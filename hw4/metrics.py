# -*- coding: UTF-8 -*-
import re
import json
from scipy import sparse
from tqdm.auto import tqdm
import pymorphy2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import pickle


tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()

parser = argparse.ArgumentParser(description='Searcher')
parser.add_argument('-d', '--indir', type=str, help='Input jsonl-file path')
parser.add_argument('-ambm', '--bm25_amatrix', type=str, help='Input pkl-file path')
parser.add_argument('-qmbm', '--bm25_qmatrix', type=str, help='Input pkl-file path')
parser.add_argument('-ambert', '--bert_amatrix', type=str, help='Input pkl-file path')
parser.add_argument('-qmbert', '--bert_qmatrix', type=str, help='Input pkl-file path')

args = parser.parse_args()

swords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def clear_text(data):
    data = data.lower()
    text = re.sub('[^а-яё]-[^а-яё]', ' ', data)
    text = re.sub('[^а-яё\-]', ' ', text)
    words = word_tokenize(text)

    lemmatized_words = []

    for word in words:
        if word not in swords:
            p = morph.parse(word)[0]
            lemmatized_words.append(p.normal_form)

    return ' '.join(lemmatized_words)


def preprocess_data(path):
    print('loading corpus...')

    with open(path, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10000]

    clear_comments = []
    clear_questions = []
    for question in tqdm(corpus, total=10000):
        answers = json.loads(question)['answers']
        auth_values = [int(answer['author_rating']['value']) for answer in answers if answer['author_rating']['value']]
        if auth_values:
            clear_comments.append(clear_text(answers[np.argmax(auth_values)]['text']))
            clear_questions.append(clear_text(json.loads(question)['question']))

    return clear_comments, clear_questions


def bert_index_data_matrix(texts, token_name, model, tokenizer):
    print('bert matrix in progress...')

    outputs = []
    for i in tqdm(range(len(texts))):
        encoded_input = tokenizer(texts[i], padding=True, truncation=True, max_length=512, return_tensors='pt')
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to('cpu')
        with torch.no_grad():
            model_output = model(**encoded_input)
        outputs.append(mean_pooling(model_output, encoded_input['attention_mask']))
    full_matrix = torch.stack(outputs, dim=0)
    pickle.dump(full_matrix, open(token_name + '_matrix.pkl', 'wb'))

    return full_matrix


def bm25_index_data_matrix(texts, token_name):
    print('bm25 matrix in progress...')

    if 'answers' in token_name:
        tf = count_vectorizer.fit_transform(texts)
        ti = tfidf_vectorizer.fit_transform(texts)
        idf = tfidf_vectorizer.idf_
        l_d = tf.sum(axis=1)
        avgdl = l_d.mean()
        k = 2
        b = 0.75

        rows = []
        cols = []
        values = []
        for i, j in zip(*tf.nonzero()):
            rows.append(i)
            cols.append(j)
            values.append(idf[j] * (tf[i, j] * (k + 1)) / (tf[i, j] + k * (1 - b + b * l_d[i, 0] / avgdl)))

        matrix = sparse.csr_matrix((values, (rows, cols)))
        pickle.dump(matrix, open(token_name + '_matrix.pkl', 'wb'))

    else:
        matrix = sparse.csr_matrix(count_vectorizer.transform(texts))
        pickle.dump(matrix, open(token_name + '_matrix.pkl', 'wb'))

    return matrix


def reshape_matrix(matrix):
    size = matrix.shape
    return matrix.reshape((size[0], size[2]))


def get_dependency_bert(amatrix, qmatrix, clear_comments):
    matrix = np.dot(amatrix, qmatrix.T)
    found = [(np.array(amatrix)[np.argmax(matrix[i], axis=0)] == np.array(amatrix[i])).all() for i in tqdm(range(len(clear_comments)))].count(True)
    return found/len(clear_comments)


def get_dependency_bm25(amatrix, qmatrix, clear_comments):
    matrix = np.dot(amatrix, qmatrix.T)
    found = [(amatrix.toarray()[np.argmax(matrix[i], axis=0)] == np.array(amatrix.toarray()[i])).all() for i in tqdm(range(len(clear_comments)))].count(True)
    return found/len(clear_comments)


def main():
    clear_comments, clear_questions = preprocess_data(args.indir)

    if args.bert_amatrix and args.bert_qmatrix:
        bert_ans_matrix = reshape_matrix(pickle.Unpickler(open(args.bert_amatrix, 'rb')).load())
        bert_quest_matrix = reshape_matrix(pickle.Unpickler(open(args.bert_qmatrix, 'rb')).load())
    else:
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        bert_ans_matrix = reshape_matrix(bert_index_data_matrix(clear_comments, 'bert_answers', model, tokenizer))
        bert_quest_matrix = reshape_matrix(bert_index_data_matrix(clear_questions, 'bert_questions', model, tokenizer))

    if args.bm25_amatrix and args.bm25_qmatrix:
        bm25_ans_matrix = pickle.Unpickler(open(args.bm25_amatrix, 'rb')).load()
        bm25_quest_matrix = pickle.Unpickler(open(args.bm25_qmatrix, 'rb')).load()
    else:
        bm25_ans_matrix = pickle.Unpickler(open(args.bm25_amatrix, 'rb')).load()
        bm25_quest_matrix = pickle.Unpickler(open(args.bm25_qmatrix, 'rb')).load()

    bert_dep = get_dependency_bert(bert_ans_matrix, bert_quest_matrix, clear_comments)
    bm25_dep = get_dependency_bm25(bm25_ans_matrix, bm25_quest_matrix, clear_comments)

    return f'Метрика для BERT: {bert_dep},\nМетрика для BM25: {bm25_dep}'


if __name__ == "__main__":
    main()
