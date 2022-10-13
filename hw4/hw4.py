# -*- coding: UTF-8 -*-
import re
import json
from tqdm.auto import tqdm
import pymorphy2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru')

parser = argparse.ArgumentParser(description='Searcher')
parser.add_argument('-d', '--indir', type=str, help='Input jsonl-file path')
parser.add_argument('-m', '--amatrix', type=str, help='Input pkl-file path')
parser.add_argument('-l', '--list',
                    action='append',
                    help='''Write query as '-l "your-query"' any times''',
                    required=True)
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
    comments = []

    for question in tqdm(corpus, total=10000):
        answers = json.loads(question)['answers']
        auth_values = [int(answer['author_rating']['value']) for answer in answers if answer['author_rating']['value']]
        if auth_values:
            clear_comments.append(clear_text(answers[np.argmax(auth_values)]['text']))
            comments.append(answers[np.argmax(auth_values)]['text'])

        return clear_comments, comments


def index_data_matrix(texts, token_name):
    print('matrix in progress..')

    outputs = []
    for i in tqdm(range(len(texts))):
        encoded_input = tokenizer(texts[i], padding=True, truncation=True, max_length=512, return_tensors='pt')
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to('cpu')
        with torch.no_grad():
            model_output = model(**encoded_input)
        outputs.append(mean_pooling(model_output, encoded_input['attention_mask']))
    full_matrix = torch.stack(outputs, dim=0)
    pickle.dump(full_matrix, open(token_name + '_matrix_' + str(len(texts)) + '.pkl', 'wb'))

    return full_matrix


def vectorize_query(query):
    clear_query = clear_text(query)
    encoded_input = tokenizer(clear_query, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input['attention_mask'])


def get_similarity(q_vec, wmatrix):
    return np.dot(wmatrix, q_vec.T)


def main():
    clear_comments, comments = preprocess_data(args.indir)
    if args.amatrix:
        ans_matrix = pickle.Unpickler(open(args.amatrix, 'rb')).load()
    else:
        ans_matrix = index_data_matrix(clear_comments, 'answers')

    for query in args.list:
        query_vec = vectorize_query(query)
        sim_docs = get_similarity(query_vec, ans_matrix)
        sorted_doc_sim = np.argsort(sim_docs, axis=0)[::-1]
        answ = np.array(comments)[sorted_doc_sim.ravel()]
        print(f"\nВыдача на запрос '{query}' в порядке убывания (показаны первые 20 документов): \n")
        for i in range(len(answ[:20])):
            print(f'{i}. {answ[i]}\n')


if __name__ == "__main__":
    main()
