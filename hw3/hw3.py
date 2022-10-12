# -*- coding: UTF-8 -*-
import re
from scipy import sparse
import json
from tqdm.auto import tqdm
import pymorphy2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import argparse

parser = argparse.ArgumentParser(description='Searcher')
parser.add_argument('indir', type=str, help='Input jsonl-file path')
parser.add_argument('-l', '--list',
                    action='append',
                    help='''Write query as '-l "your-query"' any times''',
                    required=True)
args = parser.parse_args()
swords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()


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
        corpus = list(f)[:50000]

    texts = []
    comments = []

    for question in tqdm(corpus, total=50000):
        answers = json.loads(question)['answers']
        auth_values = [int(answer['author_rating']['value']) for answer in answers if answer['author_rating']['value']]
        if auth_values:
            texts.append(clear_text(answers[np.argmax(auth_values)]['text']))
            comments.append(answers[np.argmax(auth_values)]['text'])

    return texts, comments


def index_data_matrix(texts):
    print('matrix in progress..')

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

    return matrix


def vectorize_query(query):

    clear_query = clear_text(query)
    return sparse.csr_matrix(count_vectorizer.transform([clear_query]))


def get_similarity(q_vec, wmatrix):
    return np.dot(wmatrix, q_vec.T).toarray()


def main():
    ind_text, comments = preprocess_data(args.indir)
    wt_matrix = index_data_matrix(ind_text)

    for query in args.list:
        query_vec = vectorize_query(query)
        sim_docs = get_similarity(query_vec, wt_matrix)
        sorted_doc_sim = np.argsort(sim_docs, axis=0)[::-1]
        answ = np.array(comments)[sorted_doc_sim.ravel()]
        print(f"\nВыдача на запрос '{query}' в порядке убывания (показаны первые 20 документов): \n")
        for i in range(len(answ[:20])):
            print(f'{i}. {answ[i]}\n')


if __name__ == "__main__":
    main()
