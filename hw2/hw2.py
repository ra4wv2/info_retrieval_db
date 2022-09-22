# -*- coding: UTF-8 -*-
import re
import os
import pymorphy2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

parser = argparse.ArgumentParser(description='Searcher')
parser.add_argument('indir', type=str, help='Input dir with seasons dirs')
parser.add_argument('-l', '--list',
                    action='append',
                    help='''Write query as '-l "your-query"' any times''',
                    required=True)
args = parser.parse_args()
swords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
vectorizer = TfidfVectorizer()


def preprocess_data(path):
    index_text = {}
    n = 0

    ep_dirs = os.listdir(path)

    for ep_dir in ep_dirs:
        eps_path = os.path.join(path, ep_dir)
        eps = os.listdir(eps_path)

        for ep in eps:
            filepath = os.path.join(eps_path, ep)
            n += 1
            index_text[n] = filepath

    return index_text


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

    return lemmatized_words


def index_data_matrix(index_text):
    corpus = {}

    for i in range(1, len(index_text) + 1):
        with open(index_text[i], 'r', encoding='utf-8') as ep_text:
            data = ep_text.read()
            lemmatized_data = clear_text(data)
            corpus[i] = ' '.join(lemmatized_data)

    X = vectorizer.fit_transform(list(corpus.values()))

    words = vectorizer.get_feature_names()

    return X.toarray(), words


def vectorize_query(query):
    clear_query = clear_text(query)
    X = vectorizer.transform([' '.join(clear_query)])

    return X.toarray()[0]


def get_cos_similarity(q_vec, wmatrix):
    return np.dot(wmatrix, q_vec)


def main():
    ind_text = preprocess_data(args.indir)
    wt_matrix, words = index_data_matrix(ind_text)

    for query in args.list:
        query_vec = vectorize_query(query)
        cos_sim_docs = get_cos_similarity(query_vec, wt_matrix)

        doc_cossim = {ind_text[i].split('\\')[-1]: cos_sim_docs[i - 1] for i in range(1, len(cos_sim_docs) + 1)}
        sorted_doc_cossim = {k: v for k, v in sorted(doc_cossim.items(), key=lambda item: item[1], reverse=True)}

        print(f"Выдача на запрос '{query}' в порядке убывания (найдено {len(sorted_doc_cossim)} документов): \n")
        for i in range(len(sorted_doc_cossim)):
            print(f'{i}. {list(sorted_doc_cossim)[i][:-7]}\n')


if __name__ == "__main__":
    print('loading...')
    main()
