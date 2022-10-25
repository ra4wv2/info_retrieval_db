import streamlit as st
import argparse
from time import time
import re
import pymorphy2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


swords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


parser = argparse.ArgumentParser(description='Searcher')
parser.add_argument('-d', '--indir', help='Input txt-file path', default='C:/Users/ra4wv/comments.txt', required=False)
parser.add_argument('-bmmatrix', '--bm25_matrix', help='Input pkl-file path', default='C:/Users/ra4wv/bm25_answers_matrix50000.pkl', required=False)
parser.add_argument('-tmatrix', '--tfidf_matrix', help='Input pkl-file path', default='C:/Users/ra4wv/tdidf_answers_matrix50000.pkl', required=False)
parser.add_argument('-tvec', '--tfidf_vec', help='Input pkl-file path', default='C:/Users/ra4wv/tfidf_vectorizer.pkl', required=False)
parser.add_argument('-cvec', '--count_vec', help='Input pkl-file path', default='C:/Users/ra4wv/count_vectorizer.pkl', required=False)
parser.add_argument('-mdl', '--bert_model', help='Input pkl-file path', default='C:/Users/ra4wv/bert_model.pkl',  required=False)
parser.add_argument('-tknzr', '--bert_tokenizer', help='Input pkl-file path', default='C:/Users/ra4wv/bert_tokenizer.pkl', required=False)
parser.add_argument('-bematrix', '--bert_matrix', help='Input pkl-file path', default='C:/Users/ra4wv/bert_answers_matrix50000.pkl', required=False)
args = parser.parse_args()

tfidf_vectorizer = pickle.Unpickler(open(args.tfidf_vec, 'rb')).load()
count_vectorizer = pickle.Unpickler(open(args.count_vec, 'rb')).load()


model = pickle.Unpickler(open(args.bert_model, 'rb')).load()
tokenizer = pickle.Unpickler(open(args.bert_tokenizer, 'rb')).load()


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


def reshape_matrix(matrix):
    size = matrix.shape
    return matrix.reshape((size[0], size[2]))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def bert_vectorize_query(query):
    clear_query = clear_text(query)
    encoded_input = tokenizer(clear_query, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input['attention_mask'])


def bm25_vectorize_query(query):
    clear_query = clear_text(query)

    return sparse.csr_matrix(count_vectorizer.transform([clear_query]))


def tfidf_vectorize_query(query):

    clear_query = clear_text(query)
    return sparse.csr_matrix(tfidf_vectorizer.transform([clear_query]))


def get_similarity(q_vec, wmatrix):
    return np.dot(wmatrix, q_vec.T)
    

def bert_get_similarity(q_vec, wmatrix):
    return cosine_similarity(wmatrix, q_vec)
    

def main():
    st.title("Поисковик")

    query = st.text_input('Ваш запрос')
    algorithm = st.radio("Select Gender: ", ('tf-idf', 'bm25', 'bert'))
    n_answers = st.slider('Размер выдачи', min_value=1, max_value=20)

    with open(args.indir, 'r', encoding='utf-8') as comm_file:
        comments = comm_file.read().split('\n')

    if st.button('Найти'):
        start = time()
        if algorithm == 'tf-idf':
            query_vec = tfidf_vectorize_query(query)
            tfidf_ans_matrix = pickle.Unpickler(open(args.tfidf_matrix, 'rb')).load()
            sim_docs = get_similarity(query_vec, tfidf_ans_matrix)
            sorted_doc_sim = np.argsort(sim_docs.toarray(), axis=0)[::-1]
            answs = np.array(comments)[sorted_doc_sim.ravel()][:n_answers]
        elif algorithm == 'bm25':
            query_vec = bm25_vectorize_query(query)
            bm25_ans_matrix = pickle.Unpickler(open(args.bm25_matrix, 'rb')).load()
            sim_docs = get_similarity(query_vec, bm25_ans_matrix)
            sorted_doc_sim = np.argsort(sim_docs.toarray(), axis=0)[::-1]
            answs = np.array(comments)[sorted_doc_sim.ravel()][:n_answers]
        elif algorithm == 'bert':
            query_vec = bert_vectorize_query(query)
            bert_ans_matrix = pickle.Unpickler(open(args.bert_matrix, 'rb')).load()
            sim_docs = bert_get_similarity(query_vec, reshape_matrix(bert_ans_matrix))
            sorted_doc_sim = np.argsort(sim_docs, axis=0)[::-1]
            answs = np.array(comments)[sorted_doc_sim.ravel()][:n_answers]
        st.write('Результаты поиска')
        for n, ans in enumerate(answs):
            st.write(f'{n + 1}) {ans}')
        st.write(f'Время поиска: {round(time() - start, 5)} секунд')


if __name__ == "__main__":
    main()

