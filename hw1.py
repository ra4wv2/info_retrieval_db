# -*- coding: UTF-8 -*-

from sys import argv
import re
import os
import pymorphy2
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

swords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
vectorizer = CountVectorizer()
script_param = argv
if len(script_param) == 1:
    raise Exception('Укажите путь к папке с эпизодами :(')
else:
    dir_path = script_param[1]

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


def index_data_dict(index_text):
    word_dict = {}

    for i in range(1, len(index_text) + 1):

        with open(index_text[i], 'r', encoding='utf-8') as ep_text:

            data = ep_text.read()
            lemmatized_data = clear_text(data)

            for word in lemmatized_data:
                if word in word_dict.keys():
                    if i in word_dict[word].keys():
                        word_dict[word][i] += 1
                    else:
                        word_dict[word][i] = 1
                else:
                    word_dict[word] = {i: 1}

    return word_dict
    
    
def index_data_matrix(index_text):
    corpus = {}

    for i in range(1, len(index_text) + 1):

        with open(index_text[i], 'r', encoding='utf-8') as ep_text:

            data = ep_text.read()
            lemmatized_data = clear_text(data)
            corpus[i] = ' '.join(lemmatized_data)

    X = vectorizer.fit_transform(list(corpus.values()))

    dt = {i: X.toarray()[i - 1] for i in range(1, len(corpus) + 1)}
    words = vectorizer.get_feature_names()
    wmatrix = pd.DataFrame(dt)
    wmatrix = wmatrix.set_axis(words)

    return wmatrix


def get_rare_word(m):
    raws = pd.Series(m.values.tolist())
    words = m.index
    n = 0
    word = []

    for i in range(len(words)):
        if Counter(raws[i])[0] > n:
            n = Counter(raws[i])[0]
            word = [words[i]]
        elif Counter(raws[i])[0] == n:
            word.append(words[i])

    return word, len(raws[0]) - n


def get_freq_word(m):
    raws = pd.Series(m.values.tolist())
    words = m.index
    n = 0
    word = []

    for i in range(len(words)):
        if sum(raws[i]) > n:
            n = sum(raws[i])
            word = [words[i]]
        elif sum(raws[i]) == n:
            word.append(words[i])

    return word, n


def get_const_wordlist(m):
    raws = pd.Series(m.values.tolist())
    words = m.index
    word = []

    for i in range(len(words)):
        if Counter(raws[i])[0] == 0:
            if not word:
                word = [words[i]]
            else:
                word.append(words[i])

    return word


def get_main_char(m, chars):
    raws = pd.Series(m.values.tolist())
    chars_num = {}

    for char in chars:
        eps_per_name = []

        for name in char:
            if name.lower() in list(m.index):
                ind = list(m.index).index(name.lower())
                eps_per_name.extend(raws[ind])

        chars_num[char[0]] = sum(eps_per_name)

    max_num = max(chars_num.values())
    characters = []

    for k, v in chars_num.items():
        if v == max_num:
            characters.append(k)

    return characters, max_num


def main():
    print('processing data...')
    ind_text = preprocess_data(dir_path)
    print('compiling dictionary...')
    rev_ind = index_data_dict(ind_text)
    print('making matrix...')
    wt_matrix = index_data_matrix(ind_text)
    names = [['Моника', 'Мон'],
             ['Рэйчел', 'Рейч'],
             ['Чендлер', 'Чэндлер', 'Чен'],
             ['Фиби', 'Фибс'],
             ['Росс'],
             ['Джоуи', 'Джои', 'Джо']]
    rare_words = get_rare_word(wt_matrix)
    freq_word = get_freq_word(wt_matrix)
    const_wordlist = get_const_wordlist(wt_matrix)
    chars_names, chars_n = get_main_char(wt_matrix, names)
    print(f"Пример словаря для слова 'друг': {rev_ind['друг']}")
    print(f"Матрица: {wt_matrix}")
    print(f"Список самых редких слов ({len(rare_words[0])} элем.): {', '.join(rare_words[0][:20])} (показаны первые 20)."
          f" Они встречаются {rare_words[1]} раз.")
    print(f"Список самых частотных слов ({len(freq_word[0])} элем.): {', '.join(freq_word[0])}."
          f" Они встречаются {freq_word[1]} раз.")
    print(f"Список слов, которые встречаются во всех сериях ({len(const_wordlist)} элем.): {', '.join(const_wordlist)}.")
    print(f"Из героев чаще всего ({chars_n} раз) упоминается {', '.join(chars_names)}.")


if __name__ == "__main__":
    main()
