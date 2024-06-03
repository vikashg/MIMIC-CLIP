from gensim.models import Word2Vec
import json
import os
import re
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import nltk
import matplotlib.pyplot as plt
nltk.download('stopwords')


def create_word_blob_from_data():
    data_file = './data_list_p10_'

    list_sentences = []
    for i in range(10):
        fid = open(data_file + str(i) + '.json', 'r')
        print(data_file + str(i) + '.json')
        data = json.load(fid)
        fid.close()

        for key, value in data.items():
            data_dict = value
            if data_dict == "null":
                continue
            else:
                if data_dict is None:
                    continue
                else:
                    if 'FINDINGS' in data_dict.keys():
                        if data_dict['FINDINGS'] is None:
                            continue
                        findings = data_dict['FINDINGS']
                        list_sentences.append(findings)
                    elif 'IMPRESSION' in data_dict.keys():
                        if data_dict['IMPRESSION'] is None:
                            continue
                        impression = data_dict['IMPRESSION']
                        list_sentences.append(impression)

    return list_sentences


def preprocess_data(list_sentences):
    """
    Take a list of sentences and prepare it for word2vec
    :param list_sentences:
    :return: list_of_list_of_words
    """
    all_texts = []
    for _sen in list_sentences:
        x = _sen.split('.')
        for _x in x:
            y = _x.strip().split(' ')
            if y == ['']:
                continue
            else:
                all_texts.append(y)
    return all_texts



def main():
    list_sentences = create_word_blob_from_data()

    filtered_sentence_list = []
    for _sen in list_sentences:
        filtered_sentence = remove_stopwords(_sen)
        filtered_sentence_list.append(filtered_sentence)

    preprocess_data1 = preprocess_data(filtered_sentence_list)

    """
    model = Word2Vec(preprocess_data1, min_count=1, window=5, vector_size=100, workers=4)
    print(model.wv.most_similar('pneumonia'))

    #print the vector of a word
    print(model.wv['pneumonia'])

    #print the list of words
    print(model.wv.index_to_key)
    for index, word in enumerate(model.wv.index_to_key):
        print(index, word)

    # model.train(preprocess_data1, total_examples=len(preprocess_data1), epochs=1000)

    # print(model.wv.most_similar('pneumonia'))
    """
    print(preprocess_data1)

if __name__ == '__main__':
    main()
