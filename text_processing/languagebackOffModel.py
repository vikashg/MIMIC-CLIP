import requests
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from create_word_2_vec import create_word_blob_from_data
from create_word_2_vec import preprocess_data


def main():
    nltk.download('punkt')
    list_sentences = create_word_blob_from_data()
    filtered_sentence_list = []
    for _sen in list_sentences:
        filtered_sentence = remove_stopwords(_sen)
        filtered_sentence_list.append(filtered_sentence)

    preprocess_data1 = preprocess_data(filtered_sentence_list)


    n_grams = ngrams(nltk.word_tokenize(list_sentences[1]), 2)
    print(list_sentences[1])
    x = nltk.word_tokenize(preprocess_data1[1])
    print(type(x))


if __name__=='__main__':
    main()
