import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import os, json


def main():
    corpus = ""
    for i in range(0, 10):
        filename = os.path.join('./data_list_p10_'+str(i)+'.json')
        with open(filename, 'r') as f:
            data = json.load(f)
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
                        #corpus.append(findings)
                        corpus = corpus + findings
                    elif 'IMPRESSION' in data_dict.keys():
                        if data_dict['IMPRESSION'] is None:
                            continue
                        impression = data_dict['IMPRESSION']
                        corpus = corpus + impression

        # Get the vocabulary
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(corpus)
    #print(word_tokens)
    from nltk.probability import FreqDist
    fdist = FreqDist(word_tokens)
    print(fdist.most_common(50))

    # remove punctuation
    words_no_punct = []
    for word in word_tokens:
        if word.isalpha():
            words_no_punct.append(word.lower())

    clean_words = []
    for word in words_no_punct:
        if word not in stop_words:
            clean_words.append(word)

    #frequency distribution
    fdist = FreqDist(clean_words)
    print(fdist.most_common(50))

    from nltk.stem import PorterStemmer

    porter = PorterStemmer()
    for w in clean_words:
        print(porter.stem(w))

    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    word_lemmatized = []
    for w in clean_words:
        word_lemmatized.append(lemmatizer.lemmatize(w))

    fdist = FreqDist(word_lemmatized)


if __name__=='__main__':
    main()