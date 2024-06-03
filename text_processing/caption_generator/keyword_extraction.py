from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(sentence):
    """
    Clean text by removing unnecessary characters and altering the format of words.
    :param text: The text to be cleaned
    :return: The cleaned text
    """
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', ':', ';', '(', ')', '#', '--', '...', '"', '___'])

    sentence_list = sentence.split('.')
    clean_words= []
    new_sentence_list = []

    for sen in sentence_list:
        if sen == '':
            continue
        else:
            new_sentence_list.append(sen.rstrip().lstrip())

    test = ""
    for sen in new_sentence_list:
        test = test + sen + " "
    return new_sentence_list

def get_list_reports():
    """
    Generate a list of findings and impressions from the reports
    :return:
    """
    data_file = '../data_list_p10_'

    list_reports = []
    for i in range(10):
        fid = open(data_file + str(i) + '.json', 'r')
        print(data_file + str(i) + '.json')
        data = json.load(fid)
        fid.close()

        for key, value in data.items():
            temp = ""
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
                        temp = temp + data_dict['FINDINGS']

                    elif 'IMPRESSION' in data_dict.keys():
                        if data_dict['IMPRESSION'] is None:
                            continue
                        impression = data_dict['IMPRESSION']
                        temp = temp + impression
            list_reports.append(temp)

    return list_reports

def get_sentence_list():
    data_file = '../data_list_p10_'

    list_sentences = ""
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

                        list_sentences = list_sentences + findings
                    elif 'IMPRESSION' in data_dict.keys():
                        if data_dict['IMPRESSION'] is None:
                            continue
                        impression = data_dict['IMPRESSION']
                        list_sentences = list_sentences + impression

    return list_sentences


def keyword_extractor(list_sentences):

    print("List Sentence ", type(list_sentences))
    n_gram_range = (1, 2)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', ':', ';', '(', ')', '#', '--', '...', '"', '___', '_'])
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=list(stop_words), lowercase=True,
                            ).fit(list_sentences)
    all_candidates = count.get_feature_names_out()
    import spacy
    nlp = spacy.load('en_core_web_sm')

    sen = " "
    for _sen in list_sentences:
        sen = sen + _sen + " "
    doc = nlp(sen)
    print(doc)
    noun_phrases = set(chunk.text.strip() for chunk in doc.noun_chunks)

    nouns = set()
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)

    all_nouns = nouns.union(noun_phrases)
    print(all_nouns)
    candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))
    print(all_candidates)

def main():
    """
    _list_sentences = get_sentence_list()
    list_sentences = clean_text(_list_sentences)
    keyword_extractor(list_sentences)
    """
    list_reports = get_list_reports()
    clean_reports = []
    for _report in list_reports:
        _clean_report = clean_text(_report)
        clean_reports.append(_clean_report)

    from wordwise import Extractor

    i = 72
    print(clean_reports[i])
    extractor = Extractor()

    sen = " "
    for _sen in clean_reports[i]:
        sen = sen + _sen + " "

    import yake
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(sen)

    for kw in keywords:
        print(kw)

#https://subscription.packtpub.com/book/data/9781789614381/3/ch03lvl1sec11/building-an-image-caption-generator-using-pytorch
if __name__ == '__main__':
    main()