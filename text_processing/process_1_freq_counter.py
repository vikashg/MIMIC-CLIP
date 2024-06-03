from sklearn.feature_extraction.text import CountVectorizer
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
    corpus_list = corpus.split('.')
    vectorizer = CountVectorizer()
    vocab = vectorizer.fit(corpus_list)
    X = vectorizer.transform(corpus_list)
    print(X.toarray().shape)
    print(X.toarray()[0,:])






if __name__ == '__main__':
    main()