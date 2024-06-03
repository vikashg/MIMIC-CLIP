import os, json


def get_sentence_list():
    data_file = './data_list_p10_'

    list_sentences = ""
    LIST_SENTENCES = []
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
                        LIST_SENTENCES.append(findings)

                        list_sentences = list_sentences + findings
                    elif 'IMPRESSION' in data_dict.keys():
                        if data_dict['IMPRESSION'] is None:
                            continue
                        impression = data_dict['IMPRESSION']
                        list_sentences = list_sentences + impression
                        LIST_SENTENCES.append(impression)

    return list_sentences, LIST_SENTENCES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.manifold import TSNE


def clean_text(sentence):
    """
    Clean text by removing unnecessary characters and altering the format of words.
    :param text: The text to be cleaned
    :return: The cleaned text
    """
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.',',',':',';','(',')','#','--','...','"', '___'])

    sentence_list = sentence.split('.')
    clean_words= []
    for sen in sentence_list[0:2]:
        if sen == '':
            continue
        words = word_tokenize(sen)
        for word in words:
            if word not in stop_words:
                if word.isalpha():
                    clean_words.append(word.lower())
        print(sen)

    print(clean_words)
    return clean_words


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, word_to_idx):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim*context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.word_to_idx = word_to_idx

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def predict(self, inputs):
        context_ids = torch.tensor([self.word_to_idx[w] for w in inputs], dtype=torch.long)
        res = self.forward(context_ids)
        res_arg = torch.argmax(res)
        res_val, res_idx = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_idx = res_idx[0][:3]
        for arg in zip(res_val, res_idx):
            print([(k, v, arg[0]) for k, v in self.word_to_idx.items() if v == arg[1]])

    """
    def freeze_layer(self):
        for name, child in model.named_children():
            print(name, child)
            if name == 'layer':
                for names, param in child.named_parameters():
                    param.requires_grad = False
                    print(names, param)
                    print(param.size())

    def print_layer_parameters(self):
        for name, param in model.named_parameters():
            print(name, child)
            for names, params in child.named_parameters():
                print(name, param)
                print(param.size())
    """

# https://srijithr.gitlab.io/post/word2vec/


def main():
    sentences = get_sentence_list()
    clean_words = clean_text(sentences)
    word_to_idx = {word: i for i, word in enumerate(clean_words)}
    print(word_to_idx['pneumothorax'])

    ngrams = []
    EMBEDDING_DIM = 100
    Context_size = 2
    vocab_size = len(clean_words)

    test_sentence = """Empathy for the poor may not come easily to people who never experienced it. They may blame the victims and insist their predicament can be overcome through determination and hard work.
    But they may not realize that extreme poverty can be psychologically and physically incapacitating — a perpetual cycle of bad diets, health care and education exacerbated by the shaming and self-fulfilling prophecies that define it in the public imagination.
    Gordon Parks — perhaps more than any artist — saw poverty as “the most savage of all human afflictions” and realized the power of empathy to help us understand it. It was neither an abstract problem nor political symbol, but something he endured growing up destitute in rural Kansas and having spent years documenting poverty throughout the world, including the United States.
    That sensitivity informed “Freedom’s Fearful Foe: Poverty,” his celebrated photo essay published in Life magazine in June 1961. He took readers into the lives of a Brazilian boy, Flavio da Silva, and his family, who lived in the ramshackle Catacumba favela in the hills outside Rio de Janeiro. These stark photographs are the subject of a new book, “Gordon Parks: The Flavio Story” (Steidl/The Gordon Parks Foundation), which accompanies a traveling exhibition co-organized by the Ryerson Image Centre in Toronto, where it opens this week, and the J. Paul Getty Museum. Edited with texts by the exhibition’s co-curators, Paul Roth and Amanda Maddox, the book also includes a recent interview with Mr. da Silva and essays by Beatriz Jaguaribe, Maria Alice Rezende de Carvalho and Sérgio Burgi.
    """.split()


    for i in range(len(clean_words) - Context_size):
        tup = [clean_words[j] for j in np.arange(i +1, i + Context_size + 1)]
        ngrams.append((clean_words[i], tup))

    model = CBOW(vocab_size, EMBEDDING_DIM, Context_size, word_to_idx)
    print(model)

    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(100):
        total_loss = 0
        for target, context in ngrams:
            context_ids = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
            print(context_ids, target)

            model.zero_grad()
            log_probs = model(context_ids)
            target_list = torch.tensor([word_to_idx[w] for w in target], dtype=torch.long)
            loss = loss_function(log_probs, target_list, dtype=torch.long)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()



if __name__ == '__main__':
    main()
