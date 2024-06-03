from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForSequenceClassification
from transformers import BertForPreTraining
import torch
from word2vec_Cbow import get_sentence_list
import random


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, item):
        return {key: torch.tensor(val[item]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def create_nsp_data(List_SENTENCES, bag):
    """
    bag is all the sentences in the corpus
    List_SENTENCES is a list of reports
    :param List_SENTENCES:
    :param bag:
    :return:
    """
    bag_size = len(bag.split('.'))

    sentence_a = []
    sentence_b = []
    labels = []

    for paragraph in List_SENTENCES:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != '']
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences - 2)
            if random.random() >= 0.5:
                sentence_a.append(sentences[start].lstrip().rstrip().lower())
                sentence_b.append(sentences[start + 1].lstrip().rstrip().lower())
                labels.append(0)
            else:
                index = random.randint(0,  bag_size - 1)
                sentence_a.append(sentences[start].lstrip().rstrip().lower())
                sentence_b.append(bag[index].lstrip().rstrip().lower())
                labels.append(1)

    return sentence_a, sentence_b, labels

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.train()


    sentences, LIST_SENTENCES = get_sentence_list()

    # Prepare for NSP
    bag = [item for sentence in LIST_SENTENCES for item in sentence.split('.') if item != ''][0]

    # 50/50 ISNextSentence labels
    sentence_a, sentence_b, label = create_nsp_data(LIST_SENTENCES, bag)
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    inputs['next_sentence_label'] = torch.LongTensor([label]).T

    print(inputs.keys())
    dataset = NewDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    loss = torch.nn.CrossEntropyLoss()
    from tqdm import tqdm
    epochs = 2
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, )
            # loss = loss(outputs['pooler_output'], next_sentence_label)
            print(next_sentence_label.shape, outputs['pooler_output'].shape)
            """"
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            """
            print('Epoch: ', epoch, 'Loss: ', loss.item())



if __name__ == '__main__':
    main()