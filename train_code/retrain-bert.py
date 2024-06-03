from transformers import BertTokenizer, BertForPreTraining, BertLMHeadModel
from torch.utils.data import DataLoader
import torch
import json
import random
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import torch.distributed as dist


def append_dictionary(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
    return dict


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForPreTraining.from_pretrained("bert-base-cased")

## Load Data here

data_dir = './image_text_pair'
p10_file = './image_text_pair/p10.json'
fid = open(p10_file, 'r')
image_text_pair_dict = json.load(fid)

for i in range(11, 20):
    p_file = './image_text_pair/p'+str(i)+'.json'
    fid = open(p_file, 'r')
    _image_text_pair_dict = json.load(fid)
    image_text_pair_dict.update(_image_text_pair_dict)


image_list = []
caption_list = []

for k, v in  image_text_pair_dict.items():
    print(k)
    _image = v['IMAGE']
    _caption = v['FINDINGS']
    image_list.append(_image)
    caption_list.append(_caption)

# Create a bag of sentences
# Add all the list of sentences to a list
bag = []
bag = [sentence for _finding in caption_list for sentence in _finding.split('.') if sentence != '']
num_sentences = len(bag)


# Split Sentences
sentence_a = []
sentence_b = []
label = []
for _finding in caption_list:
    sentences = [sentence for sentence in _finding.split('.') if sentence != '']
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        if random.random() >= 0.5:
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, num_sentences-1)
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)

for i in range(3):
    print(label[i])
    print(sentence_a[i])
    print(sentence_b[i])

inputs = tokenizer(sentence_a, sentence_b, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=512)

inputs['next_sentence_label'] = torch.LongTensor([label]).T
print(inputs.next_sentence_label[:10])

# Masking for MLM
inputs['label'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)
# Create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
selection = []
# Create mask replacement array
for i in range(inputs.input_ids.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

for i in range(inputs.input_ids.shape[0]):
    inputs.label[i, selection[i]] = 103

print(inputs['input_ids'][0])
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

class RadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = RadDataset(inputs)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

## Finally Training
print(inputs.keys())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

optimizer = Adam(model.parameters(), lr=0.0001)
model.to(device)
from tqdm import tqdm

num_epochs = 5

for param in model.parameters():
    param.requires_grad = True

for epoch in range(num_epochs):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        token_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        label = batch['label'].to(device)
        outputs = model(input_ids, token_type_ids=token_ids, attention_mask=attention_mask,
                            next_sentence_label=next_sentence_label, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


    torch.save(model.state_dict(), './model/bert-base-cased_' + str(num_epochs) + '.pt')

