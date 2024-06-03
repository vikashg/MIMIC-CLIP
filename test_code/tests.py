from model import ImageEncoder, TextDecoder
import torch
import json
from PIL import Image
import cv2
from transformers import GPT2Tokenizer, GPT2LMHeadModel

clip_model = 'openai/clip-vit-base-patch32'
text_model = 'gpt2-medium'

fid = open('./image_text_pair/p10.json', 'r')
data = json.load(fid)
fid.close()

keys = list(data.keys())[0]

image_fn = data[keys]['IMAGE']
findings = data[keys]['FINDINGS']
print(image_fn)
img = cv2.imread(image_fn)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_enc = ImageEncoder(clip_model, device=device)
a = img_enc(torch.tensor([img]))
print(a.shape)


tokenizer = GPT2Tokenizer.from_pretrained(text_model)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token)
cap = tokenizer(findings, return_tensors='pt', padding=True)
input_ids, attention_mask = cap['input_ids'], cap['attention_mask']
print(input_ids.shape)
print(attention_mask.shape)
text_decoder = TextDecoder(text_model, device=device)
b = text_decoder(input_ids, attention_mask=attention_mask)
print(b.shape)
