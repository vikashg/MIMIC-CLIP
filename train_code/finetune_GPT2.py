# https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171
import os
import time
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Config
from transformers import get_linear_schedule_with_warmup
import nltk
nltk.download('punkt')

from tqdm import tqdm, trange

"""
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
"""
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([Lowercase()])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=30000,
                             show_progress=True,
                             initial_alphabet=ByteLevel.alphabet(),
                             special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

        self.tokenizer.train(trainer=trainer, files = paths)

    def save_tokenizer(self, save_path, prefix=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.tokenizer.model.save(save_path, prefix)

import glob
paths = glob.glob('./processed_text/*.txt')
print(paths)

tokenizer = BPE_token()
tokenizer.bpe_train(paths)
tokenizer.save_tokenizer('./tokenizer', 'mimic_tokenizer')