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

model = BertForPreTraining.from_pretrained("bert-base-cased")
filename = './model/bert-base-cased_5.pt'
model.load_state_dict(torch.load(filename))
print(model)

