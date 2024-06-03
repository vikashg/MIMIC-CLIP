import cv2
import logging
import math
import os
import sys
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    FlaxCLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)
from transformers import BertTokenizer, BertForPreTraining, BertLMHeadModel, BertConfig, DistilBertConfig
import timm
from torch import nn
import torch, json
from PIL import Image

model_name = "resnet50"
pretrained = False
trainable = False


class ImageEncoder(nn.Module):
    """
    Encodes image to a fixed size vector
    """

    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__ (self, model_name = 'bert-base_cased', pretrained=True, trainable=True, model_path=None):
        super().__init__()
        if pretrained:
            self.model = BertForPreTraining.from_pretrained("bert-base-cased")
        else:
            self.model = BertLMHeadModel(config=BertConfig())

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        # last_hidden_state = output.last_hidden_state
        # return last_hidden_state[:, self.target_token_idx, :]
        return output

def main():
    image_dir = '/workspace/data/image/p10/p10044189/s53647608'
    image_file = os.path.join(image_dir, '1c2182ed-ca83ed53-f7ce0244-9bffb5e3-d431c131-mask.jpg')
    image = cv2.imread(image_file)
    print(image.shape)

    image_encoder = ImageEncoder(model_name, pretrained, trainable)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.permute(0, 3, 1, 2)
    print(image.shape)
    image = image.float()
    print(image.shape)
    a = image_encoder(image)
    print(a.shape)
    text_encoder = TextEncoder(model_path='/workspace/app/model/bert-base-cased_5.pt')

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    filename = '/workspace/app/image_text_pair/p10.json'
    fid = open(filename, 'r')
    data_dict = json.load(fid)
    key_list = data_dict.keys()
    for k, v in data_dict.items():
        print(k)
        _data = data_dict[k]
        image_fn = _data['IMAGE']
        findings = _data['FINDINGS']
        tokenizer_output = tokenizer(findings, return_tensors="pt")
        a = text_encoder(tokenizer_output['input_ids'], tokenizer_output['attention_mask'])
        print(a['prediction_logits'].shape)
        break




if __name__ == "__main__":
    main()