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
from transformers import BertTokenizer, BertForPreTraining, BertLMHeadModel

import torch, json
from PIL import Image

def process_text(text):
    pass


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, caption_list, tokenizer, transform):
        self.image_list = image_list
        self.caption_list = caption_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.encoded_captions = self.tokenizer(self.caption_list, padding=True, truncation=True, max_length=77)

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        print(self.image_list[idx])
        image = cv2.imread(self.image_list[idx])
        caption = self.encoded_captions['input_ids'][idx]
        attention_mask = self.encoded_captions['attention_mask'][idx]

        item = {
            key: torch.tensor(val[idx]) for key, val in self.encoded_captions.items()
        }
        item['image'] = torch.tensor(image)
        item['caption'] = self.caption_list[idx]


        return image, caption, attention_mask

def get_transforms():
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=384, width=384, p=1.0),
                A.Normalize(p=1.0),
            ],
       )
    elif mode == "valid":
        return A.Compose(
            [
                A.Resize(height=384, width=384, p=1.0),
                A.Normalize(p=1.0),
            ],
        )


class ImageEncoder(torch.nn.Module):
    """
    Image encoder module.
    """
    def __init__(self, model_name = 'resnet50', pretrained = False, trainable = False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes = 0, global_pool = 'avg')
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name = 'bert-base_uncased', pretrained = False, trainable = False):
        super().__init__()
        if pretrained:
            self.model = BertForPreTraining.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.projection = torch.nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(torch.nn.Module):
    def __init__(self, temperature, image_embedding, text_embedding):
        super().__init__()
        self.temperature = temperature
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.image_projection = ProjectionHead(
            embedding_dim=self.image_embedding.projection_dim,)
        self.text_projection = ProjectionHead(
            embedding_dim=self.text_embedding)


    def forward(self, image, text, attention_mask):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text, attention_mask)






def main():
    p10_file = './image_text_pair/p10.json'
    fid = open(p10_file, 'r')
    image_text_pair_dict = json.load(fid)
    image_list = []
    caption_list = []
    for image_name, caption in image_text_pair_dict.items():
        _caption = image_text_pair_dict[image_name]['FINDINGS']
        _image_name = image_text_pair_dict[image_name]['IMAGE']
        image_list.append(_image_name)
        caption_list.append(_caption)



    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoded = tokenizer(caption_list, padding=True, truncation=True, max_length=77, return_tensors="pt")

    dataset = CLIPDataset(image_list, caption_list, tokenizer, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for _ in dataloader:
        print(_)



if __name__ == "__main__":
    main()