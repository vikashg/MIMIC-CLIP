import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer, GPT2LMHeadModel



class ImageEncoder(nn.Module):
    def __init__(self, model, device='cpu'):
        super(ImageEncoder, self).__init__()
        self.device = device
        self.preprocesser = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        image = self.preprocesser(images= image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)

        return image_features.pooler_output

class TextDecoder(nn.Module):

    def __init__(self, model, device='cpu'):
        super(TextDecoder, self).__init__()
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embeddings, attention_mask=None):
        text_features = self.model(input_ids=embeddings, attention_mask=attention_mask)
        return text_features.logits

class Mapping(nn.Module):
    """
    Mapping from image embedding space to text embedding space
    """
    def __init__(self, ep_len,
                 num_layers,
                 embed_size,
                 n_hides,
                 forward_expansion, dropout, device='cpu'):
        super(Mapping, self).__init__()
        self.ep_len = ep_len
        self.embed_size = embed_size
        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=8, dim_feedforward=embed_size*forward_expansion,
                                       dropout=dropout, batch_first=True), num_layers=num_layers).to(self.device)

        self.mapper =nn.Linear(embed_size, ep_len *  embed_size).to(self.device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size]
                if train_mode else
                [self.ep_len, self.embed_size]
            )
        )

class Net(nn.Module):
    def __init__(selfself, clip_model, text_model, ep_len, num_layers, n_heads, forward_expansion, dropout, max_len, device='cpu'):
        """
        Model COnstructor
        :param clip_model: CLIP model
        :param text_model: Text model
        :param ep_len: Episode length
        :param num_layers: Number of transformer encoder layers
        :param n_heads: Number of heads
        :param forward_expansion: Forward expansion
        :param dropout: Dropout
        :param max_len: Maximum length of text
        :param device: Device
        """
        super(Net, self).__init__()

        self.device = device
        self.ep_len = ep_len

        self.image_encoder = ImageEncoder(clip_model, device=device)
        self.mp = Mapping(ep_len, num_layers, self.image_encoder.model.config.hidden_size, n_heads, forward_expansion, dropout, device=device)
        self.text_decoder = TextDecoder(text_model, device=device)

        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss()

    def freeze_layers(self):
        """
        Freeze layers
        """
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_decoder.parameters():
            param.requires_grad = False

    def forward(self):
        """
        Forward pass
        """
        if temperature <= 0.0:
            temperature = 1.0

        with torch.no_grad():
            img_embedding = self.image_encoder(image)
            img_mapped = self.mp(img_embedding)
            sos_emb = self.td.model.transformer.wte(torch.tensor(self.td.tokenizer.bos_token_id).to(self.device))
            sos_emb = sos_emb.unsqueeze(0)

            start_emb = torch.cat([sos_emb, img_mapped], dim=0)
            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.transformer.wte(torch.tensor(tokens).to(self.device))

