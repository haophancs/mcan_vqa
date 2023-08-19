import os
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoModel


class PretrainedTextEmbedding:
    def __init__(self, encoder_embed_dim=None, device='cpu', **kwargs):
        self.encoder_embed_dim = encoder_embed_dim
        self.embed_layer = None
        self.device = torch.device(device)

    @staticmethod
    def from_pretrained(pretrained_name, encoder_embed_dim=None, pretrained_root=None, device='cpu'):
        print(f'Load pretrained {pretrained_name} for text embedding')
        if not pretrained_root:
            pretrained_root = ''
        if pretrained_name in PretrainedTransformersEmbedding.PRETRAINED_MODELS:
            return PretrainedTransformersEmbedding(
                encoder_embed_dim=encoder_embed_dim,
                pretrained_name=pretrained_name,
                device=device
            )
        if pretrained_name in PretrainedW2VEmbedding.PRETRAINED_MODELS:
            return PretrainedW2VEmbedding(
                pretrained_name=pretrained_name,
                pretrained_root=pretrained_root,
                encoder_embed_dim=encoder_embed_dim,
                device=device
            )

    @abstractmethod
    def __call__(self, textual_tokens, **kwargs):
        raise NotImplementedError()

    def _reshape_to_embed_dim(self, embeddings):
        if self.encoder_embed_dim:
            if self.encoder_embed_dim > embeddings.shape[2]:
                embeddings = torch.cat([
                    embeddings,
                    torch.zeros(size=(
                        embeddings.shape[0],
                        embeddings.shape[1],
                        self.encoder_embed_dim - embeddings.shape[2]
                    )).to(self.device)
                ], dim=-1)
            else:
                embeddings = embeddings[:, :, :self.encoder_embed_dim]
        return embeddings


class PretrainedW2VEmbedding(PretrainedTextEmbedding):
    _NAME_TO_FILE = {
        'phow2v.syllable.100d': 'word2vec_vi_syllables_100dims.txt',
        'phow2v.syllable.300d': 'word2vec_vi_syllables_300dims.txt',
        'phow2v.word.100d': 'word2vec_vi_words_100dims.txt',
        'phow2v.word.300d': 'word2vec_vi_words_300dims.txt'
    }
    PRETRAINED_MODELS = list(_NAME_TO_FILE.keys())

    def __init__(self, pretrained_name, pretrained_root, encoder_embed_dim=None, device='cpu', **kwargs):
        import gensim
        super().__init__(encoder_embed_dim=encoder_embed_dim, device=device, **kwargs)
        self.embed_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(gensim.models.KeyedVectors.load_word2vec_format(os.path.join(
                pretrained_root,
                PretrainedW2VEmbedding._NAME_TO_FILE[pretrained_name]
            )).vectors)).to(self.device)
        self.embed_layer.requires_grad = False

    def __call__(self, textual_tokens, **kwargs):
        embeddings = self.embed_layer(textual_tokens.detach().to(self.device))
        embeddings = self._reshape_to_embed_dim(embeddings).to(textual_tokens.device)
        return embeddings


class PretrainedTransformersEmbedding(PretrainedTextEmbedding):
    PRETRAINED_MODELS = ["vinai/bartpho-syllable", "vinai/bartpho-word", "vinai/phobert-base",
                         "VietAI/vit5-base", "VietAI/vit5-large",
                         "VietAI/vit5-base-vietnews-summarization",
                         "VietAI/vit5-large-vietnews-summarization",
                         "facebook/bart-large"]

    def __init__(
            self,
            pretrained_name,
            pretrained_root='huggingface',
            encoder_embed_dim=None,
            device='cpu',
            **kwargs
    ):
        super().__init__(encoder_embed_dim=encoder_embed_dim, device=device, **kwargs)
        self.embed_layer = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name).to(self.device)
        self.embed_layer.requires_grad = False

    def __call__(self, textual_tokens, attention_mask=None, **kwargs):
        try:
            embeddings = self.embed_layer.model.encoder(
                input_ids=textual_tokens.detach().to(self.device),
                attention_mask=attention_mask.detach().to(self.device) if attention_mask is not None else None
            ).last_hidden_state
        except AttributeError:
            embeddings = self.embed_layer.encoder(
                input_ids=textual_tokens.detach().to(self.device),
                attention_mask=attention_mask.detach().to(self.device) if attention_mask is not None else None
            ).last_hidden_state
        embeddings = self._reshape_to_embed_dim(embeddings).to(textual_tokens.device)
        return embeddings


class VisualEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v):
        n, c, h, w = v.size()
        v = v.view(n, c, w*h).permute(0, 2, 1)

        v = self.proj(v)

        return v
