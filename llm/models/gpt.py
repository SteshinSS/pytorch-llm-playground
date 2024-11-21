import torch
from torch import nn, Tensor
from jaxtyping import Int, Float
from einops import rearrange

from llm.layers.dropout import Dropout
from llm.layers.transformer import TransformerBlockMy
from llm.layers.layer_norm import LayerNorm


class GPTModelMy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.pos_embeddings = nn.Embedding(cfg["context_length"], cfg["dim"])
        self.dropout = Dropout(cfg["drop_rate"])
        self.layers = nn.Sequential(*[TransformerBlockMy(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["dim"])
        self.proj = nn.Linear(cfg["dim"], cfg["vocab_size"])

    def forward(self, X: Int[Tensor, "batch seq"]):
        batch_size, context_length = X.shape
        tok_embs: Float[Tensor, "batch seq dim"] = self.tok_embeddings(X)
        pos_embs: Float[Tensor, "seq dim"] = self.pos_embeddings(torch.arange(context_length, device=X.device))
        embs: Float[Tensor, "batch seq dim"] = tok_embs + pos_embs
        embs = self.dropout(embs)
        embs = self.layers(embs)
        embs = self.final_norm(embs)
        result: Float[Tensor, "batch seq vocab_size"] = self.proj(embs)
        return result
