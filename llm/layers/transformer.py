from torch import nn, Tensor
from jaxtyping import Float

from llm.layers.attention import MultiHeadAttention
from llm.layers.feedforward import FeedForward
from llm.layers.layer_norm import LayerNorm
from llm.layers.dropout import Dropout


class TransformerBlockMy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(cfg["dim"], cfg["dim"], cfg["n_heads"], cfg["context_length"])
        self.layer_norm_first = LayerNorm(cfg["dim"])
        self.layer_norm_second = LayerNorm(cfg["dim"])
        self.dropout = Dropout(cfg["drop_rate"])
        self.feedforward = FeedForward(cfg["dim"])

    def forward(self, X: Float[Tensor, "batch seq dim"]):
        shortcut = X
        X = self.layer_norm_first(X)
        X = self.attention(X)
        X = self.dropout(X)
        X = shortcut + X
        shortcut = X
        X = self.layer_norm_second(X)
        X = self.feedforward(X)
        X = self.dropout(X)
        X = shortcut + X
        return X
