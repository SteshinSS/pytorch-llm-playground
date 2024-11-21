import torch
from torch import nn, Tensor
from jaxtyping import Float


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        pass

    def forward(self, X: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        mean: Float[Tensor, "batch seq 1"] = X.mean(dim=-1, keepdim=True)
        var: Float[Tensor, "batch seq 1"] = X.var(dim=-1, keepdim=True)
        X: Float[Tensor, "batch seq dim"] = (X - mean) / (torch.sqrt(var) + self.eps)
        return self.scale * X + self.shift
