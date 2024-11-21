import torch
from torch import nn, Tensor


class Dropout(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, X: Tensor):
        if self.training:
            mask = torch.bernoulli(
                torch.full_like(X, fill_value=self.drop_rate, device=X.device, requires_grad=False)
            ).bool()
            return X.masked_fill(mask, 0.0)
        else:
            return X / (1.0 - self.drop_rate)
