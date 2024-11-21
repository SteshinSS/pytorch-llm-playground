from torch import nn


class FeedForward(nn.Sequential):
    def __init__(self, dim):
        super().__init__(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
