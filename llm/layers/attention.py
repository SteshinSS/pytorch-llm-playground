import torch
from torch import nn, Tensor
from einops import rearrange
from jaxtyping import Float


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, max_context_length, qkv_bias=True):
        super().__init__()
        self.WQ = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.WK = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.WV = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.register_buffer(
            "mask",
            nn.triu(nn.ones(max_context_length, max_context_length), diagonal=1).bool(),
        )
        self.dim_out = dim_out**0.5

    def forward(self, X: Float[Tensor, "batch seq dim_in"]) -> Float[Tensor, "batch seq dim_out"]:
        batch_size, context_length, dim_in = X.shape
        Q: Float[Tensor, "batch seq dim_out"] = self.WQ(X)
        K: Float[Tensor, "batch seq dim_out"] = self.WK(X)
        V: Float[Tensor, "batch seq dim_out"] = self.WV(X)
        K = rearrange(K, "b s d -> b d s")
        scores: Float[Tensor, "batch seq seq"] = Q @ K / torch.tensor(self.dim_out, device=X.device)
        scores: Float[Tensor, "batch seq seq"] = scores.masked_fill(
            self.mask[:context_length, :context_length], -torch.inf
        )
        weights: Float[Tensor, "batch seq weight"] = torch.softmax(scores, dim=-1)
        result: Float[Tensor, "batch seq dim_out"] = torch.einsum("bsw,bwd->bsd", [weights, V])
        return result


class MultiHeadAttentionNaive(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, max_context_length, qkv_bias=True):
        super().__init__()
        if dim_out % n_heads != 0:
            raise ValueError(
                f"dim_out must be divisible by n_heads. But provided {dim_out} isn't divisible by {n_heads}."
            )
        dim = dim_out // n_heads
        self.layers = nn.ModuleList([SelfAttention(dim_in, dim, max_context_length, qkv_bias) for _ in range(n_heads)])
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, X: Float[Tensor, "batch seq dim_in"]) -> Float[Tensor, "batch seq dim_out"]:
        head_outs: list[Float[Tensor, "batch seq dim"]] = [head(X) for head in self.layers]
        head_outs: Float[Tensor, "batch seq dim_out"] = torch.cat(head_outs, dim=-1)
        return self.proj(head_outs)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, max_context_length, qkv_bias=True):
        super().__init__()
        if dim_out % n_heads != 0:
            raise ValueError(
                f"dim_out must be divisible by n_heads. But provided {dim_out} isn't divisible by {n_heads}."
            )
        self.n_heads = n_heads
        self.dim = (dim_out // self.n_heads) ** 0.5
        self.WQ = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.WK = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.WV = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_context_length, max_context_length), diagonal=1).bool(),
        )
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, X: Float[Tensor, "batch seq dim_in"]) -> Float[Tensor, "batch seq dim_out"]:
        batch_size, context_length, dim_in = X.shape
        Q: Float[Tensor, "batch seq dim_out"] = self.WQ(X)
        Q: Float[Tensor, "batch head seq dim"] = rearrange(Q, "b s (h d) -> b h s d", h=self.n_heads)
        K: Float[Tensor, "batch seq dim_out"] = self.WK(X)
        K: Float[Tensor, "batch head dim seq"] = rearrange(K, "b s (h d) -> b h d s", h=self.n_heads)
        V: Float[Tensor, "batch seq dim_out"] = self.WV(X)
        V: Float[Tensor, "batch head seq dim"] = rearrange(V, "b s (h d) -> b h s d", h=self.n_heads)
        scores: Float[Tensor, "batch head seq seq"] = Q @ K / torch.tensor(self.dim, device=X.device)
        scores: Float[Tensor, "batch head seq seq"] = scores.masked_fill(
            self.mask[:context_length, :context_length], -torch.inf
        )
        weights: Float[Tensor, "batch head seq seq"] = torch.softmax(scores, dim=-1)
        result: Float[Tensor, "batch head seq dim"] = weights @ V
        result: Float[Tensor, "batch seq dim_out"] = rearrange(result, "b h s d -> b s (h d)")
        return self.proj(result)
