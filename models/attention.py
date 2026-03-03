import math
import torch
from torch import nn


class TemporalBiasMultiHeadSelfAttention(nn.Module):
    """Self-attention with optional learnable temporal log-distance bias.
    A_ij = qk/sqrt(d) + beta * log(1 + |i-j|)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_temporal_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias
        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_temporal_bias:
            idx = torch.arange(s, device=x.device)
            dist = (idx[None, :] - idx[:, None]).abs().float()
            bias = self.beta * torch.log1p(dist)
            attn = attn + bias[None, None, :, :]

        probs = torch.softmax(attn, dim=-1)
        ctx = (self.dropout(probs) @ v).transpose(1, 2).reshape(b, s, d)
        return residual + self.dropout(self.out(ctx))
