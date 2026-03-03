import torch
from torch import nn

from models.moe.experts import DenseFFN, Expert
from models.moe.router import CaptionConditionedRouter


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dense_only: bool = False,
        no_caption_conditioning: bool = False,
    ):
        super().__init__()
        self.dense_only = dense_only
        self.dense = DenseFFN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.experts = nn.ModuleList([Expert(embed_dim) for _ in range(num_experts)])
        self.router = CaptionConditionedRouter(
            embed_dim,
            num_experts,
            top_k,
            no_caption_conditioning=no_caption_conditioning,
        )

    def forward(self, x: torch.Tensor, text_state: torch.Tensor):
        if self.dense_only:
            return x + self.norm(self.dense(x)), {}

        topi, topv, probs, losses = self.router(x, text_state)
        b, _, _ = x.shape
        mixed = torch.zeros_like(x)
        for bi in range(b):
            out = 0
            for k in range(topi.shape[1]):
                e = self.experts[topi[bi, k].item()]
                out = out + topv[bi, k] * e(x[bi])
            mixed[bi] = out

        losses["routing_probs"] = probs
        return x + self.norm(mixed), losses
