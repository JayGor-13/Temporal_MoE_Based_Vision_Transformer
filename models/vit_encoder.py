from torch import nn

from config import CFG
from models.attention import TemporalBiasMultiHeadSelfAttention
from models.embeddings import VideoEmbedder
from models.moe.hetero_moe_ffn import MoEFeedForward


class TemporalMoEBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int, top_k: int, dense_only: bool = False):
        super().__init__()
        self.attn = TemporalBiasMultiHeadSelfAttention(embed_dim, num_heads, CFG.dropout)
        self.ffn = MoEFeedForward(embed_dim, num_experts, top_k, dense_only=dense_only)

    def forward(self, x, text_state):
        x = self.attn(x)
        x, diag = self.ffn(x, text_state)
        return x, diag


class TemporalMoEViTEncoder(nn.Module):
    def __init__(self, dense_only: bool = False):
        super().__init__()
        self.embed = VideoEmbedder(CFG.embed_dim, CFG.num_frames)
        self.layers = nn.ModuleList(
            [TemporalMoEBlock(CFG.embed_dim, CFG.num_heads, CFG.num_experts, CFG.top_k, dense_only=dense_only) for _ in range(CFG.num_layers)]
        )
        self.norm = nn.LayerNorm(CFG.embed_dim)

    def forward(self, video, text_state):
        x = self.embed(video)
        diagnostics = {}
        for i, layer in enumerate(self.layers):
            x, d = layer(x, text_state)
            for k, v in d.items():
                diagnostics[f"layer_{i}_{k}"] = v
        return self.norm(x), diagnostics
