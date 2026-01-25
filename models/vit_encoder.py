import torch
import torch.nn as nn

from ..models.embeddings import VideoEmbedder
from ..models.attention import MultiHeadSelfAttention
from ..models.moe.hetero_moe_ffn import HeteroMoEFeedForward
from ..config import NUM_FRAMES, PATCH_SIZE


# ============================================================
# --- STEP 3: UPGRADE THE TemporalMoEBlock ---
# This block now uses our new HeteroMoEFeedForward layer.
# ============================================================

class TemporalMoEBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, top_k):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # --- THE SURGERY ---
        # Instead of the old MoE layer, we instantiate the new, powerful one.
        # Note: We've removed `num_experts` as an argument because the new class handles it internally.
        self.moe = HeteroMoEFeedForward(embed_dim, top_k=top_k)

    def forward(self, x, expert_kwargs=None, compute_router_losses=False):
        # Pass data through the layers, ensuring kwargs are passed down
        x = self.attn(x)
        x, diagnostics = self.moe(x, expert_kwargs=expert_kwargs, compute_router_losses=compute_router_losses)
        return x, diagnostics

class TemporalMoEViT_Encoder(nn.Module):
        def __init__(self, embed_dim, num_heads, top_k, num_layers):
            super().__init__()
            self.embedder = VideoEmbedder(embed_dim, NUM_FRAMES, PATCH_SIZE)
            self.layers = nn.ModuleList([
                TemporalMoEBlock(embed_dim, num_heads, top_k) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, video_frames, expert_kwargs=None, compute_router_losses=False):
            diagnostics = {}
            x = self.embedder(video_frames)
            for i, layer in enumerate(self.layers):
                x, diags = layer(x, expert_kwargs, compute_router_losses)
                if compute_router_losses:
                    for k, v in diags.items():
                        diagnostics[f'layer_{i}_{k}'] = v # Collect diagnostics from each layer
            return self.norm(x), diagnostics
