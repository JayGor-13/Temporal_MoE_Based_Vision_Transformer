import torch
import torch.nn as nn

from .experts import (
    Expert, MotionExpert, TextureExpert,
    SceneExpert, FastChangeExpert, LanguageAlignedExpert
)
from .router import HeteroRouter

# ============================================================
# --- STEP 2: DEFINE THE ADVANCED HETEROGENEOUS MOE LAYER ---
# This replaces the simple 'MoEFeedForward' class.
# ============================================================

class HeteroMoEFeedForward(nn.Module):
    def __init__(self, embed_dim, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim

        # Define the committee of specialized experts internally
        self.experts = nn.ModuleList([
            MotionExpert(embed_dim),
            TextureExpert(embed_dim),
            SceneExpert(embed_dim),
            FastChangeExpert(embed_dim),
            LanguageAlignedExpert(embed_dim),
            Expert(embed_dim), # Generic Expert 1
            Expert(embed_dim), # Generic Expert 2
            Expert(embed_dim), # Generic Expert 3
        ])
        self.num_experts = len(self.experts)

        # Create the advanced router
        self.router = HeteroRouter(embed_dim, self.num_experts, top_k=top_k)

        # Register expert costs with the router
        costs = torch.tensor([float(getattr(e, "cost", 1.0)) for e in self.experts], dtype=torch.float32)
        self.router.register_buffer("costs", costs)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, expert_kwargs=None, compute_router_losses=False):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        topk_idx, topk_probs, all_probs, router_losses = self.router(x_flat, compute_losses=compute_router_losses)
        out_flat = torch.zeros_like(x_flat)

        for e_id, expert in enumerate(self.experts):
            mask = (topk_idx == e_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            expert_in = x_flat[mask]
            
            # Safely pass kwargs only to experts that can accept them
            if expert_kwargs is not None:
                try:
                    expert_out = expert(expert_in, **expert_kwargs)
                except TypeError:
                    expert_out = expert(expert_in)
            else:
                expert_out = expert(expert_in)
            
            # Calculate correct weights and combine
            topk_idx_masked = topk_idx[mask]
            topk_probs_masked = topk_probs[mask]
            equal_mat = (topk_idx_masked == e_id)
            gate_weights = (topk_probs_masked * equal_mat.float()).sum(dim=1, keepdim=True)
            out_flat.index_add_(0, mask.nonzero().squeeze(), expert_out * gate_weights)

        out = x + self.norm(out_flat.view(B, S, D))

        # We will need these diagnostics for the full loss calculation
        diagnostics = {}
        if compute_router_losses:
            diagnostics.update(router_losses)

        return out, diagnostics