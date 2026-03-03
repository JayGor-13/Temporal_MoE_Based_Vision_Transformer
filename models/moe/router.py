import torch
from torch import nn


class CaptionConditionedRouter(nn.Module):
    """g = Softmax(W_r [h_video ; h_text]) with top-k routing."""

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2, no_caption_conditioning: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.no_caption_conditioning = no_caption_conditioning
        self.proj = nn.Linear(embed_dim * 2, num_experts)
        self.last_probs = None

    def forward(self, video_tokens: torch.Tensor, text_state: torch.Tensor):
        # video_tokens: [B,S,D], text_state: [B,D]
        h_video = video_tokens.mean(dim=1)
        if self.no_caption_conditioning:
            text_state = torch.zeros_like(text_state)

        logits = self.proj(torch.cat([h_video, text_state], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        self.last_probs = probs.detach()
        topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        aux = probs.var(dim=0, unbiased=False).mean()
        return topi, topv, probs, {"routing_entropy": entropy, "load_balance_loss": aux}
