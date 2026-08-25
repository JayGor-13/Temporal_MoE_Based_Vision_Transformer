from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """Per-crop objectness, box regression, and sub-pixel offset head."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.objectness_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))
        self.box_stem = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.xy_head = nn.Linear(hidden_dim, 2)
        self.wh_head = nn.Linear(hidden_dim, 2)
        self.offset_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.box_stem(features)
        xy = torch.sigmoid(self.xy_head(stem))
        wh = F.softplus(self.wh_head(stem)) + 1e-4
        boxes = torch.cat([xy, wh], dim=-1)
        offsets = self.offset_head(features) * 0.5
        return self.objectness_head(features), boxes, offsets
