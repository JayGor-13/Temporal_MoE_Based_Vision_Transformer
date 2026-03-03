import torch
from torch import nn


class VideoEmbedder(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int = 16, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        tokens_per_frame = (img_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, 1 + tokens_per_frame * num_frames, embed_dim))

    def _position_slice(self, token_count: int) -> torch.Tensor:
        if token_count <= self.pos.size(1):
            return self.pos[:, :token_count]

        # If runtime frame count is larger than configured count, extend by reusing
        # the last learned position embedding instead of failing hard.
        extra = token_count - self.pos.size(1)
        tail = self.pos[:, -1:, :].expand(1, extra, -1)
        return torch.cat([self.pos, tail], dim=1)

    def forward(self, x: torch.Tensor):
        # [B,T,3,H,W]
        b, t, c, h, w = x.shape
        patches = []
        for i in range(t):
            p = self.proj(x[:, i]).flatten(2).transpose(1, 2)
            patches.append(p)
        x = torch.cat(patches, dim=1)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self._position_slice(x.size(1))
