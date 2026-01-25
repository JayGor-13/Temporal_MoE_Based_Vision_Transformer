import torch
import torch.nn as nn
from einops import rearrange
from ..config import IMG_SIZE, EMBED_DIM, NUM_FRAMES, PATCH_SIZE, DEVICE


class PatchEmbed(nn.Module):
    def __init__(self, frames, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=frames * 3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: [B, T, 3, H, W]
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.proj(x)                         # [B, D, H/p, W/p]
        x = rearrange(x, 'b d ph pw -> b (ph pw) d')
        return x

# CLS token + positional embeddings
class VideoEmbedder(nn.Module):
    def __init__(self, embed_dim, num_frames, patch_size):
        super().__init__()
        self.patch_embed = PatchEmbed(num_frames, embed_dim, patch_size)
        num_patches = (IMG_SIZE // patch_size) ** 2 * num_frames
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embed(x)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.dropout(x)

# Test embedding
# embedder = VideoEmbedder(EMBED_DIM, NUM_FRAMES, PATCH_SIZE).to(DEVICE)
# tokens = embedder(videos)
# print(f"Patch-embedded tokens shape: {tokens.shape}")