import torch
import torch.nn as nn
from ..config import EMBED_DIM, NUM_HEADS, DEVICE
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        return res + self.drop(attn_out)

# Test attention
attn_block = MultiHeadSelfAttention(EMBED_DIM, NUM_HEADS).to(DEVICE)
attn_out = attn_block(tokens)
print(f"After Attention: {attn_out.shape}")
# → [2, 1569, 512]