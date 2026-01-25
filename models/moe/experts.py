import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Heterogeneous Experts --- 

#Simple Expert
class Expert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.cost = 1.0  # Assign a relative computational cost

    def forward(self, x, **kwargs): # Added **kwargs for compatibility
        return self.net(x)

# Specialized Experts
class MotionExpert(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size//2, groups=embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.cost = 1.5

    def forward(self, x, **kwargs):
        x_t = x.unsqueeze(0).transpose(1, 2)
        y = self.temporal_conv(x_t).transpose(1, 2).squeeze(0)
        return self.fc(F.gelu(y))

class TextureExpert(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.cost = 2.0

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        y = self.conv(x)
        y = y.flatten(1)
        return self.fc(y)

class SceneExpert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.cost = 1.0

    def forward(self, x, **kwargs):
        pooled = x.mean(dim=0, keepdim=True)
        return self.fc(pooled.expand_as(x))

class FastChangeExpert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.cost = 0.8

    def forward(self, x, **kwargs):
        delta = x[1:] - x[:-1]
        delta = F.pad(delta, (0, 0, 1, 0))
        return self.fc(delta)

class LanguageAlignedExpert(nn.Module):
    def __init__(self, embed_dim, lang_dim=None):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(embed_dim if lang_dim is None else lang_dim, embed_dim)
        self.cost = 2.5

    def forward(self, x, **kwargs):
        text_embedding = kwargs.get('text_embedding') # Safely get text_embedding
        if text_embedding is None:
            return x
        query = x.unsqueeze(0)
        key_value = self.proj(text_embedding).unsqueeze(0)
        y, _ = self.cross_attn(query, key_value, key_value)
        return y.squeeze(0)