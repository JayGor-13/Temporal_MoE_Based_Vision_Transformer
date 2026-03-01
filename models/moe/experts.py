from torch import nn


class DenseFFN(nn.Module):
    """FFN(x) = W2 GELU(W1 x)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class Expert(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ffn = DenseFFN(embed_dim)

    def forward(self, x):
        return self.ffn(x)
