from __future__ import annotations

from torch import Tensor, nn


class CropEncoder(nn.Module):
    """CNN patch encoder with persistent freezing and spatial preservation."""

    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self._frozen = False
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        spatial_dim = 256 * 7 * 7
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, out_dim),
        )

    def forward(self, crops: Tensor) -> Tensor:
        return self.proj(self.features(crops))

    def freeze(self) -> None:
        self._frozen = True
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self._frozen = False
        self.train(True)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "CropEncoder":
        super().train(False if self._frozen else mode)
        return self
