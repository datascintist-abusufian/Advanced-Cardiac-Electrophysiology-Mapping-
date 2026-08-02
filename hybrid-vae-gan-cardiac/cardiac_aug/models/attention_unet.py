from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import AttentionGate, Down, ResidualBlock


class UpAttention(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 8))
        self.conv = ResidualBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attended, weights = self.attention(x, skip)
        return self.conv(torch.cat([x, attended], dim=1)), weights


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32, dropout: float = 0.2):
        super().__init__()
        self.inc = ResidualBlock(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4, dropout)
        self.down3 = Down(base * 4, base * 8, dropout)
        self.down4 = Down(base * 8, base * 16, dropout)
        self.up1 = UpAttention(base * 16, base * 8, base * 8, dropout)
        self.up2 = UpAttention(base * 8, base * 4, base * 4, dropout)
        self.up3 = UpAttention(base * 4, base * 2, base * 2)
        self.up4 = UpAttention(base * 2, base, base)
        self.head = nn.Conv2d(base, num_classes, 1)
        self.last_attention_maps: list[torch.Tensor] = []

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encode(x)[-1]
        return F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encode(x)
        y, a1 = self.up1(x5, x4)
        y, a2 = self.up2(y, x3)
        y, a3 = self.up3(y, x2)
        y, a4 = self.up4(y, x1)
        self.last_attention_maps = [a1, a2, a3, a4]
        return self.head(y)
