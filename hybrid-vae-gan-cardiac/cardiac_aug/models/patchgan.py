from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .cvae import one_hot_mask


class UNetDown(nn.Module):
    def __init__(self, incoming, outgoing, normalise=True):
        super().__init__()
        layers = [nn.Conv2d(incoming, outgoing, 4, 2, 1, bias=not normalise)]
        if normalise:
            layers.append(nn.InstanceNorm2d(outgoing))
        layers.append(nn.LeakyReLU(0.2, True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(incoming, outgoing, 4, 2, 1), nn.InstanceNorm2d(outgoing), nn.ReLU(True)
        )

    def forward(self, x, skip):
        x = self.block(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)


class ConditionalRefiner(nn.Module):
    """U-Net generator conditioned on a coarse MRI and its exact one-hot mask."""

    def __init__(self, num_classes=4, base=32):
        super().__init__()
        incoming = 1 + num_classes
        self.num_classes = num_classes
        self.d1 = UNetDown(incoming, base, False)
        self.d2 = UNetDown(base, base * 2)
        self.d3 = UNetDown(base * 2, base * 4)
        self.d4 = UNetDown(base * 4, base * 8)
        self.bottleneck = UNetDown(base * 8, base * 8)
        self.u1 = UNetUp(base * 8, base * 8)
        self.u2 = UNetUp(base * 16, base * 4)
        self.u3 = UNetUp(base * 8, base * 2)
        self.u4 = UNetUp(base * 4, base)
        self.out = nn.Sequential(nn.ConvTranspose2d(base * 2, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, coarse, mask):
        x = torch.cat([coarse, one_hot_mask(mask, self.num_classes)], 1)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.bottleneck(d4)
        return self.out(self.u4(self.u3(self.u2(self.u1(b, d4), d3), d2), d1))


class PatchDiscriminator(nn.Module):
    def __init__(self, num_classes=4, base=32):
        super().__init__()
        self.num_classes = num_classes
        incoming = 1 + num_classes
        self.net = nn.Sequential(
            nn.Conv2d(incoming, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1),
            nn.InstanceNorm2d(base * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, image, mask):
        return self.net(torch.cat([image, one_hot_mask(mask, self.num_classes)], 1))
