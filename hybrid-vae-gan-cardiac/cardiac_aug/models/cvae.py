from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def one_hot_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == num_classes:
        return mask.float()
    if mask.ndim == 4:
        mask = mask[:, 0]
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()


class MaskConditionedVAE(nn.Module):
    """VAE that synthesises intensity appearance while retaining a supplied anatomy mask."""

    def __init__(self, image_size: int = 256, num_classes: int = 4, latent_dim: int = 128, base: int = 32):
        super().__init__()
        if image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.image_size, self.num_classes, self.latent_dim = image_size, num_classes, latent_dim
        channels = [base, base * 2, base * 4, base * 8]
        layers, incoming = [], 1 + num_classes
        for outgoing in channels:
            layers += [nn.Conv2d(incoming, outgoing, 4, 2, 1), nn.BatchNorm2d(outgoing), nn.LeakyReLU(0.2)]
            incoming = outgoing
        self.encoder = nn.Sequential(*layers)
        spatial = image_size // 16
        flat = channels[-1] * spatial * spatial
        self.to_mu, self.to_logvar = nn.Linear(flat, latent_dim), nn.Linear(flat, latent_dim)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(num_classes, base, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 4, 2, 1),
            nn.ReLU(),
        )
        self.from_z = nn.Linear(latent_dim, base * 4 * spatial * spatial)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.ConvTranspose2d(base, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor, mask: torch.Tensor):
        one_hot = one_hot_mask(mask, self.num_classes)
        features = self.encoder(torch.cat([image, one_hot], dim=1)).flatten(1)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, mask: torch.Tensor):
        one_hot = one_hot_mask(mask, self.num_classes)
        spatial = self.image_size // 16
        z_map = self.from_z(z).view(z.shape[0], -1, spatial, spatial)
        return self.decoder(torch.cat([z_map, self.mask_encoder(one_hot)], dim=1))

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(image, mask)
        return self.decode(self.reparameterise(mu, logvar), mask), mu, logvar

    @torch.no_grad()
    def perturb(self, image: torch.Tensor, mask: torch.Tensor, std: float = 0.1):
        mu, _ = self.encode(image, mask)
        return self.decode(mu + std * torch.randn_like(mu), mask)


def vae_loss(reconstruction, target, mu, logvar, beta: float = 1.0):
    reconstruction_loss = F.l1_loss(reconstruction, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + beta * kl, {"reconstruction": reconstruction_loss.detach(), "kl": kl.detach()}
