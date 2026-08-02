from __future__ import annotations

import random

import torch
import torch.nn.functional as F


class JointGeometricTransform:
    """Apply exactly the same spatial transform to an image and discrete mask."""

    def __init__(self, flip_p: float = 0.5, rotate90_p: float = 0.5, affine_p: float = 0.5):
        self.flip_p = flip_p
        self.rotate90_p = rotate90_p
        self.affine_p = affine_p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.flip_p:
            dims = [-1] if random.random() < 0.5 else [-2]
            image, mask = torch.flip(image, dims), torch.flip(mask, dims)
        if random.random() < self.rotate90_p:
            k = random.randint(1, 3)
            image, mask = torch.rot90(image, k, (-2, -1)), torch.rot90(mask, k, (-2, -1))
        if random.random() < self.affine_p:
            angle = random.uniform(-10.0, 10.0) * 3.141592653589793 / 180.0
            scale = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
            theta = image.new_tensor(
                [
                    [scale * torch.cos(image.new_tensor(angle)), -scale * torch.sin(image.new_tensor(angle)), tx],
                    [scale * torch.sin(image.new_tensor(angle)), scale * torch.cos(image.new_tensor(angle)), ty],
                ]
            )
            theta = theta.unsqueeze(0)
            grid = F.affine_grid(theta, (1, 1, *image.shape[-2:]), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            mask = F.grid_sample(
                mask.float()[None, None], grid, mode="nearest", padding_mode="zeros", align_corners=False
            )[0, 0].long()
        return image, mask


def intensity_augment(image: torch.Tensor) -> torch.Tensor:
    gamma = random.uniform(0.8, 1.2)
    noise = torch.randn_like(image) * random.uniform(0.0, 0.03)
    return torch.clamp(image.pow(gamma) + noise, 0.0, 1.0)
