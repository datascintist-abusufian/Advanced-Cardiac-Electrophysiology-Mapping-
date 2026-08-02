from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .logging import CSVLogger
from ..models.cvae import vae_loss


def train_vae(model, loader, device, output_dir, epochs=200, lr=1e-4, beta=1.0, amp=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(output / "training.csv", ["epoch", "loss", "reconstruction", "kl", "seconds"])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        losses, recons, kls = [], [], []
        started = time.perf_counter()
        for batch in loader:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                reconstruction, mu, logvar = model(image, mask)
                loss, parts = vae_loss(reconstruction, image, mu, logvar, beta)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(loss.item())
            recons.append(parts["reconstruction"].item())
            kls.append(parts["kl"].item())
        logger.log(
            epoch=epoch,
            loss=np.mean(losses),
            reconstruction=np.mean(recons),
            kl=np.mean(kls),
            seconds=time.perf_counter() - started,
        )
    torch.save({"model": model.state_dict(), "epochs": epochs}, output / "final.pt")


def corrupt_image(image: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool2d(image, 3, stride=1, padding=1)
    return torch.clamp(0.7 * pooled + 0.3 * image + 0.05 * torch.randn_like(image), 0, 1)


def train_refiner(
    generator,
    discriminator,
    loader,
    device,
    output_dir,
    epochs=200,
    lr_g=1e-4,
    lr_d=4e-4,
    l1_weight=100.0,
    vae=None,
    amp=True,
):
    """Train PatchGAN refinement; VAE=None gives the GAN-only denoising condition."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(output / "training.csv", ["epoch", "generator", "discriminator", "l1", "seconds"])
    generator.to(device)
    discriminator.to(device)
    if vae is not None:
        vae.to(device).eval()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_values, d_values, l1_values = [], [], []
        started = time.perf_counter()
        for batch in loader:
            real, mask = batch["image"].to(device), batch["mask"].to(device)
            with torch.no_grad():
                coarse = vae.perturb(real, mask) if vae is not None else corrupt_image(real)
            d_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                fake = generator(coarse, mask)
                real_logits, fake_logits = discriminator(real, mask), discriminator(fake.detach(), mask)
                d_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                )
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)
            scaler.update()
            g_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                fake = generator(coarse, mask)
                fake_logits = discriminator(fake, mask)
                adversarial = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                l1 = F.l1_loss(fake, real)
                g_loss = adversarial + l1_weight * l1
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()
            g_values.append(g_loss.item())
            d_values.append(d_loss.item())
            l1_values.append(l1.item())
        logger.log(
            epoch=epoch,
            generator=np.mean(g_values),
            discriminator=np.mean(d_values),
            l1=np.mean(l1_values),
            seconds=time.perf_counter() - started,
        )
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "epochs": epochs,
            "conditioning": "vae" if vae is not None else "corrupted_real",
        },
        output / "final.pt",
    )


@torch.no_grad()
def generate_candidates(loader, output_path, device, count, mode, vae=None, refiner=None, latent_std=0.1):
    if mode not in {"vae", "gan", "hybrid"}:
        raise ValueError(f"Unknown generation mode: {mode}")
    for model in (vae, refiner):
        if model is not None:
            model.to(device).eval()
    images, masks, sources, originals = [], [], [], []
    while len(images) < count:
        for batch in loader:
            real, mask = batch["image"].to(device), batch["mask"].to(device)
            if mode == "vae":
                generated = vae.perturb(real, mask, latent_std)
            elif mode == "gan":
                generated = refiner(corrupt_image(real), mask)
            else:
                generated = refiner(vae.perturb(real, mask, latent_std), mask)
            for i in range(len(generated)):
                images.append(generated[i, 0].cpu().numpy().astype(np.float32))
                masks.append(mask[i].cpu().numpy().astype(np.uint8))
                originals.append(real[i, 0].cpu().numpy().astype(np.float32))
                sources.append(batch["patient_id"][i])
                if len(images) >= count:
                    break
            if len(images) >= count:
                break
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack(images),
        masks=np.stack(masks),
        originals=np.stack(originals),
        source_patient_ids=np.asarray(sources),
    )
    return path
