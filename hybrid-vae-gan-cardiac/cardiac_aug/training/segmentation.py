from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .logging import CSVLogger
from ..evaluation.metrics import aggregate_patient_predictions


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = logits.softmax(1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs[:, 1:] * one_hot[:, 1:]).sum(dims)
    denominator = probs[:, 1:].sum(dims) + one_hot[:, 1:].sum(dims)
    return 1.0 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()


def _loss(logits, masks):
    return F.cross_entropy(logits, masks) + soft_dice_loss(logits, masks, logits.shape[1])


@torch.no_grad()
def validation_dice(model, loader, device) -> float:
    model.eval()
    scores = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        pred, target = logits.argmax(1), batch["mask"].to(device)
        for cls in range(1, logits.shape[1]):
            inter = ((pred == cls) & (target == cls)).sum().item()
            denom = (pred == cls).sum().item() + (target == cls).sum().item()
            scores.append(1.0 if denom == 0 else 2 * inter / denom)
    return float(np.mean(scores))


def train_segmentation(
    model,
    train_loader,
    validation_loader,
    device,
    output_dir,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-5,
    patience=15,
    amp=True,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(output / "training.csv", ["epoch", "train_loss", "validation_dice", "seconds"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_score, best_state, stale = -1.0, None, 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for batch in train_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss = _loss(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(loss.item())
        score = validation_dice(model, validation_loader, device)
        logger.log(
            epoch=epoch, train_loss=np.mean(losses), validation_dice=score, seconds=time.perf_counter() - started
        )
        if score > best_score:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
            torch.save({"model": best_state, "epoch": epoch, "validation_dice": score}, output / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_validation_dice": best_score, "epochs_completed": epoch}


@torch.no_grad()
def evaluate_segmentation(model, loader, device, output_csv: str | Path):
    model.eval()
    records = []
    for batch in loader:
        predictions = model(batch["image"].to(device)).argmax(1).cpu().numpy()
        targets = batch["mask"].numpy()
        for i in range(len(predictions)):
            spacing = batch["spacing"][i].numpy().tolist()
            records.append(
                {
                    "patient_id": batch["patient_id"][i],
                    "phase": batch["phase"][i],
                    "slice_index": int(batch["slice_index"][i]),
                    "spacing": spacing,
                    "prediction": predictions[i],
                    "target": targets[i],
                }
            )
    rows = aggregate_patient_predictions(records)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows
