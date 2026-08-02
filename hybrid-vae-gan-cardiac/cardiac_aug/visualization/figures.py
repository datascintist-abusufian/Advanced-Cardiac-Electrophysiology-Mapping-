from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

from ..evaluation.explainability import attention_gate_map, segmentation_input_attribution
from ..models import AttentionUNet


COLORS = ListedColormap([(0, 0, 0, 0), (0.9, 0.1, 0.1, 0.65), (0.1, 0.8, 0.2, 0.65), (0.1, 0.3, 0.95, 0.65)])


def performance_figure(metrics_csv, output_path):
    frame = pd.read_csv(metrics_csv)
    order = [
        name
        for name in ["baseline", "geometric", "vae", "gan", "hybrid_no_qc", "hybrid_qc"]
        if name in frame.condition.unique()
    ]
    summary = frame.groupby("condition")[["dice_LV", "dice_Myo", "dice_RV"]].agg(["mean", "std"])
    x = np.arange(3)
    width = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, condition in enumerate(order):
        means = summary.loc[condition].xs("mean", level=1).values
        stds = summary.loc[condition].xs("std", level=1).values
        ax.bar(
            x + (index - (len(order) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=condition.replace("_", " "),
        )
    ax.set(
        xticks=x,
        xticklabels=["LV", "Myocardium", "RV"],
        ylabel="Patient-level Dice",
        ylim=(0, 1),
        title="Cardiac MRI segmentation performance",
    )
    ax.legend(ncol=3, frameon=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _overlay(ax, image, mask, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=COLORS, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def qualitative_figure(
    sample, checkpoints: dict[str, str | Path], device, output_path, base_channels=32, num_classes=4
):
    image, target = sample["image"].unsqueeze(0).to(device), sample["mask"].numpy()
    columns = 2 + len(checkpoints) + 2
    fig, axes = plt.subplots(1, columns, figsize=(3 * columns, 3), constrained_layout=True)
    axes[0].imshow(image[0, 0].cpu(), cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    _overlay(axes[1], image[0, 0].cpu(), target, "Ground truth")
    last_model = None
    for ax, (name, checkpoint) in zip(axes[2:], checkpoints.items()):
        model = AttentionUNet(num_classes=num_classes, base=base_channels).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            prediction = model(image).argmax(1)[0].cpu().numpy()
        _overlay(ax, image[0, 0].cpu(), prediction, name.replace("_", " "))
        last_model = model
    if last_model is not None:
        attribution = segmentation_input_attribution(last_model, image, class_index=1)[0, 0].detach().cpu()
        attention = attention_gate_map(last_model, image)[0, 0].cpu()
        axes[-2].imshow(image[0, 0].cpu(), cmap="gray")
        axes[-2].imshow(attribution, cmap="inferno", alpha=0.6)
        axes[-2].set_title("LV attribution")
        axes[-2].axis("off")
        axes[-1].imshow(image[0, 0].cpu(), cmap="gray")
        axes[-1].imshow(attention, cmap="viridis", alpha=0.6)
        axes[-1].set_title("Attention gate")
        axes[-1].axis("off")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
