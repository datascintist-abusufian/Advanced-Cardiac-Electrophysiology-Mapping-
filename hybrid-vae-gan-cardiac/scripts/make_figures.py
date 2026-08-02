#!/usr/bin/env python
import argparse
from pathlib import Path

from cardiac_aug.config import load_config
from cardiac_aug.data.acdc import ACDCSliceDataset
from cardiac_aug.utils import read_json, resolve_device
from cardiac_aug.visualization.figures import performance_figure, qualitative_figure


parser = argparse.ArgumentParser(description="Create performance, qualitative, and explanation figures.")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--output-dir", default="runs/summary/figures")
args = parser.parse_args()
config = load_config(args.config)
output = Path(args.output_dir)
performance_figure(Path(config.output_dir) / "summary/patient_level_metrics.csv", output / "performance.png")
fold_dir = Path(config.output_dir) / f"seed_{args.seed}" / f"fold_{args.fold}"
split, manifest = (
    read_json(fold_dir / "split.json"),
    read_json(Path(config.data.cache_dir) / "manifest.json")["patients"],
)
dataset = ACDCSliceDataset(manifest, split["test"])
sample = next((dataset[i] for i in range(len(dataset)) if (dataset[i]["mask"] > 0).sum() > 100), dataset[0])
checkpoints = {
    name: fold_dir / "conditions" / name / "best.pt"
    for name in config.conditions
    if (fold_dir / "conditions" / name / "best.pt").exists()
}
qualitative_figure(
    sample,
    checkpoints,
    resolve_device(config.device),
    output / "qualitative_and_xai.png",
    config.model.base_channels,
    config.data.num_classes,
)
print(f"Wrote figures to {output}")
