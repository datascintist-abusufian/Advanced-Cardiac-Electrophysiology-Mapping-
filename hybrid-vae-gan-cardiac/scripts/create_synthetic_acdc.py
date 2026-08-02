#!/usr/bin/env python
"""Create tiny ACDC-shaped NIfTI data for pipeline testing only, never scientific results."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="work/synthetic_acdc")
parser.add_argument("--patients", type=int, default=10)
parser.add_argument("--size", type=int, default=64)
args = parser.parse_args()
root = Path(args.output)
rng = np.random.default_rng(42)
y, x = np.mgrid[: args.size, : args.size]
for patient_index in range(1, args.patients + 1):
    folder = root / f"patient{patient_index:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Info.cfg").write_text(f"ED: 1\nES: 2\nGroup: SYN{patient_index % 2}\n", encoding="utf-8")
    for frame, contraction in [(1, 1.0), (2, 0.78)]:
        volume = np.zeros((args.size, args.size, 3), np.float32)
        mask = np.zeros_like(volume, np.uint8)
        for z in range(3):
            cx, cy = args.size / 2 + rng.normal(0, 1), args.size / 2 + rng.normal(0, 1)
            radius = contraction * (10 + patient_index % 3 - abs(z - 1) * 2)
            lv = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            outer = (x - cx) ** 2 + (y - cy) ** 2 <= (radius + 3) ** 2
            rv = ((x - (cx + radius + 4)) / 6) ** 2 + ((y - cy) / 8) ** 2 <= 1
            mask[:, :, z][rv] = 1
            mask[:, :, z][outer & ~lv] = 2
            mask[:, :, z][lv] = 3
            volume[:, :, z] = 0.08 + 0.30 * outer + 0.50 * lv + 0.20 * rv + rng.normal(0, 0.03, (args.size, args.size))
        affine = np.diag([1.4, 1.4, 8.0, 1.0])
        prefix = folder / f"patient{patient_index:03d}_frame{frame:02d}"
        nib.save(nib.Nifti1Image(volume, affine), f"{prefix}.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), f"{prefix}_gt.nii.gz")
print(f"Created synthetic test fixture at {root}; do not use it for scientific claims.")
