from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from .transforms import JointGeometricTransform, intensity_augment
from ..utils import write_json


LABEL_MAP = {0: 0, 3: 1, 2: 2, 1: 3}  # background, LV, myocardium, RV
CLASS_NAMES = ["background", "LV", "Myo", "RV"]


def _info(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                out[key.strip()] = value.strip()
    return out


def discover_patients(raw_dir: str | Path) -> list[dict]:
    raw = Path(raw_dir)
    records = []
    for patient_dir in sorted(raw.glob("patient[0-9][0-9][0-9]")):
        info = _info(patient_dir / "Info.cfg")
        patient_id = patient_dir.name
        frame_numbers = []
        if "ED" in info:
            frame_numbers.append(("ED", int(info["ED"])))
        if "ES" in info:
            frame_numbers.append(("ES", int(info["ES"])))
        if not frame_numbers:
            for gt in sorted(patient_dir.glob(f"{patient_id}_frame*_gt.nii.gz")):
                match = re.search(r"frame(\d+)_gt", gt.name)
                if match:
                    frame_numbers.append((f"F{match.group(1)}", int(match.group(1))))
        frames = []
        for phase, number in frame_numbers:
            image = patient_dir / f"{patient_id}_frame{number:02d}.nii.gz"
            mask = patient_dir / f"{patient_id}_frame{number:02d}_gt.nii.gz"
            if image.exists() and mask.exists():
                frames.append({"phase": phase, "image": str(image), "mask": str(mask)})
        if frames:
            records.append({"patient_id": patient_id, "diagnosis": info.get("Group", "unknown"), "frames": frames})
    if not records:
        raise FileNotFoundError(f"No labelled ACDC patients found under {raw.resolve()}")
    return records


def _resize_2d(array: np.ndarray, size: int, order: int) -> np.ndarray:
    factors = (size / array.shape[0], size / array.shape[1])
    return zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def _normalise_volume(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite != 0]
    values = foreground if foreground.size else finite
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def preprocess_acdc(raw_dir: str | Path, cache_dir: str | Path, image_size: int = 256) -> list[dict]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    for patient in discover_patients(raw_dir):
        images, masks, phases, slices, spacings = [], [], [], [], []
        for frame in patient["frames"]:
            image_nii = nib.load(frame["image"])
            image = np.asarray(image_nii.dataobj, dtype=np.float32)
            mask = np.asarray(nib.load(frame["mask"]).dataobj, dtype=np.int16)
            if image.shape != mask.shape:
                raise ValueError(f"Shape mismatch for {patient['patient_id']}: {image.shape} vs {mask.shape}")
            image = _normalise_volume(image)
            remapped = np.zeros_like(mask, dtype=np.uint8)
            for source, target in LABEL_MAP.items():
                remapped[mask == source] = target
            for z in range(image.shape[2]):
                images.append(_resize_2d(image[:, :, z], image_size, order=1).astype(np.float32))
                masks.append(_resize_2d(remapped[:, :, z], image_size, order=0).astype(np.uint8))
                phases.append(frame["phase"])
                slices.append(z)
                original_spacing = image_nii.header.get_zooms()[:3]
                spacings.append(
                    (
                        original_spacing[0] * image.shape[0] / image_size,
                        original_spacing[1] * image.shape[1] / image_size,
                        original_spacing[2],
                    )
                )
        output = cache / f"{patient['patient_id']}.npz"
        np.savez_compressed(
            output,
            images=np.stack(images),
            masks=np.stack(masks),
            phases=np.asarray(phases),
            slice_indices=np.asarray(slices),
            spacings=np.asarray(spacings),
        )
        manifest.append(
            {
                "patient_id": patient["patient_id"],
                "diagnosis": patient["diagnosis"],
                "path": str(output.resolve()),
                "n_slices": len(images),
            }
        )
    write_json({"image_size": image_size, "class_names": CLASS_NAMES, "patients": manifest}, cache / "manifest.json")
    return manifest


def build_patient_folds(
    manifest: list[dict],
    output: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.15,
    stratify: bool = True,
) -> list[dict]:
    ids = np.asarray([p["patient_id"] for p in manifest])
    labels = np.asarray([p.get("diagnosis", "unknown") for p in manifest])
    can_stratify = stratify and all(np.sum(labels == value) >= n_splits for value in np.unique(labels))
    splitter = (
        StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        if can_stratify
        else KFold(n_splits, shuffle=True, random_state=seed)
    )
    folds = []
    iterator = splitter.split(ids, labels) if can_stratify else splitter.split(ids)
    for fold, (train_val_idx, test_idx) in enumerate(iterator):
        tv_ids, tv_labels = ids[train_val_idx], labels[train_val_idx]
        strat_labels = tv_labels if can_stratify and min(np.unique(tv_labels, return_counts=True)[1]) >= 2 else None
        train_ids, val_ids = train_test_split(
            tv_ids, test_size=validation_fraction, random_state=seed + fold, stratify=strat_labels
        )
        split = {
            "fold": fold,
            "seed": seed,
            "train": sorted(train_ids.tolist()),
            "validation": sorted(val_ids.tolist()),
            "test": sorted(ids[test_idx].tolist()),
            "stratified": can_stratify,
        }
        assert not (
            set(split["train"]) & set(split["validation"])
            | set(split["train"]) & set(split["test"])
            | set(split["validation"]) & set(split["test"])
        )
        folds.append(split)
    write_json(folds, output)
    return folds


class ACDCSliceDataset(Dataset):
    def __init__(
        self,
        manifest: Iterable[dict],
        patient_ids: Iterable[str],
        augment: bool = False,
        min_foreground_pixels: int = 0,
    ):
        selected = set(patient_ids)
        self.samples = []
        self.augment = augment
        self.spatial = JointGeometricTransform()
        for record in manifest:
            if record["patient_id"] not in selected:
                continue
            data = np.load(record["path"])
            for index in range(len(data["images"])):
                if np.count_nonzero(data["masks"][index]) >= min_foreground_pixels:
                    self.samples.append((record, index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict:
        record, index = self.samples[item]
        data = np.load(record["path"])
        image = torch.from_numpy(data["images"][index]).float().unsqueeze(0)
        mask = torch.from_numpy(data["masks"][index].astype(np.int64))
        if self.augment:
            image, mask = self.spatial(image, mask)
            image = intensity_augment(image)
        return {
            "image": image,
            "mask": mask,
            "patient_id": record["patient_id"],
            "phase": str(data["phases"][index]),
            "slice_index": int(data["slice_indices"][index]),
            "spacing": torch.from_numpy(data["spacings"][index]).float(),
        }


class SyntheticPairDataset(Dataset):
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.images = data["images"]
        self.masks = data["masks"]
        self.sources = data["source_patient_ids"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": torch.from_numpy(self.images[index]).float().unsqueeze(0),
            "mask": torch.from_numpy(self.masks[index]).long(),
            "patient_id": f"synthetic:{self.sources[index]}",
            "phase": "SYN",
            "slice_index": index,
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
        }
