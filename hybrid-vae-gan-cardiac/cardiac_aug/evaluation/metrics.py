from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from ..data.acdc import CLASS_NAMES


def _overlap(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...]) -> tuple[float, float]:
    if not a.any() and not b.any():
        return 0.0, 0.0
    if not a.any() or not b.any():
        # A defined worst-case penalty avoids silently dropping complete failures from averages.
        maximum = float(np.linalg.norm((np.asarray(a.shape) - 1) * np.asarray(spacing)))
        return maximum, maximum
    surface_a = np.logical_xor(a, binary_erosion(a))
    surface_b = np.logical_xor(b, binary_erosion(b))
    distance_to_b = distance_transform_edt(~surface_b, sampling=spacing)
    distance_to_a = distance_transform_edt(~surface_a, sampling=spacing)
    distances = np.concatenate([distance_to_b[surface_a], distance_to_a[surface_b]])
    return float(np.percentile(distances, 95)), float(distances.mean())


def case_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0), class_names=CLASS_NAMES
) -> dict[str, float]:
    result = {}
    for class_index, name in enumerate(class_names[1:], start=1):
        pred, true = prediction == class_index, target == class_index
        dice, iou = _overlap(pred, true)
        hd95, assd = _surface_distances(pred, true, tuple(spacing))
        result.update({f"dice_{name}": dice, f"iou_{name}": iou, f"hd95_{name}": hd95, f"assd_{name}": assd})
    for metric in ("dice", "iou", "hd95", "assd"):
        result[f"{metric}_mean"] = float(np.nanmean([result[f"{metric}_{n}"] for n in class_names[1:]]))
    return result


def aggregate_patient_predictions(records: list[dict]) -> list[dict]:
    """Compute metrics per phase, then average ED/ES results into one independent patient row."""
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["patient_id"], row["phase"])].append(row)
    phase_rows = []
    for (patient_id, phase), slices in grouped.items():
        slices.sort(key=lambda row: row["slice_index"])
        pred = np.stack([row["prediction"] for row in slices], axis=-1)
        target = np.stack([row["target"] for row in slices], axis=-1)
        xy = np.asarray(slices[0]["spacing"][:2], float)
        z = float(slices[0]["spacing"][2])
        metrics = case_metrics(pred, target, spacing=(*xy, z))
        phase_rows.append({"patient_id": patient_id, "phase": phase, **metrics})
    patients = []
    for patient_id in sorted({row["patient_id"] for row in phase_rows}):
        rows = [row for row in phase_rows if row["patient_id"] == patient_id]
        keys = [key for key in rows[0] if key not in {"patient_id", "phase"}]
        patients.append(
            {"patient_id": patient_id, **{key: float(np.nanmean([row[key] for row in rows])) for key in keys}}
        )
    return patients
