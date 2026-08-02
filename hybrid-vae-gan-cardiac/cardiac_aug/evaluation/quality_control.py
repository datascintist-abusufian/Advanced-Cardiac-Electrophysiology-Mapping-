from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA

from ..utils import write_json


def frechet_distance(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    if len(real_features) < 2 or len(generated_features) < 2:
        return float("nan")
    mu_r, mu_g = real_features.mean(0), generated_features.mean(0)
    cov_r, cov_g = np.cov(real_features, rowvar=False), np.cov(generated_features, rowvar=False)
    covariance_mean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2 * covariance_mean))


def _dice(prediction: np.ndarray, target: np.ndarray, classes=range(1, 4)) -> float:
    values = []
    for label in classes:
        a, b = prediction == label, target == label
        denominator = a.sum() + b.sum()
        if denominator:
            values.append(2 * np.logical_and(a, b).sum() / denominator)
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def _features(model, images: np.ndarray, device, batch_size=16) -> np.ndarray:
    output = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(images[start : start + batch_size, None]).float().to(device)
        output.append(model.extract_features(x).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def quality_filter(
    candidate_path,
    real_loader,
    teacher,
    device,
    output_path,
    ssim_min=0.5,
    ssim_max=0.995,
    anatomy_dice_min=0.7,
    fid_max=50.0,
    target_count=200,
    min_retained=25,
):
    """QC uses sample-level SSIM/anatomy and dataset-level domain-feature FID.

    FID is never treated as a per-image score. If the candidate set exceeds the configured
    dataset-level limit, feature-space outliers are removed and FID is recomputed.
    """
    candidates = np.load(candidate_path)
    images, masks, originals = candidates["images"], candidates["masks"], candidates["originals"]
    sources = candidates["source_patient_ids"]
    teacher.to(device).eval()
    ssim_values = np.asarray([structural_similarity(a, b, data_range=1.0) for a, b in zip(images, originals)])
    anatomy = []
    for start in range(0, len(images), 16):
        x = torch.from_numpy(images[start : start + 16, None]).float().to(device)
        pred = teacher(x).argmax(1).cpu().numpy()
        anatomy.extend(_dice(a, b) for a, b in zip(pred, masks[start : start + 16]))
    anatomy = np.asarray(anatomy)
    accepted = np.flatnonzero((ssim_values >= ssim_min) & (ssim_values <= ssim_max) & (anatomy >= anatomy_dice_min))
    real_images = []
    for batch in real_loader:
        real_images.extend(batch["image"][:, 0].numpy())
    real_images = np.asarray(real_images)
    real_features = _features(teacher, real_images, device)
    generated_features = _features(teacher, images, device)
    # Domain features can be high-dimensional. Fit dimensionality reduction on training-real
    # features only, both for numerical stability and to keep repeated set-level FID practical.
    feature_dimensions = min(64, len(real_features) - 1, real_features.shape[1])
    if feature_dimensions >= 2:
        reducer = PCA(n_components=feature_dimensions, random_state=0).fit(real_features)
        real_features = reducer.transform(real_features)
        generated_features = reducer.transform(generated_features)
    initial_fid = frechet_distance(real_features, generated_features[accepted]) if len(accepted) >= 2 else float("nan")
    # Rank only after sample filters; repeatedly recompute a dataset-level FID.
    if len(accepted):
        centre = real_features.mean(0)
        distances = np.linalg.norm(generated_features[accepted] - centre, axis=1)
        accepted = accepted[np.argsort(distances)]
    accepted = accepted[:target_count]
    while len(accepted) > min_retained:
        current_fid = frechet_distance(real_features, generated_features[accepted])
        if np.isfinite(current_fid) and current_fid <= fid_max:
            break
        accepted = accepted[: -max(1, min(10, len(accepted) - min_retained))]
    final_fid = frechet_distance(real_features, generated_features[accepted]) if len(accepted) >= 2 else float("nan")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=images[accepted],
        masks=masks[accepted],
        originals=originals[accepted],
        source_patient_ids=sources[accepted],
    )
    report = {
        "candidate_count": len(images),
        "sample_filter_count": int(
            np.sum((ssim_values >= ssim_min) & (ssim_values <= ssim_max) & (anatomy >= anatomy_dice_min))
        ),
        "retained_count": len(accepted),
        "initial_dataset_fid": initial_fid,
        "final_dataset_fid": final_fid,
        "fid_feature_source": "training-only segmentation encoder",
        "fid_passed": bool(np.isfinite(final_fid) and final_fid <= fid_max),
        "ssim_mean": float(ssim_values.mean()),
        "anatomy_dice_mean": float(anatomy.mean()),
        "thresholds": {
            "ssim_min": ssim_min,
            "ssim_max": ssim_max,
            "anatomy_dice_min": anatomy_dice_min,
            "fid_max": fid_max,
        },
    }
    write_json(report, path.with_suffix(".qc.json"))
    return report
