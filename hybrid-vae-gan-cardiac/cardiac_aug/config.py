from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/ACDC/database/training"
    cache_dir: str = "data/processed"
    image_size: int = 256
    num_classes: int = 4
    min_foreground_pixels: int = 0
    include_ed_es_only: bool = True


@dataclass
class CVConfig:
    n_splits: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 2026])
    validation_fraction: float = 0.15
    stratify_by_diagnosis: bool = True


@dataclass
class TrainConfig:
    epochs_seg: int = 100
    epochs_vae: int = 200
    epochs_gan: int = 200
    batch_size_seg: int = 8
    batch_size_gen: int = 16
    lr_seg: float = 1e-4
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    weight_decay: float = 1e-5
    patience: int = 15
    num_workers: int = 4
    amp: bool = True


@dataclass
class ModelConfig:
    latent_dim: int = 128
    base_channels: int = 32
    dropout: float = 0.2
    vae_beta: float = 1.0
    latent_noise_std: float = 0.1
    gan_l1_weight: float = 100.0


@dataclass
class QCConfig:
    ssim_min: float = 0.50
    ssim_max: float = 0.995
    anatomy_dice_min: float = 0.70
    fid_max: float = 50.0
    target_synthetic: int = 200
    candidate_multiplier: int = 2
    min_retained: int = 25


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    output_dir: str = "runs"
    device: str = "auto"
    conditions: list[str] = field(
        default_factory=lambda: ["baseline", "geometric", "vae", "gan", "hybrid_no_qc", "hybrid_qc"]
    )


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return _merge_dataclass(ExperimentConfig(), values)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
