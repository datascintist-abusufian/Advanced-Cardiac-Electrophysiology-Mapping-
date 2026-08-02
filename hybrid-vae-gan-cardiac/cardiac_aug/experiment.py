from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import ExperimentConfig, save_config
from .data.acdc import ACDCSliceDataset, SyntheticPairDataset, build_patient_folds
from .evaluation.quality_control import quality_filter
from .models import AttentionUNet, ConditionalRefiner, MaskConditionedVAE, PatchDiscriminator
from .training.generative import generate_candidates, train_refiner, train_vae
from .training.segmentation import evaluate_segmentation, train_segmentation
from .utils import provenance, read_json, seed_everything, write_json


CONDITION_SEED_OFFSET = {"baseline": 0, "geometric": 1, "vae": 2, "gan": 3, "hybrid_no_qc": 4, "hybrid_qc": 5}


def _loader(dataset, config, shuffle=False, generative=False):
    batch = config.train.batch_size_gen if generative else config.train.batch_size_seg
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
    )


def _new_segmenter(config):
    return AttentionUNet(
        num_classes=config.data.num_classes, base=config.model.base_channels, dropout=config.model.dropout
    )


def _train_condition(condition, dataset, val_loader, test_loader, config, device, directory, seed):
    seed_everything(seed)
    model = _new_segmenter(config)
    started = time.perf_counter()
    result = train_segmentation(
        model,
        _loader(dataset, config, shuffle=True),
        val_loader,
        device,
        directory,
        epochs=config.train.epochs_seg,
        lr=config.train.lr_seg,
        weight_decay=config.train.weight_decay,
        patience=config.train.patience,
        amp=config.train.amp,
    )
    result["wall_clock_hours"] = (time.perf_counter() - started) / 3600
    result["condition"] = condition
    result["training_samples"] = len(dataset)
    evaluate_segmentation(model, test_loader, device, Path(directory) / "patient_metrics.csv")
    write_json(result, Path(directory) / "runtime.json")
    return model, result


def run_fold(config: ExperimentConfig, seed: int, fold_index: int, device: torch.device):
    manifest_doc = read_json(Path(config.data.cache_dir) / "manifest.json")
    manifest = manifest_doc["patients"]
    fold_path = Path(config.output_dir) / "splits" / f"seed_{seed}.json"
    folds = (
        read_json(fold_path)
        if fold_path.exists()
        else build_patient_folds(
            manifest,
            fold_path,
            config.cv.n_splits,
            seed,
            config.cv.validation_fraction,
            config.cv.stratify_by_diagnosis,
        )
    )
    split = folds[fold_index]
    fold_dir = Path(config.output_dir) / f"seed_{seed}" / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(split, fold_dir / "split.json")
    base_train = ACDCSliceDataset(
        manifest, split["train"], augment=False, min_foreground_pixels=config.data.min_foreground_pixels
    )
    geometric_train = ACDCSliceDataset(
        manifest, split["train"], augment=True, min_foreground_pixels=config.data.min_foreground_pixels
    )
    generative_train = ACDCSliceDataset(
        manifest, split["train"], augment=True, min_foreground_pixels=max(1, config.data.min_foreground_pixels)
    )
    validation = ACDCSliceDataset(manifest, split["validation"], augment=False)
    test = ACDCSliceDataset(manifest, split["test"], augment=False)
    val_loader, test_loader = _loader(validation, config), _loader(test, config)
    condition_dir = fold_dir / "conditions"

    # A fold-local baseline is also the frozen anatomy/FID teacher for QC.
    baseline_model, _ = _train_condition(
        "baseline",
        base_train,
        val_loader,
        test_loader,
        config,
        device,
        condition_dir / "baseline",
        seed + CONDITION_SEED_OFFSET["baseline"],
    )
    if set(config.conditions) <= {"baseline", "geometric"}:
        if "geometric" in config.conditions:
            _train_condition(
                "geometric",
                geometric_train,
                val_loader,
                test_loader,
                config,
                device,
                condition_dir / "geometric",
                seed + CONDITION_SEED_OFFSET["geometric"],
            )
        return

    gen_dir = fold_dir / "generative"
    gen_loader = _loader(generative_train, config, shuffle=True, generative=True)
    vae = MaskConditionedVAE(
        config.data.image_size, config.data.num_classes, config.model.latent_dim, config.model.base_channels
    )
    train_vae(
        vae,
        gen_loader,
        device,
        gen_dir / "vae",
        config.train.epochs_vae,
        config.train.lr_g,
        config.model.vae_beta,
        config.train.amp,
    )
    gan_generator, gan_discriminator = (
        ConditionalRefiner(config.data.num_classes, config.model.base_channels),
        PatchDiscriminator(config.data.num_classes, config.model.base_channels),
    )
    train_refiner(
        gan_generator,
        gan_discriminator,
        gen_loader,
        device,
        gen_dir / "gan",
        config.train.epochs_gan,
        config.train.lr_g,
        config.train.lr_d,
        config.model.gan_l1_weight,
        None,
        config.train.amp,
    )
    hybrid_generator, hybrid_discriminator = (
        ConditionalRefiner(config.data.num_classes, config.model.base_channels),
        PatchDiscriminator(config.data.num_classes, config.model.base_channels),
    )
    train_refiner(
        hybrid_generator,
        hybrid_discriminator,
        gen_loader,
        device,
        gen_dir / "hybrid",
        config.train.epochs_gan,
        config.train.lr_g,
        config.train.lr_d,
        config.model.gan_l1_weight,
        vae,
        config.train.amp,
    )

    candidate_count = config.qc.target_synthetic * config.qc.candidate_multiplier
    generation_source = ACDCSliceDataset(
        manifest, split["train"], augment=False, min_foreground_pixels=max(1, config.data.min_foreground_pixels)
    )
    source_loader = _loader(generation_source, config, shuffle=True, generative=True)
    paths = {
        "vae": generate_candidates(
            source_loader,
            gen_dir / "vae_candidates.npz",
            device,
            candidate_count,
            "vae",
            vae=vae,
            latent_std=config.model.latent_noise_std,
        ),
        "gan": generate_candidates(
            source_loader, gen_dir / "gan_candidates.npz", device, candidate_count, "gan", refiner=gan_generator
        ),
        "hybrid_no_qc": generate_candidates(
            source_loader,
            gen_dir / "hybrid_candidates.npz",
            device,
            candidate_count,
            "hybrid",
            vae=vae,
            refiner=hybrid_generator,
            latent_std=config.model.latent_noise_std,
        ),
    }
    qc_path = gen_dir / "hybrid_qc.npz"
    quality_filter(
        paths["hybrid_no_qc"],
        _loader(generation_source, config),
        baseline_model,
        device,
        qc_path,
        config.qc.ssim_min,
        config.qc.ssim_max,
        config.qc.anatomy_dice_min,
        config.qc.fid_max,
        config.qc.target_synthetic,
        config.qc.min_retained,
    )
    paths["hybrid_qc"] = qc_path
    lineage = {}
    for name in ("vae", "gan", "hybrid_no_qc"):
        # Ablations use equal synthetic counts, selected deterministically without QC.
        data = SyntheticPairDataset(paths[name])
        data.images, data.masks, data.sources = (
            array[: config.qc.target_synthetic] for array in (data.images, data.masks, data.sources)
        )
        source_ids = {str(value) for value in data.sources.tolist()}
        if not source_ids <= set(split["train"]):
            raise RuntimeError(f"Synthetic leakage in {name}: sources outside the training fold")
        lineage[name] = {"count": len(data), "source_patients": sorted(source_ids)}
        paths[name] = data
    paths["hybrid_qc"] = SyntheticPairDataset(qc_path)
    qc_source_ids = {str(value) for value in paths["hybrid_qc"].sources.tolist()}
    if not qc_source_ids <= set(split["train"]):
        raise RuntimeError("Synthetic leakage in hybrid_qc: sources outside the training fold")
    lineage["hybrid_qc"] = {"count": len(paths["hybrid_qc"]), "source_patients": sorted(qc_source_ids)}
    write_json(lineage, gen_dir / "lineage.json")

    if "geometric" in config.conditions:
        _train_condition(
            "geometric",
            geometric_train,
            val_loader,
            test_loader,
            config,
            device,
            condition_dir / "geometric",
            seed + CONDITION_SEED_OFFSET["geometric"],
        )
    for condition in ("vae", "gan", "hybrid_no_qc", "hybrid_qc"):
        if condition in config.conditions:
            augmented = ConcatDataset([geometric_train, paths[condition]])
            _train_condition(
                condition,
                augmented,
                val_loader,
                test_loader,
                config,
                device,
                condition_dir / condition,
                seed + CONDITION_SEED_OFFSET[condition],
            )


def run_experiment(config: ExperimentConfig, seeds=None, folds=None):
    from .utils import resolve_device

    device = resolve_device(config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "resolved_config.yaml")
    write_json(provenance(), output / "provenance.json")
    selected_seeds = seeds or config.cv.seeds
    selected_folds = folds if folds is not None else list(range(config.cv.n_splits))
    for seed in selected_seeds:
        for fold in selected_folds:
            run_fold(config, seed, fold, device)
