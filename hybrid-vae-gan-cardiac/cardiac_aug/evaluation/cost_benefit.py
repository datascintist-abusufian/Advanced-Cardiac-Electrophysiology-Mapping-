from __future__ import annotations

from pathlib import Path

import pandas as pd


DEPENDENCIES = {
    "baseline": [],
    "geometric": [],
    "vae": ["vae"],
    "gan": ["gan"],
    "hybrid_no_qc": ["vae", "hybrid"],
    "hybrid_qc": ["vae", "hybrid"],
}


def _hours(csv_path: Path) -> float:
    return pd.read_csv(csv_path)["seconds"].sum() / 3600 if csv_path.exists() else 0.0


def cost_benefit(run_dir, patient_metrics_csv, output_csv):
    metrics = pd.read_csv(patient_metrics_csv)
    rows = []
    for condition in sorted(metrics.condition.unique()):
        runtime_files = list(Path(run_dir).glob(f"seed_*/fold_*/conditions/{condition}/runtime.json"))
        segmentation_hours = sum(pd.read_json(path, typ="series").get("wall_clock_hours", 0) for path in runtime_files)
        generator_hours = 0.0
        for seed_fold in Path(run_dir).glob("seed_*/fold_*"):
            generator_hours += sum(
                _hours(seed_fold / "generative" / component / "training.csv")
                for component in DEPENDENCIES.get(condition, [])
            )
        subset = metrics[metrics.condition == condition]
        training_samples = [pd.read_json(path, typ="series").get("training_samples") for path in runtime_files]
        rows.append(
            {
                "condition": condition,
                "mean_training_samples_per_fold": pd.Series(training_samples).mean(),
                "mean_dice": subset.dice_mean.mean(),
                "mean_hd95_mm": subset.hd95_mean.mean(),
                "recorded_training_hours": segmentation_hours + generator_hours,
                "note": "Wall-clock training time; generation/QC overhead excluded",
            }
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
