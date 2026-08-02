from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..utils import write_json


def bootstrap_ci(values: np.ndarray, seed=42, repetitions=10000, confidence=0.95):
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (repetitions, len(values)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return np.quantile(means, [alpha, 1 - alpha]).tolist()


def paired_comparisons(frame: pd.DataFrame, reference="hybrid_qc", metric="dice_mean", alpha=0.05):
    rows, methods = [], sorted(set(frame["condition"]) - {reference})
    # Seeds are repeated measurements, not independent patients. Average each patient's
    # repeated-seed result before the inferential test to avoid pseudoreplication.
    reference_rows = (
        frame[frame.condition == reference]
        .groupby("patient_id", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "reference"})
    )
    for method in methods:
        candidate = (
            frame[frame.condition == method]
            .groupby("patient_id", as_index=False)[metric]
            .mean()
            .rename(columns={metric: "candidate"})
        )
        paired = reference_rows.merge(candidate, on="patient_id", validate="one_to_one").dropna()
        differences = (paired.reference - paired.candidate).to_numpy()
        if len(differences) < 2:
            statistic, p_value = np.nan, np.nan
        elif np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
        ci_low, ci_high = bootstrap_ci(differences) if len(differences) else (np.nan, np.nan)
        effect = (
            np.mean(differences) / np.std(differences, ddof=1)
            if len(differences) > 1 and np.std(differences, ddof=1)
            else np.nan
        )
        rows.append(
            {
                "comparison": f"{reference} vs {method}",
                "metric": metric,
                "n_patients": len(paired),
                "mean_difference": np.mean(differences) if len(differences) else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "cohen_dz": effect,
            }
        )
    result = pd.DataFrame(rows)
    result["p_bonferroni"] = np.minimum(result.p_raw * max(len(result), 1), 1.0)
    result["significant"] = result.p_bonferroni < alpha
    return result


def validate_completeness(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    import yaml

    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    expected = {
        (seed, fold, condition)
        for seed in config.get("cv", {}).get("seeds", [])
        for fold in range(config.get("cv", {}).get("n_splits", 0))
        for condition in config.get("conditions", [])
    }
    observed = set()
    for path in run_dir.glob("seed_*/fold_*/conditions/*/patient_metrics.csv"):
        seed = int(next(part for part in path.parts if part.startswith("seed_")).split("_")[1])
        fold = int(next(part for part in path.parts if part.startswith("fold_")).split("_")[1])
        observed.add((seed, fold, path.parent.name))
    missing = sorted(expected - observed)
    return {
        "complete": bool(expected) and not missing,
        "expected_artifact_count": len(expected),
        "observed_artifact_count": len(observed),
        "missing": [{"seed": a, "fold": b, "condition": c} for a, b, c in missing],
    }


def aggregate_runs(run_dir: str | Path, output_dir: str | Path, require_complete: bool = True):
    completeness = validate_completeness(run_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(completeness, output / "completeness.json")
    if require_complete and not completeness["complete"]:
        raise RuntimeError(
            f"Run is incomplete: {len(completeness['missing'])} expected artifacts are missing. "
            "Use --allow-incomplete only for debugging, never manuscript tables."
        )
    files = list(Path(run_dir).glob("seed_*/fold_*/conditions/*/patient_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No patient_metrics.csv artifacts found under {run_dir}")
    frames = []
    for path in files:
        parts = path.parts
        seed = int(next(value for value in parts if value.startswith("seed_")).split("_")[1])
        fold = int(next(value for value in parts if value.startswith("fold_")).split("_")[1])
        condition = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, "condition", condition)
        df.insert(0, "fold", fold)
        df.insert(0, "seed", seed)
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(output / "patient_level_metrics.csv", index=False)
    metric_columns = [column for column in all_rows if column.startswith(("dice_", "iou_", "hd95_", "assd_"))]
    summary = all_rows.groupby("condition")[metric_columns].agg(["mean", "std", "count"])
    summary.to_csv(output / "performance_summary.csv")
    stats_rows = pd.concat(
        [paired_comparisons(all_rows, metric=metric) for metric in ["dice_mean", "iou_mean", "hd95_mean", "assd_mean"]],
        ignore_index=True,
    )
    stats_rows.to_csv(output / "paired_statistics.csv", index=False)
    return all_rows, summary, stats_rows
