from __future__ import annotations

from pathlib import Path

import pandas as pd


DISPLAY = {
    "baseline": "Baseline",
    "geometric": "Geometric",
    "vae": "VAE",
    "gan": "GAN",
    "hybrid_no_qc": "Hybrid (no QC)",
    "hybrid_qc": "Hybrid (QC)",
}


def write_latex_tables(patient_metrics_csv, statistics_csv, cost_csv, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(patient_metrics_csv)
    columns = [
        "dice_LV",
        "iou_LV",
        "dice_Myo",
        "iou_Myo",
        "dice_RV",
        "iou_RV",
        "dice_mean",
        "iou_mean",
        "hd95_mean",
        "assd_mean",
    ]
    table = metrics.groupby("condition")[columns].agg(["mean", "std"])
    formatted = pd.DataFrame(index=table.index)
    for column in columns:
        formatted[column] = [
            f"{table.loc[index, (column, 'mean')]:.3f} $\\pm$ {table.loc[index, (column, 'std')]:.3f}"
            for index in table.index
        ]
    formatted.index = [DISPLAY.get(index, index) for index in formatted.index]
    formatted.to_latex(
        output / "segmentation_results.tex",
        escape=False,
        caption="Patient-level segmentation performance generated from experiment artifacts.",
        label="tab:generated_segmentation",
    )
    statistics = pd.read_csv(statistics_csv)
    statistics.to_latex(
        output / "statistical_analysis.tex",
        index=False,
        float_format="%.4g",
        caption="Patient-paired comparisons with bootstrap confidence intervals and Bonferroni correction.",
        label="tab:generated_statistics",
    )
    cost = pd.read_csv(cost_csv)
    cost.to_latex(
        output / "cost_benefit.tex",
        index=False,
        float_format="%.3f",
        caption="Artifact-derived performance and recorded training-time cost.",
        label="tab:generated_cost",
    )
