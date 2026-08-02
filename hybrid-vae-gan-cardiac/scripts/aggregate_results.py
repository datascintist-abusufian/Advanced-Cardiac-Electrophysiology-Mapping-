#!/usr/bin/env python
import argparse
from pathlib import Path

from cardiac_aug.evaluation.cost_benefit import cost_benefit
from cardiac_aug.evaluation.statistics import aggregate_runs
from cardiac_aug.evaluation.tables import write_latex_tables


parser = argparse.ArgumentParser(description="Aggregate patient-level metrics and paired statistics.")
parser.add_argument("--run-dir", default="runs")
parser.add_argument("--output-dir", default="runs/summary")
parser.add_argument(
    "--allow-incomplete", action="store_true", help="Debugging only; never use partial manuscript tables."
)
args = parser.parse_args()
aggregate_runs(args.run_dir, args.output_dir, require_complete=not args.allow_incomplete)
cost_benefit(
    args.run_dir, Path(args.output_dir) / "patient_level_metrics.csv", Path(args.output_dir) / "cost_benefit.csv"
)
write_latex_tables(
    Path(args.output_dir) / "patient_level_metrics.csv",
    Path(args.output_dir) / "paired_statistics.csv",
    Path(args.output_dir) / "cost_benefit.csv",
    args.output_dir,
)
print(f"Wrote validated tables to {args.output_dir}")
