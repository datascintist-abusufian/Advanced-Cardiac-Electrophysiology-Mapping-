#!/usr/bin/env python
import argparse

from cardiac_aug.config import load_config
from cardiac_aug.experiment import run_experiment


parser = argparse.ArgumentParser(description="Run patient-level cross-validation experiments.")
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--seed", type=int, action="append", help="Run only this seed; repeat flag for several.")
parser.add_argument("--fold", type=int, action="append", help="Run only this fold; repeat flag for several.")
args = parser.parse_args()
run_experiment(load_config(args.config), args.seed, args.fold)
