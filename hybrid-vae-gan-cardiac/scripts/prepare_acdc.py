#!/usr/bin/env python
import argparse

from cardiac_aug.config import load_config
from cardiac_aug.data.acdc import preprocess_acdc


parser = argparse.ArgumentParser(description="Preprocess labelled ED/ES ACDC NIfTI volumes into patient caches.")
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
config = load_config(args.config)
patients = preprocess_acdc(config.data.raw_dir, config.data.cache_dir, config.data.image_size)
print(f"Prepared {len(patients)} patients in {config.data.cache_dir}")
