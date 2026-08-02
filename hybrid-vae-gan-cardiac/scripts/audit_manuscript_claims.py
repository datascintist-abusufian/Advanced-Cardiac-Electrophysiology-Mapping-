#!/usr/bin/env python
import argparse
from cardiac_aug.evaluation.claims import audit_claims


parser = argparse.ArgumentParser(description="Compare reported manuscript values with generated artifacts.")
parser.add_argument("--claims", default="configs/manuscript_claims.yaml")
parser.add_argument("--metrics", default="runs/summary/patient_level_metrics.csv")
parser.add_argument("--output", default="runs/summary/manuscript_claim_audit.csv")
parser.add_argument("--completeness", default="runs/summary/completeness.json")
args = parser.parse_args()
result = audit_claims(args.claims, args.metrics, args.output, args.completeness)
print(result[["claim", "status", "reported", "observed"]].to_string(index=False))
