from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def audit_claims(claims_path, patient_metrics_path, output_path, completeness_path=None):
    claims = yaml.safe_load(Path(claims_path).read_text(encoding="utf-8"))["claims"]
    metrics = pd.read_csv(patient_metrics_path)
    complete = True
    if completeness_path and Path(completeness_path).exists():
        import json

        complete = json.loads(Path(completeness_path).read_text(encoding="utf-8")).get("complete", False)
    rows = []
    for claim in claims:
        subset = metrics[metrics.condition == claim["condition"]]
        observed = subset[claim["metric"]].mean() if claim["metric"] in subset and len(subset) else None
        expected, tolerance = claim["reported"], claim.get("tolerance", 0.0005)
        if not complete:
            status = "INCOMPLETE_RUN"
        elif observed is None:
            status = "NO_EVIDENCE"
        elif abs(observed - expected) <= tolerance:
            status = "SUPPORTED_WITHIN_TOLERANCE"
        else:
            status = "NOT_REPRODUCED"
        rows.append(
            {
                **claim,
                "observed": observed,
                "difference": None if observed is None else observed - expected,
                "status": status,
            }
        )
    result = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
