import pandas as pd

from cardiac_aug.data.acdc import build_patient_folds
from cardiac_aug.evaluation.statistics import paired_comparisons


def test_patient_folds_do_not_overlap(tmp_path):
    manifest = [{"patient_id": f"p{i:03d}", "diagnosis": f"g{i % 2}"} for i in range(20)]
    folds = build_patient_folds(manifest, tmp_path / "folds.json", 5, 42, 0.2, True)
    for split in folds:
        train, val, test = map(set, (split["train"], split["validation"], split["test"]))
        assert not train & val and not train & test and not val & test
    assert sorted(patient for split in folds for patient in split["test"]) == sorted(x["patient_id"] for x in manifest)


def test_statistics_are_paired_by_seed_and_patient():
    rows = []
    for seed in [1, 2]:
        for patient in ["a", "b", "c"]:
            rows += [
                {"seed": seed, "patient_id": patient, "condition": "hybrid_qc", "dice_mean": 0.9},
                {"seed": seed, "patient_id": patient, "condition": "baseline", "dice_mean": 0.8},
            ]
    result = paired_comparisons(pd.DataFrame(rows))
    assert result.loc[0, "n_patients"] == 3
    assert abs(result.loc[0, "mean_difference"] - 0.1) < 1e-8
