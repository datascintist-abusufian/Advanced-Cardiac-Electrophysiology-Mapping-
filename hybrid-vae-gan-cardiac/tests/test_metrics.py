import numpy as np

from cardiac_aug.evaluation.metrics import case_metrics


def test_identical_masks_are_perfect():
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[7:10, 7:10] = 2
    mask[10:14, 3:7] = 3
    metrics = case_metrics(mask, mask, spacing=(1.2, 1.2, 5.0))
    assert metrics["dice_mean"] == 1.0
    assert metrics["iou_mean"] == 1.0
    assert metrics["hd95_mean"] == 0.0
    assert metrics["assd_mean"] == 0.0


def test_empty_prediction_receives_worst_case_surface_penalty():
    true = np.zeros((8, 8, 2), dtype=np.uint8)
    true[2:4, 2:4] = 1
    metrics = case_metrics(np.zeros_like(true), true)
    assert metrics["dice_LV"] == 0.0
    assert metrics["hd95_LV"] > 0
