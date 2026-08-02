# %% [markdown]
# # Artifact-backed results analysis
# Run this after `scripts/aggregate_results.py`. It reads generated patient-level artifacts only.

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUNS = Path("../runs/summary")
metrics = pd.read_csv(RUNS / "patient_level_metrics.csv")
statistics = pd.read_csv(RUNS / "paired_statistics.csv")
cost = pd.read_csv(RUNS / "cost_benefit.csv")

# %%
display(metrics.groupby("condition")[["dice_mean", "iou_mean", "hd95_mean", "assd_mean"]].agg(["mean", "std"]))

# %%
display(statistics[statistics.metric == "dice_mean"])

# %%
ax = metrics.boxplot(column="dice_mean", by="condition", rot=30, figsize=(10, 5))
ax.set_ylabel("Patient-level mean Dice")
plt.suptitle("")
plt.tight_layout()

# %%
display(cost)
