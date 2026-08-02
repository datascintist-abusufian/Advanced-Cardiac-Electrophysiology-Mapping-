# Output and evidence schema

Every result is rooted at `runs/seed_<seed>/fold_<fold>/`.

- `split.json`: exact patient membership for training, validation, and test.
- `generative/*/training.csv`: one row per VAE/GAN epoch.
- `generative/*.npz`: generated images, masks, original images, and source patient IDs.
- `generative/lineage.json`: asserted training-only source-patient lineage for each synthetic condition.
- `generative/hybrid_qc.qc.json`: all QC counts, thresholds, FID values, and pass status.
- `conditions/<name>/training.csv`: segmentation learning curve.
- `conditions/<name>/best.pt`: validation-selected weights.
- `conditions/<name>/patient_metrics.csv`: one independent row per test patient.
- `conditions/<name>/runtime.json`: sample count, early stopping, and wall time.

Aggregated files never replace the fold artifacts. A result is manuscript-eligible only if its source
file exists, the associated split is disjoint, all expected patients/seeds/folds are present, and the
claim audit supports it within a predeclared numerical tolerance.
