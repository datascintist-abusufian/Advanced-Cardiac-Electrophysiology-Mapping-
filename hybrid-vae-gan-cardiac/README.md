# Hybrid Generative Augmentation for Cardiac MRI Segmentation

This repository is a reproducible PyTorch implementation of the experiments described in
“Hybrid Generative Augmentation for Data-Efficient Cardiac MRI Segmentation Using VAE-GAN and
Attention U-Net.” It does **not** contain manuscript results as constants. Every table, significance
test, timing, and figure is derived from saved experiment artifacts.

## What is implemented

- Labelled ED/ES ACDC NIfTI loading and 2D short-axis preprocessing
- Patient-level, diagnosis-stratified five-fold cross-validation with a fold-local validation split
- Three repeat seeds (`42`, `1337`, `2026`)
- Baseline and joint geometric/intensity augmentation
- A 128-dimensional convolutional, mask-conditioned VAE
- Gaussian latent perturbation
- Conditional U-Net generator and 70×70-style PatchGAN discriminator
- Anatomically consistent synthesis: the image generator is conditioned on the exact transformed mask
- SSIM filtering, dataset-level domain-feature FID, and fold-local anatomical plausibility filtering
- Residual Attention U-Net for background, LV, myocardium, and RV
- Patient-level Dice, IoU, HD95, and ASSD in physical units
- Baseline, geometric, VAE, GAN, hybrid-without-QC, and hybrid-with-QC ablations
- CSV training logs, checkpoints, split manifests, provenance, and timing artifacts
- Patient-paired Wilcoxon tests, bootstrap confidence intervals, Cohen's dz, and Bonferroni correction
- Performance, qualitative overlay, input-attribution, and attention-gate figures
- Cost-benefit table and explicit manuscript-claim audit
- A tiny synthetic ACDC-shaped fixture for software testing; it is never valid scientific evidence

## Important methodological interpretation

The manuscript's original phrase “apply identical spatial transformations to the ground-truth mask
corresponding to a perturbed latent vector” is not sufficient to guarantee alignment. This code uses a
defensible alternative:

1. A real image and its mask receive the same geometric transformation.
2. The VAE encodes the transformed image together with the one-hot mask.
3. Only appearance is perturbed in the latent representation.
4. The decoder synthesises an image conditioned on the unchanged mask.
5. The adversarial refiner receives both the coarse image and the same mask.

Thus, the synthetic pair is `(generated image, conditioning mask)`. This preserves the requested
anatomy but does not claim that arbitrary latent movement creates a known spatial deformation.

## Repository map

```text
cardiac_aug/
  data/                 ACDC discovery, preprocessing, folds, paired transforms
  models/               VAE, conditional GAN/PatchGAN, Attention U-Net
  training/             segmentation and generative training loops/logs
  evaluation/           metrics, QC, statistics, cost, claim audit, explanations
  visualization/        manuscript-ready figures
configs/                 full, smoke-test, and claim-audit configurations
scripts/                 command-line entry points
tests/                   unit and architecture tests
notebooks/               artifact-backed exploratory analysis
```

## Installation

Python 3.10 or 3.11 and a CUDA-capable GPU are recommended.

```bash
conda env create -f environment.yml
conda activate hybrid-card-mri
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## ACDC data layout

Download the labelled ACDC training dataset under its terms, then use this layout:

```text
data/raw/ACDC/database/training/
  patient001/
    Info.cfg
    patient001_frame01.nii.gz
    patient001_frame01_gt.nii.gz
    ...
```

The loader reads the ED and ES frame identifiers from each `Info.cfg`. Original ACDC labels are
remapped from `1=RV, 2=Myo, 3=LV` to the internal order `1=LV, 2=Myo, 3=RV`. Images are clipped to
their nonzero 1st–99th percentiles, scaled to `[0,1]`, resized in-plane, and cached per patient. The
rescaled in-plane spacing and original through-plane spacing are retained for distance metrics.

Prepare the cache:

```bash
python scripts/prepare_acdc.py --config configs/default.yaml
```

## Run the study

The default configuration runs six conditions × five folds × three seeds and trains generative models
inside every fold. This is intentionally computationally expensive. To run everything:

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

To resume methodically, run one seed/fold at a time (existing artifacts are currently overwritten for
that seed/fold):

```bash
python scripts/run_experiment.py --config configs/default.yaml --seed 42 --fold 0
```

All VAE, GAN, segmentation, and QC fitting is restricted to the training patients of that fold.
Validation patients are used only for segmentation early stopping. Test patients are used only once
for final metrics. Split JSON files provide an auditable leakage boundary.

## Aggregate, analyse, and make figures

```bash
python scripts/aggregate_results.py --run-dir runs --output-dir runs/summary
python scripts/make_figures.py --config configs/default.yaml
python scripts/audit_manuscript_claims.py
```

Key outputs are:

```text
runs/summary/patient_level_metrics.csv
runs/summary/performance_summary.csv
runs/summary/paired_statistics.csv
runs/summary/cost_benefit.csv
runs/summary/manuscript_claim_audit.csv
runs/summary/completeness.json
runs/summary/segmentation_results.tex
runs/summary/statistical_analysis.tex
runs/summary/cost_benefit.tex
runs/summary/figures/performance.png
runs/summary/figures/qualitative_and_xai.png
```

The statistical unit is the patient. ED and ES are evaluated separately and averaged into one patient
row. Repeat-seed values are then averaged per patient before inferential tests, preventing seeds from
being incorrectly treated as independent patients. If one of prediction/target is empty, HD95 and ASSD
receive a documented field-of-view diagonal penalty instead of silently dropping the failure.

Aggregation is strict by default: if any configured seed/fold/condition artifact is missing, it writes
`completeness.json` and stops before producing manuscript tables. `--allow-incomplete` exists only for
pipeline debugging.

## Quality control

QC has two sample-level filters and one dataset-level filter:

- **SSIM:** accepts a configurable fidelity/diversity range relative to the source image.
- **Anatomical plausibility:** a fold-local baseline segmenter predicts each generated image; mean
  Dice against the conditioning mask must exceed the configured threshold.
- **FID:** Fréchet distance is calculated between *sets* of real and synthetic images using features
  from the same training-only cardiac segmentation encoder. PCA is fitted on real training features
  only for stable covariance estimation. It is not misrepresented as a per-image score.

The QC JSON records candidate count, retained count, both set-level FID values, thresholds, feature
source, and whether the final set actually passed. If `fid_passed` is false, the experiment is evidence
that the threshold was not achieved; the manuscript must not claim otherwise.

## Ablation definitions

| Condition | Real transforms | Synthetic source | QC |
|---|---:|---|---:|
| baseline | no | none | no |
| geometric | yes | none | no |
| VAE | yes | mask-conditioned VAE | no |
| GAN | yes | conditional refiner from corrupted real MRI | no |
| hybrid_no_qc | yes | latent-perturbed VAE → conditional refiner | no |
| hybrid_qc | yes | latent-perturbed VAE → conditional refiner | yes |

The VAE, GAN-only refiner, and hybrid refiner have distinct checkpoints. Synthetic sources are always
training patients. Ablations use the same configured synthetic count where QC has retained enough
samples; the exact count is recorded because a strict QC threshold may retain fewer.

## Smoke test and unit tests

```bash
pytest
python scripts/create_synthetic_acdc.py
python scripts/prepare_acdc.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/smoke.yaml --seed 42 --fold 0
```

Synthetic fixture output proves only that the software runs. It must not be included in the manuscript.

## Manuscript claims that remain evidence-dependent

Until a complete real-data run produces matching artifacts, the following supplied claims are
**unverified**, including: Dice values `0.867/0.881/0.896/0.902`, HD95 values `5.37/4.05 mm`, exact
confidence intervals and p-values, retained-sample range `200–250`, GPU-hour figures, the claimed
`0.915` full-data result, and any statement of statistical significance or clinical deployment.

`configs/manuscript_claims.yaml` intentionally stores a subset of reported values only for comparison.
The audit labels each as `SUPPORTED_WITHIN_TOLERANCE`, `NOT_REPRODUCED`, or `NO_EVIDENCE`. Do not edit
tolerances to force a match. Add claims only when their metric has a direct generated counterpart.

## Reproducibility notes

- The resolved configuration and software/hardware provenance are written at run start.
- Python, NumPy, and PyTorch RNGs are seeded; deterministic algorithms are requested with warnings.
- Patient folds are saved and asserted disjoint.
- No ImageNet/Inception FID is used because natural-image features are poorly matched to cardiac MRI.
- Training logs are append-only per path; remove or archive a partial run directory before restarting
  the same seed/fold if a clean log is required.
- ACDC is a single-centre benchmark. External validation and clinician assessment are not implemented
  because no external dataset or reader study was supplied. The manuscript must state this limitation.

## Citation and licence

Before publication, cite the ACDC challenge paper and every architecture used. Add a project licence
only after all contributors agree on it; none is asserted here.
