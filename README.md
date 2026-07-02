# Beyond Headline Accuracy: BreakHis Calibration and Decision Analysis

This repository contains the reproducibility package for **“Beyond Headline Accuracy: A Patient-Level Calibration and Decision-Analysis Audit of CNN Classifiers on BreakHis.”** The project is a methodological audit of CNN classifiers on BreakHis under patient-level evaluation: it measures calibration, recalibration behavior, Brier decomposition, patient-cluster uncertainty, and cost-sensitive decision utility. It is **not** intended to introduce a new classifier architecture or to redistribute the BreakHis image dataset.

The public release is designed to include source code, fixed patient-level fold definitions, cached logits, derived result tables, final figure/table artifacts, and tests. Raw BreakHis images, trained checkpoints, private repair logs, local audit backups, and local environment files are intentionally excluded.

## Repository Contents

- `src/`: PyTorch data loading, model definitions, training utilities, calibration, decision-analysis, bootstrap, plotting, and table-building code.
- `scripts/`: numbered pipeline entry points from fold construction through figure/table generation.
- `tests/`: pytest unit and smoke tests for patient-level splitting, calibration metrics, recalibrators, decision analysis, bootstrap behavior, table generation, and CLI safeguards.
- `configs/`: training and analysis grids used by the manuscript.
- `data/folds/folds.json`: fixed patient-level train/calibration/test folds.
- `outputs/logits/`: cached raw logits for the 180 trained runs. These allow analysis reproduction without retraining or redistributing images.
- `outputs/results/`: derived CSV result tables and decision-curve arrays.
- `outputs/figures/`: regenerated manuscript figures and LaTeX/CSV tables.

## Environment Setup

The project targets Python 3.11 and uses `uv` for environment management.

```bash
uv sync
uv run python --version
uv run pytest
```

The environment used for the final local run reported Python 3.11.15, PyTorch 2.11.0, torchvision 0.26.0, uv 0.11.7, and Apple MPS availability. The package constraints are recorded in `pyproject.toml` and the resolved environment is recorded in `uv.lock`.

## Data Setup

Raw BreakHis images are **not included** in this repository. Download BreakHis from the official BreakHis database page:

https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Place the extracted dataset locally as:

```text
data/raw/BreaKHis_v1/
```

The fold builder expects the standard BreakHis directory layout under `BreaKHis_v1`. If `data/raw/BreaKHis_v1/Folds.csv` is present, the fold-generation code validates that the directory-derived patient IDs match the official patient organization.

## Reproduce Patient-Level Folds

```bash
uv run python scripts/01_build_folds.py \
  --data-root data/raw \
  --output data/folds/folds.json
```

The committed fold file uses patient-directory IDs as the grouping key, not image filenames or slide IDs.

## Run Tests

```bash
uv run pytest
```

Some tests create synthetic data and temporary outputs. The full test suite does not require full retraining.

## Regenerate Tables and Figures Without Retraining

The public package includes cached logits and derived result artifacts. To regenerate the final manuscript figures and tables from those artifacts:

```bash
uv run python scripts/07_make_figures.py \
  --calibration-results outputs/results/calibration_results.csv \
  --bootstrap-cis outputs/results/bootstrap_cis.csv \
  --decision-results outputs/results/decision_results.csv \
  --architecture-significance outputs/results/architecture_significance.csv \
  --brier-summary outputs/results/brier_decomposition_summary.csv \
  --logit-root outputs/logits \
  --decision-curve-root outputs/results/decision_curves \
  --figure-root outputs/figures \
  --table-root outputs/figures/tables \
  --seed 2026
```

This command does not touch the CNN checkpoints. It reads cached logits and derived CSV/NPY artifacts.

## Rerun the Full Analysis Pipeline

If cached logits are already available:

```bash
uv run python scripts/04_run_calibration.py
uv run python scripts/04b_calibration_diagnostics.py
uv run python scripts/05_run_decision.py
uv run python scripts/06_bootstrap_cis.py --B 1000 --seed 2026
uv run python scripts/06b_architecture_significance.py
uv run python scripts/07_make_figures.py --seed 2026
```

If raw images are available and you want to rerun training from scratch:

```bash
uv run python scripts/02_train_all.py --config configs/training.yaml
uv run python scripts/03_cache_logits.py
```

Full retraining covers 3 architectures x 4 magnifications x 3 seeds x 5 folds. It is expensive and was run on an Apple M4 MacBook Air with MPS. Use `scripts/resume_report.py` to inspect checkpoint completion before or after interrupted runs.

## Reproducibility Notes

- Training seeds: `0`, `1`, and `2`.
- Bootstrap seed: `2026`.
- Fold generation is deterministic from the configured seed offsets.
- Raw logits are cached before calibration and decision analysis.
- Bitwise-identical results are not guaranteed across different PyTorch, torchvision, BLAS, hardware, or MPS/CPU backends.

## Citation

Please cite the manuscript and the BreakHis dataset paper when using this repository. A `CITATION.cff` file is included for the manuscript; the DOI field will be updated after archival release.

## License

Repository code is released under the MIT License. The BreakHis images are not redistributed here and remain governed by the dataset maintainers' terms.


## Contact

Corresponding author: Gourab Sarker  
Email: gourab.sarker@ttu.edu  
Repository: https://github.com/GourabSarker/breakhis-calibration-audit
