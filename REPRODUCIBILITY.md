# Reproducibility

## Software Environment

This project targets:

- Python: `>=3.11,<3.12`
- uv: tested locally with `uv 0.11.7`
- PyTorch: locked locally to `2.11.0`
- torchvision: locked locally to `0.26.0`

Create the environment with:

```bash
uv sync
```

Run tests with:

```bash
uv run pytest
```

The final local environment reported Python 3.11.15 and Apple MPS availability. CUDA is not required by the codebase.

## Seeds

- Training seeds: `0`, `1`, `2`
- Bootstrap seed: `2026`
- Default script seed: exposed by each CLI as `--seed` where applicable

The code sets seeds for Python `random`, NumPy, PyTorch, and CUDA seed functions. It also sets deterministic cuDNN flags where applicable. Apple MPS and library-version differences can still prevent bitwise-identical results across machines.

Recommended manuscript-safe wording:

> All analyses were run with fixed seeds and deterministic settings where supported; bitwise-identical outputs are not guaranteed across hardware backends or library versions.

## Main Reproduction Commands

Build patient-level folds after placing BreakHis at `data/raw/BreaKHis_v1/`:

```bash
uv run python scripts/01_build_folds.py
```

Train the full grid, if raw images are available:

```bash
uv run python scripts/02_train_all.py --config configs/training.yaml
```

Cache logits from completed checkpoints:

```bash
uv run python scripts/03_cache_logits.py
```

Run downstream analysis from cached logits:

```bash
uv run python scripts/04_run_calibration.py
uv run python scripts/04b_calibration_diagnostics.py
uv run python scripts/05_run_decision.py
uv run python scripts/06_bootstrap_cis.py --B 1000 --seed 2026
uv run python scripts/06b_architecture_significance.py
uv run python scripts/07_make_figures.py --seed 2026
```

Regenerate only figures and publication tables from included derived artifacts:

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

## Included Artifacts

The release includes compact derived artifacts so readers can regenerate tables and figures without retraining:

- `outputs/logits/**/*.npz`
- `outputs/results/*.csv`
- `outputs/results/decision_curves/**/*.npy`
- `outputs/figures/`

Excluded artifacts include raw BreakHis images, checkpoints, audit backups, repair-run staging outputs, private logs, virtual environments, and local cache files.
