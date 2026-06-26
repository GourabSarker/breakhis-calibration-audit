# Data Availability

Raw BreakHis images are not redistributed in this repository.

Users must download BreakHis from the official dataset source:

https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

After downloading and extracting the dataset, place it at:

```text
data/raw/BreaKHis_v1/
```

The repository includes:

- patient-level fold definitions in `data/folds/folds.json`;
- cached raw logits in `outputs/logits/`;
- derived result CSVs in `outputs/results/`;
- decision-curve arrays in `outputs/results/decision_curves/`;
- generated manuscript figures and tables in `outputs/figures/`.

The included cached logits and derived artifacts are sufficient to rerun calibration, decision-analysis summaries, statistical tables, and figure/table generation without redistributing raw images or trained model checkpoints.

The raw dataset license and citation terms are controlled by the BreakHis maintainers. Please follow the official BreakHis terms and cite the original BreakHis dataset publication when using the images.
