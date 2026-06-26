# Release Checklist

Before making this repository public:

- [ ] Confirm no raw BreakHis images are tracked.
- [ ] Confirm `data/raw/` is ignored and empty in the public clone.
- [ ] Confirm checkpoints and model weights are not tracked.
- [ ] Confirm local logs, audit backups, repair staging directories, caches, `.DS_Store`, and virtual environments are not tracked.
- [ ] Confirm no private paths, credentials, API keys, tokens, reviewer files, or collaborator communications are tracked.
- [ ] Run `uv sync`.
- [ ] Run `uv run pytest`.
- [ ] Regenerate figures and tables with `scripts/07_make_figures.py`.
- [ ] Verify Table 1 includes corrected architecture-level patient-cluster bootstrap intervals for `Net benefit@r10` and `Net benefit@r5`.
- [ ] Verify README setup and reproduction instructions on a clean clone.
- [ ] Human authors approve the MIT license before public release.
- [ ] Add final contact email to `README.md`.
- [ ] Add final public GitHub URL to the manuscript.
- [ ] Create archival release and update DOI in `CITATION.cff`.
- [ ] Update manuscript repository URL and DOI placeholders.
