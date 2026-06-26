"""Cost-ratio sweep helpers."""

from __future__ import annotations

from src.decision.thresholds import bayes_optimal_threshold


def cost_ratio_sweep(c_ratios: list[int] | tuple[int, ...] = (1, 2, 5, 10, 20)) -> list[dict]:
    """Return Bayes thresholds for c_fp=1 and c_fn in c_ratios."""
    rows: list[dict] = []
    for ratio in c_ratios:
        rows.append(
            {
                "c_ratio": int(ratio),
                "c_fp": 1.0,
                "c_fn": float(ratio),
                "tau": bayes_optimal_threshold(1.0, float(ratio)),
                "label": f"r={ratio}",
            }
        )
    return rows
