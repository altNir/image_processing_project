"""Deterministic paired statistical summaries for restoration experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_bootstrap(
    before: Sequence[float],
    after: Sequence[float],
    *,
    higher_is_better: bool = True,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, float | int | bool]:
    """Summarize paired improvement with a percentile bootstrap confidence interval.

    The reported delta is always oriented so a positive value means improvement.
    Non-finite pairs are excluded and the same sampled indices are used for both
    conditions, preserving the paired experiment design.
    """

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    first = np.asarray(before, dtype=np.float64)
    second = np.asarray(after, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Paired samples must have identical shapes")
    valid = np.isfinite(first) & np.isfinite(second)
    raw_delta = second[valid] - first[valid]
    delta = raw_delta if higher_is_better else -raw_delta
    if delta.size == 0:
        return {
            "pair_count": 0, "higher_is_better": higher_is_better,
            "mean_improvement": float("nan"), "median_improvement": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "win_rate": float("nan"), "tie_rate": float("nan"),
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(resamples, delta.size))
    bootstrap_means = np.mean(delta[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "pair_count": int(delta.size),
        "higher_is_better": higher_is_better,
        "mean_improvement": float(np.mean(delta)),
        "median_improvement": float(np.median(delta)),
        "ci_low": float(np.quantile(bootstrap_means, alpha)),
        "ci_high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "win_rate": float(np.mean(delta > 0.0)),
        "tie_rate": float(np.mean(delta == 0.0)),
    }
