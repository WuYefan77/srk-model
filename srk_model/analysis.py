"""Small analysis utilities shared by the SRK experiment scripts."""

from __future__ import annotations

import numpy as np


def aggregate_wiener_increments(fine_increments, factor: int) -> np.ndarray:
    """Aggregate fine Wiener increments into a coupled coarser path."""
    increments = np.asarray(fine_increments, dtype=np.float64)
    if increments.ndim != 1 or len(increments) < 1 or not np.all(np.isfinite(increments)):
        raise ValueError("fine_increments must be a finite one-dimensional array")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if len(increments) % factor != 0:
        raise ValueError("the increment count must be divisible by factor")
    return increments.reshape(-1, factor).sum(axis=1)


def compute_log_centroid(sigmas, values, delta: float) -> tuple[float, float, float]:
    """Summarise the near-minimum region of a positive curve in log space."""
    sigma_values = np.asarray(sigmas, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    delta = float(delta)
    if sigma_values.ndim != 1 or curve.ndim != 1 or sigma_values.shape != curve.shape:
        raise ValueError("sigmas and values must be one-dimensional arrays of equal shape")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")

    valid = (
        np.isfinite(sigma_values)
        & np.isfinite(curve)
        & (sigma_values > 0.0)
        & (curve > 0.0)
    )
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")

    log_sigmas = np.log(sigma_values[valid])
    log_values = np.log(curve[valid])
    near_minimum = log_values <= np.min(log_values) + delta
    selected = log_sigmas[near_minimum]
    return (
        float(np.exp(np.mean(selected))),
        float(np.exp(np.min(selected))),
        float(np.exp(np.max(selected))),
    )
