"""Burst timing and synchronization statistics for SRK voltage traces."""

from __future__ import annotations

import numpy as np


def _validate_trace(voltage_trace, name: str = "voltage_trace") -> np.ndarray:
    trace = np.asarray(voltage_trace, dtype=np.float64)
    if trace.ndim != 1 or len(trace) < 2 or not np.all(np.isfinite(trace)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return trace


def detect_burst_times(
    voltage_trace,
    dt: float,
    *,
    burn_in: float = 50_000.0,
    threshold: float = -35.0,
    lockout: float = 2_500.0,
) -> np.ndarray:
    """Return upward-threshold burst times in milliseconds.

    Events within ``lockout`` milliseconds of the previous accepted crossing
    are treated as belonging to the same burst.
    """
    trace = _validate_trace(voltage_trace)
    dt = float(dt)
    burn_in = float(burn_in)
    lockout = float(lockout)
    threshold = float(threshold)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(burn_in) or burn_in < 0.0:
        raise ValueError("burn_in must be finite and non-negative")
    if not np.isfinite(lockout) or lockout <= 0.0:
        raise ValueError("lockout must be finite and positive")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    burn_in_steps = int(burn_in / dt)
    if burn_in_steps >= len(trace) - 1:
        raise ValueError("burn_in must leave at least two samples to analyse")

    analysed = trace[burn_in_steps:]
    crossings = np.flatnonzero(
        (analysed[1:] > threshold) & (analysed[:-1] <= threshold)
    ) + 1
    if len(crossings) == 0:
        return np.empty(0, dtype=np.float64)

    lockout_steps = max(1, int(lockout / dt))
    accepted = np.empty(len(crossings), dtype=np.int64)
    accepted[0] = crossings[0]
    count = 1
    for crossing in crossings[1:]:
        if crossing - accepted[count - 1] > lockout_steps:
            accepted[count] = crossing
            count += 1

    return (accepted[:count] + burn_in_steps) * dt


def analyze_rhythm(
    voltage_trace,
    dt: float,
    *,
    burn_in: float = 50_000.0,
    threshold: float = -35.0,
    lockout: float = 2_500.0,
) -> float:
    """Return the coefficient of variation of inter-burst intervals."""
    burst_times = detect_burst_times(
        voltage_trace,
        dt,
        burn_in=burn_in,
        threshold=threshold,
        lockout=lockout,
    )
    if len(burst_times) < 4:
        return float("nan")
    intervals = np.diff(burst_times)
    mean_interval = float(np.mean(intervals))
    return float(np.std(intervals) / mean_interval) if mean_interval > 0.0 else float("nan")


def synchronization_index(
    voltage_1,
    voltage_2,
    dt: float,
    *,
    burn_in: float = 50_000.0,
    threshold: float = -35.0,
) -> float:
    """Return zero-lag correlation of binary above-threshold occupancy."""
    trace_1 = _validate_trace(voltage_1, "voltage_1")
    trace_2 = _validate_trace(voltage_2, "voltage_2")
    if trace_1.shape != trace_2.shape:
        raise ValueError("voltage traces must have the same shape")
    dt = float(dt)
    burn_in = float(burn_in)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(burn_in) or burn_in < 0.0:
        raise ValueError("burn_in must be finite and non-negative")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    burn_in_steps = int(burn_in / dt)
    if burn_in_steps >= len(trace_1):
        raise ValueError("burn_in must leave samples to analyse")
    occupancy_1 = (trace_1[burn_in_steps:] > threshold).astype(np.float64)
    occupancy_2 = (trace_2[burn_in_steps:] > threshold).astype(np.float64)
    centred_1 = occupancy_1 - np.mean(occupancy_1)
    centred_2 = occupancy_2 - np.mean(occupancy_2)
    denominator = np.sqrt(np.sum(centred_1**2) * np.sum(centred_2**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(centred_1 * centred_2) / denominator)
