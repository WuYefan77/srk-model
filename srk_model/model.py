"""Three-dimensional Sherman--Rinzel--Keizer model and SDE solvers."""

from __future__ import annotations

import numpy as np
from numba import njit


C_M = 20.0
G_CA = 3.6
V_CA = 25.0
G_K = 10.0
V_K = -75.0
TAU_N = 20.0
TAU_S = 20_000.0
DEFAULT_INITIAL_STATE = np.array([-60.0, 0.1, 0.3], dtype=np.float64)


@njit(cache=True)
def _logistic(argument: float) -> float:
    argument = max(min(argument, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(argument))


@njit(cache=True)
def _steady_state_gates(voltage: float) -> tuple[float, float, float]:
    m_inf = _logistic((-20.0 - voltage) / 12.0)
    n_inf = _logistic((-16.0 - voltage) / 5.6)
    s_inf = _logistic((-45.0 - voltage) / 10.0)
    return m_inf, n_inf, s_inf


@njit(cache=True)
def _simulate_single_kernel(
    initial_state: np.ndarray,
    noise: np.ndarray,
    dt: float,
    sigma: float,
    g_s: float,
) -> np.ndarray:
    voltage, n_gate, s_gate = initial_state
    history = np.empty(len(noise), dtype=np.float64)
    decay_n = np.exp(-dt / TAU_N)
    drift_scale_s = dt / TAU_S

    for step in range(len(noise)):
        m_inf, n_inf, s_inf = _steady_state_gates(voltage)
        ionic_current = (
            G_CA * m_inf * (voltage - V_CA)
            + G_K * n_gate * (voltage - V_K)
            + g_s * s_gate * (voltage - V_K)
        )

        voltage -= ionic_current * dt / C_M
        n_gate = n_inf + (n_gate - n_inf) * decay_n

        s_effective = min(max(s_gate, 0.0), 1.0)
        diffusion = sigma * np.sqrt(s_effective * (1.0 - s_effective))
        s_gate = (
            s_gate + drift_scale_s * s_inf + diffusion * noise[step]
        ) / (1.0 + drift_scale_s)
        history[step] = voltage

    return history


@njit(cache=True)
def _simulate_coupled_kernel(
    initial_states: np.ndarray,
    noise: np.ndarray,
    dt: float,
    sigma: float,
    g_s_1: float,
    g_s_2: float,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    voltage_1, n_1, s_1 = initial_states[0]
    voltage_2, n_2, s_2 = initial_states[1]
    n_steps = noise.shape[1]
    voltage_history_1 = np.empty(n_steps, dtype=np.float64)
    voltage_history_2 = np.empty(n_steps, dtype=np.float64)
    decay_n = np.exp(-dt / TAU_N)
    drift_scale_s = dt / TAU_S

    for step in range(n_steps):
        m_inf_1, n_inf_1, s_inf_1 = _steady_state_gates(voltage_1)
        m_inf_2, n_inf_2, s_inf_2 = _steady_state_gates(voltage_2)
        gap_current = coupling * (voltage_1 - voltage_2)

        ionic_current_1 = (
            G_CA * m_inf_1 * (voltage_1 - V_CA)
            + G_K * n_1 * (voltage_1 - V_K)
            + g_s_1 * s_1 * (voltage_1 - V_K)
            + gap_current
        )
        ionic_current_2 = (
            G_CA * m_inf_2 * (voltage_2 - V_CA)
            + G_K * n_2 * (voltage_2 - V_K)
            + g_s_2 * s_2 * (voltage_2 - V_K)
            - gap_current
        )

        voltage_1 -= ionic_current_1 * dt / C_M
        voltage_2 -= ionic_current_2 * dt / C_M
        n_1 = n_inf_1 + (n_1 - n_inf_1) * decay_n
        n_2 = n_inf_2 + (n_2 - n_inf_2) * decay_n

        s_effective_1 = min(max(s_1, 0.0), 1.0)
        s_effective_2 = min(max(s_2, 0.0), 1.0)
        diffusion_1 = sigma * np.sqrt(s_effective_1 * (1.0 - s_effective_1))
        diffusion_2 = sigma * np.sqrt(s_effective_2 * (1.0 - s_effective_2))
        s_1 = (
            s_1 + drift_scale_s * s_inf_1 + diffusion_1 * noise[0, step]
        ) / (1.0 + drift_scale_s)
        s_2 = (
            s_2 + drift_scale_s * s_inf_2 + diffusion_2 * noise[1, step]
        ) / (1.0 + drift_scale_s)

        voltage_history_1[step] = voltage_1
        voltage_history_2[step] = voltage_2

    return voltage_history_1, voltage_history_2


def _validate_scalar(name: str, value: float, *, non_negative: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (non_negative and value < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_initial_state(initial_state) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64)
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain three finite values")
    return state


def simulate_srk(
    noise,
    dt: float,
    sigma: float,
    *,
    g_s: float = 4.0,
    initial_state=DEFAULT_INITIAL_STATE,
) -> np.ndarray:
    """Simulate one stochastic SRK oscillator.

    ``noise`` contains pre-generated Wiener increments with variance ``dt``.
    The diffusion coefficient is evaluated at a truncated slow-gate value;
    the scheme controls, but does not mathematically exclude, stepwise boundary
    excursions.
    """
    increments = np.asarray(noise, dtype=np.float64)
    if increments.ndim != 1 or len(increments) < 1 or not np.all(np.isfinite(increments)):
        raise ValueError("noise must be a finite one-dimensional array")
    dt = _validate_scalar("dt", dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    sigma = _validate_scalar("sigma", sigma, non_negative=True)
    g_s = _validate_scalar("g_s", g_s, non_negative=True)
    state = _validate_initial_state(initial_state)
    return _simulate_single_kernel(state, increments, dt, sigma, g_s)


def simulate_coupled_srk(
    noise,
    dt: float,
    sigma: float,
    *,
    g_s_1: float = 4.2,
    g_s_2: float = 4.2,
    coupling: float = 0.1,
    initial_states=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two gap-junction-coupled SRK oscillators.

    ``noise`` must have shape ``(2, n_steps)``. Supplying the increments from
    outside the solver makes independent-noise and common-random-number
    comparisons explicit.
    """
    increments = np.asarray(noise, dtype=np.float64)
    if (
        increments.ndim != 2
        or increments.shape[0] != 2
        or increments.shape[1] < 1
        or not np.all(np.isfinite(increments))
    ):
        raise ValueError("noise must be a finite array with shape (2, n_steps)")

    dt = _validate_scalar("dt", dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    sigma = _validate_scalar("sigma", sigma, non_negative=True)
    g_s_1 = _validate_scalar("g_s_1", g_s_1, non_negative=True)
    g_s_2 = _validate_scalar("g_s_2", g_s_2, non_negative=True)
    coupling = _validate_scalar("coupling", coupling, non_negative=True)

    if initial_states is None:
        states = np.vstack((DEFAULT_INITIAL_STATE, DEFAULT_INITIAL_STATE))
    else:
        states = np.asarray(initial_states, dtype=np.float64)
    if states.shape != (2, 3) or not np.all(np.isfinite(states)):
        raise ValueError("initial_states must have shape (2, 3) and be finite")

    return _simulate_coupled_kernel(
        states,
        increments,
        dt,
        sigma,
        g_s_1,
        g_s_2,
        coupling,
    )
