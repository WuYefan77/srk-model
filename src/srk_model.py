import numpy as np
from numba import njit


@njit
def run_srk(sigma, g_S, dt, T_total, seed=0):
    np.random.seed(seed)
    steps = int(T_total / dt)
    V = -60.0
    n = 0.1
    s = 0.3
    v_history = np.zeros(steps)
    Cm, g_Ca, V_Ca, g_K, V_K = 20.0, 3.6, 25.0, 10.0, -75.0
    tau_n, tau_s = 20.0, 20000.0
    sqrt_dt = np.sqrt(dt)

    for i in range(steps):
        minf = 1.0 / (1.0 + np.exp((-20.0 - V) / 12.0))
        ninf = 1.0 / (1.0 + np.exp((-16.0 - V) / 5.6))
        sinf = 1.0 / (1.0 + np.exp((-45.0 - V) / 10.0))

        I_ion = g_Ca * minf * (V - V_Ca) + g_K * n * (V - V_K) + g_S * s * (V - V_K)
        V += ((-I_ion) / Cm) * dt
        n = ninf + (n - ninf) * np.exp(-dt / tau_n)

        dW = np.random.normal(0.0, sqrt_dt)
        s_trunc = max(0.0, min(s, 1.0))
        diffusion = sigma * np.sqrt(s_trunc * (1.0 - s_trunc)) * dW
        s = (s + (dt / tau_s) * sinf + diffusion) / (1.0 + dt / tau_s)

        v_history[i] = V
    return v_history


@njit
def run_coupled_srk(sigma, g_S1, g_S2, g_c, dt, T_total, seed=0):
    np.random.seed(seed)
    steps = int(T_total / dt)
    V1, n1, s1 = -60.0, 0.1, 0.3
    V2, n2, s2 = -60.0, 0.1, 0.3
    v1_hist = np.zeros(steps)
    v2_hist = np.zeros(steps)
    Cm, g_Ca, V_Ca, g_K, V_K = 20.0, 3.6, 25.0, 10.0, -75.0
    tau_n, tau_s = 20.0, 20000.0
    sqrt_dt = np.sqrt(dt)

    for i in range(steps):
        minf1 = 1.0 / (1.0 + np.exp((-20.0 - V1) / 12.0))
        ninf1 = 1.0 / (1.0 + np.exp((-16.0 - V1) / 5.6))
        sinf1 = 1.0 / (1.0 + np.exp((-45.0 - V1) / 10.0))
        minf2 = 1.0 / (1.0 + np.exp((-20.0 - V2) / 12.0))
        ninf2 = 1.0 / (1.0 + np.exp((-16.0 - V2) / 5.6))
        sinf2 = 1.0 / (1.0 + np.exp((-45.0 - V2) / 10.0))

        I_gap = g_c * (V1 - V2)
        I_ion1 = g_Ca * minf1 * (V1 - V_Ca) + g_K * n1 * (V1 - V_K) + g_S1 * s1 * (V1 - V_K) + I_gap
        I_ion2 = g_Ca * minf2 * (V2 - V_Ca) + g_K * n2 * (V2 - V_K) + g_S2 * s2 * (V2 - V_K) - I_gap

        V1 += ((-I_ion1) / Cm) * dt
        V2 += ((-I_ion2) / Cm) * dt
        n1 = ninf1 + (n1 - ninf1) * np.exp(-dt / tau_n)
        n2 = ninf2 + (n2 - ninf2) * np.exp(-dt / tau_n)

        dW1 = np.random.normal(0.0, sqrt_dt)
        dW2 = np.random.normal(0.0, sqrt_dt)
        s1_trunc = max(0.0, min(s1, 1.0))
        s2_trunc = max(0.0, min(s2, 1.0))
        diff1 = sigma * np.sqrt(s1_trunc * (1.0 - s1_trunc)) * dW1
        diff2 = sigma * np.sqrt(s2_trunc * (1.0 - s2_trunc)) * dW2
        s1 = (s1 + (dt / tau_s) * sinf1 + diff1) / (1.0 + dt / tau_s)
        s2 = (s2 + (dt / tau_s) * sinf2 + diff2) / (1.0 + dt / tau_s)

        v1_hist[i] = V1
        v2_hist[i] = V2
    return v1_hist, v2_hist


def analyze_rhythm(v_data, dt):
    burn_in_steps = int(50000 / dt)
    v_steady = v_data[burn_in_steps:]
    threshold = -35.0
    burst_times = []
    lockout_steps = int(2500 / dt)
    last_event_step = -lockout_steps

    for i in range(1, len(v_steady)):
        if v_steady[i] > threshold and v_steady[i-1] <= threshold:
            if i - last_event_step > lockout_steps:
                burst_times.append(i * dt)
                last_event_step = i

    if len(burst_times) < 4:
        return np.nan

    ibi = np.diff(np.array(burst_times))
    return np.std(ibi) / np.mean(ibi)


def compute_centroid(sigmas, cv_curve, delta):
    log_sigmas = np.log(sigmas)
    log_cv = np.log(cv_curve)
    min_log_cv = np.nanmin(log_cv[~np.isinf(log_cv) & ~np.isnan(log_cv)])
    mask = log_cv <= min_log_cv + delta
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    log_opt = np.nanmean(log_sigmas[mask])
    log_lo = np.min(log_sigmas[mask])
    log_hi = np.max(log_sigmas[mask])
    return np.exp(log_opt), np.exp(log_lo), np.exp(log_hi)


@njit
def calc_sync_index(v1, v2, dt):
    burn_in_steps = int(50000 / dt)
    v1_s = v1[burn_in_steps:]
    v2_s = v2[burn_in_steps:]
    threshold = -35.0
    n = len(v1_s)
    spike1 = np.empty(n)
    spike2 = np.empty(n)
    for i in range(n):
        spike1[i] = 1.0 if v1_s[i] > threshold else 0.0
        spike2[i] = 1.0 if v2_s[i] > threshold else 0.0
    mean1 = np.mean(spike1)
    mean2 = np.mean(spike2)
    if mean1 == 0.0 or mean2 == 0.0 or mean1 == 1.0 or mean2 == 1.0:
        return 0.0
    cov = np.sum((spike1 - mean1) * (spike2 - mean2))
    var1 = np.sum((spike1 - mean1)**2)
    var2 = np.sum((spike2 - mean2)**2)
    return cov / np.sqrt(var1 * var2)
