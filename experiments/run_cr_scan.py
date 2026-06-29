import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.srk_model import run_srk, analyze_rhythm, compute_centroid

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def main():
    np.random.seed(42)

    sigmas = np.logspace(-6, -1, 25)
    g_S = 4.0
    dt = 0.1
    T_sim = 600000
    cv_means = []
    cv_stds = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Coherence resonance scan: {len(sigmas)} sigma points, 10 trials each, 600s each")
    start_time = time.time()
    for idx, s_val in enumerate(sigmas):
        trial_results = []
        for t in range(10):
            v_raw = run_srk(s_val, g_S, dt, T_sim, seed=idx * 100 + t)
            res_cv = analyze_rhythm(v_raw, dt)
            if not np.isnan(res_cv):
                trial_results.append(res_cv)
        if trial_results:
            cv_means.append(np.mean(trial_results))
            cv_stds.append(np.std(trial_results))
        else:
            cv_means.append(np.nan)
            cv_stds.append(np.nan)
        print(f"  {idx+1}/{len(sigmas)}  sigma={s_val:.5f}  CV={cv_means[-1]:.4f}")

    print(f"Done in {time.time() - start_time:.1f}s")

    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(sigmas, cv_means, yerr=cv_stds, fmt='-o', capsize=5,
                color='#d62728', markerfacecolor='white', markeredgewidth=2,
                markersize=8, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Multiplicative Noise Intensity $\sigma$', fontsize=14)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=14)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    valid = ~np.isnan(cv_means)
    if np.any(valid):
        opt_sigma = sigmas[np.nanargmin(cv_means)]
        ax.axvline(opt_sigma, ls='--', color='blue', alpha=0.5, label=f'CR optimum $\\sigma \\approx {opt_sigma:.2e}$')
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'coherence_resonance.pdf'), dpi=300)
    print(f"Saved {OUTPUT_DIR}/coherence_resonance.pdf")

    np.savez(os.path.join(OUTPUT_DIR, 'cr_scan.npz'),
             sigmas=sigmas, cv_means=cv_means, cv_stds=cv_stds)


if __name__ == "__main__":
    main()
