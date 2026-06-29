import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.srk_model import run_srk, analyze_rhythm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def main():
    np.random.seed(42)

    dts = [0.5, 0.1, 0.05, 0.01]
    sigma = 1.5e-4
    g_S = 4.0
    n_trials = 5
    T_sim = 300000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for dt_val in dts:
        cvs, periods, freqs = [], [], []
        for t in range(n_trials):
            v = run_srk(sigma, g_S, dt_val, T_sim, seed=int(dt_val * 10000) + t)
            cv = analyze_rhythm(v, dt_val)
            if not np.isnan(cv):
                cvs.append(cv)
        if cvs:
            results[dt_val] = {
                'cv_mean': np.mean(cvs),
                'cv_se': np.std(cvs) / np.sqrt(len(cvs)),
                'n': len(cvs)
            }
        print(f"  dt={dt_val:5.2f} ms | CV={np.mean(cvs):.4f} ± {np.std(cvs)/np.sqrt(len(cvs)):.4f} | n={len(cvs)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dt_plot = list(results.keys())
    cv_m = [results[d]['cv_mean'] for d in dt_plot]
    cv_e = [results[d]['cv_se'] for d in dt_plot]

    axes[0].errorbar(dt_plot, cv_m, yerr=cv_e, fmt='o-', capsize=5, color='#d62728',
                     linewidth=2, markersize=8)
    axes[0].set_xlabel(r'$\Delta t$ (ms)', fontsize=13)
    axes[0].set_ylabel('Coefficient of Variation (CV)', fontsize=13)
    axes[0].set_title(r'(a) CV vs $\Delta t$', fontsize=14)
    axes[0].grid(True, ls='--', alpha=0.5)
    axes[0].set_xscale('log')

    if 0.1 in results:
        ref_cv = results[0.1]['cv_mean']
        print(f"\nConvergence (reference: dt=0.1 ms, CV={ref_cv:.4f}):")
        for d in dts:
            if d in results:
                rel_err = abs(results[d]['cv_mean'] - ref_cv) / ref_cv * 100
                print(f"  dt={d:5.2f} ms: CV={results[d]['cv_mean']:.4f}  (rel err: {rel_err:.1f}%)")
        axes[0].axhline(ref_cv, ls=':', color='gray', alpha=0.7, label='Reference dt=0.1')
        axes[0].legend()

    axes[1].text(0.1, 0.5, 'See console output for convergence table',
                 fontsize=12, va='center', ha='left')
    axes[1].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'dt_convergence.pdf'), dpi=300)
    print(f"Saved {OUTPUT_DIR}/dt_convergence.pdf")


if __name__ == "__main__":
    main()
