import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.srk_model import run_coupled_srk, calc_sync_index

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def main():
    np.random.seed(42)

    g_S1 = 4.2
    g_S2 = 4.2
    g_c = 0.1
    dt = 0.1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating waveform comparison at three noise levels...")
    sigmas_test = [1e-6, 1.5e-4, 1e-1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, sig in enumerate(sigmas_test):
        v1, v2 = run_coupled_srk(sig, g_S1, g_S2, g_c, dt, T_total=150000, seed=i)
        start_idx = int(90000 / dt)
        end_idx = int(150000 / dt)
        time_ax = np.arange(start_idx, end_idx) * dt / 1000.0
        axes[i].plot(time_ax, v1[start_idx:end_idx], color='royalblue', linewidth=1.2,
                     label=r'Cell 1 ($g_S=4.2$)')
        axes[i].plot(time_ax, v2[start_idx:end_idx], color='firebrick', linewidth=1.2,
                     alpha=0.8, label=r'Cell 2 ($g_S=4.2$)')
        axes[i].set_ylabel('V (mV)', fontsize=12)
        axes[i].set_title(rf'$\sigma = {sig:.1e}$', fontsize=11)
        axes[i].grid(True, ls='--', alpha=0.4)
        if i == 0:
            axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)', fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'coupled_waveform.pdf'), dpi=300)
    print(f"Saved {OUTPUT_DIR}/coupled_waveform.pdf")
    plt.close()

    print("Scanning synchronization index (spike-masked Pearson R)...")
    sigmas_scan = np.logspace(-6, -1.5, 25)
    sync_means = []
    start_time = time.time()

    for idx, s_val in enumerate(sigmas_scan):
        trial_syncs = []
        for t in range(5):
            v1, v2 = run_coupled_srk(s_val, g_S1, g_S2, g_c, dt, T_total=200000,
                                      seed=idx * 10 + t)
            sync = calc_sync_index(v1, v2, dt)
            trial_syncs.append(sync)
        sync_means.append(np.mean(trial_syncs))
        print(f"  sigma={s_val:.4e}  sync={sync_means[-1]:.3f}")

    print(f"Done in {time.time() - start_time:.1f}s")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas_scan, sync_means, '-o', color='purple', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Multiplicative Noise Intensity $\sigma$', fontsize=14)
    ax.set_ylabel('Synchronization Index (Pearson R)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'sync_curve.pdf'), dpi=300)
    print(f"Saved {OUTPUT_DIR}/sync_curve.pdf")


if __name__ == "__main__":
    main()
