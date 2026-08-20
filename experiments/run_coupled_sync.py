"""Compare threshold-occupancy synchronization with and without coupling."""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srk_model import simulate_coupled_srk, synchronization_index


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    sigmas = np.logspace(-6, -1.5, 25)
    couplings = np.array([0.0, 0.1])
    g_s_1 = 4.2
    g_s_2 = 4.2
    dt = 0.1
    duration = 200_000.0
    n_trials = 5
    sync_values = np.empty((len(couplings), len(sigmas), n_trials))
    rng = np.random.default_rng(42)

    OUTPUT_DIR.mkdir(exist_ok=True)
    started = time.time()
    for sigma_index, sigma in enumerate(sigmas):
        for trial in range(n_trials):
            noise = rng.normal(
                0.0,
                np.sqrt(dt),
                size=(2, int(duration / dt)),
            )
            for coupling_index, coupling in enumerate(couplings):
                voltage_1, voltage_2 = simulate_coupled_srk(
                    noise,
                    dt,
                    sigma,
                    g_s_1=g_s_1,
                    g_s_2=g_s_2,
                    coupling=float(coupling),
                )
                sync_values[coupling_index, sigma_index, trial] = synchronization_index(
                    voltage_1,
                    voltage_2,
                    dt,
                )
        means = np.mean(sync_values[:, sigma_index, :], axis=1)
        print(
            f"sigma={sigma:.4e}  uncoupled={means[0]:.3f}  "
            f"coupled={means[1]:.3f}"
        )

    sync_means = np.mean(sync_values, axis=2)
    sync_errors = np.std(sync_values, axis=2) / np.sqrt(n_trials)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for coupling_index, coupling in enumerate(couplings):
        axis.errorbar(
            sigmas,
            sync_means[coupling_index],
            yerr=sync_errors[coupling_index],
            marker="o",
            capsize=3,
            label=rf"$g_c={coupling:g}$",
        )
    axis.set_xscale("log")
    axis.set_xlabel(r"Noise intensity $\sigma$")
    axis.set_ylabel("Threshold-occupancy correlation")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "synchronization_comparison.pdf")
    plt.close(figure)

    np.savez(
        OUTPUT_DIR / "synchronization_comparison.npz",
        sigmas=sigmas,
        couplings=couplings,
        sync_values=sync_values,
        sync_means=sync_means,
        sync_errors=sync_errors,
        dt=dt,
        duration=duration,
        g_s_1=g_s_1,
        g_s_2=g_s_2,
    )
    print(f"Completed in {time.time() - started:.1f} s; outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
