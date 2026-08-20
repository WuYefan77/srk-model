"""Scan inter-burst interval variability over the noise intensity."""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srk_model import analyze_rhythm, simulate_srk


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    sigmas = np.logspace(-6, -1, 25)
    g_s = 4.0
    dt = 0.1
    duration = 600_000.0
    n_trials = 10
    cv_values = np.full((len(sigmas), n_trials), np.nan)
    rng = np.random.default_rng(42)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(
        f"CV scan: {len(sigmas)} noise levels, {n_trials} trials, "
        f"{duration / 1000:.0f} s per trial"
    )
    started = time.time()

    for sigma_index, sigma in enumerate(sigmas):
        for trial in range(n_trials):
            noise = rng.normal(0.0, np.sqrt(dt), size=int(duration / dt))
            voltage = simulate_srk(noise, dt, sigma, g_s=g_s)
            cv_values[sigma_index, trial] = analyze_rhythm(voltage, dt)
        finite = cv_values[sigma_index, np.isfinite(cv_values[sigma_index])]
        mean = np.mean(finite) if len(finite) else np.nan
        print(f"  sigma={sigma:.4e}  CV={mean:.4f}  n={len(finite)}")

    cv_means = np.array([
        np.mean(row[np.isfinite(row)]) if np.any(np.isfinite(row)) else np.nan
        for row in cv_values
    ])
    cv_stds = np.array(
        [np.std(row[np.isfinite(row)]) if np.any(np.isfinite(row)) else np.nan for row in cv_values]
    )

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.errorbar(
        sigmas,
        cv_means,
        yerr=cv_stds,
        fmt="-o",
        capsize=4,
        color="#d62728",
        markerfacecolor="white",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"Noise intensity $\sigma$")
    axis.set_ylabel("Inter-burst interval CV")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)

    valid = np.isfinite(cv_means)
    if np.any(valid):
        optimum = sigmas[np.nanargmin(cv_means)]
        axis.axvline(
            optimum,
            linestyle="--",
            alpha=0.6,
            label=rf"minimum at $\sigma={optimum:.2e}$",
        )
        axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "coherence_resonance.pdf")
    plt.close(figure)
    np.savez(
        OUTPUT_DIR / "cr_scan.npz",
        sigmas=sigmas,
        cv_values=cv_values,
        cv_means=cv_means,
        cv_stds=cv_stds,
        dt=dt,
        duration=duration,
        g_s=g_s,
    )
    print(f"Completed in {time.time() - started:.1f} s; outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
