"""Check CV sensitivity using Brownian paths coupled across time steps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srk_model import aggregate_wiener_increments, analyze_rhythm, simulate_srk


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    time_steps = np.array([0.5, 0.1, 0.05, 0.01])
    reference_dt = float(np.min(time_steps))
    sigma = 1.5e-4
    g_s = 4.0
    duration = 150_000.0
    n_trials = 5
    cv_values = np.full((len(time_steps), n_trials), np.nan)
    rng = np.random.default_rng(42)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for trial in range(n_trials):
        fine_noise = rng.normal(
            0.0,
            np.sqrt(reference_dt),
            size=int(duration / reference_dt),
        )
        for dt_index, dt in enumerate(time_steps):
            factor = int(round(dt / reference_dt))
            noise = aggregate_wiener_increments(fine_noise, factor)
            voltage = simulate_srk(noise, float(dt), sigma, g_s=g_s)
            cv_values[dt_index, trial] = analyze_rhythm(voltage, float(dt))

    cv_means = np.array([
        np.mean(row[np.isfinite(row)]) if np.any(np.isfinite(row)) else np.nan
        for row in cv_values
    ])
    cv_errors = np.array(
        [
            np.std(row[np.isfinite(row)]) / np.sqrt(np.count_nonzero(np.isfinite(row)))
            if np.any(np.isfinite(row))
            else np.nan
            for row in cv_values
        ]
    )
    reference_cv = cv_means[np.argmin(time_steps)]

    for dt, mean, error, row in zip(time_steps, cv_means, cv_errors, cv_values):
        count = np.count_nonzero(np.isfinite(row))
        relative_error = abs(mean - reference_cv) / reference_cv * 100.0
        print(
            f"dt={dt:5.2f} ms | CV={mean:.4f} +/- {error:.4f} | "
            f"relative to dt={reference_dt:.2f}: {relative_error:.1f}% | n={count}"
        )

    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.errorbar(time_steps, cv_means, yerr=cv_errors, fmt="o-", capsize=5, color="#d62728")
    axis.axhline(
        reference_cv,
        linestyle=":",
        color="gray",
        label=f"reference dt={reference_dt:g} ms",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"Time step $\Delta t$ (ms)")
    axis.set_ylabel("Inter-burst interval CV")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dt_sensitivity.pdf")
    plt.close(figure)

    np.savez(
        OUTPUT_DIR / "dt_sensitivity.npz",
        time_steps=time_steps,
        reference_dt=reference_dt,
        cv_values=cv_values,
        cv_means=cv_means,
        cv_errors=cv_errors,
        sigma=sigma,
        g_s=g_s,
        duration=duration,
    )


if __name__ == "__main__":
    main()
