"""Minimal single-oscillator example."""

import numpy as np

from srk_model import analyze_rhythm, simulate_srk


def main() -> None:
    sigma = 1.5e-4
    g_s = 4.0
    dt = 0.1
    duration = 120_000.0

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, np.sqrt(dt), size=int(duration / dt))
    voltage = simulate_srk(noise, dt, sigma, g_s=g_s)
    cv = analyze_rhythm(voltage, dt)

    print("3D SRK model with state-dependent slow-gate noise")
    print(f"g_s={g_s}, sigma={sigma:.2e}, duration={duration / 1000:.0f} s")
    print(f"inter-burst interval CV: {cv:.4f}")
    print(f"voltage range: [{voltage.min():.1f}, {voltage.max():.1f}] mV")


if __name__ == "__main__":
    main()
