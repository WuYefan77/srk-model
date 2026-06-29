import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.srk_model import run_srk, analyze_rhythm
import numpy as np


def main():
    print("=" * 60)
    print("Minimal demo: 3D SRK model with Feller noise")
    print("=" * 60)

    sigma = 1.5e-4
    g_S = 4.0
    dt = 0.1
    T_sim = 300000

    v = run_srk(sigma, g_S, dt, T_sim, seed=42)
    cv = analyze_rhythm(v, dt)

    print(f"  g_S = {g_S}, sigma = {sigma:.2e}")
    print(f"  CV of inter-burst intervals = {cv:.4f}")
    print(f"  Voltage range: [{v.min():.1f}, {v.max():.1f}] mV")
    print()
    print("A single run completes in ~30s on a modern CPU.")
    print("See experiments/ for full scan scripts.")


if __name__ == "__main__":
    main()
