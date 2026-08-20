# Stochastic SRK Model

Reusable Python components from a research project on noise-driven transitions in a three-dimensional Sherman--Rinzel--Keizer (SRK) excitable system.

This repository contains selected public components of the numerical workflow: single-oscillator and gap-junction-coupled simulations, burst timing statistics, coherence-resonance scans, a coupled pathwise step-size check and a synchronization comparison.

## Model

The slow potassium-current activation variable follows

$$
ds = \frac{s_\infty(V)-s}{\tau_s}\,dt + \sigma\sqrt{s(1-s)}\,dW_t.
$$

The state-dependent diffusion vanishes at the physical boundaries. The implementation evaluates the diffusion coefficient at a truncated slow-gate value and advances the drift semi-implicitly. This controls numerical boundary excursions without claiming that every finite step remains exactly inside $[0,1]$.

The deterministic voltage and fast-gate dynamics use a compact three-dimensional SRK conductance model. A two-cell variant adds symmetric gap-junction coupling and independent noise inputs.

## Installation

```bash
git clone https://github.com/WuYefan77/srk-model.git
cd srk-model
python -m pip install -e .
```

Install plotting and test dependencies with:

```bash
python -m pip install -e ".[plots,test]"
```

## Python API

```python
import numpy as np

from srk_model import analyze_rhythm, simulate_srk

rng = np.random.default_rng(42)
dt = 0.1
n_steps = 1_200_000
noise = rng.normal(0.0, np.sqrt(dt), size=n_steps)

voltage = simulate_srk(
    noise,
    dt,
    sigma=1.5e-4,
    g_s=4.0,
)
cv = analyze_rhythm(voltage, dt)
print(f"inter-burst interval CV: {cv:.4f}")
```

Wiener increments are generated outside the solver. This makes random seeds, independent-noise experiments and common-random-number comparisons explicit.

The public API includes:

- `simulate_srk`
- `simulate_coupled_srk`
- `detect_burst_times`
- `analyze_rhythm`
- `synchronization_index`
- `aggregate_wiener_increments`
- `compute_log_centroid`

## Included experiments

```text
experiments/
├── run_cr_scan.py           coherence-resonance CV scan
├── run_dt_convergence.py    coupled-path step-size sensitivity
└── run_coupled_sync.py      coupled versus uncoupled synchronization
```

Run the lightweight example with:

```bash
python demo.py
```

Each experiment saves generated arrays and figures under `data/`, which is excluded from version control.

## Numerical scope

This public repository is a compact, reusable subset of the research code rather than a complete reproduction archive. It focuses on the model, numerical mechanisms and selected analyses represented by the included scripts.

The experiments can be used to inspect model-specific coherence resonance, time-step sensitivity and threshold-occupancy synchronization under weak electrical coupling. Interpretation beyond these included computations belongs to the broader research analysis and is not claimed as a repository-level result.

## Author

Yefan Wu, University of Sydney
