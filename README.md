# SRK Feller CHAOS: Noise-Induced Functional Synchronization in a 3D Excitable System

Companion code for the paper *"Breakdown of Adiabatic Scaling and Noise-Induced Functional Synchronization in Deeply Quiescent Excitable Systems"* (under review, Chaos, 2026).

## Model

A 3D Sherman-Rinzel-Keizer (SRK) excitable oscillator. The slow variable $s$ (activation of the slow potassium current) is driven by Feller-type multiplicative noise:

$$ds = \frac{s_\infty(V) - s}{\tau_s} dt + \sigma \sqrt{s(1-s)}\, dW_t$$

The key features:
- The diffusion coefficient $\sqrt{s(1-s)}$ vanishes at physical bounds $s \in \{0,1\}$ (Feller boundary condition)
- $\tau_s = 20000$ ms — extreme time-scale separation from fast variables (V, n)
- Full-truncation semi-implicit Euler scheme preserves domain $[0,1]$

A two-cell network coupled by gap junctions ($I_{\text{gap}} = g_c (V_1 - V_2)$) demonstrates noise-induced functional synchronization: independent cells with independent noise converge to identical phase-locked bursting when $\sigma$ is tuned to the coherence resonance optimum.

## Repository Structure

```
src/
  srk_model.py             3D SRK model, coupled variant, stats helpers (Numba)
experiments/
  run_cr_scan.py           Coherence resonance scan (CV vs sigma)
  run_dt_convergence.py    Numerical convergence test (dt = 0.5..0.01 ms)
  run_coupled_sync.py      Two-cell gap-junction sync scan
demo.py                    Minimal single-run example
```

## Usage

```bash
pip install -r requirements.txt
python demo.py                           # ~30s single run
python experiments/run_cr_scan.py        # ~20 min, 25 sigma x 10 trials
python experiments/run_dt_convergence.py # ~5 min
python experiments/run_coupled_sync.py   # ~30 min
```

Outputs (PDF figures + .npz data) are written to `data/`.

## Key Findings

- **Coherence resonance**: CV exhibits a pronounced minimum at $\sigma^* \approx 1.5 \times 10^{-4}$, corresponding to a 7.3-second bursting period. Below $\sigma^*$, the system is quiescent; above, bursting becomes noise-jittered.
- **Adiabatic breakdown**: The standard $1/\sigma^2$ Kramers scaling fails near $\sigma^*$ — the extracted slope $R^2$ drops sharply, signaling the breakdown of the adiabatic approximation where the slow manifold is no longer a meaningful attractor at extreme quiescence.
- **Functional synchronization**: Two independent cells coupled by weak gap junctions ($g_c = 0.1$) show a peak in spike-masked Pearson correlation at $\sigma^*$, far exceeding the baseline value at all other noise levels. This synchronization is a *functional consequence* of the coherence resonance — not imposed by the coupling itself.

## Contact

Yefan Wu, `wuyefan718@gmail.com`
