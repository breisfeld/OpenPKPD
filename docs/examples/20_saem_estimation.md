# Example 20 — SAEM Estimation

**Script:** `examples/20_saem_estimation.py`

Demonstrates the Stochastic Approximation EM (SAEM) algorithm on a
theophylline-like population PK dataset (12 subjects, 1-cmt oral model).

## Key concepts

| Concept | Description |
|---------|-------------|
| Two-phase algorithm | Phase 1 (γ=1, stochastic exploration) + Phase 2 (γ_k→0, convergence) |
| Metropolis-Hastings E-step | Samples individual ETAs per chain per subject |
| Closed-form Ω M-step | `Q_Ω ← Q_Ω + γ·(SS_Ω − Q_Ω)`; no gradient needed for Ω |
| OFV convergence history | Phase-2 parameter window stability criterion |

## Usage

```python
from openpkpd.estimation.saem import SAEMMethod

result = SAEMMethod(
    n_iter_phase1=300,
    n_iter_phase2=200,
    n_chains=3,
    seed=42,
).estimate(population_model, init_params)

print(f"OFV = {result.ofv:.4f}")
print(f"Converged: {result.converged}")

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(result.ofv_history)
plt.xlabel("Iteration")
plt.ylabel("OFV")
plt.title("SAEM convergence")
plt.show()
```

## Constructor options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter_phase1` | 300 | Phase-1 stochastic exploration iterations |
| `n_iter_phase2` | 200 | Phase-2 convergence iterations |
| `n_chains` | 3 | MH chains per subject (Rao-Blackwell averaging) |
| `n_parallel` | 1 | Parallel subject processing threads |
| `mh_step_size` | 0.3 | Initial MH proposal scale (adaptive) |
| `seed` | 42 | RNG seed for reproducibility |

## Convergence criterion

SAEM declares convergence when the relative change in the full parameter vector
`φ = [θ, lower_triangle(Ω), diag(Σ)]` between successive phase-2 windows
(`_PH2_WINDOW = 50` iterations) falls below `_PH2_TOL = 1e-3`.

## Comparison with FOCE

The script fits the same dataset with both SAEM and FOCE, printing parameter
estimates side by side. SAEM is slower per iteration but does not require the
first-order Taylor approximation that FOCE/FOCEI use.
