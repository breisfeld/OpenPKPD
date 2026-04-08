"""
Example 08: ODE Model — Transit Compartment Absorption (ADVAN6)

Demonstrates:
  - Defining an ODE model with ModelBuilder using a $DES block (ADVAN6)
  - Transit compartment absorption (3 transit compartments + central)
  - Simulating data from an ADVAN2 model, then fitting with a transit model
  - Comparing estimated vs. true individual predictions

Transit compartment model structure:
  Depot → T1 → T2 → T3 → Central → (elimination)
  KTR (transit rate constant) = (N+1) / MTT
  where N = number of transit compartments, MTT = mean transit time

Reference:
  Savic RM et al. (2007). Implementation of a transit compartment model
  for describing drug absorption in pharmacokinetic studies.
  J Pharmacokinet Pharmacodyn 34:711–726.

Model parameterization ($PK block):
  MTT = THETA(1) * EXP(ETA(1))  ; Mean transit time (h)
  CL  = THETA(2) * EXP(ETA(2))  ; Clearance (L/h)
  V   = THETA(3) * EXP(ETA(3))  ; Volume (L)
  KTR = 4 / MTT                 ; 3 transits + 1 → (N+1)/MTT = 4/MTT
  K   = CL / V
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.data.event_processor import DoseEvent


N_SUBJECTS = 2
MAXEVAL = 80


# ── Simulate reference data from ADVAN2 ──────────────────────────────────────

def _simulate_data(n_subj: int = 10, seed: int = 42) -> NONMEMDataset:
    """
    Simulate oral absorption data from a 1-compartment model (ADVAN2).

    True parameters:
      KA = 1.2 h^-1, CL = 5 L/h, V = 50 L
    Variability:
      20% CV on KA, CL, V (log-normal ETAs)
    Residual:
      20% proportional error
    """
    rng = np.random.default_rng(seed)
    advan2 = ADVAN2()

    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])

    # True population parameters
    KA_pop, CL_pop, V_pop = 1.2, 5.0, 50.0
    dose = 200.0  # mg

    rows: list[dict] = []
    for i in range(1, n_subj + 1):
        # Individual parameters
        KA_i = KA_pop * np.exp(rng.normal(0, 0.2))
        CL_i = CL_pop * np.exp(rng.normal(0, 0.2))
        V_i  = V_pop  * np.exp(rng.normal(0, 0.2))
        K_i  = CL_i / V_i

        pk_params = {"KA": KA_i, "K": K_i, "V": V_i}
        sol = advan2.solve(
            pk_params,
            [DoseEvent(time=0.0, amount=dose, compartment=1)],
            obs_times,
        )

        # Add proportional residual (20%)
        dv = np.maximum(
            sol.ipred * (1.0 + rng.normal(0, 0.2, len(obs_times))),
            0.001,
        )

        # Dose row (EVID=1)
        rows.append({
            "ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1
        })

        # Observation rows (EVID=0)
        for j, t in enumerate(obs_times):
            rows.append({
                "ID": i, "TIME": t, "AMT": 0.0, "DV": float(dv[j]),
                "EVID": 0, "MDV": 0
            })

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ── Transit compartment model with ADVAN6 ────────────────────────────────────

# $DES block: 4 compartments
#  A(1) = Depot / Transit 1
#  A(2) = Transit 2
#  A(3) = Transit 3
#  A(4) = Central (output)
_DES_CODE = """
DADT(1) = -KTR * A(1)
DADT(2) =  KTR * A(1) - KTR * A(2)
DADT(3) =  KTR * A(2) - KTR * A(3)
DADT(4) =  KTR * A(3) - K   * A(4)
"""

_PK_CODE = """
MTT = THETA(1) * EXP(ETA(1))
CL  = THETA(2) * EXP(ETA(2))
V   = THETA(3) * EXP(ETA(3))
KTR = 4.0 / MTT
K   = CL / V
"""

_ERROR_CODE = """
Y = F * (1 + EPS(1))
"""


def main() -> None:
    """Run Example 08: transit compartment absorption."""
    print("=" * 60)
    print("Example 08: Transit Compartment Absorption (ADVAN6)")
    print("=" * 60)

    # 1. Simulate reference data
    print("\nSimulating data from ADVAN2 (1-cmt oral)...")
    ds = _simulate_data(n_subj=N_SUBJECTS, seed=42)
    n_obs = (ds.df["EVID"] == 0).sum()
    print(f"  Dataset: {ds.df['ID'].nunique()} subjects, {n_obs} observations")

    # 2. Build transit compartment model (ADVAN6)
    print("\nBuilding transit compartment model (ADVAN6, 4 compartments)...")
    model = (
        ModelBuilder()
        .problem("Transit absorption — ADVAN6")
        .dataset(ds)
        .subroutines(advan=6, trans=1)
        .pk(_PK_CODE)
        .des(_DES_CODE)
        .error(_ERROR_CODE)
        # THETAs: MTT (mean transit time), CL, V
        .theta([
            (0.1, 2.0, 20.0),    # MTT: initial 2 h
            (0.1, 5.0, 50.0),    # CL: initial 5 L/h
            (5.0, 50.0, 500.0),  # V: initial 50 L
        ])
        # OMEGAs: one ETA each for MTT, CL, V
        .omega([0.1, 0.1, 0.1])
        # SIGMA: proportional residual
        .sigma(0.05)
        .estimation(method="FO", maxeval=MAXEVAL)
        .build()
    )

    # Attach des_callable to population model
    from openpkpd.parser.code_compiler import NMTRANCompiler
    compiler = NMTRANCompiler()
    des_callable = compiler.compile_des(_DES_CODE, n_compartments=4)
    model.population_model.des_callable = des_callable

    # Patch the pk_subroutine to use n_compartments=4 and pass des_callable
    from openpkpd.pk.ode.advan6 import ADVAN6 as _ADVAN6
    advan6 = _ADVAN6(n_compartments=4, rtol=1e-6, atol=1e-8)
    advan6.output_compartment = 4
    model.population_model.pk_subroutine = advan6

    # Patch individual models to pass des_callable through solve
    for sid, indiv in model.population_model._individual_models.items():
        indiv.pk_subroutine = advan6
        # Override evaluate to pass des_callable
        _original_evaluate = indiv.evaluate

        def _patched_evaluate(theta, eta, sigma, trans=1, _indiv=indiv, _des=des_callable):
            if _indiv.pk_callable is not None:
                pk_params = _indiv.pk_callable(list(theta), list(eta), t=0.0)
            else:
                pk_params = {}
            if "V" in pk_params and "V4" not in pk_params:
                pk_params["V4"] = pk_params["V"]
            try:
                micro = _indiv.pk_subroutine.apply_trans(pk_params, trans)
            except Exception:
                micro = pk_params
            obs_times = _indiv.subject_events.obs_times
            if len(obs_times) == 0:
                return np.array([]), np.array([], dtype=bool), np.array([])
            pk_sol = _indiv.pk_subroutine.solve(
                micro,
                _indiv.subject_events.dose_events,
                obs_times,
                des_callable=_des,
            )
            ipred = pk_sol.ipred
            f = pk_sol.f if pk_sol.f is not None else ipred
            obs_mask = _indiv.subject_events.observation_mask()
            return ipred, obs_mask, f

        indiv.evaluate = _patched_evaluate

    print(f"\nFitting transit compartment model (FO, maxeval={MAXEVAL})...")
    try:
        result = model.fit()
        usable_fit = result.ofv < 1.0e9
        print(f"\nEstimation complete:")
        print(f"  OFV = {result.ofv:.3f}")
        print(f"  Converged: {result.converged}")
        print(f"  Usable optimum: {usable_fit}")
        print(f"  Method: {result.method}")

        print(f"\n  THETA estimates:")
        param_names = ["MTT (h)", "CL (L/h)", "V (L)"]
        for i, (name, val) in enumerate(zip(param_names, result.theta_final)):
            print(f"    THETA({i+1}) [{name}] = {val:.4f}")

        print(f"\n  OMEGA (diagonal):")
        for k in range(result.omega_final.shape[0]):
            print(f"    OMEGA({k+1},{k+1}) = {result.omega_final[k,k]:.4f}")

        print(f"\n  SIGMA:")
        print(f"    SIGMA(1,1) = {result.sigma_final[0,0]:.4f}")
        if not usable_fit:
            print("\n  Warning: the objective remained on the penalty surface.")
            print("  This example demonstrates ADVAN6 transit-model setup, not a tuned estimation workflow.")

    except Exception as exc:
        print(f"  Estimation encountered an issue: {exc}")

    print("\nExample 08 complete.")


if __name__ == "__main__":
    main()
