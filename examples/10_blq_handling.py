"""
Example 10: BLQ (Below Limit of Quantification) handling — M1 vs M3 vs M5.

Demonstrates:
  - Generating synthetic PK data from a 1-compartment oral (ADVAN2) model
    with proportional residual error and ~20% BLQ observations.
  - Fitting the same data with three BLQ handling methods:
      M1 (exclude BLQ rows)
      M3 (censored likelihood, P(Y < LLOQ))
      M5 (impute BLQ as LLOQ/2)
  - Comparing parameter estimates, OFV, AIC, and BIC across methods using
    openpkpd.inference.compare_models.
  - Illustrating that M3 recovers parameters more accurately than M1/M5
    when the fraction of BLQ observations is substantial.

All data are generated analytically (no ODE solver required) using the
1-compartment oral solution:
    C(t) = (F * Dose * KA) / (V * (KA - K10))
           * (exp(-K10 * t) - exp(-KA * t))
where K10 = CL / V.
"""

from __future__ import annotations

import io
import math
import warnings

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.blq import apply_m5_imputation, flag_blq_observations
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.inference import compare_models
from openpkpd.utils.constants import BLQMethod


# ---------------------------------------------------------------------------
# True population parameters (used for simulation)
# ---------------------------------------------------------------------------
TRUE_KA: float = 1.2    # hr^-1
TRUE_CL: float = 0.15   # L/hr
TRUE_V: float = 10.0    # L
TRUE_SIGMA: float = 0.05  # proportional CV (variance)

# IIV (log-normal)
IIV_KA: float = 0.20  # omega_KA (log-scale variance)
IIV_CL: float = 0.15  # omega_CL
IIV_V: float = 0.10   # omega_V

LLOQ: float = 0.30  # mg/L — set to capture ~20% BLQ at late time points
N_SUBJECTS: int = 20
DOSE: float = 100.0  # mg
OBS_TIMES: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
SEED: int = 42


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def one_cmt_oral(
    t: float,
    ka: float,
    cl: float,
    v: float,
    dose: float,
    f: float = 1.0,
) -> float:
    """
    Analytical 1-compartment oral concentration at time t.

    Args:
        t:    Observation time (hr).
        ka:   Absorption rate constant (hr^-1).
        cl:   Clearance (L/hr).
        v:    Volume of distribution (L).
        dose: Dose amount (mg).
        f:    Bioavailability fraction (default 1.0).

    Returns:
        Predicted concentration (mg/L). Returns 0 when t <= 0.
    """
    if t <= 0.0:
        return 0.0
    k10 = cl / v
    if abs(ka - k10) < 1e-8:
        # Handle ka ≈ k10 numerically
        ka = k10 * 1.001
    conc = (f * dose * ka) / (v * (ka - k10)) * (math.exp(-k10 * t) - math.exp(-ka * t))
    return max(conc, 0.0)


def simulate_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate a population PK dataset with proportional error and BLQ flags.

    Returns a DataFrame with columns:
        ID, TIME, AMT, DV, EVID, MDV, WT, LLOQ, BLQ

    Observations with DV < LLOQ are flagged (BLQ=1) but their true DV
    is retained so that the caller can apply different handling strategies.
    """
    records: list[dict] = []

    for subject_id in range(1, N_SUBJECTS + 1):
        # Sample individual parameters (log-normal)
        eta_ka = rng.normal(0.0, math.sqrt(IIV_KA))
        eta_cl = rng.normal(0.0, math.sqrt(IIV_CL))
        eta_v = rng.normal(0.0, math.sqrt(IIV_V))

        ka_i = TRUE_KA * math.exp(eta_ka)
        cl_i = TRUE_CL * math.exp(eta_cl)
        v_i = TRUE_V * math.exp(eta_v)

        # Dose record
        records.append(
            {
                "ID": subject_id,
                "TIME": 0.0,
                "AMT": DOSE,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "WT": 70.0,
                "LLOQ": LLOQ,
                "BLQ": 0,
            }
        )

        # Observation records
        for t in OBS_TIMES:
            ipred = one_cmt_oral(t, ka_i, cl_i, v_i, DOSE)
            # Proportional error: Y = IPRED * (1 + eps), eps ~ N(0, sigma)
            eps = rng.normal(0.0, math.sqrt(TRUE_SIGMA))
            dv_obs = ipred * (1.0 + eps)
            dv_obs = max(dv_obs, 0.0)  # concentration cannot be negative

            blq_flag = 1 if dv_obs < LLOQ else 0

            records.append(
                {
                    "ID": subject_id,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv_obs,
                    "EVID": 0,
                    "MDV": 0,
                    "WT": 70.0,
                    "LLOQ": LLOQ,
                    "BLQ": blq_flag,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset preparation per BLQ method
# ---------------------------------------------------------------------------


def prepare_m1_dataset(df_full: pd.DataFrame) -> NONMEMDataset:
    """
    M1: Exclude BLQ observations by setting MDV=1.

    This is the simplest approach: BLQ rows are excluded from the likelihood
    computation. It is conservative but discards information and can produce
    biased estimates of terminal elimination parameters.
    """
    df = df_full.copy()
    obs_mask = df["EVID"] == 0
    blq_mask = obs_mask & (df["BLQ"] == 1)
    df.loc[blq_mask, "MDV"] = 1
    return NONMEMDataset.from_dataframe(df)


def prepare_m3_dataset(df_full: pd.DataFrame) -> NONMEMDataset:
    """
    M3: Include all observations; BLQ flag and LLOQ are passed to the model.

    The censored likelihood P(Y < LLOQ) replaces the normal likelihood for
    BLQ observations. The DV for BLQ rows is set to 0 (the value is unused
    because the censored likelihood only depends on LLOQ and IPRED).
    """
    df = df_full.copy()
    obs_mask = df["EVID"] == 0
    blq_mask = obs_mask & (df["BLQ"] == 1)
    # Set DV=0 for BLQ rows; the likelihood will use LLOQ instead
    df.loc[blq_mask, "DV"] = 0.0
    return NONMEMDataset.from_dataframe(df)


def prepare_m5_dataset(df_full: pd.DataFrame) -> NONMEMDataset:
    """
    M5: Replace BLQ observations with LLOQ/2.

    A common na\"ive approach that imputes censored values with half the LLOQ.
    Easy to implement but introduces a systematic downward bias in the
    predicted concentrations at late time points.
    """
    df = df_full.copy()
    df = apply_m5_imputation(df, lloq_col="LLOQ", dv_col="DV")
    return NONMEMDataset.from_dataframe(df)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(ds: NONMEMDataset, method_label: str) -> "BuiltModel":  # noqa: F821
    """
    Assemble a 1-compartment oral FOCE model for the given dataset.

    Uses ADVAN2 TRANS2 (KA, CL, V parameterisation) with proportional
    residual error Y = F*(1 + EPS(1)).
    """
    return (
        ModelBuilder()
        .problem(f"1-cmt oral BLQ demo — {method_label}")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk(
            """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""
        )
        .error("Y = F*(1 + EPS(1))")
        .theta(
            [
                (0.1, TRUE_KA, 20.0),   # KA
                (0.01, TRUE_CL, 5.0),   # CL
                (1.0, TRUE_V, 100.0),   # V
            ]
        )
        .omega([IIV_KA, IIV_CL, IIV_V])
        .sigma(TRUE_SIGMA)
        .estimation(method="FOCE", interaction=True, maxeval=600)
        .build()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── 1. Simulate data ────────────────────────────────────────────────────
    print("=" * 60)
    print("Example 10: BLQ Handling — M1 vs M3 vs M5")
    print("=" * 60)

    df_full = simulate_dataset(rng)

    n_total_obs = int((df_full["EVID"] == 0).sum())
    n_blq = int(((df_full["EVID"] == 0) & (df_full["BLQ"] == 1)).sum())
    blq_pct = 100.0 * n_blq / n_total_obs if n_total_obs > 0 else 0.0

    print(f"\nSimulated dataset:")
    print(f"  Subjects          : {N_SUBJECTS}")
    print(f"  Total observations: {n_total_obs}")
    print(f"  BLQ observations  : {n_blq} ({blq_pct:.1f}%)")
    print(f"  LLOQ              : {LLOQ} mg/L")
    print(f"\nTrue parameters:")
    print(f"  KA = {TRUE_KA:.2f} hr-1")
    print(f"  CL = {TRUE_CL:.3f} L/hr")
    print(f"  V  = {TRUE_V:.2f} L")
    print(f"  SIGMA (prop) = {TRUE_SIGMA:.3f}")

    # ── 2. Prepare datasets for each method ─────────────────────────────────
    ds_m1 = prepare_m1_dataset(df_full)
    ds_m3 = prepare_m3_dataset(df_full)
    ds_m5 = prepare_m5_dataset(df_full)

    # ── 3. Fit models ────────────────────────────────────────────────────────
    results_dict: dict[str, object] = {}
    label_order = ["M1", "M3", "M5"]

    for label, ds in [("M1", ds_m1), ("M3", ds_m3), ("M5", ds_m5)]:
        print(f"\n{'─' * 50}")
        print(f"Fitting BLQ method: {label}")

        built = build_model(ds, label)

        # For M3: pass the LLOQ to enable censored likelihood
        if label == "M3":
            built.population_model.blq_method = BLQMethod.M3
            for subject_id in built.population_model.subject_ids():
                indiv = built.population_model.individual_model(subject_id)
                indiv.blq_method = BLQMethod.M3
                indiv.lloq = ds.lloq_values(subject_id)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = built.fit()

        # Store metadata
        n_obs_used = ds.n_observations()
        result.n_observations = n_obs_used
        result.n_subjects = ds.n_subjects()
        result.compute_n_parameters()

        results_dict[label] = result

        print(f"  OFV       = {result.ofv:.4f}")
        print(f"  Converged = {result.converged}")
        print(f"  n_obs     = {n_obs_used}")
        print(f"  AIC       = {result.aic:.4f}")
        print(f"  BIC       = {result.bic:.4f}")
        print(f"\n  Parameter estimates:")
        print(f"    KA = {result.theta_final[0]:.4f}  (true: {TRUE_KA:.2f})")
        print(f"    CL = {result.theta_final[1]:.4f}  (true: {TRUE_CL:.3f})")
        print(f"    V  = {result.theta_final[2]:.4f}  (true: {TRUE_V:.2f})")

        if len(result.eta_shrinkage) > 0:
            result.compute_shrinkage()
            sh_pct = [f"{s * 100:.1f}%" for s in result.eta_shrinkage]
            print(f"  ETA shrinkage: {sh_pct}")

    # ── 4. Compare models ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Model Comparison Table (sorted by AIC)")
    print("=" * 60)

    result_list = [results_dict[k] for k in label_order]  # type: ignore[index]
    comparison_df = compare_models(result_list, labels=label_order)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── 5. Interpretation ────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Key takeaways:")
    print(
        f"  M1 excludes {n_blq} BLQ observations, discarding information "
        f"about the terminal phase."
    )
    print(
        "  M3 uses the full censored likelihood, maximally preserving "
        "information from BLQ data."
    )
    print(
        "  M5 imputes BLQ as LLOQ/2, which can bias parameter estimates "
        "if BLQ% is high (>20%)."
    )
    best_label = comparison_df["Model"].iloc[0]
    print(f"\n  Best model by AIC: {best_label}")

    # ── 6. Optional: LRT between M3 and M1 (not nested, informational) ──────
    # Note: M1 and M3 are not nested so we cannot formally apply LRT, but
    # the OFV difference is reported for reference.
    ofv_m1 = results_dict["M1"].ofv  # type: ignore[union-attr]
    ofv_m3 = results_dict["M3"].ofv  # type: ignore[union-attr]
    print(f"\n  OFV(M1) - OFV(M3) = {ofv_m1 - ofv_m3:.2f}")
    print("  (Note: M1 vs M3 are not nested; this is illustrative only.)")


if __name__ == "__main__":
    main()
