"""
Example 13: Stepwise Covariate Search on a Theophylline PK Model.

Demonstrates:
  1. Fitting a base 1-compartment theophylline model (ADVAN2, FOCE)
  2. Manually testing a weight (WT) power effect on CL using the LRT
  3. Demonstrating SCMEngine with multiple candidate covariate relationships

All data is embedded directly — no external file is required.

Covariates tested:
  - WT on CL  (power)
  - WT on V   (power)
  - AGE on CL (linear)
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.covariate.scm import SCMEngine, _lrt_pvalue
from openpkpd.data.dataset import NONMEMDataset


MAXEVAL = 80

# ---------------------------------------------------------------------------
# Embedded theophylline dataset (12 subjects, with WT and AGE covariates)
# ---------------------------------------------------------------------------
THEO_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT,AGE
1,0,4.02,0,1,1,79.6,29
1,0.27,0,0.74,0,0,79.6,29
1,0.57,0,1.72,0,0,79.6,29
1,1.02,0,7.91,0,0,79.6,29
1,1.92,0,8.31,0,0,79.6,29
1,3.5,0,8.33,0,0,79.6,29
1,5.02,0,6.85,0,0,79.6,29
1,7.03,0,6.08,0,0,79.6,29
1,9.0,0,5.40,0,0,79.6,29
1,12.05,0,4.55,0,0,79.6,29
1,24.37,0,1.25,0,0,79.6,29
2,0,4.40,0,1,1,72.4,44
2,0.35,0,0.96,0,0,72.4,44
2,0.60,0,2.33,0,0,72.4,44
2,1.07,0,4.71,0,0,72.4,44
2,2.13,0,8.33,0,0,72.4,44
2,3.50,0,9.02,0,0,72.4,44
2,5.02,0,7.14,0,0,72.4,44
2,7.02,0,5.68,0,0,72.4,44
2,9.10,0,4.55,0,0,72.4,44
2,12.10,0,3.01,0,0,72.4,44
2,25.0,0,0.90,0,0,72.4,44
3,0,4.95,0,1,1,70.5,62
3,0.27,0,0.64,0,0,70.5,62
3,0.58,0,1.92,0,0,70.5,62
3,1.02,0,4.44,0,0,70.5,62
3,1.92,0,7.03,0,0,70.5,62
3,3.50,0,9.07,0,0,70.5,62
3,5.02,0,7.56,0,0,70.5,62
3,7.02,0,6.59,0,0,70.5,62
3,9.00,0,5.88,0,0,70.5,62
3,12.15,0,4.73,0,0,70.5,62
3,24.17,0,1.25,0,0,70.5,62
4,0,4.53,0,1,1,58.2,38
4,0.27,0,1.02,0,0,58.2,38
4,0.52,0,3.31,0,0,58.2,38
4,1.00,0,4.90,0,0,58.2,38
4,1.92,0,7.24,0,0,58.2,38
4,3.50,0,8.00,0,0,58.2,38
4,5.02,0,6.81,0,0,58.2,38
4,7.03,0,5.87,0,0,58.2,38
4,9.00,0,5.22,0,0,58.2,38
4,12.12,0,4.55,0,0,58.2,38
4,24.35,0,1.25,0,0,58.2,38
5,0,5.86,0,1,1,88.0,32
5,0.27,0,1.29,0,0,88.0,32
5,0.58,0,3.08,0,0,88.0,32
5,1.02,0,6.44,0,0,88.0,32
5,2.02,0,8.76,0,0,88.0,32
5,3.50,0,7.94,0,0,88.0,32
5,5.02,0,7.28,0,0,88.0,32
5,7.02,0,6.06,0,0,88.0,32
5,9.10,0,5.58,0,0,88.0,32
5,12.00,0,4.57,0,0,88.0,32
5,24.35,0,1.17,0,0,88.0,32
6,0,4.00,0,1,1,76.0,54
6,1.15,0,4.53,0,0,76.0,54
6,2.03,0,8.00,0,0,76.0,54
6,3.57,0,9.75,0,0,76.0,54
6,5.00,0,9.10,0,0,76.0,54
6,7.00,0,7.17,0,0,76.0,54
6,9.22,0,5.68,0,0,76.0,54
6,12.10,0,4.42,0,0,76.0,54
6,23.85,0,1.58,0,0,76.0,54
"""


# ---------------------------------------------------------------------------
# Helper: build the base ModelBuilder (not yet built)
# ---------------------------------------------------------------------------
BASE_PK_CODE = """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""

BASE_ERROR_CODE = "Y = F*(1 + EPS(1))"


def make_base_builder(ds: NONMEMDataset) -> ModelBuilder:
    """
    Return a configured (but not built) ModelBuilder for the base theophylline
    model.  This is used by SCMEngine to rebuild models with covariate additions.
    """
    return (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — covariate search")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk(BASE_PK_CODE)
        .error(BASE_ERROR_CODE)
        .theta([(0.01, 1.5, 20.0), (0.001, 0.08, 5.0), (0.1, 30.0, 500.0)])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .covariates(["WT", "AGE"])
        .estimation(method="FOCE", maxeval=MAXEVAL)
    )


# ---------------------------------------------------------------------------
# Step 1: Fit base model
# ---------------------------------------------------------------------------

def fit_base_model(ds: NONMEMDataset):  # -> EstimationResult
    """Fit the base theophylline model."""
    builder = make_base_builder(ds)
    built = builder.build()
    print(f"Fitting base model (FOCE, maxeval={MAXEVAL})...")
    result = built.fit()
    print(result.summary())
    print()
    return result


# ---------------------------------------------------------------------------
# Step 2: Manual LRT for WT on CL (power effect)
# ---------------------------------------------------------------------------

def manual_lrt_wt_on_cl(ds: NONMEMDataset, base_ofv: float) -> None:
    """
    Manually test the power effect of WT on CL.

    $PK becomes::

        CL = THETA(2)*EXP(ETA(2)) * (WT/70)^THETA(4)

    One extra THETA → 1 degree of freedom for the LRT.
    """
    print("=" * 60)
    print("Manual LRT: WT (power) on CL")
    print("=" * 60)
    print("$PK code with covariate:")
    pk_code_with_cov = (
        BASE_PK_CODE.rstrip()
        + "\n"
        + "; Power effect of WT on CL (THETA(4))\n"
        + "CL = CL * (WT/70)**THETA(4)"
    )
    print(pk_code_with_cov)
    print()

    # Build model with covariate
    built_cov = (
        ModelBuilder()
        .problem("Theophylline — WT power on CL")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk(pk_code_with_cov)
        .error(BASE_ERROR_CODE)
        .theta([(0.01, 1.5, 20.0), (0.001, 0.08, 5.0), (0.1, 30.0, 500.0),
                (-2.0, 0.0, 2.0)])   # THETA(4): power covariate
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .covariates(["WT", "AGE"])
        .estimation(method="FOCE", maxeval=MAXEVAL)
        .build()
    )

    print("Fitting model with WT→CL power covariate...")
    result_cov = built_cov.fit()
    print(result_cov.summary())

    # Likelihood ratio test
    delta_ofv = base_ofv - result_cov.ofv   # improvement (positive = better)
    p_value = _lrt_pvalue(delta_ofv, df=1)

    print()
    print(f"Base OFV          : {base_ofv:.4f}")
    print(f"Covariate OFV     : {result_cov.ofv:.4f}")
    print(f"ΔOFV              : {delta_ofv:.4f}")
    print(f"p-value (LRT, 1df): {p_value:.4f}")
    print(f"THETA(4) [WT→CL]  : {result_cov.theta_final[3]:.4f}")
    if p_value < 0.05:
        print(">>> WT (power) on CL is SIGNIFICANT at 5% level.")
    else:
        print(">>> WT (power) on CL is NOT significant at 5% level.")
    print()


# ---------------------------------------------------------------------------
# Step 3: SCMEngine automatic covariate search
# ---------------------------------------------------------------------------

def run_scm_search(ds: NONMEMDataset) -> None:
    """
    Use SCMEngine for automated forward-backward covariate search.

    Candidate relationships:
      - WT  → CL  (power)
      - WT  → V   (power)
      - AGE → CL  (linear, centered at 40 years)
    """
    print("=" * 60)
    print("SCMEngine: Automatic Stepwise Covariate Search")
    print("=" * 60)

    candidates = [
        CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        ),
        CovariateRelationship(
            parameter="V",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        ),
        CovariateRelationship(
            parameter="CL",
            covariate="AGE",
            effect=CovariateEffect.LINEAR,
            reference=40.0,
        ),
    ]

    print("Candidates to test:")
    for c in candidates:
        print(f"  {c.parameter} ~ {c.covariate} [{c.effect.value}, ref={c.reference}]")
    print(f"Forward p-value threshold : 0.05")
    print(f"Backward p-value threshold: 0.001")
    print()

    # Create the base builder (not yet built)
    base_builder = make_base_builder(ds)

    engine = SCMEngine(
        base_model_builder=base_builder,
        base_pk_code=BASE_PK_CODE,
        candidates=candidates,
        forward_pvalue=0.05,
        backward_pvalue=0.001,
        estimation_method="FOCE",
        estimation_kwargs={"maxeval": MAXEVAL},
    )

    print(f"Running SCM (this may take a moment; maxeval={MAXEVAL})...")
    scm_result = engine.run()

    print()
    print(scm_result.summary())


# ---------------------------------------------------------------------------
# Step 4: Demonstrate CovariateRelationship effect application
# ---------------------------------------------------------------------------

def demonstrate_effect_application() -> None:
    """
    Show how each effect type modifies a parameter value numerically.
    """
    print("=" * 60)
    print("Covariate Effect Parameterizations — Numerical Demo")
    print("=" * 60)

    base_cl = 5.0   # L/hr
    ref_wt = 70.0
    ref_age = 40.0

    print(f"Base CL = {base_cl} L/hr\n")

    # Power effect: WT on CL
    power_rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=ref_wt)
    for wt in [35.0, 55.0, 70.0, 90.0, 120.0]:
        cl = power_rel.apply(base_cl, wt, theta_cov=0.75)
        print(f"  Power   WT={wt:5.1f} kg → CL = {cl:.4f} L/hr "
              f"(ratio = {cl / base_cl:.3f})")

    print()

    # Linear effect: AGE on CL
    linear_rel = CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR, reference=ref_age)
    for age in [20.0, 40.0, 60.0, 80.0]:
        cl = linear_rel.apply(base_cl, age, theta_cov=0.01)
        print(f"  Linear  AGE={age:4.1f} yr → CL = {cl:.4f} L/hr "
              f"(ratio = {cl / base_cl:.3f})")

    print()

    # Exponential effect: WT on V
    exp_rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL, reference=ref_wt)
    base_v = 30.0
    for wt in [50.0, 70.0, 90.0]:
        v = exp_rel.apply(base_v, wt, theta_cov=0.02)
        print(f"  ExpSN   WT={wt:5.1f} kg → V  = {v:.4f} L   "
              f"(ratio = {v / base_v:.3f})")

    print()

    # Code generation
    print("Generated NM-TRAN $PK code snippets:")
    print()
    for rel, theta_idx in [
        (power_rel, 4),
        (linear_rel, 5),
        (exp_rel, 6),
    ]:
        code = rel.generate_pk_code(theta_idx)
        for line in code.splitlines():
            print(f"  {line}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Example 13: Covariate Search — Theophylline PK")
    print("=" * 60)
    print()

    # Load embedded dataset
    df = pd.read_csv(io.StringIO(THEO_DATA))
    ds = NONMEMDataset.from_dataframe(df)
    print(f"Dataset: {ds}")
    print()

    # Step 1: Fit base model
    base_result = fit_base_model(ds)
    base_ofv = base_result.ofv

    # Step 2: Manual LRT for WT on CL
    manual_lrt_wt_on_cl(ds, base_ofv)

    # Step 3: Demonstrate effect application numerically (fast, no fitting)
    demonstrate_effect_application()

    # Step 4: Run SCM (calls fit() multiple times — may be slow)
    try:
        run_scm_search(ds)
    except Exception as exc:
        print(f"SCM search skipped (error during fitting): {exc}")
        print("This can occur when the optimizer cannot converge on small synthetic data.")

    print("Example 13 complete.")


if __name__ == "__main__":
    main()
