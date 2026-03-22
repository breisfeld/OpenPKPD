"""
Example 12: Non-Compartmental Analysis (NCA).

Demonstrates:
  - Simulating PK data for multiple subjects (1-compartment oral model)
  - Running NCA using NCAEngine for all subjects
  - Printing a summary table with Cmax, Tmax, AUC0-t, AUC0-inf, t½, CL/F
  - Bioequivalence analysis on AUC0-inf (first 5 vs last 5 subjects)

No compartmental model fitting is required — NCA is entirely data-driven.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from openpkpd.nca import NCAEngine, average_bioequivalence


# ---------------------------------------------------------------------------
# Simulation parameters (1-compartment oral model)
# ---------------------------------------------------------------------------
# C(t) = F*Dose/V * KA/(KA-K) * [exp(-K*t) - exp(-KA*t)]
# where K = CL/V and KA is the absorption rate constant.

N_SUBJECTS = 10
DOSE = 100.0            # mg (same for all subjects)
SAMPLING_TIMES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0])

RNG_SEED = 2026
RNG = np.random.default_rng(RNG_SEED)

# Population typical values
CL_TV = 5.0    # L/hr
V_TV = 50.0    # L
KA_TV = 1.5    # hr^-1
F_TV = 1.0     # bioavailability (fraction)

# Inter-individual variability (log-normal, CV ≈ 30%)
OMEGA_CL = 0.09    # variance on log(CL)
OMEGA_V = 0.09
OMEGA_KA = 0.09

# Residual error (proportional, CV ≈ 15%)
SIGMA_PROP = 0.15


def simulate_oral_pk(
    times: np.ndarray,
    cl: float,
    v: float,
    ka: float,
    dose: float,
    f: float = 1.0,
    sigma_prop: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate concentrations for a 1-compartment oral model.

    C(t) = F*Dose/V * KA/(KA-K) * [exp(-K*t) - exp(-KA*t)]

    Args:
        times:      Sampling times (hr). Time 0 concentration is 0 for oral.
        cl:         Clearance (L/hr).
        v:          Volume of distribution (L).
        ka:         Absorption rate constant (hr^-1).
        dose:       Administered dose.
        f:          Bioavailability fraction.
        sigma_prop: Proportional residual error CV.
        rng:        Random number generator for residual noise.

    Returns:
        Simulated concentration array.
    """
    k = cl / v
    if abs(ka - k) < 1e-8:
        ka = ka + 1e-6   # avoid numerical singularity

    prefix = f * dose / v * ka / (ka - k)
    conc = prefix * (np.exp(-k * times) - np.exp(-ka * times))
    conc = np.maximum(conc, 0.0)

    if sigma_prop > 0 and rng is not None:
        eps = rng.normal(0, sigma_prop, size=len(times))
        conc = conc * (1.0 + eps)
        conc = np.maximum(conc, 0.0)

    return conc


# ---------------------------------------------------------------------------
# Simulate PK data for N_SUBJECTS
# ---------------------------------------------------------------------------

records: list[dict] = []
subject_params: list[dict] = []

for i in range(1, N_SUBJECTS + 1):
    eta_cl = RNG.normal(0, math.sqrt(OMEGA_CL))
    eta_v = RNG.normal(0, math.sqrt(OMEGA_V))
    eta_ka = RNG.normal(0, math.sqrt(OMEGA_KA))

    cl_i = CL_TV * math.exp(eta_cl)
    v_i = V_TV * math.exp(eta_v)
    ka_i = KA_TV * math.exp(eta_ka)

    subject_params.append({"ID": i, "CL": cl_i, "V": v_i, "KA": ka_i})

    # Dose record (EVID=1)
    records.append({
        "ID": i, "TIME": 0.0, "AMT": DOSE,
        "DV": 0.0, "EVID": 1, "MDV": 1,
    })

    # Observation records (skip time 0 conc for oral)
    for t in SAMPLING_TIMES[1:]:  # exclude pre-dose time 0
        conc = simulate_oral_pk(
            np.array([t]), cl_i, v_i, ka_i, DOSE,
            sigma_prop=SIGMA_PROP, rng=RNG,
        )[0]
        records.append({
            "ID": i, "TIME": t, "AMT": 0.0,
            "DV": max(conc, 0.0), "EVID": 0, "MDV": 0,
        })

df = pd.DataFrame(records)

print("=" * 70)
print("Example 12: Non-Compartmental Analysis (NCA)")
print("=" * 70)
print(f"\nSimulated {N_SUBJECTS} subjects — 1-compartment oral model")
print(f"Dose: {DOSE} mg  |  Sampling times: {SAMPLING_TIMES.tolist()} hr")
print(f"Population: CL={CL_TV} L/hr, V={V_TV} L, Ka={KA_TV} hr^-1")
print(f"IIV: CV={math.sqrt(OMEGA_CL)*100:.0f}%  |  Residual: CV={SIGMA_PROP*100:.0f}%")


# ---------------------------------------------------------------------------
# Run NCA
# ---------------------------------------------------------------------------

engine = NCAEngine(
    auc_method="linear-log",      # linear-up, log-down (standard)
    lambda_z_method="auto",       # auto-select terminal regression window
    min_points_lambda=3,
    exclude_cmax=True,
)

print("\nRunning NCA ...", flush=True)
nca_df = engine.compute_dataset(
    df,
    id_col="ID",
    time_col="TIME",
    conc_col="DV",
    dose_col="AMT",
    dose_row_col="EVID",
    route="oral",
)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("NCA Summary Table")
print("=" * 70)

header = (
    f"{'ID':>4}  {'Cmax':>8}  {'Tmax':>6}  {'AUC_last':>10}  "
    f"{'AUC_inf':>10}  {'t_half':>8}  {'CL/F':>8}  {'R2_lz':>7}"
)
print(header)
print("-" * len(header))

for _, row in nca_df.iterrows():
    sid = int(row["subject_id"])
    cmax = row["cmax"]
    tmax = row["tmax"]
    auc_last = row["auc_last"]
    auc_inf = row["auc_inf"]
    t_half = row["t_half"]
    cl_f = row["cl_f"]
    r2 = row["r_squared"]

    def fmt(x: float, fmt_str: str = ".2f") -> str:
        return f"{x:{fmt_str}}" if not math.isnan(x) else "  N/A  "

    print(
        f"{sid:>4}  {fmt(cmax):>8}  {fmt(tmax, '.1f'):>6}  "
        f"{fmt(auc_last, '.2f'):>10}  {fmt(auc_inf, '.2f'):>10}  "
        f"{fmt(t_half, '.2f'):>8}  {fmt(cl_f, '.3f'):>8}  "
        f"{fmt(r2, '.4f'):>7}"
    )

# Print geometric mean summary
valid_auc = nca_df["auc_inf"].dropna()
valid_auc = valid_auc[valid_auc > 0]
if len(valid_auc) > 0:
    gmean_auc = math.exp(np.mean(np.log(valid_auc.values)))
    print("-" * len(header))
    print(f"  Geometric mean AUC_inf: {gmean_auc:.2f}")

valid_cl = nca_df["cl_f"].dropna()
valid_cl = valid_cl[valid_cl > 0]
if len(valid_cl) > 0:
    gmean_cl = math.exp(np.mean(np.log(valid_cl.values)))
    print(f"  Geometric mean CL/F:    {gmean_cl:.3f}")

valid_thalf = nca_df["t_half"].dropna()
valid_thalf = valid_thalf[valid_thalf > 0]
if len(valid_thalf) > 0:
    print(f"  Median t½:              {valid_thalf.median():.2f} hr")


# ---------------------------------------------------------------------------
# Bioequivalence analysis (first 5 vs last 5 subjects)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Bioequivalence Analysis: Group A (subjects 1-5) vs Group B (subjects 6-10)")
print("=" * 70)
print("(Illustrative only — same population, not a true crossover design)")

group_a = nca_df[nca_df["subject_id"].isin(range(1, 6))]["auc_inf"].dropna().values
group_b = nca_df[nca_df["subject_id"].isin(range(6, 11))]["auc_inf"].dropna().values

# Ensure we have paired observations
n_pairs = min(len(group_a), len(group_b))
if n_pairs >= 2:
    be_result_auc = average_bioequivalence(
        test_values=group_a[:n_pairs],
        reference_values=group_b[:n_pairs],
        metric="AUC0-inf",
        ci_level=0.90,
    )
    print(f"\nAUC0-inf comparison:")
    print(be_result_auc.summary())
else:
    print("Not enough valid AUC_inf values for BE analysis.")

# Cmax bioequivalence
group_a_cmax = nca_df[nca_df["subject_id"].isin(range(1, 6))]["cmax"].dropna().values
group_b_cmax = nca_df[nca_df["subject_id"].isin(range(6, 11))]["cmax"].dropna().values
n_pairs_cmax = min(len(group_a_cmax), len(group_b_cmax))

if n_pairs_cmax >= 2:
    be_result_cmax = average_bioequivalence(
        test_values=group_a_cmax[:n_pairs_cmax],
        reference_values=group_b_cmax[:n_pairs_cmax],
        metric="Cmax",
        ci_level=0.90,
    )
    print(f"\nCmax comparison:")
    print(be_result_cmax.summary())


# ---------------------------------------------------------------------------
# Optional: matplotlib concentration-time plot
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Per-subject NCA parameters vs true simulation values")
print("=" * 70)

# Compare estimated CL/F with true simulated CL
sp_df = pd.DataFrame(subject_params)
sp_df = sp_df.set_index("ID")
nca_indexed = nca_df.set_index("subject_id")

print(f"\n{'ID':>4}  {'True CL':>10}  {'Est. CL/F':>10}  {'Ratio':>8}")
print("-" * 40)
for sid in range(1, N_SUBJECTS + 1):
    true_cl = sp_df.loc[sid, "CL"]
    if sid in nca_indexed.index:
        est_clf = nca_indexed.loc[sid, "cl_f"]
        if not math.isnan(est_clf):
            ratio = est_clf / true_cl
            print(f"{sid:>4}  {true_cl:>10.3f}  {est_clf:>10.3f}  {ratio:>8.3f}")
        else:
            print(f"{sid:>4}  {true_cl:>10.3f}  {'N/A':>10}")
    else:
        print(f"{sid:>4}  {true_cl:>10.3f}  {'N/A':>10}")

print("\nNote: CL/F estimates ≈ true CL (since F=1.0 in simulation)")
print("\nDone.")


if __name__ == "__main__":
    pass  # All output produced at module level above
