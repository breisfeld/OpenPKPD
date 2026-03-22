"""
Unit tests for Non-Compartmental Analysis (NCA).

Tests cover:
  - Cmax and Tmax identification
  - AUC_last computation (linear and log trapezoidal)
  - Terminal half-life estimation via log-linear regression
  - CL/F derived parameter consistency
  - Bioequivalence analysis (ABE/TOST)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from openpkpd.nca.nca import NCAEngine, NCAParameters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine_linear() -> NCAEngine:
    """NCAEngine using linear trapezoidal rule."""
    return NCAEngine(auc_method="linear-trapezoidal")


@pytest.fixture()
def engine_loglinear() -> NCAEngine:
    """NCAEngine using linear-up-log-down (default)."""
    return NCAEngine(auc_method="linear-log")


# ---------------------------------------------------------------------------
# Cmax / Tmax
# ---------------------------------------------------------------------------


def test_cmax_tmax() -> None:
    """Cmax and Tmax must match the maximum observed concentration."""
    engine = NCAEngine()
    times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = np.array([0.0, 2.5, 4.1, 5.8, 4.2, 2.1, 1.0, 0.3])
    result = engine.compute_subject(times, conc, dose=100.0, route="oral")

    assert result.cmax == pytest.approx(5.8, rel=1e-3), "Cmax incorrect"
    assert result.tmax == pytest.approx(2.0, rel=1e-3), "Tmax incorrect"


def test_cmax_at_first_timepoint() -> None:
    """IV bolus: Cmax at time 0 (or first observation)."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    conc = np.array([10.0, 7.4, 5.5, 3.0, 0.9])
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert result.cmax == pytest.approx(10.0, rel=1e-6)
    assert result.tmax == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# AUC_last
# ---------------------------------------------------------------------------


def test_auc_last_positive_finite() -> None:
    """AUC_last must be positive and finite for a valid profile."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    conc = np.array([10.0, 7.4, 5.5, 3.0, 0.9])
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert result.auc_last > 0, "AUC_last must be positive"
    assert np.isfinite(result.auc_last), "AUC_last must be finite"


def test_auc_last_linear_trapezoidal_known_value() -> None:
    """
    For a step function [10, 10] over [0, 1], AUC = 10.
    Verifies linear trapezoidal arithmetic.
    """
    engine = NCAEngine(auc_method="linear-trapezoidal")
    times = np.array([0.0, 1.0])
    conc = np.array([10.0, 10.0])
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert result.auc_last == pytest.approx(10.0, rel=1e-9)


def test_auc_last_log_trapezoidal_monoexp() -> None:
    """
    For C(t) = C0 * exp(-K*t), exact AUC[0, t_last] = C0/K * (1 - exp(-K*t_last)).
    The log trapezoidal rule should be more accurate than linear for exponential decay.
    """
    K = 0.2
    C0 = 10.0
    t_last = 8.0
    times = np.linspace(0, t_last, 9)
    conc = C0 * np.exp(-K * times)
    exact = C0 / K * (1 - np.exp(-K * t_last))

    engine_log = NCAEngine(auc_method="linear-log")
    result_log = engine_log.compute_subject(times, conc, dose=100.0, route="IV")
    assert result_log.auc_last == pytest.approx(exact, rel=0.01)

    engine_lin = NCAEngine(auc_method="linear-trapezoidal")
    result_lin = engine_lin.compute_subject(times, conc, dose=100.0, route="IV")
    # Log-trapezoidal should be at least as accurate
    err_log = abs(result_log.auc_last - exact)
    err_lin = abs(result_lin.auc_last - exact)
    assert err_log <= err_lin + 1e-6


def test_iv_bolus_monoexponential_closed_form_parameters() -> None:
    """Closed-form IV bolus profile should yield consistent core NCA endpoints."""
    k = 0.2
    c0 = 10.0
    dose = 100.0
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = c0 * np.exp(-k * times)
    result = NCAEngine(auc_method="linear-log").compute_subject(times, conc, dose=dose, route="IV")

    auc_last_exact = c0 / k * (1.0 - np.exp(-k * times[-1]))
    auc_inf_exact = c0 / k
    t_half_exact = np.log(2.0) / k
    cl_exact = dose / auc_inf_exact
    vz_exact = cl_exact / k
    mrt_exact = 1.0 / k

    assert result.auc_last == pytest.approx(auc_last_exact, rel=1e-10)
    assert result.auc_inf == pytest.approx(auc_inf_exact, rel=1e-10)
    assert result.lambda_z == pytest.approx(k, rel=1e-10)
    assert result.t_half == pytest.approx(t_half_exact, rel=1e-10)
    assert result.cl_f == pytest.approx(cl_exact, rel=1e-10)
    assert result.vz_f == pytest.approx(vz_exact, rel=1e-10)
    assert result.mrt == pytest.approx(mrt_exact, rel=0.04)


def test_partial_auc_matches_closed_form_for_monoexponential_profile() -> None:
    """Partial AUC on a monoexponential decline should match the closed form."""
    k = 0.2
    c0 = 10.0
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = c0 * np.exp(-k * times)
    engine = NCAEngine(auc_method="linear-log")

    exact = c0 / k * (np.exp(-k * 2.0) - np.exp(-k * 8.0))
    partial = engine.compute_partial_auc(times, conc, 2.0, 8.0, method="log")
    assert partial == pytest.approx(exact, rel=1e-10)


def test_compute_subject_interpolates_user_supplied_t_last_boundary() -> None:
    """AUC/AUMC should include an interpolated endpoint when t_last falls between samples."""
    k = 0.2
    c0 = 10.0
    times = np.array([0.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = c0 * np.exp(-k * times)
    cutoff = 5.0
    c_at_cutoff = c0 * np.exp(-k * cutoff)
    engine = NCAEngine(auc_method="linear-log")

    params = engine.compute_subject(times, conc, dose=100.0, route="IV", t_last=cutoff)
    expected_auc, expected_aumc = engine._compute_auc_pair(
        np.array([0.0, 2.0, 4.0, cutoff]),
        np.array([conc[0], conc[1], conc[2], c_at_cutoff]),
    )

    assert params.auc_last == pytest.approx(expected_auc, rel=1e-12)
    assert params.aumc_last == pytest.approx(expected_aumc, rel=1e-12)


def test_iv_bolus_back_extrapolates_c0_and_preserves_observed_cmax() -> None:
    """IV bolus profiles without a time-zero sample should reconstruct C0 for AUC/AUMC only."""
    k = 0.2
    c0 = 12.0
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = c0 * np.exp(-k * times)
    engine = NCAEngine(auc_method="linear-log")

    params = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert params.c0 == pytest.approx(c0, rel=1e-12)
    assert params.cmax == pytest.approx(conc[0], rel=1e-12)
    assert params.tmax == pytest.approx(times[0], rel=1e-12)
    assert params.auc_inf == pytest.approx(c0 / k, rel=1e-10)
    assert params.mrt == pytest.approx(1.0 / k, rel=1e-10)


# ---------------------------------------------------------------------------
# Terminal half-life
# ---------------------------------------------------------------------------


def test_half_life_monoexponential() -> None:
    """For C(t) = C0*exp(-K*t), t½ = ln(2)/K."""
    K = 0.1
    engine = NCAEngine(min_points_lambda=3)
    times = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0])
    conc = 100.0 * np.exp(-K * times)
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    expected_thalf = np.log(2) / K  # ≈ 6.931
    assert result.t_half == pytest.approx(expected_thalf, rel=0.05), (
        f"t_half={result.t_half:.4f}, expected={expected_thalf:.4f}"
    )


def test_lambda_z_positive() -> None:
    """lambda_z must be strictly positive for declining profiles."""
    engine = NCAEngine()
    times = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = np.array([8.0, 6.0, 4.0, 2.0, 1.0, 0.3])
    result = engine.compute_subject(times, conc, dose=50.0, route="oral")

    assert np.isfinite(result.lambda_z), "lambda_z should be estimable"
    assert result.lambda_z > 0, "lambda_z must be positive"


def test_r_squared_high_for_monoexp() -> None:
    """R² for terminal regression on monoexponential data should be near 1."""
    K = 0.15
    engine = NCAEngine()
    times = np.linspace(2, 30, 10)
    conc = 50.0 * np.exp(-K * times)
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert result.r_squared > 0.999, f"R²={result.r_squared:.6f} too low"


def test_compute_lambda_z_matches_linregress_reference_search() -> None:
    """lambda_z search should match the historical linregress-based reference."""
    engine = NCAEngine(min_points_lambda=3)
    times = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
    conc = np.array([11.2, 8.1, 6.5, 4.9, 2.7, 1.6, 0.8])

    lambda_z, r_squared, n_points = engine._compute_lambda_z(times, conc)

    best_r2_adj = -np.inf
    best_lambda = float("nan")
    best_r2 = float("nan")
    best_n = 0
    log_conc = np.log(conc)
    for k in range(engine.min_points_lambda, len(times) + 1):
        slope, _intercept, r_value, _p_value, _se_slope = stats.linregress(
            times[-k:], log_conc[-k:]
        )
        if slope >= 0:
            continue
        candidate_r2 = float(r_value**2)
        candidate_r2_adj = 1.0 - (1.0 - candidate_r2) * (k - 1) / (k - 2)
        if candidate_r2_adj > best_r2_adj:
            best_r2_adj = candidate_r2_adj
            best_lambda = -float(slope)
            best_r2 = candidate_r2
            best_n = k

    assert lambda_z == pytest.approx(best_lambda, rel=1e-12)
    assert r_squared == pytest.approx(best_r2, rel=1e-12)
    assert n_points == best_n


def test_half_life_nan_for_flat_profile() -> None:
    """Flat profile: no elimination trend, so t_half should be NaN."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0])
    conc = np.array([5.0, 5.0, 5.0, 5.0])  # constant — no decline
    result = engine.compute_subject(times, conc, dose=100.0, route="IV")

    assert np.isnan(result.t_half), "t_half should be NaN for non-declining data"


# ---------------------------------------------------------------------------
# CL/F consistency
# ---------------------------------------------------------------------------


def test_cl_f() -> None:
    """CL/F = Dose / AUC_inf must hold exactly."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = np.array([10.0, 7.4, 5.5, 3.0, 0.9, 0.27, 0.02])
    dose = 100.0
    result = engine.compute_subject(times, conc, dose=dose, route="IV")

    if np.isfinite(result.auc_inf) and result.auc_inf > 0:
        expected_cl_f = dose / result.auc_inf
        assert result.cl_f == pytest.approx(expected_cl_f, rel=1e-6), (
            f"CL/F={result.cl_f:.6f}, expected {expected_cl_f:.6f}"
        )


def test_vz_f_consistency() -> None:
    """Vz/F = CL/F / lambda_z must hold when all parameters are estimated."""
    K = 0.1
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 36.0])
    conc = 50.0 * np.exp(-K * times)
    result = engine.compute_subject(times, conc, dose=50.0, route="IV")

    if (
        np.isfinite(result.vz_f)
        and np.isfinite(result.cl_f)
        and np.isfinite(result.lambda_z)
        and result.lambda_z > 0
    ):
        expected_vz_f = result.cl_f / result.lambda_z
        assert result.vz_f == pytest.approx(expected_vz_f, rel=1e-6)


# ---------------------------------------------------------------------------
# MRT
# ---------------------------------------------------------------------------


def test_mrt_positive() -> None:
    """MRT must be positive and finite for a valid profile."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = np.array([10.0, 7.4, 5.5, 3.0, 0.9, 0.27, 0.02])
    result = engine.compute_subject(times, conc, dose=100.0)

    if np.isfinite(result.mrt):
        assert result.mrt > 0, "MRT must be positive"


def test_infusion_mrt_subtracts_half_infusion_duration() -> None:
    """IV infusion MRT should subtract the mean input time (Tinf / 2)."""
    engine = NCAEngine(auc_method="linear-log")
    times = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    conc = np.array([0.0, 1.5, 1.1, 0.8, 0.45, 0.18, 0.05])

    generic = engine.compute_subject(times, conc, dose=25.0, route="oral")
    infusion = engine.compute_subject(
        times,
        conc,
        dose=25.0,
        route="infusion",
        infusion_duration=0.25,
    )

    assert infusion.auc_inf == pytest.approx(generic.auc_inf, rel=1e-12)
    assert infusion.aumc_inf == pytest.approx(generic.aumc_inf, rel=1e-12)
    assert infusion.mrt == pytest.approx(generic.mrt - 0.125, rel=1e-12)


# ---------------------------------------------------------------------------
# NaN / missing values handling
# ---------------------------------------------------------------------------


def test_nan_concentrations_handled() -> None:
    """NaN concentrations should be silently excluded."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    conc = np.array([10.0, float("nan"), 5.5, 3.0, 0.9])
    result = engine.compute_subject(times, conc, dose=100.0)

    # Should not raise; AUC should be computable from remaining points
    assert not np.isnan(result.auc_last), "AUC_last should not be NaN after dropping NaN conc"


def test_all_nan_concentrations() -> None:
    """All-NaN concentrations should return NaN parameters without error."""
    engine = NCAEngine()
    times = np.array([0.0, 1.0, 2.0])
    conc = np.array([float("nan")] * 3)
    result = engine.compute_subject(times, conc, dose=100.0)

    assert np.isnan(result.cmax)
    assert np.isnan(result.auc_last)


def test_single_observation() -> None:
    """Single observation should not crash; AUC should be NaN or 0."""
    engine = NCAEngine()
    result = engine.compute_subject(np.array([1.0]), np.array([5.0]), dose=100.0)
    # AUC requires at least 2 points
    assert np.isnan(result.auc_last) or result.auc_last == 0.0


# ---------------------------------------------------------------------------
# NCAParameters dataclass
# ---------------------------------------------------------------------------


def test_nca_parameters_defaults() -> None:
    """Default NCAParameters should have NaN for all float fields."""
    params = NCAParameters(subject_id=1, dose=100.0, route="oral")
    assert np.isnan(params.cmax)
    assert np.isnan(params.c0)
    assert np.isnan(params.auc_last)
    assert np.isnan(params.lambda_z)
    assert params.n_points_lambda == 0


def test_nca_parameters_to_dict() -> None:
    """to_dict() should return a dictionary with the expected keys."""
    params = NCAParameters(subject_id="S01", dose=50.0, route="IV")
    d = params.to_dict()
    assert "cmax" in d
    assert "c0" in d
    assert "auc_inf" in d
    assert "cl_f" in d
    assert d["subject_id"] == "S01"


# ---------------------------------------------------------------------------
# Bioequivalence
# ---------------------------------------------------------------------------


def test_iv_iv_bioequivalence() -> None:
    """ABE test: identical datasets should give GMR = 1.0 and be bioequivalent."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    auc = np.array([100.0, 120.0, 90.0, 110.0, 95.0])
    result = average_bioequivalence(auc, auc)

    assert result.gmr == pytest.approx(1.0, abs=1e-6)
    assert result.bioequivalent, "Identical datasets must be bioequivalent"


def test_bioequivalence_ci_contains_gmr() -> None:
    """The CI must bracket the GMR."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    test_auc = np.array([95.0, 105.0, 100.0, 98.0, 102.0])
    ref_auc = np.array([100.0, 110.0, 105.0, 103.0, 108.0])
    result = average_bioequivalence(test_auc, ref_auc)

    assert result.gmr_ci_lo <= result.gmr, "CI lower must be <= GMR"
    assert result.gmr <= result.gmr_ci_hi, "GMR must be <= CI upper"


def test_bioequivalence_gmr_range() -> None:
    """GMR should be a plausible positive ratio."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    test_auc = np.array([95.0, 105.0, 100.0, 98.0, 102.0])
    ref_auc = np.array([100.0, 110.0, 105.0, 103.0, 108.0])
    result = average_bioequivalence(test_auc, ref_auc)

    assert 0.5 < result.gmr < 2.0


def test_bioequivalence_clearly_not_be() -> None:
    """Highly different formulations must not be declared bioequivalent."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    test_auc = np.array([50.0, 55.0, 45.0, 48.0, 52.0])  # ~50% of reference
    ref_auc = np.array([100.0, 105.0, 95.0, 98.0, 102.0])
    result = average_bioequivalence(test_auc, ref_auc)

    assert not result.bioequivalent, "Clearly different formulations must not pass BE"


def test_bioequivalence_exact_lower_limit_is_accepted() -> None:
    """A CI exactly at 80% should remain bioequivalent despite roundoff."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    ref_auc = np.array([100.0, 120.0, 90.0, 110.0, 95.0])
    test_auc = 0.80 * ref_auc
    result = average_bioequivalence(test_auc, ref_auc)

    assert result.gmr == pytest.approx(0.80)
    assert result.gmr_ci_lo == pytest.approx(0.80)
    assert result.gmr_ci_hi == pytest.approx(0.80)
    assert result.bioequivalent


def test_bioequivalence_invalid_input_raises() -> None:
    """Non-positive values or mismatched lengths must raise ValueError."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    with pytest.raises(ValueError, match="strictly positive"):
        average_bioequivalence(
            np.array([100.0, -10.0, 90.0]),
            np.array([100.0, 100.0, 100.0]),
        )

    with pytest.raises(ValueError, match="same shape"):
        average_bioequivalence(
            np.array([100.0, 110.0]),
            np.array([100.0]),
        )


def test_bioequivalence_summary_str() -> None:
    """BEResult.summary() should return a non-empty string."""
    from openpkpd.nca.bioequivalence import average_bioequivalence

    result = average_bioequivalence(
        np.array([100.0, 105.0, 95.0]),
        np.array([100.0, 100.0, 100.0]),
    )
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "GMR" in s


# ---------------------------------------------------------------------------
# compute_dataset
# ---------------------------------------------------------------------------


def test_compute_dataset_returns_one_row_per_subject() -> None:
    """compute_dataset should return exactly one row per unique subject."""
    import pandas as pd

    engine = NCAEngine()
    # Minimal NONMEM-format dataframe
    rows = []
    for sid in [1, 2, 3]:
        rows.append({"ID": sid, "TIME": 0.0, "DV": 0.0, "AMT": 100.0, "EVID": 1})
        for t, c in zip([1.0, 2.0, 4.0, 8.0], [8.0, 6.0, 3.5, 1.2], strict=False):
            rows.append({"ID": sid, "TIME": t, "DV": c, "AMT": 0.0, "EVID": 0})
    df = pd.DataFrame(rows)

    result_df = engine.compute_dataset(df)
    assert len(result_df) == 3, f"Expected 3 rows, got {len(result_df)}"
    assert "cmax" in result_df.columns
    assert "auc_last" in result_df.columns


def test_compute_dataset_preserves_subject_specific_closed_form_auc_inf() -> None:
    """Dataset-level NCA should preserve subject-specific exact monoexponential AUC_inf."""
    import pandas as pd

    engine = NCAEngine(auc_method="linear-log")
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    subjects = [
        {"ID": 1, "DOSE": 100.0, "C0": 10.0, "K": 0.2},
        {"ID": 2, "DOSE": 180.0, "C0": 15.0, "K": 0.25},
    ]
    rows = []
    expected_auc_inf = {}

    for subj in subjects:
        sid = subj["ID"]
        dose = subj["DOSE"]
        c0 = subj["C0"]
        k = subj["K"]
        rows.append({"ID": sid, "TIME": 0.0, "DV": 0.0, "AMT": dose, "EVID": 1})
        conc = c0 * np.exp(-k * times)
        for t, c in zip(times, conc, strict=False):
            rows.append({"ID": sid, "TIME": float(t), "DV": float(c), "AMT": 0.0, "EVID": 0})
        expected_auc_inf[sid] = c0 / k

    result_df = engine.compute_dataset(pd.DataFrame(rows), route="IV").set_index("subject_id")
    for sid, auc_inf in expected_auc_inf.items():
        assert float(result_df.loc[sid, "auc_inf"]) == pytest.approx(auc_inf, rel=1e-10)


def test_compute_dataset_forwards_infusion_duration_to_mrt() -> None:
    """Dataset-level NCA should forward infusion_duration to subject-level MRT logic."""
    import pandas as pd

    engine = NCAEngine(auc_method="linear-log")
    rows = [
        {"ID": 1, "TIME": 0.0, "DV": 0.0, "AMT": 25.0, "EVID": 1},
        {"ID": 1, "TIME": 0.0, "DV": 0.0, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 0.25, "DV": 1.5, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 0.5, "DV": 1.1, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 1.0, "DV": 0.8, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 2.0, "DV": 0.45, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 4.0, "DV": 0.18, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 8.0, "DV": 0.05, "AMT": 0.0, "EVID": 0},
    ]
    df = pd.DataFrame(rows)

    subject = engine.compute_subject(
        np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        np.array([0.0, 1.5, 1.1, 0.8, 0.45, 0.18, 0.05]),
        dose=25.0,
        route="infusion",
        infusion_duration=0.25,
    )
    result_df = engine.compute_dataset(df, route="infusion", infusion_duration=0.25)

    assert float(result_df.loc[0, "mrt"]) == pytest.approx(subject.mrt, rel=1e-12)


def test_compute_dataset_preserves_first_seen_subject_order() -> None:
    """Dataset-level NCA output should preserve the first-seen subject order."""
    import pandas as pd

    engine = NCAEngine()
    rows = [
        {"ID": 2, "TIME": 0.0, "DV": 0.0, "AMT": 100.0, "EVID": 1},
        {"ID": 2, "TIME": 1.0, "DV": 8.0, "AMT": 0.0, "EVID": 0},
        {"ID": 2, "TIME": 2.0, "DV": 5.0, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 0.0, "DV": 0.0, "AMT": 120.0, "EVID": 1},
        {"ID": 1, "TIME": 1.0, "DV": 7.0, "AMT": 0.0, "EVID": 0},
        {"ID": 1, "TIME": 2.0, "DV": 4.0, "AMT": 0.0, "EVID": 0},
        {"ID": 3, "TIME": 0.0, "DV": 0.0, "AMT": 90.0, "EVID": 1},
        {"ID": 3, "TIME": 1.0, "DV": 6.0, "AMT": 0.0, "EVID": 0},
        {"ID": 3, "TIME": 2.0, "DV": 3.0, "AMT": 0.0, "EVID": 0},
    ]

    result_df = engine.compute_dataset(pd.DataFrame(rows), route="IV")

    assert result_df["subject_id"].tolist() == [2, 1, 3]


# ---------------------------------------------------------------------------
# C4: Theophylline NCA reference values
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheophyllineNCAReference:
    """
    C4: Verify NCA parameters against analytically computed reference values
    for a Theophylline-like oral 1-compartment model.

    Reference PK parameters (Sheiner 1982):
        KA  ≈ 1.5  h⁻¹
        CL  ≈ 0.04 L/h/kg × 70 kg = 2.8 L/h
        V   ≈ 0.47 L/kg   × 70 kg = 32.9 L
        Dose = 320 mg (oral, F=1)

    Analytical 1-cmt oral solution:
        C(t) = F*D*KA / (V*(KA-K)) * (exp(-K*t) - exp(-KA*t))

    Analytical reference values:
        AUC_inf = F*D / CL               ≈ 114.3 mg·h/L
        Cmax    ≈ derived from tmax
        Tmax    = ln(KA/K) / (KA - K)   ≈ 1.90 h
        T½      = ln(2) / K             ≈ 8.13 h
    """

    KA = 1.5
    CL = 2.8
    V = 32.9
    DOSE = 320.0  # mg
    F = 1.0

    @property
    def _K(self) -> float:
        return self.CL / self.V

    @property
    def _tmax_analytical(self) -> float:
        return np.log(self.KA / self._K) / (self.KA - self._K)

    @property
    def _auc_inf_analytical(self) -> float:
        return self.F * self.DOSE / self.CL

    @property
    def _thalf_analytical(self) -> float:
        return np.log(2) / self._K

    def _generate_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate dense Theophylline concentration-time profile."""
        times = np.array(
            [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
        )
        ka, k, v, dose, f = self.KA, self._K, self.V, self.DOSE, self.F
        conc = np.where(
            times > 0,
            f * dose * ka / (v * (ka - k)) * (np.exp(-k * times) - np.exp(-ka * times)),
            0.0,
        )
        return times, conc

    def test_auc_inf_within_5pct(self) -> None:
        """AUC_inf from NCA should be within 5% of dose/CL."""
        times, conc = self._generate_profile()
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(times, conc, dose=self.DOSE, route="oral")

        ref = self._auc_inf_analytical
        assert np.isfinite(result.auc_inf), "AUC_inf should be estimable"
        assert result.auc_inf == pytest.approx(ref, rel=0.05), (
            f"AUC_inf={result.auc_inf:.2f}, expected≈{ref:.2f}"
        )

    def test_tmax_within_tolerance(self) -> None:
        """Observed Tmax should be within 0.5 h of analytical Tmax."""
        times, conc = self._generate_profile()
        engine = NCAEngine()
        result = engine.compute_subject(times, conc, dose=self.DOSE, route="oral")

        tmax_ref = self._tmax_analytical
        assert result.tmax == pytest.approx(tmax_ref, abs=0.5), (
            f"Tmax={result.tmax:.2f} h, analytical={tmax_ref:.2f} h"
        )

    def test_thalf_within_10pct(self) -> None:
        """Terminal half-life should match ln(2)/K to within 10%."""
        times, conc = self._generate_profile()
        engine = NCAEngine(min_points_lambda=4)
        result = engine.compute_subject(times, conc, dose=self.DOSE, route="oral")

        thalf_ref = self._thalf_analytical
        assert np.isfinite(result.t_half), "t_half should be estimable"
        assert result.t_half == pytest.approx(thalf_ref, rel=0.10), (
            f"t_half={result.t_half:.2f} h, analytical={thalf_ref:.2f} h"
        )

    def test_clearance_volume_and_lambda_match_reference_within_tolerance(self) -> None:
        """Derived oral NCA parameters should stay close to the analytical profile truth."""
        times, conc = self._generate_profile()
        engine = NCAEngine(auc_method="linear-log", min_points_lambda=4)
        result = engine.compute_subject(times, conc, dose=self.DOSE, route="oral")

        assert np.isfinite(result.lambda_z), "lambda_z should be estimable"
        assert np.isfinite(result.cl_f), "CL/F should be estimable"
        assert np.isfinite(result.vz_f), "Vz/F should be estimable"
        assert result.lambda_z == pytest.approx(self._K, rel=0.10)
        assert result.cl_f == pytest.approx(self.CL, rel=0.05)
        assert result.vz_f == pytest.approx(self.V, rel=0.10)

    def test_cmax_positive(self) -> None:
        """Cmax must be strictly positive."""
        times, conc = self._generate_profile()
        engine = NCAEngine()
        result = engine.compute_subject(times, conc, dose=self.DOSE, route="oral")
        assert result.cmax > 0.0
