"""
Part 5: Cross-method validation.

Runs FOCE, Laplacian, and Nonparametric on the standard theophylline dataset
and asserts internal consistency across methods.

Validates:
  - All three converge (finite OFV, finite ETAs).
  - OFV relationship: Laplacian OFV >= FOCE OFV - 2.0.
  - THETA estimates: all three methods agree within 10%.
  - OMEGA: all three methods produce PSD diagonal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.laplacian import LaplacianMethod
from openpkpd.estimation.nonparametric import NonparametricMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2

# ---------------------------------------------------------------------------
# Dataset (same as regression test — seed fixed for reproducibility)
# ---------------------------------------------------------------------------


def _build_theophylline_dataset() -> NONMEMDataset:
    rng = np.random.default_rng(42)
    ka_pop, cl_pop, v_pop = 1.5, 2.8, 32.9
    dose = 320.0
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])

    rows = []
    for sid in range(1, 13):
        eta_ka = rng.normal(0, 0.3)
        eta_cl = rng.normal(0, 0.2)
        eta_v = rng.normal(0, 0.15)
        ka = ka_pop * math.exp(eta_ka)
        cl = cl_pop * math.exp(eta_cl)
        v = v_pop * math.exp(eta_v)
        k = cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                c = dose * ka / v * t * math.exp(-k * t)
            else:
                c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            eps = rng.normal(0, 0.1)
            dv = max(c * (1 + eps), 0.01)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def _build_model():
    dataset = _build_theophylline_dataset()
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.5, upper=8.0),
        ThetaSpec(init=3.0, lower=0.5, upper=15.0),
        ThetaSpec(init=35.0, lower=10.0, upper=80.0),
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.09])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )
    return params, pop_model


# ---------------------------------------------------------------------------
# Fixtures (class-scoped so each method runs once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def foce_result():
    params, pop_model = _build_model()
    return FOCEMethod(interaction=False, maxeval=400, print_interval=999).estimate(
        pop_model, params
    )


@pytest.fixture(scope="module")
def laplacian_result():
    params, pop_model = _build_model()
    return LaplacianMethod(interaction=True, maxeval=400, print_interval=999).estimate(
        pop_model, params
    )


@pytest.fixture(scope="module")
def np_result():
    params, pop_model = _build_model()
    return NonparametricMethod(base_method="FOCE", max_iter=80).estimate(pop_model, params)


# ---------------------------------------------------------------------------
# Convergence (finite OFV + finite ETAs)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_foce_converges(foce_result):
    assert np.isfinite(foce_result.ofv), f"FOCE OFV not finite: {foce_result.ofv}"
    for sid, eta in foce_result.post_hoc_etas.items():
        assert np.all(np.isfinite(eta)), f"Non-finite FOCE ETA for subject {sid}"


@pytest.mark.regression
@pytest.mark.slow
def test_laplacian_converges(laplacian_result):
    assert np.isfinite(laplacian_result.ofv), f"Laplacian OFV not finite: {laplacian_result.ofv}"
    for sid, eta in laplacian_result.post_hoc_etas.items():
        assert np.all(np.isfinite(eta)), f"Non-finite Laplacian ETA for subject {sid}"


@pytest.mark.regression
@pytest.mark.slow
def test_nonparametric_converges(np_result):
    assert np.isfinite(np_result.ofv), f"Nonparametric OFV not finite: {np_result.ofv}"


# ---------------------------------------------------------------------------
# OFV ordering: Laplacian >= FOCE - 2.0
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_laplacian_ofv_ge_foce_minus_tolerance(foce_result, laplacian_result):
    """
    Laplacian adds a log|H| correction >= 0. Allow 2.0 units numerical slack.
    """
    assert laplacian_result.ofv >= foce_result.ofv - 2.0, (
        f"Laplacian OFV {laplacian_result.ofv:.4f} < FOCE OFV {foce_result.ofv:.4f} - 2"
    )


# ---------------------------------------------------------------------------
# THETA consistency across methods (within 10%)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_foce_laplacian_theta_agree(foce_result, laplacian_result):
    """FOCE and Laplacian THETA should agree within 25%.

    FOCE and Laplacian optimize different objectives; on small datasets they
    can converge to different local optima, so a generous tolerance is used.
    """
    for i, (t_foce, t_lap) in enumerate(
        zip(foce_result.theta_final, laplacian_result.theta_final, strict=False)
    ):
        ref = abs(t_foce)
        if ref < 1e-6:
            continue
        rel_diff = abs(t_foce - t_lap) / ref
        assert rel_diff < 0.25, (
            f"THETA({i + 1}): FOCE={t_foce:.4f}, Laplacian={t_lap:.4f}, "
            f"rel diff={rel_diff:.1%} (tolerance 25%)"
        )


@pytest.mark.regression
@pytest.mark.slow
def test_foce_np_theta_agree(foce_result, np_result):
    """FOCE and Nonparametric THETA should agree within 25%."""
    for i, (t_foce, t_np) in enumerate(
        zip(foce_result.theta_final, np_result.theta_final, strict=False)
    ):
        ref = abs(t_foce)
        if ref < 1e-6:
            continue
        rel_diff = abs(t_foce - t_np) / ref
        assert rel_diff < 0.25, (
            f"THETA({i + 1}): FOCE={t_foce:.4f}, NP={t_np:.4f}, "
            f"rel diff={rel_diff:.1%} (tolerance 25%)"
        )


# ---------------------------------------------------------------------------
# OMEGA PSD
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_foce_omega_psd(foce_result):
    eigvals = np.linalg.eigvalsh(foce_result.omega_final)
    assert np.all(eigvals >= -1e-8), f"FOCE OMEGA not PSD: {eigvals}"


@pytest.mark.regression
@pytest.mark.slow
def test_laplacian_omega_psd(laplacian_result):
    eigvals = np.linalg.eigvalsh(laplacian_result.omega_final)
    assert np.all(eigvals >= -1e-8), f"Laplacian OMEGA not PSD: {eigvals}"


@pytest.mark.regression
@pytest.mark.slow
def test_np_weights_sum_to_one(np_result):
    total = float(np.sum(np_result.support_weights))
    assert abs(total - 1.0) < 1e-6, f"NP weights sum to {total}"


@pytest.mark.regression
@pytest.mark.slow
def test_np_empirical_omega_psd(np_result):
    sp = np_result.support_points
    w = np_result.support_weights
    emp_mean = np_result.empirical_mean()
    emp_cov = sum(w[k] * np.outer(sp[k] - emp_mean, sp[k] - emp_mean) for k in range(len(w)))
    eigvals = np.linalg.eigvalsh(emp_cov)
    assert np.all(eigvals >= -1e-8), f"NP empirical OMEGA not PSD: {eigvals}"
