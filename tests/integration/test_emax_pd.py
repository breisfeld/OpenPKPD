"""
Integration test: 1-cmt IV PK + direct Emax PD model in $ERROR.

6 subjects, 3 dose levels, PK simulated from ADVAN1.
PD: effect via direct Emax in $ERROR block.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan1 import ADVAN1

# True PK/PD parameters
_TRUE_K = 0.15  # hr⁻¹
_TRUE_V = 10.0  # L
_TRUE_E0 = 2.0  # baseline effect
_TRUE_EMAX = 15.0  # maximum effect
_TRUE_EC50 = 8.0  # EC50 (same units as IPRED = mg/L)
_TRUE_W = 1.5  # additive residual SD
_DOSE_LEVELS = (50.0, 50.0, 100.0, 100.0, 200.0, 200.0)


def _relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / max(abs(truth), 1e-12)


def _simulate_pkpd_data(n_subj: int = 6, seed: int = 42) -> NONMEMDataset:
    """
    Simulate 1-cmt IV PK + direct Emax PD data.

    3 dose levels: 50, 100, 200 mg, repeated as needed for larger cohorts.
    Observation times: 1, 2, 4, 8, 12, 24 hr
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    obs_times = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan1 = ADVAN1()

    rows = []
    for i in range(1, n_subj + 1):
        dose = _DOSE_LEVELS[(i - 1) % len(_DOSE_LEVELS)]

        # Individual PK
        k_i = _TRUE_K * np.exp(rng.normal(0, 0.2))
        v_i = _TRUE_V * np.exp(rng.normal(0, 0.2))
        pk_params = {"K": k_i, "V": v_i}
        dose_ev = [DoseEvent(time=0.0, amount=dose, compartment=1)]
        sol = advan1.solve(pk_params, dose_ev, obs_times)
        conc = sol.ipred  # IPRED (concentration)

        # Direct Emax PD
        effect_pred = _TRUE_E0 + _TRUE_EMAX * conc / (_TRUE_EC50 + conc)
        effect_obs = effect_pred + rng.normal(0, _TRUE_W, len(obs_times))

        # Dose row
        rows.append(
            {
                "ID": i,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
            }
        )
        for j, t in enumerate(obs_times):
            rows.append(
                {
                    "ID": i,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": float(effect_obs[j]),
                    "EVID": 0,
                    "MDV": 0,
                }
            )

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


@pytest.fixture(scope="module")
def pkpd_dataset():
    return _simulate_pkpd_data()


# $ERROR block: 1-cmt IV PK, observe effect via Emax
_PK_CODE = """\
K = THETA(1)*EXP(ETA(1))
V = THETA(2)*EXP(ETA(2))
"""

_ERROR_CODE = """\
E0   = THETA(3)
EMAX = THETA(4)
EC50 = THETA(5)
W    = THETA(6)
IPRED = E0 + EMAX*F / (EC50 + F)
Y    = IPRED + W*EPS(1)
IRES = DV - IPRED
IWRES = IRES / W
"""

_FIXED_PK_CODE = """\
K = 0.15
V = 10.0
"""

_PD_ONLY_ERROR_CODE = """\
E0   = THETA(1)
EMAX = THETA(2)
EC50 = THETA(3)
W    = THETA(4)
IPRED = E0 + EMAX*F / (EC50 + F)
Y    = IPRED + W*EPS(1)
IRES = DV - IPRED
IWRES = IRES / W
"""


def _build_emax_fo_model(
    dataset: NONMEMDataset,
    *,
    maxeval: int = 800,
    problem: str = "1-cmt IV + Emax PD",
):
    return (
        ModelBuilder()
        .problem(problem)
        .dataset(dataset)
        .subroutines(advan=1, trans=2)
        .pk(_PK_CODE)
        .error(_ERROR_CODE)
        .theta(
            [
                (0.01, 0.15, 5.0),
                (1.0, 10.0, 100.0),
                (0.0, 2.0, 20.0),
                (1.0, 15.0, 100.0),
                (0.1, 8.0, 100.0),
                (0.1, 1.5, 20.0),
            ]
        )
        .omega([0.3, 0.3])
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=maxeval)
        .build()
    )


def _build_fixed_pk_emax_pd_fo_model(
    dataset: NONMEMDataset,
    *,
    maxeval: int = 1200,
    problem: str = "1-cmt IV fixed PK + Emax PD",
):
    return (
        ModelBuilder()
        .problem(problem)
        .dataset(dataset)
        .subroutines(advan=1, trans=2)
        .pk(_FIXED_PK_CODE)
        .error(_PD_ONLY_ERROR_CODE)
        .theta(
            [
                (0.0, 5.0, 20.0),
                (1.0, 5.0, 100.0),
                (0.1, 20.0, 100.0),
                (0.1, 3.0, 20.0),
            ]
        )
        .omega([1e-6, 1e-6], fixed=True)
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=maxeval)
        .build()
    )


@pytest.mark.integration
def test_fo_runs(pkpd_dataset):
    """FO estimation on 1-cmt IV + Emax PD should complete without error."""
    result = _build_emax_fo_model(pkpd_dataset).fit()
    assert np.isfinite(result.ofv), f"OFV not finite: {result.ofv}"
    assert result.ofv < 1e9


@pytest.mark.integration
def test_parameters_positive(pkpd_dataset):
    """All estimated PK/PD parameters should be positive."""
    result = _build_emax_fo_model(pkpd_dataset, problem="1-cmt IV + Emax PD params check").fit()
    # K, V, E0 (>=0), EMAX, EC50, W must all be > 0
    assert result.theta_final[0] > 0, "K <= 0"
    assert result.theta_final[1] > 0, "V <= 0"
    assert result.theta_final[3] > 0, "Emax <= 0"
    assert result.theta_final[4] > 0, "EC50 <= 0"
    assert result.theta_final[5] > 0, "W <= 0"


@pytest.mark.integration
def test_simulate_pkpd_data_supports_larger_subject_counts():
    """The PK/PD simulator should support larger cohorts without indexing errors."""
    dataset = _simulate_pkpd_data(n_subj=24, seed=7)
    dose_rows = dataset.df[dataset.df["EVID"] == 1]
    dose_counts = dose_rows["AMT"].value_counts().to_dict()

    assert dataset.n_subjects() == 24
    assert len(dataset.df) == 24 * 7
    assert dose_counts == {50.0: 8, 100.0: 8, 200.0: 8}


@pytest.mark.integration
def test_simulated_effect_increases_with_dose_across_seeds():
    """Simulated early effects should preserve the expected dose-response ordering."""
    for seed in [1, 7, 42]:
        dataset = _simulate_pkpd_data(n_subj=24, seed=seed)
        dose_rows = dataset.df[dataset.df["EVID"] == 1][["ID", "AMT"]].rename(
            columns={"AMT": "DOSE"}
        )
        obs_rows = dataset.df[
            (dataset.df["EVID"] == 0) & (dataset.df["TIME"].isin([1.0, 2.0, 4.0, 8.0]))
        ]
        means = (
            obs_rows.merge(dose_rows, on="ID", how="left").groupby("DOSE")["DV"].mean().to_dict()
        )

        assert means[50.0] < means[100.0] < means[200.0], (
            f"Dose-response ordering failed for seed={seed}: {means}"
        )


@pytest.mark.integration
def test_pk_recovery_stays_reasonable_across_larger_scenarios():
    """FO PK recovery should remain stable across larger Emax scenarios."""
    scenarios = [(12, 1), (12, 7), (12, 42), (24, 1), (24, 7), (24, 42)]
    k_errors = []
    v_errors = []

    for n_subj, seed in scenarios:
        dataset = _simulate_pkpd_data(n_subj=n_subj, seed=seed)
        result = _build_emax_fo_model(
            dataset,
            problem=f"1-cmt IV + Emax PD recovery seed={seed} n={n_subj}",
        ).fit()
        k_est, v_est = result.theta_final[0], result.theta_final[1]

        assert np.isfinite(result.ofv), f"OFV not finite for seed={seed}, n={n_subj}"
        assert k_est > 0 and np.isfinite(k_est), f"Bad K for seed={seed}, n={n_subj}: {k_est}"
        assert v_est > 0 and np.isfinite(v_est), f"Bad V for seed={seed}, n={n_subj}: {v_est}"

        k_errors.append(_relative_error(k_est, _TRUE_K))
        v_errors.append(_relative_error(v_est, _TRUE_V))

    assert float(np.median(k_errors)) <= 0.10, f"Median K rel err too high: {k_errors}"
    assert max(k_errors) <= 0.18, f"Worst-case K rel err too high: {k_errors}"
    assert float(np.median(v_errors)) <= 0.10, f"Median V rel err too high: {v_errors}"
    assert max(v_errors) <= 0.23, f"Worst-case V rel err too high: {v_errors}"


@pytest.mark.integration
def test_pd_recovery_stays_reasonable_when_pk_fixed_across_seeds():
    """With PK fixed, FO should recover identifiable PD parameters across seeds."""
    e0_errors = []
    emax_errors = []
    ec50_errors = []
    w_errors = []

    for seed in [1, 7, 42]:
        dataset = _simulate_pkpd_data(n_subj=48, seed=seed)
        result = _build_fixed_pk_emax_pd_fo_model(
            dataset,
            problem=f"1-cmt IV fixed PK + Emax PD recovery seed={seed}",
        ).fit()
        e0_est, emax_est, ec50_est, w_est = result.theta_final

        assert np.isfinite(result.ofv), f"OFV not finite for seed={seed}"
        assert e0_est >= 0 and np.isfinite(e0_est), f"Bad E0 for seed={seed}: {e0_est}"
        assert emax_est > 0 and np.isfinite(emax_est), f"Bad EMAX for seed={seed}: {emax_est}"
        assert ec50_est > 0 and np.isfinite(ec50_est), f"Bad EC50 for seed={seed}: {ec50_est}"
        assert w_est > 0 and np.isfinite(w_est), f"Bad W for seed={seed}: {w_est}"

        e0_errors.append(_relative_error(e0_est, _TRUE_E0))
        emax_errors.append(_relative_error(emax_est, _TRUE_EMAX))
        ec50_errors.append(_relative_error(ec50_est, _TRUE_EC50))
        w_errors.append(_relative_error(w_est, _TRUE_W))

    assert float(np.median(e0_errors)) <= 0.06, f"Median E0 rel err too high: {e0_errors}"
    assert max(e0_errors) <= 0.12, f"Worst-case E0 rel err too high: {e0_errors}"
    assert float(np.median(emax_errors)) <= 0.02, f"Median EMAX rel err too high: {emax_errors}"
    assert max(emax_errors) <= 0.15, f"Worst-case EMAX rel err too high: {emax_errors}"
    assert float(np.median(ec50_errors)) <= 0.14, f"Median EC50 rel err too high: {ec50_errors}"
    assert max(ec50_errors) <= 0.25, f"Worst-case EC50 rel err too high: {ec50_errors}"
    assert float(np.median(w_errors)) <= 0.05, f"Median W rel err too high: {w_errors}"
    assert max(w_errors) <= 0.09, f"Worst-case W rel err too high: {w_errors}"


@pytest.mark.integration
def test_emax_reasonable(pkpd_dataset):
    """Estimated Emax should be within 5x of the true value."""
    result = _build_emax_fo_model(pkpd_dataset, problem="1-cmt IV + Emax PD Emax check").fit()
    emax_est = result.theta_final[3]
    # Within a factor of 5 of the true Emax=15
    assert 3.0 < emax_est < 75.0, f"Emax estimate unreasonable: {emax_est:.2f}"
