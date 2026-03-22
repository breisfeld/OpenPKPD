"""
Integration test: 2-compartment IV (ADVAN3) and oral (ADVAN4) models.

8 subjects simulated from ADVAN3, then fitted with FO.
Also tests ADVAN4 (2-cmt oral) fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan4 import ADVAN4

# True PK parameters for simulation
_TRUE_K = 0.2
_TRUE_K12 = 0.08
_TRUE_K21 = 0.04
_TRUE_V1 = 8.0
_TRUE_DOSE = 100.0
_TRUE_CL = _TRUE_K * _TRUE_V1


# ---------------------------------------------------------------------------
# Simulate dataset from ADVAN3
# ---------------------------------------------------------------------------


def _simulate_2cmt_data(n_subj: int = 8, seed: int = 0) -> NONMEMDataset:
    """Simulate a 2-compartment IV dataset with proportional error."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan3 = ADVAN3()

    rows = []
    for i in range(1, n_subj + 1):
        # Individual PK params with log-normal variability
        k = _TRUE_K * np.exp(rng.normal(0, 0.2))
        k12 = _TRUE_K12 * np.exp(rng.normal(0, 0.15))
        k21 = _TRUE_K21 * np.exp(rng.normal(0, 0.15))
        v1 = _TRUE_V1 * np.exp(rng.normal(0, 0.2))

        pk_params = {"K": k, "K12": k12, "K21": k21, "V1": v1}
        dose_ev = [DoseEvent(time=0.0, amount=_TRUE_DOSE, compartment=1)]
        sol = advan3.solve(pk_params, dose_ev, obs_times)
        ipred = sol.ipred

        # Proportional error
        eps = rng.normal(0, 0.1, len(obs_times))
        dv = np.maximum(ipred * (1 + eps), 0)

        # Dose row
        rows.append(
            {
                "ID": i,
                "TIME": 0.0,
                "AMT": _TRUE_DOSE,
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
                    "DV": float(dv[j]),
                    "EVID": 0,
                    "MDV": 0,
                }
            )

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


def _simulate_2cmt_oral_data(n_subj: int = 6, seed: int = 1) -> NONMEMDataset:
    """Simulate a 2-compartment oral dataset (ADVAN4)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan4 = ADVAN4()

    rows = []
    for i in range(1, n_subj + 1):
        ka = 1.2 * np.exp(rng.normal(0, 0.3))
        k = _TRUE_K * np.exp(rng.normal(0, 0.2))
        k12 = _TRUE_K12 * np.exp(rng.normal(0, 0.15))
        k21 = _TRUE_K21 * np.exp(rng.normal(0, 0.15))
        v2 = _TRUE_V1 * np.exp(rng.normal(0, 0.2))

        pk_params = {"KA": ka, "K": k, "K12": k12, "K21": k21, "V2": v2}
        dose_ev = [DoseEvent(time=0.0, amount=_TRUE_DOSE, compartment=1)]
        sol = advan4.solve(pk_params, dose_ev, obs_times)
        ipred = sol.ipred

        eps = rng.normal(0, 0.1, len(obs_times))
        dv = np.maximum(ipred * (1 + eps), 0)

        rows.append(
            {
                "ID": i,
                "TIME": 0.0,
                "AMT": _TRUE_DOSE,
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
                    "DV": float(dv[j]),
                    "EVID": 0,
                    "MDV": 0,
                }
            )

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def advan3_dataset():
    return _simulate_2cmt_data()


@pytest.fixture(scope="module")
def advan4_dataset():
    return _simulate_2cmt_oral_data()


def _build_advan3_fo_model(
    dataset: NONMEMDataset,
    *,
    maxeval: int = 600,
    problem: str = "2-cmt IV FO",
):
    return (
        ModelBuilder()
        .problem(problem)
        .dataset(dataset)
        .subroutines(advan=3, trans=1)
        .pk("""
CL = THETA(1)*EXP(ETA(1))
V1 = THETA(2)*EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
K  = CL/V1
K12 = Q/V1
K21 = Q/V2
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.6, 30), (1.0, 8.0, 100), (0.1, 0.64, 10), (1.0, 8.0, 100)])
        .omega([0.4, 0.4])
        .sigma(0.05)
        .estimation(method="FO", maxeval=maxeval)
        .build()
    )


def _relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / max(abs(truth), 1e-12)


# ---------------------------------------------------------------------------
# ADVAN3 tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_advan3_fo_runs(advan3_dataset):
    """FO estimation on ADVAN3 2-cmt IV should run without errors."""
    result = _build_advan3_fo_model(advan3_dataset).fit()
    assert np.isfinite(result.ofv)
    assert result.ofv < 1e9
    assert len(result.theta_final) == 4


@pytest.mark.integration
def test_advan3_all_ipred_positive(advan3_dataset):
    """All IPRED values from ADVAN3 at converged params should be positive."""
    advan3 = ADVAN3()
    pk_params = {"K": _TRUE_K, "K12": _TRUE_K12, "K21": _TRUE_K21, "V1": _TRUE_V1}
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    dose_ev = [DoseEvent(time=0.0, amount=_TRUE_DOSE, compartment=1)]
    sol = advan3.solve(pk_params, dose_ev, obs_times)
    assert np.all(sol.ipred > 0)


@pytest.mark.integration
def test_advan3_k_recovery_stays_reasonable_across_scenarios():
    """FO recovery should remain quantitatively reasonable across seeds and N."""
    scenarios = [(8, 1), (8, 42), (24, 1), (24, 42)]
    k_errors = []
    cl_errors = []
    v1_errors = []

    for n_subj, seed in scenarios:
        dataset = _simulate_2cmt_data(n_subj=n_subj, seed=seed)
        result = _build_advan3_fo_model(
            dataset,
            maxeval=800,
            problem=f"2-cmt IV FO recovery seed={seed} n={n_subj}",
        ).fit()
        cl_est, v1_est = result.theta_final[0], result.theta_final[1]
        k_est = cl_est / v1_est

        assert np.isfinite(result.ofv), f"OFV not finite for seed={seed}, n={n_subj}"
        assert np.isfinite(k_est), f"K estimate not finite for seed={seed}, n={n_subj}"
        assert k_est > 0, f"K estimate non-positive for seed={seed}, n={n_subj}: {k_est:.3f}"

        k_errors.append(_relative_error(k_est, _TRUE_K))
        cl_errors.append(_relative_error(cl_est, _TRUE_CL))
        v1_errors.append(_relative_error(v1_est, _TRUE_V1))

    assert float(np.median(k_errors)) <= 0.15, f"Median K rel err too high: {k_errors}"
    assert max(k_errors) <= 0.25, f"Worst-case K rel err too high: {k_errors}"
    assert float(np.median(cl_errors)) <= 0.12, f"Median CL rel err too high: {cl_errors}"
    assert max(v1_errors) <= 0.15, f"Worst-case V1 rel err too high: {v1_errors}"


# ---------------------------------------------------------------------------
# ADVAN4 tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_advan4_fo_runs(advan4_dataset):
    """FO estimation on ADVAN4 2-cmt oral should run."""
    built = (
        ModelBuilder()
        .problem("2-cmt oral FO")
        .dataset(advan4_dataset)
        .subroutines(advan=4, trans=1)
        .pk("""
KA  = THETA(1)*EXP(ETA(1))
CL  = THETA(2)*EXP(ETA(2))
V2  = THETA(3)
Q   = THETA(4)
V3  = THETA(5)
K   = CL/V2
K12 = Q/V2
K21 = Q/V3
""")
        .error("Y = F*(1 + EPS(1))")
        .theta(
            [(0.01, 1.2, 20), (0.01, 1.6, 30), (1.0, 8.0, 100), (0.1, 0.64, 10), (1.0, 8.0, 100)]
        )
        .omega([0.4, 0.4])
        .sigma(0.05)
        .estimation(method="FO", maxeval=600)
        .build()
    )
    result = built.fit()
    assert np.isfinite(result.ofv)
    assert result.ofv < 1e9


@pytest.mark.integration
def test_advan4_absorption_peak_at_intermediate_time(advan4_dataset):
    """ADVAN4 absorption peak should be at an intermediate time (not t=0 or last)."""
    advan4 = ADVAN4()
    pk_params = {"KA": 1.2, "K": _TRUE_K, "K12": _TRUE_K12, "K21": _TRUE_K21, "V2": _TRUE_V1}
    obs_times = np.linspace(0.1, 24.0, 50)
    dose_ev = [DoseEvent(time=0.0, amount=_TRUE_DOSE, compartment=1)]
    sol = advan4.solve(pk_params, dose_ev, obs_times)
    peak_idx = int(np.argmax(sol.ipred))
    # Peak should not be at the first or last point
    assert 1 <= peak_idx <= len(obs_times) - 2, f"Peak at boundary index {peak_idx}"
