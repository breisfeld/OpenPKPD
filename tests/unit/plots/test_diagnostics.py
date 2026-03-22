"""Focused tests for diagnostics-table helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openpkpd.plots import diagnostics as diagnostics_mod


class _FakeSubjectEvents:
    def __init__(self, obs_times: list[float], obs_dv: list[float]) -> None:
        self.obs_times = np.asarray(obs_times, dtype=float)
        self.obs_dv = np.asarray(obs_dv, dtype=float)


class _LinearIndividual:
    def __init__(
        self,
        *,
        obs_times: list[float],
        obs_dv: list[float],
        base: list[float],
        obs_mask: list[bool],
        eta_coeff: list[list[float]] | None = None,
        symbolic_jacobian: list[list[float]] | None = None,
    ) -> None:
        self.subject_events = _FakeSubjectEvents(obs_times, obs_dv)
        self._base = np.asarray(base, dtype=float)
        self._obs_mask = np.asarray(obs_mask, dtype=bool)
        self._symbolic_jacobian = (
            None if symbolic_jacobian is None else np.asarray(symbolic_jacobian, dtype=float)
        )
        if eta_coeff is None:
            self._eta_coeff = np.zeros((len(base), 0), dtype=float)
        else:
            self._eta_coeff = np.asarray(eta_coeff, dtype=float)

    def evaluate(self, theta, eta, sigma, trans=2):
        eta = np.asarray(eta, dtype=float)
        if self._eta_coeff.shape[1] == 0:
            ipred = self._base.copy()
        else:
            ipred = self._base + self._eta_coeff @ eta
        return ipred, self._obs_mask.copy(), ipred.copy()

    def supports_prediction_eta_jacobian(self, trans=2):
        return self._symbolic_jacobian is not None

    def prediction_eta_jacobian(self, theta, eta, sigma, trans=2):
        if self._symbolic_jacobian is None:
            raise NotImplementedError
        return self._symbolic_jacobian.copy()


class _FakePopulationModel:
    def __init__(
        self,
        individuals: dict[int, _LinearIndividual],
        *,
        dataset_df: pd.DataFrame,
        covariate_columns: list[str] | None = None,
        trans: int = 2,
    ) -> None:
        self._individuals = individuals
        self.dataset = SimpleNamespace(df=dataset_df)
        self.covariate_columns = list(covariate_columns or [])
        self.trans = trans

    def subject_ids(self):
        return list(self._individuals)

    def individual_model(self, sid):
        return self._individuals[sid]


@pytest.mark.unit
def test_finite_diff_jacobian_matches_linear_eta_sensitivity() -> None:
    indiv = _LinearIndividual(
        obs_times=[1.0, 2.0, 3.0],
        obs_dv=[0.0, 0.0, 0.0],
        base=[5.0, 100.0, 9.0],
        obs_mask=[True, False, True],
        eta_coeff=[[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]],
    )

    jac = diagnostics_mod._finite_diff_jacobian(
        indiv,
        theta=np.array([0.0]),
        eta=np.array([0.2, -0.1]),
        sigma=np.array([[1.0]]),
        trans=2,
        h=1e-6,
    )

    np.testing.assert_allclose(jac, np.array([[1.0, 2.0], [3.0, -1.0]]), atol=1e-7)


@pytest.mark.unit
def test_cwres_subject_falls_back_to_iwres_when_factorization_fails(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(diagnostics_mod, "scipy_chol", _raise)

    cwres = diagnostics_mod._cwres_subject(
        dv=np.array([10.0, 20.0]),
        pred=np.array([9.0, 18.0]),
        ipred=np.array([8.0, 16.0]),
        eta_hat=np.array([0.5]),
        R_i=np.array([[1.0], [1.0]]),
        omega=np.array([[1.0]]),
        sigma_diag=np.array([4.0, 9.0]),
    )

    np.testing.assert_allclose(cwres, np.array([1.0, 4.0 / 3.0]))


@pytest.mark.unit
def test_compute_diagnostics_preserves_time_varying_covariates_per_observation() -> None:
    dataset_df = pd.DataFrame(
        {
            "ID": [1, 1],
            "TIME": [1.0, 2.0],
            "DV": [12.0, 18.0],
            "EVID": [0, 0],
            "MDV": [0, 0],
            "WT": [70.0, 80.0],
        }
    )
    population_model = _FakePopulationModel(
        {
            1: _LinearIndividual(
                obs_times=[1.0, 2.0],
                obs_dv=[12.0, 18.0],
                base=[10.0, 20.0],
                obs_mask=[True, True],
            )
        },
        dataset_df=dataset_df,
        covariate_columns=["WT"],
    )
    result = SimpleNamespace(
        theta_final=np.array([0.0]),
        omega_final=np.zeros((0, 0)),
        sigma_final=np.array([[0.25]]),
        post_hoc_etas={},
    )

    out = diagnostics_mod.compute_diagnostics(population_model, result)

    assert "ETA1" not in out.columns
    assert out["WT"].tolist() == [70.0, 80.0]
    np.testing.assert_allclose(out["PRED"], [10.0, 20.0])
    np.testing.assert_allclose(out["IPRED"], [10.0, 20.0])
    np.testing.assert_allclose(out["RES"], [2.0, -2.0])
    np.testing.assert_allclose(out["IRES"], [2.0, -2.0])
    np.testing.assert_allclose(out["WRES"], [0.4, -0.2])
    np.testing.assert_allclose(out["IWRES"], [0.4, -0.2])
    np.testing.assert_allclose(out["CWRES"], [0.4, -0.2])


@pytest.mark.unit
def test_compute_diagnostics_prefers_native_prediction_jacobian(monkeypatch) -> None:
    dataset_df = pd.DataFrame(
        {"ID": [1, 1], "TIME": [1.0, 2.0], "DV": [12.0, 18.0], "EVID": [0, 0], "MDV": [0, 0]}
    )
    population_model = _FakePopulationModel(
        {
            1: _LinearIndividual(
                obs_times=[1.0, 2.0],
                obs_dv=[12.0, 18.0],
                base=[10.0, 20.0],
                obs_mask=[True, True],
                eta_coeff=[[1.0], [2.0]],
                symbolic_jacobian=[[1.0], [2.0]],
            )
        },
        dataset_df=dataset_df,
    )
    result = SimpleNamespace(
        theta_final=np.array([0.0]),
        omega_final=np.array([[0.25]]),
        sigma_final=np.array([[0.25]]),
        post_hoc_etas={1: np.array([0.3])},
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("expected native prediction jacobian to be used")

    monkeypatch.setattr(diagnostics_mod, "_finite_diff_jacobian", _fail)

    out = diagnostics_mod.compute_diagnostics(population_model, result)

    assert list(out["ID"]) == [1, 1]
    np.testing.assert_allclose(out["PRED"], [10.0, 20.0])


@pytest.mark.unit
def test_compute_npde_preserves_duplicate_same_time_alignment(monkeypatch):
    """Repeated same-time observations should merge back one-to-one."""
    diag_df = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "TIME": [1.0, 1.0, 2.0],
            "DV": [10.0, 20.0, 30.0],
            "PRED": [9.5, 19.5, 29.5],
        }
    )
    npde_df = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "TIME": [1.0, 1.0, 2.0],
            "DV": [10.0, 20.0, 30.0],
            "PDE": [0.11, 0.22, 0.33],
            "NPDE": [-1.1, 1.2, 0.3],
        }
    )

    class MockSimulationEngine:
        def __init__(self, population_model, result, seed):
            self.population_model = population_model
            self.result = result
            self.seed = seed

    class MockNPDEEngine:
        def __init__(self, sim_engine):
            self.sim_engine = sim_engine

        def compute(self, n_replicates, seed, decorrelate):
            assert n_replicates == 25
            assert seed == 123
            assert decorrelate is False
            return SimpleNamespace(df=npde_df.copy())

    monkeypatch.setattr(
        diagnostics_mod,
        "compute_diagnostics",
        lambda population_model, result: diag_df.copy(),
    )

    import openpkpd.simulation.engine as sim_engine_mod
    import openpkpd.simulation.npde as npde_mod

    monkeypatch.setattr(sim_engine_mod, "SimulationEngine", MockSimulationEngine)
    monkeypatch.setattr(npde_mod, "NPDEEngine", MockNPDEEngine)

    out = diagnostics_mod.compute_npde(
        population_model=object(),
        result=object(),
        n_simulations=25,
        seed=123,
        decorrelate=False,
    )

    assert len(out) == len(diag_df)
    assert out["TIME"].tolist() == [1.0, 1.0, 2.0]
    assert out["PDE"].tolist() == [0.11, 0.22, 0.33]
    assert out["NPDE"].tolist() == [-1.1, 1.2, 0.3]
    assert "_OBSSEQ" not in out.columns
