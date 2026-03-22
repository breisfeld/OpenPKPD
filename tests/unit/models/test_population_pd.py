"""Tests for PopulationPDModel."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.models.pkpd import EmaxModel, InhibEmaxModel, PDData
from openpkpd.models.population_pd import PopulationPDModel, PopulationPDResult


def _relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / max(abs(truth), 1e-12)


def _make_population_subjects(
    seed: int,
    n_subjects: int = 12,
    sigma2: float = 0.01,
) -> tuple[list[PDData], dict[str, float], float]:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.5, 24.0, 24)
    concs = 12.0 * np.exp(-0.2 * times)
    true_theta = {"E0": 1.5, "Emax": 9.0, "EC50": 3.5}
    true_omega = 0.04
    subjects: list[PDData] = []

    for sid in range(n_subjects):
        eta = rng.normal(0.0, np.sqrt(true_omega))
        params = dict(true_theta)
        params["Emax"] *= np.exp(eta)
        pred = EmaxModel().predict(
            params,
            PDData(
                subject_id=sid + 1, times=times, response=np.zeros_like(times), concentrations=concs
            ),
        )
        obs = pred + rng.normal(0.0, np.sqrt(sigma2), size=len(times))
        subjects.append(PDData(subject_id=sid + 1, times=times, response=obs, concentrations=concs))

    return subjects, true_theta, sigma2


def _make_inhibitory_population_subjects(
    seed: int,
    n_subjects: int = 12,
    sigma2: float = 0.005,
) -> tuple[list[PDData], dict[str, float], float, float]:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.5, 24.0, 24)
    concs = 8.0 * np.exp(-0.15 * times)
    true_theta = {"E0": 10.0, "Imax": 0.75, "IC50": 2.5, "gamma": 1.5}
    true_omega = 0.03
    subjects: list[PDData] = []

    for sid in range(n_subjects):
        eta = rng.normal(0.0, np.sqrt(true_omega))
        params = dict(true_theta)
        params["IC50"] *= np.exp(eta)
        pred = InhibEmaxModel().predict(
            params,
            PDData(
                subject_id=sid + 1, times=times, response=np.zeros_like(times), concentrations=concs
            ),
        )
        obs = pred + rng.normal(0.0, np.sqrt(sigma2), size=len(times))
        subjects.append(PDData(subject_id=sid + 1, times=times, response=obs, concentrations=concs))

    return subjects, true_theta, sigma2, true_omega


def test_pack_unpack_round_trip_preserves_theta_omega_and_sigma2() -> None:
    model = PopulationPDModel(
        pd_model=EmaxModel(),
        eta_params=["Emax", "EC50"],
        theta_init={"E0": 1.0, "Emax": 7.0, "EC50": 2.0},
        omega_init=np.array([[0.05, 0.01], [0.01, 0.07]]),
        sigma2=0.05,
        estimate_sigma2=True,
    )
    theta = {"E0": 1.5, "Emax": 9.0, "EC50": 3.5}
    omega = np.array([[0.04, 0.015], [0.015, 0.09]])
    sigma2 = 0.02

    theta_rt, omega_rt, sigma2_rt = model._unpack(model._pack(theta, omega, sigma2))

    assert theta_rt == pytest.approx(theta)
    np.testing.assert_allclose(omega_rt, omega, rtol=1e-12, atol=1e-12)
    assert sigma2_rt == pytest.approx(sigma2)


@pytest.mark.parametrize("seed", [1, 7])
def test_population_emax_estimation_recovers_theta_and_sigma2(seed: int) -> None:
    subjects, true_theta, true_sigma2 = _make_population_subjects(seed)
    model = PopulationPDModel(
        pd_model=EmaxModel(),
        eta_params=["Emax"],
        theta_init={"E0": 1.0, "Emax": 7.0, "EC50": 2.0},
        omega_init=np.array([[0.05]]),
        sigma2=0.05,
        estimate_sigma2=True,
        maxeval=120,
    )

    result = model.estimate(subjects)
    theta_errors = np.array(
        [_relative_error(float(result.theta[name]), truth) for name, truth in true_theta.items()]
    )

    assert isinstance(result, PopulationPDResult)
    assert result.converged
    assert np.median(theta_errors) < 0.05
    assert np.max(theta_errors) < 0.10
    assert _relative_error(float(result.sigma2), true_sigma2) < 0.30
    assert result.omega.shape == (1, 1)
    assert np.isfinite(result.omega[0, 0])
    assert result.omega[0, 0] > 0.0
    assert set(result.post_hoc_etas) == {data.subject_id for data in subjects}
    assert all(eta.shape == (1,) for eta in result.post_hoc_etas.values())


def test_population_emax_estimation_respects_fixed_sigma2() -> None:
    subjects, _, _ = _make_population_subjects(seed=3, n_subjects=8, sigma2=0.015)
    fixed_sigma2 = 0.015
    model = PopulationPDModel(
        pd_model=EmaxModel(),
        eta_params=["Emax"],
        theta_init={"E0": 1.0, "Emax": 7.0, "EC50": 2.0},
        omega_init=np.array([[0.05]]),
        sigma2=fixed_sigma2,
        estimate_sigma2=False,
        maxeval=80,
    )

    result = model.estimate(subjects)

    assert result.converged
    assert result.sigma2 == pytest.approx(fixed_sigma2)
    assert result.aic == pytest.approx(result.ofv + 2.0 * 4)


def test_population_inhibitory_emax_estimation_recovers_parameters_across_seeds() -> None:
    theta_errors: list[float] = []
    sigma2_errors: list[float] = []
    omega_errors: list[float] = []

    for seed in [1, 3, 7, 11]:
        subjects, true_theta, true_sigma2, true_omega = _make_inhibitory_population_subjects(seed)
        model = PopulationPDModel(
            pd_model=InhibEmaxModel(),
            eta_params=["IC50"],
            theta_init={"E0": 9.0, "Imax": 0.6, "IC50": 2.0, "gamma": 1.2},
            omega_init=np.array([[0.05]]),
            sigma2=0.02,
            estimate_sigma2=True,
            maxeval=120,
        )

        result = model.estimate(subjects)

        assert result.converged
        assert set(result.post_hoc_etas) == {data.subject_id for data in subjects}
        assert all(eta.shape == (1,) for eta in result.post_hoc_etas.values())
        assert result.omega.shape == (1, 1)
        assert np.isfinite(result.omega[0, 0])
        assert result.omega[0, 0] > 0.0

        theta_errors.extend(
            _relative_error(float(result.theta[name]), truth) for name, truth in true_theta.items()
        )
        sigma2_errors.append(_relative_error(float(result.sigma2), true_sigma2))
        omega_errors.append(_relative_error(float(result.omega[0, 0]), true_omega))

    theta_errors_arr = np.array(theta_errors)
    sigma2_errors_arr = np.array(sigma2_errors)
    omega_errors_arr = np.array(omega_errors)

    assert np.median(theta_errors_arr) < 0.01
    assert np.max(theta_errors_arr) < 0.09
    assert np.median(sigma2_errors_arr) < 0.20
    assert np.max(sigma2_errors_arr) < 0.25
    assert np.max(omega_errors_arr) < 0.30
