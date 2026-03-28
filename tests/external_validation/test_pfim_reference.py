"""
External validation for PFIM / optimal design behavior.

These checks use exact linear-Gaussian references from optimal design theory:

- For y(t) = θ0 + θ1 t with homoscedastic variance σ², the Fisher information is
  XᵀX / σ² where X = [1, t].
- For a two-point D-optimal design on [t_min, t_max], the determinant is
  proportional to (t2 - t1)², so the optimum lies at the interval bounds.

This gives a stable external-method anchor for PFIM behavior without requiring
the PFIM R package in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.design.pfim import PFIMEngine


def _make_linear_gaussian_engine() -> tuple[PFIMEngine, float]:
    sigma_var = 4.0

    class SubjectEvents:
        def __init__(self, obs_times):
            self.obs_times = np.asarray(obs_times, dtype=float)
            self.obs_dv = np.full(len(self.obs_times), np.nan)
            self.obs_cmt = np.ones(len(self.obs_times), dtype=int)
            self.obs_mdv = np.zeros(len(self.obs_times), dtype=int)

        def observation_mask(self):
            return np.ones(len(self.obs_times), dtype=bool)

    class Indiv:
        def __init__(self):
            self.subject_events = SubjectEvents([0.5, 1.5, 3.0])

        def evaluate(self, theta, eta, sigma, trans=None):
            times = np.asarray(self.subject_events.obs_times, dtype=float)
            pred = theta[0] + theta[1] * times
            return pred, self.subject_events.observation_mask(), pred

    class Model:
        trans = None

        def subject_ids(self):
            return [1]

        def individual_model(self, sid):
            return Indiv()

    class Params:
        theta = np.array([2.0, -0.3])
        omega = np.zeros((0, 0))
        sigma = np.array([[sigma_var]])

    return PFIMEngine(population_model=Model(), init_params=Params()), sigma_var


@pytest.mark.external_validation
def test_compute_fim_matches_closed_form_linear_reference() -> None:
    engine, sigma_var = _make_linear_gaussian_engine()
    times = np.array([0.5, 1.5, 3.0], dtype=float)

    observed = engine.compute_fim(times, n_subjects=5)
    design_matrix = np.column_stack([np.ones(len(times)), times])
    expected = 5.0 * (design_matrix.T @ design_matrix) / sigma_var

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.external_validation
def test_boundary_two_point_design_beats_midpoint_reference_under_d_criterion() -> None:
    engine, _sigma_var = _make_linear_gaussian_engine()
    boundary = np.array([0.5, 1.75, 3.0], dtype=float)
    midpoint = np.array([1.25, 1.75, 2.25], dtype=float)

    d_eff = engine.efficiency(boundary, midpoint, criterion="D", n_subjects=1)

    assert d_eff > 1.40, f"Boundary D-efficiency improvement too small: {d_eff:.4f}"


@pytest.mark.external_validation
def test_boundary_two_point_design_has_larger_closed_form_determinant() -> None:
    engine, _sigma_var = _make_linear_gaussian_engine()
    boundary = np.array([0.5, 1.75, 3.0], dtype=float)
    midpoint = np.array([1.25, 1.75, 2.25], dtype=float)

    fim_boundary = engine.compute_fim(boundary, n_subjects=1)
    fim_midpoint = engine.compute_fim(midpoint, n_subjects=1)

    det_boundary = float(np.linalg.det(fim_boundary))
    det_midpoint = float(np.linalg.det(fim_midpoint))
    assert det_boundary > det_midpoint, (
        f"Expected boundary design determinant to exceed midpoint design determinant, "
        f"got {det_boundary:.6f} <= {det_midpoint:.6f}"
    )


@pytest.mark.external_validation
def test_optimize_design_returns_boundary_supported_schedule_for_linear_reference() -> None:
    engine, _sigma_var = _make_linear_gaussian_engine()

    result = engine.optimize_design(
        n_samples=3,
        t_min=0.5,
        t_max=3.0,
        n_subjects=1,
        criterion="D",
        method="L-BFGS-B",
        n_starts=8,
    )

    assert result.sampling_times[0] == pytest.approx(0.5, abs=0.15)
    assert result.sampling_times[-1] == pytest.approx(3.0, abs=0.15)
    midpoint = np.array([1.25, 1.75, 2.25], dtype=float)
    det_midpoint = float(np.linalg.det(engine.compute_fim(midpoint, n_subjects=1)))
    det_optimized = float(np.linalg.det(result.information_matrix))
    assert det_optimized > det_midpoint
