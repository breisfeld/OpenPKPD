"""Empirical BAYES(Laplace) benchmark on NONMEM run 402."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpkpd.estimation.bayes import BAYESMethod
from openpkpd.model.problem import Problem
from openpkpd.parser.control_stream import ControlStream
from tests._release_validation import require_release_fixture


TEMP_DIR = Path(__file__).parent.parent.parent / "temp" / "nonmem"
REF_PATH = Path(__file__).parent / "reference" / "nonmem_402_focei.json"


def _load_ref() -> dict:
    ref_path = require_release_fixture(REF_PATH, kind="Reference file")
    return json.loads(ref_path.read_text())


def _build_problem():
    ctl_path = require_release_fixture(TEMP_DIR / "402.ctl", kind="NONMEM control stream")
    data_path = require_release_fixture(TEMP_DIR / "402.csv", kind="NONMEM dataset")
    control_stream = ControlStream.from_file(str(ctl_path))
    return Problem.from_control_stream(control_stream, dataset_path=str(data_path))


@pytest.mark.external_validation
@pytest.mark.slow
class TestBayesLaplaceNonmem402Diagnostics:
    """Empirical BAYES(Laplace) benchmark against the NONMEM 402 parameter basin."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        problem = _build_problem()
        return BAYESMethod(
            backend="laplace",
            n_samples=50,
            seed=42,
            prior_sd_theta=1e8,
            maxeval=5,
        ).estimate(problem.population_model, problem.population_model.params)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref()

    def test_backend_and_sample_shape(self, fit_result):
        theta_samples = fit_result.posterior_samples.get("theta")
        assert fit_result.backend_used == "laplace"
        assert fit_result.method == "BAYES(Laplace)"
        assert theta_samples is not None
        assert theta_samples.shape == (50, 4)

    def test_theta_stays_in_nonmem_like_basin(self, fit_result, ref):
        observed = [float(value) for value in fit_result.theta_final]
        expected = [
            float(ref["theta"]["V1"]),
            float(ref["theta"]["CL"]),
            float(ref["theta"]["V2"]),
            float(ref["theta"]["Q"]),
        ]
        tolerances = [0.08, 0.10, 0.10, 0.10]
        for name, obs, exp, tol in zip(("V1", "CL", "V2", "Q"), observed, expected, tolerances):
            rel_err = abs(obs - exp) / exp
            assert rel_err < tol, (
                f"{name}={obs:.4f} vs NONMEM={exp:.4f} "
                f"(rel_err={rel_err:.1%}, tolerance={tol:.0%})"
            )

    def test_posterior_mean_tracks_map_estimate(self, fit_result):
        posterior_mean = fit_result.posterior_samples["theta"].mean(axis=0)
        tolerances = [0.12, 0.08, 0.90, 0.15]
        for name, obs, exp, tol in zip(
            ("V1", "CL", "V2", "Q"), posterior_mean, fit_result.theta_final, tolerances
        ):
            assert abs(float(obs) - float(exp)) < tol, (
                f"{name} posterior mean {obs:.4f} drifted from MAP {exp:.4f} by more than {tol:.2f}"
            )

    def test_sigma_and_posterior_scales_are_finite(self, fit_result):
        sigma = float(fit_result.sigma_final[0, 0])
        posterior_sd = fit_result.posterior_samples["theta"].std(axis=0, ddof=1)
        assert np.isfinite(sigma)
        assert sigma > 0.0
        assert np.isfinite(fit_result.ofv)
        assert np.all(np.isfinite(posterior_sd))
        assert np.all(posterior_sd > 0.0)
