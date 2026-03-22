"""
Unit tests for BootstrapResult CI computation.

Tests cover:
  - theta_ci shape and coverage
  - omega_diag_ci and sigma_diag_ci
  - summary() DataFrame structure
  - BootstrapResult construction from synthetic samples
  - Edge cases: single parameter, large samples
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.inference.bootstrap import BootstrapEngine, BootstrapResult

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def two_param_result() -> BootstrapResult:
    """BootstrapResult with 2 THETAs drawn from known distributions."""
    np.random.seed(42)
    samples = np.random.normal(loc=[1.0, 2.0], scale=[0.1, 0.2], size=(1000, 2))
    return BootstrapResult(
        n_boot=1000,
        n_success=1000,
        theta_samples=samples,
        omega_diag_samples=np.random.rand(1000, 1) * 0.3,
        sigma_diag_samples=np.random.rand(1000, 1) * 0.1,
    )


@pytest.fixture()
def single_param_result() -> BootstrapResult:
    """BootstrapResult with 1 THETA."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=0.5, size=(500, 1))
    return BootstrapResult(
        n_boot=500,
        n_success=500,
        theta_samples=samples,
        omega_diag_samples=rng.uniform(0.1, 0.5, size=(500, 2)),
        sigma_diag_samples=rng.uniform(0.05, 0.15, size=(500, 1)),
    )


# ── theta_ci ──────────────────────────────────────────────────────────────────


class TestThetaCI:
    """Tests for BootstrapResult.theta_ci."""

    def test_shape_two_params(self, two_param_result: BootstrapResult) -> None:
        ci = two_param_result.theta_ci
        assert ci.shape == (2, 2)

    def test_lower_less_than_upper(self, two_param_result: BootstrapResult) -> None:
        ci = two_param_result.theta_ci
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_ci_contains_true_mean_theta1(self, two_param_result: BootstrapResult) -> None:
        """95% CI for THETA(1) should contain true mean = 1.0."""
        ci = two_param_result.theta_ci
        assert ci[0, 0] < 1.0 < ci[0, 1]

    def test_ci_contains_true_mean_theta2(self, two_param_result: BootstrapResult) -> None:
        """95% CI for THETA(2) should contain true mean = 2.0."""
        ci = two_param_result.theta_ci
        assert ci[1, 0] < 2.0 < ci[1, 1]

    def test_single_param_ci_shape(self, single_param_result: BootstrapResult) -> None:
        ci = single_param_result.theta_ci
        assert ci.shape == (1, 2)

    def test_90pct_ci_narrower_than_95pct(self, two_param_result: BootstrapResult) -> None:
        """90% CI should be narrower than 95% CI."""
        result_95 = two_param_result
        result_90 = BootstrapResult(
            n_boot=result_95.n_boot,
            n_success=result_95.n_success,
            theta_samples=result_95.theta_samples,
            omega_diag_samples=result_95.omega_diag_samples,
            sigma_diag_samples=result_95.sigma_diag_samples,
            ci_level=0.90,
        )
        ci_95 = result_95.theta_ci
        ci_90 = result_90.theta_ci
        width_95 = ci_95[:, 1] - ci_95[:, 0]
        width_90 = ci_90[:, 1] - ci_90[:, 0]
        assert np.all(width_90 < width_95)

    def test_ci_percentiles_correct(self) -> None:
        """Verify percentile computation matches numpy directly."""
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(2000, 3))
        result = BootstrapResult(
            n_boot=2000,
            n_success=2000,
            theta_samples=samples,
            omega_diag_samples=np.ones((2000, 1)),
            sigma_diag_samples=np.ones((2000, 1)),
            ci_level=0.95,
        )
        ci = result.theta_ci
        expected_lower = np.percentile(samples, 2.5, axis=0)
        expected_upper = np.percentile(samples, 97.5, axis=0)
        np.testing.assert_allclose(ci[:, 0], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(ci[:, 1], expected_upper, rtol=1e-10)


# ── omega_diag_ci and sigma_diag_ci ──────────────────────────────────────────


class TestVarianceCI:
    """Tests for omega_diag_ci and sigma_diag_ci."""

    def test_omega_diag_ci_shape(self, single_param_result: BootstrapResult) -> None:
        ci = single_param_result.omega_diag_ci
        assert ci.shape == (2, 2)  # 2 omega elements

    def test_sigma_diag_ci_shape(self, single_param_result: BootstrapResult) -> None:
        ci = single_param_result.sigma_diag_ci
        assert ci.shape == (1, 2)

    def test_omega_lower_less_than_upper(self, single_param_result: BootstrapResult) -> None:
        ci = single_param_result.omega_diag_ci
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_sigma_lower_less_than_upper(self, single_param_result: BootstrapResult) -> None:
        ci = single_param_result.sigma_diag_ci
        assert np.all(ci[:, 0] < ci[:, 1])


# ── summary() ─────────────────────────────────────────────────────────────────


class TestSummaryDataFrame:
    """Tests for BootstrapResult.summary()."""

    def test_summary_returns_dataframe(self, two_param_result: BootstrapResult) -> None:
        import pandas as pd

        df = two_param_result.summary()
        assert isinstance(df, pd.DataFrame)

    def test_summary_row_count(self, two_param_result: BootstrapResult) -> None:
        """Rows = n_theta + n_omega_diag + n_sigma_diag."""
        df = two_param_result.summary()
        n_theta = two_param_result.theta_samples.shape[1]
        n_omega = two_param_result.omega_diag_samples.shape[1]
        n_sigma = two_param_result.sigma_diag_samples.shape[1]
        assert len(df) == n_theta + n_omega + n_sigma

    def test_summary_columns_present(self, two_param_result: BootstrapResult) -> None:
        df = two_param_result.summary()
        assert "parameter" in df.columns
        assert "mean" in df.columns
        assert "median" in df.columns
        assert "std" in df.columns

    def test_summary_ci_columns_present(self, two_param_result: BootstrapResult) -> None:
        df = two_param_result.summary()
        cols = df.columns.tolist()
        ci_cols = [c for c in cols if "ci" in c]
        assert len(ci_cols) >= 2

    def test_summary_theta_labels(self, two_param_result: BootstrapResult) -> None:
        df = two_param_result.summary()
        params = df["parameter"].tolist()
        assert "THETA(1)" in params
        assert "THETA(2)" in params

    def test_summary_omega_labels(self, two_param_result: BootstrapResult) -> None:
        df = two_param_result.summary()
        params = df["parameter"].tolist()
        assert "OMEGA(1,1)" in params

    def test_summary_sigma_labels(self, two_param_result: BootstrapResult) -> None:
        df = two_param_result.summary()
        params = df["parameter"].tolist()
        assert "SIGMA(1,1)" in params

    def test_summary_mean_close_to_true(self, two_param_result: BootstrapResult) -> None:
        """Bootstrap mean should be close to the true population mean."""
        df = two_param_result.summary()
        theta1_row = df[df["parameter"] == "THETA(1)"].iloc[0]
        assert abs(theta1_row["mean"] - 1.0) < 0.05


# ── BootstrapResult construction ──────────────────────────────────────────────


class TestBootstrapResultConstruction:
    """Tests for direct construction of BootstrapResult."""

    def test_basic_construction(self) -> None:
        result = BootstrapResult(
            n_boot=1000,
            n_success=1000,
            theta_samples=np.random.rand(1000, 2),
            omega_diag_samples=np.random.rand(1000, 1),
            sigma_diag_samples=np.random.rand(1000, 1),
        )
        assert result.n_boot == 1000
        assert result.n_success == 1000
        assert result.ci_level == 0.95

    def test_partial_success(self) -> None:
        """n_success < n_boot is valid."""
        result = BootstrapResult(
            n_boot=200,
            n_success=185,
            theta_samples=np.random.rand(185, 3),
            omega_diag_samples=np.random.rand(185, 2),
            sigma_diag_samples=np.random.rand(185, 1),
        )
        assert result.n_success == 185
        assert result.theta_samples.shape == (185, 3)

    def test_custom_ci_level(self) -> None:
        samples = np.random.rand(1000, 1)
        result = BootstrapResult(
            n_boot=1000,
            n_success=1000,
            theta_samples=samples,
            omega_diag_samples=np.random.rand(1000, 1),
            sigma_diag_samples=np.random.rand(1000, 1),
            ci_level=0.90,
        )
        assert result.ci_level == 0.90

    def test_repr_contains_key_info(self) -> None:
        result = BootstrapResult(
            n_boot=200,
            n_success=198,
            theta_samples=np.ones((198, 3)),
            omega_diag_samples=np.ones((198, 2)),
            sigma_diag_samples=np.ones((198, 1)),
        )
        r = repr(result)
        assert "200" in r
        assert "198" in r
        assert "95" in r


class _FakeDataset:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def subject_ids(self) -> list[int]:
        return sorted(self.df["ID"].unique().tolist())


class _FakePopulationModel:
    def __init__(
        self,
        dataset,
        pk_subroutine=None,
        params=None,
        pk_callable=None,
        error_callable=None,
        des_callable=None,
        trans=None,
        advan=None,
        covariate_columns=None,
    ) -> None:
        self.dataset = dataset
        self.pk_subroutine = pk_subroutine
        self.params = params
        self.pk_callable = pk_callable
        self.error_callable = error_callable
        self.des_callable = des_callable
        self.trans = trans
        self.advan = advan
        self.covariate_columns = list(covariate_columns or [])


def _make_bootstrap_engine(n_boot: int = 4, n_jobs: int = 1) -> BootstrapEngine:
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2, 3, 3],
            "TIME": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "DV": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        }
    )
    population_model = _FakePopulationModel(
        dataset=_FakeDataset(df),
        pk_subroutine="ADVAN1",
        params="original-params",
        pk_callable="pk",
        error_callable="err",
        des_callable="des",
        trans=2,
        advan=1,
        covariate_columns=["WT"],
    )
    return BootstrapEngine(
        population_model=population_model,
        initial_params="init-params",
        estimation_method="FOCE",
        n_boot=n_boot,
        n_jobs=n_jobs,
        seed=123,
        ci_level=0.95,
        maxeval=20,
    )


class TestBootstrapEngine:
    def test_resample_subjects_is_deterministic_and_preserves_subject_profiles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_bootstrap_engine()
        monkeypatch.setattr("openpkpd.data.dataset.NONMEMDataset", _FakeDataset)
        monkeypatch.setattr("openpkpd.model.population.PopulationModel", _FakePopulationModel)

        resampled_a = engine._resample_subjects(rng_seed=7)
        resampled_b = engine._resample_subjects(rng_seed=7)

        assert resampled_a.dataset.df.equals(resampled_b.dataset.df)
        assert resampled_a.dataset.subject_ids() == [1, 2, 3]
        assert resampled_a.params == "init-params"

        original_profiles = {
            tuple(map(tuple, group[["TIME", "DV"]].to_numpy()))
            for _, group in engine.population_model.dataset.df.groupby("ID")
        }
        for _, group in resampled_a.dataset.df.groupby("ID"):
            profile = tuple(map(tuple, group[["TIME", "DV"]].to_numpy()))
            assert profile in original_profiles

    def test_run_collects_successful_replicates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_bootstrap_engine(n_boot=3, n_jobs=1)
        outcomes = iter(
            [
                EstimationResult(
                    theta_final=np.array([1.0]),
                    omega_final=np.array([[0.2]]),
                    sigma_final=np.array([[0.05]]),
                    ofv=10.0,
                    converged=True,
                    method="FOCE",
                ),
                None,
                EstimationResult(
                    theta_final=np.array([1.5]),
                    omega_final=np.array([[0.3]]),
                    sigma_final=np.array([[0.07]]),
                    ofv=11.0,
                    converged=True,
                    method="FOCE",
                ),
            ]
        )

        monkeypatch.setattr(engine, "_fit_one", lambda seed: next(outcomes))

        result = engine.run()

        assert result.n_boot == 3
        assert result.n_success == 2
        np.testing.assert_allclose(result.theta_samples[:, 0], np.array([1.0, 1.5]))
        np.testing.assert_allclose(result.omega_diag_samples[:, 0], np.array([0.2, 0.3]))
        np.testing.assert_allclose(result.sigma_diag_samples[:, 0], np.array([0.05, 0.07]))

    def test_run_excludes_non_converged_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_bootstrap_engine(n_boot=3, n_jobs=1)
        outcomes = iter(
            [
                EstimationResult(
                    theta_final=np.array([1.0]),
                    omega_final=np.array([[0.2]]),
                    sigma_final=np.array([[0.05]]),
                    ofv=10.0,
                    converged=False,
                    method="FOCE",
                ),
                EstimationResult(
                    theta_final=np.array([1.5]),
                    omega_final=np.array([[0.3]]),
                    sigma_final=np.array([[0.07]]),
                    ofv=11.0,
                    converged=True,
                    method="FOCE",
                ),
                None,
            ]
        )

        monkeypatch.setattr(engine, "_fit_one", lambda seed: next(outcomes))

        result = engine.run()

        assert result.n_boot == 3
        assert result.n_success == 1
        np.testing.assert_allclose(result.theta_samples[:, 0], np.array([1.5]))
        np.testing.assert_allclose(result.omega_diag_samples[:, 0], np.array([0.3]))
        np.testing.assert_allclose(result.sigma_diag_samples[:, 0], np.array([0.07]))

    def test_run_raises_if_all_replicates_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_bootstrap_engine(n_boot=2, n_jobs=1)
        monkeypatch.setattr(engine, "_fit_one", lambda seed: None)

        with pytest.raises(RuntimeError, match="All bootstrap replicates failed"):
            engine.run()
