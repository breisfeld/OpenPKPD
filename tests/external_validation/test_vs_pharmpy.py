"""
External validation: openpkpd diagnostics vs Pharmpy reference.

Two validation layers
---------------------
A. Formula validation (no fitting required):
   Given the same individual ETAs and OMEGA matrix, both openpkpd and
   Pharmpy should compute identical ETA shrinkage values.  This tests
   that openpkpd implements the Karlsson & Sheiner (1993) definition
   correctly as does the pharmacometrics community standard.

B. Phenobarbital population fit (requires pharmpy, slow):
   Pharmpy bundles a pre-computed NONMEM reference fit for its
   `pheno` example (phenobarbital in 59 neonates).  We compare the
   ETA shrinkage computed by openpkpd against Pharmpy's output on the
   same ETAs, and we use the same empirical dataset as a benchmark
   surface for FOCEI and nonparametric estimation.

Shrinkage definition (Karlsson & Sheiner 1993 / NONMEM):
    shrinkage_k = 1 - SD(η̂_ik, i=1..N) / √ω_kk

Pharmpy uses the same definition in ``calculate_eta_shrinkage``.

References
----------
Karlsson MO & Sheiner LB. (1993). The importance of modeling interoccasion
  variability in population pharmacokinetic analyses. J Pharmacokinet
  Biopharm, 21(6):735-750.
Pharmpy documentation: https://pharmpy.github.io/latest/
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Layer A: Shrinkage formula cross-check (no pharmpy required)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestShrinkageFormula:
    """
    Validate openpkpd shrinkage formula against the Karlsson & Sheiner (1993)
    definition — the community standard implemented in NONMEM and Pharmpy.

    shrinkage_k = 1 - SD(η̂_k) / √ω_kk

    These tests use hand-crafted ETA/OMEGA data so they can be verified
    against the formula without any external software.
    """

    def _compute_openpkpd_shrinkage(self, post_hoc_etas: dict, omega: np.ndarray) -> np.ndarray:
        """Compute shrinkage using openpkpd's EstimationResult.compute_shrinkage."""
        from openpkpd.estimation.base import EstimationResult

        res = EstimationResult(
            theta_final=np.array([]),
            omega_final=omega,
            sigma_final=np.eye(1),
            ofv=0.0,
            converged=True,
            post_hoc_etas=post_hoc_etas,
            ofv_history=[],
            n_function_evals=0,
            elapsed_time=0.0,
            method="FOCE",
            message="",
        )
        res.compute_shrinkage()
        return res.eta_shrinkage

    def _shrinkage_formula(self, etas: np.ndarray, omega_diag: np.ndarray) -> np.ndarray:
        """Reference implementation of Karlsson & Sheiner (1993) shrinkage."""
        n_eta = etas.shape[1]
        shrinkage = np.zeros(n_eta)
        for k in range(n_eta):
            sd_k = float(np.std(etas[:, k], ddof=1))
            omega_kk = float(omega_diag[k])
            if omega_kk > 0:
                shrinkage[k] = 1.0 - sd_k / math.sqrt(omega_kk)
        return shrinkage

    def test_zero_eta_gives_full_shrinkage(self):
        """All ETAs = 0 → SD=0 → shrinkage = 1 (full shrinkage)."""
        n_subj = 10
        etas = {i: np.zeros(2) for i in range(n_subj)}
        omega = np.diag([0.3, 0.1])
        sh = self._compute_openpkpd_shrinkage(etas, omega)
        np.testing.assert_allclose(
            sh, [1.0, 1.0], atol=1e-12, err_msg="Full shrinkage expected when all ETAs=0"
        )

    def test_no_shrinkage_when_eta_matches_omega(self):
        """SD(η̂_k) = √ω_kk → shrinkage = 0 (no shrinkage).

        When posterior ETAs are drawn from N(0, Ω), their empirical SD
        approaches √ω_kk as N→∞, giving zero shrinkage.
        """
        rng = np.random.default_rng(42)
        omega_diag = np.array([0.4, 0.09])
        omega = np.diag(omega_diag)
        n_subj = 500  # large sample → empirical SD ≈ true SD
        etas_array = rng.multivariate_normal(np.zeros(2), omega, size=n_subj)
        etas = {i: etas_array[i] for i in range(n_subj)}
        sh = self._compute_openpkpd_shrinkage(etas, omega)
        # SD(sample) ≈ √ω_kk → shrinkage ≈ 0. Allow ±5% for sampling error.
        assert abs(sh[0]) < 0.05, f"ETA1 shrinkage={sh[0]:.4f} should be ≈0 when SD(η)=√ω"
        assert abs(sh[1]) < 0.05, f"ETA2 shrinkage={sh[1]:.4f} should be ≈0 when SD(η)=√ω"

    def test_shrinkage_matches_formula(self):
        """openpkpd shrinkage matches hand-computed Karlsson & Sheiner formula."""
        rng = np.random.default_rng(99)
        n_subj = 20
        omega_diag = np.array([0.5, 0.2, 0.05])
        omega = np.diag(omega_diag)
        # Shrunk ETAs: smaller SD than √ω
        etas_array = rng.multivariate_normal(np.zeros(3), omega * 0.3, size=n_subj)
        etas = {i: etas_array[i] for i in range(n_subj)}

        sh_openpkpd = self._compute_openpkpd_shrinkage(etas, omega)
        sh_ref = self._shrinkage_formula(etas_array, omega_diag)

        np.testing.assert_allclose(
            sh_openpkpd,
            sh_ref,
            atol=1e-12,
            err_msg="openpkpd shrinkage must match Karlsson & Sheiner formula exactly",
        )

    def test_single_eta_known_value(self):
        """Hand-computed shrinkage example with known result.

        5 subjects: η = {0.1, 0.2, -0.3, 0.0, 0.2}, ω = 0.25
        SD(η) = std([0.1, 0.2, -0.3, 0.0, 0.2], ddof=1)
              = std([0.1, 0.2, -0.3, 0.0, 0.2])
        √ω   = 0.5
        shrinkage = 1 - SD/0.5
        """
        eta_vals = np.array([0.1, 0.2, -0.3, 0.0, 0.2])
        sd_eta = float(np.std(eta_vals, ddof=1))
        omega_kk = 0.25
        expected = 1.0 - sd_eta / math.sqrt(omega_kk)

        etas = {i: np.array([v]) for i, v in enumerate(eta_vals)}
        omega = np.array([[omega_kk]])
        sh = self._compute_openpkpd_shrinkage(etas, omega)

        assert abs(sh[0] - expected) < 1e-12, f"Shrinkage={sh[0]:.8f}, expected={expected:.8f}"

    def test_high_shrinkage_warning_emitted(self):
        """EstimationResult should warn when shrinkage > 30%."""
        import warnings as _warnings

        from openpkpd.estimation.base import EstimationResult

        # All ETAs at 20% of SD(omega) → high shrinkage
        rng = np.random.default_rng(7)
        n_subj = 12
        omega_kk = 0.36  # SD = 0.6
        etas = {i: np.array([rng.normal(0, 0.6 * 0.1)]) for i in range(n_subj)}

        res = EstimationResult(
            theta_final=np.array([]),
            omega_final=np.array([[omega_kk]]),
            sigma_final=np.eye(1),
            ofv=0.0,
            converged=True,
            post_hoc_etas=etas,
            ofv_history=[],
            n_function_evals=0,
            elapsed_time=0.0,
            method="FOCE",
            message="",
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            res.compute_shrinkage()

        assert res.eta_shrinkage[0] > 0.30, "Test setup: shrinkage should be >30%"
        assert len(caught) > 0, "Expected a UserWarning for >30% shrinkage"
        assert any("shrinkage" in str(w.message).lower() for w in caught)


_PHARMPY_INSTALLED = importlib.util.find_spec("pharmpy") is not None


def _pharmpy_example_api_available() -> bool:
    if not _PHARMPY_INSTALLED:
        return False
    return (
        importlib.util.find_spec("pharmpy.modeling") is not None
        or importlib.util.find_spec("pharmpy.tools") is not None
    )

_pharmpy_required = pytest.mark.skipif(
    not _pharmpy_example_api_available(),
    reason=(
        "Pharmpy example APIs not available; need a Pharmpy build exposing "
        "pharmpy.modeling or pharmpy.tools"
    ),
)


def _load_pharmpy_pheno():
    """Load Pharmpy's bundled phenobarbital example model and reference fit."""
    try:
        from pharmpy.modeling import load_example_model  # type: ignore[attr-defined]
    except ImportError:
        from pharmpy.tools import load_example_model  # type: ignore[attr-defined]

    try:
        from pharmpy.modeling import load_example_modelfit_results  # type: ignore[attr-defined]
    except ImportError:
        from pharmpy.tools import load_example_modelfit_results  # type: ignore[attr-defined]

    return load_example_model("pheno"), load_example_modelfit_results("pheno")


def _extract_pharmpy_iiv_diag(results) -> np.ndarray:
    """Return diagonal Ω estimates aligned with the ETA columns."""
    estimates = dict(results.parameter_estimates.items())
    eta_cols = list(results.individual_estimates.columns)
    omega_diag = np.zeros(len(eta_cols), dtype=float)

    for idx, col in enumerate(eta_cols, start=1):
        suffix = col.split("_", 1)[-1]
        candidates = [
            f"IIV_{suffix}",
            f"OMEGA({idx},{idx})",
            f"OMEGA({idx}, {idx})",
        ]
        for name in candidates:
            if name in estimates:
                omega_diag[idx - 1] = float(estimates[name])
                break

    return omega_diag


def _build_openpkpd_pheno_model(method: str, **estimation_kwargs):
    """Build the shared phenobarbital benchmark model on Pharmpy's pheno dataset."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    model, _results = _load_pharmpy_pheno()
    dataset = NONMEMDataset.from_dataframe(model.dataset.copy())

    return (
        ModelBuilder()
        .problem(f"Pharmpy pheno {method} cross-check")
        .dataset(dataset)
        .covariates(["WGT", "APGR"])
        .subroutines(advan=1, trans=2)
        .pk(
            """
            TVCL = THETA(1) * WGT
            TVV = THETA(2) * WGT
            IF (APGR .LT. 5) TVV = TVV * (1 + THETA(3))
            CL = TVCL * EXP(ETA(1))
            V = TVV * EXP(ETA(2))
            S1 = V
            """
        )
        .error("Y = F + F * EPS(1)")
        .theta(
            [
                (0.0, 0.00469307, 0.05),
                (0.0, 1.00916, 10.0),
                (-0.99, 0.1, 5.0),
            ]
        )
        .omega([0.0309626, 0.031128])
        .sigma([0.0130865])
        .estimation(method=method, **estimation_kwargs)
        .build()
    )


@_pharmpy_required
@pytest.mark.external_validation
@pytest.mark.slow
class TestVsPharmpy:
    """
    Validate openpkpd diagnostics against Pharmpy's pheno example.

    Uses pharmpy.modeling.load_example_modelfit_results("pheno") to get
    individual ETAs and OMEGA from a NONMEM fit on the phenobarbital dataset
    (Sheiner & Beal, 1980: 59 neonates, 1-cmt IV, covariates APGR + WGT).

    Requires: pharmpy >= 0.29 (for load_example_modelfit_results API).
    """

    @pytest.fixture(scope="class")
    def pharmpy_results(self):
        return _load_pharmpy_pheno()

    @pytest.fixture(scope="class")
    def pharmpy_shrinkage(self, pharmpy_results):
        from pharmpy.modeling import calculate_eta_shrinkage

        model, results = pharmpy_results
        return calculate_eta_shrinkage(
            model,
            results.parameter_estimates,
            results.individual_estimates,
            sd=True,
        )

    @pytest.fixture(scope="class")
    def openpkpd_shrinkage(self, pharmpy_results):
        """Compute openpkpd shrinkage from pharmpy's post-hoc ETAs and OMEGA."""
        from openpkpd.estimation.base import EstimationResult

        _model, results = pharmpy_results

        # Extract ETAs from pharmpy individual estimates
        ind_ests = results.individual_estimates  # DataFrame: subjects × ETAs
        eta_dict = {int(sid): np.array(row.values, dtype=float) for sid, row in ind_ests.iterrows()}

        # Extract OMEGA from pharmpy results
        n_eta = ind_ests.shape[1]
        omega = np.diag(_extract_pharmpy_iiv_diag(results)[:n_eta])

        res = EstimationResult(
            theta_final=np.array([]),
            omega_final=omega,
            sigma_final=np.eye(1),
            ofv=0.0,
            converged=True,
            post_hoc_etas=eta_dict,
            ofv_history=[],
            n_function_evals=0,
            elapsed_time=0.0,
            method="FOCE",
            message="",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.compute_shrinkage()
        return res.eta_shrinkage

    def test_shrinkage_agrees_with_pharmpy(self, pharmpy_shrinkage, openpkpd_shrinkage):
        """openpkpd and Pharmpy shrinkage should be numerically identical.

        Both implement shrinkage_k = 1 - SD(η̂_k)/√ω_kk.  Any difference
        indicates a formula discrepancy.
        """
        pharmpy_values = np.asarray(pharmpy_shrinkage, dtype=float)
        np.testing.assert_allclose(
            openpkpd_shrinkage,
            pharmpy_values,
            atol=1e-6,
            err_msg=(
                f"Shrinkage mismatch: openpkpd={openpkpd_shrinkage}, pharmpy={pharmpy_values}"
            ),
        )

    def test_pharmpy_pheno_eta_count(self, pharmpy_results):
        """Pheno model should have exactly 2 random effects (ETA1, ETA2)."""
        _model, results = pharmpy_results
        n_etas = results.individual_estimates.shape[1]
        assert n_etas == 2, f"Expected 2 ETAs in pheno model, got {n_etas}"

    def test_pharmpy_pheno_subject_count(self, pharmpy_results):
        """Pheno dataset should have 59 subjects."""
        _model, results = pharmpy_results
        n_subjects = results.individual_estimates.shape[0]
        assert n_subjects == 59, f"Expected 59 subjects, got {n_subjects}"

    def test_openpkpd_shrinkage_is_bounded(self, openpkpd_shrinkage):
        """Shrinkage must remain in a plausible bounded range on the pheno reference."""
        assert np.all(np.isfinite(openpkpd_shrinkage))
        assert np.all(openpkpd_shrinkage > -0.5)
        assert np.all(openpkpd_shrinkage < 1.0)

    def test_pharmpy_shrinkage_eta_order_is_preserved(self, pharmpy_shrinkage, openpkpd_shrinkage):
        """Relative shrinkage ordering across ETAs should match the Pharmpy reference."""
        pharmpy_values = np.asarray(pharmpy_shrinkage, dtype=float)
        assert np.array_equal(np.argsort(pharmpy_values), np.argsort(openpkpd_shrinkage))


@_pharmpy_required
@pytest.mark.external_validation
@pytest.mark.slow
class TestEstimatorVsPharmpyPheno:
    """Cross-tool estimator comparison on Pharmpy's empirical pheno dataset."""

    @pytest.fixture(scope="class")
    def pharmpy_results(self):
        return _load_pharmpy_pheno()

    @pytest.fixture(scope="class")
    def openpkpd_result(self, pharmpy_results):
        import warnings

        built = _build_openpkpd_pheno_model("FOCEI", maxeval=300)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return built.fit()

    def test_openpkpd_converges_on_pheno(self, openpkpd_result):
        assert openpkpd_result.converged, openpkpd_result.message

    def test_pheno_fixed_effects_track_pharmpy(self, pharmpy_results, openpkpd_result):
        _model, results = pharmpy_results
        ref = dict(results.parameter_estimates.items())

        assert openpkpd_result.theta_final[0] == pytest.approx(ref["POP_CL"], rel=0.25)
        assert openpkpd_result.theta_final[1] == pytest.approx(ref["POP_VC"], rel=0.25)
        assert openpkpd_result.theta_final[2] == pytest.approx(ref["COVAPGR"], rel=0.60)

    def test_pheno_variance_terms_track_pharmpy(self, pharmpy_results, openpkpd_result):
        _model, results = pharmpy_results
        ref = dict(results.parameter_estimates.items())

        np.testing.assert_allclose(
            np.diag(openpkpd_result.omega_final),
            [ref["IIV_CL"], ref["IIV_VC"]],
            rtol=0.40,
            atol=1e-4,
            err_msg="Pheno IIV variances drifted too far from Pharmpy's reference fit",
        )
        np.testing.assert_allclose(
            np.diag(openpkpd_result.sigma_final),
            [ref["SIGMA"]],
            rtol=0.30,
            atol=1e-5,
            err_msg="Pheno residual variance drifted too far from Pharmpy's reference fit",
        )

    def test_pheno_covariate_effect_direction_matches_pharmpy(
        self, pharmpy_results, openpkpd_result
    ):
        _model, results = pharmpy_results
        ref = dict(results.parameter_estimates.items())

        assert openpkpd_result.theta_final[2] > 0.0
        assert ref["COVAPGR"] > 0.0
        assert openpkpd_result.theta_final[2] / ref["COVAPGR"] > 0.4

    def test_pheno_ofv_is_finite_and_reasonable(self, openpkpd_result):
        assert np.isfinite(openpkpd_result.ofv)
        assert openpkpd_result.ofv > 0.0
        assert openpkpd_result.ofv < 5000.0


@_pharmpy_required
@pytest.mark.external_validation
@pytest.mark.slow
class TestNonparametricVsPharmpyPheno:
    """Empirical nonparametric benchmark on Pharmpy's phenobarbital pheno dataset."""

    @pytest.fixture(scope="class")
    def pharmpy_results(self):
        return _load_pharmpy_pheno()

    @pytest.fixture(scope="class")
    def openpkpd_result(self):
        import warnings

        built = _build_openpkpd_pheno_model(
            "NONPARAMETRIC",
            base_method="FOCEI",
            maxeval=300,
            max_iter=80,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return built.fit()

    def test_nonparametric_converges_and_preserves_support_geometry(self, openpkpd_result):
        assert openpkpd_result.converged, openpkpd_result.message
        assert len(openpkpd_result.support_weights) == len(openpkpd_result.support_points) == 59
        assert float(openpkpd_result.support_weights.sum()) == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.isfinite(openpkpd_result.support_weights))
        assert np.all(openpkpd_result.support_weights >= 0.0)

    def test_nonparametric_keeps_fixed_effects_near_pharmpy_basin(
        self, pharmpy_results, openpkpd_result
    ):
        _model, results = pharmpy_results
        ref = dict(results.parameter_estimates.items())

        assert openpkpd_result.theta_final[0] == pytest.approx(ref["POP_CL"], rel=0.08)
        assert openpkpd_result.theta_final[1] == pytest.approx(ref["POP_VC"], rel=0.08)
        assert openpkpd_result.theta_final[2] == pytest.approx(ref["COVAPGR"], rel=0.15)

    def test_nonparametric_sigma_and_empirical_variance_remain_plausible(
        self, pharmpy_results, openpkpd_result
    ):
        _model, results = pharmpy_results
        ref = dict(results.parameter_estimates.items())
        empirical_var = openpkpd_result.empirical_variance()

        np.testing.assert_allclose(
            np.diag(openpkpd_result.sigma_final),
            [ref["SIGMA"]],
            rtol=0.10,
            atol=1e-5,
            err_msg="Nonparametric pheno residual variance drifted too far from Pharmpy's fit",
        )
        assert empirical_var.shape == (2,)
        assert np.all(np.isfinite(empirical_var))
        assert np.all(empirical_var > 0.0)
        assert empirical_var[0] == pytest.approx(ref["IIV_CL"], rel=0.50)
        assert empirical_var[1] == pytest.approx(ref["IIV_VC"], rel=0.25)

    def test_nonparametric_support_distribution_is_not_degenerate(self, openpkpd_result):
        weights = openpkpd_result.support_weights
        assert float(weights.max()) < 0.20
        assert int(np.sum(weights > 0.01)) >= 20

    def test_nonparametric_empirical_eta_mean_stays_near_zero(self, openpkpd_result):
        np.testing.assert_allclose(openpkpd_result.empirical_mean(), np.zeros(2), atol=0.05)
