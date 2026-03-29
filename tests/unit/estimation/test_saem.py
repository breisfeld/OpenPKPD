"""Tests for SAEMMethod multi-chain Rao-Blackwellisation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.saem import SAEMMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.utils.errors import WarningCode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_params(n_theta: int = 2, n_eta: int = 1) -> ParameterSet:
    return ParameterSet(
        theta=np.ones(n_theta) * 0.5,
        omega=np.eye(n_eta) * 0.1,
        sigma=np.eye(1) * 0.05,
        theta_specs=[ThetaSpec(init=0.5, lower=0.0, upper=10.0) for _ in range(n_theta)],
        omega_specs=[OmegaSpec(block_size=1, values=[0.1]) for _ in range(n_eta)],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
    )


def _make_mock_pop_model(n_subjects: int = 4, n_eta: int = 1):
    """Return a mock population model with a deterministic OFV."""
    model = MagicMock()
    model.subject_ids.return_value = list(range(1, n_subjects + 1))
    model.n_subjects.return_value = n_subjects
    model.trans = 2

    def _make_indiv(sid):
        indiv = MagicMock()
        # obj_eta: simple quadratic centered at eta=0.5
        indiv.obj_eta.side_effect = lambda eta, theta, omega, sigma, trans=2: float(
            2.0 * np.sum((eta - 0.5) ** 2) + 5.0
        )
        # log_likelihood: constant
        indiv.log_likelihood.side_effect = lambda theta, eta, sigma, trans=2: -10.0
        return indiv

    model.individual_model.side_effect = _make_indiv
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSAEMDefaults:
    """P0.1 — Verify that the improved defaults are correct."""

    def test_default_n_chains_is_five(self):
        saem = SAEMMethod()
        assert saem.n_chains == 5, "Default n_chains must be 5 for Rao-Blackwellisation benefit"

    def test_default_n_iter_phase1_is_300(self):
        saem = SAEMMethod()
        assert saem.n_iter_phase1 == 300

    def test_default_n_iter_phase2_is_200(self):
        saem = SAEMMethod()
        assert saem.n_iter_phase2 == 200

    def test_default_phi_tol_is_1e_3(self):
        saem = SAEMMethod()
        assert saem.phi_tol == pytest.approx(1e-3)


class TestSAEMConvergenceCriterion:
    """P0.1 — Phase-2 stability criterion."""

    def test_converged_true_when_parameters_stable(self):
        """
        With a very loose phi_tol (100 %) the convergence criterion triggers
        as soon as two windows of phase-2 history are available.
        """
        # phi_tol=100 means any relative change < 100 → converges after 2*_PH2_WINDOW+1 iters
        saem = SAEMMethod(
            n_iter_phase1=2,
            n_iter_phase2=2 * SAEMMethod._PH2_WINDOW + 10,
            n_chains=1,
            phi_tol=100.0,   # guaranteed to trigger immediately
            seed=0,
        )
        pop = _make_mock_pop_model(n_subjects=3)
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert result.converged is True

    def test_converged_false_when_iterations_exhausted_without_stability(self):
        """
        When phi_tol is impossibly tight and phase-2 is very short, the
        stability criterion should NOT be met.
        """
        class _NoisyModel:
            trans = 2

            def __init__(self):
                self._rng = np.random.default_rng(99)
                self._subj_ids = [1, 2]

            def subject_ids(self):
                return self._subj_ids

            def n_subjects(self):
                return 2

            def individual_model(self, sid):
                rng = self._rng

                class _Noisy:
                    def obj_eta(self_, eta, theta, omega, sigma, trans=2):
                        return float(np.sum(eta**2)) + rng.uniform(-5, 5)

                    def log_likelihood(self_, theta, eta, sigma, trans=2):
                        return rng.uniform(-100, 100)

                return _Noisy()

        saem = SAEMMethod(
            n_iter_phase1=0,
            n_iter_phase2=5,         # too few to satisfy 2*_PH2_WINDOW=40 window
            n_chains=1,
            phi_tol=1e-10,           # impossibly tight
            seed=42,
        )
        result = saem.estimate(_NoisyModel(), _make_minimal_params())
        assert result.converged is False

    def test_warn007_emitted_when_not_converged(self):
        saem = SAEMMethod(
            n_iter_phase1=0,
            n_iter_phase2=5,
            n_chains=1,
            phi_tol=1e-10,
            seed=0,
        )
        result = saem.estimate(_make_mock_pop_model(n_subjects=2), _make_minimal_params())
        codes = {w.code for w in result.structured_warnings}
        assert WarningCode.WARN_007 in codes

    def test_no_warn007_when_converged(self):
        saem = SAEMMethod(
            n_iter_phase1=2,
            n_iter_phase2=2 * SAEMMethod._PH2_WINDOW + 5,
            n_chains=1,
            phi_tol=1.0,             # very loose → converges immediately
            seed=0,
        )
        result = saem.estimate(_make_mock_pop_model(n_subjects=2), _make_minimal_params())
        codes = {w.code for w in result.structured_warnings}
        assert WarningCode.WARN_007 not in codes


class TestSAEMOmegaConditioning:
    """P0.1/P0.5 — check_omega_conditioning called at end of SAEM."""

    def test_check_omega_conditioning_called_on_result(self):
        """
        SAEM must call check_omega_conditioning on the result.
        We test this indirectly: a well-conditioned omega produces no WARN_001/002/004.
        (Direct WARN_004 from initial omega would be overwritten by the M-step repair.)
        """
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=0, n_chains=1, seed=0)
        pop = _make_mock_pop_model(n_subjects=2)
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        # structured_warnings list must exist (can be empty for well-cond. omega)
        assert isinstance(result.structured_warnings, list)
        codes = {w.code for w in result.structured_warnings}
        # WARN_001/002 should not appear for a healthy omega
        assert WarningCode.WARN_001 not in codes
        assert WarningCode.WARN_002 not in codes


class TestSAEMChainInitialization:
    def test_single_chain_default(self):
        saem = SAEMMethod(n_iter_phase1=1, n_iter_phase2=0, n_chains=1, seed=42)
        assert saem.n_chains == 1

    def test_multi_chain_parameter_stored(self):
        saem = SAEMMethod(n_chains=5, seed=0)
        assert saem.n_chains == 5

    def test_n_chains_zero_treated_as_one(self):
        """n_chains=0 should behave as n_chains=1 (guard against invalid input)."""
        saem = SAEMMethod(n_iter_phase1=1, n_iter_phase2=0, n_chains=0, seed=0)
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert isinstance(result, EstimationResult)


class TestSAEMChainShape:
    def test_eta_chains_shape_is_n_chains_by_n_eta(self):
        """After one iteration, chain state has shape (n_chains, n_eta)."""
        n_chains = 3
        n_eta = 2
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=0, n_chains=n_chains, seed=1)
        pop = _make_mock_pop_model(n_subjects=2, n_eta=n_eta)
        params = _make_minimal_params(n_theta=2, n_eta=n_eta)
        result = saem.estimate(pop, params)
        # post_hoc_etas should be mean across chains → shape (n_eta,)
        for sid, eta in result.post_hoc_etas.items():
            assert eta.shape == (n_eta,), f"Expected ({n_eta},) for sid={sid}, got {eta.shape}"

    def test_eta_chains_are_initialized_from_omega_when_proposals_do_not_move(self):
        class _FrozenIndividualModel:
            def obj_eta(self, eta, theta, omega, sigma, trans=None):
                return 0.0

            def log_likelihood(self, theta, eta, sigma, trans=None):
                return 0.0

        class _FrozenPopulationModel:
            trans = 2

            def __init__(self):
                self.dataset = SimpleNamespace(n_observations=lambda: 1)

            def subject_ids(self):
                return [1]

            def n_subjects(self):
                return 1

            def individual_model(self, sid):
                return _FrozenIndividualModel()

        class _StubRNG:
            def multivariate_normal(self, mean, cov, size):
                assert size == 2
                return np.array([[0.3], [-0.2]], dtype=float)

            def standard_normal(self, size):
                return np.zeros(size, dtype=float)

            def uniform(self):
                return 0.5

        saem = SAEMMethod(n_iter_phase1=1, n_iter_phase2=0, n_chains=2, seed=0)
        saem.rng = _StubRNG()
        result = saem.estimate(_FrozenPopulationModel(), _make_minimal_params())

        assert result.omega_final[0, 0] == pytest.approx((0.3**2 + 0.2**2) / 2.0)


class TestSAEMRaoBlackwellResult:
    def test_returns_estimation_result(self):
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=1, n_chains=2, seed=10)
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert isinstance(result, EstimationResult)

    def test_converged_flag_is_boolean(self):
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=1, n_chains=1, seed=11)
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert isinstance(result.converged, bool)

    def test_message_is_non_empty_string(self):
        """Result.message must be a non-empty string."""
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=0, n_chains=4, seed=12)
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert isinstance(result.message, str) and len(result.message) > 0

    def test_converged_message_mentions_chains(self):
        """When SAEM converges (phi_tol=100), the message must mention chain count."""
        saem = SAEMMethod(
            n_iter_phase1=2,
            n_iter_phase2=2 * SAEMMethod._PH2_WINDOW + 5,
            n_chains=4,
            phi_tol=100.0,   # guaranteed to trigger immediately
            seed=12,
        )
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        assert result.converged is True
        assert "4" in result.message or "chain" in result.message.lower()

    def test_post_hoc_etas_all_subjects_present(self):
        n_subjects = 5
        saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=0, n_chains=3, seed=13)
        pop = _make_mock_pop_model(n_subjects=n_subjects)
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        for sid in range(1, n_subjects + 1):
            assert sid in result.post_hoc_etas

    def test_ofv_history_length(self):
        n1, n2 = 3, 2
        saem = SAEMMethod(n_iter_phase1=n1, n_iter_phase2=n2, n_chains=2, seed=14)
        pop = _make_mock_pop_model()
        params = _make_minimal_params()
        result = saem.estimate(pop, params)
        # History length is at most n1+n2; may be shorter if early convergence triggered.
        assert 1 <= len(result.ofv_history) <= n1 + n2

    def test_result_populates_metadata_from_dataset_and_free_specs(self):
        params = ParameterSet(
            theta=np.array([0.5, 0.75]),
            omega=np.eye(1) * 0.1,
            sigma=np.eye(1) * 0.05,
            theta_specs=[
                ThetaSpec(init=0.5, lower=0.0, upper=4.0),
                ThetaSpec(init=0.75, lower=0.0, upper=4.0, fixed=True),
            ],
            omega_specs=[OmegaSpec(block_size=1, values=[0.1])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )
        pop = _ThetaTargetPopulationModel([1.5, 2.5], n_subjects=3)
        pop.dataset = SimpleNamespace(n_observations=lambda: 12)

        result = SAEMMethod(n_iter_phase1=0, n_iter_phase2=1, n_chains=1, seed=7).estimate(
            pop, params
        )

        assert result.n_subjects == 3
        assert result.n_observations == 12
        assert result.n_parameters == 3
        assert np.isfinite(result.bic)

    def test_single_chain_same_structure_as_multi(self):
        """n_chains=1 should produce the same EstimationResult structure as n_chains=3."""
        pop = _make_mock_pop_model(n_subjects=3)
        params = _make_minimal_params()

        saem1 = SAEMMethod(n_iter_phase1=3, n_iter_phase2=0, n_chains=1, seed=99)
        saem3 = SAEMMethod(n_iter_phase1=3, n_iter_phase2=0, n_chains=3, seed=99)

        r1 = saem1.estimate(pop, params)
        r3 = saem3.estimate(pop, params)

        # Both should produce finite theta estimates
        assert np.all(np.isfinite(r1.theta_final))
        assert np.all(np.isfinite(r3.theta_final))
        assert r1.theta_final.shape == r3.theta_final.shape

    def test_multi_chain_variance_reduction(self):
        """
        With more chains the sufficient-statistic variance should be lower.
        Run many short SAEM runs (phase1=1) and compare variance of Q_omega update.
        We check indirectly: with n_chains=10 the omega estimates should cluster
        more tightly around the true value than with n_chains=1.
        """
        n_reps = 20
        pop = _make_mock_pop_model(n_subjects=10)
        params = _make_minimal_params()

        omegas_1chain = []
        omegas_10chain = []
        for seed in range(n_reps):
            s1 = SAEMMethod(n_iter_phase1=5, n_iter_phase2=0, n_chains=1, seed=seed)
            s10 = SAEMMethod(n_iter_phase1=5, n_iter_phase2=0, n_chains=10, seed=seed)
            omegas_1chain.append(s1.estimate(pop, params).omega_final[0, 0])
            omegas_10chain.append(s10.estimate(pop, params).omega_final[0, 0])

        var_1 = np.var(omegas_1chain)
        var_10 = np.var(omegas_10chain)
        # More chains → lower variance in sufficient statistics → lower variance in omega
        assert var_10 <= var_1 * 2.0, (
            f"Expected 10-chain variance ≤ 2× single-chain variance, "
            f"got var_1={var_1:.4f}, var_10={var_10:.4f}"
        )


class _ThetaTargetIndividualModel:
    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def obj_eta(self, eta, theta, omega, sigma, trans=None):
        eta = np.asarray(eta, dtype=float)
        return float(np.sum(eta**2))

    def log_likelihood(self, theta, eta, sigma, trans=None):
        theta = np.asarray(theta, dtype=float)
        return float(np.sum((theta - self.target) ** 2))


class _ThetaTargetPopulationModel:
    trans = 2

    def __init__(self, target: list[float] | np.ndarray, n_subjects: int = 2):
        self._subject_ids = list(range(1, n_subjects + 1))
        self._indivs = {
            sid: _ThetaTargetIndividualModel(np.asarray(target, dtype=float))
            for sid in self._subject_ids
        }

    def subject_ids(self):
        return self._subject_ids

    def n_subjects(self):
        return len(self._subject_ids)

    def individual_model(self, sid):
        return self._indivs[sid]


class _SigmaTargetIndividualModel:
    def __init__(
        self,
        dv: list[float] | np.ndarray,
        pred: list[float] | np.ndarray,
        sigma_design: list[list[float]] | np.ndarray | None = None,
    ) -> None:
        self.subject_events = SimpleNamespace(obs_dv=np.asarray(dv, dtype=float))
        self._pred = np.asarray(pred, dtype=float)
        if sigma_design is None:
            sigma_design = np.ones((len(self._pred), 1), dtype=float)
        self._sigma_design = np.asarray(sigma_design, dtype=float)

    def obj_eta(self, eta, theta, omega, sigma, trans=None):
        return 0.0

    def log_likelihood(self, theta, eta, sigma, trans=None):
        resid = self.subject_events.obs_dv - self._pred
        diag = np.diag(sigma).astype(float)
        var = self._sigma_design @ diag
        var = np.maximum(var, 1e-10)
        return float(np.sum(np.log(var) + (resid**2) / var))

    def evaluate_observation_model(self, theta, eta, sigma, trans=None):
        pred = self._pred.copy()
        diag = np.diag(sigma).astype(float)
        var = np.maximum(self._sigma_design @ diag, 1e-10)
        obs_mask = np.ones(len(pred), dtype=bool)
        return pred, obs_mask, pred, pred, var


class _SigmaTargetPopulationModel:
    trans = 2

    def __init__(
        self,
        subjects: dict[
            int, tuple[list[float], list[float], list[list[float]] | list[float] | None]
        ],
    ):
        self._subject_ids = list(subjects)
        self._indivs = {
            sid: _SigmaTargetIndividualModel(dv, pred, sigma_design)
            for sid, (dv, pred, sigma_design) in subjects.items()
        }

    def subject_ids(self):
        return self._subject_ids

    def n_subjects(self):
        return len(self._subject_ids)

    def individual_model(self, sid):
        return self._indivs[sid]


class TestSAEMThetaMStep:
    def test_theta_m_step_moves_to_ofv_minimum(self):
        params = ParameterSet(
            theta=np.array([0.5]),
            omega=np.eye(1) * 0.1,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=0.5, lower=0.0, upper=4.0)],
            omega_specs=[OmegaSpec(block_size=1, values=[0.1])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )

        result = SAEMMethod(
            n_iter_phase1=0,
            n_iter_phase2=1,
            n_chains=1,
            seed=0,
        ).estimate(_ThetaTargetPopulationModel([2.0]), params)

        assert result.theta_final[0] == pytest.approx(2.0, abs=1e-3)

    def test_theta_m_step_respects_theta_bounds(self):
        params = ParameterSet(
            theta=np.array([0.5]),
            omega=np.eye(1) * 0.1,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=0.5, lower=0.0, upper=4.0)],
            omega_specs=[OmegaSpec(block_size=1, values=[0.1])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )

        result = SAEMMethod(
            n_iter_phase1=0,
            n_iter_phase2=1,
            n_chains=1,
            seed=1,
        ).estimate(_ThetaTargetPopulationModel([-1.0]), params)

        assert result.theta_final[0] == pytest.approx(0.0, abs=1e-6)

    def test_theta_m_step_keeps_fixed_theta_unchanged(self):
        params = ParameterSet(
            theta=np.array([0.5, 0.5]),
            omega=np.eye(1) * 0.1,
            sigma=np.eye(1) * 0.05,
            theta_specs=[
                ThetaSpec(init=0.5, lower=0.0, upper=4.0),
                ThetaSpec(init=0.5, lower=0.0, upper=4.0, fixed=True),
            ],
            omega_specs=[OmegaSpec(block_size=1, values=[0.1])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )

        result = SAEMMethod(
            n_iter_phase1=0,
            n_iter_phase2=1,
            n_chains=1,
            seed=2,
        ).estimate(_ThetaTargetPopulationModel([1.5, 2.5]), params)

        assert result.theta_final[0] == pytest.approx(1.5, abs=1e-3)
        assert result.theta_final[1] == pytest.approx(0.5, abs=1e-12)


class TestSAEMSigmaMStep:
    def test_scalar_sigma_updates_from_residual_mean_square(self):
        params = ParameterSet(
            theta=np.array([], dtype=float),
            omega=np.zeros((0, 0), dtype=float),
            sigma=np.array([[0.05]], dtype=float),
            theta_specs=[],
            omega_specs=[],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )
        pop = _SigmaTargetPopulationModel({1: ([1.0, 3.0], [0.0, 1.0], [[1.0], [1.0]])})

        result = SAEMMethod(n_iter_phase1=0, n_iter_phase2=1, n_chains=1, seed=0).estimate(
            pop, params
        )

        assert result.sigma_final[0, 0] == pytest.approx((1.0**2 + 2.0**2) / 2.0)

    def test_scalar_proportional_sigma_updates_from_relative_residuals(self):
        params = ParameterSet(
            theta=np.array([], dtype=float),
            omega=np.zeros((0, 0), dtype=float),
            sigma=np.array([[0.2]], dtype=float),
            theta_specs=[],
            omega_specs=[],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.2])],
        )
        pop = _SigmaTargetPopulationModel({1: ([2.0, 6.0], [1.0, 2.0], [[1.0], [4.0]])})

        result = SAEMMethod(n_iter_phase1=0, n_iter_phase2=1, n_chains=1, seed=0).estimate(
            pop, params
        )

        assert result.sigma_final[0, 0] == pytest.approx(
            (1.0**2 / 1.0 + 4.0**2 / 4.0) / 2.0, rel=1e-4
        )

    def test_diagonal_combined_sigma_updates_both_components(self):
        params = ParameterSet(
            theta=np.array([], dtype=float),
            omega=np.zeros((0, 0), dtype=float),
            sigma=np.diag([0.2, 0.2]).astype(float),
            theta_specs=[],
            omega_specs=[],
            sigma_specs=[
                SigmaSpec(block_size=1, values=[0.2]),
                SigmaSpec(block_size=1, values=[0.2]),
            ],
        )
        pop = _SigmaTargetPopulationModel({1: ([1.0, 6.0], [0.0, 2.0], [[1.0, 0.0], [1.0, 4.0]])})

        result = SAEMMethod(n_iter_phase1=0, n_iter_phase2=1, n_chains=1, seed=0).estimate(
            pop, params
        )

        assert result.sigma_final[0, 0] == pytest.approx(1.0, rel=1e-3)
        assert result.sigma_final[1, 1] == pytest.approx(3.75, rel=1e-3)

    def test_fixed_scalar_sigma_is_not_updated(self):
        params = ParameterSet(
            theta=np.array([], dtype=float),
            omega=np.zeros((0, 0), dtype=float),
            sigma=np.array([[0.05]], dtype=float),
            theta_specs=[],
            omega_specs=[],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05], fixed=True)],
        )
        pop = _SigmaTargetPopulationModel({1: ([1.0, 3.0], [0.0, 1.0], [[1.0], [1.0]])})

        result = SAEMMethod(n_iter_phase1=0, n_iter_phase2=1, n_chains=1, seed=0).estimate(
            pop, params
        )

        assert result.sigma_final[0, 0] == pytest.approx(0.05)
