"""
Unit tests for MixtureResult dataclass and MixtureModel structure.

Tests cover:
  - MixtureResult dataclass construction and validation
  - summary() output
  - subject_assignments() logic
  - MixtureModel construction argument validation
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.mixture.mixture import MixtureModel, MixtureResult

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_est_result(theta: list[float], ofv: float = 100.0) -> EstimationResult:
    """Create a minimal EstimationResult."""
    n = len(theta)
    return EstimationResult(
        theta_final=np.array(theta),
        omega_final=np.eye(n) * 0.3,
        sigma_final=np.eye(1) * 0.1,
        ofv=ofv,
        converged=True,
        method="FOCE",
    )


def _make_mixture_result(
    n_subpop: int = 2,
    mixture_probs: list[float] | None = None,
) -> MixtureResult:
    """Build a generic MixtureResult."""
    if mixture_probs is None:
        mixture_probs = [1.0 / n_subpop] * n_subpop
    probs_arr = np.array(mixture_probs)
    subpop_probs = {
        1: np.array([0.8, 0.2]),
        2: np.array([0.3, 0.7]),
        3: np.array([0.5, 0.5]),
    }
    subpop_results = [
        _make_est_result([1.0, 0.1, 30.0], ofv=100.0),
        _make_est_result([2.0, 0.2, 50.0], ofv=110.0),
    ]
    return MixtureResult(
        n_subpop=n_subpop,
        mixture_probs=probs_arr,
        subpop_probabilities=subpop_probs,
        subpop_results=subpop_results,
        ofv=95.0,
        converged=True,
    )


class _DummyParams:
    def __init__(self, theta: list[float]) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.omega = np.eye(1) * 0.25
        self.sigma = np.eye(1) * 0.1

    def n_eta(self) -> int:
        return 0


class _ConstantOfvIndividual:
    def __init__(self, ofv: float) -> None:
        self.ofv = float(ofv)

    def log_likelihood(self, theta, eta, sigma, trans=None):
        return self.ofv


class _ConstantOfvPopulation:
    trans = None

    def __init__(self, ofv_by_subject: dict[int, float]) -> None:
        self._ofv_by_subject = ofv_by_subject

    def subject_ids(self) -> list[int]:
        return sorted(self._ofv_by_subject)

    def individual_model(self, subject_id: int) -> _ConstantOfvIndividual:
        return _ConstantOfvIndividual(self._ofv_by_subject[subject_id])


class _SubjectListPopulation:
    trans = None

    def __init__(self, subject_ids: list[int]) -> None:
        self._subject_ids = list(subject_ids)

    def subject_ids(self) -> list[int]:
        return list(self._subject_ids)


# ── MixtureResult structure ───────────────────────────────────────────────────


class TestMixtureResultStructure:
    """Tests for MixtureResult dataclass correctness."""

    def test_n_subpop(self) -> None:
        result = _make_mixture_result()
        assert result.n_subpop == 2

    def test_mixture_probs_sum_to_one(self) -> None:
        result = _make_mixture_result()
        assert np.isclose(result.mixture_probs.sum(), 1.0)

    def test_mixture_probs_unequal(self) -> None:
        result = _make_mixture_result(mixture_probs=[0.6, 0.4])
        assert result.mixture_probs[0] == pytest.approx(0.6)
        assert result.mixture_probs[1] == pytest.approx(0.4)

    def test_ofv_stored(self) -> None:
        result = _make_mixture_result()
        assert result.ofv == pytest.approx(95.0)

    def test_converged_flag(self) -> None:
        result = _make_mixture_result()
        assert result.converged is True

    def test_subpop_results_length(self) -> None:
        result = _make_mixture_result()
        assert len(result.subpop_results) == result.n_subpop

    def test_subpop_probabilities_are_dict(self) -> None:
        result = _make_mixture_result()
        assert isinstance(result.subpop_probabilities, dict)

    def test_subpop_probabilities_shape(self) -> None:
        """Each subject's posterior array should have length n_subpop."""
        result = _make_mixture_result()
        for _sid, probs in result.subpop_probabilities.items():
            assert probs.shape == (result.n_subpop,)

    def test_est_result_fields_preserved(self) -> None:
        r = _make_est_result([1.0, 2.0], ofv=88.0)
        result = MixtureResult(
            n_subpop=2,
            mixture_probs=np.array([0.6, 0.4]),
            subpop_probabilities={1: np.array([0.8, 0.2]), 2: np.array([0.3, 0.7])},
            subpop_results=[r, r],
            ofv=95.0,
            converged=True,
        )
        assert result.subpop_results[0].ofv == pytest.approx(88.0)

    def test_repr_contains_subpop_count(self) -> None:
        result = _make_mixture_result()
        r = repr(result)
        assert "n_subpop=2" in r


# ── MixtureResult summary ─────────────────────────────────────────────────────


class TestMixtureResultSummary:
    """Tests for MixtureResult.summary()."""

    def test_summary_is_string(self) -> None:
        result = _make_mixture_result()
        s = result.summary()
        assert isinstance(s, str)

    def test_summary_contains_ofv(self) -> None:
        result = _make_mixture_result()
        s = result.summary()
        assert "OFV" in s
        assert "95.0" in s or "95" in s

    def test_summary_contains_mixing_probs(self) -> None:
        result = _make_mixture_result(mixture_probs=[0.6, 0.4])
        s = result.summary()
        assert "Mixing" in s or "proportion" in s.lower()

    def test_summary_contains_subpop_headers(self) -> None:
        result = _make_mixture_result()
        s = result.summary()
        assert "Subpop 1" in s
        assert "Subpop 2" in s

    def test_summary_converged_flag(self) -> None:
        result = _make_mixture_result()
        s = result.summary()
        assert "True" in s

    def test_summary_not_converged(self) -> None:
        result = _make_mixture_result()
        object.__setattr__(result, "converged", False)
        s = result.summary()
        assert "False" in s


# ── subject_assignments ───────────────────────────────────────────────────────


class TestSubjectAssignments:
    """Tests for MixtureResult.subject_assignments()."""

    def test_hard_assignment_subject_1(self) -> None:
        """Subject 1 has P(subpop1)=0.8, P(subpop2)=0.2 → assigned to 1."""
        result = _make_mixture_result()
        assignments = result.subject_assignments()
        assert assignments[1] == 1

    def test_hard_assignment_subject_2(self) -> None:
        """Subject 2 has P(subpop1)=0.3, P(subpop2)=0.7 → assigned to 2."""
        result = _make_mixture_result()
        assignments = result.subject_assignments()
        assert assignments[2] == 2

    def test_tie_goes_to_first_argmax(self) -> None:
        """Subject 3 has equal probabilities; argmax returns 0 (subpop 1)."""
        result = _make_mixture_result()
        assignments = result.subject_assignments()
        # Subject 3: [0.5, 0.5] → argmax = 0 → subpop 1
        assert assignments[3] == 1

    def test_assignment_values_are_1_based(self) -> None:
        result = _make_mixture_result()
        assignments = result.subject_assignments()
        for val in assignments.values():
            assert 1 <= val <= result.n_subpop


# ── MixtureModel constructor ──────────────────────────────────────────────────


class TestMixtureModelConstructor:
    """Tests for MixtureModel argument validation."""

    def test_requires_at_least_2_subpop(self) -> None:
        with pytest.raises(ValueError, match="n_subpop"):
            MixtureModel(population_model=object(), n_subpop=1)

    def test_default_n_subpop(self) -> None:
        model = MixtureModel(population_model=object())
        assert model.n_subpop == 2

    def test_default_max_iter(self) -> None:
        model = MixtureModel(population_model=object())
        assert model.max_iter == 100

    def test_custom_tol(self) -> None:
        model = MixtureModel(population_model=object(), tol=1e-6)
        assert model.tol == pytest.approx(1e-6)

    def test_stores_population_model(self) -> None:
        sentinel = object()
        model = MixtureModel(population_model=sentinel)
        assert model.population_model is sentinel

    def test_estimation_method_stored(self) -> None:
        model = MixtureModel(population_model=object(), estimation_method="FO")
        assert model.estimation_method == "FO"

    def test_estimation_kwargs_stored(self) -> None:
        model = MixtureModel(
            population_model=object(),
            estimation_kwargs={"maxeval": 100},
        )
        assert model.estimation_kwargs["maxeval"] == 100


class TestMixtureModelNumerics:
    def test_subject_log_likelihood_converts_ofv_to_loglik(self) -> None:
        model = MixtureModel(
            population_model=_ConstantOfvPopulation({1: 8.0}),
            n_subpop=2,
        )

        ll = model._subject_log_likelihood(1, _DummyParams([0.0]))

        assert ll == pytest.approx(-4.0)

    def test_e_step_matches_manual_posteriors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MixtureModel(population_model=_SubjectListPopulation([1, 2]), n_subpop=2)
        params_per_subpop = [_DummyParams([0.0]), _DummyParams([1.0])]
        mixing_probs = np.array([0.25, 0.75])
        ll_lookup = {
            (1, 0): -2.0,
            (1, 1): -1.0,
            (2, 0): -0.5,
            (2, 1): -3.0,
        }

        monkeypatch.setattr(
            model,
            "_subject_log_likelihood",
            lambda sid, params: ll_lookup[(sid, int(params.theta[0]))],
        )

        responsibilities = model._e_step(params_per_subpop, mixing_probs)
        expected = []
        for sid in [1, 2]:
            weighted = np.array([mixing_probs[k] * np.exp(ll_lookup[(sid, k)]) for k in range(2)])
            expected.append(weighted / weighted.sum())

        np.testing.assert_allclose(responsibilities, np.array(expected), rtol=1e-10)

    def test_mixture_log_likelihood_matches_manual_logsumexp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = MixtureModel(population_model=_SubjectListPopulation([1, 2]), n_subpop=2)
        params_per_subpop = [_DummyParams([0.0]), _DummyParams([1.0])]
        mixing_probs = np.array([0.4, 0.6])
        ll_lookup = {
            (1, 0): -1.5,
            (1, 1): -0.5,
            (2, 0): -0.25,
            (2, 1): -2.0,
        }

        monkeypatch.setattr(
            model,
            "_subject_log_likelihood",
            lambda sid, params: ll_lookup[(sid, int(params.theta[0]))],
        )

        ll = model._mixture_log_likelihood(params_per_subpop, mixing_probs)
        expected = 0.0
        for sid in [1, 2]:
            expected += np.log(sum(mixing_probs[k] * np.exp(ll_lookup[(sid, k)]) for k in range(2)))

        assert ll == pytest.approx(expected)

    def test_m_step_updates_mixing_and_assigned_subpopulation_parameters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = MixtureModel(
            population_model=_SubjectListPopulation([10, 20, 30]),
            n_subpop=2,
            shared_variance=True,
            estimation_kwargs={"maxeval": 7},
        )
        params_per_subpop = [_DummyParams([5.0]), _DummyParams([9.0])]
        posteriors = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )
        recorded_kwargs: dict[str, int] = {}

        class _Estimator:
            def estimate(self, restricted_model, params):
                subject_ids = restricted_model.subject_ids()
                return EstimationResult(
                    theta_final=np.array([float(len(subject_ids))]),
                    omega_final=np.eye(1) * 9.0,
                    sigma_final=np.eye(1) * 8.0,
                    ofv=0.0,
                    converged=True,
                    method="FOCE",
                )

        def _fake_get_estimation_method(method: str, **kwargs):
            recorded_kwargs.update(kwargs)
            return _Estimator()

        monkeypatch.setattr(
            "openpkpd.estimation.get_estimation_method",
            _fake_get_estimation_method,
        )
        monkeypatch.setattr(
            model,
            "_restrict_to_subjects",
            lambda ids: type(
                "RestrictedModel",
                (),
                {"subject_ids": staticmethod(lambda: list(ids))},
            )(),
        )

        new_mixing, new_params = model._m_step(
            posteriors,
            params_per_subpop,
            init_params=_DummyParams([1.0]),
            subject_ids=[10, 20, 30],
        )

        np.testing.assert_allclose(new_mixing, posteriors.mean(axis=0))
        np.testing.assert_allclose(new_params[0].theta, np.array([2.0]))
        np.testing.assert_allclose(new_params[1].theta, np.array([1.0]))
        np.testing.assert_allclose(new_params[0].omega, params_per_subpop[0].omega)
        np.testing.assert_allclose(new_params[1].sigma, params_per_subpop[1].sigma)
        assert recorded_kwargs == {"maxeval": 7}

    def test_build_subpop_results_uses_one_component_likelihood_per_subpop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = MixtureModel(population_model=_SubjectListPopulation([1, 2]), n_subpop=2)
        params_per_subpop = [_DummyParams([0.0]), _DummyParams([1.0])]
        ll_lookup = {
            (1, 0): -1.0,
            (2, 0): -2.0,
            (1, 1): -0.25,
            (2, 1): -0.75,
        }

        monkeypatch.setattr(
            model,
            "_subject_log_likelihood",
            lambda sid, params: ll_lookup[(sid, int(params.theta[0]))],
        )

        results = model._build_subpop_results(params_per_subpop)

        assert len(results) == 2
        assert results[0].ofv == pytest.approx(6.0)
        assert results[1].ofv == pytest.approx(2.0)


# ── Spec-provided test from assignment ───────────────────────────────────────


def test_mixture_result_structure() -> None:
    """Spec-mandated test: MixtureResult dataclass works correctly."""
    r = EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=100.0,
    )
    result = MixtureResult(
        n_subpop=2,
        mixture_probs=np.array([0.6, 0.4]),
        subpop_probabilities={1: np.array([0.8, 0.2]), 2: np.array([0.3, 0.7])},
        subpop_results=[r, r],
        ofv=95.0,
        converged=True,
    )
    assert result.n_subpop == 2
    assert np.isclose(result.mixture_probs.sum(), 1.0)
