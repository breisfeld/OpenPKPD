"""
External-validation tests for estimation method OFV formulas.

Tests verify that the FOCE, Laplacian and IMP methods implement their
respective mathematical formulas correctly.  For a linear-Gaussian mock
model the expected OFV can be computed analytically, providing an
independent closed-form reference.

FOCE OFV formula (Beal & Sheiner 1992; NONMEM 7 reference):
    OFV_i = n_obs·log(2π) + log|Cᵢ| + (y−f)ᵀCᵢ⁻¹(y−f) + ηᵀΩ⁻¹η
            + n_eta·log(2π) + log|Ω|

Laplacian correction (Wolfinger 1993):
    OFV_Lap_i = OFV_FOCE_base_i + log|H_i|
    where H_i = ∂²obj_eta/∂η² (Hessian of the individual objective)

IMP (importance sampling):
    OFV_i ≈ −2 log p(yᵢ) via Monte-Carlo integration;
    converges to −2·analytic_log_marginal as N_samples → ∞.

References
----------
Beal SL, Sheiner LB (1992). NONMEM User's Guides. UCSF.
Wolfinger RD (1993). Laplace's approximation for nonlinear mixed models.
  Biometrika 80:791-795.
Pinheiro JC, Bates DM (1995). Approximations to the log-likelihood function
  in the nonlinear mixed-effects model. J Comput Graph Stat 4:12-35.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.imp import IMPMethod
from openpkpd.estimation.laplacian import LaplacianMethod

# ---------------------------------------------------------------------------
# Shared mock infrastructure — linear-Gaussian individual model
# ---------------------------------------------------------------------------


class _GaussianSubjectEvents:
    def __init__(self, dv: float) -> None:
        self.obs_dv = np.array([dv], dtype=float)

    def observation_mask(self) -> np.ndarray:
        return np.array([True])


class _GaussianIndividual:
    """
    Individual model:  y ~ N(η, σ²),  η ~ N(0, ω)
    obj_eta = log(2πσ) + (y−η)²/σ + η²/ω  (= −2·log p(y|η)·p(η) )
    """

    def __init__(self, dv: float, sigma_var: float) -> None:
        self._dv = float(dv)
        self._sigma_var = float(sigma_var)
        self.subject_events = _GaussianSubjectEvents(dv)

    def obj_eta(self, eta, theta, omega, sigma, trans=None) -> float:
        eta_v = float(np.asarray(eta)[0])
        omega_v = float(omega[0, 0])
        sig_v = float(sigma[0, 0])
        return float(
            math.log(2 * math.pi * sig_v) + (self._dv - eta_v) ** 2 / sig_v + eta_v**2 / omega_v
        )

    def evaluate_observation_model(self, theta, eta, sigma, trans=None):
        eta_v = float(np.asarray(eta)[0])
        pred = np.array([eta_v])
        var = np.array([float(sigma[0, 0])])
        return pred, np.array([True]), pred, pred, var

    def log_likelihood(self, theta, eta, sigma, trans=None) -> float:
        eta_v = float(np.asarray(eta)[0])
        sig_v = float(sigma[0, 0])
        return float(math.log(2 * math.pi * sig_v) + (self._dv - eta_v) ** 2 / sig_v)


class _GaussianPopulation:
    trans = 2

    def __init__(self, dvs, sigma_var: float) -> None:
        self._subjects = {i + 1: _GaussianIndividual(dv, sigma_var) for i, dv in enumerate(dvs)}

    def subject_ids(self):
        return sorted(self._subjects)

    def individual_model(self, sid):
        return self._subjects[sid]


class _GaussianParams:
    def __init__(self, omega_var: float, sigma_var: float) -> None:
        self.theta = np.array([0.0])
        self.omega = np.array([[omega_var]], dtype=float)
        self.sigma = np.array([[sigma_var]], dtype=float)

    def n_eta(self) -> int:
        return 1


def _analytic_log_marginal(dv: float, omega_var: float, sigma_var: float) -> float:
    """
    Full log marginal for y ~ N(0, ω + σ²).

    The IMP implementation uses the full Gaussian prior normalization,
    so the analytic marginal is the standard normal density:
        log p(y) = -0.5 * [log(2π(ω+σ)) + y²/(ω+σ)]
    """
    total_var = omega_var + sigma_var
    return -0.5 * (math.log(2.0 * math.pi * total_var) + dv**2 / total_var)


def _foce_base_at_map(dv: float, omega_var: float, sigma_var: float) -> float:
    """
    FOCE OFV at MAP eta for 1 subject, 1 observation (no prior_const term).
    Formula: log(2πσ) + (dv−η_MAP)²/σ + η_MAP²/ω
    MAP eta = dv·ω/(ω+σ)
    """
    eta_map = dv * omega_var / (omega_var + sigma_var)
    return (
        math.log(2 * math.pi * sigma_var) + (dv - eta_map) ** 2 / sigma_var + eta_map**2 / omega_var
    )


def _laplacian_expected(dv: float, omega_var: float, sigma_var: float) -> float:
    """
    Expected Laplacian OFV = FOCE_base + log|H|
    where H = d²obj_eta/dη² = 2/σ + 2/ω  (second derivative of Gaussian obj_eta)
    """
    H = 2.0 / sigma_var + 2.0 / omega_var
    return _foce_base_at_map(dv, omega_var, sigma_var) + math.log(H)


# ---------------------------------------------------------------------------
# Laplacian OFV formula tests
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestLaplacianOFVFormula:
    """
    Verify Laplacian OFV = FOCE_base + log|H| against analytic computation.

    For the Gaussian mock, d²obj_eta/dη² = 2/σ + 2/ω, so the expected
    Laplacian OFV is computable in closed form.  This is the NONMEM
    Laplacian algorithm formula (Wolfinger 1993 / Beal & Sheiner 1992).
    """

    @pytest.mark.parametrize(
        "dv,omega_v,sigma_v",
        [
            (1.5, 0.4, 0.6),
            (0.5, 0.3, 0.2),
            (2.0, 1.0, 0.5),
            (-0.8, 0.5, 0.3),
            (0.0, 0.8, 0.8),
        ],
    )
    def test_single_subject_laplacian_formula(self, dv, omega_v, sigma_v):
        """
        Laplacian OFV = FOCE_base(η_MAP) + log|d²obj_eta/dη²|
        The Hessian for this mock is H = 2/σ + 2/ω, exact in closed form.
        """
        expected = _laplacian_expected(dv, omega_v, sigma_v)
        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        eta_map = dv * omega_v / (omega_v + sigma_v)
        eta_hat = {1: np.array([eta_map])}

        ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
        assert ofv == pytest.approx(expected, abs=1e-6), (
            f"dv={dv}, omega={omega_v}, sigma={sigma_v}: got {ofv:.6f}, expected {expected:.6f}"
        )

    @pytest.mark.parametrize(
        "dvs,omega_v,sigma_v",
        [
            ([0.5, 1.0, 1.5], 0.3, 0.2),
            ([-0.5, 0.8, 1.2, -0.3], 0.5, 0.5),
        ],
    )
    def test_multi_subject_sums_over_individuals(self, dvs, omega_v, sigma_v):
        """Laplacian OFV is additive: multi-subject = Σ single-subject."""
        expected = sum(_laplacian_expected(dv, omega_v, sigma_v) for dv in dvs)
        pop = _GaussianPopulation(dvs, sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        eta_hat = {
            i + 1: np.array([dvs[i] * omega_v / (omega_v + sigma_v)]) for i in range(len(dvs))
        }
        ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
        assert ofv == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("omega_v", [0.05, 0.2, 0.5, 1.0, 2.0])
    def test_laplacian_ofv_exceeds_foce_base(self, omega_v):
        """
        Laplacian OFV > FOCE_base for all ω since log|H| > 0 for H > 1
        (and H = 2/σ + 2/ω > 1 for standard PK parameters).
        """
        sigma_v, dv = 0.3, 1.0
        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        eta_hat = {1: np.array([dv * omega_v / (omega_v + sigma_v)])}

        ofv_lap = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
        H = 2.0 / sigma_v + 2.0 / omega_v
        foce_base = _foce_base_at_map(dv, omega_v, sigma_v)
        expected_lap = foce_base + math.log(H)
        assert ofv_lap == pytest.approx(expected_lap, abs=1e-6)


# ---------------------------------------------------------------------------
# FOCE OFV formula tests
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestFOCEOFVFormula:
    """
    Verify FOCE OFV = FOCE_base + log|Ω| at MAP eta.

    The inner objective omits the prior normalization constant, so the
    outer objective contributes log|Ω| but not an extra n_eta·log(2π)
    term after Laplace cancellation.
    """

    @pytest.mark.parametrize(
        "dv,omega_v,sigma_v",
        [
            (0.5, 0.3, 0.2),
            (1.5, 0.4, 0.6),
            (2.0, 1.0, 0.1),
            (-1.0, 0.5, 0.5),
        ],
    )
    def test_foce_ofv_includes_prior_const_at_map(self, dv, omega_v, sigma_v):
        """
        FOCE OFV = FOCE_base + log|Ω| at MAP eta.
        The log|Ω| term ensures large Ω is correctly penalised.
        """
        foce_base = _foce_base_at_map(dv, omega_v, sigma_v)
        prior_const = math.log(omega_v)
        expected = foce_base + prior_const

        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        eta_hat = {1: np.array([dv * omega_v / (omega_v + sigma_v)])}

        ofv = FOCEMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
        assert ofv == pytest.approx(expected, abs=1e-6), (
            f"dv={dv}, omega={omega_v}, sigma={sigma_v}: got {ofv:.6f}, expected {expected:.6f}"
        )

    def test_foce_additivity_over_subjects(self):
        """
        Multi-subject FOCE OFV = Σ per-subject FOCE OFV.
        Verifies that the outer loop correctly sums individual contributions.
        """
        dvs = [0.5, 1.0, -0.5, 1.8]
        omega_v, sigma_v = 0.4, 0.3

        # Multi-subject
        pop_multi = _GaussianPopulation(dvs, sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        eta_hat_multi = {
            i + 1: np.array([dvs[i] * omega_v / (omega_v + sigma_v)]) for i in range(len(dvs))
        }
        ofv_multi = FOCEMethod(interaction=False, maxeval=1)._outer_ofv(
            pop_multi, params, eta_hat_multi
        )

        # Per-subject sum
        total = 0.0
        for _i, dv in enumerate(dvs):
            pop_i = _GaussianPopulation([dv], sigma_v)
            eta_i = {1: np.array([dv * omega_v / (omega_v + sigma_v)])}
            total += FOCEMethod(interaction=False, maxeval=1)._outer_ofv(pop_i, params, eta_i)

        assert ofv_multi == pytest.approx(total, abs=1e-8)

    def test_foce_increases_with_residual(self):
        """OFV strictly increases when |y − ŷ| increases."""
        omega_v, sigma_v, eta_val = 0.5, 0.3, 0.3
        residuals = [0.0, 0.5, 1.0, 2.0]
        ofvs = []
        for resid in residuals:
            dv = eta_val + resid  # so (dv - η)² = resid²
            pop = _GaussianPopulation([dv], sigma_v)
            p = _GaussianParams(omega_v, sigma_v)
            ofvs.append(
                FOCEMethod(interaction=False, maxeval=1)._outer_ofv(
                    pop, p, {1: np.array([eta_val])}
                )
            )
        assert all(o1 < o2 for o1, o2 in zip(ofvs, ofvs[1:], strict=False))


# ---------------------------------------------------------------------------
# IMP large-sample convergence
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestIMPLargeSampleConvergence:
    """
    IMP with very large N_samples converges to −2·analytic_log_marginal.
    This is verifiable exactly for the linear-Gaussian case:
    p(y) = N(y; 0, ω + σ²)  →  −2·log p(y) = log(2π(ω+σ)) + y²/(ω+σ).
    """

    def test_large_isample_matches_analytic(self):
        """
        With isample=50 000, IMP OFV must match analytic value to within 0.02.
        """
        dv, omega_v, sigma_v = 1.25, 0.6, 0.4
        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)

        ofv_imp = IMPMethod(isample=50_000, seed=42)._compute_imp_ofv(pop, params)
        expected = -2.0 * _analytic_log_marginal(dv, omega_v, sigma_v)
        assert ofv_imp == pytest.approx(expected, abs=0.02)

    def test_convergence_improves_with_samples(self):
        """
        |IMP(N=5000) − analytic| < |IMP(N=100) − analytic|: more samples → better.
        """
        dv, omega_v, sigma_v = 2.0, 0.8, 0.5
        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        expected = -2.0 * _analytic_log_marginal(dv, omega_v, sigma_v)

        err_small = abs(IMPMethod(isample=100, seed=0)._compute_imp_ofv(pop, params) - expected)
        err_large = abs(IMPMethod(isample=5_000, seed=0)._compute_imp_ofv(pop, params) - expected)
        assert err_large < err_small

    def test_multi_subject_imp_sums_over_subjects(self):
        """Multi-subject IMP OFV ≈ Σ single-subject analytic for large N."""
        dvs = [0.5, 1.0, 1.5, 2.0]
        omega_v, sigma_v = 0.4, 0.6
        pop = _GaussianPopulation(dvs, sigma_v)
        params = _GaussianParams(omega_v, sigma_v)

        ofv_imp = IMPMethod(isample=20_000, seed=3)._compute_imp_ofv(pop, params)
        expected = sum(-2.0 * _analytic_log_marginal(dv, omega_v, sigma_v) for dv in dvs)
        assert ofv_imp == pytest.approx(expected, abs=0.05)

    def test_ess_implicitly_adequate_via_convergence(self):
        """
        If ESS < 10%, the IMP estimate would have high variance and miss the
        analytic value by >> 0.10.  This test guards against weight collapse
        without directly accessing internal ESS state.
        """
        dv, omega_v, sigma_v = 1.0, 0.5, 0.5
        pop = _GaussianPopulation([dv], sigma_v)
        params = _GaussianParams(omega_v, sigma_v)
        expected = -2.0 * _analytic_log_marginal(dv, omega_v, sigma_v)

        ofv = IMPMethod(isample=1_000, seed=1)._compute_imp_ofv(pop, params)
        assert ofv == pytest.approx(expected, abs=0.10)
