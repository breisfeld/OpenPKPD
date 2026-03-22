"""
External-validation tests for extended pharmacometric model families.

Each section verifies openpkpd outputs against an independent closed-form
reference (scipy.stats, scipy.linalg.expm, or published analytic formulae).
No model fitting is required for the formula-verification tests; fit-based
tests use synthetic data with known ground truth.

References
----------
- Mager DE, Jusko WJ (2001). General pharmacokinetic model for drugs
  exhibiting target-mediated drug disposition. J Pharmacokinet Pharmacodyn.
- Gibiansky L et al. (2008). Approximations of the target-mediated drug
  disposition model and identifiability of model parameters. J Pharmacokinet
  Pharmacodyn.
- Schuirmann DJ (1987). A comparison of the two one-sided tests procedure.
  J Pharmacokinet Biopharm.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import linalg as sp_linalg
from scipy.special import expit
from scipy.stats import (
    expon as scipy_expon,
)
from scipy.stats import (
    nbinom as scipy_nbinom,
)
from scipy.stats import (
    poisson as scipy_poisson,
)
from scipy.stats import (
    weibull_min as scipy_weibull,
)

from openpkpd.data.event_processor import DoseEvent
from openpkpd.models.categorical import (
    ContinuousTimeMarkovModel,
    ProportionalOddsModel,
)
from openpkpd.models.count import (
    CountData,
    NegativeBinomialModel,
    PoissonModel,
    ZeroInflatedPoissonModel,
)
from openpkpd.models.tmdd import FullTMDD, MichaelisMentenTMDD, QSSATMDDModel
from openpkpd.models.tte import ConstantHazardModel, TTEData, WeibullHazardModel

# ===========================================================================
# Section 1 — Time-to-event models
# ===========================================================================


@pytest.mark.external_validation
class TestTTESurvivalClosedForm:
    """Verify survival functions against scipy.stats."""

    @pytest.mark.parametrize(
        "lam,t",
        [
            (0.05, 1.0),
            (0.10, 5.0),
            (0.20, 10.0),
            (0.50, 2.0),
            (1.00, 0.5),
            (0.01, 50.0),
            (0.30, 3.0),
            (0.15, 7.0),
            (2.00, 0.25),
            (0.08, 20.0),
        ],
    )
    def test_constant_hazard_survival_vs_scipy_expon(self, lam, t):
        """
        S(t) = exp(−λ·t) = scipy.stats.expon.sf(t, scale=1/λ)
        Verified to floating-point precision.
        """
        model = ConstantHazardModel()
        s_openpkpd = float(model.survival(t, np.array([lam])))
        s_scipy = scipy_expon.sf(t, scale=1.0 / lam)
        assert s_openpkpd == pytest.approx(s_scipy, rel=1e-12)

    def test_constant_hazard_survival_identity_s_equals_exp_neg_H(self):
        """S(t) = exp(−H(t)) must hold exactly for all valid (λ, t)."""
        model = ConstantHazardModel()
        for lam in [0.05, 0.2, 1.0]:
            for t in [0.1, 1.0, 5.0, 20.0]:
                params = np.array([lam])
                s = float(model.survival(t, params))
                H = float(model.cumulative_hazard(t, params))
                assert s == pytest.approx(math.exp(-H), abs=1e-12)

    @pytest.mark.parametrize(
        "scale,shape,t",
        [
            (10.0, 1.0, 5.0),  # shape=1 → exponential
            (5.0, 2.0, 3.0),  # increasing hazard
            (8.0, 0.5, 4.0),  # decreasing hazard
            (3.0, 3.0, 2.0),  # strongly increasing
            (15.0, 1.5, 10.0),  # mild increase
        ],
    )
    def test_weibull_survival_vs_scipy(self, scale, shape, t):
        """
        S(t) = exp(−(t/scale)^shape) = scipy.stats.weibull_min.sf(t, c=shape, scale=scale)
        Verified to 1e-10 relative tolerance.
        """
        model = WeibullHazardModel()
        params = np.array([scale, shape])
        s_openpkpd = float(model.survival(t, params))
        s_scipy = scipy_weibull.sf(t, c=shape, scale=scale)
        assert s_openpkpd == pytest.approx(s_scipy, rel=1e-10)

    def test_weibull_survival_identity_s_equals_exp_neg_H(self):
        """S(t) = exp(−(t/scale)^shape) identity holds exactly."""
        model = WeibullHazardModel()
        for scale, shape in [(5.0, 2.0), (10.0, 0.7), (2.0, 3.0)]:
            for t in [0.5, 2.0, 8.0]:
                params = np.array([scale, shape])
                s = float(model.survival(t, params))
                H_ref = (t / scale) ** shape
                assert s == pytest.approx(math.exp(-H_ref), abs=1e-12)

    def test_weibull_at_shape_one_equals_exponential(self):
        """Weibull(scale, 1) ≡ Exponential(1/scale)."""
        scale = 7.0
        weibull = WeibullHazardModel()
        constant = ConstantHazardModel()
        for t in [1.0, 5.0, 10.0, 25.0]:
            s_w = float(weibull.survival(t, np.array([scale, 1.0])))
            s_e = float(constant.survival(t, np.array([1.0 / scale])))
            assert s_w == pytest.approx(s_e, abs=1e-12)

    def test_constant_hazard_fit_recovers_lambda(self):
        """
        Fitting to exponential data should recover λ within 20%.
        Exponential(-λ) survival times via inverse-CDF sampling.
        """
        rng = np.random.default_rng(42)
        true_lam = 0.15
        n = 200
        times = rng.exponential(1.0 / true_lam, size=n)
        data = [
            TTEData(subject_id=i, event_times=np.array([times[i]]), event_indicator=np.array([1]))
            for i in range(n)
        ]
        result = ConstantHazardModel().fit(data, np.array([0.1]))
        fitted_lam = float(result.hazard_params[0])
        assert fitted_lam == pytest.approx(true_lam, rel=0.15)


# ===========================================================================
# Section 2 — Count data models
# ===========================================================================


@pytest.mark.external_validation
class TestCountModelClosedForm:
    """Verify count model PMFs against scipy.stats."""

    @pytest.mark.parametrize(
        "k,mu",
        [
            (0, 0.5),
            (0, 2.0),
            (0, 10.0),
            (1, 1.0),
            (2, 3.0),
            (5, 3.0),
            (10, 5.0),
            (3, 8.0),
            (0, 20.0),
            (15, 10.0),
        ],
    )
    def test_poisson_log_pmf_vs_scipy(self, k, mu):
        """log P(Y=k | Poisson(μ)) matches scipy.stats.poisson.logpmf to 1e-10."""
        model = PoissonModel()
        assert model.log_pmf(k, mu) == pytest.approx(scipy_poisson.logpmf(k, mu), abs=1e-10)

    def test_poisson_k0_identity(self):
        """P(Y=0 | Poisson(μ)) = exp(−μ) exactly."""
        model = PoissonModel()
        for mu in [0.1, 1.0, 5.0, 20.0]:
            assert model.log_pmf(0, mu) == pytest.approx(-mu, abs=1e-12)

    def test_poisson_additivity(self):
        """P(Y=k | Poisson(μ)) probabilities sum to 1 over k=0..K."""
        model = PoissonModel()
        mu = 3.0
        total = sum(math.exp(model.log_pmf(k, mu)) for k in range(30))
        assert total == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "k,mu,r",
        [
            (0, 2.0, 1.0),
            (1, 2.0, 1.0),
            (3, 2.0, 5.0),
            (0, 5.0, 2.0),
            (5, 3.0, 3.0),
        ],
    )
    def test_negbin_log_pmf_vs_scipy(self, k, mu, r):
        """
        log P(Y=k | NegBin(r, μ)) matches scipy.stats.nbinom.logpmf to 1e-8.
        scipy parameterisation: nbinom(n=r, p=r/(r+μ))
        """
        model = NegativeBinomialModel(r=r)
        p = r / (r + mu)
        expected = scipy_nbinom.logpmf(k, n=r, p=p)
        assert model.log_pmf(k, mu, r=r) == pytest.approx(expected, abs=1e-8)

    def test_negbin_large_r_approaches_poisson(self):
        """NegBin(r→∞, μ) → Poisson(μ): agree to 0.001 at r=1e6."""
        nb = NegativeBinomialModel(r=1e6)
        pois = PoissonModel()
        for k, mu in [(0, 2.0), (2, 3.0), (5, 3.0), (10, 5.0)]:
            assert nb.log_pmf(k, mu, r=1e6) == pytest.approx(pois.log_pmf(k, mu), abs=1e-3)

    @pytest.mark.parametrize(
        "pi,lam",
        [
            (0.3, 2.0),
            (0.0, 3.0),
            (1.0, 1.5),
            (0.5, 5.0),
            (0.1, 0.5),
        ],
    )
    def test_zip_k0_identity(self, pi, lam):
        """P(Y=0 | ZIP(π, λ)) = π + (1−π)·exp(−λ) exactly."""
        model = ZeroInflatedPoissonModel(zero_prob=pi)
        expected_log = math.log(pi + (1.0 - pi) * math.exp(-lam))
        observed = model.log_pmf(0, lam, zero_prob=pi)
        assert observed == pytest.approx(expected_log, abs=1e-10)

    def test_zip_k_positive_equals_scaled_poisson(self):
        """For k > 0: P(Y=k | ZIP(π, λ)) = (1−π)·P(Y=k | Poisson(λ))."""
        model = ZeroInflatedPoissonModel(zero_prob=0.3)
        pois = PoissonModel()
        pi, lam = 0.3, 2.5
        for k in [1, 2, 5, 10]:
            expected = math.log(1.0 - pi) + pois.log_pmf(k, lam)
            assert model.log_pmf(k, lam, zero_prob=pi) == pytest.approx(expected, abs=1e-10)

    def test_poisson_fit_recovers_rate(self):
        """Fitting Poisson to simulated data recovers true rate within 15%."""
        rng = np.random.default_rng(7)
        true_mu = 4.5
        counts = rng.poisson(true_mu, size=300)
        data = [
            CountData(subject_id=i, counts=np.array([counts[i]]), times=np.array([0.0]))
            for i in range(len(counts))
        ]
        result = PoissonModel().fit(data, np.array([1.0]))
        assert math.exp(result.rate_params[0]) == pytest.approx(true_mu, rel=0.15)


# ===========================================================================
# Section 3 — Categorical / Markov models
# ===========================================================================


@pytest.mark.external_validation
class TestProportionalOddsReference:
    """
    Verify proportional-odds model against cumulative logit formula.
    P(Y ≤ k | x) = expit(α_k − x·β)   (Agresti 2002, Eq. 7.3)
    """

    @pytest.mark.parametrize("eta", [-2.0, -1.0, 0.0, 0.5, 1.5])
    def test_cumulative_logit_formula(self, eta):
        """P(Y ≤ k) = expit(α_k − eta) exactly for every threshold k."""
        model = ProportionalOddsModel(n_categories=4)
        # 3 raw thresholds (params[0], params[1..2] are log-delta), 1 slope
        alpha0 = -1.0
        delta1 = 1.2
        delta2 = 0.8
        params = np.array([alpha0, math.log(delta1), math.log(delta2), 1.0])
        cov = np.array([[eta]])  # x = eta so x*slope = eta

        probs = model.predict_probs(cov, params)
        assert probs.shape == (1, 4)

        alpha = np.array([alpha0, alpha0 + delta1, alpha0 + delta1 + delta2])
        cum = expit(alpha - eta)
        expected = np.diff(np.concatenate([[0.0], cum, [1.0]]))
        np.testing.assert_allclose(probs[0], expected, atol=1e-10)

    def test_probabilities_sum_to_one(self):
        """Probabilities across all K categories must sum to 1."""
        model = ProportionalOddsModel(n_categories=5)
        params = np.array([0.0, 0.5, 0.5, 0.5, 0.8])
        cov = np.array([[0.0], [1.0], [-1.0]])
        probs = model.predict_probs(cov, params)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_positive_beta_shifts_to_higher_categories(self):
        """Higher x (positive β) should increase P(Y > k) for all k."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([-0.5, 0.8, 0.5])  # positive slope
        p_low = model.predict_probs(np.array([[0.0]]), params)[0]
        p_high = model.predict_probs(np.array([[3.0]]), params)[0]
        assert p_high[-1] > p_low[-1]  # higher probability of highest category


@pytest.mark.external_validation
class TestCTMCTransitionMatrixReference:
    """
    Verify ContinuousTimeMarkovModel.transition_matrix against scipy.linalg.expm.
    P(t) = expm(Q·t)  where Q is the generator matrix with log-parameterised rates.
    """

    def _build_Q_2state(self, q01: float, q10: float) -> np.ndarray:
        return np.array([[-q01, q01], [q10, -q10]])

    def _build_Q_3state(self, rates: np.ndarray) -> np.ndarray:
        """rates = [q01, q02, q10, q12, q20, q21]."""
        Q = np.zeros((3, 3))
        off_diag = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        for idx, (i, j) in enumerate(off_diag):
            Q[i, j] = rates[idx]
        for i in range(3):
            Q[i, i] = -Q[i, :].sum()
        return Q

    @pytest.mark.parametrize(
        "q01,q10,t",
        [
            (0.10, 0.05, 1.0),
            (0.40, 0.20, 0.5),
            (1.0, 0.5, 2.0),
            (0.02, 0.08, 10.0),
            (0.5, 0.5, 0.1),
        ],
    )
    def test_2state_matches_expm(self, q01, q10, t):
        """P(t) = expm(Q·t) to 1e-10 for 2-state CTMC."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.log(np.array([q01, q10]))
        P_model = model.transition_matrix(t=t, params=params)
        Q = self._build_Q_2state(q01, q10)
        P_ref = sp_linalg.expm(Q * t)
        np.testing.assert_allclose(P_model, P_ref, atol=1e-10)

    def test_3state_matches_expm(self):
        """P(t) = expm(Q·t) to 1e-10 for 3-state CTMC."""
        rates = np.array([0.2, 0.1, 0.3, 0.15, 0.05, 0.25])
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.log(rates)
        P_model = model.transition_matrix(t=1.5, params=params)
        Q = self._build_Q_3state(rates)
        P_ref = sp_linalg.expm(Q * 1.5)
        np.testing.assert_allclose(P_model, P_ref, atol=1e-10)

    def test_chapman_kolmogorov(self):
        """P(s+t) = P(s)·P(t): Chapman-Kolmogorov equation."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.log(np.array([0.3, 0.1]))
        s, t = 1.0, 2.0
        Ps = model.transition_matrix(t=s, params=params)
        Pt = model.transition_matrix(t=t, params=params)
        Pst = model.transition_matrix(t=s + t, params=params)
        np.testing.assert_allclose(Ps @ Pt, Pst, atol=1e-10)

    def test_rows_sum_to_one(self):
        """P(t) must be a right-stochastic matrix (rows sum to 1)."""
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.full(6, -1.5)
        P = model.transition_matrix(t=2.0, params=params)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)


# ===========================================================================
# Section 4 — TMDD models
# ===========================================================================


@pytest.mark.external_validation
class TestTMDDReference:
    """
    Verify TMDD model limiting behaviours against published approximations.
    Reference: Mager & Jusko (2001), Gibiansky et al. (2008).
    """

    _OBS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    _DOSE = 100.0

    def _iv(self, amt=None):
        return [DoseEvent(time=0.0, amount=amt or self._DOSE, compartment=1)]

    def test_full_tmdd_reduces_to_advan1_when_binding_disabled(self):
        """
        When kon=0 (no target binding), FullTMDD drug compartment equals
        ADVAN1 with K = CL/V.  Verified to 1e-6.
        """
        from openpkpd.pk.analytical.advan1 import ADVAN1

        params = {
            "CL": 0.5,
            "V": 5.0,
            "kon": 0.0,
            "koff": 0.01,
            "kint": 0.2,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol_tmdd = FullTMDD().solve(params, self._iv(), self._OBS)
        sol_ref = ADVAN1().solve(
            {"K": params["CL"] / params["V"], "V": params["V"]}, self._iv(), self._OBS
        )
        np.testing.assert_allclose(sol_tmdd.ipred, sol_ref.ipred, rtol=1e-6)

    def test_qssa_reduces_to_advan1_when_target_elimination_disabled(self):
        """
        When kint=0, QSSATMDDModel target-mediated elimination is switched off;
        drug elimination is purely linear (ADVAN1).
        """
        from openpkpd.pk.analytical.advan1 import ADVAN1

        params = {"CL": 1.0, "V": 10.0, "Kss": 1.0, "kint": 0.0, "Ksyn": 0.5, "Kdeg": 0.1}
        sol_qssa = QSSATMDDModel().solve(params, self._iv(), self._OBS)
        sol_ref = ADVAN1().solve(
            {"K": params["CL"] / params["V"], "V": params["V"]}, self._iv(), self._OBS
        )
        np.testing.assert_allclose(sol_qssa.ipred, sol_ref.ipred, rtol=1e-6)

    def test_michaelis_menten_low_c_approaches_linear(self):
        """
        At very low drug concentrations (C << Km), MM elimination is linear:
        clearance ≈ Vmax/Km.  Verify that low-dose MM OFV ≈ ADVAN1 with
        effective K = CL/V + Vmax/(Km·V).
        """
        from openpkpd.pk.analytical.advan1 import ADVAN1

        Vmax, Km, CL, V = 0.01, 100.0, 0.0, 10.0
        k_eff = CL / V + Vmax / (Km * V)
        low_dose = 0.1  # C_max = 0.01 << Km = 100

        sol_mm = MichaelisMentenTMDD().solve(
            {"CL": CL, "V": V, "Vmax": Vmax, "Km": Km},
            self._iv(amt=low_dose),
            self._OBS,
        )
        sol_lin = ADVAN1().solve({"K": k_eff, "V": V}, self._iv(amt=low_dose), self._OBS)
        np.testing.assert_allclose(sol_mm.ipred, sol_lin.ipred, rtol=1e-3)

    def test_michaelis_menten_high_dose_nonlinear(self):
        """
        Nonlinearity check: at high concentration, effective CL (Vmax/(Km+C))
        is lower than at low concentration (Vmax/Km), so high-dose elimination
        is slower (normalised concentration declines less steeply early on).
        """
        Vmax, Km, V = 2.0, 5.0, 10.0
        params = {"CL": 0.0, "V": V, "Vmax": Vmax, "Km": Km}
        t_early = np.array([0.5, 1.0])

        sol_lo = MichaelisMentenTMDD().solve(params, self._iv(amt=1.0), t_early)
        sol_hi = MichaelisMentenTMDD().solve(params, self._iv(amt=100.0), t_early)

        # Normalise to initial concentration to compare elimination rate
        c0_lo = 1.0 / V
        c0_hi = 100.0 / V
        frac_lo = sol_lo.ipred / c0_lo
        frac_hi = sol_hi.ipred / c0_hi

        # High-dose fraction remaining should be higher (slower normalised elimination)
        assert frac_hi[0] > frac_lo[0]

    def test_full_tmdd_mass_conservation(self):
        """
        Total drug (free + complex) must be non-increasing after the dose
        (no creation of drug).  Also, all compartments must be non-negative.
        """
        params = {
            "CL": 0.5,
            "V": 5.0,
            "kon": 0.1,
            "koff": 0.01,
            "kint": 0.2,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol = FullTMDD().solve(params, self._iv(), self._OBS)
        assert np.all(sol.amounts >= -1e-6), "Compartment amounts must be non-negative"
        # Drug amount in cmt 1 + complex in cmt 3 should decrease over time
        total_drug = sol.amounts[:, 0] + sol.amounts[:, 2]
        assert total_drug[0] >= total_drug[-1], "Total drug must not increase"
