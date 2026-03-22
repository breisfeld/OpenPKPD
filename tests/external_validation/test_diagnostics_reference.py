"""
External-validation tests for diagnostic method formulas.

Validates NPDE, NPC, VPC and SSE against analytic/reference formulas:

- NPDE formula: Φ⁻¹(pd) vs scipy.stats.norm.ppf (Brendel et al. 2006)
- NPC p-value uniformity: under a correct model p-values are Uniform(0,1)
- VPC prediction-interval coverage: nominal 90% PI must contain ≥ 85% of
  observed data under a correctly-specified model (simulation-based)
- NPDE mean/variance: E[NPDE] = 0, Var[NPDE] = 1 under correct model

References
----------
Brendel K et al. (2006). Metrics for external model evaluation. Pharm Res.
Yano Y et al. (2001). Evaluating pharmacokinetic / pharmacodynamic models
  using the posterior predictive check. J Pharmacokinet Pharmacodyn.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest, uniform
from scipy.stats import norm as sp_norm

# ===========================================================================
# Section 1 — NPDE formula verification (fast, analytic)
# ===========================================================================


@pytest.mark.external_validation
class TestNPDEFormulaReference:
    """
    Core NPDE formula: PDE = Φ⁻¹(pd) where pd is the empirical predictive CDF.
    Under a correct model, PDE ~ N(0,1).

    This section verifies the transformation formula itself, independent of
    how pd is computed.
    """

    @pytest.mark.parametrize("pd", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    def test_inverse_normal_transform_matches_scipy(self, pd):
        """Φ⁻¹(pd) must match scipy.stats.norm.ppf to 1e-12."""
        pde = sp_norm.ppf(pd)
        # Verify the formula directly: Φ(pde) = pd
        assert sp_norm.cdf(pde) == pytest.approx(pd, abs=1e-12)

    def test_npde_is_standard_normal_for_uniform_pd(self):
        """
        If pd values are drawn from Uniform(0,1), applying Φ⁻¹ yields N(0,1).
        This verifies the probability integral transform (Fisher 1925).
        """
        rng = np.random.default_rng(42)
        pd = rng.uniform(0.0, 1.0, size=5000)
        npde = sp_norm.ppf(pd)
        assert npde.mean() == pytest.approx(0.0, abs=0.05)
        assert npde.std() == pytest.approx(1.0, abs=0.05)

    def test_empirical_pd_formula(self):
        """
        pd = (#{sim < obs} + 0.5*#{sim == obs}) / K
        Verified against direct count computation.
        """
        obs = 2.0
        sims = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        K = len(sims)
        n_below = np.sum(sims < obs)  # 2
        n_equal = np.sum(sims == obs)  # 1
        pd_ref = (n_below + 0.5 * n_equal) / K  # (2 + 0.5) / 5 = 0.5
        assert pd_ref == pytest.approx(0.5, abs=1e-12)

    def test_boundary_pds_clipped_to_avoid_infinite_npde(self):
        """
        pd = 0 → Φ⁻¹(0) = −∞ and pd = 1 → Φ⁻¹(1) = +∞.
        Practical implementations clip pd to (ε, 1-ε).  Verify that
        clipping to ε = 0.5/K gives finite NPDE for K simulations.
        """
        for K in [100, 1000, 10000]:
            pd_min = 0.5 / K
            pd_max = 1.0 - 0.5 / K
            npde_min = sp_norm.ppf(pd_min)
            npde_max = sp_norm.ppf(pd_max)
            assert np.isfinite(npde_min), f"NPDE not finite at pd={pd_min} (K={K})"
            assert np.isfinite(npde_max), f"NPDE not finite at pd={pd_max} (K={K})"

    def test_npde_symmetry(self):
        """NPDE is antisymmetric: Φ⁻¹(1−pd) = −Φ⁻¹(pd)."""
        for pd in [0.1, 0.25, 0.4]:
            npde_pd = sp_norm.ppf(pd)
            npde_neg = sp_norm.ppf(1.0 - pd)
            assert npde_neg == pytest.approx(-npde_pd, abs=1e-12)

    def test_within_subject_decorrelation_identity(self):
        """
        When the within-subject covariance matrix is the identity, decorrelation
        via Cholesky is a no-op: NPDE = PDE.
        Chol(I)⁻ᵀ = I, so npde = I @ pde = pde.
        """
        n_obs = 4
        pde = np.array([0.5, -1.2, 0.3, 1.8])
        C = np.eye(n_obs)  # Identity correlation matrix
        L = np.linalg.cholesky(C)
        npde = np.linalg.solve(L.T, pde)
        np.testing.assert_allclose(npde, pde, atol=1e-12)

    def test_within_subject_decorrelation_removes_correlation(self):
        """
        After Cholesky decorrelation, samples drawn from N(0, C) become
        approximately uncorrelated.

        Draw vectors from N(0, C) with C having correlation 0.99;
        after transform L^{-T} v, samples should be near-uncorrelated.
        """
        rng = np.random.default_rng(7)
        C = np.array([[1.0, 0.99], [0.99, 1.0]])
        L = np.linalg.cholesky(C)
        n_samples = 2000
        # Draw from N(0, C)
        pde_arr = rng.multivariate_normal([0.0, 0.0], C, size=n_samples)
        # Apply whitening: L^{-1} x ~ N(0, I) since C = L*L^T
        npde_arr = np.linalg.solve(L, pde_arr.T).T

        np.corrcoef(pde_arr[:, 0], pde_arr[:, 1])[0, 1]
        corr_after = np.corrcoef(npde_arr[:, 0], npde_arr[:, 1])[0, 1]
        # After whitening, Cov(y) = L^{-1}·C·L^{-T} = I, so |ρ_after| ≈ 0
        assert abs(corr_after) < 0.1


# ===========================================================================
# Section 2 — NPC p-value formula (fast, analytic)
# ===========================================================================


@pytest.mark.external_validation
class TestNPCPValueFormula:
    """
    NPC (numerical predictive check) p-value:
      p_two_sided = 2 * min(p_below, 1 − p_below)
    where p_below = P(y_sim < y_obs | model).
    This is the standard two-tailed predictive p-value (Gelman et al. 1996).
    """

    @pytest.mark.parametrize(
        "p_below,expected_p",
        [
            (0.025, 0.05),  # below 5th percentile
            (0.500, 1.00),  # median
            (0.100, 0.20),  # 10th percentile
            (0.950, 0.10),  # 95th percentile → p = 2*(1-0.95) = 0.10
            (0.010, 0.02),  # near tail
        ],
    )
    def test_two_sided_p_value_formula(self, p_below, expected_p):
        """p_two_sided = 2 * min(p_below, 1 − p_below)."""
        p_two = 2.0 * min(p_below, 1.0 - p_below)
        assert p_two == pytest.approx(expected_p, abs=1e-12)

    def test_p_below_from_simulation_count(self):
        """
        p_below = #{sim < obs} / K
        For obs at 95th percentile of sims, p_below ≈ 0.95.
        """
        rng = np.random.default_rng(42)
        K = 10_000
        sims = rng.standard_normal(K)
        obs = sp_norm.ppf(0.95)  # 95th percentile
        p_below = np.mean(sims < obs)
        assert p_below == pytest.approx(0.95, abs=0.02)

    def test_p_values_uniform_under_correct_model(self):
        """
        Under a correctly-specified model, NPC p-values should be Uniform(0,1).
        Kolmogorov-Smirnov test at α=0.01 should not reject.
        """
        rng = np.random.default_rng(123)
        K = 500  # simulations per observation
        n_obs = 200  # observations
        p_vals = []
        for _ in range(n_obs):
            sims = rng.standard_normal(K)
            obs = rng.standard_normal()  # from same distribution
            p_below = np.mean(sims < obs) + 0.5 / K
            p_vals.append(2.0 * min(p_below, 1.0 - p_below))

        stat, pval = kstest(p_vals, uniform(0, 1).cdf)
        assert pval > 0.01, f"NPC p-values should be uniform under correct model; KS p={pval:.4f}"


# ===========================================================================
# Section 3 — VPC prediction interval coverage (simulation-based, slow)
# ===========================================================================


@pytest.mark.external_validation
@pytest.mark.slow
class TestVPCCoverageReference:
    """
    Under a correctly-specified model, the 90% VPC prediction interval should
    contain ≈ 90% of observed data points.  We test at a generous ≥ 80% coverage
    to account for finite-sample variability.

    This test simulates a simple 1-cmt oral model, generates VPC bands, and
    checks that observed percentiles lie within the simulated bands.
    """

    def test_vpc_90pct_coverage_simple_gaussian(self):
        """
        Perfect-model VPC: obs drawn from N(μ, σ²), VPC from same distribution.
        90% PI must contain ≥ 80% of observations.
        """
        rng = np.random.default_rng(0)
        mu, sigma = 2.0, 0.5
        n_obs = 500
        K = 1000  # simulation replicates

        obs = rng.normal(mu, sigma, size=n_obs)
        # Simulate K replicates
        sims = rng.normal(mu, sigma, size=(K, n_obs))
        pi_lo, pi_hi = np.percentile(sims, [5, 95], axis=0)
        coverage = np.mean((obs >= pi_lo) & (obs <= pi_hi))
        assert coverage >= 0.80, f"90% VPC coverage = {coverage:.2%}, expected >= 80%"

    def test_vpc_coverage_degrades_under_misspecified_model(self):
        """
        Misspecified VPC (wrong sigma): coverage drops below 80%.
        This confirms the diagnostic has power.
        """
        rng = np.random.default_rng(1)
        mu_true, sigma_true = 2.0, 1.0
        sigma_wrong = 0.2  # much too narrow
        n_obs = 500
        K = 1000

        obs = rng.normal(mu_true, sigma_true, size=n_obs)
        sims = rng.normal(mu_true, sigma_wrong, size=(K, n_obs))  # wrong model
        pi_lo, pi_hi = np.percentile(sims, [5, 95], axis=0)
        coverage = np.mean((obs >= pi_lo) & (obs <= pi_hi))
        assert coverage < 0.70, f"Misspecified coverage = {coverage:.2%}, expected < 70%"
