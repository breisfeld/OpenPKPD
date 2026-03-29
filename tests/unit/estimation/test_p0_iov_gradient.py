"""
P0.6 — IOV gradient tests.

Verify that _compute_G_i correctly reflects the block-sparse structure of
inter-occasion variability (IOV) ETAs:

  ETA layout for a 2-occasion model with BSV on CL and IOV on CL:
    eta = [η_bsv_CL,  κ_occ1_CL,  κ_occ2_CL]

  Predictions for occasion 1 should depend on eta[0] and eta[1] but NOT eta[2].
  Predictions for occasion 2 should depend on eta[0] and eta[2] but NOT eta[1].

  Therefore the Jacobian G_i = ∂IPRED/∂η must satisfy:
    G_i[obs_occ1, 2] ≈ 0    (occ1 obs insensitive to κ_occ2)
    G_i[obs_occ2, 1] ≈ 0    (occ2 obs insensitive to κ_occ1)
    G_i[obs_occ1, 0] ≠ 0    (all obs sensitive to BSV eta)
    G_i[obs_occ2, 0] ≠ 0

External validation reference
------------------------------
NONMEM users guide (2009), Chapter 8: "OMEGA blocks".
IOV ETA gradients are zero for unrelated occasions by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.foce import _compute_G_i


# ─────────────────────────────────────────────────────────────────────────────
# IOV individual-model mock
# ─────────────────────────────────────────────────────────────────────────────


class _IOVIndividualModel:
    """
    2-occasion PK model.  Observation times:
      occ1: [1.0, 2.0]   (indices 0, 1)
      occ2: [5.0, 8.0]   (indices 2, 3)

    IPRED(occ=1, t) = dose * exp(-CL_occ1/V * t) / V
    IPRED(occ=2, t) = dose * exp(-CL_occ2/V * t) / V

    CL_occ1 = CL_pop * exp(eta[0] + eta[1])   (BSV + IOV_occ1)
    CL_occ2 = CL_pop * exp(eta[0] + eta[2])   (BSV + IOV_occ2)
    V = 10, CL_pop = 1, dose = 100
    """

    CL_POP = 1.0
    V = 10.0
    DOSE = 100.0

    # Observation times by occasion
    OCC1_TIMES = np.array([1.0, 2.0])
    OCC2_TIMES = np.array([5.0, 8.0])

    def _ipred_occ(self, t_arr: np.ndarray, cl: float) -> np.ndarray:
        return self.DOSE * np.exp(-cl / self.V * t_arr) / self.V

    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = np.asarray(eta, dtype=float)
        cl_occ1 = self.CL_POP * np.exp(float(eta[0]) + float(eta[1]))
        cl_occ2 = self.CL_POP * np.exp(float(eta[0]) + float(eta[2]))

        pred = np.concatenate([
            self._ipred_occ(self.OCC1_TIMES, cl_occ1),
            self._ipred_occ(self.OCC2_TIMES, cl_occ2),
        ])
        obs_mask = np.ones(len(pred), dtype=bool)
        var = np.full(len(pred), 0.1)
        return pred, obs_mask, pred, pred, var


# ─────────────────────────────────────────────────────────────────────────────
# Compute reference G_i
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def iov_G_i():
    """Pre-compute G_i for the 2-occasion IOV model at eta = [0, 0, 0]."""
    indiv = _IOVIndividualModel()
    eta = np.zeros(3)
    theta = np.array([])
    sigma = np.eye(1) * 0.1

    # Nominal prediction
    _, obs_mask, _, pred0, _ = indiv.evaluate_observation_model(theta, eta, sigma)

    G = _compute_G_i(indiv, theta, eta, sigma, 2, obs_mask, pred0[obs_mask], h=1e-5)
    return G   # shape (4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIOVGradientShape:
    def test_shape_is_n_obs_by_n_eta(self, iov_G_i):
        assert iov_G_i.shape == (4, 3), f"Expected (4,3), got {iov_G_i.shape}"


class TestIOVGradientSparsity:
    """The most important test: IOV ETAs affect only their own occasion."""

    def test_occ1_obs_insensitive_to_kappa_occ2(self, iov_G_i):
        """
        G_i[0:2, 2] ≈ 0  (occ1 observations, κ_occ2 ETA column).
        Tolerance: 1 % of the maximum gradient magnitude.
        """
        G = iov_G_i
        atol = 0.01 * np.max(np.abs(G))
        np.testing.assert_allclose(
            G[0:2, 2], 0.0, atol=atol,
            err_msg=f"Occ1 obs should be insensitive to κ_occ2; got G[0:2,2]={G[0:2,2]}"
        )

    def test_occ2_obs_insensitive_to_kappa_occ1(self, iov_G_i):
        """G_i[2:4, 1] ≈ 0  (occ2 observations, κ_occ1 ETA column)."""
        G = iov_G_i
        atol = 0.01 * np.max(np.abs(G))
        np.testing.assert_allclose(
            G[2:4, 1], 0.0, atol=atol,
            err_msg=f"Occ2 obs should be insensitive to κ_occ1; got G[2:4,1]={G[2:4,1]}"
        )

    def test_occ1_obs_sensitive_to_kappa_occ1(self, iov_G_i):
        """G_i[0:2, 1] should be nonzero (occ1 obs sensitive to their own IOV ETA)."""
        G = iov_G_i
        assert np.all(np.abs(G[0:2, 1]) > 1e-6), \
            f"Occ1 obs must depend on κ_occ1; got {G[0:2,1]}"

    def test_occ2_obs_sensitive_to_kappa_occ2(self, iov_G_i):
        """G_i[2:4, 2] should be nonzero (occ2 obs sensitive to their own IOV ETA)."""
        G = iov_G_i
        assert np.all(np.abs(G[2:4, 2]) > 1e-6), \
            f"Occ2 obs must depend on κ_occ2; got {G[2:4,2]}"

    def test_all_obs_sensitive_to_bsv_eta(self, iov_G_i):
        """All observations must depend on the BSV (between-subject) ETA."""
        G = iov_G_i
        assert np.all(np.abs(G[:, 0]) > 1e-6), \
            f"All obs must depend on BSV ETA; got col 0 = {G[:,0]}"


class TestIOVGradientAnalyticValues:
    """
    Analytic check: G_i[0, 0] for the occ1 model at eta=0, t=1.

    IPRED(t) = DOSE/V * exp(-CL_pop*exp(η_bsv+κ)/V * t)
    ∂IPRED/∂η_bsv = IPRED * (-CL_pop*exp(η_bsv+κ)/V * t)
    At η=0: = IPRED * (-CL_pop/V * t) = (DOSE/V*exp(-CL_pop/V)) * (-CL_pop/V)
    """

    def test_bsv_gradient_occ1_t1(self, iov_G_i):
        CL, V, DOSE, t = 1.0, 10.0, 100.0, 1.0
        ipred0 = DOSE / V * np.exp(-CL / V * t)
        expected = ipred0 * (-CL / V * t)
        np.testing.assert_allclose(
            iov_G_i[0, 0], expected, rtol=1e-2,
            err_msg=f"BSV gradient at occ1, t=1: expected {expected:.4f}, got {iov_G_i[0,0]:.4f}"
        )

    def test_iov_gradient_occ1_t1_equals_bsv_gradient(self, iov_G_i):
        """
        At eta=0, ∂IPRED/∂κ_occ1 = ∂IPRED/∂η_bsv because both appear only
        as sum (η_bsv + κ_occ1) in CL_occ1.
        """
        np.testing.assert_allclose(
            iov_G_i[0, 0], iov_G_i[0, 1], rtol=1e-2,
            err_msg="At eta=0, BSV and IOV gradients for occ1 must be equal"
        )
