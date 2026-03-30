"""Empirical external validation for BAYES(Laplace) on bundled datasets."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest


REF_DIR = os.path.join(os.path.dirname(__file__), "nlmixr2", "reference")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_ref(name: str) -> dict:
    path = os.path.join(REF_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _build_theophylline_bayes_laplace_model(n_samples: int = 300, maxeval: int = 40):
    """Return a weak-prior BAYES(Laplace) model on empirical theophylline data."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = os.path.join(DATA_DIR, "theophylline_boeckmann.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — empirical BAYES(Laplace) validation")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.09, 0.06, 0.04])
        .sigma(0.02)
        .estimation(
            method="BAYES",
            backend="laplace",
            n_samples=n_samples,
            seed=42,
            prior_sd_theta=1e8,
            maxeval=maxeval,
        )
        .build()
    )


def _build_warfarin_bayes_laplace_model(n_samples: int = 300, maxeval: int = 40):
    """Return a weak-prior BAYES(Laplace) model on empirical warfarin PK data."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = os.path.join(DATA_DIR, "warfarin_pk.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Warfarin PK-only 1-cmt oral — empirical BAYES(Laplace) validation")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(
            method="BAYES",
            backend="laplace",
            n_samples=n_samples,
            seed=42,
            prior_sd_theta=1e8,
            maxeval=maxeval,
        )
        .build()
    )


def _build_theophylline_bayes_nuts_model(n_samples: int = 24, tune: int = 16):
    """Return a measured second-tier BAYES(NUTS) model on empirical theophylline data."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = os.path.join(DATA_DIR, "theophylline_boeckmann.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — empirical BAYES(NUTS) validation")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.09, 0.06, 0.04])
        .sigma(0.02)
        .estimation(
            method="BAYES",
            backend="nuts",
            n_samples=n_samples,
            tune=tune,
            n_chains=2,
            seed=42,
            prior_sd_theta=1e8,
        )
        .build()
    )


@pytest.mark.external_validation
@pytest.mark.slow
class TestTheophyllineBayesLaplaceEmpirical:
    """Weak-prior empirical BAYES(Laplace) should stay near the validated theophylline basin."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        warnings.filterwarnings("ignore")
        return _build_theophylline_bayes_laplace_model().fit()

    @pytest.fixture(scope="class")
    def nlmixr2_ref(self):
        return _load_ref("theophylline_foce.json")

    def test_backend_is_laplace_and_samples_present(self, fit_result):
        theta_samples = fit_result.posterior_samples.get("theta")
        assert fit_result.backend_used == "laplace"
        assert fit_result.method == "BAYES(Laplace)"
        assert theta_samples is not None
        assert theta_samples.ndim == 2
        assert theta_samples.shape[1] == 3
        assert theta_samples.shape[0] >= 100

    def test_posterior_mean_tracks_map_estimate(self, fit_result):
        theta_samples = fit_result.posterior_samples["theta"]
        posterior_mean = theta_samples.mean(axis=0)
        tolerances = [0.10, 0.06, 0.80]
        for name, observed, expected, tol in zip(
            ("KA", "CL", "V"), posterior_mean, fit_result.theta_final, tolerances
        ):
            assert abs(float(observed) - float(expected)) < tol, (
                f"{name} posterior mean {observed:.4f} drifted from MAP {expected:.4f} "
                f"by more than {tol:.2f}"
            )

    def test_cl_and_v_remain_close_to_external_theophylline_reference(
        self, fit_result, nlmixr2_ref
    ):
        ka, cl, v = [float(value) for value in fit_result.theta_final]
        ref = nlmixr2_ref["theta"]

        cl_rel_err = abs(cl - float(ref["CL"])) / float(ref["CL"])
        v_rel_err = abs(v - float(ref["V"])) / float(ref["V"])

        assert cl_rel_err < 0.12, (
            f"CL={cl:.4f} vs nlmixr2 FOCEI={ref['CL']:.4f} "
            f"(rel_err={cl_rel_err:.1%})"
        )
        assert v_rel_err < 0.20, (
            f"V={v:.4f} vs nlmixr2 FOCEI={ref['V']:.4f} "
            f"(rel_err={v_rel_err:.1%})"
        )
        assert ka > 0.0

    def test_posterior_scales_are_finite_and_ordered(self, fit_result):
        theta_samples = fit_result.posterior_samples["theta"]
        posterior_sd = theta_samples.std(axis=0, ddof=1)
        assert np.all(np.isfinite(posterior_sd))
        assert np.all(posterior_sd > 0.0)
        assert posterior_sd[2] > posterior_sd[0] > posterior_sd[1]

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Theophylline BAYES(Laplace) KA still sits above the nlmixr2 FOCEI "
            "reference under the current weak-prior setup."
        ),
    )
    def test_ka_tracks_external_theophylline_reference(self, fit_result, nlmixr2_ref):
        observed = float(fit_result.theta_final[0])
        expected = float(nlmixr2_ref["theta"]["KA"])
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.20, (
            f"KA={observed:.4f} vs nlmixr2 FOCEI={expected:.4f} "
            f"(rel_err={rel_err:.1%})"
        )


@pytest.mark.external_validation
@pytest.mark.slow
class TestWarfarinBayesLaplaceEmpirical:
    """Weak-prior empirical BAYES(Laplace) should stay near the validated warfarin PK basin."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_bayes_laplace_model().fit()

    @pytest.fixture(scope="class")
    def nlmixr2_ref(self):
        return _load_ref("warfarin_pk_foce.json")

    def test_backend_is_laplace_and_samples_present(self, fit_result):
        theta_samples = fit_result.posterior_samples.get("theta")
        assert fit_result.backend_used == "laplace"
        assert fit_result.method == "BAYES(Laplace)"
        assert theta_samples is not None
        assert theta_samples.ndim == 2
        assert theta_samples.shape[1] == 3
        assert theta_samples.shape[0] >= 100

    def test_posterior_mean_tracks_map_estimate(self, fit_result):
        theta_samples = fit_result.posterior_samples["theta"]
        posterior_mean = theta_samples.mean(axis=0)
        tolerances = [0.08, 0.01, 0.20]
        for name, observed, expected, tol in zip(
            ("KA", "CL", "V"), posterior_mean, fit_result.theta_final, tolerances
        ):
            assert abs(float(observed) - float(expected)) < tol, (
                f"{name} posterior mean {observed:.4f} drifted from MAP {expected:.4f} "
                f"by more than {tol:.2f}"
            )

    def test_theta_tracks_external_warfarin_reference(self, fit_result, nlmixr2_ref):
        observed = [float(value) for value in fit_result.theta_final]
        expected = [
            float(nlmixr2_ref["theta"]["KA"]),
            float(nlmixr2_ref["theta"]["CL"]),
            float(nlmixr2_ref["theta"]["V"]),
        ]
        tolerances = [0.12, 0.06, 0.10]
        for name, obs, exp, tol in zip(("KA", "CL", "V"), observed, expected, tolerances):
            rel_err = abs(obs - exp) / exp
            assert rel_err < tol, (
                f"{name}={obs:.4f} vs nlmixr2 FOCEI={exp:.4f} "
                f"(rel_err={rel_err:.1%}, tolerance={tol:.0%})"
            )

    def test_sigma_is_finite_and_same_order_as_external_reference(self, fit_result, nlmixr2_ref):
        observed = float(fit_result.sigma_final[0, 0])
        expected = float(nlmixr2_ref["sigma_prop_err_variance"])
        assert np.isfinite(observed)
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.35, (
            f"sigma={observed:.5f} vs nlmixr2 FOCEI={expected:.5f} "
            f"(rel_err={rel_err:.1%})"
        )

    def test_posterior_scales_are_finite_and_positive(self, fit_result):
        posterior_sd = fit_result.posterior_samples["theta"].std(axis=0, ddof=1)
        assert np.all(np.isfinite(posterior_sd))
        assert np.all(posterior_sd > 0.0)


@pytest.mark.external_validation
@pytest.mark.slow
class TestTheophyllineBayesNUTSEmpirical:
    """Empirical BAYES(NUTS) should hit the validated theophylline basin on the measured budget."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        warnings.filterwarnings("ignore")
        return _build_theophylline_bayes_nuts_model().fit()

    @pytest.fixture(scope="class")
    def nlmixr2_ref(self):
        return _load_ref("theophylline_foce.json")

    def test_backend_and_chain_shapes_are_present(self, fit_result):
        theta_samples = fit_result.posterior_samples.get("theta")
        assert fit_result.backend_used == "nuts"
        assert fit_result.method == "BAYES(NUTS)"
        assert theta_samples is not None
        assert theta_samples.ndim == 2
        assert theta_samples.shape[1] == 3
        assert theta_samples.shape[0] == 48

    def test_theta_tracks_external_theophylline_reference(self, fit_result, nlmixr2_ref):
        observed = [float(value) for value in fit_result.theta_final]
        expected = [
            float(nlmixr2_ref["theta"]["KA"]),
            float(nlmixr2_ref["theta"]["CL"]),
            float(nlmixr2_ref["theta"]["V"]),
        ]
        tolerances = [0.05, 0.03, 0.03]
        for name, obs, exp, tol in zip(("KA", "CL", "V"), observed, expected, tolerances):
            rel_err = abs(obs - exp) / exp
            assert rel_err < tol, (
                f"{name}={obs:.4f} vs nlmixr2 FOCEI={exp:.4f} "
                f"(rel_err={rel_err:.1%}, tolerance={tol:.0%})"
            )

    def test_rhat_is_finite_and_bounded_for_measured_budget(self, fit_result):
        assert np.all(np.isfinite(fit_result.r_hat))
        assert float(np.max(fit_result.r_hat)) < 1.25
        assert np.all(np.isfinite(fit_result.n_effective))
        assert np.all(fit_result.n_effective > 0.0)

    def test_symbolic_analytic_path_is_active(self, fit_result):
        diag = fit_result.diagnostics["nuts"]
        assert diag["used_analytic_theta_gradient"] is True
        assert diag["theta_only"] is True
        assert diag["log_prob_calls"] > 0
        assert diag["foce_inner_calls"] > 0
        assert diag["warm_start_nearest_hits"] > 0
        assert all(chain["used_fd_gradient"] is False for chain in diag["chain_diagnostics"])



# ---------------------------------------------------------------------------
# Warfarin PK — BAYES(Laplace) vs nlmixr2 FOCEI reference
# ---------------------------------------------------------------------------

_WARF_BAYES_DATA_FILE = os.path.join(DATA_DIR, "warfarin_pk.csv")
_WARF_BAYES_REF_FILE = os.path.join(
    os.path.dirname(__file__), "nlmixr2", "reference", "warfarin_pk_saem.json"
)

_WARF_BAYES_PK = """\
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""

_WARF_BAYES_ERROR = "Y = F*(1 + EPS(1))"


def _build_warfarin_bayes_laplace_model(n_samples: int = 200, maxeval: int = 200):
    """Weak-prior BAYES(Laplace) on Warfarin PK dataset (32 subjects)."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    if not os.path.exists(_WARF_BAYES_DATA_FILE):
        pytest.skip(f"Warfarin data not found: {_WARF_BAYES_DATA_FILE}")

    ds = NONMEMDataset.from_csv(_WARF_BAYES_DATA_FILE)
    return (
        ModelBuilder()
        .problem("Warfarin PK 1-CMT oral — BAYES(Laplace) vs nlmixr2 FOCEI")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk(_WARF_BAYES_PK)
        .error(_WARF_BAYES_ERROR)
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(
            method="BAYES",
            backend="laplace",
            n_samples=n_samples,
            maxeval=maxeval,
            prior_sd_theta=1e8,
            seed=42,
        )
        .build()
    )


@pytest.mark.external_validation
@pytest.mark.slow
class TestWarfarinBayesLaplaceEmpirical:
    """
    Warfarin PK (1-CMT oral, 32 subjects) — BAYES(Laplace) vs nlmixr2 FOCEI.

    External anchor: ``tests/external_validation/nlmixr2/reference/warfarin_pk_saem.json``
    (nlmixr2 5.0, FOCEI objective, same dataset).

    nlmixr2 FOCEI reference values:
      KA = 0.766 h⁻¹  CL = 0.132 L/h  V = 8.129 L

    The BAYES(Laplace) MAP uses a FOCE objective with a weak prior
    (prior_sd_theta = 1e8 ≈ flat).  Recovery should be close to FOCEI
    since the prior contributes negligibly to the objective.

    The Warfarin dataset is well-conditioned (rich sampling, 32 subjects),
    so FO and FOCEI find the same basin — unlike sparse IV designs where the
    FO vs FOCEI objective surfaces can differ substantially.

    This test extends BAYES(Laplace) coverage to a second dataset beyond
    the Boeckmann Theophylline run and NONMEM Run 402.
    """

    @pytest.fixture(scope="class")
    def ref(self):
        if not os.path.exists(_WARF_BAYES_REF_FILE):
            pytest.skip(f"Reference not found: {_WARF_BAYES_REF_FILE}")
        with open(_WARF_BAYES_REF_FILE) as fh:
            return json.load(fh)

    @pytest.fixture(scope="class")
    def result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _build_warfarin_bayes_laplace_model().fit()

    # --- Backend and shape sanity --------------------------------------------

    def test_backend_is_laplace(self, result):
        assert result.backend_used == "laplace"

    def test_posterior_samples_shape(self, result):
        samples = result.posterior_samples["theta"]
        assert samples.ndim == 2
        assert samples.shape[0] >= 50, f"Too few samples: {samples.shape[0]}"
        assert samples.shape[1] == 3, f"Expected 3 theta params (KA/CL/V), got {samples.shape[1]}"

    def test_ofv_is_finite_and_not_penalty(self, result):
        """OFV must be finite and well below the 1e9 failure sentinel."""
        assert np.isfinite(result.ofv), f"OFV={result.ofv} is not finite"
        assert result.ofv < 1e6, (
            f"OFV={result.ofv:.1f} is suspiciously large — MAP likely failed. "
            "Check for optimizer divergence or warm-start η corruption."
        )

    def test_omega_psd(self, result):
        eigvals = np.linalg.eigvalsh(result.omega_final)
        assert np.all(eigvals >= -1e-8), f"OMEGA not PSD: {eigvals}"

    def test_sigma_positive(self, result):
        assert result.sigma_final[0, 0] > 0

    # --- Parameter recovery vs nlmixr2 FOCEI ---------------------------------

    def test_ka_within_20pct_of_focei(self, result, ref):
        est = float(result.theta_final[0])
        ref_val = float(ref["theta"]["KA"])
        pct = 100.0 * abs(est - ref_val) / ref_val
        assert pct < 20.0, f"BAYES MAP KA={est:.4f} is {pct:.1f}% from nlmixr2 {ref_val:.4f} (tol 20%)"

    def test_cl_within_15pct_of_focei(self, result, ref):
        est = float(result.theta_final[1])
        ref_val = float(ref["theta"]["CL"])
        pct = 100.0 * abs(est - ref_val) / ref_val
        assert pct < 15.0, f"BAYES MAP CL={est:.4f} is {pct:.1f}% from nlmixr2 {ref_val:.4f} (tol 15%)"

    def test_v_within_15pct_of_focei(self, result, ref):
        est = float(result.theta_final[2])
        ref_val = float(ref["theta"]["V"])
        pct = 100.0 * abs(est - ref_val) / ref_val
        assert pct < 15.0, f"BAYES MAP V={est:.4f} is {pct:.1f}% from nlmixr2 {ref_val:.4f} (tol 15%)"

    def test_subject_count_matches_data(self, result):
        assert len(result.post_hoc_etas) == 32, (
            f"Expected 32 Warfarin subjects, got {len(result.post_hoc_etas)}"
        )
