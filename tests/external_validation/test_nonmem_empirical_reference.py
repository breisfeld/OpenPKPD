"""Empirical external validation against bundled NONMEM reference datasets."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REF_DIR = os.path.join(os.path.dirname(__file__), "reference")

_NONMEM_504_DATA_FILE = os.path.join(DATA_DIR, "nonmem_501_iv_covariates.csv")
_NONMEM_504_REF_FILE = os.path.join(REF_DIR, "nonmem_504_focei.json")


def _load_nonmem_504_reference() -> dict:
    if not os.path.exists(_NONMEM_504_REF_FILE):
        pytest.skip(f"Reference file not found: {_NONMEM_504_REF_FILE}")
    with open(_NONMEM_504_REF_FILE) as f:
        return json.load(f)


def _build_nonmem_504_focei_model(maxeval: int = 5):
    """Return a maintained Python-API FOCEI benchmark for NONMEM Run 504."""
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    if not os.path.exists(_NONMEM_504_DATA_FILE):
        pytest.skip(f"Data file not found: {_NONMEM_504_DATA_FILE}")

    dataset = NONMEMDataset.from_csv(_NONMEM_504_DATA_FILE)
    return (
        ModelBuilder()
        .problem("NONMEM Run 504 — 1-cmt IV infusion with WT/AGE/SEX covariates")
        .dataset(dataset)
        .covariates(["WT", "AGE", "SEX"])
        .subroutines(advan=1, trans=2)
        .pk(
            "TVCL = THETA(1) * (WT/70)**THETA(3) * (AGE/50)**THETA(5) * THETA(7)**SEX\n"
            "TVV = THETA(2) * (WT/70)**THETA(4) * (AGE/50)**THETA(6) * THETA(8)**SEX\n"
            "CL = TVCL * EXP(ETA(1))\n"
            "V = TVV * EXP(ETA(2))\n"
            "S1 = V"
        )
        .error("Y = F * (1 + EPS(1))")
        .theta(
            [
                (0.5, 4.0, 10.0),
                (5.0, 30.0, 100.0),
                (-2.0, 0.8, 3.0),
                (-2.0, 0.8, 3.0),
                (-3.0, -0.5, 1.0),
                (-1.0, 0.05, 1.0),
                (0.2, 0.9, 2.0),
                (0.2, 0.95, 2.0),
            ]
        )
        .omega([[0.1, 0.001], [0.001, 0.1]])
        .sigma([[0.04]])
        .estimation(method="FOCEI", maxeval=maxeval, gtol=1e-6)
        .build()
    )


@pytest.mark.external_validation
@pytest.mark.slow
class TestNONMEM504FOCEIEmpirical:
    """The maintained API benchmark should recover the NONMEM 504 parameter basin."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        return _build_nonmem_504_focei_model().fit()

    @pytest.fixture(scope="class")
    def reference(self):
        return _load_nonmem_504_reference()

    def test_fit_is_finite_and_converged(self, fit_result):
        assert np.isfinite(fit_result.ofv)
        assert fit_result.converged, fit_result.message

    def test_theta_tracks_nonmem_reference_within_documented_uncertainty(self, fit_result, reference):
        names = (
            "CL_ref",
            "V_ref",
            "CL_WT_exponent",
            "V_WT_exponent",
            "CL_AGE_exponent",
            "V_AGE_exponent",
            "CL_SEX_multiplier",
            "V_SEX_multiplier",
        )
        tolerances = {
            "CL_ref": 0.03,
            "V_ref": 0.03,
            "CL_WT_exponent": 0.03,
            "V_WT_exponent": 0.03,
            "CL_AGE_exponent": 0.03,
            "V_AGE_exponent": 0.05,
            "CL_SEX_multiplier": 0.03,
            "V_SEX_multiplier": 0.03,
        }
        observed = [float(x) for x in fit_result.theta_final]

        for name, obs in zip(names, observed, strict=True):
            exp = float(reference["theta"][name])
            se = float(reference["theta_se"][name])
            rel_err = abs(obs - exp) / max(abs(exp), 1e-12)
            abs_err = abs(obs - exp)
            assert rel_err < tolerances[name] or abs_err <= se, (
                f"{name}={obs:.6f} differs from NONMEM {exp:.6f} by {rel_err:.1%} "
                f"(tolerance {tolerances[name]:.0%}, abs_err={abs_err:.6f}, ref_se={se:.6f})"
            )

    def test_variance_terms_stay_near_nonmem_reference(self, fit_result, reference):
        omega = fit_result.omega_final
        sigma = fit_result.sigma_final

        cl_cl = float(omega[0, 0])
        cl_cl_ref = float(reference["omega_block"]["CL_CL"])
        cl_cl_se = float(reference["omega_block_se"]["CL_CL"])
        assert cl_cl == pytest.approx(cl_cl_ref, rel=0.10) or abs(cl_cl - cl_cl_ref) <= 1.5 * cl_cl_se

        cl_v = float(omega[0, 1])
        cl_v_ref = float(reference["omega_block"]["CL_V"])
        cl_v_se = float(reference["omega_block_se"]["CL_V"])
        assert cl_v == pytest.approx(cl_v_ref, rel=0.70) or abs(cl_v - cl_v_ref) <= 1.5 * cl_v_se

        v_v = float(omega[1, 1])
        v_v_ref = float(reference["omega_block"]["V_V"])
        v_v_se = float(reference["omega_block_se"]["V_V"])
        assert v_v == pytest.approx(v_v_ref, rel=0.10) or abs(v_v - v_v_ref) <= 1.5 * v_v_se

        assert float(sigma[0, 0]) == pytest.approx(reference["sigma_diag"]["eps1"], rel=0.08)

    def test_ofv_stays_in_documented_nonmem_504_range(self, fit_result, reference):
        nm_ofv = float(reference["ofv"])
        pct_diff = 100.0 * (float(fit_result.ofv) - nm_ofv) / abs(nm_ofv)
        assert pct_diff < 25.0, (
            f"FOCEI landed in the right parameter basin but OFV gap {pct_diff:.1f}% "
            f"exceeded the documented tolerance window"
        )
