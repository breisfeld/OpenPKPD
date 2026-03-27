"""External-validation benchmarks against public PKNCA theophylline output."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from openpkpd.nca.nca import NCAEngine

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "theophylline_boeckmann.csv")
REFERENCE_PATH = os.path.join(HERE, "reference", "pknca_theophylline_summary.json")


def _load_reference() -> dict:
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _geometric_mean(values: np.ndarray) -> float:
    return float(math.exp(np.mean(np.log(values), dtype=float)))


def _geometric_cv_percent(values: np.ndarray) -> float:
    log_var = np.var(np.log(values), ddof=1, dtype=float)
    return float(math.sqrt(math.exp(log_var) - 1.0) * 100.0)


@pytest.fixture(scope="module")
def pknca_reference() -> dict:
    return _load_reference()


@pytest.fixture(scope="module")
def theophylline_nca_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    engine = NCAEngine(auc_method="linear-up-log-down", exclude_cmax=True)
    rows: list[dict[str, float]] = []
    for subject_id, group in df.groupby("ID"):
        obs_all = group[(group["EVID"] == 0) & (group["MDV"] == 0)].copy()
        obs_24 = obs_all[(obs_all["TIME"] > 0.0) & (obs_all["TIME"] <= 24.0)].copy()
        full = engine.compute_subject(
            times=obs_all["TIME"].to_numpy(float),
            conc=obs_all["DV"].to_numpy(float),
            dose=1.0,
            subject_id=subject_id,
            route="oral",
        )
        partial_24 = engine.compute_subject(
            times=obs_24["TIME"].to_numpy(float),
            conc=obs_24["DV"].to_numpy(float),
            dose=1.0,
            subject_id=subject_id,
            route="oral",
        )
        rows.append(
            {
                "ID": float(subject_id),
                "auclast_0_24": partial_24.auc_last,
                "cmax": full.cmax,
                "tmax": full.tmax,
                "half_life": full.t_half,
                "aucinf_obs": full.auc_inf,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.external_validation
class TestTheophyllineVsPKNCA:
    def test_subject_count_matches_pknca_summary(self, theophylline_nca_results, pknca_reference):
        assert len(theophylline_nca_results) == int(pknca_reference["summary_0_inf"]["n_subjects"])
        assert len(theophylline_nca_results) == int(pknca_reference["summary_0_24"]["n_subjects"])

    def test_auclast_0_24_summary_tracks_pknca(self, theophylline_nca_results, pknca_reference):
        ref = pknca_reference["summary_0_24"]["auclast"]
        values = theophylline_nca_results["auclast_0_24"].to_numpy(float)
        assert _geometric_mean(values) == pytest.approx(ref["center"], rel=0.01)
        assert _geometric_cv_percent(values) == pytest.approx(ref["dispersion"], abs=0.5)

    def test_full_interval_summary_tracks_pknca(self, theophylline_nca_results, pknca_reference):
        ref = pknca_reference["summary_0_inf"]
        cmax = theophylline_nca_results["cmax"].to_numpy(float)
        aucinf = theophylline_nca_results["aucinf_obs"].to_numpy(float)
        assert _geometric_mean(cmax) == pytest.approx(ref["cmax"]["center"], rel=0.01)
        assert _geometric_cv_percent(cmax) == pytest.approx(ref["cmax"]["dispersion"], abs=0.5)
        assert _geometric_mean(aucinf) == pytest.approx(ref["aucinf_obs"]["center"], rel=0.01)
        assert _geometric_cv_percent(aucinf) == pytest.approx(
            ref["aucinf_obs"]["dispersion"], abs=0.5
        )

    def test_tmax_and_half_life_summary_tracks_pknca(
        self, theophylline_nca_results, pknca_reference
    ):
        ref = pknca_reference["summary_0_inf"]
        tmax = theophylline_nca_results["tmax"].to_numpy(float)
        half_life = theophylline_nca_results["half_life"].to_numpy(float)
        assert float(np.median(tmax)) == pytest.approx(ref["tmax"]["center"], abs=0.02)
        assert float(np.min(tmax)) == pytest.approx(ref["tmax"]["lower"], abs=0.02)
        assert float(np.max(tmax)) == pytest.approx(ref["tmax"]["upper"], abs=0.02)
        assert float(np.mean(half_life)) == pytest.approx(ref["half_life"]["center"], abs=0.05)
        assert float(np.std(half_life, ddof=1)) == pytest.approx(
            ref["half_life"]["dispersion"], abs=0.05
        )

    def test_auc_ratio_summary_tracks_pknca_reference(self, theophylline_nca_results, pknca_reference):
        auclast_geom = _geometric_mean(theophylline_nca_results["auclast_0_24"].to_numpy(float))
        aucinf_geom = _geometric_mean(theophylline_nca_results["aucinf_obs"].to_numpy(float))
        observed_ratio = aucinf_geom / auclast_geom
        ref_ratio = (
            float(pknca_reference["summary_0_inf"]["aucinf_obs"]["center"])
            / float(pknca_reference["summary_0_24"]["auclast"]["center"])
        )
        assert observed_ratio == pytest.approx(ref_ratio, rel=0.02)
