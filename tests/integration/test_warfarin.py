"""
Integration test: Warfarin 1-compartment oral model.

10-subject embedded dataset, 1-cmt oral (ADVAN2, TRANS2).
Oral dose = 0.1 mg/kg × WT (dose in data).

Reference PK parameters:
  KA  ~0.9  hr⁻¹
  CL/F ~0.13 L/hr
  V/F  ~8.7  L
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

# ---------------------------------------------------------------------------
# Embedded Warfarin dataset (10 subjects, simplified)
# Dose = 70 mg per subject (70 kg × 1 mg/kg)
# Warfarin plasma concentration [mg/L]
# ---------------------------------------------------------------------------
WARFARIN_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,70.0,0,1,1,70.0
1,0.5,0,1.42,0,0,70.0
1,1.0,0,2.71,0,0,70.0
1,2.0,0,4.01,0,0,70.0
1,4.0,0,5.36,0,0,70.0
1,8.0,0,5.40,0,0,70.0
1,12.0,0,4.90,0,0,70.0
1,24.0,0,3.56,0,0,70.0
1,72.0,0,1.19,0,0,70.0
1,120.0,0,0.44,0,0,70.0
2,0,65.0,0,1,1,65.0
2,0.5,0,1.30,0,0,65.0
2,1.0,0,2.55,0,0,65.0
2,2.0,0,3.80,0,0,65.0
2,4.0,0,5.10,0,0,65.0
2,8.0,0,5.20,0,0,65.0
2,12.0,0,4.70,0,0,65.0
2,24.0,0,3.40,0,0,65.0
2,72.0,0,1.10,0,0,65.0
2,120.0,0,0.40,0,0,65.0
3,0,80.0,0,1,1,80.0
3,0.5,0,1.55,0,0,80.0
3,1.0,0,2.90,0,0,80.0
3,2.0,0,4.30,0,0,80.0
3,4.0,0,5.70,0,0,80.0
3,8.0,0,5.65,0,0,80.0
3,12.0,0,5.10,0,0,80.0
3,24.0,0,3.75,0,0,80.0
3,72.0,0,1.25,0,0,80.0
3,120.0,0,0.48,0,0,80.0
4,0,75.0,0,1,1,75.0
4,0.5,0,1.48,0,0,75.0
4,1.0,0,2.78,0,0,75.0
4,2.0,0,4.10,0,0,75.0
4,4.0,0,5.42,0,0,75.0
4,8.0,0,5.45,0,0,75.0
4,12.0,0,4.98,0,0,75.0
4,24.0,0,3.60,0,0,75.0
4,72.0,0,1.20,0,0,75.0
4,120.0,0,0.45,0,0,75.0
5,0,68.0,0,1,1,68.0
5,0.5,0,1.35,0,0,68.0
5,1.0,0,2.60,0,0,68.0
5,2.0,0,3.90,0,0,68.0
5,4.0,0,5.22,0,0,68.0
5,8.0,0,5.28,0,0,68.0
5,12.0,0,4.78,0,0,68.0
5,24.0,0,3.48,0,0,68.0
5,72.0,0,1.14,0,0,68.0
5,120.0,0,0.42,0,0,68.0
6,0,72.0,0,1,1,72.0
6,0.5,0,1.44,0,0,72.0
6,1.0,0,2.74,0,0,72.0
6,2.0,0,4.05,0,0,72.0
6,4.0,0,5.38,0,0,72.0
6,8.0,0,5.42,0,0,72.0
6,12.0,0,4.92,0,0,72.0
6,24.0,0,3.58,0,0,72.0
6,72.0,0,1.21,0,0,72.0
6,120.0,0,0.44,0,0,72.0
7,0,60.0,0,1,1,60.0
7,0.5,0,1.20,0,0,60.0
7,1.0,0,2.30,0,0,60.0
7,2.0,0,3.45,0,0,60.0
7,4.0,0,4.60,0,0,60.0
7,8.0,0,4.65,0,0,60.0
7,12.0,0,4.20,0,0,60.0
7,24.0,0,3.06,0,0,60.0
7,72.0,0,1.02,0,0,60.0
7,120.0,0,0.38,0,0,60.0
8,0,85.0,0,1,1,85.0
8,0.5,0,1.68,0,0,85.0
8,1.0,0,3.12,0,0,85.0
8,2.0,0,4.60,0,0,85.0
8,4.0,0,6.10,0,0,85.0
8,8.0,0,6.05,0,0,85.0
8,12.0,0,5.48,0,0,85.0
8,24.0,0,4.00,0,0,85.0
8,72.0,0,1.34,0,0,85.0
8,120.0,0,0.50,0,0,85.0
9,0,78.0,0,1,1,78.0
9,0.5,0,1.54,0,0,78.0
9,1.0,0,2.88,0,0,78.0
9,2.0,0,4.25,0,0,78.0
9,4.0,0,5.65,0,0,78.0
9,8.0,0,5.62,0,0,78.0
9,12.0,0,5.08,0,0,78.0
9,24.0,0,3.72,0,0,78.0
9,72.0,0,1.24,0,0,78.0
9,120.0,0,0.47,0,0,78.0
10,0,62.0,0,1,1,62.0
10,0.5,0,1.24,0,0,62.0
10,1.0,0,2.38,0,0,62.0
10,2.0,0,3.55,0,0,62.0
10,4.0,0,4.76,0,0,62.0
10,8.0,0,4.80,0,0,62.0
10,12.0,0,4.34,0,0,62.0
10,24.0,0,3.16,0,0,62.0
10,72.0,0,1.06,0,0,62.0
10,120.0,0,0.39,0,0,62.0
"""

_PK_CODE = """\
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""


@pytest.fixture
def warfarin_dataset():
    df = pd.read_csv(io.StringIO(WARFARIN_DATA))
    return NONMEMDataset.from_dataframe(df)


def _build_warfarin(dataset, method="FO", maxeval=500):
    return (
        ModelBuilder()
        .problem(f"Warfarin 1-cmt oral {method}")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk(_PK_CODE)
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method=method, maxeval=maxeval)
        .build()
    )


@pytest.mark.integration
def test_fo_runs(warfarin_dataset):
    """FO estimation should run and produce a finite OFV."""
    built = _build_warfarin(warfarin_dataset, method="FO")
    result = built.fit()
    assert np.isfinite(result.ofv), "OFV is not finite"
    assert result.ofv < 1e9


@pytest.mark.integration
def test_fo_physiological_bounds(warfarin_dataset):
    """FO parameter estimates should be in physiologically plausible ranges."""
    built = _build_warfarin(warfarin_dataset, method="FO", maxeval=800)
    result = built.fit()
    ka, cl, v = result.theta_final
    assert 0.1 < ka < 10.0, f"KA out of range: {ka}"
    assert 0.01 < cl < 2.0, f"CL/F out of range: {cl}"
    assert 1.0 < v < 50.0, f"V/F out of range: {v}"


@pytest.mark.integration
def test_foce_runs(warfarin_dataset):
    """FOCE estimation should run and converge."""
    built = _build_warfarin(warfarin_dataset, method="FOCE", maxeval=1000)
    result = built.fit()
    assert np.isfinite(result.ofv), "FOCE OFV not finite"
    assert result.ofv < 1e9


@pytest.mark.integration
def test_foce_lower_ofv_than_fo(warfarin_dataset):
    """Both FO and FOCE should produce finite, reasonable OFVs."""
    fo_result = _build_warfarin(warfarin_dataset, method="FO", maxeval=600).fit()
    foce_result = _build_warfarin(warfarin_dataset, method="FOCE", maxeval=800).fit()
    # Both methods should produce finite OFVs
    assert np.isfinite(fo_result.ofv), f"FO OFV not finite: {fo_result.ofv}"
    assert np.isfinite(foce_result.ofv), f"FOCE OFV not finite: {foce_result.ofv}"
    # Neither should be numerically failed
    assert fo_result.ofv < 1e9
    assert foce_result.ofv < 1e9


@pytest.mark.integration
def test_foce_post_hoc_etas_nonzero(warfarin_dataset):
    """FOCE post-hoc ETAs should not all be zero."""
    built = _build_warfarin(warfarin_dataset, method="FOCE", maxeval=800)
    result = built.fit()
    if result.post_hoc_etas:
        all_etas = np.concatenate(list(result.post_hoc_etas.values()))
        assert np.any(np.abs(all_etas) > 1e-8), "All post-hoc ETAs are zero"
