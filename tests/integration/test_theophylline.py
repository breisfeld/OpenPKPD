"""
Integration test: Theophylline 1-compartment oral model.

Uses the classic Theophylline dataset (Boeckmann et al.) to verify
that the full pipeline (parse → model assembly → FO estimation) runs
end-to-end and produces reasonable estimates.

Reference NONMEM estimates (approximate):
  THETA1 (KA):  ~1.5 hr⁻¹
  THETA2 (CL):  ~0.04 L/hr/kg
  THETA3 (V):   ~0.47 L/kg
  OFV:          ~-44 to -50 (varies by dataset)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

# Theophylline dataset (NONMEM standard example, 12 subjects)
THEO_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.27,0,0.74,0,0,79.6
1,0.57,0,1.72,0,0,79.6
1,1.02,0,7.91,0,0,79.6
1,1.92,0,8.31,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,5.02,0,6.85,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,9.0,0,5.4,0,0,79.6
1,12.05,0,4.55,0,0,79.6
1,24.37,0,1.25,0,0,79.6
2,0,4.4,0,1,1,72.4
2,0.35,0,0.96,0,0,72.4
2,0.6,0,2.33,0,0,72.4
2,1.07,0,4.71,0,0,72.4
2,2.13,0,8.33,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,5.02,0,7.14,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,9.1,0,4.55,0,0,72.4
2,12.1,0,3.01,0,0,72.4
2,25.0,0,0.9,0,0,72.4
"""


@pytest.fixture
def theo_dataset() -> NONMEMDataset:
    df = pd.read_csv(io.StringIO(THEO_DATA))
    return NONMEMDataset.from_dataframe(df)


@pytest.mark.integration
def test_fo_runs(theo_dataset):
    """FO estimation should run without errors on Theophylline data."""
    model = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral FO")
        .dataset(theo_dataset)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .estimation(method="FO", maxeval=500)
        .build()
    )
    result = model.fit()
    # Basic sanity checks
    assert result.ofv < 1e9  # Did not fail numerically
    assert len(result.theta_final) == 3
    assert result.theta_final[0] > 0  # KA > 0
    assert result.theta_final[1] > 0  # CL > 0
    assert result.theta_final[2] > 0  # V > 0


@pytest.mark.integration
def test_pk_callable_evaluates(theo_dataset):
    """The compiled $PK callable should return non-zero PK params."""
    from openpkpd.parser.code_compiler import NMTRANCompiler

    pk_code = """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""
    compiler = NMTRANCompiler()
    fn = compiler.compile_pk(pk_code)
    theta = [1.5, 0.08, 30.0]
    eta = [0.0, 0.0, 0.0]
    params = fn(theta, eta)
    assert params["KA"] == pytest.approx(1.5)
    assert params["CL"] == pytest.approx(0.08)
    assert params["V"] == pytest.approx(30.0)


@pytest.mark.integration
def test_advan2_predictions(theo_dataset):
    """ADVAN2 should produce positive IPRED values for Theophylline."""
    from openpkpd.data.event_processor import DoseEvent
    from openpkpd.pk.analytical.advan2 import ADVAN2

    advan = ADVAN2()
    pk_params = {"KA": 1.5, "K": 0.08 / 30.0, "V": 30.0}

    obs_times = np.array([0.27, 0.57, 1.02, 1.92, 3.5, 5.02, 7.03, 9.0, 12.05, 24.37])
    doses = [DoseEvent(time=0.0, amount=4.02, compartment=1)]

    sol = advan.solve(pk_params, doses, obs_times)
    assert np.all(sol.ipred > 0), "All IPREDs should be positive"
    assert sol.ipred[0] < sol.ipred[3], "Concentration should rise before falling"


@pytest.mark.integration
def test_control_stream_parse_to_problem():
    """A minimal control stream should parse into a Problem object."""

    from openpkpd.parser.control_stream import ControlStream

    cs_text = """\
$PROBLEM Test model
$DATA theo.csv
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  Y = F*(1 + EPS(1))
$THETA (0,1.5,20) (0,0.08,5) (0,30,500)
$OMEGA 0.5 0.3 0.3
$SIGMA 0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999
"""
    cs = ControlStream.from_string(cs_text)
    assert cs.problem is not None
    assert len(cs.theta_records[0].specs) == 3
    assert cs.estimation_records[0].method == "FOCE"


# Theophylline control stream written verbatim from the manuscript Step 1
# SAEM listing. Initial estimates are on the absolute-dose scale (CL/F, V/F),
# matching the bundled dataset (AMT in mg). See manuscript fix S5.
_THEO_SAEM_CTL = """\
$PROBLEM Theophylline 1-cmt oral
$INPUT  ID TIME AMT DV EVID WT MDV
$DATA   theophylline.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
   KA = THETA(1)*EXP(ETA(1))
   CL = THETA(2)*EXP(ETA(2))
   V  = THETA(3)*EXP(ETA(3))
   S2 = V
$ERROR
   Y = F*(1 + EPS(1))
$THETA
   (0, 1.5)    ; KA
   (0, 0.08)   ; CL
   (0, 30)     ; V
$OMEGA
   0.5 0.3 0.3
$SIGMA
   0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV PRED IPRED CWRES NOPRINT ONEHEADER FILE=theo.tab
"""


@pytest.mark.integration
@pytest.mark.slow
def test_control_stream_saem_recovers_focei_optimum(tmp_path):
    """Control-stream SAEM path recovers the theophylline FOCEI optimum.

    Mirrors the manuscript Step 1 SAEM listing end-to-end: the bundled
    theophylline dataset is exported as a NONMEM-style CSV (absolute mg doses),
    parsed from a control stream with absolute-dose initial estimates, and fit
    with SAEM. With the corrected initial estimates the run lands on the same
    fixed effects as the FOCEI fit (KA ~ 1.57 /h, CL/F ~ 2.79 L/h,
    V/F ~ 31.6 L), rather than the implausible optimum produced by the previous
    weight-normalized initial estimates.

    The assertions target the recovered estimates rather than
    ``result.converged``: that flag reflects the SAEM phase-2 stability
    tolerance (``phi_tol``), which is not necessarily satisfied within the
    100 phase-2 iterations used here even when the estimates are correct.
    """
    from openpkpd.data import load_theophylline
    from openpkpd.estimation.saem import SAEMMethod
    from openpkpd.model.problem import Problem
    from openpkpd.parser.control_stream import ControlStream

    cols = ["ID", "TIME", "AMT", "DV", "EVID", "WT", "MDV"]
    df = load_theophylline().df[cols]
    csv_path = tmp_path / "theophylline.csv"
    with open(csv_path, "w") as fh:
        fh.write("@" + ",".join(cols) + "\n")
        for _, row in df.iterrows():
            fh.write(",".join(str(v) for v in row.tolist()) + "\n")

    ctl_path = tmp_path / "theophylline.ctl"
    ctl_path.write_text(_THEO_SAEM_CTL)

    cs = ControlStream.from_file(str(ctl_path))
    problem = Problem.from_control_stream(cs, dataset_path=str(csv_path))
    result = SAEMMethod(
        n_iter_phase1=200, n_iter_phase2=100, n_chains=2, seed=42
    ).estimate(problem.population_model, problem.population_model.params)

    assert len(result.theta_final) == 3
    ka, cl, v = (float(t) for t in result.theta_final)
    assert ka == pytest.approx(1.565, abs=0.25)
    assert cl == pytest.approx(2.791, abs=0.25)
    assert v == pytest.approx(31.55, abs=2.5)
    assert np.isfinite(result.ofv)
