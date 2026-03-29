"""Unit tests for ControlStream parsing."""

import pytest

from openpkpd.parser.control_stream import ControlStream

SIMPLE_CTL = """\
$PROBLEM Theophylline 1-compartment oral
$DATA theo.csv IGNORE=@
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  Y = F*(1 + EPS(1))
$THETA
  (0, 1.5, 10)    ; KA
  (0, 0.08, 5)    ; CL
  (0, 30, 500)    ; V
$OMEGA
  0.5
  0.3
  0.3
$SIGMA
  0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999
$COVARIANCE
$TABLE ID TIME DV PRED IPRED CWRES NOPRINT FILE=sdtab001
"""


@pytest.mark.unit
def test_from_string():
    cs = ControlStream.from_string(SIMPLE_CTL)
    assert cs.problem is not None
    assert cs.data is not None
    assert cs.input is not None
    assert cs.pk is not None
    assert cs.error is not None
    assert len(cs.theta_records) == 1
    assert len(cs.omega_records) == 1
    assert len(cs.sigma_records) == 1
    assert len(cs.estimation_records) == 1
    assert cs.covariance is not None
    assert len(cs.table_records) == 1


@pytest.mark.unit
def test_problem_title():
    cs = ControlStream.from_string(SIMPLE_CTL)
    assert "Theophylline" in cs.problem.title


@pytest.mark.unit
def test_subroutines():
    cs = ControlStream.from_string(SIMPLE_CTL)
    assert cs.subroutines.advan == 2
    assert cs.subroutines.trans == 2


@pytest.mark.unit
def test_theta_count():
    cs = ControlStream.from_string(SIMPLE_CTL)
    specs = []
    for r in cs.theta_records:
        specs.extend(r.specs)
    assert len(specs) == 3
    assert specs[0].init == pytest.approx(1.5)


@pytest.mark.unit
def test_omega_diagonal():
    cs = ControlStream.from_string(SIMPLE_CTL)
    specs = []
    for r in cs.omega_records:
        specs.extend(r.specs)
    assert len(specs) == 3
    assert specs[0].values[0] == pytest.approx(0.5)


@pytest.mark.unit
def test_estimation_method():
    cs = ControlStream.from_string(SIMPLE_CTL)
    assert cs.estimation_records[0].method == "FOCE"
    assert cs.estimation_records[0].interaction is True


@pytest.mark.unit
def test_table_record():
    cs = ControlStream.from_string(SIMPLE_CTL)
    tbl = cs.table_records[0]
    assert "DV" in tbl.columns
    assert tbl.noprint is True
    assert tbl.file == "sdtab001"


@pytest.mark.unit
def test_abbreviations():
    """Test that abbreviated record names are resolved."""
    text = """\
$PROB Test
$EST METHOD=COND
$COV
"""
    cs = ControlStream.from_string(text)
    assert cs.problem is not None
    assert len(cs.estimation_records) == 1
    assert cs.covariance is not None


@pytest.mark.unit
def test_repr():
    cs = ControlStream.from_string(SIMPLE_CTL)
    r = repr(cs)
    assert "PROBLEM" in r
    assert "THETA" in r


@pytest.mark.unit
def test_prior_record_accessors():
    text = """\
$PROBLEM Prior accessors
$THETA 1
$OMEGA 0.1
$SIGMA 0.1
$PRIOR NWPRI NTHETA=1 NETA=1
$THETAP 1.2
$THETAPV 0.25
$OMEGAP 0.2
$OMEGAPD 4
$SIGMAP 0.1
$SIGMAPD 3
"""
    cs = ControlStream.from_string(text)

    assert cs.prior_record is not None
    assert cs.thetap_record is not None
    assert cs.thetapv_record is not None
    assert cs.omegap_record is not None
    assert cs.omegapd_record is not None
    assert cs.sigmap_record is not None
    assert cs.sigmapd_record is not None


@pytest.mark.unit
def test_simulation_record_accessor():
    cs = ControlStream.from_string(
        "$PROBLEM Sim\n$SIMULATION (12345) ONLYSIMULATION SUBPROBLEMS=3 TRUE=FINAL\n"
    )

    assert cs.simulation is not None
    assert cs.simulation.seeds == [12345]
    assert cs.simulation.onlysimulation is True
    assert cs.simulation.subproblems == 3
    assert cs.simulation.true_final is True
    sim_dict = cs.simulation.to_dict()
    assert sim_dict["onlysimulation"] is True
    assert sim_dict["true_final"] is True


@pytest.mark.unit
def test_mixture_record_accessor():
    cs = ControlStream.from_string("$PROBLEM Mix\n$MIXTURE NSPOP=3 PMIX=THETA(4)\n")

    assert cs.mixture is not None
    assert cs.mixture.nspop == 3
    assert cs.mixture.pmix_theta_index == 4
    mix_dict = cs.mixture.to_dict()
    assert mix_dict["nspop"] == 3
    assert mix_dict["pmix_theta_index"] == 4


@pytest.mark.unit
def test_to_string_round_trip_preserves_supported_structured_records():
    text = """\
$PROBLEM Round-trip
$DATA theo.csv IGNORE=@
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
$ERROR
Y = F*(1 + EPS(1))
$THETA (0.01,1.5,20) (0.001,0.08,5) (0.1,30,500)
$OMEGA BLOCK(2)
0.4
0.1 0.3
$SIGMA 0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=25 OUTEROPT=L-BFGS-B RETAINBEST
$SIMULATION (12345) SUBPROBLEMS=2 TRUE=FINAL
$PRIOR NWPRI NTHETA=3 NETA=2
$THETAP 1.5 0.08 30
$THETAPV 0.25 0.09 4.0
$OMEGAP 0.2 0.01 0.3
$OMEGAPD 4 1 5
$MIXTURE NSPOP=2 PMIX=THETA(4)
$TABLE ID TIME DV FILE=sdtab001
"""
    original = ControlStream.from_string(text)
    rendered = original.to_string()
    reparsed = ControlStream.from_string(rendered)

    assert reparsed.problem.title == "Round-trip"
    assert reparsed.data.filename == "theo.csv"
    assert reparsed.subroutines.advan == 2
    assert reparsed.subroutines.trans == 2
    assert reparsed.estimation_records[0].method == "FOCE"
    assert reparsed.estimation_records[0].interaction is True
    assert reparsed.estimation_records[0].maxeval == 25
    assert reparsed.estimation_records[0].outer_optimizer == "L-BFGS-B"
    assert reparsed.estimation_records[0].retain_best_iterate is True
    assert reparsed.simulation is not None
    assert reparsed.simulation.seeds == [12345]
    assert reparsed.simulation.subproblems == 2
    assert reparsed.simulation.true_final is True
    assert reparsed.prior_record is not None
    assert reparsed.prior_record.type == "NWPRI"
    assert reparsed.thetap_record is not None
    assert reparsed.omegap_record is not None
    assert reparsed.omegapd_record is not None
    assert reparsed.mixture is not None
    assert reparsed.mixture.nspop == 2
    assert reparsed.mixture.pmix_theta_index == 4
    assert reparsed.table_records[0].file == "sdtab001"


@pytest.mark.unit
def test_write_creates_parent_directories(tmp_path):
    cs = ControlStream.from_string(SIMPLE_CTL)
    out_path = tmp_path / "nested" / "dir" / "model.ctl"

    cs.write(str(out_path))

    assert out_path.exists()
    reparsed = ControlStream.from_file(str(out_path))
    assert reparsed.problem is not None
    assert reparsed.problem.title == "Theophylline 1-compartment oral"
