from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.cli.runner import _run_simulation_record, run_model
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.problem import Problem
from openpkpd.parser.control_stream import ControlStream
from openpkpd.simulation.engine import SimulationResult


def _write_dataset(path) -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "TIME": 0.0,
                "AMT": 320.0,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 1,
                "TIME": 0.5,
                "AMT": 0.0,
                "DV": 3.4,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 1,
                "TIME": 1.0,
                "AMT": 0.0,
                "DV": 5.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 1,
                "TIME": 2.0,
                "AMT": 0.0,
                "DV": 4.1,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 2,
                "TIME": 0.0,
                "AMT": 320.0,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 2,
                "TIME": 0.5,
                "AMT": 0.0,
                "DV": 3.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 2,
                "TIME": 1.0,
                "AMT": 0.0,
                "DV": 4.7,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
            {
                "ID": 2,
                "TIME": 2.0,
                "AMT": 0.0,
                "DV": 4.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            },
        ]
    )
    df.to_csv(path, index=False, header=False)


def _simulation_only_control_stream() -> str:
    return """\
$PROBLEM Simulation-only test
$INPUT ID TIME AMT DV EVID MDV CMT RATE ADDL II SS
$DATA dummy.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
KA = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V  = THETA(3) * EXP(ETA(3))
$ERROR
Y = F + EPS(1)
$THETA 1.5 2.8 32.9
$OMEGA 0.04
$OMEGA 0.04
$OMEGA 0.04
$SIGMA 0.01
$SIMULATION (24680) ONLYSIMULATION SUBPROBLEMS=3 TRUE=FINAL
"""


def _postfit_simulation_control_stream() -> str:
    return _simulation_only_control_stream().replace(
        "$SIMULATION (24680) ONLYSIMULATION SUBPROBLEMS=3 TRUE=FINAL",
        "$SIMULATION (13579) SUBPROBLEMS=2",
    )


@pytest.mark.integration
def test_run_model_onlysimulation_returns_simulation_result_and_writes_csv(tmp_path):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "sim_only.ctl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_dataset(data_path)
    ctl_path.write_text(_simulation_only_control_stream())

    result = run_model(
        ctl_path=str(ctl_path),
        dataset_path=str(data_path),
        output_dir=str(out_dir),
    )

    assert isinstance(result, SimulationResult)
    assert result.seed == 24680
    assert result.n_replicates == 3
    assert set(result.simulated_df["REP"].unique()) == {0, 1, 2, 3}

    sim_csv = out_dir / "sim_only.sim.csv"
    assert sim_csv.exists()
    written = pd.read_csv(sim_csv)
    assert "REP" in written.columns
    assert written["REP"].max() == 3
    assert len(written) == len(result.simulated_df)


@pytest.mark.integration
def test_run_simulation_record_writes_csv_for_existing_parameter_state(tmp_path):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "postfit_sim.ctl"

    _write_dataset(data_path)
    ctl_path.write_text(_postfit_simulation_control_stream())

    cs = ControlStream.from_file(str(ctl_path))
    problem = Problem.from_control_stream(cs, dataset_path=str(data_path))
    pop_model = problem.population_model
    params = pop_model.params
    est_result = EstimationResult(
        theta_final=params.theta.copy(),
        omega_final=params.omega.copy(),
        sigma_final=params.sigma.copy(),
        ofv=0.0,
        converged=True,
        post_hoc_etas={
            int(sid): np.zeros(params.n_eta(), dtype=float) for sid in pop_model.subject_ids()
        },
        method="FAKEFIT",
    )

    sim_result = _run_simulation_record(
        pop_model,
        params,
        cs.simulation,
        str(tmp_path / "postfit_sim"),
        result=est_result,
    )

    assert sim_result.seed == 13579
    assert sim_result.n_replicates == 2
    assert (tmp_path / "postfit_sim.sim.csv").exists()
