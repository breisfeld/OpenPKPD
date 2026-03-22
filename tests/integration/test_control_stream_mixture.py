from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from openpkpd.cli.runner import run_model
from openpkpd.estimation.base import EstimationResult
from openpkpd.mixture import MixtureResult


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
        ]
    )
    df.to_csv(path, index=False, header=False)


def _mixture_control_stream() -> str:
    return """\
$PROBLEM Mixture test
$INPUT ID TIME AMT DV EVID MDV CMT RATE ADDL II SS
$DATA dummy.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
KA = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V  = THETA(3) * EXP(ETA(3))
$ERROR
Y = F + EPS(1)
$THETA 1.5 2.8 32.9 0.5
$OMEGA 0.04
$OMEGA 0.04
$OMEGA 0.04
$SIGMA 0.01
$ESTIMATION METHOD=COND MAXEVAL=25 PRINT=7
$MIXTURE NSPOP=2 PMIX=THETA(4)
"""


def _dummy_subpop_result(theta: list[float], ofv: float) -> EstimationResult:
    return EstimationResult(
        theta_final=np.asarray(theta, dtype=float),
        omega_final=np.eye(3) * 0.04,
        sigma_final=np.eye(1) * 0.01,
        ofv=ofv,
        converged=True,
        method="FOCE",
    )


@pytest.mark.integration
def test_run_model_mixture_returns_result_and_writes_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "mix.ctl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_dataset(data_path)
    ctl_path.write_text(_mixture_control_stream())

    captured: dict[str, object] = {}

    class _FakeMixtureModel:
        def __init__(self, population_model, n_subpop, estimation_method, estimation_kwargs):
            captured["n_subpop"] = n_subpop
            captured["estimation_method"] = estimation_method
            captured["estimation_kwargs"] = estimation_kwargs

        def fit(self, init_params):
            captured["theta"] = init_params.theta.copy().tolist()
            return MixtureResult(
                n_subpop=2,
                mixture_probs=np.array([0.55, 0.45]),
                subpop_probabilities={1: np.array([0.8, 0.2]), 2: np.array([0.3, 0.7])},
                subpop_results=[
                    _dummy_subpop_result([1.4, 2.7, 31.0, 0.5], 101.0),
                    _dummy_subpop_result([1.7, 3.0, 35.0, 0.5], 99.0),
                ],
                ofv=120.5,
                converged=True,
            )

    monkeypatch.setattr("openpkpd.cli.runner.MixtureModel", _FakeMixtureModel)

    result = run_model(
        ctl_path=str(ctl_path),
        dataset_path=str(data_path),
        output_dir=str(out_dir),
    )

    assert isinstance(result, MixtureResult)
    assert captured["n_subpop"] == 2
    assert captured["estimation_method"] == "FOCE"
    assert captured["estimation_kwargs"] == {
        "interaction": False,
        "maxeval": 25,
        "n_parallel": 1,
        "sigdig": 3,
        "print_interval": 7,
        "noabort": False,
    }

    summary_path = out_dir / "mix.mix.json"
    assignments_path = out_dir / "mix.mix_assignments.csv"
    assert summary_path.exists()
    assert assignments_path.exists()

    payload = json.loads(summary_path.read_text())
    assert payload["n_subpop"] == 2
    assert payload["estimation_method"] == "FOCE"
    assert payload["mixture_probs"] == [0.55, 0.45]
    assert payload["subpopulations"][0]["theta"][0] == pytest.approx(1.4)

    assignments = pd.read_csv(assignments_path)
    assert list(assignments["assigned_subpop"]) == [1, 2]
    assert list(assignments["P_SUBPOP1"]) == pytest.approx([0.8, 0.3])
