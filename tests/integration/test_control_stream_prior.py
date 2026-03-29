from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.cli.runner import run_model
from openpkpd.estimation.base import EstimationResult
from openpkpd.prior import PriorAugmentedModel
from openpkpd.utils.errors import ParseError


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
        ]
    )
    df.to_csv(path, index=False, header=False)


def _prior_control_stream() -> str:
    return """\
$PROBLEM PRIOR runtime integration
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
$PRIOR NWPRI NTHETA=3 NETA=3
$THETAP 1.4 2.7 31.0
$THETAPV 0.25 0.25 4.0
$OMEGAP 0.03 0.03 0.03
$OMEGAPD 5 5 5
$ESTIMATION METHOD=FO MAXEVAL=10 PRINT=7
"""


def _sigmap_only_control_stream() -> str:
    return """\
$PROBLEM SIGMAP parse-only integration
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
$PRIOR NWPRI NEPS=1
$SIGMAP 0.02
$SIGMAPD 5
$ESTIMATION METHOD=FO MAXEVAL=10 PRINT=7
"""


def _nonparametric_parse_only_control_stream() -> str:
    return """\
$PROBLEM NONPARAMETRIC parse-only integration
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
$NONPARAMETRIC NPSUPP=42 MCETA=7
$ESTIMATION METHOD=FO MAXEVAL=10 PRINT=7
"""


@pytest.mark.integration
def test_run_model_prior_wraps_population_model_and_writes_standard_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "prior.ctl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_dataset(data_path)
    ctl_path.write_text(_prior_control_stream())

    captured: dict[str, object] = {}

    class _FakeEstimator:
        def estimate(self, population_model, init_params):
            assert isinstance(population_model, PriorAugmentedModel)
            captured["theta_prior"] = population_model.prior.theta_prior.copy()
            captured["omega_prior"] = population_model.prior.omega_prior.copy()
            captured["theta_prior_cov_diag"] = np.diag(population_model.prior.theta_prior_cov).copy()
            captured["omega_prior_cov_diag"] = np.diag(population_model.prior.omega_prior_cov).copy()
            return EstimationResult(
                theta_final=init_params.theta.copy(),
                omega_final=init_params.omega.copy(),
                sigma_final=init_params.sigma.copy(),
                ofv=12.34,
                converged=True,
                post_hoc_etas={
                    int(sid): np.zeros(init_params.n_eta(), dtype=float)
                    for sid in population_model.subject_ids()
                },
                method="FO",
            )

    monkeypatch.setattr("openpkpd.cli.runner.get_estimation_method", lambda *a, **k: _FakeEstimator())

    result = run_model(
        ctl_path=str(ctl_path),
        dataset_path=str(data_path),
        output_dir=str(out_dir),
    )

    assert result.converged is True
    np.testing.assert_allclose(captured["theta_prior"], np.array([1.4, 2.7, 31.0]))
    np.testing.assert_allclose(captured["omega_prior"], np.array([0.03, 0.0, 0.0, 0.03, 0.0, 0.03]))
    np.testing.assert_allclose(captured["theta_prior_cov_diag"], np.array([0.25, 0.25, 4.0]))
    np.testing.assert_allclose(
        captured["omega_prior_cov_diag"],
        np.array([0.2, 1e12, 1e12, 0.2, 1e12, 0.2]),
    )

    assert (out_dir / "prior.lst").exists()
    assert (out_dir / "prior.ext").exists()
    assert (out_dir / "prior.phi").exists()


@pytest.mark.integration
def test_run_model_prior_rejects_incomplete_theta_prior_pair(tmp_path):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "prior_invalid.ctl"

    _write_dataset(data_path)
    ctl_path.write_text(_prior_control_stream().replace("$THETAPV 0.25 0.25 4.0\n", ""))

    with pytest.raises(ParseError, match=r"Invalid prior specification: \$THETAP and \$THETAPV"):
        run_model(
            ctl_path=str(ctl_path),
            dataset_path=str(data_path),
        )


@pytest.mark.integration
def test_run_model_sigmap_records_remain_parse_only_at_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "sigmap_only.ctl"

    _write_dataset(data_path)
    ctl_path.write_text(_sigmap_only_control_stream())

    captured: dict[str, object] = {}

    class _FakeEstimator:
        def estimate(self, population_model, init_params):
            captured["is_prior_augmented"] = isinstance(population_model, PriorAugmentedModel)
            captured["sigma_init"] = init_params.sigma.copy()
            return EstimationResult(
                theta_final=init_params.theta.copy(),
                omega_final=init_params.omega.copy(),
                sigma_final=init_params.sigma.copy(),
                ofv=7.89,
                converged=True,
                post_hoc_etas={
                    int(sid): np.zeros(init_params.n_eta(), dtype=float)
                    for sid in population_model.subject_ids()
                },
                method="FO",
            )

    monkeypatch.setattr("openpkpd.cli.runner.get_estimation_method", lambda *a, **k: _FakeEstimator())

    result = run_model(
        ctl_path=str(ctl_path),
        dataset_path=str(data_path),
    )

    assert result.converged is True
    assert captured["is_prior_augmented"] is False
    np.testing.assert_allclose(captured["sigma_init"], np.array([[0.01]]))


@pytest.mark.integration
def test_run_model_nonparametric_record_remains_parse_only_at_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_path = tmp_path / "pk.csv"
    ctl_path = tmp_path / "nonparametric_parse_only.ctl"

    _write_dataset(data_path)
    ctl_path.write_text(_nonparametric_parse_only_control_stream())

    captured: dict[str, object] = {}

    class _FakeEstimator:
        def estimate(self, population_model, init_params):
            captured["population_model_type"] = type(population_model).__name__
            captured["theta_init"] = init_params.theta.copy()
            return EstimationResult(
                theta_final=init_params.theta.copy(),
                omega_final=init_params.omega.copy(),
                sigma_final=init_params.sigma.copy(),
                ofv=6.54,
                converged=True,
                post_hoc_etas={
                    int(sid): np.zeros(init_params.n_eta(), dtype=float)
                    for sid in population_model.subject_ids()
                },
                method="FO",
            )

    def _fake_get_estimation_method(method_name, **kwargs):
        captured["method_name"] = method_name
        captured["kwargs"] = dict(kwargs)
        return _FakeEstimator()

    monkeypatch.setattr("openpkpd.cli.runner.get_estimation_method", _fake_get_estimation_method)

    result = run_model(
        ctl_path=str(ctl_path),
        dataset_path=str(data_path),
    )

    assert result.converged is True
    assert captured["method_name"] == "FO"
    assert captured["kwargs"]["maxeval"] == 10
    assert captured["population_model_type"] == "PopulationModel"
    np.testing.assert_allclose(captured["theta_init"], np.array([1.5, 2.8, 32.9]))
