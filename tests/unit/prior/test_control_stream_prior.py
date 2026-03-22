"""Unit tests for control-stream prior runtime wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.model.problem import Problem
from openpkpd.parser.control_stream import ControlStream
from openpkpd.prior import PriorAugmentedModel
from openpkpd.utils.errors import ParseError


def _minimal_dataset() -> NONMEMDataset:
    df = pd.DataFrame({"ID": [1, 1], "TIME": [0.0, 1.0], "DV": [0.0, 1.0]})
    return NONMEMDataset.from_dataframe(df)


@pytest.mark.unit
def test_problem_from_control_stream_wraps_omega_diagonal_prior() -> None:
    cs = ControlStream.from_string(
        """\
$PROBLEM OMEGA prior runtime
$SUBROUTINES ADVAN1 TRANS1
$THETA 1
$OMEGA 0.1
$OMEGA 0.2
$SIGMA 0.1
$PRIOR NWPRI NETA=2
$OMEGAP 0.3 0.4
$OMEGAPD 5 10
"""
    )

    problem = Problem.from_control_stream(cs, dataset=_minimal_dataset())

    assert isinstance(problem.population_model, PriorAugmentedModel)
    prior = problem.population_model.prior
    np.testing.assert_allclose(prior.omega_prior, np.array([0.3, 0.0, 0.4]))
    assert prior.omega_prior_cov is not None
    cov_diag = np.diag(prior.omega_prior_cov)
    assert cov_diag[0] == pytest.approx(0.2)
    assert cov_diag[1] > 1e11
    assert cov_diag[2] == pytest.approx(0.1)


@pytest.mark.unit
def test_problem_from_control_stream_supports_full_lower_triangle_omega_prior() -> None:
    cs = ControlStream.from_string(
        """\
$PROBLEM Full lower triangle OMEGA prior
$SUBROUTINES ADVAN1 TRANS1
$THETA 1
$OMEGA 0.1
$OMEGA 0.2
$SIGMA 0.1
$PRIOR NWPRI NETA=2
$OMEGAP 0.3 0.05 0.4
$OMEGAPD 5 2 10
"""
    )

    problem = Problem.from_control_stream(cs, dataset=_minimal_dataset())

    prior = problem.population_model.prior
    np.testing.assert_allclose(prior.omega_prior, np.array([0.3, 0.05, 0.4]))
    assert prior.omega_prior_cov is not None
    np.testing.assert_allclose(np.diag(prior.omega_prior_cov), np.array([0.2, 0.5, 0.1]))


@pytest.mark.unit
def test_problem_from_control_stream_rejects_missing_omegapd() -> None:
    cs = ControlStream.from_string(
        """\
$PROBLEM Invalid OMEGA prior
$SUBROUTINES ADVAN1 TRANS1
$THETA 1
$OMEGA 0.1
$SIGMA 0.1
$PRIOR NWPRI NETA=1
$OMEGAP 0.3
"""
    )

    with pytest.raises(ParseError, match="\\$OMEGAP and \\$OMEGAPD"):
        Problem.from_control_stream(cs, dataset=_minimal_dataset())
