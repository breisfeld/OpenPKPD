"""Direct tests for PBPK wrapper behavior."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution
from openpkpd.pk.pbpk import FiveOrganPBPK, PBPKModel


class _NoCompartmentsPBPK(PBPKModel):
    pass


class _OrganCentralPBPK(PBPKModel):
    compartment_names = ["organ", "central"]
    output_compartment_name = "central"


class _OrganOutputPBPK(PBPKModel):
    compartment_names = ["organ", "central"]
    output_compartment_name = "organ"


class _MissingOutputPBPK(PBPKModel):
    compartment_names = ["organ", "central"]
    output_compartment_name = "missing"


class _StubAdvan6:
    def __init__(self, solution: PKSolution) -> None:
        self.solution = solution

    def solve(self, *args, **kwargs) -> PKSolution:
        return self.solution


def _zero_des(t, a, pk_params, theta, eta):
    return [0.0 for _ in a]


@pytest.mark.unit
def test_pbpk_subclass_requires_compartment_names():
    with pytest.raises(ValueError, match="compartment_names"):
        _NoCompartmentsPBPK()


@pytest.mark.unit
def test_five_organ_pbpk_exposes_expected_compartment_mapping():
    model = FiveOrganPBPK()

    assert model.n_compartments == 5
    assert model.compartment_index("lung") == 1
    assert model.compartment_index("gut") == 4
    assert model.compartment_index("central") == 5


@pytest.mark.unit
def test_pbpk_solve_uses_named_output_compartment_volume():
    model = _OrganCentralPBPK()
    obs_times = np.array([0.5, 1.0, 2.0])
    dose_events = [DoseEvent(time=0.0, amount=40.0, compartment=2)]

    sol = model.solve({"V_central": 10.0}, dose_events, obs_times, des_callable=_zero_des)

    np.testing.assert_allclose(sol.amounts[:, 1], 40.0, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, 4.0, atol=1e-10)
    np.testing.assert_allclose(sol.f, sol.ipred, atol=1e-10)


@pytest.mark.unit
def test_pbpk_solve_falls_back_to_generic_v_for_noncentral_output():
    model = _OrganOutputPBPK()
    obs_times = np.array([0.25, 1.0])
    dose_events = [DoseEvent(time=0.0, amount=30.0, compartment=1)]

    sol = model.solve({"V": 5.0}, dose_events, obs_times, des_callable=_zero_des)

    np.testing.assert_allclose(sol.amounts[:, 0], 30.0, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, 6.0, atol=1e-10)


@pytest.mark.unit
def test_pbpk_solve_clips_negative_output_predictions_to_zero():
    model = _OrganCentralPBPK()
    model._advan6 = _StubAdvan6(
        PKSolution(
            times=np.array([0.0, 1.0]),
            amounts=np.array([[1.0, -2.0], [1.0, 8.0]]),
            ipred=np.array([99.0, 99.0]),
        )
    )

    sol = model.solve({"V_central": 2.0}, [], np.array([0.0, 1.0]))

    np.testing.assert_allclose(sol.ipred, [0.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(sol.amounts, [[1.0, -2.0], [1.0, 8.0]], atol=1e-12)


@pytest.mark.unit
def test_pbpk_missing_output_name_falls_back_to_first_compartment():
    model = _MissingOutputPBPK()
    model._advan6 = _StubAdvan6(
        PKSolution(
            times=np.array([0.0, 1.0]),
            amounts=np.array([[10.0, 30.0], [5.0, 15.0]]),
            ipred=np.array([-1.0, -1.0]),
        )
    )

    sol = model.solve({"V": 5.0}, [], np.array([0.0, 1.0]))

    assert model._output_idx == 0
    np.testing.assert_allclose(sol.ipred, [2.0, 1.0], atol=1e-12)


@pytest.mark.unit
def test_pbpk_apply_trans_returns_original_micro_parameter_mapping():
    model = FiveOrganPBPK()
    params = {"Q_lung": 12.0, "V_central": 4.0}

    result = model.apply_trans(params, trans=99)

    assert result is params
