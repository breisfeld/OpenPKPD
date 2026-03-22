"""Direct tests for EventProcessor edge-case semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.columns import ADDL, AMT, CMT, DV, EVID, ID, II, MDV, RATE, SS, TIME
from openpkpd.data.event_processor import EventProcessor
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.utils.constants import EVID_DOSE, EVID_OBS, EVID_OTHER, EVID_RESET, EVID_RESET_DOSE


def _row(**overrides):
    row = {
        ID: 1,
        TIME: 0.0,
        DV: np.nan,
        AMT: 0.0,
        EVID: EVID_OBS,
        MDV: 0,
        CMT: 1,
        RATE: 0.0,
        ADDL: 0,
        II: 0.0,
        SS: 0,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_event_processor_records_other_events_as_missing_observations_with_occasions():
    df = pd.DataFrame(
        [
            _row(TIME=2.0, EVID=EVID_OBS, DV=7.5, MDV=0, CMT=2, OCC=np.nan),
            _row(TIME=1.0, EVID=EVID_OTHER, CMT=3, OCC=4),
        ]
    )

    subj = EventProcessor().process(df)[1]

    np.testing.assert_allclose(subj.obs_times, [1.0, 2.0])
    assert np.isnan(subj.obs_dv[0])
    assert subj.obs_dv[1] == pytest.approx(7.5)
    np.testing.assert_array_equal(subj.obs_mdv, [1, 0])
    np.testing.assert_array_equal(subj.obs_cmt, [3, 2])
    np.testing.assert_array_equal(subj.occasion_indices, [4, 0])
    np.testing.assert_array_equal(subj.observation_mask(), [False, True])


@pytest.mark.unit
def test_event_processor_accepts_integer_dv_columns_without_nan_cast_errors():
    df = pd.DataFrame(
        [
            _row(TIME=0.0, DV=0, EVID=EVID_OBS, MDV=0),
            _row(TIME=1.0, DV=5, EVID=EVID_OBS, MDV=0),
        ]
    )

    subj = EventProcessor().process(df)[1]

    assert subj.obs_dv.dtype.kind == "f"
    np.testing.assert_allclose(subj.obs_dv, [0.0, 5.0])


@pytest.mark.unit
def test_event_processor_preserves_same_time_reset_order_and_expands_reset_dose_addl():
    df = pd.DataFrame(
        [
            _row(TIME=5.0, EVID=EVID_RESET, MDV=1),
            _row(TIME=5.0, EVID=EVID_DOSE, AMT=20.0, CMT=2, MDV=1),
            _row(TIME=6.0, EVID=EVID_RESET_DOSE, AMT=50.0, CMT=2, SS=1, ADDL=2, II=12.0, MDV=1),
        ]
    )

    subj = EventProcessor().process(df)[1]
    events = subj.dose_events

    assert [ev.time for ev in events] == [5.0, 5.0, 6.0, 18.0, 30.0]
    assert events[0].reset is True and events[0].amount == pytest.approx(0.0)
    assert events[1].reset is False and events[1].amount == pytest.approx(20.0)
    assert events[2].reset is True and events[2].ss is True
    assert all(ev.reset is False for ev in events[3:])
    assert all(ev.ss is False for ev in events[3:])
    assert all(ev.ii == pytest.approx(12.0) for ev in events[2:])


@pytest.mark.unit
def test_event_processor_duration_based_infusion_matches_equivalent_positive_rate():
    df = pd.DataFrame(
        [
            _row(TIME=0.0, EVID=EVID_DOSE, AMT=100.0, RATE=-1.0, DUR=2.0, MDV=1),
        ]
    )
    obs_times = np.array([0.5, 1.0, 2.0, 4.0])

    processed_events = EventProcessor().process(df)[1].dose_events
    params = {"K": 0.2, "V": 10.0}
    model = ADVAN1()

    sol_processed = model.solve(params, processed_events, obs_times)
    sol_expected = model.solve(
        params,
        [
            processed_events[0].__class__(
                time=0.0, amount=100.0, rate=50.0, duration=2.0, compartment=1
            )
        ],
        obs_times,
    )

    assert processed_events[0].rate == pytest.approx(50.0)
    assert processed_events[0].infusion_end_time == pytest.approx(2.0)
    np.testing.assert_allclose(sol_processed.ipred, sol_expected.ipred, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_event_processor_preserves_per_observation_covariates_at_duplicate_times():
    df = pd.DataFrame(
        [
            _row(TIME=1.0, DV=7.0, CMT=2, DVID=1, WT=70.0),
            _row(TIME=1.0, DV=42.0, CMT=3, DVID=2, WT=70.0),
        ]
    )

    subj = EventProcessor(covariate_columns=["DVID", "WT"]).process(df)[1]

    assert subj.obs_covariates is not None
    assert subj.observation_covariates_at(0)["DVID"] == pytest.approx(1.0)
    assert subj.observation_covariates_at(1)["DVID"] == pytest.approx(2.0)
    assert subj.observation_covariates_at(0)["WT"] == pytest.approx(70.0)
    assert subj.observation_covariates_at(1)["WT"] == pytest.approx(70.0)
