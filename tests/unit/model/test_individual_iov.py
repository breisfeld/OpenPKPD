"""
Tests for occasion_indices length validation in IndividualModel (M6).
"""
from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model.individual import IndividualModel
from openpkpd.pk.analytical.advan1 import ADVAN1


def _make_subject_events(n_obs: int = 4) -> SubjectEvents:
    times = np.linspace(0.5, 8.0, n_obs)
    return SubjectEvents(
        subject_id=1,
        obs_times=times,
        obs_dv=np.ones(n_obs, dtype=float),
        obs_cmt=np.ones(n_obs, dtype=int),
        obs_mdv=np.zeros(n_obs, dtype=int),
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
    )


def _make_individual(subject_events, occasion_indices=None):
    advan1 = ADVAN1()

    def pk_callable(theta, eta, t=0.0, a=None, covariates=None):
        return {"CL": 5.0, "V": 50.0}

    return IndividualModel(
        subject_events=subject_events,
        pk_subroutine=advan1,
        pk_callable=pk_callable,
        error_callable=None,
        occasion_indices=occasion_indices,
    )


def test_matching_lengths_no_error():
    """Matching occasion_indices length raises no error and sets _unique_occasions."""
    subj = _make_subject_events(n_obs=4)
    occ = np.array([1, 1, 2, 2])
    indiv = _make_individual(subj, occasion_indices=occ)
    assert indiv._unique_occasions is not None
    np.testing.assert_array_equal(np.sort(indiv._unique_occasions), [1, 2])


def test_mismatched_lengths_raises_value_error():
    """Mismatched lengths raise ValueError mentioning both lengths."""
    subj = _make_subject_events(n_obs=4)
    occ = np.array([1, 2, 3])  # length 3, but n_obs=4
    with pytest.raises(ValueError, match="3") as exc_info:
        _make_individual(subj, occasion_indices=occ)
    assert "4" in str(exc_info.value) or "obs_times" in str(exc_info.value)


def test_none_occasion_indices_no_error():
    """occasion_indices=None sets _unique_occasions=None with no error."""
    subj = _make_subject_events(n_obs=4)
    indiv = _make_individual(subj, occasion_indices=None)
    assert indiv._unique_occasions is None


def test_numerical_two_occasion_mapping():
    """Two-occasion model sets _unique_occasions == [1, 2]."""
    n_obs = 6
    subj = _make_subject_events(n_obs=n_obs)
    occ = np.array([1, 1, 1, 2, 2, 2])
    indiv = _make_individual(subj, occasion_indices=occ)
    assert indiv._unique_occasions is not None
    np.testing.assert_array_equal(np.sort(indiv._unique_occasions), [1, 2])
