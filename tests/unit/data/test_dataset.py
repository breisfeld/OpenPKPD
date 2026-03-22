"""Unit tests for NONMEMDataset."""

import numpy as np
import pytest

from openpkpd.data.columns import DV, EVID, ID, MDV, TIME
from openpkpd.data.dataset import NONMEMDataset


@pytest.mark.unit
class TestNONMEMDataset:
    def test_from_dataframe_basic(self, theophylline_df):
        ds = NONMEMDataset.from_dataframe(theophylline_df)
        assert ds.n_subjects() == 3
        assert ID in ds.df.columns
        assert TIME in ds.df.columns
        assert DV in ds.df.columns

    def test_sorted_by_id_time(self, theophylline_df):
        """Dataset should be sorted by ID then TIME."""
        ds = NONMEMDataset.from_dataframe(theophylline_df)
        for sid, grp in ds.df.groupby(ID):
            times = grp[TIME].values
            assert np.all(np.diff(times) >= 0), f"Subject {sid} times not sorted"

    def test_evid_added(self, theophylline_df):
        """EVID column should exist after preprocessing."""
        ds = NONMEMDataset.from_dataframe(theophylline_df)
        assert EVID in ds.df.columns

    def test_mdv_added(self, theophylline_df):
        """MDV column should be auto-generated."""
        df = theophylline_df.drop(columns=["MDV"], errors="ignore")
        ds = NONMEMDataset.from_dataframe(df)
        assert MDV in ds.df.columns

    def test_subject_data(self, theophylline_dataset):
        """subject_data() should return only that subject's rows."""
        subj_df = theophylline_dataset.subject_data(1)
        assert (subj_df[ID] == 1).all()

    def test_observation_rows(self, theophylline_dataset):
        """observation_rows() should exclude dosing rows."""
        obs = theophylline_dataset.observation_rows(1)
        assert (obs[EVID] == 0).all()
        assert (obs[MDV] == 0).all()

    def test_n_subjects(self, theophylline_dataset):
        assert theophylline_dataset.n_subjects() == 3

    def test_subject_ids(self, theophylline_dataset):
        assert sorted(theophylline_dataset.subject_ids()) == [1, 2, 3]

    def test_from_csv_preserves_drop_positions_and_removes_dropped_columns(self, tmp_path):
        dataset_path = tmp_path / "drop.csv"
        dataset_path.write_text("1,0,999,70,5\n", encoding="utf-8")

        ds = NONMEMDataset.from_csv(
            str(dataset_path),
            input_columns=["ID", "TIME", "_DROP_3", "WT", "DV"],
            sep=",",
        )

        assert list(ds.df.columns[:4]) == [ID, TIME, "WT", DV]
        assert "_DROP_3" not in ds.df.columns
        assert ds.df.loc[0, "WT"] == 70
        assert ds.df.loc[0, DV] == 5


@pytest.mark.unit
class TestEventProcessor:
    def test_dose_events_extracted(self, theophylline_dataset):
        from openpkpd.data.event_processor import EventProcessor

        ep = EventProcessor()
        events = ep.process(theophylline_dataset.df)
        for sid in [1, 2, 3]:
            subj = events[sid]
            # Should have at least one dose event
            assert len(subj.dose_events) >= 1
            # All dose events should have AMT > 0
            for ev in subj.dose_events:
                assert ev.amount > 0

    def test_obs_times_match(self, theophylline_dataset):
        from openpkpd.data.event_processor import EventProcessor

        ep = EventProcessor()
        events = ep.process(theophylline_dataset.df)
        subj = events[1]
        # All obs_times should be from EVID=0 rows
        obs_rows = theophylline_dataset.observation_rows(1)
        np.testing.assert_allclose(
            np.sort(subj.obs_times),
            np.sort(obs_rows[TIME].values),
            rtol=1e-8,
        )
