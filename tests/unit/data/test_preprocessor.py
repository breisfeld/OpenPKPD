"""Direct tests for data preprocessor helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from openpkpd.data.columns import EVID, ID, MDV, TIME
from openpkpd.data.preprocessor import auto_generate_mdv, sort_by_id_time, validate_monotone_time
from openpkpd.utils.constants import EVID_DOSE, EVID_OBS


@pytest.mark.unit
class TestPreprocessorHelpers:
    def test_auto_generate_mdv_adds_expected_values_when_missing(self):
        df = pd.DataFrame(
            {
                ID: [1, 1, 1],
                TIME: [0.0, 1.0, 2.0],
                EVID: [EVID_OBS, EVID_DOSE, 2],
            }
        )

        result = auto_generate_mdv(df)

        assert MDV not in df.columns
        assert result[MDV].tolist() == [0, 1, 1]

    def test_auto_generate_mdv_preserves_existing_column(self):
        df = pd.DataFrame(
            {
                ID: [1, 1],
                TIME: [0.0, 1.0],
                EVID: [EVID_OBS, EVID_DOSE],
                MDV: [9, 8],
            }
        )

        result = auto_generate_mdv(df)

        assert result[MDV].tolist() == [9, 8]

    def test_sort_by_id_time_sorts_and_preserves_tie_order_stably(self):
        df = pd.DataFrame(
            {
                ID: [2, 1, 1, 2, 1],
                TIME: [2.0, 1.0, 1.0, 1.0, 0.0],
                "ROW": ["late-id2", "tie-a", "tie-b", "early-id2", "first-id1"],
            }
        )

        result = sort_by_id_time(df)

        assert result[ID].tolist() == [1, 1, 1, 2, 2]
        assert result[TIME].tolist() == [0.0, 1.0, 1.0, 1.0, 2.0]
        assert result["ROW"].tolist() == ["first-id1", "tie-a", "tie-b", "early-id2", "late-id2"]

    def test_validate_monotone_time_reports_only_subjects_with_reversals(self):
        df = pd.DataFrame(
            {
                ID: [2, 2, 1, 1, 3, 3],
                TIME: [0.0, 2.0, 1.0, 0.5, 1.0, 1.0],
            }
        )

        warnings = validate_monotone_time(df)

        assert warnings == ["Subject 1: non-monotone TIME values"]
