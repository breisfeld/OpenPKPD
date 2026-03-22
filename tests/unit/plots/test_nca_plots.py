"""Tests for NCA plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.plots.nca import nca_boxplot, nca_distributions


def _visible_xlabels(fig) -> list[str]:
    return [ax.get_xlabel() for ax in fig.axes if ax.get_visible()]


@pytest.mark.unit
def test_default_nca_plots_include_c0_when_available() -> None:
    import matplotlib.pyplot as plt

    nca_df = pd.DataFrame(
        {
            "c0": [2.1, 2.4, 2.8],
            "cmax": [1.5, 1.7, 1.9],
            "auc_last": [10.0, 11.0, 12.0],
        }
    )

    hist_fig = nca_distributions(nca_df, n_cols=2)
    box_fig = nca_boxplot(nca_df)

    assert _visible_xlabels(hist_fig) == ["C0", "Cmax", "AUC_last"]
    assert [ax.get_title() for ax in box_fig.axes] == ["C0", "Cmax", "AUC_last"]

    plt.close(hist_fig)
    plt.close(box_fig)


@pytest.mark.unit
def test_default_nca_plots_skip_all_nan_c0() -> None:
    import matplotlib.pyplot as plt

    nca_df = pd.DataFrame(
        {
            "c0": [np.nan, np.nan, np.nan],
            "cmax": [1.5, 1.7, 1.9],
            "auc_last": [10.0, 11.0, 12.0],
        }
    )

    hist_fig = nca_distributions(nca_df, n_cols=2)
    box_fig = nca_boxplot(nca_df)

    assert _visible_xlabels(hist_fig) == ["Cmax", "AUC_last"]
    assert [ax.get_title() for ax in box_fig.axes] == ["Cmax", "AUC_last"]

    plt.close(hist_fig)
    plt.close(box_fig)
