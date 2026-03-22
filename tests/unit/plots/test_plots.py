"""
Unit tests for the plots module.

Tests:
- compute_diagnostics(): column presence, no-crash, eta fallback
- Smoke test for every plot function (returns Figure)
- Edge cases: no overlay, missing ETA columns
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# pytest.importorskip ensures tests are skipped if matplotlib is absent
# ---------------------------------------------------------------------------
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.plots.diagnostics import compute_diagnostics
from openpkpd.plots.eta import eta_histograms, eta_pairs, eta_vs_covariate
from openpkpd.plots.gof import (
    abs_iwres_vs_ipred,
    cwres_qq,
    cwres_vs_pred,
    cwres_vs_time,
    diagnostic_panel,
    dv_vs_ipred,
    dv_vs_pred,
)
from openpkpd.plots.model_perf import (
    ofv_history,
    parameter_uncertainty_plot,
    residual_trends_plot,
)
from openpkpd.plots.model_perf import vpc as model_perf_vpc
from openpkpd.plots.pd import effect_time, emax_curve, hysteresis_loop, pd_individual
from openpkpd.plots.pk import concentration_time, mean_profile, spaghetti_plot

# ---------------------------------------------------------------------------
# Minimal dataset: 3 subjects, 1-cmt oral
# ---------------------------------------------------------------------------
MINI_DATA = """\
ID,TIME,AMT,DV,EVID,MDV
1,0,4.0,0,1,1
1,1.0,0,2.5,0,0
1,3.0,0,3.8,0,0
1,6.0,0,2.0,0,0
2,0,4.0,0,1,1
2,1.0,0,2.2,0,0
2,3.0,0,3.5,0,0
2,6.0,0,1.8,0,0
3,0,4.0,0,1,1
3,1.0,0,2.8,0,0
3,3.0,0,4.0,0,0
3,6.0,0,2.1,0,0
"""


@pytest.fixture(scope="module")
def simple_result():
    """Run a fast FO estimation on minimal data; return (pop_model, result)."""
    df = pd.read_csv(io.StringIO(MINI_DATA))
    ds = NONMEMDataset.from_dataframe(df)
    built = (
        ModelBuilder()
        .problem("Mini 1-cmt oral")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.3, 0.2, 0.2])
        .sigma(0.1)
        .estimation(method="FO", maxeval=200)
        .build()
    )
    result = built.fit()
    return built.population_model, result


@pytest.fixture(scope="module")
def diag_df(simple_result):
    pop_model, result = simple_result
    return compute_diagnostics(pop_model, result)


# ---------------------------------------------------------------------------
# compute_diagnostics tests
# ---------------------------------------------------------------------------
REQUIRED_COLS = {
    "ID",
    "TIME",
    "DV",
    "PRED",
    "IPRED",
    "RES",
    "IRES",
    "WRES",
    "IWRES",
    "CWRES",
    "MDV",
    "EVID",
}


@pytest.mark.unit
def test_compute_diagnostics_columns(diag_df):
    """All required columns must be present."""
    missing = REQUIRED_COLS - set(diag_df.columns)
    assert not missing, f"Missing columns: {missing}"


@pytest.mark.unit
def test_compute_diagnostics_no_nan_in_core(diag_df):
    """Core prediction columns should not be NaN."""
    for col in ("PRED", "IPRED", "CWRES", "WRES"):
        nan_count = diag_df[col].isna().sum()
        assert nan_count == 0, f"{col} has {nan_count} NaN values"


@pytest.mark.unit
def test_compute_diagnostics_only_observations(diag_df):
    """Returned rows should all be EVID=0, MDV=0."""
    assert (diag_df["EVID"] == 0).all()
    assert (diag_df["MDV"] == 0).all()


@pytest.mark.unit
def test_model_perf_vpc_supports_custom_vpc_quantile_labels(diag_df):
    class CustomQuantileVPCResult:
        observed_df = pd.DataFrame(
            {
                "ID": [1, 1],
                "TIME": [1.0, 2.0],
                "DV": [2.0, 3.0],
            }
        )
        obs_percentiles = pd.DataFrame(
            {
                "bin_mid": [1.0, 2.0],
                "p10": [2.0, 3.0],
                "p50": [2.0, 3.0],
                "p90": [2.0, 3.0],
            }
        )
        sim_percentiles = pd.DataFrame(
            {
                "bin_mid": [1.0, 2.0],
                "p10_lo": [1.5, 2.5],
                "p50_mid": [2.0, 3.0],
                "p90_hi": [2.5, 3.5],
            }
        )
        quantiles = (0.1, 0.5, 0.9)

    fig = model_perf_vpc(diag_df, vpc_result=CustomQuantileVPCResult())

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 4
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = {text.get_text() for text in legend.get_texts()}
    assert {"Obs p10", "Obs p50", "Obs p90", "Sim 50th %ile"}.issubset(labels)
    plt.close(fig)


@pytest.mark.unit
def test_compute_diagnostics_no_post_hoc_etas():
    """With empty post_hoc_etas, diagnostics should run without crashing (eta=0 fallback)."""
    df = pd.read_csv(io.StringIO(MINI_DATA))
    ds = NONMEMDataset.from_dataframe(df)
    built = (
        ModelBuilder()
        .problem("Mini fallback")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.3, 0.2, 0.2])
        .sigma(0.1)
        .estimation(method="FO", maxeval=5)
        .build()
    )
    result = built.fit()
    # Force empty post_hoc_etas
    result.post_hoc_etas = {}
    diag = compute_diagnostics(built.population_model, result)
    assert len(diag) > 0
    assert REQUIRED_COLS.issubset(set(diag.columns))


# ---------------------------------------------------------------------------
# Smoke tests: every plot function should return a Figure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dv_vs_ipred_returns_figure(diag_df):
    fig = dv_vs_ipred(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_dv_vs_ipred_log_scale_draws_identity_line():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "IPRED": [1.0, 2.0, 4.0],
            "DV": [1.5, 2.5, 3.5],
        }
    )

    fig = dv_vs_ipred(df, log_scale=True, title="IPRED check")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_title() == "IPRED check"
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 1.5], [2.0, 2.5], [4.0, 3.5]], dtype=float),
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.85, 4.15])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.85, 4.15])

    plt.close(fig)


@pytest.mark.unit
def test_dv_vs_pred_draws_expected_scatter_and_identity_line():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "PRED": [1.0, 2.0, 4.0],
            "DV": [1.5, 2.5, 3.5],
        }
    )

    fig = dv_vs_pred(df, title="PRED check")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    assert ax.get_xlabel() == "PRED"
    assert ax.get_ylabel() == "DV"
    assert ax.get_title() == "PRED check"
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 1.5], [2.0, 2.5], [4.0, 3.5]], dtype=float),
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.85, 4.15])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.85, 4.15])

    plt.close(fig)


@pytest.mark.unit
def test_cwres_vs_time_draws_scatter_and_reference_bands():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "TIME": [1.0, 2.0, 4.0],
            "CWRES": [-1.5, 0.0, 1.25],
        }
    )

    fig = cwres_vs_time(df, title="CW/T")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "TIME"
    assert ax.get_ylabel() == "CWRES"
    assert ax.get_title() == "CW/T"
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, -1.5], [2.0, 0.0], [4.0, 1.25]], dtype=float),
    )
    assert len(ax.lines) == 3
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 0.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [2.0, 2.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [-2.0, -2.0])

    plt.close(fig)


@pytest.mark.unit
def test_cwres_vs_pred_draws_scatter_and_reference_bands():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "PRED": [0.8, 1.6, 3.2],
            "CWRES": [-0.5, 0.25, 1.5],
        }
    )

    fig = cwres_vs_pred(df, title="CW/P")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "PRED"
    assert ax.get_ylabel() == "CWRES"
    assert ax.get_title() == "CW/P"
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[0.8, -0.5], [1.6, 0.25], [3.2, 1.5]], dtype=float),
    )
    assert len(ax.lines) == 3
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 0.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [2.0, 2.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [-2.0, -2.0])

    plt.close(fig)


@pytest.mark.unit
def test_cwres_qq_returns_figure(diag_df):
    fig = cwres_qq(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_cwres_qq_drops_nan_and_uses_probplot_reference_line():
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    df = pd.DataFrame({"CWRES": [-1.0, 0.0, 1.0, np.nan, 2.0]})

    fig = cwres_qq(df, title="QQ check")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "QQ check"

    finite = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    (osm, osr), (slope, intercept, _r) = probplot(finite, dist="norm")
    offsets = np.asarray(ax.collections[0].get_offsets(), dtype=float)
    np.testing.assert_allclose(offsets[:, 0], osm)
    np.testing.assert_allclose(offsets[:, 1], osr)

    lo, hi = float(np.min(osm)), float(np.max(osm))
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [lo, hi])
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        [slope * lo + intercept, slope * hi + intercept],
    )

    plt.close(fig)


@pytest.mark.unit
def test_abs_iwres_returns_figure(diag_df):
    fig = abs_iwres_vs_ipred(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_abs_iwres_vs_ipred_uses_absolute_values_and_reference_line():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "IPRED": [1.0, 2.0, 3.0],
            "IWRES": [-1.5, 0.25, 2.0],
        }
    )

    fig = abs_iwres_vs_ipred(df, title="Abs IWRES")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Abs IWRES"
    offsets = np.asarray(ax.collections[0].get_offsets(), dtype=float)
    np.testing.assert_allclose(offsets[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(offsets[:, 1], [1.5, 0.25, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 1.0])

    plt.close(fig)


@pytest.mark.unit
def test_diagnostic_panel_returns_figure(diag_df):
    fig = diagnostic_panel(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_diagnostic_panel_populates_all_expected_gof_axes():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "TIME": [0.5, 1.0, 0.5, 1.0],
            "DV": [1.2, 2.1, 1.5, 2.4],
            "PRED": [1.0, 2.0, 1.3, 2.2],
            "IPRED": [1.1, 2.2, 1.4, 2.3],
            "CWRES": [-1.0, 0.0, 0.75, 1.5],
            "IWRES": [-0.5, 0.25, -1.25, 1.75],
        }
    )

    fig = diagnostic_panel(df, title="GOF review")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 8
    assert fig._suptitle.get_text() == "GOF review"
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible_axes] == [
        "DV vs IPRED",
        "DV vs PRED",
        "CWRES vs TIME",
        "CWRES vs PRED",
        "CWRES Q-Q",
        "CWRES Histogram",
        "|IWRES| vs IPRED",
    ]

    plt.close(fig)


@pytest.mark.unit
def test_concentration_time_returns_figure(diag_df):
    fig = concentration_time(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_spaghetti_plot_returns_figure(diag_df):
    fig = spaghetti_plot(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_mean_profile_returns_figure(diag_df):
    fig = mean_profile(diag_df)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_concentration_time_sorts_subject_profiles_and_highlights_mean_overlay():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from openpkpd.plots._core import _IBM_COLORS

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "DV": [30.0, 20.0, 12.0, 10.0],
            "IPRED": [32.0, 22.0, 14.0, 11.0],
        }
    )

    fig = concentration_time(df, log_y=True, highlight_ids=[1], title="CT")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "CT"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"
    assert ax.get_yscale() == "log"
    assert len(ax.collections) == 2
    assert len(ax.lines) == 3
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 20.0], [2.0, 30.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(ax.collections[1].get_offsets(), dtype=float),
        np.array([[1.0, 10.0], [2.0, 12.0]]),
    )
    np.testing.assert_allclose(
        ax.collections[0].get_facecolors()[0], to_rgba("steelblue", alpha=0.4)
    )
    np.testing.assert_allclose(
        ax.collections[1].get_facecolors()[0], to_rgba(_IBM_COLORS[2], alpha=0.9)
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [22.0, 32.0])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [11.0, 14.0])
    np.testing.assert_allclose(ax.lines[2].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [16.5, 23.0])
    assert [text.get_text() for text in ax.get_legend().texts] == ["Mean IPRED"]

    plt.close(fig)


@pytest.mark.unit
def test_spaghetti_plot_sorts_profiles_and_respects_alpha_without_mean_overlay():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "IPRED": [32.0, 22.0, 14.0, 11.0],
        }
    )

    fig = spaghetti_plot(df, log_y=True, mean_overlay=False, alpha=0.2, title="SP")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "SP"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 2
    assert ax.get_legend() is None
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [22.0, 32.0])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [11.0, 14.0])
    assert ax.lines[0].get_alpha() == pytest.approx(0.2)
    assert ax.lines[1].get_alpha() == pytest.approx(0.2)

    plt.close(fig)


@pytest.mark.unit
def test_mean_profile_plots_group_mean_and_clipped_sd_band():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "IPRED": [6.0, 1.0, 4.0, 0.0],
        }
    )

    fig = mean_profile(df, log_y=True, sd_band=True, title="MP")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "MP"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.5, 5.0])
    band_vertices = ax.collections[0].get_paths()[0].vertices
    assert np.min(band_vertices[:, 1]) == pytest.approx(0.0)
    assert np.max(band_vertices[:, 1]) == pytest.approx(6.414213562373095)
    assert [text.get_text() for text in ax.get_legend().texts] == ["Mean IPRED", "±1 SD"]

    plt.close(fig)


@pytest.mark.unit
def test_ofv_history_plots_iteration_vs_ofv(simple_result):
    _, result = simple_result
    fig = ofv_history(result, title="OFV trace")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "OFV trace"
    assert ax.get_xlabel() == "Iteration"
    assert ax.get_ylabel() == "OFV"
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.arange(1, len(result.ofv_history) + 1))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), result.ofv_history)

    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.unit
def test_ofv_history_empty_no_crash():
    """ofv_history with empty list should show the fallback message."""
    import matplotlib.pyplot as plt

    from openpkpd.estimation.base import EstimationResult

    result = EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=0.0,
        ofv_history=[],
    )
    fig = ofv_history(result)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 0
    assert ax.get_title() == "OFV History"
    assert ax.texts[0].get_text() == "No OFV history available"
    plt.close(fig)


@pytest.mark.unit
def test_parameter_uncertainty_plot_prefers_standard_errors_over_covariance():
    import matplotlib.pyplot as plt

    class _Result:
        theta_final = np.array([1.0, 2.0], dtype=float)
        standard_errors = np.array([0.1, 0.2], dtype=float)
        covariance_matrix = np.diag([9.0, 16.0])

    fig = parameter_uncertainty_plot(_Result(), title="Uncertainty")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["THETA(1)", "THETA(2)"]
    assert any("±1.96 SE" in text.get_text() for text in ax.texts)

    errorbar = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
    bar_container = next(c for c in ax.containers if isinstance(c, BarContainer))
    assert [patch.get_width() for patch in bar_container.patches] == pytest.approx([1.0, 2.0])

    segments = errorbar.lines[2][0].get_segments()
    expected = [
        [1.0 - 1.96 * 0.1, 1.0 + 1.96 * 0.1],
        [2.0 - 1.96 * 0.2, 2.0 + 1.96 * 0.2],
    ]
    for seg, exp in zip(segments, expected, strict=False):
        assert seg[:, 0] == pytest.approx(exp)

    plt.close(fig)


@pytest.mark.unit
def test_parameter_uncertainty_plot_covariance_fallback_clips_negative_diagonal():
    import matplotlib.pyplot as plt

    class _Result:
        theta_final = np.array([1.0, 2.0], dtype=float)
        covariance_matrix = np.array([[0.04, 0.0], [0.0, -1.0]], dtype=float)

    fig = parameter_uncertainty_plot(_Result())

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    errorbar = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
    segments = errorbar.lines[2][0].get_segments()
    assert segments[0][:, 0] == pytest.approx([1.0 - 1.96 * 0.2, 1.0 + 1.96 * 0.2])
    assert segments[1][:, 0] == pytest.approx([2.0, 2.0])

    plt.close(fig)


@pytest.mark.unit
def test_parameter_uncertainty_plot_warns_without_uncertainty_information():
    import matplotlib.pyplot as plt

    class _Result:
        theta_final = np.array([1.0, 2.0], dtype=float)

    with pytest.warns(UserWarning, match="No covariance information found"):
        fig = parameter_uncertainty_plot(_Result())

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    bar_container = next(c for c in ax.containers if isinstance(c, BarContainer))
    assert [patch.get_width() for patch in bar_container.patches] == pytest.approx([1.0, 2.0])
    assert not any("±1.96 SE" in text.get_text() for text in ax.texts)

    plt.close(fig)


@pytest.mark.unit
def test_residual_trends_plot_raises_without_cwres():
    df = pd.DataFrame({"TIME": [1.0, 2.0], "PRED": [0.9, 1.1]})

    with pytest.raises(ValueError, match="CWRES column not found"):
        residual_trends_plot(df)


@pytest.mark.unit
def test_residual_trends_plot_raises_without_any_supported_x_column():
    df = pd.DataFrame({"CWRES": [0.0, 1.0, -1.0]})

    with pytest.raises(ValueError, match="None of TIME, PRED, IPRED columns found"):
        residual_trends_plot(df)


@pytest.mark.unit
def test_residual_trends_plot_uses_available_columns_and_filters_nonfinite_pairs():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "CWRES": [0.0, 1.0, -1.0, np.nan, 0.5],
            "TIME": [0.0, 1.0, 2.0, 3.0, 4.0],
            "PRED": [1.0, np.nan, 2.0, 3.0, 4.0],
        }
    )

    fig = residual_trends_plot(df, title="Residual check")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig._suptitle.get_text() == "Residual check"

    time_ax, pred_ax = fig.axes
    assert time_ax.get_title() == "CWRES vs TIME"
    assert pred_ax.get_title() == "CWRES vs PRED"
    assert len(time_ax.collections[0].get_offsets()) == 4
    assert len(pred_ax.collections[0].get_offsets()) == 3
    assert len(time_ax.lines) == 4
    assert len(pred_ax.lines) == 3

    plt.close(fig)


# ---------------------------------------------------------------------------
# effect_time / emax_curve with synthetic PD column
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pd_diag_df(diag_df):
    """Add a synthetic effect column to diag_df for PD plot tests."""
    df = diag_df.copy()
    rng = np.random.default_rng(0)
    emax = 10.0
    ec50 = 5.0
    c = df["IPRED"].values
    df["EFFECT"] = emax * c / (ec50 + c) + rng.normal(0, 0.5, len(c))
    return df


@pytest.mark.unit
def test_effect_time_without_overlays_draws_subject_scatter_only(pd_diag_df):
    import matplotlib.pyplot as plt

    fig = effect_time(pd_diag_df, "EFFECT", individual=False, mean_overlay=False, title="ET raw")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "ET raw"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "EFFECT"
    assert len(ax.lines) == 0
    assert len(ax.collections) == pd_diag_df["ID"].nunique()
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.unit
def test_emax_curve_with_overlay(pd_diag_df):
    import matplotlib.pyplot as plt

    fig = emax_curve(pd_diag_df, "IPRED", "EFFECT", emax=10.0, ec50=5.0)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_emax_curve_no_overlay(pd_diag_df):
    """emax_curve without emax/ec50 should only show the observed scatter."""
    import matplotlib.pyplot as plt

    fig = emax_curve(pd_diag_df, "IPRED", "EFFECT")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Emax Curve"
    assert ax.get_xlabel() == "IPRED"
    assert ax.get_ylabel() == "EFFECT"
    assert len(ax.collections) == 1
    assert len(ax.lines) == 0
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        pd_diag_df[["IPRED", "EFFECT"]].to_numpy(dtype=float),
    )
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.unit
def test_hysteresis_loop_without_time_coloring_omits_colorbar(pd_diag_df):
    import matplotlib.pyplot as plt

    fig = hysteresis_loop(pd_diag_df, "IPRED", "EFFECT", color_by_time=False, title="HY raw")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == "HY raw"
    assert ax.get_xlabel() == "IPRED"
    assert ax.get_ylabel() == "EFFECT"
    assert len(ax.collections) == pd_diag_df["ID"].nunique()
    assert len(ax.lines) == pd_diag_df["ID"].nunique()
    plt.close(fig)


@pytest.mark.unit
def test_effect_time_sorts_profiles_and_adds_mean_overlay():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "EFFECT": [5.0, 3.0, 2.0, 1.0],
        }
    )

    fig = effect_time(df, "EFFECT", individual=True, mean_overlay=True, title="ET")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "ET"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "EFFECT"
    assert len(ax.collections) == 2
    assert len(ax.lines) == 3
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 3.0], [2.0, 5.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(ax.collections[1].get_offsets(), dtype=float),
        np.array([[1.0, 1.0], [2.0, 2.0]]),
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 5.0])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [2.0, 3.5])
    assert [text.get_text() for text in ax.get_legend().texts] == ["Mean"]

    plt.close(fig)


@pytest.mark.unit
def test_emax_curve_overlay_uses_gamma_and_e0_formula():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "IPRED": [1.0, 3.0, 5.0],
            "EFFECT": [2.0, 4.0, 5.0],
        }
    )

    fig = emax_curve(df, "IPRED", "EFFECT", emax=10.0, ec50=5.0, gamma=2.0, e0=1.0, title="EM")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "EM"
    assert ax.get_xlabel() == "IPRED"
    assert ax.get_ylabel() == "EFFECT"
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 5.0]]),
    )
    assert [text.get_text() for text in ax.get_legend().texts] == ["Observed", "Emax=10, EC50=5"]
    assert ax.lines[0].get_xdata()[0] == pytest.approx(0.0)
    assert ax.lines[0].get_xdata()[-1] == pytest.approx(5.5)
    assert ax.lines[0].get_ydata()[0] == pytest.approx(1.0)
    assert ax.lines[0].get_ydata()[-1] == pytest.approx(
        1.0 + 10.0 * (5.5**2) / ((5.0**2) + (5.5**2))
    )

    plt.close(fig)


@pytest.mark.unit
def test_hysteresis_loop_sorts_subjects_and_adds_time_colorbar():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "IPRED": [5.0, 4.0, 2.0, 1.0],
            "EFFECT": [7.0, 6.0, 3.0, 2.0],
        }
    )

    fig = hysteresis_loop(df, "IPRED", "EFFECT", color_by_time=True, title="HY")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    ax = fig.axes[0]
    colorbar_ax = fig.axes[1]
    assert ax.get_title() == "HY"
    assert ax.get_xlabel() == "IPRED"
    assert ax.get_ylabel() == "EFFECT"
    assert colorbar_ax.get_ylabel() == "Time"
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets(), dtype=float),
        np.array([[4.0, 6.0], [5.0, 7.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(ax.collections[1].get_offsets(), dtype=float),
        np.array([[1.0, 2.0], [2.0, 3.0]]),
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [4.0, 5.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [6.0, 7.0])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [2.0, 3.0])

    plt.close(fig)


@pytest.mark.unit
def test_pd_individual_uses_sorted_default_subject_order_and_hides_unused_axes():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [3, 3, 1, 1, 2, 2],
            "TIME": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "IPRED": [9.0, 8.0, 3.0, 2.0, 6.0, 5.0],
            "DV": [9.5, 8.5, 3.5, 2.5, 6.5, 5.5],
            "EFFECT": [12.0, 11.0, 4.0, 3.0, 8.0, 7.0],
        }
    )

    fig = pd_individual(df, "IPRED", "EFFECT", n_cols=2, title="PDI3")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 8
    assert fig._suptitle.get_text() == "PDI3"
    assert fig.axes[0].get_title() == "Subject 1"
    assert fig.axes[1].get_title() == "Subject 2"
    assert fig.axes[4].get_title() == "Subject 3"
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [2.0, 3.0])
    np.testing.assert_allclose(
        np.asarray(fig.axes[0].collections[0].get_offsets(), dtype=float),
        np.array([[1.0, 2.5], [2.0, 3.5]]),
    )
    assert fig.axes[0].get_ylabel() == "Conc"
    assert fig.axes[2].get_xlabel() == "Time"
    assert fig.axes[2].get_ylabel() == "EFFECT"
    assert fig.axes[5].get_visible() is False
    assert fig.axes[7].get_visible() is False

    plt.close(fig)


# ---------------------------------------------------------------------------
# ETA plots
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eta_histograms_returns_figure(diag_df, simple_result):
    import matplotlib.pyplot as plt

    _, result = simple_result
    eta_cols = [c for c in diag_df.columns if c.startswith("ETA")]
    if not eta_cols:
        pytest.skip("No ETA columns in diag_df (FO result)")
    fig = eta_histograms(diag_df, result.omega_final)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_eta_histograms_no_etas_raises():
    """eta_histograms with no ETAn columns should raise ValueError."""
    df = pd.DataFrame(
        {
            "ID": [1],
            "TIME": [1.0],
            "DV": [1.0],
            "PRED": [1.0],
            "IPRED": [1.0],
            "CWRES": [0.1],
            "WRES": [0.1],
            "IWRES": [0.1],
            "RES": [0.0],
            "IRES": [0.0],
            "MDV": [0],
            "EVID": [0],
        }
    )
    with pytest.raises(ValueError, match="No ETAn columns"):
        eta_histograms(df, np.eye(1))


@pytest.mark.unit
def test_eta_pairs_no_etas_raises():
    df = pd.DataFrame({"ID": [1], "TIME": [1.0]})
    with pytest.raises(ValueError, match="No ETAn columns"):
        eta_pairs(df)


@pytest.mark.unit
def test_eta_pairs_returns_figure(diag_df):
    import matplotlib.pyplot as plt

    eta_cols = [c for c in diag_df.columns if c.startswith("ETA")]
    if not eta_cols:
        pytest.skip("No ETA columns in diag_df (FO result)")
    fig = eta_pairs(diag_df)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_eta_vs_covariate_continuous_uses_first_row_per_subject_and_zero_line():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "WT": [80.0, 81.0, 70.0, 71.0],
            "ETA1": [0.2, 9.9, -0.4, 8.8],
        }
    )

    fig = eta_vs_covariate(df, "WT", "ETA1")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "ETA1 vs WT"
    assert ax.get_xlabel() == "WT"
    assert ax.get_ylabel() == "ETA1"
    offsets = np.asarray(ax.collections[0].get_offsets(), dtype=float)
    np.testing.assert_allclose(offsets, np.array([[70.0, -0.4], [80.0, 0.2]]))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 0.0])

    plt.close(fig)


@pytest.mark.unit
def test_eta_vs_covariate_categorical_sorts_groups_and_uses_boxplot():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1, 3, 3],
            "SEX": [2, 2, 1, 1, 2, 2],
            "ETA2": [0.3, 5.0, -0.2, 4.0, 0.6, 6.0],
        }
    )

    fig = eta_vs_covariate(df, "SEX", "ETA2", categorical=True, title="Categorical ETA")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Categorical ETA"
    assert ax.get_xlabel() == "SEX"
    assert ax.get_ylabel() == "ETA2"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1", "2"]

    plt.close(fig)


@pytest.mark.unit
def test_eta_histograms_overlay_normal_and_hide_extra_axes():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "ETA1": [-0.5, 9.0, 0.5, 8.0],
            "ETA2": [0.25, 7.0, -0.25, 6.0],
        }
    )

    fig = eta_histograms(df, np.array([[0.25, 0.0], [0.0, 0.0]]), n_cols=3, title="ETA hist")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert fig._suptitle.get_text() == "ETA hist"
    assert fig.axes[0].get_xlabel() == "ETA1"
    assert fig.axes[1].get_xlabel() == "ETA2"
    assert len(fig.axes[0].patches) > 0
    assert len(fig.axes[1].patches) > 0
    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[1].lines) == 1
    assert fig.axes[2].get_visible() is False

    plt.close(fig)


@pytest.mark.unit
def test_eta_pairs_builds_diagonal_histograms_and_off_diagonal_scatter():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "ETA1": [0.3, 9.0, -0.1, 8.0],
            "ETA2": [1.2, 7.0, 0.8, 6.0],
        }
    )

    fig = eta_pairs(df, title="ETA pairs")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert fig._suptitle.get_text() == "ETA pairs"
    assert len(fig.axes[0].patches) > 0
    assert len(fig.axes[3].patches) > 0
    assert len(fig.axes[1].collections) == 1
    assert len(fig.axes[1].lines) == 2
    assert len(fig.axes[2].collections) == 1
    assert len(fig.axes[2].lines) == 2
    np.testing.assert_allclose(
        np.asarray(fig.axes[1].collections[0].get_offsets(), dtype=float),
        np.array([[0.8, -0.1], [1.2, 0.3]]),
    )
    assert fig.axes[2].get_xlabel() == "ETA1"
    assert fig.axes[2].get_ylabel() == "ETA2"
    assert fig.axes[3].get_xlabel() == "ETA2"

    plt.close(fig)
