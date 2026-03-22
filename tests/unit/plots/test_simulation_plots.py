"""
Unit tests for openpkpd.plots.simulation — simulation-based diagnostic plots.

Tests:
  - Module imports correctly with all expected attributes.
  - vpc_plot(), npde_plot(), simulation_panel(), prediction_interval_plot()
    all return Figure objects with synthetic data.
  - Edge cases: empty inputs, missing columns (handled gracefully).
  - prediction_interval_plot() validates array length consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection

# ---------------------------------------------------------------------------
# Ensure matplotlib is present; skip if not
# ---------------------------------------------------------------------------
pytest.importorskip("matplotlib")

from matplotlib.figure import Figure


def _fill_between_bounds(
    collection: PolyCollection, xs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover lower/upper y-values at given x positions from fill_between output."""
    verts = collection.get_paths()[0].vertices
    lower = []
    upper = []
    for x in np.asarray(xs, dtype=float):
        mask = np.isclose(verts[:, 0], x)
        if not np.any(mask):
            raise AssertionError(f"x={x} not found in fill_between polygon")
        ys = verts[mask, 1]
        lower.append(float(np.min(ys)))
        upper.append(float(np.max(ys)))
    return np.asarray(lower), np.asarray(upper)


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simulation_plot_import():
    """Check simulation plots module imports correctly with required attributes."""
    from openpkpd.plots import simulation

    assert hasattr(simulation, "vpc_plot"), "vpc_plot missing from simulation module"
    assert hasattr(simulation, "npde_plot"), "npde_plot missing from simulation module"
    assert hasattr(simulation, "simulation_panel"), "simulation_panel missing"
    assert hasattr(simulation, "prediction_interval_plot"), "prediction_interval_plot missing"


@pytest.mark.unit
def test_vpc_plot_is_callable():
    """vpc_plot should be a callable function."""
    from openpkpd.plots.simulation import vpc_plot

    assert callable(vpc_plot)


@pytest.mark.unit
def test_npde_plot_is_callable():
    """npde_plot should be a callable function."""
    from openpkpd.plots.simulation import npde_plot

    assert callable(npde_plot)


@pytest.mark.unit
def test_simulation_panel_is_callable():
    """simulation_panel should be a callable function."""
    from openpkpd.plots.simulation import simulation_panel

    assert callable(simulation_panel)


@pytest.mark.unit
def test_prediction_interval_plot_is_callable():
    """prediction_interval_plot should be a callable function."""
    from openpkpd.plots.simulation import prediction_interval_plot

    assert callable(prediction_interval_plot)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_times() -> np.ndarray:
    """Observation time points (hours post-dose)."""
    return np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


@pytest.fixture()
def synthetic_obs(synthetic_times: np.ndarray) -> np.ndarray:
    """Observed concentrations following a biexponential-like decay."""
    rng = np.random.default_rng(0)
    conc = 5.0 * np.exp(-0.3 * synthetic_times) + rng.normal(0, 0.2, len(synthetic_times))
    return np.maximum(conc, 0.01)


@pytest.fixture()
def synthetic_pi(synthetic_times: np.ndarray, synthetic_obs: np.ndarray):
    """Synthetic lower, median, upper prediction interval arrays."""
    median = 5.0 * np.exp(-0.3 * synthetic_times)
    lo = median * 0.6
    hi = median * 1.6
    return lo, median, hi


@pytest.fixture()
def mock_vpc_result():
    """Minimal mock VPCResult with required attributes."""
    times = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    obs_dv = np.array([4.5, 3.8, 2.5, 1.8, 1.0, 0.3])

    # observed_df
    obs_df = pd.DataFrame(
        {
            "ID": [1] * len(times),
            "TIME": times,
            "DV": obs_dv,
            "REP": [0] * len(times),
            "MDV": [0] * len(times),
        }
    )

    # obs_percentiles
    obs_pct = pd.DataFrame(
        {
            "bin_mid": [1.5, 6.0, 18.0],
            "p5": [0.5, 0.3, 0.1],
            "p50": [3.0, 1.5, 0.5],
            "p95": [5.5, 3.0, 1.2],
            "n": [8, 8, 8],
        }
    )

    # sim_percentiles
    sim_pct = pd.DataFrame(
        {
            "bin_mid": [1.5, 6.0, 18.0],
            "p5_lo": [0.3, 0.2, 0.05],
            "p5_mid": [0.5, 0.3, 0.1],
            "p5_hi": [0.7, 0.4, 0.15],
            "p50_lo": [2.5, 1.2, 0.4],
            "p50_mid": [3.0, 1.5, 0.5],
            "p50_hi": [3.5, 1.8, 0.6],
            "p95_lo": [5.0, 2.5, 1.0],
            "p95_mid": [5.5, 3.0, 1.2],
            "p95_hi": [6.0, 3.5, 1.4],
        }
    )

    class _MockVPCResult:
        observed_df = obs_df
        obs_percentiles = obs_pct
        sim_percentiles = sim_pct
        n_replicates = 100

    return _MockVPCResult()


@pytest.fixture()
def npde_df() -> pd.DataFrame:
    """Synthetic DataFrame with NPDE, TIME, PRED columns."""
    rng = np.random.default_rng(1)
    n = 30
    return pd.DataFrame(
        {
            "ID": np.repeat([1, 2, 3], n // 3),
            "TIME": np.tile(np.linspace(0.5, 24, n // 3), 3),
            "PRED": rng.uniform(0.5, 8.0, n),
            "IPRED": rng.uniform(0.5, 8.0, n),
            "DV": rng.uniform(0.2, 9.0, n),
            "CWRES": rng.normal(0, 1, n),
            "NPDE": rng.normal(0, 1, n),
        }
    )


@pytest.fixture()
def simulated_df() -> pd.DataFrame:
    """Synthetic simulated DataFrame with REP, ID, TIME, DV columns."""
    rng = np.random.default_rng(2)
    rows = []
    for rep in range(0, 11):  # REP 0 = observed, 1-10 = simulated
        for sid in range(1, 4):
            for t in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]:
                dv = max(0.0, 5.0 * np.exp(-0.3 * t) + rng.normal(0, 0.5))
                rows.append({"ID": sid, "TIME": t, "DV": dv, "REP": rep, "MDV": 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# prediction_interval_plot tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prediction_interval_plot_returns_figure(synthetic_times, synthetic_obs, synthetic_pi):
    """prediction_interval_plot should draw the expected band, median, and points."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot

    lo, med, hi = synthetic_pi
    fig = prediction_interval_plot(synthetic_times, synthetic_obs, lo, med, hi)
    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() == "Prediction Interval Plot"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"

    np.testing.assert_allclose(ax.lines[0].get_xdata(), synthetic_times)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), med)
    assert ax.lines[0].get_label() == "Sim Median"

    scatter = next(coll for coll in ax.collections if isinstance(coll, PathCollection))
    np.testing.assert_allclose(
        scatter.get_offsets(), np.column_stack([synthetic_times, synthetic_obs])
    )

    band = next(coll for coll in ax.collections if isinstance(coll, PolyCollection))
    band_lo, band_hi = _fill_between_bounds(band, synthetic_times)
    np.testing.assert_allclose(band_lo, lo)
    np.testing.assert_allclose(band_hi, hi)

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Prediction Interval",
        "Sim Median",
        "Observed",
    ]
    plt.close(fig)


@pytest.mark.unit
def test_prediction_interval_plot_with_ax(synthetic_times, synthetic_obs, synthetic_pi):
    """prediction_interval_plot should accept an existing axes object."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot

    fig0, ax0 = plt.subplots()
    lo, med, hi = synthetic_pi
    fig = prediction_interval_plot(synthetic_times, synthetic_obs, lo, med, hi, ax=ax0)
    assert isinstance(fig, Figure)
    # Should reuse the existing figure
    assert fig is fig0
    plt.close(fig)


@pytest.mark.unit
def test_prediction_interval_plot_custom_title(synthetic_times, synthetic_obs, synthetic_pi):
    """prediction_interval_plot should set a custom title on the axes."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot

    lo, med, hi = synthetic_pi
    custom_title = "My VPC Plot"
    fig = prediction_interval_plot(synthetic_times, synthetic_obs, lo, med, hi, title=custom_title)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == custom_title
    plt.close(fig)


@pytest.mark.unit
def test_prediction_interval_plot_length_mismatch():
    """prediction_interval_plot should raise ValueError on mismatched array lengths."""
    from openpkpd.plots.simulation import prediction_interval_plot

    times = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.0, 2.0])  # Wrong length
    lo = med = hi = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="same length"):
        prediction_interval_plot(times, obs, lo, med, hi)


@pytest.mark.unit
def test_prediction_interval_plot_with_nan(synthetic_times, synthetic_pi):
    """prediction_interval_plot should handle NaN observations without crashing."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot

    obs = np.array([1.0, np.nan, 2.0, np.nan, 1.5, 0.8, 0.4, 0.2])
    lo, med, hi = synthetic_pi
    fig = prediction_interval_plot(synthetic_times, obs, lo, med, hi)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_prediction_interval_plot_sorts_unsorted_inputs_before_drawing():
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot

    fig = prediction_interval_plot(
        np.array([4.0, 1.0, 2.0]),
        np.array([40.0, 10.0, 20.0]),
        np.array([30.0, 0.0, 10.0]),
        np.array([35.0, 5.0, 15.0]),
        np.array([45.0, 15.0, 25.0]),
    )

    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0, 4.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [5.0, 15.0, 35.0])

    scatter = next(coll for coll in ax.collections if isinstance(coll, PathCollection))
    np.testing.assert_allclose(
        scatter.get_offsets(),
        np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]),
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# vpc_plot tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vpc_plot_with_mock_vpc_result(mock_vpc_result):
    """vpc_plot should draw observed data, PI bands, and percentile traces."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import vpc_plot

    fig = vpc_plot(mock_vpc_result)
    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() == "Visual Predictive Check"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"

    scatter = next(coll for coll in ax.collections if isinstance(coll, PathCollection))
    np.testing.assert_allclose(
        scatter.get_offsets(),
        mock_vpc_result.observed_df[["TIME", "DV"]].to_numpy(dtype=float),
    )

    assert len(ax.lines) == 6
    expected_lines = [
        mock_vpc_result.sim_percentiles["p5_mid"].to_numpy(dtype=float),
        mock_vpc_result.sim_percentiles["p95_mid"].to_numpy(dtype=float),
        mock_vpc_result.sim_percentiles["p50_mid"].to_numpy(dtype=float),
        mock_vpc_result.obs_percentiles["p5"].to_numpy(dtype=float),
        mock_vpc_result.obs_percentiles["p50"].to_numpy(dtype=float),
        mock_vpc_result.obs_percentiles["p95"].to_numpy(dtype=float),
    ]
    for line, expected_y in zip(ax.lines, expected_lines, strict=True):
        np.testing.assert_allclose(line.get_xdata(), mock_vpc_result.obs_percentiles["bin_mid"])
        np.testing.assert_allclose(line.get_ydata(), expected_y)

    bands = [coll for coll in ax.collections if isinstance(coll, PolyCollection)]
    assert len(bands) == 3
    for band, lo_col, hi_col in [
        (bands[0], "p5_lo", "p5_hi"),
        (bands[1], "p95_lo", "p95_hi"),
        (bands[2], "p50_lo", "p50_hi"),
    ]:
        band_lo, band_hi = _fill_between_bounds(
            band,
            mock_vpc_result.sim_percentiles["bin_mid"].to_numpy(dtype=float),
        )
        np.testing.assert_allclose(band_lo, mock_vpc_result.sim_percentiles[lo_col])
        np.testing.assert_allclose(band_hi, mock_vpc_result.sim_percentiles[hi_col])

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Observed",
        "Sim 5th/95th PI",
        "Sim Median PI",
    ]
    plt.close(fig)


@pytest.mark.unit
def test_vpc_plot_with_custom_title(mock_vpc_result):
    """vpc_plot should display the custom title on the axes."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import vpc_plot

    custom_title = "Theophylline VPC"
    fig = vpc_plot(mock_vpc_result, title=custom_title)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == custom_title
    plt.close(fig)


@pytest.mark.unit
def test_vpc_plot_log_scale(mock_vpc_result):
    """vpc_plot with log_y=True should set log y-scale on the axes."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import vpc_plot

    fig = vpc_plot(mock_vpc_result, log_y=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    plt.close(fig)


@pytest.mark.unit
def test_vpc_plot_supports_custom_quantile_labels():
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import vpc_plot

    class CustomQuantileVPCResult:
        observed_df = pd.DataFrame(
            {
                "ID": [1, 1],
                "TIME": [1.0, 2.0],
                "DV": [2.0, 3.0],
                "REP": [0, 0],
                "MDV": [0, 0],
            }
        )
        obs_percentiles = pd.DataFrame(
            {
                "bin_mid": [1.0, 2.0],
                "p10": [1.0, 1.5],
                "p50": [2.0, 3.0],
                "p90": [4.0, 5.0],
                "n": [2, 2],
            }
        )
        sim_percentiles = pd.DataFrame(
            {
                "bin_mid": [1.0, 2.0],
                "p10_lo": [0.5, 1.0],
                "p10_mid": [1.0, 1.5],
                "p10_hi": [1.5, 2.0],
                "p50_lo": [1.5, 2.5],
                "p50_mid": [2.0, 3.0],
                "p50_hi": [2.5, 3.5],
                "p90_lo": [3.5, 4.5],
                "p90_mid": [4.0, 5.0],
                "p90_hi": [4.5, 5.5],
            }
        )
        n_replicates = 10
        quantiles = (0.1, 0.5, 0.9)

    fig = vpc_plot(CustomQuantileVPCResult())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 1.5])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [4.0, 5.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [2.0, 3.0])
    np.testing.assert_allclose(ax.lines[3].get_ydata(), [1.0, 1.5])
    np.testing.assert_allclose(ax.lines[4].get_ydata(), [2.0, 3.0])
    np.testing.assert_allclose(ax.lines[5].get_ydata(), [4.0, 5.0])
    plt.close(fig)


@pytest.mark.unit
def test_vpc_plot_empty_vpc_result():
    """vpc_plot with empty percentile DataFrames should not raise."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import vpc_plot

    class EmptyVPCResult:
        observed_df = pd.DataFrame(columns=["ID", "TIME", "DV"])
        obs_percentiles = pd.DataFrame(columns=["bin_mid", "p5", "p50", "p95"])
        sim_percentiles = pd.DataFrame(
            columns=[
                "bin_mid",
                "p5_lo",
                "p5_mid",
                "p5_hi",
                "p50_lo",
                "p50_mid",
                "p50_hi",
                "p95_lo",
                "p95_mid",
                "p95_hi",
            ]
        )
        n_replicates = 0

    fig = vpc_plot(EmptyVPCResult())
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# npde_plot tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_npde_plot_returns_figure(npde_df):
    """npde_plot should return a 4-panel Figure."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import npde_plot

    fig = npde_plot(npde_df)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


@pytest.mark.unit
def test_npde_plot_custom_title(npde_df):
    """npde_plot should set the custom title as the figure suptitle."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import npde_plot

    fig = npde_plot(npde_df, title="My NPDE Plot")
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_npde_plot_missing_column():
    """npde_plot should raise ValueError when NPDE column is absent."""
    from openpkpd.plots.simulation import npde_plot

    df_no_npde = pd.DataFrame(
        {
            "TIME": [1, 2, 3],
            "PRED": [1, 2, 3],
            "CWRES": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="NPDE column not found"):
        npde_plot(df_no_npde)


@pytest.mark.unit
def test_npde_plot_custom_figsize(npde_df):
    """npde_plot should respect custom figsize."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import npde_plot

    fig = npde_plot(npde_df, figsize=(8, 6))
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_npde_plot_all_nan_npde_returns_empty_panel() -> None:
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import npde_plot

    df = pd.DataFrame(
        {
            "TIME": [1.0, 2.0],
            "PRED": [1.0, 2.0],
            "NPDE": [np.nan, np.nan],
        }
    )

    fig = npde_plot(df)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


# ---------------------------------------------------------------------------
# simulation_panel tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simulation_panel_returns_figure(simulated_df):
    """simulation_panel should return a matplotlib Figure."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import simulation_panel

    fig = simulation_panel(simulated_df)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_simulation_panel_with_observed(simulated_df):
    """simulation_panel should sort replicate lines and overlay the observed profile."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import simulation_panel

    obs_df = simulated_df[simulated_df["REP"] == 0][["ID", "TIME", "DV"]].copy()
    fig = simulation_panel(simulated_df, observed_df=obs_df, n_subjects=1)
    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() == "Subject 1"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Concentration"
    assert len(ax.lines) == 11

    subj_sim = simulated_df[(simulated_df["ID"] == 1) & (simulated_df["REP"] >= 1)]
    for rep, line in zip(sorted(subj_sim["REP"].unique()), ax.lines[:-1], strict=True):
        rep_df = subj_sim[subj_sim["REP"] == rep].sort_values("TIME")
        np.testing.assert_allclose(line.get_xdata(), rep_df["TIME"])
        np.testing.assert_allclose(line.get_ydata(), rep_df["DV"])
        assert line.get_alpha() == pytest.approx(0.08)
        assert line.get_marker() == "None"

    observed_line = ax.lines[-1]
    observed_subj = obs_df[obs_df["ID"] == 1].sort_values("TIME")
    np.testing.assert_allclose(observed_line.get_xdata(), observed_subj["TIME"])
    np.testing.assert_allclose(observed_line.get_ydata(), observed_subj["DV"])
    assert observed_line.get_marker() == "o"
    assert observed_line.get_label() == "Observed"
    plt.close(fig)


@pytest.mark.unit
def test_simulation_panel_n_subjects_limit(simulated_df):
    """simulation_panel with n_subjects=1 should show only one subject's panel."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import simulation_panel

    fig = simulation_panel(simulated_df, n_subjects=1)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_simulation_panel_custom_title(simulated_df):
    """simulation_panel should apply the custom title."""
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import simulation_panel

    fig = simulation_panel(simulated_df, title="Custom Panel Title")
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.unit
def test_simulation_panel_handles_observed_only_rep0_input() -> None:
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import simulation_panel

    observed_only = pd.DataFrame(
        {
            "ID": [2, 2, 1, 1],
            "TIME": [2.0, 1.0, 2.0, 1.0],
            "DV": [1.0, 2.0, 3.0, 4.0],
            "REP": [0, 0, 0, 0],
        }
    )

    fig = simulation_panel(observed_only)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert titles[:2] == ["Subject 1", "Subject 2"]
    plt.close(fig)
