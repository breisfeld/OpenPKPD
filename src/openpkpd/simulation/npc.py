"""
Numerical Predictive Check (NPC).

Compares observed data to the empirical distribution of simulated predictions
to assess model performance. The NPC asks: what fraction of observations fall
below/above the simulated prediction interval?

Reference:
    Holford NHG. (2005). The Visual Predictive Check — superiority to standard
    diagnostic (Rorschach) plots. PAGE 14 (Abstract 738).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openpkpd.simulation.engine import SimulationResult


@dataclass
class NPCResult:
    """
    Result of a Numerical Predictive Check.

    Attributes:
        pi_lower:         Nominal lower prediction interval fraction (e.g. 0.05).
        pi_upper:         Nominal upper prediction interval fraction (e.g. 0.95).
        obs_below_lower:  Observed fraction of DVs below the lower PI bound.
        obs_above_upper:  Observed fraction of DVs above the upper PI bound.
        obs_within:       Observed fraction of DVs within the PI.
        expected_within:  Expected fraction = pi_upper - pi_lower.
        n_observations:   Number of observed data points evaluated.
        binned:           Optional per-time-bin breakdown DataFrame.
    """

    pi_lower: float
    pi_upper: float
    obs_below_lower: float
    obs_above_upper: float
    obs_within: float
    expected_within: float
    n_observations: int
    binned: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [
            "Numerical Predictive Check",
            f"  Prediction interval:  {self.pi_lower:.0%} – {self.pi_upper:.0%}",
            f"  Expected within PI:   {self.expected_within:.1%}",
            f"  Observed within PI:   {self.obs_within:.1%}",
            f"  Observed below lower: {self.obs_below_lower:.1%}",
            f"  Observed above upper: {self.obs_above_upper:.1%}",
            f"  n observations:       {self.n_observations}",
        ]
        if self.binned is not None:
            lines.append(f"  n time bins:          {len(self.binned)}")
        return "\n".join(lines)


class NPCEngine:
    """
    Numerical Predictive Check engine.

    For each observed data point, counts what fraction of simulated
    replicates are below the observed value. The resulting empirical p-values
    are compared to the nominal PI boundaries.

    Args:
        simulation_result: SimulationResult with REP=0 as observed data and
                           REP=1..n_replicates as simulated datasets.
    """

    def __init__(self, simulation_result: SimulationResult) -> None:
        self.sim_result = simulation_result

    def compute(
        self,
        pi_lower: float = 0.05,
        pi_upper: float = 0.95,
        n_bins: int | None = None,
        stratify_by: str | None = None,
        id_col: str = "ID",
        time_col: str = "TIME",
        dv_col: str = "DV",
        rep_col: str = "REP",
    ) -> NPCResult:
        """
        Compute the NPC.

        Algorithm:
            1. Split the simulation_result DataFrame into observed (REP=0) and
               simulated (REP>=1) portions.
            2. For each observed DV_obs at (subject_i, time_j):
               - Collect all simulated DV values at the same (subject, time).
               - Compute empirical_p = fraction of sim DVs < DV_obs.
            3. Observed % below lower PI = fraction of obs where empirical_p < pi_lower.
            4. Observed % above upper PI = fraction of obs where empirical_p > pi_upper.

        Args:
            pi_lower:     Nominal lower PI quantile (default 0.05).
            pi_upper:     Nominal upper PI quantile (default 0.95).
            n_bins:       Optional number of time bins for per-bin breakdown.
            stratify_by:  Optional column name to stratify the NPC.
            id_col:       Column name for subject identifier.
            time_col:     Column name for time.
            dv_col:       Column name for observed/simulated DV.
            rep_col:      Column name for replicate index (0 = observed).

        Returns:
            NPCResult with overall and optionally per-bin fractions.
        """
        df = self.sim_result.simulated_df

        if rep_col not in df.columns:
            raise ValueError(f"Column '{rep_col}' not found in simulation result.")

        obs_df = df[df[rep_col] == 0].copy()
        sim_df = df[df[rep_col] > 0].copy()

        if len(obs_df) == 0:
            raise ValueError("No observed data (REP=0) found in simulation result.")
        if len(sim_df) == 0:
            raise ValueError("No simulated data (REP>0) found in simulation result.")

        # Build per-time-point simulation lookup
        # Group simulated data by (ID, TIME) → list of DV values
        sim_lookup: dict[tuple, np.ndarray] = {}
        for (subj, time), grp in sim_df.groupby([id_col, time_col]):
            sim_lookup[(subj, float(time))] = grp[dv_col].dropna().values.astype(float)

        empirical_ps: list[float] = []
        empirical_times: list[float] = []
        n_no_sim = 0

        for _, row in obs_df.iterrows():
            dv_obs = float(row[dv_col])
            if not np.isfinite(dv_obs):
                continue

            key = (row[id_col], float(row[time_col]))
            sim_vals = sim_lookup.get(key)

            if sim_vals is None or len(sim_vals) == 0:
                # Try approximate time matching (within tolerance)
                t_obs = float(row[time_col])
                for (s, t), vals in sim_lookup.items():
                    if s == row[id_col] and abs(t - t_obs) < 1e-6:
                        sim_vals = vals
                        break

            if sim_vals is None or len(sim_vals) == 0:
                n_no_sim += 1
                continue

            emp_p = float(np.mean(sim_vals < dv_obs))
            empirical_ps.append(emp_p)
            empirical_times.append(float(row[time_col]))

        empirical_ps_arr = np.array(empirical_ps, dtype=float)
        empirical_times_arr = np.array(empirical_times, dtype=float)
        n_obs = len(empirical_ps_arr)

        if n_obs == 0:
            return NPCResult(
                pi_lower=pi_lower,
                pi_upper=pi_upper,
                obs_below_lower=float("nan"),
                obs_above_upper=float("nan"),
                obs_within=float("nan"),
                expected_within=pi_upper - pi_lower,
                n_observations=0,
            )

        obs_below = float(np.mean(empirical_ps_arr < pi_lower))
        obs_above = float(np.mean(empirical_ps_arr > pi_upper))
        obs_within = float(1.0 - obs_below - obs_above)
        expected_within = float(pi_upper - pi_lower)

        # Optional time-binned breakdown
        binned_df = None
        if n_bins is not None and n_bins > 0 and len(empirical_times_arr) > 0:
            bin_edges = np.linspace(
                float(empirical_times_arr.min()),
                float(empirical_times_arr.max()) + 1e-10,
                n_bins + 1,
            )
            bin_records = []
            for b in range(n_bins):
                t_lo, t_hi = bin_edges[b], bin_edges[b + 1]
                in_bin = (empirical_times_arr >= t_lo) & (empirical_times_arr < t_hi)
                if not np.any(in_bin):
                    continue
                bin_eps = empirical_ps_arr[in_bin]
                if len(bin_eps) == 0:
                    continue
                bin_records.append(
                    {
                        "t_lo": t_lo,
                        "t_hi": t_hi,
                        "t_mid": (t_lo + t_hi) / 2.0,
                        "n_obs": len(bin_eps),
                        "obs_below_lower": float(np.mean(bin_eps < pi_lower)),
                        "obs_above_upper": float(np.mean(bin_eps > pi_upper)),
                        "obs_within": float(np.mean((bin_eps >= pi_lower) & (bin_eps <= pi_upper))),
                    }
                )
            if bin_records:
                binned_df = pd.DataFrame(bin_records)

        return NPCResult(
            pi_lower=pi_lower,
            pi_upper=pi_upper,
            obs_below_lower=obs_below,
            obs_above_upper=obs_above,
            obs_within=obs_within,
            expected_within=expected_within,
            n_observations=n_obs,
            binned=binned_df,
        )
