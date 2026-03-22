"""
Dosing event engine for NONMEM datasets.

Converts rows of a NONMEMDataset into structured DoseEvent objects
and observation time lists for use by PK subroutines.

Supported EVID codes:
  0 - Observation
  1 - Dose event (AMT, RATE, CMT, SS, II, ADDL)
  2 - Other type (covariate change, model discontinuity)
  3 - Reset all compartments to zero
  4 - Reset then dose (SS initialization)

ADDL/II: Additional doses of same type at intervals of II
SS:      Steady-state flag; model runs to steady state before t=0
RATE>0:  Zero-order infusion at given rate for duration AMT/RATE
RATE=-1: Duration specified in DUR column
RATE=-2: Rate specified in $PK (bioavailability-equivalent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from openpkpd.data.columns import ADDL, AMT, CMT, DV, EVID, ID, II, MDV, RATE, SS, TIME
from openpkpd.utils.constants import EVID_DOSE, EVID_OBS, EVID_OTHER, EVID_RESET, EVID_RESET_DOSE

OCC = "OCC"  # Occasion column for IOV


@dataclass
class DoseEvent:
    """A single dosing event (bolus or infusion)."""

    time: float
    amount: float
    rate: float = 0.0  # 0 = bolus; >0 = infusion rate; -1 = duration-based
    duration: float = 0.0  # For RATE=-1: explicit infusion duration
    compartment: int = 1
    ss: bool = False  # Steady-state flag
    ii: float = 0.0  # Inter-dose interval (for SS or ADDL)
    reset: bool = False  # EVID=3/4: reset compartments to zero first

    @property
    def is_bolus(self) -> bool:
        return self.rate == 0.0

    @property
    def is_infusion(self) -> bool:
        return self.rate > 0

    @property
    def infusion_end_time(self) -> float:
        if self.rate > 0:
            return self.time + self.amount / self.rate
        if self.rate == -1.0 and self.duration > 0:
            return self.time + self.duration
        return self.time

    def __repr__(self) -> str:
        mode = "infusion" if self.is_infusion else "bolus"
        return (
            f"DoseEvent(t={self.time}, amt={self.amount}, {mode}, "
            f"cmt={self.compartment}, ss={self.ss})"
        )


@dataclass
class SubjectEvents:
    """
    All dosing events and observation times for one subject.
    """

    subject_id: int
    dose_events: list[DoseEvent] = field(default_factory=list)
    obs_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    obs_dv: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    obs_cmt: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    obs_mdv: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    obs_covariates: list[dict[str, Any]] | None = None
    covariate_df: pd.DataFrame | None = None
    occasion_indices: np.ndarray | None = None  # B1: per-observation occasion index

    def observation_mask(self) -> np.ndarray:
        """Boolean mask for non-missing observations."""
        return self.obs_mdv == 0

    @staticmethod
    def _normalize_covariate_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    def covariate_at(self, t: float) -> dict[str, Any]:
        """
        Return covariate values at time ``t`` using last-observation-carried-forward (LOCF).

        If ``t`` is before the first covariate measurement, the first row is returned.

        Args:
            t: Query time.

        Returns:
            Dict mapping covariate name → value.  Empty dict if no ``covariate_df``.
        """
        if self.covariate_df is None or len(self.covariate_df) == 0:
            return {}
        df = self.covariate_df
        # Binary search (O(log n)) instead of boolean mask (O(n))
        times = df["TIME"].values
        idx = int(np.searchsorted(times, t + 1e-12, side="right")) - 1
        row = df.iloc[max(0, idx)]
        cov_cols = [c for c in df.columns if c != "TIME"]
        return {c: self._normalize_covariate_value(row[c]) for c in cov_cols if not pd.isna(row[c])}

    def observation_covariates_at(self, index: int) -> dict[str, Any]:
        """Return covariates aligned to a specific observation row."""
        if self.obs_covariates is None or index >= len(self.obs_covariates):
            if index >= len(self.obs_times):
                return {}
            return self.covariate_at(float(self.obs_times[index]))
        return dict(self.obs_covariates[index])

    def covariate_change_times(self) -> list[float]:
        """
        Return a sorted list of times at which any covariate is first measured or changes.

        Used by ODE solvers to add breakpoints at covariate change events so that
        PK parameters can be updated piecewise-constantly (LOCF).

        Returns:
            Sorted list of unique times in ``covariate_df["TIME"]``.
            Empty list if no ``covariate_df``.
        """
        if self.covariate_df is None or len(self.covariate_df) == 0:
            return []
        return sorted(self.covariate_df["TIME"].unique().tolist())


class EventProcessor:
    """
    Converts a NONMEMDataset into per-subject SubjectEvents.

    Handles:
      - EVID 0/1/2/3/4 classification
      - ADDL/II expansion (repeating doses)
      - SS (steady-state) flagging
      - Infusion events (RATE > 0)
      - Reset events (EVID 3/4)
    """

    def __init__(self, covariate_columns: list[str] | None = None) -> None:
        """
        Args:
            covariate_columns: List of column names to include in covariate_df.
        """
        self.covariate_columns = covariate_columns or []

    def process(self, df: pd.DataFrame) -> dict[int, SubjectEvents]:
        """
        Process the full dataset and return per-subject events.

        Returns:
            Dict mapping subject_id → SubjectEvents.
        """
        result: dict[int, SubjectEvents] = {}
        for subj_id, subj_df in df.groupby(ID, sort=True):
            result[int(subj_id)] = self._process_subject(int(subj_id), subj_df)
        return result

    def _process_subject(self, subject_id: int, df: pd.DataFrame) -> SubjectEvents:
        df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
        n = len(df)
        available_cov = [c for c in self.covariate_columns if c in df.columns]

        # ── Convert hot columns to numpy arrays once ───────────────────────────
        time_arr = df[TIME].values.astype(float)
        evid_arr = (
            df[EVID].fillna(0).values.astype(int) if EVID in df.columns else np.zeros(n, dtype=int)
        )

        # ── Row-type masks ──────────────────────────────────────────────────────
        is_obs = evid_arr == EVID_OBS
        is_other = evid_arr == EVID_OTHER
        is_obs_row = is_obs | is_other  # rows that become observations
        is_dose = np.isin(evid_arr, [EVID_DOSE, EVID_RESET_DOSE])
        is_reset = evid_arr == EVID_RESET

        # ── Observations: bulk extraction via boolean indexing ─────────────────
        obs_time_arr = time_arr[is_obs_row]

        dv_arr = (
            pd.to_numeric(df[DV], errors="coerce").values.astype(float, copy=False)
            if DV in df.columns
            else np.full(n, np.nan)
        )
        obs_dv_arr = dv_arr[is_obs_row].copy()
        obs_dv_arr[evid_arr[is_obs_row] == EVID_OTHER] = np.nan  # EVID=2 → NaN

        cmt_arr = (
            df[CMT].fillna(1).values.astype(int) if CMT in df.columns else np.ones(n, dtype=int)
        )
        obs_cmt_arr = cmt_arr[is_obs_row]

        mdv_arr = (
            df[MDV].fillna(0).values.astype(int) if MDV in df.columns else np.zeros(n, dtype=int)
        )
        obs_mdv_arr = mdv_arr[is_obs_row].copy()
        obs_mdv_arr[evid_arr[is_obs_row] == EVID_OTHER] = 1  # EVID=2 → MDV=1

        has_occ = OCC in df.columns
        if has_occ:
            occ_arr = pd.to_numeric(df[OCC], errors="coerce").fillna(0).values.astype(int)
            obs_occ_arr = occ_arr[is_obs_row]
        else:
            obs_occ_arr = np.zeros(is_obs_row.sum(), dtype=int)

        # Covariate dicts: only iterate the (usually small) obs subset
        obs_covariates: list[dict[str, Any]] | None = None
        if available_cov:
            obs_cov_df = df.iloc[np.where(is_obs_row)[0]]
            obs_covariates = [
                {
                    c: SubjectEvents._normalize_covariate_value(row[c])
                    for c in available_cov
                    if not pd.isna(row[c])
                }
                for _, row in obs_cov_df.iterrows()
            ]

        # ── Dose events: iterate only dose/reset rows ──────────────────────────
        dose_events: list[DoseEvent] = []
        if is_dose.any() or is_reset.any():
            for _, row in df[is_dose | is_reset].iterrows():
                evid = int(row[EVID]) if EVID in df.columns else 0
                time = float(row[TIME])

                if evid in (EVID_DOSE, EVID_RESET_DOSE):
                    amt = float(row.get(AMT, 0.0))
                    rate = float(row.get(RATE, 0.0))
                    cmt = int(row.get(CMT, 1))
                    ss_flag = bool(int(row.get(SS, 0)))
                    ii_val = float(row.get(II, 0.0))
                    addl = int(row.get(ADDL, 0))
                    reset = evid == EVID_RESET_DOSE

                    # Compute infusion duration and normalize duration-based infusions
                    duration = 0.0
                    if rate > 0 and amt > 0:
                        duration = amt / rate
                    elif rate == -1.0:
                        duration = float(row.get("DUR", 0.0))
                        if duration > 0 and amt > 0:
                            rate = amt / duration

                    base_event = DoseEvent(
                        time=time,
                        amount=amt,
                        rate=rate,
                        duration=duration,
                        compartment=cmt,
                        ss=ss_flag,
                        ii=ii_val,
                        reset=reset,
                    )
                    dose_events.append(base_event)

                    # Expand ADDL doses
                    if addl > 0 and ii_val > 0:
                        for k in range(1, addl + 1):
                            dose_events.append(
                                DoseEvent(
                                    time=time + k * ii_val,
                                    amount=amt,
                                    rate=rate,
                                    duration=duration,
                                    compartment=cmt,
                                    ss=False,
                                    ii=ii_val,
                                    reset=False,
                                )
                            )

                elif evid == EVID_RESET:
                    dose_events.append(
                        DoseEvent(
                            time=time,
                            amount=0.0,
                            compartment=1,
                            reset=True,
                        )
                    )

            dose_events.sort(key=lambda e: e.time)

        # Build covariate dataframe (time-varying covariates)
        cov_df = None
        if self.covariate_columns and available_cov:
            cov_df = df[[TIME] + available_cov].copy()

        return SubjectEvents(
            subject_id=subject_id,
            dose_events=dose_events,
            obs_times=obs_time_arr,
            obs_dv=obs_dv_arr,
            obs_cmt=obs_cmt_arr,
            obs_mdv=obs_mdv_arr,
            obs_covariates=obs_covariates,
            covariate_df=cov_df,
            occasion_indices=obs_occ_arr if has_occ else None,
        )
