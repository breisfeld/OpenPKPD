"""
Non-Compartmental Analysis (NCA).

Estimates PK parameters directly from concentration-time data without
assuming a compartmental structural model. Uses standard trapezoidal
methods for AUC computation and log-linear regression for terminal phase.

References:
    Gabrielsson, J. & Weiner, D. (2006). Pharmacokinetic and Pharmacodynamic
    Data Analysis. 4th edition.
    FDA Guidance for Industry: Bioavailability and Bioequivalence Studies (2003).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class NCAParameters:
    """
    NCA parameters for one subject/profile.

    Attributes:
        subject_id:      Subject or profile identifier.
        dose:            Administered dose (in dose units).
        route:           Route of administration: 'IV', 'oral', or 'infusion'.
        cmax:            Maximum observed concentration.
        c0:              Concentration at time zero for IV bolus profiles
                          (observed if sampled, otherwise back-extrapolated
                          from the first two positive samples when possible).
        tmax:            Time of maximum observed concentration.
        auc_last:        AUC from time 0 to last quantifiable timepoint (AUC0-t).
        auc_inf:         AUC extrapolated to infinity (AUC0-inf).
        aumc_last:       AUMC from time 0 to last quantifiable timepoint.
        aumc_inf:        AUMC extrapolated to infinity.
        lambda_z:        Terminal elimination rate constant (hr^-1 or time_unit^-1).
        t_half:          Terminal half-life = ln(2) / lambda_z.
        cl_f:            Apparent clearance = Dose / AUC_inf.
        vz_f:            Apparent volume of distribution = CL/F / lambda_z.
        mrt:             Mean residence time. For IV infusion profiles with a
                          known infusion duration, OpenPKPD reports
                          AUMC_inf / AUC_inf - Tinf / 2.
        r_squared:       Coefficient of determination for lambda_z regression.
        n_points_lambda: Number of data points used in terminal regression.
        auc_extrap_pct:  Percentage of AUC extrapolated beyond last timepoint.
    """

    subject_id: int | str
    dose: float
    route: str = "IV"

    # Primary endpoints
    cmax: float = float("nan")
    c0: float = float("nan")
    tmax: float = float("nan")
    auc_last: float = float("nan")
    auc_inf: float = float("nan")
    aumc_last: float = float("nan")
    aumc_inf: float = float("nan")
    lambda_z: float = float("nan")
    t_half: float = float("nan")

    # Derived endpoints
    cl_f: float = float("nan")
    vz_f: float = float("nan")
    mrt: float = float("nan")

    # Regression quality
    r_squared: float = float("nan")
    n_points_lambda: int = 0

    # Flags
    auc_extrap_pct: float = float("nan")

    # Multiple-dose / steady-state
    auc_tau: float = float("nan")  # AUC over one dosing interval at steady state
    c_min: float = float("nan")  # Trough concentration
    c_avg: float = float("nan")  # Average concentration = auc_tau / tau
    fluctuation: float = float("nan")  # (Cmax - Cmin) / Cavg * 100
    swing: float = float("nan")  # (Cmax - Cmin) / Cmin
    r_ac: float = float("nan")  # Accumulation ratio = auc_tau_ss / auc_tau_sd

    # Partial AUC
    auc_partial: float = float("nan")
    auc_partial_t1: float = float("nan")
    auc_partial_t2: float = float("nan")

    # Dose-normalized
    norm_cmax: float = float("nan")  # Cmax / dose
    norm_auc_last: float = float("nan")  # auc_last / dose
    norm_auc_inf: float = float("nan")  # auc_inf / dose

    # Metabolite/parent ratios
    metabolite_parent_cmax_ratio: float = float("nan")
    metabolite_parent_auc_ratio: float = float("nan")

    def to_dict(self) -> dict[str, float | int | str]:
        """Return all NCA parameters as a flat dictionary."""
        return {
            "subject_id": self.subject_id,
            "dose": self.dose,
            "route": self.route,
            "cmax": self.cmax,
            "c0": self.c0,
            "tmax": self.tmax,
            "auc_last": self.auc_last,
            "auc_inf": self.auc_inf,
            "aumc_last": self.aumc_last,
            "aumc_inf": self.aumc_inf,
            "lambda_z": self.lambda_z,
            "t_half": self.t_half,
            "cl_f": self.cl_f,
            "vz_f": self.vz_f,
            "mrt": self.mrt,
            "r_squared": self.r_squared,
            "n_points_lambda": self.n_points_lambda,
            "auc_extrap_pct": self.auc_extrap_pct,
            # Multiple-dose / steady-state
            "auc_tau": self.auc_tau,
            "c_min": self.c_min,
            "c_avg": self.c_avg,
            "fluctuation": self.fluctuation,
            "swing": self.swing,
            "r_ac": self.r_ac,
            # Partial AUC
            "auc_partial": self.auc_partial,
            "auc_partial_t1": self.auc_partial_t1,
            "auc_partial_t2": self.auc_partial_t2,
            # Dose-normalized
            "norm_cmax": self.norm_cmax,
            "norm_auc_last": self.norm_auc_last,
            "norm_auc_inf": self.norm_auc_inf,
            # Metabolite/parent ratios
            "metabolite_parent_cmax_ratio": self.metabolite_parent_cmax_ratio,
            "metabolite_parent_auc_ratio": self.metabolite_parent_auc_ratio,
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"NCA Parameters — Subject {self.subject_id}",
            f"  Route:       {self.route}",
            f"  Dose:        {self.dose}",
            f"  Cmax:        {self.cmax:.4g}",
            f"  C0:          {self.c0:.4g}",
            f"  Tmax:        {self.tmax:.4g}",
            f"  AUC_last:    {self.auc_last:.4g}",
            f"  AUC_inf:     {self.auc_inf:.4g}",
            f"  t_half:      {self.t_half:.4g}",
            f"  lambda_z:    {self.lambda_z:.4g}  (R²={self.r_squared:.4f}, n={self.n_points_lambda})",
            f"  CL/F:        {self.cl_f:.4g}",
            f"  Vz/F:        {self.vz_f:.4g}",
            f"  MRT:         {self.mrt:.4g}",
            f"  AUC_extrap%: {self.auc_extrap_pct:.2f}%",
        ]
        return "\n".join(lines)


class NCAEngine:
    """
    Non-compartmental analysis engine.

    Computes NCA parameters for one or more concentration-time profiles
    without assuming a compartmental model.

    Args:
        auc_method:        Trapezoidal method for AUC computation.
                           'linear-trapezoidal': standard linear rule.
                           'linear-log': log rule for decreasing segments,
                               linear for increasing.
                           'linear-up-log-down': linear for increasing,
                               log for decreasing segments (alias of
                               'linear-log').
        lambda_z_method:   'auto' selects the terminal regression window
                           that maximizes adjusted R². 'manual' requires
                           the caller to supply terminal points explicitly.
        min_points_lambda: Minimum number of points required for terminal
                           regression. Must be >= 3.
        exclude_cmax:      If True, exclude the Cmax observation from the
                           terminal regression (standard practice).
    """

    def __init__(
        self,
        auc_method: Literal[
            "linear-trapezoidal", "linear-log", "linear-up-log-down"
        ] = "linear-log",
        lambda_z_method: Literal["auto", "manual"] = "auto",
        min_points_lambda: int = 3,
        exclude_cmax: bool = True,
    ) -> None:
        if min_points_lambda < 3:
            raise ValueError("min_points_lambda must be >= 3 for reliable regression.")
        self.auc_method = auc_method
        self.lambda_z_method = lambda_z_method
        self.min_points_lambda = min_points_lambda
        self.exclude_cmax = exclude_cmax

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_subject(
        self,
        times: np.ndarray,
        conc: np.ndarray,
        dose: float,
        subject_id: int | str = 0,
        route: str = "oral",
        t_last: float | None = None,
        infusion_duration: float | None = None,
    ) -> NCAParameters:
        """
        Compute NCA parameters for one concentration-time profile.

        Args:
            times:      Sampling times (including time 0 if available).
                        Must be sorted in ascending order.
            conc:       Observed concentrations (NaN for missing values).
            dose:       Administered dose.
            subject_id: Subject identifier (used for reporting only).
            route:      'IV', 'oral', or 'infusion'.
            t_last:     Override for the last quantifiable time point.
                        If None, automatically determined as the last
                        time point with a positive, finite concentration.
            infusion_duration:
                        Optional infusion duration (Tinf). When provided for
                        route='infusion', MRT is adjusted by subtracting
                        Tinf / 2, matching standard IV infusion reporting.

        Returns:
            NCAParameters with computed endpoints. Parameters that cannot
            be estimated (e.g. AUC_inf when terminal phase is not
            identifiable) are set to NaN.
        """
        times = np.asarray(times, dtype=float)
        conc = np.asarray(conc, dtype=float)

        if times.shape != conc.shape:
            raise ValueError("times and conc must have the same length.")
        if times.ndim != 1:
            raise ValueError("times and conc must be 1-D arrays.")

        # Remove records where time or conc is NaN
        valid_mask = np.isfinite(times) & np.isfinite(conc)
        if not np.all(valid_mask):
            times = times[valid_mask]
            conc = conc[valid_mask]

        if len(times) == 0:
            return NCAParameters(subject_id=subject_id, dose=dose, route=route)

        route_key = route.strip().lower()
        if route_key == "infusion" and infusion_duration is not None:
            infusion_duration = float(infusion_duration)
            if not np.isfinite(infusion_duration) or infusion_duration <= 0.0:
                raise ValueError(
                    "infusion_duration must be a positive finite value for infusion routes."
                )

        # Sort by time (defensive)
        if len(times) > 1 and np.any(np.diff(times) < 0.0):
            sort_idx = np.argsort(times, kind="stable")
            times = times[sort_idx]
            conc = conc[sort_idx]

        params = NCAParameters(subject_id=subject_id, dose=dose, route=route)

        # -- Cmax and Tmax ------------------------------------------------
        cmax_idx = int(np.argmax(conc))
        params.cmax = float(conc[cmax_idx])
        params.tmax = float(times[cmax_idx])

        # For IV bolus profiles, optionally reconstruct a missing time-zero
        # concentration and use it for AUC/AUMC integration. Observed Cmax/Tmax
        # and terminal regression stay anchored to the observed profile.
        auc_times = times
        auc_conc = conc
        if route_key == "iv":
            auc_times, auc_conc, params.c0 = self._augment_iv_bolus_profile(times, conc)

        # -- Determine last quantifiable timepoint (t_last) ---------------
        pos_indices = np.flatnonzero(auc_conc > 0.0)
        if len(pos_indices) == 0:
            # No positive concentrations — cannot compute AUC
            return params

        if t_last is not None:
            # Use caller-supplied t_last; find index of last obs <= t_last
            cutoff = int(np.searchsorted(auc_times, t_last, side="right"))
            eligible_idx = pos_indices[pos_indices < cutoff]
            if len(eligible_idx) == 0:
                return params
            last_idx = int(eligible_idx[-1])
            t_last_val = float(auc_times[last_idx])
            c_last = float(auc_conc[last_idx])
            t_obs = auc_times[: last_idx + 1]
            c_obs = auc_conc[: last_idx + 1]

            # If t_last falls between two observations, interpolate the
            # boundary concentration so AUClast/AUMClast honor the caller's
            # requested cutoff instead of truncating at the previous sample.
            if t_last > t_last_val:
                boundary_method: Literal["linear", "log"] = "linear"
                idx_after = int(np.searchsorted(auc_times, t_last, side="left"))
                if (
                    self.auc_method != "linear-trapezoidal"
                    and 0 < idx_after < len(auc_times)
                    and auc_conc[idx_after] < auc_conc[idx_after - 1]
                ):
                    boundary_method = "log"

                c_interp = self._interpolate_conc(
                    auc_times,
                    auc_conc,
                    float(t_last),
                    method=boundary_method,
                )
                if np.isfinite(c_interp):
                    t_last_val = float(t_last)
                    c_last = c_interp
                    t_obs = np.concatenate([t_obs, [t_last_val]])
                    c_obs = np.concatenate([c_obs, [c_last]])
        else:
            # Last positive observation
            last_idx = int(pos_indices[-1])
            t_last_val = float(auc_times[last_idx])
            c_last = float(auc_conc[last_idx])

            # Slice to [0, t_last]
            t_obs = auc_times[: last_idx + 1]
            c_obs = auc_conc[: last_idx + 1]

        # -- AUC_last -----------------------------------------------------
        params.auc_last, params.aumc_last = self._compute_auc_pair(t_obs, c_obs)

        # -- Terminal lambda_z --------------------------------------------
        # Use the AUC/AUMC profile for terminal regression as well, so IV
        # bolus profiles with a reconstructed C0 exclude the back-extrapolated
        # peak rather than the first observed sample.
        term_cmax_idx = int(np.argmax(auc_conc))
        term_start = term_cmax_idx + (1 if self.exclude_cmax else 0)
        t_term_all = auc_times[term_start:]
        c_term_all = auc_conc[term_start:]

        # Restrict to positive concentrations
        pos_term = c_term_all > 0
        if np.all(pos_term):
            t_term = t_term_all
            c_term = c_term_all
        else:
            t_term = t_term_all[pos_term]
            c_term = c_term_all[pos_term]

        lambda_z = float("nan")
        r_sq = float("nan")
        n_pts = 0

        if len(t_term) >= self.min_points_lambda:
            lambda_z, r_sq, n_pts = self._compute_lambda_z(t_term, c_term)

        params.lambda_z = lambda_z
        params.r_squared = r_sq
        params.n_points_lambda = n_pts

        # -- Half-life ----------------------------------------------------
        if np.isfinite(lambda_z) and lambda_z > 0:
            params.t_half = float(np.log(2) / lambda_z)

        # -- AUC_inf and AUMC_inf ----------------------------------------
        if np.isfinite(lambda_z) and lambda_z > 0 and c_last > 0:
            auc_extrap = self._extrapolate_auc(params.auc_last, c_last, lambda_z)
            params.auc_inf = auc_extrap

            # AUMC extrapolation: AUMC0-inf = AUMC0-t + C_last*t_last/lz + C_last/lz^2
            aumc_extrap = params.aumc_last + c_last * t_last_val / lambda_z + c_last / (lambda_z**2)
            params.aumc_inf = aumc_extrap

            # Percent extrapolated
            if params.auc_inf > 0:
                params.auc_extrap_pct = (params.auc_inf - params.auc_last) / params.auc_inf * 100.0

        # -- Derived: CL/F, Vz/F, MRT ------------------------------------
        if np.isfinite(params.auc_inf) and params.auc_inf > 0:
            params.cl_f = dose / params.auc_inf

            if np.isfinite(lambda_z) and lambda_z > 0:
                params.vz_f = params.cl_f / lambda_z

        if np.isfinite(params.aumc_inf) and np.isfinite(params.auc_inf) and params.auc_inf > 0:
            params.mrt = params.aumc_inf / params.auc_inf
            if route_key == "infusion" and infusion_duration is not None:
                params.mrt -= infusion_duration / 2.0

        return params

    def compute_dataset(
        self,
        df: pd.DataFrame,
        id_col: str = "ID",
        time_col: str = "TIME",
        conc_col: str = "DV",
        dose_col: str = "AMT",
        dose_row_col: str = "EVID",
        route: str = "oral",
        infusion_duration: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute NCA parameters for all subjects in a DataFrame.

        The DataFrame must follow NONMEM conventions:
          - Dosing rows are identified by EVID == 1 (or dose_row_col != 0).
          - Observation rows are all other rows.
          - Dose amount is read from the AMT column on the dosing row.

        Args:
            df:           Input NONMEM-formatted DataFrame.
            id_col:       Column name for subject identifier.
            time_col:     Column name for time.
            conc_col:     Column name for observed concentration.
            dose_col:     Column name for dose amount.
            dose_row_col: Column used to identify dose rows (EVID).
                          Rows with value == 1 are treated as dosing events.
            route:        Route of administration applied to all subjects.
            infusion_duration:
                          Optional infusion duration (Tinf) applied to all
                          subjects when route='infusion'.

        Returns:
            DataFrame with one row per subject containing all NCA parameters.
        """
        records: list[dict] = []
        subject_ids = pd.unique(df[id_col])
        if len(subject_ids) == 0:
            return pd.DataFrame(records)

        group_indices = df.groupby(id_col, sort=False).indices
        dose_values = df[dose_col].to_numpy(dtype=float, copy=False)
        time_values = df[time_col].to_numpy(dtype=float, copy=False)
        conc_values = df[conc_col].to_numpy(dtype=float, copy=False)
        dose_row_values = df[dose_row_col].to_numpy(copy=False)

        for sid in subject_ids:
            positions = np.asarray(group_indices[sid], dtype=int)
            subj_dose_row_values = dose_row_values[positions]
            dose_mask = subj_dose_row_values == 1
            dose = float(dose_values[positions[dose_mask]].sum()) if np.any(dose_mask) else 1.0

            obs_positions = positions[~dose_mask]
            times = time_values[obs_positions]
            conc = conc_values[obs_positions]

            if len(times) < 2:
                params = NCAParameters(subject_id=sid, dose=dose, route=route)
            else:
                params = self.compute_subject(
                    times=times,
                    conc=conc,
                    dose=dose,
                    subject_id=sid,
                    route=route,
                    infusion_duration=infusion_duration,
                )

            records.append(params.to_dict())

        return pd.DataFrame.from_records(records)

    def compute_partial_auc(
        self,
        times: np.ndarray,
        conc: np.ndarray,
        t1: float,
        t2: float,
        method: Literal["linear", "log"] = "log",
    ) -> float:
        """
        Compute partial AUC from t1 to t2 by interpolating at t1 and t2.

        For each boundary (t1, t2):
        - If the boundary falls between two observations, interpolate
          concentration using log interpolation (if method='log' and both
          concentrations > 0) or linear interpolation.
        - Then compute AUC over the subinterval using self._compute_auc.

        Returns:
            Partial AUC value. NaN if t1 >= t2 or insufficient data.
        """
        times = np.asarray(times, dtype=float)
        conc = np.asarray(conc, dtype=float)

        if t1 >= t2:
            return float("nan")

        c_at_t1 = self._interpolate_conc(times, conc, t1, method=method)
        c_at_t2 = self._interpolate_conc(times, conc, t2, method=method)

        if not np.isfinite(c_at_t1) or not np.isfinite(c_at_t2):
            return float("nan")

        # Build combined array: interpolated endpoints + interior observations
        interior_mask = (times > t1) & (times < t2)
        t_interior = times[interior_mask]
        c_interior = conc[interior_mask]

        t_combined = np.concatenate([[t1], t_interior, [t2]])
        c_combined = np.concatenate([[c_at_t1], c_interior, [c_at_t2]])

        return self._compute_auc(t_combined, c_combined)

    def compute_multidose_subject(
        self,
        times: np.ndarray,
        conc: np.ndarray,
        dose: float,
        tau: float,
        subject_id: int | str = 0,
        route: str = "oral",
        ss: bool = True,
        sd_auc_inf: float | None = None,
        infusion_duration: float | None = None,
    ) -> NCAParameters:
        """
        NCA for a multiple-dose or steady-state profile.

        Calls compute_subject for the full profile, then additionally
        computes:
        - auc_tau: AUC over last complete dosing interval
        - c_min: minimum concentration in that interval (trough)
        - c_avg: auc_tau / tau
        - fluctuation: (cmax - c_min) / c_avg * 100
        - swing: (cmax - c_min) / c_min  (NaN if c_min == 0)
        - r_ac: auc_tau / sd_auc_inf if sd_auc_inf is provided
        - norm_* fields if dose > 0

        Args:
            times:      Sampling times (may span multiple doses).
            conc:       Concentrations.
            dose:       Dose amount.
            tau:        Dosing interval.
            subject_id: Subject identifier.
            route:      Route of administration.
            ss:         Whether data represents steady state.
            sd_auc_inf: Single-dose AUC_inf for accumulation ratio
                        computation.
            infusion_duration:
                        Optional infusion duration (Tinf) forwarded to
                        compute_subject() for IV infusion MRT adjustment.

        Returns:
            NCAParameters with all standard fields plus multidose fields
            populated.
        """
        times = np.asarray(times, dtype=float)
        conc = np.asarray(conc, dtype=float)

        # First compute standard NCA
        params = self.compute_subject(
            times,
            conc,
            dose,
            subject_id,
            route,
            infusion_duration=infusion_duration,
        )

        # Determine last dosing interval: [t_last - tau, t_last]
        pos_finite_mask = np.isfinite(conc) & (conc > 0) & np.isfinite(times)
        if not np.any(pos_finite_mask):
            return params

        t_last = float(np.nanmax(times[pos_finite_mask]))
        t_start_interval = t_last - tau

        # Filter to the last complete dosing interval
        mask = (times >= t_start_interval) & (times <= t_last)
        t_interval = times[mask]
        c_interval = conc[mask]

        if len(t_interval) >= 2:
            params.auc_tau = self._compute_auc(t_interval, c_interval)
            params.c_min = float(np.nanmin(c_interval))
            if tau > 0:
                params.c_avg = params.auc_tau / tau
            if np.isfinite(params.c_avg) and params.c_avg > 0:
                params.fluctuation = (params.cmax - params.c_min) / params.c_avg * 100.0
            if params.c_min > 0:
                params.swing = (params.cmax - params.c_min) / params.c_min
            if sd_auc_inf is not None and sd_auc_inf > 0 and np.isfinite(params.auc_tau):
                params.r_ac = params.auc_tau / sd_auc_inf

        # Dose-normalized
        if dose > 0:
            if np.isfinite(params.cmax):
                params.norm_cmax = params.cmax / dose
            if np.isfinite(params.auc_last):
                params.norm_auc_last = params.auc_last / dose
            if np.isfinite(params.auc_inf):
                params.norm_auc_inf = params.auc_inf / dose

        return params

    def apply_predose_blq_rule(
        self,
        times: np.ndarray,
        conc: np.ndarray,
        lloq: float,
        rule: Literal["zero", "lloq_half", "exclude"] = "zero",
    ) -> np.ndarray:
        """
        Apply BLQ handling rule to pre-dose and post-dose BLQ concentrations.

        Rules:
        - 'zero': Set concentrations below LLOQ to 0
        - 'lloq_half': Set concentrations below LLOQ to LLOQ/2
        - 'exclude': Set concentrations below LLOQ to NaN (exclude from
          analysis)

        Args:
            times: Sampling times
            conc:  Observed concentrations
            lloq:  Lower limit of quantification
            rule:  BLQ handling rule

        Returns:
            Modified concentration array with BLQ values handled.
        """
        conc_out = np.asarray(conc, dtype=float).copy()
        blq_mask = np.isfinite(conc_out) & (conc_out < lloq)
        if rule == "zero":
            conc_out[blq_mask] = 0.0
        elif rule == "lloq_half":
            conc_out[blq_mask] = lloq / 2.0
        elif rule == "exclude":
            conc_out[blq_mask] = np.nan
        else:
            raise ValueError(f"Unknown BLQ rule: {rule!r}. Use 'zero', 'lloq_half', or 'exclude'.")
        return conc_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate_conc(
        self,
        times: np.ndarray,
        conc: np.ndarray,
        t_target: float,
        method: Literal["linear", "log"] = "log",
    ) -> float:
        """Interpolate concentration at ``t_target`` without extrapolating."""
        exact = times == t_target
        if np.any(exact):
            return float(conc[exact][0])

        idx_after = int(np.searchsorted(times, t_target, side="left"))
        if idx_after == 0 or idx_after >= len(times):
            return float("nan")

        t_lo = float(times[idx_after - 1])
        t_hi = float(times[idx_after])
        c_lo = float(conc[idx_after - 1])
        c_hi = float(conc[idx_after])
        if not (np.isfinite(c_lo) and np.isfinite(c_hi)):
            return float("nan")

        frac = (t_target - t_lo) / (t_hi - t_lo)
        if method == "log" and c_lo > 0.0 and c_hi > 0.0:
            log_c = np.log(c_lo) + frac * (np.log(c_hi) - np.log(c_lo))
            return float(np.exp(log_c))
        return float(c_lo + frac * (c_hi - c_lo))

    def _augment_iv_bolus_profile(
        self,
        times: np.ndarray,
        conc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Prepend a back-extrapolated IV bolus C0 when time zero is unobserved."""
        zero_pos = np.flatnonzero((times == 0.0) & (conc > 0.0))
        if len(zero_pos) > 0:
            return times, conc, float(conc[zero_pos[0]])

        pos_indices = np.flatnonzero(conc > 0.0)
        if len(pos_indices) < 2:
            return times, conc, float("nan")

        first_pos = int(pos_indices[0])
        second_pos = int(pos_indices[1])
        t1 = float(times[first_pos])
        t2 = float(times[second_pos])
        c1 = float(conc[first_pos])
        c2 = float(conc[second_pos])

        if t1 <= 0.0 or t2 <= t1 or c1 <= 0.0 or c2 <= 0.0:
            return times, conc, float("nan")

        lambda_initial = (np.log(c1) - np.log(c2)) / (t2 - t1)
        if not np.isfinite(lambda_initial) or lambda_initial <= 0.0:
            return times, conc, float("nan")

        c0 = float(np.exp(np.log(c1) + lambda_initial * t1))
        if not np.isfinite(c0) or c0 <= 0.0:
            return times, conc, float("nan")

        return (
            np.concatenate([[0.0], times[first_pos:]]),
            np.concatenate([[c0], conc[first_pos:]]),
            c0,
        )

    def _compute_auc(self, times: np.ndarray, conc: np.ndarray) -> float:
        """
        Compute AUC using the selected trapezoidal method.

        For 'linear-trapezoidal': uses linear rule for all intervals.
        For 'linear-log' / 'linear-up-log-down': uses linear rule when
        concentration is rising (c2 >= c1) and log trapezoidal when
        concentration is declining (c2 < c1).

        Args:
            times: Time values, must be sorted ascending.
            conc:  Concentration values corresponding to times.

        Returns:
            Scalar AUC value. Returns 0.0 if fewer than 2 valid points.
        """
        if len(times) < 2:
            return 0.0

        if len(times) <= 32:
            auc = 0.0
            linear_only = self.auc_method == "linear-trapezoidal"
            for i in range(len(times) - 1):
                t1, t2 = float(times[i]), float(times[i + 1])
                c1, c2 = float(conc[i]), float(conc[i + 1])

                if not (np.isfinite(c1) and np.isfinite(c2)):
                    continue

                if linear_only or c2 >= c1:
                    auc += (c1 + c2) * 0.5 * (t2 - t1)
                else:
                    auc += self._log_trapezoidal(t1, t2, c1, c2)
            return float(auc)

        dt = np.diff(times)
        cv1: np.ndarray = conc[:-1]
        cv2: np.ndarray = conc[1:]
        valid = np.isfinite(cv1) & np.isfinite(cv2)
        if not np.any(valid):
            return 0.0

        linear_area = 0.5 * (cv1 + cv2) * dt
        if self.auc_method == "linear-trapezoidal":
            return float(np.sum(linear_area[valid], dtype=float))

        auc = float(np.sum(linear_area[valid], dtype=float))
        decline_mask = valid & (cv2 < cv1)
        if not np.any(decline_mask):
            return auc

        positive_decline = decline_mask & (cv1 > 0.0) & (cv2 > 0.0)
        if not np.any(positive_decline):
            return auc

        tolerance = 1e-10 * np.maximum(np.maximum(np.abs(cv1), np.abs(cv2)), 1e-30)
        near_equal = positive_decline & (np.abs(cv1 - cv2) < tolerance)
        if np.any(near_equal):
            same_area = cv1 * dt
            auc += float(np.sum(same_area[near_equal] - linear_area[near_equal], dtype=float))

        log_mask = positive_decline & ~near_equal
        if np.any(log_mask):
            c1_log = cv1[log_mask]
            c2_log = cv2[log_mask]
            dt_log = dt[log_mask]
            log_area = (c1_log - c2_log) / np.log(c1_log / c2_log) * dt_log
            auc += float(np.sum(log_area - linear_area[log_mask], dtype=float))

        return auc

    def _compute_auc_pair(self, times: np.ndarray, conc: np.ndarray) -> tuple[float, float]:
        """Compute AUC and AUMC together for one subject profile."""
        if len(times) < 2:
            return 0.0, 0.0

        auc = 0.0
        aumc = 0.0
        linear_only = self.auc_method == "linear-trapezoidal"
        for i in range(len(times) - 1):
            t1, t2 = float(times[i]), float(times[i + 1])
            c1, c2 = float(conc[i]), float(conc[i + 1])
            if not (np.isfinite(c1) and np.isfinite(c2)):
                continue

            dt = t2 - t1
            if linear_only or c2 >= c1:
                auc += (c1 + c2) * 0.5 * dt
                m1 = t1 * c1
                m2 = t2 * c2
                aumc += (m1 + m2) * 0.5 * dt
            else:
                auc += self._log_trapezoidal(t1, t2, c1, c2)
                aumc += self._log_aumc_trapezoidal(t1, t2, c1, c2)

        return float(auc), float(aumc)

    def _compute_lambda_z(
        self,
        times: np.ndarray,
        conc: np.ndarray,
    ) -> tuple[float, float, int]:
        """
        Estimate terminal elimination rate constant lambda_z.

        Uses log-linear regression (ln(C) = ln(C0) - lambda_z * t) on the
        terminal phase. Automatically selects the regression window that
        maximises the adjusted R² while using at least min_points_lambda
        data points.

        Args:
            times: Times in the terminal phase (post-Cmax), positive concs only.
            conc:  Concentrations corresponding to times (all > 0).

        Returns:
            Tuple (lambda_z, r_squared, n_points).
            lambda_z > 0 if the regression slope is negative (as expected).
            Returns (nan, nan, 0) if estimation fails.
        """
        n = len(times)
        if n < self.min_points_lambda:
            return float("nan"), float("nan"), 0

        if n <= 16:
            log_conc = np.log(conc)
            if not (np.all(np.isfinite(times)) and np.all(np.isfinite(log_conc))):
                return float("nan"), float("nan"), 0

            rev_times = times[::-1]
            rev_log_conc = log_conc[::-1]
            suffix_sum_x = np.cumsum(rev_times)
            suffix_sum_y = np.cumsum(rev_log_conc)
            suffix_sum_xx = np.cumsum(rev_times * rev_times)
            suffix_sum_yy = np.cumsum(rev_log_conc * rev_log_conc)
            suffix_sum_xy = np.cumsum(rev_times * rev_log_conc)

            best_r2_adj = -np.inf
            best_lambda = float("nan")
            best_r2 = float("nan")
            best_n = 0
            for k in range(self.min_points_lambda, n + 1):
                idx = k - 1
                n_points = float(k)

                sum_x = float(suffix_sum_x[idx])
                sum_y = float(suffix_sum_y[idx])
                sum_xx = float(suffix_sum_xx[idx])
                sum_yy = float(suffix_sum_yy[idx])
                sum_xy = float(suffix_sum_xy[idx])

                sxx = sum_xx - (sum_x * sum_x) / n_points
                if sxx <= 0.0 or not np.isfinite(sxx):
                    continue
                sxy = sum_xy - (sum_x * sum_y) / n_points
                slope = sxy / sxx
                if slope >= 0.0 or not np.isfinite(slope):
                    continue
                syy = sum_yy - (sum_y * sum_y) / n_points
                if syy <= 0.0 or not np.isfinite(syy):
                    continue
                r2 = (sxy * sxy) / (sxx * syy)
                if r2 < 0.0:
                    r2 = 0.0
                elif r2 > 1.0:
                    r2 = 1.0
                r2_adj = 1.0 - (1.0 - r2) * (k - 1) / (k - 2) if k > 2 else r2
                if r2_adj > best_r2_adj:
                    best_r2_adj = r2_adj
                    best_lambda = -slope
                    best_r2 = r2
                    best_n = k

            if not np.isfinite(best_lambda) or best_lambda <= 0:
                return float("nan"), float("nan"), 0
            return float(best_lambda), float(best_r2), best_n

        log_conc = np.log(conc)
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(log_conc))):
            return float("nan"), float("nan"), 0

        rev_times = times[::-1]
        rev_log_conc = log_conc[::-1]
        suffix_sum_x = np.cumsum(rev_times)
        suffix_sum_y = np.cumsum(rev_log_conc)
        suffix_sum_xx = np.cumsum(rev_times * rev_times)
        suffix_sum_yy = np.cumsum(rev_log_conc * rev_log_conc)
        suffix_sum_xy = np.cumsum(rev_times * rev_log_conc)

        k_values = np.arange(self.min_points_lambda, n + 1)
        v_idx: np.ndarray = k_values - 1
        v_n_points: np.ndarray = k_values.astype(float)

        sum_x = suffix_sum_x[v_idx]
        sum_y = suffix_sum_y[v_idx]
        sum_xx = suffix_sum_xx[v_idx]
        sum_yy = suffix_sum_yy[v_idx]
        sum_xy = suffix_sum_xy[v_idx]

        sxx = sum_xx - (sum_x * sum_x) / v_n_points
        sxy = sum_xy - (sum_x * sum_y) / v_n_points
        syy = sum_yy - (sum_y * sum_y) / v_n_points

        valid = np.isfinite(sxx) & np.isfinite(sxy) & np.isfinite(syy) & (sxx > 0.0) & (syy > 0.0)
        v_slope: np.ndarray = np.full_like(sxx, np.nan, dtype=float)
        v_slope[valid] = sxy[valid] / sxx[valid]
        valid &= np.isfinite(v_slope) & (v_slope < 0.0)
        if not np.any(valid):
            return float("nan"), float("nan"), 0

        v_r2: np.ndarray = np.full_like(sxx, np.nan, dtype=float)
        v_r2[valid] = (sxy[valid] * sxy[valid]) / (sxx[valid] * syy[valid])
        v_r2[valid] = np.minimum(1.0, np.maximum(0.0, v_r2[valid]))

        v_r2_adj: np.ndarray = np.full_like(sxx, -np.inf, dtype=float)
        v_r2_adj[valid] = v_r2[valid]
        adjust_mask = valid & (k_values > 2)
        v_r2_adj[adjust_mask] = 1.0 - (1.0 - v_r2[adjust_mask]) * (k_values[adjust_mask] - 1) / (
            k_values[adjust_mask] - 2
        )

        best_pos = int(np.argmax(v_r2_adj))
        best_lambda = float(-v_slope[best_pos])
        best_r2 = float(v_r2[best_pos])
        best_n = int(k_values[best_pos])

        if not np.isfinite(best_lambda) or best_lambda <= 0:
            return float("nan"), float("nan"), 0

        return float(best_lambda), float(best_r2), best_n

    def _extrapolate_auc(
        self,
        auc_last: float,
        c_last: float,
        lambda_z: float,
    ) -> float:
        """
        Extrapolate AUC to infinity.

        AUC_inf = AUC_last + C_last / lambda_z

        Args:
            auc_last:  AUC from 0 to last quantifiable timepoint.
            c_last:    Concentration at last quantifiable timepoint.
            lambda_z:  Terminal elimination rate constant (> 0).

        Returns:
            AUC extrapolated to infinity.
        """
        if lambda_z <= 0:
            return float("nan")
        return float(auc_last + c_last / lambda_z)

    def _linear_trapezoidal(self, t1: float, t2: float, c1: float, c2: float) -> float:
        """
        Linear trapezoidal rule for one interval.

        Area = (c1 + c2) / 2 * (t2 - t1)
        """
        return (c1 + c2) / 2.0 * (t2 - t1)

    def _log_trapezoidal(self, t1: float, t2: float, c1: float, c2: float) -> float:
        """
        Log trapezoidal rule for one interval.

        Area = (c1 - c2) / ln(c1/c2) * (t2 - t1)

        Falls back to linear trapezoidal when c1 or c2 is non-positive,
        or when c1 and c2 are approximately equal (avoids log(1) = 0
        division).
        """
        if c1 <= 0 or c2 <= 0:
            return self._linear_trapezoidal(t1, t2, c1, c2)
        if abs(c1 - c2) < 1e-10 * max(abs(c1), abs(c2), 1e-30):
            return c1 * (t2 - t1)
        return (c1 - c2) / np.log(c1 / c2) * (t2 - t1)

    def _log_aumc_trapezoidal(self, t1: float, t2: float, c1: float, c2: float) -> float:
        """Exact AUMC segment under exponential decline between two samples."""
        if c1 <= 0.0 or c2 <= 0.0:
            return self._linear_trapezoidal(t1, t2, t1 * c1, t2 * c2)
        if abs(c1 - c2) < 1e-10 * max(abs(c1), abs(c2), 1e-30):
            return c1 * (t2 * t2 - t1 * t1) / 2.0

        lambda_z = (np.log(c1) - np.log(c2)) / (t2 - t1)
        if not np.isfinite(lambda_z) or lambda_z <= 0.0:
            return self._linear_trapezoidal(t1, t2, t1 * c1, t2 * c2)
        return ((t1 * c1 - t2 * c2) / lambda_z) + ((c1 - c2) / (lambda_z**2))
