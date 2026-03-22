"""
Urinary excretion NCA analysis.

Computes renal NCA parameters from urine collection interval data:
amount excreted, fraction excreted, renal clearance, and excretion rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpkpd.nca.nca import NCAParameters


@dataclass
class UrineNCAParameters:
    """
    NCA parameters from urinary excretion data.

    Attributes:
        subject_id:    Subject identifier.
        dose:          Administered dose.
        ae_last:       Cumulative amount excreted to last timepoint.
        ae_inf:        Extrapolated total amount excreted (Ae_inf).
        fe:            Fraction excreted = Ae_inf / dose (dimensionless).
        cl_renal:      Renal clearance = Ae_inf / AUC_inf (volume/time).
        intervals:     Per-interval data: t_start, t_end, delta_ae, rate_mid.
    """

    subject_id: int | str
    dose: float
    ae_last: float = float("nan")
    ae_inf: float = float("nan")
    fe: float = float("nan")
    cl_renal: float = float("nan")
    intervals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dose": self.dose,
            "ae_last": self.ae_last,
            "ae_inf": self.ae_inf,
            "fe": self.fe,
            "cl_renal": self.cl_renal,
            "n_intervals": len(self.intervals),
        }

    def summary(self) -> str:
        lines = [
            f"Urine NCA — Subject {self.subject_id}",
            f"  Dose:      {self.dose}",
            f"  Ae_last:   {self.ae_last:.4g}",
            f"  Ae_inf:    {self.ae_inf:.4g}",
            f"  fe:        {self.fe:.4f}",
            f"  CLrenal:   {self.cl_renal:.4g}",
        ]
        return "\n".join(lines)


class UrineNCAEngine:
    """
    Engine for urinary excretion NCA.

    Args:
        plasma_engine: Optional NCAEngine for plasma data (for renal
                       clearance).
    """

    def compute_subject(
        self,
        subject_id: int | str,
        dose: float,
        collection_times: np.ndarray,
        delta_amounts: np.ndarray,
        plasma_nca: NCAParameters | None = None,
    ) -> UrineNCAParameters:
        """
        Compute urine NCA parameters for one subject.

        Args:
            subject_id:        Subject identifier.
            dose:              Administered dose.
            collection_times:  Array of collection interval boundaries
                               (length n+1).  collection_times[i] to
                               collection_times[i+1] is interval i.
            delta_amounts:     Amount collected in each interval (length n).
                               delta_amounts[i] = amount excreted in
                               [collection_times[i], collection_times[i+1]].
            plasma_nca:        Optional plasma NCAParameters for renal
                               clearance computation.

        Returns:
            UrineNCAParameters with all computed parameters.
        """
        collection_times = np.asarray(collection_times, dtype=float)
        delta_amounts = np.asarray(delta_amounts, dtype=float)

        n = len(delta_amounts)
        if len(collection_times) != n + 1:
            raise ValueError(
                f"collection_times must have length len(delta_amounts)+1; "
                f"got {len(collection_times)} and {n}."
            )

        params = UrineNCAParameters(subject_id=subject_id, dose=dose)

        # Build per-interval data
        intervals: list[dict] = []
        cumulative = 0.0
        for i in range(n):
            t_start = float(collection_times[i])
            t_end = float(collection_times[i + 1])
            dt = t_end - t_start
            dae = float(delta_amounts[i]) if np.isfinite(delta_amounts[i]) else 0.0
            cumulative += dae
            rate_mid = dae / dt if dt > 0 else float("nan")
            t_mid = (t_start + t_end) / 2.0
            intervals.append(
                {
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_mid": t_mid,
                    "delta_ae": dae,
                    "rate_mid": rate_mid,
                    "cumulative_ae": cumulative,
                }
            )

        params.intervals = intervals
        params.ae_last = float(cumulative)
        params.ae_inf = params.ae_last

        # Extrapolate: ae_inf = ae_last + rate_last / lambda_z
        # Use the last interval rate and plasma lambda_z if available.
        if plasma_nca is not None:
            lambda_z = getattr(plasma_nca, "lambda_z", float("nan"))
            if np.isfinite(lambda_z) and lambda_z > 0 and len(intervals) > 0:
                rate_last = intervals[-1]["rate_mid"]
                if np.isfinite(rate_last):
                    params.ae_inf = params.ae_last + rate_last / lambda_z

        # Fraction excreted
        if dose > 0 and np.isfinite(params.ae_inf):
            params.fe = params.ae_inf / dose

        # Renal clearance = Ae_inf / AUC_inf
        if (
            plasma_nca is not None
            and np.isfinite(getattr(plasma_nca, "auc_inf", float("nan")))
            and plasma_nca.auc_inf > 0
            and np.isfinite(params.ae_inf)
        ):
            params.cl_renal = params.ae_inf / plasma_nca.auc_inf

        return params
