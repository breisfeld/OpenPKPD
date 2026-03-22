"""
Drug-drug interaction (DDI) index calculators.

Implements reversible competitive inhibition, time-dependent inhibition (TDI),
and induction DDI R-value calculations. Also provides a DDIStudyAnalysis class
for back-calculating Ki from observed AUC ratios.

References:
    FDA Guidance for Industry: Drug Interaction Studies -- Study Design,
        Data Analysis, Implications for Dosing, and Labeling Recommendations (2012).
    Einolf HJ. (2007). Comparison of in vitro and in vivo metabolic interactions
        with itraconazole. Xenobiotica 37(10-11):1090-1106.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class DDIResult:
    """
    Result of a DDI index calculation or study analysis.

    Attributes:
        r_value:            AUC ratio (victim AUC on/off perpetrator). R > 1 = inhibition.
        cl_ratio:           CL ratio = 1/r_value.
        perpetrator_params: Input or back-calculated perpetrator parameters.
        mechanism:          Mechanism label: 'competitive', 'TDI', or 'induction'.
        fm:                 Fraction metabolised by the inhibited pathway.
        auc_ratio:          Same as r_value (alias).
    """

    r_value: float
    cl_ratio: float
    perpetrator_params: dict[str, float]
    mechanism: str
    fm: float
    auc_ratio: float

    def summary(self) -> str:
        lines = [
            f"DDI Index ({self.mechanism})",
            f"  AUC ratio (on/off): {self.r_value:.3f}",
            f"  CL ratio:           {self.cl_ratio:.3f}",
            f"  fm:                 {self.fm:.3f}",
        ]
        for k, v in self.perpetrator_params.items():
            lines.append(f"  {k}: {v:.4g}")
        return "\n".join(lines)


def competitive_inhibition_r(
    inhibitor_conc: float,
    ki: float,
    fm: float = 1.0,
) -> float:
    """
    Compute the AUC ratio for reversible competitive inhibition (FDA static model).

    R = 1 + [I] / Ki
    AUC_ratio = 1 / (1 - fm * (1 - 1/R))

    Args:
        inhibitor_conc: Inhibitor concentration at the enzyme site (same units as Ki).
        ki:             Inhibition constant.
        fm:             Fraction of victim metabolism via the inhibited pathway (0-1).

    Returns:
        AUC ratio (victim AUC with inhibitor / without inhibitor). > 1 = inhibition.

    Raises:
        ValueError: If ki <= 0 or fm not in (0, 1].
    """
    if ki <= 0:
        raise ValueError(f"ki must be > 0; got {ki}")
    if not (0 < fm <= 1.0):
        raise ValueError(f"fm must be in (0, 1]; got {fm}")

    r = 1.0 + inhibitor_conc / ki
    auc_ratio = 1.0 / (1.0 - fm * (1.0 - 1.0 / r))
    return float(auc_ratio)


def time_dependent_inhibition_r(
    inhibitor_conc: float,
    kinact: float,
    ki_app: float,
    degradation_rate: float = 0.03,
    fm: float = 1.0,
) -> float:
    """
    Compute the AUC ratio for time-dependent inhibition (TDI) using the FDA model.

    TDI accounts for mechanism-based inactivation (MBI) where the inhibitor
    inactivates the enzyme over time.

    Effective inactivation rate: k_obs = kinact * [I] / (KI_app + [I])
    Adjusted kdeg: kdeg_adj = kdeg + k_obs
    R_TDI = kdeg_adj / kdeg  (ratio of degradation rates)
    AUC_ratio = 1 / (1 - fm * (1 - 1/R_TDI))

    Args:
        inhibitor_conc:  Inhibitor concentration at enzyme site.
        kinact:          Maximum inactivation rate (min^-1).
        ki_app:          Apparent inhibitor concentration for half-maximal inactivation.
        degradation_rate: Enzyme degradation rate constant (default 0.03 h^-1
                         -> 0.0005 min^-1; passed in same units as kinact).
        fm:              Fraction of victim via inhibited pathway.

    Returns:
        AUC ratio. > 1 = inhibition.
    """
    if ki_app <= 0:
        raise ValueError(f"ki_app must be > 0; got {ki_app}")
    if degradation_rate <= 0:
        raise ValueError(f"degradation_rate must be > 0; got {degradation_rate}")
    if not (0 < fm <= 1.0):
        raise ValueError(f"fm must be in (0, 1]; got {fm}")

    k_obs = kinact * inhibitor_conc / (ki_app + inhibitor_conc)
    kdeg_adj = degradation_rate + k_obs
    r_tdi = kdeg_adj / degradation_rate
    auc_ratio = 1.0 / (1.0 - fm * (1.0 - 1.0 / r_tdi))
    return float(auc_ratio)


def induction_r(
    inhibitor_conc: float,
    emax_ind: float,
    ec50_ind: float,
    baseline_enzyme: float = 1.0,
    fm: float = 1.0,
) -> float:
    """
    Compute the AUC ratio for CYP induction (static model).

    Fold-induction = 1 + Emax * [I] / (EC50 + [I])
    The victim AUC decreases (ratio < 1 means reduced exposure).

    R_ind = baseline_enzyme * fold_induction
    AUC_ratio = 1 / (1 - fm * (1 - 1/R_ind)) -- note: < 1 for induction

    Args:
        inhibitor_conc:  Inducer concentration at enzyme site.
        emax_ind:        Maximum fold-induction above baseline.
        ec50_ind:        Inducer concentration at half-maximal induction.
        baseline_enzyme: Baseline enzyme activity (default 1.0).
        fm:              Fraction of victim via induced pathway.

    Returns:
        AUC ratio. < 1 = induction (reduced victim exposure).
    """
    if ec50_ind <= 0:
        raise ValueError(f"ec50_ind must be > 0; got {ec50_ind}")
    if not (0 < fm <= 1.0):
        raise ValueError(f"fm must be in (0, 1]; got {fm}")

    fold_ind = 1.0 + emax_ind * inhibitor_conc / (ec50_ind + inhibitor_conc)
    r_ind = baseline_enzyme * fold_ind
    # Induction increases enzyme activity, increasing victim CL.
    # AUC_ratio = fm/r_ind + (1 - fm) = 1 - fm*(1 - 1/r_ind)
    # This yields < 1 for r_ind > 1 (i.e., when induction occurs).
    auc_ratio = 1.0 - fm * (1.0 - 1.0 / r_ind)
    return float(auc_ratio)


class DDIStudyAnalysis:
    """
    Analyse observed DDI study data to back-calculate inhibition parameters.

    Back-calculates Ki (reversible) or KI (TDI) from the observed AUC ratio
    in an in vivo DDI study using the static FDA model.
    """

    def fit_reversible_ki(
        self,
        auc_ratio_observed: float,
        inhibitor_conc: float,
        fm: float = 1.0,
    ) -> DDIResult:
        """
        Back-calculate Ki from an observed DDI AUC ratio (reversible inhibition).

        Solves: auc_ratio_obs = 1 / (1 - fm * (1 - 1/(1 + I/Ki)))  for Ki.

        Args:
            auc_ratio_observed: Observed AUC ratio (with/without inhibitor). > 1.
            inhibitor_conc:     Inhibitor concentration at enzyme site.
            fm:                 Fraction metabolised via inhibited pathway.

        Returns:
            DDIResult with back-calculated Ki.
        """
        if auc_ratio_observed <= 1.0:
            raise ValueError("auc_ratio_observed must be > 1 for inhibition.")
        if inhibitor_conc <= 0:
            raise ValueError("inhibitor_conc must be > 0.")

        # Solve: AUC_ratio = 1/(1 - fm*(1 - 1/R))  where R = 1 + I/Ki
        # -> R = fm / (fm - 1 + 1/AUC_ratio)
        # -> Ki = I / (R - 1)
        denom = fm - 1.0 + 1.0 / auc_ratio_observed
        if abs(denom) < 1e-12:
            ki = float("inf")
        else:
            r = fm / denom
            ki = inhibitor_conc / (r - 1.0) if r > 1.0 else float("inf")

        return DDIResult(
            r_value=auc_ratio_observed,
            cl_ratio=1.0 / auc_ratio_observed,
            perpetrator_params={"Ki": ki, "inhibitor_conc": inhibitor_conc},
            mechanism="competitive",
            fm=fm,
            auc_ratio=auc_ratio_observed,
        )

    def fit_tdi_ki(
        self,
        auc_ratio_observed: float,
        inhibitor_conc: float,
        kinact: float,
        degradation_rate: float = 0.03,
        fm: float = 1.0,
    ) -> DDIResult:
        """
        Back-calculate KI_app from an observed DDI AUC ratio (TDI).

        Numerically solves for KI_app given the observed AUC ratio.

        Args:
            auc_ratio_observed: Observed AUC ratio. > 1.
            inhibitor_conc:     Inhibitor concentration at enzyme site.
            kinact:             Maximum inactivation rate.
            degradation_rate:   Enzyme degradation rate.
            fm:                 Fraction via inhibited pathway.

        Returns:
            DDIResult with back-calculated KI_app.
        """
        if auc_ratio_observed <= 1.0:
            raise ValueError("auc_ratio_observed must be > 1.")
        if inhibitor_conc <= 0:
            raise ValueError("inhibitor_conc must be > 0.")
        if degradation_rate <= 0:
            raise ValueError("degradation_rate must be > 0.")

        def residual(log_ki_app: np.ndarray) -> float:
            ki_app = float(np.exp(log_ki_app[0]))
            pred = time_dependent_inhibition_r(inhibitor_conc, kinact, ki_app, degradation_rate, fm)
            return (np.log(pred) - np.log(auc_ratio_observed)) ** 2

        x0 = np.array([0.0])  # log(ki_app) = 0 -> ki_app = 1
        result = optimize.minimize(residual, x0, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"TDI KI_app optimization failed: {result.message}")
        ki_app = float(np.exp(result.x[0]))

        return DDIResult(
            r_value=auc_ratio_observed,
            cl_ratio=1.0 / auc_ratio_observed,
            perpetrator_params={
                "KI_app": ki_app,
                "kinact": kinact,
                "inhibitor_conc": inhibitor_conc,
            },
            mechanism="TDI",
            fm=fm,
            auc_ratio=auc_ratio_observed,
        )
