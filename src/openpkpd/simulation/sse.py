"""
Stochastic Simulation and Re-estimation (SSE).

Assesses estimator performance (bias, RMSE, coverage) by repeatedly
simulating datasets from a fitted model and re-fitting each replicate.

Reference:
    Holford NHG et al. (2000). Simulation of clinical trials.
    Annu Rev Pharmacol Toxicol 40:209-234.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openpkpd.estimation.base import EstimationResult
    from openpkpd.model.population import PopulationModel


def _empirical_coverage(estimates: np.ndarray, true_value: float) -> float:
    """
    Compute empirical 95% coverage using the SD of replicate estimates as a
    proxy for the per-replicate standard error.

    For each replicate i the approximate 95% CI is
    ``[est_i − 1.96·SD_all,  est_i + 1.96·SD_all]``
    where ``SD_all`` is the empirical SD across all converged replicates.
    Coverage is the fraction of replicates whose CI contains ``true_value``.

    This is the standard fallback when per-replicate covariance-step SEs are
    unavailable.  When per-replicate SEs are available, prefer:
    ``np.mean(np.abs(estimates - true_value) <= 1.96 * se_per_replicate)``.

    Args:
        estimates:   Array of replicate parameter estimates (finite values only).
        true_value:  Generating (true) parameter value.

    Returns:
        Coverage fraction in [0, 1], or ``nan`` if fewer than 2 estimates.
    """
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        # All estimates are identical — check if true value equals that estimate
        return 1.0 if np.isclose(finite[0], true_value) else 0.0
    in_ci = np.abs(finite - true_value) <= 1.96 * sd
    return float(np.mean(in_ci))


@dataclass
class SSEResult:
    """
    Results from a Stochastic Simulation and Re-estimation study.

    Attributes:
        n_replicates:     Number of simulation-estimation cycles.
        parameter_names:  Names of all estimated parameters.
        true_values:      True parameter values (from the generating model).
        estimates:        DataFrame (n_rep × n_params) with one row per converged replicate.
        bias:             Relative bias per parameter = mean(est - true) / true.
        rmse:             Root mean squared error per parameter.
        coverage_95:      Fraction of replicates where the 95% CI contains the true value.
        convergence_rate: Fraction of replicates that converged.
    """

    n_replicates: int
    parameter_names: list[str]
    true_values: dict[str, float]
    estimates: pd.DataFrame
    bias: dict[str, float]
    rmse: dict[str, float]
    coverage_95: dict[str, float]
    convergence_rate: float

    def summary(self) -> str:
        lines = [
            f"SSE Results (n={self.n_replicates}, convergence={self.convergence_rate:.1%})",
            f"{'Parameter':<20} {'Bias%':>8} {'RMSE':>10} {'Coverage95%':>12}",
            "-" * 55,
        ]
        for name in self.parameter_names:
            bias_pct = self.bias.get(name, float("nan")) * 100
            rmse = self.rmse.get(name, float("nan"))
            cov = self.coverage_95.get(name, float("nan")) * 100
            lines.append(f"  {name:<18} {bias_pct:>8.2f} {rmse:>10.4f} {cov:>11.1f}%")
        return "\n".join(lines)


class SSEEngine:
    """
    Stochastic Simulation and Re-estimation engine.

    Simulates n_replicates datasets from a fitted population model and
    re-estimates the model parameters for each replicate.

    Args:
        population_model: Assembled PopulationModel.
        true_result:      Fitted EstimationResult providing true parameter values.
        estimation_method: Estimation method name (default 'FOCE').
    """

    def __init__(
        self,
        population_model: PopulationModel,
        true_result: EstimationResult,
        estimation_method: str = "FOCE",
    ) -> None:
        self.population_model = population_model
        self.true_result = true_result
        self.estimation_method = estimation_method

    def _build_reestimation_dataset(self, rep_df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the original dataset with observation DVs replaced by one replicate."""
        dataset_df = self.population_model.dataset.df.copy()
        obs_mask = (dataset_df["EVID"] == 0) & (dataset_df["MDV"] == 0)

        orig_obs = dataset_df.loc[obs_mask, ["ID", "TIME"]].copy()
        orig_obs["_OBSSEQ"] = orig_obs.groupby(["ID", "TIME"]).cumcount()

        rep_obs = rep_df[["ID", "TIME", "DV"]].copy()
        rep_obs["_OBSSEQ"] = rep_obs.groupby(["ID", "TIME"]).cumcount()

        merged = orig_obs.merge(rep_obs, on=["ID", "TIME", "_OBSSEQ"], how="left", sort=False)
        if merged["DV"].isna().any():
            raise ValueError("Simulated replicate is missing one or more observation DV values.")

        dataset_df.loc[obs_mask, "DV"] = merged["DV"].to_numpy()
        return dataset_df

    def _clone_population_model_with_dataset(self, dataset: pd.DataFrame) -> PopulationModel:
        """Shallow-copy the population model, replace its dataset, and refresh subject caches."""
        from openpkpd.data.dataset import NONMEMDataset

        cloned = copy.copy(self.population_model)
        cloned.dataset = NONMEMDataset.from_dataframe(dataset)
        if hasattr(cloned, "_setup_subjects"):
            cloned._setup_subjects()
        return cloned

    def run(
        self,
        n_replicates: int = 100,
        seed: int = 42,
        n_jobs: int = 1,
    ) -> SSEResult:
        """
        Run the SSE: simulate n_replicates datasets and re-estimate each.

        Algorithm:
            1. Use SimulationEngine to generate n_replicates simulated datasets.
            2. For each replicate, re-run estimation with estimation_method.
            3. Collect parameter estimates and convergence flags.
            4. Compute bias, RMSE, and 95% coverage.

        Args:
            n_replicates: Number of simulation-estimation cycles.
            seed:         Random seed for reproducibility.
            n_jobs:       Number of parallel jobs (currently sequential; > 1 reserved).

        Returns:
            SSEResult with full summary statistics.
        """
        from openpkpd.estimation import get_estimation_method
        from openpkpd.simulation.engine import SimulationEngine

        rng = np.random.default_rng(seed)

        # Extract true parameter values
        theta_true = self.true_result.theta_final
        omega_true = self.true_result.omega_final
        sigma_true = self.true_result.sigma_final

        n_theta = len(theta_true)
        param_names = [f"THETA{i + 1}" for i in range(n_theta)]
        for i in range(omega_true.shape[0]):
            param_names.append(f"OMEGA({i + 1},{i + 1})")
        for i in range(sigma_true.shape[0]):
            param_names.append(f"SIGMA({i + 1},{i + 1})")

        true_values: dict[str, float] = {}
        for i, name in enumerate(param_names[:n_theta]):
            true_values[name] = float(theta_true[i])
        for i in range(omega_true.shape[0]):
            true_values[f"OMEGA({i + 1},{i + 1})"] = float(omega_true[i, i])
        for i in range(sigma_true.shape[0]):
            true_values[f"SIGMA({i + 1},{i + 1})"] = float(sigma_true[i, i])

        # Run SSE replicates
        all_estimates: list[dict[str, float]] = []
        converged_flags: list[bool] = []

        for _rep in range(n_replicates):
            rep_seed = int(rng.integers(0, 2**31))
            try:
                # Simulate one dataset
                sim_engine = SimulationEngine(
                    self.population_model,
                    self.true_result,
                    seed=rep_seed,
                )
                sim_result = sim_engine.simulate(n_replicates=1)

                # Extract replicate data (REP=1)
                rep_df = sim_result.simulated_df[sim_result.simulated_df["REP"] == 1].copy()
                rep_dataset = self._build_reestimation_dataset(rep_df)
                rep_model = self._clone_population_model_with_dataset(rep_dataset)

                # Re-estimate
                estimator = get_estimation_method(self.estimation_method)
                est_result = estimator.estimate(rep_model, rep_model.params)

                # Collect estimates
                est_row: dict[str, float] = {}
                for i, name in enumerate(param_names[:n_theta]):
                    est_row[name] = float(est_result.theta_final[i])
                for i in range(omega_true.shape[0]):
                    est_row[f"OMEGA({i + 1},{i + 1})"] = float(est_result.omega_final[i, i])
                for i in range(sigma_true.shape[0]):
                    est_row[f"SIGMA({i + 1},{i + 1})"] = float(est_result.sigma_final[i, i])

                all_estimates.append(est_row)
                converged_flags.append(bool(est_result.converged))

            except Exception:
                converged_flags.append(False)

        convergence_rate = float(sum(converged_flags) / n_replicates) if n_replicates > 0 else 0.0
        estimates_df = (
            pd.DataFrame(all_estimates) if all_estimates else pd.DataFrame(columns=param_names)
        )

        # Compute bias, RMSE, coverage
        bias: dict[str, float] = {}
        rmse: dict[str, float] = {}
        coverage_95: dict[str, float] = {}

        for name in param_names:
            if name not in estimates_df.columns or len(estimates_df) == 0:
                bias[name] = float("nan")
                rmse[name] = float("nan")
                coverage_95[name] = float("nan")
                continue
            est_vals = estimates_df[name].values
            true_val = true_values.get(name, float("nan"))
            finite_mask = np.isfinite(est_vals)
            if not np.any(finite_mask) or not np.isfinite(true_val) or true_val == 0:
                bias[name] = float("nan")
                rmse[name] = float("nan")
                coverage_95[name] = float("nan")
            else:
                finite_vals = est_vals[finite_mask]
                bias[name] = float(np.mean(finite_vals - true_val) / true_val)
                rmse[name] = float(np.sqrt(np.mean((finite_vals - true_val) ** 2)))
                coverage_95[name] = _empirical_coverage(finite_vals, true_val)

        return SSEResult(
            n_replicates=n_replicates,
            parameter_names=param_names,
            true_values=true_values,
            estimates=estimates_df,
            bias=bias,
            rmse=rmse,
            coverage_95=coverage_95,
            convergence_rate=convergence_rate,
        )
