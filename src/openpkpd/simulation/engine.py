"""
SimulationEngine: simulate replicate datasets from a fitted population model.

For each replicate:
  1. Draw ETAs ~ MVN(0, OMEGA_final) for each subject.
  2. Evaluate individual predictions (IPRED) using the PK model.
  3. Draw EPS ~ MVN(0, SIGMA_final) for each observation.
  4. Evaluate $ERROR to obtain simulated DV = Y(IPRED, EPS).

The resulting SimulationResult contains all replicates stacked in a single
DataFrame with a REP column: REP=0 is the observed data, REP=1..n_replicates
are simulated datasets.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from openpkpd.data.event_processor import DoseEvent

if TYPE_CHECKING:
    pass


@dataclass
class SimulationResult:
    """
    Result from a simulation run.

    Attributes:
        simulated_df:  DataFrame with columns ID, TIME, DV, IPRED, PRED, REP,
                       plus ETA1, ETA2, ... for each random effect.
                       REP=0 is the observed dataset; REP=1..n_replicates are simulated.
        seed:          Random seed used for reproducibility.
        n_replicates:  Number of simulated replicates generated.
    """

    simulated_df: pd.DataFrame
    seed: int
    n_replicates: int


class SimulationEngine:
    """
    Simulate replicate datasets from a fitted population PK/PD model.

    Given a PopulationModel and an EstimationResult (containing final parameter
    estimates), this engine draws new random effects (ETAs) and residuals (EPS)
    to generate Monte Carlo replicates of the observed dataset design.

    Args:
        population_model: Assembled PopulationModel with dataset, PK subroutine,
                          compiled $PK and $ERROR callables.
        result:           EstimationResult with theta_final, omega_final, sigma_final,
                          and optionally post_hoc_etas.
        seed:             Integer seed for the random number generator (default 42).
    """

    def __init__(
        self,
        population_model: Any,
        result: Any,
        seed: int = 42,
        n_parallel: int = 1,
    ) -> None:
        self.population_model = population_model
        self.result = result
        self.seed = seed
        self.n_parallel = n_parallel
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_replicates: int = 1,
        new_subjects: int | None = None,
    ) -> SimulationResult:
        """
        Simulate n_replicates copies of the observed dataset design.

        For each replicate:
          1. For each subject, draw ETA_i ~ MVN(0, OMEGA_final).
          2. Solve PK model with (theta_final, ETA_i) to obtain IPRED.
          3. For each observation, draw EPS ~ MVN(0, SIGMA_final).
          4. Evaluate $ERROR(theta, ETA, EPS, IPRED) to get simulated DV.

        The replicate index REP=0 represents the observed data (original DV values).

        Args:
            n_replicates: Number of simulated datasets to generate.
            new_subjects: Unused placeholder; the observed design is replicated.

        Returns:
            SimulationResult with stacked DataFrame.
        """
        theta = self.result.theta_final
        omega = self.result.omega_final
        sigma = self.result.sigma_final

        n_eta = omega.shape[0]
        n_eps = sigma.shape[0]

        pm = self.population_model
        subject_ids = pm.subject_ids()
        subject_models = [(sid, pm.individual_model(sid)) for sid in subject_ids]
        total_rows = sum(len(indiv.subject_events.obs_times) for _, indiv in subject_models)
        advan2_batch_plan = self._prepare_default_advan2_batch_plan(theta, subject_models)

        # --- REP=0: observed data ---
        obs_columns = self._collect_observed_data(theta, omega, subject_models, total_rows)
        all_column_blocks = [obs_columns]

        # --- REP=1..n_replicates ---
        # Pre-generate independent child seeds so parallel replicates are
        # reproducible regardless of scheduling order.
        child_seeds = self.rng.integers(0, 2**31, size=n_replicates).tolist()

        if self.n_parallel == 1 or n_replicates <= 1:
            for rep in range(1, n_replicates + 1):
                rep_rng = np.random.default_rng(child_seeds[rep - 1])
                rep_columns = self._simulate_one_replicate(
                    theta,
                    omega,
                    sigma,
                    n_eta,
                    n_eps,
                    subject_models,
                    total_rows,
                    rep,
                    rep_rng,
                    advan2_batch_plan,
                )
                all_column_blocks.append(rep_columns)
        else:
            n_workers = self.n_parallel if self.n_parallel > 0 else None
            rep_columns_by_rep: dict[int, dict[str, np.ndarray]] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        self._simulate_one_replicate,
                        theta,
                        omega,
                        sigma,
                        n_eta,
                        n_eps,
                        subject_models,
                        total_rows,
                        rep,
                        np.random.default_rng(child_seeds[rep - 1]),
                        advan2_batch_plan,
                    ): rep
                    for rep in range(1, n_replicates + 1)
                }
                for future in as_completed(futures):
                    rep = futures[future]
                    rep_columns_by_rep[rep] = future.result()
            for rep in range(1, n_replicates + 1):
                all_column_blocks.append(rep_columns_by_rep[rep])

        combined_columns = _concat_simulation_columns(all_column_blocks, n_eta)
        simulated_df = _simulation_frame_from_columns(combined_columns, n_eta)
        return SimulationResult(
            simulated_df=simulated_df,
            seed=self.seed,
            n_replicates=n_replicates,
        )

    def _collect_observed_data(
        self,
        theta: np.ndarray,
        omega: np.ndarray,
        subject_models: list[tuple[int, Any]],
        total_rows: int,
    ) -> dict[str, np.ndarray]:
        """
        Collect REP=0 rows (observed dataset with IPRED at post-hoc ETAs or ETA=0).

        Args:
            theta:       Final population THETA estimates.
            omega:       Final OMEGA matrix.
            subject_models: Cached list of (subject ID, IndividualModel).
            total_rows: Total number of observation rows across subjects.

        Returns:
            Column buffers for REP=0 rows.
        """
        pm = self.population_model
        n_eta = omega.shape[0]
        columns = _allocate_simulation_columns(total_rows, n_eta)
        offset = 0
        zero_eta = np.zeros(n_eta)

        for sid, indiv in subject_models:
            # Use post-hoc ETAs if available, else zero
            eta = self.result.post_hoc_etas.get(sid, zero_eta)
            events = indiv.subject_events
            n_obs = len(events.obs_times)
            if n_obs == 0:
                continue
            row_slice = slice(offset, offset + n_obs)

            try:
                ipred, _obs_mask, f, _amounts = indiv._evaluate_predictions(
                    theta, eta, pm.params.sigma, trans=pm.trans, include_amounts=False
                )
            except Exception:
                ipred = np.zeros(len(events.obs_times))
                f = ipred

            columns["ID"][row_slice] = sid
            columns["TIME"][row_slice] = events.obs_times
            columns["DV"][row_slice] = events.obs_dv
            columns["MDV"][row_slice] = events.obs_mdv
            columns["REP"][row_slice] = 0

            ipred_out = np.full(n_obs, np.nan, dtype=float)
            f_out = np.full(n_obs, np.nan, dtype=float)
            n_ipred = min(n_obs, len(ipred))
            n_f = min(n_obs, len(f))
            if n_ipred > 0:
                ipred_out[:n_ipred] = np.asarray(ipred[:n_ipred], dtype=float)
            if n_f > 0:
                f_out[:n_f] = np.asarray(f[:n_f], dtype=float)
            columns["IPRED"][row_slice] = ipred_out
            columns["PRED"][row_slice] = f_out

            for k, eta_val in enumerate(np.asarray(eta, dtype=float)):
                columns[f"ETA{k + 1}"][row_slice] = eta_val

            offset += n_obs

        return columns

    def _simulate_one_replicate(
        self,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        n_eta: int,
        n_eps: int,
        subject_models: list[tuple[int, Any]],
        total_rows: int,
        rep: int,
        rng: np.random.Generator | None = None,
        advan2_batch_plan: tuple[list[DoseEvent], np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate one replicated dataset.

        Args:
            theta:       Population THETA.
            omega:       OMEGA matrix (draws ETAs).
            sigma:       SIGMA matrix (draws EPSs).
            n_eta:       Number of random effects.
            n_eps:       Number of residual random effects.
            subject_models: Cached list of (subject ID, IndividualModel).
            total_rows: Total number of observation rows across subjects.
            rep:         Replicate index (1-based).
            rng:         Random generator for this replicate (default: self.rng).

        Returns:
            Column buffers for one replicate with REP=rep rows.
        """
        if rng is None:
            rng = self.rng
        pm = self.population_model
        columns = _allocate_simulation_columns(total_rows, n_eta)
        theta_seq = list(theta)
        eta_draws = _draw_mvn_batch(rng, omega, n_eta, len(subject_models))
        advan2_batch_ipred = None
        if advan2_batch_plan is not None:
            dose_events, obs_times = advan2_batch_plan
            advan2_batch_ipred = _default_advan2_batch_predictions(
                theta, eta_draws, dose_events, obs_times
            )
        offset = 0

        for subject_idx, ((sid, indiv), eta) in enumerate(
            zip(subject_models, eta_draws, strict=False)
        ):
            events = indiv.subject_events
            n_obs_total = len(events.obs_times)
            if n_obs_total == 0:
                continue
            row_slice = slice(offset, offset + n_obs_total)

            if advan2_batch_ipred is not None:
                ipred = advan2_batch_ipred[subject_idx]
                f = ipred
                amounts = None
            else:
                try:
                    ipred, _obs_mask, f, amounts = indiv._evaluate_predictions(
                        theta,
                        eta,
                        pm.params.sigma,
                        trans=pm.trans,
                        include_amounts=indiv._error_requires_amounts,
                    )
                except Exception:
                    ipred = np.zeros(len(events.obs_times))
                    f = ipred
                    amounts = None

            # Pre-draw all EPS for this subject at once (avoids per-obs MVN overhead)
            all_eps = _draw_mvn_batch(rng, sigma, n_eps, n_obs_total)

            ipred_out = np.zeros(n_obs_total, dtype=float)
            f_out = np.zeros(n_obs_total, dtype=float)
            n_ipred = min(n_obs_total, len(ipred))
            n_f = min(n_obs_total, len(f))
            if n_ipred > 0:
                ipred_out[:n_ipred] = np.asarray(ipred[:n_ipred], dtype=float)
            if n_f > 0:
                f_out[:n_f] = np.asarray(f[:n_f], dtype=float)

            if indiv.error_callable is None:
                sim_dv = ipred_out.copy()
                if n_eps > 0 and sigma.size > 0:
                    sim_dv += all_eps[:, 0]
            else:
                sim_dv = indiv.simulate_error_predictions_fast(theta, f_out, all_eps)
                if sim_dv is None:
                    eta_seq = list(eta)
                    covariates = indiv._observation_covariates
                    amount_rows = None
                    if amounts is not None:
                        amount_array = np.asarray(amounts, dtype=float)
                        if amount_array.ndim == 1:
                            amount_array = amount_array[:, None]
                        amount_rows = amount_array.tolist()
                    observed_dv = np.where(np.isfinite(events.obs_dv), events.obs_dv, 0.0)
                    sim_dv = np.empty(n_obs_total, dtype=float)
                    for i, t in enumerate(events.obs_times):
                        eps_seq = all_eps[i].tolist()
                        ipred_i = float(ipred_out[i])
                        f_i = float(f_out[i])
                        try:
                            error_out = indiv._call_error_model_prepared(
                                theta_seq=theta_seq,
                                eta_seq=eta_seq,
                                eps_seq=eps_seq,
                                f_i=f_i,
                                ipred_i=ipred_i,
                                y_obs=float(observed_dv[i]),
                                t_i=float(t),
                                a_i=None if amount_rows is None else amount_rows[i],
                                covariates=covariates[i],
                                sigma=sigma,
                            )
                            sim_dv[i] = indiv._extract_error_prediction(error_out, f_i)
                        except Exception:
                            sigma_diag = float(np.sqrt(sigma[0, 0])) if sigma.size > 0 else 0.1
                            sim_dv[i] = (
                                ipred_i + float(all_eps[i, 0]) * sigma_diag
                                if n_eps > 0
                                else ipred_i
                            )

            columns["ID"][row_slice] = sid
            columns["TIME"][row_slice] = events.obs_times
            columns["DV"][row_slice] = np.maximum(sim_dv, 0.0)
            columns["IPRED"][row_slice] = ipred_out
            columns["PRED"][row_slice] = f_out
            columns["MDV"][row_slice] = events.obs_mdv
            columns["REP"][row_slice] = rep
            for k, eta_val in enumerate(np.asarray(eta, dtype=float)):
                columns[f"ETA{k + 1}"][row_slice] = eta_val

            offset += n_obs_total

        return columns

    def _prepare_default_advan2_batch_plan(
        self,
        theta: np.ndarray,
        subject_models: list[tuple[int, Any]],
    ) -> tuple[list[DoseEvent], np.ndarray] | None:
        """Return shared design metadata for the narrow default ADVAN2 fast path."""
        pm = self.population_model
        if (
            getattr(pm.pk_subroutine, "advan", None) != 2
            or pm.trans != 2
            or pm.pk_callable is not None
            or pm.error_callable is not None
            or pm.des_callable is not None
            or len(theta) < 3
            or len(subject_models) == 0
        ):
            return None

        ref_events = subject_models[0][1].subject_events
        ref_times = np.asarray(ref_events.obs_times, dtype=float)
        ref_doses = tuple(_dose_signature(dose) for dose in ref_events.dose_events)
        if any(
            (not dose.is_bolus) or dose.reset or dose.compartment != 1 or dose.ss
            for dose in ref_events.dose_events
        ):
            return None

        for _sid, indiv in subject_models:
            events = indiv.subject_events
            if (
                indiv.pk_callable is not None
                or indiv.error_callable is not None
                or indiv.des_callable is not None
                or indiv.occasion_indices is not None
                or getattr(indiv, "_has_time_varying_covariates", False)
                or not np.array_equal(np.asarray(events.obs_times, dtype=float), ref_times)
                or tuple(_dose_signature(dose) for dose in events.dose_events) != ref_doses
            ):
                return None

        return ref_events.dose_events, ref_times

    def simulate_new_design(
        self,
        dosing_df: pd.DataFrame,
        obs_times: np.ndarray,
        n_subjects: int,
        n_replicates: int = 1,
    ) -> SimulationResult:
        """
        Simulate with a new dosing design (not the observed dataset).

        Builds a synthetic population dataset from ``dosing_df`` and
        ``obs_times`` for ``n_subjects`` virtual subjects, assembles a
        temporary :class:`PopulationModel`, and delegates to the existing
        :meth:`simulate` path so that $PK, $ERROR, covariate handling,
        TRANS, and BLQ logic are all applied correctly.

        Args:
            dosing_df:    DataFrame with dose records.  Must contain at least
                          ``TIME`` and ``AMT``; ``EVID`` defaults to 1 and
                          ``CMT`` defaults to 1 when absent.  The ``ID``
                          column is ignored — virtual subjects are numbered
                          1 … ``n_subjects``.
            obs_times:    1-D array of observation times (relative to t=0).
            n_subjects:   Number of virtual subjects to simulate.
            n_replicates: Number of Monte Carlo replicates.

        Returns:
            SimulationResult whose ``simulated_df`` contains REP=0 (all-NaN
            DV placeholder) and REP=1 … n_replicates simulated datasets.
        """
        from openpkpd.data.dataset import NONMEMDataset
        from openpkpd.model.population import PopulationModel

        pm = self.population_model
        obs_times = np.asarray(obs_times, dtype=float)

        # Build a synthetic NONMEM-style DataFrame for all virtual subjects
        frames: list[pd.DataFrame] = []
        dose_template = dosing_df.copy()
        if "EVID" not in dose_template.columns:
            dose_template["EVID"] = 1
        if "CMT" not in dose_template.columns:
            dose_template["CMT"] = 1
        if "MDV" not in dose_template.columns:
            dose_template["MDV"] = 1

        for sid in range(1, n_subjects + 1):
            # Dose rows
            dose_rows = dose_template.copy()
            dose_rows["ID"] = sid

            # Observation rows
            obs_rows = pd.DataFrame(
                {
                    "ID": sid,
                    "TIME": obs_times,
                    "AMT": 0.0,
                    "DV": np.nan,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                }
            )
            frames.append(pd.concat([dose_rows, obs_rows], ignore_index=True))

        design_df = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
            .reset_index(drop=True)
        )

        new_dataset = NONMEMDataset(df=design_df)

        # Assemble a temporary PopulationModel with the new dataset
        new_pop = PopulationModel(
            dataset=new_dataset,
            pk_subroutine=pm.pk_subroutine,
            params=pm.params,
            pk_callable=pm.pk_callable,
            error_callable=pm.error_callable,
            des_callable=pm.des_callable,
            trans=pm.trans,
            advan=pm.advan,
            covariate_columns=list(pm.covariate_columns),
            blq_method=pm.blq_method,
        )

        # Delegate to the existing simulate() path (handles $PK, $ERROR, TRANS, …)
        new_engine = SimulationEngine(new_pop, self.result, seed=self.seed)
        new_engine.rng = self.rng  # share the RNG state for reproducibility
        return new_engine.simulate(n_replicates=n_replicates)


def _draw_mvn(rng: np.random.Generator, cov: np.ndarray, n: int) -> np.ndarray:
    """
    Draw from MVN(0, cov) safely, handling near-singular covariance matrices.

    Args:
        rng:  NumPy random Generator.
        cov:  Covariance matrix, shape (n, n).
        n:    Dimension (must equal cov.shape[0]).

    Returns:
        Sample vector of shape (n,).
    """
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([rng.normal(0.0, float(np.sqrt(max(cov[0, 0], 0.0))))])
    try:
        return rng.multivariate_normal(np.zeros(n), cov)
    except np.linalg.LinAlgError:
        # Near-singular: use diagonal approximation
        stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return rng.normal(0.0, 1.0, n) * stds


def _draw_mvn_batch(rng: np.random.Generator, cov: np.ndarray, n: int, size: int) -> np.ndarray:
    """
    Draw ``size`` samples from MVN(0, cov), returning shape (size, n).

    Batching avoids per-sample Cholesky decomposition overhead.
    For n==0 returns shape (size, 0); for size==0 returns shape (0, n).
    """
    if size == 0:
        return np.zeros((0, max(n, 1)))
    if n == 0:
        return np.zeros((size, 0))
    if n == 1:
        std = float(np.sqrt(max(cov[0, 0], 0.0)))
        return rng.normal(0.0, std, size=(size, 1))
    try:
        return rng.multivariate_normal(np.zeros(n), cov, size=size)
    except np.linalg.LinAlgError:
        stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return rng.normal(0.0, 1.0, size=(size, n)) * stds


def _dose_signature(dose: DoseEvent) -> tuple[float, float, float, float, int, bool, float, bool]:
    return (
        float(dose.time),
        float(dose.amount),
        float(dose.rate),
        float(dose.duration),
        int(dose.compartment),
        bool(dose.ss),
        float(dose.ii),
        bool(dose.reset),
    )


def _default_advan2_batch_predictions(
    theta: np.ndarray,
    eta_draws: np.ndarray,
    dose_events: list[DoseEvent],
    obs_times: np.ndarray,
) -> np.ndarray:
    """Vectorized default ADVAN2/TRANS2 IPRED for shared-design simulation batches."""
    eta_arr = np.asarray(eta_draws, dtype=float)
    if eta_arr.ndim == 1:
        eta_arr = eta_arr[None, :]
    obs_arr = np.asarray(obs_times, dtype=float)
    n_subjects = len(eta_arr)
    n_obs = len(obs_arr)
    if n_subjects == 0 or n_obs == 0:
        return np.zeros((n_subjects, n_obs), dtype=float)

    ka = np.full(n_subjects, float(theta[0]), dtype=float)
    cl = np.full(n_subjects, float(theta[1]), dtype=float)
    v = np.full(n_subjects, float(theta[2]), dtype=float)
    if eta_arr.shape[1] > 0:
        ka *= np.exp(eta_arr[:, 0])
    if eta_arr.shape[1] > 1:
        cl *= np.exp(eta_arr[:, 1])
    if eta_arr.shape[1] > 2:
        v *= np.exp(eta_arr[:, 2])

    k = cl / v
    denom = ka - k
    limit_mask = np.abs(denom) < 1e-6
    ipred = np.zeros((n_subjects, n_obs), dtype=float)

    for dose in dose_events:
        if dose.reset or not dose.is_bolus:
            continue
        dt = obs_arr - float(dose.time)
        positive_idx = np.flatnonzero(dt > 0.0)
        if len(positive_idx) == 0:
            continue
        dt_pos = dt[positive_idx]
        exp_k = np.exp(-k[:, None] * dt_pos[None, :])
        exp_ka = np.exp(-ka[:, None] * dt_pos[None, :])
        contrib = np.empty((n_subjects, len(positive_idx)), dtype=float)
        if np.any(limit_mask):
            contrib[limit_mask] = (
                float(dose.amount)
                * ka[limit_mask, None]
                * dt_pos[None, :]
                * exp_k[limit_mask]
                / v[limit_mask, None]
            )
        if np.any(~limit_mask):
            contrib[~limit_mask] = (
                float(dose.amount)
                * (ka[~limit_mask] / denom[~limit_mask])[:, None]
                * (exp_k[~limit_mask] - exp_ka[~limit_mask])
                / v[~limit_mask, None]
            )
        ipred[:, positive_idx] += contrib

    return ipred


def _allocate_simulation_columns(n_rows: int, n_eta: int) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {
        "ID": np.empty(n_rows, dtype=int),
        "TIME": np.empty(n_rows, dtype=float),
        "DV": np.empty(n_rows, dtype=float),
        "IPRED": np.empty(n_rows, dtype=float),
        "PRED": np.empty(n_rows, dtype=float),
        "MDV": np.empty(n_rows, dtype=int),
        "REP": np.empty(n_rows, dtype=int),
    }
    for k in range(n_eta):
        columns[f"ETA{k + 1}"] = np.empty(n_rows, dtype=float)
    return columns


def _simulation_frame_from_columns(columns: dict[str, np.ndarray], n_eta: int) -> pd.DataFrame:
    ordered_columns = ["ID", "TIME", "DV", "IPRED", "PRED", "MDV", "REP"] + [
        f"ETA{k + 1}" for k in range(n_eta)
    ]
    return pd.DataFrame({name: columns[name] for name in ordered_columns}, columns=ordered_columns)


def _concat_simulation_columns(
    column_blocks: list[dict[str, np.ndarray]],
    n_eta: int,
) -> dict[str, np.ndarray]:
    ordered_columns = ["ID", "TIME", "DV", "IPRED", "PRED", "MDV", "REP"] + [
        f"ETA{k + 1}" for k in range(n_eta)
    ]
    return {
        name: np.concatenate([block[name] for block in column_blocks]) for name in ordered_columns
    }
