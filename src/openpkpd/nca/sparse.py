"""
Sparse sampling NCA via model-based dense-profile reconstruction.

For subjects with only a few PK samples (e.g. 2–5 time points), classical
NCA is unreliable. This module uses a fitted population model to:

1. Estimate each subject's individual parameters (post-hoc ETAs) by
   minimising the individual objective function over the sparse observations.
2. Predict a dense concentration profile at a fine time grid.
3. Apply standard NCA to the dense predicted profile.

This approach is sometimes called "model-informed sparse data NCA" or
"model-based NCA" and is suitable for preclinical sparse-sampling designs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from openpkpd.nca.nca import NCAEngine, NCAParameters


class SparseNCAEngine:
    """
    Model-based sparse sampling NCA engine.

    Reconstructs a dense concentration–time profile for each subject by
    fitting the population model to sparse observations, then delegates
    to the standard :class:`~openpkpd.nca.NCAEngine` for parameter
    computation.

    Parameters
    ----------
    population_model:
        A fitted :class:`~openpkpd.model.population.PopulationModel`
        (must have ``theta``, ``omega``, and ``sigma`` attributes set).
    dense_times:
        Time grid for dense profile prediction.  If ``None``, a 200-point
        grid from 0 to the last sparse observation time is used.
    """

    def __init__(
        self,
        population_model: Any,
        dense_times: np.ndarray | None = None,
    ) -> None:
        self.population_model = population_model
        self.dense_times = dense_times
        self._nca = NCAEngine()

    # ── Single-subject ────────────────────────────────────────────────────

    def compute_subject(
        self,
        subject_id: int | str,
        sparse_times: np.ndarray,
        sparse_conc: np.ndarray,
        dose: float,
        route: str = "IV",
        eta0: np.ndarray | None = None,
    ) -> NCAParameters:
        """
        Compute NCA parameters for one subject from sparse observations.

        Algorithm
        ---------
        1. Optimise post-hoc ETAs over the sparse data using the
           individual log-likelihood from the population model.
        2. Predict a dense profile at *dense_times* using optimal ETAs.
        3. Pass the dense profile to :meth:`NCAEngine.compute_subject`.

        Parameters
        ----------
        subject_id:
            Subject identifier (used in the returned NCAParameters).
        sparse_times:
            Observed sample times (length ≥ 2).
        sparse_conc:
            Observed concentrations at *sparse_times*.
        dose:
            Administered dose.
        route:
            Administration route: ``'IV'``, ``'oral'``, or ``'infusion'``.
        eta0:
            Initial guess for ETAs.  Defaults to the zero vector.

        Returns
        -------
        NCAParameters
            NCA parameters computed from the reconstructed dense profile.
        """
        sparse_times = np.asarray(sparse_times, dtype=float)
        sparse_conc = np.asarray(sparse_conc, dtype=float)

        n_eta = self.population_model.omega.shape[0]
        if eta0 is None:
            eta0 = np.zeros(n_eta)

        # Dense time grid
        t_last = float(np.max(sparse_times))
        dense_t = (
            self.dense_times if self.dense_times is not None else np.linspace(0.0, t_last, 200)
        )

        # Optimise ETAs against sparse observations
        def _neg_ll(eta: np.ndarray) -> float:
            try:
                ipred = self._predict(eta, sparse_times, dose=dose)
                sigma_diag = np.sqrt(np.diag(self.population_model.sigma))
                residuals = (sparse_conc - ipred) / (sigma_diag[0] * ipred + 1e-12)
                data_ll = 0.5 * float(np.sum(residuals**2))
                # Penalty: 0.5 * eta' * Omega^{-1} * eta
                omega_inv = np.linalg.pinv(self.population_model.omega)
                eta_penalty = 0.5 * float(eta @ omega_inv @ eta)
                return data_ll + eta_penalty
            except Exception:
                return 1e9

        res = minimize(_neg_ll, eta0, method="L-BFGS-B", options={"maxiter": 200})
        best_eta = res.x

        # Predict dense profile
        dense_conc = self._predict(best_eta, dense_t, dose=dose)
        dense_conc = np.maximum(dense_conc, 0.0)

        return self._nca.compute_subject(
            times=dense_t,
            conc=dense_conc,
            dose=dose,
            subject_id=subject_id,
            route=route,
        )

    # ── Dataset ───────────────────────────────────────────────────────────

    def compute_dataset(
        self,
        sparse_df: pd.DataFrame,
        dose: float,
        route: str = "IV",
        id_col: str = "ID",
        time_col: str = "TIME",
        conc_col: str = "DV",
    ) -> pd.DataFrame:
        """
        Compute NCA parameters for all subjects in a sparse dataset.

        Parameters
        ----------
        sparse_df:
            Long-format DataFrame with columns *id_col*, *time_col*,
            *conc_col*.
        dose:
            Common dose for all subjects (dose per record not yet
            supported — use :meth:`compute_subject` for heterogeneous
            dosing).
        route:
            Administration route.
        id_col, time_col, conc_col:
            Column names in *sparse_df*.

        Returns
        -------
        pd.DataFrame
            One row per subject with NCA parameters as columns.
        """
        records = []
        for subj_id, grp in sparse_df.groupby(id_col):
            grp = grp.sort_values(time_col)
            params = self.compute_subject(
                subject_id=subj_id,
                sparse_times=grp[time_col].to_numpy(),
                sparse_conc=grp[conc_col].to_numpy(),
                dose=dose,
                route=route,
            )
            records.append(params.to_dict())
        return pd.DataFrame(records)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _predict(self, eta: np.ndarray, times: np.ndarray, dose: float = 1.0) -> np.ndarray:
        """
        Call the population model to get IPRED at *times* for given *eta*.

        Tries the ``IndividualModel.evaluate()`` interface first; falls
        back to a simple analytical 1-cmt solution when the model does
        not expose that interface.
        """
        pm = self.population_model

        # Try the standard individual-model API
        try:
            ind_model = pm.get_individual_model(eta=eta)
            ipred = ind_model.evaluate(times)
            return np.asarray(ipred, dtype=float)
        except (AttributeError, TypeError):
            pass

        # Fallback: analytical 1-cmt IV bolus using theta[0]=CL, theta[1]=V
        theta = getattr(pm, "theta", np.array([1.0, 10.0]))
        cl = float(theta[0]) * np.exp(eta[0] if len(eta) > 0 else 0.0)
        v = float(theta[1]) * np.exp(eta[1] if len(eta) > 1 else 0.0)
        ke = cl / v
        return dose / v * np.exp(-ke * times)
