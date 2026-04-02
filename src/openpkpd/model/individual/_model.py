"""IndividualModel — composes all four mixins into the public class."""
from __future__ import annotations

import inspect
import logging
from typing import Any
from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import SubjectEvents
from openpkpd.pk.base import PKSubroutine
from openpkpd.utils.constants import BLQMethod
from openpkpd.utils.errors import PKError

from openpkpd.model.individual._pk_solution import PKSolutionMixin
from openpkpd.model.individual._observation import ObservationModelMixin
from openpkpd.model.individual._likelihood import LikelihoodMixin
from openpkpd.model.individual._derivatives import DerivativesMixin

logger = logging.getLogger(__name__)


class IndividualModel(
    PKSolutionMixin,
    ObservationModelMixin,
    LikelihoodMixin,
    DerivativesMixin,
):
    """
    Evaluates the individual log-likelihood and OFV contribution.

    Holds references to the population-level PK model and compiled
    callables so it can be called repeatedly during inner-loop
    optimization (EBE estimation).

    Implemented as a composition of four mixins:
      - PKSolutionMixin      (native ODE dispatch, CVODES sensitivities)
      - ObservationModelMixin (error model inference, predictions)
      - LikelihoodMixin      (log-likelihood, eta objective)
      - DerivativesMixin     (Jacobians, Hessians, supports_* capabilities)
    """

    def __init__(
        self,
        subject_events: SubjectEvents,
        pk_subroutine: PKSubroutine,
        pk_callable: Callable | None,
        error_callable: Callable | None,
        n_eps: int = 1,
        blq_method: str = BLQMethod.M1,
        lloq: float | np.ndarray | None = None,
        des_callable: Callable | None = None,
        occasion_indices: np.ndarray | None = None,
    ) -> None:
        self.subject_events = subject_events
        self.pk_subroutine = pk_subroutine
        self.pk_callable = pk_callable
        self.error_callable = error_callable
        self.des_callable = des_callable
        self.n_eps = n_eps
        self.blq_method: str = blq_method
        self.lloq: float | np.ndarray | None = lloq
        self.occasion_indices: np.ndarray | None = occasion_indices  # B1
        self._cached_ipred: np.ndarray | None = None
        self._error_call_mode: str = "unknown"
        self._compiled_error_raw = getattr(error_callable, "_call_raw", None)
        self._error_requires_amounts = self._infer_error_requires_amounts(error_callable)
        self._obs_mask = subject_events.observation_mask()
        self._dose_events = subject_events.dose_events
        self._base_covariates = subject_events.covariate_at(0.0) if pk_callable is not None else {}
        self._has_time_varying_covariates = (
            subject_events.covariate_df is not None
            and pk_callable is not None
            and des_callable is not None
        )
        self._covariate_change_times = (
            subject_events.covariate_change_times()
            if subject_events.covariate_df is not None
            else []
        )
        self._derivative_kernel_cache: dict[int, Any | None] = {}
        self._pk_param_transformers: dict[int, Callable[[dict[str, float]], dict[str, float]]] = {}
        self._observation_covariates = tuple(
            subject_events.observation_covariates_at(i)
            for i in range(len(subject_events.obs_times))
        )
        self._observation_dvid = self._build_observation_dvid()
        if occasion_indices is not None:
            if len(occasion_indices) != len(subject_events.obs_times):
                raise ValueError(
                    f"occasion_indices length ({len(occasion_indices)}) must match "
                    f"obs_times length ({len(subject_events.obs_times)})"
                )
            self._unique_occasions = np.unique(occasion_indices)
        else:
            self._unique_occasions = None
        solve_signature = inspect.signature(pk_subroutine.solve)
        self._solve_supports_return_amounts = "return_amounts" in solve_signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in solve_signature.parameters.values()
        )
        self._eta_penalty_cache_key: tuple[bytes, tuple[int, ...], int] | None = None
        self._eta_penalty_precision: np.ndarray | None = None
        self._eta_penalty_block_size: int | None = None
        self._eps_basis_vectors: tuple[tuple[float, ...], ...] = tuple(
            tuple(1.0 if i == j else 0.0 for i in range(self.n_eps)) for j in range(self.n_eps)
        )
        self._common_error_model = self._infer_common_error_model(error_callable, self.n_eps)
        self._user_ode_template: tuple[tuple[str, ...], _NativeOdeTemplate | None] | None = None
        self._native_ode_contract = self._build_native_ode_contract()

    def __getstate__(self) -> dict[str, Any]:
        """Drop rebuildable caches so worker-process pickling stays robust."""
        state = self.__dict__.copy()
        state["_pk_param_transformers"] = {}
        state["_derivative_kernel_cache"] = {}
        state["_cached_ipred"] = None
        state["_eta_penalty_cache_key"] = None
        state["_eta_penalty_precision"] = None
        state["_eta_penalty_block_size"] = None
        # User ODE template contains Python closures that can't be pickled.
        # Preserve only the param_keys so __setstate__ can rebuild eagerly in
        # the worker before the first prediction call (C-06).
        if self._user_ode_template is not None:
            cached_keys, _ = self._user_ode_template
            state["_user_ode_param_keys"] = cached_keys
        else:
            state["_user_ode_param_keys"] = None
        state["_user_ode_template"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_pk_param_transformers" not in self.__dict__:
            self._pk_param_transformers = {}
        if "_derivative_kernel_cache" not in self.__dict__:
            self._derivative_kernel_cache = {}
        if "_user_ode_template" not in self.__dict__:
            self._user_ode_template = None
        # C-06: eagerly rebuild the user ODE template (triggering Numba JIT here,
        # at worker startup) rather than lazily on the first prediction call.
        param_keys: tuple[str, ...] | None = self.__dict__.pop("_user_ode_param_keys", None)
        if param_keys is not None:
            try:
                dummy_params = {k: 1.0 for k in param_keys}
                self._try_build_user_ode_template(dummy_params)
            except Exception:  # noqa: BLE001
                # Compilation failure in the worker is non-fatal; the first
                # prediction call will fall back to the scipy integrator.
                self._user_ode_template = None

    @staticmethod
    def _infer_error_requires_amounts(error_callable: Callable | None) -> bool:
        if error_callable is None:
            return False

        uses_amounts = getattr(error_callable, "_uses_amounts", None)
        if uses_amounts is not None:
            return bool(uses_amounts)

        try:
            signature = inspect.signature(error_callable)
        except (TypeError, ValueError):
            return True

        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return True
        return "a" in signature.parameters

    def _build_observation_dvid(self) -> np.ndarray | None:
        if len(self._observation_covariates) == 0:
            return np.array([], dtype=float)
        dvid_values: list[float] = []
        for covariates in self._observation_covariates:
            if "DVID" not in covariates:
                return None
            try:
                dvid_values.append(float(covariates["DVID"]))
            except (TypeError, ValueError):
                return None
        return np.asarray(dvid_values, dtype=float)


