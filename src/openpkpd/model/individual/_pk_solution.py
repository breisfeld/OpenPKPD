"""PKSolutionMixin — native ODE dispatch, template selection, sensitivity probes."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np

from openpkpd.model.individual._base import (
    _NativeOdeTemplate,
    _NATIVE_ODE_TEMPLATES,
    _NATIVE_ELIGIBLE_ADVANS,
    _NATIVE_SUPPORTED_ERROR_MODELS,
    _apply_alag,
)
from openpkpd.pk.base import PKSolution

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PKSolutionMixin:
    """Mixin providing native ODE dispatch and CVODES sensitivity methods."""

    def _try_build_user_ode_template(
        self, pk_params: dict[str, float]
    ) -> "_NativeOdeTemplate | None":
        """Build (and cache) a native ODE template from the user's $DES callable.

        Only activates for ADVAN6 models whose ``des_callable`` is a
        :class:`~openpkpd.parser.code_compiler.CompiledDESCallable`.  Returns
        ``None`` if Numba is unavailable, if the DES cannot be compiled, or if
        no volume parameter (``V``, ``V1``, ``V2``, ``V3``) is present in the
        ``$PK`` block output.  The result is cached keyed on the sorted set of
        ``pk_params`` keys so that recompilation is avoided on repeated calls.
        """
        from openpkpd.parser.code_compiler import CompiledDESCallable  # noqa: PLC0415
        from openpkpd.pk.ode.advan6 import ADVAN6  # noqa: PLC0415

        if not isinstance(self.pk_subroutine, ADVAN6):
            return None
        if not isinstance(self.des_callable, CompiledDESCallable):
            return None

        param_keys = tuple(sorted(pk_params.keys()))
        if self._user_ode_template is not None:
            cached_keys, cached_tmpl = self._user_ode_template
            if cached_keys == param_keys:
                return cached_tmpl

        vol_name = next((v for v in ("V", "V1", "V2", "V3") if v in pk_params), None)
        if vol_name is None:
            self._user_ode_template = (param_keys, None)
            return None

        n_states = int(self.pk_subroutine.n_compartments)
        rtol = float(getattr(self.pk_subroutine, "rtol", 1e-6))
        atol = float(getattr(self.pk_subroutine, "atol", 1e-8))
        output_cmt_idx = int(getattr(self.pk_subroutine, "output_compartment", 1)) - 1

        result = self.des_callable.as_multidose_probe(param_keys, n_states, rtol, atol)
        if result is None:
            self._user_ode_template = (param_keys, None)
            return None

        state_probe, sens_probe, inf_state_probe, inf_sens_probe = result
        tmpl = _NativeOdeTemplate(
            name="user_ode",
            required_names=param_keys,
            n_states=n_states,
            output_cmt_idx=output_cmt_idx,
            vol_param_name=vol_name,
            state_probe_fn=state_probe,
            sens_probe_fn=sens_probe,
            infusion_state_probe_fn=inf_state_probe,
            infusion_sens_probe_fn=inf_sens_probe,
            eligible_advans=frozenset({6}),
        )
        self._user_ode_template = (param_keys, tmpl)
        return tmpl

    def _iter_templates(
        self, pk_params: dict[str, float]
    ) -> "list[_NativeOdeTemplate]":
        """Return the ordered list of templates to try for probe dispatch.

        Prepends the user-compiled ODE template (when available) so it takes
        precedence over the built-in Rust templates for ADVAN6 models.
        """
        user_tmpl = self._try_build_user_ode_template(pk_params)
        if user_tmpl is not None:
            return [user_tmpl, *_NATIVE_ODE_TEMPLATES]
        return _NATIVE_ODE_TEMPLATES

    def _select_template(
        self,
        pk_params: dict[str, float],
        contract: "dict[str, Any]",
        *,
        probe_attr: str = "sens_probe_fn",
        exclude_pkpd: bool = True,
        check_pcmt: bool = False,
    ) -> "_NativeOdeTemplate | None":
        """Return the first template that satisfies all eligibility criteria.

        Args:
            pk_params:    Output of the PK callable for the current eta/theta.
            contract:     Native ODE contract dict (contains ``n_compartments``
                          and ``advan``).
            probe_attr:   Name of the probe-function slot to require non-None
                          (``"state_probe_fn"`` or ``"sens_probe_fn"``).
            exclude_pkpd: When *True*, skip templates where ``is_pkpd`` is
                          True (used for pure-PK sensitivity probes).
            check_pcmt:   When *True*, enforce that the ``PCMT`` parameter in
                          *pk_params* matches the template's output compartment
                          index (used for the prediction / state probe path).

        Returns:
            The first matching :class:`_NativeOdeTemplate`, or *None*.
        """
        n_cmt = contract.get("n_compartments", -1)
        contract_advan = contract.get("advan", 6)
        for tmpl in self._iter_templates(pk_params):
            if getattr(tmpl, probe_attr) is None:
                continue
            if exclude_pkpd and tmpl.is_pkpd:
                continue
            if tmpl.n_states != n_cmt:
                continue
            if tmpl.eligible_advans and contract_advan not in tmpl.eligible_advans:
                continue
            if any(name not in pk_params for name in tmpl.required_names):
                continue
            if check_pcmt and tmpl.is_pkpd and "PCMT" in pk_params:
                if int(pk_params["PCMT"]) != tmpl.output_cmt_idx + 1:
                    continue
            return tmpl
        return None

    def _build_native_ode_contract(self) -> dict[str, Any] | None:
        # Gate 1: need either a native Rust template or a user-compiled DES callable.
        from openpkpd.parser.code_compiler import CompiledDESCallable  # noqa: PLC0415
        from openpkpd.pk.ode.advan6 import ADVAN6  # noqa: PLC0415

        _has_native_tmpl = any(t.state_probe_fn is not None for t in _NATIVE_ODE_TEMPLATES)
        _has_user_des = isinstance(self.pk_subroutine, ADVAN6) and isinstance(
            self.des_callable, CompiledDESCallable
        )
        if not _has_native_tmpl and not _has_user_des:
            return None
        if getattr(self.pk_subroutine, "advan", None) not in _NATIVE_ELIGIBLE_ADVANS:
            return None
        if (
            self._common_error_model is None
            or self._common_error_model[0] not in _NATIVE_SUPPORTED_ERROR_MODELS
        ):
            return None
        if self.occasion_indices is not None:
            return None

        cov_df = self.subject_events.covariate_df
        if cov_df is not None:
            for col in cov_df.columns:
                col_upper = str(col).upper()
                if col_upper in {"TIME", "DVID"}:
                    continue
                # Only check numeric columns for time-constancy.  Non-numeric
                # (categorical/string) columns are covariate codes that map to
                # numeric values via the user's $PK callable; they cannot be
                # time-varying in the continuous sense and should not disable
                # the native path.
                try:
                    if not np.issubdtype(cov_df[col].dtype, np.number):
                        continue
                    if cov_df[col].dropna().nunique() > 1:
                        return None
                except (TypeError, AttributeError) as _cov_e:
                    logger.warning(
                        "IndividualModel %s failed at covariate-constancy check: %s",
                        getattr(self, "subject_id", "?"), _cov_e,
                    )
                    return None  # cannot verify constancy — stay conservative

        # Doses must be into compartment 1 (the central compartment for IV
        # models; the Rust ODE probes do not model an absorption depot).
        if len(self._dose_events) == 0:
            return None
        for dose in self._dose_events:
            if int(dose.compartment) != 1:
                return None
            # Infusions are supported for IV templates; reject negative rates
            # (RATE=-1 duration-based) that were not normalised by event_processor.
            if dose.rate < 0.0:
                return None

        sorted_doses = sorted(self._dose_events, key=lambda d: float(d.time))
        dose_times = [float(d.time) for d in sorted_doses]
        dose_amts = [float(d.amount) for d in sorted_doses]
        # rate == 0.0 → bolus; rate > 0.0 → constant-rate infusion
        dose_rates = [float(d.rate) for d in sorted_doses]
        dose_compartments = [int(d.compartment) for d in sorted_doses]
        has_infusion = any(r > 0.0 for r in dose_rates)

        obs_times = np.asarray(self.subject_events.obs_times, dtype=float)
        if len(obs_times) == 0 or np.any(np.diff(obs_times) < 0.0):
            return None

        # Template matching is deferred to probe time (when pk_params are known).
        # We store the common arrays here so they are not recomputed each call.
        return {
            "dose_amount": dose_amts[0],  # backward-compat scalar
            "dose_times": dose_times,
            "dose_amts": dose_amts,
            "dose_rates": dose_rates,
            "dose_compartments": dose_compartments,
            "has_infusion": has_infusion,
            "obs_times": obs_times.copy(),
            "obs_times_list": obs_times.tolist(),
            "n_compartments": int(getattr(self.pk_subroutine, "n_compartments", 4)),
            "is_pkpd": self._common_error_model[0] == "mixed_pkpd_dvid_theta",
            "advan": int(getattr(self.pk_subroutine, "advan", 6)),
        }

    def _try_native_ode_probe(
        self,
        pk_params: dict[str, float],
        obs_times: np.ndarray,
    ) -> PKSolution | None:
        contract = self._native_ode_contract
        if contract is None:
            return None
        if (
            len(obs_times) != len(contract["obs_times"])
            or not np.array_equal(obs_times, contract["obs_times"])
        ):
            return None

        # Find the first template (most specific) whose required names are all
        # present in pk_params and whose state probe is available.
        # n_states must equal n_compartments to prevent a model with more
        # parameters from incorrectly matching a simpler template whose
        # required_names happen to be a subset of the model's pk_params.
        template = self._select_template(
            pk_params, contract,
            probe_attr="state_probe_fn",
            exclude_pkpd=False,
            check_pcmt=True,
        )
        if template is None:
            return None

        required = template.required_names
        theta = [float(pk_params[name]) for name in required]
        dose_times = _apply_alag(
            contract["dose_times"], contract["dose_compartments"], pk_params
        )
        dose_amts = contract["dose_amts"]
        dose_rates = contract["dose_rates"]
        has_infusion = contract["has_infusion"]

        try:
            if has_infusion:
                # Infusion path: requires an infusion-aware probe.
                # Fall back to Python if this template doesn't have one.
                if template.infusion_state_probe_fn is None:
                    return None
                amounts_raw = np.asarray(
                    template.infusion_state_probe_fn(
                        contract["obs_times_list"], dose_times, dose_amts, dose_rates, theta
                    ),
                    dtype=float,
                )
            else:
                amounts_raw = np.asarray(
                    template.state_probe_fn(
                        contract["obs_times_list"], dose_times, dose_amts, theta
                    ),
                    dtype=float,
                )
        except Exception as _ode_probe_e:
            logger.warning(
                "IndividualModel %s failed at native ODE probe: %s",
                getattr(self, "subject_id", "?"), _ode_probe_e,
            )
            return None

        n_states = template.n_states
        if amounts_raw.shape != (len(obs_times), n_states):
            return None

        n_comp = contract["n_compartments"]
        amounts = np.zeros((len(obs_times), max(n_comp, n_states)), dtype=float)
        amounts[:, :n_states] = amounts_raw

        V = float(pk_params[template.vol_param_name])
        if V <= 0:
            raise ValueError(
                f"Volume parameter {template.vol_param_name}={V:.6g} must be > 0 "
                f"(subject {getattr(self, 'subject_id', '?')})"
            )
        ipred = amounts_raw[:, template.output_cmt_idx] / V
        return PKSolution(times=obs_times.copy(), amounts=amounts, ipred=ipred, f=ipred.copy())

    def _try_native_pk_backend(
        self,
        pk_params: dict[str, float],
        obs_times: np.ndarray,
    ) -> PKSolution | None:
        """Try any supported native PK backend for this model shape.

        This is the generic Python-side seam. Individual native kernels can
        stay contract-specific underneath, but callers should not need to know
        which dataset or prototype first motivated the native path.
        """
        return self._try_native_ode_probe(pk_params, obs_times)

    def native_advan6_prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        obs_mask: np.ndarray,
        n_eta: int,
        eps: float = 1e-5,
    ) -> np.ndarray | None:
        """Compute G_i = ∂IPRED/∂η using the CVODES forward sensitivity probe.

        Replaces the standard ``n_eta`` full-ODE finite-difference calls in
        ``_compute_G_i`` with **one** Rust CVODES sensitivity integration plus
        ``n_eta`` cheap pk_callable evaluations.

        Algorithm (chain rule):
          G_i[:, k] = Σ_j  (∂F/∂ODE_param_j)  ×  (∂ODE_param_j/∂η_k)

        where:
          - ∂F/∂ODE_param_j  — from the CVODES sensitivity tensor (single solve)
          - ∂ODE_param_j/∂η_k — central FD on the pk_callable (no ODE)
          - F = A3/V for all single-output ADVAN6 models

        Only available for non-mixed-pkpd ADVAN6 models; returns ``None``
        otherwise so the caller can fall through to the standard FD path.

        Args:
            theta:    Population THETA vector.
            eta:      Per-subject ETA vector (evaluated at η̂_i, not η=0).
            obs_mask: Boolean mask of non-MDV observations aligned to obs_times.
            n_eta:    Length of eta vector.
            eps:      Step size for pk_callable central finite differences.

        Returns:
            G_i array of shape (n_obs, n_eta), or ``None`` if unavailable.
        """
        contract = self._native_ode_contract
        if contract is None:
            return None
        # Mixed PK/PD models use DVID-based output routing not supported here.
        if contract.get("is_pkpd", False):
            return None
        if self.pk_callable is None:
            return None

        theta_list = [float(t) for t in theta]
        eta_list = [float(e) for e in eta]

        try:
            pk_params_0 = self.pk_callable(
                theta_list, eta_list, t=0.0, covariates=self._base_covariates
            )
        except Exception as _pk_e:
            logger.warning(
                "IndividualModel %s failed at pk_callable (eta Jacobian path): %s",
                getattr(self, "subject_id", "?"), _pk_e,
            )
            return None

        template = self._select_template(pk_params_0, contract)
        if template is None:
            return None

        required = template.required_names
        ode_theta = [float(pk_params_0[name]) for name in required]
        n_ode_params = len(required)
        V = float(pk_params_0[template.vol_param_name])
        v_idx = list(required).index(template.vol_param_name)
        output_cmt = template.output_cmt_idx
        n_states = template.n_states

        dose_times = _apply_alag(
            contract["dose_times"], contract["dose_compartments"], pk_params_0
        )

        all_obs_times = np.asarray(self.subject_events.obs_times, dtype=float)
        obs_times_masked = all_obs_times[obs_mask]
        n_obs = int(obs_mask.sum())
        if n_obs == 0:
            return np.zeros((0, n_eta))

        order = np.argsort(obs_times_masked, kind="stable")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n_obs)
        sorted_times = obs_times_masked[order]

        has_infusion = contract.get("has_infusion", False)
        try:
            if has_infusion:
                # Infusion models: use the infusion-aware sensitivity probe so
                # the sensitivity computation accounts for rate-on/rate-off
                # transitions.  Fall through to FD if the template lacks it.
                if template.infusion_sens_probe_fn is None:
                    return None
                states_raw, sens_raw = template.infusion_sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    contract["dose_rates"],
                    ode_theta,
                )
            else:
                states_raw, sens_raw = template.sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    ode_theta,
                )
        except Exception as _sens_e:
            logger.warning(
                "IndividualModel %s failed at native sensitivity probe (eta Jacobian): %s",
                getattr(self, "subject_id", "?"), _sens_e,
            )
            return None

        states = np.array(states_raw, dtype=float)[inv_order]   # (n_obs, n_states)
        sens = np.array(sens_raw, dtype=float).reshape(
            n_obs, n_ode_params, n_states
        )[inv_order]                                             # (n_obs, n_params, n_states)

        A_out = states[:, output_cmt]                           # (n_obs,)
        dF_dODE = sens[:, :, output_cmt] / V                    # (n_obs, n_params)
        dF_dODE[:, v_idx] -= A_out / (V * V)                   # quotient-rule for volume

        J_pk_eta = np.zeros((n_ode_params, n_eta))
        for k in range(n_eta):
            eta_p = eta_list.copy(); eta_p[k] += eps
            eta_m = eta_list.copy(); eta_m[k] -= eps
            try:
                pp = self.pk_callable(theta_list, eta_p, t=0.0, covariates=self._base_covariates)
                pm = self.pk_callable(theta_list, eta_m, t=0.0, covariates=self._base_covariates)
                for j, name in enumerate(required):
                    J_pk_eta[j, k] = (
                        float(pp.get(name, 0.0)) - float(pm.get(name, 0.0))
                    ) / (2.0 * eps)
            except Exception as _fd_e:
                logger.warning(
                    "IndividualModel %s failed at pk_callable FD (J_pk_eta col %d): %s",
                    getattr(self, "subject_id", "?"), k, _fd_e,
                )

        return dF_dODE @ J_pk_eta   # (n_obs, n_eta)

    def _native_gauss_newton_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray | None:
        """Gauss-Newton Hessian via CVODES sensitivities.  Returns None if unavailable.

        H_i ≈ 2 G_i^T diag(1/var) G_i  +  2 Ω^{-1}

        One Rust sensitivity solve replaces the 2·n_eta·(n_eta+1) ODE calls
        that ``numerical_hessian`` would require.  Returns None (never raises)
        so that callers can fall through to the numerical Hessian path.
        """
        contract = self._native_ode_contract
        if contract is None:
            return None
        if contract.get("is_pkpd", False):
            return None

        cem = self._common_error_model
        if cem is None:
            return None

        if self.pk_callable is None:
            return None

        obs_mask = self._obs_mask
        n_eta = len(eta)
        n_obs = int(obs_mask.sum())

        if n_obs == 0:
            try:
                omega_inv: np.ndarray = np.linalg.inv(omega)
            except np.linalg.LinAlgError:
                omega_inv = np.linalg.pinv(omega)
            return 2.0 * omega_inv

        theta_list = [float(t) for t in theta]
        eta_list = [float(e) for e in eta]

        try:
            pk_params_0 = self.pk_callable(
                theta_list, eta_list, t=0.0, covariates=self._base_covariates
            )
        except Exception as _pk_gnhess_e:
            logger.warning(
                "IndividualModel %s failed at pk_callable (Gauss-Newton hessian path): %s",
                getattr(self, "subject_id", "?"), _pk_gnhess_e,
            )
            return None

        template = self._select_template(pk_params_0, contract)
        if template is None:
            return None

        required = template.required_names
        ode_theta = [float(pk_params_0[name]) for name in required]
        n_ode_params = len(required)
        V = float(pk_params_0[template.vol_param_name])
        v_idx = list(required).index(template.vol_param_name)
        output_cmt = template.output_cmt_idx
        n_states = template.n_states

        dose_times = _apply_alag(
            contract["dose_times"], contract["dose_compartments"], pk_params_0
        )

        all_obs_times = np.asarray(self.subject_events.obs_times, dtype=float)
        obs_times_masked = all_obs_times[obs_mask]
        order = np.argsort(obs_times_masked, kind="stable")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n_obs)
        sorted_times = obs_times_masked[order]

        has_infusion = contract.get("has_infusion", False)
        try:
            if has_infusion:
                if template.infusion_sens_probe_fn is None:
                    return None
                states_raw, sens_raw = template.infusion_sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    contract["dose_rates"],
                    ode_theta,
                )
            else:
                states_raw, sens_raw = template.sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    ode_theta,
                )
        except Exception as _gn_sens_e:
            logger.warning(
                "IndividualModel %s failed at native sensitivity probe (Gauss-Newton): %s",
                getattr(self, "subject_id", "?"), _gn_sens_e,
            )
            return None

        states = np.array(states_raw, dtype=float)[inv_order]         # (n_obs, n_states)
        sens   = np.array(sens_raw,   dtype=float).reshape(
            n_obs, n_ode_params, n_states
        )[inv_order]                                                    # (n_obs, n_params, n_states)

        # ── G_i = ∂IPRED/∂η  ──────────────────────────────────────────────────
        A_out = states[:, output_cmt]                                   # (n_obs,)
        dF_dODE = sens[:, :, output_cmt] / V                           # (n_obs, n_params)
        dF_dODE[:, v_idx] -= A_out / (V * V)                          # quotient-rule for V

        J_pk_eta = np.zeros((n_ode_params, n_eta))
        for k in range(n_eta):
            eta_p = eta_list.copy(); eta_p[k] += eps
            eta_m = eta_list.copy(); eta_m[k] -= eps
            try:
                pp = self.pk_callable(theta_list, eta_p, t=0.0, covariates=self._base_covariates)
                pm = self.pk_callable(theta_list, eta_m, t=0.0, covariates=self._base_covariates)
                for j, name in enumerate(required):
                    J_pk_eta[j, k] = (
                        float(pp.get(name, 0.0)) - float(pm.get(name, 0.0))
                    ) / (2.0 * eps)
            except Exception as _gn_fd_e:
                logger.warning(
                    "IndividualModel %s failed at pk_callable FD (GN J_pk_eta col %d): %s",
                    getattr(self, "subject_id", "?"), k, _gn_fd_e,
                )

        G_i = dF_dODE @ J_pk_eta  # (n_obs, n_eta)

        # ── Residual variance from detected error model (no extra ODE solve) ──
        ipred = A_out / V           # (n_obs,)
        kind, theta_idx = cem
        sigma_00 = float(sigma[0, 0]) if sigma.size > 0 else 1.0

        if kind == "proportional":
            var = np.maximum(sigma_00 * ipred ** 2, 1e-10)
        elif kind == "additive":
            var = np.full(n_obs, max(sigma_00, 1e-10))
        elif kind == "proportional_theta":
            coeff = float(theta[theta_idx[0]]) ** 2
            var = np.maximum(coeff * ipred ** 2, 1e-10)
        elif kind == "additive_theta":
            var = np.full(n_obs, max(float(theta[theta_idx[0]]) ** 2, 1e-10))
        elif kind == "combined_theta":
            prop = float(theta[theta_idx[0]])
            add  = float(theta[theta_idx[1]])
            var = np.maximum(prop ** 2 + add ** 2 * ipred ** 2, 1e-10)
        elif kind == "combined_eps":
            s01 = float(sigma[0, 1]) if sigma.size >= 4 else 0.0
            s11 = float(sigma[1, 1]) if sigma.size >= 4 else sigma_00
            var = np.maximum(sigma_00 + 2.0 * s01 * ipred + s11 * ipred ** 2, 1e-10)
        else:
            return None  # Unsupported error model; caller falls back

        # ── Gauss-Newton Hessian ───────────────────────────────────────────────
        # H_i = 2 G^T R^{-1} G + 2 Ω^{-1}
        data_hess = 2.0 * (G_i.T / var) @ G_i  # (n_eta, n_eta)
        try:
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega)
        return data_hess + 2.0 * omega_inv

    def _native_eta_objective_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        eps: float = 1e-5,
    ) -> tuple[float, np.ndarray] | None:
        """Individual MAP objective value + gradient via CVODES sensitivities.

        Returns (obj_value, grad_eta) where obj = data_obj + eta_penalty, or
        None when the native path is unavailable.  One Rust sensitivity solve
        replaces the n_eta finite-difference ODE calls that L-BFGS-B would
        otherwise require.

        Gradient formula (chain rule through error model):
            grad_data = G_i^T @ v,   v_j = ∂data_obj/∂F_j
        For any error model with ∂var_j/∂F_j = D_j:
            v_j = -2 r_j / var_j  +  D_j * (1/var_j - r_j²/var_j²)
        where r_j = y_j - F_j.
        """
        contract = self._native_ode_contract
        if contract is None:
            return None
        if contract.get("is_pkpd", False):
            return None

        cem = self._common_error_model
        if cem is None:
            return None

        if self.pk_callable is None:
            return None

        obs_mask = self._obs_mask
        n_eta = len(eta)
        n_obs = int(obs_mask.sum())

        # Observed DV for masked (non-MDV) rows
        all_dv = np.asarray(self.subject_events.obs_dv, dtype=float)
        obs_dv = all_dv[obs_mask]
        if np.any(np.isnan(obs_dv)):
            return None  # missing DV in non-MDV row; can't compute likelihood

        # ── Penalty term ──────────────────────────────────────────────────────
        eta_arr = np.asarray(eta, dtype=float)
        omega_inv, block_size = self._eta_penalty_structure(omega, n_eta)
        eta_penalty = self._eta_penalty_value(eta_arr, omega_inv, block_size)
        if block_size is None:
            grad_penalty = 2.0 * (omega_inv @ eta_arr)
        else:
            eta_blocks = eta_arr.reshape(-1, block_size)
            grad_penalty = (2.0 * (eta_blocks @ omega_inv.T)).reshape(-1)

        if n_obs == 0:
            return float(eta_penalty), grad_penalty

        theta_list = [float(t) for t in theta]
        eta_list = [float(e) for e in eta]

        try:
            pk_params_0 = self.pk_callable(
                theta_list, eta_list, t=0.0, covariates=self._base_covariates
            )
        except Exception as _pk_vg_e:
            logger.warning(
                "IndividualModel %s failed at pk_callable (eta obj value/grad path): %s",
                getattr(self, "subject_id", "?"), _pk_vg_e,
            )
            return None

        template = self._select_template(pk_params_0, contract)
        if template is None:
            return None

        required = template.required_names
        ode_theta = [float(pk_params_0[name]) for name in required]
        n_ode_params = len(required)
        V = float(pk_params_0[template.vol_param_name])
        v_idx = list(required).index(template.vol_param_name)
        output_cmt = template.output_cmt_idx
        n_states = template.n_states

        dose_times = _apply_alag(
            contract["dose_times"], contract["dose_compartments"], pk_params_0
        )

        all_obs_times = np.asarray(self.subject_events.obs_times, dtype=float)
        obs_times_masked = all_obs_times[obs_mask]
        order = np.argsort(obs_times_masked, kind="stable")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n_obs)
        sorted_times = obs_times_masked[order]

        has_infusion = contract.get("has_infusion", False)
        try:
            if has_infusion:
                if template.infusion_sens_probe_fn is None:
                    return None
                states_raw, sens_raw = template.infusion_sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    contract["dose_rates"],
                    ode_theta,
                )
            else:
                states_raw, sens_raw = template.sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    ode_theta,
                )
        except Exception as _vg_sens_e:
            logger.warning(
                "IndividualModel %s failed at native sensitivity probe (value/grad path): %s",
                getattr(self, "subject_id", "?"), _vg_sens_e,
            )
            return None

        states = np.array(states_raw, dtype=float)[inv_order]
        sens   = np.array(sens_raw,   dtype=float).reshape(
            n_obs, n_ode_params, n_states
        )[inv_order]

        A_out = states[:, output_cmt]
        dF_dODE = sens[:, :, output_cmt] / V
        dF_dODE[:, v_idx] -= A_out / (V * V)

        J_pk_eta = np.zeros((n_ode_params, n_eta))
        for k in range(n_eta):
            eta_p = eta_list.copy(); eta_p[k] += eps
            eta_m = eta_list.copy(); eta_m[k] -= eps
            try:
                pp = self.pk_callable(theta_list, eta_p, t=0.0, covariates=self._base_covariates)
                pm = self.pk_callable(theta_list, eta_m, t=0.0, covariates=self._base_covariates)
                for j, name in enumerate(required):
                    J_pk_eta[j, k] = (
                        float(pp.get(name, 0.0)) - float(pm.get(name, 0.0))
                    ) / (2.0 * eps)
            except Exception as _vg_fd_e:
                logger.warning(
                    "IndividualModel %s failed at pk_callable FD (value/grad J_pk_eta col %d): %s",
                    getattr(self, "subject_id", "?"), k, _vg_fd_e,
                )

        G_i = dF_dODE @ J_pk_eta       # (n_obs, n_eta)
        ipred = A_out / V               # (n_obs,)

        # ── Residual variance and D_j = ∂var_j/∂F_j ──────────────────────────
        kind, theta_idx = cem
        sigma_00 = float(sigma[0, 0]) if sigma.size > 0 else 1.0

        if kind == "proportional":
            var = np.maximum(sigma_00 * ipred ** 2, 1e-10)
            D   = 2.0 * sigma_00 * ipred
        elif kind == "additive":
            var = np.full(n_obs, max(sigma_00, 1e-10))
            D   = np.zeros(n_obs)
        elif kind == "proportional_theta":
            coeff = float(theta[theta_idx[0]]) ** 2
            var = np.maximum(coeff * ipred ** 2, 1e-10)
            D   = 2.0 * coeff * ipred
        elif kind == "additive_theta":
            var = np.full(n_obs, max(float(theta[theta_idx[0]]) ** 2, 1e-10))
            D   = np.zeros(n_obs)
        elif kind == "combined_theta":
            prop = float(theta[theta_idx[0]])
            add  = float(theta[theta_idx[1]])
            var = np.maximum(prop ** 2 + add ** 2 * ipred ** 2, 1e-10)
            D   = 2.0 * add ** 2 * ipred
        elif kind == "combined_eps":
            s01 = float(sigma[0, 1]) if sigma.size >= 4 else 0.0
            s11 = float(sigma[1, 1]) if sigma.size >= 4 else sigma_00
            var = np.maximum(sigma_00 + 2.0 * s01 * ipred + s11 * ipred ** 2, 1e-10)
            D   = 2.0 * s01 + 2.0 * s11 * ipred
        else:
            return None

        # ── Value and gradient ────────────────────────────────────────────────
        r = obs_dv - ipred
        data_val = float(np.sum(r ** 2 / var + np.log(var)))
        v = -2.0 * r / var + D * (1.0 / var - r ** 2 / var ** 2)   # (n_obs,)
        grad_data = G_i.T @ v                                        # (n_eta,)

        return float(data_val + eta_penalty), grad_data + grad_penalty


