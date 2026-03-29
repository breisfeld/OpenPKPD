"""
ADVAN13 — stiff ODE solving plus forward sensitivity equations.

In NONMEM, ADVAN13 integrates stiff ODEs and is associated with sensitivity-
aware estimation workflows. In OpenPKPD, ADVAN13 is implemented as a SciPy
stiff solve with tighter tolerances than ADVAN8 plus an explicit forward-
mode sensitivity solve for gradient-related workflows.

References:
    NONMEM 7.5 User's Guide — ADVAN13 documentation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.pk.base import PKSolution
from openpkpd.pk.ode.advan8 import ADVAN8


class ADVAN13(ADVAN8):
    """
    ODE solver with forward sensitivity support (ADVAN13).

    Provides the same interface as ADVAN8/ADVAN6 but uses a tighter stiff
    SciPy solve and exposes forward sensitivity equations through
    ``solve_with_sensitivity``.

    Args:
        n_compartments: Number of ODE compartments (default 10).
        rtol:           Relative ODE tolerance (default 1e-8, tighter than ADVAN8).
        atol:           Absolute ODE tolerance (default 1e-10).
    """

    advan: int = 13

    def __init__(
        self,
        n_compartments: int = 10,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> None:
        super().__init__(
            n_compartments=n_compartments,
            rtol=rtol,
            atol=atol,
            method="Radau",  # Radau is generally more accurate for stiff systems
        )

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
        covariate_fn: Callable | None = None,
        covariate_change_times: list[float] | None = None,
    ) -> PKSolution:
        """
        Solve the ODE system with the stiff SciPy path configured for ADVAN13.

        Args:
            pk_params:    PK parameter dictionary.
            dose_events:  List of DoseEvent objects.
            obs_times:    Observation times array.
            pk_callable:  Compiled $PK callable (unused; params already resolved).
            des_callable: Compiled $DES right-hand-side callable.

        Returns:
            PKSolution with ipred, amounts, f arrays.
        """
        return super().solve(pk_params, dose_events, obs_times, pk_callable, des_callable)

    # ── Forward-mode sensitivity ───────────────────────────────────────────────

    def solve_with_sensitivity(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        des_callable: Callable | None,
        param_names: list[str] | None = None,
        fd_step: float = 1e-5,
    ) -> PKSolution:
        """
        Solve the ODE and forward sensitivity equations.

        Augments the ODE ``dy/dt = f(y, t, p)`` with variational equations:

            ds_j/dt = J_y(t) @ s_j + J_p_j(t)

        where ``s_j = ∂y/∂p_j`` is the state sensitivity w.r.t. parameter
        ``p_j``, ``J_y = ∂f/∂y`` is the Jacobian of the RHS w.r.t. states, and
        ``J_p_j = ∂f/∂p_j`` is the partial derivative of the RHS w.r.t. ``p_j``.

        Both Jacobians are computed by forward finite differences on the
        ``des_callable`` at each integration step.  The augmented system is
        solved with ``scipy.integrate.solve_ivp`` using the same ``Radau``
        solver and tolerances as the base ADVAN13 solve.

        The returned :class:`~openpkpd.pk.base.PKSolution` has its
        ``sensitivity`` field populated with shape ``(n_times, n_params)``,
        where entry ``[t, j]`` is ``∂IPRED_t / ∂p_j``.

        Args:
            pk_params:    PK parameter dictionary (values are the current point).
            dose_events:  List of DoseEvent objects.
            obs_times:    1-D array of observation times.
            des_callable: Compiled $DES RHS callable.
            param_names:  Names of parameters in ``pk_params`` to differentiate.
                          When ``None``, all scalar numeric entries are used.
            fd_step:      Finite-difference step size (default 1e-5).

        Returns:
            PKSolution with ``sensitivity`` filled (shape n_times × n_params).
            Falls back to base ``solve()`` (sensitivity=None) on any error.
        """
        if des_callable is None:
            # No DES block: cannot compute sensitivities; return zero solution.
            n_obs = len(obs_times)
            return PKSolution(
                times=obs_times,
                amounts=np.zeros((n_obs, self.n_compartments)),
                ipred=np.zeros(n_obs),
                sensitivity=None,
            )

        # Determine parameter list
        if param_names is None:
            param_names = [
                k for k, v in pk_params.items() if isinstance(v, (int, float)) and k not in ("CMT",)
            ]
        n_params = len(param_names)
        n_cmt = self.n_compartments

        # Build initial conditions
        y0_base = np.zeros(n_cmt)
        for de in dose_events:
            if float(de.time) == 0.0:
                idx = int(de.compartment) - 1
                if 0 <= idx < n_cmt:
                    y0_base[idx] += float(de.amount)

        # Augmented initial state: [y (n_cmt), s_1..s_p (n_cmt each)]
        # s_j(0) = 0 (sensitivity starts at zero)
        z0 = np.concatenate([y0_base, np.zeros(n_cmt * n_params)])
        t0 = 0.0
        t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 1.0
        eps = fd_step

        _empty: list[float] = []  # theta/eta placeholders for des_callable

        def _rhs(t: float, z: np.ndarray) -> np.ndarray:
            y = z[:n_cmt]
            a_list = list(y)
            # Evaluate base RHS — signature: (t, a, pk_params, theta, eta)
            try:
                f0 = np.asarray(des_callable(t, a_list, pk_params, _empty, _empty), dtype=float)
            except Exception:
                f0 = np.zeros(n_cmt)

            # Jacobian w.r.t. state: J_y via FD
            J_y = np.zeros((n_cmt, n_cmt))
            for i in range(n_cmt):
                a_p = a_list.copy()
                a_p[i] += eps
                try:
                    fp = np.asarray(des_callable(t, a_p, pk_params, _empty, _empty), dtype=float)
                except Exception:
                    fp = f0.copy()
                J_y[:, i] = (fp - f0) / eps

            dz = np.empty_like(z)
            dz[:n_cmt] = f0

            # Sensitivity equations for each parameter
            for j, pname in enumerate(param_names):
                p0 = float(pk_params.get(pname, 0.0))
                p_step = eps * max(abs(p0), 1.0)
                pk_p = {**pk_params, pname: p0 + p_step}
                try:
                    fp_p = np.asarray(des_callable(t, a_list, pk_p, _empty, _empty), dtype=float)
                except Exception:
                    fp_p = f0.copy()
                J_p_j = (fp_p - f0) / p_step

                s_j = z[n_cmt + j * n_cmt : n_cmt + (j + 1) * n_cmt]
                dz[n_cmt + j * n_cmt : n_cmt + (j + 1) * n_cmt] = J_y @ s_j + J_p_j

            return dz

        try:
            ivp_sol = solve_ivp(
                _rhs,
                [t0, t_max],
                z0,
                method="Radau",
                t_eval=obs_times,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False,
            )

            if not ivp_sol.success or ivp_sol.y is None:
                raise RuntimeError(ivp_sol.message)

            # y_all: shape (n_cmt, n_times) → transpose to (n_times, n_cmt)
            amounts = ivp_sol.y[:n_cmt, :].T  # (n_times, n_cmt)

            # IPRED from output compartment / volume
            output_cmt = int(pk_params.get("CMT", 1)) - 1
            v_key = "V" if "V" in pk_params else "V1"
            v = float(pk_params.get(v_key, 1.0))
            if amounts.ndim == 2 and amounts.shape[1] > output_cmt:
                ipred = np.maximum(amounts[:, output_cmt] / v, 0.0)
            else:
                ipred = np.zeros(len(obs_times))

            # Sensitivities: ∂ipred/∂p_j = (∂y_{output}/∂p_j) / v
            sensitivity = np.zeros((len(obs_times), n_params))
            for j in range(n_params):
                s_j_all = ivp_sol.y[n_cmt + j * n_cmt : n_cmt + (j + 1) * n_cmt, :].T
                if s_j_all.shape[1] > output_cmt:
                    sensitivity[:, j] = s_j_all[:, output_cmt] / v

            return PKSolution(
                ipred=ipred,
                amounts=amounts,
                times=obs_times,
                sensitivity=sensitivity,
            )

        except Exception:
            # Fallback: base solve without sensitivity
            return super().solve(pk_params, dose_events, obs_times, None, des_callable)
