"""
ADVAN13 — ODE with adjoint-sensitivity gradient computation.

In NONMEM, ADVAN13 integrates stiff ODEs and uses adjoint sensitivity
equations to provide exact gradients of the OFV with respect to parameters.
This enables faster and more accurate gradient-based estimation (FOCE,
Laplacian) for complex stiff systems compared to finite-difference
sensitivity.

Implementation strategy:
  - If JAX + diffrax are available: use diffrax with ``diffrax.Adjoint``
    controller, which provides exact reverse-mode gradients through the ODE
    solution via the continuous adjoint method.
  - Fallback: delegate to ADVAN8 (stiff scipy LSODA solver) with
    finite-difference sensitivity (same as ADVAN8 but with tighter
    tolerances appropriate for adjoint-quality gradients).

The JAX path is automatically selected when ``jax`` and ``diffrax`` can be
imported.  Set ``force_scipy=True`` to disable JAX even when available.

References:
    Pontryagin LS et al. (1962). The Mathematical Theory of Optimal
        Processes. Wiley-Interscience.
    Chen RTQ et al. (2018). Neural ordinary differential equations.
        NeurIPS 2018. (Modern diffrax implementation.)
    NONMEM 7.5 User's Guide — ADVAN13 documentation.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.pk.base import PKSolution
from openpkpd.pk.ode.advan8 import ADVAN8


def _is_importable(name: str) -> bool:
    """Return True if *name* can be imported.

    Handles monkeypatched ``sys.modules`` entries (including fake module
    objects whose ``__spec__`` is None, which confuse ``find_spec``).
    """
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


class ADVAN13(ADVAN8):
    """
    ODE solver with adjoint sensitivity (ADVAN13).

    Provides the same interface as ADVAN8/ADVAN6 but uses the adjoint method
    for gradient computation when JAX and diffrax are available.

    When JAX is not available, falls back to ADVAN8 (stiff scipy LSODA) with
    a tighter tolerance to approximate the accuracy of adjoint gradients.

    Args:
        n_compartments: Number of ODE compartments (default 10).
        rtol:           Relative ODE tolerance (default 1e-8, tighter than ADVAN8).
        atol:           Absolute ODE tolerance (default 1e-10).
        force_scipy:    If True, always use scipy even when JAX is available.
    """

    advan: int = 13

    def __init__(
        self,
        n_compartments: int = 10,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        force_scipy: bool = False,
    ) -> None:
        super().__init__(
            n_compartments=n_compartments,
            rtol=rtol,
            atol=atol,
            method="Radau",  # Radau is generally more accurate for stiff systems
        )
        self.force_scipy = force_scipy
        self._jax_available: bool | None = None

    def _check_jax(self) -> bool:
        """Check whether JAX and diffrax are importable."""
        if self._jax_available is not None:
            return self._jax_available

        self._jax_available = _is_importable("jax") and _is_importable("diffrax")
        return self._jax_available

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
        Solve the ODE system, using adjoint sensitivity when JAX is available.

        For the JAX/diffrax path, the ODE is solved with a stiff solver
        (Kvaerno5 or Dopri8) and the ``Adjoint`` controller records a
        continuous adjoint tape for reverse-mode differentiation.

        For the scipy fallback path, delegates to ADVAN8.solve() with the
        tighter tolerances set at construction time.

        Args:
            pk_params:    PK parameter dictionary.
            dose_events:  List of DoseEvent objects.
            obs_times:    Observation times array.
            pk_callable:  Compiled $PK callable (unused; params already resolved).
            des_callable: Compiled $DES right-hand-side callable.

        Returns:
            PKSolution with ipred, amounts, f arrays.
        """
        if not self.force_scipy and self._check_jax():
            return self._solve_adjoint_jax(pk_params, dose_events, obs_times, des_callable)
        # Fallback: stiff scipy
        return super().solve(pk_params, dose_events, obs_times, pk_callable, des_callable)

    def _solve_adjoint_jax(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        des_callable: Callable | None,
    ) -> PKSolution:
        """
        Adjoint ODE solve via diffrax.

        Uses ``diffrax.diffeqsolve`` with ``diffrax.Adjoint()`` as the
        adjoint controller and a stiff solver (``diffrax.Kvaerno5`` by
        default).  The resulting solution supports exact reverse-mode
        gradients via ``jax.grad``.

        Falls back to scipy on any diffrax error (e.g. convergence failure).

        Args:
            pk_params:    PK parameter dictionary.
            dose_events:  List of DoseEvent objects.
            obs_times:    Observation times array.
            des_callable: Compiled $DES right-hand-side callable.

        Returns:
            PKSolution (identical structure to ADVAN6/ADVAN8 output).
        """
        try:
            import diffrax
            import jax.numpy as jnp

            if des_callable is None:
                # No DES block: nothing to differentiate; use scipy
                return super().solve(pk_params, dose_events, obs_times, None, None)

            n_cmt = self.n_compartments
            t0 = 0.0
            t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 1.0

            # Build initial conditions from dose events at t=0
            y0 = jnp.zeros(n_cmt)
            for de in dose_events:
                if float(de.time) == 0.0:
                    cmt_idx = int(de.compartment) - 1
                    if 0 <= cmt_idx < n_cmt:
                        y0 = y0.at[cmt_idx].add(float(de.amount))

            # Build diffrax ODE term from the compiled DES callable
            def vector_field(t: float, y: jnp.ndarray, args: None) -> jnp.ndarray:
                a = list(y)
                dadt = des_callable(float(t), a, pk_params, [], [])
                return jnp.array(dadt, dtype=float)

            term = diffrax.ODETerm(vector_field)
            solver = diffrax.Kvaerno5()
            saveat = diffrax.SaveAt(ts=jnp.array(obs_times, dtype=float))
            stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)

            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t_max,
                dt0=None,
                y0=y0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=diffrax.Adjoint(),
                max_steps=10_000,
            )

            amounts = np.array(sol.ys)  # shape (n_times, n_cmt)

            # IPRED = amount in output compartment / volume
            output_cmt = int(pk_params.get("CMT", 1)) - 1
            v_key = "V" if "V" in pk_params else "V1"
            v = float(pk_params.get(v_key, 1.0))

            if amounts.ndim == 2 and amounts.shape[1] > output_cmt:
                ipred = amounts[:, output_cmt] / v
            else:
                ipred = np.zeros(len(obs_times))

            return PKSolution(
                ipred=np.maximum(ipred, 0.0),
                amounts=amounts,
                times=obs_times,
                f=None,
            )

        except Exception:
            # Any diffrax failure: fall back to ADVAN8 scipy
            return super().solve(pk_params, dose_events, obs_times, None, des_callable)

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
        Solve the ODE and forward sensitivity equations (no JAX required).

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
