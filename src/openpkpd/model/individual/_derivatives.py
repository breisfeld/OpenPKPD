"""DerivativesMixin — Jacobians, Hessians, supports_* capabilities, penalty."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from openpkpd.math.autodiff import jacobian
from openpkpd.math.matrix import numerical_gradient, numerical_hessian

logger = logging.getLogger(__name__)


class DerivativesMixin:
    """Mixin providing gradient, Jacobian, and Hessian methods."""

    def supports_eta_objective_gradient(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is not None and getattr(capabilities, "eta_objective_gradient", False):
            return True
        # Native CVODES sensitivity path: available for PK-only models with a
        # matching template.  The actual probe availability is checked lazily
        # inside _native_eta_objective_value_grad; returning True here allows
        # callers (IMPMAP) to attempt the analytical gradient path.
        contract = self._native_ode_contract
        return (
            contract is not None
            and not contract.get("is_pkpd", False)
            and self._common_error_model is not None
        )

    def supports_prediction_eta_jacobian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        return bool(kernel is not None and getattr(capabilities, "prediction_eta_jacobian", False))

    def supports_theta_data_objective_gradient(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is None or not getattr(capabilities, "theta_data_objective_gradient", False):
            return False
        supports = getattr(kernel, "supports_theta_data_objective_gradient", None)
        return bool(callable(supports) and supports())

    def supports_prediction_theta_jacobian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is None or not getattr(capabilities, "prediction_theta_jacobian", False):
            return False
        supports = getattr(kernel, "supports_theta_data_objective_gradient", None)
        return bool(callable(supports) and supports())

    def supports_eta_objective_hessian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        return bool(kernel is not None and getattr(capabilities, "eta_objective_hessian", False))

    def eta_objective_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[float, np.ndarray]:
        # ── PATH 1: symbolic / autodiff derivative kernel ─────────────────────
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is not None:
            data_value, grad_data = kernel.eta_data_objective_value_grad(
                theta, np.asarray(eta, dtype=float), sigma
            )
            eta_arr = np.asarray(eta, dtype=float)
            omega_inv, block_size = self._eta_penalty_structure(omega, len(eta_arr))
            eta_penalty = self._eta_penalty_value(eta_arr, omega_inv, block_size)
            if block_size is None:
                grad_penalty = 2.0 * (omega_inv @ eta_arr)
            else:
                eta_blocks = eta_arr.reshape(-1, block_size)
                grad_penalty = (2.0 * (eta_blocks @ omega_inv.T)).reshape(-1)
            return data_value + eta_penalty, np.asarray(grad_data, dtype=float) + grad_penalty

        # ── PATH 2: native CVODES sensitivity (single Rust sensitivity solve) ─
        # Replaces n_eta finite-difference ODE calls with one sensitivity solve.
        # Returns None silently when model or extension is unsupported.
        native_result = self._native_eta_objective_value_grad(
            np.asarray(eta, dtype=float), theta, omega, sigma
        )
        if native_result is not None:
            return native_result

        # ── PATH 3: numerical fallback (finite differences on obj_eta) ────────
        # Reached when supports_eta_objective_gradient() returned True (native
        # contract present) but the probe failed at runtime (e.g. NaN obs_dv).
        # This preserves the semantics that if supports=True the method never
        # raises; callers (FOCE, IMPMAP) get a valid tuple even in edge cases.
        eta_arr = np.asarray(eta, dtype=float)
        if self._native_ode_contract is not None:
            val0 = float(self.obj_eta(eta_arr, theta, omega, sigma, trans=trans))
            grad_fd = numerical_gradient(
                lambda e: float(self.obj_eta(e, theta, omega, sigma, trans=trans)),
                eta_arr,
                eps=1e-4,
            )
            return val0, grad_fd

        raise NotImplementedError("eta objective derivative kernel is not available")

    def supports_symbolic_obj_eta(self, trans: int = 2) -> bool:
        return self.supports_eta_objective_gradient(trans)

    def symbolic_obj_eta_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[float, np.ndarray]:
        return self.eta_objective_value_grad(eta, theta, omega, sigma, trans=trans)

    def get_subject_derivative_kernel(self, trans: int = 2) -> Any | None:
        if trans in self._derivative_kernel_cache:
            return self._derivative_kernel_cache[trans]
        from openpkpd.model.derivative_kernels import build_subject_derivative_kernel

        kernel = build_subject_derivative_kernel(self, trans)
        self._derivative_kernel_cache[trans] = kernel
        return kernel

    def _get_symbolic_eta_objective(self, trans: int) -> Any | None:
        return self.get_subject_derivative_kernel(trans)

    def eta_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        if len(eta_arr) == 0:
            return np.zeros((0, 0), dtype=float)

        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is not None and getattr(capabilities, "eta_objective_hessian", False):
            data_hess = np.asarray(
                kernel.eta_data_objective_hessian(theta, eta_arr, sigma), dtype=float
            )
            omega_inv, block_size = self._eta_penalty_structure(omega, len(eta_arr))
            if block_size is None:
                penalty_hess = 2.0 * omega_inv
            else:
                n_blocks = len(eta_arr) // block_size
                penalty_hess = 2.0 * np.kron(np.eye(n_blocks), omega_inv)
            return data_hess + np.asarray(penalty_hess, dtype=float)

        # ── Native CVODES Gauss-Newton Hessian (single Rust sensitivity solve) ──
        # Replaces 2·n_eta·(n_eta+1) ODE calls with one sensitivity integration.
        # Returns None silently when the model or extension is unsupported.
        native_H = self._native_gauss_newton_hessian(theta, eta_arr, omega, sigma)
        if native_H is not None:
            return native_H

        def obj_eta_local(eta_value: np.ndarray) -> float:
            return float(self.obj_eta(eta_value, theta, omega, sigma, trans=trans))

        return numerical_hessian(obj_eta_local, eta_arr, eps=1e-4)

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        obs_mask = self.subject_events.observation_mask()
        n_obs = int(np.sum(obs_mask))
        if len(eta_arr) == 0:
            return np.zeros((n_obs, 0), dtype=float)

        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is not None and getattr(capabilities, "prediction_eta_jacobian", False):
            return np.asarray(kernel.prediction_eta_jacobian(theta, eta_arr, sigma), dtype=float)

        def pred_of_eta(eta_value: np.ndarray) -> np.ndarray:
            _, _, _, pred_eta, _ = self.evaluate_observation_model(
                theta, eta_value, sigma, trans=trans
            )
            return pred_eta[obs_mask]

        return jacobian(pred_of_eta, eta_arr, eps=1e-5)

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is None:
            raise NotImplementedError("theta data-objective derivative kernel is not available")
        return np.asarray(
            kernel.theta_data_objective_gradient(
                np.asarray(theta, dtype=float),
                np.asarray(eta, dtype=float),
                sigma,
            ),
            dtype=float,
        )

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        obs_mask = self.subject_events.observation_mask()
        n_obs = int(np.sum(obs_mask))
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is not None and self.supports_prediction_theta_jacobian(trans):
            return np.asarray(kernel.prediction_theta_jacobian(theta_arr, eta, sigma), dtype=float)
        raise NotImplementedError("prediction theta Jacobian is not available")

    @staticmethod
    def _eta_penalty_value(
        eta: np.ndarray,
        omega_inv: np.ndarray,
        block_size: int | None,
    ) -> float:
        if block_size is None:
            eta_penalty = float(eta @ omega_inv @ eta)
        else:
            eta_blocks = np.asarray(eta, dtype=float).reshape(-1, block_size)
            eta_penalty = float(np.einsum("bi,ij,bj->", eta_blocks, omega_inv, eta_blocks))
        return eta_penalty

    def _eta_penalty_structure(
        self,
        omega: np.ndarray,
        n_eta: int,
    ) -> tuple[np.ndarray, int | None]:
        omega_arr = np.ascontiguousarray(omega, dtype=float)
        cache_key = (omega_arr.tobytes(), omega_arr.shape, n_eta)
        if self._eta_penalty_cache_key == cache_key and self._eta_penalty_precision is not None:
            return self._eta_penalty_precision, self._eta_penalty_block_size

        from openpkpd.math.matrix import repair_pd

        n_bsv = omega.shape[0]
        block_size: int | None = None
        try:
            omega_inv = np.linalg.inv(repair_pd(omega_arr))
        except np.linalg.LinAlgError:
            omega_inv = np.eye(n_bsv)

        if n_eta > n_bsv and self.occasion_indices is not None:
            block_size = n_bsv

        self._eta_penalty_cache_key = cache_key
        self._eta_penalty_precision = omega_inv
        self._eta_penalty_block_size = block_size
        return omega_inv, block_size


