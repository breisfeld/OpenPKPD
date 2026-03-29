"""Reusable derivative-kernel interfaces for subject-level model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class DerivativeKernelCapabilities:
    eta_objective_gradient: bool = False
    eta_objective_hessian: bool = False
    prediction_eta_jacobian: bool = False
    theta_data_objective_gradient: bool = False
    prediction_theta_jacobian: bool = False


@runtime_checkable
class SubjectDerivativeKernel(Protocol):
    capabilities: DerivativeKernelCapabilities

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]: ...

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def eta_data_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def supports_theta_data_objective_gradient(self) -> bool: ...

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...


class BaseSubjectDerivativeKernel:
    capabilities = DerivativeKernelCapabilities()

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        values = np.empty(len(eta_arr), dtype=float)
        for i, eta in enumerate(eta_arr):
            values[i] = self.eta_data_objective_value_grad(theta, eta, sigma)[0]
        return values

    def eta_data_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def supports_theta_data_objective_gradient(self) -> bool:
        return False

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


def build_subject_derivative_kernel(indiv: Any, trans: int) -> SubjectDerivativeKernel | None:
    from openpkpd.model.symbolic_eta import (
        SympyAdvan1Trans2Objective,
        SympyAdvan2Trans2Objective,
        SympyAdvan3Trans4Objective,
        SympyAdvan4Trans1Objective,
    )

    return (
        SympyAdvan2Trans2Objective.build(indiv, trans)
        or SympyAdvan1Trans2Objective.build(indiv, trans)
        or SympyAdvan3Trans4Objective.build(indiv, trans)
        or SympyAdvan4Trans1Objective.build(indiv, trans)
    )
