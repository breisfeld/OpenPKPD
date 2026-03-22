"""
Abstract base classes for PK subroutines.

All PK models produce a PKSolution: compartment amounts at requested times.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from openpkpd.pk.trans import get_transformer


@dataclass
class PKSolution:
    """
    Computed compartment concentrations / amounts at observation times.

    Attributes:
        times:   1-D array of times at which solution was evaluated.
        amounts: 2-D array (n_times × n_compartments) of compartment amounts.
        ipred:   1-D array of individual predicted concentrations (output cmt).
        f:       Bioavailability-adjusted prediction = F * concentration.
    """

    times: np.ndarray  # shape (n_times,)
    amounts: np.ndarray  # shape (n_times, n_compartments)
    ipred: np.ndarray  # shape (n_times,)
    f: np.ndarray | None = None  # shape (n_times,) — same as ipred unless modified
    sensitivity: np.ndarray | None = None  # shape (n_times, n_params) — ∂ipred/∂p

    def __post_init__(self) -> None:
        if self.f is None:
            self.f = self.ipred.copy()


class PKSubroutine(ABC):
    """
    Abstract base class for all PK subroutines (ADVAN models).

    Subclasses implement solve() which takes PK parameters, dose events,
    and observation times and returns a PKSolution.
    """

    #: Number of compartments in this model
    n_compartments: int = 1

    #: ADVAN number (e.g., 1 for ADVAN1)
    advan: int = 0

    #: Default output compartment (1-based)
    output_compartment: int = 1

    @abstractmethod
    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,  # list[DoseEvent]
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """
        Solve the PK model for a single subject.

        Args:
            pk_params:     Dict of model parameters (e.g., {'CL': 5.0, 'V': 30.0}).
            dose_events:   List of DoseEvent objects for this subject.
            obs_times:     Times at which to evaluate the solution.
            pk_callable:   Optional compiled $PK callable (overrides defaults).
            des_callable:  Optional compiled $DES callable (for ODE models).

        Returns:
            PKSolution with amounts and IPRED at obs_times.
        """

    def apply_trans(self, raw_params: dict[str, float], trans: int) -> dict[str, float]:
        """
        Apply TRANS parameterization to convert user PK params to micro constants.

        Delegates to the appropriate trans module.
        """
        cache = getattr(self, "_transformer_cache", None)
        if cache is None:
            cache = {}
            self._transformer_cache = cache
        fn = cache.get(trans)
        if fn is None:
            fn = get_transformer(trans, self.advan)
            cache[trans] = fn
        return fn(raw_params)

    def get_output_compartment(self, pk_params: dict[str, float]) -> int:
        """Return the output compartment index (1-based). May be overridden."""
        return pk_params.get("PCMT", self.output_compartment)  # type: ignore[return-value]
