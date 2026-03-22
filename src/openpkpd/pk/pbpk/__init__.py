"""
Physiologically-based pharmacokinetic (PBPK) models.

Provides a base class wrapping ADVAN6 with organ-specific compartment naming
and a 5-organ human PBPK template (lung, liver, kidney, gut, central/blood).

Usage::

    from openpkpd.pk.pbpk import FiveOrganPBPK

    model = FiveOrganPBPK()
    sol = model.solve(pk_params, dose_events, obs_times, des_callable=my_des)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.pk.ode.advan6 import ADVAN6

# G1 ── PBPK base class ────────────────────────────────────────────────────────


class PBPKModel(PKSubroutine):
    """
    Base class for PBPK models.

    Wraps ADVAN6 (generic ODE solver) with organ-specific compartment naming.
    Subclasses define ``compartment_names`` and the DES callable describing
    inter-organ blood flow and metabolic/elimination processes.

    Attributes:
        compartment_names: Ordered list of compartment names (length must equal
                           the number of ODEs in the DES block).
        output_compartment: Name of the compartment used to compute IPRED.
                            Must appear in ``compartment_names``.
    """

    compartment_names: list[str] = []
    output_compartment_name: str = "central"

    def __init__(self) -> None:
        n = len(self.compartment_names)
        if n == 0:
            raise ValueError("PBPKModel subclass must define compartment_names")
        self.n_compartments = n
        self._advan6 = ADVAN6(n_compartments=n)
        # index (0-based) of the output compartment
        try:
            self._output_idx = self.compartment_names.index(self.output_compartment_name)
        except ValueError:
            self._output_idx = 0

    # ── Compartment helpers ──────────────────────────────────────────────────

    def compartment_index(self, name: str) -> int:
        """Return the 1-based compartment index for the given organ name."""
        return self.compartment_names.index(name) + 1

    # ── Solve ────────────────────────────────────────────────────────────────

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """
        Solve the PBPK model ODE system via ADVAN6.

        The ``des_callable`` must define dA(n)/dt for each compartment
        in the order specified by ``compartment_names``.

        IPRED is extracted from compartment ``output_compartment_name``
        divided by pk_params['V'] (or 'V_central') if present.
        """
        # Delegate integration to ADVAN6
        sol = self._advan6.solve(
            pk_params,
            dose_events,
            obs_times,
            pk_callable=pk_callable,
            des_callable=des_callable,
        )

        # Override output compartment if needed
        if sol.amounts is not None and sol.amounts.shape[1] > self._output_idx:
            v_key = f"V_{self.output_compartment_name}"
            v = pk_params.get(v_key, pk_params.get("V", 1.0))
            ipred = sol.amounts[:, self._output_idx] / max(float(v), 1e-12)
            ipred = np.maximum(ipred, 0.0)
            return PKSolution(times=sol.times, amounts=sol.amounts, ipred=ipred)

        return sol

    def apply_trans(self, pk_params: dict, trans: int = 1) -> dict:
        """PBPK models use TRANS1 (micro-parameters directly)."""
        return pk_params


# G1 ── 5-organ human PBPK template ──────────────────────────────────────────


class FiveOrganPBPK(PBPKModel):
    """
    Five-organ human PBPK template.

    Organs (compartments in order):
        0: lung
        1: liver
        2: kidney
        3: gut
        4: central (blood / plasma)

    Typical parameters (all flows in L/h, volumes in L):
        Q_lung, Q_liver, Q_kidney, Q_gut  — organ blood flows
        V_lung, V_liver, V_kidney, V_gut, V_central — organ volumes
        CL_liver, CL_kidney                — metabolic clearances
        Kp_lung, Kp_liver, Kp_kidney, Kp_gut — tissue:plasma partitioning

    DES example (to be compiled and passed as des_callable)::

        DADT(1) = Q_lung  * (C_central - A(1)/V_lung  / Kp_lung)   ; lung
        DADT(2) = Q_liver * (C_central - A(2)/V_liver / Kp_liver) - CL_liver*(A(2)/V_liver)
        DADT(3) = Q_kidney*(C_central - A(3)/V_kidney/ Kp_kidney)- CL_kidney*(A(3)/V_kidney)
        DADT(4) = Q_gut   * (C_central - A(4)/V_gut   / Kp_gut)
        DADT(5) = -(Q_lung+Q_liver+Q_kidney+Q_gut)*C_central \\
                  + Q_lung*(A(1)/V_lung/Kp_lung) \\
                  + Q_liver*(A(2)/V_liver/Kp_liver) \\
                  + Q_kidney*(A(3)/V_kidney/Kp_kidney) \\
                  + Q_gut*(A(4)/V_gut/Kp_gut)

    where C_central = A(5) / V_central.
    """

    compartment_names = ["lung", "liver", "kidney", "gut", "central"]
    output_compartment_name = "central"


# ── Exports ───────────────────────────────────────────────────────────────────

__all__ = [
    "PBPKModel",
    "FiveOrganPBPK",
]
