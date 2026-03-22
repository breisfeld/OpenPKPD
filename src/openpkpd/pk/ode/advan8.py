"""
ADVAN8 — Generic stiff ODE model via scipy solve_ivp (LSODA).

Identical to ADVAN6 but uses LSODA (or Radau) as the integration method,
which is appropriate for stiff ODE systems such as those with widely
separated eigenvalues (e.g., Michaelis-Menten elimination at low doses,
multi-compartment systems with very fast/slow transfer rates).

See advan6.py for full documentation of the solve() interface.
"""

from __future__ import annotations

from openpkpd.pk.ode.advan6 import ADVAN6


class ADVAN8(ADVAN6):
    """
    Generic stiff ODE model (ADVAN8).

    Inherits all functionality from ADVAN6 but defaults to the LSODA
    integration method, which uses automatic stiffness detection and
    switches between Adams (nonstiff) and BDF (stiff) methods.

    Parameters:
        n_compartments: Number of ODE compartments (default 10).
        rtol:           Relative tolerance (default 1e-6).
        atol:           Absolute tolerance (default 1e-9, tighter for stiff systems).
        method:         scipy solve_ivp method (default 'LSODA').
    """

    advan: int = 8

    def __init__(
        self,
        n_compartments: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        method: str = "LSODA",
    ) -> None:
        """
        Initialize the stiff ODE solver settings.

        Args:
            n_compartments: Number of ODE compartments.
            rtol:           Relative tolerance (tighter default than ADVAN6).
            atol:           Absolute tolerance (tighter default for stiff systems).
            method:         scipy integration method; 'LSODA' or 'Radau' recommended.
        """
        super().__init__(
            n_compartments=n_compartments,
            rtol=rtol,
            atol=atol,
            method=method,
        )
