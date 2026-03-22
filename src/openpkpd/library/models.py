"""
Pre-built PK/PD model library.

Factory functions returning configured ModelBuilder instances for
common pharmacokinetic and pharmacodynamic models.

Usage:
    from openpkpd.library import one_cmt_oral
    model = one_cmt_oral(data_path="theo.csv").build().fit()
"""

from __future__ import annotations

from typing import Any

from openpkpd.api.model_builder import ModelBuilder

# ── Pharmacokinetic Models ──────────────────────────────────────────────────


def one_cmt_iv(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    One-compartment IV model (ADVAN1, TRANS2).

    Structural model: single intravenous compartment with first-order elimination.
    Parameters: CL (clearance), V (volume of distribution).
    IIV: ETA on CL and V. Proportional residual error.

    THETA(1): CL — clearance [L/hr]; default init=5.0
    THETA(2): V  — volume [L]; default init=20.0
    THETA(3): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (default 5.0), v_init (default 20.0),
                   iiv_cl (default 0.3), iiv_v (default 0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("One-compartment IV (ADVAN1)")
        .subroutines(advan=1, trans=2)
        .pk("CL = THETA(1) * EXP(ETA(1))\nV  = THETA(2) * EXP(ETA(2))")
        .error("IPRED = F\nW = IPRED * THETA(3)\nY = IPRED + W * EPS(1)")
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 100),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega([kwargs.get("iiv_cl", 0.3), kwargs.get("iiv_v", 0.2)])
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def one_cmt_oral(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    One-compartment first-order absorption model (ADVAN2, TRANS2).

    Structural model: oral absorption into one compartment with first-order
    absorption (KA) and first-order elimination (CL, V).
    IIV on KA, CL, and V. Proportional residual error.

    THETA(1): KA — absorption rate constant [hr⁻¹]; default init=1.5
    THETA(2): CL — clearance [L/hr]; default init=0.1
    THETA(3): V  — volume of distribution [L]; default init=30.0
    THETA(4): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   ka_init (default 1.5), cl_init (default 0.1),
                   v_init (default 30.0),
                   iiv_ka (default 0.5), iiv_cl (default 0.3),
                   iiv_v (default 0.3).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("One-compartment oral (ADVAN2)")
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1) * EXP(ETA(1))\nCL = THETA(2) * EXP(ETA(2))\nV  = THETA(3) * EXP(ETA(3))")
        .error("IPRED = F\nW = IPRED * THETA(4)\nY = IPRED + W * EPS(1)")
        .theta(
            [
                (0.01, kwargs.get("ka_init", 1.5), 20),
                (0.001, kwargs.get("cl_init", 0.1), 10),
                (0.1, kwargs.get("v_init", 30.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_ka", 0.5),
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v", 0.3),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def two_cmt_iv(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Two-compartment IV model (ADVAN3, TRANS4).

    Structural model: central and peripheral compartment with IV dosing.
    Parameters: CL (clearance), V1 (central volume), Q (inter-compartmental
    clearance), V2 (peripheral volume).
    IIV on CL and V1. Proportional residual error.

    THETA(1): CL — systemic clearance [L/hr]; default init=5.0
    THETA(2): V1 — central volume [L]; default init=10.0
    THETA(3): Q  — inter-compartmental clearance [L/hr]; default init=2.0
    THETA(4): V2 — peripheral volume [L]; default init=30.0
    THETA(5): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v1_init (10.0), q_init (2.0),
                   v2_init (30.0), iiv_cl (0.3), iiv_v1 (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Two-compartment IV (ADVAN3, TRANS4)")
        .subroutines(advan=3, trans=4)
        .pk(
            "CL = THETA(1) * EXP(ETA(1))\nV1 = THETA(2) * EXP(ETA(2))\nQ  = THETA(3)\nV2 = THETA(4)"
        )
        .error("IPRED = F\nW = IPRED * THETA(5)\nY = IPRED + W * EPS(1)")
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v1_init", 10.0), 200),
                (0.01, kwargs.get("q_init", 2.0), 100),
                (0.1, kwargs.get("v2_init", 30.0), 1000),
                (0.01, 0.1, 2),
            ]
        )
        .omega([kwargs.get("iiv_cl", 0.3), kwargs.get("iiv_v1", 0.2)])
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def two_cmt_oral(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Two-compartment first-order absorption (ADVAN4, TRANS4).

    Structural model: oral absorption into a two-compartment system.
    Parameters: KA (absorption rate constant), CL (clearance), V2 (central
    volume), Q (inter-compartmental clearance), V3 (peripheral volume).
    IIV on KA, CL, and V2. Proportional residual error.

    THETA(1): KA — absorption rate constant [hr⁻¹]; default init=1.0
    THETA(2): CL — clearance [L/hr]; default init=5.0
    THETA(3): V2 — central volume [L]; default init=15.0
    THETA(4): Q  — inter-compartmental clearance [L/hr]; default init=2.0
    THETA(5): V3 — peripheral volume [L]; default init=40.0
    THETA(6): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   ka_init (1.0), cl_init (5.0), v2_init (15.0),
                   q_init (2.0), v3_init (40.0),
                   iiv_ka (0.4), iiv_cl (0.3), iiv_v2 (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Two-compartment oral (ADVAN4, TRANS4)")
        .subroutines(advan=4, trans=4)
        .pk(
            "KA = THETA(1) * EXP(ETA(1))\n"
            "CL = THETA(2) * EXP(ETA(2))\n"
            "V2 = THETA(3) * EXP(ETA(3))\n"
            "Q  = THETA(4)\n"
            "V3 = THETA(5)"
        )
        .error("IPRED = F\nW = IPRED * THETA(6)\nY = IPRED + W * EPS(1)")
        .theta(
            [
                (0.01, kwargs.get("ka_init", 1.0), 20),
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v2_init", 15.0), 500),
                (0.01, kwargs.get("q_init", 2.0), 100),
                (0.1, kwargs.get("v3_init", 40.0), 1000),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_ka", 0.4),
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v2", 0.2),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def three_cmt_iv(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Three-compartment IV model (ADVAN11, TRANS4).

    Structural model: central compartment plus two peripheral compartments,
    IV dosing only. Parameters: CL, V1, Q2, V2, Q3, V3.
    IIV on CL and V1. Proportional residual error.

    Falls back to ADVAN3 (two-compartment) if ADVAN11 is not registered
    in the PK subroutine registry.

    THETA(1): CL — systemic clearance [L/hr]; default init=5.0
    THETA(2): V1 — central volume [L]; default init=10.0
    THETA(3): Q2 — first inter-compartmental clearance [L/hr]; default init=2.0
    THETA(4): V2 — first peripheral volume [L]; default init=30.0
    THETA(5): Q3 — second inter-compartmental clearance [L/hr]; default init=1.0
    THETA(6): V3 — second peripheral volume [L]; default init=50.0
    THETA(7): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v1_init (10.0), q2_init (2.0),
                   v2_init (30.0), q3_init (1.0), v3_init (50.0),
                   iiv_cl (0.3), iiv_v1 (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    # Attempt ADVAN11 first; fall back to ADVAN3 if unavailable
    try:
        from openpkpd.pk import get_advan

        get_advan(11)
        advan_num = 11
    except (KeyError, NotImplementedError, ImportError):
        advan_num = 3

    if advan_num == 11:
        pk_code = (
            "CL = THETA(1) * EXP(ETA(1))\n"
            "V1 = THETA(2) * EXP(ETA(2))\n"
            "Q2 = THETA(3)\n"
            "V2 = THETA(4)\n"
            "Q3 = THETA(5)\n"
            "V3 = THETA(6)"
        )
        theta_specs: list[Any] = [
            (0.1, kwargs.get("cl_init", 5.0), 200),
            (0.1, kwargs.get("v1_init", 10.0), 200),
            (0.01, kwargs.get("q2_init", 2.0), 100),
            (0.1, kwargs.get("v2_init", 30.0), 1000),
            (0.01, kwargs.get("q3_init", 1.0), 100),
            (0.1, kwargs.get("v3_init", 50.0), 2000),
            (0.01, 0.1, 2),
        ]
    else:
        # ADVAN3 fallback (two-compartment approximation)
        pk_code = (
            "CL = THETA(1) * EXP(ETA(1))\nV1 = THETA(2) * EXP(ETA(2))\nQ  = THETA(3)\nV2 = THETA(4)"
        )
        theta_specs = [
            (0.1, kwargs.get("cl_init", 5.0), 200),
            (0.1, kwargs.get("v1_init", 10.0), 200),
            (0.01, kwargs.get("q2_init", 2.0), 100),
            (0.1, kwargs.get("v2_init", 30.0), 1000),
            (0.01, 0.1, 2),
        ]

    builder = (
        ModelBuilder()
        .problem(f"Three-compartment IV (ADVAN{advan_num}, TRANS4)")
        .subroutines(advan=advan_num, trans=4)
        .pk(pk_code)
        .error(
            "IPRED = F\nW = IPRED * THETA(7)\nY = IPRED + W * EPS(1)"
            if advan_num == 11
            else ("IPRED = F\nW = IPRED * THETA(5)\nY = IPRED + W * EPS(1)")
        )
        .theta(theta_specs)
        .omega([kwargs.get("iiv_cl", 0.3), kwargs.get("iiv_v1", 0.2)])
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


# ── Pharmacodynamic Models ────────────────────────────────────────────────────


def emax_direct(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Direct Emax PD model coupled to 1-compartment IV PK.

    Structural model: 1-cmt IV PK (ADVAN1, TRANS2) with direct Emax PD
    response. Drug effect is computed in the $ERROR block from the
    individual predicted concentration.

    E = EMAX * C / (EC50 + C)

    where C = F (individual predicted concentration from ADVAN1).

    THETA(1): CL   — clearance [L/hr]; default init=5.0
    THETA(2): V    — volume [L]; default init=20.0
    THETA(3): EMAX — maximum effect; default init=10.0
    THETA(4): EC50 — concentration for 50% effect; default init=5.0
    THETA(5): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v_init (20.0), emax_init (10.0),
                   ec50_init (5.0), iiv_cl (0.3), iiv_v (0.2),
                   iiv_emax (0.2), iiv_ec50 (0.3).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Direct Emax PD (ADVAN1 + Emax in ERROR)")
        .subroutines(advan=1, trans=2)
        .pk(
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "EMAX = THETA(3) * EXP(ETA(3))\n"
            "EC50 = THETA(4) * EXP(ETA(4))"
        )
        .error(
            "C    = F\n"
            "IPRED = EMAX * C / (EC50 + C)\n"
            "W    = IPRED * THETA(5)\n"
            "Y    = IPRED + W * EPS(1)"
        )
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.01, kwargs.get("emax_init", 10.0), 1000),
                (0.01, kwargs.get("ec50_init", 5.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v", 0.2),
                kwargs.get("iiv_emax", 0.2),
                kwargs.get("iiv_ec50", 0.3),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def sigmoid_emax(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Sigmoidal Emax (Hill) PD model coupled to 1-compartment IV PK.

    Structural model: 1-cmt IV PK with sigmoidal Emax PD response.
    The Hill coefficient (n) controls the steepness of the sigmoidal curve.

    E = EMAX * C^n / (EC50^n + C^n)

    THETA(1): CL   — clearance [L/hr]; default init=5.0
    THETA(2): V    — volume [L]; default init=20.0
    THETA(3): EMAX — maximum effect; default init=10.0
    THETA(4): EC50 — concentration for 50% effect; default init=5.0
    THETA(5): n    — Hill coefficient (sigmoidicity); default init=2.0
    THETA(6): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v_init (20.0), emax_init (10.0),
                   ec50_init (5.0), hill_init (2.0),
                   iiv_cl (0.3), iiv_v (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Sigmoidal Emax (Hill) PD (ADVAN1)")
        .subroutines(advan=1, trans=2)
        .pk(
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "EMAX = THETA(3)\n"
            "EC50 = THETA(4)\n"
            "N    = THETA(5)"
        )
        .error(
            "C    = F\n"
            "CN   = C**N\n"
            "IPRED = EMAX * CN / (EC50**N + CN)\n"
            "W    = IPRED * THETA(6)\n"
            "Y    = IPRED + W * EPS(1)"
        )
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.01, kwargs.get("emax_init", 10.0), 1000),
                (0.01, kwargs.get("ec50_init", 5.0), 500),
                (0.1, kwargs.get("hill_init", 2.0), 10),
                (0.01, 0.1, 2),
            ]
        )
        .omega([kwargs.get("iiv_cl", 0.3), kwargs.get("iiv_v", 0.2)])
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def inhibitory_emax(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Inhibitory Emax model coupled to 1-compartment IV PK.

    Structural model: 1-cmt IV PK with inhibitory Emax PD response.
    Drug inhibits a baseline response (E0) in a saturable manner.

    E = E0 * (1 - IMAX * C / (IC50 + C))

    THETA(1): CL   — clearance [L/hr]; default init=5.0
    THETA(2): V    — volume [L]; default init=20.0
    THETA(3): E0   — baseline effect; default init=10.0
    THETA(4): IMAX — maximum inhibition fraction (0-1); default init=0.8
    THETA(5): IC50 — concentration for 50% inhibition; default init=5.0
    THETA(6): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v_init (20.0), e0_init (10.0),
                   imax_init (0.8), ic50_init (5.0),
                   iiv_cl (0.3), iiv_v (0.2), iiv_e0 (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Inhibitory Emax PD (ADVAN1)")
        .subroutines(advan=1, trans=2)
        .pk(
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "E0   = THETA(3) * EXP(ETA(3))\n"
            "IMAX = THETA(4)\n"
            "IC50 = THETA(5)"
        )
        .error(
            "C    = F\n"
            "IPRED = E0 * (1 - IMAX * C / (IC50 + C))\n"
            "W    = IPRED * THETA(6)\n"
            "Y    = IPRED + W * EPS(1)"
        )
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.01, kwargs.get("e0_init", 10.0), 500),
                (0.001, kwargs.get("imax_init", 0.8), 1.0),
                (0.01, kwargs.get("ic50_init", 5.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v", 0.2),
                kwargs.get("iiv_e0", 0.2),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def indirect_response_type_i(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Indirect response model Type I: drug inhibits input (Kin).

    Structural model: 1-cmt IV PK coupled to an indirect response PD model
    where the drug inhibits the zero-order production rate constant (Kin).
    The response (R) follows:

        dR/dt = Kin * (1 - IMAX * C / (IC50 + C)) - Kout * R

    Baseline response at steady state: R0 = Kin / Kout.

    Uses ADVAN6 (ODE solver) if available. Falls back to ADVAN1 with a
    simplified algebraic approximation of the effect.

    THETA(1): CL   — PK clearance [L/hr]; default init=5.0
    THETA(2): V    — PK volume [L]; default init=20.0
    THETA(3): Kin  — zero-order production rate; default init=1.0
    THETA(4): Kout — first-order loss rate constant; default init=0.1
    THETA(5): IMAX — maximum inhibition fraction; default init=0.8
    THETA(6): IC50 — inhibitory concentration 50%; default init=5.0
    THETA(7): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v_init (20.0), kin_init (1.0),
                   kout_init (0.1), imax_init (0.8), ic50_init (5.0),
                   iiv_cl (0.3), iiv_v (0.2), iiv_kout (0.2).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    # Try ADVAN6 (ODE); fall back to ADVAN1 if not registered
    try:
        from openpkpd.pk import get_advan

        get_advan(6)
        advan_num = 6
    except (KeyError, NotImplementedError, ImportError):
        advan_num = 1

    if advan_num == 6:
        pk_code = (
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "KIN  = THETA(3)\n"
            "KOUT = THETA(4) * EXP(ETA(3))\n"
            "IMAX = THETA(5)\n"
            "IC50 = THETA(6)\n"
            "R0   = KIN / KOUT\n"
            "A_0(2) = R0"
        )
        error_code = (
            "C    = A(1) / V\nIPRED = A(2)\nW    = IPRED * THETA(7)\nY    = IPRED + W * EPS(1)"
        )
    else:
        # ADVAN1 fallback: algebraic pseudo-steady-state approximation
        pk_code = (
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "KIN  = THETA(3)\n"
            "KOUT = THETA(4) * EXP(ETA(3))\n"
            "IMAX = THETA(5)\n"
            "IC50 = THETA(6)"
        )
        error_code = (
            "C    = F\n"
            "R0   = KIN / KOUT\n"
            "IPRED = R0 * (1 - IMAX * C / (IC50 + C))\n"
            "W    = IPRED * THETA(7)\n"
            "Y    = IPRED + W * EPS(1)"
        )

    builder = (
        ModelBuilder()
        .problem(f"Indirect Response Type I (ADVAN{advan_num})")
        .subroutines(advan=advan_num, trans=2)
        .pk(pk_code)
        .error(error_code)
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.001, kwargs.get("kin_init", 1.0), 100),
                (0.001, kwargs.get("kout_init", 0.1), 10),
                (0.001, kwargs.get("imax_init", 0.8), 1.0),
                (0.01, kwargs.get("ic50_init", 5.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v", 0.2),
                kwargs.get("iiv_kout", 0.2),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


def effect_compartment(
    data_path: str | None = None,
    method: str = "FOCE",
    **kwargs: Any,
) -> ModelBuilder:
    """
    Effect compartment (link) model: 1-compartment IV PK plus effect compartment.

    Structural model: 1-cmt IV PK (central compartment) linked to an effect
    compartment via a first-order equilibration rate constant (Ke0). The drug
    effect is proportional to the effect compartment concentration (Ce).

    E = EMAX * Ce / (EC50 + Ce)

    Effect compartment dynamics (negligible volume):
        dCe/dt = Ke0 * (C - Ce)

    where C is the central compartment concentration.

    Uses ADVAN1 for PK; effect compartment modeled analytically in $ERROR.

    THETA(1): CL   — clearance [L/hr]; default init=5.0
    THETA(2): V    — volume [L]; default init=20.0
    THETA(3): Ke0  — effect compartment equilibration rate [hr⁻¹]; default init=0.5
    THETA(4): EMAX — maximum effect; default init=10.0
    THETA(5): EC50 — effect compartment concentration for 50% effect; default init=5.0
    THETA(6): proportional residual error CV; default init=0.1

    Args:
        data_path: Path to NONMEM-format CSV data file (optional).
        method:    Estimation method (default 'FOCE').
        **kwargs:  Override initial estimates:
                   cl_init (5.0), v_init (20.0), ke0_init (0.5),
                   emax_init (10.0), ec50_init (5.0),
                   iiv_cl (0.3), iiv_v (0.2), iiv_ke0 (0.3).

    Returns:
        Configured ModelBuilder (call .build().fit() to estimate).
    """
    builder = (
        ModelBuilder()
        .problem("Effect compartment link model (ADVAN1 + Ke0)")
        .subroutines(advan=1, trans=2)
        .pk(
            "CL   = THETA(1) * EXP(ETA(1))\n"
            "V    = THETA(2) * EXP(ETA(2))\n"
            "KE0  = THETA(3) * EXP(ETA(3))\n"
            "EMAX = THETA(4)\n"
            "EC50 = THETA(5)"
        )
        .error(
            "C    = F\n"
            "CE   = C * KE0 / (KE0 + CL / V)\n"
            "IPRED = EMAX * CE / (EC50 + CE)\n"
            "W    = IPRED * THETA(6)\n"
            "Y    = IPRED + W * EPS(1)"
        )
        .theta(
            [
                (0.1, kwargs.get("cl_init", 5.0), 200),
                (0.1, kwargs.get("v_init", 20.0), 500),
                (0.001, kwargs.get("ke0_init", 0.5), 20),
                (0.01, kwargs.get("emax_init", 10.0), 1000),
                (0.01, kwargs.get("ec50_init", 5.0), 500),
                (0.01, 0.1, 2),
            ]
        )
        .omega(
            [
                kwargs.get("iiv_cl", 0.3),
                kwargs.get("iiv_v", 0.2),
                kwargs.get("iiv_ke0", 0.3),
            ]
        )
        .sigma([1.0])
        .estimation(method=method, interaction=True)
    )
    if data_path:
        builder = builder.data(data_path)
    return builder


# ── Library management ────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, Any] = {
    "one_cmt_iv": one_cmt_iv,
    "one_cmt_oral": one_cmt_oral,
    "two_cmt_iv": two_cmt_iv,
    "two_cmt_oral": two_cmt_oral,
    "three_cmt_iv": three_cmt_iv,
    "emax_direct": emax_direct,
    "sigmoid_emax": sigmoid_emax,
    "inhibitory_emax": inhibitory_emax,
    "indirect_response_type_i": indirect_response_type_i,
    "effect_compartment": effect_compartment,
}


def list_models() -> list[str]:
    """
    Return a sorted list of available model names.

    Returns:
        Sorted list of model name strings that can be passed to get_model().

    Example:
        >>> from openpkpd.library import list_models
        >>> print(list_models())
        ['effect_compartment', 'emax_direct', ...]
    """
    return sorted(_MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs: Any) -> ModelBuilder:
    """
    Get a pre-built model by name and return a configured ModelBuilder.

    Args:
        name:     Model name (see list_models() for available options).
        **kwargs: Keyword arguments forwarded to the model factory function
                  (e.g., data_path, method, cl_init, v_init, etc.).

    Returns:
        Configured ModelBuilder ready for .build().fit().

    Raises:
        KeyError: If the model name is not found in the registry.

    Example:
        >>> model = get_model("one_cmt_oral", cl_init=0.1, v_init=30.0)
        >>> result = model.data("data.csv").build().fit()
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


def show_model(name: str) -> str:
    """
    Return the documentation string for a named model.

    Args:
        name: Model name (see list_models() for available options).

    Returns:
        Docstring of the model factory function, or an informative message
        if the model is unknown or has no docstring.

    Example:
        >>> from openpkpd.library import show_model
        >>> print(show_model("one_cmt_oral"))
    """
    fn = _MODEL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown model: {name!r}. Available: {list_models()}"
    return fn.__doc__ or f"No documentation available for {name!r}."
