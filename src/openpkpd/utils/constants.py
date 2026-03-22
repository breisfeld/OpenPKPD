"""Mathematical, physical, and NONMEM convention constants."""

from __future__ import annotations

import math

# ── Mathematical constants ────────────────────────────────────────────────────
LOG2PI: float = math.log(2 * math.pi)
SQRT2PI: float = math.sqrt(2 * math.pi)
INF: float = float("inf")
NEG_INF: float = float("-inf")

# ── EVID (Event ID) codes ─────────────────────────────────────────────────────
EVID_OBS: int = 0  # Observation record
EVID_DOSE: int = 1  # Dosing event (AMT, RATE, etc.)
EVID_OTHER: int = 2  # Other type (e.g. covariate change)
EVID_RESET: int = 3  # Reset all compartments to 0
EVID_RESET_DOSE: int = 4  # Reset then dose (SS entry)

# ── MDV (Missing Dependent Variable) ─────────────────────────────────────────
MDV_OBS: int = 0  # Observed (used in OFV)
MDV_MISSING: int = 1  # Missing / not used in OFV


# ── Default column names (NONMEM conventions) ────────────────────────────────
class Columns:
    ID: str = "ID"
    TIME: str = "TIME"
    DV: str = "DV"
    AMT: str = "AMT"
    RATE: str = "RATE"
    EVID: str = "EVID"
    MDV: str = "MDV"
    CMT: str = "CMT"
    ADDL: str = "ADDL"
    II: str = "II"
    SS: str = "SS"
    BLQ: str = "BLQ"
    LLOQ: str = "LLOQ"
    DUR: str = "DUR"

    # Common covariate names
    WT: str = "WT"
    AGE: str = "AGE"
    SEX: str = "SEX"
    HT: str = "HT"
    CLCR: str = "CLCR"

    # Derived / computed columns
    PRED: str = "PRED"
    IPRED: str = "IPRED"
    RES: str = "RES"
    IRES: str = "IRES"
    WRES: str = "WRES"
    IWRES: str = "IWRES"
    CWRES: str = "CWRES"
    ETA_PREFIX: str = "ETA"


# ── Tolerance and convergence defaults ───────────────────────────────────────
DEFAULT_FOCE_RELTOL: float = 1e-6
DEFAULT_FOCE_ABSTOL: float = 1e-8
DEFAULT_INNER_MAXITER: int = 200
DEFAULT_OUTER_MAXITER: int = 9999
DEFAULT_SIGDIG: int = 3
DEFAULT_SIGL: int = 6

# ── NONMEM special values ─────────────────────────────────────────────────────
NONMEM_MISSING: float = -99.0  # Default missing data indicator
NONMEM_SKIP: str = "SKIP"  # Skip column indicator

# ── Physical constants ────────────────────────────────────────────────────────
# Used in some PBPK models; standard SI units
BODY_SURFACE_AREA_REF: float = 1.73  # m^2 (reference BSA for allometric scaling)
WEIGHT_REF: float = 70.0  # kg (reference weight for allometric scaling)


# ── Estimation method names ───────────────────────────────────────────────────
class Method:
    FO: str = "FO"
    FOCE: str = "FOCE"
    FOCEI: str = "FOCEI"
    LAPLACIAN: str = "LAPLACIAN"
    SAEM: str = "SAEM"
    IMP: str = "IMP"
    IMPMAP: str = "IMPMAP"
    BAYES: str = "BAYES"
    NUTS: str = "NUTS"
    NONPARAMETRIC: str = "NONPARAMETRIC"


# ── ADVAN numbers ──────────────────────────────────────────────────────────────
class ADVAN:
    ADVAN1: int = 1
    ADVAN2: int = 2
    ADVAN3: int = 3
    ADVAN4: int = 4
    ADVAN5: int = 5
    ADVAN6: int = 6
    ADVAN7: int = 7
    ADVAN8: int = 8
    ADVAN9: int = 9
    ADVAN10: int = 10
    ADVAN11: int = 11
    ADVAN12: int = 12
    ADVAN13: int = 13


# ── TRANS numbers ─────────────────────────────────────────────────────────────
class TRANS:
    TRANS1: int = 1  # Identity (user micro constants)
    TRANS2: int = 2  # CL, V  → K, V
    TRANS3: int = 3  # CL, Vss, Q → micro rates
    TRANS4: int = 4  # CL, V1, Q, V2 → micro rates
    TRANS5: int = 5  # 3-cmt micro rate constants
    TRANS6: int = 6  # 3-cmt CL/V parameterization


# ── BLQ/LLOQ handling methods ─────────────────────────────────────────────
class BLQMethod:
    """
    Constants for Below Limit of Quantification (BLQ) handling methods.

    Implements methods M1–M7 following Beal (2001) and Ahn et al. (2008).
    """

    M1: str = "M1"  # Exclude BLQ observations (MDV=1)
    M2: str = "M2"  # Likelihood censored at LLOQ
    M3: str = "M3"  # Full likelihood-based censoring: P(Y < LLOQ)
    M4: str = "M4"  # M3 + constraint Y >= 0 (truncated normal)
    M5: str = "M5"  # Replace BLQ with LLOQ/2
    M6: str = "M6"  # Replace first BLQ with LLOQ/2, discard rest
    M7: str = "M7"  # Replace BLQ with 0
