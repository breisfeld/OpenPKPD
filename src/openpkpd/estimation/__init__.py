"""Estimation method routing."""

from __future__ import annotations

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.estimation.bayes import BAYESMethod, BayesianResult
from openpkpd.estimation import mcmc_diagnostics
from openpkpd.estimation.fo import FOMethod
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.imp import IMPMethod
from openpkpd.estimation.laplacian import LaplacianMethod
from openpkpd.estimation.nonparametric import NonparametricMethod, NonparametricResult
from openpkpd.estimation.saem import SAEMMethod
from openpkpd.utils.constants import Method
from openpkpd.utils.errors import EstimationError


def get_estimation_method(
    method: str,
    interaction: bool = False,
    **kwargs,
) -> EstimationMethod:
    """
    Return an estimation method instance for the given method name.

    Args:
        method:      Method name (FO, FOCE, FOCEI, LAPLACIAN, SAEM, IMP,
                     IMPMAP, BAYES, NONPARAMETRIC, NONPARM, NP).
        interaction: Enable FOCEI interaction term (applies to FOCE only).
        **kwargs:    Method-specific keyword arguments passed to the
                     constructor of the selected EstimationMethod subclass.

    Returns:
        EstimationMethod instance ready to call .estimate().

    Raises:
        EstimationError: If the method name is not recognised.
    """
    m = method.upper()
    if m == Method.FO:
        kwargs.pop("n_parallel", None)
        kwargs.pop("iteration_callback", None)
        return FOMethod(**kwargs)
    elif m in (Method.FOCE, Method.FOCEI):
        interact = interaction or (m == Method.FOCEI)
        return FOCEMethod(interaction=interact, **kwargs)
    elif m == Method.LAPLACIAN:
        kwargs.pop("n_parallel", None)
        kwargs.pop("iteration_callback", None)
        return LaplacianMethod(**kwargs)
    elif m == Method.SAEM:
        kwargs.pop("maxeval", None)
        kwargs.pop("n_parallel", None)
        return SAEMMethod(**kwargs)
    elif m == Method.IMP:
        return IMPMethod(is_map=False, **kwargs)
    elif m == Method.IMPMAP:
        return IMPMethod(is_map=True, **kwargs)
    elif m == Method.BAYES:
        kwargs.pop("n_parallel", None)
        kwargs.pop("iteration_callback", None)
        kwargs["interaction"] = interaction
        return BAYESMethod(**kwargs)
    elif m in (Method.NONPARAMETRIC, "NONPARM", "NP"):
        kwargs.pop("n_parallel", None)
        kwargs.pop("iteration_callback", None)
        return NonparametricMethod(**kwargs)
    else:
        raise EstimationError(
            f"Unknown estimation method: {method!r}. "
            f"Supported: FO, FOCE, FOCEI, LAPLACIAN, SAEM, IMP, IMPMAP, "
            f"BAYES, NONPARAMETRIC, NONPARM, NP"
        )


__all__ = [
    "EstimationMethod",
    "EstimationResult",
    "FOMethod",
    "FOCEMethod",
    "LaplacianMethod",
    "SAEMMethod",
    "IMPMethod",
    "BAYESMethod",
    "BayesianResult",
    "mcmc_diagnostics",
    "NonparametricMethod",
    "NonparametricResult",
    "get_estimation_method",
]
