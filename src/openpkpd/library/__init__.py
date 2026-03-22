"""
openpkpd.library — Pre-built PK/PD model library.

Factory functions that return configured ModelBuilder instances for common
pharmacokinetic and pharmacodynamic models. All functions accept an optional
data_path keyword argument and support parameter override via **kwargs.

Usage:
    from openpkpd.library import one_cmt_oral, list_models, get_model

    # Get a one-compartment oral model
    model = one_cmt_oral(data_path="theo.csv")
    result = model.build().fit()

    # Browse the library
    print(list_models())

    # Fetch by name (useful for scripting/automation)
    model = get_model("emax_direct", emax_init=20.0)
"""

from openpkpd.library.models import (
    effect_compartment,
    emax_direct,
    get_model,
    indirect_response_type_i,
    inhibitory_emax,
    list_models,
    one_cmt_iv,
    one_cmt_oral,
    show_model,
    sigmoid_emax,
    three_cmt_iv,
    two_cmt_iv,
    two_cmt_oral,
)

__all__ = [
    "one_cmt_iv",
    "one_cmt_oral",
    "two_cmt_iv",
    "two_cmt_oral",
    "three_cmt_iv",
    "emax_direct",
    "sigmoid_emax",
    "inhibitory_emax",
    "indirect_response_type_i",
    "effect_compartment",
    "list_models",
    "get_model",
    "show_model",
]
