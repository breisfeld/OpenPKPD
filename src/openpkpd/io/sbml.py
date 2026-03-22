"""
SBML model importer for OpenPKPD.

Converts a Systems Biology Markup Language (SBML) model into a DES callable
and parameter set compatible with the ADVAN6/DDESubroutine ODE integration
pipeline.

Requirements
------------
``python-libsbml`` must be installed::

    pip install python-libsbml

Supported SBML features
-----------------------
- Species (state variables → compartment amounts ``A[n]``)
- Parameters (→ ThetaSpec initial estimates)
- Compartment volumes (used for concentration → amount conversion)
- Reactions with MathML kinetic laws evaluated via ``libsbml.formulaToString``
  and then compiled to Python

Limitations
-----------
- Events, rules, and constraints are not yet supported (they will be silently
  skipped with a warning).
- Only SBML Level 2/3 is supported.
- Kinetic laws must be expressible as pure Python arithmetic after substituting
  species and parameter names.

Usage
-----
::

    from openpkpd.io import load_sbml
    from openpkpd.pk.ode.advan6 import ADVAN6

    model = load_sbml("tumor_growth.xml")
    print(model.species_names)         # ['A_tumor', 'A_drug', ...]
    print(model.parameter_names)       # ['kgrow', 'kdrug', ...]

    # Use directly with ADVAN6
    advan = ADVAN6(n_compartments=model.n_compartments)
    pk_solution = advan.solve(
        pk_params=model.default_pk_params,
        dose_events=dose_events,
        obs_times=obs_times,
        des_callable=model.des_callable,
    )

    # Or build ThetaSpecs for estimation
    from openpkpd.model.parameters import ParameterSet
    params = ParameterSet.from_specs(model.to_theta_specs(), [], [])
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SBMLModel:
    """
    Parsed SBML model ready for use with OpenPKPD estimators.

    Attributes
    ----------
    species_names : list[str]
        Ordered list of species IDs (index n → compartment A[n]).
    parameter_names : list[str]
        Ordered list of parameter IDs (keys in ``default_pk_params``).
    default_pk_params : dict[str, float]
        Parameter values extracted from the SBML file (initial estimates).
    initial_amounts : dict[str, float]
        Initial species amounts (amount, not concentration).
    n_compartments : int
        Number of ODE state variables (== ``len(species_names)``).
    des_callable : Callable
        DES function with signature
        ``(t, A_list, pk_params, theta, eta) -> dAdt_list``.
        Compatible with :class:`~openpkpd.pk.ode.advan6.ADVAN6`.
    source_path : str
        Path to the original SBML file (for reference).
    warnings : list[str]
        Non-fatal issues encountered during parsing.
    """

    species_names: list[str]
    parameter_names: list[str]
    default_pk_params: dict[str, float]
    initial_amounts: dict[str, float]
    n_compartments: int
    des_callable: Callable
    source_path: str = ""
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------

    def to_theta_specs(self) -> list[Any]:
        """
        Build a list of :class:`~openpkpd.model.parameters.ThetaSpec` objects.

        Each free parameter in the SBML model becomes one THETA.
        Initial estimates come from ``default_pk_params``; lower bounds are
        set to ``0`` for positive quantities and ``-1e6`` otherwise.

        Returns
        -------
        list[ThetaSpec]
        """
        from openpkpd.model.parameters import ThetaSpec

        specs = []
        for name in self.parameter_names:
            init = self.default_pk_params.get(name, 1.0)
            lower = 0.0 if init >= 0 else -1e6
            specs.append(ThetaSpec(init=float(init), lower=lower, label=name))
        return specs

    def pk_callable_from_theta(self, theta: list[float]) -> dict[str, float]:
        """
        Map a THETA vector (in the order of ``parameter_names``) back to a
        ``pk_params`` dict.

        Parameters
        ----------
        theta : list[float]
            Estimated THETA values in the same order as ``parameter_names``.

        Returns
        -------
        dict[str, float]
        """
        pk = dict(zip(self.parameter_names, theta, strict=False))
        return pk


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_sbml(path: str) -> SBMLModel:
    """
    Load an SBML file and return an :class:`SBMLModel`.

    Parameters
    ----------
    path : str
        Path to the ``.xml`` or ``.sbml`` file.

    Returns
    -------
    SBMLModel

    Raises
    ------
    ImportError
        If ``python-libsbml`` is not installed.
    ValueError
        If the SBML file cannot be parsed or is empty.
    """
    try:
        import libsbml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "python-libsbml is required for SBML import. "
            "Install it with: pip install python-libsbml"
        ) from exc

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(path)

    n_errors = doc.getNumErrors()
    fatal = [
        doc.getError(i).getMessage()
        for i in range(n_errors)
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    ]
    if fatal:
        raise ValueError(f"SBML file {path!r} has fatal errors:\n" + "\n".join(fatal))

    model = doc.getModel()
    if model is None:
        raise ValueError(f"No SBML model found in {path!r}")

    parse_warnings: list[str] = []
    if n_errors > 0:
        for i in range(n_errors):
            err = doc.getError(i)
            if err.getSeverity() < libsbml.LIBSBML_SEV_ERROR:
                parse_warnings.append(f"[SBML warning] {err.getMessage()}")

    # -- Compartment volumes -----------------------------------------------
    compartment_volumes: dict[str, float] = {}
    for k in range(model.getNumCompartments()):
        cpt = model.getCompartment(k)
        vol = cpt.getVolume() if cpt.isSetVolume() else 1.0
        compartment_volumes[cpt.getId()] = float(vol)

    # -- Parameters --------------------------------------------------------
    parameter_names: list[str] = []
    default_pk_params: dict[str, float] = {}
    for k in range(model.getNumParameters()):
        p = model.getParameter(k)
        pid = p.getId()
        val = p.getValue() if p.isSetValue() else 1.0
        parameter_names.append(pid)
        default_pk_params[pid] = float(val)

    # -- Species (state variables) ----------------------------------------
    species_names: list[str] = []
    initial_amounts: dict[str, float] = {}
    species_compartment: dict[str, str] = {}  # species_id → compartment_id

    for k in range(model.getNumSpecies()):
        sp = model.getSpecies(k)
        sid = sp.getId()
        species_names.append(sid)
        species_compartment[sid] = sp.getCompartment()

        if sp.isSetInitialAmount():
            initial_amounts[sid] = float(sp.getInitialAmount())
        elif sp.isSetInitialConcentration():
            vol = compartment_volumes.get(sp.getCompartment(), 1.0)
            initial_amounts[sid] = float(sp.getInitialConcentration()) * vol
        else:
            initial_amounts[sid] = 0.0

    n_compartments = len(species_names)
    if n_compartments == 0:
        raise ValueError(f"SBML model {path!r} has no species.")

    # Index map: species_id → index in A list
    species_index: dict[str, int] = {sid: i for i, sid in enumerate(species_names)}

    # -- Reactions → DES right-hand side -----------------------------------
    # For each reaction, parse the kinetic law and map to dA/dt contributions.
    # Stoichiometry: +1 for products, -1 for reactants.

    # Build contribution map: species_index → list of kinetic law strings
    dadt_exprs: dict[int, list[str]] = {i: [] for i in range(n_compartments)}

    for r_idx in range(model.getNumReactions()):
        rxn = model.getReaction(r_idx)
        kl = rxn.getKineticLaw()
        if kl is None:
            parse_warnings.append(f"Reaction {rxn.getId()!r} has no kinetic law — skipped.")
            continue

        math_ast = kl.getMath()
        formula_str = libsbml.formulaToL3String(math_ast) if math_ast else ""

        # Convert formula to Python-compatible expression
        py_formula = _sbml_formula_to_python(formula_str, species_index, parameter_names)

        # Reactants (consume)
        for sr_idx in range(rxn.getNumReactants()):
            sr = rxn.getReactant(sr_idx)
            sid = sr.getSpecies()
            if sid in species_index:
                stoich = sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0
                sign = "-" if stoich > 0 else "+"
                term = f"{sign}({abs(stoich)} * ({py_formula}))"
                dadt_exprs[species_index[sid]].append(term)

        # Products (produce)
        for sp_idx in range(rxn.getNumProducts()):
            sp = rxn.getProduct(sp_idx)
            sid = sp.getSpecies()
            if sid in species_index:
                stoich = sp.getStoichiometry() if sp.isSetStoichiometry() else 1.0
                sign = "+" if stoich > 0 else "-"
                term = f"{sign}({abs(stoich)} * ({py_formula}))"
                dadt_exprs[species_index[sid]].append(term)

    # Warn on unsupported model elements
    if model.getNumRules() > 0:
        parse_warnings.append(
            f"SBML model has {model.getNumRules()} rule(s) — rules are not yet supported and will be ignored."
        )
    if model.getNumEvents() > 0:
        parse_warnings.append(
            f"SBML model has {model.getNumEvents()} event(s) — events are not yet supported and will be ignored."
        )
    if model.getNumConstraints() > 0:
        parse_warnings.append(
            f"SBML model has {model.getNumConstraints()} constraint(s) — constraints are not yet supported and will be ignored."
        )

    for w in parse_warnings:
        warnings.warn(w, UserWarning, stacklevel=3)

    # -- Build DES callable ------------------------------------------------
    des_callable = _build_des_callable(
        dadt_exprs, species_names, species_index, parameter_names, n_compartments, parse_warnings
    )

    return SBMLModel(
        species_names=species_names,
        parameter_names=parameter_names,
        default_pk_params=default_pk_params,
        initial_amounts=initial_amounts,
        n_compartments=n_compartments,
        des_callable=des_callable,
        source_path=str(path),
        warnings=parse_warnings,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sbml_formula_to_python(
    formula: str,
    species_index: dict[str, int],
    parameter_names: list[str],
) -> str:
    """
    Convert an SBML L3 formula string to a Python expression.

    Species references like ``A_tumor`` become ``A[0]``.
    Parameter references become ``pk_params['A_tumor']``.
    SBML math functions are mapped to Python equivalents.
    """
    if not formula:
        return "0.0"

    result = formula

    # Replace SBML math functions with Python equivalents
    _SBML_FN_MAP = [
        (r"\bpower\s*\(", "pow("),
        (r"\bsqrt\s*\(", "__import__('math').sqrt("),
        (r"\bexp\s*\(", "__import__('math').exp("),
        (r"\blog\s*\(", "__import__('math').log("),
        (r"\babs\s*\(", "abs("),
        (r"\bfloor\s*\(", "__import__('math').floor("),
        (r"\bceiling\s*\(", "__import__('math').ceil("),
    ]
    for pattern, replacement in _SBML_FN_MAP:
        result = re.sub(pattern, replacement, result)

    # Replace species names with A[index] (longest names first to avoid
    # partial replacements of names that are substrings of other names)
    for sid in sorted(species_index.keys(), key=len, reverse=True):
        idx = species_index[sid]
        result = re.sub(r"\b" + re.escape(sid) + r"\b", f"A[{idx}]", result)

    # Replace parameter names with pk_params lookups
    for pname in sorted(parameter_names, key=len, reverse=True):
        result = re.sub(
            r"\b" + re.escape(pname) + r"\b",
            f"pk_params['{pname}']",
            result,
        )

    # Replace SBML boolean/comparison operators
    result = result.replace(" && ", " and ").replace(" || ", " or ")
    result = result.replace("^", "**")

    return result


def _build_des_callable(
    dadt_exprs: dict[int, list[str]],
    species_names: list[str],
    species_index: dict[str, int],
    parameter_names: list[str],
    n_compartments: int,
    parse_warnings: list[str],
) -> Callable:
    """
    Compile DES expressions into a callable with the standard signature:
        des_callable(t, A_list, pk_params, theta, eta) -> dAdt_list
    """
    import math

    # Build Python function body
    lines = [
        "def _sbml_des(t, A, pk_params, theta, eta):",
        f"    _n = {n_compartments}",
        "    dAdt = [0.0] * _n",
    ]

    for i in range(n_compartments):
        exprs = dadt_exprs.get(i, [])
        if exprs:
            combined = " ".join(exprs)
            lines.append(f"    dAdt[{i}] = {combined}")

    lines.append("    return dAdt")

    fn_code = "\n".join(lines)

    # Compile
    try:
        globs: dict[str, Any] = {"math": math, "__import__": __import__}
        exec(compile(fn_code, "<sbml_des>", "exec"), globs)  # noqa: S102
        fn = globs["_sbml_des"]
    except SyntaxError as exc:
        raise ValueError(
            f"Generated DES code has a syntax error: {exc}\n\nCode:\n{fn_code}"
        ) from exc

    return fn
