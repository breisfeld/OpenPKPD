"""
I/O utilities for importing models from external formats.

Supported formats
-----------------
SBML (Systems Biology Markup Language)
    Import via :func:`openpkpd.io.sbml.load_sbml`.
    Requires the ``python-libsbml`` package::

        pip install python-libsbml

    Returns a :class:`~openpkpd.io.sbml.SBMLModel` containing a DES callable
    and initial parameter guesses compatible with the ADVAN6/DDESubroutine
    integration pipeline.
"""

from openpkpd.io.sbml import SBMLModel, load_sbml

__all__ = ["load_sbml", "SBMLModel"]
