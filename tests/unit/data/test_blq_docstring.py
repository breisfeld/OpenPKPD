"""Tests for BLQ M3 docstring conditional independence note — D5."""

from __future__ import annotations

import openpkpd.data.blq as blq_module
from openpkpd.data.blq import blq_log_likelihood


class TestM3DocstringNote:
    def test_m3_conditional_independence_in_function_docstring(self):
        """'conditionally independent' appears in blq_log_likelihood docstring."""
        assert blq_log_likelihood.__doc__ is not None, (
            "blq_log_likelihood must have a docstring"
        )
        assert "conditionally independent" in blq_log_likelihood.__doc__, (
            "Expected 'conditionally independent' in blq_log_likelihood.__doc__"
        )

    def test_m3_beal_reference_in_docstring(self):
        """Beal (2001) reference is mentioned in the M3 docstring note."""
        doc = blq_log_likelihood.__doc__
        assert doc is not None
        assert "Beal" in doc, "Expected 'Beal' reference in docstring"

    def test_m3_sigma_bias_warning_in_docstring(self):
        """Docstring warns about biased sigma estimates under strong within-subject correlation."""
        doc = blq_log_likelihood.__doc__
        assert doc is not None
        assert "sigma" in doc.lower(), (
            "Expected mention of sigma bias in M3 docstring"
        )

    def test_module_docstring_references_beal_2001(self):
        """Module-level docstring references Beal 2001."""
        doc = blq_module.__doc__
        assert doc is not None
        assert "Beal" in doc, "Expected 'Beal' in module docstring"
