"""
Comprehensive tests for the structured estimation-warning infrastructure (P0.5).

Covers:
  - WarningCode / WarningSeverity enums
  - EstimationWarning dataclass (str, repr, fields)
  - EstimationResult.add_structured_warning()
  - EstimationResult.check_omega_conditioning() → WARN_001, WARN_002, WARN_004
  - compute_shrinkage() → WARN_005
  - Backward-compatibility: warnings list is also populated
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.utils.errors import EstimationWarning, WarningCode, WarningSeverity


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _result(**kw) -> EstimationResult:
    defaults = dict(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1) * 0.1,
        sigma_final=np.eye(1) * 0.05,
        ofv=-100.0,
    )
    defaults.update(kw)
    return EstimationResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# WarningCode / WarningSeverity enums
# ─────────────────────────────────────────────────────────────────────────────


class TestWarningEnums:
    def test_all_codes_present(self):
        expected = {f"WARN_{i:03d}" for i in range(1, 9)}
        actual = {m.value for m in WarningCode}
        assert expected.issubset(actual), f"Missing codes: {expected - actual}"

    def test_severity_levels(self):
        assert WarningSeverity.INFO.value == "INFO"
        assert WarningSeverity.WARNING.value == "WARNING"
        assert WarningSeverity.ERROR.value == "ERROR"

    def test_warning_code_enum_members_unique(self):
        values = [m.value for m in WarningCode]
        assert len(values) == len(set(values))


# ─────────────────────────────────────────────────────────────────────────────
# EstimationWarning dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimationWarning:
    def test_str_contains_code_and_message(self):
        w = EstimationWarning(WarningCode.WARN_001, "test message")
        s = str(w)
        assert "WARN_001" in s
        assert "test message" in s

    def test_default_severity_is_warning(self):
        w = EstimationWarning(WarningCode.WARN_002, "msg")
        assert w.severity is WarningSeverity.WARNING

    def test_explicit_severity_stored(self):
        w = EstimationWarning(WarningCode.WARN_004, "sev", WarningSeverity.ERROR)
        assert w.severity is WarningSeverity.ERROR

    def test_repr_contains_code_and_severity(self):
        w = EstimationWarning(WarningCode.WARN_003, "gradient issue")
        r = repr(w)
        assert "WARN_003" in r
        assert "gradient issue" in r


# ─────────────────────────────────────────────────────────────────────────────
# EstimationResult.add_structured_warning
# ─────────────────────────────────────────────────────────────────────────────


class TestAddStructuredWarning:
    def test_appends_to_structured_warnings(self):
        res = _result()
        res.add_structured_warning(WarningCode.WARN_001, "cond number high")
        assert len(res.structured_warnings) == 1
        assert res.structured_warnings[0].code is WarningCode.WARN_001

    def test_also_appends_string_to_warnings_list(self):
        """Backward compatibility: str form appears in warnings list."""
        res = _result()
        res.add_structured_warning(WarningCode.WARN_003, "grad large")
        assert any("WARN_003" in w for w in res.warnings)

    def test_severity_passed_through(self):
        res = _result()
        res.add_structured_warning(WarningCode.WARN_004, "singular", WarningSeverity.ERROR)
        assert res.structured_warnings[0].severity is WarningSeverity.ERROR

    def test_multiple_warnings_accumulated(self):
        res = _result()
        for code in [WarningCode.WARN_001, WarningCode.WARN_005, WarningCode.WARN_007]:
            res.add_structured_warning(code, f"msg for {code.value}")
        assert len(res.structured_warnings) == 3
        codes = {w.code for w in res.structured_warnings}
        assert codes == {WarningCode.WARN_001, WarningCode.WARN_005, WarningCode.WARN_007}


# ─────────────────────────────────────────────────────────────────────────────
# EstimationResult.check_omega_conditioning
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckOmegaConditioning:
    def test_well_conditioned_omega_produces_no_warning(self):
        """Condition number = 1 (identity) — no WARN_001/002/004 expected."""
        res = _result(omega_final=np.eye(3))
        res.check_omega_conditioning()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_001 not in codes
        assert WarningCode.WARN_002 not in codes
        assert WarningCode.WARN_004 not in codes

    def test_condition_number_between_1000_and_10000_gives_warn001(self):
        """Eigenvalues [1, 2000] → condition number = 2000 → WARN_001."""
        omega = np.diag([1.0, 2000.0])
        res = _result(omega_final=omega)
        res.check_omega_conditioning()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_001 in codes
        assert WarningCode.WARN_002 not in codes

    def test_condition_number_above_10000_gives_warn002(self):
        """Eigenvalues [1, 50000] → condition number = 50000 → WARN_002."""
        omega = np.diag([1.0, 50000.0])
        res = _result(omega_final=omega)
        res.check_omega_conditioning()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_002 in codes

    def test_near_singular_omega_gives_warn004(self):
        """Smallest eigenvalue < 1e-6 → WARN_004 (near-singular)."""
        omega = np.diag([1e-9, 1.0])
        res = _result(omega_final=omega)
        res.check_omega_conditioning()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_004 in codes

    def test_warn002_severity_is_error(self):
        omega = np.diag([1.0, 1e5])
        res = _result(omega_final=omega)
        res.check_omega_conditioning()
        w002 = next(w for w in res.structured_warnings if w.code is WarningCode.WARN_002)
        assert w002.severity is WarningSeverity.ERROR

    def test_warn001_severity_is_warning(self):
        omega = np.diag([1.0, 5000.0])
        res = _result(omega_final=omega)
        res.check_omega_conditioning()
        w001 = next(w for w in res.structured_warnings if w.code is WarningCode.WARN_001)
        assert w001.severity is WarningSeverity.WARNING


# ─────────────────────────────────────────────────────────────────────────────
# compute_shrinkage → WARN_005
# ─────────────────────────────────────────────────────────────────────────────


class TestShrinkageWarn005:
    def test_high_shrinkage_adds_warn005(self):
        """Shrinkage > 30% should produce WARN_005."""
        # omega_kk = 1.0, but all EBEs are 0 → SD(EBEs) = 0 → shrinkage = 1
        res = _result(omega_final=np.eye(1))
        res.post_hoc_etas = {i: np.zeros(1) for i in range(10)}
        res.compute_shrinkage()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_005 in codes

    def test_low_shrinkage_no_warn005(self):
        """If EBEs spread matches omega → shrinkage ~ 0 → no WARN_005."""
        omega = np.eye(1) * 1.0
        rng = np.random.default_rng(42)
        # EBEs drawn from N(0, omega) → SD ≈ 1 → shrinkage ≈ 0
        etas = {i: rng.normal(0, 1, 1) for i in range(100)}
        res = _result(omega_final=omega)
        res.post_hoc_etas = etas
        res.compute_shrinkage()
        codes = {w.code for w in res.structured_warnings}
        assert WarningCode.WARN_005 not in codes

    def test_warn005_message_contains_eta_number(self):
        res = _result(omega_final=np.eye(2))
        res.post_hoc_etas = {i: np.zeros(2) for i in range(10)}
        res.compute_shrinkage()
        msgs = [str(w) for w in res.structured_warnings if w.code is WarningCode.WARN_005]
        assert any("ETA1" in m or "ETA2" in m for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# Summary integration
# ─────────────────────────────────────────────────────────────────────────────


class TestSummaryWithStructuredWarnings:
    def test_summary_includes_warning_codes(self):
        res = _result()
        res.add_structured_warning(WarningCode.WARN_001, "cond number 5000")
        s = res.summary()
        assert "WARN_001" in s

    def test_summary_without_warnings_does_not_crash(self):
        res = _result()
        s = res.summary()
        assert "Method" in s
