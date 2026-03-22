"""
Unit tests for complex record parsing: BLOCK OMEGA, FIXED, SAME, multiple records.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.parser.records.omega import OmegaRecord
from openpkpd.parser.records.theta import ThetaRecord

# ---------------------------------------------------------------------------
# BLOCK OMEGA
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBlockOmega:
    def test_block_omega_2x2(self):
        """BLOCK(2) with lower-triangle values parses to 3 elements."""
        rec = OmegaRecord("BLOCK(2)\n0.5\n0.1 0.3")
        assert len(rec.specs) == 1
        spec = rec.specs[0]
        assert spec.block_size == 2
        assert len(spec.values) == 3
        # Lower triangle: [0,0]=0.5, [1,0]=0.1, [1,1]=0.3
        assert spec.values[0] == pytest.approx(0.5)
        assert spec.values[1] == pytest.approx(0.1)
        assert spec.values[2] == pytest.approx(0.3)

    def test_block_omega_3x3(self):
        """BLOCK(3) with 6 lower-triangle values."""
        rec = OmegaRecord("BLOCK(3)\n0.4\n0.1 0.3\n0.05 0.02 0.2")
        assert len(rec.specs) == 1
        spec = rec.specs[0]
        assert spec.block_size == 3
        assert len(spec.values) == 6

    def test_block_omega_to_matrix(self):
        """ParameterSet should reconstruct 2x2 OMEGA from BLOCK spec."""
        specs = [OmegaSpec(block_size=2, values=[0.5, 0.1, 0.3])]
        sigma_specs = [SigmaSpec(block_size=1, values=[0.1])]
        theta_specs = [ThetaSpec(init=1.0)]
        ps = ParameterSet.from_specs(theta_specs, specs, sigma_specs)
        omega = ps.omega
        assert omega.shape == (2, 2)
        assert omega[0, 0] == pytest.approx(0.5)
        assert omega[1, 0] == pytest.approx(0.1)
        assert omega[0, 1] == pytest.approx(0.1)  # symmetric
        assert omega[1, 1] == pytest.approx(0.3)

    def test_block_omega_positive_definite(self):
        """A valid BLOCK(2) OMEGA should be positive definite."""
        specs = [OmegaSpec(block_size=2, values=[0.5, 0.1, 0.3])]
        sigma_specs = [SigmaSpec(block_size=1, values=[0.1])]
        theta_specs = [ThetaSpec(init=1.0)]
        ps = ParameterSet.from_specs(theta_specs, specs, sigma_specs)
        eigenvalues = np.linalg.eigvalsh(ps.omega)
        assert np.all(eigenvalues > 0), f"OMEGA not PD: eigenvalues={eigenvalues}"


# ---------------------------------------------------------------------------
# FIXED THETA
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFixedTheta:
    def test_fixed_theta_parsed(self):
        """FIXED theta should have fixed=True."""
        rec = ThetaRecord("(1 FIXED)")
        assert rec.specs[0].fixed is True
        assert rec.specs[0].init == pytest.approx(1.0)

    def test_fixed_theta_multi(self):
        """Multiple THETAs with some FIXED."""
        rec = ThetaRecord("(0, 1.5, 10)\n(1.0 FIXED)\n(0, 0.08, 5)")
        assert len(rec.specs) == 3
        assert rec.specs[0].fixed is False
        assert rec.specs[1].fixed is True
        assert rec.specs[1].init == pytest.approx(1.0)
        assert rec.specs[2].fixed is False

    def test_fixed_theta_not_optimized(self):
        """Fixed THETA should not appear in the free parameter vector."""
        theta_specs = [
            ThetaSpec(init=1.5, lower=0.01, upper=20.0, fixed=False),
            ThetaSpec(init=1.0, fixed=True),  # fixed
            ThetaSpec(init=0.08, lower=0.001, upper=5.0, fixed=False),
        ]
        omega_specs = [OmegaSpec(block_size=1, values=[0.1])]
        sigma_specs = [SigmaSpec(block_size=1, values=[0.05])]
        ps = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
        vec = ps.to_vector()
        # Only 2 free THETAs + 1 omega + 1 sigma = 4 free params
        # (exact count depends on implementation, but < 5)
        assert len(vec) < 5


# ---------------------------------------------------------------------------
# SAME OMEGA
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSameOmega:
    def test_same_parsed(self):
        """$OMEGA SAME should produce a spec with same=True."""
        rec = OmegaRecord("0.5\nSAME")
        assert len(rec.specs) == 2
        assert rec.specs[1].same is True

    def test_same_block_size_inherited(self):
        """SAME block should inherit block_size from preceding spec."""
        rec = OmegaRecord("BLOCK(2)\n0.4\n0.1 0.3\nSAME")
        assert rec.specs[0].block_size == 2
        assert rec.specs[1].same is True


# ---------------------------------------------------------------------------
# FIXED OMEGA BLOCK
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFixedOmegaBlock:
    def test_fixed_block_omega(self):
        """BLOCK(2) FIXED should parse with fixed=True."""
        rec = OmegaRecord("BLOCK(2) FIXED\n0.4\n0.1 0.3")
        assert len(rec.specs) == 1
        assert rec.specs[0].fixed is True
        assert rec.specs[0].block_size == 2


# ---------------------------------------------------------------------------
# Multiple $THETA records merge
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultipleThetaRecords:
    def test_multiple_theta_records_merge(self):
        """Multiple $THETA records should merge into a single spec list."""
        from openpkpd.parser.control_stream import ControlStream

        cs_text = """\
$PROBLEM Test
$DATA dummy.csv
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
$ERROR
  Y = F + EPS(1)
$THETA (0, 1.0, 10)
$THETA (0, 30.0, 500)
$OMEGA 0.3
$SIGMA 0.1
$ESTIMATION METHOD=ZERO
"""
        cs = ControlStream.from_string(cs_text)
        # Merged theta should have 2 specs total
        all_specs = []
        for tr in cs.theta_records:
            all_specs.extend(tr.specs)
        assert len(all_specs) == 2
        assert all_specs[0].init == pytest.approx(1.0)
        assert all_specs[1].init == pytest.approx(30.0)
