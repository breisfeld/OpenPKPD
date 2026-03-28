"""Unit tests for individual record parsers."""

import pytest

from openpkpd.parser.records.data import DataRecord
from openpkpd.parser.records.estimation import EstimationRecord
from openpkpd.parser.records.input import InputRecord
from openpkpd.parser.records.omega import OmegaRecord
from openpkpd.parser.records.subroutines import SubroutinesRecord
from openpkpd.parser.records.theta import ThetaRecord


@pytest.mark.unit
class TestThetaRecord:
    def test_simple_values(self):
        rec = ThetaRecord("(0, 1.5, 10)\n(0, 0.08, 5)\n(0, 30, 500)")
        assert len(rec.specs) == 3
        assert rec.specs[0].init == 1.5
        assert rec.specs[0].lower == 0
        assert rec.specs[0].upper == 10
        assert rec.specs[1].init == 0.08
        assert rec.specs[2].init == 30

    def test_bare_value(self):
        rec = ThetaRecord("1.5")
        assert len(rec.specs) == 1
        assert rec.specs[0].init == 1.5

    def test_fixed_value(self):
        rec = ThetaRecord("(1 FIXED)")
        assert rec.specs[0].init == 1.0
        assert rec.specs[0].fixed is True

    def test_two_tuple(self):
        rec = ThetaRecord("(0, 1.5)")
        assert rec.specs[0].lower == 0
        assert rec.specs[0].init == 1.5
        import math

        assert math.isinf(rec.specs[0].upper)

    def test_multiple_blocks(self):
        rec = ThetaRecord("1.5\n0.08\n30.0")
        assert len(rec.specs) == 3


@pytest.mark.unit
class TestOmegaRecord:
    def test_diagonal(self):
        rec = OmegaRecord("0.5\n0.3\n0.3")
        assert len(rec.specs) == 3
        for s in rec.specs:
            assert s.block_size == 1

    def test_block(self):
        rec = OmegaRecord("BLOCK(2)\n0.5\n0.1 0.3")
        assert len(rec.specs) == 1
        assert rec.specs[0].block_size == 2
        assert len(rec.specs[0].values) == 3

    def test_same(self):
        rec = OmegaRecord("0.5\nSAME")
        assert rec.specs[1].same is True

    def test_fixed(self):
        rec = OmegaRecord("FIXED 0.1")
        assert rec.specs[0].fixed is True

    def test_block_to_matrix(self):
        rec = OmegaRecord("BLOCK(2)\n0.5\n0.01 0.3")
        mat = rec.specs[0].to_matrix()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(0.5)
        assert mat[1, 0] == pytest.approx(0.01)
        assert mat[0, 1] == pytest.approx(0.01)  # symmetric
        assert mat[1, 1] == pytest.approx(0.3)


@pytest.mark.unit
class TestEstimationRecord:
    def test_foce_interaction(self):
        rec = EstimationRecord("METHOD=COND INTER MAXEVAL=9999")
        assert rec.method == "FOCE"
        assert rec.interaction is True
        assert rec.maxeval == 9999

    def test_fo_method(self):
        rec = EstimationRecord("METHOD=0")
        assert rec.method == "FO"

    def test_laplace(self):
        rec = EstimationRecord("METHOD=COND LAPLACE")
        assert rec.laplace is True

    def test_saem(self):
        rec = EstimationRecord("METHOD=SAEM NITER=300")
        assert rec.method == "SAEM"

    def test_openpkpd_optimizer_extensions(self):
        rec = EstimationRecord(
            "METHOD=COND INTER OUTEROPT=Powell FALLBACKOPT=L-BFGS-B "
            "FALLBACKMAXEVAL=25 RETAINBEST RETRYONABNORMAL RETRYOMEGASCALE=0.5,0.25"
        )
        assert rec.outer_optimizer == "Powell"
        assert rec.outer_fallback_optimizer == "L-BFGS-B"
        assert rec.outer_fallback_maxeval == 25
        assert rec.retain_best_iterate is True
        assert rec.retry_on_abnormal is True
        assert rec.retry_omega_scales == pytest.approx((0.5, 0.25))


@pytest.mark.unit
class TestDataRecord:
    def test_filename(self):
        rec = DataRecord("theo.csv IGNORE=@")
        assert rec.filename == "theo.csv"
        assert rec.ignore_char == "@"

    def test_ignore_list(self):
        rec = DataRecord("data.csv IGNORE=(EVID.EQ.3)")
        assert len(rec.ignore_list) > 0

    def test_records_lrecl(self):
        rec = DataRecord("data.csv RECORDS=100 LRECL=200")
        assert rec.records == 100
        assert rec.lrecl == 200


@pytest.mark.unit
class TestInputRecord:
    def test_basic_columns(self):
        rec = InputRecord("ID TIME AMT DV EVID MDV")
        assert "ID" in rec.columns
        assert "TIME" in rec.columns
        assert "DV" in rec.columns

    def test_drop_column(self):
        rec = InputRecord("ID TIME DROP DV EVID")
        assert "_DROP_3" in rec.columns
        assert rec.dropped == [3]

    def test_alias(self):
        rec = InputRecord("ID TIME AMT DV=CONC EVID")
        assert "DV" in rec.columns


@pytest.mark.unit
class TestSubroutinesRecord:
    def test_advan_trans(self):
        rec = SubroutinesRecord("ADVAN2 TRANS2")
        assert rec.advan == 2
        assert rec.trans == 2

    def test_tol(self):
        rec = SubroutinesRecord("ADVAN6 TRANS1 TOL=9")
        assert rec.advan == 6
        assert rec.tol == 9
