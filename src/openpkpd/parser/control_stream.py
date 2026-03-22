"""
ControlStream: top-level parsed representation of a NONMEM control stream.

Parses a .ctl/.mod file into a list of typed record objects and provides
accessor methods for downstream components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openpkpd.parser.lexer import RawRecord, split_into_raw_records
from openpkpd.parser.records.abbreviated import AbbreviatedRecord
from openpkpd.parser.records.base import BaseRecord
from openpkpd.parser.records.contr import ContrRecord
from openpkpd.parser.records.covariance import CovarianceRecord
from openpkpd.parser.records.data import DataRecord
from openpkpd.parser.records.des import DESRecord
from openpkpd.parser.records.design import DesignRecord
from openpkpd.parser.records.error import ErrorRecord
from openpkpd.parser.records.estimation import EstimationRecord
from openpkpd.parser.records.input import InputRecord
from openpkpd.parser.records.mixture import MixtureRecord
from openpkpd.parser.records.nonparametric import NonparametricRecord
from openpkpd.parser.records.omega import OmegaRecord
from openpkpd.parser.records.pk import PKRecord
from openpkpd.parser.records.pred import PredRecord
from openpkpd.parser.records.prior import (
    OmegaPDRecord,
    OmegaPRecord,
    PriorRecord,
    SigmaPDRecord,
    SigmaPRecord,
    ThetaPRecord,
    ThetaPVRecord,
)

# Import all record types
from openpkpd.parser.records.problem import ProblemRecord
from openpkpd.parser.records.sigma import SigmaRecord
from openpkpd.parser.records.simulation import SimulationRecord
from openpkpd.parser.records.sizes import SizesRecord
from openpkpd.parser.records.subroutines import SubroutinesRecord
from openpkpd.parser.records.table import TableRecord
from openpkpd.parser.records.theta import ThetaRecord
from openpkpd.utils.errors import ParseError

T = TypeVar("T", bound=BaseRecord)

# Registry: canonical record name → record class
_RECORD_REGISTRY: dict[str, type[BaseRecord]] = {
    "PROBLEM": ProblemRecord,
    "DATA": DataRecord,
    "INPUT": InputRecord,
    "SUBROUTINES": SubroutinesRecord,
    "THETA": ThetaRecord,
    "OMEGA": OmegaRecord,
    "SIGMA": SigmaRecord,
    "PK": PKRecord,
    "DES": DESRecord,
    "ERROR": ErrorRecord,
    "PRED": PredRecord,
    "ESTIMATION": EstimationRecord,
    "COVARIANCE": CovarianceRecord,
    "TABLE": TableRecord,
    "SIMULATION": SimulationRecord,
    "PRIOR": PriorRecord,
    "THETAP": ThetaPRecord,
    "THETAPV": ThetaPVRecord,
    "OMEGAP": OmegaPRecord,
    "OMEGAPD": OmegaPDRecord,
    "SIGMAP": SigmaPRecord,
    "SIGMAPD": SigmaPDRecord,
    "MIXTURE": MixtureRecord,
    "ABBREVIATED": AbbreviatedRecord,
    "NONPARAMETRIC": NonparametricRecord,
    "SIZES": SizesRecord,
    "DESIGN": DesignRecord,
    "CONTR": ContrRecord,
}


class UnknownRecord(BaseRecord):
    """Placeholder for unrecognised record types."""

    def _parse(self, text: str) -> None:
        self.text = text


@dataclass
class ControlStream:
    """
    Top-level parsed representation of a NONMEM control stream.
    """

    source_text: str
    records: list[BaseRecord] = field(default_factory=list)
    source_path: str | None = None

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get(self, record_type: str) -> BaseRecord | None:
        """Return the first record of the given type, or None."""
        upper = record_type.upper()
        for rec in self.records:
            if rec.record_name.upper() == upper:
                return rec
        return None

    def get_all(self, record_type: str) -> list[BaseRecord]:
        """Return all records of the given type."""
        upper = record_type.upper()
        return [rec for rec in self.records if rec.record_name.upper() == upper]

    def get_typed(self, record_type: str, cls: type[T]) -> T | None:
        """Type-safe accessor returning a specific record class."""
        rec = self.get(record_type)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise ParseError(
                f"Expected {cls.__name__} for ${record_type}, got {type(rec).__name__}"
            )
        return rec

    def get_all_typed(self, record_type: str, cls: type[T]) -> list[T]:
        """Type-safe accessor returning a list of specific record class."""
        result = []
        for rec in self.get_all(record_type):
            if not isinstance(rec, cls):
                raise ParseError(
                    f"Expected {cls.__name__} for ${record_type}, got {type(rec).__name__}"
                )
            result.append(rec)
        return result

    # Convenience typed accessors
    @property
    def problem(self) -> ProblemRecord | None:
        return self.get_typed("PROBLEM", ProblemRecord)

    @property
    def data(self) -> DataRecord | None:
        return self.get_typed("DATA", DataRecord)

    @property
    def input(self) -> InputRecord | None:
        return self.get_typed("INPUT", InputRecord)

    @property
    def subroutines(self) -> SubroutinesRecord | None:
        return self.get_typed("SUBROUTINES", SubroutinesRecord)

    @property
    def pk(self) -> PKRecord | None:
        return self.get_typed("PK", PKRecord)

    @property
    def des(self) -> DESRecord | None:
        return self.get_typed("DES", DESRecord)

    @property
    def error(self) -> ErrorRecord | None:
        return self.get_typed("ERROR", ErrorRecord)

    @property
    def pred(self) -> PredRecord | None:
        return self.get_typed("PRED", PredRecord)

    @property
    def theta_records(self) -> list[ThetaRecord]:
        return self.get_all_typed("THETA", ThetaRecord)

    @property
    def omega_records(self) -> list[OmegaRecord]:
        return self.get_all_typed("OMEGA", OmegaRecord)

    @property
    def sigma_records(self) -> list[SigmaRecord]:
        return self.get_all_typed("SIGMA", SigmaRecord)

    @property
    def simulation(self) -> SimulationRecord | None:
        return self.get_typed("SIMULATION", SimulationRecord)

    @property
    def mixture(self) -> MixtureRecord | None:
        return self.get_typed("MIXTURE", MixtureRecord)

    @property
    def prior_record(self) -> PriorRecord | None:
        return self.get_typed("PRIOR", PriorRecord)

    @property
    def thetap_record(self) -> ThetaPRecord | None:
        return self.get_typed("THETAP", ThetaPRecord)

    @property
    def thetapv_record(self) -> ThetaPVRecord | None:
        return self.get_typed("THETAPV", ThetaPVRecord)

    @property
    def omegap_record(self) -> OmegaPRecord | None:
        return self.get_typed("OMEGAP", OmegaPRecord)

    @property
    def omegapd_record(self) -> OmegaPDRecord | None:
        return self.get_typed("OMEGAPD", OmegaPDRecord)

    @property
    def sigmap_record(self) -> SigmaPRecord | None:
        return self.get_typed("SIGMAP", SigmaPRecord)

    @property
    def sigmapd_record(self) -> SigmaPDRecord | None:
        return self.get_typed("SIGMAPD", SigmaPDRecord)

    @property
    def estimation_records(self) -> list[EstimationRecord]:
        return self.get_all_typed("ESTIMATION", EstimationRecord)

    @property
    def covariance(self) -> CovarianceRecord | None:
        return self.get_typed("COVARIANCE", CovarianceRecord)

    @property
    def table_records(self) -> list[TableRecord]:
        return self.get_all_typed("TABLE", TableRecord)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> ControlStream:
        """Load and parse a NONMEM control stream from a file."""
        if not os.path.exists(path):
            raise ParseError(f"Control stream file not found: {path}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        cs = cls.from_string(text)
        cs.source_path = os.path.abspath(path)
        return cs

    @classmethod
    def from_string(cls, text: str) -> ControlStream:
        """Parse a NONMEM control stream from a string."""
        raw_records = split_into_raw_records(text)
        records: list[BaseRecord] = []
        for raw in raw_records:
            rec = _build_record(raw)
            records.append(rec)
        return cls(source_text=text, records=records)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for debugging."""
        return {
            "source_path": self.source_path,
            "records": [r.to_dict() for r in self.records],
        }

    def __repr__(self) -> str:
        rec_names = [r.record_name for r in self.records]
        return f"ControlStream(records={rec_names})"


def _build_record(raw: RawRecord) -> BaseRecord:
    """Instantiate the appropriate record class from a RawRecord."""
    cls = _RECORD_REGISTRY.get(raw.name, UnknownRecord)
    try:
        rec = cls(raw_text=raw.raw_text, header_line=raw.header_line)
        # Patch record_name for UnknownRecord
        if isinstance(rec, UnknownRecord):
            rec.record_name = raw.name
        return rec
    except Exception as exc:
        raise ParseError(
            f"Error parsing ${raw.name} record",
            line=raw.header_line,
            context=str(exc),
        ) from exc
