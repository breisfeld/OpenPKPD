"""
Problem: top-level assembled model from a ControlStream.

Ties together the parsed control stream, dataset, PK subroutine,
compiled callables, and parameters into a PopulationModel.
"""

from __future__ import annotations

from dataclasses import dataclass

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.parser.control_stream import ControlStream
from openpkpd.parser.records.sizes import SizesRecord
from openpkpd.pk import get_absorption_model, get_advan
from openpkpd.pk.ode import ADVAN6, ADVAN8
from openpkpd.prior import PriorAugmentedModel, prior_from_control_stream
from openpkpd.utils.constants import BLQMethod
from openpkpd.utils.errors import ModelError, ParseError


@dataclass
class Problem:
    """
    Top-level assembled model from a parsed ControlStream.

    Can be constructed via Problem.from_control_stream(cs, dataset_path=...)
    or assembled programmatically via the ModelBuilder API.
    """

    control_stream: ControlStream
    population_model: PopulationModel | PriorAugmentedModel
    title: str = ""

    @classmethod
    def from_control_stream(
        cls,
        cs: ControlStream,
        dataset_path: str | None = None,
        dataset: NONMEMDataset | None = None,
    ) -> Problem:
        """
        Assemble a Problem from a parsed ControlStream.

        Args:
            cs:           Parsed ControlStream object.
            dataset_path: Path to data file (overrides $DATA filename).
            dataset:      Pre-loaded NONMEMDataset (skips file loading).
        """
        # 1. Load dataset
        if dataset is None:
            data_rec = cs.data
            if data_rec is None:
                raise ParseError("ControlStream missing $DATA record")
            path = dataset_path or data_rec.filename
            if not path:
                raise ParseError("$DATA filename is empty")
            input_rec = cs.input
            input_cols = input_rec.columns if input_rec else None
            dataset = NONMEMDataset.from_csv(
                path,
                input_columns=input_cols,
                ignore_char=data_rec.ignore_char,
            )

        # 2. Parse parameters
        theta_specs: list[ThetaSpec] = []
        for theta_rec in cs.theta_records:
            theta_specs.extend(theta_rec.specs)

        omega_specs: list[OmegaSpec] = []
        for omega_rec in cs.omega_records:
            omega_specs.extend(omega_rec.specs)

        sigma_specs: list[SigmaSpec] = []
        for sigma_rec in cs.sigma_records:
            sigma_specs.extend(sigma_rec.specs)

        if not theta_specs:
            raise ParseError("ControlStream missing $THETA parameters")
        if not omega_specs:
            # Default: single diagonal omega = 0.1
            omega_specs = [OmegaSpec(block_size=1, values=[0.1])]
        if not sigma_specs:
            # Default: single sigma = 0.1
            sigma_specs = [SigmaSpec(block_size=1, values=[0.1])]

        params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)

        # 3. Get PK subroutine
        subr_rec = cs.subroutines
        advan = subr_rec.advan if subr_rec else 2
        trans = subr_rec.trans if subr_rec else 2
        if advan is None:
            advan = 2
        if trans is None:
            trans = 2

        # A2: Read n_compartments from $SIZES PC=N for ODE subroutines
        n_compartments = 10  # NONMEM default
        sizes_rec = cs.get_typed("SIZES", SizesRecord)
        if sizes_rec is not None:
            n_compartments = int(sizes_rec.sizes.get("PC", n_compartments))

        # D1: check for non-standard absorption model first (TRANS=7/8)
        absorption_model = get_absorption_model(subr_rec)
        if absorption_model is not None:
            pk_subroutine = absorption_model
        elif advan in (6, 8):
            cls_map = {6: ADVAN6, 8: ADVAN8}
            pk_subroutine = cls_map[advan](n_compartments=n_compartments)
        else:
            pk_subroutine = get_advan(advan)

        # 4. Compile code blocks
        compiler = NMTRANCompiler(use_jax=False)
        pk_callable = None
        error_callable = None
        des_callable = None  # A1

        if cs.pk is not None:
            try:
                pk_callable = compiler.compile_pk(cs.pk.code)
            except Exception as exc:
                raise ModelError(f"Failed to compile $PK block: {exc}") from exc

        if cs.error is not None:
            try:
                error_callable = compiler.compile_error(cs.error.code)
            except Exception as exc:
                raise ModelError(f"Failed to compile $ERROR block: {exc}") from exc

        # A1: Compile $DES block if present
        if cs.des is not None:
            try:
                des_callable = compiler.compile_des(cs.des.code, n_compartments)
            except Exception as exc:
                raise ModelError(f"Failed to compile $DES block: {exc}") from exc

        # A3: Read BLQ method from $ESTIMATION
        blq_method = BLQMethod.M1
        est_records = cs.estimation_records
        if est_records:
            est_rec = est_records[0]
            blq_raw = getattr(est_rec, "blq", None)
            if blq_raw is not None:
                blq_method = str(blq_raw).upper()

        # 5. Infer covariate columns from $INPUT (non-reserved columns)
        _RESERVED_COLS = {
            "C",
            "ID",
            "L1",
            "L2",
            "DV",
            "AMT",
            "RATE",
            "TIME",
            "EVID",
            "MDV",
            "CMT",
            "ADDL",
            "II",
            "SS",
            "DROP",
            "SKIP",
            "OCC",
            "BLQ",
            "LLOQ",
        }
        input_rec2 = cs.input
        covariate_columns: list[str] = []
        if input_rec2 is not None:
            covariate_columns = [
                c for c in input_rec2.active_columns() if c.upper() not in _RESERVED_COLS
            ]

        # 6. Assemble PopulationModel
        pop_model: PopulationModel | PriorAugmentedModel = PopulationModel(
            dataset=dataset,
            pk_subroutine=pk_subroutine,
            params=params,
            pk_callable=pk_callable,
            error_callable=error_callable,
            des_callable=des_callable,
            trans=trans,
            advan=advan,
            blq_method=blq_method,
            covariate_columns=covariate_columns,
        )

        try:
            prior = prior_from_control_stream(cs, params)
        except ValueError as exc:
            raise ParseError(f"Invalid prior specification: {exc}") from exc
        if prior is not None:
            pop_model = PriorAugmentedModel(pop_model, prior)

        title = cs.problem.title if cs.problem else ""
        return cls(
            control_stream=cs,
            population_model=pop_model,
            title=title,
        )
