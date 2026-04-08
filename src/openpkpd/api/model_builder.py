"""
ModelBuilder: native Python API for defining openpkpd models without a .ctl file.

Example usage:
    from openpkpd import ModelBuilder

    model = (
        ModelBuilder()
        .problem("Theophylline 1-compartment oral")
        .data("theo.csv", id="ID", time="TIME", dv="DV", amt="AMT", evid="EVID")
        .subroutines(advan=2, trans=2)
        .pk(\"\"\"
            KA = THETA(1) * EXP(ETA(1))
            CL = THETA(2) * EXP(ETA(2))
            V  = THETA(3) * EXP(ETA(3))
        \"\"\")
        .error(\"\"\"
            Y = F * (1 + EPS(1))
        \"\"\")
        .theta([(0.1, 1.5, 10), (0.01, 0.1, 1), (0.1, 30, 500)])
        .omega([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.3]])
        .sigma([[0.1]])
        .estimation(method="FOCE", interaction=True, maxeval=9999)
        .covariance()
        .build()
    )

    result = model.fit()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.design.pfim import PFIMEngine
from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk import get_advan
from openpkpd.simulation.engine import SimulationResult
from openpkpd.utils.errors import ModelError
from openpkpd.utils.logging import get_logger

logger = get_logger("api.model_builder")


class ConfigurationError(ValueError):
    """Raised when an invalid ADVAN/TRANS combination or covariate column is specified."""
    pass


# Valid TRANS values for each ADVAN. None means any TRANS is accepted.
_VALID_ADVAN_TRANS: dict[int, set[int] | None] = {
    1:  {1, 2},           # 1-cmt IV: TRANS1 (K), TRANS2 (CL/V)
    2:  {1, 2},           # 1-cmt oral: same
    3:  {1, 3, 4, 5},     # 2-cmt IV: TRANS1 (micro), TRANS3/4/5
    4:  {1, 3, 4, 5},     # 2-cmt oral: same
    5:  {1},              # N-cmt general linear: TRANS1 (micro rate constants)
    7:  {1},              # N-cmt general linear (matrix exponential backend)
    11: {1, 3, 4, 5, 6},  # 3-cmt IV: adds TRANS6
    12: {1, 3, 4, 5, 6},  # 3-cmt oral
    6:  None,             # general ODE: no TRANS restriction
    8:  None,             # general ODE with SS: no TRANS restriction
    10: None,             # general ODE with saturable binding: no TRANS restriction
}


@dataclass
class BuiltModel:
    """
    A fully assembled model ready to fit.

    Returned by ModelBuilder.build(). Call .fit() to run estimation.
    """

    population_model: PopulationModel
    params: ParameterSet
    estimation_kwargs: dict[str, Any] = field(default_factory=dict)
    do_covariance: bool = False
    covariance_kwargs: dict[str, Any] = field(default_factory=dict)

    def design(
        self,
        sampling_times: list[float] | np.ndarray | None = None,
    ) -> PFIMEngine:
        """
        Return a :class:`~openpkpd.design.pfim.PFIMEngine` wired to this model.

        The engine can then evaluate or optimise the population Fisher
        Information Matrix (FIM) for the current parameters and model.

        Args:
            sampling_times: Default sampling times (hours).  If ``None``,
                            a 24-hour hourly grid is used as a starting
                            point for ``compute_fim`` / ``optimize_design``.

        Returns:
            :class:`~openpkpd.design.pfim.PFIMEngine` ready to call
            ``.compute_fim()`` or ``.optimize_design()``.

        Example::

            built = ModelBuilder()...build()
            result = built.fit()
            engine = built.design()
            fim = engine.compute_fim(np.linspace(0.5, 24, 8))
            optimal = engine.optimize_design(n_samples=6, t_min=0, t_max=24)
            print(optimal.summary())
        """
        times = (
            np.asarray(sampling_times, dtype=float)
            if sampling_times is not None
            else np.arange(1.0, 25.0, 1.0)
        )
        return PFIMEngine(
            population_model=self.population_model,
            init_params=self.params,
            sampling_times=times,
        )

    def simulate(
        self,
        n_replicates: int = 1,
        seed: int = 42,
        result: EstimationResult | None = None,
    ) -> SimulationResult:
        """
        Simulate replicate datasets from the fitted model.

        Convenience wrapper around SimulationEngine that optionally runs
        estimation first if no EstimationResult is provided.

        Args:
            n_replicates: Number of simulated datasets to generate (default 1).
            seed:         Random seed for reproducibility (default 42).
            result:       Pre-computed EstimationResult. If None, calls fit() first.

        Returns:
            SimulationResult with simulated_df, seed, and n_replicates.

        Example:
            model = ModelBuilder()...build()
            result = model.fit()
            sim = model.simulate(n_replicates=500, result=result)
            print(sim.simulated_df.head())
        """
        from openpkpd.simulation.engine import SimulationEngine

        if result is None:
            result = self.fit()
        engine = SimulationEngine(self.population_model, result, seed=seed)
        return engine.simulate(n_replicates=n_replicates)

    def fit(self) -> EstimationResult:
        """Run estimation and return results."""
        method_name = self.estimation_kwargs.get("method", "FOCE")
        interaction = self.estimation_kwargs.get("interaction", False)
        maxeval = self.estimation_kwargs.get("maxeval", 9999)
        kwargs = {
            k: v
            for k, v in self.estimation_kwargs.items()
            if k not in ("method", "interaction", "maxeval")
        }

        est = get_estimation_method(
            method_name,
            interaction=interaction,
            maxeval=maxeval,
            **kwargs,
        )
        result = est.estimate(self.population_model, self.params)
        dataset = getattr(self.population_model, "dataset", None)
        if dataset is not None and hasattr(dataset, "n_observations"):
            result.n_observations = int(dataset.n_observations())
        result.n_subjects = self.population_model.n_subjects()
        result.compute_n_parameters(
            theta_specs=self.params.theta_specs,
            omega_specs=self.params.omega_specs,
            sigma_specs=self.params.sigma_specs,
        )

        if self.do_covariance:
            from openpkpd.covariance.sandwich import SandwichCovariance

            cov_est = SandwichCovariance(**self.covariance_kwargs)
            final_params = ParameterSet(
                theta=result.theta_final,
                omega=result.omega_final,
                sigma=result.sigma_final,
                theta_specs=self.params.theta_specs,
                omega_specs=self.params.omega_specs,
                sigma_specs=self.params.sigma_specs,
            )
            cov_result = cov_est.compute(self.population_model, final_params, result.post_hoc_etas)
            result.warnings.extend(cov_result.warnings)

        return result


class ModelBuilder:
    """
    Fluent builder API for constructing openpkpd models in pure Python.

    Chain method calls to configure the model, then call .build() to
    produce a BuiltModel ready for estimation.
    """

    def __init__(self) -> None:
        self._title: str = ""
        self._dataset: NONMEMDataset | None = None
        self._data_path: str | None = None
        self._data_kwargs: dict[str, Any] = {}
        self._advan: int = 2
        self._trans: int = 2
        self._subroutine_kwargs: dict[str, Any] = {}
        self._pk_code: str | None = None
        self._error_code: str | None = None
        self._des_code: str | None = None
        self._theta_specs: list[ThetaSpec] = []
        self._omega_specs: list[OmegaSpec] = []
        self._sigma_specs: list[SigmaSpec] = []
        self._estimation_kwargs: dict[str, Any] = {"method": "FOCE"}
        self._do_covariance: bool = False
        self._covariance_kwargs: dict[str, Any] = {}
        self._covariate_columns: list[str] = []
        self._impute_columns: list[str] = []
        self._impute_method: str = "locf"

    def problem(self, title: str) -> ModelBuilder:
        """Set the problem title."""
        self._title = title
        return self

    def data(
        self,
        path: str,
        id: str = "ID",
        time: str = "TIME",
        dv: str = "DV",
        amt: str = "AMT",
        evid: str | None = "EVID",
        **kwargs: Any,
    ) -> ModelBuilder:
        """
        Specify the dataset file path and column mappings.

        Args:
            path: Path to CSV data file.
            id, time, dv, amt, evid: Column name overrides.
            **kwargs: Additional keyword arguments for NONMEMDataset.from_csv().
        """
        self._data_path = path
        self._data_kwargs = {
            "id_col": id,
            "time_col": time,
            "dv_col": dv,
            "amt_col": amt,
            "evid_col": evid,
            **kwargs,
        }
        return self

    def dataset(self, ds: NONMEMDataset) -> ModelBuilder:
        """Provide a pre-loaded NONMEMDataset directly."""
        self._dataset = ds
        return self

    def subroutines(self, advan: int = 2, trans: int = 2, **kwargs: Any) -> ModelBuilder:
        """Set ADVAN and TRANS subroutine numbers and optional subroutine kwargs."""
        if advan not in _VALID_ADVAN_TRANS:
            raise ConfigurationError(
                f"ADVAN={advan} is not supported. Supported: {sorted(_VALID_ADVAN_TRANS)}"
            )
        allowed = _VALID_ADVAN_TRANS[advan]
        if allowed is not None and trans not in allowed:
            raise ConfigurationError(
                f"TRANS={trans} is not valid for ADVAN={advan}. Allowed TRANS values: {sorted(allowed)}"
            )
        self._advan = advan
        self._trans = trans
        self._subroutine_kwargs = dict(kwargs)
        return self

    def pk(self, code: str) -> ModelBuilder:
        """Set the $PK code block (NM-TRAN syntax)."""
        self._pk_code = code
        return self

    def error(self, code: str) -> ModelBuilder:
        """Set the $ERROR code block (NM-TRAN syntax)."""
        self._error_code = code
        return self

    def des(self, code: str) -> ModelBuilder:
        """Set the $DES code block for ODE models (NM-TRAN syntax)."""
        self._des_code = code
        return self

    def theta(
        self,
        specs: list[tuple | float | ThetaSpec],
        labels: list[str] | None = None,
    ) -> ModelBuilder:
        """
        Set THETA initial estimates and bounds.

        Each element can be:
          - A float: initial value, unbounded
          - A 2-tuple (lower, init): lower-bounded
          - A 3-tuple (lower, init, upper): fully bounded
          - A ThetaSpec instance

        Examples:
            .theta([1.5, (0, 0.1, 1), (0, 30)])
        """
        specs_list: list[ThetaSpec] = []
        for i, s in enumerate(specs):
            label = labels[i] if labels and i < len(labels) else None
            if isinstance(s, ThetaSpec):
                specs_list.append(s)
            elif isinstance(s, (int, float)):
                specs_list.append(ThetaSpec(init=float(s), label=label))
            elif isinstance(s, (tuple, list)):
                if len(s) == 1:
                    specs_list.append(ThetaSpec(init=float(s[0]), label=label))
                elif len(s) == 2:
                    specs_list.append(ThetaSpec(lower=float(s[0]), init=float(s[1]), label=label))
                elif len(s) == 3:
                    specs_list.append(
                        ThetaSpec(
                            lower=float(s[0]), init=float(s[1]), upper=float(s[2]), label=label
                        )
                    )
                else:
                    raise ModelError(f"Invalid theta spec: {s}")
            else:
                raise ModelError(f"Invalid theta spec type: {type(s)}")
        self._theta_specs = specs_list
        return self

    def omega(
        self,
        values: Any,
        fixed: bool = False,
    ) -> ModelBuilder:
        """
        Set OMEGA initial values.

        Args:
            values: 2D matrix (full or lower-triangle). Can also be a list of
                    diagonal values [v1, v2, ...] for a diagonal OMEGA.
            fixed:  Fix OMEGA values.
        """
        arr = np.array(values, dtype=float)
        if arr.ndim == 1:
            # Diagonal: each value is a separate 1×1 block
            self._omega_specs = [
                OmegaSpec(block_size=1, values=[float(v)], fixed=fixed) for v in arr
            ]
        elif arr.ndim == 2:
            n = arr.shape[0]
            lower_tri = []
            for row in range(n):
                for col in range(row + 1):
                    lower_tri.append(float(arr[row, col]))
            self._omega_specs = [OmegaSpec(block_size=n, values=lower_tri, fixed=fixed)]
        return self

    def sigma(
        self,
        values: Any,
        fixed: bool = False,
    ) -> ModelBuilder:
        """
        Set SIGMA initial values.

        Args:
            values: Scalar, 1D list of diagonal values, or 2D matrix.
            fixed:  Fix SIGMA values.
        """
        if isinstance(values, (int, float)):
            self._sigma_specs = [SigmaSpec(block_size=1, values=[float(values)], fixed=fixed)]
        else:
            arr = np.array(values, dtype=float)
            if arr.ndim == 1:
                self._sigma_specs = [
                    SigmaSpec(block_size=1, values=[float(v)], fixed=fixed) for v in arr
                ]
            elif arr.ndim == 2:
                n = arr.shape[0]
                lower_tri = []
                for row in range(n):
                    for col in range(row + 1):
                        lower_tri.append(float(arr[row, col]))
                self._sigma_specs = [SigmaSpec(block_size=n, values=lower_tri, fixed=fixed)]
        return self

    def estimation(self, method: str = "FOCE", **kwargs: Any) -> ModelBuilder:
        """
        Set estimation method and options.

        Args:
            method:  Method name (FO, FOCE, FOCEI, LAPLACIAN, SAEM, IMP).
            **kwargs: Method-specific options (maxeval, interaction, sigdig, etc.)
        """
        self._estimation_kwargs = {"method": method, **kwargs}
        return self

    def covariance(self, matrix: str = "SR", **kwargs: Any) -> ModelBuilder:
        """Enable covariance step after estimation."""
        self._do_covariance = True
        self._covariance_kwargs = {"matrix": matrix, **kwargs}
        return self

    def covariates(self, columns: list[str]) -> ModelBuilder:
        """Specify time-varying covariate column names."""
        self._covariate_columns = columns
        return self

    def impute_covariates(
        self,
        columns: list[str],
        method: str = "locf",
    ) -> ModelBuilder:
        """
        Impute missing covariate values after loading the dataset.

        Applied automatically during :meth:`build` before model assembly.

        Args:
            columns: Column names to impute (must exist in the dataset).
            method:  Imputation strategy: ``'locf'`` (default), ``'nocb'``,
                     ``'mean'``, ``'median'``, or ``'knn'``.

        Returns:
            self (fluent interface).
        """
        self._impute_columns = list(columns)
        self._impute_method = method
        return self

    def clone(self) -> "ModelBuilder":
        """Return a copy of this builder. Callable attributes are shared by reference."""
        import copy
        new = ModelBuilder.__new__(ModelBuilder)
        for k, v in self.__dict__.items():
            if callable(v):
                setattr(new, k, v)  # share callable by reference
            else:
                try:
                    setattr(new, k, copy.deepcopy(v))
                except Exception:
                    setattr(new, k, v)  # fallback: share by reference
        return new

    def build(self) -> BuiltModel:
        """
        Assemble and validate the model, returning a BuiltModel ready to fit.
        """
        # Load dataset
        if self._dataset is None:
            if self._data_path is None:
                raise ModelError("No data specified. Call .data() first.")
            self._dataset = NONMEMDataset.from_csv(self._data_path)

        # Validate covariate columns against dataset
        if self._covariate_columns and self._dataset is not None:
            ds_cols = list(self._dataset.df.columns)
            missing = [c for c in self._covariate_columns if c not in ds_cols]
            if missing:
                raise ConfigurationError(
                    f"Covariate column(s) {missing} not found in dataset. "
                    f"Available columns: {ds_cols}"
                )

        # Apply covariate imputation if requested
        if self._impute_columns:
            self._dataset = self._dataset.impute_covariates(
                self._impute_columns, method=self._impute_method
            )
            logger.debug(
                "Imputed %d covariate column(s) using method=%r",
                len(self._impute_columns),
                self._impute_method,
            )

        # Build parameter specs
        if not self._theta_specs:
            raise ModelError("No THETA parameters specified. Call .theta() first.")
        if not self._omega_specs:
            raise ModelError("No OMEGA specified. Call .omega() first.")
        if not self._sigma_specs:
            raise ModelError("No SIGMA specified. Call .sigma() first.")

        params = ParameterSet.from_specs(self._theta_specs, self._omega_specs, self._sigma_specs)

        # Get PK subroutine
        pk_sub = get_advan(self._advan)
        if self._subroutine_kwargs:
            for key, value in self._subroutine_kwargs.items():
                if not hasattr(pk_sub, key):
                    raise ModelError(
                        f"ADVAN{self._advan} does not support subroutine option {key!r}"
                    )
                setattr(pk_sub, key, value)

        # Compile code blocks
        compiler = NMTRANCompiler()
        pk_callable = None
        error_callable = None

        if self._pk_code:
            pk_callable = compiler.compile_pk(self._pk_code)
        if self._error_code:
            error_callable = compiler.compile_error(self._error_code)

        # Compile $DES block for ODE models (ADVAN6, ADVAN8, etc.)
        des_callable = None
        if self._des_code:
            n_cmt = pk_sub.n_compartments
            des_callable = compiler.compile_des(self._des_code, n_compartments=n_cmt)

        # Assemble population model
        pop_model = PopulationModel(
            dataset=self._dataset,
            pk_subroutine=pk_sub,
            params=params,
            pk_callable=pk_callable,
            error_callable=error_callable,
            des_callable=des_callable,
            trans=self._trans,
            advan=self._advan,
            covariate_columns=self._covariate_columns,
        )

        built = BuiltModel(
            population_model=pop_model,
            params=params,
            estimation_kwargs=dict(self._estimation_kwargs),
            do_covariance=self._do_covariance,
            covariance_kwargs=dict(self._covariance_kwargs),
        )
        # Expose pk_code on the BuiltModel for SCM reference
        built._pk_code = self._pk_code or ""  # type: ignore[attr-defined]
        return built
