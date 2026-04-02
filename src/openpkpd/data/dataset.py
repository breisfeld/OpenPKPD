"""
NONMEMDataset: CSV loading, column validation, and standard preprocessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from openpkpd.data.columns import (
    ADDL,
    AMT,
    CMT,
    DV,
    EVID,
    ID,
    II,
    MDV,
    RATE,
    REQUIRED_COLUMNS,
    SS,
    TIME,
)
from openpkpd.utils.constants import EVID_DOSE, EVID_OBS, MDV_OBS, NONMEM_MISSING
from openpkpd.utils.errors import DataError


class DatasetValidationError(ValueError):
    """Raised when a required dataset column has an invalid dtype (e.g., strings in ID/TIME/DV)."""


@dataclass
class NONMEMDataset:
    """
    A validated, preprocessed NONMEM dataset.

    After construction:
      - All columns are renamed to NONMEM standard names
      - Missing numeric values (-99 or user-specified) replaced with NaN
      - EVID column added (default 0 = observation) if absent
      - MDV column added based on EVID if absent
      - CMT column added (default 1) if absent
      - Rows sorted by ID, then TIME
    """

    df: pd.DataFrame
    source_path: str | None = None
    ignore_char: str | None = None  # Character at start of line to skip
    column_map: dict[str, str] = field(default_factory=dict)  # original → standard

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str,
        input_columns: list[str] | None = None,
        ignore_char: str | None = None,
        missing_value: float = NONMEM_MISSING,
        sep: str = ",",
        impute_covariates: list[str] | None = None,
        impute_method: str = "locf",
    ) -> NONMEMDataset:
        """
        Load a NONMEM dataset from a CSV file.

        Args:
            path:              Path to CSV file.
            input_columns:     Ordered list of standard column names (from $INPUT).
                               If None, treat first row as header.
            ignore_char:       Ignore lines where the first data character matches this.
            missing_value:     Numeric value treated as missing (default -99).
            sep:               Column separator (default comma; use r'\\s+' for whitespace).
            impute_covariates: Column names to impute after loading.  When ``None``
                               (default) no imputation is performed.
            impute_method:     Imputation strategy passed to
                               :class:`~openpkpd.data.impute.CovariateImputer`:
                               ``'locf'`` (default), ``'nocb'``, ``'mean'``,
                               ``'median'``, or ``'knn'``.
        """
        if not os.path.exists(path):
            raise DataError(f"Dataset file not found: {path}")

        # Read raw CSV
        try:
            if sep == r"\s+":
                raw_df = pd.read_csv(
                    path,
                    sep=r"\s+",
                    engine="python",
                    comment=None,
                    header=0 if input_columns is None else None,
                    na_values=[str(missing_value)],
                )
            else:
                raw_df = pd.read_csv(
                    path,
                    sep=sep,
                    comment=None,
                    header=0 if input_columns is None else None,
                    na_values=[str(missing_value)],
                )
        except Exception as exc:
            raise DataError(f"Failed to read dataset {path!r}: {exc}") from exc

        # Remove ignored lines (e.g., IGNORE=@)
        if ignore_char is not None:
            first_col = raw_df.columns[0]
            mask = raw_df[first_col].astype(str).str.startswith(ignore_char)
            raw_df = raw_df[~mask].reset_index(drop=True)

        # Apply $INPUT column names if provided
        column_map: dict[str, str] = {}
        if input_columns is not None:
            input_columns = list(input_columns)
            if len(input_columns) > len(raw_df.columns):
                raise DataError(
                    f"$INPUT specifies {len(input_columns)} columns but dataset has "
                    f"{len(raw_df.columns)} columns"
                )
            raw_df = raw_df.iloc[:, : len(input_columns)].copy()
            rename = dict(zip(raw_df.columns, input_columns, strict=False))
            raw_df = raw_df.rename(columns=rename)
            raw_df = raw_df.drop(
                columns=[c for c in raw_df.columns if str(c).startswith("_DROP_")], errors="ignore"
            )
            column_map = {str(old): str(new) for old, new in rename.items()}
        else:
            # Normalise header names to uppercase
            rename = {c: c.upper() for c in raw_df.columns}
            raw_df = raw_df.rename(columns=rename)

        df = cls._validate_and_preprocess(raw_df)
        ds = cls(
            df=df, source_path=os.path.abspath(path), ignore_char=ignore_char, column_map=column_map
        )
        if impute_covariates:
            ds = ds.impute_covariates(impute_covariates, method=impute_method)
        return ds

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        missing_value: float = NONMEM_MISSING,
        impute_covariates: list[str] | None = None,
        impute_method: str = "locf",
    ) -> NONMEMDataset:
        """
        Construct from an already-loaded pandas DataFrame.

        Args:
            df:                Input DataFrame.
            missing_value:     Numeric sentinel for missing values (default -99).
            impute_covariates: Column names to impute after preprocessing.
                               When ``None`` (default) no imputation is performed.
            impute_method:     Imputation strategy: ``'locf'`` (default), ``'nocb'``,
                               ``'mean'``, ``'median'``, or ``'knn'``.
        """
        df = df.rename(columns={c: c.upper() for c in df.columns})
        df = df.replace(missing_value, float("nan"))
        df = cls._validate_and_preprocess(df)
        ds = cls(df=df)
        if impute_covariates:
            ds = ds.impute_covariates(impute_covariates, method=impute_method)
        return ds

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @classmethod
    def _validate_and_preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Validate required columns
        missing_req = REQUIRED_COLUMNS - set(df.columns)
        if missing_req:
            raise DataError(f"Dataset missing required columns: {missing_req}")

        # Validate that critical columns are numeric before coercion.
        # String values in these columns (e.g., UUID IDs, "BLQ" in DV) would be
        # silently coerced to NaN by pd.to_numeric, producing misleading results.
        # Note: NONMEM uses "." as a standard missing-value placeholder in numeric
        # columns (e.g., DV=. on dosing rows); replace it with NaN before checking
        # so it is not counted as a non-numeric value.
        _NONMEM_MISSING = "."
        _critical_cols = [ID, TIME, DV, AMT, EVID]
        for _col in _critical_cols:
            if _col not in df.columns:
                continue
            # Replace the NONMEM missing-value placeholder before validation.
            _col_data = df[_col].replace(_NONMEM_MISSING, np.nan)
            _numeric_series = pd.to_numeric(_col_data, errors="coerce")
            _n_bad = int(_numeric_series.isna().sum()) - int(_col_data.isna().sum())
            if _n_bad > 0:
                raise DatasetValidationError(
                    f"Column '{_col}' must be numeric but got dtype {df[_col].dtype} "
                    f"with {_n_bad} non-numeric value(s). "
                    f"Non-numeric IDs (e.g., UUID strings) are not supported."
                )

        # Ensure numeric types for standard columns
        numeric_cols = [ID, TIME, DV, AMT, RATE, EVID, MDV, CMT, ADDL, II, SS, "LLOQ"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add EVID if missing: infer from AMT (NONMEM convention)
        if EVID not in df.columns:
            if AMT in df.columns:
                amt_col = pd.to_numeric(df[AMT], errors="coerce")
                df[EVID] = np.where(amt_col.notna() & (amt_col != 0), EVID_DOSE, EVID_OBS)
            else:
                df[EVID] = EVID_OBS

        # Add MDV if missing: MDV=1 for dosing rows (EVID≠0), MDV=0 for observations
        if MDV not in df.columns:
            df[MDV] = (df[EVID] != EVID_OBS).astype(int)

        # Add CMT if missing (default compartment 1)
        if CMT not in df.columns:
            df[CMT] = 1

        # Add AMT if missing
        if AMT not in df.columns:
            df[AMT] = 0.0

        # Add RATE if missing
        if RATE not in df.columns:
            df[RATE] = 0.0

        # Add ADDL/II/SS if missing
        for col in (ADDL, II, SS):
            if col not in df.columns:
                df[col] = 0

        # Ensure ID is integer-compatible (may be string labels)
        try:
            df[ID] = df[ID].astype(int)
        except (ValueError, TypeError):
            # String IDs: factorize to integers, keep original in ID_LABEL
            df["ID_LABEL"] = df[ID].astype(str)
            df[ID] = pd.factorize(df[ID])[0] + 1

        # Sort by ID then TIME
        df = df.sort_values([ID, TIME], kind="stable").reset_index(drop=True)

        return df

    # ── Subject accessors ─────────────────────────────────────────────────────

    def subject_ids(self) -> list[int]:
        """Return sorted list of unique subject IDs."""
        return sorted(self.df[ID].unique().tolist())

    def subject_data(self, subject_id: int) -> pd.DataFrame:
        """Return all rows for a single subject."""
        return self.df[self.df[ID] == subject_id].reset_index(drop=True)

    def observation_rows(self, subject_id: int | None = None) -> pd.DataFrame:
        """Return observation rows (EVID=0, MDV=0)."""
        df = self.df if subject_id is None else self.subject_data(subject_id)
        return df[(df[EVID] == EVID_OBS) & (df[MDV] == MDV_OBS)].reset_index(drop=True)

    def n_subjects(self) -> int:
        return len(self.subject_ids())

    def n_observations(self) -> int:
        return int(((self.df[EVID] == EVID_OBS) & (self.df[MDV] == MDV_OBS)).sum())

    # ── LLOQ / BLQ support ────────────────────────────────────────────────

    @property
    def has_lloq(self) -> bool:
        """
        Return True if an LLOQ column is present in the dataset.

        The LLOQ column must exist and contain at least one non-NaN value
        for this property to return True.
        """
        return "LLOQ" in self.df.columns and self.df["LLOQ"].notna().any()

    def lloq_values(self, subject_id: int | None = None) -> np.ndarray:
        """
        Return LLOQ values for the dataset or a single subject.

        Only observation rows (EVID=0, MDV=0) are returned. If no LLOQ
        column is present, an array of NaN is returned with the same
        length as the observation rows.

        Args:
            subject_id: If provided, return LLOQ values only for this
                        subject. If None, return values for all subjects.

        Returns:
            NumPy array of LLOQ values, shape (n_obs_rows,).
        """
        obs = self.observation_rows(subject_id)
        if "LLOQ" in obs.columns:
            return obs["LLOQ"].to_numpy(dtype=float)
        return np.full(len(obs), float("nan"))

    # ── Covariate imputation ───────────────────────────────────────────────

    def impute_covariates(
        self,
        columns: list[str],
        method: str = "locf",
    ) -> NONMEMDataset:
        """
        Return a new NONMEMDataset with missing covariate values imputed.

        Args:
            columns: Column names to impute.
            method:  Strategy passed to :class:`~openpkpd.data.impute.CovariateImputer`:
                     ``'mean'``, ``'median'``, ``'locf'`` (default),
                     ``'nocb'``, or ``'knn'``.

        Returns:
            New NONMEMDataset; original is unchanged.
        """
        from openpkpd.data.impute import CovariateImputer

        new_df = CovariateImputer.fit_transform(self.df, columns=columns, method=method)
        return NONMEMDataset(
            df=new_df,
            source_path=self.source_path,
            ignore_char=self.ignore_char,
            column_map=self.column_map,
        )

    def __repr__(self) -> str:
        return (
            f"NONMEMDataset(n_subjects={self.n_subjects()}, "
            f"n_rows={len(self.df)}, "
            f"columns={list(self.df.columns)})"
        )
