"""
CovariateImputer — fill missing covariate values in a NONMEM dataset.

Supported methods
-----------------
``mean``    — column mean across all subjects
``median``  — column median across all subjects
``locf``    — last-observation-carried-forward per subject (forward-fill)
``nocb``    — next-observation-carried-backward per subject (back-fill)
``knn``     — k-nearest-neighbours imputation (requires scikit-learn)
"""

from __future__ import annotations

import pandas as pd


class CovariateImputer:
    """Impute missing covariate values in a NONMEM-style DataFrame."""

    @classmethod
    def fit_transform(
        cls,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "locf",
        id_col: str = "ID",
        **knn_kwargs: object,
    ) -> pd.DataFrame:
        """
        Return a copy of *df* with missing values in *columns* imputed.

        Args:
            df:         Input DataFrame (rows = NONMEM records).
            columns:    Column names to impute.
            method:     Imputation strategy: ``'mean'``, ``'median'``,
                        ``'locf'``, ``'nocb'``, or ``'knn'``.
            id_col:     Column containing subject ID (used for LOCF/NOCB).
            **knn_kwargs: Extra keyword arguments forwarded to
                          ``sklearn.impute.KNNImputer`` (e.g. ``n_neighbors``).

        Returns:
            New DataFrame with imputed values in the requested columns.

        Raises:
            ValueError:   Unknown *method*.
            ImportError:  ``method='knn'`` and scikit-learn is not installed.
        """
        df = df.copy()
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        if method == "mean":
            for col in columns:
                df[col] = df[col].fillna(df[col].mean())

        elif method == "median":
            for col in columns:
                df[col] = df[col].fillna(df[col].median())

        elif method == "locf":
            if id_col in df.columns:
                df[columns] = df.groupby(id_col, sort=False)[columns].ffill()
            else:
                df[columns] = df[columns].ffill()

        elif method == "nocb":
            if id_col in df.columns:
                df[columns] = df.groupby(id_col, sort=False)[columns].bfill()
            else:
                df[columns] = df[columns].bfill()

        elif method == "knn":
            try:
                from sklearn.impute import KNNImputer
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn is required for KNN imputation. "
                    "Install it with: pip install scikit-learn"
                ) from exc
            imputer = KNNImputer(**knn_kwargs)
            df[columns] = imputer.fit_transform(df[columns])

        else:
            raise ValueError(
                f"Unknown imputation method {method!r}. "
                "Choose from: 'mean', 'median', 'locf', 'nocb', 'knn'."
            )

        return df
