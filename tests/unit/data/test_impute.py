"""Tests for CovariateImputer and NONMEMDataset.impute_covariates()."""

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.impute import CovariateImputer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sparse_df():
    """Small DataFrame with missing covariate values."""
    return pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "TIME": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "WT": [70.0, np.nan, np.nan, 60.0, np.nan, np.nan, np.nan, 80.0, np.nan],
            "AGE": [30.0, 30.0, np.nan, 45.0, np.nan, 45.0, np.nan, np.nan, 55.0],
        }
    )


# ---------------------------------------------------------------------------
# CovariateImputer unit tests
# ---------------------------------------------------------------------------


class TestCovariateImputer:
    def test_mean(self, sparse_df):
        result = CovariateImputer.fit_transform(sparse_df, ["WT"], method="mean")
        # No remaining NaN in WT
        assert result["WT"].isna().sum() == 0
        # Mean of observed values: (70 + 60 + 80) / 3 = 70
        assert np.isclose(result["WT"].mean(), 70.0)

    def test_median(self, sparse_df):
        result = CovariateImputer.fit_transform(sparse_df, ["WT"], method="median")
        assert result["WT"].isna().sum() == 0
        # Median = 70 (observed: 70, 60, 80)
        assert np.isclose(result["WT"].iloc[1], 70.0)

    def test_locf(self, sparse_df):
        result = CovariateImputer.fit_transform(sparse_df, ["WT"], method="locf")
        # Subject 1: NaN after 70 → should be filled with 70
        subj1 = result[result["ID"] == 1]["WT"].to_numpy()
        assert np.all(subj1 == 70.0)
        # Subject 2: NaN after 60 → filled with 60
        subj2 = result[result["ID"] == 2]["WT"].to_numpy()
        assert np.all(subj2 == 60.0)
        # Subject 3: leading NaN → still NaN (no prior obs to carry forward)
        subj3 = result[result["ID"] == 3]["WT"]
        assert np.isnan(subj3.iloc[0])

    def test_nocb(self, sparse_df):
        result = CovariateImputer.fit_transform(sparse_df, ["WT"], method="nocb")
        # Subject 3: NaN before 80 → filled with 80
        subj3 = result[result["ID"] == 3]["WT"].to_numpy()
        assert subj3[0] == 80.0
        assert subj3[1] == 80.0

    def test_knn_importerror(self, sparse_df, monkeypatch):
        """KNN raises ImportError when sklearn not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="scikit-learn"):
            CovariateImputer.fit_transform(sparse_df, ["WT"], method="knn")

    def test_unknown_method(self, sparse_df):
        with pytest.raises(ValueError, match="Unknown imputation method"):
            CovariateImputer.fit_transform(sparse_df, ["WT"], method="magic")

    def test_missing_column(self, sparse_df):
        with pytest.raises(ValueError, match="Columns not found"):
            CovariateImputer.fit_transform(sparse_df, ["NOTACOL"], method="mean")

    def test_multiple_columns(self, sparse_df):
        result = CovariateImputer.fit_transform(sparse_df, ["WT", "AGE"], method="mean")
        assert result["WT"].isna().sum() == 0
        assert result["AGE"].isna().sum() == 0

    def test_original_unchanged(self, sparse_df):
        original_nan_count = sparse_df["WT"].isna().sum()
        CovariateImputer.fit_transform(sparse_df, ["WT"], method="mean")
        assert sparse_df["WT"].isna().sum() == original_nan_count


# ---------------------------------------------------------------------------
# NONMEMDataset.impute_covariates integration
# ---------------------------------------------------------------------------


class TestNONMEMDatasetImpute:
    def _make_dataset(self, sparse_df):
        # Add required NONMEM columns
        df = sparse_df.copy()
        df["DV"] = 1.0
        df["EVID"] = 0
        df["MDV"] = 0
        df["AMT"] = 0.0
        return NONMEMDataset.from_dataframe(df)

    def test_returns_new_dataset(self, sparse_df):
        ds = self._make_dataset(sparse_df)
        imputed = ds.impute_covariates(["WT"], method="mean")
        assert imputed is not ds
        assert isinstance(imputed, NONMEMDataset)

    def test_imputed_values(self, sparse_df):
        ds = self._make_dataset(sparse_df)
        imputed = ds.impute_covariates(["WT"], method="locf")
        # Subject 1: all rows should have WT=70 after LOCF
        subj1 = imputed.subject_data(1)["WT"]
        assert np.all(subj1 == 70.0)

    def test_original_dataset_unchanged(self, sparse_df):
        ds = self._make_dataset(sparse_df)
        nan_before = ds.df["WT"].isna().sum()
        ds.impute_covariates(["WT"], method="mean")
        assert ds.df["WT"].isna().sum() == nan_before


class TestFromDataframeImputation:
    """Tests for NONMEMDataset.from_dataframe impute_covariates parameter."""

    def _base_df(self):
        return pd.DataFrame(
            {
                "ID": [1, 1, 2, 2],
                "TIME": [0.0, 1.0, 0.0, 1.0],
                "DV": [1.0, 2.0, 3.0, 4.0],
                "AMT": [0.0, 0.0, 0.0, 0.0],
                "EVID": [0, 0, 0, 0],
                "WT": [70.0, np.nan, np.nan, 60.0],
            }
        )

    def test_from_dataframe_no_imputation_keeps_nan(self):
        ds = NONMEMDataset.from_dataframe(self._base_df())
        assert ds.df["WT"].isna().any()

    def test_from_dataframe_locf_fills_trailing_nan(self):
        ds = NONMEMDataset.from_dataframe(
            self._base_df(), impute_covariates=["WT"], impute_method="locf"
        )
        # Subject 1: WT=70 at t=0, NaN at t=1 → filled to 70
        subj1 = ds.df[ds.df["ID"] == 1]["WT"]
        assert not subj1.isna().any()
        assert (subj1 == 70.0).all()

    def test_from_dataframe_mean_fills_all_nan(self):
        ds = NONMEMDataset.from_dataframe(
            self._base_df(), impute_covariates=["WT"], impute_method="mean"
        )
        assert not ds.df["WT"].isna().any()

    def test_from_dataframe_returns_nonmem_dataset(self):
        ds = NONMEMDataset.from_dataframe(self._base_df(), impute_covariates=["WT"])
        assert isinstance(ds, NONMEMDataset)


class TestModelBuilderImputation:
    """Tests for ModelBuilder.impute_covariates() wired into build()."""

    def _make_builder_with_dataset(self, sparse=True):
        from openpkpd.api.model_builder import ModelBuilder

        df = pd.DataFrame(
            {
                "ID": [1, 1, 2, 2],
                "TIME": [0.0, 1.0, 0.0, 1.0],
                "DV": [1.0, 2.0, 3.0, 4.0],
                "AMT": [0.0, 0.0, 0.0, 0.0],
                "EVID": [0, 0, 0, 0],
                "WT": [70.0, np.nan, np.nan, 60.0] if sparse else [70.0, 70.0, 60.0, 60.0],
            }
        )
        ds = NONMEMDataset.from_dataframe(df)
        builder = (
            ModelBuilder()
            .dataset(ds)
            .subroutines(advan=1, trans=1)
            .pk("CL = THETA(1)")
            .error("Y = F + EPS(1)")
            .theta([1.0])
            .omega([0.1])
            .sigma([0.05])
            .estimation(method="FO")
        )
        return builder, ds

    def test_impute_covariates_method_returns_self(self):
        from openpkpd.api.model_builder import ModelBuilder

        builder = ModelBuilder()
        result = builder.impute_covariates(["WT"])
        assert result is builder

    def test_impute_stores_config(self):
        from openpkpd.api.model_builder import ModelBuilder

        builder = ModelBuilder()
        builder.impute_covariates(["WT", "AGE"], method="mean")
        assert builder._impute_columns == ["WT", "AGE"]
        assert builder._impute_method == "mean"

    def test_build_applies_imputation(self):
        """build() imputes the dataset before assembling the model."""
        builder, ds = self._make_builder_with_dataset(sparse=True)
        # WT has NaN before build
        assert ds.df["WT"].isna().any()

        builder.impute_covariates(["WT"], method="mean")
        built = builder.build()
        # Dataset on the built population model should have no NaN in WT
        model_df = built.population_model.dataset.df
        assert not model_df["WT"].isna().any()

    def test_build_without_imputation_preserves_nan(self):
        """Without .impute_covariates(), NaN values pass through unchanged."""
        builder, ds = self._make_builder_with_dataset(sparse=True)
        built = builder.build()
        model_df = built.population_model.dataset.df
        assert model_df["WT"].isna().any()

    def test_impute_default_method_is_locf(self):
        from openpkpd.api.model_builder import ModelBuilder

        builder = ModelBuilder()
        builder.impute_covariates(["WT"])
        assert builder._impute_method == "locf"
