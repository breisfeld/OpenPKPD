"""Tests for ModelBuilder.clone() — SC2."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from openpkpd.api.model_builder import ModelBuilder
from openpkpd.model.parameters import ParameterSet


def _configured_builder() -> ModelBuilder:
    """Return a fully-configured ModelBuilder (without dataset, since we test clone logic)."""
    builder = ModelBuilder()
    builder.problem("Test 1-cmt IV")
    builder.subroutines(advan=1, trans=2)
    builder.pk("CL = THETA(1) * EXP(ETA(1))\nV = THETA(2) * EXP(ETA(2))")
    builder.error("Y = F * (1 + EPS(1))")
    builder.theta([(0.01, 0.1, 10), (1, 30, 500)])
    builder.omega([[0.3, 0], [0, 0.3]])
    builder.sigma([[0.05]])
    builder.estimation(method="FOCE", maxeval=9999)
    return builder


class TestModelBuilderClone:
    def test_clone_is_independent(self):
        """Modifying the clone does not affect the original builder."""
        original = _configured_builder()
        cloned = original.clone()

        # Modify clone's title and theta specs
        cloned._title = "MODIFIED"
        cloned._theta_specs = []

        assert original._title != "MODIFIED", "Original title should not change"
        assert len(original._theta_specs) == 2, "Original theta specs should not change"

    def test_clone_with_lambda_does_not_raise(self):
        """Clone of a builder with a lambda covariate effect does not raise."""
        builder = _configured_builder()
        # Assign a lambda to a builder attribute to simulate non-picklable callables
        builder._some_callable = lambda x: x * 2  # type: ignore[attr-defined]

        clone = builder.clone()  # should not raise
        assert clone is not None

    def test_clone_dataset_is_independent(self):
        """Dataset in the clone is a different object with the same content."""
        import pandas as pd

        from openpkpd.data.dataset import NONMEMDataset

        df = pd.DataFrame({
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 1.0],
            "AMT": [100.0, 0.0],
            "EVID": [1, 0],
        })
        ds = NONMEMDataset(df)

        builder = _configured_builder()
        builder._dataset = ds

        cloned = builder.clone()

        # Must be different objects
        assert cloned._dataset is not builder._dataset
        # Must have same content
        assert cloned._dataset.df.equals(builder._dataset.df)

        # Modifying clone's dataset should not affect original
        cloned._dataset._df = cloned._dataset.df.copy()
        cloned._dataset._df["DV"] = 999.0
        assert builder._dataset.df["DV"].iloc[1] == pytest.approx(1.0)

    def test_deepcopy_with_non_picklable_raises(self):
        """deepcopy of a builder with a non-picklable object raises (demonstrates why clone is needed).

        Uses a threading.Lock which is not picklable/deepcopy-able.
        """
        import threading

        builder = _configured_builder()
        builder._lock = threading.Lock()  # type: ignore[attr-defined]

        with pytest.raises((TypeError, AttributeError, Exception)):
            copy.deepcopy(builder)

    def test_clone_with_non_picklable_does_not_raise(self):
        """Clone of a builder with a non-picklable lock does not raise."""
        import threading

        builder = _configured_builder()
        builder._lock = threading.Lock()  # type: ignore[attr-defined]

        cloned = builder.clone()  # should not raise
        assert cloned is not None

    def test_clone_theta_specs_are_deep_copies(self):
        """ThetaSpec list in the clone is a separate copy."""
        original = _configured_builder()
        cloned = original.clone()

        # Appending to clone's theta_specs must not affect original
        from openpkpd.model.parameters import ThetaSpec

        cloned._theta_specs.append(ThetaSpec(init=99.0))
        assert len(original._theta_specs) == 2, (
            "Original theta_specs was mutated by appending to clone"
        )

    def test_clone_estimation_kwargs_are_deep_copies(self):
        """Estimation kwargs dict in clone is independent from original."""
        original = _configured_builder()
        cloned = original.clone()

        cloned._estimation_kwargs["maxeval"] = 1
        assert original._estimation_kwargs.get("maxeval", 9999) != 1, (
            "Original estimation kwargs was mutated through clone"
        )

    def test_2d_omega_and_sigma_inputs_preserve_matrix_structure(self):
        builder = ModelBuilder()
        builder.omega([[0.3, 0.0], [0.0, 0.2]])
        builder.sigma([[0.1, 0.0], [0.0, 0.05]])

        params = ParameterSet.from_specs([], builder._omega_specs, builder._sigma_specs)

        np.testing.assert_allclose(params.omega, np.array([[0.3, 0.0], [0.0, 0.2]]))
        np.testing.assert_allclose(params.sigma, np.array([[0.1, 0.0], [0.0, 0.05]]))
