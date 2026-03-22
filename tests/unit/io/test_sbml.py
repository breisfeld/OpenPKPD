"""Unit tests for the SBML importer."""

from __future__ import annotations

import types

import pytest


class _FakeSBMLError:
    def __init__(self, message: str, severity: int):
        self._message = message
        self._severity = severity

    def getMessage(self):
        return self._message

    def getSeverity(self):
        return self._severity


class _FakeCompartment:
    def __init__(self, cid: str, volume: float | None = None):
        self._id = cid
        self._volume = volume

    def getId(self):
        return self._id

    def isSetVolume(self):
        return self._volume is not None

    def getVolume(self):
        return self._volume


class _FakeParameter:
    def __init__(self, pid: str, value: float | None = None):
        self._id = pid
        self._value = value

    def getId(self):
        return self._id

    def isSetValue(self):
        return self._value is not None

    def getValue(self):
        return self._value


class _FakeSpecies:
    def __init__(
        self,
        sid: str,
        compartment: str,
        *,
        initial_amount: float | None = None,
        initial_concentration: float | None = None,
    ):
        self._id = sid
        self._compartment = compartment
        self._initial_amount = initial_amount
        self._initial_concentration = initial_concentration

    def getId(self):
        return self._id

    def getCompartment(self):
        return self._compartment

    def isSetInitialAmount(self):
        return self._initial_amount is not None

    def getInitialAmount(self):
        return self._initial_amount

    def isSetInitialConcentration(self):
        return self._initial_concentration is not None

    def getInitialConcentration(self):
        return self._initial_concentration


class _FakeSpeciesRef:
    def __init__(self, species: str, stoichiometry: float | None = None):
        self._species = species
        self._stoichiometry = stoichiometry

    def getSpecies(self):
        return self._species

    def isSetStoichiometry(self):
        return self._stoichiometry is not None

    def getStoichiometry(self):
        return self._stoichiometry


class _FakeKineticLaw:
    def __init__(self, formula: str):
        self._formula = formula

    def getMath(self):
        return self._formula


class _FakeReaction:
    def __init__(
        self,
        rid: str,
        formula: str | None,
        *,
        reactants: list[_FakeSpeciesRef] | None = None,
        products: list[_FakeSpeciesRef] | None = None,
    ):
        self._id = rid
        self._formula = formula
        self._reactants = reactants or []
        self._products = products or []

    def getId(self):
        return self._id

    def getKineticLaw(self):
        if self._formula is None:
            return None
        return _FakeKineticLaw(self._formula)

    def getNumReactants(self):
        return len(self._reactants)

    def getReactant(self, idx: int):
        return self._reactants[idx]

    def getNumProducts(self):
        return len(self._products)

    def getProduct(self, idx: int):
        return self._products[idx]


class _FakeModel:
    def __init__(
        self,
        *,
        compartments: list[_FakeCompartment],
        parameters: list[_FakeParameter],
        species: list[_FakeSpecies],
        reactions: list[_FakeReaction],
        num_rules: int = 0,
        num_events: int = 0,
        num_constraints: int = 0,
    ):
        self._compartments = compartments
        self._parameters = parameters
        self._species = species
        self._reactions = reactions
        self._num_rules = num_rules
        self._num_events = num_events
        self._num_constraints = num_constraints

    def getNumCompartments(self):
        return len(self._compartments)

    def getCompartment(self, idx: int):
        return self._compartments[idx]

    def getNumParameters(self):
        return len(self._parameters)

    def getParameter(self, idx: int):
        return self._parameters[idx]

    def getNumSpecies(self):
        return len(self._species)

    def getSpecies(self, idx: int):
        return self._species[idx]

    def getNumReactions(self):
        return len(self._reactions)

    def getReaction(self, idx: int):
        return self._reactions[idx]

    def getNumRules(self):
        return self._num_rules

    def getNumEvents(self):
        return self._num_events

    def getNumConstraints(self):
        return self._num_constraints


class _FakeDoc:
    def __init__(self, model: _FakeModel, errors: list[_FakeSBMLError] | None = None):
        self._model = model
        self._errors = errors or []

    def getNumErrors(self):
        return len(self._errors)

    def getError(self, idx: int):
        return self._errors[idx]

    def getModel(self):
        return self._model


class _FakeSBMLReader:
    def __init__(self, doc: _FakeDoc):
        self._doc = doc

    def readSBMLFromFile(self, _path: str):
        return self._doc


def _install_fake_libsbml(monkeypatch, doc: _FakeDoc) -> None:
    import sys

    libsbml = types.SimpleNamespace(
        LIBSBML_SEV_ERROR=2,
        formulaToL3String=lambda math_ast: math_ast,
        SBMLReader=lambda: _FakeSBMLReader(doc),
    )
    monkeypatch.setitem(sys.modules, "libsbml", libsbml)


class TestSBMLImportError:
    """load_sbml raises ImportError when libsbml is absent."""

    def test_import_error_without_libsbml(self, monkeypatch):
        import sys

        # Remove libsbml from sys.modules if present so we can test the ImportError path
        monkeypatch.setitem(sys.modules, "libsbml", None)  # type: ignore[arg-type]

        from openpkpd.io.sbml import load_sbml

        with pytest.raises(ImportError, match="python-libsbml"):
            load_sbml("nonexistent.xml")


class TestSBMLFormulaTranslation:
    """Test internal _sbml_formula_to_python without requiring libsbml."""

    def _translate(self, formula, species_index=None, parameter_names=None):
        from openpkpd.io.sbml import _sbml_formula_to_python

        return _sbml_formula_to_python(
            formula,
            species_index=species_index or {},
            parameter_names=parameter_names or [],
        )

    def test_empty_formula(self):
        assert self._translate("") == "0.0"

    def test_species_substitution(self):
        result = self._translate(
            "A_tumor * kdrug",
            species_index={"A_tumor": 0},
            parameter_names=["kdrug"],
        )
        assert "A[0]" in result
        assert "pk_params['kdrug']" in result

    def test_parameter_substitution(self):
        result = self._translate(
            "kgrow - kdrug",
            species_index={},
            parameter_names=["kgrow", "kdrug"],
        )
        assert "pk_params['kgrow']" in result
        assert "pk_params['kdrug']" in result

    def test_power_operator(self):
        result = self._translate("A_drug ^ 2", species_index={"A_drug": 1}, parameter_names=[])
        assert "**" in result

    def test_multiple_species(self):
        result = self._translate(
            "A_drug * A_effect",
            species_index={"A_drug": 0, "A_effect": 1},
            parameter_names=[],
        )
        assert "A[0]" in result
        assert "A[1]" in result

    def test_no_partial_name_replacement(self):
        """'k' parameter should not partially replace 'kgrow'."""
        result = self._translate(
            "kgrow + k",
            species_index={},
            parameter_names=["k", "kgrow"],
        )
        # Both should appear as full pk_params references
        assert "pk_params['kgrow']" in result
        assert "pk_params['k']" in result


class TestSBMLModelDataclass:
    """Test SBMLModel without file I/O."""

    def _make_model(self):
        from openpkpd.io.sbml import SBMLModel, _build_des_callable

        species_names = ["A_central", "A_peripheral"]
        species_index = {"A_central": 0, "A_peripheral": 1}
        parameter_names = ["CL", "V", "Q", "V2"]
        default_pk_params = {"CL": 3.0, "V": 10.0, "Q": 1.0, "V2": 20.0}

        # Two-compartment model expressions
        dadt_exprs = {
            0: [
                "-(pk_params['CL'] / pk_params['V']) * A[0]",
                "-(pk_params['Q'] / pk_params['V']) * A[0]",
                "+(pk_params['Q'] / pk_params['V2']) * A[1]",
            ],
            1: [
                "+(pk_params['Q'] / pk_params['V']) * A[0]",
                "-(pk_params['Q'] / pk_params['V2']) * A[1]",
            ],
        }

        des_callable = _build_des_callable(
            dadt_exprs, species_names, species_index, parameter_names, 2, []
        )

        return SBMLModel(
            species_names=species_names,
            parameter_names=parameter_names,
            default_pk_params=default_pk_params,
            initial_amounts={"A_central": 0.0, "A_peripheral": 0.0},
            n_compartments=2,
            des_callable=des_callable,
            source_path="test",
        )

    def test_des_callable_returns_correct_length(self):
        model = self._make_model()
        A = [100.0, 0.0]
        pk = {"CL": 3.0, "V": 10.0, "Q": 1.0, "V2": 20.0}
        dadt = model.des_callable(0.0, A, pk, [], [])
        assert len(dadt) == 2

    def test_des_callable_mass_balance(self):
        """dA_central/dt + dA_peripheral/dt = -elimination."""
        model = self._make_model()
        A = [100.0, 50.0]
        pk = {"CL": 3.0, "V": 10.0, "Q": 1.0, "V2": 20.0}
        dadt = model.des_callable(0.0, A, pk, [], [])
        # Net change = -CL/V * A_central
        expected_net = -(3.0 / 10.0) * 100.0
        assert sum(dadt) == pytest.approx(expected_net, rel=1e-6)

    def test_to_theta_specs(self):
        model = self._make_model()
        specs = model.to_theta_specs()
        assert len(specs) == len(model.parameter_names)
        for spec in specs:
            assert hasattr(spec, "init")
            assert hasattr(spec, "lower")

    def test_pk_callable_from_theta(self):
        model = self._make_model()
        theta = [3.5, 11.0, 1.2, 22.0]
        pk = model.pk_callable_from_theta(theta)
        assert pk == dict(zip(model.parameter_names, theta, strict=False))

    def test_n_compartments(self):
        model = self._make_model()
        assert model.n_compartments == 2

    def test_des_compatible_with_advan6(self):
        """DES callable from SBMLModel works with ADVAN6."""
        import numpy as np

        from openpkpd.data.event_processor import DoseEvent
        from openpkpd.pk.ode.advan6 import ADVAN6

        model = self._make_model()
        advan = ADVAN6(n_compartments=2)
        dose_events = [DoseEvent(time=0.0, amount=100.0, rate=0.0, duration=0.0, compartment=1)]
        obs_times = np.array([1.0, 2.0, 4.0, 8.0])

        sol = advan.solve(
            pk_params=model.default_pk_params,
            dose_events=dose_events,
            obs_times=obs_times,
            des_callable=model.des_callable,
        )
        assert sol.ipred.shape == (4,)
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0.0)


class TestSBMLLoaderParsing:
    def test_load_sbml_builds_exact_stoichiometric_des_and_initial_amounts(self, monkeypatch):
        model = _FakeModel(
            compartments=[_FakeCompartment("c", volume=4.0)],
            parameters=[_FakeParameter("k", 0.25)],
            species=[
                _FakeSpecies("S", "c", initial_concentration=3.0),
                _FakeSpecies("P", "c", initial_amount=1.0),
            ],
            reactions=[
                _FakeReaction(
                    "r1",
                    "k * S",
                    reactants=[_FakeSpeciesRef("S", 1.0)],
                    products=[_FakeSpeciesRef("P", 2.0)],
                )
            ],
        )
        _install_fake_libsbml(monkeypatch, _FakeDoc(model))

        from openpkpd.io.sbml import load_sbml

        loaded = load_sbml("fake.xml")
        assert loaded.initial_amounts == {"S": pytest.approx(12.0), "P": pytest.approx(1.0)}

        dadt = loaded.des_callable(0.0, [12.0, 1.0], loaded.default_pk_params, [], [])
        assert dadt == pytest.approx([-3.0, 6.0], rel=1e-12)

    def test_load_sbml_collects_nonfatal_parser_warnings(self, monkeypatch):
        model = _FakeModel(
            compartments=[_FakeCompartment("c")],
            parameters=[],
            species=[_FakeSpecies("S", "c", initial_amount=0.0)],
            reactions=[],
        )
        doc = _FakeDoc(model, errors=[_FakeSBMLError("unit mismatch", severity=1)])
        _install_fake_libsbml(monkeypatch, doc)

        from openpkpd.io.sbml import load_sbml

        with pytest.warns(UserWarning, match="unit mismatch"):
            loaded = load_sbml("fake.xml")

        assert any("unit mismatch" in warning for warning in loaded.warnings)

    def test_load_sbml_warns_for_unsupported_elements_and_missing_kinetics(self, monkeypatch):
        model = _FakeModel(
            compartments=[_FakeCompartment("c")],
            parameters=[],
            species=[_FakeSpecies("S", "c", initial_amount=0.0)],
            reactions=[_FakeReaction("r_missing", None)],
            num_rules=1,
            num_events=2,
            num_constraints=3,
        )
        _install_fake_libsbml(monkeypatch, _FakeDoc(model))

        from openpkpd.io.sbml import load_sbml

        with pytest.warns(UserWarning) as record:
            loaded = load_sbml("fake.xml")

        messages = [str(item.message) for item in record]
        assert any("no kinetic law" in message for message in messages)
        assert any("1 rule(s)" in message for message in messages)
        assert any("2 event(s)" in message for message in messages)
        assert any("3 constraint(s)" in message for message in messages)
        assert any("no kinetic law" in warning for warning in loaded.warnings)
