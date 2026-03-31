from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model import symbolic_eta as sym_eta
from openpkpd.model.individual import IndividualModel
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan4 import ADVAN4


def _clear_symbolic_caches() -> None:
    sym_eta.clear_symbolic_runtime_caches()


def _make_symbolic_advan1_model() -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
        ),
        pk_subroutine=ADVAN1(),
        pk_callable=compiler.compile_pk("CL = THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(2))"),
        error_callable=compiler.compile_error("Y = F + EPS(1)"),
        n_eps=1,
    )


def _make_symbolic_advan1_covariate_model() -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
            covariate_df=pd.DataFrame({"TIME": [0.0], "WT": [70.0]}),
        ),
        pk_subroutine=ADVAN1(),
        pk_callable=compiler.compile_pk(
            "CL = THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(2))\nCL = CL * (WT/70.0)**THETA(3)"
        ),
        error_callable=compiler.compile_error("Y = F + EPS(1)"),
        n_eps=1,
    )


def _make_symbolic_advan3_trans1_model() -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
        ),
        pk_subroutine=ADVAN3(),
        pk_callable=compiler.compile_pk(
            "CL = THETA(1)*EXP(ETA(1))\n"
            "V1 = THETA(2)*EXP(ETA(2))\n"
            "Q = THETA(3)\n"
            "V2 = THETA(4)\n"
            "K = CL/V1\n"
            "K12 = Q/V1\n"
            "K21 = Q/V2"
        ),
        error_callable=compiler.compile_error("Y = F*(1 + EPS(1))"),
        n_eps=1,
    )


def _make_symbolic_advan4_trans4_model() -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=220.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([5.8, 7.1, 7.6, 6.0, 3.2], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
        ),
        pk_subroutine=ADVAN4(),
        pk_callable=compiler.compile_pk(
            "KA = THETA(1)*EXP(ETA(1))\n"
            "CL = THETA(2)*EXP(ETA(2))\n"
            "V2 = THETA(3)*EXP(ETA(3))\n"
            "Q = THETA(4)\n"
            "V3 = THETA(5)"
        ),
        error_callable=compiler.compile_error("Y = F*(1 + EPS(1))"),
        n_eps=1,
    )


def _make_symbolic_advan2_model(
    *,
    pk_code: str = "KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))",
    subject_events: SubjectEvents | None = None,
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=subject_events
        or SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
        ),
        pk_subroutine=ADVAN2(),
        pk_callable=compiler.compile_pk(pk_code),
        error_callable=compiler.compile_error("Y = F*(1 + EPS(1))"),
        n_eps=1,
    )


def _wrap_counter(counts: dict[str, int], name: str, fn: object):
    def _wrapped(*args):
        counts[name] = counts.get(name, 0) + 1
        return fn(*args)

    return _wrapped


def test_runtime_prediction_cache_reuses_last_full_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan1_model()
        kernel = model.get_subject_derivative_kernel(2)
        assert kernel is not None

        terms = sym_eta._compiled_advan1_terms()
        hess_terms = sym_eta._compiled_advan1_hessian_terms()
        counts: dict[str, int] = {}
        monkeypatch.setitem(terms, "contrib", _wrap_counter(counts, "contrib", terms["contrib"]))
        monkeypatch.setitem(
            terms,
            "contrib_grad",
            tuple(
                _wrap_counter(counts, f"grad_{idx}", fn)
                for idx, fn in enumerate(terms["contrib_grad"])
            ),
        )
        monkeypatch.setitem(
            hess_terms,
            "contrib_hess",
            tuple(
                tuple(
                    _wrap_counter(counts, f"hess_{row}{col}", fn) for col, fn in enumerate(hess_row)
                )
                for row, hess_row in enumerate(hess_terms["contrib_hess"])
            ),
        )

        theta = np.array([1.6, 11.5])
        eta = np.array([0.04, -0.03])
        sigma = np.array([[0.05]])

        kernel.eta_data_objective_hessian(theta, eta, sigma)
        counts_after_hessian = counts.copy()

        kernel.prediction_eta_jacobian(theta, eta, sigma)
        kernel.eta_data_objective_value_grad(theta, eta, sigma)

        assert counts == counts_after_hessian
    finally:
        _clear_symbolic_caches()


@pytest.mark.parametrize(
    ("factory", "trans", "theta", "sigma", "eta_batch"),
    [
        (
            _make_symbolic_advan2_model,
            2,
            np.array([1.4, 10.5, 24.0], dtype=float),
            np.array([[0.05]], dtype=float),
            np.array([[0.02, -0.03, 0.01], [-0.04, 0.01, 0.03], [0.0, 0.0, 0.0]], dtype=float),
        ),
        (
            _make_symbolic_advan1_model,
            2,
            np.array([1.6, 11.5], dtype=float),
            np.array([[0.05]], dtype=float),
            np.array([[0.04, -0.03], [-0.02, 0.05], [0.0, 0.0]], dtype=float),
        ),
        (
            _make_symbolic_advan3_trans1_model,
            1,
            np.array([2.2, 18.0, 1.4, 30.0], dtype=float),
            np.array([[0.04]], dtype=float),
            np.array([[0.03, -0.04, 0.0, 0.0], [-0.02, 0.01, 0.0, 0.0]], dtype=float),
        ),
        (
            _make_symbolic_advan4_trans4_model,
            4,
            np.array([1.1, 3.8, 20.0, 2.5, 35.0], dtype=float),
            np.array([[0.03]], dtype=float),
            np.array([[0.02, -0.03, 0.01], [-0.01, 0.02, -0.02], [0.0, 0.0, 0.0]], dtype=float),
        ),
    ],
)
def test_symbolic_kernel_batch_objective_values_match_scalar_evaluations(
    factory,
    trans: int,
    theta: np.ndarray,
    sigma: np.ndarray,
    eta_batch: np.ndarray,
) -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = factory()
        kernel = model.get_subject_derivative_kernel(trans)
        assert kernel is not None

        expected = np.array(
            [kernel.eta_data_objective_value_grad(theta, eta, sigma)[0] for eta in eta_batch],
            dtype=float,
        )
        actual = np.asarray(kernel.eta_data_objective_values(theta, eta_batch, sigma), dtype=float)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan2_theta_gradient_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan2_model()
        kernel = model.get_subject_derivative_kernel(2)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([1.4, 10.5, 24.0], dtype=float)
        eta = np.array([0.02, -0.03, 0.01], dtype=float)
        sigma = np.array([[0.05]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.theta_data_objective_gradient(theta, eta, sigma), dtype=float)
        expected = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            vp = float(kernel.eta_data_objective_value_grad(theta_p, eta, sigma)[0])
            vm = float(kernel.eta_data_objective_value_grad(theta_m, eta, sigma)[0])
            expected[i] = (vp - vm) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan2_theta_jacobian_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan2_model()
        kernel = model.get_subject_derivative_kernel(2)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([1.4, 10.5, 24.0], dtype=float)
        eta = np.array([0.02, -0.03, 0.01], dtype=float)
        sigma = np.array([[0.05]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.prediction_theta_jacobian(theta, eta, sigma), dtype=float)
        expected = np.zeros_like(analytic)

        def pred_of_theta(theta_value: np.ndarray) -> np.ndarray:
            return np.asarray(model.evaluate(theta_value, eta, sigma, trans=2)[0], dtype=float)[
                model.subject_events.observation_mask()
            ]

        base = pred_of_theta(theta)
        assert base.shape[0] == analytic.shape[0]
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            expected[:, i] = (pred_of_theta(theta_p) - pred_of_theta(theta_m)) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_source_cache_reuses_generated_code(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sympy")
    monkeypatch.setenv("OPENPKPD_SYMBOLIC_CACHE_DIR", str(tmp_path))
    _clear_symbolic_caches()
    try:
        cache_path = sym_eta._symbolic_cache_file("advan1_terms")
        assert not cache_path.exists()

        compiled = sym_eta._compiled_advan1_terms()
        assert callable(compiled["contrib"])
        assert cache_path.exists()
        assert "def contrib(" in cache_path.read_text(encoding="utf-8")

        sym_eta._compiled_advan1_terms.cache_clear()

        def _fail_symbols(*args, **kwargs):
            raise AssertionError("expected generated symbolic source to be reused from disk cache")

        monkeypatch.setattr(sym_eta.sp, "symbols", _fail_symbols)
        loaded = sym_eta._compiled_advan1_terms()
        assert callable(loaded["contrib"])
    finally:
        _clear_symbolic_caches()


def test_prewarm_symbolic_caches_materializes_expected_cache_files(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sympy")
    monkeypatch.setenv("OPENPKPD_SYMBOLIC_CACHE_DIR", str(tmp_path))
    _clear_symbolic_caches()
    try:
        warmed = sym_eta.prewarm_symbolic_caches()
        expected_names = {
            "advan2_terms",
            "advan1_terms",
            "advan1_hessian_terms",
            "advan3_terms",
            "advan3_hessian_terms",
            "advan4_terms",
            "advan4_hessian_terms",
            "advan2_limit_terms",
            "advan2_hessian_terms",
            "advan2_limit_hessian_terms",
        }
        assert {entry["cache_name"] for entry in warmed} == expected_names
        assert all(bool(entry["exists"]) for entry in warmed)
        for entry in warmed:
            assert (
                tmp_path in sym_eta.Path(str(entry["cache_path"])).parents
                or sym_eta.Path(str(entry["cache_path"])).parent == tmp_path
            )
    finally:
        _clear_symbolic_caches()


def test_symbolic_kernel_can_load_from_prewarmed_cache_without_sympy(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sympy")
    monkeypatch.setenv("OPENPKPD_SYMBOLIC_CACHE_DIR", str(tmp_path))
    _clear_symbolic_caches()
    try:
        warmed = sym_eta.prewarm_symbolic_caches()
        assert all(bool(entry["exists"]) for entry in warmed)

        _clear_symbolic_caches()
        monkeypatch.setattr(sym_eta, "SYMPY_AVAILABLE", False)
        monkeypatch.setattr(sym_eta, "sp", None)
        monkeypatch.setattr(sym_eta, "NumPyPrinter", None)

        model = _make_symbolic_advan1_model()
        kernel = model.get_subject_derivative_kernel(2)
        assert kernel is not None

        theta = np.array([1.6, 11.5], dtype=float)
        eta = np.array([0.04, -0.03], dtype=float)
        sigma = np.array([[0.05]], dtype=float)
        value, grad = kernel.eta_data_objective_value_grad(theta, eta, sigma)

        assert np.isfinite(float(value))
        np.testing.assert_equal(np.asarray(grad).shape, eta.shape)
    finally:
        _clear_symbolic_caches()


def test_generate_symbolic_function_source_uses_cse() -> None:
    pytest.importorskip("sympy")
    x, y = sym_eta.sp.symbols("x y", real=True)
    source = sym_eta._generate_symbolic_function_source(
        {"f": ((x, y), (x + y) ** 2 + sym_eta.sp.sin(x + y))}
    )
    assert "_cse" in source


def test_generate_symbolic_function_source_supports_multi_return() -> None:
    pytest.importorskip("sympy")
    x, y = sym_eta.sp.symbols("x y", real=True)
    source = sym_eta._generate_symbolic_function_source(
        {"f": ((x, y), ((x + y) ** 2, sym_eta.sp.sin(x + y)))}
    )
    loaded = sym_eta._load_symbolic_functions_from_source(source, ("f",), "<generated>")
    values = loaded["f"](2.0, 3.0)
    assert isinstance(values, tuple)
    np.testing.assert_allclose(values[0], 25.0)
    np.testing.assert_allclose(values[1], np.sin(5.0))


def test_parse_advan3_trans1_pk_source_accepts_explicit_macro_plus_micro_lines() -> None:
    parsed = sym_eta._parse_advan3_trans1_pk_source(
        "CL = theta[0]*math.exp (eta[0])\n"
        "V1 = theta[1]*math.exp (eta[1])\n"
        "Q = theta[2]\n"
        "V2 = theta[3]\n"
        "K = CL/V1\n"
        "K12 = Q/V1\n"
        "K21 = Q/V2"
    )

    assert parsed == {"CL": (0, 0), "V1": (1, 1), "Q": (2, None), "V2": (3, None)}


def test_parse_pk_source_with_static_covariates_accepts_scm_power_linear_and_exp_lines() -> None:
    parsed = sym_eta._parse_pk_source_with_static_covariates(
        "KA = theta[0]*math.exp (eta[0])\n"
        "CL = theta[1]*math.exp (eta[1])\n"
        "V = theta[2]*math.exp (eta[2])\n"
        "CL = CL * (WT/70.0)**theta[3]\n"
        "V = V * (1 + theta[4] * (AGE - 40.0))\n"
        "KA = KA * math.exp (theta[5] * (WT - 70.0))",
        ("KA", "CL", "V"),
    )

    assert parsed is not None
    assert parsed["KA"][:2] == (0, 0)
    assert parsed["CL"][:2] == (1, 1)
    assert parsed["V"][:2] == (2, 2)
    assert [adj.kind for adj in parsed["KA"][2]] == ["exponential"]
    assert [adj.kind for adj in parsed["CL"][2]] == ["power"]
    assert [adj.kind for adj in parsed["V"][2]] == ["linear"]


def test_advan3_trans1_symbolic_builder_accepts_explicit_macro_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_symbolic_advan3_trans1_model()
    monkeypatch.setattr(sym_eta, "SYMPY_AVAILABLE", True)

    kernel = sym_eta.SympyAdvan3Trans4Objective.build(model, trans=1)

    assert kernel is not None
    assert kernel.theta_idx == (0, 1, 2, 3)
    assert kernel.eta_idx == (0, 1, None, None)


def test_advan4_trans4_symbolic_builder_accepts_macro_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_symbolic_advan4_trans4_model()
    monkeypatch.setattr(sym_eta, "SYMPY_AVAILABLE", True)

    kernel = sym_eta.SympyAdvan4Trans1Objective.build(model, trans=4)

    assert kernel is not None
    assert kernel.theta_idx == (0, 1, 2, 3, 4)
    assert kernel.eta_idx == (0, 1, 2)


def test_common_symbolic_build_guards_allow_unused_covariate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_events = SubjectEvents(
        subject_id=1,
        dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
        obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
        obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
        obs_cmt=np.ones(5, dtype=int),
        obs_mdv=np.zeros(5, dtype=int),
        covariate_df=pd.DataFrame({"TIME": [0.0, 1.0, 4.0], "WT": [70.0, 70.0, 70.0]}),
    )
    model = _make_symbolic_advan2_model(subject_events=subject_events)
    monkeypatch.setattr(sym_eta, "SYMPY_AVAILABLE", True)

    assert sym_eta._common_symbolic_build_guards(model)
    assert sym_eta.SympyAdvan2Trans2Objective.build(model, trans=2) is not None


def test_static_covariate_values_reject_time_varying_covariates() -> None:
    subject_events = SubjectEvents(
        subject_id=1,
        dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
        obs_times=np.array([0.5, 1.0, 2.0], dtype=float),
        obs_dv=np.array([7.0, 6.1, 4.8], dtype=float),
        obs_cmt=np.ones(3, dtype=int),
        obs_mdv=np.zeros(3, dtype=int),
        covariate_df=pd.DataFrame({"TIME": [0.0, 1.0], "WT": [70.0, 80.0]}),
    )
    model = _make_symbolic_advan2_model(subject_events=subject_events)

    assert sym_eta._static_covariate_values(model) is None


def test_advan1_symbolic_builder_accepts_static_power_covariate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_symbolic_advan1_covariate_model()
    monkeypatch.setattr(sym_eta, "SYMPY_AVAILABLE", True)

    kernel = sym_eta.SympyAdvan1Trans2Objective.build(model, trans=2)

    assert kernel is not None
    assert kernel.theta_idx == (0, 1)
    assert kernel.eta_idx == (0, 1)
    assert kernel.static_covariates["WT"] == pytest.approx(70.0)


def test_common_symbolic_build_guards_reject_referenced_covariate_names() -> None:
    class _StubCallable:
        def __init__(self, source: str) -> None:
            self._source = source

    class _StubSubjectEvents:
        covariate_df = pd.DataFrame({"TIME": [0.0], "WT": [70.0]})

    class _StubIndiv:
        pk_callable = _StubCallable("CL = theta[0] * WT\nV = theta[1]")
        error_callable = _StubCallable("Y = F + EPS(1)")
        occasion_indices = None
        blq_method = "M1"
        lloq = None
        des_callable = None
        _error_requires_amounts = False
        _base_covariates = {"WT": 70.0}
        _observation_covariates = ({"WT": 70.0},)
        subject_events = _StubSubjectEvents()

    assert not sym_eta._common_symbolic_build_guards(_StubIndiv())


def _make_symbolic_advan3_trans4_model() -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=250.0, compartment=1)],
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([7.0, 6.1, 4.8, 2.9, 1.3], dtype=float),
            obs_cmt=np.ones(5, dtype=int),
            obs_mdv=np.zeros(5, dtype=int),
        ),
        pk_subroutine=ADVAN3(),
        pk_callable=compiler.compile_pk(
            "CL = THETA(1)*EXP(ETA(1))\n"
            "V1 = THETA(2)*EXP(ETA(2))\n"
            "Q = THETA(3)*EXP(ETA(3))\n"
            "V2 = THETA(4)*EXP(ETA(4))"
        ),
        error_callable=compiler.compile_error("Y = F*(1 + EPS(1))"),
        n_eps=1,
    )


def test_symbolic_advan3_theta_gradient_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan3_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([1.2, 12.0, 0.5, 25.0], dtype=float)
        eta = np.array([0.05, -0.02, 0.01, -0.01], dtype=float)
        sigma = np.array([[0.04]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.theta_data_objective_gradient(theta, eta, sigma), dtype=float)
        expected = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            vp = float(kernel.eta_data_objective_value_grad(theta_p, eta, sigma)[0])
            vm = float(kernel.eta_data_objective_value_grad(theta_m, eta, sigma)[0])
            expected[i] = (vp - vm) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan3_theta_jacobian_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan3_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([1.2, 12.0, 0.5, 25.0], dtype=float)
        eta = np.array([0.05, -0.02, 0.01, -0.01], dtype=float)
        sigma = np.array([[0.04]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.prediction_theta_jacobian(theta, eta, sigma), dtype=float)
        obs_mask = model.subject_events.observation_mask()

        def pred_of_theta(th: np.ndarray) -> np.ndarray:
            return np.asarray(model.evaluate(th, eta, sigma, trans=4)[0], dtype=float)[obs_mask]

        expected = np.zeros_like(analytic)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            expected[:, i] = (pred_of_theta(theta_p) - pred_of_theta(theta_m)) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan4_theta_gradient_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan4_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([0.8, 1.5, 18.0, 0.6, 30.0], dtype=float)
        eta = np.array([0.03, -0.04, 0.02], dtype=float)
        sigma = np.array([[0.04]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.theta_data_objective_gradient(theta, eta, sigma), dtype=float)
        expected = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            vp = float(kernel.eta_data_objective_value_grad(theta_p, eta, sigma)[0])
            vm = float(kernel.eta_data_objective_value_grad(theta_m, eta, sigma)[0])
            expected[i] = (vp - vm) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan4_theta_jacobian_matches_finite_difference() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan4_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.supports_theta_data_objective_gradient()

        theta = np.array([0.8, 1.5, 18.0, 0.6, 30.0], dtype=float)
        eta = np.array([0.03, -0.04, 0.02], dtype=float)
        sigma = np.array([[0.04]], dtype=float)
        eps = 1e-6

        analytic = np.asarray(kernel.prediction_theta_jacobian(theta, eta, sigma), dtype=float)
        obs_mask = model.subject_events.observation_mask()

        def pred_of_theta(th: np.ndarray) -> np.ndarray:
            return np.asarray(model.evaluate(th, eta, sigma, trans=4)[0], dtype=float)[obs_mask]

        expected = np.zeros_like(analytic)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            expected[:, i] = (pred_of_theta(theta_p) - pred_of_theta(theta_m)) / (2.0 * eps)

        np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-5)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan3_capabilities_include_theta_gradient() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan3_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.capabilities.theta_data_objective_gradient
        assert kernel.capabilities.prediction_theta_jacobian
        assert model.supports_theta_data_objective_gradient(trans=4)
    finally:
        _clear_symbolic_caches()


def test_symbolic_advan4_capabilities_include_theta_gradient() -> None:
    pytest.importorskip("sympy")
    _clear_symbolic_caches()
    try:
        model = _make_symbolic_advan4_trans4_model()
        kernel = model.get_subject_derivative_kernel(4)
        assert kernel is not None
        assert kernel.capabilities.theta_data_objective_gradient
        assert kernel.capabilities.prediction_theta_jacobian
        assert model.supports_theta_data_objective_gradient(trans=4)
    finally:
        _clear_symbolic_caches()
