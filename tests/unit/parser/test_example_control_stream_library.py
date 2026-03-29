"""Validation for curated example control-stream files."""

from __future__ import annotations

from pathlib import Path

from openpkpd.parser.control_stream import ControlStream

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "control_streams"
EXPECTED_EXAMPLE_FILES = {
    "00_theophylline_fixture_diagnostics.ctl",
    "01_theophylline_oral_fo.ctl",
    "02_warfarin_oral_focei.ctl",
    "03_two_compartment_iv_fo.ctl",
    "06_minimal_theophylline_focei.ctl",
    "08_transit_absorption_advan6_fo.ctl",
    "09_three_compartment_iv_advan11_focei.ctl",
    "20_theophylline_oral_saem.ctl",
    "23_same_omega_showcase.ctl",
    "30_nmdata_input_drop_and_body_weight_scaling.ctl",
    "31_nmdata_multiple_table_output_formats.ctl",
    "32_nmdata_saem_age_covariate.ctl",
    "33_nmdata_saem_age_covariate_block_omega.ctl",
    "34_nmsim_two_compartment_block_omega_fo.ctl",
    "35_nmsim_advan13_one_compartment_des.ctl",
    "36_nmsim_saem_multi_covariate_age_weight_sex.ctl",
    "37_focei_optimizer_controls.ctl",
    "38_prior_gaussian_subset.ctl",
    "39_onlysimulation_subproblems.ctl",
    "10_warfarin_pk_focei.ctl",
    "11_two_compartment_iv_focei.ctl",
    "12_phenobarbital_fo.ctl",
    "13_covariates_one_cmt_focei.ctl",
}

# Files that intentionally have no $ESTIMATION block (e.g. simulation-only).
_SIMULATION_ONLY_FILES = {
    "39_onlysimulation_subproblems.ctl",
}


def test_curated_control_stream_library_contains_expected_files() -> None:
    actual_files = {path.name for path in EXAMPLES_DIR.glob("*.ctl")}

    assert actual_files == EXPECTED_EXAMPLE_FILES


def test_curated_control_stream_library_parses_cleanly() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.ctl")):
        control_stream = ControlStream.from_file(path)

        assert control_stream.problem is not None, path.name
        assert control_stream.data is not None, path.name
        assert control_stream.subroutines is not None, path.name
        assert control_stream.pk is not None, path.name
        assert control_stream.error is not None, path.name
        assert control_stream.theta_records, path.name
        assert control_stream.omega_records, path.name
        assert control_stream.sigma_records, path.name
        if path.name not in _SIMULATION_ONLY_FILES:
            assert control_stream.estimation_records, path.name


def test_transit_example_includes_des_block() -> None:
    control_stream = ControlStream.from_file(EXAMPLES_DIR / "08_transit_absorption_advan6_fo.ctl")

    assert control_stream.des is not None

    advan13_control_stream = ControlStream.from_file(
        EXAMPLES_DIR / "35_nmsim_advan13_one_compartment_des.ctl"
    )

    assert advan13_control_stream.des is not None


def test_fixture_and_minimal_examples_preserve_reporting_records() -> None:
    fixture_control_stream = ControlStream.from_file(
        EXAMPLES_DIR / "00_theophylline_fixture_diagnostics.ctl"
    )
    minimal_control_stream = ControlStream.from_file(
        EXAMPLES_DIR / "06_minimal_theophylline_focei.ctl"
    )

    assert fixture_control_stream.covariance is not None
    assert fixture_control_stream.table_records
    assert minimal_control_stream.covariance is not None
    assert minimal_control_stream.table_records


def test_imported_examples_preserve_table_and_block_omega_features() -> None:
    table_control_stream = ControlStream.from_file(
        EXAMPLES_DIR / "31_nmdata_multiple_table_output_formats.ctl"
    )
    block_omega_control_stream = ControlStream.from_file(
        EXAMPLES_DIR / "33_nmdata_saem_age_covariate_block_omega.ctl"
    )

    assert len(table_control_stream.table_records) >= 3
    assert len(block_omega_control_stream.omega_records) >= 4
