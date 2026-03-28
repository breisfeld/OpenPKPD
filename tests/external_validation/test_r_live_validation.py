"""Optional live R-backed external validations.

These tests complement the bundled frozen-reference JSON checks by running
selected external-tool workflows directly when the corresponding R packages
are available locally.

They are designed to skip cleanly in environments without the required R
packages, while automatically increasing scientific validation depth where
those tools are installed.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openpkpd.r_bridge import is_r_available

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PKNCA_REF = HERE / "reference" / "pknca_theophylline_summary.json"
NLMIXR2_DIR = HERE / "nlmixr2"
LOCAL_R_LIB = ROOT / ".r-lib"


def _r_package_available(name: str) -> bool:
    if not is_r_available():
        return False
    result = subprocess.run(
        [
            "Rscript",
            "-e",
            f"cat(requireNamespace('{name}', quietly=TRUE))",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.returncode == 0 and result.stdout.strip() == "TRUE"


def _run_rscript_inline(script: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("R_LIBS_USER", "")
    libs = [str(LOCAL_R_LIB)]
    if existing:
        libs.append(existing)
    env["R_LIBS_USER"] = os.pathsep.join(libs)
    return subprocess.run(
        ["Rscript", "-e", script],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def _run_nlmixr2_reference_script(
    script_name: str,
    *,
    tmp_path: Path,
) -> tuple[dict, dict]:
    temp_dir = tmp_path / "nlmixr2"
    shutil.copytree(NLMIXR2_DIR, temp_dir)
    shutil.copytree(HERE / "data", tmp_path / "data")
    script_path = temp_dir / script_name

    env = os.environ.copy()
    existing = env.get("R_LIBS_USER", "")
    libs = [str(LOCAL_R_LIB)]
    if existing:
        libs.append(existing)
    env["R_LIBS_USER"] = os.pathsep.join(libs)

    result = subprocess.run(
        ["Rscript", str(script_path)],
        cwd=temp_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    return (
        json.loads((temp_dir / "reference" / script_name_to_fo_ref(script_name)).read_text()),
        json.loads((temp_dir / "reference" / script_name_to_focei_ref(script_name)).read_text()),
    )


def script_name_to_fo_ref(script_name: str) -> str:
    if script_name == "run_theophylline.R":
        return "theophylline_fo.json"
    if script_name == "run_warfarin.R":
        return "warfarin_pk_fo.json"
    raise ValueError(f"Unsupported nlmixr2 script: {script_name}")


def script_name_to_focei_ref(script_name: str) -> str:
    if script_name == "run_theophylline.R":
        return "theophylline_foce.json"
    if script_name == "run_warfarin.R":
        return "warfarin_pk_foce.json"
    raise ValueError(f"Unsupported nlmixr2 script: {script_name}")


@pytest.mark.external_validation
class TestLivePKNCAReference:
    @pytest.mark.skipif(not is_r_available(), reason="R / rpy2 not available")
    @pytest.mark.skipif(not _r_package_available("PKNCA"), reason="R package PKNCA not installed")
    def test_theophylline_summary_matches_frozen_pknca_reference(self) -> None:
        reference = json.loads(PKNCA_REF.read_text())
        data_path = HERE / "data" / "theophylline_boeckmann.csv"

        r_script = f"""
        suppressPackageStartupMessages(library(PKNCA))
        d <- read.csv("{data_path.as_posix()}")
        dose <- subset(d, EVID == 1, select=c(ID, TIME, AMT))
        zero_obs <- unique(dose[, c("ID", "TIME")])
        zero_obs$DV <- 0
        zero_obs$EVID <- 0
        zero_obs$MDV <- 0
        obs <- subset(d, EVID == 0 & MDV == 0, select=c(ID, TIME, DV, EVID, MDV))
        obs <- rbind(zero_obs[, c("ID", "TIME", "DV", "EVID", "MDV")], obs)
        obs <- obs[order(obs$ID, obs$TIME), ]

        conc_obj <- PKNCAconc(obs, DV ~ TIME | ID)
        dose_obj <- PKNCAdose(dose, AMT ~ TIME | ID)
        intervals <- data.frame(
          start=c(0, 0),
          end=c(24, Inf),
          auclast=c(TRUE, TRUE),
          cmax=c(FALSE, TRUE),
          tmax=c(FALSE, TRUE),
          half.life=c(FALSE, TRUE),
          aucinf.obs=c(FALSE, TRUE)
        )
        results <- as.data.frame(pk.nca(PKNCAdata(conc_obj, dose_obj, intervals=intervals)))

        get_vals <- function(code, end_value) {{
          subset(results, PPTESTCD == code & end == end_value)$PPORRES
        }}
        geomean <- function(x) exp(mean(log(x)))
        geocv <- function(x) sqrt(exp(stats::var(log(x))) - 1) * 100

        auclast <- get_vals("auclast", 24)
        cmax <- get_vals("cmax", Inf)
        tmax <- get_vals("tmax", Inf)
        half_life <- get_vals("half.life", Inf)
        aucinf <- get_vals("aucinf.obs", Inf)
        if (all(is.na(aucinf))) {{
          clast_obs <- get_vals("clast.obs", Inf)
          lambda_z <- get_vals("lambda.z", Inf)
          aucinf <- auclast + clast_obs / lambda_z
        }}

        cat(sprintf("AUCLAST_CENTER=%.10f\\n", geomean(auclast)))
        cat(sprintf("AUCLAST_CV=%.10f\\n", geocv(auclast)))
        cat(sprintf("CMAX_CENTER=%.10f\\n", geomean(cmax)))
        cat(sprintf("CMAX_CV=%.10f\\n", geocv(cmax)))
        cat(sprintf("TMAX_MEDIAN=%.10f\\n", stats::median(tmax)))
        cat(sprintf("TMAX_MIN=%.10f\\n", min(tmax)))
        cat(sprintf("TMAX_MAX=%.10f\\n", max(tmax)))
        cat(sprintf("HALFLIFE_MEAN=%.10f\\n", mean(half_life)))
        cat(sprintf("HALFLIFE_SD=%.10f\\n", stats::sd(half_life)))
        cat(sprintf("AUCINF_CENTER=%.10f\\n", geomean(aucinf)))
        cat(sprintf("AUCINF_CV=%.10f\\n", geocv(aucinf)))
        """

        result = _run_rscript_inline(r_script, cwd=ROOT)
        assert result.returncode == 0, result.stderr[-4000:]

        observed: dict[str, float] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                observed[key.strip()] = float(value.strip())

        ref_024 = reference["summary_0_24"]["auclast"]
        ref_inf = reference["summary_0_inf"]
        assert observed["AUCLAST_CENTER"] == pytest.approx(ref_024["center"], rel=0.01)
        assert observed["AUCLAST_CV"] == pytest.approx(ref_024["dispersion"], abs=0.5)
        assert observed["CMAX_CENTER"] == pytest.approx(ref_inf["cmax"]["center"], rel=0.01)
        assert observed["CMAX_CV"] == pytest.approx(ref_inf["cmax"]["dispersion"], abs=0.5)
        assert observed["TMAX_MEDIAN"] == pytest.approx(ref_inf["tmax"]["center"], abs=0.02)
        assert observed["TMAX_MIN"] == pytest.approx(ref_inf["tmax"]["lower"], abs=0.02)
        assert observed["TMAX_MAX"] == pytest.approx(ref_inf["tmax"]["upper"], abs=0.02)
        assert observed["HALFLIFE_MEAN"] == pytest.approx(ref_inf["half_life"]["center"], abs=0.05)
        assert observed["HALFLIFE_SD"] == pytest.approx(ref_inf["half_life"]["dispersion"], abs=0.05)
        assert observed["AUCINF_CENTER"] == pytest.approx(ref_inf["aucinf_obs"]["center"], rel=0.01)
        assert observed["AUCINF_CV"] == pytest.approx(ref_inf["aucinf_obs"]["dispersion"], abs=0.5)


@pytest.mark.external_validation
class TestLiveNlmixr2ReferenceGeneration:
    @pytest.mark.skipif(not is_r_available(), reason="R / rpy2 not available")
    @pytest.mark.skipif(not _r_package_available("nlmixr2"), reason="R package nlmixr2 not installed")
    @pytest.mark.skipif(not _r_package_available("nlmixr2data"), reason="R package nlmixr2data not installed")
    @pytest.mark.slow
    def test_theophylline_reference_script_reproduces_expected_json_shape(self, tmp_path: Path) -> None:
        generated_fo, generated_foce = _run_nlmixr2_reference_script(
            "run_theophylline.R", tmp_path=tmp_path
        )
        frozen_fo = json.loads((NLMIXR2_DIR / "reference" / "theophylline_fo.json").read_text())
        frozen_foce = json.loads((NLMIXR2_DIR / "reference" / "theophylline_foce.json").read_text())

        for generated, frozen in ((generated_fo, frozen_fo), (generated_foce, frozen_foce)):
            assert generated["method"] == frozen["method"]
            assert generated["software"] == "nlmixr2"
            assert generated["dataset"] == frozen["dataset"]
            assert generated["n_subjects"] == frozen["n_subjects"]
            assert generated["n_obs_in_likelihood"] == frozen["n_obs_in_likelihood"]
            for name in frozen["theta"]:
                observed = float(generated["theta"][name])
                reference = float(frozen["theta"][name])
                assert math.isfinite(observed)
                assert observed == pytest.approx(reference, rel=0.05)
            for name in frozen["omega_diag"]:
                observed = float(generated["omega_diag"][name])
                reference = float(frozen["omega_diag"][name])
                assert math.isfinite(observed)
                assert observed == pytest.approx(reference, rel=0.35)
            assert float(generated["sigma_prop_err_variance"]) == pytest.approx(
                float(frozen["sigma_prop_err_variance"]), rel=0.10
            )

    @pytest.mark.skipif(not is_r_available(), reason="R / rpy2 not available")
    @pytest.mark.skipif(not _r_package_available("nlmixr2"), reason="R package nlmixr2 not installed")
    @pytest.mark.skipif(not _r_package_available("nlmixr2data"), reason="R package nlmixr2data not installed")
    @pytest.mark.slow
    def test_warfarin_reference_script_reproduces_expected_json_shape(self, tmp_path: Path) -> None:
        generated_fo, generated_foce = _run_nlmixr2_reference_script(
            "run_warfarin.R", tmp_path=tmp_path
        )
        frozen_fo = json.loads((NLMIXR2_DIR / "reference" / "warfarin_pk_fo.json").read_text())
        frozen_foce = json.loads((NLMIXR2_DIR / "reference" / "warfarin_pk_foce.json").read_text())

        for generated, frozen in ((generated_fo, frozen_fo), (generated_foce, frozen_foce)):
            assert generated["method"] == frozen["method"]
            assert generated["software"] == "nlmixr2"
            assert generated["dataset"] == frozen["dataset"]
            assert generated["n_subjects"] == frozen["n_subjects"]
            assert generated["n_obs_in_likelihood"] == frozen["n_obs_in_likelihood"]
            for name in frozen["theta"]:
                observed = float(generated["theta"][name])
                reference = float(frozen["theta"][name])
                assert math.isfinite(observed)
                assert observed == pytest.approx(reference, rel=0.03)
            for name in frozen["omega_diag"]:
                observed = float(generated["omega_diag"][name])
                reference = float(frozen["omega_diag"][name])
                assert math.isfinite(observed)
                assert observed == pytest.approx(reference, rel=0.35)
            assert float(generated["sigma_prop_err_variance"]) == pytest.approx(
                float(frozen["sigma_prop_err_variance"]), rel=0.10
            )
            assert float(generated["ofv"]) == pytest.approx(float(frozen["ofv"]), rel=0.02)
