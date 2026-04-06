from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd.r_bridge.rscript import (
    build_rscript_env,
    detect_r_home,
    find_rscript,
    run_rscript,
)


class TestBuildRScriptEnv:
    def test_strips_conda_vars_and_sets_r_libs(self, monkeypatch):
        monkeypatch.setattr("openpkpd.r_bridge.rscript.detect_r_home", lambda: "/usr/lib/R")

        env = build_rscript_env(
            base_env={
                "CONDA_PREFIX": "/tmp/conda",
                "CONDA_SHLVL": "1",
                "R_LIBS_USER": "/existing/lib",
            },
            extra_r_libs=[Path("/repo/.r-lib"), Path("/other/lib")],
        )

        assert "CONDA_PREFIX" not in env
        assert "CONDA_SHLVL" not in env
        assert env["R_HOME"] == "/usr/lib/R"
        assert env["R_LIBS_USER"] == "/repo/.r-lib:/other/lib:/existing/lib"


class TestRunRScript:
    @pytest.mark.skipif(find_rscript() is None, reason="Rscript not available")
    def test_run_rscript_sees_extra_r_libs(self, tmp_path):
        result = run_rscript(
            ["-e", 'cat(startsWith(Sys.getenv("R_LIBS_USER"), "/repo/.r-lib"))'],
            cwd=tmp_path,
            extra_r_libs=[Path("/repo/.r-lib")],
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "TRUE"

    @pytest.mark.skipif(find_rscript() is None, reason="Rscript not available")
    def test_detect_r_home_reports_existing_directory(self):
        r_home = detect_r_home()
        assert r_home is not None
        assert Path(r_home).exists()

