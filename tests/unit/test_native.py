from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openpkpd import _native


def test_candidate_sundials_lib_dirs_prefers_env_and_existing_build_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "env_lib"
    env_dir.mkdir()

    project_root = tmp_path / "proj"
    build_lib = project_root / "rust" / "target" / "release" / "build" / "sundials-sys-abc" / "out" / "lib"
    build_lib.mkdir(parents=True)

    monkeypatch.setenv("OPENPKPD_SUNDIALS_LIBDIRS", str(env_dir))
    monkeypatch.setenv("OPENPKPD_NATIVE_DEV", "1")
    monkeypatch.setattr(_native, "__file__", str(project_root / "src" / "openpkpd" / "_native.py"))

    dirs = _native._candidate_sundials_lib_dirs()

    assert env_dir in dirs
    assert build_lib in dirs


def test_candidate_sundials_lib_dirs_includes_package_adjacent_lib_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "proj"
    package_dir = project_root / "src" / "openpkpd"
    package_dir.mkdir(parents=True)
    ext_dir = tmp_path / "site-packages" / "openpkpd"
    ext_dir.mkdir(parents=True)

    package_lib = package_dir / ".libs"
    ext_lib = ext_dir.parent / "openpkpd.libs"
    package_lib.mkdir()
    ext_lib.mkdir()

    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    monkeypatch.setattr(
        _native.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(ext_dir / "_core.cpython-312-x86_64-linux-gnu.so")),
    )

    dirs = _native._candidate_sundials_lib_dirs()

    assert package_lib in dirs
    assert ext_lib in dirs


def test_candidate_sundials_lib_dirs_defaults_to_packaged_locations_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "proj"
    package_dir = project_root / "src" / "openpkpd"
    package_dir.mkdir(parents=True)
    build_lib = project_root / "rust" / "target" / "release" / "build" / "sundials-sys-abc" / "out" / "lib"
    build_lib.mkdir(parents=True)

    monkeypatch.delenv("OPENPKPD_NATIVE_DEV", raising=False)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    monkeypatch.setattr(
        _native.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=None),
    )

    dirs = _native._candidate_sundials_lib_dirs()

    assert build_lib not in dirs
