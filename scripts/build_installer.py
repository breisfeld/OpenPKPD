#!/usr/bin/env python3
"""
Build platform-native installers for OpenPKPD.

Produces:
  Windows  →  dist/windows/OpenPKPD-<version>-windows-x64.zip
  macOS    →  dist/macos/OpenPKPD-<version>-macos.dmg
  Linux    →  dist/linux/OpenPKPD-<version>-linux-x86_64.tar.gz
              (+ AppImage if appimagetool is on PATH)

Usage:
  python scripts/build_installer.py
  python scripts/build_installer.py --version 1.2.3
  python scripts/build_installer.py --output-dir /tmp/release
  python scripts/build_installer.py --skip-gui      # CLI only, no Qt required
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "scripts" / "installer.spec"
DIST_ROOT = ROOT / "dist"
BUILD_DIR = ROOT / "build" / "pyinstaller"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=True, **kwargs)
    return result


def _read_version() -> str:
    """Extract version from pyproject.toml without importing the package."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not determine version from pyproject.toml")
    return m.group(1)


def _detect_platform() -> str:
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    if s == "windows":
        return "windows"
    return "linux"


def _arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


# ---------------------------------------------------------------------------
# PyInstaller
# ---------------------------------------------------------------------------

def _ensure_icns() -> None:
    """On macOS, regenerate icon.icns from icon.png if it is missing or stale."""
    import platform as _plat
    if _plat.system().lower() != "darwin":
        return
    icns = ROOT / "src" / "openpkpd_gui" / "resources" / "icon.icns"
    png  = ROOT / "src" / "openpkpd_gui" / "resources" / "icon.png"
    if icns.exists() and png.exists() and icns.stat().st_mtime >= png.stat().st_mtime:
        return
    make_script = ROOT / "scripts" / "packaging" / "macos" / "make_icns.sh"
    if not make_script.exists():
        print("  Warning: make_icns.sh not found; macOS app bundle will have no icon")
        return
    if not shutil.which("sips") or not shutil.which("iconutil"):
        print("  Warning: sips/iconutil not available; skipping .icns generation")
        return
    print("  Generating icon.icns from icon.png …")
    _run(["bash", str(make_script)])


def run_pyinstaller(version: str, skip_gui: bool) -> Path:
    """Run PyInstaller and return the path to the collected dist directory."""
    if not skip_gui:
        _ensure_icns()
    out_dir = DIST_ROOT / "pyinstaller"
    out_dir.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC_FILE),
        "--distpath", str(out_dir),
        "--workpath", str(BUILD_DIR),
        "--noconfirm",
        "--clean",
        f"--log-level=WARN",
    ]

    if skip_gui:
        cmd += ["--", "--skip-gui"]  # forwarded to spec via sys.argv inspection

    env_extras = {
        "OPENPKPD_VERSION": version,
        "OPENPKPD_SKIP_GUI": "1" if skip_gui else "0",
    }
    import os
    env = {**os.environ, **env_extras}

    _run(cmd, env=env)

    collected = out_dir / "OpenPKPD"
    if not collected.exists():
        raise RuntimeError(f"PyInstaller output not found at {collected}")
    return collected


# ---------------------------------------------------------------------------
# Platform packaging
# ---------------------------------------------------------------------------

def package_windows(collected: Path, version: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"OpenPKPD-{version}-windows-{_arch()}.zip"
    archive_path = output_dir / archive_name

    print(f"  Zipping {collected} → {archive_path}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(collected.rglob("*")):
            if f.is_file():
                zf.write(f, Path("OpenPKPD") / f.relative_to(collected))

    # Optionally wrap with NSIS if makensis is available
    nsi_script = ROOT / "scripts" / "packaging" / "windows" / "installer.nsi"
    if shutil.which("makensis") and nsi_script.exists():
        print("  makensis found — building NSIS installer")
        _run([
            "makensis",
            f"/DVERSION={version}",
            f"/DDIST_DIR={collected}",
            f"/DOUTPUT_DIR={output_dir}",
            str(nsi_script),
        ])

    return archive_path


def package_macos(collected: Path, version: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyInstaller BUNDLE outputs the .app alongside the COLLECT directory,
    # not inside it.  collected = dist/pyinstaller/OpenPKPD, so the app
    # bundle lives at dist/pyinstaller/OpenPKPD-gui.app.
    app_bundle = collected.parent / "OpenPKPD-gui.app"
    if not app_bundle.exists():
        # Fall back to the collected directory itself (CLI-only or unusual layout).
        print("  Warning: .app bundle not found; DMG will contain executable folder")
        app_bundle = collected

    dmg_name = f"OpenPKPD-{version}-macos-{_arch()}.dmg"
    dmg_path = output_dir / dmg_name

    settings_file = ROOT / "scripts" / "packaging" / "macos" / "dmgbuild_settings.py"
    if settings_file.exists():
        print(f"  Building DMG via dmgbuild → {dmg_path}")
        import os
        env = {
            **os.environ,
            "OPENPKPD_APP_PATH": str(app_bundle),
            "OPENPKPD_VERSION": version,
        }
        _run([
            sys.executable, "-m", "dmgbuild",
            "-s", str(settings_file),
            f"OpenPKPD {version}",
            str(dmg_path),
        ], env=env)
    else:
        # Fallback: bare hdiutil
        print(f"  Building DMG via hdiutil → {dmg_path}")
        staging = BUILD_DIR / "dmg_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copytree(app_bundle, staging / app_bundle.name, symlinks=True)
        _run([
            "hdiutil", "create",
            "-volname", f"OpenPKPD {version}",
            "-srcfolder", str(staging),
            "-ov", "-format", "UDZO",
            str(dmg_path),
        ])

    return dmg_path


def package_linux(collected: Path, version: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tarball_name = f"OpenPKPD-{version}-linux-{_arch()}.tar.gz"
    tarball_path = output_dir / tarball_name

    print(f"  Creating tarball → {tarball_path}")
    with tarfile.open(tarball_path, "w:gz") as tf:
        tf.add(collected, arcname="OpenPKPD")

    # Optionally build AppImage if appimagetool is available
    appimage_dir = ROOT / "scripts" / "packaging" / "linux"
    if shutil.which("appimagetool") and (appimage_dir / "AppDir").exists():
        print("  appimagetool found — building AppImage")
        appimage_name = f"OpenPKPD-{version}-{_arch()}.AppImage"
        appdir = BUILD_DIR / "AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        shutil.copytree(appimage_dir / "AppDir", appdir)
        # Copy frozen app into AppDir/usr/bin
        usr_bin = appdir / "usr" / "bin"
        usr_bin.mkdir(parents=True, exist_ok=True)
        for item in collected.iterdir():
            dest = usr_bin / item.name
            if item.is_dir():
                shutil.copytree(item, dest, symlinks=True)
            else:
                shutil.copy2(item, dest)
        # AppImage requires the icon at AppDir root (no extension)
        icon_src = ROOT / "src" / "openpkpd_gui" / "resources" / "icon.png"
        if icon_src.exists():
            shutil.copy2(icon_src, appdir / "openpkpd-gui.png")
        import os
        env = {**os.environ, "ARCH": _arch()}
        _run(
            ["appimagetool", str(appdir), str(output_dir / appimage_name)],
            env=env,
        )

    return tarball_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenPKPD installer")
    parser.add_argument("--version", default=None, help="Version string (default: from pyproject.toml)")
    parser.add_argument("--output-dir", default=None, help="Directory for final artifacts (default: dist/<platform>)")
    parser.add_argument("--skip-gui", action="store_true", help="Build CLI-only (no PySide6 required)")
    parser.add_argument("--platform", default="auto", choices=["auto", "windows", "macos", "linux"],
                        help="Target platform (default: auto-detect)")
    args = parser.parse_args()

    version = args.version or _read_version()
    target_platform = _detect_platform() if args.platform == "auto" else args.platform
    output_dir = Path(args.output_dir) if args.output_dir else DIST_ROOT / target_platform

    print(f"OpenPKPD installer build")
    print(f"  version   : {version}")
    print(f"  platform  : {target_platform}")
    print(f"  arch      : {_arch()}")
    print(f"  output    : {output_dir}")
    print(f"  skip-gui  : {args.skip_gui}")
    print()

    # 1. Freeze
    print("=== Step 1: PyInstaller freeze ===")
    collected = run_pyinstaller(version, args.skip_gui)
    print(f"  Frozen app: {collected}\n")

    # 2. Package
    print(f"=== Step 2: Package for {target_platform} ===")
    if target_platform == "windows":
        artifact = package_windows(collected, version, output_dir)
    elif target_platform == "macos":
        artifact = package_macos(collected, version, output_dir)
    else:
        artifact = package_linux(collected, version, output_dir)

    print()
    print(f"=== Done ===")
    print(f"  Artifact: {artifact}")
    print(f"  Size    : {artifact.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
