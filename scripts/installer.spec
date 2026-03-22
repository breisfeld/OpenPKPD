# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for OpenPKPD — CLI + GUI bundled together.
#
# Produces a single collected directory:
#
#   OpenPKPD/
#     openpkpd            ← CLI executable (console)
#     openpkpd-gui        ← GUI executable (windowed, macOS: .app)
#     _internal/          ← shared libraries, Qt plugins, Python stdlib
#
# Build via:
#   python scripts/build_installer.py
# Or directly:
#   pyinstaller scripts/installer.spec

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent          # project root
SRC  = ROOT / "src"

# ---------------------------------------------------------------------------
# Collected packages
# ---------------------------------------------------------------------------

# Core scientific stack — pull in all data + hidden imports
_all_datas    = []
_all_binaries = []
_all_hidden   = []

for pkg in (
    "openpkpd",
    "openpkpd_gui",
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "click",
    "matplotlib",
    "platformdirs",
):
    d, b, h = collect_all(pkg)
    _all_datas    += d
    _all_binaries += b
    _all_hidden   += h

# PySide6 — only when not building CLI-only
_skip_gui = os.environ.get("OPENPKPD_SKIP_GUI", "0") == "1"
if not _skip_gui:
    d, b, h = collect_all("PySide6")
    _all_datas    += d
    _all_binaries += b
    _all_hidden   += h

# Example files shipped with the package
_all_datas += [
    (str(ROOT / "examples"), "examples"),
]

# ---------------------------------------------------------------------------
# Explicit hidden imports
# ---------------------------------------------------------------------------

_all_hidden += [
    # Scipy submodules that are dynamically loaded
    "scipy.special._ufuncs",
    "scipy.special._comb",
    "scipy.linalg._decomp",
    "scipy.optimize._minimize",
    "scipy.optimize._differentiable_functions",
    # Pandas parsers / backends
    "pandas.io.formats.style",
    "pandas._libs.tslibs.timedeltas",
    # Pydantic v2
    "pydantic.deprecated.class_validators",
    "pydantic_core",
    # Click
    "click.core",
    "click.decorators",
    # Matplotlib backends — include the Qt backend for GUI and Agg for headless
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
]

# ---------------------------------------------------------------------------
# Analysis — CLI
# ---------------------------------------------------------------------------

cli_analysis = Analysis(
    [str(SRC / "openpkpd" / "cli" / "main.py")],
    pathex=[str(SRC)],
    binaries=_all_binaries,
    datas=_all_datas,
    hiddenimports=_all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "test", "unittest"],
    noarchive=False,
    optimize=1,
)

# ---------------------------------------------------------------------------
# Analysis — GUI  (skipped when OPENPKPD_SKIP_GUI=1)
# ---------------------------------------------------------------------------

if not _skip_gui:
    gui_analysis = Analysis(
        [str(SRC / "openpkpd_gui" / "app" / "main.py")],
        pathex=[str(SRC)],
        binaries=_all_binaries,
        datas=_all_datas,
        hiddenimports=_all_hidden,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=["tkinter", "test", "unittest"],
        noarchive=False,
        optimize=1,
    )

# ---------------------------------------------------------------------------
# PYZ archives
#
# NOTE: MERGE() was removed in PyInstaller 6.0. Deduplication of shared
# binaries and data files happens implicitly inside COLLECT below, since
# both analyses contribute to the same output directory.  The PYZ archives
# will each contain their own copy of pure-Python modules, adding a modest
# amount of redundancy that is acceptable compared to the build-time
# complexity of manual merging.
# ---------------------------------------------------------------------------

cli_pyz = PYZ(cli_analysis.pure)

if not _skip_gui:
    gui_pyz = PYZ(gui_analysis.pure)

# ---------------------------------------------------------------------------
# EXE — CLI
# ---------------------------------------------------------------------------

cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="openpkpd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(ROOT / "src" / "openpkpd_gui" / "resources" / "icon.ico")
        if (ROOT / "src" / "openpkpd_gui" / "resources" / "icon.ico").exists()
        else None,
)

# ---------------------------------------------------------------------------
# EXE — GUI
# ---------------------------------------------------------------------------

if not _skip_gui:
    gui_exe = EXE(
        gui_pyz,
        gui_analysis.scripts,
        [],
        exclude_binaries=True,
        name="openpkpd-gui",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,      # no terminal window
        windowed=True,
        icon=str(ROOT / "src" / "openpkpd_gui" / "resources" / "icon.ico")
            if (ROOT / "src" / "openpkpd_gui" / "resources" / "icon.ico").exists()
            else None,
        # macOS: wrap in .app bundle
        argv_emulation=sys.platform == "darwin",
        target_arch=None,
        codesign_identity=os.environ.get("MACOS_CODESIGN_IDENTITY"),
        entitlements_file=str(ROOT / "scripts" / "packaging" / "macos" / "entitlements.plist")
            if (ROOT / "scripts" / "packaging" / "macos" / "entitlements.plist").exists()
            else None,
    )

# ---------------------------------------------------------------------------
# COLLECT — single output directory
# ---------------------------------------------------------------------------

if not _skip_gui:
    coll = COLLECT(
        cli_exe,
        cli_analysis.binaries,
        cli_analysis.datas,
        gui_exe,
        gui_analysis.binaries,
        gui_analysis.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="OpenPKPD",
    )
else:
    coll = COLLECT(
        cli_exe,
        cli_analysis.binaries,
        cli_analysis.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="OpenPKPD",
    )

# ---------------------------------------------------------------------------
# macOS: BUNDLE — proper .app for the GUI executable
# ---------------------------------------------------------------------------

if sys.platform == "darwin" and not _skip_gui:
    app = BUNDLE(
        coll,
        name="OpenPKPD-gui.app",
        icon=str(ROOT / "src" / "openpkpd_gui" / "resources" / "icon.icns")
            if (ROOT / "src" / "openpkpd_gui" / "resources" / "icon.icns").exists()
            else None,
        bundle_identifier="org.openpkpd.gui",
        info_plist={
            "CFBundleName": "OpenPKPD",
            "CFBundleDisplayName": "OpenPKPD",
            "CFBundleVersion": os.environ.get("OPENPKPD_VERSION", "0.1.0"),
            "CFBundleShortVersionString": os.environ.get("OPENPKPD_VERSION", "0.1.0"),
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "12.0",
        },
    )
