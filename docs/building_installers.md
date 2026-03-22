# Building Standalone Installers

OpenPKPD can be packaged into a self-contained, redistributable application
for Windows, macOS, and Linux using PyInstaller.  The resulting artifact
requires no Python installation on the end-user machine.

## Prerequisites

Install the packaging and GUI extras (one-time per environment):

```bash
uv sync --extra packaging --extra gui
```

`packaging` pulls in PyInstaller; `gui` pulls in PySide6.  Both are needed
for a full GUI build.

**macOS only:** Xcode Command Line Tools are required to generate the
`.icns` icon file:

```bash
xcode-select --install
```

## Running the build

From the repository root:

```bash
just build-installer
```

Or directly:

```bash
uv run python scripts/build_installer.py
```

To build the CLI only (no PySide6 required — faster, smaller output):

```bash
just build-installer-cli
# equivalent:
uv run python scripts/build_installer.py --skip-gui
```

## Output artifacts

| Platform | Artifact |
|----------|----------|
| Linux    | `dist/linux/OpenPKPD-<version>-linux-x86_64.tar.gz` |
| macOS    | `dist/macos/OpenPKPD-<version>-macos-<arch>.dmg` |
| Windows  | `dist/windows/OpenPKPD-<version>-windows-x64.zip` |

Optional extras are produced automatically when the required tools are on
your `PATH`:

- **Linux:** an `.AppImage` if `appimagetool` is available.
- **macOS:** a DMG built with `dmgbuild` if installed, otherwise `hdiutil`.
- **Windows:** an NSIS `.exe` installer if `makensis` is available.

## Build steps

The build script runs two stages:

1. **Freeze (PyInstaller)** — collects the Python interpreter, all
   dependencies, Qt plugins, and application code into a single directory
   `dist/pyinstaller/OpenPKPD/`.  The directory contains two executables:
   - `openpkpd` — CLI entry point (console mode)
   - `openpkpd-gui` — GUI entry point (windowed, no terminal)
   - `_internal/` — shared libraries and Qt plugin tree

2. **Package** — wraps the frozen directory into the platform-native format
   (tarball / DMG / zip) and, where tools are available, an installer.

## Advanced options

```
--version 1.2.3       Override the version string (default: from pyproject.toml)
--output-dir /path    Write artifacts to a custom directory
--platform linux      Override platform auto-detection (linux / macos / windows)
--skip-gui            CLI-only build; PySide6 not required
```

Example:

```bash
uv run python scripts/build_installer.py --version 1.2.3 --output-dir /tmp/release
```

## Platform notes

### macOS

- `icon.icns` is generated automatically from `src/openpkpd_gui/resources/icon.png`
  before each build using `scripts/packaging/macos/make_icns.sh`.  Requires
  `sips` and `iconutil` (included with Xcode CLI tools).
- To code-sign the app bundle, set the environment variable:
  ```bash
  export MACOS_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
  just build-installer
  ```
- An `entitlements.plist` for sandboxing / hardened runtime is located at
  `scripts/packaging/macos/entitlements.plist`.

### Linux

- The tarball is always produced.  The AppImage requires
  [`appimagetool`](https://github.com/AppImage/AppImageKit) on your `PATH`.
- The AppImage launcher (`scripts/packaging/linux/AppDir/AppRun`) automatically
  selects the CLI or GUI based on how the AppImage is invoked:
  - Default invocation → GUI
  - Invoked as `openpkpd` (symlink) or with `--cli` flag → CLI

### Windows

- Build natively on Windows or via a Windows CI runner.
- The zip archive is always produced.  The NSIS installer additionally
  requires [NSIS](https://nsis.sourceforge.io/) (`makensis` on PATH) and the
  script at `scripts/packaging/windows/installer.nsi`.
