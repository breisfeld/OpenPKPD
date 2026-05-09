# dmgbuild settings for OpenPKPD macOS DMG.
# Called by build_installer.py via: dmgbuild -s this_file "OpenPKPD <ver>" out.dmg
# Environment variables injected by build_installer.py:
#   OPENPKPD_APP_PATH  — path to the .app bundle
#   OPENPKPD_VERSION   — version string

import os
from pathlib import Path

app_path = os.environ.get("OPENPKPD_APP_PATH", "dist/pyinstaller/OpenPKPD/OpenPKPD-gui.app")
version  = os.environ.get("OPENPKPD_VERSION", "0.2.8")

# Volume appearance
application = app_path
appname = Path(app_path).name

files = [app_path]
symlinks = {"Applications": "/Applications"}

# Window layout
size = (640, 400)
background = "builtin-arrow"     # simple arrow; replace with a PNG path for branding

icon_locations = {
    appname:        (160, 200),
    "Applications": (480, 200),
}

# DMG format
format = "UDZO"
filesystem = "HFS+"
volume_name = f"OpenPKPD {version}"
