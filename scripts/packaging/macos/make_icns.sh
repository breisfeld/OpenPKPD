#!/usr/bin/env bash
# Generate icon.icns from icon.png using macOS sips + iconutil.
# Run once from the repository root before building the macOS installer:
#   bash scripts/packaging/macos/make_icns.sh
#
# Requires: macOS (sips and iconutil are included with Xcode Command Line Tools)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC_PNG="$REPO_ROOT/src/openpkpd_gui/resources/icon.png"
OUT_ICNS="$REPO_ROOT/src/openpkpd_gui/resources/icon.icns"
ICONSET="$REPO_ROOT/build/icon.iconset"

if [ ! -f "$SRC_PNG" ]; then
    echo "ERROR: source icon not found at $SRC_PNG" >&2
    exit 1
fi

mkdir -p "$ICONSET"

# Generate all required sizes
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SRC_PNG" --out "$ICONSET/icon_${size}x${size}.png"       >/dev/null
    sips -z $((size*2)) $((size*2)) "$SRC_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
rm -rf "$ICONSET"

echo "Created: $OUT_ICNS"
