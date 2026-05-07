#!/usr/bin/env bash
# Build the manylinux_2_28 wheel with native-cvodes inside the maturin Docker image.
#
# Three workarounds are required for the ghcr.io/pyo3/maturin container:
#
#   1. cmake<4  — CMake 4.0 removed support for cmake_minimum_required < 3.5,
#                 which breaks the sundials-sys vendor POSIX-timers try_compile.
#
#   2. libclang — the container's system clang is 3.4.2 (a CentOS compat shim),
#                 too old for clang-sys 1.8.1.  Install a modern bundled libclang
#                 via pip and point LIBCLANG_PATH at the .so.
#                 Use /opt/python/cp312-cp312/bin/pip so the library lands in the
#                 3.12 site-packages (cosmetically consistent with the build target).
#
#   3. BINDGEN_EXTRA_CLANG_ARGS — the bundled libclang doesn't know the
#                 devtoolset-10 GCC include paths, so bindgen can't find stddef.h
#                 when parsing SUNDIALS headers.  Add -I flags explicitly.
#
# Usage (inside the maturin container):
#   bash /io/scripts/build_manylinux_wheel.sh

set -euo pipefail

PIP312="/opt/python/cp312-cp312/bin/pip"
PYTHON312="/opt/python/cp312-cp312/bin/python3.12"

echo "==> Installing build-time dependencies"
"$PIP312" install -q "cmake<4" libclang

echo "==> Locating libclang.so"
LIBCLANG_SO=$(find /opt/python/cp312-cp312 /opt/_internal -name 'libclang.so' 2>/dev/null | head -1)
if [[ -z "$LIBCLANG_SO" ]]; then
    echo "ERROR: libclang.so not found after pip install libclang" >&2
    exit 1
fi
LIBCLANG_PATH=$(dirname "$LIBCLANG_SO")
echo "    LIBCLANG_PATH=$LIBCLANG_PATH"
export LIBCLANG_PATH

echo "==> Locating GCC include dir (devtoolset-10)"
GCC_INCDIR=$(/opt/rh/devtoolset-10/root/usr/bin/gcc -print-file-name=include)
echo "    GCC_INCDIR=$GCC_INCDIR"
export BINDGEN_EXTRA_CLANG_ARGS="-I$GCC_INCDIR -I/usr/include"

echo "==> Building manylinux_2_28 wheel (CPython 3.12, native-cvodes)"
maturin build \
    --manifest-path /io/rust/Cargo.toml \
    --release \
    --features native-cvodes \
    --compatibility manylinux_2_28 \
    --interpreter "$PYTHON312" \
    --out dist/

echo "==> Done.  Wheel written to dist/"
