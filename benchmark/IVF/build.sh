#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Opt-in, separate build tree. Pass administrator-installed dependency overrides
# as quoted CMake arguments; never reconfigure an existing baseline build.
BUILD_DIR="${IVF_BUILD_DIR:-$REPO_DIR/build/ivf-benchmark}"
cmake -S "$REPO_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DIVF_BUILD_ROUTING=ON "$@"
cmake --build "$BUILD_DIR" --target ivf_build ivf_driver ivf_training_diagnostics -j"${IVF_BUILD_JOBS:-4}"
