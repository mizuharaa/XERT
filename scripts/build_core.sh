#!/usr/bin/env bash
# Requires .venv runtime dependencies, CMake >= 3.20, Make and a C++17 compiler.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cmake --preset linux
cmake --build --preset linux
