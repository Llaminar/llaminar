#!/usr/bin/env bash
# @file refresh_native_quantised_dispatch_tables.sh
# @brief Run finite exact-overlay collection while retaining generic programs.
# The Python entrypoint validates arguments before discovery or native launches.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${script_dir}/native_quantised/exact_refresh.py" "$@"
