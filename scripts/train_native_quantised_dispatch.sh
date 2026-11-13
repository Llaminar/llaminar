#!/usr/bin/env bash
# @file train_native_quantised_dispatch.sh
# @brief Reject the retired generic-fit transaction before any side effects.
# Installed generic decisions remain valid; new evidence updates exact overlays.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Generic NativeQuantised training is retired."
  echo "Use scripts/refresh_native_quantised_dispatch_tables.sh --help for exact overlays."
  exit 0
fi
echo "error: generic NativeQuantised training/refitting is retired; use scripts/refresh_native_quantised_dispatch_tables.sh for exact overlays. Installed generic runtime policies remain unchanged." >&2
exit 2
