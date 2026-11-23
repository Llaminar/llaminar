#!/usr/bin/env bash
# @file setup-codex-state.sh
# @brief Grant the remapped devcontainer user private access to its mounted Codex state.

set -euo pipefail

readonly CODEX_STATE_DIR="${CODEX_HOME:-$HOME/.codex}"

if [[ ! -d "$CODEX_STATE_DIR" ]]; then
    sudo mkdir -p "$CODEX_STATE_DIR"
fi
if [[ ! -O "$CODEX_STATE_DIR" || ! -w "$CODEX_STATE_DIR" ]]; then
    sudo chown -R "$(id -u):$(id -g)" "$CODEX_STATE_DIR"
fi
chmod 700 "$CODEX_STATE_DIR"
