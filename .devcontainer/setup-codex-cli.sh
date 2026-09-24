#!/usr/bin/env bash
# @file setup-codex-cli.sh
# @brief Install Codex after its persistent state volume is mounted.
#
# The standalone installer stores its package beneath ~/.codex and publishes a
# user-local launcher. Running it during postCreate keeps the package and the
# persistent authentication/session state on the same mounted authority.

set -euo pipefail

readonly CODEX_STATE_DIR="${CODEX_HOME:-$HOME/.codex}"

if [[ ! -d "$CODEX_STATE_DIR" ]]; then
    sudo mkdir -p "$CODEX_STATE_DIR"
fi
if [[ ! -O "$CODEX_STATE_DIR" || ! -w "$CODEX_STATE_DIR" ]]; then
    sudo chown -R "$(id -u):$(id -g)" "$CODEX_STATE_DIR"
fi
chmod 700 "$CODEX_STATE_DIR"

curl -fsSL https://chatgpt.com/codex/install.sh \
    | env CODEX_NON_INTERACTIVE=1 sh
command -v codex >/dev/null
codex --version
