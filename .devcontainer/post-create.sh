#!/usr/bin/env bash
# @file post-create.sh
# @brief Configure a freshly created Llaminar development container.
#
# Keep this lifecycle hook non-interactive so `devcontainer up` can recreate
# the environment safely from an SSH-hosted tmux session. Interactive account
# setup belongs in an explicit follow-up command after the container is ready.

set -euo pipefail

readonly WORKSPACE_DIR="/workspaces/llaminar"

sudo chmod 666 /dev/dri/render* 2>/dev/null || true
[[ -r "$WORKSPACE_DIR" && -w "$WORKSPACE_DIR" ]] || {
    printf 'post-create: %s is not readable and writable by %s\n' \
        "$WORKSPACE_DIR" "$(id -un)" >&2
    exit 1
}

if ! git config --global --get-all safe.directory | grep -Fxq "$WORKSPACE_DIR"; then
    git config --global --add safe.directory "$WORKSPACE_DIR"
fi

bash "$WORKSPACE_DIR/.devcontainer/setup-ccache.sh"
bash "$WORKSPACE_DIR/.devcontainer/setup-codex-cli.sh"
bash "$WORKSPACE_DIR/.devcontainer/setup-python-env.sh"
bash "$WORKSPACE_DIR/.devcontainer/setup-github-cli.sh" --if-authenticated

cmake --version
gcc --version
