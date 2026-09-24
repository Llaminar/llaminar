# SSH development with Codex CLI

The `scripts/llaminar-dev` launcher enters the same Docker container created by
the VS Code Dev Containers extension. It finds the container from the
`devcontainer.local_folder` label, so it is independent of Docker's generated
container name. If the host rebooted and the container is merely stopped, the
launcher starts that same container and reruns the repository's start hook.

The installed host shortcuts are:

```bash
llaminar             # persistent Codex session
llaminar-codex       # the same default entrypoint
llaminar shell       # persistent development shell
llaminar status      # container, tmux, Codex, and login state
llaminar exec COMMAND [ARG...]
llaminar rebuild     # prebuild and recreate the devcontainer
llaminar rebuild-log # inspect the last rebuild after reconnecting
```

Both interactive commands run inside `tmux`. Detach with `Ctrl-b`, then `d`;
the session and its processes continue running when SSH disconnects. Run the
same command later to reattach. A tmux process does not survive a host or
container restart; after a reboot, the launcher restarts the container and
Codex can resume its saved conversation. To begin with a prior Codex
conversation when there is no live `llaminar-codex` tmux session, use:

```bash
llaminar codex resume --last
```

From a Windows terminal, connect to the Linux host normally:

```powershell
ssh dbsanfte@HOST
llaminar
```

An ordinary interactive SSH login already allocates a terminal. For a one-line
remote invocation, request one explicitly:

```powershell
ssh -t dbsanfte@HOST llaminar
```

The `llaminar-codex-state` Docker volume is mounted at `/home/vscode/.codex`.
The standalone CLI and the Codex VS Code extension therefore share one
persistent authority for authentication, configuration, plugins, memories,
and session history. Container deletion and rebuilds do not delete this named
volume. Do not remove it with `docker volume rm` or a volume-pruning command.

The standalone Codex CLI is installed during the devcontainer's post-create
phase, after the persistent volume is mounted. If authentication expires, run
`llaminar shell`, then `codex login --device-auth`; follow the displayed device
sign-in flow from the client device. Never copy `auth.json` into the repository
or commit it.

## Rebuild without VS Code

The host has the official Dev Containers CLI and `tmux` installed. From SSH,
run:

```bash
llaminar rebuild
```

The launcher first completes `devcontainer build`. Only after the image build
succeeds does it run `devcontainer up --remove-existing-container`. The rebuild
runs in the host-side `llaminar-rebuild` tmux session, so it survives an SSH
disconnect. Run `llaminar rebuild` again while it is active to reattach, or use
`llaminar rebuild-log` after it finishes. Use `llaminar rebuild --no-cache`
only when the Docker build cache itself is suspect.

The rebuild refuses to proceed unless the existing `llaminar-codex-state`
volume exists and the configuration mounts it at `/home/vscode/.codex`. The
container is replaceable; the bind-mounted repository and named Codex volume
are not removed.

If setting up another host, install the official standalone CLI before using
the rebuild command:

```bash
curl -fsSL https://raw.githubusercontent.com/devcontainers/cli/main/scripts/install.sh \
  | sh -s -- --version 0.82.0 --prefix "$HOME/.local/devcontainers"
```

The post-create lifecycle is intentionally non-interactive. If GitHub CLI is
not already authenticated after a rebuild, configure it explicitly:

```bash
llaminar shell
bash .devcontainer/setup-github-cli.sh
```
