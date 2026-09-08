#!/usr/bin/env bash
# @file setup-github-cli.sh
# @brief Authenticate GitHub CLI when requested and configure Git identity.

set -euo pipefail

readonly AUTH_MODE="${1:-interactive}"

case "$AUTH_MODE" in
    interactive|--if-authenticated)
        ;;
    *)
        printf 'setup-github-cli: unknown mode %q\n' "$AUTH_MODE" >&2
        exit 1
        ;;
esac

command -v gh >/dev/null 2>&1 || {
    printf 'setup-github-cli: GitHub CLI is not installed\n' >&2
    exit 1
}

if ! gh auth status >/dev/null 2>&1; then
    if [[ "$AUTH_MODE" == "--if-authenticated" ]]; then
        printf '%s\n' \
            'GitHub CLI is not authenticated. After the rebuild, run:' \
            '  llaminar shell' \
            '  bash .devcontainer/setup-github-cli.sh'
        exit 0
    fi
    gh auth login
fi

github_user="$(gh api user --jq .login)"
github_name="$(gh api user --jq '.name // empty')"
github_id="$(gh api user --jq .id)"

git config --global user.name "${github_name:-$github_user}"
git config --global user.email "${github_id}+${github_user}@users.noreply.github.com"
printf 'Configured Git identity for %s.\n' "$github_user"
