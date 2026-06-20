#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    repo_root="$1"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

scan_root="${repo_root}/src"
debug_env_file="src/v2/utils/DebugEnv.h"
pattern='(^|[^[:alnum:]_])(std::)?getenv[[:space:]]*\('

# These startup/bootstrap files intentionally read process environment directly
# before DebugEnv is safe or appropriate to use.
allowed_exception_files=(
    "src/v2/app/RuntimeInitPhase.cpp"
    "src/v2/backends/ComputeBackend.cpp"
    "src/v2/backends/cuda/NvidiaContextFactory.cu"
    "src/v2/backends/rocm/AMDContextFactory.cpp"
    "src/v2/utils/CPUFeatures.h"
)

is_allowed_file()
{
    local rel_path="$1"

    if [[ "${rel_path}" == "${debug_env_file}" ]]; then
        return 0
    fi

    for allowed in "${allowed_exception_files[@]}"; do
        if [[ "${rel_path}" == "${allowed}" ]]; then
            return 0
        fi
    done

    return 1
}

matches_file="$(mktemp)"
trap 'rm -f "${matches_file}"' EXIT

grep -R -n -I -E "${pattern}" "${scan_root}" >"${matches_file}" || true

declare -A seen_exception_files=()
violations=()

while IFS= read -r match; do
    abs_path="${match%%:*}"
    rel_path="${abs_path#"${repo_root}/"}"

    if is_allowed_file "${rel_path}"; then
        seen_exception_files["${rel_path}"]=1
        continue
    fi

    violations+=("${rel_path}${match#"${abs_path}"}")
done <"${matches_file}"

stale_exceptions=()
for allowed in "${allowed_exception_files[@]}"; do
    if [[ -z "${seen_exception_files[${allowed}]:-}" ]]; then
        stale_exceptions+=("${allowed}")
    fi
done

if (( ${#violations[@]} > 0 )); then
    echo "Direct getenv calls are only allowed in ${debug_env_file} and explicit bootstrap exceptions." >&2
    echo "Move new runtime/debug environment access to DebugEnv instead." >&2
    echo "" >&2
    printf '%s\n' "${violations[@]}" >&2
    exit 1
fi

if (( ${#stale_exceptions[@]} > 0 )); then
    echo "Direct getenv exception allowlist contains files with no remaining getenv calls." >&2
    echo "Remove migrated files from scripts/check_no_direct_getenv_outside_debugenv.sh:" >&2
    echo "" >&2
    printf '%s\n' "${stale_exceptions[@]}" >&2
    exit 1
fi

echo "Direct getenv hygiene guard passed: all src/ getenv calls are centralized in DebugEnv or explicitly allowlisted bootstrap files."