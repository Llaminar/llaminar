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
mpi_schema_file="src/v2/utils/MPISessionEnvironment.h"
# MPI setup mutates its process environment before each native exec. Its tiny
# C++17 helper/module cannot consume DebugEnv's cached engine configuration.
# Permit only this closed-key bridge body, never another call in the file.
mpi_bridge_line="$(python3 - "${repo_root}/${mpi_schema_file}" <<'PY_BRIDGE'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
if path.is_file():
    text = path.read_text()
    body = ("    inline const char *processEnvironmentValue(EnvironmentKey key)\n"
            "    {\n"
            "        return std::getenv(environmentName(key));\n"
            "    }")
    matches = list(re.finditer(re.escape(body), text))
    if len(matches) == 1:
        print(text[:matches[0].start()].count("\n") + 3)
PY_BRIDGE
)"

is_mpi_accessor_file()
{
    case "$1" in
        src/v2/utils/MPISessionEnvironment.h|src/v2/utils/MPISessionOwner.cpp|src/v2/utils/NativeMPISessionSchizo.cpp)
            return 0 ;;
        *) return 1 ;;
    esac
}

# These startup/bootstrap files intentionally read process environment directly
# before DebugEnv is safe or appropriate to use.
allowed_exception_files=(
    "src/v2/app/RuntimeInitPhase.cpp"
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

    rest="${match#"${abs_path}:"}"
    line_number="${rest%%:*}"
    if [[ "${rel_path}" == "${mpi_schema_file}" && -n "${mpi_bridge_line}" &&
          "${line_number}" == "${mpi_bridge_line}" ]]; then
        continue
    fi

    if is_allowed_file "${rel_path}"; then
        seen_exception_files["${rel_path}"]=1
        continue
    fi

    violations+=("${rel_path}${match#"${abs_path}"}")
done <"${matches_file}"

# The closed MPI bootstrap reader is infrastructure-only. Engine/runtime callers
# continue to use DebugEnv even when they need a key also known to bootstrap.
grep -R -n -I -E '(^|[^[:alnum:]_])processEnvironmentValue[[:space:]]*\(' "${scan_root}" >"${matches_file}" || true
while IFS= read -r match; do
    abs_path="${match%%:*}"
    rel_path="${abs_path#"${repo_root}/"}"
    if ! is_mpi_accessor_file "${rel_path}"; then
        violations+=("MPI bootstrap accessor is infrastructure-only: ${rel_path}${match#"${abs_path}"}")
    fi
done <"${matches_file}"

stale_exceptions=()
for allowed in "${allowed_exception_files[@]}"; do
    if [[ -z "${seen_exception_files[${allowed}]:-}" ]]; then
        stale_exceptions+=("${allowed}")
    fi
done

if (( ${#violations[@]} > 0 )); then
    echo "Direct getenv calls are only allowed in ${debug_env_file} and explicit bootstrap exceptions (including the single closed-key MPI bridge)." >&2
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

echo "Direct getenv hygiene guard passed: all src/ getenv calls are centralized in DebugEnv, legacy bootstrap files, or the exact closed-key MPI bridge."
