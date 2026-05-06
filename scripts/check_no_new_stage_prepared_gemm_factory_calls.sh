#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    repo_root="$1"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

stage_dir="${repo_root}/src/v2/execution/compute_stages/stages"
pattern="KernelFactory::getOrCreatePreparedGemmWeights|getOrCreatePreparedGemmWeights\("

list_matches_with_counts() {
    if command -v rg >/dev/null 2>&1; then
        rg --count-matches "${pattern}" "${stage_dir}" || true
        return
    fi

    find "${stage_dir}" -type f \( -name '*.cpp' -o -name '*.h' \) -print0 |
        while IFS= read -r -d '' file; do
            count="$(grep -E -o "${pattern}" "${file}" 2>/dev/null | wc -l)"
            if (( count > 0 )); then
                echo "${file}:${count}"
            fi
        done
}

print_matches() {
    if command -v rg >/dev/null 2>&1; then
        rg -n "${pattern}" "${stage_dir}" >&2 || true
        return
    fi

    grep -R -n -E "${pattern}" "${stage_dir}" >&2 || true
}

declare -A allowed_counts=()

declare -A seen_counts=()

while IFS=: read -r file count; do
    [[ -n "${file}" ]] || continue
    base="$(basename "${file}")"
    seen_counts["${base}"]="${count}"
done < <(list_matches_with_counts)

failed=0

for base in "${!seen_counts[@]}"; do
    actual="${seen_counts[${base}]}"
    allowed="${allowed_counts[${base}]:-0}"
    if (( actual > allowed )); then
        echo "error: ${base} has ${actual} forbidden prepared-GEMM factory call(s), allowed is ${allowed}" >&2
        failed=1
    fi
done

for base in "${!allowed_counts[@]}"; do
    actual="${seen_counts[${base}]:-0}"
    allowed="${allowed_counts[${base}]}"
    if (( actual > allowed )); then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "" >&2
    echo "New stage-level calls to KernelFactory::getOrCreatePreparedGemmWeights are forbidden." >&2
    echo "Route model-stage GEMM resolution through PreparedWeightStore instead." >&2
    echo "Current matches:" >&2
    print_matches
    exit 1
fi

echo "Prepared-GEMM fallback guard passed: no stage file calls the deleted KernelFactory fallback."
