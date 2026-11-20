#!/usr/bin/env bash
## @file rocm-image-targets.sh
# @brief Resolve one concrete GPU target list for ROCm image builders and libraries.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/rocm-release.env"
requested="${1:-${LLAMINAR_ROCM_GPU_TARGETS}}"
if [[ "${requested}" == *$'\n'* ]]; then
    echo "ROCm image GPU targets must form one list" >&2
    exit 1
fi
requested="${requested//,/ }"
requested="${requested//;/ }"
read -r -a targets <<< "${requested}"
if (("${#targets[@]}" == 0)); then
    echo "ROCm image GPU target list must not be empty" >&2
    exit 1
fi
declare -A seen=()
for target in "${targets[@]}"; do
    if [[ ! "${target}" =~ ^gfx[0-9][0-9a-f]+$ || -n "${seen[$target]:-}" ]]; then
        echo "Invalid or duplicate ROCm image GPU target: ${target}" >&2
        exit 1
    fi
    seen["${target}"]=1
done
(IFS=';'; printf '%s\n' "${targets[*]}")
