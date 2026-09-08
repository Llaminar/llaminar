#!/usr/bin/env bash
# Release-container E2E adapter. Model, topology, axes and context geometry
# come exclusively from typed model-parity definitions, not a container matrix.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
variant=""
image=""
log_dir=""
base_port=20100
build_dir="${repo_root}/build_v2_integration"
manifest=""
dry_run=false
while (($#)); do
    case "$1" in
        --variant) variant="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --port) base_port="$2"; shift 2 ;;
        --build-dir) build_dir="$2"; shift 2 ;;
        --manifest) manifest="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help)
            echo "Usage: $0 --variant cpu|cuda|rocm|hybrid --image IMAGE [--build-dir DIR] [--log-dir DIR] [--port N] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$image" ]] || { echo "--image is required" >&2; exit 1; }
case "$variant" in
    cpu) backend='CPU' ;;
    cuda) backend='(?=.*CUDA)(?!.*ROCm).*' ;;
    rocm) backend='(?=.*ROCm)(?!.*CUDA).*' ;;
    hybrid) backend='.*CUDA.*ROCm.*' ;;
    *) echo "Invalid --variant: $variant" >&2; exit 1 ;;
esac
log_dir="${log_dir:-/tmp/llaminar-e2e-${variant}-container}"
cmd=(python3 "${script_dir}/run_model_parity_e2e.py"
    --build-dir "$build_dir" --container-image "$image" --backend "$backend"
    --report "${log_dir}/certification.json" --port "$base_port")
if [[ -n "$manifest" ]]; then
    cmd+=(--manifest "$manifest")
fi
if $dry_run; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    exec "${cmd[@]}"
fi
