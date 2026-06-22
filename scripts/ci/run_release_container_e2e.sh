#!/usr/bin/env bash
# Run serialized E2E subsets against an already-built release container image.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/run_release_container_e2e.sh --variant cpu|cuda|rocm|hybrid --image IMAGE [options]

Options:
  --variant VARIANT     E2E subset to run.
  --image IMAGE         Release container image to test.
  --models-dir DIR      Host model directory. Default: MODELS_DIR or /opt/llaminar-models.
  --log-dir DIR         E2E log directory. Default: /tmp/llaminar-e2e-<variant>-container.
  --port PORT           Base port. Default: 20100/20200/20300/20400 by variant.
  --dry-run             Print the command without running it.
  -h, --help            Show this help.

The suite lists intentionally mirror the release-container validation matrix:
CPU image covers CPU-only baseline/prefix/MTP cases, CUDA image covers CUDA and
CUDA TP cases, ROCm image covers ROCm and ROCm TP cases, and the hybrid subset
covers the full CUDA+ROCm image's pipeline-parallel CUDA/ROCm case.
EOF
}

die() {
    echo "run_release_container_e2e: error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

variant=""
image=""
models_dir="${MODELS_DIR:-/opt/llaminar-models}"
log_dir=""
base_port=""
dry_run=false

while (($#)); do
    case "$1" in
        --variant)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            variant="$2"
            shift 2
            ;;
        --image)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            image="$2"
            shift 2
            ;;
        --models-dir)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            models_dir="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            log_dir="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            base_port="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "$variant" ]] || die "--variant is required"
[[ -n "$image" ]] || die "--image is required"

case "$variant" in
    cpu)
        : "${base_port:=20100}"
        ;;
    cuda)
        : "${base_port:=20200}"
        ;;
    rocm)
        : "${base_port:=20300}"
        ;;
    hybrid)
        : "${base_port:=20400}"
        ;;
    *)
        die "unsupported variant '$variant'"
        ;;
esac
: "${log_dir:=/tmp/llaminar-e2e-${variant}-container}"

model_path() {
    printf '%s/%s' "$models_dir" "$1"
}

require_model() {
    local path
    path="$(model_path "$1")"
    [[ -f "$path" ]] || die "required model is missing: $path"
}

add_suite() {
    cmd+=(--suite "$1")
}

cmd=(
    "${repo_root}/tests/v2/e2e/server/test_server_e2e.sh"
    --container-image "$image"
    --port "$base_port"
)
e2e_env=(
    "LLAMINAR_E2E_LONG_CONTEXT=1"
    "LLAMINAR_E2E_LONG_CONTEXT_TIER=full"
    "LLAMINAR_E2E_LONG_MAX_TOKENS=2048"
    "LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=420"
    "LLAMINAR_E2E_LOG_DIR=${log_dir}"
)

case "$variant" in
    rocm)
        rocm_debug_docker_args="${LLAMINAR_E2E_DOCKER_ARGS:-}"
        rocm_debug_docker_args="${rocm_debug_docker_args:+${rocm_debug_docker_args} }-e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO"
        e2e_env+=("LLAMINAR_E2E_DOCKER_ARGS=${rocm_debug_docker_args}")
        ;;
esac

case "$variant" in
    cpu)
        require_model "qwen2.5-1.5b-instruct-q8_0.gguf"
        require_model "Qwen3.5-4B-Q8_0.gguf"
        require_model "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
        require_model "Qwen3.5-27B-Q4_K_M.gguf"
        require_model "Qwen3.6-27B-Q4_K_S.gguf"
        require_model "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
        add_suite "$(model_path qwen2.5-1.5b-instruct-q8_0.gguf)|cpu|200||qwen25-cpu-full"
        add_suite "$(model_path Qwen3.5-4B-Q8_0.gguf)|cpu|200||qwen35-4b-cpu-full"
        add_suite "$(model_path Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf)|cpu|200||qwen35-moe-cpu-full"
        add_suite "$(model_path Qwen3.5-27B-Q4_K_M.gguf)|cpu|200||qwen35-dense-cpu-full"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cpu|200||qwen36-dense-baseline-cpu-full"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cpu|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto|qwen36-dense-prefix-cpu-full|no-long-context"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cpu|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-dense-mtp-cpu-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cpu|200||qwen36-moe-baseline-cpu-full"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cpu|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint|qwen36-moe-prefix-cpu-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cpu|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-moe-mtp-cpu-full|no-long-context"
        ;;
    cuda)
        require_model "qwen2.5-1.5b-instruct-q8_0.gguf"
        require_model "qwen2.5-0.5b-instruct-q8_0.gguf"
        require_model "Qwen3.5-4B-Q8_0.gguf"
        require_model "Qwen3.5-35B-A3B-Q3_K_S.gguf"
        require_model "Qwen3.6-27B-Q4_K_S.gguf"
        require_model "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
        add_suite "$(model_path qwen2.5-1.5b-instruct-q8_0.gguf)|cuda:0|200||qwen25-cuda-full"
        add_suite "$(model_path qwen2.5-0.5b-instruct-q8_0.gguf)|cuda:0|16||qwen25-cuda-prefill-graph-full|prefill-graph-probe"
        add_suite "$(model_path Qwen3.5-4B-Q8_0.gguf)|cuda:0|200||qwen35-4b-cuda-full"
        add_suite "$(model_path Qwen3.5-35B-A3B-Q3_K_S.gguf)|cuda:0|200||qwen35-moe-q3-cuda-full"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cuda:0|200||qwen36-dense-baseline-cuda-full"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cuda:0|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto|qwen36-dense-prefix-cuda-full|no-long-context"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|cuda:0|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-dense-mtp-cuda-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cuda:0|200||qwen36-moe-baseline-cuda-full"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cuda:0|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint|qwen36-moe-prefix-cuda-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|cuda:0|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-moe-mtp-cuda-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint --tp-devices cuda:0,cuda:1|qwen36-moe-prefix-cuda2tp-full|no-long-context,no-prefill-graph-buckets"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy --tp-devices cuda:0,cuda:1|qwen36-moe-mtp-cuda2tp-full|no-long-context,no-prefill-graph-buckets"
        ;;
    rocm)
        require_model "qwen2.5-1.5b-instruct-q8_0.gguf"
        require_model "Qwen3.5-4B-Q8_0.gguf"
        require_model "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
        require_model "Qwen3.5-27B-Q4_K_M.gguf"
        require_model "Qwen3.6-27B-Q4_K_S.gguf"
        require_model "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
        add_suite "$(model_path qwen2.5-1.5b-instruct-q8_0.gguf)|rocm:0|200||qwen25-rocm-full"
        add_suite "$(model_path Qwen3.5-4B-Q8_0.gguf)|rocm:0|200||qwen35-4b-rocm-full"
        add_suite "$(model_path Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf)|rocm:0|200||qwen35-moe-rocm-full"
        add_suite "$(model_path Qwen3.5-27B-Q4_K_M.gguf)|rocm:0|200||qwen35-dense-rocm-full"
        add_suite "$(model_path Qwen3.5-27B-Q4_K_M.gguf)|tp|200|--tp-devices rocm:0,rocm:1|qwen35-dense-rocm2tp-full|no-prefill-graph-buckets"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|rocm:0|200||qwen36-dense-baseline-rocm-full"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|rocm:0|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto|qwen36-dense-prefix-rocm-full|no-long-context"
        add_suite "$(model_path Qwen3.6-27B-Q4_K_S.gguf)|rocm:0|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-dense-mtp-rocm-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|rocm:0|200||qwen36-moe-baseline-rocm-full"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|rocm:0|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint|qwen36-moe-prefix-rocm-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|rocm:0|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy|qwen36-moe-mtp-rocm-full|no-long-context"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint --tp-devices rocm:0,rocm:1|qwen36-moe-prefix-rocm2tp-full|no-long-context,no-prefill-graph-buckets"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy --tp-devices rocm:0,rocm:1|qwen36-moe-mtp-rocm2tp-full|no-long-context,no-prefill-graph-buckets"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto --prefix-cache-moe-policy placement-fingerprint --tp-devices rocm:0,rocm:1,rocm:2,rocm:3|qwen36-moe-prefix-rocm4tp-full|no-long-context,no-prefill-graph-buckets"
        add_suite "$(model_path Qwen3.6-35B-A3B-UD-IQ3_S.gguf)|tp|200|--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy --tp-devices rocm:0,rocm:1,rocm:2,rocm:3|qwen36-moe-mtp-rocm4tp-full|no-long-context,no-prefill-graph-buckets"
        ;;
    hybrid)
        require_model "Qwen3.5-27B-Q4_K_M.gguf"
        add_suite "$(model_path Qwen3.5-27B-Q4_K_M.gguf)|pp|200|--define-domain cuda_pp=cuda:0 --define-domain rocm_pp=rocm:0 --pp-stage 0=cuda_pp:0-31 --pp-stage 1=rocm_pp:32-63|qwen35-dense-localpp-cuda-rocm|no-prefill-graph-buckets"
        ;;
esac

printf '[run-release-container-e2e] '
printf '%q ' env "${e2e_env[@]}" "${cmd[@]}"
printf '\n'

if [[ "$dry_run" == "true" ]]; then
    exit 0
fi

env "${e2e_env[@]}" "${cmd[@]}"
