#!/bin/bash
# =============================================================================
# E2E Server Integration Test — Multi-Turn Inference via REST API
#
# Tests the Llaminar HTTP server (serve subcommand) with curl against the
# /v1/chat/completions endpoint for multiple model × backend combinations.
#
# Default test suites:
#   Suite 1: Qwen2.5 1.5B Q8_0 on cpu, cuda:0, rocm:0
#   Suite 2: Qwen3.5 4B   Q8_0 on cpu, cuda:0, rocm:0
#   Suite 3: Qwen3.5 35B MoE Q4_K_XL on cpu
#   Suite 4: Qwen3.5 35B MoE Q4_K_XL on rocm:0
#   Suite 5: Qwen3.5 27B dense Q4_K_M on cpu, rocm:0 (too large for single 24GB CUDA GPU)
#   Suite 6: Qwen3.5 27B dense Q4_K_M TP2 on rocm:0,rocm:1
#   Suite 7: Qwen3.5 27B dense Q4_K_M PP2 on cuda:0+rocm:0 (equal 32/32 layer split)
#   Suite 8+: Qwen3.6 dense/MoE baseline, prefix-cache, and MTP server cases
#
# Each backend test:
#   1. Starts llaminar2 serve on a unique port
#   2. Waits for /health to respond
#   3. Sends a single-turn greedy chat request, validates response
#      Thinking models are checked in both thinking and non-thinking modes.
#   4. Sends a multi-turn conversation, validates response in both modes
#   5. Sends a second independent request (tests KV cache clearing) in both modes
#   6. Validates response format (usage, finish_reason)
#   7. Tests streaming in both modes for thinking models
#   8. Tests error handling (invalid JSON, missing messages)
#   9. Optionally runs objective long-context checks for 4B+ models
#  10. Measures process RSS / GPU memory and scans the server log for WARN/ERROR
#  11. Kills server, moves to next backend
#
# Usage:
#   ./test_server_e2e.sh [--binary <path>] [--model <path>] [--backends <list>]
#   ./test_server_e2e.sh [--binary <path>] [--suite "model_path|backend1,backend2[|max_tokens[|extra_flags[|label[|suite_options]]]]]"] ...
#   ./test_server_e2e.sh --container-image llaminar:local [--model <path>] [--backends <list>]
#   LLAMINAR_E2E_LONG_CONTEXT=1 ./test_server_e2e.sh [options]
#
# Optional long-context mode runs only when the model path, basename, or label
# contains a parsed size >= LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B (default: 4B).
#
# Environment:
#   LLAMINAR_BINARY     Override binary path
#   LLAMINAR_E2E_CONTAINER_IMAGE Run server from this Docker image instead of
#                       launching LLAMINAR_BINARY on the host/devcontainer
#   LLAMINAR_E2E_DOCKER_NETWORK Docker network mode for container server
#                       (default: auto; shares devcontainer netns when inside
#                       Docker, otherwise host)
#   LLAMINAR_E2E_DOCKER_GPUS GPU flag for docker run: auto|all|none (default: auto)
#                       Uses NVIDIA Container Toolkit when available. In a
#                       GPU-enabled devcontainer, falls back to pass-through
#                       of visible NVIDIA driver libraries and device nodes.
#   LLAMINAR_E2E_DOCKER_USER User passed to docker run --user. Defaults to
#                       0:0 so nested devcontainer bind mounts are writable.
#   LLAMINAR_E2E_DOCKER_NUMA_SECCOMP
#                       Add --security-opt seccomp=unconfined so mbind/set_mempolicy
#                       NUMA syscalls work in Docker. Default: 1.
#   LLAMINAR_E2E_DOCKER_ARGS Extra docker run args, shell-split. Use this for
#                       host-specific privileges such as --cap-add or seccomp.
#   LLAMINAR_MODEL      Override model path (overrides default suite 1)
#   LLAMINAR_BACKENDS   Override backends for suite 1
#   LLAMINAR_LOG_LEVEL  Log level for server (default: WARN; ERROR is promoted
#                       to WARN so this harness can catch warnings)
#   LLAMINAR_E2E_LOG_DIR Override per-case server log directory
#   LLAMINAR_E2E_LONG_CONTEXT Enable optional long-context checks (default: 0)
#   LLAMINAR_E2E_LONG_CONTEXT_TIER lite|full long-context tier (default: full)
#   LLAMINAR_E2E_CONTEXT_LENGTH Context length passed with -c for eligible models (default: 4096)
#   LLAMINAR_E2E_LONG_MAX_TOKENS Long-generation max_tokens (default: 1024)
#   LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS Minimum helper prompt tokens (default: 900)
#   LLAMINAR_E2E_LONG_REQUEST_TIMEOUT Long helper request timeout (default: REQUEST_TIMEOUT)
#   LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B Minimum parsed model size in billions (default: 4)
#   LLAMINAR_E2E_GPU_RELEASE_TIMEOUT_SECONDS Seconds to poll for GPU VRAM release
#                       after server shutdown before declaring a leak (default: 30)
#   LLAMINAR_E2E_PERF_STATS Enable per-case PerfStats JSON artifacts and graph
#                       capture assertions for MTP GPU cases (default: 1)
#   LLAMINAR_E2E_PERF_STATS_GPU_STAGE_TIMING Include GPU stage timing in
#                       PerfStats artifacts where supported (default: 1)
#   LLAMINAR_E2E_ENABLE_REMOTE_EXPERT_OVERLAY Enable experimental remote
#                       NodeLocal CPU-cold ExpertOverlay suites. These are
#                       intentionally off until production participant graphs
#                       run matched MPI sparse dispatch/expert/return stages.
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

BINARY="${LLAMINAR_BINARY:-${REPO_ROOT}/build_v2_integration/llaminar2}"
SERVER_MODE="${LLAMINAR_E2E_SERVER_MODE:-local}"
CONTAINER_IMAGE="${LLAMINAR_E2E_CONTAINER_IMAGE:-}"
if [[ -n "$CONTAINER_IMAGE" && "${LLAMINAR_E2E_SERVER_MODE:-}" == "" ]]; then
    SERVER_MODE="docker"
fi
DOCKER_NETWORK="${LLAMINAR_E2E_DOCKER_NETWORK:-auto}"
DOCKER_GPUS="${LLAMINAR_E2E_DOCKER_GPUS:-auto}"
DOCKER_SHM_SIZE="${LLAMINAR_E2E_DOCKER_SHM_SIZE:-16g}"
DOCKER_NAME_PREFIX="${LLAMINAR_E2E_DOCKER_NAME_PREFIX:-llaminar-e2e}"
DOCKER_USER="${LLAMINAR_E2E_DOCKER_USER:-0:0}"
DOCKER_NUMA_SECCOMP="${LLAMINAR_E2E_DOCKER_NUMA_SECCOMP:-1}"
declare -a DOCKER_EXTRA_ARGS=()
if [[ -n "${LLAMINAR_E2E_DOCKER_ARGS:-}" ]]; then
    # Intentional shell-style word splitting for advanced docker run flags.
    read -r -a DOCKER_EXTRA_ARGS <<< "${LLAMINAR_E2E_DOCKER_ARGS}"
fi
LOG_LEVEL="${LLAMINAR_LOG_LEVEL:-WARN}"
if [[ "${LOG_LEVEL^^}" == "ERROR" ]]; then
    LOG_LEVEL="WARN"
fi
BASE_PORT=19080

HOST_RSS_CPU_MODEL_MULTIPLIER="${LLAMINAR_E2E_HOST_RSS_CPU_MODEL_MULTIPLIER:-5}"
HOST_RSS_GPU_MODEL_MULTIPLIER="${LLAMINAR_E2E_HOST_RSS_GPU_MODEL_MULTIPLIER:-2}"
HOST_RSS_EXTRA_MB="${LLAMINAR_E2E_HOST_RSS_EXTRA_MB:-5120}"
CPU_GPU_DELTA_LIMIT_MB="${LLAMINAR_E2E_CPU_GPU_DELTA_LIMIT_MB:-128}"
GPU_ACTIVE_MIN_MB="${LLAMINAR_E2E_GPU_ACTIVE_MIN_MB:-256}"
GPU_RELEASE_TIMEOUT_SECONDS="${LLAMINAR_E2E_GPU_RELEASE_TIMEOUT_SECONDS:-30}"
TRACE_TOKENS="${LLAMINAR_E2E_TRACE_TOKENS:-0}"
REMOTE_EXPERT_OVERLAY_E2E="${LLAMINAR_E2E_ENABLE_REMOTE_EXPERT_OVERLAY:-0}"
PERF_STATS_ENABLED="${LLAMINAR_E2E_PERF_STATS:-1}"
PERF_STATS_GPU_STAGE_TIMING="${LLAMINAR_E2E_PERF_STATS_GPU_STAGE_TIMING:-1}"
# Thinking-capable Qwen models may spend hundreds of tokens deliberating before
# answering. The E2E suite is a bounded server regression gate, so exercise the
# thinking-budget path by default instead of relying on unconstrained thinking
# to finish within the HTTP timeout. Set LLAMINAR_E2E_THINKING_BUDGET_TOKENS=""
# to request the production default of no explicit budget.
THINKING_BUDGET_TOKENS="${LLAMINAR_E2E_THINKING_BUDGET_TOKENS-16}"

# Model suites: "model_path|backend1,backend2,...[|max_tokens[|extra_flags[|label[|suite_options]]]]"
#   or shorthand: "model_path|backend1,backend2,...|extra_flags" (max_tokens omitted → defaults to 200)
# Uses '|' as delimiter (not ':') because device names contain colons (cuda:0).
# The optional 4th field (extra_flags) is passed verbatim to the server command.
# The optional 5th field (label) is a short display/log suffix for feature cases.
# The optional 6th field (suite_options) is harness metadata. Supported:
#   no-long-context  Skip duplicate optional long-context helper for feature variants.
#   no-prefill-graph-buckets  Opt this suite out of default bucketed prefill graph capture.
#   prefill-graph-probe  Send repeated same-key long-enough prompts to prove capture/replay.
#   require-prefill-graph-capture  Fail unless perfstats record prefill capture/replay.
# If the 3rd field is non-numeric, it's treated as extra_flags (max_tokens defaults to 200).
# Each --suite flag appends to the list. If none given, defaults are used.
declare -a SUITES=()
STARTED_SERVER_HANDLE=""
OVERRIDE_MODEL=""
OVERRIDE_BACKENDS=""

show_usage() {
    cat <<'EOF'
Usage:
  tests/v2/e2e/server/test_server_e2e.sh [--binary <path>] [--model <path>] [--backends <list>]
  tests/v2/e2e/server/test_server_e2e.sh --container-image <image> [--model <path>] [--backends <list>]
  tests/v2/e2e/server/test_server_e2e.sh --suite "model|backend1,backend2[|max_tokens[|extra_flags[|label[|suite_options]]]]"

Container options:
  --container                         Use llaminar:local unless LLAMINAR_E2E_CONTAINER_IMAGE is set
  --container-image, --docker-image   Run the server from this image
  --docker-network                    auto|host|bridge|container:<id> (default: auto)
  --docker-gpus                       auto|all|none|<docker --gpus value> (default: auto)
  --docker-arg                        Additional docker run argument; repeatable

Environment:
  LLAMINAR_E2E_CONTAINER_IMAGE        Docker image to run instead of the local binary
  LLAMINAR_E2E_DOCKER_NETWORK         Docker network mode (default: auto)
  LLAMINAR_E2E_DOCKER_GPUS            GPU flag for docker run (default: auto)
  LLAMINAR_E2E_DOCKER_USER            docker run --user value (default: 0:0)
  LLAMINAR_E2E_DOCKER_NUMA_SECCOMP    Add seccomp=unconfined for NUMA syscalls (default: 1)
  LLAMINAR_E2E_DOCKER_ARGS            Extra docker run args, shell-split
  LLAMINAR_MODEL                      Override default suite model
  LLAMINAR_BACKENDS                   Override default suite backends
  LLAMINAR_E2E_LONG_CONTEXT=1         Enable optional long-context checks
EOF
}

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) show_usage; exit 0 ;;
        --binary)   BINARY="$2";            shift 2 ;;
        --container) SERVER_MODE="docker"; CONTAINER_IMAGE="${CONTAINER_IMAGE:-llaminar:local}"; shift ;;
        --container-image|--docker-image)
                    CONTAINER_IMAGE="$2"; SERVER_MODE="docker"; shift 2 ;;
        --server-mode)
                    SERVER_MODE="$2";       shift 2 ;;
        --docker-network)
                    DOCKER_NETWORK="$2";    shift 2 ;;
        --docker-gpus)
                    DOCKER_GPUS="$2";       shift 2 ;;
        --docker-arg)
                    DOCKER_EXTRA_ARGS+=("$2"); shift 2 ;;
        --model)    OVERRIDE_MODEL="$2";    shift 2 ;;
        --backends) OVERRIDE_BACKENDS="$2"; shift 2 ;;
        --suite)    SUITES+=("$2");         shift 2 ;;
        --port)     BASE_PORT="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build suite list — if no explicit --suite flags, use defaults
if [ ${#SUITES[@]} -eq 0 ]; then
    # Suite 1: Qwen2.5 (small, fast — all backends)
    S1_MODEL="${OVERRIDE_MODEL:-${LLAMINAR_MODEL:-${REPO_ROOT}/models/qwen2.5-1.5b-instruct-q8_0.gguf}}"
    S1_BACKENDS="${OVERRIDE_BACKENDS:-${LLAMINAR_BACKENDS:-cpu,cuda:0,rocm:0}}"
    SUITES+=("${S1_MODEL}|${S1_BACKENDS}")
    S1_GRAPH_MODEL="${REPO_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf"
    if [ ! -f "$S1_GRAPH_MODEL" ]; then
        S1_GRAPH_MODEL="$S1_MODEL"
    fi
    if [ -f "$S1_GRAPH_MODEL" ] && [ -z "$OVERRIDE_MODEL" ] && [ -z "$OVERRIDE_BACKENDS" ] &&
       [ -z "${LLAMINAR_MODEL:-}" ] && [ -z "${LLAMINAR_BACKENDS:-}" ]; then
        SUITES+=("${S1_GRAPH_MODEL}|cuda:0|16||qwen25-cuda-prefill-graph-probe|prefill-graph-probe")
    fi

    # Suite 2: Qwen3.5 4B (hybrid GDN/FA architecture — all backends)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    S2_MODEL="${REPO_ROOT}/models/Qwen3.5-4B-Q8_0.gguf"
    if [ -f "$S2_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S2_MODEL}|cpu,cuda:0,rocm:0|200")
    fi

    # Suite 3: Qwen3.5 35B MoE (MoE + GDN/FA architecture — CPU only)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    S3_MODEL="${REPO_ROOT}/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    if [ -f "$S3_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S3_MODEL}|cpu|200")
    fi

    # Suite 4: Qwen3.5 35B MoE on ROCm (GPU MoE inference, graph capture)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    S4_MODEL="${REPO_ROOT}/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    if [ -f "$S4_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S4_MODEL}|rocm:0|200")
    fi

    # Suite 4b: Qwen3.5 35B MoE Q3_K_S on CUDA (single-device CUDA MoE path)
    # This is the proving model for CUDA MoE enablement and mirrors the parity
    # configuration that regenerates its own Q3_K_S PyTorch snapshots.
    S4B_MODEL="/opt/llaminar-models/Qwen3.5-35B-A3B-Q3_K_S.gguf"
    if [ -f "$S4B_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S4B_MODEL}|cuda:0|200")
    fi

    # Suite 5: Qwen3.5 27B dense (hybrid GDN/FA architecture — all backends)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    S5_MODEL="/opt/llaminar-models/Qwen3.5-27B-Q4_K_M.gguf"
    if [ -f "$S5_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S5_MODEL}|cpu,rocm:0|200")
    fi

    # Suite 6: Qwen3.5 27B dense TP2 on ROCm (tensor parallel across 2 GPUs)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    S6_MODEL="/opt/llaminar-models/Qwen3.5-27B-Q4_K_M.gguf"
    if [ -f "$S6_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S6_MODEL}|tp|200|--tp-devices rocm:0,rocm:1|qwen35-dense-rocm2tp|no-prefill-graph-buckets")
    fi

    # Suite 7: Qwen3.5 27B dense PP2 (pipeline parallel: cuda:0 + rocm:0, equal layer split)
    # 64 layers total: layers 0-31 on cuda:0, layers 32-63 on rocm:0.
    # Uses max_tokens=200 because Qwen3.5 is a thinking model.
    S7_MODEL="/opt/llaminar-models/Qwen3.5-27B-Q4_K_M.gguf"
    if [ -f "$S7_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S7_MODEL}|pp|200|--define-domain cuda_pp=cuda:0 --define-domain rocm_pp=rocm:0 --pp-stage 0=cuda_pp:0-31 --pp-stage 1=rocm_pp:32-63|qwen35-dense-localpp-cuda-rocm|no-prefill-graph-buckets")
    fi

    # Suite 8: Qwen3.6 27B dense baseline and feature server cases.
    # These are the current vLLM-style MTP target model family. Include
    # `cpu` explicitly: in Llaminar device selection, -d cpu exercises all
    # CPU sockets as a NodeLocal CPU domain rather than pinning one socket.
    S8_MODEL="/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf"
    S8_PREFIX_FLAGS="--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto"
    S8_MTP_FLAGS="--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy"
    if [ -f "$S8_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S8_MODEL}|cpu,cuda:0,rocm:0|200||qwen36-dense-baseline")
        SUITES+=("${S8_MODEL}|cpu,cuda:0,rocm:0|200|${S8_PREFIX_FLAGS}|qwen36-dense-prefix-ram|no-long-context")
        SUITES+=("${S8_MODEL}|cpu,cuda:0,rocm:0|200|${S8_MTP_FLAGS}|qwen36-dense-mtp-greedy-d2|no-long-context")
    fi

    # Suite 9: Qwen3.6 35B-A3B MoE baseline and feature server cases.
    # Prefix cache uses the placement fingerprint policy so routed-expert
    # placement becomes part of the restore contract.
    S9_MODEL="/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
    S9_PREFIX_FLAGS="${S8_PREFIX_FLAGS} --prefix-cache-moe-policy placement-fingerprint"
    S9_MTP_FLAGS="${S8_MTP_FLAGS}"
    S9_TP_CUDA2_FLAGS="--tp-devices cuda:0,cuda:1"
    S9_TP_ROCM2_FLAGS="--tp-devices rocm:0,rocm:1"
    S9_TP_ROCM4_FLAGS="--tp-devices rocm:0,rocm:1,rocm:2,rocm:3"
    # Remote NodeLocal CPU-cold ExpertOverlay shapes are the production target,
    # but they must not run in the default gate until non-root participant ranks
    # execute matched MPI sparse dispatch/local-expert/return-reduce stages.
    S9_OVERLAY_ROCM2_CPU2_FLAGS="--moe-expert-overlay tiered --moe-expert-overlay-continuation qwen36_moe_rocm_hot --moe-expert-overlay-base-domain qwen36_moe_rocm_hot --moe-expert-overlay-shared-domain qwen36_moe_rocm_hot --moe-expert-overlay-residency static-by-id --moe-expert-overlay-domain qwen36_moe_rocm_hot=rocm:0,rocm:1;scope=local;backend=rccl;compute=replicated_experts;owner=0 --moe-expert-overlay-domain qwen36_moe_cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=replicated_experts;ranks=0,1 --moe-expert-overlay-tier hot@qwen36_moe_rocm_hot;priority=0;max-experts-per-layer=240;memory-mb=4096 --moe-expert-overlay-tier cold@qwen36_moe_cpu_cold;priority=1;max-experts-per-layer=0;memory-mb=0;fallback=true"
    S9_OVERLAY_CUDA2_CPU2_FLAGS="--moe-expert-overlay tiered --moe-expert-overlay-continuation qwen36_moe_cuda_hot --moe-expert-overlay-base-domain qwen36_moe_cuda_hot --moe-expert-overlay-shared-domain qwen36_moe_cuda_hot --moe-expert-overlay-residency static-by-id --moe-expert-overlay-domain qwen36_moe_cuda_hot=cuda:0,cuda:1;scope=local;backend=nccl;compute=replicated_experts;owner=0 --moe-expert-overlay-domain qwen36_moe_cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=replicated_experts;ranks=0,1 --moe-expert-overlay-tier hot@qwen36_moe_cuda_hot;priority=0;max-experts-per-layer=240;memory-mb=4096 --moe-expert-overlay-tier cold@qwen36_moe_cpu_cold;priority=1;max-experts-per-layer=0;memory-mb=0;fallback=true"
    S9_OVERLAY_CUDA2_ROCM2_CPU2_FLAGS="--moe-expert-overlay tiered --moe-expert-overlay-continuation qwen36_moe_cuda_hot --moe-expert-overlay-base-domain qwen36_moe_cuda_hot --moe-expert-overlay-shared-domain qwen36_moe_cuda_hot --moe-expert-overlay-residency static-by-id --moe-expert-overlay-domain qwen36_moe_cuda_hot=cuda:0,cuda:1;scope=local;backend=nccl;compute=replicated_experts;owner=0 --moe-expert-overlay-domain qwen36_moe_rocm_warm=rocm:0,rocm:1;scope=local;backend=rccl;compute=replicated_experts;owner=0 --moe-expert-overlay-domain qwen36_moe_cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=replicated_experts;ranks=0,1 --moe-expert-overlay-tier hot@qwen36_moe_cuda_hot;priority=0;max-experts-per-layer=192;memory-mb=4096 --moe-expert-overlay-tier warm@qwen36_moe_rocm_warm;priority=1;max-experts-per-layer=64;memory-mb=4096 --moe-expert-overlay-tier cold@qwen36_moe_cpu_cold;priority=2;max-experts-per-layer=0;memory-mb=0;fallback=true"
    if [ -f "$S9_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S9_MODEL}|cpu,cuda:0,rocm:0|200||qwen36-moe-baseline")
        SUITES+=("${S9_MODEL}|cpu,cuda:0,rocm:0|200|${S9_PREFIX_FLAGS}|qwen36-moe-prefix-ram|no-long-context")
        SUITES+=("${S9_MODEL}|cpu,cuda:0,rocm:0|200|${S9_MTP_FLAGS}|qwen36-moe-mtp-greedy-d2|no-long-context")
        SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_TP_CUDA2_FLAGS}|qwen36-moe-prefix-ram-cuda2tp|no-long-context,no-prefill-graph-buckets")
        SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_TP_CUDA2_FLAGS}|qwen36-moe-mtp-greedy-d2-cuda2tp|no-long-context,no-prefill-graph-buckets")
        SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_TP_ROCM2_FLAGS}|qwen36-moe-prefix-ram-rocm2tp|no-long-context,no-prefill-graph-buckets")
        SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_TP_ROCM2_FLAGS}|qwen36-moe-mtp-greedy-d2-rocm2tp|no-long-context,no-prefill-graph-buckets")
        SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_TP_ROCM4_FLAGS}|qwen36-moe-prefix-ram-rocm4tp|no-long-context,no-prefill-graph-buckets")
        SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_TP_ROCM4_FLAGS}|qwen36-moe-mtp-greedy-d2-rocm4tp|no-long-context,no-prefill-graph-buckets")
        if [[ "$REMOTE_EXPERT_OVERLAY_E2E" == "1" ]]; then
            SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_OVERLAY_ROCM2_CPU2_FLAGS}|qwen36-moe-prefix-ram-expertoverlay-rocm2-cpu2|no-long-context,no-prefill-graph-buckets")
            SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_OVERLAY_ROCM2_CPU2_FLAGS}|qwen36-moe-mtp-greedy-d2-expertoverlay-rocm2-cpu2|no-long-context,no-prefill-graph-buckets")
            SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_OVERLAY_CUDA2_CPU2_FLAGS}|qwen36-moe-prefix-ram-expertoverlay-cuda2-cpu2|no-long-context,no-prefill-graph-buckets")
            SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_OVERLAY_CUDA2_CPU2_FLAGS}|qwen36-moe-mtp-greedy-d2-expertoverlay-cuda2-cpu2|no-long-context,no-prefill-graph-buckets")
            SUITES+=("${S9_MODEL}|tp|200|${S9_PREFIX_FLAGS} ${S9_OVERLAY_CUDA2_ROCM2_CPU2_FLAGS}|qwen36-moe-prefix-ram-expertoverlay-cuda2-rocm2-cpu2|no-long-context,no-prefill-graph-buckets")
            SUITES+=("${S9_MODEL}|tp|200|${S9_MTP_FLAGS} ${S9_OVERLAY_CUDA2_ROCM2_CPU2_FLAGS}|qwen36-moe-mtp-greedy-d2-expertoverlay-cuda2-rocm2-cpu2|no-long-context,no-prefill-graph-buckets")
        fi
    fi
fi

STARTUP_TIMEOUT=300   # seconds to wait for server startup. Most models load in
                      # <10s; the 4B Qwen3.5 GGUF on CPU needs ~60-120s for
                      # weight load + GDN init. The smaller suites finish in
                      # ~5s either way, so this is just an upper bound.
SHUTDOWN_TIMEOUT=15   # seconds to wait for graceful SIGTERM shutdown
REQUEST_TIMEOUT=180   # seconds per curl request

# Optional long-context helper controls. The helper is intentionally gated to
# 4B+ models by default so small smoke-test suites keep their fast behavior.
LONG_CONTEXT_ENABLED="${LLAMINAR_E2E_LONG_CONTEXT:-0}"
LONG_CONTEXT_TIER="${LLAMINAR_E2E_LONG_CONTEXT_TIER:-full}"
CONTEXT_LENGTH="${LLAMINAR_E2E_CONTEXT_LENGTH:-4096}"
LONG_MAX_TOKENS="${LLAMINAR_E2E_LONG_MAX_TOKENS:-1024}"
LONG_MIN_PROMPT_TOKENS="${LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS:-900}"
LONG_REQUEST_TIMEOUT="${LLAMINAR_E2E_LONG_REQUEST_TIMEOUT:-$REQUEST_TIMEOUT}"
LONG_MIN_MODEL_SIZE_B="${LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B:-4}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─── Helpers ──────────────────────────────────────────────────────────────────
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_DETAILS=""

pass() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    PASSED_TESTS=$((PASSED_TESTS + 1))
    echo -e "  ${GREEN}✓${NC} $1"
}

fail() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    FAILED_TESTS=$((FAILED_TESTS + 1))
    FAILED_DETAILS="${FAILED_DETAILS}\n  - $1"
    echo -e "  ${RED}✗${NC} $1"
}

declare -a ACTIVE_DOCKER_CONTAINERS=()
declare -a ACTIVE_LOG_FOLLOW_PIDS=()

sanitize_name() {
    echo "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

is_docker_mode() {
    [[ "$SERVER_MODE" == "docker" ]]
}

docker_handle_container() {
    local handle="$1"
    if [[ "$handle" == docker:* ]]; then
        local rest="${handle#docker:}"
        echo "${rest%%:*}"
    fi
}

docker_handle_log_pid() {
    local handle="$1"
    if [[ "$handle" == docker:*:* ]]; then
        echo "${handle##*:}"
    fi
}

pid_handle_pid() {
    local handle="$1"
    if [[ "$handle" == pid:* ]]; then
        echo "${handle#pid:}"
    else
        echo "$handle"
    fi
}

cleanup_active_docker_containers() {
    local container log_pid
    for container in "${ACTIVE_DOCKER_CONTAINERS[@]:-}"; do
        [[ -n "$container" ]] || continue
        docker rm -f "$container" >/dev/null 2>&1 || true
    done
    for log_pid in "${ACTIVE_LOG_FOLLOW_PIDS[@]:-}"; do
        [[ -n "$log_pid" ]] || continue
        kill "$log_pid" >/dev/null 2>&1 || true
        wait "$log_pid" >/dev/null 2>&1 || true
    done
}

trap cleanup_active_docker_containers EXIT INT TERM

resolve_docker_network() {
    if [[ "$DOCKER_NETWORK" != "auto" ]]; then
        echo "$DOCKER_NETWORK"
        return
    fi

    if [[ -f /.dockerenv ]] && docker inspect "$(hostname)" >/dev/null 2>&1; then
        echo "container:$(hostname)"
    else
        echo "host"
    fi
}

docker_supports_nvidia_gpus() {
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' && return 0

    # Some Docker installations expose NVIDIA GPU support through CDI or the
    # default runtime without listing a runtime named "nvidia". Probe the actual
    # run path so devcontainers do not incorrectly fall back to manual mounts.
    [[ -n "${CONTAINER_IMAGE:-}" ]] || return 1
    docker run --rm --gpus all --entrypoint /bin/true "$CONTAINER_IMAGE" >/dev/null 2>&1
}

docker_args_need_cuda() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            cuda:*|*cuda:*) return 0 ;;
        esac
    done
    return 1
}

nvidia_driver_lib_path() {
    local lib="$1"
    local path
    path="$(ldconfig -p 2>/dev/null | awk -v lib="$lib" '$1 == lib { print $NF; exit }')"
    if [[ -z "$path" && -e "/usr/lib/x86_64-linux-gnu/${lib}" ]]; then
        path="/usr/lib/x86_64-linux-gnu/${lib}"
    fi
    printf '%s' "$path"
}

nvidia_driver_libs_available() {
    local lib path
    for lib in libcuda.so.1 libnvidia-ml.so.1; do
        path="$(nvidia_driver_lib_path "$lib")"
        [[ -n "$path" && -e "$path" ]] || return 1
    done
}

append_nvidia_driver_lib_mounts() {
    local -n out_ref="$1"
    local lib path
    for lib in libcuda.so.1 libnvidia-ml.so.1; do
        path="$(nvidia_driver_lib_path "$lib")"
        [[ -n "$path" && -e "$path" ]] || return 1
        out_ref+=(-v "${path}:/usr/lib/x86_64-linux-gnu/${lib}:ro")
    done
}

nvidia_device_nodes_available() {
    [[ -e /dev/nvidiactl ]] || return 1
    local node
    for node in /dev/nvidia[0-9]*; do
        [[ -e "$node" ]] && return 0
    done
    return 1
}

append_existing_device_path() {
    local -n out_ref="$1"
    local path="$2"
    [[ -e "$path" ]] || return
    out_ref+=(--device="$path")
}

append_nvidia_device_nodes() {
    local -n out_ref="$1"
    local node
    for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
        [[ -e "$node" ]] && out_ref+=(--device="$node")
    done
    for node in /dev/nvidia[0-9]* /dev/nvidia-caps/*; do
        [[ -e "$node" ]] && out_ref+=(--device="$node")
    done
}

append_unique_group_for_path() {
    local -n out_ref="$1"
    local path="$2"
    [[ -e "$path" ]] || return
    local gid
    gid="$(stat -c '%g' "$path" 2>/dev/null || true)"
    [[ -n "$gid" ]] || return
    out_ref+=(--group-add "$gid")
}

container_model_path() {
    local model="$1"
    readlink -f "$model" 2>/dev/null || echo "$model"
}

parse_memory_to_mb() {
    local value="$1"
    python3 - "$value" <<'PY'
import re
import sys

text = sys.argv[1].split("/", 1)[0].strip()
match = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?", text)
if not match:
    print(-1)
    sys.exit(0)

amount = float(match.group(1))
unit = (match.group(2) or "b").lower()
scale = {
    "b": 1 / 1048576,
    "kb": 1 / 1024,
    "kib": 1 / 1024,
    "mb": 1,
    "mib": 1,
    "gb": 1024,
    "gib": 1024,
    "tb": 1024 * 1024,
    "tib": 1024 * 1024,
}.get(unit, 1)
print(max(0, int(amount * scale)))
PY
}

get_container_ram_mb() {
    local container="$1"
    local usage
    usage="$(docker stats --no-stream --format '{{.MemUsage}}' "$container" 2>/dev/null || true)"
    if [[ -z "$usage" ]]; then
        echo -1
        return
    fi
    parse_memory_to_mb "$usage"
}

get_container_pids() {
    local container="$1"
    docker top "$container" -eo pid 2>/dev/null | awk 'NR > 1 {print $1}' | xargs || true
}

signal_container_llaminar_processes() {
    local container="$1"
    local signal_name="${2:-TERM}"

    docker exec "$container" /bin/sh -lc '
set -eu
signal_name="$1"
sent=0
fallback_pid=""
for comm in /proc/[0-9]*/comm; do
    [ -r "$comm" ] || continue
    [ "$(cat "$comm" 2>/dev/null || true)" = "llaminar2" ] || continue
    pid="${comm%/comm}"
    pid="${pid##*/}"
    [ -n "$fallback_pid" ] || fallback_pid="$pid"
    if tr "\000" "\n" <"/proc/${pid}/environ" 2>/dev/null | grep -qx "OMPI_COMM_WORLD_RANK=0"; then
        kill "-${signal_name}" "$pid" >/dev/null 2>&1 && sent=1 || true
    fi
done
if [ "$sent" != 1 ] && [ -n "$fallback_pid" ]; then
    kill "-${signal_name}" "$fallback_pid" >/dev/null 2>&1 && sent=1 || true
fi
[ "$sent" = 1 ]
' _ "$signal_name" >/dev/null 2>&1
}

server_is_alive() {
    local handle="$1"
    if [[ "$handle" == docker:* ]]; then
        local container
        container="$(docker_handle_container "$handle")"
        [[ "$(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null || echo false)" == "true" ]]
        return
    fi

    local pid
    pid="$(pid_handle_pid "$handle")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

get_server_pids() {
    local handle="$1"
    if [[ "$handle" == docker:* ]]; then
        get_container_pids "$(docker_handle_container "$handle")"
        return
    fi

    collect_process_tree_pids "$(pid_handle_pid "$handle")" | sort -n | uniq | xargs || true
}

get_server_ram_mb() {
    local handle="$1"
    local pids="$2"
    if [[ "$handle" == docker:* ]]; then
        get_container_ram_mb "$(docker_handle_container "$handle")"
        return
    fi

    get_process_tree_ram_mb "$pids"
}

start_server_process() {
    local tag="$1"
    local port="$2"
    local host_model="$3"
    local log_path="$4"
    local safe_tag="$5"
    local -n env_ref="$6"
    local -n args_ref="$7"

    STARTED_SERVER_HANDLE=""
    if ! is_docker_mode; then
        env "${env_ref[@]}" "$BINARY" "${args_ref[@]}" >"$log_path" 2>&1 &
        STARTED_SERVER_HANDLE="pid:$!"
        return
    fi

    local model_abs model_dir log_dir_abs network_mode container_name container_id log_pid
    model_abs="$(container_model_path "$host_model")"
    model_dir="$(dirname "$model_abs")"
    log_dir_abs="$(readlink -f "$LOG_DIR" 2>/dev/null || echo "$LOG_DIR")"
    network_mode="$(resolve_docker_network)"
    container_name="${DOCKER_NAME_PREFIX}-${safe_tag}-port${port}-$$"
    container_name="$(echo "$container_name" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_.-')"

    docker rm -f "$container_name" >/dev/null 2>&1 || true

    local -a docker_args=(
        run -d
        --name "$container_name"
        --network "$network_mode"
        --ipc=host
        --shm-size="$DOCKER_SHM_SIZE"
        --ulimit core=-1
        -v "${model_dir}:${model_dir}:ro"
        -v "${log_dir_abs}:${log_dir_abs}"
    )
    if [[ "$DOCKER_NUMA_SECCOMP" != "0" ]]; then
        docker_args+=(--security-opt seccomp=unconfined)
    fi
    if [[ -n "$DOCKER_USER" ]]; then
        docker_args+=(--user "$DOCKER_USER")
    fi
    local nvidia_mode="none"
    local needs_nvidia="0"
    if docker_args_need_cuda "${args_ref[@]}"; then
        needs_nvidia="1"
    fi

    if [[ "$network_mode" == "bridge" ]]; then
        docker_args+=(-p "127.0.0.1:${port}:${port}")
    fi

    case "$DOCKER_GPUS" in
        all)
            if docker_supports_nvidia_gpus; then
                docker_args+=(--gpus all)
                nvidia_mode="runtime"
            elif nvidia_driver_libs_available; then
                append_nvidia_driver_lib_mounts docker_args
                if nvidia_device_nodes_available; then
                    append_nvidia_device_nodes docker_args
                    nvidia_mode="manual-devices"
                else
                    nvidia_mode="manual-libs"
                fi
            fi
            ;;
        auto)
            if [[ "$needs_nvidia" != "1" ]]; then
                nvidia_mode="not-needed"
            elif docker_supports_nvidia_gpus; then
                docker_args+=(--gpus all)
                nvidia_mode="runtime"
            elif nvidia_driver_libs_available; then
                append_nvidia_driver_lib_mounts docker_args
                if nvidia_device_nodes_available; then
                    append_nvidia_device_nodes docker_args
                    nvidia_mode="manual-devices"
                else
                    nvidia_mode="manual-libs"
                fi
            fi
            ;;
        none|"")
            nvidia_mode="none"
            ;;
        *)
            if docker_supports_nvidia_gpus; then
                docker_args+=(--gpus "$DOCKER_GPUS")
                nvidia_mode="runtime"
            elif nvidia_driver_libs_available; then
                append_nvidia_driver_lib_mounts docker_args
                if nvidia_device_nodes_available; then
                    append_nvidia_device_nodes docker_args
                    nvidia_mode="manual-devices"
                else
                    nvidia_mode="manual-libs"
                fi
            fi
            ;;
    esac

    if [[ -e /dev/kfd ]]; then
        docker_args+=(--device=/dev/kfd)
        append_unique_group_for_path docker_args /dev/kfd
    fi
    if [[ -e /dev/dri ]]; then
        docker_args+=(--device=/dev/dri)
        append_unique_group_for_path docker_args /dev/dri
        for render_node in /dev/dri/render*; do
            [[ -e "$render_node" ]] || continue
            append_unique_group_for_path docker_args "$render_node"
        done
    fi

    docker_args+=("${DOCKER_EXTRA_ARGS[@]}")
    for env_pair in "${env_ref[@]}"; do
        docker_args+=(-e "$env_pair")
    done
    docker_args+=("$CONTAINER_IMAGE" "${args_ref[@]}")

    echo -e "  ${BLUE}INFO${NC} [${tag}] Docker: image=${CONTAINER_IMAGE}, network=${network_mode}, nvidia=${nvidia_mode}, numa_seccomp=${DOCKER_NUMA_SECCOMP}, model=${model_abs}" >&2
    container_id="$(docker "${docker_args[@]}")"
    ACTIVE_DOCKER_CONTAINERS+=("$container_id")

    docker logs -f "$container_id" >"$log_path" 2>&1 &
    log_pid=$!
    ACTIVE_LOG_FOLLOW_PIDS+=("$log_pid")

    STARTED_SERVER_HANDLE="docker:${container_id}:${log_pid}"
}

is_thinking_model() {
    local label="$1"
    [[ "${label,,}" == *"qwen3.5"* || "${label,,}" == *"qwen35"* ||
       "${label,,}" == *"qwen3.6"* || "${label,,}" == *"qwen36"* ]]
}

is_prefix_cache_case() {
    local extra_flags="$1"
    [[ " ${extra_flags} " == *" --prefix-cache "* ]]
}

is_mtp_case() {
    local extra_flags="$1"
    [[ " ${extra_flags} " == *" --mtp "* ]]
}

suite_disables_long_context() {
    local suite_options="$1"
    [[ ",${suite_options}," == *",no-long-context,"* ]]
}

suite_disables_prefill_graph_buckets() {
    local suite_options="$1"
    [[ ",${suite_options}," == *",no-prefill-graph-buckets,"* ]]
}

suite_runs_prefill_graph_probe() {
    local suite_options="$1"
    [[ ",${suite_options}," == *",prefill-graph-probe,"* ]]
}

is_gpu_backend() {
    local backend="$1"
    [[ "$backend" == cuda:* || "$backend" == rocm:* || "$backend" == "tp" || "$backend" == "pp" ]]
}

parse_model_size_b() {
    local model="$1"
    local label="$2"

    python3 - "$model" "$label" <<'PY'
import os
import re
import sys

model = sys.argv[1]
label = sys.argv[2]
text = " ".join([model, os.path.basename(model), label])
sizes = [float(match.group(1)) for match in re.finditer(r"(?i)(?<![0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9.])", text)]
if sizes:
    print(f"{max(sizes):g}")
PY
}

model_size_meets_threshold() {
    local size_b="$1"
    local threshold_b="$2"

    python3 - "$size_b" "$threshold_b" <<'PY'
import sys

try:
    size_b = float(sys.argv[1])
    threshold_b = float(sys.argv[2])
except ValueError:
    sys.exit(1)

sys.exit(0 if size_b >= threshold_b else 1)
PY
}

print_long_context_gate() {
    local tag="$1"
    local model_size_b="$2"

    if [ -z "$model_size_b" ]; then
        echo -e "  ${YELLOW}SKIP${NC} [${tag}] Long-context: no model size parsed from path/label; require >= ${LONG_MIN_MODEL_SIZE_B}B"
    else
        echo -e "  ${YELLOW}SKIP${NC} [${tag}] Long-context: parsed model size ${model_size_b}B below ${LONG_MIN_MODEL_SIZE_B}B threshold"
    fi
}

run_long_context_checks() {
    local tag="$1"
    local port="$2"
    local thinking_model="$3"

    if LLAMINAR_E2E_LONG_CONTEXT_ARTIFACT_DIR="$LOG_DIR" \
        python3 "$SCRIPT_DIR/long_context_checks.py" \
        --base-url "http://127.0.0.1:${port}" \
        --tag "$tag" \
        --tier "$LONG_CONTEXT_TIER" \
        --min-prompt-tokens "$LONG_MIN_PROMPT_TOKENS" \
        --long-max-tokens "$LONG_MAX_TOKENS" \
        --context-length "$CONTEXT_LENGTH" \
        --request-timeout "$LONG_REQUEST_TIMEOUT" \
        --thinking-model "$thinking_model"; then
        pass "[${tag}] Long-context checks (${LONG_CONTEXT_TIER})"
    else
        fail "[${tag}] Long-context checks (${LONG_CONTEXT_TIER})"
    fi
}

mode_name() {
    if [ "$1" = "true" ]; then
        echo "thinking"
    else
        echo "non-thinking"
    fi
}

preview_text() {
    python3 -c "
import sys
text = sys.stdin.read().replace('\n', '\\n')
if len(text) > 120:
    text = text[:117] + '...'
print(text)
"
}

cleanup_server() {
    local handle=$1
    if [[ "$handle" == docker:* ]]; then
        local container log_pid
        container="$(docker_handle_container "$handle")"
        log_pid="$(docker_handle_log_pid "$handle")"
        docker rm -f "$container" >/dev/null 2>&1 || true
        if [[ -n "$log_pid" ]]; then
            kill "$log_pid" 2>/dev/null || true
            wait "$log_pid" 2>/dev/null || true
        fi
        return
    fi

    local pid
    pid="$(pid_handle_pid "$handle")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

copy_container_artifact() {
    local handle="$1"
    local container_path="$2"
    local host_path="$3"

    [[ "$handle" == docker:* ]] || return 0
    [[ -n "$container_path" && -n "$host_path" ]] || return 0
    [[ -s "$host_path" ]] && return 0

    local container
    container="$(docker_handle_container "$handle")"
    mkdir -p "$(dirname "$host_path")"
    docker cp "${container}:${container_path}" "$host_path" >/dev/null 2>&1 || true
}

# Graceful shutdown with exit code, VRAM release, and crash validation.
# Called at the end of each test case instead of bare cleanup_server.
shutdown_and_validate() {
    local tag="$1"
    local handle="$2"
    local gpu_before_mb="$3"

    # ─── Check 1: Clean SIGTERM exit ─────────────────────────────────
    local exit_code=0
    if [[ "$handle" == docker:* ]]; then
        local container log_pid
        container="$(docker_handle_container "$handle")"
        log_pid="$(docker_handle_log_pid "$handle")"

        if server_is_alive "$handle"; then
            signal_container_llaminar_processes "$container" TERM || true
            local deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
            while server_is_alive "$handle" && [ $SECONDS -lt $deadline ]; do
                sleep 0.2
            done
        fi

        if server_is_alive "$handle"; then
            if ! docker stop --time "$SHUTDOWN_TIMEOUT" "$container" >/dev/null 2>&1; then
                fail "[${tag}] Shutdown: container did not stop within ${SHUTDOWN_TIMEOUT}s after SIGTERM, removing forcefully"
                docker rm -f "$container" >/dev/null 2>&1 || true
                if [[ -n "$log_pid" ]]; then
                    kill "$log_pid" 2>/dev/null || true
                    wait "$log_pid" 2>/dev/null || true
                fi
                return
            fi
        fi

        exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container" 2>/dev/null || echo 0)"
        if [[ -n "$log_pid" ]]; then
            wait "$log_pid" 2>/dev/null || true
        fi
    else
        local pid
        pid="$(pid_handle_pid "$handle")"
        if kill -0 "$pid" 2>/dev/null; then
            # Kill the entire process tree (TP/PP spawns child ranks via mpirun)
            local tree_pids
            tree_pids=$(collect_process_tree_pids "$pid" | xargs || echo "$pid")
            kill $tree_pids 2>/dev/null || true

            # Wait with timeout — poll until root process exits or deadline
            local deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
            while kill -0 "$pid" 2>/dev/null && [ $SECONDS -lt $deadline ]; do
                sleep 0.2
            done

            if kill -0 "$pid" 2>/dev/null; then
                # Process didn't exit gracefully — force kill entire tree
                fail "[${tag}] Shutdown: process did not exit within ${SHUTDOWN_TIMEOUT}s after SIGTERM, sending SIGKILL"
                kill -9 $tree_pids 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
                return
            fi

            # Reap and get exit code (may be non-zero for SIGTERM, so suppress errexit)
            wait "$pid" 2>/dev/null && exit_code=0 || exit_code=$?
        else
            # Process already exited — get its status
            wait "$pid" 2>/dev/null && exit_code=0 || exit_code=$?
        fi
    fi

    # ─── Check 2: No crash/segfault on exit ──────────────────────────
    # exit_code meanings:
    #   0       = clean exit
    #   143     = killed by SIGTERM (128 + 15) — acceptable for servers
    #   139     = SIGSEGV (segfault)
    #   134     = SIGABRT (abort/assertion)
    #   136     = SIGFPE (floating point exception)
    #   132     = SIGILL (illegal instruction)
    local crash_signals="139 134 136 132"
    local is_crash=false
    for sig in $crash_signals; do
        if [ "$exit_code" -eq "$sig" ]; then
            is_crash=true
            break
        fi
    done

    if [ "$is_crash" = "true" ]; then
        local signal_name="unknown"
        case "$exit_code" in
            139) signal_name="SIGSEGV (segfault)" ;;
            134) signal_name="SIGABRT (abort)" ;;
            136) signal_name="SIGFPE" ;;
            132) signal_name="SIGILL" ;;
        esac
        fail "[${tag}] Shutdown: process crashed with ${signal_name} (exit code ${exit_code})"
    elif [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 1 ] || [ "$exit_code" -eq 143 ]; then
        # 0 = clean exit, 1 = mpirun wrapper reporting child signal (normal),
        # 143 = killed by SIGTERM directly (128+15)
        pass "[${tag}] Shutdown: clean exit (code ${exit_code})"
    else
        fail "[${tag}] Shutdown: unexpected exit code ${exit_code}"
    fi

    # Allow small variance (driver overhead, context caching) — 64 MiB tolerance
    local VRAM_LEAK_TOLERANCE_MB=64
    local gpu_after_mb gpu_leaked_mb release_deadline

    # ─── Check 3: GPU VRAM fully released ────────────────────────────
    # Driver teardown can lag process exit, especially on ROCm after large
    # models.  Poll for the same strict threshold instead of sampling once and
    # reporting a false leak while the driver is still releasing allocations.
    release_deadline=$((SECONDS + GPU_RELEASE_TIMEOUT_SECONDS))
    while true; do
        gpu_after_mb=$(get_total_gpu_memory_mb)
        gpu_leaked_mb=$((gpu_after_mb - gpu_before_mb))
        if [ "$gpu_leaked_mb" -le "$VRAM_LEAK_TOLERANCE_MB" ]; then
            break
        fi
        if [ $SECONDS -ge $release_deadline ]; then
            break
        fi
        sleep 1
    done

    if [ "$gpu_leaked_mb" -le "$VRAM_LEAK_TOLERANCE_MB" ]; then
        pass "[${tag}] Shutdown: GPU VRAM released (before=${gpu_before_mb} MiB, after=${gpu_after_mb} MiB, delta=${gpu_leaked_mb} MiB)"
    else
        fail "[${tag}] Shutdown: GPU VRAM leak detected (before=${gpu_before_mb} MiB, after=${gpu_after_mb} MiB, leaked=${gpu_leaked_mb} MiB)"
    fi
}

wait_for_health() {
    local port=$1
    local handle=${2:-}
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        # If we have a process/container handle, detect early exit / OOM.
        if [ -n "$handle" ] && ! server_is_alive "$handle"; then
            return 1  # Server process already exited
        fi
        if curl -s --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Extract content from chat completion JSON response
extract_content() {
    python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

extract_numeric_answer() {
    python3 -c "
import json, re, sys
try:
    data = json.load(sys.stdin)
    message = data.get('choices', [{}])[0].get('message', {})
    content = message.get('content') or ''
    matches = re.findall(r'-?\\d+', content)
    print(matches[-1] if matches else '')
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

extract_stream_content() {
    python3 -c "
import json, sys
content = []
try:
    for raw in sys.stdin.read().strip().split('\\n'):
        if not raw.startswith('data: ') or raw.strip() == 'data: [DONE]':
            continue
        chunk = json.loads(raw[6:])
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        if 'content' in delta and delta['content'] is not None:
            content.append(delta['content'])
    print(''.join(content))
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

extract_numeric_from_text() {
    python3 -c "
import re, sys
text = sys.stdin.read()
matches = re.findall(r'-?\\d+', text)
print(matches[-1] if matches else '')
"
}

make_chat_payload() {
    local messages_json="$1"
    local max_tokens="$2"
    local enable_thinking="$3"
    local stream="${4:-false}"

    python3 - "$messages_json" "$max_tokens" "$enable_thinking" "$stream" "$THINKING_BUDGET_TOKENS" <<'PY'
import json
import sys

messages = json.loads(sys.argv[1])
max_tokens = int(sys.argv[2])
enable_thinking = sys.argv[3] == "true"
stream = sys.argv[4] == "true"
thinking_budget = int(sys.argv[5]) if sys.argv[5] else -1

payload = {
    "messages": messages,
    "max_tokens": max_tokens,
    "enable_thinking": enable_thinking,
    "temperature": 0.0,
}
if stream:
    payload["stream"] = True
if enable_thinking and thinking_budget >= 0:
    payload["thinking_budget_tokens"] = thinking_budget

print(json.dumps(payload, separators=(",", ":")))
PY
}

validate_chat_response_format() {
    python3 -c "
import json, sys
d = json.load(sys.stdin)
u = d.get('usage', {})
assert u.get('prompt_tokens', 0) > 0
assert u.get('completion_tokens', 0) > 0
assert u.get('total_tokens', 0) == u['prompt_tokens'] + u['completion_tokens']
assert d.get('choices', [{}])[0].get('finish_reason') == 'stop'
print('ok')
" 2>/dev/null || echo "FAIL"
}

collect_process_tree_pids() {
    local root="$1"
    if [ -z "$root" ] || ! kill -0 "$root" 2>/dev/null; then
        return
    fi

    echo "$root"
    local child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        collect_process_tree_pids "$child"
    done
}

get_process_tree_ram_mb() {
    local pids="$1"
    local total_kb=0
    local pid ram_kb
    for pid in $pids; do
        if [ -r "/proc/${pid}/smaps_rollup" ]; then
            ram_kb=$(awk '/^Pss:/ {print $2}' "/proc/${pid}/smaps_rollup" 2>/dev/null || echo 0)
        elif [ -r "/proc/${pid}/status" ]; then
            ram_kb=$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0)
        else
            ram_kb=0
        fi
        total_kb=$((total_kb + ${ram_kb:-0}))
    done
    echo $(((total_kb + 1023) / 1024))
}

get_nvidia_total_gpu_mb() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi
    { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; } |
        awk '{gsub(/[^0-9]/, "", $1); if ($1 != "") sum += $1} END {print sum + 0}'
}

get_amd_total_gpu_mb() {
    if command -v amd-smi >/dev/null 2>&1; then
        { amd-smi metric --mem-usage --csv 2>/dev/null || true; } |
            awk -F',' 'NR > 1 && $3 ~ /^[0-9]+$/ {sum += $3} END {print sum + 0}'
        return
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        { rocm-smi --showmeminfo vram 2>/dev/null || true; } |
            awk -F': ' '/VRAM Total Used Memory/ {sum += int($2 / 1048576)} END {print sum + 0}'
        return
    fi

    echo 0
}

get_total_gpu_memory_mb() {
    local nvidia_mb amd_mb
    nvidia_mb=$(get_nvidia_total_gpu_mb)
    amd_mb=$(get_amd_total_gpu_mb)
    echo $((nvidia_mb + amd_mb))
}

get_nvidia_process_gpu_mb() {
    local pids="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi
    { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true; } |
        awk -F',' -v wanted="$pids" '
            BEGIN {
                split(wanted, arr, " ")
                for (i in arr) if (arr[i] != "") pid[arr[i]] = 1
            }
            {
                gsub(/ /, "", $1)
                gsub(/[^0-9]/, "", $2)
                if (($1 in pid) && $2 != "") sum += $2
            }
            END {print sum + 0}'
}

get_amd_process_gpu_mb() {
    local pids="$1"
    if ! command -v amd-smi >/dev/null 2>&1; then
        echo 0
        return
    fi

    local amd_json
    amd_json=$(amd-smi process --general --json 2>/dev/null || echo '[]')
    AMD_SMI_JSON="$amd_json" python3 - "$pids" <<'PY'
import json
import os
import re
import sys

wanted = {int(p) for p in sys.argv[1].split() if p.strip().isdigit()}
try:
    data = json.loads(os.environ.get("AMD_SMI_JSON", "[]"))
except Exception:
    print(0)
    sys.exit(0)

def parse_mib(value):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return 0
    amount = float(match.group(1))
    lower = text.lower()
    if "gib" in lower or "gb" in lower:
        amount *= 1024
    elif "kib" in lower or "kb" in lower:
        amount /= 1024
    elif "b" in lower and "mb" not in lower and "mib" not in lower:
        amount /= 1048576
    return int(amount)

def find_pid(obj):
    if not isinstance(obj, dict):
        return None
    for key, value in obj.items():
        normalized = key.lower().replace(" ", "_")
        if normalized in {"pid", "process_id", "processid"}:
            try:
                return int(str(value).split()[0])
            except Exception:
                return None
    return None

def walk(obj):
    total = 0
    if isinstance(obj, dict):
        pid = find_pid(obj)
        if pid in wanted:
            for key, value in obj.items():
                lower = key.lower()
                if ("mem" in lower or "vram" in lower or "gtt" in lower) and "total" not in lower and "free" not in lower:
                    total += parse_mib(value)
        for value in obj.values():
            total += walk(value)
    elif isinstance(obj, list):
        for value in obj:
            total += walk(value)
    return total

print(walk(data))
PY
}

get_process_tree_gpu_memory_mb() {
    local pids="$1"
    local nvidia_mb amd_mb
    nvidia_mb=$(get_nvidia_process_gpu_mb "$pids")
    amd_mb=$(get_amd_process_gpu_mb "$pids")
    echo $((nvidia_mb + amd_mb))
}

scan_server_log() {
    local tag="$1"
    local log_path="$2"
    local has_failure=false

    # Check 1: Application-level errors (our logging framework)
    local matches count
    matches=$(grep -nE '\[(WARN ?|ERROR|FATAL)\]' "$log_path" 2>/dev/null || true)

    if [ -n "$matches" ]; then
        count=$(printf '%s\n' "$matches" | wc -l | xargs)
        fail "[${tag}] Server log: ${count} WARN/ERROR entries in ${log_path}"
        echo -e "    ${RED}── WARN/ERROR lines ──${NC}"
        printf '%s\n' "$matches" | head -40 | sed 's/^/    /'
        if [ "$count" -gt 40 ]; then
            echo "    ... (${count} total matches; see ${log_path})"
        fi
        echo "    ─────────────────────"
        has_failure=true
    fi

    # Check 2: mpirun crash detection — this message only appears when the child
    # process exits non-zero ON ITS OWN (real error). It does NOT appear when we
    # SIGTERM mpirun externally (our normal shutdown path).
    local mpi_crash
    mpi_crash=$(grep -c "mpirun detected that one or more processes exited with non-zero status" "$log_path" 2>/dev/null || true)
    if [ "$mpi_crash" -gt 0 ]; then
        local mpi_exit_code
        mpi_exit_code=$(grep -oP 'Exit code:\s+\K\d+' "$log_path" 2>/dev/null | head -1 || echo "unknown")
        fail "[${tag}] Server log: mpirun detected child process crashed (exit code: ${mpi_exit_code})"
        has_failure=true
    fi

    if [ "$has_failure" = "false" ]; then
        pass "[${tag}] Server log: clean (${log_path})"
    fi
}

check_memory_usage() {
    local tag="$1"
    local backend="$2"
    local model="$3"
    local server_handle="$4"
    local gpu_before_mb="$5"

    local pids
    pids=$(get_server_pids "$server_handle")
    if [ -z "$pids" ] && [[ "$server_handle" != docker:* ]]; then
        fail "[${tag}] Memory: server process exited before measurement"
        return
    fi

    local ram_mb gpu_process_mb gpu_after_mb gpu_delta_mb abs_gpu_delta_mb model_mb rss_multiplier rss_limit_mb
    ram_mb=$(get_server_ram_mb "$server_handle" "$pids")
    if [ "$ram_mb" -lt 0 ]; then
        fail "[${tag}] Memory: unable to measure server RAM"
        return
    fi
    gpu_process_mb=$(get_process_tree_gpu_memory_mb "$pids")
    gpu_after_mb=$(get_total_gpu_memory_mb)
    gpu_delta_mb=$((gpu_after_mb - gpu_before_mb))
    abs_gpu_delta_mb=${gpu_delta_mb#-}
    model_mb=$(du -m "$model" 2>/dev/null | awk '{print $1}' || echo 0)

    if is_gpu_backend "$backend"; then
        rss_multiplier="$HOST_RSS_GPU_MODEL_MULTIPLIER"
    else
        rss_multiplier="$HOST_RSS_CPU_MODEL_MULTIPLIER"
    fi
    rss_limit_mb=$((model_mb * rss_multiplier + HOST_RSS_EXTRA_MB))

    if [ "$ram_mb" -le "$rss_limit_mb" ]; then
        pass "[${tag}] Memory: RAM ${ram_mb} MiB within limit ${rss_limit_mb} MiB"
    else
        fail "[${tag}] Memory: RAM ${ram_mb} MiB exceeds limit ${rss_limit_mb} MiB"
    fi

    if is_gpu_backend "$backend"; then
        if [ "$gpu_process_mb" -ge "$GPU_ACTIVE_MIN_MB" ] || [ "$abs_gpu_delta_mb" -ge "$GPU_ACTIVE_MIN_MB" ]; then
            pass "[${tag}] GPU memory: process ${gpu_process_mb} MiB, global delta ${gpu_delta_mb} MiB"
        else
            fail "[${tag}] GPU memory: expected active GPU usage, process ${gpu_process_mb} MiB, global delta ${gpu_delta_mb} MiB"
        fi
    else
        if [ "$gpu_process_mb" -le "$CPU_GPU_DELTA_LIMIT_MB" ] && [ "$abs_gpu_delta_mb" -le "$CPU_GPU_DELTA_LIMIT_MB" ]; then
            pass "[${tag}] GPU memory: CPU backend left GPU usage unchanged (process ${gpu_process_mb} MiB, global delta ${gpu_delta_mb} MiB)"
        else
            fail "[${tag}] GPU memory: CPU backend changed GPU usage (process ${gpu_process_mb} MiB, global delta ${gpu_delta_mb} MiB)"
        fi
    fi
}

validate_perf_stats() {
    local tag="$1"
    local backend="$2"
    local extra_flags="$3"
    local perf_path="$4"
    local long_context_run="${5:-false}"
    local suite_options="${6:-}"

    if [ "$PERF_STATS_ENABLED" != "1" ]; then
        return
    fi

    if [ ! -s "$perf_path" ]; then
        if is_mtp_case "$extra_flags" && is_gpu_backend "$backend"; then
            fail "[${tag}] PerfStats: missing artifact for GPU MTP case (${perf_path})"
        else
            echo -e "  ${YELLOW}SKIP${NC} [${tag}] PerfStats: no records emitted (${perf_path})"
        fi
        return
    fi

    local validation
    validation=$(python3 - "$perf_path" "$backend" "$extra_flags" "$long_context_run" "$suite_options" <<'PY'
import json
import sys

path, backend, extra_flags, long_context_run, suite_options = sys.argv[1:6]
is_gpu = backend.startswith(("cuda:", "rocm:")) or backend in {"tp", "pp"}
is_mtp = f" {extra_flags} ".find(" --mtp ") >= 0
suite_option_set = {
    option.strip()
    for option in suite_options.split(",")
    if option.strip()
}
expect_prefill_phase = (
    is_gpu
    and long_context_run == "true"
    and "no-prefill-graph-buckets" not in suite_option_set
)
require_prefill_capture = (
    "require-prefill-graph-capture" in suite_option_set
    or "prefill-graph-probe" in suite_option_set
)
require_prefill_replay = "prefill-graph-probe" in suite_option_set
expect_decode_replay = (
    is_mtp
    or long_context_run == "true"
    or "require-decode-graph-replay" in suite_option_set
)

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

records = data.get("records")
if not isinstance(records, list):
    print("FAIL: missing records array")
    sys.exit(0)

def has_record(name=None, domain=None, tags=None):
    tags = tags or {}
    for record in records:
        if name is not None and record.get("name") != name:
            continue
        if domain is not None and record.get("domain") != domain:
            continue
        record_tags = record.get("tags") or {}
        if all(record_tags.get(key) == value for key, value in tags.items()):
            return True
    return False

decode_graph_captured = (
    has_record("decode_graph_phase", "forward_graph", {"phase": "capture"})
    or has_record("decode_segmented_phase", "forward_graph", {"phase": "capture"})
)
decode_graph_replayed = (
    has_record("decode_graph_phase", "forward_graph", {"phase": "replay"})
    or has_record("decode_segmented_phase", "forward_graph", {"phase": "replay"})
)
decode_graph_explicitly_unsupported = has_record(
    "decode_capture_policy",
    "forward_graph",
    {
        "has_collectives": "true",
        "collectives_graph_capturable": "false",
    },
)

if is_mtp and not has_record(domain="mtp"):
    print("FAIL: MTP case emitted no mtp-domain counters")
    sys.exit(0)

if is_gpu:
    if not decode_graph_captured and not decode_graph_explicitly_unsupported:
        print("FAIL: GPU case emitted no decode graph capture counter")
        sys.exit(0)
    # Short arithmetic probes may finish immediately after warmup/capture,
    # especially on ROCm.  Require replay only for cases that deliberately
    # create enough decode work or explicitly ask for a replay proof.  LocalTP
    # and PP suites that include non-capturable collectives must emit an
    # explicit policy counter instead of quietly pretending graph replay was
    # available.
    if expect_decode_replay and not decode_graph_replayed and not decode_graph_explicitly_unsupported:
        print("FAIL: GPU case expected decode graph replay but emitted no replay counter")
        sys.exit(0)

if expect_prefill_phase:
    if not has_record("prefill_graph_phase", "forward_graph"):
        print("FAIL: eligible GPU long-context case emitted no prefill graph phase counters")
        sys.exit(0)

if require_prefill_capture:
    if not has_record("prefill_graph_phase", "forward_graph", {"capture_phase": "capture"}):
        print("FAIL: suite required prefill graph capture but no capture phase was recorded")
        sys.exit(0)

if require_prefill_replay:
    if not has_record("prefill_graph_phase", "forward_graph", {"capture_phase": "replay"}):
        print("FAIL: suite required prefill graph replay but no replay phase was recorded")
        sys.exit(0)

if is_gpu and is_mtp:
    if not (
        has_record("sidecar_graph_cache_hits", "mtp")
        or has_record("sidecar_graph_cache_misses", "mtp")
        or has_record("sidecar_graph_capture_path", "mtp")
    ):
        print("FAIL: GPU MTP case emitted no sidecar graph cache/capture counters")
        sys.exit(0)
    if not has_record(
        "live_prefix_replay_state_after_mutation",
        "mtp",
        {
            "operation": "clear_cache",
            "forward_replay_reset_scope": "request_boundary_preserve",
            "kernel_dynamic_state": "preserved",
        },
    ):
        print("FAIL: GPU MTP case did not preserve replay state at request-boundary clear_cache")
        sys.exit(0)

print(f"ok {len(records)}")
PY
)

    if [[ "$validation" == ok* ]]; then
        pass "[${tag}] PerfStats: captured ${validation#ok } records (${perf_path})"
    else
        fail "[${tag}] PerfStats: ${validation} (${perf_path})"
    fi
}

# ─── Validation ───────────────────────────────────────────────────────────────
case "$SERVER_MODE" in
    local|docker) ;;
    *)
        echo -e "${RED}Error: unsupported LLAMINAR_E2E_SERVER_MODE '${SERVER_MODE}' (expected local or docker)${NC}"
        exit 1
        ;;
esac

if is_docker_mode; then
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-llaminar:local}"
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker CLI not found for container E2E mode${NC}"
        exit 1
    fi
    case "$DOCKER_GPUS" in
        auto|none|"") ;;
        *)
            if ! docker_supports_nvidia_gpus && ! nvidia_device_nodes_available; then
                echo -e "${RED}Error: LLAMINAR_E2E_DOCKER_GPUS=${DOCKER_GPUS} requested NVIDIA devices, but Docker has no NVIDIA runtime and /dev/nvidia* is not visible for manual pass-through${NC}"
                echo "Install/configure nvidia-container-toolkit so 'docker run --gpus all ...' works, or expose NVIDIA device nodes to this environment."
                exit 1
            fi
            ;;
    esac
    if ! docker image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker image not found: ${CONTAINER_IMAGE}${NC}"
        echo "Build with: scripts/docker/build-runtime-image.sh --tag ${CONTAINER_IMAGE}"
        exit 1
    fi
    BINARY_ABS="/usr/local/bin/llaminar2"
    BINARY_DIR="${REPO_ROOT}/build_v2_release"
else
    if [ ! -x "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found: ${BINARY}${NC}"
    echo "Build with: cmake --build build_v2_integration --parallel"
    exit 1
    fi

    BINARY_ABS="$(readlink -f "$BINARY" 2>/dev/null || echo "$BINARY")"
    BINARY_DIR="$(dirname "$BINARY_ABS")"
fi
LOG_DIR="${LLAMINAR_E2E_LOG_DIR:-${BINARY_DIR}/e2e_server_logs}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(readlink -f "$LOG_DIR" 2>/dev/null || echo "$LOG_DIR")"

LAST_RESPONSE=""

run_chat_answer_check() {
    local tag="$1"
    local port="$2"
    local max_tokens="$3"
    local thinking_model="$4"
    local test_name="$5"
    local expected_answer="$6"
    local messages_json="$7"

    local modes=("false")
    if [ "$thinking_model" = "true" ]; then
        modes=("false" "true")
    fi

    local reference_answer=""
    local enable_thinking mode payload response content content_preview answer
    for enable_thinking in "${modes[@]}"; do
        mode=$(mode_name "$enable_thinking")
        payload=$(make_chat_payload "$messages_json" "$max_tokens" "$enable_thinking" "false")
        response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')
        LAST_RESPONSE="$response"

        content=$(printf '%s' "$response" | extract_content 2>/dev/null || echo "PARSE_ERROR")
        content_preview=$(printf '%s' "$content" | preview_text)
        answer=$(printf '%s' "$response" | extract_numeric_answer 2>/dev/null || echo "")

        if [ "$answer" = "$expected_answer" ]; then
            pass "[${tag}] ${test_name} (${mode}): got '${content_preview}' (answer ${expected_answer})"
        else
            fail "[${tag}] ${test_name} (${mode}): expected answer ${expected_answer}, got '${content_preview}'"
        fi

        if [ "$thinking_model" = "true" ] && [ "$enable_thinking" = "true" ]; then
            if printf '%s' "$content" | grep -q '</think>'; then
                fail "[${tag}] ${test_name} (${mode}): leaked thinking control tag in content '${content_preview}'"
            else
                pass "[${tag}] ${test_name} (${mode}): no leaked thinking control tags"
            fi
        fi

        if [ "$thinking_model" = "true" ]; then
            if [ -z "$reference_answer" ]; then
                reference_answer="$answer"
            elif [ "$answer" = "$reference_answer" ] && [ "$answer" = "$expected_answer" ]; then
                pass "[${tag}] ${test_name}: thinking and non-thinking answers match (${answer})"
            else
                fail "[${tag}] ${test_name}: thinking answer '${answer}' differs from non-thinking '${reference_answer}'"
            fi
        fi
    done
}

run_streaming_checks() {
    local tag="$1"
    local port="$2"
    local max_tokens="$3"
    local thinking_model="$4"
    local expected_answer="$5"
    local messages_json="$6"

    local modes=("false")
    if [ "$thinking_model" = "true" ]; then
        modes=("false" "true")
    fi

    local reference_answer=""
    local enable_thinking mode payload stream_raw stream_ok stream_meta_ok stream_content stream_preview answer
    for enable_thinking in "${modes[@]}"; do
        mode=$(mode_name "$enable_thinking")
        payload=$(make_chat_payload "$messages_json" "$max_tokens" "$enable_thinking" "true")
        stream_raw=$(curl -s --max-time "$REQUEST_TIMEOUT" -N \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo "CURL_FAILED")

        stream_ok=$(printf '%s' "$stream_raw" | python3 -c "
import sys
lines = sys.stdin.read().strip().split('\\n')
data_lines = [l for l in lines if l.startswith('data: ')]
if len(data_lines) < 2:
    print('FAIL: too few SSE lines')
    sys.exit(0)
if data_lines[-1].strip() != 'data: [DONE]':
    print('FAIL: missing [DONE] sentinel')
    sys.exit(0)
import json
first = json.loads(data_lines[0][6:])
if first.get('object') != 'chat.completion.chunk':
    print('FAIL: wrong object type')
    sys.exit(0)
delta = first.get('choices', [{}])[0].get('delta', {})
if delta.get('role') != 'assistant':
    print('FAIL: first chunk missing role')
    sys.exit(0)
for dl in data_lines[1:-1]:
    chunk = json.loads(dl[6:])
    fr = chunk.get('choices', [{}])[0].get('finish_reason')
    if fr in ('stop', 'length'):
        print('ok')
        sys.exit(0)
print('FAIL: no finish_reason chunk found')
" 2>/dev/null || echo "PARSE_ERROR")

        if [ "$stream_ok" = "ok" ]; then
            pass "[${tag}] SSE streaming (${mode}): valid chunks with role, content, finish, [DONE]"
        else
            fail "[${tag}] SSE streaming (${mode}): ${stream_ok}"
        fi

        stream_meta_ok=$(printf '%s' "$stream_raw" | python3 -c "
import json, sys
lines = sys.stdin.read().strip().split('\\n')
data_lines = [l for l in lines if l.startswith('data: ') and l.strip() != 'data: [DONE]']
if not data_lines:
    print('FAIL: no data lines'); sys.exit(0)
ids = set()
for dl in data_lines:
    chunk = json.loads(dl[6:])
    cid = chunk.get('id', '')
    if not cid.startswith('chatcmpl-'):
        print(f'FAIL: id missing chatcmpl- prefix: {cid}'); sys.exit(0)
    ids.add(cid)
    if chunk.get('system_fingerprint') != 'llaminar-v2':
        print('FAIL: wrong system_fingerprint'); sys.exit(0)
if len(ids) != 1:
    print(f'FAIL: inconsistent ids across chunks: {ids}'); sys.exit(0)
print('ok')
" 2>/dev/null || echo "PARSE_ERROR")

        if [ "$stream_meta_ok" = "ok" ]; then
            pass "[${tag}] SSE streaming (${mode}): metadata (id, system_fingerprint) consistent"
        else
            fail "[${tag}] SSE streaming metadata (${mode}): ${stream_meta_ok}"
        fi

        stream_content=$(printf '%s' "$stream_raw" | extract_stream_content 2>/dev/null || echo "PARSE_ERROR")
        stream_preview=$(printf '%s' "$stream_content" | preview_text)
        answer=$(printf '%s' "$stream_content" | extract_numeric_from_text 2>/dev/null || echo "")

        if [ "$answer" = "$expected_answer" ]; then
            pass "[${tag}] SSE streaming (${mode}): got '${stream_preview}' (answer ${expected_answer})"
        else
            fail "[${tag}] SSE streaming (${mode}): expected answer ${expected_answer}, got '${stream_preview}'"
        fi

        if [ "$thinking_model" = "true" ] && [ "$enable_thinking" = "true" ]; then
            if printf '%s' "$stream_content" | grep -q '</think>'; then
                fail "[${tag}] SSE streaming (${mode}): leaked thinking control tag in content '${stream_preview}'"
            else
                pass "[${tag}] SSE streaming (${mode}): no leaked thinking control tags"
            fi
        fi

        if [ "$thinking_model" = "true" ]; then
            if [ -z "$reference_answer" ]; then
                reference_answer="$answer"
            elif [ "$answer" = "$reference_answer" ] && [ "$answer" = "$expected_answer" ]; then
                pass "[${tag}] SSE streaming: thinking and non-thinking answers match (${answer})"
            else
                fail "[${tag}] SSE streaming: thinking answer '${answer}' differs from non-thinking '${reference_answer}'"
            fi
        fi
    done
}

run_prefill_graph_probe() {
    local tag="$1"
    local port="$2"

    local messages_json payload response validation i
    messages_json=$(python3 - <<'PY'
import json

filler = " ".join(
    "capture probe filler: alpha beta gamma delta epsilon zeta eta theta."
    for _ in range(110)
)
messages = [
    {
        "role": "system",
        "content": "You are a calculator. Reply with only the numeric answer.",
    },
    {
        "role": "user",
        "content": (
            f"{filler}\n\n"
            "Ignore the filler above. Final question: what is three plus five? "
            "Reply with only the numeric answer."
        ),
    },
]
print(json.dumps(messages, separators=(",", ":")))
PY
)

    for i in 1 2 3; do
        payload=$(make_chat_payload "$messages_json" 8 "false" "false")
        response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

        validation=$(printf '%s' "$response" | python3 -c "
import json
import re
import sys

try:
    data = json.load(sys.stdin)
    content = data.get('choices', [{}])[0].get('message', {}).get('content')
    usage = data.get('usage', {})
    prompt_tokens = int(usage.get('prompt_tokens', 0))
    completion_tokens = int(usage.get('completion_tokens', 0))
    if not isinstance(content, str):
        print('FAIL: missing assistant content')
    elif prompt_tokens < 256:
        print(f'FAIL: prompt_tokens {prompt_tokens} below prefill graph threshold')
    elif completion_tokens <= 0:
        print('FAIL: no completion tokens')
    elif not (match := re.search(r'-?\d+', content)):
        preview = content.replace('\\n', ' ')[:120]
        print(f'FAIL: no numeric answer in content {preview!r}')
    elif match.group(0) != '8':
        preview = content.replace('\\n', ' ')[:120]
        print(f'FAIL: expected answer 8, got {match.group(0)} in content {preview!r}')
    else:
        print(f'ok answer=8 prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}')
except Exception as exc:
    print(f'FAIL: malformed response: {exc}')
")
        if [[ "$validation" == ok* ]]; then
            continue
        fi
        fail "[${tag}] Prefill graph probe request ${i}: ${validation}"
        return
    done

    pass "[${tag}] Prefill graph probe: repeated same-key long prefill completed through replay request"
}

# ─── Test Runner Function ─────────────────────────────────────────────────────
# Runs the full test suite against a single model+backend combination.
# Arguments: $1=model_path $2=backend $3=port $4=model_label $5=max_tokens
run_backend_tests() {
    local model="$1"
    local backend="$2"
    local port="$3"
    local label="$4"
    local max_tokens="${5:-10}"
    local extra_flags="${6:-}"
    local suite_options="${7:-}"
    local tag="${label}/${backend}"
    local thinking_model="false"
    if is_thinking_model "$label"; then
        thinking_model="true"
    fi

    echo -e "${YELLOW}─── ${tag} (port ${port}) ───${NC}"

    local long_context_run="false"
    local model_size_b=""
    if [ "$LONG_CONTEXT_ENABLED" = "1" ]; then
        model_size_b=$(parse_model_size_b "$model" "$label")
        if suite_disables_long_context "$suite_options"; then
            echo -e "  ${YELLOW}SKIP${NC} [${tag}] Long-context: suite option no-long-context"
        elif model_size_meets_threshold "$model_size_b" "$LONG_MIN_MODEL_SIZE_B"; then
            long_context_run="true"
            echo -e "  ${BLUE}INFO${NC} [${tag}] Long-context enabled: size ${model_size_b}B, tier ${LONG_CONTEXT_TIER}, context ${CONTEXT_LENGTH}"
        else
            print_long_context_gate "$tag" "$model_size_b"
        fi
    fi

    local safe_tag log_path perf_path gpu_before_mb
    safe_tag=$(sanitize_name "$tag")
    log_path="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${safe_tag}_port${port}.log"
    perf_path="${log_path%.log}.perfstats.json"
    gpu_before_mb=$(get_total_gpu_memory_mb)
    if [ "$PERF_STATS_ENABLED" = "1" ]; then
        echo -e "  ${BLUE}INFO${NC} [${tag}] PerfStats artifact: ${perf_path}"
    fi

    local context_args=()
    if [ "$long_context_run" = "true" ]; then
        context_args=(-c "$CONTEXT_LENGTH")
    fi

    local -a server_env=(
        "LLAMINAR_LOG_LEVEL=$LOG_LEVEL"
        "LLAMINAR_TRACE_GENERATED_TOKENS=$TRACE_TOKENS"
    )
    if suite_disables_prefill_graph_buckets "$suite_options"; then
        echo -e "  ${BLUE}INFO${NC} [${tag}] Prefill graph buckets: disabled by suite option for unsupported collective topology"
        server_env+=("LLAMINAR_PREFILL_GRAPH_BUCKETS=0")
    fi
    if [ "$PERF_STATS_ENABLED" = "1" ]; then
        server_env+=("LLAMINAR_PERF_STATS_JSON=$perf_path")
        if [ "$PERF_STATS_GPU_STAGE_TIMING" = "1" ]; then
            server_env+=("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1")
        fi
    fi

    # Build server args — always pass the device explicitly unless this suite's
    # extra_flags describe a TP/PP topology.
    local server_model_path="$model"
    if is_docker_mode; then
        server_model_path="$(container_model_path "$model")"
    fi
    local -a server_args=(serve --port "$port" "${context_args[@]}")
    if [[ "$backend" != "tp" && "$backend" != "pp" ]]; then
        server_args+=(-d "$backend")
    fi
    local -a extra_arg_list=()
    if [[ -n "$extra_flags" ]]; then
        read -r -a extra_arg_list <<< "$extra_flags"
        server_args+=("${extra_arg_list[@]}")
    fi
    server_args+=(-m "$server_model_path")

    local server_handle
    start_server_process "$tag" "$port" "$model" "$log_path" "$safe_tag" server_env server_args
    server_handle="$STARTED_SERVER_HANDLE"
    if [ -z "$server_handle" ]; then
        fail "[${tag}] Server failed to start: no process handle was created"
        return
    fi

    # Wait for health (pass PID so we detect early exit / OOM)
    if ! wait_for_health "$port" "$server_handle"; then
        fail "[${tag}] Server failed to start within ${STARTUP_TIMEOUT}s"
        echo "    ── Last 80 lines of server log (${log_path}) ──"
        tail -n 80 "$log_path" 2>/dev/null | sed 's/^/    /' || echo "    (server log not available)"
        echo "    ────────────────────────────────────────────────────────────"
        scan_server_log "$tag" "$log_path"
        cleanup_server "$server_handle"
        return
    fi
    pass "[${tag}] Server started"

    # ─── Test 1: Health endpoint ──────────────────────────────────────
    local health_response
    health_response=$(curl -s --max-time 5 "http://127.0.0.1:${port}/health" 2>/dev/null || echo "CURL_FAILED")
    if echo "$health_response" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
        pass "[${tag}] GET /health returns ok"
    else
        fail "[${tag}] GET /health unexpected: ${health_response}"
    fi

    # ─── Test 2: Single-turn greedy inference ─────────────────────────
    # Simple arithmetic that works reliably across model sizes and backends.
    # Qwen3.5 thinking models are run in both thinking and non-thinking modes.
    local single_turn_messages multi_turn_messages cache_clear_messages stream_messages
    single_turn_messages='[{"role":"system","content":"You are a calculator. Reply with only the numeric answer, no explanation."},{"role":"user","content":"What is 2+2?"}]'
    run_chat_answer_check "$tag" "$port" "$max_tokens" "$thinking_model" "Single-turn" "4" "$single_turn_messages"

    # ─── Test 3: Multi-turn conversation ──────────────────────────────
    # Tests multi-turn context with simple recall.
    multi_turn_messages='[{"role":"system","content":"You are a helpful assistant. Reply briefly."},{"role":"user","content":"Remember this number: 42"},{"role":"assistant","content":"Got it, the number is 42."},{"role":"user","content":"What number did I tell you to remember? Reply with just the number."}]'
    run_chat_answer_check "$tag" "$port" "$max_tokens" "$thinking_model" "Multi-turn" "42" "$multi_turn_messages"

    # ─── Test 4: Second independent request (tests cache clear) ──────
    cache_clear_messages='[{"role":"system","content":"You are a calculator. Reply with only the numeric answer, no explanation."},{"role":"user","content":"What is 3+5?"}]'
    run_chat_answer_check "$tag" "$port" "$max_tokens" "$thinking_model" "Cache-clear" "8" "$cache_clear_messages"
    local response_format_sample="$LAST_RESPONSE"

    # ─── Prefix Cache Probe ───────────────────────────────────────────
    # Feature suites that start the server with --prefix-cache get a pair of
    # same-prefix requests. This keeps the check black-box and representative of
    # served inference while still exercising lookup, restore/bypass, harvest,
    # and ordinary response formatting through the HTTP path.
    if is_prefix_cache_case "$extra_flags"; then
        local prefix_a_messages prefix_b_messages
        prefix_a_messages='[{"role":"system","content":"You are a calculator. Reply with only the numeric answer, no explanation."},{"role":"user","content":"Shared prefix for prefix-cache E2E: keep this exact setup and answer the final arithmetic only. Final arithmetic: what is 6+7?"}]'
        prefix_b_messages='[{"role":"system","content":"You are a calculator. Reply with only the numeric answer, no explanation."},{"role":"user","content":"Shared prefix for prefix-cache E2E: keep this exact setup and answer the final arithmetic only. Final arithmetic: what is 9+5?"}]'
        run_chat_answer_check "$tag" "$port" "$max_tokens" "$thinking_model" "Prefix-cache shared-prefix A" "13" "$prefix_a_messages"
        run_chat_answer_check "$tag" "$port" "$max_tokens" "$thinking_model" "Prefix-cache shared-prefix B" "14" "$prefix_b_messages"
    fi

    # ─── Test 5: Response format validation ───────────────────────────
    local has_usage
    has_usage=$(printf '%s' "$response_format_sample" | validate_chat_response_format)

    if [ "$has_usage" = "ok" ]; then
        pass "[${tag}] Response format: valid usage + finish_reason"
    else
        fail "[${tag}] Response format: missing/invalid usage or finish_reason"
    fi

    # ─── Test 6/7: SSE streaming ─────────────────────────────────────
    stream_messages='[{"role":"system","content":"You are a calculator. Reply with only the numeric answer."},{"role":"user","content":"What is 1+1?"}]'
    run_streaming_checks "$tag" "$port" "$max_tokens" "$thinking_model" "2" "$stream_messages"

    # ─── Test 8: Error handling — invalid JSON ────────────────────────
    local error_response error_msg
    error_response=$(curl -s --max-time 5 -X POST \
        -H "Content-Type: application/json" \
        -d 'not valid json' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{}')

    error_msg=$(echo "$error_response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('error', {}).get('type', ''))
" 2>/dev/null || echo "PARSE_ERROR")

    if [ "$error_msg" = "invalid_request_error" ]; then
        pass "[${tag}] Error handling: invalid JSON returns 400"
    else
        fail "[${tag}] Error handling: expected invalid_request_error, got '${error_msg}'"
    fi

    # ─── Test 9: Error handling — missing messages ────────────────────
    error_response=$(curl -s --max-time 5 -X POST \
        -H "Content-Type: application/json" \
        -d '{"max_tokens": 10}' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{}')

    error_msg=$(echo "$error_response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('error', {}).get('type', ''))
" 2>/dev/null || echo "PARSE_ERROR")

    if [ "$error_msg" = "invalid_request_error" ]; then
        pass "[${tag}] Error handling: missing messages returns 400"
    else
        fail "[${tag}] Error handling: expected invalid_request_error, got '${error_msg}'"
    fi

    # ─── Optional Prefill Graph Capture Probe ─────────────────────────
    if suite_runs_prefill_graph_probe "$suite_options"; then
        if suite_disables_prefill_graph_buckets "$suite_options"; then
            fail "[${tag}] Prefill graph probe requested while no-prefill-graph-buckets is set"
        elif ! is_gpu_backend "$backend"; then
            fail "[${tag}] Prefill graph probe requested for non-GPU backend ${backend}"
        else
            run_prefill_graph_probe "$tag" "$port"
        fi
    fi

    # ─── Optional Long-Context Checks ─────────────────────────────────
    if [ "$long_context_run" = "true" ]; then
        run_long_context_checks "$tag" "$port" "$thinking_model"
    fi

    # ─── Test 10: Memory and server log hygiene ───────────────────────
    check_memory_usage "$tag" "$backend" "$model" "$server_handle" "$gpu_before_mb"

    # ─── Test 11: Graceful shutdown validation ────────────────────────
    shutdown_and_validate "$tag" "$server_handle" "$gpu_before_mb"
    copy_container_artifact "$server_handle" "$perf_path" "$perf_path"

    # ─── Test 12: Server log hygiene (after shutdown) ─────────────────
    # Scan AFTER shutdown so we catch errors during teardown too.
    # This covers exit code 1 from mpirun — if the child actually crashed
    # or errored, the log will have [ERROR]/[FATAL] entries.
    validate_perf_stats "$tag" "$backend" "$extra_flags" "$perf_path" "$long_context_run" "$suite_options"
    scan_server_log "$tag" "$log_path"
    echo ""
}

# ─── Run Test Suites ──────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  E2E Server Integration Test — Multi-Turn REST API${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
if is_docker_mode; then
    echo -e "  Server mode: docker"
    echo -e "  Container image: ${CONTAINER_IMAGE}"
    echo -e "  Docker network: ${DOCKER_NETWORK}"
else
    echo -e "  Server mode: local"
    echo -e "  Binary: ${BINARY}"
fi
echo -e "  Server logs: ${LOG_DIR}"
echo -e "  Server log level: ${LOG_LEVEL}"
if [ "$PERF_STATS_ENABLED" = "1" ]; then
    echo -e "  PerfStats: enabled, gpu_stage_timing=${PERF_STATS_GPU_STAGE_TIMING}"
else
    echo -e "  PerfStats: disabled"
fi
if [ "$LONG_CONTEXT_ENABLED" = "1" ]; then
    echo -e "  Long-context: enabled, tier=${LONG_CONTEXT_TIER}, context=${CONTEXT_LENGTH}, max_tokens=${LONG_MAX_TOKENS}, min_model_size=${LONG_MIN_MODEL_SIZE_B}B"
else
    echo -e "  Long-context: disabled"
fi
echo -e "  Suites: ${#SUITES[@]}"
for suite in "${SUITES[@]}"; do
    IFS='|' read -r local_model local_backends local_third local_extra local_label local_options <<< "$suite"
    # If third field isn't numeric, it's extra_flags (max_tokens was omitted)
    if [[ -n "$local_third" && ! "$local_third" =~ ^[0-9]+$ ]]; then
        local_options="$local_label"
        local_label="$local_extra"
        local_extra="$local_third"
    elif [[ -z "$local_extra" ]]; then
        local_extra=""
    fi
    local_suffix=""
    if [ -n "${local_label:-}" ]; then
        local_suffix=" [${local_label}]"
    elif [ -n "$local_extra" ]; then
        local_suffix=" [${local_extra}]"
    fi
    echo -e "    $(basename "$local_model")  →  ${local_backends}${local_suffix}"
done
echo ""

PORT=$BASE_PORT

for suite in "${SUITES[@]}"; do
    # Parse suite: "model_path|backends[|max_tokens[|extra_flags[|label[|suite_options]]]]"
    # If the third field is not numeric, treat it as extra_flags (max_tokens defaults to 200).
    IFS='|' read -r SUITE_MODEL SUITE_BACKENDS SUITE_MAX_TOKENS SUITE_EXTRA_FLAGS SUITE_LABEL_SUFFIX SUITE_OPTIONS <<< "$suite"
    if [[ -n "$SUITE_MAX_TOKENS" && ! "$SUITE_MAX_TOKENS" =~ ^[0-9]+$ ]]; then
        # Third field is not a number — it's extra_flags with max_tokens omitted
        SUITE_OPTIONS="${SUITE_LABEL_SUFFIX:-}"
        SUITE_LABEL_SUFFIX="${SUITE_EXTRA_FLAGS:-}"
        SUITE_EXTRA_FLAGS="$SUITE_MAX_TOKENS"
        SUITE_MAX_TOKENS="200"
    fi
    SUITE_MAX_TOKENS="${SUITE_MAX_TOKENS:-200}"  # Default: 200 tokens (enough for thinking models)
    SUITE_EXTRA_FLAGS="${SUITE_EXTRA_FLAGS:-}"   # Default: no extra flags
    SUITE_LABEL_SUFFIX="${SUITE_LABEL_SUFFIX:-}" # Default: derive from flags or model basename
    SUITE_OPTIONS="${SUITE_OPTIONS:-}"           # Default: no harness-only suite options
    SUITE_LABEL="$(basename "$SUITE_MODEL" .gguf)"
    if [ -n "$SUITE_LABEL_SUFFIX" ]; then
        SUITE_LABEL="${SUITE_LABEL} [${SUITE_LABEL_SUFFIX}]"
    elif [ -n "$SUITE_EXTRA_FLAGS" ]; then
        SUITE_LABEL="${SUITE_LABEL} [${SUITE_EXTRA_FLAGS}]"
    fi

    # Validate model exists
    if [ ! -f "$SUITE_MODEL" ]; then
        echo -e "${RED}Warning: Model not found: ${SUITE_MODEL} — skipping suite${NC}"
        continue
    fi

    echo -e "${BLUE}══ Model: ${SUITE_LABEL} ══${NC}"
    echo ""

    IFS=',' read -ra BACKEND_LIST <<< "$SUITE_BACKENDS"

    for BACKEND in "${BACKEND_LIST[@]}"; do
        BACKEND=$(echo "$BACKEND" | xargs)  # trim whitespace
        PORT=$((PORT + 1))
        run_backend_tests "$SUITE_MODEL" "$BACKEND" "$PORT" "$SUITE_LABEL" "$SUITE_MAX_TOKENS" "$SUITE_EXTRA_FLAGS" "$SUITE_OPTIONS"
    done
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}  ✅ ALL PASSED: ${PASSED_TESTS}/${TOTAL_TESTS} tests passed${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}  ❌ FAILED: ${FAILED_TESTS}/${TOTAL_TESTS} tests failed${NC}"
    echo -e "${RED}${FAILED_DETAILS}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
