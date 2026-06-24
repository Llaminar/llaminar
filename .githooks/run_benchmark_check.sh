#!/bin/bash
# Performance regression benchmark check for Llaminar pre-commit and CI.
#
# Local mode builds and benchmarks build_v2_release/llaminar2.
# Container mode benchmarks an already-built runtime image:
#
#   .githooks/run_benchmark_check.sh \
#     --container-image ghcr.io/owner/llaminar:develop-full-avx2 \
#     --cpu-isa AVX2
#
# CPU baselines are ISA-aware when the baseline contains:
#   devices.cpu.isa.AVX2.{prefill_tok_s,decode_tok_s}
#   devices.cpu.isa.AVX512.{prefill_tok_s,decode_tok_s}
#
# GPU baselines remain device-scoped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_FILE="$SCRIPT_DIR/benchmark_baseline.json"
RELEASE_BIN="$ROOT_DIR/build_v2_release/llaminar2"
RELEASE_CACHE="$ROOT_DIR/build_v2_release/CMakeCache.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
    cat <<'EOF'
Usage:
  .githooks/run_benchmark_check.sh [options]

Options:
  --update-baseline             Run benchmarks and overwrite the baseline values.
                                Local binary mode only.
  --container-image IMAGE       Run benchmarks inside this runtime container.
  --cpu-isa AVX2|AVX512         CPU ISA for benchmark identity/baseline lookup.
                                Required in container mode unless the image has
                                org.llaminar.cpu_isa or LLAMINAR_CPU_ISA metadata.
  --host-models-dir DIR         Host model directory mounted into containers.
                                Default: MODELS_DIR, then /opt/llaminar-models.
  --container-models-dir DIR    Container model directory. Default:
                                /opt/llaminar-models.
  -h, --help                    Show this help.
EOF
}

UPDATE_BASELINE=false
CONTAINER_IMAGE=""
CPU_ISA_OVERRIDE=""
HOST_MODELS_DIR="${MODELS_DIR:-/opt/llaminar-models}"
CONTAINER_MODELS_DIR="${LLAMINAR_CONTAINER_MODELS_DIR:-/opt/llaminar-models}"

while (($#)); do
    case "$1" in
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        --container-image)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 1; }
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --cpu-isa)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 1; }
            CPU_ISA_OVERRIDE="${2^^}"
            shift 2
            ;;
        --host-models-dir)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 1; }
            HOST_MODELS_DIR="$2"
            shift 2
            ;;
        --container-models-dir)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 1; }
            CONTAINER_MODELS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "run_benchmark_check: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

CONTAINER_MODE=false
if [[ -n "$CONTAINER_IMAGE" ]]; then
    CONTAINER_MODE=true
fi

normalize_isa() {
    local isa="${1^^}"
    case "$isa" in
        AVX2|AVX512) printf '%s' "$isa" ;;
        *) return 1 ;;
    esac
}

json_string() {
    jq -Rn --arg s "$1" '$s'
}

json_nullable_string() {
    if [[ -n "$1" ]]; then
        json_string "$1"
    else
        printf 'null'
    fi
}

csv_escape() {
    printf '%s' "$1" | sed 's/"/""/g'
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if ! command -v jq &>/dev/null; then
    echo -e "${RED}Error: jq is required but not installed. Install with: apt install jq${NC}" >&2
    exit 1
fi
if ! command -v bc &>/dev/null; then
    echo -e "${RED}Error: bc is required but not installed. Install with: apt install bc${NC}" >&2
    exit 1
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo -e "${RED}Error: Baseline file not found: $BASELINE_FILE${NC}" >&2
    echo -e "${YELLOW}Run with --update-baseline to create one.${NC}" >&2
    exit 1
fi

if $CONTAINER_MODE && $UPDATE_BASELINE; then
    echo -e "${RED}Error: --update-baseline is intentionally local-binary-only.${NC}" >&2
    exit 1
fi

if $CONTAINER_MODE; then
    if ! command -v docker &>/dev/null; then
        echo -e "${RED}Error: docker is required for --container-image mode.${NC}" >&2
        exit 1
    fi
    if ! docker image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1; then
        echo -e "${RED}Error: container image not found locally: $CONTAINER_IMAGE${NC}" >&2
        exit 1
    fi
    if [[ ! -d "$HOST_MODELS_DIR" ]]; then
        echo -e "${RED}Error: host models directory not found: $HOST_MODELS_DIR${NC}" >&2
        exit 1
    fi
else
    echo -e "${YELLOW}Building Release binary...${NC}"
    cmake -B "$ROOT_DIR/build_v2_release" -S "$ROOT_DIR/src/v2" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON > /dev/null 2>&1
    if ! cmake --build "$ROOT_DIR/build_v2_release" --parallel > /dev/null 2>&1; then
        echo -e "${RED}Error: Release build failed${NC}" >&2
        echo -e "${YELLOW}Run manually to see errors: cmake --build build_v2_release --parallel${NC}" >&2
        exit 1
    fi
    if [[ ! -x "$RELEASE_BIN" ]]; then
        echo -e "${RED}Error: Release binary not found after build${NC}" >&2
        exit 1
    fi
    echo -e "${GREEN}✓ Release build complete${NC}"
fi

CPU_ISA_TARGET="${CPU_ISA_OVERRIDE}"
if [[ -z "$CPU_ISA_TARGET" && "$CONTAINER_MODE" == "true" ]]; then
    CPU_ISA_TARGET=$(docker image inspect \
        --format '{{ index .Config.Labels "org.llaminar.cpu_isa" }}' \
        "$CONTAINER_IMAGE" 2>/dev/null || true)
    CPU_ISA_TARGET="${CPU_ISA_TARGET^^}"
    if [[ -z "$CPU_ISA_TARGET" || "$CPU_ISA_TARGET" == "<NO VALUE>" ]]; then
        CPU_ISA_TARGET=$(docker image inspect --format '{{ range .Config.Env }}{{ println . }}{{ end }}' \
            "$CONTAINER_IMAGE" 2>/dev/null | sed -n 's/^LLAMINAR_CPU_ISA=//p' | tail -1 || true)
        CPU_ISA_TARGET="${CPU_ISA_TARGET^^}"
    fi
fi

if [[ -z "$CPU_ISA_TARGET" && -f "$RELEASE_CACHE" ]]; then
    CPU_ISA_TARGET=$(grep -E '^LLAMINAR_CPU_ISA:' "$RELEASE_CACHE" | tail -1 | sed 's/.*=//' || true)
    CPU_ISA_TARGET="${CPU_ISA_TARGET^^}"
fi
CPU_ISA_TARGET="${CPU_ISA_TARGET:-AVX512}"
if ! CPU_ISA_TARGET=$(normalize_isa "$CPU_ISA_TARGET"); then
    echo -e "${RED}Error: unsupported CPU ISA '$CPU_ISA_TARGET' (expected AVX2 or AVX512).${NC}" >&2
    exit 1
fi

if $CONTAINER_MODE; then
    IMAGE_CPU_ISA=$(docker image inspect \
        --format '{{ index .Config.Labels "org.llaminar.cpu_isa" }}' \
        "$CONTAINER_IMAGE" 2>/dev/null || true)
    IMAGE_CPU_ISA="${IMAGE_CPU_ISA^^}"
    if [[ -n "$IMAGE_CPU_ISA" && "$IMAGE_CPU_ISA" != "<NO VALUE>" && "$IMAGE_CPU_ISA" != "$CPU_ISA_TARGET" ]]; then
        echo -e "${YELLOW}Warning: --cpu-isa ${CPU_ISA_TARGET} differs from image label ${IMAGE_CPU_ISA}.${NC}"
    fi
    echo -e "${BLUE}Benchmark source: container ${CONTAINER_IMAGE} (CPU ISA ${CPU_ISA_TARGET})${NC}"
else
    echo -e "${BLUE}Benchmark source: local Release build (CPU ISA ${CPU_ISA_TARGET})${NC}"
fi

# ---------------------------------------------------------------------------
# Read global settings
# ---------------------------------------------------------------------------
GLOBAL_THRESHOLD_PCT=$(jq -r '.regression_threshold_pct' "$BASELINE_FILE")
NUM_MODELS=$(jq '.models | length' "$BASELINE_FILE")

if [[ "$NUM_MODELS" -eq 0 ]]; then
    echo -e "${RED}Error: No models defined in baseline file${NC}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
parse_benchmark_output() {
    local output="$1"
    local prefill_tps decode_tps
    prefill_tps=$(echo "$output" | grep -A4 "PREFILL" | grep "Throughput" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    decode_tps=$(echo "$output" | grep -A4 "DECODE" | grep "Throughput" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    echo "${prefill_tps:-0} ${decode_tps:-0}"
}

benchmark_effective_threshold() {
    local model_idx="$1" device="$2"
    local device_threshold model_threshold
    device_threshold=$(jq -r ".models[$model_idx].devices[\"$device\"].regression_threshold_pct // empty" "$BASELINE_FILE")
    model_threshold=$(jq -r ".models[$model_idx].regression_threshold_pct // empty" "$BASELINE_FILE")
    echo "${device_threshold:-${model_threshold:-$GLOBAL_THRESHOLD_PCT}}"
}

is_direct_cpu_device() {
    local device="$1"
    [[ "$device" == "cpu" || "$device" == cpu:* ]]
}

baseline_metric() {
    local model_idx="$1" device="$2" metric="$3"
    if is_direct_cpu_device "$device" &&
       jq -e --arg isa "$CPU_ISA_TARGET" ".models[$model_idx].devices[\"$device\"].isa[\$isa]" "$BASELINE_FILE" >/dev/null 2>&1; then
        jq -r --arg isa "$CPU_ISA_TARGET" ".models[$model_idx].devices[\"$device\"].isa[\$isa].$metric // 0" "$BASELINE_FILE"
    else
        jq -r ".models[$model_idx].devices[\"$device\"].$metric // 0" "$BASELINE_FILE"
    fi
}

metric_delta_pct() {
    local baseline="$1" current="$2"
    echo "scale=4; ($current - $baseline) / $baseline * 100" | bc -l
}

is_borderline_regression() {
    local baseline="$1" current="$2" threshold="$3"
    if [[ "$baseline" == "0" || "$baseline" == "null" || "$current" == "0" || -z "$current" ]]; then
        return 1
    fi

    local delta lower_bound
    delta=$(metric_delta_pct "$baseline" "$current")
    lower_bound=$(echo "-(${threshold} + ${RECHECK_MARGIN_PCT})" | bc -l)
    if (( $(echo "$delta < -${threshold} && $delta >= $lower_bound" | bc -l) )); then
        return 0
    fi
    return 1
}

resolve_host_model_path() {
    local model="$1"
    if $CONTAINER_MODE && [[ "$model" == models/* ]]; then
        printf '%s/%s' "$HOST_MODELS_DIR" "${model#models/}"
    elif $CONTAINER_MODE && [[ "$model" == "$CONTAINER_MODELS_DIR/"* ]]; then
        printf '%s/%s' "$HOST_MODELS_DIR" "${model#"$CONTAINER_MODELS_DIR/"}"
    elif [[ "$model" == /* ]]; then
        printf '%s' "$model"
    else
        printf '%s/%s' "$ROOT_DIR" "$model"
    fi
}

resolve_container_model_path() {
    local model="$1" host_path="$2"
    if [[ "$model" == models/* ]]; then
        printf '%s/%s' "$CONTAINER_MODELS_DIR" "${model#models/}"
    elif [[ "$host_path" == "$HOST_MODELS_DIR/"* ]]; then
        printf '%s/%s' "$CONTAINER_MODELS_DIR" "${host_path#"$HOST_MODELS_DIR/"}"
    else
        printf '%s' "$host_path"
    fi
}

append_extra_flags() {
    local extra_flags="$1"
    if [[ -n "$extra_flags" ]]; then
        # Existing baseline extra_flags are simple shell words. Preserve the
        # hook's historical word-splitting behavior.
        read -r -a EXTRA_FLAG_WORDS <<< "$extra_flags"
    else
        EXTRA_FLAG_WORDS=()
    fi
}

run_benchmark_once() {
    local device="$1" model_path="$2" decode_tokens="$3" extra_flags="$4" env_prefix="$5"
    local cmd=(benchmark)
    if [[ "$device" != "tp" && "$device" != "pp" ]]; then
        cmd+=(-d "$device")
    fi
    cmd+=(-m "$model_path" -n "$decode_tokens")
    append_extra_flags "$extra_flags"
    cmd+=("${EXTRA_FLAG_WORDS[@]}")

    if ! $CONTAINER_MODE; then
        env $env_prefix "$RELEASE_BIN" "${cmd[@]}"
        return
    fi

    local docker_args=(--rm --ipc=host --shm-size=16g --cap-add SYS_NICE)
    local nvidia_args=()
    if [[ -x "$ROOT_DIR/scripts/ci/docker_gpu_run_args.sh" ]]; then
        mapfile -t nvidia_args < <("$ROOT_DIR/scripts/ci/docker_gpu_run_args.sh" --probe-image "$CONTAINER_IMAGE" || true)
        docker_args+=("${nvidia_args[@]}")
    fi
    if [[ -e /dev/kfd ]]; then
        docker_args+=(--device=/dev/kfd)
    fi
    if [[ -e /dev/dri ]]; then
        docker_args+=(--device=/dev/dri)
    fi
    docker_args+=(--group-add video --group-add render)
    docker_args+=(-e "MODELS_DIR=$CONTAINER_MODELS_DIR")
    docker_args+=(-e "LLAMINAR_CPU_ISA=$CPU_ISA_TARGET")
    local env_entry
    for env_entry in $env_prefix; do
        docker_args+=(-e "$env_entry")
    done
    docker_args+=(-v "$HOST_MODELS_DIR:$CONTAINER_MODELS_DIR:ro")

    docker run "${docker_args[@]}" "$CONTAINER_IMAGE" "${cmd[@]}"
}

display_device() {
    local device="$1"
    if is_direct_cpu_device "$device"; then
        printf '%s[%s]' "$device" "$CPU_ISA_TARGET"
    else
        printf '%s' "$device"
    fi
}

# Results are keyed by "model_idx:device" to avoid collisions across models.
declare -A RESULTS_PREFILL
declare -A RESULTS_DECODE
OVERALL_PASS=true
FAILED_CHECKS=""
RECHECK_MARGIN_PCT="${LLAMINAR_BENCHMARK_RECHECK_MARGIN_PCT:-1}"
BENCHMARK_DEFAULT_ENV="LLAMINAR_GPU_STAGE_TIMING=0"
BENCHMARK_SOURCE="local"
if $CONTAINER_MODE; then
    BENCHMARK_SOURCE="container"
fi

# ---------------------------------------------------------------------------
# Run benchmarks for all models × devices
# ---------------------------------------------------------------------------
for (( mi=0; mi<NUM_MODELS; mi++ )); do
    MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
    MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
    DECODE_TOKENS=$(jq -r ".models[$mi].decode_tokens" "$BASELINE_FILE")
    DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

    ENV_PREFIX="$BENCHMARK_DEFAULT_ENV"
    if jq -e ".models[$mi].env" "$BASELINE_FILE" > /dev/null 2>&1; then
        ENV_PREFIX="$ENV_PREFIX $(jq -r ".models[$mi].env | to_entries[] | \"\(.key)=\(.value)\"" "$BASELINE_FILE" | tr '\n' ' ')"
    fi

    HOST_MODEL_PATH=$(resolve_host_model_path "$MODEL")
    if [[ ! -f "$HOST_MODEL_PATH" ]]; then
        echo -e "${RED}✗ FAILED: ${MODEL_NAME}: model not found (${MODEL})${NC}" >&2
        OVERALL_PASS=false
        FAILED_CHECKS+="  [${MODEL_NAME}] Model file missing: ${MODEL}\n"
        continue
    fi
    MODEL_PATH="$HOST_MODEL_PATH"
    if $CONTAINER_MODE; then
        MODEL_PATH=$(resolve_container_model_path "$MODEL" "$HOST_MODEL_PATH")
    fi

    echo -e "${BLUE}Benchmarking: ${BOLD}${MODEL_NAME}${NC}"
    echo -e "${BLUE}  Model: $(basename "$MODEL"), decode tokens: ${DECODE_TOKENS}${NC}"
    echo ""

    for DEVICE in $DEVICES; do
        KEY="${mi}:${DEVICE}"
        echo -ne "  Benchmarking ${BOLD}$(display_device "$DEVICE")${NC} ... "

        EXTRA_FLAGS=""
        if jq -e ".models[$mi].devices[\"$DEVICE\"].extra_flags" "$BASELINE_FILE" > /dev/null 2>&1; then
            EXTRA_FLAGS=$(jq -r ".models[$mi].devices[\"$DEVICE\"].extra_flags" "$BASELINE_FILE")
        fi

        set +e
        BENCH_OUTPUT=$(run_benchmark_once "$DEVICE" "$MODEL_PATH" "$DECODE_TOKENS" "$EXTRA_FLAGS" "$ENV_PREFIX" 2>&1)
        BENCH_EXIT=$?
        set -e

        if [[ $BENCH_EXIT -ne 0 ]]; then
            echo -e "${RED}FAILED (exit code ${BENCH_EXIT})${NC}"
            OVERALL_PASS=false
            FAILED_CHECKS+="  [${MODEL_NAME}] ${DEVICE}: benchmark failed (exit code ${BENCH_EXIT})\n"
            continue
        fi

        read -r PREFILL DECODE <<< "$(parse_benchmark_output "$BENCH_OUTPUT")"
        if [[ "$PREFILL" == "0" || "$DECODE" == "0" ]]; then
            echo -e "${RED}FAILED (could not parse output)${NC}"
            OVERALL_PASS=false
            FAILED_CHECKS+="  [${MODEL_NAME}] ${DEVICE}: benchmark produced unparseable output\n"
            continue
        fi

        RESULTS_PREFILL[$KEY]=$PREFILL
        RESULTS_DECODE[$KEY]=$DECODE
        echo -e "prefill ${GREEN}${PREFILL}${NC} tok/s, decode ${GREEN}${DECODE}${NC} tok/s"
    done

    echo ""
done

# ---------------------------------------------------------------------------
# Recheck borderline regressions once to avoid commit blocks from benchmark
# jitter. This does not relax baselines or thresholds; severe regressions still
# fail immediately, and borderline regressions must pass on a repeat run.
# ---------------------------------------------------------------------------
RECHECK_COUNT=0
for (( mi=0; mi<NUM_MODELS; mi++ )); do
    MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
    MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
    DECODE_TOKENS=$(jq -r ".models[$mi].decode_tokens" "$BASELINE_FILE")
    DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

    ENV_PREFIX="$BENCHMARK_DEFAULT_ENV"
    if jq -e ".models[$mi].env" "$BASELINE_FILE" > /dev/null 2>&1; then
        ENV_PREFIX="$ENV_PREFIX $(jq -r ".models[$mi].env | to_entries[] | \"\(.key)=\(.value)\"" "$BASELINE_FILE" | tr '\n' ' ')"
    fi

    HOST_MODEL_PATH=$(resolve_host_model_path "$MODEL")
    MODEL_PATH="$HOST_MODEL_PATH"
    if $CONTAINER_MODE; then
        MODEL_PATH=$(resolve_container_model_path "$MODEL" "$HOST_MODEL_PATH")
    fi

    for DEVICE in $DEVICES; do
        KEY="${mi}:${DEVICE}"
        if [[ -z "${RESULTS_PREFILL[$KEY]:-}" ]]; then
            continue
        fi

        BASELINE_PREFILL=$(baseline_metric "$mi" "$DEVICE" "prefill_tok_s")
        BASELINE_DECODE=$(baseline_metric "$mi" "$DEVICE" "decode_tok_s")
        EFFECTIVE_THRESHOLD=$(benchmark_effective_threshold "$mi" "$DEVICE")

        if ! is_borderline_regression "$BASELINE_PREFILL" "${RESULTS_PREFILL[$KEY]}" "$EFFECTIVE_THRESHOLD" &&
           ! is_borderline_regression "$BASELINE_DECODE" "${RESULTS_DECODE[$KEY]}" "$EFFECTIVE_THRESHOLD"; then
            continue
        fi

        if [[ "$RECHECK_COUNT" -eq 0 ]]; then
            echo -e "${YELLOW}Rechecking borderline benchmark regressions once...${NC}"
        fi
        RECHECK_COUNT=$((RECHECK_COUNT + 1))

        EXTRA_FLAGS=""
        if jq -e ".models[$mi].devices[\"$DEVICE\"].extra_flags" "$BASELINE_FILE" > /dev/null 2>&1; then
            EXTRA_FLAGS=$(jq -r ".models[$mi].devices[\"$DEVICE\"].extra_flags" "$BASELINE_FILE")
        fi

        echo -ne "  Rechecking ${BOLD}${MODEL_NAME}${NC} on ${BOLD}$(display_device "$DEVICE")${NC} ... "
        set +e
        BENCH_OUTPUT=$(run_benchmark_once "$DEVICE" "$MODEL_PATH" "$DECODE_TOKENS" "$EXTRA_FLAGS" "$ENV_PREFIX" 2>&1)
        BENCH_EXIT=$?
        set -e

        if [[ $BENCH_EXIT -ne 0 ]]; then
            echo -e "${RED}FAILED (keeping original result)${NC}"
            continue
        fi

        read -r PREFILL DECODE <<< "$(parse_benchmark_output "$BENCH_OUTPUT")"
        if [[ "$PREFILL" == "0" || "$DECODE" == "0" ]]; then
            echo -e "${RED}FAILED (unparseable; keeping original result)${NC}"
            continue
        fi

        if (( $(echo "$PREFILL > ${RESULTS_PREFILL[$KEY]}" | bc -l) )); then
            RESULTS_PREFILL[$KEY]=$PREFILL
        fi
        if (( $(echo "$DECODE > ${RESULTS_DECODE[$KEY]}" | bc -l) )); then
            RESULTS_DECODE[$KEY]=$DECODE
        fi

        echo -e "best prefill ${GREEN}${RESULTS_PREFILL[$KEY]}${NC} tok/s, best decode ${GREEN}${RESULTS_DECODE[$KEY]}${NC} tok/s"
    done
done
if [[ "$RECHECK_COUNT" -gt 0 ]]; then
    echo ""
fi

# ---------------------------------------------------------------------------
# Emit machine-readable results JSON/CSV for CI summary tooling.
# ---------------------------------------------------------------------------
RESULTS_DIR="${LLAMINAR_BENCHMARK_RESULTS_DIR:-${ROOT_DIR}/benchmark_results}"
COMMIT_HASH=$(git -C "$ROOT_DIR" rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
mkdir -p "${RESULTS_DIR}/${COMMIT_HASH}"
RESULTS_JSON="${RESULTS_DIR}/${COMMIT_HASH}/benchmark_results.json"
RESULTS_CSV="${RESULTS_DIR}/${COMMIT_HASH}/benchmark_results.csv"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE_JSON=$(json_nullable_string "$CONTAINER_IMAGE")
SOURCE_JSON=$(json_string "$BENCHMARK_SOURCE")
ISA_JSON=$(json_string "$CPU_ISA_TARGET")

{
    echo "{"
    echo "  \"schema\": \"llaminar.benchmark.v2\","
    echo "  \"commit\": \"${COMMIT_HASH}\","
    echo "  \"timestamp\": \"${TS}\","
    echo "  \"source\": ${SOURCE_JSON},"
    echo "  \"image\": ${IMAGE_JSON},"
    echo "  \"container_cpu_isa\": ${ISA_JSON},"
    echo "  \"runs\": ["
    echo "    {"
    echo "      \"source\": ${SOURCE_JSON},"
    echo "      \"image\": ${IMAGE_JSON},"
    echo "      \"cpu_isa\": ${ISA_JSON}"
    echo "    }"
    echo "  ],"
    echo "  \"models\": ["
    FIRST_MODEL=true
    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
        MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")
        $FIRST_MODEL || echo ","
        FIRST_MODEL=false
        echo "    {"
        echo "      \"name\": $(json_string "$MODEL_NAME"),"
        echo "      \"model\": $(json_string "$MODEL"),"
        echo "      \"devices\": ["
        FIRST_DEV=true
        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            $FIRST_DEV || echo ","
            FIRST_DEV=false
            BL_P=$(baseline_metric "$mi" "$DEVICE" "prefill_tok_s")
            BL_D=$(baseline_metric "$mi" "$DEVICE" "decode_tok_s")
            CUR_P="${RESULTS_PREFILL[$KEY]:-null}"
            CUR_D="${RESULTS_DECODE[$KEY]:-null}"
            CPU_ISA_JSON="null"
            if is_direct_cpu_device "$DEVICE"; then
                CPU_ISA_JSON="$ISA_JSON"
            fi
            echo "        {"
            echo "          \"device\": $(json_string "$DEVICE"),"
            echo "          \"cpu_isa\": ${CPU_ISA_JSON},"
            echo "          \"container_cpu_isa\": ${ISA_JSON},"
            echo "          \"source\": ${SOURCE_JSON},"
            echo "          \"image\": ${IMAGE_JSON},"
            echo "          \"prefill_tok_s\": ${CUR_P},"
            echo "          \"decode_tok_s\": ${CUR_D},"
            echo "          \"baseline_prefill_tok_s\": ${BL_P},"
            echo "          \"baseline_decode_tok_s\": ${BL_D}"
            echo -n "        }"
        done
        echo
        echo "      ]"
        echo -n "    }"
    done
    echo
    echo "  ]"
    echo "}"
} > "$RESULTS_JSON"
echo -e "${BLUE}Wrote benchmark results JSON: ${RESULTS_JSON}${NC}"

{
    echo "commit,timestamp,source,image,container_cpu_isa,model_name,model_path,device,cpu_isa,prefill_tok_s,decode_tok_s,baseline_prefill_tok_s,baseline_decode_tok_s"
    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
        MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")
        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            BL_P=$(baseline_metric "$mi" "$DEVICE" "prefill_tok_s")
            BL_D=$(baseline_metric "$mi" "$DEVICE" "decode_tok_s")
            CUR_P="${RESULTS_PREFILL[$KEY]:-}"
            CUR_D="${RESULTS_DECODE[$KEY]:-}"
            ROW_CPU_ISA=""
            if is_direct_cpu_device "$DEVICE"; then
                ROW_CPU_ISA="$CPU_ISA_TARGET"
            fi
            esc_name=$(csv_escape "$MODEL_NAME")
            esc_model=$(csv_escape "$MODEL")
            esc_image=$(csv_escape "$CONTAINER_IMAGE")
            echo "${COMMIT_HASH},${TS},${BENCHMARK_SOURCE},\"${esc_image}\",${CPU_ISA_TARGET},\"${esc_name}\",\"${esc_model}\",${DEVICE},${ROW_CPU_ISA},${CUR_P},${CUR_D},${BL_P},${BL_D}"
        done
    done
} > "$RESULTS_CSV"
echo -e "${BLUE}Wrote benchmark results CSV:  ${RESULTS_CSV}${NC}"

# ---------------------------------------------------------------------------
# Update baseline mode
# ---------------------------------------------------------------------------
if $UPDATE_BASELINE; then
    echo -e "${YELLOW}Updating baseline file: $BASELINE_FILE${NC}"

    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")
        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            if [[ -z "${RESULTS_PREFILL[$KEY]:-}" ]]; then
                continue
            fi
            if is_direct_cpu_device "$DEVICE"; then
                jq --argjson mi "$mi" \
                   --arg dev "$DEVICE" \
                   --arg isa "$CPU_ISA_TARGET" \
                   --argjson pf "${RESULTS_PREFILL[$KEY]}" \
                   --argjson dc "${RESULTS_DECODE[$KEY]}" \
                   '.models[$mi].devices[$dev].isa[$isa].prefill_tok_s = $pf |
                    .models[$mi].devices[$dev].isa[$isa].decode_tok_s = $dc' \
                   "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
            else
                jq --argjson mi "$mi" \
                   --arg dev "$DEVICE" \
                   --argjson pf "${RESULTS_PREFILL[$KEY]}" \
                   --argjson dc "${RESULTS_DECODE[$KEY]}" \
                   '.models[$mi].devices[$dev].prefill_tok_s = $pf |
                    .models[$mi].devices[$dev].decode_tok_s = $dc' \
                   "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
            fi
        done
    done

    echo -e "${GREEN}✓ Baseline updated${NC}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compare against baseline
# ---------------------------------------------------------------------------
check_regression() {
    local model_idx="$1" model_name="$2" device="$3" phase="$4" baseline="$5" current="$6"

    if [[ "$current" == "0" || -z "$current" ]]; then
        return
    fi

    if [[ "$baseline" == "0" || "$baseline" == "null" ]]; then
        printf "  %-16s  %-10s  %10s    %10.1f    %8s  " "$(display_device "$device")" "$phase" "(new)" "$current" "-"
        echo -e "${YELLOW}~ no baseline${NC}"
        return
    fi

    local effective_threshold
    effective_threshold=$(benchmark_effective_threshold "$model_idx" "$device")

    local delta
    delta=$(metric_delta_pct "$baseline" "$current")
    delta=$(printf "%.1f" "$delta")

    local status="${GREEN}✓ OK${NC}"
    if (( $(echo "$delta < -${effective_threshold}" | bc -l) )); then
        status="${RED}✗ REGRESSED${NC}"
        OVERALL_PASS=false
        FAILED_CHECKS+="  [${model_name}] $(display_device "$device") ${phase}: ${baseline} → ${current} tok/s (${delta}%, threshold ${effective_threshold}%)\n"
    elif (( $(echo "$delta < 0" | bc -l) )); then
        status="${YELLOW}~ slower${NC}"
    elif (( $(echo "$delta > 0" | bc -l) )); then
        status="${GREEN}▲ faster${NC}"
    fi

    printf "  %-16s  %-10s  %10.1f    %10.1f    %+6.1f%%  " "$(display_device "$device")" "$phase" "$baseline" "$current" "$delta"
    echo -e "$status"
}

for (( mi=0; mi<NUM_MODELS; mi++ )); do
    MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
    DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

    echo -e "${BOLD}${MODEL_NAME}:${NC}"
    printf "  %-16s  %-10s  %12s  %12s  %8s  %s\n" "Device" "Phase" "Baseline" "Current" "Delta" "Status"
    printf "  %-16s  %-10s  %12s  %12s  %8s  %s\n" "------" "------" "--------" "-------" "-----" "------"

    for DEVICE in $DEVICES; do
        KEY="${mi}:${DEVICE}"
        if [[ -z "${RESULTS_PREFILL[$KEY]:-}" ]]; then
            printf "  %-16s  %-10s  %12s  %12s  %8s  " "$(display_device "$DEVICE")" "prefill" "-" "-" "-"
            echo -e "${RED}FAILED${NC}"
            printf "  %-16s  %-10s  %12s  %12s  %8s  " "$(display_device "$DEVICE")" "decode" "-" "-" "-"
            echo -e "${RED}FAILED${NC}"
            continue
        fi

        BASELINE_PREFILL=$(baseline_metric "$mi" "$DEVICE" "prefill_tok_s")
        BASELINE_DECODE=$(baseline_metric "$mi" "$DEVICE" "decode_tok_s")

        check_regression "$mi" "$MODEL_NAME" "$DEVICE" "prefill" "$BASELINE_PREFILL" "${RESULTS_PREFILL[$KEY]}"
        check_regression "$mi" "$MODEL_NAME" "$DEVICE" "decode" "$BASELINE_DECODE" "${RESULTS_DECODE[$KEY]}"
    done

    echo ""
done

if $OVERALL_PASS; then
    echo -e "${GREEN}✓ No performance regressions detected${NC}"
    echo -e "${BLUE}Baseline file unchanged. Use --update-baseline after explicit approval to rewrite baseline values.${NC}"
    exit 0
else
    echo -e "${RED}✗ Performance regression detected!${NC}"
    echo ""
    echo -e "${RED}Regressed metrics:${NC}"
    echo -e "$FAILED_CHECKS"
    echo -e "${YELLOW}If this is expected (e.g. correctness fix), update the baseline:${NC}"
    echo -e "${YELLOW}  .githooks/run_benchmark_check.sh --update-baseline${NC}"
    echo ""
    echo -e "${YELLOW}Or skip with: git commit --no-verify${NC}"
    exit 1
fi
