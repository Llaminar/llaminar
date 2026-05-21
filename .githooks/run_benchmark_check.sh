#!/bin/bash
# Performance regression benchmark check for Llaminar pre-commit hook.
#
# Runs 'llaminar2 benchmark' for each model × device combination listed in
# the baseline JSON, parses prefill/decode tok/s, and fails if any metric
# regresses beyond the configured threshold.
#
# Baseline JSON format (array of model configs):
#   { "regression_threshold_pct": 10,
#     "models": [
#       { "name": "...", "model": "...", "decode_tokens": 128,
#         "devices": { "cpu": { "prefill_tok_s": ..., "decode_tok_s": ..., "regression_threshold_pct": 25 }, ... } },
#       ...
#     ] }
#
# Device-specific thresholds override model/global thresholds. CPU entries use
# 25% tolerance to avoid noisy local-load commit blocks; GPU entries stay tight.
#
# Usage:
#   .githooks/run_benchmark_check.sh                  # normal regression check
#   .githooks/run_benchmark_check.sh --update-baseline # run benchmarks and overwrite baseline
#
# Requires: jq, release build of llaminar2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_FILE="$SCRIPT_DIR/benchmark_baseline.json"
RELEASE_BIN="$ROOT_DIR/build_v2_release/llaminar2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

UPDATE_BASELINE=false
if [[ "${1:-}" == "--update-baseline" ]]; then
    UPDATE_BASELINE=true
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if ! command -v jq &>/dev/null; then
    echo -e "${RED}Error: jq is required but not installed. Install with: apt install jq${NC}" >&2
    exit 1
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo -e "${RED}Error: Baseline file not found: $BASELINE_FILE${NC}" >&2
    echo -e "${YELLOW}Run with --update-baseline to create one.${NC}" >&2
    exit 1
fi

# Always rebuild Release to benchmark against the current source
echo -e "${YELLOW}Building Release binary...${NC}"
cmake -B "$ROOT_DIR/build_v2_release" -S "$ROOT_DIR/src/v2" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
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

# Results are keyed by "model_idx:device" to avoid collisions across models
declare -A RESULTS_PREFILL
declare -A RESULTS_DECODE
OVERALL_PASS=true
FAILED_CHECKS=""

# ---------------------------------------------------------------------------
# Run benchmarks for all models × devices
# ---------------------------------------------------------------------------
for (( mi=0; mi<NUM_MODELS; mi++ )); do
    MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
    MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
    DECODE_TOKENS=$(jq -r ".models[$mi].decode_tokens" "$BASELINE_FILE")
    DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

    # Optional per-model environment variables (e.g. LLAMINAR_MOE_REBALANCE=off)
    ENV_PREFIX=""
    if jq -e ".models[$mi].env" "$BASELINE_FILE" > /dev/null 2>&1; then
        ENV_PREFIX=$(jq -r ".models[$mi].env | to_entries[] | \"\(.key)=\(.value)\"" "$BASELINE_FILE" | tr '\n' ' ')
    fi

    MODEL_PATH="$ROOT_DIR/$MODEL"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${RED}✗ FAILED: ${MODEL_NAME}: model not found (${MODEL})${NC}" >&2
        OVERALL_PASS=false
        FAILED_CHECKS+="  [${MODEL_NAME}] Model file missing: ${MODEL}\n"
        continue
    fi

    echo -e "${BLUE}Benchmarking: ${BOLD}${MODEL_NAME}${NC}"
    echo -e "${BLUE}  Model: $(basename "$MODEL"), decode tokens: ${DECODE_TOKENS}${NC}"
    echo ""

    for DEVICE in $DEVICES; do
        KEY="${mi}:${DEVICE}"
        echo -ne "  Benchmarking ${BOLD}${DEVICE}${NC} ... "

        set +e
        BENCH_OUTPUT=$(env $ENV_PREFIX "$RELEASE_BIN" benchmark -d "$DEVICE" -m "$MODEL_PATH" -n "$DECODE_TOKENS" 2>&1)
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
# Emit machine-readable results JSON for CI summary tooling.
# Path: $LLAMINAR_BENCHMARK_RESULTS_DIR/<git-hash>/benchmark_results.json
# Schema: { "commit": "...", "timestamp": "...", "models": [
#            { "name": "...", "model": "...", "devices": [
#               { "device": "cuda:0", "prefill_tok_s": N, "decode_tok_s": N,
#                 "baseline_prefill_tok_s": N, "baseline_decode_tok_s": N } ] } ] }
# ---------------------------------------------------------------------------
RESULTS_DIR="${LLAMINAR_BENCHMARK_RESULTS_DIR:-${ROOT_DIR}/benchmark_results}"
COMMIT_HASH=$(git -C "$ROOT_DIR" rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
mkdir -p "${RESULTS_DIR}/${COMMIT_HASH}"
RESULTS_JSON="${RESULTS_DIR}/${COMMIT_HASH}/benchmark_results.json"

{
    echo "{"
    echo "  \"commit\": \"${COMMIT_HASH}\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"models\": ["
    FIRST_MODEL=true
    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
        MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")
        $FIRST_MODEL || echo ","
        FIRST_MODEL=false
        echo "    {"
        echo "      \"name\": $(jq -Rn --arg s "$MODEL_NAME" '$s'),"
        echo "      \"model\": $(jq -Rn --arg s "$MODEL" '$s'),"
        echo "      \"devices\": ["
        FIRST_DEV=true
        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            $FIRST_DEV || echo ","
            FIRST_DEV=false
            BL_P=$(jq -r ".models[$mi].devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
            BL_D=$(jq -r ".models[$mi].devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")
            CUR_P="${RESULTS_PREFILL[$KEY]:-null}"
            CUR_D="${RESULTS_DECODE[$KEY]:-null}"
            echo "        {"
            echo "          \"device\": \"${DEVICE}\","
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

# Also emit a flat CSV alongside the JSON so trend tooling can ingest the
# same per-commit history the same way the parity tests do.
RESULTS_CSV="${RESULTS_DIR}/${COMMIT_HASH}/benchmark_results.csv"
{
    echo "commit,timestamp,model_name,model_path,device,prefill_tok_s,decode_tok_s,baseline_prefill_tok_s,baseline_decode_tok_s"
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
        MODEL=$(jq -r ".models[$mi].model" "$BASELINE_FILE")
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")
        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            BL_P=$(jq -r ".models[$mi].devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
            BL_D=$(jq -r ".models[$mi].devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")
            CUR_P="${RESULTS_PREFILL[$KEY]:-}"
            CUR_D="${RESULTS_DECODE[$KEY]:-}"
            # Quote text columns that may contain commas/quotes.
            esc_name=$(printf '%s' "$MODEL_NAME" | sed 's/"/""/g')
            esc_model=$(printf '%s' "$MODEL" | sed 's/"/""/g')
            echo "${COMMIT_HASH},${TS},\"${esc_name}\",\"${esc_model}\",${DEVICE},${CUR_P},${CUR_D},${BL_P},${BL_D}"
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
            if [[ -n "${RESULTS_PREFILL[$KEY]:-}" ]]; then
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

    # Threshold priority: per-device > per-model > global
    local device_threshold model_threshold effective_threshold
    device_threshold=$(jq -r ".models[$model_idx].devices[\"$device\"].regression_threshold_pct // empty" "$BASELINE_FILE")
    model_threshold=$(jq -r ".models[$model_idx].regression_threshold_pct // empty" "$BASELINE_FILE")
    effective_threshold="${device_threshold:-${model_threshold:-$GLOBAL_THRESHOLD_PCT}}"

    local delta
    delta=$(echo "scale=4; ($current - $baseline) / $baseline * 100" | bc -l)
    delta=$(printf "%.1f" "$delta")

    local status="${GREEN}✓ OK${NC}"
    if (( $(echo "$delta < -${effective_threshold}" | bc -l) )); then
        status="${RED}✗ REGRESSED${NC}"
        OVERALL_PASS=false
        FAILED_CHECKS+="  [${model_name}] ${device} ${phase}: ${baseline} → ${current} tok/s (${delta}%, threshold ${effective_threshold}%)\n"
    elif (( $(echo "$delta < 0" | bc -l) )); then
        status="${YELLOW}~ slower${NC}"
    elif (( $(echo "$delta > 0" | bc -l) )); then
        status="${GREEN}▲ faster${NC}"
    fi

    printf "  %-10s  %-10s  %10.1f    %10.1f    %+6.1f%%  " "$device" "$phase" "$baseline" "$current" "$delta"
    echo -e "$status"
}

for (( mi=0; mi<NUM_MODELS; mi++ )); do
    MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
    DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

    echo -e "${BOLD}${MODEL_NAME}:${NC}"
    printf "  %-10s  %-10s  %12s  %12s  %8s  %s\n" "Device" "Phase" "Baseline" "Current" "Delta" "Status"
    printf "  %-10s  %-10s  %12s  %12s  %8s  %s\n" "------" "------" "--------" "-------" "-----" "------"

    for DEVICE in $DEVICES; do
        KEY="${mi}:${DEVICE}"
        if [[ -z "${RESULTS_PREFILL[$KEY]:-}" ]]; then
            printf "  %-10s  %-10s  %12s  %12s  %8s  " "$DEVICE" "prefill" "-" "-" "-"
            echo -e "${RED}FAILED${NC}"
            printf "  %-10s  %-10s  %12s  %12s  %8s  " "$DEVICE" "decode" "-" "-" "-"
            echo -e "${RED}FAILED${NC}"
            continue
        fi

        BASELINE_PREFILL=$(jq -r ".models[$mi].devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
        BASELINE_DECODE=$(jq -r ".models[$mi].devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")

        check_regression "$mi" "$MODEL_NAME" "$DEVICE" "prefill" "$BASELINE_PREFILL" "${RESULTS_PREFILL[$KEY]}"
        check_regression "$mi" "$MODEL_NAME" "$DEVICE" "decode" "$BASELINE_DECODE" "${RESULTS_DECODE[$KEY]}"
    done

    echo ""
done

if $OVERALL_PASS; then
    echo -e "${GREEN}✓ No performance regressions detected${NC}"

    # ---------------------------------------------------------------------------
    # Ratchet: auto-raise baselines when a new high-water mark is reached
    # ---------------------------------------------------------------------------
    RATCHETED=false
    for (( mi=0; mi<NUM_MODELS; mi++ )); do
        MODEL_NAME=$(jq -r ".models[$mi].name" "$BASELINE_FILE")
        DEVICES=$(jq -r ".models[$mi].devices | keys[]" "$BASELINE_FILE")

        for DEVICE in $DEVICES; do
            KEY="${mi}:${DEVICE}"
            if [[ -z "${RESULTS_PREFILL[$KEY]:-}" ]]; then
                continue
            fi

            BASELINE_PREFILL=$(jq -r ".models[$mi].devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
            BASELINE_DECODE=$(jq -r ".models[$mi].devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")
            CUR_PREFILL="${RESULTS_PREFILL[$KEY]}"
            CUR_DECODE="${RESULTS_DECODE[$KEY]}"

            RAISE_PREFILL=false
            RAISE_DECODE=false
            if (( $(echo "$CUR_PREFILL > $BASELINE_PREFILL" | bc -l) )); then
                RAISE_PREFILL=true
            fi
            if (( $(echo "$CUR_DECODE > $BASELINE_DECODE" | bc -l) )); then
                RAISE_DECODE=true
            fi

            if $RAISE_PREFILL || $RAISE_DECODE; then
                NEW_PREFILL=$( $RAISE_PREFILL && echo "$CUR_PREFILL" || echo "$BASELINE_PREFILL" )
                NEW_DECODE=$( $RAISE_DECODE && echo "$CUR_DECODE" || echo "$BASELINE_DECODE" )
                COMMIT_HASH=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
                COMMENT="High-water mark set at commit ${COMMIT_HASH} on $(date +%Y-%m-%d). Note to agents: It is FORBIDDEN to update these thresholds without explicit human approval."

                DEVICE_THRESHOLD=$(jq -r ".models[$mi].devices[\"$DEVICE\"].regression_threshold_pct // empty" "$BASELINE_FILE")

                if [[ -n "$DEVICE_THRESHOLD" ]]; then
                    jq --argjson mi "$mi" \
                       --arg dev "$DEVICE" \
                       --argjson pf "$NEW_PREFILL" \
                       --argjson dc "$NEW_DECODE" \
                       --argjson thr "$DEVICE_THRESHOLD" \
                       --arg cmt "$COMMENT" \
                       '.models[$mi].devices[$dev].prefill_tok_s = $pf |
                        .models[$mi].devices[$dev].decode_tok_s = $dc |
                        .models[$mi].devices[$dev].regression_threshold_pct = $thr |
                        .models[$mi].devices[$dev]._comment = $cmt' \
                       "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
                else
                    jq --argjson mi "$mi" \
                       --arg dev "$DEVICE" \
                       --argjson pf "$NEW_PREFILL" \
                       --argjson dc "$NEW_DECODE" \
                       --arg cmt "$COMMENT" \
                       '.models[$mi].devices[$dev].prefill_tok_s = $pf |
                        .models[$mi].devices[$dev].decode_tok_s = $dc |
                        .models[$mi].devices[$dev]._comment = $cmt' \
                       "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
                fi

                RATCHETED=true
                DETAILS=""
                $RAISE_PREFILL && DETAILS+="prefill ${BASELINE_PREFILL}→${CUR_PREFILL}"
                $RAISE_PREFILL && $RAISE_DECODE && DETAILS+=", "
                $RAISE_DECODE && DETAILS+="decode ${BASELINE_DECODE}→${CUR_DECODE}"
                echo -e "  ${GREEN}▲ [${MODEL_NAME}] ${DEVICE}: ratcheted baseline (${DETAILS})${NC}"
            fi
        done
    done

    if $RATCHETED; then
        # `git add` is only meaningful in a working git checkout (e.g. the
        # pre-commit hook context). In CI we run inside a docker container
        # that doesn't have .git mounted, so skip silently.
        if git -C "$ROOT_DIR" rev-parse --git-dir &>/dev/null; then
            git -C "$ROOT_DIR" add "$BASELINE_FILE"
            echo ""
            echo -e "${GREEN}✓ Baseline ratcheted and staged for commit${NC}"
        else
            echo ""
            echo -e "${GREEN}✓ Baseline ratcheted (not in a git repo; skipping git add)${NC}"
        fi
    fi

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
