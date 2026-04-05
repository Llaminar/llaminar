#!/bin/bash
# Performance regression benchmark check for Llaminar pre-commit hook.
#
# Runs --benchmark on each device listed in the baseline JSON, parses
# prefill/decode tok/s, and fails if any metric regresses beyond the
# configured threshold.
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
# Read baseline
# ---------------------------------------------------------------------------
MODEL=$(jq -r '.model' "$BASELINE_FILE")
DECODE_TOKENS=$(jq -r '.decode_tokens' "$BASELINE_FILE")
THRESHOLD_PCT=$(jq -r '.regression_threshold_pct' "$BASELINE_FILE")
DEVICES=$(jq -r '.devices | keys[]' "$BASELINE_FILE")

MODEL_PATH="$ROOT_DIR/$MODEL"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}Error: Model not found: $MODEL_PATH${NC}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run benchmarks and collect results
# ---------------------------------------------------------------------------
declare -A RESULTS_PREFILL
declare -A RESULTS_DECODE
OVERALL_PASS=true
FAILED_CHECKS=""

parse_benchmark_output() {
    local output="$1"
    local prefill_tps decode_tps
    # Extract tok/s values from the benchmark table output
    prefill_tps=$(echo "$output" | grep -A4 "PREFILL" | grep "Throughput" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    decode_tps=$(echo "$output" | grep -A4 "DECODE" | grep "Throughput" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    echo "${prefill_tps:-0} ${decode_tps:-0}"
}

echo -e "${BLUE}Running performance benchmarks (model: $(basename "$MODEL"))${NC}"
echo -e "${BLUE}Default regression threshold: ${THRESHOLD_PCT}% (per-device overrides may apply)${NC}"
echo ""

for DEVICE in $DEVICES; do
    echo -ne "  Benchmarking ${BOLD}${DEVICE}${NC} ... "

    set +e
    BENCH_OUTPUT=$("$RELEASE_BIN" --benchmark -d "$DEVICE" -m "$MODEL_PATH" -n "$DECODE_TOKENS" 2>&1)
    BENCH_EXIT=$?
    set -e

    if [[ $BENCH_EXIT -ne 0 ]]; then
        echo -e "${YELLOW}SKIPPED (device unavailable or error)${NC}"
        continue
    fi

    read -r PREFILL DECODE <<< "$(parse_benchmark_output "$BENCH_OUTPUT")"

    if [[ "$PREFILL" == "0" || "$DECODE" == "0" ]]; then
        echo -e "${YELLOW}SKIPPED (could not parse output)${NC}"
        continue
    fi

    RESULTS_PREFILL[$DEVICE]=$PREFILL
    RESULTS_DECODE[$DEVICE]=$DECODE
    echo -e "prefill ${GREEN}${PREFILL}${NC} tok/s, decode ${GREEN}${DECODE}${NC} tok/s"
done

echo ""

# ---------------------------------------------------------------------------
# Update baseline mode
# ---------------------------------------------------------------------------
if $UPDATE_BASELINE; then
    echo -e "${YELLOW}Updating baseline file: $BASELINE_FILE${NC}"

    # Build new JSON
    NEW_DEVICES="{"
    FIRST=true
    for DEVICE in $DEVICES; do
        if [[ -n "${RESULTS_PREFILL[$DEVICE]:-}" ]]; then
            $FIRST || NEW_DEVICES+=","
            FIRST=false
            NEW_DEVICES+="\"$DEVICE\":{\"prefill_tok_s\":${RESULTS_PREFILL[$DEVICE]},\"decode_tok_s\":${RESULTS_DECODE[$DEVICE]}}"
        fi
    done
    NEW_DEVICES+="}"

    jq --argjson devs "$NEW_DEVICES" \
       --arg comment "Canonical performance baselines for Qwen 2.5 7B Instruct Q8_0 (596-token prefill, ${DECODE_TOKENS}-token decode). Generated $(date +%Y-%m-%d). Update with: .githooks/run_benchmark_check.sh --update-baseline" \
       '.devices = $devs | ._comment = $comment' \
       "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"

    echo -e "${GREEN}✓ Baseline updated${NC}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compare against baseline
# ---------------------------------------------------------------------------
echo -e "${BOLD}Performance Regression Check:${NC}"
echo ""
printf "  %-10s  %-10s  %12s  %12s  %8s  %s\n" "Device" "Phase" "Baseline" "Current" "Delta" "Status"
printf "  %-10s  %-10s  %12s  %12s  %8s  %s\n" "------" "------" "--------" "-------" "-----" "------"

check_regression() {
    local device="$1" phase="$2" baseline="$3" current="$4"

    if [[ "$current" == "0" || -z "$current" ]]; then
        return
    fi

    # Per-device threshold overrides the global default
    local device_threshold
    device_threshold=$(jq -r ".devices[\"$device\"].regression_threshold_pct // empty" "$BASELINE_FILE")
    local effective_threshold="${device_threshold:-$THRESHOLD_PCT}"

    # Calculate percentage change: (current - baseline) / baseline * 100
    local delta
    delta=$(echo "scale=4; ($current - $baseline) / $baseline * 100" | bc -l)
    # Truncate to 1 decimal for display
    delta=$(printf "%.1f" "$delta")
    local abs_delta
    abs_delta=$(echo "$delta" | tr -d '-')

    local status="${GREEN}✓ OK${NC}"
    # Negative delta means regression (slower)
    if (( $(echo "$delta < -${effective_threshold}" | bc -l) )); then
        status="${RED}✗ REGRESSED${NC}"
        OVERALL_PASS=false
        FAILED_CHECKS+="  ${device} ${phase}: ${baseline} → ${current} tok/s (${delta}%, threshold ${effective_threshold}%)\n"
    elif (( $(echo "$delta < 0" | bc -l) )); then
        status="${YELLOW}~ slower${NC}"
    elif (( $(echo "$delta > 0" | bc -l) )); then
        status="${GREEN}▲ faster${NC}"
    fi

    printf "  %-10s  %-10s  %10.1f    %10.1f    %+6.1f%%  " "$device" "$phase" "$baseline" "$current" "$delta"
    echo -e "$status"
}

for DEVICE in $DEVICES; do
    if [[ -z "${RESULTS_PREFILL[$DEVICE]:-}" ]]; then
        printf "  %-10s  %-10s  %12s  %12s  %8s  " "$DEVICE" "prefill" "-" "-" "-"
        echo -e "${YELLOW}SKIPPED${NC}"
        printf "  %-10s  %-10s  %12s  %12s  %8s  " "$DEVICE" "decode" "-" "-" "-"
        echo -e "${YELLOW}SKIPPED${NC}"
        continue
    fi

    BASELINE_PREFILL=$(jq -r ".devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
    BASELINE_DECODE=$(jq -r ".devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")

    check_regression "$DEVICE" "prefill" "$BASELINE_PREFILL" "${RESULTS_PREFILL[$DEVICE]}"
    check_regression "$DEVICE" "decode" "$BASELINE_DECODE" "${RESULTS_DECODE[$DEVICE]}"
done

echo ""

if $OVERALL_PASS; then
    echo -e "${GREEN}✓ No performance regressions detected${NC}"

    # ---------------------------------------------------------------------------
    # Ratchet: auto-raise baselines when a new high-water mark is reached
    # ---------------------------------------------------------------------------
    RATCHETED=false
    for DEVICE in $DEVICES; do
        if [[ -z "${RESULTS_PREFILL[$DEVICE]:-}" ]]; then
            continue
        fi

        BASELINE_PREFILL=$(jq -r ".devices[\"$DEVICE\"].prefill_tok_s" "$BASELINE_FILE")
        BASELINE_DECODE=$(jq -r ".devices[\"$DEVICE\"].decode_tok_s" "$BASELINE_FILE")
        CUR_PREFILL="${RESULTS_PREFILL[$DEVICE]}"
        CUR_DECODE="${RESULTS_DECODE[$DEVICE]}"

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

            # Preserve per-device threshold if it exists
            DEVICE_THRESHOLD=$(jq -r ".devices[\"$DEVICE\"].regression_threshold_pct // empty" "$BASELINE_FILE")

            if [[ -n "$DEVICE_THRESHOLD" ]]; then
                jq --arg dev "$DEVICE" \
                   --argjson pf "$NEW_PREFILL" \
                   --argjson dc "$NEW_DECODE" \
                   --argjson thr "$DEVICE_THRESHOLD" \
                   --arg cmt "$COMMENT" \
                   '.devices[$dev].prefill_tok_s = $pf |
                    .devices[$dev].decode_tok_s = $dc |
                    .devices[$dev].regression_threshold_pct = $thr |
                    .devices[$dev]._comment = $cmt' \
                   "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
            else
                jq --arg dev "$DEVICE" \
                   --argjson pf "$NEW_PREFILL" \
                   --argjson dc "$NEW_DECODE" \
                   --arg cmt "$COMMENT" \
                   '.devices[$dev].prefill_tok_s = $pf |
                    .devices[$dev].decode_tok_s = $dc |
                    .devices[$dev]._comment = $cmt' \
                   "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" && mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
            fi

            RATCHETED=true
            DETAILS=""
            $RAISE_PREFILL && DETAILS+="prefill ${BASELINE_PREFILL}→${CUR_PREFILL}"
            $RAISE_PREFILL && $RAISE_DECODE && DETAILS+=", "
            $RAISE_DECODE && DETAILS+="decode ${BASELINE_DECODE}→${CUR_DECODE}"
            echo -e "  ${GREEN}▲ ${DEVICE}: ratcheted baseline (${DETAILS})${NC}"
        fi
    done

    if $RATCHETED; then
        # Stage the updated baseline so it becomes part of this commit
        git -C "$ROOT_DIR" add "$BASELINE_FILE"
        echo ""
        echo -e "${GREEN}✓ Baseline ratcheted and staged for commit${NC}"
    fi

    exit 0
else
    echo -e "${RED}✗ Performance regression detected!${NC}"
    echo ""
    echo -e "${RED}Regressed metrics (>${THRESHOLD_PCT}% slower):${NC}"
    echo -e "$FAILED_CHECKS"
    echo -e "${YELLOW}If this is expected (e.g. correctness fix), update the baseline:${NC}"
    echo -e "${YELLOW}  .githooks/run_benchmark_check.sh --update-baseline${NC}"
    echo ""
    echo -e "${YELLOW}Or skip with: git commit --no-verify${NC}"
    exit 1
fi
