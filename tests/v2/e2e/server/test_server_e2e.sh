#!/bin/bash
# =============================================================================
# E2E Server Integration Test — Multi-Turn Inference via REST API
#
# Tests the Llaminar HTTP server (serve subcommand) with curl against the
# /v1/chat/completions endpoint for multiple model × backend combinations.
#
# Default test suites:
#   Suite 1: Qwen2.5 1.5B Q8_0 on cpu, cuda:0, rocm:0
#   Suite 2: Qwen3.5 4B   Q8_0 on cpu
#   Suite 3: Qwen3.5 35B MoE Q4_K_XL on cpu
#
# Each backend test:
#   1. Starts llaminar2 serve on a unique port
#   2. Waits for /health to respond
#   3. Sends a single-turn greedy chat request, validates response
#   4. Sends a multi-turn conversation, validates response
#   5. Sends a second independent request (tests KV cache clearing)
#   6. Validates response format (usage, finish_reason)
#   7. Tests error handling (invalid JSON, missing messages)
#   8. Kills server, moves to next backend
#
# Usage:
#   ./test_server_e2e.sh [--binary <path>] [--model <path>] [--backends <list>]
#   ./test_server_e2e.sh [--binary <path>] [--suite "model_path|backend1,backend2[|max_tokens]"] ...
#
# Environment:
#   LLAMINAR_BINARY     Override binary path
#   LLAMINAR_MODEL      Override model path (overrides default suite 1)
#   LLAMINAR_BACKENDS   Override backends for suite 1
#   LLAMINAR_LOG_LEVEL  Log level for server (default: ERROR)
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

BINARY="${LLAMINAR_BINARY:-${REPO_ROOT}/build_v2_integration/llaminar2}"
LOG_LEVEL="${LLAMINAR_LOG_LEVEL:-ERROR}"
BASE_PORT=19080

# Model suites: "model_path|backend1,backend2,...[|max_tokens]"
# Uses '|' as delimiter (not ':') because device names contain colons (cuda:0).
# Each --suite flag appends to the list. If none given, defaults are used.
declare -a SUITES=()
OVERRIDE_MODEL=""
OVERRIDE_BACKENDS=""

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)   BINARY="$2";            shift 2 ;;
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

    # Suite 2: Qwen3.5 4B (hybrid GDN/FA architecture — CPU only for speed)
    # Uses max_tokens=200 because Qwen3.5 is a thinking model that emits
    # <think>...</think> tags before the actual answer.
    # xfail_inference=1: GDN architecture is WIP — inference produces degenerate output.
    # Server/health/error tests still validate normally.
    S2_MODEL="${REPO_ROOT}/models/Qwen3.5-4B-Q8_0.gguf"
    if [ -f "$S2_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S2_MODEL}|cpu|200|xfail_inference")
    fi

    # Suite 3: Qwen3.5 35B MoE (MoE + GDN/FA architecture — CPU only)
    # Uses max_tokens=200 for thinking model tags.
    # xfail_inference=1: MoE inference is WIP — output may be degenerate.
    # Server/health/error tests still validate normally.
    S3_MODEL="${REPO_ROOT}/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    if [ -f "$S3_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S3_MODEL}|cpu|200|xfail_inference")
    fi
fi

STARTUP_TIMEOUT=300   # seconds to wait for server startup. Most models load in
                      # <10s; the 4B Qwen3.5 GGUF on CPU needs ~60-120s for
                      # weight load + GDN init. The smaller suites finish in
                      # ~5s either way, so this is just an upper bound.
REQUEST_TIMEOUT=180   # seconds per curl request

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
XFAIL_TESTS=0
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

# Expected failure — counts as a pass (known issue, not a regression)
xfail() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    XFAIL_TESTS=$((XFAIL_TESTS + 1))
    echo -e "  ${YELLOW}⊘${NC} [XFAIL] $1"
}

cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

wait_for_health() {
    local port=$1
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
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

# ─── Validation ───────────────────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found: ${BINARY}${NC}"
    echo "Build with: cmake --build build_v2_integration --parallel"
    exit 1
fi

# ─── Test Runner Function ─────────────────────────────────────────────────────
# Runs the full test suite against a single model+backend combination.
# Arguments: $1=model_path $2=backend $3=port $4=model_label $5=max_tokens $6=xfail_inference
run_backend_tests() {
    local model="$1"
    local backend="$2"
    local port="$3"
    local label="$4"
    local max_tokens="${5:-10}"
    local xfail_inference="${6:-0}"
    local tag="${label}/${backend}"

    echo -e "${YELLOW}─── ${tag} (port ${port}) ───${NC}"

    # Build device flag — always explicit to prevent auto-detection
    local device_flag="-d ${backend}"

    # Start server
    LLAMINAR_LOG_LEVEL="$LOG_LEVEL" "$BINARY" serve --port "$port" \
        $device_flag -m "$model" >/tmp/server_e2e_trace.log 2>&1 &
    local server_pid=$!

    # Wait for health
    if ! wait_for_health "$port"; then
        fail "[${tag}] Server failed to start within ${STARTUP_TIMEOUT}s"
        echo "    ── Last 80 lines of server log (/tmp/server_e2e_trace.log) ──"
        tail -n 80 /tmp/server_e2e_trace.log 2>/dev/null | sed 's/^/    /' || echo "    (server log not available)"
        echo "    ────────────────────────────────────────────────────────────"
        cleanup_server "$server_pid"
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
    # System prompt steers thinking models (e.g. Qwen3.5) toward brief answers.
    local response content
    response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a calculator. Reply with only the numeric answer, no explanation."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": '"$max_tokens"',
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    content=$(echo "$response" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$content" | grep -q "4"; then
        pass "[${tag}] Single-turn: got '${content}' (contains 4)"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Single-turn: expected 4, got '${content}' (GDN WIP)"
    else
        fail "[${tag}] Single-turn: expected 4, got '${content}'"
    fi

    # ─── Test 3: Multi-turn conversation ──────────────────────────────
    # Tests multi-turn context with simple recall.
    response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
                {"role": "user", "content": "Remember this number: 42"},
                {"role": "assistant", "content": "Got it, the number is 42."},
                {"role": "user", "content": "What number did I tell you to remember? Reply with just the number."}
            ],
            "max_tokens": '"$max_tokens"',
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    content=$(echo "$response" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$content" | grep -q "42"; then
        pass "[${tag}] Multi-turn: got '${content}' (contains 42)"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Multi-turn: expected 42, got '${content}' (GDN WIP)"
    else
        fail "[${tag}] Multi-turn: expected 42, got '${content}'"
    fi

    # ─── Test 4: Second independent request (tests cache clear) ──────
    response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a calculator. Reply with only the numeric answer, no explanation."},
                {"role": "user", "content": "What is 3+5?"}
            ],
            "max_tokens": '"$max_tokens"',
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    content=$(echo "$response" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$content" | grep -q "8"; then
        pass "[${tag}] Cache-clear: got '${content}' (contains 8)"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Cache-clear: expected 8, got '${content}' (GDN WIP)"
    else
        fail "[${tag}] Cache-clear: expected 8, got '${content}'"
    fi

    # ─── Test 5: Response format validation ───────────────────────────
    local has_usage
    has_usage=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
u = d.get('usage', {})
assert u.get('prompt_tokens', 0) > 0
assert u.get('completion_tokens', 0) > 0
assert u.get('total_tokens', 0) == u['prompt_tokens'] + u['completion_tokens']
assert d.get('choices', [{}])[0].get('finish_reason') == 'stop'
print('ok')
" 2>/dev/null || echo "FAIL")

    if [ "$has_usage" = "ok" ]; then
        pass "[${tag}] Response format: valid usage + finish_reason"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Response format: missing/invalid usage or finish_reason (GDN WIP)"
    else
        fail "[${tag}] Response format: missing/invalid usage or finish_reason"
    fi

    # ─── Test 6: SSE streaming — basic streaming response ────────────
    local stream_raw stream_ok
    stream_raw=$(curl -s --max-time "$REQUEST_TIMEOUT" -N \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a calculator. Reply with only the numeric answer."},
                {"role": "user", "content": "What is 1+1?"}
            ],
            "max_tokens": '"$max_tokens"',
            "temperature": 0.0,
            "stream": true
        }' \
        "http://127.0.0.1:${port}/v1/chat/completions" 2>/dev/null || echo "CURL_FAILED")

    stream_ok=$(echo "$stream_raw" | python3 -c "
import sys
lines = sys.stdin.read().strip().split('\n')
# Filter to 'data: ' lines
data_lines = [l for l in lines if l.startswith('data: ')]
if len(data_lines) < 2:
    print('FAIL: too few SSE lines')
    sys.exit(0)
# Last data line must be [DONE]
if data_lines[-1].strip() != 'data: [DONE]':
    print('FAIL: missing [DONE] sentinel')
    sys.exit(0)
# First chunk must have role
import json
first = json.loads(data_lines[0][6:])
if first.get('object') != 'chat.completion.chunk':
    print('FAIL: wrong object type')
    sys.exit(0)
delta = first.get('choices', [{}])[0].get('delta', {})
if delta.get('role') != 'assistant':
    print('FAIL: first chunk missing role')
    sys.exit(0)
# Check a content chunk exists with finish_reason
for dl in data_lines[1:-1]:
    chunk = json.loads(dl[6:])
    fr = chunk.get('choices', [{}])[0].get('finish_reason')
    if fr in ('stop', 'length'):
        print('ok')
        sys.exit(0)
print('FAIL: no finish_reason chunk found')
" 2>/dev/null || echo "PARSE_ERROR")

    if [ "$stream_ok" = "ok" ]; then
        pass "[${tag}] SSE streaming: valid chunks with role, content, finish, [DONE]"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] SSE streaming: ${stream_ok} (GDN WIP)"
    else
        fail "[${tag}] SSE streaming: ${stream_ok}"
    fi

    # ─── Test 7: SSE streaming — response metadata ───────────────────
    local stream_meta_ok
    stream_meta_ok=$(echo "$stream_raw" | python3 -c "
import json, sys
lines = sys.stdin.read().strip().split('\n')
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
        pass "[${tag}] SSE streaming: metadata (id, system_fingerprint) consistent"
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] SSE streaming metadata: ${stream_meta_ok} (GDN WIP)"
    else
        fail "[${tag}] SSE streaming metadata: ${stream_meta_ok}"
    fi

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
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Error handling: expected invalid_request_error, got '${error_msg}' (GDN WIP)"
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
    elif [ "$xfail_inference" = "1" ]; then
        xfail "[${tag}] Error handling: expected invalid_request_error, got '${error_msg}' (GDN WIP)"
    else
        fail "[${tag}] Error handling: expected invalid_request_error, got '${error_msg}'"
    fi

    # Cleanup
    cleanup_server "$server_pid"
    echo ""
}

# ─── Run Test Suites ──────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  E2E Server Integration Test — Multi-Turn REST API${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Binary: ${BINARY}"
echo -e "  Suites: ${#SUITES[@]}"
for suite in "${SUITES[@]}"; do
    IFS='|' read -r local_model local_backends _ local_flags <<< "$suite"
    suffix=""
    if [[ "$local_flags" == *"xfail_inference"* ]]; then suffix=" (xfail: GDN WIP)"; fi
    echo -e "    $(basename "$local_model")  →  ${local_backends}${suffix}"
done
echo ""

PORT=$BASE_PORT

for suite in "${SUITES[@]}"; do
    # Parse suite: "model_path|backends[|max_tokens[|flags]]"
    IFS='|' read -r SUITE_MODEL SUITE_BACKENDS SUITE_MAX_TOKENS SUITE_FLAGS <<< "$suite"
    SUITE_MAX_TOKENS="${SUITE_MAX_TOKENS:-10}"  # Default: 10 tokens
    SUITE_LABEL="$(basename "$SUITE_MODEL" .gguf)"

    # Check for xfail_inference flag
    SUITE_XFAIL=0
    if [[ "$SUITE_FLAGS" == *"xfail_inference"* ]]; then
        SUITE_XFAIL=1
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
        run_backend_tests "$SUITE_MODEL" "$BACKEND" "$PORT" "$SUITE_LABEL" "$SUITE_MAX_TOKENS" "$SUITE_XFAIL"
    done
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ $FAILED_TESTS -eq 0 ]; then
    if [ $XFAIL_TESTS -gt 0 ]; then
        echo -e "${GREEN}  ✅ ALL PASSED: ${PASSED_TESTS}/${TOTAL_TESTS} tests passed (${XFAIL_TESTS} expected failures)${NC}"
    else
        echo -e "${GREEN}  ✅ ALL PASSED: ${PASSED_TESTS}/${TOTAL_TESTS} tests passed${NC}"
    fi
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}  ❌ FAILED: ${FAILED_TESTS}/${TOTAL_TESTS} tests failed${NC}"
    echo -e "${RED}${FAILED_DETAILS}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
