#!/bin/bash
# =============================================================================
# E2E Server Integration Test — Multi-Turn Inference via REST API
#
# Tests the Llaminar HTTP server (--serve) with curl against the
# /v1/chat/completions endpoint for CPU, CUDA, and ROCm backends.
#
# Each backend test:
#   1. Starts llaminar2 --serve on a unique port
#   2. Waits for /health to respond
#   3. Sends a single-turn greedy chat request, validates response
#   4. Sends a multi-turn conversation, validates response
#   5. Sends a second independent request (tests KV cache clearing)
#   6. Kills server, moves to next backend
#
# Usage:
#   ./test_server_e2e.sh [--binary <path>] [--model <path>] [--backends <list>]
#
# Environment:
#   LLAMINAR_BINARY     Override binary path
#   LLAMINAR_MODEL      Override model path
#   LLAMINAR_BACKENDS   Comma-separated backends (default: cpu,cuda:0,rocm:0)
#   LLAMINAR_LOG_LEVEL  Log level for server (default: ERROR)
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

BINARY="${LLAMINAR_BINARY:-${REPO_ROOT}/build_v2_integration/llaminar2}"
MODEL="${LLAMINAR_MODEL:-${REPO_ROOT}/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
BACKENDS="${LLAMINAR_BACKENDS:-cpu,cuda:0,rocm:0}"
LOG_LEVEL="${LLAMINAR_LOG_LEVEL:-ERROR}"
BASE_PORT=19080

# Parse CLI flags (override env vars)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)  BINARY="$2";   shift 2 ;;
        --model)   MODEL="$2";    shift 2 ;;
        --backends) BACKENDS="$2"; shift 2 ;;
        --port)    BASE_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
STARTUP_TIMEOUT=30    # seconds to wait for server startup
REQUEST_TIMEOUT=120   # seconds per curl request

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

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model not found: ${MODEL}${NC}"
    exit 1
fi

# ─── Run Tests ────────────────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  E2E Server Integration Test — Multi-Turn REST API${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Binary:   ${BINARY}"
echo -e "  Model:    $(basename "$MODEL")"
echo -e "  Backends: ${BACKENDS}"
echo ""

IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"
PORT=$BASE_PORT

for BACKEND in "${BACKEND_LIST[@]}"; do
    BACKEND=$(echo "$BACKEND" | xargs)  # trim whitespace
    PORT=$((PORT + 1))

    echo -e "${YELLOW}─── Backend: ${BACKEND} (port ${PORT}) ───${NC}"

    # Build device flag
    DEVICE_FLAG=""
    if [ "$BACKEND" != "cpu" ]; then
        DEVICE_FLAG="-d ${BACKEND}"
    fi

    # Start server
    LLAMINAR_LOG_LEVEL="$LOG_LEVEL" "$BINARY" --no-mpi-bootstrap --serve --port "$PORT" \
        $DEVICE_FLAG -m "$MODEL" >/dev/null 2>&1 &
    SERVER_PID=$!

    # Wait for health
    if ! wait_for_health "$PORT"; then
        fail "[${BACKEND}] Server failed to start within ${STARTUP_TIMEOUT}s"
        cleanup_server "$SERVER_PID"
        continue
    fi
    pass "[${BACKEND}] Server started"

    # ─── Test 1: Health endpoint ──────────────────────────────────────
    HEALTH_RESPONSE=$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "CURL_FAILED")
    if echo "$HEALTH_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
        pass "[${BACKEND}] GET /health returns ok"
    else
        fail "[${BACKEND}] GET /health unexpected: ${HEALTH_RESPONSE}"
    fi

    # ─── Test 2: Single-turn greedy inference ─────────────────────────
    RESPONSE=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 10,
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    CONTENT=$(echo "$RESPONSE" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$CONTENT" | grep -q "4"; then
        pass "[${BACKEND}] Single-turn: got '${CONTENT}' (contains 4)"
    else
        fail "[${BACKEND}] Single-turn: expected 4, got '${CONTENT}'"
    fi

    # ─── Test 3: Multi-turn conversation ──────────────────────────────
    RESPONSE=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What is 5+5? Reply with just the number."}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    CONTENT=$(echo "$RESPONSE" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$CONTENT" | grep -q "10"; then
        pass "[${BACKEND}] Multi-turn: got '${CONTENT}' (contains 10)"
    else
        fail "[${BACKEND}] Multi-turn: expected 10, got '${CONTENT}'"
    fi

    # ─── Test 4: Second independent request (tests cache clear) ──────
    RESPONSE=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [{"role": "user", "content": "What is 3+5? Reply with just the number."}],
            "max_tokens": 10,
            "temperature": 0.0
        }' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo '{"error":"curl_failed"}')

    CONTENT=$(echo "$RESPONSE" | extract_content 2>/dev/null || echo "PARSE_ERROR")

    if echo "$CONTENT" | grep -q "8"; then
        pass "[${BACKEND}] Cache-clear: got '${CONTENT}' (contains 8)"
    else
        fail "[${BACKEND}] Cache-clear: expected 8, got '${CONTENT}'"
    fi

    # ─── Test 5: Response format validation ───────────────────────────
    HAS_USAGE=$(echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
u = d.get('usage', {})
assert u.get('prompt_tokens', 0) > 0
assert u.get('completion_tokens', 0) > 0
assert u.get('total_tokens', 0) == u['prompt_tokens'] + u['completion_tokens']
assert d.get('choices', [{}])[0].get('finish_reason') == 'stop'
print('ok')
" 2>/dev/null || echo "FAIL")

    if [ "$HAS_USAGE" = "ok" ]; then
        pass "[${BACKEND}] Response format: valid usage + finish_reason"
    else
        fail "[${BACKEND}] Response format: missing/invalid usage or finish_reason"
    fi

    # ─── Test 6: Error handling — invalid JSON ────────────────────────
    ERROR_RESPONSE=$(curl -s --max-time 5 -X POST \
        -H "Content-Type: application/json" \
        -d 'not valid json' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo '{}')

    ERROR_MSG=$(echo "$ERROR_RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('error', {}).get('type', ''))
" 2>/dev/null || echo "PARSE_ERROR")

    if [ "$ERROR_MSG" = "invalid_request_error" ]; then
        pass "[${BACKEND}] Error handling: invalid JSON returns 400"
    else
        fail "[${BACKEND}] Error handling: expected invalid_request_error, got '${ERROR_MSG}'"
    fi

    # ─── Test 7: Error handling — missing messages ────────────────────
    ERROR_RESPONSE=$(curl -s --max-time 5 -X POST \
        -H "Content-Type: application/json" \
        -d '{"max_tokens": 10}' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo '{}')

    ERROR_MSG=$(echo "$ERROR_RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('error', {}).get('type', ''))
" 2>/dev/null || echo "PARSE_ERROR")

    if [ "$ERROR_MSG" = "invalid_request_error" ]; then
        pass "[${BACKEND}] Error handling: missing messages returns 400"
    else
        fail "[${BACKEND}] Error handling: expected invalid_request_error, got '${ERROR_MSG}'"
    fi

    # Cleanup
    cleanup_server "$SERVER_PID"
    echo ""
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
