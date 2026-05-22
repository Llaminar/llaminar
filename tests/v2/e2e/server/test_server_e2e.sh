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
#   Suite 4: Qwen3.5 35B MoE Q4_K_XL on rocm:0
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
#   ./test_server_e2e.sh [--binary <path>] [--suite "model_path|backend1,backend2[|max_tokens]"] ...
#   LLAMINAR_E2E_LONG_CONTEXT=1 ./test_server_e2e.sh [options]
#
# Optional long-context mode runs only when the model path, basename, or label
# contains a parsed size >= LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B (default: 4B).
#
# Environment:
#   LLAMINAR_BINARY     Override binary path
#   LLAMINAR_MODEL      Override model path (overrides default suite 1)
#   LLAMINAR_BACKENDS   Override backends for suite 1
#   LLAMINAR_LOG_LEVEL  Log level for server (default: WARN; ERROR is promoted
#                       to WARN so this harness can catch warnings)
#   LLAMINAR_E2E_LOG_DIR Override per-case server log directory
#   LLAMINAR_E2E_LONG_CONTEXT Enable optional long-context checks (default: 0)
#   LLAMINAR_E2E_LONG_CONTEXT_TIER lite|full long-context tier (default: lite)
#   LLAMINAR_E2E_CONTEXT_LENGTH Context length passed with -c for eligible models (default: 4096)
#   LLAMINAR_E2E_LONG_MAX_TOKENS Long-generation max_tokens (default: 512)
#   LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS Minimum helper prompt tokens (default: 900)
#   LLAMINAR_E2E_LONG_REQUEST_TIMEOUT Long helper request timeout (default: REQUEST_TIMEOUT)
#   LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B Minimum parsed model size in billions (default: 4)
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

BINARY="${LLAMINAR_BINARY:-${REPO_ROOT}/build_v2_integration/llaminar2}"
LOG_LEVEL="${LLAMINAR_LOG_LEVEL:-WARN}"
if [[ "${LOG_LEVEL^^}" == "ERROR" ]]; then
    LOG_LEVEL="WARN"
fi
BASE_PORT=19080

HOST_RSS_CPU_MODEL_MULTIPLIER="${LLAMINAR_E2E_HOST_RSS_CPU_MODEL_MULTIPLIER:-4}"
HOST_RSS_GPU_MODEL_MULTIPLIER="${LLAMINAR_E2E_HOST_RSS_GPU_MODEL_MULTIPLIER:-2}"
HOST_RSS_EXTRA_MB="${LLAMINAR_E2E_HOST_RSS_EXTRA_MB:-4096}"
CPU_GPU_DELTA_LIMIT_MB="${LLAMINAR_E2E_CPU_GPU_DELTA_LIMIT_MB:-128}"
GPU_ACTIVE_MIN_MB="${LLAMINAR_E2E_GPU_ACTIVE_MIN_MB:-256}"
THINKING_BUDGET_TOKENS="${LLAMINAR_E2E_THINKING_BUDGET_TOKENS:-}"

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
    S2_MODEL="${REPO_ROOT}/models/Qwen3.5-4B-Q8_0.gguf"
    if [ -f "$S2_MODEL" ] && [ -z "$OVERRIDE_MODEL" ]; then
        SUITES+=("${S2_MODEL}|cpu|200")
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
fi

STARTUP_TIMEOUT=300   # seconds to wait for server startup. Most models load in
                      # <10s; the 4B Qwen3.5 GGUF on CPU needs ~60-120s for
                      # weight load + GDN init. The smaller suites finish in
                      # ~5s either way, so this is just an upper bound.
REQUEST_TIMEOUT=180   # seconds per curl request

# Optional long-context helper controls. The helper is intentionally gated to
# 4B+ models by default so small smoke-test suites keep their fast behavior.
LONG_CONTEXT_ENABLED="${LLAMINAR_E2E_LONG_CONTEXT:-0}"
LONG_CONTEXT_TIER="${LLAMINAR_E2E_LONG_CONTEXT_TIER:-lite}"
CONTEXT_LENGTH="${LLAMINAR_E2E_CONTEXT_LENGTH:-4096}"
LONG_MAX_TOKENS="${LLAMINAR_E2E_LONG_MAX_TOKENS:-512}"
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

sanitize_name() {
    echo "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

is_thinking_model() {
    local label="$1"
    [[ "${label,,}" == *"qwen3.5"* || "${label,,}" == *"qwen35"* ]]
}

is_gpu_backend() {
    local backend="$1"
    [[ "$backend" == cuda:* || "$backend" == rocm:* ]]
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

    if python3 "$SCRIPT_DIR/long_context_checks.py" \
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
    else
        pass "[${tag}] Server log: no WARN/ERROR entries (${log_path})"
    fi
}

check_memory_usage() {
    local tag="$1"
    local backend="$2"
    local model="$3"
    local server_pid="$4"
    local gpu_before_mb="$5"

    local pids
    pids=$(collect_process_tree_pids "$server_pid" | sort -n | uniq | xargs || true)
    if [ -z "$pids" ]; then
        fail "[${tag}] Memory: server process exited before measurement"
        return
    fi

    local ram_mb gpu_process_mb gpu_after_mb gpu_delta_mb abs_gpu_delta_mb model_mb rss_multiplier rss_limit_mb
    ram_mb=$(get_process_tree_ram_mb "$pids")
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

# ─── Validation ───────────────────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found: ${BINARY}${NC}"
    echo "Build with: cmake --build build_v2_integration --parallel"
    exit 1
fi

BINARY_ABS="$(readlink -f "$BINARY" 2>/dev/null || echo "$BINARY")"
BINARY_DIR="$(dirname "$BINARY_ABS")"
LOG_DIR="${LLAMINAR_E2E_LOG_DIR:-${BINARY_DIR}/e2e_server_logs}"
mkdir -p "$LOG_DIR"

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

# ─── Test Runner Function ─────────────────────────────────────────────────────
# Runs the full test suite against a single model+backend combination.
# Arguments: $1=model_path $2=backend $3=port $4=model_label $5=max_tokens
run_backend_tests() {
    local model="$1"
    local backend="$2"
    local port="$3"
    local label="$4"
    local max_tokens="${5:-10}"
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
        if model_size_meets_threshold "$model_size_b" "$LONG_MIN_MODEL_SIZE_B"; then
            long_context_run="true"
            echo -e "  ${BLUE}INFO${NC} [${tag}] Long-context enabled: size ${model_size_b}B, tier ${LONG_CONTEXT_TIER}, context ${CONTEXT_LENGTH}"
        else
            print_long_context_gate "$tag" "$model_size_b"
        fi
    fi

    # Build device flag — always explicit to prevent auto-detection
    local device_flag="-d ${backend}"
    local safe_tag log_path gpu_before_mb
    safe_tag=$(sanitize_name "$tag")
    log_path="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${safe_tag}_port${port}.log"
    gpu_before_mb=$(get_total_gpu_memory_mb)

    # Start server
    local context_args=()
    if [ "$long_context_run" = "true" ]; then
        context_args=(-c "$CONTEXT_LENGTH")
    fi

    LLAMINAR_LOG_LEVEL="$LOG_LEVEL" "$BINARY" serve --port "$port" \
        "${context_args[@]}" $device_flag -m "$model" >"$log_path" 2>&1 &
    local server_pid=$!

    # Wait for health
    if ! wait_for_health "$port"; then
        fail "[${tag}] Server failed to start within ${STARTUP_TIMEOUT}s"
        echo "    ── Last 80 lines of server log (${log_path}) ──"
        tail -n 80 "$log_path" 2>/dev/null | sed 's/^/    /' || echo "    (server log not available)"
        echo "    ────────────────────────────────────────────────────────────"
        scan_server_log "$tag" "$log_path"
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

    # ─── Test 5: Response format validation ───────────────────────────
    local has_usage
    has_usage=$(printf '%s' "$LAST_RESPONSE" | validate_chat_response_format)

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

    # ─── Optional Long-Context Checks ─────────────────────────────────
    if [ "$long_context_run" = "true" ]; then
        run_long_context_checks "$tag" "$port" "$thinking_model"
    fi

    # ─── Test 10: Memory and server log hygiene ───────────────────────
    check_memory_usage "$tag" "$backend" "$model" "$server_pid" "$gpu_before_mb"
    scan_server_log "$tag" "$log_path"

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
echo -e "  Server logs: ${LOG_DIR}"
echo -e "  Server log level: ${LOG_LEVEL}"
if [ "$LONG_CONTEXT_ENABLED" = "1" ]; then
    echo -e "  Long-context: enabled, tier=${LONG_CONTEXT_TIER}, context=${CONTEXT_LENGTH}, max_tokens=${LONG_MAX_TOKENS}, min_model_size=${LONG_MIN_MODEL_SIZE_B}B"
else
    echo -e "  Long-context: disabled"
fi
echo -e "  Suites: ${#SUITES[@]}"
for suite in "${SUITES[@]}"; do
    IFS='|' read -r local_model local_backends _ <<< "$suite"
    echo -e "    $(basename "$local_model")  →  ${local_backends}"
done
echo ""

PORT=$BASE_PORT

for suite in "${SUITES[@]}"; do
    # Parse suite: "model_path|backends[|max_tokens]"
    IFS='|' read -r SUITE_MODEL SUITE_BACKENDS SUITE_MAX_TOKENS <<< "$suite"
    SUITE_MAX_TOKENS="${SUITE_MAX_TOKENS:-10}"  # Default: 10 tokens
    SUITE_LABEL="$(basename "$SUITE_MODEL" .gguf)"

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
        run_backend_tests "$SUITE_MODEL" "$BACKEND" "$PORT" "$SUITE_LABEL" "$SUITE_MAX_TOKENS"
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
