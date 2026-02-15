#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspaces/llaminar"
BIN="$ROOT/build_v2_release/llaminar2"
MODEL="$ROOT/models/qwen2.5-0.5b-instruct-q4_0.gguf"
PROMPT="What is the capital of France?"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not found: $BIN" >&2
  exit 1
fi

START_LAYER="${1:-0}"
END_LAYER="${2:-3}"
MODE="${3:-ffn}"
TOKENS="${TOKENS:-4}"

for ((L=START_LAYER; L<=END_LAYER; L++)); do
  BASE_LOG="/tmp/stage_l${L}_base.log"
  OPT_LOG="/tmp/stage_l${L}_optin.log"

    if [[ "$MODE" == "attn" ]]; then
        STAGES="layer${L}_attn_norm,layer${L}_qkv_proj,layer${L}_rope,layer${L}_attention,layer${L}_wo_proj"
    else
        STAGES="layer${L}_attn_residual,layer${L}_ffn_norm,layer${L}_gate_up_proj,layer${L}_swiglu,layer${L}_down_proj"
    fi

  echo "[L${L}] running baseline..."
  LLAMINAR_GPU_GRAPHS=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT_N=8 \
  LLAMINAR_STAGE_OUTPUT_PRINT_ROWS=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT_STAGES="$STAGES" \
  LLAMINAR_LOG_LEVEL=INFO \
  timeout 90 mpirun --oversubscribe -np 2 "$BIN" \
    --tp-scope global -tp 2 -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
    > "$BASE_LOG" 2>&1 || true

  echo "[L${L}] running opt-in segmented..."
  LLAMINAR_GPU_GRAPHS=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT_N=8 \
  LLAMINAR_STAGE_OUTPUT_PRINT_ROWS=1 \
  LLAMINAR_STAGE_OUTPUT_PRINT_STAGES="$STAGES" \
  LLAMINAR_LOG_LEVEL=INFO \
  timeout 90 mpirun --oversubscribe -np 2 "$BIN" \
    --tp-scope global -tp 2 -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
    > "$OPT_LOG" 2>&1 || true

  echo "[L${L}] comparing stage outputs..."
  python3 - <<PY
import re
from collections import OrderedDict

layer = ${L}
mode = "${MODE}"
base_path = "${BASE_LOG}"
opt_path = "${OPT_LOG}"

if mode == "attn":
    pat = re.compile(r"\\[StageOutput\\]\\s+(layer%s_(?:attn_norm|qkv_proj|rope|attention|wo_proj))/([^\\s]+)\\s+\\[[^\\]]+\\]\\s+row\\[0\\]:\\s*(.*)$" % layer)
else:
    pat = re.compile(r"\\[StageOutput\\]\\s+(layer%s_(?:attn_residual|ffn_norm|gate_up_proj|swiglu|down_proj))/([^\\s]+)\\s+\\[[^\\]]+\\]\\s+row\\[0\\]:\\s*(.*)$" % layer)

def first_by_key(path):
    d = OrderedDict()
    try:
        f = open(path, "r", errors="ignore")
    except FileNotFoundError:
        return d
    with f:
        for ln in f:
            m = pat.search(ln)
            if not m:
                continue
            key = (m.group(1), m.group(2))
            d.setdefault(key, m.group(3).strip())
    return d

b = first_by_key(base_path)
o = first_by_key(opt_path)
keys = sorted(set(b) | set(o))

if not keys:
    print(f"[L{layer}] no stage-output entries found")
else:
    mismatch = None
    for k in keys:
        bv = b.get(k)
        ov = o.get(k)
        if bv != ov:
            mismatch = (k, bv, ov)
            break

    if mismatch is None:
        print(f"[L{layer}] MATCH ({len(keys)} keys, mode={mode})")
    else:
        k, bv, ov = mismatch
        print(f"[L{layer}] DIFF at {k[0]}/{k[1]} (mode={mode})")
        print("  BASE:", (bv or "<missing>")[:180])
        print("  OPT :", (ov or "<missing>")[:180])

for path, label in [(base_path, "BASE"), (opt_path, "OPT")]:
    try:
        with open(path, "r", errors="ignore") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = ""
    flag = ".Names" in txt
    print(f"[L{layer}] {label} has '.Names' corruption marker: {flag}")
PY

  echo "[L${L}] logs: $BASE_LOG | $OPT_LOG"
  echo
done
