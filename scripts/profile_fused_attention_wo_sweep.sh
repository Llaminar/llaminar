#!/usr/bin/env bash
set -euo pipefail

# Sweep fused attention(+Wo) microbench under perf stat.
#
# Examples:
#   scripts/profile_fused_attention_wo_sweep.sh \
#     --bin build_v2_release/tests/v2/v2_microbench_fused_attention_wo \
#     --cores 0-27 --threads 28 --reps 2 \
#     --cases qwen32b_decode_kv512,qwen32b_decode_kv2048,qwen32b_decode_kv4096,qwen32b_decode_kv8192 \
#     --wo q8_1,q8_1_vnni_packed \
#     --out /tmp/fused_attention_sweep.csv

BIN=""
CORES=""
THREADS=""
REPS=2
CASES=""
WO="q8_1,q8_1_vnni_packed"
FA="fa2"
OUT="/tmp/fused_attention_sweep.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --cores) CORES="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --cases) CASES="$2"; shift 2;;
    --wo) WO="$2"; shift 2;;
    --fa) FA="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# *//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BIN" || -z "$CORES" || -z "$THREADS" || -z "$CASES" ]]; then
  echo "Missing required args. Use --help." >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 2
fi

# perf -x, output columns: value,unit,event,run_time,percent,metric_id
# We'll prepend our own metadata columns.
EVENTS=(
  cache-references cache-misses
  L1-dcache-loads L1-dcache-load-misses
  LLC-loads LLC-load-misses
  l2_rqsts.references l2_rqsts.miss
)
EVENTS_CSV=$(IFS=, ; echo "${EVENTS[*]}")

# Conservative iteration counts tuned to ~0.5-1.5s on large models.
# You can override by editing this map.
case_iters() {
  local name="$1"
  case "$name" in
    *kv512) echo 200;;
    *kv2048) echo 60;;
    *kv4096) echo 45;;
    *kv8192) echo 25;;
    *) echo 60;;
  esac
}

case_warmup() {
  local name="$1"
  case "$name" in
    *kv4096|*kv8192) echo 3;;
    *) echo 5;;
  esac
}

mkdir -p "$(dirname "$OUT")"

echo "case,fa,wo,cores,threads,reps,per_iter_ms,event,value,unit" > "$OUT"

echo "Sweeping: cases=$CASES wo=$WO fa=$FA threads=$THREADS cores=$CORES reps=$REPS" >&2
echo "Output: $OUT" >&2

IFS=, read -r -a CASE_ARR <<< "$CASES"
IFS=, read -r -a WO_ARR <<< "$WO"

for c in "${CASE_ARR[@]}"; do
  iters=$(case_iters "$c")
  warmup=$(case_warmup "$c")

  for wo in "${WO_ARR[@]}"; do
    # Capture per_iter_ms from stdout for convenience.
    per_iter_ms=$(taskset -c "$CORES" env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      "$BIN" --case="$c" --fa="$FA" --wo="$wo" --warmup="$warmup" --iters="$iters" \
      | awk -F= '/per_iter_ms=/{print $2}' | tail -n1)

    tmp=$(mktemp)
    taskset -c "$CORES" env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      perf stat -x, -r "$REPS" -e "$EVENTS_CSV" -o "$tmp" \
      "$BIN" --case="$c" --fa="$FA" --wo="$wo" --warmup="$warmup" --iters="$iters" \
      >/dev/null

    # perf emits comment/header lines; keep only event lines.
    # Columns: value,unit,event,run_time,percent,metric_id
    while IFS=, read -r value unit event _rest; do
      [[ -z "$event" ]] && continue
      [[ "$event" == "#"* ]] && continue
      [[ "$value" == "" ]] && continue
      echo "$c,$FA,$wo,$CORES,$THREADS,$REPS,$per_iter_ms,$event,$value,$unit" >> "$OUT"
    done < "$tmp"

    rm -f "$tmp"
    echo "Done: $c $wo per_iter_ms=$per_iter_ms" >&2
  done
done

echo "Sweep complete: $OUT" >&2
