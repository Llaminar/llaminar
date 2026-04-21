#!/usr/bin/env bash
# =============================================================================
# run_parity_slots.sh — run multiple ctest invocations in parallel.
#
# Used by .github/workflows/ci.yml to split the parity test matrix into
# parallel "slots" that exercise disjoint sets of tests, where each set:
#
#   - writes to a disjoint per-test results subdirectory
#     (tests/v2/integration/parity/results/<git-hash>/<test-name>/), so
#     concurrent CSV writes never collide
#
#   - uses at most ONE GPU type (CPU-only, or pure-CUDA, or pure-ROCm)
#     so two concurrent slots never contend for the same device
#
# Invocation:
#
#   run_parity_slots.sh NAME1 INCLUDE1 EXCLUDE1  [NAME2 INCLUDE2 EXCLUDE2 ...]
#
# Each slot uses THREE positional arguments:
#
#   NAME     Slot label (used in log filenames and section headers).
#   INCLUDE  Passed to `ctest -R`. Selects tests for this slot.
#   EXCLUDE  Passed to `ctest -E`. Removes tests that should not run in
#            this slot (e.g. a CUDA-slot excludes tests that also mention
#            ROCm so heterogeneous tests don't land here).
#            Pass "__never_match__" if you don't want any exclusions.
#
# Each slot runs as a background ctest invocation. Its output is streamed
# to /tmp/parity-slots/<name>.log. After all slots finish, every log file
# is printed with a header. Exits non-zero iff any slot exited non-zero,
# propagating the first non-zero exit code seen.
#
# Example (three CPU slots across Qwen families, run inside one container):
#
#   run_parity_slots.sh \
#     qwen2-cpu  '^V2_Integration_Parity_(Qwen2|NodeLocalTP_Qwen2)_'  'CUDA|ROCm' \
#     qwen3-cpu  '^V2_Integration_Parity_Qwen3_'                      'CUDA|ROCm' \
#     qwen35-cpu '^V2_Integration_Parity_Qwen35_'                     'CUDA|ROCm'
#
# Environment knobs:
#   BUILD_DIR       Build tree to run ctest in. Default: build_v2_integration
#   LOG_DIR         Per-slot log directory. Default: /tmp/parity-slots
#   TIMEOUT_SECS    Per-test timeout passed to ctest --timeout. Default: 1800
#   TEST_PARALLEL   Extra parallelism flags for each ctest invocation (rarely
#                   useful since slots are already parallel). Default: unset.
# =============================================================================

set -u

BUILD_DIR="${BUILD_DIR:-build_v2_integration}"
LOG_DIR="${LOG_DIR:-/tmp/parity-slots}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
TEST_PARALLEL="${TEST_PARALLEL:-}"

mkdir -p "$LOG_DIR"

if (( $# < 3 )) || (( $# % 3 != 0 )); then
  echo "usage: $0 NAME INCLUDE EXCLUDE [NAME INCLUDE EXCLUDE ...]" >&2
  echo "got $# args; must be a positive multiple of 3" >&2
  exit 2
fi

declare -a SLOT_NAMES=()
declare -a SLOT_PIDS=()
declare -a SLOT_LOGS=()

# Kick off each slot (3 args per slot). Each slot's output goes to its own
# log file; a single `tail -F` below streams every log to stdout so the
# GitHub Actions UI shows live per-test progress while the slots run.
while (( $# >= 3 )); do
  name="$1"
  include="$2"
  exclude="$3"
  shift 3

  if [[ -z "$name" || -z "$include" ]]; then
    echo "::error::Invalid slot: NAME and INCLUDE must be non-empty" >&2
    exit 2
  fi

  log="$LOG_DIR/${name}.log"
  # Pre-create the log file so `tail -F` can attach immediately (avoids a
  # race where tail spins waiting for the file to appear).
  : > "$log"

  echo ":: starting slot '${name}' (log=${log})"
  echo ":::: include: ${include}"
  echo ":::: exclude: ${exclude}"

  # Header is written to the log, not stdout, so `tail -F` streams it as
  # part of the slot's live output (complete with the `==> log <==` banner
  # that tail emits when switching between files).
  {
    echo "=== SLOT ${name} BEGIN ==="
    echo "include: ${include}"
    echo "exclude: ${exclude}"
    echo "build:   ${BUILD_DIR}"
    echo "=============================="
  } >> "$log"

  # stdbuf -oL forces ctest to line-buffer its stdout; ctest's default when
  # stdout is a regular file is full-block-buffered, which is what caused
  # the previous "no live progress" symptom.
  #
  # shellcheck disable=SC2086  # TEST_PARALLEL may be empty or "-j 8"
  (
    stdbuf -oL ctest \
      --test-dir "$BUILD_DIR" \
      --output-on-failure \
      --progress \
      --timeout "$TIMEOUT_SECS" \
      ${TEST_PARALLEL} \
      -R "$include" \
      -E "$exclude"
    rc=$?
    echo "=== SLOT ${name} END (exit=$rc) ==="
    exit "$rc"
  ) >> "$log" 2>&1 &

  SLOT_NAMES+=("$name")
  SLOT_PIDS+=("$!")
  SLOT_LOGS+=("$log")
done

# Stream all slot logs live. `tail -F` retries on missing/truncated files,
# prints `==> file <==` banners when switching between files, and is
# line-buffered when writing to a pipe. We start from the top of each file
# (-n +1) so the slot headers emitted above are included in the stream.
#
# The tail runs in the background so the script can still wait on the
# slot PIDs and collect exit codes. It is stopped explicitly after all
# slots complete.
tail -n +1 -F "${SLOT_LOGS[@]}" &
TAIL_PID=$!
# Make sure we always clean up the tail, even on abnormal exit.
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

# Wait for all slots and collect exit codes.
declare -A SLOT_RC=()
overall_rc=0

for i in "${!SLOT_PIDS[@]}"; do
  pid="${SLOT_PIDS[$i]}"
  name="${SLOT_NAMES[$i]}"
  if wait "$pid"; then
    SLOT_RC[$name]=0
  else
    rc=$?
    SLOT_RC[$name]=$rc
    if [[ $overall_rc -eq 0 ]]; then
      overall_rc=$rc
    fi
  fi
done

# Give tail a brief moment to flush any trailing lines written between the
# last tail poll interval and now, then stop it.
sleep 1
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
trap - EXIT

# All slot output already streamed live via `tail -F`. Re-emit the full log
# of any failed slot inside a GitHub Actions collapsible group so the
# failure context is easy to locate without scrolling back through the
# interleaved live stream.
for name in "${SLOT_NAMES[@]}"; do
  rc=${SLOT_RC[$name]}
  if [[ $rc -ne 0 ]]; then
    echo ""
    echo "::group::slot ${name} (exit=${rc}) — full log"
    cat "$LOG_DIR/${name}.log" || true
    echo "::endgroup::"
  fi
done

echo ""
echo "##################################################"
echo "# parallel parity summary"
echo "##################################################"
for name in "${SLOT_NAMES[@]}"; do
  rc=${SLOT_RC[$name]}
  if [[ $rc -eq 0 ]]; then
    printf "  %-20s  OK\n" "$name"
  else
    printf "  %-20s  FAILED (exit=%d)\n" "$name" "$rc"
  fi
done

exit "$overall_rc"
