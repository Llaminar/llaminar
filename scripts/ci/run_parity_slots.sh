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

# Kick off each slot (3 args per slot)
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
  echo ":: starting slot '${name}'"
  echo ":::: include: ${include}"
  echo ":::: exclude: ${exclude}"

  # shellcheck disable=SC2086  # TEST_PARALLEL may be empty or "-j 8"
  (
    echo "=== SLOT ${name} BEGIN ==="
    echo "include: ${include}"
    echo "exclude: ${exclude}"
    echo "build:   ${BUILD_DIR}"
    echo "=============================="
    ctest \
      --test-dir "$BUILD_DIR" \
      --output-on-failure \
      --timeout "$TIMEOUT_SECS" \
      ${TEST_PARALLEL} \
      -R "$include" \
      -E "$exclude"
    rc=$?
    echo "=== SLOT ${name} END (exit=$rc) ==="
    exit "$rc"
  ) >"$log" 2>&1 &

  SLOT_NAMES+=("$name")
  SLOT_PIDS+=("$!")
done

# Wait for all slots and collect exit codes
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

# Emit all logs, then a summary block
for name in "${SLOT_NAMES[@]}"; do
  echo ""
  echo "##################################################"
  echo "# slot: ${name}  (exit=${SLOT_RC[$name]})"
  echo "##################################################"
  cat "$LOG_DIR/${name}.log" || true
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
# Each slot is one argument of the form:
#
#   name|include-regex|exclude-regex
#
#   name           Slot label (used in log filenames and section headers).
#   include-regex  Passed to `ctest -R`. Selects tests for this slot.
#   exclude-regex  Passed to `ctest -E`. Removes tests that should not run in
#                  this slot (e.g. a CUDA-slot excludes tests that also
#                  mention ROCm so heterogeneous tests don't land here).
#                  Use "__never_match__" if you don't want any exclusions.
#
# Behaviour:
#   - Each slot runs as a background ctest invocation. Its output is streamed
#     to /tmp/parity-slots/<name>.log.
#   - After all slots finish, every log file is printed (delimited so you can
#     find each slot's output).
#   - Exits non-zero iff any slot exits non-zero, propagating the first
#     non-zero exit code seen.
#
# Example (three CPU slots across Qwen families, in one container):
#
#   run_parity_slots.sh \
#     "qwen2-cpu|^V2_Integration_Parity_(Qwen2|NodeLocalTP_Qwen2)_|CUDA|ROCm" \
#     "qwen3-cpu|^V2_Integration_Parity_Qwen3_|CUDA|ROCm" \
#     "qwen35-cpu|^V2_Integration_Parity_Qwen35_|CUDA|ROCm"
#
# Notes:
#   - The include and exclude regexes are POSIX extended regexes. `|` in a
#     regex is NOT a field separator; we split on the FIRST TWO `|` only.
#     Anything after the second `|` is the exclude regex.
#   - The script does not manage ctest's own -j parallelism. Pass --parallel
#     or -jN via the TEST_PARALLEL env var if desired (default: unset so
#     ctest runs its slot serially). Running slots in parallel already
#     saturates the host; intra-slot parallelism is usually not needed.
# =============================================================================

set -u

BUILD_DIR="${BUILD_DIR:-build_v2_integration}"
LOG_DIR="${LOG_DIR:-/tmp/parity-slots}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
TEST_PARALLEL="${TEST_PARALLEL:-}"

mkdir -p "$LOG_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 'name1|include1|exclude1' ['name2|include2|exclude2' ...]" >&2
  exit 2
fi

declare -a SLOT_NAMES=()
declare -a SLOT_PIDS=()

# Kick off each slot
for spec in "$@"; do
  # Split on the first TWO pipe characters only; the exclude regex may itself
  # contain pipes.
  name="${spec%%|*}"
  rest="${spec#*|}"
  include="${rest%%|*}"
  exclude="${rest#*|}"

  if [[ -z "$name" || -z "$include" ]]; then
    echo "::error::Invalid slot spec (expected 'name|include|exclude'): $spec" >&2
    exit 2
  fi

  log="$LOG_DIR/${name}.log"
  echo ":: starting slot '${name}'"
  echo ":::: include: ${include}"
  echo ":::: exclude: ${exclude}"

  # shellcheck disable=SC2086  # TEST_PARALLEL may be empty or "-j 8"
  (
    echo "=== SLOT ${name} BEGIN ==="
    echo "include: ${include}"
    echo "exclude: ${exclude}"
    echo "build:   ${BUILD_DIR}"
    echo "=============================="
    ctest \
      --test-dir "$BUILD_DIR" \
      --output-on-failure \
      --timeout "$TIMEOUT_SECS" \
      ${TEST_PARALLEL} \
      -R "$include" \
      -E "$exclude"
    rc=$?
    echo "=== SLOT ${name} END (exit=$rc) ==="
    exit "$rc"
  ) >"$log" 2>&1 &

  SLOT_NAMES+=("$name")
  SLOT_PIDS+=("$!")
done

# Wait for all slots and collect exit codes
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

# Emit all logs, then a summary block
for name in "${SLOT_NAMES[@]}"; do
  echo ""
  echo "##################################################"
  echo "# slot: ${name}  (exit=${SLOT_RC[$name]})"
  echo "##################################################"
  cat "$LOG_DIR/${name}.log" || true
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
