#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mtp_depth_hysteresis_sweep.sh --model MODEL [options] [-- extra llaminar2 benchmark args...]
  scripts/run_mtp_depth_hysteresis_sweep.sh MODEL [options] [-- extra llaminar2 benchmark args...]

Runs reusable MTP depth-policy benchmark sweeps and writes one benchmark JSON/log
per case/variant plus a machine-readable summary.tsv.

Options:
  --model PATH       GGUF model path. Also accepted as the first positional arg.
  --binary PATH      llaminar2 binary (default: build_v2_release/llaminar2).
  --device SPEC      Device spec (default: rocm:0).
  --output-dir DIR   Output directory (default: benchmark_results/mtp_depth_hysteresis/<timestamp>-<git>).
  --cases SET        smoke, short, long, code, all, or comma list (default: smoke).
  --variants SET     acceptance, grid, all, or comma list (default: acceptance).
  --deterministic    Pass --deterministic to llaminar2 benchmark. Default is production mode.
  --no-deterministic Do not pass --deterministic.
  --dry-run          Print commands without executing.
  -h, --help         Show this help.

Case sets:
  smoke: qbf_short,default
  short: qbf_short,tech_short
  long:  qbf_long,code_long,default
  code:  code_long,python_dataclass,python_cli,cpp_controller,cpp_tests
  all:   qbf_short,tech_short,qbf_long,code_long,python_dataclass,python_cli,cpp_controller,cpp_tests,default

Variants:
  acceptance: baseline,fixed_d1,fixed_d3,dynamic
  grid:       fixed_d1,fixed_d3,dynamic_p70,dynamic_p80,dynamic_p90,dynamic_p95
  all:        baseline,fixed_d1,fixed_d2,fixed_d3,dynamic,dynamic_generated_off,dynamic_p70,dynamic_p80,dynamic_p90,dynamic_p95,dynamic_p90_d70

Environment:
  LLAMINAR_MTP_HYSTERESIS_MODEL
  LLAMINAR_MTP_HYSTERESIS_DEVICE
  LLAMINAR_MTP_HYSTERESIS_CASES
  LLAMINAR_MTP_HYSTERESIS_VARIANTS
  LLAMINAR_MTP_HYSTERESIS_RESULTS_DIR
  LLAMINAR_MTP_HYSTERESIS_DETERMINISTIC
  LLAMINAR_LL2_BIN
  LLAMINAR_LOG_LEVEL

Do not use --no-mpi-bootstrap for these benchmark sweeps.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

model_path="${LLAMINAR_MTP_HYSTERESIS_MODEL:-}"
binary_path="${LLAMINAR_LL2_BIN:-${repo_root}/build_v2_release/llaminar2}"
device_spec="${LLAMINAR_MTP_HYSTERESIS_DEVICE:-rocm:0}"
case_selection="${LLAMINAR_MTP_HYSTERESIS_CASES:-smoke}"
variant_selection="${LLAMINAR_MTP_HYSTERESIS_VARIANTS:-acceptance}"
output_dir="${LLAMINAR_MTP_HYSTERESIS_RESULTS_DIR:-}"
deterministic="${LLAMINAR_MTP_HYSTERESIS_DETERMINISTIC:-0}"
dry_run=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --model)
      model_path="${2:-}"
      shift 2
      ;;
    --binary)
      binary_path="${2:-}"
      shift 2
      ;;
    --device)
      device_spec="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --cases)
      case_selection="${2:-}"
      shift 2
      ;;
    --variants)
      variant_selection="${2:-}"
      shift 2
      ;;
    --deterministic)
      deterministic=1
      shift
      ;;
    --no-deterministic)
      deterministic=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    -*)
      extra_args+=("$1")
      shift
      ;;
    *)
      if [[ -z "${model_path}" ]]; then
        model_path="$1"
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${model_path}" ]]; then
  echo "error: provide --model PATH or set LLAMINAR_MTP_HYSTERESIS_MODEL" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${model_path}" ]]; then
  echo "error: model path does not exist: ${model_path}" >&2
  exit 2
fi

if [[ ! -x "${binary_path}" ]]; then
  echo "error: llaminar2 binary is not executable: ${binary_path}" >&2
  echo "hint: cmake --build build_v2_release --parallel --target llaminar2" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to summarize benchmark JSON" >&2
  exit 2
fi

for extra_arg in "${extra_args[@]}"; do
  if [[ "${extra_arg}" == "--no-mpi-bootstrap" ]]; then
    echo "error: --no-mpi-bootstrap is for profiling/debugging, not benchmark sweeps" >&2
    exit 2
  fi
done

expand_selection() {
  local selection="$1"
  local kind="$2"

  if [[ "${kind}" == "cases" ]]; then
    case "${selection}" in
      smoke) echo "qbf_short default" ;;
      short) echo "qbf_short tech_short" ;;
      long) echo "qbf_long code_long default" ;;
      code) echo "code_long python_dataclass python_cli cpp_controller cpp_tests" ;;
      all) echo "qbf_short tech_short qbf_long code_long python_dataclass python_cli cpp_controller cpp_tests default" ;;
      *) echo "${selection}" | tr ',' ' ' ;;
    esac
  else
    case "${selection}" in
      acceptance) echo "baseline fixed_d1 fixed_d3 dynamic" ;;
      grid) echo "fixed_d1 fixed_d3 dynamic_p70 dynamic_p80 dynamic_p90 dynamic_p95" ;;
      all) echo "baseline fixed_d1 fixed_d2 fixed_d3 dynamic dynamic_generated_off dynamic_p70 dynamic_p80 dynamic_p90 dynamic_p95 dynamic_p90_d70" ;;
      *) echo "${selection}" | tr ',' ' ' ;;
    esac
  fi
}

case_args=()
describe_case() {
  local case_name="$1"
  case_args=()
  case "${case_name}" in
    qbf_short)
      case_args=(-p "The quick brown fox" -c 64 -n 48)
      ;;
    tech_short)
      case_args=(-p "Explain in two sentences why adaptive speculative decoding can speed up greedy inference." -c 128 -n 64)
      ;;
    qbf_long)
      case_args=(-p "The quick brown fox jumps over the lazy dog. The quick brown fox keeps jumping while the lazy dog watches from the warm porch. The quick brown fox then writes a tidy benchmark note about every jump and every pause." -c 256 -n 128)
      ;;
    code_long)
      case_args=(-p "Write a compact C++ function that updates a rolling acceptance-rate window, applies hysteresis before changing speculative draft depth, and explains each branch with a short comment." -c 256 -n 128)
      ;;
    python_dataclass)
      case_args=(-p "Write a Python module with dataclasses for a speculative decoding depth controller. Include configuration defaults, a rolling window class, update methods, serialization to JSON, and a small doctest-style example." -c 384 -n 192)
      ;;
    python_cli)
      case_args=(-p "Write a Python command-line tool using argparse that reads benchmark JSON files, groups rows by prompt and MTP depth policy, computes acceptance-rate summaries, prints a table, and writes a CSV report. Include helper functions and basic error handling." -c 384 -n 192)
      ;;
    cpp_controller)
      case_args=(-p "Write C++17 code for a SpeculativeDepthController class with a config struct, observation struct, stats struct, hysteresis thresholds, cooldown handling, and a recordStep method. Include header-style declarations and source-style method definitions." -c 384 -n 192)
      ;;
    cpp_tests)
      case_args=(-p "Write GoogleTest unit tests for a C++ speculative depth controller. Cover fixed mode, early demotion, promotion hysteresis, cooldown, observe mode, and budget-limited observations. Use clear fixture helpers and repetitive ASSERT/EXPECT checks." -c 384 -n 192)
      ;;
    default)
      case_args=()
      ;;
    *)
      echo "error: unknown case '${case_name}'" >&2
      exit 2
      ;;
  esac
}

variant_args=()
describe_variant() {
  local variant="$1"
  variant_args=()
  case "${variant}" in
    baseline)
      ;;
    fixed_d1)
      variant_args=(--mtp --mtp-draft-tokens 1 --mtp-depth-policy fixed)
      ;;
    fixed_d2)
      variant_args=(--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed)
      ;;
    fixed_d3)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed)
      ;;
    dynamic)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic)
      ;;
    dynamic_generated_off)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-generated-policy false)
      ;;
    dynamic_promote1)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-windows 1)
      ;;
    dynamic_promote3)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-windows 3)
      ;;
    dynamic_ms4)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-min-samples 4)
      ;;
    dynamic_ms4_promote1)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-min-samples 4 --mtp-depth-promote-windows 1)
      ;;
    dynamic_ms4_d60)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-min-samples 4 --mtp-depth-demote-acceptance 0.60)
      ;;
    dynamic_ms4_d55)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-min-samples 4 --mtp-depth-demote-acceptance 0.55)
      ;;
    dynamic_p70)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-full-accept 0.70)
      ;;
    dynamic_p80)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-full-accept 0.80)
      ;;
    dynamic_p90)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-full-accept 0.90)
      ;;
    dynamic_p95)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-full-accept 0.95)
      ;;
    dynamic_p90_d70)
      variant_args=(--mtp --mtp-draft-tokens 3 --mtp-depth-policy dynamic --mtp-depth-promote-full-accept 0.90 --mtp-depth-demote-acceptance 0.70)
      ;;
    *)
      echo "error: unknown variant '${variant}'" >&2
      exit 2
      ;;
  esac
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
git_hash="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/benchmark_results/mtp_depth_hysteresis/${timestamp}-${git_hash}"
fi
mkdir -p "${output_dir}"

summary_path="${output_dir}/summary.tsv"
commands_path="${output_dir}/commands.txt"
metadata_path="${output_dir}/metadata.txt"

{
  echo "repo_root=${repo_root}"
  echo "git_hash=${git_hash}"
  echo "timestamp_utc=${timestamp}"
  echo "binary=${binary_path}"
  echo "model=${model_path}"
  echo "device=${device_spec}"
  echo "cases=${case_selection}"
  echo "variants=${variant_selection}"
  echo "deterministic=${deterministic}"
  echo "extra_args=${extra_args[*]:-}"
  echo
  git -C "${repo_root}" status --short || true
} > "${metadata_path}"

model_name="$(basename "${model_path}")"

printf 'device\tmodel\tmode\tcase\tvariant\tsuccess\tdecode_tps\toverall_tps\tprefill_tokens\tdecode_tokens\tpolicy\tgenerated_policy\tdraft\tdepth\tmin_depth\tmax_depth\tupdates\tpromotions\tdemotions\twindows\taccepted\trejected\trollbacks\tacceptance_pct\tverifier_runs\tverifier_tokens\tjson\n' > "${summary_path}"
: > "${commands_path}"

cases=($(expand_selection "${case_selection}" cases))
variants=($(expand_selection "${variant_selection}" variants))

append_summary() {
  local case_name="$1"
  local variant="$2"
  local json_path="$3"
  jq -r \
    --arg device "${device_spec}" \
    --arg model "${model_name}" \
    --arg case "${case_name}" \
    --arg variant "${variant}" \
    --arg json "${json_path}" '[
    $device,
    $model,
    (if (.config.mtp_verify_mode // "greedy") == "speculative-sampling" then "stochastic" else (.config.mtp_verify_mode // "greedy") end),
    $case,
    $variant,
    (.success // false),
    (.throughput_tokens_per_sec.decode // 0),
    (.throughput_tokens_per_sec.overall // 0),
    (.tokens.prefill // 0),
    (.tokens.decode // 0),
    (.config.mtp_depth_policy // "none"),
    (.config.mtp_depth_generated_policy // false),
    (.config.mtp_draft_tokens // 0),
    (.mtp.current_depth // 0),
    (.mtp.min_depth // 0),
    (.mtp.max_depth // 0),
    (.mtp.depth_policy_updates // 0),
    (.mtp.depth_policy_promotions // 0),
    (.mtp.depth_policy_demotions // 0),
    (.mtp.depth_policy_windows // 0),
    (.mtp.accepted_tokens // 0),
    (.mtp.rejected_tokens // 0),
    (.mtp.rollbacks // 0),
    (((.mtp.acceptance_rate // 0) * 100)),
    (.mtp.verifier_runs // 0),
    (.mtp.verifier_token_count // 0),
    $json
  ] | @tsv' "${json_path}" >> "${summary_path}"
}

log_level="${LLAMINAR_LOG_LEVEL:-ERROR}"

for case_name in "${cases[@]}"; do
  describe_case "${case_name}"
  for variant in "${variants[@]}"; do
    describe_variant "${variant}"
    stem="${case_name}-${variant}"
    json_path="${output_dir}/${stem}.json"
    log_path="${output_dir}/${stem}.log"
    cmd=(
      "${binary_path}" benchmark
      -m "${model_path}"
      -d "${device_spec}"
      --benchmark-json-output "${json_path}"
    )
    if [[ "${deterministic}" == "1" || "${deterministic}" == "true" || "${deterministic}" == "TRUE" ]]; then
      cmd+=(--deterministic)
    fi
    cmd+=("${case_args[@]}")
    cmd+=("${variant_args[@]}")
    cmd+=("${extra_args[@]}")

    {
      printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
      printf '\n'
    } >> "${commands_path}"

    echo "== ${stem} =="
    if [[ "${dry_run}" == "1" ]]; then
      printf 'dry-run: '
      printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
      printf '\n'
      continue
    fi

    if ! LLAMINAR_LOG_LEVEL="${log_level}" "${cmd[@]}" > "${log_path}" 2>&1; then
      tail -n 80 "${log_path}" >&2 || true
      echo "error: benchmark failed for ${stem}; log: ${log_path}" >&2
      exit 1
    fi
    append_summary "${case_name}" "${variant}" "${json_path}"
    tail -n 8 "${log_path}" || true
  done
done

echo
echo "summary: ${summary_path}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${summary_path}"
else
  cat "${summary_path}"
fi
