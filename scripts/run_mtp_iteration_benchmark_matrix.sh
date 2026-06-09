#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mtp_iteration_benchmark_matrix.sh [options] [-- extra llaminar2 benchmark args...]

Runs the standard MTP iteration benchmark matrix and writes one benchmark JSON/log
per lane plus summary.tsv.

Default matrix:
  devices:  cuda:0,rocm:0,cpu:0
  models:   dense,moe
  modes:    greedy,stochastic
  variants: baseline,fixed_d1,fixed_d2,fixed_d3,dynamic

The dynamic variant starts at depth 1, allows adaptive depth-0 bypass, and may
probe/promote back up to the configured max depth 3.

Options:
  --binary PATH          llaminar2 binary (default: build_v2_release/llaminar2)
  --dense-model PATH     Dense Qwen3.6 GGUF
  --moe-model PATH       MoE Qwen3.6 GGUF
  --devices LIST         Comma list, e.g. cuda:0,rocm:0,cpu:0
  --models LIST          Comma list: dense,moe
  --modes LIST           Comma list: greedy,stochastic
  --variants LIST        Comma list: baseline,fixed_d1,fixed_d2,fixed_d3,dynamic
  --allow-partial-variants
                         Permit diagnostic variant subsets. Without this,
                         dynamic requires baseline plus fixed d1/d2/d3.
  --seed N               Seed for stochastic rows (default: 123)
  --decode-tokens N      Override benchmark decode tokens via --n-predict N
  --output-dir DIR       Output directory
  --perfstats            Capture LLAMINAR_PERF_STATS_JSON for MTP variants
  --dry-run              Print commands only
  -h, --help             Show this help

Environment aliases:
  LLAMINAR_LL2_BIN
  LLAMINAR_MTP_MATRIX_DENSE_MODEL
  LLAMINAR_MTP_MATRIX_MOE_MODEL
  LLAMINAR_MTP_MATRIX_DEVICES
  LLAMINAR_MTP_MATRIX_MODELS
  LLAMINAR_MTP_MATRIX_MODES
  LLAMINAR_MTP_MATRIX_VARIANTS
  LLAMINAR_MTP_MATRIX_SEED
  LLAMINAR_MTP_MATRIX_DECODE_TOKENS
  LLAMINAR_MTP_MATRIX_RESULTS_DIR
  LLAMINAR_MTP_MATRIX_ALLOW_PARTIAL_VARIANTS

Do not use --no-mpi-bootstrap for this benchmark matrix.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

binary_path="${LLAMINAR_LL2_BIN:-${repo_root}/build_v2_release/llaminar2}"
dense_model="${LLAMINAR_MTP_MATRIX_DENSE_MODEL:-/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf}"
moe_model="${LLAMINAR_MTP_MATRIX_MOE_MODEL:-/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
devices="${LLAMINAR_MTP_MATRIX_DEVICES:-cuda:0,rocm:0,cpu:0}"
models="${LLAMINAR_MTP_MATRIX_MODELS:-dense,moe}"
modes="${LLAMINAR_MTP_MATRIX_MODES:-greedy,stochastic}"
variants="${LLAMINAR_MTP_MATRIX_VARIANTS:-baseline,fixed_d1,fixed_d2,fixed_d3,dynamic}"
seed="${LLAMINAR_MTP_MATRIX_SEED:-123}"
decode_tokens="${LLAMINAR_MTP_MATRIX_DECODE_TOKENS:-}"
output_dir="${LLAMINAR_MTP_MATRIX_RESULTS_DIR:-}"
allow_partial_variants="${LLAMINAR_MTP_MATRIX_ALLOW_PARTIAL_VARIANTS:-0}"
perfstats=0
dry_run=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --binary)
      binary_path="${2:-}"
      shift 2
      ;;
    --dense-model)
      dense_model="${2:-}"
      shift 2
      ;;
    --moe-model)
      moe_model="${2:-}"
      shift 2
      ;;
    --devices)
      devices="${2:-}"
      shift 2
      ;;
    --models)
      models="${2:-}"
      shift 2
      ;;
    --modes)
      modes="${2:-}"
      shift 2
      ;;
    --variants)
      variants="${2:-}"
      shift 2
      ;;
    --seed)
      seed="${2:-}"
      shift 2
      ;;
    --decode-tokens)
      decode_tokens="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --perfstats)
      perfstats=1
      shift
      ;;
    --allow-partial-variants)
      allow_partial_variants=1
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
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

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
    echo "error: --no-mpi-bootstrap is for profiling/debugging, not benchmark runs" >&2
    exit 2
  fi
done

if [[ -n "${decode_tokens}" && ! "${decode_tokens}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --decode-tokens must be a positive integer, got: ${decode_tokens}" >&2
  exit 2
fi

variant_list=" $(echo "${variants}" | tr ',' ' ') "
has_variant() {
  [[ "${variant_list}" == *" $1 "* ]]
}

if [[ "${allow_partial_variants}" != "1" ]]; then
  if has_variant dynamic; then
    missing=()
    for required in baseline fixed_d1 fixed_d2 fixed_d3; do
      if ! has_variant "${required}"; then
        missing+=("${required}")
      fi
    done
    if (( ${#missing[@]} > 0 )); then
      echo "error: dynamic matrix rows require same-run baseline,fixed_d1,fixed_d2,fixed_d3; missing: ${missing[*]}" >&2
      echo "hint: use --allow-partial-variants only for local diagnostics, not iteration evidence" >&2
      exit 2
    fi
  fi
fi

split_csv() {
  echo "$1" | tr ',' ' '
}

sanitize() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

model_path_for() {
  case "$1" in
    dense) echo "${dense_model}" ;;
    moe) echo "${moe_model}" ;;
    *)
      echo "error: unknown model lane '$1'" >&2
      exit 2
      ;;
  esac
}

mode_args=()
describe_mode() {
  local mode="$1"
  mode_args=()

  case "${mode}" in
    greedy)
      mode_args=(--temperature 0 --seed "${seed}")
      ;;
    stochastic)
      mode_args=(--seed "${seed}")
      ;;
    *)
      echo "error: unknown mode '${mode}'" >&2
      exit 2
      ;;
  esac
}

variant_args=()
describe_variant() {
  local mode="$1"
  local variant="$2"
  variant_args=()

  case "${variant}" in
    baseline)
      return
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
      variant_args=(
        --mtp
        --mtp-draft-tokens 3
        --mtp-depth-policy dynamic
        --mtp-min-draft-tokens 0
        --mtp-initial-draft-tokens 1
      )
      ;;
    *)
      echo "error: unknown variant '${variant}'" >&2
      exit 2
      ;;
  esac

  case "${mode}" in
    greedy)
      variant_args+=(--mtp-verify-mode greedy)
      ;;
    stochastic)
      variant_args+=(--mtp-verify-mode speculative-sampling)
      ;;
    *)
      echo "error: unknown mode '${mode}'" >&2
      exit 2
      ;;
  esac
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
git_hash="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/benchmark_results/mtp_vllm_style/${timestamp}-iteration-matrix-${git_hash}"
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
  echo "dense_model=${dense_model}"
  echo "moe_model=${moe_model}"
  echo "devices=${devices}"
  echo "models=${models}"
  echo "modes=${modes}"
  echo "variants=${variants}"
  echo "seed=${seed}"
  echo "decode_tokens=${decode_tokens:-default}"
  echo "perfstats=${perfstats}"
  echo "allow_partial_variants=${allow_partial_variants}"
  echo "extra_args=${extra_args[*]:-}"
  echo
  git -C "${repo_root}" status --short || true
} > "${metadata_path}"

perf_summary_script="${repo_root}/scripts/summarize_mtp_perfstats.py"
if [[ ! -x "${perf_summary_script}" ]]; then
  chmod +x "${perf_summary_script}" 2>/dev/null || true
fi

printf 'device\tmodel\tmode\tvariant\tsuccess\tdecode_tps\tspeedup_vs_baseline\toverall_tps\tprefill_tokens\tdecode_tokens\tpolicy\tdraft\tdepth\taccepted\trejected\trollbacks\tacceptance_pct\tverifier_runs\tverifier_tokens\tdecode_step_ms\tverifier_ms\tcorrection_ms\tcorrection_count\tdeferred_corrections\tpublish_ms\tmain_verifier_warmup\tmain_verifier_capture\tmain_verifier_replay\treplay_resets\treplay_preserves\tjson\tperfstats\n' > "${summary_path}"
: > "${commands_path}"

append_summary() {
  local device="$1"
  local model="$2"
  local mode="$3"
  local variant="$4"
  local json_path="$5"
  local perf_path="$6"
  local baseline_decode_tps="${7:-0}"
  local perf_summary="${8:-0	0	0	0	0	0	0	0	0	0	0}"
  local base_summary
  base_summary="$(jq -r \
    --arg device "${device}" \
    --arg model "${model}" \
    --arg mode "${mode}" \
    --arg variant "${variant}" \
    --argjson baseline_decode_tps "${baseline_decode_tps}" \
    '[
      $device,
      $model,
      $mode,
      $variant,
      (.success // false),
      (.throughput_tokens_per_sec.decode // 0),
      (if $baseline_decode_tps > 0 then ((.throughput_tokens_per_sec.decode // 0) / $baseline_decode_tps) else 0 end),
      (.throughput_tokens_per_sec.overall // 0),
      (.tokens.prefill // 0),
      (.tokens.decode // 0),
      (.config.mtp_depth_policy // "none"),
      (.config.mtp_draft_tokens // 0),
      (.mtp.current_depth // 0),
      (.mtp.accepted_tokens // 0),
      (.mtp.rejected_tokens // 0),
      (.mtp.rollbacks // 0),
      (((.mtp.acceptance_rate // 0) * 100)),
      (.mtp.verifier_runs // 0),
      (.mtp.verifier_token_count // 0)
    ] | @tsv' "${json_path}")"
  printf '%s\t%s\t%s\t%s\n' "${base_summary}" "${perf_summary}" "${json_path}" "${perf_path}" >> "${summary_path}"
}

log_level="${LLAMINAR_LOG_LEVEL:-ERROR}"
declare -A baseline_decode_tps_by_lane=()

for model in $(split_csv "${models}"); do
  model_path="$(model_path_for "${model}")"
  if [[ ! -f "${model_path}" ]]; then
    echo "error: selected ${model} model path does not exist: ${model_path}" >&2
    exit 2
  fi

  for device in $(split_csv "${devices}"); do
    device_slug="$(sanitize "${device}")"
    for mode in $(split_csv "${modes}"); do
      describe_mode "${mode}"
      for variant in $(split_csv "${variants}"); do
        describe_variant "${mode}" "${variant}"
        stem="${device_slug}-${model}-${mode}-${variant}"
        json_path="${output_dir}/${stem}.json"
        log_path="${output_dir}/${stem}.log"
        perf_path=""
        if [[ "${perfstats}" == "1" && "${variant}" != "baseline" ]]; then
          perf_path="${output_dir}/${stem}.perfstats.json"
        fi

        cmd=(
          "${binary_path}" benchmark
          -m "${model_path}"
          -d "${device}"
          --benchmark-json-output "${json_path}"
        )
        if [[ -n "${decode_tokens}" ]]; then
          cmd+=(--n-predict "${decode_tokens}")
        fi
        cmd+=("${mode_args[@]}")
        cmd+=("${variant_args[@]}")
        cmd+=("${extra_args[@]}")

        {
          if [[ -n "${perf_path}" ]]; then
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "LLAMINAR_PERF_STATS_JSON=${perf_path}" "${cmd[@]}"
          else
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
          fi
          printf '\n'
        } >> "${commands_path}"

        echo "== ${stem} =="
        if [[ "${dry_run}" == "1" ]]; then
          printf 'dry-run: '
          if [[ -n "${perf_path}" ]]; then
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "LLAMINAR_PERF_STATS_JSON=${perf_path}" "${cmd[@]}"
          else
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
          fi
          printf '\n'
          continue
        fi

        if [[ -n "${perf_path}" ]]; then
          if ! LLAMINAR_LOG_LEVEL="${log_level}" LLAMINAR_PERF_STATS_JSON="${perf_path}" "${cmd[@]}" > "${log_path}" 2>&1; then
            tail -n 80 "${log_path}" >&2 || true
            echo "error: benchmark failed for ${stem}; log: ${log_path}" >&2
            exit 1
          fi
        else
          if ! LLAMINAR_LOG_LEVEL="${log_level}" "${cmd[@]}" > "${log_path}" 2>&1; then
            tail -n 80 "${log_path}" >&2 || true
            echo "error: benchmark failed for ${stem}; log: ${log_path}" >&2
            exit 1
          fi
        fi

        lane_key="${device}|${model}|${mode}"
        decode_tps="$(jq -r '(.throughput_tokens_per_sec.decode // 0)' "${json_path}")"
        baseline_decode_tps="${baseline_decode_tps_by_lane[${lane_key}]:-0}"
        if [[ "${variant}" == "baseline" ]]; then
          baseline_decode_tps="${decode_tps}"
          baseline_decode_tps_by_lane["${lane_key}"]="${decode_tps}"
        fi

        perf_summary="0	0	0	0	0	0	0	0	0	0	0"
        if [[ -n "${perf_path}" ]]; then
          perf_summary="$("${perf_summary_script}" "${perf_path}")"
        fi

        append_summary "${device}" "${model}" "${mode}" "${variant}" "${json_path}" "${perf_path}" "${baseline_decode_tps}" "${perf_summary}"
        tail -n 8 "${log_path}" || true
      done
    done
  done
done

echo "summary: ${summary_path}"
echo "commands: ${commands_path}"
