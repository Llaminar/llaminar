#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mtp_iteration_benchmark_matrix.sh [options] [-- extra llaminar2 benchmark args...]

Runs the standard MTP iteration benchmark matrix and writes one benchmark JSON/log
per lane plus summary.tsv.

Default matrix:
  topologies: single
  devices:    cuda:0,rocm:0,cpu:0
  models:     dense,moe
  modes:      greedy,stochastic
  variants:   baseline,fixed_d1,fixed_d2,fixed_d3,dynamic

The dynamic variant starts at depth 1, keeps depth 1 as the normal adaptive
floor, and may probe/promote back up to the configured max depth 3.

Options:
  --binary PATH          llaminar2 binary (default: build_v2_release/llaminar2)
  --dense-model PATH     Dense Qwen3.6 GGUF
  --moe-model PATH       MoE Qwen3.6 GGUF
  --topologies LIST      Comma list: single,localtp_rocm2,localtp_cuda2,
                         localpp_rocm2,nodelocaltp_cpu2,
                         expert_overlay_rocm2_hot,
                         expert_overlay_rocm2_cpu2
  --devices LIST         Comma list, e.g. cuda:0,rocm:0,cpu:0
                         Used only by the single topology.
  --models LIST          Comma list: dense,moe
  --modes LIST           Comma list: greedy,stochastic
  --variants LIST        Comma list: baseline,fixed_d1,fixed_d2,fixed_d3,dynamic
  --allow-partial-variants
                         Permit diagnostic variant subsets. Without this,
                         dynamic requires baseline plus fixed d1/d2/d3.
  --seed N               Seed for stochastic rows (default: 123)
  --decode-tokens N      Override benchmark decode tokens via --n-predict N
  --mtp-request-batch N  Pass --mtp-max-request-batch N on MTP variants
                         (default: 1; values >1 are Phase 8 diagnostics until
                         runner-batched speculative transactions are executable)
  --output-dir DIR       Output directory
  --perfstats            Capture LLAMINAR_PERF_STATS_JSON for MTP variants
  --gpu-stage-timing     Include graph-captured GPU stage timings in perfstats
  --dry-run              Print commands only
  -h, --help             Show this help

Environment aliases:
  LLAMINAR_LL2_BIN
  LLAMINAR_MTP_MATRIX_DENSE_MODEL
  LLAMINAR_MTP_MATRIX_MOE_MODEL
  LLAMINAR_MTP_MATRIX_TOPOLOGIES
  LLAMINAR_MTP_MATRIX_DEVICES
  LLAMINAR_MTP_MATRIX_MODELS
  LLAMINAR_MTP_MATRIX_MODES
  LLAMINAR_MTP_MATRIX_VARIANTS
  LLAMINAR_MTP_MATRIX_SEED
  LLAMINAR_MTP_MATRIX_DECODE_TOKENS
  LLAMINAR_MTP_MATRIX_REQUEST_BATCH
  LLAMINAR_MTP_MATRIX_RESULTS_DIR
  LLAMINAR_MTP_MATRIX_ALLOW_PARTIAL_VARIANTS
  LLAMINAR_MTP_MATRIX_GPU_STAGE_TIMING

Do not use --no-mpi-bootstrap for this benchmark matrix.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

binary_path="${LLAMINAR_LL2_BIN:-${repo_root}/build_v2_release/llaminar2}"
dense_model="${LLAMINAR_MTP_MATRIX_DENSE_MODEL:-/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf}"
moe_model="${LLAMINAR_MTP_MATRIX_MOE_MODEL:-/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
topologies="${LLAMINAR_MTP_MATRIX_TOPOLOGIES:-single}"
devices="${LLAMINAR_MTP_MATRIX_DEVICES:-cuda:0,rocm:0,cpu:0}"
models="${LLAMINAR_MTP_MATRIX_MODELS:-dense,moe}"
modes="${LLAMINAR_MTP_MATRIX_MODES:-greedy,stochastic}"
variants="${LLAMINAR_MTP_MATRIX_VARIANTS:-baseline,fixed_d1,fixed_d2,fixed_d3,dynamic}"
seed="${LLAMINAR_MTP_MATRIX_SEED:-123}"
decode_tokens="${LLAMINAR_MTP_MATRIX_DECODE_TOKENS:-}"
mtp_request_batch="${LLAMINAR_MTP_MATRIX_REQUEST_BATCH:-1}"
output_dir="${LLAMINAR_MTP_MATRIX_RESULTS_DIR:-}"
allow_partial_variants="${LLAMINAR_MTP_MATRIX_ALLOW_PARTIAL_VARIANTS:-0}"
gpu_stage_timing="${LLAMINAR_MTP_MATRIX_GPU_STAGE_TIMING:-0}"
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
    --topologies)
      topologies="${2:-}"
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
    --mtp-request-batch)
      mtp_request_batch="${2:-}"
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
    --gpu-stage-timing)
      gpu_stage_timing=1
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

if [[ ! "${mtp_request_batch}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --mtp-request-batch must be a positive integer, got: ${mtp_request_batch}" >&2
  exit 2
fi

if [[ "${gpu_stage_timing}" != "0" && "${gpu_stage_timing}" != "1" ]]; then
  echo "error: LLAMINAR_MTP_MATRIX_GPU_STAGE_TIMING must be 0 or 1, got: ${gpu_stage_timing}" >&2
  exit 2
fi

if [[ "${gpu_stage_timing}" == "1" && "${perfstats}" != "1" ]]; then
  echo "error: --gpu-stage-timing requires --perfstats so timing records have an output file" >&2
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

topology_model_supported() {
  local topology="$1"
  local model="$2"
  case "${topology}" in
    single)
      return 0
      ;;
    localtp_rocm2|localtp_cuda2|localpp_rocm2|nodelocaltp_cpu2)
      [[ "${model}" == "dense" ]]
      return
      ;;
    expert_overlay_rocm2_hot|expert_overlay_rocm2_cpu2)
      [[ "${model}" == "moe" ]]
      return
      ;;
    *)
      echo "error: unknown topology '${topology}'" >&2
      exit 2
      ;;
  esac
}

topology_lane_devices() {
  local topology="$1"
  case "${topology}" in
    single)
      split_csv "${devices}"
      ;;
    localtp_rocm2|localtp_cuda2|localpp_rocm2|nodelocaltp_cpu2|expert_overlay_rocm2_hot|expert_overlay_rocm2_cpu2)
      echo "${topology}"
      ;;
    *)
      echo "error: unknown topology '${topology}'" >&2
      exit 2
      ;;
  esac
}

topology_args=()
topology_device_label=""
describe_topology() {
  local topology="$1"
  local lane_device="$2"
  topology_args=()
  topology_device_label="${lane_device}"

  case "${topology}" in
    single)
      topology_args=(-d "${lane_device}")
      ;;
    localtp_rocm2)
      topology_device_label="rocm:0+rocm:1"
      topology_args=(
        --tp-devices "rocm:0,rocm:1"
        --tensor-parallelism-degree 2
        --tp-scope local
        --backend rccl
      )
      ;;
    localtp_cuda2)
      topology_device_label="cuda:0+cuda:1"
      topology_args=(
        --tp-devices "cuda:0,cuda:1"
        --tensor-parallelism-degree 2
        --tp-scope local
        --backend nccl
      )
      ;;
    localpp_rocm2)
      topology_device_label="rocm:0|rocm:1"
      topology_args=(
        --pipeline-parallelism-degree 2
        --pp-split manual
        --define-domain "stage0=rocm:0;scope=local;owner=0"
        --define-domain "stage1=rocm:1;scope=local;owner=0"
        --pp-stage "0=stage0:0-31"
        --pp-stage "1=stage1:32-63"
      )
      ;;
    nodelocaltp_cpu2)
      topology_device_label="cpu:0+cpu:1"
      topology_args=(
        --mpi-procs 2
        --device-map "0=cpu:0,1=cpu:1"
        --tensor-parallelism-degree 2
        --tp-scope node_local
        --backend mpi
      )
      ;;
    expert_overlay_rocm2_hot)
      topology_device_label="rocm:0+rocm:1"
      topology_args=(
        --moe-expert-overlay tiered
        --moe-expert-overlay-continuation qwen36_moe_rocm_hot
        --moe-expert-overlay-base-domain qwen36_moe_rocm_hot
        --moe-expert-overlay-shared-domain qwen36_moe_rocm_hot
        --moe-expert-overlay-residency static-by-id
        --moe-expert-overlay-domain "qwen36_moe_rocm_hot=rocm:0,rocm:1;scope=local;backend=rccl;compute=replicated_experts"
        --moe-expert-overlay-tier "hot@qwen36_moe_rocm_hot;priority=0;max-experts-per-layer=256;memory-mb=8192"
      )
      ;;
    expert_overlay_rocm2_cpu2)
      topology_device_label="rocm:0+rocm:1+cpu:0+cpu:1"
      topology_args=(
        --moe-expert-overlay tiered
        --moe-expert-overlay-continuation qwen36_moe_rocm_hot
        --moe-expert-overlay-base-domain qwen36_moe_rocm_hot
        --moe-expert-overlay-shared-domain qwen36_moe_rocm_hot
        --moe-expert-overlay-residency static-by-id
        --moe-expert-overlay-domain "qwen36_moe_rocm_hot=rocm:0,rocm:1;scope=local;backend=rccl;compute=replicated_experts"
        --moe-expert-overlay-domain "qwen36_moe_cpu_cold=cpu:0,cpu:1;scope=local;backend=upi;compute=replicated_experts"
        --moe-expert-overlay-tier "hot@qwen36_moe_rocm_hot;priority=0;max-experts-per-layer=240;memory-mb=4096"
        --moe-expert-overlay-tier "cold@qwen36_moe_cpu_cold;priority=1;max-experts-per-layer=0;memory-mb=0;fallback=true"
      )
      ;;
    *)
      echo "error: unknown topology '${topology}'" >&2
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
        --mtp-min-draft-tokens 1
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
stage_summary_path="${output_dir}/stage_summary.tsv"
commands_path="${output_dir}/commands.txt"
metadata_path="${output_dir}/metadata.txt"

{
  echo "repo_root=${repo_root}"
  echo "git_hash=${git_hash}"
  echo "timestamp_utc=${timestamp}"
  echo "binary=${binary_path}"
  echo "dense_model=${dense_model}"
  echo "moe_model=${moe_model}"
  echo "topologies=${topologies}"
  echo "devices=${devices}"
  echo "models=${models}"
  echo "modes=${modes}"
  echo "variants=${variants}"
  echo "seed=${seed}"
  echo "decode_tokens=${decode_tokens:-default}"
  echo "mtp_request_batch=${mtp_request_batch}"
  echo "perfstats=${perfstats}"
  echo "gpu_stage_timing=${gpu_stage_timing}"
  echo "allow_partial_variants=${allow_partial_variants}"
  echo "extra_args=${extra_args[*]:-}"
  echo
  git -C "${repo_root}" status --short || true
} > "${metadata_path}"

perf_summary_script="${repo_root}/scripts/summarize_mtp_perfstats.py"
if [[ ! -x "${perf_summary_script}" ]]; then
  chmod +x "${perf_summary_script}" 2>/dev/null || true
fi

printf 'topology\tdevice\tmodel\tmode\tvariant\tsuccess\tdecode_tps\tspeedup_vs_baseline\toverall_tps\tprefill_tokens\tdecode_tokens\tpolicy\tgenerated_policy\tdraft\tdepth\tmin_depth\tmax_depth\trequest_batch\tdepth_updates\tdepth_promotions\tdepth_demotions\tdepth_windows\tlast_depth_reason\taccepted\trejected\trollbacks\tacceptance_pct\tverifier_runs\tverifier_tokens\tdecode_step_ms\tverifier_ms\tcondition_ms\tcondition_count\tcondition_skipped_ready\tcorrection_ms\tcorrection_count\tdeferred_corrections\trejection_no_ready\tpublish_ms\tpublish_count\tpublish_avg_ms\tsidecar_ms\tsidecar_depth0_decode_ms\tshifted_initial_ms\tshifted_initial_commits\tshifted_initial_reused\tshifted_prefix_ms\tshifted_deferred_ms\tshifted_row_ms\tshifted_kv_ready_events\tshifted_kv_ready_waits\tshifted_kv_syncs_deferred\tsampling_ms\tsampling_enqueue_ms\tstochastic_batch_outcome_ms\tstochastic_batch_d2h_sync_ms\tgreedy_summary_ms\tcheckpoint_ms\tsidecar_graph_hits\tsidecar_graph_misses\tmain_decode_warmup\tmain_decode_capture\tmain_decode_replay\tmain_verifier_warmup\tmain_verifier_capture\tmain_verifier_replay\treplay_resets\treplay_preserves\treplay_reset_caches\treplay_rebind_caches\treplay_ordinary_decode_resets\treplay_verifier_rebinds\treplay_other_rebinds\tjson\tperfstats\n' > "${summary_path}"
printf 'topology\tdevice\tmodel\tmode\tvariant\tdomain\tphase\tcontext\tname\ttotal_ms\tcount\tavg_us\tstage_count\tsource\tperfstats\n' > "${stage_summary_path}"
: > "${commands_path}"

zero_perf_summary() {
  local fields=44
  local values=()
  for ((i = 0; i < fields; ++i)); do
    values+=("0")
  done
  local IFS=$'\t'
  printf '%s' "${values[*]}"
}

append_summary() {
  local topology="$1"
  local device="$2"
  local model="$3"
  local mode="$4"
  local variant="$5"
  local json_path="$6"
  local perf_path="$7"
  local baseline_decode_tps="${8:-0}"
  local perf_summary="${9:-$(zero_perf_summary)}"
  local base_summary
  base_summary="$(jq -r \
    --arg topology "${topology}" \
    --arg device "${device}" \
    --arg model "${model}" \
    --arg mode "${mode}" \
    --arg variant "${variant}" \
    --argjson baseline_decode_tps "${baseline_decode_tps}" \
    '[
      $topology,
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
      (.config.mtp_depth_generated_policy // false),
      (.config.mtp_draft_tokens // 0),
      (.mtp.current_depth // 0),
      (.mtp.min_depth // 0),
      (.mtp.max_depth // 0),
      (.config.mtp_max_request_batch // 1),
      (.mtp.depth_policy_updates // 0),
      (.mtp.depth_policy_promotions // 0),
      (.mtp.depth_policy_demotions // 0),
      (.mtp.depth_policy_windows // 0),
      (.mtp.request.last_depth_policy_reason // ""),
      (.mtp.accepted_tokens // 0),
      (.mtp.rejected_tokens // 0),
      (.mtp.rollbacks // 0),
      (((.mtp.acceptance_rate // 0) * 100)),
      (.mtp.verifier_runs // 0),
      (.mtp.verifier_token_count // 0)
    ] | @tsv' "${json_path}")"
  printf '%s\t%s\t%s\t%s\n' "${base_summary}" "${perf_summary}" "${json_path}" "${perf_path}" >> "${summary_path}"
}

append_stage_summary() {
  local topology="$1"
  local device="$2"
  local model="$3"
  local mode="$4"
  local variant="$5"
  local perf_path="$6"

  if [[ -z "${perf_path}" || ! -f "${perf_path}" ]]; then
    return
  fi

  jq -r \
    --arg topology "${topology}" \
    --arg device "${device}" \
    --arg model "${model}" \
    --arg mode "${mode}" \
    --arg variant "${variant}" \
    --arg perfstats "${perf_path}" \
    '
      (.records // [])
      | map(select(
          (.kind // "") == "timer"
          and ((.domain // "") == "mtp" or (.domain // "") == "stage_gpu")
          and ((.phase // "") == "decode")
          and ((.total_ms // 0) > 0)
        ))
      | sort_by((.total_ms // 0)) | reverse | .[:40]
      | .[]
      | [
          $topology,
          $device,
          $model,
          $mode,
          $variant,
          (.domain // ""),
          (.phase // ""),
          (.tags.context // ""),
          (.name // ""),
          (.total_ms // 0),
          (.count // 0),
          (.avg_us // 0),
          (.tags.stage_count // ""),
          (.tags.source // ""),
          $perfstats
        ] | @tsv
    ' "${perf_path}" >> "${stage_summary_path}"
}

log_level="${LLAMINAR_LOG_LEVEL:-ERROR}"
declare -A baseline_decode_tps_by_lane=()

for model in $(split_csv "${models}"); do
  model_path="$(model_path_for "${model}")"
  if [[ ! -f "${model_path}" ]]; then
    echo "error: selected ${model} model path does not exist: ${model_path}" >&2
    exit 2
  fi

  for topology in $(split_csv "${topologies}"); do
    if ! topology_model_supported "${topology}" "${model}"; then
      echo "error: topology '${topology}' is not supported for model '${model}' in this matrix script" >&2
      exit 2
    fi
    topology_slug="$(sanitize "${topology}")"
    for lane_device in $(topology_lane_devices "${topology}"); do
      describe_topology "${topology}" "${lane_device}"
      device="${topology_device_label}"
      device_slug="$(sanitize "${device}")"
    for mode in $(split_csv "${modes}"); do
      describe_mode "${mode}"
      for variant in $(split_csv "${variants}"); do
        describe_variant "${mode}" "${variant}"
        stem="${topology_slug}-${device_slug}-${model}-${mode}-${variant}"
        json_path="${output_dir}/${stem}.json"
        log_path="${output_dir}/${stem}.log"
        perf_path=""
        if [[ "${perfstats}" == "1" && "${variant}" != "baseline" ]]; then
          perf_path="${output_dir}/${stem}.perfstats.json"
        fi

        cmd=(
          "${binary_path}" benchmark
          -m "${model_path}"
          --benchmark-json-output "${json_path}"
        )
        cmd+=("${topology_args[@]}")
        if [[ -n "${decode_tokens}" ]]; then
          cmd+=(--n-predict "${decode_tokens}")
        fi
        cmd+=("${mode_args[@]}")
        cmd+=("${variant_args[@]}")
        if [[ "${variant}" != "baseline" && "${mtp_request_batch}" != "1" ]]; then
          cmd+=(--mtp-max-request-batch "${mtp_request_batch}")
        fi
        cmd+=("${extra_args[@]}")

        {
          if [[ -n "${perf_path}" ]]; then
            env_args=(
              "LLAMINAR_LOG_LEVEL=${log_level}"
              "LLAMINAR_PERF_STATS_JSON=${perf_path}"
            )
            if [[ "${gpu_stage_timing}" == "1" ]]; then
              env_args+=("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1")
            fi
            printf '%q ' "${env_args[@]}" "${cmd[@]}"
          else
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
          fi
          printf '\n'
        } >> "${commands_path}"

        echo "== ${stem} =="
        if [[ "${dry_run}" == "1" ]]; then
          printf 'dry-run: '
          if [[ -n "${perf_path}" ]]; then
            env_args=(
              "LLAMINAR_LOG_LEVEL=${log_level}"
              "LLAMINAR_PERF_STATS_JSON=${perf_path}"
            )
            if [[ "${gpu_stage_timing}" == "1" ]]; then
              env_args+=("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1")
            fi
            printf '%q ' "${env_args[@]}" "${cmd[@]}"
          else
            printf '%q ' "LLAMINAR_LOG_LEVEL=${log_level}" "${cmd[@]}"
          fi
          printf '\n'
          continue
        fi

        if [[ -n "${perf_path}" ]]; then
          env_args=(
            "LLAMINAR_LOG_LEVEL=${log_level}"
            "LLAMINAR_PERF_STATS_JSON=${perf_path}"
          )
          if [[ "${gpu_stage_timing}" == "1" ]]; then
            env_args+=("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1")
          fi
          if ! env "${env_args[@]}" "${cmd[@]}" > "${log_path}" 2>&1; then
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

        lane_key="${topology}|${device}|${model}|${mode}"
        decode_tps="$(jq -r '(.throughput_tokens_per_sec.decode // 0)' "${json_path}")"
        baseline_decode_tps="${baseline_decode_tps_by_lane[${lane_key}]:-0}"
        if [[ "${variant}" == "baseline" ]]; then
          baseline_decode_tps="${decode_tps}"
          baseline_decode_tps_by_lane["${lane_key}"]="${decode_tps}"
        fi

        perf_summary="$(zero_perf_summary)"
        if [[ -n "${perf_path}" ]]; then
          perf_summary="$("${perf_summary_script}" "${perf_path}")"
        fi

        append_summary "${topology}" "${device}" "${model}" "${mode}" "${variant}" "${json_path}" "${perf_path}" "${baseline_decode_tps}" "${perf_summary}"
        append_stage_summary "${topology}" "${device}" "${model}" "${mode}" "${variant}" "${perf_path}"
        tail -n 8 "${log_path}" || true
      done
    done
    done
  done
done

echo "summary: ${summary_path}"
echo "stage summary: ${stage_summary_path}"
echo "commands: ${commands_path}"
