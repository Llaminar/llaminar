#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bin="${LLAMINAR_BIN:-${repo_root}/build_v2_release/llaminar2}"
model="${LLAMINAR_QWEN36_MOE_MODEL:-/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
out_dir="${LLAMINAR_GPU_MOE_REBALANCE_OUT:-${repo_root}/benchmark_results/qwen36_moe_gpu_rebalance_$(date +%Y%m%d_%H%M%S)}"
backend="both"
placement="twocard"
reps="${LLAMINAR_GPU_MOE_REBALANCE_REPS:-5}"
context_length="${LLAMINAR_GPU_MOE_REBALANCE_CONTEXT:-1024}"
n_predict="${LLAMINAR_GPU_MOE_REBALANCE_N_PREDICT:-128}"
cases_csv="${LLAMINAR_GPU_MOE_REBALANCE_CASES:-static,observe,dynamic,dynamic_hot10}"
rebalance_window="${LLAMINAR_GPU_MOE_REBALANCE_WINDOW:-64}"
dry_run=0
perfstats="${LLAMINAR_GPU_MOE_REBALANCE_PERFSTATS:-0}"
stage_gpu_stats="${LLAMINAR_GPU_MOE_REBALANCE_STAGE_GPU_STATS:-0}"
capture_collectives="${LLAMINAR_GPU_MOE_REBALANCE_CAPTURE_COLLECTIVES:-1}"
defer_captured_collective_sync="${LLAMINAR_GPU_MOE_REBALANCE_DEFER_CAPTURED_COLLECTIVE_SYNC:-0}"
dense_tp="${LLAMINAR_GPU_MOE_REBALANCE_DENSE_TP:-0}"
dense_decode_replicated="${LLAMINAR_GPU_MOE_REBALANCE_DENSE_DECODE_REPLICATED:-0}"
dense_policy="${LLAMINAR_GPU_MOE_REBALANCE_DENSE_POLICY:-}"
allreduce_precision="${LLAMINAR_GPU_MOE_REBALANCE_ALLREDUCE_PRECISION:-}"
allreduce_fp16_min_elements="${LLAMINAR_GPU_MOE_REBALANCE_ALLREDUCE_FP16_MIN_ELEMENTS:-}"
small_gpu_allreduce="${LLAMINAR_GPU_MOE_REBALANCE_SMALL_GPU_ALLREDUCE:-0}"
small_gpu_allreduce_max_elements="${LLAMINAR_GPU_MOE_REBALANCE_SMALL_GPU_ALLREDUCE_MAX_ELEMENTS:-}"

usage() {
  cat <<USAGE
Usage: $0 [--backend cuda|rocm|both] [--placement single|twocard|both] [--cases LIST] [--bin PATH] [--model PATH] [--out DIR] [--reps N] [--context-length N] [--n-predict N] [--rebalance-window N] [--perfstats] [--stage-gpu-stats] [--capture-collectives] [--defer-captured-collective-sync] [--dense-tp] [--dense-decode-replicated] [--dense-policy POLICY] [--allreduce-precision fp16|fp32|bf16] [--allreduce-fp16-min-elements N] [--small-gpu-allreduce] [--small-gpu-allreduce-max-elements N] [--dry-run]

Runs the Qwen3.6 35B MoE GPU expert-rebalance proof matrix:
  static placement
  observe overhead control
  dynamic ownership
  dynamic ownership + 10% hot expert replicas

Each run uses one homogeneous 2-card LocalTP routed expert domain owned by MPI rank 0:
  CUDA: cuda:0,cuda:1; backend=nccl; compute=apportioned_experts
  ROCm: rocm:0,rocm:1; backend=rccl; compute=apportioned_experts

Single-card placement is included for 1x CUDA/ROCm baselines:
  CUDA: -d cuda:0
  ROCm: -d rocm:0

LIST is comma-separated, e.g. --cases static,dynamic_hot10.

By default rows are clean throughput measurements. Use --perfstats for
diagnostic JSON/CSV counters including tp_allreduce_bom; use --stage-gpu-stats
only when you need per-stage GPU event timing, since it perturbs throughput.

Dynamic/observe cases default to --moe-rebalance-window 64 so the standard
128-token sprint proof crosses at least one rebalance decision window. Override
with --rebalance-window or LLAMINAR_GPU_MOE_REBALANCE_WINDOW.

Homogeneous two-card LocalTP runs default to capturing NCCL/RCCL collectives
inside decode GPU graphs. Use --no-capture-collectives to force per-layer manual
collective segments for overhead diagnostics.

Use --defer-captured-collective-sync with --capture-collectives to skip the
final host stream wait after replaying a fully captured collective decode graph.
This is a diagnostic decode-overhead knob; correctness depends on a later GPU
consumer or benchmark sampling step providing the synchronization point.

Use --dense-tp to tensor-parallelize dense/non-expert projections across the
same two-card continuation domain. The default two-card proof remains
expert-overlay-only for continuity with the original sprint.

Use --dense-decode-replicated with --dense-tp to mirror full dense/non-expert
weights for decode, while keeping dense tensor parallelism and allreduce in
prefill. Routed expert reductions remain controlled by the MoE overlay.

Use --dense-policy to pass an explicit --moe-expert-overlay-dense-policy value,
for example tensor-parallel-decode-mirrored-embedding. This overrides
--dense-tp/--dense-decode-replicated for two-card runs.

Use --allreduce-precision fp16|fp32|bf16 to force the collective transport
precision for diagnostic/performance A/B runs. Omit it to use the model schema's
hybrid precision policy.

Use --allreduce-fp16-min-elements N with fp16 transport to keep tiny decode
reductions on fp32 while preserving fp16 for larger prefill reductions.

Use --small-gpu-allreduce to try the experimental two-card LocalTP peer-add
fast path for tiny fp32 decode reductions. Use
--small-gpu-allreduce-max-elements N to adjust the cutoff.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="${2:?missing --backend value}"
      shift 2
      ;;
    --placement)
      placement="${2:?missing --placement value}"
      shift 2
      ;;
    --cases)
      cases_csv="${2:?missing --cases value}"
      shift 2
      ;;
    --bin)
      bin="${2:?missing --bin value}"
      shift 2
      ;;
    --model)
      model="${2:?missing --model value}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing --out value}"
      shift 2
      ;;
    --reps)
      reps="${2:?missing --reps value}"
      shift 2
      ;;
    --context-length)
      context_length="${2:?missing --context-length value}"
      shift 2
      ;;
    --n-predict)
      n_predict="${2:?missing --n-predict value}"
      shift 2
      ;;
    --rebalance-window)
      rebalance_window="${2:?missing --rebalance-window value}"
      shift 2
      ;;
    --perfstats)
      perfstats=1
      shift
      ;;
    --no-perfstats)
      perfstats=0
      stage_gpu_stats=0
      shift
      ;;
    --stage-gpu-stats)
      perfstats=1
      stage_gpu_stats=1
      shift
      ;;
    --capture-collectives)
      capture_collectives=1
      shift
      ;;
    --no-capture-collectives)
      capture_collectives=0
      shift
      ;;
    --defer-captured-collective-sync)
      defer_captured_collective_sync=1
      shift
      ;;
    --no-defer-captured-collective-sync)
      defer_captured_collective_sync=0
      shift
      ;;
    --dense-tp)
      dense_tp=1
      shift
      ;;
    --no-dense-tp)
      dense_tp=0
      shift
      ;;
    --dense-decode-replicated)
      dense_decode_replicated=1
      shift
      ;;
    --no-dense-decode-replicated)
      dense_decode_replicated=0
      shift
      ;;
    --dense-policy)
      dense_policy="${2:?missing --dense-policy value}"
      shift 2
      ;;
    --no-dense-policy)
      dense_policy=""
      shift
      ;;
    --allreduce-precision)
      allreduce_precision="${2:?missing --allreduce-precision value}"
      shift 2
      ;;
    --no-allreduce-precision)
      allreduce_precision=""
      shift
      ;;
    --allreduce-fp16-min-elements)
      allreduce_fp16_min_elements="${2:?missing --allreduce-fp16-min-elements value}"
      shift 2
      ;;
    --no-allreduce-fp16-min-elements)
      allreduce_fp16_min_elements=""
      shift
      ;;
    --small-gpu-allreduce)
      small_gpu_allreduce=1
      shift
      ;;
    --no-small-gpu-allreduce)
      small_gpu_allreduce=0
      shift
      ;;
    --small-gpu-allreduce-max-elements)
      small_gpu_allreduce_max_elements="${2:?missing --small-gpu-allreduce-max-elements value}"
      shift 2
      ;;
    --no-small-gpu-allreduce-max-elements)
      small_gpu_allreduce_max_elements=""
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${backend}" in
  cuda|rocm|both) ;;
  *)
    echo "error: --backend must be cuda, rocm, or both" >&2
    exit 2
    ;;
esac

case "${placement}" in
  single|twocard|both) ;;
  *)
    echo "error: --placement must be single, twocard, or both" >&2
    exit 2
    ;;
esac

if [[ "${reps}" -lt 1 ]]; then
  echo "error: --reps must be >= 1" >&2
  exit 2
fi

if [[ "${context_length}" -lt 1 ]]; then
  echo "error: --context-length must be >= 1" >&2
  exit 2
fi

if [[ "${n_predict}" -lt 0 ]]; then
  echo "error: --n-predict must be >= 0" >&2
  exit 2
fi

if [[ "${rebalance_window}" -lt 1 ]]; then
  echo "error: --rebalance-window must be >= 1" >&2
  exit 2
fi

case "${allreduce_precision}" in
  ""|default|off) allreduce_precision="" ;;
  fp16|fp32|bf16) ;;
  *)
    echo "error: --allreduce-precision must be fp16, fp32, or bf16" >&2
    exit 2
    ;;
esac

if [[ -n "${allreduce_fp16_min_elements}" ]]; then
  if ! [[ "${allreduce_fp16_min_elements}" =~ ^[0-9]+$ ]]; then
    echo "error: --allreduce-fp16-min-elements must be a non-negative integer" >&2
    exit 2
  fi
fi

if [[ -n "${small_gpu_allreduce_max_elements}" ]]; then
  if ! [[ "${small_gpu_allreduce_max_elements}" =~ ^[0-9]+$ ]]; then
    echo "error: --small-gpu-allreduce-max-elements must be a non-negative integer" >&2
    exit 2
  fi
fi

if [[ ${dry_run} -eq 0 ]]; then
  [[ -x "${bin}" ]] || { echo "error: binary not executable: ${bin}" >&2; exit 1; }
  [[ -f "${model}" ]] || { echo "error: model not found: ${model}" >&2; exit 1; }
fi

mkdir -p "${out_dir}"
summary="${out_dir}/summary.tsv"
printf 'backend\tplacement\tcase\trep\texit_code\tbenchmark_json\tperf_json\tperf_csv\tlog\n' > "${summary}"

selected_backends=()
if [[ "${backend}" == "both" || "${backend}" == "cuda" ]]; then
  selected_backends+=(cuda)
fi
if [[ "${backend}" == "both" || "${backend}" == "rocm" ]]; then
  selected_backends+=(rocm)
fi

selected_placements=()
if [[ "${placement}" == "both" || "${placement}" == "single" ]]; then
  selected_placements+=(single)
fi
if [[ "${placement}" == "both" || "${placement}" == "twocard" ]]; then
  selected_placements+=(twocard)
fi

IFS=',' read -r -a case_names <<< "${cases_csv}"
for i in "${!case_names[@]}"; do
  case_names[$i]="$(printf '%s' "${case_names[$i]}" | xargs)"
done

case_args() {
  local name="$1"
  case "${name}" in
    static)
      printf '%s\n' --moe-rebalance off --moe-hot-expert-cache off
      ;;
    observe)
      printf '%s\n' --moe-rebalance observe --moe-rebalance-window "${rebalance_window}" --moe-hot-expert-cache off
      ;;
    dynamic)
      printf '%s\n' --moe-rebalance dynamic --moe-rebalance-window "${rebalance_window}" --moe-hot-expert-cache off
      ;;
    dynamic_hot10)
      printf '%s\n' --moe-rebalance dynamic --moe-rebalance-window "${rebalance_window}" --moe-hot-expert-cache 10%
      ;;
    *)
      echo "error: unknown case '${name}'" >&2
      exit 2
      ;;
  esac
}

single_device_args() {
  local be="$1"
  case "${be}" in
    cuda)
      printf '%s\n' -d cuda:0
      ;;
    rocm)
      printf '%s\n' -d rocm:0
      ;;
  esac
}

twocard_overlay_args() {
  local be="$1"
  local domain devices collective
  case "${be}" in
    cuda)
      domain="qwen36_moe_cuda_hot"
      devices="cuda:0,cuda:1"
      collective="nccl"
      ;;
    rocm)
      domain="qwen36_moe_rocm_hot"
      devices="rocm:0,rocm:1"
      collective="rccl"
      ;;
  esac

  printf '%s\n' \
    --moe-expert-overlay tiered \
    --moe-expert-overlay-continuation "${domain}" \
    --moe-expert-overlay-base-domain "${domain}" \
    --moe-expert-overlay-shared-domain "${domain}" \
    --moe-expert-overlay-residency static-by-id \
    --moe-expert-overlay-domain "${domain}=${devices};scope=local;backend=${collective};compute=apportioned_experts;owner=0" \
    --moe-expert-overlay-tier "hot@${domain};priority=0;max-experts-per-layer=256;memory-mb=8192"
}

run_one() {
  local be="$1"
  local place="$2"
  local case_name="$3"
  local rep="$4"
  local run_dir="${out_dir}/${be}/${place}/${case_name}/rep_${rep}"
  mkdir -p "${run_dir}"

  local benchmark_json="${run_dir}/benchmark.json"
  local perf_json="${run_dir}/perfstats.json"
  local perf_csv="${run_dir}/perfstats.csv"
  local log="${run_dir}/stdout_stderr.log"
  local command_file="${run_dir}/command.txt"

  local placement_args=()
  case "${place}" in
    single)
      mapfile -t placement_args < <(single_device_args "${be}")
      ;;
    twocard)
      mapfile -t placement_args < <(twocard_overlay_args "${be}")
      ;;
  esac
  mapfile -t rebalance < <(case_args "${case_name}")

  local cmd=(
    "${bin}" benchmark
    -m "${model}"
    --context-length "${context_length}"
    -n "${n_predict}"
    --benchmark-json-output "${benchmark_json}"
    --moe-release-raw-expert-weights
    "${placement_args[@]}"
    "${rebalance[@]}"
  )
  local dense_tp_enabled=0
  local dense_decode_replicated_enabled=0
  if [[ "${dense_tp}" != "0" && "${dense_tp}" != "false" && "${dense_tp}" != "off" ]]; then
    dense_tp_enabled=1
  fi
  if [[ "${dense_decode_replicated}" != "0" && "${dense_decode_replicated}" != "false" && "${dense_decode_replicated}" != "off" ]]; then
    dense_decode_replicated_enabled=1
  fi
  if [[ "${place}" == "twocard" && -n "${dense_policy}" ]]; then
    cmd+=(--moe-expert-overlay-dense-policy "${dense_policy}")
  elif [[ "${place}" == "twocard" && "${dense_tp_enabled}" == "1" && "${dense_decode_replicated_enabled}" == "1" ]]; then
    cmd+=(--moe-expert-overlay-dense-policy phase-split-hybrid-tp-ae)
  elif [[ "${place}" == "twocard" && "${dense_tp_enabled}" == "1" ]]; then
    cmd+=(--moe-expert-overlay-dense-policy tensor-parallel)
  fi
  if [[ -n "${allreduce_precision}" ]]; then
    cmd+=(--tp-allreduce-precision "${allreduce_precision}")
  fi

  local run_env=(
  )
  if [[ -n "${allreduce_fp16_min_elements}" ]]; then
    run_env+=("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS=${allreduce_fp16_min_elements}")
  fi
  if [[ "${perfstats}" != "0" && "${perfstats}" != "false" && "${perfstats}" != "off" ]]; then
    local perf_filter="moe_rebalance,moe_runtime_decode,transfer,expert_transfer,tp_allreduce_bom,tp_allreduce_runtime,tp_allreduce_small_gpu,forward_graph"
    if [[ "${stage_gpu_stats}" != "0" && "${stage_gpu_stats}" != "false" && "${stage_gpu_stats}" != "off" ]]; then
      perf_filter+=",kernel,kernel_cuda,stage_gpu,forward_pass"
    fi
    run_env+=(
      "LLAMINAR_PERF_STATS_JSON=${perf_json}"
      "LLAMINAR_PERF_STATS_CSV=${perf_csv}"
      "LLAMINAR_PERF_STATS_FILTER=${perf_filter}"
    )
  fi
  if [[ "${place}" == "twocard" ]]; then
    if [[ "${small_gpu_allreduce}" != "0" && "${small_gpu_allreduce}" != "false" && "${small_gpu_allreduce}" != "off" ]]; then
      run_env+=("LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE=1")
      if [[ -n "${small_gpu_allreduce_max_elements}" ]]; then
        run_env+=("LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE_MAX_ELEMENTS=${small_gpu_allreduce_max_elements}")
      fi
    fi
    if [[ "${capture_collectives}" != "0" && "${capture_collectives}" != "false" && "${capture_collectives}" != "off" ]]; then
      run_env+=("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1")
      if [[ -z "${LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED+x}" ]]; then
        run_env+=("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0")
      fi
    elif [[ -z "${LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED+x}" ]]; then
      run_env+=("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1")
    fi
    if [[ "${defer_captured_collective_sync}" != "0" && "${defer_captured_collective_sync}" != "false" && "${defer_captured_collective_sync}" != "off" ]]; then
      run_env+=("LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC=1")
    fi
  fi

  : > "${command_file}"
  if [[ ${#run_env[@]} -gt 0 ]]; then
    printf 'env ' >> "${command_file}"
    printf '%q ' "${run_env[@]}" >> "${command_file}"
  fi
  printf '%q ' "${cmd[@]}" >> "${command_file}"
  printf '\n' >> "${command_file}"

  echo "[qwen36-moe-gpu-rebalance] backend=${be} placement=${place} case=${case_name} rep=${rep}"
  if [[ ${dry_run} -eq 1 ]]; then
    cat "${command_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${be}" "${place}" "${case_name}" "${rep}" "dry-run" \
      "${benchmark_json}" "${perf_json}" "${perf_csv}" "${log}" >> "${summary}"
    return
  fi

  set +e
  if [[ ${#run_env[@]} -gt 0 ]]; then
    env "${run_env[@]}" "${cmd[@]}" >"${log}" 2>&1
  else
    "${cmd[@]}" >"${log}" 2>&1
  fi
  local exit_code=$?
  set -e

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${be}" "${place}" "${case_name}" "${rep}" "${exit_code}" \
    "${benchmark_json}" "${perf_json}" "${perf_csv}" "${log}" >> "${summary}"

  if [[ ${exit_code} -ne 0 ]]; then
    echo "warning: run failed; see ${log}" >&2
  fi
}

for be in "${selected_backends[@]}"; do
  for place in "${selected_placements[@]}"; do
    for case_name in "${case_names[@]}"; do
      for ((rep = 1; rep <= reps; ++rep)); do
        run_one "${be}" "${place}" "${case_name}" "${rep}"
      done
    done
  done
done

echo "summary: ${summary}"
