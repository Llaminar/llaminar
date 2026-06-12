#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/refresh_native_vnni_dispatch_tables.sh [options]

Sweeps NativeVNNI decode GEMV dispatch candidates, trains generated C++ dispatch
tables, validates the generated artifacts, and optionally installs them into
src/v2/kernels/{cuda,rocm}/gemm.

Options:
  --backend cuda|rocm|both     Backend to refresh (default: both)
  --profile quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36|all
                              Sweep breadth (default: quick)
  --output-dir DIR             Output directory for CSVs, includes, summaries
  --m-values LIST              Comma list of M buckets (default: 1,2,3,4)
  --cuda-formats LIST          Override CUDA formats
  --rocm-formats LIST          Override ROCm formats
  --shapes LIST                Override shape list shared by CUDA and ROCm
  --cuda-sweep-bin PATH        CUDA sweep binary
  --rocm-decode-bin PATH       ROCm decode trainer binary
  --skip-sweep                 Reuse existing CSVs in output-dir
  --install                    Copy generated includes into src/v2/kernels
  --dry-run                    Print commands without running them
  -h, --help                   Show this help

Build targets:
  cmake --build build_v2_release --parallel \
    --target v2_perf_cuda_blockwise_tensorcore_gemm_sweep \
             v2_perf_native_vnni_throughput

The checked-in tables must come from the generated artifacts emitted here.
Do not hand-edit per-codebook or per-shape runtime overrides.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

backend="both"
profile="quick"
output_dir=""
m_values="1,2,3,4"
cuda_formats=""
rocm_formats=""
shapes=""
cuda_families="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_FAMILIES:-wide,kpar,direct}"
cuda_sweep_bin="${repo_root}/build_v2_release/tests/v2/v2_perf_cuda_blockwise_tensorcore_gemm_sweep"
rocm_decode_bin="${repo_root}/build_v2_release/tests/v2/v2_perf_native_vnni_throughput"
skip_sweep=0
install=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      backend="${2:-}"
      shift 2
      ;;
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --m-values)
      m_values="${2:-}"
      shift 2
      ;;
    --cuda-formats)
      cuda_formats="${2:-}"
      shift 2
      ;;
    --rocm-formats)
      rocm_formats="${2:-}"
      shift 2
      ;;
    --shapes)
      shapes="${2:-}"
      shift 2
      ;;
    --cuda-sweep-bin)
      cuda_sweep_bin="${2:-}"
      shift 2
      ;;
    --rocm-decode-bin)
      rocm_decode_bin="${2:-}"
      shift 2
      ;;
    --skip-sweep)
      skip_sweep=1
      shift
      ;;
    --install)
      install=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      echo "error: unknown option: $1" >&2
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

case "${profile}" in
  quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36|all) ;;
  *)
    echo "error: --profile must be quick, family-smoke, qwen36-core, qwen36-lm-head, qwen36, or all" >&2
    exit 2
    ;;
esac

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/benchmark_results/native_vnni_dispatch/${timestamp}-${backend}-${profile}"
fi

quick_formats="Q4_1,Q5_1,Q6_K"
family_smoke_formats="Q4_0,IQ4_NL,IQ4_XS,Q4_1,Q4_K,Q5_0,Q5_1,Q5_K,Q6_K,Q3_K,Q2_K,IQ3_S,IQ3_XXS,IQ2_S,IQ2_XS,IQ2_XXS,IQ1_S,IQ1_M"
all_formats="Q4_0,IQ4_NL,IQ4_XS,Q4_1,Q4_K,Q5_0,Q5_1,Q5_K,Q6_K,Q3_K,Q2_K,IQ3_S,IQ3_XXS,IQ2_S,IQ2_XS,IQ2_XXS,IQ1_S,IQ1_M"
cuda_all_formats="${all_formats},Q8_0"
cuda_family_smoke_formats="${family_smoke_formats},Q8_0"
quick_shapes="Qwen36_FFN_DownProjection,Qwen36_GDN_OutputProjection"
family_smoke_shapes="Qwen36_GDN_TimeProjection"
qwen36_core_shapes="Qwen36_FFN_GateUp,Qwen36_FFN_DownProjection,Qwen36_GDN_InnerProjection,Qwen36_GDN_ZProjection,Qwen36_GDN_TimeProjection,Qwen36_GDN_OutputProjection"
qwen36_lm_head_shapes="Qwen36_LM_Head"
qwen36_shapes="${qwen36_core_shapes},${qwen36_lm_head_shapes}"

if [[ -z "${cuda_formats}" ]]; then
  case "${profile}" in
    quick) cuda_formats="${quick_formats}" ;;
    family-smoke) cuda_formats="${cuda_family_smoke_formats}" ;;
    qwen36-core|qwen36-lm-head|qwen36|all) cuda_formats="${cuda_all_formats}" ;;
  esac
fi

if [[ -z "${rocm_formats}" ]]; then
  case "${profile}" in
    quick) rocm_formats="${quick_formats}" ;;
    family-smoke) rocm_formats="${family_smoke_formats}" ;;
    qwen36-core|qwen36-lm-head|qwen36|all) rocm_formats="${all_formats}" ;;
  esac
fi

if [[ -z "${shapes}" ]]; then
  case "${profile}" in
    quick) shapes="${quick_shapes}" ;;
    family-smoke) shapes="${family_smoke_shapes}" ;;
    qwen36-core) shapes="${qwen36_core_shapes}" ;;
    qwen36-lm-head) shapes="${qwen36_lm_head_shapes}" ;;
    qwen36|all) shapes="${qwen36_shapes}" ;;
  esac
fi

stratified_formats=0
case "${profile}" in
  quick)
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-24}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-24}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-2}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-0.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-0.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-0.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-0.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-auto,kb1tw8,kb2tw8,kb4tw8,kb4tw12,kb8tw8,kb8tw12,kb8tw24}"
    rocm_reference_mode="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_REFERENCE:-fp32}"
    ;;
  family-smoke)
    stratified_formats=1
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-4}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-4}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-1}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-0.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-0.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-0.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-0.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-auto,kb1tw8,kb2tw8,kb4tw8,kb4tw12,kb8tw8,kb8tw12,kb8tw24}"
    rocm_reference_mode="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_REFERENCE:-fp32}"
    ;;
  qwen36-core|qwen36|all)
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-1000000}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-1000000}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-3}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-99.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-99.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-97.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-30.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-auto,kb1tw4,kb1tw8,kb2tw4,kb2tw8,kb4tw4,kb4tw8,kb4tw12,kb8tw4,kb8tw8,kb8tw12,kb8tw24}"
    rocm_reference_mode="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_REFERENCE:-fp32}"
    ;;
  qwen36-lm-head)
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-1000000}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-1000000}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-3}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-99.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-99.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-97.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-30.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-auto,kb1tw4,kb1tw8,kb2tw4,kb2tw8,kb4tw4,kb4tw8,kb4tw12,kb8tw4,kb8tw8,kb8tw12,kb8tw24}"
    rocm_reference_mode="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_REFERENCE:-native-auto}"
    ;;
esac

run_cmd() {
  local -a env_vars=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do
    env_vars+=("$1")
    shift
  done
  local -a cmd=("$@")
  if (( dry_run )); then
    printf 'dry-run:'
    for item in "${env_vars[@]}"; do
      printf ' %q' "${item}"
    done
    for item in "${cmd[@]}"; do
      printf ' %q' "${item}"
    done
    printf '\n'
    return 0
  fi
  env "${env_vars[@]}" "${cmd[@]}"
}

require_executable() {
  local path="$1"
  if (( dry_run || skip_sweep )); then
    return 0
  fi
  if [[ ! -x "${path}" ]]; then
    echo "error: executable not found: ${path}" >&2
    exit 2
  fi
}

mkdir -p "${output_dir}"

cuda_csv="${output_dir}/cuda_decode_sweep.csv"
cuda_tree_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.tree.inc"
cuda_tree_summary="${output_dir}/cuda_decode_tree_summary.txt"
cuda_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"
cuda_summary="${output_dir}/cuda_decode_dispatch_summary.txt"
rocm_csv="${output_dir}/rocm_decode_sweep.csv"
rocm_inc="${output_dir}/ROCmNativeVNNIDecodeDispatchGenerated.inc"
rocm_summary="${output_dir}/rocm_decode_dispatch_summary.txt"

dispatch_validator="${repo_root}/tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py"
cuda_tree_generator="${repo_root}/tests/v2/performance/kernels/cuda/gemm/infer_gemv_dispatch_heuristic.py"
cuda_overlay_generator="${repo_root}/tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py"
rocm_generator="${repo_root}/tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py"

csv_values() {
  printf '%s\n' "$1" | tr ',' '\n' | sed '/^$/d'
}

combine_csvs() {
  local output="$1"
  shift
  if (( dry_run )); then
    printf 'dry-run: combine-csv %q' "${output}"
    for path in "$@"; do
      printf ' %q' "${path}"
    done
    printf '\n'
    return 0
  fi

  : > "${output}"
  local wrote_header=0
  local header=""
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      echo "error: expected non-empty partial CSV: ${path}" >&2
      exit 2
    fi
    local current_header
    current_header="$(head -n 1 "${path}")"
    if (( ! wrote_header )); then
      header="${current_header}"
      printf '%s\n' "${header}" > "${output}"
      wrote_header=1
    elif [[ "${current_header}" != "${header}" ]]; then
      echo "error: CSV header mismatch while combining ${path}" >&2
      exit 2
    fi
    tail -n +2 "${path}" >> "${output}"
  done
}

refresh_cuda() {
  require_executable "${cuda_sweep_bin}"
  if (( ! skip_sweep )); then
    if (( stratified_formats )); then
      local partials=()
      while IFS= read -r format; do
        local partial="${output_dir}/cuda_decode_sweep.${format}.csv"
        partials+=("${partial}")
        run_cmd \
          "LLAMINAR_CUDA_TC_FORMATS=${format}" \
          "LLAMINAR_CUDA_TC_SHAPES=${shapes}" \
          "LLAMINAR_CUDA_TC_SWEEP_M=${m_values}" \
          "LLAMINAR_CUDA_TC_SWEEP_FAMILIES=${cuda_families}" \
          "LLAMINAR_CUDA_TC_MAX_CASES=${cuda_max_cases}" \
          "LLAMINAR_CUDA_TC_BENCH_RUNS=${cuda_bench_runs}" \
          "LLAMINAR_CUDA_TC_SWEEP_CSV=${partial}" \
          "${cuda_sweep_bin}" \
          "--gtest_filter=*Sweep_GemvDispatchCsv"
      done < <(csv_values "${cuda_formats}")
      combine_csvs "${cuda_csv}" "${partials[@]}"
    else
      run_cmd \
        "LLAMINAR_CUDA_TC_FORMATS=${cuda_formats}" \
        "LLAMINAR_CUDA_TC_SHAPES=${shapes}" \
        "LLAMINAR_CUDA_TC_SWEEP_M=${m_values}" \
        "LLAMINAR_CUDA_TC_SWEEP_FAMILIES=${cuda_families}" \
        "LLAMINAR_CUDA_TC_MAX_CASES=${cuda_max_cases}" \
        "LLAMINAR_CUDA_TC_BENCH_RUNS=${cuda_bench_runs}" \
        "LLAMINAR_CUDA_TC_SWEEP_CSV=${cuda_csv}" \
        "${cuda_sweep_bin}" \
        "--gtest_filter=*Sweep_GemvDispatchCsv"
    fi
  fi

  run_cmd python3 "${cuda_tree_generator}" \
    --input "${cuda_csv}" \
    --output "${cuda_tree_inc}" \
    --summary "${cuda_tree_summary}"

  run_cmd python3 "${cuda_overlay_generator}" \
    --input "${cuda_csv}" \
    --base-include "${cuda_tree_inc}" \
    --output "${cuda_inc}" \
    --summary "${cuda_summary}" \
    --min-overall-family-pct "${cuda_min_overall_family_pct}" \
    --min-overall-exact-pct "${cuda_min_overall_exact_pct}" \
    --min-fallback-family-pct "${cuda_min_fallback_family_pct}" \
    --min-fallback-exact-pct "${cuda_min_fallback_exact_pct}"

  run_cmd python3 "${dispatch_validator}" "${cuda_inc}"

  if (( install )); then
    run_cmd cp "${cuda_inc}" "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"
  fi
}

refresh_rocm() {
  require_executable "${rocm_decode_bin}"
  if (( ! skip_sweep )); then
    if (( stratified_formats )); then
      local partials=()
      while IFS= read -r format; do
        local partial="${output_dir}/rocm_decode_sweep.${format}.csv"
        partials+=("${partial}")
        run_cmd \
          "LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1" \
          "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${format}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${shapes}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_M=${m_values}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=${rocm_reference_mode}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${partial}" \
          "${rocm_decode_bin}" \
          "--gtest_filter=*TrainerCsv_CodebookTagged*"
      done < <(csv_values "${rocm_formats}")
      combine_csvs "${rocm_csv}" "${partials[@]}"
    else
      run_cmd \
        "LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1" \
        "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${rocm_formats}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${shapes}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_M=${m_values}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=${rocm_reference_mode}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${rocm_csv}" \
        "${rocm_decode_bin}" \
        "--gtest_filter=*TrainerCsv_CodebookTagged*"
    fi
  fi

  run_cmd python3 "${rocm_generator}" \
    --input "${rocm_csv}" \
    --output "${rocm_inc}" \
    --summary "${rocm_summary}"

  run_cmd python3 "${dispatch_validator}" "${rocm_inc}"

  if (( install )); then
    run_cmd cp "${rocm_inc}" "${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
  fi
}

case "${backend}" in
  cuda)
    refresh_cuda
    ;;
  rocm)
    refresh_rocm
    ;;
  both)
    refresh_cuda
    refresh_rocm
    ;;
esac

printf 'NativeVNNI dispatch refresh artifacts: %s\n' "${output_dir}"
