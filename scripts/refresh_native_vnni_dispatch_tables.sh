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
  --backend cuda|rocm|cpu|both|all
                              Backend to refresh (default: both; all includes cpu)
  --profile quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all
                              Sweep breadth (default: quick)
  --output-dir DIR             Output directory for CSVs, includes, summaries
  --m-values LIST              Comma list of M buckets (default: 1..16,31)
  --cuda-formats LIST          Override CUDA formats
  --rocm-formats LIST          Override ROCm formats
  --cpu-formats LIST           Override CPU formats
  --cpu-threads N              OpenMP threads per CPU policy regime; defaults
                               to detected physical cores per socket
  --shapes LIST                Override shape list shared by CUDA and ROCm
  --cuda-sweep-bin PATH        CUDA sweep binary
  --rocm-decode-bin PATH       ROCm decode trainer binary
  --cpu-avx2-sweep-bin PATH    AVX2-only CPU verifier trainer binary
  --cpu-avx512-sweep-bin PATH  AVX512 CPU verifier trainer binary; this binary
                               is run once with AVX2 and once with AVX512
                               runtime dispatch
  --skip-sweep                 Reuse existing CSVs in output-dir
  --install                    Copy generated includes into src/v2/kernels
  --dry-run                    Print commands without running them
  -h, --help                   Show this help

Build targets:
  cmake --build build_v2_release --parallel \
    --target v2_perf_cuda_blockwise_tensorcore_gemm_sweep \
             v2_perf_native_vnni_throughput
  cmake --build build_v2_release_avx2 --parallel \
    --target v2_perf_cpu_native_vnni_gemv
  cmake --build build_v2_release_avx512 --parallel \
    --target v2_perf_cpu_native_vnni_gemv

The checked-in tables must come from the generated artifacts emitted here.
Do not hand-edit per-codebook or per-shape runtime overrides.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

backend="both"
profile="quick"
output_dir=""
m_values="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
cuda_formats=""
rocm_formats=""
cpu_formats=""
cpu_threads="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_THREADS:-}"
shapes=""
cuda_families="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_FAMILIES:-wide,kpar,direct}"
cuda_sweep_bin="${repo_root}/build_v2_release/tests/v2/v2_perf_cuda_blockwise_tensorcore_gemm_sweep"
rocm_decode_bin="${repo_root}/build_v2_release/tests/v2/v2_perf_native_vnni_throughput"
cpu_avx2_sweep_bin="${repo_root}/build_v2_release_avx2/tests/v2/v2_perf_cpu_native_vnni_gemv"
cpu_avx512_sweep_bin="${repo_root}/build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_gemv"
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
    --cpu-formats)
      cpu_formats="${2:-}"
      shift 2
      ;;
    --cpu-threads)
      cpu_threads="${2:-}"
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
    --cpu-avx2-sweep-bin)
      cpu_avx2_sweep_bin="${2:-}"
      shift 2
      ;;
    --cpu-avx512-sweep-bin)
      cpu_avx512_sweep_bin="${2:-}"
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
  cuda|rocm|cpu|both|all) ;;
  *)
    echo "error: --backend must be cuda, rocm, cpu, both, or all" >&2
    exit 2
    ;;
esac

case "${profile}" in
  quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all) ;;
  *)
    echo "error: --profile must be quick, family-smoke, qwen36-core, qwen36-lm-head, qwen36-moe, qwen36, or all" >&2
    exit 2
    ;;
esac

case "${profile}" in
  quick) cpu_measurement_profile="quick" ;;
  family-smoke) cpu_measurement_profile="family-smoke" ;;
  qwen36-core|qwen36-lm-head|qwen36-moe)
    cpu_measurement_profile="partial-production"
    ;;
  qwen36|all) cpu_measurement_profile="production" ;;
esac

if (( install )) && [[ "${backend}" == "cpu" || "${backend}" == "all" ]] && \
   [[ "${cpu_measurement_profile}" != "production" ]]; then
  echo "error: installing CPU dispatch requires --profile qwen36 or all" >&2
  exit 2
fi

detect_cpu_threads_per_socket() {
  local detected
  detected="$(lscpu -p=CORE,SOCKET 2>/dev/null | awk -F, '
    $0 !~ /^#/ { seen[$2 ":" $1] = 1 }
    END {
      for (key in seen) {
        split(key, parts, ":")
        count[parts[1]]++
      }
      best = 0
      for (socket in count)
        if (count[socket] > best) best = count[socket]
      if (best > 0) print best
    }')"
  if [[ -n "${detected}" ]]; then
    printf '%s\n' "${detected}"
  else
    nproc
  fi
}

cpu_lscpu_field() {
  local label="$1"
  lscpu | awk -F: -v wanted="${label}" '
    $1 == wanted {
      value = $2
      sub(/^[[:space:]]+/, "", value)
      print value
      exit
    }'
}

cpu_architecture_class() {
  local vendor family model stepping sockets cores
  vendor="$(cpu_lscpu_field "Vendor ID")"
  family="$(cpu_lscpu_field "CPU family")"
  model="$(cpu_lscpu_field "Model")"
  stepping="$(cpu_lscpu_field "Stepping")"
  sockets="$(cpu_lscpu_field "Socket(s)")"
  cores="$(cpu_lscpu_field "Core(s) per socket")"
  printf 'x86_64-%s-family%s-model%s-step%s-%ssockets-%scores\n' \
    "${vendor}" "${family}" "${model}" "${stepping}" "${sockets}" "${cores}"
}

sha256_file_set() {
  sha256sum "$@" | sha256sum | awk '{print "sha256:" $1}'
}

if [[ -z "${cpu_threads}" ]]; then
  cpu_threads="$(detect_cpu_threads_per_socket)"
fi
if [[ ! "${cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cpu-threads must be a positive integer" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/benchmark_results/native_vnni_dispatch/${timestamp}-${backend}-${profile}"
fi

quick_formats="Q4_1,Q5_1,Q6_K"
family_smoke_formats="Q4_0,IQ4_NL,IQ4_XS,Q4_1,Q4_K,Q5_0,Q5_1,Q5_K,Q6_K,Q3_K,Q2_K,IQ3_S,IQ3_XXS,IQ2_S,IQ2_XS,IQ2_XXS,IQ1_S,IQ1_M,Q8_0,Q8_1,Q8_K"
all_formats="${family_smoke_formats}"
cuda_all_formats="${all_formats}"
cuda_family_smoke_formats="${family_smoke_formats}"
rocm_all_formats="${all_formats}"
rocm_family_smoke_formats="${family_smoke_formats}"
cpu_all_formats="${all_formats}"
cpu_family_smoke_formats="${family_smoke_formats}"
quick_shapes="Qwen36_FFN_DownProjection,Qwen36_GDN_OutputProjection"
family_smoke_shapes="Qwen36_GDN_TimeProjection"
qwen36_core_shapes="Qwen36_Attn_QKVProjection,Qwen36_FFN_GateUp,Qwen36_FFN_DownProjection,Qwen36_GDN_InnerProjection,Qwen36_GDN_ZProjection,Qwen36_GDN_TimeProjection,Qwen36_GDN_OutputProjection"
qwen36_lm_head_shapes="Qwen36_LM_Head"
qwen36_moe_shapes="35BMoE_Expert_GateUp,35BMoE_Expert_Down,Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection"
qwen36_shapes="${qwen36_core_shapes},${qwen36_lm_head_shapes},${qwen36_moe_shapes}"

if [[ -z "${cuda_formats}" ]]; then
  case "${profile}" in
    quick) cuda_formats="${quick_formats}" ;;
    family-smoke) cuda_formats="${cuda_family_smoke_formats}" ;;
    qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all) cuda_formats="${cuda_all_formats}" ;;
  esac
fi

if [[ -z "${rocm_formats}" ]]; then
  case "${profile}" in
    quick) rocm_formats="${quick_formats}" ;;
    family-smoke) rocm_formats="${rocm_family_smoke_formats}" ;;
    qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all) rocm_formats="${rocm_all_formats}" ;;
  esac
fi

if [[ -z "${cpu_formats}" ]]; then
  case "${profile}" in
    quick) cpu_formats="${quick_formats}" ;;
    family-smoke) cpu_formats="${cpu_family_smoke_formats}" ;;
    qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all) cpu_formats="${cpu_all_formats}" ;;
  esac
fi

if [[ -z "${shapes}" ]]; then
  case "${profile}" in
    quick) shapes="${quick_shapes}" ;;
    family-smoke) shapes="${family_smoke_shapes}" ;;
    qwen36-core) shapes="${qwen36_core_shapes}" ;;
    qwen36-lm-head) shapes="${qwen36_lm_head_shapes}" ;;
    qwen36-moe) shapes="${qwen36_moe_shapes}" ;;
    qwen36|all) shapes="${qwen36_shapes}" ;;
  esac
fi

stratified_formats=0
cpu_require_policy_keys=0
rocm_execution_modes="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_EXECUTION_MODES:-eager,graph_captured}"
rocm_quick_variants="kb1,kb2,kb4,kb8,kb16,kb32,kb64,inherit_serial_m1"
case "${profile}" in
  quick)
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-24}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-24}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-2}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-0.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-0.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-0.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-0.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-${rocm_quick_variants}}"
    rocm_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_WARMUPS:-2}"
    rocm_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_SAMPLES:-5}"
    cpu_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_MAX_CASES:-24}"
    cpu_warmup="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP:-2}"
    cpu_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS:-5}"
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
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-${rocm_quick_variants}}"
    rocm_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_WARMUPS:-2}"
    rocm_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_SAMPLES:-5}"
    cpu_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_MAX_CASES:-1000000}"
    cpu_warmup="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP:-2}"
    cpu_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS:-5}"
    ;;
  qwen36-core|qwen36-moe|qwen36|all)
    cpu_require_policy_keys=1
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-1000000}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-1000000}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-3}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-99.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-99.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-97.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-30.0}"
    # An empty override selects the trainer's complete KB1..64 inventory plus
    # the explicit grouped-verifier INHERIT_SERIAL_M1 candidate.
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-}"
    rocm_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_WARMUPS:-5}"
    rocm_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_SAMPLES:-30}"
    cpu_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_MAX_CASES:-1000000}"
    cpu_warmup="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP:-5}"
    cpu_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS:-30}"
    ;;
  qwen36-lm-head)
    cpu_require_policy_keys=1
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-1000000}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-1000000}"
    cuda_bench_runs="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_BENCH_RUNS:-3}"
    cuda_min_overall_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_FAMILY_PCT:-99.0}"
    cuda_min_overall_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_OVERALL_EXACT_PCT:-99.0}"
    cuda_min_fallback_family_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_FAMILY_PCT:-97.0}"
    cuda_min_fallback_exact_pct="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MIN_FALLBACK_EXACT_PCT:-30.0}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-}"
    rocm_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_WARMUPS:-5}"
    rocm_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_SAMPLES:-30}"
    cpu_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_MAX_CASES:-1000000}"
    cpu_warmup="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP:-5}"
    cpu_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS:-30}"
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
rocm_timing_csv="${output_dir}/rocm_decode_sweep.timing.csv"
rocm_inc="${output_dir}/ROCmNativeVNNIDecodeDispatchGenerated.inc"
rocm_summary="${output_dir}/rocm_decode_dispatch_summary.txt"
rocm_common_csv="${output_dir}/rocm_decode_common_observations.csv"
rocm_base_include="${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
cpu_csv="${output_dir}/cpu_verifier_rows_sweep.csv"
cpu_timing_csv="${output_dir}/cpu_verifier_rows_sweep.timing.csv"
cpu_inc="${output_dir}/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
cpu_summary="${output_dir}/cpu_verifier_rows_policy_summary.txt"
cpu_common_csv="${output_dir}/cpu_verifier_rows_common_observations.csv"

dispatch_validator="${repo_root}/tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py"
cuda_tree_generator="${repo_root}/tests/v2/performance/kernels/cuda/gemm/infer_gemv_dispatch_heuristic.py"
cuda_overlay_generator="${repo_root}/tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py"
rocm_generator="${repo_root}/tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py"
cpu_generator="${repo_root}/tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py"

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

cpu_shape_n() {
  case "$1" in
    Qwen36_FFN_GateUp) printf '17408' ;;
    Qwen36_Attn_QKVProjection) printf '12288' ;;
    Qwen36_FFN_DownProjection) printf '5120' ;;
    Qwen36_GDN_InnerProjection) printf '10240' ;;
    Qwen36_GDN_ZProjection) printf '6144' ;;
    Qwen36_GDN_TimeProjection) printf '1024' ;;
    Qwen36_GDN_OutputProjection) printf '5120' ;;
    Qwen36_LM_Head) printf '248320' ;;
    35BMoE_Expert_GateUp) printf '512' ;;
    35BMoE_Expert_Down) printf '2048' ;;
    Qwen36MoE_GDN_QKVProjection) printf '8192' ;;
    Qwen36MoE_GDN_ZProjection) printf '4096' ;;
    *)
      echo "error: unsupported CPU NativeVNNI verifier shape: $1" >&2
      exit 2
      ;;
  esac
}

cpu_shape_k() {
  case "$1" in
    Qwen36_FFN_GateUp) printf '5120' ;;
    Qwen36_Attn_QKVProjection) printf '5120' ;;
    Qwen36_FFN_DownProjection) printf '17408' ;;
    Qwen36_GDN_InnerProjection) printf '5120' ;;
    Qwen36_GDN_ZProjection) printf '5120' ;;
    Qwen36_GDN_TimeProjection) printf '5120' ;;
    Qwen36_GDN_OutputProjection) printf '6144' ;;
    Qwen36_LM_Head) printf '5120' ;;
    35BMoE_Expert_GateUp) printf '2048' ;;
    35BMoE_Expert_Down) printf '512' ;;
    Qwen36MoE_GDN_QKVProjection) printf '2048' ;;
    Qwen36MoE_GDN_ZProjection) printf '2048' ;;
    *)
      echo "error: unsupported CPU NativeVNNI verifier shape: $1" >&2
      exit 2
      ;;
  esac
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
      local timing_partials=()
      while IFS= read -r format; do
        local partial="${output_dir}/rocm_decode_sweep.${format}.csv"
        local timing_partial="${output_dir}/rocm_decode_sweep.${format}.timing.csv"
        partials+=("${partial}")
        timing_partials+=("${timing_partial}")
        run_cmd \
          "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${format}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${shapes}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_M=${m_values}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=${rocm_execution_modes}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS=${rocm_warmups}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES=${rocm_samples}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${partial}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
          "${rocm_decode_bin}" \
          "--gtest_filter=*TrainerCsv_CodebookTagged*"
      done < <(csv_values "${rocm_formats}")
      combine_csvs "${rocm_csv}" "${partials[@]}"
      combine_csvs "${rocm_timing_csv}" "${timing_partials[@]}"
    else
      run_cmd \
        "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${rocm_formats}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${shapes}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_M=${m_values}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=${rocm_execution_modes}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS=${rocm_warmups}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES=${rocm_samples}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${rocm_csv}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=${rocm_timing_csv}" \
        "${rocm_decode_bin}" \
        "--gtest_filter=*TrainerCsv_CodebookTagged*"
    fi
  fi

  local -a rocm_generator_args=(
    python3 "${rocm_generator}"
    --input "${rocm_csv}"
    --timing-sidecar "${rocm_timing_csv}"
    --output "${rocm_inc}"
    --summary "${rocm_summary}"
    --common-observations "${rocm_common_csv}"
  )
  case "${profile}" in
    qwen36-core|qwen36-lm-head|qwen36-moe)
      rocm_generator_args+=(--base-include "${rocm_base_include}")
      ;;
  esac
  run_cmd "${rocm_generator_args[@]}"

  run_cmd python3 "${dispatch_validator}" "${rocm_inc}"

  if (( install )); then
    run_cmd cp "${rocm_inc}" "${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
  fi
}

refresh_cpu() {
  require_executable "${cpu_avx2_sweep_bin}"
  require_executable "${cpu_avx512_sweep_bin}"
  local cpu_git_revision cpu_build_id cpu_compiler_id cpu_arch_class
  local cpu_device_name cpu_driver_runtime cpu_serial_policy_hash
  cpu_git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
  cpu_build_id="$(sha256_file_set \
    "${cpu_avx2_sweep_bin}" "${cpu_avx512_sweep_bin}")"
  cpu_compiler_id="$(${CXX:-c++} --version | head -n 1)"
  cpu_arch_class="$(cpu_architecture_class)"
  cpu_device_name="$(cpu_lscpu_field "Model name")"
  cpu_driver_runtime="linux-$(uname -r)-openmp"
  cpu_serial_policy_hash="$(sha256_file_set \
    "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h" \
    "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeAVX2Gemv.h" \
    "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIDecode.h" \
    "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h")"
  if (( ! skip_sweep )); then
    local partials=()
    local timing_partials=()
    local format_shards=()
    if (( stratified_formats )); then
      mapfile -t format_shards < <(csv_values "${cpu_formats}")
    else
      format_shards=("${cpu_formats}")
    fi

    local regime_specs=(
      "avx2-build.avx2-runtime|avx2|${cpu_avx2_sweep_bin}"
      "avx512-build.avx2-runtime|avx2|${cpu_avx512_sweep_bin}"
      "avx512-build.avx512-runtime|avx512|${cpu_avx512_sweep_bin}"
    )
    local format_spec shape n k regime_spec regime_name runtime_isa regime_bin
    local format_label partial timing_partial
    for format_spec in "${format_shards[@]}"; do
      if (( stratified_formats )); then
        format_label="${format_spec}"
      else
        format_label="all-formats"
      fi
      while IFS= read -r shape; do
        n="$(cpu_shape_n "${shape}")"
        k="$(cpu_shape_k "${shape}")"
        for regime_spec in "${regime_specs[@]}"; do
          IFS='|' read -r regime_name runtime_isa regime_bin <<< "${regime_spec}"
          partial="${output_dir}/cpu_verifier_rows.${format_label}.${shape}.${regime_name}.csv"
          timing_partial="${output_dir}/cpu_verifier_rows.${format_label}.${shape}.${regime_name}.timing.csv"
          partials+=("${partial}")
          timing_partials+=("${timing_partial}")
          run_cmd \
            "LLAMINAR_ISA_LEVEL=${runtime_isa}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=${format_spec}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_M=${m_values}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=${shape}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_N=${n}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_K=${k}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_MAX_CASES=${cpu_max_cases}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_THREADS=${cpu_threads}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=${cpu_warmup}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=${cpu_iters}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=${partial}" \
            "LLAMINAR_CPU_NVNNI_VERIFIER_TIMING_CSV=${timing_partial}" \
            "${regime_bin}" \
            "--gtest_filter=*TrainerCsv_StrongVerifierRows_AllFormats"
        done
      done < <(csv_values "${shapes}")
    done
    combine_csvs "${cpu_csv}" "${partials[@]}"
    combine_csvs "${cpu_timing_csv}" "${timing_partials[@]}"
  fi

  local cpu_require_args=()
  if (( cpu_require_policy_keys )); then
    while IFS= read -r format; do
      while IFS= read -r shape; do
        local n k
        n="$(cpu_shape_n "${shape}")"
        k="$(cpu_shape_k "${shape}")"
        while IFS= read -r m; do
          if (( m < 2 || m > 4 )); then
            continue
          fi
          cpu_require_args+=(--require-key "${format}:${m}:${n}:${k}")
        done < <(csv_values "${m_values}")
      done < <(csv_values "${shapes}")
    done < <(csv_values "${cpu_formats}")
  fi

  local cpu_generator_args=(
    python3 "${cpu_generator}"
    --input "${cpu_csv}"
    --timing-sidecar "${cpu_timing_csv}"
    --output "${cpu_inc}"
    --summary "${cpu_summary}"
    --common-observations "${cpu_common_csv}"
    --require-isa-matrix
    --profile "${cpu_measurement_profile}"
    --run-id "${timestamp}-cpu-${profile}"
    --git-revision "${cpu_git_revision}"
    --build-id "${cpu_build_id}"
    --compiler-id "${cpu_compiler_id}"
    --architecture-class "${cpu_arch_class}"
    --device-name "${cpu_device_name}"
    --driver-runtime "${cpu_driver_runtime}"
    --serial-m1-policy-hash "${cpu_serial_policy_hash}"
  )
  if [[ "${cpu_measurement_profile}" == "production" ||
        "${cpu_measurement_profile}" == "family-smoke" ]]; then
    cpu_generator_args+=(--require-complete)
  fi
  cpu_generator_args+=("${cpu_require_args[@]}")
  run_cmd "${cpu_generator_args[@]}"

  run_cmd python3 "${dispatch_validator}" "${cpu_inc}"

  if (( install )); then
    run_cmd cp "${cpu_inc}" "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
  fi
}

case "${backend}" in
  cuda)
    refresh_cuda
    ;;
  rocm)
    refresh_rocm
    ;;
  cpu)
    refresh_cpu
    ;;
  both)
    refresh_cuda
    refresh_rocm
    ;;
  all)
    refresh_cuda
    refresh_rocm
    refresh_cpu
    ;;
esac

printf 'NativeVNNI dispatch refresh artifacts: %s\n' "${output_dir}"
