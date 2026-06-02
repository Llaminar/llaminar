#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_moe_graph_native_overlay_benchmarks.sh [model-path] [extra llaminar2 benchmark args...]

Environment:
  LLAMINAR_MOE_OVERLAY_MODEL        Model path if not passed as the first argument.
  LLAMINAR_LL2_BIN                  Release binary path (default: build_v2_release/llaminar2).
  LLAMINAR_MOE_OVERLAY_CONFIG_DIR   Config directory (default: configs/moe_overlay).
  LLAMINAR_MOE_OVERLAY_RESULTS_DIR  Output directory override.

The script enables LLAMINAR_PROFILING=1, writes one stdout/stderr log per config,
and intentionally does not use --no-mpi-bootstrap for benchmark runs.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

model_path="${LLAMINAR_MOE_OVERLAY_MODEL:-}"
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  model_path="${1}"
  shift
fi

if [[ -z "${model_path}" ]]; then
  echo "error: provide a model path argument or set LLAMINAR_MOE_OVERLAY_MODEL" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${model_path}" ]]; then
  echo "error: model path does not exist: ${model_path}" >&2
  exit 2
fi

binary_path="${LLAMINAR_LL2_BIN:-${repo_root}/build_v2_release/llaminar2}"
config_dir="${LLAMINAR_MOE_OVERLAY_CONFIG_DIR:-${repo_root}/configs/moe_overlay}"

if [[ ! -x "${binary_path}" ]]; then
  echo "error: Release binary is not executable: ${binary_path}" >&2
  echo "hint: cmake --build build_v2_release --parallel" >&2
  exit 2
fi

if [[ ! -d "${config_dir}" ]]; then
  echo "error: config directory does not exist: ${config_dir}" >&2
  exit 2
fi

for extra_arg in "$@"; do
  if [[ "${extra_arg}" == "--no-mpi-bootstrap" ]]; then
    echo "error: --no-mpi-bootstrap is for profiling/debugging, not benchmark sweeps" >&2
    exit 2
  fi
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
git_hash="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_dir="${LLAMINAR_MOE_OVERLAY_RESULTS_DIR:-${repo_root}/benchmark_results/moe_overlay/${timestamp}-${git_hash}}"
mkdir -p "${run_dir}"

configs=(
  rocm2_replicated_static.yaml
  rocm2_cpu2_replicated_static.yaml
  cuda_hot_rocm_warm_static.yaml
  cuda_hot_rocm_warm_rebalanced.yaml
  cuda_hot_rocm_warm_cpu_cold_static.yaml
  cuda_hot_rocm_warm_cpu_cold_rebalanced.yaml
  all_gpu_capacity_low.yaml
  all_gpu_capacity_medium.yaml
  all_gpu_capacity_high.yaml
  all_gpu_capacity_all_fit.yaml
  mixed_gpu_cpu_hot_cache_low.yaml
  mixed_gpu_cpu_hot_cache_medium.yaml
  mixed_gpu_cpu_hot_cache_high.yaml
)

extract_mpi_ranks() {
  local config_path="$1"
  awk '
    /benchmark: mpi_ranks=/ {
      value=$0
      sub(/^.*mpi_ranks=/, "", value)
      gsub(/[^0-9].*$/, "", value)
      print value
      exit
    }
    /^[[:space:]]*mpi_ranks:[[:space:]]*[0-9]+/ {
      value=$0
      sub(/^.*mpi_ranks:[[:space:]]*/, "", value)
      gsub(/[^0-9].*$/, "", value)
      print value
      exit
    }
  ' "${config_path}"
}

write_inventory() {
  {
    echo "repo_root=${repo_root}"
    echo "git_hash=${git_hash}"
    echo "timestamp_utc=${timestamp}"
    echo "binary=${binary_path}"
    echo "model=${model_path}"
    echo "config_dir=${config_dir}"
    echo "extra_args=$*"
    echo
    git -C "${repo_root}" status --short || true
  } > "${run_dir}/metadata.txt"

  {
    echo "== uname =="
    uname -a || true
    echo
    echo "== CPU =="
    lscpu || true
    echo
    echo "== Memory =="
    free -h || true
    echo
    echo "== PCI =="
    if command -v lspci >/dev/null 2>&1; then
      lspci | grep -Ei 'vga|3d|display|nvidia|amd|instinct|radeon' || true
    else
      echo "lspci not available"
    fi
    echo
    echo "== NVIDIA =="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L || true
      nvidia-smi || true
    else
      echo "nvidia-smi not available"
    fi
    echo
    echo "== ROCm =="
    if command -v rocm-smi >/dev/null 2>&1; then
      rocm-smi || true
    else
      echo "rocm-smi not available"
    fi
  } > "${run_dir}/hardware_inventory.txt" 2>&1
}

write_inventory "$@"

failed=0
for config_name in "${configs[@]}"; do
  config_path="${config_dir}/${config_name}"
  if [[ ! -f "${config_path}" ]]; then
    echo "missing config: ${config_path}" | tee -a "${run_dir}/failures.txt"
    failed=1
    continue
  fi

  mpi_ranks="$(extract_mpi_ranks "${config_path}")"
  if [[ -z "${mpi_ranks}" ]]; then
    echo "missing mpi_ranks metadata in ${config_path}" | tee -a "${run_dir}/failures.txt"
    failed=1
    continue
  fi

  config_stem="${config_name%.yaml}"
  config_run_dir="${run_dir}/${config_stem}"
  mkdir -p "${config_run_dir}"
  cp "${config_path}" "${config_run_dir}/config.yaml"

  profile_csv="${config_run_dir}/moe_overlay_profile.csv"
  command_line=("${binary_path}" benchmark --config "${config_path}" --mpi-procs "${mpi_ranks}" -m "${model_path}" "$@")

  {
    printf 'LLAMINAR_PROFILING=1 LLAMINAR_MOE_EP_PROFILE_CSV=%q ' "${profile_csv}"
    printf '%q ' "${command_line[@]}"
    printf '\n'
  } > "${config_run_dir}/command.txt"

  echo "[moe-overlay] running ${config_name} with ${mpi_ranks} MPI ranks"
  set +e
  LLAMINAR_PROFILING=1 \
  LLAMINAR_MOE_EP_PROFILE_CSV="${profile_csv}" \
    "${command_line[@]}" > "${config_run_dir}/stdout_stderr.log" 2>&1
  status=$?
  set -e

  echo "${status}" > "${config_run_dir}/exit_code.txt"
  if [[ "${status}" -ne 0 ]]; then
    echo "failed: ${config_name} exit=${status}" | tee -a "${run_dir}/failures.txt"
    failed=1
  fi
done

echo "results: ${run_dir}"
exit "${failed}"
