#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/train_native_vnni_dispatch.sh --backend BACKEND [options] [-- REFRESH_OPTIONS]

Runs the complete NativeVNNI policy transaction for one backend. The command
rebuilds the trainer and both GPU policy scorers, selects a corpus by backend,
architecture, and the shared shape-inventory digest, then does one of two
things:

  * no matching corpus: benchmark, profile, fit, certify, optionally install,
    seal the evidence, and atomically publish it beneath the Git LFS corpus root;
  * matching corpus: pull/verify it, materialize a disposable fit workspace,
    refit and certify without kernel/profiler launches, and optionally install.

Options:
  --backend cpu|cpu-prefill|cuda|rocm
                              Policy backend/surface to train (required)
  --corpus-root DIR           Published Git LFS corpus root (default:
                              benchmark_results/native_vnni_dispatch/corpora)
  --workspace-root DIR        Ignored resumable collection/refit root (default:
                              benchmark_results/native_vnni_dispatch/work)
  --install                   Atomically install the certified generated .inc
  --skip-build                Reuse existing trainer/scorer binaries
  --skip-scorer-tests         Skip CUDA/ROCm scorer integration equivalence
  --no-lfs-pull               Do not run git lfs pull for an existing corpus
  --dry-run                   Print commands without executing them
  -h, --help                  Show this help

Arguments after -- are forwarded to refresh_native_vnni_dispatch_tables.sh.
Corpus identity owns those arguments after publication; changing them selects
a new shape-inventory/backend/architecture/configuration corpus generation.
Production shape subsets are intentionally forbidden here: add exact overlays
to the shared shape inventory instead of creating a backend-private corpus.

Examples:
  scripts/train_native_vnni_dispatch.sh --backend cuda --install
  scripts/train_native_vnni_dispatch.sh --backend cpu-prefill --install -- \
    --cpu-format-shards
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_root="${repo_root}/tests/v2/performance/kernels"
refresh_script="${script_dir}/refresh_native_vnni_dispatch_tables.sh"

backend=""
corpus_root="${repo_root}/benchmark_results/native_vnni_dispatch/corpora"
workspace_root="${repo_root}/benchmark_results/native_vnni_dispatch/work"
install=0
skip_build=0
skip_scorer_tests=0
lfs_pull=1
dry_run=0
refresh_arguments=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="${2:-}"
      shift 2
      ;;
    --corpus-root)
      corpus_root="${2:-}"
      shift 2
      ;;
    --workspace-root)
      workspace_root="${2:-}"
      shift 2
      ;;
    --install)
      install=1
      shift
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-scorer-tests)
      skip_scorer_tests=1
      shift
      ;;
    --no-lfs-pull)
      lfs_pull=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --)
      shift
      refresh_arguments=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option $1 (refresh options belong after --)" >&2
      exit 2
      ;;
  esac
done

case "${backend}" in
  cpu|cpu-prefill|cuda|rocm) ;;
  *)
    echo "error: --backend must be cpu, cpu-prefill, cuda, or rocm" >&2
    exit 2
    ;;
esac

for argument in "${refresh_arguments[@]}"; do
  case "${argument}" in
    --backend|--profile|--output-dir|--shapes|--shape-partition|--skip-sweep|\
    --reuse-profiler-evidence|--install|--dry-run|\
    --cpu-prefill-fit-replay-recipe|\
    --backend=*|--profile=*|--output-dir=*|--shapes=*|--shape-partition=*)
      echo "error: ${argument} is owned by the turnkey transaction; add shapes to the shared inventory" >&2
      exit 2
      ;;
  esac
done

run() {
  if (( dry_run )); then
    printf 'dry-run:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

inventory_id="$({
  PYTHONPATH="${python_root}" \
    python3 -m native_vnni_dispatch.corpus_bundle inventory-id
} | sed 's/^sha256://')"
inventory_short="${inventory_id:0:16}"
configuration_command=(
  env "PYTHONPATH=${python_root}"
  python3 -m native_vnni_dispatch.corpus_bundle configuration-id
)
for argument in "${refresh_arguments[@]}"; do
  configuration_command+=("--refresh-argument=${argument}")
done
configuration_id="$("${configuration_command[@]}" | sed 's/^sha256://')"
configuration_short="${configuration_id:0:12}"

sanitize_identity() {
  tr '[:upper:]' '[:lower:]' |
    sed 's/[^a-z0-9._-]\+/-/g; s/^-//; s/-$//'
}

architecture_identity() {
  case "${backend}" in
    cuda)
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i 0 |
          awk 'NR == 1 { print; exit }' | sanitize_identity
      else
        printf 'cuda-unknown\n'
      fi
      ;;
    rocm)
      if command -v rocminfo >/dev/null 2>&1; then
        rocminfo | awk '/Name:[[:space:]]+gfx/ { print $2; exit }' |
          sanitize_identity
      else
        printf 'rocm-unknown\n'
      fi
      ;;
    cpu|cpu-prefill)
      {
        uname -m
        lscpu | awk -F: '/Model name/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }'
      } | paste -sd- - | sanitize_identity
      ;;
  esac
}

architecture="$(architecture_identity)"
if [[ -z "${architecture}" ]]; then
  architecture="${backend}-unknown"
fi
corpus_dir="${corpus_root}/${backend}/${architecture}/${inventory_short}-${configuration_short}"
manifest="${corpus_dir}/corpus.manifest.json"
staging_dir="${workspace_root}/collect-${backend}-${architecture}-${inventory_short}-${configuration_short}"
fit_dir="${workspace_root}/fit-${backend}-${architecture}-${inventory_short}-${configuration_short}"

build_targets() {
  if (( skip_build )); then
    return
  fi

  # Both vendor scorers are always rebuilt when available. Policy fitting can
  # then partition tree-search leaves over every CUDA and ROCm device without
  # coupling the scorer vendor to the measured policy backend.
  run cmake --build "${repo_root}/build_v2_release" --parallel \
    --target v2_native_vnni_leaf_primary_scorer_cuda \
             v2_native_vnni_leaf_primary_scorer_rocm
  case "${backend}" in
    cuda)
      run cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_cuda_native_vnni_decode_trainer
      ;;
    rocm)
      run cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_native_vnni_throughput
      ;;
    cpu|cpu-prefill)
      run cmake --build "${repo_root}/build_v2_release_avx2" --parallel \
        --target v2_perf_cpu_native_vnni_gemv
      run cmake --build "${repo_root}/build_v2_release_avx512" --parallel \
        --target v2_perf_cpu_native_vnni_gemv
      ;;
  esac
}

validate_scorers() {
  if (( skip_scorer_tests )); then
    return
  fi
  run ctest --test-dir "${repo_root}/build_v2_integration" \
    -R '^V2_Integration_NativeVNNILeafPrimaryScorer_(CUDA|ROCm)$' \
    --output-on-failure --parallel
}

refresh_command() {
  local output_dir="$1"
  shift
  local command=(
    "${refresh_script}"
    --backend "${backend}"
    --profile all
    --output-dir "${output_dir}"
    "$@"
  )
  local replay_recipe="${output_dir}/cpu_prefill_replay_recipe.v1.json"
  if [[ "${backend}" == "cpu-prefill" && -s "${replay_recipe}" ]]; then
    command+=(--cpu-prefill-fit-replay-recipe "${replay_recipe}")
  fi
  if (( install )); then
    command+=(--install)
  fi
  command+=("${refresh_arguments[@]}")
  run "${command[@]}"
}

build_targets
validate_scorers

if [[ -f "${manifest}" ]]; then
  if (( lfs_pull )); then
    if ! git -C "${repo_root}" lfs version >/dev/null 2>&1; then
      echo "error: git-lfs is required to materialize ${manifest}; install it or use --no-lfs-pull only for an already-materialized checkout" >&2
      exit 2
    fi
    corpus_relative="${corpus_dir#"${repo_root}/"}"
    run git -C "${repo_root}" lfs pull --include "${corpus_relative}/**"
  fi
  run env "PYTHONPATH=${python_root}" \
    python3 -m native_vnni_dispatch.corpus_bundle verify \
      --manifest "${manifest}" --repository-root "${repo_root}"

  if (( dry_run )); then
    run cp -a --reflink=auto "${corpus_dir}/." "${fit_dir}/"
  else
    rm -rf "${fit_dir}"
    mkdir -p "${fit_dir}"
    cp -a --reflink=auto "${corpus_dir}/." "${fit_dir}/"
    rm -f "${fit_dir}/corpus.manifest.json"
  fi
  refresh_command "${fit_dir}" --skip-sweep --reuse-profiler-evidence
  printf 'NativeVNNI fit workspace: %s\n' "${fit_dir}"
  exit 0
fi

if [[ -e "${corpus_dir}" ]]; then
  echo "error: unsealed published corpus directory exists: ${corpus_dir}" >&2
  exit 2
fi
run mkdir -p "${staging_dir}"
collection_arguments=()
if [[ "${backend}" == "cpu" &&
      -s "${staging_dir}/cpu_collection_contract.sha256" ]]; then
  collection_arguments+=(--resume-cpu-partials)
elif [[ "${backend}" == "cpu-prefill" &&
        -s "${staging_dir}/cpu_prefill_collection_contract.sha256" ]]; then
  collection_arguments+=(--resume-cpu-partials)
fi
refresh_command "${staging_dir}" "${collection_arguments[@]}"

seal_command=(
  env "PYTHONPATH=${python_root}"
  python3 -m native_vnni_dispatch.corpus_bundle seal
  --directory "${staging_dir}"
  --backend "${backend}"
  --profile all
  --architecture "${architecture}"
  --repository-root "${repo_root}"
)
for argument in "${refresh_arguments[@]}"; do
  seal_command+=("--refresh-argument=${argument}")
done
run "${seal_command[@]}"
run mkdir -p "$(dirname "${corpus_dir}")"
run mv "${staging_dir}" "${corpus_dir}"
printf 'Published NativeVNNI Git LFS corpus: %s\n' "${corpus_dir}"
