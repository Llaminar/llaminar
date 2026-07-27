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
  --backend cpu|cpu-prefill|cuda|rocm|all
                              Policy backend/surface to train (required)
  --corpus-root DIR           Published Git LFS corpus root (default:
                              benchmark_results/native_vnni_dispatch/corpora)
  --workspace-root DIR        Ignored resumable collection/refit root (default:
                              benchmark_results/native_vnni_dispatch/work)
  --install                   Atomically install certified M=1/grouped .inc
                              files. Ordinary prefill remains heuristic-only;
                              cpu-prefill is an offline research surface and
                              rejects this option.
  --skip-build                Reuse existing trainer/scorer binaries
  --skip-scorer-tests         Skip CUDA/ROCm scorer integration equivalence
  --no-lfs-pull               Do not run git lfs pull for an existing corpus
  --dry-run                   Print commands without executing them
  --maximum-p95-regret-percent PERCENT
                              Per-domain p95 regret limit (default: 5)
  --minimum-passing-domain-percent PERCENT
                              Required percentage of domains below the p95
                              limit (default: 95). Use 0 for an explicit
                              best-effort install that retains all misses as
                              diagnostics while preserving hard correctness.
  --cpu-minimum-promotion-warmups N
                              Installable fixed-timing CPU evidence floor
                              (default: refresh production default of 5)
  --cpu-minimum-promotion-samples N
                              Installable fixed-timing CPU evidence floor
                              (default: refresh production default of 30)
  -h, --help                  Show this help

Arguments after -- are forwarded to refresh_native_vnni_dispatch_tables.sh.
Corpus identity owns those arguments after publication; changing them selects
a new shape-inventory/backend/architecture/configuration corpus generation.
Production shape subsets are intentionally forbidden here: add exact overlays
to the shared shape inventory instead of creating a backend-private corpus.

Examples:
  scripts/train_native_vnni_dispatch.sh --backend cuda --install
  scripts/train_native_vnni_dispatch.sh --backend cpu-prefill -- \
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
maximum_p95_regret_percent=""
minimum_passing_domain_percent=""
cpu_minimum_promotion_warmups=""
cpu_minimum_promotion_samples=""
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
    --maximum-p95-regret-percent)
      maximum_p95_regret_percent="${2:-}"
      shift 2
      ;;
    --minimum-passing-domain-percent)
      minimum_passing_domain_percent="${2:-}"
      shift 2
      ;;
    --cpu-minimum-promotion-warmups)
      cpu_minimum_promotion_warmups="${2:-}"
      shift 2
      ;;
    --cpu-minimum-promotion-samples)
      cpu_minimum_promotion_samples="${2:-}"
      shift 2
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
  cpu|cpu-prefill|cuda|rocm|all) ;;
  *)
    echo "error: --backend must be cpu, cpu-prefill, cuda, rocm, or all" >&2
    exit 2
    ;;
esac

for argument in "${refresh_arguments[@]}"; do
  case "${argument}" in
    --backend|--profile|--output-dir|--shapes|--shape-partition|--skip-sweep|\
    --reuse-profiler-evidence|--install|--dry-run|\
    --maximum-p95-regret-percent|--minimum-passing-domain-percent|\
    --cpu-minimum-promotion-warmups|--cpu-minimum-promotion-samples|\
    --cpu-prefill-fit-replay-recipe|\
    --backend=*|--profile=*|--output-dir=*|--shapes=*|--shape-partition=*|\
    --maximum-p95-regret-percent=*|--minimum-passing-domain-percent=*|\
    --cpu-minimum-promotion-warmups=*|--cpu-minimum-promotion-samples=*)
      echo "error: ${argument} is owned by the turnkey transaction; add shapes to the shared inventory" >&2
      exit 2
      ;;
  esac
done

if [[ ( -n "${cpu_minimum_promotion_warmups}" ||
        -n "${cpu_minimum_promotion_samples}" ) &&
      "${backend}" != "cpu" && "${backend}" != "all" ]]; then
  echo "error: CPU fixed-timing evidence floors require --backend cpu or all" >&2
  exit 2
fi

if [[ "${backend}" == "cpu-prefill" && ${install} -eq 1 ]]; then
  echo "error: ordinary prefill is heuristic-only; cpu-prefill cannot install a generated policy" >&2
  exit 2
fi

# Promotion criteria are fit-only controls. They must not enter the corpus
# generation identity: one immutable timing/profiler dataset can be mined and
# certified repeatedly under a stricter or more permissive installation gate.
if [[ -n "${cpu_minimum_promotion_warmups}" ]]; then
  refresh_arguments+=(
    --cpu-minimum-promotion-warmups "${cpu_minimum_promotion_warmups}"
  )
fi
if [[ -n "${cpu_minimum_promotion_samples}" ]]; then
  refresh_arguments+=(
    --cpu-minimum-promotion-samples "${cpu_minimum_promotion_samples}"
  )
fi

# A positive batch limit requests a clean, resumable collection checkpoint.
# It changes scheduling only, so the corpus identity code removes it; retain
# the value here solely to distinguish an intentional checkpoint from a
# refresh implementation that incorrectly returned before producing every
# required production artifact.
cpu_batch_limit=0
for ((argument_index = 0;
      argument_index < ${#refresh_arguments[@]};
      ++argument_index)); do
  argument="${refresh_arguments[argument_index]}"
  case "${argument}" in
    --cpu-batch-limit)
      if (( argument_index + 1 >= ${#refresh_arguments[@]} )); then
        echo "error: --cpu-batch-limit requires a value" >&2
        exit 2
      fi
      cpu_batch_limit="${refresh_arguments[argument_index + 1]}"
      argument_index=$((argument_index + 1))
      ;;
    --cpu-batch-limit=*)
      cpu_batch_limit="${argument#*=}"
      ;;
  esac
done
if [[ ! "${cpu_batch_limit}" =~ ^[0-9]+$ ]]; then
  echo "error: --cpu-batch-limit must be a non-negative integer" >&2
  exit 2
fi

if [[ "${backend}" == "all" ]]; then
  overall_status=0
  # The production transaction installs learned policies only for serial M=1
  # and grouped verifier rows. Ordinary prefill deliberately stays on each
  # backend's total legacy heuristic and is not part of this install bundle.
  for selected_backend in cpu cuda rocm; do
    command=(
      "$0"
      --backend "${selected_backend}"
      --corpus-root "${corpus_root}"
      --workspace-root "${workspace_root}"
    )
    (( install )) && command+=(--install)
    (( skip_build )) && command+=(--skip-build)
    (( skip_scorer_tests )) && command+=(--skip-scorer-tests)
    (( ! lfs_pull )) && command+=(--no-lfs-pull)
    (( dry_run )) && command+=(--dry-run)
    if [[ -n "${maximum_p95_regret_percent}" ]]; then
      command+=(
        --maximum-p95-regret-percent "${maximum_p95_regret_percent}"
      )
    fi
    if [[ -n "${minimum_passing_domain_percent}" ]]; then
      command+=(
        --minimum-passing-domain-percent "${minimum_passing_domain_percent}"
      )
    fi
    if [[ "${selected_backend}" == "cpu" &&
          -n "${cpu_minimum_promotion_warmups}" ]]; then
      command+=(
        --cpu-minimum-promotion-warmups "${cpu_minimum_promotion_warmups}"
      )
    fi
    if [[ "${selected_backend}" == "cpu" &&
          -n "${cpu_minimum_promotion_samples}" ]]; then
      command+=(
        --cpu-minimum-promotion-samples "${cpu_minimum_promotion_samples}"
      )
    fi
    if ((${#refresh_arguments[@]})); then
      # Promotion criteria are owned by the turnkey options above. Remove the
      # copies appended for a single-backend refresh before forwarding the
      # caller's remaining diagnostic/collection options.
      forwarded_refresh_arguments=()
      skip_next=0
      for argument in "${refresh_arguments[@]}"; do
        if (( skip_next )); then
          skip_next=0
          continue
        fi
        case "${argument}" in
          --maximum-p95-regret-percent|--minimum-passing-domain-percent|\
          --cpu-minimum-promotion-warmups|--cpu-minimum-promotion-samples)
            skip_next=1
            ;;
          *) forwarded_refresh_arguments+=("${argument}") ;;
        esac
      done
      if ((${#forwarded_refresh_arguments[@]})); then
        command+=(-- "${forwarded_refresh_arguments[@]}")
      fi
    fi
    printf 'NativeVNNI all-backend transaction: starting %s\n' \
      "${selected_backend}"
    if ! "${command[@]}"; then
      overall_status=1
      printf 'NativeVNNI all-backend transaction: %s failed; continuing\n' \
        "${selected_backend}" >&2
    fi
  done
  exit "${overall_status}"
fi

run() {
  if (( dry_run )); then
    printf 'dry-run:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

inventory_sources=()
if [[ "${backend}" == "cpu-prefill" ]]; then
  inventory_sources=(
    "${python_root}/native_vnni_dispatch/manifests/native_vnni_decode_shapes_v5.json"
    "${python_root}/native_vnni_dispatch/manifests/qwen35_qwen36_release_models_v1.json"
    "${python_root}/native_vnni_dispatch/prefill_matrix.py"
    "${python_root}/native_vnni_dispatch/cpu_prefill_training_plan.py"
    "${python_root}/native_vnni_dispatch/cpu_prefill_split_manifest.py"
    "${python_root}/native_vnni_dispatch/manifests/native_vnni_cpu_prefill_split_v13.json"
  )
fi
inventory_command=(
  env "PYTHONPATH=${python_root}"
  python3 -m native_vnni_dispatch.corpus_bundle inventory-id
)
if ((${#inventory_sources[@]})); then
  inventory_command+=(--repository-root "${repo_root}")
  for inventory_source in "${inventory_sources[@]}"; do
    inventory_command+=(--inventory-source "${inventory_source}")
  done
fi
inventory_id="$("${inventory_command[@]}" | sed 's/^sha256://')"
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
  if [[ -n "${maximum_p95_regret_percent}" ]]; then
    command+=(
      --maximum-p95-regret-percent "${maximum_p95_regret_percent}"
    )
  fi
  if [[ -n "${minimum_passing_domain_percent}" ]]; then
    command+=(
      --minimum-passing-domain-percent "${minimum_passing_domain_percent}"
    )
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

if (( ! dry_run )); then
  set +e
  env "PYTHONPATH=${python_root}" \
    python3 -m native_vnni_dispatch.corpus_bundle collection-complete \
      --directory "${staging_dir}" --backend "${backend}" --profile all \
      --quiet
  collection_status=$?
  set -e
  if (( collection_status == 1 && cpu_batch_limit > 0 )); then
    printf 'NativeVNNI collection checkpoint retained: %s\n' "${staging_dir}"
    exit 0
  fi
  if (( collection_status != 0 )); then
    echo "error: refresh returned without a complete publishable corpus" >&2
    exit "${collection_status}"
  fi
fi

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
for inventory_source in "${inventory_sources[@]}"; do
  seal_command+=(--inventory-source "${inventory_source}")
done
run "${seal_command[@]}"
run mkdir -p "$(dirname "${corpus_dir}")"
run mv "${staging_dir}" "${corpus_dir}"
printf 'Published NativeVNNI Git LFS corpus: %s\n' "${corpus_dir}"
