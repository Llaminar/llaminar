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
  --backend cuda|rocm|cpu|cpu-prefill|both|all
                              Backend to refresh (default: both; all includes
                              CPU verifier and ordinary-prefill policies)
  --profile quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all
                              Sweep breadth (default: quick)
  --output-dir DIR             Output directory for CSVs, includes, summaries
  --m-values LIST              Comma list of M buckets. Production requires
                               exactly serial M=1 plus verifier M=2..16,31.
  --cuda-formats LIST          Override CUDA formats
  --rocm-formats LIST          Override ROCm formats
  --cpu-formats LIST           Override CPU formats
  --cuda-measurement-lanes N   Disjoint format shards on homogeneous CUDA GPUs
                               (default: all visible GPUs for a CUDA run)
  --rocm-measurement-lanes N   Disjoint format shards on homogeneous ROCm GPUs
                               (default: all visible GPUs for a ROCm run)
  --cpu-threads N              OpenMP threads per CPU policy regime; defaults
                               to detected physical cores per socket
  --cpu-measurement-lanes N|auto
                               Independent socket-local CPU timing jobs
                               (default: every detected physical socket)
  --cpu-format-shards          Run one CPU source format per measurement job
                               instead of all formats in each shape process
  --cpu-batch-limit N          Stop successfully after N pending CPU MPMD
                               launches; zero means unlimited (default: 0)
  --backend-collection-target-seconds N
                              Target wall-clock budget for one backend's timing,
                              profiling, generation, and validation transaction;
                              overruns are reported but do not kill collection
                              (default: 7200 seconds / two hours)
  --maximum-p95-regret-percent PERCENT
                              Strict per-domain observed/UCB regret budget used
                              by fitting and sealed certification (default: 5)
  --minimum-passing-domain-percent PERCENT
                              Minimum percentage of complete generic domains
                              that must meet the p95 budget (default: 95). A
                              best-effort install may explicitly lower this;
                              correctness and structural gates remain mandatory.
  --cpu-minimum-promotion-warmups N
                              Minimum warmups admitted for installable fixed-
                              timing CPU M=1/grouped evidence (default: 5)
  --cpu-minimum-promotion-samples N
                              Minimum samples admitted for installable fixed-
                              timing CPU M=1/grouped evidence (default: 30)
  --resume-cpu-partials        Reuse completed atomic CPU shape/format/ISA
                               partials in the selected output directory
  --reuse-cuda-development     Reuse and authenticate canonical CUDA M=1
                               development timing/profiler evidence, then open
                               a fresh seal and continue the full transaction
  --cuda-development-build-change-audit NOTE
                              Reviewed explanation for retaining CUDA
                              development evidence across a harness rebuild
  --stop-after-cpu-decode      Certify and install the complete CPU M=1 policy,
                               rebuild both CPU trainers, then stop before
                               grouped-verifier collection
  --resume-after-cpu-decode    Authenticate the completed and installed CPU
                               M=1 policy in output-dir, then begin directly at
                               grouped-verifier collection. Requires the same
                               --backend cpu --profile all --install workflow.
  --cpu-decode-burned-sealed-plan PATH
                              Inspected M=1 sealed plan reused as generic-only
                              development evidence; repeat per generation
  --cpu-decode-burned-sealed-paired-dir PATH
                              Complete paired CSV directory matched by position
                              to each burned M=1 plan
  --cpu-grouped-burned-sealed-plan PATH
                              Inspected grouped sealed plan reused as generic-
                              only development evidence; repeat per generation
  --cpu-grouped-burned-sealed-paired-dir PATH
                              Complete paired CSV directory matched by position
                              to each burned grouped plan
  --cpu-grouped-max-leaves N  Maximum grouped generic-tree leaf budget
                              (default: 16). A deliberate best-effort install
                              may lower this; dispatch totality, byte equality,
                              and sealed structural gates remain mandatory.
  --resume-cpu-candidate-expansion-from DIR
                              Rebase compatible completed candidate-expansion
                              shards from an older output directory onto the
                              current timing policy before collecting gaps
  --cpu-prefill-harness-build-change-audit NOTE
                              Explicit review note allowing a candidate-
                              expansion checkpoint across a timing-harness-only
                              binary rebuild; both digests and NOTE are recorded
  --skip-cpu-prefill-baseline  Reuse a complete cpu_prefill_sweep.csv and
                               collect only post-baseline development evidence;
                               requires --backend cpu-prefill, --profile all
  --stop-after-cpu-prefill-freeze
                              Publish and validate development-only policy
                              artifacts without opening the fresh v8 holdout
  --cpu-prefill-diagnostic-max-leaves N
                              Reduce the generic-tree leaf budget for a fit-only
                              non-installable CPU prefill diagnostic. Requires
                              --skip-sweep and --stop-after-cpu-prefill-freeze.
  --cpu-prefill-ablate-profiler-features
                              Validate exact profiler evidence but withhold its
                              features from the one-leaf CPU prefill diagnostic.
                              This is the timing-only half of the controlled A/B.
  --collect-cpu-prefill-sealed-after-freeze
                              Reuse an authenticated development corpus, freeze
                              it, then collect a fresh sealed-v8 holdout before
                              certification; requires cpu-prefill/all/skip-sweep
  --collect-cpu-prefill-refinement-plan PATH
                              Collect only one already-generated CPU prefill
                              generic-refinement plan, then stop
  --cpu-prefill-refinement-source-fit PATH
                              Non-installable fit diagnostic that generated the
                              existing refinement plan
  --cpu-prefill-refinement-source-observations PATH
                              Adapted observation CSV consumed by that source
                              fit, used to authenticate its corpus digest
  --cpu-prefill-refinement-source-aggregate PATH
                              Explicit immutable predecessor aggregate. Use
                              with source-timing after a split-lineage migration.
  --cpu-prefill-refinement-source-timing PATH
                              Timing sidecar for the explicit predecessor
                              aggregate
  --cpu-prefill-refinement-round N
                              One-based round number assigned to that plan
  --cpu-prefill-split-manifest PATH
                              Reviewed CPU prefill split; historical refinement
                              resumes may name the immutable split they used
  --collect-cpu-prefill-development-lineage-plan PATH
                              Collect only the additive cells declared by an
                              authenticated CPU prefill split-migration plan
  --cpu-prefill-fit-development-lineage-plan PATH
                              Fit an already collected additive lineage using
                              the stable development-lineage combined CSVs;
                              requires --skip-sweep
  --cpu-prefill-lineage-source-aggregate PATH
                              Immutable development aggregate named by lineage
  --cpu-prefill-lineage-source-timing PATH
                              Immutable timing sidecar named by lineage
  --cpu-prefill-lineage-source-split-manifest PATH
                              Historical split manifest named by lineage
  --cpu-prefill-lineage-source-route-manifest PATH
                              Historical route manifest named by lineage;
                              repeat once per ISA regime
  --cpu-prefill-lineage-source-refinement-plan PATH
                              Historical generic-refinement plan named by
                              lineage; repeat in round order
  --cpu-prefill-lineage-source-refinement-split-manifest PATH
                              Additional historical split used by one or more
                              refinement plans; repeat for every older split
  --collect-cpu-prefill-candidate-expansion-source-aggregate PATH
                              Immutable development aggregate whose exact cells
                              receive newly registered candidates only
  --collect-cpu-prefill-candidate-expansion-source-timing PATH
                              Timing sidecar paired with the immutable candidate
                              expansion source aggregate
  --cpu-prefill-candidate-expansion-candidates LIST
                              Optional explicit missing candidate IDs; defaults
                              to every registry candidate absent from the source
  --cpu-prefill-fit-candidate-expansion-plan PATH
                              Authenticated candidate-family expansion used by
                              fit-only replay; requires the four paths below
  --cpu-prefill-fit-candidate-expansion-source-aggregate PATH
                              Immutable pre-expansion development aggregate
  --cpu-prefill-fit-candidate-expansion-source-timing PATH
                              Timing sidecar paired with the immutable source
  --cpu-prefill-fit-candidate-expansion-input PATH
                              Raw anchor-plus-new-candidate expansion aggregate
  --cpu-prefill-fit-candidate-expansion-timing PATH
                              Timing sidecar paired with the expansion aggregate
  --cpu-prefill-fit-additive-aggregate PATH
                              Ordinary complete-registry evidence collected
                              after expansion; repeat in chronological order
  --cpu-prefill-fit-additive-timing PATH
                              Timing sidecar paired by position with each
                              additive aggregate; repeat in the same order
  --cpu-prefill-fit-generic-refinement-plan PATH
                              Authenticated generic-refinement plan paired
                              with each final additive aggregate; repeat in
                              chronological order
  --cpu-prefill-fit-primary-profiler-requests PATH
                              Primary matched-anchor profiler request manifest;
                              replaces the implicit output-directory profiler base
  --cpu-prefill-fit-primary-profiler-evidence PATH
                              Evidence manifest paired with the primary requests
  --cpu-prefill-fit-primary-profiler-observations PATH
                              Compact timing witnesses paired with the primary
                              profiler catalog
  --cpu-prefill-fit-additive-profiler-requests PATH
                              Additional authenticated profiler request
                              manifest; repeat once per additive catalog
  --cpu-prefill-fit-additive-profiler-evidence PATH
                              Evidence manifest paired with each additional
                              profiler request manifest
  --cpu-prefill-fit-additive-profiler-observations PATH
                              Compact timing witnesses paired with each
                              additional profiler catalog
  --cpu-prefill-fit-replay-recipe PATH
                              Authenticated one-file replacement for every CPU
                              prefill fit lineage option above. Preflight runs
                              before route probes and explicitly owns complete,
                              prefix, or rebuild checkpoint state.
  --collect-profiler-evidence  Run isolated per-candidate ncu/rocprof/perf
                               collection after canonical timing (the default
                               for --profile all)
  --skip-profiler-evidence     Do not collect profiler sidecars. This is
                               forbidden with --install.
  --reuse-profiler-evidence    Authenticate profiler request/evidence sidecars
                               already present in output-dir without launching
                               profilers. Requires --skip-sweep and is the
                               fit-only mode used by a sealed corpus bundle.
  --profiler-cpu-list LIST     Logical CPU masks for system-wide perf collection;
                               separate per-socket lane masks with semicolons
  --ncu PATH                   Nsight Compute executable
  --rocprofv3 PATH             rocprofiler-sdk v3 executable
  --perf PATH                  Linux perf executable
  --paired-max-iterations N    Maximum new development CV/refit rounds in one
                               invocation (default: 16; completed resume rounds
                               do not consume this budget; production only)
  --policy-accelerators LIST   Policy-fit devices: auto, cpu, or a comma list
                               such as cuda:0,cuda:1,rocm:0 (default: auto)
  --policy-lanes N             Independent CPU orchestration lanes per policy
                               accelerator (default: 1)
  --policy-cuda-scorer PATH    CUDA exact leaf-primary scorer DSO
  --policy-rocm-scorer PATH    ROCm exact leaf-primary scorer DSO
  --shapes LIST                Override shape list shared by CUDA and ROCm
  --shape-partition PARTITION  For --profile all, select all,
                               fast-development, fast-sealed,
                               verifier-development, or verifier-sealed
  --cuda-sweep-bin PATH        CUDA sweep binary
  --rocm-decode-bin PATH       ROCm decode trainer binary
  --cpu-avx2-sweep-bin PATH    AVX2-only CPU verifier trainer binary
  --cpu-avx512-sweep-bin PATH  AVX512 CPU verifier trainer binary; this binary
                               is run once with AVX2 and once with AVX512
                               runtime dispatch
  --skip-sweep                 Reuse existing CSVs in output-dir
  --install                    Publish a complete --profile all artifact
  --dry-run                    Print commands without running them
  -h, --help                   Show this help

Build targets:
  cmake --build build_v2_release --parallel \
    --target v2_perf_cuda_native_vnni_decode_trainer \
             v2_perf_native_vnni_throughput \
             v2_native_vnni_leaf_primary_scorer_cuda \
             v2_native_vnni_leaf_primary_scorer_rocm
  cmake --build build_v2_release --parallel \
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
cuda_measurement_lanes="${LLAMINAR_NATIVE_VNNI_CUDA_MEASUREMENT_LANES:-auto}"
rocm_measurement_lanes="${LLAMINAR_NATIVE_VNNI_ROCM_MEASUREMENT_LANES:-auto}"
cpu_threads="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_THREADS:-}"
cpu_measurement_lanes="${LLAMINAR_NATIVE_VNNI_CPU_MEASUREMENT_LANES:-auto}"
cpu_format_shards="${LLAMINAR_NATIVE_VNNI_CPU_FORMAT_SHARDS:-0}"
cpu_batch_limit="${LLAMINAR_NATIVE_VNNI_CPU_BATCH_LIMIT:-0}"
backend_collection_target_seconds="${LLAMINAR_NATIVE_VNNI_BACKEND_COLLECTION_TARGET_SECONDS:-7200}"
maximum_p95_regret_percent="${LLAMINAR_NATIVE_VNNI_PROMOTION_P95_REGRET_PERCENT:-5}"
minimum_passing_domain_percent="${LLAMINAR_NATIVE_VNNI_PROMOTION_MIN_PASSING_DOMAIN_PERCENT:-95}"
cpu_minimum_promotion_warmups="${LLAMINAR_NATIVE_VNNI_CPU_MINIMUM_PROMOTION_WARMUPS:-5}"
cpu_minimum_promotion_samples="${LLAMINAR_NATIVE_VNNI_CPU_MINIMUM_PROMOTION_SAMPLES:-30}"
resume_cpu_partials="${LLAMINAR_NATIVE_VNNI_RESUME_CPU_PARTIALS:-0}"
reuse_cuda_development="${LLAMINAR_NATIVE_VNNI_REUSE_CUDA_DEVELOPMENT:-0}"
cuda_development_build_change_audit="${LLAMINAR_NATIVE_VNNI_CUDA_DEVELOPMENT_BUILD_CHANGE_AUDIT:-}"
stop_after_cpu_decode="${LLAMINAR_NATIVE_VNNI_STOP_AFTER_CPU_DECODE:-0}"
resume_after_cpu_decode="${LLAMINAR_NATIVE_VNNI_RESUME_AFTER_CPU_DECODE:-0}"
skip_cpu_prefill_baseline="${LLAMINAR_NATIVE_VNNI_SKIP_CPU_PREFILL_BASELINE:-0}"
stop_after_cpu_prefill_freeze="${LLAMINAR_NATIVE_VNNI_STOP_AFTER_CPU_PREFILL_FREEZE:-0}"
collect_cpu_prefill_sealed_after_freeze="${LLAMINAR_NATIVE_VNNI_COLLECT_CPU_PREFILL_SEALED_AFTER_FREEZE:-0}"
collect_cpu_prefill_refinement_plan=""
cpu_prefill_refinement_source_fit=""
cpu_prefill_refinement_source_observations=""
cpu_prefill_refinement_source_aggregate=""
cpu_prefill_refinement_source_timing=""
cpu_prefill_refinement_round=""
cpu_prefill_split_manifest_path="${LLAMINAR_NATIVE_VNNI_CPU_PREFILL_SPLIT_MANIFEST:-}"
collect_cpu_prefill_development_lineage_plan=""
cpu_prefill_fit_development_lineage_plan=""
cpu_prefill_lineage_source_aggregate=""
cpu_prefill_lineage_source_timing=""
cpu_prefill_lineage_source_split_manifest=""
cpu_prefill_lineage_source_route_manifests=()
cpu_prefill_lineage_source_refinement_plans=()
cpu_prefill_lineage_source_refinement_split_manifests=()
cpu_prefill_candidate_expansion_source_aggregate=""
cpu_prefill_candidate_expansion_source_timing=""
cpu_prefill_candidate_expansion_candidates=""
cpu_prefill_candidate_expansion_resume_from=""
cpu_prefill_harness_build_change_audit=""
cpu_prefill_fit_candidate_expansion_plan=""
cpu_prefill_fit_candidate_expansion_source_aggregate=""
cpu_prefill_fit_candidate_expansion_source_timing=""
cpu_prefill_fit_candidate_expansion_input=""
cpu_prefill_fit_candidate_expansion_timing=""
cpu_prefill_fit_additive_aggregates=()
cpu_prefill_fit_additive_timings=()
cpu_prefill_fit_generic_refinement_plans=()
cpu_prefill_fit_primary_profiler_requests=""
cpu_prefill_fit_primary_profiler_evidence=""
cpu_prefill_fit_primary_profiler_observations=""
cpu_prefill_fit_additive_profiler_requests=()
cpu_prefill_fit_additive_profiler_evidence=()
cpu_prefill_fit_additive_profiler_observations=()
cpu_prefill_fit_burned_seal_manifests=()
cpu_decode_burned_sealed_plans=()
cpu_decode_burned_sealed_paired_dirs=()
cpu_grouped_burned_sealed_plans=()
cpu_grouped_burned_sealed_paired_dirs=()
cpu_grouped_max_leaves="${LLAMINAR_NATIVE_VNNI_CPU_GROUPED_MAX_LEAVES:-16}"
cpu_prefill_fit_replay_recipe=""
cpu_prefill_diagnostic_max_leaves=""
cpu_prefill_ablate_profiler_features=0
cpu_prefill_replay_recipe_digest=""
cpu_prefill_replay_checkpoint_action=""
cpu_prefill_replay_checkpoint_path=""
cpu_prefill_replay_checkpoint_prefix_count=""
cpu_prefill_replay_context_corpus_id=""
cpu_prefill_fit_target_route_manifests=()
cpu_prefill_fit_sealed_candidate_route_manifest=""
collect_profiler_evidence=-1
reuse_profiler_evidence=0
profiler_cpu_list="${LLAMINAR_NATIVE_VNNI_PROFILER_CPU_LIST:-}"
ncu_path="${LLAMINAR_NATIVE_VNNI_NCU:-}"
rocprofv3_path="${LLAMINAR_NATIVE_VNNI_ROCPROFV3:-}"
perf_path="${LLAMINAR_NATIVE_VNNI_PERF:-}"
paired_max_iterations="${LLAMINAR_NATIVE_VNNI_PAIRED_MAX_ITERATIONS:-16}"
policy_accelerators="${LLAMINAR_NATIVE_VNNI_POLICY_ACCELERATORS:-auto}"
policy_lanes="${LLAMINAR_NATIVE_VNNI_POLICY_LANES_PER_ACCELERATOR:-1}"
policy_cuda_scorer="${LLAMINAR_NATIVE_VNNI_POLICY_CUDA_SCORER_LIBRARY:-${repo_root}/build_v2_release/tests/v2/libllaminar_native_vnni_leaf_primary_scorer_cuda.so}"
policy_rocm_scorer="${LLAMINAR_NATIVE_VNNI_POLICY_ROCM_SCORER_LIBRARY:-${repo_root}/build_v2_release/tests/v2/libllaminar_native_vnni_leaf_primary_scorer_rocm.so}"
shapes=""
shape_partition="all"
shapes_explicit=0
cuda_sweep_bin="${repo_root}/build_v2_release/tests/v2/v2_perf_cuda_native_vnni_decode_trainer"
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
    --cuda-measurement-lanes)
      cuda_measurement_lanes="${2:-}"
      shift 2
      ;;
    --rocm-measurement-lanes)
      rocm_measurement_lanes="${2:-}"
      shift 2
      ;;
    --cpu-threads)
      cpu_threads="${2:-}"
      shift 2
      ;;
    --cpu-measurement-lanes)
      cpu_measurement_lanes="${2:-}"
      shift 2
      ;;
    --cpu-format-shards)
      cpu_format_shards=1
      shift
      ;;
    --cpu-batch-limit)
      cpu_batch_limit="${2:-}"
      shift 2
      ;;
    --backend-collection-target-seconds)
      backend_collection_target_seconds="${2:-}"
      shift 2
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
    --resume-cpu-partials)
      resume_cpu_partials=1
      shift
      ;;
    --reuse-cuda-development)
      reuse_cuda_development=1
      shift
      ;;
    --cuda-development-build-change-audit)
      cuda_development_build_change_audit="${2:-}"
      shift 2
      ;;
    --stop-after-cpu-decode)
      stop_after_cpu_decode=1
      shift
      ;;
    --resume-after-cpu-decode)
      resume_after_cpu_decode=1
      shift
      ;;
    --cpu-decode-burned-sealed-plan)
      cpu_decode_burned_sealed_plans+=("${2:-}")
      shift 2
      ;;
    --cpu-decode-burned-sealed-paired-dir)
      cpu_decode_burned_sealed_paired_dirs+=("${2:-}")
      shift 2
      ;;
    --cpu-grouped-burned-sealed-plan)
      cpu_grouped_burned_sealed_plans+=("${2:-}")
      shift 2
      ;;
    --cpu-grouped-burned-sealed-paired-dir)
      cpu_grouped_burned_sealed_paired_dirs+=("${2:-}")
      shift 2
      ;;
    --cpu-grouped-max-leaves)
      cpu_grouped_max_leaves="${2:-}"
      shift 2
      ;;
    --skip-cpu-prefill-baseline)
      skip_cpu_prefill_baseline=1
      shift
      ;;
    --stop-after-cpu-prefill-freeze)
      stop_after_cpu_prefill_freeze=1
      shift
      ;;
    --cpu-prefill-diagnostic-max-leaves)
      cpu_prefill_diagnostic_max_leaves="${2:-}"
      shift 2
      ;;
    --cpu-prefill-ablate-profiler-features)
      cpu_prefill_ablate_profiler_features=1
      shift
      ;;
    --collect-cpu-prefill-sealed-after-freeze)
      collect_cpu_prefill_sealed_after_freeze=1
      shift
      ;;
    --collect-cpu-prefill-refinement-plan)
      collect_cpu_prefill_refinement_plan="${2:-}"
      shift 2
      ;;
    --cpu-prefill-refinement-source-fit)
      cpu_prefill_refinement_source_fit="${2:-}"
      shift 2
      ;;
    --cpu-prefill-refinement-source-observations)
      cpu_prefill_refinement_source_observations="${2:-}"
      shift 2
      ;;
    --cpu-prefill-refinement-source-aggregate)
      cpu_prefill_refinement_source_aggregate="${2:-}"
      shift 2
      ;;
    --cpu-prefill-refinement-source-timing)
      cpu_prefill_refinement_source_timing="${2:-}"
      shift 2
      ;;
    --cpu-prefill-refinement-round)
      cpu_prefill_refinement_round="${2:-}"
      shift 2
      ;;
    --cpu-prefill-split-manifest)
      cpu_prefill_split_manifest_path="${2:-}"
      shift 2
      ;;
    --collect-cpu-prefill-development-lineage-plan)
      collect_cpu_prefill_development_lineage_plan="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-development-lineage-plan)
      cpu_prefill_fit_development_lineage_plan="${2:-}"
      shift 2
      ;;
    --cpu-prefill-lineage-source-aggregate)
      cpu_prefill_lineage_source_aggregate="${2:-}"
      shift 2
      ;;
    --cpu-prefill-lineage-source-timing)
      cpu_prefill_lineage_source_timing="${2:-}"
      shift 2
      ;;
    --cpu-prefill-lineage-source-split-manifest)
      cpu_prefill_lineage_source_split_manifest="${2:-}"
      shift 2
      ;;
    --cpu-prefill-lineage-source-route-manifest)
      cpu_prefill_lineage_source_route_manifests+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-lineage-source-refinement-plan)
      cpu_prefill_lineage_source_refinement_plans+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-lineage-source-refinement-split-manifest)
      cpu_prefill_lineage_source_refinement_split_manifests+=("${2:-}")
      shift 2
      ;;
    --collect-cpu-prefill-candidate-expansion-source-aggregate)
      cpu_prefill_candidate_expansion_source_aggregate="${2:-}"
      shift 2
      ;;
    --collect-cpu-prefill-candidate-expansion-source-timing)
      cpu_prefill_candidate_expansion_source_timing="${2:-}"
      shift 2
      ;;
    --cpu-prefill-candidate-expansion-candidates)
      cpu_prefill_candidate_expansion_candidates="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-candidate-expansion-plan)
      cpu_prefill_fit_candidate_expansion_plan="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-candidate-expansion-source-aggregate)
      cpu_prefill_fit_candidate_expansion_source_aggregate="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-candidate-expansion-source-timing)
      cpu_prefill_fit_candidate_expansion_source_timing="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-candidate-expansion-input)
      cpu_prefill_fit_candidate_expansion_input="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-candidate-expansion-timing)
      cpu_prefill_fit_candidate_expansion_timing="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-additive-aggregate)
      cpu_prefill_fit_additive_aggregates+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-additive-timing)
      cpu_prefill_fit_additive_timings+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-generic-refinement-plan)
      cpu_prefill_fit_generic_refinement_plans+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-primary-profiler-requests)
      cpu_prefill_fit_primary_profiler_requests="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-primary-profiler-evidence)
      cpu_prefill_fit_primary_profiler_evidence="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-primary-profiler-observations)
      cpu_prefill_fit_primary_profiler_observations="${2:-}"
      shift 2
      ;;
    --cpu-prefill-fit-additive-profiler-requests)
      cpu_prefill_fit_additive_profiler_requests+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-additive-profiler-evidence)
      cpu_prefill_fit_additive_profiler_evidence+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-additive-profiler-observations)
      cpu_prefill_fit_additive_profiler_observations+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-burned-seal-manifest)
      cpu_prefill_fit_burned_seal_manifests+=("${2:-}")
      shift 2
      ;;
    --cpu-prefill-fit-replay-recipe)
      cpu_prefill_fit_replay_recipe="${2:-}"
      shift 2
      ;;
    --resume-cpu-candidate-expansion-from)
      cpu_prefill_candidate_expansion_resume_from="${2:-}"
      shift 2
      ;;
    --cpu-prefill-harness-build-change-audit)
      cpu_prefill_harness_build_change_audit="${2:-}"
      shift 2
      ;;
    --collect-profiler-evidence)
      collect_profiler_evidence=1
      shift
      ;;
    --skip-profiler-evidence)
      collect_profiler_evidence=0
      shift
      ;;
    --reuse-profiler-evidence)
      reuse_profiler_evidence=1
      shift
      ;;
    --profiler-cpu-list)
      profiler_cpu_list="${2:-}"
      shift 2
      ;;
    --ncu)
      ncu_path="${2:-}"
      shift 2
      ;;
    --rocprofv3)
      rocprofv3_path="${2:-}"
      shift 2
      ;;
    --perf)
      perf_path="${2:-}"
      shift 2
      ;;
    --paired-max-iterations)
      paired_max_iterations="${2:-}"
      shift 2
      ;;
    --policy-accelerators)
      policy_accelerators="${2:-}"
      shift 2
      ;;
    --policy-lanes)
      policy_lanes="${2:-}"
      shift 2
      ;;
    --policy-cuda-scorer)
      policy_cuda_scorer="${2:-}"
      shift 2
      ;;
    --policy-rocm-scorer)
      policy_rocm_scorer="${2:-}"
      shift 2
      ;;
    --shapes)
      shapes="${2:-}"
      shapes_explicit=1
      shift 2
      ;;
    --shape-partition)
      shape_partition="${2:-}"
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

if (( ${#cpu_decode_burned_sealed_plans[@]} !=
      ${#cpu_decode_burned_sealed_paired_dirs[@]} )); then
  echo "error: burned CPU decode plans and paired directories must pair by position" >&2
  exit 2
fi
if (( ${#cpu_grouped_burned_sealed_plans[@]} !=
      ${#cpu_grouped_burned_sealed_paired_dirs[@]} )); then
  echo "error: burned CPU grouped plans and paired directories must pair by position" >&2
  exit 2
fi

validate_promotion_percent() {
  local option_name="$1"
  local value="$2"
  local allow_zero="$3"
  if [[ ! "${value}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] ||
     ! awk -v value="${value}" -v allow_zero="${allow_zero}" 'BEGIN {
       lower_ok = allow_zero ? value >= 0.0 : value > 0.0
       exit !(lower_ok && value <= 100.0)
     }'; then
    local interval="(0, 100]"
    [[ "${allow_zero}" == "1" ]] && interval="[0, 100]"
    echo "error: ${option_name} must be a decimal percentage in ${interval}" >&2
    exit 2
  fi
}

validate_promotion_percent \
  --maximum-p95-regret-percent "${maximum_p95_regret_percent}" 0
validate_promotion_percent \
  --minimum-passing-domain-percent "${minimum_passing_domain_percent}" 1
cpu_grouped_paired_max_regret_fraction="$(awk \
  -v regret_percent="${maximum_p95_regret_percent}" \
  'BEGIN { printf "%.12g\n", regret_percent / 100.0 }')"
if [[ ! "${cpu_minimum_promotion_warmups}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cpu-minimum-promotion-warmups must be a positive integer" >&2
  exit 2
fi
if [[ ! "${cpu_minimum_promotion_samples}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cpu-minimum-promotion-samples must be a positive integer" >&2
  exit 2
fi
export LLAMINAR_NATIVE_VNNI_PROMOTION_P95_REGRET_PERCENT="${maximum_p95_regret_percent}"
export LLAMINAR_NATIVE_VNNI_PROMOTION_MIN_PASSING_DOMAIN_PERCENT="${minimum_passing_domain_percent}"
printf 'NativeVNNI promotion criteria: p95-regret<%s%% passing-domains>=%s%%\n' \
  "${maximum_p95_regret_percent}" "${minimum_passing_domain_percent}"

# A replay recipe is the production fit-only interface for an evolved CPU
# prefill corpus. Expand typed records without eval so paths cannot become
# shell syntax. The Python preflight authenticates every payload plus all
# embedded plan/source relationships before this script reaches route probes,
# policy scorers, timing binaries, or profiler tools.
if [[ -n "${cpu_prefill_fit_replay_recipe}" ]]; then
  if [[ -n "${cpu_prefill_fit_candidate_expansion_plan}" ||
        -n "${cpu_prefill_fit_candidate_expansion_source_aggregate}" ||
        -n "${cpu_prefill_fit_candidate_expansion_source_timing}" ||
        -n "${cpu_prefill_fit_candidate_expansion_input}" ||
        -n "${cpu_prefill_fit_candidate_expansion_timing}" ||
        -n "${cpu_prefill_fit_development_lineage_plan}" ||
        -n "${cpu_prefill_lineage_source_aggregate}" ||
        -n "${cpu_prefill_lineage_source_timing}" ||
        -n "${cpu_prefill_lineage_source_split_manifest}" ||
        -n "${cpu_prefill_split_manifest_path}" ||
        ${#cpu_prefill_lineage_source_route_manifests[@]} -gt 0 ||
        ${#cpu_prefill_lineage_source_refinement_plans[@]} -gt 0 ||
        ${#cpu_prefill_lineage_source_refinement_split_manifests[@]} -gt 0 ||
        ${#cpu_prefill_fit_additive_aggregates[@]} -gt 0 ||
        ${#cpu_prefill_fit_additive_timings[@]} -gt 0 ||
        ${#cpu_prefill_fit_generic_refinement_plans[@]} -gt 0 ||
        -n "${cpu_prefill_fit_primary_profiler_requests}" ||
        -n "${cpu_prefill_fit_primary_profiler_evidence}" ||
        -n "${cpu_prefill_fit_primary_profiler_observations}" ||
        ${#cpu_prefill_fit_additive_profiler_requests[@]} -gt 0 ||
        ${#cpu_prefill_fit_additive_profiler_evidence[@]} -gt 0 ||
        ${#cpu_prefill_fit_additive_profiler_observations[@]} -gt 0 ||
        ${#cpu_prefill_fit_burned_seal_manifests[@]} -gt 0 ]]; then
    echo "error: --cpu-prefill-fit-replay-recipe replaces every manual CPU prefill fit lineage option" >&2
    exit 2
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]] ||
     (( ! skip_sweep )); then
    echo "error: CPU prefill replay recipe requires --backend cpu-prefill --profile all --skip-sweep" >&2
    exit 2
  fi

  replay_records_file="$(mktemp)"
  trap 'rm -f "${replay_records_file:-}"' EXIT
  if ! PYTHONPATH="${repo_root}/tests/v2/performance/kernels" \
      python3 -m native_vnni_dispatch.cpu_prefill_replay_recipe \
        preflight \
        --recipe "${cpu_prefill_fit_replay_recipe}" \
        --emit-shell-records > "${replay_records_file}"; then
    echo "error: CPU prefill replay recipe preflight failed before expensive work" >&2
    exit 2
  fi
  while IFS=$'\t' read -r replay_kind replay_value; do
    case "${replay_kind}" in
      recipe_digest)
        cpu_prefill_replay_recipe_digest="${replay_value}"
        ;;
      checkpoint_action)
        cpu_prefill_replay_checkpoint_action="${replay_value}"
        ;;
      checkpoint_path)
        cpu_prefill_replay_checkpoint_path="${replay_value}"
        ;;
      context_corpus_id)
        cpu_prefill_replay_context_corpus_id="${replay_value}"
        ;;
      checkpoint_prefix_input_count)
        cpu_prefill_replay_checkpoint_prefix_count="${replay_value}"
        ;;
      target_split_manifest)
        cpu_prefill_split_manifest_path="${replay_value}"
        ;;
      target_route_manifest)
        cpu_prefill_fit_target_route_manifests+=("${replay_value}")
        ;;
      sealed_candidate_route_manifest)
        cpu_prefill_fit_sealed_candidate_route_manifest="${replay_value}"
        ;;
      candidate_plan)
        cpu_prefill_fit_candidate_expansion_plan="${replay_value}"
        ;;
      candidate_source_aggregate)
        cpu_prefill_fit_candidate_expansion_source_aggregate="${replay_value}"
        ;;
      candidate_source_timing)
        cpu_prefill_fit_candidate_expansion_source_timing="${replay_value}"
        ;;
      candidate_expansion_aggregate)
        cpu_prefill_fit_candidate_expansion_input="${replay_value}"
        ;;
      candidate_expansion_timing)
        cpu_prefill_fit_candidate_expansion_timing="${replay_value}"
        ;;
      additive_aggregate)
        cpu_prefill_fit_additive_aggregates+=("${replay_value}")
        ;;
      additive_timing)
        cpu_prefill_fit_additive_timings+=("${replay_value}")
        ;;
      current_generic_refinement_plan)
        cpu_prefill_fit_generic_refinement_plans+=("${replay_value}")
        ;;
      primary_profiler_requests)
        cpu_prefill_fit_primary_profiler_requests="${replay_value}"
        ;;
      primary_profiler_evidence)
        cpu_prefill_fit_primary_profiler_evidence="${replay_value}"
        ;;
      primary_profiler_observations)
        cpu_prefill_fit_primary_profiler_observations="${replay_value}"
        ;;
      profiler_requests)
        cpu_prefill_fit_additive_profiler_requests+=("${replay_value}")
        ;;
      profiler_evidence)
        cpu_prefill_fit_additive_profiler_evidence+=("${replay_value}")
        ;;
      profiler_observations)
        cpu_prefill_fit_additive_profiler_observations+=("${replay_value}")
        ;;
      burned_sealed_development_manifest)
        cpu_prefill_fit_burned_seal_manifests+=("${replay_value}")
        ;;
      lineage_plan)
        cpu_prefill_fit_development_lineage_plan="${replay_value}"
        ;;
      lineage_source_aggregate)
        cpu_prefill_lineage_source_aggregate="${replay_value}"
        ;;
      lineage_source_timing)
        cpu_prefill_lineage_source_timing="${replay_value}"
        ;;
      lineage_source_split_manifest)
        cpu_prefill_lineage_source_split_manifest="${replay_value}"
        ;;
      lineage_source_route_manifest)
        cpu_prefill_lineage_source_route_manifests+=("${replay_value}")
        ;;
      lineage_historical_refinement_plan)
        cpu_prefill_lineage_source_refinement_plans+=("${replay_value}")
        ;;
      lineage_additional_split_manifest)
        cpu_prefill_lineage_source_refinement_split_manifests+=("${replay_value}")
        ;;
      *)
        echo "error: CPU prefill replay recipe emitted unknown record ${replay_kind}" >&2
        exit 2
        ;;
    esac
  done < "${replay_records_file}"
  rm -f "${replay_records_file}"
  replay_records_file=""
  printf 'Authenticated CPU prefill replay recipe: digest=%s checkpoint_action=%s\n' \
    "${cpu_prefill_replay_recipe_digest}" \
    "${cpu_prefill_replay_checkpoint_action}"
fi

case "${backend}" in
  cuda|rocm|cpu|cpu-prefill|both|all) ;;
  *)
    echo "error: --backend must be cuda, rocm, cpu, cpu-prefill, both, or all" >&2
    exit 2
    ;;
esac
if [[ "${backend}" == "cpu-prefill" ]] &&
   [[ "${profile}" != "quick" && "${profile}" != "family-smoke" &&
      "${profile}" != "all" ]]; then
  echo "error: --backend cpu-prefill supports quick, family-smoke, or all" >&2
  exit 2
fi
if (( stop_after_cpu_decode )) &&
   [[ "${backend}" != "cpu" || "${profile}" != "all" ]]; then
  echo "error: --stop-after-cpu-decode requires --backend cpu --profile all" >&2
  exit 2
fi
if (( stop_after_cpu_decode && ! install )); then
  echo "error: --stop-after-cpu-decode requires --install" >&2
  exit 2
fi
if (( resume_after_cpu_decode )) &&
   [[ "${backend}" != "cpu" || "${profile}" != "all" ]]; then
  echo "error: --resume-after-cpu-decode requires --backend cpu --profile all" >&2
  exit 2
fi
if (( resume_after_cpu_decode && ! install )); then
  echo "error: --resume-after-cpu-decode requires --install" >&2
  exit 2
fi
if (( stop_after_cpu_decode && resume_after_cpu_decode )); then
  echo "error: --stop-after-cpu-decode and --resume-after-cpu-decode are mutually exclusive" >&2
  exit 2
fi
if (( resume_after_cpu_decode )) && [[ "${shape_partition}" != "all" ]]; then
  echo "error: --resume-after-cpu-decode cannot use --shape-partition" >&2
  exit 2
fi
if (( reuse_cuda_development )) &&
   [[ "${backend}" != "cuda" || "${profile}" != "all" ]]; then
  echo "error: --reuse-cuda-development requires --backend cuda --profile all" >&2
  exit 2
fi
if (( reuse_cuda_development && skip_sweep )); then
  echo "error: --reuse-cuda-development and --skip-sweep are mutually exclusive" >&2
  exit 2
fi
if (( reuse_cuda_development )) && [[ "${shape_partition}" != "all" ]]; then
  echo "error: --reuse-cuda-development cannot use --shape-partition" >&2
  exit 2
fi
if (( reuse_cuda_development )) &&
   [[ -z "${cuda_development_build_change_audit//[[:space:]]/}" ]]; then
  echo "error: --reuse-cuda-development requires a non-empty --cuda-development-build-change-audit" >&2
  exit 2
fi
if (( ! reuse_cuda_development )) &&
   [[ -n "${cuda_development_build_change_audit}" ]]; then
  echo "error: --cuda-development-build-change-audit requires " \
       "--reuse-cuda-development" >&2
  exit 2
fi
if [[ ! "${cpu_grouped_max_leaves}" =~ ^[1-9][0-9]*$ ]] ||
   (( cpu_grouped_max_leaves > 32 )); then
  echo "error: --cpu-grouped-max-leaves must be in [1, 32]" >&2
  exit 2
fi
if (( skip_cpu_prefill_baseline )) &&
   [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]]; then
  echo "error: --skip-cpu-prefill-baseline requires --backend cpu-prefill --profile all" >&2
  exit 2
fi
if (( skip_cpu_prefill_baseline && skip_sweep )); then
  echo "error: --skip-cpu-prefill-baseline and --skip-sweep are mutually exclusive" >&2
  exit 2
fi
if (( stop_after_cpu_prefill_freeze )) &&
   [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]]; then
  echo "error: --stop-after-cpu-prefill-freeze requires --backend cpu-prefill --profile all" >&2
  exit 2
fi
if (( stop_after_cpu_prefill_freeze && install )); then
  echo "error: --stop-after-cpu-prefill-freeze cannot install an uncertified policy" >&2
  exit 2
fi
if [[ -n "${cpu_prefill_diagnostic_max_leaves}" ]]; then
  if [[ ! "${cpu_prefill_diagnostic_max_leaves}" =~ ^[1-9][0-9]*$ ]] ||
     (( cpu_prefill_diagnostic_max_leaves > 32 )); then
    echo "error: --cpu-prefill-diagnostic-max-leaves must be in [1, 32]" >&2
    exit 2
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]] ||
     (( ! skip_sweep || ! stop_after_cpu_prefill_freeze || install )); then
    echo "error: --cpu-prefill-diagnostic-max-leaves is restricted to non-installable CPU prefill fit-only freeze diagnostics" >&2
    exit 2
  fi
fi
if (( cpu_prefill_ablate_profiler_features )) &&
   [[ -z "${cpu_prefill_diagnostic_max_leaves}" ]]; then
  echo "error: --cpu-prefill-ablate-profiler-features requires --cpu-prefill-diagnostic-max-leaves" >&2
  exit 2
fi
if (( collect_cpu_prefill_sealed_after_freeze )) &&
   [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]] ; then
  echo "error: --collect-cpu-prefill-sealed-after-freeze requires --backend cpu-prefill --profile all" >&2
  exit 2
fi
if (( collect_cpu_prefill_sealed_after_freeze && ! skip_sweep )); then
  echo "error: --collect-cpu-prefill-sealed-after-freeze requires --skip-sweep" >&2
  exit 2
fi
if (( collect_cpu_prefill_sealed_after_freeze &&
      stop_after_cpu_prefill_freeze )); then
  echo "error: sealed collection and --stop-after-cpu-prefill-freeze are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${collect_cpu_prefill_refinement_plan}" ||
      -n "${cpu_prefill_refinement_source_fit}" ||
      -n "${cpu_prefill_refinement_source_observations}" ||
      -n "${cpu_prefill_refinement_source_aggregate}" ||
      -n "${cpu_prefill_refinement_source_timing}" ||
      -n "${cpu_prefill_refinement_round}" ]]; then
  if [[ -z "${collect_cpu_prefill_refinement_plan}" ||
        -z "${cpu_prefill_refinement_source_fit}" ||
        -z "${cpu_prefill_refinement_source_observations}" ||
        -z "${cpu_prefill_refinement_round}" ]]; then
    echo "error: existing CPU prefill refinement collection requires its plan, source fit, source observations, and round" >&2
    exit 2
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]]; then
    echo "error: existing CPU prefill refinement collection requires --backend cpu-prefill --profile all" >&2
    exit 2
  fi
  if [[ -n "${cpu_prefill_refinement_source_aggregate}" ||
        -n "${cpu_prefill_refinement_source_timing}" ]]; then
    if [[ -z "${cpu_prefill_refinement_source_aggregate}" ||
          -z "${cpu_prefill_refinement_source_timing}" ]]; then
      echo "error: explicit CPU prefill refinement predecessor requires both source aggregate and timing sidecar" >&2
      exit 2
    fi
  fi
  if [[ ! "${cpu_prefill_refinement_round}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --cpu-prefill-refinement-round must be a positive integer" >&2
    exit 2
  fi
  if (( install || skip_sweep || stop_after_cpu_prefill_freeze )); then
    echo "error: existing CPU prefill refinement collection cannot fit, freeze, install, or skip measurement" >&2
    exit 2
  fi
fi
if [[ -n "${collect_cpu_prefill_development_lineage_plan}" ||
      -n "${cpu_prefill_fit_development_lineage_plan}" ||
      -n "${cpu_prefill_lineage_source_aggregate}" ||
      -n "${cpu_prefill_lineage_source_timing}" ||
      -n "${cpu_prefill_lineage_source_split_manifest}" ||
      ${#cpu_prefill_lineage_source_route_manifests[@]} -gt 0 ||
      ${#cpu_prefill_lineage_source_refinement_plans[@]} -gt 0 ||
      ${#cpu_prefill_lineage_source_refinement_split_manifests[@]} -gt 0 ]]; then
  if [[ -n "${collect_cpu_prefill_development_lineage_plan}" &&
        -n "${cpu_prefill_fit_development_lineage_plan}" ]]; then
    echo "error: CPU prefill lineage collection and fit plans are mutually exclusive" >&2
    exit 2
  fi
  if [[ -z "${collect_cpu_prefill_development_lineage_plan}" &&
        -z "${cpu_prefill_fit_development_lineage_plan}" ]] ||
     [[ -z "${cpu_prefill_lineage_source_aggregate}" ||
        -z "${cpu_prefill_lineage_source_timing}" ||
        -z "${cpu_prefill_lineage_source_split_manifest}" ||
        ${#cpu_prefill_lineage_source_route_manifests[@]} -eq 0 ]]; then
    echo "error: CPU prefill development lineage requires its plan, source aggregate, timing sidecar, source split manifest, and source route manifests" >&2
    exit 2
  fi
  if [[ -n "${collect_cpu_prefill_refinement_plan}" ]]; then
    echo "error: CPU prefill development lineage and generic refinement collection are mutually exclusive" >&2
    exit 2
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]]; then
    echo "error: CPU prefill development lineage requires --backend cpu-prefill --profile all" >&2
    exit 2
  fi
  if [[ -n "${collect_cpu_prefill_development_lineage_plan}" ]]; then
    if (( install || skip_sweep || stop_after_cpu_prefill_freeze )); then
      echo "error: CPU prefill development lineage collection cannot fit, freeze, install, or skip measurement" >&2
      exit 2
    fi
  elif (( ! skip_sweep )); then
    echo "error: CPU prefill development lineage fitting requires --skip-sweep" >&2
    exit 2
  fi
fi
if [[ -n "${cpu_prefill_candidate_expansion_source_aggregate}" ||
      -n "${cpu_prefill_candidate_expansion_source_timing}" ||
      -n "${cpu_prefill_candidate_expansion_candidates}" ||
      -n "${cpu_prefill_candidate_expansion_resume_from}" ||
      -n "${cpu_prefill_harness_build_change_audit}" ]]; then
  if [[ -z "${cpu_prefill_candidate_expansion_source_aggregate}" ||
        -z "${cpu_prefill_candidate_expansion_source_timing}" ]]; then
    echo "error: CPU prefill candidate expansion requires both source aggregate and timing sidecar" >&2
    exit 2
  fi
  if [[ -n "${collect_cpu_prefill_refinement_plan}" ||
        -n "${collect_cpu_prefill_development_lineage_plan}" ||
        -n "${cpu_prefill_fit_development_lineage_plan}" ]]; then
    echo "error: CPU prefill candidate expansion, lineage, and generic refinement collection are mutually exclusive" >&2
    exit 2
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]]; then
    echo "error: CPU prefill candidate expansion requires --backend cpu-prefill --profile all" >&2
    exit 2
  fi
  if (( install || skip_sweep || stop_after_cpu_prefill_freeze )); then
    echo "error: CPU prefill candidate expansion cannot fit, freeze, install, or skip measurement" >&2
    exit 2
  fi
  if [[ -n "${cpu_prefill_candidate_expansion_resume_from}" ]] &&
     (( ! resume_cpu_partials )); then
    echo "error: --resume-cpu-candidate-expansion-from requires --resume-cpu-partials" >&2
    exit 2
  fi
  if [[ -n "${cpu_prefill_harness_build_change_audit}" &&
        -z "${cpu_prefill_candidate_expansion_resume_from}" ]]; then
    echo "error: --cpu-prefill-harness-build-change-audit requires --resume-cpu-candidate-expansion-from" >&2
    exit 2
  fi
fi
if [[ -n "${cpu_prefill_fit_candidate_expansion_plan}" ||
      -n "${cpu_prefill_fit_candidate_expansion_source_aggregate}" ||
      -n "${cpu_prefill_fit_candidate_expansion_source_timing}" ||
      -n "${cpu_prefill_fit_candidate_expansion_input}" ||
      -n "${cpu_prefill_fit_candidate_expansion_timing}" ||
      ${#cpu_prefill_fit_additive_aggregates[@]} -gt 0 ||
      ${#cpu_prefill_fit_additive_timings[@]} -gt 0 ||
      ${#cpu_prefill_fit_generic_refinement_plans[@]} -gt 0 ||
      -n "${cpu_prefill_fit_primary_profiler_requests}" ||
      -n "${cpu_prefill_fit_primary_profiler_evidence}" ||
      -n "${cpu_prefill_fit_primary_profiler_observations}" ||
      ${#cpu_prefill_fit_additive_profiler_requests[@]} -gt 0 ||
      ${#cpu_prefill_fit_additive_profiler_evidence[@]} -gt 0 ||
      ${#cpu_prefill_fit_additive_profiler_observations[@]} -gt 0 ]]; then
  if [[ -z "${cpu_prefill_fit_candidate_expansion_plan}" ||
        -z "${cpu_prefill_fit_candidate_expansion_source_aggregate}" ||
        -z "${cpu_prefill_fit_candidate_expansion_source_timing}" ||
        -z "${cpu_prefill_fit_candidate_expansion_input}" ||
        -z "${cpu_prefill_fit_candidate_expansion_timing}" ]]; then
    echo "error: CPU prefill candidate-expansion fitting requires its plan, immutable source pair, and expansion pair" >&2
    exit 2
  fi
  if (( ${#cpu_prefill_fit_additive_aggregates[@]} !=
        ${#cpu_prefill_fit_additive_timings[@]} )); then
    echo "error: CPU prefill fit additive aggregates and timing sidecars must be paired" >&2
    exit 2
  fi
  if (( ${#cpu_prefill_fit_generic_refinement_plans[@]} >
        ${#cpu_prefill_fit_additive_aggregates[@]} )); then
    echo "error: CPU prefill generic refinement plans require one final additive aggregate each" >&2
    exit 2
  fi
  if (( ${#cpu_prefill_fit_additive_profiler_requests[@]} !=
        ${#cpu_prefill_fit_additive_profiler_evidence[@]} ||
        ${#cpu_prefill_fit_additive_profiler_requests[@]} !=
        ${#cpu_prefill_fit_additive_profiler_observations[@]} )); then
    echo "error: CPU prefill additive profiler requests, evidence, and observations must be paired" >&2
    exit 2
  fi
  if [[ -n "${cpu_prefill_fit_primary_profiler_requests}" ||
        -n "${cpu_prefill_fit_primary_profiler_evidence}" ||
        -n "${cpu_prefill_fit_primary_profiler_observations}" ]]; then
    if [[ -z "${cpu_prefill_fit_primary_profiler_requests}" ||
          -z "${cpu_prefill_fit_primary_profiler_evidence}" ||
          -z "${cpu_prefill_fit_primary_profiler_observations}" ]]; then
      echo "error: CPU prefill primary profiler requests, evidence, and observations must form one complete transaction" >&2
      exit 2
    fi
    if (( ! reuse_profiler_evidence )); then
      echo "error: CPU prefill primary profiler replacement requires --reuse-profiler-evidence" >&2
      exit 2
    fi
  fi
  if [[ "${backend}" != "cpu-prefill" || "${profile}" != "all" ]] ||
     (( ! skip_sweep )); then
    echo "error: CPU prefill candidate-expansion fitting requires --backend cpu-prefill --profile all --skip-sweep" >&2
    exit 2
  fi
  if [[ -n "${collect_cpu_prefill_refinement_plan}" ||
        -n "${collect_cpu_prefill_development_lineage_plan}" ||
        -n "${cpu_prefill_candidate_expansion_source_aggregate}" ]]; then
    echo "error: CPU prefill candidate-expansion fitting cannot collect measurement evidence" >&2
    exit 2
  fi
fi

case "${profile}" in
  quick|family-smoke|qwen36-core|qwen36-lm-head|qwen36-moe|qwen36|all) ;;
  *)
    echo "error: --profile must be quick, family-smoke, qwen36-core, qwen36-lm-head, qwen36-moe, qwen36, or all" >&2
    exit 2
    ;;
esac

case "${shape_partition}" in
  all|fast-development|fast-sealed|verifier-development|verifier-sealed) ;;
  *)
    echo "error: --shape-partition has an unsupported value" >&2
    exit 2
    ;;
esac
if (( shapes_explicit )) && [[ "${shape_partition}" != "all" ]]; then
  echo "error: --shapes and --shape-partition are mutually exclusive" >&2
  exit 2
fi
if [[ "${shape_partition}" != "all" && "${profile}" != "all" ]]; then
  echo "error: --shape-partition requires --profile all" >&2
  exit 2
fi
if (( install )) && [[ "${shape_partition}" != "all" ]]; then
  echo "error: installation requires the complete all-shape transaction" >&2
  exit 2
fi
if (( skip_sweep )) && [[ "${shape_partition}" != "all" ]]; then
  echo "error: a partitioned measurement phase cannot use --skip-sweep" >&2
  exit 2
fi
case "${profile}" in
  quick) measurement_profile="quick" ;;
  family-smoke) measurement_profile="family-smoke" ;;
  qwen36-core|qwen36-lm-head|qwen36-moe|qwen36)
    measurement_profile="partial-production"
    ;;
  all) measurement_profile="production" ;;
esac
if [[ "${measurement_profile}" == "production" &&
      "${shape_partition}" == "all" &&
      ( "${backend}" == "cpu" || "${backend}" == "all" ) ]] &&
   (( ! install )); then
  echo "error: complete CPU training requires --install because grouped " \
       "verifier collection is compiled against the newly certified M=1 policy" >&2
  exit 2
fi

canonical_m_values="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
canonical_verifier_m_values="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
partitioned_measurement=0
partition_m_values="${m_values}"
case "${shape_partition}" in
  fast-development|fast-sealed)
    partitioned_measurement=1
    partition_m_values="1"
    ;;
  verifier-development|verifier-sealed)
    partitioned_measurement=1
    partition_m_values="${canonical_verifier_m_values}"
    ;;
esac
normalize_m_values() {
  printf '%s\n' "$1" |
    tr ',' '\n' |
    sed '/^[[:space:]]*$/d' |
    awk '{ value = $1 + 0; if (value < 1 || $1 !~ /^[0-9]+$/) exit 2; seen[value] = 1 }
         END { for (value in seen) print value }' |
    sort -n |
    paste -sd, -
}

normalized_m_values="$(normalize_m_values "${m_values}")" || {
  echo "error: --m-values must contain positive integers" >&2
  exit 2
}
cpu_grouped_m_values="$(
  printf '%s\n' "${normalized_m_values}" |
    tr ',' '\n' |
    awk '$1 >= 2 { print $1 }' |
    paste -sd, -
)"
if (( install )) && [[ "${profile}" != "all" ]]; then
  echo "error: installing NativeVNNI dispatch requires --profile all" >&2
  exit 2
fi
if (( install )) && [[ "${backend}" != "cpu-prefill" ]] &&
   [[ "${normalized_m_values}" != "${canonical_m_values}" ]]; then
  echo "error: installing NativeVNNI dispatch requires the complete M=1,2..16,31 matrix" >&2
  exit 2
fi
if (( collect_profiler_evidence < 0 )); then
  if [[ "${measurement_profile}" == "production" ]]; then
    collect_profiler_evidence=1
  else
    collect_profiler_evidence=0
  fi
fi
if (( reuse_profiler_evidence && ! skip_sweep )); then
  echo "error: --reuse-profiler-evidence requires --skip-sweep" >&2
  exit 2
fi
if (( install && ! collect_profiler_evidence && ! reuse_profiler_evidence )); then
  echo "error: installation requires complete isolated profiler evidence" >&2
  exit 2
fi

detect_policy_accelerators() {
  local -a detected=()
  local ordinal
  if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r ordinal; do
      [[ "${ordinal}" =~ ^[0-9]+$ ]] && detected+=("cuda:${ordinal}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    while IFS= read -r ordinal; do
      [[ "${ordinal}" =~ ^[0-9]+$ ]] && detected+=("rocm:${ordinal}")
    done < <(
      rocm-smi --showid --csv 2>/dev/null |
        awk -F, 'NR > 1 && $1 ~ /^card[0-9]+$/ {
          sub(/^card/, "", $1)
          print $1
        }'
    )
  fi
  local joined=""
  if ((${#detected[@]})); then
    joined="$(IFS=,; printf '%s' "${detected[*]}")"
  fi
  printf '%s\n' "${joined}"
}

configure_policy_accelerators() {
  # Partition-only runs collect timing evidence but do not fit a policy.
  if [[ "${measurement_profile}" != "production" ]] ||
     [[ "${shape_partition}" != "all" ]] ||
     [[ -n "${collect_cpu_prefill_refinement_plan}" ]] ||
     [[ -n "${collect_cpu_prefill_development_lineage_plan}" ]]; then
    return
  fi
  if [[ "${policy_accelerators}" == "auto" ]]; then
    policy_accelerators="$(detect_policy_accelerators)"
    if [[ -z "${policy_accelerators}" ]]; then
      policy_accelerators="cpu"
    fi
  fi
  if [[ "${policy_accelerators}" == "cpu" ]]; then
    unset LLAMINAR_NATIVE_VNNI_POLICY_ACCELERATORS
    unset LLAMINAR_NATIVE_VNNI_POLICY_CUDA_SCORER_LIBRARY
    unset LLAMINAR_NATIVE_VNNI_POLICY_ROCM_SCORER_LIBRARY
    unset LLAMINAR_NATIVE_VNNI_POLICY_LANES_PER_ACCELERATOR
    printf 'NativeVNNI policy fitter: canonical CPU process pool\n'
    return
  fi
  if [[ -z "${policy_accelerators}" ]]; then
    echo "error: --policy-accelerators cannot be empty" >&2
    exit 2
  fi
  if [[ ! "${policy_lanes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --policy-lanes must be a positive integer" >&2
    exit 2
  fi
  local -a scorer_targets=()
  if [[ "${policy_accelerators}" == *cuda:* ]]; then
    scorer_targets+=(v2_native_vnni_leaf_primary_scorer_cuda)
  fi
  if [[ "${policy_accelerators}" == *rocm:* ]]; then
    scorer_targets+=(v2_native_vnni_leaf_primary_scorer_rocm)
  fi
  if ((${#scorer_targets[@]})); then
    printf 'NativeVNNI policy fitter: rebuilding scorer targets=%s\n' \
      "$(IFS=,; printf '%s' "${scorer_targets[*]}")"
    if (( dry_run )); then
      printf '+ cmake --build %q --parallel --target' \
        "${repo_root}/build_v2_release"
      printf ' %q' "${scorer_targets[@]}"
      printf '\n'
    else
      cmake --build "${repo_root}/build_v2_release" --parallel \
        --target "${scorer_targets[@]}"
    fi
  fi
  if [[ "${policy_accelerators}" == *cuda:* ]]; then
    if (( ! dry_run )) && [[ ! -f "${policy_cuda_scorer}" ]]; then
      echo "error: CUDA policy scorer DSO is missing: ${policy_cuda_scorer}" >&2
      exit 2
    fi
    export LLAMINAR_NATIVE_VNNI_POLICY_CUDA_SCORER_LIBRARY="${policy_cuda_scorer}"
  fi
  if [[ "${policy_accelerators}" == *rocm:* ]]; then
    if (( ! dry_run )) && [[ ! -f "${policy_rocm_scorer}" ]]; then
      echo "error: ROCm policy scorer DSO is missing: ${policy_rocm_scorer}" >&2
      exit 2
    fi
    export LLAMINAR_NATIVE_VNNI_POLICY_ROCM_SCORER_LIBRARY="${policy_rocm_scorer}"
  fi
  export LLAMINAR_NATIVE_VNNI_POLICY_ACCELERATORS="${policy_accelerators}"
  export LLAMINAR_NATIVE_VNNI_POLICY_LANES_PER_ACCELERATOR="${policy_lanes}"
  printf 'NativeVNNI policy fitter: accelerators=%s lanes-per-device=%s\n' \
    "${policy_accelerators}" "${policy_lanes}"
}

configure_policy_accelerators

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

detect_profiler_cpu_lists() {
  local requested_threads="$1"
  local requested_lanes="$2"
  lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, \
    -v wanted="${requested_threads}" -v wanted_lanes="${requested_lanes}" '
    $0 !~ /^#/ {
      if (!seen_socket[$3]++) socket_order[socket_count++] = $3
      key = $3 ":" $2
      if (seen[key]) next
      seen[key] = 1
      if (count[$3] < wanted) cpus[$3, count[$3]++] = $1
    }
    END {
      if (socket_count < wanted_lanes) exit
      for (lane = 0; lane < wanted_lanes; ++lane) {
        socket = socket_order[lane]
        if (count[socket] < wanted) exit
      }
      for (lane = 0; lane < wanted_lanes; ++lane) {
        socket = socket_order[lane]
        if (lane) printf ";"
        for (i = 0; i < wanted; ++i) {
          if (i) printf ","
          printf "%s", cpus[socket, i]
        }
      }
      printf "\n"
    }
  '
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

cpu_serial_arithmetic_contract_hash() {
  python3 - "${cpu_serial_arithmetic_contract_path}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    contract = json.load(handle)
expected = {"schema_version", "contract_hash", "description", "invariants"}
if not isinstance(contract, dict) or set(contract) != expected:
    raise SystemExit(f"{path}: invalid CPU serial arithmetic contract fields")
if contract["schema_version"] != "cpu-native-vnni-serial-m1-arithmetic-contract-v1":
    raise SystemExit(f"{path}: unsupported CPU serial arithmetic contract")
digest = contract["contract_hash"]
if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
    raise SystemExit(f"{path}: invalid CPU serial arithmetic contract hash")
if not isinstance(contract["invariants"], list) or not contract["invariants"]:
    raise SystemExit(f"{path}: CPU serial arithmetic contract has no invariants")
print(digest)
PY
}

csv_unique_field() {
  local path="$1"
  local field="$2"
  python3 - "${path}" "${field}" <<'PY'
import csv
import sys

path, field = sys.argv[1:]
values = set()
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if field not in (reader.fieldnames or ()):
        raise SystemExit(f"{path}: CSV does not contain {field!r}")
    for row in reader:
        value = row[field].strip()
        if not value:
            raise SystemExit(f"{path}: CSV contains an empty {field!r}")
        values.add(value)
        if len(values) > 1:
            raise SystemExit(f"{path}: CSV contains multiple {field!r} values")
if len(values) != 1:
    raise SystemExit(f"{path}: CSV contains no data rows")
print(next(iter(values)))
PY
}

csv_unique_field_prefix() {
  local path="$1"
  local field="$2"
  local delimiter="$3"
  python3 - "${path}" "${field}" "${delimiter}" <<'PY'
import csv
import sys

path, field, delimiter = sys.argv[1:]
values = set()
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if field not in (reader.fieldnames or ()):
        raise SystemExit(f"{path}: CSV does not contain {field!r}")
    for row in reader:
        value = row[field].strip()
        if not value:
            raise SystemExit(f"{path}: CSV contains an empty {field!r}")
        prefix, separator, _suffix = value.partition(delimiter)
        if not separator or not prefix:
            raise SystemExit(
                f"{path}: CSV {field!r} does not contain {delimiter!r}"
            )
        values.add(prefix)
        if len(values) > 1:
            raise SystemExit(
                f"{path}: CSV contains multiple {field!r} prefixes"
            )
if len(values) != 1:
    raise SystemExit(f"{path}: CSV contains no data rows")
print(next(iter(values)))
PY
}

csv_first_cpu_provenance() {
  local path="$1"
  python3 - "${path}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    required = (
        "run_id",
        "git_revision",
        "build_id",
        "compiler_id",
        "architecture_class",
        "device_name",
        "driver_runtime",
        "serial_m1_policy_hash",
    )
    missing = sorted(set(required) - set(reader.fieldnames or ()))
    if missing:
        raise SystemExit(f"{path}: CSV is missing provenance fields {missing}")
    row = next(reader, None)
    if row is None:
        raise SystemExit(f"{path}: CSV contains no data rows")
    values = [row[field].strip() for field in required]
    if any(not value or "\n" in value or "\r" in value for value in values):
        raise SystemExit(f"{path}: CSV has invalid first-row provenance")
    for index, delimiter in ((2, "|cpu_isa="), (4, "|build=")):
        prefix, separator, _suffix = values[index].partition(delimiter)
        if not prefix or not separator:
            raise SystemExit(
                f"{path}: provenance field lacks {delimiter!r}"
            )
        values[index] = prefix
    print("\n".join(values))
PY
}

csv_has_field() {
  local path="$1"
  local field="$2"
  python3 - "${path}" "${field}" <<'PY'
import csv
import sys

path, field = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    header = next(reader, ())
raise SystemExit(0 if field in header else 1)
PY
}

detect_cuda_measurement_lanes() {
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null |
    sed '/^[[:space:]]*$/d' | wc -l
}

detect_rocm_measurement_lanes() {
  rocm-smi --showproductname --csv 2>/dev/null |
    awk -F, '$1 ~ /^card[0-9]+$/ { count += 1 } END { print count + 0 }'
}

detect_rocm_device_name() {
  # rocminfo enumerates CPU agents before GPU agents on heterogeneous hosts, so
  # selecting the first "Marketing Name" silently records the host CPU as the
  # policy device.  rocm-smi's product table is GPU-only; consume the complete
  # CSV stream with the standard parser and select the first physical card.
  rocm-smi --showproductname --csv 2>/dev/null |
    python3 -c '
import csv
import sys

for row in csv.DictReader(sys.stdin):
    if not row.get("device", "").startswith("card"):
        continue
    name = row.get("Card Series", "").strip()
    if name:
        print(name)
        raise SystemExit(0)
raise SystemExit("rocm-smi did not report a physical GPU product name")
'
}

backend_uses_cuda() {
  [[ "${backend}" == "cuda" || "${backend}" == "both" || "${backend}" == "all" ]]
}

backend_uses_rocm() {
  [[ "${backend}" == "rocm" || "${backend}" == "both" || "${backend}" == "all" ]]
}

resolve_gpu_measurement_lanes() {
  local selected_backend="$1"
  local requested="$2"
  local detected=1

  if [[ "${requested}" != "auto" ]]; then
    printf '%s\n' "${requested}"
    return
  fi
  if (( dry_run )); then
    printf '1\n'
    return
  fi
  case "${selected_backend}" in
    cuda) detected="$(detect_cuda_measurement_lanes)" ;;
    rocm) detected="$(detect_rocm_measurement_lanes)" ;;
    *) echo "error: unknown GPU measurement backend ${selected_backend}" >&2; exit 2 ;;
  esac
  if [[ ! "${detected}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: no visible ${selected_backend} GPU for measurement" >&2
    exit 2
  fi
  printf '%s\n' "${detected}"
}

if [[ -z "${cpu_threads}" ]]; then
  cpu_threads="$(detect_cpu_threads_per_socket)"
fi
if [[ ! "${cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cpu-threads must be a positive integer" >&2
  exit 2
fi
if [[ "${cpu_measurement_lanes}" != "auto" ]] &&
   [[ ! "${cpu_measurement_lanes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cpu-measurement-lanes must be a positive integer or auto" >&2
  exit 2
fi
if backend_uses_cuda; then
  cuda_measurement_lanes="$(
    resolve_gpu_measurement_lanes cuda "${cuda_measurement_lanes}"
  )"
else
  cuda_measurement_lanes=1
fi
if backend_uses_rocm; then
  rocm_measurement_lanes="$(
    resolve_gpu_measurement_lanes rocm "${rocm_measurement_lanes}"
  )"
else
  rocm_measurement_lanes=1
fi
if [[ ! "${cuda_measurement_lanes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --cuda-measurement-lanes must be a positive integer or auto" >&2
  exit 2
fi
if [[ ! "${rocm_measurement_lanes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --rocm-measurement-lanes must be a positive integer or auto" >&2
  exit 2
fi
if [[ ! "${cpu_format_shards}" =~ ^[01]$ ]]; then
  echo "error: LLAMINAR_NATIVE_VNNI_CPU_FORMAT_SHARDS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${cpu_batch_limit}" =~ ^[0-9]+$ ]]; then
  echo "error: --cpu-batch-limit must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${backend_collection_target_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --backend-collection-target-seconds must be positive" >&2
  exit 2
fi
if [[ ! "${resume_cpu_partials}" =~ ^[01]$ ]]; then
  echo "error: LLAMINAR_NATIVE_VNNI_RESUME_CPU_PARTIALS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${paired_max_iterations}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --paired-max-iterations must be a positive integer" >&2
  exit 2
fi
available_cpu_sockets="$(cpu_lscpu_field "Socket(s)")"
if [[ ! "${available_cpu_sockets}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: unable to determine physical CPU socket count" >&2
  exit 2
fi
if [[ "${cpu_measurement_lanes}" == "auto" ]]; then
  cpu_measurement_lanes="${available_cpu_sockets}"
fi
if (( cpu_measurement_lanes > available_cpu_sockets )); then
  echo "error: --cpu-measurement-lanes=${cpu_measurement_lanes} exceeds the " \
       "${available_cpu_sockets} physical CPU socket(s)" >&2
  exit 2
fi
if [[ -z "${profiler_cpu_list}" ]]; then
  profiler_cpu_list="$(
    detect_profiler_cpu_lists "${cpu_threads}" "${cpu_measurement_lanes}"
  )"
fi
if [[ -z "${profiler_cpu_list}" ]]; then
  echo "error: unable to derive --profiler-cpu-list" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/benchmark_results/native_vnni_dispatch/${timestamp}-${backend}-${profile}"
fi

quick_formats="Q4_1,Q5_1,Q6_K"
all_formats="$(
  PYTHONPATH="${repo_root}/tests/v2/performance/kernels" \
    python3 -c 'from native_vnni_dispatch.format_registry import FORMAT_SPECS; print(",".join(spec.label for spec in FORMAT_SPECS))'
)"
family_smoke_formats="${all_formats}"
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
shape_manifest_path="${repo_root}/tests/v2/performance/kernels/native_vnni_dispatch/manifests/native_vnni_decode_shapes_v5.json"
cpu_serial_arithmetic_contract_path="${repo_root}/tests/v2/performance/kernels/native_vnni_dispatch/manifests/cpu_native_vnni_serial_m1_arithmetic_v1.json"
gpu_measurement_plan_path="${repo_root}/tests/v2/performance/kernels/native_vnni_dispatch/manifests/native_vnni_gpu_measurement_plan_v1.json"
if [[ -z "${cpu_prefill_split_manifest_path}" ]]; then
  cpu_prefill_split_manifest_path="${repo_root}/tests/v2/performance/kernels/native_vnni_dispatch/manifests/native_vnni_cpu_prefill_split_v12.json"
fi
shape_manifest_python_root="${repo_root}/tests/v2/performance/kernels"
cpu_prefill_training_plan_module="native_vnni_dispatch.cpu_prefill_training_plan"
cpu_prefill_generic_refinement_module="native_vnni_dispatch.cpu_prefill_generic_refinement"
cpu_prefill_development_lineage_module="native_vnni_dispatch.cpu_prefill_development_lineage"
cpu_prefill_candidate_expansion_module="native_vnni_dispatch.cpu_prefill_candidate_expansion"
cpu_prefill_candidate_expansion_rebase_module="native_vnni_dispatch.rebase_cpu_prefill_candidate_expansion_checkpoint"
declare -A native_vnni_shape_n=()
declare -A native_vnni_shape_k=()
native_vnni_shape_names=()
while IFS=$'\t' read -r manifest_name manifest_n manifest_k; do
  native_vnni_shape_names+=("${manifest_name}")
  native_vnni_shape_n["${manifest_name}"]="${manifest_n}"
  native_vnni_shape_k["${manifest_name}"]="${manifest_k}"
done < <(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest --records
)
partition_manifest_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest --names "${shape_partition}"
)"
gpu_common_development_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.measurement_plan \
      --plan "${gpu_measurement_plan_path}" \
      --shape-manifest "${shape_manifest_path}" \
      --names common-development
)"
cuda_q4_fast_refinement_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.measurement_plan \
      --plan "${gpu_measurement_plan_path}" \
      --shape-manifest "${shape_manifest_path}" \
      --names scoped-fast --backend cuda --source-format Q4_0
)"
fast_development_shapes="${gpu_common_development_shapes}"
fast_sealed_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest --names fast-sealed
)"
verifier_development_shapes="${gpu_common_development_shapes}"
verifier_sealed_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest --names verifier-sealed
)"
cpu_verifier_development_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest \
      --names verifier-development --cpu-measurement
)"
cpu_decode_development_shapes="$(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.shape_manifest \
      --names fast-development --cpu-measurement
)"
fast_all_shapes="${fast_development_shapes},${fast_sealed_shapes}"
verifier_all_shapes="${verifier_development_shapes},${verifier_sealed_shapes}"

declare -A cpu_prefill_shape_n=()
declare -A cpu_prefill_shape_k=()
declare -A cpu_prefill_shape_m=()
cpu_prefill_shape_names=()
while IFS=$'\t' read -r prefill_name prefill_n prefill_k prefill_m; do
  cpu_prefill_shape_names+=("${prefill_name}")
  cpu_prefill_shape_n["${prefill_name}"]="${prefill_n}"
  cpu_prefill_shape_k["${prefill_name}"]="${prefill_k}"
  cpu_prefill_shape_m["${prefill_name}"]="${prefill_m}"
done < <(
  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.prefill_matrix --records
)

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
    qwen36) shapes="${qwen36_shapes}" ;;
    all) shapes="${partition_manifest_shapes}" ;;
  esac
fi

stratified_formats=0
cpu_require_policy_keys=0
rocm_execution_modes="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_EXECUTION_MODES:-eager,graph_captured}"
cuda_execution_modes="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_EXECUTION_MODES:-eager,graph_captured}"
rocm_quick_variants="kb1,kb2,kb4,kb8,kb16,kb32,kb64,inherit_serial_m1"
cuda_quick_candidates="cuda.nvnni.decode.fast_m1.wide.tn128.cpt1,cuda.nvnni.decode.fast_m1.direct.tn128.cpt1,cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb1,cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb8,cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb32,cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb64,cuda.nvnni.decode.verifier.inherit_serial_m1.r2,cuda.nvnni.decode.verifier.inherit_serial_m1.r4,cuda.nvnni.decode.verifier.inherit_serial_m1.r8,cuda.nvnni.decode.verifier.inherit_serial_m1.r16,cuda.nvnni.decode.verifier.inherit_serial_m1.r32,cuda.nvnni.decode.verifier.tensor_core_mma16"
case "${profile}" in
  quick)
    cuda_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_MAX_CASES:-24}"
    rocm_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_MAX_CASES:-24}"
    cuda_candidates="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_CANDIDATES:-${cuda_quick_candidates}}"
    cuda_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_WARMUPS:-2}"
    cuda_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_SAMPLES:-5}"
    cuda_timed_replays="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_TIMED_REPLAYS:-4}"
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
    cuda_candidates="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_CANDIDATES:-${cuda_quick_candidates}}"
    cuda_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_WARMUPS:-2}"
    cuda_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_SAMPLES:-5}"
    cuda_timed_replays="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_TIMED_REPLAYS:-4}"
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
    cuda_candidates="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_CANDIDATES:-}"
    cuda_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_WARMUPS:-5}"
    cuda_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_SAMPLES:-30}"
    cuda_timed_replays="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_TIMED_REPLAYS:-16}"
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
    cuda_candidates="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_CANDIDATES:-}"
    cuda_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_WARMUPS:-5}"
    cuda_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_SAMPLES:-30}"
    cuda_timed_replays="${LLAMINAR_NATIVE_VNNI_REFRESH_CUDA_TIMED_REPLAYS:-16}"
    rocm_variants="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_VARIANTS:-}"
    rocm_warmups="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_WARMUPS:-5}"
    rocm_samples="${LLAMINAR_NATIVE_VNNI_REFRESH_ROCM_SAMPLES:-30}"
    cpu_max_cases="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_MAX_CASES:-1000000}"
    cpu_warmup="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP:-5}"
    cpu_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS:-30}"
    ;;
esac

cpu_fixed_timing_gate_args=(
  --minimum-promotion-warmups "${cpu_minimum_promotion_warmups}"
  --minimum-promotion-samples "${cpu_minimum_promotion_samples}"
)
if [[ "${measurement_profile}" == "production" &&
      ( "${backend}" == "cpu" || "${backend}" == "all" ) ]]; then
  if [[ ! "${cpu_warmup}" =~ ^[1-9][0-9]*$ ||
        ! "${cpu_iters}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: production CPU fixed-timing warmups/samples must be positive integers" >&2
    exit 2
  fi
  if (( cpu_warmup < cpu_minimum_promotion_warmups ||
        cpu_iters < cpu_minimum_promotion_samples )); then
    echo "error: production CPU collector is configured for ${cpu_warmup}/${cpu_iters} timing but the installable evidence floor is ${cpu_minimum_promotion_warmups}/${cpu_minimum_promotion_samples}; set matching --cpu-minimum-promotion-* options before collection" >&2
    exit 2
  fi
  printf 'CPU fixed-timing evidence floor: %s warmup(s), %s sample(s)\n' \
    "${cpu_minimum_promotion_warmups}" "${cpu_minimum_promotion_samples}"
fi

# Ordinary CPU prefill uses an elapsed-evidence protocol because one matrix
# spans tiny attention projections and multi-second long-context FFN GEMMs.
# `cpu_iters` remains the hard ceiling; expensive candidates may stop earlier
# only after the minimum population, measured-time floor, and median-stability
# gate have all passed.
cpu_prefill_min_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_MIN_ITERS:-5}"
cpu_prefill_max_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_MAX_ITERS:-180}"
cpu_prefill_refinement_launch_attempts="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_REFINEMENT_LAUNCH_ATTEMPTS:-5}"
# Candidate expansion compares a small additive family against interleaved
# anchors and retains at least five complete rounds per candidate. A separate
# 100 ms elapsed floor raises inexpensive kernels well beyond that population,
# while a long-running GEMM is not repeated fifteen times after it has already
# supplied several seconds of representative work. The full recovery trigger
# remains available when a very fast candidate reaches 180 rounds just before
# satisfying the elapsed floor.
cpu_prefill_candidate_expansion_max_iters="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_CANDIDATE_EXPANSION_MAX_ITERS:-180}"
cpu_prefill_candidate_expansion_timing_budget_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_CANDIDATE_EXPANSION_TIMING_BUDGET_US:-100000}"
cpu_prefill_timing_budget_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_TIMING_BUDGET_US:-100000}"
cpu_prefill_median_stability="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_MEDIAN_STABILITY:-0.02}"
cpu_prefill_default_warmup_budget_us=0
cpu_prefill_default_transition_warmup_budget_us=0
if [[ "${measurement_profile}" == "production" ||
      "${measurement_profile}" == "partial-production" ]]; then
  # Production rows retain thousands of samples for fast kernels and at least
  # the explicit minimum for slow kernels. A 250 ms budget per candidate made
  # fixed-cost warmup/timing dominate the six-thousand-cell corpus without
  # improving convergence; 50/100 ms preserves the stability gate and keeps a
  # complete backend collection near its two-hour operational target.
  cpu_prefill_default_warmup_budget_us=50000
  cpu_prefill_default_transition_warmup_budget_us=50000
fi
cpu_prefill_warmup_budget_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_WARMUP_BUDGET_US:-${cpu_prefill_default_warmup_budget_us}}"
cpu_prefill_transition_warmup_budget_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_TRANSITION_WARMUP_BUDGET_US:-${cpu_prefill_default_transition_warmup_budget_us}}"
cpu_prefill_transition_warmup_latency_multiplier="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_TRANSITION_WARMUP_LATENCY_MULTIPLIER:-60}"
cpu_prefill_transition_warmup_budget_ceiling_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_TRANSITION_WARMUP_BUDGET_CEILING_US:-500000}"
cpu_prefill_warmup_round_timeout_us="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_WARMUP_ROUND_TIMEOUT_US:-30000000}"

active_backend_target_start=-1
active_backend_target_label=""

begin_backend_collection_target() {
  active_backend_target_start=${SECONDS}
  active_backend_target_label="$1"
}

finish_backend_collection_target() {
  if (( active_backend_target_start < 0 )); then
    return
  fi
  local elapsed=$((SECONDS - active_backend_target_start))
  local status="within target"
  if (( elapsed > backend_collection_target_seconds )); then
    status="TARGET MISSED"
  fi
  printf '%s corpus transaction completed in %ds / %ds (%s)\n' \
    "${active_backend_target_label}" "${elapsed}" \
    "${backend_collection_target_seconds}" "${status}"
  active_backend_target_start=-1
  active_backend_target_label=""
}

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

require_gtest_case() {
  local binary="$1"
  local test_name="$2"
  if (( dry_run || skip_sweep )); then
    return
  fi
  if ! "${binary}" --gtest_list_tests 2>/dev/null |
       grep -Fq "${test_name}"; then
    echo "error: ${binary} does not contain required gtest ${test_name}; " \
         "rebuild the matching ISA trainer target" >&2
    exit 2
  fi
}

require_cpu_performance_governor() {
  if (( dry_run )); then
    printf 'dry-run: require every CPU frequency policy governor == performance\n'
    return
  fi
  local governor_path governor
  local found=0
  for governor_path in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    if [[ ! -r "${governor_path}" ]]; then
      continue
    fi
    found=1
    governor="$(<"${governor_path}")"
    if [[ "${governor}" != "performance" ]]; then
      echo "error: production CPU timing requires performance governor; " \
           "${governor_path} is ${governor}" >&2
      echo "hint: sudo sh -c 'for path in " \
           "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor; " \
           "do printf performance > \"\$path\"; done'" >&2
      exit 2
    fi
  done
  if (( ! found )); then
    echo "error: production CPU timing cannot read cpufreq policy governors" >&2
    exit 2
  fi
}

mkdir -p "${output_dir}"

cuda_csv="${output_dir}/cuda_decode_sweep.csv"
cuda_timing_csv="${output_dir}/cuda_decode_sweep.timing.csv"
cuda_m1_csv="${output_dir}/cuda_decode_m1.csv"
cuda_m1_timing_csv="${output_dir}/cuda_decode_m1.timing.csv"
cuda_m1_development_csv="${output_dir}/cuda_decode_m1.development.csv"
cuda_m1_development_timing_csv="${output_dir}/cuda_decode_m1.development.timing.csv"
cuda_m1_development_common_csv="${output_dir}/cuda_decode_m1.development-common.csv"
cuda_m1_development_common_timing_csv="${output_dir}/cuda_decode_m1.development-common.timing.csv"
cuda_m1_development_scoped_csv="${output_dir}/cuda_decode_m1.development-cuda-q4-refinement.csv"
cuda_m1_development_scoped_timing_csv="${output_dir}/cuda_decode_m1.development-cuda-q4-refinement.timing.csv"
cuda_m1_sealed_csv="${output_dir}/cuda_decode_m1.sealed.csv"
cuda_m1_sealed_timing_csv="${output_dir}/cuda_decode_m1.sealed.timing.csv"
cuda_verifier_csv="${output_dir}/cuda_decode_verifier.csv"
cuda_verifier_timing_csv="${output_dir}/cuda_decode_verifier.timing.csv"
cuda_verifier_development_csv="${output_dir}/cuda_decode_verifier.development.csv"
cuda_verifier_development_timing_csv="${output_dir}/cuda_decode_verifier.development.timing.csv"
cuda_verifier_sealed_csv="${output_dir}/cuda_decode_verifier.sealed.csv"
cuda_verifier_sealed_timing_csv="${output_dir}/cuda_decode_verifier.sealed.timing.csv"
cuda_m1_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.m1-staged.inc"
cuda_m1_provisional_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.m1-provisional.inc"
cuda_m1_frozen_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.m1-frozen-development.inc"
cuda_m1_summary="${output_dir}/cuda_decode_m1_summary.csv"
cuda_m1_common_csv="${output_dir}/cuda_decode_m1_common_observations.csv"
cuda_m1_frozen_policy_json="${output_dir}/cuda_decode_m1_frozen_policy.json"
cuda_m1_policy_json="${output_dir}/cuda_decode_m1_policy.json"
cuda_inc="${output_dir}/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"
cuda_summary="${output_dir}/cuda_decode_dispatch_summary.txt"
cuda_common_csv="${output_dir}/cuda_decode_common_observations.csv"
cuda_policy_json="${output_dir}/cuda_decode_policy.json"
rocm_csv="${output_dir}/rocm_decode_sweep.csv"
rocm_timing_csv="${output_dir}/rocm_decode_sweep.timing.csv"
rocm_fast_csv="${output_dir}/rocm_decode_fast.csv"
rocm_fast_timing_csv="${output_dir}/rocm_decode_fast.timing.csv"
rocm_fast_development_csv="${output_dir}/rocm_decode_fast.development.csv"
rocm_fast_development_timing_csv="${output_dir}/rocm_decode_fast.development.timing.csv"
rocm_fast_sealed_csv="${output_dir}/rocm_decode_fast.sealed.csv"
rocm_fast_sealed_timing_csv="${output_dir}/rocm_decode_fast.sealed.timing.csv"
rocm_verifier_csv="${output_dir}/rocm_decode_verifier.csv"
rocm_verifier_timing_csv="${output_dir}/rocm_decode_verifier.timing.csv"
rocm_inc="${output_dir}/ROCmNativeVNNIDecodeDispatchGenerated.inc"
rocm_provisional_inc="${output_dir}/ROCmNativeVNNIDecodeDispatchGenerated.provisional.inc"
rocm_frozen_inc="${output_dir}/ROCmNativeVNNIDecodeDispatchGenerated.frozen-development.inc"
rocm_frozen_policy_json="${output_dir}/rocm_decode_frozen_policy.json"
rocm_policy_json="${output_dir}/rocm_decode_policy.json"
rocm_summary="${output_dir}/rocm_decode_dispatch_summary.txt"
rocm_common_csv="${output_dir}/rocm_decode_common_observations.csv"
rocm_fast_common_csv="${output_dir}/rocm_decode_fast_common_observations.csv"
rocm_verifier_common_csv="${output_dir}/rocm_decode_verifier_common_observations.csv"
rocm_base_include="${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
cpu_csv="${output_dir}/cpu_verifier_rows_sweep.csv"
cpu_timing_csv="${output_dir}/cpu_verifier_rows_sweep.timing.csv"
cpu_development_csv="${output_dir}/cpu_verifier_rows.development.csv"
cpu_development_timing_csv="${output_dir}/cpu_verifier_rows.development.timing.csv"
cpu_grouped_sealed_plan_json="${output_dir}/cpu_verifier_rows.sealed-plan.json"
cpu_grouped_sealed_request_dir="${output_dir}/cpu_verifier_rows.sealed-requests"
cpu_grouped_fit_cache_dir="${output_dir}/policy_fit_cache/cpu_grouped"
cpu_inc="${output_dir}/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
cpu_provisional_inc="${output_dir}/CPUNativeVNNIVerifierRowsPolicyGenerated.provisional.inc"
cpu_frozen_inc="${output_dir}/CPUNativeVNNIVerifierRowsPolicyGenerated.frozen-development.inc"
cpu_frozen_policy_json="${output_dir}/cpu_verifier_rows_frozen_policy.json"
cpu_policy_json="${output_dir}/cpu_verifier_rows_policy.json"
cpu_grouped_certification_diagnostic="${output_dir}/cpu_verifier_rows_certification_diagnostic.json"
cpu_summary="${output_dir}/cpu_verifier_rows_policy_summary.txt"
cpu_common_csv="${output_dir}/cpu_verifier_rows_common_observations.csv"
cpu_decode_csv="${output_dir}/cpu_decode_m1_sweep.csv"
cpu_decode_timing_csv="${output_dir}/cpu_decode_m1_sweep.timing.csv"
cpu_decode_development_csv="${output_dir}/cpu_decode_m1.development.csv"
cpu_decode_development_timing_csv="${output_dir}/cpu_decode_m1.development.timing.csv"
cpu_decode_sealed_route_probe_json="${output_dir}/cpu_decode_m1.sealed-route-probe.json"
cpu_decode_sealed_plan_json="${output_dir}/cpu_decode_m1.sealed-plan.json"
cpu_decode_sealed_request_dir="${output_dir}/cpu_decode_m1.sealed-requests"
cpu_decode_inc="${output_dir}/CPUNativeVNNIDecodePolicyGenerated.inc"
cpu_decode_provisional_inc="${output_dir}/CPUNativeVNNIDecodePolicyGenerated.provisional.inc"
cpu_decode_frozen_inc="${output_dir}/CPUNativeVNNIDecodePolicyGenerated.frozen-development.inc"
cpu_decode_frozen_policy_json="${output_dir}/cpu_decode_m1_frozen_policy.json"
cpu_decode_policy_json="${output_dir}/cpu_decode_m1_policy.json"
cpu_decode_certification_diagnostic="${output_dir}/cpu_decode_m1_certification_diagnostic.json"
cpu_decode_summary="${output_dir}/cpu_decode_m1_policy_summary.csv"
cpu_decode_common_csv="${output_dir}/cpu_decode_m1_common_observations.csv"
cpu_decode_fit_cache_dir="${output_dir}/policy_fit_cache/cpu_decode"
cpu_prefill_csv="${output_dir}/cpu_prefill_sweep.csv"
cpu_prefill_timing_csv="${output_dir}/cpu_prefill_sweep.timing.csv"
cpu_prefill_legacy_v3_csv="${output_dir}/cpu_prefill_sweep.sealed.csv"
cpu_prefill_legacy_v3_timing_csv="${output_dir}/cpu_prefill_sweep.sealed.timing.csv"
cpu_prefill_refinement_csv="${output_dir}/cpu_prefill_sweep.development-refinement-v4.csv"
cpu_prefill_refinement_timing_csv="${output_dir}/cpu_prefill_sweep.development-refinement-v4.timing.csv"
cpu_prefill_development_v4_csv="${output_dir}/cpu_prefill_sweep.development-v4.csv"
cpu_prefill_development_v4_timing_csv="${output_dir}/cpu_prefill_sweep.development-v4.timing.csv"
cpu_prefill_opened_v5_csv="${output_dir}/cpu_prefill_sweep.sealed-v5.csv"
cpu_prefill_opened_v5_timing_csv="${output_dir}/cpu_prefill_sweep.sealed-v5.timing.csv"
cpu_prefill_opened_v5_csv_sha256="293c45f4b4da163724995b80a564d7cbdfe09ee392918957fb7e8c1fec195fc5"
cpu_prefill_opened_v5_timing_sha256="a865e4b90e72f69f29010713f56bcc5678596aa4ce2ecf775610b395b60fcb45"
cpu_prefill_development_v6_csv="${output_dir}/cpu_prefill_sweep.development-v6.csv"
cpu_prefill_development_v6_timing_csv="${output_dir}/cpu_prefill_sweep.development-v6.timing.csv"
cpu_prefill_opened_v6_csv="${output_dir}/cpu_prefill_sweep.sealed-v6.csv"
cpu_prefill_opened_v6_timing_csv="${output_dir}/cpu_prefill_sweep.sealed-v6.timing.csv"
cpu_prefill_opened_v6_csv_sha256="0a76a5f1fb18ce1347bcac1bff2f83a5b40250f62e165d006ef039fd8cba9e5c"
cpu_prefill_opened_v6_timing_sha256="5f95e5d7a5a02e92e822e49ea63a8a2a9b3184505452ebf1e9abd2c392b5e59f"
cpu_prefill_development_v7_csv="${output_dir}/cpu_prefill_sweep.development-v7.csv"
cpu_prefill_development_v7_timing_csv="${output_dir}/cpu_prefill_sweep.development-v7.timing.csv"
cpu_prefill_opened_v7_csv="${output_dir}/cpu_prefill_sweep.sealed-v7.csv"
cpu_prefill_opened_v7_timing_csv="${output_dir}/cpu_prefill_sweep.sealed-v7.timing.csv"
cpu_prefill_opened_v7_csv_sha256="c113ff2e6eceb747e455ecbd4f3c0eccfb27f14d66d9b77d04e60feacfd72448"
cpu_prefill_opened_v7_timing_sha256="b65a63404c5ab2f1b523ea9a290cdd591bc19de1bfafff51cee36b7f14481eeb"
cpu_prefill_development_csv="${output_dir}/cpu_prefill_sweep.development-v8.csv"
cpu_prefill_development_timing_csv="${output_dir}/cpu_prefill_sweep.development-v8.timing.csv"
if [[ -n "${cpu_prefill_fit_development_lineage_plan}" ]]; then
  cpu_prefill_development_csv="${output_dir}/cpu_prefill_sweep.development-lineage.combined.csv"
  cpu_prefill_development_timing_csv="${output_dir}/cpu_prefill_sweep.development-lineage.combined.timing.csv"
fi
cpu_prefill_sealed_csv="${output_dir}/cpu_prefill_sweep.sealed-v8.csv"
cpu_prefill_sealed_timing_csv="${output_dir}/cpu_prefill_sweep.sealed-v8.timing.csv"
cpu_prefill_inc="${output_dir}/CPUNativeVNNIPrefillPolicyGenerated.inc"
cpu_prefill_provisional_inc="${output_dir}/CPUNativeVNNIPrefillPolicyGenerated.provisional-v8.inc"
cpu_prefill_frozen_inc="${output_dir}/CPUNativeVNNIPrefillPolicyGenerated.frozen-development-v8.inc"
cpu_prefill_frozen_policy_json="${output_dir}/cpu_prefill_frozen_policy.v8.json"
cpu_prefill_sealed_witness_plan_json="${output_dir}/cpu_prefill_sealed_witness_plan.v8.json"
cpu_prefill_policy_json="${output_dir}/cpu_prefill_policy.v8.json"
cpu_prefill_certification_diagnostic="${output_dir}/cpu_prefill_certification_diagnostic.v8.json"
cpu_prefill_fit_cache_dir="${output_dir}/policy_fit_cache/cpu_prefill"
cpu_prefill_summary="${output_dir}/cpu_prefill_policy_summary.v8.csv"
cpu_prefill_common_csv="${output_dir}/cpu_prefill_common_observations.v8.csv"
cpu_prefill_development_fit_diagnostic="${output_dir}/cpu_prefill_development_fit.noninstallable.json"
cpu_prefill_max_refinement_rounds="${LLAMINAR_NATIVE_VNNI_REFRESH_CPU_PREFILL_MAX_REFINEMENT_ROUNDS:-6}"
if [[ -n "${cpu_prefill_fit_replay_recipe}" ]] &&
   [[ "$(realpath -m "${cpu_prefill_replay_checkpoint_path}")" != "$(realpath -m "${cpu_prefill_common_csv}")" ]]; then
  echo "error: CPU prefill replay recipe checkpoint must be ${cpu_prefill_common_csv}, got ${cpu_prefill_replay_checkpoint_path}" >&2
  exit 2
fi

cuda_profiler_requests="${output_dir}/cuda_profiler_requests.json"
cuda_profiler_evidence="${output_dir}/cuda_profiler_evidence.json"
cuda_profiler_features="${output_dir}/cuda_profiler_features.csv"
cuda_profiler_raw="${output_dir}/cuda_profiler_raw"
cuda_final_profiler_requests="${output_dir}/cuda_final_profiler_requests.json"
cuda_final_profiler_evidence="${output_dir}/cuda_final_profiler_evidence.json"
cuda_final_profiler_features="${output_dir}/cuda_final_profiler_features.csv"
cuda_final_profiler_raw="${output_dir}/cuda_final_profiler_raw"
rocm_profiler_requests="${output_dir}/rocm_profiler_requests.json"
rocm_profiler_evidence="${output_dir}/rocm_profiler_evidence.json"
rocm_profiler_features="${output_dir}/rocm_profiler_features.csv"
rocm_profiler_raw="${output_dir}/rocm_profiler_raw"
rocm_final_profiler_requests="${output_dir}/rocm_final_profiler_requests.json"
rocm_final_profiler_evidence="${output_dir}/rocm_final_profiler_evidence.json"
rocm_final_profiler_features="${output_dir}/rocm_final_profiler_features.csv"
rocm_final_profiler_raw="${output_dir}/rocm_final_profiler_raw"
cpu_profiler_requests="${output_dir}/cpu_profiler_requests.json"
cpu_profiler_evidence="${output_dir}/cpu_profiler_evidence.json"
cpu_profiler_features="${output_dir}/cpu_profiler_features.csv"
cpu_profiler_raw="${output_dir}/cpu_profiler_raw"
cpu_final_profiler_requests="${output_dir}/cpu_final_profiler_requests.json"
cpu_final_profiler_evidence="${output_dir}/cpu_final_profiler_evidence.json"
cpu_final_profiler_features="${output_dir}/cpu_final_profiler_features.csv"
cpu_final_profiler_raw="${output_dir}/cpu_final_profiler_raw"
cpu_decode_profiler_requests="${output_dir}/cpu_decode_profiler_requests.json"
cpu_decode_profiler_evidence="${output_dir}/cpu_decode_profiler_evidence.json"
cpu_decode_profiler_features="${output_dir}/cpu_decode_profiler_features.csv"
cpu_decode_profiler_witnesses="${output_dir}/cpu_decode_profiler_observation_witnesses.csv"
cpu_decode_profiler_raw="${output_dir}/cpu_decode_profiler_raw"
cpu_decode_development_run_id_file="${output_dir}/cpu_decode_development_run_id.txt"
cpu_grouped_development_run_id_file="${output_dir}/cpu_grouped_development_run_id.txt"
cpu_decode_final_profiler_requests="${output_dir}/cpu_decode_final_profiler_requests.json"
cpu_decode_final_profiler_evidence="${output_dir}/cpu_decode_final_profiler_evidence.json"
cpu_decode_final_profiler_features="${output_dir}/cpu_decode_final_profiler_features.csv"
cpu_decode_final_profiler_raw="${output_dir}/cpu_decode_final_profiler_raw"
cpu_prefill_profiler_requests="${output_dir}/cpu_prefill_profiler_requests.json"
cpu_prefill_profiler_evidence="${output_dir}/cpu_prefill_profiler_evidence.json"
cpu_prefill_profiler_features="${output_dir}/cpu_prefill_profiler_features.csv"
cpu_prefill_profiler_witnesses="${output_dir}/cpu_prefill_profiler_observation_witnesses.csv"
cpu_prefill_profiler_raw="${output_dir}/cpu_prefill_profiler_raw"
cpu_prefill_profiler_source_common="${output_dir}/cpu_prefill_profiler_source_observations.csv"
cpu_prefill_final_profiler_requests="${output_dir}/cpu_prefill_final_profiler_requests.json"
cpu_prefill_final_profiler_evidence="${output_dir}/cpu_prefill_final_profiler_evidence.json"
cpu_prefill_final_profiler_features="${output_dir}/cpu_prefill_final_profiler_features.csv"
cpu_prefill_final_profiler_witnesses="${output_dir}/cpu_prefill_final_profiler_observation_witnesses.csv"
cpu_prefill_final_profiler_raw="${output_dir}/cpu_prefill_final_profiler_raw"
if [[ -n "${cpu_prefill_fit_primary_profiler_requests}" ]]; then
  # A replay recipe may replace the historical implicit profiler base with a
  # complete matched-anchor transaction. Keep the derived feature table local
  # to this run; the recipe authenticates only paid requests, counters, and
  # compact timing witnesses.
  cpu_prefill_profiler_requests="${cpu_prefill_fit_primary_profiler_requests}"
  cpu_prefill_profiler_evidence="${cpu_prefill_fit_primary_profiler_evidence}"
  cpu_prefill_profiler_witnesses="${cpu_prefill_fit_primary_profiler_observations}"
  cpu_prefill_profiler_features="${output_dir}/cpu_prefill_profiler_features.active.csv"
fi

dispatch_validator="${repo_root}/tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py"
cuda_generator="${repo_root}/tests/v2/performance/kernels/cuda/gemm/analyze_cuda_native_vnni_decode_trainer.py"
paired_planner_root="${repo_root}/tests/v2/performance/kernels"
rocm_generator="${repo_root}/tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py"
cpu_generator="${repo_root}/tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py"
cpu_decode_generator="${repo_root}/tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_decode_trainer.py"
cpu_prefill_generator="${repo_root}/tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_prefill_trainer.py"
profiler_python_root="${repo_root}/tests/v2/performance/kernels"

collect_backend_profiler_evidence() {
  local selected_backend="$1"
  local common_observations="$2"
  local trainer_binary="$3"
  local request_manifest="$4"
  local evidence_manifest="$5"
  local feature_table="$6"
  local raw_directory="$7"
  local requested_mode="${8:-auto}"
  local witness_table="${9:-${feature_table%_features.csv}_observation_witnesses.csv}"

  if [[ "${requested_mode}" != "auto" &&
        "${requested_mode}" != "reuse" &&
        "${requested_mode}" != "collect" ]]; then
    echo "error: invalid profiler evidence mode: ${requested_mode}" >&2
    exit 2
  fi
  local reuse_this_evidence="${reuse_profiler_evidence}"
  if [[ "${requested_mode}" == "reuse" ]]; then
    reuse_this_evidence=1
  elif [[ "${requested_mode}" == "collect" ]]; then
    reuse_this_evidence=0
  fi

  materialize_profiler_observation_witnesses() {
    local mode="$1"
    local -a witness_source=()
    if [[ "${mode}" == "reuse" && -s "${witness_table}" ]]; then
      return
    fi
    if [[ "${mode}" == "reuse" && -s "${feature_table}" ]]; then
      # Older transactions retained the exported feature table but not its
      # compact timing witness. Recover only its strict common-observation
      # columns, verify their embedded digests, and authenticate them against
      # the immutable request/evidence manifests before any refit consumes it.
      witness_source=(--feature-table "${feature_table}")
    elif [[ -s "${common_observations}" ]] || (( dry_run )); then
      witness_source=(--observation "${common_observations}")
    else
      echo "error: profiler evidence has no timing observation source: ${common_observations}" >&2
      exit 2
    fi
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.profiler_evidence \
      compact-witnesses \
      "${witness_source[@]}" \
      --requests "${request_manifest}" \
      --evidence "${evidence_manifest}" \
      --output "${witness_table}"
  }

  if (( reuse_this_evidence )); then
    local required_profiler_path
    if (( ! dry_run )); then
      for required_profiler_path in \
          "${request_manifest}" \
          "${evidence_manifest}"; do
        if [[ ! -s "${required_profiler_path}" ]]; then
          echo "error: fit-only profiler evidence is missing: ${required_profiler_path}" >&2
          exit 2
        fi
      done
    fi
    # Re-exporting features is deliberately cheap and authenticates the
    # canonical observation digest, per-dispatch requests, and isolated
    # profiler evidence before any fit may consume the bundle.
    materialize_profiler_observation_witnesses reuse
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.profiler_evidence \
      export-features \
      --observation "${witness_table}" \
      --requests "${request_manifest}" \
      --evidence "${evidence_manifest}" \
      --output "${feature_table}"
    return
  fi

  if (( ! collect_profiler_evidence )); then
    return
  fi

  local active_request_manifest="${request_manifest}"
  local active_evidence_manifest="${evidence_manifest}"
  local active_raw_directory="${raw_directory}"
  local delta_observations=""
  local delta_witnesses=""
  local extending_profiler_transaction=0
  local profile_active_transaction=1
  local -a retained_delta_requests=()
  local -a retained_delta_evidence=()
  local -a retained_delta_witnesses=()

  if (( dry_run )); then
    # Dry-run output remains a readable description of the first transaction.
    # Existing files are deliberately not inspected because the command must
    # be reproducible on a checkout without a local corpus.
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.profiler_evidence \
      emit-requests \
      --observation "${common_observations}" \
      --output "${request_manifest}"
  elif [[ -s "${request_manifest}" && ! -s "${evidence_manifest}" ]]; then
    # Request publication precedes a potentially multi-hour counter run. An
    # interrupt in that window is an ordinary resumable initial transaction,
    # not a corrupt half of a completed immutable transaction. The collector
    # below resumes a matching journal when present or starts the still-empty
    # evidence side from the already authenticated request inventory.
    :
  elif [[ -s "${request_manifest}" || -s "${evidence_manifest}" ]]; then
    # A corpus extension must not replace an already paid profiler transaction.
    # Authenticate both old manifests and its compact timing witnesses first,
    # then derive requests only for exact physical launches absent from that
    # transaction. The collector therefore never replays unchanged candidates.
    if [[ ! -s "${request_manifest}" || ! -s "${evidence_manifest}" ]]; then
      echo "error: profiler resume requires both request and evidence manifests" >&2
      exit 2
    fi
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.profiler_evidence \
      validate-evidence \
      --requests "${request_manifest}" \
      --evidence "${evidence_manifest}"
    materialize_profiler_observation_witnesses reuse

    # A previous invocation may have completed a content-addressed delta but
    # been interrupted before the atomic canonical composition. Discover and
    # authenticate those durable transactions before deriving another delta.
    # Coverage uses exact physical launch identity, so regenerated run IDs do
    # not repay for counters while changed arithmetic or schedules still do.
    local -a published_coverage_requests=("${request_manifest}")
    local request_prefix="${request_manifest%.json}"
    local evidence_prefix="${evidence_manifest%.json}"
    local witness_prefix="${witness_table%.csv}"
    local -a delta_request_candidates=()
    shopt -s nullglob
    delta_request_candidates=("${request_prefix}".delta-*.requests.json)
    shopt -u nullglob
    local retained_request
    for retained_request in "${delta_request_candidates[@]}"; do
      local retained_tag="${retained_request#${request_prefix}.delta-}"
      retained_tag="${retained_tag%.requests.json}"
      local retained_evidence="${evidence_prefix}.delta-${retained_tag}.evidence.json"
      local retained_observations="${request_prefix}.delta-${retained_tag}.observations.csv"
      local retained_witness="${witness_prefix}.delta-${retained_tag}.csv"
      if [[ ! -s "${retained_evidence}" ]]; then
        # An in-progress journal is resumed below only when its regenerated
        # request transaction remains the active missing delta.
        continue
      fi
      run_cmd \
        "PYTHONPATH=${profiler_python_root}" \
        python3 -m native_vnni_dispatch.profiler_evidence \
        validate-evidence \
        --requests "${retained_request}" \
        --evidence "${retained_evidence}"
      local uncovered_count
      local -a uncovered_command=(
        python3 -m native_vnni_dispatch.profiler_evidence
        count-uncovered-requests
        --requests "${retained_request}"
      )
      local covered_request
      for covered_request in "${published_coverage_requests[@]}"; do
        uncovered_command+=(--covered-requests "${covered_request}")
      done
      uncovered_count="$(PYTHONPATH="${profiler_python_root}" "${uncovered_command[@]}")"
      if (( uncovered_count == 0 )); then
        continue
      fi
      if [[ ! -s "${retained_witness}" ]]; then
        if [[ ! -s "${retained_observations}" ]]; then
          echo "error: completed profiler delta has no timing witnesses: ${retained_request}" >&2
          exit 2
        fi
        run_cmd \
          "PYTHONPATH=${profiler_python_root}" \
          python3 -m native_vnni_dispatch.profiler_evidence \
          compact-witnesses \
          --observation "${retained_observations}" \
          --requests "${retained_request}" \
          --evidence "${retained_evidence}" \
          --output "${retained_witness}"
      fi
      retained_delta_requests+=("${retained_request}")
      retained_delta_evidence+=("${retained_evidence}")
      retained_delta_witnesses+=("${retained_witness}")
      published_coverage_requests+=("${retained_request}")
    done

    local delta_observations_inprogress="${request_manifest%.json}.delta.observations.inprogress.csv"
    local delta_requests_inprogress="${request_manifest%.json}.delta.requests.inprogress.json"
    rm -f "${delta_observations_inprogress}" "${delta_requests_inprogress}"
    local -a missing_request_command=(
      python3 -m native_vnni_dispatch.profiler_evidence
      emit-missing-requests
      --observation "${common_observations}"
      --output-observation "${delta_observations_inprogress}"
      --output-requests "${delta_requests_inprogress}"
    )
    local covered_request
    for covered_request in "${published_coverage_requests[@]}"; do
      missing_request_command+=(--covered-requests "${covered_request}")
    done
    run_cmd "PYTHONPATH=${profiler_python_root}" "${missing_request_command[@]}"

    if [[ ! -s "${delta_requests_inprogress}" ]]; then
      # No missing request file is the explicit proof that the retained
      # transaction and any recovered complete deltas cover the enlarged
      # timing corpus. Publish recovered deltas before exporting features.
      if (( ${#retained_delta_requests[@]} == 0 )); then
        materialize_profiler_observation_witnesses reuse
        run_cmd \
          "PYTHONPATH=${profiler_python_root}" \
          python3 -m native_vnni_dispatch.profiler_evidence \
          export-features \
          --observation "${witness_table}" \
          --requests "${request_manifest}" \
          --evidence "${evidence_manifest}" \
          --output "${feature_table}"
        return
      fi
      extending_profiler_transaction=1
      profile_active_transaction=0
    else
      # The request bytes name every durable delta artifact. Re-running the
      # same extension resumes the same evidence file and raw directory; a
      # different corpus receives a disjoint identity and cannot overwrite
      # prior counters.
      local delta_digest
      delta_digest="$(sha256sum "${delta_requests_inprogress}" | cut -d' ' -f1)"
      local delta_tag="${delta_digest:0:16}"
      delta_observations="${request_manifest%.json}.delta-${delta_tag}.observations.csv"
      active_request_manifest="${request_manifest%.json}.delta-${delta_tag}.requests.json"
      active_evidence_manifest="${evidence_manifest%.json}.delta-${delta_tag}.evidence.json"
      active_raw_directory="${raw_directory}.delta-${delta_tag}"
      delta_witnesses="${witness_table%.csv}.delta-${delta_tag}.csv"
      if [[ -s "${active_request_manifest}" ]]; then
        if ! cmp --silent "${delta_requests_inprogress}" "${active_request_manifest}" ||
           ! cmp --silent "${delta_observations_inprogress}" "${delta_observations}"; then
          echo "error: profiler delta identity collides with different contents" >&2
          exit 2
        fi
        rm -f "${delta_requests_inprogress}" "${delta_observations_inprogress}"
      else
        mv "${delta_requests_inprogress}" "${active_request_manifest}"
        mv "${delta_observations_inprogress}" "${delta_observations}"
      fi
      extending_profiler_transaction=1
    fi
  else
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.profiler_evidence \
      emit-requests \
      --observation "${common_observations}" \
      --output "${request_manifest}"
  fi

  local -a collector=(
    python3 -m native_vnni_dispatch.profiler_collectors
    --requests "${active_request_manifest}"
    --output "${active_evidence_manifest}"
    --raw-directory "${active_raw_directory}"
    --backend "${selected_backend}"
    --binary "${trainer_binary}"
    --timeout-seconds 300
    --finalize
  )
  case "${selected_backend}" in
    cuda)
      collector+=(--device-lanes "${cuda_measurement_lanes}")
      if [[ -n "${ncu_path}" ]]; then
        collector+=(--tool "${ncu_path}")
      fi
      if (( EUID != 0 )) &&
         grep -q '^RmProfilingAdminOnly:[[:space:]]*1$' \
           /proc/driver/nvidia/params 2>/dev/null; then
        collector+=(--sudo-tool)
      fi
      ;;
    rocm)
      collector+=(--device-lanes "${rocm_measurement_lanes}")
      if [[ -n "${rocprofv3_path}" ]]; then
        collector+=(--tool "${rocprofv3_path}")
      fi
      ;;
    cpu)
      collector+=(
        --cpu-list "${profiler_cpu_list}"
        --device-lanes "${cpu_measurement_lanes}"
        --cpu-avx2-binary "${cpu_avx2_sweep_bin}"
        --cpu-avx512-binary "${cpu_avx512_sweep_bin}"
      )
      if [[ -n "${perf_path}" ]]; then
        collector+=(--tool "${perf_path}")
      fi
      ;;
  esac
  if (( profile_active_transaction )); then
    if [[ -s "${active_evidence_manifest}" ||
          -s "${active_evidence_manifest}.inprogress.jsonl" ]]; then
      collector+=(--resume --retry-failures)
    fi
    run_cmd "PYTHONPATH=${profiler_python_root}" "${collector[@]}"
  fi
  if (( extending_profiler_transaction )); then
    # First compact and authenticate the new transaction independently. Only a
    # complete delta may be composed with the retained source. Composition
    # republishes canonical manifests through in-progress paths so interruption
    # leaves the previous complete transaction untouched and resumable.
    if (( profile_active_transaction )); then
      run_cmd \
        "PYTHONPATH=${profiler_python_root}" \
        python3 -m native_vnni_dispatch.profiler_evidence \
        compact-witnesses \
        --observation "${delta_observations}" \
        --requests "${active_request_manifest}" \
        --evidence "${active_evidence_manifest}" \
        --output "${delta_witnesses}"
    fi
    local composed_witnesses="${witness_table}.composed.inprogress"
    local composed_requests="${request_manifest}.composed.inprogress"
    local composed_evidence="${evidence_manifest}.composed.inprogress"
    local -a compose_command=(
      python3 -m native_vnni_dispatch.profiler_evidence
      compose-evidence
      --source-observation "${witness_table}"
      --source-requests "${request_manifest}"
      --source-evidence "${evidence_manifest}"
    )
    local source_index
    for ((source_index = 0;
         source_index < ${#retained_delta_requests[@]};
         ++source_index)); do
      compose_command+=(
        --source-observation "${retained_delta_witnesses[source_index]}"
        --source-requests "${retained_delta_requests[source_index]}"
        --source-evidence "${retained_delta_evidence[source_index]}"
      )
    done
    if (( profile_active_transaction )); then
      compose_command+=(
        --source-observation "${delta_witnesses}"
        --source-requests "${active_request_manifest}"
        --source-evidence "${active_evidence_manifest}"
      )
    fi
    compose_command+=(
      --output-observation "${composed_witnesses}"
      --output-requests "${composed_requests}"
      --output-evidence "${composed_evidence}"
    )
    run_cmd "PYTHONPATH=${profiler_python_root}" "${compose_command[@]}"
    mv "${composed_witnesses}" "${witness_table}"
    mv "${composed_requests}" "${request_manifest}"
    mv "${composed_evidence}" "${evidence_manifest}"
  else
    materialize_profiler_observation_witnesses collect
  fi
  run_cmd \
    "PYTHONPATH=${profiler_python_root}" \
    python3 -m native_vnni_dispatch.profiler_evidence \
    export-features \
    --observation "${witness_table}" \
    --requests "${request_manifest}" \
    --evidence "${evidence_manifest}" \
    --output "${feature_table}"
}

reuse_identical_backend_profiler_evidence() {
  local source_requests="$1"
  local source_evidence="$2"
  local source_features="$3"
  local final_requests="$4"
  local final_evidence="$5"
  local final_features="$6"
  local source_witnesses="${source_features%_features.csv}_observation_witnesses.csv"
  local final_witnesses="${final_features%_features.csv}_observation_witnesses.csv"

  # Certification has already consumed this profiler transaction. Re-profiled
  # "final" evidence would measure the same physical launches after the policy
  # decision and would therefore add no independent signal. The certifier has
  # already authenticated the exact observation, request, evidence, witness,
  # and feature transaction supplied here. Publish content-identical hard-link
  # aliases atomically so corpus bundling retains its explicit final-artifact
  # names without duplicating gigabytes or re-hashing every observation. A
  # genuinely enlarged candidate surface must be profiled before certification
  # instead of being silently paid for afterwards.
  if (( ! dry_run )); then
    local source_path
    for source_path in \
        "${source_requests}" \
        "${source_evidence}" \
        "${source_witnesses}" \
        "${source_features}"; do
      if [[ ! -s "${source_path}" ]]; then
        echo "error: certified profiler transaction is incomplete: ${source_path}" >&2
        exit 2
      fi
    done
  fi

  local final_requests_inprogress="${final_requests}.inprogress"
  local final_evidence_inprogress="${final_evidence}.inprogress"
  local final_witnesses_inprogress="${final_witnesses}.inprogress"
  local final_features_inprogress="${final_features}.inprogress"
  run_cmd rm -f \
    "${final_requests_inprogress}" \
    "${final_evidence_inprogress}" \
    "${final_witnesses_inprogress}" \
    "${final_features_inprogress}"
  run_cmd ln "${source_requests}" "${final_requests_inprogress}"
  run_cmd ln "${source_evidence}" "${final_evidence_inprogress}"
  run_cmd ln "${source_witnesses}" "${final_witnesses_inprogress}"
  run_cmd ln "${source_features}" "${final_features_inprogress}"
  run_cmd mv "${final_requests_inprogress}" "${final_requests}"
  run_cmd mv "${final_evidence_inprogress}" "${final_evidence}"
  run_cmd mv "${final_witnesses_inprogress}" "${final_witnesses}"
  run_cmd mv "${final_features_inprogress}" "${final_features}"
}

compose_cpu_prefill_final_profiler_evidence() {
  # The primary transaction owns a complete candidate surface at matched
  # anchors. Later profiler additions may extend it only when composition can
  # prove the same per-surface anchor inventory. The Python hard gate rejects
  # the historical pattern where an old family and a new family were profiled
  # at unrelated geometries and then presented as comparable features.
  local -a compose_command=(
    python3 -m native_vnni_dispatch.profiler_evidence
    compose-evidence
    --source-observation "${cpu_prefill_profiler_witnesses}"
    --source-requests "${cpu_prefill_profiler_requests}"
    --source-evidence "${cpu_prefill_profiler_evidence}"
  )
  local source_index
  for ((source_index = 0;
       source_index < ${#cpu_prefill_fit_additive_profiler_requests[@]};
       ++source_index)); do
    compose_command+=(
      --source-observation
      "${cpu_prefill_fit_additive_profiler_observations[${source_index}]}"
      --source-requests
      "${cpu_prefill_fit_additive_profiler_requests[${source_index}]}"
      --source-evidence
      "${cpu_prefill_fit_additive_profiler_evidence[${source_index}]}"
    )
  done
  compose_command+=(
    --output-observation "${cpu_prefill_final_profiler_witnesses}"
    --output-requests "${cpu_prefill_final_profiler_requests}"
    --output-evidence "${cpu_prefill_final_profiler_evidence}"
  )
  run_cmd "PYTHONPATH=${profiler_python_root}" "${compose_command[@]}"
  run_cmd \
    "PYTHONPATH=${profiler_python_root}" \
    python3 -m native_vnni_dispatch.profiler_evidence \
    export-features \
    --observation "${cpu_prefill_final_profiler_witnesses}" \
    --requests "${cpu_prefill_final_profiler_requests}" \
    --evidence "${cpu_prefill_final_profiler_evidence}" \
    --output "${cpu_prefill_final_profiler_features}"
}

csv_values() {
  printf '%s\n' "$1" | tr ',' '\n' | sed '/^$/d'
}

csv_contains() {
  local values="$1"
  local needle="$2"
  [[ ",${values}," == *",${needle},"* ]]
}

first_output_line() {
  # Consume the producer's complete stream before printing its first line.  A
  # direct `producer | head -n 1` closes the pipe early; under this script's
  # `set -o pipefail`, verbose tools such as hipcc then report SIGPIPE (141) and
  # abort collection before the measurement lanes are launched.
  awk 'NR == 1 { first = $0 } END { print first }'
}

partition_csv_values() {
  local values="$1"
  local lane_count="$2"
  local output_name="$3"
  local -n output="${output_name}"
  local -a items=()
  mapfile -t items < <(csv_values "${values}")
  if (( lane_count > ${#items[@]} )); then
    echo "error: ${lane_count} measurement lanes exceed ${#items[@]} format shards" >&2
    exit 2
  fi

  output=()
  local lane
  for ((lane = 0; lane < lane_count; ++lane)); do
    output+=("")
  done
  local index
  for ((index = 0; index < ${#items[@]}; ++index)); do
    lane=$((index % lane_count))
    if [[ -n "${output[${lane}]}" ]]; then
      output[${lane}]+=","
    fi
    output[${lane}]="${output[${lane}]}${items[${index}]}"
  done
}

validate_cuda_measurement_devices() {
  if (( dry_run )); then
    return
  fi
  local -a identities=()
  mapfile -t identities < <(
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
  )
  if (( cuda_measurement_lanes > ${#identities[@]} )); then
    echo "error: --cuda-measurement-lanes=${cuda_measurement_lanes} exceeds " \
         "${#identities[@]} visible CUDA GPU(s)" >&2
    exit 2
  fi
  local lane
  for ((lane = 1; lane < cuda_measurement_lanes; ++lane)); do
    if [[ "${identities[${lane}]}" != "${identities[0]}" ]]; then
      echo "error: CUDA measurement lanes must use one homogeneous architecture; " \
           "lane 0 is ${identities[0]}, lane ${lane} is ${identities[${lane}]}" >&2
      exit 2
    fi
  done
}

validate_rocm_measurement_devices() {
  if (( dry_run )); then
    return
  fi
  local -a identities=()
  mapfile -t identities < <(
    rocm-smi --showproductname --csv 2>/dev/null |
      awk -F, '$1 ~ /^card[0-9]+$/ { print $2 "|" $3 "|" $10 }'
  )
  if (( rocm_measurement_lanes > ${#identities[@]} )); then
    echo "error: --rocm-measurement-lanes=${rocm_measurement_lanes} exceeds " \
         "${#identities[@]} visible ROCm GPU(s)" >&2
    exit 2
  fi
  local lane
  for ((lane = 1; lane < rocm_measurement_lanes; ++lane)); do
    if [[ "${identities[${lane}]}" != "${identities[0]}" ]]; then
      echo "error: ROCm measurement lanes must use one homogeneous architecture; " \
           "lane 0 is ${identities[0]}, lane ${lane} is ${identities[${lane}]}" >&2
      exit 2
    fi
  done
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

# Merge compatible evidence generations without mutating either immutable
# input.  Refinement rounds may add provenance-only columns to newly measured
# rows; retaining the union lets later rounds consume one aggregate while an
# empty cell still records that an older harness never observed that field.
combine_compatible_csvs() {
  local output="$1"
  shift
  if (( dry_run )); then
    printf 'dry-run: combine-compatible-csv %q' "${output}"
    for path in "$@"; do
      printf ' %q' "${path}"
    done
    printf '\n'
    return 0
  fi

  local path
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      echo "error: expected non-empty compatible CSV: ${path}" >&2
      exit 2
    fi
  done

  PYTHONPATH="${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.combine_compatible_csvs \
      --output "${output}" --input "$@"
}

# Launch one ordered CPU prefill inventory through the canonical socket-local
# MPMD protocol. Every caller supplies complete atomic output paths, so a
# successful batch can be published with rename(2) and safely resumed later.
# Group boundaries prevent two MPI ranks from entering different ordered M
# phases, which would otherwise deadlock the complete-round timing barrier.
cpu_prefill_batches_run=0
cpu_prefill_collection_limited=0
cpu_decode_batches_run=0
cpu_decode_collection_limited=0
run_cpu_prefill_measurement_jobs() {
  local mpi_round_sync="$1"
  local group_starts_name="$2"
  local group_ends_name="$3"
  local formats_name="$4"
  local shapes_name="$5"
  local ns_name="$6"
  local ks_name="$7"
  local ms_name="$8"
  local runtime_isas_name="$9"
  local bins_name="${10}"
  local partials_name="${11}"
  local timing_partials_name="${12}"
  local -n group_starts_ref="${group_starts_name}"
  local -n group_ends_ref="${group_ends_name}"
  local -n formats_ref="${formats_name}"
  local -n shapes_ref="${shapes_name}"
  local -n ns_ref="${ns_name}"
  local -n ks_ref="${ks_name}"
  local -n ms_ref="${ms_name}"
  local -n runtime_isas_ref="${runtime_isas_name}"
  local -n bins_ref="${bins_name}"
  local -n partials_ref="${partials_name}"
  local -n timing_partials_ref="${timing_partials_name}"
  local candidate_filter="${13:-}"
  local launch_warmup_budget_us="${14:-${cpu_prefill_warmup_budget_us}}"
  local launch_transition_warmup_budget_us="${15:-${cpu_prefill_transition_warmup_budget_us}}"
  local launch_timing_budget_us="${16:-${cpu_prefill_timing_budget_us}}"
  local partial_validation_mode="${17:-}"
  local launch_transition_warmup_latency_multiplier="${18:-${cpu_prefill_transition_warmup_latency_multiplier}}"
  local launch_minimum_iterations="${19:-${cpu_prefill_min_iters}}"
  local launch_maximum_iterations="${20:-${cpu_prefill_max_iters}}"
  local launch_stationary_minimum_iterations="${21:-${launch_minimum_iterations}}"
  local launch_warmups="${22:-${cpu_warmup}}"
  local launch_attempts="${23:-1}"
  if [[ ! "${launch_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: CPU prefill launch attempts must be a positive integer" >&2
    return 2
  fi
  local -a partial_validation_args=()
  if [[ "${partial_validation_mode}" == "candidate-expansion" ]]; then
    partial_validation_args+=(--candidate-expansion)
  elif [[ -n "${partial_validation_mode}" ]]; then
    echo "error: unknown CPU prefill partial validation mode ${partial_validation_mode}" >&2
    return 2
  fi
  local -a candidate_environment=()
  if [[ "${partial_validation_mode}" == "candidate-expansion" ]]; then
    candidate_environment+=(
      "LLAMINAR_CPU_NVNNI_PREFILL_ANCHORED_EXPANSION=1"
    )
  fi
  if [[ -n "${candidate_filter}" ]]; then
    candidate_environment+=(
      "LLAMINAR_CPU_NVNNI_PREFILL_CANDIDATES=${candidate_filter}"
    )
  fi

  local job_count="${#shapes_ref[@]}"
  if (( ${#group_starts_ref[@]} != ${#group_ends_ref[@]} )); then
    echo "error: CPU prefill MPMD group boundaries are mismatched" >&2
    return 2
  fi
  local array_length
  for array_length in \
    "${#formats_ref[@]}" "${#ns_ref[@]}" "${#ks_ref[@]}" \
    "${#ms_ref[@]}" "${#runtime_isas_ref[@]}" "${#bins_ref[@]}" \
    "${#partials_ref[@]}" "${#timing_partials_ref[@]}"; do
    if (( array_length != job_count )); then
      echo "error: CPU prefill launch inventory arrays are mismatched" >&2
      return 2
    fi
  done

  cpu_prefill_batches_run=0
  cpu_prefill_collection_limited=0
  local group_index group_start group_end batch_start batch_end job_index
  for ((group_index = 0;
        group_index < ${#group_starts_ref[@]};
        ++group_index)); do
    group_start=${group_starts_ref[group_index]}
    group_end=${group_ends_ref[group_index]}
    if (( group_start < 0 || group_end < group_start || group_end > job_count )); then
      echo "error: CPU prefill MPMD group is outside the job inventory" >&2
      return 2
    fi
    for ((batch_start = group_start;
          batch_start < group_end;
          batch_start += cpu_measurement_lanes)); do
      if (( cpu_batch_limit > 0 &&
            cpu_prefill_batches_run >= cpu_batch_limit )); then
        cpu_prefill_collection_limited=1
        return 0
      fi
      batch_end=$((batch_start + cpu_measurement_lanes))
      if (( batch_end > group_end )); then
        batch_end=${group_end}
      fi

      local inprogress_paths=()
      for ((job_index = batch_start; job_index < batch_end; ++job_index)); do
        local aggregate_inprogress="${partials_ref[job_index]}.inprogress"
        local timing_inprogress="${timing_partials_ref[job_index]}.inprogress"
        inprogress_paths+=("${aggregate_inprogress}" "${timing_inprogress}")
      done
      local launch_attempt batch_validated=0
      for ((launch_attempt = 1;
            launch_attempt <= launch_attempts;
            ++launch_attempt)); do
        # A trainer flushes after every complete M phase. Authenticate any
        # interrupted prefix under the production adapter, require both MPMD
        # ranks to own the same ordered suffix, and append only that suffix.
        # This makes a multi-hour exact-overlay shard durable at M boundaries
        # without accepting a partial candidate round as evidence.
        local batch_remaining_m=""
        local batch_append=-1
        local prefix_state_valid=1
        for ((job_index = batch_start;
              job_index < batch_end;
              ++job_index)); do
          local aggregate_inprogress="${partials_ref[job_index]}.inprogress"
          local timing_inprogress="${timing_partials_ref[job_index]}.inprogress"
          local remaining_m="${ms_ref[job_index]}"
          local append_state=0
          if (( ! dry_run )) &&
             [[ -s "${aggregate_inprogress}" &&
                -s "${timing_inprogress}" ]]; then
            if remaining_m="$(
                PYTHONPATH="${shape_manifest_python_root}" \
                  python3 -m \
                    tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                    --input "${aggregate_inprogress}" \
                    --timing-sidecar "${timing_inprogress}" \
                    --planned-m-values "${ms_ref[job_index]}" \
                    --require-append-compatible \
                    --print-missing-m-values \
                    "${partial_validation_args[@]}"
              )"; then
              append_state=1
            else
              prefix_state_valid=0
              break
            fi
          elif (( ! dry_run )) &&
               [[ -e "${aggregate_inprogress}" ||
                  -e "${timing_inprogress}" ]]; then
            prefix_state_valid=0
            break
          fi

          if [[ -z "${batch_remaining_m}" ]] && (( batch_append < 0 )); then
            batch_remaining_m="${remaining_m}"
            batch_append=${append_state}
          elif [[ "${batch_remaining_m}" != "${remaining_m}" ]] ||
               (( batch_append != append_state )); then
            prefix_state_valid=0
            break
          fi
        done
        if (( ! prefix_state_valid )); then
          printf 'Discarding mismatched or non-installable CPU prefill M-prefix before retry\n' >&2
          run_cmd rm -f "${inprogress_paths[@]}"
          batch_remaining_m="${ms_ref[batch_start]}"
          batch_append=0
          for ((job_index = batch_start + 1;
                job_index < batch_end;
                ++job_index)); do
            if [[ "${ms_ref[job_index]}" != "${batch_remaining_m}" ]]; then
              echo "error: CPU prefill MPMD batch has mismatched planned M inventories" >&2
              return 2
            fi
          done
        fi

        local launch_succeeded=0
        if [[ -z "${batch_remaining_m}" ]]; then
          # A prior invocation finished every M and was interrupted before the
          # wrapper's final atomic rename. Exact validation below publishes it
          # without launching another kernel.
          launch_succeeded=1
        else
          local mpi_command=(
            mpirun --bind-to socket --map-by socket
            --mca mpi_leave_pinned 1
            --mca btl_vader_single_copy_mechanism none
            --mca orte_allowed_exit_without_sync 1
          )
          for ((job_index = batch_start;
                job_index < batch_end;
                ++job_index)); do
            if (( job_index > batch_start )); then
              mpi_command+=(":")
            fi
            local aggregate_inprogress="${partials_ref[job_index]}.inprogress"
            local timing_inprogress="${timing_partials_ref[job_index]}.inprogress"
            mpi_command+=(
              -np 1 env
              "${candidate_environment[@]}"
              "LLAMINAR_ISA_LEVEL=${runtime_isas_ref[job_index]}"
              "LLAMINAR_CPU_NVNNI_PREFILL_FORMATS=${formats_ref[job_index]}"
              "LLAMINAR_CPU_NVNNI_PREFILL_M=${batch_remaining_m}"
              "LLAMINAR_CPU_NVNNI_PREFILL_APPEND_CSV=${batch_append}"
              "LLAMINAR_CPU_NVNNI_PREFILL_SHAPE_NAME=${shapes_ref[job_index]}"
              "LLAMINAR_CPU_NVNNI_PREFILL_N=${ns_ref[job_index]}"
              "LLAMINAR_CPU_NVNNI_PREFILL_K=${ks_ref[job_index]}"
              "LLAMINAR_CPU_NVNNI_PREFILL_MAX_CASES=${cpu_max_cases}"
              "LLAMINAR_CPU_NVNNI_PREFILL_THREADS=${cpu_threads}"
              "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP=${launch_warmups}"
              "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_BUDGET_US=${launch_warmup_budget_us}"
              "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_US=${launch_transition_warmup_budget_us}"
              "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_LATENCY_MULTIPLIER=${launch_transition_warmup_latency_multiplier}"
              "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_CEILING_US=${cpu_prefill_transition_warmup_budget_ceiling_us}"
              "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_ROUND_TIMEOUT_US=${cpu_prefill_warmup_round_timeout_us}"
              "LLAMINAR_CPU_NVNNI_PREFILL_ITERS=${cpu_iters}"
              "LLAMINAR_CPU_NVNNI_PREFILL_MIN_ITERS=${launch_minimum_iterations}"
              "LLAMINAR_CPU_NVNNI_PREFILL_STATIONARY_MIN_ITERS=${launch_stationary_minimum_iterations}"
              "LLAMINAR_CPU_NVNNI_PREFILL_MAX_ITERS=${launch_maximum_iterations}"
              "LLAMINAR_CPU_NVNNI_PREFILL_TIMING_BUDGET_US=${launch_timing_budget_us}"
              "LLAMINAR_CPU_NVNNI_PREFILL_MEDIAN_STABILITY=${cpu_prefill_median_stability}"
              "LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC=${mpi_round_sync}"
              "LLAMINAR_CPU_NVNNI_PREFILL_STRONG_CSV=${aggregate_inprogress}"
              "LLAMINAR_CPU_NVNNI_PREFILL_TIMING_CSV=${timing_inprogress}"
              "${bins_ref[job_index]}"
              "--gtest_filter=*TrainerCsv_StrongPrefill_AllFormats"
            )
          done
          if run_cmd \
              "OMP_NUM_THREADS=${cpu_threads}" \
              "OMP_PLACES=cores" \
              "OMP_PROC_BIND=close" \
              "OMP_DYNAMIC=false" \
              "OMP_NESTED=false" \
              "HWLOC_COMPONENTS=-gl,-opencl" \
              "OMPI_MCA_mpi_leave_pinned=1" \
              "OMPI_MCA_btl_vader_single_copy_mechanism=none" \
              "${mpi_command[@]}"; then
            launch_succeeded=1
          fi
        fi

        batch_validated=${launch_succeeded}
        if (( launch_succeeded )); then
          # Validate the complete MPMD batch before publishing either rank.
          # A rank-local drift excursion therefore leaves only temporary files;
          # the next attempt receives a fresh timing epoch on both sockets.
          for ((job_index = batch_start;
                job_index < batch_end;
                ++job_index)); do
            if (( dry_run )); then
              printf 'dry-run: validate installable CPU prefill partial %q %q\n' \
                "${partials_ref[job_index]}.inprogress" \
                "${timing_partials_ref[job_index]}.inprogress"
            elif ! PYTHONPATH="${shape_manifest_python_root}" \
                python3 -m \
                  tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                  --input "${partials_ref[job_index]}.inprogress" \
                  --timing-sidecar "${timing_partials_ref[job_index]}.inprogress" \
                  --expected-m-values "${ms_ref[job_index]}" \
                  "${partial_validation_args[@]}"; then
              batch_validated=0
              break
            fi
          done
        fi

        if (( batch_validated )); then
          for ((job_index = batch_start;
                job_index < batch_end;
                ++job_index)); do
            run_cmd mv "${partials_ref[job_index]}.inprogress" \
              "${partials_ref[job_index]}"
            run_cmd mv "${timing_partials_ref[job_index]}.inprogress" \
              "${timing_partials_ref[job_index]}"
          done
          break
        fi

        if (( launch_attempt < launch_attempts )); then
          printf 'Retrying non-installable CPU prefill MPMD batch from its last authenticated M-prefix (attempt %d/%d)\n' \
            "$((launch_attempt + 1))" "${launch_attempts}" >&2
        fi
      done
      if (( ! batch_validated )); then
        echo "error: CPU prefill measurement remained non-installable after ${launch_attempts} attempt(s)" >&2
        return 2
      fi
      cpu_prefill_batches_run=$((cpu_prefill_batches_run + 1))
    done
  done
}

collect_cpu_prefill_generic_refinement_round() {
  local plan_path="$1"
  local round="$2"
  local aggregate="$3"
  local timing_aggregate="$4"
  local source_fit_path="${5:-}"
  local source_observations_path="${6:-}"
  local plan_token=""
  local -a plan_source_args=()
  local -a partials=() timing_partials=()
  local -a job_formats=() job_shapes=() job_ns=() job_ks=() job_ms=()
  local -a job_runtime_isas=() job_bins=() job_partials=()
  local -a job_timing_partials=() group_starts=() group_ends=()
  local pending_m_inventory=""
  local pending_group_open=0
  local format_spec shape n k regime_name runtime_isa m_values
  local partial timing_partial
  local -a lineage_inventory=()

  # A refinement round number is not a sufficient evidence identity: a failed
  # fit may regenerate that round with different shapes or M inventories. Bind
  # resumable partial names to the authenticated plan bytes so an interrupted
  # or superseded plan can never donate stale measurements to its successor.
  if (( dry_run )); then
    plan_token="dry-run-plan"
  else
    plan_token="$(sha256sum "${plan_path}" | awk '{print substr($1, 1, 16)}')"
  fi
  if [[ -n "${source_fit_path}" || -n "${source_observations_path}" ]]; then
    if [[ -z "${source_fit_path}" || -z "${source_observations_path}" ]]; then
      echo "error: refinement source fit and observations must be supplied together" >&2
      return 2
    fi
    plan_source_args+=(
      --source-fit "${source_fit_path}"
      --source-observations "${source_observations_path}"
    )
  fi

  while IFS=$'\t' read -r \
      format_spec shape n k regime_name runtime_isa m_values; do
    if ! csv_contains "${cpu_formats}" "${format_spec}"; then
      continue
    fi
    if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
       [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
      echo "error: CPU prefill generic refinement emitted unknown ISA regime ${regime_name}" >&2
      return 2
    fi
    partial="${output_dir}/cpu_prefill.generic-refinement-r${round}.${plan_token}.${format_spec}.${shape}.${regime_name}.csv"
    timing_partial="${output_dir}/cpu_prefill.generic-refinement-r${round}.${plan_token}.${format_spec}.${shape}.${regime_name}.timing.csv"
    partials+=("${partial}")
    timing_partials+=("${timing_partial}")
    if (( resume_cpu_partials )) &&
       [[ -s "${partial}" && -s "${timing_partial}" ]]; then
      if PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m \
            tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
            --input "${partial}" \
            --timing-sidecar "${timing_partial}" \
            --expected-m-values "${m_values}"; then
        continue
      fi
      printf 'Discarding non-installable CPU prefill refinement partial before resume: %s\n' \
        "${partial}" >&2
      rm -f "${partial}" "${timing_partial}"
    fi
    if (( ! pending_group_open )) ||
       [[ "${m_values}" != "${pending_m_inventory}" ]]; then
      if (( pending_group_open )); then
        group_ends+=("${#job_shapes[@]}")
      fi
      group_starts+=("${#job_shapes[@]}")
      pending_m_inventory="${m_values}"
      pending_group_open=1
    fi
    job_formats+=("${format_spec}")
    job_shapes+=("${shape}")
    job_ns+=("${n}")
    job_ks+=("${k}")
    job_ms+=("${m_values}")
    job_runtime_isas+=("${runtime_isa}")
    job_bins+=("${regime_binary[${regime_name}]}")
    job_partials+=("${partial}")
    job_timing_partials+=("${timing_partial}")
  done < <(
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m "${cpu_prefill_generic_refinement_module}" \
        --plan "${plan_path}" --records \
        --split-manifest "${cpu_prefill_split_manifest_path}" \
        "${plan_source_args[@]}" "${cpu_prefill_route_args[@]}"
  )
  if (( pending_group_open )); then
    group_ends+=("${#job_shapes[@]}")
  fi

  run_cpu_prefill_measurement_jobs \
    1 group_starts group_ends job_formats job_shapes job_ns job_ks job_ms \
    job_runtime_isas job_bins job_partials job_timing_partials \
    "" "" "" "" "" "" "" "" "" "" \
    "${cpu_prefill_refinement_launch_attempts}" || return $?
  if (( cpu_prefill_collection_limited )); then
    printf 'CPU NativeVNNI generic refinement round %d checkpoint: completed %d pending MPMD batch(es); rerun with --skip-cpu-prefill-baseline --resume-cpu-partials to continue in %s\n' \
      "${round}" "${cpu_prefill_batches_run}" "${output_dir}"
    return 3
  fi
  combine_csvs "${aggregate}" "${partials[@]}"
  combine_csvs "${timing_aggregate}" "${timing_partials[@]}"
}

collect_cpu_prefill_development_lineage_increment() {
  local plan_path="$1"
  local aggregate="$2"
  local timing_aggregate="$3"
  local plan_token=""
  local -a lineage_source_args=(
    --source-aggregate "${cpu_prefill_lineage_source_aggregate}"
    --source-timing "${cpu_prefill_lineage_source_timing}"
    --source-split-manifest "${cpu_prefill_lineage_source_split_manifest}"
    --target-split-manifest "${cpu_prefill_split_manifest_path}"
  )
  local source_route_manifest
  for source_route_manifest in \
      "${cpu_prefill_lineage_source_route_manifests[@]}"; do
    lineage_source_args+=(
      --source-route-manifest "${source_route_manifest}"
    )
  done
  local source_refinement_plan
  for source_refinement_plan in \
      "${cpu_prefill_lineage_source_refinement_plans[@]}"; do
    lineage_source_args+=(
      --source-refinement-plan "${source_refinement_plan}"
    )
  done
  local source_refinement_split_manifest
  for source_refinement_split_manifest in \
      "${cpu_prefill_lineage_source_refinement_split_manifests[@]}"; do
    lineage_source_args+=(
      --source-refinement-split-manifest \
        "${source_refinement_split_manifest}"
    )
  done

  local -a partials=() timing_partials=()
  local -a job_formats=() job_shapes=() job_ns=() job_ks=() job_ms=()
  local -a job_runtime_isas=() job_bins=() job_partials=()
  local -a job_timing_partials=() group_starts=() group_ends=()
  local pending_m_inventory=""
  local pending_group_open=0
  local format_spec shape n k regime_name runtime_isa m_values
  local partial timing_partial

  if (( dry_run )); then
    plan_token="dry-run-plan"
    printf 'dry-run: require CPU prefill lineage thread count == %s\n' \
      "${cpu_threads}"
  else
    plan_token="$(sha256sum "${plan_path}" | awk '{print substr($1, 1, 16)}')"
    local lineage_launch_records lineage_query_status=0
    lineage_launch_records="$(
        PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m "${cpu_prefill_development_lineage_module}" \
            --plan "${plan_path}" --launch-records \
            "${lineage_source_args[@]}" "${cpu_prefill_route_args[@]}"
      )" || lineage_query_status=$?
    if (( lineage_query_status != 0 )); then
      return "${lineage_query_status}"
    fi
    mapfile -t lineage_inventory <<< "${lineage_launch_records}"
    local launch_record_kind="" lineage_thread_count=""
    IFS=$'\t' read -r launch_record_kind lineage_thread_count \
      <<< "${lineage_inventory[0]:-}"
    if [[ "${launch_record_kind}" != "thread_count" ||
          ! "${lineage_thread_count}" =~ ^[1-9][0-9]*$ ]]; then
      echo "error: CPU prefill lineage emitted an invalid launch header" >&2
      return 2
    fi
    if [[ "${lineage_thread_count}" != "${cpu_threads}" ]]; then
      echo "error: CPU prefill lineage requires ${lineage_thread_count} threads, but the measurement launcher selected ${cpu_threads}" >&2
      return 2
    fi
  fi
  while IFS=$'\t' read -r \
      format_spec shape n k regime_name runtime_isa m_values; do
    if ! csv_contains "${cpu_formats}" "${format_spec}"; then
      continue
    fi
    if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
       [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
      echo "error: CPU prefill development lineage emitted unknown ISA regime ${regime_name}" >&2
      return 2
    fi
    partial="${output_dir}/cpu_prefill.development-lineage.${plan_token}.${format_spec}.${shape}.${regime_name}.csv"
    timing_partial="${output_dir}/cpu_prefill.development-lineage.${plan_token}.${format_spec}.${shape}.${regime_name}.timing.csv"
    partials+=("${partial}")
    timing_partials+=("${timing_partial}")
    if (( resume_cpu_partials )) &&
       [[ -s "${partial}" && -s "${timing_partial}" ]]; then
      if PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m \
            tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
            --input "${partial}" \
            --timing-sidecar "${timing_partial}" \
            --expected-m-values "${m_values}"; then
        continue
      fi
      printf 'Discarding non-installable CPU prefill lineage partial before resume: %s\n' \
        "${partial}" >&2
      rm -f "${partial}" "${timing_partial}"
    fi
    if (( ! pending_group_open )) ||
       [[ "${m_values}" != "${pending_m_inventory}" ]]; then
      if (( pending_group_open )); then
        group_ends+=("${#job_shapes[@]}")
      fi
      group_starts+=("${#job_shapes[@]}")
      pending_m_inventory="${m_values}"
      pending_group_open=1
    fi
    job_formats+=("${format_spec}")
    job_shapes+=("${shape}")
    job_ns+=("${n}")
    job_ks+=("${k}")
    job_ms+=("${m_values}")
    job_runtime_isas+=("${runtime_isa}")
    job_bins+=("${regime_binary[${regime_name}]}")
    job_partials+=("${partial}")
    job_timing_partials+=("${timing_partial}")
  done < <(printf '%s\n' "${lineage_inventory[@]:1}")
  if (( pending_group_open )); then
    group_ends+=("${#job_shapes[@]}")
  fi

  run_cpu_prefill_measurement_jobs \
    1 group_starts group_ends job_formats job_shapes job_ns job_ks job_ms \
    job_runtime_isas job_bins job_partials job_timing_partials || return $?
  if (( cpu_prefill_collection_limited )); then
    printf 'CPU NativeVNNI development-lineage checkpoint: completed %d pending MPMD batch(es); rerun with --resume-cpu-partials to continue in %s\n' \
      "${cpu_prefill_batches_run}" "${output_dir}"
    return 3
  fi
  combine_csvs "${aggregate}" "${partials[@]}"
  combine_csvs "${timing_aggregate}" "${timing_partials[@]}"
}

# Add newly registered physical candidates to every cell of an immutable CPU
# prefill corpus.  The plan always co-measures the established row-grid anchor;
# the analyzer uses that anchor to normalize the new cohort onto the source
# clock without rewriting or recollecting any source candidate observation.
collect_cpu_prefill_candidate_expansion() {
  local source_aggregate="$1"
  local source_timing="$2"
  local plan_path="$3"
  local aggregate="$4"
  local timing_aggregate="$5"
  local -a create_plan_args=(
    --source-aggregate "${source_aggregate}"
    --source-timing "${source_timing}"
    --collection-build-digest "${cpu_build_id}"
    --output "${plan_path}"
  )
  if [[ -n "${cpu_prefill_candidate_expansion_candidates}" ]]; then
    create_plan_args+=(
      --candidates "${cpu_prefill_candidate_expansion_candidates}"
    )
  fi

  if (( dry_run )); then
    printf 'dry-run: authenticate candidate-expansion source %q %q\n' \
      "${source_aggregate}" "${source_timing}"
    printf 'dry-run: create CPU prefill candidate-expansion plan %q\n' \
      "${plan_path}"
  else
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m \
        tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
        --input "${source_aggregate}" \
        --timing-sidecar "${source_timing}"
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m "${cpu_prefill_candidate_expansion_module}" \
        "${create_plan_args[@]}"
  fi

  local plan_token="dry-run-plan"
  local collection_candidates="${cpu_prefill_candidate_expansion_candidates}"
  declare -A rebased_candidate_expansion_records=()
  if (( ! dry_run )); then
    plan_token="$(sha256sum "${plan_path}" | awk '{print substr($1, 1, 16)}')"
    local required_threads
    required_threads="$(
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_candidate_expansion_module}" \
          --plan "${plan_path}" \
          --source-aggregate "${source_aggregate}" \
          --source-timing "${source_timing}" \
          --collection-build-digest "${cpu_build_id}" \
          --thread-count
    )"
    if [[ "${required_threads}" != "${cpu_threads}" ]]; then
      echo "error: CPU prefill candidate expansion requires ${required_threads} threads, but the launcher selected ${cpu_threads}" >&2
      return 2
    fi
    collection_candidates="$(
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_candidate_expansion_module}" \
          --plan "${plan_path}" \
          --source-aggregate "${source_aggregate}" \
          --source-timing "${source_timing}" \
          --collection-build-digest "${cpu_build_id}" \
          --candidate-list
    )"
  elif [[ -z "${collection_candidates}" ]]; then
    collection_candidates="cpu.nvnni.prefill.row_chunk_grid.full_k,cpu.nvnni.prefill.decode_equivalent_kpart.pairwise,cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k,cpu.nvnni.prefill.two_row_pair_grid.nbc2.full_k,cpu.nvnni.prefill.two_row_pair_grid.nbc4.full_k,cpu.nvnni.prefill.two_row_pair_grid.nbc8.full_k,cpu.nvnni.prefill.two_row_pair_grid.nbc16.full_k"
  else
    collection_candidates="cpu.nvnni.prefill.row_chunk_grid.full_k,${collection_candidates}"
  fi

  if [[ -n "${cpu_prefill_candidate_expansion_resume_from}" ]]; then
    if (( dry_run )); then
      printf 'dry-run: rebase CPU prefill candidate-expansion checkpoint %q into %q using %q\n' \
        "${cpu_prefill_candidate_expansion_resume_from}" \
        "${output_dir}" "${plan_path}"
      if [[ -n "${cpu_prefill_harness_build_change_audit}" ]]; then
        printf 'dry-run: audit timing-harness-only build change: %q\n' \
          "${cpu_prefill_harness_build_change_audit}"
      fi
    else
      local -a rebase_args=(
        --source-directory "${cpu_prefill_candidate_expansion_resume_from}"
        --target-directory "${output_dir}"
        --target-plan "${plan_path}"
      )
      if [[ -n "${cpu_prefill_harness_build_change_audit}" ]]; then
        rebase_args+=(
          --harness-only-build-change-audit
          "${cpu_prefill_harness_build_change_audit}"
        )
      fi
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_candidate_expansion_rebase_module}" \
          "${rebase_args[@]}"
      # The rebase transaction has already validated both the source and the
      # atomically published target sidecars under the current adapter. Load
      # its exact record identities once so the resume loop does not parse the
      # same large timing files again, serially, immediately afterward.
      local rebase_manifest
      for rebase_manifest in \
          "${output_dir}"/cpu_prefill_candidate_expansion_rebase.*.to."${plan_token}".json; do
        [[ -f "${rebase_manifest}" ]] || continue
        local rebased_record
        while IFS= read -r rebased_record; do
          rebased_candidate_expansion_records["${rebased_record}"]=1
        done < <(
          python3 -c \
            'import json, sys; records = [item["record"] for item in json.load(open(sys.argv[1], encoding="utf-8"))["published"]]; sys.stdout.write("\n".join(records) + ("\n" if records else ""))' \
            "${rebase_manifest}"
        )
      done
    fi
  fi

  local -a partials=() timing_partials=()
  local -a job_formats=() job_shapes=() job_ns=() job_ks=() job_ms=()
  local -a job_runtime_isas=() job_bins=() job_partials=()
  local -a job_timing_partials=() group_starts=() group_ends=()
  local pending_m_inventory=""
  local pending_group_open=0
  local format_spec shape n k regime_name runtime_isa m_values
  local partial timing_partial
  while IFS=$'\t' read -r \
      format_spec shape n k regime_name runtime_isa m_values; do
    if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
       [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
      echo "error: CPU prefill candidate expansion emitted unknown ISA regime ${regime_name}" >&2
      return 2
    fi
    partial="${output_dir}/cpu_prefill.candidate-expansion.${plan_token}.${format_spec}.${shape}.${regime_name}.csv"
    timing_partial="${output_dir}/cpu_prefill.candidate-expansion.${plan_token}.${format_spec}.${shape}.${regime_name}.timing.csv"
    partials+=("${partial}")
    timing_partials+=("${timing_partial}")
    if (( resume_cpu_partials )) &&
       [[ -s "${partial}" && -s "${timing_partial}" ]]; then
      local record_key="${format_spec}/${shape}/${regime_name}"
      if [[ -n "${rebased_candidate_expansion_records[${record_key}]+present}" ]]; then
        continue
      elif PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m \
            tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
            --input "${partial}" \
            --timing-sidecar "${timing_partial}" \
            --expected-m-values "${m_values}" \
            --candidate-expansion; then
        continue
      fi
      printf 'Discarding non-installable CPU prefill candidate-expansion partial before resume: %s\n' \
        "${partial}" >&2
      rm -f "${partial}" "${timing_partial}"
    fi
    if (( ! pending_group_open )) ||
       [[ "${m_values}" != "${pending_m_inventory}" ]]; then
      if (( pending_group_open )); then
        group_ends+=("${#job_shapes[@]}")
      fi
      group_starts+=("${#job_shapes[@]}")
      pending_m_inventory="${m_values}"
      pending_group_open=1
    fi
    job_formats+=("${format_spec}")
    job_shapes+=("${shape}")
    job_ns+=("${n}")
    job_ks+=("${k}")
    job_ms+=("${m_values}")
    job_runtime_isas+=("${runtime_isa}")
    job_bins+=("${regime_binary[${regime_name}]}")
    job_partials+=("${partial}")
    job_timing_partials+=("${timing_partial}")
  done < <(
    if (( dry_run )); then
      printf 'Q4_0\t0.5B_AttnOut\t896\t896\tavx2-build.avx2-runtime\tavx2\t64\n'
    else
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_candidate_expansion_module}" \
          --plan "${plan_path}" \
          --source-aggregate "${source_aggregate}" \
          --source-timing "${source_timing}" \
          --collection-build-digest "${cpu_build_id}" \
          --records
    fi
  )
  if (( pending_group_open )); then
    group_ends+=("${#job_shapes[@]}")
  fi

  run_cpu_prefill_measurement_jobs \
    1 group_starts group_ends job_formats job_shapes job_ns job_ks job_ms \
    job_runtime_isas job_bins job_partials job_timing_partials \
    "${collection_candidates}" 0 0 \
    "${cpu_prefill_candidate_expansion_timing_budget_us}" \
    candidate-expansion 0 5 \
    "${cpu_prefill_candidate_expansion_max_iters}" 5 2 3
  if (( cpu_prefill_collection_limited )); then
    printf 'CPU NativeVNNI candidate-expansion checkpoint: completed %d pending MPMD batch(es); rerun with --resume-cpu-partials to continue in %s\n' \
      "${cpu_prefill_batches_run}" "${output_dir}"
    return 3
  fi
  combine_csvs "${aggregate}" "${partials[@]}"
  combine_csvs "${timing_aggregate}" "${timing_partials[@]}"
  if (( ! dry_run )); then
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m \
        tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
        --input "${aggregate}" \
        --timing-sidecar "${timing_aggregate}" \
        --candidate-expansion
  fi
}

cpu_shape_n() {
  if [[ -z "${native_vnni_shape_n[$1]:-}" ]]; then
    echo "error: unsupported CPU NativeVNNI verifier shape: $1" >&2
    exit 2
  fi
  printf '%s\n' "${native_vnni_shape_n[$1]}"
}

cpu_shape_k() {
  if [[ -z "${native_vnni_shape_k[$1]:-}" ]]; then
    echo "error: unsupported CPU NativeVNNI verifier shape: $1" >&2
    exit 2
  fi
  printf '%s\n' "${native_vnni_shape_k[$1]}"
}

run_cuda_measurement_lanes() {
  local phase="$1"
  local phase_m_values="$2"
  local aggregate_output="$3"
  local timing_output="$4"
  local lane_formats_name="$5"
  local lane_shapes_name="$6"
  local -n lane_formats_ref="${lane_formats_name}"
  local -n lane_shapes_ref="${lane_shapes_name}"
  local lane_count="${#lane_formats_ref[@]}"
  if (( lane_count == 0 || lane_count != ${#lane_shapes_ref[@]} )); then
    echo "error: CUDA lane format/shape assignments are empty or mismatched" >&2
    return 2
  fi

  local -a partials=()
  local -a timing_partials=()
  local -a logs=()
  local -a pids=()
  local lane
  for ((lane = 0; lane < lane_count; ++lane)); do
    local partial="${output_dir}/cuda_decode_${phase}.lane${lane}.csv"
    local timing_partial="${output_dir}/cuda_decode_${phase}.lane${lane}.timing.csv"
    local log="${output_dir}/cuda_decode_${phase}.lane${lane}.log"
    partials+=("${partial}")
    timing_partials+=("${timing_partial}")
    logs+=("${log}")
    if (( dry_run )); then
      run_cmd \
        "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
        "CUDA_VISIBLE_DEVICES=${lane}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=${lane_formats_ref[${lane}]}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=${lane_shapes_ref[${lane}]}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_M=${phase_m_values}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES=${cuda_candidates}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES=${cuda_execution_modes}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES=${cuda_max_cases}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_CSV=${partial}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
        "${cuda_sweep_bin}" \
        "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
    else
      (
        run_cmd \
          "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
          "CUDA_VISIBLE_DEVICES=${lane}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=${lane_formats_ref[${lane}]}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=${lane_shapes_ref[${lane}]}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_M=${phase_m_values}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES=${cuda_candidates}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES=${cuda_execution_modes}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES=${cuda_max_cases}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_CSV=${partial}" \
          "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
          "${cuda_sweep_bin}" \
          "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
      ) >"${log}" 2>&1 &
      pids+=("$!")
    fi
  done
  if (( ! dry_run )); then
    local failed=0
    for ((lane = 0; lane < lane_count; ++lane)); do
      if ! wait "${pids[${lane}]}"; then
        echo "error: CUDA measurement lane ${lane} failed; tail of ${logs[${lane}]}:" >&2
        tail -n 40 "${logs[${lane}]}" >&2 || true
        failed=1
      fi
    done
    if (( failed )); then
      return 2
    fi
  fi
  combine_csvs "${aggregate_output}" "${partials[@]}"
  combine_csvs "${timing_output}" "${timing_partials[@]}"
}

run_cuda_shape_sharded_phase() {
  local phase="$1"
  local phase_m_values="$2"
  local aggregate_output="$3"
  local timing_output="$4"
  local phase_format="$5"
  local phase_shapes="$6"
  local -a lane_shapes=()
  partition_csv_values \
    "${phase_shapes}" "${cuda_measurement_lanes}" lane_shapes
  local -a lane_formats=()
  local lane
  for ((lane = 0; lane < cuda_measurement_lanes; ++lane)); do
    lane_formats+=("${phase_format}")
  done
  run_cuda_measurement_lanes \
    "${phase}" "${phase_m_values}" \
    "${aggregate_output}" "${timing_output}" \
    lane_formats lane_shapes
}

run_cuda_sweep_phase() {
  local phase="$1"
  local phase_m_values="$2"
  local aggregate_output="$3"
  local timing_output="$4"
  local phase_shapes="$5"

  if (( stratified_formats )); then
    local partials=()
    local timing_partials=()
    while IFS= read -r format; do
      local partial="${output_dir}/cuda_decode_${phase}.${format}.csv"
      local timing_partial="${output_dir}/cuda_decode_${phase}.${format}.timing.csv"
      partials+=("${partial}")
      timing_partials+=("${timing_partial}")
      run_cmd \
        "LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=${format}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=${phase_shapes}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_M=${phase_m_values}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES=${cuda_candidates}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES=${cuda_execution_modes}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES=${cuda_max_cases}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_CSV=${partial}" \
        "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
        "${cuda_sweep_bin}" \
        "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
    done < <(csv_values "${cuda_formats}")
    combine_csvs "${aggregate_output}" "${partials[@]}"
    combine_csvs "${timing_output}" "${timing_partials[@]}"
    return
  fi

  if (( cuda_measurement_lanes > 1 )); then
    local -a lane_formats=()
    partition_csv_values \
      "${cuda_formats}" "${cuda_measurement_lanes}" lane_formats
    local -a lane_shapes=()
    local lane
    for ((lane = 0; lane < cuda_measurement_lanes; ++lane)); do
      lane_shapes+=("${phase_shapes}")
    done
    run_cuda_measurement_lanes \
      "${phase}" "${phase_m_values}" \
      "${aggregate_output}" "${timing_output}" \
      lane_formats lane_shapes
    return
  fi

  run_cmd \
    "LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=${cuda_formats}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=${phase_shapes}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_M=${phase_m_values}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES=${cuda_candidates}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES=${cuda_execution_modes}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES=${cuda_max_cases}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_CSV=${aggregate_output}" \
    "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV=${timing_output}" \
    "${cuda_sweep_bin}" \
    "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
}

run_rocm_sweep_phase() {
  local phase="$1"
  local phase_m_values="$2"
  local aggregate_output="$3"
  local timing_output="$4"
  local phase_shapes="$5"

  if (( stratified_formats )); then
    local partials=()
    local timing_partials=()
    while IFS= read -r format; do
      local partial="${output_dir}/rocm_decode_${phase}.${format}.csv"
      local timing_partial="${output_dir}/rocm_decode_${phase}.${format}.timing.csv"
      partials+=("${partial}")
      timing_partials+=("${timing_partial}")
      run_cmd \
        "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${format}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${phase_shapes}" \
        "LLAMINAR_ROCM_NVNNI_DECODE_M=${phase_m_values}" \
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
    combine_csvs "${aggregate_output}" "${partials[@]}"
    combine_csvs "${timing_output}" "${timing_partials[@]}"
    return
  fi

  if (( rocm_measurement_lanes > 1 )); then
    local -a lane_formats=()
    partition_csv_values \
      "${rocm_formats}" "${rocm_measurement_lanes}" lane_formats
    local -a partials=()
    local -a timing_partials=()
    local -a logs=()
    local -a pids=()
    local lane
    for ((lane = 0; lane < rocm_measurement_lanes; ++lane)); do
      local partial="${output_dir}/rocm_decode_${phase}.lane${lane}.csv"
      local timing_partial="${output_dir}/rocm_decode_${phase}.lane${lane}.timing.csv"
      local log="${output_dir}/rocm_decode_${phase}.lane${lane}.log"
      partials+=("${partial}")
      timing_partials+=("${timing_partial}")
      logs+=("${log}")
      if (( dry_run )); then
        run_cmd \
          "ROCR_VISIBLE_DEVICES=${lane}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${lane_formats[${lane}]}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${phase_shapes}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_M=${phase_m_values}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=${rocm_execution_modes}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS=${rocm_warmups}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES=${rocm_samples}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${partial}" \
          "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
          "${rocm_decode_bin}" \
          "--gtest_filter=*TrainerCsv_CodebookTagged*"
      else
        (
          run_cmd \
            "ROCR_VISIBLE_DEVICES=${lane}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${lane_formats[${lane}]}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${phase_shapes}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_M=${phase_m_values}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=${rocm_execution_modes}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS=${rocm_warmups}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES=${rocm_samples}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${partial}" \
            "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=${timing_partial}" \
            "${rocm_decode_bin}" \
            "--gtest_filter=*TrainerCsv_CodebookTagged*"
        ) >"${log}" 2>&1 &
        pids+=("$!")
      fi
    done
    if (( ! dry_run )); then
      local failed=0
      for ((lane = 0; lane < rocm_measurement_lanes; ++lane)); do
        if ! wait "${pids[${lane}]}"; then
          echo "error: ROCm measurement lane ${lane} failed; tail of ${logs[${lane}]}:" >&2
          tail -n 40 "${logs[${lane}]}" >&2 || true
          failed=1
        fi
      done
      if (( failed )); then
        return 2
      fi
    fi
    combine_csvs "${aggregate_output}" "${partials[@]}"
    combine_csvs "${timing_output}" "${timing_partials[@]}"
    return
  fi

  run_cmd \
    "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=${rocm_formats}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=${phase_shapes}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_M=${phase_m_values}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES=${rocm_max_cases}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS=${rocm_variants}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=${rocm_execution_modes}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS=${rocm_warmups}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES=${rocm_samples}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_CSV=${aggregate_output}" \
    "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=${timing_output}" \
    "${rocm_decode_bin}" \
    "--gtest_filter=*TrainerCsv_CodebookTagged*"
}

# Development-only paired timing shards accepted by the final CUDA compiler.
# This array is populated by run_cuda_paired_refinement after broad M1 evidence
# has been adapted to the common schema.
cuda_paired_evidence=()

run_cuda_paired_refinement() {
  local common_observations="$1"
  local paired_dir="${output_dir}/cuda_paired_refinement"
  mkdir -p "${paired_dir}"
  local first_iteration=0
  local retained_path retained_name retained_tag retained_index
  for retained_path in "${cuda_paired_evidence[@]}"; do
    retained_name="$(basename "${retained_path}")"
    if [[ ! "${retained_name}" =~ ^iteration-([0-9]{3})[.]csv$ ]]; then
      echo "error: CUDA paired evidence has an invalid generation name: ${retained_path}" >&2
      exit 2
    fi
    retained_tag="${BASH_REMATCH[1]}"
    retained_index=$((10#${retained_tag}))
    if (( retained_index >= first_iteration )); then
      first_iteration=$((retained_index + 1))
    fi
  done

  if (( dry_run )); then
    local dry_tag
    printf -v dry_tag '%03d' "${first_iteration}"
    local requests="${paired_dir}/iteration-${dry_tag}.requests.json"
    local report="${paired_dir}/iteration-${dry_tag}.report.json"
    local evidence="${paired_dir}/iteration-${dry_tag}.csv"
    local -a dry_planner=(
      python3 -m native_vnni_dispatch.paired_requests
      "${common_observations}"
      --output "${requests}"
      --report "${report}"
      --shape-manifest "${shape_manifest_path}"
      --development-profiler-requests "${cuda_profiler_requests}"
      --development-profiler-evidence "${cuda_profiler_evidence}"
      --fit-cache-dir "${paired_dir}/fit-cache"
    )
    local dry_evidence
    for dry_evidence in "${cuda_paired_evidence[@]}"; do
      dry_planner+=(--paired-csv "${dry_evidence}")
    done
    run_cmd "PYTHONPATH=${paired_planner_root}" "${dry_planner[@]}"
    if (( skip_sweep )); then
      # A fit-only dry run demonstrates command composition only. The real
      # transaction executes this planner and requires it to report green
      # before the frozen policy can be compiled.
      return
    fi
    run_cmd \
      "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST=${requests}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_CSV=${evidence}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
      "${cuda_sweep_bin}" \
      "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
    cuda_paired_evidence=("${evidence}")
    return
  fi

  local new_iteration iteration
  for ((new_iteration = 0; new_iteration < paired_max_iterations; ++new_iteration)); do
    iteration=$((first_iteration + new_iteration))
    local iteration_tag
    printf -v iteration_tag '%03d' "${iteration}"
    local requests="${paired_dir}/iteration-${iteration_tag}.requests.json"
    local report="${paired_dir}/iteration-${iteration_tag}.report.json"
    local evidence="${paired_dir}/iteration-${iteration_tag}.csv"
    local certificate="${paired_dir}/iteration-${iteration_tag}.certificate.json"
    local -a planner=(
      python3 -m native_vnni_dispatch.paired_requests
      "${common_observations}"
      --output "${requests}"
      --report "${report}"
      --shape-manifest "${shape_manifest_path}"
      --development-profiler-requests "${cuda_profiler_requests}"
      --development-profiler-evidence "${cuda_profiler_evidence}"
      --fit-cache-dir "${paired_dir}/fit-cache"
    )
    local path
    for path in "${cuda_paired_evidence[@]}"; do
      planner+=(--paired-csv "${path}")
    done
    run_cmd "PYTHONPATH=${paired_planner_root}" "${planner[@]}"

    local status request_count cv_miss_count conflict_count
    IFS=$'\t' read -r status request_count cv_miss_count conflict_count < <(
      python3 - "${report}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(
    payload["status"],
    payload["request_count"],
    len(payload["confirmed_cv_misses"]),
    len(payload["conflicts"]),
    sep="\t",
)
PY
    )
    printf 'CUDA paired refinement iteration=%s status=%s requests=%s cv_misses=%s conflicts=%s\n' \
      "${iteration_tag}" "${status}" "${request_count}" \
      "${cv_miss_count}" "${conflict_count}"

    case "${status}" in
      green)
        if [[ "${request_count}" != "0" ]]; then
          echo "error: green paired plan still contains requests" >&2
          exit 2
        fi
        return
        ;;
      pending)
        if [[ "${request_count}" == "0" ]]; then
          echo "error: pending paired plan contains no actionable requests" >&2
          exit 2
        fi
        ;;
      evidence_conflict|insufficient_cross_validation)
        echo "error: CUDA paired refinement stopped with ${status}; see ${report}" >&2
        exit 2
        ;;
      confirmed_failure)
        echo "error: CUDA paired refinement directly confirmed an over-budget development policy; see ${report}" >&2
        exit 2
        ;;
      *)
        echo "error: unknown CUDA paired planner status ${status}" >&2
        exit 2
        ;;
    esac

    if (( skip_sweep )); then
      echo "error: CUDA corpus fit still requires ${request_count} paired " \
           "measurement request(s); publish a new evidence generation" >&2
      exit 2
    fi

    run_cmd \
      "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST=${requests}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_CSV=${evidence}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=${cuda_warmups}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=${cuda_samples}" \
      "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=${cuda_timed_replays}" \
      "${cuda_sweep_bin}" \
      "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
    run_cmd \
      "PYTHONPATH=${paired_planner_root}" \
      python3 -m native_vnni_dispatch.paired_confirmation \
      "${evidence}" \
      --output "${certificate}"
    cuda_paired_evidence+=("${evidence}")
  done

  echo "error: CUDA paired refinement did not converge in ${paired_max_iterations} new iterations from ${first_iteration}" >&2
  exit 2
}

load_cuda_development_context() {
  local observation_csv="$1"
  local output_path="$2"
  python3 - "${observation_csv}" "${output_path}" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
fields = (
    "run_id",
    "git_revision",
    "build_id",
    "compiler_id",
    "architecture_class",
    "device_name",
    "driver_runtime",
    "serial_m1_policy_hash",
)
values = {field: set() for field in fields}
row_count = 0
with source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    missing = set(fields) - set(reader.fieldnames or ())
    if missing:
        raise SystemExit(
            f"{source}: missing CUDA observation provenance columns {sorted(missing)}"
        )
    for row in reader:
        row_count += 1
        if row.get("backend") != "cuda" or row.get("semantic_contract") != "Fast":
            raise SystemExit(f"{source}: row {row_count} is not CUDA Fast evidence")
        for field in fields:
            value = row[field]
            if not value or "\n" in value or "\r" in value:
                raise SystemExit(
                    f"{source}: row {row_count} has invalid {field} provenance"
                )
            values[field].add(value)
            if len(values[field]) > 1:
                raise SystemExit(
                    f"{source}: CUDA development {field} is not uniform"
                )
if row_count == 0:
    raise SystemExit(f"{source}: CUDA development observations are empty")
destination.write_text(
    "".join(next(iter(values[field])) + "\n" for field in fields),
    encoding="utf-8",
)
PY
}

refresh_cuda() {
  begin_backend_collection_target "CUDA"
  require_executable "${cuda_sweep_bin}"
  validate_cuda_measurement_devices
  printf 'CUDA measurement lanes: %s disjoint format shard(s)\n' \
    "${cuda_measurement_lanes}"
  local cuda_git_revision cuda_build_id cuda_compiler_id cuda_arch_class
  local cuda_device_name cuda_driver_runtime cuda_serial_policy_hash
  if (( dry_run )); then
    cuda_git_revision="dry-run-git-revision"
    cuda_build_id="sha256:dry-run-cuda-build"
    cuda_compiler_id="dry-run-nvcc"
    cuda_arch_class="dry-run-sm"
    cuda_device_name="dry-run-cuda-device"
    cuda_driver_runtime="dry-run-cuda-driver-runtime"
    cuda_serial_policy_hash="sha256:dry-run-cuda-serial-policy"
  else
    cuda_git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
    cuda_build_id="$(sha256_file_set "${cuda_sweep_bin}")"
    cuda_compiler_id="$(/usr/local/cuda/bin/nvcc --version | tail -n 1)"
    cuda_arch_class="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | first_output_line | tr -d '.')"
    cuda_device_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | first_output_line)"
    cuda_driver_runtime="driver-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | first_output_line)-cuda-$(/usr/local/cuda/bin/nvcc --version | awk '/release/ && !printed {gsub(",", "", $5); print $5; printed=1}')"
    cuda_serial_policy_hash="$(sha256_file_set \
      "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu" \
      "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc" \
      "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh")"
  fi

  # A fresh seal belongs to the currently built trainer even when an audited
  # prior development corpus is retained. Keep both physical identities: the
  # learner replays development under its original context, while certification
  # records the fresh holdout under the binary that actually measured it.
  local cuda_sealed_run_id="${timestamp}-cuda-${profile}-m1-sealed"
  local cuda_sealed_git_revision="${cuda_git_revision}"
  local cuda_sealed_build_id="${cuda_build_id}"
  local cuda_sealed_compiler_id="${cuda_compiler_id}"
  local cuda_sealed_arch_class="${cuda_arch_class}"
  local cuda_sealed_device_name="${cuda_device_name}"
  local cuda_sealed_driver_runtime="${cuda_driver_runtime}"
  local cuda_sealed_serial_policy_hash="${cuda_serial_policy_hash}"
  local cuda_development_run_id="${timestamp}-cuda-${profile}-m1-development"
  local cuda_reuse_common_source="${output_dir}/cuda_decode_m1_common_observations.reuse-source.csv"
  if (( reuse_cuda_development )); then
    local -a legacy_profiler_paths=(
      "${output_dir}/cuda_decode_m1.profiler_requests.json"
      "${output_dir}/cuda_decode_m1.profiler_evidence.json"
      "${output_dir}/cuda_decode_m1.profiler_features.csv"
    )
    local -a canonical_profiler_paths=(
      "${cuda_profiler_requests}"
      "${cuda_profiler_evidence}"
      "${cuda_profiler_features}"
    )
    local legacy_profiler_schema=""
    local canonical_profiler_schema=""
    local cuda_reuse_profiler_compatible=1
    local cuda_exact_profiler_transaction=0
    if (( ! dry_run )) && [[ -s "${legacy_profiler_paths[0]}" ]]; then
      legacy_profiler_schema="$(python3 - "${legacy_profiler_paths[0]}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("schema_version", ""))
PY
)"
      if [[ "${legacy_profiler_schema}" == "native-vnni-profiler-request-v2" ]]; then
        cuda_reuse_profiler_compatible=0
        printf 'Retaining legacy CUDA profiler v2 evidence for provenance; collecting a new exact-point transaction\n'
        if [[ -s "${canonical_profiler_paths[0]}" ]]; then
          canonical_profiler_schema="$(python3 - "${canonical_profiler_paths[0]}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("schema_version", ""))
PY
)"
          if [[ "${canonical_profiler_schema}" == \
                "native-vnni-profiler-request-v4-exact-point" ]]; then
            cuda_exact_profiler_transaction=1
            printf 'Resuming current CUDA exact-point profiler transaction alongside retained v2 provenance\n'
          fi
        fi
      fi
    fi
    local profiler_index legacy_path canonical_path
    for ((profiler_index = 0;
         profiler_index < ${#legacy_profiler_paths[@]};
         ++profiler_index)); do
      legacy_path="${legacy_profiler_paths[${profiler_index}]}"
      canonical_path="${canonical_profiler_paths[${profiler_index}]}"
      if (( dry_run )); then
        run_cmd cp --reflink=auto "${legacy_path}" "${canonical_path}"
      elif (( ! cuda_reuse_profiler_compatible )); then
        if (( cuda_exact_profiler_transaction )); then
          continue
        fi
        if [[ -s "${canonical_path}" ]]; then
          if [[ ! -s "${legacy_path}" ]] ||
             ! cmp --silent "${legacy_path}" "${canonical_path}"; then
            echo "error: incompatible canonical CUDA profiler artifact is not the retained v2 generation: ${canonical_path}" >&2
            exit 2
          fi
          rm -f "${canonical_path}"
        fi
      elif [[ -s "${canonical_path}" ]]; then
        if [[ -s "${legacy_path}" ]] &&
           ! cmp --silent "${legacy_path}" "${canonical_path}"; then
          echo "error: legacy and canonical CUDA profiler artifacts conflict: " \
               "${legacy_path} ${canonical_path}" >&2
          exit 2
        fi
      elif [[ -s "${legacy_path}" ]]; then
        cp --reflink=auto "${legacy_path}" "${canonical_path}.inprogress"
        mv "${canonical_path}.inprogress" "${canonical_path}"
      fi
    done
    local required_path
    for required_path in \
        "${cuda_m1_development_csv}" \
        "${cuda_m1_development_timing_csv}" \
        "${cuda_m1_common_csv}"; do
      if (( ! dry_run )) && [[ ! -s "${required_path}" ]]; then
        echo "error: audited CUDA development reuse is missing ${required_path}" >&2
        exit 2
      fi
    done
    if (( ! dry_run && cuda_reuse_profiler_compatible )); then
      for required_path in \
          "${cuda_profiler_requests}" \
          "${cuda_profiler_evidence}"; do
        if [[ ! -s "${required_path}" ]]; then
          echo "error: audited CUDA development reuse is missing ${required_path}" >&2
          exit 2
        fi
      done
    fi
    if (( dry_run )); then
      run_cmd load_cuda_development_context \
        "${cuda_m1_common_csv}" "${output_dir}/cuda_m1_context.fields"
      cuda_development_run_id="dry-run-retained-cuda-development"
      cuda_git_revision="dry-run-retained-git-revision"
      cuda_build_id="sha256:dry-run-retained-cuda-build"
      cuda_compiler_id="dry-run-retained-nvcc"
      cuda_arch_class="dry-run-retained-sm"
      cuda_device_name="dry-run-retained-cuda-device"
      cuda_driver_runtime="dry-run-retained-cuda-runtime"
      cuda_serial_policy_hash="sha256:dry-run-retained-cuda-serial-policy"
      run_cmd cp --reflink=auto \
        "${cuda_m1_common_csv}" "${cuda_reuse_common_source}.inprogress"
      run_cmd mv -f \
        "${cuda_reuse_common_source}.inprogress" \
        "${cuda_reuse_common_source}"
    else
      local context_fields="${output_dir}/cuda_m1_context.fields"
      local cuda_reuse_context_source="${cuda_m1_common_csv}"
      # The retained snapshot owns the original development provenance. The
      # canonical path may already contain the learner-upgraded replay when a
      # previous invocation stopped during profiler collection.
      if [[ -s "${cuda_reuse_common_source}" ]]; then
        cuda_reuse_context_source="${cuda_reuse_common_source}"
      fi
      load_cuda_development_context \
        "${cuda_reuse_context_source}" "${context_fields}"
      local -a retained_context=()
      mapfile -t retained_context < "${context_fields}"
      rm -f "${context_fields}"
      if (( ${#retained_context[@]} != 8 )); then
        echo "error: retained CUDA development context is incomplete" >&2
        exit 2
      fi
      cuda_development_run_id="${retained_context[0]}"
      cuda_git_revision="${retained_context[1]}"
      cuda_build_id="${retained_context[2]}"
      cuda_compiler_id="${retained_context[3]}"
      cuda_arch_class="${retained_context[4]}"
      cuda_device_name="${retained_context[5]}"
      cuda_driver_runtime="${retained_context[6]}"
      cuda_serial_policy_hash="${retained_context[7]}"
      if [[ ! -s "${cuda_reuse_common_source}" ]]; then
        rm -f "${cuda_reuse_common_source}.inprogress"
        cp --reflink=auto \
          "${cuda_m1_common_csv}" "${cuda_reuse_common_source}.inprogress"
        mv -f \
          "${cuda_reuse_common_source}.inprogress" \
          "${cuda_reuse_common_source}"
      fi
    fi
  fi

  if [[ "${measurement_profile}" == "production" ]]; then
    if (( partitioned_measurement )); then
      local partition_csv="${output_dir}/cuda_decode_${shape_partition}.csv"
      local partition_timing_csv="${output_dir}/cuda_decode_${shape_partition}.timing.csv"
      case "${shape_partition}" in
        fast-sealed)
          partition_csv="${cuda_m1_sealed_csv}"
          partition_timing_csv="${cuda_m1_sealed_timing_csv}"
          ;;
        verifier-development)
          partition_csv="${cuda_verifier_development_csv}"
          partition_timing_csv="${cuda_verifier_development_timing_csv}"
          ;;
        verifier-sealed)
          partition_csv="${cuda_verifier_sealed_csv}"
          partition_timing_csv="${cuda_verifier_sealed_timing_csv}"
          ;;
      esac
      run_cuda_sweep_phase \
        "${shape_partition}" "${partition_m_values}" \
        "${partition_csv}" "${partition_timing_csv}" \
        "${shapes}"
      if [[ "${shape_partition}" == "verifier-sealed" ]]; then
        combine_csvs \
          "${cuda_verifier_csv}" \
          "${cuda_verifier_development_csv}" "${cuda_verifier_sealed_csv}"
        combine_csvs \
          "${cuda_verifier_timing_csv}" \
          "${cuda_verifier_development_timing_csv}" \
          "${cuda_verifier_sealed_timing_csv}"
      fi
      finish_backend_collection_target
      return
    fi

    local verifier_m_values="${canonical_verifier_m_values}"
    if (( ! skip_sweep && ! reuse_cuda_development )); then
      run_cuda_sweep_phase \
        "m1-development-common" "1" \
        "${cuda_m1_development_common_csv}" \
        "${cuda_m1_development_common_timing_csv}" \
        "${fast_development_shapes}"
      run_cuda_shape_sharded_phase \
        "m1-development-cuda-q4-refinement" "1" \
        "${cuda_m1_development_scoped_csv}" \
        "${cuda_m1_development_scoped_timing_csv}" \
        "Q4_0" "${cuda_q4_fast_refinement_shapes}"
      combine_csvs \
        "${cuda_m1_development_csv}" \
        "${cuda_m1_development_common_csv}" \
        "${cuda_m1_development_scoped_csv}"
      combine_csvs \
        "${cuda_m1_development_timing_csv}" \
        "${cuda_m1_development_common_timing_csv}" \
        "${cuda_m1_development_scoped_timing_csv}"
    elif (( skip_sweep )); then
      local cuda_fit_input
      for cuda_fit_input in \
          "${cuda_m1_development_csv}" \
          "${cuda_m1_development_timing_csv}" \
          "${cuda_m1_sealed_csv}" \
          "${cuda_m1_sealed_timing_csv}" \
          "${cuda_verifier_csv}" \
          "${cuda_verifier_timing_csv}"; do
        if [[ ! -s "${cuda_fit_input}" ]]; then
          echo "error: CUDA fit-only corpus is missing ${cuda_fit_input}" >&2
          exit 2
        fi
      done
    fi

    # First adapt the complete broad matrix without fitting an installable
    # generic policy. The paired planner consumes this common-schema corpus and
    # may refit development CV several times before any sealed data is opened.
    run_cmd python3 "${cuda_generator}" \
      --input "${cuda_m1_development_csv}" \
      --timing-sidecar "${cuda_m1_development_timing_csv}" \
      --output "${cuda_m1_provisional_inc}" \
      --common-observations "${cuda_m1_common_csv}" \
      --profile production \
      --run-id "${cuda_development_run_id}" \
      --git-revision "${cuda_git_revision}" \
      --build-id "${cuda_build_id}" \
      --compiler-id "${cuda_compiler_id}" \
      --architecture-class "${cuda_arch_class}" \
      --device-name "${cuda_device_name}" \
      --driver-runtime "${cuda_driver_runtime}" \
      --serial-m1-policy-hash "${cuda_serial_policy_hash}" \
      --exact-only \
      --require-fast-m1-complete

    collect_backend_profiler_evidence \
      cuda "${cuda_m1_common_csv}" "${cuda_sweep_bin}" \
      "${cuda_profiler_requests}" "${cuda_profiler_evidence}" \
      "${cuda_profiler_features}" "${cuda_profiler_raw}"

    if (( reuse_cuda_development )); then
      if (( dry_run )); then
        run_cmd cmp --silent \
          "${cuda_reuse_common_source}" "${cuda_m1_common_csv}"
        run_cmd \
          "PYTHONPATH=${profiler_python_root}" \
          python3 -m \
            native_vnni_dispatch.common_observation_migration \
          --retained "${cuda_reuse_common_source}" \
          --replayed "${cuda_m1_common_csv}" \
          --require-learner-transition
        run_cmd rm -f "${cuda_reuse_common_source}"
      elif cmp --silent \
          "${cuda_reuse_common_source}" "${cuda_m1_common_csv}"; then
        printf '%s\n' \
          "Authenticated byte-identical already-current CUDA common-observation replay"
        rm -f "${cuda_reuse_common_source}"
      else
        PYTHONPATH="${profiler_python_root}" \
          python3 -m native_vnni_dispatch.common_observation_migration \
            --retained "${cuda_reuse_common_source}" \
            --replayed "${cuda_m1_common_csv}" \
            --require-learner-transition
        rm -f "${cuda_reuse_common_source}"
      fi
    fi

    if (( skip_sweep || reuse_cuda_development )); then
      mapfile -t cuda_paired_evidence < <(
        find "${output_dir}/cuda_paired_refinement" -maxdepth 1 \
          -type f -name 'iteration-*.csv' -print 2>/dev/null | sort
      )
    fi
    run_cuda_paired_refinement "${cuda_m1_common_csv}"

    local -a cuda_m1_freeze=(
      python3 "${cuda_generator}"
      --input "${cuda_m1_development_csv}"
      --timing-sidecar "${cuda_m1_development_timing_csv}"
      --output "${cuda_m1_frozen_inc}"
      --summary "${cuda_m1_summary}"
      --common-observations "${cuda_m1_common_csv}"
      --policy-json "${cuda_m1_frozen_policy_json}"
      --profile production
      --run-id "${cuda_development_run_id}"
      --git-revision "${cuda_git_revision}"
      --build-id "${cuda_build_id}"
      --compiler-id "${cuda_compiler_id}"
      --architecture-class "${cuda_arch_class}"
      --device-name "${cuda_device_name}"
      --driver-runtime "${cuda_driver_runtime}"
      --serial-m1-policy-hash "${cuda_serial_policy_hash}"
      --freeze-generic
      --shape-manifest "${shape_manifest_path}"
      --measurement-plan "${gpu_measurement_plan_path}"
      --development-profiler-requests "${cuda_profiler_requests}"
      --development-profiler-evidence "${cuda_profiler_evidence}"
      --require-fast-m1-complete
    )
    if (( reuse_cuda_development )); then
      cuda_m1_freeze+=(
        --development-build-change-audit
        "${cuda_development_build_change_audit}"
      )
    fi
    local paired_path
    for paired_path in "${cuda_paired_evidence[@]}"; do
      cuda_m1_freeze+=(--paired-development-csv "${paired_path}")
    done
    run_cmd "${cuda_m1_freeze[@]}"

    # The sealed aggregate does not exist until the development-only policy has
    # been atomically published above. The certification process recomputes and
    # byte-validates that frozen artifact before it opens this sealed evidence.
    if (( ! skip_sweep )); then
      run_cuda_sweep_phase \
        "m1-sealed" "1" \
        "${cuda_m1_sealed_csv}" \
        "${cuda_m1_sealed_timing_csv}" \
        "${fast_sealed_shapes}"
    fi

    local -a cuda_m1_certify=(
      python3 "${cuda_generator}"
      --development-input "${cuda_m1_development_csv}"
      --development-timing-sidecar "${cuda_m1_development_timing_csv}"
      --sealed-input "${cuda_m1_sealed_csv}"
      --sealed-timing-sidecar "${cuda_m1_sealed_timing_csv}"
      --frozen-policy-json "${cuda_m1_frozen_policy_json}"
      --output "${cuda_m1_inc}"
      --summary "${cuda_m1_summary}"
      --common-observations "${cuda_m1_common_csv}"
      --policy-json "${cuda_m1_policy_json}"
      --profile production
      --run-id "${cuda_development_run_id}"
      --git-revision "${cuda_git_revision}"
      --build-id "${cuda_build_id}"
      --compiler-id "${cuda_compiler_id}"
      --architecture-class "${cuda_arch_class}"
      --device-name "${cuda_device_name}"
      --driver-runtime "${cuda_driver_runtime}"
      --serial-m1-policy-hash "${cuda_serial_policy_hash}"
      --certify-generic
      --shape-manifest "${shape_manifest_path}"
      --measurement-plan "${gpu_measurement_plan_path}"
      --development-profiler-requests "${cuda_profiler_requests}"
      --development-profiler-evidence "${cuda_profiler_evidence}"
      --require-fast-m1-complete
    )
    if (( reuse_cuda_development )); then
      cuda_m1_certify+=(
        --development-build-change-audit
        "${cuda_development_build_change_audit}"
        --sealed-run-id "${cuda_sealed_run_id}"
        --sealed-git-revision "${cuda_sealed_git_revision}"
        --sealed-build-id "${cuda_sealed_build_id}"
        --sealed-compiler-id "${cuda_sealed_compiler_id}"
        --sealed-architecture-class "${cuda_sealed_arch_class}"
        --sealed-device-name "${cuda_sealed_device_name}"
        --sealed-driver-runtime "${cuda_sealed_driver_runtime}"
        --sealed-serial-m1-policy-hash
        "${cuda_sealed_serial_policy_hash}"
      )
    fi
    for paired_path in "${cuda_paired_evidence[@]}"; do
      cuda_m1_certify+=(--paired-development-csv "${paired_path}")
    done
    run_cmd "${cuda_m1_certify[@]}"
    run_cmd python3 "${dispatch_validator}" "${cuda_m1_inc}"
    run_cmd \
      "PYTHONPATH=${shape_manifest_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${cuda_m1_policy_json}" \
      --include "${cuda_m1_inc}"
    combine_csvs \
      "${cuda_m1_csv}" \
      "${cuda_m1_development_csv}" "${cuda_m1_sealed_csv}"
    combine_csvs \
      "${cuda_m1_timing_csv}" \
      "${cuda_m1_development_timing_csv}" "${cuda_m1_sealed_timing_csv}"

    local cuda_source_include="${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"
    local cuda_source_backup="${output_dir}/cuda_dispatch_source_before_stage.inc"
    local staged_build_id staged_serial_policy_hash
    if (( dry_run )); then
      run_cmd cp "${cuda_source_include}" "${cuda_source_backup}"
      run_cmd cp "${cuda_m1_inc}" "${cuda_source_include}"
      run_cmd cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_cuda_native_vnni_decode_trainer
      staged_build_id="sha256:dry-run-staged-cuda-build"
      staged_serial_policy_hash="sha256:dry-run-staged-cuda-serial-policy"
    else
      cp "${cuda_source_include}" "${cuda_source_backup}"
      local restore_command
      printf -v restore_command 'cp %q %q' \
        "${cuda_source_backup}" "${cuda_source_include}"
      trap "${restore_command}" EXIT
      cp "${cuda_m1_inc}" "${cuda_source_include}"
      cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_cuda_native_vnni_decode_trainer
      staged_build_id="$(sha256_file_set "${cuda_sweep_bin}")"
      staged_serial_policy_hash="$(sha256_file_set \
        "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu" \
        "${cuda_source_include}" \
        "${repo_root}/src/v2/kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh")"
    fi

    if (( ! skip_sweep )); then
      run_cuda_sweep_phase \
        "verifier-development" "${verifier_m_values}" \
        "${cuda_verifier_development_csv}" \
        "${cuda_verifier_development_timing_csv}" \
        "${verifier_development_shapes}"
      run_cuda_sweep_phase \
        "verifier-sealed" "${verifier_m_values}" \
        "${cuda_verifier_sealed_csv}" \
        "${cuda_verifier_sealed_timing_csv}" \
        "${verifier_sealed_shapes}"
      combine_csvs \
        "${cuda_verifier_csv}" \
        "${cuda_verifier_development_csv}" "${cuda_verifier_sealed_csv}"
      combine_csvs \
        "${cuda_verifier_timing_csv}" \
        "${cuda_verifier_development_timing_csv}" \
        "${cuda_verifier_sealed_timing_csv}"
    fi

    local -a cuda_final_compile=(
      python3 "${cuda_generator}"
      --verifier-input "${cuda_verifier_csv}"
      --verifier-timing-sidecar "${cuda_verifier_timing_csv}"
      --build-id "${staged_build_id}"
      --serial-m1-policy-hash "${staged_serial_policy_hash}"
      --output "${cuda_inc}"
      --common-observations "${cuda_common_csv}"
      --profile production
      --run-id "${timestamp}-cuda-${profile}-final"
      --git-revision "${cuda_sealed_git_revision}"
      --compiler-id "${cuda_sealed_compiler_id}"
      --architecture-class "${cuda_sealed_arch_class}"
      --device-name "${cuda_sealed_device_name}"
      --driver-runtime "${cuda_sealed_driver_runtime}"
      --shape-manifest "${shape_manifest_path}"
      --measurement-plan "${gpu_measurement_plan_path}"
      --certified-m1-include "${cuda_m1_inc}"
      --certified-m1-policy-json "${cuda_m1_policy_json}"
      --require-complete
    )
    run_cmd "${cuda_final_compile[@]}"
    run_cmd python3 "${dispatch_validator}" "${cuda_inc}"
    run_cmd cp "${cuda_m1_policy_json}" "${cuda_policy_json}"
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${cuda_m1_policy_json}" \
      --include "${cuda_m1_inc}"
    collect_backend_profiler_evidence \
      cuda "${cuda_common_csv}" "${cuda_sweep_bin}" \
      "${cuda_final_profiler_requests}" "${cuda_final_profiler_evidence}" \
      "${cuda_final_profiler_features}" "${cuda_final_profiler_raw}"

    if (( dry_run )); then
      if (( install )); then
        run_cmd cp "${cuda_inc}" "${cuda_source_include}"
      else
        run_cmd cp "${cuda_source_backup}" "${cuda_source_include}"
      fi
      finish_backend_collection_target
      return
    fi

    if (( install )); then
      cp "${cuda_inc}" "${cuda_source_include}.inprogress"
      mv "${cuda_source_include}.inprogress" "${cuda_source_include}"
    else
      cp "${cuda_source_backup}" "${cuda_source_include}"
      cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_cuda_native_vnni_decode_trainer
    fi
    rm -f "${cuda_source_backup}"
    trap - EXIT
    finish_backend_collection_target
    return
  fi

  if (( ! skip_sweep )); then
    run_cuda_sweep_phase \
      "sweep" "${m_values}" "${cuda_csv}" "${cuda_timing_csv}" "${shapes}"
  fi

  local -a cuda_generator_args=(
    python3 "${cuda_generator}"
    --input "${cuda_csv}" \
    --timing-sidecar "${cuda_timing_csv}" \
    --output "${cuda_inc}" \
    --summary "${cuda_summary}" \
    --common-observations "${cuda_common_csv}" \
    --profile "${measurement_profile}" \
    --run-id "${timestamp}-cuda-${profile}" \
    --git-revision "${cuda_git_revision}" \
    --build-id "${cuda_build_id}" \
    --compiler-id "${cuda_compiler_id}" \
    --architecture-class "${cuda_arch_class}" \
    --device-name "${cuda_device_name}" \
    --driver-runtime "${cuda_driver_runtime}" \
    --serial-m1-policy-hash "${cuda_serial_policy_hash}" \
    --exact-only
  )
  if [[ "${measurement_profile}" == "partial-production" ]]; then
    cuda_generator_args+=(--require-complete)
  fi
  run_cmd "${cuda_generator_args[@]}"
  run_cmd python3 "${dispatch_validator}" "${cuda_inc}"
  collect_backend_profiler_evidence \
    cuda "${cuda_common_csv}" "${cuda_sweep_bin}" \
    "${cuda_profiler_requests}" "${cuda_profiler_evidence}" \
    "${cuda_profiler_features}" "${cuda_profiler_raw}"
  finish_backend_collection_target
}

refresh_rocm() {
  begin_backend_collection_target "ROCm"
  require_executable "${rocm_decode_bin}"
  validate_rocm_measurement_devices
  printf 'ROCm measurement lanes: %s disjoint format shard(s)\n' \
    "${rocm_measurement_lanes}"
  local rocm_git_revision rocm_build_id rocm_compiler_id rocm_arch_class
  local rocm_device_name rocm_driver_runtime rocm_serial_policy_hash
  if (( dry_run )); then
    rocm_git_revision="dry-run-git-revision"
    rocm_build_id="sha256:dry-run-rocm-build"
    rocm_compiler_id="dry-run-hipcc"
    rocm_arch_class="dry-run-gfx"
    rocm_device_name="dry-run-rocm-device"
    rocm_driver_runtime="dry-run-rocm-driver-runtime"
    rocm_serial_policy_hash="sha256:dry-run-rocm-serial-policy"
  else
    rocm_git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
    rocm_build_id="$(sha256_file_set "${rocm_decode_bin}")"
    rocm_compiler_id="$(/opt/rocm/bin/hipcc --version | first_output_line)"
    rocm_arch_class="$(rocminfo | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
    rocm_device_name="$(detect_rocm_device_name)"
    rocm_driver_runtime="linux-$(uname -r)-rocm-$(/opt/rocm/bin/hipconfig --version | first_output_line)"
    rocm_serial_policy_hash="$(sha256_file_set \
      "${repo_root}/src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip" \
      "${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc")"
  fi
  local rocm_sweep_m_values="${m_values}"
  if (( partitioned_measurement )); then
    rocm_sweep_m_values="${partition_m_values}"
  fi
  if (( ! skip_sweep )); then
    if [[ "${measurement_profile}" == "production" ]] &&
       (( ! partitioned_measurement )); then
      # Fast M=1 and grouped verifier rows are distinct semantic contracts.
      # Surface-null refinement shapes must therefore never be multiplied over
      # the verifier depth inventory merely because both contracts share one
      # ROCm trainer binary.
      run_rocm_sweep_phase \
        "fast-development" "1" \
        "${rocm_fast_development_csv}" \
        "${rocm_fast_development_timing_csv}" \
        "${fast_development_shapes}"
    else
      run_rocm_sweep_phase \
        "sweep" "${rocm_sweep_m_values}" \
        "${rocm_csv}" "${rocm_timing_csv}" "${shapes}"
    fi
  fi

  if (( partitioned_measurement )); then
    finish_backend_collection_target
    return
  fi

  if [[ "${measurement_profile}" == "production" ]]; then
    local rocm_run_id="${timestamp}-rocm-${profile}-development"
    run_cmd python3 "${rocm_generator}" \
      --input "${rocm_fast_development_csv}" \
      --timing-sidecar "${rocm_fast_development_timing_csv}" \
      --output "${rocm_provisional_inc}" \
      --common-observations "${rocm_fast_common_csv}" \
      --adapt-only \
      --profile production \
      --run-id "${rocm_run_id}" \
      --git-revision "${rocm_git_revision}" \
      --build-id "${rocm_build_id}" \
      --compiler-id "${rocm_compiler_id}" \
      --architecture-class "${rocm_arch_class}" \
      --device-name "${rocm_device_name}" \
      --driver-runtime "${rocm_driver_runtime}" \
      --serial-m1-policy-hash "${rocm_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --measurement-plan "${gpu_measurement_plan_path}"
    collect_backend_profiler_evidence \
      rocm "${rocm_fast_common_csv}" "${rocm_decode_bin}" \
      "${rocm_profiler_requests}" "${rocm_profiler_evidence}" \
      "${rocm_profiler_features}" "${rocm_profiler_raw}"

    run_cmd python3 "${rocm_generator}" \
      --input "${rocm_fast_development_csv}" \
      --timing-sidecar "${rocm_fast_development_timing_csv}" \
      --output "${rocm_frozen_inc}" \
      --summary "${rocm_summary}" \
      --common-observations "${rocm_fast_common_csv}" \
      --policy-json "${rocm_frozen_policy_json}" \
      --development-profiler-requests "${rocm_profiler_requests}" \
      --development-profiler-evidence "${rocm_profiler_evidence}" \
      --freeze-generic \
      --profile production \
      --run-id "${rocm_run_id}" \
      --git-revision "${rocm_git_revision}" \
      --build-id "${rocm_build_id}" \
      --compiler-id "${rocm_compiler_id}" \
      --architecture-class "${rocm_arch_class}" \
      --device-name "${rocm_device_name}" \
      --driver-runtime "${rocm_driver_runtime}" \
      --serial-m1-policy-hash "${rocm_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --measurement-plan "${gpu_measurement_plan_path}"

    if (( ! skip_sweep )); then
      run_rocm_sweep_phase \
        "fast-sealed" "1" \
        "${rocm_fast_sealed_csv}" "${rocm_fast_sealed_timing_csv}" \
        "${fast_sealed_shapes}"
    else
      local rocm_fit_input
      for rocm_fit_input in \
          "${rocm_fast_development_csv}" \
          "${rocm_fast_development_timing_csv}" \
          "${rocm_fast_sealed_csv}" \
          "${rocm_fast_sealed_timing_csv}" \
          "${rocm_verifier_csv}" \
          "${rocm_verifier_timing_csv}"; do
        if [[ ! -s "${rocm_fit_input}" ]]; then
          echo "error: ROCm fit-only corpus is missing ${rocm_fit_input}" >&2
          exit 2
        fi
      done
    fi
    run_cmd python3 "${rocm_generator}" \
      --development-input "${rocm_fast_development_csv}" \
      --development-timing-sidecar "${rocm_fast_development_timing_csv}" \
      --sealed-input "${rocm_fast_sealed_csv}" \
      --sealed-timing-sidecar "${rocm_fast_sealed_timing_csv}" \
      --frozen-policy-json "${rocm_frozen_policy_json}" \
      --policy-json "${rocm_policy_json}" \
      --development-profiler-requests "${rocm_profiler_requests}" \
      --development-profiler-evidence "${rocm_profiler_evidence}" \
      --output "${rocm_inc}" \
      --summary "${rocm_summary}" \
      --common-observations "${rocm_fast_common_csv}" \
      --certify-generic \
      --profile production \
      --run-id "${rocm_run_id}" \
      --git-revision "${rocm_git_revision}" \
      --build-id "${rocm_build_id}" \
      --compiler-id "${rocm_compiler_id}" \
      --architecture-class "${rocm_arch_class}" \
      --device-name "${rocm_device_name}" \
      --driver-runtime "${rocm_driver_runtime}" \
      --serial-m1-policy-hash "${rocm_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --measurement-plan "${gpu_measurement_plan_path}"
    run_cmd python3 "${dispatch_validator}" "${rocm_inc}"
    run_cmd \
      "PYTHONPATH=${profiler_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${rocm_policy_json}" \
      --include "${rocm_inc}"

    local rocm_source_include="${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
    local rocm_source_backup="${output_dir}/rocm_dispatch_source_before_stage.inc"
    local rocm_staged_build_id rocm_staged_serial_policy_hash
    if (( dry_run )); then
      run_cmd cp "${rocm_source_include}" "${rocm_source_backup}"
      run_cmd cp "${rocm_inc}" "${rocm_source_include}"
      run_cmd cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_native_vnni_throughput
      rocm_staged_build_id="sha256:dry-run-staged-rocm-build"
      rocm_staged_serial_policy_hash="sha256:dry-run-staged-rocm-serial-policy"
    else
      cp "${rocm_source_include}" "${rocm_source_backup}"
      local rocm_restore_command
      printf -v rocm_restore_command 'cp %q %q' \
        "${rocm_source_backup}" "${rocm_source_include}"
      trap "${rocm_restore_command}" EXIT
      cp "${rocm_inc}" "${rocm_source_include}"
      cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_native_vnni_throughput
      rocm_staged_build_id="$(sha256_file_set "${rocm_decode_bin}")"
      rocm_staged_serial_policy_hash="$(sha256_file_set \
        "${repo_root}/src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip" \
        "${rocm_source_include}")"
    fi

    if (( ! skip_sweep )); then
      run_rocm_sweep_phase \
        "verifier" "${canonical_verifier_m_values}" \
        "${rocm_verifier_csv}" "${rocm_verifier_timing_csv}" \
        "${verifier_all_shapes}"
    fi
    run_cmd python3 "${rocm_generator}" \
      --input "${rocm_verifier_csv}" \
      --timing-sidecar "${rocm_verifier_timing_csv}" \
      --output "${rocm_inc}" \
      --common-observations "${rocm_verifier_common_csv}" \
      --certified-policy-include "${rocm_source_include}" \
      --certified-policy-json "${rocm_policy_json}" \
      --profile production \
      --run-id "${timestamp}-rocm-${profile}-verifier" \
      --git-revision "${rocm_git_revision}" \
      --build-id "${rocm_staged_build_id}" \
      --compiler-id "${rocm_compiler_id}" \
      --architecture-class "${rocm_arch_class}" \
      --device-name "${rocm_device_name}" \
      --driver-runtime "${rocm_driver_runtime}" \
      --serial-m1-policy-hash "${rocm_staged_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --measurement-plan "${gpu_measurement_plan_path}"
    run_cmd cmp --silent "${rocm_source_include}" "${rocm_inc}"
    combine_csvs \
      "${rocm_fast_csv}" \
      "${rocm_fast_development_csv}" "${rocm_fast_sealed_csv}"
    combine_csvs "${rocm_csv}" "${rocm_fast_csv}" "${rocm_verifier_csv}"
    combine_csvs \
      "${rocm_fast_timing_csv}" \
      "${rocm_fast_development_timing_csv}" \
      "${rocm_fast_sealed_timing_csv}"
    combine_csvs \
      "${rocm_timing_csv}" \
      "${rocm_fast_timing_csv}" "${rocm_verifier_timing_csv}"
    combine_csvs \
      "${rocm_common_csv}" \
      "${rocm_fast_common_csv}" "${rocm_verifier_common_csv}"
    collect_backend_profiler_evidence \
      rocm "${rocm_common_csv}" "${rocm_decode_bin}" \
      "${rocm_final_profiler_requests}" "${rocm_final_profiler_evidence}" \
      "${rocm_final_profiler_features}" "${rocm_final_profiler_raw}"

    if (( dry_run )); then
      if (( install )); then
        run_cmd cp "${rocm_inc}" "${rocm_source_include}"
      else
        run_cmd cp "${rocm_source_backup}" "${rocm_source_include}"
      fi
      finish_backend_collection_target
      return
    fi
    if (( install )); then
      cp "${rocm_inc}" "${rocm_source_include}.inprogress"
      mv "${rocm_source_include}.inprogress" "${rocm_source_include}"
    else
      cp "${rocm_source_backup}" "${rocm_source_include}"
      cmake --build "${repo_root}/build_v2_release" --parallel \
        --target v2_perf_native_vnni_throughput
    fi
    rm -f "${rocm_source_backup}"
    trap - EXIT
    finish_backend_collection_target
    return
  fi

  local -a rocm_generator_args=(
    python3 "${rocm_generator}"
    --input "${rocm_csv}"
    --timing-sidecar "${rocm_timing_csv}"
    --output "${rocm_inc}"
    --summary "${rocm_summary}"
    --common-observations "${rocm_common_csv}"
    --profile "${measurement_profile}"
    --run-id "${timestamp}-rocm-${profile}"
    --git-revision "${rocm_git_revision}"
    --build-id "${rocm_build_id}"
    --compiler-id "${rocm_compiler_id}"
    --architecture-class "${rocm_arch_class}"
    --device-name "${rocm_device_name}"
    --driver-runtime "${rocm_driver_runtime}"
    --serial-m1-policy-hash "${rocm_serial_policy_hash}"
    --shape-manifest "${shape_manifest_path}"
    --measurement-plan "${gpu_measurement_plan_path}"
  )
  case "${profile}" in
    qwen36-core|qwen36-lm-head|qwen36-moe)
      rocm_generator_args+=(--base-include "${rocm_base_include}")
      ;;
  esac
  if [[ "${measurement_profile}" == "partial-production" ||
        "${measurement_profile}" == "production" ]]; then
    rocm_generator_args+=(--require-complete)
  fi
  run_cmd "${rocm_generator_args[@]}"

  run_cmd python3 "${dispatch_validator}" "${rocm_inc}"
  collect_backend_profiler_evidence \
    rocm "${rocm_common_csv}" "${rocm_decode_bin}" \
    "${rocm_profiler_requests}" "${rocm_profiler_evidence}" \
    "${rocm_profiler_features}" "${rocm_profiler_raw}"

  if (( install )); then
    run_cmd cp "${rocm_inc}" "${repo_root}/src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
  fi
  finish_backend_collection_target
}

rebuild_cpu_native_vnni_trainer() {
  local trainer_binary="$1"
  local build_root
  build_root="$(dirname "$(dirname "$(dirname "${trainer_binary}")")")"
  if (( dry_run )); then
    run_cmd cmake --build "${build_root}" --parallel \
      --target v2_perf_cpu_native_vnni_gemv
    return
  fi
  if [[ ! -f "${build_root}/CMakeCache.txt" ]]; then
    echo "error: cannot rebuild CPU trainer from ${trainer_binary}; " \
         "${build_root} is not a configured CMake build" >&2
    return 2
  fi
  run_cmd cmake --build "${build_root}" --parallel \
    --target v2_perf_cpu_native_vnni_gemv
}

# Development-only CPU M=1 paired timing shards accepted by both the
# incremental planner and final frozen-policy compiler. Each file is one
# architecture-homogeneous, atomically published producer transaction.
cpu_decode_paired_evidence=()
cpu_grouped_paired_evidence=()

run_cpu_decode_paired_refinement() {
  local common_observations="$1"
  local paired_dir="${output_dir}/cpu_decode_paired_refinement"
  mkdir -p "${paired_dir}"

  # Consume only complete earlier iterations. A partially collected iteration
  # is resumed from its immutable content-addressed manifests and must not
  # influence the fit that generated those manifests. This also means an
  # interruption after collection but before certification naturally advances
  # to the next fit round without rewriting iteration zero.
  cpu_decode_paired_evidence=()
  local first_iteration=0
  local candidate_iteration candidate_tag candidate_shard_dir
  for ((candidate_iteration = 0; ; ++candidate_iteration)); do
    printf -v candidate_tag '%03d' "${candidate_iteration}"
    candidate_shard_dir="${paired_dir}/iteration-${candidate_tag}.shards"
    local completed_manifests=()
    mapfile -t completed_manifests < <(
      find "${candidate_shard_dir}" -maxdepth 1 -type f \
        -name 'shard-*.requests.json' 2>/dev/null | sort
    )
    if (( ${#completed_manifests[@]} == 0 )); then
      first_iteration="${candidate_iteration}"
      break
    fi
    local candidate_complete=1
    local completed_manifest completed_evidence
    local candidate_evidence=()
    for completed_manifest in "${completed_manifests[@]}"; do
      completed_evidence="${completed_manifest%.requests.json}.csv"
      if [[ ! -s "${completed_evidence}" ]]; then
        candidate_complete=0
        break
      fi
      candidate_evidence+=("${completed_evidence}")
    done
    if (( ! candidate_complete )); then
      first_iteration="${candidate_iteration}"
      break
    fi
    cpu_decode_paired_evidence+=("${candidate_evidence[@]}")
    first_iteration=$((candidate_iteration + 1))
  done

  local iteration
  local iteration_limit=$((first_iteration + paired_max_iterations))
  for ((iteration = first_iteration;
        iteration < iteration_limit;
        ++iteration)); do
    local iteration_tag
    printf -v iteration_tag '%03d' "${iteration}"
    local requests="${paired_dir}/iteration-${iteration_tag}.requests.json"
    local report="${paired_dir}/iteration-${iteration_tag}.report.json"
    local shard_dir="${paired_dir}/iteration-${iteration_tag}.shards"
    local certificate="${paired_dir}/iteration-${iteration_tag}.certificate.json"
    local planner=(
      python3 -m native_vnni_dispatch.paired_requests
      "${common_observations}"
      --surface decode-m1
      --output "${requests}"
      --report "${report}"
      --request-shard-dir "${shard_dir}"
      --max-requests-per-shard 16
      --shape-manifest "${shape_manifest_path}"
      --fit-cache-dir "${cpu_decode_fit_cache_dir}"
    )
    local burned_index
    for burned_index in "${!cpu_decode_burned_sealed_plans[@]}"; do
      planner+=(
        --burned-sealed-plan-json
        "${cpu_decode_burned_sealed_plans[burned_index]}"
        --burned-sealed-paired-dir
        "${cpu_decode_burned_sealed_paired_dirs[burned_index]}"
      )
    done

    # The first refinement batch can be recovered directly from a complete
    # prior fit generation. This avoids replaying hours of tree search merely
    # to rediscover the held-out edges already retained in the immutable cache.
    # Once any paired ratio exists, normal content-addressed fitting is required
    # so changed domains may select new candidates and boundaries.
    local cached_cv_count=0
    if [[ -d "${cpu_decode_fit_cache_dir}/domain-cross-validation" ]]; then
      cached_cv_count="$(find \
        "${cpu_decode_fit_cache_dir}/domain-cross-validation" \
        -maxdepth 1 -type f -name '*.json' | wc -l)"
    fi
    if (( iteration == 0 &&
          ${#cpu_decode_paired_evidence[@]} == 0 &&
          ${#cpu_decode_burned_sealed_plans[@]} == 0 &&
          cached_cv_count > 0 )); then
      planner+=(
        --cached-cross-validation-dir "${cpu_decode_fit_cache_dir}"
      )
    else
      planner+=(
        --development-profiler-requests "${cpu_decode_profiler_requests}"
        --development-profiler-evidence "${cpu_decode_profiler_evidence}"
        --development-profiler-observations "${cpu_decode_profiler_witnesses}"
      )
    fi
    local paired_path
    for paired_path in "${cpu_decode_paired_evidence[@]}"; do
      planner+=(--paired-csv "${paired_path}")
    done
    run_cmd "PYTHONPATH=${paired_planner_root}" "${planner[@]}"

    if (( dry_run )); then
      printf 'dry-run: CPU paired refinement would consume %s\n' "${report}"
      return 0
    fi
    local status request_count
    status="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "${report}")"
    request_count="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["request_count"])' \
      "${report}")"
    printf 'CPU paired refinement iteration=%s status=%s requests=%s retained_shards=%s\n' \
      "${iteration}" "${status}" "${request_count}" \
      "${#cpu_decode_paired_evidence[@]}"

    case "${status}" in
      green)
        if (( request_count != 0 )); then
          echo "error: green CPU paired plan still contains requests" >&2
          return 2
        fi
        return 0
        ;;
      pending)
        if (( request_count == 0 )); then
          echo "error: pending CPU paired plan contains no actionable requests" >&2
          return 2
        fi
        ;;
      confirmed_failure|evidence_conflict|insufficient_cross_validation)
        echo "error: CPU paired refinement stopped with ${status}; see ${report}" >&2
        return 2
        ;;
      *)
        echo "error: unknown CPU paired planner status ${status}" >&2
        return 2
        ;;
    esac

    local manifests=()
    mapfile -t manifests < <(
      find "${shard_dir}" -maxdepth 1 -type f \
        -name 'shard-*.requests.json' | sort
    )
    if (( ${#manifests[@]} == 0 )); then
      echo "error: CPU paired planner emitted no request shards" >&2
      return 2
    fi

    local pending_manifests=()
    local manifest_path evidence_path
    for manifest_path in "${manifests[@]}"; do
      evidence_path="${manifest_path%.requests.json}.csv"
      if [[ -s "${evidence_path}" ]]; then
        continue
      fi
      pending_manifests+=("${manifest_path}")
    done

    local batch_start batch_end index
    for ((batch_start = 0;
          batch_start < ${#pending_manifests[@]};
          batch_start += cpu_measurement_lanes)); do
      batch_end=$((batch_start + cpu_measurement_lanes))
      if (( batch_end > ${#pending_manifests[@]} )); then
        batch_end=${#pending_manifests[@]}
      fi
      local mpi_command=(
        mpirun --bind-to socket --map-by socket
        --mca mpi_leave_pinned 1
        --mca btl_vader_single_copy_mechanism none
        --mca orte_allowed_exit_without_sync 1
      )
      local inprogress_paths=()
      local final_paths=()
      for ((index = batch_start; index < batch_end; ++index)); do
        if (( index > batch_start )); then
          mpi_command+=(":")
        fi
        manifest_path="${pending_manifests[index]}"
        evidence_path="${manifest_path%.requests.json}.csv"
        local evidence_inprogress="${evidence_path}.inprogress"
        local architecture build_isa runtime_isa trainer_bin
        architecture="$(python3 -c \
          'import json,sys; d=json.load(open(sys.argv[1])); print(d["requests"][0]["architecture_class"])' \
          "${manifest_path}")"
        case "${architecture}" in
          *'|build=AVX2|runtime=AVX2|'*)
            build_isa="AVX2"
            runtime_isa="AVX2"
            trainer_bin="${cpu_avx2_sweep_bin}"
            ;;
          *'|build=AVX512|runtime=AVX2|'*)
            build_isa="AVX512"
            runtime_isa="AVX2"
            trainer_bin="${cpu_avx512_sweep_bin}"
            ;;
          *'|build=AVX512|runtime=AVX512|'*)
            build_isa="AVX512"
            runtime_isa="AVX512"
            trainer_bin="${cpu_avx512_sweep_bin}"
            ;;
          *)
            echo "error: unsupported CPU paired architecture ${architecture}" >&2
            return 2
            ;;
        esac
        inprogress_paths+=("${evidence_inprogress}")
        final_paths+=("${evidence_path}")
        mpi_command+=(
          -np 1 env
          "LLAMINAR_PROFILING=0"
          "LLAMINAR_PERF_STATS_JSON=0"
          "LLAMINAR_PERF_STATS_CSV=0"
          "LLAMINAR_PERF_STATS_TABLE=0"
          "LLAMINAR_PERF_STATS_SUMMARY=0"
          "LLAMINAR_ISA_LEVEL=${runtime_isa}"
          "LLAMINAR_CPU_NVNNI_DECODE_WARMUP=${cpu_warmup}"
          "LLAMINAR_CPU_NVNNI_DECODE_ITERS=${cpu_iters}"
          "LLAMINAR_CPU_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST=${manifest_path}"
          "LLAMINAR_CPU_NVNNI_DECODE_PAIRED_CSV=${evidence_inprogress}"
          "${trainer_bin}"
          "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
        )
        printf 'CPU paired shard build=%s runtime=%s manifest=%s\n' \
          "${build_isa}" "${runtime_isa}" "${manifest_path}"
      done
      rm -f "${inprogress_paths[@]}"
      run_cmd \
        "OMP_NUM_THREADS=${cpu_threads}" \
        "OMP_PLACES=cores" \
        "OMP_PROC_BIND=close" \
        "OMP_DYNAMIC=false" \
        "OMP_NESTED=false" \
        "HWLOC_COMPONENTS=-gl,-opencl" \
        "OMPI_MCA_mpi_leave_pinned=1" \
        "OMPI_MCA_btl_vader_single_copy_mechanism=none" \
        "${mpi_command[@]}"
      for ((index = 0; index < ${#final_paths[@]}; ++index)); do
        mv "${inprogress_paths[index]}" "${final_paths[index]}"
      done
    done

    local completed_current_evidence=()
    for manifest_path in "${manifests[@]}"; do
      evidence_path="${manifest_path%.requests.json}.csv"
      if [[ ! -s "${evidence_path}" ]]; then
        echo "error: CPU paired collection omitted ${evidence_path}" >&2
        return 2
      fi
      completed_current_evidence+=("${evidence_path}")
    done
    cpu_decode_paired_evidence+=("${completed_current_evidence[@]}")
    run_cmd \
      "PYTHONPATH=${paired_planner_root}" \
      python3 -m native_vnni_dispatch.paired_confirmation \
      "${cpu_decode_paired_evidence[@]}" \
      --output "${certificate}"
  done

  echo "error: CPU paired refinement did not converge after " \
       "${paired_max_iterations} new iterations starting at ${first_iteration}" >&2
  return 2
}

# Execute immutable post-freeze CPU request shards without mixing ISA regimes
# or allowing partial CSVs to satisfy certification. The caller supplies the
# production surface so M=1 and grouped decode share scheduling, resume, and
# atomic-publication behavior while retaining distinct test entry points.
run_cpu_sealed_paired_shards() {
  local surface="$1"
  local shard_dir="$2"
  local output_name="$3"
  local -n output_paths="${output_name}"
  output_paths=()

  if (( dry_run )); then
    printf 'dry-run: collect CPU %s sealed paired shards from %s\n' \
      "${surface}" "${shard_dir}"
    output_paths+=("${shard_dir}/dry-run.csv")
    return 0
  fi

  local manifests=()
  mapfile -t manifests < <(
    find "${shard_dir}" -maxdepth 1 -type f \
      -name 'shard-*.requests.json' | sort
  )
  if (( ${#manifests[@]} == 0 )); then
    echo "error: CPU ${surface} sealed plan emitted no request shards" >&2
    return 2
  fi

  local pending=()
  local manifest_path evidence_path
  for manifest_path in "${manifests[@]}"; do
    evidence_path="${manifest_path%.requests.json}.csv"
    output_paths+=("${evidence_path}")
    if [[ ! -s "${evidence_path}" ]]; then
      pending+=("${manifest_path}")
    fi
  done

  local batch_start batch_end index
  for ((batch_start = 0;
        batch_start < ${#pending[@]};
        batch_start += cpu_measurement_lanes)); do
    batch_end=$((batch_start + cpu_measurement_lanes))
    if (( batch_end > ${#pending[@]} )); then
      batch_end=${#pending[@]}
    fi
    local mpi_command=(
      mpirun --bind-to socket --map-by socket
      --mca mpi_leave_pinned 1
      --mca btl_vader_single_copy_mechanism none
      --mca orte_allowed_exit_without_sync 1
    )
    local inprogress_paths=()
    local final_paths=()
    for ((index = batch_start; index < batch_end; ++index)); do
      if (( index > batch_start )); then
        mpi_command+=(":")
      fi
      manifest_path="${pending[index]}"
      evidence_path="${manifest_path%.requests.json}.csv"
      local inprogress="${evidence_path}.inprogress"
      local architecture runtime_isa trainer_bin
      architecture="$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["requests"][0]["architecture_class"])' \
        "${manifest_path}")"
      case "${architecture}" in
        *'|build=AVX2|runtime=AVX2|'*)
          runtime_isa="AVX2"
          trainer_bin="${cpu_avx2_sweep_bin}"
          ;;
        *'|build=AVX512|runtime=AVX2|'*)
          runtime_isa="AVX2"
          trainer_bin="${cpu_avx512_sweep_bin}"
          ;;
        *'|build=AVX512|runtime=AVX512|'*)
          runtime_isa="AVX512"
          trainer_bin="${cpu_avx512_sweep_bin}"
          ;;
        *)
          echo "error: unsupported CPU sealed architecture ${architecture}" >&2
          return 2
          ;;
      esac
      inprogress_paths+=("${inprogress}")
      final_paths+=("${evidence_path}")
      local paired_manifest_env paired_csv_env paired_test
      if [[ "${surface}" == "decode-m1" ]]; then
        paired_manifest_env="LLAMINAR_CPU_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST"
        paired_csv_env="LLAMINAR_CPU_NVNNI_DECODE_PAIRED_CSV"
        paired_test="TrainerCsv_StrongDecode_AllFormats"
      elif [[ "${surface}" == "grouped-decode" ]]; then
        paired_manifest_env="LLAMINAR_CPU_NVNNI_VERIFIER_PAIRED_REQUEST_MANIFEST"
        paired_csv_env="LLAMINAR_CPU_NVNNI_VERIFIER_PAIRED_CSV"
        paired_test="TrainerCsv_StrongVerifierRows_AllFormats"
      else
        echo "error: unknown CPU sealed paired surface ${surface}" >&2
        return 2
      fi
      mpi_command+=(
        -np 1 env
        "LLAMINAR_PROFILING=0"
        "LLAMINAR_PERF_STATS_JSON=0"
        "LLAMINAR_PERF_STATS_CSV=0"
        "LLAMINAR_PERF_STATS_TABLE=0"
        "LLAMINAR_PERF_STATS_SUMMARY=0"
        "LLAMINAR_ISA_LEVEL=${runtime_isa}"
        "LLAMINAR_CPU_NVNNI_DECODE_WARMUP=${cpu_warmup}"
        "LLAMINAR_CPU_NVNNI_DECODE_ITERS=${cpu_iters}"
        "LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=${cpu_warmup}"
        "LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=${cpu_iters}"
        "${paired_manifest_env}=${manifest_path}"
        "${paired_csv_env}=${inprogress}"
        "${trainer_bin}"
        "--gtest_filter=*${paired_test}"
      )
    done
    rm -f "${inprogress_paths[@]}"
    run_cmd \
      "OMP_NUM_THREADS=${cpu_threads}" \
      "OMP_PLACES=cores" \
      "OMP_PROC_BIND=close" \
      "OMP_DYNAMIC=false" \
      "OMP_NESTED=false" \
      "HWLOC_COMPONENTS=-gl,-opencl" \
      "OMPI_MCA_mpi_leave_pinned=1" \
      "OMPI_MCA_btl_vader_single_copy_mechanism=none" \
      "${mpi_command[@]}"
    for ((index = 0; index < ${#final_paths[@]}; ++index)); do
      mv "${inprogress_paths[index]}" "${final_paths[index]}"
    done
  done

  for evidence_path in "${output_paths[@]}"; do
    if [[ ! -s "${evidence_path}" ]]; then
      echo "error: CPU sealed collection omitted ${evidence_path}" >&2
      return 2
    fi
  done
}

# Refine the grouped verifier policy with the same paired, resumable evidence
# loop used by M=1 decode. Broad timing identifies plausible candidates; only
# isolated within-cell ratios are allowed to change the generic winner graph.
run_cpu_grouped_paired_refinement() {
  local common_observations="$1"
  local paired_dir="${output_dir}/cpu_grouped_paired_refinement"
  mkdir -p "${paired_dir}"

  cpu_grouped_paired_evidence=()
  local first_iteration=0
  local candidate_iteration candidate_tag candidate_shard_dir
  for ((candidate_iteration = 0; ; ++candidate_iteration)); do
    printf -v candidate_tag '%03d' "${candidate_iteration}"
    candidate_shard_dir="${paired_dir}/iteration-${candidate_tag}.shards"
    local completed_manifests=()
    mapfile -t completed_manifests < <(
      find "${candidate_shard_dir}" -maxdepth 1 -type f \
        -name 'shard-*.requests.json' 2>/dev/null | sort
    )
    if (( ${#completed_manifests[@]} == 0 )); then
      first_iteration="${candidate_iteration}"
      break
    fi
    local candidate_complete=1
    local completed_manifest completed_evidence
    local candidate_evidence=()
    for completed_manifest in "${completed_manifests[@]}"; do
      completed_evidence="${completed_manifest%.requests.json}.csv"
      if [[ ! -s "${completed_evidence}" ]]; then
        candidate_complete=0
        break
      fi
      candidate_evidence+=("${completed_evidence}")
    done
    if (( ! candidate_complete )); then
      first_iteration="${candidate_iteration}"
      break
    fi
    cpu_grouped_paired_evidence+=("${candidate_evidence[@]}")
    first_iteration=$((candidate_iteration + 1))
  done

  local iteration
  local iteration_limit=$((first_iteration + paired_max_iterations))
  for ((iteration = first_iteration;
        iteration < iteration_limit;
        ++iteration)); do
    local iteration_tag
    printf -v iteration_tag '%03d' "${iteration}"
    local requests="${paired_dir}/iteration-${iteration_tag}.requests.json"
    local report="${paired_dir}/iteration-${iteration_tag}.report.json"
    local shard_dir="${paired_dir}/iteration-${iteration_tag}.shards"
    local certificate="${paired_dir}/iteration-${iteration_tag}.certificate.json"
    local planner=(
      python3 -m native_vnni_dispatch.paired_requests
      "${common_observations}"
      --surface grouped-verifier
      --output "${requests}"
      --report "${report}"
      --request-shard-dir "${shard_dir}"
      --max-requests-per-shard 16
      --shape-manifest "${shape_manifest_path}"
      --fit-cache-dir "${cpu_grouped_fit_cache_dir}"
      --development-profiler-requests "${cpu_profiler_requests}"
      --development-profiler-evidence "${cpu_profiler_evidence}"
      --development-profiler-observations "${common_observations}"
      --max-leaves "${cpu_grouped_max_leaves}"
      --max-regret "${cpu_grouped_paired_max_regret_fraction}"
    )
    local burned_index
    for burned_index in "${!cpu_grouped_burned_sealed_plans[@]}"; do
      planner+=(
        --burned-sealed-plan-json
        "${cpu_grouped_burned_sealed_plans[burned_index]}"
        --burned-sealed-paired-dir
        "${cpu_grouped_burned_sealed_paired_dirs[burned_index]}"
      )
    done
    local paired_path
    for paired_path in "${cpu_grouped_paired_evidence[@]}"; do
      planner+=(--paired-csv "${paired_path}")
    done
    run_cmd "PYTHONPATH=${paired_planner_root}" "${planner[@]}"

    if (( dry_run )); then
      printf 'dry-run: grouped CPU paired refinement would consume %s\n' \
        "${report}"
      return 0
    fi
    local status request_count
    status="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "${report}")"
    request_count="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["request_count"])' \
      "${report}")"
    printf 'CPU grouped paired refinement iteration=%s status=%s requests=%s retained_shards=%s\n' \
      "${iteration_tag}" "${status}" "${request_count}" \
      "${#cpu_grouped_paired_evidence[@]}"

    case "${status}" in
      green)
        if (( request_count != 0 )); then
          echo "error: green grouped CPU paired plan still contains requests" >&2
          return 2
        fi
        return 0
        ;;
      pending)
        if (( request_count == 0 )); then
          echo "error: pending grouped CPU paired plan has no requests" >&2
          return 2
        fi
        ;;
      confirmed_failure|evidence_conflict|insufficient_cross_validation)
        echo "error: grouped CPU paired refinement stopped with ${status}; see ${report}" >&2
        return 2
        ;;
      *)
        echo "error: unknown grouped CPU paired status ${status}" >&2
        return 2
        ;;
    esac

    local current_evidence=()
    run_cpu_sealed_paired_shards \
      "grouped-decode" "${shard_dir}" current_evidence
    cpu_grouped_paired_evidence+=("${current_evidence[@]}")
    run_cmd \
      "PYTHONPATH=${paired_planner_root}" \
      python3 -m native_vnni_dispatch.paired_confirmation \
      "${cpu_grouped_paired_evidence[@]}" \
      --output "${certificate}"
  done

  echo "error: grouped CPU paired refinement did not converge after " \
       "${paired_max_iterations} new iterations from ${first_iteration}" >&2
  return 2
}

refresh_cpu_decode() {
  local cpu_git_revision="$1"
  local cpu_build_id="$2"
  local cpu_compiler_id="$3"
  local cpu_arch_class="$4"
  local cpu_device_name="$5"
  local cpu_driver_runtime="$6"
  local cpu_serial_policy_hash="$7"

  # Development timing and a post-freeze paired seal are distinct physical
  # generations. Preserve the binaries available for the fresh seal before a
  # fit-only replay restores the immutable provenance recorded by the timing
  # corpus.
  local cpu_measurement_git_revision="${cpu_git_revision}"
  local cpu_measurement_build_id="${cpu_build_id}"
  local cpu_measurement_compiler_id="${cpu_compiler_id}"
  local cpu_measurement_arch_class="${cpu_arch_class}"
  local cpu_measurement_device_name="${cpu_device_name}"
  local cpu_measurement_driver_runtime="${cpu_driver_runtime}"
  local cpu_measurement_serial_policy_hash="${cpu_serial_policy_hash}"

  cpu_decode_batches_run=0
  cpu_decode_collection_limited=0

  local decode_shapes="${shapes}"
  if (( partitioned_measurement )); then
    case "${shape_partition}" in
      fast-development)
        decode_shapes="${cpu_decode_development_shapes}"
        ;;
      fast-sealed)
        return 0
        ;;
      *)
        return 0
        ;;
    esac
  elif [[ "${measurement_profile}" == "production" ]]; then
    decode_shapes="${cpu_decode_development_shapes}"
  fi

  local weighted_shapes=()
  local shape n k work
  while IFS= read -r shape; do
    n="$(cpu_shape_n "${shape}")"
    k="$(cpu_shape_k "${shape}")"
    work=$((n * k))
    weighted_shapes+=("$(printf '%020d\t%s' "${work}" "${shape}")")
  done < <(csv_values "${decode_shapes}")
  local measurement_shapes
  measurement_shapes="$({
    printf '%s\n' "${weighted_shapes[@]}" |
      sort -t $'\t' -k1,1nr -k2,2 |
      cut -f2
  } | paste -sd, -)"

  local partials=()
  local timing_partials=()
  local development_partials=()
  local development_timing_partials=()
  if (( ! skip_sweep )); then
    local format_shards=()
    if (( stratified_formats || cpu_format_shards )); then
      mapfile -t format_shards < <(csv_values "${cpu_formats}")
    else
      format_shards=("${cpu_formats}")
    fi
    local regime_specs=(
      "avx2-build.avx2-runtime|avx2|${cpu_avx2_sweep_bin}"
      "avx512-build.avx2-runtime|avx2|${cpu_avx512_sweep_bin}"
      "avx512-build.avx512-runtime|avx512|${cpu_avx512_sweep_bin}"
    )
    local job_formats=()
    local job_shapes=()
    local job_ns=()
    local job_ks=()
    local job_runtime_isas=()
    local job_bins=()
    local job_partials=()
    local job_timing_partials=()
    local format_spec format_label regime_spec regime_name runtime_isa regime_bin
    local partial timing_partial
    for format_spec in "${format_shards[@]}"; do
      if (( stratified_formats || cpu_format_shards )); then
        format_label="${format_spec}"
      else
        format_label="all-formats"
      fi
      for regime_spec in "${regime_specs[@]}"; do
        IFS='|' read -r regime_name runtime_isa regime_bin <<< "${regime_spec}"
        while IFS= read -r shape; do
          n="$(cpu_shape_n "${shape}")"
          k="$(cpu_shape_k "${shape}")"
          partial="${output_dir}/cpu_decode_m1.${format_label}.${shape}.${regime_name}.csv"
          timing_partial="${output_dir}/cpu_decode_m1.${format_label}.${shape}.${regime_name}.timing.csv"
          partials+=("${partial}")
          timing_partials+=("${timing_partial}")
          if [[ "${measurement_profile}" == "production" ]]; then
            if csv_contains "${cpu_decode_development_shapes}" "${shape}"; then
              development_partials+=("${partial}")
              development_timing_partials+=("${timing_partial}")
            else
              echo "error: CPU decode shape ${shape} is not development evidence" >&2
              return 2
            fi
          fi
          if [[ -s "${partial}" && -s "${timing_partial}" ]]; then
            continue
          fi
          job_formats+=("${format_spec}")
          job_shapes+=("${shape}")
          job_ns+=("${n}")
          job_ks+=("${k}")
          job_runtime_isas+=("${runtime_isa}")
          job_bins+=("${regime_bin}")
          job_partials+=("${partial}")
          job_timing_partials+=("${timing_partial}")
        done < <(csv_values "${measurement_shapes}")
      done
    done

    local batch_start batch_end index
    for ((batch_start = 0;
          batch_start < ${#job_shapes[@]};
          batch_start += cpu_measurement_lanes)); do
      if (( cpu_batch_limit > 0 &&
            cpu_decode_batches_run >= cpu_batch_limit )); then
        cpu_decode_collection_limited=1
        break
      fi
      batch_end=$((batch_start + cpu_measurement_lanes))
      if (( batch_end > ${#job_shapes[@]} )); then
        batch_end=${#job_shapes[@]}
      fi
      local mpi_command=(
        mpirun --bind-to socket --map-by socket
        --mca mpi_leave_pinned 1
        --mca btl_vader_single_copy_mechanism none
        --mca orte_allowed_exit_without_sync 1
      )
      local inprogress_paths=()
      for ((index = batch_start; index < batch_end; ++index)); do
        if (( index > batch_start )); then
          mpi_command+=(":")
        fi
        local partial_inprogress="${job_partials[index]}.inprogress"
        local timing_inprogress="${job_timing_partials[index]}.inprogress"
        inprogress_paths+=("${partial_inprogress}" "${timing_inprogress}")
        mpi_command+=(
          -np 1 env
          "LLAMINAR_ISA_LEVEL=${job_runtime_isas[index]}"
          "LLAMINAR_CPU_NVNNI_DECODE_FORMATS=${job_formats[index]}"
          "LLAMINAR_CPU_NVNNI_DECODE_M=1"
          "LLAMINAR_CPU_NVNNI_DECODE_SHAPE_NAME=${job_shapes[index]}"
          "LLAMINAR_CPU_NVNNI_DECODE_N=${job_ns[index]}"
          "LLAMINAR_CPU_NVNNI_DECODE_K=${job_ks[index]}"
          "LLAMINAR_CPU_NVNNI_DECODE_MAX_CASES=${cpu_max_cases}"
          "LLAMINAR_CPU_NVNNI_DECODE_WARMUP=${cpu_warmup}"
          "LLAMINAR_CPU_NVNNI_DECODE_ITERS=${cpu_iters}"
          "LLAMINAR_CPU_NVNNI_DECODE_STRONG_CSV=${partial_inprogress}"
          "LLAMINAR_CPU_NVNNI_DECODE_TIMING_CSV=${timing_inprogress}"
          "${job_bins[index]}"
          "--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"
        )
      done
      run_cmd rm -f "${inprogress_paths[@]}"
      run_cmd \
        "OMP_NUM_THREADS=${cpu_threads}" \
        "OMP_PLACES=cores" \
        "OMP_PROC_BIND=close" \
        "OMP_DYNAMIC=false" \
        "OMP_NESTED=false" \
        "HWLOC_COMPONENTS=-gl,-opencl" \
        "OMPI_MCA_mpi_leave_pinned=1" \
        "OMPI_MCA_btl_vader_single_copy_mechanism=none" \
        "${mpi_command[@]}"
      for ((index = batch_start; index < batch_end; ++index)); do
        run_cmd mv "${job_partials[index]}.inprogress" "${job_partials[index]}"
        run_cmd mv \
          "${job_timing_partials[index]}.inprogress" \
          "${job_timing_partials[index]}"
      done
      cpu_decode_batches_run=$((cpu_decode_batches_run + 1))
    done

    if (( cpu_decode_collection_limited )); then
      printf 'CPU M=1 collection checkpointed after %d pending batch(es); resume from %s\n' \
        "${cpu_decode_batches_run}" "${output_dir}"
      return 0
    fi

    if [[ "${measurement_profile}" == "production" ]]; then
      combine_csvs "${cpu_decode_development_csv}" "${development_partials[@]}"
      combine_csvs \
        "${cpu_decode_development_timing_csv}" \
        "${development_timing_partials[@]}"
    else
      combine_csvs "${cpu_decode_csv}" "${partials[@]}"
      combine_csvs "${cpu_decode_timing_csv}" "${timing_partials[@]}"
    fi
  fi

  if (( partitioned_measurement )); then
    return 0
  fi

  if [[ "${measurement_profile}" == "production" ]]; then
    # Keep observation and request identities stable across resumptions of one
    # development corpus. Physical coverage is provenance-independent, but a
    # stable run ID also lets an interrupted collector reopen its exact journal
    # rather than creating a timestamp-named sibling transaction.
    local run_id
    if [[ -s "${cpu_decode_development_run_id_file}" ]]; then
      IFS= read -r run_id < "${cpu_decode_development_run_id_file}"
    else
      run_id="${timestamp}-cpu-decode-${profile}-development"
      if (( ! dry_run )); then
        printf '%s\n' "${run_id}" > "${cpu_decode_development_run_id_file}.inprogress"
        mv "${cpu_decode_development_run_id_file}.inprogress" \
          "${cpu_decode_development_run_id_file}"
      fi
    fi
    if [[ -z "${run_id}" || "${run_id}" == *$'\n'* ]]; then
      echo "error: invalid CPU decode development run ID checkpoint" >&2
      exit 2
    fi
    local reuse_cpu_decode_common=0
    if (( skip_sweep )) && [[ -s "${cpu_decode_common_csv}" ]]; then
      local recorded_run_id
      recorded_run_id="$(csv_unique_field "${cpu_decode_common_csv}" run_id)"
      if [[ "${run_id}" != "${recorded_run_id}" ]]; then
        echo "error: CPU decode run-ID checkpoint disagrees with common corpus" >&2
        exit 2
      fi
      cpu_git_revision="$(
        csv_unique_field "${cpu_decode_common_csv}" git_revision
      )"
      cpu_build_id="$(
        csv_unique_field_prefix \
          "${cpu_decode_common_csv}" build_id '|cpu_isa='
      )"
      cpu_compiler_id="$(
        csv_unique_field "${cpu_decode_common_csv}" compiler_id
      )"
      cpu_arch_class="$(
        csv_unique_field_prefix \
          "${cpu_decode_common_csv}" architecture_class '|build='
      )"
      cpu_device_name="$(
        csv_unique_field "${cpu_decode_common_csv}" device_name
      )"
      cpu_driver_runtime="$(
        csv_unique_field "${cpu_decode_common_csv}" driver_runtime
      )"
      cpu_serial_policy_hash="$(
        csv_unique_field "${cpu_decode_common_csv}" serial_m1_policy_hash
      )"
      if [[ "${cpu_serial_policy_hash}" != \
            "${cpu_measurement_serial_policy_hash}" ]]; then
        echo "error: fresh CPU decode seal cannot certify a changed serial arithmetic policy" >&2
        exit 2
      fi
      if csv_has_field "${cpu_decode_common_csv}" launch_k_tiles; then
        reuse_cpu_decode_common=1
        printf 'Fit-only CPU decode replay retains measured build provenance: %s\n' \
          "${cpu_build_id}"
      else
        printf '%s\n' \
          'CPU decode common corpus predates launch_k_tiles; re-adapting retained raw timing evidence.'
      fi
    fi
    if (( ! reuse_cpu_decode_common )); then
      run_cmd python3 "${cpu_decode_generator}" \
        --input "${cpu_decode_development_csv}" \
        --timing-sidecar "${cpu_decode_development_timing_csv}" \
        --output "${cpu_decode_provisional_inc}" \
        --common-observations "${cpu_decode_common_csv}" \
        --adapt-only --profile production \
        --run-id "${run_id}" \
        --git-revision "${cpu_git_revision}" \
        --build-id "${cpu_build_id}" \
        --compiler-id "${cpu_compiler_id}" \
        --architecture-class "${cpu_arch_class}" \
        --device-name "${cpu_device_name}" \
        --driver-runtime "${cpu_driver_runtime}" \
        --serial-m1-policy-hash "${cpu_serial_policy_hash}" \
        --shape-manifest "${shape_manifest_path}" \
        "${cpu_fixed_timing_gate_args[@]}"
    fi
    collect_backend_profiler_evidence \
      cpu "${cpu_decode_common_csv}" "${cpu_avx2_sweep_bin}" \
      "${cpu_decode_profiler_requests}" "${cpu_decode_profiler_evidence}" \
      "${cpu_decode_profiler_features}" "${cpu_decode_profiler_raw}"
    run_cpu_decode_paired_refinement "${cpu_decode_common_csv}"

    local -a cpu_decode_burned_args=()
    local burned_index
    for burned_index in "${!cpu_decode_burned_sealed_plans[@]}"; do
      cpu_decode_burned_args+=(
        --burned-sealed-plan-json
        "${cpu_decode_burned_sealed_plans[burned_index]}"
        --burned-sealed-paired-dir
        "${cpu_decode_burned_sealed_paired_dirs[burned_index]}"
      )
    done

    local cpu_decode_freeze=(
      python3 "${cpu_decode_generator}"
      --input "${cpu_decode_development_csv}" \
      --timing-sidecar "${cpu_decode_development_timing_csv}" \
      --output "${cpu_decode_frozen_inc}" \
      --summary "${cpu_decode_summary}" \
      --common-observations "${cpu_decode_common_csv}" \
      --reuse-development-common \
      --policy-json "${cpu_decode_frozen_policy_json}" \
      --sealed-route-probe-json "${cpu_decode_sealed_route_probe_json}" \
      --freeze-generic --profile production \
      --run-id "${run_id}" \
      --git-revision "${cpu_git_revision}" \
      --build-id "${cpu_build_id}" \
      --compiler-id "${cpu_compiler_id}" \
      --architecture-class "${cpu_arch_class}" \
      --device-name "${cpu_device_name}" \
      --driver-runtime "${cpu_driver_runtime}" \
      --serial-m1-policy-hash "${cpu_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --fit-cache-dir "${cpu_decode_fit_cache_dir}" \
      --development-profiler-requests "${cpu_decode_profiler_requests}" \
      --development-profiler-evidence "${cpu_decode_profiler_evidence}" \
      --development-profiler-observations "${cpu_decode_profiler_witnesses}" \
      --sealed-build-id "${cpu_measurement_build_id}" \
      "${cpu_fixed_timing_gate_args[@]}"
    )
    local cpu_paired_path
    for cpu_paired_path in "${cpu_decode_paired_evidence[@]}"; do
      cpu_decode_freeze+=(--paired-development-csv "${cpu_paired_path}")
    done
    cpu_decode_freeze+=("${cpu_decode_burned_args[@]}")
    run_cmd "${cpu_decode_freeze[@]}"

    local cpu_decode_sealed_route_args=()
    local cpu_decode_sealed_build_token="${cpu_measurement_build_id#sha256:}"
    cpu_decode_sealed_build_token="${cpu_decode_sealed_build_token:0:12}"
    local cpu_decode_sealed_route_probe_token=""
    if (( dry_run )); then
      cpu_decode_sealed_route_probe_token="dry-run-probe"
    else
      # A trainer rebuild does not imply that the learned reserve geometry is
      # unchanged. Bind each timing-free route manifest to the exact probe
      # bytes as well as the binary build so a later reserve generation can
      # never reuse a stale shape subset from the same executable.
      cpu_decode_sealed_route_probe_token="$(
        sha256sum "${cpu_decode_sealed_route_probe_json}" |
          awk '{print substr($1, 1, 12)}'
      )"
    fi
    local route_spec route_regime route_runtime_isa route_bin route_manifest
    for route_spec in \
        "avx2-build.avx2-runtime|AVX2|${cpu_avx2_sweep_bin}" \
        "avx512-build.avx2-runtime|AVX2|${cpu_avx512_sweep_bin}" \
        "avx512-build.avx512-runtime|AVX512|${cpu_avx512_sweep_bin}"; do
      IFS='|' read -r route_regime route_runtime_isa route_bin <<< "${route_spec}"
      route_manifest="${output_dir}/cpu_decode_m1.sealed-route.${route_regime}.${cpu_decode_sealed_build_token}.${cpu_decode_sealed_route_probe_token}.csv"
      if [[ ! -s "${route_manifest}" ]]; then
        run_cmd \
          "OMP_NUM_THREADS=${cpu_threads}" \
          "OMP_PLACES=cores" \
          "OMP_PROC_BIND=close" \
          "LLAMINAR_ISA_LEVEL=${route_runtime_isa}" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_CSV=${route_manifest}.inprogress" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_ISA_REGIME=${route_regime}" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_REFINEMENT_PROBE_JSON=${cpu_decode_sealed_route_probe_json}" \
          "${route_bin}" \
          "--gtest_filter=*TrainerCsv_PrefillSerialRouteManifest"
        if (( ! dry_run )); then
          mv "${route_manifest}.inprogress" "${route_manifest}"
        fi
      fi
      cpu_decode_sealed_route_args+=(--sealed-route-manifest "${route_manifest}")
    done

    local cpu_decode_plan=(
      python3 "${cpu_decode_generator}"
      --input "${cpu_decode_development_csv}"
      --timing-sidecar "${cpu_decode_development_timing_csv}"
      --output "${cpu_decode_frozen_inc}"
      --common-observations "${cpu_decode_common_csv}"
      --reuse-development-common
      --frozen-policy-json "${cpu_decode_frozen_policy_json}"
      --sealed-route-probe-json "${cpu_decode_sealed_route_probe_json}"
      --sealed-plan-json "${cpu_decode_sealed_plan_json}"
      --sealed-request-dir "${cpu_decode_sealed_request_dir}"
      --plan-sealed-pairs --profile production
      --run-id "${run_id}"
      --git-revision "${cpu_git_revision}"
      --build-id "${cpu_build_id}"
      --compiler-id "${cpu_compiler_id}"
      --architecture-class "${cpu_arch_class}"
      --device-name "${cpu_device_name}"
      --driver-runtime "${cpu_driver_runtime}"
      --serial-m1-policy-hash "${cpu_serial_policy_hash}"
      --shape-manifest "${shape_manifest_path}"
      --fit-cache-dir "${cpu_decode_fit_cache_dir}"
      --development-profiler-requests "${cpu_decode_profiler_requests}"
      --development-profiler-evidence "${cpu_decode_profiler_evidence}"
      --development-profiler-observations "${cpu_decode_profiler_witnesses}"
      --sealed-build-id "${cpu_measurement_build_id}"
      "${cpu_decode_sealed_route_args[@]}"
      "${cpu_fixed_timing_gate_args[@]}"
    )
    for cpu_paired_path in "${cpu_decode_paired_evidence[@]}"; do
      cpu_decode_plan+=(--paired-development-csv "${cpu_paired_path}")
    done
    cpu_decode_plan+=("${cpu_decode_burned_args[@]}")
    run_cmd "${cpu_decode_plan[@]}"

    local cpu_decode_sealed_paired_csvs=()
    run_cpu_sealed_paired_shards \
      "decode-m1" "${cpu_decode_sealed_request_dir}" \
      cpu_decode_sealed_paired_csvs
    local cpu_decode_certify=(
      python3 "${cpu_decode_generator}"
      --development-input "${cpu_decode_development_csv}" \
      --development-timing-sidecar "${cpu_decode_development_timing_csv}" \
      --frozen-policy-json "${cpu_decode_frozen_policy_json}" \
      --sealed-plan-json "${cpu_decode_sealed_plan_json}" \
      --policy-json "${cpu_decode_policy_json}" \
      --certification-diagnostic "${cpu_decode_certification_diagnostic}" \
      --output "${cpu_decode_inc}" \
      --summary "${cpu_decode_summary}" \
      --common-observations "${cpu_decode_common_csv}" \
      --reuse-development-common \
      --certify-generic --profile production \
      --run-id "${run_id}" \
      --git-revision "${cpu_git_revision}" \
      --build-id "${cpu_build_id}" \
      --compiler-id "${cpu_compiler_id}" \
      --architecture-class "${cpu_arch_class}" \
      --device-name "${cpu_device_name}" \
      --driver-runtime "${cpu_driver_runtime}" \
      --serial-m1-policy-hash "${cpu_serial_policy_hash}" \
      --shape-manifest "${shape_manifest_path}" \
      --fit-cache-dir "${cpu_decode_fit_cache_dir}" \
      --development-profiler-requests "${cpu_decode_profiler_requests}" \
      --development-profiler-evidence "${cpu_decode_profiler_evidence}" \
      --development-profiler-observations "${cpu_decode_profiler_witnesses}" \
      --sealed-build-id "${cpu_measurement_build_id}" \
      "${cpu_fixed_timing_gate_args[@]}"
    )
    for cpu_paired_path in "${cpu_decode_paired_evidence[@]}"; do
      cpu_decode_certify+=(--paired-development-csv "${cpu_paired_path}")
    done
    cpu_decode_certify+=(
      --sealed-paired-dir "${cpu_decode_sealed_request_dir}"
    )
    cpu_decode_certify+=("${cpu_decode_burned_args[@]}")
    run_cmd "${cpu_decode_certify[@]}"
    run_cmd python3 "${dispatch_validator}" "${cpu_decode_inc}"
    run_cmd \
      "PYTHONPATH=${shape_manifest_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${cpu_decode_policy_json}" \
      --include "${cpu_decode_inc}"
    if (( install )); then
      local target="${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIDecodePolicyGenerated.inc"
      run_cmd cp "${cpu_decode_inc}" "${target}.inprogress"
      run_cmd mv "${target}.inprogress" "${target}"
      rebuild_cpu_native_vnni_trainer "${cpu_avx2_sweep_bin}"
      rebuild_cpu_native_vnni_trainer "${cpu_avx512_sweep_bin}"
    fi
    return 0
  fi

  run_cmd python3 "${cpu_decode_generator}" \
    --input "${cpu_decode_csv}" \
    --timing-sidecar "${cpu_decode_timing_csv}" \
    --output "${cpu_decode_inc}" \
    --summary "${cpu_decode_summary}" \
    --common-observations "${cpu_decode_common_csv}" \
    --require-isa-matrix \
    --profile "${measurement_profile}" \
    --run-id "${timestamp}-cpu-decode-${profile}" \
    --git-revision "${cpu_git_revision}" \
    --build-id "${cpu_build_id}" \
    --compiler-id "${cpu_compiler_id}" \
    --architecture-class "${cpu_arch_class}" \
    --device-name "${cpu_device_name}" \
    --driver-runtime "${cpu_driver_runtime}" \
    --serial-m1-policy-hash "${cpu_serial_policy_hash}" \
    --shape-manifest "${shape_manifest_path}" \
    "${cpu_fixed_timing_gate_args[@]}"
}

authenticate_completed_cpu_decode() {
  local installed_inc="${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIDecodePolicyGenerated.inc"

  # The continuation boundary is deliberately stronger than a file-exists
  # checkpoint. The policy validator authenticates sealed certification and
  # the embedded policy digests, while byte equality proves the exact include
  # under review is already the production table consumed by both CPU builds.
  if (( ! dry_run )); then
    local required_path
    for required_path in \
        "${cpu_decode_policy_json}" \
        "${cpu_decode_inc}" \
        "${installed_inc}"; do
      if [[ ! -s "${required_path}" ]]; then
        echo "error: --resume-after-cpu-decode requires completed artifact ${required_path}" >&2
        return 2
      fi
    done
  fi

  run_cmd \
    "PYTHONPATH=${shape_manifest_python_root}" \
    python3 -m native_vnni_dispatch.policy_artifact \
    --policy-json "${cpu_decode_policy_json}" \
    --include "${cpu_decode_inc}"
  if (( dry_run )); then
    run_cmd cmp --silent "${cpu_decode_inc}" "${installed_inc}"
  elif ! cmp --silent "${cpu_decode_inc}" "${installed_inc}"; then
    echo "error: installed CPU decode policy differs from the certified output artifact" >&2
    return 2
  fi
  printf 'Authenticated installed CPU decode prerequisite: %s\n' \
    "${cpu_decode_policy_json}"
}

refresh_cpu() {
  begin_backend_collection_target "CPU grouped verifier"
  if [[ "${measurement_profile}" == "production" ||
        "${measurement_profile}" == "partial-production" ]]; then
    require_cpu_performance_governor
  fi
  require_executable "${cpu_avx2_sweep_bin}"
  require_executable "${cpu_avx512_sweep_bin}"
  local cpu_git_revision cpu_build_id cpu_compiler_id cpu_arch_class
  local cpu_device_name cpu_driver_runtime cpu_serial_policy_hash
  cpu_git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
  cpu_build_id="$(sha256_file_set \
    "${cpu_avx2_sweep_bin}" "${cpu_avx512_sweep_bin}")"
  cpu_compiler_id="$(${CXX:-c++} --version | first_output_line)"
  cpu_arch_class="$(cpu_architecture_class)"
  cpu_device_name="$(cpu_lscpu_field "Model name")"
  cpu_driver_runtime="linux-$(uname -r)-openmp"
  cpu_serial_policy_hash="$(cpu_serial_arithmetic_contract_hash)"
  # Development timings and the post-freeze grouped seal can come from two
  # different binary generations. Keep the currently installed trainers as
  # the seal identity before a fit-only replay restores the immutable
  # provenance recorded by the development corpus.
  local cpu_measurement_build_id="${cpu_build_id}"
  local cpu_measurement_serial_policy_hash="${cpu_serial_policy_hash}"
  if (( resume_after_cpu_decode )); then
    authenticate_completed_cpu_decode
  elif [[ "${shape_partition}" != verifier-* ]]; then
    refresh_cpu_decode \
      "${cpu_git_revision}" "${cpu_build_id}" "${cpu_compiler_id}" \
      "${cpu_arch_class}" "${cpu_device_name}" "${cpu_driver_runtime}" \
      "${cpu_serial_policy_hash}"
    if (( cpu_decode_collection_limited )); then
      finish_backend_collection_target
      return
    fi
    if (( stop_after_cpu_decode )) || [[ "${shape_partition}" == fast-* ]]; then
      finish_backend_collection_target
      return
    fi
    if [[ "${measurement_profile}" == "production" ]]; then
      cpu_build_id="$(sha256_file_set \
        "${cpu_avx2_sweep_bin}" "${cpu_avx512_sweep_bin}")"
      cpu_serial_policy_hash="$(cpu_serial_arithmetic_contract_hash)"
    fi
  fi
  # Grouped verifier rows begin at M=2. Keep this inventory independent from
  # the M=1 serial-decode transaction so caller-provided mixed ranges cannot
  # accidentally admit the oracle-only row count into grouped evidence.
  local cpu_sweep_m_values="${cpu_grouped_m_values}"
  local cpu_sweep_shapes="${shapes}"
  if (( partitioned_measurement )); then
    cpu_sweep_m_values="${partition_m_values}"
    case "${shape_partition}" in
      verifier-development)
        cpu_sweep_shapes="${cpu_verifier_development_shapes}"
        ;;
      verifier-sealed)
        finish_backend_collection_target
        return
        ;;
    esac
  elif [[ "${measurement_profile}" == "production" ]]; then
    # CPU currently owns only the grouped verifier contract. M=1 remains the
    # independent production serial oracle and Fast-only CUDA/ROCm refinement
    # geometries have no place in this measurement matrix.
    cpu_sweep_m_values="${canonical_verifier_m_values}"
    cpu_sweep_shapes="${cpu_verifier_development_shapes}"
  fi
  if [[ -z "${cpu_sweep_m_values}" ]]; then
    echo "error: CPU grouped-verifier collection requires at least one --m-values entry >= 2" >&2
    exit 2
  fi
  # One MPMD launch is a synchronization boundary: both ranks must finish
  # before the next pair can start. Pairing manifest neighbors can therefore
  # strand a socket when a giant vocabulary projection sits beside a compact
  # attention projection. Sort by the dominant N*K work estimate and pair
  # adjacent shapes. The ordering is deterministic, every shape occurs once,
  # and rebuilding it identically for each ISA regime keeps a shape on the
  # same rank/socket while giving the two ranks comparable work.
  local weighted_cpu_shapes=()
  local weighted_shape weighted_n weighted_k weighted_work
  while IFS= read -r weighted_shape; do
    weighted_n="$(cpu_shape_n "${weighted_shape}")"
    weighted_k="$(cpu_shape_k "${weighted_shape}")"
    weighted_work=$((weighted_n * weighted_k))
    weighted_cpu_shapes+=("$(printf '%020d\t%s' "${weighted_work}" "${weighted_shape}")")
  done < <(csv_values "${cpu_sweep_shapes}")
  local cpu_measurement_shapes
  cpu_measurement_shapes="$({
    printf '%s\n' "${weighted_cpu_shapes[@]}" |
      sort -t $'\t' -k1,1nr -k2,2 |
      cut -f2
  } | paste -sd, -)"
  local cpu_contract_path="${output_dir}/cpu_collection_contract.sha256"
  local cpu_contract_payload cpu_contract_digest
  cpu_contract_payload="$(printf '%s\n' \
    "schema=cpu-native-vnni-collection-v1" \
    "profile=${measurement_profile}" \
    "shape_partition=${shape_partition}" \
    "shapes=${cpu_measurement_shapes}" \
    "formats=${cpu_formats}" \
    "m_values=${cpu_sweep_m_values}" \
    "threads=${cpu_threads}" \
    "measurement_lanes=${cpu_measurement_lanes}" \
    "format_shards=${cpu_format_shards}" \
    "warmups=${cpu_warmup}" \
    "samples=${cpu_iters}" \
    "max_cases=${cpu_max_cases}" \
    "build_id=${cpu_build_id}" \
    "serial_policy_hash=${cpu_serial_policy_hash}" \
    "shape_manifest_hash=$(sha256sum "${shape_manifest_path}" | awk '{print $1}')")"
  cpu_contract_digest="$(
    printf '%s' "${cpu_contract_payload}" | sha256sum | awk '{print $1}'
  )"
  if (( resume_cpu_partials )); then
    if (( dry_run )); then
      printf 'dry-run: verify CPU collection contract %s == %s\n' \
        "${cpu_contract_path}" "${cpu_contract_digest}"
    else
      if [[ ! -f "${cpu_contract_path}" ]]; then
        echo "error: --resume-cpu-partials requires ${cpu_contract_path}" >&2
        exit 2
      fi
      local existing_cpu_contract
      existing_cpu_contract="$(<"${cpu_contract_path}")"
      if [[ "${existing_cpu_contract}" != "${cpu_contract_digest}" ]]; then
        echo "error: CPU partial collection contract changed; use the " \
             "original command or a new output directory" >&2
        exit 2
      fi
    fi
  elif (( ! skip_sweep )); then
    if (( dry_run )); then
      printf 'dry-run: publish CPU collection contract %s -> %s\n' \
        "${cpu_contract_digest}" "${cpu_contract_path}"
    else
      printf '%s\n' "${cpu_contract_digest}" > "${cpu_contract_path}.inprogress"
      mv "${cpu_contract_path}.inprogress" "${cpu_contract_path}"
    fi
  fi
  if (( ! skip_sweep )); then
    local partials=()
    local timing_partials=()
    local development_partials=()
    local development_timing_partials=()
    local format_shards=()
    local job_format_specs=()
    local job_shapes=()
    local job_ns=()
    local job_ks=()
    local job_runtime_isas=()
    local job_regime_bins=()
    local job_partials=()
    local job_timing_partials=()
    local job_group_starts=()
    local job_group_ends=()
    if (( stratified_formats || cpu_format_shards )); then
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
      for regime_spec in "${regime_specs[@]}"; do
        IFS='|' read -r regime_name runtime_isa regime_bin <<< "${regime_spec}"
        job_group_starts+=("${#job_shapes[@]}")
        while IFS= read -r shape; do
          n="$(cpu_shape_n "${shape}")"
          k="$(cpu_shape_k "${shape}")"
          partial="${output_dir}/cpu_verifier_rows.${format_label}.${shape}.${regime_name}.csv"
          timing_partial="${output_dir}/cpu_verifier_rows.${format_label}.${shape}.${regime_name}.timing.csv"
          partials+=("${partial}")
          timing_partials+=("${timing_partial}")
          if [[ "${measurement_profile}" == "production" ]]; then
            if csv_contains "${cpu_verifier_development_shapes}" "${shape}"; then
              development_partials+=("${partial}")
              development_timing_partials+=("${timing_partial}")
            else
              echo "error: CPU verifier shape ${shape} is not development evidence" >&2
              exit 2
            fi
          fi
          # A final sealed shard is immutable evidence under the active
          # collection contract. Always reuse a complete aggregate/timing
          # pair; an operator must never need an extra resume flag merely to
          # prevent an installation replay from overwriting hours of sealed
          # measurements. Interrupted `.inprogress` files are not accepted by
          # this test and are safely replaced by the owning job.
          if [[ -s "${partial}" && -s "${timing_partial}" ]]; then
            continue
          fi
          job_format_specs+=("${format_spec}")
          job_shapes+=("${shape}")
          job_ns+=("${n}")
          job_ks+=("${k}")
          job_runtime_isas+=("${runtime_isa}")
          job_regime_bins+=("${regime_bin}")
          job_partials+=("${partial}")
          job_timing_partials+=("${timing_partial}")
        done < <(csv_values "${cpu_measurement_shapes}")
        job_group_ends+=("${#job_shapes[@]}")
      done
    done

    local job_group_index group_start group_end
    local batch_start batch_end batch_job_index
    local cpu_batches_run=0
    local cpu_collection_limited=0
    for ((job_group_index = 0;
          job_group_index < ${#job_group_starts[@]};
          ++job_group_index)); do
      group_start=${job_group_starts[job_group_index]}
      group_end=${job_group_ends[job_group_index]}
      for ((batch_start = group_start;
            batch_start < group_end;
            batch_start += cpu_measurement_lanes)); do
        if (( cpu_batch_limit > 0 && cpu_batches_run >= cpu_batch_limit )); then
          cpu_collection_limited=1
          break 2
        fi
        batch_end=$((batch_start + cpu_measurement_lanes))
        if (( batch_end > group_end )); then
          batch_end=${group_end}
        fi

        local mpi_measurement_command=(
          mpirun
          --bind-to socket
          --map-by socket
          --mca mpi_leave_pinned 1
          --mca btl_vader_single_copy_mechanism none
          --mca orte_allowed_exit_without_sync 1
        )
        local batch_inprogress_paths=()
        for ((batch_job_index = batch_start;
              batch_job_index < batch_end;
              ++batch_job_index)); do
          if (( batch_job_index > batch_start )); then
            mpi_measurement_command+=(":")
          fi
          local partial_inprogress="${job_partials[batch_job_index]}.inprogress"
          local timing_inprogress="${job_timing_partials[batch_job_index]}.inprogress"
          batch_inprogress_paths+=(
            "${partial_inprogress}"
            "${timing_inprogress}"
          )
          mpi_measurement_command+=(
            -np 1
            env
            "LLAMINAR_ISA_LEVEL=${job_runtime_isas[batch_job_index]}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=${job_format_specs[batch_job_index]}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_M=${cpu_sweep_m_values}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=${job_shapes[batch_job_index]}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_N=${job_ns[batch_job_index]}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_K=${job_ks[batch_job_index]}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_MAX_CASES=${cpu_max_cases}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_THREADS=${cpu_threads}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=${cpu_warmup}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=${cpu_iters}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=${partial_inprogress}"
            "LLAMINAR_CPU_NVNNI_VERIFIER_TIMING_CSV=${timing_inprogress}"
            "${job_regime_bins[batch_job_index]}"
            "--gtest_filter=*TrainerCsv_StrongVerifierRows_AllFormats"
          )
        done
        run_cmd rm -f "${batch_inprogress_paths[@]}"
        run_cmd \
          "OMP_NUM_THREADS=${cpu_threads}" \
          "OMP_PLACES=cores" \
          "OMP_PROC_BIND=close" \
          "OMP_DYNAMIC=false" \
          "OMP_NESTED=false" \
          "HWLOC_COMPONENTS=-gl,-opencl" \
          "OMPI_MCA_mpi_leave_pinned=1" \
          "OMPI_MCA_btl_vader_single_copy_mechanism=none" \
          "${mpi_measurement_command[@]}"
        for ((batch_job_index = batch_start;
              batch_job_index < batch_end;
              ++batch_job_index)); do
          run_cmd mv \
            "${job_partials[batch_job_index]}.inprogress" \
            "${job_partials[batch_job_index]}"
          run_cmd mv \
            "${job_timing_partials[batch_job_index]}.inprogress" \
            "${job_timing_partials[batch_job_index]}"
        done
        cpu_batches_run=$((cpu_batches_run + 1))
      done
    done
    if (( cpu_collection_limited )); then
      printf 'CPU NativeVNNI collection checkpoint: completed %d pending MPMD batch(es); rerun with --resume-cpu-partials to continue in %s\n' \
        "${cpu_batches_run}" "${output_dir}"
      finish_backend_collection_target
      return
    fi
    if [[ "${measurement_profile}" == "production" ]]; then
      combine_csvs "${cpu_development_csv}" "${development_partials[@]}"
      combine_csvs \
        "${cpu_development_timing_csv}" \
        "${development_timing_partials[@]}"
    else
      combine_csvs "${cpu_csv}" "${partials[@]}"
      combine_csvs "${cpu_timing_csv}" "${timing_partials[@]}"
    fi
  fi

  if (( partitioned_measurement )); then
    finish_backend_collection_target
    return
  fi

  local cpu_require_args=()
  if (( cpu_require_policy_keys )); then
    cpu_require_args=(
      --require-inventory-formats "${cpu_formats}"
      --require-inventory-shapes "${cpu_sweep_shapes}"
      --require-inventory-m-values "${cpu_sweep_m_values}"
    )
  fi

  if [[ "${measurement_profile}" == "production" ]]; then
    local cpu_run_id
    local reuse_cpu_grouped_common=0
    if [[ -s "${cpu_grouped_development_run_id_file}" ]]; then
      IFS= read -r cpu_run_id < "${cpu_grouped_development_run_id_file}"
    elif [[ -s "${cpu_common_csv}" ]] &&
         (( resume_cpu_partials || skip_sweep )); then
      cpu_run_id="$(csv_unique_field "${cpu_common_csv}" run_id)"
      if (( ! dry_run )); then
        printf '%s\n' "${cpu_run_id}" > \
          "${cpu_grouped_development_run_id_file}.inprogress"
        mv "${cpu_grouped_development_run_id_file}.inprogress" \
          "${cpu_grouped_development_run_id_file}"
      fi
    else
      cpu_run_id="${timestamp}-cpu-${profile}-development"
      if (( ! dry_run )); then
        printf '%s\n' "${cpu_run_id}" > \
          "${cpu_grouped_development_run_id_file}.inprogress"
        mv "${cpu_grouped_development_run_id_file}.inprogress" \
          "${cpu_grouped_development_run_id_file}"
      fi
    fi
    if [[ -z "${cpu_run_id}" || "${cpu_run_id}" == *$'\n'* ]]; then
      echo "error: invalid CPU grouped development run ID checkpoint" >&2
      exit 2
    fi
    if [[ -s "${cpu_common_csv}" ]] &&
       (( resume_cpu_partials || skip_sweep )); then
      reuse_cpu_grouped_common=1
      local -a cpu_grouped_provenance=()
      mapfile -t cpu_grouped_provenance < <(
        csv_first_cpu_provenance "${cpu_common_csv}"
      )
      if (( ${#cpu_grouped_provenance[@]} != 8 )); then
        echo "error: invalid CPU grouped provenance checkpoint" >&2
        exit 2
      fi
      if [[ "${cpu_run_id}" != "${cpu_grouped_provenance[0]}" ]]; then
        echo "error: CPU grouped run-ID checkpoint disagrees with common corpus" >&2
        exit 2
      fi
      cpu_git_revision="${cpu_grouped_provenance[1]}"
      cpu_build_id="${cpu_grouped_provenance[2]}"
      cpu_compiler_id="${cpu_grouped_provenance[3]}"
      cpu_arch_class="${cpu_grouped_provenance[4]}"
      cpu_device_name="${cpu_grouped_provenance[5]}"
      cpu_driver_runtime="${cpu_grouped_provenance[6]}"
      cpu_serial_policy_hash="${cpu_grouped_provenance[7]}"
      if [[ "${cpu_serial_policy_hash}" != \
            "${cpu_measurement_serial_policy_hash}" ]]; then
        echo "error: fresh CPU grouped seal cannot certify a changed serial arithmetic policy" >&2
        exit 2
      fi
      printf 'Reusing authenticated CPU grouped common corpus: %s\n' \
        "${cpu_common_csv}"
      printf 'Fit-only CPU grouped replay retains measured build provenance: %s\n' \
        "${cpu_build_id}"
    fi
    if (( ! reuse_cpu_grouped_common )); then
      run_cmd python3 "${cpu_generator}" \
        --input "${cpu_development_csv}" \
        --timing-sidecar "${cpu_development_timing_csv}" \
        --output "${cpu_provisional_inc}" \
        --common-observations "${cpu_common_csv}" \
        --adapt-only \
        --profile production \
        --run-id "${cpu_run_id}" \
        --git-revision "${cpu_git_revision}" \
        --build-id "${cpu_build_id}" \
        --compiler-id "${cpu_compiler_id}" \
        --architecture-class "${cpu_arch_class}" \
        --device-name "${cpu_device_name}" \
        --driver-runtime "${cpu_driver_runtime}" \
        --serial-m1-policy-hash "${cpu_serial_policy_hash}" \
        --shape-manifest "${shape_manifest_path}" \
        "${cpu_fixed_timing_gate_args[@]}"
    fi
    collect_backend_profiler_evidence \
      cpu "${cpu_common_csv}" "${cpu_avx2_sweep_bin}" \
      "${cpu_profiler_requests}" "${cpu_profiler_evidence}" \
      "${cpu_profiler_features}" "${cpu_profiler_raw}"
    run_cpu_grouped_paired_refinement "${cpu_common_csv}"

    local -a cpu_grouped_burned_args=()
    local grouped_burned_index
    for grouped_burned_index in "${!cpu_grouped_burned_sealed_plans[@]}"; do
      cpu_grouped_burned_args+=(
        --burned-sealed-plan-json
        "${cpu_grouped_burned_sealed_plans[grouped_burned_index]}"
        --burned-sealed-paired-dir
        "${cpu_grouped_burned_sealed_paired_dirs[grouped_burned_index]}"
      )
    done

    local -a cpu_frozen_args=(
      python3 "${cpu_generator}"
      --input "${cpu_development_csv}"
      --timing-sidecar "${cpu_development_timing_csv}"
      --output "${cpu_frozen_inc}"
      --summary "${cpu_summary}"
      --common-observations "${cpu_common_csv}"
      --reuse-development-common
      --policy-json "${cpu_frozen_policy_json}"
      --sealed-plan-json "${cpu_grouped_sealed_plan_json}"
      --sealed-request-dir "${cpu_grouped_sealed_request_dir}"
      --freeze-generic
      --profile production
      --run-id "${cpu_run_id}"
      --git-revision "${cpu_git_revision}"
      --build-id "${cpu_build_id}"
      --compiler-id "${cpu_compiler_id}"
      --architecture-class "${cpu_arch_class}"
      --device-name "${cpu_device_name}"
      --driver-runtime "${cpu_driver_runtime}"
      --serial-m1-policy-hash "${cpu_serial_policy_hash}"
      --shape-manifest "${shape_manifest_path}"
      --fit-cache-dir "${cpu_grouped_fit_cache_dir}"
      --development-profiler-requests "${cpu_profiler_requests}"
      --development-profiler-evidence "${cpu_profiler_evidence}"
      --development-profiler-observations "${cpu_common_csv}"
      --generic-max-leaves "${cpu_grouped_max_leaves}"
      --sealed-build-id "${cpu_measurement_build_id}"
      "${cpu_fixed_timing_gate_args[@]}"
    )
    cpu_frozen_args+=("${cpu_grouped_burned_args[@]}")
    local cpu_grouped_development_paired_path
    for cpu_grouped_development_paired_path in \
        "${cpu_grouped_paired_evidence[@]}"; do
      cpu_frozen_args+=(
        --paired-development-csv
        "${cpu_grouped_development_paired_path}"
      )
    done
    run_cmd "${cpu_frozen_args[@]}"

    local cpu_grouped_sealed_paired_csvs=()
    run_cpu_sealed_paired_shards \
      "grouped-decode" "${cpu_grouped_sealed_request_dir}" \
      cpu_grouped_sealed_paired_csvs

    local -a cpu_certify_args=(
      python3 "${cpu_generator}"
      --development-input "${cpu_development_csv}"
      --development-timing-sidecar "${cpu_development_timing_csv}"
      --frozen-policy-json "${cpu_frozen_policy_json}"
      --sealed-plan-json "${cpu_grouped_sealed_plan_json}"
      --policy-json "${cpu_policy_json}"
      --certification-diagnostic "${cpu_grouped_certification_diagnostic}"
      --output "${cpu_inc}"
      --summary "${cpu_summary}"
      --common-observations "${cpu_common_csv}"
      --reuse-development-common
      --certify-generic
      --profile production
      --run-id "${cpu_run_id}"
      --git-revision "${cpu_git_revision}"
      --build-id "${cpu_build_id}"
      --compiler-id "${cpu_compiler_id}"
      --architecture-class "${cpu_arch_class}"
      --device-name "${cpu_device_name}"
      --driver-runtime "${cpu_driver_runtime}"
      --serial-m1-policy-hash "${cpu_serial_policy_hash}"
      --shape-manifest "${shape_manifest_path}"
      --fit-cache-dir "${cpu_grouped_fit_cache_dir}"
      --development-profiler-requests "${cpu_profiler_requests}"
      --development-profiler-evidence "${cpu_profiler_evidence}"
      --development-profiler-observations "${cpu_common_csv}"
      --generic-max-leaves "${cpu_grouped_max_leaves}"
      --sealed-build-id "${cpu_measurement_build_id}"
      "${cpu_fixed_timing_gate_args[@]}"
    )
    cpu_certify_args+=("${cpu_grouped_burned_args[@]}")
    for cpu_grouped_development_paired_path in \
        "${cpu_grouped_paired_evidence[@]}"; do
      cpu_certify_args+=(
        --paired-development-csv
        "${cpu_grouped_development_paired_path}"
      )
    done
    cpu_certify_args+=("${cpu_require_args[@]}")
    local cpu_grouped_paired_path
    cpu_certify_args+=(
      --sealed-paired-dir "${cpu_grouped_sealed_request_dir}"
    )
    run_cmd "${cpu_certify_args[@]}"
    run_cmd python3 "${dispatch_validator}" "${cpu_inc}"
    run_cmd \
      "PYTHONPATH=${shape_manifest_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${cpu_policy_json}" \
      --include "${cpu_inc}"
    reuse_identical_backend_profiler_evidence \
      "${cpu_profiler_requests}" "${cpu_profiler_evidence}" \
      "${cpu_profiler_features}" \
      "${cpu_final_profiler_requests}" "${cpu_final_profiler_evidence}" \
      "${cpu_final_profiler_features}"

    if (( install )); then
      local cpu_install_target="${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
      run_cmd cp "${cpu_inc}" "${cpu_install_target}.inprogress"
      run_cmd mv "${cpu_install_target}.inprogress" "${cpu_install_target}"
    fi
    finish_backend_collection_target
    return
  fi

  local cpu_generator_args=(
    python3 "${cpu_generator}"
    --input "${cpu_csv}"
    --timing-sidecar "${cpu_timing_csv}"
    --output "${cpu_inc}"
    --summary "${cpu_summary}"
    --common-observations "${cpu_common_csv}"
    --require-isa-matrix
    --profile "${measurement_profile}"
    --run-id "${timestamp}-cpu-${profile}"
    --git-revision "${cpu_git_revision}"
    --build-id "${cpu_build_id}"
    --compiler-id "${cpu_compiler_id}"
    --architecture-class "${cpu_arch_class}"
    --device-name "${cpu_device_name}"
    --driver-runtime "${cpu_driver_runtime}"
    --serial-m1-policy-hash "${cpu_serial_policy_hash}"
    --shape-manifest "${shape_manifest_path}"
    "${cpu_fixed_timing_gate_args[@]}"
  )
  if [[ "${measurement_profile}" == "production" ||
        "${measurement_profile}" == "family-smoke" ]]; then
    cpu_generator_args+=(--require-complete)
  fi
  cpu_generator_args+=("${cpu_require_args[@]}")
  run_cmd "${cpu_generator_args[@]}"

  run_cmd python3 "${dispatch_validator}" "${cpu_inc}"
  collect_backend_profiler_evidence \
    cpu "${cpu_common_csv}" "${cpu_avx2_sweep_bin}" \
    "${cpu_profiler_requests}" "${cpu_profiler_evidence}" \
    "${cpu_profiler_features}" "${cpu_profiler_raw}"

  if (( install )); then
    run_cmd cp "${cpu_inc}" "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
  fi
  finish_backend_collection_target
}

refresh_cpu_prefill() {
  begin_backend_collection_target "CPU prefill"
  if [[ "${measurement_profile}" == "production" ||
        "${measurement_profile}" == "partial-production" ]]; then
    require_cpu_performance_governor
  fi
  require_executable "${cpu_avx2_sweep_bin}"
  require_executable "${cpu_avx512_sweep_bin}"
  require_gtest_case \
    "${cpu_avx2_sweep_bin}" "TrainerCsv_StrongPrefill_AllFormats"
  require_gtest_case \
    "${cpu_avx512_sweep_bin}" "TrainerCsv_StrongPrefill_AllFormats"
  require_gtest_case \
    "${cpu_avx2_sweep_bin}" "TrainerCsv_PrefillSerialRouteManifest"
  require_gtest_case \
    "${cpu_avx512_sweep_bin}" "TrainerCsv_PrefillSerialRouteManifest"
  local cpu_git_revision cpu_build_id cpu_compiler_id cpu_arch_class
  local cpu_device_name cpu_driver_runtime cpu_serial_policy_hash
  local cpu_measurement_git_revision cpu_measurement_build_id
  local cpu_measurement_compiler_id cpu_measurement_arch_class
  local cpu_measurement_device_name cpu_measurement_driver_runtime
  local cpu_measurement_serial_policy_hash
  cpu_measurement_git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
  cpu_measurement_build_id="$(sha256_file_set \
    "${cpu_avx2_sweep_bin}" "${cpu_avx512_sweep_bin}")"
  cpu_measurement_compiler_id="$(${CXX:-c++} --version | first_output_line)"
  cpu_measurement_arch_class="$(cpu_architecture_class)"
  cpu_measurement_device_name="$(cpu_lscpu_field "Model name")"
  cpu_measurement_driver_runtime="linux-$(uname -r)-openmp"
  cpu_measurement_serial_policy_hash="$(cpu_serial_arithmetic_contract_hash)"
  cpu_git_revision="${cpu_measurement_git_revision}"
  cpu_build_id="${cpu_measurement_build_id}"
  cpu_compiler_id="${cpu_measurement_compiler_id}"
  cpu_arch_class="${cpu_measurement_arch_class}"
  cpu_device_name="${cpu_measurement_device_name}"
  cpu_driver_runtime="${cpu_measurement_driver_runtime}"
  cpu_serial_policy_hash="${cpu_measurement_serial_policy_hash}"

  # Fit-only replay consumes immutable measured evidence; rebuilding a test
  # binary to improve route-probe or orchestration code must not relabel that
  # evidence as though the new executable produced its timings. Recover the
  # exact measurement provenance from the authenticated compact checkpoint.
  # Live collection keeps the freshly derived identities above.
  if (( skip_sweep )) && [[ -s "${cpu_prefill_common_csv}" ]]; then
    cpu_git_revision="$(
      csv_unique_field "${cpu_prefill_common_csv}" git_revision
    )"
    cpu_build_id="$(
      csv_unique_field_prefix \
        "${cpu_prefill_common_csv}" build_id '|cpu_isa='
    )"
    cpu_compiler_id="$(
      csv_unique_field "${cpu_prefill_common_csv}" compiler_id
    )"
    cpu_arch_class="$(
      csv_unique_field_prefix \
        "${cpu_prefill_common_csv}" architecture_class '|build='
    )"
    cpu_device_name="$(
      csv_unique_field "${cpu_prefill_common_csv}" device_name
    )"
    cpu_driver_runtime="$(
      csv_unique_field "${cpu_prefill_common_csv}" driver_runtime
    )"
    cpu_serial_policy_hash="$(
      csv_unique_field \
        "${cpu_prefill_common_csv}" serial_m1_policy_hash
    )"
    printf 'Fit-only CPU prefill replay retains measured build provenance: %s\n' \
      "${cpu_build_id}"
    if (( collect_cpu_prefill_sealed_after_freeze )) &&
       [[ "${cpu_serial_policy_hash}" != "${cpu_measurement_serial_policy_hash}" ]]; then
      echo "error: fresh CPU prefill sealed collection cannot certify a changed production serial policy" >&2
      exit 2
    fi
  fi
  # Keep the immutable development identity available after the post-freeze
  # collector switches the active variables to the binaries measuring the
  # fresh holdout. Certification authenticates both partitions independently;
  # neither identity is permitted to stand in for the other.
  local cpu_development_git_revision="${cpu_git_revision}"
  local cpu_development_build_id="${cpu_build_id}"
  local cpu_development_compiler_id="${cpu_compiler_id}"
  local cpu_development_arch_class="${cpu_arch_class}"
  local cpu_development_device_name="${cpu_device_name}"
  local cpu_development_driver_runtime="${cpu_driver_runtime}"
  local cpu_development_serial_policy_hash="${cpu_serial_policy_hash}"

  # Ask the production C++ tile planner which serial arithmetic family owns
  # every shape.  This probe launches no GEMM work and keeps host cache/thread
  # policy out of the Python covering-array implementation.
  local route_probe_specs=(
    "avx2-build.avx2-runtime|avx2|${cpu_avx2_sweep_bin}"
    "avx512-build.avx2-runtime|avx2|${cpu_avx512_sweep_bin}"
    "avx512-build.avx512-runtime|avx512|${cpu_avx512_sweep_bin}"
  )
  local cpu_prefill_route_args=()
  local route_probe_spec route_regime route_runtime_isa route_binary
  local route_manifest route_manifest_inprogress
  if (( ${#cpu_prefill_fit_target_route_manifests[@]} > 0 )); then
    # A fit-only recipe already authenticated the timing-free production route
    # generation that owns every plan. Re-probing current binaries here would
    # perform work before preflight and could silently substitute a newer route
    # identity for historical evidence.
    for route_manifest in \
        "${cpu_prefill_fit_target_route_manifests[@]}"; do
      cpu_prefill_route_args+=(--route-manifest "${route_manifest}")
    done
  else
    for route_probe_spec in "${route_probe_specs[@]}"; do
      IFS='|' read -r route_regime route_runtime_isa route_binary <<< \
        "${route_probe_spec}"
      route_manifest="${output_dir}/cpu_prefill_serial_route.${route_regime}.csv"
      route_manifest_inprogress="${route_manifest}.inprogress"
      run_cmd rm -f "${route_manifest_inprogress}"
      run_cmd \
        "OMP_NUM_THREADS=${cpu_threads}" \
        "OMP_PLACES=cores" \
        "OMP_PROC_BIND=close" \
        "OMP_DYNAMIC=false" \
        "LLAMINAR_ISA_LEVEL=${route_runtime_isa}" \
        "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_CSV=${route_manifest_inprogress}" \
        "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_ISA_REGIME=${route_regime}" \
        "${route_binary}" \
        "--gtest_filter=*TrainerCsv_PrefillSerialRouteManifest"
      run_cmd mv "${route_manifest_inprogress}" "${route_manifest}"
      if (( ! dry_run )) || [[ -s "${route_manifest}" ]]; then
        cpu_prefill_route_args+=(--route-manifest "${route_manifest}")
      fi
    done
  fi

  local selected_shapes=()
  case "${profile}" in
    quick)
      selected_shapes=("0.5B_AttnOut" "7B_FFN_Up")
      ;;
    family-smoke)
      selected_shapes=("7B_FFN_Up" "14B_FFN_Up" "32B_FFN_Up")
      ;;
    all)
      selected_shapes=("${cpu_prefill_shape_names[@]}")
      ;;
  esac
  if (( shapes_explicit )); then
    selected_shapes=()
    while IFS= read -r shape; do
      if [[ -z "${cpu_prefill_shape_n[${shape}]+present}" ]]; then
        echo "error: ${shape} is not in the CPU prefill matrix" >&2
        exit 2
      fi
      selected_shapes+=("${shape}")
    done < <(csv_values "${shapes}")
  fi

  # Pair similarly expensive jobs by M*N*K rather than N*K alone. This keeps
  # one socket from waiting at MPI_Finalize while its peer times a much deeper
  # small-model prefill cell.
  local weighted_shapes=()
  local shape shape_m max_m work
  for shape in "${selected_shapes[@]}"; do
    shape_m="${cpu_prefill_shape_m[${shape}]}"
    if [[ "${profile}" != "all" ]]; then
      case "${profile}" in
        quick)
          # Quick CPU-prefill diagnostics must still exercise a depth owned by
          # the selected geometry's production tier. Sub-14B overlays begin at
          # M=512; hard-coding M=64 here would silently reintroduce a launch the
          # bounded production inventory intentionally removed.
          shape_m="$(printf '%s' "${shape_m}" | tr ',' '\n' | head -n 1)"
          ;;
        family-smoke)
          max_m="$(printf '%s' "${shape_m}" | tr ',' '\n' | tail -n 1)"
          shape_m="${max_m}"
          ;;
      esac
    fi
    if (( shapes_explicit )); then
      shape_m="${normalized_m_values}"
      if [[ ",${shape_m}," == *,1,* ]]; then
        echo "error: CPU prefill M values must all be greater than one" >&2
        exit 2
      fi
    fi
    max_m="$(printf '%s' "${shape_m}" | tr ',' '\n' | tail -n 1)"
    work=$((cpu_prefill_shape_n[${shape}] * cpu_prefill_shape_k[${shape}] * max_m))
    weighted_shapes+=("$(printf '%030d\t%s\t%s' "${work}" "${shape}" "${shape_m}")")
  done
  local sorted_shape_records=()
  mapfile -t sorted_shape_records < <(
    printf '%s\n' "${weighted_shapes[@]}" |
      sort -t $'\t' -k1,1nr -k2,2 |
      cut -f2-
  )

  local contract_path="${output_dir}/cpu_prefill_collection_contract.sha256"
  local matrix_digest training_plan_digest split_manifest_digest
  local contract_payload contract_digest
  local prefill_format_shards=0
  local cpu_prefill_mpi_round_sync=0
  local regime_specs=(
    "avx2-build.avx2-runtime|avx2|${cpu_avx2_sweep_bin}"
    "avx512-build.avx2-runtime|avx2|${cpu_avx512_sweep_bin}"
    "avx512-build.avx512-runtime|avx512|${cpu_avx512_sweep_bin}"
  )
  declare -A regime_runtime_isa=()
  declare -A regime_binary=()
  local regime_spec regime_name runtime_isa regime_bin
  for regime_spec in "${regime_specs[@]}"; do
    IFS='|' read -r regime_name runtime_isa regime_bin <<< "${regime_spec}"
    regime_runtime_isa["${regime_name}"]="${runtime_isa}"
    regime_binary["${regime_name}"]="${regime_bin}"
  done
  if [[ -n "${cpu_prefill_candidate_expansion_source_aggregate}" ]]; then
    local expansion_plan="${output_dir}/cpu_prefill_candidate_expansion.v4.json"
    local expansion_csv="${output_dir}/cpu_prefill_sweep.candidate-expansion.v4.csv"
    local expansion_timing_csv="${output_dir}/cpu_prefill_sweep.candidate-expansion.v4.timing.csv"
    local expansion_status=0
    collect_cpu_prefill_candidate_expansion \
      "${cpu_prefill_candidate_expansion_source_aggregate}" \
      "${cpu_prefill_candidate_expansion_source_timing}" \
      "${expansion_plan}" "${expansion_csv}" \
      "${expansion_timing_csv}" || expansion_status=$?
    if (( expansion_status == 3 )); then
      finish_backend_collection_target
      return
    elif (( expansion_status != 0 )); then
      return "${expansion_status}"
    fi
    printf 'CPU prefill candidate expansion collected: %s\n' \
      "${expansion_csv}"
    printf 'Use --candidate-expansion-plan-json %s --candidate-expansion-input %s --candidate-expansion-timing-sidecar %s for adaptation and fitting.\n' \
      "${expansion_plan}" "${expansion_csv}" "${expansion_timing_csv}"
    finish_backend_collection_target
    return
  fi
  if [[ -n "${collect_cpu_prefill_development_lineage_plan}" ]]; then
    # The authenticated lineage plan and target split digest, rather than a
    # version embedded in a filename, own the generation identity. Stable
    # names avoid stale v8/v9 labels whenever the declarative split advances.
    local lineage_increment_csv="${output_dir}/cpu_prefill_sweep.development-lineage.increment.csv"
    local lineage_increment_timing_csv="${output_dir}/cpu_prefill_sweep.development-lineage.increment.timing.csv"
    local lineage_development_csv="${output_dir}/cpu_prefill_sweep.development-lineage.combined.csv"
    local lineage_development_timing_csv="${output_dir}/cpu_prefill_sweep.development-lineage.combined.timing.csv"
    local -a lineage_source_args=(
      --source-aggregate "${cpu_prefill_lineage_source_aggregate}"
      --source-timing "${cpu_prefill_lineage_source_timing}"
      --source-split-manifest "${cpu_prefill_lineage_source_split_manifest}"
      --target-split-manifest "${cpu_prefill_split_manifest_path}"
    )
    local source_route_manifest
    for source_route_manifest in \
        "${cpu_prefill_lineage_source_route_manifests[@]}"; do
      lineage_source_args+=(
        --source-route-manifest "${source_route_manifest}"
      )
    done
    local source_refinement_plan
    for source_refinement_plan in \
        "${cpu_prefill_lineage_source_refinement_plans[@]}"; do
      lineage_source_args+=(
        --source-refinement-plan "${source_refinement_plan}"
      )
    done
    local source_refinement_split_manifest
    for source_refinement_split_manifest in \
        "${cpu_prefill_lineage_source_refinement_split_manifests[@]}"; do
      lineage_source_args+=(
        --source-refinement-split-manifest \
          "${source_refinement_split_manifest}"
      )
    done

    if (( dry_run )); then
      printf 'dry-run: authenticate CPU prefill development lineage %q\n' \
        "${collect_cpu_prefill_development_lineage_plan}"
    else
      local required_path
      for required_path in \
          "${cpu_prefill_lineage_source_aggregate}" \
          "${cpu_prefill_lineage_source_timing}" \
          "${cpu_prefill_lineage_source_split_manifest}" \
          "${cpu_prefill_lineage_source_route_manifests[@]}" \
          "${cpu_prefill_lineage_source_refinement_split_manifests[@]}" \
          "${cpu_prefill_lineage_source_refinement_plans[@]}"; do
        if [[ ! -s "${required_path}" ]]; then
          echo "error: CPU prefill development lineage source is missing: ${required_path}" >&2
          return 2
        fi
      done
      if [[ ! -s "${collect_cpu_prefill_development_lineage_plan}" ]]; then
        # Target route manifests are binary-owned evidence, so a lineage plan
        # cannot be authored honestly until this transaction has probed the
        # exact trainer binaries above. Build a missing plan once from those
        # manifests; resumed runs authenticate the immutable bytes instead.
        PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m "${cpu_prefill_development_lineage_module}" \
            --build \
            --output "${collect_cpu_prefill_development_lineage_plan}" \
            "${lineage_source_args[@]}" "${cpu_prefill_route_args[@]}"
      fi
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m \
          tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
          --input "${cpu_prefill_lineage_source_aggregate}" \
          --timing-sidecar "${cpu_prefill_lineage_source_timing}"
    fi

    local lineage_status=0
    collect_cpu_prefill_development_lineage_increment \
      "${collect_cpu_prefill_development_lineage_plan}" \
      "${lineage_increment_csv}" \
      "${lineage_increment_timing_csv}" || lineage_status=$?
    if (( lineage_status == 3 )); then
      finish_backend_collection_target
      return
    elif (( lineage_status != 0 )); then
      return "${lineage_status}"
    fi
    # A lineage migration crosses immutable harness generations. The column
    # names remain the semantic contract, but their physical order can change
    # as newly collected timing fields are grouped more readably. Merge by
    # field name here; the strict combiner above remains appropriate for the
    # same-generation increment shards themselves.
    combine_compatible_csvs "${lineage_development_csv}" \
      "${cpu_prefill_lineage_source_aggregate}" "${lineage_increment_csv}"
    combine_compatible_csvs "${lineage_development_timing_csv}" \
      "${cpu_prefill_lineage_source_timing}" \
      "${lineage_increment_timing_csv}"
    if (( ! dry_run )); then
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m \
          tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
          --input "${lineage_development_csv}" \
          --timing-sidecar "${lineage_development_timing_csv}"
    fi
    printf 'CPU prefill development lineage collected and published: %s\n' \
      "${lineage_development_csv}"
    finish_backend_collection_target
    return
  fi
  if [[ -n "${collect_cpu_prefill_refinement_plan}" ]]; then
    local refinement_source_csv refinement_source_timing_csv
    if [[ -n "${cpu_prefill_refinement_source_aggregate}" ]]; then
      # A split-lineage migration breaks the legacy round-number-to-filename
      # relation. Keep the authenticated migrated aggregate explicit so a new
      # refinement cannot accidentally fork from the pre-migration corpus.
      refinement_source_csv="${cpu_prefill_refinement_source_aggregate}"
      refinement_source_timing_csv="${cpu_prefill_refinement_source_timing}"
    elif (( cpu_prefill_refinement_round == 1 )); then
      refinement_source_csv="${cpu_prefill_development_csv}"
      refinement_source_timing_csv="${cpu_prefill_development_timing_csv}"
    else
      local prior_refinement_round=$((cpu_prefill_refinement_round - 1))
      refinement_source_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${prior_refinement_round}.csv"
      refinement_source_timing_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${prior_refinement_round}.timing.csv"
    fi
    local refinement_round_csv="${output_dir}/cpu_prefill_sweep.generic-refinement-r${cpu_prefill_refinement_round}.csv"
    local refinement_round_timing_csv="${output_dir}/cpu_prefill_sweep.generic-refinement-r${cpu_prefill_refinement_round}.timing.csv"
    local refinement_development_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${cpu_prefill_refinement_round}.csv"
    local refinement_development_timing_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${cpu_prefill_refinement_round}.timing.csv"

    if (( dry_run )); then
      printf 'dry-run: authenticate CPU prefill refinement source %q %q %q\n' \
        "${collect_cpu_prefill_refinement_plan}" \
        "${cpu_prefill_refinement_source_fit}" \
        "${cpu_prefill_refinement_source_observations}"
      printf 'dry-run: validate CPU prefill refinement predecessor %q %q\n' \
        "${refinement_source_csv}" "${refinement_source_timing_csv}"
    else
      for required_path in \
          "${collect_cpu_prefill_refinement_plan}" \
          "${cpu_prefill_refinement_source_fit}" \
          "${cpu_prefill_refinement_source_observations}" \
          "${refinement_source_csv}" \
          "${refinement_source_timing_csv}"; do
        if [[ ! -s "${required_path}" ]]; then
          echo "error: existing CPU prefill refinement source is missing: ${required_path}" >&2
          return 2
        fi
      done
      PYTHONPATH="${shape_manifest_python_root}" \
        python3 -m \
          tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
          --input "${refinement_source_csv}" \
          --timing-sidecar "${refinement_source_timing_csv}"
    fi

    local refinement_status=0
    collect_cpu_prefill_generic_refinement_round \
      "${collect_cpu_prefill_refinement_plan}" \
      "${cpu_prefill_refinement_round}" \
      "${refinement_round_csv}" "${refinement_round_timing_csv}" \
      "${cpu_prefill_refinement_source_fit}" \
      "${cpu_prefill_refinement_source_observations}" || refinement_status=$?
    if (( refinement_status == 3 )); then
      finish_backend_collection_target
      return
    elif (( refinement_status != 0 )); then
      return "${refinement_status}"
    fi
    combine_compatible_csvs "${refinement_development_csv}" \
      "${refinement_source_csv}" "${refinement_round_csv}"
    combine_csvs "${refinement_development_timing_csv}" \
      "${refinement_source_timing_csv}" "${refinement_round_timing_csv}"
    printf 'CPU prefill generic refinement round %d collected and published: %s\n' \
      "${cpu_prefill_refinement_round}" "${refinement_development_csv}"
    finish_backend_collection_target
    return
  fi
  if (( cpu_format_shards )) || [[ "${measurement_profile}" == "production" ]]; then
    prefill_format_shards=1
  fi
  if [[ "${profile}" == "all" ]] && (( ! shapes_explicit )); then
    cpu_prefill_mpi_round_sync=1
  fi
  matrix_digest="$(
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m native_vnni_dispatch.prefill_matrix --records |
      sha256sum | awk '{print $1}'
  )"
  training_plan_digest="$(
    PYTHONPATH="${shape_manifest_python_root}" \
      python3 -m "${cpu_prefill_training_plan_module}" --records \
        "${cpu_prefill_route_args[@]}" |
      sha256sum | awk '{print $1}'
  )"
  split_manifest_digest="$(
    sha256sum "${cpu_prefill_split_manifest_path}" | awk '{print $1}'
  )"
  contract_payload="$(printf '%s\n' \
    "schema=cpu-native-vnni-prefill-collection-v17" \
    "profile=${measurement_profile}" \
    "shape_records=$(IFS=';'; printf '%s' "${sorted_shape_records[*]}")" \
    "formats=${cpu_formats}" \
    "threads=${cpu_threads}" \
    "measurement_lanes=${cpu_measurement_lanes}" \
    "mpi_round_coordination=${cpu_prefill_mpi_round_sync}" \
    "frequency_governor=performance" \
    "format_shards=${prefill_format_shards}" \
    "warmups=${cpu_warmup}" \
    "warmup_budget_us=${cpu_prefill_warmup_budget_us}" \
    "transition_warmup_budget_us=${cpu_prefill_transition_warmup_budget_us}" \
    "transition_warmup_latency_multiplier=${cpu_prefill_transition_warmup_latency_multiplier}" \
    "transition_warmup_budget_ceiling_us=${cpu_prefill_transition_warmup_budget_ceiling_us}" \
    "warmup_round_timeout_us=${cpu_prefill_warmup_round_timeout_us}" \
    "samples=${cpu_iters}" \
    "minimum_samples=${cpu_prefill_min_iters}" \
    "maximum_samples=${cpu_prefill_max_iters}" \
    "candidate_expansion_maximum_samples=${cpu_prefill_candidate_expansion_max_iters}" \
    "candidate_expansion_timing_budget_us=${cpu_prefill_candidate_expansion_timing_budget_us}" \
    "timing_budget_us=${cpu_prefill_timing_budget_us}" \
    "median_stability=${cpu_prefill_median_stability}" \
    "max_cases=${cpu_max_cases}" \
    "collection_build_id=${cpu_measurement_build_id}" \
    "serial_policy_hash=${cpu_serial_policy_hash}" \
    "prefill_matrix_hash=${matrix_digest}" \
    "training_plan_hash=${training_plan_digest}" \
    "sealed_split_manifest_hash=${split_manifest_digest}")"
  contract_digest="$(
    printf '%s' "${contract_payload}" | sha256sum | awk '{print $1}'
  )"
  if (( skip_cpu_prefill_baseline )); then
    if (( dry_run )); then
      printf 'dry-run: require complete CPU prefill baseline %q %q\n' \
        "${cpu_prefill_csv}" "${cpu_prefill_timing_csv}"
    elif [[ ! -s "${cpu_prefill_csv}" ||
            ! -s "${cpu_prefill_timing_csv}" ]]; then
      echo "error: --skip-cpu-prefill-baseline requires complete baseline CSVs in ${output_dir}" >&2
      exit 2
    fi
  elif (( resume_cpu_partials )); then
    if (( dry_run )); then
      printf 'dry-run: verify CPU prefill collection contract %s == %s\n' \
        "${contract_path}" "${contract_digest}"
    elif [[ ! -f "${contract_path}" ]] ||
         [[ "$(<"${contract_path}")" != "${contract_digest}" ]]; then
      echo "error: CPU prefill partial collection contract changed; use the " \
           "original command or a new output directory" >&2
      exit 2
    fi
  elif (( ! skip_sweep || collect_cpu_prefill_sealed_after_freeze )); then
    if (( dry_run )); then
      printf 'dry-run: publish CPU prefill contract %s -> %s\n' \
        "${contract_digest}" "${contract_path}"
    else
      printf '%s\n' "${contract_digest}" > "${contract_path}.inprogress"
      mv "${contract_path}.inprogress" "${contract_path}"
    fi
  fi

  if (( ! skip_sweep && ! skip_cpu_prefill_baseline )); then
    local format_shards=()
    if (( prefill_format_shards )); then
      mapfile -t format_shards < <(csv_values "${cpu_formats}")
    else
      format_shards=("${cpu_formats}")
    fi
    local partials=() timing_partials=()
    local job_formats=() job_shapes=() job_ns=() job_ks=() job_ms=()
    local job_runtime_isas=() job_bins=() job_partials=() job_timing_partials=()
    local group_starts=() group_ends=()
    local format_spec format_label
    local record n k m_values partial timing_partial
    if [[ "${profile}" == "all" ]] && (( ! shapes_explicit )); then
      declare -A allowed_cpu_formats=()
      while IFS= read -r format_spec; do
        allowed_cpu_formats["${format_spec}"]=1
      done < <(csv_values "${cpu_formats}")
      local pending_m_inventory=""
      local pending_group_open=0
      while IFS=$'\t' read -r format_spec shape n k regime_name runtime_isa m_values; do
        if [[ -z "${allowed_cpu_formats[${format_spec}]+present}" ]]; then
          continue
        fi
        if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
           [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
          echo "error: CPU prefill plan emitted unknown ISA regime ${regime_name}" >&2
          exit 2
        fi
        format_label="${format_spec}"
        partial="${output_dir}/cpu_prefill.${format_label}.${shape}.${regime_name}.csv"
        timing_partial="${output_dir}/cpu_prefill.${format_label}.${shape}.${regime_name}.timing.csv"
        partials+=("${partial}")
        timing_partials+=("${timing_partial}")
        if (( resume_cpu_partials )) &&
           [[ -s "${partial}" && -s "${timing_partial}" ]]; then
          if PYTHONPATH="${shape_manifest_python_root}" \
              python3 -m \
                tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                --input "${partial}" \
                --timing-sidecar "${timing_partial}" \
                --expected-m-values "${m_values}"; then
            continue
          fi
          printf 'Discarding incomplete CPU prefill baseline partial before resume: %s\n' \
            "${partial}" >&2
          rm -f "${partial}" "${timing_partial}"
        fi
        # MPI complete-round synchronization requires every rank in one MPMD
        # launch to visit exactly the same ordered M phases.  The v4 covering
        # plan emits identical-M records contiguously; each inventory therefore
        # becomes an independent pairing group.  An odd final member runs alone
        # instead of crossing into a different phase inventory.
        if (( ! pending_group_open )) ||
           [[ "${m_values}" != "${pending_m_inventory}" ]]; then
          if (( pending_group_open )); then
            group_ends+=("${#job_shapes[@]}")
          fi
          group_starts+=("${#job_shapes[@]}")
          pending_m_inventory="${m_values}"
          pending_group_open=1
        fi
        job_formats+=("${format_spec}")
        job_shapes+=("${shape}")
        job_ns+=("${n}")
        job_ks+=("${k}")
        job_ms+=("${m_values}")
        job_runtime_isas+=("${runtime_isa}")
        job_bins+=("${regime_binary[${regime_name}]}")
        job_partials+=("${partial}")
        job_timing_partials+=("${timing_partial}")
      done < <(
        PYTHONPATH="${shape_manifest_python_root}" \
          python3 -m "${cpu_prefill_training_plan_module}" --records \
            "${cpu_prefill_route_args[@]}"
      )
      if (( pending_group_open )); then
        group_ends+=("${#job_shapes[@]}")
      fi
    else
      for format_spec in "${format_shards[@]}"; do
        format_label="all-formats"
        if (( prefill_format_shards )); then
          format_label="${format_spec}"
        fi
        for regime_spec in "${regime_specs[@]}"; do
          IFS='|' read -r regime_name runtime_isa regime_bin <<< "${regime_spec}"
          local profile_pending_m_inventory=""
          local profile_group_open=0
          for record in "${sorted_shape_records[@]}"; do
            IFS=$'\t' read -r shape m_values <<< "${record}"
            n="${cpu_prefill_shape_n[${shape}]}"
            k="${cpu_prefill_shape_k[${shape}]}"
            partial="${output_dir}/cpu_prefill.${format_label}.${shape}.${regime_name}.csv"
            timing_partial="${output_dir}/cpu_prefill.${format_label}.${shape}.${regime_name}.timing.csv"
            partials+=("${partial}")
            timing_partials+=("${timing_partial}")
            if (( resume_cpu_partials )) &&
               [[ -s "${partial}" && -s "${timing_partial}" ]]; then
              if PYTHONPATH="${shape_manifest_python_root}" \
                  python3 -m \
                    tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                    --input "${partial}" \
                    --timing-sidecar "${timing_partial}" \
                    --expected-m-values "${m_values}"; then
                continue
              fi
              printf 'Discarding incomplete CPU prefill baseline partial before resume: %s\n' \
                "${partial}" >&2
              rm -f "${partial}" "${timing_partial}"
            fi
            # Quick, family-smoke, and explicit-shape profiles may combine
            # model tiers with different depth ceilings. MPI complete-round
            # synchronization permits pairing only records with an identical
            # ordered M inventory, so close the current group before adding a
            # record from another tier.
            if (( ! profile_group_open )) ||
               [[ "${m_values}" != "${profile_pending_m_inventory}" ]]; then
              if (( profile_group_open )); then
                group_ends+=("${#job_shapes[@]}")
              fi
              group_starts+=("${#job_shapes[@]}")
              profile_pending_m_inventory="${m_values}"
              profile_group_open=1
            fi
            job_formats+=("${format_spec}")
            job_shapes+=("${shape}")
            job_ns+=("${n}")
            job_ks+=("${k}")
            job_ms+=("${m_values}")
            job_runtime_isas+=("${runtime_isa}")
            job_bins+=("${regime_bin}")
            job_partials+=("${partial}")
            job_timing_partials+=("${timing_partial}")
          done
          if (( profile_group_open )); then
            group_ends+=("${#job_shapes[@]}")
          fi
        done
      done
    fi

    run_cpu_prefill_measurement_jobs \
      "${cpu_prefill_mpi_round_sync}" \
      group_starts group_ends job_formats job_shapes job_ns job_ks job_ms \
      job_runtime_isas job_bins job_partials job_timing_partials
    if (( cpu_prefill_collection_limited )); then
      printf 'CPU NativeVNNI prefill checkpoint: completed %d pending MPMD batch(es); rerun with --resume-cpu-partials to continue in %s\n' \
        "${cpu_prefill_batches_run}" "${output_dir}"
      finish_backend_collection_target
      return
    fi
    combine_csvs "${cpu_prefill_csv}" "${partials[@]}"
    combine_csvs "${cpu_prefill_timing_csv}" "${timing_partials[@]}"
  fi

  if [[ "${measurement_profile}" == "production" ]]; then
    if (( ! skip_sweep )); then
      # Version 3 and the failed v5/v6 witness certificates are immutable
      # development evidence. Each retains its original aggregate name so a
      # later seal can never overwrite or be confused with it.
      local legacy_v3_available=0
      if [[ -s "${cpu_prefill_legacy_v3_csv}" ||
            -s "${cpu_prefill_legacy_v3_timing_csv}" ]]; then
        if [[ ! -s "${cpu_prefill_legacy_v3_csv}" ||
              ! -s "${cpu_prefill_legacy_v3_timing_csv}" ]]; then
          echo "error: opened CPU prefill v3 evidence is missing one sidecar" >&2
          exit 2
        fi
        legacy_v3_available=1
      fi
      local opened_v5_available=0
      if [[ -s "${cpu_prefill_opened_v5_csv}" ||
            -s "${cpu_prefill_opened_v5_timing_csv}" ]]; then
        if [[ ! -s "${cpu_prefill_opened_v5_csv}" ||
              ! -s "${cpu_prefill_opened_v5_timing_csv}" ]]; then
          echo "error: opened CPU prefill v5 witness evidence is missing one sidecar" >&2
          exit 2
        fi
        opened_v5_available=1
      fi
      if (( ! opened_v5_available )); then
        if (( dry_run )); then
          printf 'dry-run: require complete opened CPU prefill v5 witness evidence %q %q\n' \
            "${cpu_prefill_opened_v5_csv}" \
            "${cpu_prefill_opened_v5_timing_csv}"
          opened_v5_available=1
        else
          echo "error: CPU prefill v8 requires complete opened v5 witness evidence" >&2
          exit 2
        fi
      elif (( ! dry_run )); then
        local opened_v5_csv_actual_sha256 opened_v5_timing_actual_sha256
        opened_v5_csv_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v5_csv}" | awk '{print $1}'
        )"
        opened_v5_timing_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v5_timing_csv}" | awk '{print $1}'
        )"
        if [[ "${opened_v5_csv_actual_sha256}" != "${cpu_prefill_opened_v5_csv_sha256}" ||
              "${opened_v5_timing_actual_sha256}" != "${cpu_prefill_opened_v5_timing_sha256}" ]]; then
          echo "error: opened CPU prefill v5 witness evidence changed after certification" >&2
          exit 2
        fi
      fi
      local opened_v6_available=0
      if [[ -s "${cpu_prefill_opened_v6_csv}" ||
            -s "${cpu_prefill_opened_v6_timing_csv}" ]]; then
        if [[ ! -s "${cpu_prefill_opened_v6_csv}" ||
              ! -s "${cpu_prefill_opened_v6_timing_csv}" ]]; then
          echo "error: opened CPU prefill v6 witness evidence is missing one sidecar" >&2
          exit 2
        fi
        opened_v6_available=1
      fi
      if (( ! opened_v6_available )); then
        if (( dry_run )); then
          printf 'dry-run: require complete opened CPU prefill v6 witness evidence %q %q\n' \
            "${cpu_prefill_opened_v6_csv}" \
            "${cpu_prefill_opened_v6_timing_csv}"
          opened_v6_available=1
        else
          echo "error: CPU prefill v8 requires complete opened v6 witness evidence" >&2
          exit 2
        fi
      elif (( ! dry_run )); then
        local opened_v6_csv_actual_sha256 opened_v6_timing_actual_sha256
        opened_v6_csv_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v6_csv}" | awk '{print $1}'
        )"
        opened_v6_timing_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v6_timing_csv}" | awk '{print $1}'
        )"
        if [[ "${opened_v6_csv_actual_sha256}" != "${cpu_prefill_opened_v6_csv_sha256}" ||
              "${opened_v6_timing_actual_sha256}" != "${cpu_prefill_opened_v6_timing_sha256}" ]]; then
          echo "error: opened CPU prefill v6 witness evidence changed after certification" >&2
          exit 2
        fi
      fi

      local opened_v7_available=0
      if [[ -s "${cpu_prefill_opened_v7_csv}" ||
            -s "${cpu_prefill_opened_v7_timing_csv}" ]]; then
        if [[ ! -s "${cpu_prefill_opened_v7_csv}" ||
              ! -s "${cpu_prefill_opened_v7_timing_csv}" ]]; then
          echo "error: opened CPU prefill v7 witness evidence is missing one sidecar" >&2
          exit 2
        fi
        opened_v7_available=1
      fi
      if (( ! opened_v7_available )); then
        if (( dry_run )); then
          printf 'dry-run: require complete opened CPU prefill v7 witness evidence %q %q\n' \
            "${cpu_prefill_opened_v7_csv}" \
            "${cpu_prefill_opened_v7_timing_csv}"
          opened_v7_available=1
        else
          echo "error: CPU prefill v8 requires complete opened v7 witness evidence" >&2
          exit 2
        fi
      elif (( ! dry_run )); then
        local opened_v7_csv_actual_sha256 opened_v7_timing_actual_sha256
        opened_v7_csv_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v7_csv}" | awk '{print $1}'
        )"
        opened_v7_timing_actual_sha256="$(
          sha256sum "${cpu_prefill_opened_v7_timing_csv}" | awk '{print $1}'
        )"
        if [[ "${opened_v7_csv_actual_sha256}" != "${cpu_prefill_opened_v7_csv_sha256}" ||
              "${opened_v7_timing_actual_sha256}" != "${cpu_prefill_opened_v7_timing_sha256}" ]]; then
          echo "error: opened CPU prefill v7 witness evidence changed after certification" >&2
          exit 2
        fi
      fi

      local refinement_sources=() refinement_timing_sources=()
      if (( legacy_v3_available )); then
        refinement_sources+=("${cpu_prefill_legacy_v3_csv}")
        refinement_timing_sources+=("${cpu_prefill_legacy_v3_timing_csv}")
      fi
      local refinement_partials=() refinement_timing_partials=()
      local refinement_job_formats=() refinement_job_shapes=()
      local refinement_job_ns=() refinement_job_ks=() refinement_job_ms=()
      local refinement_job_runtime_isas=() refinement_job_bins=()
      local refinement_job_partials=() refinement_job_timing_partials=()
      local refinement_group_starts=() refinement_group_ends=()
      local refinement_pending_m_inventory=""
      local refinement_pending_group_open=0
      if (( dry_run )); then
        printf 'dry-run: diff current CPU prefill base plan against %q before development refinement\n' \
          "${cpu_prefill_csv}"
      fi
      while IFS=$'\t' read -r format_spec shape n k regime_name runtime_isa m_values; do
        if ! csv_contains "${cpu_formats}" "${format_spec}"; then
          continue
        fi
        if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
           [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
          echo "error: CPU prefill refinement plan emitted unknown ISA regime ${regime_name}" >&2
          exit 2
        fi
        # Historical aggregates already contain the opened V5, V7, V8, and V9
        # records. Skip only those jobs; focused CPUPrefillRefine_*
        # neighborhoods remain independent v4 development evidence.
        if (( legacy_v3_available )) &&
           [[ "${shape}" == V5CPUPrefillSealed_* ]]; then
          continue
        fi
        if (( opened_v5_available )) &&
           [[ "${shape}" == V7CPUPrefillSealed_* ]]; then
          continue
        fi
        if (( opened_v6_available )) &&
           [[ "${shape}" == V8CPUPrefillSealed_* ]]; then
          continue
        fi
        if (( opened_v7_available )) &&
           [[ "${shape}" == V9CPUPrefillSealed_* ]]; then
          continue
        fi
        partial="${output_dir}/cpu_prefill.refinement-v4.${format_spec}.${shape}.${regime_name}.csv"
        timing_partial="${output_dir}/cpu_prefill.refinement-v4.${format_spec}.${shape}.${regime_name}.timing.csv"
        refinement_partials+=("${partial}")
        refinement_timing_partials+=("${timing_partial}")
        if (( resume_cpu_partials )) &&
           [[ -s "${partial}" && -s "${timing_partial}" ]]; then
          if PYTHONPATH="${shape_manifest_python_root}" \
              python3 -m \
                tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                --input "${partial}" \
                --timing-sidecar "${timing_partial}" \
                --expected-m-values "${m_values}"; then
            continue
          fi
          printf 'Discarding incomplete CPU prefill refinement-v4 partial before resume: %s\n' \
            "${partial}" >&2
          rm -f "${partial}" "${timing_partial}"
        fi
        if (( ! refinement_pending_group_open )) ||
           [[ "${m_values}" != "${refinement_pending_m_inventory}" ]]; then
          if (( refinement_pending_group_open )); then
            refinement_group_ends+=("${#refinement_job_shapes[@]}")
          fi
          refinement_group_starts+=("${#refinement_job_shapes[@]}")
          refinement_pending_m_inventory="${m_values}"
          refinement_pending_group_open=1
        fi
        refinement_job_formats+=("${format_spec}")
        refinement_job_shapes+=("${shape}")
        refinement_job_ns+=("${n}")
        refinement_job_ks+=("${k}")
        refinement_job_ms+=("${m_values}")
        refinement_job_runtime_isas+=("${runtime_isa}")
        refinement_job_bins+=("${regime_binary[${regime_name}]}")
        refinement_job_partials+=("${partial}")
        refinement_job_timing_partials+=("${timing_partial}")
      done < <(
        if (( dry_run )); then
          PYTHONPATH="${shape_manifest_python_root}" \
            python3 -m "${cpu_prefill_training_plan_module}" \
              --development-refinement-records
        else
          PYTHONPATH="${shape_manifest_python_root}" \
            python3 -m "${cpu_prefill_training_plan_module}" \
              --development-update-records-from "${cpu_prefill_csv}" \
              "${cpu_prefill_route_args[@]}"
        fi
      )
      if (( refinement_pending_group_open )); then
        refinement_group_ends+=("${#refinement_job_shapes[@]}")
      fi

      run_cpu_prefill_measurement_jobs \
        1 refinement_group_starts refinement_group_ends \
        refinement_job_formats refinement_job_shapes refinement_job_ns \
        refinement_job_ks refinement_job_ms refinement_job_runtime_isas \
        refinement_job_bins refinement_job_partials \
        refinement_job_timing_partials
      if (( cpu_prefill_collection_limited )); then
        printf 'CPU NativeVNNI prefill refinement-v4 checkpoint: completed %d pending MPMD batch(es); rerun with --skip-cpu-prefill-baseline --resume-cpu-partials to continue in %s\n' \
          "${cpu_prefill_batches_run}" "${output_dir}"
        finish_backend_collection_target
        return
      fi
      refinement_sources+=("${refinement_partials[@]}")
      refinement_timing_sources+=("${refinement_timing_partials[@]}")
      combine_csvs "${cpu_prefill_refinement_csv}" \
        "${refinement_sources[@]}"
      combine_csvs "${cpu_prefill_refinement_timing_csv}" \
        "${refinement_timing_sources[@]}"
      combine_csvs "${cpu_prefill_development_v4_csv}" \
        "${cpu_prefill_csv}" "${cpu_prefill_refinement_csv}"
      combine_csvs "${cpu_prefill_development_v4_timing_csv}" \
        "${cpu_prefill_timing_csv}" "${cpu_prefill_refinement_timing_csv}"
      combine_csvs "${cpu_prefill_development_v6_csv}" \
        "${cpu_prefill_development_v4_csv}" "${cpu_prefill_opened_v5_csv}"
      combine_csvs "${cpu_prefill_development_v6_timing_csv}" \
        "${cpu_prefill_development_v4_timing_csv}" \
        "${cpu_prefill_opened_v5_timing_csv}"
      combine_csvs "${cpu_prefill_development_v7_csv}" \
        "${cpu_prefill_development_v6_csv}" "${cpu_prefill_opened_v6_csv}"
      combine_csvs "${cpu_prefill_development_v7_timing_csv}" \
        "${cpu_prefill_development_v6_timing_csv}" \
        "${cpu_prefill_opened_v6_timing_csv}"
      combine_csvs "${cpu_prefill_development_csv}" \
        "${cpu_prefill_development_v7_csv}" "${cpu_prefill_opened_v7_csv}"
      combine_csvs "${cpu_prefill_development_timing_csv}" \
        "${cpu_prefill_development_v7_timing_csv}" \
        "${cpu_prefill_opened_v7_timing_csv}"
    elif [[ -n "${cpu_prefill_fit_candidate_expansion_plan}" ]]; then
      # Candidate expansion is a separate anchored timing transaction. Its
      # source and every later complete-registry increment must remain
      # separate inputs so the adapter can normalize only the new family.
      if (( dry_run )); then
        printf 'dry-run: require complete CPU prefill candidate-expansion fit transaction %q %q %q %q %q\n' \
          "${cpu_prefill_fit_candidate_expansion_plan}" \
          "${cpu_prefill_fit_candidate_expansion_source_aggregate}" \
          "${cpu_prefill_fit_candidate_expansion_source_timing}" \
          "${cpu_prefill_fit_candidate_expansion_input}" \
          "${cpu_prefill_fit_candidate_expansion_timing}"
      else
        local candidate_fit_required_path
        for candidate_fit_required_path in \
            "${cpu_prefill_fit_candidate_expansion_plan}" \
            "${cpu_prefill_fit_candidate_expansion_source_aggregate}" \
            "${cpu_prefill_fit_candidate_expansion_source_timing}" \
            "${cpu_prefill_fit_candidate_expansion_input}" \
            "${cpu_prefill_fit_candidate_expansion_timing}" \
            "${cpu_prefill_fit_additive_aggregates[@]}" \
            "${cpu_prefill_fit_additive_timings[@]}" \
            "${cpu_prefill_fit_generic_refinement_plans[@]}" \
            "${cpu_prefill_fit_additive_profiler_requests[@]}" \
            "${cpu_prefill_fit_additive_profiler_evidence[@]}" \
            "${cpu_prefill_fit_additive_profiler_observations[@]}"; do
          if [[ ! -s "${candidate_fit_required_path}" ]]; then
            echo "error: CPU prefill candidate-expansion fit input is missing: ${candidate_fit_required_path}" >&2
            exit 2
          fi
        done
      fi
    elif (( dry_run )); then
      printf 'dry-run: require complete CPU prefill v8 development aggregate %q %q\n' \
        "${cpu_prefill_development_csv}" \
        "${cpu_prefill_development_timing_csv}"
    elif [[ ! -s "${cpu_prefill_development_csv}" ||
            ! -s "${cpu_prefill_development_timing_csv}" ]]; then
      echo "error: production --skip-sweep requires complete CPU prefill development CSVs" >&2
      exit 2
    fi

    local cpu_prefill_run_id="${timestamp}-cpu-prefill-${profile}-development"
    if (( resume_cpu_partials || skip_sweep )) &&
       [[ -s "${cpu_prefill_common_csv}" ]]; then
      cpu_prefill_run_id="$(
        csv_unique_field "${cpu_prefill_common_csv}" run_id
      )"
      printf 'Resuming CPU prefill common observation run identity: %s\n' \
        "${cpu_prefill_run_id}"
    fi
    local cpu_prefill_policy_common_csv="${cpu_prefill_common_csv}"
    local -a cpu_prefill_policy_checkpoint_args=(
      --adapted-full-development-input "${cpu_prefill_common_csv}"
    )
    local -a cpu_prefill_burned_seal_fit_args=()
    if (( ${#cpu_prefill_fit_burned_seal_manifests[@]} > 0 )); then
      cpu_prefill_policy_common_csv="${output_dir}/cpu_prefill_common_observations.burned-v1.csv"
      if [[ -s "${cpu_prefill_policy_common_csv}" ]]; then
        cpu_prefill_policy_checkpoint_args=(
          --adapted-full-development-input
          "${cpu_prefill_policy_common_csv}"
        )
      else
        cpu_prefill_policy_checkpoint_args=(
          --adapted-development-before-burned-seals-input
          "${cpu_prefill_common_csv}"
        )
      fi
      local burned_seal_manifest
      for burned_seal_manifest in \
          "${cpu_prefill_fit_burned_seal_manifests[@]}"; do
        cpu_prefill_burned_seal_fit_args+=(
          --burned-sealed-development-manifest
          "${burned_seal_manifest}"
        )
      done
      printf 'CPU prefill replay promotes %d inspected seal(s) to typed development evidence\n' \
        "${#cpu_prefill_fit_burned_seal_manifests[@]}"
    fi
    local cpu_prefill_active_development_csv="${cpu_prefill_development_csv}"
    local cpu_prefill_active_development_timing_csv="${cpu_prefill_development_timing_csv}"
    local -a cpu_prefill_active_development_csvs=(
      "${cpu_prefill_active_development_csv}"
    )
    local -a cpu_prefill_active_development_timing_csvs=(
      "${cpu_prefill_active_development_timing_csv}"
    )
    local -a cpu_prefill_candidate_expansion_fit_args=()
    if [[ -n "${cpu_prefill_fit_candidate_expansion_plan}" ]]; then
      cpu_prefill_active_development_csv="${cpu_prefill_fit_candidate_expansion_source_aggregate}"
      cpu_prefill_active_development_timing_csv="${cpu_prefill_fit_candidate_expansion_source_timing}"
      cpu_prefill_active_development_csvs=(
        "${cpu_prefill_fit_candidate_expansion_source_aggregate}"
        "${cpu_prefill_fit_additive_aggregates[@]}"
      )
      cpu_prefill_active_development_timing_csvs=(
        "${cpu_prefill_fit_candidate_expansion_source_timing}"
        "${cpu_prefill_fit_additive_timings[@]}"
      )
      cpu_prefill_candidate_expansion_fit_args=(
        --candidate-expansion-plan-json
        "${cpu_prefill_fit_candidate_expansion_plan}"
        --candidate-expansion-input
        "${cpu_prefill_fit_candidate_expansion_input}"
        --candidate-expansion-timing-sidecar
        "${cpu_prefill_fit_candidate_expansion_timing}"
      )
    fi
    local -a cpu_prefill_additive_profiler_fit_args=()
    local additive_profiler_index
    for ((additive_profiler_index = 0;
         additive_profiler_index <
           ${#cpu_prefill_fit_additive_profiler_requests[@]};
         ++additive_profiler_index)); do
      cpu_prefill_additive_profiler_fit_args+=(
        --development-profiler-requests
        "${cpu_prefill_fit_additive_profiler_requests[${additive_profiler_index}]}"
        --development-profiler-evidence
        "${cpu_prefill_fit_additive_profiler_evidence[${additive_profiler_index}]}"
        --development-profiler-observations
        "${cpu_prefill_fit_additive_profiler_observations[${additive_profiler_index}]}"
      )
    done
    local -a cpu_prefill_generic_refinement_args=()
    local cpu_prefill_fit_generic_refinement_plan
    for cpu_prefill_fit_generic_refinement_plan in \
        "${cpu_prefill_fit_generic_refinement_plans[@]}"; do
      cpu_prefill_generic_refinement_args+=(
        --generic-refinement-plan-json
        "${cpu_prefill_fit_generic_refinement_plan}"
      )
    done
    local -a cpu_prefill_development_lineage_args=()
    if [[ -n "${cpu_prefill_fit_development_lineage_plan}" ]]; then
      cpu_prefill_development_lineage_args=(
        --development-lineage-plan-json
        "${cpu_prefill_fit_development_lineage_plan}"
        --development-lineage-source-input
        "${cpu_prefill_lineage_source_aggregate}"
        --development-lineage-source-timing-sidecar
        "${cpu_prefill_lineage_source_timing}"
        --development-lineage-source-split-manifest
        "${cpu_prefill_lineage_source_split_manifest}"
      )
      local lineage_fit_source_route_manifest
      for lineage_fit_source_route_manifest in \
          "${cpu_prefill_lineage_source_route_manifests[@]}"; do
        cpu_prefill_development_lineage_args+=(
          --development-lineage-source-route-manifest
          "${lineage_fit_source_route_manifest}"
        )
      done
      local lineage_fit_source_refinement_plan
      for lineage_fit_source_refinement_plan in \
          "${cpu_prefill_lineage_source_refinement_plans[@]}"; do
        cpu_prefill_development_lineage_args+=(
          --development-lineage-source-refinement-plan-json
          "${lineage_fit_source_refinement_plan}"
        )
      done
      local lineage_fit_source_refinement_split_manifest
      for lineage_fit_source_refinement_split_manifest in \
          "${cpu_prefill_lineage_source_refinement_split_manifests[@]}"; do
        cpu_prefill_development_lineage_args+=(
          --development-lineage-source-refinement-split-manifest
          "${lineage_fit_source_refinement_split_manifest}"
        )
      done
    fi
    local cpu_prefill_refinement_round=0
    local cpu_prefill_unpromoted_domains=0
    local reuse_cpu_prefill_profiler_evidence=0

    # A completed generic-refinement aggregate is immutable development
    # evidence. Reconstruct the in-memory refinement chain when resuming so an
    # interrupted transaction starts from its latest complete corpus instead
    # of fitting the baseline again and rediscovering already-measured cells.
    # An incomplete next round is intentionally ignored here: the current fit
    # regenerates its authenticated plan, whose digest-bound partial names then
    # resume only evidence belonging to those exact plan bytes.
    if (( resume_cpu_partials )); then
      local completed_refinement_round=1
      while true; do
        local completed_round_plan="${output_dir}/cpu_prefill_generic_refinement.round${completed_refinement_round}.json"
        local completed_round_development="${output_dir}/cpu_prefill_sweep.development-generic-r${completed_refinement_round}.csv"
        local completed_round_development_timing="${output_dir}/cpu_prefill_sweep.development-generic-r${completed_refinement_round}.timing.csv"
        if [[ ! -s "${completed_round_plan}" ||
              ! -s "${completed_round_development}" ||
              ! -s "${completed_round_development_timing}" ]]; then
          break
        fi
        if (( ! dry_run )) &&
           ! PYTHONPATH="${shape_manifest_python_root}" \
              python3 -m \
                tests.v2.performance.kernels.native_vnni_dispatch.validate_cpu_prefill_partial \
                --input "${completed_round_development}" \
                --timing-sidecar "${completed_round_development_timing}"; then
          printf 'Ignoring non-installable completed CPU prefill refinement round %d during resume: %s\n' \
            "${completed_refinement_round}" \
            "${completed_round_development}" >&2
          break
        fi
        cpu_prefill_active_development_csv="${completed_round_development}"
        cpu_prefill_active_development_timing_csv="${completed_round_development_timing}"
        cpu_prefill_generic_refinement_args+=(
          --generic-refinement-plan-json "${completed_round_plan}"
        )
        cpu_prefill_refinement_round="${completed_refinement_round}"
        completed_refinement_round=$((completed_refinement_round + 1))
      done
      if (( cpu_prefill_refinement_round > 0 )); then
        printf 'Resuming CPU prefill from completed generic refinement round %d: %s\n' \
          "${cpu_prefill_refinement_round}" \
          "${cpu_prefill_active_development_csv}"
      fi
    fi
    if [[ -n "${cpu_prefill_fit_primary_profiler_requests}" ]]; then
      # Recipe preflight already authenticated this immutable triplet. Re-export
      # its derived features below, but never consult the historical implicit
      # base or require its large source-observation checkpoint.
      reuse_cpu_prefill_profiler_evidence=1
    elif [[ -s "${cpu_prefill_profiler_requests}" ||
            -s "${cpu_prefill_profiler_evidence}" ||
            -s "${cpu_prefill_profiler_source_common}" ]]; then
      if [[ ! -s "${cpu_prefill_profiler_requests}" ||
            ! -s "${cpu_prefill_profiler_evidence}" ||
            ! -s "${cpu_prefill_profiler_source_common}" ]]; then
        echo "error: reusable CPU prefill profiler evidence is missing requests, evidence, or its immutable source corpus" >&2
        exit 2
      fi
      reuse_cpu_prefill_profiler_evidence=1
    fi
    if (( reuse_profiler_evidence )); then
      # A fit-only transaction must authenticate the immutable development
      # observations and every per-candidate profiler sidecar before the
      # iterative fitter sees them.  This exports features only; it never
      # launches the trainer binary or Linux perf.
      collect_backend_profiler_evidence \
        cpu "${cpu_prefill_profiler_source_common}" \
        "${cpu_avx2_sweep_bin}" \
        "${cpu_prefill_profiler_requests}" \
        "${cpu_prefill_profiler_evidence}" \
        "${cpu_prefill_profiler_features}" \
        "${cpu_prefill_profiler_raw}" \
        reuse \
        "${cpu_prefill_profiler_witnesses}"
      reuse_cpu_prefill_profiler_evidence=1
    fi

    # Fit diagnostics are intentionally non-installable. Every failed round
    # names fresh, route-compatible geometry cells; only those timing cells
    # are appended before the next fit. Profiler evidence is already isolated
    # per physical candidate variant and remains bound to its immutable source
    # corpus, so geometry refinement never repeats hardware-counter launches.
    while true; do
      local cpu_prefill_adapt_common_csv="${cpu_prefill_common_csv}"
      local extend_cpu_prefill_checkpoint=0
      local rebuild_cpu_prefill_checkpoint=0
      local rebase_cpu_prefill_checkpoint=0
      if [[ "${cpu_prefill_replay_checkpoint_action}" == "prefix" ]]; then
        cpu_prefill_adapt_common_csv="${cpu_prefill_common_csv}.inprogress"
        extend_cpu_prefill_checkpoint=1
      elif [[ "${cpu_prefill_replay_checkpoint_action}" == "rebuild" ]]; then
        cpu_prefill_adapt_common_csv="${cpu_prefill_common_csv}.inprogress"
        rebuild_cpu_prefill_checkpoint=1
      elif [[ "${cpu_prefill_replay_checkpoint_action}" == "rebase" ]]; then
        cpu_prefill_adapt_common_csv="${cpu_prefill_common_csv}.inprogress"
        rebase_cpu_prefill_checkpoint=1
      elif [[ -z "${cpu_prefill_fit_replay_recipe}" ]] &&
           (( ${#cpu_prefill_fit_generic_refinement_plans[@]} > 0 )) &&
           [[ -s "${cpu_prefill_common_csv}" ]]; then
        # Legacy diagnostic mode retains historical inference. Production
        # replay recipes always state checkpoint lifecycle explicitly.
        cpu_prefill_adapt_common_csv="${cpu_prefill_common_csv}.inprogress"
        extend_cpu_prefill_checkpoint=1
      fi
      local -a cpu_prefill_adapt=(
        python3 "${cpu_prefill_generator}"
        --input "${cpu_prefill_active_development_csvs[@]}"
        --output "${cpu_prefill_provisional_inc}"
        --common-observations "${cpu_prefill_adapt_common_csv}"
        --split-manifest "${cpu_prefill_split_manifest_path}"
        --adapt-only
        --profile production
        --run-id "${cpu_prefill_run_id}"
        --git-revision "${cpu_git_revision}"
        --build-id "${cpu_build_id}"
        --compiler-id "${cpu_compiler_id}"
        --architecture-class "${cpu_arch_class}"
        --device-name "${cpu_device_name}"
        --driver-runtime "${cpu_driver_runtime}"
        --serial-m1-policy-hash "${cpu_serial_policy_hash}"
      )
      local active_development_timing
      for active_development_timing in \
          "${cpu_prefill_active_development_timing_csvs[@]}"; do
        cpu_prefill_adapt+=(
          --timing-sidecar "${active_development_timing}"
        )
      done
      cpu_prefill_adapt+=("${cpu_prefill_candidate_expansion_fit_args[@]}")
      cpu_prefill_adapt+=("${cpu_prefill_generic_refinement_args[@]}")
      cpu_prefill_adapt+=("${cpu_prefill_development_lineage_args[@]}")
      cpu_prefill_adapt+=("${cpu_prefill_route_args[@]}")
      if (( extend_cpu_prefill_checkpoint )); then
        cpu_prefill_adapt+=(
          --adapted-development-prefix-input "${cpu_prefill_common_csv}"
        )
        if [[ -n "${cpu_prefill_replay_checkpoint_prefix_count}" ]]; then
          cpu_prefill_adapt+=(
            --adapted-development-prefix-count
            "${cpu_prefill_replay_checkpoint_prefix_count}"
          )
        fi
        run_cmd "${cpu_prefill_adapt[@]}"
        run_cmd mv "${cpu_prefill_adapt_common_csv}" \
          "${cpu_prefill_common_csv}"
      elif (( rebuild_cpu_prefill_checkpoint )); then
        run_cmd "${cpu_prefill_adapt[@]}"
        run_cmd mv "${cpu_prefill_adapt_common_csv}" \
          "${cpu_prefill_common_csv}"
      elif (( rebase_cpu_prefill_checkpoint )); then
        cpu_prefill_adapt+=(
          --adapted-full-development-rebase-input
          "${cpu_prefill_common_csv}"
        )
        run_cmd "${cpu_prefill_adapt[@]}"
        run_cmd mv "${cpu_prefill_adapt_common_csv}" \
          "${cpu_prefill_common_csv}"
      elif (( resume_cpu_partials || skip_sweep )) &&
         [[ -s "${cpu_prefill_common_csv}" ]]; then
        printf 'Reusing authenticated CPU prefill development checkpoint: %s\n' \
          "${cpu_prefill_common_csv}"
      else
        run_cmd "${cpu_prefill_adapt[@]}"
      fi
      if [[ -n "${cpu_prefill_fit_replay_recipe}" ]] &&
         [[ "${cpu_prefill_replay_checkpoint_action}" == "prefix" ||
           "${cpu_prefill_replay_checkpoint_action}" == "rebuild" ||
           "${cpu_prefill_replay_checkpoint_action}" == "rebase" ]]; then
        run_cmd \
          "PYTHONPATH=${shape_manifest_python_root}" \
          python3 -m \
            native_vnni_dispatch.cpu_prefill_replay_recipe \
            finalize-checkpoint \
            --recipe "${cpu_prefill_fit_replay_recipe}" \
            --checkpoint "${cpu_prefill_common_csv}"
        cpu_prefill_replay_checkpoint_action="complete"
      fi

      local -a cpu_prefill_authenticated_context_args=()
      if [[ -n "${cpu_prefill_fit_replay_recipe}" ]] &&
         [[ "${cpu_prefill_replay_checkpoint_action}" == "complete" ]]; then
        if [[ ! "${cpu_prefill_replay_context_corpus_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
          echo "error: authenticated CPU prefill replay omitted its raw corpus identity" >&2
          exit 2
        fi
        cpu_prefill_authenticated_context_args=(
          --authenticated-raw-corpus-id
          "${cpu_prefill_replay_context_corpus_id}"
        )
      fi

      if (( ${#cpu_prefill_fit_burned_seal_manifests[@]} > 0 )) &&
         [[ ! -s "${cpu_prefill_policy_common_csv}" ]]; then
        # Materialize the mixed-provenance development checkpoint before
        # fitting. A failed or interrupted tree search can then resume without
        # reparsing the retained multi-gigabyte seal timing sidecar.
        local -a cpu_prefill_compose_burned_seals=(
          python3 "${cpu_prefill_generator}"
          --input "${cpu_prefill_active_development_csvs[@]}"
          --output "${cpu_prefill_provisional_inc}"
          --common-observations "${cpu_prefill_policy_common_csv}"
          --fit-cache-dir "${cpu_prefill_fit_cache_dir}"
          --split-manifest "${cpu_prefill_split_manifest_path}"
          --adapt-only
          --profile production
          --run-id "${cpu_prefill_run_id}"
          --git-revision "${cpu_git_revision}"
          --build-id "${cpu_build_id}"
          --compiler-id "${cpu_compiler_id}"
          --architecture-class "${cpu_arch_class}"
          --device-name "${cpu_device_name}"
          --driver-runtime "${cpu_driver_runtime}"
          --serial-m1-policy-hash "${cpu_serial_policy_hash}"
        )
        local burned_development_timing
        for burned_development_timing in \
            "${cpu_prefill_active_development_timing_csvs[@]}"; do
          cpu_prefill_compose_burned_seals+=(
            --timing-sidecar "${burned_development_timing}"
          )
        done
        cpu_prefill_compose_burned_seals+=(
          "${cpu_prefill_candidate_expansion_fit_args[@]}"
          "${cpu_prefill_policy_checkpoint_args[@]}"
          "${cpu_prefill_burned_seal_fit_args[@]}"
          "${cpu_prefill_authenticated_context_args[@]}"
          "${cpu_prefill_generic_refinement_args[@]}"
          "${cpu_prefill_development_lineage_args[@]}"
          "${cpu_prefill_route_args[@]}"
        )
        run_cmd "${cpu_prefill_compose_burned_seals[@]}"
        cpu_prefill_policy_checkpoint_args=(
          --adapted-full-development-input
          "${cpu_prefill_policy_common_csv}"
        )
      fi

      if (( reuse_cpu_prefill_profiler_evidence )); then
        if [[ -n "${cpu_prefill_fit_primary_profiler_requests}" ]]; then
          printf 'Reusing primary matched-anchor CPU prefill profiler evidence: %s\n' \
            "${cpu_prefill_profiler_requests}"
        else
          printf 'Reusing per-variant CPU prefill profiler evidence from %s\n' \
            "${cpu_prefill_profiler_source_common}"
        fi
      else
        collect_backend_profiler_evidence \
          cpu "${cpu_prefill_common_csv}" "${cpu_avx2_sweep_bin}" \
          "${cpu_prefill_profiler_requests}" \
          "${cpu_prefill_profiler_evidence}" \
          "${cpu_prefill_profiler_features}" "${cpu_prefill_profiler_raw}"
        run_cmd cp "${cpu_prefill_common_csv}" \
          "${cpu_prefill_profiler_source_common}.inprogress"
        run_cmd mv "${cpu_prefill_profiler_source_common}.inprogress" \
          "${cpu_prefill_profiler_source_common}"
        reuse_cpu_prefill_profiler_evidence=1
      fi

      local diagnostic_variant=""
      local diagnostic_label="profiler-informed"
      if [[ -n "${cpu_prefill_diagnostic_max_leaves}" ]]; then
        if (( cpu_prefill_ablate_profiler_features )); then
          diagnostic_variant=".profiler-ablated"
          diagnostic_label="profiler-ablated"
        else
          diagnostic_variant=".profiler-informed"
        fi
      fi
      local round_diagnostic="${cpu_prefill_development_fit_diagnostic}.round${cpu_prefill_refinement_round}${diagnostic_variant}"
      local -a cpu_prefill_development_fit=(
        python3 "${cpu_prefill_generator}"
        --input "${cpu_prefill_active_development_csvs[@]}"
        --output "${cpu_prefill_provisional_inc}"
        --development-fit-diagnostic "${round_diagnostic}"
        --common-observations "${cpu_prefill_policy_common_csv}"
        --fit-cache-dir "${cpu_prefill_fit_cache_dir}"
        --development-profiler-requests "${cpu_prefill_profiler_requests}"
        --development-profiler-evidence "${cpu_prefill_profiler_evidence}"
        --split-manifest "${cpu_prefill_split_manifest_path}"
        --profile production
        --run-id "${cpu_prefill_run_id}"
        --git-revision "${cpu_git_revision}"
        --build-id "${cpu_build_id}"
        --compiler-id "${cpu_compiler_id}"
        --architecture-class "${cpu_arch_class}"
        --device-name "${cpu_device_name}"
        --driver-runtime "${cpu_driver_runtime}"
        --serial-m1-policy-hash "${cpu_serial_policy_hash}"
      )
      for active_development_timing in \
          "${cpu_prefill_active_development_timing_csvs[@]}"; do
        cpu_prefill_development_fit+=(
          --timing-sidecar "${active_development_timing}"
        )
      done
      cpu_prefill_development_fit+=(
        "${cpu_prefill_candidate_expansion_fit_args[@]}"
      )
      cpu_prefill_development_fit+=(
        "${cpu_prefill_policy_checkpoint_args[@]}"
      )
      cpu_prefill_development_fit+=(
        "${cpu_prefill_burned_seal_fit_args[@]}"
      )
      cpu_prefill_development_fit+=(
        "${cpu_prefill_authenticated_context_args[@]}"
      )
      cpu_prefill_development_fit+=("${cpu_prefill_generic_refinement_args[@]}")
      cpu_prefill_development_fit+=("${cpu_prefill_development_lineage_args[@]}")
      if (( reuse_cpu_prefill_profiler_evidence )); then
        cpu_prefill_development_fit+=(
          --development-profiler-observations
          "${cpu_prefill_profiler_witnesses}"
        )
      fi
      cpu_prefill_development_fit+=(
        "${cpu_prefill_additive_profiler_fit_args[@]}"
      )
      cpu_prefill_development_fit+=("${cpu_prefill_route_args[@]}")
      if [[ -n "${cpu_prefill_diagnostic_max_leaves}" ]]; then
        cpu_prefill_development_fit+=(
          --generic-max-leaves "${cpu_prefill_diagnostic_max_leaves}"
        )
      fi
      if (( cpu_prefill_ablate_profiler_features )); then
        cpu_prefill_development_fit+=(--ablate-profiler-features)
      fi
      run_cmd "${cpu_prefill_development_fit[@]}"
      if (( ${#cpu_prefill_fit_burned_seal_manifests[@]} > 0 )); then
        cpu_prefill_policy_checkpoint_args=(
          --adapted-full-development-input
          "${cpu_prefill_policy_common_csv}"
        )
      fi

      if (( dry_run )); then
        run_cmd \
          "PYTHONPATH=${shape_manifest_python_root}" \
          python3 -m "${cpu_prefill_generic_refinement_module}" \
          --status "${round_diagnostic}"
        cpu_prefill_unpromoted_domains=0
      else
        cpu_prefill_unpromoted_domains="$(
          PYTHONPATH="${shape_manifest_python_root}" \
            python3 -m "${cpu_prefill_generic_refinement_module}" \
              --status "${round_diagnostic}"
        )"
      fi
      if [[ -n "${cpu_prefill_diagnostic_max_leaves}" ]]; then
        printf 'CPU prefill %s development diagnostic complete: %s (unpromoted_domains=%s)\n' \
          "${diagnostic_label}" \
          "${round_diagnostic}" \
          "${cpu_prefill_unpromoted_domains}"
        finish_backend_collection_target
        return
      fi
      if (( cpu_prefill_unpromoted_domains == 0 )); then
        break
      fi
      if (( skip_sweep )); then
        echo "error: CPU prefill fit has ${cpu_prefill_unpromoted_domains} generic domains requiring new refinement evidence" >&2
        exit 2
      fi
      if (( cpu_prefill_refinement_round >= cpu_prefill_max_refinement_rounds )); then
        echo "error: CPU prefill generic fit did not converge after ${cpu_prefill_max_refinement_rounds} refinement rounds" >&2
        exit 2
      fi

      cpu_prefill_refinement_round=$((cpu_prefill_refinement_round + 1))
      local round_plan="${output_dir}/cpu_prefill_generic_refinement.round${cpu_prefill_refinement_round}.json"
      local round_probe_plan="${output_dir}/cpu_prefill_generic_refinement.round${cpu_prefill_refinement_round}.probe.json"
      local round_csv="${output_dir}/cpu_prefill_sweep.generic-refinement-r${cpu_prefill_refinement_round}.csv"
      local round_timing_csv="${output_dir}/cpu_prefill_sweep.generic-refinement-r${cpu_prefill_refinement_round}.timing.csv"
      run_cmd \
        "PYTHONPATH=${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_generic_refinement_module}" \
        --probe-development-fit "${round_diagnostic}" \
        --output "${round_probe_plan}" --summary

      # Probe generated candidate rings with the production C++ tile helper.
      # These manifests launch no GEMM work. The final v3 refinement plan embeds
      # only selected route witnesses, which lets resumed fitting authenticate
      # old rounds without retaining or unioning transient probe-route files.
      local -a round_probe_route_args=()
      for route_probe_spec in "${route_probe_specs[@]}"; do
        IFS='|' read -r route_regime route_runtime_isa route_binary <<< \
          "${route_probe_spec}"
        route_manifest="${output_dir}/cpu_prefill_generic_refinement.round${cpu_prefill_refinement_round}.route.${route_regime}.csv"
        route_manifest_inprogress="${route_manifest}.inprogress"
        run_cmd rm -f "${route_manifest_inprogress}"
        run_cmd \
          "OMP_NUM_THREADS=${cpu_threads}" \
          "OMP_PLACES=cores" \
          "OMP_PROC_BIND=close" \
          "OMP_DYNAMIC=false" \
          "LLAMINAR_ISA_LEVEL=${route_runtime_isa}" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_CSV=${route_manifest_inprogress}" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_ISA_REGIME=${route_regime}" \
          "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_REFINEMENT_PROBE_JSON=${round_probe_plan}" \
          "${route_binary}" \
          "--gtest_filter=*TrainerCsv_PrefillSerialRouteManifest"
        run_cmd mv "${route_manifest_inprogress}" "${route_manifest}"
        if (( ! dry_run )) || [[ -s "${route_manifest}" ]]; then
          round_probe_route_args+=(
            --refinement-probe-route-manifest "${route_manifest}"
          )
        fi
      done
      run_cmd \
        "PYTHONPATH=${shape_manifest_python_root}" \
        python3 -m "${cpu_prefill_generic_refinement_module}" \
        --development-fit "${round_diagnostic}" \
        --output "${round_plan}" --summary \
        --refinement-probe-plan "${round_probe_plan}" \
        "${round_probe_route_args[@]}" \
        --split-manifest "${cpu_prefill_split_manifest_path}" \
        "${cpu_prefill_route_args[@]}"
      local refinement_status=0
      collect_cpu_prefill_generic_refinement_round \
        "${round_plan}" "${cpu_prefill_refinement_round}" \
        "${round_csv}" "${round_timing_csv}" || refinement_status=$?
      if (( refinement_status == 3 )); then
        finish_backend_collection_target
        return
      elif (( refinement_status != 0 )); then
        return "${refinement_status}"
      fi

      local next_development_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${cpu_prefill_refinement_round}.csv"
      local next_development_timing_csv="${output_dir}/cpu_prefill_sweep.development-generic-r${cpu_prefill_refinement_round}.timing.csv"
      combine_csvs "${next_development_csv}" \
        "${cpu_prefill_active_development_csv}" "${round_csv}"
      combine_csvs "${next_development_timing_csv}" \
        "${cpu_prefill_active_development_timing_csv}" "${round_timing_csv}"
      cpu_prefill_active_development_csv="${next_development_csv}"
      cpu_prefill_active_development_timing_csv="${next_development_timing_csv}"
      cpu_prefill_active_development_csvs=(
        "${cpu_prefill_active_development_csv}"
      )
      cpu_prefill_active_development_timing_csvs=(
        "${cpu_prefill_active_development_timing_csv}"
      )
      cpu_prefill_generic_refinement_args+=(
        --generic-refinement-plan-json "${round_plan}"
      )
    done

    local -a cpu_prefill_freeze=(
      python3 "${cpu_prefill_generator}"
      --input "${cpu_prefill_active_development_csvs[@]}"
      --output "${cpu_prefill_frozen_inc}"
      --summary "${cpu_prefill_summary}"
      --common-observations "${cpu_prefill_policy_common_csv}"
      --policy-json "${cpu_prefill_frozen_policy_json}"
      --sealed-witness-plan-json "${cpu_prefill_sealed_witness_plan_json}"
      --fit-cache-dir "${cpu_prefill_fit_cache_dir}"
      --development-profiler-requests "${cpu_prefill_profiler_requests}"
      --development-profiler-evidence "${cpu_prefill_profiler_evidence}"
      --split-manifest "${cpu_prefill_split_manifest_path}"
      --freeze-generic
      --profile production
      --run-id "${cpu_prefill_run_id}"
      --git-revision "${cpu_git_revision}"
      --build-id "${cpu_build_id}"
      --compiler-id "${cpu_compiler_id}"
      --architecture-class "${cpu_arch_class}"
      --device-name "${cpu_device_name}"
      --driver-runtime "${cpu_driver_runtime}"
      --serial-m1-policy-hash "${cpu_serial_policy_hash}"
    )
    local freeze_development_timing
    for freeze_development_timing in \
        "${cpu_prefill_active_development_timing_csvs[@]}"; do
      cpu_prefill_freeze+=(
        --timing-sidecar "${freeze_development_timing}"
      )
    done
    cpu_prefill_freeze+=("${cpu_prefill_candidate_expansion_fit_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_policy_checkpoint_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_burned_seal_fit_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_authenticated_context_args[@]}")
    if (( reuse_cpu_prefill_profiler_evidence )); then
      cpu_prefill_freeze+=(
        --development-profiler-observations
        "${cpu_prefill_profiler_witnesses}"
      )
    fi
    cpu_prefill_freeze+=("${cpu_prefill_additive_profiler_fit_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_generic_refinement_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_development_lineage_args[@]}")
    cpu_prefill_freeze+=("${cpu_prefill_route_args[@]}")
    if [[ -n "${cpu_prefill_fit_sealed_candidate_route_manifest}" ]]; then
      cpu_prefill_freeze+=(
        --sealed-candidate-route-manifest
        "${cpu_prefill_fit_sealed_candidate_route_manifest}"
      )
    fi
    run_cmd "${cpu_prefill_freeze[@]}"
    if (( stop_after_cpu_prefill_freeze )); then
      printf 'CPU prefill development policy frozen without opening sealed-v8 evidence: %s\n' \
        "${cpu_prefill_frozen_policy_json}"
      finish_backend_collection_target
      return
    fi

    # The certification holdout is launched only after the development policy
    # has been frozen. Splitting it from an already-open aggregate would leak
    # sealed timings into model selection and invalidate the certificate.
    if (( ! skip_sweep || collect_cpu_prefill_sealed_after_freeze )); then
      # Fit-only replay deliberately restored the immutable development
      # provenance above. Fresh sealed rows are a new measurement generation:
      # label them with the binaries and host runtime that actually execute
      # this holdout, then carry that provenance into final certification.
      cpu_git_revision="${cpu_measurement_git_revision}"
      cpu_build_id="${cpu_measurement_build_id}"
      cpu_compiler_id="${cpu_measurement_compiler_id}"
      cpu_arch_class="${cpu_measurement_arch_class}"
      cpu_device_name="${cpu_measurement_device_name}"
      cpu_driver_runtime="${cpu_measurement_driver_runtime}"
      cpu_serial_policy_hash="${cpu_measurement_serial_policy_hash}"
      if (( dry_run )); then
        printf 'dry-run: collect fresh CPU prefill sealed-v8 witness holdout after %q\n' \
          "${cpu_prefill_frozen_policy_json}"
        run_cmd \
          "PYTHONPATH=${shape_manifest_python_root}" \
          python3 -m "${cpu_prefill_training_plan_module}" \
          --sealed-witness-plan "${cpu_prefill_sealed_witness_plan_json}" \
          --coalesce-format-fixtures \
          --source-formats "${cpu_formats}" \
          "${cpu_prefill_route_args[@]}"
      else
        local sealed_partials=() sealed_timing_partials=()
        local sealed_job_formats=() sealed_job_shapes=()
        local sealed_job_ns=() sealed_job_ks=() sealed_job_ms=()
        local sealed_job_runtime_isas=() sealed_job_bins=()
        local sealed_job_partials=() sealed_job_timing_partials=()
        local sealed_group_starts=() sealed_group_ends=()
        local sealed_pending_phase_inventory=""
        local sealed_pending_group_open=0
        while IFS=$'\t' read -r format_spec format_token shape n k regime_name runtime_isa m_values format_phase_count; do
          if [[ -z "${regime_binary[${regime_name}]+present}" ]] ||
             [[ "${regime_runtime_isa[${regime_name}]}" != "${runtime_isa}" ]]; then
            echo "error: sealed CPU prefill plan emitted unknown ISA regime ${regime_name}" >&2
            exit 2
          fi
          if [[ ! "${format_phase_count}" =~ ^[1-9][0-9]*$ ]]; then
            echo "error: sealed CPU prefill plan emitted invalid format phase count ${format_phase_count}" >&2
            exit 2
          fi
          partial="${output_dir}/cpu_prefill.sealed-v8.${format_token}.${shape}.${regime_name}.csv"
          timing_partial="${output_dir}/cpu_prefill.sealed-v8.${format_token}.${shape}.${regime_name}.timing.csv"
          sealed_partials+=("${partial}")
          sealed_timing_partials+=("${timing_partial}")
          # A finalized sealed shard is immutable evidence, not disposable
          # scratch. Reuse it on every replay; --resume-cpu-partials controls
          # incomplete collection elsewhere but never authorizes overwriting
          # a complete post-freeze witness transaction.
          if [[ -s "${partial}" && -s "${timing_partial}" ]]; then
            continue
          fi
          # The coalesced serializer keeps identical format-phase counts and M
          # inventories contiguous. Preserve both boundaries in MPMD so socket
          # ranks enter the same number of globally coordinated warmup/timing
          # phases. An odd group member runs alone rather than borrowing an
          # incompatible job from the next phase inventory.
          local sealed_phase_inventory="${format_phase_count}:${m_values}"
          if (( ! sealed_pending_group_open )) ||
             [[ "${sealed_phase_inventory}" != "${sealed_pending_phase_inventory}" ]]; then
            if (( sealed_pending_group_open )); then
              sealed_group_ends+=("${#sealed_job_shapes[@]}")
            fi
            sealed_group_starts+=("${#sealed_job_shapes[@]}")
            sealed_pending_phase_inventory="${sealed_phase_inventory}"
            sealed_pending_group_open=1
          fi
          sealed_job_formats+=("${format_spec}")
          sealed_job_shapes+=("${shape}")
          sealed_job_ns+=("${n}")
          sealed_job_ks+=("${k}")
          sealed_job_ms+=("${m_values}")
          sealed_job_runtime_isas+=("${runtime_isa}")
          sealed_job_bins+=("${regime_binary[${regime_name}]}")
          sealed_job_partials+=("${partial}")
          sealed_job_timing_partials+=("${timing_partial}")
        done < <(
          PYTHONPATH="${shape_manifest_python_root}" \
            python3 -m "${cpu_prefill_training_plan_module}" \
              --sealed-witness-plan "${cpu_prefill_sealed_witness_plan_json}" \
              --coalesce-format-fixtures \
              --source-formats "${cpu_formats}" \
              "${cpu_prefill_route_args[@]}"
        )
        if (( sealed_pending_group_open )); then
          sealed_group_ends+=("${#sealed_job_shapes[@]}")
        fi
        run_cpu_prefill_measurement_jobs \
          1 sealed_group_starts sealed_group_ends sealed_job_formats \
          sealed_job_shapes sealed_job_ns sealed_job_ks sealed_job_ms \
          sealed_job_runtime_isas sealed_job_bins sealed_job_partials \
          sealed_job_timing_partials
        if (( cpu_prefill_collection_limited )); then
          printf 'CPU NativeVNNI sealed-v8 checkpoint: completed %d pending MPMD batch(es); rerun with --resume-cpu-partials to continue in %s\n' \
            "${cpu_prefill_batches_run}" "${output_dir}"
          finish_backend_collection_target
          return
        fi
        combine_csvs "${cpu_prefill_sealed_csv}" "${sealed_partials[@]}"
        combine_csvs "${cpu_prefill_sealed_timing_csv}" \
          "${sealed_timing_partials[@]}"
      fi
    fi

    local -a cpu_prefill_certify=(
      python3 "${cpu_prefill_generator}"
      --sealed-input "${cpu_prefill_sealed_csv}"
      --sealed-timing-sidecar "${cpu_prefill_sealed_timing_csv}"
      --frozen-policy-json "${cpu_prefill_frozen_policy_json}"
      --sealed-witness-plan-json "${cpu_prefill_sealed_witness_plan_json}"
      --policy-json "${cpu_prefill_policy_json}"
      --certification-diagnostic "${cpu_prefill_certification_diagnostic}"
      --fit-cache-dir "${cpu_prefill_fit_cache_dir}"
      --development-profiler-requests "${cpu_prefill_profiler_requests}"
      --development-profiler-evidence "${cpu_prefill_profiler_evidence}"
      --split-manifest "${cpu_prefill_split_manifest_path}"
      --output "${cpu_prefill_inc}"
      --summary "${cpu_prefill_summary}"
      --common-observations "${cpu_prefill_policy_common_csv}"
      --certify-generic
      --profile production
      --run-id "${cpu_prefill_run_id}"
      --git-revision "${cpu_development_git_revision}"
      --build-id "${cpu_development_build_id}"
      --compiler-id "${cpu_development_compiler_id}"
      --architecture-class "${cpu_development_arch_class}"
      --device-name "${cpu_development_device_name}"
      --driver-runtime "${cpu_development_driver_runtime}"
      --serial-m1-policy-hash "${cpu_development_serial_policy_hash}"
      --sealed-git-revision "${cpu_measurement_git_revision}"
      --sealed-build-id "${cpu_measurement_build_id}"
      --sealed-compiler-id "${cpu_measurement_compiler_id}"
      --sealed-architecture-class "${cpu_measurement_arch_class}"
      --sealed-device-name "${cpu_measurement_device_name}"
      --sealed-driver-runtime "${cpu_measurement_driver_runtime}"
      --sealed-serial-m1-policy-hash
      "${cpu_measurement_serial_policy_hash}"
    )
    local certify_development_index
    for ((certify_development_index = 0;
         certify_development_index < ${#cpu_prefill_active_development_csvs[@]};
         ++certify_development_index)); do
      cpu_prefill_certify+=(
        --development-input
        "${cpu_prefill_active_development_csvs[${certify_development_index}]}"
        --development-timing-sidecar
        "${cpu_prefill_active_development_timing_csvs[${certify_development_index}]}"
      )
    done
    cpu_prefill_certify+=("${cpu_prefill_candidate_expansion_fit_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_policy_checkpoint_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_burned_seal_fit_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_authenticated_context_args[@]}")
    if (( reuse_cpu_prefill_profiler_evidence )); then
      cpu_prefill_certify+=(
        --development-profiler-observations
        "${cpu_prefill_profiler_witnesses}"
      )
    fi
    cpu_prefill_certify+=("${cpu_prefill_additive_profiler_fit_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_generic_refinement_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_development_lineage_args[@]}")
    cpu_prefill_certify+=("${cpu_prefill_route_args[@]}")
    run_cmd "${cpu_prefill_certify[@]}"
    run_cmd python3 "${dispatch_validator}" "${cpu_prefill_inc}"
    run_cmd \
      "PYTHONPATH=${shape_manifest_python_root}" \
      python3 -m native_vnni_dispatch.policy_artifact \
      --policy-json "${cpu_prefill_policy_json}" \
      --include "${cpu_prefill_inc}"
    if (( collect_cpu_prefill_sealed_after_freeze )); then
      compose_cpu_prefill_final_profiler_evidence
    else
      collect_backend_profiler_evidence \
        cpu "${cpu_prefill_common_csv}" "${cpu_avx2_sweep_bin}" \
        "${cpu_prefill_final_profiler_requests}" \
        "${cpu_prefill_final_profiler_evidence}" \
        "${cpu_prefill_final_profiler_features}" \
        "${cpu_prefill_final_profiler_raw}" \
        auto
    fi
    if (( install )); then
      local cpu_prefill_install_target="${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc"
      run_cmd cp "${cpu_prefill_inc}" \
        "${cpu_prefill_install_target}.inprogress"
      run_cmd mv "${cpu_prefill_install_target}.inprogress" \
        "${cpu_prefill_install_target}"
    fi
    finish_backend_collection_target
    return
  fi

  local generator_args=(
    python3 "${cpu_prefill_generator}"
    --input "${cpu_prefill_csv}"
    --timing-sidecar "${cpu_prefill_timing_csv}"
    --output "${cpu_prefill_inc}"
    --summary "${cpu_prefill_summary}"
    --common-observations "${cpu_prefill_common_csv}"
    --require-isa-matrix
    --profile "${measurement_profile}"
    --run-id "${timestamp}-cpu-prefill-${profile}"
    --git-revision "${cpu_git_revision}"
    --build-id "${cpu_build_id}"
    --compiler-id "${cpu_compiler_id}"
    --architecture-class "${cpu_arch_class}"
    --device-name "${cpu_device_name}"
    --driver-runtime "${cpu_driver_runtime}"
    --serial-m1-policy-hash "${cpu_serial_policy_hash}"
  )
  generator_args+=("${cpu_prefill_route_args[@]}")
  if [[ "${measurement_profile}" == "production" ]]; then
    generator_args+=(--require-complete)
  fi
  run_cmd "${generator_args[@]}"
  run_cmd python3 "${dispatch_validator}" "${cpu_prefill_inc}"
  collect_backend_profiler_evidence \
    cpu "${cpu_prefill_common_csv}" "${cpu_avx2_sweep_bin}" \
    "${cpu_prefill_profiler_requests}" "${cpu_prefill_profiler_evidence}" \
    "${cpu_prefill_profiler_features}" "${cpu_prefill_profiler_raw}"
  if (( install )); then
    run_cmd cp "${cpu_prefill_inc}" \
      "${repo_root}/src/v2/kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc"
  fi
  finish_backend_collection_target
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
  cpu-prefill)
    refresh_cpu_prefill
    ;;
  both)
    refresh_cuda
    refresh_rocm
    ;;
  all)
    refresh_cuda
    refresh_rocm
    refresh_cpu
    refresh_cpu_prefill
    ;;
esac

printf 'NativeVNNI dispatch refresh artifacts: %s\n' "${output_dir}"
