#!/usr/bin/env python3
"""Regression tests for isolated NativeVNNI profiler sidecars."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    cpu_native_vnni_decode_registry,
    cuda_native_vnni_gemv_registry,
)
from native_vnni_dispatch.corpus import (  # noqa: E402
    ObservationCorpus,
    runtime_key,
)
from native_vnni_dispatch.format_registry import format_spec  # noqa: E402
from native_vnni_dispatch.profiler_collectors import (  # noqa: E402
    CollectorOptions,
    _append_checkpoint_journal,
    _apply_device_placement,
    _build_cpu_process_batches,
    _collect_cpu,
    _collect_cpu_batch,
    _cpu_batch_environment,
    _parse_cpu_lane_lists,
    _load_incremental_evidence,
    _profile_environment,
    _tool_command,
    _trainer_arguments,
    _write_binary_provenance,
    _write_checkpoint,
    _write_cpu_batch_plan,
    collect_request,
    parse_ncu_csv,
    parse_perf_stat,
    parse_rocprof_csvs,
)
from native_vnni_dispatch.profiler_evidence import (  # noqa: E402
    EXPECTED_PROFILER_TOOL,
    LEGACY_PROFILER_REQUEST_SCHEMA_VERSION,
    PROFILER_COLLECTOR_VERSION,
    PROFILER_METRIC_SET_VERSION,
    MetricAvailability,
    ProfiledDispatch,
    ProfiledDispatchKind,
    ProfilerEvidence,
    ProfilerEvidenceManifest,
    ProfilerEvidenceStatus,
    ProfilerMetric,
    ProfilerRequestManifest,
    build_missing_profiler_request_transaction,
    build_profiler_request_manifest,
    count_uncovered_profiler_requests,
    compose_profiler_evidence,
    compact_profiler_observation_witnesses,
    metric_definitions,
    profiler_feature_rows,
    profiler_request_for_observation,
    read_profiler_feature_observation_witnesses,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
    validate_profiler_evidence_coverage,
    write_profiler_feature_csv,
    write_profiler_evidence_manifest,
    write_profiler_request_manifest,
)
import native_vnni_dispatch.profiler_evidence as profiler_evidence  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    _center_complete_candidate_relative_auxiliary,
    _cpu_decode_schedule_features,
    _mask_held_out_profiler_anchor,
    _model_record,
    _normalized_metric_features,
    _cpu_prefill_schedule_features,
    _profiler_auxiliary_target_record,
    _profiler_model_input_record,
    apply_profiler_regret_predictions,
    build_profiler_feature_catalog,
    build_profiler_observation_index,
    load_profiler_feature_catalog,
    merge_profiler_feature_catalogs,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    Backend,
    ExecutionMode,
    FEATURE_SCHEMA_VERSION,
    LEARNER_VERSION,
    NativeVNNIObservation,
    P95_REGRET_BUDGET,
    POLICY_ABI,
    SCHEMA_VERSION,
    SemanticContract,
    classify_aspect,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    CandidatePointCost,
    PolicyFitCache,
    ProfilerInfluence,
    _domain_fold_tasks,
    _domain_profiler_prediction_requests,
    _domain_profiler_prediction_request_keys,
    _domain_profiler_teacher_fold_results,
    _fit_profiler_prediction_pool,
    _populate_profiler_prediction_cache,
    _profiler_transfer_key,
    _rank_domain_cross_validations,
    fit_domain_rules,
    fit_generic_policy,
)


def cuda_observation(
    *,
    source_format: str = "Q4_0",
    candidate_id: str = "cuda.nvnni.decode.fast_m1.wide.tn128.cpt1",
    supported: bool = True,
    forced_route_ok: bool = True,
) -> NativeVNNIObservation:
    """Build one registry-authentic timing observation for sidecar tests."""

    registry = cuda_native_vnni_gemv_registry()
    candidate = registry.resolve(candidate_id)
    spec = format_spec(source_format)
    n = 2048
    k = 4096
    result = NativeVNNIObservation(
        schema_version=SCHEMA_VERSION,
        run_id="profile-unit-run",
        corpus_id="sha256:profile-unit-corpus",
        git_revision="0123456789abcdef",
        build_id="profile-unit-release-build",
        compiler_id="profile-unit-nvcc",
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        backend=Backend.CUDA,
        architecture_class="unit-sm80-native-vnni-v1",
        device_name="unit-cuda-device",
        driver_runtime="unit-cuda-runtime",
        threading_or_stream_mode="explicit_non_default_stream",
        semantic_contract=SemanticContract.FAST,
        operation_kind="NativeVNNIDecodeProjection",
        bundle_signature="single-native-vnni-projection:fp32-output:v2",
        projection_n_vector=(n,),
        source_format=spec.label,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cuda"),
        packing_abi=spec.packing_abi("cuda"),
        runtime_codebook_id=spec.runtime_codebook("cuda"),
        shape_group_id="profile-unit-shape-group",
        shape_name="profile-unit-shape",
        execution_mode=ExecutionMode.GRAPH_CAPTURED,
        m=1,
        aggregate_n=n,
        k=k,
        aspect_ratio=float(n) / float(k),
        aspect_bucket=classify_aspect(n, k),
        work_items=n * k,
        n_tail_class="n_mod_256=0",
        k_tail_class="k_mod_256=0",
        alignment_class="cuda_gpu_prepared_native_vnni_16b_aligned",
        candidate_id=candidate.candidate_id,
        effective_candidate_id=candidate.effective_candidate_id,
        candidate_family=candidate.candidate_family,
        config_json=candidate.config_json,
        supported=supported,
        graph_capture_ok=True,
        generic_eligible=True,
        arithmetic_fingerprint=candidate.arithmetic_fingerprint,
        serial_m1_policy_id="cuda.production.serial.m1",
        serial_m1_policy_hash="sha256:serial-policy",
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction,
        uses_atomic_reduction=candidate.uses_atomic_reduction,
        trial_set_hash="sha256:profile-unit-trials",
        numerical_correctness=True,
        bitwise_equal=True,
        repeat_equal=True,
        mismatch_count=0,
        first_mismatch_index=None,
        grouped_output_digest="sha256:profile-unit-output",
        serial_output_digest="sha256:profile-unit-output",
        max_abs=0.0,
        relative_l2=0.0,
        cosine=1.0,
        symmetric_kld=0.0,
        warmup_count=5,
        sample_count=30,
        min_us=10.0,
        median_us=11.0,
        p95_us=12.0,
        mad_us=0.2,
        cv=0.01,
        timing_sample_hash="fnv1a64:0123456789abcdef",
        effective_bandwidth_gbs=100.0,
        forced_route_ok=forced_route_ok,
        observed_candidate_id=candidate.effective_candidate_id,
        route_counter_ok=forced_route_ok,
        workspace_ok=True,
        explicit_stream_ok=True,
    )
    result.validate()
    return result


def cpu_decode_observation(
    *,
    candidate_id: str = "cpu.nvnni.decode.n_chunk_grid.nbc1",
) -> NativeVNNIObservation:
    """Build one registry-authentic CPU M=1 timing observation."""

    candidate = cpu_native_vnni_decode_registry().resolve(candidate_id)
    spec = format_spec("Q4_0")
    result = dataclasses.replace(
        cuda_observation(),
        backend=Backend.CPU,
        architecture_class="unit-cascadelake|build=AVX2|runtime=AVX2|threads=28",
        device_name="unit-cpu-socket",
        driver_runtime="linux-perf-event-unit",
        threading_or_stream_mode=(
            "openmp:build=AVX2:requested=AVX2:effective=AVX2:threads=28"
        ),
        operation_kind="NativeVNNIFastM1Projection",
        bundle_signature="single-native-vnni-decode:serial-full-k:fp32-output:v1",
        source_format=spec.label,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cpu"),
        packing_abi=spec.packing_abi("cpu"),
        runtime_codebook_id=spec.runtime_codebook("cpu"),
        shape_group_id="cpu-decode:unit-shape:n2048:k4096",
        shape_name="unit-cpu-decode-shape",
        execution_mode=ExecutionMode.EAGER,
        candidate_id=candidate.candidate_id,
        effective_candidate_id=candidate.effective_candidate_id,
        candidate_family=candidate.candidate_family,
        config_json=candidate.config_json,
        graph_capture_ok=False,
        arithmetic_fingerprint=candidate.arithmetic_fingerprint,
        serial_m1_policy_id="cpu.native_vnni.serial_m1.frozen",
        serial_m1_policy_hash="sha256:cpu-serial-policy",
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction,
        uses_atomic_reduction=candidate.uses_atomic_reduction,
        observed_candidate_id=candidate.effective_candidate_id,
        explicit_stream_ok=False,
    )
    result.validate()
    return result


def complete_metrics(
    backend: Backend,
    *,
    scale: float = 1.0,
) -> tuple[ProfilerMetric, ...]:
    """Create a complete measured v1 metric inventory for one dispatch."""

    return tuple(
        ProfilerMetric(
            metric_id=definition.metric_id,
            category=definition.category,
            unit=definition.unit,
            availability=MetricAvailability.MEASURED,
            value=scale * float(index + 1),
            source_name=f"unit::{definition.metric_id}",
            reason=None,
        )
        for index, definition in enumerate(metric_definitions(backend))
    )


def cuda_dispatch(
    index: int = 0,
    name: str = "unitKernel",
    *,
    metric_scale: float = 1.0,
) -> ProfiledDispatch:
    """Create one complete synthetic CUDA profiler dispatch."""

    return ProfiledDispatch(
        dispatch_index=index,
        dispatch_kind=ProfiledDispatchKind.GPU_KERNEL,
        kernel_name=name,
        kernel_fingerprint=(
            "sha256:" + f"{index + 1:064x}"
        ),
        grid=(16, 1, 1),
        block=(128, 1, 1),
        metrics=complete_metrics(Backend.CUDA, scale=metric_scale),
    )


def complete_evidence(request, dispatches=None) -> ProfilerEvidence:
    """Build valid complete evidence for one supported request."""

    return ProfilerEvidence(
        request_id=request.request_id,
        observation_digest=request.observation_digest,
        backend=request.backend,
        status=ProfilerEvidenceStatus.COMPLETE,
        status_reason=None,
        profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
        profiler_tool_version="unit-profiler-1.0",
        metric_set_version=PROFILER_METRIC_SET_VERSION,
        collector_version=PROFILER_COLLECTOR_VERSION,
        command_digest="sha256:" + "1" * 64,
        raw_artifact_digest="sha256:" + "2" * 64,
        profiler_pass_count=3,
        target_launches_per_profiler_pass=1,
        dispatches=tuple(dispatches or (cuda_dispatch(),)),
    )


def evidence_manifest(requests, evidence) -> ProfilerEvidenceManifest:
    """Bind records to one request manifest with exact provenance."""

    return ProfilerEvidenceManifest(
        request_manifest_digest=requests.digest(),
        corpus_digest=requests.corpus_digest,
        candidate_registry_digest=requests.candidate_registry_digest,
        evidence=tuple(evidence),
    )


def profiler_model_fixture(
    *,
    second_candidate_metric_scale: float = 8.0,
):
    """Build a small complete two-candidate profiler-informed fit surface."""

    candidate_ids = (
        "cuda.nvnni.decode.fast_m1.wide.tn128.cpt1",
        "cuda.nvnni.decode.fast_m1.wide.tn128.cpt2",
    )
    dimensions = (
        (1024, 2048),
        (1536, 3072),
        (2048, 4096),
        (3072, 6144),
        (4096, 8192),
    )
    rows = []
    costs = []
    for shape_index, (n, k) in enumerate(dimensions):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            median_us = 10.0 + shape_index + candidate_index
            row = dataclasses.replace(
                cuda_observation(candidate_id=candidate_id),
                projection_n_vector=(n,),
                aggregate_n=n,
                k=k,
                aspect_ratio=float(n) / float(k),
                aspect_bucket=classify_aspect(n, k),
                work_items=n * k,
                shape_group_id=f"profile-model-shape-{shape_index}",
                shape_name=f"profile-model-{n}x{k}",
                n_tail_class=f"n_mod_256={n % 256}",
                k_tail_class=f"k_mod_256={k % 256}",
                min_us=median_us - 0.5,
                median_us=median_us,
                p95_us=median_us + 0.5,
            )
            rows.append(row)
            regret = (
                (0.01 + shape_index * 0.004)
                if candidate_index == 0
                else (0.12 - shape_index * 0.012)
            )
            costs.append(CandidatePointCost(
                runtime_key=runtime_key(row),
                shape_group_id=row.shape_group_id,
                candidate_id=row.candidate_id,
                max_surface_regret=regret,
                p95_surface_regret=regret,
                mean_surface_regret=regret,
            ))
    corpus = ObservationCorpus(tuple(rows))
    requests = build_profiler_request_manifest(corpus)
    evidence = evidence_manifest(requests, tuple(
        complete_evidence(
            request,
            dispatches=(cuda_dispatch(
                name=request.effective_candidate_id,
                metric_scale=(
                    1.0
                    if request.candidate_id == candidate_ids[0]
                    else second_candidate_metric_scale
                ),
            ),),
        )
        for request in requests.requests
    ))
    feature_rows = profiler_feature_rows(corpus, requests, evidence)
    catalog = build_profiler_feature_catalog(
        corpus,
        feature_rows,
        request_manifest_digest=requests.digest(),
        evidence_manifest_digest=evidence.digest(),
    )
    return corpus, costs, requests, evidence, feature_rows, catalog


class NativeVNNIProfilerEvidenceTest(unittest.TestCase):
    """Prove timing isolation, request identity, and complete profile coverage."""

    def test_cpu_profiler_lanes_are_disjoint_and_preserve_socket_masks(self) -> None:
        self.assertEqual(
            _parse_cpu_lane_lists("0-27;28-55", 2),
            ("0-27", "28-55"),
        )

        with self.assertRaisesRegex(ValueError, "overlaps"):
            _parse_cpu_lane_lists("0-27;27-54", 2)
        with self.assertRaisesRegex(ValueError, "defines 1"):
            _parse_cpu_lane_lists("0-27", 2)

    def test_request_is_bound_to_complete_timing_observation(self) -> None:
        observation = cuda_observation()
        request = profiler_request_for_observation(observation)

        self.assertEqual(request.observation_digest, observation.digest())
        self.assertEqual(request.timing_sample_hash, observation.timing_sample_hash)
        self.assertEqual(request.target_launches_per_profiler_pass, 1)
        self.assertTrue(request.profile_required)
        self.assertEqual(
            request.candidate_registry_surface, "cuda_native_vnni_gemv"
        )

        changed_timing = dataclasses.replace(observation, median_us=11.5)
        changed_request = profiler_request_for_observation(changed_timing)
        self.assertNotEqual(request.request_id, changed_request.request_id)
        self.assertNotEqual(
            request.observation_digest, changed_request.observation_digest
        )

    def test_large_manifest_digests_are_memoized_without_changing_json(self) -> None:
        """Repeated transaction validation must not rehash complete manifests."""

        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        evidence = evidence_manifest(
            requests,
            (complete_evidence(requests.requests[0]),),
        )
        expected_request_digest = profiler_evidence._sha256_json(
            requests.payload_mapping()
        )
        expected_evidence_digest = profiler_evidence._sha256_json(
            evidence.payload_mapping()
        )
        request_mapping = requests.canonical_mapping()
        evidence_mapping = evidence.canonical_mapping()

        # Rebuild the manifests so the first digest call below is not already
        # primed by evidence construction or canonical serialization.
        requests = ProfilerRequestManifest(
            corpus_digest=requests.corpus_digest,
            candidate_registry_digest=requests.candidate_registry_digest,
            requests=requests.requests,
            learner_version=requests.learner_version,
            feature_schema_version=requests.feature_schema_version,
            schema_version=requests.schema_version,
        )
        evidence = ProfilerEvidenceManifest(
            request_manifest_digest=requests.digest(),
            corpus_digest=evidence.corpus_digest,
            candidate_registry_digest=evidence.candidate_registry_digest,
            evidence=evidence.evidence,
            collector_version=evidence.collector_version,
        )
        # Prime neither manifest inside the patched interval. Reconstructing
        # evidence necessarily hashes the request manifest once immediately
        # before the patch, which is exactly the value later calls should reuse.
        requests = ProfilerRequestManifest(
            corpus_digest=requests.corpus_digest,
            candidate_registry_digest=requests.candidate_registry_digest,
            requests=requests.requests,
            learner_version=requests.learner_version,
            feature_schema_version=requests.feature_schema_version,
            schema_version=requests.schema_version,
        )

        with mock.patch.object(
            profiler_evidence,
            "_sha256_manifest_records",
            wraps=profiler_evidence._sha256_manifest_records,
        ) as streaming_digest:
            request_digest = requests.digest()
            self.assertEqual(request_digest, requests.digest())
            evidence_digest = evidence.digest()
            self.assertEqual(evidence_digest, evidence.digest())

        self.assertEqual(streaming_digest.call_count, 2)
        self.assertEqual(request_digest, expected_request_digest)
        self.assertEqual(evidence_digest, expected_evidence_digest)
        self.assertEqual(requests.canonical_mapping(), request_mapping)
        self.assertEqual(evidence.canonical_mapping(), evidence_mapping)

    def test_cpu_decode_request_resolves_first_class_candidate_surface(self) -> None:
        """Every Fast M=1 timing candidate must own one profiler request."""

        manifest = build_profiler_request_manifest(ObservationCorpus(tuple(
            cpu_decode_observation(candidate_id=candidate.candidate_id)
            for candidate in cpu_native_vnni_decode_registry().entries
        )))

        self.assertEqual(len(manifest.requests), 5)
        self.assertEqual(
            {request.candidate_registry_surface for request in manifest.requests},
            {"cpu_native_vnni_decode"},
        )
        self.assertEqual(
            {request.operation_kind for request in manifest.requests},
            {"NativeVNNIFastM1Projection"},
        )

    def test_manifest_separates_distinct_prepared_codebook_kernels(self) -> None:
        """A shared schedule ID must not alias different microkernels."""

        q4 = cuda_observation(source_format="Q4_0")
        q5 = cuda_observation(source_format="Q5_0")
        manifest = build_profiler_request_manifest(ObservationCorpus((q4, q5)))

        self.assertEqual(len(manifest.requests), 2)
        self.assertEqual(
            {request.source_format for request in manifest.requests},
            {"Q4_0", "Q5_0"},
        )

    def test_runtime_format_width_and_counters_are_codebook_local(self) -> None:
        """Each prepared kernel keeps both its byte geometry and counters."""

        q4 = cuda_observation(source_format="Q4_0")
        q5 = cuda_observation(source_format="Q5_0")
        corpus = ObservationCorpus((q4, q5))
        requests = build_profiler_request_manifest(corpus)
        evidence = evidence_manifest(
            requests,
            tuple(complete_evidence(request) for request in requests.requests),
        )
        rows = profiler_feature_rows(corpus, requests, evidence)
        catalog = build_profiler_feature_catalog(
            corpus,
            rows,
            request_manifest_digest=requests.digest(),
            evidence_manifest_digest=evidence.digest(),
        )
        q4_descriptor = catalog.descriptor_for(q4)
        q5_descriptor = catalog.descriptor_for(q5)

        self.assertNotEqual(q4_descriptor.key, q5_descriptor.key)
        q4_record = _model_record(
            runtime_key(q4), q4_descriptor, q4.source_format
        )
        q5_record = _model_record(
            runtime_key(q5), q5_descriptor, q5.source_format
        )

        self.assertNotEqual(
            q4_record["runtime.decode.log2_expected_bytes"],
            q5_record["runtime.decode.log2_expected_bytes"],
        )
        self.assertIn("runtime.decode.arithmetic_intensity", q4_record)

    def test_cpu_prefill_schedule_features_match_production_task_grids(self) -> None:
        """Candidate-aware features reproduce all three full-K launch formulas."""

        base = runtime_key(cuda_observation())
        key = dataclasses.replace(
            base,
            backend=Backend.CPU,
            architecture_class=(
                "unit-cpu|build=AVX512|runtime=AVX512|threads=28"
            ),
            operation_kind="NativeVNNIPrefillProjection",
            m=64,
            aggregate_n=2048,
            k=512,
            projection_n_vector=(2048,),
        )
        row_chunk = _cpu_prefill_schedule_features(
            key,
            {"config.route": "row_chunk_grid"},
        )
        n_major = _cpu_prefill_schedule_features(
            key,
            {
                "config.route": "two_row_full_output_tiles",
                "config.n_block_chunks": math.log1p(1.0),
            },
        )
        pair_grid = _cpu_prefill_schedule_features(
            key,
            {
                "config.route": "two_row_pair_grid",
                "config.n_block_chunks": math.log1p(1.0),
            },
        )

        self.assertEqual(
            row_chunk["schedule.cpu_prefill.parallel_tasks"],
            64 * 32,
        )
        self.assertEqual(
            n_major["schedule.cpu_prefill.parallel_tasks"],
            32,
        )
        self.assertEqual(
            pair_grid["schedule.cpu_prefill.parallel_tasks"],
            32 * 32,
        )
        self.assertEqual(
            row_chunk["schedule.cpu_prefill.parallel_waves"],
            74,
        )
        self.assertEqual(
            n_major["schedule.cpu_prefill.parallel_waves"],
            2,
        )
        self.assertEqual(
            pair_grid["schedule.cpu_prefill.parallel_waves"],
            37,
        )
        self.assertEqual(
            row_chunk["schedule.cpu_prefill.rows_per_task"],
            1,
        )
        self.assertEqual(
            n_major["schedule.cpu_prefill.rows_per_task"],
            64,
        )
        self.assertEqual(
            pair_grid["schedule.cpu_prefill.rows_per_task"],
            2,
        )
        self.assertAlmostEqual(
            pair_grid["schedule.cpu_prefill.final_wave_utilization"],
            16.0 / 28.0,
        )

    def test_cpu_decode_schedule_features_match_production_task_grids(self) -> None:
        """Expose full-K serial and K-part workshare transitions exactly."""

        base = runtime_key(cuda_observation())
        full_k = dataclasses.replace(
            base,
            backend=Backend.CPU,
            architecture_class=(
                "unit-cpu|build=AVX512|runtime=AVX2|threads=28"
            ),
            operation_kind="NativeVNNIFastM1Projection",
            bundle_signature=(
                "single-native-vnni-decode:serial-full-k:fp32-output:v1"
            ),
            m=1,
            aggregate_n=2048,
            k=4096,
            projection_n_vector=(2048,),
        )
        nbc1 = _cpu_decode_schedule_features(
            full_k,
            {
                "config.route": "n_chunk_grid",
                "config.n_block_chunks": math.log1p(1.0),
            },
        )
        nbc2 = _cpu_decode_schedule_features(
            full_k,
            {
                "config.route": "n_chunk_grid",
                "config.n_block_chunks": math.log1p(2.0),
            },
        )
        kpart_nbc2 = _cpu_decode_schedule_features(
            dataclasses.replace(
                full_k,
                bundle_signature=(
                    "single-native-vnni-decode:serial-kpart:fp32-output:v1"
                ),
            ),
            {
                "config.route": "n_chunk_grid",
                "config.n_block_chunks": math.log1p(2.0),
            },
        )

        self.assertEqual(nbc1["schedule.cpu_decode.n_chunks"], 32)
        self.assertEqual(nbc1["schedule.cpu_decode.n_blocks"], 32)
        self.assertEqual(nbc1["schedule.cpu_decode.n_block_waves"], 2)
        self.assertAlmostEqual(
            nbc1["schedule.cpu_decode.final_n_block_wave_utilization"],
            4.0 / 28.0,
        )
        self.assertEqual(
            nbc1["schedule.cpu_decode.serial_full_k_fast_path"],
            0.0,
        )

        # Sixteen N blocks are fewer than 28 workers, so production bypasses
        # OpenMP and executes exactly sixteen serial blocks.
        self.assertEqual(nbc2["schedule.cpu_decode.n_blocks"], 16)
        self.assertEqual(
            nbc2["schedule.cpu_decode.serial_full_k_fast_path"],
            1.0,
        )
        self.assertEqual(nbc2["schedule.cpu_decode.openmp_region"], 0.0)
        self.assertEqual(
            nbc2["schedule.cpu_decode.serial_full_k_block_count"],
            16.0,
        )

        # The same NBC under K-part stays inside OpenMP and publishes one
        # candidate-dependent N-block axis plus the exact 32-task reduction.
        self.assertEqual(
            kpart_nbc2["schedule.cpu_decode.serial_full_k_fast_path"],
            0.0,
        )
        self.assertEqual(kpart_nbc2["schedule.cpu_decode.openmp_region"], 1.0)
        self.assertEqual(
            kpart_nbc2["schedule.cpu_decode.kpart_producer_n_blocks"],
            16.0,
        )
        self.assertEqual(
            kpart_nbc2["schedule.cpu_decode.kpart_reduction_tasks"],
            32.0,
        )

    def test_profiler_record_retains_anchor_and_masks_held_dynamic_metrics(
        self,
    ) -> None:
        """A held geometry may retain static resources but not its counters."""

        observation = cuda_observation()
        corpus = ObservationCorpus((observation,))
        requests = build_profiler_request_manifest(corpus)
        evidence = evidence_manifest(
            requests,
            tuple(
                complete_evidence(request)
                for request in requests.requests
            ),
        )
        catalog = build_profiler_feature_catalog(
            corpus,
            profiler_feature_rows(corpus, requests, evidence),
            request_manifest_digest=requests.digest(),
            evidence_manifest_digest=evidence.digest(),
        )
        descriptor = catalog.descriptor_for(observation)
        record = _model_record(
            runtime_key(observation),
            descriptor,
            observation.source_format,
        )

        self.assertEqual(
            descriptor.anchor_geometry,
            (observation.aggregate_n, observation.k),
        )
        self.assertEqual(record["profile.anchor.abs_delta_log2_m"], 0.0)
        self.assertEqual(record["profile.anchor.abs_delta_log2_n"], 0.0)
        self.assertEqual(record["profile.anchor.abs_delta_log2_k"], 0.0)
        self.assertIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            record,
        )
        self.assertIn(
            "metric.gpu.registers_per_thread.maximum_log1p",
            record,
        )

        masked = _mask_held_out_profiler_anchor(
            record,
            frozenset((descriptor.anchor_geometry,)),
        )

        self.assertEqual(
            masked["profile.anchor.dynamic_metrics_available"],
            0.0,
        )
        self.assertNotIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            masked,
        )
        self.assertIn(
            "metric.gpu.registers_per_thread.maximum_log1p",
            masked,
        )
        self.assertFalse(any(
            "metric.gpu.decode.effective_gbytes_per_second_log1p" in name
            for name in masked
        ))

        model_inputs = _profiler_model_input_record(record)
        auxiliary_targets = _profiler_auxiliary_target_record(record)
        self.assertNotIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            model_inputs,
        )
        self.assertIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            auxiliary_targets,
        )
        self.assertIn(
            "metric.gpu.registers_per_thread.maximum_log1p",
            model_inputs,
        )
        self.assertNotIn(
            "metric.gpu.registers_per_thread.maximum_log1p",
            auxiliary_targets,
        )
        self.assertFalse(any(
            name.startswith("profile.anchor.") for name in model_inputs
        ))

    def test_catalog_rejects_missing_codebook_specific_profile(self) -> None:
        """One codebook's counters cannot satisfy another codebook's gate."""

        q4 = cuda_observation(source_format="Q4_0")
        q5 = cuda_observation(source_format="Q5_0")
        corpus = ObservationCorpus((q4, q5))
        q4_requests = build_profiler_request_manifest(ObservationCorpus((q4,)))
        evidence = evidence_manifest(
            q4_requests,
            tuple(
                complete_evidence(request)
                for request in q4_requests.requests
            ),
        )
        rows = profiler_feature_rows(
            ObservationCorpus((q4,)), q4_requests, evidence
        )

        with self.assertRaisesRegex(
            ValueError,
            (
                r"omits physical candidate .*"
                + q5.prepared_family_id
                + rf"/cb{q5.runtime_codebook_id}"
            ),
        ):
            build_profiler_feature_catalog(
                corpus,
                rows,
                request_manifest_digest=q4_requests.digest(),
                evidence_manifest_digest=evidence.digest(),
            )

    def test_additive_catalog_merge_requires_complete_physical_coverage(self) -> None:
        """Old and new counter transactions cover one expanded corpus jointly."""

        q4 = cuda_observation(source_format="Q4_0")
        q5 = cuda_observation(source_format="Q5_0")

        def catalog_for(observation: NativeVNNIObservation):
            source = ObservationCorpus((observation,))
            requests = build_profiler_request_manifest(source)
            evidence = evidence_manifest(
                requests,
                tuple(
                    complete_evidence(request)
                    for request in requests.requests
                ),
            )
            return build_profiler_feature_catalog(
                source,
                profiler_feature_rows(source, requests, evidence),
                request_manifest_digest=requests.digest(),
                evidence_manifest_digest=evidence.digest(),
            )

        q4_catalog = catalog_for(q4)
        q5_catalog = catalog_for(q5)
        expanded = ObservationCorpus((q4, q5))
        with self.assertRaisesRegex(ValueError, "omits physical candidate"):
            merge_profiler_feature_catalogs(expanded, (q4_catalog,))

        merged = merge_profiler_feature_catalogs(
            expanded,
            (q4_catalog, q5_catalog),
        )
        self.assertEqual(merged.corpus_digest, expanded.digest())
        self.assertEqual(len(merged.descriptors), 2)
        self.assertRegex(
            merged.request_manifest_digest,
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertRegex(
            merged.evidence_manifest_digest,
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(
            merged.descriptor_for(q4),
            q4_catalog.descriptor_for(q4),
        )
        self.assertEqual(
            merged.descriptor_for(q5),
            q5_catalog.descriptor_for(q5),
        )

    def test_additive_evidence_composition_preserves_exact_measured_launches(
        self,
    ) -> None:
        """Final publication unions source records without changing anchors."""

        sources = []
        original_request_ids = set()
        for source_index, observation in enumerate((
            cuda_observation(source_format="Q4_0"),
            cuda_observation(source_format="Q5_0"),
        )):
            corpus = ObservationCorpus((observation,))
            requests = build_profiler_request_manifest(corpus)
            if source_index == 1:
                # Candidate expansion is collected against the enlarged
                # registry while retaining the older base transaction.
                requests = dataclasses.replace(
                    requests,
                    candidate_registry_digest="sha256:" + "9" * 64,
                    learner_version="native-vnni-bounded-tree-beam-regret-v17",
                )
            evidence = evidence_manifest(
                requests,
                tuple(complete_evidence(request) for request in requests.requests),
            )
            sources.append((corpus, requests, evidence))
            original_request_ids.update(
                request.request_id for request in requests.requests
            )

        composed = compose_profiler_evidence(sources)

        self.assertEqual(len(composed.observations), 2)
        self.assertEqual(len(composed.requests.requests), 2)
        self.assertEqual(len(composed.evidence.evidence), 2)
        self.assertNotEqual(
            composed.requests.candidate_registry_digest,
            sources[0][1].candidate_registry_digest,
        )
        self.assertEqual(
            {request.request_id for request in composed.requests.requests},
            original_request_ids,
        )
        self.assertTrue(
            validate_profiler_evidence_coverage(
                composed.requests,
                composed.evidence,
            ).complete
        )
        self.assertEqual(
            len(profiler_feature_rows(
                composed.observations,
                composed.requests,
                composed.evidence,
            )),
            2,
        )

    def test_additive_composition_normalizes_compatible_feature_schemas(
        self,
    ) -> None:
        """Feature engineering changes must not trigger profiler relaunches."""

        sources = []
        for index, observation in enumerate((
            cuda_observation(source_format="Q4_0"),
            cuda_observation(source_format="Q5_0"),
        )):
            corpus = ObservationCorpus((observation,))
            requests = build_profiler_request_manifest(corpus)
            if index == 0:
                requests = dataclasses.replace(
                    requests,
                    feature_schema_version=(
                        "execution-mode-n-k-work-aspect-tile-wave-tree-v8"
                    ),
                )
            evidence = evidence_manifest(
                requests,
                tuple(
                    complete_evidence(request)
                    for request in requests.requests
                ),
            )
            sources.append((corpus, requests, evidence))

        composed = compose_profiler_evidence(sources)

        self.assertEqual(
            composed.requests.feature_schema_version,
            FEATURE_SCHEMA_VERSION,
        )
        self.assertEqual(len(composed.requests.requests), 2)
        self.assertEqual(len(composed.evidence.evidence), 2)
        self.assertEqual(
            len(profiler_feature_rows(
                composed.observations,
                composed.requests,
                composed.evidence,
            )),
            2,
        )

    def test_missing_request_transaction_profiles_only_new_exact_point(self) -> None:
        """A corpus extension keeps old launches and emits only its delta."""

        old = cuda_observation()
        new = dataclasses.replace(
            old,
            projection_n_vector=(3072,),
            aggregate_n=3072,
            aspect_ratio=3072.0 / float(old.k),
            aspect_bucket=classify_aspect(3072, old.k),
            work_items=3072 * old.k,
            shape_group_id="profile-unit-shape-group-new",
            shape_name="profile-unit-shape-new",
            trial_set_hash="sha256:profile-unit-trials-new",
            grouped_output_digest="sha256:profile-unit-output-new",
            serial_output_digest="sha256:profile-unit-output-new",
            timing_sample_hash="sha256:profile-unit-timing-new",
        )
        old_corpus = ObservationCorpus((old,))
        covered = build_profiler_request_manifest(old_corpus)

        observations, requests = build_missing_profiler_request_transaction(
            ObservationCorpus((old, new)),
            (covered,),
        )

        self.assertIsNotNone(observations)
        self.assertIsNotNone(requests)
        assert observations is not None
        assert requests is not None
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(requests.requests), 1)
        self.assertEqual(requests.requests[0].aggregate_n, 3072)

        no_rows, no_requests = build_missing_profiler_request_transaction(
            old_corpus,
            (covered,),
        )
        self.assertIsNone(no_rows)
        self.assertIsNone(no_requests)

    def test_profiler_resume_uses_physical_identity_across_run_provenance(self) -> None:
        """A timestamped replay must not profile an unchanged launch again."""

        original = cuda_observation()
        covered = build_profiler_request_manifest(ObservationCorpus((original,)))
        replay = dataclasses.replace(
            original,
            run_id="profile-unit-run-resumed",
            corpus_id="sha256:profile-unit-corpus-resumed",
            git_revision="profile-unit-revision-resumed",
        )
        replay_requests = build_profiler_request_manifest(
            ObservationCorpus((replay,))
        )

        self.assertNotEqual(
            covered.requests[0].request_id,
            replay_requests.requests[0].request_id,
        )
        self.assertEqual(
            count_uncovered_profiler_requests(replay_requests, (covered,)),
            0,
        )
        missing_rows, missing_requests = build_missing_profiler_request_transaction(
            ObservationCorpus((replay,)),
            (covered,),
        )
        self.assertIsNone(missing_rows)
        self.assertIsNone(missing_requests)

    def test_profiler_resume_invalidates_changed_kernel_contract(self) -> None:
        """Changed arithmetic or scheduling is another physical obligation."""

        current = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        request = current.requests[0]
        stale = dataclasses.replace(
            current,
            requests=(dataclasses.replace(
                request,
                arithmetic_fingerprint="sha256:changed-arithmetic-contract",
                schedule_signature=request.schedule_signature + "-changed",
            ),),
        )

        self.assertEqual(
            count_uncovered_profiler_requests(current, (stale,)),
            1,
        )

    def test_additive_evidence_composition_rejects_incomplete_source(self) -> None:
        """A missing candidate profile cannot disappear inside a final union."""

        corpus = ObservationCorpus((cuda_observation(),))
        requests = build_profiler_request_manifest(corpus)
        incomplete = evidence_manifest(requests, ())

        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            compose_profiler_evidence(((corpus, requests, incomplete),))

    def test_additive_evidence_composition_rejects_conflicting_records(self) -> None:
        """Two source transactions cannot rewrite one physical measurement."""

        corpus = ObservationCorpus((cuda_observation(),))
        requests = build_profiler_request_manifest(corpus)
        first = evidence_manifest(
            requests,
            tuple(complete_evidence(request) for request in requests.requests),
        )
        second = evidence_manifest(
            requests,
            tuple(
                complete_evidence(
                    request,
                    dispatches=(cuda_dispatch(metric_scale=2.0),),
                )
                for request in requests.requests
            ),
        )

        with self.assertRaisesRegex(ValueError, "conflicting evidence"):
            compose_profiler_evidence((
                (corpus, requests, first),
                (corpus, requests, second),
            ))

    def test_additive_evidence_accepts_independent_exact_candidate_points(self) -> None:
        """Each candidate/geometry pair retains its own counter transaction."""

        first = dataclasses.replace(
            cuda_observation(
                candidate_id="cuda.nvnni.decode.fast_m1.wide.tn128.cpt1"
            ),
            projection_n_vector=(1024,),
            aggregate_n=1024,
            k=2048,
            aspect_ratio=0.5,
            aspect_bucket=classify_aspect(1024, 2048),
            work_items=1024 * 2048,
            shape_group_id="profile-additive-first-anchor",
            shape_name="profile-additive-first-anchor",
        )
        second = dataclasses.replace(
            cuda_observation(
                candidate_id="cuda.nvnni.decode.fast_m1.wide.tn128.cpt2"
            ),
            projection_n_vector=(2048,),
            aggregate_n=2048,
            k=4096,
            aspect_ratio=0.5,
            aspect_bucket=classify_aspect(2048, 4096),
            work_items=2048 * 4096,
            shape_group_id="profile-additive-other-anchor",
            shape_name="profile-additive-other-anchor",
        )

        def source_for(observation: NativeVNNIObservation):
            corpus = ObservationCorpus((observation,))
            requests = build_profiler_request_manifest(corpus)
            evidence = evidence_manifest(
                requests,
                tuple(
                    complete_evidence(request)
                    for request in requests.requests
                ),
            )
            catalog = build_profiler_feature_catalog(
                corpus,
                profiler_feature_rows(corpus, requests, evidence),
                request_manifest_digest=requests.digest(),
                evidence_manifest_digest=evidence.digest(),
            )
            return corpus, requests, evidence, catalog

        first_source = source_for(first)
        second_source = source_for(second)
        composed = compose_profiler_evidence((
            first_source[:3],
            second_source[:3],
        ))
        self.assertEqual(len(composed.requests.requests), 2)
        merged = merge_profiler_feature_catalogs(
            ObservationCorpus((first, second)),
            (first_source[3], second_source[3]),
        )
        self.assertEqual(len(merged.descriptors), 2)

    def test_manifest_accepts_multiple_exact_points_for_one_candidate(self) -> None:
        """Changing launch geometry creates another mandatory profile."""

        manifest = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        first = manifest.requests[0]
        second = dataclasses.replace(
            first,
            request_id=first.request_id + "-other-anchor",
            observation_digest="sha256:" + "f" * 64,
            projection_n_vector=(first.aggregate_n * 2,),
            aggregate_n=first.aggregate_n * 2,
        )

        exact = ProfilerRequestManifest(
            corpus_digest=manifest.corpus_digest,
            candidate_registry_digest=manifest.candidate_registry_digest,
            requests=(first, second),
        )
        self.assertEqual(len(exact.requests), 2)
        with self.assertRaisesRegex(ValueError, "exactly one anchor"):
            dataclasses.replace(
                exact,
                schema_version=LEGACY_PROFILER_REQUEST_SCHEMA_VERSION,
            )

    def test_manifest_profiles_every_measured_work_point(self) -> None:
        base = cuda_observation()
        observations = []
        for n, k in ((1024, 2048), (2048, 4096), (4096, 8192)):
            observations.append(dataclasses.replace(
                base,
                projection_n_vector=(n,),
                aggregate_n=n,
                k=k,
                aspect_ratio=float(n) / float(k),
                aspect_bucket=classify_aspect(n, k),
                work_items=n * k,
                shape_group_id=f"profile-unit:n{n}:k{k}",
                shape_name=f"profile-unit-{n}x{k}",
                n_tail_class=f"n_mod_256={n % 256}",
                k_tail_class=f"k_mod_256={k % 256}",
            ))

        manifest = build_profiler_request_manifest(
            ObservationCorpus(observations)
        )

        self.assertEqual(len(manifest.requests), 3)
        self.assertEqual(
            {(request.aggregate_n, request.k) for request in manifest.requests},
            {(1024, 2048), (2048, 4096), (4096, 8192)},
        )

    def test_manifest_profiles_each_competing_candidate_at_each_measured_point(self) -> None:
        """No candidate's counters are borrowed from another work point."""

        candidate_geometries = {
            "cuda.nvnni.decode.fast_m1.wide.tn128.cpt1": (
                (1024, 2048),
                (2048, 4096),
                (4096, 8192),
            ),
            "cuda.nvnni.decode.fast_m1.wide.tn128.cpt2": (
                (2048, 4096),
                (4096, 8192),
                (8192, 16384),
            ),
        }
        observations = []
        for candidate_id, geometries in candidate_geometries.items():
            base = cuda_observation(candidate_id=candidate_id)
            for n, k in geometries:
                observations.append(dataclasses.replace(
                    base,
                    projection_n_vector=(n,),
                    aggregate_n=n,
                    k=k,
                    aspect_ratio=float(n) / float(k),
                    aspect_bucket=classify_aspect(n, k),
                    work_items=n * k,
                    shape_group_id=f"profile-common:{candidate_id}:n{n}:k{k}",
                    shape_name=f"profile-common-{candidate_id}-{n}x{k}",
                    n_tail_class=f"n_mod_256={n % 256}",
                    k_tail_class=f"k_mod_256={k % 256}",
                ))

        manifest = build_profiler_request_manifest(ObservationCorpus(observations))

        self.assertEqual(len(manifest.requests), 6)
        self.assertEqual(
            {(request.aggregate_n, request.k) for request in manifest.requests},
            {
                (1024, 2048),
                (2048, 4096),
                (4096, 8192),
                (8192, 16384),
            },
        )

    def test_manifest_accepts_sparse_disjoint_candidate_points(self) -> None:
        """Sparse candidates remain independent exact profiling obligations."""

        first = cuda_observation(
            candidate_id="cuda.nvnni.decode.fast_m1.wide.tn128.cpt1"
        )
        second = dataclasses.replace(
            cuda_observation(
                candidate_id="cuda.nvnni.decode.fast_m1.wide.tn128.cpt2"
            ),
            projection_n_vector=(1024,),
            aggregate_n=1024,
            k=2048,
            aspect_ratio=0.5,
            aspect_bucket=classify_aspect(1024, 2048),
            work_items=1024 * 2048,
            shape_group_id="profile-disjoint",
            shape_name="profile-disjoint",
        )

        manifest = build_profiler_request_manifest(
            ObservationCorpus((first, second))
        )
        self.assertEqual(len(manifest.requests), 2)
        self.assertEqual(
            {(request.aggregate_n, request.k) for request in manifest.requests},
            {(1024, 2048), (2048, 4096)},
        )

    def test_manifest_treats_m_as_part_of_the_profiled_invocation(self) -> None:
        """The same template and N/K still require one profile per M."""

        first = cuda_observation()
        second = dataclasses.replace(
            first,
            m=2,
            shape_group_id="profile-m2",
            shape_name="profile-m2",
            work_items=first.aggregate_n * first.k,
        )
        manifest = build_profiler_request_manifest(
            ObservationCorpus((first, second))
        )
        self.assertEqual(len(manifest.requests), 2)
        self.assertEqual({request.m for request in manifest.requests}, {1, 2})

    def test_manifest_never_profiles_an_unreachable_representative_when_launchable(self) -> None:
        """A candidate supported at any geometry receives real counter evidence."""

        launchable = cuda_observation()
        unreachable = dataclasses.replace(
            cuda_observation(supported=False, forced_route_ok=False),
            projection_n_vector=(1024,),
            aggregate_n=1024,
            k=2048,
            aspect_ratio=0.5,
            aspect_bucket=classify_aspect(1024, 2048),
            work_items=1024 * 2048,
            shape_group_id="profile-unreachable-shape",
            shape_name="profile-unreachable",
        )

        manifest = build_profiler_request_manifest(ObservationCorpus((
            unreachable,
            launchable,
        )))

        self.assertEqual(len(manifest.requests), 1)
        self.assertTrue(manifest.requests[0].profile_required)
        self.assertEqual(manifest.requests[0].aggregate_n, launchable.aggregate_n)

    def test_feature_export_joins_every_exact_request_to_full_corpus(self) -> None:
        """Every measured physical point exports its attached profile row."""

        base = cuda_observation()
        observations = ObservationCorpus(tuple(
            dataclasses.replace(
                base,
                projection_n_vector=(n,),
                aggregate_n=n,
                k=k,
                aspect_ratio=float(n) / float(k),
                aspect_bucket=classify_aspect(n, k),
                work_items=n * k,
                shape_group_id=f"profile-subset:n{n}:k{k}",
                shape_name=f"profile-subset-{n}x{k}",
                n_tail_class=f"n_mod_256={n % 256}",
                k_tail_class=f"k_mod_256={k % 256}",
            )
            for n, k in ((1024, 2048), (2048, 4096), (4096, 8192))
        ))
        requests = build_profiler_request_manifest(observations)
        evidence = evidence_manifest(
            requests,
            tuple(complete_evidence(request) for request in requests.requests),
        )

        rows = profiler_feature_rows(observations, requests, evidence)

        self.assertEqual(len(requests.requests), 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row.observation.aggregate_n for row in rows},
            {1024, 2048, 4096},
        )

    def test_compact_witnesses_authenticates_rows_from_retained_superset(self) -> None:
        """Old profiler evidence remains reusable after its corpus is expanded."""

        base = cuda_observation()
        observations = ObservationCorpus(tuple(
            dataclasses.replace(
                base,
                projection_n_vector=(n,),
                aggregate_n=n,
                k=k,
                aspect_ratio=float(n) / float(k),
                aspect_bucket=classify_aspect(n, k),
                work_items=n * k,
                shape_group_id=f"profile-witness:n{n}:k{k}",
                shape_name=f"profile-witness-{n}x{k}",
                n_tail_class=f"n_mod_256={n % 256}",
                k_tail_class=f"k_mod_256={k % 256}",
            )
            for n, k in ((1024, 2048), (2048, 4096), (4096, 8192))
        ))
        requests = build_profiler_request_manifest(observations)
        evidence = evidence_manifest(
            requests,
            tuple(complete_evidence(request) for request in requests.requests),
        )

        compact = compact_profiler_observation_witnesses(
            observations,
            requests,
            evidence,
        )

        self.assertEqual(len(compact), 3)
        self.assertEqual(compact.digest(), observations.digest())
        self.assertEqual(
            len(profiler_feature_rows(compact, requests, evidence)),
            3,
        )

        modified = ObservationCorpus(tuple(
            dataclasses.replace(row, median_us=11.5)
            if row.digest() == requests.requests[0].observation_digest
            else row
            for row in observations
        ))
        with self.assertRaisesRegex(ValueError, "omits requested timing"):
            compact_profiler_observation_witnesses(
                modified,
                requests,
                evidence,
            )

    def test_feature_table_recovers_deduplicated_timing_witnesses(self) -> None:
        """A retained feature CSV can restore lost compact source observations."""

        observations = ObservationCorpus((cuda_observation(),))
        requests = build_profiler_request_manifest(observations)
        evidence = evidence_manifest(requests, (
            complete_evidence(
                requests.requests[0],
                dispatches=(
                    cuda_dispatch(0, "nativeVnniKPartProducer"),
                    cuda_dispatch(1, "nativeVnniOrderedReducer"),
                ),
            ),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            write_profiler_feature_csv(
                path,
                observations,
                requests,
                evidence,
            )
            recovered = read_profiler_feature_observation_witnesses((path,))

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["profiler.observation_digest"] = "sha256:" + "0" * 64
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_profiler_feature_observation_witnesses((path,))

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered.digest(), observations.digest())

    def test_profiler_model_requires_every_launchable_candidate(self) -> None:
        """A partially profiled candidate matrix must never train a policy."""

        corpus, _costs, requests, evidence, rows, _catalog = (
            profiler_model_fixture()
        )
        omitted_candidate = requests.requests[-1].candidate_id
        incomplete_rows = tuple(
            row
            for row in rows
            if row.observation.candidate_id != omitted_candidate
        )

        with self.assertRaisesRegex(ValueError, "omits physical candidate"):
            build_profiler_feature_catalog(
                corpus,
                incomplete_rows,
                request_manifest_digest=requests.digest(),
                evidence_manifest_digest=evidence.digest(),
            )

    def test_profiler_model_ignores_metrics_without_auxiliary_signal(self) -> None:
        """Config-only separability must not masquerade as counter influence."""

        corpus, costs, *_unused, first_catalog = profiler_model_fixture(
            second_candidate_metric_scale=3.0
        )
        _corpus, _costs, *_unused, flat_catalog = profiler_model_fixture(
            second_candidate_metric_scale=1.0
        )

        first = apply_profiler_regret_predictions(costs, corpus, first_catalog)
        self.assertNotEqual(first_catalog.digest, flat_catalog.digest)
        self.assertTrue(any(
            row.profiler_predicted_regret is not None for row in first
        ))
        flat = apply_profiler_regret_predictions(costs, corpus, flat_catalog)
        self.assertEqual(
            [row.profiler_predicted_regret for row in flat],
            [row.profiler_predicted_regret for row in costs],
        )
        self.assertTrue(all(
            (row.selection_regret < P95_REGRET_BUDGET)
            == (row.max_surface_regret < P95_REGRET_BUDGET)
            for row in first
        ))

    def test_auxiliary_centering_rejects_partial_candidate_metrics(self) -> None:
        """An unavailable counter must not look like a zero-cost candidate."""

        observations = [
            dataclasses.replace(
                cuda_observation(),
                shape_group_id=f"shape-{shape}",
                shape_name=f"shape-{shape}",
                candidate_id=candidate,
                effective_candidate_id=candidate,
            )
            for shape in range(2)
            for candidate in ("candidate-a", "candidate-b")
        ]
        point_members = {
            (runtime_key(observations[0]), "shape-0"): [0, 1],
            (runtime_key(observations[2]), "shape-1"): [2, 3],
        }
        centered = _center_complete_candidate_relative_auxiliary(
            [
                {"stable": 2.0, "partial": 7.0},
                {"stable": 4.0},
                {"stable": 10.0},
                {"stable": 14.0},
            ],
            point_members,
        )

        self.assertEqual(
            [record["stable"] for record in centered],
            [-1.0, 1.0, -2.0, 2.0],
        )
        self.assertTrue(all("partial" not in record for record in centered))

    def test_profiler_model_digest_ignores_non_model_provenance(self) -> None:
        """Additive timing identity must not invalidate unchanged fit inputs."""

        _corpus, _costs, *_unused, catalog = profiler_model_fixture()
        rebound = dataclasses.replace(
            catalog,
            corpus_digest="sha256:" + "1" * 64,
            request_manifest_digest="sha256:" + "2" * 64,
            evidence_manifest_digest="sha256:" + "3" * 64,
        )

        self.assertNotEqual(catalog.digest, rebound.digest)
        self.assertEqual(catalog.model_digest, rebound.model_digest)

    def test_profiler_catalog_whole_catalog_digests_are_memoized(self) -> None:
        """Per-domain cache lookup must not reserialize immutable evidence."""

        _corpus, _costs, *_unused, catalog = profiler_model_fixture()
        with mock.patch.object(json, "dumps", wraps=json.dumps) as dumps:
            first_provenance = catalog.digest
            second_provenance = catalog.digest
            first_model = catalog.model_digest
            second_model = catalog.model_digest

        self.assertEqual(first_provenance, second_provenance)
        self.assertEqual(first_model, second_model)
        self.assertEqual(dumps.call_count, 2)

    def test_profiler_model_digest_changes_with_normalized_features(self) -> None:
        """A descriptor change must still invalidate every dependent model."""

        _corpus, _costs, *_unused, catalog = profiler_model_fixture()
        key, descriptor = next(iter(catalog.descriptors.items()))
        changed_features = dict(descriptor.features)
        changed_features["metric.unit.incremental-cache-probe"] = 1.0
        changed = dataclasses.replace(
            catalog,
            descriptors={
                **catalog.descriptors,
                key: dataclasses.replace(
                    descriptor,
                    features=changed_features,
                ),
            },
        )

        self.assertNotEqual(catalog.model_digest, changed.model_digest)

    def test_profiler_pool_digest_ignores_unreachable_descriptors(self) -> None:
        """A descriptor outside one pool must not evict that pool's CV cache."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        selected = corpus.observations[0]
        other_key = next(
            key
            for key in catalog.descriptors
            if key.effective_candidate_id != selected.effective_candidate_id
        )
        other_descriptor = catalog.descriptors[other_key]
        changed_features = dict(other_descriptor.features)
        changed_features["metric.unit.other-pool-probe"] = 2.0
        changed = dataclasses.replace(
            catalog,
            descriptors={
                **catalog.descriptors,
                other_key: dataclasses.replace(
                    other_descriptor,
                    features=changed_features,
                ),
            },
        )

        self.assertNotEqual(catalog.model_digest, changed.model_digest)
        self.assertEqual(
            catalog.model_digest_for((selected,)),
            changed.model_digest_for((selected,)),
        )

    def test_model_catalog_excludes_raw_representative_counter_magnitudes(self) -> None:
        """Admit normalized economics while retaining raw values only in evidence."""

        _corpus, _costs, *_unused, catalog = profiler_model_fixture()
        descriptor = next(iter(catalog.descriptors.values()))
        names = set(descriptor.features)

        self.assertEqual(
            descriptor.features["profile.execution_regime"],
            "decode",
        )
        self.assertIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            names,
        )
        self.assertIn(
            "metric.gpu.registers_per_thread.maximum_log1p",
            names,
        )
        self.assertFalse(any(
            name.startswith("profile.representative_") for name in names
        ))
        self.assertFalse(any(
            name.startswith("profile.grid_threads.") for name in names
        ))
        self.assertNotIn("metric.gpu.duration_ns.mean", names)
        self.assertNotIn("metric.gpu.duration_ns.sum", names)

    def test_decode_and_prefill_normalize_profiler_time_by_distinct_work(self) -> None:
        """Use bytes/second for GEMV and operations/second for GEMM evidence."""

        decode = cuda_observation()
        metric_values = {
            metric.metric_id: [float(metric.value)]
            for metric in complete_metrics(Backend.CUDA)
            if metric.value is not None
        }
        prefill = dataclasses.replace(
            decode,
            operation_kind="NativeVNNIPrefillProjection",
            m=64,
        )

        decode_features = _normalized_metric_features(decode, metric_values)
        prefill_features = _normalized_metric_features(prefill, metric_values)

        self.assertIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            decode_features,
        )
        self.assertIn(
            "metric.gpu.decode.effective_gops_log1p",
            decode_features,
        )
        self.assertNotIn(
            "metric.gpu.prefill.effective_gops_log1p",
            decode_features,
        )
        self.assertIn(
            "metric.gpu.prefill.effective_gops_log1p",
            prefill_features,
        )
        self.assertNotIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            prefill_features,
        )

    def test_cpu_and_rocm_features_follow_the_same_regime_contract(self) -> None:
        """Keep backend-specific counters symmetric around decode/prefill work."""

        decode = cuda_observation()
        prefill = dataclasses.replace(
            decode,
            operation_kind="NativeVNNIPrefillProjection",
            m=64,
        )

        def values_for(backend: Backend) -> dict[str, list[float]]:
            values = {
                metric.metric_id: [float(metric.value)]
                for metric in complete_metrics(backend)
                if metric.value is not None
            }
            if backend == Backend.CPU:
                values["cpu.wall_clock_ns"] = [1_000_000.0]
                values["cpu.task_clock_ns"] = [20_000_000.0]
            return values

        cpu_decode_observation = dataclasses.replace(
            decode,
            architecture_class="unit-cpu|threads=28",
        )
        cpu_prefill_observation = dataclasses.replace(
            prefill,
            architecture_class="unit-cpu|threads=28",
        )
        cpu_decode = _normalized_metric_features(
            cpu_decode_observation, values_for(Backend.CPU)
        )
        cpu_prefill = _normalized_metric_features(
            cpu_prefill_observation, values_for(Backend.CPU)
        )
        rocm_decode = _normalized_metric_features(
            decode, values_for(Backend.ROCM)
        )
        rocm_prefill = _normalized_metric_features(
            prefill, values_for(Backend.ROCM)
        )

        self.assertIn(
            "metric.cpu.decode.cycles_per_expected_mib_log1p",
            cpu_decode,
        )
        self.assertIn(
            "metric.cpu.decode.cycles_per_million_macs_log1p",
            cpu_decode,
        )
        self.assertIn(
            "metric.cpu.prefill.cycles_per_million_macs_log1p",
            cpu_prefill,
        )
        self.assertIn(
            "metric.gpu.decode.effective_gbytes_per_second_log1p",
            rocm_decode,
        )
        self.assertIn("metric.gpu.fetch_to_expected_byte_ratio", rocm_decode)
        self.assertIn("metric.gpu.vgpr_count.maximum_log1p", rocm_decode)
        self.assertIn(
            "metric.gpu.prefill.effective_gops_log1p",
            rocm_prefill,
        )

    def test_short_cpu_profile_masks_duration_sensitive_counter_features(self) -> None:
        """Control overhead cannot become a learned short-kernel property."""

        observation = dataclasses.replace(
            cuda_observation(),
            backend=Backend.CPU,
            architecture_class="unit-cpu|threads=28",
            operation_kind="NativeVNNIPrefillProjection",
            m=64,
        )
        values = {
            metric.metric_id: [float(metric.value)]
            for metric in complete_metrics(Backend.CPU)
            if metric.value is not None
        }
        values["cpu.wall_clock_ns"] = [100_000.0]
        values["cpu.task_clock_ns"] = [3_400_000.0]
        features = _normalized_metric_features(observation, values)

        self.assertEqual(
            features["profile.cpu.duration_features_reliable"],
            0.0,
        )
        self.assertIn(
            "metric.cpu.prefill.instructions_per_million_macs_log1p",
            features,
        )
        self.assertIn(
            "metric.cpu.prefill.l1d_loads_per_million_macs_log1p",
            features,
        )
        self.assertIn(
            "metric.cpu.prefill.l1d_load_misses_per_million_macs_log1p",
            features,
        )
        self.assertIn(
            "metric.cpu.prefill.llc_load_misses_per_million_macs_log1p",
            features,
        )
        self.assertIn("metric.cpu.l1d_load_miss_fraction", features)
        self.assertNotIn("metric.cpu.ipc", features)
        self.assertNotIn(
            "metric.cpu.prefill.cycles_per_million_macs_log1p",
            features,
        )

        auxiliary = _profiler_auxiliary_target_record(features)
        self.assertNotIn("profile.cpu.duration_features_reliable", auxiliary)
        self.assertFalse(any("wall_clock_ns_" in name for name in auxiliary))
        self.assertFalse(any("task_clock_ns_" in name for name in auxiliary))

    def test_profiler_prediction_preserves_measured_hard_budget_topology(self) -> None:
        """Bounded shrinkage must never create a synthetic pass or failure."""

        corpus, costs, *_unused = profiler_model_fixture()
        passing = next(
            cost
            for cost in costs
            if cost.max_surface_regret < P95_REGRET_BUDGET
        )
        failing = next(
            cost
            for cost in costs
            if cost.max_surface_regret >= P95_REGRET_BUDGET
        )

        passing = dataclasses.replace(
            passing,
            profiler_predicted_regret=0.75,
        )
        failing = dataclasses.replace(
            failing,
            profiler_predicted_regret=0.75,
        )

        self.assertLess(passing.selection_regret, P95_REGRET_BUDGET)
        self.assertGreater(failing.selection_regret, P95_REGRET_BUDGET)
        self.assertEqual(
            failing.selection_regret,
            failing.max_surface_regret + P95_REGRET_BUDGET,
        )

        optimistic_failure = dataclasses.replace(
            failing,
            profiler_predicted_regret=0.0,
        )
        self.assertGreater(
            optimistic_failure.selection_regret,
            P95_REGRET_BUDGET,
        )

        exact_boundary = dataclasses.replace(
            passing,
            max_surface_regret=P95_REGRET_BUDGET,
            profiler_predicted_regret=0.0,
        )
        self.assertGreater(
            exact_boundary.selection_regret,
            P95_REGRET_BUDGET,
        )

    def test_profiler_prediction_can_rank_measured_passing_candidates(self) -> None:
        """Profiler economics may resolve near-ties without crossing the gate."""

        corpus, costs, *_unused = profiler_model_fixture()
        passing = next(
            cost
            for cost in costs
            if cost.max_surface_regret < P95_REGRET_BUDGET
        )
        optimistic = dataclasses.replace(
            passing,
            profiler_predicted_regret=0.0,
        )
        pessimistic = dataclasses.replace(
            passing,
            profiler_predicted_regret=0.02,
        )

        self.assertLess(optimistic.selection_regret, pessimistic.selection_regret)
        self.assertLessEqual(
            pessimistic.selection_regret,
            P95_REGRET_BUDGET,
        )

    def test_profiler_blend_strength_is_explicit_and_budget_bounded(self) -> None:
        """CV may trust a useful prior more strongly without forging evidence."""

        _corpus, costs, *_unused = profiler_model_fixture()
        cost = dataclasses.replace(
            next(
                item
                for item in costs
                if item.max_surface_regret >= P95_REGRET_BUDGET
            ),
            max_surface_regret=0.10,
        )
        quarter = dataclasses.replace(
            cost,
            profiler_predicted_regret=0.025,
            profiler_blend_weight=0.25,
        )
        half = dataclasses.replace(quarter, profiler_blend_weight=0.5)
        full = dataclasses.replace(quarter, profiler_blend_weight=1.0)

        self.assertGreater(quarter.selection_regret, half.selection_regret)
        self.assertGreater(half.selection_regret, full.selection_regret)
        self.assertGreater(full.selection_regret, P95_REGRET_BUDGET)
        self.assertGreaterEqual(
            full.selection_regret,
            cost.max_surface_regret - P95_REGRET_BUDGET,
        )
        with self.assertRaisesRegex(ValueError, "blend weight"):
            _ = dataclasses.replace(
                quarter,
                profiler_blend_weight=1.01,
            ).selection_regret

    def test_emitted_rule_reports_measured_not_profiler_penalty_regret(self) -> None:
        """Profiler regularization must not inflate the published timing claim."""

        corpus, costs, *_unused = profiler_model_fixture()
        penalized = [
            dataclasses.replace(cost, profiler_predicted_regret=0.75)
            for cost in costs
        ]
        rules = fit_domain_rules(
            corpus.generic_domains()[0],
            penalized,
            corpus,
            max_leaves=1,
            min_shape_groups_per_leaf=1,
        )

        self.assertEqual(len(rules), 1)
        selected = rules[0].candidate_id
        self.assertEqual(
            rules[0].development_max_regret,
            max(
                cost.max_surface_regret
                for cost in costs
                if cost.candidate_id == selected
            ),
        )
        self.assertLess(rules[0].development_max_regret, 0.75)

    def test_profiler_fold_fit_never_receives_held_out_timing_rows(self) -> None:
        """Grouped CV profiler labels are restricted to training shape groups."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        observed_calls = []
        observed_indices = []
        observed_model_pools = []
        original_codebook = costs[0].runtime_key.runtime_codebook_id
        foreign_codebook = original_codebook + 100
        foreign_costs = [
            dataclasses.replace(
                cost,
                runtime_key=dataclasses.replace(
                    cost.runtime_key,
                    prepared_family_id="NativeVNNI_cuda_CB1",
                    packing_abi="native-vnni-cuda-cb1-v1",
                    runtime_codebook_id=foreign_codebook,
                ),
            )
            for cost in costs
        ]
        other_m = domain.m * 2
        cross_m_costs = [
            dataclasses.replace(
                cost,
                runtime_key=dataclasses.replace(cost.runtime_key, m=other_m),
            )
            for cost in (*costs, *foreign_costs)
        ]

        def capture(
            training_costs,
            training_corpus,
            feature_catalog,
            *,
            observation_index=None,
            model_training_costs=None,
            model_record_index=None,
            excluded_profiler_geometries=frozenset(),
        ):
            self.assertIs(feature_catalog, catalog)
            self.assertIsNotNone(observation_index)
            self.assertIsNotNone(model_record_index)
            prediction_groups = {
                row.shape_group_id for row in training_costs
            }
            corpus_groups = {row.shape_group_id for row in training_corpus}
            pool_groups = {
                row.shape_group_id for row in model_training_costs
            }
            self.assertEqual(prediction_groups, corpus_groups)
            self.assertLess(pool_groups, corpus_groups)
            prediction_geometries = {
                (row.runtime_key.aggregate_n, row.runtime_key.k)
                for row in training_costs
            }
            model_geometries = {
                (row.runtime_key.aggregate_n, row.runtime_key.k)
                for row in model_training_costs
            }
            self.assertLess(model_geometries, prediction_geometries)
            self.assertEqual(
                excluded_profiler_geometries,
                frozenset(prediction_geometries - model_geometries),
            )
            self.assertEqual(
                len(model_training_costs),
                4 * sum(
                    row.shape_group_id in pool_groups for row in costs
                ),
            )
            self.assertEqual(
                {row.runtime_key.runtime_codebook_id for row in model_training_costs},
                {original_codebook, foreign_codebook},
            )
            self.assertEqual(
                {row.runtime_key.m for row in model_training_costs},
                {domain.m, other_m},
            )
            observed_calls.append(frozenset(pool_groups))
            observed_indices.append(observation_index)
            observed_model_pools.append(frozenset(pool_groups))
            return list(training_costs)

        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "apply_profiler_regret_predictions",
            side_effect=capture,
        ), mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_model_record_index",
            return_value={"materialized": "once"},
        ) as build_records:
            prediction_cache = {}
            training_pool = tuple((*costs, *foreign_costs, *cross_m_costs))
            pool_key = _profiler_transfer_key(domain)
            request_keys = _domain_profiler_prediction_request_keys(
                domain,
                costs,
                seed="profiler-fold-unit",
            )
            workers = _populate_profiler_prediction_cache(
                prediction_cache,
                request_keys,
                {pool_key: training_pool},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=1,
            )
            repeated_workers = _populate_profiler_prediction_cache(
                prediction_cache,
                request_keys,
                {pool_key: training_pool},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=1,
            )
            tasks, fold_count = _domain_fold_tasks(
                domain,
                costs,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                seed="profiler-fold-unit",
                profiler_training_pool_key=pool_key,
                profiler_prediction_cache=prediction_cache,
            )
            repeated_tasks, repeated_fold_count = _domain_fold_tasks(
                domain,
                costs,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                seed="profiler-fold-unit",
                profiler_training_pool_key=pool_key,
                profiler_prediction_cache=prediction_cache,
            )

        all_groups = {row.shape_group_id for row in corpus}
        self.assertEqual(workers, 1)
        self.assertEqual(repeated_workers, 0)
        build_records.assert_called_once()
        self.assertEqual(len(observed_calls), fold_count)
        self.assertEqual(repeated_fold_count, fold_count)
        self.assertEqual(repeated_tasks, tasks)
        self.assertEqual(
            {task[5] for task in tasks},
            {
                ProfilerInfluence.MEASURED_ONLY,
                ProfilerInfluence.BOUNDED_PRIOR,
                ProfilerInfluence.BOUNDED_PRIOR_HALF,
                ProfilerInfluence.BOUNDED_PRIOR_FULL,
            },
        )
        self.assertTrue(all(
            index is observed_indices[0] for index in observed_indices
        ))
        self.assertEqual(observed_calls, observed_model_pools)
        self.assertTrue(tasks)
        self.assertTrue(all(groups < all_groups for groups in observed_calls))

    def test_profiler_prediction_surfaces_are_built_concurrently(self) -> None:
        """Independent held-out forests use the configured host worker pool."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        base_pool_key = _profiler_transfer_key(domain)
        pool_keys = tuple(
            (*base_pool_key, f"unit-pool-{index}")
            for index in range(4)
        )
        request_keys = tuple(
            (pool_key, ((1000 + index, 2000 + index),))
            for index, pool_key in enumerate(pool_keys)
        )

        def capture(*args, **kwargs):
            time.sleep(0.05)
            return [
                dataclasses.replace(
                    cost,
                    profiler_predicted_regret=float(os.getpid()),
                )
                for cost in args[0]
            ]

        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "apply_profiler_regret_predictions",
            side_effect=capture,
        ), mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_model_record_index",
            return_value={"materialized": "once"},
        ):
            prediction_cache = {}
            workers = _populate_profiler_prediction_cache(
                prediction_cache,
                request_keys,
                {pool_key: tuple(costs) for pool_key in pool_keys},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=4,
            )

        self.assertEqual(workers, 4)
        self.assertEqual(set(prediction_cache), set(request_keys))
        worker_pids = {
            prediction
            for predictions in prediction_cache.values()
            for prediction in predictions.values()
            if prediction is not None
        }
        self.assertGreater(len(worker_pids), 1)

    def test_profiler_fold_surfaces_include_held_teacher_points(self) -> None:
        """A fold predicts held choices without admitting their timing labels."""

        corpus, costs, *_unused = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        expected_points = frozenset(
            (
                cost.runtime_key,
                cost.shape_group_id,
                cost.candidate_id,
            )
            for cost in costs
        )

        requests = _domain_profiler_prediction_requests(
            domain,
            costs,
            seed="profiler-teacher-inventory-unit",
        )

        self.assertTrue(requests)
        self.assertTrue(all(
            points == expected_points for points in requests.values()
        ))

    def test_profiler_teacher_choices_are_scored_by_held_timing(self) -> None:
        """The surrogate chooses; held canonical evidence alone judges it."""

        corpus, costs, *_unused = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        pool_key = _profiler_transfer_key(domain)
        first_candidate = costs[0].candidate_id
        prediction_cache = {}
        for request_key, points in _domain_profiler_prediction_requests(
            domain,
            costs,
            seed="profiler-teacher-choice-unit",
        ).items():
            prediction_cache[request_key] = {
                point: (0.0 if point[2] == first_candidate else 1.0)
                for point in points
            }

        results = _domain_profiler_teacher_fold_results(
            domain,
            costs,
            max_leaves=3,
            seed="profiler-teacher-choice-unit",
            profiler_training_pool_key=pool_key,
            profiler_prediction_cache=prediction_cache,
        )

        self.assertTrue(results)
        self.assertTrue(all(
            result.profiler_influence
            == ProfilerInfluence.CROSS_FITTED_TEACHER
            for result in results
        ))
        for result in results:
            self.assertEqual(len(result.complexities), 3)
            self.assertTrue(all(
                complexity.decisions == result.complexities[0].decisions
                for complexity in result.complexities
            ))
            for selected, exact in result.complexities[0].decisions:
                self.assertEqual(selected.candidate_id, first_candidate)
                self.assertLessEqual(
                    exact.max_surface_regret,
                    selected.max_surface_regret,
                )

    def test_generic_fit_consumes_cross_fitted_teacher_results(self) -> None:
        """Profiler fitting cannot silently omit the direct teacher candidate."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        ranked_influences = []

        def rank_with_witness(*args, **kwargs):
            fold_results = tuple(args[3])
            ranked_influences.append({
                result.profiler_influence for result in fold_results
            })
            return _rank_domain_cross_validations(
                args[0],
                args[1],
                args[2],
                fold_results,
                **kwargs,
            )

        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "_domain_profiler_teacher_fold_results",
            wraps=_domain_profiler_teacher_fold_results,
        ) as teacher, mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "_rank_domain_cross_validations",
            side_effect=rank_with_witness,
        ):
            policy = fit_generic_policy(
                corpus,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                policy_accelerators=(),
                profiler_feature_catalog=catalog,
                fit_final_rules=False,
            )

        self.assertTrue(policy.cross_validation)
        teacher.assert_called_once()
        self.assertTrue(ranked_influences)
        self.assertTrue(all(
            ProfilerInfluence.CROSS_FITTED_TEACHER in influences
            for influences in ranked_influences
        ))

    def test_one_cross_m_pool_parallelizes_independent_surfaces(self) -> None:
        """One large transfer pool must not collapse fitting to one process."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        pool_key = _profiler_transfer_key(domain)
        request_keys = _domain_profiler_prediction_request_keys(
            domain,
            costs,
            seed="single-pool-parallel-unit",
        )
        self.assertGreaterEqual(len(request_keys), 4)

        def capture(*args, **kwargs):
            time.sleep(0.05)
            return [
                dataclasses.replace(
                    cost,
                    profiler_predicted_regret=float(os.getpid()),
                )
                for cost in args[0]
            ]

        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "apply_profiler_regret_predictions",
            side_effect=capture,
        ), mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_model_record_index",
            return_value={"materialized": "once"},
        ) as build_records:
            prediction_cache = {}
            workers = _populate_profiler_prediction_cache(
                prediction_cache,
                request_keys,
                {pool_key: tuple(costs)},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=4,
            )

        self.assertEqual(workers, 4)
        build_records.assert_called_once()
        worker_pids = {
            prediction
            for predictions in prediction_cache.values()
            for prediction in predictions.values()
            if prediction is not None
        }
        self.assertGreater(len(worker_pids), 1)

    def test_profiler_transfer_pool_is_generic_across_prefill_m(self) -> None:
        """M is a learned numeric feature, not a transfer-pool boundary."""

        corpus, _costs, *_unused = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        other_m_domain = dataclasses.replace(domain, m=domain.m * 8)

        self.assertEqual(
            _profiler_transfer_key(other_m_domain),
            _profiler_transfer_key(domain),
        )
        self.assertNotEqual(
            _profiler_transfer_key(dataclasses.replace(
                other_m_domain,
                operation_kind=f"{domain.operation_kind}.other-regime",
            )),
            _profiler_transfer_key(domain),
        )

    def test_profiler_prediction_cache_is_worker_count_invariant(self) -> None:
        """Fork scheduling cannot change deterministic ExtraTrees predictions."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        base_pool_key = _profiler_transfer_key(domain)
        pool_keys = tuple(
            (*base_pool_key, f"unit-pool-{index}")
            for index in range(4)
        )
        request_keys = tuple((pool_key, ()) for pool_key in pool_keys)
        pools = {pool_key: tuple(costs) for pool_key in pool_keys}
        observation_index = build_profiler_observation_index(corpus)
        serial_cache = {}
        parallel_cache = {}

        serial_workers = _populate_profiler_prediction_cache(
            serial_cache,
            request_keys,
            pools,
            corpus,
            catalog,
            observation_index,
            requested_workers=1,
        )
        parallel_workers = _populate_profiler_prediction_cache(
            parallel_cache,
            request_keys,
            pools,
            corpus,
            catalog,
            observation_index,
            requested_workers=4,
        )

        self.assertEqual(serial_workers, 1)
        self.assertEqual(parallel_workers, 4)
        self.assertEqual(parallel_cache, serial_cache)

    def test_profiler_prediction_surfaces_persist_compact_point_maps(self) -> None:
        """A paid forest reloads without fitting or storing unrelated rows."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        pool_key = _profiler_transfer_key(domain)
        request_key = _domain_profiler_prediction_request_keys(
            domain,
            costs,
            seed="persistent-profiler-surface-unit",
        )[0]
        requested_points = frozenset({(
            costs[0].runtime_key,
            costs[0].shape_group_id,
            costs[0].candidate_id,
        )})
        pool_digest = "sha256:" + "1" * 64
        model_digest = "sha256:" + "2" * 64

        def capture(prediction_costs, *_args, **_kwargs):
            self.assertEqual(len(prediction_costs), 1)
            return [
                dataclasses.replace(
                    prediction_costs[0],
                    profiler_predicted_regret=0.125,
                )
            ]

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "apply_profiler_regret_predictions",
            side_effect=capture,
        ) as apply_predictions, mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_model_record_index",
            return_value={"materialized": "once"},
        ) as build_records:
            fit_cache = PolicyFitCache(Path(directory))
            first_cache = {}
            first_workers = _populate_profiler_prediction_cache(
                first_cache,
                (request_key,),
                {pool_key: tuple(costs)},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=1,
                prediction_points_by_request={
                    request_key: requested_points,
                },
                fit_cache=fit_cache,
                training_pool_digests={pool_key: pool_digest},
                profiler_model_digests={pool_key: model_digest},
            )
            replay_cache = {}
            replay_workers = _populate_profiler_prediction_cache(
                replay_cache,
                (request_key,),
                {pool_key: tuple(costs)},
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
                requested_workers=1,
                prediction_points_by_request={
                    request_key: requested_points,
                },
                fit_cache=fit_cache,
                training_pool_digests={pool_key: pool_digest},
                profiler_model_digests={pool_key: model_digest},
            )

            surface_files = tuple(
                (Path(directory) / "profiler-prediction-surface").glob(
                    "*.json"
                )
            )

        self.assertEqual(first_workers, 1)
        self.assertEqual(replay_workers, 0)
        self.assertEqual(replay_cache, first_cache)
        self.assertEqual(set(first_cache[request_key]), set(requested_points))
        self.assertEqual(len(surface_files), 1)
        apply_predictions.assert_called_once()
        build_records.assert_called_once()

    def test_generic_fit_reuses_surfaces_when_only_cv_is_missing(self) -> None:
        """Incremental CV replay consumes its durable profiler surfaces."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        arguments = {
            "max_leaves": 2,
            "min_shape_groups_per_leaf": 1,
            "policy_accelerators": (),
            "profiler_feature_catalog": catalog,
            "fit_final_rules": False,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1",
                "LLAMINAR_NATIVE_VNNI_PROFILER_MODEL_WORKERS": "1",
            },
        ):
            fit_cache = PolicyFitCache(Path(directory))
            baseline = fit_generic_policy(
                corpus,
                fit_cache=fit_cache,
                **arguments,
            )
            surface_files = tuple(
                (Path(directory) / "profiler-prediction-surface").glob(
                    "*.json"
                )
            )
            validation_files = tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )
            self.assertTrue(surface_files)
            self.assertTrue(validation_files)
            for path in validation_files:
                path.unlink()

            with mock.patch(
                "native_vnni_dispatch.segmented_policy."
                "_fit_profiler_prediction_pool",
                side_effect=AssertionError("profiler surface was rebuilt"),
            ):
                replay = fit_generic_policy(
                    corpus,
                    fit_cache=fit_cache,
                    **arguments,
                )

        self.assertEqual(replay, baseline)

    def test_profiler_cv_task_assembly_rejects_missing_surface(self) -> None:
        """A serial hidden model fit cannot return to CV task construction."""

        corpus, costs, *_unused = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        with self.assertRaisesRegex(RuntimeError, "was not completed"):
            _domain_fold_tasks(
                domain,
                costs,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                seed="profiler-fold-unit",
                profiler_training_pool_key=_profiler_transfer_key(domain),
                profiler_prediction_cache={},
            )

    def test_profiler_pool_worker_rejects_foreign_request_key(self) -> None:
        """A scheduler mismatch cannot train one pool under another identity."""

        corpus, costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        pool_key = _profiler_transfer_key(domain)
        foreign_key = (*pool_key, "foreign-unit-pool")
        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_model_record_index",
            return_value={},
        ), self.assertRaisesRegex(ValueError, "different transfer pool"):
            _fit_profiler_prediction_pool(
                pool_key,
                ((foreign_key, ()),),
                tuple(costs),
                corpus,
                catalog,
                build_profiler_observation_index(corpus),
            )

    def test_profiler_informed_catalog_drives_complete_generic_fit(self) -> None:
        """The authenticated sidecar reaches both CV and final tree fitting."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        with mock.patch(
            "native_vnni_dispatch.segmented_policy."
            "build_profiler_observation_index",
            wraps=build_profiler_observation_index,
        ) as build_index:
            policy = fit_generic_policy(
                corpus,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                policy_accelerators=(),
                profiler_feature_catalog=catalog,
            )

        build_index.assert_called_once_with(corpus)
        self.assertTrue(policy.rules)
        self.assertFalse(policy.unpromoted_domains)
        self.assertEqual(len(policy.cross_validation), 1)
        self.assertEqual(
            policy.cross_validation[0].covered_point_count,
            policy.cross_validation[0].required_point_count,
        )

    def test_profiler_cache_identity_ignores_unlaunchable_diagnostics(self) -> None:
        """Unsupported route rows do not require physical counter evidence."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        exemplar = corpus.observations[0]
        diagnostic_id = "cuda.nvnni.unit.unsupported-diagnostic"
        diagnostic = dataclasses.replace(
            exemplar,
            candidate_id=diagnostic_id,
            effective_candidate_id=diagnostic_id,
            observed_candidate_id=diagnostic_id,
            candidate_family="unit-unsupported",
            supported=False,
            forced_route_ok=False,
        )
        expanded = ObservationCorpus((*tuple(corpus), diagnostic))
        rebound = catalog.rebind(expanded)

        expected = fit_generic_policy(
            corpus,
            max_leaves=2,
            min_shape_groups_per_leaf=1,
            policy_accelerators=(),
            profiler_feature_catalog=catalog,
        )
        actual = fit_generic_policy(
            expanded,
            max_leaves=2,
            min_shape_groups_per_leaf=1,
            policy_accelerators=(),
            profiler_feature_catalog=rebound,
        )

        self.assertEqual(actual, expected)

    def test_profiler_pool_change_invalidates_every_dependent_cv_domain(self) -> None:
        """Adding one format retrains every M in its shared prefill pool."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        original_m = corpus.observations[0].m
        other_m = 31 if original_m != 31 else 15
        other_m_rows = tuple(
            dataclasses.replace(
                row,
                m=other_m,
                shape_group_id=f"other-m-{row.shape_group_id}",
                shape_name=f"other-m-{row.shape_name}",
            )
            for row in corpus
        )
        baseline = ObservationCorpus((*tuple(corpus), *other_m_rows))
        baseline_requests = build_profiler_request_manifest(baseline)
        baseline_evidence = evidence_manifest(
            baseline_requests,
            tuple(
                complete_evidence(
                    request,
                    dispatches=(cuda_dispatch(
                        name=request.effective_candidate_id,
                        metric_scale=(
                            1.0
                            if request.candidate_id.endswith(".cpt1")
                            else 8.0
                        ),
                    ),),
                )
                for request in baseline_requests.requests
            ),
        )
        baseline_catalog = build_profiler_feature_catalog(
            baseline,
            profiler_feature_rows(
                baseline,
                baseline_requests,
                baseline_evidence,
            ),
            request_manifest_digest=baseline_requests.digest(),
            evidence_manifest_digest=baseline_evidence.digest(),
        )
        q5_rows = []
        for row in corpus:
            base = cuda_observation(
                source_format="Q5_0",
                candidate_id=row.candidate_id,
            )
            q5_rows.append(dataclasses.replace(
                base,
                projection_n_vector=row.projection_n_vector,
                aggregate_n=row.aggregate_n,
                k=row.k,
                aspect_ratio=row.aspect_ratio,
                aspect_bucket=row.aspect_bucket,
                work_items=row.work_items,
                shape_group_id=f"q5-{row.shape_group_id}",
                shape_name=f"q5-{row.shape_name}",
                n_tail_class=row.n_tail_class,
                k_tail_class=row.k_tail_class,
                min_us=row.min_us + 1.0,
                median_us=row.median_us + 1.0,
                p95_us=row.p95_us + 1.0,
            ))
        extended = ObservationCorpus((*tuple(baseline), *q5_rows))
        q5_corpus = ObservationCorpus(q5_rows)
        q5_requests = build_profiler_request_manifest(q5_corpus)
        q5_evidence = evidence_manifest(q5_requests, tuple(
            complete_evidence(
                request,
                dispatches=(cuda_dispatch(
                    name=request.effective_candidate_id,
                    metric_scale=(
                        1.0
                        if request.candidate_id.endswith(".cpt1")
                        else 8.0
                    ),
                ),),
            )
            for request in q5_requests.requests
        ))
        q5_feature_rows = profiler_feature_rows(
            q5_corpus,
            q5_requests,
            q5_evidence,
        )
        q5_catalog = build_profiler_feature_catalog(
            q5_corpus,
            q5_feature_rows,
            request_manifest_digest=q5_requests.digest(),
            evidence_manifest_digest=q5_evidence.digest(),
        )
        extended_catalog = merge_profiler_feature_catalogs(
            extended,
            (baseline_catalog, q5_catalog),
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            fit_generic_policy(
                baseline,
                max_leaves=2,
                min_shape_groups_per_leaf=1,
                policy_accelerators=(),
                profiler_feature_catalog=baseline_catalog,
                fit_final_rules=False,
                fit_cache=cache,
            )
            evaluated = []
            original = _domain_fold_tasks

            def record_domain(domain, *args, **kwargs):
                evaluated.append(domain)
                return original(domain, *args, **kwargs)

            with mock.patch(
                "native_vnni_dispatch.segmented_policy._domain_fold_tasks",
                side_effect=record_domain,
            ):
                fit_generic_policy(
                    extended,
                    max_leaves=2,
                    min_shape_groups_per_leaf=1,
                    policy_accelerators=(),
                    profiler_feature_catalog=extended_catalog,
                    fit_final_rules=False,
                    fit_cache=cache,
                )

        self.assertEqual(
            {
                (domain.runtime_codebook_id, domain.m)
                for domain in evaluated
            },
            {(0, original_m), (0, other_m), (6, original_m)},
        )

    def test_profiler_fit_cache_migrates_legacy_catalog_without_pool_key(self) -> None:
        """A paid full-catalog CV fit migrates without any model recompute."""

        corpus, _costs, *_unused, catalog = profiler_model_fixture()
        domain = corpus.generic_domains()[0]
        arguments = {
            "max_leaves": 2,
            "min_shape_groups_per_leaf": 1,
            "cross_validation_seed": (
                "native-vnni-development-cv-v9-cross-fitted-publication"
            ),
            "policy_accelerators": (),
            "profiler_feature_catalog": catalog,
            "fit_final_rules": False,
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            baseline = fit_generic_policy(
                corpus,
                fit_cache=cache,
                **arguments,
            )
            cost_files = tuple(
                (Path(directory) / "candidate-costs").glob("*.json")
            )
            validation_files = tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )
            self.assertEqual(len(cost_files), 1)
            self.assertEqual(len(validation_files), 1)

            cost_key = cost_files[0].stem
            current_validation_key = validation_files[0].stem
            legacy_validation_key = cache.validation_key(
                domain,
                cost_key,
                max_leaves=arguments["max_leaves"],
                min_shape_groups_per_leaf=(
                    arguments["min_shape_groups_per_leaf"]
                ),
                cross_validation_seed=arguments["cross_validation_seed"],
                profiler_feature_catalog_digest=catalog.digest,
                fit_final_rules=False,
                profiler_training_pool_digest=None,
            )
            self.assertNotEqual(legacy_validation_key, current_validation_key)

            validation_files[0].unlink()
            cache.store_validation(
                legacy_validation_key,
                domain,
                baseline.cross_validation[0],
            )

            with mock.patch.object(
                PolicyFitCache,
                "load_costs",
                side_effect=AssertionError("legacy costs were deserialized"),
            ), mock.patch(
                "native_vnni_dispatch.segmented_policy."
                "_fit_profiler_prediction_pool",
                side_effect=AssertionError("profiler model was rebuilt"),
            ), mock.patch(
                "native_vnni_dispatch.segmented_policy."
                "_evaluate_placement_fold",
                side_effect=AssertionError("legacy CV was recomputed"),
            ):
                migrated = fit_generic_policy(
                    corpus,
                    fit_cache=cache,
                    **arguments,
                )

            self.assertEqual(migrated, baseline)
            self.assertTrue(
                cache._path(
                    "domain-cross-validation",
                    current_validation_key,
                ).exists()
            )

    def test_shape_resolved_formula_is_not_mislabeled_as_physical_launch(self) -> None:
        observation = cuda_observation()
        formula = next(
            candidate
            for candidate in cuda_native_vnni_gemv_registry().entries
            if candidate.config_json.get("family") == "kpar_formula"
        )
        projected = dataclasses.replace(
            observation,
            candidate_id=formula.candidate_id,
            effective_candidate_id=formula.effective_candidate_id,
            candidate_family=formula.candidate_family,
            config_json=formula.config_json,
            arithmetic_fingerprint=formula.arithmetic_fingerprint,
            candidate_policy_hash=formula.candidate_policy_hash(),
            observed_candidate_id=formula.effective_candidate_id,
        )
        projected.validate()

        with self.assertRaisesRegex(ValueError, "not a physical profiler launch"):
            profiler_request_for_observation(projected)

    def test_request_and_evidence_manifests_authenticate_round_trip(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        record = complete_evidence(requests.requests[0])
        sidecar = evidence_manifest(requests, (record,))

        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.json"
            evidence_path = Path(directory) / "evidence.json"
            write_profiler_request_manifest(request_path, requests)
            write_profiler_evidence_manifest(evidence_path, sidecar)
            loaded_requests = read_profiler_request_manifest(request_path)
            loaded_evidence = read_profiler_evidence_manifest(evidence_path)

        self.assertEqual(loaded_requests.digest(), requests.digest())
        self.assertEqual(loaded_evidence.digest(), sidecar.digest())
        report = validate_profiler_evidence_coverage(
            loaded_requests, loaded_evidence
        )
        self.assertTrue(report.complete)
        self.assertEqual(report.complete_count, 1)

    def test_immutable_v1_evidence_remains_valid_for_offline_refitting(self) -> None:
        """Collector upgrades must not force hardware-counter recollection."""

        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        collector_version = "native-vnni-isolated-profiler-v1"
        record = dataclasses.replace(
            complete_evidence(requests.requests[0]),
            collector_version=collector_version,
        )
        sidecar = ProfilerEvidenceManifest(
            request_manifest_digest=requests.digest(),
            corpus_digest=requests.corpus_digest,
            candidate_registry_digest=requests.candidate_registry_digest,
            evidence=(record,),
            collector_version=collector_version,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-evidence.json"
            write_profiler_evidence_manifest(path, sidecar)
            loaded = read_profiler_evidence_manifest(path)

        self.assertEqual(loaded.collector_version, collector_version)
        self.assertEqual(loaded.digest(), sidecar.digest())
        self.assertTrue(
            validate_profiler_evidence_coverage(requests, loaded).complete
        )

    def test_immutable_v9_request_manifest_survives_v10_refitting(self) -> None:
        """Reading old counters must preserve their authenticated request hash."""

        current = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        legacy = ProfilerRequestManifest(
            corpus_digest=current.corpus_digest,
            candidate_registry_digest=current.candidate_registry_digest,
            requests=current.requests,
            schema_version=LEGACY_PROFILER_REQUEST_SCHEMA_VERSION,
            learner_version="native-vnni-bounded-tree-beam-regret-v9",
            feature_schema_version=(
                "execution-mode-n-k-work-aspect-tile-occupancy-tree-v5"
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-requests.json"
            write_profiler_request_manifest(path, legacy)
            loaded = read_profiler_request_manifest(path)

        self.assertEqual(loaded.digest(), legacy.digest())
        self.assertEqual(loaded.learner_version, legacy.learner_version)
        self.assertEqual(
            loaded.feature_schema_version,
            legacy.feature_schema_version,
        )

    def test_request_manifest_rejects_unreviewed_training_provenance(self) -> None:
        """Compatible reuse remains an explicit allowlist, not a version bypass."""

        current = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        with self.assertRaisesRegex(ValueError, "learner_version"):
            dataclasses.replace(current, learner_version="unreviewed-learner")
        with self.assertRaisesRegex(ValueError, "feature_schema_version"):
            dataclasses.replace(
                current,
                feature_schema_version="unreviewed-feature-schema",
            )

    def test_profiler_catalog_rebinds_only_after_source_authentication(self) -> None:
        """Schema projections reuse counters without weakening corpus binding."""

        source, _costs, requests, evidence, *_unused = profiler_model_fixture()
        target = ObservationCorpus(tuple(
            dataclasses.replace(row, corpus_id="sha256:projected-corpus")
            for row in source
        ))

        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.json"
            evidence_path = Path(directory) / "evidence.json"
            write_profiler_request_manifest(request_path, requests)
            write_profiler_evidence_manifest(evidence_path, evidence)

            with self.assertRaisesRegex(ValueError, "another timing corpus"):
                load_profiler_feature_catalog(
                    target,
                    request_path,
                    evidence_path,
                )

            rebound = load_profiler_feature_catalog(
                target,
                request_path,
                evidence_path,
                source_corpus=source,
            )
            self.assertEqual(rebound.corpus_digest, target.digest())
            self.assertEqual(
                len(rebound.descriptors),
                len(requests.requests),
            )

            request_digests = {
                request.observation_digest for request in requests.requests
            }
            witnesses = source.subset(
                lambda row: row.digest() in request_digests
            )
            self.assertEqual(witnesses.digest(), source.digest())
            witness_rebound = load_profiler_feature_catalog(
                target,
                request_path,
                evidence_path,
                source_corpus=witnesses,
            )
            self.assertEqual(witness_rebound.digest, rebound.digest)

            incomplete_witnesses = ObservationCorpus(
                witnesses.observations[:-1]
            )
            with self.assertRaisesRegex(ValueError, "exactly the request"):
                load_profiler_feature_catalog(
                    target,
                    request_path,
                    evidence_path,
                    source_corpus=incomplete_witnesses,
                )

            candidate = cuda_native_vnni_gemv_registry().resolve(
                "cuda.nvnni.decode.fast_m1.wide.tn256.cpt2"
            )
            unknown = dataclasses.replace(
                next(iter(target)),
                candidate_id=candidate.candidate_id,
                effective_candidate_id=candidate.effective_candidate_id,
                candidate_family=candidate.candidate_family,
                config_json=candidate.config_json,
                arithmetic_fingerprint=candidate.arithmetic_fingerprint,
                candidate_policy_hash=candidate.candidate_policy_hash(),
                ordered_reduction=candidate.ordered_reduction,
                uses_atomic_reduction=candidate.uses_atomic_reduction,
                observed_candidate_id=candidate.effective_candidate_id,
            )
            with self.assertRaisesRegex(ValueError, "omits physical candidate"):
                load_profiler_feature_catalog(
                    ObservationCorpus((*target, unknown)),
                    request_path,
                    evidence_path,
                    source_corpus=source,
                )

    def test_profiler_catalog_cache_reuses_only_identical_source_artifacts(self) -> None:
        """Repeat fits bypass giant JSON parsing without weakening provenance."""

        source, _costs, requests, evidence, *_unused = profiler_model_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.json"
            evidence_path = root / "evidence.json"
            cache_path = root / "fit-cache" / "profiler-catalog.json"
            write_profiler_request_manifest(request_path, requests)
            write_profiler_evidence_manifest(evidence_path, evidence)

            first = load_profiler_feature_catalog(
                source,
                request_path,
                evidence_path,
                source_corpus=source,
                cache_path=cache_path,
            )
            self.assertTrue(cache_path.is_file())
            with mock.patch(
                "native_vnni_dispatch.profiler_model.read_profiler_request_manifest",
                side_effect=AssertionError("request JSON was reparsed"),
            ), mock.patch(
                "native_vnni_dispatch.profiler_model.read_profiler_evidence_manifest",
                side_effect=AssertionError("evidence JSON was reparsed"),
            ):
                second = load_profiler_feature_catalog(
                    source,
                    request_path,
                    evidence_path,
                    source_corpus=source,
                    cache_path=cache_path,
                )
            self.assertEqual(second.digest, first.digest)
            self.assertEqual(second.model_digest, first.model_digest)
            self.assertEqual(second.descriptors, first.descriptors)

    def test_request_manifest_rejects_content_tampering(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            write_profiler_request_manifest(path, requests)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["requests"][0]["k"] += 32
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                read_profiler_request_manifest(path)

    def test_multi_kernel_candidate_retains_ordered_dispatch_evidence(self) -> None:
        observations = ObservationCorpus((cuda_observation(),))
        requests = build_profiler_request_manifest(observations)
        record = complete_evidence(
            requests.requests[0],
            dispatches=(
                cuda_dispatch(0, "nativeVnniKPartProducer"),
                cuda_dispatch(1, "nativeVnniOrderedReducer"),
            ),
        )
        report = validate_profiler_evidence_coverage(
            requests, evidence_manifest(requests, (record,))
        )

        self.assertTrue(report.complete)
        self.assertEqual(len(record.dispatches), 2)

        rows = profiler_feature_rows(
            observations, requests, evidence_manifest(requests, (record,))
        )
        self.assertEqual([row.dispatch_index for row in rows], [0, 1])
        self.assertEqual(rows[0].dispatch_count, 2)
        self.assertEqual(rows[0].observation.median_us, 11.0)
        self.assertEqual(rows[1].kernel_name, "nativeVnniOrderedReducer")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            write_profiler_feature_csv(
                path,
                observations,
                requests,
                evidence_manifest(requests, (record,)),
            )
            with path.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))

        self.assertEqual(len(exported), 2)
        self.assertEqual(exported[0]["median_us"], "11.0")
        self.assertEqual(exported[0]["profiler.dispatch_count"], "2")
        self.assertEqual(
            exported[0]["profiler.kernel_name"], "nativeVnniKPartProducer"
        )
        self.assertEqual(
            exported[0]["gpu.duration_ns.availability"], "measured"
        )

    def test_feature_export_rejects_a_different_timing_corpus(self) -> None:
        observations = ObservationCorpus((cuda_observation(),))
        requests = build_profiler_request_manifest(observations)
        record = complete_evidence(requests.requests[0])
        changed = ObservationCorpus(
            (dataclasses.replace(cuda_observation(), median_us=11.5),)
        )

        with self.assertRaisesRegex(ValueError, "another timing corpus"):
            profiler_feature_rows(
                changed, requests, evidence_manifest(requests, (record,))
            )

    def test_required_metric_cannot_be_silently_unavailable(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        dispatch = cuda_dispatch()
        metrics = list(dispatch.metrics)
        definition = next(
            item for item in metric_definitions(Backend.CUDA) if item.required
        )
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.metric_id == definition.metric_id
        )
        metrics[index] = dataclasses.replace(
            metrics[index],
            availability=MetricAvailability.UNSUPPORTED_BY_TOOL,
            value=None,
            reason="unit tool omitted a required metric",
        )
        bad_dispatch = dataclasses.replace(dispatch, metrics=tuple(metrics))
        bad_record = complete_evidence(
            requests.requests[0], dispatches=(bad_dispatch,)
        )

        with self.assertRaisesRegex(ValueError, "required profiler metric"):
            validate_profiler_evidence_coverage(
                requests, evidence_manifest(requests, (bad_record,))
            )

    def test_optional_metric_unavailability_is_explicit_and_accepted(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        dispatch = cuda_dispatch()
        metrics = list(dispatch.metrics)
        definition = next(
            item for item in metric_definitions(Backend.CUDA) if not item.required
        )
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.metric_id == definition.metric_id
        )
        metrics[index] = dataclasses.replace(
            metrics[index],
            availability=MetricAvailability.UNSUPPORTED_BY_TOOL,
            value=None,
            reason="counter is unavailable on this architecture",
        )
        record = complete_evidence(
            requests.requests[0],
            dispatches=(dataclasses.replace(dispatch, metrics=tuple(metrics)),),
        )

        report = validate_profiler_evidence_coverage(
            requests, evidence_manifest(requests, (record,))
        )
        self.assertTrue(report.complete)

    def test_tool_failure_is_visible_during_incremental_collection(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        request = requests.requests[0]
        failure = ProfilerEvidence(
            request_id=request.request_id,
            observation_digest=request.observation_digest,
            backend=request.backend,
            status=ProfilerEvidenceStatus.TOOL_UNAVAILABLE,
            status_reason="ncu is not installed",
            profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
            profiler_tool_version="",
            metric_set_version=PROFILER_METRIC_SET_VERSION,
            collector_version=PROFILER_COLLECTOR_VERSION,
            command_digest="",
            raw_artifact_digest="",
            profiler_pass_count=0,
            target_launches_per_profiler_pass=1,
            dispatches=(),
        )
        sidecar = evidence_manifest(requests, (failure,))

        report = validate_profiler_evidence_coverage(
            requests, sidecar, require_complete=False
        )
        self.assertFalse(report.complete)
        self.assertEqual(report.failed_request_ids, (request.request_id,))
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            validate_profiler_evidence_coverage(requests, sidecar)

    def test_unreachable_candidate_has_explicit_non_profile_state(self) -> None:
        observation = cuda_observation(supported=False, forced_route_ok=False)
        requests = build_profiler_request_manifest(ObservationCorpus((observation,)))
        request = requests.requests[0]
        self.assertFalse(request.profile_required)
        unsupported = ProfilerEvidence(
            request_id=request.request_id,
            observation_digest=request.observation_digest,
            backend=request.backend,
            status=ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
            status_reason="canonical timing row proved the forced route unreachable",
            profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
            profiler_tool_version="",
            metric_set_version=PROFILER_METRIC_SET_VERSION,
            collector_version=PROFILER_COLLECTOR_VERSION,
            command_digest="",
            raw_artifact_digest="",
            profiler_pass_count=0,
            target_launches_per_profiler_pass=1,
            dispatches=(),
        )

        report = validate_profiler_evidence_coverage(
            requests, evidence_manifest(requests, (unsupported,))
        )
        self.assertTrue(report.complete)
        self.assertEqual(report.unsupported_count, 1)

    def test_missing_candidate_evidence_blocks_complete_gate(self) -> None:
        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        sidecar = evidence_manifest(requests, ())

        report = validate_profiler_evidence_coverage(
            requests, sidecar, require_complete=False
        )
        self.assertEqual(
            report.missing_request_ids, (requests.requests[0].request_id,)
        )
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            validate_profiler_evidence_coverage(requests, sidecar)

    def test_profiler_checkpoint_journal_recovers_complete_prefix(self) -> None:
        """A torn final append cannot discard earlier exact request records."""

        requests = build_profiler_request_manifest(
            ObservationCorpus((cuda_observation(),))
        )
        item = complete_evidence(requests.requests[0])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            _append_checkpoint_journal(output, requests, (item,))
            journal = Path(str(output) + ".inprogress.jsonl")
            self.assertFalse(output.exists())
            with journal.open("ab") as handle:
                handle.write(b'{"record_type":"evidence"')

            recovered = _load_incremental_evidence(
                output, requests, resume=True
            )

            self.assertEqual(recovered[item.request_id], item)
            self.assertTrue(journal.read_bytes().endswith(b"\n"))
            manifest = _write_checkpoint(output, requests, recovered)
            self.assertTrue(output.exists())
            self.assertFalse(journal.exists())
            self.assertEqual(manifest.evidence, (item,))

    def test_cuda_collector_environment_selects_exactly_one_surface(self) -> None:
        request = profiler_request_for_observation(cuda_observation())
        environment = _profile_environment(request, Path("/tmp/profile-unit"))

        self.assertEqual(
            environment["LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID"],
            request.request_id,
        )
        self.assertEqual(
            environment["LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES"],
            request.effective_candidate_id,
        )
        self.assertEqual(environment["LLAMINAR_CUDA_NVNNI_DECODE_M"], "1")
        self.assertEqual(
            environment["LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES"],
            "graph_captured",
        )
        self.assertEqual(environment["LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES"], "1")

    def test_profiler_command_preserves_explicit_noninteractive_sudo_prefix(self) -> None:
        """Admin-only NVIDIA counters retain privilege in command provenance."""

        options = CollectorOptions(
            backend=Backend.CUDA,
            binary=Path("/tmp/trainer"),
            tool=Path("/usr/local/cuda/bin/ncu"),
            raw_directory=Path("/tmp/raw"),
            tool_command_prefix=("sudo", "-n", "-E"),
        )

        self.assertEqual(
            _tool_command(options, "--version"),
            ["sudo", "-n", "-E", "/usr/local/cuda/bin/ncu", "--version"],
        )

    def test_gpu_profiler_lane_placement_is_backend_specific(self) -> None:
        """Concurrent CUDA and ROCm workers bind only their own device API."""

        cuda_environment = {}
        _apply_device_placement(cuda_environment, CollectorOptions(
            backend=Backend.CUDA,
            binary=Path("/tmp/cuda-trainer"),
            tool=Path("/tmp/ncu"),
            raw_directory=Path("/tmp/cuda-raw"),
            device_ordinal=1,
        ))
        rocm_environment = {}
        _apply_device_placement(rocm_environment, CollectorOptions(
            backend=Backend.ROCM,
            binary=Path("/tmp/rocm-trainer"),
            tool=Path("/tmp/rocprofv3"),
            raw_directory=Path("/tmp/rocm-raw"),
            device_ordinal=3,
        ))

        self.assertEqual(cuda_environment["CUDA_VISIBLE_DEVICES"], "1")
        self.assertNotIn("ROCR_VISIBLE_DEVICES", cuda_environment)
        self.assertEqual(rocm_environment["ROCR_VISIBLE_DEVICES"], "3")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", rocm_environment)

    def test_cpu_prefill_collector_selects_prefill_not_verifier_surface(self) -> None:
        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            backend=Backend.CPU,
            operation_kind="NativeVNNIPrefillProjection",
            m=64,
            effective_candidate_id=(
                "cpu.nvnni.prefill.two_row_tiles.nbc4.full_k"
            ),
            threading_or_stream_mode=(
                "openmp:build=AVX512:requested=AVX512:"
                "effective=AVX512:threads=28"
            ),
        )

        environment = _profile_environment(request, Path("/tmp/profile-unit"))

        self.assertEqual(
            environment["LLAMINAR_CPU_NVNNI_PREFILL_CANDIDATES"],
            request.effective_candidate_id,
        )
        self.assertEqual(environment["LLAMINAR_CPU_NVNNI_PREFILL_M"], "64")
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS", environment)

    def test_cpu_decode_collector_selects_first_class_m1_surface(self) -> None:
        """M=1 evidence must not run through the grouped-verifier trainer."""

        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            backend=Backend.CPU,
            operation_kind="NativeVNNIFastM1Projection",
            execution_mode=ExecutionMode.EAGER,
            m=1,
            effective_candidate_id="cpu.nvnni.decode.n_chunk_grid.nbc4",
            threading_or_stream_mode=(
                "openmp:build=AVX512:requested=AVX512:"
                "effective=AVX512:threads=28"
            ),
        )

        environment = _profile_environment(request, Path("/tmp/profile-unit"))

        self.assertEqual(
            environment["LLAMINAR_CPU_NVNNI_DECODE_CANDIDATES"],
            request.effective_candidate_id,
        )
        self.assertEqual(environment["LLAMINAR_CPU_NVNNI_DECODE_M"], "1")
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS", environment)
        self.assertEqual(
            _trainer_arguments(request),
            ["--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"],
        )

    def test_cpu_profiler_runs_direct_process_scoped_counter_harness(self) -> None:
        """The collector never wraps an exact launch in system-wide perf."""

        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            backend=Backend.CPU,
            effective_candidate_id="cpu.nvnni.verifier.pairwise",
            threading_or_stream_mode=(
                "openmp:build=AVX2:requested=AVX2:effective=AVX2:threads=1"
            ),
        )
        rows = {
            "cycles": 1000,
            "ref-cycles": 900,
            "instructions": 2500,
            "task-clock": 1,
            "wall-clock": 1,
            "branches": 200,
            "branch-misses": 3,
            "cache-references": 50,
            "cache-misses": 5,
            "L1-dcache-loads": 400,
            "L1-dcache-load-misses": 20,
            "LLC-loads": 40,
            "LLC-load-misses": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            options = CollectorOptions(
                backend=Backend.CPU,
                binary=Path("/bin/true"),
                tool=Path("/usr/bin/perf"),
                raw_directory=raw,
                cpu_list="0",
            )

            def run_perf(
                command: list[str],
                environment: dict[str, str],
                *_args: object,
                **_kwargs: object,
            ) -> mock.Mock:
                self.assertEqual(command[:3], ["taskset", "-c", "0"])
                self.assertNotIn("stat", command)
                self.assertEqual(
                    environment["LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH"],
                    str(raw / "perf-stat.csv"),
                )
                (raw / "perf-stat.csv").write_text(
                    "".join(
                        f"{value};{'nsec' if event in {'task-clock', 'wall-clock'} else ''};"
                        f"{event};1;100.00;;\n"
                        for event, value in rows.items()
                    ),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            with mock.patch(
                "native_vnni_dispatch.profiler_collectors._run_command",
                side_effect=run_perf,
            ), mock.patch(
                "native_vnni_dispatch.profiler_collectors.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout="perf version unit\n"),
            ):
                evidence = _collect_cpu(request, options, raw)

            self.assertEqual(evidence.status, ProfilerEvidenceStatus.COMPLETE)
            self.assertFalse(any(path.is_fifo() for path in raw.iterdir()))
            self.assertTrue(all(path.is_file() for path in raw.iterdir()))

    def test_cpu_profiler_batch_amortizes_setup_without_merging_points(self) -> None:
        """Candidate/M members share a process but retain exact TSV outputs."""

        base = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            request_id="batch-request-a",
            backend=Backend.CPU,
            operation_kind="NativeVNNIPrefillProjection",
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aggregate_n=2048,
            projection_n_vector=(2048,),
            k=512,
            candidate_id="cpu.nvnni.prefill.row_chunk_grid.full_k",
            effective_candidate_id=(
                "cpu.nvnni.prefill.row_chunk_grid.full_k"
            ),
            threading_or_stream_mode=(
                "openmp:build=AVX512:requested=AVX512:"
                "effective=AVX512:threads=28"
            ),
        )
        second = dataclasses.replace(
            base,
            request_id="batch-request-b",
            observation_digest="sha256:" + "b" * 64,
            candidate_id="cpu.nvnni.prefill.two_row_pair_grid.nbc4.full_k",
            effective_candidate_id=(
                "cpu.nvnni.prefill.two_row_pair_grid.nbc4.full_k"
            ),
            candidate_policy_hash="sha256:" + "c" * 64,
        )
        third = dataclasses.replace(
            base,
            request_id="batch-request-c",
            observation_digest="sha256:" + "d" * 64,
            m=256,
        )
        different_geometry = dataclasses.replace(
            base,
            request_id="batch-request-d",
            observation_digest="sha256:" + "e" * 64,
            aggregate_n=4096,
            projection_n_vector=(4096,),
        )

        batches = _build_cpu_process_batches(
            (base, second, third, different_geometry),
            maximum_size=3,
        )

        self.assertEqual(sorted(len(batch.requests) for batch in batches), [1, 3])
        shared = next(batch for batch in batches if len(batch.requests) == 3)
        self.assertEqual(
            {request.request_id for request in shared.requests},
            {"batch-request-a", "batch-request-b", "batch-request-c"},
        )
        cell_packed = _build_cpu_process_batches(
            (base, second, third), maximum_size=2
        )
        self.assertEqual(len(cell_packed), 2)
        self.assertEqual(
            [request.request_id for request in cell_packed[0].requests],
            ["batch-request-a", "batch-request-b"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_directories = {}
            for request in shared.requests:
                raw = root / request.request_id
                raw.mkdir()
                raw_directories[request.request_id] = raw
            plan = root / "requests.tsv"
            digest = _write_cpu_batch_plan(plan, shared, raw_directories)
            environment = _cpu_batch_environment(
                shared, plan, digest, root / "batch"
            )

            self.assertEqual(len(plan.read_text(encoding="utf-8").splitlines()), 4)
            self.assertEqual(
                environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_DIGEST"],
                digest,
            )
            self.assertNotIn(
                "LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID", environment
            )
            self.assertNotIn(
                "LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH", environment
            )
            self.assertEqual(
                environment["LLAMINAR_CPU_NVNNI_PREFILL_M"], "64,256"
            )
            self.assertEqual(
                environment["LLAMINAR_CPU_NVNNI_PREFILL_MAX_CASES"], "2"
            )

    def test_cpu_profiler_batch_preserves_completed_members_after_later_failure(
        self,
    ) -> None:
        """A failed later cell cannot erase an already published exact report."""

        base = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            request_id="batch-salvage-a",
            backend=Backend.CPU,
            operation_kind="NativeVNNIPrefillProjection",
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aggregate_n=2048,
            projection_n_vector=(2048,),
            k=512,
            candidate_id="cpu.nvnni.prefill.row_chunk_grid.full_k",
            effective_candidate_id=(
                "cpu.nvnni.prefill.row_chunk_grid.full_k"
            ),
            threading_or_stream_mode=(
                "openmp:build=AVX2:requested=AVX2:"
                "effective=AVX2:threads=1"
            ),
        )
        later = dataclasses.replace(
            base,
            request_id="batch-salvage-b",
            observation_digest="sha256:" + "7" * 64,
            m=256,
        )
        batch = _build_cpu_process_batches((base, later), 2)[0]
        counters = {
            "cycles": 1000,
            "ref-cycles": 900,
            "instructions": 2500,
            "task-clock": 1000,
            "wall-clock": 1000,
            "L1-dcache-loads": 400,
            "L1-dcache-load-misses": 20,
            "LLC-load-misses": 4,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = CollectorOptions(
                backend=Backend.CPU,
                binary=Path("/bin/true"),
                cpu_avx2_binary=Path("/bin/true"),
                tool=Path("/usr/bin/perf"),
                raw_directory=root,
                cpu_list="0",
            )

            def partial_batch(
                _command: list[str],
                environment: dict[str, str],
                *_args: object,
                **_kwargs: object,
            ) -> mock.Mock:
                with Path(
                    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_PATH"]
                ).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                Path(rows[0]["output_path"]).write_text(
                    "".join(
                        f"{value};{'nsec' if event in {'task-clock', 'wall-clock'} else ''};"
                        f"{event};1;100.0;;\n"
                        for event, value in counters.items()
                    ),
                    encoding="utf-8",
                )
                line = (
                    "[NativeVNNIProfiler][CPU_PREFILL] "
                    f"request={rows[0]['request_id']} candidate=test "
                    "format=Q4_K M=64 N=2048 K=512 launches=1\n"
                )
                return mock.Mock(returncode=1, stdout="", stderr=line)

            with mock.patch(
                "native_vnni_dispatch.profiler_collectors._run_command",
                side_effect=partial_batch,
            ), mock.patch(
                "native_vnni_dispatch.profiler_collectors._tool_version",
                return_value="perf unit",
            ):
                evidence = _collect_cpu_batch(batch, options)

        self.assertEqual(
            evidence["batch-salvage-a"].status,
            ProfilerEvidenceStatus.COMPLETE,
        )
        self.assertEqual(
            evidence["batch-salvage-b"].status,
            ProfilerEvidenceStatus.LAUNCH_FAILED,
        )

    def test_linux_perf_parser_normalizes_one_controlled_region(self) -> None:
        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            backend=Backend.CPU,
            effective_candidate_id="cpu.nvnni.verifier.pairwise",
        )
        rows = {
            "cycles": (1000, ""),
            "ref-cycles": (900, ""),
            "instructions": (2500, ""),
            "task-clock": (0.5, "msec"),
            "wall-clock": (250000, "nsec"),
            "branches": (200, ""),
            "branch-misses": (3, ""),
            "cache-references": (50, ""),
            "cache-misses": (5, ""),
            "L1-dcache-loads": (400, ""),
            "L1-dcache-load-misses": (20, ""),
            "LLC-loads": (40, ""),
            "LLC-load-misses": (4, ""),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf.csv"
            path.write_text(
                "".join(
                    f"{value};{unit};{event};1;100.00;;\n"
                    for event, (value, unit) in rows.items()
                ),
                encoding="utf-8",
            )
            dispatches = parse_perf_stat(path, request)

        metrics = {metric.metric_id: metric for metric in dispatches[0].metrics}
        self.assertEqual(metrics["cpu.cycles"].value, 1000.0)
        self.assertEqual(metrics["cpu.ref_cycles"].value, 900.0)
        self.assertEqual(metrics["cpu.instructions"].value, 2500.0)
        self.assertEqual(metrics["cpu.task_clock_ns"].value, 500000.0)
        self.assertEqual(metrics["cpu.wall_clock_ns"].value, 250000.0)

    def test_ncu_parser_retains_pipeline_dispatch_order_and_required_metrics(self) -> None:
        request = profiler_request_for_observation(cuda_observation())
        required_rows = (
            ("gpu__time_duration.sum", "nsecond", "1200"),
            ("launch__registers_per_thread", "register/thread", "48"),
            ("launch__shared_mem_per_block_static", "byte/block", "0"),
            ("launch__shared_mem_per_block_dynamic", "byte/block", "4096"),
            ("sm__maximum_warps_per_active_cycle_pct", "%", "50"),
            ("sm__warps_active.avg.pct_of_peak_sustained_active", "%", "42"),
            ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "71"),
            ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "36"),
        )
        header = (
            '"ID","Kernel Name","Block Size","Grid Size","Metric Name",'
            '"Metric Unit","Metric Value"\n'
        )
        body = ""
        for identifier, kernel in ((0, "producer"), (1, "orderedReducer")):
            for metric_name, unit, value in required_rows:
                body += (
                    f'"{identifier}","{kernel}","(128, 1, 1)",'
                    f'"(16, 1, 1)","{metric_name}","{unit}","{value}"\n'
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu.csv"
            path.write_text("==PROF== unit banner\n" + header + body, encoding="utf-8")
            dispatches = parse_ncu_csv(path, request)

        self.assertEqual(
            [dispatch.kernel_name for dispatch in dispatches],
            ["producer", "orderedReducer"],
        )
        self.assertEqual(dispatches[0].block, (128, 1, 1))
        self.assertEqual(dispatches[0].grid, (16, 1, 1))

    def test_ncu_parser_accepts_2025_wide_raw_metric_export(self) -> None:
        """Nsight Compute 2025.3 emits one metric column per dispatch row."""

        request = profiler_request_for_observation(cuda_observation())
        metric_columns = (
            "gpu__time_duration.sum",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "launch__shared_mem_per_block_dynamic",
            "sm__maximum_warps_per_active_cycle_pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        )
        columns = (
            "ID",
            "Kernel Name",
            "Block Size",
            "Grid Size",
            *metric_columns,
        )
        units = (
            "",
            "",
            "",
            "",
            "us",
            "register/thread",
            "byte/block",
            "byte/block",
            "%",
            "%",
            "%",
            "%",
        )
        values = ("3", "nativeVnni", "(128, 1, 1)", "(16, 1, 1)",
                  "1.25", "40", "0", "1024", "50", "42", "71", "36")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu-wide.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow(units)
                writer.writerow(values)
            dispatches = parse_ncu_csv(path, request)

        metrics = {metric.metric_id: metric for metric in dispatches[0].metrics}
        self.assertEqual(dispatches[0].kernel_name, "nativeVnni")
        self.assertEqual(metrics["gpu.duration_ns"].value, 1250.0)
        self.assertEqual(metrics["gpu.registers_per_thread"].value, 40.0)
        self.assertEqual(metrics["gpu.dram_throughput_pct_of_peak"].value, 36.0)

    def test_rocprof_parser_normalizes_static_resources_and_dynamic_counters(self) -> None:
        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            backend=Backend.ROCM,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rocprof.csv"
            path.write_text(
                "KernelName,Index,wgr,arch_vgpr,sgpr,lds,scr,DurationNs,"
                "GPUBusy,VALUBusy,MemUnitBusy\n"
                "nativeVnni,0,256,64,48,8192,0,2500,92,73,66\n",
                encoding="utf-8",
            )
            dispatches = parse_rocprof_csvs((path,), request)

        metrics = {metric.metric_id: metric for metric in dispatches[0].metrics}
        self.assertEqual(metrics["gpu.vgpr_count"].value, 64.0)
        self.assertEqual(metrics["gpu.sgpr_count"].value, 48.0)
        self.assertEqual(metrics["gpu.gpu_busy_pct"].value, 92.0)
        self.assertEqual(dispatches[0].block, (256, 1, 1))

    def test_rocprof_counter_ranges_join_to_physical_trace_by_dispatch_order(self) -> None:
        request = dataclasses.replace(
            profiler_request_for_observation(cuda_observation()),
            request_id="rocm-range-unit",
            backend=Backend.ROCM,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / "trace"
            counter_dir = root / "counters-0"
            trace_dir.mkdir()
            counter_dir.mkdir()
            (trace_dir / "profile_kernel_trace.csv").write_text(
                "Kernel_Name,Dispatch_Id,Start_Timestamp,End_Timestamp,"
                "LDS_Block_Size,Scratch_Size,VGPR_Count,SGPR_Count,"
                "Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z,"
                "Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n"
                "physicalQuantize,8,100,140,0,0,8,32,32,1,1,896,1,1\n"
                "physicalGemm,9,150,250,4096,0,64,48,128,1,1,16,1,1\n",
                encoding="utf-8",
            )
            counter_header = (
                "Kernel_Name,Dispatch_Id,Counter_Name,Counter_Value,"
                "Start_Timestamp,End_Timestamp,Workgroup_Size,Grid_Size,"
                "LDS_Block_Size,Scratch_Size,VGPR_Count,SGPR_Count\n"
            )
            counter_rows = []
            for dispatch_id, values in (
                (16, {"GPUBusy": 90, "VALUBusy": 40, "MemUnitBusy": 25}),
                (17, {"GPUBusy": 95, "VALUBusy": 70, "MemUnitBusy": 50}),
            ):
                for name, value in values.items():
                    counter_rows.append(
                        f"NativeVNNIProfile::rocm-range-unit,{dispatch_id},"
                        f"{name},{value},100,200,64,896,0,0,20,48\n"
                    )
            (counter_dir / "profile_counter_collection.csv").write_text(
                counter_header + "".join(counter_rows), encoding="utf-8"
            )
            dispatches = parse_rocprof_csvs(root.rglob("*.csv"), request)

        self.assertEqual(
            [dispatch.kernel_name for dispatch in dispatches],
            ["physicalQuantize", "physicalGemm"],
        )
        first = {metric.metric_id: metric for metric in dispatches[0].metrics}
        second = {metric.metric_id: metric for metric in dispatches[1].metrics}
        self.assertEqual(first["gpu.gpu_busy_pct"].value, 90.0)
        self.assertEqual(second["gpu.gpu_busy_pct"].value, 95.0)
        self.assertEqual(second["gpu.vgpr_count"].value, 64.0)

    def test_missing_profiler_tool_is_typed_evidence_not_silent_omission(self) -> None:
        request = profiler_request_for_observation(cuda_observation())
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trainer"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            result = collect_request(
                request,
                CollectorOptions(
                    backend=Backend.CUDA,
                    binary=binary,
                    tool=Path(directory) / "missing-ncu",
                    raw_directory=Path(directory) / "raw",
                ),
            )

        self.assertEqual(result.status, ProfilerEvidenceStatus.TOOL_UNAVAILABLE)
        self.assertIn("unavailable", result.status_reason or "")

    def test_raw_profile_authenticates_the_actual_trainer_binary(self) -> None:
        request = profiler_request_for_observation(cuda_observation())
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trainer"
            binary.write_bytes(b"unit-profiler-binary")
            record = Path(directory) / "profiler-binary.json"

            _write_binary_provenance(record, request, binary)
            payload = json.loads(record.read_text(encoding="utf-8"))

        self.assertEqual(payload["timing_build_id"], request.build_id)
        self.assertEqual(
            payload["candidate_policy_hash"], request.candidate_policy_hash
        )
        self.assertEqual(
            payload["profiler_binary_digest"],
            "sha256:c984df6010ad99eefdd2fee14a48d0d649d78029f26a2fddb0b2fbf86dd8f17e",
        )


if __name__ == "__main__":
    unittest.main()
