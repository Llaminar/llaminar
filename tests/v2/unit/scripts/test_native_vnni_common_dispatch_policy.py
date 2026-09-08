#!/usr/bin/env python3
"""Regression tests for the common NativeVNNI dispatch policy compiler."""

from __future__ import annotations

import dataclasses
import csv
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from enum import Enum
from fractions import Fraction
from unittest import mock
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch import certification as certification_module  # noqa: E402
from native_vnni_dispatch.certification import certify_generic_policy  # noqa: E402
from native_vnni_dispatch import corpus as corpus_module  # noqa: E402
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_decode_registry,
    cpu_native_vnni_prefill_registry,
    cpu_native_vnni_verifier_registry,
    cuda_native_vnni_gemv_registry,
    rocm_moe_grouped_prefill_registry,
    rocm_native_vnni_decode_formula_registry,
    rocm_native_vnni_decode_registry,
)
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    certify_frozen_policy,
    compile_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import (  # noqa: E402
    GenericDomain,
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
    generic_domain,
    runtime_key,
)
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    predicate_condition,
)
from native_vnni_dispatch.cuda_shape_resolved import (  # noqa: E402
    _nearest_factor_kb,
    project_cuda_shape_resolved_candidates,
    resolve_cuda_formula_kb,
    resolve_cuda_concrete_candidate_id,
)
from native_vnni_dispatch.exact_oracle import (  # noqa: E402
    build_exact_winner,
    build_exact_winners,
    candidate_is_eligible,
)
from native_vnni_dispatch import exact_oracle as exact_oracle_module  # noqa: E402
from native_vnni_dispatch.format_registry import (  # noqa: E402
    FORMAT_SPECS,
    format_spec,
    registry_digest,
    runtime_aliases,
)
from native_vnni_dispatch.policy_ir import (  # noqa: E402
    ExactDispatchEntry,
    PolicyIR,
    _normalize_policy_inventory,
    make_policy_ir,
)
from native_vnni_dispatch.rocm_shape_resolved import (  # noqa: E402
    project_rocm_shape_resolved_candidates,
    resolve_rocm_formula_kb,
)
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_frozen_policy_file,
    validate_installable_policy_artifact,
    write_certification_diagnostic,
    write_frozen_policy,
    write_compiled_policy,
)
from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    PairedCellEvidence,
    PairedCellKey,
    paired_timing_comparisons,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    FEATURE_SCHEMA_VERSION,
    LEARNER_VERSION,
    NativeVNNIObservation,
    P95_REGRET_BUDGET,
    POLICY_ABI,
    SemanticContract,
    classify_aspect,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    BoundaryPlacement,
    FeatureAxis,
    FeaturePolicy,
    FeaturePredicate,
    FeatureThreshold,
    GenericPolicy,
    GenericDispatchRule,
    PolicyFitCache,
    build_candidate_point_costs,
    fit_generic_policy,
    generic_domain_corpus_digest,
    validate_generic_rule_partition,
)
from native_vnni_dispatch.splits import (  # noqa: E402
    Partition,
    SplitManifest,
)
import native_vnni_dispatch.segmented_policy as segmented_policy  # noqa: E402
from native_vnni_dispatch.validation import (  # noqa: E402
    require_candidate_matrix_complete,
    require_exact_overlay_scope,
    require_verifier_m_matrix,
)
from native_vnni_dispatch.shape_manifest import load_shape_manifest  # noqa: E402
from native_vnni_dispatch.adapters.rocm_moe import (  # noqa: E402
    ROCmMoEAdapterContext,
    adapt_rocm_moe_csv,
    adapt_rocm_moe_row,
    raw_corpus_id,
    timing_sample_digest,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerCandidateDescriptor,
    ProfilerFeatureCatalog,
    _physical_key,
)


SERIAL_HASH = "sha256:serial-m1-policy-v1"


@dataclasses.dataclass(frozen=True)
class _SchedulerProbeSpec:
    """Minimal device identity consumed by the scheduler-only fork probe."""

    label: str


def _scheduler_probe_worker(
    spec,
    task_kind,
    connection,
    expected_parent_pid,
) -> None:
    """Model one slow lane and one fast lane without loading a GPU runtime."""

    del task_kind, expected_parent_pid
    connection.send(("ready",))
    lane_sequence = 0
    while True:
        message = connection.recv()
        if message[0] == "stop":
            connection.send(("done",))
            connection.close()
            return
        task_index = message[1]
        if spec.label == "slow" and task_index == 0:
            time.sleep(0.2)
        else:
            time.sleep(0.005)
        connection.send((
            "result",
            task_index,
            (spec.label, task_index),
            0,
            lane_sequence,
            0.2 if spec.label == "slow" and task_index == 0 else 0.005,
        ))
        lane_sequence += 1


def observation(
    *,
    source_format: str = "Q4_0",
    candidate: str = "candidate.a",
    family: str = "family-a",
    shape_group: str = "shape-0",
    shape_name: str | None = None,
    n: int = 512,
    k: int = 2048,
    m: int = 2,
    latency_us: float = 10.0,
    p95_scale: float = 1.0,
    cv: float = 0.0,
    mode: ExecutionMode = ExecutionMode.GRAPH_CAPTURED,
    contract: SemanticContract = SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
    bitwise_equal: bool = True,
    repeat_equal: bool = True,
    uses_atomic: bool = False,
    route_ok: bool = True,
    generic_eligible: bool = True,
    serial_hash: str = SERIAL_HASH,
    backend: Backend = Backend.ROCM,
) -> NativeVNNIObservation:
    """Build one complete strict observation for compact synthetic corpora."""

    spec = format_spec(source_format)
    backend_name = backend.value
    runtime_codebook = spec.runtime_codebook(backend_name)
    mismatch_count = 0 if bitwise_equal else 1
    result = NativeVNNIObservation(
        schema_version=1,
        run_id="unit-run",
        corpus_id="sha256:unit-corpus",
        git_revision="0123456789abcdef",
        build_id="unit-release-build",
        compiler_id="unit-compiler",
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        backend=backend,
        architecture_class=f"unit-{backend_name}-native-vnni-v1",
        device_name=f"unit-{backend_name}-device",
        driver_runtime=f"unit-{backend_name}-runtime",
        threading_or_stream_mode="explicit_non_default_stream",
        semantic_contract=contract,
        operation_kind="SingleProjection",
        bundle_signature=f"single:cb{runtime_codebook}:fp32:v1",
        projection_n_vector=(n,),
        source_format=spec.label,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family(backend_name),
        packing_abi=spec.packing_abi(backend_name),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=shape_group,
        shape_name=shape_name or shape_group,
        execution_mode=mode,
        m=m,
        aggregate_n=n,
        k=k,
        aspect_ratio=float(n) / float(k),
        aspect_bucket=classify_aspect(n, k),
        work_items=n * k,
        n_tail_class=f"n_mod_256={n % 256}",
        k_tail_class=f"k_mod_256={k % 256}",
        alignment_class="prepared_16b_aligned",
        candidate_id=candidate,
        effective_candidate_id=candidate,
        candidate_family=family,
        config_json={"candidate": candidate},
        supported=True,
        graph_capture_ok=True,
        generic_eligible=generic_eligible,
        arithmetic_fingerprint=f"sha256:arithmetic:{candidate}",
        serial_m1_policy_id="serial.m1.production",
        serial_m1_policy_hash=serial_hash,
        candidate_policy_hash=f"sha256:policy:{candidate}",
        ordered_reduction=not uses_atomic,
        uses_atomic_reduction=uses_atomic,
        trial_set_hash="sha256:unit-trials",
        numerical_correctness=True,
        bitwise_equal=bitwise_equal,
        repeat_equal=repeat_equal,
        mismatch_count=mismatch_count,
        first_mismatch_index=None if bitwise_equal else 7,
        grouped_output_digest=(
            "sha256:equal-output" if bitwise_equal else "sha256:one-ulp-output"
        ),
        serial_output_digest="sha256:equal-output",
        max_abs=0.0 if bitwise_equal else 1.0e-7,
        relative_l2=0.0 if bitwise_equal else 1.0e-9,
        cosine=1.0,
        symmetric_kld=0.0,
        warmup_count=5,
        sample_count=30,
        min_us=latency_us,
        median_us=latency_us,
        p95_us=latency_us * p95_scale,
        mad_us=latency_us * cv * 0.67,
        cv=cv,
        timing_sample_hash=f"sha256:timings:{candidate}:{shape_group}:{source_format}:{mode.value}",
        effective_bandwidth_gbs=100.0,
        forced_route_ok=route_ok,
        observed_candidate_id=candidate,
        route_counter_ok=route_ok,
        workspace_ok=True,
        explicit_stream_ok=True,
    )
    result.validate()
    return result


def candidate_rows_for_aliases(
    candidate: str,
    latencies: dict[str, float],
    **kwargs,
) -> list[NativeVNNIObservation]:
    """Create one graph-captured row for each named source-format alias."""

    return [
        observation(
            source_format=source_format,
            candidate=candidate,
            family=candidate,
            latency_us=latency,
            **kwargs,
        )
        for source_format, latency in latencies.items()
    ]


class NativeVNNICommonDispatchPolicyTest(unittest.TestCase):
    def test_immutable_policy_keys_memoize_only_their_structural_hash(self) -> None:
        """Repeated set joins must reuse hashes without changing value identity."""
        row = observation(shape_group="key-hash", n=384, k=1024)
        for key in (runtime_key(row), generic_domain(row)):
            with self.subTest(key=type(key).__name__):
                fields = dataclasses.fields(key)
                expected = hash(tuple(getattr(key, field.name) for field in fields))
                self.assertEqual(hash(key), expected)
                self.assertEqual(key.__dict__["_cached_structural_hash"], expected)
                with mock.patch("native_vnni_dispatch.corpus.fields",
                                side_effect=AssertionError("warmed hash recomputed fields")):
                    self.assertEqual(hash(key), expected)
                    self.assertIn(key, {key})
                self.assertNotIn("_cached_structural_hash", dataclasses.asdict(key))
                self.assertNotIn("_cached_structural_hash", key.__getstate__())
                copied = pickle.loads(pickle.dumps(key))
                self.assertNotIn("_cached_structural_hash", copied.__dict__)
                self.assertEqual(copied, key)
                self.assertEqual(hash(copied), expected)
                changed = dataclasses.replace(key, architecture_class="another-architecture")
                self.assertNotEqual(changed, key)
                self.assertEqual(hash(changed), hash(tuple(
                    getattr(changed, field.name) for field in fields)))

    def test_immutable_policy_key_pickle_rehashes_under_another_python_seed(self) -> None:
        """A memoized salted hash may never cross spawn/interpreter boundaries."""
        row = observation(shape_group="key-hash-seed", n=384, k=1024)
        keys = (runtime_key(row), generic_domain(row))
        for key in keys:
            hash(key)
        encoded = pickle.dumps({key: "present" for key in keys}).hex()
        program = (
            "import dataclasses,pickle,sys; "
            "values=pickle.loads(bytes.fromhex(sys.argv[1])); "
            "assert len(values)==2; "
            "assert all(values[dataclasses.replace(key)]=='present' for key in values); "
            "assert all(hash(key)==hash(tuple(getattr(key,f.name) for f in "
            "dataclasses.fields(key))) for key in values)"
        )
        for seed in ("1", "8675309"):
            with self.subTest(seed=seed):
                subprocess.run([sys.executable, "-c", program, encoded], check=True,
                    capture_output=True, text=True, timeout=30,
                    env={**os.environ, "PYTHONHASHSEED": seed,
                         "PYTHONPATH": str(KERNEL_PERF_ROOT)})

    def test_parallel_policy_serializer_preserves_legacy_digest(self) -> None:
        """Fast publication must retain the historical policy artifact bytes."""

        row = observation(shape_group="serializer-shape", n=384, k=1024)
        key = runtime_key(row)
        domain = generic_domain(row)
        cell = segmented_policy.CrossValidationCell(
            runtime_key=key,
            shape_group_id=row.shape_group_id,
            selected_candidate_id=row.effective_candidate_id,
            exact_candidate_id=row.effective_candidate_id,
            observed_broad_regret=0.0,
        )
        validation = segmented_policy.DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=1,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=(
                segmented_policy.ProfilerInfluence.MEASURED_ONLY
            ),
            fold_count=1,
            shape_group_count=1,
            required_point_count=1,
            covered_point_count=1,
            max_regret=0.0,
            p95_regret=0.0,
            mean_regret=0.0,
            worst_shape_group_id=row.shape_group_id,
            worst_aggregate_n=row.aggregate_n,
            worst_k=row.k,
            worst_selected_candidate_id=row.effective_candidate_id,
            worst_exact_candidate_id=row.effective_candidate_id,
            cells=(cell,),
        )
        rule = GenericDispatchRule(
            domain=domain,
            predicates=(),
            candidate_id=row.effective_candidate_id,
            arithmetic_fingerprint=row.arithmetic_fingerprint,
            development_shape_groups=(row.shape_group_id,),
            development_max_regret=0.0,
            development_p95_regret=0.0,
            development_mean_regret=0.0,
        )
        exact = ExactDispatchEntry(
            key=key,
            candidate_id=row.effective_candidate_id,
            arithmetic_fingerprint=row.arithmetic_fingerprint,
            config_json={"nested": {"candidate": row.effective_candidate_id}},
        )
        policy = PolicyIR(
            policy_abi=POLICY_ABI,
            learner_version=LEARNER_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            exact_entries=(exact,),
            generic_rules=(rule,),
            unpromoted_domains=(),
            cross_validation=(validation,),
            metadata={"regime": ExecutionMode.GRAPH_CAPTURED},
        )

        def legacy_normalize(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {
                    str(name): legacy_normalize(item)
                    for name, item in sorted(value.items())
                }
            if isinstance(value, (tuple, list)):
                return [legacy_normalize(item) for item in value]
            if hasattr(value, "__dataclass_fields__"):
                return legacy_normalize(dataclasses.asdict(value))
            return value

        expected_generic = {
            "policy_abi": policy.policy_abi,
            "learner_version": policy.learner_version,
            "feature_schema_version": policy.feature_schema_version,
            "generic_rules": legacy_normalize(policy.generic_rules),
            "unpromoted_domains": legacy_normalize(policy.unpromoted_domains),
            "cross_validation": legacy_normalize(policy.cross_validation),
        }
        expected_complete = dict(expected_generic)
        expected_complete["exact_entries"] = legacy_normalize(
            policy.exact_entries
        )
        expected_complete["metadata"] = legacy_normalize(dict(policy.metadata))

        self.assertEqual(policy.canonical_mapping(generic_only=True), expected_generic)
        self.assertEqual(policy.canonical_mapping(), expected_complete)
        self.assertIs(policy.canonical_mapping(), policy.canonical_mapping())
        self.assertIs(
            policy.canonical_mapping(generic_only=True),
            policy.canonical_mapping(generic_only=True),
        )
        for generic_only, expected in (
            (True, expected_generic),
            (False, expected_complete),
        ):
            encoded = json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ).encode()
            expected_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
            self.assertEqual(
                policy.digest(generic_only=generic_only), expected_digest
            )

        repeated = (validation,) * 8
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_SERIALIZATION_WORKERS": "2"},
        ):
            parallel = _normalize_policy_inventory(
                repeated, items_per_worker=1
            )
        self.assertEqual(parallel, legacy_normalize(repeated))

    def test_policy_worker_default_counts_affinity_visible_physical_cores(
        self,
    ) -> None:
        """SMT siblings share one default process-pool worker slot."""

        topology = (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 1),
            (3, 0, 1),
            (4, 1, 0),
            (5, 1, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu, package_id, core_id in topology:
                cpu_topology = root / f"cpu{cpu}" / "topology"
                cpu_topology.mkdir(parents=True)
                (cpu_topology / "physical_package_id").write_text(
                    f"{package_id}\n", encoding="utf-8"
                )
                (cpu_topology / "core_id").write_text(
                    f"{core_id}\n", encoding="utf-8"
                )

            self.assertEqual(
                segmented_policy._physical_core_worker_count(
                    affinity=range(6), topology_root=root
                ),
                3,
            )
            self.assertEqual(
                segmented_policy._physical_core_worker_count(
                    affinity=(0, 1, 4, 5), topology_root=root
                ),
                2,
            )

            # Incomplete topology fails conservatively to the visible logical
            # count instead of guessing an SMT width.
            (root / "cpu5" / "topology" / "core_id").unlink()
            self.assertEqual(
                segmented_policy._physical_core_worker_count(
                    affinity=range(6), topology_root=root
                ),
                6,
            )

    def test_cv_reduction_workers_never_exceed_physical_cores(self) -> None:
        """Explicit tuning cannot schedule reduction work onto SMT siblings."""

        with mock.patch.object(
            segmented_policy,
            "_physical_core_worker_count",
            return_value=3,
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "99"},
        ):
            self.assertEqual(
                segmented_policy._cv_reduction_worker_count(10),
                3,
            )
            self.assertEqual(
                segmented_policy._cv_reduction_worker_count(
                    2, requested_workers=99
                ),
                2,
            )
            self.assertEqual(
                segmented_policy._cv_reduction_worker_count(
                    10, requested_workers=1
                ),
                1,
            )
            self.assertEqual(
                segmented_policy._cv_reduction_worker_count(0),
                0,
            )

    def test_corpus_full_subset_reuses_validated_immutable_indices(self) -> None:
        """A phase-only file must not rebuild its complete corpus indices."""

        corpus = ObservationCorpus((observation(),))
        self.assertIs(corpus.subset(lambda row: True), corpus)

    def test_corpus_proper_subset_preserves_runtime_visibility(self) -> None:
        """A real subset rebuilds indices without changing key semantics."""

        first = observation(shape_group="first", shape_name="first")
        second = observation(shape_group="second", shape_name="second")
        corpus = ObservationCorpus(
            (first, second),
            distinguish_execution_mode=False,
            distinguish_aspect_bucket=False,
        )
        selected = corpus.subset(lambda row: row.shape_name == "first")

        self.assertEqual(selected.observations, (first,))
        self.assertFalse(selected.distinguishes_execution_mode)
        self.assertFalse(selected.distinguishes_aspect_bucket)

    def test_parallel_corpus_digest_preserves_legacy_bytes_and_is_memoized(
        self,
    ) -> None:
        """Parallel canonicalization must not fork the evidence identity."""

        # Repeating a validated immutable row is enough to cross the production
        # parallelization threshold without manufacturing thousands of unique
        # runtime surfaces.  Two workers exercise ordered shard reduction.
        row = observation()
        corpus = ObservationCorpus((row,) * 8192)
        canonical = json.dumps(
            row.canonical_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        )
        historical_bytes = (
            "[" + ",".join((canonical,) * 8192) + "]"
        ).encode()
        expected = "sha256:" + hashlib.sha256(historical_bytes).hexdigest()

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "2"},
        ):
            first = corpus.digest()
        self.assertEqual(first, expected)

        # A second call must be a constant-time identity lookup.  An invalid
        # worker setting would fail if digest computation ran a second time.
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "0"},
        ):
            self.assertEqual(corpus.digest(), expected)

    def test_parallel_corpus_digest_caps_workers_at_physical_cores(self) -> None:
        """Offline hashing must not schedule one process per SMT sibling."""

        corpus = ObservationCorpus((observation(),) * 8192)
        with (
            mock.patch.object(
                corpus_module,
                "_physical_core_count",
                return_value=1,
            ),
            mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "99"},
            ),
            mock.patch.object(
                corpus_module,
                "ProcessPoolExecutor",
            ) as executor,
        ):
            corpus.digest()
        executor.assert_not_called()

    def test_parallel_corpus_digest_reuses_one_process_pool(self) -> None:
        """Serialization, partitioning, and merge share one fork lifetime."""

        corpus = ObservationCorpus((observation(),) * 8192)
        real_executor = corpus_module.ProcessPoolExecutor
        with (
            mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "2"},
            ),
            mock.patch.object(
                corpus_module,
                "ProcessPoolExecutor",
                side_effect=lambda *args, **kwargs: real_executor(
                    *args, **kwargs
                ),
            ) as executor_factory,
        ):
            corpus.digest()

        self.assertEqual(executor_factory.call_count, 1)

    def test_parallel_corpus_digest_preserves_collapsed_prefix_order(self) -> None:
        """Mode/aspect visibility markers retain their historical hash order."""

        row = observation()
        corpus = ObservationCorpus(
            (row,) * 8192,
            distinguish_execution_mode=False,
            distinguish_aspect_bucket=False,
        )
        canonical = json.dumps(
            row.canonical_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        )
        historical_bytes = (
            "aspect-collapsed:mode-collapsed:["
            + ",".join((canonical,) * 8192)
            + "]"
        ).encode()
        expected = "sha256:" + hashlib.sha256(historical_bytes).hexdigest()

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "2"},
        ):
            self.assertEqual(corpus.digest(), expected)

    def test_parallel_corpus_digest_range_merge_preserves_global_order(self) -> None:
        """Disjoint merge buckets must reproduce one monolithic lexical sort."""

        exemplars = tuple(
            observation(
                shape_group=f"digest-shape-{index}",
                shape_name=f"digest-shape-{index}",
            )
            for index in range(8)
        )
        rows = tuple(exemplars[index % len(exemplars)] for index in range(8192))
        corpus = ObservationCorpus(rows)
        expected = "sha256:" + hashlib.sha256(
            (
                "["
                + ",".join(sorted(row._cached_canonical_json for row in rows))
                + "]"
            ).encode()
        ).hexdigest()

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "2"},
        ):
            self.assertEqual(corpus.digest(), expected)

    def test_row_and_corpus_digests_share_one_canonical_serialization(self) -> None:
        """Profiler-row authentication must not encode immutable rows twice."""

        row = observation()
        row_digest = row.digest()
        canonical = row._cached_canonical_json
        expected_corpus = "sha256:" + hashlib.sha256(
            ("[" + canonical + "]").encode()
        ).hexdigest()

        with mock.patch.object(
            NativeVNNIObservation,
            "canonical_mapping",
            side_effect=AssertionError("canonical observation was reserialized"),
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS": "1"},
        ):
            self.assertEqual(ObservationCorpus((row,)).digest(), expected_corpus)
            self.assertEqual(row.digest(), row_digest)

    def test_parallel_domain_cache_identities_match_serial_generation(self) -> None:
        """Parallel fit-cache preparation must preserve every cache key input."""

        exemplars = tuple(
            observation(
                source_format=source_format,
                shape_group=f"shape-{index}",
            )
            for index, source_format in enumerate(
                ("Q4_0", "Q5_0", "Q8_0", "IQ4_NL")
            )
        )
        corpus = ObservationCorpus(tuple(
            row for exemplar in exemplars for row in (exemplar,) * 2048
        )).with_collapsed_aspect_domains()
        serial_hashes = {
            corpus.runtime_key_for(row): SERIAL_HASH for row in exemplars
        }

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            serial = segmented_policy._domain_cache_identity_inputs(
                corpus,
                corpus.generic_domains(),
                None,
                {},
                serial_hashes,
            )
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "4"},
        ):
            parallel = segmented_policy._domain_cache_identity_inputs(
                corpus,
                corpus.generic_domains(),
                None,
                {},
                serial_hashes,
            )

        self.assertEqual(parallel, serial)

    def test_parallel_candidate_cost_cache_loads_match_serial_generation(
        self,
    ) -> None:
        """Publication cache parsing must preserve every typed regret row."""

        corpus = ObservationCorpus(tuple(
            observation(
                source_format=source_format,
                shape_group=f"cost-cache-{index}",
            )
            for index, source_format in enumerate(
                ("Q4_0", "Q5_0", "Q8_0", "IQ4_NL")
            )
        )).with_collapsed_aspect_domains()
        costs = build_candidate_point_costs(corpus)
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            tasks = []
            for index, domain in enumerate(corpus.generic_domains()):
                content_key = f"unit-cost-key-{index}"
                cache.store_costs(content_key, domain, costs[domain])
                tasks.append((domain, content_key, ()))

            with mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
            ):
                serial = segmented_policy._load_cached_costs_parallel(
                    cache, tasks
                )
            with mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "4"},
            ):
                parallel = segmented_policy._load_cached_costs_parallel(
                    cache, tasks
                )

        self.assertEqual(parallel, serial)

    def test_parallel_candidate_cost_builds_match_serial_generation(self) -> None:
        """Independent domain builders preserve canonical regret row order."""

        exemplars = tuple(
            observation(
                source_format=source_format,
                shape_group=f"parallel-cost-{index}",
            )
            for index, source_format in enumerate(
                ("Q4_0", "Q5_0", "Q8_0", "IQ4_NL")
            )
        )
        corpus = ObservationCorpus(tuple(
            row for exemplar in exemplars for row in (exemplar,) * 1024
        )).with_collapsed_aspect_domains()
        domains = corpus.generic_domains()
        domain_corpora = {domain: corpus for domain in domains}

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            serial = segmented_policy._build_domain_candidate_costs_parallel(
                domain_corpora,
                domains,
                serial_m1_hashes=None,
                paired_comparisons=None,
                supplemental_costs={},
            )
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "4"},
        ):
            parallel = segmented_policy._build_domain_candidate_costs_parallel(
                domain_corpora,
                domains,
                serial_m1_hashes=None,
                paired_comparisons=None,
                supplemental_costs={},
            )

        self.assertEqual(parallel, serial)

    def test_parallel_profiler_prediction_requests_match_serial_generation(
        self,
    ) -> None:
        """Independent fold inventories preserve every prediction point."""

        corpus = ObservationCorpus(tuple(
            observation(
                source_format=source_format,
                shape_group=f"prediction-request-{format_index}-{shape_index}",
                n=256 * (shape_index + 1),
            )
            for format_index, source_format in enumerate(
                ("Q4_0", "Q5_0", "Q8_0", "IQ4_NL")
            )
            for shape_index in range(5)
        )).with_collapsed_aspect_domains()
        costs = build_candidate_point_costs(corpus)
        domains = corpus.generic_domains()

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            serial = (
                segmented_policy
                ._build_domain_profiler_prediction_requests_parallel(
                    domains,
                    costs,
                    seed="parallel-profiler-request-test",
                )
            )
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "4"},
        ):
            parallel = (
                segmented_policy
                ._build_domain_profiler_prediction_requests_parallel(
                    domains,
                    costs,
                    seed="parallel-profiler-request-test",
                )
            )

        self.assertEqual(parallel, serial)

    def test_p95_uses_exact_nearest_rank_for_every_accelerated_population(
        self,
    ) -> None:
        """Host and device searches must select the same observed p95 row."""

        for population_size in range(1, 65):
            values = tuple(float(index) for index in range(population_size))
            rank = (95 * population_size + 99) // 100
            with self.subTest(population_size=population_size):
                self.assertEqual(
                    segmented_policy._percentile(values, 0.95),
                    values[rank - 1],
                )

    def test_accelerator_scheduler_refills_the_first_free_device_lane(self) -> None:
        """A fast lane must consume tail tasks while another lane is busy."""

        specs = (_SchedulerProbeSpec("slow"), _SchedulerProbeSpec("fast"))
        task_timings = []
        with mock.patch.object(
            segmented_policy,
            "_accelerated_worker",
            _scheduler_probe_worker,
        ):
            results = segmented_policy._run_accelerated_tasks(
                specs,
                (100, 90, 80, 70),
                "cv",
                task_timings,
            )

        self.assertEqual(
            results,
            [
                ("slow", 0),
                ("fast", 1),
                ("fast", 2),
                ("fast", 3),
            ],
        )
        self.assertEqual(
            {timing.task_index for timing in task_timings},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {timing.accelerator_label for timing in task_timings},
            {"slow", "fast"},
        )
        self.assertTrue(all(
            timing.elapsed_seconds > 0.0 for timing in task_timings
        ))

    def test_accelerated_cv_groups_reuse_one_geometry_across_influences(
        self,
    ) -> None:
        """Keep profiler variants together without changing original indices."""

        domain = object()
        heldout = frozenset(("held-a", "held-b"))
        tasks = []
        for feature_policy in (
            segmented_policy.FeaturePolicy.CONTINUOUS,
            segmented_policy.FeaturePolicy.TILE_64,
        ):
            for influence in segmented_policy.TREE_PROFILER_INFLUENCES:
                tasks.append((
                    domain,
                    2,
                    heldout,
                    feature_policy,
                    segmented_policy.BoundaryPlacement.MIDPOINT,
                    influence,
                    (f"costs-{influence.value}",),
                    16,
                    2,
                ))

        groups = segmented_policy._group_accelerated_cv_tasks(tasks)

        self.assertEqual(len(groups), 2)
        self.assertTrue(all(
            len(group) == len(segmented_policy.TREE_PROFILER_INFLUENCES)
            for group in groups
        ))
        self.assertEqual(
            tuple(index for group in groups for index, _task in group),
            tuple(range(len(tasks))),
        )
        self.assertTrue(all(
            len({task[3] for _index, task in group}) == 1
            for group in groups
        ))

    def test_accelerated_threshold_scalar_builder_is_exact(self) -> None:
        """Allocation-free pair arithmetic must preserve rational ABI fields."""

        cases = (
            (
                segmented_policy.FeatureAxis.AGGREGATE_N,
                Fraction(17, 3),
                Fraction(29, 4),
            ),
            (
                segmented_policy.FeatureAxis.N_PARALLEL_WAVES_64,
                Fraction(2, 1),
                Fraction(5, 1),
            ),
            (
                segmented_policy.FeatureAxis.MN_FINAL_PARALLEL_WAVE_UTILIZATION_64,
                Fraction(3, 28),
                Fraction(11, 28),
            ),
        )
        for axis, lower, upper in cases:
            for placement in segmented_policy.BOUNDARY_PLACEMENTS:
                with self.subTest(axis=axis, placement=placement):
                    expected = segmented_policy._threshold_between(
                        axis,
                        lower,
                        upper,
                        placement,
                        28,
                        15,
                    )
                    actual = (
                        segmented_policy._accelerated_threshold_components_between(
                            axis,
                            lower,
                            upper,
                            placement,
                            28,
                            15,
                        )
                    )
                    self.assertEqual(
                        actual,
                        (
                            expected.numerator,
                            expected.denominator,
                            expected.parallelism_width,
                            expected.task_multiplier,
                        ),
                    )

    def test_accelerated_axis_policy_is_compact_and_device_generated(self) -> None:
        """The host must publish O(axis) policy, never O(axis*point^2) tables."""

        axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            segmented_policy.FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY
        ]
        descriptors = tuple(
            segmented_policy._accelerated_feature_axis_descriptor(axis)
            for axis in axes
        )
        self.assertEqual(len(descriptors), len(axes))
        self.assertEqual(
            len({descriptor.axis_priority for descriptor in descriptors}),
            len(axes),
        )
        self.assertFalse(
            hasattr(segmented_policy, "_accelerated_tree_feature_metadata")
        )

    def test_cached_cuda_nearest_factor_matches_brute_force_resolver(self) -> None:
        """Faster formula projection must preserve every exact KB decision."""

        def brute_force(
            *,
            grid_n,
            k_groups,
            target_blocks,
            min_kgroups_per_cta,
            max_kb,
        ):
            kb = max(2, (target_blocks + grid_n - 1) // grid_n)
            kb_max = max(2, k_groups // min_kgroups_per_cta)
            kb = min(kb, kb_max)
            if k_groups % kb:
                lower = next((
                    kb - distance
                    for distance in range(1, kb)
                    if kb - distance >= 2
                    and k_groups % (kb - distance) == 0
                ), -1)
                upper = next((
                    kb + distance
                    for distance in range(1, kb_max - kb + 1)
                    if k_groups % (kb + distance) == 0
                ), -1)
                if lower > 0 and upper > 0:
                    lower_distance = abs(grid_n * lower - target_blocks)
                    upper_distance = abs(grid_n * upper - target_blocks)
                    kb = upper if upper_distance < lower_distance else lower
                elif lower > 0:
                    kb = lower
                elif upper > 0:
                    kb = upper
            return min(max_kb, k_groups, max(1, kb))

        for grid_n in (1, 2, 3, 7, 16, 65):
            for k_groups in (
                1, 2, 3, 17, 31, 32, 33, 64, 83, 128, 152, 160, 161,
                256, 512, 1792,
            ):
                for target_blocks in range(1, 161):
                    for minimum in (1, 2, 4, 8):
                        arguments = {
                            "grid_n": grid_n,
                            "k_groups": k_groups,
                            "target_blocks": target_blocks,
                            "min_kgroups_per_cta": minimum,
                            "max_kb": 256,
                        }
                        self.assertEqual(
                            _nearest_factor_kb(**arguments),
                            brute_force(**arguments),
                            arguments,
                        )

    def test_strict_csv_rows_are_validated_once_before_corpus_indexing(self) -> None:
        """Immutable ownership transfer must avoid duplicate schema validation."""

        row = observation()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.csv"
            write_observation_csv(path, (row,))
            original_validate = NativeVNNIObservation.validate
            validated = []

            def counted_validate(instance) -> None:
                validated.append(instance.candidate_id)
                original_validate(instance)

            with mock.patch.object(
                NativeVNNIObservation,
                "validate",
                counted_validate,
            ):
                corpus = read_observation_csv((path,))

        self.assertEqual(len(corpus), 1)
        self.assertEqual(validated, [row.candidate_id])

    def test_observation_digest_is_memoized_without_changing_canonical_bytes(self) -> None:
        """Repeated profiler joins must reuse the exact historical row digest."""

        row = observation()
        canonical_before = row.canonical_mapping()
        with mock.patch(
            "native_vnni_dispatch.schema.hashlib.sha256",
            wraps=hashlib.sha256,
        ) as sha256:
            first = row.digest()
            second = row.digest()

        self.assertEqual(first, second)
        self.assertEqual(sha256.call_count, 1)
        self.assertEqual(row.canonical_mapping(), canonical_before)
        self.assertNotIn("_cached_digest", row.canonical_mapping())

    def test_parallel_observation_writer_is_byte_identical_to_serial(self) -> None:
        """Parallel checkpoint emission must preserve canonical row ordering."""

        rows = tuple(
            observation(
                candidate=f"candidate.{index}",
                family=f"family-{index % 2}",
                shape_group=f"shape-{index}",
                n=128 + index * 16,
                k=1024 + index * 32,
                m=1 + index,
            )
            for index in range(7)
        )
        with tempfile.TemporaryDirectory() as temporary:
            serial_path = Path(temporary) / "serial.csv"
            parallel_path = Path(temporary) / "parallel.csv"
            write_observation_csv(
                serial_path,
                rows,
                workers=1,
                parallel_threshold=1,
            )
            write_observation_csv(
                parallel_path,
                rows,
                workers=3,
                parallel_threshold=1,
            )
            self.assertEqual(
                parallel_path.read_bytes(),
                serial_path.read_bytes(),
            )
            self.assertEqual(
                tuple(read_observation_csv((parallel_path,))),
                rows,
            )

    def test_validated_csv_handoff_still_rejects_candidate_identity_drift(self) -> None:
        """Skipping repeat row checks cannot skip cross-row identity checks."""

        first = observation()
        changed = dataclasses.replace(first, candidate_family="changed-family")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity-drift.csv"
            write_observation_csv(path, (first, changed))
            with self.assertRaisesRegex(
                ValueError, "candidate identity changed"
            ):
                read_observation_csv((path,))

    def test_cv_folds_withhold_geometry_aliases_in_coherent_regions(self) -> None:
        """CV must not train on an alias or neighbor of held-out geometry."""

        exemplar = observation(
            backend=Backend.CUDA,
            contract=SemanticContract.FAST,
            mode=ExecutionMode.EAGER,
            m=1,
            n=128,
            k=32,
        )
        domain = generic_domain(exemplar)
        exemplar_key = runtime_key(exemplar)
        costs = []
        groups_by_geometry = {}
        for k in (32, 4096):
            for point_index in range(10):
                n = 128 + point_index * 32
                geometry = (n, k)
                groups = (
                    f"shape-{n}x{k}-primary",
                    f"shape-{n}x{k}-alias",
                )
                groups_by_geometry[geometry] = frozenset(groups)
                key = dataclasses.replace(
                    exemplar_key,
                    projection_n_vector=(n,),
                    aggregate_n=n,
                    k=k,
                )
                costs.extend(
                    segmented_policy.CandidatePointCost(
                        runtime_key=key,
                        shape_group_id=group,
                        candidate_id="candidate.a",
                        max_surface_regret=0.0,
                        p95_surface_regret=0.0,
                        mean_surface_regret=0.0,
                    )
                    for group in groups
                )

        folds = segmented_policy._domain_folds(
            domain,
            costs,
            seed="unit-coherent-regions",
        )
        self.assertEqual(len(folds), 5)
        all_groups = set().union(*folds)
        self.assertEqual(len(all_groups), sum(map(len, folds)))
        self.assertEqual(
            all_groups,
            {
                group
                for groups in groups_by_geometry.values()
                for group in groups
            },
        )
        fold_by_group = {
            group: fold_index
            for fold_index, fold in enumerate(folds)
            for group in fold
        }
        for groups in groups_by_geometry.values():
            self.assertEqual(
                {fold_by_group[group] for group in groups},
                {fold_by_group[next(iter(groups))]},
            )

        k_by_group = {
            group: k
            for (_n, k), groups in groups_by_geometry.items()
            for group in groups
        }
        self.assertTrue(all(
            len({k_by_group[group] for group in fold}) == 1
            for fold in folds
        ))

    def test_leaf_primary_pruning_is_exact_and_skips_dominated_percentiles(
        self,
    ) -> None:
        """Dominated candidates cannot reach later lexicographic score keys."""

        candidates = {
            "candidate.a": (0.10, 0.02, 0.00, 0.00, 0.00),
            "candidate.b": (0.10, 0.01, 0.01, 0.00, 0.00),
            "candidate.dominated": (0.20, 0.00, 0.00, 0.00, 0.00),
        }
        costs = []
        for point_index in range(5):
            n = 128 + point_index * 32
            key = RuntimeKey(
                backend=Backend.CUDA,
                architecture_class="unit-sm",
                semantic_contract=SemanticContract.FAST,
                operation_kind="NativeVNNIDecodeProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=1,
                aggregate_n=n,
                k=256,
            )
            for candidate, regrets in candidates.items():
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"shape-{point_index}",
                    candidate_id=candidate,
                    max_surface_regret=regrets[point_index],
                    p95_surface_regret=regrets[point_index],
                    mean_surface_regret=regrets[point_index],
                ))

        matrix = segmented_policy._point_matrix(costs)
        points = tuple(sorted(
            matrix,
            key=lambda point: (
                point[0].aggregate_n,
                point[0].k,
                point[1],
            ),
        ))
        exhaustive = []
        for candidate in sorted(candidates):
            rows = [matrix[point][candidate] for point in points]
            regrets = tuple(row.max_surface_regret for row in rows)
            measured_p95 = segmented_policy._percentile(
                (row.p95_surface_regret for row in rows), 0.95
            )
            exhaustive.append((
                int(measured_p95 >= P95_REGRET_BUDGET),
                segmented_policy._percentile(regrets, 0.95),
                statistics.fmean(regrets),
                measured_p95,
                statistics.fmean(row.mean_surface_regret for row in rows),
                sum(
                    row.max_surface_regret >= P95_REGRET_BUDGET
                    for row in rows
                ),
                max(row.max_surface_regret for row in rows),
                candidate,
                regrets,
            ))
        expected = min(exhaustive, key=lambda item: item[:8])

        with mock.patch.object(
            segmented_policy,
            "_percentile",
            wraps=segmented_policy._percentile,
        ) as percentile:
            fitted = segmented_policy._best_leaf(points, matrix)

        self.assertIsNotNone(fitted)
        self.assertEqual(fitted.root.candidate_id, expected[7])
        self.assertEqual(fitted.regrets, expected[8])
        # Every candidate needs the two exact primary percentiles. Only the two
        # primary survivors reach the complete measured diagnostic score.
        self.assertEqual(percentile.call_count, 10)

    def test_batched_primary_scorer_preserves_complete_tree_sequence(self) -> None:
        """GPU plumbing may batch primary keys but cannot change fitted trees."""

        costs = []
        for point_index, n in enumerate(
            (128, 160, 192, 224, 288, 320, 352, 384)
        ):
            key = RuntimeKey(
                backend=Backend.CUDA,
                architecture_class="unit-sm",
                semantic_contract=SemanticContract.FAST,
                operation_kind="NativeVNNIDecodeProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=1,
                aggregate_n=n,
                k=2048,
            )
            lower = point_index < 4
            for candidate, regret in (
                ("candidate.low", 0.0 if lower else 0.40),
                ("candidate.high", 0.40 if lower else 0.0),
                ("candidate.middle", 0.12),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"accelerated-shape-{point_index}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        expected = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
        )
        matrix = segmented_policy._point_matrix(costs)
        ordered_points = tuple(sorted(
            matrix,
            key=lambda point: (
                point[0].aggregate_n,
                point[0].k,
                point[1],
            ),
        ))
        candidates = tuple(sorted({
            cost.candidate_id for cost in costs
        }))

        def encode(fit):
            leaf_masks = []
            candidate_indices = []
            structure = []
            split_thresholds = []

            def visit(node):
                if isinstance(node, segmented_policy._LeafNode):
                    structure.append(0)
                    split_thresholds.append(None)
                    leaf_masks.append(node.point_mask)
                    candidate_indices.append(candidates.index(node.candidate_id))
                    return
                structure.append(1)
                split_thresholds.append(
                    segmented_policy._accelerated_threshold_descriptor(
                        node.threshold
                    )
                )
                visit(node.left)
                visit(node.right)

            visit(fit.root)
            return segmented_policy.AcceleratedTreeFitResult(
                tuple(leaf_masks),
                tuple(candidate_indices),
                tuple(structure),
                tuple(split_thresholds),
            )

        class ExactFakeTreeScorer:
            """Publish the CPU oracle through the compact device result ABI."""

            def __init__(self) -> None:
                self.calls = 0

            def fit_tree_budgets(self, *args, **kwargs):
                self.calls += 1
                self.point_count = kwargs["point_count"]
                self.feature_axes = kwargs["feature_axes"]
                return tuple(encode(fit) for fit in expected)

        scorer = ExactFakeTreeScorer()
        actual = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
            primary_scorer=scorer,
        )

        self.assertEqual(actual, expected)
        self.assertEqual(scorer.calls, 1)
        self.assertEqual(scorer.point_count, len(ordered_points))
        self.assertEqual(
            len(scorer.feature_axes),
            len(segmented_policy.FEATURE_AXES_BY_POLICY[
                segmented_policy.FeaturePolicy.CONTINUOUS
            ]),
        )

        def leaf_nodes(root):
            if isinstance(root, segmented_policy._LeafNode):
                return (root,)
            return (*leaf_nodes(root.left), *leaf_nodes(root.right))

        for fit in actual:
            masks = 0
            for leaf in leaf_nodes(fit.root):
                self.assertEqual(masks & leaf.point_mask, 0)
                masks |= leaf.point_mask
                self.assertEqual(
                    leaf.points,
                    tuple(
                        point
                        for index, point in enumerate(ordered_points)
                        if leaf.point_mask & (1 << index)
                    ),
                )
            self.assertEqual(masks, (1 << len(ordered_points)) - 1)

    def test_grouped_cv_uses_fused_accelerator_evaluation_contract(self) -> None:
        """Accelerated CV must consume compact decisions, never temporary trees."""

        heldout_groups = frozenset(("fused-unit-2", "fused-unit-7"))
        costs = []
        for point_index in range(10):
            n = 128 + point_index * 64
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit|threads=28",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-fused-contract",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=1024,
            )
            for candidate, regret in (
                ("candidate.alpha", 0.01),
                ("candidate.beta", 0.04),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"fused-unit-{point_index}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class="unit|threads=28",
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature="unit-fused-contract",
            prepared_family_id="unit-prepared",
            packing_abi="unit-packing",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        arguments = (
            domain,
            0,
            heldout_groups,
            FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY,
            BoundaryPlacement.MIDPOINT,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
            costs,
            3,
            2,
        )
        expected = segmented_policy._evaluate_placement_fold(arguments)

        class FusedOnlyScorer:
            """Expose only the compact ABI so tree-download use fails loudly."""

            def __init__(self) -> None:
                self.calls = 0

            def fit_tree_budgets_and_evaluate(self, *args, **kwargs):
                self.calls += 1
                heldout_count = len(kwargs["heldout_aggregate_n"])
                self.feature_axes = kwargs["feature_axes"]
                return tuple(
                    segmented_policy.AcceleratedFoldEvaluation(
                        selected_candidate_indices=(0,) * heldout_count,
                        exact_candidate_indices=(0,) * heldout_count,
                        covered_point_count=heldout_count,
                        required_point_count=heldout_count,
                    )
                    for _ in range(kwargs["max_leaves"])
                )

        scorer = FusedOnlyScorer()
        actual = segmented_policy._evaluate_placement_fold(arguments, scorer)

        self.assertEqual(actual, expected)
        self.assertEqual(scorer.calls, 1)
        self.assertTrue(scorer.feature_axes)
        self.assertTrue(all(
            isinstance(
                descriptor.operation,
                segmented_policy.TreeThresholdOperation,
            )
            for descriptor in scorer.feature_axes
        ))

    def test_full_launch_geometry_policies_are_monotonic(self) -> None:
        """Row-grid features extend rather than replace the proven N model."""

        n_only_axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.FULL_LAUNCH_GEOMETRY
        ]
        row_grid_axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY
        ]
        row_grid_only_axes = (
            set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
        )

        self.assertEqual(tuple(dict.fromkeys(n_only_axes)), n_only_axes)
        self.assertEqual(tuple(dict.fromkeys(row_grid_axes)), row_grid_axes)
        self.assertEqual(set(n_only_axes), set(FeatureAxis) - row_grid_only_axes)
        self.assertEqual(set(row_grid_axes), set(FeatureAxis))
        self.assertLessEqual(
            len(row_grid_axes),
            segmented_policy.TREE_MAXIMUM_FEATURE_AXES,
        )

    def test_kpart_chunk_grid_policy_contains_only_physical_schedule_axes(
        self,
    ) -> None:
        """Serial-K-part routing must model the five forceable N task grids."""

        axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.KPART_CHUNK_GRID_SCHEDULES
        ]
        expected = (
            *segmented_policy.BASE_FEATURE_AXES,
            *(axis for axis, width in
              segmented_policy.N_TILE_WIDTH_BY_AXIS.items() if width >= 64),
            *(axis for axis, width in
              segmented_policy.K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS.items()
              if width >= 64),
            *(axis for axis, width in
              segmented_policy.N_FINAL_TILE_WIDTH_BY_AXIS.items()
              if width >= 64),
            *(axis for axis, width in
              segmented_policy.K_FINAL_TILE_WIDTH_BY_AXIS.items()
              if width >= 64),
            *(axis for axis, width in
              segmented_policy.N_TILE_UTILIZATION_WIDTH_BY_AXIS.items()
              if width >= 64),
            *(axis for axis, width in
              segmented_policy.N_TILE_ALIGNED_WIDTH_BY_AXIS.items()
              if width >= 64),
            *segmented_policy.N_PARALLEL_WAVE_WIDTH_BY_AXIS,
            *segmented_policy.N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
        )

        self.assertEqual(axes, expected)
        self.assertEqual(tuple(dict.fromkeys(axes)), axes)
        self.assertFalse(
            set(axes)
            & (
                set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
                | set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
            )
        )
        self.assertNotIn(FeatureAxis.N_TILES_32, axes)
        self.assertLessEqual(len(axes), 64)

    def test_kpart_producer_policy_contains_partition_span_and_tail(self) -> None:
        """Producer-grid fitting must see both task count and K work span."""

        axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.KPART_PRODUCER_GRID_SCHEDULES
        ]
        self.assertEqual(tuple(dict.fromkeys(axes)), axes)
        self.assertTrue(
            set(segmented_policy.KPART_PARTITION_GEOMETRY_AXES).issubset(axes)
        )
        self.assertTrue(
            set(segmented_policy.KPART_PRODUCER_WAVE_WIDTH_BY_AXIS).issubset(
                axes
            )
        )
        self.assertTrue(
            set(
                segmented_policy.KPART_FINAL_PRODUCER_WAVE_WIDTH_BY_AXIS
            ).issubset(axes)
        )

    def test_kpart_complete_policy_combines_both_physical_schedule_grids(
        self,
    ) -> None:
        """One tree must express joint N-chunk and K-partition rollovers."""

        chunk_axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.KPART_CHUNK_GRID_SCHEDULES
        ]
        producer_axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.KPART_PRODUCER_GRID_SCHEDULES
        ]
        complete_axes = segmented_policy.FEATURE_AXES_BY_POLICY[
            FeaturePolicy.KPART_COMPLETE_SCHEDULES
        ]
        expected = tuple(dict.fromkeys((*chunk_axes, *producer_axes)))

        self.assertEqual(complete_axes, expected)
        self.assertLess(set(chunk_axes), set(complete_axes))
        self.assertLess(set(producer_axes), set(complete_axes))
        self.assertNotIn(FeatureAxis.N_TILES_32, complete_axes)
        self.assertFalse(
            set(complete_axes)
            & (
                set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
                | set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
            )
        )
        self.assertLessEqual(
            len(complete_axes),
            segmented_policy.TREE_MAXIMUM_FEATURE_AXES,
        )

    def test_tree_beam_prioritizes_segment_p95_over_failure_count(self) -> None:
        """The same percentile used by installation must own beam ranking."""

        costs = []
        for point_index, n in enumerate((128, 256, 384, 512, 640)):
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx2",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=2048,
            )
            for candidate, regret in (
                ("candidate.one-outlier", 0.40 if point_index == 0 else 0.0),
                ("candidate.four-failures", 0.06 if point_index < 4 else 0.0),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"multi-cliff-{point_index}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        fit = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=1,
            min_shape_groups_per_leaf=2,
        )[0]

        self.assertIsNotNone(fit)
        self.assertEqual(fit.root.candidate_id, "candidate.four-failures")
        self.assertEqual(fit.primary_objective, (1, 0.06))

    def test_measured_leaf_p95_gate_precedes_maximum_surface_regret(self) -> None:
        """A p95-safe candidate cannot lose to a low-maximum failing leaf.

        This is the natural shape of the round-9 final-fit defect. The safe
        candidate has one expensive source alias at every geometry but remains
        below budget at p95. The other candidate has a lower maximum yet misses
        the installation p95 at every geometry. Maximum-first fitting selected
        the latter and only discovered the rejection after final rule emission.
        """

        costs = []
        for point_index, n in enumerate((128, 160, 192, 224, 256, 288)):
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx2",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=6,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=512,
            )
            for candidate, maximum, p95, mean in (
                ("candidate.p95-safe", 0.10, 0.01, 0.02),
                ("candidate.low-max-failing", 0.06, 0.06, 0.06),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"surface-p95-{point_index}",
                    candidate_id=candidate,
                    max_surface_regret=maximum,
                    p95_surface_regret=p95,
                    mean_surface_regret=mean,
                ))

        fit = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=1,
            min_shape_groups_per_leaf=2,
        )[0]

        self.assertIsNotNone(fit)
        self.assertEqual(fit.root.candidate_id, "candidate.p95-safe")
        self.assertEqual(fit.root.p95_regret, 0.01)
        self.assertEqual(fit.root.measured_max_regret, 0.10)
        self.assertEqual(fit.primary_objective[0], 0)

    def test_tree_prefers_every_p95_safe_leaf_over_lower_global_max(self) -> None:
        """Final-fit search must optimize the gate applied to emitted leaves.

        A cut after three shapes produces a superficially attractive tree with
        one 4% miss and a lower global maximum, but that three-shape leaf fails
        p95. A balanced cut leaves one 10% diagnostic outlier among twenty
        shapes and therefore passes nearest-rank p95 in both leaves. The former
        count-then-maximum objective selected the rejected two-shape tree.
        """

        costs = []
        for point_index in range(40):
            n = 128 + point_index * 32
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx2",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=6,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=512,
            )
            candidate_regrets = {
                "candidate.small-left": (
                    0.06 if point_index == 0
                    else 0.0 if point_index < 3
                    else 0.50
                ),
                "candidate.large-right": (
                    0.06 if point_index < 3 else 0.0
                ),
                "candidate.balanced-left": (
                    0.10 if point_index == 0
                    else 0.0 if point_index < 20
                    else 0.50
                ),
            }
            for candidate, regret in candidate_regrets.items():
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"leaf-gate-{point_index:02d}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        fit = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=2,
            min_shape_groups_per_leaf=2,
        )[-1]

        self.assertIsNotNone(fit)
        self.assertEqual(fit.leaf_count, 2)
        self.assertEqual(fit.root.failed_leaf_count, 0)
        self.assertLess(fit.root.worst_leaf_p95_regret, P95_REGRET_BUDGET)
        self.assertIsInstance(fit.root, segmented_policy._SplitNode)
        self.assertTrue(fit.root.threshold.matches_less_equal(736, 512))
        self.assertFalse(fit.root.threshold.matches_less_equal(768, 512))

    def test_leaf_capacity_experiment_does_not_change_production_default(self) -> None:
        """Permit a reviewed 32-leaf study while defaulting production to 16."""

        self.assertEqual(segmented_policy.DEFAULT_TREE_LEAVES, 16)
        self.assertEqual(segmented_policy.MAX_TREE_LEAVES, 32)
        self.assertEqual(
            len(segmented_policy._fit_tree_budgets(
                [],
                max_leaves=32,
                min_shape_groups_per_leaf=2,
            )),
            32,
        )
        with self.assertRaisesRegex(ValueError, "32 leaves"):
            segmented_policy._fit_tree_budgets(
                [],
                max_leaves=33,
                min_shape_groups_per_leaf=2,
            )

    """Prove numerical filtering, robust regret, and sealed policy behavior."""

    def test_registry_matches_full_quantized_inventory_and_q8_alias_semantics(self) -> None:
        self.assertEqual(len(FORMAT_SPECS), 21)
        self.assertEqual(format_spec("Q8_K").source_codebook_id, 21)
        self.assertEqual(format_spec("Q8_K").runtime_codebook("rocm"), 19)
        self.assertEqual(format_spec("Q8_1").runtime_codebook("cuda"), 19)
        self.assertEqual(format_spec("Q8_1").runtime_codebook("cpu"), 20)
        self.assertEqual(runtime_aliases("rocm", 19), ("Q8_0", "Q8_1", "Q8_K"))
        self.assertEqual(registry_digest(), registry_digest())

    def test_candidate_registries_are_explicit_complete_and_digestible(self) -> None:
        rocm_moe = rocm_moe_grouped_prefill_registry()
        rocm_decode = rocm_native_vnni_decode_registry()
        cpu_decode = cpu_native_vnni_decode_registry()
        cpu = cpu_native_vnni_verifier_registry()
        cpu_prefill = cpu_native_vnni_prefill_registry()
        cuda = cuda_native_vnni_gemv_registry()

        self.assertEqual(len(rocm_moe.entries), 12)
        self.assertEqual(len(rocm_decode.entries), 65)
        self.assertEqual(len(cpu_decode.entries), 5)
        self.assertEqual(len(cpu.entries), 9)
        self.assertEqual(
            {entry.config_json["policy"] for entry in cpu.entries},
            {
                "Pairwise",
                "WideRows",
                "FullKRowChunkGrid",
                "FullKTwoRowNbc1",
                "FullKTwoRowNbc2",
                "FullKTwoRowPairGridNbc1",
                "FullKTwoRowPairGridNbc2",
                "FullKTwoRowPairGridNbc4",
                "FullKTwoRowPairGridNbc8",
            },
        )
        self.assertEqual(
            sum(
                entry.config_json.get("route") == "two_row_pair_grid"
                for entry in cpu.entries
            ),
            4,
        )
        self.assertEqual(len(cpu_prefill.entries), 12)
        self.assertEqual(
            cpu_prefill.entries[0].candidate_id,
            "cpu.nvnni.prefill.row_chunk_grid.full_k",
        )
        self.assertEqual(
            {entry.config_json["k_tile_policy"] for entry in cpu_prefill.entries},
            {"full_k", "inherit_serial_m1"},
        )
        self.assertEqual(
            sum(
                entry.config_json.get("route") == "two_row_pair_grid"
                for entry in cpu_prefill.entries
            ),
            5,
        )
        self.assertEqual(
            {
                entry.config_json["n_block_chunks"]
                for entry in cpu_prefill.entries
                if entry.config_json.get("route")
                == "two_row_full_output_tiles"
            },
            {1, 2, 4, 8},
        )
        self.assertEqual(len(cuda.entries), 4495 + 64 + 32 + 2 * (40 * 4 * 2 + 64))
        self.assertEqual(
            {
                (entry.config_json["tile_n"], entry.config_json["exact_kb"])
                for entry in cuda.entries
                if entry.config_json.get("family") == "fused_kpar"
            },
            {(columns, kb) for columns in (16, 32) for kb in range(1, 1024 // columns + 1)},
        )
        self.assertEqual(
            {
                entry.config_json["n_block_chunks"]
                for entry in cpu_decode.entries
            },
            {1, 2, 4, 8, 16},
        )
        self.assertEqual(
            {
                entry.config_json["k_partition_policy"]
                for entry in cpu_decode.entries
            },
            {"frozen_serial_m1"},
        )
        self.assertTrue(all(
            entry.supports_contract(SemanticContract.FAST)
            for entry in cpu_decode.entries
        ))
        self.assertEqual(
            sum(
                entry.config_json.get("formula_kind")
                == "canonical_target_blocks"
                for entry in cuda.entries
            ),
            9 * 40 * 4,
        )
        self.assertTrue(candidate_registry_digest().startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "not an explicit candidate"):
            rocm_decode.resolve("AUTO")

        rocm_fast = rocm_decode.resolve("KB32")
        self.assertEqual(rocm_fast, rocm_decode.resolve("KB32/TW24"))
        self.assertTrue(rocm_fast.supports_contract(SemanticContract.FAST))
        self.assertFalse(rocm_fast.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        rocm_verifier = rocm_decode.resolve("INHERIT_SERIAL_M1")
        self.assertFalse(rocm_verifier.supports_contract(SemanticContract.FAST))
        self.assertTrue(rocm_verifier.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))

        public_m1 = cuda.resolve(
            "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        )
        self.assertTrue(public_m1.supports_contract(SemanticContract.FAST))
        self.assertFalse(public_m1.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        self.assertEqual(public_m1.config_json["target_waves"], 0)
        self.assertEqual(public_m1.config_json["exact_kb"], 8)
        self.assertEqual(public_m1.config_json["force_two_phase"], 1)
        self.assertTrue(public_m1.ordered_reduction)
        self.assertFalse(public_m1.uses_atomic_reduction)

        cuda_verifier = cuda.resolve(
            "cuda.nvnni.decode.verifier.inherit_serial_m1.r2"
        )
        self.assertFalse(cuda_verifier.supports_contract(SemanticContract.FAST))
        self.assertTrue(cuda_verifier.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        self.assertEqual(cuda_verifier.config_json["grouped_rows"], 2)
        self.assertTrue(cuda_verifier.ordered_reduction)
        self.assertFalse(cuda_verifier.uses_atomic_reduction)
        with self.assertRaisesRegex(ValueError, "unknown forceable candidate"):
            cuda.resolve("cuda.nvnni.kpar.tn256.cpt4.tw8.mkg4.kb4.phase2")

        occupancy = cuda.resolve(
            "cuda.nvnni.decode.fast_m1.kpar_formula."
            "tn128.cpt1.tb328.mkg1"
        )
        canonical_occupancy = cuda.resolve(
            "cuda.nvnni.decode.fast_m1.kpar_formula."
            "tn128.cpt1.ctb328.mkg1"
        )
        groups_per_partition = cuda.resolve(
            "cuda.nvnni.decode.fast_m1.kpar_formula."
            "tn128.cpt1.bpp4"
        )
        self.assertEqual(resolve_cuda_formula_kb(occupancy, 192, 256), 8)
        # K/32=83 is prime. The legacy even-split formula jumps to KB83,
        # while the canonical uneven-width candidate retains the desired
        # five K groups per partition and resolves to economical KB17.
        self.assertEqual(resolve_cuda_formula_kb(occupancy, 2400, 2656), 83)
        self.assertEqual(
            resolve_cuda_formula_kb(canonical_occupancy, 2400, 2656),
            17,
        )
        self.assertEqual(
            resolve_cuda_formula_kb(groups_per_partition, 192, 256),
            2,
        )

    def test_shape_resolved_cuda_evidence_remains_concrete_for_exact_policy(self) -> None:
        """Formula evidence may train generic rules but cannot replace exact routes."""

        concrete_id = "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb8"
        concrete = dataclasses.replace(
            observation(
                candidate=concrete_id,
                family="cuda_public_m1_kpar",
                n=192,
                k=256,
                m=1,
                mode=ExecutionMode.GRAPH_CAPTURED,
                contract=SemanticContract.FAST,
            ),
            backend=Backend.CUDA,
            prepared_family_id=format_spec("Q4_0").prepared_family("cuda"),
            packing_abi=format_spec("Q4_0").packing_abi("cuda"),
            runtime_codebook_id=format_spec("Q4_0").runtime_codebook("cuda"),
            observed_candidate_id=concrete_id,
            config_json={
                "family": "kpar",
                "tile_n": 128,
                "cpt": 1,
                "exact_kb": 8,
            },
        )
        projected = project_cuda_shape_resolved_candidates(
            ObservationCorpus((concrete,))
        )
        for projected_row in projected:
            projected_row.validate()
        formula_id = (
            "cuda.nvnni.decode.fast_m1.kpar_formula."
            "tn128.cpt1.tb328.mkg1"
        )
        formulas = [row for row in projected if row.candidate_id == formula_id]

        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0].effective_candidate_id, concrete_id)
        self.assertEqual(formulas[0].observed_candidate_id, concrete_id)
        self.assertEqual(
            formulas[0].timing_sample_hash,
            concrete.timing_sample_hash,
        )
        exact = build_exact_winners(projected)
        self.assertEqual(next(iter(exact.values())).candidate_id, concrete_id)
        costs = build_candidate_point_costs(projected)
        generic_candidates = {
            cost.candidate_id for rows in costs.values() for cost in rows
        }
        self.assertIn(
            formula_id,
            generic_candidates,
        )
        self.assertNotIn(concrete_id, generic_candidates)

    def test_fused_cuda_formula_inventory_is_total_and_physically_admitted(self) -> None:
        """Every recipe resolves to a registered CTA even below measured K.

        The exact bridge never clamps a concrete count. Geometry dependence is
        explicit in the nominal policy and resolves before physical dispatch.
        """

        registry = cuda_native_vnni_gemv_registry()
        formulas = [entry for entry in registry.entries
                    if entry.config_json.get("family") == "fused_kpar_formula"]
        self.assertEqual(len(formulas), 2 * (40 * 4 * 2 + 64))
        for formula in formulas:
            for n in (1, 15, 16, 31, 32, 193, 1024, 248320, 1048576):
                for k in (32, 64, 96, 128, 992, 1024, 1056, 2016, 2048, 2080, 2656, 1048576):
                    kb = resolve_cuda_formula_kb(formula, n, k)
                    physical = registry.resolve(resolve_cuda_concrete_candidate_id(formula, n, k))
                    config = physical.config_json
                    self.assertEqual(config["family"], "fused_kpar")
                    self.assertEqual(config["exact_kb"], kb)
                    self.assertGreater(kb, 0)
                    self.assertLessEqual(kb, k // 32)
                    self.assertLessEqual(kb * int(config["tile_n"]), 1024)
                    self.assertNotIn("ordered_kpart_partials", physical.prepared_resources)

    def test_fused_cuda_generic_rules_require_shape_resolved_evidence(self) -> None:
        """Literal CTA partition counts cannot own unseen smaller K values.

        Every fused formula must retain its concrete timing and output witness.
        Exact overlays still name that physical launch; generic costs instead
        name a total formula with the CTA-specific partition capacity.
        """

        registry = cuda_native_vnni_gemv_registry()
        for columns in (16, 32):
            with self.subTest(columns=columns):
                physical = registry.resolve(
                    f"cuda.nvnni.decode.fast_m1.fused_kpar.tn{columns}.cpt1.kb8"
                )
                concrete = dataclasses.replace(
                    observation(
                        candidate=physical.candidate_id,
                        family=physical.candidate_family,
                        n=192, k=256, m=1,
                        mode=ExecutionMode.GRAPH_CAPTURED,
                        contract=SemanticContract.FAST,
                    ),
                    backend=Backend.CUDA,
                    prepared_family_id=format_spec("Q4_0").prepared_family("cuda"),
                    packing_abi=format_spec("Q4_0").packing_abi("cuda"),
                    runtime_codebook_id=format_spec("Q4_0").runtime_codebook("cuda"),
                    observed_candidate_id=physical.candidate_id,
                    config_json=physical.config_json,
                    candidate_policy_hash=physical.candidate_policy_hash(),
                    arithmetic_fingerprint=physical.arithmetic_fingerprint,
                )
                projected = project_cuda_shape_resolved_candidates((concrete,))
                self.assertFalse(projected.observations[0].generic_eligible)
                formula_id = (
                    f"cuda.nvnni.decode.fast_m1.fused_kpar_formula.tn{columns}.cpt1.bpp1"
                )
                formulas = [row for row in projected if row.candidate_id == formula_id]
                self.assertEqual(len(formulas), 1)
                witness = formulas[0]
                witness.validate()
                self.assertEqual(witness.effective_candidate_id, physical.candidate_id)
                self.assertEqual(witness.observed_candidate_id, physical.candidate_id)
                self.assertEqual(witness.timing_sample_hash, concrete.timing_sample_hash)
                self.assertEqual(
                    next(iter(build_exact_winners(projected).values())).candidate_id,
                    physical.candidate_id,
                )
                candidates = {
                    cost.candidate_id
                    for costs in build_candidate_point_costs(projected).values()
                    for cost in costs
                }
                self.assertNotIn(physical.candidate_id, candidates)
                self.assertIn(formula_id, candidates)

    def test_shape_resolved_cuda_aliases_share_registry_config_storage(self) -> None:
        """Formula projection must not allocate one config mapping per alias."""

        concrete_id = "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb8"
        concrete = dataclasses.replace(
            observation(
                candidate=concrete_id,
                family="cuda_public_m1_kpar",
                n=192,
                k=256,
                m=1,
                contract=SemanticContract.FAST,
            ),
            backend=Backend.CUDA,
            prepared_family_id=format_spec("Q4_0").prepared_family("cuda"),
            packing_abi=format_spec("Q4_0").packing_abi("cuda"),
            runtime_codebook_id=format_spec("Q4_0").runtime_codebook("cuda"),
            observed_candidate_id=concrete_id,
            config_json={
                "family": "kpar",
                "tile_n": 128,
                "cpt": 1,
                "exact_kb": 8,
            },
        )
        sibling = dataclasses.replace(
            concrete,
            shape_group_id="shape-registry-config-sibling",
            shape_name="shape-registry-config-sibling",
        )
        projected = project_cuda_shape_resolved_candidates(
            ObservationCorpus((concrete, sibling))
        )
        formula_id = (
            "cuda.nvnni.decode.fast_m1.kpar_formula."
            "tn128.cpt1.tb328.mkg1"
        )
        formulas = tuple(
            row for row in projected if row.candidate_id == formula_id
        )

        self.assertEqual(len(formulas), 2)
        self.assertIs(formulas[0].config_json, formulas[1].config_json)
        for row in formulas:
            row.validate()

    def test_shape_clamped_rocm_formula_uses_concrete_measured_evidence(self) -> None:
        """Generic requested KB must resolve to the launch ROCm actually runs."""

        physical = rocm_native_vnni_decode_registry().resolve(
            "rocm.nvnni.decode.fast.kb8"
        )
        concrete = dataclasses.replace(
            observation(
                candidate=physical.candidate_id,
                family=physical.candidate_family,
                n=192,
                k=256,
                m=1,
                mode=ExecutionMode.GRAPH_CAPTURED,
                contract=SemanticContract.FAST,
            ),
            config_json=physical.config_json,
            arithmetic_fingerprint=physical.arithmetic_fingerprint,
            candidate_policy_hash=physical.candidate_policy_hash(),
            observed_candidate_id=physical.candidate_id,
        )
        projected = project_rocm_shape_resolved_candidates(
            (concrete,),
            distinguish_execution_mode=False,
        )
        self.assertFalse(projected.distinguishes_execution_mode)
        formula = rocm_native_vnni_decode_formula_registry().resolve(
            "rocm.nvnni.decode.fast.clamped_formula.kb64"
        )
        formula_rows = tuple(
            row for row in projected if row.candidate_id == formula.candidate_id
        )

        self.assertEqual(resolve_rocm_formula_kb(formula, 256), 8)
        self.assertEqual(len(formula_rows), 1)
        self.assertEqual(
            formula_rows[0].effective_candidate_id,
            physical.candidate_id,
        )
        self.assertEqual(
            formula_rows[0].timing_sample_hash,
            concrete.timing_sample_hash,
        )
        self.assertFalse(projected.observations[0].generic_eligible)
        for row in projected:
            row.validate()
        self.assertEqual(
            next(iter(build_exact_winners(projected).values())).candidate_id,
            physical.candidate_id,
        )

        formula_row = formula_rows[0]
        rule = GenericDispatchRule(
            domain=projected.generic_domain_for(formula_row),
            predicates=(),
            candidate_id=formula.candidate_id,
            arithmetic_fingerprint=formula.arithmetic_fingerprint,
            development_shape_groups=(formula_row.shape_group_id,),
            development_max_regret=0.0,
            development_p95_regret=0.0,
            development_mean_regret=0.0,
        )
        policy = make_policy_ir(
            build_exact_winners(projected),
            GenericPolicy(rules=(rule,), unpromoted_domains=()),
            metadata={"test": "ROCm formula sealed certification"},
        )

        report = certify_generic_policy(policy, projected)

        self.assertEqual(report.covered_cell_count, 1)
        self.assertEqual(report.unexercised_rule_count, 0)
        self.assertEqual(
            report.cells[0].selected_candidate_id,
            formula.candidate_id,
        )
        self.assertEqual(report.cells[0].exact_candidate_id, physical.candidate_id)
        self.assertEqual(report.cells[0].observed_worst_surface_regret, 0.0)

    def test_rocm_formula_projection_is_candidate_total_across_k_sizes(self) -> None:
        """Every requested-KB policy must have evidence at small and large K."""

        physical_registry = rocm_native_vnni_decode_registry()
        rows = []
        for shape_group, n, k, maximum_kb in (
            ("small-k", 192, 256, 8),
            ("large-k", 1536, 2048, 64),
        ):
            for kb in range(1, maximum_kb + 1):
                physical = physical_registry.resolve(
                    f"rocm.nvnni.decode.fast.kb{kb}"
                )
                rows.append(dataclasses.replace(
                    observation(
                        candidate=physical.candidate_id,
                        family=physical.candidate_family,
                        shape_group=shape_group,
                        shape_name=shape_group,
                        n=n,
                        k=k,
                        m=1,
                        contract=SemanticContract.FAST,
                    ),
                    config_json=physical.config_json,
                    arithmetic_fingerprint=physical.arithmetic_fingerprint,
                    candidate_policy_hash=physical.candidate_policy_hash(),
                    observed_candidate_id=physical.candidate_id,
                ))
        projected = project_rocm_shape_resolved_candidates(rows)
        costs = build_candidate_point_costs(projected)
        expected = {
            candidate.candidate_id
            for candidate in rocm_native_vnni_decode_formula_registry().entries
        }

        self.assertEqual(len(costs), 1)
        point_candidates = {
            (cost.runtime_key, cost.shape_group_id): set()
            for cost in next(iter(costs.values()))
        }
        for cost in next(iter(costs.values())):
            point_candidates[(cost.runtime_key, cost.shape_group_id)].add(
                cost.candidate_id
            )
        self.assertEqual(len(point_candidates), 2)
        self.assertTrue(all(ids == expected for ids in point_candidates.values()))

    @staticmethod
    def rocm_moe_raw_row(**overrides) -> dict[str, str]:
        """Build one strong-evidence row from the production ROCm speedometer."""

        row = {
            "backend": "rocm",
            "phase": "grouped_prefill",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "19",
            "shape": "gate_ratio_1_4",
            "role": "gateup",
            "candidate_id": "tm12_tn256",
            "m": "32",
            "n": "512",
            "k": "2048",
            "tile_m": "12",
            "tile_n": "256",
            "warmup_count": "2",
            "sample_count": "3",
            "timed_replays": "12",
            "min_us": "330.0",
            "graph_us": "331.0",
            "p95_us": "333.0",
            "mad_us": "1.0",
            "cv": "0.004",
            "pipeline_gops": "4863.0",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_bit_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal",
            "serial_output_digest": "fnv1a64:equal",
            "timing_sample_digest": "fnv1a64:timings",
            "route_counter_ok": "1",
            "observed_candidate_id": "tm12_tn256",
            "is_winner": "1",
        }
        row.update({name: str(value) for name, value in overrides.items()})
        return row

    def test_rocm_moe_adapter_preserves_alias_bundle_and_route_proof(self) -> None:
        context = ROCmMoEAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "1" * 64
        )
        result = adapt_rocm_moe_row(self.rocm_moe_raw_row(), context)

        self.assertEqual(result.source_codebook_id, 21)
        self.assertEqual(result.runtime_codebook_id, 19)
        self.assertEqual(result.projection_n_vector, (512, 512))
        self.assertEqual(result.aggregate_n, 1024)
        self.assertEqual(
            result.candidate_id,
            "rocm.moe.grouped_prefill.tm12.tn256",
        )
        self.assertEqual(result.observed_candidate_id, result.effective_candidate_id)
        self.assertTrue(result.forced_route_ok)
        self.assertTrue(result.bitwise_equal)
        self.assertTrue(result.repeat_equal)

    def test_rocm_moe_adapter_retains_failed_route_as_ineligible_evidence(self) -> None:
        context = ROCmMoEAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "2" * 64
        )
        result = adapt_rocm_moe_row(
            self.rocm_moe_raw_row(route_counter_ok=0, observed_candidate_id="missing"),
            context,
        )
        self.assertFalse(result.forced_route_ok)
        self.assertFalse(candidate_is_eligible(result, context.serial_m1_policy_hash))

    def test_installable_rocm_moe_profile_rejects_smoke_timing(self) -> None:
        context = ROCmMoEAdapterContext(
            profile=MeasurementProfile.PRODUCTION,
            run_id="run-20260711",
            corpus_id="sha256:" + "3" * 64,
            git_revision="0123456789abcdef",
            build_id="release-20260711",
            compiler_id="hip-clang-7.1",
            architecture_class="gfx906-native-vnni-v1",
            device_name="AMD-Instinct-MI50",
            driver_runtime="ROCm-7.1",
            serial_m1_policy_hash=SERIAL_HASH,
            raw_timing_sidecar_retained=True,
        )
        context.validate()
        with self.assertRaisesRegex(ValueError, "at least 5 warmups/30 samples"):
            adapt_rocm_moe_row(self.rocm_moe_raw_row(), context)

    def test_rocm_moe_raw_timing_sidecar_is_digest_and_statistic_checked(self) -> None:
        samples_ms = (0.330, 0.331, 0.332)
        mean_ms = sum(samples_ms) / len(samples_ms)
        cv = math.sqrt(
            sum((value - mean_ms) ** 2 for value in samples_ms) / len(samples_ms)
        ) / mean_ms
        aggregate = self.rocm_moe_raw_row(
            min_us=330.0,
            graph_us=331.0,
            p95_us=332.0,
            cv=f"{cv:.6f}",
            timing_sample_digest=timing_sample_digest(samples_ms),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate_path = root / "aggregate.csv"
            timing_path = root / "timing.csv"
            with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(aggregate))
                writer.writeheader()
                writer.writerow(aggregate)

            timing_fields = (
                "backend", "phase", "source_format", "source_codebook",
                "execution_codebook", "shape", "role", "candidate_id", "m",
                "n", "k", "tile_m", "tile_n", "sample_index", "timed_replays",
                "latency_us", "latency_ms_hex",
            )
            with timing_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=timing_fields)
                writer.writeheader()
                for index, value in enumerate(samples_ms):
                    writer.writerow({
                        **{name: aggregate[name] for name in timing_fields
                           if name in aggregate},
                        "sample_index": index,
                        "timed_replays": 4,
                        "latency_us": f"{value * 1000.0:.9f}",
                        "latency_ms_hex": value.hex(),
                    })

            context = ROCmMoEAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate_path, timing_path))
            )
            corpus = adapt_rocm_moe_csv(
                (aggregate_path,), context, timing_sidecars=(timing_path,)
            )
            self.assertEqual(len(corpus), 1)
            self.assertEqual(next(iter(corpus)).timing_sample_hash,
                             timing_sample_digest(samples_ms))

            broken = dict(aggregate)
            broken["timing_sample_digest"] = "fnv1a64:0000000000000000"
            with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(broken))
                writer.writeheader()
                writer.writerow(broken)
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                adapt_rocm_moe_csv(
                    (aggregate_path,), context, timing_sidecars=(timing_path,)
                )

    def test_schema_rejects_missing_field_and_auto_candidate(self) -> None:
        raw = observation().canonical_mapping()
        raw.pop("route_counter_ok")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            NativeVNNIObservation.from_mapping(raw)

        with self.assertRaisesRegex(ValueError, "AUTO"):
            dataclasses.replace(observation(), candidate_id="AUTO").validate()

    def test_current_learner_accepts_v8_measurements_but_rejects_unknown_versions(
        self,
    ) -> None:
        """Search evolution may reuse immutable evidence, not unknown schemas."""

        dataclasses.replace(
            observation(),
            learner_version="native-vnni-bounded-tree-beam-regret-v8",
        ).validate()
        with self.assertRaisesRegex(ValueError, "accepted observation versions"):
            dataclasses.replace(
                observation(),
                learner_version="native-vnni-bounded-tree-beam-regret-v7",
            ).validate()

    def test_one_ulp_candidate_is_ineligible_despite_perfect_diagnostics(self) -> None:
        one_ulp = observation(
            candidate="candidate.one_ulp",
            bitwise_equal=False,
            latency_us=1.0,
        )
        safe = observation(candidate="candidate.safe", latency_us=2.0)

        self.assertFalse(candidate_is_eligible(one_ulp, SERIAL_HASH))
        winner = build_exact_winner([one_ulp, safe], current_serial_m1_hash=SERIAL_HASH)
        self.assertEqual(winner.candidate_id, "candidate.safe")

    def test_atomic_and_stale_oracle_candidates_are_filtered(self) -> None:
        atomic = observation(candidate="candidate.atomic", uses_atomic=True)
        stale = observation(candidate="candidate.stale", serial_hash="sha256:old")
        self.assertFalse(candidate_is_eligible(atomic, SERIAL_HASH))
        self.assertFalse(candidate_is_eligible(stale, SERIAL_HASH))

    def test_opposing_alias_winners_choose_worst_surface_robust_candidate(self) -> None:
        rows = []
        rows += candidate_rows_for_aliases(
            "candidate.a", {"Q4_1": 1.0, "Q4_K": 2.0}
        )
        rows += candidate_rows_for_aliases(
            "candidate.b", {"Q4_1": 2.0, "Q4_K": 1.0}
        )
        rows += candidate_rows_for_aliases(
            "candidate.compromise", {"Q4_1": 1.2, "Q4_K": 1.2}
        )

        winner = build_exact_winner(rows, current_serial_m1_hash=SERIAL_HASH)
        self.assertEqual(winner.candidate_id, "candidate.compromise")
        self.assertAlmostEqual(winner.max_surface_regret, 0.2)

    def test_eager_and_graph_captured_modes_publish_independent_exact_winners(self) -> None:
        """Policy ABI v2 must retain launch-mode-specific economical schedules."""

        rows = ObservationCorpus([
            observation(
                candidate="candidate.eager",
                mode=ExecutionMode.EAGER,
                latency_us=1.0,
            ),
            observation(
                candidate="candidate.captured",
                mode=ExecutionMode.EAGER,
                latency_us=2.0,
            ),
            observation(
                candidate="candidate.eager",
                mode=ExecutionMode.GRAPH_CAPTURED,
                latency_us=4.0,
            ),
            observation(
                candidate="candidate.captured",
                mode=ExecutionMode.GRAPH_CAPTURED,
                latency_us=1.0,
            ),
        ])

        winners = build_exact_winners(rows)

        self.assertEqual(len(winners), 2)
        by_mode = {key.execution_mode: winner for key, winner in winners.items()}
        self.assertEqual(
            by_mode[ExecutionMode.EAGER].candidate_id,
            "candidate.eager",
        )
        self.assertEqual(
            by_mode[ExecutionMode.GRAPH_CAPTURED].candidate_id,
            "candidate.captured",
        )

    def test_certification_uses_mode_collapsed_runtime_key_for_exact_oracle(
        self,
    ) -> None:
        """A mode-robust ABI must certify both physical launch surfaces."""

        development_rows = []
        for group_index, n in enumerate((256, 384, 448)):
            for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
                development_rows.extend((
                    observation(
                        candidate="candidate.robust",
                        shape_group=f"dev-{group_index}",
                        shape_name=f"dev-{group_index}",
                        n=n,
                        mode=mode,
                        latency_us=10.0,
                    ),
                    observation(
                        candidate="candidate.slow",
                        shape_group=f"dev-{group_index}",
                        shape_name=f"dev-{group_index}",
                        n=n,
                        mode=mode,
                        latency_us=12.0,
                    ),
                ))
        development = ObservationCorpus(
            development_rows,
            distinguish_execution_mode=False,
        )
        generic = fit_generic_policy(development, max_leaves=1)
        policy = make_policy_ir(
            build_exact_winners(development),
            generic,
            metadata={"test": "mode-collapsed certification"},
        )
        sealed = ObservationCorpus(
            tuple(
                observation(
                    candidate=candidate,
                    shape_group="sealed",
                    shape_name="sealed",
                    n=512,
                    mode=mode,
                    latency_us=latency,
                )
                for mode in (
                    ExecutionMode.EAGER,
                    ExecutionMode.GRAPH_CAPTURED,
                )
                for candidate, latency in (
                    ("candidate.robust", 10.0),
                    ("candidate.slow", 12.0),
                )
            ),
            distinguish_execution_mode=False,
        )

        report = certify_generic_policy(policy, sealed)

        self.assertEqual(report.covered_cell_count, 1)
        self.assertEqual(report.cells[0].execution_mode_count, 2)
        self.assertEqual(report.cells[0].exact_candidate_id, "candidate.robust")

    def test_parallel_certification_matches_serial_point_order(self) -> None:
        """Physical-core cell reduction must preserve certificate bytes."""

        development = ObservationCorpus(tuple(
            observation(
                candidate=candidate,
                family=candidate,
                shape_group=f"cert-dev-{index}",
                shape_name=f"cert-dev-{index}",
                n=256 + index * 32,
                latency_us=latency,
            )
            for index in range(6)
            for candidate, latency in (
                ("candidate.fast", 10.0),
                ("candidate.slow", 12.0),
            )
        ))
        policy = make_policy_ir(
            build_exact_winners(development),
            fit_generic_policy(development, max_leaves=1),
            metadata={"test": "parallel sealed certification"},
        )
        sealed = ObservationCorpus(tuple(
            observation(
                candidate=candidate,
                family=candidate,
                shape_group=f"cert-sealed-{index}",
                shape_name=f"cert-sealed-{index}",
                n=512 + index * 32,
                latency_us=latency,
            )
            for index in range(12)
            for candidate, latency in (
                ("candidate.fast", 10.0),
                ("candidate.slow", 12.0),
            )
        ))

        with mock.patch.object(
            certification_module,
            "_physical_core_count",
            return_value=2,
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CERTIFICATION_WORKERS": "1"},
        ):
            serial = certify_generic_policy(policy, sealed)
        with mock.patch.object(
            certification_module,
            "_physical_core_count",
            return_value=2,
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_CERTIFICATION_WORKERS": "2"},
        ):
            parallel = certify_generic_policy(policy, sealed)

        self.assertEqual(parallel, serial)

    def test_parallel_exact_winners_match_serial_runtime_key_order(self) -> None:
        """Physical-core reduction must preserve exact policy bytes and order."""

        rows = ObservationCorpus(tuple(
            observation(
                candidate=f"candidate.{candidate}",
                family=f"family.{candidate}",
                shape_group=f"shape-{shape:03d}",
                n=256 + shape * 16,
                k=1024 + shape * 32,
                m=1,
                latency_us=1.0 + candidate * 0.25,
            )
            for shape in range(40)
            for candidate in range(2)
        ))
        with mock.patch.object(
            exact_oracle_module,
            "_physical_core_count",
            return_value=2,
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_EXACT_WINNER_WORKERS": "1"},
        ):
            serial = build_exact_winners(rows)
        with mock.patch.object(
            exact_oracle_module,
            "_physical_core_count",
            return_value=2,
        ), mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_EXACT_WINNER_WORKERS": "2"},
        ):
            parallel = build_exact_winners(rows)

        self.assertEqual(tuple(parallel), tuple(serial))
        self.assertEqual(parallel, serial)

    def test_missing_alias_surface_cannot_win_or_quietly_fallback(self) -> None:
        rows = candidate_rows_for_aliases(
            "candidate.partial", {"Q4_1": 1.0}
        )
        required = frozenset({
            SurfaceKey("Q4_1", ExecutionMode.GRAPH_CAPTURED),
            SurfaceKey("Q4_K", ExecutionMode.GRAPH_CAPTURED),
        })
        with self.assertRaisesRegex(ValueError, "no eligible candidate"):
            build_exact_winner(
                rows,
                required_surfaces=required,
                current_serial_m1_hash=SERIAL_HASH,
            )

    def test_regret_learner_rejects_more_than_five_percent_large_misses(
        self,
    ) -> None:
        rows = []
        large_miss_indices = {5, 25, 45, 65, 85, 99}
        for index in range(100):
            group = f"shape-{index:03d}"
            n = 512 + 16 * index
            a_latency = 1.40 if index in large_miss_indices else 1.0
            rows.append(observation(
                candidate="candidate.modal",
                family="modal",
                shape_group=group,
                n=n,
                k=4 * n,
                latency_us=a_latency,
            ))
            rows.append(observation(
                candidate="candidate.robust",
                family="robust",
                shape_group=group,
                n=n,
                k=4 * n,
                latency_us=1.02,
            ))
        policy = fit_generic_policy(ObservationCorpus(rows), max_leaves=1)
        self.assertEqual(len(policy.rules), 1)
        self.assertEqual(policy.rules[0].candidate_id, "candidate.robust")
        self.assertLessEqual(policy.rules[0].development_max_regret, 0.02 + 1.0e-12)

    def test_burned_seal_costs_change_generic_fit_without_exact_overlay(
        self,
    ) -> None:
        """Ratio-only failed-seal evidence may influence only generic rules."""

        rows = []
        for index in range(20):
            n = 512 + 32 * index
            for candidate, latency in (
                ("candidate.a", 10.0),
                ("candidate.b", 10.1),
            ):
                rows.append(observation(
                    candidate=candidate,
                    family=candidate,
                    shape_group=f"broad-{index}",
                    n=n,
                    k=4 * n,
                    latency_us=latency,
                ))
        corpus = ObservationCorpus(rows)
        domain = corpus.generic_domains()[0]
        supplemental = []
        supplemental_dimensions = ((3008, 12032), (3040, 12160))
        exemplar = runtime_key(rows[0])
        for index, (n, k) in enumerate(supplemental_dimensions):
            key = dataclasses.replace(
                exemplar,
                projection_n_vector=(n,),
                aggregate_n=n,
                k=k,
            )
            for candidate, regret in (
                ("candidate.a", 0.30),
                ("candidate.b", 0.0),
            ):
                supplemental.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"cpu-burned-seal:unit-{index}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        baseline = fit_generic_policy(corpus, max_leaves=1)
        adapted = fit_generic_policy(
            corpus,
            max_leaves=1,
            supplemental_development_costs={domain: tuple(supplemental)},
        )
        frozen = freeze_policy(
            corpus,
            sealed_commitment="sha256:" + "1" * 64,
            split_manifest_digest="sha256:" + "2" * 64,
            supplemental_development_costs={domain: tuple(supplemental)},
            max_leaves=1,
            require_promotable=False,
        )

        self.assertEqual(baseline.rules[0].candidate_id, "candidate.a")
        self.assertFalse(adapted.rules)
        self.assertIn(
            adapted.cross_validation[0].worst_shape_group_id,
            {"cpu-burned-seal:unit-0", "cpu-burned-seal:unit-1"},
        )
        exact_dimensions = {
            (entry.key.aggregate_n, entry.key.k)
            for entry in frozen.policy_ir.exact_entries
        }
        self.assertTrue(set(supplemental_dimensions).isdisjoint(exact_dimensions))

    def test_low_winner_label_accuracy_can_pass_when_all_regret_is_below_three_pct(self) -> None:
        rows = []
        for index in range(8):
            group = f"alternating-{index}"
            n = 512 + 32 * index
            rows.extend([
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=group,
                    n=n,
                    k=4 * n,
                    latency_us=1.0 if index % 2 == 0 else 1.02,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=group,
                    n=n,
                    k=4 * n,
                    latency_us=1.02 if index % 2 == 0 else 1.0,
                ),
                observation(
                    candidate="candidate.never_exact",
                    family="stable",
                    shape_group=group,
                    n=n,
                    k=4 * n,
                    latency_us=1.01,
                ),
            ])
        policy = fit_generic_policy(ObservationCorpus(rows), max_leaves=1)
        self.assertEqual(policy.rules[0].candidate_id, "candidate.never_exact")
        self.assertLess(
            policy.rules[0].development_max_regret,
            P95_REGRET_BUDGET,
        )

    def test_single_shape_domain_remains_exact_only(self) -> None:
        corpus = ObservationCorpus([
            observation(candidate="candidate.a"),
            observation(candidate="candidate.b", latency_us=11.0),
        ])
        policy = fit_generic_policy(corpus)
        self.assertFalse(policy.rules)
        self.assertEqual(len(policy.unpromoted_domains), 1)

    def test_failed_cv_domain_remains_explicitly_unpromoted(self) -> None:
        rows = []
        for index in range(6):
            a_is_fast = index < 3
            n = 512 + 64 * index
            rows.extend([
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=f"rejected-cv-{index}",
                    n=n,
                    k=4 * n,
                    latency_us=1.0 if a_is_fast else 2.0,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=f"rejected-cv-{index}",
                    n=n,
                    k=4 * n,
                    latency_us=2.0 if a_is_fast else 1.0,
                ),
            ])

        policy = fit_generic_policy(ObservationCorpus(rows), max_leaves=1)

        self.assertFalse(policy.rules)
        self.assertEqual(len(policy.unpromoted_domains), 1)
        self.assertEqual(len(policy.cross_validation), 1)
        rejected = policy.cross_validation[0]
        self.assertEqual(rejected.selected_max_leaves, 1)
        self.assertEqual(rejected.covered_point_count, rejected.required_point_count)
        self.assertGreater(rejected.max_regret, P95_REGRET_BUDGET)
        self.assertTrue(rejected.worst_shape_group_id.startswith("rejected-cv-"))
        self.assertIn(
            rejected.worst_selected_candidate_id,
            {"candidate.a", "candidate.b"},
        )
        self.assertIn(
            rejected.worst_exact_candidate_id,
            {"candidate.a", "candidate.b"},
        )
        self.assertNotEqual(
            rejected.worst_selected_candidate_id,
            rejected.worst_exact_candidate_id,
        )
        self.assertEqual(len(rejected.cells), rejected.required_point_count)
        self.assertEqual(len(policy.promotion_diagnostics), 1)
        diagnostic = policy.promotion_diagnostics[0]
        self.assertEqual(diagnostic.rejection_stage, "cross_validation_p95")
        self.assertEqual(diagnostic.cv_p95_regret, rejected.p95_regret)
        self.assertEqual(diagnostic.final_rule_count, 0)
        self.assertIsNone(diagnostic.final_worst_p95_regret)
        self.assertIsNone(diagnostic.final_worst_aggregate_n)

        p95_pass = dataclasses.replace(
            rejected,
            max_regret=0.50,
            p95_regret=0.049,
        )
        self.assertTrue(
            segmented_policy.domain_cross_validation_is_promotable(p95_pass)
        )
        self.assertFalse(
            segmented_policy.domain_cross_validation_is_promotable(
                dataclasses.replace(
                    p95_pass,
                    p95_regret=P95_REGRET_BUDGET,
                )
            )
        )
        worst_cell = max(
            rejected.cells,
            key=lambda cell: (
                cell.observed_broad_regret,
                cell.shape_group_id,
                cell.selected_candidate_id,
            ),
        )
        self.assertEqual(
            worst_cell.shape_group_id,
            rejected.worst_shape_group_id,
        )
        self.assertEqual(
            worst_cell.selected_candidate_id,
            rejected.worst_selected_candidate_id,
        )
        self.assertEqual(
            worst_cell.exact_candidate_id,
            rejected.worst_exact_candidate_id,
        )

    def test_domain_promotion_quota_is_exactly_ninety_five_percent(self) -> None:
        """The corpus gate is inclusive while every domain gate stays strict."""

        self.assertTrue(
            segmented_policy.domain_promotion_quota_is_satisfied(95, 100)
        )
        self.assertTrue(
            segmented_policy.domain_promotion_quota_is_satisfied(19, 20)
        )
        self.assertFalse(
            segmented_policy.domain_promotion_quota_is_satisfied(94, 99)
        )
        self.assertFalse(
            segmented_policy.domain_promotion_quota_is_satisfied(0, 0)
        )
        with self.assertRaisesRegex(ValueError, "outside the required domain"):
            segmented_policy.domain_promotion_quota_is_satisfied(101, 100)

    def test_manual_domain_quota_preserves_structural_nonempty_requirement(self) -> None:
        """A zero best-effort quota admits misses but never an empty corpus."""

        self.assertTrue(
            segmented_policy.domain_promotion_quota_is_satisfied(
                0,
                100,
                minimum_passing_fraction=0.0,
            )
        )
        self.assertFalse(
            segmented_policy.domain_promotion_quota_is_satisfied(
                0,
                0,
                minimum_passing_fraction=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "fraction must be in"):
            segmented_policy.domain_promotion_quota_is_satisfied(
                1,
                1,
                minimum_passing_fraction=1.01,
            )

    def test_freeze_counts_structural_domain_without_cross_validation(self) -> None:
        """A missing CV result remains one rejected domain, never a negative count."""

        development = ObservationCorpus([
            observation(candidate="candidate.a"),
            observation(candidate="candidate.b", latency_us=11.0),
        ])
        structural_policy = fit_generic_policy(development)
        self.assertFalse(structural_policy.cross_validation)
        self.assertEqual(len(structural_policy.promotion_diagnostics), 1)
        self.assertEqual(
            structural_policy.promotion_diagnostics[0].rejection_stage,
            "cross_validation_missing",
        )

        with mock.patch(
            "native_vnni_dispatch.compiler.fit_generic_policy",
            return_value=structural_policy,
        ), mock.patch(
            "native_vnni_dispatch.compiler.MINIMUM_PASSING_DOMAIN_FRACTION",
            0.0,
        ):
            frozen = freeze_policy(
                development,
                sealed_commitment="sha256:sealed-shape-inventory",
                split_manifest_digest="sha256:split-manifest",
            )

        self.assertEqual(
            frozen.policy_ir.metadata["development_required_domain_count"],
            1,
        )
        self.assertEqual(
            frozen.policy_ir.metadata["development_passing_domain_count"],
            0,
        )
        self.assertEqual(
            frozen.policy_ir.metadata["development_passing_domain_fraction"],
            0.0,
        )
        self.assertTrue(
            frozen.policy_ir.metadata[
                "development_domain_promotion_quota_satisfied"
            ]
        )
        self.assertEqual(len(frozen.policy_ir.unpromoted_domains), 1)

    def test_zero_quota_freezes_complete_over_budget_domain(self) -> None:
        """A performance exception keeps its real tree and explicit diagnostic."""

        rows = []
        for index in range(6):
            a_is_fast = index < 3
            rows.extend((
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=f"best-effort-{index}",
                    n=512 + 64 * index,
                    latency_us=1.0 if a_is_fast else 2.0,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=f"best-effort-{index}",
                    n=512 + 64 * index,
                    latency_us=2.0 if a_is_fast else 1.0,
                ),
            ))
        development = ObservationCorpus(rows)
        quota = segmented_policy.domain_promotion_quota_is_satisfied
        with mock.patch.object(
            segmented_policy,
            "domain_promotion_quota_is_satisfied",
            side_effect=lambda passing, required: quota(
                passing,
                required,
                minimum_passing_fraction=0.0,
            ),
        ):
            performance_policy = fit_generic_policy(
                development,
                max_leaves=1,
            )
        self.assertFalse(performance_policy.unpromoted_domains)
        self.assertEqual(len(performance_policy.promotion_diagnostics), 1)
        self.assertTrue(performance_policy.rules)

        with mock.patch(
            "native_vnni_dispatch.compiler.fit_generic_policy",
            return_value=performance_policy,
        ), mock.patch(
            "native_vnni_dispatch.compiler.MINIMUM_PASSING_DOMAIN_FRACTION",
            0.0,
        ):
            frozen = freeze_policy(
                development,
                sealed_commitment="sha256:sealed-shape-inventory",
                split_manifest_digest="sha256:split-manifest",
            )

        self.assertEqual(
            frozen.policy_ir.metadata["development_required_domain_count"],
            1,
        )
        self.assertEqual(
            frozen.policy_ir.metadata["development_passing_domain_count"],
            0,
        )
        self.assertTrue(
            frozen.policy_ir.metadata[
                "development_domain_promotion_quota_satisfied"
            ]
        )
        self.assertFalse(frozen.policy_ir.unpromoted_domains)
        self.assertEqual(len(frozen.promotion_diagnostics), 1)

    def test_promotion_percent_environment_is_parsed_in_one_common_module(self) -> None:
        """Every analyzer process must observe identical manual criteria."""

        environment = {
            **os.environ,
            "PYTHONPATH": str(KERNEL_PERF_ROOT),
            "LLAMINAR_NATIVE_VNNI_PROMOTION_P95_REGRET_PERCENT": "12.5",
            "LLAMINAR_NATIVE_VNNI_PROMOTION_MIN_PASSING_DOMAIN_PERCENT": "0",
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from native_vnni_dispatch.schema import "
                    "P95_REGRET_BUDGET,MINIMUM_PASSING_DOMAIN_FRACTION;"
                    "print(P95_REGRET_BUDGET,MINIMUM_PASSING_DOMAIN_FRACTION)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.stdout.strip(), "0.125 0.0")

    def test_ninety_five_percent_policy_keeps_exception_domains_total(self) -> None:
        """A permitted performance exception must still own a generic tree."""

        rows = []
        for m in range(1, 101):
            is_exception = m > 95
            shape_count = 6 if is_exception else 3
            for shape_index in range(shape_count):
                exception_a_is_fast = shape_index < shape_count // 2
                rows.extend((
                    observation(
                        candidate="candidate.a",
                        family="a",
                        shape_group=f"quota-{m}-{shape_index}",
                        m=m,
                        n=256 + 64 * shape_index,
                        latency_us=(
                            1.0
                            if not is_exception or exception_a_is_fast
                            else 2.0
                        ),
                    ),
                    observation(
                        candidate="candidate.b",
                        family="b",
                        shape_group=f"quota-{m}-{shape_index}",
                        m=m,
                        n=256 + 64 * shape_index,
                        latency_us=(
                            1.10
                            if not is_exception
                            else (2.0 if exception_a_is_fast else 1.0)
                        ),
                    ),
                ))

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            policy = fit_generic_policy(
                ObservationCorpus(rows),
                max_leaves=1,
            )

        self.assertFalse(policy.unpromoted_domains)
        self.assertEqual(len(policy.promotion_diagnostics), 5)
        self.assertEqual(
            {diagnostic.domain.m for diagnostic in policy.promotion_diagnostics},
            {96, 97, 98, 99, 100},
        )
        self.assertTrue(all(
            diagnostic.rejection_stage == "cross_validation_p95"
            for diagnostic in policy.promotion_diagnostics
        ))
        self.assertEqual(len({rule.domain for rule in policy.rules}), 100)
        for diagnostic in policy.promotion_diagnostics:
            self.assertIsNotNone(
                policy.resolve(diagnostic.domain, 256, 2048)
            )

    def test_final_fit_p95_failure_retains_separate_diagnostic(self) -> None:
        """A full-development leaf failure must not look like a CV miss."""

        rows = []
        for index, n in enumerate((256, 320, 384, 448, 512, 576)):
            rows.extend((
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=f"final-diagnostic-{index}",
                    n=n,
                    latency_us=10.0,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=f"final-diagnostic-{index}",
                    n=n,
                    latency_us=11.0,
                ),
            ))
        original = segmented_policy._fit_final_domain

        def poison_final_p95(task, primary_scorer=None):
            domain, rules, validation = original(task, primary_scorer)
            return domain, tuple(
                dataclasses.replace(
                    rule,
                    development_p95_regret=P95_REGRET_BUDGET,
                    development_max_regret=0.07,
                )
                for rule in rules
            ), validation

        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ), mock.patch.object(
            segmented_policy,
            "_fit_final_domain",
            side_effect=poison_final_p95,
        ):
            policy = fit_generic_policy(
                ObservationCorpus(rows),
                max_leaves=1,
            )

        self.assertFalse(policy.rules)
        self.assertEqual(len(policy.unpromoted_domains), 1)
        self.assertEqual(len(policy.promotion_diagnostics), 1)
        diagnostic = policy.promotion_diagnostics[0]
        self.assertEqual(diagnostic.rejection_stage, "final_fit_p95")
        self.assertLess(diagnostic.cv_p95_regret, P95_REGRET_BUDGET)
        self.assertEqual(diagnostic.final_rule_count, 1)
        self.assertEqual(
            diagnostic.final_worst_p95_regret,
            P95_REGRET_BUDGET,
        )
        self.assertEqual(diagnostic.final_worst_max_regret, 0.07)
        self.assertEqual(
            diagnostic.final_worst_shape_group_id,
            "final-diagnostic-5",
        )
        self.assertEqual(diagnostic.final_worst_aggregate_n, 576)
        self.assertEqual(diagnostic.final_worst_k, 2048)
        self.assertEqual(
            diagnostic.final_worst_candidate_id,
            "candidate.a",
        )

    def test_paired_ratio_corrects_broad_cost_without_absolute_clock_mix(self) -> None:
        """Development pairing re-anchors a ratio to the same broad surface."""

        rows = [
            observation(
                candidate="candidate.selected",
                family="selected",
                shape_group="paired-cost-shape",
                latency_us=11.0,
            ),
            observation(
                candidate="candidate.exact",
                family="exact",
                shape_group="paired-cost-shape",
                latency_us=10.0,
            ),
        ]
        corpus = ObservationCorpus(rows)
        broad = next(iter(build_candidate_point_costs(corpus).values()))
        broad_by_candidate = {cost.candidate_id: cost for cost in broad}
        self.assertAlmostEqual(
            broad_by_candidate["candidate.selected"].max_surface_regret,
            0.10,
        )

        exemplar = rows[0]
        pair_key = PairedCellKey(
            backend=exemplar.backend.value,
            source_format=exemplar.source_format,
            source_codebook=exemplar.source_codebook_id,
            execution_codebook=exemplar.runtime_codebook_id,
            shape=exemplar.shape_name,
            execution_mode=exemplar.execution_mode.value,
            m=exemplar.m,
            n=exemplar.aggregate_n,
            k=exemplar.k,
        )
        paired = paired_timing_comparisons((PairedCellEvidence(
            key=pair_key,
            selected_candidate_id="candidate.selected",
            exact_candidate_id="candidate.exact",
            selected_latency_us=(9.0,) * 30,
            exact_latency_us=(10.0,) * 30,
            selected_first_count=15,
            exact_first_count=15,
            selected_ran_first=tuple(index % 2 == 0 for index in range(30)),
        ),))
        corrected = next(iter(build_candidate_point_costs(
            corpus,
            paired_comparisons=paired,
        ).values()))
        corrected_by_candidate = {
            cost.candidate_id: cost for cost in corrected
        }
        self.assertEqual(
            corrected_by_candidate["candidate.selected"].max_surface_regret,
            0.0,
        )
        self.assertAlmostEqual(
            corrected_by_candidate["candidate.exact"].max_surface_regret,
            1.0 / 0.9 - 1.0,
        )

    def test_paired_ratios_pool_true_source_aliases_by_runtime_identity(self) -> None:
        """Packed aliases cannot teach opposite policies for one CPU launch."""

        rows = []
        for source_format in ("IQ4_NL", "IQ4_XS"):
            rows.extend((
                observation(
                    backend=Backend.CPU,
                    contract=SemanticContract.FAST,
                    mode=ExecutionMode.EAGER,
                    m=1,
                    source_format=source_format,
                    candidate="candidate.selected",
                    family="selected",
                    shape_group="paired-alias-shape",
                    latency_us=11.0,
                ),
                observation(
                    backend=Backend.CPU,
                    contract=SemanticContract.FAST,
                    mode=ExecutionMode.EAGER,
                    m=1,
                    source_format=source_format,
                    candidate="candidate.exact",
                    family="exact",
                    shape_group="paired-alias-shape",
                    latency_us=10.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        cells = []
        for source_format, ratio in (("IQ4_NL", 0.9), ("IQ4_XS", 1.1)):
            exemplar = next(
                row for row in rows if row.source_format == source_format
            )
            pair_key = PairedCellKey(
                backend=exemplar.backend.value,
                source_format=exemplar.source_format,
                source_codebook=exemplar.source_codebook_id,
                execution_codebook=exemplar.runtime_codebook_id,
                shape=exemplar.shape_name,
                execution_mode=exemplar.execution_mode.value,
                m=exemplar.m,
                n=exemplar.aggregate_n,
                k=exemplar.k,
                architecture_class=exemplar.architecture_class,
            )
            cells.append(PairedCellEvidence(
                key=pair_key,
                selected_candidate_id="candidate.selected",
                exact_candidate_id="candidate.exact",
                selected_latency_us=(ratio * 10.0,) * 30,
                exact_latency_us=(10.0,) * 30,
                selected_first_count=15,
                exact_first_count=15,
                selected_ran_first=tuple(
                    index % 2 == 0 for index in range(30)
                ),
            ))

        corrected = next(iter(build_candidate_point_costs(
            corpus,
            paired_comparisons=paired_timing_comparisons(cells),
        ).values()))
        by_candidate = {cost.candidate_id: cost for cost in corrected}
        pooled_ratio = math.sqrt(0.9 * 1.1)
        self.assertEqual(
            by_candidate["candidate.selected"].max_surface_regret,
            0.0,
        )
        self.assertAlmostEqual(
            by_candidate["candidate.exact"].max_surface_regret,
            1.0 / pooled_ratio - 1.0,
        )

    def test_nominal_aliases_of_one_effective_launch_have_zero_regret(self) -> None:
        """Repeated policy names cannot make one physical launch beat itself."""

        rows = tuple(
            dataclasses.replace(
                observation(
                    candidate=candidate,
                    family="nominal-alias",
                    shape_group="effective-alias-shape",
                    latency_us=latency,
                ),
                effective_candidate_id="physical.launch.same",
                observed_candidate_id="physical.launch.same",
            )
            for candidate, latency in (
                ("candidate.alias-a", 10.0),
                ("candidate.alias-b", 20.0),
            )
        )

        costs = next(iter(build_candidate_point_costs(
            ObservationCorpus(rows)
        ).values()))

        self.assertEqual(
            {cost.candidate_id: cost.max_surface_regret for cost in costs},
            {
                "candidate.alias-a": 0.0,
                "candidate.alias-b": 0.0,
            },
        )

    def test_grouped_cv_selects_two_leaves_and_uses_midpoint_boundary(self) -> None:
        rows = []
        lower_ns = (128, 160, 192)
        upper_ns = (288, 320, 384)
        for index, n in enumerate((*lower_ns, *upper_ns)):
            group = f"cv-shape-{index}"
            lower = n in lower_ns
            rows.extend([
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=group,
                    n=n,
                    latency_us=10.0 if lower else 14.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=group,
                    n=n,
                    latency_us=14.0 if lower else 10.0,
                ),
            ])

        policy = fit_generic_policy(ObservationCorpus(rows), max_leaves=2)
        self.assertEqual(len(policy.rules), 2)
        self.assertEqual(len(policy.cross_validation), 1)
        cv = policy.cross_validation[0]
        self.assertEqual(cv.selected_max_leaves, 2)
        self.assertEqual(
            cv.selected_boundary_placement,
            BoundaryPlacement.MIDPOINT,
        )
        self.assertEqual(
            cv.selected_profiler_influence,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
        )
        self.assertEqual(cv.fold_count, 6)
        self.assertEqual(cv.covered_point_count, cv.required_point_count)
        self.assertEqual(len(cv.cells), cv.required_point_count)
        predicates = {
            predicate
            for rule in policy.rules
            for predicate in rule.predicates
        }
        self.assertEqual(len(predicates), 2)
        for predicate in predicates:
            self.assertEqual(predicate.threshold.axis, FeatureAxis.AGGREGATE_N)
            self.assertEqual(predicate.threshold.numerator, 240)
            self.assertEqual(predicate.threshold.denominator, 1)
        lower = next(rule for rule in policy.rules if rule.matches(192, 2048))
        upper = next(rule for rule in policy.rules if rule.matches(288, 2048))
        self.assertEqual(lower.candidate_id, "candidate.low")
        self.assertEqual(upper.candidate_id, "candidate.high")

    def test_final_publication_tree_distills_out_of_fold_candidates(self) -> None:
        """Publication boundaries must be learned from non-leaking OOF labels."""

        rows = []
        for index, n in enumerate((128, 160, 192, 288, 320, 384)):
            lower = n <= 192
            rows.extend((
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=f"publication-oof-{index}",
                    n=n,
                    latency_us=10.0 if lower else 14.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=f"publication-oof-{index}",
                    n=n,
                    latency_us=14.0 if lower else 10.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        domain = corpus.generic_domains()[0]
        costs = build_candidate_point_costs(corpus)[domain]
        exact_by_point = {}
        for cost in costs:
            point = (cost.runtime_key, cost.shape_group_id)
            previous = exact_by_point.get(point)
            if previous is None or cost.max_surface_regret < previous.max_surface_regret:
                exact_by_point[point] = cost
        cells = tuple(
            segmented_policy.CrossValidationCell(
                runtime_key=key,
                shape_group_id=shape_group,
                selected_candidate_id=cost.candidate_id,
                exact_candidate_id=cost.candidate_id,
                observed_broad_regret=cost.max_surface_regret,
            )
            for (key, shape_group), cost in sorted(exact_by_point.items())
        )
        validation = segmented_policy.DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=2,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=(
                segmented_policy.ProfilerInfluence.MEASURED_ONLY
            ),
            fold_count=6,
            shape_group_count=6,
            required_point_count=6,
            covered_point_count=6,
            max_regret=0.0,
            p95_regret=0.0,
            mean_regret=0.0,
            worst_shape_group_id=cells[0].shape_group_id,
            worst_aggregate_n=cells[0].runtime_key.aggregate_n,
            worst_k=cells[0].runtime_key.k,
            worst_selected_candidate_id=cells[0].selected_candidate_id,
            worst_exact_candidate_id=cells[0].exact_candidate_id,
            cells=cells,
        )

        rules, fitted_validation = (
            segmented_policy._fit_cross_fitted_publication_rules(
                domain,
                costs,
                rows,
                validation,
                max_leaves=2,
                min_shape_groups_per_leaf=2,
            )
        )
        (
            candidate_tasks,
            prepared_frontiers,
            targets_by_context,
        ) = segmented_policy._prepare_accelerated_final_candidates(((
            domain,
            costs,
            tuple(rows),
            (validation,),
            (),
            2,
            2,
        ),))
        split_results = tuple(
            (
                context_index,
                segmented_policy._fit_cross_fitted_publication_candidate(
                    candidate_args
                ),
            )
            for context_index, candidate_args in candidate_tasks
        )
        split_fits = segmented_policy._reduce_accelerated_final_candidates(
            split_results,
            prepared_frontiers,
            targets_by_context,
        )

        self.assertEqual(len(rules), 2)
        self.assertEqual(split_fits, [(domain, rules, fitted_validation)])
        self.assertEqual(fitted_validation.publication_oof_p95_regret, 0.0)
        self.assertEqual(fitted_validation.publication_oof_mismatch_count, 0)
        self.assertEqual(
            fitted_validation.publication_boundary_placement,
            BoundaryPlacement.MIDPOINT,
        )
        self.assertEqual(
            next(rule for rule in rules if rule.matches(192, 2048)).candidate_id,
            "candidate.low",
        )
        self.assertEqual(
            next(rule for rule in rules if rule.matches(288, 2048)).candidate_id,
            "candidate.high",
        )

    def test_publication_frontier_contains_only_durable_cv_winner(self) -> None:
        """Cold publication must not fit runner-ups absent from a resumed fit."""

        winner = mock.Mock(
            spec=segmented_policy.DomainCrossValidation,
            required_point_count=12,
            covered_point_count=12,
        )
        runner_up = mock.Mock(
            spec=segmented_policy.DomainCrossValidation,
            required_point_count=12,
            covered_point_count=12,
        )

        frontier = segmented_policy._publication_validation_frontier(winner)

        self.assertEqual(frontier, (winner,))
        self.assertNotIn(runner_up, frontier)
        self.assertEqual(
            segmented_policy._publication_validation_frontier(None), ()
        )
        incomplete = mock.Mock(
            spec=segmented_policy.DomainCrossValidation,
            required_point_count=12,
            covered_point_count=11,
        )
        self.assertEqual(
            segmented_policy._publication_validation_frontier(incomplete), ()
        )

    def test_final_publication_tree_can_represent_union_of_fold_boundaries(self) -> None:
        """OOF distillation may need more leaves than each independent fold."""

        rows = []
        selected_by_group = {}
        for index, n in enumerate(range(128, 384, 32)):
            selected = "candidate.low" if (index // 2) % 2 == 0 else "candidate.high"
            selected_by_group[f"publication-bound-{index}"] = selected
            rows.extend((
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=f"publication-bound-{index}",
                    n=n,
                    latency_us=10.0 if selected == "candidate.low" else 14.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=f"publication-bound-{index}",
                    n=n,
                    latency_us=10.0 if selected == "candidate.high" else 14.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        domain = corpus.generic_domains()[0]
        costs = build_candidate_point_costs(corpus)[domain]
        cells = tuple(
            segmented_policy.CrossValidationCell(
                runtime_key=cost.runtime_key,
                shape_group_id=cost.shape_group_id,
                selected_candidate_id=selected_by_group[cost.shape_group_id],
                exact_candidate_id=selected_by_group[cost.shape_group_id],
                observed_broad_regret=0.0,
            )
            for cost in costs
            if cost.candidate_id == selected_by_group[cost.shape_group_id]
        )
        validation = segmented_policy.DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=2,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=(
                segmented_policy.ProfilerInfluence.MEASURED_ONLY
            ),
            fold_count=5,
            shape_group_count=len(cells),
            required_point_count=len(cells),
            covered_point_count=len(cells),
            max_regret=0.0,
            p95_regret=0.0,
            mean_regret=0.0,
            worst_shape_group_id=cells[0].shape_group_id,
            worst_aggregate_n=cells[0].runtime_key.aggregate_n,
            worst_k=cells[0].runtime_key.k,
            worst_selected_candidate_id=cells[0].selected_candidate_id,
            worst_exact_candidate_id=cells[0].exact_candidate_id,
            cells=cells,
        )

        rules, fitted = segmented_policy._fit_cross_fitted_publication_rules(
            domain,
            costs,
            rows,
            validation,
            max_leaves=4,
            min_shape_groups_per_leaf=2,
        )

        # Each held-out fold selected at most two leaves, but their alternating
        # cross-fitted decisions form four stable regions once all OOF labels
        # are assembled. Artificially retaining the fold-local two-leaf cap
        # makes the final generic tree merge unlike regions and fail its hard
        # per-leaf measured-p95 gate.
        self.assertGreater(len(rules), validation.selected_max_leaves)
        self.assertLessEqual(len(rules), 4)
        self.assertEqual(fitted.publication_max_leaves, len(rules))
        self.assertEqual(fitted.publication_oof_p95_regret, 0.0)
        self.assertTrue(all(
            rule.development_p95_regret < P95_REGRET_BUDGET
            for rule in rules
        ))

    def test_grouped_cv_retains_competitive_model_edges_for_one_batch(self) -> None:
        """The planner sees alternate model misses before the fit discards them."""

        exemplar = observation(
            backend=Backend.CUDA,
            contract=SemanticContract.FAST,
            mode=ExecutionMode.EAGER,
            m=1,
            candidate="candidate.exact",
            shape_group="frontier-shape",
        )
        key = runtime_key(exemplar)
        domain = generic_domain(exemplar)

        def cost(candidate: str, regret: float):
            return segmented_policy.CandidatePointCost(
                runtime_key=key,
                shape_group_id=exemplar.shape_group_id,
                candidate_id=candidate,
                max_surface_regret=regret,
                p95_surface_regret=regret,
                mean_surface_regret=regret,
            )

        exact = cost("candidate.exact", 0.0)
        selected_a = cost("candidate.a", 0.06)
        selected_b = cost("candidate.b", 0.07)

        def fold_result(feature_policy, selected):
            return segmented_policy._PlacementFoldResult(
                domain=domain,
                fold_index=0,
                feature_policy=feature_policy,
                placement=BoundaryPlacement.MIDPOINT,
                profiler_influence=(
                    segmented_policy.ProfilerInfluence.MEASURED_ONLY
                ),
                complexities=(segmented_policy._ComplexityFoldResult(
                    complexity=1,
                    decisions=((selected, exact),),
                    uncovered=0,
                    required=1,
                ),),
            )

        fold_results = (
            fold_result(FeaturePolicy.CONTINUOUS, selected_a),
            fold_result(FeaturePolicy.TILE_32, selected_b),
        )
        frontier = segmented_policy._rank_domain_cross_validations(
            domain,
            [exact, selected_a, selected_b],
            1,
            fold_results,
            max_leaves=1,
        )
        validation = frontier[0]

        self.assertIsNotNone(validation)
        self.assertEqual(
            validation.cells[0].selected_candidate_id,
            "candidate.a",
        )
        self.assertEqual(
            {
                cell.selected_candidate_id
                for cell in validation.competitive_cells
            },
            {"candidate.a", "candidate.b"},
        )
        self.assertTrue(validation.cells)
        self.assertTrue(all(not item.cells for item in frontier[1:]))
        self.assertTrue(all(
            item.competitive_cells == validation.competitive_cells
            for item in frontier
        ))

        lean_frontier = segmented_policy._rank_domain_cross_validations(
            domain,
            [exact, selected_a, selected_b],
            1,
            fold_results,
            max_leaves=1,
            retain_model_cells=False,
        )
        self.assertTrue(all(not item.cells for item in lean_frontier))
        self.assertTrue(all(
            not item.competitive_cells for item in lean_frontier
        ))
        hydrated = segmented_policy._hydrate_cross_validation_cells(
            lean_frontier[0],
            fold_results,
        )
        self.assertEqual(hydrated.cells, validation.cells)

    def test_final_fit_advances_to_next_cv_certified_model(self) -> None:
        """A one-shot final partition miss must not discard a stable CV model."""

        exemplar = observation(
            backend=Backend.CPU,
            contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            mode=ExecutionMode.EAGER,
            m=64,
            candidate="candidate.exact",
            shape_group="final-frontier-shape",
        )
        key = runtime_key(exemplar)
        domain = generic_domain(exemplar)

        def cost(candidate: str, regret: float):
            return segmented_policy.CandidatePointCost(
                runtime_key=key,
                shape_group_id=exemplar.shape_group_id,
                candidate_id=candidate,
                max_surface_regret=regret,
                p95_surface_regret=regret,
                mean_surface_regret=regret,
            )

        exact = cost("candidate.exact", 0.0)
        selected_a = cost("candidate.a", 0.01)
        selected_b = cost("candidate.b", 0.02)

        def fold_result(feature_policy, selected):
            return segmented_policy._PlacementFoldResult(
                domain=domain,
                fold_index=0,
                feature_policy=feature_policy,
                placement=BoundaryPlacement.MIDPOINT,
                profiler_influence=(
                    segmented_policy.ProfilerInfluence.MEASURED_ONLY
                ),
                complexities=(segmented_policy._ComplexityFoldResult(
                    complexity=1,
                    decisions=((selected, exact),),
                    uncovered=0,
                    required=1,
                ),),
            )

        fold_results = (
            fold_result(FeaturePolicy.CONTINUOUS, selected_a),
            fold_result(FeaturePolicy.TILE_32, selected_b),
        )
        frontier = segmented_policy._rank_domain_cross_validations(
            domain,
            [exact, selected_a, selected_b],
            1,
            fold_results,
            max_leaves=1,
        )
        self.assertEqual(
            tuple(item.selected_feature_policy for item in frontier),
            (FeaturePolicy.CONTINUOUS, FeaturePolicy.TILE_32),
        )

        def rule(candidate: str, p95: float):
            return (GenericDispatchRule(
                domain=domain,
                predicates=(),
                candidate_id=candidate,
                arithmetic_fingerprint=exemplar.arithmetic_fingerprint,
                development_shape_groups=(exemplar.shape_group_id,),
                development_max_regret=max(p95, 0.05),
                development_p95_regret=p95,
                development_mean_regret=p95,
            ),)

        def fit_publication(
            _domain,
            _costs,
            _observations,
            validation,
            **_kwargs,
        ):
            if validation.selected_feature_policy == FeaturePolicy.CONTINUOUS:
                return rule("candidate.a", 0.01), dataclasses.replace(
                    validation,
                    publication_oof_p95_regret=P95_REGRET_BUDGET,
                    publication_oof_mean_regret=P95_REGRET_BUDGET,
                    publication_oof_mismatch_count=1,
                )
            return rule("candidate.b", 0.01), dataclasses.replace(
                validation,
                publication_oof_p95_regret=0.01,
                publication_oof_mean_regret=0.01,
                publication_oof_mismatch_count=0,
            )

        with mock.patch.object(
            segmented_policy,
            "_fit_cross_fitted_publication_rules",
            side_effect=fit_publication,
        ) as fit:
            fitted_domain, rules, selected = segmented_policy._fit_final_domain((
                domain,
                [exact, selected_a, selected_b],
                (exemplar,),
                frontier,
                fold_results,
                1,
                1,
            ))

        self.assertEqual(fitted_domain, domain)
        self.assertEqual(rules[0].candidate_id, "candidate.b")
        self.assertEqual(selected.selected_feature_policy, FeaturePolicy.TILE_32)
        self.assertEqual(fit.call_count, 2)

    def test_grouped_cv_selects_profiler_influence_only_when_helpful(self) -> None:
        """Profiler economics cannot displace a better measured-only model."""

        exemplar = observation(
            backend=Backend.CUDA,
            contract=SemanticContract.FAST,
            mode=ExecutionMode.EAGER,
            m=1,
            candidate="candidate.exact",
            shape_group="profiler-influence-shape",
        )
        key = runtime_key(exemplar)
        domain = generic_domain(exemplar)

        def cost(candidate: str, regret: float):
            return segmented_policy.CandidatePointCost(
                runtime_key=key,
                shape_group_id=exemplar.shape_group_id,
                candidate_id=candidate,
                max_surface_regret=regret,
                p95_surface_regret=regret,
                mean_surface_regret=regret,
            )

        exact = cost("candidate.exact", 0.0)

        def select(measured_regret: float, prior_regret: float):
            measured = cost("candidate.measured", measured_regret)
            prior = cost("candidate.profiler", prior_regret)

            def result(influence, selected):
                return segmented_policy._PlacementFoldResult(
                    domain=domain,
                    fold_index=0,
                    feature_policy=FeaturePolicy.CONTINUOUS,
                    placement=BoundaryPlacement.MIDPOINT,
                    profiler_influence=influence,
                    complexities=(segmented_policy._ComplexityFoldResult(
                        complexity=1,
                        decisions=((selected, exact),),
                        uncovered=0,
                        required=1,
                    ),),
                )

            return segmented_policy._select_domain_cross_validation(
                domain,
                [exact, measured, prior],
                1,
                (
                    result(
                        segmented_policy.ProfilerInfluence.MEASURED_ONLY,
                        measured,
                    ),
                    result(
                        segmented_policy.ProfilerInfluence.BOUNDED_PRIOR,
                        prior,
                    ),
                ),
                max_leaves=1,
            )

        harmful = select(0.01, 0.20)
        helpful = select(0.20, 0.01)
        self.assertEqual(
            harmful.selected_profiler_influence,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
        )
        self.assertEqual(
            helpful.selected_profiler_influence,
            segmented_policy.ProfilerInfluence.BOUNDED_PRIOR,
        )

    def test_cv_tree_evaluation_matches_flattened_production_rules(self) -> None:
        """Observation-free CV traversal preserves every leaf decision exactly."""

        rows = []
        for index, n in enumerate((128, 160, 192, 288, 320, 384)):
            lower = n <= 192
            rows.extend([
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=f"tree-eval-{index}",
                    n=n,
                    latency_us=10.0 if lower else 14.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=f"tree-eval-{index}",
                    n=n,
                    latency_us=14.0 if lower else 10.0,
                ),
            ])
        corpus = ObservationCorpus(rows)
        domain = corpus.generic_domains()[0]
        costs = build_candidate_point_costs(corpus)[domain]
        fit = segmented_policy._fit_tree(
            costs,
            max_leaves=2,
            min_shape_groups_per_leaf=2,
            boundary_placement=BoundaryPlacement.MIDPOINT,
            feature_policy=FeaturePolicy.CONTINUOUS,
        )
        exemplars = {row.candidate_id: row for row in rows}
        rules = segmented_policy._flatten_rules(
            domain, fit.root, exemplars
        )

        self.assertEqual(
            segmented_policy._evaluate_tree(fit.root, costs),
            segmented_policy._evaluate_rules(rules, costs),
        )

    def test_parallel_fold_and_reduction_are_identical_to_serial_fit(self) -> None:
        """Forked fitting and domain reduction may never change policy bytes."""

        rows = []
        for mode in (
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ):
            for source_format in ("Q4_0", "Q5_0"):
                for index, n in enumerate((128, 160, 192, 288, 320, 384)):
                    lower = n <= 192
                    for candidate, latency in (
                        ("candidate.low", 10.0 if lower else 14.0),
                        ("candidate.high", 14.0 if lower else 10.0),
                    ):
                        rows.append(observation(
                            source_format=source_format,
                            candidate=candidate,
                            family=candidate,
                            shape_group=(
                                f"parallel-cv-{mode.value}-{source_format}-"
                                f"{index}"
                            ),
                            n=n,
                            latency_us=latency,
                            mode=mode,
                        ))
        corpus = ObservationCorpus(rows)
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            serial = fit_generic_policy(corpus, max_leaves=2)
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "8"},
        ):
            parallel = fit_generic_policy(corpus, max_leaves=2)
        self.assertEqual(parallel, serial)
        self.assertTrue(all(
            validation.cells for validation in serial.cross_validation
        ))
        self.assertTrue(any(
            validation.competitive_cells
            for validation in serial.cross_validation
        ))

    def test_parallel_domain_fold_plans_match_serial_and_cap_workers(self) -> None:
        """Parallel task assembly preserves order and excludes SMT workers."""

        rows = []
        for mode in (
            ExecutionMode.EAGER,
            ExecutionMode.GRAPH_CAPTURED,
        ):
            for source_format in ("Q4_0", "Q5_0"):
                for index, n in enumerate((128, 160, 192, 288, 320, 384)):
                    for candidate, latency in (
                        ("candidate.low", 10.0),
                        ("candidate.high", 11.0),
                    ):
                        rows.append(observation(
                            source_format=source_format,
                            candidate=candidate,
                            family=candidate,
                            shape_group=(
                                f"parallel-plan-{mode.value}-{source_format}-"
                                f"{index}"
                            ),
                            n=n,
                            latency_us=latency,
                            mode=mode,
                        ))
        corpus = ObservationCorpus(rows)
        costs = build_candidate_point_costs(corpus)
        plan_tasks = tuple(
            (
                domain,
                costs[domain],
                2,
                2,
                "unit-parallel-domain-plan",
                None,
            )
            for domain in corpus.generic_domains()
        )
        serial, serial_workers = segmented_policy._construct_domain_fold_plans(
            plan_tasks,
            profiler_prediction_cache=None,
            requested_workers=1,
        )
        with mock.patch.object(
            segmented_policy,
            "_physical_core_worker_count",
            return_value=2,
        ):
            parallel, parallel_workers = (
                segmented_policy._construct_domain_fold_plans(
                    plan_tasks,
                    profiler_prediction_cache=None,
                    requested_workers=99,
                )
            )

        self.assertEqual(serial_workers, 1)
        self.assertEqual(parallel_workers, 2)
        self.assertEqual(parallel, serial)

    def test_fit_cache_retrains_only_the_domain_touched_by_paired_evidence(self) -> None:
        """A new tournament edge cannot force unrelated mode domains to refit."""

        rows = []
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
            for index, n in enumerate((128, 160, 192, 224, 256, 288)):
                for candidate, latency in (
                    ("candidate.low", 10.0),
                    ("candidate.high", 11.0),
                ):
                    rows.append(observation(
                        backend=Backend.CUDA,
                        contract=SemanticContract.FAST,
                        m=1,
                        mode=mode,
                        candidate=candidate,
                        family=candidate,
                        shape_group=f"cache-{mode.value}-{index}",
                        n=n,
                        latency_us=latency,
                    ))
        corpus = ObservationCorpus(rows)
        eager_exemplar = next(
            row
            for row in rows
            if row.execution_mode == ExecutionMode.EAGER
            and row.candidate_id == "candidate.low"
        )
        paired_key = PairedCellKey(
            backend=eager_exemplar.backend.value,
            source_format=eager_exemplar.source_format,
            source_codebook=eager_exemplar.source_codebook_id,
            execution_codebook=eager_exemplar.runtime_codebook_id,
            shape=eager_exemplar.shape_name,
            execution_mode=eager_exemplar.execution_mode.value,
            m=eager_exemplar.m,
            n=eager_exemplar.aggregate_n,
            k=eager_exemplar.k,
        )
        paired = paired_timing_comparisons((PairedCellEvidence(
            key=paired_key,
            selected_candidate_id="candidate.low",
            exact_candidate_id="candidate.high",
            selected_latency_us=(9.0,) * 30,
            exact_latency_us=(10.0,) * 30,
            selected_first_count=15,
            exact_first_count=15,
            selected_ran_first=tuple(index % 2 == 0 for index in range(30)),
        ),))

        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            with mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
            ):
                def domain_corpus(domain):
                    return ObservationCorpus._from_validated(
                        corpus.rows_for_generic_domain(domain)
                    )

                fit_generic_policy(
                    corpus,
                    max_leaves=1,
                    fit_final_rules=False,
                    fit_cache=cache,
                    domain_corpus_provider=domain_corpus,
                    domain_corpus_digest_provider=lambda domain: (
                        generic_domain_corpus_digest(corpus, domain)
                    ),
                )
                cost_cache_files = tuple(
                    (Path(directory) / "candidate-costs").glob("*.json")
                )
                self.assertEqual(len(cost_cache_files), 2)
                for path in cost_cache_files:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertIn("candidate_ids", payload)
                    self.assertIn("points", payload)
                    self.assertNotIn("costs", payload)

                evaluated_domains = []
                projected_domains = []
                original = segmented_policy._evaluate_placement_fold

                def record_domain(task, primary_scorer=None):
                    evaluated_domains.append(task[0])
                    return original(task, primary_scorer)

                def record_projection(domain):
                    projected_domains.append(domain)
                    return domain_corpus(domain)

                with mock.patch.object(
                    segmented_policy,
                    "_evaluate_placement_fold",
                    side_effect=record_domain,
                ):
                    paired_fit = fit_generic_policy(
                        corpus,
                        paired_comparisons=paired,
                        max_leaves=1,
                        fit_final_rules=False,
                        fit_cache=cache,
                        domain_corpus_provider=record_projection,
                        domain_corpus_digest_provider=lambda domain: (
                            generic_domain_corpus_digest(corpus, domain)
                        ),
                    )

                self.assertTrue(evaluated_domains)
                self.assertEqual(
                    {domain.execution_mode for domain in evaluated_domains},
                    {ExecutionMode.EAGER},
                )
                self.assertEqual(
                    {domain.execution_mode for domain in projected_domains},
                    {ExecutionMode.EAGER},
                )

                def fail_projection(_domain):
                    raise AssertionError("cached domain was projected")

                with mock.patch.object(
                    segmented_policy,
                    "_evaluate_placement_fold",
                    side_effect=AssertionError("cached domain was refitted"),
                ), mock.patch.object(
                    PolicyFitCache,
                    "load_costs",
                    side_effect=AssertionError("cached costs were deserialized"),
                ):
                    cached_fit = fit_generic_policy(
                        corpus,
                        paired_comparisons=paired,
                        max_leaves=1,
                        fit_final_rules=False,
                        fit_cache=cache,
                        domain_corpus_provider=fail_projection,
                        domain_corpus_digest_provider=lambda domain: (
                            generic_domain_corpus_digest(corpus, domain)
                        ),
                    )
                self.assertEqual(cached_fit, paired_fit)

    def test_larger_leaf_budget_refits_only_unpromoted_domains(self) -> None:
        """A split expansion must retain already-promotable one-leaf domains."""

        rows = []
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
            for index, n in enumerate((128, 160, 192, 288, 320, 384)):
                split_domain = mode == ExecutionMode.GRAPH_CAPTURED
                lower = n < 240
                rows.extend([
                    observation(
                        backend=Backend.CUDA,
                        contract=SemanticContract.FAST,
                        m=1,
                        mode=mode,
                        candidate="candidate.low",
                        family="low",
                        shape_group=f"leaf-cache-{mode.value}-{index}",
                        n=n,
                        latency_us=(
                            10.0
                            if not split_domain or lower
                            else 20.0
                        ),
                    ),
                    observation(
                        backend=Backend.CUDA,
                        contract=SemanticContract.FAST,
                        m=1,
                        mode=mode,
                        candidate="candidate.high",
                        family="high",
                        shape_group=f"leaf-cache-{mode.value}-{index}",
                        n=n,
                        latency_us=(
                            11.0
                            if not split_domain
                            else (20.0 if lower else 10.0)
                        ),
                    ),
                ])
        corpus = ObservationCorpus(rows)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            one_leaf = fit_generic_policy(
                corpus,
                max_leaves=1,
                fit_cache=cache,
            )
            self.assertEqual(
                {domain.execution_mode for domain in one_leaf.unpromoted_domains},
                {ExecutionMode.GRAPH_CAPTURED},
            )

            evaluated_domains = []
            original = segmented_policy._evaluate_placement_fold

            def record_domain(task, primary_scorer=None):
                evaluated_domains.append(task[0])
                return original(task, primary_scorer)

            with mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                side_effect=record_domain,
            ):
                two_leaf = fit_generic_policy(
                    corpus,
                    max_leaves=2,
                    fit_cache=cache,
                )

        self.assertTrue(evaluated_domains)
        self.assertEqual(
            {domain.execution_mode for domain in evaluated_domains},
            {ExecutionMode.GRAPH_CAPTURED},
        )
        self.assertFalse(two_leaf.unpromoted_domains)
        self.assertEqual(
            {
                validation.domain.execution_mode:
                validation.selected_max_leaves
                for validation in two_leaf.cross_validation
            },
            {
                ExecutionMode.EAGER: 1,
                ExecutionMode.GRAPH_CAPTURED: 2,
            },
        )

    def test_current_fit_cache_miss_never_hashes_full_profiler_provenance(
        self,
    ) -> None:
        """Current model keys must not rebuild the obsolete catalog identity."""

        rows = tuple(
            observation(
                candidate=candidate,
                family=candidate,
                shape_group=f"profiler-cache-{point}",
                n=128 + point * 32,
                latency_us=latency,
            )
            for point in range(6)
            for candidate, latency in (
                ("candidate.low", 10.0),
                ("candidate.high", 11.0),
            )
        )
        corpus = ObservationCorpus(rows)
        descriptors = {}
        for row in rows:
            key = _physical_key(row)
            descriptors[key] = ProfilerCandidateDescriptor(
                key=key,
                anchor_m=row.m,
                anchor_n=row.aggregate_n,
                anchor_k=row.k,
                features={},
            )
        catalog = ProfilerFeatureCatalog(
            corpus_digest=corpus.digest(),
            request_manifest_digest="sha256:requests",
            evidence_manifest_digest="sha256:evidence",
            descriptors=descriptors,
        )

        def reject_obsolete_digest(_catalog) -> str:
            raise AssertionError("full profiler provenance digest was requested")

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(
                os.environ,
                {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
            ),
            mock.patch.object(
                ProfilerFeatureCatalog,
                "digest",
                new=property(reject_obsolete_digest),
            ),
        ):
            policy = fit_generic_policy(
                corpus,
                max_leaves=1,
                fit_final_rules=False,
                policy_accelerators=(),
                fit_cache=PolicyFitCache(Path(directory)),
                profiler_feature_catalog=catalog,
            )

        self.assertEqual(len(policy.cross_validation), 1)

    def test_fit_cache_publishes_only_the_final_stable_cv_model(self) -> None:
        """Immutable cache keys cannot first receive a provisional CV winner."""

        rows = []
        for index, n in enumerate((128, 160, 192, 224, 256, 288)):
            for candidate, latency in (
                ("candidate.low", 10.0),
                ("candidate.high", 11.0),
            ):
                rows.append(observation(
                    backend=Backend.CPU,
                    contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                    mode=ExecutionMode.EAGER,
                    m=64,
                    candidate=candidate,
                    family=candidate,
                    shape_group=f"final-cache-{index}",
                    n=n,
                    latency_us=latency,
                ))
        corpus = ObservationCorpus(rows)
        with mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            baseline = fit_generic_policy(corpus, max_leaves=1)
        selected = baseline.cross_validation[0]
        alternate = dataclasses.replace(
            selected,
            selected_feature_policy=FeaturePolicy.TILE_32,
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ), mock.patch.object(
            segmented_policy,
            "_rank_domain_cross_validations",
            return_value=(selected, alternate),
        ), mock.patch.object(
            segmented_policy,
            "_fit_final_domain",
            return_value=(
                selected.domain,
                baseline.rules,
                alternate,
            ),
        ) as final_fit:
            cache = PolicyFitCache(Path(directory))
            fitted = fit_generic_policy(
                corpus,
                max_leaves=1,
                fit_cache=cache,
            )

            cache_files = tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )
            self.assertEqual(len(cache_files), 1)
            cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
            final_cache_files = tuple(
                (Path(directory) / "domain-final-fit").glob("*.json")
            )
            self.assertEqual(len(final_cache_files), 1)
            cached_final = json.loads(
                final_cache_files[0].read_text(encoding="utf-8")
            )

            final_fit.reset_mock()
            final_fit.side_effect = AssertionError(
                "stable final publication tree was refitted"
            )
            with mock.patch.object(
                PolicyFitCache,
                "load_costs",
                side_effect=AssertionError(
                    "stable final publication costs were deserialized"
                ),
            ):
                replay = fit_generic_policy(
                    corpus,
                    max_leaves=1,
                    fit_cache=cache,
                )

        self.assertEqual(
            fitted.cross_validation[0].selected_feature_policy,
            FeaturePolicy.TILE_32,
        )
        self.assertEqual(
            cached["validation"]["selected_feature_policy"],
            selected.selected_feature_policy.value,
        )
        self.assertEqual(
            cached_final["validation"]["selected_feature_policy"],
            FeaturePolicy.TILE_32.value,
        )
        self.assertIsNone(cached["final_rules"])
        self.assertEqual(
            len(cached_final["final_rules"]), len(baseline.rules)
        )
        self.assertEqual(replay, fitted)

    def test_fit_cache_identity_changes_only_with_its_domain_rows(self) -> None:
        """Appending targeted evidence must preserve unrelated cache keys."""

        rows = []
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
            for candidate, latency in (
                ("candidate.low", 10.0),
                ("candidate.high", 11.0),
            ):
                rows.append(observation(
                    backend=Backend.CUDA,
                    contract=SemanticContract.FAST,
                    mode=mode,
                    candidate=candidate,
                    family=candidate,
                    shape_group=f"domain-digest-{mode.value}",
                    n=256,
                    latency_us=latency,
                ))
        baseline = ObservationCorpus(rows)
        extension = [
            observation(
                backend=Backend.CUDA,
                contract=SemanticContract.FAST,
                mode=ExecutionMode.GRAPH_CAPTURED,
                candidate=candidate,
                family=candidate,
                shape_group="domain-digest-graph-extension",
                n=320,
                latency_us=latency,
            )
            for candidate, latency in (
                ("candidate.low", 12.0),
                ("candidate.high", 10.0),
            )
        ]
        extended = ObservationCorpus((*rows, *extension))
        domains = {
            domain.execution_mode: domain
            for domain in baseline.generic_domains()
        }
        cache = PolicyFitCache(Path("unused-domain-cache"))

        eager = domains[ExecutionMode.EAGER]
        eager_before = generic_domain_corpus_digest(baseline, eager)
        eager_after = generic_domain_corpus_digest(extended, eager)
        self.assertEqual(eager_after, eager_before)
        self.assertEqual(
            cache.cost_key(
                eager, eager_before, "sha256:paired", "sha256:serial"
            ),
            cache.cost_key(
                eager, eager_after, "sha256:paired", "sha256:serial"
            ),
        )

        graph = domains[ExecutionMode.GRAPH_CAPTURED]
        graph_before = generic_domain_corpus_digest(baseline, graph)
        graph_after = generic_domain_corpus_digest(extended, graph)
        self.assertNotEqual(graph_after, graph_before)
        self.assertNotEqual(
            cache.cost_key(
                graph, graph_before, "sha256:paired", "sha256:serial"
            ),
            cache.cost_key(
                graph, graph_after, "sha256:paired", "sha256:serial"
            ),
        )

    def test_fit_cache_domain_digest_ignores_global_corpus_identity(self) -> None:
        """Aggregate relabeling must neither recompute nor refit one domain."""

        rows = []
        for point, n in enumerate((256, 320, 384, 448)):
            for candidate, latency in (
                ("candidate.low", 10.0),
                ("candidate.high", 11.0),
            ):
                rows.append(observation(
                    backend=Backend.CUDA,
                    contract=SemanticContract.FAST,
                    mode=ExecutionMode.EAGER,
                    candidate=candidate,
                    family=candidate,
                    shape_group=f"corpus-label-cache-{point}",
                    n=n,
                    latency_us=latency,
                ))
        original = ObservationCorpus(rows)
        relabeled = ObservationCorpus(
            dataclasses.replace(
                row,
                corpus_id="sha256:larger-additive-aggregate",
            )
            for row in rows
        )
        domain = original.generic_domains()[0]
        self.assertEqual(
            generic_domain_corpus_digest(original, domain),
            generic_domain_corpus_digest(relabeled, domain),
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            # Reproduce the aggregate-coupled key written by the first v15
            # implementation, then prove both migration and future aggregate
            # relabeling are lookup-only operations.
            with mock.patch.object(
                segmented_policy,
                "generic_domain_corpus_digest",
                side_effect=segmented_policy._legacy_generic_domain_corpus_digest,
            ):
                expected = fit_generic_policy(
                    original,
                    max_leaves=1,
                    fit_cache=cache,
                )

            with mock.patch.object(
                segmented_policy,
                "build_domain_candidate_point_costs",
                side_effect=AssertionError("legacy costs were recomputed"),
            ), mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                side_effect=AssertionError("legacy validation was recomputed"),
            ):
                migrated = fit_generic_policy(
                    original,
                    max_leaves=1,
                    fit_cache=cache,
                )
                relabeled_fit = fit_generic_policy(
                    relabeled,
                    max_leaves=1,
                    fit_cache=cache,
                )

            self.assertEqual(migrated, expected)
            self.assertEqual(relabeled_fit, expected)
            # A cached final tree makes the candidate matrix unnecessary.
            # Leave its old generation in place and migrate it lazily only if
            # a later affected-pool refit actually needs the costs.
            self.assertEqual(len(tuple(
                (Path(directory) / "candidate-costs").glob("*.json")
            )), 1)
            self.assertEqual(len(tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )), 2)

    def test_fit_cache_serial_hash_identity_is_domain_local(self) -> None:
        """Changing one domain's serial oracle must not evict another domain."""

        rows = []
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
            for point, n in enumerate((256, 320, 384)):
                for candidate, latency in (
                    ("candidate.low", 10.0),
                    ("candidate.high", 11.0),
                ):
                    rows.append(observation(
                        backend=Backend.CUDA,
                        contract=SemanticContract.FAST,
                        mode=mode,
                        candidate=candidate,
                        family=candidate,
                        shape_group=f"serial-cache-{mode.value}-{point}",
                        n=n,
                        latency_us=latency,
                    ))
        corpus = ObservationCorpus(rows)
        hashes = {
            corpus.runtime_key_for(row): SERIAL_HASH for row in rows
        }
        graph_keys = {
            corpus.runtime_key_for(row)
            for row in rows
            if row.execution_mode == ExecutionMode.GRAPH_CAPTURED
        }
        changed_hashes = {
            key: (
                "sha256:changed-graph-serial"
                if key in graph_keys
                else policy_hash
            )
            for key, policy_hash in hashes.items()
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            fit_generic_policy(
                corpus,
                serial_m1_hashes=hashes,
                max_leaves=1,
                fit_final_rules=False,
                fit_cache=cache,
            )
            initial_costs = len(tuple(
                (Path(directory) / "candidate-costs").glob("*.json")
            ))
            initial_validations = len(tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            ))

            fit_generic_policy(
                corpus,
                serial_m1_hashes=changed_hashes,
                max_leaves=1,
                fit_final_rules=False,
                fit_cache=cache,
            )

            self.assertEqual(initial_costs, 2)
            self.assertEqual(initial_validations, 2)
            self.assertEqual(len(tuple(
                (Path(directory) / "candidate-costs").glob("*.json")
            )), initial_costs + 1)
            self.assertEqual(len(tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )), initial_validations + 1)

    def test_fit_cache_cv_identity_includes_profiler_transfer_pool(self) -> None:
        """Cross-format profiler training changes invalidate dependent CV."""

        domain = generic_domain(observation())
        cache = PolicyFitCache(Path("unused-profiler-pool-cache"))
        arguments = {
            "max_leaves": 4,
            "min_shape_groups_per_leaf": 2,
            "cross_validation_seed": "unit-profiler-pool",
            "profiler_feature_catalog_digest": "sha256:catalog",
            "fit_final_rules": True,
        }
        baseline = cache.validation_key(
            domain,
            "cost-key",
            profiler_training_pool_digest="sha256:pool-a",
            **arguments,
        )
        changed = cache.validation_key(
            domain,
            "cost-key",
            profiler_training_pool_digest="sha256:pool-b",
            **arguments,
        )
        legacy = cache.validation_key(
            domain,
            "cost-key",
            **arguments,
        )

        self.assertNotEqual(changed, baseline)
        self.assertNotEqual(legacy, baseline)

    def test_additive_feature_policy_scores_only_new_family(self) -> None:
        """A cached prior tournament must not replay unchanged tree families."""

        rows = []
        for point, n in enumerate((128, 160, 192, 224, 256, 288)):
            rows.extend((
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=f"incremental-policy-{point}",
                    n=n,
                    latency_us=10.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=f"incremental-policy-{point}",
                    n=n,
                    latency_us=11.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        added_policy = FeaturePolicy.KPART_CHUNK_GRID_SCHEDULES
        predecessor_policies = tuple(
            policy
            for policy in segmented_policy.FEATURE_POLICIES
            if policy != added_policy
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            with mock.patch.object(
                segmented_policy,
                "FEATURE_POLICIES",
                predecessor_policies,
            ):
                fit_generic_policy(
                    corpus,
                    max_leaves=2,
                    fit_final_rules=False,
                    policy_accelerators=(),
                    fit_cache=cache,
                )

            with mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                wraps=segmented_policy._evaluate_placement_fold,
            ) as evaluate:
                incremental = fit_generic_policy(
                    corpus,
                    max_leaves=2,
                    fit_final_rules=False,
                    policy_accelerators=(),
                    fit_cache=cache,
                )
            fresh = fit_generic_policy(
                corpus,
                max_leaves=2,
                fit_final_rules=False,
                policy_accelerators=(),
            )

        self.assertTrue(evaluate.call_args_list)
        self.assertEqual(
            {
                call.args[0][3]
                for call in evaluate.call_args_list
            },
            {added_policy},
        )
        self.assertEqual(
            segmented_policy._cross_validation_mapping(
                incremental.cross_validation[0]
            ),
            segmented_policy._cross_validation_mapping(
                fresh.cross_validation[0]
            ),
        )

    def test_one_leaf_cv_removes_predicate_only_tournament_duplicates(
        self,
    ) -> None:
        """An unsplit tree must score one canonical feature/threshold pair."""

        rows = []
        for point, n in enumerate((128, 160, 192, 224, 256, 288)):
            rows.extend((
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=f"one-leaf-{point}",
                    n=n,
                    latency_us=10.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=f"one-leaf-{point}",
                    n=n,
                    latency_us=11.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        domain = generic_domain(rows[0])
        costs = build_candidate_point_costs(corpus)[domain]

        one_leaf_tasks, fold_count = segmented_policy._domain_fold_tasks(
            domain,
            costs,
            max_leaves=1,
            min_shape_groups_per_leaf=2,
            seed="one-leaf-dedup",
        )
        split_tasks, split_fold_count = segmented_policy._domain_fold_tasks(
            domain,
            costs,
            max_leaves=2,
            min_shape_groups_per_leaf=2,
            seed="one-leaf-dedup",
        )

        self.assertEqual(fold_count, split_fold_count)
        self.assertEqual(len(one_leaf_tasks), fold_count)
        self.assertEqual(
            {(task[3], task[4]) for task in one_leaf_tasks},
            {(FeaturePolicy.CONTINUOUS, BoundaryPlacement.MIDPOINT)},
        )
        self.assertEqual(
            len(split_tasks),
            fold_count
            * len(segmented_policy.FEATURE_POLICIES)
            * len(segmented_policy.BOUNDARY_PLACEMENTS),
        )

    def test_parallel_profiler_prediction_cache_keys_match_serial(self) -> None:
        """Forked cache lookup preserves every persistent surface identity."""

        exemplar = observation()
        runtime = runtime_key(exemplar)
        pool_key = ("cpu", "unit-pool")
        request_keys = tuple(
            (pool_key, ((128 + index * 32, 4096),))
            for index in range(4)
        )
        prediction_points = {
            request_key: frozenset((
                (runtime, f"shape-{point}", f"candidate-{point}"),
            ))
            for point, request_key in enumerate(request_keys)
        }
        training_digests = {pool_key: "sha256:training"}
        model_digests = {pool_key: "sha256:model"}

        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            serial_cache = {}
            serial_keys, serial_workers, serial_hits = (
                segmented_policy._load_persistent_profiler_prediction_cache(
                    request_keys,
                    prediction_points,
                    serial_cache,
                    cache,
                    training_digests,
                    model_digests,
                    requested_workers=1,
                )
            )
            for request_index, request_key in enumerate(request_keys):
                points = prediction_points[request_key]
                cache.store_profiler_predictions(
                    serial_keys[request_key],
                    training_pool_digest=training_digests[pool_key],
                    profiler_model_digest=model_digests[pool_key],
                    held_out_geometries=request_key[1],
                    predictions={
                        point: float(request_index) for point in points
                    },
                    prediction_points=points,
                )
            with mock.patch.object(
                segmented_policy,
                "_physical_core_worker_count",
                return_value=2,
            ):
                parallel_cache = {}
                parallel_keys, parallel_workers, parallel_hits = (
                    segmented_policy._load_persistent_profiler_prediction_cache(
                        request_keys,
                        prediction_points,
                        parallel_cache,
                        cache,
                        training_digests,
                        model_digests,
                        requested_workers=99,
                    )
                )

        self.assertEqual(serial_workers, 1)
        self.assertEqual(parallel_workers, 2)
        self.assertEqual(serial_hits, 0)
        self.assertEqual(parallel_hits, len(request_keys))
        self.assertEqual(parallel_keys, serial_keys)
        self.assertEqual(len(parallel_cache), len(request_keys))
        for request_index, request_key in enumerate(request_keys):
            point = next(iter(prediction_points[request_key]))
            self.assertEqual(
                parallel_cache[request_key][point],
                float(request_index),
            )

    def test_prediction_inventory_reuses_exact_runtime_json_fragments(self) -> None:
        """Every runtime discriminator survives amortized canonical serialization."""

        base = runtime_key(observation())
        mutations = {
            "backend": Backend.CUDA,
            "architecture_class": 'arch-"\\\n\u2603',
            "semantic_contract": SemanticContract.FAST,
            "operation_kind": "operation-\t\u00e9",
            "bundle_signature": "bundle-\r\ud800",
            "projection_n_vector": (33, 65),
            "prepared_family_id": "another-prepared-family",
            "packing_abi": "another-packing-abi",
            "runtime_codebook_id": base.runtime_codebook_id + 1,
            "execution_mode": ExecutionMode.EAGER,
            "m": base.m + 1,
            "aggregate_n": base.aggregate_n + 1,
            "k": base.k + 1,
            "launch_k_tiles": base.launch_k_tiles + 1,
        }
        self.assertEqual(set(mutations), {field.name for field in dataclasses.fields(base)})
        keys = (base, *(dataclasses.replace(base, **{name: value})
                        for name, value in mutations.items()))
        names = ("", 'escaped-"\\\n', "unicode-\u2603-\ud800")
        points = frozenset((key, shape, candidate)
                           for key in keys for shape in names for candidate in names)
        expected = hashlib.sha256()
        for key, shape, candidate in sorted(points):
            # Independent historical encoder: do not build the oracle with the
            # fragment implementation whose byte identity is being verified.
            encoded = json.dumps({
                "runtime_key": segmented_policy._runtime_key_mapping(key),
                "shape_group_id": shape,
                "candidate_id": candidate,
            }, sort_keys=True, separators=(",", ":")).encode()
            expected.update(len(encoded).to_bytes(8, byteorder="little"))
            expected.update(encoded)
        segmented_policy._profiler_prediction_point_inventory.cache_clear()
        with mock.patch.object(segmented_policy, "_runtime_key_mapping",
                               wraps=segmented_policy._runtime_key_mapping) as mapping, \
             mock.patch.object(type(base), "__lt__",
                               side_effect=AssertionError("inventory used slow dataclass ordering")):
            actual = segmented_policy._profiler_prediction_point_inventory(points)
        self.assertEqual(actual.points, tuple(sorted(points)))
        self.assertEqual(actual.digest, "sha256:" + expected.hexdigest())
        self.assertEqual(mapping.call_count, len(keys),
                         "candidate aliases must not re-encode their runtime key")
        empty = segmented_policy._profiler_prediction_point_inventory(frozenset())
        self.assertEqual(empty.points, ())
        self.assertEqual(empty.digest, "sha256:" + hashlib.sha256().hexdigest())
        segmented_policy._profiler_prediction_point_inventory.cache_clear()

    def test_memmap_teacher_totality_is_vectorized_and_exact(self) -> None:
        """Batched memmap checks reject missing and non-finite predictions."""

        runtime = runtime_key(observation())
        costs = [
            segmented_policy.CandidatePointCost(
                runtime_key=runtime,
                shape_group_id=f"teacher-shape-{index}",
                candidate_id=f"teacher-candidate-{index}",
                max_surface_regret=0.0,
                p95_surface_regret=0.0,
                mean_surface_regret=0.0,
            )
            for index in range(3)
        ]
        points = frozenset(
            (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
            for cost in costs
        )
        inventory = segmented_policy._profiler_prediction_point_inventory(
            points
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            finite_path = directory_path / "finite.f64"
            np.asarray(
                [1.0] * len(inventory.points), dtype="<f8"
            ).tofile(finite_path)
            finite = segmented_policy.ProfilerPredictionSurface(
                inventory, finite_path
            )

            nonfinite_path = directory_path / "nonfinite.f64"
            nonfinite_values = np.asarray(
                [1.0] * len(inventory.points), dtype="<f8"
            )
            nonfinite_values[1] = np.nan
            nonfinite_values.tofile(nonfinite_path)
            nonfinite = segmented_policy.ProfilerPredictionSurface(
                inventory, nonfinite_path
            )

            missing_inventory = (
                segmented_policy._profiler_prediction_point_inventory(
                    frozenset(tuple(points)[:-1])
                )
            )
            missing_path = directory_path / "missing.f64"
            np.asarray(
                [1.0] * len(missing_inventory.points), dtype="<f8"
            ).tofile(missing_path)
            missing = segmented_policy.ProfilerPredictionSurface(
                missing_inventory, missing_path
            )

            self.assertTrue(
                segmented_policy._profiler_teacher_surface_is_complete(
                    costs, finite
                )
            )
            self.assertFalse(
                segmented_policy._profiler_teacher_surface_is_complete(
                    costs, nonfinite
                )
            )
            self.assertFalse(
                segmented_policy._profiler_teacher_surface_is_complete(
                    costs, missing
                )
            )

    def test_fit_cache_shares_cv_identity_between_planning_and_publication(self) -> None:
        """Publication is a derivative of planning CV, not a second search."""

        domain = generic_domain(observation())
        cache = PolicyFitCache(Path("unused-selection-mode-cache"))
        arguments = {
            "max_leaves": 4,
            "min_shape_groups_per_leaf": 2,
            "cross_validation_seed": "unit-selection-mode",
            "profiler_feature_catalog_digest": None,
        }

        production = cache.validation_key(
            domain,
            "cost-key",
            fit_final_rules=True,
            **arguments,
        )
        planning = cache.validation_key(
            domain,
            "cost-key",
            fit_final_rules=False,
            **arguments,
        )

        self.assertEqual(production, planning)

    def test_planning_cache_satisfies_production_cv_without_refitting(self) -> None:
        """Freeze fits final leaves while reusing every certified CV fold."""

        rows = []
        for point, n in enumerate((128, 160, 192, 224, 256, 288)):
            for candidate, latency in (
                ("candidate.low", 10.0),
                ("candidate.high", 11.0),
            ):
                rows.append(observation(
                    candidate=candidate,
                    family=candidate,
                    shape_group=f"cache-mode-{point}",
                    n=n,
                    latency_us=latency,
                ))
        corpus = ObservationCorpus(rows)
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            fit_generic_policy(
                corpus,
                max_leaves=1,
                fit_final_rules=False,
                fit_cache=cache,
            )
            planning_records = tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )
            with mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                side_effect=AssertionError("certified CV was recomputed"),
            ), mock.patch.object(
                segmented_policy,
                "_populate_profiler_prediction_cache",
                side_effect=AssertionError("cached CV rebuilt profiler models"),
            ):
                production = fit_generic_policy(
                    corpus,
                    max_leaves=1,
                    fit_final_rules=True,
                    fit_cache=cache,
                )
            production_records = tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )
            final_records = tuple(
                (Path(directory) / "domain-final-fit").glob("*.json")
            )

            with mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                side_effect=AssertionError("certified CV was recomputed"),
            ), mock.patch.object(
                segmented_policy,
                "_fit_final_domain",
                side_effect=AssertionError("final leaves were recomputed"),
            ), mock.patch.object(
                PolicyFitCache,
                "load_costs",
                side_effect=AssertionError("cached final fit loaded costs"),
            ):
                replay = fit_generic_policy(
                    corpus,
                    max_leaves=1,
                    fit_final_rules=True,
                    fit_cache=cache,
                )

        self.assertEqual(len(planning_records), 1)
        self.assertEqual(len(production_records), 1)
        self.assertEqual(len(final_records), 1)
        self.assertTrue(production.cross_validation[0].cells)
        self.assertEqual(replay, production)

    def test_fit_cache_promotes_legacy_global_serial_hash_keys(self) -> None:
        """Current v6 global-key records migrate without recomputing policy."""

        rows = []
        for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED):
            for point, n in enumerate((256, 320, 384)):
                for candidate, latency in (
                    ("candidate.low", 10.0),
                    ("candidate.high", 11.0),
                ):
                    rows.append(observation(
                        backend=Backend.CUDA,
                        contract=SemanticContract.FAST,
                        mode=mode,
                        candidate=candidate,
                        family=candidate,
                        shape_group=f"legacy-cache-{mode.value}-{point}",
                        n=n,
                        latency_us=latency,
                    ))
        corpus = ObservationCorpus(rows)
        hashes = {
            corpus.runtime_key_for(row): SERIAL_HASH for row in rows
        }
        digest = segmented_policy._serial_m1_hash_digest

        def legacy_global_digest(serial_hashes, runtime_keys=None):
            del runtime_keys
            return digest(serial_hashes)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LLAMINAR_NATIVE_VNNI_POLICY_WORKERS": "1"},
        ):
            cache = PolicyFitCache(Path(directory))
            with mock.patch.object(
                segmented_policy,
                "_serial_m1_hash_digest",
                side_effect=legacy_global_digest,
            ):
                expected = fit_generic_policy(
                    corpus,
                    serial_m1_hashes=hashes,
                    max_leaves=1,
                    fit_cache=cache,
                )

            with mock.patch.object(
                segmented_policy,
                "build_domain_candidate_point_costs",
                side_effect=AssertionError("legacy costs were recomputed"),
            ), mock.patch.object(
                segmented_policy,
                "_evaluate_placement_fold",
                side_effect=AssertionError("legacy validation was recomputed"),
            ):
                migrated = fit_generic_policy(
                    corpus,
                    serial_m1_hashes=hashes,
                    max_leaves=1,
                    fit_cache=cache,
                )

            self.assertEqual(migrated, expected)
            # Both domains reuse their stable final trees, so their legacy
            # candidate matrices remain lazy instead of being deserialized
            # merely to republish an otherwise unused cache generation.
            self.assertEqual(len(tuple(
                (Path(directory) / "candidate-costs").glob("*.json")
            )), 2)
            self.assertEqual(len(tuple(
                (Path(directory) / "domain-cross-validation").glob("*.json")
            )), 4)

    def test_fit_cache_rejects_identity_mismatch_instead_of_refitting(self) -> None:
        """A corrupt content-addressed entry is fatal, never a quiet cache miss."""

        exemplar = observation()
        domain = generic_domain(exemplar)
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            content_key = cache.cost_key(
                domain,
                "sha256:" + "1" * 64,
                "sha256:unit-paired",
                "sha256:unit-serial",
            )
            path = cache._path("candidate-costs", content_key)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "schema_version": (
                    segmented_policy.POLICY_FIT_CACHE_SCHEMA_VERSION
                ),
                "kind": "candidate-costs",
                "content_key": "wrong-content-key",
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cache.load_costs(content_key, domain)

    def test_fit_cache_ignores_obsolete_schema_during_domain_scan(self) -> None:
        """Derived v13 state must not block a resumable v14 corpus refit."""

        exemplar = observation()
        domain = generic_domain(exemplar)
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            path = cache._path("domain-cross-validation", "1" * 64)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "schema_version": "native-vnni-policy-fit-cache-v13",
                "kind": "domain-cross-validation",
                "content_key": "1" * 64,
                "domain": segmented_policy._generic_domain_mapping(domain),
                "validation": None,
                "final_rules": None,
            }), encoding="utf-8")

            self.assertFalse(cache.has_validation_entry_for_domain(domain))
            self.assertTrue(path.exists(), "obsolete derived state is retained")

    def test_fit_cache_indexes_validation_domains_once_and_tracks_writes(
        self,
    ) -> None:
        """Legacy migration lookup stays linear and sees same-run writes."""

        exemplar = observation()
        first = generic_domain(exemplar)
        second = dataclasses.replace(first, m=first.m + 1)
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            first_key = "1" * 64
            cache.store_validation(first_key, first, None)

            self.assertTrue(cache.has_validation_entry_for_domain(first))
            index = cache._validation_domain_index
            self.assertFalse(cache.has_validation_entry_for_domain(second))
            self.assertIs(cache._validation_domain_index, index)

            second_key = "2" * 64
            cache.store_validation(second_key, second, None)
            self.assertTrue(cache.has_validation_entry_for_domain(second))
            self.assertIs(cache._validation_domain_index, index)

    def test_equivalent_dimension_cuts_prefer_k_partition_axis(self) -> None:
        """Correlated N must not hide the K dimension governing reduction."""

        rows = []
        lower_ns = (128, 160, 192)
        upper_ns = (288, 320, 384)
        for index, n in enumerate((*lower_ns, *upper_ns)):
            lower = n in lower_ns
            rows.extend([
                observation(
                    candidate="candidate.low-k",
                    family="low-k",
                    shape_group=f"correlated-k-{index}",
                    n=n,
                    k=n * 2,
                    latency_us=10.0 if lower else 14.0,
                ),
                observation(
                    candidate="candidate.high-k",
                    family="high-k",
                    shape_group=f"correlated-k-{index}",
                    n=n,
                    k=n * 2,
                    latency_us=14.0 if lower else 10.0,
                ),
            ])

        policy = fit_generic_policy(ObservationCorpus(rows), max_leaves=2)

        self.assertEqual(len(policy.rules), 2)
        predicates = {
            predicate
            for rule in policy.rules
            for predicate in rule.predicates
        }
        self.assertEqual(len(predicates), 2)
        self.assertTrue(
            all(
                predicate.threshold.axis == FeatureAxis.K
                for predicate in predicates
            )
        )

    def test_shape_group_split_never_leaks_candidate_alias_or_m_rows(self) -> None:
        rows = []
        for group in ("development-a", "development-b", "sealed-c"):
            for m in (2, 3, 4):
                for source_format in ("Q4_1", "Q4_K"):
                    rows.append(observation(
                        source_format=source_format,
                        candidate="candidate.a",
                        shape_group=group,
                        m=m,
                    ))
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "development-a": Partition.DEVELOPMENT,
                "development-b": Partition.DEVELOPMENT,
                "sealed-c": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        self.assertEqual(set(development.shape_groups()), {"development-a", "development-b"})
        self.assertFalse(sealed.opened)
        opened = sealed.open("sha256:frozen-generic")
        self.assertEqual(opened.shape_groups(), ("sealed-c",))

    def test_compile_freezes_generic_before_sealed_and_does_not_refit(self) -> None:
        rows = []
        for group, n in (
            ("dev-a", 256),
            ("dev-b", 384),
            ("dev-c", 448),
            ("sealed", 512),
        ):
            rows.extend([
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=group,
                    n=n,
                    latency_us=10.0,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=group,
                    n=n,
                    latency_us=11.0,
                ),
            ])
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "dev-a": Partition.DEVELOPMENT,
                "dev-b": Partition.DEVELOPMENT,
                "dev-c": Partition.DEVELOPMENT,
                "sealed": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        compiled = compile_policy(development, sealed)
        frozen = compiled.certification.frozen_generic_policy_digest
        self.assertEqual(compiled.policy_ir.digest(generic_only=True), frozen)
        self.assertEqual(compiled.certification.coverage, 1.0)
        self.assertEqual(compiled.certification.max_observed_regret, 0.0)

    def test_explicit_freeze_boundary_accepts_no_sealed_rows(self) -> None:
        """The two-process API binds a digest before certification can run."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(
                candidate="candidate.b", shape_group="dev-a", n=256,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(
                candidate="candidate.b", shape_group="dev-b", n=384,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(
                candidate="candidate.b", shape_group="dev-c", n=448,
                latency_us=11.0,
            ),
        ])
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(
                candidate="candidate.b", shape_group="sealed", n=512,
                latency_us=11.0,
            ),
        ])

        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        before = frozen.generic_digest
        compiled = certify_frozen_policy(frozen, development, sealed)

        self.assertEqual(compiled.certification.frozen_generic_policy_digest, before)
        self.assertEqual(compiled.policy_ir.digest(generic_only=True), before)
        self.assertEqual(compiled.certification.coverage, 1.0)

    def test_certification_honors_collapsed_aspect_domain_projection(self) -> None:
        """Sealed resolution must use the same cross-aspect view as fitting."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(
                candidate="candidate.b", shape_group="dev-a", n=256,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(
                candidate="candidate.b", shape_group="dev-b", n=384,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(
                candidate="candidate.b", shape_group="dev-c", n=448,
                latency_us=11.0,
            ),
        ]).with_collapsed_aspect_domains()
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(
                candidate="candidate.b", shape_group="sealed", n=512,
                latency_us=11.0,
            ),
        ]).with_collapsed_aspect_domains()

        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        compiled = certify_frozen_policy(frozen, development, sealed)

        self.assertEqual(compiled.certification.coverage, 1.0)
        self.assertEqual(compiled.certification.unexercised_rule_count, 0)
        self.assertFalse(compiled.policy_ir.metadata["sealed_max_regret"])

    def test_unpromoted_domains_remain_diagnostic_but_block_certification(self) -> None:
        """Exact overlays cannot waive mandatory generic dispatch coverage."""

        development_rows = []
        for group, n in (("dev-a", 256), ("dev-b", 384), ("dev-c", 448)):
            development_rows.extend((
                observation(candidate="candidate.a", shape_group=group, n=n),
                observation(
                    candidate="candidate.b", shape_group=group, n=n,
                    latency_us=11.0,
                ),
            ))
        development_rows.extend((
            observation(
                candidate="candidate.a", shape_group="exact-only-dev", m=3,
            ),
            observation(
                candidate="candidate.b", shape_group="exact-only-dev", m=3,
                latency_us=11.0,
            ),
        ))
        development = ObservationCorpus(development_rows)
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="generic-sealed", n=512),
            observation(
                candidate="candidate.b", shape_group="generic-sealed", n=512,
                latency_us=11.0,
            ),
            observation(
                candidate="candidate.a", shape_group="exact-only-sealed", m=3,
            ),
            observation(
                candidate="candidate.b", shape_group="exact-only-sealed", m=3,
                latency_us=11.0,
            ),
        ])

        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        compiled = certify_frozen_policy(
            frozen,
            development,
            sealed,
            require_promotable=False,
        )
        report = compiled.certification

        self.assertEqual(report.sealed_cell_count, 2)
        self.assertEqual(report.required_cell_count, 1)
        self.assertEqual(report.out_of_scope_cell_count, 1)
        self.assertEqual(report.covered_cell_count, 1)
        self.assertEqual(report.unpromoted_domain_count, 1)
        with self.assertRaisesRegex(ValueError, "lack generic scope"):
            report.require_promotable()
        self.assertEqual(report.coverage, 1.0)

    def test_installable_artifact_rejects_every_certificate_gate_failure(self) -> None:
        """No backend may publish a coverage-only or over-budget policy."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(candidate="candidate.b", shape_group="dev-a", n=256, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(candidate="candidate.b", shape_group="dev-b", n=384, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(candidate="candidate.b", shape_group="dev-c", n=448, latency_us=11.0),
        ])
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(candidate="candidate.b", shape_group="sealed", n=512, latency_us=11.0),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        compiled = certify_frozen_policy(frozen, development, sealed)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path = root / "policy.json"
            include_path = root / "policy.inc"
            write_compiled_policy(policy_path, compiled)
            include_path.write_text(
                f"// Common policy digest: {compiled.policy_ir.digest()}\n"
                f"// Frozen generic policy digest: {frozen.generic_digest}\n",
                encoding="utf-8",
            )
            validate_installable_policy_artifact(
                policy_path,
                include_path=include_path,
            )

            pristine = json.loads(policy_path.read_text(encoding="utf-8"))
            diagnostic_outlier = json.loads(json.dumps(pristine))
            diagnostic_outlier["certification"]["max_observed_regret"] = 0.50
            diagnostic_outlier["certification"][
                "max_simultaneous_95pct_upper_regret"
            ] = 0.60
            diagnostic_outlier["certification"]["p95_observed_regret"] = 0.50
            diagnostic_outlier["certification"][
                "p95_simultaneous_95pct_upper_regret"
            ] = 0.60
            policy_path.write_text(
                json.dumps(diagnostic_outlier),
                encoding="utf-8",
            )
            validate_installable_policy_artifact(
                policy_path,
                include_path=include_path,
            )

            failures = {
                "coverage": ("covered_cell_count", 0),
                "byte equality": ("verifier_bitwise_failures", 1),
                "rule exercise": ("unexercised_rule_count", 1),
                "generic totality": ("unpromoted_domain_count", 1),
                "domain count": ("required_domain_count", 0),
                "passing domain count": ("passing_domain_count", 0),
                "passing domain fraction": ("passing_domain_fraction", 0.0),
                "domain quota": ("domain_promotion_quota_satisfied", False),
            }
            for label, (field, value) in failures.items():
                with self.subTest(label=label):
                    poisoned = json.loads(json.dumps(pristine))
                    poisoned["certification"][field] = value
                    policy_path.write_text(
                        json.dumps(poisoned),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        validate_installable_policy_artifact(
                            policy_path,
                            include_path=include_path,
                        )

            failed_domain = json.loads(json.dumps(pristine))
            result = failed_domain["certification"]["domain_results"][0]
            result["p95_observed_regret"] = P95_REGRET_BUDGET
            result["passes_p95_budget"] = False
            failed_domain["certification"]["passing_domain_count"] = 0
            failed_domain["certification"]["passing_domain_fraction"] = 0.0
            failed_domain["certification"][
                "domain_promotion_quota_satisfied"
            ] = False
            policy_path.write_text(json.dumps(failed_domain), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not reach 95%"):
                validate_installable_policy_artifact(
                    policy_path,
                    include_path=include_path,
                )

    def test_best_effort_artifact_retains_misses_and_uses_frozen_quota(self) -> None:
        """A manual zero quota installs measured trees without hiding regret."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(candidate="candidate.b", shape_group="dev-a", n=256, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(candidate="candidate.b", shape_group="dev-b", n=384, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(candidate="candidate.b", shape_group="dev-c", n=448, latency_us=11.0),
        ])
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(candidate="candidate.b", shape_group="sealed", n=512, latency_us=11.0),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        compiled = certify_frozen_policy(frozen, development, sealed)
        failed_cells = tuple(
            dataclasses.replace(
                cell,
                observed_worst_surface_regret=0.50,
                simultaneous_95pct_upper_regret=0.60,
            )
            for cell in compiled.certification.cells
        )
        metadata = {
            **compiled.policy_ir.metadata,
            "promotion_minimum_passing_domain_fraction": 0.0,
        }
        best_effort = CompiledPolicy(
            dataclasses.replace(compiled.policy_ir, metadata=metadata),
            dataclasses.replace(
                compiled.certification,
                cells=failed_cells,
                minimum_passing_domain_fraction=0.0,
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path = root / "policy.json"
            include_path = root / "policy.inc"
            write_compiled_policy(policy_path, best_effort)
            include_path.write_text(
                f"// Common policy digest: {best_effort.policy_ir.digest()}\n"
                "// Frozen generic policy digest: "
                f"{best_effort.policy_ir.digest(generic_only=True)}\n",
                encoding="utf-8",
            )
            payload = validate_installable_policy_artifact(
                policy_path,
                include_path=include_path,
            )

        certificate = payload["certification"]
        self.assertEqual(certificate["passing_domain_count"], 0)
        self.assertTrue(certificate["domain_promotion_quota_satisfied"])
        self.assertFalse(
            certificate["domain_results"][0]["passes_p95_budget"]
        )

    def test_sealed_p95_allows_exactly_five_percent_diagnostic_outliers(self) -> None:
        """Nearest-rank p95, not the maximum, owns the installation decision."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(candidate="candidate.b", shape_group="dev-a", n=256, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(candidate="candidate.b", shape_group="dev-b", n=384, latency_us=11.0),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(candidate="candidate.b", shape_group="dev-c", n=448, latency_us=11.0),
        ])
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(candidate="candidate.b", shape_group="sealed", n=512, latency_us=11.0),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        report = certify_frozen_policy(frozen, development, sealed).certification
        exemplar = report.cells[0]
        cells = tuple(
            dataclasses.replace(
                exemplar,
                shape_group_id=f"sealed-p95-{index}",
                observed_worst_surface_regret=(0.50 if index >= 95 else 0.01),
                simultaneous_95pct_upper_regret=(0.60 if index >= 95 else 0.02),
            )
            for index in range(100)
        )
        p95_pass = dataclasses.replace(
            report,
            cells=cells,
            sealed_cell_count=100,
            required_cell_count=100,
            covered_cell_count=100,
        )

        self.assertEqual(p95_pass.p95_observed_regret, 0.01)
        self.assertEqual(p95_pass.max_observed_regret, 0.50)
        p95_pass.require_promotable()

        six_outliers = dataclasses.replace(
            p95_pass,
            cells=(
                *(
                    dataclasses.replace(
                        cell,
                        observed_worst_surface_regret=0.50,
                        simultaneous_95pct_upper_regret=0.60,
                    )
                    if index == 94 else cell
                    for index, cell in enumerate(p95_pass.cells)
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "p95 observed regret"):
            six_outliers.require_promotable()

    def test_sealed_promotion_requires_ninety_five_percent_of_domains(self) -> None:
        """Five of 100 over-budget domains are diagnostic; six block promotion."""

        development = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="dev-a", n=256),
            observation(
                candidate="candidate.b",
                shape_group="dev-a",
                n=256,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-b", n=384),
            observation(
                candidate="candidate.b",
                shape_group="dev-b",
                n=384,
                latency_us=11.0,
            ),
            observation(candidate="candidate.a", shape_group="dev-c", n=448),
            observation(
                candidate="candidate.b",
                shape_group="dev-c",
                n=448,
                latency_us=11.0,
            ),
        ])
        sealed = ObservationCorpus([
            observation(candidate="candidate.a", shape_group="sealed", n=512),
            observation(
                candidate="candidate.b",
                shape_group="sealed",
                n=512,
                latency_us=11.0,
            ),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        report = certify_frozen_policy(frozen, development, sealed).certification
        exemplar = report.cells[0]
        cells = tuple(
            dataclasses.replace(
                exemplar,
                domain=dataclasses.replace(exemplar.domain, m=index + 1),
                shape_group_id=f"sealed-domain-{index + 1}",
                observed_worst_surface_regret=(0.50 if index == 99 else 0.01),
                simultaneous_95pct_upper_regret=(0.60 if index == 99 else 0.02),
            )
            for index in range(100)
        )
        ninety_five_percent = dataclasses.replace(
            report,
            cells=tuple(
                dataclasses.replace(
                    cell,
                    observed_worst_surface_regret=0.50,
                    simultaneous_95pct_upper_regret=0.60,
                )
                if index >= 95 else cell
                for index, cell in enumerate(cells)
            ),
            sealed_cell_count=100,
            required_cell_count=100,
            covered_cell_count=100,
        )

        self.assertEqual(ninety_five_percent.required_domain_count, 100)
        self.assertEqual(ninety_five_percent.passing_domain_count(), 95)
        ninety_five_percent.require_promotable()

        ninety_four_percent = dataclasses.replace(
            ninety_five_percent,
            cells=tuple(
                dataclasses.replace(
                    cell,
                    observed_worst_surface_regret=0.50,
                    simultaneous_95pct_upper_regret=0.60,
                )
                if index == 94 else cell
                for index, cell in enumerate(ninety_five_percent.cells)
            ),
        )
        with self.assertRaisesRegex(ValueError, "94/100.*does not reach 95%"):
            ninety_four_percent.require_promotable()

    def test_installable_compiler_rejects_short_timing_corpus(self) -> None:
        rows = []
        for group, n in (("dev-a", 256), ("dev-b", 384), ("sealed", 512)):
            rows.extend([
                dataclasses.replace(
                    observation(
                        candidate="candidate.a",
                        shape_group=group,
                        n=n,
                        latency_us=10.0,
                    ),
                    warmup_count=1,
                    sample_count=2,
                ),
                dataclasses.replace(
                    observation(
                        candidate="candidate.b",
                        shape_group=group,
                        n=n,
                        latency_us=11.0,
                    ),
                    warmup_count=1,
                    sample_count=2,
                ),
            ])
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "dev-a": Partition.DEVELOPMENT,
                "dev-b": Partition.DEVELOPMENT,
                "sealed": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        with self.assertRaisesRegex(ValueError, "non-promotable timing"):
            compile_policy(development, sealed)

    def test_exact_overlay_cannot_mask_broken_generic_policy(self) -> None:
        development = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="dev-a", latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-a", latency_us=14.0),
            observation(candidate="candidate.fast", shape_group="dev-b", n=384, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-b", n=384, latency_us=14.0),
            observation(candidate="candidate.fast", shape_group="dev-c", n=448, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-c", n=448, latency_us=14.0),
        ])
        exact = build_exact_winners(development)
        generic = fit_generic_policy(development, max_leaves=1)
        poisoned_rule = dataclasses.replace(
            generic.rules[0], candidate_id="candidate.slow"
        )
        poisoned = make_policy_ir(
            exact,
            GenericPolicy(rules=(poisoned_rule,), unpromoted_domains=()),
            metadata={"test": "exact overlay must be bypassed"},
        )
        sealed = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="sealed", n=512, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="sealed", n=512, latency_us=14.0),
        ])
        report = certify_generic_policy(poisoned, sealed)
        self.assertAlmostEqual(report.max_observed_regret, 0.4)
        self.assertEqual(len(report.rule_coverage), 1)
        self.assertEqual(report.rule_coverage[0].sealed_hit_count, 1)
        with self.assertRaisesRegex(ValueError, "not promotable"):
            report.require_promotable()

    def test_frozen_policy_mismatch_reports_first_canonical_difference(self) -> None:
        """A stale pre-seal artifact must identify why reconstruction changed."""

        development = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="dev-a", n=256),
            observation(candidate="candidate.slow", shape_group="dev-a", n=256,
                        latency_us=12.0),
            observation(candidate="candidate.fast", shape_group="dev-b", n=384),
            observation(candidate="candidate.slow", shape_group="dev-b", n=384,
                        latency_us=12.0),
            observation(candidate="candidate.fast", shape_group="dev-c", n=448),
            observation(candidate="candidate.slow", shape_group="dev-c", n=448,
                        latency_us=12.0),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed",
            split_manifest_digest="sha256:split",
            metadata={"generation": "original"},
        )
        reconstructed = dataclasses.replace(
            frozen,
            policy_ir=dataclasses.replace(
                frozen.policy_ir,
                metadata={
                    **frozen.policy_ir.metadata,
                    "generation": "reconstructed",
                },
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_frozen_policy(path, frozen)
            with self.assertRaisesRegex(
                ValueError,
                r"persisted_policy_digest=.*reconstructed_policy_digest=.*"
                r"first_difference=\$\.policy\.metadata\.generation",
            ):
                validate_frozen_policy_file(path, reconstructed)

    def test_failed_certificate_writes_noninstallable_rule_diagnostics(self) -> None:
        """A failed generic fit must retain cells without becoming publishable."""

        development = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="dev-a", n=256),
            observation(
                candidate="candidate.slow", shape_group="dev-a", n=256,
                latency_us=14.0,
            ),
            observation(candidate="candidate.fast", shape_group="dev-b", n=384),
            observation(
                candidate="candidate.slow", shape_group="dev-b", n=384,
                latency_us=14.0,
            ),
            observation(candidate="candidate.fast", shape_group="dev-c", n=448),
            observation(
                candidate="candidate.slow", shape_group="dev-c", n=448,
                latency_us=14.0,
            ),
        ])
        frozen = freeze_policy(
            development,
            sealed_commitment="sha256:sealed-shape-inventory",
            split_manifest_digest="sha256:split-manifest",
        )
        poisoned_rule = dataclasses.replace(
            frozen.policy_ir.generic_rules[0], candidate_id="candidate.slow"
        )
        poisoned = dataclasses.replace(
            frozen,
            policy_ir=dataclasses.replace(
                frozen.policy_ir,
                generic_rules=(poisoned_rule,),
            ),
        )
        sealed = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="sealed", n=512),
            observation(
                candidate="candidate.slow", shape_group="sealed", n=512,
                latency_us=14.0,
            ),
        ])
        compiled = certify_frozen_policy(
            poisoned,
            development,
            sealed,
            require_promotable=False,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed-certificate.json"
            write_certification_diagnostic(path, compiled)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertFalse(payload["promotable"])
            self.assertIn("p95 observed regret", payload["promotion_error"])
            self.assertEqual(
                payload["state"],
                "sealed_certification_diagnostic_noninstallable",
            )
            self.assertEqual(
                payload["certification"]["rule_coverage"][0][
                    "sealed_hit_count"
                ],
                1,
            )
            with self.assertRaisesRegex(ValueError, "not promotable"):
                write_compiled_policy(
                    Path(temporary) / "must-not-publish.json",
                    compiled,
                )
            with self.assertRaisesRegex(ValueError, "sealed_certified"):
                validate_installable_policy_artifact(path)

    def test_inventory_validators_reject_missing_m_and_candidate_surface(self) -> None:
        corpus = ObservationCorpus([
            observation(source_format="Q4_1", candidate="candidate.a", m=2),
            observation(source_format="Q4_K", candidate="candidate.b", m=2),
        ])
        with self.assertRaisesRegex(ValueError, "M matrix incomplete"):
            require_verifier_m_matrix(corpus)
        with self.assertRaisesRegex(ValueError, "candidate matrix"):
            require_candidate_matrix_complete(corpus)

    def test_exact_refresh_requires_declared_all_format_scope(self) -> None:
        """A missing whole codebook or depth cannot shrink an additive plan."""

        manifest = load_shape_manifest()
        shape = next(shape for shape in manifest.shapes if shape.exact_overlay)
        scope = dict(
            shape_names=(shape.name,), m_values=(3, 16),
            execution_modes=(ExecutionMode.GRAPH_CAPTURED,),
            contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )
        for backend in Backend:
            rows = [observation(
                source_format=spec.label, n=shape.n, k=shape.k, m=m,
                shape_name=shape.name, backend=backend,
            ) for spec in FORMAT_SPECS for m in scope["m_values"]]
            require_exact_overlay_scope(ObservationCorpus(rows), manifest, **scope)
            for incomplete in (
                [row for row in rows if row.source_format != "Q8_0"],
                [row for row in rows if row.m != 16],
                rows + [dataclasses.replace(rows[0], m=7)],
                [observation(source_format=row.source_format, n=shape.n + 32,
                             k=shape.k, m=row.m, shape_name=shape.name, backend=backend)
                 for row in rows],
            ):
                with self.assertRaisesRegex(ValueError, "surface inventory"):
                    require_exact_overlay_scope(ObservationCorpus(incomplete), manifest, **scope)
            for invalid in (
                dict(scope, m_values=()), dict(scope, m_values=(3, 3)),
                dict(scope, m_values=(1,)), dict(scope, shape_names=()),
            ):
                with self.assertRaises(ValueError):
                    require_exact_overlay_scope(ObservationCorpus(rows), manifest, **invalid)

    def test_counted_row_oracle_preserves_each_occupancy_surface(self) -> None:
        """Full-batch wins cannot hide a slower partially filled retained graph."""

        rows = [dataclasses.replace(observation(
            candidate=candidate, latency_us=latency, m=16,
            generic_eligible=False,
        ), active_rows=active) for candidate, active, latency in (
            ("candidate.tc", 16, 210.0), ("candidate.tc", 3, 182.0),
            ("candidate.rows8", 16, 222.0), ("candidate.rows8", 3, 77.0),
        )]
        corpus = ObservationCorpus(rows)
        self.assertEqual(len(corpus.runtime_keys()), 1)
        winner = build_exact_winner(rows, current_serial_m1_hash=SERIAL_HASH)
        self.assertEqual(winner.candidate_id, "candidate.rows8")
        self.assertEqual({surface.surface.active_rows for surface in winner.surfaces}, {3, 16})
        with self.assertRaisesRegex(ValueError, "candidate matrix"):
            require_candidate_matrix_complete(ObservationCorpus(rows[:-1]))
        legacy = observation()
        self.assertNotIn("active_rows", legacy.canonical_mapping())
        self.assertEqual(NativeVNNIObservation.from_mapping(legacy.canonical_mapping()), legacy)
        counted = rows[0]
        self.assertEqual(NativeVNNIObservation.from_mapping(counted.canonical_mapping()), counted)
        self.assertNotEqual(counted.digest(), dataclasses.replace(counted, active_rows=3).digest())
        for invalid in (0, 17, True):
            with self.assertRaisesRegex(ValueError, "active_rows"):
                dataclasses.replace(counted, active_rows=invalid).validate()
        with self.assertRaisesRegex(ValueError, "exact-only"):
            dataclasses.replace(counted, generic_eligible=True).validate()


    def test_cpp_predicates_match_python_for_every_axis_and_boundary(self) -> None:
        """Every generated branch must preserve the learner's exact arithmetic."""

        thresholds = {
            FeatureAxis.AGGREGATE_N: (100, 1),
            FeatureAxis.K: (160, 1),
            FeatureAxis.WORK_ITEMS: (100000, 3),
            FeatureAxis.ASPECT_RATIO: (5, 4),
            FeatureAxis.N_TILES_32: (5, 2),
            FeatureAxis.N_TILES_64: (5, 2),
            FeatureAxis.N_TILES_128: (5, 2),
            FeatureAxis.N_TILES_256: (5, 2),
            FeatureAxis.N_TILES_512: (5, 2),
            FeatureAxis.N_TILES_1024: (5, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_32: (3, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_64: (3, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_128: (3, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_256: (3, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_512: (3, 2),
            FeatureAxis.K_GROUPS_PER_N_TILE_1024: (3, 2),
            FeatureAxis.N_FINAL_TILE_VALUES_32: (17, 1),
            FeatureAxis.N_FINAL_TILE_VALUES_64: (33, 1),
            FeatureAxis.N_FINAL_TILE_VALUES_128: (65, 1),
            FeatureAxis.N_FINAL_TILE_VALUES_256: (129, 1),
            FeatureAxis.N_FINAL_TILE_VALUES_512: (257, 1),
            FeatureAxis.N_FINAL_TILE_VALUES_1024: (513, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_32: (17, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_64: (33, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_128: (65, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_256: (129, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_512: (257, 1),
            FeatureAxis.K_FINAL_TILE_VALUES_1024: (513, 1),
            FeatureAxis.N_TILE_UTILIZATION_32: (3, 4),
            FeatureAxis.N_TILE_UTILIZATION_64: (3, 4),
            FeatureAxis.N_TILE_UTILIZATION_128: (3, 4),
            FeatureAxis.N_TILE_UTILIZATION_256: (3, 4),
            FeatureAxis.N_TILE_UTILIZATION_512: (3, 4),
            FeatureAxis.N_TILE_UTILIZATION_1024: (3, 4),
            FeatureAxis.N_TILE_ALIGNED_32: (1, 2),
            FeatureAxis.N_TILE_ALIGNED_64: (1, 2),
            FeatureAxis.N_TILE_ALIGNED_128: (1, 2),
            FeatureAxis.N_TILE_ALIGNED_256: (1, 2),
            FeatureAxis.N_TILE_ALIGNED_512: (1, 2),
            FeatureAxis.N_TILE_ALIGNED_1024: (1, 2),
            FeatureAxis.N_PARALLEL_WAVES_64: (3, 1),
            FeatureAxis.N_PARALLEL_WAVES_128: (3, 1),
            FeatureAxis.N_PARALLEL_WAVES_256: (3, 1),
            FeatureAxis.N_PARALLEL_WAVES_512: (3, 1),
            FeatureAxis.N_PARALLEL_WAVES_1024: (3, 1),
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64: (4, 7),
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_128: (4, 7),
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_256: (4, 7),
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_512: (4, 7),
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_1024: (4, 7),
            FeatureAxis.KPART_PRODUCER_WAVES_64: (3, 1),
            FeatureAxis.KPART_PRODUCER_WAVES_128: (3, 1),
            FeatureAxis.KPART_PRODUCER_WAVES_256: (3, 1),
            FeatureAxis.KPART_PRODUCER_WAVES_512: (3, 1),
            FeatureAxis.KPART_PRODUCER_WAVES_1024: (3, 1),
            FeatureAxis.KPART_FINAL_PRODUCER_WAVE_UTILIZATION_64: (4, 7),
            FeatureAxis.KPART_FINAL_PRODUCER_WAVE_UTILIZATION_128: (4, 7),
            FeatureAxis.KPART_FINAL_PRODUCER_WAVE_UTILIZATION_256: (4, 7),
            FeatureAxis.KPART_FINAL_PRODUCER_WAVE_UTILIZATION_512: (4, 7),
            FeatureAxis.KPART_FINAL_PRODUCER_WAVE_UTILIZATION_1024: (4, 7),
            FeatureAxis.KPART_K_BLOCKS_PER_TILE: (8, 1),
            FeatureAxis.KPART_FINAL_K_TILE_BLOCKS: (4, 1),
            FeatureAxis.KPART_FINAL_K_TILE_UTILIZATION: (1, 2),
            FeatureAxis.KPART_K_TILE_COUNT: (7, 1),
            FeatureAxis.MN_PARALLEL_WAVES_64: (3, 1),
            FeatureAxis.MN_FINAL_PARALLEL_WAVE_UTILIZATION_64: (4, 7),
        }
        self.assertEqual(set(thresholds), set(FeatureAxis))
        wave_axes = (
            set(segmented_policy.N_PARALLEL_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.KPART_PRODUCER_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.KPART_FINAL_PRODUCER_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
        )
        mn_wave_axes = (
            set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
            | set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
        )
        predicates = tuple(
            FeaturePredicate(
                FeatureThreshold(
                    axis,
                    numerator,
                    denominator,
                    28
                    if axis in wave_axes
                    else 1,
                    64 if axis in mn_wave_axes else 1,
                ),
                require_less_equal,
            )
            for axis, (numerator, denominator) in thresholds.items()
            for require_less_equal in (True, False)
        )
        points = (
            (31, 32, 0),
            (32, 64, 1),
            (33, 96, 2),
            (127, 256, 3),
            (128, 128, 4),
            (129, 64, 5),
            (511, 2656, 7),
            (512, 2656, 8),
            (513, 2656, 9),
            (2400, 2656, 13),
            (42496, 2656, 16),
        )
        source = ["int main() {"]
        failure = 1
        for predicate in predicates:
            condition = predicate_condition(
                predicate,
                k_tiles_expression="k_tiles",
            )
            for n, k, k_tiles in points:
                expected = (
                    "true"
                    if predicate.matches(n, k, k_tiles)
                    else "false"
                )
                source.extend((
                    "  {",
                    f"    const int n = {n};",
                    f"    const int k = {k};",
                    f"    const int k_tiles = {k_tiles};",
                    "    const long long work_items =",
                    "        static_cast<long long>(n) * static_cast<long long>(k);",
                    f"    if (({condition}) != {expected}) return {failure};",
                    "  }",
                ))
                failure += 1

        aspect_points = (
            (16, 1),
            (159, 10),
            (20, 10),
            (19, 10),
            (3, 4),
            (2, 3),
            (300, 400),
        )
        for n, k in aspect_points:
            expected_bucket = classify_aspect(n, k)
            for bucket in AspectBucket:
                expected = "true" if bucket == expected_bucket else "false"
                source.extend((
                    "  {",
                    f"    const int n = {n};",
                    f"    const int k = {k};",
                    (
                        f"    if (({aspect_condition(bucket)}) != {expected}) "
                        f"return {failure};"
                    ),
                    "  }",
                ))
                failure += 1
        source.extend(("  return 0;", "}"))

        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "predicate_equivalence"
            compiled = subprocess.run(
                ["g++", "-std=c++20", "-x", "c++", "-o", str(executable), "-"],
                input="\n".join(source),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                executed.returncode,
                0,
                f"generated predicate mismatch at check {executed.returncode}",
            )

    def test_parallel_wave_features_express_cpu_prefill_occupancy_transition(
        self,
    ) -> None:
        """A low-utilization second wave must not alias a full first wave."""

        threshold = FeatureThreshold(
            FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64,
            4,
            7,
            28,
        )

        # N=896 has 14 tasks in its only wave. N=2080 has 33 tasks, so its
        # second wave contains only five tasks. Both need the fine-grained row
        # grid. N=1152 and N=4864 leave 18 and 20 active workers respectively
        # in their final waves and can amortize the two-row kernel.
        self.assertTrue(threshold.matches_less_equal(896, 896))
        self.assertTrue(threshold.matches_less_equal(2080, 480))
        self.assertFalse(threshold.matches_less_equal(1152, 896))
        self.assertFalse(threshold.matches_less_equal(4864, 896))

    def test_tile_alignment_expresses_tail_free_decode_grid(self) -> None:
        """A full nbc8 grid must not interpolate onto a partial final task."""

        partial_grid = FeatureThreshold(
            FeatureAxis.N_TILE_ALIGNED_512,
            1,
            2,
        )

        # NBC8 owns eight adjacent 64-column chunks, hence one 512-column
        # task tile. The failed CPU M1 holdout demonstrated that N=2048 can
        # economically use NBC8 while N=1984 must retain NBC1. A continuous N
        # threshold cannot infer that unseen boundary, but exact grid alignment
        # is stable for every shape and is available before dispatch.
        self.assertTrue(partial_grid.matches_less_equal(1920, 11008))
        self.assertTrue(partial_grid.matches_less_equal(1984, 11008))
        self.assertFalse(partial_grid.matches_less_equal(2048, 11008))
        self.assertFalse(partial_grid.matches_less_equal(2560, 11008))

    def test_k_final_tile_expresses_periodic_kpart_geometry(self) -> None:
        """A generic tree must distinguish unseen K-partition tail widths."""

        k_tail_at_most_192 = FeatureThreshold(
            FeatureAxis.K_FINAL_TILE_VALUES_256,
            192,
            1,
        )

        # The sparse AVX512 k-part corpus contains the same N=2048 grid at
        # neighboring K values whose winning launch families differ. A
        # continuous K threshold cannot generalize that periodic boundary;
        # the final 256-value tile is exact for every seen or unseen shape.
        self.assertTrue(k_tail_at_most_192.matches_less_equal(2048, 10880))
        self.assertTrue(k_tail_at_most_192.matches_less_equal(2048, 10944))
        self.assertFalse(k_tail_at_most_192.matches_less_equal(2048, 11008))
        self.assertTrue(k_tail_at_most_192.matches_less_equal(2048, 11072))
        self.assertTrue(k_tail_at_most_192.matches_less_equal(2048, 11136))

    def test_kpart_span_and_tail_match_frozen_serial_partition(self) -> None:
        """K-part predicates must retain short and empty terminal launches."""

        span_at_most_four = FeatureThreshold(
            FeatureAxis.KPART_K_BLOCKS_PER_TILE,
            4,
        )
        tile_count_at_most_one = FeatureThreshold(
            FeatureAxis.KPART_K_TILE_COUNT,
            1,
        )
        final_at_most_one = FeatureThreshold(
            FeatureAxis.KPART_FINAL_K_TILE_BLOCKS,
            1,
        )
        final_at_most_half_full = FeatureThreshold(
            FeatureAxis.KPART_FINAL_K_TILE_UTILIZATION,
            1,
            2,
        )

        # K=320 contains ten NativeVNNI blocks. Missing telemetry and one tile
        # both own all ten blocks. Three tiles own 4,4,2 blocks; four own
        # 3,3,3,1; and eight preserve all eight producer launches even though
        # the final ceil-div partition starts beyond K and therefore does no
        # arithmetic.
        self.assertFalse(span_at_most_four.matches_less_equal(64, 320, 0))
        self.assertFalse(span_at_most_four.matches_less_equal(64, 320, 1))
        self.assertTrue(span_at_most_four.matches_less_equal(64, 320, 3))
        self.assertTrue(tile_count_at_most_one.matches_less_equal(64, 320, 0))
        self.assertTrue(tile_count_at_most_one.matches_less_equal(64, 320, 1))
        self.assertFalse(tile_count_at_most_one.matches_less_equal(64, 320, 3))
        self.assertFalse(final_at_most_one.matches_less_equal(64, 320, 3))
        self.assertTrue(final_at_most_one.matches_less_equal(64, 320, 4))
        self.assertTrue(final_at_most_one.matches_less_equal(64, 320, 8))
        self.assertTrue(
            final_at_most_half_full.matches_less_equal(64, 320, 3)
        )
        self.assertTrue(
            final_at_most_half_full.matches_less_equal(64, 320, 8)
        )

    def test_mn_parallel_wave_features_match_row_chunk_grid_geometry(
        self,
    ) -> None:
        """Row-grid occupancy must include all M independently scheduled rows."""

        threshold = FeatureThreshold(
            FeatureAxis.MN_FINAL_PARALLEL_WAVE_UTILIZATION_64,
            1,
            2,
            28,
            64,
        )

        # Production RowChunkGrid schedules `M * ceil(N / 64)` OpenMP tasks.
        # At M=64, N=4096 leaves eight of 28 workers in the final wave, while
        # N=4160 leaves sixteen. The old N-only feature saw 64 and 65 tasks and
        # therefore modeled a completely different occupancy boundary.
        self.assertTrue(threshold.matches_less_equal(4096, 1024))
        self.assertFalse(threshold.matches_less_equal(4160, 1024))

        count_threshold = FeatureThreshold(
            FeatureAxis.MN_PARALLEL_WAVES_64,
            147,
            1,
            28,
            64,
        )
        self.assertTrue(count_threshold.matches_less_equal(4096, 1024))
        self.assertFalse(count_threshold.matches_less_equal(4160, 1024))

    def test_generic_tree_partition_is_total_beyond_fitted_geometry(self) -> None:
        """Complementary leaves must dispatch tiny and arbitrarily large shapes."""

        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class="test|build=AVX512|runtime=AVX512|threads=28",
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature="serial-full-k",
            prepared_family_id="NativeVNNI_cpu_CB0",
            packing_abi="native-vnni-cpu-cb0-v1",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        threshold = FeatureThreshold(
            FeatureAxis.WORK_ITEMS,
            896 * 896,
        )

        def rule(require_less_equal: bool, candidate: str) -> GenericDispatchRule:
            return GenericDispatchRule(
                domain=domain,
                predicates=(FeaturePredicate(
                    threshold,
                    require_less_equal=require_less_equal,
                ),),
                candidate_id=candidate,
                arithmetic_fingerprint="sha256:test",
                development_shape_groups=("development:test",),
                development_max_regret=0.0,
                development_p95_regret=0.0,
                development_mean_regret=0.0,
            )

        rules = (
            rule(True, "small-work"),
            rule(False, "large-work"),
        )
        validate_generic_rule_partition(rules)
        for n, k in ((1, 32), (895, 896), (896, 896), (1 << 20, 1 << 20)):
            self.assertEqual(
                sum(item.matches(n, k) for item in rules),
                1,
                f"N={n} K={k} did not resolve exactly one generic leaf",
            )

        with self.assertRaisesRegex(ValueError, "unseen geometry uncovered"):
            validate_generic_rule_partition((rules[0],))

    def test_parallel_wave_policy_covers_every_cpu_two_row_schedule(self) -> None:
        """Every forceable N-block candidate must expose its physical waves."""

        expected_widths = {64, 128, 256, 512, 1024}
        self.assertEqual(
            set(segmented_policy.N_PARALLEL_WAVE_WIDTH_BY_AXIS.values()),
            expected_widths,
        )
        self.assertEqual(
            set(
                segmented_policy.N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS.values()
            ),
            expected_widths,
        )
        axes = set(
            segmented_policy.FEATURE_AXES_BY_POLICY[
                FeaturePolicy.PARALLEL_WAVE_SCHEDULES
            ]
        )
        self.assertTrue(
            set(segmented_policy.N_PARALLEL_WAVE_WIDTH_BY_AXIS) <= axes
        )
        self.assertTrue(
            set(segmented_policy.N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS) <= axes
        )
        self.assertEqual(
            set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS.values()),
            {64},
        )
        self.assertEqual(
            set(
                segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS.values()
            ),
            {64},
        )
        self.assertTrue(
            set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS).isdisjoint(
                axes
            )
        )
        self.assertTrue(
            set(
                segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS
            ).isdisjoint(axes)
        )
        row_grid_axes = set(
            segmented_policy.FEATURE_AXES_BY_POLICY[
                FeaturePolicy.ROW_GRID_PARALLEL_WAVE_SCHEDULES
            ]
        )
        self.assertTrue(
            set(segmented_policy.MN_PARALLEL_WAVE_WIDTH_BY_AXIS)
            <= row_grid_axes
        )
        self.assertTrue(
            set(segmented_policy.MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS)
            <= row_grid_axes
        )


if __name__ == "__main__":
    unittest.main()
