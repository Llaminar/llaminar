#!/usr/bin/env python3
"""Regression tests for the common-backed CPU prefill policy generator."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

ANALYZER = (
    KERNEL_PERF_ROOT / "cpu" / "analyze_cpu_native_vnni_prefill_trainer.py"
)
SPEC = importlib.util.spec_from_file_location("cpu_prefill_analyzer", ANALYZER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from native_vnni_dispatch.profiles import (  # noqa: E402
    ADAPTIVE_TIMING_CEILING_POLICY,
    ADAPTIVE_TIMING_PROTOCOL,
)
from native_vnni_dispatch.cpu_prefill_split_manifest import (  # noqa: E402
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    V9_SCHEMA_VERSION,
    V9_QWEN36_MOE_PRODUCTION_SHAPES,
    load_cpu_prefill_split_manifest,
)
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
    CPUPrefillSerialRoute,
    CPUPrefillSerialRouteManifest,
)
from native_vnni_dispatch.cpu_prefill_training_plan import (  # noqa: E402
    CPUPrefillSourceTrainingRecord,
)
from native_vnni_dispatch.corpus import (  # noqa: E402
    GenericDomain,
    ObservationCorpus,
)
from native_vnni_dispatch.cpu_prefill_candidate_expansion import (  # noqa: E402
    CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
    CPUPrefillCandidateExpansionPlan,
    CPUPrefillCandidateExpansionRecord,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    cpu_native_vnni_prefill_registry,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    write_observation_csv,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    cpu_prefill_measurements,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    FeatureAxis,
    FeaturePredicate,
    FeatureThreshold,
    GenericDispatchRule,
)
from native_vnni_dispatch.shape_manifest import load_shape_manifest  # noqa: E402
from tests.v2.unit.scripts.test_native_vnni_common_dispatch_policy import (  # noqa: E402
    observation,
)


class CPUNativeVNNIPrefillTrainerTest(unittest.TestCase):
    """Exercise robust winner emission and shape-aware runtime M bucketing."""

    def test_authenticated_context_identity_skips_duplicate_raw_hash(self) -> None:
        """Recipe-owned fit replay consumes the identity preflight already proved."""

        corpus_id = "sha256:" + "a" * 64
        with mock.patch.object(
            MODULE,
            "raw_corpus_id",
            side_effect=AssertionError("raw evidence was hashed twice"),
        ):
            context = MODULE._context_from_args(
                SimpleNamespace(profile="quick"),
                (Path("aggregate.csv"),),
                (Path("timing.csv"),),
                authenticated_corpus_id=corpus_id,
            )

        self.assertEqual(context.corpus_id, corpus_id)

    def test_sealed_context_uses_explicit_measurement_provenance(self) -> None:
        """Certification must not authenticate fresh rows as a development build."""

        args = SimpleNamespace(
            profile="production",
            run_id="development-run",
            git_revision="development-git",
            build_id="sha256:development-build",
            compiler_id="development-compiler",
            architecture_class="development-architecture",
            device_name="development-device",
            driver_runtime="development-runtime",
            serial_m1_policy_hash="sha256:serial-policy",
            sealed_git_revision="sealed-git",
            sealed_build_id="sha256:sealed-build",
            sealed_compiler_id="sealed-compiler",
            sealed_architecture_class="sealed-architecture",
            sealed_device_name="sealed-device",
            sealed_driver_runtime="sealed-runtime",
            sealed_serial_m1_policy_hash="sha256:serial-policy",
        )
        context = MODULE._context_from_args(
            args,
            (Path("sealed.csv"),),
            (Path("sealed.timing.csv"),),
            run_id="development-run-sealed",
            authenticated_corpus_id="sha256:" + "b" * 64,
            provenance_prefix="sealed",
        )

        self.assertEqual(context.run_id, "development-run-sealed")
        self.assertEqual(context.git_revision, "sealed-git")
        self.assertEqual(context.build_id, "sha256:sealed-build")
        self.assertEqual(context.compiler_id, "sealed-compiler")
        self.assertEqual(context.architecture_class, "sealed-architecture")
        self.assertEqual(context.device_name, "sealed-device")
        self.assertEqual(context.driver_runtime, "sealed-runtime")
        self.assertEqual(context.serial_m1_policy_hash, "sha256:serial-policy")

    def test_development_fit_does_not_rewrite_its_full_checkpoint(self) -> None:
        """A read-only fit preserves an already-authenticated checkpoint."""

        checkpoint = Path("/tmp/native-vnni-checkpoint.csv")
        self.assertFalse(MODULE._common_checkpoint_requires_write(
            checkpoint,
            checkpoint,
        ))
        self.assertTrue(MODULE._common_checkpoint_requires_write(
            Path("/tmp/native-vnni-checkpoint-copy.csv"),
            checkpoint,
        ))
        self.assertTrue(MODULE._common_checkpoint_requires_write(
            checkpoint,
            None,
        ))
        self.assertFalse(MODULE._common_checkpoint_requires_write(None, None))

    def test_parallel_checkpoint_reader_is_order_and_byte_deterministic(self) -> None:
        """Byte-range parsing must reproduce the serial canonical corpus."""

        rows = tuple(
            observation(
                backend=Backend.CPU,
                mode=ExecutionMode.EAGER,
                shape_group=f"parallel-checkpoint-{index}",
                shape_name=f"parallel-checkpoint-{index}",
                n=256 + 32 * index,
            )
            for index in range(48)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.csv"
            write_observation_csv(path, rows)
            serial = MODULE._read_observation_checkpoint(
                path,
                requested_workers=1,
                minimum_parallel_bytes=0,
            )
            parallel = MODULE._read_observation_checkpoint(
                path,
                requested_workers=4,
                minimum_parallel_bytes=0,
            )

        self.assertEqual(parallel.observations, serial.observations)
        self.assertEqual(parallel.digest(), serial.digest())

    def test_full_development_checkpoint_is_bound_to_raw_corpus_identity(self) -> None:
        """Compact replay accepts only rows adapted from the declared inputs."""

        context = SimpleNamespace(
            run_id="unit-run",
            corpus_id="sha256:unit-corpus",
            git_revision="unit-git",
            build_id="unit-build",
            compiler_id="unit-compiler",
            architecture_class="unit-cpu",
            device_name="unit-device",
            driver_runtime="unit-runtime",
            serial_m1_policy_hash="sha256:unit-serial",
            validate=mock.Mock(),
        )
        row = replace(
            observation(
                backend=Backend.CPU,
                mode=ExecutionMode.EAGER,
                serial_hash=context.serial_m1_policy_hash,
            ),
            run_id=context.run_id,
            corpus_id=context.corpus_id,
            git_revision=context.git_revision,
            build_id=f"{context.build_id}|cpu_isa=AVX2",
            compiler_id=context.compiler_id,
            architecture_class=(
                f"{context.architecture_class}|build=AVX2|runtime=AVX2|"
                "threads=28"
            ),
            device_name=context.device_name,
            driver_runtime=context.driver_runtime,
        )

        MODULE._validate_adapted_development_checkpoint(
            ObservationCorpus((row,)),
            context,
        )
        context.validate.assert_called_once_with()

        stale = ObservationCorpus((replace(
            row,
            corpus_id="sha256:another-corpus",
        ),))
        with self.assertRaisesRegex(ValueError, "another transaction: corpus_id"):
            MODULE._validate_adapted_development_checkpoint(stale, context)

    def test_prefix_checkpoint_infers_and_adapts_new_development_suffix(self) -> None:
        """Plan lineage locates an authenticated prefix without replay."""

        old_context = SimpleNamespace(
            run_id="unit-run",
            corpus_id="sha256:old-corpus",
            git_revision="unit-git",
            build_id="unit-build",
            compiler_id="unit-compiler",
            architecture_class="unit-cpu",
            device_name="unit-device",
            driver_runtime="unit-runtime",
            serial_m1_policy_hash="sha256:unit-serial",
            validate=mock.Mock(),
        )
        new_context = SimpleNamespace(
            **{
                **vars(old_context),
                "corpus_id": "sha256:new-corpus",
                "validate": mock.Mock(),
            }
        )
        old_row = replace(
            observation(
                backend=Backend.CPU,
                mode=ExecutionMode.EAGER,
                serial_hash=old_context.serial_m1_policy_hash,
                m=64,
            ),
            run_id=old_context.run_id,
            corpus_id=old_context.corpus_id,
            git_revision=old_context.git_revision,
            build_id=f"{old_context.build_id}|cpu_isa=AVX2",
            compiler_id=old_context.compiler_id,
            architecture_class=(
                f"{old_context.architecture_class}|build=AVX2|runtime=AVX2|"
                "threads=28"
            ),
            device_name=old_context.device_name,
            driver_runtime=old_context.driver_runtime,
        )
        suffix_row = replace(
            old_row,
            corpus_id=new_context.corpus_id,
            shape_group_id="new-refinement-cell",
            shape_name="NewRefinementCell",
            m=256,
            timing_sample_hash="sha256:new-refinement-timings",
        )
        args = SimpleNamespace(
            candidate_expansion_plan_json=None,
            candidate_expansion_input=None,
            candidate_expansion_timing_sidecar=None,
            adapted_development_input=None,
            adapted_full_development_input=None,
            adapted_development_prefix_input=Path("checkpoint.csv"),
            adapted_development_prefix_count=None,
        )
        inputs = (Path("base.csv"), Path("refinement.csv"))
        timings = (Path("base.timing.csv"), Path("refinement.timing.csv"))
        checkpoint = ObservationCorpus((old_row,))
        refinement_plan = SimpleNamespace(
            source_development_corpus_digest=(
                checkpoint.with_collapsed_aspect_domains().digest()
            ),
        )

        with (
            mock.patch.object(
                MODULE,
                "read_observation_csv",
                return_value=checkpoint,
            ),
            mock.patch.object(
                MODULE,
                "_context_from_args",
                return_value=old_context,
            ) as context_from_args,
            mock.patch.object(
                MODULE,
                "adapt_cpu_prefill_csv",
                return_value=ObservationCorpus((suffix_row,)),
            ) as adapt,
        ):
            actual, plan = MODULE._adapt_development_with_candidate_expansion(
                args,
                inputs,
                timings,
                new_context,
                (refinement_plan,),
            )

        self.assertIsNone(plan)
        self.assertEqual(len(actual), 2)
        rebound = actual.observations[0]
        self.assertEqual(rebound.corpus_id, new_context.corpus_id)
        self.assertEqual(rebound.timing_sample_hash, old_row.timing_sample_hash)
        context_from_args.assert_called_once_with(
            args,
            (inputs[0],),
            (timings[0],),
        )
        adapt.assert_called_once_with(
            (inputs[1],),
            new_context,
            timing_sidecars=(timings[1],),
        )

    def test_burned_seal_appends_under_its_original_transaction_identity(self) -> None:
        """A failed holdout may become development data but is never relabeled."""

        serial_hash = "sha256:" + "5" * 64
        base_context = SimpleNamespace(
            corpus_id="sha256:" + "1" * 64,
            serial_m1_policy_hash=serial_hash,
        )
        burned_context = SimpleNamespace(
            corpus_id="sha256:" + "2" * 64,
            serial_m1_policy_hash=serial_hash,
        )
        base_row = replace(
            observation(backend=Backend.CPU, mode=ExecutionMode.EAGER),
            corpus_id=base_context.corpus_id,
            build_id="base-build|cpu_isa=AVX2",
        )
        burned_row = replace(
            base_row,
            corpus_id=burned_context.corpus_id,
            build_id="sealed-build|cpu_isa=AVX2",
            shape_group_id="failed-seal-shape",
            shape_name="FailedSealShape",
        )
        transaction = SimpleNamespace(
            aggregate=SimpleNamespace(
                resolve=mock.Mock(return_value=Path("sealed.csv"))
            ),
            timing=SimpleNamespace(
                resolve=mock.Mock(return_value=Path("sealed.timing.csv"))
            ),
            witness_plan=SimpleNamespace(
                resolve=mock.Mock(return_value=Path("witness.json"))
            ),
            path=Path("burned.json"),
        )
        args = SimpleNamespace(
            profile="production",
            burned_sealed_development_transactions=(transaction,),
            adapted_full_development_input=None,
            adapted_development_before_burned_seals_input=Path("base-common.csv"),
        )

        with (
            mock.patch.object(
                MODULE,
                "_burned_seal_context",
                return_value=burned_context,
            ),
            mock.patch.object(
                MODULE,
                "_adapt_development_with_candidate_expansion",
                return_value=(ObservationCorpus((base_row,)), mock.sentinel.candidate_plan),
            ) as adapt_base,
            mock.patch.object(
                MODULE,
                "_read_observation_checkpoint",
                return_value=ObservationCorpus((base_row,)),
            ),
            mock.patch.object(
                MODULE,
                "adapt_cpu_prefill_csv",
                return_value=ObservationCorpus((burned_row,)),
            ) as adapt_burned,
            mock.patch.object(
                MODULE,
                "read_cpu_prefill_sealed_witness_plan",
                return_value=mock.sentinel.witness_plan,
            ),
            mock.patch.object(MODULE, "_require_partition") as require_partition,
        ):
            corpus, candidate_plan, burned_plans = (
                MODULE._adapt_development_with_burned_seals(
                    args,
                    (Path("base.csv"),),
                    (Path("base.timing.csv"),),
                    base_context,
                    mock.sentinel.route_manifest,
                    mock.sentinel.split_manifest,
                )
            )

        self.assertEqual(
            [row.corpus_id for row in corpus],
            [base_context.corpus_id, burned_context.corpus_id],
        )
        self.assertEqual(
            [row.build_id for row in corpus],
            ["base-build|cpu_isa=AVX2", "sealed-build|cpu_isa=AVX2"],
        )
        self.assertIs(candidate_plan, mock.sentinel.candidate_plan)
        self.assertEqual(burned_plans, (mock.sentinel.witness_plan,))
        replay_args = adapt_base.call_args.args[0]
        self.assertEqual(
            replay_args.adapted_full_development_input,
            Path("base-common.csv"),
        )
        adapt_burned.assert_called_once_with(
            (Path("sealed.csv"),),
            burned_context,
            timing_sidecars=(Path("sealed.timing.csv"),),
        )
        require_partition.assert_called_once_with(
            mock.ANY,
            mock.sentinel.route_manifest,
            mock.sentinel.split_manifest,
            sealed=True,
            sealed_witness_plan=mock.sentinel.witness_plan,
        )

    def test_full_checkpoint_replays_historical_expansion_plan(self) -> None:
        """Fit replay authenticates plan provenance without current-file drift."""

        checkpoint = ObservationCorpus((observation(
            source_format="Q4_0",
            candidate="cpu.nvnni.prefill.row_chunk_grid.full_k",
            family="cpu-prefill-unit",
            mode=ExecutionMode.EAGER,
            backend=Backend.CPU,
        ),))
        args = SimpleNamespace(
            candidate_expansion_plan_json=Path("expansion.json"),
            candidate_expansion_input=Path("expansion.csv"),
            candidate_expansion_timing_sidecar=Path("expansion.timing.csv"),
            adapted_development_input=None,
            adapted_full_development_input=Path("complete.csv"),
        )
        inputs = (Path("source.csv"), Path("refinement.csv"))
        timings = (Path("source.timing.csv"), Path("refinement.timing.csv"))

        with (
            mock.patch.object(
                MODULE,
                "read_cpu_prefill_candidate_expansion_plan",
                return_value=mock.sentinel.plan,
            ) as read_plan,
            mock.patch.object(
                MODULE,
                "read_observation_csv",
                return_value=checkpoint,
            ),
            mock.patch.object(
                MODULE,
                "_validate_adapted_development_checkpoint",
            ),
            mock.patch.object(
                MODULE,
                "_candidate_expansion_expected_cells",
                return_value={"cell"},
            ),
            mock.patch.object(
                MODULE,
                "_candidate_expansion_cell",
                return_value="cell",
            ),
            mock.patch.object(
                MODULE,
                "validate_normalized_cpu_prefill_candidate_expansion",
            ) as validate_normalized,
        ):
            actual, plan = MODULE._adapt_development_with_candidate_expansion(
                args,
                inputs,
                timings,
                mock.sentinel.context,
            )

        self.assertIs(actual, checkpoint)
        self.assertIs(plan, mock.sentinel.plan)
        read_plan.assert_called_once_with(
            args.candidate_expansion_plan_json,
            source_aggregate=inputs[0],
            source_timing=timings[0],
            collection_build_digest=None,
            authenticate_current_implementation=False,
        )
        validate_normalized.assert_called_once()

    def test_complete_checkpoint_rebase_changes_provenance_only(self) -> None:
        """Workspace relocation never reinterprets historical raw expansion timing."""

        context = SimpleNamespace(
            run_id="unit-run",
            corpus_id="sha256:new-path-corpus",
            git_revision="unit-git",
            build_id="unit-build",
            compiler_id="unit-compiler",
            architecture_class="unit-cpu",
            device_name="unit-device",
            driver_runtime="unit-runtime",
            serial_m1_policy_hash="sha256:unit-serial",
            validate=mock.Mock(),
        )
        original = replace(
            observation(
                backend=Backend.CPU,
                mode=ExecutionMode.EAGER,
                serial_hash=context.serial_m1_policy_hash,
            ),
            run_id=context.run_id,
            corpus_id="sha256:old-path-corpus",
            git_revision=context.git_revision,
            build_id=f"{context.build_id}|cpu_isa=AVX2",
            compiler_id=context.compiler_id,
            architecture_class=(
                f"{context.architecture_class}|build=AVX2|runtime=AVX2|"
                "threads=28"
            ),
            device_name=context.device_name,
            driver_runtime=context.driver_runtime,
        )
        checkpoint = ObservationCorpus((original,))
        args = SimpleNamespace(
            candidate_expansion_plan_json=Path("expansion.json"),
            candidate_expansion_input=Path("expansion.csv"),
            candidate_expansion_timing_sidecar=Path("expansion.timing.csv"),
            adapted_development_input=None,
            adapted_full_development_input=None,
            adapted_full_development_rebase_input=Path("complete.csv"),
            adapted_development_prefix_input=None,
            adapted_development_prefix_count=None,
        )

        with (
            mock.patch.object(
                MODULE,
                "read_cpu_prefill_candidate_expansion_plan",
                return_value=mock.sentinel.plan,
            ) as read_plan,
            mock.patch.object(
                MODULE,
                "_read_observation_checkpoint",
                return_value=checkpoint,
            ),
            mock.patch.object(
                MODULE,
                "_validate_checkpoint_candidate_expansion",
            ) as validate_expansion,
            mock.patch.object(
                MODULE,
                "adapt_cpu_prefill_csv",
            ) as adapt_raw,
        ):
            actual, plan = MODULE._adapt_development_with_candidate_expansion(
                args,
                (Path("source.csv"), Path("refinement.csv")),
                (Path("source.timing.csv"), Path("refinement.timing.csv")),
                context,
            )

        self.assertIs(plan, mock.sentinel.plan)
        self.assertEqual(actual.observations[0].corpus_id, context.corpus_id)
        self.assertEqual(
            actual.observations[0].timing_sample_hash,
            original.timing_sample_hash,
        )
        read_plan.assert_called_once_with(
            args.candidate_expansion_plan_json,
            source_aggregate=Path("source.csv"),
            source_timing=Path("source.timing.csv"),
            collection_build_digest=None,
            authenticate_current_implementation=False,
        )
        validate_expansion.assert_called_once_with(
            checkpoint,
            mock.sentinel.plan,
        )
        adapt_raw.assert_not_called()

    def test_exact_overlay_uses_only_declared_production_shape_names(self) -> None:
        """Synthetic refinement aliases train generic rules but are not overlays."""

        candidate = "cpu.nvnni.prefill.row_chunk_grid.full_k"
        common = replace(
            observation(
                source_format="Q4_0",
                candidate=candidate,
                family="cpu-prefill-unit",
                shape_group="fast-sealed-4096",
                shape_name="FastSealed_Balanced_4096x4096",
                n=4096,
                k=4096,
                m=16384,
                latency_us=10.0,
                mode=ExecutionMode.EAGER,
                backend=Backend.CPU,
            ),
            architecture_class=(
                "unit-host|build=AVX2|runtime=AVX2|threads=28"
            ),
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
        )
        model_alias = replace(
            common,
            shape_group_id="qwen35-release-4096",
            shape_name="Qwen35Release_4096x4096",
            timing_sample_hash="sha256:qwen35-release-timings",
        )

        entries = MODULE.select_entries(
            ObservationCorpus((common, model_alias)),
            common.serial_m1_policy_hash,
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0].shape_names,
            ("Qwen35Release_4096x4096",),
        )

    def test_synthetic_refinement_geometry_has_no_exact_overlay(self) -> None:
        """A learned boundary probe must never leak into production exact dispatch."""

        candidate = "cpu.nvnni.prefill.row_chunk_grid.full_k"
        synthetic = replace(
            observation(
                source_format="Q5_0",
                candidate=candidate,
                family="cpu-prefill-unit",
                shape_group="cpu-prefill:CPUPrefillAutoRefine_N512_K288:n512:k288",
                shape_name="CPUPrefillAutoRefine_N512_K288",
                n=512,
                k=288,
                m=2048,
                latency_us=10.0,
                mode=ExecutionMode.EAGER,
                backend=Backend.CPU,
            ),
            architecture_class=(
                "unit-host|build=AVX2|runtime=AVX2|threads=28"
            ),
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
        )

        entries = MODULE.select_entries(
            ObservationCorpus((synthetic,)),
            synthetic.serial_m1_policy_hash,
        )

        self.assertEqual(entries, [])

    def test_lineage_refinements_use_their_authored_split(self) -> None:
        """Mixed historical rounds are authenticated against v8 and v9."""

        class FakeSplit:
            def __init__(self, digest: str) -> None:
                self._digest = digest

            def digest(self) -> str:
                return self._digest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v8_plan = root / "round-v8.json"
            v9_plan = root / "round-v9.json"
            v8_plan.write_text(
                '{"split_manifest_digest":"sha256:v8"}\n',
                encoding="utf-8",
            )
            v9_plan.write_text(
                '{"split_manifest_digest":"sha256:v9"}\n',
                encoding="utf-8",
            )
            seen: list[tuple[str, str]] = []

            def authenticate(path, _route, split):
                seen.append((path.name, split.digest()))
                return path.name

            with mock.patch.object(
                MODULE,
                "read_cpu_prefill_generic_refinement_plan",
                side_effect=authenticate,
            ):
                plans = MODULE._read_lineage_source_refinement_plans(
                    [v8_plan, v9_plan],
                    mock.sentinel.source_route,
                    (FakeSplit("sha256:v9"), FakeSplit("sha256:v8")),
                )

            self.assertEqual(plans, ("round-v8.json", "round-v9.json"))
            self.assertEqual(
                seen,
                [
                    ("round-v8.json", "sha256:v8"),
                    ("round-v9.json", "sha256:v9"),
                ],
            )

    def test_lineage_refinement_rejects_an_unprovided_split(self) -> None:
        """A historical round cannot silently inherit the wrong split."""

        split = mock.Mock()
        split.digest.return_value = "sha256:v9"
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "round-v8.json"
            plan.write_text(
                '{"split_manifest_digest":"sha256:v8"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no lineage source split"):
                MODULE._read_lineage_source_refinement_plans(
                    [plan],
                    mock.sentinel.source_route,
                    (split,),
                )

    def test_candidate_expansion_preserves_generic_geometry_and_full_overlays(self) -> None:
        """Historical cells retain source candidates; production adds all new ones."""

        registry_candidates = tuple(
            entry.candidate_id
            for entry in cpu_native_vnni_prefill_registry().entries
        )
        expansion_candidates = tuple(
            candidate
            for candidate in registry_candidates
            if ".two_row_pair_grid." in candidate
        )
        source_candidates = tuple(
            candidate
            for candidate in registry_candidates
            if candidate not in expansion_candidates
        )
        regimes = (
            ("avx2-build.avx2-runtime", "AVX2", "AVX2"),
            ("avx512-build.avx2-runtime", "AVX512", "AVX2"),
            ("avx512-build.avx512-runtime", "AVX512", "AVX512"),
        )

        def row(candidate: str, build: str, runtime: str, shape: str):
            return replace(
                observation(
                    source_format="Q4_0",
                    candidate=candidate,
                    family="cpu-prefill-unit",
                    shape_group=f"cpu-prefill:{shape}:n1024:k2048",
                    shape_name=shape,
                    n=1024,
                    k=2048,
                    m=64,
                    latency_us=10.0,
                    mode=ExecutionMode.EAGER,
                    backend=Backend.CPU,
                ),
                architecture_class=(
                    f"unit-host|build={build}|runtime={runtime}|threads=28"
                ),
            )

        records = tuple(
            CPUPrefillCandidateExpansionRecord(
                source_format="Q4_0",
                shape_name="production-shape",
                n=1024,
                k=2048,
                isa_regime=regime,
                runtime_isa=runtime.lower(),
                threads=28,
                m_values=(64,),
            )
            for regime, _, runtime in regimes
        )
        plan = CPUPrefillCandidateExpansionPlan(
            schema_version=CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
            source_aggregate_sha256="sha256:source",
            source_timing_sha256="sha256:timing",
            candidate_registry_digest=cpu_native_vnni_prefill_registry().digest(),
            implementation_digest="sha256:implementation",
            collection_build_digest="sha256:" + "b" * 64,
            source_cell_count=4,
            source_cell_digest="sha256:source-cells",
            selection_policy="split-development-cells-v1",
            selected_cell_count=3,
            selected_cell_digest="sha256:selected-cells",
            source_candidate_ids=source_candidates,
            anchor_candidate_ids=(
                "cpu.nvnni.prefill.row_chunk_grid.full_k",
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
            ),
            expansion_candidate_ids=expansion_candidates,
            records=records,
        )
        source = ObservationCorpus(tuple(
            [
                row(candidate, build, runtime, "production-shape")
                for _, build, runtime in regimes
                for candidate in source_candidates
            ]
            + [
                row(candidate, "AVX2", "AVX2", "historical-refinement")
                for candidate in source_candidates
            ]
        ))

        retained = MODULE._select_candidate_expansion_source(source, plan)

        self.assertEqual(len(retained), 3 * len(source_candidates))
        self.assertEqual(
            {item.shape_name for item in retained},
            {"production-shape"},
        )
        combined = ObservationCorpus((
            *retained.observations,
            *(
                row(candidate, build, runtime, "production-shape")
                for _, build, runtime in regimes
                for candidate in expansion_candidates
            ),
        ))
        MODULE.validate_candidate_expansion_complete(combined, plan)
        with self.assertRaisesRegex(ValueError, "candidate coverage"):
            MODULE.validate_candidate_expansion_complete(
                ObservationCorpus(combined.observations[:-1]),
                plan,
            )
        with self.assertRaisesRegex(ValueError, "selected cell count"):
            MODULE.validate_candidate_expansion_complete(
                ObservationCorpus((
                    *combined.observations,
                    row(
                        expansion_candidates[0],
                        "AVX2",
                        "AVX2",
                        "historical-refinement",
                    ),
                )),
                plan,
            )

    def test_candidate_expansion_checkpoint_appends_raw_refinement(self) -> None:
        """A compact normalized base may gain ordinary full-matrix evidence."""

        base = ObservationCorpus((observation(
            source_format="Q4_0",
            candidate="cpu.nvnni.prefill.row_chunk_grid.full_k",
            family="cpu-prefill-unit",
            shape_group="cpu-prefill:base",
            shape_name="base-shape",
            mode=ExecutionMode.EAGER,
            backend=Backend.CPU,
        ),))
        refinement = ObservationCorpus((observation(
            source_format="Q4_0",
            candidate="cpu.nvnni.prefill.row_chunk_grid.full_k",
            family="cpu-prefill-unit",
            shape_group="cpu-prefill:refinement",
            shape_name="refinement-shape",
            n=576,
            mode=ExecutionMode.EAGER,
            backend=Backend.CPU,
        ),))
        plan = object()
        context = SimpleNamespace(build_id="sha256:unit-build")
        args = SimpleNamespace(
            candidate_expansion_plan_json=Path("expansion.json"),
            candidate_expansion_input=Path("expansion.csv"),
            candidate_expansion_timing_sidecar=Path("expansion.timing.csv"),
            adapted_development_input=Path("normalized.csv"),
        )
        inputs = (Path("source.csv"), Path("refinement.csv"))
        timings = (
            Path("source.timing.csv"),
            Path("refinement.timing.csv"),
        )

        with (
            mock.patch.object(
                MODULE,
                "read_cpu_prefill_candidate_expansion_plan",
                return_value=plan,
            ) as read_plan,
            mock.patch.object(
                MODULE,
                "read_observation_csv",
                return_value=base,
            ),
            mock.patch.object(
                MODULE,
                "validate_normalized_cpu_prefill_candidate_expansion",
            ) as validate_base,
            mock.patch.object(
                MODULE,
                "adapt_cpu_prefill_csv",
                return_value=refinement,
            ) as adapt_refinement,
        ):
            combined, restored_plan = (
                MODULE._adapt_development_with_candidate_expansion(
                    args,
                    inputs,
                    timings,
                    context,
                )
            )

        self.assertIs(restored_plan, plan)
        self.assertEqual(
            {row.shape_name for row in combined},
            {"base-shape", "refinement-shape"},
        )
        read_plan.assert_called_once_with(
            args.candidate_expansion_plan_json,
            source_aggregate=inputs[0],
            source_timing=timings[0],
            collection_build_digest=None,
            authenticate_current_implementation=False,
        )
        validate_base.assert_called_once_with(base, plan)
        adapt_refinement.assert_called_once_with(
            inputs[1:],
            context,
            timing_sidecars=timings[1:],
        )

    def test_candidate_expansion_refinement_requires_all_candidates(self) -> None:
        """Every additive boundary cell has the complete physical registry."""

        registry_candidates = tuple(
            entry.candidate_id
            for entry in cpu_native_vnni_prefill_registry().entries
        )
        expansion_candidates = tuple(
            candidate
            for candidate in registry_candidates
            if ".two_row_pair_grid." in candidate
        )
        source_candidates = tuple(
            candidate
            for candidate in registry_candidates
            if candidate not in expansion_candidates
        )
        regimes = (
            ("avx2-build.avx2-runtime", "AVX2", "AVX2"),
            ("avx512-build.avx2-runtime", "AVX512", "AVX2"),
            ("avx512-build.avx512-runtime", "AVX512", "AVX512"),
        )

        def row(
            candidate: str,
            build: str,
            runtime: str,
            shape: str,
            n: int,
        ):
            return replace(
                observation(
                    source_format="Q4_0",
                    candidate=candidate,
                    family="cpu-prefill-unit",
                    shape_group=f"cpu-prefill:{shape}:n{n}:k2048",
                    shape_name=shape,
                    n=n,
                    k=2048,
                    m=64,
                    latency_us=10.0,
                    mode=ExecutionMode.EAGER,
                    backend=Backend.CPU,
                ),
                architecture_class=(
                    f"unit-host|build={build}|runtime={runtime}|threads=28"
                ),
            )

        expansion_records = tuple(
            CPUPrefillCandidateExpansionRecord(
                source_format="Q4_0",
                shape_name="production-shape",
                n=1024,
                k=2048,
                isa_regime=regime,
                runtime_isa=runtime.lower(),
                threads=28,
                m_values=(64,),
            )
            for regime, _, runtime in regimes
        )
        expansion_plan = CPUPrefillCandidateExpansionPlan(
            schema_version=CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
            source_aggregate_sha256="sha256:source",
            source_timing_sha256="sha256:timing",
            candidate_registry_digest=cpu_native_vnni_prefill_registry().digest(),
            implementation_digest="sha256:implementation",
            collection_build_digest="sha256:" + "b" * 64,
            source_cell_count=3,
            source_cell_digest="sha256:source-cells",
            selection_policy="split-development-cells-v1",
            selected_cell_count=3,
            selected_cell_digest="sha256:selected-cells",
            source_candidate_ids=source_candidates,
            anchor_candidate_ids=(
                "cpu.nvnni.prefill.row_chunk_grid.full_k",
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
            ),
            expansion_candidate_ids=expansion_candidates,
            records=expansion_records,
        )
        refinement_record = CPUPrefillSourceTrainingRecord(
            source_format="Q4_0",
            shape_name="refinement-shape",
            n=1088,
            k=2048,
            isa_regime="avx2-build.avx2-runtime",
            runtime_isa="avx2",
            m_values=(64,),
        )
        refinement_plan = SimpleNamespace(
            records=(refinement_record,),
            refinement_shapes=(
                SimpleNamespace(name="refinement-shape", n=1088, k=2048),
            ),
            digest=lambda: "sha256:refinement",
        )
        lineage_record = CPUPrefillSourceTrainingRecord(
            source_format="Q4_0",
            shape_name="lineage-shape",
            n=1152,
            k=2048,
            isa_regime="avx2-build.avx2-runtime",
            runtime_isa="avx2",
            m_values=(64,),
        )
        lineage_plan = SimpleNamespace(
            source_records=(),
            increment_records=(lineage_record,),
        )
        split = SimpleNamespace(
            shape_names=lambda *, sealed: (
                frozenset() if sealed else frozenset({"production-shape"})
            ),
            require_partition=mock.Mock(),
        )
        base_rows = tuple(
            row(candidate, build, runtime, "production-shape", 1024)
            for _, build, runtime in regimes
            for candidate in registry_candidates
        )
        refinement_rows = tuple(
            row(candidate, "AVX2", "AVX2", "refinement-shape", 1088)
            for candidate in registry_candidates
        )
        lineage_rows = tuple(
            row(candidate, "AVX2", "AVX2", "lineage-shape", 1152)
            for candidate in registry_candidates
        )
        complete = ObservationCorpus((
            *base_rows,
            *refinement_rows,
            *lineage_rows,
        ))

        self.assertIs(
            MODULE._require_partition(
                complete,
                object(),
                split,
                sealed=False,
                generic_refinement_plans=(refinement_plan,),
                development_lineage=lineage_plan,
                candidate_expansion_plan=expansion_plan,
            ),
            complete,
        )
        with self.assertRaisesRegex(ValueError, "registry candidate coverage"):
            MODULE._require_partition(
                ObservationCorpus((*base_rows, *refinement_rows[:-1])),
                object(),
                split,
                sealed=False,
                generic_refinement_plans=(refinement_plan,),
                development_lineage=lineage_plan,
                candidate_expansion_plan=expansion_plan,
            )

    def test_reviewed_split_covers_every_prefill_geometry_once(self) -> None:
        manifest = load_cpu_prefill_split_manifest()
        expected = {
            measurement.shape.name
            for measurement in cpu_prefill_measurements()
        }

        self.assertEqual(
            set(manifest.development_shapes) & expected,
            expected,
        )
        self.assertFalse(
            set(manifest.development_shapes) & set(manifest.sealed_shapes)
        )
        self.assertIn("32B_FFN_Dn", manifest.development_shapes)
        self.assertIn("3B_FFN_Up", manifest.development_shapes)
        self.assertEqual(manifest.schema_version, SCHEMA_VERSION)
        self.assertTrue(manifest.include_all_production_shapes)
        self.assertTrue(
            V9_QWEN36_MOE_PRODUCTION_SHAPES.issubset(
                manifest.development_shapes
            )
        )
        self.assertIn(
            "V5CPUPrefillSealed_Tall_176x352", manifest.development_shapes
        )
        self.assertIn(
            "CPUPrefillRefine_0_5B_FFN_Up_Nm64_4800x896",
            manifest.development_shapes,
        )
        self.assertIn(
            "V8CPUPrefillSealed_VeryWide_17600x544",
            manifest.sealed_shapes,
        )
        self.assertIn(
            "V9CPUPrefillSealed_Tall_208x416",
            manifest.development_shapes,
        )
        self.assertIn(
            "V10CPUPrefillSealed_Tall_200x416",
            manifest.sealed_shapes,
        )
        self.assertNotIn(
            "V4FastSealed_Balanced_608x704", manifest.sealed_shapes
        )
        self.assertEqual(
            manifest.sealed_m_values,
            (64, 256, 1024, 2048, 4096, 8192, 16384),
        )

    def test_v8_split_remains_readable_for_digest_bound_refinement(self) -> None:
        """Adding production anchors cannot rewrite an in-flight plan's split."""

        path = (
            KERNEL_PERF_ROOT
            / "native_vnni_dispatch"
            / "manifests"
            / "native_vnni_cpu_prefill_split_v8.json"
        )
        manifest = load_cpu_prefill_split_manifest(path)

        self.assertEqual(manifest.schema_version, LEGACY_SCHEMA_VERSION)
        self.assertEqual(
            manifest.digest(),
            "sha256:d60ac640183b5211600458e7f0067efb185e4438898f8c0b278c9c468efb200e",
        )
        self.assertTrue(
            V9_QWEN36_MOE_PRODUCTION_SHAPES.isdisjoint(
                manifest.development_shapes
            )
        )

    def test_v9_split_remains_readable_for_digest_bound_refinement(self) -> None:
        """Declarative v10 ownership must not rewrite completed v9 evidence."""

        path = (
            KERNEL_PERF_ROOT
            / "native_vnni_dispatch"
            / "manifests"
            / "native_vnni_cpu_prefill_split_v9.json"
        )
        manifest = load_cpu_prefill_split_manifest(path)

        self.assertEqual(manifest.schema_version, V9_SCHEMA_VERSION)
        self.assertFalse(manifest.include_all_production_shapes)
        self.assertEqual(
            manifest.digest(),
            "sha256:a0654a4e60443b6d1389e69063d56b20bb594a89b86ce75e86a9e505e50ede0d",
        )
        self.assertTrue(
            V9_QWEN36_MOE_PRODUCTION_SHAPES.issubset(
                manifest.development_shapes
            )
        )
        self.assertLess(
            len(manifest.development_shapes),
            len(load_cpu_prefill_split_manifest().development_shapes),
        )

    def test_sealed_plan_reaches_every_frozen_generic_leaf(self) -> None:
        """A timing-free preflight must reject structurally untested rules."""

        split = load_cpu_prefill_split_manifest()
        shapes = load_shape_manifest()
        routes = CPUPrefillSerialRouteManifest(
            CPUPrefillSerialRoute(
                execution_codebook=0,
                shape_name=shape_name,
                n=shapes.by_name(shape_name).n,
                k=shapes.by_name(shape_name).k,
                isa_regime="avx512-build.avx2-runtime",
                payload_bytes=16,
                threads=28,
                k_tiles=0,
                bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
            )
            for shape_name in split.sealed_shapes
        )
        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class=(
                "test-host|build=AVX512|runtime=AVX2|threads=28"
            ),
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
            prepared_family_id="NativeVNNI_cpu_CB0",
            packing_abi="native-vnni-cpu-cb0-v1",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=16384,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )

        def rule_above(work_items: int) -> GenericDispatchRule:
            return GenericDispatchRule(
                domain=domain,
                predicates=(FeaturePredicate(
                    FeatureThreshold(FeatureAxis.WORK_ITEMS, work_items),
                    require_less_equal=False,
                ),),
                candidate_id="cpu.nvnni.prefill.two_row_tiles.nbc1.full_k",
                arithmetic_fingerprint="sha256:test",
                development_shape_groups=("development:test",),
                development_max_regret=0.01,
                development_p95_regret=0.01,
                development_mean_regret=0.01,
            )

        plan = MODULE.build_cpu_prefill_sealed_witness_plan(
            (rule_above(9_500_000),),
            "sha256:frozen",
            split,
            routes,
        )
        self.assertEqual(len(plan.rule_witnesses), 1)
        self.assertEqual(len(plan.records), 1)
        self.assertEqual(
            plan.records[0].shape_name,
            "V8CPUPrefillSealed_VeryWide_17600x544",
        )
        with self.assertRaisesRegex(ValueError, "cannot exercise"):
            MODULE.build_cpu_prefill_sealed_witness_plan(
                (rule_above(10_000_000),),
                "sha256:frozen",
                split,
                routes,
            )

    @staticmethod
    def row(candidate: str, median_us: float) -> dict[str, object]:
        nbc = int(candidate.split("nbc", 1)[1].split(".", 1)[0]) if "nbc" in candidate else 10
        normalized = nbc == 16
        observed = (
            "cpu.nvnni.prefill.row_chunk_grid.full_k"
            if normalized or "row_chunk_grid" in candidate
            else candidate
        )
        supported = not normalized
        return {
            "backend": "cpu",
            "phase": "prefill_gemm",
            "source_format": "Q4_K",
            "source_codebook": 5,
            "execution_codebook": 5,
            "shape": "7B_FFN_Up",
            "execution_mode": "eager",
            "m": 64,
            "n": 18944,
            "k": 3584,
            "candidate_id": candidate,
            "build_isa": "AVX512",
            "runtime_isa_requested": "AVX512",
            "runtime_isa_effective": "AVX512",
            "threads": 28,
            "measurement_coordination": "process-local-complete-round-v1",
            "mpi_world_size": 1,
            "mpi_rank": 0,
            "weight_bytes": 1000000,
            "serial_oracle_policy": "boundary-quartile-sentinel-v1",
            "serial_oracle_rows": 16,
            "preconditioning_policy": "disabled",
            "preconditioning_m": 0,
            "preconditioning_budget_us": 0,
            "preconditioning_duration_us": 0,
            "warmup_count": 2 if supported else 0,
            "warmup_round_count": 0,
            "warmup_budget_policy": "disabled",
            "warmup_probe_latency_us": median_us,
            "warmup_budget_floor_us": 0,
            "warmup_latency_multiplier": 0,
            "warmup_budget_ceiling_us": 0,
            "warmup_budget_us": 0,
            "warmup_duration_us": 0,
            "global_warmup_round_count": 0,
            "warmup_wall_duration_us": 0,
            "warmup_max_round_duration_us": 0,
            "warmup_round_timeout_us": 30000000,
            "timing_order_seed": 12345,
            "global_timing_round_count": 3 if supported else 1,
            "sample_count": 3 if supported else 1,
            "timing_protocol": ADAPTIVE_TIMING_PROTOCOL,
            "timing_ceiling_policy": ADAPTIVE_TIMING_CEILING_POLICY,
            "min_sample_count": 3 if supported else 1,
            "stable_sample_count": 3 if supported else 1,
            "max_sample_count": 3 if supported else 1,
            "timing_budget_us": 0,
            "timed_duration_us": median_us * (3 if supported else 1),
            "stationary_sample_begin": 0,
            "stationary_sample_count": 3 if supported else 1,
            "stationary_duration_us": median_us * (3 if supported else 1),
            "median_stability_limit": 0.02,
            "median_relative_drift": 0,
            "timing_converged": 1 if supported else 0,
            "timing_stop_reason": (
                "stationary_window" if supported else "fixed_samples"
            ),
            "min_us": median_us,
            "median_us": median_us,
            "p95_us": median_us,
            "mad_us": 0,
            "cv": 0.01,
            "serial_median_us": 20,
            "speedup": 20 / median_us,
            "bit_mismatches": 0,
            "first_bit_mismatch": 0,
            "repeat_byte_mismatches": 0,
            "max_abs": 0,
            "relative_l2": 0,
            "cosine": 1,
            "symmetric_kld": 0,
            "grouped_output_digest": "fnv1a64:equal",
            "serial_output_digest": "fnv1a64:equal",
            "timing_sample_digest": f"fnv1a64:{nbc:016x}",
            "route_counter_ok": 1,
            "observed_candidate_id": observed,
            "k_tiles": 1,
            "k_tile_blocks": 112,
            "n_block_chunks": nbc,
            "numerical_correctness": 1,
            "correctness_pass": 1 if supported else 0,
            "is_winner": 0,
        }

    @classmethod
    def rows(cls) -> list[dict[str, object]]:
        return [
            cls.row("cpu.nvnni.prefill.row_chunk_grid.full_k", 9.0),
            cls.row("cpu.nvnni.prefill.two_row_tiles.nbc1.full_k", 8.0),
            cls.row("cpu.nvnni.prefill.two_row_tiles.nbc2.full_k", 7.0),
            cls.row("cpu.nvnni.prefill.two_row_tiles.nbc4.full_k", 5.0),
            cls.row("cpu.nvnni.prefill.two_row_tiles.nbc8.full_k", 6.0),
            cls.row("cpu.nvnni.prefill.two_row_tiles.nbc16.full_k", 4.0),
        ]

    def run_analyzer(self, *extra: str) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "prefill.csv"
            output = root / "generated.inc"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(self.rows()[0]))
                writer.writeheader()
                writer.writerows(self.rows())
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(source),
                    "--output",
                    str(output),
                    *extra,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            generated = output.read_text(encoding="utf-8") if output.exists() else ""
            return result, generated

    def test_fastest_forceable_full_k_candidate_is_emitted(self) -> None:
        result, generated = self.run_analyzer()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPUNativeVNNIPrefillPolicy::TwoRowNbc4", generated)
        self.assertNotIn(
            "7B_FFN_Up cpu.nvnni.prefill.two_row_tiles.nbc16.full_k",
            generated,
        )

    def test_generated_bucket_clamps_real_32b_ffn_but_not_small_models(self) -> None:
        _, generated = self.run_analyzer()

        self.assertIn("(n == 27648 && k == 5120)", generated)
        self.assertIn("maximum_m = 1024", generated)
        self.assertIn("return 16384", generated)

    def test_generated_bucket_is_total_for_unseen_runtime_m(self) -> None:
        """Compile the emitted resolver at interval and geometry boundaries."""

        result, generated = self.run_analyzer()
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include = root / "generated.inc"
            source = root / "bucket_test.cpp"
            include.write_text(generated, encoding="utf-8")
            source.write_text(
                """
#include "generated.inc"

using llaminar2::cpu::native_vnni::generated::bucketCPUNativeVNNIPrefillM;

static_assert(bucketCPUNativeVNNIPrefillM(1, 896, 896) == 64);
static_assert(bucketCPUNativeVNNIPrefillM(64, 896, 896) == 64);
static_assert(bucketCPUNativeVNNIPrefillM(65, 896, 896) == 256);
static_assert(bucketCPUNativeVNNIPrefillM(256, 896, 896) == 256);
static_assert(bucketCPUNativeVNNIPrefillM(257, 896, 896) == 1024);
static_assert(bucketCPUNativeVNNIPrefillM(1024, 896, 896) == 1024);
static_assert(bucketCPUNativeVNNIPrefillM(1025, 896, 896) == 2048);
static_assert(bucketCPUNativeVNNIPrefillM(2048, 896, 896) == 2048);
static_assert(bucketCPUNativeVNNIPrefillM(2049, 896, 896) == 4096);
static_assert(bucketCPUNativeVNNIPrefillM(4096, 896, 896) == 4096);
static_assert(bucketCPUNativeVNNIPrefillM(4097, 896, 896) == 8192);
static_assert(bucketCPUNativeVNNIPrefillM(8192, 896, 896) == 8192);
static_assert(bucketCPUNativeVNNIPrefillM(8193, 896, 896) == 16384);
static_assert(bucketCPUNativeVNNIPrefillM(16384, 896, 896) == 16384);
static_assert(bucketCPUNativeVNNIPrefillM(2147483647, 896, 896) == 16384);
static_assert(bucketCPUNativeVNNIPrefillM(2048, 27648, 5120) == 1024);
static_assert(bucketCPUNativeVNNIPrefillM(8192, 5120, 5120) == 4096);

int main() { return 0; }
""".lstrip(),
                encoding="utf-8",
            )
            compiled = subprocess.run(
                ["c++", "-std=c++20", "-fsyntax-only", str(source)],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)

    def test_prefill_key_preserves_sixteen_m_bits(self) -> None:
        low = MODULE._pack_key(5, 1024, 27648, 5120)
        high = MODULE._pack_key(5, 16384, 27648, 5120)

        self.assertNotEqual(low, high)
        self.assertEqual((high >> 40) & 0xFFFF, 16384)

    def test_generated_selector_prefers_geometry_overlay_before_generic_rule(self) -> None:
        """Runtime dispatch is codebook/geometry/work, exact first, then generic."""

        entry = MODULE.PrefillPolicyEntry(
            build_isa="AVX2",
            runtime_isa="AVX2",
            threads=28,
            codebook=0,
            m=64,
            n=512,
            k=512,
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
            policy="TwoRowNbc1",
            candidate_id="cpu.nvnni.prefill.two_row_tiles.nbc1.full_k",
            shape_name="QwenRelease_ExactOverlay",
            max_surface_regret=0.0,
            max_cv=0.0,
        )
        rule = GenericDispatchRule(
            domain=GenericDomain(
                backend=Backend.CPU,
                architecture_class=(
                    "test-host|build=AVX2|runtime=AVX2|threads=28"
                ),
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
                prepared_family_id="NativeVNNI_cpu_CB0",
                packing_abi="native-vnni-cpu-cb0-v1",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aspect_bucket=AspectBucket.BALANCED,
                all_aspects=True,
            ),
            predicates=(),
            candidate_id="cpu.nvnni.prefill.row_chunk_grid.full_k",
            arithmetic_fingerprint="serial-m1-test",
            development_shape_groups=("generic-a", "generic-b", "generic-c"),
            development_max_regret=0.0,
            development_p95_regret=0.0,
            development_mean_regret=0.0,
        )
        generated = MODULE.generate_include(
            [entry],
            [MODULE.CPUPrefillGenericRule("AVX2", "AVX2", 28, rule)],
            corpus_digest="sha256:corpus",
            registry_digest="sha256:registry",
            profile=MODULE.MeasurementProfile.QUICK,
        )

        self.assertLess(
            generated.index("switch (key)"),
            generated.index("const long long work_items"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include = root / "generated.inc"
            source = root / "dispatch_test.cpp"
            include.write_text(generated, encoding="utf-8")
            source.write_text(
                """
#include "generated.inc"

using namespace llaminar2::cpu::native_vnni::generated;

int main()
{
    CPUNativeVNNIPrefillPolicy policy =
        CPUNativeVNNIPrefillPolicy::KPartWideRows;
    if (!selectCPUNativeVNNIPrefillGeneratedPolicy(
            CPUNativeVNNIPrefillBuildISA::AVX2,
            CPUNativeVNNIPrefillRuntimeISA::AVX2,
            28, 0, 64, 512, 512, false, policy))
        return 1;
    if (policy != CPUNativeVNNIPrefillPolicy::TwoRowNbc1)
        return 2;
    if (!selectCPUNativeVNNIPrefillGeneratedPolicy(
            CPUNativeVNNIPrefillBuildISA::AVX2,
            CPUNativeVNNIPrefillRuntimeISA::AVX2,
            28, 0, 64, 640, 512, false, policy))
        return 3;
    return policy == CPUNativeVNNIPrefillPolicy::RowChunkGrid ? 0 : 4;
}
""".lstrip(),
                encoding="utf-8",
            )
            executable = root / "dispatch_test"
            compiled = subprocess.run(
                ["c++", "-std=c++20", str(source), "-o", str(executable)],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(executable)],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_generator_has_explicit_serial_kpart_policies(self) -> None:
        self.assertEqual(
            MODULE._candidate_policy(
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise"
            ),
            "KPartPairwise",
        )
        self.assertEqual(
            MODULE._candidate_policy(
                "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows"
            ),
            "KPartWideRows",
        )

    def test_generated_abi_authenticates_route_and_m2_capability(self) -> None:
        """Exact entries cannot ignore the production serial arithmetic route."""

        result, generated = self.run_analyzer()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("enum class CPUNativeVNNIPrefillBuildISA", generated)
        self.assertIn("enum class CPUNativeVNNIPrefillRuntimeISA", generated)
        self.assertNotIn("enum class CPUNativeVNNIBuildISA", generated)
        self.assertIn(
            "#define LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED 0",
            generated,
        )
        self.assertIn("if (serial_kpart && m < 3)", generated)
        self.assertIn("if (serial_kpart != false)", generated)
        self.assertNotIn(
            "if (serial_kpart)\n        {\n"
            "            policy = CPUNativeVNNIPrefillPolicy::KPartPairwise;",
            generated,
        )

    def test_only_sealed_certificate_enables_production_policy(self) -> None:
        """A frozen development include must never masquerade as certified."""

        certificate = MODULE.CertificationReport(
            frozen_generic_policy_digest="sha256:frozen",
            cells=(),
            sealed_cell_count=0,
            out_of_scope_cell_count=0,
            required_cell_count=0,
            covered_cell_count=0,
            verifier_bitwise_failures=0,
            unexercised_rule_count=0,
            unpromoted_domain_count=0,
        )
        generated = MODULE.generate_include(
            [],
            [],
            corpus_digest="sha256:corpus",
            registry_digest="sha256:registry",
            profile=MODULE.MeasurementProfile.PRODUCTION,
            policy_digest="sha256:policy",
            certification=certificate,
        )

        self.assertIn(
            "#define LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED 1",
            generated,
        )

    def test_require_complete_rejects_a_tiny_smoke_corpus(self) -> None:
        result, _ = self.run_analyzer("--require-complete")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires production C++ serial-route manifests", result.stderr)

    def test_production_rejects_obsolete_one_pass_policy_emission(self) -> None:
        result, _ = self.run_analyzer("--profile", "production")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires development freeze and separate sealed certification",
            result.stderr,
        )

    def test_production_freeze_requires_profiler_evidence(self) -> None:
        result, _ = self.run_analyzer(
            "--profile",
            "production",
            "--freeze-generic",
            "--policy-json",
            "/tmp/unused-cpu-prefill-policy.json",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires complete development profiler evidence",
            result.stderr,
        )

    def test_nontotal_generic_freeze_writes_no_artifacts(self) -> None:
        """Totality must fail before witness derivation or file publication."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "frozen.inc"
            policy = root / "frozen.policy.json"
            witness = root / "sealed.witness.json"
            frozen = SimpleNamespace(
                policy_ir=SimpleNamespace(generic_rules=()),
                generic_digest="sha256:incomplete",
            )
            context = SimpleNamespace(
                serial_m1_policy_hash="sha256:serial",
                profile=MODULE.MeasurementProfile.PRODUCTION,
            )
            argv = [
                str(ANALYZER),
                str(root / "development.csv"),
                "--output",
                str(output),
                "--profile",
                "production",
                "--freeze-generic",
                "--policy-json",
                str(policy),
                "--sealed-witness-plan-json",
                str(witness),
                "--route-manifest",
                str(root / "routes.csv"),
                "--development-profiler-requests",
                str(root / "profiler.requests.json"),
                "--development-profiler-evidence",
                str(root / "profiler.evidence.json"),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    MODULE,
                    "read_cpu_prefill_route_manifests",
                    return_value=object(),
                ),
                mock.patch.object(
                    MODULE,
                    "load_cpu_prefill_split_manifest",
                    return_value=object(),
                ),
                mock.patch.object(
                    MODULE,
                    "_context_from_args",
                    return_value=context,
                ),
                mock.patch.object(
                    MODULE,
                    "adapt_cpu_prefill_csv",
                    return_value=object(),
                ),
                mock.patch.object(
                    MODULE,
                    "_load_development_profiler_catalogs",
                    return_value=object(),
                ),
                mock.patch.object(
                    MODULE,
                    "freeze_cpu_prefill_policy",
                    return_value=frozen,
                ),
                mock.patch.object(MODULE, "select_entries", return_value=[]),
                mock.patch.object(MODULE, "validate_emitter_inputs"),
                mock.patch.object(
                    MODULE,
                    "build_cpu_prefill_sealed_witness_plan",
                ) as build_witness,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "incomplete ISA groups",
                ):
                    MODULE.main()

            build_witness.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(policy.exists())
            self.assertFalse(witness.exists())

    def test_fit_cache_option_preserves_nonproduction_workflow(self) -> None:
        result, _ = self.run_analyzer("--fit-cache-dir", "/tmp/prefill-fit-cache")

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generic_leaf_budget_rejects_unreviewed_capacity(self) -> None:
        """The CLI must not bypass the common compiler's reviewed tree bound."""

        result, _ = self.run_analyzer(
            "--generic-max-leaves",
            str(MODULE.MAX_TREE_LEAVES + 1),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            f"must be between 1 and {MODULE.MAX_TREE_LEAVES}",
            result.stderr,
        )

    def test_freeze_records_and_forwards_generic_leaf_budget(self) -> None:
        """Fit capacity is policy provenance, not an unrecorded CLI detail."""

        generic_development = object()
        development = mock.Mock()
        development.with_collapsed_aspect_domains.return_value = (
            generic_development
        )
        split_manifest = SimpleNamespace(
            schema_version="cpu-prefill-split-test",
            sealed_shapes=(),
            sealed_m_values=(),
            digest=lambda: "sha256:split",
        )
        expected = SimpleNamespace(
            policy_ir=mock.sentinel.unfiltered_policy_ir,
            promotion_diagnostics=(mock.sentinel.diagnostic,),
        )
        with (
            mock.patch.object(
                MODULE,
                "_require_partition",
                return_value=development,
            ),
            mock.patch.object(MODULE, "_serial_hashes", return_value={}),
            mock.patch.object(
                MODULE,
                "freeze_policy",
                return_value=expected,
            ) as freeze_policy,
            mock.patch.object(
                MODULE,
                "_retain_production_exact_overlays",
                return_value=mock.sentinel.filtered_policy_ir,
            ) as retain_overlays,
        ):
            actual = MODULE.freeze_cpu_prefill_policy(
                development,
                object(),
                split_manifest,
                "sha256:serial",
                None,
                max_leaves=MODULE.MAX_TREE_LEAVES,
            )

        self.assertIs(actual.policy_ir, mock.sentinel.filtered_policy_ir)
        self.assertEqual(actual.promotion_diagnostics, expected.promotion_diagnostics)
        retain_overlays.assert_called_once_with(expected.policy_ir)
        self.assertEqual(
            freeze_policy.call_args.kwargs["max_leaves"],
            MODULE.MAX_TREE_LEAVES,
        )
        self.assertEqual(
            freeze_policy.call_args.kwargs["metadata"]["generic_max_leaves"],
            MODULE.MAX_TREE_LEAVES,
        )

    def test_profiler_ablation_is_restricted_to_development_fit(self) -> None:
        """Profiler omission must be an explicit experiment, never production."""

        result, _ = self.run_analyzer("--ablate-profiler-features")

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "restricted to development fit mode",
            result.stderr,
        )

    @staticmethod
    def _total_policy_fixture() -> tuple[
        list[object],
        list[MODULE.CPUPrefillGenericRule],
    ]:
        """Build a complete synthetic policy with irrelevant exact overlays."""

        codebooks = sorted({
            cell.runtime_codebook
            for cell in MODULE.cpu_prefill_runtime_training_cells()
        })
        groups = (
            ("AVX2", "AVX2", 28),
            ("AVX512", "AVX2", 28),
            ("AVX512", "AVX512", 28),
        )
        entries = [
            MODULE.PrefillPolicyEntry(
                build_isa=build,
                runtime_isa=runtime,
                threads=threads,
                codebook=codebook,
                m=m,
                n=measurement.shape.n,
                k=measurement.shape.k,
                bundle_signature=MODULE.CPU_PREFILL_FULL_K_BUNDLE,
                policy="RowChunkGrid",
                candidate_id="cpu.nvnni.prefill.row_chunk_grid.full_k",
                shape_name=measurement.shape.name,
                max_surface_regret=0.0,
                max_cv=0.0,
            )
            for build, runtime, threads in groups
            for codebook in codebooks
            for measurement in MODULE.cpu_prefill_measurements()
            for m in measurement.m_values
        ]
        m_values = sorted({
            m
            for measurement in MODULE.cpu_prefill_measurements()
            for m in measurement.m_values
        })
        generic_rules = [
            MODULE.CPUPrefillGenericRule(
                build_isa=build,
                runtime_isa=runtime,
                threads=threads,
                rule=GenericDispatchRule(
                    domain=GenericDomain(
                        backend=Backend.CPU,
                        architecture_class=(
                            f"test-host|build={build}|runtime={runtime}|"
                            f"threads={threads}"
                        ),
                        semantic_contract=(
                            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                        ),
                        operation_kind="NativeVNNIPrefillProjection",
                        bundle_signature=bundle,
                        prepared_family_id=f"NativeVNNI_cpu_CB{codebook}",
                        packing_abi=f"native-vnni-cpu-cb{codebook}-v1",
                        runtime_codebook_id=codebook,
                        execution_mode=ExecutionMode.EAGER,
                        m=m,
                        aspect_bucket=AspectBucket.BALANCED,
                        all_aspects=True,
                    ),
                    predicates=(),
                    candidate_id=(
                        "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise"
                        if bundle == CPU_PREFILL_KPART_BUNDLE
                        else "cpu.nvnni.prefill.row_chunk_grid.full_k"
                    ),
                    arithmetic_fingerprint="sha256:test",
                    development_shape_groups=("development:test",),
                    development_max_regret=0.0,
                    development_p95_regret=0.0,
                    development_mean_regret=0.0,
                ),
            )
            for build, runtime, threads in groups
            for codebook in codebooks
            for m in m_values
            for bundle in (
                CPU_PREFILL_FULL_K_BUNDLE,
                CPU_PREFILL_KPART_BUNDLE,
            )
        ]

        return entries, generic_rules

    def test_exact_overlays_do_not_satisfy_generic_totality(self) -> None:
        """A complete exact table cannot substitute for learned dispatch."""

        entries, generic_rules = self._total_policy_fixture()

        MODULE.validate_total_policy(entries, generic_rules)
        with self.assertRaisesRegex(ValueError, "generic policy is not total"):
            MODULE.validate_total_policy(entries, generic_rules[:-1])

    def test_generic_policy_requires_every_runtime_codebook(self) -> None:
        """Deleting one codebook must leave a visible generic coverage hole."""

        entries, generic_rules = self._total_policy_fixture()
        missing_codebook = min(
            item.rule.domain.runtime_codebook_id for item in generic_rules
        )
        incomplete = [
            item
            for item in generic_rules
            if item.rule.domain.runtime_codebook_id != missing_codebook
        ]

        with self.assertRaisesRegex(ValueError, "generic policy is not total"):
            MODULE.validate_total_policy(entries, incomplete)

    def test_generic_policy_requires_every_runtime_m_bucket(self) -> None:
        """Deleting one measured M domain must not create a dispatch interval hole."""

        entries, generic_rules = self._total_policy_fixture()
        target = generic_rules[0]
        incomplete = [
            item
            for item in generic_rules
            if not (
                item.build_isa == target.build_isa
                and item.runtime_isa == target.runtime_isa
                and item.threads == target.threads
                and item.rule.domain.runtime_codebook_id
                == target.rule.domain.runtime_codebook_id
                and item.rule.domain.m == target.rule.domain.m
                and item.rule.domain.bundle_signature
                == target.rule.domain.bundle_signature
            )
        ]

        with self.assertRaisesRegex(ValueError, "generic policy is not total"):
            MODULE.validate_total_policy(entries, incomplete)

    def test_generic_policy_rejects_nk_partition_hole(self) -> None:
        """One-sided geometry predicates cannot claim unseen N-by-K totality."""

        entries, generic_rules = self._total_policy_fixture()
        target = generic_rules[0]
        one_sided_rule = replace(
            target.rule,
            predicates=(FeaturePredicate(
                FeatureThreshold(FeatureAxis.WORK_ITEMS, 896 * 896),
                require_less_equal=True,
            ),),
        )
        malformed = [
            replace(item, rule=one_sided_rule) if item == target else item
            for item in generic_rules
        ]

        with self.assertRaisesRegex(ValueError, "unseen geometry uncovered"):
            MODULE.validate_total_policy(entries, malformed)


if __name__ == "__main__":
    unittest.main()
