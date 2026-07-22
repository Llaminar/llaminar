#!/usr/bin/env python3
"""Regress fresh-route CPU M=1 frozen-leaf certification plans."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.cpu_decode_sealed_plan import (  # noqa: E402
    CPU_DECODE_SEALED_PROBE_SCHEMA,
    CPU_DECODE_SEALED_GEOMETRY_PREFIX,
    LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA,
    SERIAL_FULL_K_BUNDLE,
    SERIAL_KPART_BUNDLE,
    build_cpu_decode_sealed_plan,
    build_cpu_decode_sealed_route_probe,
    cpu_decode_burned_seal_costs,
    read_cpu_decode_sealed_plan,
    read_cpu_decode_sealed_route_probe,
    validate_cpu_decode_sealed_plan,
    write_cpu_decode_sealed_plan,
    write_cpu_decode_sealed_route_probe,
)
from native_vnni_dispatch.cpu_sealed_paired import (  # noqa: E402
    fresh_geometry_candidates_for_rule,
    write_cpu_sealed_request_shards,
)
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
    REQUIRED_COLUMNS,
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


ARCHITECTURE = "test-host|build=AVX512|runtime=AVX512|threads=28"
ISA_REGIME = "avx512-build.avx512-runtime"
FROZEN_DIGEST = "sha256:" + "d" * 64
DEVELOPMENT_DIGEST = "sha256:" + "e" * 64


class _Development:
    def __init__(self, rows, domains=(), digest=DEVELOPMENT_DIGEST):
        self._rows = tuple(rows)
        self._domains = tuple(domains)
        self._digest = digest

    def __iter__(self):
        return iter(self._rows)

    def digest(self) -> str:
        return self._digest

    def generic_domains(self):
        return self._domains


class _Manifest:
    maximum_cpu_measurement_weight_elements = 32_000_000
    shapes = (SimpleNamespace(n=256, k=256),)

    @staticmethod
    def digest() -> str:
        return "sha256:" + "f" * 64


def _rule(
    index: int,
    *,
    bundle: str,
    candidate: str,
    predicates=(),
) -> GenericDispatchRule:
    return GenericDispatchRule(
        domain=GenericDomain(
            backend=Backend.CPU,
            architecture_class=ARCHITECTURE,
            semantic_contract=SemanticContract.FAST,
            operation_kind="NativeVNNIFastM1Projection",
            bundle_signature=bundle,
            prepared_family_id="NativeVNNI_cpu_CB4",
            packing_abi="native-vnni-cpu-cb4-v1",
            runtime_codebook_id=4,
            execution_mode=ExecutionMode.EAGER,
            m=1,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        ),
        predicates=predicates,
        candidate_id=candidate,
        arithmetic_fingerprint=f"sha256:test-decode-{index}",
        development_shape_groups=(f"group-{index}",),
        development_max_regret=0.01,
        development_p95_regret=0.01,
        development_mean_regret=0.01,
    )


class CPUDecodeSealedPlanTest(unittest.TestCase):
    """Prove M=1 seals bind unseen geometry to the physical serial route."""

    def setUp(self) -> None:
        self.sealed_build_id = "sha256:" + "b" * 64
        self.rules = (
            _rule(
                0,
                bundle=SERIAL_FULL_K_BUNDLE,
                candidate="cpu.nvnni.decode.n_chunk_grid.nbc4",
            ),
            _rule(
                1,
                bundle=SERIAL_KPART_BUNDLE,
                candidate="cpu.nvnni.decode.n_chunk_grid.nbc16",
                predicates=(FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.KPART_PRODUCER_WAVES_64,
                        2,
                        1,
                        28,
                    ),
                    False,
                ),),
            ),
        )
        self.development = _Development((
            SimpleNamespace(
                shape_group_id="group-0", aggregate_n=512, k=512,
                architecture_class=ARCHITECTURE,
                runtime_codebook_id=4,
                bundle_signature=SERIAL_FULL_K_BUNDLE,
                operation_kind="NativeVNNIFastM1Projection",
                execution_mode=ExecutionMode.EAGER,
                m=1,
                launch_k_tiles=0,
            ),
            SimpleNamespace(
                shape_group_id="group-1", aggregate_n=2048, k=4096,
                architecture_class=ARCHITECTURE,
                runtime_codebook_id=4,
                bundle_signature=SERIAL_KPART_BUNDLE,
                operation_kind="NativeVNNIFastM1Projection",
                execution_mode=ExecutionMode.EAGER,
                m=1,
                launch_k_tiles=2,
            ),
        ))
        self.manifest = _Manifest()
        self.probe = build_cpu_decode_sealed_route_probe(
            self.rules,
            FROZEN_DIGEST,
            self.development,
            self.manifest,
            self.sealed_build_id,
        )

    def test_cpp_route_planner_accepts_current_probe_schema(self) -> None:
        """Keep the Python reserve producer and C++ route reader in lockstep."""

        source = (
            REPO_ROOT
            / "tests/v2/performance/kernels/cpu/native_vnni"
            / "Perf__CPUNativeVNNI_GEMV.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn(
            f'probe_schema == "{CPU_DECODE_SEALED_PROBE_SCHEMA}"',
            source,
        )

    @staticmethod
    def _write_routes(path: Path, probe, *, omit_large: bool = False) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
            writer.writeheader()
            for shape in probe.shapes:
                kpart = shape.n > 1000
                if omit_large and kpart:
                    continue
                writer.writerow({
                    "schema_version": "cpu-prefill-serial-route-v1",
                    "shape": shape.name,
                    "n": shape.n,
                    "k": shape.k,
                    "execution_codebook": 4,
                    "payload_bytes": 16,
                    "threads": 28,
                    "isa_regime": ISA_REGIME,
                    "k_tiles": 2 if kpart else 0,
                    "bundle_signature": (
                        CPU_PREFILL_KPART_BUNDLE
                        if kpart
                        else CPU_PREFILL_FULL_K_BUNDLE
                    ),
                })

    def test_probe_and_plan_round_trip_cover_both_serial_routes(self) -> None:
        self.assertTrue(all(
            shape.name.startswith(CPU_DECODE_SEALED_GEOMETRY_PREFIX)
            for shape in self.probe.shapes
        ))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_path = root / "probe.json"
            routes = root / "routes.csv"
            plan_path = root / "plan.json"
            write_cpu_decode_sealed_route_probe(probe_path, self.probe)
            restored_probe = read_cpu_decode_sealed_route_probe(probe_path)
            self._write_routes(routes, restored_probe)
            plan = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                restored_probe,
                (routes,),
                self.sealed_build_id,
            )
            validate_cpu_decode_sealed_plan(
                plan,
                self.rules,
                self.development,
                self.manifest,
                self.sealed_build_id,
            )
            shards = write_cpu_decode_sealed_plan(
                plan_path, plan, root / "shards", max_requests_per_shard=8
            )
            restored_plan = read_cpu_decode_sealed_plan(plan_path)

        self.assertEqual(
            {item.bundle_signature for item in restored_plan.rule_witnesses},
            {SERIAL_FULL_K_BUNDLE, SERIAL_KPART_BUNDLE},
        )
        self.assertGreater(len(shards), 0)

    def test_replanned_request_directory_removes_orphaned_evidence(self) -> None:
        """A live seal directory must contain only one plan generation."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            routes = root / "routes.csv"
            self._write_routes(routes, self.probe)
            plan = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                self.probe,
                (routes,),
                self.sealed_build_id,
            )
            shard_dir = root / "shards"
            old_paths = write_cpu_sealed_request_shards(
                shard_dir,
                plan.requests,
                plan.development_corpus_digest,
                plan.plan_digest,
                "unit-index-v1",
            )
            stale_csv = old_paths[0].with_name(
                old_paths[0].name.removesuffix(".requests.json") + ".csv"
            )
            stale_inprogress = stale_csv.with_name(
                stale_csv.name + ".inprogress"
            )
            stale_csv.write_text("completed stale evidence\n", encoding="utf-8")
            stale_inprogress.write_text(
                "partial stale evidence\n", encoding="utf-8"
            )
            changed_first = replace(
                plan.requests[0],
                request=replace(
                    plan.requests[0].request,
                    request_id=plan.requests[0].request.request_id + "-changed",
                ),
            )
            new_paths = write_cpu_sealed_request_shards(
                shard_dir,
                (changed_first, *plan.requests[1:]),
                plan.development_corpus_digest,
                plan.plan_digest,
                "unit-index-v1",
            )

            self.assertFalse(stale_csv.exists())
            self.assertFalse(stale_inprogress.exists())
            self.assertNotEqual(old_paths[0].name, new_paths[0].name)
            self.assertEqual(
                len(tuple(shard_dir.glob("shard-*.requests.json"))),
                len(new_paths),
            )

    def test_probe_indexes_development_corpus_once(self) -> None:
        """Leaf planning must not rescan the complete corpus per rule."""

        class CountingDevelopment(_Development):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        development = CountingDevelopment(tuple(self.development))
        probe = build_cpu_decode_sealed_route_probe(
            self.rules,
            FROZEN_DIGEST,
            development,
            self.manifest,
            self.sealed_build_id,
        )

        self.assertTrue(probe.shapes)
        self.assertEqual(development.iterations, 1)

    def test_legacy_plan_recovers_exact_k_tiles_from_authenticated_routes(
        self,
    ) -> None:
        """A v2 burned seal must not invent missing K-part geometry."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            burned = root / "burned-generation"
            burned.mkdir()
            routes = root / (
                "cpu_decode_m1.sealed-route."
                "avx512-build.avx512-runtime.0123456789ab.csv"
            )
            self._write_routes(routes, self.probe)
            current = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                self.probe,
                (routes,),
                self.sealed_build_id,
            )
            raw = current.canonical_mapping()
            raw["schema_version"] = LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA
            for witness in raw["rule_witnesses"]:
                witness.pop("k_tiles")
            seed = dict(raw)
            seed.pop("plan_digest")
            seed.pop("requests")
            raw["plan_digest"] = "sha256:" + hashlib.sha256(json.dumps(
                seed, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            plan_path = burned / "cpu_decode_m1.sealed-plan.json"
            plan_path.write_text(json.dumps(raw), encoding="utf-8")

            restored = read_cpu_decode_sealed_plan(plan_path)

        self.assertEqual(
            tuple(item.k_tiles for item in restored.rule_witnesses),
            tuple(item.k_tiles for item in current.rule_witnesses),
        )

    def test_legacy_plan_rejects_missing_authenticated_routes(self) -> None:
        """Absent v2 geometry lineage must fail instead of becoming zero."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            routes = root / "routes.csv"
            self._write_routes(routes, self.probe)
            current = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                self.probe,
                (routes,),
                self.sealed_build_id,
            )
            raw = current.canonical_mapping()
            raw["schema_version"] = LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA
            for witness in raw["rule_witnesses"]:
                witness.pop("k_tiles")
            seed = dict(raw)
            seed.pop("plan_digest")
            seed.pop("requests")
            raw["plan_digest"] = "sha256:" + hashlib.sha256(json.dumps(
                seed, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            plan_path = root / "cpu_decode_m1.sealed-plan.json"
            plan_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "requires its authenticated route manifests"
            ):
                read_cpu_decode_sealed_plan(plan_path)

    def test_missing_physical_route_rejects_a_frozen_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, self.probe, omit_large=True)
            with self.assertRaisesRegex(ValueError, "cannot exercise frozen rule"):
                build_cpu_decode_sealed_plan(
                    self.rules,
                    FROZEN_DIGEST,
                    self.development,
                    self.manifest,
                    self.probe,
                    (routes,),
                    self.sealed_build_id,
                )

    def test_burned_dimensions_are_excluded_from_the_next_route_probe(self) -> None:
        """A previously inspected seal geometry can never seal a later fit."""

        burned = {
            f"burned-{index}": (shape.n, shape.k)
            for index, shape in enumerate(self.probe.shapes)
        }
        next_probe = build_cpu_decode_sealed_route_probe(
            self.rules,
            FROZEN_DIGEST,
            self.development,
            self.manifest,
            self.sealed_build_id,
            burned,
        )

        self.assertTrue(all(
            (shape.n, shape.k) not in set(burned.values())
            for shape in next_probe.shapes
        ))

    def test_burned_kpart_anchor_retains_its_launch_tile_geometry(self) -> None:
        """A later leaf may be anchored solely by inspected K-part evidence."""

        predicate = FeaturePredicate(
            FeatureThreshold(FeatureAxis.KPART_K_TILE_COUNT, 1),
            False,
        )
        rule = replace(
            _rule(
                0,
                bundle=SERIAL_KPART_BUNDLE,
                candidate="cpu.nvnni.decode.n_chunk_grid.nbc4",
                predicates=(predicate,),
            ),
            development_shape_groups=("burned-kpart-only",),
        )
        development = _Development((), (rule.domain,))
        dimensions = {"burned-kpart-only": (2048, 4096)}
        runtime = SimpleNamespace(
            architecture_class=rule.domain.architecture_class,
            runtime_codebook_id=rule.domain.runtime_codebook_id,
            bundle_signature=rule.domain.bundle_signature,
            operation_kind=rule.domain.operation_kind,
            execution_mode=rule.domain.execution_mode,
            m=rule.domain.m,
            launch_k_tiles=2,
        )
        supplemental = {
            rule.domain: (
                SimpleNamespace(
                    shape_group_id="burned-kpart-only",
                    runtime_key=runtime,
                ),
            ),
        }

        # Missing launch telemetry reproduces the old coherence bug: the
        # anchor is interpreted as one full-K tile and no point can satisfy the
        # leaf's explicit multi-tile predicate.
        with self.assertRaisesRegex(ValueError, "found=0 required=1"):
            build_cpu_decode_sealed_route_probe(
                (rule,),
                FROZEN_DIGEST,
                development,
                self.manifest,
                self.sealed_build_id,
                dimensions,
            )

        probe = build_cpu_decode_sealed_route_probe(
            (rule,),
            FROZEN_DIGEST,
            development,
            self.manifest,
            self.sealed_build_id,
            dimensions,
            supplemental,
        )
        self.assertGreaterEqual(len(probe.shapes), 12)
        self.assertTrue(all(
            rule.matches(shape.n, shape.k, runtime.launch_k_tiles)
            for shape in probe.shapes
        ))

    def test_burned_seal_informs_an_additively_extended_corpus(self) -> None:
        """An M=1 burned generation must survive later generic geometry."""

        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, self.probe)
            plan = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                self.probe,
                (routes,),
                self.sealed_build_id,
            )
        extended = _Development(
            (*tuple(self.development), SimpleNamespace(
                shape_group_id="later-development-geometry",
                aggregate_n=2080,
                k=11040,
            )),
            tuple(rule.domain for rule in self.rules),
            digest="sha256:" + "a" * 64,
        )
        self.assertNotEqual(plan.development_corpus_digest, extended.digest())

        def project_domains(
            _plan_digest,
            witnesses,
            _requests,
            _evidence_paths,
            domain_for_witness,
            _expected_path,
        ):
            return {
                domain_for_witness(witness): ()
                for witness in witnesses
            }

        with mock.patch(
            "native_vnni_dispatch.cpu_decode_sealed_plan."
            "cpu_sealed_development_costs",
            side_effect=project_domains,
        ):
            costs = cpu_decode_burned_seal_costs(extended, plan, ())

        self.assertEqual(set(costs), {rule.domain for rule in self.rules})

    def test_plan_uses_first_geometry_that_preserves_normalized_schedule(
        self,
    ) -> None:
        """Witness selection must mirror the runtime power-of-two boundary."""

        rule = _rule(
            0,
            bundle=SERIAL_FULL_K_BUNDLE,
            candidate="cpu.nvnni.decode.n_chunk_grid.nbc4",
        )
        development = _Development((SimpleNamespace(
            shape_group_id="group-0", aggregate_n=256, k=512
        ),))
        probe = build_cpu_decode_sealed_route_probe(
            (rule,),
            FROZEN_DIGEST,
            development,
            self.manifest,
            self.sealed_build_id,
        )
        self.assertTrue(any(shape.n <= 192 for shape in probe.shapes))
        self.assertTrue(any(shape.n >= 129 for shape in probe.shapes))
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, probe)
            plan = build_cpu_decode_sealed_plan(
                (rule,),
                FROZEN_DIGEST,
                development,
                self.manifest,
                probe,
                (routes,),
                self.sealed_build_id,
            )

        self.assertGreaterEqual(plan.rule_witnesses[0].n, 129)

    def test_nbc16_is_forceable_for_thirteen_n_chunks(self) -> None:
        """Regress the production N=800 rule that first exposed this drift."""

        rule = _rule(
            0,
            bundle=SERIAL_FULL_K_BUNDLE,
            candidate="cpu.nvnni.decode.n_chunk_grid.nbc16",
        )
        development = _Development((SimpleNamespace(
            shape_group_id="group-0", aggregate_n=800, k=2944
        ),))
        probe = build_cpu_decode_sealed_route_probe(
            (rule,),
            FROZEN_DIGEST,
            development,
            self.manifest,
            self.sealed_build_id,
        )
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, probe)
            plan = build_cpu_decode_sealed_plan(
                (rule,),
                FROZEN_DIGEST,
                development,
                self.manifest,
                probe,
                (routes,),
                self.sealed_build_id,
            )

        witness = plan.rule_witnesses[0]
        self.assertGreaterEqual(witness.n, 513)
        self.assertIn(rule.candidate_id, witness.forceable_candidate_ids)

    def test_plan_requires_a_physical_challenger_for_every_leaf(self) -> None:
        """Even an nbc1 leaf must select a geometry with two schedules."""

        rule = _rule(
            0,
            bundle=SERIAL_FULL_K_BUNDLE,
            candidate="cpu.nvnni.decode.n_chunk_grid.nbc1",
        )
        development = _Development((SimpleNamespace(
            shape_group_id="group-0", aggregate_n=96, k=512
        ),))
        probe = build_cpu_decode_sealed_route_probe(
            (rule,),
            FROZEN_DIGEST,
            development,
            self.manifest,
            self.sealed_build_id,
        )
        self.assertTrue(any(shape.n <= 64 for shape in probe.shapes))
        self.assertTrue(any(shape.n >= 65 for shape in probe.shapes))
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, probe)
            plan = build_cpu_decode_sealed_plan(
                (rule,),
                FROZEN_DIGEST,
                development,
                self.manifest,
                probe,
                (routes,),
                self.sealed_build_id,
            )

        self.assertGreaterEqual(plan.rule_witnesses[0].n, 65)
        self.assertGreaterEqual(
            len(plan.rule_witnesses[0].forceable_candidate_ids), 2
        )
        self.assertGreater(len(plan.requests), 0)

    def test_missing_alias_challenger_edge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, self.probe)
            plan = build_cpu_decode_sealed_plan(
                self.rules,
                FROZEN_DIGEST,
                self.development,
                self.manifest,
                self.probe,
                (routes,),
                self.sealed_build_id,
            )
        incomplete = replace(plan, requests=plan.requests[1:])
        with self.assertRaisesRegex(ValueError, "challenger matrix incomplete"):
            validate_cpu_decode_sealed_plan(
                incomplete,
                self.rules,
                self.development,
                self.manifest,
                self.sealed_build_id,
            )

    def test_fresh_reserve_covers_narrow_two_dimensional_leaf(self) -> None:
        """A narrow work interval must not be limited to four search rays."""

        base_n = 2048
        base_k = 11008
        narrow = replace(
            self.rules[0],
            predicates=(
                FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.WORK_ITEMS,
                        numerator=22_188_032,
                    ),
                    require_less_equal=False,
                ),
                FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.WORK_ITEMS,
                        numerator=22_609_920,
                    ),
                    require_less_equal=True,
                ),
            ),
            development_shape_groups=("narrow-work-leaf",),
        )
        forbidden = frozenset({(base_n, base_k), (2048, 10_944)})
        arguments = (
            narrow,
            {"narrow-work-leaf": (base_n, base_k)},
            forbidden,
            32_000_000,
            12,
        )

        first = fresh_geometry_candidates_for_rule(*arguments)
        second = fresh_geometry_candidates_for_rule(*arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertTrue(all(item not in forbidden for item in first))
        self.assertTrue(all(narrow.matches(n, k) for n, k in first))
        self.assertTrue(all(k % 32 == 0 for _, k in first))
        self.assertTrue(all(n * k <= 32_000_000 for n, k in first))
        self.assertTrue(any(
            n != base_n
            and k != base_k
            and abs(n - base_n) != abs(k - base_k)
            for n, k in first
        ))

    def test_decode_reserve_keeps_finite_leaf_with_ten_fresh_points(self) -> None:
        """Twelve is probe redundancy, not a minimum seal sample count."""

        groups = ("finite-leaf-0", "finite-leaf-1")
        rule = replace(
            self.rules[1],
            candidate_id="cpu.nvnni.decode.n_chunk_grid.nbc4",
            predicates=(
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.AGGREGATE_N, numerator=128),
                    require_less_equal=False,
                ),
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.AGGREGATE_N, numerator=140),
                    require_less_equal=True,
                ),
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.K, numerator=480),
                    require_less_equal=False,
                ),
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.K, numerator=512),
                    require_less_equal=True,
                ),
            ),
            development_shape_groups=groups,
        )
        dimensions = {
            groups[0]: (129, 512),
            groups[1]: (130, 512),
        }

        candidates = fresh_geometry_candidates_for_rule(
            rule,
            dimensions,
            frozenset(dimensions.values()),
            self.manifest.maximum_cpu_measurement_weight_elements,
            12,
            n_quanta=(1, 32),
            minimum_count=1,
        )

        self.assertEqual(len(candidates), 10)
        self.assertEqual(candidates[0], (131, 512))
        self.assertEqual(candidates[-1], (140, 512))
        self.assertTrue(all(rule.matches(n, k) for n, k in candidates))

    def test_decode_reserve_uses_complete_unit_n_lattice(self) -> None:
        """A valid narrow-N leaf must retain twelve unseen holdout points.

        This is the geometry of the production failure that originally left
        the frozen seal with zero candidates. Both development anchors are
        multiples of 32, but their learned aspect/work interval contains no
        other point on a 32-value N lattice. NativeVNNI supports every positive
        output-row count, so the M=1 reserve must search N at unit granularity
        while retaining the quantized K block boundary.
        """

        groups = ("narrow-n-0", "narrow-n-1")
        predicates = (
            FeaturePredicate(
                FeatureThreshold(
                    FeatureAxis.ASPECT_RATIO,
                    numerator=1567,
                    denominator=8526,
                ),
                require_less_equal=False,
            ),
            FeaturePredicate(
                FeatureThreshold(
                    FeatureAxis.ASPECT_RATIO,
                    numerator=2427,
                    denominator=13148,
                ),
                require_less_equal=False,
            ),
            FeaturePredicate(
                FeatureThreshold(FeatureAxis.K, numerator=10_912),
                require_less_equal=False,
            ),
            FeaturePredicate(
                FeatureThreshold(
                    FeatureAxis.WORK_ITEMS,
                    numerator=22_511_616,
                ),
                require_less_equal=True,
            ),
        )
        rule = replace(
            self.rules[1],
            candidate_id="cpu.nvnni.decode.n_chunk_grid.nbc4",
            predicates=predicates,
            development_shape_groups=groups,
        )
        dimensions = {
            groups[0]: (2048, 10_944),
            groups[1]: (2048, 10_976),
        }
        forbidden = frozenset(dimensions.values())
        launch_k_tiles = {group: 7 for group in groups}

        with self.assertRaisesRegex(ValueError, "found=0 required=12"):
            fresh_geometry_candidates_for_rule(
                rule,
                dimensions,
                forbidden,
                self.manifest.maximum_cpu_measurement_weight_elements,
                12,
                launch_k_tiles_by_group=launch_k_tiles,
            )

        rows = tuple(
            SimpleNamespace(
                shape_group_id=group,
                aggregate_n=n,
                k=k,
                architecture_class=ARCHITECTURE,
                runtime_codebook_id=4,
                bundle_signature=SERIAL_KPART_BUNDLE,
                operation_kind="NativeVNNIFastM1Projection",
                execution_mode=ExecutionMode.EAGER,
                m=1,
                launch_k_tiles=7,
            )
            for group, (n, k) in dimensions.items()
        )
        probe = build_cpu_decode_sealed_route_probe(
            (rule,),
            FROZEN_DIGEST,
            _Development(rows, (rule.domain,)),
            self.manifest,
            self.sealed_build_id,
        )

        self.assertGreaterEqual(len(probe.shapes), 12)
        self.assertTrue(all(
            rule.matches(shape.n, shape.k, 7) for shape in probe.shapes
        ))
        self.assertTrue(all(shape.k % 32 == 0 for shape in probe.shapes))
        self.assertTrue(any(shape.n % 32 != 0 for shape in probe.shapes))

    def test_sealed_build_change_burns_probe_and_plan(self) -> None:
        """A harness rebuild cannot reuse route or paired timing evidence."""

        changed = "sha256:" + "c" * 64
        with tempfile.TemporaryDirectory() as temporary:
            routes = Path(temporary) / "routes.csv"
            self._write_routes(routes, self.probe)
            with self.assertRaisesRegex(ValueError, "another sealed build"):
                build_cpu_decode_sealed_plan(
                    self.rules,
                    FROZEN_DIGEST,
                    self.development,
                    self.manifest,
                    self.probe,
                    (routes,),
                    changed,
                )


if __name__ == "__main__":
    unittest.main()
