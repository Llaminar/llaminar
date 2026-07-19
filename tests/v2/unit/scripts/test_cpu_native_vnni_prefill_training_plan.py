#!/usr/bin/env python3
"""Regression tests for the two-hour CPU prefill covering plan."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import groupby
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.cpu_prefill_training_plan import (  # noqa: E402
    CPUPrefillSourceTrainingRecord,
    ISA_REGIMES,
    _generic_domain_shape_group_counts,
    _required_minimum_geometry_anchor_cells,
    _source_record_schedule_key,
    build_cpu_prefill_sealed_witness_plan,
    coalesce_cpu_prefill_source_training_records,
    cpu_prefill_arithmetic_refinement_source_training_records,
    cpu_prefill_development_refinement_source_training_records,
    cpu_prefill_missing_base_source_training_records,
    cpu_prefill_opened_witness_v6_source_training_records,
    cpu_prefill_opened_witness_v7_source_training_records,
    cpu_prefill_runtime_training_cells,
    cpu_prefill_source_training_records,
    read_cpu_prefill_sealed_witness_plan,
    validate_cpu_prefill_runtime_training_cells,
    validate_cpu_prefill_sealed_witness_plan,
    validate_cpu_prefill_sealed_witness_plan_file,
    write_cpu_prefill_sealed_witness_plan,
)
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
    CPUPrefillSerialRoute,
    CPUPrefillSerialRouteManifest,
)
from native_vnni_dispatch.format_registry import (  # noqa: E402
    FORMAT_SPECS,
    format_spec,
    runtime_aliases,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    cpu_prefill_measurements,
)
from native_vnni_dispatch.cpu_prefill_split_manifest import (  # noqa: E402
    load_cpu_prefill_split_manifest,
)
from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
)
from native_vnni_dispatch.shape_manifest import load_shape_manifest  # noqa: E402


class CPUNativeVNNIPrefillTrainingPlanTest(unittest.TestCase):
    """Lock coverage strength and bounded collection cardinality."""

    def test_format_fixture_coalescing_is_complete_and_phase_safe(self) -> None:
        """Coalescing must remove only compatible process boundaries."""

        records = (
            CPUPrefillSourceTrainingRecord(
                source_format="Q4_0",
                shape_name="shape-a",
                n=512,
                k=1024,
                isa_regime="avx512-build.avx512-runtime",
                runtime_isa="avx512",
                m_values=(64, 256),
            ),
            CPUPrefillSourceTrainingRecord(
                source_format="Q4_1",
                shape_name="shape-a",
                n=512,
                k=1024,
                isa_regime="avx512-build.avx512-runtime",
                runtime_isa="avx512",
                m_values=(64, 256),
            ),
            CPUPrefillSourceTrainingRecord(
                source_format="Q5_0",
                shape_name="shape-b",
                n=256,
                k=1024,
                isa_regime="avx512-build.avx512-runtime",
                runtime_isa="avx512",
                m_values=(64, 256),
            ),
            CPUPrefillSourceTrainingRecord(
                source_format="Q8_0",
                shape_name="shape-c",
                n=256,
                k=1024,
                isa_regime="avx512-build.avx512-runtime",
                runtime_isa="avx512",
                m_values=(1024,),
            ),
        )

        jobs = coalesce_cpu_prefill_source_training_records(records)
        self.assertEqual(
            jobs,
            coalesce_cpu_prefill_source_training_records(tuple(reversed(records))),
        )
        self.assertEqual(len(jobs), 3)
        self.assertEqual(
            sorted(
                source_format
                for job in jobs
                for source_format in job.source_formats
            ),
            sorted(record.source_format for record in records),
        )
        combined = next(job for job in jobs if job.shape_name == "shape-a")
        self.assertEqual(combined.source_formats, ("Q4_0", "Q4_1"))
        self.assertEqual(combined.format_phase_count, 2)
        self.assertRegex(combined.format_token, r"^formats-[0-9a-f]{16}$")
        singleton = next(job for job in jobs if job.shape_name == "shape-b")
        self.assertEqual(singleton.format_token, "Q5_0")

        phase_keys = [
            (job.m_values, job.format_phase_count) for job in jobs
        ]
        self.assertEqual(
            len(list(groupby(phase_keys))),
            len(set(phase_keys)),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _production_like_route_manifest() -> CPUPrefillSerialRouteManifest:
        """Model the two finite K-part geometries observed by production C++."""

        codebooks = sorted({
            spec.cpu_execution_codebook_id for spec in FORMAT_SPECS
        })
        kpart_shapes = {
            "1.5B_FFN_Dn",
            "3B_FFN_Dn",
            "V4FastSealed_Tall_1536x5632",
        }
        split = load_cpu_prefill_split_manifest()
        manifest = load_shape_manifest()
        shapes = tuple(
            manifest.by_name(name)
            for name in dict.fromkeys((
                *split.development_shapes,
                *split.sealed_shapes,
            ))
        )
        return CPUPrefillSerialRouteManifest(
            CPUPrefillSerialRoute(
                execution_codebook=codebook,
                shape_name=shape.name,
                n=shape.n,
                k=shape.k,
                isa_regime=regime,
                payload_bytes=16 + codebook,
                threads=28,
                k_tiles=4 if shape.name in kpart_shapes else 0,
                bundle_signature=(
                    CPU_PREFILL_KPART_BUNDLE
                    if shape.name in kpart_shapes
                    else CPU_PREFILL_FULL_K_BUNDLE
                ),
            )
            for codebook in codebooks
            for shape in shapes
            for regime in ISA_REGIMES
        )

    def test_runtime_covering_array_satisfies_declared_obligations(self) -> None:
        cells = cpu_prefill_runtime_training_cells()

        validate_cpu_prefill_runtime_training_cells(cells)
        self.assertEqual(len(cells), 2818)
        self.assertEqual(len(cells), len(set(cells)))

    def test_every_generic_domain_has_enough_lower_geometry_anchors(self) -> None:
        """Every policy domain must have a lower edge and three shape groups."""

        cells = set(cpu_prefill_runtime_training_cells())
        codebooks = tuple(sorted({cell.runtime_codebook for cell in cells}))
        required = _required_minimum_geometry_anchor_cells(
            cpu_prefill_measurements(),
            codebooks,
        )
        counts = _generic_domain_shape_group_counts(
            cells,
            cpu_prefill_measurements(),
        )

        self.assertTrue(required)
        self.assertTrue(required.issubset(cells))
        self.assertTrue(counts)
        self.assertGreaterEqual(min(counts.values()), 3)

    def test_sparse_kpart_domains_receive_generic_route_anchors(self) -> None:
        """A two-production-shape family gains a third measured CV group."""

        manifest = self._production_like_route_manifest()
        cells = cpu_prefill_runtime_training_cells(manifest)
        validate_cpu_prefill_runtime_training_cells(cells, manifest)

        kpart_shapes = {"1.5B_FFN_Dn", "3B_FFN_Dn"}
        selected = {
            (cell.runtime_codebook, cell.m, cell.isa_regime, cell.shape_name)
            for cell in cells
            if cell.shape_name in kpart_shapes
        }
        for codebook in sorted({cell.runtime_codebook for cell in cells}):
            for regime in ISA_REGIMES:
                for measurement in cpu_prefill_measurements():
                    if measurement.shape.name not in kpart_shapes:
                        continue
                    for m in measurement.m_values:
                        self.assertIn(
                            (codebook, m, regime, measurement.shape.name),
                            selected,
                        )

        anchors = cpu_prefill_arithmetic_refinement_source_training_records(
            manifest
        )
        self.assertEqual(len(anchors), 18 * len(ISA_REGIMES))
        self.assertEqual(
            {record.shape_name for record in anchors},
            {"V4FastSealed_Tall_1536x5632"},
        )
        self.assertEqual(
            {record.m_values for record in anchors},
            {load_cpu_prefill_split_manifest().sealed_m_values},
        )

    def test_source_aliases_are_co_measured(self) -> None:
        records = cpu_prefill_source_training_records()
        cells_by_format: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
        for record in records:
            for m in record.m_values:
                cells_by_format[record.source_format].add(
                    (record.shape_name, m, record.isa_regime)
                )
        for codebook in sorted({item.cpu_execution_codebook_id for item in FORMAT_SPECS}):
            aliases = runtime_aliases("cpu", codebook)
            exemplar = cells_by_format[aliases[0]]
            for alias in aliases[1:]:
                self.assertEqual(cells_by_format[alias], exemplar)

    def test_historical_base_diff_emits_only_missing_m_cells(self) -> None:
        """A route-plan extension must not recollect an observed base corpus."""

        manifest = self._production_like_route_manifest()
        records = cpu_prefill_source_training_records(manifest)
        target = records[0]
        missing_m = target.m_values[len(target.m_values) // 2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical-base.csv"
            fieldnames = (
                "source_format",
                "shape",
                "m",
                "build_isa",
                "runtime_isa_effective",
            )
            raw_isa = {
                "avx2-build.avx2-runtime": ("AVX2", "AVX2"),
                "avx512-build.avx2-runtime": ("AVX512", "AVX2"),
                "avx512-build.avx512-runtime": ("AVX512", "AVX512"),
            }
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    build_isa, runtime_isa = raw_isa[record.isa_regime]
                    for m in record.m_values:
                        if record == target and m == missing_m:
                            continue
                        writer.writerow({
                            "source_format": record.source_format,
                            "shape": record.shape_name,
                            "m": m,
                            "build_isa": build_isa,
                            "runtime_isa_effective": runtime_isa,
                        })

            missing = cpu_prefill_missing_base_source_training_records(
                manifest,
                (path,),
            )

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].source_format, target.source_format)
        self.assertEqual(missing[0].shape_name, target.shape_name)
        self.assertEqual(missing[0].isa_regime, target.isa_regime)
        self.assertEqual(missing[0].m_values, (missing_m,))

    def test_sealed_plan_projects_rules_and_expands_source_aliases(self) -> None:
        """One cheapest witness must preserve every alias and grouped M value."""

        codebook = 4
        split = load_cpu_prefill_split_manifest()
        rules = tuple(
            GenericDispatchRule(
                domain=GenericDomain(
                    backend=Backend.CPU,
                    architecture_class=(
                        "test-host|build=AVX512|runtime=AVX512|threads=28"
                    ),
                    semantic_contract=(
                        SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                    ),
                    operation_kind="NativeVNNIPrefillProjection",
                    bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
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
                    "cpu.nvnni.prefill.two_row_tiles.nbc1.full_k"
                ),
                arithmetic_fingerprint="sha256:test",
                development_shape_groups=("development:test",),
                development_max_regret=0.01,
                development_p95_regret=0.01,
                development_mean_regret=0.01,
            )
            for m in (64, 256)
        )
        plan = build_cpu_prefill_sealed_witness_plan(
            rules,
            "sha256:frozen",
            split,
            self._production_like_route_manifest(),
        )

        self.assertEqual(len(plan.rule_witnesses), 2)
        self.assertEqual(
            {witness.rule_index for witness in plan.rule_witnesses},
            {0, 1},
        )
        self.assertEqual(len(plan.records), len(runtime_aliases("cpu", codebook)))
        self.assertEqual(
            {record.m_values for record in plan.records},
            {(64, 256)},
        )
        self.assertEqual(
            {record.source_format for record in plan.records},
            set(runtime_aliases("cpu", codebook)),
        )
        work_by_shape = {
            shape_name: (
                load_shape_manifest().by_name(shape_name).n
                * load_shape_manifest().by_name(shape_name).k
            )
            for shape_name in split.sealed_shapes
        }
        self.assertEqual(
            {record.shape_name for record in plan.records},
            {min(work_by_shape, key=work_by_shape.get)},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness-plan.json"
            write_cpu_prefill_sealed_witness_plan(path, plan)
            self.assertEqual(read_cpu_prefill_sealed_witness_plan(path), plan)
            validate_cpu_prefill_sealed_witness_plan_file(path, plan)
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "sha256:frozen", "sha256:tampered"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "changed after generic freeze"):
                validate_cpu_prefill_sealed_witness_plan_file(path, plan)

    def test_sealed_plan_embeds_fresh_generated_route_per_frozen_leaf(self) -> None:
        """A generated witness must be self-contained and absent from fitting."""

        codebook = 4
        split = load_cpu_prefill_split_manifest()
        production_routes = self._production_like_route_manifest()
        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class=(
                "test-host|build=AVX512|runtime=AVX512|threads=28"
            ),
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
            prepared_family_id=f"NativeVNNI_cpu_CB{codebook}",
            packing_abi=f"native-vnni-cpu-cb{codebook}-v1",
            runtime_codebook_id=codebook,
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        rule = GenericDispatchRule(
            domain=domain,
            predicates=(),
            candidate_id="cpu.nvnni.prefill.two_row_tiles.nbc1.full_k",
            arithmetic_fingerprint="sha256:test",
            development_shape_groups=("development:test",),
            development_max_regret=0.01,
            development_p95_regret=0.01,
            development_mean_regret=0.01,
        )
        candidate = CPUPrefillSerialRoute(
            execution_codebook=codebook,
            shape_name="CPUPrefillAutoRefine_N16_K32",
            n=16,
            k=32,
            isa_regime="avx512-build.avx512-runtime",
            payload_bytes=16 + codebook,
            threads=28,
            k_tiles=0,
            bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
        )
        candidates = CPUPrefillSerialRouteManifest((candidate,))
        # The generated point is cheaper than every checked-in sealed shape,
        # proving that it participates in deterministic witness selection.
        plan = build_cpu_prefill_sealed_witness_plan(
            (rule,),
            "sha256:frozen",
            split,
            production_routes,
            candidate_route_manifest=candidates,
            development_dimensions=frozenset({(896, 896)}),
        )

        self.assertEqual(plan.rule_witnesses[0].route, candidate)
        self.assertEqual(
            plan.candidate_route_manifest_digest,
            candidates.digest(),
        )
        validate_cpu_prefill_sealed_witness_plan(
            plan,
            (rule,),
            split,
            production_routes,
            development_dimensions=frozenset({(896, 896)}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated-witness-plan.json"
            write_cpu_prefill_sealed_witness_plan(path, plan)
            self.assertEqual(read_cpu_prefill_sealed_witness_plan(path), plan)

        with self.assertRaisesRegex(ValueError, "visible to fitting"):
            validate_cpu_prefill_sealed_witness_plan(
                plan,
                (rule,),
                split,
                production_routes,
                development_dimensions=frozenset({(16, 32)}),
            )

    def test_development_refinement_reuses_opened_rows_and_densifies_cliffs(
        self,
    ) -> None:
        """The v4 increment must not rebuild the original development sweep."""

        records = cpu_prefill_development_refinement_source_training_records(
            self._production_like_route_manifest()
        )
        opened = [
            record
            for record in records
            if record.shape_name.startswith("V5CPUPrefillSealed_")
        ]
        focused = [
            record
            for record in records
            if record.shape_name.startswith("CPUPrefillRefine_")
        ]
        opened_v5 = [
            record
            for record in records
            if record.shape_name.startswith("V7CPUPrefillSealed_")
        ]
        opened_v6 = [
            record
            for record in records
            if record.shape_name.startswith("V8CPUPrefillSealed_")
        ]
        opened_v7 = [
            record
            for record in records
            if record.shape_name.startswith("V9CPUPrefillSealed_")
        ]
        arithmetic_anchors = [
            record
            for record in records
            if record.shape_name == "V4FastSealed_Tall_1536x5632"
        ]

        self.assertEqual(len(opened), 8 * len(FORMAT_SPECS) * len(ISA_REGIMES))
        self.assertEqual({record.m_values for record in opened}, {(64,)})
        self.assertEqual(
            {record.source_format for record in opened},
            {spec.label for spec in FORMAT_SPECS},
        )
        self.assertEqual(len(focused), 4 * 18 * len(ISA_REGIMES))
        self.assertEqual(
            {record.m_values for record in focused},
            {load_cpu_prefill_split_manifest().sealed_m_values},
        )
        canonical_aliases = {
            runtime_aliases("cpu", codebook)[0]
            for codebook in {
                spec.cpu_execution_codebook_id for spec in FORMAT_SPECS
            }
        }
        self.assertEqual(
            {record.source_format for record in focused},
            canonical_aliases,
        )
        self.assertEqual(len(opened_v5), 37)
        self.assertEqual(sum(len(record.m_values) for record in opened_v5), 86)
        self.assertEqual(
            {record.shape_name for record in opened_v5},
            {
                "V7CPUPrefillSealed_Tall_240x416",
                "V7CPUPrefillSealed_Wide_2208x544",
                "V7CPUPrefillSealed_VeryWide_5664x352",
                "V7CPUPrefillSealed_VeryWide_17280x544",
            },
        )
        self.assertEqual(
            Counter(opened_v6),
            Counter(cpu_prefill_opened_witness_v6_source_training_records(
                self._production_like_route_manifest()
            )),
        )
        self.assertEqual(len(opened_v6), 21)
        self.assertEqual(sum(len(record.m_values) for record in opened_v6), 34)
        self.assertEqual(
            {record.shape_name for record in opened_v6},
            {
                "V8CPUPrefillSealed_Tall_224x416",
                "V8CPUPrefillSealed_Wide_2272x544",
                "V8CPUPrefillSealed_VeryWide_5792x352",
            },
        )
        self.assertEqual(
            Counter(opened_v7),
            Counter(cpu_prefill_opened_witness_v7_source_training_records(
                self._production_like_route_manifest()
            )),
        )
        self.assertEqual(len(opened_v7), 17)
        self.assertEqual(sum(len(record.m_values) for record in opened_v7), 30)
        self.assertEqual(
            {record.shape_name for record in opened_v7},
            {
                "V9CPUPrefillSealed_Tall_208x416",
                "V9CPUPrefillSealed_Wide_2336x544",
                "V9CPUPrefillSealed_VeryWide_5920x352",
            },
        )
        for shape_name in {record.shape_name for record in focused}:
            for regime in ISA_REGIMES:
                codebooks = {
                    format_spec(record.source_format).cpu_execution_codebook_id
                    for record in focused
                    if record.shape_name == shape_name
                    and record.isa_regime == regime
                }
                self.assertEqual(len(codebooks), 18)
        self.assertEqual(len(arithmetic_anchors), 18 * len(ISA_REGIMES))
        self.assertEqual(
            {record.m_values for record in arithmetic_anchors},
            {load_cpu_prefill_split_manifest().sealed_m_values},
        )
        self.assertEqual(len(records), 849)
        self.assertEqual(sum(len(record.m_values) for record in records), 2544)

    def test_plan_is_balanced_and_far_smaller_than_cartesian_product(self) -> None:
        runtime = cpu_prefill_runtime_training_cells()
        records = cpu_prefill_source_training_records()
        source_cells = sum(len(record.m_values) for record in records)
        regime_loads = Counter(cell.isa_regime for cell in runtime)

        full_cartesian_cells = (
            len(FORMAT_SPECS)
            * sum(
                len(measurement.m_values)
                for measurement in cpu_prefill_measurements()
            )
            * len(ISA_REGIMES)
        )

        self.assertEqual(source_cells, 3229)
        self.assertLessEqual(max(regime_loads.values()) - min(regime_loads.values()), 10)
        self.assertEqual(set(regime_loads), set(ISA_REGIMES))
        self.assertLess(source_cells, full_cartesian_cells // 4)

    def test_process_jobs_form_identical_m_inventory_mpmd_groups(self) -> None:
        records = cpu_prefill_source_training_records()
        groups = [
            tuple(group)
            for _, group in groupby(records, key=lambda record: record.m_values)
        ]
        estimated_group_work = []
        for group in groups:
            self.assertTrue(group)
            self.assertEqual(len({record.m_values for record in group}), 1)
            estimated_work = [
                -_source_record_schedule_key(record)[0]
                for record in group
            ]
            self.assertEqual(estimated_work, sorted(estimated_work, reverse=True))
            estimated_group_work.append(estimated_work[0])

        self.assertEqual(len(groups), 54)
        self.assertEqual(
            sum((len(group) + 1) // 2 for group in groups),
            953,
        )
        self.assertEqual(
            estimated_group_work,
            sorted(estimated_group_work, reverse=True),
        )


if __name__ == "__main__":
    unittest.main()
