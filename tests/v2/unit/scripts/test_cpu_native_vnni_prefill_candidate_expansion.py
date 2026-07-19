#!/usr/bin/env python3
"""Regression tests for additive CPU prefill candidate collection."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    cpu_native_vnni_prefill_registry,
)
from native_vnni_dispatch.cpu_prefill_candidate_expansion import (  # noqa: E402
    CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
    DEFAULT_ANCHOR_CANDIDATES,
    CPUPrefillCandidateExpansionPlan,
    build_cpu_prefill_candidate_expansion_plan,
    normalize_cpu_prefill_candidate_expansion,
    read_cpu_prefill_candidate_expansion_plan,
    validate_normalized_cpu_prefill_candidate_expansion,
    write_cpu_prefill_candidate_expansion_plan,
)
from native_vnni_dispatch.corpus import ObservationCorpus  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    Backend,
    ExecutionMode,
)
from tests.v2.unit.scripts.test_native_vnni_common_dispatch_policy import (  # noqa: E402
    observation,
)


SOURCE_CANDIDATES = (
    "cpu.nvnni.prefill.row_chunk_grid.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc1.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc2.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc4.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc8.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc16.full_k",
    "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
    "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
)
FULL_K_ANCHOR = DEFAULT_ANCHOR_CANDIDATES[0]
UNIT_BUILD_DIGEST = "sha256:" + "b" * 64


class CPUNativeVNNIPrefillCandidateExpansionTest(unittest.TestCase):
    """Prove candidate expansion is complete, additive, and resumable."""

    @staticmethod
    def _write_source(path: Path, *, omit_last_candidate: bool = False) -> None:
        fieldnames = (
            "backend",
            "phase",
            "source_format",
            "source_codebook",
            "execution_codebook",
            "shape",
            "execution_mode",
            "m",
            "n",
            "k",
            "candidate_id",
            "build_isa",
            "runtime_isa_requested",
            "runtime_isa_effective",
            "threads",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for build_isa, runtime_isa in (
                ("AVX2", "AVX2"),
                ("AVX512", "AVX2"),
                ("AVX512", "AVX512"),
            ):
                for m in (2, 15):
                    candidates = SOURCE_CANDIDATES
                    if omit_last_candidate and m == 15 and build_isa == "AVX2":
                        candidates = candidates[:-1]
                    for candidate in candidates:
                        writer.writerow({
                            "backend": "cpu",
                            "phase": "prefill_gemm",
                            "source_format": "Q4_0",
                            "source_codebook": 2,
                            "execution_codebook": 2,
                            "shape": "unit-shape",
                            "execution_mode": "eager",
                            "m": m,
                            "n": 1024,
                            "k": 2048,
                            "candidate_id": candidate,
                            "build_isa": build_isa,
                            "runtime_isa_requested": runtime_isa,
                            "runtime_isa_effective": runtime_isa,
                            "threads": 28,
                        })

    def test_plan_adds_only_five_pair_grid_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            output = root / "expansion.json"
            self._write_source(source)
            timing.write_text("timing\nretained\n", encoding="utf-8")

            plan = build_cpu_prefill_candidate_expansion_plan(
                source,
                timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )

            self.assertEqual(plan.source_cell_count, 6)
            self.assertEqual(
                plan.anchor_candidate_ids,
                DEFAULT_ANCHOR_CANDIDATES,
            )
            self.assertEqual(len(plan.records), 3)
            self.assertEqual({record.m_values for record in plan.records}, {(2, 15)})
            self.assertEqual({record.threads for record in plan.records}, {28})
            self.assertEqual(len(plan.expansion_candidate_ids), 5)
            self.assertTrue(all(
                ".two_row_pair_grid." in candidate
                for candidate in plan.expansion_candidate_ids
            ))
            self.assertEqual(
                set(plan.source_candidate_ids).union(plan.expansion_candidate_ids),
                {entry.candidate_id for entry in cpu_native_vnni_prefill_registry().entries},
            )
            write_cpu_prefill_candidate_expansion_plan(output, plan)
            restored = read_cpu_prefill_candidate_expansion_plan(
                output,
                source_aggregate=source,
                source_timing=timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )
            self.assertEqual(restored.canonical_mapping(), plan.canonical_mapping())
            self.assertEqual(restored.digest(), plan.digest())

    def test_unrelated_global_registry_version_preserves_prefill_plan(self) -> None:
        """A new decode/backend family must not invalidate prefill evidence."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            output = root / "expansion.json"
            self._write_source(source)
            timing.write_text("timing\nretained\n", encoding="utf-8")
            current = build_cpu_prefill_candidate_expansion_plan(
                source,
                timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )
            registry = cpu_native_vnni_prefill_registry()
            historical = replace(
                current,
                candidate_registry_digest=registry.digest(
                    registry_version="native-vnni-candidates-v8"
                ),
            )
            write_cpu_prefill_candidate_expansion_plan(output, historical)

            restored = read_cpu_prefill_candidate_expansion_plan(
                output,
                source_aggregate=source,
                source_timing=timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
                authenticate_current_implementation=False,
            )

            self.assertEqual(restored, historical)
            self.assertTrue(
                registry.matches_digest(historical.candidate_registry_digest)
            )
            self.assertFalse(registry.matches_digest("sha256:" + "0" * 64))

    def test_development_selection_retains_reviewed_fit_cells_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            self._write_source(source)
            with source.open(newline="", encoding="utf-8") as handle:
                synthetic_rows = list(csv.DictReader(handle))
                fieldnames = tuple(synthetic_rows[0])
            production_rows = [
                {**row, "shape": "35BMoE_Expert_Down"}
                for row in synthetic_rows
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((*synthetic_rows, *production_rows))
            timing.write_text("timing\nretained\n", encoding="utf-8")

            plan = build_cpu_prefill_candidate_expansion_plan(
                source,
                timing,
                development_only=True,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )

            self.assertEqual(plan.source_cell_count, 12)
            self.assertEqual(plan.selected_cell_count, 6)
            self.assertEqual(
                plan.selection_policy,
                "split-development-cells-v1",
            )
            self.assertEqual(
                {record.shape_name for record in plan.records},
                {"35BMoE_Expert_Down"},
            )

    def test_plan_rejects_an_incomplete_source_candidate_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            self._write_source(source, omit_last_candidate=True)
            timing.write_text("timing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "one complete candidate inventory"):
                build_cpu_prefill_candidate_expansion_plan(
                    source,
                    timing,
                    collection_build_digest=UNIT_BUILD_DIGEST,
                )

    def test_plan_rejects_source_or_plan_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            output = root / "expansion.json"
            self._write_source(source)
            timing.write_text("timing\nretained\n", encoding="utf-8")
            plan = build_cpu_prefill_candidate_expansion_plan(
                source,
                timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )
            write_cpu_prefill_candidate_expansion_plan(output, plan)

            timing.write_text("timing\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source timing changed"):
                read_cpu_prefill_candidate_expansion_plan(
                    output,
                    source_aggregate=source,
                    source_timing=timing,
                )

            with self.assertRaisesRegex(ValueError, "trainer binaries changed"):
                read_cpu_prefill_candidate_expansion_plan(
                    output,
                    collection_build_digest="sha256:" + "c" * 64,
                )

            raw = json.loads(output.read_text(encoding="utf-8"))
            raw["source_cell_count"] += 1
            output.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plan digest changed"):
                read_cpu_prefill_candidate_expansion_plan(output)

    def test_historical_plan_can_skip_only_current_implementation_check(self) -> None:
        """Checkpoint migration may inspect old policy identity, never ignore it."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            output = root / "historical.json"
            self._write_source(source)
            timing.write_text("timing\nretained\n", encoding="utf-8")
            current = build_cpu_prefill_candidate_expansion_plan(
                source,
                timing,
                collection_build_digest=UNIT_BUILD_DIGEST,
            )
            historical = replace(
                current,
                implementation_digest="sha256:" + "d" * 64,
            )
            write_cpu_prefill_candidate_expansion_plan(output, historical)

            with self.assertRaisesRegex(ValueError, "implementation changed"):
                read_cpu_prefill_candidate_expansion_plan(output)
            restored = read_cpu_prefill_candidate_expansion_plan(
                output,
                authenticate_current_implementation=False,
            )

            self.assertEqual(restored, historical)

    def test_existing_candidate_cannot_be_recollected_as_an_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            timing = root / "source.timing.csv"
            self._write_source(source)
            timing.write_text("timing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "may not recollect"):
                build_cpu_prefill_candidate_expansion_plan(
                    source,
                    timing,
                    expansion_candidate_ids=(FULL_K_ANCHOR,),
                    collection_build_digest=UNIT_BUILD_DIGEST,
                )

    @staticmethod
    def _normalization_plan(target: str) -> CPUPrefillCandidateExpansionPlan:
        return CPUPrefillCandidateExpansionPlan(
            schema_version=CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
            source_aggregate_sha256="sha256:source-aggregate",
            source_timing_sha256="sha256:source-timing",
            candidate_registry_digest=cpu_native_vnni_prefill_registry().digest(),
            implementation_digest="sha256:unit-implementation",
            collection_build_digest=UNIT_BUILD_DIGEST,
            source_cell_count=1,
            source_cell_digest="sha256:source-cells",
            selection_policy="unit-all-cells",
            selected_cell_count=1,
            selected_cell_digest="sha256:selected-cells",
            source_candidate_ids=DEFAULT_ANCHOR_CANDIDATES,
            anchor_candidate_ids=DEFAULT_ANCHOR_CANDIDATES,
            expansion_candidate_ids=(target,),
            records=(),
        )

    @staticmethod
    def _cpu_observation(candidate: str, latency: float):
        return observation(
            candidate=candidate,
            family="cpu-prefill-unit",
            shape_group="cpu-prefill:unit-shape:n1024:k2048",
            shape_name="unit-shape",
            n=1024,
            k=2048,
            m=15,
            latency_us=latency,
            mode=ExecutionMode.EAGER,
            backend=Backend.CPU,
        )

    def test_new_candidate_is_scaled_to_the_immutable_anchor_clock(self) -> None:
        target = "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k"
        source = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 10.0),
        ))
        expansion = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 20.0),
            self._cpu_observation(target, 15.0),
        ))

        combined = normalize_cpu_prefill_candidate_expansion(
            source,
            expansion,
            self._normalization_plan(target),
        )

        self.assertEqual(len(combined), 2)
        normalized = next(row for row in combined if row.candidate_id == target)
        self.assertEqual(normalized.median_us, 7.5)
        self.assertEqual(normalized.min_us, 7.5)
        self.assertEqual(normalized.p95_us, 7.5)
        proof = normalized.adaptive_timing_evidence[
            "candidate_expansion_normalization"
        ]
        self.assertEqual(proof["status"], "scaled_to_immutable_source_anchor")
        self.assertEqual(float.fromhex(proof["scale_hex"]), 0.5)

    def test_plan_digest_and_anchor_scale_are_reused_across_cell_candidates(
        self,
    ) -> None:
        """Keep normalization linear in cells rather than plan bytes times rows."""

        targets = (
            "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k",
            "cpu.nvnni.prefill.two_row_pair_grid.nbc2.full_k",
        )
        source = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 10.0),
        ))
        expansion = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 20.0),
            self._cpu_observation(targets[0], 15.0),
            self._cpu_observation(targets[1], 12.0),
        ))
        delegate = replace(
            self._normalization_plan(targets[0]),
            expansion_candidate_ids=targets,
        )

        class CountingPlan:
            """Delegate plan fields while exposing expensive digest call count."""

            def __init__(self, wrapped: CPUPrefillCandidateExpansionPlan):
                self.wrapped = wrapped
                self.digest_calls = 0

            def __getattr__(self, name: str):
                return getattr(self.wrapped, name)

            def digest(self) -> str:
                self.digest_calls += 1
                return self.wrapped.digest()

        plan = CountingPlan(delegate)
        combined = normalize_cpu_prefill_candidate_expansion(
            source,
            expansion,
            plan,
        )

        self.assertEqual(plan.digest_calls, 1)
        latencies = {
            row.candidate_id: row.median_us
            for row in combined
            if row.candidate_id in targets
        }
        self.assertEqual(latencies, {targets[0]: 7.5, targets[1]: 6.0})

    def test_normalized_checkpoint_requires_complete_plan_provenance(self) -> None:
        target = "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k"
        source = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 10.0),
        ))
        expansion = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 20.0),
            self._cpu_observation(target, 15.0),
        ))
        plan = replace(
            self._normalization_plan(target),
            source_candidate_ids=(FULL_K_ANCHOR,),
            anchor_candidate_ids=(FULL_K_ANCHOR,),
        )
        combined = normalize_cpu_prefill_candidate_expansion(
            source,
            expansion,
            plan,
        )

        validate_normalized_cpu_prefill_candidate_expansion(combined, plan)

        normalized = next(row for row in combined if row.candidate_id == target)
        forged_proof = {
            **normalized.adaptive_timing_evidence,
            "candidate_expansion_normalization": {
                **normalized.adaptive_timing_evidence[
                    "candidate_expansion_normalization"
                ],
                "plan_digest": "sha256:" + "0" * 64,
            },
        }
        forged = replace(
            normalized,
            adaptive_timing_evidence=forged_proof,
        )
        with self.assertRaisesRegex(ValueError, "another plan"):
            validate_normalized_cpu_prefill_candidate_expansion(
                ObservationCorpus((source.observations[0], forged)),
                plan,
            )

    def test_supported_candidate_without_anchor_fails_closed(self) -> None:
        target = "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k"
        source = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 10.0),
        ))
        expansion = ObservationCorpus((
            self._cpu_observation(target, 15.0),
        ))

        with self.assertRaisesRegex(ValueError, "exactly one route-matched"):
            normalize_cpu_prefill_candidate_expansion(
                source,
                expansion,
                self._normalization_plan(target),
            )

    def test_unsupported_candidate_is_retained_without_fabricated_timing(self) -> None:
        target = "cpu.nvnni.prefill.two_row_pair_grid.nbc1.full_k"
        source = ObservationCorpus((
            self._cpu_observation(FULL_K_ANCHOR, 10.0),
        ))
        unsupported = replace(
            self._cpu_observation(target, 15.0),
            supported=False,
            forced_route_ok=False,
            route_counter_ok=False,
            observed_candidate_id=FULL_K_ANCHOR,
        )
        unsupported.validate()
        expansion = ObservationCorpus((unsupported,))

        combined = normalize_cpu_prefill_candidate_expansion(
            source,
            expansion,
            self._normalization_plan(target),
        )

        retained = next(row for row in combined if row.candidate_id == target)
        self.assertEqual(retained.median_us, 15.0)
        proof = retained.adaptive_timing_evidence[
            "candidate_expansion_normalization"
        ]
        self.assertEqual(proof["status"], "candidate_not_supported_on_serial_route")


if __name__ == "__main__":
    unittest.main()
