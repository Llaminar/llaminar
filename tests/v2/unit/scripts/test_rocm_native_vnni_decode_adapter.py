#!/usr/bin/env python3
"""Focused regressions for strong ROCm NativeVNNI decode adaptation."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v2.performance.kernels.native_vnni_dispatch.adapters.evidence import (
    summarize_sorted_timing,
)
from tests.v2.performance.kernels.native_vnni_dispatch.adapters.rocm_decode import (
    ROCmDecodeAdapterContext,
    adapt_rocm_decode_csv,
    adapt_rocm_decode_row,
    raw_corpus_id,
    read_rocm_decode_timing_sidecars,
    ROCM_Q8_GEOMETRY_GENERATION,
)
from tests.v2.performance.kernels.native_vnni_dispatch.geometry_timing import (
    PAIRED_GEOMETRY_GENERATION, GEOMETRY_ORDER_PROTOCOL, geometry_timing_orders,
    geometry_timing_position, paired_geometry_summary,
)
from tests.v2.performance.kernels.native_vnni_dispatch.exact_oracle import (
    candidate_is_eligible,
)
from tests.v2.performance.kernels.native_vnni_dispatch.profiles import MeasurementProfile
from tests.v2.performance.kernels.native_vnni_dispatch.schema import (
    SemanticContract, NativeVNNIObservation,
)

from tests.v2.performance.kernels.native_vnni_dispatch.candidate_registry import (
    rocm_native_vnni_decode_registry,
    rocm_native_q8_serial_geometry_diagnostic_registry,
)
from tests.v2.performance.kernels.native_vnni_dispatch.candidate_observation import (
    write_observation_csv, read_observation_csv,
)


class ROCmNativeVNNIDecodeAdapterTest(unittest.TestCase):
    """Prove contract normalization, exactness, and sidecar verification."""

    @staticmethod
    def context() -> ROCmDecodeAdapterContext:
        return ROCmDecodeAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "1" * 64
        )

    @staticmethod
    def row(*, m: int = 2, candidate: str = "INHERIT_SERIAL_M1") -> dict[str, str]:
        """Build one valid strong-evidence Q8 grouped-verifier row."""

        return {
            "backend": "rocm",
            "phase": "decode",
            "source_format": "Q8_0",
            "source_codebook": "19",
            "execution_codebook": "19",
            "shape": "35BMoE_Expert_GateUp",
            "execution_mode": "eager",
            "m": str(m),
            "n": "512",
            "k": "2048",
            "candidate_id": candidate,
            "kb": "32",
            "target_waves": "4",
            "weight_bytes": "1114112",
            "warmup_count": "2",
            "sample_count": "3",
            "min_us": "10",
            "median_us": "11",
            "p95_us": "13",
            "mad_us": "1",
            "cv": "0.1",
            "effective_bandwidth_gbs": "30",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal00000000000",
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": "fnv1a64:timing0000000000",
            "route_counter_ok": "1",
            "observed_candidate_id": candidate,
            "observed_path": "split_reduce",
            "serial_m1_kb": "32",
            "serial_m1_target_waves": "4",
            "serial_route_counter_ok": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1",
            "is_winner": "1",
        }

    def geometry_row(self, tn: int = 64, cpt: int = 1) -> tuple[dict[str, str], tuple[float, ...]]:
        """Source-shaped device-free receipt with un-compacted empty K planes."""
        row = self.row(m=1)
        samples = (10.0, 11.0, 13.0)
        summary = summarize_sorted_timing(samples)
        identity = f"rocm.nvnni.decode.native_q8_geometry.tn{tn}.cpt{cpt}"
        row.update(candidate_id=identity, observed_candidate_id=identity,
            execution_mode="graph_captured", n="257", k="640", kb="16",
            serial_m1_kb="16", evidence_generation=ROCM_Q8_GEOMETRY_GENERATION,
            min_us=str(summary.minimum), median_us=str(summary.median),
            p95_us=str(summary.p95), mad_us=str(summary.mad), cv=repr(summary.cv),
            timing_sample_digest=summary.digest)

        def launch(tile: int, columns: int, function: int) -> dict:
            def kernel(pointer: int, reducer: bool) -> dict:
                return {"function": pointer,
                    "specialization": f"gemv_native_vnni_{'reduce_' if reducer else ''}kernel_t<{tile},{columns}>",
                    "grid": [(257 + tile - 1) // tile, 1 if reducer else 16, 1],
                    "block": [tile // columns, 1, 1]}
            producer, reducer = kernel(function, False), kernel(function + 1, True)
            def resources(value: dict) -> dict:
                return {"function": value["function"], "specialization": value["specialization"],
                    "block_threads": tile // columns, "registers_per_thread": 32,
                    "local_memory_bytes_per_thread": 0, "static_shared_memory_bytes": 0,
                    "max_threads_per_block": 256, "max_active_blocks_per_sm": 4}
            return {"tile_n": tile, "columns_per_thread": columns,
                "physical_n": 257, "arithmetic_policy_n": 257, "k": 640,
                "device_ordinal": 0, "execution_codebook": 19, "arithmetic_policy_codebook": 19,
                "k_partitions": 16, "blocks_per_partition": 2, "nonempty_partitions": 10,
                "partial_elements": 16 * 257, "stream": 4096, "producer": producer,
                "reducer": reducer, "resources": {"producer": resources(producer), "reducer": resources(reducer)},
                "captured_kernels": [copy.deepcopy(producer), copy.deepcopy(reducer)]}
        baseline = launch(128, 1, 100)
        observed = copy.deepcopy(baseline) if (tn, cpt) == (128, 1) else launch(tn, cpt, 200)
        row["physical_launch_evidence"] = json.dumps({"generation": ROCM_Q8_GEOMETRY_GENERATION,
            "prepared_family": "NativeVNNI_K32_CB19", "baseline": baseline, "launch": observed})
        return row, samples

    def paired_geometry_row(self, operations: int = 1, candidate_index: int = 0):
        """Build paired evidence from registry identity, including true batch counts."""
        entry = rocm_native_q8_serial_geometry_diagnostic_registry().entries[candidate_index]
        row, samples = self.geometry_row(entry.config_json["tile_n"], entry.config_json["columns_per_thread"])
        row.update(evidence_generation=PAIRED_GEOMETRY_GENERATION, operations_per_sample=str(operations))
        evidence = json.loads(row["physical_launch_evidence"])
        evidence["generation"] = PAIRED_GEOMETRY_GENERATION
        for role, count in (("baseline", 1), ("launch", operations)):
            value = evidence[role]
            value["captured_operations"] = count
            value["captured_kernels"].append({"function": 50, "grid": [20, 1, 1], "block": [32, 1, 1]})
            value["captured_kernels"] *= count
        row["physical_launch_evidence"] = json.dumps(evidence)
        base = {name: row[name] for name in ("backend", "phase", "source_format", "source_codebook",
            "execution_codebook", "shape", "execution_mode", "m", "n", "k", "candidate_id", "kb",
            "target_waves", "evidence_generation", "operations_per_sample")}
        timings = [base | {"sample_index": str(index), "timed_replays": str(operations),
            "latency_us": str(value), "latency_us_hex": value.hex(), "round_index": str(index),
            "order_position": str(geometry_timing_position(index, candidate_index)),
            "interval_us": str(value * operations), "interval_us_hex": (value * operations).hex(),
            "ordering_protocol": GEOMETRY_ORDER_PROTOCOL} for index, value in enumerate(samples)]
        return row, timings

    def adapt_paired_fixture(self, row, timings):
        """Exercise the real aggregate/raw parser, not an independently built proof."""
        with tempfile.TemporaryDirectory() as tmp:
            aggregate, sidecar = Path(tmp) / "aggregate.csv", Path(tmp) / "timing.csv"
            for path, values in ((aggregate, [row]), (sidecar, timings)):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(values[0]))
                    writer.writeheader(); writer.writerows(values)
            return adapt_rocm_decode_csv((aggregate,), self.context(), timing_sidecars=(sidecar,), diagnostic_geometry=True)

    def test_shared_geometry_order_balances_positions_and_both_pair_directions(self):
        orders = geometry_timing_orders()
        for order in orders:
            self.assertEqual(sorted(order), list(range(5)))
        for a in range(5):
            for position in range(5):
                self.assertEqual(sum(order[position] == a for order in orders), len(orders) // 5)
            for b in range(a + 1, 5):
                self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), len(orders) // 2)
        for round_index in range(3 * len(orders)):
            for candidate in range(5):
                self.assertEqual(orders[round_index % len(orders)][geometry_timing_position(round_index, candidate)], candidate)

    def test_paired_geometry_retains_every_round_interval_and_separate_batch_regime(self):
        for operations in (1, 32):
            for candidate in range(5):
                row, timings = self.paired_geometry_row(operations, candidate)
                # Latency sorting need not preserve chronological round order.
                timings[0]["round_index"], timings[2]["round_index"] = "2", "0"
                for raw in timings:
                    raw["order_position"] = str(geometry_timing_position(int(raw["round_index"]), candidate))
                corpus = self.adapt_paired_fixture(row, timings)
                observation = corpus.observations[0]
                sampling = observation.physical_launch_evidence["sampling"]
                self.assertEqual([value["round_index"] for value in sampling["rounds"]], [0, 1, 2])
                self.assertEqual(sampling["operations_per_sample"], operations)
                self.assertEqual(sum(value["interval_us"] for value in sampling["rounds"]), 34.0 * operations)
                self.assertFalse(observation.generic_eligible)
                self.assertEqual(observation.config_json, rocm_native_q8_serial_geometry_diagnostic_registry().entries[candidate].config_json)
                paired = paired_geometry_summary(sampling, sampling)
                self.assertEqual(paired["paired_rounds"], len(timings))
                self.assertEqual(paired["paired_median_speedup"], 1.0)

    def test_paired_geometry_rejects_coordinate_count_and_normalization_forgeries(self):
        row, timings = self.paired_geometry_row(32)
        for change in ("position", "duplicate_round", "negative_round", "interval", "replays", "protocol", "missing_round"):
            broken = copy.deepcopy(timings)
            if change == "position": broken[0]["order_position"] = str((int(broken[0]["order_position"]) + 1) % 5)
            elif change == "duplicate_round":
                broken[1]["round_index"] = broken[0]["round_index"]
                broken[1]["order_position"] = broken[0]["order_position"]
            elif change == "negative_round": broken[0]["round_index"] = "-1"
            elif change == "interval": broken[0]["interval_us_hex"] = (321.0).hex()
            elif change == "replays": broken[0]["timed_replays"] = "1"
            elif change == "protocol": broken[0]["ordering_protocol"] = "fixed-candidate-blocks"
            elif change == "missing_round": broken.pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.adapt_paired_fixture(row, broken)
        evidence = json.loads(row["physical_launch_evidence"])
        missing_quantizer = copy.deepcopy(evidence)
        missing_quantizer["launch"]["captured_kernels"][-1]["function"] += 1
        with self.assertRaisesRegex(ValueError, "quantizer"):
            self.adapt_paired_fixture(row | {"physical_launch_evidence": json.dumps(missing_quantizer)}, timings)
        for role in ("baseline", "launch"):
            broken = copy.deepcopy(evidence)
            broken[role]["captured_operations"] = 2
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "projection count"):
                self.adapt_paired_fixture(row | {"physical_launch_evidence": json.dumps(broken)}, timings)

    def test_paired_geometry_cannot_mix_one_and_32_operation_surfaces(self):
        one, one_samples = self.paired_geometry_row(1)
        batch, batch_samples = self.paired_geometry_row(32)
        with tempfile.TemporaryDirectory() as tmp:
            aggregate, sidecar = Path(tmp) / "aggregate.csv", Path(tmp) / "timing.csv"
            for path, values in ((aggregate, [one, batch]), (sidecar, one_samples + batch_samples)):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(values[0]))
                    writer.writeheader(); writer.writerows(values)
            with self.assertRaisesRegex(ValueError, "separate analysis surfaces"):
                adapt_rocm_decode_csv((aggregate,), self.context(), timing_sidecars=(sidecar,), diagnostic_geometry=True)

    def test_paired_summary_uses_matching_rounds_instead_of_unpaired_medians(self):
        candidate = {"protocol": GEOMETRY_ORDER_PROTOCOL, "operations_per_sample": 1,
            "rounds": [{"round_index": i, "latency_us": x, "interval_us": x} for i, x in enumerate((1., 50., 100.))]}
        baseline = copy.deepcopy(candidate)
        for record, value in zip(baseline["rounds"], (2., 100., 1.)):
            record.update(latency_us=value, interval_us=value)
        result = paired_geometry_summary(candidate, baseline)
        self.assertEqual(result["paired_rounds"], 3)
        self.assertAlmostEqual(result["paired_median_speedup"], 2.)
        self.assertNotEqual(result["paired_median_speedup"], 2. / 50.)
        baseline["rounds"].pop()
        with self.assertRaises(ValueError): paired_geometry_summary(candidate, baseline)

    def test_fixed_arithmetic_geometry_uses_five_distinct_common_candidates(self) -> None:
        registry = rocm_native_q8_serial_geometry_diagnostic_registry()
        self.assertEqual(len(registry.entries), 5)
        self.assertEqual(len({entry.arithmetic_fingerprint for entry in registry.entries}), 1)
        observed = set()
        for entry in registry.entries:
            row, samples = self.geometry_row(entry.config_json["tile_n"], entry.config_json["columns_per_thread"])
            observation = adapt_rocm_decode_row(row, self.context(), samples, diagnostic_geometry=True)
            self.assertEqual(observation.effective_candidate_id, entry.candidate_id)
            self.assertTrue(observation.bitwise_equal)
            self.assertFalse(observation.generic_eligible)
            self.assertEqual(observation.physical_launch_evidence["launch"]["partial_elements"], 16 * 257)
            observed.add(observation.effective_candidate_id)
            with self.assertRaises(ValueError):
                rocm_native_vnni_decode_registry().resolve(entry.candidate_id)
        self.assertEqual(len(observed), 5)
        for alias in ("KB16", "KB16/TW4", "TN64CPT1", "AUTO"):
            with self.assertRaises(ValueError): registry.resolve(alias)

    def test_geometry_requires_explicit_generation_raw_samples_and_diagnostic_mode(self) -> None:
        row, samples = self.geometry_row()
        with self.assertRaisesRegex(ValueError, "explicit diagnostic"):
            adapt_rocm_decode_row(row, self.context(), samples)
        with self.assertRaisesRegex(ValueError, "every raw timing"):
            adapt_rocm_decode_row(row, self.context(), diagnostic_geometry=True)
        old = self.row(m=1, candidate="KB16")
        with self.assertRaisesRegex(ValueError, "cannot reinterpret old"):
            adapt_rocm_decode_row(old, self.context(), samples, diagnostic_geometry=True)
        for fields in ({"execution_mode": "eager"}, {"m": "2"},
                       {"source_format": "Q4_0"}, {"evidence_generation": "old-generation"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                adapt_rocm_decode_row(row | fields, self.context(), samples, diagnostic_geometry=True)
        corrupted = row | {"bit_mismatches": "1", "first_bit_mismatch": "0",
                           "grouped_output_digest": "fnv1a64:different000000"}
        with self.assertRaisesRegex(ValueError, "byte identical"):
            adapt_rocm_decode_row(corrupted, self.context(), samples, diagnostic_geometry=True)

    def test_geometry_rejects_installable_context_and_timing_generation_aliases(self) -> None:
        row, samples = self.geometry_row()
        context = replace(self.context(), profile=MeasurementProfile.PRODUCTION)
        with self.assertRaisesRegex(ValueError, "explicit diagnostic"):
            adapt_rocm_decode_row(row, context, samples, diagnostic_geometry=True)
        base = {name: row[name] for name in ("backend", "phase", "source_format",
            "source_codebook", "execution_codebook", "shape", "execution_mode",
            "m", "n", "k", "candidate_id", "kb", "target_waves", "evidence_generation")}
        timing_rows = [base | {"sample_index": str(index), "timed_replays": "1",
            "latency_us": str(value), "latency_us_hex": value.hex()}
            for index, value in enumerate(samples)]
        with tempfile.TemporaryDirectory() as tmp:
            aggregate, sidecar = Path(tmp) / "aggregate.csv", Path(tmp) / "timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row)); writer.writeheader(); writer.writerow(row)
            def write_timing(rows) -> None:
                with sidecar.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(timing_rows[0])); writer.writeheader(); writer.writerows(rows)
            write_timing(timing_rows)
            corpus = adapt_rocm_decode_csv((aggregate,), self.context(),
                timing_sidecars=(sidecar,), diagnostic_geometry=True)
            self.assertEqual(len(corpus), 1)
            for generation in ("", "old-kb-generation"):
                write_timing([value | {"evidence_generation": generation} for value in timing_rows])
                with self.subTest(generation=generation), self.assertRaisesRegex(ValueError, "generation/candidate"):
                    read_rocm_decode_timing_sidecars((sidecar,), workers=1)

    def test_geometry_rejects_requested_only_or_changed_physical_evidence(self) -> None:
        row, samples = self.geometry_row()
        for change in ("requested_geometry", "capture_function", "capture_grid", "duplicate_node",
                       "private_memory", "partition_count", "compacted_planes", "foreign_preparation", "default_stream"):
            broken = json.loads(row["physical_launch_evidence"])
            launch = broken["launch"]
            if change == "requested_geometry": launch["tile_n"] = 256
            elif change == "capture_function": launch["captured_kernels"][0]["function"] += 1
            elif change == "capture_grid": launch["captured_kernels"][0]["grid"][0] += 1
            elif change == "duplicate_node": launch["captured_kernels"].append(copy.deepcopy(launch["captured_kernels"][0]))
            elif change == "private_memory": launch["resources"]["producer"]["local_memory_bytes_per_thread"] = 4
            elif change == "partition_count": launch["k_partitions"] = 10
            elif change == "compacted_planes":
                launch["partial_elements"] = broken["baseline"]["partial_elements"] = 10 * 257
            elif change == "foreign_preparation": broken["prepared_family"] = "Int8Blockwise"
            elif change == "default_stream": launch["stream"] = broken["baseline"]["stream"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                adapt_rocm_decode_row(row | {"physical_launch_evidence": json.dumps(broken)},
                    self.context(), samples, diagnostic_geometry=True)

    def test_geometry_rejects_coordinated_baseline_and_candidate_tree_mutation(self) -> None:
        """Two matching forged receipts cannot replace the original fixed tree."""
        row, samples = self.geometry_row()
        evidence = json.loads(row["physical_launch_evidence"])
        for value in (evidence["baseline"], evidence["launch"]):
            value["k_partitions"] = 10
            value["partial_elements"] = 10 * 257
            value["producer"]["grid"][1] = 10
            value["captured_kernels"][0]["grid"][1] = 10
        changed = row | {"kb": "10", "serial_m1_kb": "10",
                         "physical_launch_evidence": json.dumps(evidence)}
        with self.assertRaisesRegex(ValueError, "canonical original serial"):
            adapt_rocm_decode_row(changed, self.context(), samples, diagnostic_geometry=True)
        # The original helper legitimately bounds sixteen by real K32 blocks.
        for value in (evidence["baseline"], evidence["launch"]):
            value.update(k=160, k_partitions=5, blocks_per_partition=1,
                         nonempty_partitions=5, partial_elements=5 * 257)
            value["producer"]["grid"][1] = 5
            value["captured_kernels"][0]["grid"][1] = 5
        bounded = row | {"k": "160", "kb": "5", "serial_m1_kb": "5",
                         "physical_launch_evidence": json.dumps(evidence)}
        observation = adapt_rocm_decode_row(bounded, self.context(), samples, diagnostic_geometry=True)
        self.assertEqual(observation.physical_launch_evidence["launch"]["k_partitions"], 5)

    def test_geometry_common_observation_roundtrip_preserves_actual_launch_receipts(self) -> None:
        row, samples = self.geometry_row()
        observation = adapt_rocm_decode_row(row, self.context(), samples, diagnostic_geometry=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.csv"
            write_observation_csv(path, (observation,), workers=1)
            restored = tuple(read_observation_csv((path,)))[0]
        self.assertEqual(restored.canonical_mapping(), observation.canonical_mapping())
        self.assertEqual(restored.digest(), observation.digest())
        legacy = adapt_rocm_decode_row(self.row(), self.context())
        self.assertNotIn("physical_launch_evidence", legacy.canonical_mapping())
        self.assertEqual(NativeVNNIObservation.from_mapping(legacy.canonical_mapping()).digest(), legacy.digest())

    def test_verifier_inherits_serial_policy_and_is_exactly_eligible(self) -> None:
        observation = adapt_rocm_decode_row(self.row(), self.context())

        self.assertEqual(
            observation.semantic_contract,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )
        self.assertEqual(
            observation.candidate_id,
            "rocm.nvnni.decode.verifier.inherit_serial_m1",
        )
        self.assertTrue(candidate_is_eligible(
            observation, self.context().serial_m1_policy_hash
        ))

    def test_extended_runtime_m_keeps_the_grouped_verifier_contract(self) -> None:
        observation = adapt_rocm_decode_row(self.row(m=31), self.context())

        self.assertEqual(observation.m, 31)
        self.assertEqual(
            observation.semantic_contract,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )

    def test_explicit_kb_cannot_masquerade_as_grouped_verifier_candidate(self) -> None:
        row = self.row(candidate="KB32")
        row["observed_candidate_id"] = "KB32"

        with self.assertRaisesRegex(ValueError, "does not support Verifier"):
            adapt_rocm_decode_row(row, self.context())

    def test_atomic_publication_evidence_is_rejected(self) -> None:
        """A retired completion-order reduction cannot enter a corpus."""

        row = self.row()
        row["observed_path"] = "atomic_reduce"

        with self.assertRaisesRegex(ValueError, "unsupported publication path"):
            adapt_rocm_decode_row(row, self.context())

    def test_fast_m1_candidate_may_be_numerically_correct_without_byte_identity(self) -> None:
        row = self.row(m=1, candidate="KB8/TW24")
        row.update({
            "kb": "8",
            "observed_candidate_id": "KB8",
            "bit_mismatches": "3",
            "first_bit_mismatch": "2",
            "max_abs": "0.000001",
            "relative_l2": "0.0000001",
            "grouped_output_digest": "fnv1a64:different000000",
        })

        observation = adapt_rocm_decode_row(row, self.context())

        self.assertEqual(observation.semantic_contract, SemanticContract.FAST)
        self.assertEqual(
            observation.candidate_id,
            "rocm.nvnni.decode.fast.kb8",
        )
        self.assertFalse(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(observation))

    def test_exact_timing_sidecar_reconstructs_digest_and_robust_statistics(self) -> None:
        samples = (10.0, 11.0, 13.0)
        summary = summarize_sorted_timing(samples)
        row = self.row()
        row.update({
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": repr(summary.cv),
            "timing_sample_digest": summary.digest,
        })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate = root / "decode.csv"
            sidecar = root / "decode-timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            timing_rows = []
            for index, value in enumerate(samples):
                timing_rows.append({
                    "backend": "rocm",
                    "phase": "decode",
                    "source_format": "Q8_0",
                    "source_codebook": "19",
                    "execution_codebook": "19",
                    "shape": "35BMoE_Expert_GateUp",
                    "execution_mode": "eager",
                    "m": "2",
                    "n": "512",
                    "k": "2048",
                    "candidate_id": "INHERIT_SERIAL_M1",
                    "kb": "32",
                    "target_waves": "4",
                    "sample_index": str(index),
                    "timed_replays": "1",
                    "latency_us": f"{value:.9f}",
                    "latency_us_hex": value.hex(),
                })
            with sidecar.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(timing_rows[0]))
                writer.writeheader()
                writer.writerows(timing_rows)

            context = ROCmDecodeAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate, sidecar))
            )
            corpus = adapt_rocm_decode_csv(
                (aggregate,), context, timing_sidecars=(sidecar,)
            )
            self.assertEqual(len(corpus), 1)
            self.assertEqual(corpus.observations[0].timing_sample_hash, summary.digest)

            row["mad_us"] = "99"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "aggregate mad_us"):
                adapt_rocm_decode_csv(
                    (aggregate,), context, timing_sidecars=(sidecar,)
                )

    def test_parallel_timing_reader_preserves_complete_trial_sequences(self) -> None:
        """A byte split inside one trial must advance to the next trial."""

        rows = []
        for shape, samples in (
            ("LargeFirstTrial", tuple(float(index + 1) for index in range(20))),
            ("SmallSecondTrial", (101.0, 102.0, 103.0)),
        ):
            for sample_index, latency in enumerate(samples):
                rows.append({
                    "backend": "rocm",
                    "phase": "decode",
                    "source_format": "Q8_0",
                    "source_codebook": "19",
                    "execution_codebook": "19",
                    "shape": shape,
                    "execution_mode": "eager",
                    "m": "1",
                    "n": "512",
                    "k": "2048",
                    "candidate_id": "KB32/TW4",
                    "kb": "32",
                    "target_waves": "4",
                    "sample_index": str(sample_index),
                    "timed_replays": "1",
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                })

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rocm-timing.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            serial = read_rocm_decode_timing_sidecars((path,), workers=1)
            parallel = read_rocm_decode_timing_sidecars((path,), workers=2)

        self.assertEqual(parallel, serial)
        self.assertEqual(len(parallel), 2)
        self.assertEqual(sorted(len(samples) for samples in parallel.values()), [3, 20])

    def test_timing_reader_rejects_nonpositive_worker_count(self) -> None:
        """Explicit analyzer parallelism must remain a positive contract."""

        with self.assertRaisesRegex(ValueError, "worker count must be positive"):
            read_rocm_decode_timing_sidecars((Path("unused.csv"),), workers=0)


if __name__ == "__main__":
    unittest.main()
