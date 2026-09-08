"""Unit regressions for strict CPU NativeVNNI M=1 evidence adaptation."""

from __future__ import annotations

import csv
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
CPU_ANALYZER_ROOT = KERNEL_PERF_ROOT / "cpu"
for root in (KERNEL_PERF_ROOT, CPU_ANALYZER_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import analyze_cpu_native_vnni_decode_trainer as decode_analyzer  # noqa: E402
from analyze_cpu_native_vnni_decode_trainer import (  # noqa: E402
    _load_reused_development_common,
)

from native_vnni_dispatch.adapters.cpu_decode import (  # noqa: E402
    CPUDecodeAdapterContext,
    adapt_cpu_decode_csv,
    adapt_cpu_decode_csv_to_common,
    adapt_cpu_decode_row,
    read_cpu_decode_timing_sidecars,
)
from native_vnni_dispatch.adapters.evidence import (  # noqa: E402
    native_double_digest,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    write_observation_csv,
)
from native_vnni_dispatch.format_registry import format_spec  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import NativeVNNIObservation  # noqa: E402


def context() -> CPUDecodeAdapterContext:
    """Return non-installable provenance suitable for pure adapter tests."""

    return CPUDecodeAdapterContext(
        profile=MeasurementProfile.QUICK,
        run_id="unit-run",
        corpus_id="sha256:" + "1" * 64,
        git_revision="unit-revision",
        build_id="unit-build",
        compiler_id="unit-compiler",
        architecture_class="unit-cpu",
        device_name="unit-device",
        driver_runtime="unit-runtime",
        frozen_serial_policy_hash="sha256:" + "2" * 64,
        raw_timing_sidecar_retained=False,
    )


def raw_row() -> dict[str, str]:
    """Return one byte-exact, route-authenticated NBC4 observation."""

    spec = format_spec("Q4_K")
    return {
        "backend": "cpu",
        "phase": "decode_m1",
        "source_format": "Q4_K",
        "source_codebook": str(spec.source_codebook_id),
        "execution_codebook": str(spec.runtime_codebook("cpu")),
        "shape": "UnitDecode",
        "execution_mode": "eager",
        "m": "1",
        "n": "5120",
        "k": "5120",
        "candidate_id": "cpu.nvnni.decode.n_chunk_grid.nbc4",
        "build_isa": "AVX512",
        "runtime_isa_requested": "AVX512",
        "runtime_isa_effective": "AVX512",
        "threads": "28",
        "weight_bytes": "4194304",
        "warmup_count": "1",
        "sample_count": "1",
        "min_us": "10.0",
        "median_us": "10.0",
        "p95_us": "10.0",
        "mad_us": "0.0",
        "cv": "0.0",
        "bit_mismatches": "0",
        "first_bit_mismatch": "0",
        "repeat_byte_mismatches": "0",
        "max_abs": "0.0",
        "relative_l2": "0.0",
        "cosine": "1.0",
        "symmetric_kld": "0.0",
        "output_digest": "sha256:" + "3" * 64,
        "oracle_output_digest": "sha256:" + "3" * 64,
        "timing_sample_digest": "sha256:" + "4" * 64,
        "route_counter_ok": "1",
        "observed_candidate_id": "Nbc4",
        "k_tiles": "0",
        "n_block_chunks": "4",
        "serial_kpart": "0",
        "numerical_correctness": "1",
        "correctness_pass": "1",
        "is_winner": "1",
    }


class CPUDecodeAdapterTest(unittest.TestCase):
    """Protect byte, route, ISA, and arithmetic-bundle admission gates."""

    def test_corpus_identity_canonicalizes_relative_path_aliases(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            evidence = Path(directory) / "evidence.csv"
            evidence.write_bytes(b"candidate,timing\nnbc4,1.0\n")
            relative = evidence.relative_to(Path.cwd())

            self.assertEqual(
                raw_corpus_id((relative,)),
                raw_corpus_id((evidence.resolve(),)),
            )

            moved = evidence.with_name("moved.csv")
            moved.write_bytes(evidence.read_bytes())
            self.assertNotEqual(
                raw_corpus_id((relative,)),
                raw_corpus_id((moved,)),
            )

    def test_adapts_forceable_byte_exact_candidate(self) -> None:
        observation = adapt_cpu_decode_row(raw_row(), context())

        self.assertTrue(observation.supported)
        self.assertTrue(observation.bitwise_equal)
        self.assertEqual(observation.m, 1)
        self.assertEqual(
            observation.operation_kind, "NativeVNNIFastM1Projection"
        )
        self.assertIn("serial-full-k", observation.bundle_signature)
        self.assertEqual(observation.launch_k_tiles, 0)

    def test_normalized_nominal_candidate_is_not_supported(self) -> None:
        raw = raw_row()
        raw["candidate_id"] = "cpu.nvnni.decode.n_chunk_grid.nbc8"
        raw["correctness_pass"] = "0"

        observation = adapt_cpu_decode_row(raw, context())

        self.assertFalse(observation.supported)
        self.assertFalse(observation.forced_route_ok)

    def test_rejects_digest_mismatch_hidden_behind_zero_count(self) -> None:
        raw = raw_row()
        raw["oracle_output_digest"] = "sha256:" + "5" * 64

        with self.assertRaisesRegex(ValueError, "mismatch count"):
            adapt_cpu_decode_row(raw, context())

    def test_rejects_non_m1_evidence(self) -> None:
        raw = raw_row()
        raw["m"] = "2"

        with self.assertRaisesRegex(ValueError, "requires M=1"):
            adapt_cpu_decode_row(raw, context())

    def test_kpart_bundle_is_bound_to_route_counter(self) -> None:
        raw = raw_row()
        raw["k_tiles"] = "4"
        raw["serial_kpart"] = "1"

        observation = adapt_cpu_decode_row(raw, context())

        self.assertIn("serial-kpart", observation.bundle_signature)
        self.assertEqual(observation.launch_k_tiles, 4)

    def test_rejects_inconsistent_kpart_marker(self) -> None:
        raw = raw_row()
        raw["k_tiles"] = "4"

        with self.assertRaisesRegex(ValueError, "serial_kpart"):
            adapt_cpu_decode_row(raw, context())

    def test_installable_timing_floor_is_explicit_and_overrideable(self) -> None:
        """Reviewed best-effort M=1 evidence still verifies its raw sample."""

        samples = (10.0,)
        raw = raw_row()
        raw["timing_sample_digest"] = native_double_digest(samples)
        production = dataclasses.replace(
            context(),
            profile=MeasurementProfile.PRODUCTION,
            run_id="unit-production",
            git_revision="deadbeef",
            build_id="sha256:" + "3" * 64,
            compiler_id="gcc-13",
            architecture_class="x86_64-unit",
            device_name="unit-cpu",
            driver_runtime="linux-unit",
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(ValueError, "requires 5/30 timing"):
            adapt_cpu_decode_row(raw, production, samples)

        best_effort = dataclasses.replace(
            production,
            minimum_promotion_warmups=1,
            minimum_promotion_samples=1,
        )
        best_effort.validate()
        observation = adapt_cpu_decode_row(raw, best_effort, samples)
        self.assertEqual(observation.warmup_count, 1)
        self.assertEqual(observation.sample_count, 1)

    def test_parallel_timing_and_adaptation_match_serial_bytes(self) -> None:
        """Large-corpus worker paths preserve every observation and CSV byte."""

        samples = (10.0, 10.0, 10.0)
        rows = []
        timing_rows = []
        for index in range(8):
            raw = raw_row()
            raw["shape"] = f"UnitDecode{index}"
            raw["n"] = str(5120 + 32 * index)
            raw["sample_count"] = str(len(samples))
            raw["timing_sample_digest"] = native_double_digest(samples)
            rows.append(raw)
            for sample_index, latency in enumerate(samples):
                timing_rows.append({
                    "backend": raw["backend"],
                    "phase": raw["phase"],
                    "source_format": raw["source_format"],
                    "source_codebook": raw["source_codebook"],
                    "execution_codebook": raw["execution_codebook"],
                    "shape": raw["shape"],
                    "execution_mode": raw["execution_mode"],
                    "m": raw["m"],
                    "n": raw["n"],
                    "k": raw["k"],
                    "candidate_id": raw["candidate_id"],
                    "build_isa": raw["build_isa"],
                    "runtime_isa_requested": raw["runtime_isa_requested"],
                    "runtime_isa_effective": raw["runtime_isa_effective"],
                    "sample_index": str(sample_index),
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate.csv"
            timing = root / "timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with timing.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=tuple(timing_rows[0])
                )
                writer.writeheader()
                writer.writerows(timing_rows)

            serial_timing = read_cpu_decode_timing_sidecars(
                (timing,), workers=1
            )
            parallel_timing = read_cpu_decode_timing_sidecars(
                (timing,), workers=2, parallel_threshold_bytes=0
            )
            self.assertEqual(parallel_timing, serial_timing)

            serial = adapt_cpu_decode_csv(
                (aggregate,), context(), timing_sidecars=(timing,)
            )
            expected = root / "expected.csv"
            write_observation_csv(expected, serial, workers=1)
            actual = root / "actual.csv"
            with mock.patch.object(
                NativeVNNIObservation,
                "from_mapping",
                side_effect=AssertionError(
                    "parallel adapter reparsed its canonical CSV"
                ),
            ):
                parallel = adapt_cpu_decode_csv_to_common(
                    (aggregate,),
                    context(),
                    actual,
                    timing_sidecars=(timing,),
                    workers=2,
                    parallel_threshold=0,
                )

            self.assertEqual(parallel.observations, serial.observations)
            self.assertEqual(actual.read_bytes(), expected.read_bytes())
            reused = _load_reused_development_common(actual, context())
            self.assertEqual(reused.observations, serial.observations)
            with self.assertRaisesRegex(
                ValueError, "changed raw/build provenance"
            ):
                _load_reused_development_common(
                    actual,
                    dataclasses.replace(context(), build_id="stale-build"),
                )

    def test_prepared_cross_aspect_corpus_is_the_fitted_object(self) -> None:
        """Freeze must preserve the corpus whose digest probe publication reuses."""

        source = mock.Mock()
        validated = mock.Mock()
        prepared = mock.Mock()
        prepared.distinguishes_aspect_bucket = False
        validated.with_collapsed_aspect_domains.return_value = prepared
        manifest = mock.Mock()
        manifest.schema_version = "unit-shape-manifest"
        manifest.digest.return_value = "sha256:" + "a" * 64
        frozen = object()

        with mock.patch.object(
            decode_analyzer,
            "_require_partition",
            return_value=validated,
        ):
            actual = decode_analyzer._prepare_cpu_decode_policy_corpus(
                source, manifest
            )
        self.assertIs(actual, prepared)

        with (
            mock.patch.object(
                decode_analyzer,
                "cpu_decode_sealed_reserve_commitment",
                return_value="sha256:" + "b" * 64,
            ),
            mock.patch.object(
                decode_analyzer,
                "paired_comparison_digest",
                return_value="sha256:" + "c" * 64,
            ),
            mock.patch.object(
                decode_analyzer,
                "freeze_policy",
                return_value=frozen,
            ) as fit,
        ):
            result = decode_analyzer._freeze_prepared_cpu_decode_policy(
                prepared, manifest
            )

        self.assertIs(result, frozen)
        self.assertIs(fit.call_args.args[0], prepared)


if __name__ == "__main__":
    unittest.main()
