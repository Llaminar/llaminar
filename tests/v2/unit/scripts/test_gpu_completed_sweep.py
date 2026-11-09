"""Device-free GPU receipt reuse and missing-shape launch regressions.

Authored tiny CSVs isolate lifecycle/provenance behavior; validator spies never
certify their values as native measurements. Real strong adapters separately
reject truncated artifacts. Production arithmetic/profiler proof stays native.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/performance/kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_quantised_dispatch import gpu_completed_sweep as reuse
from native_quantised_dispatch.corpus_provenance import sha256_file

spec = importlib.util.spec_from_file_location("gpu_completed_public", ROOT / "scripts/native_quantised/exact_refresh.py")
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


def csv_rows(path, rows):
    """Write inert identity rows; tests stub only native numerical validation."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shape", "candidate_id", "latency_us_hex", "sample_index"))
        for shape in rows:
            writer.writerow((shape, "original-candidate", float(12.5).hex(), "0"))


def contract(shapes=("shared", "missing")):
    """Author a complete published context, keeping every discriminator explicit."""
    return dict(build=dict(build_id="sha256:" + "1" * 64, git_revision="a" * 40,
        compiler_id="published-native-compiler", architecture_class="sm_86",
        device_name="published-device", driver_runtime="published-runtime",
        serial_m1_policy_hash="sha256:" + "2" * 64), shapes=list(shapes), m="1",
        devices=[0, 1], formats="original-source-formats", warmups=5, samples=30, repeats=8)


def original(root, data):
    """Commit inert source files with the existing native collection receipt ABI."""
    root.mkdir(parents=True)
    (root / "collection-contract.json").write_text(json.dumps(data))
    for name in ("aggregate.csv", "timing.csv"):
        csv_rows(root / name, data["shapes"])
    (root / "collection-complete.json").write_text(json.dumps({
        "aggregate": sha256_file(root / "aggregate.csv"), "timing": sha256_file(root / "timing.csv")}))
    return {str(p): p.read_bytes() for p in root.iterdir()}


def strong_grouped_fixture(root, backend, *, raw_warmups=5, raw_samples=30, raw_replays=None):
    """Expand existing strong-reader builders into one complete authored M=2 cell.

    Every production source alias, eager/captured mode and reachable verifier
    candidate is present. These inert fixtures assert reader/schema behavior;
    their authored counters are never published as native device qualification.
    """
    from native_quantised_dispatch.adapters.evidence import summarize_sorted_timing
    from native_quantised_dispatch.format_registry import FORMAT_SPECS
    from test_cuda_native_quantised_decode_adapter import CUDANativeQuantisedDecodeAdapterTest
    from test_rocm_native_quantised_decode_adapter import ROCmNativeQuantisedDecodeAdapterTest
    shape = "35BMoE_Expert_GateUp"
    data = contract((shape,));data["m"] = "2"
    data["formats"] = ",".join(spec.label for spec in FORMAT_SPECS)
    if backend == "rocm":
        data["build"]["architecture_class"] = "gfx906"
    repeats = (8 if backend == "cuda" else 1) if raw_replays is None else raw_replays
    samples = (10.5,) * raw_samples
    summary = summarize_sorted_timing(samples)
    aggregate, timing = [], []
    for spec in FORMAT_SPECS:
        for mode in ("eager", "graph_captured"):
            builders = ((CUDANativeQuantisedDecodeAdapterTest.verifier_row,
                         CUDANativeQuantisedDecodeAdapterTest.tensor_core_verifier_row)
                        if backend == "cuda" else (ROCmNativeQuantisedDecodeAdapterTest.row,))
            for order, builder in enumerate(builders):
                row = builder()
                row.update(shape=shape, n=512, k=2048, m=2, source_format=spec.label,
                    source_codebook=spec.source_codebook_id, execution_codebook=spec.runtime_codebook(backend),
                    execution_mode=mode, warmup_count=raw_warmups, sample_count=raw_samples,
                    min_us=summary.minimum, median_us=summary.median, p95_us=summary.p95,
                    mad_us=summary.mad, cv=summary.cv, timing_sample_digest=summary.digest)
                if backend == "cuda":
                    if row["family"] == "inherit_serial_m1":
                        identity = "cuda.native_quantised.decode.verifier.inherit_serial_m1.r2"
                        row.update(candidate_id=identity, observed_candidate_id=identity, grouped_rows=2)
                    row.update(measurement_order=order, measurement_order_seed=123456789)
                aggregate.append(row)
                for index, latency in enumerate(samples):
                    sample = dict(row, sample_index=index, timed_replays=repeats,
                        latency_us=latency, latency_us_hex=latency.hex())
                    if backend == "cuda":
                        sample.update(sample_measurement_order=order, sample_order_seed=9000 + index)
                    timing.append(sample)
    root.mkdir()
    for name, rows in (("aggregate.csv", aggregate), ("timing.csv", timing)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return data, (shape,)


class GPUCompletedSweepTest(unittest.TestCase):
    def test_real_cuda_validator_accepts_complete_strong_matrix_and_checks_original_floors(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "native-reader-fixture"
            old, names = strong_grouped_fixture(source, "cuda")
            arguments = ("cuda", source, source / "aggregate.csv", source / "timing.csv")
            self.assertEqual(reuse._validated_shapes(*arguments, old, names, old), names)
            # Counts are already at the general production floor; the original
            # collection's stricter promise must independently remain true.
            for field in ("warmups", "samples"):
                with self.subTest(original_floor=field):
                    stricter = copy.deepcopy(old);stricter[field] += 1
                    self.assertEqual(reuse._validated_shapes(*arguments, stricter, names, old), ())
            changed_replays = copy.deepcopy(old);changed_replays["repeats"] += 1
            with self.assertRaisesRegex(ValueError, "replay divisor"):
                reuse._validated_shapes(*arguments, changed_replays, names, old)

    def test_real_rocm_validator_accepts_complete_strong_matrix_and_checks_original_floors(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "native-reader-fixture"
            old, names = strong_grouped_fixture(source, "rocm")
            arguments = ("rocm", source, source / "aggregate.csv", source / "timing.csv")
            self.assertEqual(reuse._validated_shapes(*arguments, old, names, old), names)
            for field in ("warmups", "samples"):
                with self.subTest(original_floor=field):
                    stricter = copy.deepcopy(old);stricter[field] += 1
                    self.assertEqual(reuse._validated_shapes(*arguments, stricter, names, old), ())
            # The unchanged ROCm native protocol uses one launch/sample, even
            # though public repeats remains an orchestration contract field.
            timing_path = source / "timing.csv"
            with timing_path.open(newline="") as stream:
                reader = csv.DictReader(stream);fields = reader.fieldnames;rows = list(reader)
            self.assertEqual({row["timed_replays"] for row in rows}, {"1"})
            rows[0]["timed_replays"] = "8"
            with timing_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "replay count"):
                reuse._validated_shapes(*arguments, old, names, old)

    def test_superset_projection_retains_original_context_samples_and_source_bytes(self):
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                source, target = base / "old" / backend / "decode/m1", base / "new"
                target.mkdir()
                wanted = contract(("shared",))
                old = contract(("legacy-only", "shared", "other"))
                old.update(samples=40, warmups=7)
                original_bytes = original(source, old)
                with patch.object(reuse, "_validated_shapes", side_effect=lambda b, r, a, t, c, s, w: s) as validate:
                    outputs, missing = reuse.retain_completed_decode_shapes(base / "old", backend, wanted, target)
                self.assertEqual(missing, ())
                self.assertEqual(validate.call_args.args[4], old)
                self.assertEqual(validate.call_args.args[5], ("shared",))
                self.assertEqual(len(outputs), 1)
                for path in outputs[0]:
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([row["shape"] for row in rows], ["shared"])
                    self.assertEqual(rows[0]["latency_us_hex"], float(12.5).hex())
                    self.assertEqual(rows[0]["candidate_id"], "original-candidate")
                receipt = json.loads((target / "collection-reuse.json").read_text())
                self.assertEqual(receipt["sources"][0]["original_contract"], old)
                self.assertEqual(receipt["requested_contract"], wanted)
                reuse.verify_retained_decode_sources(target)
                self.assertEqual(reuse.resume_retained_decode_shapes(target, backend, wanted), (outputs, ()))
                self.assertEqual({name: Path(name).read_bytes() for name in original_bytes}, original_bytes)

    def test_every_actual_producer_hardware_phase_and_timing_axis_is_required(self):
        mutations = (("build", "build_id"), ("build", "architecture_class"),
            ("build", "device_name"), ("build", "compiler_id"), ("build", "driver_runtime"),
            ("build", "serial_m1_policy_hash"), ("build", "git_revision"),
            ("devices", None), ("formats", None), ("m", None), ("repeats", None),
            ("warmups", None), ("samples", None))
        for axis, field in mutations:
            with self.subTest(axis=axis, field=field), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary); wanted = contract(); old = copy.deepcopy(wanted)
                if field: old[axis][field] = "changed"
                else: old[axis] = {"devices": [1, 0], "formats": "changed", "m": "2", "repeats": 1,
                                  "warmups": 1, "samples": 3}[axis]
                source, target = base / "old", base / "new";target.mkdir();original(source, old)
                with patch.object(reuse, "_validated_shapes", side_effect=AssertionError("incompatible data adapted")):
                    outputs, missing = reuse.retain_completed_decode_shapes(source, "cuda", wanted, target)
                self.assertEqual(outputs, [])
                self.assertEqual(missing, tuple(wanted["shapes"]))

    def test_incomplete_changed_and_second_generation_sources_are_never_imported(self):
        for failure in ("missing-complete", "changed-aggregate", "second-generation"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
                wanted = contract();original(source, wanted)
                if failure == "missing-complete": (source / "collection-complete.json").unlink()
                elif failure == "changed-aggregate": (source / "aggregate.csv").write_text("changed")
                else: (source / "collection-reuse.json").write_text('{}')
                with patch.object(reuse, "_validated_shapes", side_effect=AssertionError("inadmissible source adapted")):
                    outputs, missing = reuse.retain_completed_decode_shapes(source, "cuda", wanted, target)
                self.assertFalse(outputs);self.assertEqual(missing, tuple(wanted["shapes"]))

    def test_original_or_retained_mutation_invalidates_resume(self):
        for field in ("aggregate.csv", "timing.csv", "collection-contract.json", "collection-complete.json", "retained"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
                wanted = contract();original(source, wanted)
                with patch.object(reuse, "_validated_shapes", side_effect=lambda b, r, a, t, c, s, w: s):
                    outputs, _ = reuse.retain_completed_decode_shapes(source, "cuda", wanted, target)
                changed = outputs[0][0] if field == "retained" else source / field
                changed.write_text(changed.read_text() + "changed")
                with self.assertRaises((ValueError, OSError)):
                    reuse.resume_retained_decode_shapes(target, "cuda", wanted)

    def test_failed_or_incomplete_shapes_are_missing_even_if_sibling_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
            wanted = contract();original(source, wanted)
            with patch.object(reuse, "_validated_shapes", return_value=("shared",)):
                outputs, missing = reuse.retain_completed_decode_shapes(source, "cuda", wanted, target)
            self.assertEqual(missing, ("missing",))
            self.assertEqual(json.loads((target / "collection-reuse.json").read_text())["sources"][0]["shapes"], ["shared"])
            with outputs[0][0].open(newline="") as stream:
                self.assertEqual([row["shape"] for row in csv.DictReader(stream)], ["shared"])

    def test_actual_strong_adapters_reject_truncated_csv_instead_of_reusing_it(self):
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
                wanted = contract();original(source, wanted)
                outputs, missing = reuse.retain_completed_decode_shapes(source, backend, wanted, target)
                self.assertFalse(outputs);self.assertEqual(missing, tuple(wanted["shapes"]))
                self.assertTrue(json.loads((target / "collection-reuse.json").read_text())["declined"])

    def test_collector_launches_only_missing_shapes_and_resumes_original_references(self):
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
                args = public.parser().parse_args(["--backend", backend, "--output-dir", str(base / "public"),
                    "--reuse-completed-root", str(source), "--model-scope", "all-retained"])
                names = public.decode_shape_names("modern")[:2]
                wanted = contract(names)
                wanted["formats"] = ",".join(spec.label for spec in public.FORMAT_SPECS)
                original(source, wanted)
                observed = []
                def native(command, *, env, log):
                    prefix = "LLAMINAR_" + backend.upper() + "_NATIVE_QUANTISED_DECODE_"
                    shapes = env[prefix + "SHAPES"].split(",");observed.extend(shapes)
                    for field in ("CSV", "TIMING_CSV"):csv_rows(Path(env[prefix + field]), shapes)
                with patch.object(reuse, "_validated_shapes", return_value=(names[0],)), patch.object(public, "run", side_effect=native):
                    result = public.collect(args, backend, Path("/not-a-native-binary"), target,
                        "1", names, (0, 1), wanted["build"])
                self.assertEqual(observed, [names[1]])
                completion = json.loads((target / "collection-complete.json").read_text())
                self.assertIn("retained_originals", completion)
                args.resume = True;args.reuse_completed_root = None
                with patch.object(public, "run", side_effect=AssertionError("paid shape repeated")):
                    self.assertEqual(public.collect(args, backend, Path("/unused"), target,
                        "1", names, (0, 1), wanted["build"]), result)
                # An interrupted later native launch does not invalidate original retained clocks.
                (target / "collection-complete.json").unlink();observed.clear()
                with patch.object(public, "run", side_effect=native):
                    public.collect(args, backend, Path("/unused"), target, "1", names, (0, 1), wanted["build"])
                self.assertEqual(observed, [names[1]])

    def test_complete_reuse_never_launches_native_and_binds_its_reference_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary);source, target = base / "old", base / "new";target.mkdir()
            args = public.parser().parse_args(["--backend", "cuda", "--output-dir", str(base / "public"),
                "--reuse-completed-root", str(source), "--model-scope", "all-retained"])
            names = public.decode_shape_names("modern")[:1];wanted = contract(names)
            wanted["formats"] = ",".join(spec.label for spec in public.FORMAT_SPECS)
            original(source, wanted)
            with patch.object(reuse, "_validated_shapes", return_value=names), patch.object(public, "run", side_effect=AssertionError("paid shape repeated")):
                public.collect(args, "cuda", Path("/unused"), target, "1", names, (0, 1), wanted["build"])
            args.resume = True
            path = target / "collection-reuse.json";path.write_text(path.read_text() + " ")
            with patch.object(public, "run", side_effect=AssertionError("changed evidence launched")):
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    public.collect(args, "cuda", Path("/unused"), target, "1", names, (0, 1), wanted["build"])


if __name__ == "__main__":
    unittest.main()
