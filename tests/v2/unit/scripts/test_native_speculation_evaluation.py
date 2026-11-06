#!/usr/bin/env python3
## @file test_native_speculation_evaluation.py
## @brief Device-free contracts for the actual offline speculation evaluator.
##
## Synthetic native-shaped JSON exercises identity admission, exact-token
## comparison and cost arithmetic through the production consumer. These are
## evidence-contract tests, not recorded inference replay or performance proof.
"""Adversarial native speculation evaluation contracts; no runtime or models."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
evaluator = importlib.import_module("evaluate_native_speculation")


class EvidenceContractTests(unittest.TestCase):
    """Exercise the same frozen-file interface used by an independent evaluator."""

    def setUp(self):
        """Create isolated synthetic native outputs and three paired blocks."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.plan_path = self.directory / "plan.json"
        self.evidence_path = self.directory / "evidence.json"
        self.build = {"build_type": "Release", "source_revision": "a" * 40,
                      "binary_sha256": "a" * 64, "compiler": {"cxx": "fixture-compiler"},
                      "runtime_libraries": {"openmp": "fixture-runtime-digest"}, "cpu_isa": "AVX512"}
        self.config = {"benchmark_mode": True, "prefix_cache_enabled": True, "mtp_enabled": True,
                       "mtp_verify_mode": "speculative-sampling", "mtp_depth_policy": "dynamic",
                       "mtp_draft_tokens": 3, "mtp_graph_capacity_draft_tokens": 15, "mtp_max_request_batch": 1}
        implementations = {}
        for role in evaluator.Role:
            build = self.build if role is not evaluator.Role.CANDIDATE else {
                **self.build, "source_revision": "b" * 40, "binary_sha256": "b" * 64}
            implementations[role.value] = {
                "build": self.save(f"{role.value}-build.json", build),
                "config": {**self.config, "mtp_enabled": role is not evaluator.Role.AR}}
        self.plan_document = {
            "schema": evaluator.PLAN_SCHEMA,
            "deployment": {key: self.save(f"{key}.json", {"synthetic": key})
                           for key in ("model", "hardware", "execution")},
            "implementations": implementations, "pairs": ["a", "b", "c"],
            "requests": [{"id": split.value, "split": split.value, "workload": {
                "schema": 1, "prompt_unit": split.value, "prompt_repetitions": 1,
                "decode_tokens": 8, "warmup_iterations": 1, "measurement_iterations": 3,
                "seed": 42, "temperature": 0.8, "top_k": 40, "top_p": 0.9}}
                         for split in evaluator.Split]}
        self.refresh_plan()
        self.evidence = {"schema": evaluator.EVIDENCE_SCHEMA, "plan_sha256": self.plan.sha256, "observations": []}
        for request in self.plan_document["requests"]:
            for pair in self.plan_document["pairs"]:
                for role in evaluator.Role:
                    filename = f"{request['id']}-{pair}-{role.value}.json"
                    data = self.measurement(request, role)
                    # Native outputs normally differ in timing/epochs. The
                    # fixture marker makes independent synthetic files unique.
                    data["fixture_identity"] = filename
                    self.evidence["observations"].append({
                        "request": request["id"], "pair": pair, "role": role.value,
                        "deployment_sha256": self.plan.deployment_sha256,
                        "build_sha256": self.plan.builds[role.value], "measurement": "unprofiled",
                        "benchmark": self.save(filename, data)})
        self.save("evidence.json", self.evidence)

    def save(self, name, document):
        """Persist fixture bytes and return their real artifact identity."""
        payload = (json.dumps(document, sort_keys=True) + "\n").encode()
        (self.directory / name).write_bytes(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    def refresh_plan(self):
        """Freeze fixture intent before any observation is admitted."""
        reference = self.save("plan.json", self.plan_document)
        self.plan = evaluator.load_plan(self.plan_path, reference["sha256"])

    def measurement(self, request, role):
        """Mirror BenchmarkRunner's single-request numerator and counter schema."""
        work = request["workload"]
        elapsed = {evaluator.Role.AR: 100, evaluator.Role.STOCK: 50, evaluator.Role.CANDIDATE: 40}[role]
        counters = dict(zip(evaluator.COUNTERS, (0, 0, 0, 0, 0) if role is evaluator.Role.AR else (8, 4, 4, 2, 16)))
        prompt = (work["prompt_unit"] * work["prompt_repetitions"]).encode()
        return {"schema": "llaminar.benchmark.v1", "success": True,
                "prefill_success": True, "decode_success": True,
                "measurement_iterations": 3, "warmup_iterations": 1,
                "prompt": {"sha256": hashlib.sha256(prompt).hexdigest(), "bytes": len(prompt)},
                "config": {**self.plan_document["implementations"][role.value]["config"],
                           "n_predict": 8, "sampling": {key: work[key] for key in ("seed", "temperature", "top_k", "top_p")},
                           "model_path": "/fixture/model.gguf", "benchmark_json_output_path": "fixture.json"},
                "iterations": [{"iteration": index, "tokens": {"prefill": 16, "decode": 8, "decode_after_prefill": 7},
                                "timing_ms": {"prefill": 10, "decode": elapsed * scale},
                                "throughput_tokens_per_sec": {"prefill": 1600, "decode_after_prefill": 7000 / (elapsed * scale)},
                                "generated_token_ids": list(range(8)), "mtp": counters.copy()}
                               for index, scale in enumerate((0.98, 1, 1.02), 1)],
                "mtp": {**{key: value * 3 for key, value in counters.items()},
                        "config_enabled": role is not evaluator.Role.AR, "bypassed": False,
                        "transaction_validation_failures": 0, "unsafe_verifier_state_rejections": 0},
                "perf_stats": {"schema": "llaminar.perf_stats.v1", "enabled": True, "records": [
                    {"kind": "timer", "domain": "stage_gpu", "name": name, "phase": "decode", "device": "cuda:0",
                     "tags": {"context": "fixture"}, "count": 3, "total_ns": 1_000_000_000}
                    for name in ("parent", "overlapping_child")]}}

    def mutate_benchmark(self, change, index=2):
        """Repin changed fixture evidence, exercising semantic rather than hash checks."""
        observation = self.evidence["observations"][index]
        reference = observation["benchmark"]
        data = json.loads((self.directory / reference["path"]).read_text())
        change(data)
        observation["benchmark"] = self.save(reference["path"], data)
        self.save("evidence.json", self.evidence)

    def evaluate(self):
        """Invoke the public consumer with the currently persisted evidence."""
        self.save("evidence.json", self.evidence)
        return evaluator.evaluate(self.plan, self.evidence_path)

    def test_frozen_plan_views_cannot_mutate_validated_intent(self):
        """Consumer-owned views cannot change the meaning of retained digests."""
        original_document = copy.deepcopy(self.plan.document)
        original_requests = copy.deepcopy(self.plan.requests)
        original_identity = copy.deepcopy(self.plan.identity())
        mutations = (
            lambda plan: plan.document["implementations"]["candidate"]["config"].update(mtp_enabled=False),
            lambda plan: plan.requests["held_out"]["workload"].update(seed=123),
            lambda plan: plan.builds.update(candidate="0" * 64),
            lambda plan: plan.identity()["build_sha256"].update(stock="0" * 64),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(view=index):
                self.plan = evaluator.load_plan(self.plan_path, original_identity["plan_sha256"])
                mutate(self.plan)
                self.assertEqual(self.plan.document, original_document)
                self.assertEqual(self.plan.requests, original_requests)
                self.assertEqual(self.plan.identity(), original_identity)
                self.assertEqual(self.evaluate()["status"], "compared")

    def test_metadata_types_are_part_of_frozen_identity(self):
        """Bool/int equality cannot disguise compiler or execution drift."""
        original = copy.deepcopy(self.plan_document)
        for changed in ("compiler", "runtime_libraries", "execution"):
            with self.subTest(changed=changed):
                self.plan_document = copy.deepcopy(original)
                if changed == "execution":
                    for name, entry in self.plan_document["implementations"].items():
                        entry["config"]["future_native_option"] = 1 if name == "candidate" else True
                else:
                    for name, entry in self.plan_document["implementations"].items():
                        build = {**self.build, changed: {"typed_option": 1 if name == "candidate" else True}}
                        entry["build"] = self.save(f"{name}-typed-build.json", build)
                with self.assertRaisesRegex(ValueError, "identity differs|execution policy differs"):
                    self.refresh_plan()

    def test_small_rates_still_require_relative_arithmetic_agreement(self):
        """An absolute epsilon must not admit arbitrarily wrong tiny rates."""
        def change(data):
            for row in data["iterations"]:
                row["timing_ms"]["decode"] = 1e18
                row["throughput_tokens_per_sec"]["decode_after_prefill"] = 1e-9
        self.mutate_benchmark(change)
        with self.assertRaisesRegex(ValueError, "arithmetic disagrees"):
            self.evaluate()

    def test_small_valid_rates_remain_comparable(self):
        """Scale-aware arithmetic retains valid low-throughput measurements."""
        def change(data):
            for row in data["iterations"]:
                row["timing_ms"]["decode"] = 1e18
                row["throughput_tokens_per_sec"]["decode_after_prefill"] = 7000 / 1e18
        self.mutate_benchmark(change)
        self.assertEqual(self.evaluate()["status"], "compared")

    def test_native_integer_boundaries_are_inclusive(self):
        """The parser admits native endpoints without allowing wider JSON ints."""
        for maximum in (evaluator.INT32_MAX, evaluator.UINT64_MAX):
            for value in (0, 1, maximum):
                with self.subTest(value=value, maximum=maximum):
                    self.assertEqual(evaluator.integer(value, "boundary", maximum=maximum), value)
            for value in (-1, maximum + 1, True, 1.5):
                with self.subTest(invalid=value, maximum=maximum), self.assertRaisesRegex(ValueError, "invalid integer"):
                    evaluator.integer(value, "boundary", maximum=maximum)

    def test_native_integer_ranges_reject_unrepresentable_evidence(self):
        """Synthetic JSON integers must fit the producer's native storage."""
        original = copy.deepcopy(self.evidence)
        path = original["observations"][2]["benchmark"]["path"]
        data = json.loads((self.directory / path).read_text())
        for field in ("counter", "token"):
            with self.subTest(field=field):
                self.evidence = copy.deepcopy(original)
                self.save(path, data)
                def change(payload):
                    for row in payload["iterations"]:
                        if field == "counter":
                            row["mtp"]["draft_steps"] = 2**64
                        else:
                            row["generated_token_ids"][0] = 2**31
                    if field == "counter":
                        payload["mtp"]["draft_steps"] = 3 * 2**64
                # Keep all streams identical for the token case: a separate
                # AR mismatch must not mask the native representability gate.
                indices = range(len(self.evidence["observations"])) if field == "token" else (2,)
                for index in indices:
                    self.mutate_benchmark(change, index)
                with self.assertRaisesRegex(ValueError, "invalid integer"):
                    self.evaluate()

    def test_valid_costs_distinguish_ar_stock_and_useful_tokens(self):
        """Whole decode includes all work; draft acceptance is not throughput."""
        result = self.evaluate()
        self.assertFalse(result["certification_eligible"])
        self.assertEqual(result["correctness"]["serial_row_and_state_bytes"], "not_observed")
        summary = result["decode_speedup_geomean_by_split"]["held_out"]
        self.assertAlmostEqual(summary["stock_vs_ar"], 2)
        self.assertAlmostEqual(summary["candidate_vs_ar"], 2.5)
        self.assertAlmostEqual(summary["candidate_vs_stock"], 1.25)
        candidate = result["requests"][0]["blocks"][0]["roles"]["candidate"]
        self.assertAlmostEqual(candidate["decode_ms_per_emitted_token"], 40 / 7)
        self.assertAlmostEqual(candidate["mtp"]["whole_decode_ms_per_verifier_invocation"], 20)
        self.assertEqual(candidate["mtp"]["emitted_tokens_per_verifier_invocation"], 3.5)
        self.assertEqual(candidate["mtp"]["accepted_fraction"], 0.5)
        self.assertEqual(len(candidate["timers_nonadditive"]), 2)
        self.assertNotIn("output", candidate)
        self.assertNotIn("headroom_speedup", result)

    def test_higher_acceptance_can_still_be_slower(self):
        """A costly candidate is reported honestly, never promoted by acceptance."""
        for index in range(2, len(self.evidence["observations"]), 3):
            def change(data):
                for row in data["iterations"]:
                    row["timing_ms"]["decode"] *= 2
                    row["throughput_tokens_per_sec"]["decode_after_prefill"] /= 2
                    row["mtp"]["accepted_tokens"] = 7
                    row["mtp"]["rejected_tokens"] = 1
                data["mtp"].update(accepted_tokens=21, rejected_tokens=3)
            self.mutate_benchmark(change, index)
        result = self.evaluate()
        self.assertAlmostEqual(result["decode_speedup_geomean_by_split"]["held_out"]["candidate_vs_stock"], 0.625)

    def test_token_mismatch_fails_even_with_faster_timing(self):
        """A consistent wrong candidate stream still fails the exact AR control."""
        self.mutate_benchmark(lambda data: [row["generated_token_ids"].__setitem__(3, 99) for row in data["iterations"]])
        with self.assertRaisesRegex(ValueError, "exact AR token mismatch"):
            self.evaluate()

    def test_repeat_token_drift_fails(self):
        """A changed stochastic repeat cannot hide behind a matching first row."""
        self.mutate_benchmark(lambda data: data["iterations"][0]["generated_token_ids"].__setitem__(0, 99))
        with self.assertRaisesRegex(ValueError, "seeded output changed"):
            self.evaluate()

    def test_missing_duplicate_extra_or_reused_observations_fail(self):
        """The complete planned Cartesian product is mandatory and unique."""
        original = copy.deepcopy(self.evidence)
        for mutation, diagnostic in (
            (lambda rows: rows.pop(), "incomplete evidence"),
            (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate observation"),
            (lambda rows: rows[0].update(pair="unplanned"), "unexpected/duplicate"),
            (lambda rows: rows[3].update(benchmark=rows[0]["benchmark"]), "reused benchmark"),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.evidence = copy.deepcopy(original)
                mutation(self.evidence["observations"])
                with self.assertRaisesRegex(ValueError, diagnostic):
                    self.evaluate()

    def test_profiled_stale_build_or_different_deployment_fail(self):
        """A valid payload cannot waive its collector-bound provenance."""
        original = copy.deepcopy(self.evidence)
        for field, value, diagnostic in (
            ("measurement", "profiled", "profiled timings"),
            ("build_sha256", "0" * 64, "implementation identity"),
            ("deployment_sha256", "0" * 64, "deployment identity"),
        ):
            with self.subTest(field=field):
                self.evidence = copy.deepcopy(original)
                self.evidence["observations"][0][field] = value
                with self.assertRaisesRegex(ValueError, diagnostic):
                    self.evaluate()

    def test_changed_artifact_or_plan_bytes_fail(self):
        """Even semantically harmless byte changes invalidate external pins."""
        reference = self.evidence["observations"][0]["benchmark"]
        with (self.directory / reference["path"]).open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
            self.evaluate()
        with self.plan_path.open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
            evaluator.load_plan(self.plan_path, self.plan.sha256)

    def test_evidence_plan_mismatch_fails(self):
        """An old collection cannot be reinterpreted under new evaluation intent."""
        self.evidence["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.evaluate()

    def test_leaked_held_out_prompt_even_with_changed_seed_fails(self):
        """Seed changes do not turn a seen prompt into held-out evidence."""
        held = self.plan_document["requests"][1]["workload"]
        held.update(prompt_unit="development", seed=123)
        with self.assertRaisesRegex(ValueError, "prompt leakage"):
            self.refresh_plan()

    def test_release_build_compiler_and_ar_control_are_matched(self):
        """Reject mixed build modes, toolchains, ISAs and AR implementations."""
        original = copy.deepcopy(self.plan_document)
        for key, value, diagnostic in (
            ("build_type", "Integration", "only Release"),
            ("compiler", {"cxx": "changed"}, "compiler/ISA/runtime"),
            ("cpu_isa", "AVX2", "compiler/ISA/runtime"),
            ("runtime_libraries", {"openmp": "changed"}, "compiler/ISA/runtime"),
        ):
            with self.subTest(key=key):
                self.plan_document = copy.deepcopy(original)
                self.plan_document["implementations"]["candidate"]["build"] = self.save("changed-build.json", {**self.build, key: value})
                with self.assertRaisesRegex(ValueError, diagnostic):
                    self.refresh_plan()
        self.plan_document = copy.deepcopy(original)
        self.plan_document["implementations"]["ar"]["build"] = self.save("changed-build.json", {**self.build, "binary_sha256": "c" * 64})
        with self.assertRaisesRegex(ValueError, "identical build"):
            self.refresh_plan()

    def test_non_mtp_policy_drift_and_incomplete_plan_fail(self):
        """Evaluation intent cannot omit controls or change unrelated policy."""
        original = copy.deepcopy(self.plan_document)
        for change, diagnostic in (
            (lambda plan: plan["implementations"]["candidate"]["config"].update(prefix_cache_enabled=False), "non-MTP"),
            (lambda plan: plan.update(pairs=["a", "a", "b"]), "unique paired"),
            (lambda plan: plan["requests"].pop(), "disjoint development"),
            (lambda plan: plan["implementations"].pop("ar"), "name ar, stock and candidate"),
            (lambda plan: plan["deployment"].pop("model"), "deployment needs"),
        ):
            self.plan_document = copy.deepcopy(original)
            change(self.plan_document)
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, diagnostic):
                self.refresh_plan()

    def test_failed_partial_or_incompatible_native_results_fail(self):
        """Repinned but malformed native data fails its semantic contract."""
        original = copy.deepcopy(self.evidence)
        path = original["observations"][2]["benchmark"]["path"]
        data = json.loads((self.directory / path).read_text())
        cases = [
            (lambda d: d.update(success=False), "did not complete"),
            (lambda d: d.update(schema="old"), "did not complete"),
            (lambda d: d.update(warmup_iterations=0), "omitted warmup"),
            (lambda d: d["iterations"].pop(), "omitted warmup"),
            (lambda d: d["iterations"][0]["tokens"].update(decode=7), "truncated"),
            (lambda d: d["iterations"][0]["tokens"].update(decode_after_prefill=8), "terminal-prefill"),
            (lambda d: d["iterations"][0].update(iteration=2), "iteration order"),
            (lambda d: d["iterations"][0]["generated_token_ids"].pop(), "truncated generated"),
            (lambda d: d["config"]["sampling"].update(seed=99), "sampler differs"),
            (lambda d: d["config"].update(mtp_draft_tokens=15), "configuration differs"),
            (lambda d: d["config"].update(mtp_max_request_batch=True), "configuration differs"),
            (lambda d: d["config"]["sampling"].update(seed=True), "invalid integer"),
            (lambda d: d["mtp"].update(bypassed=True), "bypassed"),
            (lambda d: d["mtp"].update(transaction_validation_failures=1), "transaction_validation"),
            (lambda d: d["mtp"].update(verifier_runs=99), "aggregate includes"),
            (lambda d: d["iterations"][0]["mtp"].update(verifier_token_count=True), "invalid integer"),
            (lambda d: d["iterations"][0]["timing_ms"].update(decode=0), "finite positive"),
            (lambda d: d["iterations"][0]["timing_ms"].update(decode=1), "arithmetic disagrees"),
        ]
        for change, diagnostic in cases:
            self.evidence = copy.deepcopy(original)
            self.save(path, data)
            self.mutate_benchmark(change)
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, diagnostic):
                self.evaluate()

    def test_disabled_perfstats_is_unknown_not_zero_headroom(self):
        """Missing timing observations must not become a zero-cost estimate."""
        self.mutate_benchmark(lambda data: data["perf_stats"].update(enabled=False, records=[]))
        result = self.evaluate()
        self.assertEqual(result["requests"][0]["blocks"][0]["roles"]["candidate"]["timers_nonadditive"], [])
        self.assertIn("kernel_or_overhead", result["unobserved_headroom"])

    def test_duplicate_json_keys_and_nonfinite_values_fail(self):
        """Ambiguous or non-standard JSON is never an authenticated document."""
        path = self.directory / "invalid.json"
        for payload in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            path.write_text(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                evaluator.read_document(path)

    def test_cross_block_control_drift_is_not_a_valid_pair(self):
        """Every block must reproduce the original same-build AR token stream."""
        self.mutate_benchmark(lambda data: [row["generated_token_ids"].__setitem__(0, 99)
                                            for row in data["iterations"]], index=3)
        with self.assertRaisesRegex(ValueError, "exact AR token mismatch"):
            self.evaluate()

    def test_duplicate_workload_cannot_reweight_geomean(self):
        """Renamed duplicate requests cannot overweight a favorable sample."""
        duplicate = copy.deepcopy(self.plan_document["requests"][0])
        duplicate["id"] = "renamed"
        self.plan_document["requests"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate request workload"):
            self.refresh_plan()

    def test_numeric_workload_spelling_cannot_reweight_geomean(self):
        """JSON 1 and 1.0 temperatures describe the same request policy."""
        self.plan_document["requests"][0]["workload"]["temperature"] = 1
        duplicate = copy.deepcopy(self.plan_document["requests"][0])
        duplicate["id"] = "float-temperature"
        duplicate["workload"]["temperature"] = 1.0
        self.plan_document["requests"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate request workload"):
            self.refresh_plan()

    def test_mtp_without_actual_verifier_work_fails(self):
        """Configured MTP is not evidence that the speculative path executed."""
        def change(data):
            for row in data["iterations"]:
                row["mtp"] = dict.fromkeys(evaluator.COUNTERS, 0)
            data["mtp"].update(dict.fromkeys(evaluator.COUNTERS, 0))
        self.mutate_benchmark(change)
        with self.assertRaisesRegex(ValueError, "selected AR/MTP path"):
            self.evaluate()

    def test_split_ratios_are_independent_and_full_cycle_cost_is_visible(self):
        """Held-out regressions and prefill costs cannot disappear in a global win."""
        for index in range(11, 18, 3):
            def change(data):
                for row in data["iterations"]:
                    row["timing_ms"]["decode"] *= 2
                    row["throughput_tokens_per_sec"]["decode_after_prefill"] /= 2
                    row["timing_ms"]["prefill"] *= 4
                    row["throughput_tokens_per_sec"]["prefill"] /= 4
            self.mutate_benchmark(change, index)
        report = self.evaluate()
        self.assertAlmostEqual(report["decode_speedup_geomean_by_split"]["development"]["candidate_vs_stock"], 1.25)
        self.assertAlmostEqual(report["decode_speedup_geomean_by_split"]["held_out"]["candidate_vs_stock"], 0.625)
        full_cycle = report["requests"][1]["speedup"]["candidate_vs_stock"]["prefill_plus_decode_ms"]["median"]
        self.assertAlmostEqual(full_cycle, 0.5)

    def test_cli_success_no_overwrite_and_failure_publish_no_green(self):
        """CLI report creation is distinct from evidence and qualification."""
        report = self.directory / "report.json"
        temporary_sibling = self.directory / "report.json.tmp"
        temporary_sibling.write_text("an existing input, not disposable temporary storage")
        args = ["evaluate", "--plan", str(self.plan_path), "--plan-sha256", self.plan.sha256,
                "--evidence", str(self.evidence_path), "--report", str(report)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evaluator.main(args), 0)
            saved = report.read_bytes()
            self.assertEqual(temporary_sibling.read_text(), "an existing input, not disposable temporary storage")
            self.assertEqual(evaluator.main(args), 1)
            self.assertEqual(saved, report.read_bytes())
            self.evidence["observations"].pop()
            self.save("evidence.json", self.evidence)
            args[-1] = str(self.directory / "failed.json")
            self.assertEqual(evaluator.main(args), 1)
            self.assertFalse((self.directory / "failed.json").exists())
        self.assertFalse(json.loads(saved)["certification_eligible"])

    def test_cli_inspect_and_external_cwd(self):
        """Artifact paths are document-relative, never dependent on cwd."""
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/ci/evaluate_native_speculation.py"),
                                    "inspect", "--plan", str(self.plan_path), "--plan-sha256", self.plan.sha256],
                                   cwd=self.directory, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), self.plan.identity())


if __name__ == "__main__":
    unittest.main()
