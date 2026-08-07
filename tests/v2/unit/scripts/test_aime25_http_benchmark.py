#!/usr/bin/env python3
"""Fast, device-free regressions for the AIME25 HTTP benchmark workflow."""

from __future__ import annotations

import importlib.util
import dataclasses
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "scripts/benchmarks/aime25_http_benchmark.py"
SPEC = importlib.util.spec_from_file_location("aime25_http_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class Aime25HttpBenchmarkTest(unittest.TestCase):
    """Lock strict scoring, selection, resume, and summary semantics."""

    def test_explicit_answer_extraction_never_uses_incidental_numbers(self) -> None:
        self.assertIsNone(benchmark.extract_explicit_answer("I tried 17 and 42."))
        self.assertEqual(
            benchmark.extract_explicit_answer(
                "Maybe \\boxed{111}. After checking:\nAnswer: 007"
            ),
            7,
        )
        self.assertEqual(
            benchmark.extract_explicit_answer("Thus \\boxed{999}."),
            999,
        )

    def test_problem_selection_is_sorted_unique_and_total(self) -> None:
        problems = tuple(
            benchmark.AimeProblem(str(index), f"problem {index}", index)
            for index in range(30)
        )
        self.assertEqual(
            benchmark.selected_problem_ids("5,1-3", problems),
            ("1", "2", "3", "5"),
        )
        self.assertEqual(len(benchmark.selected_problem_ids(None, problems)), 30)
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark.selected_problem_ids("1,1", problems)

    def test_append_only_resume_rejects_duplicate_problem_ids(self) -> None:
        manifest_digest = "sha256:manifest"
        row = {
            "schema_version": benchmark.RESULT_SCHEMA,
            "manifest_digest": manifest_digest,
            "problem_id": "4",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            benchmark.append_jsonl_durable(path, row)
            self.assertEqual(
                set(benchmark.completed_results(path, manifest_digest)), {"4"}
            )
            benchmark.append_jsonl_durable(path, row)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                benchmark.completed_results(path, manifest_digest)

    def test_resume_rejects_results_from_another_manifest(self) -> None:
        row = {
            "schema_version": benchmark.RESULT_SCHEMA,
            "manifest_digest": "sha256:old",
            "problem_id": "4",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            benchmark.append_jsonl_durable(path, row)
            with self.assertRaisesRegex(ValueError, "manifest"):
                benchmark.completed_results(path, "sha256:new")

    def test_review_receives_reasoning_and_content_from_first_turn(self) -> None:
        first = benchmark.ChatReply(
            content="Answer: 007",
            reasoning_content="The complete derivation.",
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            elapsed_seconds=1.0,
        )
        review = benchmark.ChatReply(
            content="Answer: 007",
            reasoning_content="Checked independently.",
            finish_reason="stop",
            usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            elapsed_seconds=1.0,
        )
        calls = []

        def fake_post_chat(_base_url, messages, **_kwargs):
            calls.append(messages)
            return first if len(calls) == 1 else review

        with mock.patch.object(benchmark, "post_chat", side_effect=fake_post_chat):
            result = benchmark.solve_problem(
                "http://127.0.0.1:1",
                benchmark.AimeProblem("0", "Compute it.", 7),
                manifest_digest="sha256:manifest",
                first_max_tokens=32,
                review_max_tokens=32,
                enable_thinking=True,
                temperature=0.0,
                seed=1,
                request_timeout=1.0,
            )

        assistant = calls[1][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertIn("The complete derivation.", assistant["content"])
        self.assertIn("Answer: 007", assistant["content"])
        self.assertEqual(result["manifest_digest"], "sha256:manifest")

    def test_first_turn_checkpoint_is_manifest_and_problem_bound(self) -> None:
        reply = benchmark.ChatReply(
            content="Answer: 007",
            reasoning_content="Derivation.",
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            elapsed_seconds=1.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "turns/0.first.json"
            benchmark.write_first_turn_checkpoint(
                path, "sha256:manifest", "0", reply
            )
            loaded = benchmark.load_first_turn_checkpoint(
                path, "sha256:manifest", "0"
            )
            self.assertEqual(loaded, reply)
            with self.assertRaisesRegex(ValueError, "manifest"):
                benchmark.load_first_turn_checkpoint(
                    path, "sha256:other", "0"
                )
            with self.assertRaisesRegex(ValueError, "problem ID"):
                benchmark.load_first_turn_checkpoint(
                    path, "sha256:manifest", "1"
                )

    def test_history_canary_is_manifest_bound_and_proves_nonce(self) -> None:
        row = {
            "schema_version": benchmark.CANARY_SCHEMA,
            "manifest_digest": "sha256:manifest",
            "observed_answer": 731,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canary.json"
            benchmark.write_json_atomic(path, row)
            self.assertEqual(
                benchmark.validate_history_canary(path, "sha256:manifest"),
                row,
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                benchmark.validate_history_canary(path, "sha256:other")

            row["observed_answer"] = 999
            benchmark.write_json_atomic(path, row)
            with self.assertRaisesRegex(ValueError, "expected nonce"):
                benchmark.validate_history_canary(path, "sha256:manifest")

    def test_prefix_cache_canary_replays_an_identical_http_request(self) -> None:
        """The cache certificate must not rely on chat-history similarity."""

        reply = benchmark.ChatReply(
            content="cache-ready",
            reasoning_content="",
            finish_reason="stop",
            usage={"prompt_tokens": 24, "completion_tokens": 3, "total_tokens": 27},
            elapsed_seconds=0.25,
        )
        calls = []

        def fake_post_chat(_base_url, messages, **kwargs):
            calls.append((json.loads(json.dumps(messages)), dict(kwargs)))
            return reply

        with mock.patch.object(
            benchmark, "post_chat", side_effect=fake_post_chat
        ):
            result = benchmark.run_prefix_cache_restore_canary(
                "http://127.0.0.1:1", 1.0, 17
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(result["first"]["content"], "cache-ready")
        self.assertEqual(result["replay"]["content"], "cache-ready")

    def test_prefix_cache_canary_fails_on_changed_replay_output(self) -> None:
        first = benchmark.ChatReply(
            content="cache-ready",
            reasoning_content="",
            finish_reason="stop",
            usage={"prompt_tokens": 24, "completion_tokens": 3, "total_tokens": 27},
            elapsed_seconds=0.25,
        )
        changed = dataclasses.replace(first, content="changed")
        with (
            mock.patch.object(
                benchmark, "post_chat", side_effect=(first, changed)
            ),
            self.assertRaisesRegex(RuntimeError, "changed deterministic"),
        ):
            benchmark.run_prefix_cache_restore_canary(
                "http://127.0.0.1:1", 1.0, 17
            )

    def test_prefix_cache_canary_artifact_is_manifest_bound(self) -> None:
        reply = {
            "content": "cache-ready",
            "reasoning_content": "",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 24, "completion_tokens": 3},
            "elapsed_seconds": 0.25,
        }
        row = {
            "schema_version": benchmark.PREFIX_CACHE_CANARY_SCHEMA,
            "manifest_digest": "sha256:manifest",
            "first": reply,
            "replay": dict(reply),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix-canary.json"
            benchmark.write_json_atomic(path, row)
            self.assertEqual(
                benchmark.validate_prefix_cache_canary(
                    path, "sha256:manifest"
                ),
                row,
            )
            with self.assertRaisesRegex(ValueError, "another manifest"):
                benchmark.validate_prefix_cache_canary(path, "sha256:other")

            row["replay"] = {**reply, "content": "changed"}
            benchmark.write_json_atomic(path, row)
            with self.assertRaisesRegex(ValueError, "not deterministic"):
                benchmark.validate_prefix_cache_canary(
                    path, "sha256:manifest"
                )

    def test_summary_keeps_first_and_review_accuracy_separate(self) -> None:
        rows = (
            {
                "problem_id": "0",
                "first_correct": False,
                "review_correct": True,
                "first": {
                    "elapsed_seconds": 2.0,
                    "usage": {"completion_tokens": 20},
                },
                "review": {
                    "elapsed_seconds": 1.0,
                    "usage": {"completion_tokens": 10},
                },
            },
            {
                "problem_id": "1",
                "first_correct": True,
                "review_correct": False,
                "first": {
                    "elapsed_seconds": 3.0,
                    "usage": {"completion_tokens": 30},
                },
                "review": {
                    "elapsed_seconds": 4.0,
                    "usage": {"completion_tokens": 40},
                },
            },
        )
        summary = benchmark.make_summary(rows, ("0", "1"))
        self.assertEqual(summary["first_correct"], 1)
        self.assertEqual(summary["review_correct"], 1)
        self.assertEqual(summary["completion_tokens"], 100)
        self.assertAlmostEqual(summary["completion_tokens_per_second"], 10.0)

    def test_manifest_digest_is_key_order_independent(self) -> None:
        self.assertEqual(
            benchmark.mapping_digest({"a": 1, "b": 2}),
            benchmark.mapping_digest({"b": 2, "a": 1}),
        )

    def test_device_parser_is_total_across_supported_backends(self) -> None:
        for device, backend in (
            ("cpu:0", "cpu"),
            ("cuda:1", "cuda"),
            ("rocm:3", "rocm"),
        ):
            with self.subTest(device=device):
                self.assertEqual(
                    benchmark.parse_device_backend(device), backend
                )
        for device in ("", "gpu:0", "cuda", "rocm:-1", "cpu:x"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "cpu:N"):
                    benchmark.parse_device_backend(device)

    def test_managed_mtp_depth_three_and_prefix_policy_are_explicit(self) -> None:
        arguments = benchmark.managed_server_arguments(3, True)
        self.assertEqual(
            arguments,
            (
                "--mtp",
                "--mtp-draft-tokens", "3",
                "--mtp-depth-policy", "fixed",
                "--mtp-verify-mode", "speculative-sampling",
                "--prefix-cache",
                "--prefix-cache-storage", "ram",
                "--prefix-cache-ram-budget-mb", "1024",
                "--prefix-cache-terminal-state", "auto",
                "--prefix-cache-moe-policy", "placement-fingerprint",
            ),
        )

    def test_managed_feature_policy_rejects_conflicting_raw_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "owned by --mtp-depth"):
            benchmark.reject_managed_server_argument_conflicts(
                ("--mtp-draft-tokens=4",),
                3,
                False,
            )
        with self.assertRaisesRegex(ValueError, "owned by --prefix-cache"):
            benchmark.reject_managed_server_argument_conflicts(
                ("--prefix-cache-storage", "device"),
                None,
                True,
            )

    def test_release_manifest_can_bind_resolved_core_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llaminar2"
            core = root / "libllaminar2_core.so"
            binary.write_bytes(b"binary")
            core.write_bytes(b"core")
            completed = benchmark.subprocess.CompletedProcess(
                ("ldd", str(binary)),
                0,
                stdout=(
                    f"libllaminar2_core.so => {core} (0x0001)\n"
                    "libc.so.6 => /lib/libc.so.6 (0x0002)\n"
                ),
            )
            with mock.patch.object(
                benchmark.subprocess, "run", return_value=completed
            ):
                self.assertEqual(
                    benchmark.resolve_linked_library(
                        binary, "libllaminar2_core.so"
                    ),
                    core.resolve(),
                )

    def test_server_owning_evidence_requires_release_build_tree(self) -> None:
        for build_type, accepted in (("Release", True), ("Integration", False)):
            with (
                self.subTest(build_type=build_type),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                binary = root / "tests/v2/llaminar2"
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"binary")
                (root / "CMakeCache.txt").write_text(
                    f"CMAKE_BUILD_TYPE:STRING={build_type}\n",
                    encoding="utf-8",
                )
                if accepted:
                    self.assertEqual(
                        benchmark.require_release_build(binary), "Release"
                    )
                else:
                    with self.assertRaisesRegex(
                        ValueError, "requires a Release"
                    ):
                        benchmark.require_release_build(binary)

    def test_manifest_records_only_execution_relevant_inherited_knobs(self) -> None:
        selected = benchmark.inherited_runtime_environment({
            "PATH": "/bin",
            "HOME": "/tmp/home",
            "LLAMINAR_GPU_GRAPHS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_DEBUG": "WARN",
            "OMP_NUM_THREADS": "8",
        })
        self.assertEqual(
            selected,
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "LLAMINAR_GPU_GRAPHS": "1",
                "NCCL_DEBUG": "WARN",
                "OMP_NUM_THREADS": "8",
            },
        )

    def test_gpu_execution_contract_requires_monolithic_capture_and_replay(self) -> None:
        records = [
            {
                "domain": "forward_graph",
                "name": "decode_graph_phase",
                "device": "ROCm:0",
                "value": 1,
                "tags": {"phase": "capture"},
            },
            {
                "domain": "forward_graph",
                "name": "decode_graph_phase",
                "device": "ROCm:0",
                "value": 10,
                "tags": {"phase": "replay"},
            },
            {
                "domain": "forward_graph",
                "name": "prefill_graph_phase",
                "device": "ROCm:0",
                "value": 1,
                "tags": {"capture_phase": "capture"},
            },
            {
                "domain": "forward_graph",
                "name": "prefill_graph_phase",
                "device": "ROCm:0",
                "value": 4,
                "tags": {"capture_phase": "replay"},
            },
            {
                "domain": "forward_graph",
                "name": "decode_capture_policy",
                "device": "ROCm:0",
                "value": 11,
                "tags": {
                    "allow_graph_replay": "true",
                    "collective_segmented": "false",
                    "replay_plan_policy": "require_full_graph",
                },
            },
            {
                "domain": "forward_graph",
                "name": "durable_output_publications",
                "phase": "decode",
                "device": "ROCm:0",
                "value": 11,
                "tags": {},
            },
            {
                "domain": "forward_graph",
                "name": "durable_output_publications",
                "phase": "prefill",
                "device": "ROCm:0",
                "value": 5,
                "tags": {},
            },
            {
                "domain": "transfer",
                "name": "d2h_bytes",
                "device": "ROCm:0",
                "value": 4,
                "tags": {"boundary": "terminal_result"},
            },
            {
                "domain": "mtp",
                "name": "decode_transaction_depth_selections",
                "device": "ROCm:0",
                "value": 2,
                "tags": {
                    "depth_policy": "fixed",
                    "requested_depth": "3",
                    "capture_depth": "3",
                },
            },
            {
                "domain": "mtp",
                "name": "device_resident_generation_requests",
                "device": "ROCm:0",
                "value": 2,
                "tags": {
                    "path": "device_resident_generation_loop",
                    "depth": "3",
                    "capture_depth": "3",
                    "final_depth": "3",
                    "transactions": "5",
                    "state_commits": "5",
                },
            },
            {
                "domain": "mtp",
                "name": "verifier_runs",
                "device": "ROCm:0",
                "value": 5,
                "tags": {
                    "path": "device_resident_generation_loop",
                    "depth": "3",
                },
            },
            {
                "domain": "prefix_cache",
                "name": "harvest_inserts",
                "device": "ROCm:0",
                "value": 4,
                "tags": {},
            },
            {
                "domain": "prefix_cache",
                "name": "lookup_results",
                "device": "ROCm:0",
                "value": 1,
                "tags": {
                    "hit_type": "partial",
                    "cached_tokens": "64",
                },
            },
            {
                "domain": "prefix_cache",
                "name": "populate_restores",
                "device": "ROCm:0",
                "value": 1,
                "tags": {"includes_mtp_state": "true"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perfstats.json"
            path.write_text(json.dumps({
                "schema": "llaminar.perf_stats.v1",
                "records": records,
            }), encoding="utf-8")
            report = benchmark.validate_execution_contract(
                path,
                "rocm:0",
                required_mtp_depth=3,
                require_prefix_cache=True,
            )
            self.assertEqual(
                report["schema_version"],
                benchmark.EXECUTION_CONTRACT_SCHEMA,
            )
            self.assertEqual(report["prefill_replay"], 4.0)
            self.assertEqual(report["required_mtp_depth"], 3)
            self.assertEqual(report["prefix_cache_restores"], 1.0)

    def test_execution_contract_rejects_wrong_mtp_depth_and_missing_prefix_hit(self) -> None:
        records = [
            {
                "domain": "mtp",
                "name": "decode_transaction_depth_selections",
                "device": "CPU:0",
                "value": 1,
                "tags": {
                    "depth_policy": "fixed",
                    "requested_depth": "2",
                    "capture_depth": "2",
                },
            },
            {
                "domain": "mtp",
                "name": "verifier_runs",
                "device": "CPU:0",
                "value": 1,
                "tags": {"path": "device_resident_generation_loop"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perfstats.json"
            path.write_text(json.dumps({
                "schema": "llaminar.perf_stats.v1",
                "records": records,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed depth 3"):
                benchmark.validate_execution_contract(
                    path,
                    "cpu:0",
                    required_mtp_depth=3,
                )

            records[0]["tags"].update({
                "requested_depth": "3",
                "capture_depth": "3",
            })
            path.write_text(json.dumps({
                "schema": "llaminar.perf_stats.v1",
                "records": records,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix-cache population"):
                benchmark.validate_execution_contract(
                    path,
                    "cpu:0",
                    required_mtp_depth=3,
                    require_prefix_cache=True,
                )

    def test_gpu_execution_contract_rejects_segmentation_and_intermediate_d2h(self) -> None:
        base_records = [
            {
                "domain": "forward_graph",
                "name": "decode_graph_phase",
                "device": "CUDA:0",
                "value": 1,
                "tags": {"phase": "capture"},
            },
            {
                "domain": "forward_graph",
                "name": "decode_graph_phase",
                "device": "CUDA:0",
                "value": 1,
                "tags": {"phase": "replay"},
            },
            {
                "domain": "forward_graph",
                "name": "prefill_graph_phase",
                "device": "CUDA:0",
                "value": 1,
                "tags": {"capture_phase": "capture"},
            },
            {
                "domain": "forward_graph",
                "name": "prefill_graph_phase",
                "device": "CUDA:0",
                "value": 1,
                "tags": {"capture_phase": "replay"},
            },
            {
                "domain": "forward_graph",
                "name": "decode_capture_policy",
                "device": "CUDA:0",
                "value": 2,
                "tags": {
                    "allow_graph_replay": "true",
                    "collective_segmented": "false",
                    "replay_plan_policy": "require_full_graph",
                },
            },
            {
                "domain": "forward_graph",
                "name": "durable_output_publications",
                "phase": "decode",
                "device": "CUDA:0",
                "value": 2,
                "tags": {},
            },
            {
                "domain": "forward_graph",
                "name": "durable_output_publications",
                "phase": "prefill",
                "device": "CUDA:0",
                "value": 2,
                "tags": {},
            },
        ]
        violations = (
            ({
                "domain": "forward_graph",
                "name": "decode_capture_policy",
                "device": "gpu:0",
                "value": 1,
                "tags": {"collective_segmented": "true"},
            }, "segmented"),
            ({
                "domain": "mtp",
                "name": "intermediate_state_d2h_sync",
                "device": "cuda:0",
                "value": 4,
                "tags": {"boundary": "intermediate_state"},
            }, "intermediate D2H"),
        )
        for violation, expected in violations:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "perfstats.json"
                path.write_text(json.dumps({
                    "schema": "llaminar.perf_stats.v1",
                    "records": [*base_records, violation],
                }), encoding="utf-8")
                expected_pattern = (
                    "intermediate device-to-host"
                    if expected == "intermediate D2H" else expected
                )
                with self.assertRaisesRegex(ValueError, expected_pattern):
                    benchmark.validate_execution_contract(path, "cuda:0")


if __name__ == "__main__":
    unittest.main()
