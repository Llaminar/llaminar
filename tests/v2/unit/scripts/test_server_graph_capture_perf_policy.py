#!/usr/bin/env python3
"""Device-free regressions for the production HTTP evidence contract.

Synthetic records test rank/graph identity, ownership and completed runtime
features without loading models or occupying accelerators. A shell request
observer also executes the prefix scenario to prove exact seed/replay identity
and preservation of the different-answer check independently of HTTP serving.
"""

from __future__ import annotations

import re
import json
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SERVER_E2E_DIR = (
    Path(__file__).resolve().parents[2] / "e2e" / "server"
)
REPO_ROOT = SERVER_E2E_DIR.parents[3]
sys.path.insert(0, str(SERVER_E2E_DIR))

from graph_capture_perf_policy import (  # noqa: E402
    DecodeGraphRequirement,
    _incomplete_graph_contexts,
    _missing_prefill_phases,
    device_kinds_for_cell,
    validate_graph_capture_policy,
)
from ranked_perf_artifacts import (  # noqa: E402
    collect_ranked_perf_stats,
    collect_and_publish_ranked_perf_stats,
    publish_ranked_perf_stats,
    validate_memory_authority,
)
from moe_route_scratch_perf_policy import validate_moe_route_scratch_policy  # noqa: E402
from flash_attention_perf_policy import (  # noqa: E402
    attention_device_kinds_for_cell,
    validate_cpu_flash_attention_execution_policy,
    validate_flash_attention_plan_policy,
)
from gpu_host_transfer_perf_policy import (  # noqa: E402
    DEVICE_GENERATION_DISPATCH_TICKET_ABI_VERSION,
    DEVICE_GENERATION_DISPATCH_TICKET_BYTES,
    DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT,
    validate_gpu_host_transfer_policy,
)
from llep_verifier_perf_policy import (  # noqa: E402
    validate_llep_verifier_policy,
)
from mtp_device_generation_perf_policy import (  # noqa: E402
    validate_cuda_dynamic_mtp_device_generation_policy,
    validate_rocm_host_scheduled_mtp_device_generation_policy,
)
from request_input_lifetime_perf_policy import (  # noqa: E402
    validate_request_input_lifetime_policy,
)
from runtime_feature_perf_policy import MovementEvidence, validate_runtime_feature_policy  # noqa: E402


def counter(
    name: str,
    *,
    value: float = 1.0,
    domain: str = "forward_graph",
    tags: dict[str, str] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Build the minimal PerfStats counter shape consumed by the validator."""

    record: dict[str, object] = {
        "name": name,
        "domain": domain,
        "value": value,
        "tags": tags or {},
    }
    if device is not None:
        record["device"] = device
    return record


def host_movement_records() -> list[dict]:
    """One published wave with bounded, independently authored ordering proof."""
    scope = {"rank": 0, "device": "priority0/priority1", "phase": "maintenance"}
    tags = {"purpose": "placement_change"}
    sequence = {"kind": "ordered_sequence", "count": 1, "sequence_word_count": 2,
                "sequence_digest_lo": 123, "sequence_digest_hi": 456}
    return [counter("expert_migration_edges", domain="moe_overlay_residency", value=2) | scope,
            counter("placement_published_payload_bytes", domain="moe_overlay_residency",
                    value=4096, tags=tags) | scope,
            *[counter(name, domain="moe_overlay_residency", tags=tags) | scope | sequence
              for name in ("placement_transport_publications", "placement_owner_publications")]]


def device_overlay_movement_records() -> list[dict]:
    """One completed all-GPU wave; unrelated rank/transaction rows cannot join."""
    scope = {"rank": 0, "device": "priority0/priority1", "phase": "maintenance"}
    tags = {"transaction": "3", "candidate_epoch": "4", "policy_owner": "device"}
    return [counter(name, domain="moe_overlay_controller", tags=tags) | scope
            for name in ("dynamic_movement_transactions", "dynamic_migration_edges", "dynamic_physical_bytes")]


def device_generation_ticket_tags() -> dict[str, str]:
    """Build the reviewed cross-language immutable-ticket ABI tags."""

    return {
        "bytes": str(DEVICE_GENERATION_DISPATCH_TICKET_BYTES),
        "abi_version": str(DEVICE_GENERATION_DISPATCH_TICKET_ABI_VERSION),
        "word_count": str(DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT),
        "authority": "immutable_scheduler_snapshot",
        "state_payload": "false",
    }


class TestServerGraphCapturePerfPolicy(unittest.TestCase):
    """Prove topology and PerfStats jointly control production certification."""

    def test_log_scan_preserves_failures_without_broken_pipe(self) -> None:
        """Large diagnostics stay bounded without aborting the final certificate."""
        script = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        body = script.split("scan_server_log() {", 1)[1].split("\n}\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            for warnings, mpi_crash in ((0, False), (1, False), (199, True)):
                with self.subTest(warnings=warnings, mpi_crash=mpi_crash):
                    # Exceed pipe capacity, as the real per-lane timeout report
                    # did. head must not SIGPIPE the producer under pipefail.
                    path.write_text(("[WARN ] " + "pending " * 100 + "\n") * warnings
                                    + ("mpirun detected that one or more processes exited with non-zero status\n"
                                       "Exit code: 17\n" if mpi_crash else ""))
                    command = ("set -euo pipefail\nRED= NC=\nfailures=0\n"
                               "fail() { failures=$((failures+1)); printf 'FAIL %s\\n' \"$*\"; }\n"
                               "pass() { printf 'PASS %s\\n' \"$*\"; }\n"
                               "scan_server_log() {" + body + "\n}\n"
                               "scan_server_log test " + shlex.quote(str(path))
                               + "\nprintf 'COMPLETE failures=%s\\n' \"$failures\"\n"
                               "test \"$failures\" -eq 0\n")
                    result = subprocess.run(["bash", "-c", command],
                                            capture_output=True, text=True, timeout=10)
                    expected = int(warnings > 0) + int(mpi_crash)
                    self.assertEqual(result.returncode, int(expected > 0), result.stderr)
                    self.assertIn(f"COMPLETE failures={expected}", result.stdout)
                    if warnings:
                        self.assertIn(f"{warnings} WARN/ERROR entries", result.stdout)
                        self.assertEqual(result.stdout.count("[WARN ]"), min(warnings, 40))
                    if mpi_crash:
                        self.assertIn("child process crashed (exit code: 17)", result.stdout)

    def test_server_disposal_cannot_restore_a_discarded_model(self) -> None:
        """The HTTP surface exports no reusable model contract at shutdown."""
        for name in ("prepared_context_restore_movement_waves", "prepared_context_restore_certifications"):
            for device in ("cuda:0", "rocm:0"):
                with self.subTest(name=name, device=device):
                    record = counter(name, domain="moe_overlay_controller") | {"device": device}
                    self.assertIn("without a retained model", validate_runtime_feature_policy(
                        [record], "", MovementEvidence.NOT_APPLICABLE) or "")
                    self.assertIsNone(validate_runtime_feature_policy(
                        [record | {"value": 0}], "", MovementEvidence.NOT_APPLICABLE))
        host = counter("prepared_context_restoration_edges", domain="moe_overlay_residency")
        self.assertIn("without a retained model", validate_runtime_feature_policy(
            [host], "", MovementEvidence.NOT_APPLICABLE) or "")
        self.assertIsNone(validate_runtime_feature_policy([host | {"value": 0}], "", MovementEvidence.NOT_APPLICABLE))

    def test_runtime_features_require_execution_not_lookup_or_domain_presence(self) -> None:
        """A cache hit or arbitrary MTP counter is not a completed feature."""
        records = [counter("harvest_inserts", domain="prefix_cache"),
                   counter("block_hits", domain="prefix_cache"),
                   counter("sidecar_graph_cache_hits", domain="mtp")]
        self.assertIn("actual restore", validate_runtime_feature_policy(records, "--prefix-cache --mtp", MovementEvidence.NOT_APPLICABLE) or "")
        self.assertIn("attempted and accepted", validate_runtime_feature_policy(records, "--mtp", MovementEvidence.NOT_APPLICABLE) or "")

    def test_runtime_features_require_mtp_state_in_prefix_restore(self) -> None:
        """Restoring only ordinary KV cannot certify a speculative request."""
        records = [counter("harvest_inserts", domain="prefix_cache"),
                   counter("populate_restores", domain="prefix_cache",
                           tags={"includes_mtp_state": "false"})]
        self.assertIn("MTP-bearing", validate_runtime_feature_policy(records, "--prefix-cache --mtp", MovementEvidence.NOT_APPLICABLE) or "")

    def test_runtime_features_accept_completed_host_and_device_operations(self) -> None:
        """All backend paths owe restores, accepted drafts and physical commits."""
        common = [counter("harvest_inserts", domain="prefix_cache"),
                  counter("populate_restores", domain="prefix_cache", tags={"includes_mtp_state": "true"}),
                  counter("accepted_tokens", domain="mtp"),
                  counter("depth_policy_windows", domain="mtp")]
        host = [counter("draft_steps", domain="mtp"), *host_movement_records()]
        native = [counter("device_generation_terminal_attempted_draft_tokens", domain="mtp"),
                  counter("device_rebalance_request_copied_payload_lower_bound", domain="moe_rebalance"),
                  counter("device_rebalance_request_applied_payload_lower_bound", domain="moe_rebalance"),
                  counter("device_rebalance_request_useful_payload_bytes_lower_bound", domain="moe_rebalance")]
        for backend, completed in (("cpu", host), ("cuda", native), ("rocm", native)):
            with self.subTest(backend=backend):
                records = [record | {"device": backend + ":0"} for record in common + completed]
                self.assertIsNone(validate_runtime_feature_policy(iter(records),
                    "--prefix-cache --mtp --mtp-depth-policy=dynamic --moe-residency-maintenance dynamic", MovementEvidence.REQUIRED))

    def test_runtime_features_reject_unexecuted_dynamic_depth(self) -> None:
        """Fixed-depth activity cannot stand in for an adaptive controller."""
        records = [counter("draft_steps", domain="mtp"), counter("accepted_tokens", domain="mtp")]
        self.assertIn("depth-controller", validate_runtime_feature_policy(records, "--mtp --mtp-depth-policy dynamic", MovementEvidence.NOT_APPLICABLE) or "")

    def test_host_movement_needs_published_bytes_and_matching_bounded_sequences(self) -> None:
        """Committed edges, calibration copies and mismatched publications fail."""
        records = host_movement_records()
        self.assertIsNone(validate_runtime_feature_policy(records, "", MovementEvidence.REQUIRED))
        for omitted in range(len(records)):
            with self.subTest(omitted=omitted):
                self.assertIsNotNone(validate_runtime_feature_policy(
                    records[:omitted] + records[omitted + 1:], "", MovementEvidence.REQUIRED))
        for field, value in (("rank", 1), ("device", "different-tier"), ("phase", "prefill"),
                             ("count", 2), ("sequence_word_count", 4),
                             ("sequence_digest_lo", 999), ("sequence_digest_hi", 999),
                             ("kind", "counter"), ("tags", {"purpose": "economy_calibration"})):
            with self.subTest(field=field):
                broken = [*records[:-1], records[-1] | {field: value}]
                self.assertIsNotNone(validate_runtime_feature_policy(broken, "", MovementEvidence.REQUIRED))
        for purpose in ("economy_calibration", "prepared_context_restoration", None):
            broken = [records[0], *[r | {"tags": {"purpose": purpose}} for r in records[1:]]]
            self.assertIsNotNone(validate_runtime_feature_policy(broken, "", MovementEvidence.REQUIRED))
        for value in (0, -1, True, float("nan"), float("inf"), "invalid"):
            broken = [records[0], records[1] | {"value": value}, *records[2:]]
            self.assertIsNotNone(validate_runtime_feature_policy(broken, "", MovementEvidence.REQUIRED))

    def test_static_rejects_completed_placement_payload_without_an_owner_commit(self) -> None:
        """Actual copies are forbidden even if publication later aborts."""
        for name in ("placement_transfer_operations_completed", "placement_transfer_payload_bytes_completed",
                     "placement_published_payload_bytes"):
            with self.subTest(name=name):
                record = counter(name, domain="moe_overlay_residency", tags={"purpose": "placement_change"})
                self.assertIn("static", validate_runtime_feature_policy([record], "", MovementEvidence.FORBIDDEN) or "")
                # A topology probe is distinct from a placement transaction.
                record["tags"] = {"purpose": "economy_calibration"}
                self.assertIsNone(validate_runtime_feature_policy([record], "", MovementEvidence.FORBIDDEN))

    def test_host_publications_keep_noop_ranks_explicit_and_do_not_add_rank_mirrors(self) -> None:
        """No-op rank payloads may be zero, never missing, malformed or unmatched."""
        records = host_movement_records()
        follower = [r | {"rank": 1} for r in records]
        follower[1] = follower[1] | {"value": 0}
        self.assertIsNone(validate_runtime_feature_policy(records + follower, "", MovementEvidence.REQUIRED))
        for invalid in (None, True, -1, float("nan")):
            changed = [follower[0], follower[1] | {"value": invalid}, *follower[2:]]
            self.assertIsNotNone(validate_runtime_feature_policy(records + changed, "", MovementEvidence.REQUIRED))
        self.assertIsNotNone(validate_runtime_feature_policy(records + [follower[0], *follower[2:]], "", MovementEvidence.REQUIRED))
        self.assertIsNotNone(validate_runtime_feature_policy(records + [records[-1]], "", MovementEvidence.REQUIRED))

    def test_runtime_features_native_movement_survives_empty_final_wave(self) -> None:
        """Terminal scratch reuse must not erase a qualified physical commit."""
        names = ("device_rebalance_request_copied_payload_lower_bound",
                 "device_rebalance_request_applied_payload_lower_bound",
                 "device_rebalance_request_useful_payload_bytes_lower_bound")
        for backend in ("CUDA", "ROCm"):
            records = [counter(name, domain="moe_rebalance", device=f"{backend}:0",
                               value=value, tags={"launch_count": "384"}) |
                       {"rank": 0, "phase": "decode"}
                       for name, value in zip(names, (2, 1, 3277312))]
            scratch = counter("device_rebalance_transfer_useful_payload_bytes",
                              domain="moe_rebalance", value=0)
            with self.subTest(backend=backend):
                self.assertIsNone(validate_runtime_feature_policy(records + [scratch], "",
                                  MovementEvidence.REQUIRED))
                self.assertIn("static", validate_runtime_feature_policy(records, "",
                              MovementEvidence.FORBIDDEN) or "")
                for omitted in range(3):
                    partial = records[:omitted] + records[omitted + 1:]
                    self.assertIn("committed physical", validate_runtime_feature_policy(
                        partial, "", MovementEvidence.REQUIRED) or "")
                # Pairing a copy from one participant/request with another's
                # apply can fabricate a transaction that never completed.
                for field, value in (("device", f"{backend}:1"), ("rank", 1),
                                     ("phase", "prefill"),
                                     ("tags", {"launch_count": "768"})):
                    mismatched = [records[0] | {field: value}, *records[1:]]
                    self.assertIn("committed physical", validate_runtime_feature_policy(
                        mismatched, "", MovementEvidence.REQUIRED) or "")
                for invalid in (0, -1, float("nan"), float("inf"), "invalid"):
                    incomplete = [records[0] | {"value": invalid}, *records[1:]]
                    self.assertIn("committed physical", validate_runtime_feature_policy(
                        incomplete, "", MovementEvidence.REQUIRED) or "")

    def test_runtime_features_certify_device_overlay_commits_symmetrically(self) -> None:
        """The sole overlay authority emits a completed transaction/edge/byte trio."""
        for backend in ("cuda", "rocm"):
            records = [record | {"device": backend + ":0"} for record in device_overlay_movement_records()]
            with self.subTest(backend=backend):
                self.assertIsNone(validate_runtime_feature_policy(records, "",
                    MovementEvidence.REQUIRED))
                self.assertIn("static", validate_runtime_feature_policy(records, "",
                    MovementEvidence.FORBIDDEN) or "")
                for omitted in range(len(records)):
                    partial = records[:omitted] + records[omitted + 1:]
                    self.assertIn("committed physical", validate_runtime_feature_policy(
                        partial, "", MovementEvidence.REQUIRED) or "")

    def test_device_overlay_completion_cannot_join_unrelated_transactions_or_ranks(self) -> None:
        """Global positive totals cannot fabricate one completed transaction."""
        records = device_overlay_movement_records()
        for field, value in (("rank", 1), ("device", "another-domain"), ("phase", "prefill")):
            with self.subTest(field=field):
                self.assertIsNotNone(validate_runtime_feature_policy(
                    [records[0] | {field: value}, *records[1:]], "", MovementEvidence.REQUIRED))
        for field, value in (("transaction", "4"), ("candidate_epoch", "5"), ("policy_owner", "host")):
            with self.subTest(field=field):
                self.assertIsNotNone(validate_runtime_feature_policy(
                    [records[0] | {"tags": records[0]["tags"] | {field: value}}, *records[1:]],
                    "", MovementEvidence.REQUIRED))

    def test_runtime_features_reject_device_overlay_proposals(self) -> None:
        """Submitted commands and prepared arrivals cannot certify a commit."""
        records = [counter(name, domain="moe_overlay_controller") for name in
                   ("dynamic_movement_commands", "prepared_arrival_descriptor_publications",
                    "physical_wave_parallel_operations_started")]
        self.assertIn("committed physical", validate_runtime_feature_policy(records, "",
            MovementEvidence.REQUIRED) or "")

    def test_runtime_features_reject_proposals_and_unapplied_payloads(self) -> None:
        """Neither a proposal nor transferred bytes prove published placement."""
        for name in ("device_rebalance_dynamic_ownership_swap_accepts",
                     "device_rebalance_transfer_useful_payload_bytes",
                     "device_rebalance_wave_applied_arrivals_total"):
            with self.subTest(name=name):
                self.assertIn("committed physical", validate_runtime_feature_policy(
                    [counter(name, domain="moe_rebalance")], "--moe-residency-maintenance dynamic", MovementEvidence.REQUIRED) or "")

    def test_runtime_features_static_rejects_movement(self) -> None:
        """Static keeps its negative movement obligation."""
        self.assertIsNone(validate_runtime_feature_policy([], "--moe-residency-maintenance off", MovementEvidence.FORBIDDEN))
        self.assertIn("static", validate_runtime_feature_policy(
            [counter("expert_migration_edges", domain="moe_overlay_residency")],
            "--moe-residency-maintenance=off", MovementEvidence.FORBIDDEN) or "")

    def test_runtime_features_reject_nonfinite_or_malformed_success(self) -> None:
        """Invalid diagnostic values cannot create accepted-token evidence."""
        for value in (float("nan"), float("inf"), "invalid", -1, 0):
            with self.subTest(value=value):
                records = [counter("draft_steps", domain="mtp"),
                           counter("accepted_tokens", domain="mtp", value=value)]
                self.assertIn("attempted and accepted", validate_runtime_feature_policy(records, "--mtp", MovementEvidence.NOT_APPLICABLE) or "")

    def test_runtime_features_do_not_invent_unconfigured_requirements(self) -> None:
        """KV/weight flags are not activation of prefix or speculative state."""
        self.assertIsNone(validate_runtime_feature_policy([], "--kv-cache-precision fp16 --mtp-max-draft-tokens 15", MovementEvidence.NOT_APPLICABLE))

    def test_runtime_movement_obligation_is_not_inferred_from_defaults(self) -> None:
        """The same Dynamic CLI default has different typed topology obligations."""
        flags = "--moe-residency-maintenance dynamic"
        self.assertIsNone(validate_runtime_feature_policy([], flags, MovementEvidence.NOT_APPLICABLE))
        self.assertIn("committed physical", validate_runtime_feature_policy([], flags, MovementEvidence.REQUIRED) or "")
        self.assertIn("typed movement", validate_runtime_feature_policy([], flags, "not_applicable") or "")

    def test_prefix_http_scenario_seeds_changes_answer_then_repeats_exactly(self) -> None:
        """Execute the shell scenario with a request observer instead of a server."""
        script = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        body = script.split("run_prefix_cache_checks() {", 1)[1].split("\n}\n", 1)[0]
        observer = "run_chat_answer_check() { printf '%s\\n' \"$5\" \"$6\" \"$7\"; }\n"
        result = subprocess.run(["bash", "-c", observer + "run_prefix_cache_checks() {" + body
                                 + "\n}\nrun_prefix_cache_checks tag 1234 200 true\n"],
                                check=True, text=True, capture_output=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 9)
        self.assertEqual(lines[1::3], ["13", "14", "13"])
        self.assertEqual(json.loads(lines[2]), json.loads(lines[8]))
        self.assertNotEqual(json.loads(lines[2]), json.loads(lines[5]))
        self.assertIn("exact-repeat", lines[6])

    def test_http_exchange_artifacts_preserve_exact_bytes_without_overwriting(self) -> None:
        """Full responses must survive repeated prompts and malformed replies."""
        script = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        body = script.split("record_http_exchange() {", 1)[1].split("\n}\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            request = '{"messages":[{"content":"quoted \\\"text\\\""}],"max_tokens":200}'
            responses = ['{"choices":[{"finish_reason":"length"}]}', 'not-json']
            commands = ["HTTP_CASE_SEQUENCE=0", "LOG_DIR=" + shlex.quote(directory),
                        "record_http_exchange() {" + body + "\n}"]
            commands += ["record_http_exchange " + shlex.quote(request) + " " + shlex.quote(response)
                         for response in responses]
            subprocess.run(["bash", "-c", "\n".join(commands)], check=True)
            for index, response in enumerate(responses, 1):
                stem = Path(directory) / f"http_{index:06d}"
                self.assertEqual(stem.with_suffix(".request.json").read_text(), request + "\n")
                self.assertEqual(stem.with_suffix(".response.json").read_text(), response + "\n")

    def test_short_arithmetic_rejects_budget_exhausted_answer_loops(self) -> None:
        """A correct last number cannot hide generation that never reached EOS."""
        script = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        body = script.split("extract_numeric_answer() {", 1)[1].split("\n}\n", 1)[0]
        for finish in ("stop", "length", "tool_calls", None):
            with self.subTest(finish=finish):
                response = json.dumps({"choices": [{"finish_reason": finish,
                    "message": {"content": "13\n\n13"}}]})
                result = subprocess.run(["bash", "-c", "extract_numeric_answer() {" + body
                    + "\n}\nextract_numeric_answer"], input=response, text=True,
                    capture_output=True, check=True)
                self.assertEqual(result.stdout.strip(), "13" if finish == "stop" else "")

    def test_shared_route_arena_accepts_both_runtime_ownership_models(self) -> None:
        """One expert runtime may serve many graphs without duplicating tables."""
        allocation = counter("moe_serial_route_scratch_arena_allocations", domain="memory",
            device="ROCm:0", tags={"bytes": "1024", "ownership": "per_device_serial_graph_domain",
                                    "immutable": "true", "largest_participant": "true"}) | {"rank": 1}
        binding = counter("moe_serial_route_scratch_runtime_table_bindings", domain="memory",
                          device="ROCm:0") | {"rank": 1}
        def graph(role: str) -> dict:
            return counter("materialized_graphs", domain="moe_overlay_participant_graph",
                           device="ROCm:0,ROCm:1", tags={"graph_role": role, "immutable": "true",
                                                       "allocation_policy": "setup_only"}) | {"rank": 1}
        self.assertIsNone(validate_moe_route_scratch_policy([allocation, binding | {"value": 2}]))
        self.assertIsNone(validate_moe_route_scratch_policy([allocation, binding, graph("main"), graph("mtp_draft")]))
        for bad in (
            [allocation], [allocation, binding],
            [allocation, binding, graph("main")],
            [allocation, binding, graph("main"), graph("mtp_draft") | {"rank": 0}],
            [allocation, binding | {"rank": 0, "value": 2}],
            [allocation, allocation, binding | {"value": 2}],
            [allocation | {"value": 2}, binding | {"value": 2}],
        ):
            with self.subTest(records=bad):
                self.assertIsNotNone(validate_moe_route_scratch_policy(bad))

    def test_mapped_route_arena_requires_native_families_and_one_runtime(self) -> None:
        """Declared shapes alone never certify model-lifetime runtime reuse."""
        def record(name: str, domain: str, tags: dict) -> dict:
            return counter(name, domain=domain, device="ROCm:0", tags=tags) | {"rank": 1}
        allocation = record("moe_serial_route_scratch_arena_allocations", "memory",
            {"bytes": "1024", "ownership": "per_device_serial_graph_domain",
             "immutable": "true", "largest_participant": "true"})
        binding = record("moe_serial_route_scratch_runtime_table_bindings", "memory", {})
        runtime = record("follower_runtime_tables_materialized", "moe_overlay_controller",
                         {"blocking_hot_path": "false"})
        def family(ordinal: str) -> dict:
            return record("materialized_mapped_follower_families", "moe_overlay_participant_graph",
                {"graph_family": ordinal, "standalone_progress_launch": "false",
                 "setup_materialized_gpu_transactions": "4"})
        sealed = record("mapped_follower_serving_graph_family_completions", "moe_overlay_participant_graph",
                        {"authority_installation": "before_native_capture"})
        good = [allocation, binding, runtime, family("0"), family("1"), sealed]
        self.assertIsNone(validate_moe_route_scratch_policy(good))
        for index in range(len(good)):
            with self.subTest(missing=index):
                self.assertIsNotNone(validate_moe_route_scratch_policy(good[:index] + good[index + 1:]))
        for index in (2, 3, 4, 5):
            with self.subTest(wrong_rank=index):
                bad = list(good)
                bad[index] = bad[index] | {"rank": 0}
                self.assertIsNotNone(validate_moe_route_scratch_policy(bad))
        for tags in ({"graph_family": "1", "setup_materialized_gpu_transactions": "0"},
                     {"graph_family": "1", "setup_materialized_gpu_transactions": "bad"}):
            self.assertIsNotNone(validate_moe_route_scratch_policy(
                good[:4] + [family("1") | {"tags": tags}] + good[5:]))

    def test_server_harness_embedded_python_is_syntactically_valid(self) -> None:
        """Compile every quoted Python heredoc used by the server matrix.

        The graph policy is invoked from a large inline validator. A syntax
        error in that shell-owned Python would prevent every GPU cell from
        reaching the standalone policy module even though the module's own
        tests remained green.
        """

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY(?:\n|$)",
            harness,
            flags=re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks):
            try:
                compile(
                    block,
                    f"test_server_e2e.sh:python-heredoc-{index}",
                    "exec",
                )
            except SyntaxError as error:
                self.fail(
                    f"embedded Python heredoc {index} is invalid: {error}"
                )

    def test_flash_attention_policy_accepts_rocm_query_and_context_plans(
        self,
    ) -> None:
        """ROCm capture inventory authenticates both three-node plan shapes."""

        common_tags = {
            "requested_axis": "geometry_selected",
            "batch_size": "1",
            "query_rows": "64",
            "local_query_heads": "16",
            "head_dim": "128",
            "kv_capacity": "8192",
            "query_grid_blocks": "64",
            "tile_q": "16",
            "tile_kv": "32",
            "kv_storage": "FP16",
        }
        query = counter(
            "rocm_fa2_parallel_plan_selections",
            domain="gpu_graph_inventory",
            device="rocm:0",
            tags=common_tags
            | {
                "selected_axis": "query_sequence",
                "context_partitions": "0",
                "context_partition_slots": "0",
                "context_phase_block_slots": "0",
                "device_direct_partition_limit": "0",
                "reducer_dimension_wavefronts": "0",
                "reducer_block_slots": "0",
            },
        ) | {"phase": "capture_setup"}
        context = counter(
            "rocm_fa2_parallel_plan_selections",
            domain="gpu_graph_inventory",
            device="rocm:0",
            tags=common_tags
            | {
                "selected_axis": "key_value_context",
                "context_partitions": "32",
                "context_partition_slots": "32",
                "context_phase_block_slots": "60",
                "device_direct_partition_limit": "1",
                "reducer_dimension_wavefronts": "4",
                "reducer_block_slots": "896",
            },
        ) | {"phase": "capture_setup"}

        result = validate_flash_attention_plan_policy(
            [query, context],
            "rocm:0",
            "",
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.expected_backends, frozenset({"rocm"}))
        self.assertEqual(result.observed_backends, frozenset({"rocm"}))
        self.assertEqual(result.plan_count, 2)
        self.assertEqual(
            result.selected_modes,
            frozenset({"query_sequence", "key_value_context"}),
        )

    def test_flash_attention_policy_rejects_legacy_query_only_request(
        self,
    ) -> None:
        """A graph that pins the retired backend policy fails closed."""

        record = counter(
            "rocm_fa2_parallel_plan_selections",
            domain="gpu_graph_inventory",
            device="rocm:0",
            tags={
                "requested_axis": "query_sequence",
                "selected_axis": "query_sequence",
                "batch_size": "1",
                "query_rows": "256",
                "local_query_heads": "16",
                "head_dim": "256",
                "kv_capacity": "4096",
                "query_grid_blocks": "256",
                "context_partitions": "0",
                "context_partition_slots": "0",
                "context_phase_block_slots": "0",
                "device_direct_partition_limit": "0",
                "reducer_dimension_wavefronts": "0",
                "reducer_block_slots": "0",
                "tile_q": "16",
                "tile_kv": "16",
                "kv_storage": "FP16",
            },
        ) | {"phase": "capture_setup"}

        result = validate_flash_attention_plan_policy(
            [record],
            "rocm:0",
            "",
        )
        self.assertIn("geometry-selected", result.error or "")

    def test_flash_attention_policy_requires_every_declared_gpu_backend(
        self,
    ) -> None:
        """A mixed CUDA/ROCm graph cannot omit one backend's FA2 evidence."""

        result = validate_flash_attention_plan_policy(
            [],
            "pp",
            "--define-domain mixed=cuda:0,rocm:0",
        )
        self.assertIn("cuda, rocm", result.error or "")

    def test_overlay_attention_ownership_follows_base_domain_not_expert_tiers(self) -> None:
        """GPU roles can reverse; an expert tier does not acquire attention."""
        for owner, expert in (("cuda", "rocm"), ("rocm", "cuda"), ("cpu", "cuda")):
            flags = shlex.join([
                "--moe-routed-expert-placement", "tiered-overlay",
                "--moe-routed-expert-continuation-domain", "model",
                "--moe-routed-expert-domain", f"model=localhost:-1:{owner}:0;scope=auto",
                "--moe-routed-expert-domain", f"experts=localhost:-1:{expert}:0;scope=auto",
            ])
            with self.subTest(owner=owner):
                self.assertEqual(attention_device_kinds_for_cell("tp", flags), {owner})
                evidence = validate_flash_attention_plan_policy([], "tp", flags)
                if owner == "cpu":
                    self.assertIsNone(evidence.error)
                else:
                    self.assertEqual(evidence.expected_backends, {owner})
                    self.assertIn(f"capture plan for: {owner}", evidence.error or "")

    def test_overlay_explicit_base_domain_owns_attention(self) -> None:
        """Use the same explicit base-domain precedence as the engine plan."""
        flags = shlex.join([
            "--moe-routed-expert-placement", "tiered-overlay",
            "--moe-routed-expert-continuation-domain", "continuation",
            "--moe-routed-expert-base-model-domain=base",
            "--define-domain", "base=localhost:0:rocm:0;scope=rank-local",
            "--moe-routed-expert-domain", "continuation=localhost:0:cuda:0",
        ])
        self.assertEqual(attention_device_kinds_for_cell("tp", flags), {"rocm"})

    def test_overlay_attention_joins_matching_domain_namespaces(self) -> None:
        """General orchestration and expert placement may name the same domain."""
        for kind in ("cpu", "cuda", "rocm"):
            for joined_option in (False, True):
                members = f"localhost:-1:{kind}:0,localhost:-1:{kind}:1"
                flags = [
                    "--moe-routed-expert-placement", "single-domain",
                    "--moe-routed-expert-continuation-domain", "model",
                    "--define-domain", f"model={members};scope=rank_local",
                ]
                expert_spec = f"model={members};routed_compute=apportioned"
                flags += ([f"--moe-routed-expert-domain={expert_spec}"]
                          if joined_option else ["--moe-routed-expert-domain", expert_spec])
                with self.subTest(kind=kind, joined_option=joined_option):
                    self.assertEqual(
                        attention_device_kinds_for_cell("tp", shlex.join(flags)),
                        {kind},
                    )

    def test_overlay_attention_rejects_duplicate_within_one_namespace(self) -> None:
        """Cross-namespace corroboration must not admit duplicate definitions."""
        for option in ("--define-domain", "--moe-routed-expert-domain"):
            flags = [
                "--moe-routed-expert-placement", "single-domain",
                "--moe-routed-expert-continuation-domain", "model",
                option, "model=cuda:0,cuda:1",
                option, "model=cuda:0,cuda:1",
            ]
            with self.subTest(option=option), self.assertRaises(ValueError):
                attention_device_kinds_for_cell("tp", shlex.join(flags))

    def test_overlay_attention_rejects_conflicting_cross_namespace_membership(self) -> None:
        """Backend kind alone cannot authenticate participant identity or order."""
        for members in ("cuda:0,cuda:2", "cuda:1,cuda:0", "rocm:0,rocm:1"):
            flags = [
                "--moe-routed-expert-placement", "single-domain",
                "--moe-routed-expert-continuation-domain", "model",
                "--define-domain", "model=cuda:0,cuda:1",
                "--moe-routed-expert-domain", f"model={members}",
            ]
            with self.subTest(members=members), self.assertRaises(ValueError):
                attention_device_kinds_for_cell("tp", shlex.join(flags))

    def test_overlay_attention_ownership_rejects_missing_and_ambiguous_domains(self) -> None:
        """Missing evidence cannot silently reclassify a continuation as expert-only."""
        prefix = "--moe-routed-expert-placement tiered-overlay"
        for suffix in ("", " --moe-routed-expert-continuation-domain",
                       " --moe-routed-expert-continuation-domain absent",
                       " --moe-routed-expert-continuation-domain model --moe-routed-expert-domain model=auto",
                       " --moe-routed-expert-continuation-domain model --moe-routed-expert-domain model=cuda:0 --define-domain model=rocm:0"):
            with self.subTest(suffix=suffix):
                self.assertIn("attention", validate_flash_attention_plan_policy([], "tp", prefix + suffix).error or "")

    def test_cpu_flash_attention_policy_accepts_both_physical_modes(
        self,
    ) -> None:
        """A long CPU lane proves prefill and context-split decode execution."""

        common_tags = {
            "requested_axis": "geometry_selected",
            "local_query_heads": "8",
            "head_dim": "256",
            "physical_workers": "28",
            "context_partition_rows": "256",
            "physical_kv_tile": "256",
        }
        query = counter(
            "cpu_fa2_parallel_plan_executions",
            value=96.0,
            domain="kernel",
            device="cpu",
            tags=common_tags
            | {
                "selected_mode": "query_sequence",
                "query_rows": "4096",
                "arithmetic_partitions": "16",
                "context_partitions": "1",
            },
        ) | {"phase": "execute"}
        context = counter(
            "cpu_fa2_parallel_plan_executions",
            value=808.0,
            domain="kernel",
            device="cpu",
            tags=common_tags
            | {
                "selected_mode": "key_value_context",
                "query_rows": "1",
                "arithmetic_partitions": "16",
                "context_partitions": "4",
            },
        ) | {"phase": "execute"}

        result = validate_cpu_flash_attention_execution_policy(
            [query, context]
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.plan_count, 2)
        self.assertEqual(result.execution_count, 904.0)
        self.assertEqual(
            result.selected_modes,
            frozenset({"query_sequence", "key_value_context"}),
        )

    def test_cpu_flash_attention_policy_requires_context_execution(
        self,
    ) -> None:
        """A declaration or query-only run cannot certify context splitting."""

        query = counter(
            "cpu_fa2_parallel_plan_executions",
            domain="kernel",
            device="cpu",
            tags={
                "requested_axis": "geometry_selected",
                "selected_mode": "query_sequence",
                "query_rows": "128",
                "local_query_heads": "8",
                "head_dim": "256",
                "physical_workers": "28",
                "arithmetic_partitions": "4",
                "context_partitions": "1",
                "context_partition_rows": "256",
                "physical_kv_tile": "128",
            },
        ) | {"phase": "execute"}

        result = validate_cpu_flash_attention_execution_policy([query])
        self.assertIn("never exercised K/V-context", result.error or "")

    def test_cpu_flash_attention_policy_rejects_false_context_record(
        self,
    ) -> None:
        """Context mode must represent multiple real physical producers."""

        context = counter(
            "cpu_fa2_parallel_plan_executions",
            domain="kernel",
            device="cpu",
            tags={
                "requested_axis": "geometry_selected",
                "selected_mode": "key_value_context",
                "query_rows": "1",
                "local_query_heads": "8",
                "head_dim": "256",
                "physical_workers": "28",
                "arithmetic_partitions": "8",
                "context_partitions": "1",
                "context_partition_rows": "256",
                "physical_kv_tile": "256",
            },
        ) | {"phase": "execute"}

        result = validate_cpu_flash_attention_execution_policy(
            [context],
            require_query_sequence=False,
        )
        self.assertIn("multiple context producers", result.error or "")

    def test_cpu_flash_attention_policy_rejects_diagnostic_axis(
        self,
    ) -> None:
        """Forced tournament controls cannot authenticate production policy."""

        context = counter(
            "cpu_fa2_parallel_plan_executions",
            domain="kernel",
            device="cpu",
            tags={
                "requested_axis": "key_value_context",
                "selected_mode": "key_value_context",
                "query_rows": "1",
                "local_query_heads": "8",
                "head_dim": "256",
                "physical_workers": "28",
                "arithmetic_partitions": "8",
                "context_partitions": "4",
                "context_partition_rows": "256",
                "physical_kv_tile": "256",
            },
        ) | {"phase": "execute"}

        result = validate_cpu_flash_attention_execution_policy(
            [context],
            require_query_sequence=False,
        )
        self.assertIn("geometry-selected", result.error or "")

    def test_gpu_host_transfer_policy_accepts_only_response_materialization(
        self,
    ) -> None:
        """Compact response mailboxes are the normal GPU-to-host boundary."""

        records = [
            counter(
                "stochastic_request_batch_summary_d2h_sync",
                domain="mtp",
            ),
            counter(
                "grouped_outcome_stochastic_device_outcome_host_bridge",
                domain="mtp",
                tags={"timing": "post_publication_response_bridge"},
            ),
            {
                "kind": "timer",
                "name": "device_generation_terminal_d2h_enqueue",
                "domain": "mtp",
                "value": 0,
                "count": 1,
                "total_ns": 500,
                "tags": {"requests": "1"},
            },
            {
                "kind": "timer",
                "name": "device_generation_terminal_d2h_wait",
                "domain": "mtp",
                "value": 0,
                "count": 1,
                "total_ns": 700,
                "tags": {"requests": "1"},
            },
            counter("d2h_bytes", value=4096.0, domain="transfer"),
        ]
        result = validate_gpu_host_transfer_policy(records)
        self.assertIsNone(result.error)
        self.assertEqual(
            result.final_response_operations,
            (
                "device_generation_terminal_d2h_enqueue",
                "device_generation_terminal_d2h_wait",
                "grouped_outcome_stochastic_device_outcome_host_bridge",
                "stochastic_request_batch_summary_d2h_sync",
            ),
        )

    def test_gpu_host_transfer_policy_accepts_authenticated_rocm_dispatch_ticket(
        self,
    ) -> None:
        """HIP may expose only its immutable graph-branch scheduler ticket."""

        result = validate_gpu_host_transfer_policy(
            [
                counter(
                    "device_generation_dispatch_ticket_d2h_submissions",
                    domain="mtp",
                    device="ROCm:0",
                    tags=device_generation_ticket_tags(),
                )
            ]
        )
        self.assertIsNone(result.error)
        self.assertEqual(
            result.scheduler_dispatch_operations,
            ("device_generation_dispatch_ticket_d2h_submissions",),
        )

    def test_device_generation_ticket_policy_matches_cpp_typed_abi(self) -> None:
        """The Python gate must track the sole C++ ticket ABI declaration."""

        source = (
            REPO_ROOT / "src/v2/kernels/common/SamplingMath.h"
        ).read_text(encoding="utf-8")
        version = re.search(r"kABIVersion\s*=\s*(\d+)u;", source)
        word_count = re.search(r"kWordCount\s*=\s*(\d+)u;", source)
        self.assertIsNotNone(version)
        self.assertIsNotNone(word_count)
        assert version is not None
        assert word_count is not None
        self.assertEqual(
            DEVICE_GENERATION_DISPATCH_TICKET_ABI_VERSION,
            int(version.group(1)),
        )
        self.assertEqual(
            DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT,
            int(word_count.group(1)),
        )
        self.assertEqual(
            DEVICE_GENERATION_DISPATCH_TICKET_BYTES,
            DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT * 4,
        )

    def test_gpu_host_transfer_policy_accepts_authenticated_rocm_moe_ticket(
        self,
    ) -> None:
        """HIP MoE cadence dispatch may expose only its 60-byte decision."""

        result = validate_gpu_host_transfer_policy(
            [
                counter(
                    "device_moe_rebalance_dispatch_ticket_d2h_submissions",
                    domain="moe_rebalance",
                    device="ROCm:1",
                    tags={
                        "bytes": "60",
                        "authority": "immutable_scheduler_snapshot",
                        "state_payload": "false",
                    },
                )
            ]
        )
        self.assertIsNone(result.error)
        self.assertEqual(
            result.scheduler_dispatch_operations,
            ("device_moe_rebalance_dispatch_ticket_d2h_submissions",),
        )

    def test_gpu_host_transfer_policy_rejects_malformed_rocm_moe_ticket(
        self,
    ) -> None:
        """MoE tickets remain fail-closed over backend, domain, and ABI size."""

        canonical_tags = {
            "bytes": "60",
            "authority": "immutable_scheduler_snapshot",
            "state_payload": "false",
        }
        invalid_records = {
            "cuda": counter(
                "device_moe_rebalance_dispatch_ticket_d2h_submissions",
                domain="moe_rebalance",
                device="CUDA:0",
                tags=canonical_tags,
            ),
            "wrong_domain": counter(
                "device_moe_rebalance_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags,
            ),
            "expanded_payload": counter(
                "device_moe_rebalance_dispatch_ticket_d2h_submissions",
                domain="moe_rebalance",
                device="ROCm:0",
                tags=canonical_tags | {"bytes": "64"},
            ),
        }
        for case, record in invalid_records.items():
            with self.subTest(case=case):
                result = validate_gpu_host_transfer_policy([record])
                self.assertIn(
                    "device_moe_rebalance_dispatch_ticket_d2h_submissions",
                    result.error or "",
                )

    def test_gpu_host_transfer_policy_rejects_noncanonical_dispatch_ticket(
        self,
    ) -> None:
        """Backend, ABI size, and no-state authority tags are fail-closed."""

        canonical_tags = device_generation_ticket_tags()
        invalid_records = {
            "cuda": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="CUDA:0",
                tags=canonical_tags,
            ),
            "expanded_payload": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags | {"bytes": "64"},
            ),
            "stale_abi_version": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags | {"abi_version": "1"},
            ),
            "stale_word_count": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags | {"word_count": "12"},
            ),
            "state_payload": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags | {"state_payload": "true"},
            ),
            "mutable_authority": counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                domain="mtp",
                device="ROCm:0",
                tags=canonical_tags | {"authority": "mutable_host_shadow"},
            ),
        }
        for case, record in invalid_records.items():
            with self.subTest(case=case):
                result = validate_gpu_host_transfer_policy([record])
                self.assertIn(
                    "device_generation_dispatch_ticket_d2h_submissions",
                    result.error or "",
                )

    def test_cuda_heterogeneous_ticket_requires_same_owner_retained_family(self) -> None:
        """A genuine mixed-device boundary permits only the exact immutable ABI."""
        ticket = counter("device_generation_dispatch_ticket_d2h_submissions",
                         domain="mtp", device="CUDA:0",
                         tags=device_generation_ticket_tags()) | {"rank": 0}
        family = counter("device_generation_loop_graph_materializations",
                         domain="mtp", device="CUDA:0",
                         tags={"execution": "hosted_captured_transactions_with_ticket_only_dispatch",
                               "fragments": "12"}) | {"rank": 0}
        mixed = frozenset({"cuda", "rocm"})
        self.assertIsNone(validate_gpu_host_transfer_policy([ticket, family], device_kinds=mixed).error)
        for kinds, proof, sample in (
            (frozenset({"cuda"}), family, ticket),
            (mixed, family | {"rank": 1}, ticket),
            (mixed, family | {"device": "CUDA:1"}, ticket),
            (mixed, family | {"value": 0}, ticket),
            (mixed, family | {"tags": {"execution": "native_conditional_parent", "fragments": "12"}}, ticket),
            (mixed, family, ticket | {"tags": device_generation_ticket_tags() | {"bytes": "64"}}),
            (mixed, family, ticket | {"tags": device_generation_ticket_tags() | {"state_payload": "true"}}),
            (mixed, family, ticket | {"name": "device_moe_rebalance_dispatch_ticket_d2h_submissions", "domain": "moe_rebalance"}),
        ):
            with self.subTest(kinds=kinds, proof=proof, sample=sample):
                self.assertIsNotNone(validate_gpu_host_transfer_policy([sample, proof], device_kinds=kinds).error)
        self.assertIsNotNone(validate_gpu_host_transfer_policy([ticket], device_kinds=mixed).error)

    def test_llep_verifier_policy_accepts_grouped_static_owner_execution(
        self,
    ) -> None:
        """Canonical LLEP keeps verifier rows on ordinary expert owners."""

        result = validate_llep_verifier_policy(
            [
                counter(
                    "static_owner_grouped_verifier_calls",
                    domain="moe_routed_execution",
                    tags={
                        "execution_policy": "static_owner_grouped",
                        "assignment": "static_owner",
                        "runtime_grouping": "runtime_table",
                        "row_execution_policy": "participant_assigned",
                        "current_batch_transport": "none",
                    },
                )
                | {"phase": "verifier"}
            ]
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.policy, "static_owner_grouped")

    def test_cuda_dynamic_mtp_policy_accepts_exact_switch_terminal_ledger(
        self,
    ) -> None:
        """The native selector and terminal totals jointly prove device ownership."""

        device = "CUDA:0"
        records = [
            counter(
                "device_generation_loop_graph_materializations",
                domain="mtp",
                device=device,
                tags={
                    "backend": "CUDA",
                    "depth_policy": "dynamic",
                    "execution": "native_device_controlled_selector_while",
                    "minimum_draft_depth": "1",
                    "maximum_draft_depth": "15",
                    "draft_depth": "15",
                    "verifier_rows": "16",
                    "sampling_mode": "stochastic",
                },
            ),
            counter(
                "device_generation_loop_graph_launches",
                domain="mtp",
                device=device,
                tags={
                    "execution": "single_async_native_selector_while_launch",
                    "minimum_draft_depth": "1",
                    "maximum_draft_depth": "15",
                },
            ),
            counter(
                "dynamic_device_generation_capacity_capture_transactions",
                domain="mtp",
                device=device,
                tags={
                    "capture_depth": "15",
                    "selected_depth": "4",
                    "authority": "device_generation_controller",
                },
            ),
            counter(
                "device_generation_terminal_transactions",
                value=3,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_attempted_draft_tokens",
                value=8,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_verifier_tokens",
                value=11,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_depth_updates",
                value=1,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_depth_evaluated_windows",
                value=2,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_depth_demotions",
                value=1,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_response_bridges",
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_compact_outcome_reductions",
                value=2,
                domain="mtp",
                device=device,
                tags={
                    "authority": "device_generation_controller",
                    "accounting_role": "captured_graph_replay_multiplier",
                    "source": "captured_stochastic_compact_outcome",
                    "execution": "native_conditional_graph",
                },
            ),
            counter(
                "device_generation_terminal_compact_outcome_reductions",
                value=1,
                domain="mtp",
                device=device,
                tags={
                    "authority": "device_generation_controller",
                    "accounting_role": "captured_graph_replay_multiplier",
                    "source": "captured_greedy_compact_outcome",
                    "execution": "native_conditional_graph",
                },
            ),
            counter(
                "device_generation_terminal_consumed_verifier_rows",
                value=6,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_accepted_speculative_tokens",
                value=4,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_rejected_transactions",
                value=1,
                domain="mtp",
                device=device,
            ),
        ]

        result = validate_cuda_dynamic_mtp_device_generation_policy(
            records,
            expected_minimum_depth=1,
            expected_maximum_depth=15,
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.devices, (device,))

        missing_stochastic_parent = list(records)
        missing_stochastic_parent[0] = records[0] | {
            "tags": (records[0].get("tags") or {}) | {"sampling_mode": "greedy"}
        }
        result = validate_cuda_dynamic_mtp_device_generation_policy(
            missing_stochastic_parent,
            expected_minimum_depth=1,
            expected_maximum_depth=15,
        )
        self.assertIn("native stochastic sampling parent", result.error or "")

        missing_stochastic_outcome = [
            record
            if (record.get("tags") or {}).get("source")
            != "captured_stochastic_compact_outcome"
            else record
            | {
                "tags": (record.get("tags") or {})
                | {"source": "captured_greedy_compact_outcome"}
            }
            for record in records
        ]
        result = validate_cuda_dynamic_mtp_device_generation_policy(
            missing_stochastic_outcome,
            expected_minimum_depth=1,
            expected_maximum_depth=15,
        )
        self.assertIn("compact-outcome provenance", result.error or "")

        records[5] = counter(
            "device_generation_terminal_verifier_tokens",
            value=12,
            domain="mtp",
            device=device,
        )
        result = validate_cuda_dynamic_mtp_device_generation_policy(
            records,
            expected_minimum_depth=1,
            expected_maximum_depth=15,
        )
        self.assertIn("terminal depth ledger", result.error or "")

    def test_rocm_dynamic_mtp_policy_accepts_authenticated_host_dispatch(
        self,
    ) -> None:
        """HIP host scheduling exposes decisions while state stays on device."""

        device = "ROCm:0"
        records = [
            counter(
                "device_generation_loop_graph_materializations",
                domain="mtp",
                device=device,
                tags={
                    "backend": "HIP",
                    "depth_policy": "dynamic",
                    "execution":
                        "hosted_captured_transactions_with_ticket_only_dispatch",
                    "conditional_fragments": "0",
                    "fragments": "12",
                    "minimum_draft_depth": "3",
                    "maximum_draft_depth": "3",
                    "draft_depth": "3",
                    "verifier_rows": "4",
                    "physical_verifier_rows": "4",
                    "sampling_mode": "stochastic",
                },
            ),
            counter(
                "device_generation_execution_policy_selections",
                domain="mtp",
                device=device,
                tags={
                    "policy": "host_scheduled_captured_transactions",
                    "selection_boundary": "pre_first_draft",
                    "topology": "dynamic_depth",
                },
            ),
            counter(
                "device_generation_loop_graph_launches",
                domain="mtp",
                device=device,
                tags={
                    "backend": "HIP",
                    "execution": "hosted_ticket_selected_captured_transactions",
                    "conditional_fragments": "0",
                    "fragments": "12",
                    "minimum_draft_depth": "3",
                    "maximum_draft_depth": "3",
                },
            ),
            counter(
                "dynamic_device_generation_capacity_capture_transactions",
                domain="mtp",
                tags={
                    "capture_depth": "3",
                    "selected_depth": "3",
                    "authority": "device_generation_controller",
                },
            ),
            counter(
                "device_generation_dispatch_ticket_d2h_submissions",
                value=3,
                domain="mtp",
                device=device,
                tags=device_generation_ticket_tags(),
            ),
            counter(
                "device_generation_dispatch_tickets_observed",
                value=3,
                domain="mtp",
                device=device,
                tags={
                    "transaction": "1",
                    "next_depth": "3",
                    "complete": "false",
                    "maintenance_due": "false",
                },
            ),
            counter(
                "hosted_device_generation_transaction_submissions",
                value=2,
                domain="mtp",
                device=device,
                tags={
                    "depth": "3",
                    "dynamic_depth_source": "device_controller_ticket",
                    "fragments": "12",
                },
            ),
            counter(
                "hosted_device_generation_terminal_submissions",
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_transactions",
                value=3,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_attempted_draft_tokens",
                value=9,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_verifier_tokens",
                value=12,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_depth_evaluated_windows",
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_response_bridges",
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_compact_outcome_reductions",
                value=3,
                domain="mtp",
                device=device,
                tags={
                    "authority": "device_generation_controller",
                    "accounting_role": "captured_graph_replay_multiplier",
                    "source": "captured_stochastic_compact_outcome",
                    "execution": "host_scheduled_captured_transactions",
                },
            ),
            counter(
                "device_generation_terminal_consumed_verifier_rows",
                value=7,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_accepted_speculative_tokens",
                value=4,
                domain="mtp",
                device=device,
            ),
            counter(
                "device_generation_terminal_rejected_transactions",
                domain="mtp",
                device=device,
            ),
        ]

        result = validate_rocm_host_scheduled_mtp_device_generation_policy(
            records,
            expected_minimum_depth=3,
            expected_maximum_depth=3,
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.devices, (device,))

        records[4] = records[4] | {
            "tags": (records[4].get("tags") or {}) | {"bytes": "64"}
        }
        result = validate_rocm_host_scheduled_mtp_device_generation_policy(
            records,
            expected_minimum_depth=3,
            expected_maximum_depth=3,
        )
        self.assertIn("scheduler ticket boundary", result.error or "")

    def test_llep_verifier_policy_rejects_prefill_assignment_bleed(self) -> None:
        """Least-loaded current-batch assignment is never a verifier policy."""

        result = validate_llep_verifier_policy(
            [
                counter(
                    "device_rebalance_llep_resident_assignment_calls",
                    domain="moe_rebalance",
                    tags={
                        "assignment": "logical_position_resident",
                        "current_batch_transport": "none",
                    },
                )
                | {"phase": "verifier"}
            ]
        )
        self.assertIn("leaked prefill assignment", result.error or "")

    def test_llep_verifier_policy_rejects_replicated_or_incomplete_evidence(
        self,
    ) -> None:
        """The canonical lane rejects alternate expert-placement policies."""

        mirrored = counter(
            "fully_replicated_local_verifier_execution_calls",
            domain="moe_rebalance",
            tags={
                "execution_policy": "fully_replicated_local",
                "participant_assignment": "none",
                "current_batch_transport": "none",
                "routed_result_collective": "none",
            },
        ) | {"phase": "verifier"}
        static_owner = counter(
            "static_owner_grouped_verifier_calls",
            domain="moe_routed_execution",
            tags={
                "execution_policy": "static_owner_grouped",
                "assignment": "static_owner",
                "runtime_grouping": "runtime_table",
                "row_execution_policy": "participant_assigned",
                "current_batch_transport": "none",
            },
        ) | {"phase": "verifier"}

        self.assertIn(
            "replicated-expert policy",
            validate_llep_verifier_policy([mirrored, static_owner]).error or "",
        )
        incomplete = dict(static_owner)
        incomplete["tags"] = {
            "execution_policy": "static_owner_grouped",
            "assignment": "static_owner",
            "runtime_grouping": "runtime_table",
            "current_batch_transport": "none",
        }
        self.assertIn(
            "incomplete",
            validate_llep_verifier_policy([incomplete]).error or "",
        )

    def test_gpu_host_transfer_policy_rejects_draft_proposal_readback(
        self,
    ) -> None:
        """A draft token remains device-owned until target verification."""

        result = validate_gpu_host_transfer_policy(
            [
                {
                    "kind": "timer",
                    "name": "stochastic_draft_greedy_proposal_d2h_sync",
                    "domain": "mtp",
                    "value": 0,
                    "count": 17,
                    "total_ns": 1200,
                    "tags": {"slot": "3"},
                }
            ]
        )
        self.assertIn("draft_greedy_proposal", result.error or "")

    def test_gpu_host_transfer_policy_rejects_draft_shadow_readback(
        self,
    ) -> None:
        """Diagnostic host shadows cannot enter the production serving path."""

        result = validate_gpu_host_transfer_policy(
            [
                counter(
                    "request_batch_sidecar_device_draft_shadow_d2h_sync",
                    domain="mtp",
                )
            ]
        )
        self.assertIn("draft_shadow", result.error or "")

    def test_gpu_host_transfer_policy_rejects_full_host_logits_access(
        self,
    ) -> None:
        """A full-vocabulary host view is forbidden even without D2H in its name."""

        result = validate_gpu_host_transfer_policy(
            [
                counter(
                    "host_logits_access",
                    domain="sampling",
                    tags={
                        "source": "device_graph_logits",
                        "rows": "1",
                        "cols": "248320",
                    },
                )
            ]
        )
        self.assertIn("host_logits_access", result.error or "")

    def test_gpu_host_transfer_policy_fails_closed_for_unknown_d2h(self) -> None:
        """New D2H boundaries require an explicit architectural review."""

        result = validate_gpu_host_transfer_policy(
            [counter("new_intermediate_state_d2h", domain="decode")]
        )
        self.assertIn("new_intermediate_state_d2h", result.error or "")

    def test_gpu_host_transfer_policy_distinguishes_cpu_tail_logits(self) -> None:
        """CPU-owned logits are legal; a CPU tier cannot authorize GPU logits downloads."""
        for gpu in ("cuda", "rocm"):
            for device in ("CPU", "CPU:0", "CPU:1", "CUDA:0", "ROCm:0", "", "CPU:invalid"):
                with self.subTest(gpu=gpu, device=device):
                    row = counter("host_logits_access", domain="sampling", device=device,
                                  tags={"source": "rank_orchestrator_logits"})
                    result = validate_gpu_host_transfer_policy([row], device_kinds=frozenset({gpu, "cpu"}))
                    self.assertEqual(result.error is None, device in {"CPU", "CPU:0", "CPU:1"})
                    self.assertIsNotNone(validate_gpu_host_transfer_policy([row], device_kinds=frozenset({gpu})).error)

    def test_gpu_host_transfer_policy_accepts_explicit_cache_tier_movement(
        self,
    ) -> None:
        """RAM/disk prefix tiers are intentional storage, not host mirrors."""

        result = validate_gpu_host_transfer_policy(
            [
                counter(
                    "prefix_cache_block_d2h",
                    domain="prefix_cache",
                    tags={"tier": "ram"},
                )
            ]
        )
        self.assertIsNone(result.error)
        self.assertEqual(
            result.prefix_cache_operations,
            ("prefix_cache_block_d2h",),
        )

    def test_server_harness_invokes_gpu_host_transfer_policy(self) -> None:
        """The canonical matrix must invoke the standalone fail-closed gate."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from gpu_host_transfer_perf_policy import "
            "validate_gpu_host_transfer_policy",
            harness,
        )
        self.assertIn(
            "host_transfer_validation = "
            "validate_gpu_host_transfer_policy(\n"
            "        records, device_kinds=graph_capture_validation.device_kinds)",
            harness,
        )

    def test_heterogeneous_activation_collective_is_not_a_host_state_mirror(self) -> None:
        """The reviewed wire requires topology, exact payload tags and local mapping."""
        for backend in ("cuda", "rocm"):
            for rank in (0, 1):
                with self.subTest(backend=backend, rank=rank):
                    mapped = dict(counter("mapped_regions_registered",
                        domain="moe_overlay_activation_epoch", tags={
                            "scope": "node_local", "blocking": "false",
                            "mapping": "typed_external_host_pages"}), rank=rank)
                    transfer = dict(counter("shared_physical_dispatch_d2h_bytes", value=4096,
                        domain="moe_overlay_activation_epoch", device=f"{backend}:0", tags={
                            "host_blocking": "false", "payload_layout": "shared_physical_rows",
                            "payload_path": "shared_physical_mapped", "role": "shared dispatch lane batch"}), rank=rank)
                    kinds = frozenset({backend, "cpu"})
                    result = validate_gpu_host_transfer_policy([mapped, transfer], device_kinds=kinds)
                    self.assertIsNone(result.error)
                    self.assertEqual(result.activation_collective_operations,
                                     ("shared_physical_dispatch_d2h_bytes",))
                    self.assertIsNotNone(validate_gpu_host_transfer_policy([mapped, transfer]).error)
                    self.assertIsNotNone(validate_gpu_host_transfer_policy(
                        [mapped, transfer], device_kinds=frozenset({backend})).error)
                    self.assertIsNotNone(validate_gpu_host_transfer_policy([transfer], device_kinds=kinds).error)
                    for key in transfer["tags"]:
                        bad = dict(transfer, tags={**transfer["tags"], key: "wrong"})
                        self.assertIsNotNone(validate_gpu_host_transfer_policy([mapped, bad], device_kinds=kinds).error)
                    for key in mapped["tags"]:
                        bad = dict(mapped, tags={**mapped["tags"], key: "wrong"})
                        self.assertIsNotNone(validate_gpu_host_transfer_policy([bad, transfer], device_kinds=kinds).error)
                    for key, value in (("rank", 1 - rank), ("domain", "mtp"),
                                       ("name", "verifier_state_d2h_bytes"), ("device", "cpu:0")):
                        bad = dict(transfer, **{key: value})
                        self.assertIsNotNone(validate_gpu_host_transfer_policy([mapped, bad], device_kinds=kinds).error)

    def test_canonical_certification_keeps_prefill_buckets_enabled(self) -> None:
        """Tagged E2E cells cannot inherit the historical prefill opt-out."""
        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        self.assertIn("e2e-certification", harness)
        driver = (SERVER_E2E_DIR.parents[3] / "scripts/ci/run_model_parity_e2e.py").read_text()
        self.assertNotIn("no-prefill-graph-buckets", driver)
        self.assertIn('"LLAMINAR_E2E_PERF_STATS": "1"', driver)

    def test_canonical_overlay_export_names_movement_policy_explicitly(self) -> None:
        """Actual round-trip tests, not shell-table patterns, prove the axes."""
        root = Path(__file__).resolve().parents[4]
        source = (root / "tests/v2/integration/parity/ModelParityRuntimeExport.h").read_text()
        self.assertIn('add("--moe-residency-maintenance", moeRebalanceRuntimeModeToString(config.moe_rebalance.mode))', source)
        self.assertIn("cell.expert_overlay->owner_order", source)
        self.assertIn("cell.expert_overlay->movement", source)

    def test_harness_has_no_independent_default_policy_matrix(self) -> None:
        """Unfinished LLEP must not enter certification through legacy defaults."""
        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
        self.assertNotIn("S9_LLEP_OVERLAY_", harness)
        self.assertNotIn("S9_STATIC_FLAGS=", harness)
        self.assertIn('exec python3 "$REPO_ROOT/scripts/ci/run_model_parity_e2e.py"', harness)

    def test_prefill_graph_probe_defeats_full_prefix_hits_at_fixed_geometry(
        self,
    ) -> None:
        """Prefix-cache cells must execute warmup, capture, and replay.

        Repeating one byte-identical prompt lets the RAM prefix tier answer the
        second and third requests without launching prefill at all. The probe
        therefore needs distinct first-block keys and an explicit equal-token
        assertion so every request exercises the same prefill graph key.
        """

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        probe_match = re.search(
            r"run_prefill_graph_probe\(\) \{(.*?)\n\}",
            harness,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(probe_match)
        probe = probe_match.group(1)
        for marker in ('probe_marker="A"', 'probe_marker="B"', 'probe_marker="C"'):
            self.assertIn(marker, probe)
        self.assertIn(
            'observed_prompt_tokens" != "$reference_prompt_tokens',
            probe,
        )
        self.assertIn(
            'f"{sys.argv[1]} You are a calculator.',
            probe,
        )

    def test_llep_perf_gate_requires_native_raw_allgather(self) -> None:
        """LLEP must prove native NCCL/RCCL transport without host rendezvous."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'record.get("domain") == "tp_raw_allgather_runtime"',
            harness,
        )
        self.assertIn(
            'record_tags.get("path") != "native_single_device_on_stream"',
            harness,
        )
        self.assertIn(
            'record_tags.get("backend_primitive") != expected_allgather_primitive',
            harness,
        )
        self.assertIn(
            'record_tags.get("host_rendezvous") != "false"',
            harness,
        )
        self.assertIn(
            'expected_allgather_primitive = "ncclAllGather"',
            harness,
        )
        self.assertIn(
            'expected_allgather_primitive = "rcclAllGather"',
            harness,
        )

    def test_movement_probe_uses_one_compact_payload_by_default(self) -> None:
        """The correctness probe must not inflate every captured MoE transfer."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "LLAMINAR_E2E_MOE_REBALANCE_COMPACT_PAYLOAD_SLOTS:-1",
            harness,
        )
        self.assertNotIn(
            "LLAMINAR_E2E_MOE_REBALANCE_COMPACT_PAYLOAD_SLOTS:-32",
            harness,
        )

    def test_release_ci_selects_typed_backend_tags(self) -> None:
        """Container variants only filter the canonical discovery inventory."""
        root = Path(__file__).resolve().parents[4]
        harness = root / "scripts/ci/run_release_container_e2e.sh"
        for backend, expected in (("cpu", "CPU"), ("cuda", "(?=.*CUDA)(?!.*ROCm).*"), ("rocm", "(?=.*ROCm)(?!.*CUDA).*"), ("hybrid", ".*CUDA.*ROCm.*")):
            result = subprocess.run(
                [str(harness), "--variant", backend, "--image", "unused:test-image", "--dry-run"],
                check=True, capture_output=True, text=True)
            command = shlex.split(result.stdout)
            self.assertEqual(command[0], "python3")
            self.assertTrue(command[1].endswith("run_model_parity_e2e.py"))
            self.assertEqual(command[command.index("--backend") + 1], expected)
            self.assertNotIn("--suite", command)

    def test_release_backend_partition_includes_cpu_tiers_without_duplicates(self) -> None:
        """GPU+CPU tags belong to the GPU image; mixed GPU tags require both vendors."""
        root = Path(__file__).resolve().parents[4]
        harness = root / "scripts/ci/run_release_container_e2e.sh"
        selected = {}
        for backend in ("cuda", "rocm", "hybrid"):
            result = subprocess.run(
                [str(harness), "--variant", backend, "--image", "unused:image", "--dry-run"],
                check=True, capture_output=True, text=True)
            command = shlex.split(result.stdout)
            pattern = command[command.index("--backend") + 1]
            selected[backend] = {signature for signature in
                ("CPU", "CUDA", "ROCm", "CPU+CUDA", "CPU+ROCm", "CUDA+ROCm", "CPU+CUDA+ROCm")
                if re.fullmatch(pattern, signature)}
        self.assertEqual(selected["cuda"], {"CUDA", "CPU+CUDA"})
        self.assertEqual(selected["rocm"], {"ROCm", "CPU+ROCm"})
        self.assertEqual(selected["hybrid"], {"CUDA+ROCm", "CPU+CUDA+ROCm"})

    def test_canonical_runner_does_not_expand_a_second_configuration_matrix(self) -> None:
        """The runner transports typed arguments and pins the full helper."""
        root = Path(__file__).resolve().parents[4]
        driver = (root / "scripts/ci/run_model_parity_e2e.py").read_text()
        inventory = (root / "scripts/ci/model_parity_inventory.py").read_text()
        self.assertIn("discover_inventory(args, InventoryScope.E2E)", driver)
        self.assertIn("parity.discover_campaigns(", inventory)
        self.assertIn('record["e2e"]["server_args"]', driver)
        self.assertIn('"LLAMINAR_E2E_LONG_CONTEXT_TIER": "full"', driver)
        self.assertNotIn("itertools.product", driver)
        self.assertNotIn("itertools.product", inventory)
        self.assertNotIn("Qwen3.6-", driver)

    def test_stochastic_probe_requires_device_resident_outcome_evidence(
        self,
    ) -> None:
        """The server gate rejects labels that do not execute stochastic GPU MTP."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            'flag_value("--mtp-verify-mode") != "speculative-sampling"',
            'flag_value("--mtp-depth-policy") != "dynamic"',
            'record.get("name") == "stochastic_accept_tests"',
            'get("device_resident") == "true"',
            '"stochastic_verify_request_batch_outcomes"',
            '"FAIL: stochastic GPU MTP entered the retired request-batched "',
            'get("sampling_mode") == "stochastic"',
            '"hosted_captured_transactions_with_ticket_only_dispatch"',
            'validate_rocm_host_scheduled_mtp_device_generation_policy',
            '"device_generation_terminal_compact_outcome_reductions"',
            '"captured_stochastic_compact_outcome"',
            '"stochastic_serial_equivalent_host_verifier_rows"',
            '"depth_policy_windows"',
        ):
            self.assertIn(marker, harness)

        runner = (
            REPO_ROOT / "src/v2/execution/runner/OrchestrationRunner.cpp"
        ).read_text(encoding="utf-8")
        terminal_ledger_start = runner.index(
            "GenerationResult OrchestrationRunner::completeDeviceResidentGeneration"
        )
        terminal_ledger_end = runner.index(
            "GenerationResult OrchestrationRunner::decodeStepMTP",
            terminal_ledger_start,
        )
        terminal_ledger = runner[terminal_ledger_start:terminal_ledger_end]
        for marker in (
            "DeviceGenerationSamplingMode::Stochastic",
            '"device_resident",',
            '"true"',
            '"grouped_decode_equivalent_stochastic"',
            '"device_resident_generation_terminal_ledger"',
            '"stochastic_accept_tests"',
            "stochastic_tags",
        ):
            self.assertIn(marker, terminal_ledger)

    def test_stochastic_probe_can_complete_a_dynamic_depth_window(self) -> None:
        """The HTTP workload must outlive the four-sample controller window."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        probe_start = harness.index("run_stochastic_mtp_probe()")
        probe_end = harness.index("run_prefill_graph_probe()", probe_start)
        probe = harness[probe_start:probe_end]

        self.assertIn(
            "LLAMINAR_E2E_STOCHASTIC_MTP_PROBE_MAX_TOKENS:-64",
            probe,
        )
        self.assertIn(
            "LLAMINAR_E2E_STOCHASTIC_MTP_PROBE_REPETITIONS:-1",
            probe,
        )
        self.assertIn("at least sixty-four lowercase", probe)
        self.assertIn("English words", probe)
        self.assertIn(
            '"$messages_json" "$probe_max_tokens" '
            '"false" "false" "stochastic" "12345"',
            probe,
        )
        self.assertIn('printf -v probe_marker "trial%02d"', probe)
        self.assertIn("probe_iteration <= probe_repetitions", probe)
        self.assertIn("common_prefix_chars", probe)
        self.assertIn('"sha256"', probe)

    def test_stochastic_http_probe_runs_only_after_health_publication(
        self,
    ) -> None:
        """HTTP inference must not race model loading and health publication."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        runner_start = harness.index("run_backend_tests()")
        runner_end = harness.index("# ─── Run Test Suites", runner_start)
        runner = harness[runner_start:runner_end]

        health_wait = runner.index(
            'if ! wait_for_health "$port" "$server_handle"; then'
        )
        health_pass = runner.index(
            'pass "[${tag}] Server started"',
            health_wait,
        )
        stochastic_probe = runner.index(
            'run_stochastic_mtp_probe "$tag" "$port"',
            health_pass,
        )
        self.assertLess(health_wait, health_pass)
        self.assertLess(health_pass, stochastic_probe)

    def test_device_kind_parser_covers_all_matrix_domain_syntaxes(self) -> None:
        flags = (
            "--tp-devices cuda:0,cuda:1 "
            "--define-domain stage=rocm:0,rocm:1;backend=rccl "
            "--moe-routed-expert-domain cold=0:cpu:0,1:cpu:0;scope=node_local"
        )
        self.assertEqual(
            device_kinds_for_cell("tp", flags),
            frozenset({"cuda", "rocm", "cpu"}),
        )

    def test_homogeneous_full_graph_passes(self) -> None:
        records = [
            counter(
                "decode_graph_phase",
                tags={"context": "main_decode", "phase": "capture"},
            ),
            counter(
                "decode_graph_phase",
                tags={"context": "main_decode", "phase": "replay"},
            ),
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "main_decode",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices rocm:0,rocm:1",
        )
        self.assertIsNone(result.error)
        self.assertTrue(result.has_full_graph_plan)
        self.assertTrue(result.has_nonempty_full_graph_executable)
        self.assertFalse(result.has_segmented_execution)
        self.assertTrue(result.prefill_lifecycle_complete)
        self.assertEqual(result.missing_prefill_phases, ())
        self.assertEqual(result.incomplete_contexts, ())

    def test_full_tier_gpu_requires_complete_prefill_lifecycle(self) -> None:
        """Decode capture cannot mask a prefill executable stuck before replay."""

        records = [
            counter(
                "decode_graph_phase",
                tags={"context": "main_decode", "phase": "capture"},
            ),
            counter(
                "decode_graph_phase",
                tags={"context": "main_decode", "phase": "replay"},
            ),
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "main_decode",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
            counter(
                "prefill_graph_phase",
                tags={"capture_phase": "capture"},
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
            require_prefill_lifecycle=True,
        )
        self.assertIn("missing phases: replay", result.error or "")
        self.assertFalse(result.prefill_lifecycle_complete)
        self.assertEqual(result.missing_prefill_phases, ("replay",))

    def test_full_tier_gpu_accepts_complete_prefill_lifecycle(self) -> None:
        """Capture and replay evidence closes the prefill contract without eager work."""

        records = [
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "main_decode",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
        ]
        records.extend(
            counter(
                "prefill_graph_phase",
                tags={"capture_phase": phase},
            )
            for phase in ("capture", "replay")
        )
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices rocm:0,rocm:1",
            require_prefill_lifecycle=True,
        )
        self.assertIsNone(result.error)
        self.assertTrue(result.prefill_lifecycle_complete)
        self.assertEqual(result.missing_prefill_phases, ())

    def test_setup_materialized_prefill_requires_same_bucket_and_no_eager_warmup(self) -> None:
        records = [counter("full_graph_plan_graphs", tags={"type": "capturable"}),
                   counter("full_graph_capture_executable_nodes", value=17, tags={
                       "context": "main_decode", "source": "full_graph_capture", "type": "captured_executable"}),
                   counter("prefill_graph_phase", tags={"bucket_seq_len": "256", "cache_phase": "ready",
                       "capture_phase": "materialized_without_launch"}),
                   counter("prefill_graph_phase", tags={"bucket_seq_len": "256", "capture_phase": "replay"})]
        for backend in ("cuda", "rocm"):
            self.assertIsNone(validate_graph_capture_policy(records, backend, "", require_prefill_lifecycle=True).error)
        records[-1]["tags"]["bucket_seq_len"] = "512"
        self.assertIn("capture", validate_graph_capture_policy(records, "cuda", "", require_prefill_lifecycle=True).missing_prefill_phases)
        records[-1]["tags"]["bucket_seq_len"] = "256"
        records.append(counter("prefill_graph_phase", tags={"capture_phase": "warmup"}))
        self.assertIn("retired eager warmup", validate_graph_capture_policy(records, "cuda", "", require_prefill_lifecycle=True).error)

    def test_native_setup_shapes_are_not_inference_invocations(self) -> None:
        """Unused materializations never demand synthetic inference to certify."""
        for device in ("cuda:0", "rocm:0"):
            rows = [
                counter("decode_graph_phase", value=2, device=device,
                        tags={"context": "condition_batch", "phase": "capture"}),
                dict(counter("full_graph_capture_executable_nodes", value=2859, device=device,
                             tags={"context": "condition_batch", "source": "full_graph_capture",
                                   "type": "materialized_unlaunched_executable"}), count=2),
            ]
            self.assertEqual(_incomplete_graph_contexts(rows), ())
            # Actual repeated execution is still required to produce replay.
            rows.append(counter("decode_capture_policy", value=2, device=device,
                                tags={"context": "condition_batch"}))
            self.assertIn("missing replay", str(_incomplete_graph_contexts(rows)))
            rows[-1]["value"] = 1
            self.assertIn("missing launch", str(_incomplete_graph_contexts(rows)))
            rows.append(counter("decode_graph_phase", device=device,
                                tags={"context": "condition_batch", "phase": "replay"}))
            self.assertEqual(_incomplete_graph_contexts(rows), ())

    def test_materialized_only_family_passes_full_policy_only_with_matching_launch(self) -> None:
        """Exercise the public validator, not only its inner lifecycle helper."""
        for device in ("cuda:0", "rocm:0"):
            rows = [counter("full_graph_plan_graphs", tags={"type": "capturable"}),
                    dict(counter("full_graph_capture_executable_nodes", value=1175, device=device,
                            tags={"context": "main_decode", "source": "full_graph_capture",
                                  "type": "materialized_unlaunched_executable"}), count=1),
                    counter("decode_graph_phase", device=device,
                            tags={"context": "main_decode", "phase": "capture"}),
                    counter("decode_capture_policy", value=1532, device=device,
                            tags={"context": "main_decode"}),
                    counter("decode_graph_phase", value=1532, device=device,
                            tags={"context": "main_decode", "phase": "replay"})]
            self.assertIsNone(validate_graph_capture_policy(rows, device, "").error)
            self.assertIn("missing launch", validate_graph_capture_policy(rows[:-1], device, "").error)
            for field, bad in (("context", "unrelated"), ("type", "captured_child_template"),
                               ("source", "segmented_graph_capture")):
                changed = copy.deepcopy(rows)
                changed[1]["tags"][field] = bad
                self.assertIsNotNone(validate_graph_capture_policy(changed, device, "").error)
            changed = copy.deepcopy(rows)
            changed[1]["value"] = 0
            self.assertIsNotNone(validate_graph_capture_policy(changed, device, "").error)
            changed = copy.deepcopy(rows)
            changed[1]["domain"] = "unrelated"
            self.assertIsNotNone(validate_graph_capture_policy(changed, device, "").error)

    def test_native_materialization_count_cannot_mask_another_owner_or_launch(self) -> None:
        """Rank/context identity and exact counts prevent over-crediting setup."""
        rows = [dict(counter("decode_graph_phase", value=4, device="rocm:0",
                             tags={"context": "condition_batch", "phase": "capture"}), rank=1),
                dict(counter("full_graph_capture_executable_nodes", value=1024, device="rocm:0",
                             tags={"context": "condition_batch", "source": "full_graph_capture",
                                   "type": "materialized_unlaunched_executable"}), count=2, rank=1)]
        self.assertIn("missing replay", str(_incomplete_graph_contexts(rows)))
        rows[0]["value"] = 2
        self.assertEqual(_incomplete_graph_contexts(rows), ())
        rows[1]["rank"] = 0
        self.assertIn("context-matched", str(_incomplete_graph_contexts(rows)))
        rows[1]["rank"] = 1
        rows[1]["tags"]["context"] = "unrelated"
        self.assertIn("context-matched", str(_incomplete_graph_contexts(rows)))

    def test_verifier_replay_requires_its_own_setup_materialized_executable(self) -> None:
        records = [counter("full_graph_plan_graphs", tags={"type": "capturable"}),
                   counter("full_graph_capture_executable_nodes", value=17, tags={
                       "context": "helper", "source": "full_graph_capture", "type": "captured_executable"}),
                   counter("full_graph_capture_executable_nodes", value=31, tags={
                       "context": "main_verifier", "source": "full_graph_capture", "type": "materialized_unlaunched_executable"}),
                   counter("decode_graph_phase", value=10, tags={"context": "main_verifier", "phase": "replay"})]
        self.assertIsNone(validate_graph_capture_policy(records, "rocm", "").error)
        records[2]["tags"]["context"] = "main_decode"
        self.assertIn("main_verifier", validate_graph_capture_policy(records, "rocm", "").error)

    def test_helper_executable_cannot_mask_legacy_verifier_warmup(self) -> None:
        """Context attribution must expose a verifier using retired eager warmup."""

        records = [
            counter(
                "decode_graph_phase",
                value=8.0,
                tags={"context": "main_verifier", "phase": "warmup"},
            ),
            counter(
                "decode_graph_phase",
                tags={"context": "mtp_helper", "phase": "capture"},
            ),
            counter(
                "decode_graph_phase",
                tags={"context": "mtp_helper", "phase": "replay"},
            ),
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=11.0,
                tags={
                    "context": "mtp_helper",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "cuda:0",
            "",
        )
        self.assertIn("main_verifier", result.error or "")
        self.assertEqual(len(result.incomplete_contexts), 1)

    def test_transaction_zero_requires_context_matched_capture(self) -> None:
        """Transaction-zero capture must own its matching executable."""

        records = [
            counter(
                "decode_graph_phase",
                tags={"context": "main_decode", "phase": "capture"},
            ),
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=5.0,
                tags={
                    "context": "unrelated",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
        ]
        result = validate_graph_capture_policy(records, "rocm:0", "")
        self.assertIn("context-matched executable", result.error or "")

    def test_repeated_mtp_sidecar_rebuilds_fail(self) -> None:
        """A sliding source pointer must not masquerade as graph progress."""

        records = [
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "mtp_shifted_prefill",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                value=32.0,
                domain="mtp",
                tags={
                    "context": "mtp_shifted_prefill",
                    "path": "plain_after_build",
                    "seq_len": "3",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                value=64.0,
                domain="mtp",
                tags={
                    "context": "mtp_shifted_prefill",
                    "path": "full_graph",
                    "seq_len": "3",
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
        )
        self.assertIn("rebuilt graph 32 times", result.error or "")
        self.assertEqual(len(result.incomplete_contexts), 1)

    def test_mtp_sidecar_allows_one_build_before_full_graph_replay(self) -> None:
        """One ordinary build is the capture lifecycle, not a fallback."""

        records = [
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "mtp_shifted_prefill",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                domain="mtp",
                tags={
                    "context": "mtp_shifted_prefill",
                    "path": "plain_after_build",
                    "seq_len": "3",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                value=2.0,
                domain="mtp",
                tags={
                    "context": "mtp_shifted_prefill",
                    "path": "plain",
                    "seq_len": "3",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                value=64.0,
                domain="mtp",
                tags={
                    "context": "mtp_shifted_prefill",
                    "path": "full_graph",
                    "seq_len": "3",
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.incomplete_contexts, ())

    def test_mtp_sidecar_groups_logical_aliases_by_physical_graph(self) -> None:
        """Logical callers cannot conceal duplicate physical graph builds."""

        records = [
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=17.0,
                tags={
                    "context": "mtp_decode_sidecar",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                domain="mtp",
                tags={
                    "context": "mtp_decode_sidecar_chain_device_token",
                    "graph_context": "mtp_decode_sidecar",
                    "path": "plain_after_build",
                    "seq_len": "1",
                },
            ),
            counter(
                "sidecar_graph_capture_path",
                domain="mtp",
                tags={
                    "context": "mtp_decode_sidecar_resident_logical_state",
                    "graph_context": "mtp_decode_sidecar",
                    "path": "plain_after_build",
                    "seq_len": "1",
                },
            ),
        ]
        result = validate_graph_capture_policy(records, "cuda:0", "")
        self.assertIn("rebuilt graph 2 times", result.error or "")
        self.assertEqual(
            result.incomplete_contexts,
            (
                "unknown:mtp_decode_sidecar[seq_len=1] "
                "(rebuilt graph 2 times for one stable shape)",
            ),
        )

    def test_full_graph_plan_without_instantiated_nodes_fails(self) -> None:
        records = [
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
        )
        self.assertIn(
            "non-empty instantiated GPU graph executable",
            result.error or "",
        )

    def test_homogeneous_segmented_plan_fails_even_with_collectives(self) -> None:
        records = [
            counter(
                "decode_capture_policy",
                tags={"has_collectives": "true"},
            ),
            counter(
                "segmented_plan_segments",
                tags={"type": "capturable"},
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
        )
        self.assertIn("forbidden for a homogeneous", result.error or "")

    def test_homogeneous_segmented_policy_tag_fails_without_plan_records(
        self,
    ) -> None:
        """Policy admission itself must fail closed if plan telemetry is absent."""

        records = [
            counter(
                "decode_capture_policy",
                tags={
                    "has_collectives": "true",
                    "heterogeneous_segmented": "true",
                    "replay_plan_policy": (
                        "allow_heterogeneous_boundary_segmentation"
                    ),
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices rocm:0,rocm:1",
        )
        self.assertTrue(result.has_segmented_execution)
        self.assertIn("forbidden for a homogeneous", result.error or "")

    def test_homogeneous_full_graph_policy_tags_remain_admissible(self) -> None:
        records = [
            counter(
                "decode_capture_policy",
                tags={
                    "has_collectives": "true",
                    "heterogeneous_segmented": "false",
                    "replay_plan_policy": "require_full_graph",
                },
            ),
            counter(
                "full_graph_plan_graphs",
                tags={"type": "capturable"},
            ),
            counter(
                "full_graph_capture_executable_nodes",
                value=9.0,
                tags={
                    "context": "main_verifier",
                    "source": "full_graph_capture",
                    "type": "captured_executable",
                },
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            "--tp-devices cuda:0,cuda:1",
        )
        self.assertIsNone(result.error)
        self.assertFalse(result.has_segmented_execution)

    def test_heterogeneous_segmentation_requires_collective_evidence(self) -> None:
        records = [
            counter(
                "graph_replay_plan_segments",
                domain="stage_gpu",
                tags={"source": "segmented_graph_capture"},
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "pp",
            (
                "--define-domain cuda_pp=cuda:0 "
                "--define-domain rocm_pp=rocm:0"
            ),
        )
        self.assertIn("runtime-proven collective", result.error or "")

    @staticmethod
    def pipeline_boundary_records(device: str = "cuda:0") -> list[dict]:
        """One captured GPU child plus one CPU child in a frozen rank-local PP plan."""
        geometry = {"total_segments": "2", "native_segments": "1", "host_segments": "1",
                    "heterogeneous_segmented": "true", "scope": "pipeline_coordinator",
                    "boundary_authority": "rank_pipeline_graph_plan"}
        rows = [counter("segmented_plan_segments", value=2, device="pipeline_coordinator", tags=geometry),
                counter("segmented_graph_capture_segments", device="pipeline_coordinator", tags=geometry)]
        for row in rows:
            row["phase"] = "setup"
        for phase in ("prefill", "decode"):
            rows.append(dict(counter("segmented_replay_segments", value=2, device="pipeline_coordinator",
                                     tags={"segments_per_transaction": "2", "transactions": "1",
                                           "heterogeneous_segmented": "true",
                                           "boundary_authority": "rank_pipeline_graph_plan"}), phase=phase))
        rows += [counter("full_graph_plan_graphs", device=device, tags={"type": "capturable"}),
                 counter("full_graph_capture_executable_nodes", value=17, device=device,
                         tags={"context": "main_decode", "source": "full_graph_capture", "type": "captured_executable"}),
                 counter("decode_graph_phase", device=device, tags={"context": "main_decode", "phase": "capture"}),
                 counter("decode_graph_phase", device=device, tags={"context": "main_decode", "phase": "replay"})]
        return [dict(row, rank=0) for row in rows]

    def test_pipeline_boundary_preserves_full_native_child_proof(self) -> None:
        """PP coordination is not an in-child collective on either GPU vendor."""
        for device in ("cuda:0", "rocm:0"):
            rows = self.pipeline_boundary_records(device)
            result = validate_graph_capture_policy(rows, "pp", f"{device} cpu:0")
            self.assertIsNone(result.error)
            self.assertTrue(result.has_pipeline_boundary_evidence)
            self.assertFalse(result.has_collective_evidence)
            without_executable = [r for r in rows if r["name"] != "full_graph_capture_executable_nodes"]
            self.assertIsNotNone(validate_graph_capture_policy(without_executable, "pp", f"{device} cpu:0").error)
            self.assertIsNotNone(validate_graph_capture_policy(rows, "pp", f"{device}").error)

    def test_pipeline_boundary_rejects_incomplete_or_cross_rank_evidence(self) -> None:
        """A policy tag or a neighboring rank cannot certify missing PP work."""
        mutations = (
            lambda r: r.pop(0),
            lambda r: r.pop(1),
            lambda r: r.pop(2),
            lambda r: r.pop(3),
            lambda r: r[1].update(rank=1),
            lambda r: r[1].update(value=2),
            lambda r: r[3].update(value=1),
            lambda r: r[3]["tags"].update(segments_per_transaction="3"),
            lambda r: r[0]["tags"].update(total_segments="wrong"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                rows = copy.deepcopy(self.pipeline_boundary_records())
                mutate(rows)
                # Even a nested TP child's valid collective cannot hide an
                # incomplete outer pipeline lifecycle.
                rows.append(counter("decode_capture_policy", tags={"has_collectives": "true"}))
                self.assertIsNotNone(validate_graph_capture_policy(rows, "pp", "cuda:0 cpu:0").error)

    @classmethod
    def overlay_boundary_records(cls, device: str = "cuda:0", rank: int = 0,
                                 native: int = 1, host: int = 1,
                                 generation_phase: str = "decode") -> list[dict]:
        """Reduce a real sparse-overlay server artifact without fabricating TP.

        The coordinator emits one frozen rank plan and one retirement per
        sequence; many sequences may share one command. GPU parents separately
        prove physical capture and launches.
        Exercise CPU/GPU and mixed-vendor GPU boundaries using the same schema.
        """
        total = native + host
        identity = {"authority": "typed_overlay_transaction_plan",
                    "scope": "cross_rank_expert_overlay"}
        geometry = identity | {"continuation_rank": str(rank), "graph_family_generation": "1",
                               "follower_segments": str(total - 1), "native_segments": str(native),
                               "eager_host_segments": str(host), "native_participants": str(native)}
        rows = [dict(counter("segmented_plan_segments", value=total, device="continuation_rank",
                             tags=dict(geometry)), rank=rank, phase="setup"),
                dict(counter("segmented_graph_capture_segments", value=native, device="continuation_rank",
                             tags=dict(geometry)), rank=rank, phase="setup")]
        for command, phase in enumerate(("prefill", generation_phase), 1):
            groups = 3 if phase == "mtp" else 1
            rows.append(dict(counter("segmented_replay_segments", value=total * groups,
                device="continuation_rank", tags=identity | {
                    "command": str(command), "sequence": str(command),
                    "draft_depth": "2" if phase == "mtp" else "0",
                    "graph_groups": str(groups), "plan_segments": str(total),
                    "terminal": "sparse_return_retired"}), rank=rank, phase=phase))
        for offset in range(native):
            participant = device if offset == 0 else ("rocm:0" if device == "cuda:0" else "cuda:0")
            child = cls.retained_parent_records(participant, rank + offset)
            for row in child:
                row["tags"]["has_collectives"] = "false"
            rows.extend(child)
        return copy.deepcopy(rows)

    def test_overlay_boundary_accepts_retired_sparse_transactions_without_child_tp(self) -> None:
        """Both vendors, shifted continuation ranks, and two/three tiers work."""
        for device in ("cuda:0", "rocm:0"):
            for rank in (0, 1):
                for native, host in ((1, 1), (2, 0), (2, 1)):
                    for phase in ("decode", "mtp"):
                        with self.subTest(device=device, rank=rank, native=native, host=host, phase=phase):
                            rows = self.overlay_boundary_records(device, rank, native, host, phase)
                            flags = f"{device} " + ("cpu:0" if host else "cuda:0 rocm:0")
                            result = validate_graph_capture_policy(
                                rows, "tp", flags, decode_requirement=DecodeGraphRequirement.REPLAY)
                            self.assertIsNone(result.error)
                            self.assertTrue(result.has_overlay_boundary_evidence)
                            self.assertFalse(result.has_collective_evidence)
                            self.assertFalse(result.has_pipeline_boundary_evidence)

    def test_overlay_boundary_rejects_missing_malformed_and_foreign_evidence(self) -> None:
        """Plan, physical preparation and retirement cannot borrow other owners."""
        mutations = (
            lambda r: r.pop(0), lambda r: r.pop(1), lambda r: r.pop(2), lambda r: r.pop(3),
            lambda r: r[1].update(rank=9), lambda r: r[3].update(rank=9),
            lambda r: r[0]["tags"].update(continuation_rank="9"),
            lambda r: r[1]["tags"].update(graph_family_generation="2"),
            lambda r: r[0]["tags"].update(graph_family_generation="0"),
            lambda r: r[0]["tags"].update(native_participants="0"),
            lambda r: r[0]["tags"].update(eager_host_segments="-1"),
            lambda r: r[0]["tags"].update(follower_segments="invalid"),
            lambda r: r[0].update(count=2), lambda r: r[1].update(value=0),
            lambda r: r[3].update(value=1), lambda r: r[3].update(value=float("nan")),
            lambda r: r[3]["tags"].update(terminal="submitted"),
            lambda r: r[3]["tags"].update(plan_segments="3"),
            lambda r: r[3]["tags"].update(command="0"),
            lambda r: r[3]["tags"].update(sequence="1"),
            lambda r: r[3]["tags"].update(sequence="0"),
            lambda r: r[3]["tags"].pop("sequence"),
            lambda r: r[3].update(count=2, value=4),
            lambda r: r[3]["tags"].update(graph_groups="0"),
            lambda r: r[3]["tags"].update(draft_depth="2"),
            lambda r: r[3]["tags"].update(scope="unrelated"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                rows = self.overlay_boundary_records()
                mutate(rows)
                # A child's TP counter must not hide a broken outer boundary.
                rows.append(counter("decode_capture_policy", tags={"has_collectives": "true"}))
                self.assertIn("Incomplete ExpertOverlay", validate_graph_capture_policy(
                    rows, "tp", "cuda:0 cpu:0").error or "")

    def test_overlay_boundary_distinguishes_many_sequences_in_one_command(self) -> None:
        """Four-row prefill and ticketed MTP retire sequences, not commands.

        The failing server artifact coalesced 54 prefill returns because its
        counter omitted the already authoritative sequence ID. Require that ID
        for every return; accepting a multiplied count would hide duplicates.
        """
        for device in ("cuda:0", "rocm:0"):
            for phase in ("prefill", "decode", "mtp"):
                with self.subTest(device=device, phase=phase):
                    rows = self.overlay_boundary_records(
                        device, generation_phase="mtp" if phase == "mtp" else "decode")
                    original = rows[2 if phase == "prefill" else 3]
                    for sequence in range(3, 56):
                        additional = copy.deepcopy(original)
                        additional["tags"]["sequence"] = str(sequence)
                        rows.append(additional)
                    flags = f"{device} cpu:0"
                    self.assertIsNone(validate_graph_capture_policy(rows, "tp", flags).error)
                    # Distinct commands cannot recycle a sequence either.
                    duplicate = copy.deepcopy(rows[-1])
                    duplicate["tags"]["command"] = "99"
                    rows.append(duplicate)
                    self.assertIn("Incomplete ExpertOverlay", validate_graph_capture_policy(
                        rows, "tp", flags).error or "")

    def test_overlay_boundary_keeps_native_executable_and_topology_obligations(self) -> None:
        """A retired command cannot certify missing GPU execution or eager work."""
        for device in ("cuda:0", "rocm:0"):
            rows = self.overlay_boundary_records(device)
            self.assertIn("homogeneous", validate_graph_capture_policy(rows, device, "").error or "")
            for missing in ("retained_parent_executable_nodes", "retained_parent_transaction_zero_launches",
                            "retained_parent_replays"):
                with self.subTest(device=device, missing=missing):
                    incomplete = [row for row in rows if row["name"] != missing]
                    self.assertIn("lifecycle", validate_graph_capture_policy(
                        incomplete, "tp", f"{device} cpu:0").error or "")
            rows.append(dict(counter("decode_graph_phase", device=device,
                tags={"context": "main_verifier", "phase": "warmup"}), rank=0))
            self.assertIn("eager warmup", validate_graph_capture_policy(
                rows, "tp", f"{device} cpu:0").error or "")

    @classmethod
    def local_ticket_boundary_records(cls, device: str = "cuda:0", rank: int = 0) -> list[dict]:
        """A rank-local CPU service retires alongside one captured GPU parent.

        Physical lowering combines many logical CPU cutpoints into one service
        program. Do not require a fake cross-rank coordinator or TP collective.
        The executor publishes the same sealed geometry on nodes and successful
        initial/repeat submissions, after joining the CPU service worker.
        """
        rows = cls.retained_parent_records(device, rank)
        for row in rows:
            row["tags"]["context"] = "main_decode"
            if row["name"] == "decode_capture_policy":
                row["tags"]["has_collectives"] = "false"
            if row["name"] in {"retained_parent_executable_nodes",
                               "retained_parent_transaction_zero_launches", "retained_parent_replays"}:
                row["tags"].update(boundary_authority="concurrent_ticket_service", ticket_service_units="1")
                row["count"] = 3 if row["name"] == "retained_parent_replays" else 1
        return rows

    def test_rank_local_overlay_boundary_uses_completed_ticket_service(self) -> None:
        """CUDA/CPU and ROCm/CPU need no cross-rank transaction coordinator."""
        for device in ("cuda:0", "rocm:0"):
            for rank in (0, 1):
                with self.subTest(device=device, rank=rank):
                    result = validate_graph_capture_policy(self.local_ticket_boundary_records(device, rank),
                        "tp", f"{device} cpu:0", decode_requirement=DecodeGraphRequirement.REPLAY)
                    self.assertIsNone(result.error)
                    self.assertTrue(result.has_overlay_boundary_evidence)
                    self.assertFalse(result.has_collective_evidence)
                    self.assertFalse(result.has_pipeline_boundary_evidence)

    def test_rank_local_overlay_boundary_rejects_missing_or_borrowed_retirement(self) -> None:
        """Neither a flag, a sibling, nor unrelated TP can certify this boundary."""
        mutations = (
            lambda r: r.pop(2), lambda r: r.pop(4), lambda r: r.pop(5),
            lambda r: r[2].update(rank=9), lambda r: r[4].update(device="rocm:7"),
            lambda r: r[5]["tags"].update(context="unrelated_helper"),
            lambda r: r[4]["tags"].update(child_units="48"),
            lambda r: r[4]["tags"].update(ticket_service_units="2"),
            lambda r: r[2]["tags"].update(ticket_service_units="0"),
            lambda r: r[4]["tags"].pop("boundary_authority"),
            lambda r: r[4].update(count=2), lambda r: r[4].update(value=0),
            lambda r: r[4].update(value=float("nan")),
            lambda r: r[2].update(domain="unrelated"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                rows = self.local_ticket_boundary_records()
                mutate(rows)
                rows.append(counter("decode_capture_policy", tags={"has_collectives": "true"}))
                self.assertIsNotNone(validate_graph_capture_policy(rows, "tp", "cuda:0 cpu:0").error)

    def test_rank_local_ticket_boundary_never_permits_homogeneous_segmentation(self) -> None:
        """Even valid CPU-service records cannot change the requested topology."""
        for device in ("cuda:0", "rocm:0"):
            self.assertIn("homogeneous", validate_graph_capture_policy(
                self.local_ticket_boundary_records(device), device, "").error or "")

    def test_decode_replay_gate_uses_native_family_and_not_prefill_or_sidecar(self) -> None:
        """One graph policy owns replay validation for every executable family."""
        for device in ("cuda:0", "rocm:0"):
            for family in ("full", "retained", "segmented"):
                with self.subTest(device=device, family=family):
                    if family == "retained":
                        rows = self.overlay_boundary_records(device)
                    elif family == "segmented":
                        rows = self.segmented_executable_records(device)
                    else:
                        rows = self.pipeline_boundary_records(device)
                    self.assertIsNone(validate_graph_capture_policy(
                        rows, "tp", f"{device} cpu:0", decode_requirement=DecodeGraphRequirement.REPLAY).error)
                    for context in ("prefill_bucket", "mtp_decode_sidecar"):
                        unrelated = copy.deepcopy(rows)
                        for row in unrelated:
                            if "context" in row["tags"]:
                                row["tags"]["context"] = context
                        self.assertIn("no context-matched decode", validate_graph_capture_policy(
                            unrelated, "tp", f"{device} cpu:0",
                            decode_requirement=DecodeGraphRequirement.REPLAY).error or "")

    def test_unused_retained_family_does_not_satisfy_required_decode_replay(self) -> None:
        """An instantiated but unlaunched setup family cannot certify inference."""
        rows = [row for row in self.overlay_boundary_records()
                if row["name"] not in {"decode_capture_policy", "retained_parent_replays",
                                       "retained_parent_transaction_zero_launches"}]
        self.assertIsNone(validate_graph_capture_policy(rows, "tp", "cuda:0 cpu:0").error)
        self.assertIn("expected context-matched decode", validate_graph_capture_policy(
            rows, "tp", "cuda:0 cpu:0", decode_requirement=DecodeGraphRequirement.REPLAY).error or "")

    def test_short_probe_still_requires_decode_capture_not_only_a_plan(self) -> None:
        """Consolidation preserves the shell's unconditional GPU capture gate."""
        rows = [counter("full_graph_plan_graphs", tags={"type": "capturable"}),
                counter("full_graph_capture_executable_nodes", value=17,
                        tags={"context": "main_decode", "source": "full_graph_capture",
                              "type": "captured_executable"})]
        self.assertIn("no context-matched decode", validate_graph_capture_policy(
            rows, "cuda:0", "", decode_requirement=DecodeGraphRequirement.CAPTURE).error or "")
        rows.append(counter("decode_graph_phase", tags={"context": "main_decode", "phase": "capture"}))
        self.assertIsNone(validate_graph_capture_policy(
            rows, "cuda:0", "", decode_requirement=DecodeGraphRequirement.CAPTURE).error)
        self.assertIn("expected context-matched decode", validate_graph_capture_policy(
            rows, "cuda:0", "", decode_requirement=DecodeGraphRequirement.REPLAY).error or "")

    @staticmethod
    def segmented_executable_records(device: str, rank: int = 1,
                                     context: str = "main_decode") -> list[dict]:
        """Reduce the real mixed-vendor TP artifact to two captured segments.

        Setup instantiates both units without executing arithmetic. Transaction
        zero and subsequent inference submit those same native executables.
        One segment cannot provide another segment's physical-node evidence.
        """
        rows = [counter("decode_capture_policy", value=3, device=device, tags={
                    "context": context, "has_collectives": "true",
                    "replay_plan_policy": "allow_heterogeneous_boundary_segmentation"}),
                counter("decode_graph_phase", device=device,
                        tags={"context": context, "phase": "capture"}),
                counter("decode_graph_phase", value=3, device=device,
                        tags={"context": context, "phase": "replay"}),
                counter("materialized_graph_transaction_zero_launches", device=device,
                        tags={"context": context, "segments": "3"})]
        for first, last in (("embedding", "router"), ("ffn", "output")):
            tags = {"context": context, "first_stage": first,
                    "last_stage": last, "stage_count": "2"}
            rows += [counter("segmented_graph_capture_executable_nodes", value=8,
                             device=device, tags=tags | {
                                 "source": "segmented_graph_capture",
                                 "type": "materialized_unlaunched_executable"}),
                     counter("segmented_replay_segments", value=3, device=device,
                             tags=tags | {"type": "capturable"})]
        return [dict(row, rank=rank) for row in rows]

    def test_heterogeneous_segments_require_matching_physical_executables(self) -> None:
        """Both vendors and decode/prefill contexts obey the same native proof."""
        for device in ("cuda:0", "rocm:0"):
            for rank in (0, 1):
                for context in ("main_decode", "prefill_bucket"):
                    with self.subTest(device=device, rank=rank, context=context):
                        rows = self.segmented_executable_records(device, rank, context)
                        result = validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0")
                        self.assertIsNone(result.error)
                        self.assertTrue(result.has_segmented_execution)
                        self.assertFalse(result.has_nonempty_full_graph_executable)

    def test_heterogeneous_segment_lifecycle_rejects_incomplete_or_wrong_owner(self) -> None:
        """A complete neighboring unit cannot hide a missing capture or launch."""
        for device in ("cuda:0", "rocm:0"):
            for missing in ("segmented_graph_capture_executable_nodes",
                            "segmented_replay_segments",
                            "materialized_graph_transaction_zero_launches"):
                with self.subTest(device=device, missing=missing):
                    rows = self.segmented_executable_records(device)
                    rows.remove(next(row for row in rows if row["name"] == missing))
                    self.assertIsNotNone(validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0").error)
            for field, value in (("rank", 0), ("device", "rocm:7"),
                                 ("context", "unrelated_helper"), ("source", "full_graph_capture"),
                                 ("first_stage", "different"), ("last_stage", "different"),
                                 ("stage_count", "3"), ("stage_count", "bad"),
                                 ("type", "captured_child_template"), ("value", 0),
                                 ("value", float("nan")), ("value", float("inf"))):
                with self.subTest(device=device, field=field, value=value):
                    rows = self.segmented_executable_records(device)
                    node = next(row for row in rows if row["name"] == "segmented_graph_capture_executable_nodes")
                    (node if field in {"rank", "device", "value"} else node["tags"])[field] = value
                    self.assertIsNotNone(validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0").error)

    def test_unused_segmented_family_needs_no_synthetic_launch(self) -> None:
        """Setup counts are physical units/shapes, never inference invocations."""
        rows = [row for row in self.segmented_executable_records("cuda:0")
                if row["name"] not in {"decode_capture_policy", "segmented_replay_segments",
                                       "materialized_graph_transaction_zero_launches"}
                and row["tags"].get("phase") != "replay"]
        rows.append(counter("decode_collective_graph_capture_policy",
                            tags={"has_collectives": "true"}))
        self.assertIsNone(validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0").error)

    def test_each_segment_replays_after_repeated_execution(self) -> None:
        """Repeated context submissions cannot borrow one neighbor's launches."""
        rows = self.segmented_executable_records("rocm:0")
        next(row for row in rows if row["name"] == "segmented_replay_segments")["value"] = 1
        self.assertIn("missing replay", validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0").error or "")

    def test_segment_proof_cannot_hide_eager_or_homogeneous_execution(self) -> None:
        """Physical segments are not permission to change the topology policy."""
        for device in ("cuda:0", "rocm:0"):
            rows = self.segmented_executable_records(device)
            self.assertIn("homogeneous", validate_graph_capture_policy(rows, device, "").error or "")
            rows.append(dict(counter("decode_graph_phase", device=device,
                tags={"context": "main_decode", "phase": "warmup"}), rank=1))
            self.assertIn("eager warmup", validate_graph_capture_policy(rows, "tp", "cuda:0 rocm:0").error or "")

    @staticmethod
    def retained_parent_records(device: str, rank: int = 1) -> list[dict]:
        """Minimal real retained-parent lifecycle, with process-local ownership."""
        context = "main_verifier"
        rows = [
            counter("decode_capture_policy", value=4, device=device, tags={
                "context": context, "has_collectives": "true",
                "heterogeneous_segmented": "true",
                "replay_plan_policy": "require_retained_parent_with_concurrent_ticket_service"}),
            counter("decode_graph_phase", device=device,
                    tags={"context": context, "phase": "capture"}),
            counter("retained_parent_executable_nodes", value=2781, device=device,
                    tags={"context": context, "child_units": "49"}),
            counter("retained_parent_materialized_without_launch", device=device,
                    tags={"context": context, "child_units": "49", "parent_nodes": "2781"}),
            counter("retained_parent_transaction_zero_launches", device=device,
                    tags={"context": context, "child_units": "49"}),
            counter("retained_parent_replays", value=3, device=device,
                    tags={"context": context, "child_units": "49"}),
        ]
        return [dict(row, rank=rank) for row in rows]

    def test_retained_parent_lifecycle_is_symmetric_and_rank_local(self) -> None:
        """Full parent nodes and launches certify either heterogeneous GPU backend."""
        for device in ("cuda:0", "rocm:0"):
            for rank in (0, 1):
                with self.subTest(device=device, rank=rank):
                    rows = self.retained_parent_records(device, rank)
                    result = validate_graph_capture_policy(rows, "tp", f"--tp-devices {device},cpu:0")
                    self.assertIsNone(result.error)

    def test_retained_parent_requires_its_own_nodes_and_launches(self) -> None:
        """A child template, wrong owner, or missing replay cannot certify a parent."""
        for device in ("cuda:0", "rocm:0"):
            for missing in ("retained_parent_executable_nodes",
                            "retained_parent_transaction_zero_launches",
                            "retained_parent_replays"):
                with self.subTest(device=device, missing=missing):
                    rows = [row for row in self.retained_parent_records(device) if row["name"] != missing]
                    rows.append(dict(counter("retained_parent_child_graph_nodes", value=2781,
                        device=device, tags={"context": "main_verifier"}), rank=1))
                    result = validate_graph_capture_policy(rows, "tp", f"--tp-devices {device},cpu:0")
                    self.assertIsNotNone(result.error)
            for field, value in (("rank", 0), ("device", "rocm:7"), ("context", "unrelated_helper")):
                with self.subTest(device=device, field=field):
                    rows = self.retained_parent_records(device)
                    nodes = next(row for row in rows if row["name"] == "retained_parent_executable_nodes")
                    if field == "context":
                        nodes["tags"][field] = value
                    else:
                        nodes[field] = value
                    self.assertIsNotNone(validate_graph_capture_policy(
                        rows, "tp", f"--tp-devices {device},cpu:0").error)

    def test_retained_parent_sidecar_requires_matching_physical_parent(self) -> None:
        """Logical sidecar aliases use native parent's replay evidence, not plain execution."""
        rows = self.retained_parent_records("rocm:0")
        rows.append(dict(counter("sidecar_graph_capture_path", value=4, domain="mtp", device="rocm:0",
            tags={"context": "logical_alias", "graph_context": "main_verifier",
                  "seq_len": "1", "path": "retained_parent"}), rank=1))
        self.assertIsNone(validate_graph_capture_policy(rows, "tp", "rocm:0 cpu:0").error)
        rows[-1]["tags"]["graph_context"] = "foreign_parent"
        self.assertIn("no matching certified retained parent",
                      validate_graph_capture_policy(rows, "tp", "rocm:0 cpu:0").error or "")

    def test_unused_retained_family_requires_no_synthetic_launch(self) -> None:
        """Multiple setup shapes are not repeated inference invocations."""
        rows = [row for row in self.retained_parent_records("cuda:0")
                if row["name"] not in {"decode_capture_policy", "retained_parent_replays",
                                       "retained_parent_transaction_zero_launches"}]
        for row in rows:
            row["value"] *= 2
        rows.append(dict(counter("decode_collective_graph_capture_policy", device="cuda:0",
            tags={"has_collectives": "true"}), rank=1))
        self.assertIsNone(validate_graph_capture_policy(rows, "tp", "cuda:0 cpu:0").error)

    def test_retained_parent_cannot_hide_homogeneous_segmentation_or_warmup(self) -> None:
        for device in ("cuda:0", "rocm:0"):
            rows = self.retained_parent_records(device)
            self.assertIn("homogeneous", validate_graph_capture_policy(rows, device, "").error or "")
            rows.append(dict(counter("decode_graph_phase", device=device,
                tags={"context": "main_verifier", "phase": "warmup"}), rank=1))
            self.assertIn("eager warmup", validate_graph_capture_policy(rows, "tp", f"{device} cpu:0").error or "")

    def test_heterogeneous_collective_segmentation_passes(self) -> None:
        records = [
            counter(
                "decode_capture_policy",
                tags={"has_collectives": "true"},
            ),
            counter(
                "segmented_replay_launches",
                tags={"source": "segmented_graph_capture"},
            ),
        ]
        result = validate_graph_capture_policy(
            records,
            "tp",
            (
                "--moe-routed-expert-domain "
                "hot=cuda:0,cuda:1;scope=rank_local "
                "--moe-routed-expert-domain "
                "cold=0:cpu:0,1:cpu:0;scope=node_local"
            ),
        )
        self.assertIsNone(result.error)
        self.assertTrue(result.heterogeneous_device_mix)
        self.assertTrue(result.has_collective_evidence)

    def test_gpu_cell_without_any_replay_plan_fails(self) -> None:
        result = validate_graph_capture_policy(
            [],
            "cuda:0",
            "",
        )
        self.assertIn("neither a full-graph plan", result.error or "")

    def test_request_input_lifetime_accepts_complete_multi_request_chain(
        self,
    ) -> None:
        records = [
            counter(
                "device_owned_input_rows",
                value=128.0,
                domain="request_admission",
            ),
            counter(
                "device_input_event_waits",
                value=4.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reader_admissions",
                value=10.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reuse_publications",
                value=10.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reuse_waits",
                value=3.0,
                domain="request_admission",
            ),
        ]
        result = validate_request_input_lifetime_policy(records)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.devices), 1)

    def test_request_input_lifetime_rejects_missing_final_reader_release(
        self,
    ) -> None:
        records = [
            counter(
                "device_owned_input_rows",
                value=128.0,
                domain="request_admission",
            ),
            counter(
                "device_input_event_waits",
                value=4.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reader_admissions",
                value=4.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reuse_publications",
                value=3.0,
                domain="request_admission",
            ),
            counter(
                "device_input_reuse_waits",
                value=2.0,
                domain="request_admission",
            ),
        ]
        result = validate_request_input_lifetime_policy(records)
        self.assertIn("reader admissions (4) != releases (3)", result.error or "")

    def test_request_input_lifetime_rejects_unexercised_reuse_wait(
        self,
    ) -> None:
        records = [
            counter(
                "device_owned_input_rows",
                value=32.0,
                domain="request_admission",
            ),
            counter(
                "device_input_event_waits",
                domain="request_admission",
            ),
            counter(
                "device_input_reader_admissions",
                domain="request_admission",
            ),
            counter(
                "device_input_reuse_publications",
                domain="request_admission",
            ),
        ]
        result = validate_request_input_lifetime_policy(records)
        self.assertIn("no release-to-next-writer event waits", result.error or "")


class TestRankedPerfArtifacts(unittest.TestCase):
    """Require complete runtime membership without imposing a GPU rank layout."""

    def setUp(self) -> None:
        """Give each test a unique cell artifact namespace."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "cell.perfstats.json"

    def write_rank(self, rank: int, *, size: int = 2, authority: int = 1,
                   rows: list[dict] | None = None) -> Path:
        """Emit the same membership row and schema as ServerMode."""
        path = self.output.with_name(f"cell.perfstats.rank-{rank}.json")
        identity = counter("rank_membership", domain="server", tags={
            "rank": str(rank), "world_size": str(size), "authority_rank": str(authority)})
        path.write_text(json.dumps({"schema": "llaminar.perf_stats.v1",
                                    "records": [identity, *(rows or [])]}))
        return path

    def test_nonzero_gpu_authority_and_every_raw_row_are_preserved(self) -> None:
        """No rank-zero convention, reduction or device renumbering is allowed."""
        for authority in (0, 1):
            with self.subTest(authority=authority):
                cpu = counter("compute", device="cpu:0", value=7)
                gpu = counter("compute", device="rocm:0", value=13)
                self.write_rank(0, authority=authority, rows=[cpu])
                self.write_rank(1, authority=authority, rows=[gpu])
                data = collect_ranked_perf_stats(self.output)
                self.assertEqual(data["world_size"], 2)
                self.assertEqual(data["authority_rank"], authority)
                self.assertIn(dict(cpu, rank=0), data["records"])
                self.assertIn(dict(gpu, rank=1), data["records"])
                self.assertEqual(len(data["records"]), 4)
                self.assertFalse(self.output.exists())

    def test_single_rank_is_the_same_collection_protocol(self) -> None:
        self.write_rank(0, size=1, authority=0)
        self.assertEqual(collect_ranked_perf_stats(self.output)["world_size"], 1)

    def test_publication_and_validation_share_one_parse_per_rank(self) -> None:
        """Preserve all rows and raw bytes while returning the published owner."""
        files = [self.write_rank(rank, rows=[counter("ordered", value=rank + 7)])
                 for rank in range(2)]
        raw = [path.read_bytes() for path in files]
        with mock.patch("ranked_perf_artifacts.json.loads", wraps=json.loads) as parse, \
             mock.patch("ranked_perf_artifacts.publish_ranked_perf_stats",
                        wraps=publish_ranked_perf_stats) as publish:
            data = collect_and_publish_ranked_perf_stats(self.output)
            self.assertEqual(parse.call_count, 2)
            publish.assert_called_once()
            self.assertIs(publish.call_args.args[1], data)
        self.assertEqual(json.loads(self.output.read_text()), data)
        self.assertEqual([path.read_bytes() for path in files], raw)
        self.assertEqual(len(data["records"]), 4)
        self.assertFalse(list(self.output.parent.glob("*.tmp")))

    def test_failed_publication_preserves_previous_aggregate_and_removes_only_temporary(self) -> None:
        """A failed writer cannot advertise partial or replace valid evidence."""
        self.write_rank(0, size=1, authority=0)
        previous = '{"previous":"complete evidence"}\n'
        self.output.write_text(previous)
        before = set(self.output.parent.iterdir())
        with mock.patch("ranked_perf_artifacts.os.replace", side_effect=OSError("injected publish failure")):
            with self.assertRaisesRegex(OSError, "injected publish failure"):
                collect_and_publish_ranked_perf_stats(self.output)
        self.assertEqual(self.output.read_text(), previous)
        self.assertEqual(set(self.output.parent.iterdir()), before)

    def test_invalid_membership_never_publishes_or_reuses_stale_evidence(self) -> None:
        """The same one-pass boundary fails closed before touching the output."""
        self.write_rank(0)
        previous = '{"previous":"not a substitute for missing rank"}\n'
        self.output.write_text(previous)
        with mock.patch("ranked_perf_artifacts.publish_ranked_perf_stats") as publish:
            with self.assertRaisesRegex(ValueError, "missing rank evidence"):
                collect_and_publish_ranked_perf_stats(self.output)
            publish.assert_not_called()
        self.assertEqual(self.output.read_text(), previous)

    def test_missing_last_or_middle_rank_fails_closed(self) -> None:
        self.write_rank(0, size=3, authority=2)
        with self.assertRaisesRegex(ValueError, "missing rank evidence"):
            collect_ranked_perf_stats(self.output)
        self.write_rank(2, size=3, authority=2)
        with self.assertRaisesRegex(ValueError, "missing rank evidence"):
            collect_ranked_perf_stats(self.output)

    def test_conflicting_world_or_authority_is_rejected(self) -> None:
        self.write_rank(0)
        for size, authority in ((3, 1), (2, 0)):
            self.write_rank(1, size=size, authority=authority)
            with self.assertRaisesRegex(ValueError, "conflicting server"):
                collect_ranked_perf_stats(self.output)

    def test_filename_cannot_impersonate_another_rank(self) -> None:
        source = self.write_rank(0)
        source.rename(source.with_name("cell.perfstats.rank-1.json"))
        with self.assertRaisesRegex(ValueError, "outside declared communicator"):
            collect_ranked_perf_stats(self.output)

    def test_missing_duplicate_invalid_membership_and_malformed_rows_rejected(self) -> None:
        path = self.write_rank(0, size=1, authority=0)
        valid = json.loads(path.read_text())
        for records in ([], valid["records"] * 2, [None], {},
                        [dict(valid["records"][0], tags={"rank": "0"})],
                        [dict(valid["records"][0], value=0)]):
            path.write_text(json.dumps(dict(valid, records=records)))
            with self.subTest(records=records), self.assertRaises(ValueError):
                collect_ranked_perf_stats(self.output)

    def test_unqualified_stale_aggregate_is_not_participant_evidence(self) -> None:
        self.output.write_text('{"schema":"llaminar.perf_stats.v1","records":[]}')
        with self.assertRaisesRegex(ValueError, "missing rank-qualified"):
            collect_ranked_perf_stats(self.output)

    def test_prefill_capture_on_one_rank_cannot_certify_another(self) -> None:
        capture = counter("prefill_graph_phase", device="cuda:0",
                          tags={"capture_phase": "capture", "bucket_seq_len": "64"})
        replay = counter("prefill_graph_phase", device="cuda:0",
                         tags={"capture_phase": "replay", "bucket_seq_len": "64"})
        self.assertIn("capture", _missing_prefill_phases(
            [dict(capture, rank=0), dict(replay, rank=1)]))
        self.assertEqual(_missing_prefill_phases(
            [dict(capture, rank=1), dict(replay, rank=1)]), ())

    def test_decode_capture_on_one_rank_cannot_certify_another(self) -> None:
        capture = counter("decode_graph_phase", device="cuda:0",
                          tags={"phase": "capture", "context": "main"})
        replay = counter("decode_graph_phase", device="cuda:0",
                         tags={"phase": "replay", "context": "main"})
        executable = counter("full_graph_capture_executable_nodes", device="cuda:0",
                             tags={"context": "main", "source": "full_graph_capture",
                                   "type": "captured_executable"})
        failures = _incomplete_graph_contexts([dict(capture, rank=0),
                                              dict(executable, rank=0), dict(replay, rank=1)])
        self.assertEqual(len(failures), 1)
        self.assertIn("rank=1/cuda:0:main", failures[0])
        self.assertEqual(_incomplete_graph_contexts([
            dict(row, rank=1) for row in (capture, replay, executable)]), ())

    def memory_evidence(self) -> dict:
        """Build a lazy pool's coherent, authority-produced owner snapshot."""
        resource = counter("resource_admission", domain="physical_memory", device="cpu:0",
                           tags={"rank": "0", "incremental_bytes": "100", "available_bytes": "100",
                                 "owner_count": "1"})
        owner = counter("owner_attestation", domain="physical_memory", device="cpu:0",
                        tags={"rank": "0", "owner": "routed_expert_weights", "planned_new_bytes": "100",
                              "committed_new_bytes": "100", "materialized_new_bytes": "40",
                              "planned_resident_bytes": "20", "adopted_resident_bytes": "20"})
        return {"world_size": 1, "records": [dict(resource, rank=0), dict(owner, rank=0)]}

    def test_lazy_memory_pool_does_not_need_to_be_fully_materialized(self) -> None:
        validate_memory_authority(self.memory_evidence())

    def test_memory_authority_admission_and_owner_bounds_are_required(self) -> None:
        for index, field, value in ((0, "incremental_bytes", "101"),
                                    (1, "committed_new_bytes", "101"),
                                    (1, "materialized_new_bytes", "101"),
                                    (1, "adopted_resident_bytes", "21"),
                                    (1, "materialized_new_bytes", "-1")):
            evidence = self.memory_evidence()
            evidence["records"][index]["tags"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_memory_authority(evidence)

    def test_memory_authority_missing_rank_resource_owner_and_duplicate_rejected(self) -> None:
        valid = self.memory_evidence()
        for evidence in (dict(valid, world_size=2), dict(valid, records=valid["records"][:1]),
                         dict(valid, records=valid["records"][1:]),
                         dict(valid, records=valid["records"] * 2)):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_memory_authority(evidence)

    def test_shutdown_post_has_explicit_empty_body_and_observes_acceptance(self) -> None:
        """Exercise the actual shell request against a strict local HTTP peer."""
        observed = []

        class ShutdownPeer(BaseHTTPRequestHandler):
            """Require the same body framing that cpp-httplib expects."""

            def do_POST(self) -> None:
                observed.append((self.path, self.headers.get("Content-Length")))
                self.send_response(202 if observed[-1] == ("/admin/shutdown", "0") else 400)
                self.end_headers()

            def log_message(self, *_args) -> None:
                """Keep successful unit tests quiet."""

        with HTTPServer(("127.0.0.1", 0), ShutdownPeer) as server:
            worker = threading.Thread(target=server.handle_request)
            server.timeout = 5
            worker.start()
            shell = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text()
            function = shell.split("request_server_shutdown() {", 1)[1].split("\n}\n", 1)[0]
            try:
                result = subprocess.run(["bash", "-c", "server_base_url() { echo http://127.0.0.1:$1; }\n"
                    + "request_server_shutdown() {" + function + "\n}\n"
                    + f"request_server_shutdown {server.server_port}"], timeout=8, capture_output=True, text=True)
            finally:
                worker.join(timeout=6)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(observed, [("/admin/shutdown", "0")])


if __name__ == "__main__":
    unittest.main()
