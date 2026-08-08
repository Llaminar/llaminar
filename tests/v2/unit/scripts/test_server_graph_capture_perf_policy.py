#!/usr/bin/env python3
"""Regression tests for the server E2E graph-capture PerfStats gate."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SERVER_E2E_DIR = (
    Path(__file__).resolve().parents[2] / "e2e" / "server"
)
REPO_ROOT = SERVER_E2E_DIR.parents[3]
sys.path.insert(0, str(SERVER_E2E_DIR))

from graph_capture_perf_policy import (  # noqa: E402
    device_kinds_for_cell,
    validate_graph_capture_policy,
)
from gpu_host_transfer_perf_policy import (  # noqa: E402
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


class TestServerGraphCapturePerfPolicy(unittest.TestCase):
    """Prove topology and PerfStats jointly control segmentation admission."""

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
                    tags={
                        "bytes": "48",
                        "authority": "immutable_scheduler_snapshot",
                        "state_payload": "false",
                    },
                )
            ]
        )
        self.assertIsNone(result.error)
        self.assertEqual(
            result.scheduler_dispatch_operations,
            ("device_generation_dispatch_ticket_d2h_submissions",),
        )

    def test_gpu_host_transfer_policy_rejects_noncanonical_dispatch_ticket(
        self,
    ) -> None:
        """Backend, ABI size, and no-state authority tags are fail-closed."""

        canonical_tags = {
            "bytes": "48",
            "authority": "immutable_scheduler_snapshot",
            "state_payload": "false",
        }
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
                    "execution": "native_device_controlled_switch_while",
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
                    "execution": "single_async_native_switch_while_launch",
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
                tags={
                    "bytes": "48",
                    "authority": "immutable_scheduler_snapshot",
                    "state_payload": "false",
                },
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
            "validate_gpu_host_transfer_policy(records)",
            harness,
        )

    def test_qwen36_homogeneous_tp_cells_require_prefill_graph_probe(self) -> None:
        """Same-backend TP must not retain the retired prefill opt-out."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        homogeneous_qwen36_tp_rows = [
            line
            for line in harness.splitlines()
            if "SUITES+=" in line
            and "qwen36-moe-" in line
            and re.search(r"-(?:cuda2tp|rocm2tp|rocm4tp)\|", line)
        ]
        self.assertGreater(len(homogeneous_qwen36_tp_rows), 0)
        for row in homogeneous_qwen36_tp_rows:
            self.assertNotIn("no-prefill-graph-buckets", row)
            self.assertIn("prefill-graph-probe", row)

    def test_qwen36_moe_cells_name_residency_maintenance_explicitly(self) -> None:
        """Do not let a CLI default silently change an independent policy axis."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'S9_STATIC_FLAGS="--moe-residency-maintenance off"',
            harness,
        )
        qwen36_moe_rows = [
            line
            for line in harness.splitlines()
            if "SUITES+=" in line and "qwen36-moe-" in line
        ]
        self.assertGreater(len(qwen36_moe_rows), 0)
        for row in qwen36_moe_rows:
            has_explicit_mode = any(
                marker in row
                for marker in (
                    "${S9_STATIC_FLAGS}",
                    "--moe-residency-maintenance dynamic",
                    "--moe-residency-maintenance off",
                    "${S9_DYNAMIC_RESIDENCY_FLAGS}",
                )
            )
            self.assertTrue(
                has_explicit_mode,
                f"Qwen3.6 MoE matrix row inherits an ambiguous CLI mode: {row}",
            )

    def test_llep_cells_declare_economical_policy_tuple_explicitly(self) -> None:
        """Current-batch LLEP must not imply expert or verifier replication."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        for backend in ("CUDA", "ROCM"):
            definition = next(
                line
                for line in harness.splitlines()
                if line.startswith(f"    S9_LLEP_OVERLAY_{backend}2_FLAGS=")
            )
            self.assertIn("--moe-continuation-dense-policy tensor-parallel", definition)
            self.assertIn("--mtp-terminal-head-policy mirrored-full-vocabulary", definition)
            self.assertIn("routed_compute=apportioned", definition)
            self.assertIn(
                "routed_phase=uniform",
                definition,
            )
            self.assertIn(
                "routed_decode_assignment=static-owner",
                definition,
            )
            self.assertIn(
                "routed_prefill_assignment=least-loaded-resident",
                definition,
            )
            self.assertNotIn("routed_assignment=", definition)

        llep_rows = [
            line
            for line in harness.splitlines()
            if "SUITES+=" in line and "qwen36-moe-llep-" in line
        ]
        self.assertGreater(len(llep_rows), 0)
        for row in llep_rows:
            self.assertRegex(row, r"\$\{S9_LLEP_OVERLAY_(?:CUDA|ROCM)2_FLAGS\}")
            self.assertNotRegex(row, r"\$\{S9_OVERLAY_(?:CUDA|ROCM)2_FLAGS\}")
            self.assertIn("--moe-residency-maintenance off", row)
            self.assertIn("${S9_LLEP_MOVEMENT_FLAGS}", row)

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

    def test_release_ci_has_combined_dynamic_llep_prefix_mtp_gpu_cells(
        self,
    ) -> None:
        """Release CI must expand the two strongest MTP MoE cells per GPU."""

        repo_root = Path(__file__).resolve().parents[4]
        harness = repo_root / "scripts" / "ci" / "run_release_container_e2e.sh"
        required_models = set(
            re.findall(
                r'require_model "([^"]+)"',
                harness.read_text(encoding="utf-8"),
            )
        )

        with tempfile.TemporaryDirectory() as models_dir:
            for model in required_models:
                (Path(models_dir) / model).touch()

            for backend in ("cuda", "rocm"):
                completed = subprocess.run(
                    [
                        str(harness),
                        "--variant",
                        backend,
                        "--image",
                        "unused:test-image",
                        "--models-dir",
                        models_dir,
                        "--dry-run",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                command = shlex.split(
                    completed.stdout.removeprefix(
                        "[run-release-container-e2e] "
                    )
                )
                suites = [
                    command[index + 1]
                    for index, token in enumerate(command[:-1])
                    if token == "--suite"
                ]
                for mode in ("dynamic", "llep"):
                    label = (
                        f"qwen36-moe-{mode}-prefix-mtp-stochastic-d4to15-"
                        f"{backend}2tp-full"
                    )
                    matching = [
                        suite
                        for suite in suites
                        if f"|{label}|" in suite
                    ]
                    self.assertEqual(len(matching), 1, label)
                    _, cell_backend, _, flags, _, options = matching[0].split(
                        "|", 5
                    )
                    self.assertEqual(cell_backend, "tp")
                    self.assertIn("--prefix-cache", flags)
                    self.assertIn("--mtp", flags)
                    self.assertIn(
                        "--mtp-verify-mode speculative-sampling",
                        flags,
                    )
                    self.assertIn("--mtp-depth-policy dynamic", flags)
                    self.assertIn("--mtp-max-draft-tokens 15", flags)
                    expected_maintenance = "dynamic" if mode == "dynamic" else "off"
                    self.assertIn(
                        f"--moe-residency-maintenance {expected_maintenance}",
                        flags,
                    )
                    self.assertNotIn("--tp-devices", flags)
                    self.assertIn(
                        "--moe-routed-expert-domain "
                        f"qwen36_moe_{backend}_hot="
                        f"{backend}:0,{backend}:1;",
                        flags,
                    )
                    self.assertIn("routed_compute=apportioned", flags)
                    self.assertIn("routed_phase=uniform", flags)
                    self.assertIn(
                        "routed_decode_assignment=static-owner",
                        flags,
                    )
                    if mode == "llep":
                        self.assertIn(
                            "routed_prefill_assignment=least-loaded-resident",
                            flags,
                        )
                        self.assertIn(
                            "--mtp-terminal-head-policy "
                            "mirrored-full-vocabulary",
                            flags,
                        )
                    else:
                        self.assertIn(
                            "routed_prefill_assignment=static-owner",
                            flags,
                        )
                    self.assertNotIn("routed_assignment=", flags)
                    self.assertIn("prefill-graph-probe", options)
                    self.assertIn(
                        "prefix-cache-rebalance-clear-probe",
                        options,
                    )
                    self.assertIn(
                        "moe-rebalance-movement-probe",
                        options,
                    )
                    self.assertIn("stochastic-mtp-probe", options)
                    self.assertNotIn("no-long-context", options)

    def test_default_gpu_rebalance_matrix_uses_stochastic_mtp(self) -> None:
        """The strongest four local-GPU MoE cells exercise production sampling."""

        harness = (SERVER_E2E_DIR / "test_server_e2e.sh").read_text(
            encoding="utf-8"
        )
        rows = [
            line
            for line in harness.splitlines()
            if "SUITES+=" in line
            and "prefix-mtp-stochastic-d4to15-" in line
        ]
        self.assertEqual(len(rows), 4)
        for backend in ("cuda2tp", "rocm2tp"):
            for mode in ("dynamic", "llep"):
                matching = [
                    row
                    for row in rows
                    if f"qwen36-moe-{mode}-prefix-mtp-" in row
                    and f"-{backend}|" in row
                ]
                self.assertEqual(len(matching), 1)
                row = matching[0]
                self.assertIn("${S9_STOCHASTIC_MTP_FLAGS}", row)
                self.assertIn("stochastic-mtp-probe", row)

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
                tags={"capture_phase": "warmup"},
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
        """Warmup, capture, and replay evidence closes the prefill contract."""

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
            for phase in ("warmup", "capture", "replay")
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
                    "collective_segmented": "true",
                    "replay_plan_policy": (
                        "allow_heterogeneous_collective_segmentation"
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
                    "collective_segmented": "false",
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
                "hot=cuda:0,cuda:1;scope=local "
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


if __name__ == "__main__":
    unittest.main()
