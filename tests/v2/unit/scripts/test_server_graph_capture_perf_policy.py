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
sys.path.insert(0, str(SERVER_E2E_DIR))

from graph_capture_perf_policy import (  # noqa: E402
    device_kinds_for_cell,
    validate_graph_capture_policy,
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
) -> dict[str, object]:
    """Build the minimal PerfStats counter shape consumed by the validator."""

    return {
        "name": name,
        "domain": domain,
        "value": value,
        "tags": tags or {},
    }


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
                        f"qwen36-moe-{mode}-prefix-mtp-greedy-d2-"
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
                    self.assertIn(f"--moe-rebalance {mode}", flags)
                    self.assertNotIn("--tp-devices", flags)
                    self.assertIn(
                        "--moe-routed-expert-domain "
                        f"qwen36_moe_{backend}_hot="
                        f"{backend}:0,{backend}:1;",
                        flags,
                    )
                    self.assertIn("prefill-graph-probe", options)
                    self.assertIn(
                        "prefix-cache-rebalance-clear-probe",
                        options,
                    )
                    self.assertIn(
                        "moe-rebalance-movement-probe",
                        options,
                    )
                    self.assertNotIn("no-long-context", options)

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
                tags={"context": "main_decode", "phase": "warmup"},
            ),
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
                tags={"context": "main_decode", "phase": "warmup"},
            ),
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

    def test_helper_executable_cannot_mask_repeated_verifier_warmup(self) -> None:
        """Context attribution must expose a verifier that never captures."""

        records = [
            counter(
                "decode_graph_phase",
                value=8.0,
                tags={"context": "main_verifier", "phase": "warmup"},
            ),
            counter(
                "decode_graph_phase",
                tags={"context": "mtp_helper", "phase": "warmup"},
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

    def test_two_invocations_require_context_matched_capture(self) -> None:
        """A second invocation must advance warmup to a real capture."""

        records = [
            counter(
                "decode_graph_phase",
                value=2.0,
                tags={"context": "main_decode", "phase": "warmup"},
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
        self.assertIn("missing capture", result.error or "")

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
                "device_input_reuse_publications",
                value=4.0,
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
        self.assertIn("consumed admissions (4) != releases (3)", result.error or "")

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
                "device_input_reuse_publications",
                domain="request_admission",
            ),
        ]
        result = validate_request_input_lifetime_policy(records)
        self.assertIn("no release-to-next-writer event waits", result.error or "")


if __name__ == "__main__":
    unittest.main()
