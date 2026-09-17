#!/usr/bin/env python3
"""Device-free adversarial tests for benchmark and image certification policy.

Mocked processes prove orchestration decisions, not inference performance.
Prefix-state obligations come from typed model declarations; exact output
tokens cannot excuse missing recurrent state or a response-selected contract.
The real local CI run remains the only source of image certificates.
Inventory regressions include untagged models: a complete E2E projection
cannot stand in for the full model inventory or omit its shard identity pins.
Container mount translation must preserve that inventory exactly and reject
path traversal instead of silently selecting a different model namespace.
Native acquisition audits check original responses without approving baselines;
failed retries, stale configurations and duplicate passes remain visible.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import re
import shlex
import signal
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import docker_paths
import cross_host_artifacts as remote_artifacts
import generation_tokens as generation
import generation_regression_http as generation_http
import generation_corpus
import audit_generation_acquisition as acquisition
import export_generation_controls as control_export
import run_model_parity_generation as generation_runner
import prebuilt_test_image
import production_artifacts as artifacts
import run_model_parity_benchmarks as benchmark
import run_production_pipeline as pipeline
import run_production_cross_host_e2e as remote_e2e
import cross_host_containers as remote_containers
import cross_host_network as remote_network
import run_production_parity_campaigns as parity
import run_production_prerequisites as prerequisites_runner
from test_generation_movement_ledger import empty_ledger, ledger as movement_ledger
from test_model_parity_inventory import remote_cases


def cell(name="cell"):
    """Minimal tagged configuration with a complete split-model manifest."""
    runtime = GenerationHTTPTests().record()["runtime"]
    runtime["generation"]["serial_control_id"] = name
    return {"case": name, "campaign": "campaign", "backends": "CUDA+ROCm",
            "model_files": ["/src/models/model-1.gguf", "/src/models/model-2.gguf"],
            "configuration": {"model_parity_schema": 1, "id": name,
                "cross_host_e2e": [], "runtime": runtime,
                "model": "/src/models/model-1.gguf", "e2e": {"context_length": 8192,
                    "server_args": ["--define-domain", "arbitrary;devices=rocm:0,cuda:0", "--mtp"]}}}


def manifest():
    return {"schema": 1, "scope": "e2e", "source_revision": "revision", "cells": [cell()]}


def full_manifest():
    """Include a non-E2E model with independent shards, without another matrix."""
    untagged = cell("untagged")
    untagged["model_files"] = ["/src/models/other-1.gguf", "/src/models/other-2.gguf"]
    untagged["configuration"].update(model=untagged["model_files"][0], e2e=None)
    return {**manifest(), "scope": "all", "cells": [cell(), untagged]}


def parity_report():
    """Complete mock numerical evidence uses the same full fixture membership."""
    return {"correctness_passed": True, "performance_requirements_met": True,
            "artifact_contract_passed": True, "preflight_return_code": 0,
            "preflight_tests": ["V2_Unit_A", "V2_Integration_B"], "preflight_test_count": 2,
            "exact_matrix_cell_count": len(full_manifest()["cells"]),
            "campaigns": [{"campaign": "campaign", "return_code": 0, "outcome": "completed",
                "artifact_contract_passed": True,
                "gtest_cases": [row["case"] for row in full_manifest()["cells"]]}]}


def prerequisite_report():
    """Canonical model-free receipt, separate from mathematical diagnostics."""
    return {"preflight_return_code": 0, "preflight_elapsed_seconds": 1,
            "preflight_build_directory": "/src/build_v2_integration", "preflight_completed_ns": 123,
            "preflight_tests": ["V2_Unit_A", "V2_Integration_B"], "preflight_test_count": 2}


def generation_report(inventory=None, isa="AVX512", image="runtime-id"):
    """Synthetic scheduling evidence; token/protocol validation has separate tests."""
    inventory = inventory or full_manifest()
    rows = [row for row in inventory["cells"] if generation_corpus.InventoryScope.GENERATION.accepts(row["configuration"])]
    return {"schema": 1, "mode": "regression", "complete": True, "passed": True,
            "certification_eligible": True, "image": image, "cpu_isa": isa, "corpus_digest": "c" * 64,
            "source_revision": inventory["source_revision"], "inventory_digest": artifacts.digest(inventory),
            "prerequisite_report_digest": artifacts.digest(prerequisite_report()), "selected": len(rows),
            "cells": [{"case": row["case"], "configuration": copy.deepcopy(row["configuration"]),
                       "return_code": 0, "evidence_error": None} for row in rows]}


def install_corpus_io_fixture(test, *, mock_stat=False):
    """Mock only external corpus/model I/O; keep report admission real.

    ApprovedGenerationCorpusTests independently exercise actual files, approval
    catalogs, full configuration binding, provenance and token validation.
    These pipeline tests must not depend on a downloaded corpus or real GGUFs.
    """
    loader = patch.object(pipeline.ApprovedGenerationCorpus, "load_reviewed",
                          return_value=argparse.Namespace(pin=argparse.Namespace(document_digest="c" * 64)))
    loader.start()
    test.addCleanup(loader.stop)
    if mock_stat:
        stat = patch.object(pipeline, "model_identities", return_value={})
        stat.start()
        test.addCleanup(stat.stop)


def image_pair(source, isa="AVX512"):
    """Independent mock Docker metadata for the test and Release siblings."""
    common = {"org.opencontainers.image.revision": source["revision"],
              "org.llaminar.source_tree": source["tree"], "org.llaminar.cpu_isa": isa,
              "org.llaminar.build_type": "Release", "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON"}
    return {"builder": {"id": "builder-id", "layers": ["test-layer"],
                        "labels": {**common, "org.llaminar.image_role": "builder",
                                   "org.llaminar.integration_skipped": "0"}},
            "runtime": {"id": "runtime-id", "layers": ["runtime-layer"],
                        "labels": {**common, "org.llaminar.image_role": "runtime"}}}


def remote_runtime_identity(*, identifier="runtime-id", isa="AVX2", revision="revision",
                            source_tree="source-tree"):
    """Build the full label contract admitted before a remote CPU image transfer."""
    return {"id": identifier, "labels": {
        "org.opencontainers.image.revision": revision,
        "org.llaminar.source_tree": source_tree,
        "org.llaminar.cpu_isa": isa,
        "org.llaminar.image_role": "runtime",
        "org.llaminar.build_type": "Release",
        "org.llaminar.cuda": "ON",
        "org.llaminar.rocm": "ON",
    }}


def e2e_report(inventory=None):
    """A passing report explicitly bound to its entire manifest and image."""
    inventory = inventory or manifest()
    return {"schema": 1, "image": "runtime-id", "source_revision": "revision",
            "manifest_digest": artifacts.digest(inventory), "correctness_passed": True,
            "selected": len(inventory["cells"]),
            "cells": [{"case": row["case"], "return_code": 0, "outcome": "passed",
                       "configuration": copy.deepcopy(row["configuration"])} for row in inventory["cells"]]}


def non_remote_documents(inventory, image="runtime-id"):
    """Give untagged fixtures the same explicit empty-phase receipt as CI."""
    remote = pipeline.cross_host_e2e_projection(inventory, inventory["source_revision"])
    assert not remote["cells"], "remote fixtures must supply executed-scenario evidence explicitly"
    return {"cross-host-manifest": remote,
            "cross-host-e2e": {"schema": 1, "eligible": False, "complete": True,
                "source_revision": inventory["source_revision"], "image": image,
                "manifest_digest": artifacts.digest(remote), "scenarios": [],
                "all_resources_retired": True}}


def workload():
    return json.loads((ROOT / "benchmarks/production/workload.json").read_text())


def measurement():
    """A complete runtime JSON with deliberately unsorted sample times."""
    work = workload()
    prompt = benchmark.validate_workload(work).encode()
    return {"schema": "llaminar.benchmark.v1", "success": True,
            "prefill_success": True, "decode_success": True,
            "measurement_iterations": 3, "warmup_iterations": 1,
            "prompt": {"sha256": hashlib.sha256(prompt).hexdigest(), "bytes": len(prompt)},
            "iterations": [{"tokens": {"prefill": 512, "decode": 256},
                            "throughput_tokens_per_sec": {"prefill": value, "decode_after_prefill": value / 10}}
                           for value in (110, 90, 100)]}


def certified_variants(directory, source, baseline):
    """Create internally consistent mock evidence for both publication lanes."""
    finals = {}
    for isa in pipeline.SHIPPING_ISAS:
        path = directory / isa.lower()
        inventory = {**manifest(), "source_revision": source["revision"]}
        images = image_pair(source, isa)
        images["runtime"]["id"] = "sha256:" + isa
        images["builder"]["id"] = "builder-" + isa
        evidence = {**e2e_report(inventory), "source_revision": source["revision"],
                    "image": images["runtime"]["id"]}
        row = {"case": "cell", "identity": {"cpu_isa": isa},
               "tokens_per_second": {"prefill": 100, "decode": 10}}
        documents = {"manifest": inventory,
            "all-cells": {**full_manifest(), "source_revision": source["revision"]}, "e2e": evidence,
            "prerequisites/prerequisites": prerequisite_report(),
            "generation/report": generation_report({**full_manifest(), "source_revision": source["revision"]},
                                                    isa, images["runtime"]["id"]),
            "benchmarks": {"passed": True, "complete": True, "diagnostic": False,
                           "image": images["runtime"]["id"], "manifest_digest": artifacts.digest(inventory),
                           "e2e_report_digest": artifacts.digest(evidence), "cells": [row],
                           "baseline_digest": artifacts.digest(baseline)}}
        documents.update(non_remote_documents(documents["all-cells"], images["runtime"]["id"]))
        for name, document in documents.items():
            artifacts.write_json(path / f"{name}.json", document)
        final = {"tag": "registry/image:" + isa.lower(), "id": "sha256:certified-" + isa,
                 "certificate": pipeline.certificates(source, images, path, cpu_isa=isa)}
        artifacts.write_json(path / "pipeline.json", {"certified": True, "images": images, "final": final,
            "identity": {"corpus_root": str(ROOT / "corpora"), "diagnostic_mathematical_parity": False}})
        finals[isa] = final
    return finals


class GenerationTokenTests(unittest.TestCase):
    """Exact token regression must neither hide drift nor certify distributions."""

    def response(self):
        """A terminal response includes the stop token hidden from visible text."""
        return {"object": "chat.completion", "token_ids": {"prompt": [7, 8], "completion": [10, 11, 0]},
                "choices": [{"index": 0, "message": {"content": "same text"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

    def test_response_preserves_raw_ids_and_termination(self):
        trace = generation.TokenTrace.from_response(self.response(), requested_max_tokens=10)
        self.assertEqual(trace, generation.TokenTrace((7, 8), (10, 11, 0), "stop"))
        self.assertIsNone(generation.compare_tokens(trace, trace))

    def test_invalid_or_missing_token_evidence_cannot_pass_via_matching_text(self):
        for mutate in (lambda r: r.pop("token_ids"), lambda r: r.update(object="chat.completion.chunk"),
                       lambda r: r.update(error="failure"), lambda r: r["token_ids"].update(prompt=[]),
                       lambda r: r["usage"].update(total_tokens=4),
                       lambda r: r["usage"].update(completion_tokens=True),
                       lambda r: r["choices"][0].update(finish_reason=None),
                       lambda r: r["choices"][0].update(finish_reason="length"),
                       lambda r: r["choices"].append(r["choices"][0])):
            response = self.response()
            mutate(response)
            with self.subTest(response=response), self.assertRaises(ValueError):
                generation.TokenTrace.from_response(response, requested_max_tokens=10)
        for token in (True, 1.0, "1", -1, 2147483648):
            response = self.response()
            response["token_ids"]["completion"][1] = token
            with self.subTest(token=token), self.assertRaises(ValueError):
                generation.TokenTrace.from_response(response, requested_max_tokens=10)

    def test_every_position_length_and_finish_reason_are_checked(self):
        expected = generation.TokenTrace((7, 8), tuple(range(2048)), "length")
        for position in (0, 31, 511, 1023, 2047):
            tokens = list(expected.completion)
            tokens[position] = 3000
            actual = generation.TokenTrace(expected.prompt, tuple(tokens), "length")
            mismatch = generation.compare_tokens(expected, actual)
            self.assertEqual(mismatch.phase, generation.TokenTracePhase.COMPLETION)
            self.assertEqual(mismatch.position, position)
            self.assertEqual(mismatch.expected, position)
            self.assertEqual(mismatch.observed, 3000)
        for tokens in (expected.completion[:-1], (*expected.completion, 2048), ()):
            mismatch = generation.compare_tokens(expected, generation.TokenTrace(expected.prompt, tokens, "stop"))
            self.assertEqual(mismatch.position, min(len(tokens), len(expected.completion)))
        mismatch = generation.compare_tokens(expected, generation.TokenTrace((7, 9), expected.completion, "length"))
        self.assertEqual(mismatch.phase, generation.TokenTracePhase.PROMPT)
        mismatch = generation.compare_tokens(expected, generation.TokenTrace(expected.prompt, expected.completion, "stop"))
        self.assertEqual(mismatch.phase, generation.TokenTracePhase.TERMINATION)

    def test_continuous_horizon_rejects_short_matching_or_early_eos_runs(self):
        workload = generation.GenerationWorkload.from_record({"runtime": {
            "generation": {"max_tokens": 512, "minimum_completion_tokens": 384}}})
        for count in (0, 128, 200, 383, 384, 512):
            response = self.response()
            response["token_ids"]["completion"] = [10] * count
            response["usage"].update(completion_tokens=count, total_tokens=count + 2)
            for finish in ("stop", "tool_calls", "length"):
                response["choices"][0]["finish_reason"] = finish
                with self.subTest(count=count, finish=finish):
                    if count < 384 or (finish == "length" and count != 512):
                        with self.assertRaises(ValueError):
                            workload.observe(response)
                    else:
                        self.assertEqual(len(workload.observe(response).completion), count)
        # Three identical 128-token traces still cannot establish one 384-token
        # causal horizon. Lists are not accepted as a substitute response.
        with self.assertRaises(ValueError):
            workload.observe([self.response()] * 3)

    def test_generation_workload_has_no_python_default_or_test_name_heuristic(self):
        for record in (None, {}, {"runtime": None}, {"runtime": {}},
                       {"id": "MTPDepth15", "runtime": {"generation": {"max_tokens": 384}}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                generation.GenerationWorkload.from_record(record)
        for maximum, minimum in ((384, 0), (383, 384), (True, 1), (384, False),
                                 (384.0, 384), (384, "384")):
            with self.subTest(maximum=maximum, minimum=minimum), self.assertRaises(ValueError):
                generation.GenerationWorkload(maximum, minimum)
        # The typed producer may request a longer horizon; the reader preserves
        # it verbatim instead of introducing its own table of supported lengths.
        self.assertEqual(generation.GenerationWorkload.from_record({"runtime": {
            "generation": {"max_tokens": 2048, "minimum_completion_tokens": 1536}}}),
            generation.GenerationWorkload(2048, 1536))

    def test_delayed_drift_and_last_token_drift_fail_the_384_token_horizon(self):
        workload = generation.GenerationWorkload(384, 384)
        response = self.response()
        response["token_ids"]["completion"] = list(range(384))
        response["usage"].update(completion_tokens=384, total_tokens=386)
        response["choices"][0]["finish_reason"] = "length"
        expected = workload.observe(response)
        for position in (199, 200, 201, 382, 383):
            changed = copy.deepcopy(response)
            changed["token_ids"]["completion"][position] += 1
            mismatch = generation.compare_tokens(expected, workload.observe(changed))
            self.assertIsNotNone(mismatch)
            self.assertEqual(mismatch.position, position)
            self.assertEqual(mismatch.phase, generation.TokenTracePhase.COMPLETION)


class GenerationHTTPTests(unittest.TestCase):
    """Public requests and persisted controls reject self-certification and drift."""

    def record(self, mtp=False):
        """Small synthetic messages exercise protocol only, never model quality."""
        bodies = [{"messages": [{"role": "system", "content": "shared prefix"},
                                 {"role": "user", "content": suffix}], "max_tokens": 384,
                   "seed": 4242, "temperature": 0.7, "top_k": 40, "top_p": 0.9,
                   "return_token_ids": True, "return_runtime_summary": True} for suffix in ("harbor", "mountain")]
        bodies[1]["messages"] = [*copy.deepcopy(bodies[0]["messages"]),
                                  {"role": "assistant", "content": "mountain"}]
        requests = [{"id": name, "prefix": role, "body": copy.deepcopy(bodies[variant])}
                    for name, role, variant in (("seed", "fresh", 0), ("hit", "full", 0),
                                               ("partial", "partial", 1), ("hit_b", "full", 1))]
        return {"model_parity_schema": 1, "model": "/models/model.gguf", "id": "mtp" if mtp else "serial",
                "e2e": None, "runtime": {"context_length": 4096, "movement_evidence": "not_applicable",
                    "server_args": ["opaque topology"], "generation": {"max_tokens": 384,
                    "minimum_completion_tokens": 384, "readiness_timeout_seconds": 60,
                    "mtp_policy": "depth_2" if mtp else "off", "serial_control_id": "serial",
                    "mtp_verify_mode": "speculative-sampling",
                    "prefix_state": "hybrid_recurrent",
                    "requests": requests}}}

    def response(self, url, body, timeout):
        """Return distinct streams for distinct prompts, exact for repeats."""
        self.assertGreater(timeout, 0)
        variant = int(body["messages"][-1]["content"] == "mountain")
        prompt = [7, 8, 0] + ([9, 10] if variant else [])
        # A new mocked server handles each ordered four-request sequence.
        phase = getattr(self, "response_index", 0) % 4
        self.response_index = phase + 1
        matched = (0, 3, 3, 5)[phase]
        policy = getattr(self, "mtp_policy", generation_http.MTPPolicy.OFF)
        active = policy is not generation_http.MTPPolicy.OFF
        dynamic = policy is generation_http.MTPPolicy.DYNAMIC
        depth = {generation_http.MTPPolicy.OFF: 1, generation_http.MTPPolicy.DEPTH1: 1,
                 generation_http.MTPPolicy.DEPTH2: 2, generation_http.MTPPolicy.DEPTH3: 3,
                 generation_http.MTPPolicy.DEPTH15: 15, generation_http.MTPPolicy.DYNAMIC: 2}[policy]
        return {"object": "chat.completion", "token_ids": {"prompt": prompt,
                "completion": [variant * 1000 + index for index in range(384)]},
                "runtime_summary": {"schema": 1, "expert_movement": empty_ledger(),
                    "expert_movement_topology": {"schema": 1, "scope": "model_lifetime", "authority": "none", "available_axes": []}, "mtp": {
                    "enabled": active, "bypassed": False, "bypass_reason": "",
                    "stochastic_verify": active, "verify_mode": "speculative-sampling" if active else "greedy",
                    "adaptive_depth_enabled": dynamic, "depth_policy_mode": "dynamic" if dynamic else "fixed",
                    "current_depth": depth, "min_depth": 1 if dynamic else depth, "max_depth": 15 if dynamic else depth,
                    "depth_policy_updates": 128 if dynamic else 0,
                    "draft_steps": 384 if active else 0, "verifier_runs": 128 if active else 0,
                    "verifier_token_count": 384 if active else 0, "accepted_tokens": 128 if active else 0,
                    "stochastic_accept_tests": 384 if active else 0}, "prefix_cache": {
                    "enabled": True, "bypassed": False, "bypass_reason": "",
                    "hit": phase in (1, 3), "partial_hit": phase == 2,
                    "requested_tokens": len(prompt), "matched_tokens": matched, "matched_blocks": int(matched > 0),
                    "terminal_logits_restored": phase in (1, 3), "terminal_hidden_restored": phase in (1, 3),
                    "mtp_state_restored": active and phase != 0, "hybrid_state_restored": phase != 0,
                    "storage_tier": "ram" if phase else "none", "admission_epoch_earliest": 0,
                    "admission_epoch_latest": 0, "completion_movement_epoch": 0}},
                "choices": [{"finish_reason": "length"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 384, "total_tokens": len(prompt) + 384}}

    def collect(self, directory):
        """Acquire only an unapproved serial observation using the real comparator."""
        self.mtp_policy = generation_http.MTPPolicy.OFF
        with patch.object(generation_http, "post_completion", side_effect=self.response):
            return generation_http.run_probes(self.record(), "http://unused", directory)

    def test_completed_http_response_survives_a_hard_watchdog_kill(self):
        """A later stuck request must not erase already returned runtime proof.

        Use a real process death, not an exception that runs Python finally
        blocks. The mock transport blocks only after the first valid response
        has been processed. The surviving document must remain incomplete and
        cannot be admitted as a passing serial control.
        """
        context = multiprocessing.get_context("fork")
        receive, send = context.Pipe(duplex=False)
        release = context.Event()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "interrupted"

            def worker():
                """Exercise the real reporter with a bounded mock HTTP stall."""
                calls = 0
                def response(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        send.send("second-request-started")
                        release.wait(5)
                        raise RuntimeError("test did not retire its stalled child")
                    return self.response(*args)
                with patch.object(generation_http, "post_completion", side_effect=response):
                    generation_http.run_probes(self.record(), "http://unused", output)

            process = context.Process(target=worker)
            process.start()
            try:
                self.assertTrue(receive.poll(5), "first response did not finish")
                self.assertEqual(receive.recv(), "second-request-started")
                process.kill()
                process.join(5)
                self.assertEqual(process.exitcode, -signal.SIGKILL)
                path = output / "observations.json"
                self.assertTrue(path.is_file(), "hard timeout lost a completed HTTP response")
                observed = json.loads(path.read_text())
                self.assertFalse(observed["complete"])
                self.assertFalse(observed["certification_eligible"])
                self.assertFalse(observed["repeatability_passed"])
                self.assertEqual([row["id"] for row in observed["requests"]], ["seed"])
                self.assertEqual(len(observed["requests"][0]["response"]["token_ids"]["completion"]), 384)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    generation_http.observation_traces(self.record(), observed)
            finally:
                if process.is_alive():
                    process.kill()
                    process.join(5)
                receive.close()
                send.close()

    def test_non_mtp_early_eos_fails_live_and_saved_continuous_horizon(self):
        record = self.record()

        def short_response(*args):
            response = self.response(*args)
            response["token_ids"]["completion"] = response["token_ids"]["completion"][:13]
            response["usage"].update(completion_tokens=13, total_tokens=response["usage"]["prompt_tokens"] + 13)
            response["choices"][0]["finish_reason"] = "stop"
            return response

        with tempfile.TemporaryDirectory() as folder, patch.object(generation_http, "post_completion", side_effect=short_response):
            with self.assertRaisesRegex(ValueError, "insufficient continuous"):
                generation_http.run_probes(record, "http://unused", Path(folder) / "short")
        self.response_index = 0
        with tempfile.TemporaryDirectory() as folder:
            report = self.collect(Path(folder) / "complete")
        report["requests"][0]["response"] = short_response("unused", record["runtime"]["generation"]["requests"][0]["body"], 1)
        with self.assertRaisesRegex(ValueError, "insufficient continuous"):
            generation_http.observation_traces(record, report)

    def test_canonical_generation_rejects_caller_stop_or_tool_substitutions(self):
        record = self.record()
        for name, value in (("stop", ["end"]), ("tools", [{"type": "function"}]), ("tool_choice", "auto")):
            altered = copy.deepcopy(record)
            altered["runtime"]["generation"]["requests"][0]["body"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                generation_http.generation_profile(altered)

    def test_collect_and_compare_keep_exact_requests_without_certifying_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = self.collect(root / "serial")
            self.assertTrue(control["complete"])
            self.assertFalse(control["certification_eligible"])
            self.assertIsNone(control["serial_comparison_passed"])
            for policy in generation_http.MTPPolicy:
                self.mtp_policy = policy
                record = self.record(policy is not generation_http.MTPPolicy.OFF)
                record["runtime"]["generation"]["mtp_policy"] = policy.value
                with patch.object(generation_http, "post_completion", side_effect=self.response) as post:
                    report = generation_http.run_probes(record, "http://unused", root / policy.value, control)
                self.assertEqual(post.call_count, 4)
                self.assertTrue(report["serial_comparison_passed"])
                self.assertFalse(report["certification_eligible"])
                self.assertEqual([call.args[1] for call in post.call_args_list],
                                 [request["body"] for request in record["runtime"]["generation"]["requests"]])
            with self.assertRaises(FileExistsError):
                self.collect(root / "serial")

    def test_mtp_without_a_control_fails_before_http_or_artifact_creation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(generation_http, "post_completion") as post:
            path = Path(folder) / "missing"
            with self.assertRaisesRegex(ValueError, "requires its serial control"):
                generation_http.run_probes(self.record(True), "http://unused", path)
            self.assertFalse(path.exists())
            post.assert_not_called()

    def test_movement_obligation_is_required_before_http_admission(self):
        """Neither CLI defaults nor descriptive cell names can choose the gate."""
        for value in (None, "", "dynamic", True, 0, [], {}):
            record = self.record()
            record["runtime"]["movement_evidence"] = value
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder, patch.object(
                    generation_http, "post_completion") as post:
                with self.assertRaises(ValueError):
                    generation_http.run_probes(record, "http://unused", Path(folder) / "bad")
                post.assert_not_called()

    def test_missing_movement_journal_rejects_live_and_persisted_exact_tokens(self):
        """An older token-only pass cannot silently acquire new movement proof."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = self.collect(root / "serial")
            changed = copy.deepcopy(control)
            del changed["requests"][0]["response"]["runtime_summary"]["expert_movement"]
            with self.assertRaisesRegex(ValueError, "movement ledger"):
                generation_http.admit_control(self.record(True), changed)
            def missing(url, body, timeout):
                observed = self.response(url, body, timeout)
                del observed["runtime_summary"]["expert_movement"]
                return observed
            with patch.object(generation_http, "post_completion", side_effect=missing) as post:
                with self.assertRaisesRegex(ValueError, "movement ledger"):
                    generation_http.run_probes(self.record(), "http://unused", root / "bad")
                post.assert_called_once()
            observed = json.loads((root / "bad/observations.json").read_text())
            self.assertFalse(observed["complete"])
            self.assertEqual(len(observed["requests"]), 1)

    def test_static_rejects_completed_movement_even_when_tokens_match(self):
        """Completed ownership changes cannot be excused by a correct answer."""
        record = self.record()
        record["runtime"]["movement_evidence"] = "forbidden"
        def moved(url, body, timeout):
            observed = self.response(url, body, timeout)
            observed["runtime_summary"]["expert_movement"] = movement_ledger()
            return observed
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=moved) as post:
            with self.assertRaisesRegex(ValueError, "forbidden"):
                generation_http.run_probes(record, "http://unused", Path(folder) / "bad")
            post.assert_called_once()

    def test_dynamic_requires_cohort_movement_and_preserves_owner_history(self):
        """Late real movement is valid; missing or rewritten publication is not."""
        record = self.record()
        record["runtime"]["movement_evidence"] = "required"
        for authority in ("host", "device"):
            self.response_index = 0
            def moved(url, body, timeout):
                observed = self.response(url, body, timeout)
                observed["runtime_summary"]["expert_movement_topology"].update(
                    authority=authority, available_axes=["tier_residency", "participant_placement"])
                if self.response_index >= 3:
                    observed["runtime_summary"]["expert_movement"] = movement_ledger(authority)
                return observed
            with tempfile.TemporaryDirectory() as folder, patch.object(
                    generation_http, "post_completion", side_effect=moved):
                observed = generation_http.run_probes(record, "http://unused", Path(folder) / "good")
            self.assertEqual(len(generation_http.observation_traces(record, observed)), 4)
            for patch_ledger in (lambda x: x["economy"].clear(),
                                 lambda x: x["edges"][0].update(activation_count=1)):
                changed = copy.deepcopy(observed)
                patch_ledger(changed["requests"][3]["response"]["runtime_summary"]["expert_movement"])
                with self.subTest(authority=authority), self.assertRaisesRegex(ValueError, "movement"):
                    generation_http.observation_traces(record, changed)
        self.response_index = 0
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=self.response) as post:
            with self.assertRaisesRegex(ValueError, "required movement"):
                generation_http.run_probes(record, "http://unused", Path(folder) / "empty")
            self.assertEqual(post.call_count, 4)

    def test_first_mismatch_is_preserved_and_no_later_requests_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = self.collect(root / "serial")
            self.mtp_policy = generation_http.MTPPolicy.DEPTH2
            def drift(url, body, timeout):
                response = self.response(url, body, timeout)
                response["token_ids"]["completion"][200] += 1
                return response
            with patch.object(generation_http, "post_completion", side_effect=drift) as post:
                with self.assertRaisesRegex(ValueError, "token drift.*200"):
                    generation_http.run_probes(self.record(True), "http://unused", root / "bad", control)
            post.assert_called_once()
            evidence = json.loads((root / "bad/observations.json").read_text())
            self.assertFalse(evidence["complete"])
            self.assertEqual(evidence["requests"][0]["mismatch"]["position"], 200)

    def test_template_rewriting_a_complete_prefix_cannot_claim_partial_coverage(self):
        """Well-formed messages do not establish an identical encoded prefix."""
        def rewrite(url, body, timeout):
            response = self.response(url, body, timeout)
            if body["messages"][-1]["role"] == "assistant":
                response["token_ids"]["prompt"][2] = 999
            return response
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=rewrite) as post:
            with self.assertRaisesRegex(ValueError, "earlier complete token prefix"):
                generation_http.run_probes(self.record(), "http://unused", Path(folder) / "bad")
            self.assertEqual(post.call_count, 3)

    def test_control_revalidates_responses_not_just_success_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            control = self.collect(Path(folder) / "serial")
        for mutate in (lambda c: c.update(complete=False), lambda c: c["requests"].pop(),
                       lambda c: c["requests"].reverse(),
                       lambda c: c["requests"].append(c["requests"][0]),
                       lambda c: c["requests"][1]["response"]["token_ids"]["completion"].__setitem__(383, 9999),
                       lambda c: c["configuration"].update(model="wrong-model"),
                       lambda c: c["configuration"].update(id="mtp"),
                       lambda c: c["configuration"]["runtime"]["generation"].update(mtp_policy="depth_2")):
            changed = copy.deepcopy(control)
            mutate(changed)
            with self.subTest(control=changed), self.assertRaises(ValueError):
                generation_http.admit_control(self.record(True), changed)

    def test_canonical_request_geometry_and_sampler_are_mandatory(self):
        for mutate in (lambda r: r.update(requests=[]), lambda r: r.update(readiness_timeout_seconds=901),
                       lambda r: r.update(mtp_verify_mode="greedy"),
                       lambda r: r["requests"][0].update(prefix="full"),
                       lambda r: r["requests"][2].update(prefix="fresh"),
                       lambda r: r["requests"][2]["body"].update(
                           messages=copy.deepcopy(r["requests"][0]["body"]["messages"]), seed=7),
                       lambda r: r["requests"][1].update(id="seed"),
                       lambda r: r["requests"][0]["body"].update(seed=None),
                       lambda r: r["requests"][0]["body"].update(seed=0),
                       lambda r: r["requests"][0]["body"].update(seed=-1),
                       lambda r: r["requests"][0]["body"].update(seed=2**32),
                       lambda r: r["requests"][0]["body"].update(seed=True),
                       lambda r: r["requests"][0]["body"].update(max_tokens=383),
                       lambda r: r["requests"][0]["body"].update(temperature=0),
                       lambda r: r["requests"][0]["body"].update(temperature=float("nan")),
                       lambda r: r["requests"][0]["body"].update(return_token_ids=False),
                       lambda r: r["requests"][0]["body"].update(return_runtime_summary=False),
                       lambda r: r["requests"][0]["body"].update(stream=True)):
            record = self.record()
            mutate(record["runtime"]["generation"])
            with self.subTest(record=record), self.assertRaises(ValueError):
                generation_http.generation_profile(record)

    def test_exact_tokens_do_not_waive_missing_or_malformed_restore_evidence(self):
        """Re-admission validates physical outcomes even if every token matches."""
        with tempfile.TemporaryDirectory() as folder:
            control = self.collect(Path(folder) / "serial")
        for index, patch_prefix in ((0, {"hit": True}), (0, {"matched_tokens": 1}),
                (1, {"enabled": False}), (1, {"bypassed": True}), (1, {"hit": False}),
                (1, {"partial_hit": True}), (1, {"terminal_logits_restored": False}),
                (1, {"storage_tier": "none"}), (1, {"storage_tier": "invented"}),
                (1, {"requested_tokens": 4}), (1, {"matched_tokens": 2}),
                (1, {"matched_tokens": True}), (1, {"hit": 1}), (1, {"bypass_reason": "invalidated"}),
                (1, {"admission_epoch_earliest": 2}), (1, {"admission_epoch_latest": 2}),
                (1, {"completion_movement_epoch": -1}), (2, {"partial_hit": False}),
                (2, {"hit": True}), (2, {"matched_tokens": 1}), (2, {"matched_tokens": 5})):
            changed = copy.deepcopy(control)
            changed["requests"][index]["response"]["runtime_summary"]["prefix_cache"].update(patch_prefix)
            with self.subTest(index=index, patch=patch_prefix), self.assertRaises(ValueError):
                generation_http.admit_control(self.record(True), changed)
        for value in (None, {}, {"schema": True}, {"schema": 2}, {"schema": 1, "prefix_cache": None}):
            changed = copy.deepcopy(control)
            changed["requests"][1]["response"]["runtime_summary"] = value
            with self.subTest(summary=value), self.assertRaises(ValueError):
                generation_http.admit_control(self.record(True), changed)

    def test_restore_miss_stops_at_offending_response_even_with_matching_tokens(self):
        def miss(url, body, timeout):
            response = self.response(url, body, timeout)
            if self.response_index == 2:
                response["runtime_summary"]["prefix_cache"]["hit"] = False
            return response
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=miss) as post:
            path = Path(folder) / "bad"
            with self.assertRaisesRegex(ValueError, "restore did not occur"):
                generation_http.run_probes(self.record(), "http://unused", path)
            self.assertEqual(post.call_count, 2)
            evidence = json.loads((path / "observations.json").read_text())
            self.assertFalse(evidence["complete"])
            self.assertEqual(len(evidence["requests"]), 2)

    def test_partial_restore_early_eos_preserves_failure_without_sending_another_request(self):
        """Observed short replies and horizon edges cannot borrow prior tokens."""
        for count in (1, 56, 69, 85, 89, 97, 208, 357, 383):
            self.response_index = 0
            def early_stop(url, body, timeout):
                response = self.response(url, body, timeout)
                if self.response_index == 3:
                    response["token_ids"]["completion"] = response["token_ids"]["completion"][:count]
                    response["choices"][0]["finish_reason"] = "stop"
                    response["usage"].update(completion_tokens=count,
                                             total_tokens=count + len(response["token_ids"]["prompt"]))
                return response
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder, patch.object(
                    generation_http, "post_completion", side_effect=early_stop) as post:
                output = Path(folder) / "early-eos"
                with self.assertRaisesRegex(ValueError, f"observed {count} committed tokens"):
                    generation_http.run_probes(self.record(), "http://unused", output)
                self.assertEqual(post.call_count, 3)
                observed = json.loads((output / "observations.json").read_text())
                self.assertFalse(observed["complete"])
                self.assertEqual([row["response"]["usage"]["completion_tokens"] for row in observed["requests"]],
                                 [384, 384, count])
                self.assertTrue(observed["requests"][-1]["response"]["runtime_summary"]["prefix_cache"]["partial_hit"])

    def test_fresh_request_rejects_stale_restoration_flags_and_storage(self):
        """A cold request cannot inherit the preceding request's cache summary."""
        with tempfile.TemporaryDirectory() as folder:
            control = self.collect(Path(folder) / "serial")
        for fields in ({"terminal_logits_restored": True}, {"terminal_hidden_restored": True},
                       {"mtp_state_restored": True}, {"hybrid_state_restored": True},
                       {"matched_blocks": 1}, {"storage_tier": "ram"}):
            changed = copy.deepcopy(control)
            changed["requests"][0]["response"]["runtime_summary"]["prefix_cache"].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                generation_http.admit_control(self.record(True), changed)

    def test_every_mtp_policy_requires_shifted_state_on_full_and_partial_restore(self):
        """Exact tokens cannot certify omitted sidecar state or terminal hidden."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = self.collect(root / "serial")
            for policy in generation_http.MTPPolicy:
                if policy is generation_http.MTPPolicy.OFF:
                    continue
                self.mtp_policy = policy
                record = self.record(True)
                record["runtime"]["generation"]["mtp_policy"] = policy.value
                with patch.object(generation_http, "post_completion", side_effect=self.response):
                    observed = generation_http.run_probes(record, "http://unused", root / policy.value, control)
                # Both online admission and persisted-evidence admission use
                # the same validator. Replacing a flag must fail even though
                # every one of the 384 committed token IDs remains unchanged.
                for index, fields in ((1, {"mtp_state_restored": False}),
                                      (1, {"terminal_hidden_restored": False}),
                                      (2, {"mtp_state_restored": False}),
                                      (3, {"mtp_state_restored": False}),
                                      (3, {"terminal_hidden_restored": False})):
                    changed = copy.deepcopy(observed)
                    changed["requests"][index]["response"]["runtime_summary"]["prefix_cache"].update(fields)
                    with self.subTest(policy=policy, index=index, fields=fields), self.assertRaises(ValueError):
                        generation_http.observation_traces(record, changed)

    def test_hybrid_restore_is_required_independently_of_mtp(self):
        """All policies require main-model state at full and partial boundaries."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = self.collect(root / "serial")
            for policy in generation_http.MTPPolicy:
                self.mtp_policy = policy
                record = self.record(policy is not generation_http.MTPPolicy.OFF)
                record["runtime"]["generation"]["mtp_policy"] = policy.value
                with patch.object(generation_http, "post_completion", side_effect=self.response):
                    observed = generation_http.run_probes(record, "http://unused", root / policy.value, control)
                for index in (1, 2, 3):
                    changed = copy.deepcopy(observed)
                    changed["requests"][index]["response"]["runtime_summary"]["prefix_cache"]["hybrid_state_restored"] = False
                    with self.subTest(policy=policy, index=index), self.assertRaisesRegex(ValueError, "hybrid"):
                        generation_http.observation_traces(record, changed)

    def test_prefix_state_contract_cannot_be_omitted_or_inferred(self):
        """Neither a descriptive filename nor an MTP flag defines model state."""
        for value in (None, "", "unknown", True, 0, [], {}):
            record = self.record()
            record["model"] = "/models/hybrid-model.gguf"
            record["runtime"]["generation"]["prefix_state"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation_http.generation_profile(record)
        record = self.record()
        del record["runtime"]["generation"]["prefix_state"]
        with self.assertRaises(ValueError):
            generation_http.generation_profile(record)

    def test_kv_only_restore_contract_is_explicit_and_part_of_control_identity(self):
        """Attention-only models do not inherit hybrid state from other requests."""
        record = self.record()
        record["runtime"]["generation"]["prefix_state"] = "attention_kv"
        def attention_only(url, body, timeout):
            response = self.response(url, body, timeout)
            response["runtime_summary"]["prefix_cache"]["hybrid_state_restored"] = False
            return response
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=attention_only):
            observed = generation_http.run_probes(record, "http://unused", Path(folder) / "serial")
        self.assertTrue(observed["complete"])
        with self.assertRaisesRegex(ValueError, "workload/model"):
            generation_http.admit_control(self.record(True), observed)
        for index in (1, 2, 3):
            changed = copy.deepcopy(observed)
            changed["requests"][index]["response"]["runtime_summary"]["prefix_cache"]["hybrid_state_restored"] = True
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "attention-only"):
                generation_http.observation_traces(record, changed)

    def test_missing_hybrid_restore_fails_online_without_retry(self):
        """A token-exact response must fail at its first missing state boundary."""
        def missing_state(url, body, timeout):
            response = self.response(url, body, timeout)
            response["runtime_summary"]["prefix_cache"]["hybrid_state_restored"] = False
            return response
        with tempfile.TemporaryDirectory() as folder, patch.object(
                generation_http, "post_completion", side_effect=missing_state) as post:
            destination = Path(folder) / "missing-state"
            with self.assertRaisesRegex(ValueError, "hybrid"):
                generation_http.run_probes(self.record(), "http://unused", destination)
            self.assertEqual(post.call_count, 2)
            observed = json.loads((destination / "observations.json").read_text())
            self.assertFalse(observed["complete"])
            self.assertEqual(len(observed["requests"]), 2)

    def test_dynamic_policy_requires_actual_terminal_controller_activity(self):
        """A dynamic flag without a completed policy observation is no proof."""
        record = self.record(True)
        record["runtime"]["generation"]["mtp_policy"] = "dynamic"
        self.mtp_policy = generation_http.MTPPolicy.DYNAMIC
        response = self.response("http://unused", record["runtime"]["generation"]["requests"][0]["body"], 1)
        generation_http.validate_mtp_outcome(record["runtime"]["generation"], response)
        for value in (0, -1, True, None, 1.5):
            changed = copy.deepcopy(response)
            changed["runtime_summary"]["mtp"]["depth_policy_updates"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation_http.validate_mtp_outcome(record["runtime"]["generation"], changed)
        changed = copy.deepcopy(response)
        changed["runtime_summary"]["mtp"].pop("depth_policy_updates")
        with self.assertRaises(ValueError):
            generation_http.validate_mtp_outcome(record["runtime"]["generation"], changed)

    def test_terminal_mtp_evidence_rejects_bypass_wrong_policy_and_inactive_drafts(self):
        """A requested MTP mode is not proof that its stochastic work happened."""
        record = self.record(True)
        self.mtp_policy = generation_http.MTPPolicy.DEPTH2
        response = self.response("http://unused", record["runtime"]["generation"]["requests"][0]["body"], 1)
        profile = record["runtime"]["generation"]
        generation_http.validate_mtp_outcome(profile, response)
        for changed_fields in ({"enabled": False}, {"bypassed": True}, {"bypass_reason": "not greedy"},
                {"stochastic_verify": False}, {"verify_mode": "greedy"}, {"draft_steps": 0},
                {"verifier_runs": 0}, {"verifier_token_count": 0}, {"accepted_tokens": 0},
                {"stochastic_accept_tests": 0}, {"current_depth": 1}, {"min_depth": 3},
                {"max_depth": 15}, {"depth_policy_mode": "dynamic"}, {"adaptive_depth_enabled": True},
                {"draft_steps": True}, {"enabled": 1}, {"verifier_runs": -1}):
            changed = copy.deepcopy(response)
            changed["runtime_summary"]["mtp"].update(changed_fields)
            with self.subTest(fields=changed_fields), self.assertRaises(ValueError):
                generation_http.validate_mtp_outcome(profile, changed)
        changed = copy.deepcopy(response)
        changed["runtime_summary"].pop("mtp")
        with self.assertRaisesRegex(ValueError, "missing terminal MTP"):
            generation_http.validate_mtp_outcome(profile, changed)
        with self.assertRaisesRegex(ValueError, "serial control executed"):
            generation_http.validate_mtp_outcome(self.record()["runtime"]["generation"], response)

    def test_selectors_resolve_an_existing_control_without_synthesizing_argv(self):
        campaign = parity.CampaignCell("campaign", parity.CampaignGroup("CUDA", "ALL"))
        inventory = [(campaign, "Suite.serial", self.record()), (campaign, "Suite.mtp", self.record(True))]
        inventory[1][2]["runtime"]["generation"]["mtp_policy"] = "dynamic"
        args = argparse.Namespace(mode=generation_runner.RunMode.COLLECT, backend="CUDA", campaign=".*", cell="Suite.mtp")
        with patch.object(generation_runner, "discover", return_value=inventory):
            self.assertEqual(generation_runner.select_cells(args), [generation_runner.GenerationCell(
                *inventory[0], serial_control=inventory[0][2])])
            args.mode = generation_runner.RunMode.COMPARE
            self.assertEqual(generation_runner.select_cells(args), [generation_runner.GenerationCell(
                *inventory[1], serial_control=inventory[0][2])])
        with patch.object(generation_runner, "discover", return_value=inventory[1:]):
            with self.assertRaisesRegex(ValueError, "omitted serial control"):
                generation_runner.select_cells(args)

    def test_control_identity_includes_current_runtime_policy_not_only_its_name(self):
        """Changed defaults must not inherit an old same-name serial baseline."""
        with tempfile.TemporaryDirectory() as folder:
            observed = self.collect(Path(folder) / "serial")
        campaign = parity.CampaignCell("campaign", parity.CampaignGroup("CUDA", "ALL"))
        selected = generation_runner.GenerationCell(campaign, "Suite.mtp", self.record(True), self.record())
        self.assertEqual(len(selected.admit_control(observed)), 4)
        # These still pass the HTTP-only workload check, which cannot discover
        # another cell. The outer runner owns the complete inventory binding.
        for mutate in (lambda r: r["runtime"]["server_args"].append("changed-economy-policy"),
                       lambda r: r["runtime"].update(movement_evidence="required"),
                       lambda r: r["runtime"]["generation"].update(readiness_timeout_seconds=180)):
            canonical = self.record()
            mutate(canonical)
            changed = generation_runner.GenerationCell(campaign, "Suite.mtp", self.record(True), canonical)
            with self.subTest(configuration=canonical), self.assertRaisesRegex(ValueError, "current canonical control"):
                changed.admit_control(observed)
        with self.assertRaisesRegex(ValueError, "current canonical control"):
            selected.admit_control(None)

    def test_stale_control_fails_before_prerequisites_or_model_admission(self):
        """A stale observation must cost no device time and create no new run."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observed = self.collect(root / "observed")
            control_file = root / "control.json"
            artifacts.write_json(control_file, observed)
            canonical = self.record()
            canonical["runtime"]["server_args"].append("changed-policy")
            campaign = parity.CampaignCell("campaign", parity.CampaignGroup("CUDA", "ALL"))
            selected = generation_runner.GenerationCell(campaign, "Suite.mtp", self.record(True), canonical)
            output = root / "must-not-exist"
            with patch.object(generation_runner, "select_cells", return_value=[selected]), \
                    patch.object(generation_runner, "control_path", return_value=control_file), \
                    patch.object(parity, "run_production_parity_preflight") as preflight, \
                    patch.object(parity, "model_staging_workspace") as staging:
                with self.assertRaisesRegex(ValueError, "current canonical control"):
                    generation_runner.main(["--mode", "compare-controls", "--controls", str(root),
                                            "--output", str(output)])
                preflight.assert_not_called()
                staging.assert_not_called()
                self.assertFalse(output.exists())


class GenerationAcquisitionAuditTests(unittest.TestCase):
    """Count original native proof without turning a progress audit into approval."""

    def setUp(self):
        """Use the real HTTP observer with synthetic device-free server responses."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.http = GenerationHTTPTests(methodName="runTest")
        self.records = {name: self.http.record(name == "mtp") for name in ("serial", "mtp")}
        self.inventory = {"schema": 1, "scope": "all", "source_revision": "revision", "cells": [
            {"case": "Suite." + name, "campaign": "campaign", "backends": "CPU",
             "configuration": record, "model_files": [record["model"]]}
            for name, record in self.records.items()]}
        self.manifest = self.root / "inventory.json"
        artifacts.write_json(self.manifest, self.inventory)
        self.serial = self.http.collect(self.root / "serial/row/generation")
        self.http.mtp_policy = generation_http.MTPPolicy.DEPTH2
        with patch.object(generation_http, "post_completion", side_effect=self.http.response):
            generation_http.run_probes(self.records["mtp"], "http://unused",
                                      self.root / "mtp/row/generation", self.serial)
        self.reports = []
        for name in self.records:
            report = {"schema": 1, "mode": "collect-controls" if name == "serial" else "compare-controls",
                "source_revision": "revision", "certification_eligible": False, "complete": True,
                "passed": True, "selected": 1, "preflight_return_code": 0, "preflight_test_count": 2,
                "preflight_tests": ["V2_Unit_A", "V2_Integration_B"], "cells": [{
                    "case": "Suite." + name, "configuration": self.records[name], "return_code": 0,
                    "evidence_error": None, "artifacts": str(self.root / name / "row")}]}
            path = self.root / name / "report.json"
            artifacts.write_json(path, report)
            self.reports.append(path)

    def test_complete_counts_are_original_exact_proof_not_corpus_approval(self):
        result = acquisition.audit(self.manifest, self.reports)
        self.assertEqual(result["counts"], {kind: {"expected": 1, "passed": 1, "not_complete": 0}
                                            for kind in ("serial", "mtp")})
        self.assertTrue(result["acquisition_complete"])
        self.assertFalse(result["certification_eligible"])
        self.assertEqual(result["unresolved_failed_cases"], [])
        self.assertEqual(result["unseen_cases"], [])
        self.assertEqual(len(result["reports"]), 2)
        self.assertEqual(result["inventory_digest"], artifacts.digest(self.inventory))
        partial = acquisition.audit(self.manifest, self.reports[:1])
        self.assertFalse(partial["acquisition_complete"])
        self.assertEqual(partial["unseen_cases"], ["Suite.mtp"])

    def test_export_preserves_inputs_mapping_and_tokens_without_approval(self):
        sources = {self.records["serial"]["model"]: [1, 2, 4096, 3, 4]}
        with patch.object(control_export, "model_identities", return_value=sources):
            exported = control_export.export_controls(self.manifest, self.reports, "AVX512")
            with self.assertRaisesRegex(ValueError, "complete audited"):
                control_export.export_controls(self.manifest, self.reports[:1], "AVX512")
        self.assertEqual(exported["kind"], "unapproved_generation_controls")
        self.assertEqual([row["id"] for row in exported["controls"]], ["serial"])
        expected = generation_corpus.portable_inventory(self.inventory, sources)
        self.assertEqual(exported["inventory"], expected)
        for actual, original in zip(exported["controls"][0]["requests"], self.serial["requests"]):
            self.assertEqual(actual["response"]["token_ids"], original["response"]["token_ids"])
            self.assertNotIn("runtime_summary", actual["response"])
        pin = generation_corpus.ApprovedCorpusPin("AVX512", artifacts.digest(exported))
        with self.assertRaisesRegex(ValueError, "approval schema"):
            generation_corpus.ApprovedGenerationCorpus.admit(exported, pin, self.inventory, sources)
        output = self.root / "existing.json"
        output.write_text("preserve")
        with patch.object(control_export, "export_controls") as acquire:
            with self.assertRaises(FileExistsError):
                control_export.main(["--manifest", str(self.manifest), "--reports", str(self.reports[0]),
                                     "--cpu-isa", "AVX512", "--output", str(output)])
            acquire.assert_not_called()
        self.assertEqual(output.read_text(), "preserve")

    def test_partial_aggregate_keeps_completed_cells_but_not_unfinished_responses(self):
        report = json.loads(self.reports[1].read_text())
        report.update(complete=False, passed=False, selected=2)
        artifacts.write_json(self.reports[1], report)
        self.assertTrue(acquisition.audit(self.manifest, self.reports)["acquisition_complete"])
        observation = self.root / "mtp/row/generation/observations.json"
        changed = json.loads(observation.read_text())
        changed["complete"] = False
        artifacts.write_json(observation, changed)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            acquisition.audit(self.manifest, self.reports)

    def test_later_failure_is_not_hidden_by_an_earlier_pass_and_retry_can_resolve_it(self):
        failed = json.loads(self.reports[1].read_text())
        failed.update(complete=False, passed=False)
        failed["cells"][0]["return_code"] = 1
        failure_path = self.root / "failure.json"
        artifacts.write_json(failure_path, failed)
        result = acquisition.audit(self.manifest, [*self.reports, failure_path])
        self.assertFalse(result["acquisition_complete"])
        self.assertEqual(result["counts"]["mtp"]["passed"], 0)
        self.assertEqual(result["unresolved_failed_cases"], ["Suite.mtp"])
        retried = acquisition.audit(self.manifest, [self.reports[0], failure_path, self.reports[1]])
        self.assertTrue(retried["acquisition_complete"])
        self.assertEqual(retried["unresolved_failed_cases"], [])

    def test_reports_and_successful_cells_cannot_be_duplicated(self):
        with self.assertRaisesRegex(ValueError, "distinct explicit"):
            acquisition.audit(self.manifest, [*self.reports, self.reports[1]])
        duplicate = self.root / "mtp/duplicate.json"
        artifacts.write_json(duplicate, json.loads(self.reports[1].read_text()))
        with self.assertRaisesRegex(ValueError, "duplicate passing"):
            acquisition.audit(self.manifest, [*self.reports, duplicate])

    def test_acquisition_preserves_old_source_provenance_without_claiming_a_current_build_proof(self):
        report = json.loads(self.reports[0].read_text())
        report["source_revision"] = "original-control-source"
        artifacts.write_json(self.reports[0], report)
        result = acquisition.audit(self.manifest, self.reports)
        self.assertTrue(result["acquisition_complete"])
        self.assertTrue(result["mixed_source_revisions"])
        self.assertEqual(result["source_revisions"], ["original-control-source", "revision"])
        self.assertFalse(result["certification_eligible"])

    def test_full_inventory_and_current_configuration_cannot_be_replaced_by_a_name(self):
        changed = copy.deepcopy(self.inventory)
        changed["scope"] = "e2e"
        artifacts.write_json(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "all-cell manifest"):
            acquisition.audit(self.manifest, self.reports)
        artifacts.write_json(self.manifest, self.inventory)
        report = json.loads(self.reports[1].read_text())
        report["cells"][0]["configuration"]["runtime"]["server_args"].append("changed-policy")
        artifacts.write_json(self.reports[1], report)
        with self.assertRaisesRegex(ValueError, "canonical configuration"):
            acquisition.audit(self.manifest, self.reports)

    def test_token_drift_and_missing_serial_control_do_not_count_as_passes(self):
        with self.assertRaisesRegex(ValueError, "green canonical serial control"):
            acquisition.audit(self.manifest, self.reports[1:])
        observation = self.root / "mtp/row/generation/observations.json"
        changed = json.loads(observation.read_text())
        # Change both repeats to retain self-repeatability while violating the
        # independent serial answer at the final token of the required horizon.
        for row in changed["requests"][:2]:
            row["response"]["token_ids"]["completion"][-1] += 1
        artifacts.write_json(observation, changed)
        with self.assertRaisesRegex(ValueError, "original serial control"):
            acquisition.audit(self.manifest, self.reports)

    def test_report_flags_do_not_replace_prerequisites_or_original_artifact_ownership(self):
        original = json.loads(self.reports[1].read_text())
        for mutation in ({"preflight_return_code": 1}, {"preflight_tests": []},
                         {"preflight_tests": ["V2_Unit_A", "V2_Unit_A"]}, {"preflight_test_count": 0},
                         {"source_revision": ""}, {"selected": 2}, {"certification_eligible": True}):
            artifacts.write_json(self.reports[1], original | mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                acquisition.audit(self.manifest, self.reports)
        escaped = copy.deepcopy(original)
        escaped["cells"][0]["artifacts"] = str(self.root / "serial/row")
        artifacts.write_json(self.reports[1], escaped)
        with self.assertRaisesRegex(ValueError, "escape their original"):
            acquisition.audit(self.manifest, self.reports)


class ApprovedGenerationCorpusTests(unittest.TestCase):
    """A reviewed token file cannot choose or weaken its consuming inventory."""

    def setUp(self):
        """Keep compact synthetic streams, without model files or runtime proof."""
        helper = GenerationHTTPTests()
        records = [helper.record(), helper.record(True)]
        files = [records[0]["model"], "/models/second-shard.gguf"]
        self.inventory = {"schema": 1, "scope": "all", "source_revision": "candidate",
            "cells": [{"case": "Suite." + record["id"], "campaign": "campaign", "backends": "CPU",
                       "model_files": list(files), "configuration": record} for record in records]}
        self.sources = {path: [1, index + 2, 4096 + index, 100, 200] for index, path in enumerate(files)}
        requests = []
        for request in generation_http.generation_profile(records[0])["requests"]:
            response = helper.response("unused", request["body"], 1)
            # Expected tokens are not yesterday's prefix, movement or MTP
            # receipts. Every real candidate still needs fresh runtime proof.
            response.pop("runtime_summary")
            requests.append({"id": request["id"], "response": response})
        self.document = {"schema": 1, "kind": "approved_generation_controls", "cpu_isa": "AVX512",
            "inventory": generation_corpus.portable_inventory(self.inventory, self.sources),
            "provenance": {kind: [artifacts.digest({"synthetic_proof": kind})]
                           for kind in ("numerical", "serial", "mtp")},
            "controls": [{"id": "serial", "requests": requests}]}
        self.pin = generation_corpus.ApprovedCorpusPin("AVX512", artifacts.digest(self.document))

    def admit(self, document=None, *, repin=False, inventory=None, sources=None):
        """Only tests repin malformed fixtures to reach each structural guard."""
        document = self.document if document is None else document
        pin = (generation_corpus.ApprovedCorpusPin("AVX512", artifacts.digest(document))
               if repin else self.pin)
        return generation_corpus.ApprovedGenerationCorpus.admit(
            document, pin, self.inventory if inventory is None else inventory,
            self.sources if sources is None else sources)

    def record(self, name="mtp", *, inventory=None):
        """Select the exact fixture record, without reconstructing its policy."""
        rows = (self.inventory if inventory is None else inventory)["cells"]
        return next(row["configuration"] for row in rows if row["configuration"]["id"] == name)

    def test_serial_and_mtp_share_immutable_expected_streams_without_runtime_receipts(self):
        corpus = self.admit()
        serial = corpus.expected(self.record("serial"))
        self.assertEqual(corpus.expected(self.record()), serial)
        self.assertEqual(sum(len(trace.completion) for trace in serial.values()), 1536)
        self.document["controls"][0]["requests"][0]["response"]["token_ids"]["completion"][383] = 999999
        serial.clear()
        self.assertEqual(len(corpus.expected(self.record("serial"))), 4)
        self.assertEqual(corpus.expected(self.record("serial"))["seed"].completion[383], 383)
        with self.assertRaisesRegex(ValueError, "outside the approved"):
            corpus.expected({**self.record(), "id": "unseen"})

    def test_expected_document_roundtrip_and_live_mtp_prefix_evidence(self):
        """Golden tokens never substitute for a candidate's actual runtime proof."""
        helper = GenerationHTTPTests()
        record = self.record()
        corpus = self.admit()
        expected = corpus.expectations_document(record)
        self.assertEqual(generation_http.admit_expected_tokens(record, expected), corpus.expected(record))
        helper.mtp_policy = generation_http.MTPPolicy.DEPTH2
        with tempfile.TemporaryDirectory() as temp, patch.object(generation_http, "post_completion", side_effect=helper.response):
            result = generation_http.run_probes(record, "http://unused", Path(temp) / "probe", expected_tokens=expected)
            self.assertTrue(result["serial_comparison_passed"])
        for mutation in (
            lambda doc: doc["requests"].pop(),
            lambda doc: doc["configuration"]["runtime"]["generation"]["requests"][0]["body"].update(seed=99),
            lambda doc: doc.update(serial_control_id="different"),
            lambda doc: doc["requests"][0]["completion"].pop(),
        ):
            changed = copy.deepcopy(expected)
            mutation(changed)
            with self.assertRaises(ValueError):
                generation_http.admit_expected_tokens(record, changed)
        # The final token is just as binding as the first; checking a matching
        # prefix or concatenating short outputs must not satisfy the horizon.
        changed = copy.deepcopy(expected)
        for request in changed["requests"]:
            request["completion"][383] = 999999
        with tempfile.TemporaryDirectory() as temp, patch.object(generation_http, "post_completion", side_effect=helper.response):
            with self.assertRaises(ValueError):
                generation_http.run_probes(record, "http://unused", Path(temp) / "probe", expected_tokens=changed)



    def test_same_named_cell_cannot_change_its_admitted_configuration(self):
        corpus = self.admit()
        mutations = (lambda x: x.update(model="/models/foreign.gguf"),
                     lambda x: x["runtime"]["server_args"].append("changed-topology"),
                     lambda x: x["runtime"]["generation"]["requests"][0]["body"].update(seed=42),
                     lambda x: x["runtime"]["generation"].update(mtp_policy="off"),
                     lambda x: x["runtime"].update(movement_evidence="different-policy"))
        for mutate in mutations:
            record = copy.deepcopy(self.record())
            mutate(record)
            with self.assertRaisesRegex(ValueError, "admitted configuration"):
                corpus.expected(record)
        # A bare ID cannot authenticate the contract, nor can changing the
        # caller's original inventory mutate the corpus's admission snapshot.
        with self.assertRaises(TypeError):
            corpus.expected("mtp")
        self.record()["runtime"]["server_args"].append("changed-after-admission")
        with self.assertRaisesRegex(ValueError, "admitted configuration"):
            corpus.expected(self.record())

    def test_changed_payload_or_untyped_pin_fails_before_inventory_admission(self):
        changed = copy.deepcopy(self.document)
        changed["controls"][0]["requests"][0]["response"]["token_ids"]["completion"][383] += 1
        with patch.object(generation_corpus, "portable_inventory") as inventory:
            with self.assertRaisesRegex(ValueError, "reviewed metadata pin"):
                self.admit(changed)
            inventory.assert_not_called()
        with self.assertRaises(TypeError):
            generation_corpus.ApprovedGenerationCorpus.admit(self.document, {}, self.inventory, self.sources)
        for value in ("", "abc", "A" * 64, None):
            with self.subTest(pin=value), self.assertRaises(ValueError):
                generation_corpus.ApprovedCorpusPin("AVX512", value)

    def test_both_isas_have_explicit_independent_approval(self):
        for isa in ("AVX2", "AVX512"):
            document = {**self.document, "cpu_isa": isa}
            pin = generation_corpus.ApprovedCorpusPin(isa, artifacts.digest(document))
            corpus = generation_corpus.ApprovedGenerationCorpus.admit(document, pin, self.inventory, self.sources)
            self.assertEqual(corpus.pin.cpu_isa, isa)
            wrong = generation_corpus.ApprovedCorpusPin("AVX512" if isa == "AVX2" else "AVX2", pin.document_digest)
            with self.assertRaisesRegex(ValueError, "CPU ISA"):
                generation_corpus.ApprovedGenerationCorpus.admit(document, wrong, self.inventory, self.sources)

    def test_mount_and_source_revision_changes_do_not_change_regression_identity(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["source_revision"] = "new-source-improvement"
        sources = {}
        for row in inventory["cells"]:
            row["configuration"]["model"] = row["configuration"]["model"].replace("/models/", "/container/models/")
            row["model_files"] = [path.replace("/models/", "/container/models/") for path in row["model_files"]]
        for path, stat in self.sources.items():
            sources[path.replace("/models/", "/container/models/")] = [9, stat[1] + 50, stat[2], 900, 901]
        corpus = self.admit(inventory=inventory, sources=sources)
        self.assertEqual(corpus.inventory_digest, self.admit().inventory_digest)
        self.assertEqual(corpus.expected(self.record(inventory=inventory)), self.admit().expected(self.record()))
        with self.assertRaisesRegex(ValueError, "admitted configuration"):
            corpus.expected(self.record())
        self.assertEqual(self.inventory["cells"][0]["configuration"]["model"], "/models/model.gguf")

    def test_same_model_filename_with_changed_shard_size_is_rejected(self):
        for path in self.sources:
            sources = copy.deepcopy(self.sources)
            sources[path][2] += 1
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "model/configuration inventory"):
                self.admit(sources=sources)

    def test_container_mount_adapter_preserves_reviewed_tokens_and_all_shard_identity(self):
        """Join real mapping and corpus admission without trusting a mount alias."""
        container = copy.deepcopy(self.inventory)
        for row in container["cells"]:
            row["configuration"]["model"] = "/src/models/" + Path(row["configuration"]["model"]).name
            row["model_files"] = ["/src/models/" + Path(path).name for path in row["model_files"]]
        mounted = pipeline.remap_manifest(container, Path("/runtime/model pool"))
        sources = {"/runtime/model pool/" + Path(path).name: [9, stat[1] + 50, stat[2], 900, 901]
                   for path, stat in self.sources.items()}
        corpus = self.admit(inventory=mounted, sources=sources)
        self.assertEqual(corpus.expected(self.record(inventory=mounted)),
                         self.admit().expected(self.record()))
        with self.assertRaisesRegex(ValueError, "admitted configuration"):
            corpus.expected(self.record(inventory=container))
        # Translation cannot legitimize a substituted secondary shard merely
        # because its filename is the same inside the new mount.
        sources["/runtime/model pool/second-shard.gguf"][2] += 1
        with self.assertRaisesRegex(ValueError, "model/configuration inventory"):
            self.admit(inventory=mounted, sources=sources)

    def test_missing_extra_invalid_or_ambiguous_shard_pins_fail(self):
        for mutate in (lambda x: x.pop("/models/second-shard.gguf"),
                       lambda x: x.update({"/extra.gguf": [1, 2, 3, 4, 5]}),
                       lambda x: x["/models/model.gguf"].__setitem__(2, 0),
                       lambda x: x["/models/model.gguf"].__setitem__(1, True)):
            sources = copy.deepcopy(self.sources)
            mutate(sources)
            with self.assertRaises(ValueError):
                self.admit(sources=sources)
        inventory = copy.deepcopy(self.inventory)
        for row in inventory["cells"]:
            row["model_files"][1] = "/other/model.gguf"
        sources = {"/models/model.gguf": self.sources["/models/model.gguf"],
                   "/other/model.gguf": self.sources["/models/second-shard.gguf"]}
        with self.assertRaisesRegex(ValueError, "filenames are ambiguous"):
            self.admit(inventory=inventory, sources=sources)

    def test_full_inventory_and_exact_runtime_policy_are_required(self):
        mutations = (lambda x: x.update(scope="e2e"),
                     lambda x: x["cells"].pop(),
                     lambda x: x["cells"][0]["configuration"].pop("e2e"),
                     lambda x: x["cells"].append(copy.deepcopy(x["cells"][0])),
                     lambda x: x["cells"][1]["configuration"]["runtime"]["server_args"].append("different-topology"),
                     lambda x: x["cells"][0]["configuration"]["runtime"]["generation"]["requests"][0]["body"].update(seed=42),
                     lambda x: x["cells"][1]["configuration"]["runtime"]["generation"].update(serial_control_id="absent"))
        for mutate in mutations:
            inventory = copy.deepcopy(self.inventory)
            mutate(inventory)
            with self.assertRaises(ValueError):
                self.admit(inventory=inventory)

    def test_serial_control_requires_the_same_secondary_shards(self):
        """Matching primary model/workload cannot hide a different split model."""
        inventory = copy.deepcopy(self.inventory)
        inventory["cells"][1]["model_files"][1] = "/models/replaced-second.gguf"
        sources = {**self.sources, "/models/replaced-second.gguf": [1, 99, 4097, 100, 200]}
        with self.assertRaisesRegex(ValueError, "exact canonical serial control"):
            generation_corpus.portable_inventory(inventory, sources)

    def test_only_exact_serial_controls_can_supply_answers(self):
        for mutate in (lambda x: x["controls"].clear(),
                       lambda x: x["controls"].append(copy.deepcopy(x["controls"][0])),
                       lambda x: x["controls"][0].update(id="mtp"),
                       lambda x: x["controls"][0].update(id="unknown")):
            document = copy.deepcopy(self.document)
            mutate(document)
            with self.assertRaisesRegex(ValueError, "exactly the canonical serial controls"):
                self.admit(document, repin=True)

    def test_provenance_and_approval_kind_cannot_be_omitted(self):
        for mutate in (lambda x: x.update(kind="unapproved_observations"),
                       lambda x: x.update(schema=True),
                       lambda x: x.pop("provenance"),
                       lambda x: x["provenance"].pop("numerical"),
                       lambda x: x["provenance"].update(mtp=[]),
                       lambda x: x["provenance"].update(serial=["not-a-proof"]),
                       lambda x: x["provenance"].update(serial=x["provenance"]["serial"] * 2)):
            document = copy.deepcopy(self.document)
            mutate(document)
            with self.assertRaises(ValueError):
                self.admit(document, repin=True)

    def test_requests_must_be_complete_ordered_and_byte_repeatable(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.reverse(),
                       lambda rows: rows[1].update(id=rows[0]["id"]),
                       lambda rows: rows[1]["response"]["token_ids"]["completion"].__setitem__(383, 9876),
                       lambda rows: rows[2]["response"]["token_ids"]["prompt"].__setitem__(0, 9876)):
            document = copy.deepcopy(self.document)
            mutate(document["controls"][0]["requests"])
            with self.assertRaises(ValueError):
                self.admit(document, repin=True)

    def test_a_review_pin_cannot_waive_the_continuous_token_horizon(self):
        document = copy.deepcopy(self.document)
        response = document["controls"][0]["requests"][0]["response"]
        response["token_ids"]["completion"] = response["token_ids"]["completion"][:13]
        response["usage"].update(completion_tokens=13, total_tokens=response["usage"]["prompt_tokens"] + 13)
        response["choices"][0]["finish_reason"] = "stop"
        with self.assertRaisesRegex(ValueError, "insufficient continuous"):
            self.admit(document, repin=True)

    def test_loading_requires_existing_materialized_bytes_and_never_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.json"
            with self.assertRaises(FileNotFoundError):
                generation_corpus.ApprovedGenerationCorpus.load(path, self.pin, self.inventory, self.sources)
            artifacts.write_json(path, self.document)
            before = path.read_bytes()
            loaded = generation_corpus.ApprovedGenerationCorpus.load(path, self.pin, self.inventory, self.sources)
            self.assertEqual(loaded.expected(self.record()), loaded.expected(self.record("serial")))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            # An unmaterialized LFS pointer is not an empty or downloadable
            # corpus. The caller must explicitly provision the selected data.
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 123\n")
            with self.assertRaises(json.JSONDecodeError):
                generation_corpus.ApprovedGenerationCorpus.load(path, self.pin, self.inventory, self.sources)

    def approval_fixture(self, root):
        """Install synthetic review metadata separately from candidate results."""
        source, data = root / "source", root / "mounted-corpora"
        catalog = {"schema": 1, "corpora": {"AVX512": {
            "path": "generation/reviewed/avx512.json", "document_digest": self.pin.document_digest}}}
        artifacts.write_json(source / generation_corpus.APPROVAL_CATALOG, catalog)
        artifacts.write_json(data / catalog["corpora"]["AVX512"]["path"], self.document)
        return source, data, catalog

    def test_source_catalog_selects_each_isa_without_fetching_the_other_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            source, data, catalog = self.approval_fixture(Path(folder))
            avx2 = {**self.document, "cpu_isa": "AVX2"}
            catalog["corpora"]["AVX2"] = {"path": "generation/reviewed/avx2.json",
                                           "document_digest": artifacts.digest(avx2)}
            artifacts.write_json(source / generation_corpus.APPROVAL_CATALOG, catalog)
            before = {path: path.read_bytes() for path in Path(folder).rglob("*") if path.is_file()}
            reader = generation_corpus.ApprovedGenerationCorpus.load_reviewed
            result = reader(source, data, "AVX512", self.inventory, self.sources)
            self.assertEqual(result.expected(self.record()), self.admit().expected(self.record()))
            with self.assertRaises(FileNotFoundError):
                reader(source, data, "AVX2", self.inventory, self.sources)
            self.assertEqual(before, {path: path.read_bytes() for path in Path(folder).rglob("*") if path.is_file()})
            artifacts.write_json(data / catalog["corpora"]["AVX2"]["path"], avx2)
            self.assertEqual(reader(source, data, "AVX2", self.inventory, self.sources).pin.cpu_isa, "AVX2")

    def test_candidate_payload_cannot_choose_its_own_review_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            source, data, catalog = self.approval_fixture(Path(folder))
            candidate = copy.deepcopy(self.document)
            candidate["controls"][0]["requests"][0]["response"]["token_ids"]["completion"][383] += 1
            artifacts.write_json(data / catalog["corpora"]["AVX512"]["path"], candidate)
            # Even a valid-looking catalog next to candidate data is not the
            # source-owned approval catalog consulted by the consumer.
            catalog["corpora"]["AVX512"]["document_digest"] = artifacts.digest(candidate)
            artifacts.write_json(data / generation_corpus.APPROVAL_CATALOG, catalog)
            with self.assertRaisesRegex(ValueError, "reviewed metadata pin"):
                generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                    source, data, "AVX512", self.inventory, self.sources)

    def test_missing_or_unapproved_source_catalog_never_selects_another_isa(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                    root, root, "AVX512", self.inventory, self.sources)
            source, data, _ = self.approval_fixture(root)
            with self.assertRaisesRegex(ValueError, "no reviewed generation corpus"):
                generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                    source, data, "AVX2", self.inventory, self.sources)
            with self.assertRaisesRegex(ValueError, "shipping CPU ISA"):
                generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                    source, data, "NATIVE", self.inventory, self.sources)

    def test_invalid_catalog_metadata_and_path_spellings_fail_before_payload_read(self):
        with tempfile.TemporaryDirectory() as folder:
            source, data, catalog = self.approval_fixture(Path(folder))
            malformed = [[], {**catalog, "schema": True}, {**catalog, "unreviewed": True},
                         {"schema": 1, "corpora": {"NATIVE": {}}}]
            for path in ("", ".", "../candidate.json", "a/../../candidate.json", "/tmp/candidate.json",
                         "./relative.json", "a//relative.json", "a\\relative.json"):
                changed = copy.deepcopy(catalog)
                changed["corpora"]["AVX512"]["path"] = path
                malformed.append(changed)
            for field, value in (("document_digest", "not-a-pin"), ("candidate_pin", "0" * 64)):
                changed = copy.deepcopy(catalog)
                changed["corpora"]["AVX512"][field] = value
                malformed.append(changed)
            with patch.object(generation_corpus.ApprovedGenerationCorpus, "load") as load:
                for document in malformed:
                    with self.subTest(document=document):
                        artifacts.write_json(source / generation_corpus.APPROVAL_CATALOG, document)
                        with self.assertRaises(ValueError):
                            generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                                source, data, "AVX512", self.inventory, self.sources)
                load.assert_not_called()

    def test_source_and_payload_symlinks_cannot_escape_their_admitted_roots(self):
        for escape_catalog in (False, True):
            with self.subTest(catalog=escape_catalog), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, data, catalog = self.approval_fixture(root)
                path = (source / generation_corpus.APPROVAL_CATALOG if escape_catalog
                        else data / catalog["corpora"]["AVX512"]["path"])
                outside = root / "candidate.json"
                path.rename(outside)
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    generation_corpus.ApprovedGenerationCorpus.load_reviewed(
                        source, data, "AVX512", self.inventory, self.sources)


class RoutineGenerationPolicyTests(unittest.TestCase):
    """Default scheduling and source-bound evidence cannot silently lose cells."""

    def test_default_selects_off_and_dynamic_not_fixed_depths(self):
        helper = GenerationHTTPTests()
        campaign = parity.CampaignCell("campaign", parity.CampaignGroup("CUDA", "ALL"))
        records = [helper.record()]
        for policy in ("depth_1", "depth_2", "depth_3", "depth_15", "dynamic"):
            record = helper.record(True)
            record["id"] = policy
            record["runtime"]["generation"]["mtp_policy"] = policy
            records.append(record)
        inventory = [(campaign, "Suite." + record["id"], record) for record in records]
        args = argparse.Namespace(mode=generation_runner.RunMode.REGRESSION, backend=".*", campaign=".*", cell=".*")
        with patch.object(generation_runner, "discover", return_value=inventory):
            self.assertEqual([cell.configuration["id"] for cell in generation_runner.select_cells(args)],
                             ["serial", "dynamic"])
            self.assertEqual(len(generation_runner.select_cells(args, generation_corpus.InventoryScope.ALL)), 6)

    def test_mathematical_matrix_is_explicit_and_receipts_cannot_change_policy(self):
        self.assertNotIn(pipeline.Phase.PARITY, pipeline.pipeline_phases())
        self.assertIn(pipeline.Phase.PARITY, pipeline.pipeline_phases(True))
        names = [phase.value for phase in pipeline.pipeline_phases(True)]
        pipeline.validate_phase_prefix(dict.fromkeys(names), True)
        with self.assertRaises(ValueError):
            pipeline.validate_phase_prefix(dict.fromkeys(names))
        with patch.object(pipeline, "source_identity") as source:
            with self.assertRaises(SystemExit):
                pipeline.main(["--output", "/unused", "--through", "parity"])
            source.assert_not_called()

    def test_complete_report_requires_reviewed_answers_exact_cells_and_shared_gate(self):
        good = generation_report()
        generation_runner.validate_regression_report(good, full_manifest(), prerequisite_report(),
                                                     "runtime-id", "AVX512", "c" * 64)
        for mutate in (
            lambda r: r.update(mode="collect-controls"), lambda r: r.update(mode="compare-controls"),
            lambda r: r.update(corpus_digest="new-self-approved-answers"),
            lambda r: r.update(cpu_isa="AVX2"), lambda r: r.update(image="another-image"),
            lambda r: r.update(certification_eligible=False), lambda r: r.update(passed=False),
            lambda r: r.update(prerequisite_report_digest="stale"), lambda r: r.update(cells=[]),
            lambda r: r["cells"].__setitem__(1, copy.deepcopy(r["cells"][0])),
            lambda r: r["cells"][0]["configuration"]["runtime"]["generation"]["requests"][0]["body"].update(seed=99),
        ):
            report = copy.deepcopy(good)
            mutate(report)
            with self.assertRaises(ValueError):
                generation_runner.validate_regression_report(report, full_manifest(), prerequisite_report(),
                                                             "runtime-id", "AVX512", "c" * 64)


class PrerequisiteEntrypointTests(unittest.TestCase):
    """The model-free command delegates to one complete existing authority."""

    def test_local_and_installed_paths_execute_both_complete_namespaces(self):
        """Mock external operations, not the canonical gate's phase transitions."""
        for installed in (False, True):
            with self.subTest(installed=installed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                build = root / "build"
                build.mkdir()
                output = root / "results"
                receipt = root / "installed.json"
                artifacts.write_json(receipt, {"fixture": True})
                argv = ["--build-dir", str(build), "--output", str(output)]
                if installed:
                    argv += ["--installed-build-receipt", str(receipt)]
                with patch.object(parity, "discover_production_parity_unit_tests", return_value=("V2_Unit_A",)), \
                     patch.object(parity, "discover_production_parity_preflight_tests", return_value=("V2_Integration_B",)), \
                     patch.object(prebuilt_test_image, "validate") as validate, \
                     patch.object(parity, "_run_process", return_value=0) as execute:
                    self.assertEqual(prerequisites_runner.main(argv), 0)
                commands = [call.args[0] for call in execute.call_args_list]
                self.assertEqual(len(commands), 2 if installed else 3)
                if installed:
                    validate.assert_called_once_with(receipt, build)
                else:
                    validate.assert_not_called()
                    self.assertEqual(commands[0], ["cmake", "--build", str(build), "--parallel", "--target",
                                                  parity.PRODUCTION_PARITY_UNIT_BUILD_TARGET,
                                                  parity.PRODUCTION_PARITY_PREFLIGHT_BUILD_TARGET])
                unit, preflight = commands[-2:]
                self.assertEqual(unit[unit.index("-R") + 1], "^V2_Unit_")
                self.assertEqual(preflight[preflight.index("-L") + 1], "^ProductionParityPreflight$")
                for command in (unit, preflight):
                    self.assertIn("--output-log", command)
                    self.assertIn("--output-junit", command)
                    self.assertIn("--no-tests=error", command)
                result = json.loads((output / "prerequisites.json").read_text())
                self.assertEqual(result["preflight_tests"], ["V2_Unit_A", "V2_Integration_B"])
                self.assertEqual(result["preflight_return_code"], 0)
                self.assertFalse(result["certification_eligible"])

    def test_existing_output_is_never_replaced_or_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "results"
            output.mkdir()
            artifacts.write_json(output / "prerequisites.json", {"prior": "evidence"})
            with patch.object(prerequisites_runner, "run_production_parity_preflight") as execute:
                with self.assertRaises(FileExistsError):
                    prerequisites_runner.main(["--build-dir", str(root), "--output", str(output)])
                execute.assert_not_called()
            self.assertEqual(json.loads((output / "prerequisites.json").read_text()), {"prior": "evidence"})

    def test_failed_gate_exit_code_is_preserved_without_synthesized_success(self):
        for code in (1, 2, 124):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                output = root / "results"
                with patch.object(prerequisites_runner, "run_production_parity_preflight",
                                  return_value=(code, 1.0, ())) as execute:
                    self.assertEqual(prerequisites_runner.main([
                        "--build-dir", str(root), "--output", str(output)]), code)
                    execute.assert_called_once_with(root, None, installed_build_receipt=None,
                                                    artifact_directory=output)
                self.assertEqual(list(output.iterdir()), [])

    def test_missing_or_invalid_installed_receipt_cannot_rebuild_around_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = root / "installed.json"
            with patch.object(prerequisites_runner, "run_production_parity_preflight") as authority:
                with self.assertRaises(FileNotFoundError):
                    prerequisites_runner.main(["--build-dir", str(root), "--output", str(root / "missing"),
                                               "--installed-build-receipt", str(receipt)])
                authority.assert_not_called()
            artifacts.write_json(receipt, {"invalid": True})
            with patch.object(parity, "discover_production_parity_unit_tests", return_value=("V2_Unit_A",)), \
                 patch.object(parity, "discover_production_parity_preflight_tests", return_value=("V2_Integration_B",)), \
                 patch.object(prebuilt_test_image, "validate", side_effect=ValueError("stale installation")), \
                 patch.object(parity, "_run_process") as execute:
                with self.assertRaisesRegex(ValueError, "stale installation"):
                    prerequisites_runner.main(["--build-dir", str(root), "--output", str(root / "invalid"),
                                               "--installed-build-receipt", str(receipt)])
                execute.assert_not_called()

    def test_cli_has_no_inventory_selector_or_skip_switch(self):
        for flag in ("--cell", "--skip-preflight", "--backend", "--collect-controls"):
            with self.subTest(flag=flag), patch.object(prerequisites_runner, "run_production_parity_preflight") as execute:
                with self.assertRaises(SystemExit) as error:
                    prerequisites_runner.main(["--build-dir", "unused", "--output", "unused", flag])
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()


class BenchmarkPolicyTests(unittest.TestCase):
    def test_e2e_admission_requires_full_same_image_evidence(self):
        inventory = manifest()
        inventory["cells"].append(cell("second"))
        evidence = e2e_report(inventory)
        artifacts.validate_image_e2e(evidence, inventory, "runtime-id")
        for mutate in (lambda r: r.update(correctness_passed=False),
                       lambda r: r.update(selected=1), lambda r: r["cells"].pop(),
                       lambda r: r.update(image="other-build-of-same-revision"),
                       lambda r: r.update(source_revision="stale"),
                       lambda r: r.update(manifest_digest="stale"),
                       lambda r: r["cells"][0].update(return_code=1),
                       lambda r: r["cells"][0].update(outcome="skipped")):
            invalid = copy.deepcopy(evidence)
            mutate(invalid)
            with self.subTest(report=invalid), self.assertRaises(ValueError):
                artifacts.validate_image_e2e(invalid, inventory, "runtime-id")

    def test_cli_blocks_missing_e2e_before_touching_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifacts.write_json(path / "manifest.json", manifest())
            argv = ["--manifest", str(path / "manifest.json"), "--source-revision", "revision",
                    "--image", "tag", "--report", str(path / "out.json")]
            with patch.object(benchmark, "image_identity") as inspect:
                with self.assertRaises(SystemExit) as error:
                    benchmark.main(argv)
                self.assertEqual(error.exception.code, 2)
                inspect.assert_not_called()
                self.assertEqual(benchmark.main([*argv, "--list"]), 0)
                inspect.assert_not_called()

    def test_cli_diagnostic_is_explicit_and_bypasses_only_e2e_admission(self):
        # Stop at hardware discovery, before model staging/device execution.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifacts.write_json(path / "manifest.json", manifest())
            argv = ["--manifest", str(path / "manifest.json"), "--source-revision", "revision",
                    "--image", "tag", "--report", str(path / "out.json")]
            runtime = {"id": "runtime-id", "labels": {"org.opencontainers.image.revision": "revision"}}
            with patch.object(benchmark, "image_identity", return_value=runtime), \
                 patch.object(benchmark, "hardware_identity", side_effect=RuntimeError("admitted")) as hardware:
                with self.assertRaisesRegex(RuntimeError, "admitted"):
                    benchmark.main([*argv, "--diagnostic"])
                hardware.assert_called_once()
                hardware.reset_mock()
                artifacts.write_json(path / "e2e.json", e2e_report())
                with self.assertRaisesRegex(RuntimeError, "admitted"):
                    benchmark.main([*argv, "--e2e-report", str(path / "e2e.json")])
                hardware.assert_called_once()
                hardware.reset_mock()
                artifacts.write_json(path / "e2e.json", {**e2e_report(), "cells": []})
                with self.assertRaises(ValueError):
                    benchmark.main([*argv, "--e2e-report", str(path / "e2e.json")])
                hardware.assert_not_called()

    def test_timeout_retires_the_exact_container_and_never_inherits_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(benchmark.docker_paths, "device_args", return_value=[]), \
                 patch.object(benchmark.docker_paths, "mounts", return_value=[]), \
                 patch.object(benchmark.subprocess, "run", side_effect=[
                     subprocess.TimeoutExpired("docker", 600), None]) as execute:
                with self.assertRaises(subprocess.TimeoutExpired):
                    benchmark.run_cell("immutable-image", "CUDA", path, path / "model.gguf",
                                       cell()["configuration"], workload())
            launched = execute.call_args_list[0].args[0]
            retired = execute.call_args_list[1].args[0]
            self.assertEqual(retired[:3], ["docker", "rm", "-f"])
            self.assertEqual(retired[3], launched[launched.index("--name") + 1])
            self.assertEqual(execute.call_args_list[0].kwargs["timeout"], 900)
            environment = [launched[i + 1] for i, value in enumerate(launched) if value == "-e"]
            self.assertEqual(environment, ["LLAMINAR_BENCHMARK_ITERATIONS=3",
                                           "LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1"])

    def test_benchmarks_reuse_canonical_argv_verbatim(self):
        record = cell()["configuration"]
        args = benchmark.benchmark_arguments(record, "/tmp/model", "/tmp/out", workload())
        canonical = record["e2e"]["server_args"]
        self.assertEqual(args[1:1 + len(canonical)], canonical)
        self.assertNotIn("--perf-stats", args)
        self.assertNotIn("--deterministic", args)

    def test_untagged_empty_duplicate_and_stale_manifests_are_rejected(self):
        for mutate in (lambda m: m.update(cells=[]),
                       lambda m: m["cells"].append(cell()),
                       lambda m: m["cells"][0]["configuration"].update(e2e=None),
                       lambda m: m["cells"][0].update(model_files=[]),
                       lambda m: m.update(source_revision="stale")):
            doc = manifest()
            mutate(doc)
            with self.assertRaises(ValueError):
                artifacts.validate_manifest(doc, "revision")

    def test_median_not_best_retry_is_the_canonical_measurement(self):
        values = benchmark.validate_measurement(measurement(), workload())
        self.assertEqual(values["tokens_per_second"], {"prefill": 100, "decode": 10})
        self.assertEqual(values["prefill_tokens"], 512)

    def test_short_prompt_missing_iterations_eos_and_failed_runtime_are_red(self):
        for mutate in (lambda d: d.update(success=False),
                       lambda d: d.update(warmup_iterations=0),
                       lambda d: d["prompt"].update(bytes=1),
                       lambda d: d["prompt"].update(sha256="wrong"),
                       lambda d: d["iterations"].pop(),
                       lambda d: d["iterations"][0]["tokens"].update(decode=10),
                       lambda d: d["iterations"][0]["tokens"].update(prefill=511)):
            data = measurement()
            mutate(data)
            with self.assertRaises(ValueError):
                benchmark.validate_measurement(data, workload())

    def test_nonfinite_zero_and_boolean_throughput_cannot_pass(self):
        for invalid in (None, True, 0, -1, float("nan"), float("inf")):
            data = measurement()
            data["iterations"][0]["throughput_tokens_per_sec"]["prefill"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                benchmark.validate_measurement(data, workload())

    def test_ratchet_is_upward_only_and_reports_both_axes(self):
        empty = {"schema": 1, "regression_threshold_pct": 10, "entries": {}}
        row = {"case": "cell", "identity": {"cell": "cell", "hardware": "A"},
               "tokens_per_second": {"prefill": 100, "decode": 100}}
        first, comparison = artifacts.ratchet(empty, [row])
        self.assertTrue(all(item["status"] == "new_baseline" for item in comparison))
        next_row = {**row, "tokens_per_second": {"prefill": 105, "decode": 95}}
        second, comparison = artifacts.ratchet(first, [next_row])
        marks = second["entries"][artifacts.digest(row["identity"])]["tokens_per_second"]
        self.assertEqual(marks, {"prefill": 105, "decode": 100})
        self.assertTrue(all(item["passed"] for item in comparison))
        self.assertEqual(empty["entries"], {})
        _, red = artifacts.ratchet(second, [{**row, "tokens_per_second": {"prefill": 105, "decode": 89}}])
        self.assertFalse(red[1]["passed"])

    def test_ratchet_isolated_by_hardware_and_rejects_duplicate_or_empty_run(self):
        base = {"schema": 1, "regression_threshold_pct": 10, "entries": {}}
        row = {"case": "cell", "identity": {"hardware": "A"},
               "tokens_per_second": {"prefill": 100, "decode": 100}}
        base, _ = artifacts.ratchet(base, [row])
        other, evidence = artifacts.ratchet(base, [{**row, "identity": {"hardware": "B"}}])
        self.assertEqual(len(other["entries"]), 2)
        self.assertTrue(all(item["status"] == "new_baseline" for item in evidence))
        for rows in ([], [row, row]):
            with self.assertRaises(ValueError):
                artifacts.ratchet(base, rows)

    def test_manifest_translation_preserves_all_shards_and_execution_policy(self):
        before = manifest()
        after = pipeline.remap_manifest(before, Path("/actual models"))
        self.assertEqual(after["cells"][0]["model_files"],
                         ["/actual models/model-1.gguf", "/actual models/model-2.gguf"])
        self.assertEqual(after["cells"][0]["configuration"]["e2e"], before["cells"][0]["configuration"]["e2e"])


class PipelineInventoryTests(unittest.TestCase):
    def setUp(self):
        install_corpus_io_fixture(self)

    """One discovered full inventory owns model pins and its exact E2E subset."""

    def test_container_model_translation_rejects_escape_and_ambiguous_paths(self):
        """Reject malformed primary and secondary paths before model admission."""
        for filename in ("/src/models/../outside.gguf",
                         "/src/models/nested/../../outside.gguf",
                         "/src/models/./model.gguf", "/src/models//model.gguf",
                         "/src/models/model.gguf/", "/other/model.gguf",
                         "relative.gguf", "/src/models", "/src/models/",
                         "/src/models/bad\x00.gguf", "/src/models/bad\n.gguf"):
            for index in (0, 1):
                full = full_manifest()
                row = full["cells"][1]  # Untagged models need the same boundary.
                row["model_files"][index] = filename
                if index == 0:
                    row["configuration"]["model"] = filename
                before = copy.deepcopy(full)
                with self.subTest(filename=filename, shard=index), self.assertRaises(ValueError):
                    pipeline.remap_manifest(full, Path("/admitted/models"))
                self.assertEqual(full, before)

    def test_container_model_translation_requires_an_absolute_resolved_mount(self):
        """The caller supplies the admitted mount, never a working-directory alias."""
        for destination in (Path("relative"), Path("/admitted/../outside")):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                pipeline.remap_manifest(full_manifest(), destination)

    def test_container_model_translation_preserves_nested_inventory_and_policy(self):
        """Translation changes only model mount spelling, including untagged shards."""
        full = full_manifest()
        for row in full["cells"]:
            row["model_files"] = [path.replace("/src/models/", "/src/models/nested folder/")
                                  for path in row["model_files"]]
            row["configuration"]["model"] = row["model_files"][0]
        before = copy.deepcopy(full)
        translated = pipeline.remap_manifest(full, Path("/mounted models"))
        expected = copy.deepcopy(full)
        for row in expected["cells"]:
            row["model_files"] = [path.replace("/src/models/", "/mounted models/")
                                  for path in row["model_files"]]
            row["configuration"]["model"] = row["model_files"][0]
        self.assertEqual(translated, expected)
        self.assertEqual(full, before)
        self.assertEqual(len(pipeline.e2e_projection(translated, "revision")["cells"]), 1)

    def test_build_rejects_untagged_path_escape_before_model_stat_admission(self):
        """Exercise the real build transition, not merely the translation helper."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            full = full_manifest()
            full["cells"][1]["model_files"][1] = "/src/models/../unmounted.gguf"
            args = argparse.Namespace(models=models, model_ramdisk_root=root,
                reference_cache_root=root / "references", output=root / "results",
                cpu_isa="AVX512", image=None, resume=False, through="build")
            source = {"revision": "revision", "tree": "tree", "dirty": False}

            def export(images, command, args, output, log):
                """Supply the malformed installed inventory at its real boundary."""
                artifacts.write_json(output / "container-all-cells.json", full)

            with patch.object(pipeline, "source_identity", return_value=source), \
                 patch.object(pipeline, "build", return_value=image_pair(source)), \
                 patch.object(pipeline, "run_builder", side_effect=export), \
                 patch.object(pipeline, "model_identities") as pin:
                with self.assertRaisesRegex(ValueError, "noncanonical path"):
                    pipeline.run_variant(args, source)
                pin.assert_not_called()
            state = json.loads((args.output / "pipeline.json").read_text())
            self.assertEqual(state["phases"], {})
            self.assertFalse(state["certified"])

    def test_tagged_projection_preserves_records_without_mutating_full_inventory(self):
        """Eligibility selects whole rows; it cannot rewrite runtime policy."""
        full = full_manifest()
        before = copy.deepcopy(full)
        projected = pipeline.e2e_projection(full, "revision")
        self.assertEqual(projected, manifest())
        self.assertEqual(full, before)
        projected["cells"][0]["configuration"]["e2e"]["server_args"].append("changed")
        self.assertEqual(full, before)

    def test_remote_projection_keeps_canonical_policy_and_rejects_stale_or_duplicate_tags(self):
        """Remote discovery is an immutable projection, never another model table."""
        full = full_manifest()
        full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        original = copy.deepcopy(full)
        projected = pipeline.cross_host_e2e_projection(full, "revision")
        self.assertEqual(projected, {**full, "scope": "cross-host-e2e", "cells": full["cells"][:1]})
        projected["cells"][0]["configuration"]["cross_host_e2e"][0]["server_policy_args"].append("changed")
        self.assertEqual(full, original)
        for mutate in (
            lambda m: m["cells"][1]["configuration"].pop("cross_host_e2e"),
            lambda m: m["cells"][0]["configuration"]["cross_host_e2e"].pop(),
            lambda m: m["cells"][1]["configuration"].update(
                e2e={}, cross_host_e2e=remote_cases()),
        ):
            broken = copy.deepcopy(full)
            mutate(broken)
            with self.assertRaises(ValueError):
                pipeline.cross_host_e2e_projection(broken, "revision")

    def test_full_inventory_rejects_wrong_scope_and_malformed_untagged_members(self):
        """Validate untagged members too, before filtering would hide a defect."""
        for index, mutate in enumerate((
                lambda m: m.update(scope="e2e"), lambda m: m.pop("scope"),
                lambda m: m.update(source_revision="other"), lambda m: m.update(schema=True),
                lambda m: m.update(cells=[]), lambda m: m["cells"].append(copy.deepcopy(m["cells"][1])),
                lambda m: m["cells"][1]["configuration"].update(id="cell"),
                lambda m: m["cells"][1]["configuration"].pop("e2e"),
                lambda m: m["cells"][1]["configuration"].update(e2e=False),
                lambda m: m["cells"][1].update(model_files=[]),
                lambda m: m["cells"][1]["model_files"].append(m["cells"][1]["model_files"][0]),
                lambda m: m["cells"][1]["configuration"].update(model="/src/models/absent.gguf"))):
            full = full_manifest()
            mutate(full)
            with self.subTest(index=index), self.assertRaises(ValueError):
                pipeline.e2e_projection(full, "revision")

    def test_build_exports_full_inventory_and_pins_untagged_model_shards(self):
        """Exercise the real build transition with only external Docker mocked."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            full = full_manifest()
            full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
            for row in full["cells"]:
                for filename in row["model_files"]:
                    (models / Path(filename).name).write_text("model fixture")
            args = argparse.Namespace(models=models, model_ramdisk_root=root,
                reference_cache_root=root / "references", output=root / "results",
                cpu_isa="AVX512", image=None, resume=False, through="build")
            source = {"revision": "revision", "tree": "tree", "dirty": False}
            images = image_pair(source)

            def export(images, command, args, output, log):
                """Materialize the metadata-only container command's declared output."""
                self.assertEqual(command[:2], ["python3", "scripts/ci/model_parity_inventory.py"])
                self.assertEqual(command[command.index("--scope") + 1], "all")
                self.assertFalse({"--cell", "--campaign", "--backend"}.intersection(command))
                filename = Path(command[command.index("--export-manifest") + 1]).name
                artifacts.write_json(output / filename, full)

            with patch.object(pipeline, "source_identity", return_value=source), \
                 patch.object(pipeline, "build", return_value=images) as build, \
                 patch.object(pipeline, "run_builder", side_effect=export) as discover, \
                 patch.object(pipeline, "image_identity", side_effect=lambda identity: next(
                     item for item in images.values() if item["id"] == identity)):
                state = pipeline.run_variant(args, source)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(discover.call_count, 1)
                self.assertEqual(len(state["model_sources"]), 4)
                self.assertEqual(set(state["phases"]["build"]["files"]),
                                 {"container-all-cells.json", "all-cells.json", "manifest.json", "cross-host-manifest.json"})
                observed = json.loads((args.output / "all-cells.json").read_text())
                self.assertEqual(observed, pipeline.remap_manifest(full, models))
                projected = json.loads((args.output / "manifest.json").read_text())
                self.assertEqual(projected, pipeline.e2e_projection(observed, "revision"))
                remote = json.loads((args.output / "cross-host-manifest.json").read_text())
                self.assertEqual(remote, pipeline.cross_host_e2e_projection(observed, "revision"))
                self.assertEqual(len(remote["cells"][0]["configuration"]["cross_host_e2e"]), 4)
                self.assertFalse(state["certified"])
                args.resume = True
                self.assertEqual(pipeline.run_variant(args, source), state)
                # A receipt cannot invent metadata for an immutable Docker ID.
                # Re-inspection must compare labels/layers as well as the ID.
                for field, value in (("labels", {**images["builder"]["labels"],
                                                "org.llaminar.cpu_isa": "AVX2"}),
                                     ("layers", ["foreign-layer"])):
                    forged = copy.deepcopy(state)
                    forged["images"]["builder"][field] = value
                    artifacts.write_json(args.output / "pipeline.json", forged)
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity/labels/layers"):
                        pipeline.run_variant(args, source)
                artifacts.write_json(args.output / "pipeline.json", state)
                # A model used only by numerical/generation cells must invalidate
                # resume just as decisively as an E2E-tagged model does.
                (models / "other-2.gguf").write_text("changed untagged shard")
                with self.assertRaisesRegex(ValueError, "model files changed"):
                    pipeline.run_variant(args, source)
                self.assertEqual(build.call_count, 1)

    def test_host_phases_use_frozen_scripts_and_source_changes_stop_the_next_phase(self):
        """Exercise real phase admission; mock external execution, not policy.

        Builder tests are installed in their image. Host-driven HTTP and
        benchmarks must instead run the matching archived scripts, even if a
        developer edits the workspace while a long device phase is running.
        These synthetic reports prove scheduling only, never inference.
        """
        for isa in pipeline.SHIPPING_ISAS:
            for changed in (False, True):
                with self.subTest(isa=isa, changed=changed), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    models = root / "models"
                    models.mkdir()
                    full = full_manifest()
                    for filename in {name for row in full["cells"] for name in row["model_files"]}:
                        (models / Path(filename).name).write_text("model fixture")
                    args = argparse.Namespace(models=models, model_ramdisk_root=root,
                        reference_cache_root=root / "references", output=root / "results",
                        cpu_isa=isa, image=None, resume=False, through="benchmarks")
                    source = {"revision": "revision", "tree": "tree", "dirty": False}
                    current_source = copy.deepcopy(source)
                    images = image_pair(source, isa)

                    def builder(images, command, args, output, log):
                        """Stand in for only installed discovery and test execution."""
                        name = "container-all-cells.json" if log == "discovery.log" else "prerequisites/prerequisites.json"
                        artifacts.write_json(output / name, full if log == "discovery.log" else prerequisite_report())

                    def host(command, log, **kwargs):
                        """Assert launch provenance before producing a mock receipt."""
                        phase = log.stem
                        expected = args.output / "source/scripts/ci" / f"run_model_parity_{phase}.py"
                        self.assertEqual(command[:2], [sys.executable, str(expected)])
                        self.assertFalse({"--cell", "--campaign", "--backend", "--diagnostic"}.intersection(command))
                        flag = "--container-image" if phase in ("e2e", "generation") else "--image"
                        self.assertEqual(command[command.index(flag) + 1], images["runtime"]["id"])
                        manifest_path = Path(command[command.index("--manifest") + 1])
                        self.assertEqual(manifest_path, args.output / ("all-cells.json" if phase == "generation" else "manifest.json"))
                        projected = json.loads(manifest_path.read_text())
                        if phase == "generation":
                            self.assertEqual(projected, pipeline.remap_manifest(full, models))
                            self.assertEqual(command[command.index("--prerequisite-report") + 1],
                                             str(args.output / "prerequisites/prerequisites.json"))
                            artifacts.write_json(Path(command[command.index("--output") + 1]) / "report.json",
                                                 generation_report(projected, isa))
                            return
                        self.assertEqual(projected, pipeline.e2e_projection(pipeline.remap_manifest(full, models), "revision"))
                        if phase == "e2e":
                            result = e2e_report(projected)
                            if changed:
                                current_source["tree"] = "workspace-edited-during-e2e"
                        else:
                            self.assertEqual(phase, "benchmarks")
                            self.assertEqual(command[command.index("--e2e-report") + 1], str(args.output / "e2e.json"))
                            result = {"synthetic_benchmark_receipt": True}
                        artifacts.write_json(Path(command[command.index("--report") + 1]), result)

                    with patch.object(pipeline, "source_identity", side_effect=lambda: copy.deepcopy(current_source)), \
                         patch.object(pipeline, "build", return_value=images), \
                         patch.object(pipeline, "run_builder", side_effect=builder), \
                         patch.object(pipeline, "run", side_effect=host) as run:
                        if changed:
                            with self.assertRaisesRegex(ValueError, "source changed after admission"):
                                pipeline.run_variant(args, source)
                        else:
                            state = pipeline.run_variant(args, source)
                            self.assertFalse(state["certified"])
                        self.assertEqual([call.args[1].stem for call in run.call_args_list],
                                         ["generation", "e2e"] if changed else ["generation", "e2e", "benchmarks"])

    def test_cross_host_phase_follows_http_and_precedes_benchmarks(self):
        """A declared remote projection is a real ordered certification phase."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            full = full_manifest()
            full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
            for filename in {name for row in full["cells"] for name in row["model_files"]}:
                (models / Path(filename).name).write_text("model fixture")
            args = argparse.Namespace(models=models, model_ramdisk_root=root,
                reference_cache_root=root / "references", output=root / "results",
                cpu_isa="AVX512", image=None, resume=False, through="benchmarks",
                resolved_remote_cpu_image="remote-avx2-id")
            source = {"revision": "revision", "tree": "tree", "dirty": False}
            images = image_pair(source, "AVX512")

            def builder(images, command, args, output, log):
                artifacts.write_json(output / ("container-all-cells.json" if log == "discovery.log" else "prerequisites/prerequisites.json"),
                                     full if log == "discovery.log" else prerequisite_report())

            def host(command, log, **kwargs):
                phase = log.stem
                if phase == "generation":
                    artifacts.write_json(Path(command[command.index("--output") + 1]) / "report.json",
                                         generation_report(pipeline.remap_manifest(full, models)))
                elif phase == "e2e":
                    artifacts.write_json(Path(command[command.index("--report") + 1]),
                                         e2e_report(pipeline.e2e_projection(pipeline.remap_manifest(full, models), "revision")))
                elif phase == "cross-host-e2e":
                    remote = pipeline.cross_host_e2e_projection(pipeline.remap_manifest(full, models), "revision")
                    runtime = images["runtime"]["id"]
                    scenarios = [scenario for row in remote["cells"]
                                 for scenario in row["configuration"]["cross_host_e2e"]]
                    artifacts.write_json(Path(command[command.index("--report") + 1]), {
                        "schema": 1, "eligible": True, "complete": True,
                        "source_revision": "revision", "image": runtime,
                        "remote_cpu_image": "remote-avx2-id",
                        "manifest_digest": artifacts.digest(remote), "all_resources_retired": True,
                        "scenarios": [{"id": item["id"], "frontend": item["frontend"],
                                       "topology": item["topology"], "return_code": 0,
                                       "outcome": "passed", "resource_retired": True,
                                       "transport_proof": True, "http_proof": True}
                                      for item in scenarios]})
                else:
                    artifacts.write_json(Path(command[command.index("--report") + 1]), {"synthetic": True})

            with patch.object(pipeline, "source_identity", return_value=source), \
                 patch.object(pipeline, "build", return_value=images), \
                 patch.object(pipeline, "run_builder", side_effect=builder), \
                 patch.object(pipeline, "run", side_effect=host) as execute:
                state = pipeline.run_variant(args, source)
            self.assertEqual([call.args[1].stem for call in execute.call_args_list],
                             ["generation", "e2e", "cross-host-e2e", "benchmarks"])
            cross_host = execute.call_args_list[2].args[0]
            self.assertEqual(cross_host[cross_host.index("--remote-cpu-image") + 1], "remote-avx2-id")
            self.assertIn("cross-host-e2e", state["phases"])


class CrossHostRunnerTests(unittest.TestCase):
    """Device-free checks for remote launcher admission and artifact safety."""

    def test_remote_cpu_runtime_requires_matching_release_source_and_real_isa_support(self):
        """Reject a doomed AVX-512 peer before image/model transfer or MPI startup."""
        controller_identity = remote_runtime_identity(identifier="sha256:controller", isa="AVX512")
        avx2_identity = remote_runtime_identity(identifier="sha256:peer", isa="AVX2")
        controller = remote_e2e.RuntimeImage(controller_identity, remote_e2e.CPUISA.AVX512)
        peer = remote_e2e.RuntimeImage(avx2_identity, remote_e2e.CPUISA.AVX2)
        host = [{"public_ip": "192.0.2.10"}]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(remote_e2e, "remote_cpu_features",
                              return_value=remote_e2e.CPU_ISA_FEATURES[remote_e2e.CPUISA.AVX2]):
                remote_e2e.admit_remote_cpu_runtime(controller, peer, host, Path("/key"), directory)
            evidence = json.loads((directory / "remote-cpu-runtime.json").read_text())
            self.assertEqual(evidence["controller_cpu_isa"], "AVX512")
            self.assertEqual(evidence["remote_cpu_isa"], "AVX2")
            self.assertEqual(evidence["hosts"][0]["supported_cpu_isas"], ["AVX2"])
            with patch.object(remote_e2e, "remote_cpu_features",
                              return_value=remote_e2e.CPU_ISA_FEATURES[remote_e2e.CPUISA.AVX2]):
                with self.assertRaisesRegex(ValueError, "supplied CPU runtime requires AVX512"):
                    remote_e2e.admit_remote_cpu_runtime(
                        controller, remote_e2e.RuntimeImage(controller_identity, remote_e2e.CPUISA.AVX512),
                        host, Path("/key"), directory)
        foreign = remote_runtime_identity(identifier="sha256:foreign", source_tree="other-source-tree")
        with self.assertRaisesRegex(ValueError, "different immutable source tree"):
            remote_e2e.admit_remote_cpu_runtime(
                controller, remote_e2e.RuntimeImage(foreign, remote_e2e.CPUISA.AVX2), host, Path("/key"), Path("/tmp"))

    def test_runtime_image_admission_rejects_stale_or_non_runtime_metadata(self):
        """Image labels are an admission contract, never an advisory transfer tag."""
        admitted = remote_runtime_identity(identifier="sha256:admitted")
        corruptions = (
            ("org.opencontainers.image.revision", "old-revision", "requested full-backend Release source"),
            ("org.llaminar.image_role", "builder", "requested full-backend Release source"),
            ("org.llaminar.cpu_isa", "SSE2", "supported explicit CPU ISA"),
            ("org.llaminar.source_tree", "", "source tree identity"),
        )
        with patch.object(remote_e2e, "image_identity", return_value=admitted):
            runtime = remote_e2e.admit_runtime_image("runtime", "revision")
        self.assertEqual(runtime.cpu_isa, remote_e2e.CPUISA.AVX2)
        for key, value, expected in corruptions:
            with self.subTest(key=key):
                invalid = copy.deepcopy(admitted)
                invalid["labels"][key] = value
                with patch.object(remote_e2e, "image_identity", return_value=invalid), \
                     self.assertRaisesRegex(ValueError, expected):
                    remote_e2e.admit_runtime_image("runtime", "revision")

    def test_peer_evidence_downloads_overlap_and_preserve_exact_records(self):
        import threading
        for collision in (False, True):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                peers = [{"directory": f"/owned/{index}", "container": f"peer-{index}",
                          "public_ip": f"192.0.2.{index}"} for index in (1, 2)]
                rendezvous = threading.Barrier(2)
                payload = b'{"records":[{"name":"exact","value":17}]}\n'

                def execute(argv, **kwargs):
                    if argv[0] != "scp":
                        return ""
                    self.assertIn("-C", argv)
                    self.assertEqual(kwargs["timeout"], 60)
                    target = Path(argv[-1])
                    # A serial implementation times out at this rendezvous.
                    rendezvous.wait(timeout=2)
                    self.assertEqual(list(root.glob("*.json")), [])
                    name = "duplicate.json" if collision else target.name + ".json"
                    (target / name).write_bytes(payload)
                    return ""

                with patch.object(remote_containers, "execute", side_effect=execute):
                    if collision:
                        with self.assertRaisesRegex(ValueError, "duplicated an artifact"):
                            remote_containers.collect_peer_evidence(
                                {"artifact": temp, "key": "/key", "peers": peers})
                        self.assertEqual(list(root.glob("*.json")), [])
                    else:
                        remote_containers.collect_peer_evidence(
                            {"artifact": temp, "key": "/key", "peers": peers})
                        self.assertEqual(sorted(path.name for path in root.glob("*.json")),
                                         ["peer-1.json", "peer-2.json"])
                        self.assertTrue(all(path.read_bytes() == payload for path in root.glob("*.json")))

    def test_case_priority_reorders_without_weakening_complete_projection(self):
        full = full_manifest()
        cases = remote_cases()
        full["cells"][0]["configuration"]["cross_host_e2e"] = cases
        projected = pipeline.cross_host_e2e_projection(full, "revision")
        original = copy.deepcopy(projected)
        canonical = remote_e2e.selected_rows(projected)
        first = [cases[-1]["id"], cases[0]["id"]]
        ordered = remote_e2e.selected_rows(projected, first)
        self.assertEqual([row[1]["id"] for row in ordered],
                         first + [row[1]["id"] for row in canonical if row[1]["id"] not in first])
        self.assertCountEqual(ordered, canonical)
        self.assertEqual(projected, original)
        for invalid in (["not-a-canonical-case"], [first[0], first[0]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                remote_e2e.selected_rows(projected, invalid)
        projected["cells"].append(copy.deepcopy(projected["cells"][0]))
        with self.assertRaisesRegex(ValueError, "repeats a scenario"):
            remote_e2e.selected_rows(projected)

    def test_private_key_cli_destination_survives_pipeline_forwarding(self):
        args = argparse.Namespace(ssh_private_key="/private/task key", azure_subscription="subscription",
                                  azure_pricing="spot", remote_cpu_image="operator-peer-image",
                                  resolved_remote_cpu_image="sha256:built-avx2")
        forwarded = pipeline.cross_host_cli_arguments(args)
        self.assertEqual(forwarded[forwarded.index("--ssh-private-key") + 1], args.ssh_private_key)
        self.assertEqual(pipeline.cross_host_identity(args)["ssh_private_key"], args.ssh_private_key)
        self.assertEqual(forwarded[forwarded.index("--azure-pricing") + 1], "spot")
        self.assertEqual(pipeline.cross_host_identity(args)["azure_pricing"], "spot")
        self.assertEqual(forwarded[forwarded.index("--remote-cpu-image") + 1], "sha256:built-avx2")
        self.assertEqual(pipeline.cross_host_identity(args)["remote_cpu_image"], "operator-peer-image")

    def test_partial_long_context_and_wrong_remote_policy_never_pass(self):
        scenario = remote_cases()[0]
        profile = {"context_length": 4096, "minimum_prompt_tokens": 900, "generation_tokens": 384}
        complete = {"schema": 1, "tier": "full", "complete": True, **profile,
                    "results": [{"passed": True} for _ in range(8)]}
        proof = {"case": scenario["id"], "scenario": scenario, "execution": {"observed": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifacts.write_json(path / "long_context_results.json", complete)
            artifacts.write_json(path / "runtime.cross-host.json", proof)
            remote_e2e.validate_case_evidence(path, scenario, profile)
            for mutate in (lambda long, proof: long.update(complete=False),
                           lambda long, proof: long["results"].pop(),
                           lambda long, proof: long["results"][0].update(passed=False),
                           lambda long, proof: long.update(context_length=8192),
                           lambda long, proof: proof.update(case="other"),
                           lambda long, proof: proof.update(execution={}),
                           lambda long, proof: proof["scenario"].update(frontend="auto-serve")):
                bad_long, bad_proof = copy.deepcopy(complete), copy.deepcopy(proof)
                mutate(bad_long, bad_proof)
                artifacts.write_json(path / "long_context_results.json", bad_long)
                artifacts.write_json(path / "runtime.cross-host.json", bad_proof)
                with self.assertRaises(ValueError):
                    remote_e2e.validate_case_evidence(path, scenario, profile)

    def test_report_artifacts_survive_staging_cleanup_on_success_and_failure(self):
        """Exercise the runner and real Azure lease owner with a fake provider.

        Only network/model execution is replaced. Evidence is written through
        the actual per-cell lifecycle, and cloud retirement must run on both
        ordinary completion and a rejected cell without deleting HTTP logs.
        """
        from test_azure_cross_host_resources import FakeAzure, PUBLIC_KEY, SUBSCRIPTION
        for failure in (None, "cell", "import", "fleet", "plan", "planning-only"):
            planning_only = failure == "planning-only"
            failed = failure is not None and not planning_only
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = root / "model.gguf"
                model.write_bytes(b"fixture")
                private = root / "key"
                private.write_text("fixture: not used by mocked transport")
                private.chmod(0o600)
                public = root / "key.pub"
                public.write_text(PUBLIC_KEY)
                profile = {"context_length": 4096, "minimum_prompt_tokens": 900,
                    "generation_tokens": 384, "readiness_timeout_seconds": 60,
                    "request_timeout_seconds": 60, "thinking_modes": "non-thinking",
                    "movement_evidence": "required"}
                parent = cell()
                parent.update(model_files=[str(model)])
                parent["configuration"].update(model=str(model), e2e=profile,
                    cross_host_e2e=remote_cases())
                manifest_path = root / "manifest.json"
                artifacts.write_json(manifest_path, {"schema": 1, "scope": "cross-host-e2e",
                    "source_revision": "revision", "cells": [parent]})
                report_path = root / "evidence" / "report.json"
                cli = FakeAzure()

                budgets = {}

                def harness(command, env, log, *, budget):
                    directory = Path(env["LLAMINAR_E2E_LOG_DIR"])
                    self.assertIsInstance(budget, remote_e2e.E2ECellBudget)
                    self.assertIs(budgets.setdefault(directory, budget), budget)
                    if "prepare" in command:
                        log.write("preserved plan diagnostic\n")
                        return 1 if failure == "plan" else 0
                    ident = command[command.index("--cross-host-case") + 1]
                    scenario = next(row for row in parent["configuration"]["cross_host_e2e"]
                                    if row["id"] == ident)
                    log.write("preserved HTTP diagnostic\n")
                    artifacts.write_json(directory / "runtime.cross-host.json", {
                        "case": ident, "scenario": scenario, "execution": {"observed": True}})
                    artifacts.write_json(directory / "long_context_results.json", {
                        "schema": 1, "tier": "full", "complete": not failed,
                        **profile, "results": [{"passed": True} for _ in range(8)]})
                    return 0

                @contextmanager
                def fleet(**kwargs):
                    self.assertIn(len(kwargs["hosts"]), (1, 2))
                    self.assertTrue(kwargs["workspace"].is_relative_to(
                        remote_e2e.CROSS_HOST_CONTAINER_SCRATCH))
                    self.assertTrue(kwargs["artifact"].is_relative_to(
                        remote_e2e.CROSS_HOST_CONTAINER_SCRATCH))
                    if failure == "fleet":
                        raise RuntimeError("container fleet rejected the owned remote mount")
                    scenario = next(row for row in parent["configuration"]["cross_host_e2e"]
                                    if row["id"] == kwargs["artifact"].name)
                    plan = kwargs["plan_args"]
                    declared = scenario["server_policy_args"]
                    self.assertEqual(plan[3:3 + len(declared)], declared)
                    applied = json.loads((kwargs["artifact"] / "server-args.json").read_text())
                    if kwargs["frontend_mode"] == "plan-apply":
                        self.assertEqual(applied, ["--config", str(remote_containers.CASE_ROOT / "plan.json")])
                    else:
                        self.assertEqual(applied[:len(declared)], declared)
                    workspace = kwargs["workspace"]
                    workspace.mkdir(parents=True, exist_ok=True)
                    artifacts.write_json(workspace / "plan.json", {
                        "kind": "llaminar.orchestration-config", "configuration": {}})
                    yield root / "unused-launcher"

                @contextmanager
                def connection(*args):
                    yield remote_network.PrivateMPIConnection("10.207.14.1")

                def stage_image(image, hosts, key, directory):
                    (directory / "runtime-import-peer.log").write_text("preserved import diagnostic\n")
                    if failure == "import":
                        raise RuntimeError("runtime import failed")
                    artifacts.write_json(directory / "runtime-import.json", {"source_image": image})
                    return {host["public_ip"]: "sha256:remote-fixture" for host in hosts}

                with patch.object(remote_e2e, "AzureCLI", return_value=cli), \
                     patch.object(remote_e2e, "validate_local_network"), \
                     patch.object(remote_e2e, "private_mpi_connection", side_effect=connection), \
                     patch.object(remote_e2e, "image_identity", return_value=remote_runtime_identity()), \
                     patch.object(remote_e2e, "validate_attached_execution"), \
                     patch.object(remote_e2e, "wait_remote_runtime"), \
                     patch.object(remote_e2e, "remote_cpu_features",
                                  return_value=remote_e2e.CPU_ISA_FEATURES[remote_e2e.CPUISA.AVX2]), \
                     patch.object(remote_e2e, "stage_image", side_effect=stage_image) as image_stage, \
                     patch.object(remote_e2e, "distribute") as model_transfer, \
                     patch.object(remote_e2e, "ssh"), patch.object(remote_e2e, "upload_artifact"), \
                     patch.object(remote_containers, "container_fleet", side_effect=fleet) as fleets, \
                     patch.object(remote_e2e, "run_e2e_process", side_effect=harness), \
                     patch.object(remote_e2e.time, "sleep"):
                    argv = ["--manifest", str(manifest_path), "--source-revision", "revision",
                        "--container-image", "runtime-id", "--models", str(root),
                        "--model-ramdisk-root", str(root / "staging"), "--report", str(report_path),
                        "--azure-subscription", SUBSCRIPTION, "--azure-ssh-source", "8.8.8.8/32",
                        "--azure-ssh-public-key", str(public), "--ssh-private-key", str(private)]
                    if planning_only:
                        argv.append("--planning-only")
                    if failed:
                        expected = ("runtime import failed" if failure == "import" else
                                    "container fleet rejected" if failure == "fleet" else
                                    "cross-host scenario failed")
                        with self.assertRaisesRegex(RuntimeError, expected):
                            remote_e2e.main(argv)
                    else:
                        self.assertEqual(remote_e2e.main(argv), 0)
                self.assertIsNone(cli.group)
                image_stage.assert_called_once()
                if planning_only:
                    # Planning metadata is root-published through MPI. A
                    # remote model transfer would conceal a follower-side
                    # GGUF dependency and turn a control-plane check into a
                    # multi-gigabyte data-plane operation.
                    model_transfer.assert_not_called()
                else:
                    model_transfer.assert_called_once()
                self.assertEqual([len(call.kwargs["hosts"]) for call in fleets.call_args_list],
                                 [] if failure == "import" else [1] if failed else
                                 [1, 2] if planning_only else [1, 1, 2, 2])
                self.assertEqual(list((root / "staging").iterdir()), [])
                report = json.loads(report_path.read_text())
                self.assertEqual(report["complete"], not failed)
                self.assertTrue(report["all_resources_retired"])
                self.assertEqual(len(report["scenarios"]),
                                 0 if failure in ("import", "fleet") else
                                 1 if failed else 2 if planning_only else 4)
                self.assertEqual(report["diagnostic_planning_only"], planning_only)
                if failure in ("import", "fleet"):
                    message = ("runtime import failed" if failure == "import" else
                               "container fleet rejected the owned remote mount")
                    self.assertEqual(report["infrastructure_failure"], {
                        "type": "RuntimeError",
                        "message": message})
                else:
                    self.assertIsNone(report["infrastructure_failure"])
                import_logs = list(report_path.parent.rglob("runtime-import-peer.log"))
                self.assertEqual(len(import_logs), 1)
                self.assertEqual(import_logs[0].read_text(), "preserved import diagnostic\n")
                for observed in report["scenarios"]:
                    evidence = Path(observed["artifacts"])
                    self.assertTrue(evidence.is_relative_to(report_path.parent))
                    if failure in ("plan", "planning-only"):
                        self.assertEqual((evidence / "plan.log").read_text(), "preserved plan diagnostic\n")
                        self.assertFalse((evidence / "harness.log").exists())
                        self.assertFalse((evidence / "long_context_results.json").exists())
                        self.assertEqual(observed["planning_proof"], planning_only)
                        continue
                    self.assertEqual((evidence / "harness.log").read_text(), "preserved HTTP diagnostic\n")
                    self.assertTrue((evidence / "long_context_results.json").is_file())
                receipt = next(report_path.parent.rglob("azure-lease.json"))
                self.assertEqual(json.loads(receipt.read_text())["state"], "retired")

    def test_safe_names_are_bounded_path_components(self):
        self.assertEqual(remote_e2e.safe_name("a/b:c"), "a_b_c")
        self.assertNotIn("/", remote_e2e.safe_name("../"))

    def test_remote_command_failure_preserves_captured_diagnostics(self):
        process = MagicMock()
        process.communicate.return_value = ("bounded stdout", "remote failure")
        process.returncode = 1
        with patch.object(remote_e2e.subprocess, "Popen", return_value=process):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                remote_e2e.run(["ssh"])
        self.assertEqual(caught.exception.output, "bounded stdout")
        self.assertEqual(caught.exception.stderr, "remote failure")

    def test_remote_command_interrupt_retires_its_exact_process_group(self):
        process = MagicMock()
        process.pid = 713
        process.poll.return_value = None
        process.communicate.side_effect = InterruptedError("controller cancellation")
        with patch.object(remote_e2e.subprocess, "Popen", return_value=process), \
             patch.object(remote_e2e.os, "killpg") as terminate:
            with self.assertRaisesRegex(InterruptedError, "controller cancellation"):
                remote_e2e.run(["rsync", "model.gguf"])
        process.wait.assert_called_once_with(timeout=5)
        terminate.assert_called_once_with(713, remote_e2e.signal.SIGTERM)

    def test_mpi_interface_selection_is_exact_and_rejects_ambiguous_inventory(self):
        def interface(name, address, flags=("UP",)):
            return {"ifname": name, "flags": list(flags),
                    "addr_info": [{"family": "inet", "local": address}]}
        inventory = [interface("lo", "127.0.0.1"), interface("eth0", "172.17.0.2"),
                     interface("tun4123", "10.207.14.1")]
        self.assertEqual(remote_containers.mpi_parameters("10.207.14.1", inventory),
                         "oob_tcp_if_include=tun4123\nbtl_tcp_if_include=tun4123\n")
        self.assertEqual(remote_containers.mpi_parameters("10.221.0.4",
                [interface("enP1s0", "10.221.0.4"), interface("tun9", "10.207.14.2")]),
                         "oob_tcp_if_include=enP1s0\nbtl_tcp_if_include=enP1s0\n")
        for invalid in ([], [interface("eth0", "10.207.14.1", ())],
                        [*inventory, interface("other0", "10.207.14.1")],
                        [interface("eth0,lo", "10.207.14.1")]):
            with self.subTest(inventory=invalid), self.assertRaises(ValueError):
                remote_containers.mpi_parameters("10.207.14.1", invalid)
        for address in ("127.0.0.1", "0.0.0.0", "224.0.0.1", "::1"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                remote_containers.mpi_parameters(address, inventory)

    def test_server_frontend_never_runs_plan_inside_readiness(self):
        """A saved-plan server launch must not secretly launch another MPI job."""
        configuration = {"controller": "owned-controller", "frontend": "plan-apply",
                         "host_model": "/models/source.gguf", "container_model": "/models/runtime.gguf",
                         "plan_args": ["plan", "--output", "/run/plan.json"], "peers": []}
        process = Mock()
        process.wait.return_value = 0
        with patch.object(remote_containers.subprocess, "run") as plan, \
             patch.object(remote_containers.subprocess, "Popen", return_value=process) as server, \
             patch.object(remote_containers, "execute"), \
             patch.object(remote_containers, "collect_peer_evidence"):
            self.assertEqual(remote_containers.frontend(configuration, ["serve", "--config", "/run/plan.json"]), 0)
        plan.assert_not_called()
        self.assertEqual(server.call_args.args[0][-3:], ["serve", "--config", "/run/plan.json"])

    def test_plan_preparation_publishes_only_after_success_and_always_retires_mpi(self):
        """Failure/cancellation cannot distribute an incomplete apply document."""
        for error in (None, subprocess.CalledProcessError(1, ["plan"]), InterruptedError("cancelled")):
            with self.subTest(error=error):
                configuration = {"controller": "owned-controller", "frontend": "plan-apply",
                    "host_model": "/models/source.gguf", "container_model": "/models/runtime.gguf",
                    "plan_args": ["plan", "--output", "/run/plan.json"], "key": "/key",
                    "workspace": "/owned", "peers": [{"public_ip": "20.0.0.1", "directory": "/peer-1"},
                                                       {"public_ip": "20.0.0.2", "directory": "/peer-2"}]}
                with patch.object(remote_containers.subprocess, "run", side_effect=error) as plan, \
                     patch.object(remote_containers, "execute") as execute:
                    if error is None:
                        self.assertEqual(remote_containers.prepare_plan(configuration), 0)
                    else:
                        with self.assertRaises(type(error)):
                            remote_containers.prepare_plan(configuration)
                self.assertEqual(plan.call_args.args[0][-3:], ["plan", "--output", "/run/plan.json"])
                copies = [call for call in execute.call_args_list if call.args[0][0] == "scp"]
                self.assertEqual(len(copies), 2 if error is None else 0)
                self.assertEqual(execute.call_args.args[0], ["docker", "exec", "owned-controller", "python3",
                    remote_containers.CONTROL, "stop", remote_containers.CONFIG])

    def test_auto_serve_cannot_request_a_separate_plan_job(self):
        with patch.object(remote_containers.subprocess, "run") as plan:
            with self.assertRaises(ValueError):
                remote_containers.prepare_plan({"frontend": "auto-serve"})
        plan.assert_not_called()

    def test_container_fleet_encloses_mpi_and_retires_every_acquired_owner(self):
        for backend in ("cuda", "rocm"):
            for count in (1, 2):
                for fail_start in (False, True):
                    with self.subTest(backend=backend, count=count, fail_start=fail_start), \
                         tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        hosts = [{"public_ip": f"20.0.0.{i + 1}", "private_ip": f"10.1.0.{i + 2}",
                                  "runtime_image": "sha256:remote-fixture"}
                                 for i in range(count)]
                        acquired, retired, creates = [], [], []

                        def execute(command, **kwargs):
                            if command[0] == "ssh":
                                command = shlex.split(command[-1])
                            if command[:4] == ["ip", "-j", "-4", "address"]:
                                return json.dumps([{"ifname": f"nic{i}", "flags": ["UP"],
                                    "addr_info": [{"family": "inet", "local": f"10.1.0.{i}"}]}
                                    for i in range(1, 4)])
                            if command[:2] == ["docker", "create"]:
                                creates.append(command)
                                acquired.append(command[command.index("--name") + 1])
                            if command[:2] == ["docker", "start"] and fail_start and len(acquired) == 2:
                                raise subprocess.CalledProcessError(1, command)
                            if command[:2] == ["docker", "rm"]:
                                retired.append(command[-1])
                            return ""

                        with patch.object(remote_containers, "execute", side_effect=execute), \
                             patch.object(docker_paths, "device_args", return_value=["--network", "host"]) as devices, \
                             patch.object(docker_paths, "containing_container", return_value={"Id": "controller-ns"}), \
                             patch.object(docker_paths, "host_path", side_effect=str):
                            def launch():
                                with remote_containers.container_fleet(image="sha256:fixture", hosts=hosts,
                                        key=root / "key", workspace=root / "launch", artifact=root / "evidence",
                                        model_dir=root / "models", model_name="fixture.gguf", frontend_mode="auto-serve",
                                        plan_args=[], backend=backend, controller_address="10.1.0.1",
                                        continuation_devices=1) as launcher:
                                    self.assertEqual(subprocess.run(["sh", "-n", str(launcher)]).returncode, 0)
                                    config = json.loads((launcher.parent / "fleet.json").read_text())
                                    self.assertEqual(len(config["peers"]), count)
                                    self.assertEqual(len((launcher.parent / "hosts").read_text().splitlines()), count + 1)
                                    self.assertEqual((launcher.parent / "mpi.conf").read_text(),
                                        "oob_tcp_if_include=nic1\nbtl_tcp_if_include=nic1\n")
                                    for index in range(count):
                                        self.assertEqual((launcher.parent / f"mpi-peer-{index+1}.conf").read_text(),
                                            f"oob_tcp_if_include=nic{index+2}\nbtl_tcp_if_include=nic{index+2}\n")
                            if fail_start:
                                with self.assertRaises(subprocess.CalledProcessError):
                                    launch()
                            else:
                                launch()
                            devices.assert_called_once_with("sha256:fixture", {"cuda": "CUDA", "rocm": "ROCm"}[backend])
                        self.assertEqual(retired, list(reversed(acquired)))
                        for index, command in enumerate(creates):
                            self.assertEqual(command[-3:], ["/bin/sleep",
                                "sha256:fixture" if index == 0 else "sha256:remote-fixture", "infinity"])
                        self.assertIn("container:controller-ns", creates[0])
                        visibility = "CUDA_VISIBLE_DEVICES=0" if backend == "cuda" else "ROCR_VISIBLE_DEVICES=0"
                        self.assertIn(visibility, creates[0])
                        for command in creates[1:]:
                            self.assertNotIn(visibility, command)

    def test_mpi_daemon_runs_inside_the_declared_peer_image(self):
        peer = {"mpi_ip": "10.1.0.2", "public_ip": "20.0.0.1", "container": "owned-peer"}
        with patch.object(remote_containers.os, "execvp") as launch:
            remote_containers.daemon({"peers": [peer]}, ["10.1.0.2", "orted -mca ess env"])
        self.assertEqual(shlex.split(launch.call_args.args[1][-1]),
            ["docker", "exec", "owned-peer", "/bin/sh", "-c", "orted -mca ess env"])
        with self.assertRaisesRegex(ValueError, "outside the admitted"):
            remote_containers.daemon({"peers": [peer]}, ["10.1.0.3", "orted"])

    def test_cloud_controller_environment_never_reaches_inference(self):
        self.assertEqual(remote_containers.runtime_environment({
            "LLAMINAR_PERF_STATS_JSON": "rank-{rank}.json", "LLAMINAR_ENABLE_PERF_STATS": "1",
            "LLAMINAR_AZURE_AUTH_TOKEN": "secret", "LLAMINAR_AZURE_SSH_PRIVATE_KEY": "private-path",
            "AZURE_CLIENT_SECRET": "secret", "PATH": "/bin"}),
            {"LLAMINAR_PERF_STATS_JSON": "rank-{rank}.json", "LLAMINAR_ENABLE_PERF_STATS": "1"})

    def test_image_import_authenticates_content_across_docker_stores(self):
        source = {"Id": "sha256:source", "Config": {"Env": ["POLICY=correct"], "User": "llaminar"},
                  "RootFS": {"Type": "layers", "Layers": ["sha256:layer1", "sha256:layer2"]},
                  "Architecture": "amd64", "Os": "linux"}
        for corruption in (None, "layer", "order", "config", "architecture"):
            remote = json.loads(json.dumps(source))
            remote["Id"] = "sha256:containerd-manifest"
            remote["Config"]["Entrypoint"] = None
            if corruption == "layer": remote["RootFS"]["Layers"][0] = "sha256:foreign"
            if corruption == "order": remote["RootFS"]["Layers"].reverse()
            if corruption == "config": remote["Config"]["Env"] = ["POLICY=wrong"]
            if corruption == "architecture": remote["Architecture"] = "arm64"
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp:
                commands = []
                def local(command, **kwargs):
                    commands.append(command)
                    return json.dumps([source]) if command[:3] == ["docker", "image", "inspect"] else ""
                with patch.object(remote_e2e, "run", side_effect=local), \
                     patch.object(remote_e2e, "cached_runtime_image", return_value=None), \
                     patch.object(remote_e2e, "import_runtime_archives") as import_archives, \
                     patch.object(remote_e2e, "ssh", return_value=json.dumps([remote])), \
                     patch.object(remote_e2e, "upload_artifact"), \
                     patch.object(remote_e2e, "distribute"), \
                     patch.object(remote_e2e, "image_identity", return_value={"id": source["Id"]}):
                    if corruption:
                        with self.assertRaisesRegex(ValueError, "different runtime"):
                            remote_e2e.stage_image(source["Id"], [{"public_ip": "peer"}], Path("/key"), Path(temp))
                    else:
                        self.assertEqual(remote_e2e.stage_image(source["Id"], [{"public_ip": "peer"}], Path("/key"), Path(temp)),
                                         {"peer": remote["Id"]})
                        self.assertTrue((Path(temp) / "runtime-import.json").is_file())
                tag = next(command[-1] for command in commands if command[:2] == ["docker", "tag"])
                self.assertIn(["docker", "save", "-o", str(Path(temp) / "runtime-image.tar"), tag], commands)
                self.assertEqual(commands[-1], ["docker", "image", "rm", tag])
                import_archives.assert_called_once_with(
                    [{"public_ip": "peer"}], Path("/key"), Path(temp))

    def test_full_runtime_cache_requires_exact_content_and_avoids_export_and_import(self):
        source = {"Id": "sha256:source", "Config": {"Env": ["POLICY=correct"]},
                  "RootFS": {"Type": "layers", "Layers": ["sha256:layer"]},
                  "Architecture": "amd64", "Os": "linux"}
        remote = {**source, "Id": "sha256:daemon-local"}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(remote_e2e, "run", return_value=json.dumps([source])) as local, \
             patch.object(remote_e2e, "ssh", side_effect=[remote["Id"] + "\n", json.dumps([remote])]) as commands, \
             patch.object(remote_e2e, "distribute") as transfer:
            self.assertEqual(remote_e2e.stage_image(source["Id"], [{"public_ip": "peer"}], Path("/key"), Path(temp)),
                             {"peer": remote["Id"]})
            transfer.assert_not_called()
            local.assert_called_once_with(["docker", "image", "inspect", source["Id"]])
            self.assertEqual(len(commands.call_args_list), 2)
            receipt = json.loads((Path(temp) / "runtime-import.json").read_text())
            self.assertEqual(receipt["cached_hosts"], ["peer"])
        remote["Config"] = {"Env": ["POLICY=wrong"]}
        with patch.object(remote_e2e, "ssh", side_effect=[remote["Id"] + "\n", json.dumps([remote])]):
            self.assertIsNone(remote_e2e.cached_runtime_image("peer", Path("/key"), artifacts.runtime_image_content(source)))
        with patch.object(remote_e2e, "ssh", side_effect=[remote["Id"] + "\n", "[]"]):
            with self.assertRaisesRegex(ValueError, "requested identities"):
                remote_e2e.cached_runtime_image("peer", Path("/key"), artifacts.runtime_image_content(source))

    def test_retained_archive_is_outside_boot_cleaned_temporary_and_model_directories(self):
        archive = Path(remote_e2e.REMOTE_IMAGE_ARCHIVE)
        self.assertTrue(archive.is_relative_to("/home/llaminar/.cache/llaminar-cross-host"))
        self.assertFalse(archive.is_relative_to("/tmp"))
        self.assertFalse(archive.is_relative_to("/opt/llaminar-models"))

    def test_model_staging_checks_complete_shards_and_duplicate_basenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "one.gguf"
            source.write_text("fixture")
            parent = {"model_files": [str(source)]}
            (root / "stage").mkdir()
            staged = remote_e2e.stage_models([parent], [], root / "unused-key", root / "stage")
            self.assertFalse((staged / source.name).is_symlink())
            self.assertEqual((staged / source.name).read_text(), source.read_text())
            self.assertIs(remote_e2e.stage_models([parent], [], root / "unused-key", root / "stage",
                                                  model_dir=staged), staged)
            peers = [{"public_ip": "peer", "private_ip": "10.1.0.2"}]
            with patch.object(remote_e2e, "ssh"), patch.object(remote_e2e, "distribute") as distribute:
                remote_e2e.stage_models([parent], peers, root / "unused-key", root / "stage", model_dir=staged)
            self.assertEqual(distribute.call_args.args[:3],
                (staged / source.name, "/opt/llaminar-models/one.gguf", peers))
            other = root / "other" / source.name
            other.parent.mkdir()
            other.write_text("different")
            (root / "stage2").mkdir()
            with self.assertRaises(ValueError):
                remote_e2e.stage_models([{"model_files": [str(source), str(other)]}], [], root / "unused-key", root / "stage2")

    def test_remote_runtime_probe_uses_fresh_authenticated_connection(self):
        with patch.object(remote_e2e, "ssh", return_value="") as probe:
            remote_e2e.wait_remote_runtime("203.0.113.10", Path("/tmp/key"), timeout=1)
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[:2], ("203.0.113.10", Path("/tmp/key")))

    def test_loader_consumes_the_canonical_projection_shape(self):
        full = full_manifest()
        full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        projected = pipeline.cross_host_e2e_projection(full, "revision")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross-host.json"
            artifacts.write_json(path, projected)
            loaded = remote_e2e.load_manifest(path, "revision")
            self.assertEqual(loaded["cells"][0]["configuration"]["model"],
                             full["cells"][0]["configuration"]["model"])

class RemoteImageImportTests(unittest.TestCase):
    """Prove concurrent launch and bounded cleanup without Docker, SSH or devices."""

    def test_all_peers_launch_before_polling_and_every_client_is_joined(self):
        processes = [Mock(pid=1001), Mock(pid=1002)]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(remote_e2e.subprocess, "Popen", side_effect=processes) as launch, \
             patch.object(remote_e2e.os, "killpg") as kill:
            def complete():
                self.assertEqual(launch.call_count, 2, "imports must not serialize across peers")
                return 0
            for process in processes:
                process.poll.side_effect = complete
            remote_e2e.import_runtime_archives(
                [{"public_ip": "peer-a"}, {"public_ip": "peer-b"}], Path("/key"), Path(temp))
            kill.assert_not_called()
            for process in processes:
                process.wait.assert_called_once_with(timeout=5)
            for call in launch.call_args_list:
                self.assertTrue(call.kwargs["start_new_session"])
                self.assertTrue(call.kwargs["stdout"].closed)
                self.assertEqual(shlex.split(call.args[0][-1]),
                                 ["docker", "load", "--input", remote_e2e.REMOTE_IMAGE_ARCHIVE])

    def test_failure_deadline_and_cancellation_retire_other_imports(self):
        for reason, expected in (("failure", RuntimeError), ("deadline", TimeoutError),
                                 ("cancel", InterruptedError), ("spawn", OSError)):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                processes = [Mock(pid=1001), Mock(pid=1002)]
                for process in processes:
                    process.poll.return_value = None
                if reason == "failure":
                    processes[0].poll.return_value = 7
                events = []
                def kill(pid, signum):
                    events.append(("signal", pid, signum))
                    processes[pid - 1001].poll.return_value = -signum
                for process in processes:
                    process.wait.side_effect = lambda timeout, pid=process.pid: events.append(("join", pid))
                spawns = [processes[0], OSError("spawn failed")] if reason == "spawn" else processes
                with patch.object(remote_e2e.subprocess, "Popen", side_effect=spawns) as launch, \
                     patch.object(remote_e2e.os, "killpg", side_effect=kill), \
                     patch.object(remote_e2e.time, "monotonic", side_effect=[0, 2]), \
                     patch.object(remote_e2e.time, "sleep", side_effect=InterruptedError("cancelled")):
                    with self.assertRaises(expected):
                        remote_e2e.import_runtime_archives(
                            [{"public_ip": "peer-a"}, {"public_ip": "peer-b"}],
                            Path("/key"), Path(temp), timeout=1 if reason == "deadline" else 600)
                expected_owned = processes[:1] if reason == "spawn" else processes
                for process in expected_owned:
                    process.wait.assert_called_once_with(timeout=5)
                signals = [index for index, event in enumerate(events) if event[0] == "signal"]
                joins = [index for index, event in enumerate(events) if event[0] == "join"]
                self.assertLess(max(signals), min(joins), "signal every live peer before waiting")
                for call in launch.call_args_list:
                    self.assertTrue(call.kwargs["stdout"].closed)

    def test_client_ignoring_termination_is_killed_and_joined(self):
        process = Mock(pid=1001)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned ssh", 5), 0]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(remote_e2e.subprocess, "Popen", return_value=process), \
             patch.object(remote_e2e.os, "killpg") as kill, \
             patch.object(remote_e2e.time, "sleep", side_effect=InterruptedError("cancelled")):
            with self.assertRaises(InterruptedError):
                remote_e2e.import_runtime_archives([{"public_ip": "peer"}], Path("/key"), Path(temp))
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(1001, signal.SIGTERM), (1001, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 2)

    def test_duplicate_or_ambiguous_peers_and_invalid_deadlines_fail_before_launch(self):
        for peers, timeout in (([], 600), (["same", "same"], 600),
                               (["a/b", "a?b"], 600), (["peer"], 0), (["peer"], 601)):
            with self.subTest(peers=peers, timeout=timeout), \
                 patch.object(remote_e2e.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    remote_e2e.import_runtime_archives(
                        [{"public_ip": peer} for peer in peers], Path("/key"), Path("/unused"), timeout)
                launch.assert_not_called()

    def test_unreaped_client_does_not_prevent_joining_other_owned_clients(self):
        processes = [Mock(pid=1001), Mock(pid=1002)]
        for process in processes:
            process.poll.return_value = None
        processes[0].wait.side_effect = subprocess.TimeoutExpired("owned ssh", 5)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(remote_e2e.subprocess, "Popen", side_effect=processes) as launch, \
             patch.object(remote_e2e.os, "killpg") as kill, \
             patch.object(remote_e2e.time, "sleep", side_effect=InterruptedError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "client retirement failed"):
                remote_e2e.import_runtime_archives(
                    [{"public_ip": "peer-a"}, {"public_ip": "peer-b"}], Path("/key"), Path(temp))
        self.assertEqual(processes[0].wait.call_count, 2)
        processes[1].wait.assert_called_once_with(timeout=5)
        self.assertIn((1002, signal.SIGTERM), [call.args for call in kill.call_args_list])
        self.assertTrue(all(call.kwargs["stdout"].closed for call in launch.call_args_list))


class CrossHostArtifactTests(unittest.TestCase):
    """Exercise real streaming/retirement locally; mock only the Azure SSH hop."""

    @staticmethod
    def local_command(peer, key, arguments):
        """Run the exact peer helper without a device, image, SSH key or VM."""
        return [sys.executable, "-u", str(Path(remote_artifacts.__file__)), *arguments]

    def test_concurrent_private_reads_and_owner_close(self):
        from concurrent.futures import ThreadPoolExecutor
        from urllib.error import HTTPError, URLError
        from urllib.request import urlopen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            payload = bytes(range(256)) * 4096 + b"final unequal chunk"
            source.write_bytes(payload)
            peer = {"private_ip": "127.0.0.1"}
            with patch.object(remote_artifacts, "remote_command", side_effect=self.local_command):
                with remote_artifacts.private_source(peer, root / "unused-key", str(source),
                                                    [peer], len(payload)) as endpoint:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(endpoint.url, timeout=1)
                    self.assertEqual(error.exception.code, 403)
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        list(pool.map(lambda index: remote_artifacts.fetch(endpoint, root / f"copy-{index}"), range(3)))
                    for index in range(3):
                        self.assertEqual((root / f"copy-{index}").read_bytes(), payload)
                    self.assertFalse(list(root.glob("*.partial-*")))
                with self.assertRaises(URLError):
                    urlopen(endpoint.url, timeout=1)

    def test_receiver_failure_retires_source_and_preserves_existing_destination(self):
        from urllib.error import URLError
        from urllib.request import urlopen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "destination"
            source.write_bytes(b"admitted input")
            destination.write_bytes(b"unrelated existing data")
            peer = {"private_ip": "127.0.0.1"}
            with patch.object(remote_artifacts, "remote_command", side_effect=self.local_command):
                with self.assertRaises(FileExistsError):
                    with remote_artifacts.private_source(peer, root / "key", str(source),
                                                        [peer], source.stat().st_size) as endpoint:
                        remote_artifacts.fetch(endpoint, destination)
                self.assertEqual(destination.read_bytes(), b"unrelated existing data")
                with self.assertRaises(URLError):
                    urlopen(endpoint.url, timeout=1)

    def test_incomplete_transfer_never_publishes_a_model(self):
        import io
        from unittest.mock import MagicMock
        for declared in ("3", "16"):
            with self.subTest(content_length=declared), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                endpoint = remote_artifacts.ArtifactEndpoint("http://private/artifact", "token", 16)
                response = io.BytesIO(b"abc")
                response.status = 200
                response.geturl = lambda: endpoint.url
                response.headers = {"Content-Length": declared}
                opener = MagicMock()
                opener.open.return_value = response
                with patch.object(remote_artifacts, "build_opener", return_value=opener), \
                     self.assertRaisesRegex(ValueError, "byte count|complete input"):
                    remote_artifacts.fetch(endpoint, root / "model.gguf")
                self.assertEqual(list(root.iterdir()), [])

    def test_distribution_crosses_wan_once_for_arbitrary_peer_count(self):
        from unittest.mock import Mock
        for count in (1, 2, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "model.gguf"
                source.write_bytes(b"same complete input")
                peers = [{"name": f"cpu-{i}", "public_ip": f"203.0.113.{i+1}", "private_ip": f"10.1.0.{i+1}"}
                         for i in range(count)]
                upload = Mock()
                retired = []
                @contextmanager
                def private_source(*args):
                    try:
                        self.assertEqual(args[0], peers[0])
                        self.assertEqual(args[3], peers[1:])
                        yield remote_artifacts.ArtifactEndpoint("http://private/artifact", "token", source.stat().st_size)
                    finally:
                        retired.append(True)
                with patch.object(remote_artifacts, "private_source", side_effect=private_source), \
                     patch.object(remote_artifacts, "remote_command", return_value=["mock-fetch"]), \
                     patch.object(remote_artifacts, "remote_stamp", side_effect=[None] * count +
                                  [remote_artifacts.file_stamp(source)]), \
                     patch.object(remote_artifacts.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 0, "")) as fetch:
                    remote_artifacts.distribute(source, "/opt/models/model.gguf", peers, root / "key", upload)
                upload.assert_called_once_with(peers[0]["public_ip"], root / "key", source, "/opt/models/model.gguf")
                self.assertEqual(fetch.call_count, count - 1)
                self.assertEqual(retired, [True] if count > 1 else [])

    def test_unchanged_owned_replicas_require_no_transfer(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.gguf"
            source.write_bytes(b"stable input")
            peers = [{"name": f"cpu-{i}", "public_ip": f"203.0.113.{i+1}", "private_ip": f"10.1.0.{i+1}"}
                     for i in range(2)]
            upload = Mock()
            with patch.object(remote_artifacts, "remote_stamp", return_value=remote_artifacts.file_stamp(source)), \
                 patch.object(remote_artifacts, "private_source") as transfer:
                remote_artifacts.distribute(source, "/opt/models/model.gguf", peers, Path("/key"), upload)
            upload.assert_not_called()
            transfer.assert_not_called()

    def test_owned_replacement_is_atomic_and_preserves_source_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "destination"
            source.write_bytes(b"new exact input")
            destination.write_bytes(b"previous owned cache")
            previous = remote_artifacts.file_stamp(destination)
            peer = {"private_ip": "127.0.0.1"}
            with patch.object(remote_artifacts, "remote_command", side_effect=self.local_command):
                with remote_artifacts.private_source(peer, root / "key", str(source), [peer], source.stat().st_size) as endpoint:
                    remote_artifacts.fetch(endpoint, destination, replacing=previous,
                                           mtime_ns=source.stat().st_mtime_ns)
                    with self.assertRaises(FileExistsError):
                        remote_artifacts.fetch(endpoint, destination, replacing=previous)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(remote_artifacts.file_stamp(destination), remote_artifacts.file_stamp(source))
            self.assertFalse(list(root.glob("*.partial-*")))

    def test_owned_cache_rejects_symlinks_even_when_target_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"input")
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                remote_artifacts.file_stamp(link)

    def test_incremental_upload_preserves_times_without_inplace_or_hash_gate(self):
        with patch.object(remote_e2e, "run", return_value="transfer evidence") as upload:
            remote_e2e.upload_artifact("20.0.0.1", Path("/keys/private"), Path("/source/model"), "/opt/models/model")
        argv = upload.call_args.args[0]
        self.assertEqual(argv[0], "rsync")
        self.assertIn("--times", argv)
        self.assertIn("--modify-window=-1", argv)
        self.assertNotIn("--inplace", argv)
        self.assertNotIn("--checksum", argv)

    def test_distribution_rejects_nonprivate_or_duplicate_peers_before_upload(self):
        from unittest.mock import Mock
        for addresses in (("8.8.8.8",), ("127.0.0.1",), ("0.0.0.0",), ("10.1.0.1", "10.1.0.1")):
            with self.subTest(addresses=addresses), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "model.gguf"
                source.write_bytes(b"payload")
                peers = [{"public_ip": f"203.0.113.{i+1}", "private_ip": address}
                         for i, address in enumerate(addresses)]
                upload = Mock()
                with self.assertRaisesRegex(ValueError, "distinct private"):
                    remote_artifacts.distribute(source, "/opt/models/model.gguf", peers, Path("/key"), upload)
                upload.assert_not_called()


class CertificateTests(unittest.TestCase):
    def test_certificate_layer_cannot_hide_a_runtime_overwrite(self):
        pipeline.validate_certificate_changes("C /usr/local/share\nA /usr/local/share/llaminar/certificates/production.json\n")
        for invalid in ("C /usr/local/bin/llaminar2", "A /usr/local/lib/libllaminar2_core.so",
                        "D /usr/local/share/llaminar/certificates/production.json",
                        "A /usr/local/share/llaminar/certificates/unexpected.sh"):
            with self.subTest(change=invalid), self.assertRaises(ValueError):
                pipeline.validate_certificate_changes(invalid)

    def setUp(self):
        install_corpus_io_fixture(self, mock_stat=True)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.source = {"revision": "revision", "tree": "tree", "dirty": False}
        self.images = image_pair(self.source)
        self.documents = {
            "manifest": manifest(),
            "all-cells": full_manifest(),
            "prerequisites/prerequisites": prerequisite_report(),
            "generation/report": generation_report(),
            "e2e": e2e_report(),
            "benchmarks": {"passed": True, "complete": True, "image": "runtime-id",
                           "diagnostic": False, "e2e_report_digest": artifacts.digest(e2e_report()),
                           "manifest_digest": artifacts.digest(manifest()), "cells": [{"case": "cell"}]}}
        self.documents.update(non_remote_documents(self.documents["all-cells"]))

    def write(self):
        for name, document in self.documents.items():
            artifacts.write_json(self.path / f"{name}.json", document)

    def test_exact_complete_evidence_can_certify(self):
        self.write()
        result = pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")
        self.assertTrue(result["certified"])
        self.assertEqual(result["tested_image"], "runtime-id")
        self.assertEqual(result["inventory_digest"], artifacts.digest(full_manifest()))
        self.assertEqual(result["generation"]["exact_cells"], 2)
        self.assertIsNone(result["diagnostic_mathematical_parity"])

    def test_cross_host_report_requires_exact_routes_transport_and_retirement(self):
        """Remote certification is an E2E proof, not an empty cloud receipt."""
        full = full_manifest()
        full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        remote = pipeline.cross_host_e2e_projection(full, self.source["revision"])
        image = self.images["runtime"]["id"]
        expected = [scenario for scenario in remote_cases()]
        report = {"schema": 1, "eligible": True, "complete": True,
                  "source_revision": self.source["revision"], "image": image,
                  "manifest_digest": artifacts.digest(remote), "all_resources_retired": True,
                  "scenarios": [{"id": scenario["id"], "frontend": scenario["frontend"],
                                 "topology": scenario["topology"], "return_code": 0,
                                 "outcome": "passed", "resource_retired": True,
                                 "transport_proof": True, "http_proof": True}
                                for scenario in expected]}
        pipeline.validate_cross_host_report(report, remote, image, self.source["revision"])
        for field, value in (("all_resources_retired", False), ("transport_proof", False),
                             ("http_proof", False), ("return_code", 1), ("outcome", "failed")):
            bad = copy.deepcopy(report)
            (bad if field == "all_resources_retired" else bad["scenarios"][0])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipeline.validate_cross_host_report(bad, remote, image, self.source["revision"])

    def test_pipeline_owned_cross_host_manifest_requires_a_report(self):
        """A build receipt cannot silently skip the phase before benchmarks."""
        full = full_manifest()
        full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        remote = pipeline.cross_host_e2e_projection(full, self.source["revision"])
        self.documents["all-cells"] = full
        self.documents["manifest"] = pipeline.e2e_projection(full, self.source["revision"])
        self.documents.pop("cross-host-e2e")
        self.write()
        artifacts.write_json(self.path / "cross-host-manifest.json", remote)
        with self.assertRaisesRegex(ValueError, "no evidence report"):
            pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_missing_remote_manifest_cannot_erase_declared_remote_obligations(self):
        full = full_manifest()
        full["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        self.documents["all-cells"] = full
        self.documents["manifest"] = pipeline.e2e_projection(full, self.source["revision"])
        self.documents.pop("cross-host-manifest")
        self.documents.pop("cross-host-e2e")
        self.write()
        with self.assertRaisesRegex(ValueError, "no canonical manifest"):
            pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_tagged_projection_cannot_replace_or_differ_from_full_inventory(self):
        """Even internally consistent E2E results need their full-inventory parent."""
        original = copy.deepcopy(self.documents)
        for index, mutate in enumerate((
                lambda d: d.update({"all-cells": manifest()}),
                lambda d: d["all-cells"]["cells"][0]["configuration"]["e2e"].update(context_length=4096),
                lambda d: d["manifest"]["cells"].append(cell("extra")),
                lambda d: d["generation/report"].update(selected=1),
                lambda d: d["generation/report"].update(selected=True))):
            self.documents = copy.deepcopy(original)
            mutate(self.documents)
            self.write()
            with self.subTest(index=index), self.assertRaises(ValueError):
                pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_numerical_membership_cannot_be_replaced_by_counts_or_another_selection(self):
        """A green summary/count cannot hide a missing or failed exact member."""
        original = parity_report()
        for index, mutate in enumerate((
                lambda p: p.pop("campaigns"), lambda p: p.update(campaigns=[]),
                lambda p: p["campaigns"][0].update(gtest_cases=["cell", "cell"]),
                lambda p: p["campaigns"][0].update(gtest_cases=["cell", "different"]),
                lambda p: p["campaigns"][0].update(campaign="different"),
                lambda p: p["campaigns"][0].update(return_code=1),
                lambda p: p["campaigns"][0].update(outcome="cancelled"),
                lambda p: p["campaigns"][0].update(artifact_contract_passed=False))):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                pipeline.validate_parity(changed, full_manifest()["cells"])

    def test_every_missing_report_blocks_certification(self):
        for name in self.documents:
            self.write()
            (self.path / f"{name}.json").unlink()
            with self.subTest(name=name), self.assertRaises((OSError, ValueError)):
                pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_regression_subset_wrong_image_or_stale_manifest_cannot_certify(self):
        original = copy.deepcopy(self.documents)
        for bad in ({"passed": False}, {"complete": False}, {"image": "other-image"},
                    {"diagnostic": True}, {"diagnostic": None}, {"e2e_report_digest": "stale"},
                    {"manifest_digest": "stale"}, {"cells": []}, {"cells": [{"case": "wrong"}]}):
            self.documents = copy.deepcopy(original)
            self.documents["benchmarks"].update(bad)
            self.write()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_exit_zero_e2e_with_changed_configuration_is_rejected(self):
        self.documents["e2e"]["cells"][0]["configuration"]["e2e"]["server_args"] = []
        self.write()
        with self.assertRaises(ValueError):
            pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_missing_unit_preflight_never_certifies(self):
        original = copy.deepcopy(self.documents)
        for bad in ({"preflight_tests": []}, {"preflight_test_count": 0},
                    {"preflight_return_code": 1}):
            self.documents = copy.deepcopy(original)
            self.documents["prerequisites/prerequisites"].update(bad)
            self.write()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.certificates(self.source, self.images, self.path, cpu_isa="AVX512")

    def test_local_and_dirty_runs_cannot_publish_or_commit(self):
        with patch.dict(pipeline.os.environ, {}, clear=True), patch.object(pipeline.subprocess, "check_output") as execute:
            with self.assertRaises(ValueError):
                pipeline.publish(self.source, self.path, {})
            execute.assert_not_called()

    def test_image_requires_release_both_backends_correct_tree_and_isa(self):
        labels = {"org.opencontainers.image.revision": "revision", "org.llaminar.source_tree": "tree",
                  "org.llaminar.build_type": "Release", "org.llaminar.cpu_isa": "AVX512",
                  "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON", "org.llaminar.image_role": "runtime"}
        pipeline.require_image({"labels": labels}, self.source, "AVX512", pipeline.ImageRole.RUNTIME)
        for field in labels:
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipeline.require_image({"labels": {**labels, field: "wrong"}}, self.source,
                                       "AVX512", pipeline.ImageRole.RUNTIME)

    def test_builder_identity_is_required_before_certifying_complete_reports(self):
        """A matching runtime cannot excuse a stale or untested builder image."""
        self.write()
        for key in self.images["builder"]["labels"]:
            changed = copy.deepcopy(self.images)
            changed["builder"]["labels"][key] = "wrong"
            with self.subTest(label=key), self.assertRaises(ValueError):
                pipeline.certificates(self.source, changed, self.path, cpu_isa="AVX512")
        # The caller's shipping slot remains authority even if both images
        # consistently advertise the other ISA.
        with self.assertRaises(ValueError):
            pipeline.certificates(self.source, image_pair(self.source, "AVX2"), self.path, cpu_isa="AVX512")


class ImageIdentityTests(unittest.TestCase):
    """Expected source/ISA/role comes from admission, not the inspected image."""

    def test_docker_roles_export_the_actual_build_arguments(self):
        """Guard producer metadata; runtime image admission tests its consumers."""
        dockerfile = (ROOT / "Dockerfile").read_text()
        builder, runtime = dockerfile.split("FROM toolchain AS builder", 1)[1].split(
            "FROM ubuntu:24.04 AS runtime", 1)
        fields = {
            "org.opencontainers.image.revision": "VCS_REF",
            "org.llaminar.source_tree": "LLAMINAR_SOURCE_TREE",
            "org.llaminar.cpu_isa": "LLAMINAR_CPU_ISA",
            "org.llaminar.build_type": "LLAMINAR_BUILD_TYPE",
            "org.llaminar.cuda": "LLAMINAR_ENABLE_CUDA",
            "org.llaminar.rocm": "LLAMINAR_ENABLE_ROCM",
        }
        for role, stage in ((pipeline.ImageRole.BUILDER, builder),
                            (pipeline.ImageRole.RUNTIME, runtime)):
            with self.subTest(role=role):
                self.assertIn(f'org.llaminar.image_role="{role.value}"', stage)
                for label, argument in fields.items():
                    self.assertIn(f'{label}="${{{argument}}}"', stage)
        self.assertIn('org.llaminar.integration_skipped="${LLAMINAR_SKIP_INTEGRATION}"', builder)

    def test_both_shipping_pairs_require_every_declared_build_property(self):
        """Exercise both roles and both ISAs, including absent identity fields."""
        source = {"revision": "revision", "tree": "tree", "dirty": False}
        for isa in pipeline.SHIPPING_ISAS:
            images = image_pair(source, isa)
            pipeline.require_image_pair(images, source, isa)
            for role in pipeline.ImageRole:
                for key in images[role.value]["labels"]:
                    for remove in (False, True):
                        changed = copy.deepcopy(images)
                        if remove:
                            changed[role.value]["labels"].pop(key)
                        else:
                            changed[role.value]["labels"][key] = "wrong"
                        with self.subTest(isa=isa, role=role, field=key, remove=remove), self.assertRaises(ValueError):
                            pipeline.require_image_pair(changed, source, isa)
        with self.assertRaises(TypeError):
            pipeline.require_image(images["runtime"], source, isa, "runtime")
        with self.assertRaises(ValueError):
            pipeline.require_image_pair(images, source, "native")

    def test_missing_swapped_and_shared_image_roles_are_rejected(self):
        """Two role names cannot authenticate one image or the opposite sibling."""
        source = {"revision": "revision", "tree": "tree", "dirty": False}
        images = image_pair(source)
        for changed in ({}, {"runtime": images["runtime"]},
                        {"runtime": images["builder"], "builder": images["runtime"]},
                        {**images, "extra": images["builder"]},
                        {**images, "builder": {**images["builder"], "id": images["runtime"]["id"]}}):
            with self.subTest(roles=list(changed)), self.assertRaises(ValueError):
                pipeline.require_image_pair(changed, source, "AVX512")

    def test_bad_builder_stops_before_building_runtime(self):
        """Fail at the first wrong image, rather than paying for a second build."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            source = {"revision": "revision", "tree": "tree", "dirty": False}
            args = argparse.Namespace(cpu_isa="AVX512")
            wrong = image_pair(source, "AVX2")["builder"]
            with patch.object(pipeline, "run") as execute, \
                 patch.object(pipeline, "image_identity", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "builder image"):
                    pipeline.build(args, source, root)
            self.assertEqual(execute.call_count, 1)
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--target") + 1], "builder")


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        install_corpus_io_fixture(self, mock_stat=True)

    def test_all_isa_e2e_suites_precede_any_benchmark_and_certification(self):
        args = argparse.Namespace(output=Path("reports"), cpu_isas=list(pipeline.SHIPPING_ISAS),
                                  through="certify", image="registry/image:version", resume=False)
        def advance(variant, _source):
            return {"certified": True,
                    "images": {"runtime": {"id": "runtime-" + variant.cpu_isa.lower()}}}
        with patch.object(pipeline, "run_variant", side_effect=advance) as execute:
            result = pipeline.drive_variants(args, {})
        self.assertEqual(set(result), set(pipeline.SHIPPING_ISAS))
        order = [(call.args[0].cpu_isa, call.args[0].through) for call in execute.call_args_list]
        self.assertEqual(order, [(isa, phase.value) for phase in pipeline.pipeline_phases() for isa in pipeline.SHIPPING_ISAS])
        for call in execute.call_args_list:
            variant = call.args[0]
            self.assertEqual(variant.output, args.output / variant.cpu_isa.lower())
            self.assertEqual(variant.resume, variant.through != "build")
            self.assertEqual(variant.image, "registry/image:version" + ("-avx2" if variant.cpu_isa == "AVX2" else ""))
            if variant.through == "cross-host-e2e":
                self.assertEqual(variant.resolved_remote_cpu_image, "runtime-avx2")

    def test_failing_second_e2e_suite_blocks_both_benchmarks(self):
        args = argparse.Namespace(output=Path("reports"), cpu_isas=list(pipeline.SHIPPING_ISAS),
                                  through="certify", image=None, resume=False)
        def advance(variant, source):
            if variant.cpu_isa == "AVX2" and variant.through == "e2e":
                raise ValueError("AVX2 E2E red")
            return {"certified": False}
        with patch.object(pipeline, "run_variant", side_effect=advance) as execute:
            with self.assertRaisesRegex(ValueError, "AVX2 E2E red"):
                pipeline.drive_variants(args, {})
        self.assertFalse(any(call.args[0].through in ("benchmarks", "certify") for call in execute.call_args_list))

    def test_shipping_requires_both_isas_and_combines_ratchets_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = {"revision": "revision", "tree": "tree", "dirty": False}
            baseline = {"schema": 1, "entries": {}, "regression_threshold_pct": 10}
            finals = certified_variants(path, source, baseline)
            payloads = pipeline.publication_payloads(source, path, finals, baseline)
            marks = payloads["benchmarks/production/high_water.json"]["entries"]
            self.assertEqual(len(payloads), 3)
            self.assertEqual({entry["identity"]["cpu_isa"] for entry in marks.values()}, set(pipeline.SHIPPING_ISAS))
            with self.assertRaisesRegex(ValueError, "both AVX512 and AVX2"):
                pipeline.publication_payloads(source, path, {"AVX512": finals["AVX512"]}, baseline)
            with self.assertRaises(ValueError):
                pipeline.publication_payloads(source, path, {"AVX512": finals["AVX2"], "AVX2": finals["AVX512"]}, baseline)

    def test_official_single_isa_publication_is_rejected_before_build(self):
        with patch.object(pipeline, "source_identity") as source:
            with self.assertRaises(SystemExit) as error:
                pipeline.main(["--output", "unused", "--cpu-isa", "AVX2", "--publish", "--image", "registry/image:tag"])
            self.assertEqual(error.exception.code, 2)
            source.assert_not_called()

    def test_generated_reference_metadata_cannot_change_source_identity(self):
        """New model families must not turn additive HF evidence into source edits."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"], cwd=path, check=True)
            with patch.object(pipeline, "ROOT", path):
                before = pipeline.source_identity()
                for name in ("pytorch_ornith15_moe_LocalTP_RCCL_2xROCm_ExpertOverlay_snapshots",
                             "pytorch_future_model_snapshots"):
                    pack = path / name
                    pack.mkdir()
                    (pack / "mtp_sidecar_snapshot_schema.txt").write_text("5\n")
                    (pack / "mtp_sidecar_branch_overrides.json").write_text('{"branch": [1, 2]}\n')
                    (pack / "metadata.txt").write_text("reference_engine: pytorch\n")
                self.assertEqual(pipeline.source_identity(), before)
                (path / "src").mkdir()
                (path / "src/new_implementation.cpp").write_text("// new source\n")
                self.assertNotEqual(pipeline.source_identity()["tree"], before["tree"])

    def test_builder_installs_the_source_identity_policy(self):
        """The functional Git-identity fixture must also have its data in Docker.

        This packaging check catches the omitted COPY during the fast native
        gate; the preceding functional regression executes again in the real
        installed builder and proves that the input actually arrived there.
        """
        dockerfile = (ROOT / "Dockerfile").read_text()
        builder = dockerfile.split("FROM toolchain AS builder", 1)[1].split("FROM ubuntu:24.04 AS runtime", 1)[0]
        inputs = [part for line in builder.splitlines() if line.startswith("COPY ")
                  for part in shlex.split(line)[1:-1]]
        for policy in ("Dockerfile", ".dockerignore", ".gitignore"):
            self.assertIn(policy, inputs)

    def test_default_cli_certifies_both_and_keeps_local_publication_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            def advance(variant, _source):
                return {"certified": True,
                        "images": {"runtime": {"id": "runtime-" + variant.cpu_isa.lower()}}}
            with patch.object(pipeline, "source_identity", return_value={"revision": "revision", "tree": "tree"}), \
                 patch.object(pipeline, "device_lease"), \
                 patch.object(pipeline.docker_paths, "publish_model_cache"), \
                 patch.object(pipeline, "run_variant", side_effect=advance) as variants, \
                 patch.object(pipeline, "publish") as publish:
                self.assertEqual(pipeline.main(["--output", str(output)]), 0)
            self.assertEqual(variants.call_count, 2 * len(pipeline.pipeline_phases()))
            publish.assert_not_called()
            state = json.loads((output / "pipeline.json").read_text())
            self.assertTrue(state["certified"])
            self.assertTrue(state["shipping_set_complete"])
            self.assertEqual(set(state["variants"]), set(pipeline.SHIPPING_ISAS))

    def test_shared_core_defers_cuda_driver_binding(self):
        # CPU-only cluster members use the same full-backend shared core.
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        public_blocks = re.findall(r"target_link_libraries\(llaminar2_core\s+PUBLIC\s+([^)]*)\)", cmake)
        cuda_contract = [block for block in public_blocks if "CUDA::cudart" in block]
        self.assertEqual(len(cuda_contract), 1)
        self.assertNotIn("CUDA::cuda_driver", cmake)
        self.assertIn("backends/cuda/CUDADriverApi.cpp", cmake)

    def test_driver_stub_is_not_needed_for_build_or_runtime(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        directory = "/tmp/llaminar-build-discovery-driver"
        self.assertNotIn(directory, dockerfile)
        self.assertNotIn("stubs/libcuda", dockerfile)
        runtime = dockerfile.split("FROM ubuntu:24.04 AS runtime", 1)[1]
        self.assertNotIn(directory, runtime)
        self.assertNotIn("stubs", runtime)

    def test_installed_tests_keep_host_cache_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(models=root, model_ramdisk_root=root, reference_cache_root=root)
            with patch.object(pipeline.os, "getuid", return_value=1003), \
                 patch.object(pipeline.os, "getgid", return_value=1004), \
                 patch.object(pipeline.os, "getgroups", return_value=[1004, 109]), \
                 patch.object(pipeline.docker_paths, "device_args", return_value=["--user", "1003:1004"]) as devices, \
                 patch.object(pipeline.docker_paths, "mounts", return_value=[]), \
                 patch.object(pipeline, "run") as execute, patch.object(pipeline.subprocess, "run"):
                pipeline.run_builder({"builder": {"id": "image"}}, ["python3", "gate.py"], args, root, "gate.log")
            devices.assert_called_once_with("image", "CPU+CUDA+ROCm", user="1003:1004")
            argv = execute.call_args.args[0]
            self.assertEqual([argv[i + 1] for i, part in enumerate(argv) if part == "--group-add"], ["1004", "109"])

    def test_runtime_and_direct_docker_builds_share_complete_collective_selection(self):
        pattern = (ROOT / "scripts/docker/rccl-functions.txt").read_text().strip()
        selections = {entry.split()[0]: entry.split() for entry in pattern.split("|")}
        for operation in ("AllReduce", "Reduce", "ReduceScatter"):
            self.assertEqual(set(selections[operation][3].split("/")), {"Sum", "MinMax"})
            self.assertEqual(set(selections[operation][4].split("/")), {"i8", "i32", "f16", "f32", "bf16"})
        wrapper = ROOT / "scripts/docker/build-runtime-image.sh"
        output = subprocess.check_output(["bash", str(wrapper), "--variant", "full", "--dry-run"], text=True)
        self.assertIn("MinMax", output)
        full = subprocess.check_output(["bash", str(wrapper), "--variant", "full", "--full-rccl-funcs", "--dry-run"], text=True)
        self.assertIn("RCCL_ONLY_FUNCS=", full)
        self.assertNotIn("MinMax", full)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("COPY scripts/docker/rccl-functions.txt", dockerfile)
        self.assertIn("ARG RCCL_ONLY_FUNCS=default", dockerfile)

    def test_rccl_dependency_is_selected_once_for_builder_and_runtime(self):
        # Runtime search lists previously ignored the configured source-built
        # library and loaded broken packaged gfx906 code in relocated images.
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        self.assertIn('LLAMINAR_RCCL_LIBRARY_PATH=\\"${RCCL_LIBRARY}\\"', cmake)
        loader = (ROOT / "src/v2/collective/backends/RCCLDynamicLoader.cpp").read_text()
        self.assertIn("path = LLAMINAR_RCCL_LIBRARY_PATH;", loader)
        self.assertNotIn("lib_paths", loader)
        self.assertNotIn('"/workspaces/', loader)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("install -m 0755 /src/external/rccl/build/librccl.so.1.0 "
                      "/usr/local/lib/librccl.so.1.0", dockerfile)
        self.assertEqual(dockerfile.count("-DRCCL_LIBRARY=/usr/local/lib/librccl.so.1"), 2)

    def test_build_planning_never_probes_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "docker"
            path.write_text("#!/bin/sh\nexit 99\n")
            path.chmod(0o755)
            result = subprocess.run(["bash", str(ROOT / "scripts/docker/build-runtime-image.sh"), "--dry-run"],
                env={**pipeline.os.environ, "PATH": directory + ":" + pipeline.os.environ["PATH"]},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("docker buildx build", result.stdout)

    def test_official_result_commit_is_scoped_and_parented_to_tested_source(self):
        # A real private Git remote exercises the publication transaction. Docker
        # is mocked: this test is never evidence of an actual image certificate.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            remote = Path(directory) / "remote.git"
            def git(*args, input=None):
                return subprocess.check_output(["git", *args], cwd=root, text=True,
                                               input=input, stderr=subprocess.DEVNULL).strip()
            git("init", "-b", "develop")
            git("init", "--bare", str(remote))
            git("config", "user.name", "test")
            git("config", "user.email", "test@example.invalid")
            base = {"schema": 1, "entries": {}, "regression_threshold_pct": 10}
            artifacts.write_json(root / "benchmarks/production/high_water.json", base)
            (root / "keep.txt").write_text("unchanged\n")
            git("add", ".")
            git("commit", "-m", "source")
            revision = git("rev-parse", "HEAD")
            git("remote", "add", "origin", str(remote))
            git("push", "origin", "develop")
            result = root / "reports"
            result.mkdir()
            source = {"revision": revision, "tree": git("rev-parse", "HEAD^{tree}"), "dirty": False}
            finals = certified_variants(result, source, base)
            executed = []
            def command(argv, _log, **kwargs):
                executed.append(argv)
                if argv[0] == "git":
                    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, **kwargs)
            env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push",
                   "GITHUB_REF": "refs/heads/develop", "GITHUB_SHA": revision}
            with patch.object(pipeline, "ROOT", root), patch.dict(pipeline.os.environ, env), \
                 patch.object(pipeline, "run", side_effect=command), \
                 patch.object(pipeline, "image_identity", side_effect=lambda tag: next(
                     final for final in finals.values() if final["tag"] == tag)):
                pipeline.publish(source, result, finals)
            publication = json.loads((result / "publication.json").read_text())
            commit = publication["result_commit"]
            self.assertEqual(git("rev-parse", "HEAD"), revision)
            self.assertEqual(git("rev-parse", commit + "^"), revision)
            self.assertEqual(git("ls-remote", "origin", "refs/heads/develop").split()[0], commit)
            changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
            self.assertTrue(changed)
            self.assertEqual(len(changed), 3)
            self.assertTrue(all(name.startswith("benchmarks/production/") for name in changed))
            self.assertEqual(executed[-3:-1], [["docker", "push", finals[isa]["tag"]] for isa in pipeline.SHIPPING_ISAS])
            self.assertEqual(executed[-1][:3], ["git", "push", "origin"])

    def test_registry_failure_never_advances_git_or_high_water(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {"schema": 1, "entries": {}, "regression_threshold_pct": 10}
            artifacts.write_json(root / "benchmarks/production/high_water.json", base)
            source = {"revision": "revision", "tree": "tree", "dirty": False}
            finals = certified_variants(root, source, base)
            env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push",
                   "GITHUB_REF": "refs/heads/develop", "GITHUB_SHA": "revision"}
            with patch.object(pipeline, "ROOT", root), patch.dict(pipeline.os.environ, env), \
                 patch.object(pipeline.subprocess, "check_output", return_value="revision\trefs/heads/develop\n"), \
                 patch.object(pipeline.subprocess, "run"), \
                 patch.object(pipeline, "image_identity", side_effect=lambda tag: next(
                     final for final in finals.values() if final["tag"] == tag)), \
                 patch.object(pipeline, "run", side_effect=[None, subprocess.CalledProcessError(1, "docker")]) as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.publish(source, root, finals)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(execute.call_args.args[0], ["docker", "push", finals["AVX2"]["tag"]])
            self.assertFalse((root / "publication.json").exists())
            self.assertEqual(json.loads((root / "benchmarks/production/high_water.json").read_text()), base)

    def test_resume_cannot_skip_a_phase(self):
        names = [phase.value for phase in pipeline.pipeline_phases()]
        for length in range(len(names) + 1):
            pipeline.validate_phase_prefix(dict.fromkeys(names[:length]))
        for impossible in ({"certify": {}}, {"build": {}, "e2e": {}}, {"unknown": {}}):
            with self.assertRaises(ValueError):
                pipeline.validate_phase_prefix(impossible)

    def test_prebuilt_receipt_rejects_changed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "installed.json"
            artifacts.write_json(path, {"schema": 1, "inventory": {"tests": ["one"]}})
            with patch.object(prebuilt_test_image, "inventory", return_value={"tests": ["two"]}):
                with self.assertRaises(ValueError):
                    prebuilt_test_image.validate(path, Path(directory))

    def test_seal_rejects_invalid_gate_registration_before_writing_a_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            with patch.object(prebuilt_test_image, "discover_production_parity_unit_tests", return_value=("V2_Unit_A",)), \
                 patch.object(prebuilt_test_image, "discover_production_parity_preflight_tests", side_effect=RuntimeError("invalid timeout")), \
                 patch.object(prebuilt_test_image, "inventory") as installed:
                with self.assertRaisesRegex(RuntimeError, "invalid timeout"):
                    prebuilt_test_image.seal(receipt, Path(directory))
                installed.assert_not_called()
            self.assertFalse(receipt.exists())

    def test_container_process_evidence_preserves_failure_and_literal_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            command = [sys.executable, "-c", "import sys; print(sys.argv[1]); print('stderr', file=sys.stderr); sys.exit(7)",
                       "literal $HOME; $(false)"]
            completed = subprocess.run(pipeline.logged_container_command(command, str(log)),
                                       text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 7)
            self.assertIn("literal $HOME; $(false)", log.read_text())
            self.assertIn("stderr", log.read_text())
            self.assertEqual(completed.stdout, log.read_text())

    def test_installed_seal_rejects_a_registered_but_unbuilt_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            registration = {"tests": [{"name": "V2_Unit_HostOnly", "command": [str(path / "v2_missing")]}]}
            with patch.object(prebuilt_test_image.subprocess, "check_output", return_value=json.dumps(registration)):
                with self.assertRaisesRegex(ValueError, "installed test executable is missing"):
                    prebuilt_test_image.inventory(path)

    def test_installed_seal_rejects_missing_ctest_command_for_each_gate(self):
        for name, labels in (
            ("V2_Unit_HostOnly", []),
            ("V2_Integration_Device", ["ProductionParityPreflight"]),
            ("V2_Integration_Parity_ProductionCampaign_CPU", ["Campaign"]),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                registration = {"tests": [{"name": name, "properties": {"LABELS": labels}}]}
                with patch.object(prebuilt_test_image.subprocess, "check_output", return_value=json.dumps(registration)):
                    with self.assertRaisesRegex(ValueError, "CTest omitted command"):
                        prebuilt_test_image.inventory(Path(directory))

    def test_installed_seal_allows_unbuilt_tests_outside_pipeline_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "CMakeCache.txt").write_text("configured")
            (path / "compile_commands.json").write_text("[]")
            (path / "build.ninja").write_text("configured")
            (path / "CMakeFiles").mkdir()
            (path / "CMakeFiles/rules.ninja").write_text("compiler launchers")
            registration = {"tests": [
                {"name": "V2_Integration_Unrelated", "command": [str(path / "v2_unbuilt")]},
                {"name": "V2_Perf_Unrelated"},
            ]}
            with patch.object(prebuilt_test_image.subprocess, "check_output", return_value=json.dumps(registration)):
                sealed = prebuilt_test_image.inventory(path)
            self.assertEqual(sealed["tests"], registration["tests"])
            self.assertIn("CMakeFiles/rules.ninja", sealed["files"])
            self.assertIn("build.ninja", sealed["files"])

    def test_docker_context_excludes_nested_interpreter_caches(self):
        # Importing archived helpers must not change the next COPY tests/scripts
        # layer and rebuild both sibling images. Docker needs explicit recursion.
        patterns = (ROOT / ".dockerignore").read_text().splitlines()
        for pattern in ("**/__pycache__/", "**/*.pyc", "**/*.pyo"):
            self.assertIn(pattern, patterns)

    def test_prebuilt_image_still_executes_both_complete_gates(self):
        with patch.object(parity, "discover_production_parity_unit_tests", return_value=("V2_Unit_A",)), \
             patch.object(parity, "discover_production_parity_preflight_tests", return_value=("V2_Integration_B",)), \
             patch.object(parity.os, "sched_getaffinity", return_value=set(range(8))), \
             patch.object(prebuilt_test_image, "validate") as validate, \
             patch.object(parity, "_run_process", return_value=0) as execute:
            status, _, tests = parity.run_production_parity_preflight(Path("build"), None, Path("receipt"))
        self.assertEqual(status, 0)
        self.assertEqual(len(tests), 2)
        validate.assert_called_once()
        self.assertEqual(execute.call_count, 2)
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertTrue(all(cmd[0] == "ctest" and "--no-tests=error" in cmd for cmd in commands))
        self.assertIn("^V2_Unit_", commands[0])
        self.assertIn("^ProductionParityPreflight$", commands[1])
        for command in commands:
            self.assertEqual(command[command.index("--parallel") + 1], "8")

    def test_remote_docker_is_not_a_node_local_device_transport(self):
        with patch.dict(docker_paths.os.environ, {"DOCKER_HOST": "tcp://remote:2376"}):
            with self.assertRaises(ValueError):
                docker_paths.host_path(ROOT)

    def test_attached_execution_requires_delayed_output_and_exact_exit(self):
        for outcome in ("complete", "premature", "missing-output", "timeout"):
            with self.subTest(outcome=outcome):
                calls = []

                def execute(command, **kwargs):
                    calls.append(command)
                    if command[:2] == ["docker", "exec"]:
                        if outcome == "timeout":
                            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                        return subprocess.CompletedProcess(command,
                            0 if outcome == "premature" else 23,
                            command[-1] + "\n" if outcome == "complete" else "")
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(docker_paths.subprocess, "run", side_effect=execute):
                    if outcome == "complete":
                        docker_paths.validate_attached_execution("image")
                    else:
                        expected = subprocess.TimeoutExpired if outcome == "timeout" else RuntimeError
                        with self.assertRaises(expected):
                            docker_paths.validate_attached_execution("image")
                self.assertEqual(calls[-1][:4], ["docker", "rm", "--force", "--volumes"])
                self.assertEqual(calls[-1][-1], calls[0][3])
                self.assertIn("--network", calls[0])
                self.assertEqual(calls[0][calls[0].index("--network") + 1], "host")
                self.assertEqual(sum(command[:2] == ["docker", "exec"] for command in calls), 1)

    def test_devcontainer_uses_the_direct_mounted_docker_socket(self):
        configuration = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
        self.assertEqual(configuration["containerEnv"]["DOCKER_HOST"],
                         "unix:///var/run/docker-host.sock")

    def test_driver_injection_follows_selected_backend_not_image_linkage(self):
        for backend in ("CPU", "ROCm", "CPU+ROCm", "CUDA", "CPU+CUDA+ROCm"):
            with self.subTest(backend=backend), \
                 patch.object(docker_paths, "rocm_device_group_ids", return_value=(992,)), \
                 patch.object(docker_paths.subprocess, "check_output", return_value="--gpus\nall\n") as inject:
                arguments = docker_paths.device_args("full-image", backend)
                self.assertEqual("--gpus" in arguments, "CUDA" in backend)
                if "CUDA" in backend:
                    self.assertIn("--required", inject.call_args.args[0])
                else:
                    inject.assert_not_called()
                self.assertEqual("--device=/dev/kfd" in arguments, "ROCm" in backend)
                self.assertEqual("992" in arguments, "ROCm" in backend)

    def test_rocm_groups_come_from_daemon_device_metadata(self):
        with patch.object(docker_paths, "daemon_device_metadata", return_value="992\n44\n992\n") as probe:
            self.assertEqual(docker_paths.rocm_device_group_ids("image"), (44, 992))
            self.assertEqual(probe.call_args.args[0], "image")
            self.assertIn("stat -c %g", probe.call_args.args[1])
        for invalid in ("", "render\n", "992\ninvalid\n"):
            with self.subTest(output=invalid), \
                 patch.object(docker_paths, "daemon_device_metadata", return_value=invalid):
                with self.assertRaises(ValueError):
                    docker_paths.rocm_device_group_ids("image")

    def test_device_probe_reads_completed_artifact_not_attached_stdout(self):
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 30)
            if command[1] == "cp":
                # Reproduce Docker's empty attached output while its result
                # file is complete. No stdout or log replay supplies metadata.
                Path(command[-1]).write_text("992\n44\n")
            return subprocess.CompletedProcess(command, 0, stdout="")

        def wait(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[1], "wait")
            return "0\n"

        with patch.object(docker_paths.subprocess, "run", side_effect=execute), \
             patch.object(docker_paths.subprocess, "check_output", side_effect=wait):
            self.assertEqual(docker_paths.daemon_device_metadata("image", "probe script"), "992\n44\n")
        self.assertEqual([command[1] for command in calls], ["create", "start", "wait", "cp", "rm"])
        create = calls[0]
        self.assertIn("--network", create)
        self.assertEqual(create[create.index("--network") + 1], "host")
        self.assertIn("type=bind,src=/dev,dst=/host-dev,readonly", create)
        self.assertIn("probe script", create)
        name = create[create.index("--name") + 1]
        self.assertTrue(name.startswith("llaminar-device-probe-"))
        self.assertEqual(calls[1][-1], name)
        self.assertEqual(calls[2][-1], name)
        self.assertTrue(calls[3][2].startswith(name + ":"))
        self.assertEqual(calls[4], ["docker", "rm", "--force", "--volumes", name])
        self.assertFalse(Path(calls[3][-1]).exists())

    def test_device_probe_failure_cleans_up_without_retry(self):
        for failure in ("start", "wait", "exit-status", "cp", "missing-artifact"):
            with self.subTest(failure=failure):
                calls = []

                def execute(command, **kwargs):
                    calls.append(command)
                    if command[1] == failure:
                        raise subprocess.CalledProcessError(1, command)
                    return subprocess.CompletedProcess(command, 0)

                def wait(command, **kwargs):
                    calls.append(command)
                    if failure == "wait":
                        raise subprocess.TimeoutExpired(command, 30)
                    return "7\n" if failure == "exit-status" else "0\n"

                with patch.object(docker_paths.subprocess, "run", side_effect=execute), \
                     patch.object(docker_paths.subprocess, "check_output", side_effect=wait):
                    with self.assertRaises((subprocess.SubprocessError, RuntimeError, FileNotFoundError)):
                        docker_paths.daemon_device_metadata("image", "probe script")
                self.assertEqual(calls[-1][1:4], ["rm", "--force", "--volumes"])
                self.assertEqual(sum(command[1] == "create" for command in calls), 1)
                self.assertEqual(sum(command[1] == "start" for command in calls), 1)
                if failure in ("start", "wait", "exit-status"):
                    self.assertNotIn("cp", [command[1] for command in calls])

    def test_nvidia_nodes_use_the_same_completed_metadata_contract(self):
        with patch.object(docker_paths, "daemon_device_metadata",
                          return_value="/dev/nvidia1\n/dev/nvidiactl\n/dev/nvidia1\n") as probe:
            self.assertEqual(docker_paths.nvidia_device_nodes("image"), ("/dev/nvidia1", "/dev/nvidiactl"))
            self.assertEqual(probe.call_args.args[0], "image")
            self.assertIn("/host-dev/nvidia-caps/*", probe.call_args.args[1])
        with patch.object(docker_paths, "daemon_device_metadata", return_value=""):
            self.assertEqual(docker_paths.nvidia_device_nodes("image"), ())
        for invalid in ("/dev/kfd\n", "/dev/nvidia-caps/../../mem\n", "\n"):
            with self.subTest(output=invalid), \
                 patch.object(docker_paths, "daemon_device_metadata", return_value=invalid):
                with self.assertRaises(ValueError):
                    docker_paths.nvidia_device_nodes("image")

    def test_nvidia_shell_launcher_propagates_metadata_failure(self):
        # Exported functions isolate the actual shell launcher from Docker and
        # Python. Even optional injection must not swallow a failed probe as
        # though a successful inventory had reported no NVIDIA devices.
        shell = ('docker() { return 1; }; python3() { return 9; }; '
                 'export -f docker python3; exec bash "$1" --probe-image image')
        result = subprocess.run(["bash", "-c", shell, "probe-test",
                                 str(ROOT / "scripts/ci/docker_gpu_run_args.sh")],
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("NVIDIA device metadata probe failed", result.stderr)

    def test_nvidia_runtime_probe_uses_the_inference_network_mode(self):
        """GPU admission must not depend on unrelated bridge/iptables setup."""
        # Simulate a host whose native GPU injection works but whose bridge
        # creation does not. Run the real shell helper; no source spelling
        # assertion can prove which Docker command it actually submits.
        shell = ('docker() { case "$1" in '
                 'info) printf "{}\\n" ;; '
                 'run) case " $* " in *" --network host "*) return 0 ;; '
                 '*) return 1 ;; esac ;; *) return 1 ;; esac; }; '
                 'python3() { return 9; }; export -f docker python3; '
                 'exec bash "$1" --probe-image image --required')
        result = subprocess.run(["bash", "-c", shell, "probe-test",
                                 str(ROOT / "scripts/ci/docker_gpu_run_args.sh")],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "--gpus\nall\n")

    def test_cpu_rank_requires_no_nvidia_injection(self):
        with patch.object(docker_paths.subprocess, "check_output") as inject:
            self.assertNotIn("--gpus", docker_paths.device_args("cpu-image", "CPU"))
            inject.assert_not_called()

    def test_server_harness_uses_request_intent_for_driver_injection(self):
        harness = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        self.assertNotIn('--cuda-driver-required', harness)
        self.assertIn('if docker_args_need_cuda "${args_ref[@]}"; then', harness)

    def test_develop_workflow_has_one_model_free_driver_and_no_certification_axes(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("scripts/ci/run_develop_image_gate.py", workflow)
        self.assertNotIn("scripts/ci/run_production_pipeline.py", workflow)
        self.assertNotIn("run_benchmark_check.sh", workflow)
        self.assertNotIn("run_model_parity", workflow)
        self.assertNotIn("azure", workflow.lower())
        self.assertNotIn("matrix:", workflow)
        self.assertIn("branches: [develop]", workflow)
        self.assertIn("--publish", workflow)
        self.assertIn("submodules: false", workflow)
        self.assertIn("lfs: false", workflow)


if __name__ == "__main__":
    unittest.main()
