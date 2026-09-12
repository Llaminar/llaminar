#!/usr/bin/env python3
"""Device-free adversarial tests for benchmark and image certification policy.

Mocked processes prove orchestration decisions, not inference performance.
Prefix-state obligations come from typed model declarations; exact output
tokens cannot excuse missing recurrent state or a response-selected contract.
The real local CI run remains the only source of image certificates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import docker_paths
import generation_tokens as generation
import generation_regression_http as generation_http
import run_model_parity_generation as generation_runner
import prebuilt_test_image
import production_artifacts as artifacts
import run_model_parity_benchmarks as benchmark
import run_production_pipeline as pipeline
import run_production_parity_campaigns as parity
from test_generation_movement_ledger import empty_ledger, ledger as movement_ledger


def cell(name="cell"):
    """Minimal tagged configuration with a complete split-model manifest."""
    return {"case": name, "campaign": "campaign", "backends": "CUDA+ROCm",
            "model_files": ["/src/models/model-1.gguf", "/src/models/model-2.gguf"],
            "configuration": {"model_parity_schema": 1, "id": name,
                "model": "/src/models/model-1.gguf", "e2e": {"context_length": 8192,
                    "server_args": ["--define-domain", "arbitrary;devices=rocm:0,cuda:0", "--mtp"]}}}


def manifest():
    return {"schema": 1, "source_revision": "revision", "cells": [cell()]}


def e2e_report(inventory=None):
    """A passing report explicitly bound to its entire manifest and image."""
    inventory = inventory or manifest()
    return {"schema": 1, "image": "runtime-id", "source_revision": "revision",
            "manifest_digest": artifacts.digest(inventory), "correctness_passed": True,
            "selected": len(inventory["cells"]),
            "cells": [{"case": row["case"], "return_code": 0, "outcome": "passed",
                       "configuration": copy.deepcopy(row["configuration"])} for row in inventory["cells"]]}


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
        images = {"runtime": {"id": "sha256:" + isa, "labels": {"org.llaminar.cpu_isa": isa}},
                  "builder": {"id": "builder-" + isa}}
        evidence = {**e2e_report(inventory), "source_revision": source["revision"],
                    "image": images["runtime"]["id"]}
        row = {"case": "cell", "identity": {"cpu_isa": isa},
               "tokens_per_second": {"prefill": 100, "decode": 10}}
        documents = {"manifest": inventory, "e2e": evidence,
            "parity": {"correctness_passed": True, "performance_requirements_met": True,
                       "artifact_contract_passed": True, "preflight_return_code": 0,
                       "preflight_tests": ["V2_Unit_A", "V2_Integration_B"],
                       "preflight_test_count": 2, "exact_matrix_cell_count": 1},
            "benchmarks": {"passed": True, "complete": True, "diagnostic": False,
                           "image": images["runtime"]["id"], "manifest_digest": artifacts.digest(inventory),
                           "e2e_report_digest": artifacts.digest(evidence), "cells": [row],
                           "baseline_digest": artifacts.digest(baseline)}}
        for name, document in documents.items():
            artifacts.write_json(path / f"{name}.json", document)
        final = {"tag": "registry/image:" + isa.lower(), "id": "sha256:certified-" + isa,
                 "certificate": pipeline.certificates(source, images, path)}
        artifacts.write_json(path / "pipeline.json", {"certified": True, "images": images, "final": final})
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
        for mutate in (lambda r: r.update(requests=[]), lambda r: r.update(readiness_timeout_seconds=601),
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
            self.assertEqual(execute.call_args_list[0].kwargs["timeout"], 600)
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


class CertificateTests(unittest.TestCase):
    def test_certificate_layer_cannot_hide_a_runtime_overwrite(self):
        pipeline.validate_certificate_changes("C /usr/local/share\nA /usr/local/share/llaminar/certificates/production.json\n")
        for invalid in ("C /usr/local/bin/llaminar2", "A /usr/local/lib/libllaminar2_core.so",
                        "D /usr/local/share/llaminar/certificates/production.json",
                        "A /usr/local/share/llaminar/certificates/unexpected.sh"):
            with self.subTest(change=invalid), self.assertRaises(ValueError):
                pipeline.validate_certificate_changes(invalid)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.source = {"revision": "revision", "tree": "tree", "dirty": False}
        self.images = {"runtime": {"id": "runtime-id", "labels": {"org.llaminar.cpu_isa": "AVX512"}},
                       "builder": {"id": "builder-id"}}
        self.documents = {
            "manifest": manifest(),
            "parity": {"correctness_passed": True, "performance_requirements_met": True,
                       "artifact_contract_passed": True, "preflight_return_code": 0,
                       "preflight_tests": ["V2_Unit_A", "V2_Integration_B"],
                       "preflight_test_count": 2, "exact_matrix_cell_count": 1},
            "e2e": e2e_report(),
            "benchmarks": {"passed": True, "complete": True, "image": "runtime-id",
                           "diagnostic": False, "e2e_report_digest": artifacts.digest(e2e_report()),
                           "manifest_digest": artifacts.digest(manifest()), "cells": [{"case": "cell"}]}}

    def write(self):
        for name, document in self.documents.items():
            artifacts.write_json(self.path / f"{name}.json", document)

    def test_exact_complete_evidence_can_certify(self):
        self.write()
        result = pipeline.certificates(self.source, self.images, self.path)
        self.assertTrue(result["certified"])
        self.assertEqual(result["tested_image"], "runtime-id")

    def test_every_missing_report_blocks_certification(self):
        for name in self.documents:
            self.write()
            (self.path / f"{name}.json").unlink()
            with self.subTest(name=name), self.assertRaises(OSError):
                pipeline.certificates(self.source, self.images, self.path)

    def test_regression_subset_wrong_image_or_stale_manifest_cannot_certify(self):
        original = copy.deepcopy(self.documents)
        for bad in ({"passed": False}, {"complete": False}, {"image": "other-image"},
                    {"diagnostic": True}, {"diagnostic": None}, {"e2e_report_digest": "stale"},
                    {"manifest_digest": "stale"}, {"cells": []}, {"cells": [{"case": "wrong"}]}):
            self.documents = copy.deepcopy(original)
            self.documents["benchmarks"].update(bad)
            self.write()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.certificates(self.source, self.images, self.path)

    def test_exit_zero_e2e_with_changed_configuration_is_rejected(self):
        self.documents["e2e"]["cells"][0]["configuration"]["e2e"]["server_args"] = []
        self.write()
        with self.assertRaises(ValueError):
            pipeline.certificates(self.source, self.images, self.path)

    def test_missing_unit_preflight_or_economy_never_certifies(self):
        original = copy.deepcopy(self.documents)
        for bad in ({"preflight_tests": []}, {"preflight_test_count": 0},
                    {"preflight_return_code": 1}, {"performance_requirements_met": False}):
            self.documents = copy.deepcopy(original)
            self.documents["parity"].update(bad)
            self.write()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.certificates(self.source, self.images, self.path)

    def test_local_and_dirty_runs_cannot_publish_or_commit(self):
        with patch.dict(pipeline.os.environ, {}, clear=True), patch.object(pipeline.subprocess, "check_output") as execute:
            with self.assertRaises(ValueError):
                pipeline.publish(self.source, self.path, {})
            execute.assert_not_called()

    def test_image_requires_release_both_backends_correct_tree_and_isa(self):
        labels = {"org.opencontainers.image.revision": "revision", "org.llaminar.source_tree": "tree",
                  "org.llaminar.build_type": "Release", "org.llaminar.cpu_isa": "AVX512",
                  "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON"}
        pipeline.require_full_runtime({"labels": labels}, self.source, "AVX512")
        for field in labels:
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipeline.require_full_runtime({"labels": {**labels, field: "wrong"}}, self.source, "AVX512")


class InfrastructureTests(unittest.TestCase):
    def test_all_isa_e2e_suites_precede_any_benchmark_and_certification(self):
        args = argparse.Namespace(output=Path("reports"), cpu_isas=list(pipeline.SHIPPING_ISAS),
                                  through="certify", image="registry/image:version", resume=False)
        with patch.object(pipeline, "run_variant", return_value={"certified": True}) as execute:
            result = pipeline.drive_variants(args, {})
        self.assertEqual(set(result), set(pipeline.SHIPPING_ISAS))
        order = [(call.args[0].cpu_isa, call.args[0].through) for call in execute.call_args_list]
        self.assertEqual(order, [(isa, phase.value) for phase in pipeline.Phase for isa in pipeline.SHIPPING_ISAS])
        for call in execute.call_args_list:
            variant = call.args[0]
            self.assertEqual(variant.output, args.output / variant.cpu_isa.lower())
            self.assertEqual(variant.resume, variant.through != "build")
            self.assertEqual(variant.image, "registry/image:version" + ("-avx2" if variant.cpu_isa == "AVX2" else ""))

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
            source = {"revision": "revision", "dirty": False}
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

    def test_default_cli_certifies_both_and_keeps_local_publication_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with patch.object(pipeline, "source_identity", return_value={"revision": "revision", "tree": "tree"}), \
                 patch.object(pipeline, "device_lease"), \
                 patch.object(pipeline.docker_paths, "publish_model_cache"), \
                 patch.object(pipeline, "run_variant", return_value={"certified": True}) as variants, \
                 patch.object(pipeline, "publish") as publish:
                self.assertEqual(pipeline.main(["--output", str(output)]), 0)
            self.assertEqual(variants.call_count, 2 * len(pipeline.Phase))
            publish.assert_not_called()
            state = json.loads((output / "pipeline.json").read_text())
            self.assertTrue(state["certified"])
            self.assertTrue(state["shipping_set_complete"])
            self.assertEqual(set(state["variants"]), set(pipeline.SHIPPING_ISAS))

    def test_shared_core_exports_the_cuda_driver_link_contract(self):
        # Source-policy guard for the driver-free Docker build. cuda_backend's
        # private whole-archive inclusion does not propagate this SDK dependency.
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        public_blocks = re.findall(r"target_link_libraries\(llaminar2_core\s+PUBLIC\s+([^)]*)\)", cmake)
        cuda_contract = [block for block in public_blocks if "CUDA::cudart" in block]
        self.assertEqual(len(cuda_contract), 1)
        self.assertIn("CUDA::cuda_driver", cuda_contract[0])

    def test_driver_stub_is_confined_to_build_time_test_name_discovery(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        directory = "/tmp/llaminar-build-discovery-driver"
        self.assertIn(f'LD_LIBRARY_PATH="{directory}:${{LD_LIBRARY_PATH}}"', dockerfile)
        self.assertIn(f"rm {directory}/libcuda.so.1", dockerfile)
        self.assertIn(f"rmdir {directory}", dockerfile)
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
            source = {"revision": revision, "dirty": False}
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
            source = {"revision": "revision", "dirty": False}
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
        names = [phase.value for phase in pipeline.Phase]
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

    def test_driver_injection_follows_image_linkage_not_only_requested_backend(self):
        for backend in ("CPU", "ROCm", "CPU+ROCm", "CUDA"):
            with self.subTest(backend=backend), \
                 patch.object(docker_paths, "image_identity", return_value={"labels": {"org.llaminar.cuda": "ON"}}), \
                 patch.object(docker_paths, "rocm_device_group_ids", return_value=(992,)), \
                 patch.object(docker_paths.subprocess, "check_output", return_value="--gpus\nall\n") as inject:
                arguments = docker_paths.device_args("full-image", backend)
                self.assertIn("--gpus", arguments)
                self.assertIn("--required", inject.call_args.args[0])
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

    def test_backend_subset_image_does_not_require_an_unused_cuda_driver(self):
        with patch.object(docker_paths, "image_identity", return_value={"labels": {"org.llaminar.cuda": "OFF"}}), \
             patch.object(docker_paths.subprocess, "check_output") as inject:
            self.assertNotIn("--gpus", docker_paths.device_args("cpu-image", "CPU"))
            inject.assert_not_called()

    def test_server_harness_uses_the_same_image_driver_contract_as_benchmarks(self):
        harness = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        self.assertIn('--cuda-driver-required "$CONTAINER_IMAGE"', harness)

    def test_workflow_has_one_driver_no_parallel_model_matrix_or_old_ratchet(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("scripts/ci/run_production_pipeline.py", workflow)
        self.assertNotIn("run_benchmark_check.sh", workflow)
        self.assertNotIn("matrix:", workflow)
        self.assertIn("submodules: false", workflow)
        self.assertIn("lfs: false", workflow)


if __name__ == "__main__":
    unittest.main()
