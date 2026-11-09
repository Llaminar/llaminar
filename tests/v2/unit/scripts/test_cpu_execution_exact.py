#!/usr/bin/env python3
"""Device-free replay of CPU exact identity, bundle admission and retention.

These tests supply metadata only. They prove wrong producers, underfilled or
foreign bundles, incomplete candidate/holdout evidence and stale retained bases
cannot acquire exact runtime precedence. No native process or model is loaded.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/performance/kernels"))
from native_quantised_dispatch.cpu_exact_overlay import (
    CPUExecutionExactKey, RetainedCPUProgram, authenticate_retained_programs,
    file_digest, generate_execution_exact_include, merge_exact_entries,
    select_execution_exact_entries, validate_filled_bundle,
)
from native_quantised_dispatch.adapters.cpu_execution_exact import adapt_cpu_execution_exact_row, adapt_cpu_execution_exact_native
from native_quantised_dispatch.adapters.cpu_verifier import CPUVerifierAdapterContext
import test_cpu_native_quantised_verifier_adapter as strong_fixture


def key(**changes):
    """A complete actual-kind expert contract; dimensions are test metadata."""
    original = CPUExecutionExactKey(2, 1, 5, 1, 0, 1, 1, 1, 28, 64,
                                    320, 320, 2560, 16, 1, 1, 1, 1)
    return replace(original, **changes)


def row(choice="Pairwise", *, contract=None, members=None, holdout=False):
    """Extend the existing strong-adapter fixture rather than invent an oracle."""
    contract = contract or key()
    members = members or [contract] * 2
    result = strong_fixture.CPUNativeQuantisedVerifierAdapterTest.row(choice)
    result.update(source_format="Q4_K", source_codebook="5", execution_codebook="5",
                  m=str(contract.rows), n=str(contract.physical_n), k=str(contract.k),
                  build_isa="AVX512", runtime_isa_requested="AVX512", runtime_isa_effective="AVX512",
                  threads=str(contract.workers), k_tiles=str(contract.k_tiles),
                  n_block_chunks=str(contract.ambient_n_block_chunks),
                  observed_candidate_id=choice, correctness_pass="1",
                  exact_key=json.dumps(contract.scalars()),
                  descriptor_keys=json.dumps([member.scalars() for member in members]),
                  exact_key_verified="1", numerical_policy="GPUAlignedExpert", purpose="prefill",
                  partial_route="OutputTileLocal", team_lifetime="retained",
                  output_layout="aligned_expert_major_v1", holdout=str(int(holdout)),
                  descriptor_policies=json.dumps([int(choice == "WideRows") if member == contract else 0 for member in members]),
                  baseline_descriptor_policies=json.dumps([0] * len(members)))
    return result


def observe(raw):
    """Use the common strong adapter's intentionally non-installable smoke context."""
    result = adapt_cpu_execution_exact_row(raw, CPUVerifierAdapterContext.workflow_smoke(
        corpus_id="sha256:" + "2" * 64))
    # Authored common-witness metadata for clean batch admission controls. This
    # is not a measured native producer or a claim about legacy CSV clocks.
    return replace(result, physical_launch_evidence={**result.physical_launch_evidence,
        "native_timing_owner": "CPUExecutionExactBatch",
        "native_timing_schema": "cpu_execution_exact_native_v1",
        "timed_collection_disabled": True, "timed_projection_observations": 0})


class CPUExecutionExactTest(unittest.TestCase):
    """Exercise actual discriminators and adversarial publication boundaries."""

    def test_source_arithmetic_purpose_m_and_kpart_are_distinct(self):
        original = key()
        for changes in ({"source_superblock": 0}, {"prepared_encoding": 1},
                        {"operation": 1}, {"rows": 256}, {"policy_n": 640},
                        {"k_tiles": 8}, {"team_entry": 0}, {"workers": 14}):
            with self.subTest(changes=changes):
                other = replace(original, **changes)
                self.assertNotEqual(original.scalars(), other.scalars())
                self.assertNotEqual(original.signature(), other.signature())
        self.assertEqual(CPUExecutionExactKey.from_scalars(original.scalars()), original)

    def test_no_truncated_or_backend_native_key(self):
        for changes in ({"rows": -1}, {"rows": 0}, {"rows": 2**32},
                        {"numerical_policy": 0}, {"k_tiles": 1},
                        {"layout": 0}, {"bundle_class": 0},
                        {"physical_n": 321}, {"build_isa": 0, "runtime_isa": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(key(), **changes)
        with self.assertRaises(ValueError):
            CPUExecutionExactKey.from_scalars(key().scalars()[:-1])

    def test_filled_class_uses_all_wide_owners_and_every_member(self):
        small = key(rows=3)
        with self.assertRaisesRegex(ValueError, "lower bound"):
            validate_filled_bundle(small, [small] * 5)  # 25 owners for28 workers
        self.assertEqual(len(validate_filled_bundle(small, [small] * 6)), 6)
        for foreign in (key(team_entry=0), key(operation=1), key(k=1280)):
            with self.assertRaisesRegex(ValueError, "changed execution class"):
                validate_filled_bundle(key(), [key(), foreign])

    def test_strong_adapter_keeps_whole_bundle_and_rejects_wrong_source(self):
        result = observe(row())
        self.assertFalse(result.generic_eligible)
        self.assertEqual(result.physical_launch_evidence["timing_basis"], "whole_fused_bundle")
        self.assertEqual(len(result.physical_launch_evidence["descriptors"]), 2)
        self.assertEqual(result.median_us, 11)  # no division by descriptors/leaves
        for changes in ({"purpose": "decode"}, {"numerical_policy": "BackendNative"},
                        {"exact_key_verified": "0"}, {"source_format": "Q4_1"},
                        {"threads": "14"}, {"partial_route": "SharedBank"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observe({**row(), **changes})

    def test_mixed_holdout_required_and_excluded_from_winner(self):
        training = [observe(row()), observe(row("WideRows"))]
        with self.assertRaisesRegex(ValueError, "holdout"):
            select_execution_exact_entries(training)
        held_key = key()
        mixed = [held_key, key(rows=1)]
        holdout = [observe(row(candidate, contract=held_key, members=mixed, holdout=True))
                   for candidate in ("Pairwise", "WideRows")]
        entries, receipt = select_execution_exact_entries(training + holdout)
        self.assertEqual(set(entries), {key()})
        self.assertFalse(receipt["new_generic_certification"])
        self.assertFalse(receipt["entries"][0]["latency_divided_per_leaf"])
        self.assertEqual(len(receipt["entries"][0]["mixed_holdout_digests"]), 2)
        foreign = row("WideRows", contract=held_key, members=mixed, holdout=True)
        foreign["descriptor_policies"] = json.dumps([1, 1])
        with self.assertRaisesRegex(ValueError, "unrelated descriptor"):
            observe(foreign)
        with self.assertRaisesRegex(ValueError, "distinct training"):
            select_execution_exact_entries(training + [observe(row(members=[key()] * 3))] + holdout)

    def test_scope_preserves_unrelated_entries_and_rejects_partial_refresh(self):
        old = {(1,): 0, (2,): 1}
        self.assertEqual(merge_exact_entries(old, {(1,): 1}, [(1,)]), {(1,): 1, (2,): 1})
        for scope in ([], [(1,), (1,)], [(1,), (3,)]):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                merge_exact_entries(old, {(1,): 1}, scope)

    def test_emitter_orders_full_keys_and_has_no_generic_decision(self):
        text = generate_execution_exact_include({key(rows=512): 1, key(): 0})
        self.assertIn("std::array<std::uint32_t, 18>", text)
        self.assertLess(text.index("64u, 320u"), text.index("512u, 320u"))
        self.assertIn("return false;", text)
        self.assertNotIn("selectVerifierRowsPolicy", text)
        with self.assertRaises(ValueError):
            generate_execution_exact_include({key(): 2})
        with self.assertRaises(ValueError):
            generate_execution_exact_include({key(rows=1): 0})

    def test_retained_bases_require_both_authentic_programs(self):
        with tempfile.TemporaryDirectory() as directory:
            programs = []
            for kind, selector, abi in (("decode", "selectCPUNativeQuantisedDecodeGeneratedPolicy", "DECODE"),
                                        ("verifier", "selectCPUNativeQuantisedVerifierRowsGeneratedPolicy", "VERIFIER")):
                path = Path(directory) / (kind + ".inc")
                path.write_text(f"#define LLAMINAR_CPU_NATIVE_QUANTISED_{abi}_POLICY_ABI 1\n"
                                "// Frozen generic policy digest: sha256:original\n"
                                f"bool {selector}();\n")
                programs.append(RetainedCPUProgram(kind, path, file_digest(path)))
            receipt = authenticate_retained_programs(programs)
            self.assertTrue(all(not item["new_generic_certification"] for item in receipt))
            with self.assertRaises(ValueError):
                authenticate_retained_programs(programs[:1])
            programs[0].include.write_text("mutated after baseline freeze")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                authenticate_retained_programs(programs)


class CPUExecutionNativeEvidenceTest(unittest.TestCase):
    """Native owner JSON remains the sole sample/byte/placement authority."""

    @staticmethod
    def payload(m=64):
        from native_quantised_dispatch.adapters.evidence import summarize_sorted_timing
        contract = key(rows=m); source = row(contract=contract)
        summary = summarize_sorted_timing((10.0, 11.0, 13.0))
        return dict(schema='cpu_execution_exact_native_v1', request_id='synthetic-batch-request', exact_key=contract.scalars(),
            descriptor_keys=[contract.scalars()] * 2, descriptor_policies=[0, 0],
            baseline_descriptor_policies=[0, 0], target_descriptor=0, candidate='Pairwise',
            exact_key_verified=True, source_format='Q4_K', source_superblock=True,
            prepared_codebook=5, prepared_encoding=0, numerical_policy='GPUAlignedExpert',
            purpose='prefill', partial_route='OutputTileLocal', bundle_class=1,
            team_lifetime='retained', output_layout='aligned_expert_major_v1', output_role='projection',
            expert_rows=[m, m], member_rows=[m, m], weight_bytes=1000000, shape='ProtocolFixture',
            raw_samples_us=[13.0, 10.0, 11.0], sorted_samples_us=[10.0, 11.0, 13.0],
            timing=dict(sample_count=3, repeat_count=2, warmup_count=1,
                        min_us=summary.minimum, median_us=summary.median, p95_us=summary.p95,
                        mad_us=summary.mad, cv=summary.cv, sorted_native_digest=summary.digest),
            correctness=dict(mismatch_count=0, nonfinite_count=0, actual_digest=source['grouped_output_digest'],
                expected_digest=source['serial_output_digest'], max_abs=0, relative_l2=0, cosine=1,
                symmetric_kld=0, guard_replays=2, bitwise_equal=True),
            counters=dict(enabled=True, projection_observations=4, expected_projection_observations=4,
                timed_collection_disabled=True, timed_projection_observations=0,
                sample_observation_basis='untimed_guard_calls_only',
                authentication_calls=1, authentication_outside_timing=True,
                authenticated_projection_observations=2, route_records=[dict(count=2)]),
            placement=dict(workers=28, admitted_physical_cores=28, disjoint_single_core_masks=True))

    def adapt(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.json'; path.write_text(json.dumps(payload))
            return adapt_cpu_execution_exact_native(path, CPUVerifierAdapterContext.workflow_smoke(
                corpus_id='sha256:' + '2' * 64), holdout=False)

    def test_batch_timing_requires_true_collection_disabled_and_exact_zero(self):
        from copy import deepcopy
        original = self.payload()
        for name, value in (("timed_collection_disabled", None),
                ("timed_collection_disabled", False), ("timed_collection_disabled", 1),
                ("timed_projection_observations", None), ("timed_projection_observations", 1),
                ("timed_projection_observations", False), ("timed_projection_observations", "0")):
            bad = deepcopy(original)
            if value is None: bad['counters'].pop(name)
            else: bad['counters'][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'disabled timed collection'):
                self.adapt(bad)
        result = self.adapt(original)
        self.assertIs(result.physical_launch_evidence.get('timed_collection_disabled'), True)
        self.assertEqual(result.physical_launch_evidence['timed_projection_observations'], 0)
        for changes in ({'projection_observations': 0, 'expected_projection_observations': 0},
                        {'sample_observation_basis': 'screen_and_final_timed_calls_only'}):
            bad = deepcopy(original); bad['counters'].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'untimed guards'):
                self.adapt(bad)

    def test_publication_rechecks_batch_raw_markers_and_rejects_nonbatch_diagnostic(self):
        from copy import deepcopy
        from native_quantised_dispatch.cpu_exact_overlay import require_execution_exact_publication_timing, canonical_digest
        context = CPUVerifierAdapterContext.workflow_smoke(corpus_id='sha256:' + '2' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'original.json'; payload = self.payload()
            path.write_text(json.dumps(payload))
            result = adapt_cpu_execution_exact_native(path, context, holdout=False)
            require_execution_exact_publication_timing(result)
            # A reserialized common row cannot lose the additive marker.
            for field in ('timed_collection_disabled', 'timed_projection_observations'):
                witness = dict(result.physical_launch_evidence); witness.pop(field)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'disabled timed collection'):
                    require_execution_exact_publication_timing(replace(result, physical_launch_evidence=witness))
            # Historical common rows lack the new owner tag. Original typed raw
            # batch provenance still rejects dirty clocks; no path-name heuristic.
            bad = deepcopy(payload); bad['counters'].pop('timed_collection_disabled')
            path.write_text(json.dumps(bad)); witness = dict(result.physical_launch_evidence)
            for field in ('native_timing_owner', 'timed_collection_disabled', 'timed_projection_observations'):
                witness.pop(field)
            witness['native_evidence_mapping_digest'] = canonical_digest(bad)
            with self.assertRaisesRegex(ValueError, 'disabled timed collection'):
                require_execution_exact_publication_timing(replace(result, physical_launch_evidence=witness))
            # The separate standalone diagnostic remains parseable/profileable,
            # but its declared instrumented clocks cannot publish NEW exact rows.
            standalone = deepcopy(payload); standalone.pop('request_id')
            standalone['counters'].pop('timed_collection_disabled')
            standalone['counters'].pop('timed_projection_observations')
            standalone['counters'].pop('sample_observation_basis')
            path.write_text(json.dumps(standalone))
            diagnostic = adapt_cpu_execution_exact_native(path, context, holdout=False)
            self.assertEqual(diagnostic.median_us, 11.)
            from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation
            self.assertTrue(profiler_request_for_observation(diagnostic).request_id)
            with self.assertRaisesRegex(ValueError, 'nonbatch.*diagnostic'):
                select_execution_exact_entries([diagnostic])

    def test_winner_publication_requires_clean_timing_on_every_member(self):
        contract = key(); mixed = [contract, key(rows=1)]
        clean = [observe(row(candidate)) for candidate in ('Pairwise', 'WideRows')]
        clean += [observe(row(candidate, members=mixed, holdout=True)) for candidate in ('Pairwise', 'WideRows')]
        self.assertTrue(select_execution_exact_entries(clean)[0])
        for index in range(len(clean)):
            witness = dict(clean[index].physical_launch_evidence)
            witness.pop('timed_collection_disabled')
            bad = list(clean); bad[index] = replace(bad[index], physical_launch_evidence=witness)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'disabled timed collection'):
                select_execution_exact_entries(bad)
        raw_diagnostic = adapt_cpu_execution_exact_row(row(), CPUVerifierAdapterContext.workflow_smoke(
            corpus_id='sha256:' + '2' * 64))
        with self.assertRaisesRegex(ValueError, 'known uninstrumented'):
            select_execution_exact_entries([raw_diagnostic])

    def test_actual_samples_remain_whole_bundle_and_round_trip_profile_identity(self):
        from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation, ProfilerRequest
        from native_quantised_dispatch.profiler_collectors import _cpu_execution_exact_profile_fields
        observation = self.adapt(self.payload())
        self.assertEqual(observation.median_us, 11.0)
        request = profiler_request_for_observation(observation)
        restored = ProfilerRequest.from_mapping(request.canonical_mapping())
        self.assertEqual(restored.execution_witness, request.execution_witness)
        with tempfile.TemporaryDirectory() as directory:
            fields = _cpu_execution_exact_profile_fields(request, Path(directory))
            self.assertEqual(fields['LLAMINAR_CPU_FUSED_VERIFIER_EXPERT_ROWS'], '64,64')
            self.assertEqual(fields['LLAMINAR_CPU_FUSED_VERIFIER_PURPOSE'], 'prefill')
            self.assertEqual(fields['LLAMINAR_CPU_FUSED_VERIFIER_EXPECTED_K_TILES'], '16')

    def test_native_timing_counter_and_byte_corruption_fail(self):
        for name in ('samples', 'route', 'bytes', 'placement'):
            payload = self.payload()
            if name == 'samples': payload['raw_samples_us'][0] = 14.0
            elif name == 'route': payload['counters']['authentication_calls'] = 2
            elif name == 'bytes': payload['correctness']['mismatch_count'] = 1
            else: payload['placement']['workers'] = 14
            with self.subTest(name=name), self.assertRaises(ValueError): self.adapt(payload)

    def test_profiler_batch_retains_each_request_vector_and_own_counter_path(self):
        from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation
        from native_quantised_dispatch.profiler_collectors import (
            CPUProfilerProcessBatch, _cpu_process_batch_key, _write_cpu_batch_plan)
        requests = tuple(profiler_request_for_observation(self.adapt(self.payload(m))) for m in (32, 64))
        self.assertEqual(_cpu_process_batch_key(requests[0]), _cpu_process_batch_key(requests[1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = root / 'requests.json'
            _write_cpu_batch_plan(plan, CPUProfilerProcessBatch(requests),
                                  {request.request_id: root / request.request_id for request in requests})
            content = json.loads(plan.read_text())
            self.assertEqual(content['schema'], 'cpu_execution_exact_batch_v1')
            self.assertEqual({tuple(r['expert_rows']) for r in content['requests']}, {(32, 32), (64, 64)})
            self.assertEqual(len({r['profile_output_path'] for r in content['requests']}), 2)

    def test_execution_exact_literal_roundtrip_preserves_full_m_and_unrelated_keys(self):
        from native_quantised_dispatch.cpu_exact_overlay import read_execution_exact_include
        entries = {key(): 0, key(rows=512): 1}
        self.assertEqual(read_execution_exact_include(generate_execution_exact_include(entries)), entries)
        with self.assertRaises(ValueError):
            read_execution_exact_include(generate_execution_exact_include(entries).replace('512u', '64u'))


    def test_installed_auto_requires_real_hits_and_the_measured_winner(self):
        payload = self.payload(); payload['candidate'] = 'Auto'
        payload['descriptor_policies'] = [1, 1]
        payload['baseline_descriptor_policies'] = [1, 1]
        payload['counters']['execution_exact_hits'] = 2
        context = CPUVerifierAdapterContext.workflow_smoke(corpus_id='sha256:'+'2'*64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'auto.json'; path.write_text(json.dumps(payload))
            result = adapt_cpu_execution_exact_native(path, context, holdout=False, installed_choices={key():1})
            self.assertEqual(result.physical_launch_evidence['requested_candidate'], 'Auto')
            self.assertTrue(result.physical_launch_evidence['installed_generated_lookup_verified'])
            payload['counters']['execution_exact_hits'] = 1; path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'generated full-key'):
                adapt_cpu_execution_exact_native(path, context, holdout=False, installed_choices={key():1})
            payload['counters']['execution_exact_hits'] = 2
            payload['descriptor_policies'] = [0, 0]; path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'descriptor policy'):
                adapt_cpu_execution_exact_native(path, context, holdout=False, installed_choices={key():1})

    def test_cpu_sudo_prefix_targets_native_owner_without_external_perf(self):
        from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation
        from native_quantised_dispatch.profiler_collectors import (
            CollectorOptions, _cpu_owner_command, _resolve_tool, _tool_version)
        from native_quantised_dispatch.schema import Backend
        request = profiler_request_for_observation(self.adapt(self.payload()))
        options = CollectorOptions(backend=Backend.CPU, binary=Path('/tmp/native-owner'),
            tool=Path('perf_event_open'), raw_directory=Path('/tmp/raw'), cpu_list='0-27',
            tool_command_prefix=('sudo','-n','-E'))
        command = _cpu_owner_command(options, options.binary, request)
        self.assertEqual(command[:7], ['sudo','-n','-E','taskset','-c','0-27','/tmp/native-owner'])
        with mock.patch('native_quantised_dispatch.profiler_collectors.subprocess.run', side_effect=AssertionError('external perf')):
            self.assertIn('perf_event_open', _tool_version(options))
            self.assertEqual(_resolve_tool(Backend.CPU, None), Path('perf_event_open'))
        with self.assertRaisesRegex(ValueError, 'not applicable'):
            _resolve_tool(Backend.CPU, Path('/usr/bin/perf'))


    def test_fused_profile_completion_requires_one_exact_target_marker(self):
        from native_quantised_dispatch.profiler_collectors import _cpu_profile_completion_count
        from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation
        request = profiler_request_for_observation(self.adapt(self.payload()))
        marker = f"execution_exact_request_complete,id={request.request_id},profile=1"
        wrong = [marker.replace(request.request_id, request.request_id + "foreign"),
                 marker.replace("profile=1", "profile=0"),
                 "prefix " + marker, marker + ",foreign=1",
                 f"request={request.request_id} launches=1"]
        self.assertEqual(_cpu_profile_completion_count(request, wrong), 0)
        self.assertEqual(_cpu_profile_completion_count(request, wrong + [marker]), 1)
        self.assertEqual(_cpu_profile_completion_count(request, [marker, marker]), 2)
        legacy = replace(request, operation_kind="NativeQuantisedDecodeProjection", execution_witness={})
        self.assertEqual(_cpu_profile_completion_count(legacy, [marker]), 0)
        self.assertEqual(_cpu_profile_completion_count(legacy, [f"request={request.request_id} launches=1"]), 1)
        with self.assertRaisesRegex(ValueError, 'original execution witness'):
            _cpu_profile_completion_count(replace(request, execution_witness={}), [marker])


    def test_native_profile_key_vector_policy_and_one_call_epoch_are_bound(self):
        from native_quantised_dispatch.profiler_collectors import _validate_cpu_execution_exact_profile
        from native_quantised_dispatch.profiler_evidence import profiler_request_for_observation
        payload = self.payload(); request = profiler_request_for_observation(self.adapt(payload))
        payload.update(schema='cpu_execution_exact_profile_v1', request_id=request.request_id,
            profile_request_id=request.request_id, profiled_calls=1, preconditioning_calls=2,
            activation_layout='aligned_tightly_packed_expert_rows_v1', activation_row_offsets=[0, 64])
        payload['counters'].update(projection_observations=2, expected_projection_observations=2)
        payload['counters'].pop('timed_collection_disabled')
        payload['counters'].pop('timed_projection_observations')
        payload['counters']['route_records'][0]['tags'] = dict(m='64', n='320', policy_n='320',
            k='2560', codebook='5', k_tiles='16', ambient_n_block_chunks='1',
            isa='AVX512', purpose='prefill', effective_verifier_policy='Pairwise')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native-profile.json'
            def check(value, required=True):
                path.write_text(json.dumps(value))
                _validate_cpu_execution_exact_profile(request, Path(directory), require_route_counters=required)
            check(payload)
            for field, value in (('request_id', 'foreign'), ('profiled_calls', 2),
                    ('exact_key_verified', 1), ('activation_row_offsets', [0, 0]),
                    ('exact_key', key(rows=63).scalars()),
                    ('descriptor_keys', [key(rows=63).scalars()] * 2),
                    ('descriptor_policies', [1, 1]), ('baseline_descriptor_policies', [1, 1])):
                changed = dict(payload); changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError): check(changed)
            for mutation in ('disabled', 'missing', 'extra', 'empty'):
                changed = json.loads(json.dumps(payload))
                if mutation == 'disabled': changed['counters']['enabled'] = False
                elif mutation == 'missing': changed.pop('counters')
                elif mutation == 'extra': changed['counters']['route_records'][0]['count'] = 3
                else: changed['counters']['route_records'] = []
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): check(changed)
            changed = json.loads(json.dumps(payload))
            changed['counters']['route_records'][0]['tags']['effective_verifier_policy'] = 'WideRows'
            with self.assertRaisesRegex(ValueError, 'route counters differ'): check(changed)
            single = dict(payload); single.pop('counters'); single.pop('profile_request_id')
            check(single, False)
            legacy = replace(request, operation_kind='NativeQuantisedDecodeProjection', execution_witness={})
            path.unlink()
            _validate_cpu_execution_exact_profile(legacy, Path(directory), require_route_counters=True)


class CPULegacyExactRetentionTest(unittest.TestCase):
    """Replay additive literal composition without fitting or native execution."""

    @staticmethod
    def verifier(keys, policies):
        return ("#define LLAMINAR_CPU_NATIVE_QUANTISED_VERIFIER_POLICY_ABI 3\n"
                "inline constexpr uint64_t kCPUNativeQuantisedVerifierExactAVX512AVX512T28Keys[] =\n{\n    "
                + ", ".join(f"0x{k:016x}ULL" for k in keys) + ",\n};\n"
                "inline constexpr uint8_t kCPUNativeQuantisedVerifierExactAVX512AVX512T28Policies[] =\n{\n    "
                + ", ".join(str(v) for v in policies) + ",\n};\n"
                "bool untouched_generic(int n) { return n < 73; }\n")

    @staticmethod
    def decode(rows):
        cases = "".join(f"            case 0x{k:016x}ULL:\n"
                        f"                // original {k}\n"
                        f"                policy = CPUNativeQuantisedDecodePolicy::Nbc{v};\n"
                        "                return true;\n" for k, v in rows)
        return ("#define LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_POLICY_ABI 1\n"
                "if (build_isa == CPUNativeQuantisedDecodeBuildISA::AVX512 &&\n"
                "        runtime_isa == CPUNativeQuantisedDecodeRuntimeISA::AVX512)\n"
                "{\n        if (threads == 28)\n        {\n            switch (key)\n            {\n"
                + cases + "            default:\n                break;\n            }\n        }\n"
                "    if (n < 73) return true; // immutable generic\n}\n")

    def test_verifier_refresh_preserves_unrelated_keys_and_generic(self):
        from native_quantised_dispatch.cpu_exact_retention import refresh_verifier_exact, _verifier_arrays
        base = self.verifier([1, 2], [0, 1])
        out, receipt = refresh_verifier_exact(base, self.verifier([2, 3], [0, 1]))
        arrays = _verifier_arrays(out)
        self.assertEqual(arrays['kCPUNativeQuantisedVerifierExactAVX512AVX512T28Keys'][1], (1, 2, 3))
        self.assertEqual(arrays['kCPUNativeQuantisedVerifierExactAVX512AVX512T28Policies'][1], (0, 0, 1))
        self.assertTrue(out.endswith('bool untouched_generic(int n) { return n < 73; }\n'))
        self.assertTrue(receipt['generic_program_unchanged'])

    def test_verifier_foreign_or_duplicate_literal_fails(self):
        from native_quantised_dispatch.cpu_exact_retention import refresh_verifier_exact
        base = self.verifier([1, 2], [0, 1])
        for fresh in (self.verifier([2, 2], [0, 1]), self.verifier([2], [9]),
                      self.verifier([2], [0]).replace('0x0000000000000002ULL', 'unknown()')):
            with self.subTest(fresh=fresh), self.assertRaises(ValueError):
                refresh_verifier_exact(base, fresh)

    def test_decode_refresh_preserves_original_case_and_generic(self):
        from native_quantised_dispatch.cpu_exact_retention import refresh_decode_exact, _decode_exact_arms
        base = self.decode([(1, 1), (2, 2)])
        out, receipt = refresh_decode_exact(base, self.decode([(2, 4), (3, 8)]))
        arms = _decode_exact_arms(out)
        rows = next(iter(arms.values()))[2]
        self.assertEqual(list(rows), [1, 2, 3])
        self.assertEqual(rows[1], next(iter(_decode_exact_arms(base).values()))[2][1])
        self.assertIn('Nbc4', rows[2]); self.assertTrue(receipt['generic_program_unchanged'])
        self.assertTrue(out.endswith('    if (n < 73) return true; // immutable generic\n}\n'))

    def test_decode_foreign_statement_or_thread_arm_fails(self):
        from native_quantised_dispatch.cpu_exact_retention import refresh_decode_exact
        base = self.decode([(1, 1)])
        for fresh in (self.decode([(2, 2)]).replace('threads == 28', 'threads == 14'),
                      self.decode([(2, 2)]).replace('default:', 'arbitrary();\n            default:')):
            with self.subTest(fresh=fresh), self.assertRaises(ValueError):
                refresh_decode_exact(base, fresh)


class CPUExactSweepTransactionTest(unittest.TestCase):
    """Exercise full-stage dependency/resume semantics without any native job."""

    def fixture(self, directory):
        from native_quantised_dispatch import cpu_exact_sweep as sweep
        root = Path(directory) / 'source'; owner = root / 'src/v2/kernels/cpu/gemm'; owner.mkdir(parents=True)
        for kind, name, selector, abi in (
                ('decode', 'Decode', 'selectCPUNativeQuantisedDecodeGeneratedPolicy', 'DECODE'),
                ('verifier', 'VerifierRows', 'selectCPUNativeQuantisedVerifierRowsGeneratedPolicy', 'VERIFIER')):
            (owner / f'CPUNativeQuantised{name}PolicyGenerated.inc').write_text(
                f'#define LLAMINAR_CPU_NATIVE_QUANTISED_{abi}_POLICY_ABI 1\n'
                '// Frozen generic policy digest: sha256:original\n'
                f'bool {selector}();\n')
        (owner / 'CPUNativeQuantisedExecutionExactGenerated.inc').write_text(
            'inline bool selectCPUExecutionExactPolicy(\n'
            'const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice) noexcept\n'
            '{ (void)key; (void)choice; return false; }\n')
        output = Path(directory) / 'output'; output.mkdir()
        args = SimpleNamespace(output_dir=output, stage='all', install=True, emit_only=False)
        plan = dict(plan_digest='sha256:plan', total_jobs=3,
                    job_counts=dict(legacy_decode=1, legacy_grouped=1, expert_rows=1))
        events = []; generation = {'value': dict(digest='before', programs={})}
        def emit(args, plan, kind):
            events.append('emit:' + kind)
            name = 'Decode' if kind == 'decode' else 'VerifierRows'
            target = output / 'generated' / f'CPUNativeQuantised{name}PolicyGenerated.inc'
            target.parent.mkdir(exist_ok=True)
            target.write_text((owner / target.name).read_text() + '// new exact selection\n')
            return target
        def rebuild(args):
            events.append('rebuild')
            generation['value'] = dict(digest='after-' + str(events.count('rebuild')), programs={})
        def collect(args, plan, original, stage):
            events.append('collect:' + stage)
            if stage == 'rows':
                self.assertIn('// new exact selection', (owner / 'CPUNativeQuantisedDecodePolicyGenerated.inc').read_text())
                self.assertNotEqual(original['digest'], 'before')
            return 0
        patches = [mock.patch.object(sweep, 'ROOT', root),
            mock.patch.object(sweep, '_producer_generation', side_effect=lambda _: generation['value']),
            mock.patch.object(sweep, '_emit_legacy', side_effect=emit),
            mock.patch.object(sweep, '_rebuild_after_decode', side_effect=rebuild),
            mock.patch.object(sweep, '_collect_stage', side_effect=collect),
            mock.patch.object(sweep, '_load_stage_corpus', return_value=[observe(row())]),
            mock.patch.object(sweep, '_profile', return_value={'authenticated': True}),
            mock.patch.object(sweep, '_post_install_auto_validation',
                              side_effect=lambda *args: (events.append('validate:auto') or 0)),
            mock.patch('native_quantised_dispatch.cpu_exact_overlay.select_execution_exact_entries',
                       return_value=({key(): 0}, {'entries': []}))]
        return sweep, args, plan, owner, events, patches

    def test_full_transaction_installs_m1_rebuilds_then_rows_and_completed_resume_is_passive(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            sweep, args, plan, owner, events, patches = self.fixture(directory)
            with ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                self.assertEqual(events, ['collect:decode', 'emit:decode', 'rebuild',
                                         'collect:rows', 'emit:verifier', 'rebuild', 'validate:auto'])
                before = list(events)
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                self.assertEqual(events, before)
                state = json.loads((args.output_dir / 'state.json').read_text())
                self.assertTrue(state['complete']); self.assertFalse(state['generic_fit'])
                self.assertFalse(state['ordinary_backend_native_prefill_installed'])
                (owner / 'CPUNativeQuantisedDecodePolicyGenerated.inc').write_text('foreign source mutation')
                with self.assertRaisesRegex(ValueError, 'generated program changed'):
                    sweep.run_transaction(args, plan)

    def test_rows_checkpoint_resume_never_recollects_old_m1_generation(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            sweep, args, plan, owner, events, patches = self.fixture(directory)
            with ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                original = sweep._collect_stage.side_effect
                def collect(args, plan, generation, stage):
                    if stage == 'rows' and 'checkpoint' not in events:
                        events.append('checkpoint'); return 2
                    return original(args, plan, generation, stage)
                sweep._collect_stage.side_effect = collect
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                self.assertFalse(json.loads((args.output_dir / 'state.json').read_text())['complete'])
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                self.assertEqual(events.count('collect:decode'), 1)
                self.assertEqual(events.count('emit:decode'), 1)
                self.assertEqual(events.count('rebuild'), 2)

    def test_rows_without_installed_m1_generation_fails_before_collecting(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            sweep, args, plan, owner, events, patches = self.fixture(directory)
            args.stage = 'rows'
            with ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                with self.assertRaisesRegex(ValueError, 'completed installed exact M1'):
                    sweep.run_transaction(args, plan)
                self.assertEqual(events, [])

    def test_plan_only_does_not_discover_or_launch_native_binary(self):
        from native_quantised_dispatch import cpu_exact_sweep as sweep
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(schema='plan', plan_digest='sha256:plan', job_counts={}, total_jobs=0)
            with mock.patch.object(sweep, 'build_plan', return_value=plan), \
                 mock.patch.object(sweep, '_producer_generation', side_effect=AssertionError('native discovery')), \
                 mock.patch.object(sweep.subprocess, 'run', side_effect=AssertionError('native launch')):
                self.assertEqual(sweep.main(['--output-dir', directory, '--threads', '2', '--plan-only']), 0)
                self.assertEqual(json.loads((Path(directory) / sweep.PLAN_NAME).read_text()), plan)

    def test_batch_command_uses_actual_existing_owner_and_distinct_request_outputs(self):
        from native_quantised_dispatch import cpu_exact_sweep as sweep
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); binary = root / 'build/tests/v2/v2_perf_cpu_fused_verifier'
            args = SimpleNamespace(fused_avx512_bin=binary)
            request = dict(kind='expert_rows',build_isa='AVX512',runtime_isa='AVX512',
                source_format='Q4_K',shape='Shared',n=320,k=2560,policy_n=320,role='gate_up',
                operation=2,purpose='prefill',m=3,candidate='Pairwise',holdout=False,expert_rows=[3,3,3])
            other = {**request, 'm':4, 'holdout':True, 'candidate':'WideRows', 'expert_rows':[4,4,4,1]}
            job = {**request, 'requests':[request,other]}
            command, env = sweep._command(args,dict(threads=2,warmups=1,samples=3,repeats=2),job,root)
            self.assertEqual(command[0], str(root / 'build/llaminar_mpi_session'))
            self.assertIn('--bind-to', command); self.assertEqual(env['OMP_NUM_THREADS'], '2')
            content = json.loads((root / 'requests.json').read_text())
            self.assertEqual(len(content['requests']),2)
            self.assertEqual(len({r['evidence_path'] for r in content['requests']}),2)
            self.assertEqual(content['requests'][1]['expert_rows'],[4,4,4,1])
            self.assertIn('kernel.cpu_native_quantised_execution_exact_policy_hits', env['LLAMINAR_PERF_STATS_FILTER'])


    def test_installed_auto_checkpoint_uses_final_producer_and_cannot_complete_early(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            sweep, args, plan, owner, events, patches = self.fixture(directory)
            with ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                sweep._post_install_auto_validation.side_effect = [2, 0]
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                state = json.loads((args.output_dir/'state.json').read_text())
                self.assertFalse(state['complete']); self.assertEqual(state['stage'],'post_install_auto')
                self.assertEqual(events.count('rebuild'),2)
                # Resume must not compare final compiled generation after-2 to
                # row timing generation after-1, nor rerun old forced timings.
                self.assertEqual(sweep.run_transaction(args, plan), 0)
                self.assertEqual(events.count('collect:rows'),1)
                self.assertEqual(events.count('rebuild'),2)
                state = json.loads((args.output_dir/'state.json').read_text())
                self.assertTrue(state['complete']); self.assertTrue(state['installed_auto_dispatch_verified'])

    def test_exact_key_import_does_not_load_retired_generic_fitter(self):
        import subprocess
        code = ("import sys;sys.path.insert(0," + repr(str(ROOT/'tests/v2/performance/kernels')) + ");"
                "import native_quantised_dispatch.cpu_exact_overlay;"
                "import native_quantised_dispatch.cpu_exact_sweep;"
                "assert 'native_quantised_dispatch.segmented_policy' not in sys.modules")
        subprocess.run([sys.executable,'-c',code],check=True,timeout=10)


class CPUAdaptiveExecutionExactContract(unittest.TestCase):
    """Authored timing-grade fixtures exercise existing exact selection without native claims."""

    @staticmethod
    def observations(*, holdout=False):
        from native_quantised_dispatch.adapters.evidence import native_double_digest
        contract=key()
        result=[]
        pair='cpu.native_quantised.verifier.pairwise'
        wide='cpu.native_quantised.verifier.wide_rows'
        for name,candidate in (('Pairwise',pair),('WideRows',wide)):
            original=observe(row(name,contract=contract,
                members=[contract,replace(contract,rows=1)] if holdout else None,holdout=holdout))
            final=name=='Pairwise'
            screen=[1.,1.,1.] if final else [10.,10.,10.]
            witness=dict(schema='native_quantised_tournament_v1',timing_stage='final' if final else 'screening',
                promotion_grade=final,finalist_reason=['screen_winner','installed_baseline'] if final else ['screened_out'],
                candidate_ids=[pair,wide],baseline_candidate_id=pair,winner_candidate_id=pair,
                finalist_candidate_ids=[pair],screen_samples_us=screen,
                screen_sorted_native_digest=native_double_digest(screen),
                contest_digest='sha256:'+('b' if holdout else 'a')*64,
                all_candidate_correctness_passed=True)
            clock=100. if final else 10.
            result.append(replace(original,min_us=clock,median_us=clock,p95_us=clock,
                physical_launch_evidence={**original.physical_launch_evidence,'tournament':witness}))
        return result

    def test_screening_clock_cannot_win_without_original_full_timing_floor(self):
        training=self.observations();holdout=self.observations(holdout=True)
        entries,_=select_execution_exact_entries(training+holdout)
        self.assertEqual(entries,{key():0}) # Faster screen-only WideRows is not winner evidence.
        with self.assertRaisesRegex(ValueError,'partial|matrix'):
            select_execution_exact_entries(training[:1]+holdout)
        foreign=replace(training[1],numerical_correctness=False)
        with self.assertRaises(ValueError):select_execution_exact_entries([training[0],foreign]+holdout)
        with self.assertRaisesRegex(ValueError,'mixes original'):
            select_execution_exact_entries([training[0],observe(row('WideRows'))]+holdout)

    def test_same_complete_contest_and_correct_holdout_remain_mandatory(self):
        training=self.observations();holdout=self.observations(holdout=True)
        wrong=replace(training[1],physical_launch_evidence={**training[1].physical_launch_evidence,
            'tournament':{**training[1].physical_launch_evidence['tournament'],'contest_digest':'sha256:'+'c'*64}})
        with self.assertRaisesRegex(ValueError,'contest'):
            select_execution_exact_entries([training[0],wrong]+holdout)
        with self.assertRaises(ValueError):select_execution_exact_entries(training)


if __name__ == "__main__":
    unittest.main()
