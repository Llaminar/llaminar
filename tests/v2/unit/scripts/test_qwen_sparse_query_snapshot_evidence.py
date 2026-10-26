#!/usr/bin/env python3
"""Tiny stdlib-only query snapshot schema regressions with AST-isolated validators.

No project module, native runtime, reference generator or payload is imported.
Fixtures describe original histories and acquired metadata independently.
"""
import ast
import copy
from enum import Enum, auto
import math
import json
import csv
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[4]


def pure_validators():
    """Compile only the two pure canonical validator definitions, never module imports."""
    path=ROOT/'scripts/ci/run_production_parity_campaigns.py'
    names={'expected_query_snapshot_interval','validate_stage_checkpoint_evidence','validate_query_program_omission'}
    body=[node for node in ast.parse(path.read_text()).body
          if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in body}!=names:
        raise AssertionError('Missing canonical pure query validators')
    scope={'math':math}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope


def fixture(prefix):
    """Describe both required original query checkpoints at one canonical M1 boundary."""
    columns=('backend,step,layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,snr_db,rmse,error_entropy,'
        'is_routing,routing_overlap,routing_top1_match,routing_weight_l1,proof_authority,canonical_passed,'
        'route_conditioned_evaluated,route_conditioned_passed,route_conditioned_cosine,route_conditioned_rel_l2,'
        'route_conditioned_max_abs,route_weights_max_error,hf_reconstruction_rel_l2,scalar_type,integer_mismatch_count,passed,'
        'llaminar_elements,llaminar_nan_count,llaminar_inf_count,pytorch_elements,pytorch_nan_count,pytorch_inf_count,'
        'checkpoint_evidence_kind,structural_valid,query_empty_reason,history_prompt_rows,history_decode_rows_before,'
        'history_live_rows,history_source_capacity,observed_request_count,observed_prefix_rows,observed_live_rows,'
        'observed_row_capacity,observed_source_capacity,observed_first_required_logical_row,observed_row_offset,'
        'observed_first_row,observed_end_row,observed_payload_row_offset,observed_physical_rows,observed_repeats_per_row,'
        'observed_slice_first_row,observed_payload_rows,observed_payload_columns,observed_payload_elements,'
        'reference_rows,reference_first_row,query_program_observation').split(',')
    empty=prefix<2051
    result=[columns]
    for stage in ('QSA_INDEXER_RAW_QUERY','QSA_INDEXER_QUERY'):
        row=dict.fromkeys(columns,'')
        row.update(backend='CPU0',step='0',layer='3',stage=stage,scalar_type='FP32',is_routing='0',
            route_conditioned_evaluated='0',checkpoint_evidence_kind='unauthored_query' if empty else 'authored_query',
            structural_valid='1',query_empty_reason='ranking_not_required' if empty else 'none',
            proof_authority='structural_query_interval' if empty else 'canonical_hf',
            canonical_passed='0' if empty else '1',passed='0' if empty else '1',
            history_prompt_rows=str(prefix),history_decode_rows_before='0',history_live_rows='1',history_source_capacity='4096',
            observed_request_count='1',observed_prefix_rows=str(prefix),observed_live_rows='1',observed_row_capacity='16',
            observed_source_capacity='4096',observed_first_required_logical_row='2051',observed_row_offset='0',
            observed_first_row='1' if empty else '0',observed_end_row='1',observed_payload_row_offset='0',
            observed_physical_rows='16',observed_repeats_per_row='1',observed_slice_first_row='0',
            observed_payload_rows='0' if empty else '1',observed_payload_columns='512',
            observed_payload_elements='0' if empty else '512',reference_rows='1',reference_first_row='1' if empty else '0')
        if not empty:
            row.update(cosine='1',cosine_drop='0',rel_l2='0',max_abs_diff='0',snr_db='inf',rmse='0',error_entropy='0',
                llaminar_elements='512',llaminar_nan_count='0',llaminar_inf_count='0',
                pytorch_elements='512',pytorch_nan_count='0',pytorch_inf_count='0')
        result.append([row[column] for column in columns])
    return result


def set_value(rows,field,value,row=1):
    """Mutate one independent fixture cell without invoking production code."""
    rows[row][rows[0].index(field)]=value


class QwenSparseQuerySnapshotEvidence(unittest.TestCase):
    """Original-history false-skip controls and preserved typed numerical evidence."""
    @classmethod
    def setUpClass(cls):
        cls.pure=pure_validators()

    def validate(self,rows,decode=True):
        return self.pure['validate_stage_checkpoint_evidence'](rows,decode=decode)

    def test_independent_pool_law_covers_crossing_and_empty_buckets(self):
        derive=self.pure['expected_query_snapshot_interval']
        for prefix in range(2047,2054):
            for live in (*range(17),17,64):
                expected=[row for row in range(live) if (prefix+row+1)//4>512]
                first=expected[0] if expected else live
                self.assertEqual(derive(prefix,0,live,4096),(prefix,first,live))
        for values in ((2050,0,-1,4096),(2050,0,3,2052),(2147483647,1,1,2147483647),
                       (True,0,1,4096),(2050,0,1,0)):
            with self.assertRaises(ValueError):derive(*values)

    def test_empty_and_first_needed_keep_both_checkpoint_identities(self):
        self.assertIsNone(self.validate(fixture(2050)))
        self.assertIsNone(self.validate(fixture(2051)))
        self.assertIsNone(self.validate(fixture(4095)))

    def test_wrong_zero_at2052_and_extra_authored_at2051_fail(self):
        wrong=fixture(2050)
        for row in (1,2):
            set_value(wrong,'history_prompt_rows','2051',row)
            set_value(wrong,'observed_prefix_rows','2051',row)
        self.assertIsNotNone(self.validate(wrong))
        wrong=fixture(2051)
        for row in (1,2):
            set_value(wrong,'history_prompt_rows','2050',row)
            set_value(wrong,'observed_prefix_rows','2050',row)
        self.assertIsNotNone(self.validate(wrong))

    def test_stale_frontier_wrong_policy_missing_or_malformed_record_fails(self):
        for field,value in (('observed_prefix_rows','2047'),('observed_first_required_logical_row','2052'),
            ('observed_request_count','0'),('observed_request_count','2'),('observed_source_capacity','8192'),
            ('observed_live_rows','0'),('observed_row_offset','1'),('observed_payload_row_offset','1'),
            ('observed_repeats_per_row','4'),('observed_slice_first_row','1'),('query_empty_reason','none'),
            ('history_prompt_rows','not-an-int'),('history_decode_rows_before','1'),('structural_valid','0')):
            with self.subTest(field=field,value=value):
                rows=fixture(2048);set_value(rows,field,value)
                self.assertIsNotNone(self.validate(rows))
        rows=fixture(2050);rows[0][rows[0].index('query_empty_reason')]='missing_reason'
        self.assertIsNotNone(self.validate(rows))

    def test_original_reference_window_is_not_native_last_size_crop(self):
        rows=fixture(2051)
        for row in (1,2):
            set_value(rows,'reference_rows','2052',row)
            set_value(rows,'reference_first_row','2051',row)
        self.assertIsNone(self.validate(rows))
        set_value(rows,'reference_first_row','2052')
        self.assertIsNotNone(self.validate(rows))

    def test_missing_duplicate_wrong_phase_and_wrong_kind_rejected(self):
        rows=fixture(2050)
        self.assertIsNotNone(self.validate(rows[:-1]))
        self.assertIsNotNone(self.validate(rows+[rows[1][:]]))
        self.assertIsNotNone(self.validate(rows,decode=False))
        set_value(rows,'checkpoint_evidence_kind','compared_values')
        self.assertIsNotNone(self.validate(rows))
        rows=fixture(2050);set_value(rows,'stage','ATTENTION_OUTPUT')
        self.assertIsNotNone(self.validate(rows))

    def test_structural_values_cannot_invent_numeric_equality(self):
        for field,value in (('passed','1'),('canonical_passed','1'),('cosine','1'),
                            ('llaminar_elements','0'),('pytorch_nan_count','0'),
                            ('route_conditioned_passed','1'),('proof_authority','canonical_hf')):
            rows=fixture(2050);set_value(rows,field,value)
            self.assertIsNotNone(self.validate(rows))

    def test_useful_queries_keep_real_finite_numeric_witnesses(self):
        for field,value in (('cosine','nan'),('rel_l2','inf'),('max_abs_diff','nan'),
                           ('llaminar_nan_count','1'),('pytorch_inf_count','1'),
                           ('llaminar_elements','0'),('pytorch_elements','1024')):
            rows=fixture(2051);set_value(rows,field,value)
            self.assertIsNotNone(self.validate(rows))

    def test_nonquery_and_prefill_values_keep_existing_policy(self):
        for decode in (False,True):
            rows=fixture(2051)
            start=rows[0].index('checkpoint_evidence_kind')
            for row in (1,2):
                for column in range(start,len(rows[0])):rows[row][column]=''
                set_value(rows,'checkpoint_evidence_kind','compared_values',row)
                set_value(rows,'stage','ATTENTION_OUTPUT' if row==1 else 'FFN_RESIDUAL',row)
            self.assertIsNone(self.validate(rows,decode=decode))

    def test_actual_reference_enum_preserves_distinct_raw_and_prepared_identity(self):
        path=ROOT/'python/reference/pipeline_stages.py'
        tree=ast.parse(path.read_text())
        body=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='PipelineStage']
        body += [node for node in tree.body if isinstance(node,ast.AnnAssign) and
                 isinstance(node.target,ast.Name) and node.target.id=='_STAGE_TO_STRING']
        scope={'Enum':Enum,'auto':auto,'Dict':dict}
        exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
        stage=scope['PipelineStage'];mapping=scope['_STAGE_TO_STRING']
        self.assertEqual(mapping[stage.QSA_INDEXER_RAW_QUERY],'QSA_INDEXER_RAW_QUERY')
        self.assertEqual(mapping[stage.QSA_INDEXER_QUERY],'QSA_INDEXER_QUERY')
        self.assertNotEqual(stage.QSA_INDEXER_RAW_QUERY,stage.QSA_INDEXER_QUERY)



def omission_fixture():
    """Author actual path metadata independently of native snapshot coordinates."""
    rows = fixture(2050)
    tags = {"program": "proven_no_ranking", "role": "0", "physical_rows": "1",
            "observation": "successful_forward_submission_not_native_replay_count"}
    witness = {"schema": "qsa.program_omission.v1", "participant_source": "admitted_rank_local_model_devices",
        "expected_devices": ["CPU"], "before": [{"device": "CPU", "phase": "decode", "kind": 0, "count": 7, "tags": tags}],
        "after": [{"device": "CPU", "phase": "decode", "kind": 0, "count": 8, "tags": tags}]}
    for values in rows[1:]:
        for i, field in enumerate(rows[0]):
            if field.startswith("observed_"): values[i] = ""
            if field == "query_empty_reason": values[i] = "program_omitted_ranking_not_required"
            if field == "query_program_observation": values[i] = json.dumps(witness)
    return rows


class ProgramOmissionCSV(unittest.TestCase):
    def test_admitted_cuda_rocm_and_mixed_participant_sets_are_exact(self):
        """Every actual participant witness needs one fresh matching path record."""
        check = pure_validators()["validate_stage_checkpoint_evidence"]
        sets = [[f"{backend}:{ordinal}" for ordinal in range(degree)]
                for backend in ("CUDA", "ROCm") for degree in (1, 2, 4, 8)]
        sets.append(["CPU", "CUDA:0", "ROCm:1"])
        for devices in sets:
            with self.subTest(devices=devices):
                rows = omission_fixture()
                index = rows[0].index("query_program_observation")
                for values in rows[1:]:
                    witness = json.loads(values[index])
                    before, after = witness["before"][0], witness["after"][0]
                    witness["expected_devices"] = devices
                    witness["before"] = [{**before, "device": device} for device in devices]
                    witness["after"] = [{**after, "device": device} for device in devices]
                    values[index] = json.dumps(witness)
                self.assertIsNone(check(rows, decode=True))
                mutations = (
                    lambda w: w["after"].pop(),
                    lambda w: w["expected_devices"].append(devices[0]),
                    lambda w: w["expected_devices"].pop(),
                    lambda w: w["after"].append({**w["after"][0], "device": "CUDA:99", "count": 1}),
                    lambda w: w["after"].append(copy.deepcopy(w["after"][0])),
                    lambda w: w["after"][-1].update(count=6),
                    lambda w: w["after"][-1].update(count=7),
                )
                for mutation in mutations:
                    broken = copy.deepcopy(rows)
                    for values in broken[1:]:
                        witness = json.loads(values[index])
                        mutation(witness)
                        values[index] = json.dumps(witness)
                    self.assertIsNotNone(check(broken, decode=True))

    def test_typed_program_rows_require_both_path_and_history(self):
        check = pure_validators()["validate_stage_checkpoint_evidence"]
        rows = omission_fixture()
        self.assertIsNone(check(rows, decode=True))
        for mutate in (lambda w: w["after"][0].update(count=7),
                       lambda w: w["after"][0].update(device="CUDA:9"),
                       lambda w: w["expected_devices"].append("CPU"),
                       lambda w: w["after"][0]["tags"].update(program="ranking_possible"),
                       lambda w: w["after"].append({**w["after"][0], "tags": {**w["after"][0]["tags"], "role": "1"}})):
            broken = copy.deepcopy(rows)
            index = broken[0].index("query_program_observation")
            for values in broken[1:]:
                witness = json.loads(values[index]); mutate(witness); values[index] = json.dumps(witness)
            self.assertIsNotNone(check(broken, decode=True))
        for field, value in (("history_prompt_rows", "2051"), ("observed_payload_elements", "0"),
                             ("reference_first_row", "0"), ("cosine", "1")):
            broken = copy.deepcopy(rows)
            for values in broken[1:]: values[broken[0].index(field)] = value
            self.assertIsNotNone(check(broken, decode=True))



if __name__=='__main__':unittest.main(verbosity=2)
