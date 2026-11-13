#!/usr/bin/env python3
"""Fast metadata/metric contracts for the original pooled-ngram trace tooling.

Tiny authored records below are validator controls, not a corpus, cache replay,
physical source permit, observed GPU result, or performance measurement.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[2] / "performance/kernels/qwen4exp/qfn_ngram_trace.py"
spec = importlib.util.spec_from_file_location("qfn_ngram_trace", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def trace():
    """A deliberately small saved-file schema control; no external corpus payload."""
    return {"schema": 1, "origin": "saved_pytorch_ple_indices", "native_execution_proven": False,
            "timing_qualified": False, "source": {"tensor": "per_layer_token_embd.weight", "format": "IQ4_NL",
            "rows": 1024, "row_elements": 160, "native_row_bytes": 90}, "maximum_original_ids": 32,
            "histories": [{"id": "metadata-control", "prompt_ids": [7, 11], "continuation_ids": [13]}],
            "frames": [{"id": "prefill", "history_id": "metadata-control", "phase": "prefill", "step": 0,
            "layer": 1, "tokens": [7, 11], "ids": list(range(32)), "intent": "current", "outcome": "consumed",
            "snapshot_path": "metadata-only-placeholder", "snapshot_sha256": "0" * 64}],
            "unsupported_coverage": ["native_execution", "future_read_ahead", "cancelled_mtp", "timing"]}


def sample(t):
    """Counter-shaped consistency input, explicitly not measured execution."""
    f = t["frames"][0]
    counters = dict.fromkeys(m.COUNTERS, 0)
    counters.update(read_original_ids=len(f["ids"]), consumed_original_ids=len(f["ids"]))
    return {"mode": "combined", "consumer": "rank0/cuda0", "frame_id": f["id"], "counter_deltas": counters,
            "fetched_native_rows": len(set(f["ids"])), "native_payload_bytes": len(set(f["ids"])) * 90,
            "peak_response_bytes": 32 * 90, "response_capacity_bytes": 32 * 90,
            "promotions": 0, "evictions": 0}


def evidence(s):
    return {"schema": 1, "scope": "terminal_consistency_only", "frames": [s]}


def budgets():
    """Equal original allocator ceilings; modes can require different exact BOMs."""
    return [{"mode": mode, "pma_resources": [
        {"world_rank": 0, "device": "cpu:0", "capacity_bytes": 8192, "complete_bom_bytes": 4096},
        {"world_rank": 0, "device": "cuda:0", "capacity_bytes": 16384, "complete_bom_bytes": 12345}]}
            for mode in sorted(m.MODES)]


def campaign_selection():
    """Tiny source-selection metadata control; never an authenticated real corpus."""
    from types import SimpleNamespace
    exact = "original.ProductionParity/Off"
    return SimpleNamespace(exact=exact, record={"runtime": {"generation": {"mtp_policy": "off"}}},
        corpus={"manifest": {"source": {"identity": "metadata-control"}, "histories": [
            {"id": "first", "context_group": "short", "prompt": "Natural first.", "prompt_ids": [1, 2]},
            {"id": "second", "context_group": "heldout", "prompt": "Changed conversation.", "prompt_ids": [3, 4]}]}},
        topology={"canonical_cell": exact, "expected_config": {"configuration": {
            "temperature": 0, "top_k": 1, "top_p": 1, "seed": 42}}})


class QFNNgramTraceContract(unittest.TestCase):
    def test_campaign_seed_requires_a_reproducible_public_cohort(self):
        """The real default -1 must not survive metadata projection into a positive-seed campaign."""
        selection = campaign_selection()
        config = selection.topology["expected_config"]["configuration"]
        config["seed"] = -1
        original = copy.deepcopy(selection.__dict__)
        with self.assertRaisesRegex(ValueError, "positive 32-bit seed"):
            m.campaign_from_selection(selection, ["first"], Path("/private/output"))
        fixed = m.campaign_from_selection(selection, ["first"], Path("/private/output"), seed=17)
        self.assertEqual(fixed["sampling"], {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 17})
        self.assertEqual(selection.__dict__, original)
        for invalid in (-1, 0, True, 1.5, "17", 2147483648):
            with self.subTest(seed=invalid), self.assertRaisesRegex(ValueError, "positive 32-bit seed"):
                m.campaign_from_selection(selection, ["first"], Path("/private/output"), seed=invalid)
        for seed in (1, 2147483647):
            config["seed"] = seed
            preserved = m.campaign_from_selection(selection, ["first"], Path("/private/output"))
            self.assertEqual(preserved["sampling"]["seed"], seed)

    def test_campaign_reuses_original_natural_text_and_expected_ids(self):
        selection = campaign_selection()
        result = m.campaign_from_selection(selection, ["first", "second"], Path("/private/output"), repetitions=2)
        self.assertEqual(result["schema"], "llaminar.benchmark-campaign.v1")
        self.assertEqual([x["prompt_id"] for x in result["cells"]], ["first", "second", "first", "second"])
        self.assertEqual(result["prompts"][0]["text"], "Natural first.")
        self.assertEqual(result["prompts"][0]["frozen_history"]["prompt_ids"], [1, 2])
        self.assertEqual(result["prompts"][1]["frozen_history"]["canonical_cell"], selection.exact)
        self.assertTrue(all(x["partition"] == "holdout" for x in result["prompts"]))
        self.assertTrue(all(x["mtp"] == {"mode": "off"} for x in result["cells"]))

    def test_campaign_rejects_unknown_duplicate_or_foreign_provenance(self):
        for ids in (["unknown"], ["first", "first"], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                m.campaign_from_selection(campaign_selection(), ids, Path("/private/output"))
        for fault in ("cell", "duplicate", "tokens", "mtp", "identity"):
            selection = campaign_selection()
            if fault == "cell": selection.topology["canonical_cell"] = "foreign"
            if fault == "duplicate": selection.corpus["manifest"]["histories"] *= 2
            if fault == "tokens": selection.corpus["manifest"]["histories"][0]["prompt_ids"] = [True]
            if fault == "mtp": selection.record["runtime"]["generation"]["mtp_policy"] = "fixed"
            if fault == "identity": selection.corpus["manifest"]["histories"][0]["context_group"] = "../escape"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                m.campaign_from_selection(selection, ["first"], Path("/private/output"))

    def test_original_live_rows_and_trace_scope(self):
        t = trace()
        self.assertEqual(m.validate_trace(t)["original_ids"], 32)
        self.assertFalse(m.validate_trace(t)["native_execution_proven"])
        self.assertFalse(m.validate_terminal_metrics(t, evidence(sample(t)))["timing_qualified"])

    def test_original_phase_token_and_source_bound(self):
        for field, value in (("phase", "draft"), ("tokens", [7, 12]), ("step", 1), ("ids", [1024] * 32)):
            t = trace(); t["frames"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.validate_trace(t)
        t = trace(); t["frames"][0].update(phase="decode", step=0, tokens=[13], ids=list(range(16)))
        self.assertEqual(m.validate_trace(t)["original_ids"], 16)
        t["frames"][0]["step"] = 1
        with self.assertRaises(ValueError): m.validate_trace(t)

    def test_real_reference_cannot_invent_future_cancelled_or_native_work(self):
        for key, value in (("intent", "future"), ("outcome", "cancelled")):
            t = trace(); t["frames"][0][key] = value
            with self.assertRaises(ValueError): m.validate_trace(t)
        for field in ("native_execution_proven", "timing_qualified"):
            t = trace(); t[field] = True
            with self.assertRaises(ValueError): m.validate_trace(t)
        t = trace(); t["unsupported_coverage"].remove("cancelled_mtp")
        with self.assertRaises(ValueError): m.validate_trace(t)

    def test_exact_bounded_query_and_integer_identity(self):
        for maximum in (0, 16, 31, True):
            t = trace(); t["maximum_original_ids"] = maximum
            with self.assertRaises(ValueError): m.validate_trace(t)
        t = trace(); t["frames"][0]["ids"][0] = True
        with self.assertRaises(ValueError): m.validate_trace(t)
        for key in ("histories", "frames"):
            t = trace(); t[key].append(copy.deepcopy(t[key][0]))
            with self.assertRaises(ValueError): m.validate_trace(t)

    def test_cancelled_adversarial_control_never_credits_hot_policy(self):
        t = trace(); t["origin"] = "adversarial_metadata_control"
        t["frames"][0].update(intent="future", outcome="cancelled")
        s = sample(t); s["counter_deltas"]["consumed_original_ids"] = 0
        s["counter_deltas"]["future_wasted_native_rows"] = 32
        m.validate_terminal_metrics(t, evidence(s))
        for key in ("consumed_original_ids", "future_useful_native_rows"):
            bad = copy.deepcopy(s); bad["counter_deltas"][key] = 1
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(bad))
        for key in ("promotions", "evictions"):
            bad = copy.deepcopy(s); bad[key] = 1
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(bad))

    def test_future_consumed_control_needs_real_completion_classification(self):
        t = trace(); t["origin"] = "adversarial_metadata_control"; t["frames"][0]["intent"] = "future"
        s = sample(t); s["counter_deltas"].update(future_on_time_native_rows=32, future_useful_native_rows=32)
        m.validate_terminal_metrics(t, evidence(s))
        s["counter_deltas"]["future_on_time_native_rows"] = 0
        with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(s))

    def test_payload_uses_actual_unique_native_rows_not_capacity(self):
        t = trace(); t["frames"][0]["ids"] = [7] * 32
        s = sample(t); self.assertEqual(s["native_payload_bytes"], 90)
        m.validate_terminal_metrics(t, evidence(s))
        for key, value in (("fetched_native_rows", 32), ("native_payload_bytes", 2880), ("response_capacity_bytes", 90)):
            bad = copy.deepcopy(s); bad[key] = value
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(bad))

    def test_staging_hits_and_consumption_cannot_be_forged(self):
        t = trace(); s = sample(t)
        for key, value in (("peak_response_bytes", 2881), ("peak_response_bytes", 0)):
            bad = copy.deepcopy(s); bad[key] = value
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(bad))
        for key, value in (("local_hit_original_ids", 33), ("all_hit_wait_avoided_original_ids", 1),
                           ("read_original_ids", 0), ("future_late_native_rows", 1)):
            bad = copy.deepcopy(s); bad["counter_deltas"][key] = value
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(bad))

    def test_local_counters_do_not_relabel_other_paths(self):
        t = trace(); s = sample(t)
        for mode in ("consumer_local", "shared_only"):
            s["mode"] = mode
            with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(s))
        s = sample(t); s["counter_deltas"]["invented_counter"] = 0
        with self.assertRaises(ValueError): m.validate_terminal_metrics(t, evidence(s))

    def test_same_pma_budget_per_allocator_not_merely_total(self):
        b = budgets(); m.validate_equal_budgets(b)
        b[0]["pma_resources"][0]["capacity_bytes"] += 1
        b[0]["pma_resources"][1]["capacity_bytes"] -= 1
        with self.assertRaises(ValueError): m.validate_equal_budgets(b)
        b = budgets(); b[0]["pma_resources"][0]["device"] = "cpu:1"
        with self.assertRaises(ValueError): m.validate_equal_budgets(b)
        b = budgets(); b[0]["pma_resources"].append(copy.deepcopy(b[0]["pma_resources"][0]))
        with self.assertRaises(ValueError): m.validate_equal_budgets(b)
        b = budgets(); b[0]["pma_resources"][0]["complete_bom_bytes"] = 8193
        with self.assertRaises(ValueError): m.validate_equal_budgets(b)

    def test_duplicate_json_authority_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(ValueError): m.load_document(path)

    def test_snapshot_metadata_duplicate_authority_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.txt"
            path.write_text("token_ids: 7,11\ntoken_ids: 13\n")
            with self.assertRaises(ValueError): m.metadata(path)


if __name__ == "__main__":
    unittest.main()
