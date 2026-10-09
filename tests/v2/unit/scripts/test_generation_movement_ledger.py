#!/usr/bin/env python3
"""Device-free adversarial checks of authority-owned HTTP movement evidence.

Synthetic journals exercise schema, closed-cycle, economic and immutable-history
contracts, including different HTTP and terminal publication cutoffs. Exact
interior identities cannot be replaced by equal totals or endpoint ranges.
They do not certify physical transfers or any model topology.
The canonical generation runner additionally retains real server path evidence.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
from generation_movement_ledger import (MovementLedgerObserver, MovementRequirement,
    validate_movement_transport_mirrors, native_movement_sequence, native_movement_edge_words,
    validate_controller_movement_transport, controller_movement_words, controller_movement_edge_words,
    CONTROLLER_MOVEMENT_TAGS, _CONTROLLER_FIELDS, _CONTROLLER_COUNTER_FIELDS)
from runtime_feature_perf_policy import MovementEvidence, validate_runtime_feature_policy
from server_execution_contract import RuntimeFeaturePolicy


def ledger(authority="device", transaction=1, epoch=2, axis="combined"):
    """One balanced two-edge owner publication, optionally with host admission."""
    if authority == "host":
        epoch = transaction
    wave = {"authority": authority, "transaction": transaction, "candidate_epoch": epoch}
    edges = [{**wave, "layer": 3, "expert": expert, "cycle_index": 0, "cycle_size": 2,
              "movement_axis": axis, "direction": direction, "source_participant": source,
              "destination_participant": destination, "source_priority": source_priority,
              "destination_priority": destination_priority, "source_device": source_device,
              "destination_device": destination_device, "source_world_rank": source,
              "destination_world_rank": destination, "estimated_weight_bytes": 1024,
              "activation_count": 0, "blocking_inference": False}
             for expert, source, destination, source_priority, destination_priority, direction, source_device, destination_device in (
                 (7, 0, 1, -20, 17, "demotion", "CUDA:0", "ROCm:0"),
                 (8, 1, 0, 17, -20, "promotion", "ROCm:0", "CUDA:0"))]
    economy = [{**wave, "command_count": 2, "cycle_count": 1, "policy": "time_ns",
                "projected_service_gain_ns": 100, "projected_transfer_and_repack_ns": 30,
                "projected_inference_interference_ns": 10, "projected_net_benefit_ns": 60}]
    counts = {"tier_residency": 0, "participant_placement": 0, "combined": 0}
    counts[axis] = 1
    admissions = [] if authority == "device" else [{**wave,
        "cycle_capacity_kind": "bounded", "maximum_concurrent_cycles": 1,
        "candidate_cycles": 1, "policy_eligible_cycles": 1, "policy_eligible_axes": counts,
        "admitted_candidate_cycles": 1, "admitted_candidate_axes": counts,
        "admitted_physical_cycles": 1, "admitted_physical_axes": counts,
        "individual_policy_rejected_cycles": 0, "dependent_payoff_rejected_cycles": 0,
        "capacity_rejected_cycles": 0, "participant_axis_budget_rejected_cycles": 0,
        "dependent_cohort_candidates": 0, "dependent_cohort_payoff_rejections": 0,
        "physical_cycle_recomposition": False, "capacity_bounded": False, "policy_bounded": False}]
    return {"schema": 2, "scope": "model_lifetime", "complete": True, "discarded_edges": 0,
            "discarded_economy_records": 0, "discarded_host_admission_records": 0,
            "edges": edges, "economy": economy, "host_admissions": admissions}


def empty_ledger():
    """The same versioned schema without published movement, not missing proof."""
    value = ledger()
    for name in ("edges", "economy", "host_admissions"):
        value[name] = []
    return value


def topology(authority="device", axes=("tier_residency", "participant_placement")):
    """Explicit synthetic admitted geometry, independent of any journal content."""
    return {"schema": 1, "scope": "model_lifetime", "authority": authority, "available_axes": list(axes)}


def response(value, expected_topology=None):
    """Wrap only the terminal boundary consumed by this focused interpreter."""
    return {"runtime_summary": {"schema": 1, "expert_movement": value,
                               "expert_movement_topology": topology() if expected_topology is None else expected_topology}}


def controller_transport(journal, ranks=2):
    """Completed topology-wide history with exactly one policy-owning rank."""
    publications = []
    for economy in journal["economy"]:
        edges = [edge for edge in journal["edges"] if edge["transaction"] == economy["transaction"]]
        receipt = dict.fromkeys(_CONTROLLER_FIELDS, 0)
        receipt.update(base_epoch=economy["candidate_epoch"] - 1,
            promotions=sum(e["direction"] == "promotion" for e in edges),
            demotions=sum(e["direction"] == "demotion" for e in edges),
            same_priority_moves=sum(e["direction"] == "same_priority" for e in edges),
            cross_domain_moves=sum(e["source_priority"] != e["destination_priority"] for e in edges),
            cross_rank_moves=sum(e["source_world_rank"] is not None and e["destination_world_rank"] is not None
                                 and e["source_world_rank"] != e["destination_world_rank"] for e in edges),
            cross_backend_moves=sum(e["source_device"].split(":")[0] != e["destination_device"].split(":")[0] for e in edges),
            snapshot_observations=64, accepted_cycles=economy["cycle_count"], physical_cycles=economy["cycle_count"], changed_layers=1,
            edges_checked=len(edges), participant_coordinates_checked=2, tier_coordinates_checked=2)
        receipt.update({key: value for key, value in economy.items() if key.endswith("_ns")})
        publications.append({"transaction": economy["transaction"], "candidate_epoch": economy["candidate_epoch"],
            "command_count": economy["command_count"], "physical_payload_bytes": 10007, "controller": receipt})
    wave = native_movement_sequence(controller_movement_words(p) for p in publications)
    edge = native_movement_sequence(controller_movement_edge_words(e) for e in journal["edges"])
    records, terminals = [], []
    for rank in range(ranks):
        terminal = {"schema": 1, "scope": "terminal_model_lifetime", "rank": rank,
                    "movement": copy.deepcopy(journal), "device_publications": copy.deepcopy(publications)}
        if rank:
            terminal["movement"]["economy"] = []
        terminals.append(terminal)
        for name, field in _CONTROLLER_COUNTER_FIELDS.items():
            value = float(len(journal["edges"])) if field == "edges" else 0.0
            if field != "edges":
                for p in publications:
                    value += float(1 if field is None else p.get(field, p["controller"].get(field)))
            records.append({"kind": "counter", "domain": "moe_overlay_controller", "name": name, "rank": rank,
                "device": "overlay", "phase": "maintenance", "tags": dict(CONTROLLER_MOVEMENT_TAGS),
                "value": value, **(edge if field == "edges" else wave)})
    return records, terminals


class MovementLedgerTests(unittest.TestCase):
    """Every rejection is independent of token equality or optional PerfStats."""

    @staticmethod
    def native_load_ledger():
        """A native swap improves layer spread even when participant totals worsen."""
        value = ledger(axis="participant_placement")
        for edge in value["edges"]:
            edge.update(source_priority=0, destination_priority=0, direction="same_priority",
                        source_device=f'CUDA:{edge["source_participant"]}',
                        destination_device=f'CUDA:{edge["destination_participant"]}')
        value["economy"] = [{k: v for k, v in value["economy"][0].items() if not k.endswith("_ns")}]
        value["economy"][0].update(
            policy="native_load_spread", accepted_spread_improvement=40,
            pre_wave_spread=100, post_wave_spread=60, pre_wave_total=200, post_wave_total=200,
            pre_participant_spread=0, post_participant_spread=20,
            pre_participant_total=200, post_participant_total=200,
            requested_payload_slots=1, minimum_improvement_per_slot=40,
            maximum_post_spread_per_mille=300, ownership_swap_accepts=1)
        return value

    def test_native_load_policy_preserves_units_and_all_exact_admission_equations(self):
        value = self.native_load_ledger()
        native_topology = topology(axes=("participant_placement",))
        self.check(value, expected_topology=native_topology)
        for name, bad in (("policy", "time_ns"), ("policy", None),
                          ("accepted_spread_improvement", 39), ("ownership_swap_accepts", 2),
                          ("post_wave_spread", 100), ("post_wave_total", 201),
                          ("post_participant_total", 201), ("requested_payload_slots", 0),
                          ("requested_payload_slots", True), ("maximum_post_spread_per_mille", 299),
                          ("minimum_improvement_per_slot", 2**32), ("projected_net_benefit_ns", 1)):
            changed = copy.deepcopy(value)
            changed["economy"][0][name] = bad
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(changed, expected_topology=native_topology)
        huge = copy.deepcopy(value)
        huge["economy"][0].update(pre_wave_total=(2**64-1)//300+1, post_wave_total=(2**64-1)//300+1)
        self.check(huge, expected_topology=native_topology)

    def test_native_load_proof_cannot_certify_host_or_cross_tier_movement(self):
        for authority in ("host", "device"):
            changed = ledger(authority)
            proof = self.native_load_ledger()["economy"][0]
            changed["economy"][0].update({k: v for k, v in proof.items()
                                        if k not in ("authority", "transaction", "candidate_epoch")})
            changed["economy"][0] = {k: v for k, v in changed["economy"][0].items() if not k.endswith("_ns")}
            with self.subTest(authority=authority), self.assertRaises(ValueError):
                self.check(changed)
        changed = ledger()
        changed["economy"][0]["pre_wave_spread"] = 100
        with self.assertRaises(ValueError):
            self.check(changed)

    @staticmethod
    def transport_mirrors(journal):
        """Build commit-site mirrors, not substitute transport or owner state."""
        return [{"domain": "moe_overlay_residency" if edge["authority"] == "host" else "moe_overlay_controller",
                 "name": "expert_migration_edges" if edge["authority"] == "host" else "dynamic_migration_edges",
                 "value": 1, "rank": 0, "tags": {**{k: str(v) for k, v in edge.items()},
                    "policy_owner": edge["authority"], "transaction_purpose": "live_placement_change"}}
                for edge in journal["edges"]]

    def test_http_journal_is_bound_to_the_completed_transport_edge_identities(self):
        for owner in ("host",):
            journal = ledger(owner)
            mirrors = self.transport_mirrors(journal)
            validate_movement_transport_mirrors(journal, mirrors)
            # Extra later waves and rank mirrors do not replace missing edges.
            validate_movement_transport_mirrors(journal, mirrors + self.transport_mirrors(ledger(owner, transaction=9, epoch=10)))
            validate_movement_transport_mirrors(journal, mirrors + [r | {"rank": 1} for r in mirrors])
            with self.assertRaisesRegex(ValueError, "matching completed transport"):
                validate_movement_transport_mirrors(journal, [mirrors[0], mirrors[0] | {"rank": 1}])
            for field in ("candidate_epoch", "expert", "source_participant", "destination_participant",
                          "source_priority", "destination_priority", "source_device", "destination_device",
                          "movement_axis", "direction", "policy_owner"):
                changed = copy.deepcopy(mirrors)
                changed[0]["tags"][field] = "wrong"
                with self.subTest(owner=owner, field=field), self.assertRaises(ValueError):
                    validate_movement_transport_mirrors(journal, changed)
            for invalid in (0, -1, True, "1", float("nan"), float("inf")):
                with self.subTest(owner=owner, invalid=invalid), self.assertRaises(ValueError):
                    validate_movement_transport_mirrors(journal, [mirrors[0] | {"value": invalid}, mirrors[1]])
        validate_movement_transport_mirrors(empty_ledger(), [])

    def test_calibration_or_physical_bytes_cannot_supply_a_different_http_journal(self):
        journal = ledger("host")
        changed = self.transport_mirrors(journal)
        changed[0]["tags"]["transaction_purpose"] = "economy_calibration"
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(journal, changed)
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(journal, [{"domain": "moe_overlay_residency",
                "name": "placement_published_payload_bytes", "value": 999999999}])

    @staticmethod
    def movement_history(authority, transactions):
        """Repeat real closed-cycle fixture geometry with distinct exact epochs."""
        history = empty_ledger()
        for transaction in transactions:
            wave = ledger(authority, transaction=transaction, epoch=transaction + 1)
            for name in ("edges", "economy", "host_admissions"):
                history[name].extend(wave[name])
        return history

    def test_transport_history_rejects_missing_interior_wave_with_equal_bounds_and_totals(self):
        """First/last IDs, cardinality and bytes cannot certify missing history."""
        for authority in ("host",):
            first = 2**54 + 1
            journal = self.movement_history(authority, (first, first + 2, first + 4))
            unrelated = self.movement_history(authority, (first, first + 3, first + 4))
            expected = self.transport_mirrors(journal)
            changed = self.transport_mirrors(unrelated)
            for field in ("transaction", "candidate_epoch"):
                expected_ids = [int(row["tags"][field]) for row in expected]
                changed_ids = [int(row["tags"][field]) for row in changed]
                self.assertEqual((min(expected_ids), max(expected_ids), len(expected_ids)),
                                 (min(changed_ids), max(changed_ids), len(changed_ids)))
            self.assertEqual(sum(row["value"] for row in expected), sum(row["value"] for row in changed))
            self.assertEqual(sum(edge["estimated_weight_bytes"] for edge in journal["edges"]),
                             sum(edge["estimated_weight_bytes"] for edge in unrelated["edges"]))
            with self.subTest(authority=authority), self.assertRaisesRegex(ValueError, "matching completed transport"):
                validate_movement_transport_mirrors(journal, changed)

    def test_transport_history_accepts_later_terminal_waves_without_losing_http_cutoff(self):
        """Maintenance may drain after the last HTTP observation; that is valid."""
        for authority in ("host",):
            first = 2**54 + 1
            journal = self.movement_history(authority, (first, first + 1, first + 2))
            terminal = self.movement_history(authority, (first, first + 1, first + 2, first + 3, first + 4))
            mirrors = self.transport_mirrors(terminal)
            validate_movement_transport_mirrors(journal, mirrors)
            # Every rank can mirror the later history, but none of those rows
            # supplies the missing earlier command required by the HTTP reply.
            missing = [row for row in mirrors if row["tags"]["transaction"] != str(first + 1)]
            mirrored = missing + [row | {"rank": 1} for row in missing]
            with self.subTest(authority=authority), self.assertRaisesRegex(ValueError, "matching completed transport"):
                validate_movement_transport_mirrors(journal, mirrored)

    def test_transport_history_keeps_adjacent_uint64_publications_distinct(self):
        """Above double precision, neighboring integer identities must not alias."""
        for authority in ("host",):
            first = 2**54
            journal = self.movement_history(authority, (first, first + 1))
            mirrors = self.transport_mirrors(journal)
            self.assertEqual(float(first), float(first + 1))
            validate_movement_transport_mirrors(journal, mirrors)
            # The host protocol's candidate epoch is its transaction identity;
            # an optional diagnostic transaction tag is not an authority.
            for field in (("candidate_epoch",) if authority == "host" else ("transaction", "candidate_epoch")):
                changed = copy.deepcopy(mirrors)
                for row in changed:
                    row["tags"][field] = str(int(float(row["tags"][field])))
                with self.subTest(authority=authority, field=field), self.assertRaisesRegex(ValueError, "matching completed transport"):
                    validate_movement_transport_mirrors(journal, changed)

    def test_controller_bounded_transport_covers_every_rank_and_http_prefix(self):
        first = 2**54
        earlier = self.movement_history("device", (first, first + 1))
        complete = self.movement_history("device", (first, first + 1, first + 2, first + 3))
        records, terminal = controller_transport(complete)
        validate_movement_transport_mirrors(earlier, records, terminal)
        for rank in (0, 1):
            for name in _CONTROLLER_COUNTER_FIELDS:
                partial = [r for r in records if (r["rank"], r["name"]) != (rank, name)]
                with self.subTest(rank=rank, name=name), self.assertRaises(ValueError):
                    validate_controller_movement_transport(partial, terminal)
        for evidence in (MovementEvidence.NOT_APPLICABLE, MovementEvidence.REQUIRED):
            self.assertIsNone(validate_runtime_feature_policy(records, RuntimeFeaturePolicy(), evidence,
                terminal_movement=terminal))
            self.assertIsNotNone(validate_runtime_feature_policy([], RuntimeFeaturePolicy(), evidence,
                terminal_movement=terminal))
        for changed in (terminal[1:], terminal[:1], terminal * 2):
            with self.assertRaises(ValueError):
                validate_controller_movement_transport(records, changed)
        no_leader = copy.deepcopy(terminal)
        no_leader[0]["movement"]["economy"] = []
        with self.assertRaisesRegex(ValueError, "exactly one policy leader"):
            validate_controller_movement_transport(records, no_leader)
        duplicate_leader = copy.deepcopy(terminal)
        duplicate_leader[1]["movement"]["economy"] = duplicate_leader[0]["movement"]["economy"]
        with self.assertRaisesRegex(ValueError, "exactly one policy leader"):
            validate_controller_movement_transport(records, duplicate_leader)

    def test_controller_history_rejects_equal_bounds_and_totals_with_missing_interior_wave(self):
        first = 2**54
        expected = self.movement_history("device", (first, first + 2, first + 4))
        unrelated = self.movement_history("device", (first, first + 3, first + 4))
        records, terminal = controller_transport(expected)
        changed, wrong_terminal = controller_transport(unrelated)
        self.assertEqual(records[0]["sequence_minimum_words"], changed[0]["sequence_minimum_words"])
        self.assertEqual(records[0]["sequence_maximum_words"], changed[0]["sequence_maximum_words"])
        self.assertEqual([r["value"] for r in records], [r["value"] for r in changed])
        with self.assertRaisesRegex(ValueError, "exact completed transport sequence"):
            validate_controller_movement_transport(records, wrong_terminal)
        with self.assertRaisesRegex(ValueError, "matching completed transport prefix"):
            validate_movement_transport_mirrors(expected, changed, wrong_terminal)
        for field in ("transaction", "candidate_epoch"):
            changed = copy.deepcopy(terminal)
            for t in changed:
                t["device_publications"][1][field] += 1
            with self.assertRaises(ValueError):
                validate_controller_movement_transport(records, changed)
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(expected, self.transport_mirrors(expected), terminal)

    def test_controller_receipts_preserve_capacity_economy_edges_and_actual_bytes(self):
        records, terminal = controller_transport(ledger())
        for key in ("physical_payload_bytes", "command_count", "transaction", "candidate_epoch"):
            for value in (True, float(terminal[0]["device_publications"][0][key]), 0, 2**64):
                changed = copy.deepcopy(terminal)
                changed[0]["device_publications"][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_controller_movement_transport(records, changed)
        for key in _CONTROLLER_FIELDS:
            changed = copy.deepcopy(terminal)
            changed[0]["device_publications"][0]["controller"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_controller_movement_transport(records, changed)
        for key in ("cycle_index", "cycle_size", "source_device", "destination_device", "activation_count",
                    "source_world_rank", "destination_world_rank", "expert", "estimated_weight_bytes", "movement_axis"):
            changed = copy.deepcopy(terminal)
            value = changed[0]["movement"]["edges"][0][key]
            changed[0]["movement"]["edges"][0][key] = value + 1 if isinstance(value, int) else "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_controller_movement_transport(records, changed)
        for key, value in (("count", True), ("sequence_word_count", 34), ("sequence_digest_lo", -1),
                           ("sequence_digest_hi", 0), ("value", float("nan")), ("value", True)):
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_controller_movement_transport(changed, terminal)
        changed = copy.deepcopy(records)
        changed[0]["sequence_minimum_words"][0] = True
        with self.assertRaises(ValueError):
            validate_controller_movement_transport(changed, terminal)

    @classmethod
    def native_history(cls, transactions):
        """Keep native equations and edge geometry fixed while exact identities advance."""
        value = empty_ledger()
        for transaction in transactions:
            wave = cls.native_load_ledger()
            for name in ("edges", "economy"):
                for row in wave[name]:
                    row.update(transaction=transaction, candidate_epoch=transaction)
                value[name].extend(wave[name])
        return value

    @staticmethod
    def native_transport(journal):
        """Build bounded metadata fixtures; C++/Python encoding also has a native probe."""
        publications = [{"transaction": wave["transaction"], "candidate_epoch": wave["candidate_epoch"],
                         "command_count": wave["command_count"], "physical_payload_bytes": 6000}
                        for wave in journal["economy"]]
        words = [(p["transaction"], p["candidate_epoch"], p["command_count"], p["physical_payload_bytes"])
                 for p in publications]
        wave = native_movement_sequence(words)
        edges = native_movement_sequence(native_movement_edge_words(edge) for edge in journal["edges"])
        tags = {"policy_owner": "device", "policy": "native_load_spread", "encoding": "completed_wave_v1",
                "first_model_layer": "0", "layer_count": "8"}
        values = {"dynamic_movement_transactions": len(publications),
                  "dynamic_physical_bytes": sum(p["physical_payload_bytes"] for p in publications),
                  "dynamic_migration_edges": len(journal["edges"]),
                  "dynamic_migration_edge_identities": len(journal["edges"])}
        records = [{"domain": "moe_overlay_controller", "name": name, "kind": "counter", "value": value,
                    "rank": 0, "device": "CUDA:0", "phase": "maintenance", "tags": tags.copy(),
                    **(edges if name.endswith("identities") else wave)} for name, value in values.items()]
        terminal = [{"schema": 1, "scope": "terminal_model_lifetime", "rank": 0,
                     "movement": copy.deepcopy(journal), "device_publications": publications}]
        return records, terminal

    def test_native_completed_receipts_bind_bounded_counters_to_exact_terminal_history(self):
        """The same physical identity join survives removal of per-transaction keys."""
        journal = self.native_load_ledger()
        records, terminal = self.native_transport(journal)
        validate_movement_transport_mirrors(journal, records, terminal)
        for omitted in range(len(records)):
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                validate_movement_transport_mirrors(journal, records[:omitted] + records[omitted + 1:], terminal)
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(journal, records)
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(journal, records, terminal * 2)
        for name in ("transaction", "candidate_epoch", "command_count", "physical_payload_bytes"):
            for bad in (0, True, 2**64, float(2**54)):
                changed = copy.deepcopy(terminal)
                changed[0]["device_publications"][0][name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    validate_movement_transport_mirrors(journal, records, changed)

    def test_native_terminal_history_accepts_later_waves_but_rejects_missing_interior_identity(self):
        """Equal extrema, byte totals and row counts cannot replace one missing wave."""
        first = 2**54 + 1
        earlier = self.native_history((first, first + 2, first + 4))
        complete = self.native_history((first, first + 2, first + 4, first + 5))
        records, terminal = self.native_transport(complete)
        validate_movement_transport_mirrors(earlier, records, terminal)
        changed_records, changed_terminal = self.native_transport(self.native_history((first, first + 3, first + 4, first + 5)))
        for a, b in zip(records, changed_records):
            self.assertEqual(a["count"], b["count"])
            self.assertEqual(a["value"], b["value"])
            self.assertEqual(a["sequence_minimum_words"], b["sequence_minimum_words"])
            self.assertEqual(a["sequence_maximum_words"], b["sequence_maximum_words"])
        with self.assertRaisesRegex(ValueError, "matching completed transport sequence"):
            validate_movement_transport_mirrors(earlier, records, changed_terminal)
        with self.assertRaisesRegex(ValueError, "matching completed transport prefix"):
            validate_movement_transport_mirrors(earlier, changed_records, changed_terminal)

    def test_native_transport_preserves_adjacent_uint64_fields_and_actual_bytes(self):
        """Real copies cannot be reconstructed from padded/estimated edge sizes."""
        journal = self.native_history((2**54, 2**54 + 1))
        records, terminal = self.native_transport(journal)
        validate_movement_transport_mirrors(journal, records, terminal)
        for field, bad in (("transaction", 2**54), ("candidate_epoch", 2**54),
                           ("physical_payload_bytes", 2048)):
            changed = copy.deepcopy(terminal)
            changed[0]["device_publications"][1][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_movement_transport_mirrors(journal, records, changed)
        for index in range(len(records)):
            changed = copy.deepcopy(records)
            changed[index]["sequence_digest_lo"] ^= 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_movement_transport_mirrors(journal, changed, terminal)
        changed = copy.deepcopy(records)
        changed[-1]["sequence_minimum_words"][16] = True
        with self.assertRaises(ValueError):
            validate_movement_transport_mirrors(journal, changed, terminal)
        for field in ("activation_count", "estimated_weight_bytes", "source_world_rank"):
            changed = copy.deepcopy(terminal)
            changed[0]["movement"]["edges"][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_movement_transport_mirrors(journal, records, changed)

    def test_native_runtime_gate_requires_complete_transport_even_when_movement_is_optional(self):
        """Optional movement cannot turn malformed positive publication into a pass."""
        journal = self.native_load_ledger()
        records, terminal = self.native_transport(journal)
        for requirement in (MovementEvidence.REQUIRED, MovementEvidence.NOT_APPLICABLE):
            self.assertIsNone(validate_runtime_feature_policy(records, RuntimeFeaturePolicy(), requirement,
                terminal_movement=terminal))
            self.assertIsNotNone(validate_runtime_feature_policy(records, RuntimeFeaturePolicy(), requirement))
            self.assertIsNotNone(validate_runtime_feature_policy([], RuntimeFeaturePolicy(), requirement,
                terminal_movement=terminal))
            for omitted in range(len(records)):
                self.assertIsNotNone(validate_runtime_feature_policy(records[:omitted] + records[omitted + 1:],
                    RuntimeFeaturePolicy(), requirement, terminal_movement=terminal))
        self.assertIn("static", validate_runtime_feature_policy(records, RuntimeFeaturePolicy(),
            MovementEvidence.FORBIDDEN, terminal_movement=terminal))

    def check(self, value, requirement=MovementRequirement.REQUIRED, expected_topology=None):
        observer = MovementLedgerObserver(requirement)
        observer.observe(response(value, expected_topology))
        observer.finish()

    def test_host_and_device_publish_all_objective_axes(self):
        for authority in ("host", "device"):
            for axis in ("tier_residency", "participant_placement", "combined"):
                with self.subTest(authority=authority, axis=axis):
                    axes = ("tier_residency", "participant_placement") if axis == "combined" else (axis,)
                    self.check(ledger(authority, axis=axis), expected_topology=topology(authority, axes))

    def test_required_is_cohort_wide_and_exact_full_hits_need_no_new_wave(self):
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        observer.observe(response(empty_ledger()))
        observer.observe(response(ledger()))
        observer.observe(response(ledger()))
        observer.finish()

    def test_recomposed_physical_cycles_retain_per_command_objectives(self):
        """One physical component can contain differently scored logical moves."""
        for authority in ("host", "device"):
            changed = ledger(authority)
            changed["edges"][0]["movement_axis"] = "tier_residency"
            changed["edges"][1]["movement_axis"] = "participant_placement"
            self.check(changed, expected_topology=topology(authority))

    def test_static_and_not_applicable_require_empty_authoritative_arrays(self):
        for mode in (MovementRequirement.FORBIDDEN, MovementRequirement.NOT_APPLICABLE):
            self.check(empty_ledger(), mode)
            with self.assertRaises(ValueError):
                self.check(ledger(), mode)
        with self.assertRaises(ValueError):
            self.check(empty_ledger())

    def test_missing_malformed_and_truncated_snapshots_fail_closed(self):
        for value in (None, {}, [], False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(value)
        for name, value in (("schema", True), ("schema", 1), ("scope", "request"), ("complete", False),
                            ("discarded_edges", 1), ("discarded_economy_records", 1),
                            ("discarded_host_admission_records", 1), ("edges", {}),
                            ("economy", None), ("host_admissions", False)):
            changed = ledger()
            changed[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.check(changed)
        for name in empty_ledger():
            changed = empty_ledger()
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.check(changed, MovementRequirement.FORBIDDEN)

    def test_edge_fields_reject_untyped_or_impossible_identities(self):
        for name, value in (("authority", "none"), ("transaction", 0), ("candidate_epoch", True),
                            ("layer", -1), ("expert", 2**31), ("cycle_index", -1), ("cycle_size", 0),
                            ("movement_axis", "same_priority"), ("direction", "same_priority"),
                            ("source_participant", 1), ("source_device", ""), ("destination_device", "cuda"),
                            ("source_world_rank", -1), ("estimated_weight_bytes", 0),
                            ("activation_count", 2**64), ("blocking_inference", True)):
            changed = ledger()
            changed["edges"][0][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.check(changed)
        changed = ledger()
        for edge in changed["edges"]:
            edge["source_world_rank"] = edge["destination_world_rank"] = None
        self.check(changed)

    def test_cycle_coverage_identity_and_conservation_are_exact(self):
        mutations = (
            lambda x: x["edges"].pop(),
            lambda x: x["edges"].append(copy.deepcopy(x["edges"][0])),
            lambda x: x["edges"][1].update(cycle_index=1),
            lambda x: x["edges"][1].update(cycle_size=3),
            lambda x: x["edges"][1].update(layer=4),
            lambda x: x["edges"][1].update(destination_participant=2),
            lambda x: x["edges"][1].update(authority="host"),
        )
        for mutate in mutations:
            changed = ledger()
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check(changed)

    def test_every_wave_has_one_matching_profitable_economy_proof(self):
        mutations = (
            lambda x: x["economy"].clear(),
            lambda x: x["economy"].append(copy.deepcopy(x["economy"][0])),
            lambda x: x["economy"][0].update(command_count=3),
            lambda x: x["economy"][0].update(cycle_count=2),
            lambda x: x["economy"][0].update(transaction=3),
            lambda x: x["economy"][0].update(projected_net_benefit_ns=61),
            lambda x: x["economy"][0].update(projected_inference_interference_ns=100),
            lambda x: x["economy"][0].update(projected_transfer_and_repack_ns=2**64 - 1),
            lambda x: x["economy"][0].update(projected_service_gain_ns=100.0),
        )
        for mutate in mutations:
            changed = ledger()
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check(changed)
        changed = ledger()
        changed["economy"][0].update(projected_service_gain_ns=2**64-1, projected_net_benefit_ns=2**64-41)
        self.check(changed)

    def test_host_admission_requires_exact_classification_capacity_and_flags(self):
        for name, value in (("candidate_epoch", 3), ("cycle_capacity_kind", "unbounded"),
                            ("maximum_concurrent_cycles", 0), ("candidate_cycles", 2),
                            ("policy_eligible_cycles", 0), ("admitted_candidate_cycles", 2),
                            ("admitted_physical_cycles", 2), ("capacity_rejected_cycles", 1),
                            ("dependent_cohort_payoff_rejections", 1), ("physical_cycle_recomposition", True),
                            ("capacity_bounded", True), ("policy_bounded", True),
                            ("policy_eligible_axes", {"combined": 1}),
                            ("admitted_physical_axes", {"combined": 0, "tier_residency": 1, "participant_placement": 0})):
            changed = ledger("host")
            changed["host_admissions"][0][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.check(changed)
        changed = ledger("host")
        changed["host_admissions"].clear()
        with self.assertRaises(ValueError):
            self.check(changed)
        changed = ledger()
        changed["host_admissions"] = ledger("host")["host_admissions"]
        with self.assertRaises(ValueError):
            self.check(changed)

    def test_repeated_snapshots_are_immutable_prefixes_not_summable_counters(self):
        prior = ledger()
        extended = copy.deepcopy(prior)
        following = ledger(transaction=2, epoch=3)
        for name in ("edges", "economy", "host_admissions"):
            extended[name] += following[name]
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        observer.observe(response(prior))
        observer.observe(response(extended))
        observer.finish()
        with self.assertRaises(ValueError):
            observer.observe(response(prior))
        # Caller mutation cannot rewrite the observer's accepted history.
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        observer.observe(response(prior))
        prior["edges"][0]["activation_count"] += 1
        with self.assertRaises(ValueError):
            observer.observe(response(prior))

    def test_phase_api_rejects_untyped_requirement_and_empty_observation_sequence(self):
        with self.assertRaises(TypeError):
            MovementLedgerObserver("required")
        with self.assertRaises(ValueError):
            MovementLedgerObserver(MovementRequirement.FORBIDDEN).finish()

    def test_topology_requires_the_right_owner_and_every_available_axis(self):
        for authority in ("host", "device"):
            for axis in ("tier_residency", "participant_placement"):
                observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
                observer.observe(response(ledger(authority, axis=axis), topology(authority)))
                with self.subTest(authority=authority, axis=axis), self.assertRaisesRegex(ValueError, "every.*axis"):
                    observer.finish()
            with self.assertRaisesRegex(ValueError, "contradicts"):
                self.check(ledger(authority), expected_topology=topology("device" if authority == "host" else "host"))
            with self.assertRaisesRegex(ValueError, "contradicts"):
                self.check(ledger(authority), expected_topology=topology(authority, ("tier_residency",)))
        for invalid in (None, {}, {**topology(), "schema": True}, topology("none"),
                        topology("device", ("combined",)), topology("host", ("tier_residency", "tier_residency"))):
            observed = response(empty_ledger())
            observed["runtime_summary"]["expert_movement_topology"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MovementLedgerObserver(MovementRequirement.FORBIDDEN).observe(observed)
        self.check(empty_ledger(), MovementRequirement.NOT_APPLICABLE, topology("none", ()))
        self.check(empty_ledger(), MovementRequirement.FORBIDDEN, topology())

    def test_topology_cannot_change_even_before_any_movement_occurs(self):
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        initial = topology()
        observer.observe(response(empty_ledger(), initial))
        initial["authority"] = "host"
        with self.assertRaisesRegex(ValueError, "topology changed"):
            observer.observe(response(empty_ledger(), initial))

    @classmethod
    def pipeline_evidence(cls, devices=4, reverse=False, shared=False, mtp=False):
        """Independent real scopes may reuse a device and the same uint64 wave IDs."""
        movement = {"schema": 3, "scope": "model_lifetime", "complete": True, "stages": []}
        geometry = {"schema": 2, "scope": "model_lifetime", "authority": "pipeline",
                    "available_axes": ["participant_placement"], "stages": []}
        terminal = {"schema": 2, "scope": "terminal_model_lifetime", "rank": 0, "stages": []}
        records = []
        for stage in range(2):
            backend = "rocm" if (bool(stage) != reverse and not shared) else "cuda"
            end = 4 * stage + 4 + int(stage == 1 and mtp)
            identity = {"stage_index": stage, "first_layer": 4 * stage, "main_last_layer": 4 * stage + 4,
                        "routed_last_layer": end, "terminal": stage == 1,
                        "participants": [f"localhost:-1:{backend}:{i}" for i in range(devices // 2)]}
            journal = cls.native_history((2**54 + 1, 2**54 + 2))
            for edge in journal["edges"]:
                edge["layer"] = end - 1
                for endpoint in ("source", "destination"):
                    edge[endpoint + "_device"] = f'{"ROCm" if backend == "rocm" else "CUDA"}:{edge[endpoint + "_participant"]}'
            owned, final = cls.native_transport(journal)
            for record in owned:
                record["device"] = "ROCm:0" if backend == "rocm" else "CUDA:0"
                record["tags"].update(first_model_layer=str(stage * 4), layer_count=str(end - stage * 4))
            records.extend(owned)
            movement["stages"].append({"identity": copy.deepcopy(identity), "movement": journal})
            geometry["stages"].append({"identity": copy.deepcopy(identity),
                                       "topology": topology(axes=("participant_placement",))})
            del final[0]["rank"]
            terminal["stages"].append({"identity": copy.deepcopy(identity), "transport": final[0]})
        return movement, geometry, records, [terminal]

    def test_pipeline_four_and_eight_device_scopes_preserve_colliding_native_ids(self):
        """Both vendor orders, retained MTP and shared devices keep separate owners."""
        for devices in (4, 8):
            for reverse in (False, True):
                for shared in (False, True):
                    for mtp in (False, True):
                        with self.subTest(devices=devices, reverse=reverse, shared=shared, mtp=mtp):
                            journal, geometry, records, terminal = self.pipeline_evidence(devices, reverse, shared, mtp)
                            observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
                            observer.observe(response(journal, geometry))
                            observer.observe(response(journal, geometry))
                            observer.finish()
                            validate_movement_transport_mirrors(journal, records, terminal)
                            self.assertEqual(validate_controller_movement_transport(records, terminal), [])

    def test_pipeline_missing_reordered_and_foreign_scopes_fail_atomically(self):
        """Scope corruption must not replace the observer's last good snapshot."""
        journal, geometry, _, _ = self.pipeline_evidence()
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        observer.observe(response(journal, geometry))
        for mutation in range(10):
            changed = copy.deepcopy(journal)
            if mutation == 0:
                changed["stages"].pop()
            elif mutation == 1:
                changed["stages"].reverse()
            elif mutation == 2:
                changed["stages"][1]["identity"]["stage_index"] = 0
            elif mutation == 3:
                changed["stages"][1]["identity"]["first_layer"] = 3
            elif mutation == 4:
                changed["stages"][0]["identity"]["routed_last_layer"] = 5
            elif mutation == 5:
                changed["stages"][1]["movement"]["edges"][0]["layer"] = 3
            elif mutation == 6:
                changed["stages"][1]["movement"]["edges"][0]["destination_device"] = "CUDA:1"
            elif mutation == 7:
                changed["stages"][1]["identity"]["participants"].append("localhost:-1:rocm:0")
            elif mutation == 8:
                changed["stages"][0]["movement"] = copy.deepcopy(journal)
            else:
                changed["edges"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observer.observe(response(changed, geometry))
        observer.observe(response(journal, geometry))
        observer.finish()

    def test_pipeline_each_stage_requires_its_own_history_and_progress(self):
        """An advancing first stage cannot conceal lost or empty sibling history."""
        journal, geometry, _, _ = self.pipeline_evidence(8)
        observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
        observer.observe(response(journal, geometry))
        truncated = copy.deepcopy(journal)
        for name in ("edges", "economy", "host_admissions"):
            truncated["stages"][1]["movement"][name] = []
        with self.assertRaisesRegex(ValueError, "regressed"):
            observer.observe(response(truncated, geometry))
        fresh = MovementLedgerObserver(MovementRequirement.REQUIRED)
        fresh.observe(response(truncated, geometry))
        with self.assertRaisesRegex(ValueError, "stage axis"):
            fresh.finish()
        observer.finish()

    def test_pipeline_transport_cannot_borrow_a_sibling_receipt_or_counter(self):
        """Equal transaction IDs, counts and bytes do not authenticate another stage."""
        journal, geometry, records, terminal = self.pipeline_evidence(8, shared=True, mtp=True)
        validate_movement_transport_mirrors(journal, records, terminal)
        for mutation in range(9):
            changed_records, changed_terminal = copy.deepcopy(records), copy.deepcopy(terminal)
            if mutation == 0:
                changed_records.pop()
            elif mutation == 1:
                changed_records[4]["tags"]["first_model_layer"] = "0"
            elif mutation == 2:
                changed_records[4]["tags"]["layer_count"] = "4"
            elif mutation == 3:
                changed_terminal[0]["stages"].pop()
            elif mutation == 4:
                changed_terminal[0]["stages"][1]["transport"] = copy.deepcopy(changed_terminal[0]["stages"][0]["transport"])
            elif mutation == 5:
                changed_terminal[0]["stages"][1]["transport"]["device_publications"].pop()
            elif mutation == 6:
                changed_records[4]["tags"]["first_model_layer"] = "04"
            elif mutation == 7:
                changed_terminal[0]["stages"][1]["transport"]["rank"] = 0
            else:
                changed_records[4]["device"] = "ROCm:0"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_movement_transport_mirrors(journal, changed_records, changed_terminal)

    def test_pipeline_topology_cannot_certify_one_stage_with_another_axis(self):
        """Per-stage ownership survives the parent union of movement opportunities."""
        journal, geometry, _, _ = self.pipeline_evidence()
        geometry["stages"][1]["topology"]["available_axes"] = ["tier_residency"]
        geometry["available_axes"].append("tier_residency")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            MovementLedgerObserver(MovementRequirement.REQUIRED).observe(response(journal, geometry))


if __name__ == "__main__":
    unittest.main()
