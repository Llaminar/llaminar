#!/usr/bin/env python3
"""Device-free adversarial checks of authority-owned HTTP movement evidence.

Synthetic journals exercise schema, closed-cycle, economic and immutable-history
contracts only. They do not certify physical transfers or any model topology.
The canonical generation runner additionally retains real server path evidence.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from generation_movement_ledger import MovementLedgerObserver, MovementRequirement, validate_movement_transport_mirrors


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
        for owner in ("host", "device"):
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

    def test_native_completed_receipts_use_the_same_identity_join(self):
        """Native load units require no alternate transport-evidence protocol."""
        journal = self.native_load_ledger()
        mirrors = self.transport_mirrors(journal)
        for record in mirrors:
            record["tags"]["policy"] = "native_load_spread"
        validate_movement_transport_mirrors(journal, mirrors)
        with self.assertRaisesRegex(ValueError, "matching completed transport"):
            validate_movement_transport_mirrors(journal, mirrors[:1])
        mirrors[0]["tags"]["transaction"] = "999"
        with self.assertRaisesRegex(ValueError, "matching completed transport"):
            validate_movement_transport_mirrors(journal, mirrors)

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


if __name__ == "__main__":
    unittest.main()
