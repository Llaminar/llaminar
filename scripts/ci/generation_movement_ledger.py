"""Validate passive terminal movement journals from the production HTTP path.

The host/device placement owner authors these records. This consumer never
drives maintenance, reconstructs movement from PerfStats, estimates bandwidth,
or changes admission policy. It verifies the existing owner equations and
closed-cycle publication as evidence, then checks immutable model-lifetime
history across requests. The runner's frozen-plan geometry names the expected
authority and axes independently of the observed moves. Physical transport still
needs its independent production-path checks; this is not an image
certificate or a replacement placement authority.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from enum import Enum
import math
import re
from typing import Iterable, Mapping, Any


class MovementRequirement(str, Enum):
    """Exact matrix-owned obligation, independent of names and CLI defaults."""
    NOT_APPLICABLE = "not_applicable"
    FORBIDDEN = "forbidden"
    REQUIRED = "required"


class MovementAuthority(str, Enum):
    """Only actual placement owners can author completed movement evidence."""
    HOST = "host"
    DEVICE = "device"


class MovementAxis(str, Enum):
    """Logical objective is independent of the endpoints' physical direction."""
    TIER = "tier_residency"
    PARTICIPANT = "participant_placement"
    COMBINED = "combined"


_U64_MAX = 2**64 - 1
_I32_MAX = 2**31 - 1
_ARRAYS = ("edges", "economy", "host_admissions")
_ECONOMY_IDENTITY = {"authority", "transaction", "candidate_epoch", "command_count", "cycle_count", "policy"}
_TIME_ECONOMY = {"projected_service_gain_ns", "projected_transfer_and_repack_ns",
                 "projected_inference_interference_ns", "projected_net_benefit_ns"}
_LOAD_ECONOMY = {"accepted_spread_improvement", "pre_wave_spread", "post_wave_spread",
                 "pre_wave_total", "post_wave_total", "pre_participant_spread", "post_participant_spread",
                 "pre_participant_total", "post_participant_total", "requested_payload_slots",
                 "minimum_improvement_per_slot", "maximum_post_spread_per_mille", "ownership_swap_accepts"}


def _integer(record: dict, name: str, low: int = 0, high: int = _U64_MAX) -> int:
    """Preserve exact wire integers, rejecting bools, floats and overflow."""
    value = record.get(name)
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid movement integer: " + name)
    return value


def _flag(record: dict, name: str) -> bool:
    """Flags must be explicit JSON booleans, never truthy strings or counts."""
    if type(record.get(name)) is not bool:
        raise ValueError("invalid movement flag: " + name)
    return record[name]


def _wave(record: dict) -> tuple[MovementAuthority, int, int]:
    """The owner, transaction and candidate epoch jointly identify publication."""
    if not isinstance(record, dict):
        raise ValueError("movement record must be an object")
    return (MovementAuthority(record.get("authority")), _integer(record, "transaction", 1),
            _integer(record, "candidate_epoch", 1))


def _axes(record: dict, name: str, expected: int) -> Counter:
    """Verify exclusive owner-authored cycle buckets, not inferred objectives."""
    value = record.get(name)
    if not isinstance(value, dict):
        raise ValueError("missing movement axis counts: " + name)
    counts = Counter({axis.value: _integer(value, axis.value) for axis in MovementAxis})
    if sum(counts.values()) != expected:
        raise ValueError("movement axis counts do not match their total: " + name)
    return counts


def _validate_edge(edge: dict) -> tuple:
    """Check typed identity and physical direction without guessing intent."""
    wave = _wave(edge)
    layer = _integer(edge, "layer", 0, _I32_MAX)
    expert = _integer(edge, "expert", 0, _I32_MAX)
    _integer(edge, "cycle_index")
    _integer(edge, "cycle_size", 1)
    MovementAxis(edge.get("movement_axis"))
    source = _integer(edge, "source_participant", 0, _I32_MAX)
    destination = _integer(edge, "destination_participant", 0, _I32_MAX)
    if source == destination:
        raise ValueError("movement edge has no destination change")
    source_priority = _integer(edge, "source_priority", -2**31, _I32_MAX)
    destination_priority = _integer(edge, "destination_priority", -2**31, _I32_MAX)
    direction = ("promotion" if destination_priority < source_priority else
                 "demotion" if destination_priority > source_priority else "same_priority")
    if edge.get("direction") != direction:
        raise ValueError("movement direction disagrees with endpoint priorities")
    for side in ("source", "destination"):
        device = edge.get(side + "_device")
        if not isinstance(device, str) or not re.fullmatch(r"CPU|(?:CUDA|ROCm):[0-9]+", device):
            raise ValueError("invalid movement device identity: " + side)
        rank = side + "_world_rank"
        if rank not in edge:
            raise ValueError("movement rank must be an integer or explicit null")
        if edge[rank] is not None:
            _integer(edge, rank, 0, _I32_MAX)
    # This remains an estimated payload size, not a measured transport counter.
    _integer(edge, "estimated_weight_bytes", 1)
    _integer(edge, "activation_count")
    if _flag(edge, "blocking_inference"):
        raise ValueError("generation movement blocked inference")
    return wave, layer, expert, source, destination


def _validate_economy(record: dict, edges: list[dict], cycles: dict) -> None:
    """Require the exact profitable owner equation for the same completed wave."""
    if (_integer(record, "command_count", 1) != len(edges)
            or _integer(record, "cycle_count", 1) != len(cycles)):
        raise ValueError("movement economy does not cover every published edge/cycle")
    policy = record.get("policy")
    if policy == "native_load_spread":
        if set(record) != _ECONOMY_IDENTITY | _LOAD_ECONOMY:
            raise ValueError("native load proof has missing, unknown or mixed-unit fields")
        _validate_native_load_spread(record, edges)
        return
    if policy != "time_ns":
        raise ValueError("movement economy has no supported unit-qualified policy")
    if set(record) != _ECONOMY_IDENTITY | _TIME_ECONOMY:
        raise ValueError("time economy has missing, unknown or mixed-unit fields")
    gain = _integer(record, "projected_service_gain_ns", 1)
    transfer = _integer(record, "projected_transfer_and_repack_ns")
    interference = _integer(record, "projected_inference_interference_ns")
    benefit = _integer(record, "projected_net_benefit_ns", 1)
    cost = transfer + interference
    if cost > _U64_MAX or gain <= cost or benefit != gain - cost:
        raise ValueError("movement economy has no exact positive net benefit")


def _validate_native_load_spread(record: dict, edges: list[dict]) -> None:
    """Check the sealed native policy in routed-work units, never invented ns.

    The device policy owns admission. These independent equations only validate
    its terminal proof. Participant totals can worsen when cancelling layer
    skews improve; conservation and the serial-layer critical path remain gates.
    """
    if record["authority"] != "device" or any(name.endswith("_ns") for name in record):
        raise ValueError("native load proof has a wrong authority or mixed units")
    if any(edge["movement_axis"] != "participant_placement"
           or edge["direction"] != "same_priority" for edge in edges):
        raise ValueError("native load proof cannot certify tier movement")
    swaps = _integer(record, "ownership_swap_accepts", 1, 2**32 - 1)
    slots = _integer(record, "requested_payload_slots", 1, 2**32 - 1)
    floor = _integer(record, "minimum_improvement_per_slot", 0, 2**32 - 1)
    ceiling = _integer(record, "maximum_post_spread_per_mille", 0, 2**32 - 1)
    gain = _integer(record, "accepted_spread_improvement", 1)
    pre = _integer(record, "pre_wave_spread")
    post = _integer(record, "post_wave_spread")
    total = _integer(record, "pre_wave_total", 1)
    post_total = _integer(record, "post_wave_total", 1)
    participant_total = _integer(record, "pre_participant_total", 1)
    post_participant_total = _integer(record, "post_participant_total", 1)
    _integer(record, "pre_participant_spread")
    _integer(record, "post_participant_spread")
    if (swaps != record["cycle_count"] or swaps * 2 != len(edges)
            or gain < slots * floor or pre <= post or total != post_total
            or participant_total != post_participant_total):
        raise ValueError("native load proof violates swap, conservation or improvement policy")
    if ceiling and (post > _U64_MAX // 1000 or post * 1000 > total * ceiling):
        raise ValueError("native load proof exceeds its configured spread ceiling")


def _cycle_axis(edges: list[dict]) -> str:
    """Union recorded objectives without inferring them from physical direction.

    Device publication can recompose independently scored logical cycles into
    a larger closed physical component. Its commands retain their original
    axes; requiring identical labels would reject valid combined movement.
    """
    axes = {MovementAxis(edge["movement_axis"]) for edge in edges}
    return (next(iter(axes)).value if len(axes) == 1 else MovementAxis.COMBINED.value)


def _validate_admission(record: dict, cycles: dict) -> None:
    """Recheck existing host capacity/classification equations without planning."""
    authority, transaction, epoch = _wave(record)
    if authority is not MovementAuthority.HOST or epoch != transaction:
        raise ValueError("invalid host movement admission identity")
    if record.get("cycle_capacity_kind") != "bounded":
        raise ValueError("production host movement requires bounded physical capacity")
    maximum = _integer(record, "maximum_concurrent_cycles", 1)
    candidates = _integer(record, "candidate_cycles", 1)
    eligible = _integer(record, "policy_eligible_cycles")
    admitted = _integer(record, "admitted_candidate_cycles", 1)
    physical = _integer(record, "admitted_physical_cycles", 1)
    _axes(record, "policy_eligible_axes", eligible)
    _axes(record, "admitted_candidate_axes", admitted)
    physical_axes = _axes(record, "admitted_physical_axes", physical)
    individual = _integer(record, "individual_policy_rejected_cycles")
    dependent = _integer(record, "dependent_payoff_rejected_cycles")
    capacity = _integer(record, "capacity_rejected_cycles")
    participant = _integer(record, "participant_axis_budget_rejected_cycles")
    cohorts = _integer(record, "dependent_cohort_candidates")
    rejected_cohorts = _integer(record, "dependent_cohort_payoff_rejections")
    if (physical > maximum or physical != len(cycles) or candidates != individual + eligible
            or eligible != admitted + dependent + capacity + participant or rejected_cohorts > cohorts
            or physical_axes != Counter(_cycle_axis(edges) for edges in cycles.values())):
        raise ValueError("host movement admission does not classify the completed physical cycles")
    if (_flag(record, "physical_cycle_recomposition") != (physical != admitted)
            or _flag(record, "capacity_bounded") != (capacity != 0)
            or _flag(record, "policy_bounded") != bool(individual or dependent or participant)):
        raise ValueError("host movement admission flags disagree with owner accounting")


def _index_unique(records: list, label: str) -> dict:
    """Reject duplicated authority proofs even when their payloads are equal."""
    indexed = {}
    for record in records:
        key = _wave(record)
        if key in indexed:
            raise ValueError("duplicate movement " + label + " record")
        indexed[key] = record
    return indexed


def validate_movement_ledger(value: dict) -> None:
    """Validate a complete schema-v2 owner snapshot, including empty snapshots."""
    if (not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 2
            or value.get("scope") != "model_lifetime" or value.get("complete") is not True):
        raise ValueError("missing, unsupported or incomplete movement ledger")
    for name in ("discarded_edges", "discarded_economy_records", "discarded_host_admission_records"):
        if _integer(value, name) != 0:
            raise ValueError("movement ledger discarded evidence")
    for name in _ARRAYS:
        if not isinstance(value.get(name), list):
            raise ValueError("missing movement ledger array: " + name)
    waves, identities = defaultdict(list), set()
    for edge in value["edges"]:
        identity = _validate_edge(edge)
        if identity in identities:
            raise ValueError("duplicate completed movement edge")
        identities.add(identity)
        waves[identity[0]].append(edge)
    economy = _index_unique(value["economy"], "economy")
    admissions = _index_unique(value["host_admissions"], "host admission")
    if set(economy) != set(waves):
        raise ValueError("movement edges and economy waves differ")
    if len({wave[0] for wave in waves}) > 1:
        raise ValueError("one model-lifetime movement ledger has conflicting authorities")
    # One transaction cannot be relabelled under several candidate epochs.
    if len({wave[:2] for wave in waves}) != len(waves):
        raise ValueError("movement transaction has conflicting candidate epochs")
    host_waves = {wave for wave in waves if wave[0] is MovementAuthority.HOST}
    if set(admissions) != host_waves:
        raise ValueError("host movement admission does not cover exactly the host-owned waves")
    for wave, edges in waves.items():
        cycles = defaultdict(list)
        for edge in edges:
            cycles[edge["cycle_index"]].append(edge)
        for members in cycles.values():
            if (len({(edge["layer"], edge["cycle_size"]) for edge in members}) != 1
                    or len(members) != members[0]["cycle_size"]
                    or Counter(edge["source_participant"] for edge in members)
                    != Counter(edge["destination_participant"] for edge in members)):
                raise ValueError("movement cycle is incomplete, inconsistent or not closed")
        _validate_economy(economy[wave], edges, cycles)
        if wave in admissions:
            _validate_admission(admissions[wave], cycles)


def validate_movement_topology(value: dict) -> frozenset[MovementAxis]:
    """Read immutable geometry; no expert quota, placement or cost is recomputed."""
    if (not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1
            or value.get("scope") != "model_lifetime" or value.get("authority") not in ("none", "host", "device")):
        raise ValueError("missing or invalid movement topology")
    axes = value.get("available_axes")
    if (not isinstance(axes, list) or any(axis not in ("tier_residency", "participant_placement") for axis in axes)
            or len(set(axes)) != len(axes) or (value["authority"] == "none" and axes)):
        raise ValueError("invalid movement topology axes")
    return frozenset(MovementAxis(axis) for axis in axes)


def validate_movement_transport_mirrors(ledger: dict, records: Iterable[Mapping[str, Any]]) -> None:
    """Bind HTTP-owned edges to the independently checked transport publication.

    The server harness first authenticates completed bytes and matched host
    publication sequences or device transaction trios. This additional join
    prevents that physical evidence from certifying a different HTTP journal.
    It never reconstructs a placement from PerfStats or treats estimated expert
    sizes as transferred bytes. Rank mirrors collapse by exact edge identity;
    they cannot supply another missing expert. Final process diagnostics may
    contain later legitimate waves than the last request snapshot, so the
    immutable HTTP journal must be covered, not equal the later full history.
    """
    validate_movement_ledger(ledger)
    fields = ("candidate_epoch", "layer", "expert", "source_participant", "destination_participant",
              "source_priority", "destination_priority", "source_device", "destination_device",
              "movement_axis", "direction")
    observed = set()
    for record in records:
        family = (record.get("domain"), record.get("name"))
        if family == ("moe_overlay_residency", "expert_migration_edges"):
            authority = "host"
        elif family == ("moe_overlay_controller", "dynamic_migration_edges"):
            authority = "device"
        else:
            continue
        value = record.get("value")
        if type(value) not in (int, float) or not 0 < value < 2**64 or not math.isfinite(value):
            continue
        tags = record.get("tags")
        if not isinstance(tags, dict) or tags.get("policy_owner") != authority:
            continue
        if authority == "host" and tags.get("transaction_purpose") != "live_placement_change":
            continue
        transaction = tags.get("candidate_epoch" if authority == "host" else "transaction")
        projected = (transaction, *(tags.get(field) for field in fields))
        if all(isinstance(value, str) for value in projected):
            observed.add((authority, *projected))
    for edge in ledger["edges"]:
        identity = (edge["authority"], str(edge["transaction"]), *(str(edge[field]) for field in fields))
        if identity not in observed:
            raise ValueError("HTTP movement edge lacks matching completed transport publication: "
                             f"{edge['authority']} transaction={edge['transaction']} "
                             f"epoch={edge['candidate_epoch']} layer={edge['layer']} expert={edge['expert']}")


def _observed_axes(edges: list[dict]) -> set[MovementAxis]:
    """Unpack owner-authored combined objectives, independently of direction."""
    observed = {MovementAxis(edge["movement_axis"]) for edge in edges}
    if MovementAxis.COMBINED in observed:
        observed.remove(MovementAxis.COMBINED)
        observed.update((MovementAxis.TIER, MovementAxis.PARTICIPANT))
    return observed


class MovementLedgerObserver:
    """Validate a request sequence without adding together overlapping snapshots.

No counters drive inference and no optional telemetry supplies missing records.
The observer owns a private copy of its previous diagnostic snapshot so later
caller mutation cannot change accepted history. Required movement is checked
at cohort completion, not demanded gratuitously on every restored request.
"""

    def __init__(self, requirement: MovementRequirement):
        """Require explicit matrix policy; strings cannot silently select defaults."""
        if not isinstance(requirement, MovementRequirement):
            raise TypeError("movement requirement must be typed")
        self._requirement = requirement
        self._previous = None
        self._topology = None

    def observe(self, response: dict) -> None:
        """Accept one whole immutable snapshot or fail before changing history."""
        summary = response.get("runtime_summary") if isinstance(response, dict) else None
        if not isinstance(summary, dict) or type(summary.get("schema")) is not int or summary["schema"] != 1:
            raise ValueError("missing terminal runtime summary for movement")
        current = summary.get("expert_movement")
        validate_movement_ledger(current)
        if self._requirement is not MovementRequirement.REQUIRED and any(current[name] for name in _ARRAYS):
            raise ValueError("movement occurred in a forbidden or non-applicable cell")
        topology = summary.get("expert_movement_topology")
        available = validate_movement_topology(topology)
        if (any(edge["authority"] != topology["authority"] for edge in current["edges"])
                or not _observed_axes(current["edges"]) <= available):
            raise ValueError("movement authority or objective contradicts the frozen topology")
        if self._topology is not None and topology != self._topology:
            raise ValueError("movement topology changed within a model lifetime")
        if self._previous is not None:
            for name in _ARRAYS:
                previous = self._previous[name]
                if current[name][:len(previous)] != previous:
                    raise ValueError("movement ledger history regressed or changed: " + name)
        self._previous = copy.deepcopy(current)
        self._topology = copy.deepcopy(topology)

    def finish(self) -> None:
        """Require a complete observed cohort and positive movement when selected."""
        if self._previous is None:
            raise ValueError("no movement ledger observations")
        if self._requirement is MovementRequirement.REQUIRED:
            required = validate_movement_topology(self._topology)
            if not required or not required <= _observed_axes(self._previous["edges"]):
                raise ValueError("required movement does not cover every frozen-topology axis")
