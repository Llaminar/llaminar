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


def _pipeline_entries(value: dict, field: str) -> list[dict]:
    """Authenticate complete ordered stage namespaces without renumbering waves."""
    entries = value.get("stages")
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing pipeline movement stages")
    expected_first = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"identity", field}:
            raise ValueError("invalid pipeline movement stage fields")
        identity, child = entry["identity"], entry[field]
        if (not isinstance(identity, dict) or set(identity) !=
                {"stage_index", "first_layer", "main_last_layer", "routed_last_layer", "participants", "terminal"}
                or not isinstance(child, dict) or "stages" in child):
            raise ValueError("invalid or nested pipeline movement namespace")
        first = _integer(identity, "first_layer", 0, _I32_MAX)
        main_end = _integer(identity, "main_last_layer", first + 1, _I32_MAX)
        routed_end = _integer(identity, "routed_last_layer", main_end, _I32_MAX)
        participants = identity["participants"]
        if (_integer(identity, "stage_index", 0, _I32_MAX) != index or first != expected_first
                or type(identity["terminal"]) is not bool or identity["terminal"] != (index + 1 == len(entries))
                or (index + 1 < len(entries) and routed_end != main_end)
                or not isinstance(participants, list) or not participants
                or any(not isinstance(device, str) or not re.fullmatch(r".+:(?:cuda|rocm|cpu):[0-9]+", device) for device in participants)
                or len(set(participants)) != len(participants)):
            raise ValueError("missing, overlapping or foreign pipeline movement scope")
        expected_first = main_end
    return entries


def _scoped_edges(identity: dict, journal: dict) -> None:
    """Bind global layers and exact participant coordinates to a frozen stage."""
    devices = [":".join(value.split(":")[-2:]).lower() for value in identity["participants"]]
    for edge in journal["edges"]:
        if not identity["first_layer"] <= edge["layer"] < identity["routed_last_layer"]:
            raise ValueError("movement edge escaped its stage layer interval")
        for endpoint in ("source", "destination"):
            participant = edge[endpoint + "_participant"]
            if not 0 <= participant < len(devices) or edge[endpoint + "_device"].lower() != devices[participant]:
                raise ValueError("movement edge escaped its stage participant domain")


def _ledger_leaves(value: dict) -> list[tuple[dict | None, dict]]:
    """Return owner values paired with their namespace, preserving every identity."""
    return ([(entry["identity"], entry["movement"]) for entry in _pipeline_entries(value, "movement")]
            if value["schema"] == 3 else [(None, value)])


def _terminal_scopes(documents: Iterable[Mapping[str, Any]]) -> list[tuple[dict | None, dict]]:
    """Project complete terminal documents while retaining process and stage scope."""
    result, ranks = [], set()
    for document in documents:
        if (not isinstance(document, dict) or type(document.get("schema")) is not int
                or document["schema"] not in (1, 2) or document.get("scope") != "terminal_model_lifetime"):
            raise ValueError("invalid terminal movement transport artifact")
        rank = _integer(document, "rank", 0, _I32_MAX)
        if rank in ranks:
            raise ValueError("duplicate terminal movement rank")
        ranks.add(rank)
        if document["schema"] == 1:
            if "stages" in document:
                raise ValueError("unversioned pipeline terminal movement")
            result.append((None, document))
        else:
            if set(document) != {"schema", "scope", "rank", "stages"}:
                raise ValueError("pipeline terminal movement contains unscoped records")
            for entry in _pipeline_entries(document, "transport"):
                child = entry["transport"]
                if type(child.get("schema")) is not int or child["schema"] != 1 or child.get("scope") != "terminal_model_lifetime" or "rank" in child:
                    raise ValueError("invalid terminal pipeline stage transport")
                result.append((entry["identity"], {**child, "rank": rank}))
    return result


def validate_terminal_movement_metadata(document: dict, *, expected_rank: int) -> None:
    """Authenticate a sidecar's rank and stage envelopes before aggregation.

    Collection and transport validation share one stage-scope parser. The
    collector preserves the original document; it does not flatten independent
    transaction namespaces or infer completed moves from counter totals. Full
    journal and physical transport joins remain the movement observer's job.
    """
    scopes = _terminal_scopes((document,))
    if type(expected_rank) is not int or document["rank"] != expected_rank:
        raise ValueError("terminal movement rank disagrees with its artifact")
    for _, terminal in scopes:
        if (not isinstance(terminal.get("movement"), dict)
                or not isinstance(terminal.get("device_publications"), list)):
            raise ValueError("invalid terminal movement journal or publications")


def native_movement_scope(record: Mapping[str, Any]) -> tuple[int, int, int]:
    """Read frozen controller geometry from bounded metadata counter keys."""
    tags = record.get("tags")
    base = {"policy_owner": "device", "policy": "native_load_spread", "encoding": "completed_wave_v1"}
    if (not isinstance(tags, dict) or set(tags) != set(base) | {"first_model_layer", "layer_count"}
            or any(tags.get(key) != value for key, value in base.items())):
        raise ValueError("invalid native movement scope tags")
    values = []
    for name in ("first_model_layer", "layer_count"):
        value = tags[name]
        if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise ValueError("invalid native movement layer scope")
        values.append(int(value))
    if values[1] <= 0 or sum(values) > _I32_MAX:
        raise ValueError("invalid native movement layer extent")
    return (_integer(record, "rank", 0, _I32_MAX), *values)


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


def validate_movement_ledger(value: dict, *, device_follower: bool = False) -> None:
    """Validate a complete schema-v2 owner snapshot, including empty snapshots."""
    if isinstance(value, dict) and type(value.get("schema")) is int and value["schema"] == 3:
        if set(value) != {"schema", "scope", "complete", "stages"} or value.get("scope") != "model_lifetime" or value.get("complete") is not True:
            raise ValueError("invalid pipeline movement ledger envelope")
        for entry in _pipeline_entries(value, "movement"):
            validate_movement_ledger(entry["movement"], device_follower=device_follower)
            _scoped_edges(entry["identity"], entry["movement"])
        return
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
    if device_follower and (economy or any(wave[0] is not MovementAuthority.DEVICE for wave in waves)):
        raise ValueError("device follower cannot claim policy ownership or host movement")
    if not device_follower and set(economy) != set(waves):
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
        if not device_follower:
            _validate_economy(economy[wave], edges, cycles)
        if wave in admissions:
            _validate_admission(admissions[wave], cycles)


def validate_movement_topology(value: dict) -> frozenset[MovementAxis]:
    """Read immutable geometry; no expert quota, placement or cost is recomputed."""
    if isinstance(value, dict) and type(value.get("schema")) is int and value["schema"] == 2:
        if (set(value) != {"schema", "scope", "authority", "available_axes", "stages"}
                or value.get("scope") != "model_lifetime" or value.get("authority") != "pipeline"):
            raise ValueError("invalid pipeline movement topology envelope")
        axes = frozenset().union(*(validate_movement_topology(entry["topology"])
                                  for entry in _pipeline_entries(value, "topology")))
        if (not isinstance(value["available_axes"], list) or len(value["available_axes"]) != len(axes)
                or set(value["available_axes"]) != {axis.value for axis in axes}):
            raise ValueError("pipeline topology axes disagree with its stages")
        return axes
    if (not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1
            or value.get("scope") != "model_lifetime" or value.get("authority") not in ("none", "host", "device")):
        raise ValueError("missing or invalid movement topology")
    axes = value.get("available_axes")
    if (not isinstance(axes, list) or any(axis not in ("tier_residency", "participant_placement") for axis in axes)
            or len(set(axes)) != len(axes) or (value["authority"] == "none" and axes)):
        raise ValueError("invalid movement topology axes")
    return frozenset(MovementAxis(axis) for axis in axes)


def native_movement_sequence(steps: Iterable[Iterable[int]]) -> dict:
    """Independently encode the collector's delimited uint64 metadata witness.

    Only transaction metadata enters these two lanes. The encoding uses explicit
    little-endian bytes, so neither native struct padding nor host endian enters
    the artifact. No model, tensor or cache payload is read.
    """
    lo, hi, count, word_count = 14695981039346656037, 7809847782465536322, 0, 0
    minimum, maximum = [], []

    def mix(value: int, word: int, prime: int) -> int:
        for shift in range(0, 64, 8):
            value = ((value ^ ((word >> shift) & 255)) * prime) & _U64_MAX
        return value

    for values in steps:
        words = tuple(values)
        if (not words or any(type(word) is not int or not 0 <= word <= _U64_MAX for word in words)
                or (count and len(words) != len(minimum))):
            raise ValueError("invalid native movement sequence words")
        lo = mix(lo, 0x6c6c616d696e6172 ^ len(words), 1099511628211)
        hi = mix(hi, (~0x6c6c616d696e6172 & _U64_MAX) ^ len(words), 14029467366897019727)
        for word in words:
            lo = mix(lo, word, 1099511628211)
            hi = mix(hi, ~word & _U64_MAX, 14029467366897019727)
        lo = mix(lo, 0x73657175656e6365, 1099511628211)
        hi = mix(hi, ~0x73657175656e6365 & _U64_MAX, 14029467366897019727)
        minimum = [min(a, b) for a, b in zip(minimum, words)] if count else list(words)
        maximum = [max(a, b) for a, b in zip(maximum, words)] if count else list(words)
        count += 1
        word_count += len(words)
    return {"count": count, "sequence_word_count": word_count,
            "sequence_digest_lo": lo if count else 0, "sequence_digest_hi": hi if count else 0,
            "sequence_minimum_words": minimum, "sequence_maximum_words": maximum}


def native_movement_edge_words(edge: dict) -> tuple[int, ...]:
    """Project the complete semantic native edge identity into its versioned words."""
    _validate_edge(edge)
    if (edge["authority"] != "device" or edge["direction"] != "same_priority"
            or edge["movement_axis"] != "participant_placement"):
        raise ValueError("native movement witness contains a different authority or objective")
    source, destination = edge["source_device"].split(":"), edge["destination_device"].split(":")
    if len(source) != 2 or len(destination) != 2 or source[0] not in {"CUDA", "ROCm"} or source[0] != destination[0]:
        raise ValueError("native movement witness contains a heterogeneous device edge")
    source_rank, destination_rank = edge["source_world_rank"], edge["destination_world_rank"]
    backend = 1 if source[0] == "CUDA" else 2
    return (edge["transaction"], edge["candidate_epoch"], edge["layer"], edge["expert"],
            edge["cycle_index"], edge["cycle_size"], 0, 1,
            edge["source_participant"], edge["destination_participant"],
            edge["source_priority"] & _U64_MAX, edge["destination_priority"] & _U64_MAX,
            backend, int(source[1]), backend, int(destination[1]),
            int(source_rank is not None), source_rank if source_rank is not None else 0,
            int(destination_rank is not None), destination_rank if destination_rank is not None else 0,
            edge["estimated_weight_bytes"], edge["activation_count"], 0)


def _validate_native_movement_transport_owner(records: Iterable[Mapping[str, Any]], terminal_movement: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Bind every native counter/edge witness to its exact final owner journal.

    The terminal journal can extend the last HTTP response. It supplies actual
    per-wave copied bytes independently of edge size estimates, while the
    ordered counter witnesses reject missing interior waves and reordered edges.
    One native root publishes each rank-local domain's archive; duplicate scope
    claims cannot certify a missing participant or a different transaction.
    """
    names = {"dynamic_movement_transactions", "dynamic_physical_bytes", "dynamic_migration_edges",
             "dynamic_migration_edge_identities"}
    scopes = {}
    for record in records:
        tags = record.get("tags") or {}
        if (record.get("domain") != "moe_overlay_controller" or record.get("name") not in names
                or tags.get("policy") != "native_load_spread"):
            continue
        rank = _integer(record, "rank", 0, _I32_MAX)
        if (record.get("phase") != "maintenance" or not isinstance(record.get("device"), str)
                or not record["device"]):
            raise ValueError("invalid bounded native transport publication scope")
        native_movement_scope(record)
        scope = scopes.setdefault(rank, {"device": record["device"], "records": {}})
        if scope["device"] != record["device"] or record["name"] in scope["records"]:
            raise ValueError("duplicate or conflicting native transport publication scope")
        scope["records"][record["name"]] = record

    validated, seen_ranks, matched_ranks = [], set(), set()
    for terminal in terminal_movement:
        if (not isinstance(terminal, dict) or type(terminal.get("schema")) is not int
                or terminal["schema"] != 1 or terminal.get("scope") != "terminal_model_lifetime"):
            raise ValueError("invalid terminal native movement transport artifact")
        rank = _integer(terminal, "rank", 0, _I32_MAX)
        if rank in seen_ranks:
            raise ValueError("duplicate terminal movement rank")
        seen_ranks.add(rank)
        journal = terminal.get("movement")
        if not isinstance(journal, dict) or not isinstance(journal.get("economy"), list):
            raise ValueError("missing terminal movement owner journal")
        native = any(item.get("policy") == "native_load_spread" for item in journal["economy"] if isinstance(item, dict))
        if not native and rank not in scopes:
            continue
        validate_movement_ledger(journal)
        if not native or any(item["policy"] != "native_load_spread" for item in journal["economy"]):
            raise ValueError("native transport scope disagrees with the terminal policy owner")
        publications = terminal.get("device_publications")
        if not isinstance(publications, list) or len(publications) != len(journal["economy"]):
            raise ValueError("terminal native movement lost physical publication history")
        wave_steps, expected_order = [], []
        for publication, economy in zip(publications, journal["economy"]):
            fields = ("transaction", "candidate_epoch", "command_count", "physical_payload_bytes")
            if not isinstance(publication, dict) or set(publication) != set(fields):
                raise ValueError("invalid terminal native physical publication fields")
            words = tuple(_integer(publication, field, 1) for field in fields)
            if words[:3] != tuple(economy[field] for field in fields[:3]):
                raise ValueError("terminal native physical publication disagrees with its economy wave")
            wave_steps.append(words)
            expected_order.extend([(words[0], words[1])] * words[2])
        if expected_order != [(edge["transaction"], edge["candidate_epoch"]) for edge in journal["edges"]]:
            raise ValueError("terminal native physical publication disagrees with edge order")
        scope = scopes.get(rank, {}).get("records", {})
        if set(scope) != names:
            raise ValueError("native movement lacks matching completed transport counter families")
        wave_witness = native_movement_sequence(wave_steps)
        edge_witness = native_movement_sequence(native_movement_edge_words(edge) for edge in journal["edges"])
        # Reproduce counter addition in publication order. IDs and actual per-wave
        # bytes remain uint64 in the witness even above double's exact range.
        totals = {name: 0.0 for name in names}
        for _, _, commands, actual_bytes in wave_steps:
            totals["dynamic_movement_transactions"] += 1.0
            totals["dynamic_physical_bytes"] += float(actual_bytes)
            totals["dynamic_migration_edges"] += float(commands)
        totals["dynamic_migration_edge_identities"] = float(len(journal["edges"]))
        for name, record in scope.items():
            expected = edge_witness if name == "dynamic_migration_edge_identities" else wave_witness
            if (record.get("kind") != "counter" or type(record.get("value")) not in (int, float)
                    or record["value"] != totals[name]
                    or any(type(record.get(field)) is not type(value) or record[field] != value
                           or (isinstance(value, list) and any(type(word) is not int for word in record[field]))
                           for field, value in expected.items())):
                raise ValueError("native movement lacks matching completed transport sequence: " + name)
        matched_ranks.add(rank)
        validated.append(journal)
    if matched_ranks != set(scopes):
        raise ValueError("native movement counter has no matching terminal transport journal")
    return validated


def validate_native_movement_transport(records: Iterable[Mapping[str, Any]],
                                       terminal_movement: Iterable[Mapping[str, Any]]) -> list[tuple[dict | None, dict]]:
    """Authenticate each independent native publisher against its complete scope."""
    groups = defaultdict(list)
    for record in records:
        if ((record.get("tags") or {}).get("policy") == "native_load_spread"
                and record.get("domain") == "moe_overlay_controller"
                and record.get("name") in {"dynamic_movement_transactions", "dynamic_physical_bytes",
                                           "dynamic_migration_edges", "dynamic_migration_edge_identities"}):
            groups[native_movement_scope(record)].append(record)
    validated, matched = [], set()
    for identity, terminal in _terminal_scopes(terminal_movement):
        rank = terminal["rank"]
        if identity is None:
            candidates = [key for key in groups if key[0] == rank]
            if len(candidates) > 1:
                raise ValueError("multiple native domains require scoped terminal evidence")
        else:
            candidates = [(rank, identity["first_layer"], identity["routed_last_layer"] - identity["first_layer"])]
        key = candidates[0] if candidates else None
        owned = groups.get(key, [])
        result = _validate_native_movement_transport_owner(owned, [terminal])
        if result:
            if key in matched:
                raise ValueError("duplicate native controller namespace")
            matched.add(key)
            if identity is not None:
                _scoped_edges(identity, result[0])
                devices = {":".join(value.split(":")[-2:]).lower() for value in identity["participants"]}
                if any(record["device"].lower() not in devices for record in owned):
                    raise ValueError("native publication root escaped its stage domain")
            if any(not key[1] <= edge["layer"] < key[1] + key[2] for edge in result[0]["edges"]):
                raise ValueError("native publication contains a foreign global layer")
            validated.append((identity, result[0]))
    if matched != set(groups):
        raise ValueError("native movement counter has no matching scoped terminal journal")
    return validated


_CONTROLLER_FIELDS = (
    "base_epoch", "promotions", "demotions", "same_priority_moves", "cross_domain_moves", "cross_rank_moves",
    "cross_backend_moves", "snapshot_observations", "priority_cost_before", "priority_cost_after",
    "same_priority_makespan_before", "same_priority_makespan_after", "accepted_cycles", "physical_cycles", "rejected_cycles",
    "phase_tradeoff_candidates", "improvement_floor_rejected_cycles", "payoff_rejected_cycles",
    "residency_rejected_cycles", "projected_service_gain_ns", "projected_transfer_and_repack_ns",
    "projected_inference_interference_ns", "projected_net_benefit_ns", "changed_layers", "layer_scan_start",
    "layer_scan_next", "edges_checked", "participant_coordinates_checked", "tier_coordinates_checked",
    "malformed_edges", "participant_flow_violations", "tier_flow_violations")
_PUBLICATION_FIELDS = ("transaction", "candidate_epoch", "command_count", "physical_payload_bytes")
_CONTROLLER_COUNTER_FIELDS = {
    "dynamic_movement_transactions": None, "dynamic_movement_commands": "command_count",
    "dynamic_physical_bytes": "physical_payload_bytes", "dynamic_promotions": "promotions",
    "dynamic_demotions": "demotions", "dynamic_same_priority_moves": "same_priority_moves",
    "dynamic_cross_domain_moves": "cross_domain_moves", "dynamic_cross_rank_moves": "cross_rank_moves",
    "dynamic_cross_backend_moves": "cross_backend_moves", "dynamic_capacity_conservation_certifications": None,
    "dynamic_migration_edges": "command_count", "dynamic_migration_edge_identities": "edges"}
CONTROLLER_MOVEMENT_TAGS = {
    "policy_owner": "device", "policy": "time_ns", "encoding": "controller_wave_v1",
    "parallel_submission": "true", "bounded_device_phases": "true", "resident_external_waits": "0",
    "prearmed_cross_device_fanin": "true", "retirement_attempts": "1", "retirement_busy_retries": "0",
    "blocking_inference": "false", "direction_counts_are_capacity_proof": "false"}


def controller_movement_words(publication: dict) -> tuple[int, ...]:
    """Validate and encode the full physical, economy and capacity receipt."""
    if not isinstance(publication, dict) or set(publication) != set(_PUBLICATION_FIELDS) | {"controller"}:
        raise ValueError("invalid controller physical publication fields")
    physical = tuple(_integer(publication, field, 1) for field in _PUBLICATION_FIELDS)
    receipt = publication["controller"]
    if not isinstance(receipt, dict) or set(receipt) != set(_CONTROLLER_FIELDS):
        raise ValueError("invalid completed controller receipt fields")
    words = tuple(_integer(receipt, field) for field in _CONTROLLER_FIELDS)
    commands = physical[2]
    gain, cost, net = receipt["projected_service_gain_ns"], (receipt["projected_transfer_and_repack_ns"] +
        receipt["projected_inference_interference_ns"]), receipt["projected_net_benefit_ns"]
    if (receipt["base_epoch"] == 0 or physical[1] != receipt["base_epoch"] + 1
            or sum(receipt[field] for field in ("promotions", "demotions", "same_priority_moves")) != commands
            or any(receipt[field] > commands for field in ("cross_domain_moves", "cross_rank_moves", "cross_backend_moves"))
            or not 0 < receipt["accepted_cycles"] <= commands or not 0 < receipt["physical_cycles"] <= commands
            or not 0 < net == gain - cost
            or cost > _U64_MAX or receipt["edges_checked"] != commands
            or not receipt["participant_coordinates_checked"] or not receipt["tier_coordinates_checked"]
            or any(receipt[field] for field in ("malformed_edges", "participant_flow_violations", "tier_flow_violations"))):
        raise ValueError("controller receipt has invalid epoch, economy, counts or capacity proof")
    return physical + words


def controller_movement_edge_words(edge: dict) -> tuple[int, ...]:
    """Encode all device-owned edges, including cross-vendor and cross-rank cycles."""
    _validate_edge(edge)
    source, destination = edge["source_device"].split(":"), edge["destination_device"].split(":")
    if (edge["authority"] != "device" or len(source) != 2 or len(destination) != 2
            or source[0] not in {"CUDA", "ROCm"} or destination[0] not in {"CUDA", "ROCm"}):
        raise ValueError("controller movement requires device-owned GPU edges")
    backend = {"CUDA": 1, "ROCm": 2}
    source_rank, destination_rank = edge["source_world_rank"], edge["destination_world_rank"]
    return (edge["transaction"], edge["candidate_epoch"], edge["layer"], edge["expert"],
            edge["cycle_index"], edge["cycle_size"], {"same_priority": 0, "promotion": 1, "demotion": 2}[edge["direction"]],
            {"tier_residency": 0, "participant_placement": 1, "combined": 2}[edge["movement_axis"]],
            edge["source_participant"], edge["destination_participant"],
            edge["source_priority"] & _U64_MAX, edge["destination_priority"] & _U64_MAX,
            backend[source[0]], int(source[1]), backend[destination[0]], int(destination[1]),
            int(source_rank is not None), source_rank if source_rank is not None else 0,
            int(destination_rank is not None), destination_rank if destination_rank is not None else 0,
            edge["estimated_weight_bytes"], edge["activation_count"], 0)


def validate_controller_movement_transport(records: Iterable[Mapping[str, Any]],
                                           terminal_movement: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Authenticate every rank's bounded controller counters against the full journal.

    Exactly one policy leader retains the admitting economy. Followers retain
    the same physical receipts/edges and cannot replace that leader. Per-wave
    classification and cycle coverage are validated before sequence comparison;
    endpoint ranges or totals alone never establish membership.
    """
    owner_documents = []
    for identity, document in _terminal_scopes(terminal_movement):
        if identity is None:
            owner_documents.append(document)
        elif (any("controller" in publication for publication in document.get("device_publications", []))
              or any(proof.get("authority") == "device" and proof.get("policy") == "time_ns"
                     for proof in document.get("movement", {}).get("economy", []))):
            raise ValueError("pipeline controller publication requires its own scoped transport witness")
    terminal_movement = owner_documents
    scopes = {}
    for record in records:
        tags = record.get("tags") or {}
        if (record.get("domain") != "moe_overlay_controller" or record.get("name") not in _CONTROLLER_COUNTER_FIELDS
                or tags.get("policy") == "native_load_spread"):
            continue
        rank = _integer(record, "rank", 0, _I32_MAX)
        if (record.get("phase") != "maintenance" or not isinstance(record.get("device"), str)
                or not record["device"] or tags != CONTROLLER_MOVEMENT_TAGS):
            raise ValueError("invalid bounded controller transport publication scope")
        scope = scopes.setdefault(rank, {"device": record["device"], "records": {}})
        if scope["device"] != record["device"] or record["name"] in scope["records"]:
            raise ValueError("duplicate controller transport publication scope")
        scope["records"][record["name"]] = record
    validated, seen, matched, leaders = [], set(), set(), []
    for terminal in terminal_movement:
        if (not isinstance(terminal, dict) or type(terminal.get("schema")) is not int or terminal["schema"] != 1
                or terminal.get("scope") != "terminal_model_lifetime"):
            raise ValueError("invalid terminal controller movement transport artifact")
        rank = _integer(terminal, "rank", 0, _I32_MAX)
        if rank in seen:
            raise ValueError("duplicate terminal controller movement rank")
        seen.add(rank)
        journal, publications = terminal.get("movement"), terminal.get("device_publications")
        if not isinstance(journal, dict) or not isinstance(publications, list):
            raise ValueError("missing terminal controller owner journal")
        is_controller = any(isinstance(p, dict) and "controller" in p for p in publications)
        device_time = any(isinstance(e, dict) and e.get("authority") == "device" and e.get("policy") == "time_ns"
                          for e in journal.get("economy", []))
        if not is_controller and not device_time and rank not in scopes:
            continue
        if not publications:
            raise ValueError("controller movement lost its physical publication history")
        follower = not journal.get("economy")
        validate_movement_ledger(journal, device_follower=follower)
        if not follower:
            leaders.append(terminal)
        wave_steps, offset = [], 0
        for publication in publications:
            words = controller_movement_words(publication)
            receipt, command_count = publication["controller"], publication["command_count"]
            edges = journal["edges"][offset:offset + command_count]
            offset += command_count
            if len(edges) != command_count or any((e["transaction"], e["candidate_epoch"]) != words[:2] for e in edges):
                raise ValueError("controller physical publication disagrees with exact edge order")
            direction = Counter(e["direction"] for e in edges)
            cycles = {e["cycle_index"] for e in edges}
            cross_rank = sum(e["source_world_rank"] is not None and e["destination_world_rank"] is not None
                             and e["source_world_rank"] != e["destination_world_rank"] for e in edges)
            cross_backend = sum(e["source_device"].split(":")[0] != e["destination_device"].split(":")[0] for e in edges)
            if (any(direction[key] != receipt[field] for key, field in
                    (("promotion", "promotions"), ("demotion", "demotions"), ("same_priority", "same_priority_moves")))
                    or len(cycles) != receipt["physical_cycles"] or cross_rank != receipt["cross_rank_moves"]
                    or cross_backend != receipt["cross_backend_moves"]):
                raise ValueError("controller completed edge classification disagrees with receipt")
            if not follower:
                economy = next((e for e in journal["economy"] if (e["transaction"], e["candidate_epoch"]) == words[:2]), None)
                if (economy is None or economy["policy"] != "time_ns" or economy["command_count"] != command_count
                        or economy["cycle_count"] != receipt["physical_cycles"]
                        or any(economy[field] != receipt[field] for field in _TIME_ECONOMY)):
                    raise ValueError("controller receipt disagrees with policy leader economy")
            wave_steps.append(words)
        if offset != len(journal["edges"]):
            raise ValueError("controller edge history has no matching completed publication")
        scope = scopes.get(rank, {}).get("records", {})
        if set(scope) != set(_CONTROLLER_COUNTER_FIELDS):
            raise ValueError("controller movement lacks completed transport counter families")
        wave_witness = native_movement_sequence(wave_steps)
        edge_witness = native_movement_sequence(controller_movement_edge_words(e) for e in journal["edges"])
        for name, field in _CONTROLLER_COUNTER_FIELDS.items():
            expected = edge_witness if field == "edges" else wave_witness
            total = float(len(journal["edges"])) if field == "edges" else 0.0
            if field != "edges":
                for publication in publications:
                    total += float(1 if field is None else publication.get(field, publication["controller"].get(field)))
            record = scope[name]
            if (record.get("kind") != "counter" or type(record.get("value")) not in (int, float) or record["value"] != total
                    or any(type(record.get(key)) is not type(value) or record[key] != value
                           or (isinstance(value, list) and any(type(word) is not int for word in record[key]))
                           for key, value in expected.items())):
                raise ValueError("controller counter disagrees with exact completed transport sequence: " + name)
        matched.add(rank)
        validated.append(terminal)
    if matched != set(scopes):
        raise ValueError("controller counter has no matching terminal history")
    if validated:
        if len(leaders) != 1:
            raise ValueError("controller transport requires exactly one policy leader")
        for terminal in validated:
            if (terminal["movement"]["edges"] != leaders[0]["movement"]["edges"]
                    or terminal["device_publications"] != leaders[0]["device_publications"]):
                raise ValueError("controller follower disagrees with the policy leader's complete history")
    return [terminal["movement"] for terminal in leaders]


def validate_movement_transport_mirrors(ledger: dict, records: Iterable[Mapping[str, Any]],
                                       terminal_movement: Iterable[Mapping[str, Any]] = ()) -> None:
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
    records, terminal_movement = list(records), list(terminal_movement)
    if ledger["schema"] == 3:
        completed = validate_native_movement_transport(records, terminal_movement)
        terminal_scopes = _terminal_scopes(terminal_movement)
        for identity, leaf in _ledger_leaves(ledger):
            candidates = [document for scope, document in terminal_scopes if scope == identity]
            if not candidates:
                raise ValueError("HTTP pipeline stage has no matching terminal namespace")
            if any(record["policy"] == "native_load_spread" for record in leaf["economy"]):
                if not any(scope == identity and all(final[name][:len(leaf[name])] == leaf[name] for name in _ARRAYS)
                           for scope, final in completed):
                    raise ValueError("HTTP pipeline movement lacks its exact completed transport prefix")
            else:
                validate_movement_transport_mirrors(leaf, records, candidates)
        return
    if any(record["policy"] == "native_load_spread" for record in ledger["economy"]):
        completed = validate_native_movement_transport(records, terminal_movement)
        if not any(all(terminal[name][:len(ledger[name])] == ledger[name] for name in _ARRAYS)
                   for identity, terminal in completed if identity is None):
            raise ValueError("HTTP movement history lacks matching completed transport prefix")
        return
    if any(edge["authority"] == "device" for edge in ledger["edges"]):
        completed = validate_controller_movement_transport(records, terminal_movement)
        if not any(all(terminal[name][:len(ledger[name])] == ledger[name] for name in _ARRAYS)
                   for terminal in completed):
            raise ValueError("HTTP controller movement lacks matching completed transport prefix")
        return
    fields = ("candidate_epoch", "layer", "expert", "source_participant", "destination_participant",
              "source_priority", "destination_priority", "source_device", "destination_device",
              "movement_axis", "direction")
    observed = set()
    for record in records:
        family = (record.get("domain"), record.get("name"))
        if family == ("moe_overlay_residency", "expert_migration_edges"):
            authority = "host"
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
        if not isinstance(summary, dict) or type(summary.get("schema")) is not int or summary["schema"] not in (1, 2):
            raise ValueError("missing terminal runtime summary for movement")
        current = summary.get("expert_movement")
        validate_movement_ledger(current)
        leaves = _ledger_leaves(current)
        if self._requirement is not MovementRequirement.REQUIRED and any(leaf[name] for _, leaf in leaves for name in _ARRAYS):
            raise ValueError("movement occurred in a forbidden or non-applicable cell")
        topology = summary.get("expert_movement_topology")
        validate_movement_topology(topology)
        scopes = ([(entry["identity"], entry["topology"]) for entry in _pipeline_entries(topology, "topology")]
                  if topology["schema"] == 2 else [(None, topology)])
        if [identity for identity, _ in leaves] != [identity for identity, _ in scopes]:
            raise ValueError("movement and frozen topology disagree on stage ownership")
        for (_, leaf), (_, geometry) in zip(leaves, scopes):
            if (any(edge["authority"] != geometry["authority"] for edge in leaf["edges"])
                    or not _observed_axes(leaf["edges"]) <= validate_movement_topology(geometry)):
                raise ValueError("movement authority or objective contradicts the frozen topology")
        if self._topology is not None and topology != self._topology:
            raise ValueError("movement topology changed within a model lifetime")
        if self._previous is not None:
            previous_leaves = _ledger_leaves(self._previous)
            if [identity for identity, _ in previous_leaves] != [identity for identity, _ in leaves]:
                raise ValueError("movement stage ownership changed within a model lifetime")
            for (_, previous), (_, leaf) in zip(previous_leaves, leaves):
                for name in _ARRAYS:
                    if leaf[name][:len(previous[name])] != previous[name]:
                        raise ValueError("movement ledger history regressed or changed: " + name)
        self._previous = copy.deepcopy(current)
        self._topology = copy.deepcopy(topology)

    def finish(self) -> None:
        """Require a complete observed cohort and positive movement when selected."""
        if self._previous is None:
            raise ValueError("no movement ledger observations")
        if self._requirement is MovementRequirement.REQUIRED:
            scopes = ([entry["topology"] for entry in _pipeline_entries(self._topology, "topology")]
                      if self._topology["schema"] == 2 else [self._topology])
            for (_, leaf), geometry in zip(_ledger_leaves(self._previous), scopes):
                required = validate_movement_topology(geometry)
                if not required or not required <= _observed_axes(leaf["edges"]):
                    raise ValueError("required movement does not cover every frozen-topology stage axis")
