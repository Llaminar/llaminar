#!/usr/bin/env python3
"""Collect complete participant-local server evidence after clean shutdown.

Each server rank writes the existing rank-qualified PerfStats format. Runtime
membership records identify the actual communicator and request authority;
neither device kind nor filename order is allowed to choose that authority.
Collection validates membership before publishing the cell's aggregate and
keeps every original row, with its rank attached, without summing measurements
or conflating process-local graph identities. Raw files remain diagnostic
artifacts alongside the aggregate. No inference state is read or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile


def _natural(value: object, field: str) -> int:
    """Read the non-negative decimal tags emitted by the C++ collector."""
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"invalid server membership {field}: {value!r}")
    return int(value)


def collect_ranked_perf_stats(output: Path) -> dict:
    """Validate all rank files associated with one exact aggregate path.

    Missing terminal ranks, conflicting communicator/authority declarations,
    malformed exports and filename/record mismatches fail closed. Membership
    comes from the runtime, so auto topology works without a second CLI planner.
    The returned document is not written until every participant is validated.
    """
    prefix = output.name.removesuffix(".json") + ".rank-"
    files = sorted(p for p in output.parent.iterdir()
                   if p.name.startswith(prefix) and p.name.endswith(".json"))
    if not files:
        raise ValueError("missing rank-qualified PerfStats artifacts")
    participants: dict[int, list[dict]] = {}
    membership: tuple[int, int] | None = None
    for path in files:
        rank = _natural(path.name[len(prefix):-5], "filename rank")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != "llaminar.perf_stats.v1":
            raise ValueError(f"invalid PerfStats schema: {path.name}")
        records = data.get("records")
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError(f"invalid PerfStats records: {path.name}")
        identities = [row for row in records if row.get("domain") == "server"
                      and row.get("name") == "rank_membership"]
        if len(identities) != 1:
            raise ValueError(f"expected one server membership record: {path.name}")
        identity = identities[0]
        tags = identity.get("tags")
        if not isinstance(tags, dict) or identity.get("value") != 1:
            raise ValueError(f"invalid server membership record: {path.name}")
        declared_rank = _natural(tags.get("rank"), "rank")
        size = _natural(tags.get("world_size"), "world_size")
        authority = _natural(tags.get("authority_rank"), "authority_rank")
        if declared_rank != rank or rank >= size or authority >= size:
            raise ValueError(f"server membership outside declared communicator: {path.name}")
        if membership is not None and membership != (size, authority):
            raise ValueError("conflicting server communicator/authority membership")
        membership = (size, authority)
        if rank in participants:
            raise ValueError(f"duplicate rank evidence: {rank}")
        # Rank is an identity component, never a tag to be guessed from device
        # ordinal. Preserve counters, timers and ordered witnesses byte-for-byte
        # in value; measurements from different processes are not coalesced.
        for record in records:
            if "rank" in record and record["rank"] != rank:
                raise ValueError(f"conflicting record rank: {path.name}")
            # This freshly parsed document has one owner. Attach provenance in
            # place instead of copying hundreds of thousands of dictionaries.
            # Original rank files and all measurement values remain unchanged.
            record["rank"] = rank
        participants[rank] = records
    assert membership is not None
    size, authority = membership
    if set(participants) != set(range(size)):
        raise ValueError(f"missing rank evidence: expected {size} ranks, got {sorted(participants)}")
    return {"schema": "llaminar.perf_stats.v1", "world_size": size,
            "authority_rank": authority,
            "records": [row for rank in sorted(participants) for row in participants[rank]]}


def publish_ranked_perf_stats(output: Path, data: dict) -> None:
    """Atomically publish the complete document without replacing good evidence on failure.

    Encode once using the standard library's C encoder, then write the newline
    separately to avoid another aggregate-sized string copy. The temporary file
    is a sibling so replacement is atomic on the artifact filesystem. A failed
    writer owns and removes only its own unpublished temporary file.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=output.parent, prefix=output.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(data, separators=(",", ":")))
            handle.write("\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def collect_and_publish_ranked_perf_stats(output: Path) -> dict:
    """Return the same document used for publication and all feature validators.

    The HTTP harness calls this boundary once after shutdown. Returning its
    owned document avoids a second process parsing the large aggregate just to
    check evidence which was already in memory. No rows are filtered/coalesced.
    """
    data = collect_ranked_perf_stats(output)
    publish_ranked_perf_stats(output, data)
    return data


def validate_memory_authority(data: dict) -> None:
    """Require each rank's canonical admission and bounded live-owner proof.

    This validates exported authority relationships; it never estimates bytes
    from model files, sums feature budgets, reserves space or makes allocation
    decisions. RSS remains independent OS telemetry because shared file-backed
    pages and runtime libraries are not synonymous with engine-owned storage.
    """
    size = data.get("world_size")
    if type(size) is not int or size <= 0:
        raise ValueError("physical memory evidence requires ranked server membership")
    resources: dict[tuple[int, str], int] = {}
    owners: dict[tuple[int, str], set[str]] = {}
    for row in data["records"]:
        if row.get("domain") != "physical_memory":
            continue
        name = row.get("name")
        if name not in {"resource_admission", "owner_attestation"}:
            continue
        tags = row.get("tags") or {}
        rank = row.get("rank")
        if (type(rank) is not int or rank < 0 or rank >= size
                or _natural(tags.get("rank"), "physical rank") != rank
                or row.get("value") != 1):
            raise ValueError("invalid physical memory authority identity")
        key = (rank, row.get("device", ""))
        if not key[1]:
            raise ValueError("physical memory authority omitted allocator device")
        if name == "resource_admission":
            if key in resources:
                raise ValueError("duplicate physical memory resource admission")
            resources[key] = _natural(tags.get("owner_count"), "owner_count")
            if (not resources[key] or _natural(tags.get("incremental_bytes"), "incremental_bytes")
                    > _natural(tags.get("available_bytes"), "available_bytes")):
                raise ValueError(f"physical memory admission exceeded capacity: {key}")
        else:
            owner = tags.get("owner")
            if not isinstance(owner, str) or not owner or owner in owners.setdefault(key, set()):
                raise ValueError("missing or duplicate physical memory owner")
            owners[key].add(owner)
            values = {field: _natural(tags.get(field), field) for field in (
                "planned_new_bytes", "planned_resident_bytes", "materialized_new_bytes",
                "committed_new_bytes", "adopted_resident_bytes")}
            if (not values["materialized_new_bytes"] <= values["committed_new_bytes"] <= values["planned_new_bytes"]
                    or values["adopted_resident_bytes"] > values["planned_resident_bytes"]):
                raise ValueError(f"physical memory owner exceeded authority envelope: {key}/{owner}")
    if ({rank for rank, _ in resources} != set(range(size)) or owners.keys() != resources.keys()
            or any(len(owners[key]) != count for key, count in resources.items())):
        raise ValueError("incomplete physical memory authority resource/owner evidence")


def main() -> int:
    """Publish the validated aggregate for the existing feature validators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        collect_and_publish_ranked_perf_stats(args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"PerfStats rank collection failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
