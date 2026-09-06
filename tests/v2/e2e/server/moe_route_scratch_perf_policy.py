"""Authenticate shared MoE route storage across production graph roles.

Continuation graphs bind separate main/MTP runtime tables to one arena.
Expert-only participants instead bind one model-lifetime runtime table reused
by all retained graph roles. Neither count may be inferred from a device vendor
or borrowed from another rank. Allocation bytes remain memory-authority evidence;
this module validates them without maintaining another ledger.
"""

from typing import Any, Iterable, Mapping


def validate_moe_route_scratch_policy(records: Iterable[Mapping[str, Any]]) -> str | None:
    """Require one immutable largest-participant arena and proven role reuse."""
    records = tuple(records)

    def value(record: Mapping[str, Any]) -> float:
        try:
            return float(record.get("value", record.get("count", 0)))
        except (TypeError, ValueError):
            return 0.0

    def owner(record: Mapping[str, Any]) -> tuple[object, str]:
        return record.get("rank"), str(record.get("device", "")).lower()

    allocations = [r for r in records if r.get("domain") == "memory"
                   and r.get("name") == "moe_serial_route_scratch_arena_allocations"]
    bindings = [r for r in records if r.get("domain") == "memory"
                and r.get("name") == "moe_serial_route_scratch_runtime_table_bindings"]
    if not allocations:
        return "GPU MoE+MTP cell emitted no immutable shared route-scratch arena allocation evidence"
    seen = set()
    for allocation in allocations:
        identity = owner(allocation)
        tags = allocation.get("tags") or {}
        try:
            size = int(tags.get("bytes", 0))
        except (TypeError, ValueError):
            size = 0
        if (identity in seen or value(allocation) != 1 or size <= 0
                or tags.get("ownership") != "per_device_serial_graph_domain"
                or tags.get("immutable") != "true" or tags.get("largest_participant") != "true"):
            return f"GPU MoE route scratch is not exactly one immutable largest-participant arena for {identity}"
        seen.add(identity)
        count = sum(value(r) for r in bindings if owner(r) == identity)
        if count > 1:
            continue

        # One follower runtime table is deliberately shared by every retained
        # main/verifier and sidecar family. Require explicit same-owner evidence
        # for both graph roles; a missing binding or graph can never pass.
        roles = set()
        mapped_families = set()
        sealed_mapped_families = False
        follower_runtime = False
        for record in records:
            graph_tags = record.get("tags") or {}
            if (record.get("domain") == "moe_overlay_controller"
                    and record.get("name") == "follower_runtime_tables_materialized"
                    and owner(record) == identity and value(record) == 1
                    and graph_tags.get("blocking_hot_path") == "false"):
                follower_runtime = True
            same_graph_owner = (record.get("rank") == identity[0]
                and identity[1] in str(record.get("device", "")).lower().split(",")
                and record.get("domain") == "moe_overlay_participant_graph"
                and value(record) > 0)
            if same_graph_owner:
                if (record.get("name") == "materialized_mapped_follower_families"
                        and graph_tags.get("standalone_progress_launch") == "false"):
                    try:
                        if int(graph_tags.get("setup_materialized_gpu_transactions", 0)) > 0:
                            mapped_families.add(graph_tags.get("graph_family"))
                    except (TypeError, ValueError):
                        pass
                if (record.get("name") == "mapped_follower_serving_graph_family_completions"
                        and graph_tags.get("authority_installation") == "before_native_capture"):
                    sealed_mapped_families = True
            if (record.get("domain") == "moe_overlay_participant_graph"
                    and record.get("name") == "materialized_graphs"
                    and record.get("rank") == identity[0] and value(record) > 0
                    and identity[1] in str(record.get("device", "")).lower().split(",")
                    and graph_tags.get("immutable") == "true"
                    and graph_tags.get("allocation_policy") == "setup_only"):
                roles.add(graph_tags.get("graph_role"))
        # Family zero is the main model; positive ordinals name its MTP
        # sidecars. Declaration alone is insufficient: the final native
        # materializations and the sealed serving-family barrier must exist.
        mapped_reuse = (follower_runtime and sealed_mapped_families
                        and {"0", "1"}.issubset(mapped_families))
        if count != 1 or not ({"main", "mtp_draft"}.issubset(roles) or mapped_reuse):
            return f"GPU MoE+MTP has no proven shared route-arena graph-role reuse for {identity}"
    return None
