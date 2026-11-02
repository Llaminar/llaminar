#!/usr/bin/env python3
"""Original selected-worker birth diagnostics, using the existing owned-child observer.

This never certifies a bound. External metadata-only observation is mandatory,
and begins before Popen/any child SDK API. Exact stack/object roots, backend-wide
startup, selected original worker constructors, partial rollback, retained pools
and actual runtime reset remain distinct observations. Native execution is an
explicit root/user operation, not an ordinary Unit/preflight side effect.
"""
from __future__ import annotations
import argparse
import json
import platform
import stat
from pathlib import Path
import run_read_ahead_resource_calibration as common

SCHEMA = "llaminar.native_context_birth_calibration.v1"


LOOKUP_PROGRAM_COUNTS = {"owner-c0":3,"owner-positive":4,"coordinator-c0":3,"coordinator-positive":4,
                         "source":3,"boundary":1,"local-c0":10,"local-mixed":11,"local-positive":12,"remote-c0":3,"remote-positive":4}


def validate_lookup_events(events: list[dict], geometry: dict) -> int:
    """Validate observed raw existing API prefixes; this never supplies a production certificate."""
    program=geometry.get("lookup_event_program","")
    expected=LOOKUP_PROGRAM_COUNTS.get(program,0)
    if (program and not expected) or geometry.get("lookup_event_count",0)!=expected:
        raise ValueError("Unknown/incompatible closed lookup event census")
    prefix=geometry.get("lookup_event_prefix")
    prefix=expected if prefix is None else prefix
    if type(prefix) is not int or not 0<=prefix<=expected:
        raise ValueError("Invalid raw lookup prefix")
    aux=[row for row in events if row["stage"].startswith(("lookup_","before_lookup_","after_lookup_"))]
    if not program:
        if aux:raise ValueError("Unrequested lookup event API population")
        return 0
    names=[row["stage"] for row in events]
    for name in ("lookup_program_baseline","lookup_program_prefix_ready","lookup_program_handles_destroyed"):
        if names.count(name)!=1:raise ValueError("Missing lookup event lifetime boundary")
    closure=events[0]["closure"]
    resource=closure.get("lookup_program_family","")
    if resource!="native-auxiliary-operations" or closure.get("lookup_receipt_role")!="raw_existing_ordering_program_no_certificate":raise ValueError("Lookup diagnostic lost its closed native operation family/role")
    baseline=names.index("lookup_program_baseline")
    baseline_extra=events[baseline]["extra"]
    if baseline_extra.get("original_ordering_events")!=32 or baseline_extra.get("program")!=program or baseline_extra.get("simultaneous_events")!=expected or baseline_extra.get("requested_prefix")!=prefix:raise ValueError("Lookup baseline changed exact simultaneous population")
    ready=next(row["extra"] for row in events if row["stage"]=="lookup_program_prefix_ready")
    if ready.get("live_events")!=prefix or ready.get("partial_prefix")!=(prefix<expected):raise ValueError("Lookup prefix population drifted")
    if baseline<=max(i for i,name in enumerate(names) if name in ("native_first_use_complete","native_reuse_complete")):
        raise ValueError("Lookup events preceded original32 event use baseline")
    before=[r["extra"] for r in events if r["stage"]=="before_lookup_event_creator"]
    after=[r["extra"] for r in events if r["stage"]=="after_lookup_event_creator"]
    destroyed=[r["extra"] for r in events if r["stage"]=="after_lookup_event_destroy"]
    if len(before)!=prefix or len(after)!=prefix or len(destroyed)!=prefix:
        raise ValueError("Incomplete actual lookup SDK prefix")
    for index,(begin,end) in enumerate(zip(before,after),36):
        if any(r.get("creator")!=index or r.get("role")!="ordering_event" or not r.get("caller") for r in (begin,end)) or end.get("success") is not True or end.get("native_status")!=0 or not end.get("handle"):
            raise ValueError("Lookup prefix lost exact actual creator order/status")
    if len({r["handle"] for r in after})!=prefix or {(r["creator"],r["handle"]) for r in after}!={(r.get("creator"),r.get("handle")) for r in destroyed} or any(r.get("destroy_success") is not True or r.get("destroy_status")!=0 for r in destroyed):
        raise ValueError("Lookup prefix cleanup lost actual SDK handle")
    created=[r["extra"].get("event") for r in events if r["stage"]=="lookup_event_created"]
    uses=[r["extra"] for r in events if r["stage"]=="lookup_program_use_complete"]
    if created!=list(range(prefix)) or [r.get("replay") for r in uses]!=list(range(geometry["replays"])) or any(r.get("ordering_events_used")!=prefix or r.get("rerecord")!=(i!=0) for i,r in enumerate(uses)):
        raise ValueError("Lookup re-record/wait program incomplete")
    terminal=next(r["extra"] for r in events if r["stage"]=="lookup_program_handles_destroyed")
    if terminal.get("destroyed_events")!=prefix or terminal.get("physical_release_claimed") is not False or names.index("lookup_program_handles_destroyed")>=names.index("original_worker_destroyed"):
        raise ValueError("Lookup handles outlived original worker or claimed physical release")
    return prefix


def validate_records(records: list[dict], device: str, fail_creator: int) -> None:
    """Require actual original creation/use/cleanup/reset; never synthesize absent data."""
    if not records:
        raise ValueError("Missing original worker birth observations")
    first = records[0]
    geometry = first["geometry"]
    if any(geometry.get(key) != value for key, value in {"primary_streams": 1, "primary_blas_handles": 1, "secondary_blas_handles": 1, "ordering_events": 32, "selected_workers": 1, "pool_admission": False, "opaque_family_excludes_exact_stack_object": True, "fail_creator": fail_creator}.items()):
        raise ValueError("Wrong original selected-worker family")
    if not 1 <= geometry.get("lifetimes", 0) <= 128 or not 1 <= geometry.get("replays", 0) <= 1024:
        raise ValueError("Invalid original family repetition geometry")
    if fail_creator and (not 1 <= fail_creator <= 35 or geometry["lifetimes"] != 1):
        raise ValueError("Invalid partial original creator prefix")
    reset_between = geometry.get("reset_between_lifetimes", False)
    if type(reset_between) is not bool:
        raise ValueError("Invalid reset-between-lifetimes policy")
    for index, record in enumerate(records):
        if record.get("schema") != SCHEMA or record.get("sequence") != index or record.get("pid") != first["pid"] or record.get("geometry") != geometry:
            raise ValueError("Reordered/incompatible native birth trace")
        if record.get("qualified") is not False or record.get("production_admission") is not False or record.get("certified_limit") is not None:
            raise ValueError("Raw diagnostic attempted to certify production")
        if record["closure"].get("device", "").lower() != device.lower():
            raise ValueError("Diagnostic changed physical endpoint")
    stages = [record["stage"] for record in records]
    if stages[0] != "before_backend_lookup" or stages[-1] != "exclusive_context_retirement_complete" or stages.count("startup_metadata_complete") != 1:
        raise ValueError("Missing pre-API startup/reset boundaries")
    causes=[record for record in records if record["stage"]=="original_constructor_failed"]
    if fail_creator:
        if len(causes)!=1 or causes[0]["extra"].get("lifetime")!=0 or not isinstance(causes[0]["extra"].get("reason"),str) or not causes[0]["extra"]["reason"].strip():
            raise ValueError("Injected original constructor failure lost its unique nonempty primary reason")
    elif causes:
        raise ValueError("A successful birth trace contains an original constructor failure")
    startup = stages.index("startup_metadata_complete")
    closure = records[startup]["closure"]
    for key in ("physical_uuid", "architecture", "runtime_version", "driver_api_version", "startup_visible_devices"):
        if not closure.get(key):
            raise ValueError("Missing actual startup closure: " + key)
    for key in ("resource_source_sha256", "calibration_source_sha256"):
        value = closure.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(letter not in "0123456789abcdef" for letter in value):
            raise ValueError("Missing original source identity")
    if any(record["closure"] != closure for record in records[startup:]):
        raise ValueError("Selected original source/runtime identity drifted")
    for lifetime in range(geometry["lifetimes"]):
        events = [record for record in records if record["extra"].get("lifetime") == lifetime]
        names = [record["stage"] for record in events]
        for stage in ("exact_stack_object_root_ready", "original_worker_destroyed", "exact_stack_object_root_released"):
            if names.count(stage) != 1:
                raise ValueError("Missing original exact-root/destruction boundary")
        root = next(record["extra"] for record in events if record["stage"] == "exact_stack_object_root_ready")
        if min(root.get("stack_bytes", 0), root.get("object_bytes", 0), root.get("object_alignment", 0)) <= 0 or root.get("exact_pma_bytes", 0) <= root["stack_bytes"]:
            raise ValueError("Missing exact original stack/object root geometry")
        before = [record for record in events if record["stage"] == "before_original_creator"]
        after = [record for record in events if record["stage"] == "after_original_creator"]
        count = fail_creator or 35
        if len(before) != count or len(after) != count:
            raise ValueError("Incomplete actual original creator prefix/interposition")
        for index, (begin, end) in enumerate(zip(before, after), 1):
            role = ("primary_stream", "primary_blas", "secondary_blas")[index-1] if index <= 3 else "ordering_event"
            for record in (begin, end):
                if record["extra"].get("creator") != index or record["extra"].get("role") != role or not record["extra"].get("caller"):
                    raise ValueError("Wrong actual original creator order/source")
            injected = index == fail_creator
            status=end["extra"].get("native_status")
            if not isinstance(status,int) or (status==0)==injected or end["extra"].get("injected") != injected or end["extra"].get("success") is not (not injected) or (not injected and not end["extra"].get("handle")):
                raise ValueError("Partial original creator failure did not execute")
            if records.index(begin) >= records.index(end):
                raise ValueError("Creator completion preceded original call")
        released = [record for record in events if record["stage"] == "after_original_destroy"]
        expected_releases = {(row["extra"]["creator"], row["extra"]["role"], row["extra"]["handle"]) for row in after if row["extra"]["success"]}
        actual_releases = {(row["extra"].get("creator"), row["extra"].get("role"), row["extra"].get("handle")) for row in released}
        if len(released) != count-bool(fail_creator) or actual_releases != expected_releases or any(row["extra"].get("destroy_success") is not True or row["extra"].get("destroy_status")!=0 for row in released):
            raise ValueError("Original rollback/destruction omitted or duplicated native owners")
        lookup_prefix=validate_lookup_events(events,geometry)
        destroyed = next(record for record in events if record["stage"] == "original_worker_destroyed")
        if destroyed["extra"].get("rollback_complete") is not True or destroyed["extra"].get("created_handles") != len(expected_releases)+lookup_prefix or destroyed["extra"].get("destroyed_handles") != len(expected_releases)+lookup_prefix:
            raise ValueError("Original worker joined without complete native owner retirement")
        if any(records.index(row) >= records.index(destroyed) for row in released):
            raise ValueError("Original owner outlived worker destruction")
        if fail_creator:
            if names.count("worker_birth_failed_cleanup_complete") != 1 or "selected_worker_ready" in names:
                raise ValueError("Partial original worker escaped rollback")
        else:
            if names.count("selected_worker_ready") != 1:
                raise ValueError("Missing actual selected worker publication")
            uses = [record["extra"].get("replay") for record in events if record["stage"] in ("native_first_use_complete", "native_reuse_complete")]
            if uses != list(range(geometry["replays"])):
                raise ValueError("Missing actual ordering-event family use")
        if not names.index("exact_stack_object_root_ready") < names.index("before_original_creator") < names.index("original_worker_destroyed") < names.index("exact_stack_object_root_released"):
            raise ValueError("Exact-root lifetime does not retain original worker")
    resets = [record for record in records if record["stage"] == "exclusive_context_retirement_complete"]
    if len(resets) != (geometry["lifetimes"] if reset_between else 1):
        raise ValueError("Missing or duplicated requested physical reset boundaries")
    previous_successor = None
    for index, reset in enumerate(resets):
        terminal = reset["extra"]
        if terminal.get("runtime_post_reset_state") != "quiescent" or terminal.get("reset_invoked") is not True or type(terminal.get("retired_runtime_generation")) is not int or terminal["retired_runtime_generation"] <= 0 or terminal.get("successor_runtime_generation") != terminal["retired_runtime_generation"] + 1:
            raise ValueError("Actual backend physical reset receipt incomplete")
        if terminal.get("retired_context_generation") != 0 or terminal.get("raw_direct_owner") is not True or terminal.get("canonical_allocations_before_runtime_reset") != 0 or terminal.get("canonical_bytes_before_runtime_reset") != 0 or terminal.get("diagnostic_failed") is not False:
            raise ValueError("Raw direct constructor was confused with admitted pool ownership")
        if terminal.get("injected_failure") != bool(fail_creator):
            raise ValueError("Wrong partial-creator retirement receipt")
        lifetime = index if reset_between else geometry["lifetimes"] - 1
        if reset_between and terminal.get("lifetime") != lifetime:
            raise ValueError("Reset does not name its original lifetime")
        released = next(row for row in records if row["stage"] == "exact_stack_object_root_released" and row["extra"].get("lifetime") == lifetime)
        if records.index(released) >= records.index(reset):
            raise ValueError("Physical reset preceded original root destruction")
        if reset_between:
            if previous_successor is not None and terminal["retired_runtime_generation"] != previous_successor:
                raise ValueError("Reset/rebirth runtime generation chain drifted")
            previous_successor = terminal["successor_runtime_generation"]
            if lifetime + 1 < geometry["lifetimes"]:
                next_root = next(row for row in records if row["stage"] == "exact_stack_object_root_ready" and row["extra"].get("lifetime") == lifetime + 1)
                if records.index(next_root) != records.index(reset) + 1:
                    raise ValueError("Post-reset checkpoint did not precede next requested birth")


def validate_source_snapshot(path: Path) -> dict:
    """Authenticate regular immutable source copies; no native library/device is loaded."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Creator source snapshot must be a regular manifest")
    value=json.loads(path.read_text())
    if value.get("schema")!="llaminar.native_context_birth_source_snapshot.v1" or not value.get("inputs"):
        raise ValueError("Missing original creator snapshot")
    text=""
    for item in value["inputs"]:
        artifact=(path.parent/item["snapshot"]).resolve(strict=True)
        if not artifact.is_relative_to(path.parent.resolve()) or not stat.S_ISREG(artifact.stat().st_mode) or common.digest(artifact)!=item["sha256"]:
            raise ValueError("Original creator source snapshot changed or escaped its evidence owner")
        text+=item["path"]+":"+item["sha256"]+";"
    if common.hashlib.sha256(text.encode()).hexdigest()!=value.get("source_sha256"):
        raise ValueError("Original creator source closure is incomplete")
    return value


def validate_checkpoint(record: dict, device: str, source_sha256: str, reset_between_lifetimes: bool = False) -> None:
    """Reject a mismatched constructor binary before acknowledging its first SDK operation."""
    if record.get("schema")!=SCHEMA or record.get("closure",{}).get("resource_source_sha256")!=source_sha256 or record.get("closure",{}).get("device","").lower()!=device.lower():
        raise ValueError("Linked diagnostic does not match original creator source snapshot")
    if record.get("qualified") is not False or record.get("production_admission") is not False or record.get("certified_limit") is not None:
        raise ValueError("Raw creator diagnostic attempted production admission")
    if record.get("geometry", {}).get("reset_between_lifetimes", False) is not reset_between_lifetimes:
        raise ValueError("Linked diagnostic changed requested reset policy")


def validate_caller_identity(identity: dict) -> None:
    """Require actual ELF content identity; build-ID notes are optional ELF metadata.

    Project and SDK builds may intentionally omit GNU build-ID notes. The
    observer has already checked regular-file type and ELF magic; its full
    content digest and extent remain mandatory regardless of optional notes.
    """
    digest=identity.get("sha256")
    if (not isinstance(digest,str) or len(digest)!=64 or
            any(value not in "0123456789abcdef" for value in digest) or
            type(identity.get("size")) is not int or identity["size"]<=0 or
            not isinstance(identity.get("elf_build_ids"),list)):
        raise ValueError("Original creator caller lacks its actual regular ELF identity")


def validate_requested_lookup(record: dict,args) -> None:
    """Reject child/request drift before acknowledging another native API."""
    geometry=record.get("geometry",{})
    if geometry.get("lookup_event_program","")!=getattr(args,"lookup_event_program","") or geometry.get("lookup_event_prefix")!=getattr(args,"lookup_event_prefix",None):
        raise ValueError("Child changed requested raw lookup program/prefix")


def run_one(args, label: str, lifetimes: int, failure: int = 0) -> dict:
    """Reuse the canonical observer/session owner, including mandatory prelaunch GPU sampling."""
    command = [str(args.binary), "--device", args.device, "--lifetimes", str(lifetimes), "--replays", str(args.replays), "--handshake"]
    if getattr(args,"lookup_event_program",""):
        command += ["--lookup-event-program",args.lookup_event_program]
        if getattr(args,"lookup_event_prefix",None) is not None:command += ["--lookup-event-prefix",str(args.lookup_event_prefix)]
    if getattr(args, "reset_between_lifetimes", False):
        command.append("--reset-between-lifetimes")
    result = common.run_one(args, label, lifetimes, failure, child_command=command, expected_schema=SCHEMA, record_validator=validate_records,
        checkpoint_validator=lambda record: (validate_requested_lookup(record,args),validate_checkpoint(record,args.device,args.creator_source["source_sha256"],getattr(args,"reset_between_lifetimes",False))))
    # Authenticate every observed project caller against actual regular ELF maps.
    # This never opens driver character nodes or invokes a vendor device API.
    for path in (args.output / label).glob("checkpoint-*.json"):
        value = json.loads(path.read_text())
        child = value["child"]
        if child["stage"] in ("before_original_creator", "after_original_creator", "after_original_destroy", "before_lookup_event_creator", "after_lookup_event_creator", "after_lookup_event_destroy"):
            caller = child["extra"]["caller"]
            identity = value["mapped_library_identities"].get(caller, {})
            validate_caller_identity(identity)
        if not value["external_device"].get("supported"):
            raise ValueError("Missing external metadata-only GPU observation")
    return result


def main() -> None:
    """Run only explicitly requested native children; measurements cannot install limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--creator-source-manifest",type=Path,required=True,help="Immutable matching original creator snapshot; checked before first SDK acknowledgment")
    parser.add_argument("--external-observer", type=Path, required=True, help="Root-selected sysfs or read-only NVML provider; must never call CUDA/HIP")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cold-processes", type=int, default=20)
    parser.add_argument("--warm-lifetimes", type=int, default=20)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--reset-between-lifetimes", action="store_true", help="Reset the actual original runtime after each worker lifetime; observe externally before next requested birth")
    parser.add_argument("--lookup-event-program", choices=tuple(LOOKUP_PROGRAM_COUNTS),default="")
    parser.add_argument("--lookup-event-prefix",type=int,help="Diagnostic stop after exact successful prefix; does not simulate SDK failure")
    parser.add_argument("--creator-rejection-controls", action="store_true", help="Exercise every actual stream/BLAS/Lt/32-event partial constructor prefix")
    args = parser.parse_args()
    args.device = args.device.lower()
    if not args.device.startswith(("cuda:", "rocm:")) or int(args.device.split(":")[1]) < 0 or min(args.replays, args.cold_processes, args.warm_lifetimes, args.sample_ms) <= 0 or max(args.replays, args.cold_processes, args.warm_lifetimes) > 128:
        parser.error("Invalid original selected-worker diagnostic geometry")
    if args.lookup_event_program and args.creator_rejection_controls:parser.error("Source birth failure controls and lookup program must be separate observations")
    if args.lookup_event_prefix is not None and (not args.lookup_event_program or not 0<=args.lookup_event_prefix<=LOOKUP_PROGRAM_COUNTS[args.lookup_event_program]):parser.error("Prefix outside selected closed lookup program")
    args.binary = args.binary.resolve(strict=True)
    args.external_observer = args.external_observer.resolve(strict=True)
    for path in (args.binary,args.external_observer):
        if not stat.S_ISREG(path.stat().st_mode):
            parser.error("Diagnostic inputs must be regular files, never device nodes/FIFOs")
    if args.binary.open("rb").read(4)!=b"\x7fELF":
        parser.error("Original native birth diagnostic must be its actual linked ELF")
    args.creator_source_manifest=args.creator_source_manifest.resolve(strict=True)
    args.creator_source=validate_source_snapshot(args.creator_source_manifest)
    args.identity_cache = {}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "llaminar.native_context_birth_dataset.v1", "binary": str(args.binary), "binary_sha256": common.digest(args.binary), "driver_script_sha256": common.digest(Path(__file__)), "owned_child_driver_sha256": common.digest(Path(common.__file__)), "external_observer_sha256": common.digest(args.external_observer), "platform": platform.uname()._asdict(), "device": args.device, "qualified": False, "certified_limit": None, "production_admission": False,"reset_between_lifetimes":args.reset_between_lifetimes,"creator_source_manifest":str(args.creator_source_manifest),"creator_source_manifest_sha256":common.digest(args.creator_source_manifest),"creator_source_sha256":args.creator_source["source_sha256"],"lookup_event_program":args.lookup_event_program,"lookup_event_prefix":args.lookup_event_prefix}
    for name,path in {"nvidia_kernel_driver":"/proc/driver/nvidia/version","amdgpu_kernel_version":"/sys/module/amdgpu/version","amdgpu_srcversion":"/sys/module/amdgpu/srcversion"}.items():
        try:manifest[name]=Path(path).read_text()
        except OSError as error:manifest[name]={"unavailable":str(error)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    try:
        for index in range(args.cold_processes):
            results.append(run_one(args, f"cold-{index:03d}", 1))
        results.append(run_one(args, "warm-repeated", args.warm_lifetimes))
        if args.creator_rejection_controls:
            for creator in range(1, 36):
                results.append(run_one(args, f"reject-creator-{creator:02d}", 1, creator))
    finally:
        (args.output / "dataset.json").write_text(json.dumps({"manifest": manifest, "processes": results, "qualified": False, "certified_limit": None}, indent=2) + "\n")


if __name__ == "__main__":
    main()
