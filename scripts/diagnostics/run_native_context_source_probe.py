#!/usr/bin/env python3
"""Explicit uninstrumented original-source projection/lifecycle observer; never installs limits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import stat
import run_read_ahead_resource_calibration as common
import run_native_context_birth_calibration as raw
SCHEMA="llaminar.native_context_source_probe.v1"

def validate_birth_key(closure: dict, device: str) -> None:
    """Authenticate native runtime/geometry and original live identity; application source is optional evidence before permitting worker birth."""
    key = closure.get("birth_key")
    if not isinstance(key, dict) or key.get("schema") != "llaminar.native_context_birth_key.v1" or key.get("device", "").lower() != device.lower():
        raise ValueError("Missing exact original production BirthKey projection")
    if key.get("population") != "additional_context" or type(key.get("source_program_version")) is not int or key["source_program_version"] != 1:
        raise ValueError("Wrong selected original BirthKey population/version")
    for name in ("visible_device_count", "compute_units", "native_wave_width", "worker_stack_bytes", "context_object_bytes", "context_object_alignment"):
        if type(key.get(name)) is not int or key[name] <= 0:
            raise ValueError("Incomplete production BirthKey geometry: " + name)
    if type(key.get("worker_guard_bytes")) is not int or key["worker_guard_bytes"] < 0 or key["context_object_alignment"] & (key["context_object_alignment"]-1):
        raise ValueError("Invalid production BirthKey exact root geometry")
    for name in ("physical_uuid", "architecture", "runtime_version", "driver_version"):
        if not isinstance(key.get(name), str) or not key[name].strip():
            raise ValueError("Missing actual production BirthKey closure: " + name)
    for name in ("backend_sha256", "runtime_sha256", "driver_sha256", "runtime_policy_sha256", "visibility_identity", "pthread_defaults_identity"):
        value = key.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) or set(value) == {"0"}:
            raise ValueError("Missing authenticated production BirthKey digest: " + name)
    hsa = key.get("hsa_sha256")
    if device.lower().startswith("rocm:"):
        if not isinstance(hsa, str) or len(hsa) != 64 or any(ch not in "0123456789abcdef" for ch in hsa) or set(hsa) == {"0"}:
            raise ValueError("Missing production BirthKey loaded HSA closure")
    elif hsa != "":
        raise ValueError("CUDA production BirthKey has a foreign HSA closure")
    for name, expected in {"primary_streams": 1, "primary_blas_handles": 1, "secondary_blas_handles": 1, "dependency_events": 32}.items():
        if type(key.get(name)) is not int or key[name] != expected:
            raise ValueError("Changed production BirthKey original creator population")

def validate_projection(closure: dict, device: str) -> None:
    """Reject interposed/raw keys and bind the production key to exact native SDK providers."""
    if closure.get("projection_role")!="uninstrumented_production_birth_key" or closure.get("creator_symbols_interposed") is not False or closure.get("existing_pool_worker") is not False:
        raise ValueError("Source key is not an uninstrumented original production projection")
    for name in ("resource_source_sha256","diagnostic_source_sha256"):
        value=closure.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(ch not in "0123456789abcdef" for ch in value):raise ValueError("Missing exact original source/diagnostic identity")
    validate_birth_key(closure,device)
    providers=closure.get("native_blas_providers")
    symbols=("cublasCreate_v2","cublasLtCreate") if device.lower().startswith("cuda:") else ("hipblasCreate","hipblasLtCreate")
    if not isinstance(providers,list) or len(providers)!=2:
        raise ValueError("Missing actual native SDK provider evidence")
    for provider,symbol in zip(providers,symbols):
        if provider.get("symbol")!=symbol or provider.get("matches_configured_provider") is not True or provider.get("provider_is_executable") is not False or not provider.get("path") or provider.get("path")!=provider.get("expected_path"):
            raise ValueError("Original creator symbol is interposed or differs from configured native SDK")
        digest=provider.get("sha256")
        if not isinstance(digest,str) or len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Original native SDK provider lacks actual ELF identity")
    combined=hashlib.sha256((providers[0]["sha256"]+":"+providers[1]["sha256"]).encode()).hexdigest()
    if closure["birth_key"]["backend_sha256"]!=combined:
        raise ValueError("Production BirthKey does not bind the actual configured native SDK providers")


def validate_records(records: list[dict],device: str,fail_creator: int=0) -> None:
    """Authenticate role, original same-state PMA/worker ordering and closed physical lifetime."""
    if not records or fail_creator:raise ValueError("Missing original source observations or foreign creator-fault mode")
    first=records[0];geometry=first.get("geometry",{});role=geometry.get("role")
    if role not in ("key_only","source_lifecycle") or geometry.get("selected_workers")!=1:raise ValueError("Unknown original source role")
    for index,row in enumerate(records):
        if row.get("schema")!=SCHEMA or row.get("sequence")!=index or row.get("pid")!=first.get("pid") or row.get("geometry")!=geometry or row.get("numerical_inference_executed") is not False or row.get("closure",{}).get("device","").lower()!=device.lower():raise ValueError("Incompatible original source trace")
    names=[row["stage"] for row in records]
    if names[:2]!=["before_backend_lookup","startup_metadata_complete"] or "birth_key" in first["closure"]:raise ValueError("Source projection preceded immutable metadata")
    closure=records[1]["closure"];validate_projection(closure,device)
    if any(row["closure"]!=closure for row in records[1:]):raise ValueError("Original source class/live identity drifted")
    if role=="key_only":
        if names!=["before_backend_lookup","startup_metadata_complete","production_key_complete"] or geometry.get("replays")!=0 or any(row.get("production_admission") is not False for row in records):raise ValueError("Metadata projection attempted source admission")
        if any(records[-1]["extra"].get(name) is not False for name in ("source_created","catalog_invoked","pool_worker_created")):raise ValueError("Key-only projection created a source/catalog/worker")
        return
    replays=geometry.get("replays")
    if type(replays) is not int or not 1<=replays<=1024:raise ValueError("Invalid source event replay geometry")
    expected=["before_backend_lookup","startup_metadata_complete","preworker_observations_ready","source_prepared","original_worker_ready","same_pma_admitted"]+["original_event_family_used"]*replays+["original_permit_released","original_source_retired"]
    if names!=expected:raise ValueError("Source creation/admission/use/permit/reset ordering incomplete")
    prepared=records[3]["extra"];host=prepared.get("contract_host_upper");gpu=prepared.get("contract_device_upper");lifetime=prepared.get("physical_lifetime")
    if type(host) is not int or host<=0 or type(gpu) is not int or gpu<=0 or lifetime not in ("ContextGeneration","ProcessRuntime"):raise ValueError("Missing closed original source commitment/lifetime")
    exact=prepared.get("pma",{}).get("exact_preparation_bytes")
    if type(exact) is not int or exact<=closure["birth_key"]["worker_stack_bytes"]:raise ValueError("Original source exact stack/object was not materialized")
    for index in (3,4,5):
        state=records[index]["extra"].get("pma",{})
        if state.get("exact_preparation_bytes")!=exact or state.get("host_opaque_bytes")!=host or state.get("device_opaque_bytes")!=gpu or state.get("opaque_committed_bytes")!=host+gpu or state.get("phase")!=("Admitted" if index==5 else "Preparing"):raise ValueError("Original source PMA obligations changed across admission")
    ready=records[4]["extra"];generation=ready.get("generation")
    if type(generation) is not int or generation<=0 or ready.get("exact_original_worker") is not True or ready.get("original_key_alias_verified") is not True:raise ValueError("Missing actual original prepared pool generation/key alias")
    if any(records[5]["extra"].get(name) is not True for name in ("same_authority_controlblock","same_certificate")):raise ValueError("Final source admission replaced original PMA/certificate")
    for index,row in enumerate(records):
        if row.get("production_admission") is not (index>=5):raise ValueError("Source admission role drifted")
    for replay,row in enumerate(records[6:6+replays]):
        if row["extra"]!={"replay":replay,"ordering_events":32,"same_retained_worker":True,"exact_primary_stream":True}:raise ValueError("Actual original event family was not exercised")
    terminal=records[-1]["extra"];state=terminal.get("pma",{})
    if terminal.get("generation")!=generation or terminal.get("retired_context_generation")!=generation or terminal.get("reset_invoked") is not True or terminal.get("runtime_post_reset_state")!="quiescent" or terminal.get("physical_lifetime")!=lifetime or terminal.get("canonical_allocations_before_runtime_reset")!=0 or terminal.get("canonical_bytes_before_runtime_reset")!=0:raise ValueError("Original native-only retirement receipt incomplete")
    retired=terminal.get("retired_runtime_generation")
    if type(retired) is not int or retired<=0 or terminal.get("successor_runtime_generation")!=retired+1:raise ValueError("Actual runtime reset generation missing")
    remaining=(host,gpu) if lifetime=="ProcessRuntime" else (0,0)
    if state.get("exact_preparation_bytes")!=0 or state.get("host_opaque_bytes")!=remaining[0] or state.get("device_opaque_bytes")!=remaining[1] or state.get("opaque_committed_bytes")!=sum(remaining):raise ValueError("Opaque source lifetime was released or retained against closed contract")


def validate_checkpoint(row: dict,device: str,source: str) -> None:
    """Fail before acknowledging SDK work when source/binding identity is incompatible."""
    if row.get("schema")!=SCHEMA or row.get("closure",{}).get("resource_source_sha256")!=source or row.get("closure",{}).get("device","").lower()!=device.lower():raise ValueError("Wrong original source probe/source snapshot")
    if row.get("stage")!="before_backend_lookup":validate_projection(row.get("closure",{}),device)


def main() -> None:
    """Run only root-selected native processes with original external observation/cleanup authority."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--device",required=True)
    parser.add_argument("--role",choices=("key-only","source-lifecycle"),required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--creator-source-manifest",type=Path,required=True)
    parser.add_argument("--external-observer",type=Path,required=True);parser.add_argument("--replays",type=int,default=2);parser.add_argument("--sample-ms",type=int,default=20)
    args=parser.parse_args();args.device=args.device.lower()
    if not args.device.startswith(("cuda:","rocm:")) or int(args.device.split(":")[1])<0 or not 1<=args.replays<=1024 or args.sample_ms<=0:parser.error("Invalid selected source observation geometry")
    args.binary=args.binary.resolve(strict=True);args.external_observer=args.external_observer.resolve(strict=True)
    for path in (args.binary,args.external_observer):
        if not stat.S_ISREG(path.stat().st_mode):parser.error("Native observation input must be a regular file")
    if args.binary.open("rb").read(4)!=b"\x7fELF":parser.error("Native source probe must be actual linked ELF")
    source=raw.validate_source_snapshot(args.creator_source_manifest.resolve(strict=True))["source_sha256"]
    args.identity_cache={};args.output.mkdir(parents=True,exist_ok=False)
    command=[str(args.binary),"--device",args.device,"--"+args.role,"--replays",str(args.replays),"--handshake"]
    result=common.run_one(args,args.role,1,child_command=command,expected_schema=SCHEMA,record_validator=validate_records,
        checkpoint_validator=lambda row:validate_checkpoint(row,args.device,source))
    for path in (args.output/args.role).glob("checkpoint-*.json"):
        checkpoint=json.loads(path.read_text())
        if not checkpoint["external_device"].get("supported"):raise ValueError("Missing external metadata-only native observation")
        child=checkpoint["child"]
        for provider in child.get("closure",{}).get("native_blas_providers",[]):
            observed=checkpoint["mapped_library_identities"].get(provider["path"],{})
            if observed.get("sha256")!=provider["sha256"]:raise ValueError("Native provider receipt differs from actual externally observed mapped ELF")
    (args.output/"source-probe-result.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
