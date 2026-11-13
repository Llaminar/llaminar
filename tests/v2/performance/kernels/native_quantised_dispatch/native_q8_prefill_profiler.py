"""Bind native-Q8 dense profiling to sealed original full-projection timings.

This is an additive operation of the existing profiler request/collector owner.
The measured winner and the actual unforced Auto baseline retain independent
requests on every original source alias. Profiling never supplies timing data.
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from .corpus_provenance import mapping_digest, sha256_file
from .gpu_architecture import GPUArchitecture
from .profiler_evidence import (
    PROFILE_PROTOCOL, ProfilerRequest,
    ProfilerRequestManifest, ProfilerEvidenceStatus, read_profiler_request_manifest,
    read_profiler_evidence_manifest, validate_profiler_evidence_coverage,
    write_profiler_request_manifest,
)
from .schema import Backend, ExecutionMode, SemanticContract

OPERATION = "ROCmNativeQ8DensePrefill"
SCHEMA = "rocm-native-q8-dense-prefill-profile-v1"


def validate_request(request: ProfilerRequest) -> None:
    """Reject relabeled decode, expanded-Q8 and forced-Auto profile requests."""
    config = request.config_json
    cell = config.get("cell", {})
    launch = config.get("launch", ())
    if (request.backend != Backend.ROCM or request.operation_kind != OPERATION
            or config.get("schema") != SCHEMA
            or config.get("candidate_id") != request.effective_candidate_id
            or config.get("requested_candidate") not in {request.effective_candidate_id, "Auto"}
            or cell.get("prepared_family") != "native_quantised"
            or cell.get("gpu_architecture") != request.architecture_class
            or (cell.get("m"), cell.get("n"), cell.get("k"), cell.get("source_format"), cell.get("shape")) !=
               (request.m, request.aggregate_n, request.k, request.source_format, request.shape_name)
            or request.runtime_codebook_id != 19 or len(launch) != 6 or launch[5] != 0
            or not config.get("native_participant") or not config.get("timing_artifacts")
            or type(config.get("visibility_ordinal")) is not int or config["visibility_ordinal"] < 0
            or mapping_digest(config) != request.candidate_policy_hash):
        raise ValueError("native Q8 profiler requires its complete original physical-operation witness")
    GPUArchitecture.parse("rocm", request.architecture_class)


def build_manifest(root: Path, *, warmups: int = 5, samples: int = 30) -> ProfilerRequestManifest:
    """Project complete authenticated source cells into winner/baseline obligations."""
    from . import production_dense_prefill_sweep as sweep, dense_production_overlay as owner
    cells, plan = sweep.load_planned_dense_prefill_cells(root / sweep.SWEEP_PLAN_FILENAME, "rocm")
    if any(cell.prepared_family != sweep.DensePreparedWeightFamily.NATIVE_QUANTISED or cell.gpu_architecture is None for cell in cells):
        raise ValueError("native Q8 profile owner cannot reinterpret a legacy or expanded corpus")
    entries = owner.collect_dense_overlay_entries(root, "rocm", cells=cells, warmup_runs=warmups, bench_runs=samples)
    winners = {entry.key: entry for entry in entries}
    trainer = plan["trainer_provenance"]
    requests = []
    for cell in cells:
        key = owner.DenseOverlayKey(19, cell.m, cell.shape.n, cell.shape.k,
            cell.effective_prepared_family, GPUArchitecture.parse("rocm", cell.gpu_architecture))
        winner = winners[key]
        paths = sweep.production_dense_prefill_cell_paths(root, cell)
        rows = sweep._read_aggregate_rows(paths.aggregate, cell)
        artifacts = {path.name: sha256_file(path) for path in (paths.aggregate, paths.timing, paths.manifest)}
        for purpose, requested, launch in (("winner", winner.candidate_id, winner.launch),
                                           ("baseline", "Auto", owner._launch_tuple("rocm", rows["Auto"]))):
            # Common requests always name a physical candidate. The original
            # request mode remains separate, so the baseline still executes Auto.
            concrete = sorted(candidate for candidate, row in rows.items()
                              if candidate != "Auto" and owner._launch_tuple("rocm", row) == launch)
            if not concrete:
                raise ValueError("profiled native Q8 physical launch lacks its sealed forced control")
            candidate = requested if requested != "Auto" else concrete[0]
            row = rows[requested]
            participant = {name: row[name] for name in ("gpu_uuid", "gpu_pci", "native_ordinal", "gpu_architecture")}
            config = dict(schema=SCHEMA, cell=cell.canonical_mapping(), purpose=purpose,
                candidate_id=candidate, requested_candidate=requested, launch=list(launch),
                native_participant=participant, timing_artifacts=artifacts,
                visibility_ordinal=json.loads(paths.manifest.read_text())["native_process"]["device_ordinal"])
            identity = mapping_digest(config)
            request = ProfilerRequest(
                request_id="native-q8-prefill-" + identity.removeprefix("sha256:"), observation_digest=identity,
                run_id=cell.cell_id, corpus_id=mapping_digest(artifacts), git_revision="unrecorded-in-original-trainer",
                build_id=mapping_digest(trainer), compiler_id="Release-executable-closure:" + trainer["binary_sha256"],
                backend=Backend.ROCM, architecture_class=cell.gpu_architecture,
                device_name="native-uuid:" + row["gpu_uuid"] + ";pci:" + row["gpu_pci"],
                driver_runtime="unrecorded-in-original-trainer;collector-retains-tool-version",
                threading_or_stream_mode="retained-exact-nondefault-stream", semantic_contract=SemanticContract.FAST,
                operation_kind=OPERATION, bundle_signature=identity, projection_n_vector=(cell.shape.n,),
                source_format=cell.source_format.label, source_codebook_id=cell.source_format.source_codebook_id,
                prepared_family_id="NativeQuantised", packing_abi="ROCm-native-Q8", runtime_codebook_id=19,
                shape_group_id=OPERATION, shape_name=cell.shape.name, execution_mode=ExecutionMode.GRAPH_CAPTURED,
                m=cell.m, aggregate_n=cell.shape.n, k=cell.shape.k, candidate_registry_surface=OPERATION,
                candidate_id=candidate, effective_candidate_id=candidate, candidate_family="NativeCooperative",
                config_json=config, schedule_signature=identity, workspace_signature="retained-native-dense-prefill-arena",
                prepared_resources=("NativeQuantised",), arithmetic_fingerprint="byte-proved-original-source:" + identity,
                candidate_policy_hash=identity, trial_set_hash=mapping_digest(artifacts), timing_sample_hash=mapping_digest(artifacts),
                supported=True, graph_capture_ok=True, forced_route_ok=True, observed_candidate_id=candidate,
                profile_protocol=PROFILE_PROTOCOL, target_launches_per_profiler_pass=1)
            request.validate()
            validate_request(request)
            requests.append(request)
    return ProfilerRequestManifest(corpus_digest=mapping_digest([r.observation_digest for r in requests]),
        candidate_registry_digest=mapping_digest([r.config_json for r in requests]), requests=tuple(requests))


def environment(request: ProfilerRequest, raw_dir: Path) -> dict[str, str]:
    """Lower one sealed original dense request into the same native trainer."""
    validate_request(request)
    config = request.config_json
    env = {name: value for name, value in os.environ.items() if not name.startswith((
        "LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_", "LLAMINAR_ROCM_NATIVE_QUANTISED_DECODE_"))}
    prefix = "LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_"
    env.update({prefix + "FORMATS": request.source_format, prefix + "SHAPES": request.shape_name,
        prefix + "M": str(request.m), prefix + "MAX_CASES": "1", prefix + "WARMUP": "5", prefix + "BENCH": "1",
        prefix + "PREPARED_FAMILY": "native_quantised", prefix + "ARCHITECTURE": request.architecture_class,
        "LLAMINAR_NATIVE_QUANTISED_PROFILE_REQUEST_ID": request.request_id,
        "LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_CANDIDATE": config["requested_candidate"],
        "LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_EXPECTED_LAUNCH": ",".join(map(str, config["launch"])),
        "LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_EXPECTED_UUID": config["native_participant"]["gpu_uuid"]})
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(config["visibility_ordinal"])
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("LLAMINAR_NATIVE_QUANTISED_TOURNAMENT", "LLAMINAR_NATIVE_QUANTISED_TOURNAMENT_RECEIPT_PATH",
                 "LLAMINAR_NATIVE_QUANTISED_PREFILL_ROW_POLICY"):
        env.pop(name, None)
    return env


def require_profile(root: Path, requests: Path, evidence: Path, *, warmups: int, samples: int) -> None:
    """Require complete current winner/actual-Auto evidence before new publication."""
    expected = build_manifest(root, warmups=warmups, samples=samples)
    actual = read_profiler_request_manifest(requests)
    if expected.canonical_mapping() != actual.canonical_mapping():
        raise ValueError("native Q8 profile requests do not bind the current sealed timing corpus")
    observed = read_profiler_evidence_manifest(evidence)
    # The reader authenticates the serialized root metric-set version; coverage
    # validates every typed record's version against its exact request below.
    validate_profiler_evidence_coverage(actual, observed, require_complete=True)
    if any(record.status != ProfilerEvidenceStatus.COMPLETE for record in observed.evidence):
        raise ValueError("native Q8 winner and actual Auto require complete separate profiling")
    for record in observed.evidence:
        for dispatch in record.dispatches:
            for metric in dispatch.metrics:
                if metric.metric_id in {"gpu.scratch_bytes", "gpu.local_memory_bytes_per_thread", "gpu.local_memory_spill_requests"} and metric.value is not None and metric.value > 0:
                    raise ValueError("native Q8 profile reports nonzero local/scratch traffic")


def main() -> int:
    """Write original common-profiler requests from committed native-Q8 cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--bench-runs", type=int, default=30)
    args = parser.parse_args()
    write_profiler_request_manifest(args.output, build_manifest(args.work_dir, warmups=args.warmup_runs, samples=args.bench_runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
