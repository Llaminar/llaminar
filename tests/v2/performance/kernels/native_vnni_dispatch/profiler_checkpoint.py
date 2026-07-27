#!/usr/bin/env python3
"""Recover immutable profiler transactions from interrupted journals.

Long NativeVNNI profiler sweeps append one authenticated evidence record after
each completed request.  The collector normally turns that journal into a
single evidence manifest only after every request reaches a terminal state.
This module preserves useful work when the timing corpus is regenerated before
that final publication: it publishes the completed journal members as a new,
self-contained subset transaction without rewriting any request or evidence
record.

Recovery is deliberately based on exact physical launch identity.  Already
published request manifests may be supplied as coverage, in which case an
overlapping journal member is omitted.  A changed arithmetic fingerprint,
schedule, workspace, prepared codebook, execution mode, or M/N/K geometry is a
different launch and therefore cannot inherit old counters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .candidate_observation import read_observation_csv, write_observation_csv
from .corpus import ObservationCorpus
from .profiler_collectors import _load_incremental_evidence
from .profiler_evidence import (
    ProfilerCoverageReport,
    ProfilerEvidenceManifest,
    ProfilerEvidenceStatus,
    ProfilerRequestManifest,
    _profiled_exact_launch_key,
    read_profiler_request_coverage_keys,
    read_profiler_request_manifest,
    validate_profiler_evidence_coverage,
    write_profiler_evidence_manifest,
    write_profiler_request_manifest,
)


@dataclass(frozen=True)
class RecoveredCheckpointReport:
    """Describe one authenticated interrupted-journal recovery."""

    source_record_count: int
    terminal_record_count: int
    covered_record_count: int
    recovered_record_count: int
    request_manifest_digest: str | None
    evidence_manifest_digest: str | None


def recover_complete_checkpoint(
    *,
    source_observations: Path,
    source_requests: Path,
    source_evidence: Path,
    covered_requests: Sequence[Path],
    output_observations: Path,
    output_requests: Path,
    output_evidence: Path,
) -> RecoveredCheckpointReport:
    """Publish uncovered terminal journal records as one complete transaction.

    The source request manifest and every complete newline in its journal are
    authenticated by the ordinary collector resume reader.  Successful and
    explicitly unsupported terminal records retain their original request ID,
    observation digest, command digest, raw-artifact digest, and counter data.
    Failed or missing requests remain obligations for the next collector run.
    """

    requests = read_profiler_request_manifest(source_requests)
    observations = read_observation_csv((source_observations,))
    if observations.digest() != requests.corpus_digest:
        raise ValueError(
            "profiler checkpoint observations belong to another request corpus"
        )
    records = _load_incremental_evidence(
        source_evidence,
        requests,
        resume=True,
    )
    terminal_statuses = {
        ProfilerEvidenceStatus.COMPLETE,
        ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
    }
    terminal = {
        request_id: evidence
        for request_id, evidence in records.items()
        if evidence.status in terminal_statuses
    }
    covered_keys: set[tuple[object, ...]] = set()
    for path in covered_requests:
        covered_keys.update(read_profiler_request_coverage_keys(path))

    recovered_requests = tuple(
        request
        for request in requests.requests
        if request.request_id in terminal
        and _profiled_exact_launch_key(request) not in covered_keys
    )
    covered_count = len(terminal) - len(recovered_requests)
    if not recovered_requests:
        return RecoveredCheckpointReport(
            source_record_count=len(records),
            terminal_record_count=len(terminal),
            covered_record_count=covered_count,
            recovered_record_count=0,
            request_manifest_digest=None,
            evidence_manifest_digest=None,
        )

    observations_by_digest = {
        observation.digest(): observation for observation in observations
    }
    recovered_observations = ObservationCorpus(tuple(
        observations_by_digest[request.observation_digest]
        for request in recovered_requests
    ))
    recovered_request_manifest = ProfilerRequestManifest(
        corpus_digest=recovered_observations.digest(),
        candidate_registry_digest=requests.candidate_registry_digest,
        requests=recovered_requests,
        learner_version=requests.learner_version,
        feature_schema_version=requests.feature_schema_version,
        schema_version=requests.schema_version,
    )
    recovered_evidence_manifest = ProfilerEvidenceManifest(
        request_manifest_digest=recovered_request_manifest.digest(),
        corpus_digest=recovered_request_manifest.corpus_digest,
        candidate_registry_digest=recovered_request_manifest.candidate_registry_digest,
        evidence=tuple(
            terminal[request.request_id] for request in recovered_requests
        ),
        collector_version=next(iter(terminal.values())).collector_version,
    )
    coverage: ProfilerCoverageReport = validate_profiler_evidence_coverage(
        recovered_request_manifest,
        recovered_evidence_manifest,
        require_complete=True,
    )
    if not coverage.complete:
        raise AssertionError("recovered profiler transaction is incomplete")

    output_observations.parent.mkdir(parents=True, exist_ok=True)
    output_requests.parent.mkdir(parents=True, exist_ok=True)
    output_evidence.parent.mkdir(parents=True, exist_ok=True)
    write_observation_csv(
        output_observations,
        recovered_observations.observations,
    )
    write_profiler_request_manifest(
        output_requests,
        recovered_request_manifest,
    )
    write_profiler_evidence_manifest(
        output_evidence,
        recovered_evidence_manifest,
    )
    return RecoveredCheckpointReport(
        source_record_count=len(records),
        terminal_record_count=len(terminal),
        covered_record_count=covered_count,
        recovered_record_count=len(recovered_requests),
        request_manifest_digest=recovered_request_manifest.digest(),
        evidence_manifest_digest=recovered_evidence_manifest.digest(),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the interrupted-checkpoint recovery command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-observations", required=True)
    parser.add_argument("--source-requests", required=True)
    parser.add_argument("--source-evidence", required=True)
    parser.add_argument("--covered-requests", action="append", default=[])
    parser.add_argument("--output-observations", required=True)
    parser.add_argument("--output-requests", required=True)
    parser.add_argument("--output-evidence", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Recover one journal and print its machine-readable disposition."""

    args = build_argument_parser().parse_args(argv)
    report = recover_complete_checkpoint(
        source_observations=Path(args.source_observations),
        source_requests=Path(args.source_requests),
        source_evidence=Path(args.source_evidence),
        covered_requests=tuple(Path(path) for path in args.covered_requests),
        output_observations=Path(args.output_observations),
        output_requests=Path(args.output_requests),
        output_evidence=Path(args.output_evidence),
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
