#!/usr/bin/env python3
"""Validate and resume immutable NativeVNNI paired-refinement generations.

The paired planner publishes one numbered report and, for a pending result,
one or more content-addressed request shards.  Collection may be interrupted
between any two shards, and a later implementation repair may legitimately
continue after a terminal zero-request diagnostic generation.  This module
owns that lifecycle so shell drivers do not infer state merely from whether a
shard directory happens to contain a manifest.

Completed pending generations contribute all of their evidence atomically.
An incomplete pending generation is returned for collection without rerunning
the fit that authored its immutable requests.  A green generation is terminal
for refinement, while a failed diagnostic generation may be followed by a new
generation after the underlying planner or evidence contract is repaired.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TERMINAL_DIAGNOSTIC_STATUSES = frozenset({
    "confirmed_failure",
    "evidence_conflict",
    "insufficient_cross_validation",
})
REPORT_PATTERN = re.compile(r"^iteration-(\d+)\.report\.json$")


@dataclass(frozen=True)
class PairedRefinementHistory:
    """One validated resume decision and its complete prior evidence."""

    mode: str
    iteration: int
    evidence_paths: tuple[Path, ...]


def _read_object(path: Path) -> Mapping[str, object]:
    """Read one required JSON object with path-rich diagnostics."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read paired refinement artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"paired refinement artifact is not an object: {path}")
    return payload


def _required_string(payload: Mapping[str, object], field: str, path: Path) -> str:
    """Return one non-empty string field from a refinement artifact."""

    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: field {field!r} must be a non-empty string")
    return value


def _required_nonnegative_int(
    payload: Mapping[str, object], field: str, path: Path
) -> int:
    """Return one strict non-negative integer field."""

    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: field {field!r} must be a non-negative integer")
    return value


def _request_count(path: Path) -> int:
    """Return the exact request cardinality from a planner or shard JSON."""

    payload = _read_object(path)
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{path}: field 'requests' must be an array")
    return len(requests)


def _report_indices(paired_dir: Path) -> tuple[int, ...]:
    """Discover report generations while rejecting ambiguous filenames."""

    indices = []
    if not paired_dir.exists():
        return ()
    for path in paired_dir.glob("iteration-*.report.json"):
        match = REPORT_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid paired refinement report name: {path}")
        indices.append(int(match.group(1)))
    return tuple(sorted(indices))


def scan_paired_refinement_history(paired_dir: Path) -> PairedRefinementHistory:
    """Validate a refinement history and return its deterministic resume state.

    Modes are ``plan`` for a fresh fit generation, ``collect`` for an existing
    incomplete pending generation, and ``green`` when refinement has already
    converged.  Evidence paths contain only complete *earlier* pending
    generations; partially collected current shards never leak into a refit.
    """

    paired_dir = Path(paired_dir)
    report_indices = _report_indices(paired_dir)
    report_index_set = frozenset(report_indices)
    retained_evidence: list[Path] = []
    iteration = 0

    while True:
        tag = f"{iteration:03d}"
        report_path = paired_dir / f"iteration-{tag}.report.json"
        if iteration not in report_index_set:
            later = tuple(index for index in report_indices if index > iteration)
            if later:
                raise ValueError(
                    "paired refinement history has a report gap before "
                    f"iteration {later[0]:03d}"
                )
            return PairedRefinementHistory(
                mode="plan",
                iteration=iteration,
                evidence_paths=tuple(retained_evidence),
            )

        report = _read_object(report_path)
        status = _required_string(report, "status", report_path)
        declared_requests = _required_nonnegative_int(
            report, "request_count", report_path
        )
        requests_path = paired_dir / f"iteration-{tag}.requests.json"
        shard_dir = paired_dir / f"iteration-{tag}.shards"
        manifests = tuple(sorted(shard_dir.glob("shard-*.requests.json")))

        if status == "pending":
            if declared_requests == 0:
                raise ValueError(f"{report_path}: pending generation has no requests")
            observed_requests = _request_count(requests_path)
            if observed_requests != declared_requests:
                raise ValueError(
                    f"{requests_path}: request count {observed_requests} does not "
                    f"match report count {declared_requests}"
                )
            if not manifests:
                raise ValueError(f"{report_path}: pending generation has no shards")
            sharded_requests = sum(_request_count(path) for path in manifests)
            if sharded_requests != declared_requests:
                raise ValueError(
                    f"{shard_dir}: shard request count {sharded_requests} does not "
                    f"match report count {declared_requests}"
                )
            generation_evidence = tuple(
                path.with_suffix("").with_suffix(".csv") for path in manifests
            )
            if not all(path.is_file() and path.stat().st_size > 0 for path in generation_evidence):
                return PairedRefinementHistory(
                    mode="collect",
                    iteration=iteration,
                    evidence_paths=tuple(retained_evidence),
                )
            retained_evidence.extend(generation_evidence)
            iteration += 1
            continue

        if status == "green":
            if declared_requests != 0:
                raise ValueError(f"{report_path}: green generation contains requests")
            if manifests:
                raise ValueError(f"{report_path}: green generation published request shards")
            later = tuple(index for index in report_indices if index > iteration)
            if later:
                raise ValueError(
                    f"{report_path}: green generation is followed by iteration "
                    f"{later[0]:03d}"
                )
            return PairedRefinementHistory(
                mode="green",
                iteration=iteration,
                evidence_paths=tuple(retained_evidence),
            )

        if status in TERMINAL_DIAGNOSTIC_STATUSES:
            if declared_requests != 0 or manifests:
                raise ValueError(
                    f"{report_path}: terminal diagnostic generation contains work"
                )
            iteration += 1
            continue

        raise ValueError(f"{report_path}: unknown paired refinement status {status!r}")


def main() -> int:
    """Print a shell-friendly or JSON representation of the resume state."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_dir", type=Path)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    args = parser.parse_args()
    history = scan_paired_refinement_history(args.paired_dir)
    if args.format == "lines":
        print(f"{history.mode}\t{history.iteration}")
        for path in history.evidence_paths:
            rendered = str(path)
            if "\n" in rendered or "\r" in rendered:
                raise ValueError("paired evidence path contains a newline")
            print(rendered)
        return 0
    print(json.dumps({
        "mode": history.mode,
        "iteration": history.iteration,
        "evidence_paths": [str(path) for path in history.evidence_paths],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
