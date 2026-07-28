#!/usr/bin/env python3
"""Reject eventless GPU-authority publication after a blocking boundary.

Asynchronous GPU tensor writes are ordered by a completion event recorded on
the exact producer stream. A blocking stream/device synchronization followed by
an eventless authority publication is not an equivalent protocol: the
synchronization stalls the host, while flags-only publication then discards the
producer dependency needed by later device consumers.

This source sanitizer resolves both operations to their enclosing C++ caller.
It rejects the sequence even when the two calls are separated by branches,
comments, or unrelated statements. Dedicated unit tests that exercise the
coherence state machine are outside the scanned roots; production, integration,
and performance sources receive the same policy. Both the retired raw
``transitionTo(DEVICE_AUTHORITATIVE)`` spelling and the encapsulated graph-owned
publication APIs are recognized so an API cleanup cannot accidentally disable
the architectural check.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import dataclasses
import os
import pathlib
import re
import sys
from collections.abc import Iterable

from check_gpu_blocking_sync_policy import (
    SOURCE_SUFFIXES,
    SYNC_PATTERNS,
    function_intervals,
    strip_comments_and_literals,
)


SCANNED_ROOTS = (
    "src/v2",
    "tests/v2/integration",
    "tests/v2/performance",
)

EVENTLESS_PUBLICATION_PATTERN = re.compile(
    r"""
    (?:
        (?:->|\.)
        transitionTo
        \s*\(
        \s*
        (?:[A-Za-z_][A-Za-z0-9_:]*::)?
        DEVICE_AUTHORITATIVE
        \b
      |
        (?:->|\.|::)
        (?:
            publishGraphOwnedDeviceWrite
          | publishGraphOwnedCurrentDeviceWrite
          | markWrittenFlagsOnly
        )
        \s*\(
      |
        \b markOutputsDirtyFlagsOnly \s*\(
    )
    """,
    re.VERBOSE,
)

KERNEL_PUBLICATION_PATTERN = re.compile(
    r"""
    \b
    (?:[A-Za-z_][A-Za-z0-9_]*::)*
    TransferEngine::
    (?P<method>publishDeviceWrite|publishCurrentDeviceWrite)
    \s*\(
    """,
    re.VERBOSE,
)

GPU_PUBLICATION_ROOTS = SCANNED_ROOTS

COMPLETED_PUBLICATION_PATTERN = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*::)*TransferEngine::publishCompletedDeviceWrite\s*\("
)

COMPLETED_PUBLICATION_ALLOWLIST = (
    "src/v2/transfer/TransferEngine.cpp",
    "src/v2/utils/MPIStager.cpp",
    "src/v2/execution/local_execution/coherence/CrossDomainTransfer.cpp",
    "src/v2/execution/local_execution/collective/CollectiveContext.cpp",
    "src/v2/collective/LocalTPContext.cpp",
)


@dataclasses.dataclass(frozen=True, order=True)
class Operation:
    """One caller-attributed synchronization or publication operation."""

    position: int
    line: int
    scope_start: int
    scope_end: int
    caller: str
    kind: str


@dataclasses.dataclass(frozen=True, order=True)
class Violation:
    """One eventless publication ordered after an earlier blocking operation."""

    path: str
    caller: str
    synchronization_kind: str
    synchronization_line: int
    publication_line: int


@dataclasses.dataclass(frozen=True, order=True)
class KernelPublicationViolation:
    """One asynchronous GPU publication with invalid stream provenance."""

    path: str
    line: int
    reason: str


def source_files(repo_root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield all policy-covered C++ source files in deterministic order."""

    for relative_root in SCANNED_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def line_starts(source: str) -> list[int]:
    """Return character offsets used to map lexical matches to source lines."""

    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(source) if char == "\n")
    return starts


def line_number(starts: list[int], position: int) -> int:
    """Map a source character offset to a one-based line number."""

    return bisect.bisect_right(starts, position)


def call_arguments(source: str, open_paren: int) -> list[str] | None:
    """Split one already-sanitized C++ call into top-level arguments.

    The policy only needs lexical structure, not a complete C++ grammar.
    Tracking nested delimiters is nevertheless essential because tensor,
    device, and stream expressions may contain function calls or template
    initializers whose commas must not be mistaken for argument separators.
    """

    matching = {"(": ")", "[": "]", "{": "}"}
    stack = ["("]
    argument_start = open_paren + 1
    arguments: list[str] = []

    for index in range(open_paren + 1, len(source)):
        char = source[index]
        if char in matching:
            stack.append(char)
            continue
        if char in matching.values():
            if not stack or matching[stack[-1]] != char:
                return None
            stack.pop()
            if not stack:
                final = source[argument_start:index].strip()
                if final or arguments:
                    arguments.append(final)
                return arguments
            continue
        if char == "," and len(stack) == 1:
            arguments.append(source[argument_start:index].strip())
            argument_start = index + 1

    return None


def scan_gpu_kernel_publications(
    repo_root: pathlib.Path,
    path: pathlib.Path,
    raw_source: str | None = None,
) -> list[KernelPublicationViolation]:
    """Reject implicit/default-stream publication from GPU kernel code.

    Asynchronous GPU producers in this tree launch on explicit stage-owned
    streams. Publishing their output on an implicit backend stream records an
    unrelated event and lets a cross-stream consumer observe incomplete bytes.
    Requiring the stream argument throughout production and GPU-bearing tests
    makes launch and publication provenance visibly inseparable. The
    explicit-device spelling carries ``(tensor, device, stream)`` while the
    current-device spelling carries ``(tensor, stream)``.
    """

    relative_path = path.relative_to(repo_root).as_posix()
    if not any(
        relative_path == root or relative_path.startswith(root + "/")
        for root in GPU_PUBLICATION_ROOTS
    ):
        return []

    if raw_source is None:
        raw_source = path.read_text(encoding="utf-8", errors="replace")
    source = strip_comments_and_literals(raw_source)
    starts = line_starts(source)
    violations: list[KernelPublicationViolation] = []

    for match in KERNEL_PUBLICATION_PATTERN.finditer(source):
        method = match.group("method")
        expected_arguments = 3 if method == "publishDeviceWrite" else 2
        stream_index = expected_arguments - 1
        open_paren = source.find("(", match.start(), match.end())
        arguments = call_arguments(source, open_paren)
        if arguments is None:
            violations.append(
                KernelPublicationViolation(
                    relative_path,
                    line_number(starts, match.start()),
                    "could not parse publication arguments",
                )
            )
            continue
        if len(arguments) < expected_arguments:
            violations.append(
                KernelPublicationViolation(
                    relative_path,
                    line_number(starts, match.start()),
                    f"{method} omits the exact producer stream",
                )
            )
            continue
        stream_argument = arguments[stream_index].strip()
        if stream_argument in {"nullptr", "NULL", "0"}:
            violations.append(
                KernelPublicationViolation(
                    relative_path,
                    line_number(starts, match.start()),
                    f"{method} uses a null/default producer stream",
                )
            )

    if relative_path not in COMPLETED_PUBLICATION_ALLOWLIST:
        for match in COMPLETED_PUBLICATION_PATTERN.finditer(source):
            violations.append(
                KernelPublicationViolation(
                    relative_path,
                    line_number(starts, match.start()),
                    "uses completed-boundary publication outside an audited "
                    "synchronous transfer or collective owner",
                )
            )

    return violations


def enclosing_scope(
    intervals: Iterable[tuple[int, int, str]],
    position: int,
    source_size: int,
) -> tuple[int, int, str]:
    """Return the exact enclosing body interval and its diagnostic name.

    Caller names alone are insufficient because each GoogleTest macro expands
    lexically as ``TEST`` or ``TEST_F``. The source interval keeps independent
    test bodies from sharing synchronization state merely because they have the
    same macro name.
    """

    candidates = [
        interval
        for interval in intervals
        if interval[0] < position < interval[1]
    ]
    if not candidates:
        return (0, source_size, "<global>")
    return min(candidates, key=lambda interval: interval[1] - interval[0])


def scan_file(
    repo_root: pathlib.Path,
    path: pathlib.Path,
    raw_source: str | None = None,
) -> list[Violation]:
    """Find blocking-sync-then-eventless-publication sequences in one file."""

    if raw_source is None:
        raw_source = path.read_text(encoding="utf-8", errors="replace")
    if not any(
        token in raw_source
        for token in (
            "DEVICE_AUTHORITATIVE",
            "publishGraphOwnedDeviceWrite",
            "publishGraphOwnedCurrentDeviceWrite",
            "markWrittenFlagsOnly",
            "markOutputsDirtyFlagsOnly",
        )
    ):
        return []

    source = strip_comments_and_literals(raw_source)
    intervals = function_intervals(source)
    starts = line_starts(source)

    synchronizations: list[Operation] = []
    for kind, pattern in SYNC_PATTERNS:
        for match in pattern.finditer(source):
            scope_start, scope_end, caller = enclosing_scope(
                intervals,
                match.start(),
                len(source),
            )
            synchronizations.append(
                Operation(
                    position=match.start(),
                    line=line_number(starts, match.start()),
                    scope_start=scope_start,
                    scope_end=scope_end,
                    caller=caller,
                    kind=kind,
                )
            )

    publications: list[Operation] = []
    for match in EVENTLESS_PUBLICATION_PATTERN.finditer(source):
        scope_start, scope_end, caller = enclosing_scope(
            intervals,
            match.start(),
            len(source),
        )
        publications.append(
            Operation(
                position=match.start(),
                line=line_number(starts, match.start()),
                scope_start=scope_start,
                scope_end=scope_end,
                caller=caller,
                kind="eventless_publication",
            )
        )

    relative_path = path.relative_to(repo_root).as_posix()
    violations: list[Violation] = []
    for publication in publications:
        preceding = [
            synchronization
            for synchronization in synchronizations
            if synchronization.scope_start == publication.scope_start
            and synchronization.scope_end == publication.scope_end
            and synchronization.position < publication.position
        ]
        if not preceding:
            continue

        nearest = max(preceding, key=lambda operation: operation.position)
        violations.append(
            Violation(
                path=relative_path,
                caller=publication.caller,
                synchronization_kind=nearest.kind,
                synchronization_line=nearest.line,
                publication_line=publication.line,
            )
        )
    return violations


def physical_core_count() -> int:
    """Return a conservative physical-core ceiling for sanitizer workers."""

    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return max(1, affinity_count)

    physical_cores: set[tuple[str, str]] = set()
    physical_id = ""
    core_id = ""
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if not line:
            if physical_id and core_id:
                physical_cores.add((physical_id, core_id))
            physical_id = ""
            core_id = ""
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "physical id":
            physical_id = value.strip()
        elif key.strip() == "core id":
            core_id = value.strip()

    if not physical_cores:
        return max(1, affinity_count)
    return max(1, min(affinity_count, len(physical_cores)))


def scan_policy_path(
    repo_root: pathlib.Path,
    path: pathlib.Path,
) -> tuple[list[Violation], list[KernelPublicationViolation]]:
    """Read one source once and run both independent publication policies."""

    raw_source = path.read_text(encoding="utf-8", errors="replace")
    return (
        scan_file(repo_root, path, raw_source),
        scan_gpu_kernel_publications(repo_root, path, raw_source),
    )


def scan_policy_path_task(
    task: tuple[pathlib.Path, pathlib.Path],
) -> tuple[list[Violation], list[KernelPublicationViolation]]:
    """Pickle-friendly process-pool entry point for one source path."""

    return scan_policy_path(*task)


def validate(repo_root: pathlib.Path) -> list[str]:
    """Return actionable diagnostics for every forbidden publication sequence."""

    paths = list(source_files(repo_root))
    max_workers = min(8, physical_core_count(), max(1, len(paths)))
    if len(paths) < 32 or max_workers == 1:
        scan_results = [
            scan_policy_path(repo_root, path)
            for path in paths
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            scan_results = list(
                executor.map(
                    scan_policy_path_task,
                    ((repo_root, path) for path in paths),
                    chunksize=8,
                )
            )

    violations = [
        violation
        for file_violations, _ in scan_results
        for violation in file_violations
    ]
    failures = [
        (
            f"{violation.path}:{violation.publication_line} in "
            f"{violation.caller}: eventless DEVICE_AUTHORITATIVE publication "
            f"follows {violation.synchronization_kind} blocking operation at "
            f"line {violation.synchronization_line}; publish the producer "
            "event with TransferEngine::publishDeviceWrite() and order "
            "consumers with a stream wait"
        )
        for violation in sorted(violations)
    ]
    kernel_violations = [
        violation
        for _, file_violations in scan_results
        for violation in file_violations
    ]
    failures.extend(
        (
            f"{violation.path}:{violation.line}: GPU publication "
            f"{violation.reason}; pass the exact explicit "
            "stream that enqueued the write"
        )
        for violation in sorted(kernel_violations)
    )
    return failures


def parse_args() -> argparse.Namespace:
    """Parse the repository root used by CTest and local audits."""

    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=default_root,
        help="Repository root containing src/v2 (default: inferred)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the policy and report every caller that violates event ordering."""

    args = parse_args()
    failures = validate(args.repo_root.resolve())
    if failures:
        print(
            "GPU eventless publication policy violations:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("GPU eventless publication policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
