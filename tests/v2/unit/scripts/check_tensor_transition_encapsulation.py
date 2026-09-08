#!/usr/bin/env python3
"""Reject direct tensor coherence mutation outside its owning subsystem.

TensorBase owns coherence state, while TransferEngine owns the public movement
and publication protocol. Direct calls to ``transitionTo()`` or
``transitionToWithEvent()`` let callers split byte movement, event ordering, and
state publication across unrelated files. Raw completion-event clearing also
discarded backend-owned handles and could leak cross-backend resources. These
were the sources of stale-host, eventless-device-authority, and lifecycle bugs.

Only TransferEngine may initiate raw transitions or publish external host
writes. TensorSlice is allowed to delegate the private virtual hooks to its
wrapped tensor; it does not expose a public mutation API. Every other C++
source, including test and performance harnesses, must use a typed
TransferEngine publication method.
"""

from __future__ import annotations

import argparse
import bisect
import pathlib
import re
import sys
from collections.abc import Iterable

from check_gpu_blocking_sync_policy import (
    SOURCE_SUFFIXES,
    strip_comments_and_literals,
)


SCANNED_ROOTS = (
    "src/v2",
    "tests/v2",
)

ALLOWED_FILES = frozenset(
    {
        "src/v2/tensors/TensorBase.cpp",
        "src/v2/tensors/TensorClasses.h",
        "src/v2/tensors/TensorSlice.h",
        "src/v2/transfer/TransferEngine.cpp",
    }
)

RAW_COHERENCE_MUTATION_PATTERN = re.compile(
    r"\b("
    r"(?:transitionTo(?:WithEvent)?)"
    r"|mark_host_dirty"
    r"|clearCompletionEvent"
    r"|publishDeviceWriteStateWithEvent"
    r"|publishGraphOwnedDeviceWriteState"
    r"|publishHostWriteState"
    r"|publishSynchronizedState"
    r")\s*\("
)


def source_files(repo_root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield policy-covered C++ files in deterministic order."""

    for relative_root in SCANNED_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def line_number(source: str, position: int) -> int:
    """Return the one-based line containing a lexical match."""

    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(source) if char == "\n")
    return bisect.bisect_right(starts, position)


def scan_file(repo_root: pathlib.Path, path: pathlib.Path) -> list[str]:
    """Return direct-mutation diagnostics for one source file."""

    relative_path = path.relative_to(repo_root).as_posix()
    if relative_path in ALLOWED_FILES:
        return []

    source = path.read_text(encoding="utf-8", errors="replace")
    if RAW_COHERENCE_MUTATION_PATTERN.search(source) is None:
        return []

    # Most translation units contain none of the guarded API names. Keep that
    # overwhelmingly common path inside the C regex engine and run the exact
    # comment/string lexer only for candidate files. This preserves identical
    # diagnostics while avoiding a Python character-by-character pass over the
    # complete source tree during every unit gate.
    source = strip_comments_and_literals(source)
    return [
        (
            f"{relative_path}:{line_number(source, match.start())}: direct "
            f"{match.group(1)}() bypasses TransferEngine; use a typed "
            "publishDeviceWrite(), publishGraphOwnedDeviceWrite(), "
            "publishHostWrite(), or publishSynchronized() contract"
        )
        for match in RAW_COHERENCE_MUTATION_PATTERN.finditer(source)
    ]


def validate(repo_root: pathlib.Path) -> list[str]:
    """Return all policy violations under the repository root."""

    return [
        failure
        for path in source_files(repo_root)
        for failure in scan_file(repo_root, path)
    ]


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
    """Run the encapsulation policy and print actionable diagnostics."""

    args = parse_args()
    failures = validate(args.repo_root.resolve())
    if failures:
        print("Tensor coherence encapsulation violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Tensor coherence encapsulation policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
