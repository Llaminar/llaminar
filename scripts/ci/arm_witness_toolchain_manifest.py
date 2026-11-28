#!/usr/bin/env python3
# @file arm_witness_toolchain_manifest.py
# @brief Capture and authenticate the exact ARM64 native witness toolchain closure.
#
# The NEON register witness proof binds the SHA-256 of every compiler program
# (GNU driver, cc1plus, as), the exact symlink/alternatives chain that selects
# them, and the system headers of the dependency closure. The test-runner image
# intentionally omits the ~20 GB toolchain; it therefore cannot reproduce that
# closure through a later apt install, because Ubuntu publishes security
# updates (binutils, gcc) between the cached toolchain context build and the
# test-runner build. The proof then correctly, but uselessly, reports a
# compiler program change.
#
# Instead the builder stages its own witness closure under
# /src/witness-toolchain (exact bytes and symlink/alternatives shape) and this
# tool records every entry in a manifest. The test-runner copies the staged
# tree to the root and verifies it against the manifest: a drifted, retargeted
# or missing program fails the IMAGE build with a precise diagnostic, before
# the sealed tests run. Run `capture` in the builder stage and `verify` in the
# arm64 test-runner stage.
#
# The manifest is byte-bound to the builder that compiled the witness; it
# does not claim any compiler version is good, and it never bypasses the
# proof's own program authentication.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = 1
TOOL = "arm-witness-toolchain-manifest"
_CHUNK = 1 << 20


def _digest(path: Path) -> str:
    """SHA-256 of one file, streamed in bounded chunks.

    The proof side already hashes the same bytes; keeping one chunked reader
    here avoids materializing multi-hundred-MB header trees in memory.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def capture(root: Path) -> dict:
    """Record every file and symlink under the staged closure root.

    Directories are implied by their entries. Symlinks are recorded by their
    link text (never followed), because the proof authenticates the exact
    symlink chain the driver resolves. Deterministic ordering keeps repeated
    captures byte-identical.
    """
    if not root.is_dir():
        raise ValueError(f"closure root is not a directory: {root}")
    entries: dict[str, dict] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            link = path.lstat().st_mode & 0o170000
            if path.is_symlink():
                entries[rel] = {"type": "symlink", "target": os.readlink(path)}
            elif link == 0o100000:
                entries[rel] = {"type": "file", "sha256": _digest(path)}
            else:
                entries[rel] = {"type": "special", "mode": oct(path.lstat().st_mode & 0o7777)}
    if not entries:
        raise ValueError(f"closure root is empty: {root}")
    return {"schema": SCHEMA, "tool": TOOL, "entries": entries}


def verify(root: Path, manifest: dict) -> None:
    """Authenticate the installed closure against a captured manifest.

    Only the manifest entries are inspected (the image contains many unrelated
    files); each must exist with the same kind, symlink target or content hash.
    Any difference raises ValueError naming the offending entry, so image
    builds fail closed with the exact drift visible.
    """
    if manifest.get("schema") != SCHEMA or manifest.get("tool") != TOOL:
        raise ValueError("foreign witness toolchain manifest schema or tool")
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("witness toolchain manifest has no entries")
    problems: list[str] = []
    for rel, expected in sorted(entries.items()):
        path = root / rel
        if not isinstance(expected, dict) or expected.get("type") not in (None, "file", "symlink", "special"):
            problems.append(f"{rel}: malformed record")
            continue
        try:
            stat = path.lstat()
        except FileNotFoundError:
            problems.append(f"{rel}: missing")
            continue
        kind = expected["type"]
        if path.is_symlink():
            if kind != "symlink":
                problems.append(f"{rel}: expected {kind}, found symlink")
                continue
            actual_target = os.readlink(path)
            if actual_target != expected.get("target"):
                problems.append(f"{rel}: symlink retargeted {actual_target!r} != {expected.get('target')!r}")
        elif stat.st_mode & 0o170000 == 0o100000:
            if kind != "file":
                problems.append(f"{rel}: expected {kind}, found file")
                continue
            if _digest(path) != expected.get("sha256"):
                problems.append(f"{rel}: file content changed")
        elif kind != "special" or oct(stat.st_mode & 0o7777) != expected.get("mode"):
            problems.append(f"{rel}: special file changed")
    if problems:
        shown = "\n  ".join(problems[:40])
        raise ValueError(f"witness toolchain closure differs from its manifest:\n  {shown}")


def main(argv: list[str] | None = None) -> int:
    """CLI: capture a staged closure or verify an installed one."""
    parser = argparse.ArgumentParser(
        description="Capture or verify the byte-exact ARM64 witness toolchain closure.")
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("--root", type=Path, required=True,
                        help="staged closure root (capture) or filesystem root (verify)")
    parser.add_argument("--out", type=Path, help="manifest output path (capture)")
    parser.add_argument("--manifest", type=Path, help="manifest input path (verify)")
    args = parser.parse_args(argv)
    try:
        if args.mode == "capture":
            if not args.out:
                parser.error("capture requires --out")
            args.out.write_text(json.dumps(capture(args.root), indent=2, sort_keys=True) + "\n")
            print(f"[witness-toolchain] captured {args.root} to {args.out}", flush=True)
            return 0
        if not args.manifest:
            parser.error("verify requires --manifest")
        verify(args.root, json.loads(args.manifest.read_text()))
        print(f"[witness-toolchain] closure at {args.root} matches {args.manifest}", flush=True)
        return 0
    except (ValueError, OSError) as error:
        print(f"[witness-toolchain] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
