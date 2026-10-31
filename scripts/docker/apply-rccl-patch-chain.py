#!/usr/bin/env python3
"""Validate a complete canonical RCCL patch transition before changing its source.

Only Git-parsed affected paths are copied into a disposable index. Exact current
patches are reversed there in reverse order, and the complete desired chain is
validated/applied there. Fresh, current and explicitly ordered predecessor
generations are supported. Unknown/partial states fail before any real write. A single
whole-diff Git application preserves unrelated files and non-conflicting edits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile


class PatchChainError(RuntimeError):
    """An unsupported source state or failed validation; no alternate patch mode."""


@dataclass(frozen=True)
class FileState:
    """Exact affected-file content/mode; absent files have no copied root."""
    digest: str
    mode: int


def git(directory, arguments, *, data=None, check=True):
    """Run the actual Git parser/applicator without inherited index/worktree overrides."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
                             "-c", "core.fsmonitor=false", "-C", str(directory), *arguments],
                            input=data, capture_output=True, env=environment)
    if check and result.returncode:
        raise PatchChainError(result.stderr.decode(errors="replace").strip()
                              or "Git rejected the canonical patch transition")
    return result


def patch_paths(patch):
    """Use Git's NUL-delimited path inventory; reject unsupported path/patch forms."""
    # Git otherwise filters paths by the caller's repository prefix when this
    # script lives under scripts/docker. Parse from an explicit private root.
    with tempfile.TemporaryDirectory(prefix="llaminar-rccl-patch-paths-") as directory:
        parser_root = Path(directory)
        git(parser_root, ["init", "--quiet"])
        parsed = git(parser_root, ["apply", "--numstat", "-z", str(patch)]).stdout
    paths = []
    for entry in parsed.split(b"\0"):
        if not entry:
            continue
        fields = entry.split(b"\t", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise PatchChainError("Canonical RCCL patches must contain ordinary text paths")
        name = os.fsdecode(fields[2])
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix() != name
                or any(part in ("..", ".git") for part in path.parts)):
            raise PatchChainError("Unsafe or unsupported Git patch path: " + repr(name))
        paths.append(name)
    if not paths:
        raise PatchChainError("Canonical patch has no affected paths")
    # Git accepts a patch stream with multiple complete sections for one file.
    # Copy that actual path once; Git remains responsible for section ordering.
    return list(dict.fromkeys(paths))


def file_state(source, name):
    """Reject symlink traversal and snapshot one real affected file before planning."""
    path = source
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        path = path / part
        if path.is_symlink():
            raise PatchChainError("Affected RCCL path traverses a symlink: " + name)
        if path.exists() and index + 1 != len(parts) and not path.is_dir():
            raise PatchChainError("Affected RCCL parent is not a directory: " + name)
    if not path.exists():
        return None
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PatchChainError("Affected RCCL path is not a regular file: " + name)
    return FileState(hashlib.sha256(path.read_bytes()).hexdigest(), stat.S_IMODE(info.st_mode))


def apply_chain(source, patches, upgrade_patches):
    """Apply one validated transition, or return False without changing current files.

    The original affected files and patch bytes are rechecked before final
    validation. Git prepares/checks the entire combined diff before one actual
    application; no real per-patch mutation, reset, fuzz or reject mode is used.
    """
    source = source.resolve(strict=True)
    patches = tuple(path.resolve(strict=True) for path in patches)
    upgrade_patches = tuple(path.resolve(strict=True) for path in upgrade_patches)
    if not source.is_dir() or not patches or len(set(patches)) != len(patches):
        raise PatchChainError("Canonical source/ordered patch set is invalid")
    if (not upgrade_patches or len(set(upgrade_patches)) != len(upgrade_patches)
            or any(path not in patches for path in upgrade_patches)):
        raise PatchChainError("Ordered predecessor patches must be unique members of the canonical chain")
    # Each authored predecessor removes the next newest addition. This is a
    # closed linear history, never permission to accept arbitrary patch subsets.
    # Several changes can land in one source commit while private qualification
    # builds contain intermediate generations; both exact states remain useful.
    complete = frozenset(patches)
    predecessors = {complete - frozenset(upgrade_patches[:count])
                    for count in range(len(upgrade_patches) + 1)}
    patch_hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in patches}
    names = sorted({name for patch in patches for name in patch_paths(patch)})
    original = {name: file_state(source, name) for name in names}
    with tempfile.TemporaryDirectory(prefix="llaminar-rccl-patch-chain-") as directory:
        temporary = Path(directory)
        git(temporary, ["init", "--quiet"])
        for name, state in original.items():
            if state is not None:
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        git(temporary, ["add", "--all"])
        before = git(temporary, ["write-tree"]).stdout.decode().strip()
        reversed_patches = set()
        for patch in reversed(patches):
            if git(temporary, ["apply", "--reverse", "--check", str(patch)], check=False).returncode == 0:
                git(temporary, ["apply", "--reverse", str(patch)])
                reversed_patches.add(patch)
        if reversed_patches and frozenset(reversed_patches) not in predecessors:
            raise PatchChainError("Unsupported partial RCCL patch chain; actual source is unchanged")
        for patch in patches:
            git(temporary, ["apply", "--check", str(patch)])
            git(temporary, ["apply", str(patch)])
        git(temporary, ["add", "--all"])
        delta = git(temporary, ["diff", "--cached", "--binary", "--full-index",
                                "--no-ext-diff", "--no-renames", before]).stdout
        if any(file_state(source, name) != value for name, value in original.items()):
            raise PatchChainError("RCCL affected source changed during patch validation")
        if any(hashlib.sha256(path.read_bytes()).digest() != value
               for path, value in patch_hashes.items()):
            raise PatchChainError("Canonical RCCL patch inputs changed during validation")
        if not delta:
            return False
        # Anchor the actual source as this private repository's worktree. This
        # avoids Git's subdirectory path filtering and never opens its real index.
        scope = ["--git-dir=" + str(temporary / ".git"), "--work-tree=" + str(source)]
        git(source, [*scope, "apply", "--check"], data=delta)
        git(source, [*scope, "apply"], data=delta)
        return True


def main():
    """Keep canonical ordering in the caller; this helper owns only the transaction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", required=True)
    parser.add_argument("--upgrade-patch", type=Path, action="append", required=True,
                        help="Repeat newest-first for each exact supported predecessor generation")
    arguments = parser.parse_args()
    try:
        changed = apply_chain(arguments.source, arguments.patch, arguments.upgrade_patch)
    except (PatchChainError, OSError) as error:
        parser.exit(1, "RCCL patch chain rejected: " + str(error) + "\n")
    print("RCCL patch chain: " + ("updated" if changed else "already current"))


if __name__ == "__main__":
    main()
