#!/usr/bin/env python3
"""Reject nvcc split-compile on spill-guarded GPU device compilation.

nvcc ``--split-compile`` (and ``--split-compile-extended``) partitions one
translation unit's NVVM module across compiler workers. The partitioning is
not reproducible: identical compiles of the former monolithic CUDA prefill TU
produced 4, 30 or 56 spilling kernels on sm_120/121 (each target landing in a
121- or 128-register allocation), and shipped sm_80..90 SASS differed between
identical builds. The fatal GPU spill guard cannot certify an allocation that
the next identical build may not reproduce.

Compile throughput for large kernel families is therefore obtained from
persistent translation-unit shards (AGENTS.md maxim 34), never from
compiler-internal partitioning. This script proves that contract twice:

* every CMake file in the repository declares no split-compile option, so a
  configuration not exercised by the current build cannot reintroduce it;
* the configured build's actual compile database contains no split-compile
  flag on any CUDA/HIP command, including flags injected through cache
  variables such as CMAKE_CUDA_FLAGS.

``--self-test`` runs the negative controls without a repository or build.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import sys
import tempfile

SPLIT_COMPILE = re.compile(r"(?<![\w-])-{1,2}split-compile(?:-extended)?(?:[=\s]|$)")
CMAKE_GLOBS = ("**/CMakeLists.txt", "**/*.cmake")
EXCLUDED_ROOTS = ("build", "external", "corpora", ".git")
DEVICE_SUFFIXES = (".cu", ".hip")


def _excluded(path: pathlib.Path, repo_root: pathlib.Path) -> bool:
    """Return whether a path belongs to generated, vendored or data trees."""
    relative = path.relative_to(repo_root).parts
    return bool(relative) and (
        relative[0] in EXCLUDED_ROOTS or relative[0].startswith("build")
    )


def cmake_violations(repo_root: pathlib.Path) -> list[str]:
    """Return every CMake declaration of nvcc split-compile under the repository.

    Comments are ignored so explanatory prose may name the forbidden flag.
    """
    violations: list[str] = []
    for pattern in CMAKE_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if not path.is_file() or _excluded(path, repo_root):
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                code = line.split("#", 1)[0]
                if SPLIT_COMPILE.search(code):
                    violations.append(
                        f"{path.relative_to(repo_root)}:{number}: nvcc split-compile "
                        "is nondeterministic; shard the translation unit instead"
                    )
    return violations


def _command_tokens(entry: dict[str, object]) -> list[str]:
    """Normalize either compile_commands.json command representation."""
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return [str(argument) for argument in arguments]
    command = entry.get("command")
    if isinstance(command, str):
        return shlex.split(command)
    raise AssertionError(f"compile command has no command/arguments field: {entry}")


def command_violations(entries: list[dict[str, object]]) -> list[str]:
    """Return every device compile command that requests split-compile."""
    violations: list[str] = []
    for entry in entries:
        source = str(entry.get("file", ""))
        if not source.endswith(DEVICE_SUFFIXES):
            continue
        for token in _command_tokens(entry):
            if SPLIT_COMPILE.search(token + " "):
                violations.append(f"{source}: generated command uses {token}")
    return violations


def verify(repo_root: pathlib.Path, build_dir: pathlib.Path | None) -> list[str]:
    """Run the source and (when supplied) generated-command contracts."""
    violations = cmake_violations(repo_root)
    if build_dir is not None:
        database = build_dir / "compile_commands.json"
        if not database.is_file():
            raise AssertionError(f"compile database does not exist: {database}")
        violations += command_violations(
            json.loads(database.read_text(encoding="utf-8"))
        )
    return violations


def run_self_test() -> int:
    """Prove that both forms of the defect are detected and clean inputs pass."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "src").mkdir()
        (root / "src" / "CMakeLists.txt").write_text(
            "# --split-compile=8 in a comment is prose, not configuration\n"
            "set_source_files_properties(a.cu PROPERTIES COMPILE_OPTIONS \"--split-compile=8\")\n",
            encoding="utf-8",
        )
        (root / "cmake").mkdir()
        (root / "cmake" / "Extended.cmake").write_text(
            "target_compile_options(t PRIVATE --split-compile-extended=4)\n",
            encoding="utf-8",
        )
        (root / "build_x").mkdir()
        (root / "build_x" / "CMakeLists.txt").write_text(
            "set(X --split-compile=0)\n", encoding="utf-8"
        )
        found = cmake_violations(root)
        assert len(found) == 2, found
        assert any("src/CMakeLists.txt:2" in item for item in found), found
        assert any("Extended.cmake:1" in item for item in found), found

    entries = [
        {"file": "/s/a.cu", "command": "nvcc -O3 --split-compile=0 -c a.cu"},
        {"file": "/s/b.cu", "arguments": ["nvcc", "-split-compile", "8", "-c", "b.cu"]},
        {"file": "/s/c.cu", "command": "nvcc -O3 -c c.cu"},
        {"file": "/s/d.cpp", "command": "c++ --split-compile=8 d.cpp"},
        {"file": "/s/e.hip", "command": "clang++ --split-compile-extended=2 e.hip"},
    ]
    found = command_violations(entries)
    assert [item.split(":")[0] for item in found] == ["/s/a.cu", "/s/b.cu", "/s/e.hip"], found
    assert not command_violations([entries[2], entries[3]])
    print("deterministic GPU compilation policy self-test: PASS")
    return 0


def main() -> int:
    """Parse arguments and run the requested contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--build-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.repo_root is None:
        parser.error("--repo-root is required unless --self-test is given")
    violations = verify(args.repo_root.resolve(), args.build_dir)
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        print("deterministic GPU compilation policy: FAIL", file=sys.stderr)
        return 1
    print("deterministic GPU compilation policy: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"deterministic GPU compilation policy: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
