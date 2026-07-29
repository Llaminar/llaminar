#!/usr/bin/env python3
"""Regress optimized accelerator flags and compiler-cache wiring.

Integration is a custom CMake build type. CMake does not synthesize CUDA or HIP
flags for custom configurations, so a configuration that defines only C/C++
flags silently compiles device kernels with the accelerator compiler's
unoptimized defaults. Accelerator translation units are also expensive enough
that continuously evicting or bypassing ccache makes ordinary iteration
needlessly slow. This test checks both declarative contracts and, when supplied,
the generated compile database used by the current test binary.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import sys


REQUIRED_CONFIGURATION_FLAGS = {
    "CMAKE_CUDA_FLAGS_INTEGRATION": (
        "-O3",
        "-g",
        "-DNDEBUG",
        "-DLLAMINAR_ENABLE_ASSERTIONS",
    ),
    "CMAKE_HIP_FLAGS_INTEGRATION": (
        "-O3",
        "-g",
        "-DNDEBUG",
        "-DLLAMINAR_ENABLE_ASSERTIONS",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse paths supplied by CTest or a developer's direct invocation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--compile-commands", type=pathlib.Path)
    parser.add_argument("--build-dir", type=pathlib.Path)
    return parser.parse_args()


def declared_cache_value(cmake_text: str, variable: str) -> str:
    """Return the quoted cache value assigned to one CMake variable."""

    match = re.search(
        rf"set\(\s*{re.escape(variable)}\s+\"([^\"]*)\"\s+CACHE\s+STRING",
        cmake_text,
    )
    if match is None:
        raise AssertionError(f"{variable} is not declared as a cached string")
    return match.group(1)


def verify_declarations(cmake_path: pathlib.Path) -> None:
    """Require optimized CUDA/HIP definitions for every custom build type."""

    cmake_text = cmake_path.read_text(encoding="utf-8")
    for variable, required_flags in REQUIRED_CONFIGURATION_FLAGS.items():
        value = declared_cache_value(cmake_text, variable)
        tokens = set(shlex.split(value))
        missing = [flag for flag in required_flags if flag not in tokens]
        if missing:
            raise AssertionError(
                f"{variable} is missing required flags {missing}: {value!r}"
            )


def verify_ccache_declarations(repo_root: pathlib.Path) -> None:
    """Require all compilers and development containers to use a durable cache."""

    cmake_path = repo_root / "src/v2/CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")
    for language in ("C", "CXX", "CUDA", "HIP"):
        required = (
            f'set(CMAKE_{language}_COMPILER_LAUNCHER "${{CCACHE_PROGRAM}}")'
        )
        if required not in cmake_text:
            raise AssertionError(f"{cmake_path} does not cache {language} compilation")

    setup_path = repo_root / ".devcontainer/setup-ccache.sh"
    setup_text = setup_path.read_text(encoding="utf-8")
    if 'CCACHE_MAX_SIZE="${LLAMINAR_CCACHE_MAXSIZE:-50G}"' not in setup_text:
        raise AssertionError(f"{setup_path} does not retain the 50G cache default")

    devcontainer_path = repo_root / ".devcontainer/devcontainer.json"
    devcontainer = json.loads(devcontainer_path.read_text(encoding="utf-8"))
    container_env = devcontainer.get("containerEnv", {})
    required_container_env = {
        "CCACHE_DIR": "/home/vscode/.ccache",
        "CCACHE_MAXSIZE": "50G",
        "CCACHE_BASEDIR": "/workspaces/llaminar",
        "CCACHE_NOHASHDIR": "1",
    }
    for variable, expected in required_container_env.items():
        actual = container_env.get(variable)
        if actual != expected:
            raise AssertionError(
                f"{devcontainer_path} must publish {variable}={expected!r} "
                f"to every container process, got {actual!r}"
            )

    dockerfile_path = repo_root / "Dockerfile"
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    if "CCACHE_MAXSIZE=50G" not in dockerfile:
        raise AssertionError(f"{dockerfile_path} does not retain the 50G cache default")


def command_text(entry: dict[str, object]) -> str:
    """Normalize either compile_commands.json command representation."""

    command = entry.get("command")
    if isinstance(command, str):
        return command
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return " ".join(str(argument) for argument in arguments)
    raise AssertionError(f"compile command has no command/arguments field: {entry}")


def verify_generated_commands(compile_commands_path: pathlib.Path) -> None:
    """Check accelerator optimization in commands emitted for this build tree."""

    entries = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    accelerator_entries: dict[str, list[dict[str, object]]] = {
        "CUDA": [],
        "HIP": [],
    }
    for entry in entries:
        source = str(entry.get("file", ""))
        if source.endswith(".cu"):
            accelerator_entries["CUDA"].append(entry)
        elif source.endswith(".hip"):
            accelerator_entries["HIP"].append(entry)

    for backend, backend_entries in accelerator_entries.items():
        if not backend_entries:
            raise AssertionError(f"compile database contains no {backend} entries")
        for entry in backend_entries:
            command = command_text(entry)
            tokens = set(shlex.split(command))
            if "-O3" not in tokens:
                raise AssertionError(
                    f"{backend} Integration kernel lacks -O3: {entry.get('file')}"
                )


def verify_generated_launcher(build_dir: pathlib.Path) -> None:
    """Prove Ninja's executable rules invoke ccache for host and GPU compilers."""

    cache_path = build_dir / "CMakeCache.txt"
    cache = cache_path.read_text(encoding="utf-8")
    ccache_match = re.search(
        r"^CCACHE_PROGRAM:FILEPATH=(.+)$",
        cache,
        flags=re.MULTILINE,
    )
    if ccache_match is None:
        raise AssertionError(f"{cache_path} does not resolve CCACHE_PROGRAM")

    build_ninja_path = build_dir / "build.ninja"
    build_ninja = build_ninja_path.read_text(encoding="utf-8")
    launcher = f"LAUNCHER = {ccache_match.group(1)} "
    if launcher not in build_ninja:
        raise AssertionError(f"{build_ninja_path} has no ccache launcher edges")

    rules_path = build_dir / "CMakeFiles/rules.ninja"
    rules = rules_path.read_text(encoding="utf-8")
    for language in ("CXX", "CUDA", "HIP"):
        rule_pattern = re.compile(
            rf"rule {language}_COMPILER__[^\n]+\n"
            rf"(?:  [^\n]*\n)*?"
            rf"  command = .*?\$\{{LAUNCHER\}}",
        )
        if rule_pattern.search(rules) is None:
            raise AssertionError(
                f"{rules_path} does not launch {language} compilation through ccache"
            )


def main() -> int:
    """Run the source contract and optional generated-command regression."""

    args = parse_args()
    verify_declarations(args.repo_root / "src" / "v2" / "CMakeLists.txt")
    verify_ccache_declarations(args.repo_root)
    if args.compile_commands is not None:
        if not args.compile_commands.is_file():
            raise AssertionError(
                f"compile database does not exist: {args.compile_commands}"
            )
        verify_generated_commands(args.compile_commands)
    if args.build_dir is not None:
        verify_generated_launcher(args.build_dir)
    print("accelerator custom-build flags: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"accelerator custom-build flags: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
