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
  configuration not exercised by the current build cannot reintroduce it.
  Comments are removed with a CMake-aware lexer (``#`` inside quoted
  arguments and bracket arguments is data, not a comment);
* the configured build's actual compile database contains no split-compile
  flag on any device compilation. Device compilation is identified by the
  compiler (nvcc/hipcc/clang with CUDA or HIP language) or the source suffix,
  never by the suffix alone, and option sources are expanded first: nvcc
  ``--options-file``/``-optf`` and ``@response`` files are read relative to
  the command's directory. An option source that cannot be read is itself a
  violation, because its contents cannot be certified.

Every invocation runs the negative controls first, so the preflight cell
executes them too. ``--self-test`` runs only the controls.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import sys
import tempfile

SPLIT_COMPILE = re.compile(r"^-{1,2}split-compile(?:-extended)?(?:=.*)?$")
CMAKE_GLOBS = ("**/CMakeLists.txt", "**/*.cmake")
EXCLUDED_ROOTS = ("external", "corpora", ".git")
DEVICE_SUFFIXES = (".cu", ".hip", ".cu.inc")
DEVICE_COMPILERS = ("nvcc", "hipcc")
DEVICE_LANGUAGES = ("cu", "cuda", "hip")
OPTION_FILE_FLAGS = ("--options-file", "-optf")
BRACKET_OPEN = re.compile(r"\[(=*)\[")
BRACKET_COMMENT = re.compile(r"#\[(=*)\[")


def _excluded(path: pathlib.Path, repo_root: pathlib.Path) -> bool:
    """Return whether a path belongs to generated, vendored or data trees."""
    relative = path.relative_to(repo_root).parts
    return bool(relative) and (
        relative[0] in EXCLUDED_ROOTS or relative[0].startswith("build")
    )


def cmake_code(text: str) -> str:
    """Return CMake source with comments blanked, preserving line numbers.

    Handles quoted arguments (with escapes), bracket arguments ``[=[...]=]``
    and bracket comments ``#[[...]]``; ``#`` inside an argument is data.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        bracket = BRACKET_OPEN.match(text, i) if ch == "[" else None
        if bracket:
            close = "]" + bracket.group(1) + "]"
            j = text.find(close, i + len(bracket.group(0)))
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            i = j
            continue
        if ch == "#":
            comment = BRACKET_COMMENT.match(text, i)
            if comment:
                close = "]" + comment.group(1) + "]"
                j = text.find(close, i)
                j = n if j < 0 else j + len(close)
            else:
                j = text.find("\n", i)
                j = n if j < 0 else j
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _cmake_tokens(line: str) -> list[str]:
    """Split one comment-free CMake line into unquoted argument words."""
    words = re.findall(r'"((?:[^"\\]|\\.)*)"|([^\s()"]+)', line)
    tokens: list[str] = []
    for quoted, bare in words:
        tokens.extend((quoted or bare).replace(";", " ").split())
    return tokens


def cmake_violations(repo_root: pathlib.Path) -> list[str]:
    """Return every CMake declaration of nvcc split-compile under the repository."""
    violations: list[str] = []
    for pattern in CMAKE_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if not path.is_file() or _excluded(path, repo_root):
                continue
            code = cmake_code(path.read_text(encoding="utf-8", errors="replace"))
            for number, line in enumerate(code.splitlines(), 1):
                if any(SPLIT_COMPILE.match(token) for token in _cmake_tokens(line)):
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


def _expand_option_sources(
    tokens: list[str], directory: pathlib.Path, unreadable: list[str], depth: int = 0
) -> list[str]:
    """Inline nvcc option files and @response files relative to the command directory."""
    expanded: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        source: str | None = None
        for flag in OPTION_FILE_FLAGS:
            if token == flag and i + 1 < len(tokens):
                source = tokens[i + 1]
                i += 1
            elif token.startswith(flag + "="):
                source = token.split("=", 1)[1]
        if source is None and token.startswith("@") and len(token) > 1:
            source = token[1:]
        if source is None:
            expanded.append(token)
            i += 1
            continue
        path = pathlib.Path(source)
        path = path if path.is_absolute() else directory / path
        try:
            if depth > 8:
                raise OSError("option sources nest too deeply")
            inner = shlex.split(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            unreadable.append(f"{path}: {error}")
        else:
            expanded.extend(_expand_option_sources(inner, path.parent, unreadable, depth + 1))
        i += 1
    return expanded


def _is_device_compilation(tokens: list[str], source: str) -> bool:
    """Identify device compilation from the compiler and language, not the suffix alone."""
    if source.endswith(DEVICE_SUFFIXES):
        return True
    executables = [pathlib.Path(token).name for token in tokens[:2]]
    if any(name.startswith(DEVICE_COMPILERS) for name in executables):
        return True
    for i, token in enumerate(tokens):
        language = None
        if token == "-x" and i + 1 < len(tokens):
            language = tokens[i + 1]
        elif token.startswith("-x") and len(token) > 2:
            language = token[2:]
        elif token.startswith("--language="):
            language = token.split("=", 1)[1]
        if language in DEVICE_LANGUAGES:
            return True
    return False


def command_violations(entries: list[dict[str, object]]) -> list[str]:
    """Return every device compile command that requests split-compile."""
    violations: list[str] = []
    for entry in entries:
        source = str(entry.get("file", ""))
        directory = pathlib.Path(str(entry.get("directory", ".")))
        unreadable: list[str] = []
        tokens = _expand_option_sources(_command_tokens(entry), directory, unreadable)
        if not _is_device_compilation(tokens, source):
            continue
        for problem in unreadable:
            violations.append(f"{source}: unreadable option source {problem}")
        for token in tokens:
            if SPLIT_COMPILE.match(token):
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
    """Prove that every form of the defect is detected and clean inputs pass."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "src").mkdir()
        (root / "src" / "CMakeLists.txt").write_text(
            "# --split-compile=8 in a comment is prose, not configuration\n"
            "set_source_files_properties(a.cu PROPERTIES COMPILE_OPTIONS \"--split-compile=8\")\n"
            "target_compile_options(t PRIVATE \"-DNOTE=#\" \"--split-compile=8\")\n"
            "#[[ --split-compile=2 inside a bracket comment ]]\n"
            "set(OPTS \"-O3;--split-compile=0\")\n"
            "set(SAFE \"-DTEXT=--split-compile-is-only-a-word\")\n",
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
        lines = sorted(item.split(":")[0] + ":" + item.split(":")[1] for item in found)
        assert lines == [
            "cmake/Extended.cmake:1",
            "src/CMakeLists.txt:2",
            "src/CMakeLists.txt:3",
            "src/CMakeLists.txt:5",
        ], found

        build = root / "build"
        build.mkdir()
        (build / "gpu.opts").write_text("-O3 --split-compile=8\n", encoding="utf-8")
        (build / "clean.opts").write_text("-O3\n", encoding="utf-8")
        (build / "nested.rsp").write_text("--options-file gpu.opts\n", encoding="utf-8")
        d = str(build)
        entries = [
            {"directory": d, "file": "/s/a.cu", "command": "nvcc -O3 --split-compile=0 -c a.cu"},
            {"directory": d, "file": "/s/b.cu", "arguments": ["nvcc", "-split-compile", "8", "-c", "b.cu"]},
            {"directory": d, "file": "/s/c.cu", "command": "nvcc -O3 -c c.cu"},
            {"directory": d, "file": "/s/d.cpp", "command": "c++ --split-compile=8 d.cpp"},
            {"directory": d, "file": "/s/e.hip", "command": "clang++ --split-compile-extended=2 e.hip"},
            {"directory": d, "file": "/s/f.cu", "command": "nvcc --options-file gpu.opts -c f.cu"},
            {"directory": d, "file": "/s/g.cu", "command": "nvcc -optf=clean.opts -c g.cu"},
            {"directory": d, "file": "/s/h.cpp", "command": "nvcc -x cu --split-compile=8 -c h.cpp"},
            {"directory": d, "file": "/s/i.cpp", "command": "clang++ -xcuda --split-compile=8 -c i.cpp"},
            {"directory": d, "file": "/s/j.cu", "command": "nvcc @nested.rsp -c j.cu"},
            {"directory": d, "file": "/s/k.cu", "command": "nvcc --options-file missing.opts -c k.cu"},
            {"directory": d, "file": "/s/l.cu.inc", "command": "/usr/bin/ccache nvcc --split-compile=1 -c l.cu.inc"},
        ]
        found = command_violations(entries)
        offenders = sorted({item.split(":")[0] for item in found})
        assert offenders == [
            "/s/a.cu", "/s/b.cu", "/s/e.hip", "/s/f.cu", "/s/h.cpp",
            "/s/i.cpp", "/s/j.cu", "/s/k.cu", "/s/l.cu.inc",
        ], found
        assert not command_violations([entries[2], entries[3], entries[6]])
    print("deterministic GPU compilation policy self-test: PASS")
    return 0


def main() -> int:
    """Parse arguments, run the negative controls, then the requested contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--build-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_self_test()
    if args.self_test:
        return 0
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
