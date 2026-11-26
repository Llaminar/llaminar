#!/usr/bin/env python3
"""
Enforce the Llaminar V2 translation-unit line-count limit (<= 7,000 lines).

Giant C++, CUDA, and HIP translation units serialize Ninja builds and inflate
compiler memory usage. Every compiled translation unit (.c, .cc, .cpp, .cxx,
.cu, .hip) under src/v2/ must stay at or below MAX_TU_LINES (7,000 lines),
with no permanent allowlist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

MAX_TU_LINES = 7000
TU_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".cu", ".hip"})


def count_lines(path: Path) -> int:
    """Count physical lines in one UTF-8/ASCII source file."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def scan_repo(repo_root: Path, max_lines: int = MAX_TU_LINES) -> list[tuple[int, str]]:
    """Return sorted (line_count, relative_path) violations for src/v2 translation units."""
    src_v2 = repo_root / "src" / "v2"
    if not src_v2.is_dir():
        raise FileNotFoundError(f"Expected src/v2 directory under {repo_root}")

    violations: list[tuple[int, str]] = []
    for path in sorted(src_v2.rglob("*")):
        if not path.is_file() or path.suffix not in TU_SUFFIXES:
            continue
        lines = count_lines(path)
        if lines > max_lines:
            rel = path.relative_to(repo_root).as_posix()
            violations.append((lines, rel))
    violations.sort(key=lambda item: (-item[0], item[1]))
    return violations


def run_self_test() -> int:
    """Verify the policy accepts a 7,000-line TU and rejects a 7,001-line TU."""
    with tempfile.TemporaryDirectory(prefix="llaminar_tu_limit_selftest_") as tmp:
        root = Path(tmp)
        src_dir = root / "src" / "v2" / "synthetic"
        src_dir.mkdir(parents=True, exist_ok=True)

        exact_limit = src_dir / "ExactLimit.cpp"
        exact_limit.write_text("// line\n" * MAX_TU_LINES, encoding="utf-8")
        over_limit_cpp = src_dir / "OverLimit.cpp"
        over_limit_cpp.write_text("// line\n" * (MAX_TU_LINES + 1), encoding="utf-8")
        over_limit_cu = src_dir / "OverLimit.cu"
        over_limit_cu.write_text("// line\n" * (MAX_TU_LINES + 5), encoding="utf-8")
        over_limit_hip = src_dir / "OverLimit.hip"
        over_limit_hip.write_text("// line\n" * (MAX_TU_LINES + 10), encoding="utf-8")
        # Included generated data tables (.inc) are not standalone TUs.
        generated_inc = src_dir / "GeneratedTable.inc"
        generated_inc.write_text("// line\n" * (MAX_TU_LINES + 100), encoding="utf-8")

        violations = scan_repo(root)
        got = {rel for _, rel in violations}
        expected = {
            "src/v2/synthetic/OverLimit.cpp",
            "src/v2/synthetic/OverLimit.cu",
            "src/v2/synthetic/OverLimit.hip",
        }
        if got != expected:
            print(
                f"FAIL: self-test expected violations {sorted(expected)}, got {sorted(got)}",
                file=sys.stderr,
            )
            return 1
    print(f"PASS: TU line-limit policy self-test verified (limit={MAX_TU_LINES})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Llaminar V2 TU line-count limit.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root containing src/v2.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic positive/negative checks.")
    parser.add_argument("--max-lines", type=int, default=MAX_TU_LINES, help="Maximum permitted lines per TU.")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    repo_root = args.repo_root
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    violations = scan_repo(repo_root, max_lines=args.max_lines)
    if violations:
        print(
            f"FAIL: {len(violations)} translation unit(s) exceed the {args.max_lines}-line limit:",
            file=sys.stderr,
        )
        for lines, rel in violations:
            print(f"  {lines:6d} lines  {rel} (+{lines - args.max_lines})", file=sys.stderr)
        return 1

    print(f"PASS: all src/v2 translation units are <= {args.max_lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
