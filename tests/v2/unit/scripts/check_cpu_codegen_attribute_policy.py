#!/usr/bin/env python3
"""Admit only arithmetic-preserving per-function code-generation attributes in src/v2.

Function-level `optimize(...)` attributes change how GCC compiles a function. The
exact NEON tile programs use them to pin fp-contract=off/no-associative-math and to
select register-pressure options (no-thread-jumps, no-schedule-insns) that reorder
or duplicate instructions without changing arithmetic. This lexical gate admits:

* `optimize(...)` (any spelling: __attribute__, [[gnu::...]], __optimize__) whose
  arguments are string literals made only of ADMITTED tokens;
* an attribute carrying a register-pressure or vectoriser token (REQUIRES_EXACT) only
  together with fp-contract=off and no-associative-math. The single exception is
  SLP_PAIR_EXEMPTIONS: the four vectoriser-only `no-tree-slp-vectorize` pins of the
  x86 packed-prefill TU, named by exact file and function. Every exemption must
  match exactly one attribute; a renamed or moved exempt function fails closed
  (its attribute loses the waiver and the stale exemption is reported).

It rejects target/target_clones attributes, `#pragma GCC optimize|target|push_options`
(including _Pragma), `#pragma clang optimize`, every FP-semantic option
(fast-math, fp-contract=fast/on, associative/reciprocal/finite/unsafe math, ...) and
any non-literal optimize argument (a macro could hide a forbidden option).
It is a lexical scan, not C++ semantic analysis; unknown forms fail closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx", ".inl", ".inc", ".ipp", ".cu", ".cuh", ".hip"}
ADMITTED = {"fp-contract=off", "no-associative-math", "no-thread-jumps", "no-schedule-insns",
            "no-tree-slp-vectorize"}
REQUIRES_EXACT = {"no-thread-jumps", "no-schedule-insns", "no-tree-slp-vectorize"}
EXACT = {"fp-contract=off", "no-associative-math"}
SLP_TOKEN = "no-tree-slp-vectorize"
# Pre-existing x86 packed-prefill pins: they select the loop vectoriser over SLP for the
# ordered panel family and carry no exact pair. Their contraction is proven separately by
# the x86 contraction witness. Keyed by (repository-relative file, declared function name).
X86_PACKED_PREFILL = "src/v2/kernels/cpu/gemm/CPUNativeX86PackedPrefill.cpp"
SLP_PAIR_EXEMPTIONS = frozenset((X86_PACKED_PREFILL, name)
                                for name in ("orderedPanel", "stripeOrdered", "run", "pooledChunk"))
FP_SEMANTIC = re.compile(
    r"^(?:O(?:fast|[0-3sgz]?)$|(?:no-)?fast-math$|fp-contract=(?:fast|on)$|associative-math$|reciprocal-math$|"
    r"(?:no-)?finite-math-only$|unsafe-math-optimizations$|no-signed-zeros$|no-trapping-math$|"
    r"(?:no-)?rounding-math$|cx-limited-range$|excess-precision=|no-math-errno$|"
    r"(?:no-)?signaling-nans$|float-store$|no-float-store$|fp-model)")
FORBIDDEN_ATTRIBUTE = {"target", "target_clones", "target_version"}
OPTIMIZE_ATTRIBUTE = {"optimize"}
PRAGMA = re.compile(r"#\s*pragma\s+(?:GCC\s+(?:optimize|target|push_options|pop_options|reset_options)|"
                    r"clang\s+optimize|clang\s+attribute)\b|"
                    r"\b_Pragma\s*\(\s*\"\s*(?:GCC\s+(?:optimize|target|push_options|pop_options|reset_options)|"
                    r"clang\s+optimize|clang\s+attribute)\b")
FP_PRAGMA = re.compile(r"(?:#\s*pragma|_Pragma\s*\(\s*\")\s*(?:clang\s+fp\s+(?:contract\s*\(\s*(?:fast|on)|"
                       r"reassociate\s*\(\s*on|exceptions\s*\(\s*ignore)|STDC\s+FP_CONTRACT\s+ON|"
                       r"float_control\s*\(\s*(?:precise\s*,\s*off|except\s*,\s*off)|fp_contract\s*\(\s*on)")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


LEXEME = re.compile(r"//[^\n]*|/\*.*?(?:\*/|\Z)|\"(?:[^\"\\\n]|\\.)*\"?|'(?:[^'\\\n]|\\.)*'?", re.S)


def blank_comments(text: str) -> str:
    """Replace comments by spaces (newlines kept); string and character literals stay intact."""
    def blank(match: re.Match) -> str:
        lexeme = match.group(0)
        if lexeme.startswith("//") or lexeme.startswith("/*"):
            return re.sub(r"[^\n]", " ", lexeme)
        return lexeme
    return LEXEME.sub(blank, text)


def balanced(text: str, open_index: int) -> int:
    """Index just past the parenthesis/bracket group opening at open_index (string-aware)."""
    pairs = {"(": ")", "[": "]"}
    stack, index = [pairs[text[open_index]]], open_index + 1
    while index < len(text) and stack:
        char = text[index]
        if char in "\"'":
            index += 1
            while index < len(text) and text[index] != char:
                index += 2 if text[index] == "\\" else 1
        elif char in pairs:
            stack.append(pairs[char])
        elif char == stack[-1]:
            stack.pop()
        index += 1
    if stack:
        raise ValueError("unbalanced attribute")
    return index


STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
ATTRIBUTE_NAME = re.compile(r"\s*(?:(?:gnu|__gnu__|clang)\s*::\s*)?([A-Za-z_]\w*)\s*(\()?")


def split_top(body: str) -> list[str]:
    """Split an attribute list on top-level commas."""
    parts, depth, start, index = [], 0, 0, 0
    while index < len(body):
        char = body[index]
        if char in "\"'":
            index += 1
            while index < len(body) and body[index] != char:
                index += 2 if body[index] == "\\" else 1
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
        index += 1
    parts.append(body[start:])
    return [part for part in parts if part.strip()]


def option_tokens(arguments: str) -> list[str] | None:
    """Normalized option tokens of optimize(...) arguments; None when any argument is not a string literal."""
    tokens = []
    for argument in split_top(arguments):
        literals = STRING.findall(argument)
        if not literals or STRING.sub("", argument).strip():
            return None
        for option in "".join(literals).split(","):
            option = option.strip()
            if option.startswith("-f"):
                option = option[2:]
            elif option.startswith("-"):
                option = option[1:]
            tokens.append(option)
    return tokens


def check_attribute_list(body: str, line: int, path: str, function: str | None = None,
                         exempt_used: list[tuple[str, str]] | None = None) -> list[Violation]:
    """Violations of one attribute list declared before @p function (None when not a function)."""
    violations = []
    for item in split_top(body):
        match = ATTRIBUTE_NAME.match(item)
        if not match:
            continue
        name = match.group(1)
        if name.startswith("__") and name.endswith("__") and len(name) > 4:
            name = name[2:-2]
        if name in FORBIDDEN_ATTRIBUTE:
            violations.append(Violation(path, line, f"forbidden code-generation attribute {name}(...)"))
            continue
        if name not in OPTIMIZE_ATTRIBUTE:
            continue
        if not match.group(2):
            violations.append(Violation(path, line, "optimize attribute without a literal argument list"))
            continue
        start = match.end() - 1
        arguments = item[start + 1:balanced(item, start) - 1]
        tokens = option_tokens(arguments)
        if tokens is None:
            violations.append(Violation(path, line, "optimize(...) arguments must be string literals"))
            continue
        for token in tokens:
            if FP_SEMANTIC.match(token):
                violations.append(Violation(path, line, f"FP-semantic optimize option {token!r}"))
            elif token not in ADMITTED:
                violations.append(Violation(path, line, f"optimize option {token!r} is not admitted"))
        unpaired = REQUIRES_EXACT & set(tokens) if not EXACT <= set(tokens) else set()
        if SLP_TOKEN in unpaired and function is not None and (path, function) in SLP_PAIR_EXEMPTIONS:
            unpaired.discard(SLP_TOKEN)
            if exempt_used is not None:
                exempt_used.append((path, function))
        if unpaired:
            violations.append(Violation(path, line, "optimize options " + ", ".join(sorted(unpaired))
                                        + " require fp-contract=off and no-associative-math in the same attribute"))
    return violations


ATTRIBUTE_OPEN = re.compile(r"__attribute__\s*\(\s*\(|\[\[")
DECLARATOR = re.compile(r"[^;{}()#=\[\]]*?\b([A-Za-z_]\w*)\s*\(")


def declared_function(code: str, end: int) -> str | None:
    """Name of the function whose declarator directly follows an attribute ending at @p end."""
    match = DECLARATOR.match(code, end)
    return match.group(1) if match else None
BARE = re.compile(r"\b(?:__)?(?:optimize|target_clones|target_version)(?:__)?\s*\(|\b__target__\s*\(")


def scan_text(path: str, text: str, code: str | None = None,
              exempt_used: list[tuple[str, str]] | None = None) -> list[Violation]:
    code = blank_comments(text) if code is None else code
    line_of = lambda offset: code.count("\n", 0, offset) + 1  # noqa: E731
    violations: list[Violation] = []
    covered: list[tuple[int, int]] = []
    for match in ATTRIBUTE_OPEN.finditer(code) if "__attribute__" in code or "[[" in code else ():
        if any(lo <= match.start() < hi for lo, hi in covered):
            continue
        if match.group(0) == "[[":
            open_index = match.start()
            try:
                end = balanced(code, open_index)
            except ValueError:
                violations.append(Violation(path, line_of(match.start()), "unbalanced [[...]] attribute"))
                continue
            inner = code[open_index + 2:end - 2]
            if inner.lstrip().startswith("using"):
                inner = inner.split(":", 1)[1] if ":" in inner else inner
        else:
            open_index = code.index("(", match.start())
            try:
                end = balanced(code, open_index)
            except ValueError:
                violations.append(Violation(path, line_of(match.start()), "unbalanced __attribute__"))
                continue
            inner = code[code.index("(", open_index + 1) + 1:end - 2]
        covered.append((match.start(), end))
        violations += check_attribute_list(inner, line_of(match.start()), path, declared_function(code, end),
                                           exempt_used)
    for match in BARE.finditer(code) if "optimize" in code or "target_" in code or "__target__" in code else ():
        if not any(lo <= match.start() < hi for lo, hi in covered):
            violations.append(Violation(path, line_of(match.start()),
                                        "code-generation attribute outside __attribute__((...)) or [[...]]"))
    pragmas = "pragma" in code or "Pragma" in code
    for match in PRAGMA.finditer(code) if pragmas else ():
        violations.append(Violation(path, line_of(match.start()), "forbidden optimization/target pragma"))
    for match in FP_PRAGMA.finditer(code) if pragmas else ():
        violations.append(Violation(path, line_of(match.start()), "FP-semantic pragma"))
    return violations


CANDIDATE = re.compile(r"optimize|target|pragma|_Pragma")


def scan(repo_root: Path) -> tuple[list[Violation], int, int]:
    """(violations, files scanned, optimize attributes seen) over src/v2.

    Every SLP_PAIR_EXEMPTIONS entry must waive exactly one attribute; otherwise the
    exemption is stale (function renamed, moved or duplicated) and is reported.
    """
    violations, files, attributes = [], 0, 0
    exempt_used: list[tuple[str, str]] = []
    for path in sorted((repo_root / "src/v2").rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        files += 1
        if not CANDIDATE.search(text):
            continue  # Cheap filter only: every attribute or pragma form contains one of these words.
        relative = path.relative_to(repo_root).as_posix()
        code = blank_comments(text)
        attributes += len(re.findall(r"\boptimize\s*\(", code))
        violations += scan_text(relative, text, code, exempt_used)
    for path, function in sorted(SLP_PAIR_EXEMPTIONS):
        count = exempt_used.count((path, function))
        if count != 1:
            violations.append(Violation(path, 0, f"{SLP_TOKEN} pair exemption for {function}() matched {count} "
                                                 "attributes, expected exactly 1"))
    return violations, files, attributes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(argv)
    violations, files, attributes = scan(args.repo_root.resolve())
    if files == 0:
        print("no src/v2 sources found", file=sys.stderr)
        return 2
    if attributes == 0:
        print("no optimize(...) attribute found: the scan cannot see the exact-FP kernels", file=sys.stderr)
        return 2
    for violation in violations:
        print(violation, file=sys.stderr)
    print(f"{files} files, {attributes} optimize attributes, {len(violations)} violations")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
