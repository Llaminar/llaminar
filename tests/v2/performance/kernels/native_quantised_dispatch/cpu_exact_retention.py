"""Emitter-owned additive refresh of legacy CPU exact tables.

Only the literal tables emitted by the existing CPU analyzers are accepted.
The complete non-exact C++ program is retained byte-for-byte. New rows replace
only explicitly measured keys; unrelated exact rows retain their original text
or literal values. This does not fit or certify a new generic policy.
"""
from __future__ import annotations

import hashlib
import re
from typing import Mapping


def _sha(source: str) -> str:
    """Name an emitter program/evidence string independently of file location."""
    return "sha256:" + hashlib.sha256(source.encode()).hexdigest()


def _abi(source: str, kind: str) -> str:
    """Require one unchanged installed selector ABI before composing literals."""
    token = "DECODE" if kind == "decode" else "VERIFIER"
    matches = re.findall(r"^#define LLAMINAR_CPU_NATIVE_QUANTISED_" + token + r"_POLICY_ABI (\d+)$", source, re.M)
    if len(matches) != 1:
        raise ValueError("CPU exact retention requires one generated ABI declaration")
    return matches[0]


def _scope_provenance(source: str, measured: str) -> str:
    """Keep the old certificate scoped to retained generic data, never new rows."""
    corpus = re.search(r"^// Common corpus digest: (sha256:[0-9a-f]+)$", measured, re.M)
    note = "// Exact-only refresh; retained generic certificate does not certify these new exact rows.\n"
    if corpus: note += "// Exact refresh corpus digest: " + corpus[1] + "\n"
    return note + source.replace("// Common policy digest:", "// Retained base policy digest:").replace(
        "// Common corpus digest:", "// Retained base corpus digest:").replace(
        "// Sealed generic certificate:", "// Retained generic certificate:")


def _literal_array_body(body: str, hexadecimal: bool) -> tuple[int, ...]:
    """Reject foreign C++ instead of treating arbitrary source as a table."""
    pattern = r"0x[0-9a-fA-F]+ULL" if hexadecimal else r"\d+"
    tokens = re.findall(pattern, body)
    remainder = re.sub(pattern, "", body)
    if not tokens or re.sub(r"[\s,]", "", remainder):
        raise ValueError("CPU exact retention found a foreign/empty literal table")
    return tuple(int(value[:-3], 16) if hexadecimal else int(value) for value in tokens)


_ARRAY = re.compile(
    r"inline constexpr (uint64_t|uint8_t) (kCPUNativeQuantisedVerifierExact\w+)(Keys|Policies)\[\] =\n\{(.*?)\n\};", re.S)


def _verifier_arrays(source: str) -> dict[str, tuple[re.Match, tuple[int, ...]]]:
    """Read only paired, sorted, unique verifier exact arrays from this emitter."""
    arrays = {}
    for match in _ARRAY.finditer(source):
        name = match[2] + match[3]
        if name in arrays or (match[1] == "uint64_t") != (match[3] == "Keys"):
            raise ValueError("CPU verifier exact array identity is ambiguous")
        arrays[name] = (match, _literal_array_body(match[4], match[3] == "Keys"))
    groups = {name.removesuffix("Keys") for name in arrays if name.endswith("Keys")}
    if not groups or set(arrays) != {group + suffix for group in groups for suffix in ("Keys", "Policies")}:
        raise ValueError("CPU verifier exact arrays are incomplete")
    for group in groups:
        keys = arrays[group + "Keys"][1]; policies = arrays[group + "Policies"][1]
        if len(keys) != len(policies) or tuple(sorted(set(keys))) != keys or any(value > 8 for value in policies):
            raise ValueError("CPU verifier exact literal key/policy ABI changed")
    return arrays


def _render_array_body(values: tuple[int, ...], hexadecimal: bool) -> str:
    """Keep generated formatting bounded while preserving full literal values."""
    width = 4 if hexadecimal else 24
    tokens = [f"0x{v:016x}ULL" if hexadecimal else str(v) for v in values]
    return "\n" + "\n".join("    " + ", ".join(tokens[i:i + width]) + "," for i in range(0, len(tokens), width))


def refresh_verifier_exact(base: str, measured: str) -> tuple[str, dict]:
    """Compose newly measured legacy grouped keys, never regenerate generics."""
    if _abi(base, "verifier") != _abi(measured, "verifier"):
        raise ValueError("CPU verifier exact refresh cannot change selector ABI")
    original = _verifier_arrays(base); new = _verifier_arrays(measured)
    if not set(new).issubset(original):
        raise ValueError("CPU exact refresh introduced an unregistered ISA/thread table")
    replacements = []; changed = 0
    for name in sorted(new):
        if not name.endswith("Keys"):
            continue
        group = name.removesuffix("Keys")
        old_rows = dict(zip(original[name][1], original[group + "Policies"][1]))
        fresh_rows = dict(zip(new[name][1], new[group + "Policies"][1]))
        old_rows.update(fresh_rows); changed += len(fresh_rows)
        ordered = tuple(sorted(old_rows))
        for suffix, values, hexadecimal in (("Keys", ordered, True), ("Policies", tuple(old_rows[k] for k in ordered), False)):
            match = original[group + suffix][0]
            replacements.append((match.start(4), match.end(4), _render_array_body(values, hexadecimal)))
    result = base
    for start, finish, body in sorted(replacements, reverse=True):
        result = result[:start] + body + result[finish:]
    def program(source):
        return _ARRAY.sub(lambda m: m[0][:m.start(4) - m.start()] + "<retained-exact-data>" + m[0][m.end(4) - m.start():], source)
    if program(base) != program(result):
        raise AssertionError("CPU verifier non-exact program changed")
    result = _scope_provenance(result, measured)
    return result, {"kind": "verifier", "base_sha256": _sha(base), "result_sha256": _sha(result),
                    "retained_non_exact_sha256": _sha(program(base)), "measured_exact_keys": changed,
                    "generic_program_unchanged": True, "new_generic_certification": False}


def _closing_brace(source: str, start: int) -> int:
    """Match braces only inside this emitter's fixed literal-only exact arm."""
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{": depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0: return index
    raise ValueError("unterminated CPU decode exact arm")


_DECODE_GROUP = re.compile(r"if \(build_isa == CPUNativeQuantisedDecodeBuildISA::(AVX2|AVX512) &&\n        runtime_isa == CPUNativeQuantisedDecodeRuntimeISA::(AVX2|AVX512)\)")
_DECODE_CASE = re.compile(r"            case (0x[0-9a-fA-F]{16})ULL:\n(?:                //[^\n]*\n)*                policy = CPUNativeQuantisedDecodePolicy::Nbc(?:1|2|4|8|16);\n                return true;\n")


def _decode_exact_arms(source: str) -> dict[tuple[str, str, int], tuple[int, int, dict[int, str]]]:
    """Read exact switch arms without accepting a second C++ implementation."""
    arms = {}
    for isa in _DECODE_GROUP.finditer(source):
        start = source.index("{", isa.end()); finish = _closing_brace(source, start)
        body = source[start:finish]
        for thread in re.finditer(r"if \(threads == (\d+)\)\n        \{\n            switch \(key\)\n            \{", body):
            opening = start + thread.end() - 1; closing = _closing_brace(source, opening)
            begin = opening + 1; literal = source[begin:closing]
            cases = list(_DECODE_CASE.finditer(literal)); rows = {int(m[1], 16): m[0] for m in cases}
            if not cases or len(rows) != len(cases) or re.sub(r"\s", "", _DECODE_CASE.sub("", literal)) != "default:break;":
                raise ValueError("CPU decode exact arm contains foreign or duplicate statements")
            group = (isa[1], isa[2], int(thread[1]))
            if group in arms: raise ValueError("duplicate CPU decode ISA/thread arm")
            arms[group] = (begin, closing, rows)
    if not arms: raise ValueError("CPU decode include has no supported exact arms")
    return arms


def refresh_decode_exact(base: str, measured: str) -> tuple[str, dict]:
    """Overlay measured M1 switch literals; preserve every other program byte."""
    if _abi(base, "decode") != _abi(measured, "decode"):
        raise ValueError("CPU decode exact refresh cannot change selector ABI")
    original = _decode_exact_arms(base); fresh = _decode_exact_arms(measured)
    if not set(fresh).issubset(original):
        raise ValueError("CPU exact refresh introduced an unregistered ISA/thread arm")
    replacements = []; changed = 0
    for group, (_, _, new_rows) in fresh.items():
        begin, finish, old_rows = original[group]
        rows = dict(old_rows); rows.update(new_rows); changed += len(new_rows)
        # Keep original unrelated case strings/order. Append newly admitted keys
        # deterministically; switch dispatch does not require numeric sorting.
        order = list(old_rows) + sorted(set(new_rows).difference(old_rows))
        replacements.append((begin, finish, "\n" + "".join(rows[key] for key in order) + "            default:\n                break;\n            "))
    result = base
    for start, finish, body in sorted(replacements, reverse=True): result = result[:start] + body + result[finish:]
    def program(source):
        spans = _decode_exact_arms(source)
        for begin, finish, _ in sorted(spans.values(), reverse=True): source = source[:begin] + "<retained-exact-data>" + source[finish:]
        return source
    if program(base) != program(result): raise AssertionError("CPU decode generic program changed")
    result = _scope_provenance(result, measured)
    return result, {"kind": "decode", "base_sha256": _sha(base), "result_sha256": _sha(result),
                    "retained_non_exact_sha256": _sha(program(base)), "measured_exact_keys": changed,
                    "generic_program_unchanged": True, "new_generic_certification": False}
