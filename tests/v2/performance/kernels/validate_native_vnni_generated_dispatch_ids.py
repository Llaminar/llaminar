#!/usr/bin/env python3
"""Validate generated NativeVNNI dispatch includes against canonical codebook ids."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import CODEBOOK_TO_FORMAT  # noqa: E402

# The persistent ROCm small-M workspace owns one partial-output slot for every
# K partition and is sized for 64 partitions.  Do not confuse this independent
# launch axis with the grouped verifier's M=2..16 row extent.
ROCM_DECODE_GRAPH_SAFE_KB_CAP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("include", type=Path, nargs="+", help="Generated C++ include(s)")
    return parser.parse_args()


def _known_aliases(codebook: int) -> set[str]:
    label = CODEBOOK_TO_FORMAT.get(codebook)
    if not label:
        return set()
    return {part.strip() for part in label.split("/") if part.strip()}


def _extract_codebooks(text: str) -> set[int]:
    ids: set[int] = set()
    patterns = [
        r"\bCB\s*==\s*(\d+)\b",
        r"\bCB=(\d+)\b",
        r"\bcodebook\s*==\s*(\d+)\b",
        r"\bselectTuning_CB(\d+)\s*\(",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            ids.add(int(match.group(1)))
    if "packCPUNativeVNNIPrefillPolicyKey" in text:
        for match in re.finditer(
            r"\bcase\s+0x([0-9a-fA-F]{16})ULL\s*:", text
        ):
            ids.add((int(match.group(1), 16) >> 56) & 0xFF)
    return ids


def _validate_cpu_verifier_packed_tables(path: Path, text: str) -> set[int]:
    """Validate compact grouped-verifier exact keys and policy arrays.

    The grouped verifier packs ``codebook/M/K/N`` as 8/8/24/24 bits. Each
    runtime surface owns a strictly sorted key array and one policy byte per
    key. ABI v2's historical Pairwise/WideRows bitset remains readable while a
    checked-in corpus is upgraded; ABI v3 requires the byte array so every
    registered grouped family has an unambiguous exact-overlay identity. The
    validator decodes the generated data itself so compact binary-search tables
    cannot weaken codebook, geometry, ordering, or cardinality checks.
    """

    if "packCPUNativeVNNIVerifierRowsPolicyKey" not in text:
        return set()

    keys_pattern = re.compile(
        r"inline\s+constexpr\s+uint64_t\s+"
        r"kCPUNativeVNNIVerifierExact(?P<surface>[A-Za-z0-9]+)Keys\[\]\s*=\s*"
        r"\{(?P<body>.*?)\n\};",
        re.DOTALL,
    )
    mask_pattern = re.compile(
        r"inline\s+constexpr\s+uint64_t\s+"
        r"kCPUNativeVNNIVerifierExact(?P<surface>[A-Za-z0-9]+)WideRowsMask\[\]\s*=\s*"
        r"\{(?P<body>.*?)\n\};",
        re.DOTALL,
    )
    policies_pattern = re.compile(
        r"inline\s+constexpr\s+uint8_t\s+"
        r"kCPUNativeVNNIVerifierExact(?P<surface>[A-Za-z0-9]+)Policies\[\]\s*=\s*"
        r"\{(?P<body>.*?)\n\};",
        re.DOTALL,
    )
    key_tables = {
        match.group("surface"): [
            int(raw, 16)
            for raw in re.findall(
                r"\b0x([0-9a-fA-F]{16})ULL\b", match.group("body")
            )
        ]
        for match in keys_pattern.finditer(text)
    }
    mask_tables = {
        match.group("surface"): [
            int(raw, 16)
            for raw in re.findall(
                r"\b0x([0-9a-fA-F]{16})ULL\b", match.group("body")
            )
        ]
        for match in mask_pattern.finditer(text)
    }
    policy_tables = {
        match.group("surface"): [
            int(raw)
            for raw in re.findall(r"\b(\d+)\b", match.group("body"))
        ]
        for match in policies_pattern.finditer(text)
    }
    if not key_tables:
        raise SystemExit(
            f"{path}: CPU verifier selector has no packed exact-key tables"
        )
    abi_match = re.search(
        r"#define\s+LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI\s+(\d+)", text
    )
    abi = int(abi_match.group(1)) if abi_match else 0
    expected_tables = policy_tables if abi >= 3 else mask_tables
    expected_name = "policy arrays" if abi >= 3 else "legacy policy masks"
    if set(key_tables) != set(expected_tables):
        missing = sorted(set(key_tables) - set(expected_tables))
        orphan = sorted(set(expected_tables) - set(key_tables))
        raise SystemExit(
            f"{path}: CPU verifier exact table pairing is incomplete "
            f"representation={expected_name} missing={missing} orphan={orphan}"
        )
    enum_match = re.search(
        r"enum\s+class\s+CPUNativeVNNIVerifierRowsPolicy\s*:\s*uint8_t\s*"
        r"\{(?P<body>.*?)\};",
        text,
        re.DOTALL,
    )
    policy_ordinals = {
        int(raw)
        for raw in re.findall(
            r"[A-Za-z0-9_]+\s*=\s*(\d+)",
            enum_match.group("body") if enum_match else "",
        )
    }
    if abi >= 3 and not policy_ordinals:
        raise SystemExit(f"{path}: CPU verifier ABI v3 has no policy enum values")

    codebooks: set[int] = set()
    for surface, keys in sorted(key_tables.items()):
        if not keys:
            raise SystemExit(
                f"{path}: CPU verifier exact table {surface} is empty"
            )
        for left, right in zip(keys, keys[1:]):
            if left >= right:
                raise SystemExit(
                    f"{path}: CPU verifier exact table {surface} is not strictly "
                    f"sorted (0x{left:016x} before 0x{right:016x})"
                )
        for key in keys:
            codebook = (key >> 56) & 0xFF
            m = (key >> 48) & 0xFF
            k = (key >> 24) & 0xFFFFFF
            n = key & 0xFFFFFF
            if codebook not in CODEBOOK_TO_FORMAT:
                raise SystemExit(
                    f"{path}: CPU verifier key references unknown codebook {codebook}"
                )
            if m < 2 or n == 0 or k == 0:
                raise SystemExit(
                    f"{path}: malformed CPU verifier key M={m} N={n} K={k}"
                )
            codebooks.add(codebook)

        if abi >= 3:
            policies = policy_tables[surface]
            if len(policies) != len(keys):
                raise SystemExit(
                    f"{path}: CPU verifier exact table {surface} has "
                    f"{len(policies)} policies for {len(keys)} keys"
                )
            unknown = sorted(set(policies) - policy_ordinals)
            if unknown:
                raise SystemExit(
                    f"{path}: CPU verifier exact table {surface} contains "
                    f"unknown policy ordinals {unknown}"
                )
        else:
            masks = mask_tables[surface]
            required_words = (len(keys) + 63) // 64
            if len(masks) != required_words:
                raise SystemExit(
                    f"{path}: CPU verifier exact table {surface} has {len(masks)} "
                    f"policy words for {len(keys)} keys; expected {required_words}"
                )
            used_bits = len(keys) % 64
            if used_bits and masks[-1] >> used_bits:
                raise SystemExit(
                    f"{path}: CPU verifier exact table {surface} sets unused policy bits"
                )
    return codebooks


def _validate_cpu_prefill_packed_keys(path: Path, text: str) -> None:
    """Decode and validate the staged CPU prefill selector's 8/16/20/20 ABI."""

    if "packCPUNativeVNNIPrefillPolicyKey" not in text:
        return
    keys = [
        int(match.group(1), 16)
        for match in re.finditer(
            r"\bcase\s+0x([0-9a-fA-F]{16})ULL\s*:", text
        )
    ]
    if not keys:
        raise SystemExit(f"{path}: CPU prefill selector has no packed exact keys")
    for switch_index, match in enumerate(re.finditer(
        r"switch\s*\(key\)\s*\{(?P<body>.*?)\bdefault\s*:",
        text,
        re.DOTALL,
    ), start=1):
        switch_keys = re.findall(
            r"\bcase\s+0x([0-9a-fA-F]{16})ULL\s*:",
            match.group("body"),
        )
        if len(switch_keys) != len(set(switch_keys)):
            raise SystemExit(
                f"{path}: CPU prefill switch #{switch_index} has duplicate keys"
            )
    for key in keys:
        codebook = (key >> 56) & 0xFF
        m = (key >> 40) & 0xFFFF
        k = (key >> 20) & 0xFFFFF
        n = key & 0xFFFFF
        if codebook not in CODEBOOK_TO_FORMAT:
            raise SystemExit(
                f"{path}: CPU prefill key references unknown codebook {codebook}"
            )
        if m < 2 or n == 0 or k == 0:
            raise SystemExit(
                f"{path}: malformed CPU prefill key M={m} N={n} K={k}"
            )


def _validate_labeled_branches(path: Path, text: str) -> None:
    branch_patterns = [
        r"CB\s*==\s*(\d+)\)\s*\{\s*//\s*([A-Za-z0-9_./-]+)",
        r"CB=(\d+)\s*\(([A-Za-z0-9_./-]+)",
        r"CB=(\d+)\s*\(\s*([A-Za-z0-9_./-]+)",
    ]
    for pattern in branch_patterns:
        for match in re.finditer(pattern, text):
            codebook = int(match.group(1))
            label = match.group(2).strip()
            aliases = _known_aliases(codebook)
            if not aliases:
                raise SystemExit(
                    f"{path}: generated dispatch references unknown codebook {codebook} "
                    f"with label {label!r}"
                )
            if label not in aliases:
                raise SystemExit(
                    f"{path}: codebook {codebook} label {label!r} does not match "
                    f"canonical alias set {sorted(aliases)}"
                )


def _validate_rocm_decode_graph_safe_kb(path: Path, text: str) -> None:
    if (
        "ROCmNativeVNNIDecodeDispatchConfig" not in text and
        "ROCmNativeVNNIBatchedDecodeDispatchConfig" not in text
    ):
        return

    for match in re.finditer(r"\{0x[0-9a-fA-F]+ULL,\s*\{(\d+),\s*(\d+)\}\}", text):
        kb = int(match.group(1))
        if kb > ROCM_DECODE_GRAPH_SAFE_KB_CAP:
            raise SystemExit(
                f"{path}: ROCm NativeVNNI decode generated kb={kb} exceeds "
                f"graph-safe small-M cap {ROCM_DECODE_GRAPH_SAFE_KB_CAP}"
            )
    for match in re.finditer(
        r"\{\s*\d+\s*,\s*\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,"
        r"\s*\{[^{}]*\}\s*,\s*\{[^{}]*\}\s*,\s*\{(\d+)\s*,\s*(\d+)\}\s*\}",
        text,
    ):
        kb = int(match.group(1))
        if kb > ROCM_DECODE_GRAPH_SAFE_KB_CAP:
            raise SystemExit(
                f"{path}: ROCm NativeVNNI batched decode generated kb={kb} exceeds "
                f"graph-safe small-M cap {ROCM_DECODE_GRAPH_SAFE_KB_CAP}"
            )


def _validate_binary_search_tables_sorted(path: Path, text: str) -> None:
    """Ensure generated tables stay sorted for their binary-search helpers."""

    table_pattern = re.compile(
        r"static\s+constexpr\s+[A-Za-z0-9_:<>]+\s+kTable\[\]\s*=\s*\{(?P<body>.*?)\n\s*\};",
        re.DOTALL,
    )
    for table_index, match in enumerate(table_pattern.finditer(text), start=1):
        keys = [int(raw, 16) for raw in re.findall(r"\{\s*0x([0-9a-fA-F]+)ULL\s*,", match.group("body"))]
        if len(keys) < 2:
            continue
        for left, right in zip(keys, keys[1:]):
            if left > right:
                raise SystemExit(
                    f"{path}: generated table #{table_index} is not sorted for binary search "
                    f"(0x{left:x} appears before 0x{right:x})"
                )


def validate_file(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"generated include not found: {path}")

    text = path.read_text()
    codebooks = _extract_codebooks(text)
    codebooks.update(_validate_cpu_verifier_packed_tables(path, text))
    if not codebooks:
        raise SystemExit(f"{path}: found no generated codebook dispatch branches")

    known = set(CODEBOOK_TO_FORMAT)
    unknown = sorted(codebooks - known)
    if unknown:
        raise SystemExit(
            f"{path}: generated dispatch references unknown codebook id(s): "
            f"{', '.join(str(value) for value in unknown)}"
        )

    _validate_labeled_branches(path, text)
    _validate_cpu_prefill_packed_keys(path, text)
    _validate_rocm_decode_graph_safe_kb(path, text)
    _validate_binary_search_tables_sorted(path, text)
    return len(codebooks)


def main() -> int:
    args = parse_args()
    total = 0
    for path in args.include:
        total += validate_file(path)
    print(f"validated {total} generated NativeVNNI dispatch codebook reference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
