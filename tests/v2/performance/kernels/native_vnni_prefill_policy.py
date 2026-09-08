"""Load the canonical NativeVNNI grouped-decode and prefill M policy.

The C++ runtime owns its physical prefill buckets through an X-macro ``.def``
file so graph capture and every backend consume one inventory.  Trainer tools
must follow that include instead of scraping only integer literals from the
array initializer: the initializer intentionally contains no literal rows once
the X-macro is expanded by the compiler.
"""

from __future__ import annotations

import re
from pathlib import Path


_BUCKET_INVOCATION = re.compile(
    r"^\s*LLAMINAR_PREFILL_GRAPH_BUCKET\(([1-9][0-9]*)\)\s*$"
)


def _parse_int_array(text: str, symbol: str, path: Path) -> list[int]:
    """Parse an ordinary literal C++ integer-array initializer."""

    pattern = rf"{re.escape(symbol)}[^=]*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(f"{path}: could not find {symbol}")
    values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    if not values:
        raise SystemExit(f"{path}: {symbol} was empty")
    return values


def _resolve_bucket_definition(policy_header: Path, include_path: str) -> Path:
    """Resolve the quoted X-macro include using the runtime include roots."""

    candidates = (
        policy_header.parent / Path(include_path).name,
        policy_header.parent.parent / include_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"{policy_header}: could not resolve prefill bucket definition "
        f"{include_path!r}"
    )


def _parse_prefill_buckets(text: str, path: Path) -> list[int]:
    """Read literal buckets or expand the canonical X-macro definition."""

    pattern = r"kDefaultPrefillGraphBucketSizes[^=]*=\s*\{([^}]*)\}"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(
            f"{path}: could not find kDefaultPrefillGraphBucketSizes"
        )

    literal_values = [
        int(value) for value in re.findall(r"-?\d+", match.group(1))
    ]
    if literal_values:
        return literal_values

    include = re.search(
        r'#include\s+"([^"]*PrefillGraphBuckets\.def)"',
        match.group(1),
    )
    if not include:
        raise SystemExit(
            f"{path}: kDefaultPrefillGraphBucketSizes was empty"
        )

    definition = _resolve_bucket_definition(path, include.group(1))
    buckets = [
        int(bucket.group(1))
        for line in definition.read_text(encoding="utf-8").splitlines()
        if (bucket := _BUCKET_INVOCATION.fullmatch(line))
    ]
    if not buckets:
        raise SystemExit(f"{definition}: no prefill graph buckets")
    if buckets != sorted(set(buckets)):
        raise SystemExit(
            f"{definition}: prefill graph buckets must be unique and ordered"
        )
    return buckets


def load_native_vnni_m_policy(policy_header: Path) -> list[int]:
    """Return every canonical positive grouped-decode and prefill row count."""

    text = policy_header.read_text(encoding="utf-8")
    small = _parse_int_array(
        text, "kDefaultNativeVNNISmallMRows", policy_header
    )
    buckets = _parse_prefill_buckets(text, policy_header)
    return sorted({value for value in [*small, *buckets] if value > 0})
