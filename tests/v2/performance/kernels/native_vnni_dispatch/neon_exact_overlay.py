"""Generate native ARM NativeVNNI exact overlays from authenticated NEON sweeps.

Native NEON projections reduce every row over full K in fixed register trees.
Their only schedule dimensions -- the shared-row tile (``Pair``/``Four``) and
the adjacent 64-column chunks per dynamically claimed task (``nbc``) -- are
numerically neutral, so an overlay may never change a byte. The runtime total
policy is ``Four``/``nbc=1`` for every geometry; this generator emits an exact
entry only where a measured candidate beats it.

Input rows come from ``Perf__CPUNativeNEONExactOverlay`` (schema
``neon-exact-overlay-v1``). For each runtime key (execution codebook, source
superblock layout, N, K, physical M, workers) every measured source alias must
provide the complete candidate set with byte-exact output. Aliases sharing one
runtime key are pooled by the geometric mean of their median latency
normalised to the total policy. An entry is emitted only when:

- the winner differs from the total policy,
- its geometric-mean normalised latency improves by at least
  ``--min-improvement``, and
- it is not slower than the total policy for any alias.

Missing candidates, non-exact bytes, duplicate measurements or mixed ISA
identities fail closed. Unmeasured keys stay on the total policy; nothing is
interpolated or fitted. The receipt records every input digest and decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = "neon-exact-overlay-v1"
TOTAL_POLICY = ("Four", 1)
TILES = ("Pair", "Four")
CHUNKS = (1, 2, 4, 8)
FIELDS = (
    "schema", "isa", "codebook", "superblock", "source_format", "shape", "n", "k",
    "m", "workers", "tile", "nbc", "samples", "median_us", "p10_us", "p90_us",
    "byte_exact",
)


class OverlayError(ValueError):
    """Reject incomplete, non-exact or ambiguous overlay evidence."""


@dataclass(frozen=True, order=True)
class OverlayKey:
    """Runtime-visible exact key of one native projection invocation."""

    codebook: int
    superblock: bool
    n: int
    k: int
    m: int
    workers: int


@dataclass(frozen=True)
class Entry:
    """One emitted exact schedule and its supporting statistics."""

    key: OverlayKey
    tile: str
    nbc: int
    geomean_ratio: float
    sources: tuple[str, ...]


def expected_candidates(m: int) -> tuple[tuple[str, int], ...]:
    """Return the complete candidate set measured for one physical M."""

    if m == 1:
        return tuple(("Four", chunks) for chunks in CHUNKS)
    return tuple((tile, chunks) for chunks in CHUNKS for tile in TILES)


def read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Read every CSV row and the SHA-256 of each input file."""

    rows: list[dict[str, str]] = []
    digests: dict[str, str] = {}
    for path in paths:
        data = path.read_bytes()
        digests[str(path)] = hashlib.sha256(data).hexdigest()
        reader = csv.DictReader(data.decode("utf-8").splitlines())
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise OverlayError(f"{path}: unexpected header {reader.fieldnames}")
        rows.extend(reader)
    if not rows:
        raise OverlayError("no overlay rows")
    return rows, digests


def collect(rows: Iterable[Mapping[str, str]]):
    """Group authenticated rows by runtime key, source alias and candidate."""

    isas = set()
    grouped: dict[OverlayKey, dict[str, dict[tuple[str, int], float]]] = {}
    for row in rows:
        if row["schema"] != SCHEMA:
            raise OverlayError(f"unsupported schema {row['schema']!r}")
        if row["byte_exact"] != "1":
            raise OverlayError(f"non-exact candidate output: {row}")
        isas.add(row["isa"])
        key = OverlayKey(int(row["codebook"]), row["superblock"] == "1", int(row["n"]),
                         int(row["k"]), int(row["m"]), int(row["workers"]))
        if min(key.n, key.k, key.m, key.workers) <= 0:
            raise OverlayError(f"non-positive geometry: {row}")
        candidate = (row["tile"], int(row["nbc"]))
        if candidate not in expected_candidates(key.m):
            raise OverlayError(f"unknown candidate {candidate} for M={key.m}")
        median = float(row["median_us"])
        if not math.isfinite(median) or median <= 0.0 or int(row["samples"]) <= 0:
            raise OverlayError(f"invalid timing: {row}")
        sources = grouped.setdefault(key, {})
        candidates = sources.setdefault(row["source_format"], {})
        if candidate in candidates:
            raise OverlayError(f"duplicate measurement for {key} {row['source_format']} {candidate}")
        candidates[candidate] = median
    if len(isas) != 1:
        raise OverlayError(f"overlay inputs mix ISA identities {sorted(isas)}")
    return grouped, isas.pop()


def decide(grouped, min_improvement: float) -> tuple[list[Entry], list[dict]]:
    """Choose exact winners; return emitted entries and every key decision."""

    entries: list[Entry] = []
    decisions: list[dict] = []
    for key in sorted(grouped):
        sources = grouped[key]
        expected = expected_candidates(key.m)
        for source, candidates in sources.items():
            missing = [c for c in expected if c not in candidates]
            if missing:
                raise OverlayError(f"{key} {source} lacks candidates {missing}")
        ratios = {}
        for candidate in expected:
            logs = [math.log(candidates[candidate] / candidates[TOTAL_POLICY])
                    for candidates in sources.values()]
            ratios[candidate] = math.exp(sum(logs) / len(logs))
        # Deterministic: lowest pooled ratio, total policy first on ties.
        order = sorted(expected, key=lambda c: (ratios[c], c != TOTAL_POLICY, c))
        winner = order[0]
        no_alias_regression = all(candidates[winner] <= candidates[TOTAL_POLICY]
                                  for candidates in sources.values())
        emit = (winner != TOTAL_POLICY and ratios[winner] <= 1.0 - min_improvement
                and no_alias_regression)
        decisions.append({
            "key": key.__dict__, "winner": list(winner), "ratio": ratios[winner],
            "sources": sorted(sources), "emitted": emit,
        })
        if emit:
            entries.append(Entry(key, winner[0], winner[1], ratios[winner], tuple(sorted(sources))))
    return entries, decisions


def render(entries: Sequence[Entry], isa: str) -> str:
    """Render the installed C++ table, sorted by runtime key."""

    lines = [
        "// Generated by tests/v2/performance/kernels/native_vnni_dispatch/neon_exact_overlay.py.",
        f"// Do not edit. {len(entries)} authenticated native ARM ({isa}) exact overlays; every",
        "// other key resolves to the total four-row/one-chunk native policy.",
        "inline constexpr std::array<NativeNEONExactOverlayEntry, "
        f"{len(entries)}> kGeneratedNativeNEONExactOverlay{{{{",
    ]
    for entry in sorted(entries, key=lambda e: e.key):
        k = entry.key
        lines.append(
            f"    {{.codebook_id = {k.codebook}, .is_superblock = {'true' if k.superblock else 'false'}, "
            f".n = {k.n}, .k = {k.k}, .m = {k.m}, .workers = {k.workers}, "
            f".tile = NEONProjectionRowTile::{entry.tile}, .n_block_chunks = {entry.nbc}}}, "
            f"// ratio {entry.geomean_ratio:.4f} over {'/'.join(entry.sources)}")
    lines.append("}};")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the installed table and its receipt from authenticated CSVs."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_improvement < 1.0:
        raise OverlayError("--min-improvement must be in [0, 1)")
    rows, digests = read_rows(args.csv)
    grouped, isa = collect(rows)
    entries, decisions = decide(grouped, args.min_improvement)
    rendered = render(entries, isa)
    args.output.write_text(rendered, encoding="utf-8")
    receipt = {
        "schema": "neon-exact-overlay-receipt-v1",
        "isa": isa,
        "inputs": digests,
        "min_improvement": args.min_improvement,
        "keys": len(decisions),
        "emitted": len(entries),
        "output_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "decisions": decisions,
    }
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"neon exact overlay: {len(entries)} of {len(decisions)} keys emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
