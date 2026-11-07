#!/usr/bin/env python3
"""@file test_neon_exact_overlay.py
@brief Device-free regressions for the native ARM exact-overlay generator.

The generator may only replace the total four-row/one-chunk NEON policy for a
fully measured runtime key whose byte-exact winner pools across every source
alias and independent replica without regressing any of them. Incomplete,
non-exact, duplicated, unreplicated or mixed-ISA evidence fails; nothing is
interpolated.
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch import neon_exact_overlay as overlay  # noqa: E402


def rows_for(source: str, m: int, latencies: dict, *, codebook=5, superblock=1, n=2048, k=4096,
             workers=20, isa="NEON_DOTPROD", exact="1"):
    """Return one complete candidate set for a key; ``latencies`` overrides defaults."""

    result = []
    for tile, chunks in overlay.expected_candidates(m):
        median = latencies.get((tile, chunks), 100.0)
        result.append({
            "schema": overlay.SCHEMA, "isa": isa, "codebook": str(codebook),
            "superblock": str(superblock), "source_format": source, "shape": "S", "n": str(n),
            "k": str(k), "m": str(m), "workers": str(workers), "tile": tile, "nbc": str(chunks),
            "samples": "15", "median_us": f"{median:.3f}", "p10_us": f"{median:.3f}",
            "p90_us": f"{median:.3f}", "byte_exact": exact,
        })
    return result


def write(rows) -> Path:
    """Write rows to a temporary CSV with the canonical header."""

    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(handle, fieldnames=overlay.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    return Path(handle.name)


class NeonExactOverlayTest(unittest.TestCase):
    """Exact-only, alias-pooled and fail-closed generator semantics."""

    def generate(self, rows, *extra, replica_rows=None):
        """Run the generator over ``rows`` and an independent second replica.

        The second replica defaults to the same latencies with a distinct
        per-file shape label, so the identical-bytes guard is not tripped.
        """
        if replica_rows is None:
            replica_rows = [dict(row, shape="S2") for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "table.inc"
            receipt = Path(directory) / "receipt.json"
            overlay.main(["--csv", str(write(rows)), "--csv", str(write(replica_rows)),
                          "--output", str(out), "--receipt", str(receipt), *extra])
            return out.read_text(), receipt.read_text()

    def test_single_replica_missing_key_or_replica_regression_is_rejected(self):
        rows = rows_for("Q4_K", 4, {("Pair", 2): 80.0})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "t.inc"
            receipt = Path(directory) / "r.json"
            with self.assertRaises(overlay.OverlayError):
                overlay.main(["--csv", str(write(rows)), "--output", str(out), "--receipt", str(receipt)])
            same = write(rows)
            with self.assertRaises(overlay.OverlayError):
                overlay.main(["--csv", str(same), "--csv", str(same), "--output", str(out),
                              "--receipt", str(receipt)])
        # A key measured by only one replica fails closed.
        with self.assertRaises(overlay.OverlayError):
            self.generate(rows, replica_rows=rows_for("Q4_K", 8, {}))
        # Per-format replica files combine: each alias needs its own two replicas.
        q4 = rows_for("Q4_K", 4, {("Pair", 2): 80.0})
        q8 = rows_for("Q8_0", 4, {("Pair", 4): 70.0}, codebook=19, superblock=0)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "t.inc"
            receipt = Path(directory) / "r.json"
            overlay.main(["--csv", str(write(q4)), "--csv", str(write([dict(r, shape="S2") for r in q4])),
                          "--csv", str(write(q8)), "--csv", str(write([dict(r, shape="S2") for r in q8])),
                          "--output", str(out), "--receipt", str(receipt)])
            self.assertIn("std::array<NativeNEONExactOverlayEntry, 2>", out.read_text())
        # A noisy first-run winner that regresses in the replica is never installed.
        table, _ = self.generate(rows, replica_rows=rows_for("Q4_K", 4, {("Pair", 2): 101.0}))
        self.assertIn("std::array<NativeNEONExactOverlayEntry, 0>", table)
        table, receipt = self.generate(rows, replica_rows=rows_for("Q4_K", 4, {("Pair", 2): 90.0}))
        self.assertIn("std::array<NativeNEONExactOverlayEntry, 1>", table)
        self.assertIn('"replicas": 2', receipt)

    def test_winner_emitted_only_when_it_beats_total_policy_for_every_alias(self):
        rows = (rows_for("Q4_K", 4, {("Pair", 2): 80.0}) +
                rows_for("Q4_1", 4, {("Pair", 2): 90.0}, superblock=1))
        table, receipt = self.generate(rows)
        self.assertIn("std::array<NativeNEONExactOverlayEntry, 1>", table)
        self.assertIn(".m = 4, .workers = 20, .tile = NEONProjectionRowTile::Pair, .n_block_chunks = 2", table)
        self.assertIn('"emitted": 1', receipt)

    def test_total_policy_or_small_or_alias_regressing_wins_are_not_emitted(self):
        for rows in (
            rows_for("Q4_K", 4, {}),                                  # all tie: total policy
            rows_for("Q4_K", 4, {("Pair", 2): 98.5}),                  # below 3% threshold
            rows_for("Q4_K", 4, {("Pair", 2): 50.0}) +                 # one alias regresses
            rows_for("Q4_1", 4, {("Pair", 2): 120.0}),
        ):
            table, _ = self.generate(rows)
            self.assertIn("std::array<NativeNEONExactOverlayEntry, 0>", table)

    def test_decode_candidates_are_chunk_only_and_entries_sorted(self):
        rows = (rows_for("Q8_0", 1, {("Four", 8): 70.0}, codebook=19, superblock=0, n=512) +
                rows_for("Q4_K", 15, {("Four", 4): 70.0}))
        table, _ = self.generate(rows)
        first = table.index(".codebook_id = 5,")
        second = table.index(".codebook_id = 19,")
        self.assertLess(first, second)
        self.assertIn(".m = 1, .workers = 20, .tile = NEONProjectionRowTile::Four, .n_block_chunks = 8", table)

    def test_incomplete_nonexact_duplicate_or_mixed_evidence_fails(self):
        complete = rows_for("Q4_K", 4, {})
        for rows in (
            complete[:-1],
            rows_for("Q4_K", 4, {}, exact="0"),
            complete + complete[:1],
            complete + rows_for("Q4_1", 4, {}, isa="NEON"),
        ):
            with self.assertRaises(overlay.OverlayError):
                self.generate(rows)

    def test_rendered_header_matches_runtime_entry_type(self):
        table, _ = self.generate(rows_for("Q4_K", 4, {("Pair", 1): 50.0}))
        self.assertTrue(table.startswith("// Generated by"))
        self.assertIn("kGeneratedNativeNEONExactOverlay{{", table)
        self.assertTrue(table.rstrip().endswith("}};"))


if __name__ == "__main__":
    unittest.main()
