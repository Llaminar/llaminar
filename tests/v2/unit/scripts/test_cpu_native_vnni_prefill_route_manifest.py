#!/usr/bin/env python3
"""Regression tests for production C++ CPU prefill route evidence."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
    read_cpu_prefill_route_manifests,
)


FIELDNAMES = (
    "schema_version",
    "shape",
    "n",
    "k",
    "execution_codebook",
    "payload_bytes",
    "threads",
    "isa_regime",
    "k_tiles",
    "bundle_signature",
)


class CPUNativeVNNIPrefillRouteManifestTest(unittest.TestCase):
    """Prove route manifests fail closed before timing collection."""

    def _write(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _row(*, shape: str, k_tiles: int) -> dict[str, object]:
        return {
            "schema_version": "cpu-prefill-serial-route-v1",
            "shape": shape,
            "n": 1536,
            "k": 8960,
            "execution_codebook": 0,
            "payload_bytes": 16,
            "threads": 28,
            "isa_regime": "avx2-build.avx2-runtime",
            "k_tiles": k_tiles,
            "bundle_signature": (
                CPU_PREFILL_KPART_BUNDLE
                if k_tiles > 1
                else CPU_PREFILL_FULL_K_BUNDLE
            ),
        }

    def test_reads_typed_routes_and_stable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv"
            second = Path(temporary) / "second.csv"
            first_row = self._row(shape="1.5B_FFN_Dn", k_tiles=7)
            second_row = self._row(shape="1.5B_AttnOut", k_tiles=0)
            self._write(first, [first_row])
            self._write(second, [second_row])

            forward = read_cpu_prefill_route_manifests((first, second))
            reverse = read_cpu_prefill_route_manifests((second, first))

            route = forward.route_for(
                0,
                "1.5B_FFN_Dn",
                "avx2-build.avx2-runtime",
            )
            self.assertTrue(route.serial_kpart)
            self.assertEqual(route.k_tiles, 7)
            self.assertEqual(forward.thread_count(), 28)
            self.assertEqual(forward.digest(), reverse.digest())

    def test_rejects_mixed_openmp_widths(self) -> None:
        """One route digest cannot represent two incompatible CPU teams."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.csv"
            first = self._row(shape="1.5B_FFN_Dn", k_tiles=7)
            second = self._row(shape="1.5B_AttnOut", k_tiles=0)
            second["threads"] = 36
            self._write(path, [first, second])

            with self.assertRaisesRegex(ValueError, "inconsistent thread counts"):
                read_cpu_prefill_route_manifests((path,))

    def test_rejects_bundle_that_disagrees_with_k_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            row = self._row(shape="1.5B_FFN_Dn", k_tiles=7)
            row["bundle_signature"] = CPU_PREFILL_FULL_K_BUNDLE
            self._write(path, [row])

            with self.assertRaisesRegex(ValueError, "K tiles disagree"):
                read_cpu_prefill_route_manifests((path,))

    def test_rejects_duplicate_route_keys_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv"
            second = Path(temporary) / "second.csv"
            row = self._row(shape="1.5B_FFN_Dn", k_tiles=7)
            self._write(first, [row])
            self._write(second, [row])

            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_cpu_prefill_route_manifests((first, second))


if __name__ == "__main__":
    unittest.main()
