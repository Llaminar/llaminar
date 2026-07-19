#!/usr/bin/env python3
"""Regressions for typed failed-seal development transactions."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.cpu_prefill_burned_seal import (  # noqa: E402
    BurnedSealProvenance,
    read_cpu_prefill_burned_seal_transaction,
    write_cpu_prefill_burned_seal_transaction,
)


class CPUNativeVNNIPrefillBurnedSealTest(unittest.TestCase):
    """Require complete original provenance and immutable payload identities."""

    @staticmethod
    def _provenance() -> BurnedSealProvenance:
        """Return one complete production-like provenance fixture."""

        return BurnedSealProvenance(
            run_id="unit-burned-seal",
            git_revision="0123456789abcdef",
            build_id="sha256:" + "1" * 64,
            compiler_id="unit-compiler",
            architecture_class="unit-cpu",
            device_name="unit-device",
            driver_runtime="unit-runtime",
            serial_m1_policy_hash="sha256:" + "2" * 64,
        )

    def test_round_trip_authenticates_every_failed_seal_payload(self) -> None:
        """The transaction must preserve raw evidence and its measuring build."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "sealed.csv"
            timing = root / "sealed.timing.csv"
            witness = root / "witness.json"
            manifest = root / "burned.json"
            aggregate.write_text("aggregate\n", encoding="utf-8")
            timing.write_text("timing\n", encoding="utf-8")
            witness.write_text("{}\n", encoding="utf-8")

            written = write_cpu_prefill_burned_seal_transaction(
                manifest,
                aggregate,
                timing,
                witness,
                self._provenance(),
            )
            loaded = read_cpu_prefill_burned_seal_transaction(manifest)

            self.assertEqual(loaded, written)
            self.assertEqual(loaded.provenance, self._provenance())
            self.assertEqual(
                loaded.aggregate.resolve(manifest).read_text(encoding="utf-8"),
                "aggregate\n",
            )

    def test_payload_or_provenance_tampering_is_rejected(self) -> None:
        """Neither timing bytes nor typed provenance may drift after creation."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "sealed.csv"
            timing = root / "sealed.timing.csv"
            witness = root / "witness.json"
            manifest = root / "burned.json"
            aggregate.write_text("aggregate\n", encoding="utf-8")
            timing.write_text("timing\n", encoding="utf-8")
            witness.write_text("{}\n", encoding="utf-8")
            write_cpu_prefill_burned_seal_transaction(
                manifest,
                aggregate,
                timing,
                witness,
                self._provenance(),
            )

            timing.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload digest changed"):
                read_cpu_prefill_burned_seal_transaction(manifest)

            timing.write_text("timing\n", encoding="utf-8")
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["provenance"]["build_id"] = "sha256:" + "3" * 64
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transaction digest changed"):
                read_cpu_prefill_burned_seal_transaction(manifest)


if __name__ == "__main__":
    unittest.main()
