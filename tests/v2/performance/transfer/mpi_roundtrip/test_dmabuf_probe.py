#!/usr/bin/env python3
"""Audit native DMA-BUF receipts without confusing import success with data transfer.

The complete allocation/flag/HCA Cartesian product is required even when device
GDR is unavailable. Tests exercise evidence semantics, not source-code spelling.
"""
import json
import sys
import unittest
from pathlib import Path

ALLOCATIONS = (
    "host-mapped", "host-write-combined", "host-registered", "device",
    "managed-cpu-touched", "managed-gpu-touched", "device-async",
    "vmm-device", "vmm-device-gdr", "vmm-host", "vmm-host-numa",
)
CAPABILITIES = {"device_gdr", "device_dmabuf", "host_dmabuf", "vmm_gdr", "vmm"}


def audit(text: str) -> dict:
    """Reject incomplete, duplicated or internally contradictory native receipts."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or rows[0].get("event") != "identity" or rows[-1].get("event") != "complete":
        raise ValueError("missing startup or retirement receipt")
    if sum(row.get("event") == "identity" for row in rows) != 1 or sum(row.get("event") == "complete" for row in rows) != 1:
        raise ValueError("duplicate lifecycle receipt")
    page = rows[0]["page_bytes"]
    if page <= 0 or page & (page - 1):
        raise ValueError("invalid native page geometry")
    caps = [row for row in rows if row.get("event") == "capability"]
    if len(caps) != len(CAPABILITIES) or {row["name"] for row in caps} != CAPABILITIES:
        raise ValueError("incomplete independent capability observations")
    controls = [row for row in rows if row.get("event") == "negative-control"]
    nics = {row["nic"] for row in controls}
    if not nics or len(nics) != len(controls) or any(row["code"] == 0 for row in controls):
        raise ValueError("missing invalid-FD rejection")
    expected = {(allocation, flags, nic) for allocation in ALLOCATIONS for flags in (0, 1) for nic in nics}
    cases = [row for row in rows if row.get("event") == "case"]
    actual = {(row["allocation"], row["flags"], row["nic"]) for row in cases}
    if actual != expected or len(cases) != len(expected):
        raise ValueError("incomplete or duplicate allocation/flag/HCA coverage")
    exports = [row for row in rows if row.get("event") == "export"]
    export_keys = {(row["allocation"], row["flags"], row["bytes"]) for row in exports}
    if len(exports) != len(export_keys):
        raise ValueError("duplicate export")
    for row in cases:
        if row["bytes"] <= 0 or row["bytes"] % page:
            raise ValueError("invalid export extent")
        if (row["stage"] == "registered") != (row["code"] == 0):
            raise ValueError("contradictory native result")
        if row["stage"] in ("registered", "registration") and (row["allocation"], row["flags"], row["bytes"]) not in export_keys:
            raise ValueError("NIC import without export evidence")
    successes = sum(row["stage"] == "registered" for row in cases)
    if rows[-1]["registration_successes"] != successes or rows[-1]["transfer_qualified"] is not False:
        raise ValueError("registration count mismatch or unproved transfer claim")
    return {"host": rows[0]["host"], "cases": len(cases), "registrations": successes,
            "capabilities": {row["name"]: {"code": row["code"], "value": row["value"]} for row in caps},
            "transfer_qualified": False}


class EvidenceTests(unittest.TestCase):
    """Exercise capability independence, native controls and complete matrix accounting."""

    def fixture(self):
        """Construct a receipt with supported host import but unavailable device GDR."""
        rows = [{"event": "identity", "host": "fixture", "page_bytes": 4096}]
        rows += [{"event": "capability", "name": name, "code": 0, "value": int(name == "host_dmabuf")} for name in sorted(CAPABILITIES)]
        rows += [{"event": "negative-control", "nic": "hca", "code": 9}]
        for name in ALLOCATIONS:
            for flags in (0, 1):
                supported = name == "host-mapped" and flags == 0
                if supported:
                    rows.append({"event": "export", "allocation": name, "flags": flags, "bytes": 1048576, "fd_inode": 1})
                rows.append({"event": "case", "allocation": name, "flags": flags, "nic": "hca", "bytes": 1048576,
                             "stage": "registered" if supported else "export", "code": 0 if supported else 801})
        rows.append({"event": "complete", "registration_successes": 1, "transfer_qualified": False})
        return rows

    def check_rows(self, rows):
        """Run the same auditor used for the native JSON-lines artifact."""
        return audit("\n".join(json.dumps(row) for row in rows))

    def test_host_support_is_independent_of_device_support(self):
        result = self.check_rows(self.fixture())
        self.assertEqual(result["registrations"], 1)
        self.assertEqual(result["capabilities"]["device_gdr"]["value"], 0)

    def test_missing_combination_is_rejected(self):
        rows = self.fixture(); del rows[-2]
        with self.assertRaises(ValueError): self.check_rows(rows)

    def test_missing_retirement_is_rejected(self):
        with self.assertRaises(ValueError): self.check_rows(self.fixture()[:-1])

    def test_duplicate_combination_is_rejected(self):
        rows = self.fixture(); rows.insert(-1, dict(rows[-2]))
        with self.assertRaises(ValueError): self.check_rows(rows)

    def test_import_without_export_is_rejected(self):
        rows = [row for row in self.fixture() if row["event"] != "export"]
        with self.assertRaises(ValueError): self.check_rows(rows)

    def test_registration_never_certifies_transfer(self):
        rows = self.fixture(); rows[-1]["transfer_qualified"] = True
        with self.assertRaises(ValueError): self.check_rows(rows)

    def test_wrong_success_count_is_rejected(self):
        rows = self.fixture(); rows[-1]["registration_successes"] = 2
        with self.assertRaises(ValueError): self.check_rows(rows)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--audit":
        print(json.dumps(audit(Path(sys.argv[2]).read_text()), indent=2))
    else:
        unittest.main()
