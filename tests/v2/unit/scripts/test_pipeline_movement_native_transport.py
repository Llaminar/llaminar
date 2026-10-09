#!/usr/bin/env python3
"""Bind actual C++ PP movement/counter JSON to the independent Python observer.

The supplied Unit binary emits metadata fixtures through production serializers.
No model, GPU, native transport, or inference work is performed. Both four- and
eight-participant declarations preserve adjacent large IDs and terminal MTP rows.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from generation_movement_ledger import MovementLedgerObserver, MovementRequirement, validate_movement_transport_mirrors


def main(binary: str) -> None:
    """Require both native projections to survive scope and exact-history validation."""
    with tempfile.TemporaryDirectory(prefix="llaminar-pipeline-movement-") as directory:
        report = Path(directory) / "native.xml"
        result = subprocess.run([binary, "--gtest_filter=*IndependentPublishersPreserveCollidingTransactionsAndGlobalLayers*",
                                 f"--gtest_output=xml:{report}"], text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        xml = ET.parse(report).getroot()
        if int(xml.get("tests", 0)) != 2 or int(xml.get("failures", 1)):
            raise AssertionError("native movement fixture did not execute both topology widths")
        properties = xml.findall(".//property[@name='pipeline_movement_fixture']")
        if len(properties) != 2:
            raise AssertionError("missing native pipeline movement metadata")
        for prop in properties:
            value = json.loads(prop.attrib["value"])
            for schema in (1, 2):
                observer = MovementLedgerObserver(MovementRequirement.REQUIRED)
                observer.observe({"runtime_summary": {"schema": schema, "expert_movement": value["http_movement"],
                                                      "expert_movement_topology": value["http_topology"]}})
                observer.finish()
            validate_movement_transport_mirrors(value["http_movement"], value["records"], value["terminal_movement"])
    print("C++/Python pipeline movement transport: both topology widths passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_pipeline_movement_native_transport.py UNIT_BINARY")
    main(sys.argv[1])
