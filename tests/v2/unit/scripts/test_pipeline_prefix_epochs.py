#!/usr/bin/env python3
"""Check production PP prefix metadata against the independent HTTP validator.

The supplied Unit binary runs real nested rank lookup code with metadata-only
participants. Four/eight declarations and both vendor orders need no devices.
Mutation controls prove that a large sibling epoch never legitimizes missing,
reversed, truncated, duplicated, nested, or imprecise stage observations.
"""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from generation_regression_http import validate_prefix_movement_epochs, validate_prefix_outcome, PrefixState, MTPPolicy
from generation_tokens import TokenTrace


def check(envelope: dict) -> int:
    """Authenticate a real C++ value, then reject each corrupt independent copy."""
    names = ("admission_epoch_earliest", "admission_epoch_latest", "completion_movement_epoch")
    prefix = {**dict.fromkeys(names), "movement_epochs": envelope}
    validate_prefix_movement_epochs(prefix, 2)
    trace = TokenTrace((1, 2, 3), (4,), "stop")
    fresh = {**prefix, "enabled": True, "bypassed": False, "bypass_reason": "",
             "hit": False, "partial_hit": False, "requested_tokens": 3, "matched_tokens": 0,
             "matched_blocks": 0, "storage_tier": "none", "terminal_logits_restored": False,
             "terminal_hidden_restored": False, "mtp_state_restored": False, "hybrid_state_restored": False}
    validate_prefix_outcome({}, {"prefix": "fresh"},
                            {"runtime_summary": {"schema": 2, "prefix_cache": fresh}}, trace, ())
    full = {**fresh, "hit": True, "matched_tokens": 3, "matched_blocks": 1, "storage_tier": "ram",
            "terminal_logits_restored": True, "terminal_hidden_restored": True,
            "mtp_state_restored": True, "hybrid_state_restored": True}
    profile = {"prefix_state": PrefixState.HYBRID_RECURRENT.value, "mtp_policy": MTPPolicy.DYNAMIC.value}
    validate_prefix_outcome(profile, {"id": "repeat", "prefix": "full"},
                            {"runtime_summary": {"schema": 2, "prefix_cache": full}}, trace, (trace,))
    try:
        validate_prefix_outcome(profile, {"id": "repeat", "prefix": "full"},
                                {"runtime_summary": {"schema": 2, "prefix_cache": fresh}}, trace, (trace,))
    except ValueError:
        pass
    else:
        raise AssertionError("pipeline movement metadata excused a missing prefix restore")
    mutations = [
        lambda p: p.update(completion_movement_epoch=100),
        lambda p: p.pop("admission_epoch_latest"),
        lambda p: p["movement_epochs"].update(schema=True),
        lambda p: p["movement_epochs"]["stages"].pop(),
        lambda p: p["movement_epochs"]["stages"].reverse(),
        lambda p: p["movement_epochs"]["stages"].append(copy.deepcopy(p["movement_epochs"]["stages"][-1])),
        lambda p: p["movement_epochs"]["stages"][1]["identity"].update(first_layer=0),
        lambda p: p["movement_epochs"]["stages"][1]["identity"].update(terminal=False),
        lambda p: p["movement_epochs"]["stages"][1]["identity"].update(participants=[]),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(stages=[]),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(completion_movement_epoch=1),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(completion_movement_epoch=2**64),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(completion_movement_epoch=True),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(admission_epoch_earliest=-1),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].update(admission_epoch_latest=2.0),
        lambda p: p["movement_epochs"]["stages"][1]["epochs"].pop("completion_movement_epoch"),
    ]
    for mutation in mutations:
        changed = copy.deepcopy(prefix)
        mutation(changed)
        try:
            validate_prefix_movement_epochs(changed, 2)
        except ValueError:
            continue
        raise AssertionError("corrupt pipeline prefix epochs were accepted: " + repr(changed))
    for schema in (1, 3):
        try:
            validate_prefix_movement_epochs(prefix, schema)
        except ValueError:
            continue
        raise AssertionError("pipeline prefix epochs lost their schema")
    # Adjacent exact integer epochs remain metadata, never a float or hash.
    precise = copy.deepcopy(prefix)
    precise["movement_epochs"]["stages"][1]["epochs"] = dict(zip(names, (2**53 + 1, 2**53 + 2, 2**53 + 3)))
    validate_prefix_movement_epochs(precise, 2)
    return len(mutations) + 2


def main(binary: str) -> None:
    """Execute actual lookups and check every width/order emitted by C++."""
    with tempfile.TemporaryDirectory(prefix="llaminar-prefix-epochs-") as directory:
        report = Path(directory) / "native.xml"
        result = subprocess.run([binary, "--gtest_filter=*PrefixMovement*", f"--gtest_output=xml:{report}"],
                                text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        xml = ET.parse(report).getroot()
        if int(xml.get("tests", 0)) != 4 or int(xml.get("failures", 1)):
            raise AssertionError("pipeline prefix fixture omitted a width or lifecycle case")
        properties = [p for p in xml.findall(".//property") if p.attrib["name"].startswith("prefix_epochs_")]
        if len(properties) != 4:
            raise AssertionError("pipeline prefix fixture omitted a vendor order")
        controls = sum(check(json.loads(p.attrib["value"])) for p in properties)
    print(f"Pipeline prefix epochs: 4 C++ cases, 4 width/order publications, {controls} rejected corruptions passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_pipeline_prefix_epochs.py UNIT_BINARY")
    main(sys.argv[1])
