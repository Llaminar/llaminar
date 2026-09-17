#!/usr/bin/env python3
"""Audit unapproved native acquisition without executing or certifying a cell.

The existing inventory reader owns membership and serial-control relationships.
Reports are supplied explicitly, oldest first; a later failed attempt cannot
be hidden by an earlier pass. Every counted pass revalidates its saved HTTP
responses and exact serial tokens. Aggregate failures may contain valid earlier
cells, but incomplete individual observations never count. This tool neither
approves provenance nor writes expected tokens, fetches corpora or issues image
certificates. It is a read-only feedback aid before the separate corpus review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation_regression_http import MTPPolicy, generation_profile, observation_traces
from generation_tokens import compare_tokens
from production_artifacts import digest
from model_parity_inventory import InventoryScope
from run_model_parity_generation import RunMode, select_cells


def audit(manifest: Path, reports: list[Path]) -> dict:
    """Join exact current cells to completed evidence, retaining failed retries.

    Report order is explicit acquisition chronology, not filesystem mtime. A
    duplicated passing case is rejected instead of inflating the coverage count
    or silently choosing one potentially different answer. Saved observations
    remain inside their original report directory and retain their configuration;
    this audit never remaps old evidence into a newer model or runtime policy.
    Explicit acquisition inputs may span source revisions when their complete
    configurations still agree. Preserve those revisions as review provenance;
    such a union is not a fresh proof of the current source or shipping image.
    """
    inventory = json.loads(manifest.read_text())
    args = argparse.Namespace(manifest=manifest, source_revision=inventory.get("source_revision"),
                              mode=RunMode.COMPARE, backend=".*", campaign=".*", cell=".*")
    if not isinstance(args.source_revision, str) or not args.source_revision:
        raise ValueError("acquisition inventory omitted its source revision")
    # Historical acquisition includes fixed depths. Auditing those immutable
    # observations does not execute them in the routine gate.
    canonical = {cell.exact: cell for cell in select_cells(args, InventoryScope.ALL)}
    paths = [path.resolve(strict=True) for path in reports]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("acquisition requires distinct explicit report files in chronological order")
    passes, latest, provenance = {}, {}, []
    for path in paths:
        report = json.loads(path.read_text())
        if (type(report.get("schema")) is not int or report["schema"] != 1
                or not isinstance(report.get("source_revision"), str) or not report["source_revision"]
                or report.get("certification_eligible") is not False
                or report.get("mode") not in {mode.value for mode in RunMode}
                or type(report.get("complete")) is not bool or type(report.get("passed")) is not bool
                or type(report.get("selected")) is not int or report["selected"] <= 0
                or not isinstance(report.get("cells"), list) or len(report["cells"]) > report["selected"]):
            raise ValueError("malformed or foreign native acquisition report: " + str(path))
        if report["passed"] and (not report["complete"] or len(report["cells"]) != report["selected"]):
            raise ValueError("passing acquisition report omitted completed cells")
        provenance.append({"report": str(path), "report_digest": digest(report),
                           "source_revision": report["source_revision"]})
        seen = set()
        for row in report["cells"]:
            if (not isinstance(row, dict) or not isinstance(row.get("case"), str)
                    or row["case"] not in canonical or row["case"] in seen
                    or type(row.get("return_code")) is not int
                    or row.get("configuration") != canonical[row["case"]].configuration):
                raise ValueError("acquisition row is duplicated, unknown or differs from the canonical configuration")
            exact = row["case"]
            seen.add(exact)
            cell = canonical[exact]
            serial = MTPPolicy(generation_profile(cell.configuration)["mtp_policy"]) is MTPPolicy.OFF
            if report["mode"] == RunMode.COLLECT.value and not serial:
                raise ValueError("serial acquisition report contains an MTP-specific answer")
            passed = row["return_code"] == 0 and row.get("evidence_error") is None
            if report["passed"] and not passed:
                raise ValueError("passing aggregate contains a failed cell")
            latest[exact] = passed
            if not passed:
                continue
            if exact in passes:
                raise ValueError("duplicate passing acquisition evidence: " + exact)
            tests = report.get("preflight_tests")
            if (type(report.get("preflight_return_code")) is not int or report["preflight_return_code"] != 0
                    or not isinstance(tests, list) or any(not isinstance(name, str) for name in tests)
                    or len(set(tests)) != len(tests) or report.get("preflight_test_count") != len(tests)
                    or not any(name.startswith("V2_Unit_") for name in tests)
                    or not any(name.startswith("V2_Integration_") for name in tests)):
                raise ValueError("passing acquisition lacks its successful shared Unit/preflight receipt")
            artifact = row.get("artifacts")
            if not isinstance(artifact, str) or not Path(artifact).is_absolute():
                raise ValueError("acquisition omitted its original absolute artifact directory")
            observed_path = (Path(artifact) / "generation/observations.json").resolve(strict=True)
            if not observed_path.is_relative_to(path.parent):
                raise ValueError("acquisition observations escape their original report directory")
            observed = json.loads(observed_path.read_text())
            traces = observation_traces(cell.configuration, observed)
            # Retain only immutable token traces after the complete observer
            # validates this cell. Large movement journals need not accumulate
            # in the auditor or be re-parsed for every speculative policy.
            passes[exact] = traces

    by_id = {cell.configuration["id"]: exact for exact, cell in canonical.items()}
    for exact, actual in passes.items():
        cell = canonical[exact]
        control = by_id[generation_profile(cell.configuration)["serial_control_id"]]
        if control not in passes or not latest[control]:
            raise ValueError("acquisition comparison lacks a currently green canonical serial control: " + exact)
        # select_cells authenticated this complete canonical Off relationship;
        # observation_traces checked its saved configuration and runtime proof.
        # No historical configuration or mount alias is substituted here.
        expected = passes[control]
        if any(compare_tokens(expected[name], trace) is not None for name, trace in actual.items()):
            raise ValueError("acquisition tokens differ from the original serial control: " + exact)

    green = {exact for exact in passes if latest[exact]}
    unresolved = {exact for exact, passed in latest.items() if not passed}
    groups = {}
    for name, is_serial in (("serial", True), ("mtp", False)):
        expected = {exact for exact, cell in canonical.items()
                    if (MTPPolicy(generation_profile(cell.configuration)["mtp_policy"]) is MTPPolicy.OFF) == is_serial}
        groups[name] = {"expected": len(expected), "passed": len(expected & green),
                        "not_complete": len(expected - green)}
    revisions = sorted({entry["source_revision"] for entry in provenance})
    return {"schema": 1, "kind": "unapproved_generation_acquisition_audit", "certification_eligible": False,
            "source_revisions": revisions, "mixed_source_revisions": len(revisions) > 1,
            "inventory_digest": digest(inventory), "reports": provenance, "counts": groups,
            "passed_cases": sorted(green), "unresolved_failed_cases": sorted(unresolved),
            "unseen_cases": sorted(set(canonical) - latest.keys()),
            "acquisition_complete": len(green) == len(canonical)}


def main(argv: list[str] | None = None) -> int:
    """Print evidence only; a partial, valid acquisition is not an audit error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="Original native reports in oldest-to-newest order; no glob or auto-selection")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.manifest, args.reports), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
