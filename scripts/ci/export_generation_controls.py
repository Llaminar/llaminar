#!/usr/bin/env python3
"""Archive existing HTTP controls for review; never generate or approve answers.

The full canonical inventory owns prompts, seeds and cell/control mappings.
The acquisition auditor authenticates saved HTTP responses, live-path evidence
and serial/MTP agreement before export. This produces an explicitly unapproved
corpus, not an image certificate or an ISA approval. Publish it in the optional
corpus repository; independent numerical/ISA review must precede any approval
catalog change. Existing generations are immutable and cannot be overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from audit_generation_acquisition import audit
from generation_corpus import portable_inventory
from generation_regression_http import MTPPolicy, generation_profile
from production_artifacts import model_identities


def export_controls(manifest: Path, reports: list[Path], cpu_isa: str) -> dict:
    """Retain exact acquired tokens and explicit mappings, excluding old telemetry.

    The declared acquisition ISA is provenance only. It never approves another
    ISA or authorizes CI to replace its expected answers. An incomplete or stale
    acquisition fails before any payload is emitted.
    """
    if cpu_isa not in ("AVX2", "AVX512"):
        raise ValueError("control export requires the acquisition CPU ISA")
    evidence = audit(manifest, reports)
    if evidence["acquisition_complete"] is not True:
        raise ValueError("control export requires complete audited acquisition")
    inventory = json.loads(manifest.read_text())
    controls = []
    for path in reports:
        report = json.loads(path.read_text())
        for row in report["cells"]:
            record = row["configuration"]
            if (row["return_code"] != 0 or row.get("evidence_error") is not None
                    or MTPPolicy(generation_profile(record)["mtp_policy"]) is not MTPPolicy.OFF):
                continue
            observed = json.loads((Path(row["artifacts"]) / "generation/observations.json").read_text())
            requests = []
            for request in observed["requests"]:
                response = request["response"]
                # Runtime receipts must be produced again by the candidate.
                # Keep only terminal token/termination evidence as the answer.
                terminal = {key: copy.deepcopy(response[key]) for key in ("object", "token_ids", "usage")}
                terminal["choices"] = [{"finish_reason": response["choices"][0]["finish_reason"]}]
                requests.append({"id": request["id"], "response": terminal})
            controls.append({"id": record["id"], "requests": requests})
    return {"schema": 1, "kind": "unapproved_generation_controls", "cpu_isa": cpu_isa,
            "inventory": portable_inventory(inventory, model_identities(inventory)),
            "controls": sorted(controls, key=lambda row: row["id"]),
            "acquisition": {"counts": evidence["counts"], "source_revisions": evidence["source_revisions"],
                "inventory_digest": evidence["inventory_digest"],
                "reports": [{"name": Path(row["report"]).parent.name,
                             "report_digest": row["report_digest"], "source_revision": row["source_revision"]}
                            for row in evidence["reports"]]}}


def main(argv: list[str] | None = None) -> int:
    """Write a new review artifact without altering source approval or old tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--cpu-isa", choices=("AVX2", "AVX512"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("control generation already exists: " + str(args.output))
    document = export_controls(args.manifest, args.reports, args.cpu_isa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(document, stream, separators=(",", ":"), sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
