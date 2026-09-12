#!/usr/bin/env python3
"""Shared, fail-closed identities for production CI artifacts.

Certificates describe one immutable build and one canonical cell inventory.
They are evidence, never a substitute for running a gate. Large numerical
evidence remains a CI artifact; only compact certificates and ratchets enter
source control. No model bytes are hashed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess


def digest(value: object) -> str:
    """Authenticate small structured metadata using a deterministic encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON atomically, retaining the prior file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value: object) -> float:
    """Reject missing, boolean, non-finite and non-positive performance values."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"expected a finite positive measurement, got {value!r}")
    return float(value)


def image_identity(image: str) -> dict:
    """Resolve a mutable Docker tag once; every test subsequently uses its ID."""
    data = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    labels = data["Config"].get("Labels") or {}
    if not data["Id"].startswith("sha256:"):
        raise ValueError("Docker did not return an immutable image identity")
    return {"id": data["Id"], "layers": data["RootFS"]["Layers"], "labels": labels}


def validate_manifest(manifest: dict, revision: str) -> list[dict]:
    """Accept only nonempty, unique, revision-bound E2E selections."""
    if manifest.get("schema") != 1 or manifest.get("source_revision") != revision:
        raise ValueError("canonical E2E manifest has the wrong schema/source revision")
    cells = manifest.get("cells", [])
    if not cells or len({row["case"] for row in cells}) != len(cells):
        raise ValueError("canonical E2E inventory is empty or duplicated")
    for row in cells:
        config = row["configuration"]
        if config.get("model_parity_schema") != 1 or not config.get("e2e"):
            raise ValueError("benchmark/certificate selection contains an untagged cell")
        if config["model"] not in row.get("model_files", []):
            raise ValueError("canonical selection omits its complete model-file manifest; re-export it")
    return cells


def validate_e2e(report: dict, cells: list[dict]) -> None:
    """A subset, duplicate, stale configuration or exit-zero omission is red."""
    expected = {row["case"]: row["configuration"] for row in cells}
    rows = report.get("cells", [])
    if (report.get("correctness_passed") is not True or report.get("selected") != len(expected)
            or len(rows) != len(expected) or {row["case"] for row in rows} != set(expected)):
        raise ValueError("E2E certificate does not cover the complete selected inventory")
    for row in rows:
        if (row.get("return_code") != 0 or row.get("outcome") != "passed"
                or row.get("configuration") != expected[row["case"]]):
            raise ValueError(f"failed or stale E2E evidence: {row['case']}")


def validate_image_e2e(report: dict, manifest: dict, image: str) -> None:
    """Admit benchmarks only after the entire same-image E2E suite passed.

    Checking against the full manifest, not a benchmark selector, prevents a
    passing one-cell diagnostic from authorizing the production benchmark run.
    The immutable image ID also rejects results from an earlier dirty build of
    the same Git revision.
    """
    cells = validate_manifest(manifest, manifest["source_revision"])
    validate_e2e(report, cells)
    if (report.get("schema") != 1 or report.get("image") != image
            or report.get("source_revision") != manifest["source_revision"]
            or report.get("manifest_digest") != digest(manifest)):
        raise ValueError("full E2E evidence does not belong to this image and manifest")


def ratchet(baseline: dict, results: list[dict]) -> tuple[dict, list[dict]]:
    """Compare all samples before proposing an upward-only high-water update.

    Each key includes hardware, workload and full cell policy, but deliberately
    excludes the source/image revision: performance must be comparable across
    implementations. A first observation is explicitly a new baseline, never
    a claim to have beaten a prior implementation. The caller may persist the
    proposal only after the complete production pipeline passes.
    """
    if baseline.get("schema") != 1:
        raise ValueError("unsupported benchmark high-water schema")
    threshold = positive(baseline["regression_threshold_pct"])
    if threshold >= 100:
        raise ValueError("regression threshold must be below 100 percent")
    entries = dict(baseline["entries"])
    comparisons = []
    seen = set()
    for row in results:
        key = digest(row["identity"])
        if key in seen:
            raise ValueError("duplicate benchmark identity")
        seen.add(key)
        previous = entries.get(key)
        if previous is not None and previous["identity"] != row["identity"]:
            raise ValueError("high-water identity mismatch")
        marks = {}
        for phase in ("prefill", "decode"):
            current = positive(row["tokens_per_second"][phase])
            prior = positive(previous["tokens_per_second"][phase]) if previous else None
            passed = prior is None or current >= prior * (1 - threshold / 100)
            comparisons.append({"case": row["case"], "phase": phase, "current": current,
                                "high_water": prior, "passed": passed,
                                "status": "new_baseline" if prior is None else
                                          "improved" if current > prior else
                                          "within_tolerance" if passed else "regressed"})
            marks[phase] = max(current, prior or current)
        entries[key] = {"identity": row["identity"], "tokens_per_second": marks}
    if not seen:
        raise ValueError("empty benchmark run cannot advance high-water marks")
    return {**baseline, "entries": entries}, comparisons
