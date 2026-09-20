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


def model_identities(manifest: dict) -> dict:
    """Pin every declared source shard by stat identity, never payload hashing."""
    result = {}
    for filename in sorted({path for row in manifest["cells"] for path in row["model_files"]}):
        stat = Path(filename).stat()
        result[filename] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


def validate_prerequisites(report: dict) -> None:
    """Require the canonical completed model-free receipt, not a cell pass flag."""
    tests = report.get("preflight_tests")
    if (type(report.get("preflight_return_code")) is not int or report["preflight_return_code"] != 0
            or not isinstance(tests, list)
            or not tests or any(not isinstance(name, str) for name in tests)
            or len(set(tests)) != len(tests) or type(report.get("preflight_test_count")) is not int
            or report["preflight_test_count"] != len(tests)
            or not any(name.startswith("V2_Unit_") for name in tests)
            or not any(name.startswith("V2_Integration_") for name in tests)):
        raise ValueError("canonical Unit/production-preflight receipt is incomplete or failed")


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


def runtime_image_content(inspection: dict) -> dict:
    """Identify runnable bytes and configuration independently of Docker's store.

    Classic Docker identifies a config blob; the containerd store can identify
    its manifest instead. Neither daemon-local ID is a portable comparison.
    Ordered uncompressed layer digests authenticate the entire filesystem, and
    the complete runtime config authenticates entrypoint, environment, user,
    labels and other launch defaults. Import history and storage-driver paths
    are not runtime content. Missing/null optional config fields are equivalent
    in Docker's API; non-null values (including empty strings/lists) stay exact.
    """
    def without_nulls(value):
        """Normalize only API-omitted optional fields, never runtime values."""
        if isinstance(value, dict):
            return {key: without_nulls(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [without_nulls(item) for item in value]
        return value

    if (not isinstance(inspection.get("Config"), dict)
            or not isinstance(inspection.get("RootFS"), dict)
            or inspection["RootFS"].get("Type") != "layers"
            or not isinstance(inspection["RootFS"].get("Layers"), list)
            or not inspection.get("Architecture") or not inspection.get("Os")):
        raise ValueError("Docker inspection omitted its complete runtime content")
    return {"config": without_nulls(inspection["Config"]), "rootfs": inspection["RootFS"],
            "architecture": inspection["Architecture"], "os": inspection["Os"],
            "variant": inspection.get("Variant") or "", "os_version": inspection.get("OsVersion") or ""}


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


def validate_e2e_coverage(report: dict, cells: list[dict]) -> None:
    """Authenticate every attempted cell, including failures retained for diagnosis.

    Completeness is separate from success: a failed cell still contributes its
    exact identity and evidence, but can never authorize a benchmark.
    """
    expected = {row["case"]: row["configuration"] for row in cells}
    rows = report.get("cells", [])
    if (type(report.get("correctness_passed")) is not bool
            or report.get("selected") != len(expected)
            or len(rows) != len(expected) or {row["case"] for row in rows} != set(expected)):
        raise ValueError("E2E certificate does not cover the complete selected inventory")
    for row in rows:
        code = row.get("return_code")
        outcome = row.get("outcome")
        if (type(code) is not int or row.get("configuration") != expected[row["case"]]
                or (outcome == "passed") != (code == 0)
                or (outcome == "cell_timeout") != (code == 124)
                or outcome not in ("passed", "failed", "cell_timeout")):
            raise ValueError(f"failed or stale E2E evidence: {row['case']}")
    if report["correctness_passed"] != all(row["outcome"] == "passed" for row in rows):
        raise ValueError("E2E aggregate outcome disagrees with its complete cell evidence")


def validate_e2e(report: dict, cells: list[dict]) -> None:
    """Only a complete, all-green report can certify an image."""
    validate_e2e_coverage(report, cells)
    if (report["correctness_passed"] is not True
            or any(row["outcome"] != "passed" for row in report["cells"])):
        raise ValueError("failed E2E evidence cannot certify the image")


def validate_image_e2e_coverage(report: dict, manifest: dict, image: str) -> None:
    """Validate complete image-bound evidence without discarding failed cells."""
    cells = validate_manifest(manifest, manifest["source_revision"])
    validate_e2e_coverage(report, cells)
    if (report.get("schema") != 1 or report.get("image") != image
            or report.get("source_revision") != manifest["source_revision"]
            or report.get("manifest_digest") != digest(manifest)):
        raise ValueError("full E2E evidence does not belong to this image and manifest")


def validate_image_e2e(report: dict, manifest: dict, image: str) -> None:
    """Admit benchmarks only after the entire same-image E2E suite passed.

    Checking against the full manifest, not a benchmark selector, prevents a
    passing one-cell diagnostic from authorizing the production benchmark run.
    The immutable image ID also rejects results from an earlier dirty build of
    the same Git revision.
    """
    validate_image_e2e_coverage(report, manifest, image)
    validate_e2e(report, manifest["cells"])


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
