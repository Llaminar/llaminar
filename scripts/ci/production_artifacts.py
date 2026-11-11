#!/usr/bin/env python3
"""Shared, fail-closed identities for production CI artifacts.

Certificates describe one immutable build and one canonical cell inventory.
They are evidence, never a substitute for running a gate. Large numerical
evidence remains a CI artifact; only compact certificates and ratchets enter
source control. No model bytes are hashed by this module.
"""
from __future__ import annotations

import hashlib
from enum import Enum
import json
import math
import os
import re
from pathlib import Path
import subprocess


def artifact_identity_path(raw_name: str) -> str:
    """Encode the shared C++ artifact owner's bounded lossless path; preserve short names."""
    if not isinstance(raw_name, str) or not raw_name or raw_name in ('.', '..') or '\x00' in raw_name:
        raise ValueError('invalid logical artifact identity')
    component = re.sub(r'[/\\:*?"<>|]', "_", raw_name)
    encoded = os.fsencode(component)
    namespace = "long-identities"
    if len(encoded) <= 255 and component != namespace:
        return component
    # Mirror the writer's byte chunks and terminal leaf. The full legacy
    # identity remains recoverable; no digest, truncation or second ID table.
    chunks = ["c" + os.fsdecode(encoded[offset:offset + 240])
              for offset in range(0, len(encoded), 240)]
    return "/".join([namespace, *chunks, "artifacts"])


class CPUISA(str, Enum):
    """The explicit whole-runtime ISA contracts of the two shippable images.

    Test inventory and physical host capabilities do not identify a tested
    binary's ISA. Consumers use its build declaration or immutable OCI label.
    """

    AVX2 = "AVX2"
    AVX512 = "AVX512"


def image_cpu_isa(identity: dict) -> CPUISA:
    """Read the admitted runtime's ISA, rejecting missing or unknown labels."""
    try:
        return CPUISA(identity["labels"]["org.llaminar.cpu_isa"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("runtime image has no supported explicit CPU ISA") from error


def model_identities(manifest: dict) -> dict:
    """Pin every declared source shard by stat identity, never payload hashing."""
    result = {}
    for filename in sorted({path for row in manifest["cells"] for path in row["model_files"]}):
        stat = Path(filename).stat()
        result[filename] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


def validate_optional_native_cohort_requirement(requirement: dict) -> None:
    """Validate the typed optional geometry shared by registration and receipt readers."""
    if isinstance(requirement, dict) and "backend_degrees" in requirement:
        degrees = requirement["backend_degrees"]
        if (set(requirement) != {"backend_degrees", "locality", "mpi_ranks"}
                or not isinstance(degrees, dict) or set(degrees) != {"cuda", "rocm"}
                or any(type(degree) is not int or degree <= 0 for degree in degrees.values())
                or requirement["locality"] not in ("original-numa-local-rank", "two-distinct-numa-local-ranks")
                or type(requirement["mpi_ranks"]) is not int or requirement["mpi_ranks"] <= 0
                or (requirement["locality"] == "two-distinct-numa-local-ranks" and requirement["mpi_ranks"] != 2)):
            raise ValueError("invalid optional mixed-family preflight geometry")
        return
    required = {"backend", "degree", "locality"}
    optional = {"native_isa", "stage_sizes", "rank_sizes", "mpi_ranks", "peer_access"}
    if (not isinstance(requirement, dict) or not required <= requirement.keys()
            or not requirement.keys() <= required | optional):
        raise ValueError("invalid optional preflight requirement")
    backend, degree, locality = (requirement[key] for key in ("backend", "degree", "locality"))
    if (backend not in ("cuda", "rocm") or type(degree) is not int or degree <= 0
            or locality not in ("original-numa-local-rank", "two-distinct-numa-local-ranks")
            or (locality == "two-distinct-numa-local-ranks" and degree != 2 and "rank_sizes" not in requirement)):
        raise ValueError("unsupported optional preflight geometry")
    if "native_isa" in requirement:
        isa = requirement["native_isa"]
        syntax = r"gfx[0-9][0-9a-f]*" if backend == "rocm" else r"sm_?[0-9]+[a-z]?"
        if not isinstance(isa, str) or re.fullmatch(syntax, isa) is None:
            raise ValueError("invalid optional preflight native ISA")
    if "peer_access" in requirement:
        if (requirement["peer_access"] not in ("directed-edge", "no-directed-pair") or degree < 2
                or (requirement["peer_access"] == "no-directed-pair" and degree != 2)
                or locality != "original-numa-local-rank"
                or "stage_sizes" in requirement or "rank_sizes" in requirement
                or type(requirement.get("mpi_ranks")) is not int or requirement["mpi_ranks"] <= 0):
            raise ValueError("invalid optional directed peer geometry")
    if "rank_sizes" in requirement:
        sizes = requirement["rank_sizes"]
        if (locality != "two-distinct-numa-local-ranks" or "stage_sizes" in requirement
                or not isinstance(sizes, list) or len(sizes) != 2
                or any(type(size) is not int or size <= 0 for size in sizes)
                or sum(sizes) != degree or type(requirement.get("mpi_ranks")) is not int
                or requirement["mpi_ranks"] != 2):
            raise ValueError("invalid optional preflight original rank geometry")
    elif "stage_sizes" in requirement:
        stages, ranks = requirement["stage_sizes"], requirement.get("mpi_ranks")
        if (locality != "original-numa-local-rank" or not isinstance(stages, list) or len(stages) < 2
                or any(type(size) is not int or size <= 0 for size in stages) or sum(stages) != degree
                or type(ranks) is not int or ranks <= 0):
            raise ValueError("invalid optional preflight stage/discovery geometry")
    elif "mpi_ranks" in requirement and "peer_access" not in requirement:
        raise ValueError("optional preflight discovery count requires stage geometry")


def optional_native_cohort_absence(requirement: dict, hardware: dict) -> dict | None:
    """Recompute exact optional eligibility from the original observed hardware.

    Local native collectives never pool devices across NUMA ranks. The explicit
    distinct-rank case uses two separate rank transport endpoints. A staged
    observed request also proves its original CPU discovery coverage and cannot
    borrow a different ISA or an ambiguous second matching NUMA owner.
    """
    validate_optional_native_cohort_requirement(requirement)
    from run_qfn_source_admission import (observed_native_isa_cohorts,
        observed_pipeline_hardware_requirement, source_hardware_devices)
    observed = source_hardware_devices(hardware)
    absence = {"qualification": "hardware-ineligible-unqualified", "requirement": requirement}
    if "backend_degrees" in requirement:
        cohorts = {backend: {} for backend in requirement["backend_degrees"]}
        unknown = {backend: [] for backend in cohorts}
        cpu_nodes = {device["numa_node"] for address, device in observed.items()
                     if address[0] == "cpu" and device["numa_node"] >= 0}
        for (backend, ordinal, _), device in observed.items():
            if backend not in cohorts:
                continue
            numa = device["numa_node"]
            if numa < 0:
                unknown[backend].append(ordinal)
            else:
                cohorts[backend].setdefault(numa, []).append(ordinal)
        eligible = {backend: {numa for numa, ordinals in groups.items()
                              if numa in cpu_nodes and len(ordinals) >= requirement["backend_degrees"][backend]}
                    for backend, groups in cohorts.items()}
        cuda_nodes, rocm_nodes = eligible["cuda"], eligible["rocm"]
        legal_owners = (bool(cuda_nodes & rocm_nodes) if requirement["locality"] == "original-numa-local-rank"
                        else any(cuda != rocm for cuda in cuda_nodes for rocm in rocm_nodes))
        if legal_owners and len(cpu_nodes) >= requirement["mpi_ranks"]:
            return None
        return {**absence, "observed_cpu_discovery_nodes": sorted(cpu_nodes),
                "observed_family_numa_cohorts": {backend: [
                    {"numa_node": numa, "ordinals": sorted(groups[numa])} for numa in sorted(groups)]
                    for backend, groups in cohorts.items()},
                "unknown_locality_ordinals": {backend: sorted(ordinals) for backend, ordinals in unknown.items()}}
    backend, degree, locality = (requirement[key] for key in ("backend", "degree", "locality"))
    if "stage_sizes" in requirement and "native_isa" in requirement:
        # This is the same observed selector used for QFN source qualification.
        # The request has no invented addresses; runtime still binds the exact
        # original discovery rank and stage devices before the native body runs.
        missing = observed_pipeline_hardware_requirement({"participants": [],
            "mpi_ranks": requirement["mpi_ranks"], "observed_pipeline_request": {
                key: requirement[key] for key in ("backend", "native_isa", "stage_sizes")}}, observed)
        return {**absence, "observed_pipeline_cohort": missing[0]} if missing else None
    cohorts: dict[int, list[int]] = {}
    unknown = []
    if "native_isa" in requirement:
        cohorts = {numa: [device["ordinal"] for device in devices]
                   for numa, devices in observed_native_isa_cohorts(observed, backend, requirement["native_isa"]).items()}
    else:
        for (family, ordinal, _), device in observed.items():
            if family != backend:
                continue
            numa = device["numa_node"]
            if numa < 0:
                unknown.append(ordinal)
            else:
                cohorts.setdefault(numa, []).append(ordinal)
    if "peer_access" in requirement:
        # Consume the complete original publication; missing facts cannot become
        # a negative capability observation or trigger another discovery probe.
        matrix = hardware.get("directed_peer_access")
        if not isinstance(matrix, list):
            raise ValueError("missing original directed peer matrix")
        facts = {}
        for edge in matrix:
            if (not isinstance(edge, dict) or set(edge) != {"backend", "accessor", "peer", "available"}
                    or edge["backend"] not in ("cuda", "rocm")
                    or type(edge["accessor"]) is not int or type(edge["peer"]) is not int
                    or edge["accessor"] == edge["peer"] or type(edge["available"]) is not bool):
                raise ValueError("malformed original directed peer matrix")
            key = (edge["backend"], edge["accessor"], edge["peer"])
            if key in facts:
                raise ValueError("duplicate original directed peer edge")
            facts[key] = edge["available"]
        expected = {(family, a, b) for family in ("cuda", "rocm")
                    for f, a, _ in observed for g, b, _ in observed
                    if f == g == family and a != b}
        if set(facts) != expected:
            raise ValueError("incomplete or foreign original directed peer matrix")
        cpu_nodes = {device["numa_node"] for address, device in observed.items()
                     if address[0] == "cpu" and device["numa_node"] >= 0}
        eligible = len(cpu_nodes) >= requirement["mpi_ranks"] and any(
            numa in cpu_nodes and len(ordinals) >= degree
            and any((facts[(backend, destination, source)] if requirement["peer_access"] == "directed-edge"
                     else not facts[(backend, destination, source)] and not facts[(backend, source, destination)])
                    for source in ordinals for destination in ordinals if source != destination)
            for numa, ordinals in cohorts.items())
        absence["observed_cpu_discovery_nodes"] = sorted(cpu_nodes)
        absence["observed_directed_peer_access"] = matrix
    elif "rank_sizes" in requirement:
        cpu_nodes = {device["numa_node"] for address, device in observed.items()
                     if address[0] == "cpu" and device["numa_node"] >= 0}
        first, second = requirement["rank_sizes"]
        eligible = any(left != right and left in cpu_nodes and right in cpu_nodes
                       and len(left_devices) >= first and len(right_devices) >= second
                       for left, left_devices in cohorts.items() for right, right_devices in cohorts.items())
        absence["observed_cpu_discovery_nodes"] = sorted(cpu_nodes)
    elif "stage_sizes" in requirement:
        cpu_nodes = {device["numa_node"] for address, device in observed.items() if address[0] == "cpu"}
        eligible_owners = [numa for numa, ordinals in cohorts.items() if numa in cpu_nodes and len(ordinals) >= degree]
        if len(eligible_owners) > 1:
            raise ValueError("observed pipeline request has ambiguous original NUMA ownership")
        eligible = bool(eligible_owners) and len(cpu_nodes) >= requirement["mpi_ranks"]
        absence["observed_cpu_discovery_nodes"] = sorted(cpu_nodes)
    else:
        eligible = (len(cohorts) >= 2 if locality == "two-distinct-numa-local-ranks"
                    else any(len(ordinals) >= degree for ordinals in cohorts.values()))
    if eligible:
        return None
    return {**absence, "observed_count": (sum(map(len, cohorts.values())) if "native_isa" in requirement
                else sum(address[0] == backend for address in observed)),
        "observed_numa_cohorts": [{"numa_node": numa, "ordinals": sorted(cohorts[numa])}
            for numa in sorted(cohorts)], "unknown_locality_ordinals": sorted(unknown)}


def validate_preflight_qualification(tests: list[str], qualification: dict) -> None:
    """Authenticate full registration coverage and exact absent physical geometries.

    The original native hardware reader owns address/schema validation. Every
    optional requirement is recomputed against that saved observation; omissions,
    invented absence and selected/unqualified overlap fail closed.
    """
    if (not isinstance(qualification, dict) or set(qualification) != {
            "schema", "registered", "selected", "requirements", "hardware", "unqualified"}
            or qualification["schema"] != 1 or type(qualification["schema"]) is not int
            or qualification["registered"] != tests):
        raise ValueError("missing or stale preflight hardware qualification")
    requirements = qualification["requirements"]
    if not isinstance(requirements, dict) or not set(requirements).issubset(tests):
        raise ValueError("foreign optional preflight geometry")
    if requirements:
        from run_qfn_source_admission import source_hardware_devices
        observed = source_hardware_devices(qualification["hardware"])
    else:
        if qualification["hardware"] is not None:
            raise ValueError("unexpected preflight hardware observation")
        observed = {}
    unavailable = {}
    for name, requirement in requirements.items():
        if not name.startswith("V2_Integration_"):
            raise ValueError("invalid optional preflight requirement")
        absence = optional_native_cohort_absence(requirement, qualification["hardware"])
        if absence is not None:
            unavailable[name] = absence
    if qualification["unqualified"] != unavailable:
        raise ValueError("preflight unqualified evidence differs from actual native hardware")
    if qualification["selected"] != [name for name in tests if name not in unavailable]:
        raise ValueError("preflight selected/unqualified partition is incomplete or overlapping")


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
    validate_preflight_qualification(tests, report.get("preflight_qualification"))


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
