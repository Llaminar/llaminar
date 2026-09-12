#!/usr/bin/env python3
"""Export canonical model cells without executing numerical or generation tests.

CTest owns exact membership; GoogleTest's typed JSON owns each configuration.
Generation regression consumes the entire inventory, while HTTP certification
and benchmarks consume only its explicitly tagged subset. Neither consumer
may reconstruct a topology from a test name or maintain another axis expander.
Discovery never loads model tensors, stages weights, or initializes devices.
"""
from __future__ import annotations

import argparse
import dataclasses
from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_production_parity_campaigns as parity
from production_artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]


class InventoryScope(str, Enum):
    """Select a projection of existing cells, never a different configuration."""
    ALL = "all"
    E2E = "e2e"

    def accepts(self, record: dict) -> bool:
        """Require an explicit eligibility field even for an untagged cell."""
        if "e2e" not in record:
            raise ValueError("canonical cell omitted its explicit E2E eligibility")
        if record["e2e"] is not None and not isinstance(record["e2e"], dict):
            raise ValueError("canonical E2E eligibility must be a profile or null")
        return self == InventoryScope.ALL or record["e2e"] is not None


def parse_parameters(output: str) -> dict[str, dict]:
    """Read complete typed parameters; console output can truncate these values."""
    records = {}
    for suite in json.loads(output)["testsuites"]:
        for test in suite["testsuite"]:
            if not test["name"].startswith("ProductionParity/"):
                continue
            exact = suite["name"] + "." + test["name"]
            record = json.loads(test["value_param"])
            if record.get("model_parity_schema") != 1:
                raise ValueError(f"unsupported parameter schema: {exact}")
            InventoryScope.ALL.accepts(record)
            if exact in records:
                raise ValueError(f"duplicate parameter: {exact}")
            records[exact] = record
    return records


def source_revision(args: argparse.Namespace) -> str:
    """Containers without Git receive the revision of their admitted snapshot."""
    return getattr(args, "source_revision", None) or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def matrix_binary(campaign: parity.CampaignCell) -> Path:
    """Resolve one registered executable, independently of launcher topology."""
    binaries = [Path(arg) for arg in campaign.command
                if Path(arg).name.startswith("v2_integration_parity_")
                and Path(arg).name.endswith("_matrix")]
    if len(binaries) != 1:
        raise ValueError(f"ambiguous matrix executable: {campaign.name}")
    return binaries[0]


def discover(args: argparse.Namespace, scope: InventoryScope) -> list[tuple[parity.CampaignCell, str, dict]]:
    """Join exact CTest membership to typed metadata once per matrix executable.

    A saved manifest declares whether it covers all cells or only E2E tags.
    An E2E-only manifest must never masquerade as the full generation matrix.
    Model stat identity handles mount aliases without reading GGUF payloads.
    """
    if not isinstance(scope, InventoryScope):
        raise TypeError("inventory scope must be explicit and typed")
    selected = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("schema") != 1 or manifest.get("source_revision") != source_revision(args):
            raise ValueError("canonical manifest must be generated from this source revision")
        # Existing E2E manifests predate the general projection. They remain
        # valid E2E inputs, but cannot assert coverage of untagged cells.
        declared = InventoryScope(manifest.get("scope", "e2e"))
        if scope == InventoryScope.ALL and declared != InventoryScope.ALL:
            raise ValueError("the full matrix requires an all-cell manifest")
        seen = set()
        for row in manifest["cells"]:
            exact = row["case"]
            if exact in seen:
                raise ValueError(f"duplicate canonical manifest cell: {exact}")
            seen.add(exact)
            record = row["configuration"]
            if record.get("model_parity_schema") != 1 or not declared.accepts(record):
                raise ValueError(f"invalid configuration in {declared.value} manifest: {exact}")
            if record.get("model") not in row.get("model_files", []):
                raise ValueError(f"canonical manifest omitted the declared model: {exact}")
            if (scope.accepts(record) and re.fullmatch(args.backend, row["backends"])
                    and re.fullmatch(args.campaign, row["campaign"])
                    and re.fullmatch(args.cell, exact)):
                campaign = parity.CampaignCell(row["campaign"],
                    parity.CampaignGroup(row["backends"], "ALL"),
                    model_files=tuple(row["model_files"]))
                selected.append((campaign, exact, record))
    else:
        # Audit the complete registration for every binary we inspect. Applying
        # selectors before this join would hide cells silently dropped by a
        # broken CTest name parser, even when GoogleTest still contains them.
        registered = parity.discover_campaigns(args.build_dir)
        registered_by_binary = {}
        for campaign in registered:
            names = registered_by_binary.setdefault(matrix_binary(campaign), set())
            if names.intersection(campaign.gtest_cases):
                raise ValueError(f"duplicate canonical CTest cell in {campaign.name}")
            names.update(campaign.gtest_cases)
        campaigns = [campaign for campaign in registered
                     if re.fullmatch(args.backend, campaign.group.backends)
                     and re.fullmatch(args.campaign, campaign.name)]
        inventories = {}
        seen = set()
        for campaign in campaigns:
            binary = matrix_binary(campaign)
            if binary not in inventories:
                with tempfile.TemporaryDirectory(prefix="llaminar-cell-discovery-") as directory:
                    output = Path(directory) / "parameters.json"
                    subprocess.run(
                        [str(binary), "--gtest_list_tests", f"--gtest_output=json:{output}"],
                        cwd=campaign.working_directory,
                        env={**os.environ, "LLAMINAR_FORCE_CPU_ONLY_STARTUP": "1",
                             "HWLOC_COMPONENTS": "-gl,-opencl",
                             "OMPI_MCA_btl_vader_single_copy_mechanism": "none"},
                        check=True, capture_output=True, text=True, timeout=30)
                    inventories[binary] = parse_parameters(output.read_text())
                actual = set(inventories[binary])
                expected = registered_by_binary[binary]
                if actual != expected:
                    raise ValueError(f"CTest/GoogleTest inventory mismatch for {binary}: "
                                     f"unregistered={sorted(actual - expected)} "
                                     f"missing_from_binary={sorted(expected - actual)}")
            for exact in campaign.gtest_cases:
                if exact in seen:
                    raise ValueError(f"duplicate canonical CTest cell: {exact}")
                seen.add(exact)
                if exact not in inventories[binary]:
                    raise ValueError(f"stale CTest registration, rebuild {binary}: {exact}")
                record = inventories[binary][exact]
                if not scope.accepts(record) or not re.fullmatch(args.cell, exact):
                    continue
                matches = [path for path in campaign.model_files if Path(record["model"]).samefile(path)]
                if not matches:
                    raise ValueError(f"model is not in the canonical GGUF manifest: {exact}")
                # One aggregate can contain a fine-tune as well as its base
                # model. Retain only this cell's model and complete split set.
                model_campaign = dataclasses.replace(campaign, model_files=(matches[0],))
                model_campaign = dataclasses.replace(model_campaign, model_files=tuple(
                    str(path) for path in parity.selected_model_files([model_campaign])))
                # Shard expansion resolves symlinks. The model field must use
                # that same spelling, or our exported manifest would reject its
                # own model even though both paths name identical file bytes.
                primary = next(path for path in model_campaign.model_files
                               if Path(path).samefile(matches[0]))
                record = {**record, "model": primary}
                selected.append((model_campaign, exact, record))
    if not selected:
        raise ValueError(f"no {scope.value} canonical cells selected")
    return selected


def export_manifest(selected: list, revision: str, scope: InventoryScope) -> dict:
    """Retain the projection's scope and full configurations for other drivers."""
    if not selected or len({exact for _, exact, _ in selected}) != len(selected):
        raise ValueError("cannot export an empty or duplicate canonical inventory")
    for _, exact, record in selected:
        if not scope.accepts(record):
            raise ValueError(f"cell is outside the exported scope: {exact}")
    return {"schema": 1, "source_revision": revision, "scope": scope.value,
            "cells": [{"campaign": campaign.name, "backends": campaign.group.backends,
                       "case": exact, "configuration": record,
                       "model_files": list(campaign.model_files)}
                      for campaign, exact, record in selected]}


def main(argv: list[str] | None = None) -> int:
    """List/export metadata only; this command cannot run or certify inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_v2_integration")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", type=InventoryScope, choices=list(InventoryScope), default=InventoryScope.ALL)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--backend", default=".*")
    parser.add_argument("--campaign", default=".*")
    parser.add_argument("--cell", default=".*")
    args = parser.parse_args(argv)
    selected = discover(args, args.scope)
    if args.export_manifest:
        write_json(args.export_manifest, export_manifest(selected, source_revision(args), args.scope))
    for _, exact, _ in selected:
        print(exact)
    print(f"[model-parity-inventory] scope={args.scope.value} cells={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
