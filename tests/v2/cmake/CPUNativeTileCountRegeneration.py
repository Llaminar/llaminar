#!/usr/bin/env python3
"""Regenerate the committed GNU-14 NEON tile fingerprints from driver attempts: a maintainer tool, never run by CI.

The committed counts (CPUNativeTileOrderedAddSites.json and CPUNativeTileChainCounts.json) are structural
fingerprints of the tile programs: they fail every attempt until a reviewed diff changes them. After a kernel
change, this tool writes them from the checker's own analysis of driver attempts (CPUNativeTileProofDriver.py
`build` output directories), so no count is ever edited by hand:

- ordered-add sites: len(ordered_adds[key]) of the attempt's record (seal or unqualified analysis record);
- tile_chain: the record's chain_witness.tile_chain;
- fused: fused_evidence() (pre-link scope, probe copy) of the attempt's bound object, disassembly and the
  assembly whose `.loc` locations the record names (the Release line witness when present).

Every attempt of one envelope must agree on all three parts (e.g. its Integration and Release attempts), else the
tool fails. Envelopes without an attempt are left unchanged; other compiler identities are never touched. Each
changed entry is printed (before -> after) for the commit message.

Usage: CPUNativeTileCountRegeneration.py --identity GNU-14 DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence",
                                               HERE / "CPUNativeTileRegisterEvidence.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def attempt(directory: Path) -> tuple[str, str, dict]:
    """(identity, mode, {"sites", "tile_chain", "fused"}) of one driver attempt directory."""
    proof = directory / "proof.json"
    document = json.loads((proof if proof.exists() else ev.analysis_path(proof)).read_text())
    located, _ = ev.located_assembly(proof, document)
    fused = ev.fused_evidence((directory / "proof.o").read_bytes(), (directory / "proof.dis").read_text(),
                              located.read_text(errors="replace"), ev.PROBE_COPY,
                              ev.closure_sources(document["inputs"]), scope="prelink")
    suffix = f"@{ev.PROBE_COPY}"
    return document["identity"], document["mode"], {
        "sites": {key: len(indices) for key, indices in sorted(document["ordered_adds"].items())},
        "tile_chain": dict(sorted(document["chain_witness"]["tile_chain"].items())),
        "fused": {key[:-len(suffix)]: value for key, value in sorted(fused.items())}}


def changes(name: str, before: dict, after: dict) -> list[str]:
    return [f"{name} {key}: {json.dumps(before.get(key))} -> {json.dumps(after.get(key))}"
            for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--identity", required=True)
    parser.add_argument("attempts", nargs="+", type=Path)
    args = parser.parse_args()
    found: dict[str, dict] = {}
    for directory in args.attempts:
        identity, mode, parts = attempt(directory)
        if identity != args.identity:
            raise SystemExit(f"{directory}: attempt of compiler {identity}, not {args.identity}")
        if mode in found and found[mode] != parts:
            raise SystemExit(f"{directory}: the {mode} attempts disagree; regenerate from one consistent tree")
        found[mode] = parts
    sites_path, chain_path = ev.ORDERED_ADD_SITES_PATH, ev.CHAIN_COUNTS_PATH
    sites, chain = json.loads(sites_path.read_text()), json.loads(chain_path.read_text())
    for mode, parts in sorted(found.items()):
        for line in changes(f"{mode} ordered-add sites", sites[args.identity][mode], parts["sites"]) + \
                changes(f"{mode} tile_chain", chain[args.identity][mode]["tile_chain"], parts["tile_chain"]) + \
                changes(f"{mode} fused", chain[args.identity][mode]["fused"], parts["fused"]):
            print(line)
        sites[args.identity][mode] = parts["sites"]
        chain[args.identity][mode]["tile_chain"] = parts["tile_chain"]
        chain[args.identity][mode]["fused"] = parts["fused"]
    # The committed layouts: two-space ordered-add sites, one-space chain counts.
    sites_path.write_text(json.dumps(sites, indent=2) + "\n")
    chain_path.write_text(json.dumps(chain, indent=1) + "\n")


if __name__ == "__main__":
    main()
