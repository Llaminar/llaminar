#!/usr/bin/env python3
"""Read explicitly reviewed serial-token corpora without acquiring new answers.

The source-controlled approval pin authenticates one small corpus document.
This consumer cannot collect, approve, repair, fetch or publish a baseline.
Its provenance references reviewed HF, serial and MTP evidence; the reader is
not a substitute for those acquisition-time proofs. Runtime path, movement,
prefix and MTP evidence must still be collected from every candidate server.

Model mount paths are deployment details. Portable identity retains the full
canonical configuration and each declared shard's filename and byte length,
using the pipeline's existing stat pins rather than rereading GGUF payloads.
Stat pins still protect the current run against replacement. Filename/length
metadata alone does not prove weight-content equivalence: that is established
by the independent numerical evidence reviewed before approving the corpus.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re

from generation_regression_http import (
    MTPPolicy, generation_profile, serial_workload_identity, validate_partial_prompt,
)
from generation_tokens import GenerationWorkload, TokenTrace, compare_tokens
from model_parity_inventory import InventoryScope
from production_artifacts import digest


APPROVAL_CATALOG = Path("scripts/ci/approved_generation_corpora.json")


def _metadata_digest(value: object) -> bool:
    """Recognize a complete metadata identity, never a model-content checksum."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ApprovedCorpusPin:
    """Reviewed caller-owned identity; never derive this from candidate output.

    The consuming pipeline must load this pin from its admitted source snapshot.
    An arbitrary digest computed from today's observations is not an approval.
    Keeping the expected ISA here prevents a corpus selecting its own contract.
    """
    cpu_isa: str
    document_digest: str

    def __post_init__(self) -> None:
        """Reject incomplete approval intent before opening a corpus payload."""
        if not isinstance(self.cpu_isa, str) or not self.cpu_isa or not _metadata_digest(self.document_digest):
            raise ValueError("generation corpus requires an explicit ISA and reviewed metadata pin")


def portable_inventory(inventory: dict, model_sources: dict) -> list[dict]:
    """Remove mount spelling only, retaining every canonical cell and shard.

    The source revision is provenance, not a compatibility key: otherwise
    every source improvement would invalidate its own regression baseline.
    Model descriptors consume the canonical admission ledger's five stat fields
    (device, inode, bytes, mtime_ns, ctime_ns). No second file scan, capacity
    ledger, model-name parser or topology expander is introduced here.
    """
    if (not isinstance(inventory, dict) or type(inventory.get("schema")) is not int
            or inventory["schema"] != 1 or inventory.get("scope") != "all"
            or not isinstance(inventory.get("cells"), list) or not inventory["cells"]
            or not isinstance(model_sources, dict)):
        raise ValueError("generation corpus admission requires the complete canonical inventory and model pins")
    result, identities, cases, declared_files = [], set(), set(), set()
    for row in inventory["cells"]:
        if (not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                for key in ("case", "campaign", "backends"))):
            raise ValueError("canonical corpus inventory has an invalid cell identity")
        config, files = row.get("configuration"), row.get("model_files")
        if (not isinstance(config, dict) or type(config.get("model_parity_schema")) is not int
                or config["model_parity_schema"] != 1 or not isinstance(files, list) or not files
                or any(not isinstance(path, str) or not path for path in files)
                or len(set(files)) != len(files) or config.get("model") not in files):
            raise ValueError("canonical corpus cell omitted its full configuration or model shards")
        generation_profile(config)
        InventoryScope.ALL.accepts(config)
        if config["id"] in identities or row["case"] in cases:
            raise ValueError("canonical corpus inventory contains duplicate cells")
        identities.add(config["id"])
        cases.add(row["case"])
        descriptors = {}
        for path in files:
            stat = model_sources.get(path)
            if (not isinstance(stat, (list, tuple)) or len(stat) != 5
                    or any(type(value) is not int or value < 0 for value in stat) or stat[2] <= 0):
                raise ValueError("canonical corpus model shard lacks its admitted stat identity: " + path)
            descriptors[path] = {"filename": Path(path).name, "bytes": stat[2]}
        if len({item["filename"] for item in descriptors.values()}) != len(files):
            raise ValueError("canonical model shard filenames are ambiguous across mounts")
        declared_files.update(files)
        normalized = copy.deepcopy(row)
        normalized["configuration"]["model"] = descriptors[config["model"]]
        normalized["model_files"] = sorted(descriptors.values(), key=lambda item: item["filename"])
        result.append(normalized)
    if set(model_sources) != declared_files:
        raise ValueError("corpus admission model pins differ from the full declared shard set")
    by_id = {row["configuration"]["id"]: row for row in result}
    for row in by_id.values():
        record = row["configuration"]
        profile = generation_profile(record)
        control_row = by_id.get(profile["serial_control_id"])
        control = control_row["configuration"] if control_row is not None else None
        if (control is None or row["model_files"] != control_row["model_files"]
                or MTPPolicy(generation_profile(control)["mtp_policy"]) is not MTPPolicy.OFF
                or serial_workload_identity(control) != serial_workload_identity(record)):
            raise ValueError("corpus inventory requires each cell's exact canonical serial control")
    return sorted(result, key=lambda row: (row["campaign"], row["case"]))


@dataclass(frozen=True)
class ApprovedGenerationCorpus:
    """Immutable admitted tokens shared by every serial-equivalent MTP policy.

    The complete inventory is checked at admission, including untagged cells.
    Store immutable token tuples rather than retaining caller-owned JSON or old
    runtime receipts that could accidentally stand in for today's live checks.
    """
    pin: ApprovedCorpusPin
    inventory_digest: str
    _cell_controls: tuple[tuple[str, str, str], ...]
    _controls: tuple[tuple[str, tuple[tuple[str, TokenTrace], ...]], ...]

    @classmethod
    def load_reviewed(cls, source_snapshot: Path, corpus_root: Path, cpu_isa: str,
                      inventory: dict, model_sources: dict) -> ApprovedGenerationCorpus:
        """Select approval only from the pipeline's admitted source snapshot.

        The caller owns admission of the immutable source tree and requested
        shipping ISA. The optional data repository can have a different mount
        path, but neither that location nor candidate output chooses the pin.
        Only the selected payload must be materialized; this reader does not
        initialize the submodule or fetch another ISA's corpus.
        """
        if cpu_isa not in ("AVX2", "AVX512"):
            raise ValueError("generation corpus requires an explicit shipping CPU ISA")
        source = source_snapshot.resolve(strict=True)
        catalog_path = (source / APPROVAL_CATALOG).resolve(strict=True)
        if not catalog_path.is_relative_to(source):
            raise ValueError("generation approval catalog escapes the admitted source snapshot")
        catalog = json.loads(catalog_path.read_text())
        if (not isinstance(catalog, dict) or set(catalog) != {"schema", "corpora"}
                or type(catalog["schema"]) is not int or catalog["schema"] != 1
                or not isinstance(catalog["corpora"], dict)
                or not set(catalog["corpora"]).issubset({"AVX2", "AVX512"})):
            raise ValueError("generation approval catalog has an invalid schema or ISA inventory")
        selected = {}
        for isa, entry in catalog["corpora"].items():
            if not isinstance(entry, dict) or set(entry) != {"path", "document_digest"}:
                raise ValueError("generation approval requires a corpus-relative path and reviewed pin")
            path = entry["path"]
            if (not isinstance(path, str) or not path or "\\" in path
                    or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                    or path == "." or PurePosixPath(path).as_posix() != path):
                raise ValueError("generation approval requires a canonical corpus-relative path")
            selected[isa] = (path, ApprovedCorpusPin(isa, entry["document_digest"]))
        if cpu_isa not in selected:
            raise ValueError("no reviewed generation corpus is approved for " + cpu_isa)
        relative, pin = selected[cpu_isa]
        root = corpus_root.resolve(strict=True)
        payload = (root / relative).resolve(strict=True)
        # A relative spelling is insufficient: a payload symlink must not
        # redirect admission into candidate results outside the data mount.
        if not payload.is_relative_to(root):
            raise ValueError("approved generation payload escapes the corpus root")
        return cls.load(payload, pin, inventory, model_sources)

    @classmethod
    def load(cls, path: Path, pin: ApprovedCorpusPin, inventory: dict,
             model_sources: dict) -> ApprovedGenerationCorpus:
        """Read an already materialized payload once; missing/LFS pointers fail."""
        if not isinstance(pin, ApprovedCorpusPin):
            raise TypeError("generation corpus admission requires a typed reviewed pin")
        return cls.admit(json.loads(path.read_text()), pin, inventory, model_sources)

    @classmethod
    def admit(cls, document: dict, pin: ApprovedCorpusPin, inventory: dict,
              model_sources: dict) -> ApprovedGenerationCorpus:
        """Authenticate review, full compatibility and every continuous stream.

        Proof digests reference acquisition evidence covered by explicit review;
        this is deliberately not a validator for arbitrary unreviewed reports.
        The publishing workflow must establish those proofs before installing
        a pin. Ordinary certification receives no approval/write operation.
        """
        if not isinstance(pin, ApprovedCorpusPin):
            raise TypeError("generation corpus admission requires a typed reviewed pin")
        if not isinstance(document, dict) or digest(document) != pin.document_digest:
            raise ValueError("generation corpus differs from the reviewed metadata pin")
        if (type(document.get("schema")) is not int or document["schema"] != 1
                or document.get("kind") != "approved_generation_controls"
                or document.get("cpu_isa") != pin.cpu_isa):
            raise ValueError("generation corpus has the wrong approval schema or CPU ISA")
        normalized = portable_inventory(inventory, model_sources)
        if document.get("inventory") != normalized:
            raise ValueError("generation corpus differs from the full canonical model/configuration inventory")
        provenance = document.get("provenance")
        if not isinstance(provenance, dict) or set(provenance) != {"numerical", "serial", "mtp"}:
            raise ValueError("generation corpus omitted independent acquisition provenance")
        for kind, proofs in provenance.items():
            if (not isinstance(proofs, list) or not proofs or any(not _metadata_digest(value) for value in proofs)
                    or len(set(proofs)) != len(proofs)):
                raise ValueError("generation corpus has incomplete reviewed " + kind + " provenance")
        configs = {row["configuration"]["id"]: row["configuration"] for row in normalized}
        serials = {name: config for name, config in configs.items()
                   if MTPPolicy(generation_profile(config)["mtp_policy"]) is MTPPolicy.OFF}
        controls = document.get("controls")
        if (not isinstance(controls, list) or len(controls) != len(serials)
                or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in controls)
                or {row["id"] for row in controls} != set(serials)):
            raise ValueError("generation corpus must contain exactly the canonical serial controls, never MTP answers")
        admitted = []
        for control in controls:
            config = serials[control["id"]]
            requests = generation_profile(config)["requests"]
            rows = control.get("requests")
            if (not isinstance(rows, list) or len(rows) != len(requests)
                    or any(not isinstance(row, dict) for row in rows)
                    or [row.get("id") for row in rows] != [request["id"] for request in requests]):
                raise ValueError("generation corpus omitted, duplicated or reordered a serial request")
            traces, by_body = [], {}
            for request, row in zip(requests, rows):
                trace = GenerationWorkload.from_record(config).observe(row.get("response"))
                validate_partial_prompt(request, trace, (previous for _, previous in traces))
                body = digest(request["body"])
                if body in by_body and compare_tokens(by_body[body], trace) is not None:
                    raise ValueError("generation corpus repeated request is not token-exact")
                by_body[body] = trace
                traces.append((row["id"], trace))
            admitted.append((control["id"], tuple(traces)))
        # Corpus compatibility ignores mount spelling, but execution must use
        # the exact current record admitted above. Retain its immutable digest
        # so a same-named cell cannot swap topology, sampler or model afterward.
        return cls(pin, digest(normalized),
                   tuple((row["configuration"]["id"], digest(row["configuration"]),
                          generation_profile(row["configuration"])["serial_control_id"])
                         for row in inventory["cells"]),
                   tuple(admitted))

    def expected(self, record: dict) -> dict[str, TokenTrace]:
        """Bind immutable serial tokens to the complete current configuration.

        A cell name is an index, not an admission contract. A consumer cannot
        reuse expected tokens after changing runtime policy or the admitted
        model mount. A different deployment is normalized at full-inventory
        admission, never implicitly during an individual request lookup.
        """
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise TypeError("generation expectations require the complete admitted configuration")
        entry = next((item for item in self._cell_controls if item[0] == record["id"]), None)
        if entry is None:
            raise ValueError("cell is outside the approved full generation inventory: " + record["id"])
        if digest(record) != entry[1]:
            raise ValueError("generation cell differs from its complete admitted configuration")
        return dict(dict(self._controls)[entry[2]])

    def expectations_document(self, record: dict) -> dict:
        """Export exact admitted serial tokens without forging old HTTP evidence."""
        return {"schema": 1, "kind": "generation_expected_tokens", "configuration": copy.deepcopy(record),
                "serial_control_id": generation_profile(record)["serial_control_id"],
                "requests": [{"id": name, "prompt": list(trace.prompt), "completion": list(trace.completion),
                              "finish_reason": trace.finish_reason} for name, trace in self.expected(record).items()]}
