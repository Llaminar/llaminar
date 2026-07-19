"""Add newly implemented CPU prefill candidates to an immutable corpus.

Kernel tuning is iterative: a completed corpus can reveal a missing physical
schedule, and implementing that schedule must not force every unchanged
candidate to be timed again.  This module defines the fail-closed transaction
for that case.

An expansion plan inventories every measurement cell in the source aggregate,
records the candidates already present, and selects only registry candidates
that are absent. Collection times those new candidates together with one
existing anchor for each physical serial route. Route-complete anchors are
required because absolute timings from runs collected hours or days apart are
not comparable at a five-percent regret gate. During adaptation, each
supported new candidate's timing distribution is multiplied by
``source_anchor_median / expansion_anchor_median`` for the exact route, CPU ISA,
source-format, shape, and M surface. Raw CSVs and timing sidecars are never
rewritten; the derived observation records the scale and both anchor medians in
its adaptive-timing provenance.

Unsupported candidates are retained as explicit matrix evidence without a
scale.  They cannot participate in economy selection, but their authenticated
route failure proves that the registry candidate was considered at that cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_registry import cpu_native_vnni_prefill_registry
from .corpus import ObservationCorpus, RuntimeKey, SurfaceKey
from .cpu_prefill_split_manifest import load_cpu_prefill_split_manifest
from .profiles import (
    CANDIDATE_EXPANSION_EVIDENCE_KEY,
    CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
)
from .schema import NativeVNNIObservation


CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA = (
    "cpu-prefill-candidate-expansion-plan-v4"
)
DEFAULT_ANCHOR_CANDIDATES = (
    "cpu.nvnni.prefill.row_chunk_grid.full_k",
    "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
)

_REQUIRED_SOURCE_COLUMNS = frozenset({
    "backend",
    "phase",
    "source_format",
    "source_codebook",
    "execution_codebook",
    "shape",
    "execution_mode",
    "m",
    "n",
    "k",
    "candidate_id",
    "build_isa",
    "runtime_isa_requested",
    "runtime_isa_effective",
    "threads",
})

_ISA_REGIME_BY_PAIR = {
    ("AVX2", "AVX2"): "avx2-build.avx2-runtime",
    ("AVX512", "AVX2"): "avx512-build.avx2-runtime",
    ("AVX512", "AVX512"): "avx512-build.avx512-runtime",
}

_IMPLEMENTATION_PATHS = (
    Path("src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h"),
    Path(
        "tests/v2/performance/kernels/cpu/native_vnni/"
        "Perf__CPUNativeVNNI_GEMV.cpp"
    ),
    Path("tests/v2/utils/NativeVNNITrainerEvidence.h"),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/"
        "candidate_registry.py"
    ),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/"
        "cpu_prefill_candidate_expansion.py"
    ),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/"
        "adapters/cpu_prefill.py"
    ),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/profiles.py"
    ),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/"
        "validate_cpu_prefill_partial.py"
    ),
    Path(
        "tests/v2/performance/kernels/native_vnni_dispatch/manifests/"
        "native_vnni_decode_shapes_v5.json"
    ),
    Path("scripts/refresh_native_vnni_dispatch_tables.sh"),
)


def _sha256_file(path: Path) -> str:
    """Return a streaming content identity for one retained evidence file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_mapping(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def cpu_prefill_candidate_expansion_plan_token(path: Path) -> str:
    """Return the short on-disk plan identity used in partial filenames.

    The complete plan contains its own canonical digest, but the refresh
    driver historically names partials with the first sixteen hexadecimal
    characters of the serialized plan file's SHA-256.  Checkpoint migration
    must reproduce that exact naming rule; deriving it here gives the Python
    transaction and the shell launcher one authoritative interpretation.
    """

    return _sha256_file(Path(path)).removeprefix("sha256:")[:16]


def _implementation_digest() -> str:
    """Bind resumable shards to every byte that defines their meaning."""

    repository_root = Path(__file__).resolve().parents[5]
    identities = {
        str(relative): _sha256_file(repository_root / relative)
        for relative in _IMPLEMENTATION_PATHS
    }
    return _sha256_mapping(identities)


@dataclass(frozen=True, order=True)
class CPUPrefillCandidateExpansionRecord:
    """One process launch after all M values for a packed tensor are grouped."""

    source_format: str
    shape_name: str
    n: int
    k: int
    isa_regime: str
    runtime_isa: str
    threads: int
    m_values: tuple[int, ...]


@dataclass(frozen=True)
class CPUPrefillCandidateExpansionPlan:
    """Content-addressed inventory for one additive candidate transaction."""

    schema_version: str
    source_aggregate_sha256: str
    source_timing_sha256: str
    candidate_registry_digest: str
    implementation_digest: str
    collection_build_digest: str
    source_cell_count: int
    source_cell_digest: str
    selection_policy: str
    selected_cell_count: int
    selected_cell_digest: str
    source_candidate_ids: tuple[str, ...]
    anchor_candidate_ids: tuple[str, ...]
    expansion_candidate_ids: tuple[str, ...]
    records: tuple[CPUPrefillCandidateExpansionRecord, ...]

    @property
    def collection_candidate_ids(self) -> tuple[str, ...]:
        """Return the anchor followed by every newly implemented candidate."""

        return (*self.anchor_candidate_ids, *self.expansion_candidate_ids)

    def canonical_mapping(self) -> dict[str, object]:
        """Return the complete deterministic representation used for hashing."""

        return {
            "schema_version": self.schema_version,
            "source_aggregate_sha256": self.source_aggregate_sha256,
            "source_timing_sha256": self.source_timing_sha256,
            "candidate_registry_digest": self.candidate_registry_digest,
            "implementation_digest": self.implementation_digest,
            "collection_build_digest": self.collection_build_digest,
            "source_cell_count": self.source_cell_count,
            "source_cell_digest": self.source_cell_digest,
            "selection_policy": self.selection_policy,
            "selected_cell_count": self.selected_cell_count,
            "selected_cell_digest": self.selected_cell_digest,
            "source_candidate_ids": list(self.source_candidate_ids),
            "anchor_candidate_ids": list(self.anchor_candidate_ids),
            "expansion_candidate_ids": list(self.expansion_candidate_ids),
            "records": [
                {
                    **asdict(record),
                    "m_values": list(record.m_values),
                }
                for record in self.records
            ],
        }

    def digest(self) -> str:
        """Return the immutable identity used by resumable partial filenames."""

        return _sha256_mapping(self.canonical_mapping())


def _raw_cell_mapping(raw: Mapping[str, str]) -> dict[str, object]:
    """Normalize every discriminator that can select a different CPU launch."""

    return {
        "source_format": raw["source_format"].strip().upper(),
        "source_codebook": int(raw["source_codebook"]),
        "execution_codebook": int(raw["execution_codebook"]),
        "shape": raw["shape"].strip(),
        "execution_mode": raw["execution_mode"].strip().lower(),
        "m": int(raw["m"]),
        "n": int(raw["n"]),
        "k": int(raw["k"]),
        "build_isa": raw["build_isa"].strip().upper(),
        "runtime_isa_requested": raw["runtime_isa_requested"].strip().upper(),
        "runtime_isa_effective": raw["runtime_isa_effective"].strip().upper(),
        "threads": int(raw["threads"]),
    }


def _cell_identity(mapping: Mapping[str, object]) -> str:
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"))


def build_cpu_prefill_candidate_expansion_plan(
    source_aggregate: Path,
    source_timing: Path,
    *,
    anchor_candidate_ids: Iterable[str] = DEFAULT_ANCHOR_CANDIDATES,
    expansion_candidate_ids: Iterable[str] | None = None,
    development_only: bool = False,
    collection_build_digest: str,
) -> CPUPrefillCandidateExpansionPlan:
    """Derive a missing-candidate inventory from an immutable corpus.

    A complete refresh can retain historical exploratory cells in its immutable
    source without making them part of every later fit. ``development_only``
    selects the reviewed development split: every production overlay plus the
    systematic geometry witnesses reserved for generic held-out fitting. Every
    selected cell receives the same registry candidate inventory, preventing CV
    from learning labels produced by incomparable candidate universes.
    """

    source_aggregate = Path(source_aggregate)
    source_timing = Path(source_timing)
    registry = cpu_native_vnni_prefill_registry()
    if (
        not collection_build_digest.startswith("sha256:")
        or len(collection_build_digest) != 71
        or any(
            character not in "0123456789abcdef"
            for character in collection_build_digest.removeprefix("sha256:")
        )
    ):
        raise ValueError("CPU prefill expansion build digest is invalid")
    anchors = tuple(
        registry.resolve(candidate_id).candidate_id
        for candidate_id in anchor_candidate_ids
    )
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("CPU prefill expansion anchors must be unique")

    candidates_by_cell: dict[str, set[str]] = defaultdict(set)
    cells: dict[str, dict[str, object]] = {}
    with source_aggregate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_SOURCE_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "CPU prefill expansion source is missing columns "
                f"{sorted(missing)}"
            )
        for row_number, raw in enumerate(reader, start=2):
            if raw["backend"].strip().lower() != "cpu" or (
                raw["phase"].strip() != "prefill_gemm"
            ):
                raise ValueError(
                    f"{source_aggregate}:{row_number}: wrong policy surface"
                )
            cell = _raw_cell_mapping(raw)
            if cell["execution_mode"] != "eager":
                raise ValueError(
                    f"{source_aggregate}:{row_number}: CPU prefill must be eager"
                )
            if min(cell["m"], cell["n"], cell["k"], cell["threads"]) <= 0:
                raise ValueError(
                    f"{source_aggregate}:{row_number}: invalid launch dimensions"
                )
            identity = _cell_identity(cell)
            candidate = registry.resolve(raw["candidate_id"]).candidate_id
            if candidate in candidates_by_cell[identity]:
                raise ValueError(
                    f"{source_aggregate}:{row_number}: duplicate candidate/cell"
                )
            candidates_by_cell[identity].add(candidate)
            cells.setdefault(identity, cell)

    if not cells:
        raise ValueError("CPU prefill expansion source contains no cells")
    source_candidate_sets = {
        tuple(sorted(candidate_ids))
        for candidate_ids in candidates_by_cell.values()
    }
    if len(source_candidate_sets) != 1:
        raise ValueError(
            "CPU prefill expansion source does not have one complete candidate "
            "inventory at every measurement cell"
        )
    source_candidates = next(iter(source_candidate_sets))
    missing_anchors = sorted(set(anchors).difference(source_candidates))
    if missing_anchors:
        raise ValueError(
            "CPU prefill expansion anchors are absent from the source: "
            f"{missing_anchors}"
        )

    if expansion_candidate_ids is None:
        expansion_candidates = tuple(
            entry.candidate_id
            for entry in registry.entries
            if entry.candidate_id not in source_candidates
        )
    else:
        expansion_candidates = tuple(
            registry.resolve(candidate_id).candidate_id
            for candidate_id in expansion_candidate_ids
        )
    if not expansion_candidates:
        raise ValueError("CPU prefill corpus already covers every registry candidate")
    if len(set(expansion_candidates)) != len(expansion_candidates):
        raise ValueError("CPU prefill expansion candidates must be unique")
    overlap = set(expansion_candidates).intersection(source_candidates)
    if overlap:
        raise ValueError(
            "CPU prefill expansion may not recollect existing candidates: "
            f"{sorted(overlap)}"
        )

    selected_cells = cells
    selection_policy = "all-immutable-source-cells-v1"
    if development_only:
        development_names = frozenset(
            load_cpu_prefill_split_manifest().development_shapes
        )
        selected_cells = {
            identity: cell
            for identity, cell in cells.items()
            if str(cell["shape"]) in development_names
        }
        selection_policy = "split-development-cells-v1"
    if not selected_cells:
        raise ValueError(
            "CPU prefill candidate-expansion selection contains no cells"
        )

    grouped_m: dict[tuple[object, ...], set[int]] = defaultdict(set)
    for cell in selected_cells.values():
        build_runtime = (
            str(cell["build_isa"]), str(cell["runtime_isa_effective"])
        )
        try:
            isa_regime = _ISA_REGIME_BY_PAIR[build_runtime]
        except KeyError as exc:
            raise ValueError(
                f"unsupported CPU prefill ISA pair {build_runtime}"
            ) from exc
        group = (
            str(cell["source_format"]),
            str(cell["shape"]),
            int(cell["n"]),
            int(cell["k"]),
            isa_regime,
            str(cell["runtime_isa_effective"]).lower(),
            int(cell["threads"]),
        )
        grouped_m[group].add(int(cell["m"]))

    records = tuple(sorted(
        (
            CPUPrefillCandidateExpansionRecord(
                source_format=str(group[0]),
                shape_name=str(group[1]),
                n=int(group[2]),
                k=int(group[3]),
                isa_regime=str(group[4]),
                runtime_isa=str(group[5]),
                threads=int(group[6]),
                m_values=tuple(sorted(m_values)),
            )
            for group, m_values in grouped_m.items()
        ),
        key=lambda record: (
            record.m_values,
            -(record.n * record.k * max(record.m_values)),
            record.shape_name,
            record.source_format,
            record.isa_regime,
        ),
    ))
    ordered_cells = [cells[identity] for identity in sorted(cells)]
    ordered_selected_cells = [
        selected_cells[identity] for identity in sorted(selected_cells)
    ]
    return CPUPrefillCandidateExpansionPlan(
        schema_version=CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA,
        source_aggregate_sha256=_sha256_file(source_aggregate),
        source_timing_sha256=_sha256_file(source_timing),
        candidate_registry_digest=registry.digest(),
        implementation_digest=_implementation_digest(),
        collection_build_digest=collection_build_digest,
        source_cell_count=len(cells),
        source_cell_digest=_sha256_mapping(ordered_cells),
        selection_policy=selection_policy,
        selected_cell_count=len(selected_cells),
        selected_cell_digest=_sha256_mapping(ordered_selected_cells),
        source_candidate_ids=source_candidates,
        anchor_candidate_ids=anchors,
        expansion_candidate_ids=expansion_candidates,
        records=records,
    )


def write_cpu_prefill_candidate_expansion_plan(
    path: Path,
    plan: CPUPrefillCandidateExpansionPlan,
) -> None:
    """Atomically publish one plan together with its self-authenticating digest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **plan.canonical_mapping(),
        "plan_digest": plan.digest(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def read_cpu_prefill_candidate_expansion_plan(
    path: Path,
    *,
    source_aggregate: Path | None = None,
    source_timing: Path | None = None,
    collection_build_digest: str | None = None,
    authenticate_current_implementation: bool = True,
) -> CPUPrefillCandidateExpansionPlan:
    """Read and authenticate one immutable candidate-expansion plan.

    Normal collection and resume must retain the default implementation check:
    a partial is reusable only while every byte defining collection semantics
    is unchanged. The checkpoint-rebase transaction and a non-installable fit
    replay from a previously normalized checkpoint may disable that one check.
    Both still authenticate the plan's self-digest, current candidate registry,
    source identities, selected-cell inventory, and trainer-binary identity;
    rebase also independently revalidates every raw shard under current policy.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        CPUPrefillCandidateExpansionRecord(
            source_format=str(record["source_format"]),
            shape_name=str(record["shape_name"]),
            n=int(record["n"]),
            k=int(record["k"]),
            isa_regime=str(record["isa_regime"]),
            runtime_isa=str(record["runtime_isa"]),
            threads=int(record["threads"]),
            m_values=tuple(int(value) for value in record["m_values"]),
        )
        for record in raw["records"]
    )
    plan = CPUPrefillCandidateExpansionPlan(
        schema_version=str(raw["schema_version"]),
        source_aggregate_sha256=str(raw["source_aggregate_sha256"]),
        source_timing_sha256=str(raw["source_timing_sha256"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        implementation_digest=str(raw["implementation_digest"]),
        collection_build_digest=str(raw["collection_build_digest"]),
        source_cell_count=int(raw["source_cell_count"]),
        source_cell_digest=str(raw["source_cell_digest"]),
        selection_policy=str(raw["selection_policy"]),
        selected_cell_count=int(raw["selected_cell_count"]),
        selected_cell_digest=str(raw["selected_cell_digest"]),
        source_candidate_ids=tuple(str(item) for item in raw["source_candidate_ids"]),
        anchor_candidate_ids=tuple(
            str(item) for item in raw["anchor_candidate_ids"]
        ),
        expansion_candidate_ids=tuple(
            str(item) for item in raw["expansion_candidate_ids"]
        ),
        records=records,
    )
    if plan.schema_version != CPU_PREFILL_CANDIDATE_EXPANSION_SCHEMA:
        raise ValueError("unsupported CPU prefill candidate-expansion schema")
    if raw.get("plan_digest") != plan.digest():
        raise ValueError("CPU prefill candidate-expansion plan digest changed")
    if not cpu_native_vnni_prefill_registry().matches_digest(
        plan.candidate_registry_digest
    ):
        raise ValueError("CPU prefill candidate registry changed after planning")
    if authenticate_current_implementation and (
        plan.implementation_digest != _implementation_digest()
    ):
        raise ValueError(
            "CPU prefill candidate-expansion implementation changed after planning"
        )
    if collection_build_digest is not None and (
        plan.collection_build_digest != collection_build_digest
    ):
        raise ValueError(
            "CPU prefill candidate-expansion trainer binaries changed after planning"
        )
    if source_aggregate is not None and (
        _sha256_file(Path(source_aggregate)) != plan.source_aggregate_sha256
    ):
        raise ValueError("CPU prefill expansion source aggregate changed")
    if source_timing is not None and (
        _sha256_file(Path(source_timing)) != plan.source_timing_sha256
    ):
        raise ValueError("CPU prefill expansion source timing changed")
    if not plan.records or not plan.expansion_candidate_ids:
        raise ValueError("CPU prefill candidate-expansion plan is empty")
    if (
        not plan.anchor_candidate_ids
        or len(set(plan.anchor_candidate_ids)) != len(plan.anchor_candidate_ids)
        or not set(plan.anchor_candidate_ids).issubset(plan.source_candidate_ids)
    ):
        raise ValueError(
            "CPU prefill candidate-expansion anchors are incomplete"
        )
    if plan.selected_cell_count <= 0 or (
        sum(len(record.m_values) for record in plan.records)
        != plan.selected_cell_count
    ):
        raise ValueError(
            "CPU prefill candidate-expansion selected-cell count disagrees"
        )
    return plan


def _normalization_key(
    corpus: ObservationCorpus,
    row: NativeVNNIObservation,
) -> tuple[RuntimeKey, SurfaceKey, str]:
    """Select the exact runtime and source-alias surface sharing one clock."""

    return (
        corpus.runtime_key_for(row),
        SurfaceKey(row.source_format, row.execution_mode),
        row.shape_group_id,
    )


def normalize_cpu_prefill_candidate_expansion(
    source: ObservationCorpus,
    expansion: ObservationCorpus,
    plan: CPUPrefillCandidateExpansionPlan,
) -> ObservationCorpus:
    """Scale new timings through contemporaneous anchors and combine corpora.

    The returned corpus contains every source observation and every expansion
    target observation, but not the duplicate expansion anchor.  Supported
    targets must have a supported anchor on both sides.  This makes an omitted
    or failed anchor a hard evidence error instead of silently accepting an
    absolute cross-session comparison.
    """

    anchors = frozenset(plan.anchor_candidate_ids)
    targets = frozenset(plan.expansion_candidate_ids)
    # The plan is a large, content-addressed measurement inventory.  Its
    # digest is identical for every normalized row, so serializing and hashing
    # the complete plan inside the target loop turns a linear normalization
    # pass into tens of gigabytes of redundant Python work on a production
    # corpus.  Compute the immutable identity once and reuse it in every row's
    # provenance record.
    plan_digest = plan.digest()
    source_anchor: dict[
        tuple[RuntimeKey, SurfaceKey, str],
        dict[str, list[NativeVNNIObservation]],
    ] = defaultdict(lambda: defaultdict(list))
    expansion_anchor: dict[
        tuple[RuntimeKey, SurfaceKey, str],
        dict[str, list[NativeVNNIObservation]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in source:
        if row.candidate_id in anchors and row.supported and row.forced_route_ok:
            source_anchor[_normalization_key(source, row)][
                row.candidate_id
            ].append(row)
    for row in expansion:
        if row.candidate_id in anchors and row.supported and row.forced_route_ok:
            expansion_anchor[_normalization_key(expansion, row)][
                row.candidate_id
            ].append(row)

    unexpected = sorted({
        row.candidate_id
        for row in expansion
        if row.candidate_id not in anchors and row.candidate_id not in targets
    })
    if unexpected:
        raise ValueError(
            f"CPU prefill expansion contains undeclared candidates {unexpected}"
        )

    # All five pair-grid candidates in one runtime cell share the same source
    # and contemporaneous anchor clocks.  Resolve that relationship once per
    # cell rather than sorting anchors and recomputing the same medians once
    # per candidate.  dict.fromkeys preserves first-observation order, keeping
    # fail-closed diagnostics deterministic if more than one cell is invalid.
    supported_target_keys = dict.fromkeys(
        _normalization_key(expansion, row)
        for row in expansion
        if row.candidate_id in targets
        and row.supported
        and row.forced_route_ok
    )
    normalization_by_key: dict[
        tuple[RuntimeKey, SurfaceKey, str],
        tuple[str, float, float, float],
    ] = {}
    for key in supported_target_keys:
        matching_anchors = sorted(
            candidate_id
            for candidate_id in anchors
            if source_anchor.get(key, {}).get(candidate_id)
            and expansion_anchor.get(key, {}).get(candidate_id)
        )
        if len(matching_anchors) != 1:
            raise ValueError(
                "supported CPU prefill expansion candidate requires exactly "
                "one route-matched source/contemporaneous anchor: "
                f"{key}, anchors={matching_anchors}"
            )
        anchor = matching_anchors[0]
        source_median = statistics.median(
            item.median_us for item in source_anchor[key][anchor]
        )
        expansion_median = statistics.median(
            item.median_us for item in expansion_anchor[key][anchor]
        )
        scale = source_median / expansion_median
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(f"CPU prefill expansion scale is invalid for {key}")
        normalization_by_key[key] = (
            anchor,
            source_median,
            expansion_median,
            scale,
        )

    normalized: list[NativeVNNIObservation] = []
    for row in expansion:
        if row.candidate_id in anchors:
            continue
        if row.candidate_id not in targets:
            continue
        if not row.supported or not row.forced_route_ok:
            evidence = {
                **row.adaptive_timing_evidence,
                CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                    "schema_version": CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
                    "plan_digest": plan_digest,
                    "status": "candidate_not_supported_on_serial_route",
                },
            }
            normalized.append(replace(row, adaptive_timing_evidence=evidence))
            continue
        key = _normalization_key(expansion, row)
        anchor, source_median, expansion_median, scale = normalization_by_key[key]
        evidence = {
            **row.adaptive_timing_evidence,
            CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                "schema_version": CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
                "plan_digest": plan_digest,
                "status": "scaled_to_immutable_source_anchor",
                "anchor_candidate_id": anchor,
                "source_anchor_median_us_hex": source_median.hex(),
                "expansion_anchor_median_us_hex": expansion_median.hex(),
                "scale_hex": scale.hex(),
            },
        }
        normalized.append(replace(
            row,
            min_us=row.min_us * scale,
            median_us=row.median_us * scale,
            p95_us=row.p95_us * scale,
            mad_us=row.mad_us * scale,
            effective_bandwidth_gbs=row.effective_bandwidth_gbs / scale,
            adaptive_timing_evidence=evidence,
        ))

    expected_target_rows = sum(
        1 for row in expansion if row.candidate_id in targets
    )
    if len(normalized) != expected_target_rows:
        raise ValueError("CPU prefill expansion lost target observations")
    return ObservationCorpus((
        *source.observations,
        *normalized,
    ))


def validate_normalized_cpu_prefill_candidate_expansion(
    corpus: ObservationCorpus,
    plan: CPUPrefillCandidateExpansionPlan,
) -> None:
    """Authenticate a compact normalized corpus for sidecar-free refitting.

    Raw timing sidecars remain the evidence authority during adaptation, but
    repeatedly parsing millions of unchanged samples before every tree-search
    experiment is unnecessary.  The canonical observation CSV may serve as a
    refit checkpoint only when it proves the complete plan matrix: every
    selected cell has one row for every source and expansion candidate, no
    undeclared candidate exists, and each expansion row carries the exact
    normalization-plan identity and disposition.
    """

    candidates = (*plan.source_candidate_ids, *plan.expansion_candidate_ids)
    expected_candidates = frozenset(candidates)
    counts = Counter(row.candidate_id for row in corpus)
    unexpected = sorted(set(counts) - expected_candidates)
    if unexpected:
        raise ValueError(
            "normalized CPU prefill candidate expansion contains undeclared "
            f"candidates {unexpected}"
        )
    expected_count = plan.selected_cell_count
    incomplete = {
        candidate_id: counts.get(candidate_id, 0)
        for candidate_id in candidates
        if counts.get(candidate_id, 0) != expected_count
    }
    if incomplete:
        raise ValueError(
            "normalized CPU prefill candidate expansion is not a complete "
            f"plan matrix: expected={expected_count}, actual={incomplete}"
        )
    expected_rows = expected_count * len(candidates)
    if len(corpus) != expected_rows:
        raise ValueError(
            "normalized CPU prefill candidate expansion row count changed: "
            f"expected={expected_rows}, actual={len(corpus)}"
        )

    plan_digest = plan.digest()
    targets = frozenset(plan.expansion_candidate_ids)
    for row in corpus:
        if row.candidate_id not in targets:
            continue
        proof = row.adaptive_timing_evidence.get(
            CANDIDATE_EXPANSION_EVIDENCE_KEY
        )
        if not isinstance(proof, Mapping):
            raise ValueError(
                "normalized CPU prefill expansion row lacks plan provenance"
            )
        if proof.get("schema_version") != CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA:
            raise ValueError(
                "normalized CPU prefill expansion row uses another provenance schema"
            )
        if proof.get("plan_digest") != plan_digest:
            raise ValueError(
                "normalized CPU prefill expansion row belongs to another plan"
            )
        expected_status = (
            "scaled_to_immutable_source_anchor"
            if row.supported and row.forced_route_ok
            else "candidate_not_supported_on_serial_route"
        )
        if proof.get("status") != expected_status:
            raise ValueError(
                "normalized CPU prefill expansion disposition disagrees with "
                f"candidate support: expected={expected_status!r}"
            )


def _parse_candidates(raw: str) -> tuple[str, ...] | None:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or None


def main() -> int:
    """Build, inspect, or authenticate one candidate-expansion plan."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-aggregate", type=Path)
    parser.add_argument("--source-timing", type=Path)
    parser.add_argument("--collection-build-digest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--anchor-candidates",
        default=",".join(DEFAULT_ANCHOR_CANDIDATES),
    )
    parser.add_argument("--candidates", default="")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Collect the reviewed development shapes, including every production "
            "overlay and every systematic generic-fit geometry witness"
        ),
    )
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--candidate-list", action="store_true")
    parser.add_argument("--thread-count", action="store_true")
    args = parser.parse_args()

    if args.output:
        if (
            args.plan
            or not args.source_aggregate
            or not args.source_timing
            or not args.collection_build_digest
        ):
            parser.error(
                "plan creation requires source aggregate, source timing, and "
                "collection build digest with --output without --plan"
            )
        plan = build_cpu_prefill_candidate_expansion_plan(
            args.source_aggregate,
            args.source_timing,
            anchor_candidate_ids=_parse_candidates(args.anchor_candidates) or (),
            expansion_candidate_ids=_parse_candidates(args.candidates),
            development_only=args.development_only,
            collection_build_digest=args.collection_build_digest,
        )
        write_cpu_prefill_candidate_expansion_plan(args.output, plan)
        print(
            f"planned {plan.selected_cell_count}/{plan.source_cell_count} cells, "
            f"{len(plan.records)} launches, "
            f"and {len(plan.expansion_candidate_ids)} new candidates -> {args.output}"
        )
        return 0

    if not args.plan:
        parser.error("inspection requires --plan")
    plan = read_cpu_prefill_candidate_expansion_plan(
        args.plan,
        source_aggregate=args.source_aggregate,
        source_timing=args.source_timing,
        collection_build_digest=args.collection_build_digest,
    )
    selected_modes = sum((args.records, args.candidate_list, args.thread_count))
    if selected_modes != 1:
        parser.error("select exactly one of --records, --candidate-list, --thread-count")
    if args.candidate_list:
        print(",".join(plan.collection_candidate_ids))
    elif args.thread_count:
        thread_counts = {record.threads for record in plan.records}
        if len(thread_counts) != 1:
            raise ValueError(
                "CPU prefill expansion spans multiple thread-count policy regimes"
            )
        print(next(iter(thread_counts)))
    else:
        for record in plan.records:
            print("\t".join((
                record.source_format,
                record.shape_name,
                str(record.n),
                str(record.k),
                record.isa_regime,
                record.runtime_isa,
                ",".join(str(value) for value in record.m_values),
            )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
