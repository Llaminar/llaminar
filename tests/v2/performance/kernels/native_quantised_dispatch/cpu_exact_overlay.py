"""Compose CPU execution-exact evidence without refitting installed generics.

The ordinary decode/verifier programs remain separate immutable authorities.
This module authenticates those programs and composes only fully named exact
execution keys. A refresh replaces its declared keys, retaining every unrelated
exact entry. New measurements never inherit the retained generic certificate.
Collection, arithmetic admission and winner selection continue to use the common
NativeQuantised adapters and exact oracle; this is their publication boundary.
New batch timings must prove collector suppression; instrumented standalone
history remains diagnostic and cannot acquire new winner authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


CPU_EXACT_SCHEMA = "cpu-execution-exact-v1"


def canonical_digest(value: object) -> str:
    """Hash a deterministic evidence mapping, not any model payload."""
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Stream-hash one explicitly named tooling/evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class RetainedCPUProgram:
    """Frozen installed program identity, independent of new exact evidence."""
    kind: str
    include: Path
    sha256: str
    policy_json: Path | None = None

    def authenticate(self) -> dict[str, object]:
        """Reject stale/foreign bases before any collection or publication.

        An optional original policy artifact authenticates its sealed certificate.
        A hash-only frozen source receipt preserves the already installed program
        but deliberately makes no new generic certification claim.
        """
        if self.kind not in {"decode", "verifier"}:
            raise ValueError("retained CPU program kind must be decode or verifier")
        if file_digest(self.include) != self.sha256:
            raise ValueError(f"retained CPU {self.kind} include identity changed")
        source = self.include.read_text(encoding="utf-8")
        marker = ("LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_POLICY_ABI" if self.kind == "decode"
                  else "LLAMINAR_CPU_NATIVE_QUANTISED_VERIFIER_POLICY_ABI")
        selector = ("selectCPUNativeQuantisedDecodeGeneratedPolicy" if self.kind == "decode"
                    else "selectCPUNativeQuantisedVerifierRowsGeneratedPolicy")
        if source.count("#define " + marker + " ") != 1 or source.count(selector + "(") != 1:
            raise ValueError("retained CPU program has an unsupported generated ABI")
        if "// Frozen generic policy digest: sha256:" not in source:
            raise ValueError("retained CPU program is missing its generic identity")
        if self.policy_json is not None:
            from .policy_artifact import validate_installable_policy_artifact
            validate_installable_policy_artifact(self.policy_json, include_path=self.include)
        return {"kind": self.kind, "include": str(self.include.resolve()),
                "sha256": self.sha256,
                "original_policy_artifact": str(self.policy_json.resolve()) if self.policy_json else None,
                "new_generic_certification": False}


def authenticate_retained_programs(
    programs: Iterable[RetainedCPUProgram],
) -> tuple[dict[str, object], ...]:
    """Require both original CPU programs, without duplicate/shadow identities."""
    indexed: dict[str, RetainedCPUProgram] = {}
    for program in programs:
        if program.kind in indexed:
            raise ValueError("duplicate retained CPU program")
        indexed[program.kind] = program
    if set(indexed) != {"decode", "verifier"}:
        raise ValueError("exact refresh requires retained decode and verifier programs")
    return tuple(indexed[kind].authenticate() for kind in sorted(indexed))


def merge_exact_entries(
    retained: Mapping[tuple[int, ...], int],
    measured: Mapping[tuple[int, ...], int],
    declared_scope: Iterable[tuple[int, ...]],
) -> dict[tuple[int, ...], int]:
    """Replace exactly a complete declared scope; preserve unrelated exact rows.

    The caller owns typed key/candidate validation and authenticated measurements.
    An empty, duplicate, partial or foreign scope must never become an installable
    overlay. Requiring exact equality also prevents quietly deleting an old key
    when its new measurement failed.
    """
    ordered = tuple(declared_scope)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("exact refresh scope must be nonempty and unique")
    scope = set(ordered)
    if set(measured) != scope:
        raise ValueError("measured exact keys do not match the complete declared scope")
    result = dict(retained)
    result.update(measured)
    if any(result[key] != value for key, value in retained.items() if key not in scope):
        raise AssertionError("unrelated exact policy changed during composition")
    return dict(sorted(result.items()))

# The order is the C++ CPUExecutionExactKey ABI, not a packed/hash surrogate.
CPU_EXACT_KEY_FIELDS = (
    "operation", "numerical_policy", "source_codebook", "source_superblock",
    "prepared_encoding", "asymmetric", "build_isa", "runtime_isa", "workers",
    "rows", "physical_n", "policy_n", "k", "k_tiles", "ambient_n_block_chunks",
    "layout", "team_entry", "bundle_class",
)


@dataclass(frozen=True, order=True)
class CPUExecutionExactKey:
    """Full-width runtime branch identity shared with the production C++ key.

    Source codebook is the actual packed source/dispatch authority, preserved
    even when preparation changes encoding. Superblock identity disambiguates
    source tuples sharing a codebook. The filled execution class is certified
    from the whole live descriptor set before any row policy is selected.
    """
    operation: int
    numerical_policy: int
    source_codebook: int
    source_superblock: int
    prepared_encoding: int
    asymmetric: int
    build_isa: int
    runtime_isa: int
    workers: int
    rows: int
    physical_n: int
    policy_n: int
    k: int
    k_tiles: int
    ambient_n_block_chunks: int
    layout: int
    team_entry: int
    bundle_class: int

    def __post_init__(self) -> None:
        """Reject unsupported or lossy execution identities immediately."""
        if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in self.scalars()):
            raise ValueError("CPU execution exact key requires full uint32 scalars")
        if self.operation not in (0, 1, 2) or self.numerical_policy != 1:
            raise ValueError("row exact v1 owns GPUAlignedExpert decode/verifier/prefill only")
        if not 0 <= self.source_codebook <= 255 or self.source_superblock not in (0, 1):
            raise ValueError("CPU exact source tuple is invalid")
        if self.prepared_encoding not in range(6) or self.asymmetric not in (0, 1):
            raise ValueError("CPU exact prepared identity is invalid")
        if self.build_isa not in (0, 1) or self.runtime_isa not in (0, 1) or self.runtime_isa > self.build_isa:
            raise ValueError("CPU exact ISA route is unavailable in its build")
        if min(self.workers, self.physical_n, self.policy_n, self.k, self.ambient_n_block_chunks) <= 0:
            raise ValueError("CPU exact geometry and worker width must be positive")
        if self.rows < 1 or self.k % 32 or self.k_tiles <= 1:
            raise ValueError("CPU exact descriptor requires positive rows and actual K partitions")
        if self.layout != 1 or self.physical_n % 16 or self.bundle_class != 1 or self.team_entry not in (0, 1):
            raise ValueError("row exact v1 requires the admitted aligned filled execution class")

    def scalars(self) -> tuple[int, ...]:
        """Return C++ aggregate field order without truncation or inference."""
        return tuple(getattr(self, name) for name in CPU_EXACT_KEY_FIELDS)

    @classmethod
    def from_scalars(cls, values: Iterable[int]) -> "CPUExecutionExactKey":
        """Read one original emitted key; incomplete keys fail closed."""
        values = tuple(values)
        if len(values) != len(CPU_EXACT_KEY_FIELDS):
            raise ValueError("CPU execution exact key ABI must contain18 scalars")
        return cls(*values)

    def signature(self) -> str:
        """Use existing common bundle identity for exact-only observations."""
        return CPU_EXACT_SCHEMA + ":" + json.dumps(self.scalars(), separators=(",", ":"))


def validate_filled_bundle(
    key: CPUExecutionExactKey,
    descriptors: Iterable[CPUExecutionExactKey],
) -> tuple[CPUExecutionExactKey, ...]:
    """Authenticate policy-invariant OutputTileLocal class from live metadata.

    These are original descriptor observations, never model payloads. Requiring
    every member to be aligned, GPUAligned and K-partitioned prevents a measured
    homogeneous bundle from silently authorizing a differently owned partial
    route. Existing-region/standalone is observed entry context, not inferred
    retained-team lifetime. The lower bound uses the all-Wide physical owners.
    """
    members = tuple(descriptors)
    if not members or key not in members:
        raise ValueError("CPU exact witness must contain its actual selected descriptor")
    shared = (key.operation, key.numerical_policy, key.k, key.build_isa, key.runtime_isa,
              key.workers, key.layout, key.team_entry, key.bundle_class)
    for member in members:
        if (member.operation, member.numerical_policy, member.k, member.build_isa, member.runtime_isa,
            member.workers, member.layout, member.team_entry, member.bundle_class) != shared:
            raise ValueError("CPU exact witness changed execution class inside its bundle")
    owners = sum(((d.rows + 3) // 4) * ((d.physical_n + 63) // 64) for d in members)
    if owners < key.workers:
        raise ValueError("all-Wide output-owner lower bound does not fill the actual team")
    return members


def generate_execution_exact_include(entries: Mapping[CPUExecutionExactKey, int]) -> str:
    """Emit the additive sorted full-key table; installed generics are untouched."""
    if any(key.rows < 2 for key in entries):
        raise ValueError("execution-exact row overlay requires grouped target rows")
    if any(type(choice) is not int or choice not in (0, 1) for choice in entries.values()):
        raise ValueError("execution-exact choice must be physical Pairwise or WideRows")
    ordered = sorted(entries.items())
    lines = [
        "/**", " * @file CPUNativeQuantisedExecutionExactGenerated.inc",
        " * @brief Generated exact CPU execution overlays; no generic policy is fitted.",
        " * Full typed keys preserve source/prepared arithmetic and execution ownership.",
        " * An exact miss leaves the existing certified policy program in authority.",
        " */", "#pragma once", "#include <array>", "#include <cstddef>",
        "#include <cstdint>", "", "namespace llaminar2::cpu::native_quantised::generated", "{",
        "struct CPUExecutionExactGeneratedEntry", "{",
        "    std::array<std::uint32_t, 18> key;", "    CPUExecutionExactChoice choice;", "};",
        f"inline constexpr std::array<CPUExecutionExactGeneratedEntry, {len(ordered)}> kCPUExecutionExactEntries{{{{",
    ]
    for key, choice in ordered:
        values = ", ".join(str(value) + "u" for value in key.scalars())
        name = "Pairwise" if choice == 0 else "WideRows"
        lines.append(f"    {{{{{values}}}, CPUExecutionExactChoice::{name}}},")
    lines += ["}};", "", "/** Resolve one immutable execution key without allocation or policy mutation. */",
              "inline bool selectCPUExecutionExactPolicy(",
              "    const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice) noexcept", "{",
              "    const auto scalars = key.scalars();"]
    lines += [ "    std::size_t first = 0u;", "    std::size_t last = kCPUExecutionExactEntries.size();",
              "    while (first < last)", "    {", "        const std::size_t middle = first + (last - first) / 2u;",
              "        if (kCPUExecutionExactEntries[middle].key < scalars)", "            first = middle + 1u;",
              "        else", "            last = middle;", "    }",
              "    if (first == kCPUExecutionExactEntries.size() || kCPUExecutionExactEntries[first].key != scalars)",
              "        return false;", "    choice = kCPUExecutionExactEntries[first].choice;", "    return true;", "}", "}", ""]
    return "\n".join(lines)



def is_execution_exact_batch_timing(payload: Mapping) -> bool:
    """Identify the existing typed batch producer, never its filename/location.

    Native v2 is batch-only. The shared native v1 schema also has an older
    standalone diagnostic producer; only its batch form carries request_id.
    Neither an absent new marker nor an old path name grants clean clocks.
    """
    return (payload.get("schema") == "cpu_execution_exact_native_v2" or
            (payload.get("schema") == "cpu_execution_exact_native_v1" and
             "request_id" in payload))


def require_execution_exact_batch_timing_collection_disabled(payload: Mapping) -> None:
    """Reject batch clocks unless native collection was disabled and observed zero.

    These additive native counters concern ONLY timed calls. Fresh route
    authentication remains enabled outside timing. Bool/int types are exact so
    strings, truthy integers and bool-as-zero cannot acquire timing authority.
    """
    if not is_execution_exact_batch_timing(payload):
        raise ValueError("CPU exact timing requires original native-batch identity")
    if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
        raise ValueError("CPU exact batch timing requires its original request_id")
    counters = payload.get("counters", {})
    if (not isinstance(counters, Mapping) or
            counters.get("timed_collection_disabled") is not True or
            type(counters.get("timed_projection_observations")) is not int or
            counters["timed_projection_observations"] != 0):
        raise ValueError("CPU exact batch timing requires disabled timed collection and zero timed projection observations")


def require_execution_exact_publication_timing(observation) -> None:
    """Admit new winners only from a known uninstrumented exact timing contract.

    Reopen explicit original-native provenance when present. Historical batch
    rows cannot evade this gate by lacking the new common-witness tag. Ordinary
    nonbatch diagnostic parsing/profiling stays valid, but its counter-bearing
    clocks do not qualify NEW exact winners. Installed literal rows are separate
    retained authority and never pass through this function.
    """
    witness = observation.physical_launch_evidence
    path = witness.get("native_evidence_path")
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if canonical_digest(payload) != witness.get("native_evidence_mapping_digest"):
            raise ValueError("CPU exact original native evidence mapping changed")
        if not is_execution_exact_batch_timing(payload):
            raise ValueError("nonbatch CPU exact timing remains diagnostic, not new winner publication evidence")
        require_execution_exact_batch_timing_collection_disabled(payload)
        if witness.get("native_timing_schema") != payload["schema"]:
            raise ValueError("CPU exact common/native timing schemas disagree")
    if (witness.get("native_timing_owner") != "CPUExecutionExactBatch" or
            witness.get("native_timing_schema") not in
                {"cpu_execution_exact_native_v1", "cpu_execution_exact_native_v2"}):
        raise ValueError("CPU exact publication lacks a known uninstrumented batch timing contract")
    if (witness.get("timed_collection_disabled") is not True or
            type(witness.get("timed_projection_observations")) is not int or
            witness["timed_projection_observations"] != 0):
        raise ValueError("CPU exact publication requires disabled timed collection and zero timed projection observations")


def execution_key_from_observation(observation) -> CPUExecutionExactKey:
    """Recover only an adapter-authenticated exact key, never infer a producer."""
    witness = observation.physical_launch_evidence
    if (observation.operation_kind != "CPUExecutionExactFusedBundle" or
        witness.get("schema") != "cpu-execution-exact-witness-v1" or
        witness.get("timing_basis") != "whole_fused_bundle"):
        raise ValueError("CPU exact row lacks its original execution witness")
    key = CPUExecutionExactKey.from_scalars(witness.get("key", ()))
    if observation.bundle_signature != key.signature() or observation.generic_eligible:
        raise ValueError("CPU exact observation has a foreign fitting/runtime identity")
    validate_filled_bundle(key, (CPUExecutionExactKey.from_scalars(v)
                                for v in witness.get("descriptors", ())))
    return key


def _holdout_domain(key: CPUExecutionExactKey) -> tuple[int, ...]:
    """A held-out mixed vector must exercise the actual proposed target key."""
    return key.scalars()


def _execution_tournament_matrix(rows, required: set[str]):
    """Authenticate a complete adaptive roster without upgrading screening clocks.

    The native adapter already validates actual samples, output/route identity,
    original finalist timing floors and the whole contest. This owner retains
    the same candidate/holdout completeness rule using that typed receipt; only
    final observations can enter the existing performance oracle.
    """
    witnesses = [row.physical_launch_evidence.get("tournament") for row in rows]
    if not any(w is not None for w in witnesses):
        return None
    if not all(w is not None for w in witnesses):
        raise ValueError("CPU exact key mixes original and adaptive contest populations")
    from .adaptive_tournament import validate_adaptive_tournament_witness, is_promotion_grade
    for row, witness in zip(rows, witnesses):
        validate_adaptive_tournament_witness(witness)
        if set(witness["candidate_ids"]) != required:
            raise ValueError("CPU adaptive roster differs from its actual forceable candidate matrix")
        if row.candidate_id not in required:
            raise ValueError("CPU adaptive observation has a foreign candidate")
        if is_promotion_grade(row) != (row.candidate_id in witness["finalist_candidate_ids"]):
            raise ValueError("CPU adaptive finalist grade disagrees with its complete contest")
        if not (row.supported and row.numerical_correctness and row.bitwise_equal and row.repeat_equal and
                row.mismatch_count == 0 and row.forced_route_ok and row.route_counter_ok and
                row.workspace_ok and row.explicit_stream_ok and row.ordered_reduction and
                not row.uses_atomic_reduction):
            raise ValueError("CPU adaptive matrix cannot discard a failed candidate")
    shared = {(w["contest_digest"], tuple(w["candidate_ids"]), w["baseline_candidate_id"],
               w["winner_candidate_id"], tuple(w["finalist_candidate_ids"])) for w in witnesses}
    if len(shared) != 1 or {r.candidate_id for r in rows} != required:
        raise ValueError("CPU adaptive candidate matrix is partial or crosses contest identities")
    return {r.candidate_id for r in rows if is_promotion_grade(r)}


def _execution_promotion_grade(row) -> bool:
    """Keep original fully timed evidence valid; adaptive screens remain diagnostics."""
    if row.physical_launch_evidence.get("tournament") is None:
        return True
    from .adaptive_tournament import is_promotion_grade
    return is_promotion_grade(row)


def select_execution_exact_entries(corpus) -> tuple[dict[CPUExecutionExactKey, int], dict]:
    """Use the common exact oracle, with mandatory separate mixed-bundle proof.

    The scalar key is a leaf policy basis inside a certified filled class, while
    every latency remains whole-bundle. Until the common oracle admits separate
    execution-witness surfaces, one original training vector per key is required;
    a mixed-M holdout is always excluded from winner selection.
    """
    from .exact_oracle import build_exact_winner, candidate_is_eligible
    grouped: dict[CPUExecutionExactKey, list] = {}
    holdouts: dict[tuple[int, ...], list] = {}
    for row in corpus:
        require_execution_exact_publication_timing(row)
        key = execution_key_from_observation(row)
        target = holdouts.setdefault(_holdout_domain(key), []) if row.physical_launch_evidence["holdout"] else grouped.setdefault(key, [])
        target.append(row)
    if not grouped:
        raise ValueError("CPU exact refresh contains no measured overlay keys")
    entries = {}; receipts = []
    for key, rows in sorted(grouped.items()):
        vectors = {canonical_digest(row.physical_launch_evidence["descriptors"]) for row in rows}
        if len(vectors) != 1:
            raise ValueError("distinct training bundle vectors require separate exact-oracle surfaces")
        required = {"cpu.native_quantised.verifier.pairwise"}
        if key.runtime_isa == 1 and key.rows >= 3:
            required.add("cpu.native_quantised.verifier.wide_rows")
        tournament_finalists = _execution_tournament_matrix(rows, required)
        valid = {row.candidate_id for row in rows
                 if candidate_is_eligible(row) and _execution_promotion_grade(row)}
        if valid != (required if tournament_finalists is None else tournament_finalists):
            raise ValueError("CPU exact candidate matrix is incomplete or foreign")
        mixed = [row for row in holdouts.get(_holdout_domain(key), ())
                 if len({v[9] for v in row.physical_launch_evidence["descriptors"]}) > 1]
        mixed_finalists = _execution_tournament_matrix(mixed, required)
        mixed_valid = {row.candidate_id for row in mixed if candidate_is_eligible(row)}
        if mixed_finalists is None and not required.issubset(mixed_valid):
            raise ValueError("CPU exact key lacks complete byte-exact mixed-bundle holdout")
        winner = build_exact_winner([row for row in rows
            if candidate_is_eligible(row) and _execution_promotion_grade(row)])
        choice = {"cpu.native_quantised.verifier.pairwise": 0, "cpu.native_quantised.verifier.wide_rows": 1}[winner.candidate_id]
        entries[key] = choice
        receipts.append({"key": key.scalars(), "choice": choice,
                         "candidate_id": winner.candidate_id,
                         "training_bundle_digests": sorted(vectors),
                         "mixed_holdout_digests": sorted({canonical_digest(row.physical_launch_evidence) for row in mixed}),
                         "timing_basis": "whole_fused_bundle", "latency_divided_per_leaf": False,
                         "max_surface_regret": winner.max_surface_regret})
    return entries, {"schema": CPU_EXACT_SCHEMA, "entries": receipts,
                     "new_generic_certification": False}


def read_execution_exact_include(source: str) -> dict[CPUExecutionExactKey, int]:
    """Read only this emitter's literal table or the reviewed initial empty stub."""
    import re
    table = re.search(r"std::array<CPUExecutionExactGeneratedEntry, (\d+)> kCPUExecutionExactEntries\{\{(.*?)\}\};", source, re.S)
    if table is None:
        stub = re.search(r"inline bool selectCPUExecutionExactPolicy\(\s*const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice\) noexcept\s*\{([^{}]*)\}", source, re.S)
        if stub and re.sub(r"\s", "", stub[1]) == "(void)key;(void)choice;returnfalse;":
            return {}
        raise ValueError("unsupported prior CPU execution-exact generated program")
    rows = {}; body = table[2]
    pattern = re.compile(r"\{\{([\d,u\s]+)\}, CPUExecutionExactChoice::(Pairwise|WideRows)\},")
    ordered = []
    for match in pattern.finditer(body):
        literals = [value.strip() for value in match[1].split(',') if value.strip()]
        if any(not re.fullmatch(r"\d+u", value) for value in literals):
            raise ValueError("CPU execution-exact table contains a non-uint32 literal")
        key = CPUExecutionExactKey.from_scalars(int(value[:-1]) for value in literals)
        if key.rows < 2 or key in rows:
            raise ValueError("CPU execution-exact table has a foreign or duplicate key")
        ordered.append(key); rows[key] = int(match[2] == "WideRows")
    if re.sub(r"\s", "", pattern.sub("", body)) or len(rows) != int(table[1]) or ordered != sorted(ordered):
        raise ValueError("CPU execution-exact literal inventory changed or is incomplete")
    return rows
