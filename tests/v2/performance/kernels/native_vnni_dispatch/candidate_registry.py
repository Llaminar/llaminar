"""Explicit forceable candidate registries for NativeVNNI policy training.

The learned-policy compiler is allowed to choose only candidates represented by
one of these registries.  In particular, ``Auto`` and an already-generated
policy are resolver behavior, not candidates: neither has a stable arithmetic
or launch identity that can be measured and reproduced later.

Candidate products are generated from reviewed constant axes to keep the CUDA
registry readable.  The resulting entries are still explicit and digestible;
changing any axis changes the registry digest and invalidates stale corpora.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

from .schema import Backend, SemanticContract


CANDIDATE_REGISTRY_VERSION = "native-vnni-candidates-v13"
COMPATIBLE_CANDIDATE_REGISTRY_VERSIONS = tuple(
    f"native-vnni-candidates-v{version}" for version in range(1, 14)
)

ROCM_MOE_GROUPED_PREFILL = "rocm_moe_grouped_prefill"
ROCM_NATIVE_VNNI_DECODE = "rocm_native_vnni_decode"
ROCM_NATIVE_VNNI_DECODE_FORMULA = "rocm_native_vnni_decode_formula"
CUDA_NATIVE_VNNI_GEMV = "cuda_native_vnni_gemv"
CPU_NATIVE_VNNI_DECODE = "cpu_native_vnni_decode"
CPU_NATIVE_VNNI_VERIFIER_ROWS = "cpu_native_vnni_verifier_rows"
CPU_NATIVE_VNNI_PREFILL_GEMM = "cpu_native_vnni_prefill_gemm"

# Keep this measured boundary equal to
# kDecodeScheduleUnderfillCrossoverElements in CPUNativeVNNIGemv.h. The
# candidate-registry regression reads both definitions so corpus tooling cannot
# silently diverge from the production resolver.
CPU_DECODE_UNDERFILL_CROSSOVER_ELEMENTS = 32 * 1024 * 1024
CPU_DECODE_N_BLOCK_CHUNKS = (1, 2, 4, 8, 16)


def resolve_cpu_native_vnni_decode_n_block_chunks(
    requested_n_block_chunks: int,
    *,
    n: int,
    k: int,
    k_tiles: int,
    threads: int,
) -> int:
    """Resolve one nominal CPU M=1 schedule to its physical task width.

    This is the corpus-side form of ``resolveDecodeSchedulePolicy()``. Widths
    that cover the complete N inventory collapse to one physical power-of-two
    route. Above the measured 32-Mi-element crossover, a coarse width is then
    narrowed until its ``(N block, K tile)`` producer grid exposes every worker
    that NBC1 could use. The operation is deterministic and total: NBC1 is
    always a physical identity, while small underfilled shapes retain the
    measured task-overhead tradeoff.
    """

    if requested_n_block_chunks not in CPU_DECODE_N_BLOCK_CHUNKS:
        raise ValueError("unsupported CPU decode N-block width")
    if n <= 0 or k <= 0 or k_tiles < 0 or threads <= 0:
        raise ValueError(
            "CPU decode schedule requires positive N, K, and threads plus "
            "non-negative K tiles"
        )

    n_chunks = (n + 63) // 64
    if n_chunks <= 1:
        effective = 1
    elif requested_n_block_chunks < n_chunks:
        effective = requested_n_block_chunks
    elif n_chunks <= 2:
        effective = 2
    elif n_chunks <= 4:
        effective = 4
    elif n_chunks <= 8:
        effective = 8
    else:
        effective = 16

    physical_k_tiles = max(1, k_tiles)
    target_tasks = min(threads, n_chunks * physical_k_tiles)

    def producer_tasks(width: int) -> int:
        return ((n_chunks + width - 1) // width) * physical_k_tiles

    if n * k > CPU_DECODE_UNDERFILL_CROSSOVER_ELEMENTS:
        while producer_tasks(effective) < target_tasks and effective > 1:
            effective //= 2
    return effective


def cpu_native_vnni_decode_physical_candidate_ids(
    *,
    n: int,
    k: int,
    k_tiles: int,
    threads: int,
) -> frozenset[str]:
    """Return every distinct forceable M=1 schedule for one geometry.

    Registry entries describe the union of launch schedules that can exist
    across the complete geometry domain. At one concrete geometry, multiple
    nominal widths can resolve to the same physical OpenMP task grid. Timing or
    requiring those aliases as separate candidates would give one launch
    several identities and distort both coverage and fit weights. Resolve the
    complete registry axis first, then retain exactly one canonical candidate
    ID for each distinct physical width.

    Args:
        n: Physical output-column count.
        k: Physical reduction dimension.
        k_tiles: Frozen serial-M1 K-partition count, where zero means full-K.
        threads: Positive OpenMP worker count for the runtime surface.

    Returns:
        The complete set of physically distinct candidate IDs for this key.
    """

    physical_widths = {
        resolve_cpu_native_vnni_decode_n_block_chunks(
            width,
            n=n,
            k=k,
            k_tiles=k_tiles,
            threads=threads,
        )
        for width in CPU_DECODE_N_BLOCK_CHUNKS
    }
    return frozenset(
        f"cpu.nvnni.decode.n_chunk_grid.nbc{width}"
        for width in physical_widths
    )


def _sha256_json(value: Any) -> str:
    """Return a stable SHA-256 identity for one JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateSpec:
    """One production-forceable launch candidate and its static contract.

    ``candidate_id`` names the requested candidate.  ``effective_candidate_id``
    names the normalized launch after removing nominal settings that do not
    alter execution.  Most current entries are already normalized and use the
    same value for both fields.
    """

    backend: Backend
    registry_surface: str
    candidate_id: str
    effective_candidate_id: str
    aliases: tuple[str, ...]
    candidate_family: str
    config_items: tuple[tuple[str, Any], ...]
    arithmetic_fingerprint: str
    schedule_signature: str
    supported_contracts: tuple[SemanticContract, ...]
    graph_capture_supported: bool
    ordered_reduction: bool
    uses_atomic_reduction: bool
    prepared_resources: tuple[str, ...]
    workspace_signature: str

    @property
    def config_json(self) -> dict[str, Any]:
        """Expose the immutable launcher configuration as a regular mapping."""

        return dict(self.config_items)

    def supports_contract(self, contract: SemanticContract) -> bool:
        """Return whether this exact arithmetic route supports ``contract``."""

        return contract in self.supported_contracts

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a deterministic serializable representation."""

        result = asdict(self)
        result["backend"] = self.backend.value
        result["supported_contracts"] = [
            contract.value for contract in self.supported_contracts
        ]
        result["config_items"] = [list(item) for item in self.config_items]
        return result

    @lru_cache(maxsize=None)
    def candidate_policy_hash(self) -> str:
        """Bind emitted policy evidence to this complete registry entry."""

        return _sha256_json(self.canonical_mapping())


class CandidateRegistry:
    """Validated candidate inventory for one backend launcher surface."""

    def __init__(self, entries: Iterable[CandidateSpec]):
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("candidate registry must not be empty")

        backends = {entry.backend for entry in self._entries}
        surfaces = {entry.registry_surface for entry in self._entries}
        if len(backends) != 1 or len(surfaces) != 1:
            raise ValueError("one candidate registry must describe one backend surface")

        lookup: dict[str, CandidateSpec] = {}
        for entry in self._entries:
            if not entry.candidate_id.strip() or not entry.effective_candidate_id.strip():
                raise ValueError("candidate IDs must not be empty")
            for identifier in (entry.candidate_id, entry.effective_candidate_id, *entry.aliases):
                normalized = identifier.strip().lower()
                if normalized == "auto":
                    raise ValueError("Auto is resolver behavior, not a forceable candidate")
                previous = lookup.get(normalized)
                if previous is not None and previous != entry:
                    raise ValueError(f"candidate alias {identifier!r} is ambiguous")
                lookup[normalized] = entry
            if entry.uses_atomic_reduction and entry.ordered_reduction:
                raise ValueError(
                    f"{entry.candidate_id}: an atomic FP reduction cannot claim ordered reduction"
                )
            if not entry.schedule_signature.strip() or not entry.workspace_signature.strip():
                raise ValueError(f"{entry.candidate_id}: incomplete launch identity")

        self._lookup = lookup

    @property
    def backend(self) -> Backend:
        return self._entries[0].backend

    @property
    def surface(self) -> str:
        return self._entries[0].registry_surface

    @property
    def entries(self) -> tuple[CandidateSpec, ...]:
        return self._entries

    def resolve(self, identifier: str) -> CandidateSpec:
        """Resolve a canonical ID or reviewed trainer alias, failing closed."""

        normalized = identifier.strip().lower()
        if not normalized or normalized == "auto":
            raise ValueError(f"{self.surface}: {identifier!r} is not an explicit candidate")
        try:
            return self._lookup[normalized]
        except KeyError as exc:
            raise ValueError(
                f"{self.surface}: unknown forceable candidate {identifier!r}"
            ) from exc

    def digest(self, *, registry_version: str = CANDIDATE_REGISTRY_VERSION) -> str:
        """Hash this surface's complete ordered inventory.

        ``registry_version`` is explicit for immutable-plan migration. The
        project-wide version historically appeared in every surface digest,
        so adding an unrelated backend surface changed an otherwise identical
        CPU-prefill identity. Readers may recompute a historical version over
        the *current complete entries*; any real change to this surface still
        changes the digest and fails closed.
        """

        payload = {
            "version": registry_version,
            "backend": self.backend.value,
            "surface": self.surface,
            "entries": [entry.canonical_mapping() for entry in self._entries],
        }
        return _sha256_json(payload)

    def matches_digest(self, expected: str) -> bool:
        """Accept an exact current or historical-version surface identity.

        Compatibility covers only the old project-wide version stamp. The
        backend, surface, ordered candidate inventory, and every candidate
        field are always recomputed from current code.
        """

        return any(
            self.digest(registry_version=version) == expected
            for version in COMPATIBLE_CANDIDATE_REGISTRY_VERSIONS
        )


def _candidate(
    *,
    backend: Backend,
    surface: str,
    candidate_id: str,
    aliases: tuple[str, ...],
    family: str,
    config: dict[str, Any],
    arithmetic: str,
    schedule: str,
    contracts: tuple[SemanticContract, ...],
    graph_capture: bool = True,
    ordered: bool = True,
    atomics: bool = False,
    resources: tuple[str, ...] = ("native_vnni_prepared_weights",),
    workspace: str = "declared_graph_workspace-v1",
) -> CandidateSpec:
    """Build one normalized entry while keeping registry declarations compact."""

    return CandidateSpec(
        backend=backend,
        registry_surface=surface,
        candidate_id=candidate_id,
        effective_candidate_id=candidate_id,
        aliases=aliases,
        candidate_family=family,
        config_items=tuple(sorted(config.items())),
        arithmetic_fingerprint=_sha256_json({"arithmetic": arithmetic}),
        schedule_signature=schedule,
        supported_contracts=contracts,
        graph_capture_supported=graph_capture,
        ordered_reduction=ordered,
        uses_atomic_reduction=atomics,
        prepared_resources=resources,
        workspace_signature=workspace,
    )


@lru_cache(maxsize=1)
def rocm_moe_grouped_prefill_registry() -> CandidateRegistry:
    """Return all 12 production HIP row-tile/column-tile combinations."""

    entries = []
    for tile_m in (4, 8, 12, 16):
        for tile_n in (64, 128, 256):
            raw_id = f"tm{tile_m}_tn{tile_n}"
            entries.append(_candidate(
                backend=Backend.ROCM,
                surface=ROCM_MOE_GROUPED_PREFILL,
                candidate_id=f"rocm.moe.grouped_prefill.tm{tile_m}.tn{tile_n}",
                aliases=(raw_id,),
                family="rocm_moe_decode_equivalent_row_tile",
                config={"tile_m": tile_m, "tile_n": tile_n},
                arithmetic=(
                    "ordered increasing-K NativeVNNI decode, FP32 scaling, "
                    "serial route-order publication, no floating-point atomics-v1"
                ),
                schedule=f"hip-gfx906-row-reuse-tm{tile_m}-tn{tile_n}-v1",
                contracts=(
                    SemanticContract.FAST,
                    SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                ),
                resources=(
                    "native_vnni_expert_tables",
                    "device_group_directory",
                    "device_grouped_activation_scratch",
                ),
                workspace="MoEWorkspaceBuffers.rocmMoE-v1",
            ))
    return CandidateRegistry(entries)


@lru_cache(maxsize=1)
def rocm_native_vnni_decode_registry() -> CandidateRegistry:
    """Return normalized ROCm serial-decode and grouped-verifier candidates.

    An explicit KB changes the M=1 split/reduction schedule and is therefore a
    real Fast candidate.  Once KB is explicit, target-wave settings no longer
    change the launch; reviewed historical KB/TW spellings are aliases of the
    same effective KB candidate.  Grouped verifier rows have one independent
    schedule choice: inherit the exact frozen serial-M1 split policy.  Giving
    verifier rows a different KB would change FP32 reduction order and cannot
    satisfy the bitwise contract.
    """

    entries = []
    for kb in range(1, 65):
        legacy_aliases = tuple(
            alias
            for target_waves in (4, 8, 12, 24, 40)
            for alias in (
                f"KB{kb}/TW{target_waves}",
                f"kb{kb}tw{target_waves}",
            )
        )
        entries.append(_candidate(
            backend=Backend.ROCM,
            surface=ROCM_NATIVE_VNNI_DECODE,
            candidate_id=f"rocm.nvnni.decode.fast.kb{kb}",
            aliases=(f"KB{kb}", f"kb{kb}", *legacy_aliases),
            family="rocm_native_vnni_serial_kpart",
            config={"kb": kb},
            arithmetic="ordered K-part partials and serial-order FP32 reduction-v1",
            schedule=f"hip-native-vnni-serial-kb{kb}-v2",
            contracts=(SemanticContract.FAST,),
            resources=("native_vnni_prepared_weights", "ordered_kpart_partials"),
            workspace="ROCmNativeVNNIKPartWorkspace-v1",
        ))

    entries.append(_candidate(
        backend=Backend.ROCM,
        surface=ROCM_NATIVE_VNNI_DECODE,
        candidate_id="rocm.nvnni.decode.verifier.inherit_serial_m1",
        aliases=("INHERIT_SERIAL_M1", "inherit_serial_m1", "serial_m1"),
        family="rocm_native_vnni_grouped_rows",
        config={"split_policy": "inherit_serial_m1"},
        arithmetic=(
            "grouped row-reuse decode with the frozen serial-M1 K partition, "
            "ordered per-row FP32 partial reduction-v2"
        ),
        schedule="hip-native-vnni-grouped-rows-inherit-serial-m1-v2",
        contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
        resources=("native_vnni_prepared_weights", "ordered_kpart_partials"),
        workspace="ROCmNativeVNNIKPartWorkspace-v1",
    ))
    return CandidateRegistry(entries)


@lru_cache(maxsize=1)
def rocm_native_vnni_decode_formula_registry() -> CandidateRegistry:
    """Return total generic ROCm policies backed by concrete KB evidence.

    The production launcher interprets a generated policy KB as a requested
    partition count and clamps it to ``K / 32`` before launch. A literal KB row
    is therefore an exact-shape candidate, while this registry names the
    geometry-dependent operation that generic dispatch actually performs.
    Projection resolves every formula to a directly measured concrete row; no
    timing or correctness evidence is synthesized.
    """

    return CandidateRegistry(
        _candidate(
            backend=Backend.ROCM,
            surface=ROCM_NATIVE_VNNI_DECODE_FORMULA,
            candidate_id=(
                "rocm.nvnni.decode.fast.clamped_formula."
                f"kb{requested_kb}"
            ),
            aliases=(),
            family="rocm_native_vnni_serial_clamped_kpart_formula",
            config={
                "family": "clamped_kb_formula",
                "formula_kind": "requested_kb_clamped_to_k_groups",
                "kb": requested_kb,
                "k_group_width": 32,
            },
            arithmetic=(
                "ordered K-part partials with requested KB clamped to K/32 "
                "and serial-order FP32 reduction-v1"
            ),
            schedule=(
                "hip-native-vnni-serial-requested-kb-"
                f"{requested_kb}-shape-clamped-v1"
            ),
            contracts=(SemanticContract.FAST,),
            resources=(
                "native_vnni_prepared_weights",
                "ordered_kpart_partials",
            ),
            workspace="ROCmNativeVNNIKPartWorkspace-v1",
        )
        for requested_kb in range(1, 65)
    )


@lru_cache(maxsize=1)
def cpu_native_vnni_decode_registry() -> CandidateRegistry:
    """Return forceable CPU M=1 task-ownership schedules.

    CPU batch invariance freezes the production serial-M1 arithmetic partition:
    a geometry is either full-K or uses the existing ordered K-part boundaries.
    Learned decode dispatch may change how many adjacent 64-column chunks one
    OpenMP task owns, but it may not change those K boundaries or the ascending
    FP32 reduction tree.  The same five candidates therefore apply to both
    arithmetic bundles without making a different numerical promise.
    """

    common = {
        "backend": Backend.CPU,
        "surface": CPU_NATIVE_VNNI_DECODE,
        "family": "cpu_native_vnni_decode_n_chunk_grid",
        "arithmetic": (
            "frozen serial-M1 full-K or ordered K-part accumulation; "
            "schedule cannot change K partition boundaries-v1"
        ),
        "contracts": (SemanticContract.FAST,),
        "graph_capture": False,
        "resources": (
            "native_vnni_prepared_weights",
            "prequantized_q8_1_row",
            "ordered_kpart_partials",
        ),
        "workspace": "cpu-native-vnni-m1-persistent-partials-v1",
    }
    return CandidateRegistry(
        _candidate(
            **common,
            candidate_id=f"cpu.nvnni.decode.n_chunk_grid.nbc{n_block_chunks}",
            aliases=(f"NBC{n_block_chunks}",),
            config={
                "route": "n_chunk_grid",
                "n_block_chunks": n_block_chunks,
                "k_partition_policy": "frozen_serial_m1",
            },
            schedule=(
                "cpu-native-vnni-m1-n-chunk-grid-"
                f"nbc{n_block_chunks}-underfill-resolved-v2"
            ),
        )
        for n_block_chunks in CPU_DECODE_N_BLOCK_CHUNKS
    )


@lru_cache(maxsize=1)
def cpu_native_vnni_verifier_registry() -> CandidateRegistry:
    """Return every economical, serial-row-exact CPU verifier schedule.

    Pairwise and WideRows preserve the production serial-M1 K partition and
    therefore remain available for long-K domains.  The full-K families reuse
    packed-weight decode across rows when serial M1 owns one K tile.  Their
    distinct task grids and N-block widths are real runtime identities: the
    all-format admission sweep found winners in every family retained here.
    """

    common = {
        "backend": Backend.CPU,
        "surface": CPU_NATIVE_VNNI_VERIFIER_ROWS,
        "arithmetic": "serial-M1-shaped ordered dot-product per row-v1",
        "contracts": (SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
        "graph_capture": False,
        "resources": (
            "native_vnni_prepared_weights",
            "prequantized_q8_1_rows",
        ),
        "workspace": "cpu-thread-local-accumulators-v1",
    }
    entries = [
        _candidate(
            **common,
            candidate_id="cpu.nvnni.verifier.pairwise",
            aliases=("Pairwise", "verifier_rows_pairwise_policy"),
            family="cpu_pairwise_rows",
            config={
                "policy": "Pairwise",
                "route": "decode_equivalent_kpart_rows",
                "row_tile": 2,
                "k_tile_policy": "inherit_serial_m1",
            },
            schedule="cpu-avx512-vnni-pairwise-workshare-v1",
        ),
        _candidate(
            **common,
            candidate_id="cpu.nvnni.verifier.wide_rows",
            aliases=("WideRows", "verifier_rows_wide_policy"),
            family="cpu_wide_rows",
            config={
                "policy": "WideRows",
                "route": "decode_equivalent_kpart_rows",
                "row_tile": 4,
                "k_tile_policy": "inherit_serial_m1",
            },
            schedule="cpu-avx512-vnni-wide-row-workshare-v1",
        ),
        _candidate(
            **common,
            candidate_id="cpu.nvnni.verifier.full_k.row_chunk_grid",
            aliases=("FullKRowChunkGrid",),
            family="cpu_native_vnni_grouped_full_k_row_chunk_grid",
            config={
                "policy": "FullKRowChunkGrid",
                "route": "row_chunk_grid",
                "row_tile": 1,
                "k_tile_policy": "full_k",
            },
            schedule="cpu-native-vnni-grouped-row-chunk-grid-full-k-v1",
        ),
    ]
    for n_block_chunks in (1, 2):
        entries.append(_candidate(
            **common,
            candidate_id=(
                "cpu.nvnni.verifier.full_k.two_row_n_major."
                f"nbc{n_block_chunks}"
            ),
            aliases=(f"FullKTwoRowNbc{n_block_chunks}",),
            family="cpu_native_vnni_grouped_full_k_two_row_n_major",
            config={
                "policy": f"FullKTwoRowNbc{n_block_chunks}",
                "route": "two_row_n_major",
                "row_tile": 2,
                "n_block_chunks": n_block_chunks,
                "k_tile_policy": "full_k",
            },
            schedule=(
                "cpu-native-vnni-grouped-two-row-n-major-"
                f"nbc{n_block_chunks}-full-k-v1"
            ),
        ))
    for n_block_chunks in (1, 2, 4, 8):
        entries.append(_candidate(
            **common,
            candidate_id=(
                "cpu.nvnni.verifier.full_k.two_row_pair_grid."
                f"nbc{n_block_chunks}"
            ),
            aliases=(f"FullKTwoRowPairGridNbc{n_block_chunks}",),
            family="cpu_native_vnni_grouped_full_k_two_row_pair_grid",
            config={
                "policy": f"FullKTwoRowPairGridNbc{n_block_chunks}",
                "route": "two_row_pair_grid",
                "row_tile": 2,
                "n_block_chunks": n_block_chunks,
                "k_tile_policy": "full_k",
            },
            schedule=(
                "cpu-native-vnni-grouped-two-row-pair-grid-"
                f"nbc{n_block_chunks}-full-k-v1"
            ),
        ))
    return CandidateRegistry(entries)


@lru_cache(maxsize=1)
def cpu_native_vnni_prefill_registry() -> CandidateRegistry:
    """Return byte-proven physical CPU prefill schedules.

    The row-chunk grid ignores N blocking after route selection, so it is one
    candidate. Two distinct two-row families retain N blocking as a physical
    scheduling axis. N-major tasks own one N block and visit every row pair;
    pair-grid tasks own one `(row pair, N block)` tile and expose M parallelism
    to OpenMP. Both invoke the same byte-exact two-row microkernel. Long-K
    shapes use explicit Pairwise/Wide grouped K-part candidates whose
    independent row partials are reduced in the exact serial-M1 tile order.
    """

    common = {
        "backend": Backend.CPU,
        "surface": CPU_NATIVE_VNNI_PREFILL_GEMM,
        "contracts": (
            SemanticContract.FAST,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        ),
        "graph_capture": False,
        "resources": (
            "native_vnni_prepared_weights",
            "prequantized_q8_1_rows",
        ),
        "workspace": "cpu-thread-local-prefill-accumulators-v2",
    }
    entries = [_candidate(
        **common,
        candidate_id="cpu.nvnni.prefill.row_chunk_grid.full_k",
        aliases=(),
        family="cpu_native_vnni_row_chunk_grid",
        config={
            "route": "row_chunk_grid",
            "n_block_policy": "minimum_for_row_chunk_grid",
            "k_tile_policy": "full_k",
        },
        arithmetic="independent ordered serial-M1 K accumulation per row-v1",
        schedule="cpu-native-vnni-row-chunk-grid-full-k-v1",
    )]
    # NBC16 is intentionally absent from the N-major family. Across 1,827
    # historical cells it never won, and the focused short-prefill audit found
    # its best result still 36.5% behind the cell winner. Keep NBC16 in the
    # pair-grid family below because that physically different schedule has
    # demonstrated rare wins.
    for n_block_chunks in (1, 2, 4, 8):
        entries.append(_candidate(
            **common,
            candidate_id=(
                "cpu.nvnni.prefill.two_row_tiles."
                f"nbc{n_block_chunks}.full_k"
            ),
            aliases=(),
            family="cpu_native_vnni_two_row_output_tiles",
            config={
                "route": "two_row_full_output_tiles",
                "n_block_chunks": n_block_chunks,
                "k_tile_policy": "full_k",
            },
            arithmetic="independent ordered serial-M1 K accumulation per row-v1",
            schedule=(
                "cpu-native-vnni-two-row-output-tiles-"
                f"nbc{n_block_chunks}-full-k-v1"
            ),
        ))
    for n_block_chunks in (1, 2, 4, 8, 16):
        entries.append(_candidate(
            **common,
            candidate_id=(
                "cpu.nvnni.prefill.two_row_pair_grid."
                f"nbc{n_block_chunks}.full_k"
            ),
            aliases=(),
            family="cpu_native_vnni_two_row_pair_grid",
            config={
                "route": "two_row_pair_grid",
                "n_block_chunks": n_block_chunks,
                "row_tile": 2,
                "k_tile_policy": "full_k",
            },
            arithmetic="independent ordered serial-M1 K accumulation per row-v1",
            schedule=(
                "cpu-native-vnni-two-row-pair-grid-"
                f"nbc{n_block_chunks}-full-k-v1"
            ),
        ))
    for policy, family, row_tile in (
        ("Pairwise", "cpu_native_vnni_kpart_pairwise", 2),
        ("WideRows", "cpu_native_vnni_kpart_wide_rows", 4),
    ):
        entries.append(_candidate(
            **common,
            candidate_id=(
                "cpu.nvnni.prefill.decode_equivalent_kpart."
                + ("pairwise" if policy == "Pairwise" else "wide_rows")
            ),
            aliases=(),
            family=family,
            config={
                "route": "decode_equivalent_kpart_rows",
                "policy": policy,
                "row_tile": row_tile,
                "k_tile_policy": "inherit_serial_m1",
            },
            arithmetic=(
                "independent serial-M1 K partials with ordered exact reduction-v1"
            ),
            schedule=(
                "cpu-native-vnni-decode-equivalent-kpart-"
                + ("pairwise-v1" if policy == "Pairwise" else "wide-rows-v1")
            ),
        ))
    return CandidateRegistry(entries)


@lru_cache(maxsize=1)
def cuda_native_vnni_gemv_registry() -> CandidateRegistry:
    """Return effective public-M1 and grouped-verifier CUDA candidates.

    Public CUDA M1 decode uses ordered KPAR publication, but no longer
    collapses every candidate to KB1. An exact KB is part of the arithmetic
    identity because it fixes partition boundaries and FP32 parenthesization.
    The forceable inventory is dense through KB256 so existing and newly learned
    production schedules always resolve to an explicit arithmetic identity.
    Sweep profiles may use a coarse subset followed by local refinement, but the
    emitted winner must be one of these exact candidates. A winner at KB256 is a
    boundary signal and must expand this axis before policy promotion.

    Fused KPAR preserves that exact arithmetic with CTA-local partials and one
    output writer. Its two physical widths have different native thread limits;
    those limits constrain admission, never silently clamp the selected KB.

    Grouped runtime-M verifier publication is a separate production candidate. It
    inherits the frozen public-M1 family and tuning for the exact runtime key;
    it is not an independently retuned KPAR launch.  ROWPAR is intentionally
    absent until immutable row-major preparation is owned by both public M1 and
    grouped verifier execution.  Floating-point atomic publication is never a
    candidate on this serial-M1-bitwise surface.
    """

    entries: list[CandidateSpec] = []
    for family, tile_pairs in (
        ("wide", ((128, 1), (128, 2), (256, 2), (256, 4), (512, 4))),
        ("direct", ((128, 1), (64, 1), (32, 1))),
    ):
        for tile_n, cpt in tile_pairs:
            candidate_id = f"cuda.nvnni.decode.fast_m1.{family}.tn{tile_n}.cpt{cpt}"
            entries.append(_candidate(
                backend=Backend.CUDA,
                surface=CUDA_NATIVE_VNNI_GEMV,
                candidate_id=candidate_id,
                aliases=(),
                family=f"cuda_public_m1_{family}",
                config={
                    "family": family,
                    "tile_n": tile_n,
                    "cpt": cpt,
                    "target_waves": 0,
                    "min_kgroups_per_cta": 0,
                    "max_kb": 0,
                    "exact_kb": 0,
                    "force_two_phase": 0,
                },
                arithmetic="public-M1 ordered NativeVNNI dot-product-v2",
                schedule=f"cuda-public-m1-{family}-tn{tile_n}-cpt{cpt}-v2",
                contracts=(SemanticContract.FAST,),
                workspace="CUDANativeVNNIGemvWorkspace-v1",
            ))

    reviewed_kblocks = range(1, 257)
    for tile_n, cpt in (
        (128, 1), (128, 2), (256, 2), (256, 4),
        (64, 1), (64, 2), (32, 1),
    ):
        for kb in reviewed_kblocks:
            entries.append(_candidate(
                backend=Backend.CUDA,
                surface=CUDA_NATIVE_VNNI_GEMV,
                candidate_id=(
                    f"cuda.nvnni.decode.fast_m1.kpar."
                    f"tn{tile_n}.cpt{cpt}.kb{kb}"
                ),
                aliases=(),
                family="cuda_public_m1_kpar",
                config={
                    "family": "kpar",
                    "tile_n": tile_n,
                    "cpt": cpt,
                    "target_waves": 0,
                    "min_kgroups_per_cta": 0,
                    "max_kb": 0,
                    "exact_kb": kb,
                    "force_two_phase": 1,
                },
                arithmetic=(
                    "public-M1 disjoint K-partition outputs and ascending FP32 "
                    f"reduction, KB={kb}-v3"
                ),
                schedule=(
                    f"cuda-public-m1-kpar-tn{tile_n}-cpt{cpt}-kb{kb}-v3"
                ),
                contracts=(SemanticContract.FAST,),
                resources=("native_vnni_prepared_weights", "ordered_kpart_partials"),
                workspace="CUDANativeVNNIKPartWorkspace-v2",
            ))

    for columns in (16, 32):
        for kb in range(1, 1024 // columns + 1):
            entries.append(_candidate(
                backend=Backend.CUDA,
                surface=CUDA_NATIVE_VNNI_GEMV,
                candidate_id=(
                    f"cuda.nvnni.decode.fast_m1.fused_kpar.tn{columns}.cpt1.kb{kb}"
                ),
                aliases=(),
                family="cuda_public_m1_fused_kpar",
                config={
                    "family": "fused_kpar",
                    "tile_n": columns,
                    "cpt": 1,
                    "target_waves": 0,
                    "min_kgroups_per_cta": 0,
                    "max_kb": 0,
                    "exact_kb": kb,
                    # Historical ABI spelling authenticates ordered partials;
                    # the physical family distinguishes shared/global storage.
                    "force_two_phase": 1,
                },
                arithmetic=(
                    "public-M1 disjoint K-partition outputs and ascending FP32 "
                    f"reduction, KB={kb}-v3"
                ),
                schedule=f"cuda-public-m1-fused-kpar-tn{columns}-kb{kb}-v1",
                contracts=(SemanticContract.FAST,),
                resources=("native_vnni_prepared_weights",),
                workspace="CUDANativeVNNIGemvWorkspace-v1",
            ))

    # Generic policy candidates resolve a concrete exact KB from N and K. The
    # selected formula is deterministic and M-independent; grouped verifier
    # publication therefore receives the same concrete partition tree as
    # serial M1. Their costs are projected only from directly measured exact-KB
    # rows by cuda_shape_resolved.py.
    formula_geometries = (
        *(("kpar", n, cpt, 256) for n, cpt in (
            (128, 1), (128, 2), (256, 2), (256, 4),
            (64, 1), (64, 2), (32, 1),
        )),
        *(("fused_kpar", n, 1, 1024 // n) for n in (16, 32)),
    )
    for physical_family, tile_n, cpt, maximum_kb in formula_geometries:
        # Only the nominal formula varies with geometry. Its resolved launch
        # remains one immutable, directly measured exact-KB registry entry.
        formula_family = f"{physical_family}_formula"
        schedule_family = physical_family.replace("_", "-")
        resources = ("native_vnni_prepared_weights",)
        workspace = "CUDANativeVNNIGemvWorkspace-v1"
        if physical_family == "kpar":
            resources += ("ordered_kpart_partials",)
            workspace = "CUDANativeVNNIKPartWorkspace-v2"
        for target_waves in range(1, 41):
            for min_kgroups_per_cta in (1, 2, 4, 8):
                target_blocks = 82 * target_waves
                candidate_id = (
                    f"cuda.nvnni.decode.fast_m1.{formula_family}."
                    f"tn{tile_n}.cpt{cpt}.tb{target_blocks}."
                    f"mkg{min_kgroups_per_cta}"
                )
                entries.append(_candidate(
                    backend=Backend.CUDA,
                    surface=CUDA_NATIVE_VNNI_GEMV,
                    candidate_id=candidate_id,
                    aliases=(),
                    family=f"cuda_public_m1_{formula_family}",
                    config={
                        "family": formula_family,
                        "formula_kind": "target_blocks",
                        "tile_n": tile_n,
                        "cpt": cpt,
                        "target_blocks": target_blocks,
                        "min_kgroups_per_cta": min_kgroups_per_cta,
                        "max_kb": maximum_kb,
                        "force_two_phase": 1,
                    },
                    arithmetic=(
                        "N/K-resolved disjoint K partitions with ascending FP32 "
                        "reduction; absolute target-block nearest-factor formula-v1"
                    ),
                    schedule=(
                        f"cuda-public-m1-{schedule_family}-formula-tn{tile_n}-cpt{cpt}-"
                        f"tb{target_blocks}-mkg{min_kgroups_per_cta}-v1"
                    ),
                    contracts=(SemanticContract.FAST,),
                    resources=resources,
                    workspace=workspace,
                ))
                canonical_id = (
                    f"cuda.nvnni.decode.fast_m1.{formula_family}."
                    f"tn{tile_n}.cpt{cpt}.ctb{target_blocks}."
                    f"mkg{min_kgroups_per_cta}"
                )
                entries.append(_candidate(
                    backend=Backend.CUDA,
                    surface=CUDA_NATIVE_VNNI_GEMV,
                    candidate_id=canonical_id,
                    aliases=(),
                    family=f"cuda_public_m1_{formula_family}",
                    config={
                        "family": formula_family,
                        "formula_kind": "canonical_target_blocks",
                        "tile_n": tile_n,
                        "cpt": cpt,
                        "target_blocks": target_blocks,
                        "min_kgroups_per_cta": min_kgroups_per_cta,
                        "max_kb": maximum_kb,
                        "force_two_phase": 1,
                    },
                    arithmetic=(
                        "N/K-resolved disjoint K partitions with ascending FP32 "
                        "reduction; absolute target-block canonical uneven "
                        "partition-width formula-v1"
                    ),
                    schedule=(
                        f"cuda-public-m1-{schedule_family}-formula-tn{tile_n}-cpt{cpt}-"
                        f"ctb{target_blocks}-mkg{min_kgroups_per_cta}-v1"
                    ),
                    contracts=(SemanticContract.FAST,),
                    resources=resources,
                    workspace=workspace,
                ))
        for blocks_per_partition in range(1, 65):
            candidate_id = (
                f"cuda.nvnni.decode.fast_m1.{formula_family}."
                f"tn{tile_n}.cpt{cpt}.bpp{blocks_per_partition}"
            )
            entries.append(_candidate(
                backend=Backend.CUDA,
                surface=CUDA_NATIVE_VNNI_GEMV,
                candidate_id=candidate_id,
                aliases=(),
                family=f"cuda_public_m1_{formula_family}",
                config={
                    "family": formula_family,
                    "formula_kind": "blocks_per_partition",
                    "tile_n": tile_n,
                    "cpt": cpt,
                    "blocks_per_partition": blocks_per_partition,
                    "max_kb": maximum_kb,
                    "force_two_phase": 1,
                },
                arithmetic=(
                    "N/K-resolved disjoint K partitions with ascending FP32 "
                    "reduction; fixed K-groups-per-partition formula-v1"
                ),
                schedule=(
                    f"cuda-public-m1-{schedule_family}-formula-tn{tile_n}-cpt{cpt}-"
                    f"bpp{blocks_per_partition}-v1"
                ),
                contracts=(SemanticContract.FAST,),
                resources=resources,
                workspace=workspace,
            ))

    for grouped_rows in (2, 4, 8, 16, 32, 64):
        entries.append(_candidate(
            backend=Backend.CUDA,
            surface=CUDA_NATIVE_VNNI_GEMV,
            candidate_id=(
                "cuda.nvnni.decode.verifier.inherit_serial_m1."
                f"r{grouped_rows}"
            ),
            aliases=(),
            family="cuda_native_vnni_grouped_rows",
            config={
                "family": "inherit_serial_m1",
                "grouped_rows": grouped_rows,
            },
            arithmetic=(
                "grouped row-dimension decode with frozen public-M1 family, "
                "tile, exact K partition, and ascending per-row FP32 "
                "publication-v4"
            ),
            schedule=(
                "cuda-native-vnni-grouped-rows-inherit-serial-m1-"
                f"r{grouped_rows}-v4"
            ),
            contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
            resources=(
                "native_vnni_prepared_weights",
                "ordered_kpart_partials",
            ),
            workspace="CUDANativeVNNIKPartWorkspace-v2",
        ))
    entries.append(_candidate(
        backend=Backend.CUDA,
        surface=CUDA_NATIVE_VNNI_GEMV,
        candidate_id="cuda.nvnni.decode.verifier.tensor_core_mma16",
        aliases=(),
        family="cuda_native_vnni_grouped_tensor_core",
        config={"family": "tensor_core_mma16", "warps_per_block": 4},
        arithmetic=(
            "integer m16n8k32 MMA dot products with production NativeVNNI "
            "all-format correction, private per-warp tiles, and FP32 "
            "publication"
        ),
        schedule="cuda-native-vnni-grouped-tensor-core-mma16-w4-v2",
        contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
        resources=(
            "native_vnni_prepared_weights",
            "ordered_kpart_partials",
        ),
        workspace="CUDANativeVNNIKPartWorkspace-v2",
    ))
    return CandidateRegistry(entries)


@lru_cache(maxsize=1)
def candidate_registry_digest() -> str:
    """Hash every initial backend registry as one transaction input."""

    registries = (
        cpu_native_vnni_decode_registry(),
        cpu_native_vnni_verifier_registry(),
        cpu_native_vnni_prefill_registry(),
        cuda_native_vnni_gemv_registry(),
        rocm_native_vnni_decode_registry(),
        rocm_native_vnni_decode_formula_registry(),
        rocm_moe_grouped_prefill_registry(),
    )
    return _sha256_json({
        "version": CANDIDATE_REGISTRY_VERSION,
        "registries": [registry.digest() for registry in registries],
    })
