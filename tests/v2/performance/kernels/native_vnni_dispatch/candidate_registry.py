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
from typing import Any, Iterable

from .schema import Backend, SemanticContract


CANDIDATE_REGISTRY_VERSION = "native-vnni-candidates-v4"

ROCM_MOE_GROUPED_PREFILL = "rocm_moe_grouped_prefill"
ROCM_NATIVE_VNNI_DECODE = "rocm_native_vnni_decode"
CUDA_NATIVE_VNNI_GEMV = "cuda_native_vnni_gemv"
CPU_NATIVE_VNNI_VERIFIER_ROWS = "cpu_native_vnni_verifier_rows"


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

    def digest(self) -> str:
        """Hash the complete ordered inventory for corpus provenance."""

        payload = {
            "version": CANDIDATE_REGISTRY_VERSION,
            "backend": self.backend.value,
            "surface": self.surface,
            "entries": [entry.canonical_mapping() for entry in self._entries],
        }
        return _sha256_json(payload)


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


def cpu_native_vnni_verifier_registry() -> CandidateRegistry:
    """Return the two currently forceable CPU grouped verifier schedules."""

    return CandidateRegistry((
        _candidate(
            backend=Backend.CPU,
            surface=CPU_NATIVE_VNNI_VERIFIER_ROWS,
            candidate_id="cpu.nvnni.verifier.pairwise",
            aliases=("Pairwise", "verifier_rows_pairwise_policy"),
            family="cpu_pairwise_rows",
            config={"policy": "Pairwise"},
            arithmetic="serial-M1-shaped ordered dot-product per row-v1",
            schedule="cpu-avx512-vnni-pairwise-workshare-v1",
            contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
            graph_capture=False,
            workspace="cpu-thread-local-accumulators-v1",
        ),
        _candidate(
            backend=Backend.CPU,
            surface=CPU_NATIVE_VNNI_VERIFIER_ROWS,
            candidate_id="cpu.nvnni.verifier.wide_rows",
            aliases=("WideRows", "verifier_rows_wide_policy"),
            family="cpu_wide_rows",
            config={"policy": "WideRows"},
            arithmetic="serial-M1-shaped ordered dot-product per row-v1",
            schedule="cpu-avx512-vnni-wide-row-workshare-v1",
            contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
            graph_capture=False,
            workspace="cpu-thread-local-accumulators-v1",
        ),
    ))


def cuda_native_vnni_gemv_registry() -> CandidateRegistry:
    """Return effective public-M1 and grouped-verifier CUDA candidates.

    Public CUDA M1 decode uses ordered two-phase KPAR publication, but no longer
    collapses every candidate to KB1. An exact KB is part of the arithmetic
    identity because it fixes partition boundaries and FP32 parenthesization.
    The forceable inventory is dense through KB256 so existing and newly learned
    production schedules always resolve to an explicit arithmetic identity.
    Sweep profiles may use a coarse subset followed by local refinement, but the
    emitted winner must be one of these exact candidates. A winner at KB256 is a
    boundary signal and must expand this axis before policy promotion.

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

    entries.append(_candidate(
        backend=Backend.CUDA,
        surface=CUDA_NATIVE_VNNI_GEMV,
        candidate_id="cuda.nvnni.decode.verifier.inherit_serial_m1",
        aliases=("INHERIT_SERIAL_M1", "inherit_serial_m1", "serial_m1"),
        family="cuda_native_vnni_grouped_rows",
        config={"family": "inherit_serial_m1"},
        arithmetic=(
            "grouped row-dimension decode with frozen public-M1 family, tile, "
            "exact K partition, and ascending per-row FP32 publication-v3"
        ),
        schedule="cuda-native-vnni-grouped-rows-inherit-serial-m1-v3",
        contracts=(SemanticContract.VERIFIER_SERIAL_M1_BITWISE,),
        resources=("native_vnni_prepared_weights", "ordered_kpart_partials"),
        workspace="CUDANativeVNNIKPartWorkspace-v2",
    ))
    return CandidateRegistry(entries)


def candidate_registry_digest() -> str:
    """Hash every initial backend registry as one transaction input."""

    registries = (
        cpu_native_vnni_verifier_registry(),
        cuda_native_vnni_gemv_registry(),
        rocm_native_vnni_decode_registry(),
        rocm_moe_grouped_prefill_registry(),
    )
    return _sha256_json({
        "version": CANDIDATE_REGISTRY_VERSION,
        "registries": [registry.digest() for registry in registries],
    })
