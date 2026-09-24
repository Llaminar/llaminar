"""Validated production MoE codebook mixtures derived from pinned GGUFs.

Runtime dispatch must remain model-agnostic. The source GGUF inventory records
model names and layer numbers only as provenance; this module collapses those
uses onto the actual execution key: projection geometry plus the routed and
shared gate/up/down source formats. Identical Qwen 3.5 and Qwen 3.6 mixtures
therefore share one exact-overlay measurement instead of teaching dispatch to
recognize checkpoint names.

The loader is intentionally strict. Every layer in every checked-in variant
must belong to exactly one pattern, every LFS object must have immutable SHA256
provenance, and NativeVNNI eligibility is recomputed from the canonical format
registry. F16, BF16, MXFP4, or future formats remain visible all-format cases;
they are never mislabeled as NativeVNNI or silently discarded.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from .format_registry import FORMAT_BY_LABEL, format_spec
from .hf_gguf_moe_inventory import (
    MANIFEST_SCHEMA_VERSION,
    PINNED_QWEN_MOE_REPOSITORIES,
)
from .qwen_release_geometry import QWEN_RELEASE_MODELS, QwenModelKind


QWEN_MOE_GGUF_PATTERN_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "qwen_moe_gguf_codebook_patterns_v1.json"
)


@dataclass(frozen=True, order=True)
class MoEProjectionFormats:
    """Source formats for one gate/up/down expert projection family."""

    gate: str
    up: str
    down: str

    @property
    def all(self) -> tuple[str, str, str]:
        """Return formats in stable gate, up, down order."""

        return self.gate, self.up, self.down

    @property
    def native_vnni_sweepable(self) -> bool:
        """Return whether every role has a NativeVNNI execution path."""

        return all(label in FORMAT_BY_LABEL for label in self.all)

    def runtime_codebooks(self, backend: str) -> tuple[int, int, int]:
        """Resolve backend-visible codebooks, failing for non-NativeVNNI roles."""

        if not self.native_vnni_sweepable:
            unsupported = sorted(set(self.all) - set(FORMAT_BY_LABEL))
            raise ValueError(
                f"{backend} NativeVNNI cannot dispatch formats {unsupported}"
            )
        return tuple(
            format_spec(label).runtime_codebook(backend) for label in self.all
        )


@dataclass(frozen=True, order=True)
class QwenMoEPatternUse:
    """One checkpoint/layer set represented by a model-agnostic mixture."""

    release_id: str
    variant_id: str
    layers: tuple[int, ...]


@dataclass(frozen=True, order=True)
class QwenMoEPrefillMixture:
    """One unique production MoE prefill pipeline geometry and format tuple."""

    hidden_size: int
    routed_expert_width: int
    shared_expert_width: int
    expert_count: int
    experts_per_token: int
    routed: MoEProjectionFormats
    shared: MoEProjectionFormats
    uses: tuple[QwenMoEPatternUse, ...]

    @property
    def native_vnni_sweepable(self) -> bool:
        """Return whether all six production projections use NativeVNNI."""

        return (
            self.routed.native_vnni_sweepable
            and self.shared.native_vnni_sweepable
        )

    @property
    def observed_formats(self) -> tuple[str, ...]:
        """Return the sorted unique source formats in this pipeline."""

        return tuple(sorted(set(self.routed.all + self.shared.all)))

    @property
    def overlay_key(self) -> tuple[object, ...]:
        """Return the complete model-independent exact-dispatch identity."""

        return (
            self.hidden_size,
            self.routed_expert_width,
            self.shared_expert_width,
            self.expert_count,
            self.experts_per_token,
            *self.routed.all,
            *self.shared.all,
        )

    def runtime_overlay_key(self, backend: str, m: int) -> tuple[int, ...]:
        """Return backend codebooks, geometry, and M for generated C++ lookup."""

        if m <= 0:
            raise ValueError("MoE prefill overlay M must be positive")
        return (
            *self.routed.runtime_codebooks(backend),
            *self.shared.runtime_codebooks(backend),
            self.hidden_size,
            self.routed_expert_width,
            self.shared_expert_width,
            self.expert_count,
            self.experts_per_token,
            m,
        )


@dataclass(frozen=True, order=True)
class QwenMoERoutedPrefillCase:
    """One source-format key consumed by the grouped routed-expert launcher."""

    hidden_size: int
    routed_expert_width: int
    expert_count: int
    experts_per_token: int
    routed: MoEProjectionFormats
    mixture_overlay_keys: tuple[tuple[object, ...], ...]

    @property
    def evidence_id(self) -> str:
        """Return the cross-language stable identity used by sweep cells."""

        return (
            f"h{self.hidden_size}_r{self.routed_expert_width}"
            f"_e{self.expert_count}_k{self.experts_per_token}"
            f"_g{self.routed.gate}_u{self.routed.up}_d{self.routed.down}"
        )

    @property
    def source_key(self) -> tuple[object, ...]:
        """Return the exact source-format tournament identity."""

        return (
            self.hidden_size,
            self.routed_expert_width,
            self.expert_count,
            self.experts_per_token,
            *self.routed.all,
        )

    def runtime_key(self, backend: str, m: int) -> tuple[int, ...]:
        """Return the model-independent grouped-launch exact-overlay key."""

        if m <= 0:
            raise ValueError("MoE routed-prefill M must be positive")
        return (
            *self.routed.runtime_codebooks(backend),
            self.hidden_size,
            self.routed_expert_width,
            self.expert_count,
            self.experts_per_token,
            m,
        )


@dataclass(frozen=True)
class QwenMoEGGUFPatternInventory:
    """Validated immutable source records and deduplicated execution mixtures."""

    schema_version: str
    source_digest: str
    sources: tuple[Mapping[str, str], ...]
    raw_variants: tuple[Mapping[str, object], ...]
    mixtures: tuple[QwenMoEPrefillMixture, ...]


def _require_exact_keys(
    record: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    """Reject manifest drift rather than partially interpreting new fields."""

    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"{context} schema mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _projection_formats(raw: object, context: str) -> MoEProjectionFormats:
    """Parse one exact gate/up/down object."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(raw, {"gate", "up", "down"}, context)
    labels = tuple(str(raw[name]) for name in ("gate", "up", "down"))
    if any(
        not label or not re.fullmatch(r"[A-Z0-9_]+", label)
        for label in labels
    ):
        raise ValueError(f"{context} contains an invalid format label")
    return MoEProjectionFormats(*labels)


def _manifest_digest(raw: Mapping[str, object]) -> str:
    """Hash canonical source bytes for downstream policy provenance."""

    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_inventory(path: Path) -> QwenMoEGGUFPatternInventory:
    """Load and fully validate one reviewed production-pattern manifest."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Qwen MoE GGUF manifest root must be an object")
    _require_exact_keys(raw, {"schema_version", "sources", "variants"}, "root")
    if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Qwen MoE GGUF manifest version is unsupported")

    sources_raw = raw["sources"]
    variants_raw = raw["variants"]
    if not isinstance(sources_raw, list) or not isinstance(variants_raw, list):
        raise ValueError("Qwen MoE GGUF sources and variants must be arrays")
    sources = []
    for index, source in enumerate(sources_raw):
        if not isinstance(source, Mapping):
            raise ValueError(f"source {index} must be an object")
        _require_exact_keys(
            source,
            {"release_id", "repository", "revision"},
            f"source {index}",
        )
        sources.append({key: str(source[key]) for key in source})
    expected_sources = tuple(
        {
            "release_id": release_id,
            "repository": repository,
            "revision": revision,
        }
        for release_id, repository, revision in PINNED_QWEN_MOE_REPOSITORIES
    )
    if tuple(sources) != expected_sources:
        raise ValueError("Qwen MoE GGUF source revisions are not the pinned set")

    release_models = {
        model.release_id: model
        for model in QWEN_RELEASE_MODELS
        if model.kind == QwenModelKind.MOE
    }
    source_by_release = {source["release_id"]: source for source in sources}
    mixture_uses: dict[
        tuple[object, ...], list[QwenMoEPatternUse]
    ] = defaultdict(list)
    raw_variants: list[Mapping[str, object]] = []
    variant_identities: set[tuple[str, str]] = set()

    for variant_index, variant in enumerate(variants_raw):
        if not isinstance(variant, Mapping):
            raise ValueError(f"variant {variant_index} must be an object")
        _require_exact_keys(variant, {
            "release_id", "repository", "revision", "variant_id", "files",
            "block_count", "expert_count", "experts_per_token", "expert_width",
            "shared_expert_width", "observed_formats", "native_vnni_sweepable",
            "unsupported_formats", "layer_patterns",
        }, f"variant {variant_index}")
        release_id = str(variant["release_id"])
        variant_id = str(variant["variant_id"])
        identity = release_id, variant_id
        if identity in variant_identities:
            raise ValueError(f"duplicate GGUF variant {identity}")
        variant_identities.add(identity)
        if release_id not in release_models or release_id not in source_by_release:
            raise ValueError(f"{identity}: release is not a reviewed MoE geometry")
        source = source_by_release[release_id]
        if (
            variant["repository"] != source["repository"]
            or variant["revision"] != source["revision"]
        ):
            raise ValueError(f"{identity}: repository provenance mismatch")

        files = variant["files"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"{identity}: files must be a nonempty array")
        for file_index, file in enumerate(files):
            if not isinstance(file, Mapping):
                raise ValueError(f"{identity}: file {file_index} must be an object")
            _require_exact_keys(
                file,
                {"path", "size_bytes", "sha256"},
                f"{identity} file {file_index}",
            )
            if (
                not str(file["path"]).endswith(".gguf")
                or int(file["size_bytes"]) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(file["sha256"]))
            ):
                raise ValueError(f"{identity}: invalid LFS file provenance")

        block_count = int(variant["block_count"])
        expert_width = int(variant["expert_width"])
        shared_width = int(variant["shared_expert_width"])
        expert_count = int(variant["expert_count"])
        experts_per_token = int(variant["experts_per_token"])
        model = release_models[release_id]
        if (
            block_count <= 0
            or expert_width != model.feed_forward_size
            or shared_width <= 0
            or expert_count <= 0
            or experts_per_token <= 0
            or experts_per_token > expert_count
        ):
            raise ValueError(f"{identity}: MoE geometry metadata is inconsistent")

        patterns = variant["layer_patterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"{identity}: layer patterns must be nonempty")
        covered_layers = []
        variant_formats: set[str] = set()
        for pattern_index, pattern in enumerate(patterns):
            if not isinstance(pattern, Mapping):
                raise ValueError(f"{identity}: pattern {pattern_index} must be an object")
            _require_exact_keys(
                pattern,
                {"layers", "routed", "shared"},
                f"{identity} pattern {pattern_index}",
            )
            layers_raw = pattern["layers"]
            if not isinstance(layers_raw, list) or not layers_raw:
                raise ValueError(f"{identity}: pattern layers must be nonempty")
            layers = tuple(int(layer) for layer in layers_raw)
            if tuple(sorted(set(layers))) != layers:
                raise ValueError(f"{identity}: pattern layers are not sorted unique")
            covered_layers.extend(layers)
            routed = _projection_formats(
                pattern["routed"],
                f"{identity} pattern {pattern_index} routed",
            )
            shared = _projection_formats(
                pattern["shared"],
                f"{identity} pattern {pattern_index} shared",
            )
            variant_formats.update(routed.all + shared.all)
            key = (
                model.hidden_size,
                expert_width,
                shared_width,
                expert_count,
                experts_per_token,
                routed,
                shared,
            )
            mixture_uses[key].append(QwenMoEPatternUse(
                release_id=release_id,
                variant_id=variant_id,
                layers=layers,
            ))

        if sorted(covered_layers) != list(range(block_count)):
            raise ValueError(f"{identity}: layer patterns are not total and disjoint")
        observed_formats = sorted(
            str(label) for label in variant["observed_formats"]
        )
        if observed_formats != sorted(variant_formats):
            raise ValueError(f"{identity}: observed format summary is stale")
        unsupported = sorted(variant_formats - set(FORMAT_BY_LABEL))
        if unsupported != sorted(
            str(label) for label in variant["unsupported_formats"]
        ):
            raise ValueError(f"{identity}: unsupported format summary is stale")
        if bool(variant["native_vnni_sweepable"]) != (not unsupported):
            raise ValueError(f"{identity}: NativeVNNI eligibility is stale")
        raw_variants.append(variant)

    mixtures = tuple(sorted(
        (
            QwenMoEPrefillMixture(
                hidden_size=int(key[0]),
                routed_expert_width=int(key[1]),
                shared_expert_width=int(key[2]),
                expert_count=int(key[3]),
                experts_per_token=int(key[4]),
                routed=key[5],
                shared=key[6],
                uses=tuple(sorted(uses)),
            )
            for key, uses in mixture_uses.items()
        ),
        key=lambda mixture: mixture.overlay_key,
    ))
    return QwenMoEGGUFPatternInventory(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source_digest=_manifest_digest(raw),
        sources=tuple(sources),
        raw_variants=tuple(raw_variants),
        mixtures=mixtures,
    )


@lru_cache(maxsize=1)
def load_qwen_moe_gguf_pattern_inventory() -> QwenMoEGGUFPatternInventory:
    """Return the memoized, fully validated production pattern inventory."""

    return _load_inventory(QWEN_MOE_GGUF_PATTERN_MANIFEST_PATH)


def qwen_moe_prefill_mixtures(
    *,
    native_vnni_only: bool = False,
) -> tuple[QwenMoEPrefillMixture, ...]:
    """Return all unique production mixtures, optionally restricted by path."""

    mixtures = load_qwen_moe_gguf_pattern_inventory().mixtures
    if not native_vnni_only:
        return mixtures
    return tuple(
        mixture for mixture in mixtures if mixture.native_vnni_sweepable
    )


@lru_cache(maxsize=1)
def qwen_moe_routed_prefill_cases() -> tuple[QwenMoERoutedPrefillCase, ...]:
    """Collapse full GGUF mixtures onto routed grouped-launch source keys.

    A full mixture whose shared expert uses BF16/F16/MXFP4 still has a valid
    NativeVNNI routed path. Shared-expert execution is a dense GEMM surface, so
    it must not suppress or duplicate grouped routed evidence.
    """

    grouped: dict[tuple[object, ...], list[tuple[object, ...]]] = defaultdict(list)
    routed_by_key: dict[tuple[object, ...], MoEProjectionFormats] = {}
    for mixture in qwen_moe_prefill_mixtures():
        if not mixture.routed.native_vnni_sweepable:
            continue
        key = (
            mixture.hidden_size,
            mixture.routed_expert_width,
            mixture.expert_count,
            mixture.experts_per_token,
            *mixture.routed.all,
        )
        grouped[key].append(mixture.overlay_key)
        routed_by_key[key] = mixture.routed

    return tuple(
        QwenMoERoutedPrefillCase(
            hidden_size=int(key[0]),
            routed_expert_width=int(key[1]),
            expert_count=int(key[2]),
            experts_per_token=int(key[3]),
            routed=routed_by_key[key],
            mixture_overlay_keys=tuple(sorted(set(overlay_keys))),
        )
        for key, overlay_keys in sorted(grouped.items())
    )
