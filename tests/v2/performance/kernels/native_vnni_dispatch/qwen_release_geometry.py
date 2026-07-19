"""Declarative Qwen 3.5/3.6 NativeVNNI production geometry catalog.

The learned dispatcher must never branch on a model name.  Model releases are
still valuable evidence sources, however: every matrix geometry used by a
released dense or MoE checkpoint should receive an exact measured overlay in
addition to remaining covered by generic rules.  This module keeps those two
concerns separate.  It records architecture parameters from the official Qwen
configs, derives the quantized text-backbone matrix dimensions, and collapses
identical ``(N, K)`` pairs before they enter the trainer shape manifest.

The catalog intentionally excludes FP32 router matrices, norms, convolutions,
and recurrent state tensors because they do not dispatch through NativeVNNI.
It includes the fused attention-Q/gate projection, K/V projection, attention
output, all four GDN projection geometries, dense or expert FFN projections,
the MTP hidden/embedding projection, and the language-model head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


QWEN_RELEASE_CATALOG_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "qwen35_qwen36_release_models_v1.json"
)


class QwenModelKind(str, Enum):
    """Text-backbone feed-forward architecture used by one release."""

    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True, order=True)
class QwenReleaseModel:
    """Architecture parameters needed to derive NativeVNNI matrices.

    Attributes:
        release_id: Stable official release name used only for evidence
            provenance and tests.  It never participates in runtime dispatch.
        kind: Whether the feed-forward block is dense or expert-routed.
        hidden_size: Text-backbone residual width.
        feed_forward_size: Dense intermediate width or per-expert MoE width.
        attention_head_count: Number of full-attention query heads.
        attention_kv_head_count: Number of full-attention K and V heads.
        gdn_value_head_count: Number of GDN value heads.  Each value head has
            the shared 128-element value dimension.
        gdn_time_step_rank: Width of each learned GDN alpha/beta projection.
    """

    release_id: str
    kind: QwenModelKind
    hidden_size: int
    feed_forward_size: int
    attention_head_count: int
    attention_kv_head_count: int
    gdn_value_head_count: int
    gdn_time_step_rank: int

    def __post_init__(self) -> None:
        """Reject malformed release data before it can alter a policy corpus."""

        dimensions = (
            self.hidden_size,
            self.feed_forward_size,
            self.attention_head_count,
            self.attention_kv_head_count,
            self.gdn_value_head_count,
            self.gdn_time_step_rank,
        )
        if not self.release_id or any(value <= 0 for value in dimensions):
            raise ValueError("Qwen release geometry fields must be positive")

    @property
    def attention_query_width(self) -> int:
        """Return the unfused full-attention query width."""

        return self.attention_head_count * QWEN_ATTENTION_HEAD_DIMENSION

    @property
    def attention_kv_width(self) -> int:
        """Return the width of either the K or V projection."""

        return self.attention_kv_head_count * QWEN_ATTENTION_HEAD_DIMENSION

    @property
    def gdn_inner_width(self) -> int:
        """Return the recurrent GDN value/gate width."""

        return self.gdn_value_head_count * QWEN_GDN_KEY_HEAD_DIMENSION

    @property
    def gdn_qkv_width(self) -> int:
        """Return the packed GDN Q/K/V output width.

        Qwen's GDN projection contains one key-width Q component, one
        key-width K component, and the complete value-head payload.
        """

        key_width = QWEN_GDN_KEY_HEAD_COUNT * QWEN_GDN_KEY_HEAD_DIMENSION
        return 2 * key_width + self.gdn_inner_width


@dataclass(frozen=True, order=True)
class QwenProjectionUse:
    """One model/projection reason that requires a matrix geometry."""

    release_id: str
    projection: str


@dataclass(frozen=True, order=True)
class QwenReleaseGeometry:
    """One deduplicated runtime geometry and all release uses that own it."""

    n: int
    k: int
    uses: tuple[QwenProjectionUse, ...]

    @property
    def shape_name(self) -> str:
        """Return a model-agnostic trainer name for the exact-overlay row."""

        return f"Qwen35Release_{self.n}x{self.k}"


def _load_release_catalog() -> tuple[
    int,
    int,
    int,
    int,
    tuple[QwenReleaseModel, ...],
]:
    """Load the cross-language release catalog used by every trainer backend."""

    raw = json.loads(QWEN_RELEASE_CATALOG_PATH.read_text(encoding="utf-8"))
    required_root = {
        "schema_version",
        "vocabulary_size",
        "attention_head_dimension",
        "gdn_key_head_count",
        "gdn_key_head_dimension",
        "releases",
    }
    if not isinstance(raw, dict) or set(raw) != required_root:
        raise ValueError("Qwen release catalog root schema is invalid")
    if raw["schema_version"] != "qwen35-qwen36-release-models-v1":
        raise ValueError("Qwen release catalog version is unsupported")
    required_release = {
        "release_id",
        "kind",
        "hidden_size",
        "feed_forward_size",
        "attention_head_count",
        "attention_kv_head_count",
        "gdn_value_head_count",
        "gdn_time_step_rank",
    }
    releases = raw["releases"]
    if not isinstance(releases, list) or not releases:
        raise ValueError("Qwen release catalog must contain releases")
    models = []
    for record in releases:
        if not isinstance(record, dict) or set(record) != required_release:
            raise ValueError("Qwen release catalog model schema is invalid")
        models.append(QwenReleaseModel(
            release_id=str(record["release_id"]),
            kind=QwenModelKind(str(record["kind"])),
            hidden_size=int(record["hidden_size"]),
            feed_forward_size=int(record["feed_forward_size"]),
            attention_head_count=int(record["attention_head_count"]),
            attention_kv_head_count=int(record["attention_kv_head_count"]),
            gdn_value_head_count=int(record["gdn_value_head_count"]),
            gdn_time_step_rank=int(record["gdn_time_step_rank"]),
        ))
    scalar_values = tuple(int(raw[name]) for name in (
        "vocabulary_size",
        "attention_head_dimension",
        "gdn_key_head_count",
        "gdn_key_head_dimension",
    ))
    if any(value <= 0 for value in scalar_values):
        raise ValueError("Qwen release catalog constants must be positive")
    return (*scalar_values, tuple(models))


(
    QWEN_VOCABULARY_SIZE,
    QWEN_ATTENTION_HEAD_DIMENSION,
    QWEN_GDN_KEY_HEAD_COUNT,
    QWEN_GDN_KEY_HEAD_DIMENSION,
    QWEN_RELEASE_MODELS,
) = _load_release_catalog()


def model_projection_geometries(
    model: QwenReleaseModel,
) -> tuple[tuple[str, int, int], ...]:
    """Derive every NativeVNNI matrix used by one text backbone.

    The returned tuple is deliberately expressed only as projection role,
    output columns ``N``, and reduction width ``K``.  Tensor names and model
    identities remain corpus metadata; generated C++ dispatch consumes only
    codebook, geometry, and work size.
    """

    hidden = model.hidden_size
    projections = [
        ("attention_q_gate", 2 * model.attention_query_width, hidden),
        ("attention_kv", model.attention_kv_width, hidden),
        ("attention_output", hidden, model.attention_query_width),
        ("gdn_qkv", model.gdn_qkv_width, hidden),
        ("gdn_z", model.gdn_inner_width, hidden),
        ("gdn_time", model.gdn_time_step_rank, hidden),
        ("gdn_output", hidden, model.gdn_inner_width),
        # NextN concatenates normalized hidden and embedding rows, hence K=2H.
        ("mtp_hidden_embedding", hidden, 2 * hidden),
    ]
    if model.kind == QwenModelKind.DENSE:
        projections.extend((
            ("ffn_gate_up", model.feed_forward_size, hidden),
            ("ffn_down", hidden, model.feed_forward_size),
        ))
    else:
        projections.extend((
            ("expert_gate_up", model.feed_forward_size, hidden),
            ("expert_down", hidden, model.feed_forward_size),
        ))
    projections.append(("lm_head", QWEN_VOCABULARY_SIZE, hidden))
    return tuple(projections)


def qwen_release_geometries() -> tuple[QwenReleaseGeometry, ...]:
    """Return the stable deduplicated geometry inventory for all releases."""

    uses_by_geometry: dict[tuple[int, int], list[QwenProjectionUse]] = {}
    for model in QWEN_RELEASE_MODELS:
        for projection, n, k in model_projection_geometries(model):
            uses_by_geometry.setdefault((n, k), []).append(
                QwenProjectionUse(model.release_id, projection)
            )
    return tuple(
        QwenReleaseGeometry(
            n=n,
            k=k,
            uses=tuple(sorted(set(uses))),
        )
        for (n, k), uses in sorted(uses_by_geometry.items())
    )
