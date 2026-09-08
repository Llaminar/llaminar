"""Shared authenticated Qwen3.6 MTP sidecar reference utilities.

Dense and MoE Qwen3.6 models use different decoder-layer implementations, but
their recursive predictor corpus has one lifecycle contract: validate typed
branch requests, replay only the necessary speculative depth, publish additive
NPY files atomically, and advance the authenticated canonical-depth marker only
after generation completes.  Keeping those operations here prevents the two
reference generators from acquiring subtly different failure semantics.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from .snapshot_metadata import write_metadata_atomically


def normalize_mtp_branch_override_batches(raw_overrides):
    """Return validated batches of decode-step recursive token overrides.

    One mapping remains accepted for direct developer use.  Campaign runners
    may pass an array of mappings to amortize one loaded sidecar context across
    several independently observed production branches.
    """

    raw_batches = (
        raw_overrides if isinstance(raw_overrides, list) else [raw_overrides]
    )
    if not raw_batches or any(not isinstance(batch, dict) for batch in raw_batches):
        raise ValueError(
            "MTP branch overrides must be a JSON object or a non-empty "
            "array of JSON objects"
        )

    normalized_batches = []
    for batch in raw_batches:
        normalized = {}
        for step, tokens in batch.items():
            if not isinstance(tokens, list) or any(
                isinstance(token, (list, dict)) for token in tokens
            ):
                raise ValueError(
                    "Every MTP branch override must be one flat token array"
                )
            normalized[int(step)] = [int(token) for token in tokens]
        normalized_batches.append(normalized)
    return normalized_batches


def promote_mtp_sidecar_metadata(metadata_path: Path, maximum_depth: int) -> None:
    """Atomically publish newly completed canonical sidecar capacity.

    The existing metadata authenticates the expensive main-model trajectory.
    Sidecar-only repair preserves every field except the independently
    repairable maximum predictor depth.  Callers invoke this only after all NPY
    files and the sidecar schema marker are durable.
    """

    if not metadata_path.is_file():
        raise ValueError(
            "Sidecar-only generation requires existing canonical metadata"
        )
    key = "mtp_sidecar_max_draft_depth:"
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    replacement = f"{key} {maximum_depth}"
    matches = [index for index, line in enumerate(lines) if line.startswith(key)]
    if len(matches) > 1:
        raise ValueError(
            "Canonical metadata contains duplicate MTP sidecar capacity fields"
        )
    if matches:
        lines[matches[0]] = replacement
    else:
        lines.append(replacement)
    write_metadata_atomically(metadata_path, lines)


def mtp_sidecar_replay_depth(
    step: int,
    max_draft_depth: int,
    draft_token_overrides: dict[int, list[int]],
) -> int:
    """Return the minimum predictor depth needed by one additive branch pass.

    MTP0 must be replayed at every preceding committed position to preserve the
    shifted predictor cache.  Deeper rows are speculative and are needed only
    at the decode step whose production branch is being materialized.
    """

    if not draft_token_overrides:
        return max_draft_depth
    tokens = draft_token_overrides.get(step)
    return 1 if tokens is None else len(tokens) + 1


def save_mtp_snapshot_atomic(path: Path, payload: np.ndarray) -> None:
    """Atomically publish one additive MTP reference tensor."""

    destination = os.fspath(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=os.path.dirname(destination),
            delete=False,
        ) as output:
            temporary = output.name
            np.save(output, payload)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
