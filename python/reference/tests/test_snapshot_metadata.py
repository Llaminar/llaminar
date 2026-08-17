"""Regressions for production parity reference-pack identity."""

from __future__ import annotations

import hashlib

import pytest

from python.reference.snapshot_metadata import (
    REFERENCE_DEVICE,
    REFERENCE_DTYPE,
    REFERENCE_ENGINE,
    REFERENCE_IDENTITY_VERSION,
    build_reference_identity,
    write_metadata_atomically,
)


def test_reference_identity_binds_exact_model_prompt_tokens_and_depth(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"real model weights\x00\xff")
    prompt = "first line\nsecond line"
    identity = build_reference_identity(model, prompt, [7, 11, 13], 5)

    fields = dict(
        line.split(": ", 1) for line in identity.metadata_lines()
    )
    assert fields == {
        "reference_identity_version": str(REFERENCE_IDENTITY_VERSION),
        "reference_engine": REFERENCE_ENGINE,
        "reference_device": REFERENCE_DEVICE,
        "reference_dtype": REFERENCE_DTYPE,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(b"7,11,13").hexdigest(),
        "decode_steps": "5",
    }


@pytest.mark.parametrize(
    ("tokens", "decode_steps", "message"),
    [([], 1, "no token IDs"), ([1], -1, "non-negative")],
)
def test_reference_identity_rejects_incomplete_inputs(
    tmp_path, tokens, decode_steps, message
):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    with pytest.raises(ValueError, match=message):
        build_reference_identity(model, "prompt", tokens, decode_steps)


def test_metadata_is_published_as_one_complete_marker(tmp_path):
    metadata = tmp_path / "snapshots" / "metadata.txt"
    write_metadata_atomically(metadata, ["snapshot_version: 4", "decode_steps: 2"])

    assert metadata.read_bytes() == b"snapshot_version: 4\ndecode_steps: 2\n"
    assert not list(metadata.parent.glob(".metadata.txt.*.tmp"))
