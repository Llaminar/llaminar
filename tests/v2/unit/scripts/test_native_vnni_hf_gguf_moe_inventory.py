#!/usr/bin/env python3
"""Device-free regressions for bounded production GGUF inventory extraction."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.hf_gguf_moe_inventory import (  # noqa: E402
    GGUFHeaderTruncated,
    GGUFInventoryError,
    HubGGUFFile,
    HubGGUFVariant,
    extract_moe_variant_record,
    group_hub_gguf_variants,
    parse_gguf_header,
)


def _string(value: str) -> bytes:
    """Encode one synthetic GGUF string."""

    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_string(key: str, value: str) -> bytes:
    """Encode one string metadata entry."""

    return _string(key) + struct.pack("<I", 8) + _string(value)


def _metadata_u32(key: str, value: int) -> bytes:
    """Encode one uint32 metadata entry."""

    return _string(key) + struct.pack("<II", 4, value)


def _metadata_string_array(key: str, values: tuple[str, ...]) -> bytes:
    """Encode a tokenizer-like string array that the parser must skip."""

    return (
        _string(key)
        + struct.pack("<IIQ", 9, 8, len(values))
        + b"".join(_string(value) for value in values)
    )


def _tensor(name: str, dimensions: tuple[int, ...], type_id: int) -> bytes:
    """Encode one payload-free tensor-info record."""

    return (
        _string(name)
        + struct.pack("<I", len(dimensions))
        + struct.pack("<" + "Q" * len(dimensions), *dimensions)
        + struct.pack("<IQ", type_id, 0)
    )


def _gguf(
    tensors: tuple[tuple[str, tuple[int, ...], int], ...],
    *,
    name: str = "Qwen3.6-35B-A3B",
    block_count: int = 2,
) -> bytes:
    """Construct a small valid GGUF v3 header entirely in memory."""

    metadata = (
        _metadata_string("general.architecture", "qwen35moe"),
        _metadata_string("general.name", name),
        _metadata_u32("qwen35moe.block_count", block_count),
        _metadata_u32("qwen35moe.expert_count", 256),
        _metadata_u32("qwen35moe.expert_used_count", 8),
        _metadata_u32("qwen35moe.expert_feed_forward_length", 512),
        _metadata_u32("qwen35moe.expert_shared_feed_forward_length", 512),
        _metadata_string_array(
            "tokenizer.ggml.tokens", ("alpha", "beta", "gamma")
        ),
    )
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + b"".join(metadata)
        + b"".join(_tensor(*tensor) for tensor in tensors)
    )
    padding = (-len(header)) % 32
    return header + b"\x00" * padding


def _layer_tensors(
    layer: int,
    *,
    routed_down: int,
) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    """Construct all routed/shared role descriptors for one MoE layer."""

    return (
        (f"blk.{layer}.ffn_gate_exps.weight", (2048, 512, 256), 22),
        (f"blk.{layer}.ffn_up_exps.weight", (2048, 512, 256), 22),
        (f"blk.{layer}.ffn_down_exps.weight", (512, 2048, 256), routed_down),
        (f"blk.{layer}.ffn_gate_shexp.weight", (2048, 512), 14),
        (f"blk.{layer}.ffn_up_shexp.weight", (2048, 512), 14),
        (f"blk.{layer}.ffn_down_shexp.weight", (512, 2048), 14),
    )


class NativeVNNIHFGGUFMoeInventoryTest(unittest.TestCase):
    """Prove bounded parsing, split grouping, and exact MoE signatures."""

    def test_parser_skips_tokenizer_arrays_and_retains_tensor_info(self) -> None:
        payload = _gguf(_layer_tensors(0, routed_down=21), block_count=1)

        parsed = parse_gguf_header(payload)

        self.assertEqual(parsed.version, 3)
        self.assertEqual(parsed.metadata["general.name"], "Qwen3.6-35B-A3B")
        self.assertNotIn("tokenizer.ggml.tokens", parsed.metadata)
        self.assertEqual(len(parsed.tensors), 6)
        self.assertEqual(parsed.tensors[0].dimensions, (2048, 512, 256))
        self.assertEqual(parsed.tensors[0].format_label, "IQ2_S")
        self.assertEqual(parsed.data_offset, len(payload))

    def test_truncated_prefix_reports_required_byte_boundary(self) -> None:
        payload = _gguf(_layer_tensors(0, routed_down=21), block_count=1)

        with self.assertRaises(GGUFHeaderTruncated) as raised:
            parse_gguf_header(payload[:64])

        self.assertGreater(raised.exception.required_bytes, 64)

    def test_split_paths_are_grouped_without_payload_sizes_affecting_identity(self) -> None:
        siblings = [
            {
                "rfilename": (
                    "UD-IQ3_S/Qwen3.5-122B-A10B-UD-IQ3_S-00002-of-00002.gguf"
                ),
                "size": 50_000_000_000,
                "lfs": {"sha256": "b" * 64},
            },
            {
                "rfilename": (
                    "UD-IQ3_S/Qwen3.5-122B-A10B-UD-IQ3_S-00001-of-00002.gguf"
                ),
                "size": 11_000_000,
                "lfs": {"sha256": "a" * 64},
            },
            {
                "rfilename": "imatrix_unsloth.gguf",
                "size": 10,
                "lfs": {"sha256": "c" * 64},
            },
        ]

        variants = group_hub_gguf_variants("Qwen3.5-122B-A10B", siblings)

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].variant_id, "UD-IQ3_S")
        self.assertEqual(
            [file.sha256 for file in variants[0].files], ["a" * 64, "b" * 64]
        )

    def test_real_style_mixed_layer_patterns_are_preserved(self) -> None:
        payload = _gguf(
            _layer_tensors(0, routed_down=21)
            + _layer_tensors(1, routed_down=23)
        )
        source = HubGGUFFile("model.gguf", len(payload), "a" * 64)

        record = extract_moe_variant_record(
            release_id="Qwen3.6-35B-A3B",
            repository="unsloth/Qwen3.6-35B-A3B-GGUF",
            revision="1" * 40,
            variant=HubGGUFVariant("UD-IQ3_S", (source,)),
            headers=(parse_gguf_header(payload),),
        )

        self.assertTrue(record["native_vnni_sweepable"])
        self.assertEqual(record["observed_formats"], ["IQ2_S", "IQ3_S", "IQ4_XS", "Q6_K"])
        self.assertEqual(record["layer_patterns"], [
            {
                "layers": [0],
                "routed": {"gate": "IQ2_S", "up": "IQ2_S", "down": "IQ3_S"},
                "shared": {"gate": "Q6_K", "up": "Q6_K", "down": "Q6_K"},
            },
            {
                "layers": [1],
                "routed": {"gate": "IQ2_S", "up": "IQ2_S", "down": "IQ4_XS"},
                "shared": {"gate": "Q6_K", "up": "Q6_K", "down": "Q6_K"},
            },
        ])

    def test_unknown_execution_format_is_explicitly_not_sweepable(self) -> None:
        tensors = tuple(
            (name, dimensions, 39 if "gate_exps" in name else type_id)
            for name, dimensions, type_id in _layer_tensors(0, routed_down=21)
        )
        payload = _gguf(tensors, block_count=1)
        source = HubGGUFFile("model.gguf", len(payload), "a" * 64)

        record = extract_moe_variant_record(
            release_id="Qwen3.6-35B-A3B",
            repository="unsloth/Qwen3.6-35B-A3B-GGUF",
            revision="1" * 40,
            variant=HubGGUFVariant("MXFP4_MOE", (source,)),
            headers=(parse_gguf_header(payload),),
        )

        self.assertFalse(record["native_vnni_sweepable"])
        self.assertEqual(record["unsupported_formats"], ["MXFP4"])

    def test_missing_role_is_fatal_instead_of_inventing_a_grouping(self) -> None:
        payload = _gguf(_layer_tensors(0, routed_down=21)[:-1], block_count=1)
        source = HubGGUFFile("model.gguf", len(payload), "a" * 64)

        with self.assertRaisesRegex(GGUFInventoryError, "expected"):
            extract_moe_variant_record(
                release_id="Qwen3.6-35B-A3B",
                repository="unsloth/Qwen3.6-35B-A3B-GGUF",
                revision="1" * 40,
                variant=HubGGUFVariant("UD-IQ3_S", (source,)),
                headers=(parse_gguf_header(payload),),
            )


if __name__ == "__main__":
    unittest.main()
