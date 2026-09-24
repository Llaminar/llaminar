#!/usr/bin/env python3
"""Regressions for production-derived all-format MoE prefill mixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.hf_gguf_moe_inventory import (  # noqa: E402
    PINNED_QWEN_MOE_REPOSITORIES,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    CPU_14B_PLUS_PREFILL_M_BUCKETS,
    GPU_PREFILL_M_BUCKETS,
    moe_prefill_mixture_measurements,
)
from native_vnni_dispatch.qwen_moe_gguf_patterns import (  # noqa: E402
    load_qwen_moe_gguf_pattern_inventory,
    qwen_moe_routed_prefill_cases,
    qwen_moe_prefill_mixtures,
)


class NativeVNNIQwenMoEPrefillPatternsTest(unittest.TestCase):
    """Lock source provenance, layer totality, and backend matrix coverage."""

    def test_inventory_uses_only_reviewed_pinned_repositories(self) -> None:
        inventory = load_qwen_moe_gguf_pattern_inventory()

        self.assertEqual(
            tuple(
                (
                    source["release_id"],
                    source["repository"],
                    source["revision"],
                )
                for source in inventory.sources
            ),
            PINNED_QWEN_MOE_REPOSITORIES,
        )
        self.assertTrue(inventory.source_digest.startswith("sha256:"))
        self.assertEqual(len(inventory.raw_variants), 66)

    def test_every_release_variant_and_layer_contributes_to_a_mixture(self) -> None:
        inventory = load_qwen_moe_gguf_pattern_inventory()
        represented = {
            (use.release_id, use.variant_id, layer)
            for mixture in inventory.mixtures
            for use in mixture.uses
            for layer in use.layers
        }
        expected = {
            (variant["release_id"], variant["variant_id"], layer)
            for variant in inventory.raw_variants
            for layer in range(int(variant["block_count"]))
        }

        self.assertEqual(represented, expected)

    def test_real_qwen36_iq3s_exception_layers_are_concrete(self) -> None:
        matching = [
            (mixture, use)
            for mixture in qwen_moe_prefill_mixtures()
            for use in mixture.uses
            if use.release_id == "Qwen3.6-35B-A3B"
            and use.variant_id == "UD-IQ3_S"
        ]

        self.assertEqual(len(matching), 4)
        by_signature = {
            (mixture.routed.all, mixture.shared.all): use.layers
            for mixture, use in matching
        }
        self.assertEqual(
            by_signature[
                (("IQ2_S", "IQ2_S", "IQ4_XS"),
                 ("Q6_K", "Q6_K", "Q6_K"))
            ],
            tuple(
                layer for layer in range(41)
                if layer not in {34, 38, 39, 40}
            ),
        )
        self.assertEqual(
            by_signature[
                (("IQ2_S", "IQ2_S", "Q6_K"),
                 ("Q6_K", "Q6_K", "Q6_K"))
            ],
            (34, 38),
        )
        self.assertEqual(
            by_signature[
                (("IQ3_S", "IQ3_S", "Q6_K"),
                 ("Q8_0", "Q8_0", "Q8_0"))
            ],
            (39,),
        )
        self.assertEqual(
            by_signature[
                (("Q2_K", "Q2_K", "Q4_K"),
                 ("Q6_K", "Q6_K", "Q6_K"))
            ],
            (40,),
        )
        for mixture, _ in matching:
            self.assertEqual(mixture.expert_count, 256)
            self.assertEqual(mixture.experts_per_token, 8)

    def test_mixture_deduplication_is_model_agnostic(self) -> None:
        mixtures = qwen_moe_prefill_mixtures()

        self.assertEqual(len(mixtures), 63)
        self.assertEqual(len({mixture.overlay_key for mixture in mixtures}), 63)
        self.assertTrue(any(
            {use.release_id for use in mixture.uses}
            >= {"Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B"}
            for mixture in mixtures
        ))

    def test_non_native_formats_remain_visible_but_never_claim_coverage(self) -> None:
        all_mixtures = qwen_moe_prefill_mixtures()
        native_mixtures = qwen_moe_prefill_mixtures(native_vnni_only=True)

        self.assertEqual(len(native_mixtures), 55)
        self.assertEqual(len(all_mixtures) - len(native_mixtures), 8)
        non_native_formats = {
            label
            for mixture in all_mixtures
            if not mixture.native_vnni_sweepable
            for label in mixture.observed_formats
        }
        self.assertTrue({"F16", "BF16", "MXFP4"}.issubset(non_native_formats))
        for mixture in native_mixtures:
            for backend in ("cpu", "cuda", "rocm"):
                key = mixture.runtime_overlay_key(backend, 64)
                self.assertEqual(len(key), 12)

    def test_routing_geometry_is_part_of_every_execution_key(self) -> None:
        mixtures = qwen_moe_prefill_mixtures()

        self.assertTrue(all(
            mixture.expert_count == 256 and mixture.experts_per_token == 8
            for mixture in mixtures
        ))
        self.assertEqual(
            len({mixture.overlay_key for mixture in mixtures}),
            len(mixtures),
        )

    def test_grouped_routed_cases_retain_non_native_shared_provenance(self) -> None:
        cases = qwen_moe_routed_prefill_cases()

        self.assertEqual(len(cases), 49)
        self.assertEqual(len({case.source_key for case in cases}), 49)
        self.assertTrue(all(case.routed.gate == case.routed.up for case in cases))
        represented = {
            overlay_key
            for case in cases
            for overlay_key in case.mixture_overlay_keys
        }
        expected = {
            mixture.overlay_key
            for mixture in qwen_moe_prefill_mixtures()
            if mixture.routed.native_vnni_sweepable
        }
        self.assertEqual(represented, expected)

    def test_every_backend_receives_every_mixture_without_bucket_holes(self) -> None:
        for backend in ("cuda", "rocm"):
            measurements = moe_prefill_mixture_measurements(backend)
            self.assertEqual(len(measurements), 63)
            self.assertTrue(all(
                measurement.m_values == GPU_PREFILL_M_BUCKETS
                for measurement in measurements
            ))
            self.assertEqual(
                sum(
                    len(measurement.m_values)
                    for measurement in measurements
                ),
                63 * len(GPU_PREFILL_M_BUCKETS),
            )

        cpu = moe_prefill_mixture_measurements("cpu")
        self.assertEqual(len(cpu), 63)
        self.assertTrue(all(
            measurement.m_values == CPU_14B_PLUS_PREFILL_M_BUCKETS
            for measurement in cpu
        ))


if __name__ == "__main__":
    unittest.main()
