/**
 * @file Qwen3ModelParityDefinitions.h
 * @brief Authenticated Qwen3 identities and numerical contracts for parity.
 *
 * Every Qwen3 topology joins the same real GGUF and Hugging Face reference
 * pack through this typed declaration.  Test files describe only physical
 * participants and topology-specific numerical evidence; precision products
 * are expanded by the canonical model-parity generator.
 */

#pragma once

#include "../ModelParityDefinition.h"
#include "../LongFormJournalPrompt.h"

#include <utility>

namespace llaminar2::test::parity::qwen3
{
    /** Distribution threshold for production compressed-KV/reference parity. */
    inline constexpr float kQwen3CompressedKVKLThreshold = 0.15f;

    /**
     * @return Canonical Qwen3-0.6B Q8_0 model/reference identity.
     *
     * Keep the generation assistant history terminal: another user turn makes
     * this model's template remove its earlier non-thinking assistant header,
     * changing the seed token prefix. The observer proves the actual complete
     * prefix and output horizon independently of this source-owned prompt.
     */
    inline ModelParityModelDefinition qwen3Q80ParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen3_Q8_0",
            .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
            .reference_directory = "pytorch_qwen3_snapshots",
            .prompt = "The quick brown fox jumps over the lazy dog",
            .token_ids = {
                785, 3974, 13876, 38835, 34208,
                916, 279, 15678, 5562,
            },
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 28,
            .attention_heads = 16,
            .kv_heads = 8,
            .prefix_state = ModelParityPrefixState::AttentionKV,
            .generation_prompt = longFormRevisionGenerationPrompt(),
        };
    }

    /**
     * @brief Create the common Qwen3 numerical contract.
     *
     * @param kl_threshold Maximum output-distribution divergence.
     * @param min_top1_accuracy Minimum top-one agreement percentage.
     * @param min_top5_accuracy Minimum top-five agreement percentage.
     * @return Fully initialized backend thresholds.
     */
    inline BackendThresholds qwen3ParityThresholds(
        float kl_threshold,
        float min_top1_accuracy = 80.0f,
        float min_top5_accuracy = 95.0f)
    {
        return BackendThresholds{
            .cosine_threshold = 0.94f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = kl_threshold,
            .min_top1_accuracy = min_top1_accuracy,
            .min_top5_accuracy = min_top5_accuracy,
        };
    }

    /**
     * @brief Join Qwen3 to one typed topology and precision declaration.
     *
     * @param topology Exact participants, rank ownership, and collective.
     * @param thresholds Default numerical contract.
     * @param kv_precisions Supported KV-cache formats for this topology.
     * @param overrides Pair-specific numerical contracts.
     * @return Definition ready for canonical matrix expansion.
     */
    inline ModelParityDefinition qwen3Q80ParityDefinition(
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds,
        std::vector<KVCachePrecision> kv_precisions = {
            KVCachePrecision::FP16,
        },
        std::vector<ModelParityPrecisionThresholdOverride> overrides = {})
    {
        ModelParityDefinition definition;
        definition.model = qwen3Q80ParityModel();
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = std::move(kv_precisions);
        definition.precisions.threshold_overrides = std::move(overrides);
        return definition;
    }
} // namespace llaminar2::test::parity::qwen3
