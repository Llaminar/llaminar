/**
 * @file Qwen2ModelParityDefinitions.h
 * @brief Authenticated Qwen2 model identities shared by typed parity matrices.
 *
 * Topology files declare only their participants and numerical contract. The
 * real GGUF, Hugging Face reference pack, prompt, tokens, and architectural
 * dimensions live here so every generated Qwen2 cell proves the same model.
 */

#pragma once

#include "../ModelParityDefinition.h"

#include <utility>

namespace llaminar2::test::parity::qwen2
{
    /** @return Canonical Qwen2.5-0.5B Q4_0 real-model/reference identity. */
    inline ModelParityModelDefinition qwen2Q40ParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen2_Q4_0",
            .model_path =
                "models/qwen2.5-0.5b-instruct-q4_0.gguf",
            .reference_directory =
                "pytorch_qwen2_5_0_5b_instruct_q4_0_snapshots",
            .prompt = "The quick brown fox jumps over the lazy dog",
            .token_ids = {
                785, 3974, 13876, 38835, 34208,
                916, 279, 15678, 5562,
            },
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 24,
            .attention_heads = 14,
            .kv_heads = 2,
        };
    }

    /**
     * @brief Join the canonical Qwen2 model to one typed topology.
     *
     * @param topology Exact participants, ranks, and collective policy.
     * @param thresholds Numerical acceptance contract for that topology.
     * @return Definition ready for central matrix expansion.
     */
    inline ModelParityDefinition qwen2Q40ParityDefinition(
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds)
    {
        ModelParityDefinition definition;
        definition.model = qwen2Q40ParityModel();
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        return definition;
    }
} // namespace llaminar2::test::parity::qwen2
