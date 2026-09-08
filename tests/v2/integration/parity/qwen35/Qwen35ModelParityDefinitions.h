/**
 * @file Qwen35ModelParityDefinitions.h
 * @brief Authenticated dense Qwen3.5 identities and parity contracts.
 *
 * Dense Qwen3.5 parity topologies consume these declarations instead of
 * repeating model paths, Hugging Face reference directories, prompts, and
 * head geometry in each test binary. The geometry is authenticated against
 * the production GGUF metadata and is used to reject impossible TP/PP shapes
 * before a campaign reaches model loading.
 */

#pragma once

#include "../ModelParityDefinition.h"

#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test::parity::qwen35
{
    /** Exact prompt shared by the dense Qwen3.5 reference packs. */
    inline constexpr const char *kQwen35ParityPrompt =
        "The quick brown fox jumps over the lazy dog";

    /** @return Authenticated tokenization of @ref kQwen35ParityPrompt. */
    inline std::vector<int> qwen35ParityTokenIds()
    {
        return {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388};
    }

    /** @return Canonical Qwen3.5-0.8B Q4_0 model/reference identity. */
    inline ModelParityModelDefinition qwen35_08B_Q40ParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen35_08B_Q4_0",
            .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
            .reference_directory = "pytorch_qwen35_snapshots",
            .prompt = kQwen35ParityPrompt,
            .token_ids = qwen35ParityTokenIds(),
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 24,
            .attention_heads = 8,
            .kv_heads = 2,
        };
    }

    /** @return Canonical Qwen3.5-4B Q8_0 model/reference identity. */
    inline ModelParityModelDefinition qwen35_4B_Q80ParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen35_4B_Q8_0",
            .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
            .reference_directory = "pytorch_qwen35_4b_snapshots",
            .prompt = kQwen35ParityPrompt,
            .token_ids = qwen35ParityTokenIds(),
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 32,
            .attention_heads = 16,
            .kv_heads = 4,
        };
    }

    /** @return Extended 20-step Qwen3.5-4B Q8_0 reference identity. */
    inline ModelParityModelDefinition qwen35_4B_Q80Decode20ParityModel()
    {
        auto model = qwen35_4B_Q80ParityModel();
        model.test_id = "Qwen35_4B_Q8_0_Decode20";
        model.reference_directory = "pytorch_qwen35_4b_decode20_snapshots";
        model.decode_steps = 20;
        return model;
    }

    /** @return Canonical Qwen3.5-27B Q8_0 model/reference identity. */
    inline ModelParityModelDefinition qwen35_27B_Q80ParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen35_27B_Q8_0",
            .model_path = "models/Qwen3.5-27B-Q8_0.gguf",
            .reference_directory = "pytorch_qwen35_27b_snapshots",
            .prompt = kQwen35ParityPrompt,
            .token_ids = qwen35ParityTokenIds(),
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 64,
            .attention_heads = 24,
            .kv_heads = 4,
        };
    }

    /** @return Canonical Qwen3.5-27B Q4_K_M model/reference identity. */
    inline ModelParityModelDefinition qwen35_27B_Q4KMParityModel()
    {
        auto model = qwen35_27B_Q80ParityModel();
        model.test_id = "Qwen35_27B_Q4_K_M";
        model.model_path = "models/Qwen3.5-27B-Q4_K_M.gguf";
        model.reference_directory = "pytorch_qwen35_27b_q4km_snapshots";
        return model;
    }

    /**
     * @brief Join one dense Qwen3.5 identity to topology and precision axes.
     *
     * @param model Authenticated GGUF/reference identity.
     * @param topology Exact physical participants and orchestration shape.
     * @param thresholds Default numerical contract.
     * @param kv_precisions Supported KV-cache formats for this topology.
     * @param overrides Pair-specific numerical contracts.
     * @return Definition ready for canonical matrix expansion.
     */
    inline ModelParityDefinition qwen35ParityDefinition(
        ModelParityModelDefinition model,
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds,
        std::vector<KVCachePrecision> kv_precisions = {
            KVCachePrecision::FP16,
        },
        std::vector<ModelParityPrecisionThresholdOverride> overrides = {})
    {
        ModelParityDefinition definition;
        definition.model = std::move(model);
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = std::move(kv_precisions);
        definition.precisions.threshold_overrides = std::move(overrides);
        return definition;
    }
} // namespace llaminar2::test::parity::qwen35
