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
#include "../LongFormJournalPrompt.h"

#include <utility>

namespace llaminar2::test::parity::qwen2
{
    /**
     * @return The complete 0.5B family identity and its fixed acquisition task.
     * @param test_id Stable format-qualified model identity.
     * @param model_path Exact GGUF artifact.
     * @param reference_directory Independent deep mathematical reference pack.
     * @param generation_prompt Exact model-owned workload, independent of topology.
     * @param generation_seed Fixed acquisition seed inherited by every generated cell.
     *
     * Existing model identities retain their proven translation workload. A
     * model definition may explicitly select different text during acquisition;
     * runtime format/backend dispatch must never choose or retry that text.
     */
    inline ModelParityModelDefinition qwen2ParityModel(
        std::string test_id, std::string model_path, std::string reference_directory,
        ModelParityGenerationPrompt generation_prompt = longFormTranslationGenerationPrompt(),
        ModelParityGenerationSeed generation_seed = ModelParityGenerationSeed{})
    {
        return ModelParityModelDefinition{
            .test_id = std::move(test_id),
            .model_path = std::move(model_path),
            .reference_directory = std::move(reference_directory),
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
            .prefix_state = ModelParityPrefixState::AttentionKV,
            .generation_prompt = std::move(generation_prompt),
            .generation_seed = generation_seed,
        };
    }

    /**
     * @return Canonical Qwen2.5-0.5B Q4_0 real-model/reference identity.
     *
     * A fixed next user turn requests the full revision after closed assistant
     * history. Every topology and KV format proves the preserved token prefix;
     * the positive model-owned seed is shared by all of them. EOS stays enabled
     * and each response still owes 384 committed tokens.
     */
    inline ModelParityModelDefinition qwen2Q40ParityModel()
    {
        return qwen2ParityModel("Qwen2_Q4_0", "models/qwen2.5-0.5b-instruct-q4_0.gguf",
                               "pytorch_qwen2_5_0_5b_instruct_q4_0_snapshots",
                               longFormRevisionGenerationPrompt(
                                   "Please provide the complete revised English journal now, including all nine paragraphs in full."),
                               ModelParityGenerationSeed{17});
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
