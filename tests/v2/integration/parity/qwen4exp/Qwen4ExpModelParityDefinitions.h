/**
 * @file Qwen4ExpModelParityDefinitions.h
 * @brief Exact Flash Next main-GGUF identity for canonical parity expansion.
 *
 * Main weights declare qwen4exp with48 decoder blocks, four residual branches,
 * and separate QSA/GDN and PLE state. The independently exported MTP GGUF must
 * acquire an authenticated sidecar manifest before a definition enables its
 * predictor axis; a missing sidecar never changes the requested runtime mode.
 */
#pragma once

#include "../ModelParityDefinition.h"

namespace llaminar2::test::parity::qwen4exp
{
    /** First split owns exact metadata while its directory aggregates all four shards. */
    inline constexpr const char *kMainModelPath =
        "/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

    /** Separate source identity; never silently bound to the main-model descriptor. */
    inline constexpr const char *kPredictorModelPath =
        "/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/MTP/"
        "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

    /**
     * @brief Declare one main-model numerical cell through the canonical expander.
     * @param topology Explicit participant, expert ownership and collective policy.
     * @param reference_directory CPU/FP32 pack generated from this exact split set.
     * @return Definition with strict inherited numerical gates and explicit hybrid restore.
     *
     * No topology is registered here: runtime storage and placement remain the
     * typed caller's responsibility. This identity cannot advertise an MTP pack
     * until the separate predictor descriptor is authenticated by the framework.
     */
    inline ModelParityDefinition qwen4ExpMainParityDefinition(
        ModelParityTopologyDefinition topology, std::string reference_directory)
    {
        ModelParityDefinition definition;
        definition.model = {
            .test_id = "Qwen38FlashNext_UDQ4KXL",
            .model_path = kMainModelPath,
            .reference_directory = std::move(reference_directory),
            .prompt = "The quick brown fox jumps over the lazy dog",
            .decode_steps = 2,
            .max_seq_len = 4096,
            .transformer_layers = 48,
            .attention_heads = 24,
            .kv_heads = 2,
            .maximum_mtp_draft_depth = 0,
            .prefix_state = ModelParityPrefixState::HybridRecurrent,
        };
        definition.topology = std::move(topology);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Disabled;
        definition.features.ngram_storage = {
            ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM},
            ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::MemoryMapped},
        };
        return definition;
    }
    /**
     * @return Exact explicit host-cache qualification axes: SSD-only response and bounded positive RAM capacity.
     * @note These are requested ceilings, not reservations or a promise that PMA may change earlier owners.
     */
    inline std::vector<ImmutableRowTablePolicy> qwen4ExpHostAdaptiveStorageAxes()
    {
        ImmutableRowTablePolicy zero{.storage=ImmutableRowStoragePreference::HostAdaptiveRAMSSD};
        auto bounded=zero;
        bounded.host_adaptive={.maximum_cache_rows=64,.cache_physical_byte_limit=1u<<20};
        return {zero,bounded};
    }
} // namespace llaminar2::test::parity::qwen4exp
