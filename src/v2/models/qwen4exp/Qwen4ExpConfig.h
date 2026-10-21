/**
 * @file Qwen4ExpConfig.h
 * @brief Authenticated GGUF geometry for the Qwen3.8 Flash Next text decoder.
 *
 * Parsing consumes only the loader's immutable directory and metadata. Compute,
 * residual-trunk, and terminal-head widths remain separate; the PLE table retains
 * its actual native source type and row geometry, independent of model file labels.
 * This contract does not advertise a backend or materialize any tensor payload.
 */
#pragma once

#include "../../loaders/ModelLoader.h"
#include <array>
#include <cstdint>
#include <vector>

namespace llaminar2
{
    /** @brief Explicit activation widths at the three model graph boundaries. */
    struct Qwen4ExpActivationGeometry
    {
        uint32_t compute;
        uint32_t trunk;
        uint32_t terminal_head;
    };

    /** @brief HC branch count and low-rank read/write gate bottleneck. */
    struct Qwen4ExpHyperConnectionGeometry
    {
        uint32_t branches;
        uint32_t gate_rank;
    };

    /** @brief Gated DeltaNet head and convolution dimensions from ssm metadata. */
    struct Qwen4ExpDeltaNetGeometry
    {
        uint32_t key_heads;
        uint32_t value_heads;
        uint32_t head_dimension;
        uint32_t value_width;
        uint32_t convolution_kernel;
    };

    /** @brief Core sparse attention and compressed indexer geometry. */
    struct Qwen4ExpSparseAttentionGeometry
    {
        uint32_t query_heads;
        uint32_t kv_heads;
        uint32_t head_dimension;
        uint32_t rotary_dimension;
        uint32_t indexer_query_heads;
        uint32_t indexer_key_heads;
        uint32_t indexer_dimension;
        uint32_t compression_ratio;
        uint32_t selected_token_budget;
        uint32_t selected_complete_blocks;
        uint32_t maximum_selected_tokens;
    };

    /** @brief Routed and shared expert projection geometry. */
    struct Qwen4ExpExpertGeometry
    {
        uint32_t count;
        uint32_t selected;
        uint32_t intermediate;
        uint32_t shared_intermediate;
    };

    /**
     * @brief Exact PLE hash, table and convolution contract.
     *
     * Multipliers/ranges retain all 64 bits. Table coordinates are canonical
     * ModelLoader row-major [rows, row_elements], not raw GGML dimension order.
     * Physical storage identity and mapping lifetime remain loader/store-owned.
     */
    struct Qwen4ExpNgramGeometry
    {
        uint32_t decoder_layer;
        uint32_t maximum_order;
        uint32_t heads_per_order;
        uint32_t heads;
        uint32_t row_elements;
        uint32_t convolution_kernel;
        uint32_t convolution_dilation;
        uint32_t convolution_history;
        uint32_t hash_reset_token;
        std::vector<uint64_t> multipliers;
        std::vector<uint64_t> head_offsets;
        std::vector<uint64_t> head_vocab_sizes;
        GGUFTensorInfo table;
        uint64_t row_bytes;
    };

    /** @brief Per-layer mixer identity; predictor blocks always use sparse attention. */
    enum class Qwen4ExpMixer : uint8_t { GatedDeltaNet, SparseAttention };

    /**
     * @brief Validated immutable configuration for the published Flash Next family.
     *
     * The pinned converter stores total blocks including optional nextn blocks.
     * An absent nextn_predict_layers key explicitly denotes an export without MTP.
     * Required HC/QSA/PLE metadata is never synthesized from defaults.
     */
    class Qwen4ExpConfig final
    {
    public:
        /**
         * @brief Authenticate the loaded GGUF metadata and PLE row directory.
         * @param model Complete split-merged loader directory, without tensor materialization.
         * @throws std::invalid_argument With the offending metadata/tensor name on failure.
         */
        explicit Qwen4ExpConfig(const GGUFModel &model);

        /** @brief Return distinct compute, trunk and vocabulary-head widths. */
        const Qwen4ExpActivationGeometry &activations() const noexcept { return activations_; }
        /** @brief Return residual-branch and gate geometry. */
        const Qwen4ExpHyperConnectionGeometry &hyperConnections() const noexcept { return hc_; }
        /** @brief Return recurrent layer geometry. */
        const Qwen4ExpDeltaNetGeometry &deltaNet() const noexcept { return gdn_; }
        /** @brief Return selected-index sparse attention geometry. */
        const Qwen4ExpSparseAttentionGeometry &sparseAttention() const noexcept { return qsa_; }
        /** @brief Return routed/shared expert geometry. */
        const Qwen4ExpExpertGeometry &experts() const noexcept { return experts_; }
        /** @brief Return exact hash constants and immutable source table descriptor. */
        const Qwen4ExpNgramGeometry &ngrams() const noexcept { return ple_; }
        /** @brief Return mixer kinds for main blocks followed by exported predictor blocks. */
        const std::vector<Qwen4ExpMixer> &mixers() const noexcept { return mixers_; }
        /** @brief Return the number of main decoder blocks, excluding predictor blocks. */
        uint32_t mainLayers() const noexcept { return main_layers_; }
        /** @brief Return the exported predictor count; zero means intentionally omitted. */
        uint32_t predictorLayers() const noexcept { return predictor_layers_; }
        /** @brief Return native maximum context length. */
        uint32_t contextLength() const noexcept { return context_length_; }
        /** @brief Return vocabulary size authenticated against embedding/head shapes. */
        uint32_t vocabularySize() const noexcept { return vocabulary_size_; }
        /** @brief Return exported norm epsilon; GGUF gains already include zero-centering. */
        float normEpsilon() const noexcept { return norm_epsilon_; }
        /** @brief Return exported rotary frequency base. */
        float rotaryBase() const noexcept { return rotary_base_; }
        /** @brief Return native interleaved mRoPE section lengths. */
        const std::array<uint32_t, 3> &rotarySections() const noexcept { return rotary_sections_; }

        // These are checkpoint generation semantics, not missing-metadata defaults.
        // Pinned generation_config.json includes both IDs; PLE resets only on 248044.
        static constexpr std::array<uint32_t, 2> generationStopTokens{248044, 248046};

    private:
        Qwen4ExpActivationGeometry activations_{};
        Qwen4ExpHyperConnectionGeometry hc_{};
        Qwen4ExpDeltaNetGeometry gdn_{};
        Qwen4ExpSparseAttentionGeometry qsa_{};
        Qwen4ExpExpertGeometry experts_{};
        Qwen4ExpNgramGeometry ple_{};
        std::vector<Qwen4ExpMixer> mixers_;
        uint32_t main_layers_ = 0;
        uint32_t predictor_layers_ = 0;
        uint32_t context_length_ = 0;
        uint32_t vocabulary_size_ = 0;
        float norm_epsilon_ = 0;
        float rotary_base_ = 0;
        std::array<uint32_t, 3> rotary_sections_{};
    };
}
