/**
 * @file ModelMemoryProfile.h
 * @brief Declares the compact, serializable model geometry used by memory planning.
 *
 * A ModelMemoryProfile contains only metadata and tensor inventory facts. It
 * deliberately owns no loaded weights, runtime graph objects, or backend
 * handles, which allows rank zero to parse GGUF once and broadcast an exact
 * planning description to every participant before device allocation begins.
 */

#pragma once
#include "models/qwen4exp/Qwen4ExpPredictorPolicy.h"
#include "loaders/SharedPredictorGGUFSource.h"
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace llaminar2
{

    // Forward declarations
    struct GGUFModel;
    class ModelLoader;
    class Qwen4ExpConfig;

    struct TensorSizeInfo
    {
        std::string name;
        size_t native_bytes = 0;
        std::string quant_type; // "F32", "Q8_0", "Q4_0", "IQ4_NL", etc.
        size_t elements = 0;
        /**
         * @brief Inner dimension of one logical matrix.
         *
         * This is the normalized second axis for ordinary 2-D weights and the
         * first GGUF axis for a 3-D routed-expert parent `[K, N, experts]`.
         * It is deliberately not just `dimensions.back()`: that axis is the
         * expert count for routed parents.
         */
        size_t K = 0;
        int layer_index = -1; // -1 for non-layer tensors (embedding, lm_head, norms)
    };

    /**
     * @brief Exact HC/QSA/PLE allocation inputs retained without runtime state.
     *
     * This is a typed metadata BOM contribution, not live memory accounting.
     * Table source capacity is separate from ordinary staged model weights.
     * Request history sizes retain token/channel geometry so every allocator
     * and archive format can derive its own admitted representation exactly.
     */
    struct Qwen4ExpMemoryGeometry
    {
        uint32_t residual_width = 0;
        uint32_t terminal_head_width = 0;
        uint32_t residual_branches = 0;
        uint32_t residual_gate_rank = 0;
        uint32_t indexer_query_heads = 0;
        uint32_t indexer_key_heads = 0;
        uint32_t indexer_head_dimension = 0;
        uint32_t compression_ratio = 0;
        uint32_t selected_token_budget = 0;
        uint32_t maximum_selected_tokens = 0;
        uint32_t ple_decoder_layer = 0;
        uint32_t ple_heads = 0;
        uint32_t ple_row_elements = 0;
        uint32_t ple_convolution_kernel = 0;
        uint32_t ple_convolution_dilation = 0;
        uint32_t ple_convolution_history = 0;
        uint32_t ple_hash_history_tokens = 0;
        uint32_t ple_table_gguf_type = 0;
        uint64_t ple_table_rows = 0;
        uint64_t ple_table_row_bytes = 0;
        uint64_t ple_table_native_bytes = 0;
        /** Original immutable head metadata; these are source facts, not derived storage ranges. */
        std::vector<std::uint64_t> ple_head_offsets;
        std::vector<std::uint64_t> ple_head_vocab_sizes;
        /** Main and exported predictor sparse layers in canonical block order. */
        std::vector<uint32_t> sparse_decoder_layers;
        /** @brief Project the validated immutable model config into this metadata-only contract. */
        [[nodiscard]] static Qwen4ExpMemoryGeometry fromConfig(const Qwen4ExpConfig &config);
        bool operator==(const Qwen4ExpMemoryGeometry &) const = default;
    };

    /**
     * @brief Backend-neutral model geometry and native-weight inventory.
     *
     * Every field is sourced from GGUF metadata or the GGUF tensor directory.
     * Memory estimators use this structure before graph construction, so any
     * geometry that changes a persistent or workspace allocation belongs here
     * rather than in a late graph-specific heuristic.
     */
    struct ModelMemoryProfile
    {
        // Architecture
        std::string architecture;
        int n_layers = 0;
        int d_model = 0;
        int d_ff = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int vocab_size = 0;
        int max_seq_len = 0;

        /** Validated Flash Next metadata; absent for every other model family. */
        std::optional<Qwen4ExpMemoryGeometry> flash_next;

        /** @brief Number of routed experts addressable by each MoE router. */
        int expert_count = 0;
        /** @brief Number of routed experts selected for each token. */
        int expert_used_count = 0;
        /** @brief Intermediate width of one routed expert FFN. */
        int expert_feed_forward_length = 0;
        /** @brief Intermediate width of the optional shared expert FFN. */
        int expert_shared_feed_forward_length = 0;

        /**
         * @brief Hybrid recurrent and MTP geometry copied from GGUF metadata.
         *
         * Keeping these values in the compact profile lets rank-local memory
         * planning reproduce the same cache construction performed later by
         * Qwen35GraphConfigBuilder without loading model tensors on every rank.
         */
        int mtp_layer_count = 0;
        /** @return Main decoder blocks under the architecture's explicit GGUF block-count contract. */
        [[nodiscard]] int mainLayerCount() const;
        /** @return Main plus separately declared trailing predictor blocks, without double counting. */
        [[nodiscard]] int totalLayerCount() const;
        int full_attention_interval = 0;
        int gdn_conv_kernel_size = 0;
        int gdn_state_size = 0;
        int gdn_inner_size = 0;
        int gdn_group_count = 0;
        int gdn_time_step_rank = 0;

        // Weight sizing
        size_t total_native_bytes = 0;
        std::vector<TensorSizeInfo> tensors;

        /** Source-authenticated predictor policy; presence is mandatory for a QFN learned block. */
        std::optional<Qwen4ExpPredictorMetadata> flash_next_predictor;

        /** @return Number of QSA layers in an inclusive interval; dense predictors contribute zero. */
        size_t flashNextSparseLayerCount(int first_layer, int last_layer) const;

        /**
         * @brief Resolve exact main or authenticated predictor state identity.
         * @param layer Original global source layer, including an admitted trailing predictor.
         * @return GDN, QSA or dense attention; predictors never inherit absence-from-sparse-list defaults.
         * @throws std::invalid_argument If geometry, source policy or layer index is invalid.
         */
        Qwen4ExpLayerKind flashNextLayerKind(int layer) const;

        /**
         * @brief Authenticate original main and separate predictor directories and project active native roles.
         * @param main Unmodified complete target directory.
         * @param predictor Unmodified explicit shared predictor directory.
         * @param materialization Active predictor roles or metadata-only source identity.
         * @return Planning metadata with unchanged main count and typed predictor policy.
         */
        static ModelMemoryProfile fromGGUF(const GGUFModel &main, const GGUFModel &predictor,
            SharedPredictorMaterialization materialization);

        /**
         * @brief Project one sealed loader source set into graph-consumed native BOM inventory.
         * @param loader Exact target and optional authenticated shared predictor.
         * @return Metadata with unchanged main boundaries and explicit predictor policy.
         */
        static ModelMemoryProfile fromLoader(const ModelLoader &loader);

        /**
         * @brief Build an exact planning profile from parsed GGUF metadata.
         * @param model Parsed GGUF model header and tensor inventory.
         * @return A self-contained profile suitable for local planning or MPI broadcast.
         */
        static ModelMemoryProfile fromGGUF(const GGUFModel &model);

        // Query helpers
        size_t weightBytesForLayers(int first_layer, int last_layer) const;
        size_t embeddingBytes() const;
        size_t lmHeadBytes() const;
        size_t normBytes() const;
        size_t layerWeightBytes(int layer) const;

        /**
         * @brief Serialize this profile using the current versioned wire format.
         * @return Byte buffer suitable for MPI broadcast or explicit plan storage.
         */
        std::vector<uint8_t> serialize() const;

        /**
         * @brief Deserialize one current-version profile.
         * @param data First byte of the serialized profile.
         * @param size Number of available bytes.
         * @return Reconstructed profile.
         * @throws std::runtime_error for truncation, bad magic, or stale versions.
         */
        static ModelMemoryProfile deserialize(const uint8_t *data, size_t size);
    };

} // namespace llaminar2
