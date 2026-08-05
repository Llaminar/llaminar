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
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{

    // Forward declarations
    struct GGUFModel;
    class ModelLoader;

    struct TensorSizeInfo
    {
        std::string name;
        size_t native_bytes = 0;
        std::string quant_type; // "F32", "Q8_0", "Q4_0", "IQ4_NL", etc.
        size_t elements = 0;
        size_t K = 0;         // Inner dimension (last dim) for packed size estimation
        int layer_index = -1; // -1 for non-layer tensors (embedding, lm_head, norms)
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
        int full_attention_interval = 0;
        int gdn_conv_kernel_size = 0;
        int gdn_state_size = 0;
        int gdn_inner_size = 0;
        int gdn_group_count = 0;
        int gdn_time_step_rank = 0;

        // Weight sizing
        size_t total_native_bytes = 0;
        std::vector<TensorSizeInfo> tensors;

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
