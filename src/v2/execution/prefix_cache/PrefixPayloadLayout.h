/**
 * @file PrefixPayloadLayout.h
 * @brief Immutable payload geometry and archive organization for one participant.
 *
 * Attention caches need an ordered chain of token blocks. A participant with
 * only recurrent state needs one complete checkpoint at a known prompt frontier,
 * not synthetic K/V storage or empty ancestor payloads. Geometry comes from
 * the live cache owners and never creates a parallel physical-memory ledger.
 */
#pragma once

#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "tensors/TensorLayout.h"

#include <cstddef>

namespace llaminar2
{
    class IKVCache;

    /** @brief Which archived records are necessary to reconstruct a prefix. */
    enum class PrefixPayloadOrganization
    {
        AttentionBlocks,    ///< Main or shifted attention needs every preceding block.
        RecurrentCheckpoint ///< One complete recurrent image covers its entire prefix.
    };

    /** @brief Canonical section sizes shared by archive producers and consumers. */
    struct PrefixPayloadLayout
    {
        DeviceId device = DeviceId::cpu();
        int block_size = 64;
        int first_layer_index = 0;
        int total_layers = 0;
        int fa_layers = 0;
        int gdn_layers = 0;
        int local_kv_heads = 0;
        int kv_head_start = 0;
        int head_dim = 0;
        ActivationPrecision k_precision = ActivationPrecision::FP32;
        ActivationPrecision v_precision = ActivationPrecision::FP32;
        TensorLayout kv_layout = TensorLayout::KV_POS_HEAD_DIM;
        size_t bytes_per_fa_layer_k = 0;
        size_t bytes_per_fa_layer_v = 0;
        size_t hybrid_host_state_bytes = 0;
        size_t hybrid_device_state_bytes = 0;
        size_t hybrid_state_bytes = 0;
        int mtp_layers = 0;
        int mtp_local_kv_heads = 0;
        int mtp_kv_head_start = 0;
        int mtp_head_dim = 0;
        ActivationPrecision mtp_k_precision = ActivationPrecision::FP32;
        ActivationPrecision mtp_v_precision = ActivationPrecision::FP32;
        TensorLayout mtp_kv_layout = TensorLayout::KV_POS_HEAD_DIM;
        size_t bytes_per_mtp_layer_k = 0;
        size_t bytes_per_mtp_layer_v = 0;
        size_t mtp_kv_bytes = 0;
        size_t terminal_hidden_bytes = 0;
        size_t terminal_logits_bytes = 0;
        bool includes_hybrid_state = false;
        bool includes_mtp_state = false;
        bool includes_terminal_hidden = false;
        bool includes_terminal_logits = false;

        /** @return Bytes of main attention payload, zero for recurrent-only shards. */
        size_t faKVBytes() const;
        /** @return Bytes of the participant-owned shifted predictor cache. */
        size_t mtpKVBytes() const;
        /** @return Total bytes of sections included by this particular record. */
        size_t totalBytes() const;
        /** @return Whether the main cache exposes all state required for restore. */
        [[nodiscard]] bool hasRestorableMainState() const;
        /** @return Archive organization derived from main and shifted cache ownership. */
        [[nodiscard]] PrefixPayloadOrganization organization() const;
        /**
         * @brief Compare immutable shape while allowing omitted terminal-only sections.
         * @param other Runtime or record layout whose geometry must match.
         * @return True when records can belong to the same authenticated prefix chain.
         */
        bool compatiblePayloadShape(const PrefixPayloadLayout &other) const;
    };

    /**
     * @brief Project actual attention/recurrent cache owners into serialized sections.
     * @param kv_cache Main or shifted cache whose physical geometry is authoritative.
     * @param device Participant that produces the archive.
     * @param block_size Logical token width of attention blocks and hash-chain chunks.
     * @param terminal_hidden_bytes Hidden row owned by a terminal MTP participant.
     * @param terminal_logits_bytes Logit row owned by a terminal participant.
     * @return Layout without allocating storage or observing device execution state.
     */
    PrefixPayloadLayout buildDensePrefixPayloadLayout(
        const IKVCache &kv_cache,
        DeviceId device,
        int block_size,
        size_t terminal_hidden_bytes = 0,
        size_t terminal_logits_bytes = 0);

    /**
     * @brief Resolve a payload's dense attention ordinal into its global model layer.
     * @param kv_cache Participant-local cache with a possibly nonzero PP offset.
     * @param fa_index Zero-based ordinal among this participant's attention layers.
     * @return Global model layer, or -1 for an out-of-range attention ordinal.
     * All cache queries use global identity; local offsets can overlap that
     * namespace and must never be passed to a hybrid cache as model indices.
     */
    int prefixFALayerForIndex(const IKVCache &kv_cache, int fa_index);

    /** @return Global first attention layer, or the first layer for a GDN-only shard. */
    int firstRestorablePrefixLayer(const IKVCache &kv_cache);

    /** @return Host-visible cached count at an explicitly diagnostic/CPU cache boundary. */
    int restorablePrefixCachedTokens(const IKVCache &kv_cache, int seq_idx = 0);

} // namespace llaminar2
