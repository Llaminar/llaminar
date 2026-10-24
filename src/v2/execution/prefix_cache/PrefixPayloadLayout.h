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
#include "execution/mtp/MTPShiftedCacheOrigin.h"
#include "kernels/common/GDNRecurrenceStateLayout.h"
#include "kernels/qwen4exp/Qwen4ExpArchiveGeometry.h"

#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "tensors/TensorLayout.h"
#include "kernels/kvcache/CacheAuxiliaryState.h"
#include <optional>

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

    /** @brief Compact model-owned archive sections, distinct from local rollback capacity. */
    struct PrefixAuxiliaryPayloadLayout
    {
        std::optional<CacheAuxiliaryStateKind> kind;
        size_t bytes_per_attention_layer = 0; ///< Only pool rows completed in this logical block.
        size_t endpoint_bytes = 0; ///< Compact tails/histories at the immutable prefix frontier.
        bool includes_endpoint = false; ///< Endpoint sections are absent from earlier row-block reads.
        /** @return Checked block extent; zero is valid for PLE-only or incomplete pools. */
        [[nodiscard]] size_t blockBytes(int attention_layers) const;
        /** @return Exact endpoint read extent, independent of rollback capacity. */
        [[nodiscard]] size_t endpointBytes() const { return includes_endpoint ? endpoint_bytes : 0; }
        bool operator==(const PrefixAuxiliaryPayloadLayout &) const = default;
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
        GDNRecurrenceStateLayout hybrid_host_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Authenticated serialized host
                                                               ///< orientation.
        GDNRecurrenceStateLayout hybrid_device_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Authenticated serialized device
                                                               ///< orientation.
        MTPShiftedCacheOrigin mtp_origin=MTPShiftedCacheOrigin::SuccessorPairsOnly; ///< Source-authenticated cache block coordinates.
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
        PrefixAuxiliaryPayloadLayout main_auxiliary; ///< Main QSA pools and compact PLE/QSA endpoint.
        PrefixAuxiliaryPayloadLayout mtp_auxiliary; ///< Shifted predictor pools and compact endpoint.
        bool includes_hybrid_state = false;
        bool includes_mtp_state = false;
        bool includes_terminal_hidden = false;
        bool includes_terminal_logits = false;

        /** @return Every immutable serialized geometry field, without observing payload bytes. */
        bool operator==(const PrefixPayloadLayout &) const = default;

        /** @return Bytes of main attention payload, zero for recurrent-only shards. */
        size_t faKVBytes() const;
        /** @return Bytes of the participant-owned shifted predictor cache. */
        size_t mtpKVBytes() const;
        /** @return Exact compact model auxiliary section bytes. */
        [[nodiscard]] size_t modelAuxiliaryBytes() const;
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
     * @brief Resolve one valid largest QFN record layout for fixed-slot admission.
     * @param base Actual common KV/recurrent/terminal section geometry.
     * @param main Main-cache participant-local auxiliary ownership.
     * @param mtp Shifted predictor auxiliary ownership, empty for dense KV-only or absent predictors.
     * @param context_tokens Exact admitted main-cache horizon.
     * @return A maximum valid correlated record layout; never a transfer extent.
     *
     * Token starts are block boundaries and terminal counts may be partial.
     * Only modulo-four representatives are needed, including the distinct
     * first shifted block. Live records retain their own exact compact layout.
     */
    PrefixPayloadLayout qwen4ExpPrefixCapacityLayout(const PrefixPayloadLayout &base,
        qwen4exp::Qwen4ExpArchiveGeometry main, qwen4exp::Qwen4ExpArchiveGeometry mtp,
        int context_tokens);

    /**
     * @brief Project exact cache-owned immutable descriptors into the same slot contract.
     * @param base Actual common section layout, without changing its record semantics.
     * @param main Main cache that owns its auxiliary binding manifest.
     * @param mtp Optional shifted predictor cache owned by this participant.
     * @return Largest valid layout for these caches, or base for ordinary models.
     */
    PrefixPayloadLayout prefixPayloadCapacityLayout(const PrefixPayloadLayout &base,
        const IKVCache &main, const IKVCache *mtp);

    /**
     * @brief Resolve live auxiliary sections from immutable logical archive coordinates.
     * @param layout Existing attention/GDN/MTP geometry to extend without allocation.
     * @param main_cache Canonical main state owner.
     * @param mtp_cache Optional canonical shifted predictor state owner.
     * @param sequence Cache request slot, independent of graph row capacity.
     * @param first_token First logical token represented by the block.
     * @param token_count Actual block rows, never allocation padding.
     * @param logical_prefix Exact immutable endpoint frontier.
     * @param include_endpoint Whether this record owns the selected prefix endpoint.
     */
    void applyPrefixAuxiliaryLayout(PrefixPayloadLayout &layout, const IKVCache &main_cache,
        const IKVCache *mtp_cache, int sequence, int first_token, int token_count,
        int logical_prefix, bool include_endpoint);

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
