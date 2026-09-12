/**
 * @file KVCacheMemoryEstimator.h
 * @brief Persistent KV-cache allocation contract used by memory admission.
 *
 * This estimator mirrors backend cache ownership and deliberately excludes
 * graph workspace, which is a separate MemoryPlanner line item. Unsupported
 * precision/backend combinations fail closed instead of assuming FP16.
 */

#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <string>

namespace llaminar2
{
    /**
     * @brief Concrete cache construction family, independent of storage precision.
     *
     * GPU hybrid caches currently store linear Q8_1 K/V, whereas attention-only
     * Q8 caches store anchored AQ8 keys. CPU caches use anchored keys in both
     * families. Every admission caller must name the family explicitly: a dtype
     * alone is not a complete physical allocation or serialization contract.
     */
    enum class KVCacheFamily
    {
        AttentionOnly, ///< Ordinary attention and the shifted MTP sidecar.
        Hybrid,        ///< Main cache with the model's FA/GDN layer mapping.
    };

    /** @brief Native packed bytes for one logical GPU prefix-cache layer. */
    struct GPULogicalKVBlockEstimate
    {
        /** Key payload, including the immutable AQ8 anchor when compressed. */
        std::size_t k_bytes = 0;
        /** Value payload in the configured native cache codec. */
        std::size_t v_bytes = 0;

        /** @return Combined K/V payload bytes. */
        [[nodiscard]] std::size_t totalBytes() const noexcept
        {
            return k_bytes + v_bytes;
        }
    };

    /** @brief Exact backend-aware persistent KV-cache memory estimator. */
    class KVCacheMemoryEstimator
    {
    public:
        /**
         * @brief Estimate persistent KV-cache ownership for one device.
         * @param family Concrete cache family selected by production construction.
         * @param n_layers Full-attention layers resident on this device.
         * @param batch_size Maximum concurrent request slots.
         * @param max_seq_len Ring horizon per request.
         * @param n_kv_heads Local KV heads after tensor sharding.
         * @param head_dim Coordinates in one KV head.
         * @param kv_precision Canonical `fp32`, `bf16`, `fp16`, `q8_1`,
         *        `q16_1`, `tq4`, or `tq` storage token.
         * @param device Concrete CPU, CUDA, or ROCm owner.
         * @return Exact requested bytes for persistent cache allocations.
         * @throws std::invalid_argument for unsupported formats, devices, or
         *         format/backend geometry.
         * @throws std::overflow_error when the requested geometry cannot be
         *         represented by `size_t`.
         */
        static std::size_t estimate(
            KVCacheFamily family,
            int n_layers,
            int batch_size,
            int max_seq_len,
            int n_kv_heads,
            int head_dim,
            const std::string &kv_precision,
            DeviceId device);

        /**
         * @brief Estimate one GPU cache layer's canonical archive payload.
         *
         * This mirrors `CUDARingKVCache::logicalBlockLayout()` and its ROCm
         * counterpart, including the per-block AQ8 key anchor used by Q8/TQ
         * caches. Metadata, rotation matrices, and live ring storage are not
         * part of a serialized prefix payload and remain in @ref estimate.
         *
         * @param token_count Tokens serialized in the prefix block.
         * @param family Concrete cache family, not merely the model's family.
         * @param n_kv_heads Participant-local KV head count.
         * @param head_dim Coordinates per head.
         * @param kv_precision Canonical runtime storage precision.
         * @param device Exact CUDA or ROCm owner.
         * @return Exact K and V byte counts for one full-attention layer.
         * @throws std::invalid_argument for CPU, unsupported codecs, or
         *         invalid positive geometry.
         * @throws std::overflow_error when byte arithmetic overflows.
         */
        static GPULogicalKVBlockEstimate estimateGPULogicalBlock(
            KVCacheFamily family,
            int token_count,
            int n_kv_heads,
            int head_dim,
            const std::string &kv_precision,
            DeviceId device);

        /**
         * @brief Return codec bytes/element when that value is geometry-free.
         *
         * TurboQuant formats are head/backend dependent and therefore reject
         * this scalar query; callers needing admission bytes use estimate().
         */
        static float getBytesPerElement(
            const std::string &kv_precision);
    };

} // namespace llaminar2
