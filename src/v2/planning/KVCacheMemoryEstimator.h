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

    /** @brief Exact backend-aware persistent KV-cache memory estimator. */
    class KVCacheMemoryEstimator
    {
    public:
        /**
         * @brief Estimate persistent KV-cache ownership for one device.
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
            int n_layers,
            int batch_size,
            int max_seq_len,
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
