/**
 * @file TurboQuantContext.h
 * @brief Lifecycle management for TurboQuant random matrices
 *
 * Created once at model load time, held by the graph orchestrator, and
 * passed to KVCacheAppendStage (quantize) and attention stages (dequantize)
 * via stage params.
 *
 * The decorrelating rotation is deterministic, so any process with the same
 * head_dim and seed produces the same quantize/dequantize behavior.
 */

#pragma once

#include "TurboQuantRotation.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace llaminar2
{

    /**
     * @brief Holds the TurboQuant rotation matrix for a model's head_dim.
     *
     * Usage:
     *   // At model load time (DeviceGraphOrchestrator):
     *   auto tq_ctx = std::make_shared<TurboQuantContext>(head_dim);
     *
     *   // Pass to stages:
     *   kv_append_params.turboquant_ctx = tq_ctx.get();
     *   attention_params.turboquant_ctx = tq_ctx.get();
     */
    class TurboQuantContext
    {
    public:
        /**
         * @brief Construct TurboQuantContext for the given head dimension.
         */
        explicit TurboQuantContext(int head_dim,
                                   uint64_t rotation_seed = 0,
                                   [[maybe_unused]] uint64_t projection_seed = 0)
            : rotation_seed_(rotation_seed == 0 ? 31ULL : rotation_seed),
              rotation_(generate_rotation_matrix(head_dim, rotation_seed_)),
              head_dim_(head_dim)
        {
        }

        /// Access the decorrelating rotation matrix
        const TurboQuantRotation &rotation() const { return rotation_; }

        /// Head dimension this context was created for
        int head_dim() const { return head_dim_; }

        /// Deterministically derive and cache an independent context per layer.
        const TurboQuantContext &for_layer(int layer_idx) const
        {
            if (layer_idx < 0)
                return *this;

            std::lock_guard<std::mutex> lock(derived_mutex_);
            return get_or_create_layer_unlocked(layer_idx);
        }

        /**
         * @brief Resolve several derived contexts while taking the cache lock once.
         *
         * TurboQuant KV kernels reuse one deterministic rotation per KV head.
         * Resolving that context inside a `(row, head)` loop used to acquire the
         * same mutex once per head per row, obscuring otherwise useful SIMD work.
         * This bulk API copies stable context pointers into caller-owned storage
         * under one lock. The caller can then execute a parallel hot loop without
         * synchronization or temporary allocation.
         *
         * @param count Number of consecutive contexts, starting at index zero.
         * @param[out] out Caller-owned pointer array.
         * @param capacity Number of entries available in `out`.
         * @throws std::invalid_argument for an invalid output contract.
         */
        void resolve_layer_contexts(
            int count,
            const TurboQuantContext **out,
            size_t capacity) const
        {
            if (count < 0 || !out || capacity < static_cast<size_t>(count))
            {
                throw std::invalid_argument(
                    "TurboQuantContext::resolve_layer_contexts: invalid output contract");
            }

            std::lock_guard<std::mutex> lock(derived_mutex_);
            for (int index = 0; index < count; ++index)
            {
                out[index] = &get_or_create_layer_unlocked(index);
            }
        }

    private:
        /**
         * @brief Resolve one derived context while `derived_mutex_` is held.
         */
        const TurboQuantContext &get_or_create_layer_unlocked(int layer_idx) const
        {
            auto it = derived_contexts_.find(layer_idx);
            if (it != derived_contexts_.end())
                return *it->second;

            const uint64_t rot_seed =
                mix_seed(rotation_seed_, static_cast<uint64_t>(layer_idx) + 1ULL);
            auto derived = std::make_shared<TurboQuantContext>(head_dim_, rot_seed);
            const TurboQuantContext &ref = *derived;
            derived_contexts_.emplace(layer_idx, std::move(derived));
            return ref;
        }

        static uint64_t mix_seed(uint64_t base, uint64_t salt)
        {
            uint64_t z = base + 0x9E3779B97F4A7C15ULL + (salt << 6) + (salt >> 2);
            z ^= salt + 0x9E3779B97F4A7C15ULL + (z << 6) + (z >> 2);
            z ^= (z >> 30);
            z *= 0xBF58476D1CE4E5B9ULL;
            z ^= (z >> 27);
            z *= 0x94D049BB133111EBULL;
            z ^= (z >> 31);
            return z ? z : 1ULL;
        }

        uint64_t rotation_seed_;
        TurboQuantRotation rotation_;
        int head_dim_;
        mutable std::mutex derived_mutex_;
        mutable std::unordered_map<int, std::shared_ptr<TurboQuantContext>> derived_contexts_;
    };

} // namespace llaminar2
