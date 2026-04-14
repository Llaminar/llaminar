/**
 * @file HybridKVCacheConfig.h
 * @brief Configuration and GDN state for hybrid KV caches
 *
 * Hybrid KV caches are specialized subclasses of the standard ring buffer
 * KV caches (CPU, CUDA, ROCm) that support models with heterogeneous layer
 * types, such as Qwen 3.5 which mixes full-attention (FA) layers with
 * Gated Delta Network (GDN) layers.
 *
 * Key design:
 * - FA layers use the standard KV cache (inherited from the parent class)
 * - GDN layers store fixed-size recurrence + conv state (O(1) per layer)
 * - The parent cache is constructed with only the FA layer count, saving
 *   memory by not allocating KV entries for GDN layers
 * - All layer-indexed methods remap global layer indices to compressed
 *   KV cache indices for FA layers
 *
 * @see CPUHybridRingKVCache, CUDAHybridRingKVCache, ROCmHybridRingKVCache
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../utils/Logger.h"

namespace llaminar2
{
    class ITensorShortConvolution;
    class ITensorGatedDeltaNet;

    /**
     * @brief Per-layer GDN state (recurrence + short conv) for hybrid caches
     *
     * This state is fixed-size O(1) per layer regardless of sequence length:
     * - Recurrence state: delta-rule S matrix, updated in-place each step
     * - Conv state: causal convolution sliding window of (kernel-1) tokens
     *
     * Always stored in FP32. The --kv-cache-precision flag has no effect
     * on GDN state precision.
     */
    struct HybridGDNLayerState
    {
        int n_v_heads = 0; ///< Value head count (recurrence runs with this)
        int n_k_heads = 0; ///< Key head count (for QKV split)
        int d_k = 0;       ///< Key/query dimension per head
        int d_v = 0;       ///< Value dimension per head
        int conv_kernel_size = 0;

        /// Recurrence state S: [n_v_heads, d_k, d_v] (FP32)
        std::vector<float> recurrence_state;

        /// Short convolution state: [qkv_dim, conv_kernel-1] (FP32)
        std::vector<float> conv_state;

        /// Kernel instances for this layer (owned by the cache)
        std::shared_ptr<ITensorShortConvolution> conv_kernel;
        std::shared_ptr<ITensorGatedDeltaNet> rec_kernel;

        /// Initialize (zero-fill) all state
        void initialize(int qkv_dim)
        {
            const size_t s_size = static_cast<size_t>(n_v_heads) *
                                  static_cast<size_t>(d_k) *
                                  static_cast<size_t>(d_v);
            recurrence_state.assign(s_size, 0.0f);

            if (conv_kernel_size > 1)
            {
                const size_t c_size = static_cast<size_t>(qkv_dim) *
                                      static_cast<size_t>(conv_kernel_size - 1);
                conv_state.assign(c_size, 0.0f);
            }
        }

        /// Reset host-side state to zero (for new sequence)
        /// NOTE: GPU kernel state must be reset separately via resetGPUKernelState()
        void reset()
        {
            std::fill(recurrence_state.begin(), recurrence_state.end(), 0.0f);
            std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        }

        /// Reset GPU-resident kernel state (call after reset() for GPU backends)
        /// Requires full ITensorShortConvolution/ITensorGatedDeltaNet definitions.
        void resetGPUKernelState();

        /// Total memory in bytes
        size_t memoryBytes() const
        {
            return (recurrence_state.size() + conv_state.size()) * sizeof(float);
        }
    };

    /**
     * @brief Configuration for creating a hybrid KV cache
     *
     * Extends the standard KVCacheConfig with layer type information and
     * GDN state sizing parameters.
     */
    struct HybridKVCacheConfig
    {
        /// Per-layer type: "full_attention" or "gdn"
        /// Size must equal total n_layers.
        std::vector<std::string> layer_types;

        // GDN state sizing (from GGUF ssm.* metadata)
        int gdn_conv_kernel_size = 0; ///< Short conv kernel width (e.g. 4)
        int gdn_state_size = 0;       ///< Recurrence state dim (d_k = d_v)
        int gdn_inner_size = 0;       ///< Inner projection size
        int gdn_group_count = 0;      ///< Key head count (ssm.group_count)
        int gdn_time_step_rank = 0;   ///< Value head count (ssm.time_step_rank)

        /// TP-aware local head count (0 = use full n_heads, no sharding)
        int local_n_heads = 0;
        int n_heads = 0; ///< Total attention head count

        /// Device for GDN kernel instances (convolution + recurrence)
        int device_type = 0;    ///< KernelFactory device type enum
        int device_ordinal = 0; ///< Device index within the type

        /// Returns true if this config represents a hybrid model
        bool isHybrid() const { return !layer_types.empty(); }

        /// Count the number of full-attention layers
        int countKVLayers() const
        {
            int count = 0;
            for (const auto &t : layer_types)
            {
                if (t == "full_attention")
                    ++count;
            }
            return count;
        }
    };

    /**
     * @brief Layer mapping utility for hybrid KV caches
     *
     * Provides bidirectional mapping between global layer indices (0..n_layers-1)
     * and compressed KV cache layer indices (0..kv_layer_count-1).
     *
     * Example for Qwen 3.5 4B with 36 layers, full_attention_interval=4:
     *   Global:  0  1  2  3  4  5  6  7  8  ...  35
     *   Type:    G  G  G  FA G  G  G  FA G  ...  FA
     *   KV idx:  -  -  -  0  -  -  -  1  -  ...  8
     *   GDN idx: 0  1  2  -  3  4  5  -  6  ...  -
     */
    class HybridLayerMap
    {
    public:
        HybridLayerMap() = default;

        /**
         * @brief Build the layer mapping from layer type strings
         * @param layer_types Per-layer type ("full_attention" or "gdn")
         */
        void build(const std::vector<std::string> &layer_types)
        {
            const int n = static_cast<int>(layer_types.size());
            global_to_kv_.resize(n, -1);
            global_to_gdn_.resize(n, -1);
            is_fa_.resize(n, false);

            int kv_idx = 0;
            int gdn_idx = 0;
            for (int i = 0; i < n; ++i)
            {
                if (layer_types[i] == "full_attention")
                {
                    global_to_kv_[i] = kv_idx++;
                    is_fa_[i] = true;
                }
                else
                {
                    global_to_gdn_[i] = gdn_idx++;
                }
            }
            n_total_ = n;
            n_kv_ = kv_idx;
            n_gdn_ = gdn_idx;

            LOG_DEBUG("[HybridLayerMap] Built mapping: " << n_total_ << " total layers, "
                                                         << n_kv_ << " KV (FA), " << n_gdn_ << " GDN");
        }

        /// Map global layer index to KV cache index (-1 if GDN layer)
        int toKVIndex(int global_layer) const
        {
            if (global_layer < 0 || global_layer >= n_total_)
                return -1;
            return global_to_kv_[global_layer];
        }

        /// Map global layer index to GDN state index (-1 if FA layer)
        int toGDNIndex(int global_layer) const
        {
            if (global_layer < 0 || global_layer >= n_total_)
                return -1;
            return global_to_gdn_[global_layer];
        }

        /// Check if a global layer is a full-attention layer
        bool isFullAttention(int global_layer) const
        {
            if (global_layer < 0 || global_layer >= n_total_)
                return false;
            return is_fa_[global_layer];
        }

        int totalLayers() const { return n_total_; }
        int kvLayerCount() const { return n_kv_; }
        int gdnLayerCount() const { return n_gdn_; }

    private:
        std::vector<int> global_to_kv_;  ///< Global layer → KV cache index (-1 for GDN)
        std::vector<int> global_to_gdn_; ///< Global layer → GDN state index (-1 for FA)
        std::vector<bool> is_fa_;        ///< True if full-attention layer
        int n_total_ = 0;
        int n_kv_ = 0;
        int n_gdn_ = 0;
    };

} // namespace llaminar2
