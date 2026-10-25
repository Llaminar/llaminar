/**
 * @file HybridCacheManager.h
 * @brief Dual KV/GDN state manager for heterogeneous layer architectures
 *
 * Models like Qwen 3.5 mix full-attention layers (which need KV cache)
 * with GDN layers (which need recurrence state + conv state). This
 * manager provides a unified interface for per-layer cache allocation
 * and state management.
 *
 * Key design:
 * - Each layer is tagged as either KV_CACHE or GDN_STATE
 * - KV cache layers delegate to the existing IKVCache implementation
 * - GDN state layers own complete CPU checkpoint/history + conv state tensors
 * - The manager does NOT own the IKVCache; it wraps one provided externally
 */

#pragma once

#include "IHybridCacheManager.h"
#include "kernels/CPUGDNStateLayout.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class IKVCache;

    /**
     * @brief Layer state type for heterogeneous models
     */
    enum class LayerStateType
    {
        KV_CACHE,  ///< Full attention — needs standard KV cache
        GDN_STATE, ///< Gated Delta Network — needs recurrence + conv state
        NONE       ///< No persistent state (e.g., embedding-only)
    };

    /**
     * @brief Per-layer GDN state (recurrence + short conv)
     *
     * This is small and persistent across decode steps:
     * - Recurrence state: complete fixed-epoch CPU checkpoint and raw history
     * - Conv state: [n_heads, conv_kernel-1, head_dim] (causal conv history)
     */
    struct GDNLayerState
    {
        int n_heads = 0;
        int head_dim = 0;
        int conv_kernel_size = 0;

        /// Complete CPUGDNStateLayout bank; snapshots never flush pending history.
        std::vector<float> recurrence_state;

        /// Short convolution state: [n_heads, (conv_kernel-1), head_dim] (FP32)
        std::vector<float> conv_state;

        /// Initialize (zero-fill) all state
        void initialize()
        {
            const size_t s_size = CPUGDNStateLayout(n_heads,head_dim,head_dim).floats();
            recurrence_state.assign(s_size, 0.0f);

            if (conv_kernel_size > 1)
            {
                const size_t c_size = static_cast<size_t>(n_heads) *
                                      static_cast<size_t>(conv_kernel_size - 1) *
                                      static_cast<size_t>(head_dim);
                conv_state.assign(c_size, 0.0f);
            }
        }

        /// Reset state to zero (for new sequence)
        void reset()
        {
            std::fill(recurrence_state.begin(), recurrence_state.end(), 0.0f);
            std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        }

        /// Total memory in bytes
        size_t memoryBytes() const
        {
            return (recurrence_state.size() + conv_state.size()) * sizeof(float);
        }
    };

    /**
     * @brief Unified cache manager for heterogeneous layer architectures
     *
     * Wraps an IKVCache for full-attention layers and manages GDN state
     * for GDN layers. Layer type assignment is fixed at construction time
     * based on the model's layer_types configuration.
     */
    class HybridCacheManager : public IHybridCacheManager
    {
    public:
        /**
         * @brief Configuration for the hybrid cache
         */
        struct Config
        {
            int n_layers = 0;                    ///< Total number of layers
            std::vector<std::string> layer_types; ///< Per-layer type ("full_attention", "gdn")
            int n_heads = 0;                     ///< Attention heads (for GDN state sizing)
            int head_dim = 0;                    ///< Head dimension
            int conv_kernel_size = 0;            ///< GDN conv kernel width (e.g., 4)
        };

        /**
         * @brief Construct with configuration.
         * @param config Layer type and dimension configuration
         * @param kv_cache External KV cache for full-attention layers (not owned)
         */
        HybridCacheManager(Config config, IKVCache *kv_cache);

        // -- IHybridCacheManager interface --

        LayerStateType getLayerStateType(int layer_idx) const override;
        IKVCache *kvCache() const override { return kv_cache_; }
        GDNLayerState *getGDNState(int layer_idx) override;
        const GDNLayerState *getGDNState(int layer_idx) const override;
        void reset() override;
        size_t gdnMemoryBytes() const override;
        int kvCacheLayerCount() const override;
        int gdnLayerCount() const override;
        int totalLayers() const override { return static_cast<int>(layer_state_types_.size()); }
        int toKVCacheLayerIdx(int layer_idx) const override;

    private:
        IKVCache *kv_cache_ = nullptr;
        std::vector<LayerStateType> layer_state_types_;
        std::vector<GDNLayerState> gdn_states_;
        std::vector<int> gdn_layer_map_;  ///< layer_idx → gdn_states_ index (-1 if not GDN)
        std::vector<int> kv_layer_map_;   ///< layer_idx → KV cache layer index (-1 if not KV)
    };

} // namespace llaminar2
