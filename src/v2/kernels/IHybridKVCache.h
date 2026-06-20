/**
 * @file IHybridKVCache.h
 * @brief Interface for hybrid KV caches that combine FA and GDN layer state
 *
 * Extends IKVCache with GDN state access methods. Stages can dynamic_cast
 * an IKVCache* to IHybridKVCache* to access per-layer GDN state:
 *
 *   auto* hybrid = dynamic_cast<IHybridKVCache*>(kv_cache);
 *   if (hybrid && hybrid->isGDNLayer(layer_idx)) {
 *       float* conv_state = hybrid->getConvState(layer_idx);
 *       ITensorShortConvolution* kernel = hybrid->getConvKernel(layer_idx);
 *       ...
 *   }
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    class ITensorShortConvolution;
    class ITensorGatedDeltaNet;
    struct HybridGDNLayerState;

    struct HybridPrefixStateMetadata
    {
        int total_layers = 0;
        int gdn_layers = 0;
        size_t host_bytes = 0;
        size_t device_bytes = 0;
        bool has_device_kernel_state = false;
    };

    struct HybridPrefixStateDescriptor
    {
        int seq_idx = 0;
        int logical_token_count = 0;
        void *stream = nullptr;
        // Prefix-cache persistence needs completed payloads on return; live rollback
        // checkpoints can use stream ordering and avoid a host-side sync.
        bool synchronize = true;
        bool include_host_state = true;
        bool include_device_state = true;
    };

    /**
     * @brief Extended KV cache interface with GDN state management
     *
     * Implemented by CPUHybridRingKVCache, CUDAHybridRingKVCache,
     * ROCmHybridRingKVCache. Provides per-layer GDN state access
     * for hybrid models (e.g., Qwen 3.5) that mix full-attention
     * and GDN layers.
     */
    class IHybridKVCache
    {
    public:
        virtual ~IHybridKVCache() = default;

        // =====================================================================
        // Layer Type Queries
        // =====================================================================

        /// Check if a layer is a GDN (linear attention) layer
        virtual bool isGDNLayer(int layer) const = 0;

        /// Check if a layer is a full-attention layer
        virtual bool isFullAttentionLayer(int layer) const = 0;

        /// Number of KV cache (full-attention) layers
        virtual int kvLayerCount() const = 0;

        /// Number of GDN layers
        virtual int gdnLayerCount() const = 0;

        // =====================================================================
        // GDN State Access
        // =====================================================================

        /// Get full GDN state for a layer (nullptr if FA layer)
        virtual HybridGDNLayerState *getGDNState(int layer) = 0;
        virtual const HybridGDNLayerState *getGDNState(int layer) const = 0;

        /// Get mutable recurrence state [n_v_heads, d_k, d_v] (nullptr if FA)
        virtual float *getRecurrenceState(int layer) = 0;

        /// Get mutable conv state [qkv_dim, conv_kernel-1] (nullptr if FA)
        virtual float *getConvState(int layer) = 0;

        // =====================================================================
        // GDN Kernel Access
        // =====================================================================

        /// Get short convolution kernel for a GDN layer (nullptr if FA)
        virtual ITensorShortConvolution *getConvKernel(int layer) = 0;

        /// Get gated delta net recurrence kernel for a GDN layer (nullptr if FA)
        virtual ITensorGatedDeltaNet *getRecurrenceKernel(int layer) = 0;

        // =====================================================================
        // GDN Lifecycle
        // =====================================================================

        /// Reset all GDN states to zero (for new sequence)
        virtual void resetGDNStates() = 0;

        /// Total GDN state memory in bytes
        virtual size_t gdnMemoryBytes() const = 0;

        // =====================================================================
        // Prefix/Rollback State
        // =====================================================================

        virtual HybridPrefixStateMetadata hybridPrefixStateMetadata() const = 0;

        virtual bool exportHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            void *dst_host,
            void *dst_device) const = 0;

        virtual bool importHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            const void *src_host,
            const void *src_device) = 0;
    };

} // namespace llaminar2
