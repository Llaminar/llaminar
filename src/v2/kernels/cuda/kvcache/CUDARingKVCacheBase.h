/**
 * @file CUDARingKVCacheBase.h
 * @brief Common base class for CUDA ring buffer KV caches
 * @author David Sanftenberg
 *
 * Extracts shared ring buffer bookkeeping, graph capture support,
 * and common IKVCache implementations from CUDARingKVCache<P> and
 * CUDARingKVCacheTQ into a single base class.
 *
 * Hierarchy:
 *   IKVCache
 *   └── CUDARingKVCacheBase (this class)
 *       ├── ICUDARingKVCache (typed CUDA device pointer APIs)
 *       │   └── CUDARingKVCache<P>
 *       └── CUDARingKVCacheTQ (TurboQuant asymmetric precision)
 */

#pragma once

#include "../../IKVCache.h"

namespace llaminar2
{

    /**
     * @brief Common base for CUDA ring buffer KV caches.
     *
     * Stores core dimensions (layers, batch, seq_len, heads, etc.),
     * manages graph capture head params (d_head_params_/h_head_params_),
     * and provides shared IKVCache implementations via abstract
     * entry accessors that derived classes implement.
     *
     * Ring buffer state (head, count per entry) is owned by derived classes
     * in their type-specific entry structs. This base accesses them through
     * protected pure virtual methods, avoiding any state duplication.
     */
    class CUDARingKVCacheBase : public IKVCache
    {
    public:
        virtual ~CUDARingKVCacheBase();

        // Non-copyable, non-movable (owns GPU memory)
        CUDARingKVCacheBase(const CUDARingKVCacheBase &) = delete;
        CUDARingKVCacheBase &operator=(const CUDARingKVCacheBase &) = delete;

        // Bring the base-class single-arg overload into scope so it isn't
        // hidden by the two-arg override below (silences NVCC #611-D).
        using IKVCache::clear_sequence;

        // =====================================================================
        // IKVCache implementations
        // =====================================================================

        int n_layers() const override { return n_layers_; }
        int max_seq_len() const override { return max_seq_len_; }
        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;

        // =====================================================================
        // Graph Capture Support (IKVCache overrides)
        // =====================================================================

        bool isGraphCaptureReady() const override { return d_head_params_ != nullptr; }
        void setDynamicHead(int layer, int seq_idx, void *gpu_stream) override;
        void advanceHead(int layer, int seq_idx, int num_tokens) override;

        // =====================================================================
        // Common Accessors
        // =====================================================================

        int batch_size() const { return batch_size_; }
        int n_kv_heads() const { return n_kv_heads_; }
        int head_dim() const { return head_dim_; }
        int kv_dim() const { return kv_dim_; }
        int device_id() const { return device_id_; }

        /// Backward-compatible alias for n_layers()
        int num_layers() const { return n_layers_; }

        int get_head_position(int layer, int seq_idx = 0) const;
        bool is_wrapped(int layer, int seq_idx = 0) const;

    protected:
        CUDARingKVCacheBase(int n_layers, int batch_size, int max_seq_len,
                            int n_kv_heads, int head_dim, int kv_dim, int device_id);

        // Core parameters
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;
        int head_dim_;
        int kv_dim_;
        int device_id_;

        // Graph capture device params
        // Layout: [n_layers_ * batch_size_] ints
        int *d_head_params_ = nullptr; ///< Device-side head position buffer
        int *h_head_params_ = nullptr; ///< Pinned host-side head position buffer
        bool wrap_warned_ = false;     ///< One-time warning when ring buffer wraps

        void allocateDeviceParams();
        void freeDeviceParams();

        bool validLayerSeq(int layer, int seq_idx) const
        {
            return layer >= 0 && layer < n_layers_ &&
                   seq_idx >= 0 && seq_idx < batch_size_;
        }

        // =====================================================================
        // Ring State Access (implemented by derived classes)
        // =====================================================================

        /// Get the head (write) position for an entry
        virtual int entryHead(int layer, int seq_idx) const = 0;

        /// Get the count (valid tokens) for an entry
        virtual int entryCount(int layer, int seq_idx) const = 0;

        /// Set the head position for an entry
        virtual void setEntryHead(int layer, int seq_idx, int value) = 0;

        /// Set the count for an entry
        virtual void setEntryCount(int layer, int seq_idx, int value) = 0;

        /// Reset an entry to empty state (head=0, count=0, plus type-specific cleanup)
        virtual void resetEntry(int layer, int seq_idx) = 0;

        // =====================================================================
        // Hooks for derived class behaviors
        // =====================================================================

        /// Called after an entry is cleared (for scratch/shadow invalidation)
        virtual void onClearSequence(int layer, int seq_idx) {}

        /// Called when tokens are evicted during advanceHead
        virtual void onEviction(int layer, int seq_idx, int num_evicted) {}

        /// Called after head is advanced and count is updated
        virtual void onAdvanceComplete(int layer, int seq_idx) {}
    };

} // namespace llaminar2
