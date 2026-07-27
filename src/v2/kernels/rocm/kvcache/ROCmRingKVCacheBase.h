/**
 * @file ROCmRingKVCacheBase.h
 * @brief Common base class for ROCm ring buffer KV caches
 * @author David Sanftenberg
 *
 * Extracts shared ring buffer bookkeeping, graph capture support,
 * and common IKVCache implementations from ROCmRingKVCache<P> and
 * ROCmRingKVCacheTQ into a single base class.
 *
 * Hierarchy:
 *   IKVCache
 *   └── ROCmRingKVCacheBase (this class)
 *       ├── IROCmRingKVCache (typed ROCm device pointer APIs)
 *       │   └── ROCmRingKVCache<P>
 *       └── ROCmRingKVCacheTQ (TurboQuant asymmetric precision)
 */

#pragma once

#include "../../IKVCache.h"
#include <vector>

namespace llaminar2
{

    /**
     * @brief Common base for ROCm ring buffer KV caches.
     *
     * Stores core dimensions (layers, batch, seq_len, heads, etc.), owns the
     * canonical device-resident head/count rows, and provides shared IKVCache
     * control-boundary operations. Derived cache entries own payload pointers
     * only; they must never retain a second host copy of mutable ring state.
     */
    class ROCmRingKVCacheBase : public IKVCache
    {
    public:
        virtual ~ROCmRingKVCacheBase();

        // Non-copyable, non-movable (owns GPU memory)
        ROCmRingKVCacheBase(const ROCmRingKVCacheBase &) = delete;
        ROCmRingKVCacheBase &operator=(const ROCmRingKVCacheBase &) = delete;

        // =====================================================================
        // IKVCache implementations
        // =====================================================================

        int n_layers() const override { return n_layers_; }
        int max_seq_len() const override { return max_seq_len_; }
        int get_cached_tokens(int layer, int seq_idx = 0) const override;
        KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override;
        bool truncateSequence(
            int seq_idx,
            int cached_tokens,
            void *stream = nullptr) override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;

        // =====================================================================
        // Graph Capture Support (IKVCache overrides)
        // =====================================================================

        bool isGraphCaptureReady() const override
        {
            return d_head_params_ != nullptr && d_count_params_ != nullptr;
        }
        bool supportsDeviceResidentSequenceStatePublication() const override
        {
            return d_head_params_ != nullptr && d_count_params_ != nullptr;
        }
        bool bindGraphAppendCountSource(
            int layer,
            int seq_idx,
            const int32_t *append_tokens_device,
            int captured_max_tokens,
            void *gpu_stream) override;
        const int *deviceCachedTokenCountPtr(int layer, int seq_idx = 0) const override;
        const int *deviceRingHeadPtr(int layer, int seq_idx = 0) const override;
        size_t deviceSequenceStateCheckpointBytes() const override;
        bool captureDeviceSequenceStateCheckpoint(
            int seq_idx,
            void *checkpoint_device,
            size_t checkpoint_bytes,
            void *stream,
            std::string *error = nullptr) const override;
        bool restoreDeviceSequenceStateCheckpoint(
            int seq_idx,
            const void *checkpoint_device,
            size_t checkpoint_bytes,
            void *stream,
            std::string *error = nullptr) override;
        bool publishSequenceStateFromDeviceMetadata(
            const DeviceSequenceStatePublicationRequest &request,
            std::string *error = nullptr) override;

        // =====================================================================
        // Common Accessors
        // =====================================================================

        int batch_size() const { return batch_size_; }
        int n_kv_heads() const override { return n_kv_heads_; }
        int head_dim() const { return head_dim_; }
        int kv_dim() const { return kv_dim_; }
        int device_id() const { return device_id_; }

        /// Backward-compatible alias for n_layers()
        int num_layers() const { return n_layers_; }

        int get_head_position(int layer, int seq_idx = 0) const;
        int ring_head(int layer, int seq_idx = 0) const override { return get_head_position(layer, seq_idx); }
        bool is_wrapped(int layer, int seq_idx = 0) const;

    protected:
        ROCmRingKVCacheBase(int n_layers, int batch_size, int max_seq_len,
                            int n_kv_heads, int head_dim, int kv_dim, int device_id);

        // Core parameters
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;
        int head_dim_;
        int kv_dim_;
        int device_id_;

        // Canonical graph-captured device sequence state.
        // Layout: [n_layers_ * batch_size_] ints
        int *d_head_params_ = nullptr;  ///< Device-side head position buffer
        int *d_count_params_ = nullptr; ///< Device-side cached-token count buffer
        /** Stable arena-owned device count source per cache entry, or nullptr. */
        std::vector<const int32_t *> append_count_sources_;

        /**
         * @brief Establish this cache's allocation device on the calling thread.
         *
         * LocalTP invokes participant caches serially from one coordinator
         * thread. HIP's ambient current device can therefore name a sibling
         * when this object receives an out-of-band checkpoint, restore, or
         * publication request. Every base-class runtime or kernel operation
         * must call this method before using an owned pointer or stream.
         *
         * @param operation Human-readable operation name used in diagnostics.
         * @param error Optional detailed failure destination for public APIs.
         * @return true when the HIP runtime selected @ref device_id_.
         */
        bool activateOwningDevice(
            const char *operation,
            std::string *error = nullptr) const;

        void allocateDeviceParams();
        void freeDeviceParams();
        const int *deviceDynamicAppendCountPtr(int layer, int seq_idx) const;
        bool setDeviceSequenceState(
            int layer,
            int seq_idx,
            int head,
            int count,
            void *gpu_stream);

        /**
         * @brief Remove oldest visible rows using only canonical device state.
         *
         * The HIP kernel saturates the device-owned count at zero while
         * preserving the ring head. No host metadata mirror is consulted and
         * the method never synchronizes the supplied explicit stream.
         *
         * @param layer Local cache layer index.
         * @param seq_idx Request index within the cache batch.
         * @param num_tokens Number of oldest visible rows to discard.
         * @param gpu_stream Explicit HIP stream that owns the mutation.
         * @return true when the metadata kernel was accepted for launch.
         */
        bool evictOldestDeviceSequenceState(
            int layer,
            int seq_idx,
            int num_tokens,
            void *gpu_stream);

        /**
         * @brief Materialize one immutable diagnostic snapshot from device state.
         *
         * This method is intentionally synchronous and must never be called by
         * graph construction, capture, or replay. It creates a temporary value,
         * not a cache-owned host mirror.
         */
        bool observeDeviceSequenceState(
            int layer,
            int seq_idx,
            KVCacheSequenceState *state) const;

        bool validLayerSeq(int layer, int seq_idx) const
        {
            return layer >= 0 && layer < n_layers_ &&
                   seq_idx >= 0 && seq_idx < batch_size_;
        }

        // =====================================================================
        // Hooks for derived class behaviors
        // =====================================================================

        /// Called after an entry is cleared (for scratch/shadow invalidation)
        virtual void onClearSequence(int layer, int seq_idx) {}

    };

} // namespace llaminar2
