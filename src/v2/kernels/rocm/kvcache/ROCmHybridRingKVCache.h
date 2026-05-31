/**
 * @file ROCmHybridRingKVCache.h
 * @brief Hybrid KV cache for ROCm: KV entries only for FA layers, GDN state for GDN layers
 *
 * Inherits from ROCmRingKVCache and adds:
 * - Layer index remapping (global → compressed KV index for FA layers)
 * - Per-layer GDN state management (recurrence + conv state)
 * - Memory savings by not allocating GPU KV entries for GDN layers
 *
 * @see HybridKVCacheConfig.h for HybridLayerMap and HybridGDNLayerState
 */

#pragma once

#include "ROCmRingKVCache.h"
#include "../../HybridKVCacheConfig.h"
#include "../../IHybridKVCache.h"
#include "../../../tensors/TensorKernels.h"

#include <cstdint>
#include <cstring>

namespace llaminar2
{

    /**
     * @brief ROCm ring-buffer KV cache with hybrid FA/GDN layer support
     *
     * For a model with N total layers where only K are full-attention:
     * - Parent ROCmRingKVCache is constructed with n_layers=K (FA layers only)
     * - This class reports n_layers()=N (total layers)
     * - All layer-indexed KV methods remap to the compressed [0,K) space
     * - GDN layers return 0 cached tokens and no-op on append/clear
     *
     * @tparam Precision Activation storage format (FP32, FP16, BF16)
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class ROCmHybridRingKVCache : public ROCmRingKVCache<Precision>,
                                  public IHybridKVCache
    {
        using Base = ROCmRingKVCache<Precision>;
        using DataT = typename Base::DataT;

    public:
        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct a non-sharded hybrid ROCm KV cache
         */
        ROCmHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int head_dim, int device_id = 0)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, head_dim, device_id),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        /**
         * @brief Construct a non-sharded hybrid ROCm KV cache with device context
         */
        ROCmHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int head_dim, IWorkerGPUContext *ctx)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, head_dim, ctx),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        /**
         * @brief Construct a sharded hybrid ROCm KV cache (tensor parallelism)
         */
        ROCmHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int local_n_kv_heads, int kv_head_start,
            int head_dim, int device_id = 0)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, local_n_kv_heads, kv_head_start,
                   head_dim, device_id),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        /**
         * @brief Construct a sharded hybrid ROCm KV cache with device context
         */
        ROCmHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int local_n_kv_heads, int kv_head_start,
            int head_dim, IWorkerGPUContext *ctx)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, local_n_kv_heads, kv_head_start,
                   head_dim, ctx),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        // =====================================================================
        // IKVCache Overrides — Total Layer Count
        // =====================================================================

        int n_layers() const override { return total_layers_; }

        // =====================================================================
        // IKVCache Overrides — Layer-Indexed Methods
        // =====================================================================

        int get_cached_tokens(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return 0;
            return Base::get_cached_tokens(kv_idx, seq_idx);
        }

        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                if (out_kv_len)
                    *out_kv_len = 0;
                return false;
            }
            return Base::get_kv(kv_idx, seq_idx, out_k, out_v, out_kv_len);
        }

        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                if (out_kv_len)
                    *out_kv_len = 0;
                return false;
            }
            return Base::get_kv(kv_idx, seq_idx, out_k, out_v, out_kv_len);
        }

        ITensor *get_k(int layer, int seq_idx = 0) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return nullptr;
            return Base::get_k(kv_idx, seq_idx);
        }

        const ITensor *get_k(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return nullptr;
            return Base::get_k(kv_idx, seq_idx);
        }

        ITensor *get_v(int layer, int seq_idx = 0) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return nullptr;
            return Base::get_v(kv_idx, seq_idx);
        }

        const ITensor *get_v(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return nullptr;
            return Base::get_v(kv_idx, seq_idx);
        }

        // ROCm-specific append (device pointer version)
        bool append(int layer, int seq_idx,
                    const void *d_k, const void *d_v,
                    int num_tokens, hipStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return true; // GDN layer — no-op
            return Base::append(kv_idx, seq_idx, d_k, d_v, num_tokens, stream);
        }

        bool get_kv_for_attention(int layer, int seq_idx,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, hipStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
            {
                if (d_k_out)
                    *d_k_out = nullptr;
                if (d_v_out)
                    *d_v_out = nullptr;
                if (kv_len)
                    *kv_len = 0;
                return false;
            }
            return Base::get_kv_for_attention(kv_idx, seq_idx, d_k_out, d_v_out, kv_len, stream);
        }

        bool linearize_to(int layer, int seq_idx,
                          void *d_k_out, void *d_v_out,
                          int *kv_len, hipStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return false;
            return Base::linearize_to(kv_idx, seq_idx, d_k_out, d_v_out, kv_len, stream);
        }

        void evict_oldest(int layer, int seq_idx, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return;
            Base::evict_oldest(kv_idx, seq_idx, num_tokens);
        }

        void evict_oldest_layer(int layer, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return;
            Base::evict_oldest_layer(kv_idx, num_tokens);
        }

        int gather_kv_batched(int layer, int num_seqs,
                              void *d_k_out, void *d_v_out,
                              int *kv_lens, int max_kv_len,
                              hipStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return -1;
            return Base::gather_kv_batched(kv_idx, num_seqs, d_k_out, d_v_out,
                                           kv_lens, max_kv_len, stream);
        }

        void clear() override
        {
            Base::clear();
            for (auto &state : gdn_states_)
            {
                state.reset();
                state.resetGPUKernelState();
            }
        }

        void clear_sequence(int layer, int seq_idx) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return;
            // Qualify fully to avoid name hiding from 'using IKVCache::clear_sequence' in Base
            ROCmRingKVCacheBase::clear_sequence(kv_idx, seq_idx);
        }

        void clear_layer(int layer) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx >= 0)
            {
                for (int seq = 0; seq < this->batch_size_; ++seq)
                {
                    ROCmRingKVCacheBase::clear_sequence(kv_idx, seq);
                }
            }
            else
            {
                int gdn_idx = layer_map_.toGDNIndex(layer);
                if (gdn_idx >= 0 && gdn_idx < static_cast<int>(gdn_states_.size()))
                {
                    gdn_states_[gdn_idx].reset();
                    gdn_states_[gdn_idx].resetGPUKernelState();
                }
            }
        }

        typename IKVCache::KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const override
        {
            int kv_idx = layer_map_.toKVIndex(global_layer);
            if (kv_idx < 0)
                return {};
            return Base::logicalBlockLayout(kv_idx, token_count);
        }

        typename IKVCache::KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override
        {
            int kv_idx = layer_map_.toKVIndex(global_layer);
            if (kv_idx < 0)
                return {};
            return Base::sequenceState(kv_idx, seq_idx);
        }

        bool exportLogicalBlock(const IKVCache::KVCacheLogicalBlockDescriptor &desc,
                                void *dst_k,
                                void *dst_v) const override
        {
            int kv_idx = layer_map_.toKVIndex(desc.layer);
            if (kv_idx < 0)
                return false;
            auto remapped = desc;
            remapped.layer = kv_idx;
            return Base::exportLogicalBlock(remapped, dst_k, dst_v);
        }

        bool importLogicalBlock(const IKVCache::KVCacheLogicalBlockDescriptor &desc,
                                const void *src_k,
                                const void *src_v) override
        {
            int kv_idx = layer_map_.toKVIndex(desc.layer);
            if (kv_idx < 0)
                return false;
            auto remapped = desc;
            remapped.layer = kv_idx;
            return Base::importLogicalBlock(remapped, src_k, src_v);
        }

        bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr) override
        {
            return Base::truncateSequence(seq_idx, cached_tokens, stream);
        }

        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len = nullptr,
                              const typename IKVCache::KVReadParams *rope = nullptr) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                if (out_kv_len)
                    *out_kv_len = 0;
                return false;
            }
            return Base::get_kv_converted(kv_idx, seq_idx, target, out_k, out_v, out_kv_len, rope);
        }

        // Graph capture support with remapping
        void setDynamicHead(int layer, int seq_idx, void *gpu_stream) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return;
            Base::setDynamicHead(kv_idx, seq_idx, gpu_stream);
        }

        void advanceHead(int layer, int seq_idx, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return;
            Base::advanceHead(kv_idx, seq_idx, num_tokens);
        }

        // =====================================================================
        // GDN State Access (IHybridKVCache)
        // =====================================================================

        bool isGDNLayer(int layer) const override { return !layer_map_.isFullAttention(layer); }
        bool isFullAttentionLayer(int layer) const override { return layer_map_.isFullAttention(layer); }

        HybridGDNLayerState *getGDNState(int layer) override
        {
            int gdn_idx = layer_map_.toGDNIndex(layer);
            if (gdn_idx < 0)
                return nullptr;
            return &gdn_states_[gdn_idx];
        }

        const HybridGDNLayerState *getGDNState(int layer) const override
        {
            int gdn_idx = layer_map_.toGDNIndex(layer);
            if (gdn_idx < 0)
                return nullptr;
            return &gdn_states_[gdn_idx];
        }

        float *getRecurrenceState(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->recurrence_state.data() : nullptr;
        }

        float *getConvState(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->conv_state.data() : nullptr;
        }

        ITensorShortConvolution *getConvKernel(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->conv_kernel.get() : nullptr;
        }

        ITensorGatedDeltaNet *getRecurrenceKernel(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->rec_kernel.get() : nullptr;
        }

        void resetGDNStates() override
        {
            for (auto &state : gdn_states_)
            {
                state.reset();
                state.resetGPUKernelState();
            }
        }

        int kvLayerCount() const override { return layer_map_.kvLayerCount(); }
        int gdnLayerCount() const override { return layer_map_.gdnLayerCount(); }

        size_t gdnMemoryBytes() const override
        {
            size_t total = 0;
            for (const auto &state : gdn_states_)
                total += state.memoryBytes();
            return total;
        }

        HybridPrefixStateMetadata hybridPrefixStateMetadata() const override
        {
            return buildHybridPrefixStateMetadata();
        }

        bool exportHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            void *dst_host,
            void *dst_device) const override
        {
            if (desc.seq_idx < 0)
                return false;

            HybridPrefixStateMetadata metadata = buildHybridPrefixStateMetadata();
            if ((metadata.host_bytes > 0 && !dst_host) ||
                (metadata.device_bytes > 0 && !dst_device))
            {
                return false;
            }

            auto *host_cursor = reinterpret_cast<uint8_t *>(dst_host);
            auto *device_cursor = reinterpret_cast<uint8_t *>(dst_device);
            return exportHybridStatePayload(host_cursor, device_cursor, desc.stream);
        }

        bool importHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            const void *src_host,
            const void *src_device) override
        {
            if (desc.seq_idx < 0)
                return false;

            HybridPrefixStateMetadata metadata = buildHybridPrefixStateMetadata();
            if ((metadata.host_bytes > 0 && !src_host) ||
                (metadata.device_bytes > 0 && !src_device))
            {
                return false;
            }

            const auto *host_cursor = reinterpret_cast<const uint8_t *>(src_host);
            const auto *device_cursor = reinterpret_cast<const uint8_t *>(src_device);
            return importHybridStatePayload(host_cursor, device_cursor, desc.stream);
        }

        const HybridLayerMap &layerMap() const { return layer_map_; }

    private:
        int total_layers_;
        HybridLayerMap layer_map_;
        std::vector<HybridGDNLayerState> gdn_states_;

        HybridPrefixStateMetadata buildHybridPrefixStateMetadata() const
        {
            HybridPrefixStateMetadata metadata;
            metadata.total_layers = total_layers_;
            metadata.gdn_layers = layer_map_.gdnLayerCount();
            metadata.host_bytes = gdnMemoryBytes();

            for (int layer = 0; layer < total_layers_; ++layer)
            {
                const auto *state = getGDNState(layer);
                if (!state)
                    continue;
                if (state->conv_kernel)
                    metadata.device_bytes += state->conv_kernel->stateBytes();
                if (state->rec_kernel)
                    metadata.device_bytes += state->rec_kernel->stateBytes();
            }
            metadata.has_device_kernel_state = metadata.device_bytes > 0;
            return metadata;
        }

        bool exportHybridStatePayload(
            uint8_t *&host_cursor,
            uint8_t *&device_cursor,
            void *stream) const
        {
            for (int layer = 0; layer < total_layers_; ++layer)
            {
                const auto *state = getGDNState(layer);
                if (!state)
                    continue;

                const size_t recurrence_bytes = state->recurrence_state.size() * sizeof(float);
                const size_t conv_bytes = state->conv_state.size() * sizeof(float);
                if (recurrence_bytes > 0)
                {
                    std::memcpy(host_cursor, state->recurrence_state.data(), recurrence_bytes);
                    host_cursor += recurrence_bytes;
                }
                if (conv_bytes > 0)
                {
                    std::memcpy(host_cursor, state->conv_state.data(), conv_bytes);
                    host_cursor += conv_bytes;
                }

                if (state->conv_kernel)
                {
                    const size_t bytes = state->conv_kernel->stateBytes();
                    if (bytes > 0)
                    {
                        if (!state->conv_kernel->exportState(nullptr, device_cursor, stream))
                            return false;
                        device_cursor += bytes;
                    }
                }
                if (state->rec_kernel)
                {
                    const size_t bytes = state->rec_kernel->stateBytes();
                    if (bytes > 0)
                    {
                        if (!state->rec_kernel->exportState(nullptr, device_cursor, stream))
                            return false;
                        device_cursor += bytes;
                    }
                }
            }
            return true;
        }

        bool importHybridStatePayload(
            const uint8_t *&host_cursor,
            const uint8_t *&device_cursor,
            void *stream)
        {
            for (int layer = 0; layer < total_layers_; ++layer)
            {
                auto *state = getGDNState(layer);
                if (!state)
                    continue;

                const size_t recurrence_bytes = state->recurrence_state.size() * sizeof(float);
                const size_t conv_bytes = state->conv_state.size() * sizeof(float);
                if (recurrence_bytes > 0)
                {
                    std::memcpy(state->recurrence_state.data(), host_cursor, recurrence_bytes);
                    host_cursor += recurrence_bytes;
                }
                if (conv_bytes > 0)
                {
                    std::memcpy(state->conv_state.data(), host_cursor, conv_bytes);
                    host_cursor += conv_bytes;
                }

                if (state->conv_kernel)
                {
                    const size_t bytes = state->conv_kernel->stateBytes();
                    if (bytes > 0)
                    {
                        if (!state->conv_kernel->importState(nullptr, device_cursor, stream))
                            return false;
                        device_cursor += bytes;
                    }
                }
                if (state->rec_kernel)
                {
                    const size_t bytes = state->rec_kernel->stateBytes();
                    if (bytes > 0)
                    {
                        if (!state->rec_kernel->importState(nullptr, device_cursor, stream))
                            return false;
                        device_cursor += bytes;
                    }
                }
            }
            return true;
        }

        void initHybrid(const HybridKVCacheConfig &config)
        {
            layer_map_.build(config.layer_types);

            const int n_gdn = layer_map_.gdnLayerCount();
            if (n_gdn <= 0)
                return;

            // Compute GDN dimensions (same logic as Qwen35Graph::ensureGDNStates)
            const int n_k_heads_full = config.gdn_group_count > 0
                                           ? config.gdn_group_count
                                           : config.n_heads;
            const int n_v_heads_full = config.gdn_time_step_rank > 0
                                           ? config.gdn_time_step_rank
                                           : n_k_heads_full;

            int n_k_heads = n_k_heads_full;
            int n_v_heads = n_v_heads_full;
            const bool gdn_modular_repeat = (n_v_heads_full > n_k_heads_full);

            if (config.local_n_heads > 0 && config.n_heads > 0 &&
                config.local_n_heads < config.n_heads)
            {
                n_v_heads = n_v_heads_full * config.local_n_heads / config.n_heads;
                if (n_v_heads <= 0)
                    n_v_heads = 1;
                if (!gdn_modular_repeat)
                {
                    n_k_heads = n_k_heads_full * config.local_n_heads / config.n_heads;
                    if (n_k_heads <= 0)
                        n_k_heads = 1;
                }
            }

            const int d_v = config.gdn_state_size;
            const int d_k = d_v;
            const int key_dim = n_k_heads * d_k;
            const int value_dim = config.gdn_inner_size > 0
                                      ? (config.gdn_inner_size * n_v_heads / n_v_heads_full)
                                      : n_v_heads * d_v;
            const int qkv_dim = 2 * key_dim + value_dim;

            gdn_states_.resize(n_gdn);
            for (auto &state : gdn_states_)
            {
                state.n_v_heads = n_v_heads;
                state.n_k_heads = n_k_heads;
                state.d_k = d_k;
                state.d_v = d_v;
                state.conv_kernel_size = config.gdn_conv_kernel_size;
                state.initialize(qkv_dim);
            }

            LOG_DEBUG("[ROCmHybridRingKVCache] Created: " << total_layers_ << " total layers, "
                                                          << layer_map_.kvLayerCount() << " KV (FA), "
                                                          << n_gdn << " GDN. "
                                                          << "GDN state: " << (gdnMemoryBytes() / 1024) << " KB");
        }
    };

    // =========================================================================
    // Convenience Type Aliases
    // =========================================================================

    using ROCmHybridRingKVCacheFP32 = ROCmHybridRingKVCache<ActivationPrecision::FP32>;
    using ROCmHybridRingKVCacheFP16 = ROCmHybridRingKVCache<ActivationPrecision::FP16>;
    using ROCmHybridRingKVCacheBF16 = ROCmHybridRingKVCache<ActivationPrecision::BF16>;
    using ROCmHybridRingKVCacheQ8_1 = ROCmHybridRingKVCache<ActivationPrecision::Q8_1>;

} // namespace llaminar2
