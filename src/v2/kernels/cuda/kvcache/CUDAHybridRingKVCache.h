/**
 * @file CUDAHybridRingKVCache.h
 * @brief Hybrid KV cache for CUDA: KV entries only for FA layers, GDN state for GDN layers
 *
 * Inherits from CUDARingKVCache and adds:
 * - Layer index remapping (global → compressed KV index for FA layers)
 * - Per-layer GDN state management (recurrence + conv state)
 * - Memory savings by not allocating GPU KV entries for GDN layers
 *
 * @see HybridKVCacheConfig.h for HybridLayerMap and HybridGDNLayerState
 */

#pragma once

#include "CUDARingKVCache.h"
#include "../../HybridGDNDeviceStateArena.h"
#include "../../HybridKVCacheConfig.h"
#include "../../IHybridKVCache.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../backends/GPUDeviceContextPool.h"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <utility>
#include <unistd.h>

namespace llaminar2
{

    /**
     * @brief CUDA ring-buffer KV cache with hybrid FA/GDN layer support
     *
     * For a model with N total layers where only K are full-attention:
     * - Parent CUDARingKVCache is constructed with n_layers=K (FA layers only)
     * - This class reports n_layers()=N (total layers)
     * - All layer-indexed KV methods remap to the compressed [0,K) space
     * - GDN layers return 0 cached tokens and no-op on append/clear
     *
     * @tparam Precision Activation storage format (FP32, FP16, BF16)
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class CUDAHybridRingKVCache : public CUDARingKVCache<Precision>,
                                  public IHybridKVCache
    {
        using Base = CUDARingKVCache<Precision>;
        using DataT = typename Base::DataT;

    public:
        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct a non-sharded hybrid CUDA KV cache
         */
        CUDAHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int head_dim, int device_id,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, head_dim, device_id),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config, std::move(memory_authority));
        }

        /**
         * @brief Construct a non-sharded hybrid CUDA KV cache with device context
         */
        CUDAHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int head_dim, IWorkerGPUContext *ctx,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, head_dim, ctx),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config, std::move(memory_authority));
        }

        /**
         * @brief Construct a sharded hybrid CUDA KV cache (tensor parallelism)
         */
        CUDAHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int local_n_kv_heads, int kv_head_start,
            int head_dim, int device_id,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, local_n_kv_heads, kv_head_start,
                   head_dim, device_id),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config, std::move(memory_authority));
        }

        /**
         * @brief Construct a sharded hybrid CUDA KV cache with device context
         */
        CUDAHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int local_n_kv_heads, int kv_head_start,
            int head_dim, IWorkerGPUContext *ctx,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
            : Base(hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, local_n_kv_heads, kv_head_start,
                   head_dim, ctx),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config, std::move(memory_authority));
        }

        // =====================================================================
        // IKVCache Overrides — Total Layer Count
        // =====================================================================

        int n_layers() const override { return total_layers_; }
        int first_layer_index() const override { return first_layer_index_; }

        // =====================================================================
        // IKVCache Overrides — Layer-Indexed Methods
        // =====================================================================

        int get_cached_tokens(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return 0;
            return Base::get_cached_tokens(kv_idx, seq_idx);
        }

        /**
         * @brief Device pointer to the compressed FA slot's live cached-token count.
         *
         * Hybrid caches allocate CUDA KV entries only for full-attention layers,
         * so model-layer ids must be remapped before exposing device-owned
         * sequence metadata.  Attention and graph replay use this pointer to
         * derive dynamic KV lengths on device; returning the un-remapped base
         * pointer would make the payload path and metadata path disagree.
         */
        const int *deviceCachedTokenCountPtr(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return nullptr;
            return Base::deviceCachedTokenCountPtr(kv_idx, seq_idx);
        }

        /**
         * @brief Device pointer to the compressed FA slot's live ring head.
         */
        const int *deviceRingHeadPtr(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return nullptr;
            return Base::deviceRingHeadPtr(kv_idx, seq_idx);
        }

        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
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
            ITensor *k = nullptr;
            ITensor *v = nullptr;
            const bool ok = const_cast<CUDAHybridRingKVCache<Precision> *>(this)->get_kv(
                layer, seq_idx, &k, &v, out_kv_len);
            if (ok)
            {
                if (out_k)
                    *out_k = k;
                if (out_v)
                    *out_v = v;
            }
            return ok;
        }

        bool get_kv_snapshot_view(int layer, int seq_idx,
                                  int token_count,
                                  ITensor **out_k, ITensor **out_v,
                                  int *out_kv_len = nullptr) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
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
            return Base::get_kv_snapshot_view(kv_idx, seq_idx, token_count, out_k, out_v, out_kv_len);
        }

        bool get_kv_snapshot_view(int layer, int seq_idx,
                                  int token_count,
                                  const ITensor **out_k, const ITensor **out_v,
                                  int *out_kv_len = nullptr) const override
        {
            ITensor *k = nullptr;
            ITensor *v = nullptr;
            const bool ok = const_cast<CUDAHybridRingKVCache<Precision> *>(this)->get_kv_snapshot_view(
                layer, seq_idx, token_count, &k, &v, out_kv_len);
            if (ok)
            {
                if (out_k)
                    *out_k = k;
                if (out_v)
                    *out_v = v;
            }
            return ok;
        }

        ITensor *get_k(int layer, int seq_idx = 0) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return nullptr;
            return Base::get_k(kv_idx, seq_idx);
        }

        const ITensor *get_k(int layer, int seq_idx = 0) const override
        {
            return const_cast<CUDAHybridRingKVCache<Precision> *>(this)->get_k(layer, seq_idx);
        }

        ITensor *get_v(int layer, int seq_idx = 0) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return nullptr;
            return Base::get_v(kv_idx, seq_idx);
        }

        const ITensor *get_v(int layer, int seq_idx = 0) const override
        {
            return const_cast<CUDAHybridRingKVCache<Precision> *>(this)->get_v(layer, seq_idx);
        }

        // CUDA-specific append (device pointer version)
        bool append(int layer, int seq_idx,
                    const void *d_k, const void *d_v,
                    int num_tokens, cudaStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return true; // GDN layer — no-op
            return Base::append(kv_idx, seq_idx, d_k, d_v, num_tokens, stream);
        }

        bool appendConvertedWithStream(int layer, int seq_idx,
                                       const void *d_k_src, const void *d_v_src,
                                       TensorType src_type,
                                       int num_tokens, cudaStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return true; // GDN layer — no KV payload to publish.
            return Base::appendConvertedWithStream(
                kv_idx, seq_idx, d_k_src, d_v_src, src_type, num_tokens, stream);
        }

        /**
         * @brief Publish grouped verifier K/V rows through the hybrid FA map.
         *
         * Qwen3.6 hybrid graphs address stages by global model layer, while the
         * base GPU ring cache stores only full-attention layers in compressed
         * FA-slot order.  Keeping this remap in the hybrid cache makes normal
         * decode, grouped verifier decode, prefix restore, and graph-captured
         * replay all use the same ownership rule.
         */
        bool appendVerifierRowsDecodeEquivalent(int layer,
                                                int seq_idx,
                                                const ITensor *K,
                                                const ITensor *V,
                                                int verifier_rows,
                                                void *gpu_stream = nullptr) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return true; // GDN layer — no KV payload to publish.
            return Base::appendVerifierRowsDecodeEquivalent(
                kv_idx, seq_idx, K, V, verifier_rows, gpu_stream);
        }

        bool get_kv_for_attention(int layer, int seq_idx,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, cudaStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
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
                          int *kv_len, cudaStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return false;
            return Base::linearize_to(kv_idx, seq_idx, d_k_out, d_v_out, kv_len, stream);
        }

        void evict_oldest(int layer, int seq_idx, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return;
            Base::evict_oldest(kv_idx, seq_idx, num_tokens);
        }

        void evict_oldest_layer(int layer, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return;
            Base::evict_oldest_layer(kv_idx, num_tokens);
        }

        int gather_kv_batched(int layer, int num_seqs,
                              void *d_k_out, void *d_v_out,
                              int *kv_lens, int max_kv_len,
                              cudaStream_t stream) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return -1;
            return Base::gather_kv_batched(kv_idx, num_seqs, d_k_out, d_v_out,
                                           kv_lens, max_kv_len, stream);
        }

        /**
         * @brief Remap a model-layer direct ring read to its compressed FA slot.
         *
         * Hybrid storage allocates physical rings only for full-attention
         * layers. The graph nevertheless addresses this interface with the
         * model layer id, so payload, head, and count must all be resolved
         * through the same immutable layer map before capture embeds them.
         *
         * @return true when @p layer names a full-attention layer with a
         *         complete native floating ring contract.
         */
        bool get_kv_device_ring_view(
            int layer,
            int seq_idx,
            ITensor **out_k,
            ITensor **out_v,
            const int **device_head,
            const int **device_count,
            int *physical_capacity,
            void *gpu_stream) override
        {
            const int kv_idx =
                layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                if (device_head)
                    *device_head = nullptr;
                if (device_count)
                    *device_count = nullptr;
                if (physical_capacity)
                    *physical_capacity = 0;
                return false;
            }
            return Base::get_kv_device_ring_view(
                kv_idx,
                seq_idx,
                out_k,
                out_v,
                device_head,
                device_count,
                physical_capacity,
                gpu_stream);
        }

        bool get_kv_batched_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ITensor **out_k,
            ITensor **out_v,
            void *gpu_stream) override
        {
            const int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                return false;
            }
            return Base::get_kv_batched_device_view(
                kv_idx,
                first_seq_idx,
                request_count,
                out_k,
                out_v,
                gpu_stream);
        }

        /**
         * @brief Remap a converted resident read to the compressed FA slot.
         *
         * RoPE-on-read calls this virtual interface with the graph's global
         * model-layer id. The parent CUDA ring owns only full-attention layers,
         * so forwarding that id unchanged can select a different compressed
         * slot that happens to be numerically in range. Payload pointers,
         * device head/count metadata, conversion, and grouped RoPE must all use
         * the same remapped slot established by @ref HybridLayerMap.
         *
         * @param layer Global or pipeline-local model layer accepted by this
         *        hybrid cache.
         * @param first_seq_idx First independent request bank to materialize.
         * @param request_count Number of contiguous request banks.
         * @param target Converted activation precision requested by attention.
         * @param out_k Receives the fixed-stride resident K view.
         * @param out_v Receives the fixed-stride resident V view.
         * @param read Device stream, RoPE geometry, and logical head metadata.
         * @return true when the global layer is full-attention and the parent
         *         cache enqueued the converted grouped read.
         */
        bool get_kv_batched_converted_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ActivationPrecision target,
            ITensor **out_k,
            ITensor **out_v,
            const typename IKVCache::KVReadParams &read) override
        {
            const int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
            {
                if (out_k)
                    *out_k = nullptr;
                if (out_v)
                    *out_v = nullptr;
                return false;
            }
            return Base::get_kv_batched_converted_device_view(
                kv_idx,
                first_seq_idx,
                request_count,
                target,
                out_k,
                out_v,
                read);
        }

        bool resetRequestState(
            const typename IKVCache::StateResetContext &context) override
        {
            if (!Base::resetRequestState(context))
                return false;
            for (size_t gdn_index = 0; gdn_index < gdn_states_.size(); ++gdn_index)
            {
                auto &state = gdn_states_[gdn_index];
                if (!state.resetGPUKernelState(context.execution_stream))
                {
                    LOG_ERROR("[CUDAHybridRingKVCache] Failed to enqueue cache-owned GDN request reset"
                              << " reason=" << context.reason
                              << " gdn_index=" << gdn_index
                              << " stream=" << context.execution_stream);
                    const char message[] =
                        "[FATAL] CUDA hybrid cache request reset failed: "
                        "owner=gdn\n";
                    (void)::write(
                        STDERR_FILENO,
                        message,
                        sizeof(message) - 1);
                    return false;
                }
            }
            return true;
        }

        bool resetLayerSequenceState(
            int layer,
            int seq_idx,
            const typename IKVCache::StateResetContext &context) override
        {
            IKVCache::requireGPUExecutionStream(
                context.execution_stream,
                "CUDAHybridRingKVCache::resetLayerSequenceState");
            if (!context.permitsLayerSequenceReset() ||
                !context.hasReason() ||
                seq_idx < 0 || seq_idx >= this->batch_size_)
                return false;
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
            {
                const int gdn_idx =
                    layer_map_.toGDNIndex(normalizeLayerIndex(layer));
                return gdn_idx >= 0 &&
                       gdn_idx < static_cast<int>(gdn_states_.size());
            }
            return CUDARingKVCacheBase::resetLayerSequenceState(
                kv_idx,
                seq_idx,
                context);
        }

        bool resetLayerState(
            int layer,
            const typename IKVCache::StateResetContext &context) override
        {
            IKVCache::requireGPUExecutionStream(
                context.execution_stream,
                "CUDAHybridRingKVCache::resetLayerState");
            if (!context.permitsLayerReset() || !context.hasReason())
            {
                return false;
            }
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx >= 0)
            {
                return CUDARingKVCacheBase::resetLayerState(kv_idx, context);
            }
            int gdn_idx = layer_map_.toGDNIndex(normalizeLayerIndex(layer));
            if (gdn_idx < 0 || gdn_idx >= static_cast<int>(gdn_states_.size()))
                return false;
            if (!gdn_states_[gdn_idx].resetGPUKernelState(
                    context.execution_stream))
            {
                LOG_ERROR("[CUDAHybridRingKVCache] Failed to enqueue layer GDN reset"
                          << " layer=" << layer);
                return false;
            }
            return true;
        }

        typename IKVCache::KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(global_layer));
            if (kv_idx < 0)
                return {};
            return Base::logicalBlockLayout(baseLayerIndexForKVIndex(kv_idx), token_count);
        }

        typename IKVCache::KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(global_layer));
            if (kv_idx < 0)
                return {};
            return Base::sequenceState(baseLayerIndexForKVIndex(kv_idx), seq_idx);
        }

        bool exportLogicalBlock(const IKVCache::KVCacheLogicalBlockDescriptor &desc,
                                void *dst_k,
                                void *dst_v) const override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(desc.layer));
            if (kv_idx < 0)
                return false;
            auto remapped = desc;
            remapped.layer = baseLayerIndexForKVIndex(kv_idx);
            return Base::exportLogicalBlock(remapped, dst_k, dst_v);
        }

        bool importLogicalBlock(const IKVCache::KVCacheLogicalBlockDescriptor &desc,
                                const void *src_k,
                                const void *src_v) override
        {
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(desc.layer));
            if (kv_idx < 0)
                return false;
            auto remapped = desc;
            remapped.layer = baseLayerIndexForKVIndex(kv_idx);
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
            int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
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

        /** @brief Remap one graph append-count binding to the compressed FA slot. */
        bool bindGraphAppendCountSource(
            int layer,
            int seq_idx,
            const int32_t *append_tokens_device,
            int captured_max_tokens,
            void *gpu_stream) override
        {
            const int kv_idx = layer_map_.toKVIndex(normalizeLayerIndex(layer));
            if (kv_idx < 0)
                return false;
            return Base::bindGraphAppendCountSource(
                kv_idx,
                seq_idx,
                append_tokens_device,
                captured_max_tokens,
                gpu_stream);
        }

        // =====================================================================
        // GDN State Access (IHybridKVCache)
        // =====================================================================

        bool isGDNLayer(int layer) const override
        {
            const int local_layer = normalizeLayerIndex(layer);
            return local_layer >= 0 && local_layer < total_layers_ &&
                   !layer_map_.isFullAttention(local_layer);
        }
        bool isFullAttentionLayer(int layer) const override
        {
            return layer_map_.isFullAttention(normalizeLayerIndex(layer));
        }

        HybridGDNLayerState *getGDNState(int layer) override
        {
            int gdn_idx = layer_map_.toGDNIndex(normalizeLayerIndex(layer));
            if (gdn_idx < 0)
                return nullptr;
            return &gdn_states_[gdn_idx];
        }

        const HybridGDNLayerState *getGDNState(int layer) const override
        {
            int gdn_idx = layer_map_.toGDNIndex(normalizeLayerIndex(layer));
            if (gdn_idx < 0)
                return nullptr;
            return &gdn_states_[gdn_idx];
        }

        float *getRecurrenceState(int layer) override
        {
            (void)layer;
            return nullptr;
        }

        float *getConvState(int layer) override
        {
            (void)layer;
            return nullptr;
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
            void *const state_stream = gdnStateStream();
            for (auto &state : gdn_states_)
            {
                if (!state.resetGPUKernelState(state_stream))
                    throw std::runtime_error(
                        "[CUDAHybridRingKVCache] Failed to reset GDN state");
            }
        }

        int kvLayerCount() const override { return layer_map_.kvLayerCount(); }
        int gdnLayerCount() const override { return layer_map_.gdnLayerCount(); }

        size_t gdnMemoryBytes() const override
        {
            return gdn_state_arena_.bytes();
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

            const HybridPrefixStateMetadata metadata = buildHybridPrefixStateMetadata();
            if (!desc.include_device_state || metadata.device_bytes == 0)
                return true;
            if (!dst_host && !dst_device)
                return false;

            void *effective_stream = desc.stream;
            if (!effective_stream && this->device_id() >= 0)
            {
                LOG_ERROR("[CUDAHybridRingKVCache] exportHybridPrefixState requires an explicit GPU stream");
                return false;
            }

            auto *host_cursor = dst_device
                                    ? nullptr
                                    : reinterpret_cast<uint8_t *>(dst_host);
            auto *device_cursor = reinterpret_cast<uint8_t *>(dst_device);
            const bool ok = exportHybridStatePayload(
                host_cursor,
                device_cursor,
                effective_stream,
                false,
                true);
            if (ok && effective_stream && desc.synchronize)
                GPUDeviceContextPool::instance()
                    .getNvidiaContext(this->device_id())
                    .synchronizeStream(effective_stream);
            return ok;
        }

        bool importHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            const void *src_host,
            const void *src_device) override
        {
            if (desc.seq_idx < 0)
                return false;

            const HybridPrefixStateMetadata metadata = buildHybridPrefixStateMetadata();
            if (!desc.include_device_state || metadata.device_bytes == 0)
                return true;
            if (!src_host && !src_device)
                return false;

            void *effective_stream = desc.stream;
            if (!effective_stream && this->device_id() >= 0)
            {
                LOG_ERROR("[CUDAHybridRingKVCache] importHybridPrefixState requires an explicit GPU stream");
                return false;
            }

            const auto *host_cursor = src_device
                                          ? nullptr
                                          : reinterpret_cast<const uint8_t *>(src_host);
            const auto *device_cursor = reinterpret_cast<const uint8_t *>(src_device);
            const bool ok = importHybridStatePayload(
                host_cursor,
                device_cursor,
                effective_stream,
                false,
                true);
            if (ok && effective_stream && desc.synchronize)
                GPUDeviceContextPool::instance()
                    .getNvidiaContext(this->device_id())
                    .synchronizeStream(effective_stream);
            return ok;
        }

        const HybridLayerMap &layerMap() const { return layer_map_; }

    private:
        int total_layers_;
        int first_layer_index_ = 0;
        HybridLayerMap layer_map_;
        HybridGDNStateGeometry gdn_state_geometry_;
        HybridGDNDeviceStateArena gdn_state_arena_;
        std::vector<HybridGDNLayerState> gdn_states_;

        /**
         * @brief Resolve the cache's explicit state-management stream.
         *
         * Cache construction and request reset enqueue state initialization on
         * this stream. Kernel stages later use the same worker context, making
         * ordering visible without a host-side stream synchronization.
         */
        void *gdnStateStream() const
        {
            if (this->deviceContext())
                return this->deviceContext()->defaultStream();
            return GPUDeviceContextPool::instance()
                .getNvidiaContext(this->device_id())
                .defaultStream();
        }

        int normalizeLayerIndex(int layer) const
        {
            if (layer >= first_layer_index_ &&
                layer < first_layer_index_ + total_layers_)
            {
                return layer - first_layer_index_;
            }
            return layer;
        }

        int baseLayerIndexForKVIndex(int kv_idx) const
        {
            return first_layer_index_ + kv_idx;
        }

        HybridPrefixStateMetadata buildHybridPrefixStateMetadata() const
        {
            HybridPrefixStateMetadata metadata;
            metadata.total_layers = total_layers_;
            metadata.gdn_layers = layer_map_.gdnLayerCount();
            metadata.host_bytes = 0;
            metadata.device_bytes =
                gdn_state_geometry_.deviceSerializedPayloadBytes(
                    metadata.gdn_layers);
            metadata.has_device_kernel_state = metadata.device_bytes > 0;
            return metadata;
        }

        bool exportHybridStatePayload(
            uint8_t *&host_cursor,
            uint8_t *&device_cursor,
            void *stream,
            bool include_host_state = false,
            bool include_device_state = true) const
        {
            if (include_host_state)
                return false;
            if (!include_device_state)
                return true;

            for (int layer = 0; layer < total_layers_; ++layer)
            {
                const int gdn_idx = layer_map_.toGDNIndex(layer);
                if (gdn_idx < 0)
                    continue;
                const auto &state = gdn_states_[static_cast<size_t>(gdn_idx)];

                if (state.conv_kernel)
                {
                    if (!exportDeviceBank(
                            *state.conv_kernel,
                            state.local_conv_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                    if (state.full_conv_state_size != state.local_conv_state_size &&
                        !exportDeviceBank(
                            *state.conv_kernel,
                            state.full_conv_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                }
                if (state.rec_kernel)
                {
                    if (!exportDeviceBank(
                            *state.rec_kernel,
                            state.local_recurrence_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                    if (state.full_recurrence_state_size !=
                            state.local_recurrence_state_size &&
                        !exportDeviceBank(
                            *state.rec_kernel,
                            state.full_recurrence_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                }
            }
            return true;
        }

        /**
         * @brief Export one kernel-owned state bank to device or archive memory.
         *
         * A non-null device cursor keeps live checkpoints device resident. A
         * host cursor is accepted only as an explicit prefix-archive transport;
         * it is never retained or adopted as live GPU state.
         */
        template <typename Kernel>
        static bool exportDeviceBank(
            Kernel &kernel,
            int state_size,
            uint8_t *&host_cursor,
            uint8_t *&device_cursor,
            void *stream)
        {
            if (state_size <= 0)
                return true;
            const size_t bytes = static_cast<size_t>(state_size) * sizeof(float);
            if (device_cursor)
            {
                if (!kernel.exportStateForSize(
                        state_size, nullptr, device_cursor, stream))
                    return false;
                device_cursor += bytes;
                return true;
            }
            if (!host_cursor ||
                !kernel.exportStateForSize(
                    state_size, host_cursor, nullptr, stream))
                return false;
            host_cursor += bytes;
            return true;
        }

        bool importHybridStatePayload(
            const uint8_t *&host_cursor,
            const uint8_t *&device_cursor,
            void *stream,
            bool include_host_state = false,
            bool include_device_state = true)
        {
            if (include_host_state)
                return false;
            if (!include_device_state)
                return true;

            for (int layer = 0; layer < total_layers_; ++layer)
            {
                const int gdn_idx = layer_map_.toGDNIndex(layer);
                if (gdn_idx < 0)
                    continue;
                auto &state = gdn_states_[static_cast<size_t>(gdn_idx)];

                if (state.conv_kernel)
                {
                    if (!importDeviceBank(
                            *state.conv_kernel,
                            state.local_conv_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                    if (state.full_conv_state_size != state.local_conv_state_size &&
                        !importDeviceBank(
                            *state.conv_kernel,
                            state.full_conv_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                }
                if (state.rec_kernel)
                {
                    if (!importDeviceBank(
                            *state.rec_kernel,
                            state.local_recurrence_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                    if (state.full_recurrence_state_size !=
                            state.local_recurrence_state_size &&
                        !importDeviceBank(
                            *state.rec_kernel,
                            state.full_recurrence_state_size,
                            host_cursor,
                            device_cursor,
                            stream))
                        return false;
                }
            }
            return true;
        }

        /// @brief Import one serialized bank without ever retaining its host address.
        template <typename Kernel>
        static bool importDeviceBank(
            Kernel &kernel,
            int state_size,
            const uint8_t *&host_cursor,
            const uint8_t *&device_cursor,
            void *stream)
        {
            if (state_size <= 0)
                return true;
            const size_t bytes = static_cast<size_t>(state_size) * sizeof(float);
            if (device_cursor)
            {
                if (!kernel.importStateForSize(
                        state_size, nullptr, device_cursor, stream))
                    return false;
                device_cursor += bytes;
                return true;
            }
            if (!host_cursor ||
                !kernel.importStateForSize(
                    state_size, host_cursor, nullptr, stream))
                return false;
            host_cursor += bytes;
            return true;
        }

        /**
         * @brief Materialize GDN banks from canonical geometry and authority.
         * @param config Immutable hybrid model/participant geometry.
         * @param memory_authority Sole ledger for the GPU allocation.
         */
        void initHybrid(
            const HybridKVCacheConfig &config,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
        {
            first_layer_index_ = config.first_layer_index;
            layer_map_.build(config.layer_types);

            const int n_gdn = layer_map_.gdnLayerCount();
            if (n_gdn <= 0)
                return;

            gdn_state_geometry_ = config.gdnStateGeometry();
            const HybridGDNStateGeometry &geometry =
                gdn_state_geometry_;

            gdn_states_.resize(n_gdn);
            for (auto &state : gdn_states_)
            {
                state.n_v_heads = geometry.local_value_heads;
                state.n_k_heads = geometry.local_key_heads;
                state.d_k = geometry.d_k;
                state.d_v = geometry.d_v;
                state.conv_kernel_size = config.gdn_conv_kernel_size;
                state.full_recurrence_state_size =
                    geometry.full_recurrence_state_floats;
                state.full_conv_state_size =
                    geometry.full_conv_state_floats;
                state.initializeShape(geometry.local_qkv_dim);
            }

            gdn_state_arena_.initialize(
                DeviceId::cuda(this->device_id()),
                this->batch_size_,
                geometry,
                gdn_states_,
                gdnStateStream(),
                std::move(memory_authority));

            LOG_DEBUG("[CUDAHybridRingKVCache] Created: " << total_layers_ << " total layers, "
                                                          << layer_map_.kvLayerCount() << " KV (FA), "
                                                          << n_gdn << " GDN. "
                                                          << "GDN state: " << (gdnMemoryBytes() / 1024) << " KB");
        }
    };

    // =========================================================================
    // Convenience Type Aliases
    // =========================================================================

    using CUDAHybridRingKVCacheFP32 = CUDAHybridRingKVCache<ActivationPrecision::FP32>;
    using CUDAHybridRingKVCacheFP16 = CUDAHybridRingKVCache<ActivationPrecision::FP16>;
    using CUDAHybridRingKVCacheBF16 = CUDAHybridRingKVCache<ActivationPrecision::BF16>;
    using CUDAHybridRingKVCacheQ8_1 = CUDAHybridRingKVCache<ActivationPrecision::Q8_1>;

} // namespace llaminar2
