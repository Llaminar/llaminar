/**
 * @file CPUHybridRingKVCache.h
 * @brief Hybrid KV cache for CPU: KV entries only for FA layers, GDN state for GDN layers
 *
 * Inherits from CPURingKVCache and adds:
 * - Layer index remapping (global → compressed KV index for FA layers)
 * - Per-layer GDN state management (recurrence + conv state)
 * - Memory savings by not allocating KV entries for GDN layers
 *
 * All KV cache operations are delegated to the parent CPURingKVCache with
 * remapped layer indices. GDN state is accessed via dedicated methods.
 *
 * @see HybridKVCacheConfig.h for HybridLayerMap and HybridGDNLayerState
 */

#pragma once

#include "CPURingKVCache.h"
#include "../HybridKVCacheConfig.h"
#include "../IHybridKVCache.h"

namespace llaminar2
{

    /**
     * @brief CPU ring-buffer KV cache with hybrid FA/GDN layer support
     *
     * For a model with N total layers where only K are full-attention:
     * - Parent CPURingKVCache is constructed with n_layers=K (FA layers only)
     * - This class reports n_layers()=N (total layers)
     * - All layer-indexed KV methods remap to the compressed [0,K) space
     * - GDN layers return 0 cached tokens and no-op on append/clear
     *
     * @tparam KPrecision Activation storage format for K cache
     * @tparam VPrecision Activation storage format for V cache (defaults to KPrecision)
     */
    template <ActivationPrecision KPrecision = ActivationPrecision::FP32,
              ActivationPrecision VPrecision = KPrecision>
    class CPUHybridRingKVCache : public CPURingKVCache<KPrecision, VPrecision>,
                                 public IHybridKVCache
    {
        using Base = CPURingKVCache<KPrecision, VPrecision>;

    public:
        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct a non-sharded hybrid KV cache
         *
         * @param hybrid_config  Hybrid config with layer types and GDN params
         * @param mpi_ctx        MPI context for tensor allocation
         * @param n_layers       Total number of layers (FA + GDN)
         * @param batch_size     Maximum batch size
         * @param max_seq_len    Ring buffer capacity per sequence
         * @param n_kv_heads     Total number of KV attention heads
         * @param head_dim       Dimension of each attention head
         * @param device         Device placement (default: CPU)
         * @param layout_mode    Memory layout (default: POSITION_MAJOR)
         */
        CPUHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int head_dim, DeviceId device = DeviceId::cpu(),
            KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR)
            : Base(mpi_ctx, hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, head_dim, device, layout_mode),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        /**
         * @brief Construct a sharded hybrid KV cache (tensor parallelism)
         */
        CPUHybridRingKVCache(
            const HybridKVCacheConfig &hybrid_config,
            const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
            int n_kv_heads, int local_n_kv_heads, int kv_head_start,
            int head_dim, DeviceId device = DeviceId::cpu(),
            KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR)
            : Base(mpi_ctx, hybrid_config.countKVLayers(), batch_size, max_seq_len,
                   n_kv_heads, local_n_kv_heads, kv_head_start,
                   head_dim, device, layout_mode),
              total_layers_(n_layers)
        {
            initHybrid(hybrid_config);
        }

        // =====================================================================
        // IKVCache Overrides — Total Layer Count
        // =====================================================================

        int n_layers() const override { return total_layers_; }
        int num_layers() const override { return total_layers_; }

        // =====================================================================
        // IKVCache Overrides — Layer-Indexed Methods
        // =====================================================================

        int get_cached_tokens(int layer, int seq_idx = 0) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return 0; // GDN layer — no KV cache tokens
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
                return false; // GDN layer
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

        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return true; // GDN layer — no-op
            return Base::append_kv(kv_idx, seq_idx, new_k, new_v);
        }

        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return true; // GDN layer — no-op
            return Base::append_kv(kv_idx, seq_idx, new_k, new_v, num_tokens);
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
                return; // GDN — no per-sequence clear needed (state is global)
            Base::clear_sequence(kv_idx, seq_idx);
        }

        void clear_layer(int layer) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx >= 0)
            {
                Base::clear_layer(kv_idx);
            }
            else
            {
                // Reset GDN state for this layer
                int gdn_idx = layer_map_.toGDNIndex(layer);
                if (gdn_idx >= 0 && gdn_idx < static_cast<int>(gdn_states_.size()))
                {
                    gdn_states_[gdn_idx].reset();
                    gdn_states_[gdn_idx].resetGPUKernelState();
                }
            }
        }

        int gather_kv_batched(int layer, int num_sequences,
                              TensorBase *out_k, TensorBase *out_v,
                              std::vector<int> &out_kv_lens) override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return -1; // GDN layer
            return Base::gather_kv_batched(kv_idx, num_sequences, out_k, out_v, out_kv_lens);
        }

        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len = nullptr,
                              const typename Base::KVReadParams *rope = nullptr) override
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

        DeviceId get_layer_device(int layer) const override
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return DeviceId::cpu(); // GDN state is on CPU
            return Base::get_layer_device(kv_idx);
        }

        int ring_head(int layer, int seq_idx = 0) const
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return 0;
            return Base::ring_head(kv_idx, seq_idx);
        }

        int ring_size(int layer, int seq_idx = 0) const
        {
            int kv_idx = layer_map_.toKVIndex(layer);
            if (kv_idx < 0)
                return 0;
            return Base::ring_size(kv_idx, seq_idx);
        }

        // =====================================================================
        // GDN State Access
        // =====================================================================

        /// Check if a layer is a GDN layer
        bool isGDNLayer(int layer) const override { return !layer_map_.isFullAttention(layer); }

        /// Check if a layer is a full-attention layer
        bool isFullAttentionLayer(int layer) const override { return layer_map_.isFullAttention(layer); }

        /// Get GDN state for a layer (nullptr if FA layer)
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

        /// Get mutable recurrence state pointer for a GDN layer
        float *getRecurrenceState(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->recurrence_state.data() : nullptr;
        }

        /// Get mutable conv state pointer for a GDN layer
        float *getConvState(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->conv_state.data() : nullptr;
        }

        // =====================================================================
        // GDN Kernel Access
        // =====================================================================

        /// Get short convolution kernel for a GDN layer (nullptr if FA)
        ITensorShortConvolution *getConvKernel(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->conv_kernel.get() : nullptr;
        }

        /// Get gated delta net recurrence kernel for a GDN layer (nullptr if FA)
        ITensorGatedDeltaNet *getRecurrenceKernel(int layer) override
        {
            auto *state = getGDNState(layer);
            return state ? state->rec_kernel.get() : nullptr;
        }

        /// Reset all GDN states (for new sequence)
        void resetGDNStates() override
        {
            for (auto &state : gdn_states_)
            {
                state.reset();
                state.resetGPUKernelState();
            }
        }

        /// Number of KV cache (FA) layers
        int kvLayerCount() const override { return layer_map_.kvLayerCount(); }

        /// Number of GDN layers
        int gdnLayerCount() const override { return layer_map_.gdnLayerCount(); }

        /// Total GDN state memory in bytes
        size_t gdnMemoryBytes() const override
        {
            size_t total = 0;
            for (const auto &state : gdn_states_)
                total += state.memoryBytes();
            return total;
        }

        /// Access the layer map
        const HybridLayerMap &layerMap() const { return layer_map_; }

    private:
        int total_layers_;
        HybridLayerMap layer_map_;
        std::vector<HybridGDNLayerState> gdn_states_;

        void initHybrid(const HybridKVCacheConfig &config)
        {
            layer_map_.build(config.layer_types);

            // Initialize GDN states
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

            LOG_DEBUG("[CPUHybridRingKVCache] Created: " << total_layers_ << " total layers, "
                                                         << layer_map_.kvLayerCount() << " KV (FA), "
                                                         << n_gdn << " GDN. "
                                                         << "GDN state: " << (gdnMemoryBytes() / 1024) << " KB");
        }
    };

    // =========================================================================
    // Convenience Type Aliases
    // =========================================================================

    using CPUHybridRingKVCacheFP32 = CPUHybridRingKVCache<ActivationPrecision::FP32>;
    using CPUHybridRingKVCacheBF16 = CPUHybridRingKVCache<ActivationPrecision::BF16>;
    using CPUHybridRingKVCacheFP16 = CPUHybridRingKVCache<ActivationPrecision::FP16>;
    using CPUHybridRingKVCacheQ8_1 = CPUHybridRingKVCache<ActivationPrecision::Q8_1>;
    using CPUHybridRingKVCacheQ16_1 = CPUHybridRingKVCache<ActivationPrecision::Q16_1>;
    using CPUHybridRingKVCacheTQ4 = CPUHybridRingKVCache<ActivationPrecision::TQ4>;
    using CPUHybridRingKVCacheTQ8 = CPUHybridRingKVCache<ActivationPrecision::TQ8>;
    using CPUHybridRingKVCacheTQ = CPUHybridRingKVCache<ActivationPrecision::TQ8, ActivationPrecision::TQ4>;

} // namespace llaminar2
