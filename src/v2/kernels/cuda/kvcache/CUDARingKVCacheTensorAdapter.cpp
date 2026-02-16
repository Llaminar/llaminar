/**
 * @file CUDARingKVCacheTensorAdapter.cpp
 * @brief ITensor adapter for CUDA KV Cache
 *
 * This file is compiled by the regular C++ compiler (not nvcc) so it can
 * include heavy headers like CPUTensors.h without MPI header issues.
 *
 * Provides:
 * - ICUDARingKVCache::append(ITensor*) implementation
 * - CUDARingKVCache<>::get_k() and get_v() implementations using GpuTensorView
 */

#include "CUDARingKVCache.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../backends/DeviceId.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KVCacheProfiler.h"

#include <chrono>
#include <cuda_runtime.h>

namespace llaminar2
{

    extern "C" bool cuda_convert_tensor_to_fp16(
        const void *d_src,
        TensorType src_type,
        uint16_t *d_dst,
        int count,
        cudaStream_t stream);

    extern "C" bool cuda_convert_tensor_to_q8_1(
        const void *d_src,
        TensorType src_type,
        Q8_1Block *d_dst,
        int rows,
        int cols,
        cudaStream_t stream);

    // =========================================================================
    // ICUDARingKVCache destructor + conversion scratch buffer management
    // =========================================================================

    ICUDARingKVCache::~ICUDARingKVCache()
    {
        freeConvScratch();
    }

    bool ICUDARingKVCache::ensureConvScratch(size_t bytes)
    {
        if (bytes <= conv_scratch_capacity_)
            return true;

        // Grow to requested size (round up to 4KB for alignment)
        const size_t alloc_size = (bytes + 4095) & ~size_t(4095);

        void *new_k = nullptr;
        void *new_v = nullptr;
        if (cudaMalloc(&new_k, alloc_size) != cudaSuccess ||
            cudaMalloc(&new_v, alloc_size) != cudaSuccess)
        {
            if (new_k)
                cudaFree(new_k);
            if (new_v)
                cudaFree(new_v);
            LOG_ERROR("[ICUDARingKVCache] Failed to allocate conversion scratch buffers ("
                      << alloc_size << " bytes each)");
            return false;
        }

        // Free old buffers
        if (conv_scratch_k_)
            cudaFree(conv_scratch_k_);
        if (conv_scratch_v_)
            cudaFree(conv_scratch_v_);

        conv_scratch_k_ = new_k;
        conv_scratch_v_ = new_v;
        conv_scratch_capacity_ = alloc_size;

        LOG_DEBUG("[ICUDARingKVCache] Allocated conversion scratch: "
                  << alloc_size << " bytes each (" << (alloc_size * 2 / 1024) << " KB total)");
        return true;
    }

    void ICUDARingKVCache::freeConvScratch()
    {
        if (conv_scratch_k_)
        {
            cudaFree(conv_scratch_k_);
            conv_scratch_k_ = nullptr;
        }
        if (conv_scratch_v_)
        {
            cudaFree(conv_scratch_v_);
            conv_scratch_v_ = nullptr;
        }
        conv_scratch_capacity_ = 0;
    }

    // =========================================================================
    // ICUDARingKVCache::append(ITensor*) implementation
    // =========================================================================
    // NOTE: This is in a separate .cpp file (not .cu) because nvcc has issues
    // with some C++ headers. The ITensor interface is lightweight and doesn't
    // require heavy includes.

    bool ICUDARingKVCache::append(int layer, int seq_idx,
                                  const ITensor *K, const ITensor *V,
                                  int num_tokens)
    {
        if (!K || !V)
        {
            LOG_DEBUG("[ICUDARingKVCache::append(ITensor)] Null K or V tensor");
            return false;
        }

        const auto target = DeviceId::cuda(device_id());

        // Try existing GPU pointers first
        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        // If missing, force residency on cache device
        if (!d_k)
        {
            auto *k_mut = const_cast<ITensor *>(K);
            if (!k_mut->ensureOnDevice(target))
            {
                LOG_ERROR("[ICUDARingKVCache::append(ITensor)] Failed to ensure K on "
                          << target.toString());
                return false;
            }
            d_k = K->gpu_data_ptr();
        }
        if (!d_v)
        {
            auto *v_mut = const_cast<ITensor *>(V);
            if (!v_mut->ensureOnDevice(target))
            {
                LOG_ERROR("[ICUDARingKVCache::append(ITensor)] Failed to ensure V on "
                          << target.toString());
                return false;
            }
            d_v = V->gpu_data_ptr();
        }

        if (!d_k || !d_v)
        {
            LOG_ERROR("[ICUDARingKVCache::append(ITensor)] K or V tensor lacks GPU data after ensureOnDevice().");
            return false;
        }

        // Delegate to the device pointer version (with default stream 0)
        return append(layer, seq_idx, d_k, d_v, num_tokens, 0);
    }

    bool ICUDARingKVCache::appendWithStream(int layer, int seq_idx,
                                            const ITensor *K, const ITensor *V,
                                            int num_tokens, void *gpu_stream)
    {
        if (!K || !V)
        {
            LOG_DEBUG("[ICUDARingKVCache::appendWithStream] Null K or V tensor");
            return false;
        }

        const auto target = DeviceId::cuda(device_id());

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k)
        {
            auto *k_mut = const_cast<ITensor *>(K);
            if (!k_mut->ensureOnDevice(target))
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Failed to ensure K on "
                          << target.toString());
                return false;
            }
            d_k = K->gpu_data_ptr();
        }
        if (!d_v)
        {
            auto *v_mut = const_cast<ITensor *>(V);
            if (!v_mut->ensureOnDevice(target))
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Failed to ensure V on "
                          << target.toString());
                return false;
            }
            d_v = V->gpu_data_ptr();
        }

        if (!d_k || !d_v)
        {
            LOG_ERROR("[ICUDARingKVCache::appendWithStream] K or V tensor lacks GPU data after ensureOnDevice().");
            return false;
        }

        const auto stream = static_cast<cudaStream_t>(gpu_stream);

        if (precision() == ActivationPrecision::FP16 &&
            (K->native_type() != TensorType::FP16 || V->native_type() != TensorType::FP16))
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Invalid K/V shape for FP16 conversion");
                return false;
            }

            const int kv_dim = static_cast<int>(k_shape[1]);
            const int elements = num_tokens * kv_dim;
            if (elements <= 0)
            {
                return append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            }

            // --- Profiling: ensure scratch buffers ---
            const auto alloc_start = std::chrono::high_resolution_clock::now();

            const size_t buf_bytes = static_cast<size_t>(elements) * sizeof(uint16_t);
            if (!ensureConvScratch(buf_bytes))
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Failed to ensure FP16 conversion scratch");
                return false;
            }
            auto *d_k_fp16 = static_cast<uint16_t *>(conv_scratch_k_);
            auto *d_v_fp16 = static_cast<uint16_t *>(conv_scratch_v_);

            const auto alloc_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: FP16 conversion kernels ---
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const bool k_ok = cuda_convert_tensor_to_fp16(d_k, K->native_type(), d_k_fp16, elements, stream);
            const bool v_ok = cuda_convert_tensor_to_fp16(d_v, V->native_type(), d_v_fp16, elements, stream);

            if (!k_ok || !v_ok)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] GPU FP16 conversion failed");
                return false;
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: ring buffer append ---
            const auto append_start = std::chrono::high_resolution_clock::now();
            const bool ok = append(layer, seq_idx, d_k_fp16, d_v_fp16, num_tokens, stream);
            const auto append_end = std::chrono::high_resolution_clock::now();

            // Record profiling breakdown
            {
                auto to_ns = [](auto d) -> uint64_t
                {
                    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
                };
                const uint64_t alloc_ns = to_ns(alloc_end - alloc_start);
                const uint64_t conv_ns = to_ns(conv_end - conv_start);
                const uint64_t append_ns = to_ns(append_end - append_start);
                const uint64_t bytes = static_cast<uint64_t>(elements) * sizeof(uint16_t) * 2;
                KVCacheProfiler::record(KVCacheOpType::GPU_ALLOC, alloc_ns);
                KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_FP16, conv_ns, static_cast<uint64_t>(num_tokens), bytes);
                KVCacheProfiler::record(KVCacheOpType::APPEND, append_ns, static_cast<uint64_t>(num_tokens), bytes);
            }

            return ok;
        }

        if (precision() == ActivationPrecision::Q8_1 &&
            (K->native_type() != TensorType::Q8_1 || V->native_type() != TensorType::Q8_1))
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Invalid K/V shape for Q8_1 conversion");
                return false;
            }

            const int kv_dim = static_cast<int>(k_shape[1]);
            const int blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const size_t block_count = static_cast<size_t>(num_tokens) * static_cast<size_t>(blocks_per_row);
            if (block_count == 0)
            {
                return append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            }

            // --- Profiling: ensure scratch buffers ---
            const auto alloc_start = std::chrono::high_resolution_clock::now();

            const size_t buf_bytes = block_count * sizeof(Q8_1Block);
            if (!ensureConvScratch(buf_bytes))
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] Failed to ensure Q8_1 conversion scratch");
                return false;
            }
            auto *d_k_q8 = static_cast<Q8_1Block *>(conv_scratch_k_);
            auto *d_v_q8 = static_cast<Q8_1Block *>(conv_scratch_v_);

            const auto alloc_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: Q8_1 conversion kernels ---
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const bool k_ok = cuda_convert_tensor_to_q8_1(d_k, K->native_type(), d_k_q8, num_tokens, kv_dim, stream);
            const bool v_ok = cuda_convert_tensor_to_q8_1(d_v, V->native_type(), d_v_q8, num_tokens, kv_dim, stream);
            if (!k_ok || !v_ok)
            {
                LOG_ERROR("[ICUDARingKVCache::appendWithStream] GPU Q8_1 conversion failed");
                return false;
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: ring buffer append ---
            const auto append_start = std::chrono::high_resolution_clock::now();
            const bool ok = append(layer, seq_idx, d_k_q8, d_v_q8, num_tokens, stream);
            const auto append_end = std::chrono::high_resolution_clock::now();

            // Record profiling breakdown
            {
                auto to_ns = [](auto d) -> uint64_t
                {
                    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
                };
                const uint64_t alloc_ns = to_ns(alloc_end - alloc_start);
                const uint64_t conv_ns = to_ns(conv_end - conv_start);
                const uint64_t append_ns = to_ns(append_end - append_start);
                const uint64_t bytes = block_count * sizeof(Q8_1Block) * 2;
                KVCacheProfiler::record(KVCacheOpType::GPU_ALLOC, alloc_ns);
                KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q8_1, conv_ns, static_cast<uint64_t>(num_tokens), bytes);
                KVCacheProfiler::record(KVCacheOpType::APPEND, append_ns, static_cast<uint64_t>(num_tokens), bytes);
            }

            return ok;
        }

        // No conversion needed - profile just the append
        {
            const auto start = std::chrono::high_resolution_clock::now();
            const bool ok = append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            const auto end = std::chrono::high_resolution_clock::now();
            const uint64_t ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            KVCacheProfiler::record(KVCacheOpType::APPEND, ns, static_cast<uint64_t>(num_tokens), 0);
            return ok;
        }
    }

    // =========================================================================
    // CUDARingKVCache<Precision>::get_k() / get_v() implementations
    // =========================================================================
    // These create GpuTensorView wrappers around the cached device buffers.
    // Views are stored in tensor_views_ and reused on subsequent calls.

    template <ActivationPrecision Precision>
    ITensor *CUDARingKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[CUDARingKVCache::get_k] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via get_kv_for_attention
        const void *d_k = nullptr;
        const void *d_v = nullptr;
        int kv_len = 0;

        if (!get_kv_for_attention(layer, seq_idx, &d_k, &d_v, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_k] get_kv_for_attention failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        if (!d_k || kv_len == 0)
        {
            // Empty cache - valid state, return nullptr
            return nullptr;
        }

        // Convert ActivationPrecision to TensorType at compile time
        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        // Create or update the view
        auto &view = tensor_views_[layer][seq_idx][0]; // Index 0 = K

        // Check if view needs to be created or updated (pointer or size changed)
        if (!view ||
            view->gpu_data_ptr() != d_k ||
            view->rows() != static_cast<size_t>(kv_len))
        {
            // Create new view wrapping the device buffer
            view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_k), // GpuTensorView needs non-const for interface
                static_cast<size_t>(kv_len),
                view_cols,
                tensor_type,
                device_id_);

            LOG_TRACE("[CUDARingKVCache::get_k] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CUDARingKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<CUDARingKVCache<Precision> *>(this)->get_k(layer, seq_idx);
    }

    template <ActivationPrecision Precision>
    ITensor *CUDARingKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[CUDARingKVCache::get_v] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via get_kv_for_attention
        const void *d_k = nullptr;
        const void *d_v = nullptr;
        int kv_len = 0;

        if (!get_kv_for_attention(layer, seq_idx, &d_k, &d_v, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_v] get_kv_for_attention failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        if (!d_v || kv_len == 0)
        {
            // Empty cache - valid state, return nullptr
            return nullptr;
        }

        // Convert ActivationPrecision to TensorType at compile time
        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }();

        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        // Create or update the view
        auto &view = tensor_views_[layer][seq_idx][1]; // Index 1 = V

        // Check if view needs to be created or updated (pointer or size changed)
        if (!view ||
            view->gpu_data_ptr() != d_v ||
            view->rows() != static_cast<size_t>(kv_len))
        {
            // Create new view wrapping the device buffer
            view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_v), // GpuTensorView needs non-const for interface
                static_cast<size_t>(kv_len),
                view_cols,
                tensor_type,
                device_id_);

            LOG_TRACE("[CUDARingKVCache::get_v] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CUDARingKVCache<Precision>::get_v(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<CUDARingKVCache<Precision> *>(this)->get_v(layer, seq_idx);
    }

    // Explicit template instantiations
    template ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP32>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::FP16>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::BF16>::get_v(int, int) const;

    template ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_k(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_k(int, int) const;
    template ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_v(int, int);
    template const ITensor *CUDARingKVCache<ActivationPrecision::Q8_1>::get_v(int, int) const;

} // namespace llaminar2
