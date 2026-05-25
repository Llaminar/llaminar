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
#include "../../../tensors/TensorClasses.h"
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
        (void)layer;
        (void)seq_idx;
        (void)K;
        (void)V;
        (void)num_tokens;
        LOG_ERROR("[ICUDARingKVCache::append(ITensor)] Explicit CUDA stream required; use appendWithStream()");
        return false;
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
        if (!gpu_stream)
        {
            LOG_ERROR("[ICUDARingKVCache::appendWithStream] Null CUDA stream is not allowed");
            return false;
        }

        const auto target = DeviceId::cuda(device_id());

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k)
        {
            auto *k_mut = const_cast<ITensor *>(K);
            auto *k_tensor = dynamic_cast<TensorBase *>(k_mut);
            if (!(k_tensor ? k_tensor->ensureOnDevice(target, gpu_stream) : k_mut->ensureOnDevice(target)))
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
            auto *v_tensor = dynamic_cast<TensorBase *>(v_mut);
            if (!(v_tensor ? v_tensor->ensureOnDevice(target, gpu_stream) : v_mut->ensureOnDevice(target)))
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

        // NOTE: We use k_precision() for both K and V conversion paths below.
        // For symmetric caches (k_precision() == v_precision(), which is the
        // default), this is correct. For asymmetric caches such as
        // CUDARingKVCacheTQ (TQ8 K + TQ4 V) the TQ path is not handled by
        // these blocks at all - those tensors fall through to the native
        // append() call below. If a future asymmetric FP16/Q8_1 cache is
        // added, the conversion paths must be split into separate K and V
        // gates using k_precision() / v_precision() respectively.
        if (k_precision() == ActivationPrecision::FP16 &&
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

        if (k_precision() == ActivationPrecision::Q8_1 &&
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

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. CUDAHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_k] get_kv_typed failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        const void *d_k = d_k_typed;
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

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. CUDAHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[CUDARingKVCache::get_v] get_kv_typed failed for layer="
                     << layer << " seq_idx=" << seq_idx);
            return nullptr;
        }

        const void *d_v = d_v_typed;
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

    // =========================================================================
    // get_kv(): single-pass K+V ITensor access
    // =========================================================================

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv(
        int layer, int seq_idx,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        // Single call gets both K and V device pointers (non-virtual to avoid
        // double-mapping when called from derived class with layer remapping)
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
            return false;

        const void *d_k = d_k_typed;
        const void *d_v = d_v_typed;

        if (kv_len == 0 || !d_k || !d_v)
        {
            if (out_kv_len)
                *out_kv_len = 0;
            return true;
        }

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
        const size_t rows = static_cast<size_t>(kv_len);

        // Update K view (index 0)
        auto &k_view = tensor_views_[layer][seq_idx][0];
        if (!k_view || k_view->gpu_data_ptr() != d_k || k_view->rows() != rows)
        {
            k_view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_k), rows, view_cols, tensor_type, device_id_);
        }

        // Update V view (index 1)
        auto &v_view = tensor_views_[layer][seq_idx][1];
        if (!v_view || v_view->gpu_data_ptr() != d_v || v_view->rows() != rows)
        {
            v_view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_v), rows, view_cols, tensor_type, device_id_);
        }

        if (out_k)
            *out_k = k_view.get();
        if (out_v)
            *out_v = v_view.get();
        if (out_kv_len)
            *out_kv_len = kv_len;
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv(
        int layer, int seq_idx,
        const ITensor **out_k, const ITensor **out_v,
        int *out_kv_len) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        ITensor *k = nullptr;
        ITensor *v = nullptr;
        bool ok = const_cast<CUDARingKVCache<Precision> *>(this)->get_kv(layer, seq_idx, &k, &v, out_kv_len);
        if (ok)
        {
            if (out_k)
                *out_k = k;
            if (out_v)
                *out_v = v;
        }
        return ok;
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

    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv(int, int, ITensor **, ITensor **, int *);
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv(int, int, const ITensor **, const ITensor **, int *) const;

    // =========================================================================
    // get_kv_converted(): FP16 shadow buffers with optional RoPE
    // =========================================================================

    extern "C" bool cuda_rope_apply_fp16(
        __half *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    extern "C" bool cuda_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::ensureRoPEShadow(int layer, int seq_idx) const
    {
        // Lazy init the outer vectors
        if (rope_shadows_.empty())
        {
            rope_shadows_.resize(n_layers_);
            for (auto &layer_shadows : rope_shadows_)
                layer_shadows.resize(batch_size_);
        }

        auto &shadow = rope_shadows_[layer][seq_idx];
        if (!shadow.d_K)
        {
            const size_t buf_bytes = static_cast<size_t>(max_seq_len_) * kv_dim_ * sizeof(__half);
            cudaMalloc(&shadow.d_K, buf_bytes);
            cudaMalloc(&shadow.d_V, buf_bytes);
            shadow.converted_count = 0;
            shadow.last_head = -1;
            shadow.rope_applied = false;
        }
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::invalidateRoPEShadow(int layer, int seq_idx) const
    {
        if (rope_shadows_.empty())
            return;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        auto &shadow = rope_shadows_[layer][seq_idx];
        shadow.converted_count = 0;
        shadow.last_head = -1;
        shadow.rope_applied = false;
        shadow.k_view.reset();
        shadow.v_view.reset();
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        (void)target; // We always produce FP16 on GPU

        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        const auto &entry = entries_[layer][seq_idx];
        if (entry.count == 0)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return true;
        }

        // If no RoPE is requested, the raw cache tensors already have the
        // required representation.
        // Use qualified call to avoid virtual dispatch — CUDAHybridRingKVCache
        // overrides get_kv() with layer remapping, but the layer index passed here
        // has already been remapped by the hybrid override of get_kv_converted().
        const bool want_rope = (rope && rope->rope_theta > 0.0f);
        if (!want_rope)
        {
            return CUDARingKVCache::get_kv(layer, seq_idx, out_k, out_v, out_kv_len);
        }

        cudaSetDevice(device_id_);
        const cudaStream_t stream = getEffectiveStream(
            rope ? static_cast<cudaStream_t>(rope->gpu_stream) : nullptr);

        ensureRoPEShadow(layer, seq_idx);
        auto &shadow = rope_shadows_[layer][seq_idx];

        // Incremental update: only process new tokens since last call.
        // Detect whether we can do an incremental update or need a full rebuild.
        // Full rebuild if: first call, eviction happened (count shrunk), or ring reset.
        const int new_tokens = entry.count - shadow.converted_count;
        const bool need_full_rebuild = (shadow.converted_count == 0 ||
                                        new_tokens < 0 ||
                                        shadow.converted_count > entry.count);

        if constexpr (Precision == ActivationPrecision::FP16)
        {
            if (need_full_rebuild)
            {
                // Full rebuild: linearize all K/V to shadow + apply RoPE to all K
                int kv_len = 0;
                // Qualified call avoids virtual dispatch — prevents CUDAHybridRingKVCache
                // from double-remapping the already-remapped layer index.
                if (!CUDARingKVCache::linearize_to(layer, seq_idx, shadow.d_K, shadow.d_V, &kv_len, stream))
                    return false;

                cuda_rope_apply_fp16(shadow.d_K, kv_len, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start, stream,
                                     rope->rope_dim);

                shadow.converted_count = kv_len;
            }
            else if (new_tokens > 0)
            {
                // Incremental: linearize all data from ring buffer into shadow.
                // IMPORTANT: linearize_to overwrites the ENTIRE shadow with fresh
                // (non-RoPE'd) data from the ring buffer. Unlike FP32/Q8_1/BF16
                // paths which use a separate scratch buffer and only copy new tokens
                // to the shadow, the FP16 path writes directly to the shadow.
                // Therefore we must re-apply RoPE to ALL tokens, not just the new ones.
                int kv_len = 0;
                if (!CUDARingKVCache::linearize_to(layer, seq_idx, shadow.d_K, shadow.d_V, &kv_len, stream))
                    return false;

                // Apply RoPE to all tokens (linearize overwrote previously RoPE'd data)
                cuda_rope_apply_fp16(shadow.d_K, kv_len, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start, stream,
                                     rope->rope_dim);

                shadow.converted_count = kv_len;
            }
            // else: new_tokens == 0, shadow is already up-to-date
        }
        else if constexpr (Precision == ActivationPrecision::FP32)
        {
            // FP32 cache → linearize to scratch, apply RoPE, convert to FP16 shadow
            const size_t row_bytes = static_cast<size_t>(kv_dim_) * sizeof(float);
            const size_t total_bytes = static_cast<size_t>(entry.count) * row_bytes;

            if (!ensureConvScratch(total_bytes))
                return false;

            auto *d_temp_k = static_cast<float *>(conv_scratch_k_);
            auto *d_temp_v = static_cast<float *>(conv_scratch_v_);

            int kv_len = 0;
            if (!CUDARingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            if (need_full_rebuild)
            {
                cuda_rope_apply_fp32(d_temp_k, kv_len, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start, stream,
                                     rope->rope_dim);

                cuda_convert_tensor_to_fp16(d_temp_k, TensorType::FP32,
                                            reinterpret_cast<uint16_t *>(shadow.d_K),
                                            kv_len * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(d_temp_v, TensorType::FP32,
                                            reinterpret_cast<uint16_t *>(shadow.d_V),
                                            kv_len * kv_dim_, stream);
            }
            else if (new_tokens > 0)
            {
                // RoPE only new tokens in scratch, convert only new tokens to shadow
                const int old_count = shadow.converted_count;
                float *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_dim_;
                float *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_dim_;

                cuda_rope_apply_fp32(new_k_start, new_tokens, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start + old_count, stream,
                                     rope->rope_dim);

                __half *shadow_k_new = reinterpret_cast<__half *>(shadow.d_K) + static_cast<size_t>(old_count) * kv_dim_;
                __half *shadow_v_new = reinterpret_cast<__half *>(shadow.d_V) + static_cast<size_t>(old_count) * kv_dim_;

                cuda_convert_tensor_to_fp16(new_k_start, TensorType::FP32,
                                            reinterpret_cast<uint16_t *>(shadow_k_new),
                                            new_tokens * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(new_v_start, TensorType::FP32,
                                            reinterpret_cast<uint16_t *>(shadow_v_new),
                                            new_tokens * kv_dim_, stream);
            }

            shadow.converted_count = kv_len;
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            // Q8_1 cache → linearize to scratch, dequant to FP16 shadow, RoPE
            const size_t q8_row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(Q8_1Block);
            const size_t q8_total = static_cast<size_t>(entry.count) * q8_row_bytes;

            if (!ensureConvScratch(q8_total))
                return false;

            auto *d_temp_k = static_cast<Q8_1Block *>(conv_scratch_k_);
            auto *d_temp_v = static_cast<Q8_1Block *>(conv_scratch_v_);

            int kv_len = 0;
            if (!CUDARingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            if (need_full_rebuild)
            {
                cuda_convert_tensor_to_fp16(d_temp_k, TensorType::Q8_1,
                                            reinterpret_cast<uint16_t *>(shadow.d_K),
                                            kv_len * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(d_temp_v, TensorType::Q8_1,
                                            reinterpret_cast<uint16_t *>(shadow.d_V),
                                            kv_len * kv_dim_, stream);

                cuda_rope_apply_fp16(shadow.d_K, kv_len, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start, stream,
                                     rope->rope_dim);
            }
            else if (new_tokens > 0)
            {
                // Dequant only new tokens, RoPE only new tokens
                const int old_count = shadow.converted_count;
                Q8_1Block *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_storage_dim_;
                Q8_1Block *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_storage_dim_;

                __half *shadow_k_new = shadow.d_K + static_cast<size_t>(old_count) * kv_dim_;
                __half *shadow_v_new = shadow.d_V + static_cast<size_t>(old_count) * kv_dim_;

                cuda_convert_tensor_to_fp16(new_k_start, TensorType::Q8_1,
                                            reinterpret_cast<uint16_t *>(shadow_k_new),
                                            new_tokens * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(new_v_start, TensorType::Q8_1,
                                            reinterpret_cast<uint16_t *>(shadow_v_new),
                                            new_tokens * kv_dim_, stream);

                cuda_rope_apply_fp16(shadow_k_new, new_tokens, n_kv_heads_, head_dim_,
                                     rope->rope_theta,
                                     rope->position_start + old_count, stream,
                                     rope->rope_dim);
            }

            shadow.converted_count = kv_len;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            // BF16 → linearize to scratch, convert to FP16 shadow, RoPE
            const size_t bf16_bytes = static_cast<size_t>(entry.count) * kv_dim_ * sizeof(__nv_bfloat16);

            if (!ensureConvScratch(bf16_bytes))
                return false;

            auto *d_temp_k = static_cast<__nv_bfloat16 *>(conv_scratch_k_);
            auto *d_temp_v = static_cast<__nv_bfloat16 *>(conv_scratch_v_);

            int kv_len = 0;
            if (!CUDARingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            if (need_full_rebuild)
            {
                cuda_convert_tensor_to_fp16(d_temp_k, TensorType::BF16,
                                            reinterpret_cast<uint16_t *>(shadow.d_K),
                                            kv_len * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(d_temp_v, TensorType::BF16,
                                            reinterpret_cast<uint16_t *>(shadow.d_V),
                                            kv_len * kv_dim_, stream);

                cuda_rope_apply_fp16(shadow.d_K, kv_len, n_kv_heads_, head_dim_,
                                     rope->rope_theta, rope->position_start, stream,
                                     rope->rope_dim);
            }
            else if (new_tokens > 0)
            {
                const int old_count = shadow.converted_count;
                __nv_bfloat16 *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_dim_;
                __nv_bfloat16 *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_dim_;

                __half *shadow_k_new = shadow.d_K + static_cast<size_t>(old_count) * kv_dim_;
                __half *shadow_v_new = shadow.d_V + static_cast<size_t>(old_count) * kv_dim_;

                cuda_convert_tensor_to_fp16(new_k_start, TensorType::BF16,
                                            reinterpret_cast<uint16_t *>(shadow_k_new),
                                            new_tokens * kv_dim_, stream);
                cuda_convert_tensor_to_fp16(new_v_start, TensorType::BF16,
                                            reinterpret_cast<uint16_t *>(shadow_v_new),
                                            new_tokens * kv_dim_, stream);

                cuda_rope_apply_fp16(shadow_k_new, new_tokens, n_kv_heads_, head_dim_,
                                     rope->rope_theta,
                                     rope->position_start + old_count, stream,
                                     rope->rope_dim);
            }

            shadow.converted_count = kv_len;
        }

        shadow.last_head = entry.head;
        shadow.rope_applied = true;

        // Create/update GpuTensorViews
        if (!shadow.k_view || shadow.k_view->shape()[0] != static_cast<size_t>(shadow.converted_count))
        {
            shadow.k_view = std::make_unique<GpuTensorView>(
                shadow.d_K, shadow.converted_count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        if (!shadow.v_view || shadow.v_view->shape()[0] != static_cast<size_t>(shadow.converted_count))
        {
            shadow.v_view = std::make_unique<GpuTensorView>(
                shadow.d_V, shadow.converted_count, kv_dim_,
                TensorType::FP16, device_id_);
        }

        if (out_k)
            *out_k = shadow.k_view.get();
        if (out_v)
            *out_v = shadow.v_view.get();
        if (out_kv_len)
            *out_kv_len = shadow.converted_count;

        return true;
    }

    // Explicit template instantiations for get_kv_converted
    template bool CUDARingKVCache<ActivationPrecision::FP32>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::FP16>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::BF16>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);
    template bool CUDARingKVCache<ActivationPrecision::Q8_1>::get_kv_converted(int, int, ActivationPrecision, ITensor **, ITensor **, int *, const KVReadParams *);

    // Explicit template instantiations for shadow helpers
    template void CUDARingKVCache<ActivationPrecision::FP32>::ensureRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::FP16>::ensureRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::BF16>::ensureRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::Q8_1>::ensureRoPEShadow(int, int) const;

    template void CUDARingKVCache<ActivationPrecision::FP32>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::FP16>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::BF16>::invalidateRoPEShadow(int, int) const;
    template void CUDARingKVCache<ActivationPrecision::Q8_1>::invalidateRoPEShadow(int, int) const;

} // namespace llaminar2
