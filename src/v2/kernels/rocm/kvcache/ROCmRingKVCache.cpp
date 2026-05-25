/**
 * @file ROCmRingKVCache.cpp
 * @brief ROCm Ring Buffer KV Cache implementation (C++ adapter)
 * @author Llaminar Team
 * @date January 2026
 *
 * This file is compiled by the regular C++ compiler (not hipcc) so it can
 * include heavy headers like CPUTensors.h.
 *
 * The HIP kernels are defined in ROCmRingKVCacheKernels.hip and linked
 * via extern "C" declarations.
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include "ROCmRingKVCache.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../ROCmKernelBase.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/KVCacheProfiler.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // External HIP Kernel Declarations
    // =========================================================================

    // FP32 kernel wrappers
    extern "C" void hip_ring_append_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_fp32(
        float *d_out,
        const float *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_gather_batched_fp32(
        float *d_K_out, float *d_V_out,
        const float *const *d_K_caches, const float *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream);

    // FP16 kernel wrappers
    extern "C" void hip_ring_append_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_fp16(
        _Float16 *d_out,
        const _Float16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_gather_batched_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const _Float16 *const *d_K_caches, const _Float16 *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream);

    // BF16 kernel wrappers
    extern "C" void hip_ring_append_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_bf16(
        hip_bfloat16 *d_out,
        const hip_bfloat16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_gather_batched_bf16(
        hip_bfloat16 *d_K_out, hip_bfloat16 *d_V_out,
        const hip_bfloat16 *const *d_K_caches, const hip_bfloat16 *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream);

    // Q8_1 kernel wrappers (block-based)
    extern "C" void hip_ring_append_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        int head, int max_seq_len, int kv_blocks, int num_tokens,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_q8_1(
        Q8_1Block *d_out,
        const Q8_1Block *d_cache,
        int tail, int count, int max_seq_len, int kv_blocks,
        hipStream_t stream);

    extern "C" void hip_ring_gather_batched_q8_1(
        Q8_1Block *d_K_out, Q8_1Block *d_V_out,
        const Q8_1Block *const *d_K_caches, const Q8_1Block *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_blocks,
        hipStream_t stream);

    // Dynamic head append wrappers (graph-capturable)
    extern "C" void hip_ring_append_dynamic_fp32(
        float *, float *, const float *, const float *,
        const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_fp16(
        _Float16 *, _Float16 *, const _Float16 *, const _Float16 *,
        const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_bf16(
        hip_bfloat16 *, hip_bfloat16 *, const hip_bfloat16 *, const hip_bfloat16 *,
        const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        const int *, int, int, int, hipStream_t);

    extern "C" bool hip_convert_tensor_to_fp16(
        const void *d_src,
        TensorType src_type,
        uint16_t *d_dst,
        int count,
        hipStream_t stream);

    extern "C" bool hip_convert_tensor_to_q8_1(
        const void *d_src,
        TensorType src_type,
        Q8_1Block *d_dst,
        int rows,
        int cols,
        hipStream_t stream);

    // =========================================================================
    // IROCmRingKVCache destructor + conversion scratch buffer management
    // =========================================================================

    IROCmRingKVCache::~IROCmRingKVCache()
    {
        freeConvScratch();
    }

    bool IROCmRingKVCache::ensureConvScratch(size_t bytes)
    {
        if (bytes <= conv_scratch_capacity_)
            return true;

        // Grow to requested size (round up to 4KB for alignment)
        const size_t alloc_size = (bytes + 4095) & ~size_t(4095);

        void *new_k = nullptr;
        void *new_v = nullptr;
        if (hipMalloc(&new_k, alloc_size) != hipSuccess ||
            hipMalloc(&new_v, alloc_size) != hipSuccess)
        {
            if (new_k)
                hipFree(new_k);
            if (new_v)
                hipFree(new_v);
            LOG_ERROR("[IROCmRingKVCache] Failed to allocate conversion scratch buffers ("
                      << alloc_size << " bytes each)");
            return false;
        }

        // Free old buffers
        if (conv_scratch_k_)
            hipFree(conv_scratch_k_);
        if (conv_scratch_v_)
            hipFree(conv_scratch_v_);

        conv_scratch_k_ = new_k;
        conv_scratch_v_ = new_v;
        conv_scratch_capacity_ = alloc_size;

        LOG_DEBUG("[IROCmRingKVCache] Allocated conversion scratch: "
                  << alloc_size << " bytes each (" << (alloc_size * 2 / 1024) << " KB total)");
        return true;
    }

    void IROCmRingKVCache::freeConvScratch()
    {
        if (conv_scratch_k_)
        {
            hipFree(conv_scratch_k_);
            conv_scratch_k_ = nullptr;
        }
        if (conv_scratch_v_)
        {
            hipFree(conv_scratch_v_);
            conv_scratch_v_ = nullptr;
        }
        conv_scratch_capacity_ = 0;
    }

    // =========================================================================
    // IROCmRingKVCache::append(ITensor*) implementation
    // =========================================================================

    bool IROCmRingKVCache::append(int layer, int seq_idx,
                                  const ITensor *K, const ITensor *V,
                                  int num_tokens)
    {
        (void)layer;
        (void)seq_idx;
        (void)K;
        (void)V;
        (void)num_tokens;
        LOG_ERROR("[IROCmRingKVCache::append(ITensor)] Explicit HIP stream required; use appendWithStream()");
        return false;
    }

    bool IROCmRingKVCache::appendWithStream(int layer, int seq_idx,
                                            const ITensor *K, const ITensor *V,
                                            int num_tokens, void *gpu_stream)
    {
        if (!K || !V)
        {
            LOG_DEBUG("[IROCmRingKVCache::appendWithStream] Null K or V tensor");
            return false;
        }
        if (!gpu_stream)
        {
            LOG_ERROR("[IROCmRingKVCache::appendWithStream] Null HIP stream is not allowed");
            return false;
        }

        const auto target = DeviceId::rocm(device_id());

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k)
        {
            auto *k_mut = const_cast<ITensor *>(K);
            auto *k_tensor = dynamic_cast<TensorBase *>(k_mut);
            if (!(k_tensor ? k_tensor->ensureOnDevice(target, gpu_stream) : k_mut->ensureOnDevice(target)))
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Failed to ensure K on "
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
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Failed to ensure V on "
                          << target.toString());
                return false;
            }
            d_v = V->gpu_data_ptr();
        }

        if (!d_k || !d_v)
        {
            LOG_ERROR("[IROCmRingKVCache::appendWithStream] K or V tensor lacks GPU data after ensureOnDevice().");
            return false;
        }

        const auto stream = static_cast<hipStream_t>(gpu_stream);

        if (precision() == ActivationPrecision::FP16 &&
            (K->native_type() != TensorType::FP16 || V->native_type() != TensorType::FP16))
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Invalid K/V shape for FP16 conversion");
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
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Failed to ensure FP16 conversion scratch");
                return false;
            }
            auto *d_k_fp16 = static_cast<uint16_t *>(conv_scratch_k_);
            auto *d_v_fp16 = static_cast<uint16_t *>(conv_scratch_v_);

            const auto alloc_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: FP16 conversion kernels ---
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const bool k_ok = hip_convert_tensor_to_fp16(
                d_k,
                K->native_type(),
                d_k_fp16,
                elements,
                stream);
            const bool v_ok = hip_convert_tensor_to_fp16(
                d_v,
                V->native_type(),
                d_v_fp16,
                elements,
                stream);
            if (!k_ok || !v_ok)
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] GPU FP16 conversion failed");
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
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Invalid K/V shape for Q8_1 conversion");
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
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Failed to ensure Q8_1 conversion scratch");
                return false;
            }
            auto *d_k_q8 = static_cast<Q8_1Block *>(conv_scratch_k_);
            auto *d_v_q8 = static_cast<Q8_1Block *>(conv_scratch_v_);

            const auto alloc_end = std::chrono::high_resolution_clock::now();

            // --- Profiling: Q8_1 conversion kernels ---
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const bool k_ok = hip_convert_tensor_to_q8_1(
                d_k,
                K->native_type(),
                d_k_q8,
                num_tokens,
                kv_dim,
                stream);
            const bool v_ok = hip_convert_tensor_to_q8_1(
                d_v,
                V->native_type(),
                d_v_q8,
                num_tokens,
                kv_dim,
                stream);
            if (!k_ok || !v_ok)
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] GPU Q8_1 conversion failed");
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
    // ROCmRingKVCache Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    ROCmRingKVCache<Precision>::ROCmRingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id)
        : IROCmRingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(n_kv_heads), kv_head_start_(0),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (n_kv_heads * head_dim)),
          is_sharded_(false), device_ctx_(nullptr)
    {
        LOG_DEBUG("[ROCmRingKVCache] Creating cache: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", kv_dim=" << kv_dim_
                  << ", precision=" << static_cast<int>(Precision));

        HipDeviceGuard::setDevice(device_id_);

        allocate_all_entries();
    }

    template <ActivationPrecision Precision>
    ROCmRingKVCache<Precision>::ROCmRingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, IWorkerGPUContext *ctx)
        : IROCmRingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, n_kv_heads * head_dim, 0),
          local_n_kv_heads_(n_kv_heads), kv_head_start_(0),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (n_kv_heads * head_dim)),
          is_sharded_(false), device_ctx_(nullptr)
    {
        if (!ctx)
        {
            throw std::runtime_error("[ROCmRingKVCache] Device context is null");
        }
        if (!ctx->isInitialized())
        {
            throw std::runtime_error("[ROCmRingKVCache] Device context is not initialized");
        }

        device_ctx_ = ctx;
        device_id_ = ctx->deviceOrdinal();

        LOG_DEBUG("[ROCmRingKVCache] Creating cache with device context: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", kv_dim=" << kv_dim_
                  << ", device=" << device_id_
                  << ", precision=" << static_cast<int>(Precision));

        HipDeviceGuard::setDevice(device_id_);

        allocate_all_entries();
    }

    template <ActivationPrecision Precision>
    ROCmRingKVCache<Precision>::ROCmRingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_id)
        : IROCmRingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, local_n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((local_n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (local_n_kv_heads * head_dim)),
          is_sharded_(local_n_kv_heads != n_kv_heads), device_ctx_(nullptr)
    {
        LOG_DEBUG("[ROCmRingKVCache] Creating sharded cache: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", total_kv_heads=" << n_kv_heads
                  << ", local_kv_heads=" << local_n_kv_heads << ", kv_head_start=" << kv_head_start
                  << ", local_kv_dim=" << kv_dim_
                  << ", precision=" << static_cast<int>(Precision));

        HipDeviceGuard::setDevice(device_id_);

        allocate_all_entries();
    }

    template <ActivationPrecision Precision>
    ROCmRingKVCache<Precision>::ROCmRingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, IWorkerGPUContext *ctx)
        : IROCmRingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, local_n_kv_heads * head_dim, 0),
          local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((local_n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (local_n_kv_heads * head_dim)),
          is_sharded_(local_n_kv_heads != n_kv_heads), device_ctx_(nullptr)
    {
        if (!ctx)
        {
            throw std::runtime_error("[ROCmRingKVCache] Device context is null");
        }
        if (!ctx->isInitialized())
        {
            throw std::runtime_error("[ROCmRingKVCache] Device context is not initialized");
        }

        device_ctx_ = ctx;
        device_id_ = ctx->deviceOrdinal();

        LOG_DEBUG("[ROCmRingKVCache] Creating sharded cache with device context: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", total_kv_heads=" << n_kv_heads
                  << ", local_kv_heads=" << local_n_kv_heads << ", kv_head_start=" << kv_head_start
                  << ", local_kv_dim=" << kv_dim_
                  << ", device=" << device_id_
                  << ", precision=" << static_cast<int>(Precision));

        HipDeviceGuard::setDevice(device_id_);

        allocate_all_entries();
    }

    template <ActivationPrecision Precision>
    ROCmRingKVCache<Precision>::~ROCmRingKVCache()
    {
        // Check if HIP runtime is shutting down
        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_));
        if (set_err == hipErrorDeinitialized || set_err == hipErrorNoDevice)
        {
            // Runtime is shutting down, skip cleanup
            return;
        }

        // Free RoPE shadow buffers
        for (auto &layer_shadows : rope_shadows_)
        {
            for (auto &shadow : layer_shadows)
            {
                if (shadow.d_K)
                    hipFree(shadow.d_K);
                if (shadow.d_V)
                    hipFree(shadow.d_V);
            }
        }
        rope_shadows_.clear();

        if (pool_base_)
        {
            // All entries point into the single pool — free it once
            free_pool();
            // Null out entry pointers (they're dangling now)
            for (auto &layer_entries : entries_)
            {
                for (auto &entry : layer_entries)
                {
                    entry.d_K = nullptr;
                    entry.d_V = nullptr;
                }
            }
        }
        else
        {
            // Fallback: individually allocated entries
            for (auto &layer_entries : entries_)
            {
                for (auto &entry : layer_entries)
                {
                    free_entry(entry);
                }
            }
        }
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::allocate_pool()
    {
        size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                             static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        size_t total_entries = static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        // 2 buffers per entry: persistent K and V. Wrapped-ring reads borrow
        // the cache-level conversion scratch instead of reserving per-entry
        // linearization buffers for every full-attention layer.
        pool_size_ = total_entries * 2 * buffer_size;

        if (pool_size_ == 0)
        {
            pool_base_ = nullptr;
            return;
        }

        hipError_t err = hipMalloc(&pool_base_, pool_size_);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache] Failed to allocate pooled KV cache ("
                      << (pool_size_ / (1024 * 1024)) << " MB): "
                      << hipGetErrorString(err));
            pool_base_ = nullptr;
            pool_size_ = 0;
            return;
        }

        // Zero-initialize the entire pool
        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        hipMemsetAsync(pool_base_, 0, pool_size_, init_stream);
        hipStreamSynchronize(init_stream);

        LOG_DEBUG("[ROCmRingKVCache] Pooled KV cache: 1 hipMalloc for "
                  << total_entries << " entries × 2 buffers = "
                  << (pool_size_ / (1024 * 1024)) << " MB (replaced "
                  << (total_entries * 2) << " individual hipMalloc calls)");
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::free_pool()
    {
        if (pool_base_)
        {
            hipError_t err = hipFree(pool_base_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(pool_base_) failed: %s\n",
                        hipGetErrorString(err));
            }
            pool_base_ = nullptr;
            pool_size_ = 0;
        }
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::assign_entry_from_pool(EntryT &entry, int linear_index)
    {
        size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                             static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        char *base = static_cast<char *>(pool_base_);
        size_t entry_offset = static_cast<size_t>(linear_index) * 2 * buffer_size;

        entry.d_K = reinterpret_cast<DataT *>(base + entry_offset + 0 * buffer_size);
        entry.d_V = reinterpret_cast<DataT *>(base + entry_offset + 1 * buffer_size);
        entry.head = 0;
        entry.count = 0;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::allocate_all_entries()
    {
        // Single pooled allocation for all KV cache entries
        allocate_pool();

        entries_.resize(n_layers_);
        int linear_idx = 0;
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer].resize(batch_size_);
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                if (pool_base_)
                {
                    assign_entry_from_pool(entries_[layer][seq], linear_idx++);
                }
                else
                {
                    allocate_entry(entries_[layer][seq]); // Fallback if pool alloc failed
                }
            }
        }

        // Initialize tensor_views_ storage for get_k()/get_v() wrappers
        tensor_views_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            tensor_views_[layer].resize(batch_size_);
        }

        LOG_DEBUG("[ROCmRingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 2 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total");

        allocateDeviceParams(); // Base class method (ROCmRingKVCacheBase)
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::allocate_entry(EntryT &entry)
    {
        size_t buffer_size = static_cast<size_t>(max_seq_len_) * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);

        // Main K/V buffers
        hipMalloc(&entry.d_K, buffer_size);
        hipMalloc(&entry.d_V, buffer_size);

        // Initialize state
        entry.head = 0;
        entry.count = 0;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::free_entry(EntryT &entry)
    {
        if (entry.d_K)
        {
            hipError_t err = hipFree(entry.d_K);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(d_K) failed: %s\n", hipGetErrorString(err));
            }
        }
        if (entry.d_V)
        {
            hipError_t err = hipFree(entry.d_V);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(d_V) failed: %s\n", hipGetErrorString(err));
            }
        }
        entry.d_K = nullptr;
        entry.d_V = nullptr;
    }

    // get_cached_tokens(), get_head_position(), is_wrapped() are now
    // provided by ROCmRingKVCacheBase via entry accessor overrides.

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::append(
        int layer, int seq_idx,
        const void *d_k, const void *d_v,
        int num_tokens, hipStream_t stream)
    {
        return append_typed(layer, seq_idx,
                            static_cast<const DataT *>(d_k),
                            static_cast<const DataT *>(d_v),
                            num_tokens, stream);
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::append_typed(
        int layer, int seq_idx,
        const DataT *d_k, const DataT *d_v,
        int num_tokens, hipStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::append] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        EntryT &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();

        // Track how many tokens to skip from input (earliest tokens that would be overwritten)
        int tokens_to_skip = 0;
        int tokens_to_write = num_tokens;

        // Check if we would exceed capacity (ring buffer overwrites oldest)
        if (!capture_active && entry.count + num_tokens > max_seq_len_)
        {
            // Calculate how many existing tokens to evict
            int to_evict = entry.count + num_tokens - max_seq_len_;

            // If eviction would be more than current count, we're also skipping input tokens
            if (to_evict > entry.count)
            {
                // Skip input tokens that would be immediately overwritten
                tokens_to_skip = to_evict - entry.count;
                tokens_to_write = num_tokens - tokens_to_skip;
                // Count both existing cache tokens AND skipped input tokens as evicted
                total_evicted_ += entry.count + tokens_to_skip;
                entry.count = 0;
                LOG_DEBUG("[ROCmRingKVCache::append] Skipping " << tokens_to_skip
                                                                << " input tokens, writing " << tokens_to_write);
            }
            else
            {
                entry.count -= to_evict;
                total_evicted_ += to_evict;
                LOG_DEBUG("[ROCmRingKVCache::append] Auto-evicted " << to_evict << " tokens");
            }
        }

        // Launch append kernel with adjusted pointers (skip earliest tokens)
        if (tokens_to_write > 0)
        {
            const DataT *d_k_adjusted = d_k + static_cast<size_t>(tokens_to_skip) * kv_storage_dim_;
            const DataT *d_v_adjusted = d_v + static_cast<size_t>(tokens_to_skip) * kv_storage_dim_;

            // Use dynamic head params only while recording a graph. In regular
            // execution, the scalar-head append path avoids an unnecessary
            // pinned-host H2D dependency and snapshots the host head value at
            // launch time. Captured graphs still need the dynamic path so replay
            // can refresh the head through setDynamicHead() before the captured
            // H2D copy runs.
            if (capture_active && d_head_params_ && h_head_params_ && stream)
            {
                int idx = layer * batch_size_ + seq_idx;
                h_head_params_[idx] = entry.head;
                hipMemcpyAsync(&d_head_params_[idx], &h_head_params_[idx],
                               sizeof(int), hipMemcpyHostToDevice, stream);
                launch_append_kernel_dynamic(entry, d_k_adjusted, d_v_adjusted,
                                             &d_head_params_[idx], tokens_to_write, stream);
            }
            else
            {
                // Fallback: scalar head argument (non-graph path)
                launch_append_kernel(entry, d_k_adjusted, d_v_adjusted, tokens_to_write, stream);
            }
        }

        if (!capture_active)
        {
            // Update host metadata only for real execution. Captured graph
            // launches use replay callbacks to advance by real (unpadded) tokens.
            entry.head = (entry.head + tokens_to_write) % max_seq_len_;
            entry.count += tokens_to_write;
        }

        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::appendConvertedWithStream(
        int layer, int seq_idx,
        const void *d_k_src, const void *d_v_src,
        TensorType src_type,
        int num_tokens, hipStream_t stream)
    {
        (void)layer;
        (void)seq_idx;
        (void)d_k_src;
        (void)d_v_src;
        (void)src_type;
        (void)num_tokens;
        (void)stream;
        return false;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_append_kernel(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        int num_tokens, hipStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            hip_ring_append_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_append_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_append_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_append_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
    }

    // allocate_device_params(), free_device_params(), setDynamicHead(), advanceHead()
    // are now provided by ROCmRingKVCacheBase.

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_append_kernel_dynamic(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        const int *d_head, int num_tokens, hipStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            hip_ring_append_dynamic_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_append_dynamic_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_append_dynamic_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_append_dynamic_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_for_attention(
        int layer, int seq_idx,
        const void **d_k_out, const void **d_v_out,
        int *kv_len, hipStream_t stream)
    {
        const DataT *k_typed;
        const DataT *v_typed;
        bool result = get_kv_typed(layer, seq_idx, &k_typed, &v_typed, kv_len, stream);
        *d_k_out = k_typed;
        *d_v_out = v_typed;
        return result;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_typed(
        int layer, int seq_idx,
        const DataT **d_k_out, const DataT **d_v_out,
        int *kv_len, hipStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        EntryT &entry = entries_[layer][seq_idx];
        *kv_len = entry.count;

        if (entry.count == 0)
        {
            *d_k_out = nullptr;
            *d_v_out = nullptr;
            return true;
        }

        // Optimization: if not wrapped, return direct pointers
        if (!entry.is_wrapped(max_seq_len_))
        {
            int tail = entry.tail(max_seq_len_);
            *d_k_out = entry.d_K + static_cast<size_t>(tail) * kv_storage_dim_;
            *d_v_out = entry.d_V + static_cast<size_t>(tail) * kv_storage_dim_;
            return true;
        }

        if (!linearize_entry(entry, stream))
            return false;

        *d_k_out = static_cast<DataT *>(conv_scratch_k_);
        *d_v_out = static_cast<DataT *>(conv_scratch_v_);
        ++linearization_count_;
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::linearize_entry(EntryT &entry, hipStream_t stream)
    {
        const size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                                   static_cast<size_t>(kv_storage_dim_) *
                                   sizeof(DataT);
        if (!ensureConvScratch(buffer_size))
            return false;

        launch_linearize_kernel(entry,
                                static_cast<DataT *>(conv_scratch_k_),
                                static_cast<DataT *>(conv_scratch_v_),
                                stream);
        return true;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_linearize_kernel(
        const EntryT &entry, DataT *d_k_out, DataT *d_v_out,
        hipStream_t stream)
    {
        int tail = entry.tail(max_seq_len_);

        if constexpr (Precision == ActivationPrecision::FP32)
        {
            // Linearize K
            hip_ring_linearize_fp32(
                d_k_out, entry.d_K,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
            // Linearize V
            hip_ring_linearize_fp32(
                d_v_out, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_linearize_fp16(
                d_k_out, entry.d_K,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_fp16(
                d_v_out, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_linearize_bf16(
                d_k_out, entry.d_K,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_bf16(
                d_v_out, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_linearize_q8_1(
                d_k_out, entry.d_K,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_q8_1(
                d_v_out, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::linearize_to(
        int layer, int seq_idx,
        void *d_k_out, void *d_v_out,
        int *kv_len, hipStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::linearize_to] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        const EntryT &entry = entries_[layer][seq_idx];
        *kv_len = entry.count;

        if (entry.count == 0)
        {
            return true;
        }

        launch_linearize_kernel(entry,
                                static_cast<DataT *>(d_k_out),
                                static_cast<DataT *>(d_v_out),
                                stream);
        return true;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }

        EntryT &entry = entries_[layer][seq_idx];
        int to_evict = std::min(num_tokens, entry.count);

        // O(1) eviction - just update count (tail moves implicitly)
        entry.count -= to_evict;
        total_evicted_ += to_evict;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::evict_oldest_layer(int layer, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }

        for (int seq = 0; seq < batch_size_; ++seq)
        {
            evict_oldest(layer, seq, num_tokens);
        }
    }

    template <ActivationPrecision Precision>
    int ROCmRingKVCache<Precision>::gather_kv_batched(
        int layer, int num_seqs,
        void *d_k_out, void *d_v_out,
        int *kv_lens, int max_kv_len,
        hipStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || num_seqs > batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::gather_kv_batched] Invalid layer=" << layer);
            return -1;
        }

        // Collect entry pointers and metadata
        std::vector<EntryT *> entry_ptrs(num_seqs);
        int actual_max_kv_len = 0;

        for (int seq = 0; seq < num_seqs; ++seq)
        {
            entry_ptrs[seq] = &entries_[layer][seq];
            kv_lens[seq] = entry_ptrs[seq]->count;
            actual_max_kv_len = std::max(actual_max_kv_len, kv_lens[seq]);
        }

        if (actual_max_kv_len == 0)
        {
            return 0;
        }

        // Use provided max_kv_len or actual
        int out_max_kv_len = (max_kv_len > 0) ? max_kv_len : actual_max_kv_len;

        launch_gather_kernel(entry_ptrs,
                             static_cast<DataT *>(d_k_out),
                             static_cast<DataT *>(d_v_out),
                             kv_lens, out_max_kv_len,
                             num_seqs, stream);

        return actual_max_kv_len;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_gather_kernel(
        const std::vector<EntryT *> &entries,
        DataT *d_k_out, DataT *d_v_out,
        int *kv_lens, int max_kv_len,
        int num_seqs, hipStream_t stream)
    {
        // Device arrays for kernel
        DataT **d_k_caches = nullptr;
        DataT **d_v_caches = nullptr;
        int *d_tails = nullptr;
        int *d_counts = nullptr;

        if (!validateROCmWorkspaceBinding(workspace_, device_id_, "ROCmRingKVCache"))
        {
            return;
        }

        // Use pre-allocated workspace buffers (fast path - workspace is required)
        d_k_caches = reinterpret_cast<DataT **>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::K_CACHE_PTRS));
        d_v_caches = reinterpret_cast<DataT **>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::V_CACHE_PTRS));
        d_tails = reinterpret_cast<int *>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::TAILS));
        d_counts = reinterpret_cast<int *>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::COUNTS));

        LOG_TRACE("[ROCmRingKVCache] Using workspace buffers for gather, num_seqs=" << num_seqs);

        // Prepare host arrays
        std::vector<DataT *> h_k_caches(num_seqs);
        std::vector<DataT *> h_v_caches(num_seqs);
        std::vector<int> h_tails(num_seqs);
        std::vector<int> h_counts(num_seqs);

        for (int i = 0; i < num_seqs; ++i)
        {
            h_k_caches[i] = entries[i]->d_K;
            h_v_caches[i] = entries[i]->d_V;
            h_tails[i] = entries[i]->tail(max_seq_len_);
            h_counts[i] = entries[i]->count;
        }

        // Copy to device
        hipMemcpyAsync(d_k_caches, h_k_caches.data(), num_seqs * sizeof(DataT *),
                       hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_v_caches, h_v_caches.data(), num_seqs * sizeof(DataT *),
                       hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_tails, h_tails.data(), num_seqs * sizeof(int),
                       hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_counts, h_counts.data(), num_seqs * sizeof(int),
                       hipMemcpyHostToDevice, stream);

        // Launch kernel (via extern "C" wrapper)
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            hip_ring_gather_batched_fp32(
                d_k_out, d_v_out,
                const_cast<const float *const *>(d_k_caches),
                const_cast<const float *const *>(d_v_caches),
                d_tails, d_counts,
                num_seqs, max_kv_len, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_gather_batched_fp16(
                d_k_out, d_v_out,
                const_cast<const _Float16 *const *>(d_k_caches),
                const_cast<const _Float16 *const *>(d_v_caches),
                d_tails, d_counts,
                num_seqs, max_kv_len, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_gather_batched_bf16(
                d_k_out, d_v_out,
                const_cast<const hip_bfloat16 *const *>(d_k_caches),
                const_cast<const hip_bfloat16 *const *>(d_v_caches),
                d_tails, d_counts,
                num_seqs, max_kv_len, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_gather_batched_q8_1(
                d_k_out, d_v_out,
                const_cast<const Q8_1Block *const *>(d_k_caches),
                const_cast<const Q8_1Block *const *>(d_v_caches),
                d_tails, d_counts,
                num_seqs, max_kv_len, max_seq_len_, kv_storage_dim_, stream);
        }

        // Removed hipStreamSynchronize() - caller manages coherence via events
        // Workspace buffers are caller-owned, no cleanup needed
    }

    // clear(), clear_sequence(), clear_layer() are now provided by
    // ROCmRingKVCacheBase via entry accessor overrides (resetEntry, onClearSequence).

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::clear()
    {
        // Reset every request-shaped sidecar owned by the persistent cache
        // object. Reconstructing the cache naturally drops these buffers; clear()
        // must preserve the same invariant without invalidating the main pool
        // pointers that cached ComputeGraphs depend on.
        HipDeviceGuard::setDevice(device_id_);
        hipStream_t clear_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());

        // Conversion scratch is sized by the previous append/read path and can
        // carry stale lanes if a later request converts fewer tokens. Drop it so
        // the next conversion starts from fresh allocation just like a new cache.
        freeConvScratch();

        // Dynamic head params are used by both captured and non-captured append
        // paths. Keep the allocations stable, but reset host and device values
        // to match empty ring entries before the next request's first append.
        const int num_entries = n_layers_ * batch_size_;
        if (h_head_params_ && num_entries > 0)
        {
            std::memset(h_head_params_, 0, static_cast<size_t>(num_entries) * sizeof(int));
        }
        if (d_head_params_ && num_entries > 0)
        {
            hipError_t head_err = hipMemsetAsync(d_head_params_, 0,
                                                 static_cast<size_t>(num_entries) * sizeof(int),
                                                 clear_stream);
            if (head_err != hipSuccess)
            {
                LOG_WARN("[ROCmRingKVCache::clear] head params memset failed: "
                         << hipGetErrorString(head_err));
            }
        }

        // RoPE-on-read shadows and tensor views are cheap metadata wrappers
        // around request contents. Free/reset them so no view can expose stale
        // rows after a shorter prompt follows a longer one.
        for (auto &layer_shadows : rope_shadows_)
        {
            for (auto &shadow : layer_shadows)
            {
                if (shadow.d_K)
                {
                    hipFree(shadow.d_K);
                    shadow.d_K = nullptr;
                }
                if (shadow.d_V)
                {
                    hipFree(shadow.d_V);
                    shadow.d_V = nullptr;
                }
                shadow.converted_count = 0;
                shadow.last_head = -1;
                shadow.rope_applied = false;
                shadow.k_view.reset();
                shadow.v_view.reset();
            }
        }
        rope_shadows_.clear();
        for (auto &layer_views : tensor_views_)
        {
            for (auto &views : layer_views)
            {
                views[0].reset();
                views[1].reset();
            }
        }

        if (pool_base_ && pool_size_ > 0)
        {
            hipError_t err = hipMemsetAsync(pool_base_, 0, pool_size_, clear_stream);
            if (err != hipSuccess)
            {
                LOG_WARN("[ROCmRingKVCache::clear] pooled KV memset failed: "
                         << hipGetErrorString(err));
            }
        }
        else
        {
            const size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                                       static_cast<size_t>(kv_storage_dim_) *
                                       sizeof(DataT);
            for (auto &layer_entries : entries_)
            {
                for (auto &entry : layer_entries)
                {
                    if (entry.d_K)
                        hipMemsetAsync(entry.d_K, 0, buffer_size, clear_stream);
                    if (entry.d_V)
                        hipMemsetAsync(entry.d_V, 0, buffer_size, clear_stream);
                }
            }
        }

        hipError_t sync_err = hipStreamSynchronize(clear_stream);
        if (sync_err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCache::clear] KV buffer clear sync failed: "
                     << hipGetErrorString(sync_err));
        }

        // Do not call ROCmRingKVCacheBase::clear() here: it loops through
        // clear_layer(), which is virtual. ROCmHybridRingKVCache overrides
        // clear_layer() to accept global model layer ids, so dispatching there
        // with compressed FA indices would skip/reset the wrong entries. Clear
        // compressed entries directly while still using the base sequence helper
        // for resetEntry()/onClearSequence() bookkeeping.
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                ROCmRingKVCacheBase::clear_sequence(layer, seq);
            }
        }
    }

    // =========================================================================
    // get_k() / get_v() implementations
    // =========================================================================

    template <ActivationPrecision Precision>
    ITensor *ROCmRingKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[ROCmRingKVCache::get_k] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. ROCmHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[ROCmRingKVCache::get_k] get_kv_typed failed for layer="
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
                static_cast<size_t>(kv_dim_),
                tensor_type,
                device_id_);

            LOG_TRACE("[ROCmRingKVCache::get_k] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *ROCmRingKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<ROCmRingKVCache<Precision> *>(this)->get_k(layer, seq_idx);
    }

    template <ActivationPrecision Precision>
    ITensor *ROCmRingKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_WARN("[ROCmRingKVCache::get_v] Invalid layer=" << layer
                                                               << " seq_idx=" << seq_idx);
            return nullptr;
        }

        // Get device pointers via non-virtual get_kv_typed to avoid double-mapping
        // when called from a derived class (e.g. ROCmHybridRingKVCache) that overrides
        // the virtual get_kv_for_attention with its own layer remapping.
        const DataT *d_k_typed = nullptr;
        const DataT *d_v_typed = nullptr;
        int kv_len = 0;

        if (!get_kv_typed(layer, seq_idx, &d_k_typed, &d_v_typed, &kv_len, 0))
        {
            LOG_WARN("[ROCmRingKVCache::get_v] get_kv_typed failed for layer="
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
                static_cast<size_t>(kv_dim_),
                tensor_type,
                device_id_);

            LOG_TRACE("[ROCmRingKVCache::get_v] Created view for layer=" << layer
                                                                         << " seq=" << seq_idx << " kv_len=" << kv_len);
        }

        return view.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *ROCmRingKVCache<Precision>::get_v(int layer, int seq_idx) const
    {
        // Delegate to non-const version (tensor_views_ is mutable)
        return const_cast<ROCmRingKVCache<Precision> *>(this)->get_v(layer, seq_idx);
    }

    // =========================================================================
    // IWorkspaceConsumer Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    WorkspaceRequirements ROCmRingKVCache<Precision>::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        // m = batch size (number of sequences in gather operation)
        // n, k unused for KV cache
        // Default to batch_size_ if m is 0
        const int actual_batch_size = (m > 0) ? m : batch_size_;

        WorkspaceRequirements reqs;

        // Buffer for K cache pointers: DataT* per sequence
        reqs.buffers.push_back({
            KVCacheWorkspaceBuffers::K_CACHE_PTRS,
            static_cast<size_t>(actual_batch_size) * sizeof(DataT *),
            sizeof(void *) // Pointer alignment
        });

        // Buffer for V cache pointers: DataT* per sequence
        reqs.buffers.push_back({
            KVCacheWorkspaceBuffers::V_CACHE_PTRS,
            static_cast<size_t>(actual_batch_size) * sizeof(DataT *),
            sizeof(void *) // Pointer alignment
        });

        // Buffer for tail indices: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::TAILS,
                                static_cast<size_t>(actual_batch_size) * sizeof(int),
                                sizeof(int)});

        // Buffer for count values: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::COUNTS,
                                static_cast<size_t>(actual_batch_size) * sizeof(int),
                                sizeof(int)});

        LOG_DEBUG("[ROCmRingKVCache] Workspace requirements: batch_size="
                  << actual_batch_size
                  << " K_CACHE_PTRS=" << actual_batch_size * sizeof(DataT *)
                  << " V_CACHE_PTRS=" << actual_batch_size * sizeof(DataT *)
                  << " TAILS=" << actual_batch_size * sizeof(int)
                  << " COUNTS=" << actual_batch_size * sizeof(int));

        return reqs;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
        LOG_DEBUG("[ROCmRingKVCache] Workspace bound: " << (workspace ? "yes" : "no"));
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::hasWorkspace() const
    {
        return workspace_ != nullptr;
    }

    template <ActivationPrecision Precision>
    DeviceWorkspaceManager *ROCmRingKVCache<Precision>::getWorkspace() const
    {
        return workspace_;
    }

    // =========================================================================
    // get_kv() implementations (IKVCache interface)
    // =========================================================================

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv(int layer, int seq_idx,
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
    bool ROCmRingKVCache<Precision>::get_kv(int layer, int seq_idx,
                                            const ITensor **out_k, const ITensor **out_v,
                                            int *out_kv_len) const
    {
        return const_cast<ROCmRingKVCache<Precision> *>(this)->get_kv(
            layer, seq_idx,
            const_cast<ITensor **>(out_k),
            const_cast<ITensor **>(out_v),
            out_kv_len);
    }

    // =========================================================================
    // RoPE Shadow Buffer Helpers (for get_kv_converted)
    // =========================================================================

    extern "C" bool hip_rope_apply_fp16(
        _Float16 *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim = 0);

    extern "C" bool hip_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim = 0);

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::ensureRoPEShadow(int layer, int seq_idx) const
    {
        if (rope_shadows_.empty())
        {
            rope_shadows_.resize(n_layers_);
            for (auto &layer_shadows : rope_shadows_)
                layer_shadows.resize(batch_size_);
        }

        auto &shadow = rope_shadows_[layer][seq_idx];
        if (!shadow.d_K)
        {
            const size_t buf_bytes = static_cast<size_t>(max_seq_len_) * kv_dim_ * sizeof(_Float16);
            hipSetDevice(device_id_);
            hipMalloc(&shadow.d_K, buf_bytes);
            hipMalloc(&shadow.d_V, buf_bytes);
            shadow.converted_count = 0;
            shadow.last_head = -1;
            shadow.rope_applied = false;
        }
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::invalidateRoPEShadow(int layer, int seq_idx) const
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

    // =========================================================================
    // get_kv_converted(): FP16 shadow buffers with optional RoPE
    // =========================================================================

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_converted(
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
        // Use qualified call to avoid virtual dispatch — ROCmHybridRingKVCache
        // overrides get_kv() with layer remapping, but the layer index passed here
        // has already been remapped by the hybrid override of get_kv_converted().
        const bool want_rope = (rope && rope->rope_theta > 0.0f);
        if (!want_rope)
        {
            return ROCmRingKVCache::get_kv(layer, seq_idx, out_k, out_v, out_kv_len);
        }

        hipSetDevice(device_id_);
        const hipStream_t stream = getEffectiveStream(
            rope ? static_cast<hipStream_t>(rope->gpu_stream) : nullptr);

        ensureRoPEShadow(layer, seq_idx);
        auto &shadow = rope_shadows_[layer][seq_idx];

        // Detect incremental vs full rebuild
        const int new_tokens = entry.count - shadow.converted_count;
        const bool need_full_rebuild = (shadow.converted_count == 0 ||
                                        new_tokens < 0 ||
                                        shadow.converted_count > entry.count);

        if constexpr (Precision == ActivationPrecision::FP16)
        {
            auto *shadow_k = static_cast<_Float16 *>(shadow.d_K);
            auto *shadow_v = static_cast<_Float16 *>(shadow.d_V);

            if (need_full_rebuild)
            {
                int kv_len = 0;
                // Qualified call avoids virtual dispatch — prevents ROCmHybridRingKVCache
                // from double-remapping the already-remapped layer index.
                if (!ROCmRingKVCache::linearize_to(layer, seq_idx, shadow_k, shadow_v, &kv_len, stream))
                    return false;

                hip_rope_apply_fp16(shadow_k, kv_len, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start, stream,
                                    rope->rope_dim);

                shadow.converted_count = kv_len;
            }
            else if (new_tokens > 0)
            {
                int kv_len = 0;
                if (!ROCmRingKVCache::linearize_to(layer, seq_idx, shadow_k, shadow_v, &kv_len, stream))
                    return false;

                // Apply RoPE to ALL tokens — linearize_to overwrites the entire
                // shadow with fresh (non-RoPE'd) data from the ring buffer.
                hip_rope_apply_fp16(shadow_k, kv_len, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start, stream,
                                    rope->rope_dim);

                shadow.converted_count = kv_len;
            }
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
            if (!ROCmRingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            auto *shadow_k = static_cast<_Float16 *>(shadow.d_K);
            auto *shadow_v = static_cast<_Float16 *>(shadow.d_V);

            if (need_full_rebuild)
            {
                hip_rope_apply_fp32(d_temp_k, kv_len, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start, stream,
                                    rope->rope_dim);

                hip_convert_tensor_to_fp16(d_temp_k, TensorType::FP32,
                                           reinterpret_cast<uint16_t *>(shadow_k),
                                           kv_len * kv_dim_, stream);
                hip_convert_tensor_to_fp16(d_temp_v, TensorType::FP32,
                                           reinterpret_cast<uint16_t *>(shadow_v),
                                           kv_len * kv_dim_, stream);
            }
            else if (new_tokens > 0)
            {
                const int old_count = shadow.converted_count;
                float *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_dim_;
                float *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_dim_;

                hip_rope_apply_fp32(new_k_start, new_tokens, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start + old_count, stream,
                                    rope->rope_dim);

                _Float16 *shadow_k_new = shadow_k + static_cast<size_t>(old_count) * kv_dim_;
                _Float16 *shadow_v_new = shadow_v + static_cast<size_t>(old_count) * kv_dim_;

                hip_convert_tensor_to_fp16(new_k_start, TensorType::FP32,
                                           reinterpret_cast<uint16_t *>(shadow_k_new),
                                           new_tokens * kv_dim_, stream);
                hip_convert_tensor_to_fp16(new_v_start, TensorType::FP32,
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
            if (!ROCmRingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            auto *shadow_k = static_cast<_Float16 *>(shadow.d_K);
            auto *shadow_v = static_cast<_Float16 *>(shadow.d_V);

            if (need_full_rebuild)
            {
                hip_convert_tensor_to_fp16(d_temp_k, TensorType::Q8_1,
                                           reinterpret_cast<uint16_t *>(shadow_k),
                                           kv_len * kv_dim_, stream);
                hip_convert_tensor_to_fp16(d_temp_v, TensorType::Q8_1,
                                           reinterpret_cast<uint16_t *>(shadow_v),
                                           kv_len * kv_dim_, stream);

                hip_rope_apply_fp16(shadow_k, kv_len, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start, stream,
                                    rope->rope_dim);
            }
            else if (new_tokens > 0)
            {
                const int old_count = shadow.converted_count;
                Q8_1Block *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_storage_dim_;
                Q8_1Block *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_storage_dim_;

                _Float16 *shadow_k_new = shadow_k + static_cast<size_t>(old_count) * kv_dim_;
                _Float16 *shadow_v_new = shadow_v + static_cast<size_t>(old_count) * kv_dim_;

                hip_convert_tensor_to_fp16(new_k_start, TensorType::Q8_1,
                                           reinterpret_cast<uint16_t *>(shadow_k_new),
                                           new_tokens * kv_dim_, stream);
                hip_convert_tensor_to_fp16(new_v_start, TensorType::Q8_1,
                                           reinterpret_cast<uint16_t *>(shadow_v_new),
                                           new_tokens * kv_dim_, stream);

                hip_rope_apply_fp16(shadow_k_new, new_tokens, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta,
                                    rope->position_start + old_count, stream,
                                    rope->rope_dim);
            }

            shadow.converted_count = kv_len;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            // BF16 → linearize to scratch, convert to FP16 shadow, RoPE
            const size_t bf16_bytes = static_cast<size_t>(entry.count) * kv_dim_ * sizeof(hip_bfloat16);

            if (!ensureConvScratch(bf16_bytes))
                return false;

            auto *d_temp_k = static_cast<hip_bfloat16 *>(conv_scratch_k_);
            auto *d_temp_v = static_cast<hip_bfloat16 *>(conv_scratch_v_);

            int kv_len = 0;
            if (!ROCmRingKVCache::linearize_to(layer, seq_idx, d_temp_k, d_temp_v, &kv_len, stream))
                return false;

            auto *shadow_k = static_cast<_Float16 *>(shadow.d_K);
            auto *shadow_v = static_cast<_Float16 *>(shadow.d_V);

            if (need_full_rebuild)
            {
                hip_convert_tensor_to_fp16(d_temp_k, TensorType::BF16,
                                           reinterpret_cast<uint16_t *>(shadow_k),
                                           kv_len * kv_dim_, stream);
                hip_convert_tensor_to_fp16(d_temp_v, TensorType::BF16,
                                           reinterpret_cast<uint16_t *>(shadow_v),
                                           kv_len * kv_dim_, stream);

                hip_rope_apply_fp16(shadow_k, kv_len, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta, rope->position_start, stream,
                                    rope->rope_dim);
            }
            else if (new_tokens > 0)
            {
                const int old_count = shadow.converted_count;
                hip_bfloat16 *new_k_start = d_temp_k + static_cast<size_t>(old_count) * kv_dim_;
                hip_bfloat16 *new_v_start = d_temp_v + static_cast<size_t>(old_count) * kv_dim_;

                _Float16 *shadow_k_new = shadow_k + static_cast<size_t>(old_count) * kv_dim_;
                _Float16 *shadow_v_new = shadow_v + static_cast<size_t>(old_count) * kv_dim_;

                hip_convert_tensor_to_fp16(new_k_start, TensorType::BF16,
                                           reinterpret_cast<uint16_t *>(shadow_k_new),
                                           new_tokens * kv_dim_, stream);
                hip_convert_tensor_to_fp16(new_v_start, TensorType::BF16,
                                           reinterpret_cast<uint16_t *>(shadow_v_new),
                                           new_tokens * kv_dim_, stream);

                hip_rope_apply_fp16(shadow_k_new, new_tokens, local_n_kv_heads_, head_dim_,
                                    rope->rope_theta,
                                    rope->position_start + old_count, stream,
                                    rope->rope_dim);
            }

            shadow.converted_count = kv_len;
        }

        shadow.last_head = entry.head;
        shadow.rope_applied = true;

        // Create/update GpuTensorViews for the shadow buffers
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

    // =========================================================================
    // Explicit Template Instantiations
    // =========================================================================

    template class ROCmRingKVCache<ActivationPrecision::FP32>;
    template class ROCmRingKVCache<ActivationPrecision::FP16>;
    template class ROCmRingKVCache<ActivationPrecision::BF16>;
    template class ROCmRingKVCache<ActivationPrecision::Q8_1>;

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<IROCmRingKVCache> createROCmRingKVCache(
        ActivationPrecision precision,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<ROCmRingKVCacheFP32>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<ROCmRingKVCacheFP16>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<ROCmRingKVCacheBF16>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmRingKVCacheQ8_1>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        default:
            LOG_ERROR("[createROCmRingKVCache] Unsupported precision: "
                      << static_cast<int>(precision));
            return nullptr;
        }
    }

    // =========================================================================
    // Sharded Factory Function (for Tensor Parallelism)
    // =========================================================================

    std::unique_ptr<IROCmRingKVCache> createShardedROCmRingKVCache(
        ActivationPrecision precision,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<ROCmRingKVCacheFP32>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<ROCmRingKVCacheFP16>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<ROCmRingKVCacheBF16>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmRingKVCacheQ8_1>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        default:
            LOG_ERROR("[createShardedROCmRingKVCache] Unsupported precision: "
                      << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
