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
#include "../../kvcache/KVCacheWorkspaceContract.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../ROCmKernelBase.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/KVCacheProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../kvcache/KVCacheLogicalBlockCodec.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Check whether the opt-in MTP publication trace is enabled.
         *
         * HIP graph capture records the append kernels while replay-time
         * metadata remains device resident.  This diagnostic gate lets the
         * capture path report the dynamic scalar pointers it records without
         * changing production stream or ownership behavior.
         */
        bool mtpKVPublicationDiagnosticsEnabled()
        {
            return debugEnv().runtime_debug.mtp_publication_diagnostics;
        }
    } // namespace

    // =========================================================================
    // External HIP Kernel Declarations
    // =========================================================================

    // Scalar diagnostic linearization wrappers.
    extern "C" void hip_ring_linearize_fp32(
        float *d_out,
        const float *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_fp16(
        _Float16 *d_out,
        const _Float16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_bf16(
        hip_bfloat16 *d_out,
        const hip_bfloat16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream);

    extern "C" void hip_ring_linearize_q8_1(
        Q8_1Block *d_out,
        const Q8_1Block *d_cache,
        int tail, int count, int max_seq_len, int kv_blocks,
        hipStream_t stream);

    extern "C" hipError_t hip_ring_logical_block_export_device_fp32(
        const float *, const float *, float *, float *,
        const int *, const int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_export_device_fp16(
        const _Float16 *, const _Float16 *, _Float16 *, _Float16 *,
        const int *, const int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_export_device_bf16(
        const hip_bfloat16 *, const hip_bfloat16 *,
        hip_bfloat16 *, hip_bfloat16 *,
        const int *, const int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_export_device_q8_1(
        const Q8_1Block *, const Q8_1Block *, Q8_1Block *, Q8_1Block *,
        const int *, const int *, int, int, int, int, hipStream_t);

    extern "C" hipError_t hip_ring_logical_block_import_device_fp32(
        float *, float *, const float *, const float *,
        int *, int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_import_device_fp16(
        _Float16 *, _Float16 *, const _Float16 *, const _Float16 *,
        int *, int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_import_device_bf16(
        hip_bfloat16 *, hip_bfloat16 *,
        const hip_bfloat16 *, const hip_bfloat16 *,
        int *, int *, int, int, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_logical_block_import_device_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        int *, int *, int, int, int, int, hipStream_t);

    extern "C" hipError_t hip_ring_gather_batched_device_state_fp32(
        float *, float *, const float *const *, const float *const *,
        const int *, const int *, int, int, int, int, int, bool, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_device_state_fp16(
        _Float16 *, _Float16 *, const _Float16 *const *, const _Float16 *const *,
        const int *, const int *, int, int, int, int, int, bool, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_device_state_bf16(
        hip_bfloat16 *, hip_bfloat16 *,
        const hip_bfloat16 *const *, const hip_bfloat16 *const *,
        const int *, const int *, int, int, int, int, int, bool, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_device_state_q8_1(
        Q8_1Block *, Q8_1Block *,
        const Q8_1Block *const *, const Q8_1Block *const *,
        const int *, const int *, int, int, int, int, int, bool, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_fp32(
        _Float16 *, _Float16 *, const float *const *, const float *const *,
        const int *, const int *, int, int, int, int, int, int, int,
        float, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_fp16(
        _Float16 *, _Float16 *, const _Float16 *const *, const _Float16 *const *,
        const int *, const int *, int, int, int, int, int, int, int,
        float, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_bf16(
        _Float16 *, _Float16 *,
        const hip_bfloat16 *const *, const hip_bfloat16 *const *,
        const int *, const int *, int, int, int, int, int, int, int,
        float, int, int, hipStream_t);
    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_q8_1(
        _Float16 *, _Float16 *,
        const Q8_1Block *const *, const Q8_1Block *const *,
        const int *, const int *, int, int, int, int, int, int, int,
        float, int, int, hipStream_t);

    // Dynamic head append wrappers (graph-capturable)
    extern "C" void hip_ring_append_dynamic_fp32(
        float *, float *, const float *, const float *,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_fp16(
        _Float16 *, _Float16 *, const _Float16 *, const _Float16 *,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_bf16(
        hip_bfloat16 *, hip_bfloat16 *, const hip_bfloat16 *, const hip_bfloat16 *,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" void hip_ring_append_dynamic_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" bool hip_ring_append_converted_fp32(
        float *, float *, const void *, const void *, TensorType,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" bool hip_ring_append_converted_fp16(
        _Float16 *, _Float16 *, const void *, const void *, TensorType,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" bool hip_ring_append_converted_bf16(
        hip_bfloat16 *, hip_bfloat16 *, const void *, const void *, TensorType,
        const int *, const int *, int, int, int, hipStream_t);
    extern "C" void hip_kv_sequence_state_advance(
        int *, int *, int, int, hipStream_t);
    extern "C" void hip_kv_sequence_state_advance_dynamic(
        int *, int *, const int *, int, int, hipStream_t);

    extern "C" void hip_ring_append_verifier_rows_dynamic_fp32(
        float *, float *, const float *, const float *,
        const int *, const int *, int, int, int, int, bool, bool, hipStream_t);
    extern "C" void hip_ring_append_verifier_rows_dynamic_fp16(
        _Float16 *, _Float16 *, const _Float16 *, const _Float16 *,
        const int *, const int *, int, int, int, int, bool, bool, hipStream_t);
    extern "C" void hip_ring_append_verifier_rows_dynamic_bf16(
        hip_bfloat16 *, hip_bfloat16 *, const hip_bfloat16 *, const hip_bfloat16 *,
        const int *, const int *, int, int, int, int, bool, bool, hipStream_t);
    extern "C" void hip_ring_append_verifier_rows_dynamic_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        const int *, const int *, int, int, int, int, bool, bool, hipStream_t);

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
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(this))
        {
            DeviceWorkspaceManager *workspace = consumer->getWorkspace();
            if (workspace && workspace->isAllocated())
            {
                void *workspace_k = workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
                void *workspace_v = workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
                const size_t workspace_k_size = workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
                const size_t workspace_v_size = workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
                if (!workspace_k || !workspace_v || workspace_k_size < bytes || workspace_v_size < bytes)
                {
                    LOG_ERROR("[IROCmRingKVCache] Bound workspace is missing conversion scratch: required="
                              << bytes << " bytes, K_available=" << workspace_k_size
                              << " V_available=" << workspace_v_size);
                    return false;
                }

                conv_scratch_k_ = workspace_k;
                conv_scratch_v_ = workspace_v;
                conv_scratch_capacity_ = std::min(workspace_k_size, workspace_v_size);
                conv_scratch_workspace_backed_ = true;
                return true;
            }
        }

        LOG_ERROR("[IROCmRingKVCache] Conversion requires pre-bound graph workspace: required="
                  << bytes << " bytes per K/V buffer");
        return false;
    }

    void IROCmRingKVCache::freeConvScratch()
    {
        // Conversion scratch is always workspace-owned.
        conv_scratch_k_ = nullptr;
        conv_scratch_v_ = nullptr;
        conv_scratch_capacity_ = 0;
        conv_scratch_workspace_backed_ = false;
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
        requireGPUExecutionStream(
            gpu_stream,
            "IROCmRingKVCache::appendWithStream");
        if (!K || !V)
        {
            LOG_DEBUG("[IROCmRingKVCache::appendWithStream] Null K or V tensor");
            return false;
        }
        const auto target = DeviceId::rocm(device_id());

        const auto require_input = [&](const ITensor *tensor, const char *label)
        {
            if (const auto *prepared =
                    dynamic_cast<const PreparedGpuTensorView *>(tensor))
            {
                if (!prepared->isPreparedFor(target, gpu_stream))
                {
                    throw std::runtime_error(
                        std::string("[IROCmRingKVCache::appendWithStream] Prepared ") +
                        label + " slice does not match the ROCm consumer device/stream");
                }
                return;
            }

            /*
             * Ordinary tensors still join their canonical producer event here.
             * The stage-created prepared view is the only exception: its type
             * carries the exact backend-qualified device and stream on which
             * the parent tensor was already ordered.
             */
            TransferEngine::requireDeviceInput(
                const_cast<ITensor *>(tensor), target, gpu_stream);
        };
        require_input(K, "K");
        require_input(V, "V");

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k || !d_v)
        {
            LOG_ERROR("[IROCmRingKVCache::appendWithStream] K or V tensor lacks validated GPU storage");
            return false;
        }

        const auto stream = static_cast<hipStream_t>(gpu_stream);

        const ActivationPrecision destination_precision = k_precision();
        const bool floating_cache =
            destination_precision == ActivationPrecision::FP32 ||
            destination_precision == ActivationPrecision::FP16 ||
            destination_precision == ActivationPrecision::BF16;
        const bool source_matches_cache =
            (destination_precision == ActivationPrecision::FP32 &&
             K->native_type() == TensorType::FP32 &&
             V->native_type() == TensorType::FP32) ||
            (destination_precision == ActivationPrecision::FP16 &&
             K->native_type() == TensorType::FP16 &&
             V->native_type() == TensorType::FP16) ||
            (destination_precision == ActivationPrecision::BF16 &&
             K->native_type() == TensorType::BF16 &&
             V->native_type() == TensorType::BF16);

        if (floating_cache && !source_matches_cache)
        {
            const auto &k_shape = K->shape();
            const auto &v_shape = V->shape();
            if (k_shape.size() < 2 || v_shape.size() < 2)
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Invalid K/V shape for floating cache conversion");
                return false;
            }

            const int kv_dim = static_cast<int>(k_shape[1]);
            const int elements = num_tokens * kv_dim;
            if (elements <= 0)
            {
                return append(layer, seq_idx, d_k, d_v, num_tokens, stream);
            }
            if (K->native_type() != V->native_type())
            {
                LOG_ERROR("[IROCmRingKVCache::appendWithStream] Asymmetric K/V source types are unsupported for fused floating append: K="
                          << static_cast<int>(K->native_type())
                          << " V=" << static_cast<int>(V->native_type()));
                return false;
            }

            const auto append_start = std::chrono::high_resolution_clock::now();
            const bool ok = appendConvertedWithStream(layer, seq_idx, d_k, d_v,
                                                      K->native_type(), num_tokens, stream);
            const auto append_end = std::chrono::high_resolution_clock::now();
            {
                auto to_ns = [](auto d) -> uint64_t
                {
                    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
                };
                const uint64_t append_ns = to_ns(append_end - append_start);
                const uint64_t destination_element_bytes =
                    destination_precision == ActivationPrecision::FP32
                        ? sizeof(float)
                        : sizeof(uint16_t);
                const uint64_t bytes =
                    static_cast<uint64_t>(elements) *
                    destination_element_bytes * 2;
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

        // RoPE shadows are non-owning views over graph-planned conversion
        // workspace. Their vector destructors release only host-side wrappers.
        rope_shadows_.clear();

        // The entry tables contain device pointers into this cache's pool, but
        // are separate cache-owned allocations. Release them before the pool
        // so no stale topology survives while the persistent payload is freed.
        releaseBatchedEntryPointerTables();

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

        auto *const backend = getROCmBackend();
        if (!backend)
        {
            throw std::runtime_error(
                "[ROCmRingKVCache] ROCm backend unavailable during canonical pool allocation");
        }
        pool_base_ = backend->allocate(pool_size_, device_id_);
        if (!pool_base_)
        {
            pool_size_ = 0;
            throw std::runtime_error(
                "[ROCmRingKVCache] Failed to allocate canonical pooled KV storage");
        }

        // Zero-initialize the entire pool
        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        const hipError_t memset_status =
            init_stream
                ? hipMemsetAsync(pool_base_, 0, pool_size_, init_stream)
                : hipErrorInvalidResourceHandle;
        const hipError_t sync_status =
            memset_status == hipSuccess
                ? hipStreamSynchronize(init_stream)
                : memset_status;
        if (memset_status != hipSuccess || sync_status != hipSuccess)
        {
            backend->free(pool_base_, device_id_);
            pool_base_ = nullptr;
            pool_size_ = 0;
            throw std::runtime_error(
                "[ROCmRingKVCache] Failed to initialize canonical pooled KV storage");
        }

        LOG_DEBUG("[ROCmRingKVCache] Pooled KV cache: one backend allocation for "
                  << total_entries << " entries × 2 buffers = "
                  << (pool_size_ / (1024 * 1024)) << " MB (replaced "
                  << (total_entries * 2) << " individual hipMalloc calls)");
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::free_pool()
    {
        if (pool_base_)
        {
            auto *const backend = getROCmBackend();
            if (!backend)
            {
                LOG_ERROR("[ROCmRingKVCache] ROCm backend unavailable while releasing canonical pool");
                std::terminate();
            }
            try
            {
                backend->free(pool_base_, device_id_);
                pool_base_ = nullptr;
                pool_size_ = 0;
            }
            catch (...)
            {
                LOG_ERROR("[ROCmRingKVCache] ROCm backend threw while releasing canonical pool");
                std::terminate();
            }
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
    }

    /**
     * @brief Publish the permanent pool topology consumed by grouped gather kernels.
     *
     * HIP graph capture may begin from an outer orchestration transaction after
     * this cache has been constructed. Therefore this routine deliberately uses
     * synchronous allocation/publication during construction and is never
     * reachable from workspace binding or a grouped cache read.
     */
    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::initializeBatchedEntryPointerTables()
    {
        if (batched_pointer_tables_ready_ ||
            d_batched_k_entry_table_ ||
            d_batched_v_entry_table_)
        {
            throw std::logic_error(
                "[ROCmRingKVCache] Immutable entry pointer tables may only be initialized once");
        }

        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        if (entry_count == 0)
        {
            // An empty hybrid-cache topology has no pointers to publish, but it
            // has still completed the one-time publication lifecycle.
            batched_pointer_tables_ready_ = true;
            return;
        }

        std::vector<DataT *> h_k_table(entry_count);
        std::vector<DataT *> h_v_table(entry_count);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                const size_t index =
                    static_cast<size_t>(layer) * static_cast<size_t>(batch_size_) +
                    static_cast<size_t>(seq);
                h_k_table[index] = entries_[layer][seq].d_K;
                h_v_table[index] = entries_[layer][seq].d_V;
                if (!h_k_table[index] || !h_v_table[index])
                {
                    throw std::runtime_error(
                        "[ROCmRingKVCache] Cannot publish a null permanent KV entry");
                }
            }
        }

        const hipError_t device_error =
            static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_));
        if (device_error != hipSuccess)
        {
            throw std::runtime_error(
                std::string("[ROCmRingKVCache] Failed to select device for immutable entry publication: ") +
                hipGetErrorString(device_error));
        }

        auto *const backend = getROCmBackend();
        if (!backend)
        {
            throw std::runtime_error(
                "[ROCmRingKVCache] ROCm backend unavailable while publishing immutable entry topology");
        }

        const size_t table_bytes = entry_count * sizeof(DataT *);
        IWorkerGPUContext &setup_context =
            device_ctx_
                ? *device_ctx_
                : GPUDeviceContextPool::instance().getAMDContext(device_id_);
        void *const setup_stream = setup_context.defaultStream();
        requireGPUExecutionStream(
            setup_stream,
            "ROCmRingKVCache::initializeBatchedEntryPointerTables");
        d_batched_k_entry_table_ =
            static_cast<DataT **>(backend->allocate(table_bytes, device_id_));
        d_batched_v_entry_table_ =
            static_cast<DataT **>(backend->allocate(table_bytes, device_id_));
        if (!d_batched_k_entry_table_ ||
            !d_batched_v_entry_table_ ||
            !backend->hostToDevice(
                d_batched_k_entry_table_, h_k_table.data(),
                table_bytes, device_id_, setup_stream) ||
            !backend->hostToDevice(
                d_batched_v_entry_table_, h_v_table.data(),
                table_bytes, device_id_, setup_stream))
        {
            releaseBatchedEntryPointerTables();
            throw std::runtime_error(
                "[ROCmRingKVCache] Failed to publish immutable batched entry pointer tables");
        }

        batched_pointer_tables_ready_ = true;
    }

    /**
     * @brief Release immutable grouped-gather topology without throwing.
     */
    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::releaseBatchedEntryPointerTables() noexcept
    {
        auto *const backend = getROCmBackend();
        if ((d_batched_k_entry_table_ || d_batched_v_entry_table_) && !backend)
        {
            LOG_ERROR("[ROCmRingKVCache] ROCm backend unavailable while releasing immutable entry topology");
            std::terminate();
        }
        try
        {
            if (d_batched_k_entry_table_)
                backend->free(d_batched_k_entry_table_, device_id_);
            if (d_batched_v_entry_table_)
                backend->free(d_batched_v_entry_table_, device_id_);
        }
        catch (...)
        {
            LOG_ERROR("[ROCmRingKVCache] ROCm backend threw while releasing immutable entry topology");
            std::terminate();
        }
        d_batched_k_entry_table_ = nullptr;
        d_batched_v_entry_table_ = nullptr;
        batched_pointer_tables_ready_ = false;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::allocate_all_entries()
    {
        const int total_entries = n_layers_ * batch_size_;

        /*
         * A hybrid model may contain no full-attention layers at all. Such a
         * cache legitimately owns no K/V payload and therefore needs no pool;
         * this is distinct from a failed allocation for a non-empty topology.
         */
        if (total_entries > 0)
        {
            allocate_pool();
            if (!pool_base_)
            {
                throw std::runtime_error(
                    "[ROCmRingKVCache] Canonical pooled KV allocation is required");
            }
        }

        try
        {
            entries_.resize(n_layers_);
            int linear_idx = 0;
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                entries_[layer].resize(batch_size_);
                for (int seq = 0; seq < batch_size_; ++seq)
                {
                    assign_entry_from_pool(entries_[layer][seq], linear_idx++);
                }
            }

            // Initialize tensor_views_ storage for get_k()/get_v() wrappers.
            tensor_views_.resize(n_layers_);
            device_ring_views_.resize(n_layers_);
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                tensor_views_[layer].resize(batch_size_);
                device_ring_views_[layer].resize(batch_size_);
            }

            initializeBatchedEntryPointerTables();
            allocateDeviceParams(); // Base class method (ROCmRingKVCacheBase)
        }
        catch (...)
        {
            // A derived-class destructor is not called when its constructor
            // throws. Explicitly release raw device ownership before allowing
            // the initialization error to escape.
            releaseBatchedEntryPointerTables();
            if (pool_base_)
                free_pool();
            for (auto &layer_entries : entries_)
            {
                for (auto &entry : layer_entries)
                {
                    entry.d_K = nullptr;
                    entry.d_V = nullptr;
                }
            }
            throw;
        }

        LOG_DEBUG("[ROCmRingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 2 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total");
    }

    // Sequence-state observations are provided by ROCmRingKVCacheBase from
    // the canonical device rows; entries contain payload topology only.

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
        requireGPUExecutionStream(
            static_cast<void *>(stream),
            "ROCmRingKVCache::append");
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::append] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        EntryT &entry = entries_[layer][seq_idx];

        if (num_tokens > 0)
        {
            if (!d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[ROCmRingKVCache::append] Canonical device sequence state is unavailable");
                return false;
            }
            const int idx = layer * batch_size_ + seq_idx;
            const int *d_append_count = deviceDynamicAppendCountPtr(layer, seq_idx);
            if (mtpKVPublicationDiagnosticsEnabled())
            {
                LOG_INFO("[MTPPublicationDiagnostics] phase=rocm_kv_append_capture_enqueue"
                         << " path=typed"
                         << " layer=" << layer
                         << " seq_idx=" << seq_idx
                         << " captured_num_tokens=" << num_tokens
                         << " d_head=" << static_cast<const void *>(&d_head_params_[idx])
                         << " d_count=" << static_cast<const void *>(&d_count_params_[idx])
                         << " d_append_count=" << static_cast<const void *>(d_append_count)
                         << " stream=" << static_cast<void *>(stream));
            }
            launch_append_kernel_dynamic(entry, d_k, d_v,
                                         &d_head_params_[idx], d_append_count,
                                         num_tokens, stream);
            hip_kv_sequence_state_advance_dynamic(
                &d_head_params_[idx], &d_count_params_[idx],
                d_append_count,
                num_tokens, max_seq_len_, stream);
        }

        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::appendVerifierRowsDecodeEquivalent(
        int layer,
        int seq_idx,
        const ITensor *K,
        const ITensor *V,
        int verifier_rows,
        void *gpu_stream)
    {
        requireGPUExecutionStream(
            gpu_stream,
            "ROCmRingKVCache::appendVerifierRowsDecodeEquivalent");
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            verifier_rows < 1 ||
            !K || !V)
        {
            LOG_ERROR("[ROCmRingKVCache] Invalid verifier append request: layer="
                      << layer << " n_layers=" << n_layers_
                      << " first_layer=" << first_layer_index()
                      << " seq_idx=" << seq_idx << " batch_size=" << batch_size_
                      << " rows=" << verifier_rows
                      << " K=" << (K ? "set" : "null")
                      << " V=" << (V ? "set" : "null")
                      << " stream=" << gpu_stream);
            return false;
        }

        const auto layout_for = [&](const ITensor *tensor,
                                    const char *label,
                                    bool *head_major,
                                    int *convert_rows,
                                    int *convert_cols) -> bool
        {
            if (!tensor || !head_major || !convert_rows || !convert_cols ||
                tensor->shape().size() < 2)
            {
                return false;
            }

            const size_t rows = tensor->shape()[0];
            const size_t cols = tensor->shape()[1];
            const bool position_major =
                rows >= static_cast<size_t>(verifier_rows) &&
                cols == static_cast<size_t>(kv_dim_);
            const bool verifier_head_major =
                rows == static_cast<size_t>(local_n_kv_heads_) *
                            static_cast<size_t>(verifier_rows) &&
                cols == static_cast<size_t>(head_dim_);

            if (!position_major && !verifier_head_major)
            {
                LOG_ERROR("[ROCmRingKVCache] Unsupported verifier " << label
                                                                    << " source shape ["
                                                                    << rows << "," << cols
                                                                    << "] rows=" << verifier_rows
                                                                    << " local_heads="
                                                                    << local_n_kv_heads_
                                                                    << " head_dim=" << head_dim_
                                                                    << " kv_dim=" << kv_dim_);
                return false;
            }

            *head_major = verifier_head_major && !position_major;
            *convert_rows = *head_major ? local_n_kv_heads_ * verifier_rows : verifier_rows;
            *convert_cols = *head_major ? head_dim_ : kv_dim_;
            return true;
        };

        bool k_head_major = false;
        bool v_head_major = false;
        int k_convert_rows = 0;
        int k_convert_cols = 0;
        int v_convert_rows = 0;
        int v_convert_cols = 0;
        if (!layout_for(K, "K", &k_head_major, &k_convert_rows, &k_convert_cols) ||
            !layout_for(V, "V", &v_head_major, &v_convert_rows, &v_convert_cols))
        {
            return false;
        }

        const void *d_k_src = K->gpu_data_ptr();
        const void *d_v_src = V->gpu_data_ptr();
        if (!d_k_src || !d_v_src)
        {
            LOG_ERROR("[ROCmRingKVCache] Verifier append requires device-resident K/V tensors");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(gpu_stream);
        const auto target_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                return TensorType::FP32;
            else if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else
                return TensorType::Q8_1;
        }();

        const DataT *typed_k = static_cast<const DataT *>(d_k_src);
        const DataT *typed_v = static_cast<const DataT *>(d_v_src);
        if (K->native_type() != target_type || V->native_type() != target_type)
        {
            if constexpr (Precision == ActivationPrecision::FP16)
            {
                const size_t k_bytes = static_cast<size_t>(k_convert_rows) *
                                       static_cast<size_t>(k_convert_cols) * sizeof(uint16_t);
                const size_t v_bytes = static_cast<size_t>(v_convert_rows) *
                                       static_cast<size_t>(v_convert_cols) * sizeof(uint16_t);
                if (!ensureConvScratch(std::max(k_bytes, v_bytes)))
                    return false;
                if (!hip_convert_tensor_to_fp16(d_k_src, K->native_type(),
                                                static_cast<uint16_t *>(conv_scratch_k_),
                                                k_convert_rows * k_convert_cols, stream) ||
                    !hip_convert_tensor_to_fp16(d_v_src, V->native_type(),
                                                static_cast<uint16_t *>(conv_scratch_v_),
                                                v_convert_rows * v_convert_cols, stream))
                {
                    LOG_ERROR("[ROCmRingKVCache] FP16 verifier append conversion failed");
                    return false;
                }
                typed_k = static_cast<const DataT *>(conv_scratch_k_);
                typed_v = static_cast<const DataT *>(conv_scratch_v_);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const auto q8_bytes = [](int rows, int cols) -> size_t
                {
                    const int blocks = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    return static_cast<size_t>(rows) * static_cast<size_t>(blocks) * sizeof(Q8_1Block);
                };
                if (!ensureConvScratch(std::max(q8_bytes(k_convert_rows, k_convert_cols),
                                                q8_bytes(v_convert_rows, v_convert_cols))))
                    return false;
                if (!hip_convert_tensor_to_q8_1(d_k_src, K->native_type(),
                                                static_cast<Q8_1Block *>(conv_scratch_k_),
                                                k_convert_rows, k_convert_cols, stream) ||
                    !hip_convert_tensor_to_q8_1(d_v_src, V->native_type(),
                                                static_cast<Q8_1Block *>(conv_scratch_v_),
                                                v_convert_rows, v_convert_cols, stream))
                {
                    LOG_ERROR("[ROCmRingKVCache] Q8_1 verifier append conversion failed");
                    return false;
                }
                typed_k = static_cast<const DataT *>(conv_scratch_k_);
                typed_v = static_cast<const DataT *>(conv_scratch_v_);
            }
            else
            {
                LOG_ERROR("[ROCmRingKVCache] Verifier append conversion is unsupported for cache precision "
                          << static_cast<int>(Precision));
                return false;
            }
        }

        EntryT &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();

        const int head_storage_dim =
            (Precision == ActivationPrecision::Q8_1)
                ? (head_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE
                : head_dim_;

        auto launch_dynamic = [&](const int *d_head,
                                  const int *d_append_count)
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                hip_ring_append_verifier_rows_dynamic_fp32(entry.d_K, entry.d_V, typed_k, typed_v,
                                                           d_head, d_append_count,
                                                           max_seq_len_, kv_storage_dim_, head_storage_dim,
                                                           verifier_rows,
                                                           k_head_major, v_head_major, stream);
            else if constexpr (Precision == ActivationPrecision::FP16)
                hip_ring_append_verifier_rows_dynamic_fp16(entry.d_K, entry.d_V, typed_k, typed_v,
                                                           d_head, d_append_count,
                                                           max_seq_len_, kv_storage_dim_, head_storage_dim,
                                                           verifier_rows,
                                                           k_head_major, v_head_major, stream);
            else if constexpr (Precision == ActivationPrecision::BF16)
                hip_ring_append_verifier_rows_dynamic_bf16(entry.d_K, entry.d_V, typed_k, typed_v,
                                                           d_head, d_append_count,
                                                           max_seq_len_, kv_storage_dim_, head_storage_dim,
                                                           verifier_rows,
                                                           k_head_major, v_head_major, stream);
            else
                hip_ring_append_verifier_rows_dynamic_q8_1(entry.d_K, entry.d_V, typed_k, typed_v,
                                                           d_head, d_append_count,
                                                           max_seq_len_, kv_storage_dim_, head_storage_dim,
                                                           verifier_rows,
                                                           k_head_major, v_head_major, stream);
        };

        if (!d_head_params_ || !d_count_params_)
        {
            LOG_ERROR("[ROCmRingKVCache] Grouped verifier requires canonical device sequence state");
            return false;
        }
        const int idx = layer * batch_size_ + seq_idx;
        const int *d_append_count =
            deviceDynamicAppendCountPtr(layer, seq_idx);
        launch_dynamic(&d_head_params_[idx], d_append_count);
        hip_kv_sequence_state_advance_dynamic(
            &d_head_params_[idx], &d_count_params_[idx],
            d_append_count, verifier_rows, max_seq_len_, stream);

        const hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache] Verifier append kernel launch failed: "
                      << hipGetErrorString(launch_err));
            return false;
        }

        // The all-format MTP sweep consumes this route record to prove that a
        // successful byte comparison came from the grouped GPU publication
        // kernel and the canonical device-state metadata path.
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_kv_cache_grouped_verifier_append_calls",
            1.0,
            "verifier",
            "rocm",
            {{"cache_format", activationPrecisionToString(Precision)},
             {"source_k_format", K->dtype_name()},
             {"source_v_format", V->dtype_name()},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"source_k_layout", k_head_major ? "head_major" : "position_major"},
             {"source_v_layout", v_head_major ? "head_major" : "position_major"},
             {"execution_mode", capture_active ? "graph_captured" : "eager_device_state"},
             {"row_count_policy", d_append_count ? "resident_device_count" : "captured_exact_shape"},
             {"topology", (local_n_kv_heads_ != n_kv_heads_ || kv_head_start_ != 0)
                              ? "local_tp_shard"
                              : "replicated"},
             {"commit_policy", "single_grouped_metadata_commit"},
             {"local_kv_heads", std::to_string(local_n_kv_heads_)},
             {"head_dim", std::to_string(head_dim_)},
             {"kv_head_start", std::to_string(kv_head_start_)}});

        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::appendConvertedWithStream(
        int layer, int seq_idx,
        const void *d_k_src, const void *d_v_src,
        TensorType src_type,
        int num_tokens, hipStream_t stream)
    {
        if constexpr (
            Precision != ActivationPrecision::FP32 &&
            Precision != ActivationPrecision::FP16 &&
            Precision != ActivationPrecision::BF16)
        {
            (void)layer;
            (void)seq_idx;
            (void)d_k_src;
            (void)d_v_src;
            (void)src_type;
            (void)num_tokens;
            (void)stream;
            LOG_ERROR("[ROCmRingKVCache::appendConvertedWithStream] Converted append requires floating cache storage");
            return false;
        }
        else
        {
            requireGPUExecutionStream(
                static_cast<void *>(stream),
                "ROCmRingKVCache::appendConvertedWithStream");
            if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            {
                LOG_ERROR("[ROCmRingKVCache::appendConvertedWithStream] Invalid layer=" << layer
                                                                                        << " or seq_idx=" << seq_idx);
                return false;
            }
            if (!d_k_src || !d_v_src || num_tokens < 0)
            {
                LOG_ERROR("[ROCmRingKVCache::appendConvertedWithStream] Invalid source pointers or token count");
                return false;
            }
            if (num_tokens == 0)
                return true;

            EntryT &entry = entries_[layer][seq_idx];
            const int idx = layer * batch_size_ + seq_idx;
            if (!d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[ROCmRingKVCache::appendConvertedWithStream] Canonical device sequence state is unavailable");
                return false;
            }
            const int *d_head = &d_head_params_[idx];
            const int *d_append_count =
                deviceDynamicAppendCountPtr(layer, seq_idx);

            bool launch_ok = false;
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                launch_ok = hip_ring_append_converted_fp32(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                launch_ok = hip_ring_append_converted_fp16(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                launch_ok = hip_ring_append_converted_bf16(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }

            if (!launch_ok)
            {
                LOG_ERROR("[ROCmRingKVCache::appendConvertedWithStream] Fused converted append launch failed");
                return false;
            }

            hip_kv_sequence_state_advance_dynamic(
                &d_head_params_[idx], &d_count_params_[idx],
                d_append_count,
                num_tokens, max_seq_len_, stream);

            return true;
        }
    }

    // Device sequence-state allocation and graph bindings are provided by
    // ROCmRingKVCacheBase.

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_append_kernel_dynamic(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        const int *d_head, const int *d_append_count, int num_tokens, hipStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            hip_ring_append_dynamic_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_append_dynamic_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_append_dynamic_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_append_dynamic_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
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
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv] Could not observe canonical device sequence state");
            return false;
        }
        const int count = state.cached_tokens;
        const int head = state.implementation_head;
        *kv_len = count;

        if (count == 0)
        {
            *d_k_out = nullptr;
            *d_v_out = nullptr;
            return true;
        }

        // Optimization: if not wrapped, return direct pointers
        if (!state.wrapped)
        {
            const int tail = (head - count + max_seq_len_) % max_seq_len_;
            *d_k_out = entry.d_K + static_cast<size_t>(tail) * kv_storage_dim_;
            *d_v_out = entry.d_V + static_cast<size_t>(tail) * kv_storage_dim_;
            return true;
        }

        if (!linearize_entry(entry, head, count, stream))
            return false;

        *d_k_out = static_cast<DataT *>(conv_scratch_k_);
        *d_v_out = static_cast<DataT *>(conv_scratch_v_);
        ++linearization_count_;
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::linearize_entry(
        EntryT &entry,
        int head,
        int count,
        hipStream_t stream)
    {
        const size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                                   static_cast<size_t>(kv_storage_dim_) *
                                   sizeof(DataT);
        if (!ensureConvScratch(buffer_size))
            return false;

        launch_linearize_kernel(entry, head, count,
                                static_cast<DataT *>(conv_scratch_k_),
                                static_cast<DataT *>(conv_scratch_v_),
                                stream);
        return true;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::launch_linearize_kernel(
        const EntryT &entry,
        int head,
        int count,
        DataT *d_k_out,
        DataT *d_v_out,
        hipStream_t stream)
    {
        const int tail = (head - count + max_seq_len_) % max_seq_len_;

        if constexpr (Precision == ActivationPrecision::FP32)
        {
            // Linearize K
            hip_ring_linearize_fp32(
                d_k_out, entry.d_K,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
            // Linearize V
            hip_ring_linearize_fp32(
                d_v_out, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            hip_ring_linearize_fp16(
                d_k_out, entry.d_K,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_fp16(
                d_v_out, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            hip_ring_linearize_bf16(
                d_k_out, entry.d_K,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_bf16(
                d_v_out, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            hip_ring_linearize_q8_1(
                d_k_out, entry.d_K,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
            hip_ring_linearize_q8_1(
                d_v_out, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
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
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return false;
        *kv_len = state.cached_tokens;

        if (state.cached_tokens == 0)
        {
            return true;
        }

        launch_linearize_kernel(entry,
                                state.implementation_head,
                                state.cached_tokens,
                                static_cast<DataT *>(d_k_out),
                                static_cast<DataT *>(d_v_out),
                                stream);
        return true;
    }

    template <ActivationPrecision Precision>
    typename IKVCache::KVCacheLogicalBlockLayout
    ROCmRingKVCache<Precision>::logicalBlockLayout(int global_layer, int token_count) const
    {
        KVCacheLogicalBlockLayout layout;
        layout.k_precision = Precision;
        layout.v_precision = Precision;
        layout.layout = TensorLayout::KV_POS_HEAD_DIM;
        layout.local_kv_heads = local_n_kv_heads_;
        layout.kv_head_start = kv_head_start_;
        layout.head_dim = head_dim_;
        layout.device_resident = true;

        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ || batch_size_ <= 0 || token_count <= 0)
        {
            return layout;
        }

        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        layout.k_bytes = static_cast<size_t>(token_count) * row_bytes;
        layout.v_bytes = static_cast<size_t>(token_count) * row_bytes;
        return layout;
    }

    template <ActivationPrecision Precision>
    typename IKVCache::KVCacheSequenceState
    ROCmRingKVCache<Precision>::sequenceState(int global_layer, int seq_idx) const
    {
        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_)
        {
            return {};
        }
        return ROCmRingKVCacheBase::sequenceState(local_layer, seq_idx);
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::exportLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const
    {
        requireGPUExecutionStream(
            desc.stream,
            "ROCmRingKVCache::exportLogicalBlock");
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0)
        {
            return false;
        }

        (void)hipSetDevice(device_id_);
        hipStream_t stream = static_cast<hipStream_t>(desc.stream);
        const auto &entry = entries_[local_layer][desc.seq_idx];
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (desc.token_count == 0)
                return true;
            if (!dst_k || !dst_v || !entry.d_K || !entry.d_V ||
                !d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] device-domain logical export storage unavailable");
                return false;
            }

            const int entry_index =
                local_layer * batch_size_ + desc.seq_idx;
            hipError_t launch_error = hipErrorInvalidValue;
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                launch_error = hip_ring_logical_block_export_device_fp32(
                    entry.d_K, entry.d_V,
                    static_cast<float *>(dst_k),
                    static_cast<float *>(dst_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                launch_error = hip_ring_logical_block_export_device_fp16(
                    entry.d_K, entry.d_V,
                    static_cast<_Float16 *>(dst_k),
                    static_cast<_Float16 *>(dst_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                launch_error = hip_ring_logical_block_export_device_bf16(
                    entry.d_K, entry.d_V,
                    static_cast<hip_bfloat16 *>(dst_k),
                    static_cast<hip_bfloat16 *>(dst_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                launch_error = hip_ring_logical_block_export_device_q8_1(
                    entry.d_K, entry.d_V,
                    static_cast<Q8_1Block *>(dst_k),
                    static_cast<Q8_1Block *>(dst_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            if (launch_error != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] device-domain gather launch failed: "
                          << hipGetErrorString(launch_error));
                return false;
            }
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "rocm_device_logical_kv_exports",
                1.0,
                "harvest",
                "rocm:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)},
                 {"precision", std::to_string(static_cast<int>(Precision))}});
            return true;
        }

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(local_layer, desc.seq_idx, &state))
            return false;
        const int export_head = state.implementation_head;
        const int export_count = state.cached_tokens;

        if (export_count < 0 || export_count > max_seq_len_ ||
            export_head < 0 || export_head >= max_seq_len_ ||
            desc.logical_token_start > export_count ||
            desc.token_count > export_count - desc.logical_token_start)
        {
            LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] logical export bounds rejected"
                      << " global_layer=" << desc.layer
                      << " local_layer=" << local_layer
                      << " seq_idx=" << desc.seq_idx
                      << " logical_start=" << desc.logical_token_start
                      << " token_count=" << desc.token_count
                      << " device_head=" << export_head
                      << " device_count=" << export_count
                      << " max_seq_len=" << max_seq_len_);
            return false;
        }
        if (desc.token_count == 0)
        {
            return true;
        }
        if (!dst_k || !dst_v || !entry.d_K || !entry.d_V)
        {
            LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] logical export storage unavailable"
                      << " global_layer=" << desc.layer
                      << " local_layer=" << local_layer
                      << " seq_idx=" << desc.seq_idx
                      << " token_count=" << desc.token_count
                      << " dst_k=" << dst_k
                      << " dst_v=" << dst_v
                      << " entry_k=" << entry.d_K
                      << " entry_v=" << entry.d_V);
            return false;
        }

        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        int tail = export_head - export_count;
        tail %= max_seq_len_;
        if (tail < 0)
        {
            tail += max_seq_len_;
        }
        auto *out_k = static_cast<uint8_t *>(dst_k);
        auto *out_v = static_cast<uint8_t *>(dst_v);

        for (int i = 0; i < desc.token_count; ++i)
        {
            const int logical = desc.logical_token_start + i;
            const int phys = (tail + logical) % max_seq_len_;
            const size_t dst_offset = static_cast<size_t>(i) * row_bytes;
            const size_t src_offset = static_cast<size_t>(phys) * static_cast<size_t>(kv_storage_dim_);

            hipError_t err = hipMemcpyAsync(out_k + dst_offset,
                                            entry.d_K + src_offset,
                                            row_bytes,
                                            hipMemcpyDeviceToHost,
                                            stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] K copy failed: "
                          << hipGetErrorString(err));
                return false;
            }
            err = hipMemcpyAsync(out_v + dst_offset,
                                 entry.d_V + src_offset,
                                 row_bytes,
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] V copy failed: "
                          << hipGetErrorString(err));
                return false;
            }
        }

        const hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] stream sync failed: "
                      << hipGetErrorString(sync_err));
            return false;
        }
        const size_t payload_bytes = static_cast<size_t>(desc.token_count) * row_bytes;
        if (!kv_cache_codec::canonicalizeFloatingZeros(dst_k, payload_bytes, Precision) ||
            !kv_cache_codec::canonicalizeFloatingZeros(dst_v, payload_bytes, Precision))
        {
            LOG_ERROR("[ROCmRingKVCache::exportLogicalBlock] logical payload canonicalization failed");
            return false;
        }
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::importLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v)
    {
        requireGPUExecutionStream(
            desc.stream,
            "ROCmRingKVCache::importLogicalBlock");
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start)
        {
            return false;
        }

        auto &entry = entries_[local_layer][desc.seq_idx];
        hipStream_t stream = static_cast<hipStream_t>(desc.stream);
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (desc.token_count == 0)
            {
                if (desc.logical_token_start != 0)
                    return true;
                if (!setDeviceSequenceState(
                        local_layer,
                        desc.seq_idx,
                        0,
                        0,
                        stream))
                {
                    return false;
                }
                invalidateRoPEShadow(local_layer, desc.seq_idx);
                return true;
            }
            if (!src_k || !src_v || !entry.d_K || !entry.d_V ||
                !d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[ROCmRingKVCache::importLogicalBlock] device-domain logical import storage unavailable");
                return false;
            }

            (void)hipSetDevice(device_id_);
            const int entry_index =
                local_layer * batch_size_ + desc.seq_idx;
            hipError_t launch_error = hipErrorInvalidValue;
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                launch_error = hip_ring_logical_block_import_device_fp32(
                    entry.d_K, entry.d_V,
                    static_cast<const float *>(src_k),
                    static_cast<const float *>(src_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                launch_error = hip_ring_logical_block_import_device_fp16(
                    entry.d_K, entry.d_V,
                    static_cast<const _Float16 *>(src_k),
                    static_cast<const _Float16 *>(src_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                launch_error = hip_ring_logical_block_import_device_bf16(
                    entry.d_K, entry.d_V,
                    static_cast<const hip_bfloat16 *>(src_k),
                    static_cast<const hip_bfloat16 *>(src_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                launch_error = hip_ring_logical_block_import_device_q8_1(
                    entry.d_K, entry.d_V,
                    static_cast<const Q8_1Block *>(src_k),
                    static_cast<const Q8_1Block *>(src_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_,
                    stream);
            }
            if (launch_error != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCache::importLogicalBlock] device-domain scatter launch failed: "
                          << hipGetErrorString(launch_error));
                return false;
            }
            invalidateRoPEShadow(local_layer, desc.seq_idx);
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "rocm_device_logical_kv_imports",
                1.0,
                "restore",
                "rocm:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)},
                 {"precision", std::to_string(static_cast<int>(Precision))}});
            return true;
        }

        if (desc.token_count == 0)
        {
            if (desc.logical_token_start == 0)
            {
                if (!setDeviceSequenceState(
                        local_layer, desc.seq_idx, 0, 0, stream))
                    return false;
                invalidateRoPEShadow(local_layer, desc.seq_idx);
            }
            return true;
        }
        if (!src_k || !src_v || !entry.d_K || !entry.d_V)
        {
            return false;
        }
        /*
         * RAM/SSD restore owns the logical block order. Keep that intentional
         * H2D boundary asynchronous and never observe canonical metadata on
         * the host just to repeat the restore planner's ordering decision.
         */
        (void)hipSetDevice(device_id_);
        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        const size_t bytes = static_cast<size_t>(desc.token_count) * row_bytes;
        const size_t dst_offset = static_cast<size_t>(desc.logical_token_start) *
                                  static_cast<size_t>(kv_storage_dim_);

        hipError_t err = hipMemcpyAsync(entry.d_K + dst_offset,
                                        src_k,
                                        bytes,
                                        hipMemcpyHostToDevice,
                                        stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache::importLogicalBlock] K copy failed: "
                      << hipGetErrorString(err));
            return false;
        }
        err = hipMemcpyAsync(entry.d_V + dst_offset,
                             src_v,
                             bytes,
                             hipMemcpyHostToDevice,
                             stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache::importLogicalBlock] V copy failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const int new_count = desc.logical_token_start + desc.token_count;
        const int new_head = new_count % max_seq_len_;
        if (!setDeviceSequenceState(
                local_layer, desc.seq_idx, new_head, new_count, stream))
            return false;

        invalidateRoPEShadow(local_layer, desc.seq_idx);
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::truncateSequence(int seq_idx, int cached_tokens, void *stream)
    {
        return ROCmRingKVCacheBase::truncateSequence(
            seq_idx,
            cached_tokens,
            stream);
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            num_tokens < 0)
        {
            return;
        }

        hipStream_t stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance()
                .getAMDContext(device_id_)
                .defaultStream());
        if (!evictOldestDeviceSequenceState(
                layer,
                seq_idx,
                num_tokens,
                stream))
        {
            LOG_ERROR("[ROCmRingKVCache::evict_oldest] Device metadata eviction launch failed");
            return;
        }
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
        requireGPUExecutionStream(
            static_cast<void *>(stream),
            "ROCmRingKVCache::gather_kv_batched");
        if (layer < 0 || layer >= n_layers_ || num_seqs > batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCache::gather_kv_batched] Invalid layer=" << layer);
            return -1;
        }

        int actual_max_kv_len = 0;

        for (int seq = 0; seq < num_seqs; ++seq)
        {
            KVCacheSequenceState state;
            if (!observeDeviceSequenceState(layer, seq, &state))
                return -1;
            kv_lens[seq] = state.cached_tokens;
            actual_max_kv_len = std::max(actual_max_kv_len, kv_lens[seq]);
        }

        if (actual_max_kv_len == 0)
        {
            return 0;
        }

        // Use provided max_kv_len or actual
        int out_max_kv_len = (max_kv_len > 0) ? max_kv_len : actual_max_kv_len;

        if (!d_k_out || !d_v_out ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_ ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_)
            return -1;

        const int entry_offset = layer * batch_size_;
        hipError_t launch_error = hipErrorInvalidValue;
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            launch_error = hip_ring_gather_batched_device_state_fp32(
                static_cast<float *>(d_k_out), static_cast<float *>(d_v_out),
                const_cast<const float *const *>(d_batched_k_entry_table_),
                const_cast<const float *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, num_seqs,
                out_max_kv_len, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/true, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            launch_error = hip_ring_gather_batched_device_state_fp16(
                static_cast<_Float16 *>(d_k_out), static_cast<_Float16 *>(d_v_out),
                const_cast<const _Float16 *const *>(d_batched_k_entry_table_),
                const_cast<const _Float16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, num_seqs,
                out_max_kv_len, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/true, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            launch_error = hip_ring_gather_batched_device_state_bf16(
                static_cast<hip_bfloat16 *>(d_k_out), static_cast<hip_bfloat16 *>(d_v_out),
                const_cast<const hip_bfloat16 *const *>(d_batched_k_entry_table_),
                const_cast<const hip_bfloat16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, num_seqs,
                out_max_kv_len, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/true, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            launch_error = hip_ring_gather_batched_device_state_q8_1(
                static_cast<Q8_1Block *>(d_k_out), static_cast<Q8_1Block *>(d_v_out),
                const_cast<const Q8_1Block *const *>(d_batched_k_entry_table_),
                const_cast<const Q8_1Block *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, num_seqs,
                out_max_kv_len, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/true, stream);
        }
        if (launch_error != hipSuccess)
            return -1;

        return actual_max_kv_len;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_device_ring_view(
        int layer,
        int seq_idx,
        ITensor **out_k,
        ITensor **out_v,
        const int **device_head,
        const int **device_count,
        int *physical_capacity,
        void *gpu_stream)
    {
        requireGPUExecutionStream(
            gpu_stream,
            "ROCmRingKVCache::get_kv_device_ring_view");
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

        if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            return false;
        }
        else
        {
            if (layer < 0 || layer >= n_layers_ || seq_idx < 0 ||
                seq_idx >= batch_size_ || max_seq_len_ <= 0 ||
                device_ring_views_.size() !=
                    static_cast<std::size_t>(n_layers_) ||
                !ROCmRingKVCacheBase::deviceRingHeadPtr(layer, seq_idx) ||
                !ROCmRingKVCacheBase::deviceCachedTokenCountPtr(
                    layer, seq_idx))
            {
                LOG_ERROR("[ROCmRingKVCache::get_kv_device_ring_view] Incomplete native ring contract"
                          << " layer=" << layer
                          << " seq=" << seq_idx
                          << " capacity=" << max_seq_len_
                          << " stream=" << gpu_stream);
                return false;
            }

            auto &entry = entries_[layer][seq_idx];
            if (!entry.d_K || !entry.d_V)
            {
                LOG_ERROR("[ROCmRingKVCache::get_kv_device_ring_view] Native ring storage is missing"
                          << " layer=" << layer << " seq=" << seq_idx);
                return false;
            }

            constexpr TensorType tensor_type = []() constexpr
            {
                if constexpr (Precision == ActivationPrecision::FP16)
                    return TensorType::FP16;
                if constexpr (Precision == ActivationPrecision::BF16)
                    return TensorType::BF16;
                return TensorType::FP32;
            }();
            auto &views = device_ring_views_[layer][seq_idx];
            if (!views[0])
            {
                views[0] = std::make_unique<GpuTensorView>(
                    entry.d_K,
                    static_cast<std::size_t>(max_seq_len_),
                    static_cast<std::size_t>(kv_dim_),
                    tensor_type,
                    DeviceId::rocm(device_id_));
            }
            if (!views[1])
            {
                views[1] = std::make_unique<GpuTensorView>(
                    entry.d_V,
                    static_cast<std::size_t>(max_seq_len_),
                    static_cast<std::size_t>(kv_dim_),
                    tensor_type,
                    DeviceId::rocm(device_id_));
            }

            if (out_k)
                *out_k = views[0].get();
            if (out_v)
                *out_v = views[1].get();
            if (device_head)
                *device_head =
                    ROCmRingKVCacheBase::deviceRingHeadPtr(layer, seq_idx);
            if (device_count)
                *device_count =
                    ROCmRingKVCacheBase::deviceCachedTokenCountPtr(
                        layer, seq_idx);
            if (physical_capacity)
                *physical_capacity = max_seq_len_;
            return true;
        }
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_batched_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ITensor **out_k,
        ITensor **out_v,
        void *gpu_stream)
    {
        requireGPUExecutionStream(
            gpu_stream,
            "ROCmRingKVCache::get_kv_batched_device_view");
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;

        const bool request_range_valid =
            first_seq_idx >= 0 && request_count > 0 &&
            request_count <= batch_size_ &&
            first_seq_idx <= batch_size_ - request_count;
        if (layer < 0 || layer >= n_layers_ || !request_range_valid ||
            !workspace_ || !workspace_->isAllocated() ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_device_view] Invalid resident gather contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " batch_capacity=" << batch_size_
                      << " max_seq_len=" << max_seq_len_
                      << " stream=" << gpu_stream
                      << " workspace=" << (workspace_ ? "bound" : "missing")
                      << " pointer_tables="
                      << (batched_pointer_tables_ready_ ? "ready" : "missing"));
            return false;
        }

        const size_t rows =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(max_seq_len_);
        if (kv_storage_dim_ <= 0 ||
            rows > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(kv_storage_dim_) ||
            rows * static_cast<size_t>(kv_storage_dim_) >
                std::numeric_limits<size_t>::max() / sizeof(DataT))
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_device_view] Gather byte size overflow");
            return false;
        }
        const size_t required_bytes =
            rows * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        if (!ensureConvScratch(required_bytes) ||
            !conv_scratch_k_ || !conv_scratch_v_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_device_view] Native gather scratch is unavailable"
                      << " required_bytes=" << required_bytes);
            return false;
        }

        if (!d_batched_k_entry_table_ || !d_batched_v_entry_table_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_device_view] Missing cache-owned immutable entry pointer tables");
            return false;
        }

        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const auto stream = static_cast<hipStream_t>(gpu_stream);
        hipError_t launch_error = hipErrorInvalidValue;
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            launch_error = hip_ring_gather_batched_device_state_fp32(
                static_cast<float *>(conv_scratch_k_),
                static_cast<float *>(conv_scratch_v_),
                const_cast<const float *const *>(d_batched_k_entry_table_),
                const_cast<const float *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/false, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            launch_error = hip_ring_gather_batched_device_state_fp16(
                static_cast<_Float16 *>(conv_scratch_k_),
                static_cast<_Float16 *>(conv_scratch_v_),
                const_cast<const _Float16 *const *>(d_batched_k_entry_table_),
                const_cast<const _Float16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/false, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            launch_error = hip_ring_gather_batched_device_state_bf16(
                static_cast<hip_bfloat16 *>(conv_scratch_k_),
                static_cast<hip_bfloat16 *>(conv_scratch_v_),
                const_cast<const hip_bfloat16 *const *>(d_batched_k_entry_table_),
                const_cast<const hip_bfloat16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/false, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            launch_error = hip_ring_gather_batched_device_state_q8_1(
                static_cast<Q8_1Block *>(conv_scratch_k_),
                static_cast<Q8_1Block *>(conv_scratch_v_),
                const_cast<const Q8_1Block *const *>(d_batched_k_entry_table_),
                const_cast<const Q8_1Block *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_storage_dim_,
                /*zero_inactive_rows=*/false, stream);
        }
        if (launch_error != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_device_view] Gather launch failed: "
                      << hipGetErrorString(launch_error));
            return false;
        }

        constexpr TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            return TensorType::FP32;
        }();

        const size_t columns = static_cast<size_t>(kv_storage_dim_);
        if (!batched_k_view_ ||
            batched_k_view_->gpu_data_ptr() != conv_scratch_k_ ||
            batched_k_view_->shape().size() < 2 ||
            batched_k_view_->shape()[0] != rows ||
            batched_k_view_->shape()[1] != columns)
        {
            batched_k_view_ = std::make_unique<GpuTensorView>(
                conv_scratch_k_, rows, columns, tensor_type,
                DeviceId::rocm(device_id_));
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != conv_scratch_v_ ||
            batched_v_view_->shape().size() < 2 ||
            batched_v_view_->shape()[0] != rows ||
            batched_v_view_->shape()[1] != columns)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                conv_scratch_v_, rows, columns, tensor_type,
                DeviceId::rocm(device_id_));
        }

        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_batched_converted_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ActivationPrecision target,
        ITensor **out_k,
        ITensor **out_v,
        const KVReadParams &read)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;

        const int requested_heads =
            read.n_kv_heads > 0 ? read.n_kv_heads : local_n_kv_heads_;
        const int requested_head_dim =
            read.head_dim > 0 ? read.head_dim : head_dim_;
        const int effective_rope_dim =
            read.rope_dim > 0 ? read.rope_dim : requested_head_dim;
        if (layer < 0 || layer >= n_layers_ || first_seq_idx < 0 ||
            request_count <= 0 ||
            first_seq_idx > batch_size_ - request_count ||
            target != ActivationPrecision::FP16 || !read.gpu_stream ||
            requested_heads != local_n_kv_heads_ ||
            requested_head_dim != head_dim_ ||
            effective_rope_dim <= 0 || effective_rope_dim > head_dim_ ||
            (effective_rope_dim % 2) != 0 ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_ ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_converted_device_view] Invalid grouped conversion contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " target=" << activationPrecisionToString(target)
                      << " heads=" << requested_heads
                      << " head_dim=" << requested_head_dim
                      << " rope_dim=" << effective_rope_dim
                      << " stream=" << read.gpu_stream);
            return false;
        }

        const size_t rows =
            static_cast<size_t>(request_count) * max_seq_len_;
        const size_t required_bytes =
            rows * static_cast<size_t>(kv_dim_) * sizeof(_Float16);
        if (!ensureConvScratch(required_bytes) ||
            !conv_scratch_k_ || !conv_scratch_v_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_converted_device_view] FP16 grouped workspace is unavailable"
                      << " required_bytes=" << required_bytes);
            return false;
        }

        auto *k_output = static_cast<_Float16 *>(conv_scratch_k_);
        auto *v_output = static_cast<_Float16 *>(conv_scratch_v_);
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        auto stream = static_cast<hipStream_t>(read.gpu_stream);
        hipError_t status = hipErrorInvalidValue;
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            status = hip_ring_gather_batched_converted_device_state_fp32(
                k_output, v_output,
                const_cast<const float *const *>(d_batched_k_entry_table_),
                const_cast<const float *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_dim_, local_n_kv_heads_, head_dim_,
                read.rope_theta, read.position_start, effective_rope_dim, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            status = hip_ring_gather_batched_converted_device_state_fp16(
                k_output, v_output,
                const_cast<const _Float16 *const *>(d_batched_k_entry_table_),
                const_cast<const _Float16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_dim_, local_n_kv_heads_, head_dim_,
                read.rope_theta, read.position_start, effective_rope_dim, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            status = hip_ring_gather_batched_converted_device_state_bf16(
                k_output, v_output,
                const_cast<const hip_bfloat16 *const *>(d_batched_k_entry_table_),
                const_cast<const hip_bfloat16 *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_dim_, local_n_kv_heads_, head_dim_,
                read.rope_theta, read.position_start, effective_rope_dim, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            status = hip_ring_gather_batched_converted_device_state_q8_1(
                k_output, v_output,
                const_cast<const Q8_1Block *const *>(d_batched_k_entry_table_),
                const_cast<const Q8_1Block *const *>(d_batched_v_entry_table_),
                d_head_params_, d_count_params_, entry_offset, request_count,
                max_seq_len_, max_seq_len_, kv_dim_, local_n_kv_heads_, head_dim_,
                read.rope_theta, read.position_start, effective_rope_dim, stream);
        }
        if (status != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_batched_converted_device_view] Grouped conversion launch failed: "
                      << hipGetErrorString(status));
            return false;
        }

        if (!converted_batched_k_view_ ||
            converted_batched_k_view_->gpu_data_ptr() != k_output ||
            converted_batched_k_view_->shape().size() < 2 ||
            converted_batched_k_view_->shape()[0] != rows)
        {
            converted_batched_k_view_ = std::make_unique<GpuTensorView>(
                k_output, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (!converted_batched_v_view_ ||
            converted_batched_v_view_->gpu_data_ptr() != v_output ||
            converted_batched_v_view_->shape().size() < 2 ||
            converted_batched_v_view_->shape()[0] != rows)
        {
            converted_batched_v_view_ = std::make_unique<GpuTensorView>(
                v_output, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (out_k)
            *out_k = converted_batched_k_view_.get();
        if (out_v)
            *out_v = converted_batched_v_view_.get();
        return true;
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
                DeviceId::rocm(device_id_));

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
                DeviceId::rocm(device_id_));

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
        /*
         * New callers pass m as the active graph bucket and n as the request
         * batch. Resident attention gathers the complete post-append KV horizon,
         * which can exceed the current prefill chunk after the first chunk. Size
         * conversion scratch to the cache's configured sequence horizon so graph
         * replay never outgrows its stable workspace pointers.
         *
         * Legacy one-argument callers use m as a batch hint. Preserve that API
         * and reserve at least the cache's configured batch capacity.
         */
        (void)k;
        const auto geometry = kv_cache_workspace::ConversionGeometry{
            .configured_batch_size = batch_size_,
            .configured_context_rows = max_seq_len_,
            .requested_graph_rows = m,
            .requested_batch_size = n,
            .conversion_row_bytes =
                static_cast<size_t>(kv_dim_) * sizeof(float),
            .native_row_bytes =
                static_cast<size_t>(kv_storage_dim_) * sizeof(DataT),
        };
        WorkspaceRequirements reqs =
            kv_cache_workspace::conversionRequirements(geometry);
        const size_t conversion_scratch_bytes =
            kv_cache_workspace::conversionBufferBytes(geometry);

        LOG_TRACE("[ROCmRingKVCache] Workspace requirements: configured_batch="
                  << batch_size_
                  << " configured_context=" << max_seq_len_
                  << " requested_graph_rows=" << m
                  << " requested_batch=" << n
                  << " CONV_SCRATCH(each)=" << conversion_scratch_bytes);

        return reqs;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        if (!batched_pointer_tables_ready_ ||
            (entry_count > 0 &&
             (!d_batched_k_entry_table_ || !d_batched_v_entry_table_)))
        {
            throw std::logic_error(
                "[ROCmRingKVCache] Cannot bind workspace before immutable entry topology is published");
        }

        // WorkspaceAllocator intentionally re-presents the current manager on
        // every graph execution. Preserve all wrappers and scratch ownership
        // when the identity is unchanged.
        if (workspace_ == workspace)
            return;

        if (isGraphCaptureActive())
        {
            throw std::runtime_error(
                "[ROCmRingKVCache] Workspace ownership cannot change during HIP graph capture");
        }
        if (workspace && !workspace->isAllocated())
        {
            throw std::invalid_argument(
                "[ROCmRingKVCache] A bound workspace must be fully allocated");
        }

        void *new_scratch_k = nullptr;
        void *new_scratch_v = nullptr;
        size_t new_scratch_capacity = 0;
        if (workspace)
        {
            new_scratch_k =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            new_scratch_v =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            const size_t k_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            const size_t v_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            if (!new_scratch_k || !new_scratch_v ||
                k_capacity == 0 || v_capacity == 0)
            {
                throw std::runtime_error(
                    "[ROCmRingKVCache] Bound workspace lacks mandatory grouped conversion scratch");
            }
            new_scratch_capacity = std::min(k_capacity, v_capacity);
        }

        // Detach or release the old scratch before publishing the new binding.
        // This transition is lifecycle-only; identical hot-path binds returned
        // above without allocation, deallocation, or view invalidation.
        freeConvScratch();
        workspace_ = workspace;
        conv_scratch_k_ = new_scratch_k;
        conv_scratch_v_ = new_scratch_v;
        conv_scratch_capacity_ = new_scratch_capacity;
        conv_scratch_workspace_backed_ = workspace != nullptr;
        batched_k_view_.reset();
        batched_v_view_.reset();
        converted_batched_k_view_.reset();
        converted_batched_v_view_.reset();

        LOG_TRACE("[ROCmRingKVCache] Workspace bound: "
                  << (workspace ? "yes" : "no"));
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
                const_cast<void *>(d_k), rows, view_cols, tensor_type,
                DeviceId::rocm(device_id_));
        }

        // Update V view (index 1)
        auto &v_view = tensor_views_[layer][seq_idx][1];
        if (!v_view || v_view->gpu_data_ptr() != d_v || v_view->rows() != rows)
        {
            v_view = std::make_unique<GpuTensorView>(
                const_cast<void *>(d_v), rows, view_cols, tensor_type,
                DeviceId::rocm(device_id_));
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

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_snapshot_view(int layer, int seq_idx,
                                                          int token_count,
                                                          ITensor **out_k, ITensor **out_v,
                                                          int *out_kv_len)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (out_kv_len)
            *out_kv_len = 0;

        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            token_count <= 0 || token_count > max_seq_len_)
        {
            return false;
        }

        const EntryT &entry = entries_[layer][seq_idx];
        KVCacheSequenceState state;
        if (!entry.d_K || !entry.d_V ||
            !observeDeviceSequenceState(layer, seq_idx, &state) ||
            token_count < state.cached_tokens)
            return false;
        const int tail =
            (state.implementation_head - state.cached_tokens + max_seq_len_) %
            max_seq_len_;
        if (state.cached_tokens > 0 && tail != 0)
            return false;
        if (state.implementation_head != state.cached_tokens)
            return false;

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

        if (snapshot_tensor_views_.empty())
        {
            snapshot_tensor_views_.resize(n_layers_);
            for (int l = 0; l < n_layers_; ++l)
                snapshot_tensor_views_[l].resize(batch_size_);
        }

        const size_t rows = static_cast<size_t>(token_count);
        const size_t view_cols = (Precision == ActivationPrecision::Q8_1)
                                     ? static_cast<size_t>(kv_storage_dim_)
                                     : static_cast<size_t>(kv_dim_);

        auto &k_view = snapshot_tensor_views_[layer][seq_idx][0];
        if (!k_view || k_view->gpu_data_ptr() != entry.d_K || k_view->rows() != rows)
        {
            k_view = std::make_unique<GpuTensorView>(
                static_cast<void *>(entry.d_K), rows, view_cols, tensor_type,
                DeviceId::rocm(device_id_));
        }

        auto &v_view = snapshot_tensor_views_[layer][seq_idx][1];
        if (!v_view || v_view->gpu_data_ptr() != entry.d_V || v_view->rows() != rows)
        {
            v_view = std::make_unique<GpuTensorView>(
                static_cast<void *>(entry.d_V), rows, view_cols, tensor_type,
                DeviceId::rocm(device_id_));
        }

        if (out_k)
            *out_k = k_view.get();
        if (out_v)
            *out_v = v_view.get();
        if (out_kv_len)
            *out_kv_len = token_count;
        return true;
    }

    template <ActivationPrecision Precision>
    bool ROCmRingKVCache<Precision>::get_kv_snapshot_view(int layer, int seq_idx,
                                                          int token_count,
                                                          const ITensor **out_k, const ITensor **out_v,
                                                          int *out_kv_len) const
    {
        ITensor *k = nullptr;
        ITensor *v = nullptr;
        const bool ok = const_cast<ROCmRingKVCache<Precision> *>(this)->get_kv_snapshot_view(
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

    // =========================================================================
    // RoPE Shadow Buffer Helpers (for get_kv_converted)
    // =========================================================================

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::ensureRoPEShadow(int layer, int seq_idx)
    {
        if (rope_shadows_.empty())
        {
            rope_shadows_.resize(n_layers_);
            for (auto &layer_shadows : rope_shadows_)
                layer_shadows.resize(batch_size_);
        }

        auto &shadow = rope_shadows_[layer][seq_idx];
        const size_t fp16_bytes =
            static_cast<size_t>(max_seq_len_) *
            static_cast<size_t>(kv_dim_) * sizeof(_Float16);
        if (!ensureConvScratch(fp16_bytes))
        {
            throw std::runtime_error(
                "[ROCmRingKVCache] RoPE conversion requires bound K/V workspace");
        }
        shadow.d_K = conv_scratch_k_;
        shadow.d_V = conv_scratch_v_;
    }

    template <ActivationPrecision Precision>
    void ROCmRingKVCache<Precision>::invalidateRoPEShadow(int layer, int seq_idx) const
    {
        if (rope_shadows_.empty())
            return;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        auto &shadow = rope_shadows_[layer][seq_idx];
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

        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (out_kv_len)
            *out_kv_len = 0;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_converted] Scalar conversion is forbidden during graph capture; use get_kv_batched_converted_device_view");
            return false;
        }

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return false;
        const int requested_token_count =
            rope && rope->requested_token_count > 0
                ? rope->requested_token_count
                : 0;
        if (requested_token_count > 0 &&
            requested_token_count != state.cached_tokens)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_converted] Requested scalar span does not match canonical device count"
                      << " requested=" << requested_token_count
                      << " device_count=" << state.cached_tokens);
            return false;
        }
        const int read_count = state.cached_tokens;
        if (read_count == 0)
            return true;

        // If no RoPE is requested, the raw cache tensors already have the
        // required representation.
        // Use qualified call to avoid virtual dispatch — ROCmHybridRingKVCache
        // overrides get_kv() with layer remapping, but the layer index passed here
        // has already been remapped by the hybrid override of get_kv_converted().
        const bool want_rope = (rope && rope->rope_theta > 0.0f);
        if (!want_rope)
            return ROCmRingKVCache::get_kv(layer, seq_idx, out_k, out_v, out_kv_len);

        (void)hipSetDevice(device_id_);
        const hipStream_t stream = getEffectiveStream(
            rope ? static_cast<hipStream_t>(rope->gpu_stream) : nullptr);

        /*
         * Reuse the canonical grouped device-state conversion even for this
         * scalar observation boundary. The grouped kernel reads the native ring
         * directly and writes FP16+RoPE output once, so FP32/BF16/Q8_1 never
         * need a second native-format temporary or an allocation fallback.
         */
        KVReadParams grouped_read = *rope;
        grouped_read.gpu_stream = stream;
        grouped_read.n_kv_heads = local_n_kv_heads_;
        grouped_read.head_dim = head_dim_;
        ITensor *grouped_k = nullptr;
        ITensor *grouped_v = nullptr;
        if (!get_kv_batched_converted_device_view(
                layer,
                seq_idx,
                1,
                ActivationPrecision::FP16,
                &grouped_k,
                &grouped_v,
                grouped_read))
        {
            return false;
        }

        ensureRoPEShadow(layer, seq_idx);
        auto &shadow = rope_shadows_[layer][seq_idx];
        if (!shadow.d_K || !shadow.d_V || read_count > max_seq_len_)
        {
            LOG_ERROR("[ROCmRingKVCache::get_kv_converted] Scalar conversion storage is unavailable");
            return false;
        }

        if (hipGetLastError() != hipSuccess)
            return false;

        // Create/update GpuTensorViews for the shadow buffers
        if (!shadow.k_view || shadow.k_view->shape()[0] != static_cast<size_t>(read_count))
        {
            shadow.k_view = std::make_unique<GpuTensorView>(
                shadow.d_K, read_count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (!shadow.v_view || shadow.v_view->shape()[0] != static_cast<size_t>(read_count))
        {
            shadow.v_view = std::make_unique<GpuTensorView>(
                shadow.d_V, read_count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }

        if (out_k)
            *out_k = shadow.k_view.get();
        if (out_v)
            *out_v = shadow.v_view.get();
        if (out_kv_len)
            *out_kv_len = read_count;

        return true;
    }

    // =========================================================================
    // Explicit Template Instantiations
    // =========================================================================

    template class ROCmRingKVCache<ActivationPrecision::FP32>;
    template class ROCmRingKVCache<ActivationPrecision::FP16>;
    template class ROCmRingKVCache<ActivationPrecision::BF16>;
    template class ROCmRingKVCache<ActivationPrecision::Q8_1>;

} // namespace llaminar2
