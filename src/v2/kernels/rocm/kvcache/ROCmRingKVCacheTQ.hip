/**
 * @file ROCmRingKVCacheTQ.hip
 * @brief HIP implementation of ROCm TQ KV cache
 * @author David Sanftenberg
 *
 * V2: Per-layer FP16 scratch, fused incremental kernel, graph capture support.
 */

#include "ROCmRingKVCacheTQ.h"
#include "ROCmTurboQuantKernels.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    extern "C" void hip_kv_sequence_state_advance(
        int *d_head, int *d_count, int num_tokens, int max_seq_len,
        hipStream_t stream);
    extern "C" void hip_kv_sequence_state_advance_dynamic(
        int *d_head, int *d_count, const int *d_append_count,
        int captured_num_tokens, int max_seq_len, hipStream_t stream);

    /**
     * @brief Gather asymmetric TurboQuant K/V rows into device prefix storage.
     *
     * TurboQuant stores K and V with different native row widths. Treating the
     * payload as raw bytes preserves every quantized code and scale exactly
     * while allowing one launch to cover both arrays. The canonical ring head
     * and count remain device-owned inputs, so graph replay never consults a
     * potentially stale host mirror.
     */
    __global__ void rocm_tq_ring_logical_block_export_device_kernel(
        const uint8_t *__restrict__ ring_k,
        const uint8_t *__restrict__ ring_v,
        uint8_t *__restrict__ block_k,
        uint8_t *__restrict__ block_v,
        const int *__restrict__ ring_head,
        const int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len,
        size_t k_row_bytes,
        size_t v_row_bytes)
    {
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count < 0 || count > max_seq_len ||
            head < 0 || head >= max_seq_len ||
            logical_token_start < 0 ||
            logical_token_start > count ||
            token_count < 0 ||
            token_count > count - logical_token_start)
        {
            return;
        }

        int tail = (head - count) % max_seq_len;
        if (tail < 0)
            tail += max_seq_len;

        const size_t k_bytes =
            static_cast<size_t>(token_count) * k_row_bytes;
        const size_t v_bytes =
            static_cast<size_t>(token_count) * v_row_bytes;
        const size_t total = k_bytes + v_bytes;
        for (size_t linear =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const bool is_k = linear < k_bytes;
            const size_t local = is_k ? linear : linear - k_bytes;
            const size_t row_bytes = is_k ? k_row_bytes : v_row_bytes;
            const int token = static_cast<int>(local / row_bytes);
            const size_t byte_in_row = local % row_bytes;
            const int physical =
                (tail + logical_token_start + token) % max_seq_len;
            const size_t source =
                static_cast<size_t>(physical) * row_bytes + byte_in_row;
            if (is_k)
                block_k[local] = ring_k[source];
            else
                block_v[local] = ring_v[source];
        }
    }

    /**
     * @brief Scatter a device-resident TurboQuant block into the live ring.
     *
     * The kernel validates the live device metadata before writing. A stale or
     * out-of-order restore therefore cannot mutate payload storage, and the
     * stream-ordered publication kernel below will likewise refuse to advance
     * the metadata.
     */
    __global__ void rocm_tq_ring_logical_block_import_device_kernel(
        uint8_t *__restrict__ ring_k,
        uint8_t *__restrict__ ring_v,
        const uint8_t *__restrict__ block_k,
        const uint8_t *__restrict__ block_v,
        const int *__restrict__ ring_head,
        const int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len,
        size_t k_row_bytes,
        size_t v_row_bytes)
    {
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count != logical_token_start ||
            head != (logical_token_start % max_seq_len))
        {
            return;
        }

        const size_t k_bytes =
            static_cast<size_t>(token_count) * k_row_bytes;
        const size_t v_bytes =
            static_cast<size_t>(token_count) * v_row_bytes;
        const size_t total = k_bytes + v_bytes;
        for (size_t linear =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const bool is_k = linear < k_bytes;
            const size_t local = is_k ? linear : linear - k_bytes;
            const size_t row_bytes = is_k ? k_row_bytes : v_row_bytes;
            const int token = static_cast<int>(local / row_bytes);
            const size_t byte_in_row = local % row_bytes;
            const int physical = (head + token) % max_seq_len;
            const size_t destination =
                static_cast<size_t>(physical) * row_bytes + byte_in_row;
            if (is_k)
                ring_k[destination] = block_k[local];
            else
                ring_v[destination] = block_v[local];
        }
    }

    /**
     * @brief Publish TQ ring metadata after asynchronous device payload import.
     *
     * This one-thread kernel is deliberately separate from the scatter. Stream
     * ordering guarantees that no consumer can observe the new count before
     * every payload byte has reached its canonical ring location.
     */
    __global__ void rocm_tq_ring_logical_block_import_publish_kernel(
        int *__restrict__ ring_head,
        int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count != logical_token_start ||
            head != (logical_token_start % max_seq_len))
        {
            return;
        }
        const int new_count = logical_token_start + token_count;
        *cached_tokens = new_count;
        *ring_head = new_count % max_seq_len;
    }

    ROCmRingKVCacheTQ::ROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx, int device_id,
        TurboQuantKVMode mode)
        : ROCmRingKVCacheTQ(n_layers, batch_size, max_seq_len,
                            n_kv_heads, n_kv_heads, 0,
                            head_dim, tq_ctx, device_id, mode)
    {
    }

    ROCmRingKVCacheTQ::ROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx, int device_id,
        TurboQuantKVMode mode)
        : ROCmRingKVCacheBase(n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, local_n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(local_n_kv_heads),
          kv_head_start_(kv_head_start),
          mode_(mode)
    {
        if (local_n_kv_heads <= 0 || local_n_kv_heads > n_kv_heads ||
            kv_head_start < 0 || kv_head_start + local_n_kv_heads > n_kv_heads)
        {
            throw std::invalid_argument("ROCmRingKVCacheTQ: invalid LocalTP KV-head range");
        }
        (void)hipSetDevice(device_id_);

        auto construction_cleanup = [this](void *) noexcept
        {
            releaseOwnedDeviceStorage();
        };
        std::unique_ptr<void, decltype(construction_cleanup)>
            construction_guard(this, construction_cleanup);

        if (head_dim_ == 64)
        {
            tq8_block_size_ = sizeof(TQ8Block<64>);
            v_block_size_ = mode_ == TurboQuantKVMode::TQ8_K_TQ8_V
                                ? sizeof(TQ8Block<64>)
                                : sizeof(TQ4Block<64>);
        }
        else if (head_dim_ == 128)
        {
            tq8_block_size_ = sizeof(TQ8Block<128>);
            v_block_size_ = mode_ == TurboQuantKVMode::TQ8_K_TQ8_V
                                ? sizeof(TQ8Block<128>)
                                : sizeof(TQ4Block<128>);
        }
        else if (head_dim_ == 256)
        {
            tq8_block_size_ = sizeof(TQ8Block<256>);
            v_block_size_ = mode_ == TurboQuantKVMode::TQ8_K_TQ8_V
                                ? sizeof(TQ8Block<256>)
                                : sizeof(TQ4Block<256>);
        }
        else
        {
            throw std::runtime_error(
                "ROCmRingKVCacheTQ: head_dim must be 64, 128, or 256");
        }

        if (!tq_ctx)
        {
            throw std::runtime_error(
                "ROCmRingKVCacheTQ requires TurboQuantContext");
        }
        const uint64_t rotation_seed = tq_ctx->rotation().seed;

        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        if (!init_stream)
            throw std::runtime_error("ROCmRingKVCacheTQ: explicit initialization stream unavailable");

        if (!hip_tq_upload_codebooks(init_stream))
            throw std::runtime_error(
                "ROCmRingKVCacheTQ: failed to enqueue constant codebook publication");
        rotations_ = hip_tq_create_rotations(n_layers_, local_n_kv_heads_, head_dim_,
                                              rotation_seed, device_id_, init_stream, kv_head_start_);

        const size_t k_entry_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * tq8_block_size_;
        const size_t v_entry_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * v_block_size_;
        auto *const backend = getROCmBackend();
        if (!backend)
        {
            throw std::runtime_error(
                "ROCmRingKVCacheTQ: ROCm backend unavailable during cache allocation");
        }

        entries_.resize(n_layers_);

        for (int l = 0; l < n_layers_; ++l)
        {
            entries_[l].resize(batch_size_);

            for (int s = 0; s < batch_size_; ++s)
            {
                auto &entry = entries_[l][s];
                entry.d_K = backend->allocate(k_entry_bytes, device_id_);
                entry.d_V = backend->allocate(v_entry_bytes, device_id_);
                if (!entry.d_K || !entry.d_V ||
                    hipMemsetAsync(entry.d_K, 0, k_entry_bytes, init_stream) != hipSuccess ||
                    hipMemsetAsync(entry.d_V, 0, v_entry_bytes, init_stream) != hipSuccess)
                {
                    throw std::runtime_error(
                        "ROCmRingKVCacheTQ: failed to allocate or initialize compressed entry storage");
                }
            }
        }
        if (!publishBatchedEntryTables(init_stream))
        {
            throw std::runtime_error(
                "ROCmRingKVCacheTQ: failed to publish resident batched entry topology");
        }

        // Grouped verifier graph replay and device-resident MTP publication
        // share these canonical base head/count scalars.
        allocateDeviceParams();

        // Conversion addresses arrive from the graph workspace. The cache
        // keeps only per-layer wrappers over that shared producer/consumer
        // storage, never one private full-context allocation per layer.
        layer_scratch_.resize(n_layers_);
        for (auto &scratch : layer_scratch_)
            scratch.invalidate();

        const hipError_t init_sync = hipStreamSynchronize(init_stream);
        if (init_sync != hipSuccess)
            throw std::runtime_error(std::string("ROCmRingKVCacheTQ initialization failed: ") +
                                     hipGetErrorString(init_sync));

        construction_guard.release();
        LOG_DEBUG("ROCmRingKVCacheTQ: created on device " << device_id_
                  << " (" << n_layers_ << " layers, " << batch_size_ << " batch, "
                  << max_seq_len_ << " maxseq, local/total_heads="
                  << local_n_kv_heads_ << "/" << n_kv_heads_ << ", head_dim=" << head_dim_
                  << ", TQ8+TQ4, FP16 scratch)");
    }

    ROCmRingKVCacheTQ::~ROCmRingKVCacheTQ()
    {
        (void)hipSetDevice(device_id_);
        releaseOwnedDeviceStorage();
    }

    void ROCmRingKVCacheTQ::releaseOwnedDeviceStorage() noexcept
    {
        auto *const backend = getROCmBackend();
        bool owns_device_storage =
            d_batched_k_entry_table_ || d_batched_v_entry_table_ ||
            rotations_.d_rotations || rotations_.d_rotations_t;
        for (const auto &layer : entries_)
            for (const auto &entry : layer)
                owns_device_storage = owns_device_storage || entry.d_K || entry.d_V;
        if (owns_device_storage && !backend)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] ROCm backend unavailable during cache teardown");
            std::terminate();
        }

        try
        {
            batched_k_view_.reset();
            batched_v_view_.reset();
            if (d_batched_k_entry_table_)
                backend->free(d_batched_k_entry_table_, device_id_);
            if (d_batched_v_entry_table_)
                backend->free(d_batched_v_entry_table_, device_id_);
            d_batched_k_entry_table_ = nullptr;
            d_batched_v_entry_table_ = nullptr;

            for (auto &layer : entries_)
            {
                for (auto &entry : layer)
                {
                    if (entry.d_K)
                        backend->free(entry.d_K, device_id_);
                    if (entry.d_V)
                        backend->free(entry.d_V, device_id_);
                    entry.d_K = nullptr;
                    entry.d_V = nullptr;
                }
            }

            for (auto &scratch : layer_scratch_)
            {
                scratch.k_view.reset();
                scratch.v_view.reset();
                scratch.d_K = nullptr;
                scratch.d_V = nullptr;
            }
            layer_scratch_.clear();
            workspace_ = nullptr;
            scratch_capacity_bytes_ = 0;

            hip_tq_free_rotations(rotations_);
        }
        catch (...)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] ROCm backend threw during cache teardown");
            std::terminate();
        }
    }

    WorkspaceRequirements ROCmRingKVCacheTQ::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        (void)k;
        const bool has_token_hint = n > 0;
        const int hinted_batch =
            has_token_hint ? n : ((m > 0) ? m : batch_size_);
        const int scratch_tokens =
            has_token_hint ? std::max(m, max_seq_len_) : max_seq_len_;
        const size_t rows =
            static_cast<size_t>(std::max(hinted_batch, batch_size_)) *
            static_cast<size_t>(std::max(1, scratch_tokens));
        const size_t bytes =
            rows * static_cast<size_t>(kv_dim_) * sizeof(_Float16);

        WorkspaceRequirements requirements;
        requirements.buffers.emplace_back(
            KVCacheWorkspaceBuffers::CONV_SCRATCH_K, bytes, 256, true);
        requirements.buffers.emplace_back(
            KVCacheWorkspaceBuffers::CONV_SCRATCH_V, bytes, 256, true);
        return requirements;
    }

    void ROCmRingKVCacheTQ::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        if (workspace_ == workspace)
            return;
        if (isGraphCaptureActive())
        {
            throw std::runtime_error(
                "ROCmRingKVCacheTQ workspace ownership cannot change during HIP graph capture");
        }
        if (workspace && !workspace->isAllocated())
        {
            throw std::invalid_argument(
                "ROCmRingKVCacheTQ requires a fully allocated workspace");
        }

        void *scratch_k = nullptr;
        void *scratch_v = nullptr;
        size_t capacity = 0;
        if (workspace)
        {
            scratch_k =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            scratch_v =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            const size_t k_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            const size_t v_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            const size_t required =
                static_cast<size_t>(batch_size_) * max_seq_len_ * kv_dim_ *
                sizeof(_Float16);
            if (!scratch_k || !scratch_v ||
                k_capacity < required || v_capacity < required)
            {
                throw std::runtime_error(
                    "ROCmRingKVCacheTQ workspace lacks full-horizon FP16 K/V conversion storage");
            }
            capacity = std::min(k_capacity, v_capacity);
        }

        batched_k_view_.reset();
        batched_v_view_.reset();
        for (auto &scratch : layer_scratch_)
        {
            scratch.invalidate();
            scratch.d_K = static_cast<_Float16 *>(scratch_k);
            scratch.d_V = static_cast<_Float16 *>(scratch_v);
        }
        workspace_ = workspace;
        scratch_capacity_bytes_ = capacity;
    }

    bool ROCmRingKVCacheTQ::publishBatchedEntryTables(hipStream_t stream)
    {
        requireGPUExecutionStream(
            static_cast<void *>(stream),
            "ROCmRingKVCacheTQ::publishBatchedEntryTables");
        if (n_layers_ <= 0 || batch_size_ <= 0)
            return false;

        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        const size_t table_bytes = entry_count * sizeof(void *);
        std::vector<void *> host_k(entry_count);
        std::vector<void *> host_v(entry_count);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int request = 0; request < batch_size_; ++request)
            {
                const size_t index =
                    static_cast<size_t>(layer) * batch_size_ + request;
                host_k[index] = entries_[layer][request].d_K;
                host_v[index] = entries_[layer][request].d_V;
            }
        }

        auto *const backend = getROCmBackend();
        if (!backend)
            return false;

        d_batched_k_entry_table_ = static_cast<void **>(
            backend->allocate(table_bytes, device_id_));
        d_batched_v_entry_table_ = static_cast<void **>(
            backend->allocate(table_bytes, device_id_));
        if (!d_batched_k_entry_table_ ||
            !d_batched_v_entry_table_ ||
            !backend->hostToDevice(
                d_batched_k_entry_table_, host_k.data(),
                table_bytes, device_id_, stream) ||
            !backend->hostToDevice(
                d_batched_v_entry_table_, host_v.data(),
                table_bytes, device_id_, stream))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Resident entry-table publication failed");
            if (d_batched_k_entry_table_)
                backend->free(d_batched_k_entry_table_, device_id_);
            if (d_batched_v_entry_table_)
                backend->free(d_batched_v_entry_table_, device_id_);
            d_batched_k_entry_table_ = nullptr;
            d_batched_v_entry_table_ = nullptr;
            return false;
        }
        return true;
    }

    // =========================================================================
    // Query — get_cached_tokens(), get_head_position(), is_wrapped() are now
    // provided by ROCmRingKVCacheBase via entry accessor overrides.
    // =========================================================================

    // =========================================================================
    // Append
    // =========================================================================

    bool ROCmRingKVCacheTQ::append(int layer, int seq_idx,
                                    const ITensor *k, const ITensor *v,
                                    int num_tokens)
    {
        LOG_ERROR("[ROCmRingKVCacheTQ::append] Explicit stream required; use appendWithStream()");
        return false;
    }

    bool ROCmRingKVCacheTQ::appendWithStream(
        int layer, int seq_idx, const ITensor *k, const ITensor *v,
        int num_tokens, void *stream)
    {
        requireGPUExecutionStream(
            stream,
            "ROCmRingKVCacheTQ::appendWithStream");
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (!k || !v)
            return false;
        (void)hipSetDevice(device_id_);
        const auto hip_stream = static_cast<hipStream_t>(stream);
        cached_stream_ = hip_stream;
        auto &entry = entries_[layer][seq_idx];
        const int index = layer * batch_size_ + seq_idx;
        const int *d_append_count =
            deviceDynamicAppendCountPtr(layer, seq_idx);
        if (!d_head_params_ || !d_count_params_)
            return false;

        const bool prepared_tq =
            k->native_type() == TensorType::TQ8 &&
            v->native_type() ==
                (mode_ == TurboQuantKVMode::TQ8_K_TQ8_V
                     ? TensorType::TQ8
                     : TensorType::TQ4);
        if (prepared_tq)
        {
            if (!k->gpu_data_ptr() || !v->gpu_data_ptr() ||
                k->shape().empty() || v->shape().empty() ||
                k->shape()[0] < static_cast<size_t>(num_tokens) ||
                v->shape()[0] < static_cast<size_t>(num_tokens))
            {
                LOG_ERROR("[ROCmRingKVCacheTQ::appendWithStream] Prepared TQ rows must already be device resident");
                return false;
            }

            const size_t k_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
            const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * v_block_size_;
            const bool copy_ok = hip_tq_copy_prepared_rows_ring_dynamic(
                k->gpu_data_ptr(), v->gpu_data_ptr(), entry.d_K, entry.d_V,
                &d_head_params_[index], d_append_count,
                max_seq_len_, num_tokens,
                k_row_bytes, v_row_bytes, false, false,
                local_n_kv_heads_, hip_stream);
            if (copy_ok)
            {
                hip_kv_sequence_state_advance_dynamic(
                    &d_head_params_[index], &d_count_params_[index],
                    d_append_count, num_tokens, max_seq_len_, hip_stream);
            }
            if (!copy_ok)
                return false;
            return true;
        }

        if (k->native_type() != TensorType::FP32 || v->native_type() != TensorType::FP32 ||
            !k->gpu_data_ptr() || !v->gpu_data_ptr())
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::appendWithStream] TQ cache append requires device-resident FP32 K/V tensors");
            return false;
        }
        if (k->shape().size() < 2 || v->shape().size() < 2 ||
            k->shape()[0] < static_cast<size_t>(num_tokens) ||
            v->shape()[0] < static_cast<size_t>(num_tokens) ||
            k->shape()[1] != static_cast<size_t>(kv_dim_) ||
            v->shape()[1] != static_cast<size_t>(kv_dim_))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::appendWithStream] Position-major K/V shape does not match the local cache shard");
            return false;
        }

        const auto *d_k = static_cast<const float *>(k->gpu_data_ptr());
        const auto *d_v = static_cast<const float *>(v->gpu_data_ptr());
        const size_t rotation_offset = static_cast<size_t>(layer) *
                                       local_n_kv_heads_ * head_dim_ * head_dim_;
        const float *d_rotations = rotations_.d_rotations + rotation_offset;
        const bool ok = hip_tq_quantize_grouped_ring_dynamic(
            d_k, d_v, d_rotations, entry.d_K, entry.d_V,
            &d_head_params_[index], d_append_count,
            max_seq_len_, num_tokens,
            local_n_kv_heads_, head_dim_, false, false, mode_, hip_stream);
        if (ok)
        {
            hip_kv_sequence_state_advance_dynamic(
                &d_head_params_[index], &d_count_params_[index],
                d_append_count, num_tokens, max_seq_len_, hip_stream);
        }
        if (!ok)
            return false;

        return true;
    }

    bool ROCmRingKVCacheTQ::appendVerifierRowsDecodeEquivalent(
        int layer,
        int seq_idx,
        const ITensor *K,
        const ITensor *V,
        int verifier_rows,
        void *gpu_stream)
    {
        requireGPUExecutionStream(
            gpu_stream,
            "ROCmRingKVCacheTQ::appendVerifierRowsDecodeEquivalent");
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ ||
            verifier_rows < 1 || !K || !V)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Invalid grouped verifier append request");
            return false;
        }
        const bool fp32_sources =
            K->native_type() == TensorType::FP32 &&
            V->native_type() == TensorType::FP32;
        const bool prepared_tq_sources =
            K->native_type() == TensorType::TQ8 &&
            V->native_type() ==
                (mode_ == TurboQuantKVMode::TQ8_K_TQ8_V
                     ? TensorType::TQ8
                     : TensorType::TQ4);
        if (!fp32_sources && !prepared_tq_sources)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Grouped verifier publication requires device-resident FP32/FP32 or prepared TQ8/TQ4 K/V; got K="
                      << K->dtype_name() << " V=" << V->dtype_name());
            return false;
        }

        const auto source_layout = [&](const ITensor *tensor,
                                       const char *label,
                                       bool *head_major) -> bool
        {
            if (!tensor || !head_major || tensor->shape().size() < 2)
                return false;
            const size_t rows = tensor->shape()[0];
            const size_t cols = tensor->shape()[1];
            const bool position_major =
                rows >= static_cast<size_t>(verifier_rows) &&
                cols == static_cast<size_t>(kv_dim_);
            const bool verifier_head_major =
                rows == static_cast<size_t>(local_n_kv_heads_ * verifier_rows) &&
                cols == static_cast<size_t>(head_dim_);
            if (!position_major && !verifier_head_major)
            {
                LOG_ERROR("[ROCmRingKVCacheTQ] Unsupported grouped " << label
                          << " shape [" << rows << "," << cols << "] for M="
                          << verifier_rows << " local_heads=" << local_n_kv_heads_
                          << " head_dim=" << head_dim_);
                return false;
            }
            *head_major = verifier_head_major && !position_major;
            return true;
        };

        bool k_head_major = false;
        bool v_head_major = false;
        if (!source_layout(K, "K", &k_head_major) ||
            !source_layout(V, "V", &v_head_major))
        {
            return false;
        }

        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();
        if (!d_k || !d_v)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Grouped verifier K/V must already be device resident");
            return false;
        }

        (void)hipSetDevice(device_id_);
        const auto stream = static_cast<hipStream_t>(gpu_stream);
        cached_stream_ = stream;
        auto &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();
        const size_t rotation_offset = static_cast<size_t>(layer) *
                                       local_n_kv_heads_ * head_dim_ * head_dim_;
        const float *d_rotations = rotations_.d_rotations + rotation_offset;

        if (!d_head_params_ || !d_count_params_)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Grouped verifier requires canonical device sequence state");
            return false;
        }
        bool ok = false;
        const int index = layer * batch_size_ + seq_idx;
        const int *d_append_count =
            deviceDynamicAppendCountPtr(layer, seq_idx);
        if (prepared_tq_sources)
        {
            ok = hip_tq_copy_prepared_rows_ring_dynamic(
                d_k, d_v, entry.d_K, entry.d_V,
                &d_head_params_[index], d_append_count,
                max_seq_len_, verifier_rows,
                static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_,
                static_cast<size_t>(local_n_kv_heads_) * v_block_size_,
                k_head_major, v_head_major, local_n_kv_heads_, stream);
        }
        else
        {
            ok = hip_tq_quantize_grouped_ring_dynamic(
                static_cast<const float *>(d_k),
                static_cast<const float *>(d_v),
                d_rotations, entry.d_K, entry.d_V,
                &d_head_params_[index], d_append_count,
                max_seq_len_, verifier_rows,
                local_n_kv_heads_, head_dim_, k_head_major, v_head_major,
                mode_, stream);
        }
        if (ok)
        {
            hip_kv_sequence_state_advance_dynamic(
                &d_head_params_[index], &d_count_params_[index],
                d_append_count, verifier_rows, max_seq_len_, stream);
        }
        if (!ok)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Grouped verifier publication launch failed");
            return false;
        }

        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_kv_cache_grouped_verifier_append_calls",
            1.0,
            "verifier",
            "rocm",
            {{"cache_format", turboQuantKVModeName(mode_)},
             {"source_k_format", K->dtype_name()},
             {"source_v_format", V->dtype_name()},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"source_k_layout", k_head_major ? "head_major" : "position_major"},
             {"source_v_layout", v_head_major ? "head_major" : "position_major"},
             {"execution_mode", capture_active ? "graph_captured" : "eager_device_state"},
             {"row_count_policy", d_append_count ? "resident_device_count" : "captured_exact_shape"},
             {"topology", is_sharded() ? "local_tp_shard" : "replicated"},
             {"commit_policy", "single_grouped_metadata_commit"},
             {"local_kv_heads", std::to_string(local_n_kv_heads_)},
             {"head_dim", std::to_string(head_dim_)},
             {"kv_head_start", std::to_string(kv_head_start_)}});
        return true;
    }

    IKVCache::KVCacheLogicalBlockLayout
    ROCmRingKVCacheTQ::logicalBlockLayout(int global_layer, int token_count) const
    {
        KVCacheLogicalBlockLayout layout{
            .k_precision = ActivationPrecision::TQ8,
            .v_precision = ActivationPrecision::TQ4,
            .layout = TensorLayout::KV_POS_HEAD_DIM,
            .local_kv_heads = local_n_kv_heads_,
            .kv_head_start = kv_head_start_,
            .head_dim = head_dim_,
            .device_resident = true,
        };
        if (remapLayerIndex(global_layer) < 0 || token_count <= 0)
            return layout;
        layout.k_bytes = static_cast<size_t>(token_count) * local_n_kv_heads_ * tq8_block_size_;
        layout.v_bytes = static_cast<size_t>(token_count) * local_n_kv_heads_ * v_block_size_;
        return layout;
    }

    IKVCache::KVCacheSequenceState
    ROCmRingKVCacheTQ::sequenceState(int global_layer, int seq_idx) const
    {
        const int layer = remapLayerIndex(global_layer);
        if (layer < 0 || seq_idx < 0 || seq_idx >= batch_size_)
            return {};
        return ROCmRingKVCacheBase::sequenceState(layer, seq_idx);
    }

    bool ROCmRingKVCacheTQ::exportLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc,
        void *dst_k,
        void *dst_v) const
    {
        requireGPUExecutionStream(
            desc.stream,
            "ROCmRingKVCacheTQ::exportLogicalBlock");
        const int layer = remapLayerIndex(desc.layer);
        if (layer < 0 || desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0)
        {
            return false;
        }
        const auto &entry = entries_[layer][desc.seq_idx];
        if (desc.token_count == 0)
        {
            if (desc.payload_domain ==
                KVCacheLogicalBlockPayloadDomain::Device)
            {
                return true;
            }
            KVCacheSequenceState state;
            return observeDeviceSequenceState(layer, desc.seq_idx, &state) &&
                   desc.logical_token_start <= state.cached_tokens;
        }
        if (!dst_k || !dst_v || !entry.d_K || !entry.d_V)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::exportLogicalBlock] Non-empty export requires destinations and an explicit stream");
            return false;
        }

        (void)hipSetDevice(device_id_);
        const auto stream = static_cast<hipStream_t>(desc.stream);
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (!d_head_params_ || !d_count_params_)
                return false;

            const int entry_index =
                layer * batch_size_ + desc.seq_idx;
            constexpr int threads = 256;
            const size_t k_row_bytes =
                static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
            const size_t v_row_bytes =
                static_cast<size_t>(local_n_kv_heads_) * v_block_size_;
            const size_t bytes =
                static_cast<size_t>(desc.token_count) *
                (k_row_bytes + v_row_bytes);
            const int blocks = std::max(
                1,
                std::min<int>(
                    65535,
                    static_cast<int>((bytes + threads - 1) / threads)));
            hipLaunchKernelGGL(
                rocm_tq_ring_logical_block_export_device_kernel,
                dim3(blocks),
                dim3(threads),
                0,
                stream,
                static_cast<const uint8_t *>(entry.d_K),
                static_cast<const uint8_t *>(entry.d_V),
                static_cast<uint8_t *>(dst_k),
                static_cast<uint8_t *>(dst_v),
                &d_head_params_[entry_index],
                &d_count_params_[entry_index],
                desc.logical_token_start,
                desc.token_count,
                max_seq_len_,
                k_row_bytes,
                v_row_bytes);
            const hipError_t launch_error = hipGetLastError();
            if (launch_error != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCacheTQ::exportLogicalBlock] device-domain gather launch failed: "
                          << hipGetErrorString(launch_error));
                return false;
            }
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "rocm_device_logical_tq_kv_exports",
                1.0,
                "harvest",
                "rocm:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)}});
            return true;
        }

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, desc.seq_idx, &state))
            return false;
        const int export_head = state.implementation_head;
        const int export_count = state.cached_tokens;
        if (export_head < 0 || export_head >= max_seq_len_ || export_count < 0 ||
            export_count > max_seq_len_ || desc.logical_token_start > export_count ||
            desc.token_count > export_count - desc.logical_token_start)
        {
            return false;
        }

        int tail = (export_head - export_count) % max_seq_len_;
        if (tail < 0)
            tail += max_seq_len_;
        const size_t k_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
        const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * v_block_size_;
        auto *out_k = static_cast<uint8_t *>(dst_k);
        auto *out_v = static_cast<uint8_t *>(dst_v);
        const auto *ring_k = static_cast<const uint8_t *>(entry.d_K);
        const auto *ring_v = static_cast<const uint8_t *>(entry.d_V);
        for (int row = 0; row < desc.token_count; ++row)
        {
            const int logical_row = desc.logical_token_start + row;
            const int physical_row = (tail + logical_row) % max_seq_len_;
            if (hipMemcpyAsync(out_k + static_cast<size_t>(row) * k_row_bytes,
                               ring_k + static_cast<size_t>(physical_row) * k_row_bytes,
                               k_row_bytes, hipMemcpyDeviceToHost, stream) != hipSuccess ||
                hipMemcpyAsync(out_v + static_cast<size_t>(row) * v_row_bytes,
                               ring_v + static_cast<size_t>(physical_row) * v_row_bytes,
                               v_row_bytes, hipMemcpyDeviceToHost, stream) != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCacheTQ::exportLogicalBlock] TQ payload copy failed");
                return false;
            }
        }
        return hipStreamSynchronize(stream) == hipSuccess;
    }

    bool ROCmRingKVCacheTQ::importLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc,
        const void *src_k,
        const void *src_v)
    {
        requireGPUExecutionStream(
            desc.stream,
            "ROCmRingKVCacheTQ::importLogicalBlock");
        const int layer = remapLayerIndex(desc.layer);
        if (layer < 0 || desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start)
        {
            return false;
        }
        auto &entry = entries_[layer][desc.seq_idx];
        const auto stream = static_cast<hipStream_t>(desc.stream);
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (desc.token_count == 0)
            {
                if (desc.logical_token_start != 0)
                    return true;
                layer_scratch_[layer].invalidate();
                return setDeviceSequenceState(
                    layer, desc.seq_idx, 0, 0, stream);
            }
            if (!src_k || !src_v || !entry.d_K || !entry.d_V ||
                !d_head_params_ || !d_count_params_)
            {
                return false;
            }

            const int entry_index =
                layer * batch_size_ + desc.seq_idx;
            constexpr int threads = 256;
            const size_t k_row_bytes =
                static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
            const size_t v_row_bytes =
                static_cast<size_t>(local_n_kv_heads_) * v_block_size_;
            const size_t bytes =
                static_cast<size_t>(desc.token_count) *
                (k_row_bytes + v_row_bytes);
            const int blocks = std::max(
                1,
                std::min<int>(
                    65535,
                    static_cast<int>((bytes + threads - 1) / threads)));
            hipLaunchKernelGGL(
                rocm_tq_ring_logical_block_import_device_kernel,
                dim3(blocks),
                dim3(threads),
                0,
                stream,
                static_cast<uint8_t *>(entry.d_K),
                static_cast<uint8_t *>(entry.d_V),
                static_cast<const uint8_t *>(src_k),
                static_cast<const uint8_t *>(src_v),
                &d_head_params_[entry_index],
                &d_count_params_[entry_index],
                desc.logical_token_start,
                desc.token_count,
                max_seq_len_,
                k_row_bytes,
                v_row_bytes);
            hipLaunchKernelGGL(
                rocm_tq_ring_logical_block_import_publish_kernel,
                dim3(1),
                dim3(1),
                0,
                stream,
                &d_head_params_[entry_index],
                &d_count_params_[entry_index],
                desc.logical_token_start,
                desc.token_count,
                max_seq_len_);
            const hipError_t launch_error = hipGetLastError();
            if (launch_error != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCacheTQ::importLogicalBlock] device-domain scatter launch failed: "
                          << hipGetErrorString(launch_error));
                return false;
            }
            layer_scratch_[layer].invalidate();
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "rocm_device_logical_tq_kv_imports",
                1.0,
                "restore",
                "rocm:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)}});
            return true;
        }

        if (desc.token_count == 0)
        {
            if (desc.logical_token_start != 0)
                return true;
            layer_scratch_[layer].invalidate();
            return setDeviceSequenceState(
                layer, desc.seq_idx, 0, 0, stream);
        }
        if (!src_k || !src_v || !entry.d_K || !entry.d_V)
        {
            return false;
        }

        (void)hipSetDevice(device_id_);
        const size_t k_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
        const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * v_block_size_;
        const size_t k_bytes = static_cast<size_t>(desc.token_count) * k_row_bytes;
        const size_t v_bytes = static_cast<size_t>(desc.token_count) * v_row_bytes;
        auto *ring_k = static_cast<uint8_t *>(entry.d_K);
        auto *ring_v = static_cast<uint8_t *>(entry.d_V);
        if (hipMemcpyAsync(ring_k + static_cast<size_t>(desc.logical_token_start) * k_row_bytes,
                           src_k, k_bytes, hipMemcpyHostToDevice, stream) != hipSuccess ||
            hipMemcpyAsync(ring_v + static_cast<size_t>(desc.logical_token_start) * v_row_bytes,
                           src_v, v_bytes, hipMemcpyHostToDevice, stream) != hipSuccess)
        {
            return false;
        }
        const int new_count = desc.logical_token_start + desc.token_count;
        const int new_head = new_count % max_seq_len_;
        layer_scratch_[layer].invalidate();
        if (!setDeviceSequenceState(
                layer, desc.seq_idx, new_head, new_count, stream))
            return false;
        return true;
    }

    // =========================================================================
    // Per-Layer FP16 Scratch with Fused Incremental Dequant
    // =========================================================================

    hipStream_t ROCmRingKVCacheTQ::clearStream() const
    {
        if (cached_stream_)
            return cached_stream_;
        return static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
    }

    bool ROCmRingKVCacheTQ::dequant_to_scratch(
        int layer, int seq_idx, float rope_theta, int position_start,
        int rope_dim, hipStream_t stream, int *out_count) const
    {
        const size_t required =
            static_cast<size_t>(max_seq_len_) * kv_dim_ * sizeof(_Float16);
        if (!out_count || isGraphCaptureActive() || !workspace_ ||
            scratch_capacity_bytes_ < required)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::dequant_to_scratch] Scalar observation requires bound full-horizon workspace outside graph capture");
            return false;
        }

        auto &scratch = layer_scratch_[layer];
        const auto &entry = entries_[layer][seq_idx];
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return false;
        *out_count = state.cached_tokens;
        if (state.cached_tokens == 0)
            return true;

        const int tail =
            (state.implementation_head - state.cached_tokens + max_seq_len_) %
            max_seq_len_;
        const int D = head_dim_;
        const size_t layer_rot_offset = static_cast<size_t>(layer) * local_n_kv_heads_ * D * D;
        const float *d_K_rot_t = rotations_.d_rotations_t + layer_rot_offset;
        const float *d_V_rot_t = d_K_rot_t;
        const float *d_K_rot = rotations_.d_rotations + layer_rot_offset;
        const float *d_V_rot = d_K_rot;

        return hip_tq_ring_linearize_dequant_fp16(
            scratch.d_K, scratch.d_V,
            entry.d_K, entry.d_V,
            d_K_rot_t, d_V_rot_t,
            d_K_rot, d_V_rot,
            tail, state.cached_tokens, max_seq_len_,
            local_n_kv_heads_, head_dim_,
            rope_theta, position_start, rope_dim,
            mode_,
            stream);
    }

    // =========================================================================
    // get_k / get_v / get_kv
    // =========================================================================

    ITensor *ROCmRingKVCacheTQ::get_k(int layer, int seq_idx)
    {
        return const_cast<ITensor *>(std::as_const(*this).get_k(layer, seq_idx));
    }

    ITensor *ROCmRingKVCacheTQ::get_v(int layer, int seq_idx)
    {
        return const_cast<ITensor *>(std::as_const(*this).get_v(layer, seq_idx));
    }

    const ITensor *ROCmRingKVCacheTQ::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        (void)hipSetDevice(device_id_);
        int count = 0;
        if (!dequant_to_scratch(
                layer, seq_idx, 0.0f, 0, 0, clearStream(), &count) ||
            count == 0)
            return nullptr;
        auto &scratch = layer_scratch_[layer];

        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.k_view.get())->update_view(
                scratch.d_K, count);
        }
        return scratch.k_view.get();
    }

    const ITensor *ROCmRingKVCacheTQ::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        (void)hipSetDevice(device_id_);
        int count = 0;
        if (!dequant_to_scratch(
                layer, seq_idx, 0.0f, 0, 0, clearStream(), &count) ||
            count == 0)
            return nullptr;
        auto &scratch = layer_scratch_[layer];

        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.v_view.get())->update_view(
                scratch.d_V, count);
        }
        return scratch.v_view.get();
    }

    bool ROCmRingKVCacheTQ::get_kv(int layer, int seq_idx,
                                    ITensor **out_k, ITensor **out_v,
                                    int *out_kv_len)
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k) *out_k = k;
        if (out_v) *out_v = v;
        if (out_kv_len) *out_kv_len = static_cast<int>(k->rows());
        return true;
    }

    bool ROCmRingKVCacheTQ::get_kv(int layer, int seq_idx,
                                    const ITensor **out_k, const ITensor **out_v,
                                    int *out_kv_len) const
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k) *out_k = k;
        if (out_v) *out_v = v;
        if (out_kv_len) *out_kv_len = static_cast<int>(k->rows());
        return true;
    }

    bool ROCmRingKVCacheTQ::get_kv_batched_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ITensor **out_k,
        ITensor **out_v,
        void *gpu_stream)
    {
        requireGPUExecutionStream(
            gpu_stream,
            "ROCmRingKVCacheTQ::get_kv_batched_device_view");
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (layer < 0 || layer >= n_layers_ || first_seq_idx < 0 ||
            request_count <= 0 ||
            first_seq_idx > batch_size_ - request_count ||
            !d_head_params_ || !d_count_params_ ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_ ||
            !rotations_.d_rotations || !workspace_ ||
            scratch_capacity_bytes_ <
                static_cast<size_t>(request_count) * max_seq_len_ * kv_dim_ *
                    sizeof(_Float16))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::get_kv_batched_device_view] Invalid resident gather contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " stream=" << gpu_stream);
            return false;
        }

        (void)hipSetDevice(device_id_);
        auto &scratch = layer_scratch_[layer];
        const size_t rotation_offset =
            static_cast<size_t>(layer) * local_n_kv_heads_ * head_dim_ * head_dim_;
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const auto stream = static_cast<hipStream_t>(gpu_stream);
        if (!hip_tq_batched_ring_dequant_fp16_device_state(
                scratch.d_K,
                scratch.d_V,
                const_cast<const void *const *>(d_batched_k_entry_table_),
                const_cast<const void *const *>(d_batched_v_entry_table_),
                d_head_params_,
                d_count_params_,
                rotations_.d_rotations + rotation_offset,
                entry_offset,
                request_count,
                max_seq_len_,
                max_seq_len_,
                local_n_kv_heads_,
                head_dim_,
                /*rope_theta=*/0.0f,
                /*position_start=*/0,
                /*rope_dim=*/0,
                mode_,
                stream))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::get_kv_batched_device_view] Grouped TQ dequant launch failed");
            return false;
        }

        scratch.invalidate();
        cached_stream_ = stream;
        const size_t rows =
            static_cast<size_t>(request_count) * max_seq_len_;
        if (!batched_k_view_ ||
            batched_k_view_->gpu_data_ptr() != scratch.d_K ||
            batched_k_view_->shape().empty() ||
            batched_k_view_->shape()[0] != rows)
        {
            batched_k_view_ = std::make_unique<GpuTensorView>(
                scratch.d_K, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != scratch.d_V ||
            batched_v_view_->shape().empty() ||
            batched_v_view_->shape()[0] != rows)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                scratch.d_V, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    bool ROCmRingKVCacheTQ::get_kv_batched_converted_device_view(
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
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_ ||
            !rotations_.d_rotations || !workspace_ ||
            scratch_capacity_bytes_ <
                static_cast<size_t>(request_count) * max_seq_len_ * kv_dim_ *
                    sizeof(_Float16))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::get_kv_batched_converted_device_view] Invalid grouped conversion contract"
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

        (void)hipSetDevice(device_id_);
        auto &scratch = layer_scratch_[layer];
        const size_t rotation_offset =
            static_cast<size_t>(layer) * local_n_kv_heads_ * head_dim_ * head_dim_;
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const auto stream = static_cast<hipStream_t>(read.gpu_stream);
        if (!hip_tq_batched_ring_dequant_fp16_device_state(
                scratch.d_K,
                scratch.d_V,
                const_cast<const void *const *>(d_batched_k_entry_table_),
                const_cast<const void *const *>(d_batched_v_entry_table_),
                d_head_params_,
                d_count_params_,
                rotations_.d_rotations + rotation_offset,
                entry_offset,
                request_count,
                max_seq_len_,
                max_seq_len_,
                local_n_kv_heads_,
                head_dim_,
                read.rope_theta,
                read.position_start,
                effective_rope_dim,
                mode_,
                stream))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::get_kv_batched_converted_device_view] Grouped TQ conversion/RoPE launch failed");
            return false;
        }

        scratch.invalidate();
        cached_stream_ = stream;
        const size_t rows =
            static_cast<size_t>(request_count) * max_seq_len_;
        if (!batched_k_view_ ||
            batched_k_view_->gpu_data_ptr() != scratch.d_K ||
            batched_k_view_->shape().empty() ||
            batched_k_view_->shape()[0] != rows)
        {
            batched_k_view_ = std::make_unique<GpuTensorView>(
                scratch.d_K, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != scratch.d_V ||
            batched_v_view_->shape().empty() ||
            batched_v_view_->shape()[0] != rows)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                scratch.d_V, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    // =========================================================================
    // get_kv_converted (dequant + optional RoPE → FP16 scratch)
    // =========================================================================

    bool ROCmRingKVCacheTQ::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        (void)target;
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (out_kv_len)
            *out_kv_len = 0;

        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        (void)hipSetDevice(device_id_);

        const float rope_theta = (rope && rope->rope_theta > 0.0f) ? rope->rope_theta : 0.0f;
        const int position_start = (rope && rope_theta > 0.0f) ? rope->position_start : 0;
        const int rope_dim = (rope && rope_theta > 0.0f)
                                 ? (rope->rope_dim > 0 ? rope->rope_dim : head_dim_)
                                 : 0;
        hipStream_t stream = (rope && rope->gpu_stream)
                                 ? static_cast<hipStream_t>(rope->gpu_stream)
                                 : clearStream();
        int count = 0;
        if (!dequant_to_scratch(
                layer, seq_idx, rope_theta, position_start, rope_dim,
                stream, &count))
            return false;
        if (count == 0)
            return true;

        auto &scratch = layer_scratch_[layer];

        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.k_view.get())->update_view(
                scratch.d_K, count);
        }
        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, count, kv_dim_,
                TensorType::FP16, DeviceId::rocm(device_id_));
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.v_view.get())->update_view(
                scratch.d_V, count);
        }

        if (out_k) *out_k = scratch.k_view.get();
        if (out_v) *out_v = scratch.v_view.get();
        if (out_kv_len) *out_kv_len = count;

        return true;
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    void ROCmRingKVCacheTQ::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            num_tokens < 0)
            return;

        const auto stream = clearStream();
        if (!evictOldestDeviceSequenceState(
                layer,
                seq_idx,
                num_tokens,
                stream))
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::evict_oldest] Device metadata eviction launch failed");
            return;
        }

        layer_scratch_[layer].invalidate();
    }

    // =========================================================================
    // Factory function (callable from non-HIP code via Factory header)
    // =========================================================================

    std::unique_ptr<IKVCache> createROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx,
        int device_id,
        TurboQuantKVMode mode)
    {
        return std::make_unique<ROCmRingKVCacheTQ>(
            n_layers, batch_size, max_seq_len,
            n_kv_heads, head_dim, tq_ctx, device_id, mode);
    }

    std::unique_ptr<IKVCache> createShardedROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx,
        int device_id,
        TurboQuantKVMode mode)
    {
        return std::make_unique<ROCmRingKVCacheTQ>(
            n_layers, batch_size, max_seq_len,
            n_kv_heads, local_n_kv_heads, kv_head_start,
            head_dim, tq_ctx, device_id, mode);
    }

} // namespace llaminar2
