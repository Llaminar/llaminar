/**
 * @file CUDARingKVCacheTQ.cu
 * @brief Implementation of CUDARingKVCacheTQ - TurboQuant KV cache on CUDA
 * @author David Sanftenberg
 */

#include "CUDARingKVCacheTQ.h"
#include "CUDATurboQuantKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    extern "C" void cuda_kv_sequence_state_advance(
        int *d_head, int *d_count, int num_tokens, int max_seq_len,
        cudaStream_t stream);
    extern "C" void cuda_kv_sequence_state_advance_dynamic(
        int *d_head, int *d_count, const int *d_append_count,
        int captured_num_tokens, int max_seq_len, cudaStream_t stream);

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    CUDARingKVCacheTQ::CUDARingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx,
        int device_id)
        : CUDARingKVCacheTQ(n_layers, batch_size, max_seq_len,
                            n_kv_heads, n_kv_heads, 0,
                            head_dim, tq_ctx, device_id)
    {
    }

    CUDARingKVCacheTQ::CUDARingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx,
        int device_id)
        : CUDARingKVCacheBase(n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, local_n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(local_n_kv_heads),
          kv_head_start_(kv_head_start),
          tq_ctx_(tq_ctx)
    {
        if (local_n_kv_heads <= 0 || local_n_kv_heads > n_kv_heads ||
            kv_head_start < 0 || kv_head_start + local_n_kv_heads > n_kv_heads)
        {
            throw std::invalid_argument("CUDARingKVCacheTQ: invalid LocalTP KV-head range");
        }
        cudaSetDevice(device_id);

        // Determine block sizes
        if (head_dim == 64)
        {
            k_block_size_ = sizeof(TQ8Block<64>);
            v_block_size_ = sizeof(TQ4Block<64>);
        }
        else if (head_dim == 128)
        {
            k_block_size_ = sizeof(TQ8Block<128>);
            v_block_size_ = sizeof(TQ4Block<128>);
        }
        else
        {
            LOG_ERROR("CUDARingKVCacheTQ: unsupported head_dim=" << head_dim);
            throw std::runtime_error("CUDARingKVCacheTQ: head_dim must be 64 or 128");
        }

        k_pos_bytes_ = static_cast<size_t>(local_n_kv_heads_) * k_block_size_;
        v_pos_bytes_ = static_cast<size_t>(local_n_kv_heads_) * v_block_size_;

        LOG_DEBUG("CUDARingKVCacheTQ: K block=" << k_block_size_ << "B, V block=" << v_block_size_
                                                << "B, per-position: K=" << k_pos_bytes_ << "B V=" << v_pos_bytes_ << "B"
                                                << " (vs FP16: " << (local_n_kv_heads_ * head_dim * 2 * 2) << "B)");

        cudaStream_t init_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id).defaultStream());
        if (!init_stream)
            throw std::runtime_error("CUDARingKVCacheTQ: explicit initialization stream unavailable");

        // Upload persistent codebooks and shard-specific rotations on one
        // explicit stream.  The constructor fences that stream before return,
        // so the first graph capture cannot race partially initialized state.
        cuda_tq_upload_codebooks(init_stream);

        // Create GPU rotation matrices from TurboQuantContext
        if (tq_ctx)
        {
            rotations_ = cuda_tq_create_rotations(
                n_layers, local_n_kv_heads_, head_dim,
                tq_ctx->rotation().seed, device_id, init_stream, kv_head_start_);

            // Diagnostic: verify GPU rotation matches CPU rotation
            {
                const auto &layer0_ctx = tq_ctx->for_layer(0);
                const auto &head0_ctx = layer0_ctx.for_layer(kv_head_start_);
                const auto &cpu_rot = head0_ctx.rotation();
                float gpu_rot_val = 0.0f;
                cudaMemcpyAsync(&gpu_rot_val, rotations_.d_rotations,
                                sizeof(float), cudaMemcpyDeviceToHost, init_stream);
                cudaStreamSynchronize(init_stream);
                LOG_DEBUG("[CUDARingKVCacheTQ] Rotation check: CPU[0,0]=" << cpu_rot.matrix[0]
                                                                          << " GPU[0,0]=" << gpu_rot_val
                                                                          << " match=" << (std::abs(cpu_rot.matrix[0] - gpu_rot_val) < 1e-6f)
                                                                          << " seed=" << tq_ctx->rotation().seed);
            }
        }
        else
        {
            LOG_ERROR("CUDARingKVCacheTQ: null TurboQuantContext");
            throw std::runtime_error("CUDARingKVCacheTQ requires TurboQuantContext");
        }

        // Allocate per-layer TQ ring buffers
        entries_.resize(n_layers);
        for (int l = 0; l < n_layers; ++l)
        {
            entries_[l].resize(batch_size);
            for (int b = 0; b < batch_size; ++b)
            {
                allocate_entry(entries_[l][b]);
            }
        }
        if (!publishBatchedEntryTables(init_stream))
        {
            for (auto &layer : entries_)
                for (auto &entry : layer)
                    free_entry(entry);
            cuda_tq_free_rotations(rotations_);
            throw std::runtime_error(
                "CUDARingKVCacheTQ: failed to publish resident batched entry topology");
        }

        // Allocate per-layer FP16 scratch buffers (enables incremental dequant + FP16 flash attention)
        {
            // Scalar reads use the first request span. Grouped reads use every
            // request span in-place, so one allocation supports both paths.
            const size_t scratch_bytes =
                static_cast<size_t>(batch_size) * max_seq_len * kv_dim_ *
                sizeof(__half);
            layer_scratch_.resize(n_layers);
            for (int l = 0; l < n_layers; ++l)
            {
                cudaMalloc(&layer_scratch_[l].d_K, scratch_bytes);
                cudaMalloc(&layer_scratch_[l].d_V, scratch_bytes);
                layer_scratch_[l].invalidate();
            }

            const size_t total_tq_bytes = static_cast<size_t>(n_layers) * batch_size *
                                          max_seq_len * (k_pos_bytes_ + v_pos_bytes_);
            const size_t total_scratch_bytes = static_cast<size_t>(n_layers) * 2 * scratch_bytes;
            LOG_DEBUG("CUDARingKVCacheTQ VRAM: TQ caches="
                      << (total_tq_bytes / 1024) << "KB, per-layer FP16 scratch="
                      << (total_scratch_bytes / (1024 * 1024)) << "MB (" << n_layers << " layers)");
        }

        // Canonical device head/count and append-count mailboxes are allocated
        // by CUDARingKVCacheBase. Incremental dequant consumes those values
        // directly, so TQ owns no parallel host/device replay metadata.
        allocateDeviceParams();

        const cudaError_t init_sync = cudaStreamSynchronize(init_stream);
        if (init_sync != cudaSuccess)
            throw std::runtime_error(std::string("CUDARingKVCacheTQ initialization failed: ") +
                                     cudaGetErrorString(init_sync));

        LOG_DEBUG("CUDARingKVCacheTQ created: " << n_layers << " layers, "
                                                << max_seq_len << " max_seq_len, " << local_n_kv_heads_
                                                << "/" << n_kv_heads << " local/total KV heads, "
                                                << head_dim << " head_dim on cuda:" << device_id);
    }

    CUDARingKVCacheTQ::~CUDARingKVCacheTQ()
    {
        cudaSetDevice(device_id_);

        for (auto &layer : entries_)
        {
            for (auto &entry : layer)
                free_entry(entry);
        }

        batched_k_view_.reset();
        batched_v_view_.reset();
        if (d_batched_k_entry_table_)
            cudaFree(d_batched_k_entry_table_);
        if (d_batched_v_entry_table_)
            cudaFree(d_batched_v_entry_table_);
        d_batched_k_entry_table_ = nullptr;
        d_batched_v_entry_table_ = nullptr;

        // Free per-layer scratch buffers
        for (auto &scratch : layer_scratch_)
        {
            scratch.k_view.reset();
            scratch.v_view.reset();
            if (scratch.d_K)
                cudaFree(scratch.d_K);
            if (scratch.d_V)
                cudaFree(scratch.d_V);
        }
        layer_scratch_.clear();

        cuda_tq_free_rotations(rotations_);
    }

    bool CUDARingKVCacheTQ::publishBatchedEntryTables(cudaStream_t stream)
    {
        if (!stream || n_layers_ <= 0 || batch_size_ <= 0)
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

        cudaError_t status = cudaMalloc(
            reinterpret_cast<void **>(&d_batched_k_entry_table_), table_bytes);
        if (status == cudaSuccess)
        {
            status = cudaMalloc(
                reinterpret_cast<void **>(&d_batched_v_entry_table_),
                table_bytes);
        }
        if (status == cudaSuccess)
        {
            status = cudaMemcpyAsync(
                d_batched_k_entry_table_, host_k.data(), table_bytes,
                cudaMemcpyHostToDevice, stream);
        }
        if (status == cudaSuccess)
        {
            status = cudaMemcpyAsync(
                d_batched_v_entry_table_, host_v.data(), table_bytes,
                cudaMemcpyHostToDevice, stream);
        }
        if (status != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheTQ] Resident entry-table publication failed: "
                      << cudaGetErrorString(status));
            if (d_batched_k_entry_table_)
                cudaFree(d_batched_k_entry_table_);
            if (d_batched_v_entry_table_)
                cudaFree(d_batched_v_entry_table_);
            d_batched_k_entry_table_ = nullptr;
            d_batched_v_entry_table_ = nullptr;
            return false;
        }
        return true;
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    void CUDARingKVCacheTQ::allocate_entry(TQEntry &entry)
    {
        const size_t k_bytes = static_cast<size_t>(max_seq_len_) * k_pos_bytes_;
        const size_t v_bytes = static_cast<size_t>(max_seq_len_) * v_pos_bytes_;

        cudaMalloc(&entry.d_K, k_bytes);
        cudaMalloc(&entry.d_V, v_bytes);
        cudaStream_t alloc_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
        cudaMemsetAsync(entry.d_K, 0, k_bytes, alloc_stream);
        cudaMemsetAsync(entry.d_V, 0, v_bytes, alloc_stream);
        cudaStreamSynchronize(alloc_stream);

    }

    void CUDARingKVCacheTQ::free_entry(TQEntry &entry)
    {
        if (entry.d_K)
        {
            cudaFree(entry.d_K);
            entry.d_K = nullptr;
        }
        if (entry.d_V)
        {
            cudaFree(entry.d_V);
            entry.d_V = nullptr;
        }
    }

    // =========================================================================
    // IKVCache Basic Operations (now in CUDARingKVCacheBase)
    // get_cached_tokens, clear, clear_sequence, clear_layer,
    // get_head_position, is_wrapped are all inherited.
    // =========================================================================

    // =========================================================================
    // Append (FP32 → TQ8 K / TQ4 V on GPU)
    // =========================================================================

    bool CUDARingKVCacheTQ::append(int layer, int seq_idx,
                                   const ITensor *K, const ITensor *V,
                                   int num_tokens)
    {
        LOG_ERROR("[CUDARingKVCacheTQ::append] Explicit stream required; use appendWithStream()");
        return false;
    }

    bool CUDARingKVCacheTQ::appendWithStream(int layer, int seq_idx,
                                             const ITensor *K, const ITensor *V,
                                             int num_tokens, void *gpu_stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (num_tokens <= 0)
            return true;
        if (!gpu_stream)
        {
            LOG_ERROR("[CUDARingKVCacheTQ::appendWithStream] Null stream is not allowed");
            return false;
        }

        cudaSetDevice(device_id_);
        cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream);
        cached_stream_ = stream;

        TQEntry &entry = entries_[layer][seq_idx];
        const int index = layer * batch_size_ + seq_idx;
        const int *d_append_count =
            deviceDynamicAppendCountPtr(layer, seq_idx);
        if (!d_head_params_ || !d_count_params_)
            return false;

        const bool prepared_tq =
            K->native_type() == TensorType::TQ8 &&
            V->native_type() == TensorType::TQ4;
        if (prepared_tq)
        {
            if (!K->gpu_data_ptr() || !V->gpu_data_ptr() ||
                K->shape().empty() || V->shape().empty() ||
                K->shape()[0] < static_cast<size_t>(num_tokens) ||
                V->shape()[0] < static_cast<size_t>(num_tokens))
            {
                LOG_ERROR("[CUDARingKVCacheTQ::appendWithStream] Prepared TQ rows must already be device resident");
                return false;
            }

            const bool copy_ok = cuda_tq_copy_prepared_rows_ring_dynamic(
                K->gpu_data_ptr(), V->gpu_data_ptr(), entry.d_K, entry.d_V,
                &d_head_params_[index], d_append_count,
                max_seq_len_, num_tokens,
                k_pos_bytes_, v_pos_bytes_, false, false,
                local_n_kv_heads_, stream);
            if (copy_ok)
            {
                cuda_kv_sequence_state_advance_dynamic(
                    &d_head_params_[index], &d_count_params_[index],
                    d_append_count, num_tokens, max_seq_len_, stream);
            }
            if (!copy_ok)
                return false;
            return true;
        }

        if (K->native_type() != TensorType::FP32 || V->native_type() != TensorType::FP32 ||
            !K->gpu_data_ptr() || !V->gpu_data_ptr())
        {
            LOG_ERROR("[CUDARingKVCacheTQ::appendWithStream] TQ cache append requires device-resident FP32 K/V tensors");
            return false;
        }
        if (K->shape().size() < 2 || V->shape().size() < 2 ||
            K->shape()[0] < static_cast<size_t>(num_tokens) ||
            V->shape()[0] < static_cast<size_t>(num_tokens) ||
            K->shape()[1] != static_cast<size_t>(kv_dim_) ||
            V->shape()[1] != static_cast<size_t>(kv_dim_))
        {
            LOG_ERROR("[CUDARingKVCacheTQ::appendWithStream] Position-major K/V shape does not match the local cache shard");
            return false;
        }

        const auto *d_k = static_cast<const float *>(K->gpu_data_ptr());
        const auto *d_v = static_cast<const float *>(V->gpu_data_ptr());
        const size_t rotation_offset = static_cast<size_t>(layer) *
                                       local_n_kv_heads_ * head_dim_ * head_dim_;
        const float *d_rotations = rotations_.d_rotations + rotation_offset;
        const bool ok = cuda_tq_quantize_grouped_ring_dynamic(
            d_k, d_v, d_rotations, entry.d_K, entry.d_V,
            &d_head_params_[index], d_append_count,
            max_seq_len_, num_tokens,
            local_n_kv_heads_, head_dim_, false, false, stream);
        if (ok)
        {
            cuda_kv_sequence_state_advance_dynamic(
                &d_head_params_[index], &d_count_params_[index],
                d_append_count, num_tokens, max_seq_len_, stream);
        }
        if (!ok)
            return false;

        return true;
    }

    bool CUDARingKVCacheTQ::appendVerifierRowsDecodeEquivalent(
        int layer,
        int seq_idx,
        const ITensor *K,
        const ITensor *V,
        int verifier_rows,
        void *gpu_stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ ||
            verifier_rows < 1 || !K || !V || !gpu_stream)
        {
            LOG_ERROR("[CUDARingKVCacheTQ] Invalid grouped verifier append request");
            return false;
        }
        const bool fp32_sources =
            K->native_type() == TensorType::FP32 &&
            V->native_type() == TensorType::FP32;
        const bool prepared_tq_sources =
            K->native_type() == TensorType::TQ8 &&
            V->native_type() == TensorType::TQ4;
        if (!fp32_sources && !prepared_tq_sources)
        {
            LOG_ERROR("[CUDARingKVCacheTQ] Grouped verifier publication requires device-resident FP32/FP32 or prepared TQ8/TQ4 K/V; got K="
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
                LOG_ERROR("[CUDARingKVCacheTQ] Unsupported grouped " << label
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
            LOG_ERROR("[CUDARingKVCacheTQ] Grouped verifier K/V must already be device resident");
            return false;
        }

        cudaSetDevice(device_id_);
        const auto stream = static_cast<cudaStream_t>(gpu_stream);
        cached_stream_ = stream;
        auto &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();
        const size_t rotation_offset = static_cast<size_t>(layer) *
                                       local_n_kv_heads_ * head_dim_ * head_dim_;
        const float *d_rotations = rotations_.d_rotations + rotation_offset;

        if (!d_head_params_ || !d_count_params_)
        {
            LOG_ERROR("[CUDARingKVCacheTQ] Grouped verifier requires canonical device sequence state");
            return false;
        }
        bool ok = false;
        const int index = layer * batch_size_ + seq_idx;
        if (prepared_tq_sources)
        {
            ok = cuda_tq_copy_prepared_rows_ring_dynamic(
                d_k, d_v, entry.d_K, entry.d_V,
                &d_head_params_[index], /*d_row_count=*/nullptr,
                max_seq_len_, verifier_rows,
                k_pos_bytes_, v_pos_bytes_, k_head_major, v_head_major,
                local_n_kv_heads_, stream);
        }
        else
        {
            ok = cuda_tq_quantize_grouped_ring_dynamic(
                static_cast<const float *>(d_k),
                static_cast<const float *>(d_v),
                d_rotations, entry.d_K, entry.d_V,
                &d_head_params_[index], /*d_row_count=*/nullptr,
                max_seq_len_, verifier_rows,
                local_n_kv_heads_, head_dim_, k_head_major, v_head_major, stream);
        }
        if (ok)
        {
            cuda_kv_sequence_state_advance(
                &d_head_params_[index], &d_count_params_[index],
                verifier_rows, max_seq_len_, stream);
        }
        if (!ok)
        {
            LOG_ERROR("[CUDARingKVCacheTQ] Grouped verifier publication launch failed");
            return false;
        }

        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_kv_cache_grouped_verifier_append_calls",
            1.0,
            "verifier",
            "cuda",
            {{"cache_format", "TQ8/TQ4"},
             {"source_k_format", K->dtype_name()},
             {"source_v_format", V->dtype_name()},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"source_k_layout", k_head_major ? "head_major" : "position_major"},
             {"source_v_layout", v_head_major ? "head_major" : "position_major"},
             {"execution_mode", capture_active ? "graph_captured" : "eager_device_state"},
             {"topology", is_sharded() ? "local_tp_shard" : "replicated"},
             {"commit_policy", "single_grouped_metadata_commit"},
             {"local_kv_heads", std::to_string(local_n_kv_heads_)},
             {"head_dim", std::to_string(head_dim_)},
             {"kv_head_start", std::to_string(kv_head_start_)}});
        return true;
    }

    IKVCache::KVCacheLogicalBlockLayout
    CUDARingKVCacheTQ::logicalBlockLayout(int global_layer, int token_count) const
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
        layout.k_bytes = static_cast<size_t>(token_count) * k_pos_bytes_;
        layout.v_bytes = static_cast<size_t>(token_count) * v_pos_bytes_;
        return layout;
    }

    IKVCache::KVCacheSequenceState
    CUDARingKVCacheTQ::sequenceState(int global_layer, int seq_idx) const
    {
        const int layer = remapLayerIndex(global_layer);
        if (layer < 0 || seq_idx < 0 || seq_idx >= batch_size_)
            return {};
        return CUDARingKVCacheBase::sequenceState(layer, seq_idx);
    }

    bool CUDARingKVCacheTQ::exportLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc,
        void *dst_k,
        void *dst_v) const
    {
        const int layer = remapLayerIndex(desc.layer);
        if (layer < 0 || desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0)
        {
            return false;
        }
        const auto &entry = entries_[layer][desc.seq_idx];
        if (desc.token_count == 0)
        {
            KVCacheSequenceState state;
            return observeDeviceSequenceState(layer, desc.seq_idx, &state) &&
                   desc.logical_token_start <= state.cached_tokens;
        }
        if (!desc.stream || !dst_k || !dst_v || !entry.d_K || !entry.d_V)
        {
            LOG_ERROR("[CUDARingKVCacheTQ::exportLogicalBlock] Non-empty export requires destinations and an explicit stream");
            return false;
        }

        cudaSetDevice(device_id_);
        const auto stream = static_cast<cudaStream_t>(desc.stream);
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
        auto *out_k = static_cast<uint8_t *>(dst_k);
        auto *out_v = static_cast<uint8_t *>(dst_v);
        const auto *ring_k = static_cast<const uint8_t *>(entry.d_K);
        const auto *ring_v = static_cast<const uint8_t *>(entry.d_V);
        for (int row = 0; row < desc.token_count; ++row)
        {
            const int logical_row = desc.logical_token_start + row;
            const int physical_row = (tail + logical_row) % max_seq_len_;
            if (cudaMemcpyAsync(out_k + static_cast<size_t>(row) * k_pos_bytes_,
                                ring_k + static_cast<size_t>(physical_row) * k_pos_bytes_,
                                k_pos_bytes_, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaMemcpyAsync(out_v + static_cast<size_t>(row) * v_pos_bytes_,
                                ring_v + static_cast<size_t>(physical_row) * v_pos_bytes_,
                                v_pos_bytes_, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCacheTQ::exportLogicalBlock] TQ payload copy failed");
                return false;
            }
        }
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }

    bool CUDARingKVCacheTQ::importLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc,
        const void *src_k,
        const void *src_v)
    {
        const int layer = remapLayerIndex(desc.layer);
        if (layer < 0 || desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start || !desc.stream)
        {
            return false;
        }
        auto &entry = entries_[layer][desc.seq_idx];
        const auto stream = static_cast<cudaStream_t>(desc.stream);
        if (desc.token_count == 0)
        {
            if (desc.logical_token_start != 0)
                return true;
            layer_scratch_[layer].invalidate();
            return setDeviceSequenceState(
                layer, desc.seq_idx, 0, 0, stream);
        }
        KVCacheSequenceState state;
        if (!src_k || !src_v || !entry.d_K || !entry.d_V ||
            !observeDeviceSequenceState(layer, desc.seq_idx, &state) ||
            desc.logical_token_start != state.cached_tokens ||
            state.implementation_head != state.cached_tokens % max_seq_len_)
        {
            return false;
        }

        cudaSetDevice(device_id_);
        const size_t k_bytes = static_cast<size_t>(desc.token_count) * k_pos_bytes_;
        const size_t v_bytes = static_cast<size_t>(desc.token_count) * v_pos_bytes_;
        auto *ring_k = static_cast<uint8_t *>(entry.d_K);
        auto *ring_v = static_cast<uint8_t *>(entry.d_V);
        if (cudaMemcpyAsync(ring_k + static_cast<size_t>(desc.logical_token_start) * k_pos_bytes_,
                            src_k, k_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(ring_v + static_cast<size_t>(desc.logical_token_start) * v_pos_bytes_,
                            src_v, v_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return false;
        }
        const int new_count = desc.logical_token_start + desc.token_count;
        const int new_head = new_count % max_seq_len_;
        layer_scratch_[layer].invalidate();
        if (!setDeviceSequenceState(
                layer, desc.seq_idx, new_head, new_count, stream))
            return false;
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }

    // =========================================================================
    // Per-Layer Scratch Buffer Management
    // =========================================================================

    cudaStream_t CUDARingKVCacheTQ::clearStream() const
    {
        if (cached_stream_)
            return cached_stream_;
        return static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
    }

    void CUDARingKVCacheTQ::clearEntryStorage(int layer, int seq_idx, cudaStream_t stream)
    {
        auto &entry = entries_[layer][seq_idx];
        const size_t k_bytes = static_cast<size_t>(max_seq_len_) * k_pos_bytes_;
        const size_t v_bytes = static_cast<size_t>(max_seq_len_) * v_pos_bytes_;

        if (entry.d_K)
        {
            cudaError_t err = cudaMemsetAsync(entry.d_K, 0, k_bytes, stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] K memset failed: " << cudaGetErrorString(err));
        }
        if (entry.d_V)
        {
            cudaError_t err = cudaMemsetAsync(entry.d_V, 0, v_bytes, stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] V memset failed: " << cudaGetErrorString(err));
        }
    }

    void CUDARingKVCacheTQ::clearScratchStorage(int layer, cudaStream_t stream)
    {
        auto &scratch = layer_scratch_[layer];
        const size_t scratch_bytes =
            static_cast<size_t>(batch_size_) * max_seq_len_ * kv_dim_ *
            sizeof(__half);

        if (scratch.d_K)
        {
            cudaError_t err = cudaMemsetAsync(scratch.d_K, 0, scratch_bytes, stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] scratch K memset failed: " << cudaGetErrorString(err));
        }
        if (scratch.d_V)
        {
            cudaError_t err = cudaMemsetAsync(scratch.d_V, 0, scratch_bytes, stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] scratch V memset failed: " << cudaGetErrorString(err));
        }

        scratch.k_view.reset();
        scratch.v_view.reset();
        scratch.invalidate();
    }

    void CUDARingKVCacheTQ::clearDynamicParams(int layer, int seq_idx, cudaStream_t stream)
    {
        const int entry_idx = layer * batch_size_ + seq_idx;
        append_count_sources_[static_cast<size_t>(entry_idx)] = nullptr;
        if (d_head_params_)
        {
            cudaError_t err = cudaMemsetAsync(&d_head_params_[entry_idx], 0, sizeof(int), stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] head param memset failed: " << cudaGetErrorString(err));
        }
        if (d_count_params_)
        {
            cudaError_t err = cudaMemsetAsync(&d_count_params_[entry_idx], 0, sizeof(int), stream);
            if (err != cudaSuccess)
                LOG_WARN("[CUDARingKVCacheTQ::clear] count param memset failed: " << cudaGetErrorString(err));
        }

    }

    void CUDARingKVCacheTQ::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        cudaSetDevice(device_id_);
        cudaStream_t stream = clearStream();
        clearEntryStorage(layer, seq_idx, stream);
        clearScratchStorage(layer, stream);
        clearDynamicParams(layer, seq_idx, stream);

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
            LOG_WARN("[CUDARingKVCacheTQ::clear_sequence] sync failed: " << cudaGetErrorString(sync_err));

        cached_stream_ = nullptr;
        CUDARingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void CUDARingKVCacheTQ::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
            return;

        cudaSetDevice(device_id_);
        cudaStream_t stream = clearStream();
        for (int seq = 0; seq < batch_size_; ++seq)
        {
            clearEntryStorage(layer, seq, stream);
            clearDynamicParams(layer, seq, stream);
        }
        clearScratchStorage(layer, stream);

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
            LOG_WARN("[CUDARingKVCacheTQ::clear_layer] sync failed: " << cudaGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int seq = 0; seq < batch_size_; ++seq)
            CUDARingKVCacheBase::clear_sequence(layer, seq);
    }

    void CUDARingKVCacheTQ::clear_sequence(int seq_idx)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_)
            return;

        cudaSetDevice(device_id_);
        cudaStream_t stream = clearStream();
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            clearEntryStorage(layer, seq_idx, stream);
            clearScratchStorage(layer, stream);
            clearDynamicParams(layer, seq_idx, stream);
        }

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
            LOG_WARN("[CUDARingKVCacheTQ::clear_sequence] sync failed: " << cudaGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
            CUDARingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void CUDARingKVCacheTQ::clear()
    {
        cudaSetDevice(device_id_);
        cudaStream_t stream = clearStream();

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                clearEntryStorage(layer, seq, stream);
                clearDynamicParams(layer, seq, stream);
            }
            clearScratchStorage(layer, stream);
        }

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
            LOG_WARN("[CUDARingKVCacheTQ::clear] sync failed: " << cudaGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
                CUDARingKVCacheBase::clear_sequence(layer, seq);
        }
    }

    bool CUDARingKVCacheTQ::dequant_to_scratch(
        int layer, int seq_idx, float rope_theta, int position_start,
        int rope_dim, cudaStream_t stream, int *out_count) const
    {
        if (!out_count || isGraphCaptureActive())
        {
            LOG_ERROR("[CUDARingKVCacheTQ::dequant_to_scratch] Scalar observation is unavailable during graph capture");
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

        return cuda_tq_ring_linearize_dequant_fp16(
            scratch.d_K, scratch.d_V,
            entry.d_K, entry.d_V,
            d_K_rot_t, d_V_rot_t,
            d_K_rot, d_V_rot,
            tail, state.cached_tokens, max_seq_len_,
            local_n_kv_heads_, head_dim_,
            rope_theta, position_start, rope_dim,
            stream);
    }

    // =========================================================================
    // get_k / get_v (dequant to shared FP32 scratch, return view)
    // =========================================================================
    //
    // NOTE: The returned ITensor* points into the per-layer scratch buffer.
    // Each layer has its own scratch, so there are no cross-layer conflicts.
    //

    ITensor *CUDARingKVCacheTQ::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        cudaSetDevice(device_id_);

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
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.k_view.get())->update_view(scratch.d_K, count);
        }

        return scratch.k_view.get();
    }

    const ITensor *CUDARingKVCacheTQ::get_k(int layer, int seq_idx) const
    {
        return const_cast<CUDARingKVCacheTQ *>(this)->get_k(layer, seq_idx);
    }

    ITensor *CUDARingKVCacheTQ::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        cudaSetDevice(device_id_);

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
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.v_view.get())->update_view(scratch.d_V, count);
        }

        return scratch.v_view.get();
    }

    const ITensor *CUDARingKVCacheTQ::get_v(int layer, int seq_idx) const
    {
        return const_cast<CUDARingKVCacheTQ *>(this)->get_v(layer, seq_idx);
    }

    // =========================================================================
    // get_kv (unified K+V access)
    // =========================================================================

    bool CUDARingKVCacheTQ::get_kv(int layer, int seq_idx,
                                   ITensor **out_k, ITensor **out_v,
                                   int *out_kv_len)
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k)
            *out_k = k;
        if (out_v)
            *out_v = v;
        if (out_kv_len)
            *out_kv_len = static_cast<int>(k->rows());
        return true;
    }

    bool CUDARingKVCacheTQ::get_kv(int layer, int seq_idx,
                                   const ITensor **out_k, const ITensor **out_v,
                                   int *out_kv_len) const
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k)
            *out_k = k;
        if (out_v)
            *out_v = v;
        if (out_kv_len)
            *out_kv_len = static_cast<int>(k->rows());
        return true;
    }

    bool CUDARingKVCacheTQ::get_kv_batched_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ITensor **out_k,
        ITensor **out_v,
        void *gpu_stream)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;
        if (layer < 0 || layer >= n_layers_ || first_seq_idx < 0 ||
            request_count <= 0 ||
            first_seq_idx > batch_size_ - request_count ||
            !gpu_stream ||
            !d_head_params_ || !d_count_params_ ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_ ||
            !rotations_.d_rotations)
        {
            LOG_ERROR("[CUDARingKVCacheTQ::get_kv_batched_device_view] Invalid resident gather contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " stream=" << gpu_stream);
            return false;
        }

        cudaSetDevice(device_id_);
        auto &scratch = layer_scratch_[layer];
        const size_t rotation_offset =
            static_cast<size_t>(layer) * local_n_kv_heads_ * head_dim_ * head_dim_;
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const auto stream = static_cast<cudaStream_t>(gpu_stream);
        if (!cuda_tq_batched_ring_dequant_fp16_device_state(
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
                stream))
        {
            LOG_ERROR("[CUDARingKVCacheTQ::get_kv_batched_device_view] Grouped TQ dequant launch failed");
            return false;
        }

        // Grouped materialization overwrites the shared layer scratch, so a
        // later scalar read must rebuild its own sequence-local contents.
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
                TensorType::FP16, device_id_);
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != scratch.d_V ||
            batched_v_view_->shape().empty() ||
            batched_v_view_->shape()[0] != rows)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                scratch.d_V, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, device_id_);
        }
        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    bool CUDARingKVCacheTQ::get_kv_batched_converted_device_view(
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
            !rotations_.d_rotations)
        {
            LOG_ERROR("[CUDARingKVCacheTQ::get_kv_batched_converted_device_view] Invalid grouped conversion contract"
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

        cudaSetDevice(device_id_);
        auto &scratch = layer_scratch_[layer];
        const size_t rotation_offset =
            static_cast<size_t>(layer) * local_n_kv_heads_ * head_dim_ * head_dim_;
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const auto stream = static_cast<cudaStream_t>(read.gpu_stream);
        if (!cuda_tq_batched_ring_dequant_fp16_device_state(
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
                stream))
        {
            LOG_ERROR("[CUDARingKVCacheTQ::get_kv_batched_converted_device_view] Grouped TQ conversion/RoPE launch failed");
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
                TensorType::FP16, device_id_);
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != scratch.d_V ||
            batched_v_view_->shape().empty() ||
            batched_v_view_->shape()[0] != rows)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                scratch.d_V, rows, static_cast<size_t>(kv_dim_),
                TensorType::FP16, device_id_);
        }
        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    // =========================================================================
    // get_kv_converted (dequant + optional RoPE → FP32 scratch)
    // =========================================================================

    bool CUDARingKVCacheTQ::get_kv_converted(
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

        cudaSetDevice(device_id_);

        // Determine RoPE parameters
        float rope_theta = 0.0f;
        int position_start = 0;
        int rope_dim = 0;
        if (rope && rope->rope_theta > 0.0f)
        {
            rope_theta = rope->rope_theta;
            position_start = rope->position_start;
            rope_dim = rope->rope_dim > 0 ? rope->rope_dim : head_dim_;
        }

        // Dequant to per-layer FP32 scratch with optional RoPE on an explicit stream.
        cudaStream_t stream = (rope && rope->gpu_stream)
                                  ? static_cast<cudaStream_t>(rope->gpu_stream)
                                  : clearStream();
        int count = 0;
        if (!dequant_to_scratch(
                layer, seq_idx, rope_theta, position_start, rope_dim,
                stream, &count))
            return false;
        if (count == 0)
            return true;

        auto &scratch = layer_scratch_[layer];

        // Create/update FP16 scratch views (enables FP16 flash attention path)
        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.k_view.get())->update_view(scratch.d_K, count);
        }
        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.v_view.get())->update_view(scratch.d_V, count);
        }

        if (out_k)
            *out_k = scratch.k_view.get();
        if (out_v)
            *out_v = scratch.v_view.get();
        if (out_kv_len)
            *out_kv_len = count;

        return true;
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    void CUDARingKVCacheTQ::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return;
        const int evict = std::min(num_tokens, state.cached_tokens);
        const auto stream = clearStream();
        if (!setDeviceSequenceState(
                layer, seq_idx, state.implementation_head,
                state.cached_tokens - evict, stream) ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheTQ::evict_oldest] Device metadata update failed");
            return;
        }

        // Invalidate per-layer scratch (eviction changes the tail position)
        layer_scratch_[layer].invalidate();
    }

} // namespace llaminar2
