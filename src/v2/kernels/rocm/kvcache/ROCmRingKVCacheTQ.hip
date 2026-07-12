/**
 * @file ROCmRingKVCacheTQ.hip
 * @brief HIP implementation of ROCm TQ KV cache
 * @author David Sanftenberg
 *
 * V2: Per-layer FP16 scratch, fused incremental kernel, graph capture support.
 */

#include "ROCmRingKVCacheTQ.h"
#include "ROCmTurboQuantKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include <hip/hip_runtime.h>
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

    ROCmRingKVCacheTQ::ROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx, int device_id)
        : ROCmRingKVCacheTQ(n_layers, batch_size, max_seq_len,
                            n_kv_heads, n_kv_heads, 0,
                            head_dim, tq_ctx, device_id)
    {
    }

    ROCmRingKVCacheTQ::ROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx, int device_id)
        : ROCmRingKVCacheBase(n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, local_n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(local_n_kv_heads),
          kv_head_start_(kv_head_start)
    {
        if (local_n_kv_heads <= 0 || local_n_kv_heads > n_kv_heads ||
            kv_head_start < 0 || kv_head_start + local_n_kv_heads > n_kv_heads)
        {
            throw std::invalid_argument("ROCmRingKVCacheTQ: invalid LocalTP KV-head range");
        }
        (void)hipSetDevice(device_id_);

        if (head_dim_ == 64)
        {
            tq8_block_size_ = sizeof(TQ8Block<64>);
            tq4_block_size_ = sizeof(TQ4Block<64>);
        }
        else if (head_dim_ == 128)
        {
            tq8_block_size_ = sizeof(TQ8Block<128>);
            tq4_block_size_ = sizeof(TQ4Block<128>);
        }
        else
        {
            LOG_ERROR("ROCmRingKVCacheTQ: unsupported head_dim=" << head_dim_);
            return;
        }

        const uint64_t rotation_seed = tq_ctx ? tq_ctx->rotation().seed : 42;

        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        if (!init_stream)
            throw std::runtime_error("ROCmRingKVCacheTQ: explicit initialization stream unavailable");

        hip_tq_upload_codebooks(init_stream);
        rotations_ = hip_tq_create_rotations(n_layers_, local_n_kv_heads_, head_dim_,
                                              rotation_seed, device_id_, init_stream, kv_head_start_);

        const size_t k_entry_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * tq8_block_size_;
        const size_t v_entry_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * tq4_block_size_;

        entries_.resize(n_layers_);

        for (int l = 0; l < n_layers_; ++l)
        {
            entries_[l].resize(batch_size_);

            for (int s = 0; s < batch_size_; ++s)
            {
                (void)hipMalloc(&entries_[l][s].d_K, k_entry_bytes);
                (void)hipMalloc(&entries_[l][s].d_V, v_entry_bytes);
            }
        }
        if (!publishBatchedEntryTables(init_stream))
        {
            for (auto &layer : entries_)
                for (auto &entry : layer)
                {
                    if (entry.d_K)
                        (void)hipFree(entry.d_K);
                    if (entry.d_V)
                        (void)hipFree(entry.d_V);
                }
            hip_tq_free_rotations(rotations_);
            throw std::runtime_error(
                "ROCmRingKVCacheTQ: failed to publish resident batched entry topology");
        }

        // Grouped verifier graph replay and device-resident MTP publication
        // share these canonical base head/count scalars.
        allocateDeviceParams();

        // Allocate per-layer FP16 scratch buffers
        {
            const size_t scratch_bytes =
                static_cast<size_t>(batch_size_) * max_seq_len_ * kv_dim_ *
                sizeof(_Float16);
            layer_scratch_.resize(n_layers_);
            for (int l = 0; l < n_layers_; ++l)
            {
                (void)hipMalloc(&layer_scratch_[l].d_K, scratch_bytes);
                (void)hipMalloc(&layer_scratch_[l].d_V, scratch_bytes);
                layer_scratch_[l].invalidate();
            }
            LOG_DEBUG("ROCmRingKVCacheTQ: per-layer FP16 scratch: "
                     << n_layers_ << " x " << (2 * scratch_bytes / 1024) << "KB");
        }

        const hipError_t init_sync = hipStreamSynchronize(init_stream);
        if (init_sync != hipSuccess)
            throw std::runtime_error(std::string("ROCmRingKVCacheTQ initialization failed: ") +
                                     hipGetErrorString(init_sync));

        LOG_DEBUG("ROCmRingKVCacheTQ: created on device " << device_id_
                  << " (" << n_layers_ << " layers, " << batch_size_ << " batch, "
                  << max_seq_len_ << " maxseq, local/total_heads="
                  << local_n_kv_heads_ << "/" << n_kv_heads_ << ", head_dim=" << head_dim_
                  << ", TQ8+TQ4, FP16 scratch)");
    }

    ROCmRingKVCacheTQ::~ROCmRingKVCacheTQ()
    {
        (void)hipSetDevice(device_id_);

        for (auto &layer : entries_)
            for (auto &entry : layer)
            {
                if (entry.d_K) (void)hipFree(entry.d_K);
                if (entry.d_V) (void)hipFree(entry.d_V);
            }

        batched_k_view_.reset();
        batched_v_view_.reset();
        if (d_batched_k_entry_table_)
            (void)hipFree(d_batched_k_entry_table_);
        if (d_batched_v_entry_table_)
            (void)hipFree(d_batched_v_entry_table_);
        d_batched_k_entry_table_ = nullptr;
        d_batched_v_entry_table_ = nullptr;

        // Free per-layer scratch
        for (auto &scratch : layer_scratch_)
        {
            scratch.k_view.reset();
            scratch.v_view.reset();
            if (scratch.d_K) (void)hipFree(scratch.d_K);
            if (scratch.d_V) (void)hipFree(scratch.d_V);
        }

        hip_tq_free_rotations(rotations_);
    }

    bool ROCmRingKVCacheTQ::publishBatchedEntryTables(hipStream_t stream)
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

        hipError_t status = hipMalloc(
            reinterpret_cast<void **>(&d_batched_k_entry_table_), table_bytes);
        if (status == hipSuccess)
        {
            status = hipMalloc(
                reinterpret_cast<void **>(&d_batched_v_entry_table_),
                table_bytes);
        }
        if (status == hipSuccess)
        {
            status = hipMemcpyAsync(
                d_batched_k_entry_table_, host_k.data(), table_bytes,
                hipMemcpyHostToDevice, stream);
        }
        if (status == hipSuccess)
        {
            status = hipMemcpyAsync(
                d_batched_v_entry_table_, host_v.data(), table_bytes,
                hipMemcpyHostToDevice, stream);
        }
        if (status != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Resident entry-table publication failed: "
                      << hipGetErrorString(status));
            if (d_batched_k_entry_table_)
                (void)hipFree(d_batched_k_entry_table_);
            if (d_batched_v_entry_table_)
                (void)hipFree(d_batched_v_entry_table_);
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
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (!k || !v)
            return false;
        if (!stream)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::appendWithStream] Null stream is not allowed");
            return false;
        }

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
            v->native_type() == TensorType::TQ4;
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
            const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq4_block_size_;
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
            local_n_kv_heads_, head_dim_, false, false, hip_stream);
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
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ ||
            verifier_rows < 1 || !K || !V || !gpu_stream)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ] Invalid grouped verifier append request");
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
        if (prepared_tq_sources)
        {
            ok = hip_tq_copy_prepared_rows_ring_dynamic(
                d_k, d_v, entry.d_K, entry.d_V,
                &d_head_params_[index], /*d_row_count=*/nullptr,
                max_seq_len_, verifier_rows,
                static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_,
                static_cast<size_t>(local_n_kv_heads_) * tq4_block_size_,
                k_head_major, v_head_major, local_n_kv_heads_, stream);
        }
        else
        {
            ok = hip_tq_quantize_grouped_ring_dynamic(
                static_cast<const float *>(d_k),
                static_cast<const float *>(d_v),
                d_rotations, entry.d_K, entry.d_V,
                &d_head_params_[index], /*d_row_count=*/nullptr,
                max_seq_len_, verifier_rows,
                local_n_kv_heads_, head_dim_, k_head_major, v_head_major, stream);
        }
        if (ok)
        {
            hip_kv_sequence_state_advance(
                &d_head_params_[index], &d_count_params_[index],
                verifier_rows, max_seq_len_, stream);
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
        layout.v_bytes = static_cast<size_t>(token_count) * local_n_kv_heads_ * tq4_block_size_;
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
            LOG_ERROR("[ROCmRingKVCacheTQ::exportLogicalBlock] Non-empty export requires destinations and an explicit stream");
            return false;
        }

        (void)hipSetDevice(device_id_);
        const auto stream = static_cast<hipStream_t>(desc.stream);
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
        const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq4_block_size_;
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
        const int layer = remapLayerIndex(desc.layer);
        if (layer < 0 || desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start || !desc.stream)
        {
            return false;
        }
        auto &entry = entries_[layer][desc.seq_idx];
        const auto stream = static_cast<hipStream_t>(desc.stream);
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

        (void)hipSetDevice(device_id_);
        const size_t k_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq8_block_size_;
        const size_t v_row_bytes = static_cast<size_t>(local_n_kv_heads_) * tq4_block_size_;
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
        return hipStreamSynchronize(stream) == hipSuccess;
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

    void ROCmRingKVCacheTQ::clearEntryStorage(int layer, int seq_idx, hipStream_t stream)
    {
        auto &entry = entries_[layer][seq_idx];
        const size_t k_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * tq8_block_size_;
        const size_t v_bytes = static_cast<size_t>(max_seq_len_) * local_n_kv_heads_ * tq4_block_size_;

        if (entry.d_K)
        {
            hipError_t err = hipMemsetAsync(entry.d_K, 0, k_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] K memset failed: " << hipGetErrorString(err));
        }
        if (entry.d_V)
        {
            hipError_t err = hipMemsetAsync(entry.d_V, 0, v_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] V memset failed: " << hipGetErrorString(err));
        }
    }

    void ROCmRingKVCacheTQ::clearScratchStorage(int layer, hipStream_t stream)
    {
        auto &scratch = layer_scratch_[layer];
        const size_t scratch_bytes =
            static_cast<size_t>(batch_size_) * max_seq_len_ * kv_dim_ *
            sizeof(_Float16);

        if (scratch.d_K)
        {
            hipError_t err = hipMemsetAsync(scratch.d_K, 0, scratch_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] scratch K memset failed: " << hipGetErrorString(err));
        }
        if (scratch.d_V)
        {
            hipError_t err = hipMemsetAsync(scratch.d_V, 0, scratch_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] scratch V memset failed: " << hipGetErrorString(err));
        }

        scratch.k_view.reset();
        scratch.v_view.reset();
        scratch.invalidate();
    }

    void ROCmRingKVCacheTQ::clearDynamicParams(int layer, int seq_idx, hipStream_t stream)
    {
        const int entry_idx = layer * batch_size_ + seq_idx;
        append_count_sources_[static_cast<size_t>(entry_idx)] = nullptr;
        if (d_head_params_)
        {
            hipError_t err = hipMemsetAsync(&d_head_params_[entry_idx], 0, sizeof(int), stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] head param memset failed: " << hipGetErrorString(err));
        }
        if (d_count_params_)
        {
            hipError_t err = hipMemsetAsync(&d_count_params_[entry_idx], 0, sizeof(int), stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] count param memset failed: " << hipGetErrorString(err));
        }

    }

    void ROCmRingKVCacheTQ::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        (void)hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        clearEntryStorage(layer, seq_idx, stream);
        clearScratchStorage(layer, stream);
        clearDynamicParams(layer, seq_idx, stream);

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_sequence] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        ROCmRingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void ROCmRingKVCacheTQ::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
            return;

        (void)hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        for (int seq = 0; seq < batch_size_; ++seq)
        {
            clearEntryStorage(layer, seq, stream);
            clearDynamicParams(layer, seq, stream);
        }
        clearScratchStorage(layer, stream);

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_layer] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int seq = 0; seq < batch_size_; ++seq)
            ROCmRingKVCacheBase::clear_sequence(layer, seq);
    }

    void ROCmRingKVCacheTQ::clear_sequence(int seq_idx)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_)
            return;

        (void)hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            clearEntryStorage(layer, seq_idx, stream);
            clearScratchStorage(layer, stream);
            clearDynamicParams(layer, seq_idx, stream);
        }

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_sequence] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
            ROCmRingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void ROCmRingKVCacheTQ::clear()
    {
        (void)hipSetDevice(device_id_);
        hipStream_t stream = clearStream();

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                clearEntryStorage(layer, seq, stream);
                clearDynamicParams(layer, seq, stream);
            }
            clearScratchStorage(layer, stream);
        }

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
                ROCmRingKVCacheBase::clear_sequence(layer, seq);
        }
    }

    bool ROCmRingKVCacheTQ::dequant_to_scratch(
        int layer, int seq_idx, float rope_theta, int position_start,
        int rope_dim, hipStream_t stream, int *out_count) const
    {
        if (!out_count || isGraphCaptureActive())
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::dequant_to_scratch] Scalar observation is unavailable during graph capture");
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
                TensorType::FP16, device_id_);
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
                TensorType::FP16, device_id_);
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
            !rotations_.d_rotations)
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
                TensorType::FP16, device_id_);
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
                TensorType::FP16, device_id_);
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
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return;
        if (num_tokens >= state.cached_tokens)
        {
            clear_sequence(layer, seq_idx);
            return;
        }

        const auto stream = clearStream();
        if (!setDeviceSequenceState(
                layer, seq_idx, state.implementation_head,
                state.cached_tokens - num_tokens, stream) ||
            hipStreamSynchronize(stream) != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::evict_oldest] Device metadata update failed");
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
        int device_id)
    {
        return std::make_unique<ROCmRingKVCacheTQ>(
            n_layers, batch_size, max_seq_len,
            n_kv_heads, head_dim, tq_ctx, device_id);
    }

    std::unique_ptr<IKVCache> createShardedROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx,
        int device_id)
    {
        return std::make_unique<ROCmRingKVCacheTQ>(
            n_layers, batch_size, max_seq_len,
            n_kv_heads, local_n_kv_heads, kv_head_start,
            head_dim, tq_ctx, device_id);
    }

} // namespace llaminar2
