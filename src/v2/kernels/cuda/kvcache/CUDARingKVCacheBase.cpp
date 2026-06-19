/**
 * @file CUDARingKVCacheBase.cpp
 * @brief Implementation of CUDARingKVCacheBase common ring buffer operations
 *
 * Uses CUDA runtime API for device param allocation (cudaMalloc, cudaFree,
 * cudaMallocHost, cudaFreeHost). No custom kernels — just memory management
 * and host-side bookkeeping.
 */

#include "CUDARingKVCacheBase.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace llaminar2
{
    extern "C" bool cuda_kv_sequence_state_publish(
        int *d_heads,
        int *d_counts,
        const int32_t *target_cached_tokens,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int n_layers,
        int batch_size,
        int first_seq_idx,
        int request_count,
        int max_seq_len,
        cudaStream_t stream);

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    CUDARingKVCacheBase::CUDARingKVCacheBase(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int kv_dim, int device_id)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), head_dim_(head_dim), kv_dim_(kv_dim),
          device_id_(device_id)
    {
    }

    CUDARingKVCacheBase::~CUDARingKVCacheBase()
    {
        // Derived class destructors have already run and called cudaSetDevice().
        // cudaFree/cudaFreeHost work on the allocating device's memory.
        freeDeviceParams();
    }

    // =========================================================================
    // Graph Capture Device Params Management
    // =========================================================================

    void CUDARingKVCacheBase::allocateDeviceParams()
    {
        int num_entries = n_layers_ * batch_size_;
        cudaError_t err;

        err = cudaMalloc(&d_head_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate device head params: "
                     << cudaGetErrorString(err) << " - graph capture disabled");
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            return;
        }

        err = cudaMalloc(&d_count_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate device count params: "
                     << cudaGetErrorString(err) << " - device-resident KV sequence publication disabled");
            cudaFree(d_head_params_);
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            d_count_params_ = nullptr;
            h_count_params_ = nullptr;
            return;
        }

        err = cudaMallocHost(&h_head_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate pinned head params: "
                     << cudaGetErrorString(err) << " - graph capture disabled");
            cudaFree(d_head_params_);
            cudaFree(d_count_params_);
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            d_count_params_ = nullptr;
            h_count_params_ = nullptr;
            return;
        }

        err = cudaMallocHost(&h_count_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate pinned count params: "
                     << cudaGetErrorString(err) << " - device-resident KV sequence publication disabled");
            cudaFreeHost(h_head_params_);
            cudaFree(d_head_params_);
            cudaFree(d_count_params_);
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            d_count_params_ = nullptr;
            h_count_params_ = nullptr;
            return;
        }

        cudaStream_t init_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
        cudaMemsetAsync(d_head_params_, 0, num_entries * sizeof(int), init_stream);
        cudaMemsetAsync(d_count_params_, 0, num_entries * sizeof(int), init_stream);
        cudaStreamSynchronize(init_stream);
        std::memset(h_head_params_, 0, num_entries * sizeof(int));
        std::memset(h_count_params_, 0, num_entries * sizeof(int));

        LOG_DEBUG("[CUDARingKVCacheBase] Allocated device params for graph capture: "
                  << num_entries << " entries (" << num_entries * sizeof(int) * 2 << " bytes)");
    }

    void CUDARingKVCacheBase::freeDeviceParams()
    {
        if (d_head_params_)
        {
            cudaError_t err = cudaFree(d_head_params_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_head_params_) failed: %s\n", cudaGetErrorString(err));
            }
            d_head_params_ = nullptr;
        }
        if (h_head_params_)
        {
            cudaError_t err = cudaFreeHost(h_head_params_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFreeHost(h_head_params_) failed: %s\n", cudaGetErrorString(err));
            }
            h_head_params_ = nullptr;
        }
        if (d_count_params_)
        {
            cudaError_t err = cudaFree(d_count_params_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_count_params_) failed: %s\n", cudaGetErrorString(err));
            }
            d_count_params_ = nullptr;
        }
        if (h_count_params_)
        {
            cudaError_t err = cudaFreeHost(h_count_params_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFreeHost(h_count_params_) failed: %s\n", cudaGetErrorString(err));
            }
            h_count_params_ = nullptr;
        }
    }

    // =========================================================================
    // IKVCache Basic Operations
    // =========================================================================

    int CUDARingKVCacheBase::get_cached_tokens(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx))
            return 0;
        return entryCount(layer, seq_idx);
    }

    int CUDARingKVCacheBase::get_head_position(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx))
            return 0;
        return entryHead(layer, seq_idx);
    }

    bool CUDARingKVCacheBase::is_wrapped(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx))
            return false;
        int count = entryCount(layer, seq_idx);
        if (count == 0)
            return false;
        int head = entryHead(layer, seq_idx);
        int tail = (head - count + max_seq_len_) % max_seq_len_;
        return tail >= head && count > 0;
    }

    void CUDARingKVCacheBase::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
            clear_layer(layer);
        wrap_warned_ = false;
    }

    void CUDARingKVCacheBase::clear_sequence(int layer, int seq_idx)
    {
        if (!validLayerSeq(layer, seq_idx))
            return;
        resetEntry(layer, seq_idx);
        refreshHostDeviceParamMirror(layer, seq_idx);
        onClearSequence(layer, seq_idx);
    }

    void CUDARingKVCacheBase::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
            return;
        for (int seq = 0; seq < batch_size_; ++seq)
            clear_sequence(layer, seq);
    }

    // =========================================================================
    // Graph Capture Support
    // =========================================================================

    void CUDARingKVCacheBase::setDynamicHead(int layer, int seq_idx, void *gpu_stream)
    {
        if (!d_head_params_ || !h_head_params_)
            return;
        if (!validLayerSeq(layer, seq_idx))
            return;

        int idx = layer * batch_size_ + seq_idx;
        refreshHostDeviceParamMirror(layer, seq_idx);
        if (!gpu_stream)
            return;
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDARingKVCacheBase] Refusing to upload dynamic KV head inside CUDA graph capture");
            return;
        }

        cudaError_t err = cudaMemcpyAsync(
            &d_head_params_[idx],
            &h_head_params_[idx],
            sizeof(int),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(gpu_stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheBase] Failed to upload dynamic KV head: "
                      << cudaGetErrorString(err));
        }
        if (d_count_params_ && h_count_params_)
        {
            err = cudaMemcpyAsync(
                &d_count_params_[idx],
                &h_count_params_[idx],
                sizeof(int),
                cudaMemcpyHostToDevice,
                static_cast<cudaStream_t>(gpu_stream));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCacheBase] Failed to upload dynamic KV count: "
                          << cudaGetErrorString(err));
            }
        }
    }

    void CUDARingKVCacheBase::advanceHead(int layer, int seq_idx, int num_tokens)
    {
        if (!validLayerSeq(layer, seq_idx))
            return;

        int count = entryCount(layer, seq_idx);

        // Handle auto-eviction (same logic as append)
        if (count + num_tokens > max_seq_len_)
        {
            int to_evict = count + num_tokens - max_seq_len_;
            setEntryCount(layer, seq_idx, count - to_evict);
            onEviction(layer, seq_idx, to_evict);
            count -= to_evict;
        }

        int head = entryHead(layer, seq_idx);
        setEntryHead(layer, seq_idx, (head + num_tokens) % max_seq_len_);
        setEntryCount(layer, seq_idx, count + num_tokens);
        refreshHostDeviceParamMirror(layer, seq_idx);

        onAdvanceComplete(layer, seq_idx);
    }

    const int *CUDARingKVCacheBase::deviceCachedTokenCountPtr(int layer, int seq_idx) const
    {
        if (!d_count_params_ || !validLayerSeq(layer, seq_idx))
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return &d_count_params_[idx];
    }

    const int *CUDARingKVCacheBase::deviceRingHeadPtr(int layer, int seq_idx) const
    {
        if (!d_head_params_ || !validLayerSeq(layer, seq_idx))
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return &d_head_params_[idx];
    }

    void CUDARingKVCacheBase::refreshHostDeviceParamMirror(int layer, int seq_idx)
    {
        if (!validLayerSeq(layer, seq_idx))
            return;
        const int idx = layer * batch_size_ + seq_idx;
        if (h_head_params_)
            h_head_params_[idx] = entryHead(layer, seq_idx);
        if (h_count_params_)
            h_count_params_[idx] = entryCount(layer, seq_idx);
    }

    bool CUDARingKVCacheBase::uploadHostDeviceParamMirror(int layer, int seq_idx, void *gpu_stream)
    {
        if (!validLayerSeq(layer, seq_idx))
            return false;
        if (!gpu_stream || !d_head_params_ || !h_head_params_ ||
            !d_count_params_ || !h_count_params_)
            return false;
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDARingKVCacheBase] Refusing to upload KV sequence-state mirror inside CUDA graph capture");
            return false;
        }

        refreshHostDeviceParamMirror(layer, seq_idx);
        const int idx = layer * batch_size_ + seq_idx;
        cudaError_t err = cudaMemcpyAsync(
            &d_head_params_[idx],
            &h_head_params_[idx],
            sizeof(int),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(gpu_stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheBase] Failed to upload KV head mirror: "
                      << cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpyAsync(
            &d_count_params_[idx],
            &h_count_params_[idx],
            sizeof(int),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(gpu_stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheBase] Failed to upload KV count mirror: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDARingKVCacheBase::publishSequenceStateFromDeviceMetadata(
        const DeviceSequenceStatePublicationRequest &request,
        std::string *error)
    {
        if (!request.valid())
        {
            if (error)
            {
                *error =
                    "invalid CUDA KV device sequence-state publication request";
            }
            return false;
        }
        if (!d_head_params_ || !d_count_params_)
        {
            if (error)
            {
                *error =
                    "CUDA KV device sequence-state publication requires device head/count mirrors";
            }
            return false;
        }
        if (request.first_seq_idx + request.request_count > batch_size_)
        {
            if (error)
            {
                *error =
                    "CUDA KV device sequence-state publication request exceeds batch size";
            }
            return false;
        }

        const bool enqueued = cuda_kv_sequence_state_publish(
            d_head_params_,
            d_count_params_,
            request.target_cached_tokens_device,
            request.accepted_state_counts_device,
            request.publication_ok_flags_device,
            n_layers_,
            batch_size_,
            request.first_seq_idx,
            request.request_count,
            max_seq_len_,
            static_cast<cudaStream_t>(request.stream));
        if (!enqueued && error)
        {
            *error = "failed to enqueue CUDA KV device sequence-state publication";
        }
        return enqueued;
    }

    bool CUDARingKVCacheBase::adoptSequenceStateFromHostMetadata(
        const HostSequenceStatePublicationRequest &request,
        std::string *error)
    {
        if (!request.valid())
        {
            if (error)
            {
                *error =
                    "invalid CUDA KV host sequence-state adoption request";
            }
            return false;
        }
        if (request.first_seq_idx + request.request_count > batch_size_)
        {
            if (error)
            {
                *error =
                    "CUDA KV host sequence-state adoption request exceeds batch size";
            }
            return false;
        }

        for (int request_idx = 0; request_idx < request.request_count; ++request_idx)
        {
            if (request.publication_ok_flags[static_cast<size_t>(request_idx)] == 0)
                continue;

            const int seq_idx = request.first_seq_idx + request_idx;
            const int accepted_count =
                request.accepted_state_counts[static_cast<size_t>(request_idx)];
            const int target_count =
                request.target_cached_tokens[static_cast<size_t>(request_idx)];
            if (accepted_count < 0 ||
                target_count < 0 ||
                target_count > max_seq_len_)
            {
                if (error)
                {
                    *error =
                        "CUDA KV host sequence-state adoption received invalid counts";
                }
                return false;
            }

            for (int layer = 0; layer < n_layers_; ++layer)
            {
                const int old_count = entryCount(layer, seq_idx);
                const int evicted = std::max(0, old_count - target_count);
                if (evicted > 0)
                    onEviction(layer, seq_idx, evicted);

                const int old_head = entryHead(layer, seq_idx);
                int tail = old_head - old_count;
                tail %= max_seq_len_;
                if (tail < 0)
                    tail += max_seq_len_;
                setEntryHead(
                    layer,
                    seq_idx,
                    (tail + target_count) % max_seq_len_);
                setEntryCount(layer, seq_idx, target_count);
                refreshHostDeviceParamMirror(layer, seq_idx);
                onAdvanceComplete(layer, seq_idx);
            }
        }

        return true;
    }

} // namespace llaminar2
