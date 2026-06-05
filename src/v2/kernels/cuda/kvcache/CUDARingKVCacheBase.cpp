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
#include <cstring>
#include <cstdio>

namespace llaminar2
{

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

        err = cudaMallocHost(&h_head_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate pinned head params: "
                     << cudaGetErrorString(err) << " - graph capture disabled");
            cudaFree(d_head_params_);
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            return;
        }

        cudaStream_t init_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
        cudaMemsetAsync(d_head_params_, 0, num_entries * sizeof(int), init_stream);
        cudaStreamSynchronize(init_stream);
        std::memset(h_head_params_, 0, num_entries * sizeof(int));

        LOG_DEBUG("[CUDARingKVCacheBase] Allocated device params for graph capture: "
                  << num_entries << " entries (" << num_entries * sizeof(int) << " bytes)");
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
        h_head_params_[idx] = entryHead(layer, seq_idx);
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

        onAdvanceComplete(layer, seq_idx);
    }

} // namespace llaminar2
