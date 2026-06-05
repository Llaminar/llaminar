/**
 * @file ROCmRingKVCacheBase.cpp
 * @brief Implementation of ROCmRingKVCacheBase common ring buffer operations
 *
 * Uses HIP runtime API for device param allocation (hipMalloc, hipFree,
 * hipHostMalloc, hipHostFree). No custom kernels — just memory management
 * and host-side bookkeeping.
 */

#include "ROCmRingKVCacheBase.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    ROCmRingKVCacheBase::ROCmRingKVCacheBase(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int kv_dim, int device_id)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), head_dim_(head_dim), kv_dim_(kv_dim),
          device_id_(device_id)
    {
    }

    ROCmRingKVCacheBase::~ROCmRingKVCacheBase()
    {
        // Derived class destructors have already run and called hipSetDevice().
        // hipFree/hipHostFree work on the allocating device's memory.
        freeDeviceParams();
    }

    // =========================================================================
    // Graph Capture Device Params Management
    // =========================================================================

    void ROCmRingKVCacheBase::allocateDeviceParams()
    {
        int num_entries = n_layers_ * batch_size_;
        hipError_t err;

        err = hipMalloc(&d_head_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate device head params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            return;
        }

        err = hipHostMalloc(&h_head_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate pinned head params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            hipFree(d_head_params_);
            d_head_params_ = nullptr;
            h_head_params_ = nullptr;
            return;
        }

        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        hipMemsetAsync(d_head_params_, 0, num_entries * sizeof(int), init_stream);
        hipStreamSynchronize(init_stream);
        std::memset(h_head_params_, 0, num_entries * sizeof(int));

        LOG_DEBUG("[ROCmRingKVCacheBase] Allocated device params for graph capture: "
                  << num_entries << " entries (" << num_entries * sizeof(int) << " bytes)");
    }

    void ROCmRingKVCacheBase::freeDeviceParams()
    {
        if (d_head_params_)
        {
            hipError_t err = hipFree(d_head_params_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(d_head_params_) failed: %s\n", hipGetErrorString(err));
            }
            d_head_params_ = nullptr;
        }
        if (h_head_params_)
        {
            hipError_t err = hipHostFree(h_head_params_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipHostFree(h_head_params_) failed: %s\n", hipGetErrorString(err));
            }
            h_head_params_ = nullptr;
        }
    }

    // =========================================================================
    // IKVCache Basic Operations
    // =========================================================================

    int ROCmRingKVCacheBase::get_cached_tokens(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx))
            return 0;
        return entryCount(layer, seq_idx);
    }

    int ROCmRingKVCacheBase::get_head_position(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx))
            return 0;
        return entryHead(layer, seq_idx);
    }

    bool ROCmRingKVCacheBase::is_wrapped(int layer, int seq_idx) const
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

    void ROCmRingKVCacheBase::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
            clear_layer(layer);
    }

    void ROCmRingKVCacheBase::clear_sequence(int layer, int seq_idx)
    {
        if (!validLayerSeq(layer, seq_idx))
            return;
        resetEntry(layer, seq_idx);
        onClearSequence(layer, seq_idx);
    }

    void ROCmRingKVCacheBase::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
            return;
        for (int seq = 0; seq < batch_size_; ++seq)
            clear_sequence(layer, seq);
    }

    // =========================================================================
    // Graph Capture Support
    // =========================================================================

    void ROCmRingKVCacheBase::setDynamicHead(int layer, int seq_idx, void *gpu_stream)
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
            LOG_ERROR("[ROCmRingKVCacheBase] Refusing to upload dynamic KV head inside HIP graph capture");
            return;
        }

        hipError_t err = hipMemcpyAsync(
            &d_head_params_[idx],
            &h_head_params_[idx],
            sizeof(int),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(gpu_stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to upload dynamic KV head: "
                      << hipGetErrorString(err));
        }
    }

    void ROCmRingKVCacheBase::advanceHead(int layer, int seq_idx, int num_tokens)
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
