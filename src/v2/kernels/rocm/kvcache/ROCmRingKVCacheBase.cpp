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
#include <algorithm>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace llaminar2
{
    extern "C" bool hip_kv_sequence_state_publish(
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
        hipStream_t stream);

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

        err = hipMalloc(&d_count_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate device count params: "
                     << hipGetErrorString(err) << " - device-resident KV sequence publication disabled");
            d_count_params_ = nullptr;
            h_count_params_ = nullptr;
            freeDeviceParams();
            return;
        }

        err = hipHostMalloc(&h_head_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate pinned head params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            h_head_params_ = nullptr;
            h_count_params_ = nullptr;
            freeDeviceParams();
            return;
        }

        err = hipHostMalloc(&h_count_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate pinned count params: "
                     << hipGetErrorString(err) << " - device-resident KV sequence publication disabled");
            h_count_params_ = nullptr;
            freeDeviceParams();
            return;
        }

        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        err = hipMemsetAsync(d_head_params_, 0, num_entries * sizeof(int), init_stream);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to initialize device head params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            freeDeviceParams();
            return;
        }
        err = hipMemsetAsync(d_count_params_, 0, num_entries * sizeof(int), init_stream);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to initialize device count params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            freeDeviceParams();
            return;
        }
        err = hipStreamSynchronize(init_stream);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to synchronize device param initialization: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            freeDeviceParams();
            return;
        }
        std::memset(h_head_params_, 0, num_entries * sizeof(int));
        std::memset(h_count_params_, 0, num_entries * sizeof(int));

        LOG_DEBUG("[ROCmRingKVCacheBase] Allocated device params for graph capture: "
                  << num_entries << " entries (" << num_entries * sizeof(int) * 2 << " bytes)");
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
        if (d_count_params_)
        {
            hipError_t err = hipFree(d_count_params_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(d_count_params_) failed: %s\n", hipGetErrorString(err));
            }
            d_count_params_ = nullptr;
        }
        if (h_count_params_)
        {
            hipError_t err = hipHostFree(h_count_params_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipHostFree(h_count_params_) failed: %s\n", hipGetErrorString(err));
            }
            h_count_params_ = nullptr;
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
        refreshHostDeviceParamMirror(layer, seq_idx);
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
        refreshHostDeviceParamMirror(layer, seq_idx);
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
        if (d_count_params_ && h_count_params_)
        {
            err = hipMemcpyAsync(
                &d_count_params_[idx],
                &h_count_params_[idx],
                sizeof(int),
                hipMemcpyHostToDevice,
                static_cast<hipStream_t>(gpu_stream));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRingKVCacheBase] Failed to upload dynamic KV count: "
                          << hipGetErrorString(err));
            }
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
        refreshHostDeviceParamMirror(layer, seq_idx);

        onAdvanceComplete(layer, seq_idx);
    }

    const int *ROCmRingKVCacheBase::deviceCachedTokenCountPtr(int layer, int seq_idx) const
    {
        if (!d_count_params_ || !validLayerSeq(layer, seq_idx))
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return &d_count_params_[idx];
    }

    const int *ROCmRingKVCacheBase::deviceRingHeadPtr(int layer, int seq_idx) const
    {
        if (!d_head_params_ || !validLayerSeq(layer, seq_idx))
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return &d_head_params_[idx];
    }

    void ROCmRingKVCacheBase::refreshHostDeviceParamMirror(int layer, int seq_idx)
    {
        if (!validLayerSeq(layer, seq_idx))
            return;
        const int idx = layer * batch_size_ + seq_idx;
        if (h_head_params_)
            h_head_params_[idx] = entryHead(layer, seq_idx);
        if (h_count_params_)
            h_count_params_[idx] = entryCount(layer, seq_idx);
    }

    bool ROCmRingKVCacheBase::uploadHostDeviceParamMirror(int layer, int seq_idx, void *gpu_stream)
    {
        if (!validLayerSeq(layer, seq_idx))
            return false;
        if (!gpu_stream || !d_head_params_ || !h_head_params_ ||
            !d_count_params_ || !h_count_params_)
            return false;
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Refusing to upload KV sequence-state mirror inside HIP graph capture");
            return false;
        }

        refreshHostDeviceParamMirror(layer, seq_idx);
        const int idx = layer * batch_size_ + seq_idx;
        hipError_t err = hipMemcpyAsync(
            &d_head_params_[idx],
            &h_head_params_[idx],
            sizeof(int),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(gpu_stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to upload KV head mirror: "
                      << hipGetErrorString(err));
            return false;
        }
        err = hipMemcpyAsync(
            &d_count_params_[idx],
            &h_count_params_[idx],
            sizeof(int),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(gpu_stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to upload KV count mirror: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmRingKVCacheBase::publishSequenceStateFromDeviceMetadata(
        const DeviceSequenceStatePublicationRequest &request,
        std::string *error)
    {
        if (!request.valid())
        {
            if (error)
            {
                *error =
                    "invalid ROCm KV device sequence-state publication request";
            }
            return false;
        }
        if (!d_head_params_ || !d_count_params_)
        {
            if (error)
            {
                *error =
                    "ROCm KV device sequence-state publication requires device head/count mirrors";
            }
            return false;
        }
        if (request.first_seq_idx + request.request_count > batch_size_)
        {
            if (error)
            {
                *error =
                    "ROCm KV device sequence-state publication request exceeds batch size";
            }
            return false;
        }

        const bool enqueued = hip_kv_sequence_state_publish(
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
            static_cast<hipStream_t>(request.stream));
        if (!enqueued && error)
        {
            *error = "failed to enqueue ROCm KV device sequence-state publication";
        }
        return enqueued;
    }

    bool ROCmRingKVCacheBase::adoptSequenceStateFromHostMetadata(
        const HostSequenceStatePublicationRequest &request,
        std::string *error)
    {
        if (!request.valid())
        {
            if (error)
            {
                *error =
                    "invalid ROCm KV host sequence-state adoption request";
            }
            return false;
        }
        if (request.first_seq_idx + request.request_count > batch_size_)
        {
            if (error)
            {
                *error =
                    "ROCm KV host sequence-state adoption request exceeds batch size";
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
                        "ROCm KV host sequence-state adoption received invalid counts";
                }
                return false;
            }

            for (int layer = 0; layer < n_layers_; ++layer)
            {
                const int old_count = entryCount(layer, seq_idx);
                const int evicted =
                    std::max(0, old_count + accepted_count - target_count);
                if (evicted > 0)
                    onEviction(layer, seq_idx, evicted);

                const int old_head = entryHead(layer, seq_idx);
                setEntryHead(
                    layer,
                    seq_idx,
                    (old_head + accepted_count) % max_seq_len_);
                setEntryCount(layer, seq_idx, target_count);
                refreshHostDeviceParamMirror(layer, seq_idx);
                onAdvanceComplete(layer, seq_idx);
            }
        }

        return true;
    }

} // namespace llaminar2
