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
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Check whether the opt-in MTP publication trace is enabled.
         *
         * ROCm KV sequence metadata should normally remain fully device-owned.
         * This helper gates the diagnostic D2H probes that make publication
         * bugs observable while preserving the production path's ownership
         * model when the trace is not explicitly requested.
         */
        bool mtpKVPublicationDiagnosticsEnabled()
        {
            return debugEnv().runtime_debug.mtp_publication_diagnostics;
        }
    } // namespace

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

    extern "C" bool hip_kv_sequence_state_set(
        int *d_head,
        int *d_count,
        int head,
        int count,
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
          device_id_(device_id),
          append_count_sources_(static_cast<size_t>(std::max(0, n_layers)) *
                                    static_cast<size_t>(std::max(0, batch_size)),
                                nullptr)
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
        const int num_entries = n_layers_ * batch_size_;
        if (num_entries == 0)
        {
            // All-GDN hybrid caches have no FA ring metadata to publish.
            return;
        }
        hipError_t err = hipMalloc(&d_head_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate device head params: "
                     << hipGetErrorString(err) << " - graph capture disabled");
            d_head_params_ = nullptr;
            return;
        }

        err = hipMalloc(&d_count_params_, num_entries * sizeof(int));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmRingKVCacheBase] Failed to allocate device count params: "
                     << hipGetErrorString(err) << " - device-resident KV sequence publication disabled");
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
        if (d_count_params_)
        {
            hipError_t err = hipFree(d_count_params_);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                fprintf(stderr, "WARNING: hipFree(d_count_params_) failed: %s\n", hipGetErrorString(err));
            }
            d_count_params_ = nullptr;
        }
        std::fill(
            append_count_sources_.begin(),
            append_count_sources_.end(),
            nullptr);
    }

    // =========================================================================
    // IKVCache Basic Operations
    // =========================================================================

    int ROCmRingKVCacheBase::get_cached_tokens(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state)
                   ? state.cached_tokens
                   : 0;
    }

    int ROCmRingKVCacheBase::get_head_position(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state)
                   ? state.implementation_head
                   : 0;
    }

    bool ROCmRingKVCacheBase::is_wrapped(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state) && state.wrapped;
    }

    IKVCache::KVCacheSequenceState ROCmRingKVCacheBase::sequenceState(
        int global_layer,
        int seq_idx) const
    {
        KVCacheSequenceState state;
        (void)observeDeviceSequenceState(global_layer, seq_idx, &state);
        return state;
    }

    bool ROCmRingKVCacheBase::observeDeviceSequenceState(
        int layer,
        int seq_idx,
        KVCacheSequenceState *state) const
    {
        if (!state || !validLayerSeq(layer, seq_idx) ||
            !d_head_params_ || !d_count_params_)
        {
            return false;
        }
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Host sequence-state observation is forbidden during graph capture");
            return false;
        }

        (void)hipSetDevice(device_id_);
        const int index = layer * batch_size_ + seq_idx;
        int head = 0;
        int count = 0;
        hipError_t status = hipDeviceSynchronize();
        if (status == hipSuccess)
        {
            status = hipMemcpy(
                &head,
                &d_head_params_[index],
                sizeof(head),
                hipMemcpyDeviceToHost);
        }
        if (status == hipSuccess)
        {
            status = hipMemcpy(
                &count,
                &d_count_params_[index],
                sizeof(count),
                hipMemcpyDeviceToHost);
        }
        if (status != hipSuccess || head < 0 || head >= max_seq_len_ ||
            count < 0 || count > max_seq_len_)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to observe canonical device sequence state"
                      << " layer=" << layer
                      << " seq_idx=" << seq_idx
                      << " status=" << hipGetErrorString(status)
                      << " head=" << head
                      << " count=" << count);
            return false;
        }

        const int tail = count > 0
                             ? (head - count + max_seq_len_) % max_seq_len_
                             : head;
        *state = KVCacheSequenceState{
            .cached_tokens = count,
            .implementation_head = head,
            .wrapped = count > 0 && tail >= head,
        };
        return true;
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
        append_count_sources_[static_cast<size_t>(
            layer * batch_size_ + seq_idx)] = nullptr;
        hipStream_t stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance()
                .getAMDContext(device_id_)
                .defaultStream());
        if (!setDeviceSequenceState(layer, seq_idx, 0, 0, stream) ||
            hipStreamSynchronize(stream) != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to reset canonical device sequence state"
                      << " layer=" << layer
                      << " seq_idx=" << seq_idx);
        }
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

    bool ROCmRingKVCacheBase::bindGraphAppendCountSource(
        int layer,
        int seq_idx,
        const int32_t *append_tokens_device,
        int captured_max_tokens,
        void *gpu_stream)
    {
        if (!validLayerSeq(layer, seq_idx) ||
            captured_max_tokens <= 0 ||
            !gpu_stream ||
            !d_head_params_ ||
            !d_count_params_)
        {
            return false;
        }
        const int idx = layer * batch_size_ + seq_idx;
        append_count_sources_[static_cast<size_t>(idx)] =
            append_tokens_device;
        if (mtpKVPublicationDiagnosticsEnabled())
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=rocm_kv_graph_append_binding"
                     << " mode="
                     << (append_tokens_device ? "resident_device_count" : "captured_exact_shape")
                     << " layer=" << layer
                     << " seq_idx=" << seq_idx
                     << " captured_max_tokens=" << captured_max_tokens
                     << " count_source="
                     << static_cast<const void *>(append_tokens_device)
                     << " stream=" << gpu_stream);
        }
        return true;
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

    const int *ROCmRingKVCacheBase::deviceDynamicAppendCountPtr(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx) || append_count_sources_.empty())
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return append_count_sources_[static_cast<size_t>(idx)];
    }

    bool ROCmRingKVCacheBase::setDeviceSequenceState(
        int layer,
        int seq_idx,
        int head,
        int count,
        void *gpu_stream)
    {
        if (!validLayerSeq(layer, seq_idx) ||
            !gpu_stream ||
            !d_head_params_ ||
            !d_count_params_ ||
            head < 0 ||
            head >= max_seq_len_ ||
            count < 0 ||
            count > max_seq_len_)
            return false;
        const int idx = layer * batch_size_ + seq_idx;
        return hip_kv_sequence_state_set(
            &d_head_params_[idx],
            &d_count_params_[idx],
            head,
            count,
            max_seq_len_,
            static_cast<hipStream_t>(gpu_stream));
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
                    "ROCm KV device sequence-state publication requires canonical device head/count rows";
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
        if (enqueued && mtpKVPublicationDiagnosticsEnabled())
        {
            const int seq_idx = request.first_seq_idx;
            const int idx = seq_idx;
            int device_head = -1;
            int device_count = -1;
            hipError_t diag_err = hipMemcpyAsync(
                &device_head,
                &d_head_params_[idx],
                sizeof(int),
                hipMemcpyDeviceToHost,
                static_cast<hipStream_t>(request.stream));
            if (diag_err == hipSuccess)
            {
                diag_err = hipMemcpyAsync(
                    &device_count,
                    &d_count_params_[idx],
                    sizeof(int),
                    hipMemcpyDeviceToHost,
                    static_cast<hipStream_t>(request.stream));
            }
            if (diag_err == hipSuccess)
                diag_err = hipStreamSynchronize(static_cast<hipStream_t>(request.stream));
            if (diag_err == hipSuccess)
            {
                LOG_INFO("[MTPPublicationDiagnostics] phase=rocm_kv_sequence_state_publish"
                         << " first_seq_idx=" << request.first_seq_idx
                         << " request_count=" << request.request_count
                         << " layer0_seq0_head_after=" << device_head
                         << " layer0_seq0_count_after=" << device_count
                         << " stream=" << request.stream);
            }
            else
            {
                LOG_ERROR("[MTPPublicationDiagnostics] phase=rocm_kv_sequence_state_publish"
                          << " diagnostic_copy_failed=" << hipGetErrorString(diag_err)
                          << " first_seq_idx=" << request.first_seq_idx
                          << " request_count=" << request.request_count);
            }
        }
        return enqueued;
    }

} // namespace llaminar2
