/**
 * @file CUDARingKVCacheBase.cpp
 * @brief Implementation of CUDARingKVCacheBase common ring buffer operations
 *
 * Owns canonical device sequence metadata and graph append-count bindings.
 * Persistent host sequence mirrors are intentionally not part of this layer.
 */

#include "CUDARingKVCacheBase.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <algorithm>
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
         * The dynamic KV metadata path is normally device-owned and should not
         * synchronize just to make logs pretty.  This helper is intentionally
         * local to the CUDA KV cache translation unit so diagnostic D2H probes
         * stay gated behind the same explicit environment variable used by the
         * orchestrator publication trace.
         */
        bool mtpKVPublicationDiagnosticsEnabled()
        {
            return debugEnv().runtime_debug.mtp_publication_diagnostics;
        }
    } // namespace

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

    extern "C" bool cuda_kv_sequence_state_set(
        int *d_head,
        int *d_count,
        int head,
        int count,
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
          device_id_(device_id),
          append_count_sources_(static_cast<size_t>(std::max(0, n_layers)) *
                                    static_cast<size_t>(std::max(0, batch_size)),
                                nullptr)
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
        const int num_entries = n_layers_ * batch_size_;
        if (num_entries == 0)
        {
            // All-GDN hybrid caches have no FA ring metadata to publish.
            return;
        }
        cudaError_t err = cudaMalloc(&d_head_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate device head params: "
                     << cudaGetErrorString(err) << " - graph capture disabled");
            d_head_params_ = nullptr;
            return;
        }

        err = cudaMalloc(&d_count_params_, num_entries * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to allocate device count params: "
                     << cudaGetErrorString(err) << " - device-resident KV sequence publication disabled");
            freeDeviceParams();
            return;
        }

        cudaStream_t init_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
        err = cudaMemsetAsync(
            d_head_params_, 0, num_entries * sizeof(int), init_stream);
        if (err == cudaSuccess)
        {
            err = cudaMemsetAsync(
                d_count_params_, 0, num_entries * sizeof(int), init_stream);
        }
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(init_stream);
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCacheBase] Failed to initialize canonical device sequence state: "
                     << cudaGetErrorString(err));
            freeDeviceParams();
            return;
        }

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
        if (d_count_params_)
        {
            cudaError_t err = cudaFree(d_count_params_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_count_params_) failed: %s\n", cudaGetErrorString(err));
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

    int CUDARingKVCacheBase::get_cached_tokens(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state)
                   ? state.cached_tokens
                   : 0;
    }

    int CUDARingKVCacheBase::get_head_position(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state)
                   ? state.implementation_head
                   : 0;
    }

    bool CUDARingKVCacheBase::is_wrapped(int layer, int seq_idx) const
    {
        KVCacheSequenceState state;
        return observeDeviceSequenceState(layer, seq_idx, &state) && state.wrapped;
    }

    IKVCache::KVCacheSequenceState CUDARingKVCacheBase::sequenceState(
        int global_layer,
        int seq_idx) const
    {
        KVCacheSequenceState state;
        (void)observeDeviceSequenceState(global_layer, seq_idx, &state);
        return state;
    }

    bool CUDARingKVCacheBase::observeDeviceSequenceState(
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
            LOG_ERROR("[CUDARingKVCacheBase] Host sequence-state observation is forbidden during graph capture");
            return false;
        }

        (void)cudaSetDevice(device_id_);
        const int index = layer * batch_size_ + seq_idx;
        int head = 0;
        int count = 0;
        cudaError_t status = cudaDeviceSynchronize();
        if (status == cudaSuccess)
        {
            status = cudaMemcpy(
                &head,
                &d_head_params_[index],
                sizeof(head),
                cudaMemcpyDeviceToHost);
        }
        if (status == cudaSuccess)
        {
            status = cudaMemcpy(
                &count,
                &d_count_params_[index],
                sizeof(count),
                cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess || head < 0 || head >= max_seq_len_ ||
            count < 0 || count > max_seq_len_)
        {
            LOG_ERROR("[CUDARingKVCacheBase] Failed to observe canonical device sequence state"
                      << " layer=" << layer
                      << " seq_idx=" << seq_idx
                      << " status=" << cudaGetErrorString(status)
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
        append_count_sources_[static_cast<size_t>(
            layer * batch_size_ + seq_idx)] = nullptr;
        cudaStream_t stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance()
                .getNvidiaContext(device_id_)
                .defaultStream());
        if (!setDeviceSequenceState(layer, seq_idx, 0, 0, stream) ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCacheBase] Failed to reset canonical device sequence state"
                      << " layer=" << layer
                      << " seq_idx=" << seq_idx);
        }
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

    bool CUDARingKVCacheBase::bindGraphAppendCountSource(
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
            LOG_INFO("[MTPPublicationDiagnostics] phase=cuda_kv_graph_append_binding"
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

    const int *CUDARingKVCacheBase::deviceDynamicAppendCountPtr(int layer, int seq_idx) const
    {
        if (!validLayerSeq(layer, seq_idx) || append_count_sources_.empty())
            return nullptr;
        const int idx = layer * batch_size_ + seq_idx;
        return append_count_sources_[static_cast<size_t>(idx)];
    }

    bool CUDARingKVCacheBase::setDeviceSequenceState(
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
        return cuda_kv_sequence_state_set(
            &d_head_params_[idx],
            &d_count_params_[idx],
            head,
            count,
            max_seq_len_,
            static_cast<cudaStream_t>(gpu_stream));
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
                    "CUDA KV device sequence-state publication requires canonical device head/count rows";
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
        if (enqueued && mtpKVPublicationDiagnosticsEnabled())
        {
            const int seq_idx = request.first_seq_idx;
            const int idx = seq_idx;
            int device_head = -1;
            int device_count = -1;
            cudaError_t diag_err = cudaMemcpyAsync(
                &device_head,
                &d_head_params_[idx],
                sizeof(int),
                cudaMemcpyDeviceToHost,
                static_cast<cudaStream_t>(request.stream));
            if (diag_err == cudaSuccess)
            {
                diag_err = cudaMemcpyAsync(
                    &device_count,
                    &d_count_params_[idx],
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(request.stream));
            }
            if (diag_err == cudaSuccess)
                diag_err = cudaStreamSynchronize(static_cast<cudaStream_t>(request.stream));
            if (diag_err == cudaSuccess)
            {
                LOG_INFO("[MTPPublicationDiagnostics] phase=cuda_kv_sequence_state_publish"
                         << " first_seq_idx=" << request.first_seq_idx
                         << " request_count=" << request.request_count
                         << " layer0_seq0_head_after=" << device_head
                         << " layer0_seq0_count_after=" << device_count
                         << " stream=" << request.stream);
            }
            else
            {
                LOG_ERROR("[MTPPublicationDiagnostics] phase=cuda_kv_sequence_state_publish"
                          << " diagnostic_copy_failed=" << cudaGetErrorString(diag_err)
                          << " first_seq_idx=" << request.first_seq_idx
                          << " request_count=" << request.request_count);
            }
        }
        return enqueued;
    }

} // namespace llaminar2
