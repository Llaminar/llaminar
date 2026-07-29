/**
 * @file ROCmRingKVCacheBase.cpp
 * @brief Implementation of ROCmRingKVCacheBase common ring buffer operations
 *
 * Device sequence metadata is allocated through the canonical ROCm backend.
 * No custom kernels are defined here.
 */

#include "ROCmRingKVCacheBase.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
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

    extern "C" bool hip_kv_sequence_state_checkpoint_capture(
        const int *d_heads,
        const int *d_counts,
        int *checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx,
        hipStream_t stream);

    extern "C" bool hip_kv_sequence_state_checkpoint_restore(
        int *d_heads,
        int *d_counts,
        const int *checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx,
        hipStream_t stream);

    extern "C" bool hip_kv_sequence_state_truncate(
        int *d_heads,
        int *d_counts,
        int n_layers,
        int batch_size,
        int seq_idx,
        int cached_tokens,
        int max_seq_len,
        hipStream_t stream);

    extern "C" bool hip_kv_sequence_state_evict_oldest(
        int *d_heads,
        int *d_counts,
        int n_layers,
        int batch_size,
        int layer,
        int seq_idx,
        int num_tokens,
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

    bool ROCmRingKVCacheBase::activateOwningDevice(
        const char *operation,
        std::string *error) const
    {
        /*
         * This is an execution boundary rather than an inner kernel loop.
         * Force the runtime selection so direct HIP callers cannot leave the
         * thread-local HipDeviceGuard cache disagreeing with the HIP runtime.
         */
        const hipError_t status = static_cast<hipError_t>(
            HipDeviceGuard::forceSetDevice(device_id_));
        if (status == hipSuccess)
            return true;

        const std::string message =
            std::string("ROCm KV ") +
            (operation && operation[0] != '\0' ? operation : "device operation") +
            " could not activate owning device " +
            std::to_string(device_id_) + ": " +
            hipGetErrorString(status);
        if (error)
            *error = message;
        LOG_ERROR("[ROCmRingKVCacheBase] " << message);
        return false;
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
        if (!activateOwningDevice("device-parameter allocation"))
            return;

        auto *backend = getROCmBackend();
        if (!backend)
        {
            throw std::runtime_error(
                "[ROCmRingKVCacheBase] ROCm backend unavailable during metadata allocation");
        }

        const size_t metadata_bytes =
            static_cast<size_t>(num_entries) * sizeof(int);
        d_head_params_ =
            static_cast<int *>(backend->allocate(metadata_bytes, device_id_));
        d_count_params_ =
            static_cast<int *>(backend->allocate(metadata_bytes, device_id_));
        if (!d_head_params_ || !d_count_params_)
        {
            freeDeviceParams();
            throw std::runtime_error(
                "[ROCmRingKVCacheBase] Failed to allocate mandatory device sequence metadata");
        }

        hipStream_t init_stream = static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
        if (!init_stream ||
            !backend->memset(
                d_head_params_, 0, metadata_bytes, device_id_, init_stream) ||
            !backend->memset(
                d_count_params_, 0, metadata_bytes, device_id_, init_stream))
        {
            freeDeviceParams();
            throw std::runtime_error(
                "[ROCmRingKVCacheBase] Failed to initialize mandatory device sequence metadata");
        }
        LOG_DEBUG("[ROCmRingKVCacheBase] Allocated device params for graph capture: "
                  << num_entries << " entries (" << num_entries * sizeof(int) * 2 << " bytes)");
    }

    void ROCmRingKVCacheBase::freeDeviceParams()
    {
        auto *backend = getROCmBackend();
        if ((d_head_params_ || d_count_params_) && !backend)
        {
            throw std::runtime_error(
                "[ROCmRingKVCacheBase] ROCm backend unavailable during metadata release");
        }
        if (d_head_params_)
        {
            backend->free(d_head_params_, device_id_);
            d_head_params_ = nullptr;
        }
        if (d_count_params_)
        {
            backend->free(d_count_params_, device_id_);
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

    bool ROCmRingKVCacheBase::truncateSequence(
        int seq_idx,
        int cached_tokens,
        void *stream)
    {
        if (!stream ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            cached_tokens < 0 || cached_tokens > max_seq_len_ ||
            !d_head_params_ || !d_count_params_)
        {
            return false;
        }
        if (!activateOwningDevice("sequence-state truncation"))
            return false;

        if (!hip_kv_sequence_state_truncate(
                d_head_params_,
                d_count_params_,
                n_layers_,
                batch_size_,
                seq_idx,
                cached_tokens,
                max_seq_len_,
                static_cast<hipStream_t>(stream)))
        {
            return false;
        }
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            onResetLayerSequenceState(layer, seq_idx);
        }
        return true;
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

        if (!activateOwningDevice("sequence-state observation"))
            return false;

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

    bool ROCmRingKVCacheBase::resetRequestState(
        const StateResetContext &context)
    {
        if (!context.permitsRequestReset() ||
            !context.execution_stream || !context.hasReason() ||
            !d_head_params_ || !d_count_params_ ||
            !activateOwningDevice("request-state reset"))
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Request-state reset requires an explicit stream, reason, and device metadata");
            return false;
        }

        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        const size_t metadata_bytes = entry_count * sizeof(int);
        auto stream = static_cast<hipStream_t>(context.execution_stream);

        /*
         * Keep the two publications separate even though they share a stream.
         * A prior asynchronous kernel fault is reported by the next HIP API
         * that observes it; preserving the exact failing operation tells an
         * E2E crash whether the head reset itself failed or whether the first
         * reset succeeded and count publication then failed.  Do not retry or
         * clear the sticky error: request reset is an ownership boundary and
         * must fail hard when either canonical metadata bank is not reset.
         */
        const hipError_t head_status =
            hipMemsetAsync(d_head_params_, 0, metadata_bytes, stream);
        if (head_status != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to enqueue request-state head reset"
                      << " reason=" << context.reason
                      << " device=" << device_id_
                      << " bytes=" << metadata_bytes
                      << " status=" << hipGetErrorString(head_status));
            std::fprintf(
                stderr,
                "[FATAL] ROCm KV request reset failed: bank=head device=%d "
                "stream=%p bytes=%zu status=%d (%s) reason=%s\n",
                device_id_,
                context.execution_stream,
                metadata_bytes,
                static_cast<int>(head_status),
                hipGetErrorString(head_status),
                context.reason);
            std::fflush(stderr);
            return false;
        }

        const hipError_t count_status =
            hipMemsetAsync(d_count_params_, 0, metadata_bytes, stream);
        if (count_status != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Failed to enqueue request-state count reset"
                      << " reason=" << context.reason
                      << " device=" << device_id_
                      << " bytes=" << metadata_bytes
                      << " status=" << hipGetErrorString(count_status));
            std::fprintf(
                stderr,
                "[FATAL] ROCm KV request reset failed: bank=count device=%d "
                "stream=%p bytes=%zu status=%d (%s) reason=%s\n",
                device_id_,
                context.execution_stream,
                metadata_bytes,
                static_cast<int>(count_status),
                hipGetErrorString(count_status),
                context.reason);
            std::fflush(stderr);
            return false;
        }

        std::fill(append_count_sources_.begin(), append_count_sources_.end(), nullptr);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
                onResetLayerSequenceState(layer, seq_idx);
        }
        return true;
    }

    bool ROCmRingKVCacheBase::resetSequenceState(
        int seq_idx,
        const StateResetContext &context)
    {
        if (!context.permitsSequenceReset() ||
            !context.execution_stream || !context.hasReason() ||
            seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Sequence-state reset has invalid ownership"
                      << " seq_idx=" << seq_idx);
            return false;
        }
        if (!truncateSequence(seq_idx, 0, context.execution_stream))
            return false;

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            append_count_sources_[static_cast<size_t>(
                layer * batch_size_ + seq_idx)] = nullptr;
        }
        return true;
    }

    bool ROCmRingKVCacheBase::resetLayerSequenceState(
        int layer,
        int seq_idx,
        const StateResetContext &context)
    {
        if (!context.permitsLayerSequenceReset() ||
            !context.execution_stream || !context.hasReason() ||
            !validLayerSeq(layer, seq_idx))
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Layer/sequence reset has invalid ownership"
                      << " layer=" << layer
                      << " seq_idx=" << seq_idx);
            return false;
        }
        append_count_sources_[static_cast<size_t>(
            layer * batch_size_ + seq_idx)] = nullptr;
        if (!setDeviceSequenceState(
                layer,
                seq_idx,
                0,
                0,
                context.execution_stream))
        {
            return false;
        }
        onResetLayerSequenceState(layer, seq_idx);
        return true;
    }

    bool ROCmRingKVCacheBase::resetLayerState(
        int layer,
        const StateResetContext &context)
    {
        if (!context.permitsLayerReset() ||
            !context.execution_stream || !context.hasReason() ||
            layer < 0 || layer >= n_layers_ ||
            !d_head_params_ || !d_count_params_ ||
            !activateOwningDevice("layer-state reset"))
        {
            LOG_ERROR("[ROCmRingKVCacheBase] Layer-state reset has invalid ownership"
                      << " layer=" << layer);
            return false;
        }

        const size_t first_entry =
            static_cast<size_t>(layer) * static_cast<size_t>(batch_size_);
        const size_t metadata_bytes =
            static_cast<size_t>(batch_size_) * sizeof(int);
        auto stream = static_cast<hipStream_t>(context.execution_stream);
        if (hipMemsetAsync(
                d_head_params_ + first_entry,
                0,
                metadata_bytes,
                stream) != hipSuccess ||
            hipMemsetAsync(
                d_count_params_ + first_entry,
                0,
                metadata_bytes,
                stream) != hipSuccess)
        {
            return false;
        }

        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            append_count_sources_[first_entry + static_cast<size_t>(seq_idx)] =
                nullptr;
            onResetLayerSequenceState(layer, seq_idx);
        }
        return true;
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

    size_t ROCmRingKVCacheBase::deviceSequenceStateCheckpointBytes() const
    {
        if (!d_head_params_ || !d_count_params_ || n_layers_ <= 0)
            return 0;
        return sizeof(int32_t) * static_cast<size_t>(n_layers_) * 2u;
    }

    bool ROCmRingKVCacheBase::captureDeviceSequenceStateCheckpoint(
        int seq_idx,
        void *checkpoint_device,
        size_t checkpoint_bytes,
        void *stream,
        std::string *error) const
    {
        const size_t required_bytes = deviceSequenceStateCheckpointBytes();
        if (!checkpoint_device || !stream ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            required_bytes == 0 || checkpoint_bytes < required_bytes)
        {
            if (error)
            {
                *error =
                    "invalid ROCm device sequence-state checkpoint capture request";
            }
            return false;
        }
        if (!activateOwningDevice(
                "device sequence-state checkpoint capture",
                error))
        {
            return false;
        }

        const bool enqueued = hip_kv_sequence_state_checkpoint_capture(
            d_head_params_,
            d_count_params_,
            static_cast<int *>(checkpoint_device),
            n_layers_,
            batch_size_,
            seq_idx,
            static_cast<hipStream_t>(stream));
        if (!enqueued && error)
        {
            *error =
                "failed to enqueue ROCm device sequence-state checkpoint capture";
        }
        return enqueued;
    }

    bool ROCmRingKVCacheBase::restoreDeviceSequenceStateCheckpoint(
        int seq_idx,
        const void *checkpoint_device,
        size_t checkpoint_bytes,
        void *stream,
        std::string *error)
    {
        const size_t required_bytes = deviceSequenceStateCheckpointBytes();
        if (!checkpoint_device || !stream ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            required_bytes == 0 || checkpoint_bytes < required_bytes)
        {
            if (error)
            {
                *error =
                    "invalid ROCm device sequence-state checkpoint restore request";
            }
            return false;
        }
        if (!activateOwningDevice(
                "device sequence-state checkpoint restore",
                error))
        {
            return false;
        }

        const bool enqueued = hip_kv_sequence_state_checkpoint_restore(
            d_head_params_,
            d_count_params_,
            static_cast<const int *>(checkpoint_device),
            n_layers_,
            batch_size_,
            seq_idx,
            static_cast<hipStream_t>(stream));
        if (!enqueued)
        {
            if (error)
            {
                *error =
                    "failed to enqueue ROCm device sequence-state checkpoint restore";
            }
            return false;
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            onResetLayerSequenceState(layer, seq_idx);
        }
        return true;
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
        if (!activateOwningDevice("sequence-state replacement"))
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

    bool ROCmRingKVCacheBase::evictOldestDeviceSequenceState(
        int layer,
        int seq_idx,
        int num_tokens,
        void *gpu_stream)
    {
        if (!validLayerSeq(layer, seq_idx) ||
            num_tokens < 0 ||
            !gpu_stream ||
            !d_head_params_ ||
            !d_count_params_)
        {
            return false;
        }
        if (!activateOwningDevice("oldest-sequence-state eviction"))
            return false;

        return hip_kv_sequence_state_evict_oldest(
            d_head_params_,
            d_count_params_,
            n_layers_,
            batch_size_,
            layer,
            seq_idx,
            num_tokens,
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
        if (!activateOwningDevice(
                "device sequence-state publication",
                error))
        {
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
