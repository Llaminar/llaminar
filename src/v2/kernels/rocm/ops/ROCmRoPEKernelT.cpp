/**
 * @file ROCmRoPEKernelT.cpp
 * @brief ROCm RoPE kernel implementation
 *
 * Implementation of the ROCmRoPEKernelT template specializations.
 * Calls the extern "C" HIP wrapper functions defined in ROCmRoPEKernels.hip.
 *
 * OPTIMIZATIONS (v2):
 * - Pre-computed inverse frequency table cached per (head_dim, freq_base, device)
 * - Workspace-based position_ids buffer (no per-call hipMalloc/hipFree)
 * - Fused Q+K kernel launch to reduce overhead
 *
 * @author Llaminar Team
 * @date 2025-01-17
 */

#include "ROCmRoPEKernelT.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/ROCmKernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../ROCmKernelBase.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../kernels/rope/RoPEDeviceParams.h"
#include <hip/hip_runtime.h>
#include <bit>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

// Forward declare extern "C" HIP wrappers (v2 - with inv_freq parameter)
extern "C"
{
    bool hipOps_rope_fp32_v2(
        float *Q,
        float *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream);

    bool hipOps_rope_bf16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream);

    bool hipOps_rope_fp16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream);

    // Decode-optimized versions (scalar position, no H2D copy)
    bool hipOps_rope_fp32_decode(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

    bool hipOps_rope_bf16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

    bool hipOps_rope_fp16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

    // Contiguous position versions (zero-copy: positions computed on GPU)
    bool hipOps_rope_fp32_contiguous(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    bool hipOps_rope_bf16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    bool hipOps_rope_fp16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    // Workspace-aware inverse frequency population
    bool hipOps_rope_populate_inv_freq(
        float *d_inv_freq,
        int head_dim,
        float freq_base,
        int device_idx, void *stream);
    bool hipOps_rope_publish_device_params(
        llaminar2::rope::RoPEDeviceParams *device_params,
        int pos_offset,
        int device_idx,
        void *stream);
}

namespace
{
    constexpr const char *kROCmRoPEInvariantPublicationDomain =
        "rocm_rope_inverse_frequency_tables";

    /** @brief Return the collision-free FP32 identity used by publications. */
    std::uint32_t ropeThetaBits(float theta) noexcept
    {
        return std::bit_cast<std::uint32_t>(theta);
    }

    bool isHIPStreamCapturing(hipStream_t stream, const char *context)
    {
        if (llaminar2::isGraphCaptureActive())
            return true;

        if (!stream)
            return false;

        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t err = hipStreamIsCapturing(stream, &status);
        if (err != hipSuccess)
        {
            LOG_ERROR("[" << context << "] hipStreamIsCapturing failed: " << hipGetErrorString(err));
            return true;
        }
        return status == hipStreamCaptureStatusActive;
    }

    bool publishHIPRoPEDeviceParams(
        llaminar2::DeviceWorkspaceManager *workspace,
        hipStream_t stream,
        bool &device_valid,
        int &device_offset,
        int pos_offset,
        int device_idx,
        const char *context)
    {
        device_valid = false;

        if (!stream)
        {
            LOG_ERROR("[" << context << "] Cannot publish RoPE params on a null/default HIP stream");
            return false;
        }
        if (!workspace)
        {
            LOG_ERROR("[" << context << "] Cannot publish RoPE params without a bound workspace");
            return false;
        }
        if (isHIPStreamCapturing(stream, context))
        {
            LOG_ERROR("[" << context << "] Refusing to record mutable RoPE-param publication inside HIP graph capture");
            return false;
        }

        auto *d_params = workspace->getBuffer(llaminar2::RoPEWorkspaceBuffers::DEVICE_PARAMS);
        if (!d_params)
        {
            LOG_ERROR("[" << context << "] Missing workspace buffer "
                          << llaminar2::RoPEWorkspaceBuffers::DEVICE_PARAMS);
            return false;
        }

        if (!hipOps_rope_publish_device_params(
                static_cast<llaminar2::rope::RoPEDeviceParams *>(d_params),
                pos_offset,
                device_idx,
                stream))
        {
            LOG_ERROR("[" << context << "] Device-owned RoPE-param publication failed");
            return false;
        }

        device_offset = pos_offset;
        device_valid = true;
        return true;
    }

    bool uploadHIPRoPEPositionIds(
        llaminar2::DeviceWorkspaceManager *workspace,
        const int *position_ids,
        int seq_len,
        hipStream_t stream,
        bool &device_valid,
        int &device_seq_len,
        const char *context)
    {
        device_valid = false;
        device_seq_len = 0;

        if (!position_ids || seq_len <= 0)
        {
            LOG_ERROR("[" << context << "] Cannot upload empty RoPE position_ids");
            return false;
        }
        if (!stream)
        {
            LOG_ERROR("[" << context << "] Cannot upload RoPE position_ids on a null/default HIP stream");
            return false;
        }
        if (!workspace)
        {
            LOG_ERROR("[" << context << "] Cannot upload RoPE position_ids without a bound workspace");
            return false;
        }
        if (isHIPStreamCapturing(stream, context))
        {
            LOG_ERROR("[" << context << "] Refusing to record RoPE position_ids H2D inside HIP graph capture");
            return false;
        }

        auto *d_position_ids = workspace->getBuffer(llaminar2::RoPEWorkspaceBuffers::POSITION_IDS);
        if (!d_position_ids)
        {
            LOG_ERROR("[" << context << "] Missing workspace buffer "
                          << llaminar2::RoPEWorkspaceBuffers::POSITION_IDS);
            return false;
        }

        const size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
        const hipError_t copy_err =
            hipMemcpyAsync(d_position_ids, position_ids, pos_bytes,
                           hipMemcpyHostToDevice, stream);
        if (copy_err != hipSuccess)
        {
            LOG_ERROR("[" << context << "] hipMemcpyAsync failed for RoPE position_ids: "
                          << hipGetErrorString(copy_err));
            return false;
        }

        device_seq_len = seq_len;
        device_valid = true;
        return true;
    }

    /**
     * @brief Publish or adopt one immutable ROCm RoPE frequency table.
     *
     * Publication happens once during graph setup on the exact producer stream.
     * Every graph-local adopter receives a stable workspace slot and waits on
     * the producer event once per stream binding. Graph replay subsequently
     * uses only the retained device pointer and performs no registry work.
     */
    bool prepareHIPRoPEInvariantPublication(
        llaminar2::DeviceWorkspaceManager *workspace,
        hipStream_t stream,
        int device_idx,
        int rotary_dim,
        float rope_theta,
        std::shared_ptr<llaminar2::rope::RoPEInvariantWorkspacePublication>
            &current,
        void *&adoption_stream,
        const char *context)
    {
        if (!workspace || !stream || device_idx < 0 || rotary_dim <= 0 ||
            (rotary_dim % 2) != 0 ||
            rotary_dim / 2 > llaminar2::rope::kMaxInverseFrequencyValues ||
            !std::isfinite(rope_theta) || rope_theta <= 0.0f)
        {
            LOG_ERROR("[" << context << "] Invalid immutable RoPE publication"
                          << " workspace=" << workspace
                          << " stream=" << static_cast<void *>(stream)
                          << " device=" << device_idx
                          << " rotary_dim=" << rotary_dim
                          << " theta=" << rope_theta);
            return false;
        }
        if (!workspace->device().is_rocm() ||
            workspace->device().rocm_ordinal() != device_idx)
        {
            LOG_ERROR("[" << context << "] RoPE publication workspace/device mismatch: "
                          << workspace->device().to_string()
                          << " requested rocm:" << device_idx);
            return false;
        }

        const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
        const bool exact_current =
            current && current->device_values && current->ready_event &&
            current->rotary_dim == rotary_dim &&
            current->theta_bits == theta_bits;
        if (exact_current && adoption_stream == static_cast<void *>(stream))
            return true;

        if (isHIPStreamCapturing(stream, context))
        {
            LOG_ERROR("[" << context << "] Immutable RoPE state was not adopted "
                          "on the exact HIP stream before graph capture");
            return false;
        }

        if (!exact_current)
        {
            llaminar2::PersistentWorkspacePublicationKey key{
                .word0 = static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(rotary_dim)),
                .word1 = static_cast<std::uint64_t>(theta_bits),
                .word2 = static_cast<std::uint64_t>(
                    llaminar2::rope::kMaxInverseFrequencyValues),
                .word3 = 1U,
            };
            const auto result = workspace->getOrCreatePersistentPublication(
                kROCmRoPEInvariantPublicationDomain,
                key,
                llaminar2::rope::kInvariantPublicationSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    const std::size_t payload_bytes =
                        static_cast<std::size_t>(rotary_dim / 2) *
                        sizeof(float);
                    auto *device_values = static_cast<float *>(
                        workspace->getPersistentSlotBuffer(
                            llaminar2::RoPEWorkspaceBuffers::INV_FREQ,
                            llaminar2::rope::kInvariantPublicationSlots,
                            slot,
                            payload_bytes));
                    if (!device_values)
                        return {};

                    hipEvent_t ready_event = nullptr;
                    hipError_t err = hipEventCreateWithFlags(
                        &ready_event,
                        hipEventDisableTiming);
                    if (err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[" << context
                                      << "] Failed to create RoPE readiness event: "
                                      << hipGetErrorString(err));
                        return {};
                    }

                    auto publication = std::shared_ptr<
                        llaminar2::rope::RoPEInvariantWorkspacePublication>(
                        new llaminar2::rope::RoPEInvariantWorkspacePublication{
                            .ready_event = static_cast<void *>(ready_event),
                            .device_values = device_values,
                            .rotary_dim = rotary_dim,
                            .theta_bits = theta_bits,
                            .workspace_slot = slot,
                        },
                        [](llaminar2::rope::RoPEInvariantWorkspacePublication *value)
                        {
                            if (value && value->ready_event)
                            {
                                (void)hipEventDestroy(
                                    static_cast<hipEvent_t>(
                                        value->ready_event));
                            }
                            delete value;
                        });

                    if (!hipOps_rope_populate_inv_freq(
                            device_values,
                            rotary_dim,
                            rope_theta,
                            device_idx,
                            stream))
                    {
                        LOG_ERROR("[" << context
                                      << "] Device RoPE frequency producer failed");
                        return {};
                    }
                    err = hipEventRecord(ready_event, stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[" << context
                                      << "] Cannot publish RoPE readiness: "
                                      << hipGetErrorString(err));
                        // Device work was submitted; rollback without its event
                        // would expose an unordered slot to a later publisher.
                        std::terminate();
                    }
                    return publication;
                });
            if (!result)
            {
                LOG_ERROR("[" << context
                              << "] Failed to publish immutable RoPE table");
                return false;
            }
            current = std::static_pointer_cast<
                llaminar2::rope::RoPEInvariantWorkspacePublication>(
                result.publication);
            adoption_stream = nullptr;
        }

        const hipError_t wait_err = hipStreamWaitEvent(
            stream,
            static_cast<hipEvent_t>(current->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[" << context
                          << "] Failed to adopt immutable RoPE publication: "
                          << hipGetErrorString(wait_err));
            return false;
        }
        adoption_stream = static_cast<void *>(stream);
        return true;
    }

    /**
     * @brief Record the production ROCm RoPE route for an MTP row group.
     *
     * Serial M=1 calls are intentionally omitted. A grouped byte proof must
     * observe either the contiguous device-scalar route or the explicit
     * device-row route, plus one native fused Q/K launch.
     */
    void recordROCmGroupedRoPECall(
        const char *tensor_format,
        int verifier_rows,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device,
        const char *position_route)
    {
        if (verifier_rows < 2)
            return;

        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "rocm_rope_grouped_verifier_rows_calls",
            1.0,
            "verifier",
            llaminar2::DeviceId::rocm(device).to_string(),
            {{"tensor_format", tensor_format},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"q_heads", std::to_string(n_heads)},
             {"kv_heads", std::to_string(n_kv_heads)},
             {"head_dim", std::to_string(head_dim)},
             {"rotary_dim", std::to_string(rotary_dim)},
             {"position_route", position_route},
             {"capture_mode", llaminar2::isGraphCaptureActive() ? "graph_capture" : "direct"},
             {"invocation_policy", "single_grouped_launch"}});
    }
} // namespace

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // ROCmRoPEKernelT<FP32> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::prepareInvariantDeviceState(
            int rotary_dim,
            float rope_theta)
        {
            const int dev = device_idx_ >= 0
                                ? device_idx_
                                : (workspace_ ? workspace_->device().rocm_ordinal() : -1);
            return prepareHIPRoPEInvariantPublication(
                workspace_,
                static_cast<hipStream_t>(gpu_stream_),
                dev,
                rotary_dim,
                rope_theta,
                inv_freq_publication_,
                inv_freq_adoption_stream_,
                "ROCmRoPEKernelT<FP32>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::setDynamicPosOffset(int pos_offset)
        {
            if (!publishHIPRoPEDeviceParams(
                    workspace_, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    device_idx_, "ROCmRoPEKernelT<FP32>"))
            {
                throw std::runtime_error(
                    "ROCmRoPEKernelT<FP32> failed to publish device-owned dynamic position");
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::setDynamicPositionIds(
            const int *position_ids,
            int seq_len)
        {
            dynamic_position_ids_device_ptr_ = nullptr;
            uploadHIPRoPEPositionIds(
                workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                "ROCmRoPEKernelT<FP32>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::setDynamicDevicePositionIds(
            const void *position_ids_device,
            int seq_len)
        {
            dynamic_position_ids_device_valid_ = false;
            dynamic_position_ids_seq_len_ = 0;
            dynamic_position_ids_device_ptr_ = nullptr;

            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Cannot bind device position_ids on a null/default HIP stream");
                return;
            }
            if (!position_ids_device || seq_len <= 0)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Cannot bind empty device position_ids");
                return;
            }

            dynamic_position_ids_device_ptr_ =
                static_cast<const int *>(position_ids_device);
            dynamic_position_ids_seq_len_ = seq_len;
            dynamic_position_ids_device_valid_ = true;
        }

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::FP32>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            // Position IDs buffer - m is max sequence length
            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                RoPEWorkspaceBuffers::POSITION_IDS,
                pos_ids_bytes,
                256, // HIP alignment
                true // Required
            });
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({
                RoPEWorkspaceBuffers::INV_FREQ,
                rope::kInvariantPublicationSlots *
                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                256, // HIP alignment
                true, // Required
                WorkspaceExecutionRegime::Any,
                WorkspaceContentLifetime::SerialGraphFamily
            });
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // The graph is rebuilt every forward() call with new stage objects,
            // but the workspace pointer stays the same. Resetting unconditionally
            // forces a synchronous hipMemcpy every RoPE call, which blocks until
            // all pending GPU work (GEMM) completes — causing ~4ms overhead per call.
            if (workspace_ != ws)
            {
                inv_freq_publication_.reset();
                inv_freq_adoption_stream_ = nullptr;
                dynamic_pos_device_valid_ = false;
                dynamic_position_ids_device_valid_ = false;
                dynamic_position_ids_device_ptr_ = nullptr;
            }
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::FP32>::getWorkspace() const
        {
            return workspace_;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP32>"))
            {
                return false;
            }

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<FP32>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            return hipOps_rope_fp32_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Q must be FP32Tensor");
                return false;
            }

            auto *q_fp32 = dynamic_cast<FP32Tensor *>(Q);
            FP32Tensor *k_fp32 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] K must be FP32Tensor");
                    return false;
                }
                k_fp32 = dynamic_cast<FP32Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            HipDeviceGuard::setDevice(dev);

            // Get GPU pointers - coherence is handled by DeviceGraphExecutor
            float *d_Q = static_cast<float *>(q_fp32->gpu_data_ptr());
            float *d_K = k_fp32 ? static_cast<float *>(k_fp32->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP32>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
            auto complete_launch = [&](bool ok, const char *position_route)
            {
                if (ok)
                {
                    recordROCmGroupedRoPECall(
                        "FP32", seq_len, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, position_route);
                }
                return ok;
            };

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            const bool has_device_position_ids = dynamic_position_ids_device_ptr_ != nullptr;
            const bool force_device_positions =
                (gpu_stream_ != nullptr && (position_ids != nullptr || has_device_position_ids));

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy.
            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos
            // would be frozen in the captured graph. Fall through to contiguous path which
            // uses pre-uploaded device_params for graph replay.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return complete_launch(
                    hipOps_rope_fp32_decode(
                        d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, gpu_stream_),
                    "decode_scalar");
            }

            // CONTIGUOUS DETECTION: Check if positions are sequential (pos_offset, pos_offset+1, ...)
            // If so, use the zero-copy contiguous kernel which computes positions on-the-fly on GPU.
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                // Explicit graph rows are mutable workspace data.  Keep them
                // on the row-buffer path even if today's values are contiguous.
                if (!force_device_positions && !is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture/replay, use the pre-uploaded device params buffer
                    // so pos_offset can change without recording H2D nodes.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[ROCmRoPEKernelT<FP32>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[ROCmRoPEKernelT<FP32>] RoPE device params were not ready before HIP graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    return complete_launch(
                        hipOps_rope_fp32_contiguous(
                            d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                            n_heads, n_kv_heads, head_dim, eff_rotary,
                            dev, gpu_stream_, d_params),
                        "contiguous_device_scalar");
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array (rare: batched with padding/reordering)
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<FP32>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            // Call the optimized kernel
            return complete_launch(
                hipOps_rope_fp32_v2(
                    d_Q, d_K, d_inv_freq, d_position_ids, seq_len,
                    n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_),
                "explicit_device_rows");
        }

        // =========================================================================
        // ROCmRoPEKernelT<BF16> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::prepareInvariantDeviceState(
            int rotary_dim,
            float rope_theta)
        {
            const int dev = device_idx_ >= 0
                                ? device_idx_
                                : (workspace_ ? workspace_->device().rocm_ordinal() : -1);
            return prepareHIPRoPEInvariantPublication(
                workspace_,
                static_cast<hipStream_t>(gpu_stream_),
                dev,
                rotary_dim,
                rope_theta,
                inv_freq_publication_,
                inv_freq_adoption_stream_,
                "ROCmRoPEKernelT<BF16>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::setDynamicPosOffset(int pos_offset)
        {
            if (!publishHIPRoPEDeviceParams(
                    workspace_, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    device_idx_, "ROCmRoPEKernelT<BF16>"))
            {
                throw std::runtime_error(
                    "ROCmRoPEKernelT<BF16> failed to publish device-owned dynamic position");
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::setDynamicPositionIds(
            const int *position_ids,
            int seq_len)
        {
            dynamic_position_ids_device_ptr_ = nullptr;
            uploadHIPRoPEPositionIds(
                workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                "ROCmRoPEKernelT<BF16>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::setDynamicDevicePositionIds(
            const void *position_ids_device,
            int seq_len)
        {
            dynamic_position_ids_device_valid_ = false;
            dynamic_position_ids_seq_len_ = 0;
            dynamic_position_ids_device_ptr_ = nullptr;

            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Cannot bind device position_ids on a null/default HIP stream");
                return;
            }
            if (!position_ids_device || seq_len <= 0)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Cannot bind empty device position_ids");
                return;
            }

            dynamic_position_ids_device_ptr_ =
                static_cast<const int *>(position_ids_device);
            dynamic_position_ids_seq_len_ = seq_len;
            dynamic_position_ids_device_valid_ = true;
        }

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::BF16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({RoPEWorkspaceBuffers::POSITION_IDS,
                                    pos_ids_bytes,
                                    256,
                                    true});
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({RoPEWorkspaceBuffers::INV_FREQ,
                                    rope::kInvariantPublicationSlots *
                                        static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                                    256,
                                    true,
                                    WorkspaceExecutionRegime::Any,
                                    WorkspaceContentLifetime::SerialGraphFamily});
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // See FP32 bindWorkspace() for detailed explanation.
            if (workspace_ != ws)
            {
                inv_freq_publication_.reset();
                inv_freq_adoption_stream_ = nullptr;
                dynamic_pos_device_valid_ = false;
                dynamic_position_ids_device_valid_ = false;
                dynamic_position_ids_device_ptr_ = nullptr;
            }
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::BF16>::getWorkspace() const
        {
            return workspace_;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<BF16>"))
            {
                return false;
            }

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<BF16>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            return hipOps_rope_bf16_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::BF16)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Q must be BF16Tensor");
                return false;
            }

            auto *q_bf16 = dynamic_cast<BF16Tensor *>(Q);
            BF16Tensor *k_bf16 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::BF16)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] K must be BF16Tensor");
                    return false;
                }
                k_bf16 = dynamic_cast<BF16Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            HipDeviceGuard::setDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_bf16->gpu_data_ptr());
            uint16_t *d_K = k_bf16 ? static_cast<uint16_t *>(k_bf16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<BF16>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
            auto complete_launch = [&](bool ok, const char *position_route)
            {
                if (ok)
                {
                    recordROCmGroupedRoPECall(
                        "BF16", seq_len, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, position_route);
                }
                return ok;
            };

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            const bool has_device_position_ids = dynamic_position_ids_device_ptr_ != nullptr;
            const bool force_device_positions =
                (gpu_stream_ != nullptr && (position_ids != nullptr || has_device_position_ids));

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return complete_launch(
                    hipOps_rope_bf16_decode(
                        d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, gpu_stream_),
                    "decode_scalar");
            }

            // CONTIGUOUS DETECTION: Avoid synchronous hipMemcpy pipeline drain
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                // Explicit graph rows are mutable workspace data.  Keep them
                // on the row-buffer path even if today's values are contiguous.
                if (!force_device_positions && !is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture/replay, use the pre-uploaded device params buffer
                    // so pos_offset can change without recording H2D nodes.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[ROCmRoPEKernelT<BF16>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[ROCmRoPEKernelT<BF16>] RoPE device params were not ready before HIP graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    return complete_launch(
                        hipOps_rope_bf16_contiguous(
                            d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                            n_heads, n_kv_heads, head_dim, eff_rotary,
                            dev, gpu_stream_, d_params),
                        "contiguous_device_scalar");
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            // Workspace is already verified above, just get the buffer
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<BF16>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            return complete_launch(
                hipOps_rope_bf16_v2(
                    d_Q, d_K, d_inv_freq, d_position_ids, seq_len,
                    n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_),
                "explicit_device_rows");
        }

        // =========================================================================
        // ROCmRoPEKernelT<FP16> Implementation
        // =========================================================================

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::prepareInvariantDeviceState(
            int rotary_dim,
            float rope_theta)
        {
            const int dev = device_idx_ >= 0
                                ? device_idx_
                                : (workspace_ ? workspace_->device().rocm_ordinal() : -1);
            return prepareHIPRoPEInvariantPublication(
                workspace_,
                static_cast<hipStream_t>(gpu_stream_),
                dev,
                rotary_dim,
                rope_theta,
                inv_freq_publication_,
                inv_freq_adoption_stream_,
                "ROCmRoPEKernelT<FP16>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::setDynamicPosOffset(int pos_offset)
        {
            if (!publishHIPRoPEDeviceParams(
                    workspace_, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    device_idx_, "ROCmRoPEKernelT<FP16>"))
            {
                throw std::runtime_error(
                    "ROCmRoPEKernelT<FP16> failed to publish device-owned dynamic position");
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::setDynamicPositionIds(
            const int *position_ids,
            int seq_len)
        {
            dynamic_position_ids_device_ptr_ = nullptr;
            uploadHIPRoPEPositionIds(
                workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                "ROCmRoPEKernelT<FP16>");
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::setDynamicDevicePositionIds(
            const void *position_ids_device,
            int seq_len)
        {
            dynamic_position_ids_device_valid_ = false;
            dynamic_position_ids_seq_len_ = 0;
            dynamic_position_ids_device_ptr_ = nullptr;

            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Cannot bind device position_ids on a null/default HIP stream");
                return;
            }
            if (!position_ids_device || seq_len <= 0)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Cannot bind empty device position_ids");
                return;
            }

            dynamic_position_ids_device_ptr_ =
                static_cast<const int *>(position_ids_device);
            dynamic_position_ids_seq_len_ = seq_len;
            dynamic_position_ids_device_valid_ = true;
        }

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::FP16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({RoPEWorkspaceBuffers::POSITION_IDS,
                                    pos_ids_bytes,
                                    256,
                                    true});
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({RoPEWorkspaceBuffers::INV_FREQ,
                                    rope::kInvariantPublicationSlots *
                                        static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                                    256,
                                    true,
                                    WorkspaceExecutionRegime::Any,
                                    WorkspaceContentLifetime::SerialGraphFamily});
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // See FP32 bindWorkspace() for detailed explanation.
            if (workspace_ != ws)
            {
                inv_freq_publication_.reset();
                inv_freq_adoption_stream_ = nullptr;
                dynamic_pos_device_valid_ = false;
                dynamic_position_ids_device_valid_ = false;
                dynamic_position_ids_device_ptr_ = nullptr;
            }
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::FP16>::getWorkspace() const
        {
            return workspace_;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP16>"))
            {
                return false;
            }

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<FP16>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            return hipOps_rope_fp16_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::FP16)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Q must be FP16Tensor");
                return false;
            }

            auto *q_fp16 = dynamic_cast<FP16Tensor *>(Q);
            FP16Tensor *k_fp16 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::FP16)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] K must be FP16Tensor");
                    return false;
                }
                k_fp16 = dynamic_cast<FP16Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            HipDeviceGuard::setDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_fp16->gpu_data_ptr());
            uint16_t *d_K = k_fp16 ? static_cast<uint16_t *>(k_fp16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP16>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
            auto complete_launch = [&](bool ok, const char *position_route)
            {
                if (ok)
                {
                    recordROCmGroupedRoPECall(
                        "FP16", seq_len, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, position_route);
                }
                return ok;
            };

            const std::uint32_t theta_bits = ropeThetaBits(rope_theta);
            if (!inv_freq_publication_ ||
                inv_freq_publication_->rotary_dim != eff_rotary ||
                inv_freq_publication_->theta_bits != theta_bits)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] Immutable RoPE state was not prepared before graph capture");
                    return false;
                }
                if (!prepareInvariantDeviceState(eff_rotary, rope_theta))
                    return false;
            }
            float *d_inv_freq = inv_freq_publication_->device_values;

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            const bool has_device_position_ids = dynamic_position_ids_device_ptr_ != nullptr;
            const bool force_device_positions =
                (gpu_stream_ != nullptr && (position_ids != nullptr || has_device_position_ids));

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return complete_launch(
                    hipOps_rope_fp16_decode(
                        d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads,
                        head_dim, eff_rotary, dev, gpu_stream_),
                    "decode_scalar");
            }

            // CONTIGUOUS DETECTION: Avoid synchronous hipMemcpy pipeline drain
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                // Explicit graph rows are mutable workspace data.  Keep them
                // on the row-buffer path even if today's values are contiguous.
                if (!force_device_positions && !is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture/replay, use the pre-uploaded device params buffer
                    // so pos_offset can change without recording H2D nodes.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[ROCmRoPEKernelT<FP16>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[ROCmRoPEKernelT<FP16>] RoPE device params were not ready before HIP graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    return complete_launch(
                        hipOps_rope_fp16_contiguous(
                            d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                            n_heads, n_kv_heads, head_dim, eff_rotary,
                            dev, gpu_stream_, d_params),
                        "contiguous_device_scalar");
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            // Workspace is already verified above, just get the buffer
            const int *d_position_ids = dynamic_position_ids_device_ptr_;
            if (!d_position_ids)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            }
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            if (!dynamic_position_ids_device_valid_ ||
                dynamic_position_ids_seq_len_ < seq_len)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] RoPE position_ids were not ready before HIP graph capture"
                              << " requested_seq_len=" << seq_len
                              << " prepared_seq_len=" << dynamic_position_ids_seq_len_
                              << " device_valid=" << dynamic_position_ids_device_valid_);
                    return false;
                }
                uploadHIPRoPEPositionIds(
                    workspace_, position_ids, seq_len, static_cast<hipStream_t>(gpu_stream_),
                    dynamic_position_ids_device_valid_, dynamic_position_ids_seq_len_,
                    "ROCmRoPEKernelT<FP16>");
                if (!dynamic_position_ids_device_valid_)
                    return false;
            }

            return complete_launch(
                hipOps_rope_fp16_v2(
                    d_Q, d_K, d_inv_freq, d_position_ids, seq_len,
                    n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_),
                "explicit_device_rows");
        }

    } // namespace rocm
} // namespace llaminar2
