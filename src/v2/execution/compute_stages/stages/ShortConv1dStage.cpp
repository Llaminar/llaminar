/**
 * @file ShortConv1dStage.cpp
 * @brief Implementation of causal depthwise conv1d + SiLU for GDN layers
 *
 * Delegates to ITensorShortConvolution kernel for the actual computation.
 * Stage handles tensor extraction, null checks, and buffer contract management.
 *
 * GPU path: Uses ensureOnDevice() / allocateOnDevice() / gpu_data_ptr() to
 * keep data on-device. No H2D/D2H copies in the hot path.
 *
 * CPU path: Uses data() / mutable_data() host pointers.
 */

#include "ShortConv1dStage.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/ops/CUDARowSelectKernels.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ops/ROCmRowSelectKernels.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

namespace llaminar2
{
    namespace
    {
        std::atomic<uint32_t> g_shortconv_workspace_slice_counter{0};
    }

    struct ShortConv1dStage::GpuEffectiveSeqLenState
    {
        DeviceId device = DeviceId::invalid();   ///< Device that owns device_effective_seq_len.
        int *host_effective_seq_len = nullptr;   ///< Pinned host scalar uploaded before capture/replay.
        int *device_effective_seq_len = nullptr; ///< Device scalar read by the short-conv kernel.
        bool device_value_uploaded = false;       ///< True once the current host scalar is resident.
    };

    ShortConv1dStage::ShortConv1dStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          workspace_slice_id_(g_shortconv_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
    {
    }

    ShortConv1dStage::~ShortConv1dStage()
    {
        releaseGpuEffectiveSeqLenState();
    }

    WorkspaceRequirements ShortConv1dStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)n;
        (void)k;

        WorkspaceRequirements reqs;
        if (params_.channels <= 0)
            return reqs;

        const int max_seq_len = std::max(1, m > 0 ? m : params_.seq_len);
        const int speculative_slot_rows = requestedSpeculativeStateSlotRows();
        if (speculative_slot_rows > 0 && params_.kernel_size > 1)
        {
            const int rows = std::min(speculative_slot_rows, max_seq_len);
            const size_t state_floats =
                static_cast<size_t>(params_.channels) *
                static_cast<size_t>(params_.kernel_size - 1);
            reqs.buffers.push_back({speculativeStateSlotsBufferName(),
                                    static_cast<size_t>(rows) * state_floats * sizeof(float),
                                    256,
                                    true});
            if (params_.device_id.is_gpu())
            {
                reqs.buffers.push_back({speculativeStateWorkBufferName(),
                                        state_floats * sizeof(float),
                                        256,
                                        true});
            }
        }

        if (!params_.device_id.is_gpu())
            return reqs;

        if (max_seq_len > 1)
            reqs.buffers.push_back({effectiveSeqLenScalarBufferName(), sizeof(int), alignof(int), true});

        const size_t bytes = static_cast<size_t>(max_seq_len) *
                             static_cast<size_t>(params_.channels) * sizeof(float);
        reqs.buffers.push_back({WS_INPLACE_PREFILL_SCRATCH, bytes, 256, true});
        return reqs;
    }

    void ShortConv1dStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (gpu_effective_seq_len_state_)
        {
            gpu_effective_seq_len_state_->device_effective_seq_len = nullptr;
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        }
        bindKernelWorkspace();
    }

    void ShortConv1dStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
        bindKernelWorkspace();
    }

    void ShortConv1dStage::bindKernelWorkspace()
    {
        if (!params_.kernel)
            return;

        float *scratch = nullptr;
        int scratch_floats = 0;
        if (bound_workspace_ && bound_workspace_->hasBuffer(WS_INPLACE_PREFILL_SCRATCH))
        {
            scratch = static_cast<float *>(bound_workspace_->getBuffer(WS_INPLACE_PREFILL_SCRATCH));
            const size_t available_floats =
                bound_workspace_->getBufferSize(WS_INPLACE_PREFILL_SCRATCH) / sizeof(float);
            scratch_floats = static_cast<int>(std::min<size_t>(
                available_floats,
                static_cast<size_t>(std::numeric_limits<int>::max())));
        }

        params_.kernel->bindScratchWorkspace(scratch, scratch_floats);

        const int capture_state_size = params_.channels * std::max(0, params_.kernel_size - 1);
        verifier_capture_workspace_bound_ = false;
        speculative_state_work_bound_ = false;
        verifier_capture_rows_bound_ = 0;
        verifier_capture_state_size_bound_ = capture_state_size;

        const int speculative_slot_rows = requestedSpeculativeStateSlotRows();
        if (speculative_slot_rows <= 0)
        {
            // GDN kernels are owned by the hybrid KV cache and reused across
            // verifier and normal decode graphs. A normal decode graph must
            // actively clear verifier workspaces that may have been bound by a
            // previous all-position verifier graph; otherwise the shared
            // kernel keeps running against speculative state scratch and stops
            // publishing real recurrent state.
            params_.kernel->bindVerifierStateCaptureWorkspace(
                nullptr,
                0,
                capture_state_size);
            params_.kernel->bindSpeculativeStateWorkspace(
                nullptr,
                capture_state_size);
            return;
        }

        float *capture = nullptr;
        int capture_rows = 0;
        if (bound_workspace_ && bound_workspace_->hasBuffer(speculativeStateSlotsBufferName()))
        {
            const std::string capture_name = speculativeStateSlotsBufferName();
            capture = static_cast<float *>(bound_workspace_->getBuffer(capture_name));
            const size_t available_floats =
                bound_workspace_->getBufferSize(capture_name) / sizeof(float);
            if (capture_state_size > 0)
            {
                capture_rows = static_cast<int>(std::min<size_t>(
                    static_cast<size_t>(speculative_slot_rows),
                    available_floats / static_cast<size_t>(capture_state_size)));
            }
        }
        verifier_capture_workspace_bound_ =
            capture != nullptr && capture_rows > 0 && capture_state_size > 0;
        verifier_capture_rows_bound_ = capture_rows;
        params_.kernel->bindVerifierStateCaptureWorkspace(
            capture,
            capture_rows,
            capture_state_size);

        float *speculative_work = nullptr;
        if (bound_workspace_ && bound_workspace_->hasBuffer(speculativeStateWorkBufferName()))
        {
            const std::string work_name = speculativeStateWorkBufferName();
            const size_t available_floats =
                bound_workspace_->getBufferSize(work_name) / sizeof(float);
            if (available_floats >= static_cast<size_t>(std::max(0, capture_state_size)))
            {
                speculative_work = static_cast<float *>(bound_workspace_->getBuffer(work_name));
            }
        }
        speculative_state_work_bound_ =
            speculative_work != nullptr || !params_.device_id.is_gpu();
        params_.kernel->bindSpeculativeStateWorkspace(
            speculative_work,
            capture_state_size);
    }

    int ShortConv1dStage::effectivePrefillSeqLen() const
    {
        if (params_.seq_len <= 0)
            return 0;
        if (!prefill_replay_params_set_ || prefill_effective_seq_len_ <= 0)
            return params_.seq_len;
        return std::clamp(prefill_effective_seq_len_, 1, params_.seq_len);
    }

    bool ShortConv1dStage::shouldUseRealLengthContract() const
    {
        return params_.seq_len > 1 &&
               prefill_replay_params_set_ &&
               prefill_bucket_seq_len_ == params_.seq_len &&
               prefill_effective_seq_len_ > 0 &&
               prefill_effective_seq_len_ < params_.seq_len &&
               params_.kernel &&
               params_.kernel->supportsPaddedPrefillRealLength();
    }

    std::string ShortConv1dStage::workspaceStableId() const
    {
        if (params_.layer_idx >= 0)
            return std::string("layer") + std::to_string(params_.layer_idx);
        return std::string("slice") + std::to_string(workspace_slice_id_);
    }

    std::string ShortConv1dStage::effectiveSeqLenScalarBufferName() const
    {
        return std::string(WS_EFFECTIVE_SEQ_LEN_SCALAR) + "_" + workspaceStableId();
    }

    std::string ShortConv1dStage::speculativeStateSlotsBufferName() const
    {
        return std::string(WS_SPECULATIVE_STATE_SLOTS) + "_" + workspaceStableId();
    }

    std::string ShortConv1dStage::speculativeStateWorkBufferName() const
    {
        return std::string(WS_SPECULATIVE_STATE_WORK) + "_" + workspaceStableId();
    }

    int ShortConv1dStage::requestedSpeculativeStateSlotRows() const
    {
        return std::max(
            params_.speculative_state_slot_rows,
            params_.verifier_state_capture_rows);
    }

    void ShortConv1dStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        prefill_replay_params_set_ = true;
        prefill_bucket_seq_len_ = replay.bucket_seq_len > 0 ? replay.bucket_seq_len : params_.seq_len;
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        prefill_effective_seq_len_ = std::clamp(real_seq_len, 1, std::max(1, params_.seq_len));
        refreshPinnedEffectiveSeqLen();
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        if (params_.device_id.is_gpu() && gpuStream() && bound_workspace_)
            (void)(ensureGpuEffectiveSeqLenStateInitialized() && uploadGpuEffectiveSeqLen());
    }

    bool ShortConv1dStage::supportsPaddedPrefillRealLengthContract() const
    {
        return params_.kernel && params_.kernel->supportsPaddedPrefillRealLength();
    }

    bool ShortConv1dStage::hasVerifierStateCapture() const
    {
        return params_.kernel &&
               verifierStateCaptureWorkspaceRequired() &&
               params_.conv_state != nullptr;
    }

    bool ShortConv1dStage::verifierStateCaptureWorkspaceRequired() const
    {
        return requestedSpeculativeStateSlotRows() > 0 &&
               params_.kernel_size > 1;
    }

    bool ShortConv1dStage::ensureVerifierStateCaptureWorkspaceBound() const
    {
        if (!verifierStateCaptureWorkspaceRequired())
            return true;
        if (verifier_capture_workspace_bound_ && speculative_state_work_bound_)
            return true;

        LOG_ERROR("[ShortConv1dStage] Missing required verifier state capture workspace '"
                  << speculativeStateSlotsBufferName()
                  << "' or speculative state workspace '"
                  << speculativeStateWorkBufferName()
                  << "' (requested_rows=" << requestedSpeculativeStateSlotRows()
                  << ", bound_rows=" << verifier_capture_rows_bound_
                  << ", speculative_work_bound=" << speculative_state_work_bound_
                  << ", state_size=" << verifier_capture_state_size_bound_
                  << ", layer=" << params_.layer_idx
                  << ", device=" << params_.device_id.toString() << ")");
        return false;
    }

    bool ShortConv1dStage::restoreVerifierStateCaptureRow(int row, void *stream)
    {
        if (!hasVerifierStateCapture())
            return false;
        return params_.kernel->restoreVerifierStateCaptureRow(
            params_.conv_state,
            row,
            stream ? stream : gpuStream());
    }

    bool ShortConv1dStage::publishAcceptedSpeculativeStateFromDeviceMetadata(
        const int32_t *device_accepted_state_slot_indices,
        int request_index,
        void *stream)
    {
        if (!hasVerifierStateCapture() || !params_.kernel ||
            !device_accepted_state_slot_indices ||
            request_index < 0 || !stream)
        {
            return false;
        }
        return params_.kernel->publishAcceptedSpeculativeStateFromDeviceMetadata(
            params_.conv_state,
            device_accepted_state_slot_indices,
            request_index,
            stream);
    }

    bool ShortConv1dStage::ensureGpuEffectiveSeqLenStateInitialized()
    {
        const std::string scalar_buffer = effectiveSeqLenScalarBufferName();
        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(scalar_buffer) ||
            bound_workspace_->getBufferSize(scalar_buffer) < sizeof(int))
        {
            LOG_ERROR("[ShortConv1dStage] Missing required graph workspace buffer '"
                      << scalar_buffer << "' for effective sequence length on "
                      << params_.device_id.toString());
            return false;
        }

        auto *device_effective_seq_len =
            static_cast<int *>(bound_workspace_->getBuffer(scalar_buffer));
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ShortConv1dStage] Graph workspace buffer '"
                      << scalar_buffer << "' resolved to null on "
                      << params_.device_id.toString());
            return false;
        }

        if (gpu_effective_seq_len_state_)
        {
            gpu_effective_seq_len_state_->device_effective_seq_len = device_effective_seq_len;
            if (!device_effective_seq_len)
                gpu_effective_seq_len_state_->device_value_uploaded = false;
            return true;
        }

        auto state = std::make_unique<GpuEffectiveSeqLenState>();
        state->device = params_.device_id;
        state->device_effective_seq_len = device_effective_seq_len;

        bool allocated = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            allocated = cuda::allocateRowSelectHostParam(
                params_.device_id.cuda_ordinal(),
                &state->host_effective_seq_len);
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            allocated = rocm::allocateRowSelectHostParam(
                params_.device_id.rocm_ordinal(),
                &state->host_effective_seq_len);
#endif
        }

        if (!allocated || !state->host_effective_seq_len || !state->device_effective_seq_len)
        {
            LOG_ERROR("[ShortConv1dStage] Failed to allocate pinned effective length replay scalar on "
                      << params_.device_id.toString());
            return false;
        }

        gpu_effective_seq_len_state_ = std::move(state);
        refreshPinnedEffectiveSeqLen();
        return true;
    }

    void ShortConv1dStage::refreshPinnedEffectiveSeqLen()
    {
        if (gpu_effective_seq_len_state_ && gpu_effective_seq_len_state_->host_effective_seq_len)
            *gpu_effective_seq_len_state_->host_effective_seq_len = effectivePrefillSeqLen();
    }

    bool ShortConv1dStage::uploadGpuEffectiveSeqLen()
    {
        if (!gpu_effective_seq_len_state_)
            return false;
        refreshPinnedEffectiveSeqLen();

        if (isGraphCaptureActive())
        {
            if (!gpu_effective_seq_len_state_->device_value_uploaded)
            {
                LOG_ERROR("[ShortConv1dStage] Effective sequence length scalar was not uploaded before graph capture");
                return false;
            }
            return true;
        }

        bool uploaded = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            uploaded = cuda::uploadRowSelectParam(
                gpu_effective_seq_len_state_->device_effective_seq_len,
                gpu_effective_seq_len_state_->host_effective_seq_len,
                gpuStream());
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            uploaded = rocm::uploadRowSelectParam(
                gpu_effective_seq_len_state_->device_effective_seq_len,
                gpu_effective_seq_len_state_->host_effective_seq_len,
                gpuStream());
#endif
        }
        gpu_effective_seq_len_state_->device_value_uploaded = uploaded;
        return uploaded;
    }

    void ShortConv1dStage::releaseGpuEffectiveSeqLenState()
    {
        if (!gpu_effective_seq_len_state_)
            return;

        if (gpu_effective_seq_len_state_->device.is_cuda())
        {
#ifdef HAVE_CUDA
            cuda::freeRowSelectHostParam(
                gpu_effective_seq_len_state_->device.cuda_ordinal(),
                gpu_effective_seq_len_state_->host_effective_seq_len);
#endif
        }
        else if (gpu_effective_seq_len_state_->device.is_rocm())
        {
#ifdef HAVE_ROCM
            rocm::freeRowSelectHostParam(
                gpu_effective_seq_len_state_->device.rocm_ordinal(),
                gpu_effective_seq_len_state_->host_effective_seq_len);
#endif
        }

        gpu_effective_seq_len_state_.reset();
    }

    bool ShortConv1dStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "ShortConv1dStage"))
            return false;

        if (!ensureRequiredPointers("ShortConv1dStage",
                                    {{"input", params_.input},
                                     {"output", params_.output},
                                     {"weight", params_.weight}}))
            return false;

        if (!params_.kernel)
        {
            LOG_ERROR("[ShortConv1dStage] kernel (ITensorShortConvolution) not set");
            return false;
        }

        // Bind stage stream to kernel before execution
        params_.kernel->setGPUStream(gpuStream());
        bindKernelWorkspace();
        if (!ensureVerifierStateCaptureWorkspaceBound())
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *weight_base = requireTensorBasePtr(params_.weight, "weight");

        if (!input_base || !output_base || !weight_base)
            return false;

        // =================================================================
        // GPU path: keep data on-device, pass device pointers to kernel
        // =================================================================
        if (device().is_gpu())
        {
            // Coherence is handled by the executor via bufferContract():
            //   - Arena inputs (GDN_QKV) via prepareForRead
            //   - Model weights (conv1d weight, bias) via contract.weight_tensors
            //   - Output (GDN_QKV) via prepareForWrite + markWritten
            const float *d_input = static_cast<const float *>(input_base->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_base->gpu_data_ptr());
            float *d_output = static_cast<float *>(const_cast<TensorBase *>(output_base)->gpu_data_ptr());

            const float *d_bias = nullptr;
            if (params_.bias)
            {
                auto *bias_base = dynamic_cast<const TensorBase *>(params_.bias);
                if (bias_base)
                    d_bias = static_cast<const float *>(bias_base->gpu_data_ptr());
            }

            const int effective_seq_len = effectivePrefillSeqLen();
            const bool padded_effective_len =
                params_.seq_len > 1 && prefill_replay_params_set_ && effective_seq_len < params_.seq_len;
            const bool use_real_length_contract = shouldUseRealLengthContract();
            if (padded_effective_len && !use_real_length_contract)
            {
                LOG_ERROR("[ShortConv1dStage] Padded prefill requires a backend real-length contract");
                return false;
            }

            bool ok = false;
            if (use_real_length_contract)
            {
                if (!ensureGpuEffectiveSeqLenStateInitialized() || !uploadGpuEffectiveSeqLen())
                {
                    LOG_ERROR("[ShortConv1dStage] Failed to update GPU effective length scalar");
                    return false;
                }
                ok = params_.kernel->forwardWithEffectiveSeqLen(
                    d_input, d_weight, d_bias,
                    d_output, params_.conv_state,
                    params_.seq_len, params_.channels, params_.kernel_size,
                    gpu_effective_seq_len_state_->device_effective_seq_len,
                    /*apply_silu=*/true);
            }
            else
            {
                ok = params_.kernel->forward(
                    d_input, d_weight, d_bias,
                    d_output, params_.conv_state,
                    params_.seq_len, params_.channels, params_.kernel_size,
                    /*apply_silu=*/true);
            }

            if (!ok)
            {
                LOG_ERROR("[ShortConv1dStage] GPU kernel forward() failed");
                return false;
            }

            LOG_DEBUG("[ShortConv1dStage] GPU: seq_len=" << params_.seq_len
                                                         << " channels=" << params_.channels
                                                         << " effective_seq_len=" << effective_seq_len
                                                         << " kernel=" << params_.kernel_size
                                                         << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));
            return true;
        }

        // =================================================================
        // CPU path: use host pointers
        // =================================================================
        const float *input_data = input_base->data();
        float *output_data = output_base->mutable_data();
        const float *weight_data = weight_base->data();

        const float *bias_data = nullptr;
        if (params_.bias)
        {
            auto *bias_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.bias));
            if (bias_base)
                bias_data = bias_base->data();
        }

        if (!input_data || !output_data || !weight_data)
        {
            LOG_ERROR("[ShortConv1dStage] Null data pointer");
            return false;
        }

        const int effective_seq_len = effectivePrefillSeqLen();
        const int kernel_seq_len = params_.seq_len > 1 ? effective_seq_len : params_.seq_len;
        const bool padded_effective_len =
            params_.seq_len > 1 && prefill_replay_params_set_ && effective_seq_len < params_.seq_len;
        if (padded_effective_len && !supportsPaddedPrefillRealLengthContract())
        {
            LOG_ERROR("[ShortConv1dStage] Padded CPU prefill requires a real-length contract");
            return false;
        }

        bool ok = params_.kernel->forward(
            input_data, weight_data, bias_data,
            output_data, params_.conv_state,
            kernel_seq_len, params_.channels, params_.kernel_size,
            /*apply_silu=*/true);

        if (!ok)
        {
            LOG_ERROR("[ShortConv1dStage] Kernel forward() failed");
            return false;
        }

        if (kernel_seq_len < params_.seq_len)
        {
            const size_t first_pad = static_cast<size_t>(kernel_seq_len) * static_cast<size_t>(params_.channels);
            const size_t pad_count = static_cast<size_t>(params_.seq_len - kernel_seq_len) * static_cast<size_t>(params_.channels);
            std::memset(output_data + first_pad, 0, pad_count * sizeof(float));
        }

        LOG_DEBUG("[ShortConv1dStage] Executed: seq_len=" << params_.seq_len
                                                          << " effective_seq_len=" << kernel_seq_len
                                                          << " channels=" << params_.channels
                                                          << " kernel=" << params_.kernel_size
                                                          << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));
        return true;
    }

    size_t ShortConv1dStage::estimatedFlops() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t C = static_cast<size_t>(params_.channels);
        const size_t K = static_cast<size_t>(params_.kernel_size);
        // Per output: K MAC + SiLU (4 ops)
        return S * C * (2 * K + 4);
    }

    size_t ShortConv1dStage::estimatedMemoryBytes() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t C = static_cast<size_t>(params_.channels);
        const size_t K = static_cast<size_t>(params_.kernel_size);
        return (S * C + C * K + S * C) * sizeof(float); // input + weight + output
    }

    bool ShortConv1dStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo ShortConv1dStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // The backing arena tensor is sized for the maximum sequence length;
        // snapshots need the logical rows touched by this stage execution.
        auto *out_base = dynamic_cast<const TensorBase *>(params_.output);
        if (out_base)
            info.addOutput("output", params_.output,
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.channels));

        return info;
    }

    StageBufferRequirements ShortConv1dStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract ShortConv1dStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        // Conv weights are model parameters, not arena-managed
        if (params_.weight)
            contract.addWeight(const_cast<ITensor *>(params_.weight));
        if (params_.bias)
            contract.addWeight(const_cast<ITensor *>(params_.bias));
        return contract;
    }

} // namespace llaminar2
