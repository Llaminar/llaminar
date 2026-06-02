/**
 * @file HiddenStateRowSelectStage.cpp
 * @brief Implementation of graph-capturable hidden-state row selection.
 *
 * Key algorithm: copy the dynamic last-real-token row into a stable one-row
 * scratch tensor before LM head. On GPU, the dynamic row lives in a pinned host
 * scalar whose H2D copy is captured with the row-copy kernel; replay changes the
 * selected row by changing the pinned scalar contents before graph launch.
 */

#include "HiddenStateRowSelectStage.h"

#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../tensors/Tensors.h"
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
#include <utility>

namespace llaminar2
{
    namespace
    {
        std::atomic<uint32_t> g_row_select_workspace_slice_counter{0};
    }

    struct HiddenStateRowSelectStage::GpuParamState
    {
        DeviceId device = DeviceId::invalid(); ///< Device that owns device_selected_row.
        int *host_selected_row = nullptr;      ///< Pinned host scalar captured by H2D memcpy.
        int *device_selected_row = nullptr;    ///< Device scalar read by row-select kernel.
    };

    HiddenStateRowSelectStage::HiddenStateRowSelectStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          workspace_slice_id_(g_row_select_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
    {
        selected_row_idx_ = normalizeSelectedRow(params_.selected_row_idx);
    }

    HiddenStateRowSelectStage::~HiddenStateRowSelectStage()
    {
        releaseGpuParamState();
    }

    int HiddenStateRowSelectStage::normalizeSelectedRow(int requested_row) const
    {
        if (params_.seq_len <= 0)
            return 0;
        if (requested_row < 0)
            return params_.seq_len - 1;
        return std::clamp(requested_row, 0, params_.seq_len - 1);
    }

    std::string HiddenStateRowSelectStage::selectedRowScalarBufferName() const
    {
        return std::string(WS_SELECTED_ROW_SCALAR) + "_" + std::to_string(workspace_slice_id_);
    }

    WorkspaceRequirements HiddenStateRowSelectStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;

        WorkspaceRequirements reqs;
        if (params_.device_id.is_gpu())
            reqs.buffers.push_back({selectedRowScalarBufferName(), sizeof(int), alignof(int), true});
        return reqs;
    }

    void HiddenStateRowSelectStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (gpu_state_)
            gpu_state_->device_selected_row = nullptr;
    }

    void HiddenStateRowSelectStage::unbindWorkspace()
    {
        bindWorkspace(nullptr);
    }

    void HiddenStateRowSelectStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        // real_seq_len is authoritative for bucketed prefill. If it is absent,
        // fall back to the captured bucket shape so exact legacy paths still use
        // the final bucket row.
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        selected_row_idx_ = normalizeSelectedRow(real_seq_len - 1);
        refreshPinnedSelectedRow();
    }

    bool HiddenStateRowSelectStage::validateCommon(TensorBase **input_base, TensorBase **output_base)
    {
        if (!ensureRequiredPointers("HiddenStateRowSelectStage", {
                                                                     {"input", params_.input},
                                                                     {"output", params_.output},
                                                                 }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Invalid dimensions: seq_len=" << params_.seq_len
                                                                                 << " d_model=" << params_.d_model);
            return false;
        }

        auto *resolved_input = const_cast<TensorBase *>(requireTensorBasePtr(params_.input, "input"));
        auto *resolved_output = requireTensorBasePtr(params_.output, "output");
        if (!resolved_input || !resolved_output)
            return false;

        if (resolved_input->native_type() != TensorType::FP32 || resolved_output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Only FP32 input/output tensors are supported, got input="
                      << resolved_input->dtype_name() << " output=" << resolved_output->dtype_name());
            return false;
        }

        if (resolved_input->rows() < static_cast<size_t>(params_.seq_len) ||
            resolved_input->cols() < static_cast<size_t>(params_.d_model) ||
            resolved_output->rows() < 1 ||
            resolved_output->cols() < static_cast<size_t>(params_.d_model))
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Tensor shapes do not cover requested row select: input="
                      << resolved_input->rows() << "x" << resolved_input->cols()
                      << " output=" << resolved_output->rows() << "x" << resolved_output->cols()
                      << " requested=" << params_.seq_len << "x" << params_.d_model);
            return false;
        }

        *input_base = resolved_input;
        *output_base = resolved_output;
        return true;
    }

    bool HiddenStateRowSelectStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        TensorBase *input_base = nullptr;
        TensorBase *output_base = nullptr;
        if (!validateCommon(&input_base, &output_base))
            return false;

        if (params_.device_id.is_gpu())
            return executeGPU(input_base, output_base);
        return executeCPU(input_base, output_base);
    }

    bool HiddenStateRowSelectStage::executeCPU(TensorBase *input_base, TensorBase *output_base)
    {
        const float *input_data = input_base->data();
        float *output_data = output_base->mutable_data();
        if (!input_data || !output_data)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Missing CPU data pointers");
            return false;
        }

        // One contiguous row copy keeps the CPU path simple and allocation-free.
        const size_t row_offset = static_cast<size_t>(selected_row_idx_) * static_cast<size_t>(params_.d_model);
        std::memcpy(output_data, input_data + row_offset, static_cast<size_t>(params_.d_model) * sizeof(float));
        return true;
    }

    bool HiddenStateRowSelectStage::ensureGpuParamStateInitialized()
    {
        const std::string scalar_buffer = selectedRowScalarBufferName();
        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(scalar_buffer) ||
            bound_workspace_->getBufferSize(scalar_buffer) < sizeof(int))
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Missing required graph workspace buffer '"
                      << scalar_buffer << "' for selected-row scalar on "
                      << params_.device_id.toString());
            return false;
        }

        auto *device_selected_row =
            static_cast<int *>(bound_workspace_->getBuffer(scalar_buffer));
        if (!device_selected_row)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Graph workspace buffer '"
                      << scalar_buffer << "' resolved to null on "
                      << params_.device_id.toString());
            return false;
        }

        if (gpu_state_)
        {
            gpu_state_->device_selected_row = device_selected_row;
            return true;
        }

        auto state = std::make_unique<GpuParamState>();
        state->device = params_.device_id;
        state->device_selected_row = device_selected_row;

        bool allocated = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            allocated = cuda::allocateRowSelectHostParam(
                params_.device_id.cuda_ordinal(),
                &state->host_selected_row);
#else
            allocated = false;
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            allocated = rocm::allocateRowSelectHostParam(
                params_.device_id.rocm_ordinal(),
                &state->host_selected_row);
#else
            allocated = false;
#endif
        }

        if (!allocated || !state->host_selected_row || !state->device_selected_row)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Failed to allocate pinned selected-row replay scalar on "
                      << params_.device_id.toString());
            return false;
        }

        gpu_state_ = std::move(state);
        refreshPinnedSelectedRow();
        return true;
    }

    void HiddenStateRowSelectStage::refreshPinnedSelectedRow()
    {
        if (gpu_state_ && gpu_state_->host_selected_row)
            *gpu_state_->host_selected_row = selected_row_idx_;
    }

    bool HiddenStateRowSelectStage::executeGPU(TensorBase *input_base, TensorBase *output_base)
    {
        if (!ensureGpuParamStateInitialized())
            return false;

        refreshPinnedSelectedRow();

        // Direct tests may bypass DeviceGraphExecutor coherence. These calls are
        // no-ops after warmup when the arena has already prepared the buffers.
        if (!input_base->ensureOnDevice(params_.device_id) || !output_base->ensureOnDevice(params_.device_id))
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Failed to ensure tensors on " << params_.device_id.toString());
            return false;
        }

        const auto *input_device = static_cast<const float *>(input_base->gpu_data_ptr());
        auto *output_device = static_cast<float *>(output_base->gpu_data_ptr());
        if (!input_device || !output_device)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] Missing GPU data pointers");
            return false;
        }

        bool uploaded = false;
        bool launched = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            auto stream = gpuStream();
            uploaded = cuda::uploadRowSelectParam(gpu_state_->device_selected_row, gpu_state_->host_selected_row, stream);
            launched = uploaded && cuda::launchRowSelectFP32(
                                       input_device,
                                       output_device,
                                       gpu_state_->device_selected_row,
                                       params_.seq_len,
                                       params_.d_model,
                                       stream);
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            auto stream = gpuStream();
            uploaded = rocm::uploadRowSelectParam(gpu_state_->device_selected_row, gpu_state_->host_selected_row, stream);
            launched = uploaded && rocm::launchRowSelectFP32(
                                       input_device,
                                       output_device,
                                       gpu_state_->device_selected_row,
                                       params_.seq_len,
                                       params_.d_model,
                                       stream);
#endif
        }

        if (!uploaded || !launched)
        {
            LOG_ERROR("[HiddenStateRowSelectStage] GPU row-select launch failed on " << params_.device_id.toString());
            return false;
        }

        output_base->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, params_.device_id, gpuStream());
        return true;
    }

    void HiddenStateRowSelectStage::releaseGpuParamState()
    {
        if (!gpu_state_)
            return;

        if (gpu_state_->device.is_cuda())
        {
#ifdef HAVE_CUDA
            cuda::freeRowSelectHostParam(
                gpu_state_->device.cuda_ordinal(),
                gpu_state_->host_selected_row);
#endif
        }
        else if (gpu_state_->device.is_rocm())
        {
#ifdef HAVE_ROCM
            rocm::freeRowSelectHostParam(
                gpu_state_->device.rocm_ordinal(),
                gpu_state_->host_selected_row);
#endif
        }

        gpu_state_.reset();
    }

    size_t HiddenStateRowSelectStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(params_.d_model) * sizeof(float) * 2;
    }

    bool HiddenStateRowSelectStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo HiddenStateRowSelectStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("hidden_states", params_.input, params_.seq_len, params_.d_model);
        if (params_.output)
            info.addOutput("selected_row", params_.output, 1, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("selected_row_idx", selected_row_idx_);
        return info;
    }

    StageBufferRequirements HiddenStateRowSelectStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        requirements.addInput(
            "hidden_states",
            {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model)},
            BufferTensorType::FP32);
        requirements.addOutput(
            "selected_row",
            {1, static_cast<size_t>(params_.d_model)},
            BufferTensorType::FP32);
        return requirements;
    }

    StageBufferContract HiddenStateRowSelectStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_buffer_id)
            return {};
        return StageBufferContract::build()
            .addInput(*params_.input_buffer_id, "FP32")
            .addOutput(*params_.output_buffer_id, "FP32");
    }

} // namespace llaminar2
