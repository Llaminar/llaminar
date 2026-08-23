/**
 * @file HiddenStateRowsSelectStage.cpp
 * @brief Implementation of compact hidden-state multi-row selection.
 */

#include "HiddenStateRowsSelectStage.h"

#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
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
        std::atomic<uint32_t> g_rows_select_workspace_slice_counter{0};
    }

    struct HiddenStateRowsSelectStage::GpuParamState
    {
        DeviceId device = DeviceId::invalid();
        int *host_selected_rows = nullptr;
        int *device_selected_rows = nullptr;
        bool device_value_uploaded = false;
    };

    HiddenStateRowsSelectStage::HiddenStateRowsSelectStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          workspace_slice_id_(g_rows_select_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
    {
        selected_row_count_ = params_.selected_row_count > 0
                                  ? params_.selected_row_count
                                  : static_cast<int>(params_.selected_row_indices.size());
        if (selected_row_count_ > 0 &&
            static_cast<int>(params_.selected_row_indices.size()) == selected_row_count_)
        {
            selected_rows_.reserve(static_cast<size_t>(selected_row_count_));
            for (const int row : params_.selected_row_indices)
                selected_rows_.push_back(normalizeSelectedRow(row));
        }
        else
        {
            selected_rows_ = defaultSelectedRows();
        }
    }

    HiddenStateRowsSelectStage::~HiddenStateRowsSelectStage()
    {
        releaseGpuParamState();
    }

    int HiddenStateRowsSelectStage::normalizeSelectedRow(int requested_row) const
    {
        if (params_.seq_len <= 0)
            return 0;
        if (requested_row < 0)
            return 0;
        return std::clamp(requested_row, 0, params_.seq_len - 1);
    }

    std::vector<int> HiddenStateRowsSelectStage::defaultSelectedRows() const
    {
        std::vector<int> rows;
        if (selected_row_count_ <= 0)
            return rows;
        rows.reserve(static_cast<size_t>(selected_row_count_));
        const int first_row = std::max(0, params_.seq_len - selected_row_count_);
        for (int i = 0; i < selected_row_count_; ++i)
            rows.push_back(normalizeSelectedRow(first_row + i));
        return rows;
    }

    std::string HiddenStateRowsSelectStage::selectedRowsBufferName() const
    {
        if (!params_.workspace_buffer_name.empty())
            return params_.workspace_buffer_name;
        return std::string(WS_SELECTED_ROWS_ARRAY) + "_" + std::to_string(workspace_slice_id_);
    }

    WorkspaceRequirements HiddenStateRowsSelectStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;

        WorkspaceRequirements reqs;
        if (params_.device_id.is_gpu() &&
            selected_row_count_ > 0 &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::StageOwnedIndices)
        {
            reqs.buffers.push_back({
                selectedRowsBufferName(),
                static_cast<size_t>(selected_row_count_) * sizeof(int),
                alignof(int),
                true});
        }
        return reqs;
    }

    void HiddenStateRowsSelectStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (gpu_state_)
        {
            gpu_state_->device_selected_rows = nullptr;
            gpu_state_->device_value_uploaded = false;
        }
    }

    void HiddenStateRowsSelectStage::unbindWorkspace()
    {
        bindWorkspace(nullptr);
    }

    bool HiddenStateRowsSelectStage::setSelectedRowsForReplay(const std::vector<int> &selected_row_indices)
    {
        if (static_cast<int>(selected_row_indices.size()) != selected_row_count_)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Cannot change selected-row count from "
                      << selected_row_count_ << " to " << selected_row_indices.size());
            return false;
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source !=
                DeviceRowIndexSource::StageOwnedIndices)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Cannot mutate selected rows through the stage when the device owns row metadata");
            return false;
        }

        std::vector<int> normalized;
        normalized.reserve(selected_row_indices.size());
        for (const int row : selected_row_indices)
            normalized.push_back(normalizeSelectedRow(row));

        selected_rows_ = std::move(normalized);
        refreshPinnedSelectedRows();
        if (gpu_state_)
            gpu_state_->device_value_uploaded = false;

        // Keep replay mutation side-effect free with respect to GPU workspace.
        // Cached graphs can retain a stage after the previous workspace manager
        // has been destroyed or replaced.  The executor is the only code that
        // may bind a current workspace and explicit stream; executeGPU() uploads
        // the dirty row array after that binding is in place.
        return true;
    }

    bool HiddenStateRowsSelectStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (!params_.device_id.is_gpu())
            return true;
        if (stream)
            setGPUStream(stream);
        (void)requireGPUStream();
        if (params_.device_row_index_source ==
                DeviceRowIndexSource::RequestTerminalLengths ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::FixedContiguousRange ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillKVProgress ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillTransaction)
        {
            /*
             * Request-terminal mode reads current resident lengths. Fixed-range
             * mode records an immutable D2D source offset. Neither contract has
             * launch-time host metadata to upload.
             */
            return true;
        }
        return uploadGpuSelectedRows();
    }

    bool HiddenStateRowsSelectStage::validateCommon(TensorBase **input_base, TensorBase **output_base)
    {
        if (!ensureRequiredPointers("HiddenStateRowsSelectStage", {
                                                                      {"input", params_.input},
                                                                      {"output", params_.output},
                                                                  }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || params_.d_model <= 0 || selected_row_count_ <= 0)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Invalid dimensions: seq_len=" << params_.seq_len
                                                                                  << " d_model=" << params_.d_model
                                                                                  << " selected_row_count=" << selected_row_count_);
            return false;
        }

        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::RequestTerminalLengths)
        {
            const bool static_geometry =
                params_.request_row_stride_source ==
                RequestRowStrideSource::StaticGraphGeometry;
            const bool device_geometry =
                params_.request_row_stride_source ==
                RequestRowStrideSource::ExternalDeviceScalar;
            const bool valid_request_geometry =
                params_.request_sequence_lengths_device != nullptr &&
                ((static_geometry &&
                  params_.request_row_stride > 0 &&
                  selected_row_count_ * params_.request_row_stride ==
                      params_.seq_len &&
                  params_.request_row_stride_device == nullptr) ||
                 (device_geometry &&
                  params_.request_row_stride == 0 &&
                  params_.request_row_stride_device != nullptr &&
                  selected_row_count_ <= params_.seq_len));
            if (!valid_request_geometry)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Request-terminal row selection requires "
                          << "one unambiguous resident geometry owner: seq_len="
                          << params_.seq_len
                          << " request_count=" << selected_row_count_
                          << " request_row_stride=" << params_.request_row_stride
                          << " lengths=" << params_.request_sequence_lengths_device
                          << " stride_device=" << params_.request_row_stride_device
                          << " stride_source="
                          << (static_geometry ? "static_graph" : "external_device"));
                return false;
            }
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::FixedContiguousRange)
        {
            const bool valid_fixed_range =
                params_.fixed_contiguous_row_start >= 0 &&
                params_.fixed_contiguous_row_start <=
                    params_.seq_len - selected_row_count_;
            if (!valid_fixed_range)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Fixed contiguous GPU row selection exceeds source geometry: start="
                          << params_.fixed_contiguous_row_start
                          << " count=" << selected_row_count_
                          << " seq_len=" << params_.seq_len);
                return false;
            }
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillKVProgress)
        {
            const bool valid_progress_binding =
                params_.request_sequence_lengths_device != nullptr &&
                params_.request_row_stride_source ==
                    RequestRowStrideSource::ExternalDeviceScalar &&
                params_.request_row_stride == 0 &&
                params_.request_row_stride_device != nullptr &&
                params_.main_cached_tokens_device != nullptr &&
                params_.shifted_cached_tokens_device != nullptr &&
                params_.request_index >= 0 &&
                selected_row_count_ <= params_.seq_len;
            if (!valid_progress_binding)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Shifted-prefill row selection requires canonical device KV progress and resident request geometry: request="
                          << params_.request_index
                          << " rows=" << selected_row_count_
                          << " seq_capacity=" << params_.seq_len
                          << " main_count=" << params_.main_cached_tokens_device
                          << " shifted_count=" << params_.shifted_cached_tokens_device
                          << " lengths=" << params_.request_sequence_lengths_device
                          << " stride=" << params_.request_row_stride_device);
                return false;
            }
        }
        if (params_.device_row_index_source ==
            DeviceRowIndexSource::ShiftedPrefillTransaction)
        {
            const bool valid_transaction =
                params_.device_id.is_gpu() && params_.request_count > 0 &&
                params_.seq_len % params_.request_count == 0 &&
                selected_row_count_ == params_.seq_len &&
                params_.request_sequence_lengths_device &&
                params_.request_row_stride_source ==
                    RequestRowStrideSource::ExternalDeviceScalar &&
                params_.request_row_stride == 0 &&
                params_.request_row_stride_device &&
                params_.input_token_ids_device &&
                params_.input_position_ids_device &&
                params_.shifted_token_ids_output_device &&
                params_.shifted_position_ids_output_device &&
                params_.shifted_append_lengths_output_device &&
                params_.terminal_hidden_archive &&
                params_.terminal_hidden_archive_buffer_id.has_value() &&
                params_.main_cached_tokens_by_request.size() ==
                    static_cast<size_t>(params_.request_count) &&
                params_.shifted_cached_tokens_by_request.size() ==
                    static_cast<size_t>(params_.request_count) &&
                std::all_of(
                    params_.main_cached_tokens_by_request.begin(),
                    params_.main_cached_tokens_by_request.end(),
                    [](const int32_t *ptr) { return ptr != nullptr; }) &&
                std::all_of(
                    params_.shifted_cached_tokens_by_request.begin(),
                    params_.shifted_cached_tokens_by_request.end(),
                    [](const int32_t *ptr) { return ptr != nullptr; });
            if (!valid_transaction)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Shifted-prefill transaction requires complete immutable device ownership"
                          << " requests=" << params_.request_count
                          << " seq_capacity=" << params_.seq_len
                          << " selected_rows=" << selected_row_count_
                          << " token_input=" << params_.input_token_ids_device
                          << " position_input=" << params_.input_position_ids_device
                          << " append_output=" << params_.shifted_append_lengths_output_device
                          << " archive=" << params_.terminal_hidden_archive);
                return false;
            }
            if (params_.terminal_hidden_archive->native_type() !=
                    TensorType::FP32 ||
                params_.terminal_hidden_archive->rows() <
                    static_cast<size_t>(params_.request_count) ||
                params_.terminal_hidden_archive->cols() <
                    static_cast<size_t>(params_.d_model))
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Shifted-prefill terminal archive does not cover request geometry");
                return false;
            }
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::WorkspaceBoundDeviceIndices &&
            (params_.workspace_buffer_name.empty() ||
             params_.external_device_row_indices != nullptr))
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Workspace-bound device row indices require exactly one named workspace owner");
            return false;
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source ==
                DeviceRowIndexSource::ExternalDeviceIndices &&
            (!params_.external_device_row_indices ||
             !params_.workspace_buffer_name.empty()))
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] External device row indices require exactly one explicit producer-owned device pointer");
            return false;
        }
        if (params_.device_id.is_gpu() &&
            params_.device_row_index_source !=
                DeviceRowIndexSource::ExternalDeviceIndices &&
            params_.external_device_row_indices != nullptr)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] A producer-owned row-index pointer is valid only for ExternalDeviceIndices");
            return false;
        }

        if (static_cast<int>(selected_rows_.size()) != selected_row_count_)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Selected-row vector size does not match fixed count");
            return false;
        }

        auto *resolved_input = const_cast<TensorBase *>(requireTensorBasePtr(params_.input, "input"));
        auto *resolved_output = requireTensorBasePtr(params_.output, "output");
        if (!resolved_input || !resolved_output)
            return false;

        if (resolved_input->native_type() != TensorType::FP32 || resolved_output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Only FP32 input/output tensors are supported, got input="
                      << resolved_input->dtype_name() << " output=" << resolved_output->dtype_name());
            return false;
        }

        if (resolved_input->rows() < static_cast<size_t>(params_.seq_len) ||
            resolved_input->cols() < static_cast<size_t>(params_.d_model) ||
            resolved_output->rows() < static_cast<size_t>(selected_row_count_) ||
            resolved_output->cols() < static_cast<size_t>(params_.d_model))
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Tensor shapes do not cover requested row select: input="
                      << resolved_input->rows() << "x" << resolved_input->cols()
                      << " output=" << resolved_output->rows() << "x" << resolved_output->cols()
                      << " requested_input=" << params_.seq_len << "x" << params_.d_model
                      << " requested_output=" << selected_row_count_ << "x" << params_.d_model);
            return false;
        }

        *input_base = resolved_input;
        *output_base = resolved_output;
        return true;
    }

    bool HiddenStateRowsSelectStage::execute(IDeviceContext *ctx)
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

    bool HiddenStateRowsSelectStage::executeCPU(TensorBase *input_base, TensorBase *output_base)
    {
        const float *input_data = input_base->data();
        float *output_data = output_base->mutable_data();
        if (!input_data || !output_data)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Missing CPU data pointers");
            return false;
        }

        const size_t row_bytes = static_cast<size_t>(params_.d_model) * sizeof(float);
        for (int output_row = 0; output_row < selected_row_count_; ++output_row)
        {
            const size_t source_offset =
                static_cast<size_t>(selected_rows_[static_cast<size_t>(output_row)]) *
                static_cast<size_t>(params_.d_model);
            const size_t output_offset =
                static_cast<size_t>(output_row) * static_cast<size_t>(params_.d_model);
            std::memcpy(output_data + output_offset, input_data + source_offset, row_bytes);
        }
        return true;
    }

    bool HiddenStateRowsSelectStage::ensureGpuParamStateInitialized()
    {
        if (params_.device_row_index_source ==
                DeviceRowIndexSource::RequestTerminalLengths ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillKVProgress ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillTransaction)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Request-terminal row selection must not bind a row-index workspace");
            return false;
        }

        if (params_.device_row_index_source ==
            DeviceRowIndexSource::ExternalDeviceIndices)
        {
            if (!params_.external_device_row_indices)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] External row-index owner published a null device pointer on "
                          << params_.device_id.toString());
                return false;
            }

            if (!gpu_state_)
                gpu_state_ = std::make_unique<GpuParamState>();
            gpu_state_->device = params_.device_id;
            gpu_state_->device_selected_rows =
                const_cast<int *>(params_.external_device_row_indices);
            gpu_state_->device_value_uploaded = true;
            return true;
        }

        const std::string rows_buffer = selectedRowsBufferName();
        const size_t expected_bytes = static_cast<size_t>(selected_row_count_) * sizeof(int);
        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(rows_buffer) ||
            bound_workspace_->getBufferSize(rows_buffer) < expected_bytes)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Missing required graph workspace buffer '"
                      << rows_buffer << "' for selected-row array on "
                      << params_.device_id.toString());
            return false;
        }

        auto *device_selected_rows =
            static_cast<int *>(bound_workspace_->getBuffer(rows_buffer));
        if (!device_selected_rows)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Graph workspace buffer '"
                      << rows_buffer << "' resolved to null on "
                      << params_.device_id.toString());
            return false;
        }

        if (gpu_state_)
        {
            gpu_state_->device_selected_rows = device_selected_rows;
            if (!device_selected_rows)
                gpu_state_->device_value_uploaded = false;
            return true;
        }

        auto state = std::make_unique<GpuParamState>();
        state->device = params_.device_id;
        state->device_selected_rows = device_selected_rows;
        if (params_.device_row_index_source ==
            DeviceRowIndexSource::WorkspaceBoundDeviceIndices)
        {
            // The graph-family metadata producer owns and updates this named
            // row-index array. This stage only reads the pointer resolved from
            // the same explicitly shared workspace during graph capture.
            state->device_value_uploaded = true;
            gpu_state_ = std::move(state);
            return true;
        }

        bool allocated = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            allocated = cuda::allocateRowSelectHostParams(
                params_.device_id.cuda_ordinal(),
                &state->host_selected_rows,
                selected_row_count_);
#else
            allocated = false;
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            allocated = rocm::allocateRowSelectHostParams(
                params_.device_id.rocm_ordinal(),
                &state->host_selected_rows,
                selected_row_count_);
#else
            allocated = false;
#endif
        }

        if (!allocated || !state->host_selected_rows || !state->device_selected_rows)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Failed to allocate pinned selected-row replay array on "
                      << params_.device_id.toString());
            return false;
        }

        gpu_state_ = std::move(state);
        refreshPinnedSelectedRows();
        return true;
    }

    bool HiddenStateRowsSelectStage::uploadGpuSelectedRows()
    {
        if (params_.device_row_index_source ==
                DeviceRowIndexSource::RequestTerminalLengths ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::ShiftedPrefillKVProgress)
        {
            return true;
        }
        if (!ensureGpuParamStateInitialized())
            return false;
        if (params_.device_row_index_source ==
                DeviceRowIndexSource::ExternalDeviceIndices ||
            params_.device_row_index_source ==
                DeviceRowIndexSource::WorkspaceBoundDeviceIndices)
        {
            gpu_state_->device_value_uploaded = true;
            return true;
        }
        (void)requireGPUStream();

        refreshPinnedSelectedRows();

        if (isGraphCaptureActive())
        {
            if (!gpu_state_->device_value_uploaded)
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Selected-row array was not uploaded before graph capture");
                return false;
            }
            return true;
        }

        bool uploaded = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            uploaded = cuda::uploadRowSelectParams(
                gpu_state_->device_selected_rows,
                gpu_state_->host_selected_rows,
                selected_row_count_,
                gpuStream());
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            uploaded = rocm::uploadRowSelectParams(
                gpu_state_->device_selected_rows,
                gpu_state_->host_selected_rows,
                selected_row_count_,
                gpuStream());
#endif
        }

        gpu_state_->device_value_uploaded = uploaded;
        return uploaded;
    }

    void HiddenStateRowsSelectStage::refreshPinnedSelectedRows()
    {
        if (!gpu_state_ || !gpu_state_->host_selected_rows)
            return;
        for (int i = 0; i < selected_row_count_; ++i)
            gpu_state_->host_selected_rows[i] = selected_rows_[static_cast<size_t>(i)];
    }

    bool HiddenStateRowsSelectStage::executeGPU(TensorBase *input_base, TensorBase *output_base)
    {
        const bool request_terminal_rows =
            params_.device_row_index_source ==
            DeviceRowIndexSource::RequestTerminalLengths;
        const bool device_request_geometry =
            request_terminal_rows &&
            params_.request_row_stride_source ==
                RequestRowStrideSource::ExternalDeviceScalar;
        const bool fixed_contiguous_rows =
            params_.device_row_index_source ==
            DeviceRowIndexSource::FixedContiguousRange;
        const bool shifted_prefill_progress =
            params_.device_row_index_source ==
            DeviceRowIndexSource::ShiftedPrefillKVProgress;
        const bool shifted_prefill_transaction =
            params_.device_row_index_source ==
            DeviceRowIndexSource::ShiftedPrefillTransaction;
        if (!request_terminal_rows && !fixed_contiguous_rows &&
            !shifted_prefill_progress && !shifted_prefill_transaction &&
            !ensureGpuParamStateInitialized())
            return false;

        const bool graph_managed = params_.input_buffer_id.has_value() && params_.output_buffer_id.has_value();
        if (!graph_managed)
        {
            // The production verifier path is graph-managed: BufferArena and
            // StageBufferContract perform coherence before this stage runs.
            // Direct tensor GPU execution is kept only for focused kernel tests
            // and requires the caller to prepare device pointers explicitly.
            if (!input_base->gpu_data_ptr() || !output_base->gpu_data_ptr())
            {
                LOG_ERROR("[HiddenStateRowsSelectStage] Direct GPU execution requires caller-prepared device pointers on "
                          << params_.device_id.toString()
                          << "; graph execution should pass input/output buffer ids so arena coherence owns movement");
                return false;
            }
        }

        const auto *input_device = static_cast<const float *>(input_base->gpu_data_ptr());
        auto *output_device = static_cast<float *>(output_base->gpu_data_ptr());
        if (!input_device || !output_device)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Missing GPU data pointers"
                      << (graph_managed ? " after graph-managed arena coherence" : " after direct tensor preparation")
                      << " device=" << params_.device_id.toString()
                      << " input=" << static_cast<const void *>(input_device)
                      << " output=" << static_cast<void *>(output_device)
                      << " input_buffer="
                      << (params_.input_buffer_id
                              ? bufferIdName(*params_.input_buffer_id)
                              : "<direct>")
                      << " output_buffer="
                      << (params_.output_buffer_id
                              ? bufferIdName(*params_.output_buffer_id)
                              : "<direct>"));
            return false;
        }

        if (!request_terminal_rows && !fixed_contiguous_rows &&
            !shifted_prefill_progress && !shifted_prefill_transaction &&
            !uploadGpuSelectedRows())
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] Failed to update GPU selected-row array");
            return false;
        }

        bool launched = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            if (shifted_prefill_transaction)
            {
                auto *archive_device = static_cast<float *>(
                    params_.terminal_hidden_archive->gpu_data_ptr());
                if (!archive_device)
                {
                    LOG_ERROR("[HiddenStateRowsSelectStage] Shifted-prefill transaction is missing its device terminal archive");
                    return false;
                }
                if (!DebugEnv::isFalseyEnv(
                        "LLAMINAR_PREFIX_CACHE_TRACE"))
                {
                    LOG_DEBUG(
                        "[ShiftedMTPArchiveTrace] device="
                        << params_.device_id.toString()
                        << " input_tensor=" << input_base
                        << " input_source=" << input_device
                        << " archive_tensor="
                        << params_.terminal_hidden_archive
                        << " archive_source=" << archive_device
                        << " packed_tensor=" << output_base
                        << " packed_destination=" << output_device
                        << " captured_rows=" << params_.seq_len
                        << " requests=" << params_.request_count);
                }
                const int row_stride = params_.seq_len / params_.request_count;
                launched = true;
                for (int request = 0; request < params_.request_count; ++request)
                {
                    launched = launched && cuda::launchShiftedMTPPrefillPrepareFP32(
                        input_device,
                        archive_device,
                        output_device,
                        params_.input_token_ids_device,
                        params_.input_position_ids_device,
                        params_.shifted_token_ids_output_device,
                        params_.shifted_position_ids_output_device,
                        params_.shifted_append_lengths_output_device,
                        params_.main_cached_tokens_by_request[static_cast<size_t>(request)],
                        params_.shifted_cached_tokens_by_request[static_cast<size_t>(request)],
                        params_.request_sequence_lengths_device,
                        params_.request_row_stride_device,
                        request,
                        params_.request_count,
                        row_stride,
                        params_.seq_len,
                        params_.d_model,
                        gpuStream());
                }
            }
            else if (shifted_prefill_progress)
            {
                launched = cuda::launchDeviceKVProgressRowsSelectFP32(
                    input_device,
                    output_device,
                    params_.main_cached_tokens_device,
                    params_.shifted_cached_tokens_device,
                    params_.request_sequence_lengths_device,
                    params_.request_row_stride_device,
                    params_.request_index,
                    params_.seq_len,
                    params_.d_model,
                    selected_row_count_,
                    gpuStream());
            }
            else if (request_terminal_rows)
            {
                launched = device_request_geometry
                               ? cuda::launchDeviceGeometryRequestTerminalRowsSelectFP32(
                                     input_device,
                                     output_device,
                                     params_.request_sequence_lengths_device,
                                     params_.request_row_stride_device,
                                     params_.seq_len,
                                     params_.d_model,
                                     selected_row_count_,
                                     gpuStream())
                               : cuda::launchRequestTerminalRowsSelectFP32(
                                     input_device,
                                     output_device,
                                     params_.request_sequence_lengths_device,
                                     params_.seq_len,
                                     params_.request_row_stride,
                                     params_.d_model,
                                     selected_row_count_,
                                     gpuStream());
            }
            else if (fixed_contiguous_rows)
            {
                launched = cuda::launchFixedRowsSelectFP32(
                    input_device,
                    output_device,
                    params_.fixed_contiguous_row_start,
                    selected_row_count_,
                    params_.seq_len,
                    params_.d_model,
                    gpuStream());
            }
            else
            {
                launched = cuda::launchRowsSelectFP32(
                    input_device,
                    output_device,
                    gpu_state_->device_selected_rows,
                    params_.seq_len,
                    params_.d_model,
                    selected_row_count_,
                    gpuStream());
            }
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            if (shifted_prefill_transaction)
            {
                auto *archive_device = static_cast<float *>(
                    params_.terminal_hidden_archive->gpu_data_ptr());
                if (!archive_device)
                {
                    LOG_ERROR("[HiddenStateRowsSelectStage] Shifted-prefill transaction is missing its device terminal archive");
                    return false;
                }
                if (!DebugEnv::isFalseyEnv(
                        "LLAMINAR_PREFIX_CACHE_TRACE"))
                {
                    LOG_DEBUG(
                        "[ShiftedMTPArchiveTrace] device="
                        << params_.device_id.toString()
                        << " input_tensor=" << input_base
                        << " input_source=" << input_device
                        << " archive_tensor="
                        << params_.terminal_hidden_archive
                        << " archive_source=" << archive_device
                        << " packed_tensor=" << output_base
                        << " packed_destination=" << output_device
                        << " captured_rows=" << params_.seq_len
                        << " requests=" << params_.request_count);
                }
                const int row_stride = params_.seq_len / params_.request_count;
                launched = true;
                for (int request = 0; request < params_.request_count; ++request)
                {
                    launched = launched && rocm::launchShiftedMTPPrefillPrepareFP32(
                        input_device,
                        archive_device,
                        output_device,
                        params_.input_token_ids_device,
                        params_.input_position_ids_device,
                        params_.shifted_token_ids_output_device,
                        params_.shifted_position_ids_output_device,
                        params_.shifted_append_lengths_output_device,
                        params_.main_cached_tokens_by_request[static_cast<size_t>(request)],
                        params_.shifted_cached_tokens_by_request[static_cast<size_t>(request)],
                        params_.request_sequence_lengths_device,
                        params_.request_row_stride_device,
                        request,
                        params_.request_count,
                        row_stride,
                        params_.seq_len,
                        params_.d_model,
                        gpuStream());
                }
            }
            else if (shifted_prefill_progress)
            {
                launched = rocm::launchDeviceKVProgressRowsSelectFP32(
                    input_device,
                    output_device,
                    params_.main_cached_tokens_device,
                    params_.shifted_cached_tokens_device,
                    params_.request_sequence_lengths_device,
                    params_.request_row_stride_device,
                    params_.request_index,
                    params_.seq_len,
                    params_.d_model,
                    selected_row_count_,
                    gpuStream());
            }
            else if (request_terminal_rows)
            {
                launched = device_request_geometry
                               ? rocm::launchDeviceGeometryRequestTerminalRowsSelectFP32(
                                     input_device,
                                     output_device,
                                     params_.request_sequence_lengths_device,
                                     params_.request_row_stride_device,
                                     params_.seq_len,
                                     params_.d_model,
                                     selected_row_count_,
                                     gpuStream())
                               : rocm::launchRequestTerminalRowsSelectFP32(
                                     input_device,
                                     output_device,
                                     params_.request_sequence_lengths_device,
                                     params_.seq_len,
                                     params_.request_row_stride,
                                     params_.d_model,
                                     selected_row_count_,
                                     gpuStream());
            }
            else if (fixed_contiguous_rows)
            {
                launched = rocm::launchFixedRowsSelectFP32(
                    input_device,
                    output_device,
                    params_.fixed_contiguous_row_start,
                    selected_row_count_,
                    params_.seq_len,
                    params_.d_model,
                    gpuStream());
            }
            else
            {
                launched = rocm::launchRowsSelectFP32(
                    input_device,
                    output_device,
                    gpu_state_->device_selected_rows,
                    params_.seq_len,
                    params_.d_model,
                    selected_row_count_,
                    gpuStream());
            }
#endif
        }

        if (!launched)
        {
            LOG_ERROR("[HiddenStateRowsSelectStage] GPU rows-select launch failed on " << params_.device_id.toString());
            return false;
        }

        if (!graph_managed)
            gpuExecution().publish(output_base);
        return true;
    }

    void HiddenStateRowsSelectStage::releaseGpuParamState()
    {
        if (!gpu_state_)
            return;

        if (gpu_state_->device.is_cuda())
        {
#ifdef HAVE_CUDA
            cuda::freeRowSelectHostParam(
                gpu_state_->device.cuda_ordinal(),
                gpu_state_->host_selected_rows);
#endif
        }
        else if (gpu_state_->device.is_rocm())
        {
#ifdef HAVE_ROCM
            rocm::freeRowSelectHostParam(
                gpu_state_->device.rocm_ordinal(),
                gpu_state_->host_selected_rows);
#endif
        }

        gpu_state_.reset();
    }

    size_t HiddenStateRowsSelectStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(selected_row_count_) *
               static_cast<size_t>(params_.d_model) *
               sizeof(float) *
               2;
    }

    bool HiddenStateRowsSelectStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo HiddenStateRowsSelectStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("hidden_states", params_.input, params_.seq_len, params_.d_model);
        if (params_.output)
            info.addOutput("selected_rows", params_.output, selected_row_count_, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("selected_row_count", selected_row_count_);
        info.addScalarInt(
            "device_row_index_source",
            static_cast<int>(params_.device_row_index_source));
        info.addScalarInt(
            "fixed_contiguous_row_start",
            params_.fixed_contiguous_row_start);
        info.addScalarInt("request_row_stride", params_.request_row_stride);
        return info;
    }

    StageBufferRequirements HiddenStateRowsSelectStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        requirements.addInput(
            "hidden_states",
            {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model)},
            BufferTensorType::FP32);
        requirements.addOutput(
            "selected_rows",
            {static_cast<size_t>(std::max(0, selected_row_count_)), static_cast<size_t>(params_.d_model)},
            BufferTensorType::FP32);
        return requirements;
    }

    StageBufferContract HiddenStateRowsSelectStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_buffer_id)
            return {};
        if (params_.device_row_index_source ==
            DeviceRowIndexSource::ShiftedPrefillTransaction)
        {
            if (!params_.terminal_hidden_archive_buffer_id)
                return {};
            return StageBufferContract::build()
                .addInput(*params_.input_buffer_id, "FP32")
                .addInput(BufferId::REQUEST_TOKEN_IDS, "INT32")
                .addInput(BufferId::REQUEST_POSITION_IDS, "INT32")
                .addInput(BufferId::REQUEST_BATCH_GEOMETRY, "INT32")
                .addPreallocatedInOut(
                    *params_.terminal_hidden_archive_buffer_id,
                    "FP32")
                .addOutput(*params_.output_buffer_id, "FP32")
                .addOutput(BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS, "INT32")
                .addOutput(BufferId::MTP_SHIFTED_PREFILL_POSITION_IDS, "INT32")
                .addOutput(BufferId::MTP_SHIFTED_PREFILL_APPEND_LENGTHS, "INT32");
        }
        return StageBufferContract::build()
            .addInput(*params_.input_buffer_id, "FP32")
            .addOutput(*params_.output_buffer_id, "FP32");
    }

} // namespace llaminar2
