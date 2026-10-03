/**
 * @file NativeAllGatherStage.cpp
 * @brief Lossless native collective lowering over existing GPU arena storage.
 *
 * Setup authenticates complete membership and byte extents. Recording borrows
 * the exact stage stream and producer publication, enqueues NCCL/RCCL once,
 * then records the receive publication on that same stream. Native capture
 * owns replay ordering; there is no per-token host protocol or memory owner.
 */
#include "NativeAllGatherStage.h"

#include "collective/ILocalTPContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "memory/StageBufferContract.h"
#include "tensors/ITensor.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/KernelProfiler.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    NativeAllGatherStage::NativeAllGatherStage(Params params)
        : CapturedAllGatherStage(params.device_id), params_(std::move(params)),
          members_(params_.tp_ctx ? params_.tp_ctx->devices() : std::vector<GlobalDeviceAddress>{})
    {
        validateMembership();
        if (params_.bytes_per_participant == 0 || params_.stage_name.empty())
            throw std::invalid_argument("Native allgather requires a nonempty message and stage identity");
        if (params_.live_rows && (params_.live_rows->bankElements() != params_.bytes_per_participant ||
                !params_.live_rows->byteGeometryValid(1, static_cast<int>(members_.size()))))
            throw std::invalid_argument("Native allgather live rows disagree with the admitted byte bank");
        const auto limit = std::numeric_limits<std::size_t>::max();
        // Include the send prefix in this bound, not only the receive bank:
        // estimated traffic must not wrap after receive geometry was admitted.
        if (params_.bytes_per_participant > limit / (members_.size() + 1))
            throw std::overflow_error("Native allgather byte geometry overflows size_t");
        receive_bytes_ = params_.bytes_per_participant * members_.size();
        traffic_bytes_ = params_.bytes_per_participant + receive_bytes_;
        validateBuffers();
    }

    void NativeAllGatherStage::validateMembership() const
    {
        if (!params_.device_id.is_gpu() || !params_.tp_ctx ||
            params_.participant < 0 || members_.size() < 2 ||
            static_cast<std::size_t>(params_.participant) >= members_.size())
            throw std::invalid_argument("Native allgather requires one exact GPU participant and native context");
        if (params_.tp_ctx->devices() != members_ ||
            params_.tp_ctx->degree() != static_cast<int>(members_.size()))
            throw std::logic_error("Native allgather membership changed after graph construction");
        const auto expected_backend = params_.device_id.is_cuda()
            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        if (params_.tp_ctx->backend() != expected_backend ||
            members_[params_.participant].toLocalDeviceId() != params_.device_id ||
            !params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture())
            throw std::invalid_argument("Native allgather requires matching capturable NCCL/RCCL membership");
        // Ordinal uniqueness is within this existing rank-local communicator.
        // Do not derive physical-node membership from hostname strings here.
        for (std::size_t i = 0; i < members_.size(); ++i)
        {
            if (members_[i].toLocalDeviceId().type != params_.device_id.type)
                throw std::invalid_argument("Native allgather cannot join different device backends");
            for (std::size_t j = 0; j < i; ++j)
                if (members_[i].toLocalDeviceId() == members_[j].toLocalDeviceId())
                    throw std::invalid_argument("Native allgather requires distinct physical GPUs");
        }
    }

    void NativeAllGatherStage::validateBuffers() const
    {
        if (!params_.local_input || !params_.rank_major_output ||
            params_.local_input == params_.rank_major_output ||
            !params_.input_buffer_id || !params_.output_buffer_id ||
            params_.input_buffer_id == params_.output_buffer_id ||
            params_.local_input->size_bytes() < params_.bytes_per_participant ||
            params_.rank_major_output->size_bytes() < receive_bytes_)
            throw std::invalid_argument("Native allgather requires disjoint declared buffers covering exact message extents");
        const auto send = reinterpret_cast<std::uintptr_t>(params_.local_input->gpu_data_ptr());
        const auto receive = reinterpret_cast<std::uintptr_t>(params_.rank_major_output->gpu_data_ptr());
        // Construction may precede arena materialization. When storage exists,
        // authenticate physical ranges as well as the distinct logical IDs.
        // execute() separately requires both allocations on the exact GPU.
        if (send && receive)
        {
            constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
            if (send > limit - params_.bytes_per_participant || receive > limit - receive_bytes_ ||
                (send < receive + receive_bytes_ && receive < send + params_.bytes_per_participant))
                throw std::invalid_argument("Native allgather device byte ranges overlap or overflow");
        }
    }

    bool NativeAllGatherStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "NativeAllGatherStage")) return false;
        if (ctx->deviceId() != params_.device_id)
            throw std::logic_error("Native allgather execution context belongs to another device");
        const auto execution = gpuExecution();
        validateEnqueue(execution);
        if (!enqueueOnStream(execution.nativeStream())) return false;
        execution.publish(params_.rank_major_output);
        return true;
    }

    bool NativeAllGatherStage::enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const
    {
        if (!input.valid() || input.device() != params_.device_id ||
            input.sourceOwner() != params_.local_input)
            throw std::invalid_argument("Native allgather requires its exact acquired input fork");
        validateMembership();
        validateBuffers();
        TransferEngine::requireDeviceOutput(params_.rank_major_output, input.device(), input.consumerStream());
        return enqueueOnStream(input.consumerStream());
    }

    bool NativeAllGatherStage::enqueueOnStream(void *stream) const
    {
        const auto *send = params_.local_input->gpu_data_ptr();
        auto *receive = params_.rank_major_output->gpu_data_ptr();

        KERNEL_PROFILE_SCOPE(KernelType::ALLGATHER);
        // INT8 here is a byte transport, not an activation precision. It keeps
        // arbitrary packed payloads, FP scales, NaNs and signed zeros unchanged.
        // Allreduce's configured wire precision must never alter this message.
        if (params_.live_rows)
            return params_.tp_ctx->nativeRowsOnStream(NativeRowCollective::AllGather, send, receive,
                *params_.live_rows, CollectiveDataType::INT8, CollectiveOp::ALLGATHER,
                params_.participant, stream, params_.stage_name);
        return params_.tp_ctx->allgatherRawOnStream(send, receive,
                params_.bytes_per_participant, CollectiveDataType::INT8,
                params_.participant, stream, params_.stage_name);
    }

    void NativeAllGatherStage::validateEnqueue(const StageGPUExecution &execution) const
    {
        if (execution.device() != params_.device_id)
            throw std::logic_error("Native allgather producer frontier belongs to another device");
        validateMembership();
        validateBuffers();
        execution.requirePreparedInput(params_.local_input);
        execution.requirePreparedOutput(params_.rank_major_output);
    }

    bool NativeAllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda() && backend == ComputeBackendType::GPU_CUDA) return true;
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm() && backend == ComputeBackendType::GPU_ROCM) return true;
#endif
        return false;
    }

    bool NativeAllGatherStage::isGraphCapturable() const
    {
        validateMembership();
        return supportsBackend(params_.device_id.is_cuda()
            ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
    }

    StageBufferRequirements NativeAllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        requirements.addInput("local_input", params_.local_input->shape(),
            toBufferTensorType(params_.local_input->native_type()));
        requirements.addOutput("rank_major_output", params_.rank_major_output->shape(),
            toBufferTensorType(params_.rank_major_output->native_type()));
        return requirements;
    }

    StageBufferContract NativeAllGatherStage::bufferContract() const
    {
        return StageBufferContract::build()
            .addInput(*params_.input_buffer_id).addOutput(*params_.output_buffer_id);
    }

    StageDumpInfo NativeAllGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("participant", params_.participant)
            .addScalarInt("degree", static_cast<int>(members_.size()));
        info.addInput("local_input", params_.local_input,
            params_.local_input->rows(), params_.local_input->cols());
        info.addOutput("rank_major_output", params_.rank_major_output,
            params_.rank_major_output->rows(), params_.rank_major_output->cols());
        return info;
    }
} // namespace llaminar2
