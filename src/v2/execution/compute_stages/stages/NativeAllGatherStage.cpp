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
#include "execution/local_execution/graph/GraphCaptureGuard.h"
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
        if (params_.live_rows && params_.arena_rows)
            throw std::invalid_argument("Native allgather accepts one fixed/raw/arena row alternative");
        validateCountDeclaration();
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
        validateCountDeclaration();
    }

    void NativeAllGatherStage::validateCountDeclaration() const
    {
        if(params_.payload_receipt) {
            params_.payload_receipt->requireFor(params_.device_id);
            if(!params_.arena_rows || params_.payload_receipt->bankBytes()!=params_.bytes_per_participant ||
                params_.payload_receipt->owner()==params_.local_input ||
                params_.payload_receipt->owner()==params_.rank_major_output ||
                params_.payload_receipt->owner()==params_.arena_rows->owner ||
                params_.payload_receipt->bufferId()==params_.input_buffer_id ||
                params_.payload_receipt->bufferId()==params_.output_buffer_id ||
                params_.payload_receipt->bufferId()==params_.arena_rows->buffer)
                throw std::invalid_argument("Native collective receipt requires its exact counted arena frame");
            const auto controller=reinterpret_cast<std::uintptr_t>(params_.payload_receipt->owner()->gpu_data_ptr());
            const auto controller_bytes=params_.payload_receipt->owner()->size_bytes();
            if(controller>std::numeric_limits<std::uintptr_t>::max()-controller_bytes)
                throw std::invalid_argument("Native collective controller storage range overflows");
            for(const auto &[tensor,bytes]:{std::pair{params_.local_input,params_.bytes_per_participant},
                    std::pair{params_.rank_major_output,receive_bytes_}}) {
                const auto bank=reinterpret_cast<std::uintptr_t>(tensor?tensor->gpu_data_ptr():nullptr);
                if(bank && (bank>std::numeric_limits<std::uintptr_t>::max()-bytes ||
                    (controller<bank+bytes && bank<controller+controller_bytes)))
                    throw std::invalid_argument("Native collective receipt aliases a transport bank");
            }
        }
        if (!params_.arena_rows) return;
        const auto &count = *params_.arena_rows;
        if (!count.owner || count.buffer >= BufferId::_COUNT ||
            count.owner->transferStorageOwner() != count.owner || count.owner->is_view() ||
            count.owner->native_type() != TensorType::INT32 ||
            count.byte_offset % alignof(std::int32_t) ||
            count.byte_offset > count.owner->size_bytes() ||
            sizeof(std::int32_t) > count.owner->size_bytes() - count.byte_offset ||
            count.capacity <= 0 || !count.bytes_per_row ||
            count.bytes_per_row > std::numeric_limits<std::uint32_t>::max() ||
            count.bytes_per_row > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(count.capacity) ||
            count.bytes_per_row * static_cast<std::size_t>(count.capacity) != params_.bytes_per_participant ||
            count.owner == params_.local_input || count.owner == params_.rank_major_output ||
            count.buffer == params_.input_buffer_id || count.buffer == params_.output_buffer_id)
            throw std::invalid_argument("Native allgather arena count requires a distinct canonical INT32 owner and exact byte geometry");
        const auto *base = count.owner->gpu_data_ptr();
        if (!base) return; // Normal output preparation, not this declaration, materializes the root.
        const auto address = reinterpret_cast<std::uintptr_t>(base);
        if (count.owner->current_device() != params_.device_id ||
            count.byte_offset > std::numeric_limits<std::uintptr_t>::max() - sizeof(std::int32_t) ||
            address > std::numeric_limits<std::uintptr_t>::max() - count.byte_offset - sizeof(std::int32_t) ||
            (address + count.byte_offset) % alignof(std::int32_t))
            throw std::invalid_argument("Native allgather arena count has foreign or unaligned native storage");
        // Distinct declarations may not hide physical aliases. Only the live
        // scalar bytes are read, but they must survive both complete message banks.
        const auto scalar = address + count.byte_offset;
        if(params_.payload_receipt) {
            const auto controller=reinterpret_cast<std::uintptr_t>(params_.payload_receipt->owner()->gpu_data_ptr());
            const auto bytes=params_.payload_receipt->owner()->size_bytes();
            if(controller<scalar+sizeof(std::int32_t) && scalar<controller+bytes)
                throw std::invalid_argument("Native collective receipt aliases its live count");
        }
        for (const auto &[tensor, bytes] : {std::pair{params_.local_input, params_.bytes_per_participant},
                std::pair{params_.rank_major_output, receive_bytes_}})
        {
            const auto bank = reinterpret_cast<std::uintptr_t>(tensor ? tensor->gpu_data_ptr() : nullptr);
            if (bank && (bank > std::numeric_limits<std::uintptr_t>::max() - bytes ||
                    (scalar < bank + bytes && bank < scalar + sizeof(std::int32_t))))
                throw std::invalid_argument("Native allgather count aliases a transport bank");
        }
    }

    bool NativeAllGatherStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        if (!ctx || ctx->deviceId() != params_.device_id || !stream)
            throw std::invalid_argument("Native allgather preparation requires its exact participant and explicit stream");
        if (params_.arena_rows)
        {
            if (isGraphCaptureActive())
                throw std::logic_error("Native allgather arena count must be prepared before capture");
            validateCountDeclaration();
            if (count_seal_) validateCountSeal();
            else
            {
                auto *owner = params_.arena_rows->owner;
                auto *backend = owner->resolveBackend(params_.device_id);
                const auto generation = backend ? backend->deviceRuntimeGeneration(params_.device_id.ordinal) : 0;
                if (!owner->gpu_data_ptr() || owner->current_device() != params_.device_id || !generation)
                    throw std::invalid_argument("Native allgather count requires ordinary prepared GPU storage and runtime generation");
                count_seal_ = CountSeal{owner->gpu_data_ptr(), owner->size_bytes(), backend, generation};
            }
        }
        return IComputeStage::prepareGraphLaunch(ctx, stream);
    }

    void NativeAllGatherStage::validateCountSeal() const
    {
        if (!params_.arena_rows) return;
        validateCountDeclaration();
        if (!count_seal_)
            throw std::logic_error("Native allgather arena count has not been prepared for capture");
        const auto &seal = *count_seal_;
        auto *owner = params_.arena_rows->owner;
        if (owner->gpu_data_ptr() != seal.base || owner->size_bytes() != seal.owner_bytes ||
            owner->current_device() != params_.device_id || owner->resolveBackend(params_.device_id) != seal.backend ||
            seal.backend->deviceRuntimeGeneration(params_.device_id.ordinal) != seal.runtime_generation)
            throw std::logic_error("Native allgather count owner/address/device/runtime changed after capture preparation");
    }

    NativeCollectiveRows NativeAllGatherStage::arenaRows() const
    {
        validateCountSeal();
        const auto &declaration = *params_.arena_rows;
        const auto *count = reinterpret_cast<const std::int32_t *>(
            static_cast<const std::byte *>(count_seal_->base) + declaration.byte_offset);
        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(declaration.capacity, count), declaration.bytes_per_row);
        return params_.payload_receipt?rows.withPayloadReceipt(params_.payload_receipt->native(params_.device_id)):rows;
    }

    std::uint64_t NativeAllGatherStage::graphCaptureVariantSignature() const
    {
        validateMembership();
        validateCountDeclaration();
        if (count_seal_) validateCountSeal();
        // Keep this declaration stable before/after ordinary materialization.
        // The immutable physical seal is checked on every identity query and
        // enqueue; it is not installed by changing the cache key mid-setup.
        std::uint64_t signature = 1469598103934665603ULL;
        const auto mix = [&](std::uint64_t word) { signature ^= word; signature *= 1099511628211ULL; };
        mix(1); mix(static_cast<std::uint64_t>(params_.device_id.type)); mix(params_.device_id.ordinal);
        mix(reinterpret_cast<std::uintptr_t>(params_.tp_ctx)); mix(params_.participant); mix(members_.size());
        for (const auto &member : members_) { mix(static_cast<std::uint64_t>(member.toLocalDeviceId().type)); mix(member.toLocalDeviceId().ordinal); }
        mix(reinterpret_cast<std::uintptr_t>(params_.local_input)); mix(reinterpret_cast<std::uintptr_t>(params_.rank_major_output));
        mix(params_.bytes_per_participant); mix(static_cast<std::uint64_t>(*params_.input_buffer_id)); mix(static_cast<std::uint64_t>(*params_.output_buffer_id));
        mix(params_.arena_rows ? 2 : params_.live_rows ? 1 : 0);
        if (params_.arena_rows)
        {
            const auto &count = *params_.arena_rows;
            mix(reinterpret_cast<std::uintptr_t>(count.owner)); mix(static_cast<std::uint64_t>(count.buffer));
            mix(count.byte_offset); mix(count.capacity); mix(count.bytes_per_row); mix(count.owner->size_bytes());
        }
        else if (params_.live_rows)
        {
            const auto identity = params_.live_rows->rows().captureIdentity();
            mix(reinterpret_cast<std::uintptr_t>(identity.count)); mix(identity.count_capacity);
            mix(identity.first_physical_row); mix(identity.physical_rows); mix(identity.repeats_per_row); mix(identity.schema_version);
            mix(params_.live_rows->elementsPerRow()); mix(reinterpret_cast<std::uintptr_t>(params_.live_rows->payloadReceipt()));
        }
        if(params_.payload_receipt)mix(params_.payload_receipt->captureIdentity());
        return signature ? signature : 1;
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
        if(params_.payload_receipt)execution.publish(params_.payload_receipt->owner());
        return true;
    }

    bool NativeAllGatherStage::enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const
    {
        if (!input.valid() || input.device() != params_.device_id ||
            input.sourceOwner() != params_.local_input)
            throw std::invalid_argument("Native allgather requires its exact acquired input fork");
        validateMembership();
        validateBuffers();
        validateCountSeal();
        if (params_.arena_rows && !input.includesSource(params_.arena_rows->owner))
            throw std::invalid_argument("Native allgather acquired fork omits its exact arena count producer");
        if(params_.payload_receipt && !input.includesSource(params_.payload_receipt->owner()))
            throw std::invalid_argument("Native allgather acquired fork omits its exact controller observation producer");
        TransferEngine::requireDeviceOutput(params_.rank_major_output, input.device(), input.consumerStream());
        if(params_.payload_receipt)
            TransferEngine::requireDeviceOutput(params_.payload_receipt->owner(),input.device(),input.consumerStream());
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
        if (params_.arena_rows)
            return params_.tp_ctx->nativeRowsOnStream(NativeRowCollective::AllGather, send, receive,
                arenaRows(), CollectiveDataType::INT8, CollectiveOp::ALLGATHER,
                params_.participant, stream, params_.stage_name);
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
        validateCountSeal();
        execution.requirePreparedInput(params_.local_input);
        if (params_.arena_rows) execution.requirePreparedInput(params_.arena_rows->owner);
        execution.requirePreparedOutput(params_.rank_major_output);
        if(params_.payload_receipt) {
            execution.requirePreparedInput(params_.payload_receipt->owner());
            execution.requirePreparedOutput(params_.payload_receipt->owner());
        }
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
        validateCountDeclaration();
        if (params_.arena_rows && !count_seal_) return false;
        if (count_seal_) validateCountSeal();
        return supportsBackend(params_.device_id.is_cuda()
            ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
    }

    bool NativeAllGatherStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        validateMembership();
        validateCountDeclaration();
        return supportsBackend(params_.device_id.is_cuda()
            ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
    }

    StageBufferRequirements NativeAllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        requirements.addInput("local_input", params_.local_input->shape(),
            toBufferTensorType(params_.local_input->native_type()));
        if (params_.arena_rows)
            requirements.addInput("transport_rows", params_.arena_rows->owner->shape(), BufferTensorType::INT32);
        if(params_.payload_receipt) {
            requirements.addInput("request_observation",params_.payload_receipt->owner()->shape(),BufferTensorType::INT32);
            requirements.addOutput("request_observation_after",params_.payload_receipt->owner()->shape(),BufferTensorType::INT32);
        }
        requirements.addOutput("rank_major_output", params_.rank_major_output->shape(),
            toBufferTensorType(params_.rank_major_output->native_type()));
        return requirements;
    }

    StageBufferContract NativeAllGatherStage::bufferContract() const
    {
        auto contract = StageBufferContract::build().addInput(*params_.input_buffer_id).addOutput(*params_.output_buffer_id);
        if (params_.arena_rows) contract.addInput(params_.arena_rows->buffer, "INT32");
        if(params_.payload_receipt)contract.addPreallocatedInOut(params_.payload_receipt->bufferId(),"INT32");
        return contract;
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
