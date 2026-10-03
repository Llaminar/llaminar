/**
 * @file TPColumnReduceScatterStage.cpp
 * @brief Column-owner native sum over existing arena banks and exact event frontiers.
 *
 * Packing and conversion are local kernels ordered on the native collective's
 * caller stream. The operation preserves the canonical row-based wire policy;
 * rank partitioning changes the returned extent, not the sum's input precision.
 * The paired overlap builder may move this entire transaction to an auxiliary
 * stream, but only its final event join may publish the local result.
 */
#include "TPColumnReduceScatterStage.h"
#include "collective/ILocalTPContext.h"
#include "collective/AllreducePrecisionPolicy.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/common/ColumnShardAssemblyKernels.h"
#include "memory/StageBufferContract.h"
#include "tensors/ITensor.h"
#include "transfer/TransferEngine.h"
#include "utils/DebugEnv.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
extern "C" cudaError_t cudaCastFP32ToFP16(const float *, void *, size_t, int, cudaStream_t);
extern "C" cudaError_t cudaCastFP16ToFP32(const void *, float *, size_t, int, cudaStream_t);
#endif
#ifdef HAVE_ROCM
extern "C" int rocmCastFP32ToFP16(const float *, void *, size_t, int, void *);
extern "C" int rocmCastFP16ToFP32(const void *, float *, size_t, int, void *);
#endif

namespace llaminar2
{
    TPColumnReduceScatterStage::TPColumnReduceScatterStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params)),
          members_(params_.tp_ctx ? params_.tp_ctx->devices() : std::vector<GlobalDeviceAddress>{})
    {
        validateMembership();
        if (params_.rows <= 0 || params_.model_columns <= 0 ||
            params_.model_columns % members_.size() != 0 || params_.stage_name.empty() ||
            (!params_.precision.empty() && params_.precision != "fp16" && params_.precision != "fp32"))
            throw std::invalid_argument("Column reduce-scatter requires whole equal columns and resolved FP32/FP16 wire policy");
        if (std::size_t(params_.rows) > std::numeric_limits<std::size_t>::max() /
                sizeof(float) / std::size_t(params_.model_columns))
            throw std::overflow_error("Column reduce-scatter row geometry overflows");
        elements_ = std::size_t(params_.rows) * params_.model_columns;
        if (params_.live_rows && (params_.live_rows->capacity() != params_.rows ||
                params_.live_rows->physicalRows() != params_.rows))
            throw std::invalid_argument("Column reduce-scatter live rows disagree with physical row geometry");
        fp16_ = fp32SumUsesFP16Transport(params_.precision, elements_, params_.model_columns,
            debugEnv().allreduce_fp16_min_elements);
        validateBuffers();
    }

    void TPColumnReduceScatterStage::validateMembership() const
    {
        if (!params_.device_id.is_gpu() || !params_.tp_ctx || members_.size() < 2 ||
            params_.participant < 0 || std::size_t(params_.participant) >= members_.size())
            throw std::invalid_argument("Column reduce-scatter requires one native multi-GPU participant");
        if (params_.tp_ctx->devices() != members_ || params_.tp_ctx->degree() != int(members_.size()))
            throw std::logic_error("Column reduce-scatter membership changed after construction");
        const auto native = params_.device_id.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        if (params_.tp_ctx->scope() != TPScope::RANK_LOCAL || params_.tp_ctx->backend() != native ||
            members_[params_.participant].toLocalDeviceId() != params_.device_id)
            throw std::invalid_argument("Column reduce-scatter requires exact rank-local native membership");
        for (std::size_t i = 0; i < members_.size(); ++i)
        {
            if (members_[i].toLocalDeviceId().type != params_.device_id.type)
                throw std::invalid_argument("Column reduce-scatter cannot mix GPU backends");
            for (std::size_t j = 0; j < i; ++j)
                if (members_[i].toLocalDeviceId() == members_[j].toLocalDeviceId())
                    throw std::invalid_argument("Column reduce-scatter requires distinct physical devices");
        }
    }

    void TPColumnReduceScatterStage::validateBuffers() const
    {
        const auto bytes = elements_ * sizeof(float);
        if (!params_.tensor || !params_.packing || params_.tensor == params_.packing ||
            !params_.tensor_buffer_id || !params_.packing_buffer_id ||
            params_.tensor_buffer_id == params_.packing_buffer_id ||
            params_.tensor->native_type() != TensorType::FP32 || params_.packing->native_type() != TensorType::FP32 ||
            params_.tensor->size_bytes() < bytes || params_.packing->size_bytes() < bytes)
            throw std::invalid_argument("Column reduce-scatter needs two disjoint admitted full-row FP32 banks");
        const auto a = reinterpret_cast<std::uintptr_t>(params_.tensor->gpu_data_ptr());
        const auto b = reinterpret_cast<std::uintptr_t>(params_.packing->gpu_data_ptr());
        constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
        if ((a && (a % alignof(float) || a > maximum - bytes)) ||
            (b && (b % alignof(float) || b > maximum - bytes)) ||
            (a && b && a < b + bytes && b < a + bytes))
            throw std::invalid_argument("Column reduce-scatter physical banks overlap, misalign or overflow");
    }

    void TPColumnReduceScatterStage::validateEnqueue(const StageGPUExecution &execution) const
    {
        if (execution.device() != params_.device_id)
            throw std::logic_error("Column reduce-scatter producer belongs to another GPU");
        validateMembership();
        validateBuffers();
        execution.requirePreparedInput(params_.tensor);
        execution.requirePreparedOutput(params_.tensor);
        execution.requirePreparedOutput(params_.packing);
    }

    bool TPColumnReduceScatterStage::enqueue(void *stream) const
    {
        if (!stream) throw std::invalid_argument("Column reduce-scatter requires its exact non-null stream");
        auto *partial = static_cast<float *>(params_.tensor->gpu_data_ptr());
        auto *packed = static_cast<float *>(params_.packing->gpu_data_ptr());
        const int degree = static_cast<int>(members_.size());
        const auto count = elements_ / members_.size();
        const auto sum = [&](const void *send, void *receive, CollectiveDataType dtype) {
            if (params_.live_rows)
                return params_.tp_ctx->nativeRowsOnStream(NativeRowCollective::ReduceScatter,
                    send, receive, NativeCollectiveRows(*params_.live_rows, params_.model_columns / degree),
                    dtype, CollectiveOp::ALLREDUCE_SUM, params_.participant, stream, params_.stage_name);
            return params_.tp_ctx->reduceScatterRawOnStream(send, receive, count,
                dtype, params_.participant, stream, params_.stage_name);
        };
        // Transpose [row,rank,column] to [rank,row,column]. Exchanging the
        // outer dimensions reuses the existing exact-copy deinterleave kernel.
        if (!assembleColumnShardsFP32(params_.device_id, partial, packed,
                degree, params_.rows, params_.model_columns / degree, stream)) return false;
        if (!fp16_)
            return sum(packed, partial, CollectiveDataType::FLOAT32);

        // Packing has consumed every original partial. Reuse that dead bank
        // for FP16 send values, then reuse packed storage for FP16 receives.
        // The final cast restores a contiguous FP32 local-column prefix.
        bool cast = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda()) cast = cudaCastFP32ToFP16(packed, partial,
            elements_, params_.device_id.ordinal, static_cast<cudaStream_t>(stream)) == cudaSuccess;
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm()) cast = rocmCastFP32ToFP16(packed, partial,
            elements_, params_.device_id.ordinal, stream) == 0;
#endif
        if (!cast || !sum(partial, packed, CollectiveDataType::FLOAT16)) return false;
        cast = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda()) cast = cudaCastFP16ToFP32(packed, partial,
            count, params_.device_id.ordinal, static_cast<cudaStream_t>(stream)) == cudaSuccess;
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm()) cast = rocmCastFP16ToFP32(packed, partial,
            count, params_.device_id.ordinal, stream) == 0;
#endif
        return cast;
    }

    bool TPColumnReduceScatterStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || ctx->deviceId() != params_.device_id)
            throw std::invalid_argument("Column reduce-scatter requires its participant context");
        const auto execution = gpuExecution();
        validateEnqueue(execution);
        if (!enqueue(execution.nativeStream())) { params_.tp_ctx->requestAbort(); return false; }
        execution.publish(params_.tensor);
        execution.publish(params_.packing);
        return true;
    }

    bool TPColumnReduceScatterStage::enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const
    {
        if (!input.valid() || input.sourceOwner() != params_.tensor || input.device() != params_.device_id)
            throw std::invalid_argument("Column reduce-scatter fork requires its exact acquired partial");
        validateMembership();
        validateBuffers();
        if (!enqueue(input.consumerStream())) { params_.tp_ctx->requestAbort(); return false; }
        return true;
    }

    bool TPColumnReduceScatterStage::supportsBackend(ComputeBackendType backend) const
    {
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda() && backend == ComputeBackendType::GPU_CUDA) return true;
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm() && backend == ComputeBackendType::GPU_ROCM) return true;
#endif
        return false;
    }

    bool TPColumnReduceScatterStage::isGraphCapturable() const
    {
        validateMembership();
        return supportsBackend(params_.device_id.is_cuda() ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
    }

    StageBufferContract TPColumnReduceScatterStage::bufferContract() const
    {
        return StageBufferContract::build().addInOut(*params_.tensor_buffer_id)
            .addOutput(*params_.packing_buffer_id);
    }

    StageBufferRequirements TPColumnReduceScatterStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        requirements.addInput("partial", params_.tensor->shape(), BufferTensorType::FP32);
        requirements.addOutput("local_columns", params_.tensor->shape(), BufferTensorType::FP32);
        requirements.addOutput("packing", params_.packing->shape(), BufferTensorType::FP32);
        return requirements;
    }

    StageDumpInfo TPColumnReduceScatterStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("participant", params_.participant).addScalarInt("degree", members_.size());
        info.addOutput("local_columns", params_.tensor, params_.rows, params_.model_columns / members_.size());
        return info;
    }
}
