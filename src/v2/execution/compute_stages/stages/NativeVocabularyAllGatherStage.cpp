/**
 * @file NativeVocabularyAllGatherStage.cpp
 * @brief Capture-safe lossless publication of sharded terminal logits.
 *
 * Native live-row allgather retains fixed rank strides while clipping wire
 * payloads on device. Assembly reads precisely that same live prefix, leaves
 * inactive output rows untouched, and publishes on the collective's stream.
 * No host count, blocking wait, temporary allocation or precision conversion
 * participates in this transaction.
 */
#include "NativeVocabularyAllGatherStage.h"
#include "collective/ILocalTPContext.h"
#include "collective/VocabularyGatherWorkspaceContract.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/common/ColumnShardAssemblyKernels.h"
#include "memory/StageBufferContract.h"
#include "tensors/ITensor.h"
#include "tensors/TensorClasses.h"
#include "utils/KernelProfiler.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Validate two immutable byte ranges without allocating or dereferencing them. */
        void requireDisjoint(const void *first, std::size_t first_bytes,
            const void *second, std::size_t second_bytes)
        {
            const auto a = reinterpret_cast<std::uintptr_t>(first);
            const auto b = reinterpret_cast<std::uintptr_t>(second);
            constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
            if ((a && (a % alignof(float) || a > maximum - first_bytes)) ||
                (b && (b % alignof(float) || b > maximum - second_bytes)) ||
                (a && b && a < b + second_bytes && b < a + first_bytes))
                throw std::invalid_argument("Native vocabulary gather banks overlap, misalign or overflow");
        }
    }

    NativeVocabularyAllGatherStage::NativeVocabularyAllGatherStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params)),
          members_(params_.tp_ctx ? params_.tp_ctx->devices() : std::vector<GlobalDeviceAddress>{}),
          rows_(params_.rows, static_cast<std::size_t>(params_.local_vocabulary))
    {
        validateMembership();
        if (params_.stage_name.empty() || !rows_.byteGeometryValid(sizeof(float), params_.tp_ctx->degree()) ||
            rows_.bankElements() > std::numeric_limits<std::size_t>::max() / sizeof(float) / (members_.size() + 1) ||
            params_.local_vocabulary <= 0 || params_.vocabulary <= 0 ||
            params_.vocabulary % params_.tp_ctx->degree() != 0 ||
            params_.vocabulary / params_.tp_ctx->degree() != params_.local_vocabulary)
            throw std::invalid_argument("Native vocabulary gather requires equal, complete vocabulary shards");
        validateBuffers();
    }

    void NativeVocabularyAllGatherStage::validateMembership() const
    {
        if (!params_.device_id.is_gpu() || !params_.tp_ctx || members_.size() < 2 ||
            params_.participant < 0 || static_cast<std::size_t>(params_.participant) >= members_.size())
            throw std::invalid_argument("Native vocabulary gather requires an exact local GPU participant");
        if (params_.tp_ctx->devices() != members_ || params_.tp_ctx->degree() != static_cast<int>(members_.size()))
            throw std::logic_error("Native vocabulary gather membership changed after construction");
        if (params_.tp_ctx->backend() != (params_.device_id.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL) ||
            !params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture() ||
            members_[params_.participant].toLocalDeviceId() != params_.device_id)
            throw std::invalid_argument("Native vocabulary gather requires matching capturable NCCL/RCCL membership");
        for (std::size_t i = 0; i < members_.size(); ++i)
        {
            if (members_[i].toLocalDeviceId().type != params_.device_id.type)
                throw std::invalid_argument("Native vocabulary gather cannot mix GPU backends");
            for (std::size_t j = 0; j < i; ++j)
                if (members_[i].toLocalDeviceId() == members_[j].toLocalDeviceId())
                    throw std::invalid_argument("Native vocabulary gather requires distinct physical GPUs");
        }
    }

    void NativeVocabularyAllGatherStage::validateBuffers() const
    {
        const auto local_bytes = rows_.bankElements() * sizeof(float);
        const auto full_bytes = local_bytes * members_.size();
        if (!params_.local_logits || !params_.full_logits || params_.local_logits == params_.full_logits ||
            params_.input_id == params_.output_id ||
            params_.local_logits->native_type() != TensorType::FP32 || params_.full_logits->native_type() != TensorType::FP32 ||
            params_.local_logits->size_bytes() < local_bytes || params_.full_logits->size_bytes() < full_bytes)
            throw std::invalid_argument("Native vocabulary gather needs disjoint admitted local/full FP32 tensors");
        requireDisjoint(params_.local_logits->gpu_data_ptr(), local_bytes, params_.full_logits->gpu_data_ptr(), full_bytes);
    }

    bool NativeVocabularyAllGatherStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || ctx->deviceId() != params_.device_id)
            throw std::invalid_argument("Native vocabulary gather requires its participant context");
        validateMembership();
        validateBuffers();
        const auto execution = gpuExecution();
        execution.requirePreparedInput(params_.local_logits);
        execution.requirePreparedOutput(params_.full_logits);
        const auto full_bytes = rows_.bankElements() * members_.size() * sizeof(float);
        auto *output = static_cast<float *>(params_.full_logits->gpu_data_ptr());
        auto *rank_major = output;
        if (params_.rows.physicalRows() > 1)
        {
            if (!workspace_ || workspace_->device() != params_.device_id ||
                workspace_->getBufferSize(VocabularyGatherWorkspaceContract::rankMajorBank) < full_bytes)
                throw std::logic_error("Native vocabulary gather transpose bank was not admitted before capture");
            rank_major = static_cast<float *>(workspace_->getBuffer(VocabularyGatherWorkspaceContract::rankMajorBank));
            if (!rank_major)
                throw std::logic_error("Native vocabulary gather transpose bank has no resident pointer");
            requireDisjoint(rank_major, full_bytes, output, full_bytes);
            requireDisjoint(rank_major, full_bytes, params_.local_logits->gpu_data_ptr(), rows_.bankElements() * sizeof(float));
        }
        KERNEL_PROFILE_SCOPE(KernelType::ALLGATHER);
        if (!params_.tp_ctx->nativeRowsOnStream(NativeRowCollective::AllGather,
                params_.local_logits->gpu_data_ptr(), rank_major, rows_, CollectiveDataType::FLOAT32,
                CollectiveOp::ALLGATHER, params_.participant, execution.nativeStream(), params_.stage_name, params_.payload_bytes))
        {
            params_.tp_ctx->requestAbort();
            return false;
        }
        if (params_.rows.physicalRows() > 1 &&
            !assembleLiveColumnShardsFP32(params_.device_id, rank_major, output, params_.rows,
                static_cast<int>(members_.size()), params_.local_vocabulary, execution.nativeStream()))
        {
            params_.tp_ctx->requestAbort();
            return false;
        }
        execution.publish(params_.full_logits);
        PerfStatsCollector::addCounter("collective", "native_vocabulary_allgather_graph_nodes", 1.0, "capture",
            params_.device_id.toString(), {{"rows_capacity", std::to_string(params_.rows.capacity())},
                {"vocabulary", std::to_string(params_.vocabulary)}, {"live_rows", params_.rows.countOwner() ? "device" : "fixed"}});
        return true;
    }

    bool NativeVocabularyAllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda() && backend == ComputeBackendType::GPU_CUDA) return true;
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm() && backend == ComputeBackendType::GPU_ROCM) return true;
#endif
        return false;
    }

    bool NativeVocabularyAllGatherStage::isGraphCapturable() const
    {
        validateMembership();
        return supportsBackend(params_.device_id.is_cuda() ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
    }

    std::size_t NativeVocabularyAllGatherStage::estimatedMemoryBytes() const
    {
        return rows_.bankElements() * sizeof(float) * (members_.size() + 1);
    }

    StageBufferRequirements NativeVocabularyAllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements result;
        // Arena parents can retain more rows than this operation owns. Declare
        // the frozen operation axes, so capacity never becomes executable M.
        result.addInput("local_logits", {static_cast<std::size_t>(params_.rows.capacity()),
            static_cast<std::size_t>(params_.local_vocabulary)}, toBufferTensorType(TensorType::FP32));
        result.addOutput("full_logits", {static_cast<std::size_t>(params_.rows.capacity()),
            static_cast<std::size_t>(params_.vocabulary)}, toBufferTensorType(TensorType::FP32));
        return result;
    }

    StageBufferContract NativeVocabularyAllGatherStage::bufferContract() const
    {
        return StageBufferContract::build().addInput(params_.input_id).addOutput(params_.output_id);
    }

    StageDumpInfo NativeVocabularyAllGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo result;
        result.addScalarInt("participant", params_.participant).addScalarInt("rows_capacity", params_.rows.capacity());
        result.addInput("local_logits", params_.local_logits, params_.local_logits->rows(), params_.local_logits->cols());
        result.addOutput("full_logits", params_.full_logits, params_.full_logits->rows(), params_.full_logits->cols());
        return result;
    }

    WorkspaceRequirements NativeVocabularyAllGatherStage::getWorkspaceRequirements(int m, int, int) const
    {
        if (m <= 0)
            throw std::invalid_argument("Native vocabulary workspace preview needs positive operation rows");
        // Serial decode may publish the shared bank before the larger retained
        // verifier exists. Price its declared M now; execution keeps frozen rows.
        return VocabularyGatherWorkspaceContract::requirements(
            std::max(m, params_.rows.capacity()), params_.vocabulary);
    }
}
