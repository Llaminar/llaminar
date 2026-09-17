/**
 * @file DecodePositionSnapshotStage.cpp
 * @brief Captured scalar KV-position publication using the shared GPU primitive.
 *
 * A position snapshot is model work, not a host launch prerequisite. Capturing
 * it ahead of every model root allows a retained generation parent to repeat a
 * decode child safely after the previous iteration advanced the cache.
 */
#include "DecodePositionSnapshotStage.h"
#include "../../../backends/IBackend.h"
#include "../../../memory/BufferId.h"
#include <stdexcept>
#include <utility>

namespace llaminar2
{
DecodePositionSnapshotStage::DecodePositionSnapshotStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
    if (!params_.device_id.is_gpu() || !params_.binding.valid() ||
        params_.binding.backend->backendDeviceType() != params_.device_id.type)
        throw std::invalid_argument("DecodePositionSnapshotStage: invalid resident binding");
}

bool DecodePositionSnapshotStage::execute(IDeviceContext *ctx)
{
    if (!ensureContext(ctx, "DecodePositionSnapshotStage"))
        return false;
    void *const stream = requireGPUStream();
    return stream && params_.binding.backend->enqueuePrepareMTPVerifierPositionIds(
        params_.binding.cached_tokens, 1, 1, params_.device_id.gpu_ordinal(),
        stream, params_.binding.position);
}

bool DecodePositionSnapshotStage::supportsBackend(ComputeBackendType backend) const
{
    return backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM;
}

StageBufferContract DecodePositionSnapshotStage::bufferContract() const
{
    StageBufferContract contract;
    contract.addOutput(BufferId::REQUEST_POSITION_IDS);
    return contract;
}

StageDumpInfo DecodePositionSnapshotStage::buildDumpInfoImpl() const
{
    StageDumpInfo info;
    info.addScalarInt("rows", 1);
    return info;
}
}
