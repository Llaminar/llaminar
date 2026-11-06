/**
 * @file PooledImmutableLocalHotReadyRoot.cpp
 * @brief Original-source validation and actual fixed LocalReady lease observations.
 *
 * A ready page is neither GPU cache payload nor shared CPU source storage. Its
 * independent CPU staging claim survives through every mapped alias; the
 * existing mapped-region owner frees native storage before releasing that claim.
 */
#include "PooledImmutableLocalHotReadyRoot.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeContextPreparation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <stdexcept>

namespace llaminar2::qwen4exp
{
    std::shared_ptr<PooledImmutableLocalHotReadyRoot> PooledImmutableLocalHotReadyRoot::create(
        PhysicalMemoryAllocatorIdentity consumer, std::shared_ptr<PhysicalMemoryAuthority> authority)
    {
        return create(consumer, std::move(authority), TransferEngine::instance());
    }

    std::shared_ptr<PooledImmutableLocalHotReadyRoot> PooledImmutableLocalHotReadyRoot::create(
        PhysicalMemoryAllocatorIdentity consumer, std::shared_ptr<PhysicalMemoryAuthority> authority,
        const TransferEngine& transfers)
    {
        if (isGraphCaptureActive())
            throw std::logic_error("Local ready storage must be materialized before graph capture");
        if (!authority || consumer.world_rank != authority->worldRank() ||
            !consumer.device.is_valid() || !consumer.device.is_gpu())
            throw std::invalid_argument("Local ready storage requires its original rank/GPU and canonical authority");
        // Retain an already-born original source; this can never mint a worker
        // or accept an equal-sized replacement PMA before native registration.
        const auto original = GPUDeviceContextPool::instance().retainExistingPreparedContext(consumer.device,authority);
        if (original.device() != consumer.device || original.memory().get() != authority.get())
            throw std::invalid_argument("Local ready storage changed its original prepared GPU authority");
        auto root = std::shared_ptr<PooledImmutableLocalHotReadyRoot>(
            new PooledImmutableLocalHotReadyRoot(consumer,std::move(authority)));
        root->region_ = transfers.allocateMappedHostRegion(*root,root->authority_);
        return root;
    }

    ImmutableLookupNativeOwnerObservation PooledImmutableLocalHotReadyRoot::retainedOwnerObservation() const
    {
        if (!region_ || !region_->isBound() || !region_->hasDevice(consumer_.device))
            throw std::logic_error("Local ready observation requires its actual retained mapped owner");
        return {authority_,{consumer_.world_rank,DeviceId::cpu()},0u,region_->sizeBytes()};
    }
}
