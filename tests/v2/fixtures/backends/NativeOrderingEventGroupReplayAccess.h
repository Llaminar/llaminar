/** @file NativeOrderingEventGroupReplayAccess.h
 * @brief Recorded immutable event-budget inputs for actual production arithmetic/lifetime replay.
 * No production table is changed and these synthetic bounds never certify native memory.
 */
#pragma once
#include "backends/NativeOrderingEventGroup.h"
#include "backends/NativeOrderingEventBudgetProjection.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
namespace llaminar2::test {
struct NativeOrderingEventGroupReplayAccess {
    /** @return Sealed recorded rows for precisely the caller's fixture device observations. */
    static NativeOrderingEventBudgetProjection projection(std::span<const DeviceId> devices,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration){return projectionImpl(devices,lifetime,false);}
    /** @return Explicit distinct graph-family metadata as well as event rows; this grants no native graph permission. */
    static NativeOrderingEventBudgetProjection projectionWithRemoteGraphs(std::span<const DeviceId> devices,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration){return projectionImpl(devices,lifetime,true);}
    /** @brief One private fixture decoder; graph costs are independent from ordering-event caps. */
    static NativeOrderingEventBudgetProjection projectionImpl(std::span<const DeviceId> devices,NativePhysicalLifetime lifetime,bool graphs){
        std::vector<NativeOrderingEventBudgetProjection::RecordedBound> rows;
        const std::array programs{NativeOrderingEventProgram::pooledOwner(false),NativeOrderingEventProgram::pooledOwner(true),
            NativeOrderingEventProgram::pooledCoordinator(false),NativeOrderingEventProgram::pooledCoordinator(true),NativeOrderingEventProgram::pooledSource(),
            NativeOrderingEventProgram::lookupSubmissionRetirement(),NativeOrderingEventProgram::readAheadParentTerminal()};
        for(const auto device:devices)for(const auto program:programs){
            const auto n=program.orderingEvents();
            // Explicit recorded nonlinear bounds exercise growth, not an inferred native per-event price.
            const auto host=n==1?1024u:n==3?3072u:5120u;
            rows.push_back({device,program,host,host*2,std::string(64,'a'),std::string(64,'b'),lifetime});}
        std::vector<NativeOrderingEventBudgetProjection::RecordedRemoteBound> remote;
        if(graphs)for(const auto device:devices){
            const NativeAuxiliaryRemoteFamilyScope family{PooledRemoteGraphProgramClass::familyVersion(),
                {4096,4095,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)},PooledRemoteGraphProgramClass::argumentLayout()};
            // Synthetic graph trace, not an estimate from event counts or a native certificate.
            remote.push_back({device,4096,sizeof(testing::MockGPUGraphCapture),std::string(64,'c'),std::string(64,'d'),family});
        }
        return NativeOrderingEventBudgetProjection::fromRecorded(std::move(rows),std::move(remote));
    }
    /** @return Actual group claims using the sealed recorded input source. */
    static std::unique_ptr<NativeOrderingEventGroup> claim(std::shared_ptr<PhysicalMemoryAuthority> memory,const GPUContextProvenance& original,NativeOrderingEventProgram program,const NativeOrderingEventBudgetProjection& projection){
        return NativeOrderingEventGroup::claimWithProjection(std::move(memory),original,program,projection);
    }
    static NativeAuxiliaryResourceSelection select(const NativeContextBirthKey& key,std::uint64_t generation,NativeAuxiliaryResourceProgram program,std::string_view resource,std::span<const detail::NativeAuxiliaryQualifiedClass> rows){return NativeAuxiliaryResourceCatalog::select(key,generation,program,resource,rows);}
    static void begin(NativeOrderingEventGroup& group){group.beginNativeAttempt();}
    static void complete(NativeOrderingEventGroup& group)noexcept{group.completeNativeHandles();}
    static void requireOriginal(const NativeOrderingEventGroup& group,const std::shared_ptr<PhysicalMemoryAuthority>& memory,const GPUContextProvenance& original,NativeOrderingEventProgram program){group.requireOriginal(memory,original,program);}
    static void validatedReset(ExclusiveGPUDeviceGenerationRetirement& scope){NativeContextPreparationReplayAccess::physicalReset(scope);}
};
}
