/**
 * @file PooledImmutableFactoryReplayAccess.h
 * @brief Closed recorded claims consumed by original pooled graph factory bodies.
 *
 * These fixtures retain ordinary hardware-free pool workers, never fabricate an
 * admitted PreparedBirth. Every group uses the same final PMA and original pool
 * generation, and all native API calls remain explicit test doubles.
 */
#pragma once
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "kernels/qwen4exp/PooledImmutableNativeStorage.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
namespace llaminar2::test {
struct PooledImmutableFactoryReplayAccess final {
    /**
     * @brief Withdraw only the recorded boundary's weak authority to replay missing terminal proof.
     * @param boundary Original fixture boundary; its actual cohort owner remains retained by the test.
     */
    static void withdrawCohortForTerminalReplay(qwen4exp::ImmutableLookupExecutionBoundary &boundary)
    {
        boundary.completeCohort_.store(
            std::weak_ptr<qwen4exp::PooledImmutableRetirementAuthority>{}, std::memory_order_release);
    }

    /** @brief Append the actual plan's closed program using sealed recorded observations. */
    static void addOrdering(PhysicalMemoryPlanBuilder& builder,NativeOrderingEventProgram program,
        PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        const std::array devices{gpu.device};const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices);
        const auto contribution=projection.bom(program,host,gpu);
        for(const auto& line:contribution.resources())builder.add(line);
    }
    /** @return A closed claim bound to the actual ordinary worker/control block and original same-PMA generation. */
    static auto claim(const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device,NativeOrderingEventProgram program){
        auto& pool=GPUDeviceContextPool::instance();(void)pool.getContext(device);
        const auto original=pool.retainExistingContextProvenance(device);const std::array devices{device};
        return NativeOrderingEventGroupReplayAccess::claim(memory,original,program,
            NativeOrderingEventGroupReplayAccess::projection(devices));
    }
    /** @return Same owner body for adversarial closed-claim provenance/program replay. */
    static auto ownerClaim(qwen4exp::PooledImmutableOwnerNativePlan plan,std::shared_ptr<PhysicalMemoryAuthority> memory,
        void* stream,std::unique_ptr<NativeOrderingEventGroup> group){
        if(!group)throw std::invalid_argument("Recorded owner requires its preclaimed closed group");
        return qwen4exp::PooledImmutableOwnerPreparedStorage::createImpl(std::move(plan),std::move(memory),stream,std::move(group));
    }
    /** @return The production owner factory body with its exact original preclaimed group. */
    static auto owner(qwen4exp::PooledImmutableOwnerNativePlan plan,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream){
        auto group=claim(memory,plan.allocator().device,plan.orderingProgram());
        return ownerClaim(std::move(plan),std::move(memory),stream,std::move(group));
    }
    /** @return The production coordinator factory body with its exact original preclaimed group. */
    static auto coordinator(qwen4exp::PooledImmutableCoordinatorNativePlan plan,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream){
        auto group=claim(memory,plan.allocator().device,plan.orderingProgram());
        return qwen4exp::PooledImmutableCoordinatorPreparedStorage::createImpl(std::move(plan),std::move(memory),stream,std::move(group));
    }
    /** @return The unchanged full source/boundary creator program with both independently priced groups. */
    static auto boundary(std::shared_ptr<qwen4exp::ImmutableLookupService> service,qwen4exp::PooledImmutableSourceGeometry geometry,
        std::vector<std::shared_ptr<qwen4exp::PooledImmutableOwnerPreparedStorage>> owners,
        std::shared_ptr<qwen4exp::PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream){
        auto source=claim(memory,geometry.allocator().device,geometry.orderingProgram());
        auto terminal=claim(memory,geometry.allocator().device,NativeOrderingEventProgram::lookupSubmissionRetirement());
        return qwen4exp::ImmutableLookupExecutionBoundary::createPooledImpl(std::move(service),std::move(geometry),std::move(owners),
            std::move(coordinator),std::move(memory),stream,std::move(source),std::move(terminal));
    }
    /** @brief Retire only generations in the isolated fixture's declared two-backend/eight-ordinal mock population.
     * All graph/storage/backend aliases must already have drained before this last-owner scope runs. */
    static void retireFixtureGenerations(){
        for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(int ordinal=0;ordinal<8;++ordinal){
            auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement({type,ordinal});
            if(reset.receipt().retiredLiveContext())NativeOrderingEventGroupReplayAccess::validatedReset(reset);
        }
    }
};
}
