/**
 * @file FollowerPredecessorDiagnosticGraphResources.h
 * @brief Original-PMA General backing for two raw diagnostic capture streams.
 *
 * This component fixture exercises the original cache/event/epoch bodies. It
 * does not claim RetainedParticipantGraphExecutable source/owner coverage. Each
 * actual raw creator has its own original General host/device reservation,
 * transferred to the existing ProcessRuntime retirement owner before creation.
 * The known native wrapper object has a separate exact allocation claim.
 */
#pragma once
#include "backends/GPUGraphMemoryContract.h"
#include "backends/GPUFlatGraphStructure.h"
#include "backends/NativeContextPreparation.h"
#include "backends/NativeContextRetirement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <atomic>
#include <limits>

namespace llaminar2::test {
/** @brief Closed fixture ownership of the two actual diagnostic General creators. */
struct FollowerPredecessorDiagnosticGraphResources final {
    /**
     * @brief Price exactly the fixture's two raw General graph definitions.
     * @param host Original same-rank CPU allocator observation.
     * @param gpu Original addressed GPU observation.
     * @return Canonical General host/device units and exact native-wrapper bytes.
     */
    static PhysicalMemoryPlan bom(PhysicalMemoryResource host,PhysicalMemoryResource gpu) {
        if(!host.valid() || host.device!=DeviceId::cpu() || !gpu.valid() ||
            !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
            throw std::invalid_argument("Diagnostic General graphs require original same-rank CPU/GPU observations");
        PhysicalMemoryPlanBuilder plan;
        for(unsigned index=0;index<2u;++index) {
            plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(gpu.device));
            plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,nativeGraphCaptureObjectBytes(gpu.device));
            plan.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(gpu.device,GPUGraphExecutableClass::General));
        }
        return plan.build();
    }

    /**
     * @brief Transfer existing canonical reservations before one actual raw creator.
     * @param original Actual admitted original source/PMA/worker capability.
     * @return Sole retirement slot; caller arms immediately before native creation
     * and completes only after proven terminal observation and handle destruction.
     * No certificate or runtime authority is manufactured by this diagnostic.
     */
    static NativeContextRetirementSlot prepare(const PreparedGPUContextUsePermit& original) {
        const auto memory=original.memory();const auto worker=original.worker();
        if(!memory || !worker || !original.device().is_gpu() ||
            memory->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("Diagnostic General backing requires its actual admitted original source");
        static std::atomic<std::uint64_t> next{1u};
        auto instance=next.load(std::memory_order_relaxed);
        do {
            if(instance==std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Diagnostic General process population exhausted");
        } while(!next.compare_exchange_weak(instance,instance+1u,std::memory_order_relaxed));
        const std::string policy="diagnostic-follower-predecessor-original-general/v1";
        auto device=memory->reserveNewAllocations(original.device(),PhysicalMemoryOwner::NativeGraphExecutable,
            GPUGraphMemoryContract::reservationBytesPerExecutable(original.device(),GPUGraphExecutableClass::General));
        auto host=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,
            GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(original.device()));
        auto slot=original.provenance().reserveNativeRetirementSlot(memory,policy);
        slot.prepareProcessRuntime("diagnostic-follower-predecessor-general-instance:"+std::to_string(instance));
        slot.adopt(std::make_unique<Obligation>(memory,original.device(),original.generation(),policy,
            std::move(device),std::move(host)));
        return slot;
    }
private:
    /** @brief Only original PMA tokens; actual OS reclamation owns their physical release. */
    class Obligation final:public IContextNativeRetirementObligation {
    public:
        /** @brief Move the actual original General reservations without copying accounting. */
        Obligation(std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,std::uint64_t generation,
            std::string policy,PhysicalMemoryOwnerReservation gpu,PhysicalMemoryOwnerReservation host)
            :memory_(std::move(memory)),device_(device),generation_(generation),policy_(std::move(policy)),
             gpu_(std::move(gpu)),host_(std::move(host)) {}
        /** @copydoc IContextNativeRetirementObligation::originalAuthority */
        const std::shared_ptr<PhysicalMemoryAuthority>& originalAuthority()const noexcept override{return memory_;}
        /** @copydoc IContextNativeRetirementObligation::device */
        DeviceId device()const noexcept override{return device_;}
        /** @copydoc IContextNativeRetirementObligation::poolGeneration */
        std::uint64_t poolGeneration()const noexcept override{return generation_;}
        /** @copydoc IContextNativeRetirementObligation::certificateIdentity */
        const std::string& certificateIdentity()const noexcept override{return policy_;}
        /** @copydoc IContextNativeRetirementObligation::physicalLifetime */
        NativePhysicalLifetime physicalLifetime()const noexcept override{return NativePhysicalLifetime::ProcessRuntime;}
    private:
        std::shared_ptr<PhysicalMemoryAuthority> memory_;
        DeviceId device_;
        std::uint64_t generation_;
        std::string policy_;
        PhysicalMemoryOwnerReservation gpu_,host_;
    };
};
} // namespace llaminar2::test
