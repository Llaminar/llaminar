/**
 * @file AdaptiveImmutableLookupGeometry.h
 * @brief Exact single-GPU split-owner/coordinator/source BOM through the existing PMA candidate.
 *
 * The original resolved layout remains the sole source/slot declaration. Each
 * physical root is priced by its own native factory before source IDs exist.
 * C0 execution is fixed before optional admission; a candidate adds only exact
 * root growth and optional cache roots. No aggregate owner line is removed,
 * physical capacity subtracted, or alternative live ledger maintained here.
 */
#pragma once
#include "CpuAdaptiveImmutableRowCache.h"
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>
namespace llaminar2
{
/** @brief Source-owned exact physical root projections; neither admission nor runtime residency state. */
class AdaptiveImmutableLookupGeometry final
{
public:
    /**
     * @brief Project the existing admitted single-GPU policy onto separated native authorities.
     * @param layout Original source/owner/consumer layout, including exact selected slots.
     * @param source Original native source metadata, never payload or a fabricated runtime identity.
     * @param request Original bounded GPU/RAM policies.
     * @return Exact immutable roots and their C0 physical extent partitions.
     * @throws std::exception For incompatible source, participants, policy or extents.
     */
    static AdaptiveImmutableLookupGeometry fromLayout(const ImmutableRowTableLayoutGeometry& layout,
        const ImmutableRowTableSource& source,const AdaptiveVRAMRAMSSDRequest& request)
    {
        if(layout.consumers().size()!=1 || !layout.consumers().front().device.is_gpu() ||
           layout.sourceServices().size()!=1 || !layout.maximumInflightRows() || layout.maximumInflightRows()%16 ||
           layout.maximumInflightRows()>qwen4exp::adaptive_lookup::MaximumSlots)
            throw std::invalid_argument("Adaptive lookup requires one original GPU/source domain and complete sixteen-head query geometry");
        const auto requester=layout.consumers().front();
        const std::array order{requester};
        const auto physical=qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,requester);
        const qwen4exp::adaptive_lookup::Policy policy{request.gpu_policy.aging_accesses,
            request.gpu_policy.protected_numerator,request.gpu_policy.protected_denominator};
        const auto coordinator=qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(physical,policy,
            static_cast<std::uint32_t>(layout.maximumInflightRows()));
        const auto host=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,
            layout.sourceServices().front().allocator,requester,request.host.policy);
        const auto packets=qwen4exp::PooledImmutableSourceGeometry::rootOffsets(physical,coordinator,0,source);
        // A temporary C0 declaration derives the fixed portion of these SAME
        // roots. It creates no new placement registry, permit or allocation.
        auto owners=std::vector<ImmutableRowTableLayoutOwner>(layout.owners().begin(),layout.owners().end());
        for(auto& owner:owners)if(owner.key.tier==ImmutableRowStorage::VRAM)owner.cache_rows=0;
        const auto zero=ImmutableRowTableLayoutGeometry::checked(layout.extent(),layout.tiers(),layout.residency(),
            std::move(owners),{requester},{},{layout.sourceServices().begin(),layout.sourceServices().end()},
            {},std::nullopt,layout.maximumInflightRows());
        const auto floor=qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(zero,order,requester);
        const auto floorPolicy=qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(floor,policy,
            static_cast<std::uint32_t>(layout.maximumInflightRows()));
        if(physical.owners().front().frame.bytes!=floor.owners().front().frame.bytes ||
           coordinator.frameOffsets().bytes!=floorPolicy.frameOffsets().bytes)
            throw std::logic_error("Adaptive fixed frame declarations unexpectedly depend on optional slot capacity");
        return {physical,coordinator,host,packets,
            qwen4exp::PooledImmutableOwnerNativePlan::rootOffsets(physical,coordinator,0).bytes,
            qwen4exp::PooledImmutableCoordinatorNativePlan::rootOffsets(physical,coordinator).bytes,
            {qwen4exp::PooledImmutableOwnerNativePlan::rootOffsets(floor,floorPolicy,0).bytes,
             qwen4exp::PooledImmutableCoordinatorNativePlan::rootOffsets(floor,floorPolicy).bytes,
             qwen4exp::PooledImmutableSourceGeometry::rootOffsets(floor,floorPolicy,0,source).frame_bytes}};
    }
    /** @return Original single-GPU physical slot and grant-root declaration. */
    const qwen4exp::PooledImmutablePhysicalOwnerGeometry& physical()const noexcept{return physical_;}
    /** @return Independent policy/frame declaration on the original coordinator. */
    const qwen4exp::PooledImmutableCoordinatorGeometry& coordinator()const noexcept{return coordinator_;}
    /** @return Original bounded CPU RAM/SSD provider declaration. */
    const CpuAdaptiveImmutableRowCacheGeometry& host()const noexcept{return host_;}
    /** @return Original complete ID envelope; runtime counts remain device-owned. */
    std::uint32_t maximumIds()const noexcept{return coordinator_.parameters().maximumOriginal();}
    /** @return Exact source protocol2 envelope, without another mapped channel owner. */
    qwen4exp::ImmutableLookupLayout channelLayout()const{
        return qwen4exp::ImmutableLookupLayout::createSubordinate(coordinator_.parameters().maximumUnique(),physical_.rowBytes());
    }
    /** @return Full optional native owner payload/slot AND independent policy roots, excluding staging. */
    std::size_t gpuCacheBytes()const{return plus(physical_.owners().front().cache.bytes,coordinator_.policyOffsets().bytes);}
    /** @return Complete bytes charged to the explicit optional GPU limit, including exact staging growth. */
    std::size_t gpuSlotDependentBytes()const{
        std::size_t bytes=gpuCacheBytes();
        for(const auto growth:rootGrowth())bytes=plus(bytes,growth);
        return bytes;
    }
    /** @return Exact complete native staging root requests, including actual capacity-dependent banks. */
    std::size_t mandatoryDeviceBytes()const{
        std::size_t total=0;
        for(const auto bytes:{physical_.owners().front().frame.bytes,coordinator_.frameOffsets().bytes,
            owner_native_,coordinator_native_,source_.frame_bytes,source_.native_bytes})total=plus(total,bytes);
        return total;
    }
    /**
     * @brief Price every actual candidate execution/source root, including its selected bank sizes.
     * @param gpu Original PMA GPU observation. @param host Original CPU source observation.
     * @return Full exact mandatory candidate BOM; at C0 this is the fixed pre-quota floor.
     */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource gpu,PhysicalMemoryResource host)const{
        return mandatory(gpu,host,{owner_native_,coordinator_native_,source_.frame_bytes});
    }
    /**
     * @brief Price only optional native GPU/policy and CPU cache roots.
     * @param gpu Original exact GPU observation. @param host Original CPU observation.
     * @return Typed roots; mandatory staging is contributed separately.
     */
    PhysicalMemoryPlan cacheBOM(PhysicalMemoryResource gpu,PhysicalMemoryResource host)const{
        PhysicalMemoryPlanBuilder result;
        for(const auto& plan:{physical_.cacheBOM(0,gpu),coordinator_.policyBOM(gpu),host_.cacheBOM(host)})
            for(const auto& line:plan.resources())result.add(line);
        return result.build();
    }
    /**
     * @brief Add exact optional roots and slot-dependent physical bank portions to the frozen C0 floor.
     * @param gpu Same original PMA GPU observation. @param host Same CPU observation.
     * @return Complete candidate continuation contribution; PMA alone evaluates fit.
     * @note Differences partition exact source-authored physical root declarations.
     * They never remove aggregate owner charges or subtract from available memory.
     */
    PhysicalMemoryPlan slotDependentBOM(PhysicalMemoryResource gpu,PhysicalMemoryResource host)const{
        PhysicalMemoryPlanBuilder result(cacheBOM(gpu,host));
        for(const auto growth:rootGrowth())result.add(gpu,PhysicalMemoryOwner::ImmutableLookupStaging,growth);
        return result.build();
    }
private:
    /** @return The three exact declared physical root portions beyond C0, with no capacity observation. */
    std::array<std::size_t,3> rootGrowth()const{
        std::array actual{owner_native_,coordinator_native_,source_.frame_bytes};
        for(std::size_t root=0;root<actual.size();++root){
            if(actual[root]<floor_[root])throw std::logic_error("Adaptive physical root is smaller than its original C0 declaration");
            actual[root]-=floor_[root];
        }
        return actual;
    }
    /** @return Checked physical extent sum; never physical capacity or a live balance. */
    static std::size_t plus(std::size_t a,std::size_t b){
        if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("Adaptive physical root extent overflows");
        return a+b;
    }
    /** @brief Retain original declarations and three exact native root partitions. */
    AdaptiveImmutableLookupGeometry(qwen4exp::PooledImmutablePhysicalOwnerGeometry physical,
        qwen4exp::PooledImmutableCoordinatorGeometry coordinator,CpuAdaptiveImmutableRowCacheGeometry host,
        qwen4exp::PooledSourceOffsets source,std::size_t ownerNative,std::size_t coordinatorNative,std::array<std::size_t,3> floor)
        :physical_(std::move(physical)),coordinator_(std::move(coordinator)),host_(std::move(host)),source_(source),
         owner_native_(ownerNative),coordinator_native_(coordinatorNative),floor_(floor){}
    /** @return Full fixed-frame/source/channel BOM with exact supplied native root extents.
     * @param gpu Original observed GPU allocator. @param host Original CPU source observation.
     * @param roots Exact owner-native, coordinator-native and source-frame requests. */
    PhysicalMemoryPlan mandatory(PhysicalMemoryResource gpu,PhysicalMemoryResource host,std::array<std::size_t,3> roots)const{
        PhysicalMemoryPlanBuilder result;
        for(const auto& plan:{physical_.mandatoryBOM(0,gpu),coordinator_.mandatoryBOM(gpu),host_.mandatoryBOM(host)})
            for(const auto& line:plan.resources())result.add(line);
        for(const auto bytes:roots)result.add(gpu,PhysicalMemoryOwner::ImmutableLookupStaging,bytes);
        result.add(gpu,PhysicalMemoryOwner::ImmutableLookupStaging,source_.native_bytes);
        result.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,qwen4exp::ImmutableLookupService::hostBytes(channelLayout()));
        return result.build();
    }
    qwen4exp::PooledImmutablePhysicalOwnerGeometry physical_;
    qwen4exp::PooledImmutableCoordinatorGeometry coordinator_;
    CpuAdaptiveImmutableRowCacheGeometry host_;
    qwen4exp::PooledSourceOffsets source_;
    std::size_t owner_native_,coordinator_native_;
    std::array<std::size_t,3> floor_; ///< Immutable physical root declarations, not a physical-byte authority.
};
}
