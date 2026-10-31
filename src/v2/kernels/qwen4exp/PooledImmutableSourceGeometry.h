/**
 * @file PooledImmutableSourceGeometry.h
 * @brief Exact original RAM/SSD source response and accepted-owner delivery storage.
 *
 * Geometry contributes typed staging to the existing PMA; it is not source,
 * transport or allocation permission. The existing subordinate service binding
 * authenticates these declarations against its actual provider/channel before
 * any producer is created. Directed transport remains an explicit graph edge.
 */
#pragma once
#include "PooledImmutableNativePlan.h"
#include "tensors/TensorType.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
namespace llaminar2 {struct ImmutableRowTableSource;}
namespace llaminar2::qwen4exp {
/** @brief Exact private source packet banks; only acquired live prefixes are consumed. */
struct PooledSourceOffsets {
    std::size_t coordinates=0,responses=0,invalidations=0,fills=0,payload=0,frame_bytes=0;
    std::size_t protocol=0,frontier=64,record_bytes=128,payload_bytes=136,invocation=144,response_stage=152,native_bytes=192;
    bool operator==(const PooledSourceOffsets&)const=default;
};
/** @brief Complete immutable source-native declarations; actual authority is retained by its service-issued binder. */
struct PooledSourceParameters {
    pooled_native::OwnerParameters owner;
    pooled_native::CoordinatorParameters coordinator;
    PooledSourceOffsets offsets;
    TensorType format;
    std::uint32_t row_elements;
    std::uint64_t identity;
};
/** @brief Closed exact source/frame geometry independent of any host cache decision or mirrored GPU state. */
class PooledImmutableSourceGeometry final {
public:
    /**
     * @brief Price exact response and local delivery banks from original source and accepted native declarations.
     * @param owner Original storage-owner declaration, independent of consumer/coordinator placement.
     * @param coordinator Original global coalescing declaration.
     * @param source Actual immutable provider source metadata; payload is neither read nor hashed.
     * @return Checked typed geometry; the source-issued binder separately proves the actual source/channel/roots.
     * @throws std::exception For incompatible format/table/source/count/owner geometry or overflow.
     */
    static PooledImmutableSourceGeometry fromPlans(const PooledImmutableOwnerNativePlan& owner,
        const PooledImmutableCoordinatorNativePlan& coordinator,const ImmutableRowTableSource& source);
    /** @return Complete frozen native operands, including source precision and exact physical offsets. */
    const PooledSourceParameters& parameters()const noexcept{return parameters_;}
    /** @return Original coordinator allocator that owns the source request/private response staging. */
    PhysicalMemoryAllocatorIdentity allocator()const noexcept{return allocator_;}
    /**
     * @brief Contribute only new exact GPU source staging, never provider/channel/table roots already priced elsewhere.
     * @param resource Original observed coordinator resource from the same PMA plan.
     * @return Typed mandatory staging; no reservation or capacity decision occurs.
     * @throws std::invalid_argument For another allocator.
     */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource resource)const;
    /**
     * @brief Price the same exact source staging before the actual retained service identity exists.
     * @param physical Original complete independent owner/coordinator geometry.
     * @param coordinator Matching global policy and original-ID admission bound.
     * @param coordinate Actual owner edge whose immutable packet capacity is declared.
     * @param source Original immutable source metadata; no payload is read or permission minted.
     * @param resource Exact original coordinator resource observed by the sole PMA.
     * @return Typed mandatory staging only; runtime service/native identities are authenticated later.
     */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,const ImmutableRowTableSource& source,
        PhysicalMemoryResource resource);
    /** @return The existing protocol2 mapped-channel layout; hostBytes remains the existing service's authority. */
    ImmutableLookupLayout channelLayout()const;
private:
    /** @brief Only the checked factory constructs source geometry. */
    PooledImmutableSourceGeometry(PooledSourceParameters parameters,PhysicalMemoryAllocatorIdentity allocator)
        :parameters_(parameters),allocator_(allocator){}
    PooledSourceParameters parameters_;
    PhysicalMemoryAllocatorIdentity allocator_;
};
}
