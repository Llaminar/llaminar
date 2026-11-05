/**
 * @file PooledImmutableSourceGeometry.h
 * @brief Exact original RAM/SSD source response and accepted-owner delivery storage.
 *
 * Geometry contributes typed staging to the existing PMA; it is not source,
 * transport or allocation permission. The existing subordinate service binding
 * authenticates these declarations against its actual provider/channel before
 * any producer is created. Directed transport remains an explicit graph edge.
 * Reporting scratch preserves actual operation receipts before grant retirement;
 * the canonical terminal row is written only after every physical owner retires.
 */
#pragma once
#include <optional>
#include <type_traits>
#include <stdexcept>
#include "backends/NativeOrderingEventProgram.h"
#include "PooledImmutableNativePlan.h"
#include "tensors/TensorType.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include <vector>
#include <span>
namespace llaminar2 {struct ImmutableRowTableSource;}
namespace llaminar2::qwen4exp {
/** @brief Exact private source packet banks; only acquired live prefixes are consumed. */
struct PooledSourceOffsets {
    std::size_t coordinates=0,responses=0,invalidations=0,fills=0,payload=0,frame_bytes=0;
    std::size_t protocol=0,frontier=64,record_bytes=128,payload_bytes=136,invocation=144,response_stage=152;
    std::size_t observation=192,observation_ordinal=264,reported_ordinal=272,native_bytes=320;
    std::size_t progress=160,seal_receipt=0; ///< Serial acquired owner receipt, distinct from canonical source frontier.
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
    std::uint32_t local_owner=adaptive_lookup::None,packet_capacity=0;
    std::uint32_t consumers=1; ///< Actual compute readers; never physical owner or CPU source-channel population.
};
/** @brief Closed exact source/frame geometry independent of any host cache decision or mirrored GPU state. */
class PooledImmutableSourceGeometry final {
public:
    /**
     * @brief Price exact response and local delivery banks from original source and accepted native declarations.
     * @param owners Original storage-owner declarations, independent of consumer/coordinator placement.
     * @param coordinator Original global coalescing declaration.
     * @param source Actual immutable provider source metadata; payload is neither read nor hashed.
     * @param consumers Original declared compute-reader cohort, independent of the one CPU source channel.
     * @return Checked typed geometry; the source-issued binder separately proves the actual source/channel/roots.
     * @throws std::exception For incompatible format/table/source/count/owner geometry or overflow.
     */
    static PooledImmutableSourceGeometry fromPlans(std::span<const PooledImmutableOwnerNativePlan> owners,
        const PooledImmutableCoordinatorNativePlan& coordinator,const ImmutableRowTableSource& source,
        std::uint32_t consumers);
    /**
     * @brief Validate a complete one-owner cohort with an explicitly declared reader population.
     * @param owner Original sole storage-owner declaration.
     * @param coordinator Original global coalescing declaration.
     * @param source Actual immutable provider source metadata.
     * @param consumers Original compute-reader count, independent of the storage-owner count.
     * @return The same checked geometry produced by the complete-cohort overload.
     * @throws std::exception For a partial cohort or incompatible source geometry.
     */
    static PooledImmutableSourceGeometry fromPlans(const PooledImmutableOwnerNativePlan& owner,
        const PooledImmutableCoordinatorNativePlan& coordinator,const ImmutableRowTableSource& source,std::uint32_t consumers)
    {return fromPlans(std::span{&owner,1},coordinator,source,consumers);}
    /**
     * @brief Validate the same complete source identity used at materialization.
     * @param owners Original ordered storage owners, independent of compute readers.
     * @param coordinator Original coordinator policy and source generation.
     * @param source Original immutable source metadata.
     * @param expected_compute_readers Expected declared compute-reader population, never inferred from storage owners.
     * @throws std::exception For malformed declarations or a changed source/policy/cohort/reader identity.
     */
    void requireOriginalPlans(std::span<const PooledImmutableOwnerNativePlan> owners,
        const PooledImmutableCoordinatorNativePlan& coordinator,const ImmutableRowTableSource& source,
        std::uint32_t expected_compute_readers)const;
    /**
     * @brief Project only the original native roots physically retained on one allocator.
     * @param allocator Exact physical endpoint, independent of compute-reader entries.
     * @param source Read-only getter for the existing source/coordinator/local-owner aggregate.
     * @param owner Read-only getter for an original storage owner by immutable coordinate.
     * @return One unchanged observation or no native roots. No capacity or bytes are inferred.
     * @note Source selection comes first because its aggregate already includes any local owner.
     */
    template<class SourceObservation,class OwnerObservation>
    [[nodiscard]] auto projectRetainedOwnerObservation(PhysicalMemoryAllocatorIdentity allocator,
        SourceObservation&& source,OwnerObservation&& owner)const
    {
        using Observation=std::invoke_result_t<SourceObservation&>;
        static_assert(std::is_same_v<Observation,std::invoke_result_t<OwnerObservation&,std::size_t>>);
        const auto checked=[&](Observation observed){
            if(observed.allocator()!=allocator)
                throw std::logic_error("Adaptive retained observation has a foreign original allocator");
            return std::optional<Observation>{observed};
        };
        if(allocator==allocator_)return checked(source());
        for(std::size_t index=0;index<owners_.size();++index)
            if(owners_[index].allocator()==allocator)return checked(owner(index));
        return std::optional<Observation>{};
    }
    /** @return Original immutable ordered cohort; this is not an ownership/location registry. */
    std::span<const PooledImmutableOwnerNativePlan> originalOwners()const noexcept{return owners_;}
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
     * @param source Original immutable source metadata; no payload is read or permission minted.
     * @param resource Exact original coordinator resource observed by the sole PMA.
     * @return Typed mandatory staging only; runtime service/native identities are authenticated later.
     */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,const ImmutableRowTableSource& source,
        PhysicalMemoryResource resource);
    /** @return Exact identity-free physical source roots from the runtime factory's checked layout.
     * @param physical Original complete physical declaration. @param coordinator Matching coalescing policy.
     * @param source Original source metadata for the single complete-cohort source root.
     * @throws std::exception For incompatible source/owner declarations or byte overflow. */
    static PooledSourceOffsets rootOffsets(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,const ImmutableRowTableSource& source);
    /** @return Closed source setup plus its two original Tensor publication events. */
    NativeOrderingEventProgram orderingProgram()const;
    /**
     * @brief Derive the same closed source program before native materialization.
     * @param physical Original ordered physical owner geometry.
     * @param coordinator Matching complete coalescing policy.
     * @param source Original source metadata; no payload is read.
     * @return Setup plus publication of the exact frame/native roots.
     * @throws std::exception For incompatible source/cohort geometry.
     */
    static NativeOrderingEventProgram orderingProgram(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,const ImmutableRowTableSource& source);
    /** @return The existing protocol2 mapped-channel layout; hostBytes remains the existing service's authority. */
    ImmutableLookupLayout channelLayout()const;
private:
    /** @brief Only the checked factory constructs source geometry. */
    PooledImmutableSourceGeometry(PooledSourceParameters parameters,PhysicalMemoryAllocatorIdentity allocator,std::vector<PooledImmutableOwnerNativePlan> owners)
        :parameters_(parameters),allocator_(allocator),owners_(std::move(owners)){}
    PooledSourceParameters parameters_;
    PhysicalMemoryAllocatorIdentity allocator_;
    std::vector<PooledImmutableOwnerNativePlan> owners_;
};
}
