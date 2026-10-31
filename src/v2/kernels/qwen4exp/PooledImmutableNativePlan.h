/**
 * @file PooledImmutableNativePlan.h
 * @brief Exact native adapter BOMs and complete immutable launch geometry.
 *
 * These factories project declared metadata, not source/Rank/materialization
 * permissions. They make no capacity choice and maintain no resource balance.
 * Existing frozen owner/coordinator BOMs remain separate inputs to the sole PMA.
 */
#pragma once
#include "PooledImmutableNativeTypes.h"
namespace llaminar2::qwen4exp
{
/** @brief Closed native owner geometry and only its additional mandatory staging contribution. */
class PooledImmutableOwnerNativePlan final
{
public:
    /**
     * @brief Project exact native owner metadata from the existing frozen geometry.
     * @param physical Original checked owner order and unchanged source extent.
     * @param coordinator Matching shared coordinator geometry with an explicit original-ID bound.
     * @param coordinate Original storage-owner coordinate, including C0.
     * @param table Original table identity from the actual source binder; this metadata is not a capability.
     * @param storage_generation Original source storage generation, distinct from source I/O sequences.
     * @return Exact native geometry on the declared owner allocator.
     */
    static PooledImmutableOwnerNativePlan fromGeometry(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,
        std::uint64_t table,std::uint64_t storage_generation);
    /** @return Complete native geometry for argument/capture identity. */
    const pooled_native::OwnerParameters& parameters() const noexcept{return parameters_;}
    /** @return Exact original allocator, independent of coordinator/consumer placement. */
    PhysicalMemoryAllocatorIdentity allocator() const noexcept{return allocator_;}
    /** @return Additional native bytes only; frozen cache/frame bytes are not charged twice. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource observed) const;
    /**
     * @brief Price the same canonical native layout before source/channel materialization creates its actual service identity.
     * @param physical Original checked physical owner geometry.
     * @param coordinator Matching independent policy/coalescing declaration.
     * @param coordinate Original storage-owner coordinate, including C0.
     * @param observed Exact original allocator observation for the sole PMA.
     * @return Typed staging BOM only; no placeholder source identity or runtime permission is created.
     */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate,PhysicalMemoryResource observed);
private:
    /** @brief Only fromGeometry constructs a complete native declaration. */
    PooledImmutableOwnerNativePlan(pooled_native::OwnerParameters p,PhysicalMemoryAllocatorIdentity a)
        :parameters_(p),allocator_(a){}
    pooled_native::OwnerParameters parameters_;
    PhysicalMemoryAllocatorIdentity allocator_;
};
/** @brief Closed native coordinator geometry and only its additional mandatory staging contribution. */
class PooledImmutableCoordinatorNativePlan final
{
public:
    /**
     * @brief Project exact native coordinator roots without owner=consumer or vendor assumptions.
     * @param physical Original checked explicit owner/coordinator order.
     * @param coordinator Matching original policy/coalescing geometry.
     * @param table Original table identity from the future actual source binder.
     * @param storage_generation Original immutable source storage generation.
     * @return Native argument/BOM declaration on the exact independent coordinator.
     */
    static PooledImmutableCoordinatorNativePlan fromGeometry(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint64_t table,std::uint64_t storage_generation);
    /** @return Complete native geometry. */
    const pooled_native::CoordinatorParameters& parameters() const noexcept{return parameters_;}
    /** @return Exact original coordinator allocator. */
    PhysicalMemoryAllocatorIdentity allocator() const noexcept{return allocator_;}
    /** @return Original immutable owner geometry copied only during setup; never a mutable residency shadow. */
    std::span<const adaptive_lookup::pooled::CoordinatorOwnerGeometry> owners() const{return owners_;}
    /** @return Only additional native staging, with all frozen roots contributed separately once. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource observed) const;
    /**
     * @brief Price the same independent native coordinator layout before the actual service ID exists.
     * @param physical Original declared owner/coordinator geometry.
     * @param coordinator Matching policy and original-ID bound.
     * @param observed Same independent coordinator resource supplied to the sole PMA.
     * @return Exact staging BOM; source identities are bound only after actual service creation.
     */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,PhysicalMemoryResource observed);
private:
    /** @brief Only fromGeometry constructs checked native staging and owner-order metadata. */
    PooledImmutableCoordinatorNativePlan(pooled_native::CoordinatorParameters p,
        PhysicalMemoryAllocatorIdentity a,std::vector<adaptive_lookup::pooled::CoordinatorOwnerGeometry> owners)
        :parameters_(p),allocator_(a),owners_(std::move(owners)){}
    pooled_native::CoordinatorParameters parameters_;
    PhysicalMemoryAllocatorIdentity allocator_;
    std::vector<adaptive_lookup::pooled::CoordinatorOwnerGeometry> owners_;
};
}
