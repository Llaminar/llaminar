/**
 * @file PooledImmutableCoordinatorGeometry.h
 * @brief Exact sole-PMA contributions for one global policy and live two-wave coordinator frame.
 *
 * Policy entries name physical locators but never reserve cache slots or bytes.
 * Source, transport, payload and native executable storage remain separate typed
 * inputs to the same PhysicalMemoryAuthority. No serving mode is installed here.
 */
#pragma once
#include "PooledImmutablePhysicalOwnerGeometry.h"
#include "PooledImmutableCoordinatorState.h"
namespace llaminar2::qwen4exp
{
/** @brief Disjoint optional policy subranges on the independent declared coordinator allocator. */
struct PooledCoordinatorOffsets
{
    std::size_t control = 0, entries = 0, nodes = 0, epochs = 0, counts = 0, bytes = 0;
};
/** @brief Mandatory exact K/N/P roots; original-ID work is priced independently of distinct rows. */
struct PooledCoordinatorFrameOffsets
{
    std::size_t control = 0, unique = 0, sort_a = 0, sort_b = 0, radix_counts = 0;
    std::size_t original_to_unique = 0, owner_plans = 0, decisions = 0, owners = 0, bytes = 0;
};
/** @brief Immutable checked geometry; it neither admits an allocation nor authenticates a native cohort. */
class PooledImmutableCoordinatorGeometry final
{
public:
    /**
     * @brief Project exact ordered owners and explicit original-ID admission bound without selecting capacity.
     * @param owners Original checked physical-owner declaration, including C0 and independent coordinator.
     * @param policy Explicit shared frequency/queue policy.
     * @param maximumOriginalIds Admitted original logical IDs, including duplicates; at least the distinct-row bound.
     * @return Exact metadata and native geometry contributed once to the sole PMA.
     * @throws std::invalid_argument For incompatible policy/bounds or unsupported index geometry.
     * @throws std::overflow_error For any nonrepresentable byte extent.
     */
    static PooledImmutableCoordinatorGeometry fromOwners(const PooledImmutablePhysicalOwnerGeometry &owners,
        adaptive_lookup::Policy policy, std::uint32_t maximumOriginalIds);
    /** @return Exact declared coordinator allocator; this metadata grants no allocation permission. */
    PhysicalMemoryAllocatorIdentity coordinator() const noexcept { return coordinator_; }
    /** @return Native metadata projection without mutable residency state. */
    adaptive_lookup::pooled::CoordinatorParameters parameters() const { return parameters_; }
    /** @return Immutable exact owner-order geometry; production must retain/materialize its own native copy. */
    std::span<const adaptive_lookup::pooled::CoordinatorOwnerGeometry> owners() const { return owners_; }
    /** @return Optional policy root only; physical owner bytes are contributed separately. */
    PooledCoordinatorOffsets policyOffsets() const { return policy_; }
    /** @return Mandatory frame and coalescing roots, including C0/U0. */
    PooledCoordinatorFrameOffsets frameOffsets() const { return frame_; }
    /**
     * @brief Contribute optional policy bytes to the exact coordinator allocator.
     * @param observed Canonical PMA resource observation for that allocator.
     * @return Typed policy BOM without admission or allocation decisions.
     * @throws std::invalid_argument For a foreign/invalid allocator observation.
     */
    PhysicalMemoryPlan policyBOM(PhysicalMemoryResource observed) const;
    /**
     * @brief Contribute mandatory K/N/P frame bytes to the exact coordinator allocator.
     * @param observed Canonical PMA resource observation for that allocator.
     * @return Typed staging BOM; source/channel/transport storage remains separately owned.
     * @throws std::invalid_argument For a foreign/invalid allocator observation.
     */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource observed) const;
private:
    /** @brief Construct complete immutable geometry only through fromOwners. */
    PooledImmutableCoordinatorGeometry() = default;
    /** @brief Reject foreign resource observations before constructing a PMA contribution. */
    void require(PhysicalMemoryResource observed) const;
    PhysicalMemoryAllocatorIdentity coordinator_{};
    adaptive_lookup::pooled::CoordinatorParameters parameters_{};
    std::vector<adaptive_lookup::pooled::CoordinatorOwnerGeometry> owners_;
    PooledCoordinatorOffsets policy_{};
    PooledCoordinatorFrameOffsets frame_{};
};
}
