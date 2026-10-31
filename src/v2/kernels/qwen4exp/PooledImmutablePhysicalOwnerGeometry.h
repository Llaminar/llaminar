/**
 * @file PooledImmutablePhysicalOwnerGeometry.h
 * @brief Checked source-owned BOM for a pooled physical-owner prerequisite.
 *
 * This metadata projects an existing checked layout and explicitly supplied
 * ordered owner cohort. It neither proves actual Rank membership nor selects
 * a serving mode, source service, global policy or physical allocation. PMA is
 * the sole admission/materialization ledger. Coordinator directory/policy,
 * source channel and ragged transport executable BOMs are separate mandatory
 * prerequisites and are not claimed complete by this physical-only geometry.
 */
#pragma once
#include "PooledImmutablePhysicalOwnerState.h"
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include <optional>
#include <span>
#include <vector>

namespace llaminar2::qwen4exp
{
/** @brief Exact named root subranges; zero optional capacity has no optional root. */
struct PooledPhysicalOwnerOffsets
{
    std::size_t control = 0, slots = 0, payload = 0, bytes = 0;
    bool operator==(const PooledPhysicalOwnerOffsets &) const = default;
};

/** @brief Mandatory frame and bounded command/reply operands, not communication extents. */
struct PooledPhysicalFrameOffsets
{
    std::size_t control = 0, commands = 0, fill_order = 0, replies = 0, acknowledgements = 0;
    std::size_t allocation_control = 0, accepted_plan = 0, initial_pins = 0;
    std::size_t validation_a = 0, validation_b = 0, validation_counts = 0;
    std::size_t pin_headers = 0, publish_headers = 0, free_headers = 0, victim_headers = 0, release_receipts = 0, bytes = 0;
    bool operator==(const PooledPhysicalFrameOffsets &) const = default;
};

/** @brief Immutable per-coordinate physical declaration; contains no live residency counters. */
struct PooledPhysicalOwnerDeclaration
{
    PhysicalMemoryAllocatorIdentity allocator;
    std::uint32_t coordinate, slots;
    PooledPhysicalOwnerOffsets cache;
    PooledPhysicalFrameOffsets frame;
    std::optional<std::size_t> physical_byte_limit; ///< Original optional-cache ceiling, not another budget.
    bool operator==(const PooledPhysicalOwnerDeclaration &) const = default;
};

/** @brief Source geometry only, shared by later CPU/CUDA/HIP materialization wrappers. */
class PooledImmutablePhysicalOwnerGeometry final
{
public:
    /**
     * @brief Project explicit actual-owner order through the existing checked adaptive declaration.
     * @param layout Existing adaptive VRAM/SSD declaration, no fixed ranges or replicas.
     * @param orderedOwners Explicit storage-owner order, independent of consumer rank/vendor, including C0.
     * @param requester Existing coordinator identity, or actual requester when all capacities are zero.
     * @return Exact physical-only offsets for the representable owner count and exact native header bank.
     * @throws std::invalid_argument For foreign/missing owners, unsupported topology or incomplete bounds.
     * @throws std::overflow_error For any nonrepresentable byte extent, before returning geometry.
     * @note This accepts declared metadata, not a test-created proof of actual Rank/communicator membership.
     */
    static PooledImmutablePhysicalOwnerGeometry fromLayout(
        const ImmutableRowTableLayoutGeometry &layout,
        std::span<const PhysicalMemoryAllocatorIdentity> orderedOwners,
        PhysicalMemoryAllocatorIdentity requester);

    /** @return Original source extent; no payload is opened or hashed. */
    ImmutableRowTableExtent extent() const noexcept { return extent_; }
    /** @return Exact original native bytes; physical slot padding is never communication data. */
    std::size_t rowBytes() const noexcept { return extent_.rowBytes(); }
    /** @return Four-byte-aligned allocation stride; payload exchange still uses rowBytes(). */
    std::size_t slotStride() const noexcept { return stride_; }
    /** @return Checked row32/row64 geometry, never truncated IDs. */
    adaptive_lookup::RowKeyWidth rowKeyWidth() const noexcept { return width_; }
    /** @return Explicit declared requester coordinate, independent of physical device sorting. */
    std::optional<std::uint32_t> requesterCoordinate() const noexcept
    { return requester_ == adaptive_lookup::None ? std::nullopt : std::optional(requester_); }
    /** @return Independent coordinator allocator; it need not own physical slots. */
    PhysicalMemoryAllocatorIdentity coordinator() const noexcept { return coordinator_; }
    /** @return Actual declared K bound, not a per-step live count or source-row quota. */
    std::uint32_t maximumUniqueRows() const noexcept { return maximum_unique_; }
    /** @return Exactly P ordered owner declarations, including C0. */
    std::span<const PooledPhysicalOwnerDeclaration> owners() const noexcept { return owners_; }

    /**
     * @brief Project one native owner geometry from this same checked declared order.
     * @param coordinate Actual declared participant coordinate, never a sorted physical ordinal.
     * @return Immutable common CPU/device parameters, not a source/Rank/materialization permit.
     * @throws std::out_of_range For a foreign coordinate.
     */
    adaptive_lookup::pooled::OwnerParameters ownerParameters(std::uint32_t coordinate) const;

    /**
     * @brief Contribute only this owner's mandatory frame/command operands before optional cache sizing.
     * @param coordinate Checked owner coordinate in the declared order.
     * @param observed Canonical exact allocator resource observation, never a copied free budget.
     * @return ImmutableLookupStaging BOM; no allocation or admission decision is made.
     * @throws std::invalid_argument For foreign resources or invalid coordinate.
     */
    PhysicalMemoryPlan mandatoryBOM(std::uint32_t coordinate, PhysicalMemoryResource observed) const;

    /**
     * @brief Contribute original native payload plus canonical physical slot/free/gen metadata only.
     * @return ImmutableTableStorage BOM, empty at C0; global policy/directory is deliberately not duplicated.
     * @param coordinate Checked declared owner coordinate.
     * @param observed Exact PMA resource observation for that allocator.
     * @throws std::invalid_argument For foreign resources or invalid coordinate.
     */
    PhysicalMemoryPlan cacheBOM(std::uint32_t coordinate, PhysicalMemoryResource observed) const;

private:
    /** @brief Construct immutable metadata only after all layout/order arithmetic checks. */
    PooledImmutablePhysicalOwnerGeometry(ImmutableRowTableExtent extent) : extent_(extent) {}
    /** @return Exact declaration after validating both coordinate and resource identity. */
    const PooledPhysicalOwnerDeclaration &require(std::uint32_t coordinate,
                                                  PhysicalMemoryResource observed) const;
    PhysicalMemoryAllocatorIdentity coordinator_;
    ImmutableRowTableExtent extent_;
    std::size_t stride_ = 0;
    adaptive_lookup::RowKeyWidth width_ = adaptive_lookup::RowKeyWidth::UInt32;
    std::uint32_t requester_ = adaptive_lookup::None, maximum_unique_ = 0;
    std::vector<PooledPhysicalOwnerDeclaration> owners_;
};
} // namespace llaminar2::qwen4exp
