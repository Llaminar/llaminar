/**
 * @file CPUStorageRoot.h
 * @brief Immutable CPU physical-root declarations and their single allocator.
 *
 * A declaration binds element capacity, exact CPU allocator, typed physical
 * owner, first-touch placement and initialization before admission. Its byte
 * extent comes from AlignedVector's allocator geometry. Materialization claims
 * that exact extent before allocation and binds the lease to the actual vector;
 * tensor adoption, moves and views preserve that physical owner. This interface
 * owns no live ledger and accepts no ambient or optional admission authority.
 */
#pragma once
#include "NUMAAllocator.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/AlignedVector.h"
#include <cstring>
#include <stdexcept>
#include <type_traits>
namespace llaminar2
{
/** @brief Whether the producer requires zero values or overwrites every element. */
enum class CPUStorageInitialization : uint8_t { Uninitialized, Zero };
template <typename T> class CPUStorageRoot;
/** @brief Immutable physical allocation and NUMA first-touch policy, without accounting. */
class CPUStoragePlacement final
{
public:
    /** @return Native host heap placement with ordinary allocator rounding. */
    static CPUStoragePlacement local() noexcept { return CPUStoragePlacement(-1); }
    /**
     * @brief Declare dedicated page mappings certified on an exact NUMA node.
     * @param node Nonnegative physical NUMA node, distinct from CPU DeviceId.
     * @return Placement policy applied before the producer's first write.
     */
    static CPUStoragePlacement onNode(int node)
    {
        if (node < 0) throw std::invalid_argument("CPU storage requires a nonnegative NUMA node");
        return CPUStoragePlacement(node);
    }
    /** @return Structural equality of complete placement policy. */
    bool operator==(const CPUStoragePlacement &) const = default;
    /** @return Exact node or -1 for native heap placement. */
    [[nodiscard]] int node() const noexcept { return node_; }
    /**
     * @brief Resolve exact physical capacity through the allocator's geometry.
     * @tparam T Native storage element.
     * @param count Final element capacity.
     * @return Physical payload bytes, excluding inaccessible virtual guards.
     */
    template <typename T> [[nodiscard]] size_t physicalBytes(size_t count) const
    {
        using Storage = AlignedVector<T>;
        return Storage::requiredAllocationBytes(count, Storage::ALIGNMENT,
            node_ < 0 ? Storage::StorageKind::Heap : Storage::StorageKind::AnonymousPageMapping);
    }
    /**
     * @brief Explicit low-level allocation without an admission contract.
     * @tparam T Native trivially copyable storage element.
     * @param count Required final capacity.
     * @param initialization Producer initialization contract.
     * @return Native storage for setup infrastructure or device-free fixtures.
     *
     * Production admitted roots use CPUStorageRoot::allocate instead. This
     * named primitive preserves the explicit local/onNode low-level contract;
     * it is never selected after an admission or materialization failure.
     */
    template <typename T> [[nodiscard]] AlignedVector<T> allocateUnadmitted(size_t count,
        CPUStorageInitialization initialization = CPUStorageInitialization::Uninitialized) const
    {
        return materialize<T>(count, initialization, nullptr, DeviceId::invalid(), PhysicalMemoryOwner::Count);
    }
private:
    template <typename T> friend class CPUStorageRoot;
    /** @brief Construct only through explicit placement factories. */
    explicit CPUStoragePlacement(int node) noexcept : node_(node) {}
    /**
     * @brief Single physical allocator for admitted and explicitly low-level CPU storage.
     * @tparam T Native trivially copyable storage element.
     * @param count Exact element capacity.
     * @param initialization Whether to zero after first-touch certification.
     * @param authority Sole PMA issuer, absent only for named low-level allocation.
     * @param allocator Exact admitted CPU allocator identity.
     * @param owner Explicit physical owner.
     * @return Backing vector owning its physical bytes and any admission lease.
     */
    template <typename T> [[nodiscard]] AlignedVector<T> materialize(size_t count,
        CPUStorageInitialization initialization, std::shared_ptr<PhysicalMemoryAuthority> authority,
        DeviceId allocator, PhysicalMemoryOwner owner) const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (initialization != CPUStorageInitialization::Zero && initialization != CPUStorageInitialization::Uninitialized)
            throw std::invalid_argument("CPU storage initialization policy is invalid");
        const size_t bytes = physicalBytes<T>(count);
        std::shared_ptr<PhysicalMemoryAllocationLease> lease;
        if (authority && bytes)
            lease = std::make_shared<PhysicalMemoryAllocationLease>(authority->claimNewAllocation(allocator, owner, bytes));
        AlignedVector<T> storage;
        if (node_ < 0) storage.resize_uninitialized(count);
        else storage = AlignedVector<T>::pageMappedUninitialized(count);
        if (storage.allocationBytes() != bytes)
            throw std::logic_error("CPU root storage differs from preclaimed allocator geometry");
        if (authority) storage.retainAllocationLifetime(std::move(lease), bytes, authority, allocator, owner);
        // NUMA certification precedes the first producer write, including zero initialization.
        if (node_ >= 0 && count && !NUMAAllocator::instance().prepareExternalReceiveRangeOnNode(
            storage.data(), count * sizeof(T), node_))
            throw std::runtime_error("CPU root first-touch placement failed");
        if (initialization == CPUStorageInitialization::Zero && count)
            std::memset(static_cast<void *>(storage.data()), 0, count * sizeof(T));
        return storage;
    }
    int node_; ///< Physical placement policy, never a CPU allocator identity.
};
/** @brief Complete immutable declaration of one admitted CPU physical allocation. */
template <typename T> class CPUStorageRoot final
{
public:
    /**
     * @brief Declare final CPU storage before contributing its physical BOM.
     * @param capacity Element capacity, including an explicitly declared zero root.
     * @param owner Typed PMA owner of the physical bytes.
     * @param allocator Exact CPU DeviceId carried by the declaring schema.
     * @param placement Native heap or exact-node page-mapping policy.
     * @param initialization Tensor zero values or producer-overwritten bytes.
     */
    CPUStorageRoot(size_t capacity, PhysicalMemoryOwner owner, DeviceId allocator,
        CPUStoragePlacement placement = CPUStoragePlacement::local(),
        CPUStorageInitialization initialization = CPUStorageInitialization::Zero)
        : capacity_(capacity), owner_(owner), allocator_(allocator), placement_(placement), initialization_(initialization)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!allocator.is_valid() || !allocator.is_cpu() || owner >= PhysicalMemoryOwner::Count ||
            (initialization != CPUStorageInitialization::Zero && initialization != CPUStorageInitialization::Uninitialized))
            throw std::invalid_argument("CPU storage root requires an exact CPU allocator, physical owner and initialization policy");
        (void)physicalBytes(); // Reject overflow at declaration, before admission.
    }
    /** @return Structural equality of complete immutable root identity. */
    bool operator==(const CPUStorageRoot &) const = default;
    /** @return Immutable native element capacity. */
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    /** @return Typed physical owner declared by the schema. */
    [[nodiscard]] PhysicalMemoryOwner owner() const noexcept { return owner_; }
    /** @return Exact admitted CPU allocator identity. */
    [[nodiscard]] DeviceId allocator() const noexcept { return allocator_; }
    /** @return Immutable physical placement policy. */
    [[nodiscard]] CPUStoragePlacement placement() const noexcept { return placement_; }
    /** @return Immutable initialization contract. */
    [[nodiscard]] CPUStorageInitialization initialization() const noexcept { return initialization_; }
    /** @return Exact physical bytes produced by the common allocator. */
    [[nodiscard]] size_t physicalBytes() const { return placement_.physicalBytes<T>(capacity_); }
    /**
     * @brief Preclaim and materialize exactly the declared physical root.
     * @param authority Mandatory immutable-admission/live-allocation authority.
     * @return Final vector whose actual backing owns the typed allocation lease.
     * @throws std::invalid_argument For absent authority or a foreign allocator.
     */
    [[nodiscard]] AlignedVector<T> allocate(std::shared_ptr<PhysicalMemoryAuthority> authority) const
    {
        if (!authority || !authority->contains(allocator_))
            throw std::invalid_argument("CPU storage root requires its admitted allocator authority");
        return placement_.materialize<T>(capacity_, initialization_, std::move(authority), allocator_, owner_);
    }
private:
    const size_t capacity_; ///< Final immutable element capacity.
    const PhysicalMemoryOwner owner_; ///< Exact BOM owner, never inferred from a caller.
    const DeviceId allocator_; ///< Exact admitted physical allocator.
    const CPUStoragePlacement placement_; ///< NUMA policy independent of allocator identity.
    const CPUStorageInitialization initialization_; ///< First producer's write contract.
};
}
