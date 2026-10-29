/**
 * @file CPUWeightStoragePlacement.h
 * @brief Placement contract for final CPU prepared-weight storage.
 *
 * Exact-node weights receive dedicated pages, certified by the common NUMA
 * first-touch authority before any packing or native-byte copy. No live bytes
 * are migrated, discarded, or copied to repair placement after publication.
 * All allocation and initialization mechanics delegate to CPUStorageRoot and
 * CPUStoragePlacement, shared with tensor arenas and serial workspaces.
 * This policy owns no capacity ledger: the prepared-weight owner retains the
 * allocation and reports it through PhysicalMemoryAuthority as usual.
 */
#pragma once

#include "CPUStorageRoot.h"

#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
    /** @brief Immutable choice of ordinary host storage or an exact NUMA owner. */
    class CPUWeightStoragePlacement final
    {
    public:
        /** @return Ordinary host storage with no single-node placement promise. */
        static CPUWeightStoragePlacement local() noexcept { return CPUWeightStoragePlacement(CPUStoragePlacement::local()); }

        /**
         * @brief Select exact-node placement for final CPU execution storage.
         * @param node Nonnegative physical NUMA node identity.
         * @return Immutable placement contract.
         * @throws std::invalid_argument If node is negative.
         */
        static CPUWeightStoragePlacement onNode(int node)
        {
            if (node < 0)
                throw std::invalid_argument("CPU weight placement requires a non-negative NUMA node");
            return CPUWeightStoragePlacement(CPUStoragePlacement::onNode(node));
        }

        /**
         * @brief Bind final storage allocations to the admitted CPU owner.
         * @param authority Sole live physical-memory authority.
         * @param allocator Exact admitted CPU allocator identity, independent of NUMA placement.
         * @param owner Persistent, migration-staging or shadow owner charged before allocation.
         * @param placement Native local or exact-node allocation policy.
         * @return Placement carrying the allocation claim policy.
         */
        static CPUWeightStoragePlacement accounted(
            std::shared_ptr<PhysicalMemoryAuthority> authority,
            DeviceId allocator, PhysicalMemoryOwner owner,
            CPUWeightStoragePlacement placement)
        {
            if (!authority || !allocator.is_valid() || !allocator.is_cpu() ||
                !authority->contains(allocator) || (owner != PhysicalMemoryOwner::PrimaryModelWeights &&
                owner != PhysicalMemoryOwner::AdditionalModelWeights &&
                owner != PhysicalMemoryOwner::RoutedExpertWeights &&
                owner != PhysicalMemoryOwner::ExpertMigrationStaging &&
                owner != PhysicalMemoryOwner::ExpertShadowSlots) || placement.authority_)
                throw std::invalid_argument("CPU weight accounting requires one admitted prepared-storage owner");
            placement.authority_ = std::move(authority);
            placement.owner_ = owner;
            placement.allocator_ = allocator;
            return placement;
        }

        /**
         * @brief Resolve exactly the allocator-rounded final storage extent.
         * @tparam T Native element type.
         * @param count Required element capacity.
         * @return Physical bytes, excluding unfaultable virtual mapping guards.
         */
        template <typename T>
        [[nodiscard]] size_t requiredAllocationBytes(size_t count) const
        {
            return placement_.physicalBytes<T>(count);
        }

        /**
         * @brief Allocate and certify unpublished storage before its producer writes.
         * @tparam T Trivially copyable native element type.
         * @param count Number of elements the final owner requires.
         * @return Final storage ready for the producer to overwrite every element.
         * @throws std::runtime_error If requested first-touch placement fails.
         */
        template <typename T>
        [[nodiscard]] AlignedVector<T> allocate(size_t count) const
        {
            if (authority_)
                return CPUStorageRoot<T>(count, owner_, allocator_, placement_,
                    CPUStorageInitialization::Uninitialized).allocate(authority_);
            return placement_.allocateUnadmitted<T>(count, CPUStorageInitialization::Uninitialized);
        }

    private:
        /** @brief Construct only through the explicit placement factories. */
        explicit CPUWeightStoragePlacement(CPUStoragePlacement placement) noexcept : placement_(placement) {}
        std::shared_ptr<PhysicalMemoryAuthority> authority_; ///< Sole admission issuer, never a parallel ledger.
        DeviceId allocator_ = DeviceId::invalid(); ///< Exact admitted CPU allocator, never a NUMA node.
        PhysicalMemoryOwner owner_ = PhysicalMemoryOwner::PrimaryModelWeights; ///< Typed accounting owner.
        CPUStoragePlacement placement_; ///< Common immutable allocator/NUMA contract.
    };
}
