/**
 * @file CPUWeightStoragePlacement.h
 * @brief Placement contract for final CPU prepared-weight storage.
 *
 * Exact-node weights receive dedicated pages, certified by the common NUMA
 * first-touch authority before any packing or native-byte copy. No live bytes
 * are migrated, discarded, or copied to repair placement after publication.
 * This policy owns no capacity ledger: the prepared-weight owner retains the
 * allocation and reports it through PhysicalMemoryAuthority as usual.
 */
#pragma once

#include "NUMAAllocator.h"
#include "tensors/AlignedVector.h"

#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
    /** @brief Immutable choice of ordinary host storage or an exact NUMA owner. */
    class CPUWeightStoragePlacement final
    {
    public:
        /** @return Ordinary host storage with no single-node placement promise. */
        static CPUWeightStoragePlacement local() noexcept { return CPUWeightStoragePlacement(-1); }

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
            return CPUWeightStoragePlacement(node);
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
            static_assert(std::is_trivially_copyable_v<T>);
            if (node_ < 0)
            {
                AlignedVector<T> storage;
                storage.resize_uninitialized(count);
                return storage;
            }
            auto storage = AlignedVector<T>::pageMappedUninitialized(count);
            // Fresh exclusive mappings cannot share recycled/pinned heap pages.
            // Certification happens before the source decoder writes its bytes.
            if (count && !NUMAAllocator::instance().prepareExternalReceiveRangeOnNode(
                    storage.data(), storage.size() * sizeof(T), node_))
                throw std::runtime_error("CPU prepared-weight first-touch placement failed");
            return storage;
        }

    private:
        /** @brief Construct only through the explicit placement factories. */
        explicit CPUWeightStoragePlacement(int node) noexcept : node_(node) {}
        int node_; ///< Negative means no exact-node placement was requested.
    };
}
