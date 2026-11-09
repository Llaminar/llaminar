/**
 * @file CPUProjectionStorageProof.h
 * @brief Allocation-free byte/metadata alias proof for CPU projection phases.
 *
 * Each phase declares its actual write envelope against retained reads and
 * original request publications. This is pointer/shape validation, not a live
 * memory ledger or owner: PhysicalMemoryAuthority and publication leases remain
 * authoritative. No count/frontier is copied or re-published by these checks.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised::storage_proof
{
    /** @brief Bounded immutable byte envelope; no lifetime/ownership extension. */
    struct Bytes
    {
        const void *data = nullptr;
        std::size_t size = 0;
    };

    /** @brief Compare two bounded byte envelopes without overflowing end addresses.
     * @param a First live root/extent.
     * @param b Second live root/extent.
     * @return Whether nonempty envelopes intersect.
     */
    [[nodiscard]] inline bool overlaps(Bytes a, Bytes b) noexcept
    {
        if (!a.size || !b.size) return false;
        const auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
        return x <= y ? y - x < a.size : x - y < b.size;
    }

    /** @brief Require a write envelope to retain another original storage authority.
     * @param write Actual live bytes authored by this phase.
     * @param read Original immutable values needed by the operation/owner.
     * @throws std::invalid_argument If a phase could corrupt retained storage.
     */
    inline void requireDisjoint(Bytes write, Bytes read)
    {
        if (overlaps(write, read))
            throw std::invalid_argument("CPU projection write bank aliases a retained source/publication authority");
    }

    /** @brief Keep written data outside original work descriptors and publications.
     * @param write Complete actual write envelope of this execution phase.
     * @param work Original host-owned request layout, retained unchanged throughout execution.
     * @throws std::invalid_argument For overlap with any work object/table/word.
     * @details Table entries are borrowed original addresses; this cold proof
     * does not maintain a live count mirror or infer stride across allocations.
     */
    inline void requireOriginalWorkReadOnly(Bytes write, const ProjectionRowWork *work)
    {
        if (!work) return;
        const auto id = work->captureIdentity();
        const auto reject = [&](const void *source, std::size_t size) {
            if (source) requireDisjoint(write, {source, size});
        };
        reject(work, sizeof(*work));
        reject(id.table, std::size_t(id.requests) * sizeof(ProjectionRequestPublication));
        for (int request = 0; request < id.requests; ++request)
        {
            const auto offset = std::size_t(request) * id.publication_stride_words;
            reject(id.table ? id.table[request].live : id.counts ? id.counts + offset : nullptr, sizeof(std::int32_t));
            reject(id.table ? id.table[request].frontier : id.frontiers ? id.frontiers + offset : nullptr, sizeof(std::int32_t));
            reject(id.inherited_counts ? id.inherited_counts + std::size_t(request) * id.inherited_count_stride : nullptr,
                sizeof(std::int32_t));
        }
    }
} // namespace llaminar2::cpu::native_quantised::storage_proof
