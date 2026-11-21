/**
 * @file NativeGraphObjectGeometry.h
 * @brief Owner-published host bytes of one native graph-capture adapter per GPU.
 *
 * Every retained graph definition keeps one host adapter object
 * (CUDAGraphCapture or HIPGraphCapture) on the rank that owns its GPU. That
 * object is compiled into the owner's binary: its size depends on the owner's
 * backend and ABI (std::mutex alone differs between x86-64 and AArch64), and a
 * binary without that backend cannot name it at all. A planner pricing another
 * rank's graph memory therefore never derives the size from its own compile.
 *
 * Owners publish the size with their native context declarations
 * (PlanningNativeContextDeclaration::graph_capture_object_bytes). A live
 * configuration that holds its own prepared native source is that owner and
 * answers from its compiled adapter, resolved only when a GPU is priced. Lookups for an allocator without an
 * owner entry are fatal; there is no default size.
 */
#pragma once
#include "backends/GPUFlatGraphStructure.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>

namespace llaminar2 {
namespace test { struct NativeGraphObjectGeometryRecordedAccess; }

/** @brief Immutable-after-construction map of owner-published adapter sizes. */
class NativeGraphObjectGeometry final {
public:
    /**
     * @brief Publish this process's own adapter size for a device it owns.
     * @param owner A GPU this process has prepared (its live native source);
     *        the backend is therefore compiled into this binary.
     * @return Geometry holding exactly that owner.
     * @throws std::invalid_argument When this binary does not compile the backend.
     */
    static NativeGraphObjectGeometry localOwner(PhysicalMemoryAllocatorIdentity owner)
    {
        NativeGraphObjectGeometry geometry;
        geometry.own(owner);
        return geometry;
    }

    /**
     * @brief Mark one GPU whose live native source this process holds.
     * @param owner GPU allocator of this process's own rank; the caller has
     *        proven ownership (a live preparation of that rank), so the
     *        compiled adapter answers its lookup.
     * @throws std::invalid_argument For a non-GPU owner.
     */
    void publishLocalOwner(PhysicalMemoryAllocatorIdentity owner) { own(owner); }

    /**
     * @brief Record one owner's adapter size.
     * @param owner Original GPU allocator (world rank and device).
     * @param bytes Owner-reported sizeof of its native graph-capture adapter.
     * @throws std::invalid_argument For a non-GPU owner, zero bytes, or a
     *         second, different value for the same owner.
     */
    void publish(PhysicalMemoryAllocatorIdentity owner, std::size_t bytes)
    {
        if (!owner.valid() || !owner.device.is_gpu() || bytes == 0)
            throw std::invalid_argument("Native graph object geometry needs a GPU owner and positive bytes");
        const auto [entry, inserted] = bytes_.emplace(key(owner), bytes);
        if (!inserted && entry->second != bytes)
            throw std::invalid_argument("Native graph object geometry received conflicting owner values for " +
                                        owner.device.toString());
    }

    /**
     * @param owner Original GPU allocator whose graph memory is being priced.
     * @return The owner's published adapter size.
     * @throws std::invalid_argument When the owner published nothing.
     */
    [[nodiscard]] std::size_t bytes(PhysicalMemoryAllocatorIdentity owner) const
    {
        const auto found = bytes_.find(key(owner));
        if (found == bytes_.end() && owned_.contains(key(owner)))
            return nativeGraphCaptureObjectBytes(owner.device); // This process is the owner.
        if (found == bytes_.end() && recorded_ && owner.valid() && owner.device.is_gpu())
            return *recorded_;
        if (found == bytes_.end())
            throw std::invalid_argument("Native graph object size for rank " + std::to_string(owner.world_rank) +
                                        " " + owner.device.toString() + " has no owner declaration");
        return found->second;
    }

    /**
     * @brief Renumber discovery-rank evidence into a candidate's execution ranks.
     * @param discovery_ranks Original discovery rank of each execution rank, in order.
     * @return Published entries of the selected ranks keyed by execution rank;
     *         unselected ranks are dropped, locally owned entries are not projected.
     */
    [[nodiscard]] NativeGraphObjectGeometry projectedOnto(std::span<const int> discovery_ranks) const
    {
        NativeGraphObjectGeometry projected;
        for (const auto &[key, size] : bytes_)
            for (std::size_t execution = 0; execution < discovery_ranks.size(); ++execution)
                if (std::get<0>(key) == discovery_ranks[execution])
                    projected.bytes_.emplace(Key{static_cast<int>(execution), std::get<1>(key), std::get<2>(key)}, size);
        projected.recorded_ = recorded_;
        return projected;
    }

    /** @return Whether the owner published a size. */
    [[nodiscard]] bool contains(PhysicalMemoryAllocatorIdentity owner) const
    {
        return bytes_.contains(key(owner)) || owned_.contains(key(owner)) ||
               (recorded_ && owner.valid() && owner.device.is_gpu());
    }

private:
    friend struct test::NativeGraphObjectGeometryRecordedAccess;
    /** @brief Mark a GPU this process owns; its compiled adapter answers lookups. */
    void own(PhysicalMemoryAllocatorIdentity owner)
    {
        if (!owner.valid() || !owner.device.is_gpu())
            throw std::invalid_argument("Native graph object geometry needs a GPU owner");
        owned_.insert(key(owner));
    }

    using Key = std::tuple<int, int, int>;
    static Key key(PhysicalMemoryAllocatorIdentity owner) noexcept
    {
        return {owner.world_rank, static_cast<int>(owner.device.type), owner.device.ordinal};
    }
    std::map<Key, std::size_t> bytes_;
    std::set<Key> owned_;
    /** Recorded owner value for every GPU; only the named test access creates it. */
    std::optional<std::size_t> recorded_;
};

} // namespace llaminar2
