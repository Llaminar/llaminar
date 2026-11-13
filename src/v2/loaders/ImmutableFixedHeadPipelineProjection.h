/**
 * @file ImmutableFixedHeadPipelineProjection.h
 * @brief Immutable original native pipeline ownership for fixed PLE head ranges.
 *
 * The source head order, original pipeline member order and PLE reader order
 * are different coordinates. This checked projection connects them without
 * adding compute participants or creating a second source or PMA ledger.
 */
#pragma once

#include "planning/PhysicalMemoryAuthority.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2 {

/** @brief One actual fixed range owner and the PLE consumer assembling its rows. */
struct ImmutableFixedHeadPipelineOwner final {
    PhysicalMemoryAllocatorIdentity allocator;
    int stage_index;
    int participant_index;
    int consumer_index;
    int first_head;
    int head_count;
    bool operator==(const ImmutableFixedHeadPipelineOwner&) const = default;
};

/** @brief Exact disjoint slices of one participant-local native scratch bank. */
struct ImmutableFixedHeadPipelineEdgeScratch final {
    std::size_t ids_offset, ids_bytes, native_offset, native_bytes;
    std::optional<std::size_t> service_rows_offset;
};

/** @brief Original fixed-source roles, ordered by the source's logical head intervals. */
class ImmutableFixedHeadPipelineProjection final {
public:
    /**
     * @brief Assign a nonempty whole-head range to every original GPU.
     * @param rank Original executing MPI rank.
     * @param consumer_stage Stage owning the authenticated PLE decoder layer.
     * @param heads Number of original source heads.
     * @param maximum_rows Common admitted prefill/verifier graph-row envelope.
     * @param stages Original ordered stage/member roster authenticated by the caller's inventory.
     * @return Source-ordered storage owners and exact ordered PLE-stage consumers.
     * @throws std::invalid_argument If any stage is empty, foreign, heterogeneous or duplicated, or if the source has too few heads.
     */
    static ImmutableFixedHeadPipelineProjection checked(int rank, int consumer_stage, int heads, int maximum_rows,
        std::span<const std::vector<PhysicalMemoryAllocatorIdentity>> stages)
    {
        if (rank < 0 || consumer_stage < 0 || static_cast<std::size_t>(consumer_stage) >= stages.size() ||
            stages.size() < 2 || stages.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            heads <= 0 || maximum_rows <= 0 || stages[consumer_stage].empty())
            throw std::invalid_argument("Fixed PLE pipeline requires an original consumer stage and source heads");
        const auto& consumers = stages[consumer_stage];
        if (consumers.size() > static_cast<std::size_t>(heads))
            throw std::invalid_argument("Fixed PLE source has fewer heads than original consumers");
        const auto type = consumers.front().device.type;
        std::set<DeviceId> seen;
        std::vector<std::vector<ImmutableFixedHeadPipelineOwner>> groups(consumers.size());
        for (std::size_t index = 0; index < consumers.size(); ++index)
            groups[index].push_back({consumers[index], consumer_stage, static_cast<int>(index),
                static_cast<int>(index), 0, 0});
        std::size_t next_reader = 0;
        for (std::size_t stage = 0; stage < stages.size(); ++stage) {
            const auto& members = stages[stage];
            if (members.empty() || members.size() > static_cast<std::size_t>(heads))
                throw std::invalid_argument("Fixed PLE pipeline lost a nonempty original stage");
            for (std::size_t index = 0; index < members.size(); ++index) {
                const auto& allocator = members[index];
                if (!allocator.valid() || allocator.world_rank != rank || !allocator.device.is_gpu() ||
                    allocator.device.type != type || !seen.insert(allocator.device).second ||
                    seen.size() > static_cast<std::size_t>(heads))
                    throw std::invalid_argument("Fixed PLE pipeline requires unique original homogeneous GPU owners with nonempty heads");
                if (static_cast<int>(stage) == consumer_stage)
                    continue;
                const auto reader = next_reader++ % consumers.size();
                groups[reader].push_back({allocator, static_cast<int>(stage), static_cast<int>(index),
                    static_cast<int>(reader), 0, 0});
            }
        }
        ImmutableFixedHeadPipelineProjection result(rank, consumer_stage, maximum_rows, consumers);
        result.owners_.reserve(seen.size());
        const int population = static_cast<int>(seen.size());
        int first = 0;
        for (auto& group : groups)
            for (auto& owner : group) {
                // Pairing precedes range assignment, so one reader's useful
                // local and remote rows form a contiguous logical head block.
                owner.first_head = first;
                owner.head_count = heads / population +
                    (result.owners_.size() < static_cast<std::size_t>(heads % population) ? 1 : 0);
                first += owner.head_count;
                result.owners_.push_back(owner);
            }
        return result;
    }

    /** @return Original rank owning every native endpoint. */
    int worldRank() const noexcept { return rank_; }
    /** @return Common admitted prefill/verifier row capacity. */
    int maximumRows() const noexcept { return maximum_rows_; }
    /** @return Original stage that actually computes PLE. */
    int consumerStage() const noexcept { return consumer_stage_; }
    /** @return Source-order, disjoint fixed range owners. */
    std::span<const ImmutableFixedHeadPipelineOwner> owners() const noexcept { return owners_; }
    /** @return Actual PLE consumers in original stage-local order. */
    std::span<const PhysicalMemoryAllocatorIdentity> consumers() const noexcept { return consumers_; }
    /**
     * @param allocator Exact original physical identity.
     * @return Its immutable storage/reader routing role.
     * @throws std::invalid_argument If the allocator does not own a range.
     */
    const ImmutableFixedHeadPipelineOwner& owner(PhysicalMemoryAllocatorIdentity allocator) const
    {
        const auto found = std::find_if(owners_.begin(), owners_.end(),
            [&](const auto& member) { return member.allocator == allocator; });
        if (found == owners_.end())
            throw std::invalid_argument("Fixed PLE pipeline queried a foreign original allocator");
        return *found;
    }
    /**
     * @param allocator Actual storage owner or PLE reader.
     * @return Exact incident request and response native byte frames.
     * @throws std::invalid_argument If the allocator is foreign.
     */
    std::size_t endpointCount(PhysicalMemoryAllocatorIdentity allocator) const
    {
        const auto& member = owner(allocator);
        if (member.stage_index != consumer_stage_)
            return 2;
        return 2 * std::count_if(owners_.begin(), owners_.end(), [&](const auto& remote) {
            return remote.stage_index != consumer_stage_ && remote.consumer_index == member.consumer_index;
        });
    }
    /**
     * @param consumer_index Actual index in the original PLE stage.
     * @return Number of contiguous source heads assembled by this reader.
     * @throws std::out_of_range For a foreign stage-local index.
     */
    int assembledHeadCount(int consumer_index) const
    {
        if (consumer_index < 0 || static_cast<std::size_t>(consumer_index) >= consumers_.size())
            throw std::out_of_range("Fixed PLE pipeline has no such original reader");
        int heads = 0;
        for (const auto& member : owners_)
            if (member.consumer_index == consumer_index)
                heads += member.head_count;
        return heads;
    }
    /** @return Largest assembled reader contribution in original heads. */
    int maximumAssembledHeads() const
    {
        int heads = 0;
        for (std::size_t reader = 0; reader < consumers_.size(); ++reader)
            heads = std::max(heads, assembledHeadCount(static_cast<int>(reader)));
        return heads;
    }
    /** @return Whether actual reader packet widths require variable native gather. */
    bool requiresVariableGather() const
    {
        if (consumers_.size() <= 1)
            return false;
        const auto first = assembledHeadCount(0);
        for (std::size_t reader = 1; reader < consumers_.size(); ++reader)
            if (assembledHeadCount(static_cast<int>(reader)) != first)
                return true;
        return false;
    }
    /**
     * @param consumer_index Actual original reader index.
     * @return Exact gathered arena head count, excluding local-only packet ownership.
     * @throws std::out_of_range For a foreign reader.
     */
    int gatheredHeadCount(int consumer_index) const
    {
        const auto local = assembledHeadCount(consumer_index);
        if (consumers_.size() == 1)
            return 0;
        const auto total = owners_.back().first_head + owners_.back().head_count;
        return requiresVariableGather() ? total - local : total;
    }
    /**
     * @param allocator Actual local reader or storage-only owner on the edge.
     * @param storage_owner Actual storage-only peer owning the selected range.
     * @param maximum_rows Retained capacity of the original graph family.
     * @param native_row_bytes Byte width of one original native table row.
     * @return Exact local scratch subviews for the pair.
     * @throws std::exception For foreign roles, invalid dimensions or overflow.
     */
    ImmutableFixedHeadPipelineEdgeScratch edgeScratch(PhysicalMemoryAllocatorIdentity allocator,
        PhysicalMemoryAllocatorIdentity storage_owner, std::size_t maximum_rows, std::size_t native_row_bytes) const
    {
        std::optional<ImmutableFixedHeadPipelineEdgeScratch> selected;
        (void)visitScratch(allocator, maximum_rows, native_row_bytes, [&](const auto& remote, auto edge) {
            if (remote.allocator == storage_owner)
                selected = edge;
        });
        if (!selected)
            throw std::invalid_argument("Fixed PLE scratch names a foreign reader/storage edge");
        return *selected;
    }
    /**
     * @param allocator Actual local reader or storage-only owner.
     * @param maximum_rows Retained capacity of the original graph family.
     * @param native_row_bytes Byte width of one original native table row.
     * @return Exact aligned scratch root byte size, excluding native completion frames.
     * @throws std::exception For foreign roles, invalid dimensions or overflow.
     */
    std::size_t scratchBytes(PhysicalMemoryAllocatorIdentity allocator,
        std::size_t maximum_rows, std::size_t native_row_bytes) const
    {
        return visitScratch(allocator, maximum_rows, native_row_bytes, [](const auto&, auto) {});
    }
    /**
     * @param allocator Actual local reader.
     * @param maximum_rows Retained common prefill/verifier rows.
     * @param native_row_bytes Original native row stride.
     * @return One reusable reply-count field after this reader's disjoint packets, if it has peers.
     * @throws std::exception For invalid dimensions, allocator or overflow.
     */
    std::optional<std::size_t> readerResponseRowsOffset(PhysicalMemoryAllocatorIdentity allocator,
        std::size_t maximum_rows, std::size_t native_row_bytes) const
    {
        if (owner(allocator).stage_index != consumer_stage_)
            throw std::invalid_argument("Fixed PLE reply count requires its original reader");
        std::optional<std::size_t> offset;
        (void)visitScratch(allocator, maximum_rows, native_row_bytes, [](const auto&, auto) {}, &offset);
        return offset;
    }
    bool operator==(const ImmutableFixedHeadPipelineProjection&) const = default;

private:
    /**
     * @brief Lay out the exact native messages in source order for one endpoint.
     * @param allocator Actual original endpoint.
     * @param rows Common retained row capacity.
     * @param stride Source-native table row bytes.
     * @param visitor Observer of immutable edge offsets.
     * @param reader_response Optional projection of the reader's reusable native receipt count.
     * @return Complete aligned root extent.
     * @throws std::exception For foreign membership or invalid byte arithmetic.
     */
    template<class Visitor>
    std::size_t visitScratch(PhysicalMemoryAllocatorIdentity allocator, std::size_t rows,
        std::size_t stride, Visitor&& visitor,
        std::optional<std::size_t>* reader_response = nullptr) const
    {
        const auto& local = owner(allocator);
        if (rows != static_cast<std::size_t>(maximum_rows_) || !stride)
            throw std::invalid_argument("Fixed PLE scratch requires its admitted row envelope and native stride");
        const auto add = [](std::size_t left, std::size_t right) {
            if (right > std::numeric_limits<std::size_t>::max() - left)
                throw std::overflow_error("Fixed PLE scratch byte offset overflow");
            return left + right;
        };
        const auto multiply = [](std::size_t left, std::size_t right) {
            if (right && left > std::numeric_limits<std::size_t>::max() / right)
                throw std::overflow_error("Fixed PLE scratch byte extent overflow");
            return left * right;
        };
        constexpr auto alignment = alignof(std::int64_t);
        const auto align = [&](std::size_t bytes) {
            return add(bytes, (alignment - bytes % alignment) % alignment);
        };
        std::size_t offset = 0;
        const bool reader = local.stage_index == consumer_stage_;
        for (const auto& remote : owners_) {
            if (remote.stage_index == consumer_stage_ ||
                (reader ? remote.consumer_index != local.consumer_index : remote.allocator != allocator))
                continue;
            const auto count = multiply(rows, static_cast<std::size_t>(remote.head_count));
            ImmutableFixedHeadPipelineEdgeScratch edge{offset, multiply(count, sizeof(std::int64_t)),
                0, multiply(count, stride), {}};
            edge.native_offset = add(offset, edge.ids_bytes);
            offset = align(add(edge.native_offset, edge.native_bytes));
            if (!reader) {
                edge.service_rows_offset = offset;
                offset = align(add(offset, sizeof(std::int32_t)));
            }
            visitor(remote, edge);
        }
        if (reader && offset) {
            if (reader_response) *reader_response = offset;
            offset = align(add(offset, sizeof(std::int32_t)));
        }
        return offset;
    }

    /**
     * @brief Start construction from the already checked original reader stage.
     * @param rank Original owning rank.
     * @param consumer_stage Actual PLE stage index.
     * @param maximum_rows Common admitted row envelope.
     * @param consumers Original ordered stage reader membership.
     */
    ImmutableFixedHeadPipelineProjection(int rank, int consumer_stage, int maximum_rows,
        const std::vector<PhysicalMemoryAllocatorIdentity>& consumers)
        : rank_(rank), consumer_stage_(consumer_stage), maximum_rows_(maximum_rows), consumers_(consumers) {}
    int rank_;
    int consumer_stage_;
    int maximum_rows_;
    std::vector<PhysicalMemoryAllocatorIdentity> consumers_;
    std::vector<ImmutableFixedHeadPipelineOwner> owners_;
};

} // namespace llaminar2
