/**
 * @file PrefixHostArena.cpp
 * @brief Reuse admitted prefix bytes without native allocation or hot-path waits.
 *
 * Retirement only changes an existing placement node. A later admission queries
 * each exact producer event once and coalesces completed ranges; it never waits
 * for GPU progress or reclaims bytes still held by a request or disk writer.
 * Archive sections have independent lifetime tokens. Admission and acquisition
 * share one placement algorithm, so a restore can retain its KV readers while
 * unrelated recurrent images are reused across physically separate free holes.
 */
#include "execution/prefix_cache/PrefixHostArena.h"
#include "utils/Logger.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace llaminar2
{
    std::shared_ptr<PrefixHostArena> PrefixHostArena::create(
        std::shared_ptr<void> backing, size_t bytes, std::shared_ptr<void> memory_claim)
    {
        if (!backing || bytes == 0u)
            throw std::invalid_argument("Prefix host arena requires materialized backing");
        return std::shared_ptr<PrefixHostArena>(
            new PrefixHostArena(std::move(backing), bytes, std::move(memory_claim)));
    }

    PrefixHostArena::PrefixHostArena(
        std::shared_ptr<void> backing, size_t bytes, std::shared_ptr<void> memory_claim)
        : memory_claim_(std::move(memory_claim)), backing_(std::move(backing))
    {
        extents_.emplace(0u, Extent{.bytes = bytes});
    }

    PrefixHostArena::~PrefixHostArena()
    {
        // Every active lease retains this arena, so only free/retired ranges
        // can remain. Waiting is legal here: this is physical owner teardown.
        for (const auto &[offset, extent] : extents_)
        {
            (void)offset;
            if (extent.state == State::Leased ||
                (extent.readiness && extent.readiness->prepared() && !extent.readiness->published()) ||
                (extent.readiness && !extent.readiness->waitOnHost()))
            {
                LOG_ERROR("[PrefixHostArena] unsafe final backing retirement at offset=" << offset
                          << " bytes=" << extent.bytes
                          << ": live lease or unpublished/failed producer event");
                std::terminate();
            }
        }
    }

    PrefixHostArena::Lease::~Lease()
    {
        if (arena)
            arena->retire(offset);
    }

    void PrefixHostArena::retire(size_t offset) noexcept
    {
        std::lock_guard lock(mutex_);
        auto it = extents_.find(offset);
        if (it == extents_.end() || it->second.state != State::Leased)
        {
            LOG_ERROR("[PrefixHostArena] invalid last-alias retirement at offset=" << offset);
            std::terminate();
        }
        it->second.state = State::Retired;
    }

    void PrefixHostArena::collectRetired()
    {
        for (auto &[offset, extent] : extents_)
        {
            (void)offset;
            if (extent.state != State::Retired)
                continue;
            bool ready = true;
            if (extent.readiness && !extent.readiness->queryComplete(&ready))
                throw std::runtime_error("Prefix host arena producer-event query failed");
            if (ready)
            {
                extent.readiness.reset();
                extent.state = State::Free;
            }
        }
        for (auto it = extents_.begin(); it != extents_.end();)
        {
            auto next = std::next(it);
            if (next != extents_.end() && it->second.state == State::Free &&
                next->second.state == State::Free)
            {
                it->second.bytes += next->second.bytes;
                extents_.erase(next);
            }
            else
                it = next;
        }
    }

    size_t PrefixHostArena::availableBytes()
    {
        std::lock_guard lock(mutex_);
        collectRetired();
        size_t largest = 0u;
        for (const auto &[offset, extent] : extents_)
        {
            (void)offset;
            if (extent.state == State::Free)
                largest = std::max(largest, extent.bytes);
        }
        return largest;
    }

    size_t PrefixHostArena::availableStorageBytes()
    {
        std::lock_guard lock(mutex_);
        collectRetired();
        size_t available = 0u;
        for (const auto &[offset, extent] : extents_)
        {
            (void)offset;
            if (extent.state == State::Free)
                available += extent.bytes;
        }
        return available;
    }

    std::optional<std::vector<size_t>> PrefixHostArena::planSections(
        std::span<const size_t> sections) const
    {
        // This transient geometry describes existing free nodes, not a second
        // physical capacity ledger. Only committed extents own placement.
        std::vector<std::pair<size_t, size_t>> free_ranges;
        free_ranges.reserve(extents_.size());
        for (const auto &[offset, extent] : extents_)
            if (extent.state == State::Free)
                free_ranges.emplace_back(offset, extent.bytes);

        std::vector<size_t> offsets(sections.size(), 0u);
        for (size_t section = 0u; section < sections.size(); ++section)
        {
            const size_t bytes = sections[section];
            if (bytes == 0u)
                continue;
            auto range = std::find_if(free_ranges.begin(), free_ranges.end(),
                [bytes](const auto &candidate) { return bytes <= candidate.second; });
            if (range == free_ranges.end())
                return std::nullopt;
            offsets[section] = range->first;
            range->first += bytes;
            range->second -= bytes;
        }
        return offsets;
    }

    bool PrefixHostArena::canAcquireSections(std::span<const size_t> sections)
    {
        std::lock_guard lock(mutex_);
        collectRetired();
        return planSections(sections).has_value();
    }

    std::vector<std::shared_ptr<void>> PrefixHostArena::acquireSections(
        std::span<const size_t> sections,
        std::shared_ptr<PrefixPayloadReadiness> readiness)
    {
        std::lock_guard lock(mutex_);
        collectRetired();
        const auto offsets = planSections(sections);
        if (!offsets)
            return {};

        // Prepare every token and replacement map node before changing live
        // placement. Metadata allocation failure cannot leave a partial archive
        // or a Leased node whose final owner was never constructed.
        std::vector<std::shared_ptr<Lease>> leases(sections.size());
        std::vector<std::shared_ptr<void>> owners(sections.size());
        std::map<size_t, Extent> replacements;
        std::vector<size_t> replaced_offsets;
        replaced_offsets.reserve(sections.size());
        const auto self = shared_from_this();
        for (size_t section = 0u; section < sections.size(); ++section)
        {
            if (sections[section] == 0u)
                continue;
            leases[section] = std::make_shared<Lease>();
            leases[section]->offset = (*offsets)[section];
            owners[section] = std::shared_ptr<void>(leases[section],
                static_cast<uint8_t *>(backing_.get()) + (*offsets)[section]);
            if (!replacements.emplace((*offsets)[section], Extent{
                    .bytes = sections[section], .state = State::Leased,
                    .readiness = readiness}).second)
                throw std::logic_error("Prefix section placement overlaps a live section");
        }
        for (const auto &[offset, extent] : extents_)
        {
            if (extent.state != State::Free)
                continue;
            size_t consumed = 0u;
            for (size_t section = 0u; section < sections.size(); ++section)
            {
                const size_t placement = (*offsets)[section];
                if (sections[section] == 0u || placement < offset ||
                    placement - offset >= extent.bytes)
                    continue;
                if (placement != offset + consumed ||
                    sections[section] > extent.bytes - consumed)
                    throw std::logic_error("Prefix section placement violates its free-range plan");
                consumed += sections[section];
            }
            if (consumed == 0u)
                continue;
            replaced_offsets.push_back(offset);
            // Serialized sections are DMA/memcpy bytes, not typed tensor loads.
            // Exact odd-byte placement adds no unadmitted alignment padding.
            if (consumed < extent.bytes &&
                !replacements.emplace(offset + consumed,
                    Extent{.bytes = extent.bytes - consumed}).second)
                throw std::logic_error("Prefix section placement overlaps its free suffix");
        }

        for (const size_t offset : replaced_offsets)
            extents_.erase(offset);
        extents_.merge(replacements);
        if (!replacements.empty())
        {
            LOG_ERROR("[PrefixHostArena] atomic section placement conflicts with retained extents");
            std::terminate();
        }
        for (const auto &lease : leases)
            if (lease)
                lease->arena = self;
        return owners;
    }

    std::shared_ptr<void> PrefixHostArena::acquire(
        size_t bytes, std::shared_ptr<PrefixPayloadReadiness> readiness)
    {
        if (bytes == 0u)
            return {};
        const std::array sections{bytes};
        auto owners = acquireSections(sections, std::move(readiness));
        return owners.empty() ? std::shared_ptr<void>{} : std::move(owners.front());
    }
} // namespace llaminar2
