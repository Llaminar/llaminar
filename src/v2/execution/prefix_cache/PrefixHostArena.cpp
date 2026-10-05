/**
 * @file PrefixHostArena.cpp
 * @brief Reuse admitted prefix bytes without native allocation or hot-path waits.
 *
 * Retirement only changes an existing placement node. A later admission queries
 * each exact producer event once and coalesces completed ranges; it never waits
 * for GPU progress or reclaims bytes still held by a request or disk writer.
 */
#include "execution/prefix_cache/PrefixHostArena.h"
#include "utils/Logger.h"

#include <algorithm>
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

    std::shared_ptr<void> PrefixHostArena::acquire(
        size_t bytes, std::shared_ptr<PrefixPayloadReadiness> readiness)
    {
        if (bytes == 0u)
            return {};
        std::lock_guard lock(mutex_);
        collectRetired();
        for (auto &[offset, extent] : extents_)
        {
            if (extent.state != State::Free || bytes > extent.bytes)
                continue;

            // Construct the lease before publishing placement. No destructor
            // can observe a Leased entry whose lifetime token was not created.
            auto lease = std::make_shared<Lease>();
            // These are serialized bytes, consumed through DMA/memcpy, never
            // typed tensor loads. Exact placement admits odd MoE extensions
            // without unplanned alignment padding or a second byte budget.
            if (bytes < extent.bytes)
                extents_.emplace(offset + bytes, Extent{.bytes = extent.bytes - bytes});
            extent.bytes = bytes;
            extent.readiness = std::move(readiness);
            extent.state = State::Leased;
            lease->offset = offset;
            lease->arena = shared_from_this();
            return std::shared_ptr<void>(std::move(lease),
                static_cast<uint8_t *>(backing_.get()) + offset);
        }
        return {};
    }
} // namespace llaminar2
