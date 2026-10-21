/**
 * @file HostImmutableRowProvider.cpp
 * @brief Preserve the exact native host owner and successful per-call observation.
 *
 * Adoption authenticates existing source ownership without gathering rows.
 * Dispatch adds no mutex, physical allocation, capacity choice, fallback,
 * scheduler, mutable directory or provider-side counter/ordinal authority.
 */
#include "HostImmutableRowProvider.h"
#include <stdexcept>

namespace llaminar2
{
std::shared_ptr<HostImmutableRowProvider> HostImmutableRowProvider::fromWholeTable(WholeOwner table)
{
    if (!table || (table->storage() != ImmutableRowStorage::RAM &&
                   table->storage() != ImmutableRowStorage::MemoryMapped))
        throw std::invalid_argument("Host native provider requires one complete RAM/mmap owner");
    table->authenticateSource();
    return std::shared_ptr<HostImmutableRowProvider>(new HostImmutableRowProvider(std::move(table)));
}

std::shared_ptr<HostImmutableRowProvider> HostImmutableRowProvider::fromAdaptiveCache(AdaptiveOwner cache)
{
    if (!cache || cache->subordinateRequester())
        throw std::invalid_argument("Host native provider requires one complete independent adaptive owner");
    cache->authenticateSource();
    return std::shared_ptr<HostImmutableRowProvider>(new HostImmutableRowProvider(std::move(cache)));
}

std::shared_ptr<HostImmutableRowProvider> HostImmutableRowProvider::fromSubordinateAdaptiveCache(AdaptiveOwner cache)
{
    if(!cache || !cache->subordinateRequester())
        throw std::invalid_argument("Subordinate native provider requires exact GPU requester geometry");
    cache->authenticateSource();
    return std::shared_ptr<HostImmutableRowProvider>(new HostImmutableRowProvider(std::move(cache)));
}
std::optional<PhysicalMemoryAllocatorIdentity> HostImmutableRowProvider::subordinateRequester() const noexcept
{
    if(const auto *cache=std::get_if<AdaptiveOwner>(&owner_))return (*cache)->subordinateRequester();
    return std::nullopt;
}
void HostImmutableRowProvider::bindSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel)
{
    if(!subordinateRequester())throw std::invalid_argument("Independent source cannot bind a subordinate processor");
    std::get<AdaptiveOwner>(owner_)->bindSubordinate(channel);
}
void HostImmutableRowProvider::retireSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel) noexcept
{
    if(auto *cache=std::get_if<AdaptiveOwner>(&owner_))(*cache)->retireSubordinate(channel);
}
CpuAdaptiveImmutableRowCache::ReadGrant HostImmutableRowProvider::authorizeSubordinate(
    const qwen4exp::ImmutableLookupHostProcessor *channel,std::uint64_t generation,std::uint64_t sequence,
    std::span<const std::int64_t> ids,std::span<const ImmutableRowSubordinateIntent> intents,
    std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements)
{
    return std::get<AdaptiveOwner>(owner_)->authorizeSubordinate(channel,generation,sequence,ids,intents,acknowledgements);
}
CpuImmutableRowGatherObservation HostImmutableRowProvider::gatherSubordinate(
    const CpuAdaptiveImmutableRowCache::ReadGrant &grant,std::span<std::byte> destination)
{
    return std::get<AdaptiveOwner>(owner_)->gatherImpl(grant.ids_,destination,&grant);
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> HostImmutableRowProvider::pinSubordinate(
    const CpuAdaptiveImmutableRowCache::ReadGrant &grant,std::size_t response_row)
{
    return std::get<AdaptiveOwner>(owner_)->pinSubordinate(grant,response_row);
}

HostImmutableRowMode HostImmutableRowProvider::mode() const noexcept
{
    if (const auto *table = std::get_if<WholeOwner>(&owner_))
        return (*table)->storage() == ImmutableRowStorage::RAM
            ? HostImmutableRowMode::WholeRAM : HostImmutableRowMode::WholeMemoryMapped;
    return HostImmutableRowMode::AdaptiveRAMSSD;
}

std::optional<std::size_t> HostImmutableRowProvider::maximumLiveRows() const noexcept
{
    if(const auto *cache=std::get_if<AdaptiveOwner>(&owner_))return (*cache)->maximumLiveRows();
    return std::nullopt;
}
ImmutableRowStorage HostImmutableRowProvider::wholeStorage() const
{
    if(const auto *table=std::get_if<WholeOwner>(&owner_))return (*table)->storage();
    throw std::logic_error("Adaptive RAM/SSD is not a whole mmap tier");
}
ImmutableRowTableStats HostImmutableRowProvider::sourceStats() const
{
    if(const auto *table=std::get_if<WholeOwner>(&owner_))return (*table)->stats();
    return std::get<AdaptiveOwner>(owner_)->sourceStats();
}
const ImmutableRowTableSource &HostImmutableRowProvider::source() const noexcept
{
    return std::visit([](const auto &owner) -> const ImmutableRowTableSource & { return owner->source(); }, owner_);
}

const ImmutableRowFileIdentity &HostImmutableRowProvider::identity() const noexcept
{
    return std::visit([](const auto &owner) -> const ImmutableRowFileIdentity & { return owner->identity(); }, owner_);
}

void HostImmutableRowProvider::authenticateSource() const
{
    std::visit([](const auto &owner) { owner->authenticateSource(); }, owner_);
}

HostImmutableRowOwnerObservation HostImmutableRowProvider::retainedOwnerObservation() const
{
    return std::visit([](const auto &owner) { return owner->retainedOwnerObservation(); }, owner_);
}

HostImmutableRowGatherObservation HostImmutableRowProvider::gather(
    std::span<const std::int64_t> ids, std::span<std::byte> destination)
{
    if (const auto *table = std::get_if<WholeOwner>(&owner_))
    {
        const auto work = (*table)->gather(ids, destination);
        return WholeImmutableRowGatherObservation{(*table)->storage(), ids.size(), destination.size(), work};
    }
    return std::get<AdaptiveOwner>(owner_)->gather(ids, destination);
}
} // namespace llaminar2
