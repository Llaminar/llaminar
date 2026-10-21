/**
 * @file HostImmutableRowProvider.h
 * @brief Closed ownership of whole-tier or adaptive CPU native-row service.
 *
 * This component dispatches to one already-admitted owner. It neither selects
 * capacity nor creates a source, scheduler, payload allocation, or admission ledger.
 * Consumers retain this shared facade through their existing terminal boundary;
 * its private unique owner prevents exposing the adaptive cache's SSD service.
 * A subordinate cache uses this same owner but rejects independent gather;
 * only a bound existing host processor can acquire its closed source capability.
 */
#pragma once
#include "CpuAdaptiveImmutableRowCache.h"
#include "HostImmutableRowMode.h"
#include <variant>

namespace llaminar2
{
/** @brief Exact successful whole-tier operation, preserving its concrete mode. */
struct WholeImmutableRowGatherObservation final
{
    ImmutableRowStorage storage; ///< Actual RAM or MemoryMapped table; never Automatic/VRAM/adaptive.
    std::uint64_t rows, bytes; ///< Requested native rows/bytes, including original duplicate order.
    ImmutableRowGatherObservation windows; ///< This call's mapping work, never physical SSD reads.
};
/** @brief A successful owner's unmodified per-call work; no lifetime-counter delta is sampled. */
using HostImmutableRowGatherObservation =
    std::variant<WholeImmutableRowGatherObservation, CpuImmutableRowGatherObservation>;

/** @brief One immutable owner alternative, shared by existing synchronous host consumers. */
class HostImmutableRowProvider final
{
public:
    /**
     * @brief Adopt one already-admitted ordinary RAM/mmap table without another source owner.
     * @param table Complete unique native table; ownership transfers even if authentication fails.
     * @return Shared provider retaining exactly that physical/source owner.
     * @throws std::exception For null, non-host or stale source; no provider is published on failure.
     */
    static std::shared_ptr<HostImmutableRowProvider> fromWholeTable(std::unique_ptr<ImmutableRowTable> table);
    /**
     * @brief Adopt one checked host-coordinated adaptive cache, retaining its private SSD service.
     * @param cache Complete unique cache built from its exact layout and PMA; no separate SSD table.
     * @return Shared provider retaining exactly that physical/source owner.
     * @throws std::exception For null, retired, failed or stale source; no provider is published on failure.
     */
    static std::shared_ptr<HostImmutableRowProvider> fromAdaptiveCache(std::unique_ptr<CpuAdaptiveImmutableRowCache> cache);
    /**
     * @brief Adopt the same checked cache as a closed GPU-subordinate CPU source.
     * @param cache Unique cache with exact subordinate endpoint geometry and existing PMA leases.
     * @return Same provider owner, with independent gather forbidden; no storage-mode substitution.
     * @throws std::exception For independent, null, retired, failed or stale ownership.
     */
    static std::shared_ptr<HostImmutableRowProvider> fromSubordinateAdaptiveCache(std::unique_ptr<CpuAdaptiveImmutableRowCache> cache);
    /** @return Declared subordinate GPU requester; absent means ordinary independent host ownership. */
    std::optional<PhysicalMemoryAllocatorIdentity> subordinateRequester() const noexcept;
    /** @brief Release the actual owner only after the final retained provider handle retires. */
    ~HostImmutableRowProvider() = default;
    HostImmutableRowProvider(const HostImmutableRowProvider &) = delete;
    HostImmutableRowProvider &operator=(const HostImmutableRowProvider &) = delete;
    /** @return Actual retained implementation; adaptive service is never reported as whole mmap. */
    HostImmutableRowMode mode() const noexcept;
    /** @return Exact bounded adaptive ID capacity; whole tables need no shared request arrays. */
    std::optional<std::size_t> maximumLiveRows() const noexcept;
    /** @return Concrete whole tier; rejects adaptive instead of erasing its mode. */
    ImmutableRowStorage wholeStorage() const;
    /** @return Read-only original source-service observations, including partial failed work; never per-client evidence. */
    ImmutableRowTableStats sourceStats() const;
    /** @return Borrowed original source metadata, valid while a provider handle is retained. */
    const ImmutableRowTableSource &source() const noexcept;
    /** @return Borrowed actual file metadata identity, without payload hashing or a second identity record. */
    const ImmutableRowFileIdentity &identity() const noexcept;
    /**
     * @brief Authenticate the actual owner at a ready/reuse boundary without a gather.
     * @throws std::exception For a stale/retired/failed owner; adaptive failure poisons that same cache.
     * @note This never mints a cache ordinal, changes replacement policy or publishes successful lookup work.
     */
    void authenticateSource() const;
    /**
     * @brief Delegate ready validation and actual owner-lease observation without copying ownership.
     * @return Exact retained host/locality/PMA and storage/staging amounts for the actual arm.
     * @throws std::exception For stale/retired/failed ownership.
     * @note No source escape, new lease, allocation permit, gather ordinal or provider counter is created.
     */
    HostImmutableRowOwnerObservation retainedOwnerObservation() const;
    /**
     * @brief Delegate exact native response production to the retained actual owner.
     * @param ids Complete row IDs, preserving duplicate output order.
     * @param destination Exact admitted caller-owned native response; no payload scratch is created here.
     * @return Owner-specific per-call observation after successful copies, without changing its mode or authority.
     * @throws std::exception For subordinate access or validation/source/owner failure; no observation is returned for partial work.
     */
    HostImmutableRowGatherObservation gather(std::span<const std::int64_t> ids, std::span<std::byte> destination);
private:
    friend class qwen4exp::ImmutableLookupHostProcessor;
    /** @brief Bind one actual acquired-request processor to the retained cache authority. */
    void bindSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel);
    /** @brief Retire the sole channel without freeing old pinned reader backing. */
    void retireSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel) noexcept;
    /** @brief Mint a private capability only after the existing processor validates its actual frame. */
    CpuAdaptiveImmutableRowCache::ReadGrant authorizeSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel,
        std::uint64_t generation,std::uint64_t sequence,std::span<const std::int64_t> ids,
        std::span<const ImmutableRowSubordinateIntent> intents,
        std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements);
    /** @brief Delegate native fill and exclusive invalidation under the same cache mutex. */
    CpuImmutableRowGatherObservation gatherSubordinate(const CpuAdaptiveImmutableRowCache::ReadGrant &grant,
        std::span<std::byte> destination);
    /** @brief Borrow a resident row already authorized by the current outstanding source request. */
    std::optional<CpuAdaptiveImmutableRowCache::Pin> pinSubordinate(
        const CpuAdaptiveImmutableRowCache::ReadGrant &grant,std::size_t response_row);
    using WholeOwner = std::unique_ptr<ImmutableRowTable>;
    using AdaptiveOwner = std::unique_ptr<CpuAdaptiveImmutableRowCache>;
    /** @brief Publish only a nonnull authenticated complete whole-table owner. */
    explicit HostImmutableRowProvider(WholeOwner table) : owner_(std::move(table)) {}
    /** @brief Publish only a nonnull authenticated complete adaptive owner. */
    explicit HostImmutableRowProvider(AdaptiveOwner cache) : owner_(std::move(cache)) {}
    // Alternative and source are immutable after construction; all operational
    // serialization and policy/lifetime state remain with the existing owner.
    std::variant<WholeOwner, AdaptiveOwner> owner_;
};
} // namespace llaminar2
