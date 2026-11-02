/**
 * @file ImmutableRowTablePolicy.h
 * @brief Requested ngram preference and concrete runtime storage are separate types.
 *
 * Runtime configuration and lossless documents retain the requested preference.
 * Canonical PMA planning resolves Automatic only after all higher-priority
 * obligations have been frozen. Concrete runtime values never contain Automatic;
 * a failed allocation cannot retry a different storage mode. Mapping dimensions
 * bound SSD windows; native source capacity is not an ordinary prepared weight.
 */
#pragma once

#include "CpuImmutableRowCachePolicy.h"
#include "backends/DeviceId.h"
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
/** @brief Concrete materialized storage; preserve the existing RAM/mmap wire values. */
enum class ImmutableRowStorage : std::uint8_t { RAM = 0, MemoryMapped = 1, VRAM = 2 };
/** @brief Actual lookup transport, independent of whole-tier versus adaptive storage. */
enum class ImmutableRowLookupExecution : std::uint8_t
{
    HostProvider, ///< Complete host provider, including bounded adaptive RAM/SSD.
    ResidentGPU, ///< One complete original-source native GPU copy.
    FixedHeadShardGPU, ///< One checked partial native range; requires explicit fixed-head routing.
    AdaptiveGPUCache ///< Explicit GPU-coordinated native cache over one subordinate rank-local RAM/SSD source.
};
/**
 * @brief Classify physical native ownership without erasing whole/range lookup semantics.
 * @param execution Checked participant-local lookup execution alternative.
 * @return Whether this participant owns native GPU storage rather than a host provider.
 * @throws std::invalid_argument For unknown alternatives, never choosing a fallback.
 */
bool immutableRowLookupUsesNativeDeviceStorage(ImmutableRowLookupExecution execution);
/** @brief Project a concrete whole tier to its execution contract, rejecting unknown tiers. */
ImmutableRowLookupExecution immutableRowLookupExecution(ImmutableRowStorage storage);

/** @brief Explicit lookup distribution; no automatic entry into distributed storage. */
enum class ImmutableRowTableDistribution : std::uint8_t { Replicated, FixedHeadSharded };
/** @brief Planner provenance of the actual declaration, not a selectable topology. */
enum class ImmutableRowTableTPTopology : std::uint8_t { None, HomogeneousFullMainRankLocal };

/** @brief Requested setup preference; Automatic is never a runtime/kernel storage kind. */
enum class ImmutableRowStoragePreference : std::uint8_t { Automatic, VRAM, RAM, MemoryMapped, HostAdaptiveRAMSSD, AdaptiveVRAMRAMSSD };

/**
 * @brief Explicit bounded host RAM/SSD request; ceilings are not admitted slot counts.
 * Zero ceilings request a valid zero-slot host cache with its mandatory SSD service.
 */
struct HostAdaptiveRAMSSDRequest final
{
    std::uint64_t maximum_cache_rows = 0;
    std::size_t cache_physical_byte_limit = 0; ///< Full cache payload+metadata ceiling, excluding mandatory SSD/request arrays.
    CpuImmutableRowCachePolicy policy;
    bool operator==(const HostAdaptiveRAMSSDRequest &) const = default;
};

/** @brief Storage placement is explicit and never inferred from an empty owner vector. */
enum class AdaptiveGPUCachePlacementMode : std::uint8_t { ConsumerLocal, DeclaredRankLocal };
/** @brief Source/cache participant identity, independent of any compute or expert domain. */
struct AdaptiveGPUCacheParticipant final
{
    int world_rank = -1;
    DeviceType backend = DeviceType::CPU;
    int ordinal = -1;
    /** @return Exact declared physical endpoint; validation never initializes a runtime. */
    DeviceId device() const noexcept { return {backend,ordinal}; }
    bool operator==(const AdaptiveGPUCacheParticipant &) const = default;
};
/** @brief One ordered optional-cache request; PMA alone chooses admitted slots. */
struct AdaptiveGPUCacheOwnerRequest final
{
    AdaptiveGPUCacheParticipant participant;
    std::uint64_t maximum_cache_rows = 0;
    std::size_t cache_physical_byte_limit = 0;
    bool operator==(const AdaptiveGPUCacheOwnerRequest &) const = default;
};
/**
 * @brief Lossless source-storage declaration, not permission to create a compute domain.
 * Declared owners retain their order at C0 as well as bounded capacity. All
 * endpoints are rank-local; the exact coordinator is an independently named
 * query authority. No runtime capacity or native handle belongs in this value.
 */
struct AdaptiveGPUCachePlacementRequest final
{
    AdaptiveGPUCachePlacementMode mode = AdaptiveGPUCachePlacementMode::ConsumerLocal;
    std::vector<AdaptiveGPUCacheOwnerRequest> owners;
    std::optional<AdaptiveGPUCacheParticipant> coordinator;
    /** @brief Reject mixed arms, duplicate/foreign endpoints, and unsupported same-vendor remote owners. */
    void validate() const;
    /** @return Ordered unique declared source participants; local consumer arm returns an empty vector. */
    std::vector<AdaptiveGPUCacheParticipant> declaredParticipants() const;
    bool operator==(const AdaptiveGPUCachePlacementRequest &) const = default;
};
/**
 * @brief Parse the one public lossless placement grammar before planning.
 * @param text `consumer-local`, or `rank:cuda|rocm:ordinal/rows/bytes,...@rank:cuda|rocm:ordinal`.
 * @return Validated ordered owners and exact coordinator, without probing any device.
 * @throws std::invalid_argument For malformed, duplicate, foreign or unsupported endpoints.
 */
AdaptiveGPUCachePlacementRequest parseAdaptiveGPUCachePlacement(const std::string &text);

/**
 * @brief Explicit GPU-coordinated cache with independent storage owners and subordinate RAM/SSD.
 * GPU cache payload plus metadata is admitted first, then RAM cache payload plus
 * metadata, both after all frozen mandatory/weight/expert/KV obligations. Zero
 * ceilings retain this same mode; they do not select ordinary mmap execution.
 */
struct AdaptiveVRAMRAMSSDRequest final
{
    std::uint64_t maximum_gpu_cache_rows = 0;
    std::size_t gpu_cache_physical_byte_limit = 0;
    CpuImmutableRowCachePolicy gpu_policy; ///< One explicit admission/recency policy, executed by the exact coordinator.
    HostAdaptiveRAMSSDRequest host; ///< Independent CPU cache request; its SSD/request scratch is mandatory.
    AdaptiveGPUCachePlacementRequest placement; ///< Explicit owner policy; scalar GPU ceilings belong only to ConsumerLocal.
    /** @return Source-owned CPU worker label for plural independent consumers, otherwise the exact GPU consumer. */
    DeviceId lookupIOConsumer(DeviceId requester, std::size_t actual_consumers) const
    {
        if (!requester.is_gpu() || !actual_consumers)
            throw std::invalid_argument("Adaptive lookup service requires actual GPU consumers");
        return placement.mode == AdaptiveGPUCachePlacementMode::ConsumerLocal && actual_consumers > 1
            ? DeviceId::cpu() : requester;
    }
    bool operator==(const AdaptiveVRAMRAMSSDRequest &) const = default;
};

inline constexpr std::size_t kImmutableRowDefaultWindowBytes = 64u * 1024u;
inline constexpr std::size_t kImmutableRowDefaultWindowCount = 1024u;

/**
 * @brief Checked concrete mode/cache geometry, without an allocation/admission claim.
 *
 * The loader owns byte sizing for this value, including RAM page alignment and
 * the exact mmap directory. Candidate planning may construct it before admission;
 * only a sealed placement authorizes actual storage materialization.
 */
class ImmutableRowTableStoragePolicy final
{
public:
    /**
     * @brief Validate one concrete storage mode and exact retained-window geometry.
     * @param storage Actual VRAM, RAM or bounded mmap mode; no Automatic exists here.
     * @param window_bytes Page-aligned bytes per mmap window, retained losslessly.
     * @param window_count Positive bounded window count.
     * @return Immutable concrete sizing input, never permission to allocate.
     * @throws std::invalid_argument For unknown mode, invalid windows or overflow.
     */
    static ImmutableRowTableStoragePolicy forStorage(ImmutableRowStorage storage,
        std::size_t window_bytes = kImmutableRowDefaultWindowBytes,
        std::size_t window_count = kImmutableRowDefaultWindowCount);
    /** @return Concrete materialized storage kind. */
    ImmutableRowStorage storage() const noexcept { return storage_; }
    /** @return Exact configured bytes per bounded mmap window. */
    std::size_t windowBytes() const noexcept { return window_bytes_; }
    /** @return Exact configured maximum retained mmap windows. */
    std::size_t windowCount() const noexcept { return window_count_; }
    bool operator==(const ImmutableRowTableStoragePolicy &) const = default;
private:
    /** @brief Only the validating factory constructs concrete sizing inputs. */
    ImmutableRowTableStoragePolicy(ImmutableRowStorage storage, std::size_t bytes, std::size_t count)
        : storage_(storage), window_bytes_(bytes), window_count_(count) {}
    ImmutableRowStorage storage_;
    std::size_t window_bytes_, window_count_;
};

/** @brief Lossless requested policy, resolved once by canonical physical-memory admission. */
struct ImmutableRowTablePolicy
{
    ImmutableRowStoragePreference storage = ImmutableRowStoragePreference::Automatic;
    std::size_t window_bytes = kImmutableRowDefaultWindowBytes; ///< Page-aligned extent of one SSD mapping.
    std::size_t window_count = kImmutableRowDefaultWindowCount; ///< Maximum simultaneously retained SSD windows.

    HostAdaptiveRAMSSDRequest host_adaptive; ///< Used only by the explicit HostAdaptiveRAMSSD request arm.

    ImmutableRowTableDistribution distribution = ImmutableRowTableDistribution::Replicated;

    AdaptiveVRAMRAMSSDRequest gpu_adaptive; ///< Used only by explicit AdaptiveVRAMRAMSSD; no Automatic entry.

    /** @return Whether the request explicitly selects fixed native head shards. */
    bool isFixedHeadSharded() const noexcept { return distribution == ImmutableRowTableDistribution::FixedHeadSharded; }
    /** @return Whether this checked request chooses explicit CPU-owned adaptive RAM/SSD. */
    bool isHostAdaptive() const noexcept { return storage == ImmutableRowStoragePreference::HostAdaptiveRAMSSD; }
    /** @return Whether this request explicitly selects GPU-coordinated mixed native caching. */
    bool isGPUAdaptive() const noexcept { return storage == ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD; }
    /**
     * @brief Identify the original I/O worker declaration for this authored lookup policy.
     * @param requester Actual execution consumer, never a storage-only owner.
     * @param actual_consumers Exact concurrent consumer population of this source.
     * @return CPU0 source identity for grouped ConsumerLocal channels; otherwise original consumer.
     * @note This immutable label is not a new allocation or a global GPU coordinator.
     */
    DeviceId lookupIOConsumer(DeviceId requester, std::size_t actual_consumers) const
    {
        if (!actual_consumers) throw std::invalid_argument("Lookup service has no actual consumers");
        return isGPUAdaptive() ? gpu_adaptive.lookupIOConsumer(requester, actual_consumers) : requester;
    }
    /** @brief Reject unknown preferences, mixed arms, invalid policy and bounded-cache overflow. */
    void validate() const;
    bool operator==(const ImmutableRowTablePolicy &) const = default;
};
} // namespace llaminar2
