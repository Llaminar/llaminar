/**
 * @file AdaptiveImmutableLookupGeometry.h
 * @brief Exact pooled GPU owner, coordinator and RAM/SSD source declarations.
 *
 * The resolved layout and explicit participant order feed the original native
 * root factories. PhysicalMemoryAuthority remains the sole admission ledger.
 * An empty-cache declaration separates fixed execution storage from optional
 * root growth without subtracting available memory or changing model/KV/graph
 * admission. Directed transport contributes its own source-owned BOM separately;
 * this declaration grants no transport or serving capability. The existing
 * single-device spelling uses the same factory. Existing ordering programs are
 * priced by a sealed source-bound projection in both the C0 floor and optional
 * quota growth; native staging bytes cannot stand in for their commitments.
 */
#pragma once
#include "backends/NativeOrderingEventBudgetProjection.h"
#include "CpuAdaptiveImmutableRowCache.h"
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include "execution/qwen4exp/ImmutableLookupServiceGeometry.h"
#include "execution/qwen4exp/PooledImmutableReplicatedReaderGeometry.h"
#include "execution/qwen4exp/PooledImmutableLocalHotTransportGeometry.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotGeometry.h"
#include <optional>
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
class MemoryPlanner;
class ImmutableRowTablePlacementPlanner;
namespace test {struct AdaptiveImmutableLookupGeometryReplayAccess;}
/** @brief Immutable physical declarations, with no live residency or capacity balance. */
class AdaptiveImmutableLookupGeometry final
{
public:
    /**
     * @brief Declare one original source around a singleton or explicitly shared reader cohort.
     * @param layout Original source, owner and consumer layout.
     * @param source Native source metadata; payload is never inspected.
     * @param request Original bounded GPU/RAM policies.
     * @return The same checked geometry used by explicit owner cohorts.
     * @throws std::invalid_argument For a missing consumer or additional unbound owners.
     */
    static AdaptiveImmutableLookupGeometry fromLayout(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request)
    {
        return fromLayoutWithLocalHot(layout, source, request, {});
    }

    /**
     * @brief Declare candidate local cache geometry over exact original consumers.
     * @param layout Original shared placement, without changing its pool owners.
     * @param source Original native format/row geometry.
     * @param request Lossless local policy and overrides.
     * @param slots Candidate/admitted local slots in original consumer order, not requested caps.
     * @return Static byte declarations only, with no materialization permission.
     * @throws std::exception For incomplete/foreign identity or unrepresentable geometry.
     */
    static std::vector<qwen4exp::PooledImmutableLocalHotGeometry> declareLocalHotCaches(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request, std::span<const std::size_t> slots)
    {
        if (!request.local_hot_cache)
        {
            if (!slots.empty()) throw std::invalid_argument("Shared-only placement cannot declare local slots");
            return {};
        }
        if (request.placement.mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal || slots.size() != layout.consumers().size())
            throw std::invalid_argument("Local hot slots require every original shared compute reader");
        std::vector<AdaptiveGPUCacheParticipant> consumers;
        for (const auto id : layout.consumers()) consumers.push_back({id.world_rank,id.device.type,id.device.ordinal});
        request.local_hot_cache->requireConsumers(consumers);
        const auto& p = request.local_hot_cache->policy;
        const qwen4exp::adaptive_lookup::Policy policy{p.aging_accesses,p.protected_numerator,p.protected_denominator};
        std::vector<qwen4exp::PooledImmutableLocalHotGeometry> result;
        result.reserve(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            if (slots[i] > qwen4exp::adaptive_lookup::MaximumSlots || layout.maximumInflightRows() > qwen4exp::adaptive_lookup::MaximumSlots)
                throw std::invalid_argument("Local hot geometry exceeds the original native index bound");
            result.push_back(qwen4exp::PooledImmutableLocalHotGeometry::declare(source,layout.consumers()[i],
                static_cast<std::uint32_t>(layout.maximumInflightRows()),static_cast<std::uint32_t>(slots[i]),policy));
        }
        return result;
    }

    /**
     * @brief Price only an explicitly named cold mandatory local C0 floor.
     * @param layout Original shared layout. @param source Original native source.
     * @param request Original local/shared policy.
     * @return Complete mandatory geometry; never an implicit runtime replacement for admitted local slots.
     * @throws std::exception For invalid original declarations.
     */
    static AdaptiveImmutableLookupGeometry fromLayoutFloor(const ImmutableRowTableLayoutGeometry& layout,
        const ImmutableRowTableSource& source, const AdaptiveVRAMRAMSSDRequest& request)
    {
        const std::vector<std::size_t> zero(request.local_hot_cache ? layout.consumers().size() : 0u,0u);
        const auto local = declareLocalHotCaches(layout,source,request,zero);
        return fromLayoutWithLocalHot(layout,source,request,local);
    }

    /**
     * @brief Bind original admitted local cache declarations without selecting slots.
     * @param layout Same original shared-pool/source placement.
     * @param source Exact immutable native source geometry.
     * @param request Same lossless shared/local intent.
     * @param local_hot Exact actual-consumer order and already-admitted local geometries, including C0.
     * @return One complete local/pool/source declaration, with no additional authority.
     * @throws std::exception For absent/foreign local declarations or incompatible bounds.
     */
    static AdaptiveImmutableLookupGeometry fromLayoutWithLocalHot(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request,
        std::span<const qwen4exp::PooledImmutableLocalHotGeometry> local_hot)
    {
        if (layout.consumers().empty())
            throw std::invalid_argument("Adaptive lookup requires original GPU consumers");
        if(request.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
        {
            request.placement.validate();
            (void)sourceLayout(layout,request); // Authored coordinator membership, never consumer enumeration order.
            std::vector<PhysicalMemoryAllocatorIdentity> owners;
            for(const auto &owner:request.placement.owners)
                owners.push_back({owner.participant.world_rank,owner.participant.device()});
            return fromLayout(layout,source,request,owners,local_hot);
        }
        if(layout.consumers().size()!=1)
            throw std::invalid_argument("ConsumerLocal plural geometry requires exact per-requester projection");
        const std::array owners{layout.consumers().front()};
        return fromLayout(layout, source, request, owners,local_hot);
    }

    /**
     * @brief Project one authored source channel without cloning the shared storage layout.
     * @param layout Full original owner/reader layout, retained by resolved placement.
     * @param request Original policy naming the exact shared coordinator.
     * @return Coordinator-only CPU-source view preserving every physical storage owner.
     * @throws std::invalid_argument For foreign coordinator, incompatible reader domain or unbound policy.
     */
    static ImmutableRowTableLayoutGeometry sourceLayout(
        const ImmutableRowTableLayoutGeometry& layout,const AdaptiveVRAMRAMSSDRequest& request)
    {
        if(layout.sourceServices().size()!=1 || layout.sourceServices().front().windows.storage()!=request.sourceStorage())
            throw std::invalid_argument("Adaptive source differs from its explicit original backing");
        if(layout.consumers().empty())throw std::invalid_argument("Adaptive source lost its original consumers");
        if(request.placement.mode!=AdaptiveGPUCachePlacementMode::DeclaredRankLocal){
            if(layout.consumers().size()!=1)
                throw std::invalid_argument("Plural ConsumerLocal sources require their grouped provider contract");
            return layout;
        }
        request.placement.validate();
        const PhysicalMemoryAllocatorIdentity coordinator{request.placement.coordinator->world_rank,request.placement.coordinator->device()};
        if(layout.adaptiveGPUDomain()!=ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator ||
           std::find(layout.consumers().begin(),layout.consumers().end(),coordinator)==layout.consumers().end())
            throw std::invalid_argument("Shared source coordinator is not its authored original compute consumer");
        for(const auto consumer:layout.consumers())
            if(consumer.world_rank!=coordinator.world_rank || consumer.device.type!=coordinator.device.type)
                throw std::invalid_argument("Shared source readers must be original homogeneous rank-local GPUs");
        return ImmutableRowTableLayoutGeometry::checked(layout.extent(),layout.tiers(),layout.residency(),
            {layout.owners().begin(),layout.owners().end()},{coordinator},{},
            {layout.sourceServices().begin(),layout.sourceServices().end()},{},layout.gpuCoordinator(),
            layout.maximumInflightRows(),ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,layout.peerAccess());
    }

    /**
     * @brief Project an explicitly consumer-local cache without losing the plural source layout.
     * @param layout Original plural owner/consumer geometry; retained by placement and caller.
     * @param source Original native row metadata.
     * @param request Explicit ConsumerLocal policy, never a shared-owner declaration.
     * @param requester Exact original consumer key, independent of TP enumeration order.
     * @return Existing single-requester GPU roots around that consumer's own cache.
     * @throws std::invalid_argument For shared-cohort policy or foreign requester identity.
     * @note Its host contribution is a view of shared backing, not permission to sum it per consumer.
     */
    static AdaptiveImmutableLookupGeometry forRequester(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request, PhysicalMemoryAllocatorIdentity requester)
    {
        if (request.placement.mode != AdaptiveGPUCachePlacementMode::ConsumerLocal ||
            layout.adaptiveGPUDomain() != ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal ||
            std::find(layout.consumers().begin(), layout.consumers().end(), requester) == layout.consumers().end())
            throw std::invalid_argument("Consumer-local geometry requires its exact original requester and policy");
        std::vector<ImmutableRowTableLayoutOwner> owners;
        for (const auto& owner : layout.owners())
            if (owner.key.tier != ImmutableRowStorage::VRAM || owner.key.allocator == requester)
                owners.push_back(owner);
        const auto local = ImmutableRowTableLayoutGeometry::checked(
            layout.extent(), layout.tiers(), layout.residency(), std::move(owners), {requester}, {},
            {layout.sourceServices().begin(), layout.sourceServices().end()}, {},
            layout.gpuCoordinatorFor(requester), layout.maximumInflightRows());
        return fromLayout(local, source, request);
    }

    /**
     * @brief Declare independent GPU storage owners around the original query coordinator.
     * @param layout Resolved layout, including zero-capacity owner entries.
     * @param source Original source format/extents, without a fabricated runtime identity.
     * @param request Original GPU/host policies; geometry makes no placement decision.
     * @param ordered_owners Complete physical ordering, independent of TP indices.
     * @return Exact per-allocator roots and their fixed empty-cache portions.
     * @throws std::exception For incomplete ownership, incompatible geometry or overflow.
     * @note The coordinator may own slots or be separate from every storage owner.
     * This factory does not authorize native, inter-vendor or cross-rank transport.
     */
    static AdaptiveImmutableLookupGeometry fromLayout(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request,
        std::span<const PhysicalMemoryAllocatorIdentity> ordered_owners,
        std::span<const qwen4exp::PooledImmutableLocalHotGeometry> local_hot = {})
    {
        const auto source_layout=sourceLayout(layout,request);
        if (source_layout.consumers().size() != 1 || !source_layout.consumers().front().device.is_gpu() ||
            layout.sourceServices().size() != 1 || !layout.maximumInflightRows() ||
            layout.maximumInflightRows() % 16 ||
            layout.maximumInflightRows() > qwen4exp::adaptive_lookup::MaximumSlots)
            throw std::invalid_argument("Adaptive lookup requires one original GPU/source domain and complete sixteen-head query geometry");
        const auto requester = source_layout.consumers().front();
        const auto physical = qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(
            source_layout, ordered_owners, requester);
        const qwen4exp::adaptive_lookup::Policy policy{
            request.gpu_policy.aging_accesses, request.gpu_policy.protected_numerator,
            request.gpu_policy.protected_denominator};
        std::optional<qwen4exp::PooledImmutableReplicatedReaderGeometry> readers;
        if (layout.consumers().size() > 1)
        {
            if (source.row_elements > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Shared reader output width exceeds its original ABI");
            readers = qwen4exp::PooledImmutableReplicatedReaderGeometry::native(requester, layout.consumers(),
                static_cast<std::uint32_t>(layout.maximumInflightRows()), static_cast<std::uint32_t>(source.row_elements));
        }
        std::optional<qwen4exp::PooledImmutableLocalHotTransportGeometry> compact;
        if (request.local_hot_cache)
        {
            if (request.placement.mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal ||
                local_hot.size() != layout.consumers().size())
                throw std::invalid_argument("Local hot geometry requires every original declared shared consumer, including C0");
            std::vector<AdaptiveGPUCacheParticipant> consumers;
            for (const auto consumer : layout.consumers())
                consumers.push_back({consumer.world_rank, consumer.device.type, consumer.device.ordinal});
            request.local_hot_cache->requireConsumers(consumers);
            const auto& local_policy = request.local_hot_cache->policy;
            const qwen4exp::adaptive_lookup::Policy policy{local_policy.aging_accesses,
                local_policy.protected_numerator, local_policy.protected_denominator};
            for (std::size_t i = 0; i < local_hot.size(); ++i)
            {
                const auto expected = qwen4exp::PooledImmutableLocalHotGeometry::declare(source,layout.consumers()[i],
                    static_cast<std::uint32_t>(layout.maximumInflightRows()),local_hot[i].slots(),policy);
                if (local_hot[i] != expected)
                    throw std::invalid_argument("Local hot geometry changed its original consumer/source/query/policy identity");
            }
            compact = readers
                ? qwen4exp::PooledImmutableLocalHotTransportGeometry::declare(*readers,layout.extent())
                : qwen4exp::PooledImmutableLocalHotTransportGeometry::singleton(requester,
                    static_cast<std::uint32_t>(layout.maximumInflightRows()),layout.extent());
        }
        else if (!local_hot.empty())
            throw std::invalid_argument("Shared-only lookup cannot retain undeclared local hot geometry");
        // N contains compact per-reader misses; U and CPU source work remain
        // bounded by the one original replicated model query, never P*U.
        const auto maximum_pool_original = compact ? compact->maximumPoolOriginalRows()
            : static_cast<std::uint32_t>(layout.maximumInflightRows());
        const auto coordinator = qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(
            physical, policy, maximum_pool_original);
        const auto host = CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(
            source_layout, layout.sourceServices().front().allocator, requester, request.host.policy);

        // Project the SAME cohort with no optional GPU slots. Source/query/owner
        // ordering survives; this is not another placement solver or ledger.
        auto owners = std::vector<ImmutableRowTableLayoutOwner>(layout.owners().begin(), layout.owners().end());
        for (auto& owner : owners)
            if (owner.key.tier == ImmutableRowStorage::VRAM) owner.cache_rows = 0;
        const auto zero = ImmutableRowTableLayoutGeometry::checked(
            layout.extent(), layout.tiers(), layout.residency(), std::move(owners),
            {requester}, {}, {layout.sourceServices().begin(), layout.sourceServices().end()},
            {}, std::nullopt, layout.maximumInflightRows(), ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator, layout.peerAccess());
        const auto floor = qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(
            zero, ordered_owners, requester);
        const auto floor_policy = qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(
            floor, policy, maximum_pool_original);
        for (std::size_t owner = 0; owner < physical.owners().size(); ++owner)
            if (physical.owners()[owner].frame.bytes != floor.owners()[owner].frame.bytes)
                throw std::logic_error("Adaptive fixed owner frame unexpectedly depends on optional slot capacity");
        if (coordinator.frameOffsets().bytes != floor_policy.frameOffsets().bytes)
            throw std::logic_error("Adaptive fixed coordinator frame unexpectedly depends on optional slot capacity");
        auto actual_roots = NativeRoots::from(physical, coordinator, source);
        auto fixed_roots = NativeRoots::from(floor, floor_policy, source);
        if (actual_roots.source.native_bytes != fixed_roots.source.native_bytes)
            throw std::logic_error("Adaptive source metadata unexpectedly depends on optional slot capacity");
        return {physical, coordinator, host, std::move(actual_roots), std::move(fixed_roots),
            std::move(readers), {local_hot.begin(),local_hot.end()}, std::move(compact)};
    }

    /**
     * @brief Declare one CPU source/cache with all original consumer grant lanes.
     * @param layout Exact plural ConsumerLocal layout, retained by the placement owner.
     * @param request Original independent GPU and shared host policy.
     * @return Shared backing geometry; consumers must not materialize per-requester copies.
     */
    static CpuAdaptiveImmutableRowCacheGeometry sharedHostGeometry(
        const ImmutableRowTableLayoutGeometry& layout, const AdaptiveVRAMRAMSSDRequest& request)
    {
        if (request.placement.mode != AdaptiveGPUCachePlacementMode::ConsumerLocal ||
            layout.adaptiveGPUDomain() != ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal ||
            layout.sourceServices().size() != 1 || layout.sourceServices().front().windows.storage()!=request.sourceStorage())
            throw std::invalid_argument("Grouped CPU source requires exact consumer-local placement and original backing");
        return CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPUs(
            layout, layout.sourceServices().front().allocator, request.host.policy);
    }

    /**
     * @brief Price concurrent local GPU programs, shared CPU backing and disjoint source channels once.
     * @param layout Full original ConsumerLocal source/owner/consumer geometry.
     * @param source Exact original native format and row extents.
     * @param request Authored independent GPU and shared host policies.
     * @param devices Complete unique original GPU resource observations.
     * @param host Original canonical CPU0 observation for the shared source.
     * @param projection Sealed original or recorded event budget projection.
     * @return Mandatory typed contributions before optional quotas.
     * @throws std::exception For incompatible endpoints, source geometry or arithmetic overflow.
     */
    static PhysicalMemoryPlan consumerLocalMandatoryBOM(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request, std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host, const NativeOrderingEventBudgetProjection& projection)
    {
        return consumerLocalBOM(layout, source, request, devices, host, projection, ConsumerLocalContribution::Mandatory);
    }
    /**
     * @brief Price optional local GPU cache roots and one shared CPU cache, without another source/channel.
     * @param layout Complete original candidate layout, including C0 physical owners.
     * @param source Original native format and row extents.
     * @param request Authored independent GPU and shared host policies.
     * @param devices Complete unique original GPU resource observations.
     * @param host Original canonical CPU0 source observation.
     * @param projection Sealed event projection shared with the mandatory floor.
     * @return C0-relative cache/root growth on the actual allocators.
     * @throws std::exception For incompatible endpoints, source geometry or arithmetic overflow.
     */
    static PhysicalMemoryPlan consumerLocalSlotDependentBOM(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request, std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host, const NativeOrderingEventBudgetProjection& projection)
    {
        return consumerLocalBOM(layout, source, request, devices, host, projection, ConsumerLocalContribution::Optional);
    }

    /** @return Optional exact result/ACK cohort; absence retains existing singleton semantics. */
    const std::optional<qwen4exp::PooledImmutableReplicatedReaderGeometry>& readerGeometry()const noexcept{return readers_;}
    /** @return Exact admitted local roots, in original compute-reader order; empty in shared-only mode. */
    std::span<const qwen4exp::PooledImmutableLocalHotGeometry> localHotCaches() const noexcept { return local_hot_; }
    /** @return Compact native request/result transport replacing old FP32 reader channels when engaged. */
    const std::optional<qwen4exp::PooledImmutableLocalHotTransportGeometry>& localHotTransport() const noexcept { return compact_; }
    /** @return Actual compute readers, independent of the single original CPU source channel. */
    std::uint32_t computeReaderCount()const noexcept{return readers_?static_cast<std::uint32_t>(readers_->consumers().size()):1u;}

    /** @return Original ordered physical declarations, including C0 participants. */
    const qwen4exp::PooledImmutablePhysicalOwnerGeometry& physical() const noexcept { return physical_; }
    /** @return Original coordinator policy/frame declaration, charged once. */
    const qwen4exp::PooledImmutableCoordinatorGeometry& coordinator() const noexcept { return coordinator_; }
    /** @return Original bounded CPU RAM/SSD provider declaration, charged once. */
    const CpuAdaptiveImmutableRowCacheGeometry& host() const noexcept { return host_; }
    /** @return Complete ID envelope; runtime counts remain device-owned. */
    std::uint32_t maximumIds() const noexcept { return physical_.maximumUniqueRows(); }
    /** @return Exact subordinate source envelope, independent of storage-owner count. */
    qwen4exp::ImmutableLookupLayout channelLayout() const
    {
        return qwen4exp::ImmutableLookupLayout::createSubordinate(
            coordinator_.parameters().maximumUnique(), physical_.rowBytes());
    }

    /** @return One-device cache/policy extent; larger cohorts must name an allocator. */
    std::size_t gpuCacheBytes() const { return gpuCacheBytes(singleAllocator()); }
    /** @return One-device optional growth; never charge different GPUs to one capacity. */
    std::size_t gpuSlotDependentBytes() const { return gpuSlotDependentBytes(singleAllocator()); }
    /** @return One-device execution/source extent; larger cohorts must name an allocator. */
    std::size_t mandatoryDeviceBytes() const { return mandatoryDeviceBytes(singleAllocator()); }

    /**
     * @brief Project optional cache/policy roots onto one original GPU allocator.
     * @param allocator Actual storage owner or coordinator.
     * @return Physical cache bytes plus policy bytes only when this GPU owns them.
     * @throws std::invalid_argument For a foreign allocator.
     */
    std::size_t gpuCacheBytes(PhysicalMemoryAllocatorIdentity allocator) const
    {
        requireAllocator(allocator);
        std::size_t bytes = allocator == physical_.coordinator() ? coordinator_.policyOffsets().bytes : 0;
        for (const auto& owner : physical_.owners())
            if (owner.allocator == allocator) bytes = plus(bytes, owner.cache.bytes);
        return bytes;
    }
    /**
     * @brief Project optional roots and exact bank growth onto their actual allocator.
     * @param allocator Original owner or coordinator; no physical capacity is read.
     * @return GPU contribution beyond the same cohort's C0 declaration.
     * @throws std::exception For a foreign allocator, invalid growth or overflow.
     */
    std::size_t gpuSlotDependentBytes(PhysicalMemoryAllocatorIdentity allocator) const
    { return plus(gpuCacheBytes(allocator), nativeGrowth(allocator)); }
    /**
     * @brief Project mandatory owner/source execution roots onto one allocator.
     * @param allocator Actual original owner or coordinator.
     * @return Exact extents; shared source roots appear only on the coordinator.
     * @throws std::exception For a foreign allocator or extent overflow.
     */
    std::size_t mandatoryDeviceBytes(PhysicalMemoryAllocatorIdentity allocator) const
    {
        requireAllocator(allocator);
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < physical_.owners().size(); ++i)
            if (physical_.owners()[i].allocator == allocator)
                bytes = plus(bytes, plus(physical_.owners()[i].frame.bytes, actual_.owners[i]));
        if (allocator == physical_.coordinator())
            for (const auto size : {coordinator_.frameOffsets().bytes, actual_.coordinator,
                                   actual_.source.frame_bytes, actual_.source.native_bytes})
                bytes = plus(bytes, size);
        return bytes;
    }

    /**
     * @brief Project actual local mandatory categories independently of the shared source.
     * @param devices Same complete original allocator observations.
     * @param host Original CPU resource; its LookupStaging contribution names only LocalReady pages.
     * @param projection Original installed or closed recorded event-resource origin.
     * @return Separate local GPU frame/native and CPU ready-page BOM; opaque event category remains distinct.
     * @throws std::exception For missing/foreign observations or unavailable native bounds.
     */
    PhysicalMemoryPlan localHotMandatoryBOM(std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host, const NativeOrderingEventBudgetProjection& projection) const
    {
        requireDevices(devices);
        PhysicalMemoryPlanBuilder result;
        for (const auto& local : local_hot_)
            append(result,local.mandatoryBOM(resource(devices,local.allocator()),host,projection));
        return result.build();
    }

    /**
     * @brief Project only admitted local optional roots/event growth, independent of pool caps.
     * @param devices Same complete original allocator observations.
     * @param host Original CPU resource for setup events and bounded probe publication.
     * @param projection Original installed or closed recorded native-resource origin.
     * @return Exact per-consumer optional L1 BOM; absent local intent contributes nothing.
     * @throws std::exception For missing/foreign resources or unavailable original bounds.
     */
    PhysicalMemoryPlan localHotSlotDependentBOM(std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host, const NativeOrderingEventBudgetProjection& projection) const
    {
        requireDevices(devices);
        PhysicalMemoryPlanBuilder result;
        for (const auto& local : local_hot_)
            append(result,local.slotDependentBOM(resource(devices,local.allocator()),host,projection));
        return result.build();
    }

    /** @return Existing one-GPU spelling of the exact mandatory root contributor. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource gpu, PhysicalMemoryResource host) const
    { return mandatoryBOM(std::span{&gpu, 1}, host); }
    /** @return Existing one-GPU spelling of the optional cache root contributor. */
    PhysicalMemoryPlan cacheBOM(PhysicalMemoryResource gpu, PhysicalMemoryResource host) const
    { return cacheBOM(std::span{&gpu, 1}, host); }
    /** @return Existing one-GPU spelling of the C0-relative optional contribution. */
    PhysicalMemoryPlan slotDependentBOM(PhysicalMemoryResource gpu, PhysicalMemoryResource host) const
    { return slotDependentBOM(std::span{&gpu, 1}, host); }

    /**
     * @brief Contribute every owner, coordinator and source execution root once.
     * @param devices Exact unique owner/coordinator observations, in any order.
     * @param host Original source CPU observation.
     * @return Typed cache/source roots; directed transport contributes its own BOM.
     * @throws std::exception For incomplete/foreign observations or overflow.
     */
    PhysicalMemoryPlan mandatoryBOM(std::span<const PhysicalMemoryResource> devices,
                                    PhysicalMemoryResource host) const
    { return mandatoryBOMWithOrderingProjection(devices,host,NativeOrderingEventBudgetProjection::installed()); }
    /**
     * @brief Contribute optional physical cache/policy roots without execution duplicates.
     * @param devices Exact unique owner/coordinator resource observations.
     * @param host Original CPU source observation.
     * @return Native cache roots, one coordinator policy root and one host cache.
     * @throws std::exception For incomplete or foreign observations.
     */
    PhysicalMemoryPlan cacheBOM(std::span<const PhysicalMemoryResource> devices,
                                PhysicalMemoryResource host) const
    {
        requireDevices(devices);
        PhysicalMemoryPlanBuilder result;
        for (std::size_t i = 0; i < physical_.owners().size(); ++i)
            append(result, physical_.cacheBOM(static_cast<std::uint32_t>(i),
                resource(devices, physical_.owners()[i].allocator)));
        append(result, coordinator_.policyBOM(resource(devices, physical_.coordinator())));
        append(result, host_.cacheBOM(host));
        return result.build();
    }
    /**
     * @brief Add optional roots/growth to the same cohort's frozen C0 floor.
     * @param devices Exact owner/coordinator observations; PMA alone decides fit.
     * @param host Original CPU source observation.
     * @return Optional contribution apportioned to its actual physical allocators.
     * @throws std::exception For incomplete observations, invalid growth or overflow.
     */
    PhysicalMemoryPlan slotDependentBOM(std::span<const PhysicalMemoryResource> devices,
                                        PhysicalMemoryResource host) const
    { return slotDependentBOMWithOrderingProjection(devices,host,NativeOrderingEventBudgetProjection::installed()); }

private:
    /** @brief Select fixed versus optional contributions without duplicating channel/cache ownership. */
    enum class ConsumerLocalContribution { Mandatory, Optional };
    /** @brief Sum actual concurrent requester roots while charging shared backing exactly once. */
    static PhysicalMemoryPlan consumerLocalBOM(
        const ImmutableRowTableLayoutGeometry& layout, const ImmutableRowTableSource& source,
        const AdaptiveVRAMRAMSSDRequest& request, std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host, const NativeOrderingEventBudgetProjection& projection,
        ConsumerLocalContribution contribution)
    {
        const auto shared = sharedHostGeometry(layout, request);
        if (devices.size() != layout.consumers().size())
            throw std::invalid_argument("Consumer-local resources differ from original requester population");
        PhysicalMemoryPlanBuilder result;
        std::vector<DeviceId> channels;
        std::optional<qwen4exp::ImmutableLookupLayout> channel;
        for (const auto requester : layout.consumers()) {
            const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& value) {
                return PhysicalMemoryAllocatorIdentity{value.world_rank, value.device} == requester;
            });
            if (found == devices.end() || std::count_if(devices.begin(), devices.end(), [&](const auto& value) {
                    return PhysicalMemoryAllocatorIdentity{value.world_rank, value.device} == requester;
                }) != 1)
                throw std::invalid_argument("Consumer-local resource is missing or duplicated");
            const auto local = forRequester(layout, source, request, requester);
            const auto localBOM = contribution == ConsumerLocalContribution::Mandatory
                ? local.mandatoryBOMWithOrderingProjection(std::span{&*found, 1}, host, projection)
                : local.slotDependentBOMWithOrderingProjection(std::span{&*found, 1}, host, projection);
            for (const auto& row : localBOM.resources())
                for (std::size_t index = 0; index < PhysicalMemoryBOM::ownerCount(); ++index) {
                    const auto owner = static_cast<PhysicalMemoryOwner>(index);
                    if (row.resource().device.is_cpu() &&
                        (owner == PhysicalMemoryOwner::ImmutableTableStorage || owner == PhysicalMemoryOwner::ImmutableLookupStaging))
                        continue;
                    result.add(row.resource(), owner, row.bytes(owner));
                }
            const auto layoutForChannel = local.channelLayout();
            if (channel && (channel->maximumIds() != layoutForChannel.maximumIds() ||
                channel->rowBytes() != layoutForChannel.rowBytes() || channel->protocol() != layoutForChannel.protocol()))
                throw std::invalid_argument("Grouped source channels disagree on original envelope");
            channel = layoutForChannel;
            channels.push_back(requester.device);
        }
        const auto hostBOM = contribution == ConsumerLocalContribution::Mandatory
            ? shared.mandatoryBOM(host) : shared.cacheBOM(host);
        append(result, hostBOM);
        if (contribution == ConsumerLocalContribution::Mandatory)
            result.add(host, PhysicalMemoryOwner::ImmutableLookupStaging,
                qwen4exp::ImmutableLookupServiceGeometry::checked(*channel, channels).hostBytes());
        return result.build();
    }
    friend struct test::AdaptiveImmutableLookupGeometryReplayAccess;
    friend class MemoryPlanner;
    friend class ImmutableRowTablePlacementPlanner;
    /** @brief Shared production/replay implementation; the projection is sealed, not caller-authored caps. */
    PhysicalMemoryPlan mandatoryBOMWithOrderingProjection(std::span<const PhysicalMemoryResource> devices,
                                    PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection) const
    {
        requireDevices(devices);
        PhysicalMemoryPlanBuilder result;
        for (std::size_t i = 0; i < physical_.owners().size(); ++i)
        {
            const auto gpu = resource(devices, physical_.owners()[i].allocator);
            append(result, physical_.mandatoryBOM(static_cast<std::uint32_t>(i), gpu));
            result.add(gpu, PhysicalMemoryOwner::ImmutableLookupStaging, actual_.owners[i]);
        }
        const auto coordinator = resource(devices, physical_.coordinator());
        append(result, coordinator_.mandatoryBOM(coordinator));
        for (const auto bytes : {actual_.coordinator, actual_.source.frame_bytes, actual_.source.native_bytes})
            result.add(coordinator, PhysicalMemoryOwner::ImmutableLookupStaging, bytes);
        append(result, orderingBOM(devices,host,projection));
        append(result, host_.mandatoryBOM(host));
        if (compact_) append(result, compact_->completeBOM(host,devices));
        else if (readers_) append(result,readers_->completeBOM(host,devices));
        append(result,localHotMandatoryBOM(devices,host,projection));
        result.add(host, PhysicalMemoryOwner::ImmutableLookupStaging,
                   qwen4exp::ImmutableLookupService::hostBytes(channelLayout()));
        return result.build();
    }
    /** @brief Shared production/replay implementation; the projection is sealed, not caller-authored caps. */
    PhysicalMemoryPlan slotDependentBOMWithOrderingProjection(std::span<const PhysicalMemoryResource> devices,
                                        PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection) const
    {
        PhysicalMemoryPlanBuilder result(cacheBOM(devices, host));
        for (const auto& device : devices)
            result.add(device, PhysicalMemoryOwner::ImmutableLookupStaging,
                nativeGrowth({device.world_rank, device.device}));
        append(result, orderingGrowthBOM(devices,host,projection));
        append(result,localHotSlotDependentBOM(devices,host,projection));
        return result.build();
    }
    /**
     * @brief Price the existing closed event programs from actual nonempty roots.
     * @param devices Original allocator observations, already cohort-validated.
     * @param host Original CPU observation for the same PMA.
     * @return One owner group per coordinate, one coordinator/source group and original source boundary.
     * @note Legacy CompleteCohort remote boundary groups belong to that separate composition.
     */
    PhysicalMemoryPlan orderingBOM(std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection)const
    {
        PhysicalMemoryPlanBuilder result;
        for(std::size_t i=0;i<physical_.owners().size();++i)
            append(result,projection.bom(
                qwen4exp::PooledImmutableOwnerNativePlan::orderingProgram(physical_,coordinator_,static_cast<std::uint32_t>(i)),
                host,resource(devices,physical_.owners()[i].allocator)));
        const auto coordinator=resource(devices,physical_.coordinator());
        append(result,projection.bom(
            qwen4exp::PooledImmutableCoordinatorNativePlan::orderingProgram(physical_,coordinator_),host,coordinator));
        const auto source=NativeOrderingEventProgram::pooledSource();
        source.requireRootBytes({0,actual_.source.frame_bytes,actual_.source.native_bytes});
        append(result,projection.bom(source,host,coordinator));
        append(result,projection.bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,coordinator));
        return result.build();
    }
    /**
     * @brief Price optional publication-event growth through the same closed certificate authority.
     * @param devices Original complete owner/coordinator observations.
     * @param host Original CPU observation.
     * @return Exactly the event-group growth above this cohort's C0 floor, before quota admission.
     */
    PhysicalMemoryPlan orderingGrowthBOM(std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryResource host,const NativeOrderingEventBudgetProjection& projection)const
    {
        PhysicalMemoryPlanBuilder result;
        for(std::size_t i=0;i<physical_.owners().size();++i)
            append(result,projection.growthBOM(NativeOrderingEventProgram::pooledOwner(false),
                qwen4exp::PooledImmutableOwnerNativePlan::orderingProgram(physical_,coordinator_,static_cast<std::uint32_t>(i)),
                host,resource(devices,physical_.owners()[i].allocator)));
        append(result,projection.growthBOM(NativeOrderingEventProgram::pooledCoordinator(false),
            qwen4exp::PooledImmutableCoordinatorNativePlan::orderingProgram(physical_,coordinator_),
            host,resource(devices,physical_.coordinator())));
        return result.build();
    }
    /** @brief Original physical root sizes, never allocations or live balances. */
    struct NativeRoots
    {
        std::vector<std::size_t> owners;
        std::size_t coordinator;
        qwen4exp::PooledSourceOffsets source;
        /** @return Exact native root extents from their original identity-free factories. */
        static NativeRoots from(const qwen4exp::PooledImmutablePhysicalOwnerGeometry& physical,
            const qwen4exp::PooledImmutableCoordinatorGeometry& coordinator,
            const ImmutableRowTableSource& source)
        {
            NativeRoots roots{{}, qwen4exp::PooledImmutableCoordinatorNativePlan::rootOffsets(
                physical, coordinator).bytes,
                qwen4exp::PooledImmutableSourceGeometry::rootOffsets(physical, coordinator, source)};
            for (std::size_t i = 0; i < physical.owners().size(); ++i)
                roots.owners.push_back(qwen4exp::PooledImmutableOwnerNativePlan::rootOffsets(
                    physical, coordinator, static_cast<std::uint32_t>(i)).bytes);
            return roots;
        }
    };
    /** @brief Retain original declarations without native operations or reservations. */
    AdaptiveImmutableLookupGeometry(qwen4exp::PooledImmutablePhysicalOwnerGeometry physical,
        qwen4exp::PooledImmutableCoordinatorGeometry coordinator, CpuAdaptiveImmutableRowCacheGeometry host,
        NativeRoots actual, NativeRoots floor,
        std::optional<qwen4exp::PooledImmutableReplicatedReaderGeometry> readers={},
        std::vector<qwen4exp::PooledImmutableLocalHotGeometry> local_hot={},
        std::optional<qwen4exp::PooledImmutableLocalHotTransportGeometry> compact={})
        : physical_(std::move(physical)), coordinator_(std::move(coordinator)), host_(std::move(host)),
          actual_(std::move(actual)), floor_(std::move(floor)),readers_(std::move(readers)),local_hot_(std::move(local_hot)),compact_(std::move(compact)) {}
    /** @return Checked extent sum, never available-memory arithmetic. */
    static std::size_t plus(std::size_t a, std::size_t b)
    {
        if (b > std::numeric_limits<std::size_t>::max() - a)
            throw std::overflow_error("Adaptive physical root extent overflows");
        return a + b;
    }
    /** @return Exact optional portion of one declared physical root. */
    static std::size_t growth(std::size_t actual, std::size_t floor)
    {
        if (actual < floor)
            throw std::logic_error("Adaptive physical root is smaller than its original C0 declaration");
        return actual - floor;
    }
    /** @brief Append original typed BOM lines without another capacity ledger. */
    static void append(PhysicalMemoryPlanBuilder& result, const PhysicalMemoryPlan& original)
    { for (const auto& line : original.resources()) result.add(line); }
    /** @brief Reject an allocator outside the original owners, coordinator and declared reader cohort. */
    void requireAllocator(PhysicalMemoryAllocatorIdentity allocator) const
    {
        if (allocator != physical_.coordinator() && std::none_of(physical_.owners().begin(), physical_.owners().end(),
            [&](const auto& owner) { return owner.allocator == allocator; }) &&
            (!readers_ || std::find(readers_->consumers().begin(),readers_->consumers().end(),allocator)==readers_->consumers().end()))
            throw std::invalid_argument("Adaptive root projection names a foreign GPU allocator");
    }
    /** @return Original one-device owner; reject ambiguous projections for larger cohorts. */
    PhysicalMemoryAllocatorIdentity singleAllocator() const
    {
        if (readers_ || physical_.owners().size() != 1 || physical_.owners().front().allocator != physical_.coordinator())
            throw std::invalid_argument("Pooled adaptive root bytes require an explicit physical allocator");
        return physical_.coordinator();
    }
    /** @return Existing validated observation for exactly one original allocator. */
    static PhysicalMemoryResource resource(std::span<const PhysicalMemoryResource> devices,
                                            PhysicalMemoryAllocatorIdentity allocator)
    {
        const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& value) {
            return PhysicalMemoryAllocatorIdentity{value.world_rank, value.device} == allocator;
        });
        if (found == devices.end()) throw std::invalid_argument("Adaptive BOM is missing an original GPU observation");
        return *found;
    }
    /** @brief Require the exact observation set, independent of presentation order. */
    void requireDevices(std::span<const PhysicalMemoryResource> devices) const
    {
        std::vector<PhysicalMemoryAllocatorIdentity> expected{physical_.coordinator()};
        for(const auto& owner:physical_.owners())if(std::find(expected.begin(),expected.end(),owner.allocator)==expected.end())expected.push_back(owner.allocator);
        if(readers_)for(const auto reader:readers_->consumers())if(std::find(expected.begin(),expected.end(),reader)==expected.end())expected.push_back(reader);
        if (devices.size() != expected.size())
            throw std::invalid_argument("Adaptive BOM requires every original GPU owner, coordinator and reader");
        for (std::size_t i = 0; i < devices.size(); ++i)
        {
            const PhysicalMemoryAllocatorIdentity allocator{devices[i].world_rank, devices[i].device};
            if (!devices[i].valid() || !devices[i].device.is_gpu())
                throw std::invalid_argument("Adaptive BOM contains an invalid GPU observation");
            requireAllocator(allocator);
            for (std::size_t prior = 0; prior < i; ++prior)
                if (PhysicalMemoryAllocatorIdentity{devices[prior].world_rank, devices[prior].device} == allocator)
                    throw std::invalid_argument("Adaptive BOM repeats a GPU allocator observation");
        }
    }
    /** @return Native growth owned here; shared source growth appears only once. */
    std::size_t nativeGrowth(PhysicalMemoryAllocatorIdentity allocator) const
    {
        requireAllocator(allocator);
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < physical_.owners().size(); ++i)
            if (physical_.owners()[i].allocator == allocator)
                bytes = plus(bytes, growth(actual_.owners[i], floor_.owners[i]));
        if (allocator == physical_.coordinator())
        {
            bytes = plus(bytes, growth(actual_.coordinator, floor_.coordinator));
            bytes = plus(bytes, growth(actual_.source.frame_bytes, floor_.source.frame_bytes));
        }
        return bytes;
    }

    qwen4exp::PooledImmutablePhysicalOwnerGeometry physical_;
    qwen4exp::PooledImmutableCoordinatorGeometry coordinator_;
    CpuAdaptiveImmutableRowCacheGeometry host_;
    NativeRoots actual_, floor_;
    std::optional<qwen4exp::PooledImmutableReplicatedReaderGeometry> readers_;
    std::vector<qwen4exp::PooledImmutableLocalHotGeometry> local_hot_;
    std::optional<qwen4exp::PooledImmutableLocalHotTransportGeometry> compact_;
};
}
