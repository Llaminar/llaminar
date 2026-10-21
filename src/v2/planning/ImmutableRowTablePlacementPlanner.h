/**
 * @file ImmutableRowTablePlacementPlanner.h
 * @brief Lowest-priority native table admission over one complete immutable PMA plan.
 *
 * No runtime allocation, capacity subtraction or policy retry occurs here. The
 * caller freezes higher-priority rows, weights, experts and KV before selecting
 * one concrete rank-wide mode. Adaptive host storage admits its mandatory SSD and
 * request scratch before quotas, then sizes only optional native cache slots.
 * Mode-specific arena/transport/upload contributions
 * are canonical BOMs authored by their existing owners.
 */
#pragma once
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "planning/ImmutableRowTableExecutionProvision.h"
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <utility>
namespace llaminar2
{
/** @brief Authenticated source extent and observed rank-local PLE consumer resources. */
class ImmutableRowTablePlacementInput final
{
public:
    /**
     * @brief Bind the actual continuation consumers after execution placement is frozen.
     * @param extent Original loader-authenticated native source metadata.
     * @param host Exact CPU allocator resource for the shared host candidates and upload transport.
     * @param consumers Actual PLE graph resources, excluding expert-only participants.
     * @param host_placement Explicit RAM locality; default task policy makes no exact-node claim.
     * @return Checked input with deterministic unique physical consumers.
     * @throws std::invalid_argument For empty/duplicate/foreign or invalid observations.
     */
    static ImmutableRowTablePlacementInput forConsumers(ImmutableRowTableExtent extent,
        PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> consumers,
        ImmutableRowTableHostPlacement host_placement = ImmutableRowTableHostPlacement::taskPolicy());
    /** @return Immutable locality policy; never interpreted as an allocator ordinal. */
    const ImmutableRowTableHostPlacement &hostPlacement() const noexcept { return host_placement_; }
    /** @return Original source geometry; no payload read is performed. */
    const ImmutableRowTableExtent &extent() const noexcept { return extent_; }
    /** @return Exact rank-local CPU allocator observation. */
    const PhysicalMemoryResource &host() const noexcept { return host_; }
    /** @return Deterministically ordered physical consumers. */
    std::span<const PhysicalMemoryResource> consumers() const noexcept { return consumers_; }
    /**
     * @brief Build one bounded host-adaptive candidate geometry without allocating.
     * @param requested Exact explicit adaptive request; ceilings remain request identity.
     * @param slots Candidate admitted RAM slots, bounded by request and original source.
     * @param maximum_live_ids Exact pre-quota maximum ID response envelope.
     * @return One authoritative source/host/consumer layout, without a duplicate byte ceiling.
     */
    ImmutableRowTableLayoutGeometry hostAdaptiveLayout(const ImmutableRowTablePolicy &requested,
        std::size_t slots, std::size_t maximum_live_ids) const;
    /**
     * @brief Declare one explicit GPU+subordinate host domain without observing residual capacity.
     * @param requested Lossless explicit mixed request and independent slot/physical-byte ceilings.
     * @param gpu_slots Final candidate GPU slots, including zero.
     * @param host_slots Final candidate subordinate RAM slots, including zero.
     * @param maximum_live_ids Original pre-quota ID envelope, never compact miss count.
     * @return Sole checked source/owner/consumer layout; no allocation/admission is issued.
     * @throws std::invalid_argument For multiple/non-GPU consumers or excessive slot/request geometry.
     */
    ImmutableRowTableLayoutGeometry gpuAdaptiveLayout(const ImmutableRowTablePolicy &requested,
        std::size_t gpu_slots,std::size_t host_slots,std::size_t maximum_live_ids) const;

    /** @return Whether every actual PLE consumer is a GPU. */
    bool permitsVRAM() const noexcept;
private:
    /**
     * @brief Only the checked factory binds source and observed resources.
     * @param extent Authenticated source extent.
     * @param host Canonical CPU allocation resource.
     * @param consumers Unique actual PLE execution resources.
     * @param host_placement Immutable RAM locality, separate from the allocator.
     */
    ImmutableRowTablePlacementInput(ImmutableRowTableExtent extent, PhysicalMemoryResource host,
        std::vector<PhysicalMemoryResource> consumers, ImmutableRowTableHostPlacement host_placement)
        : host_placement_(host_placement), extent_(extent), host_(host), consumers_(std::move(consumers)) {}
    ImmutableRowTableHostPlacement host_placement_;
    ImmutableRowTableExtent extent_;
    PhysicalMemoryResource host_;
    std::vector<PhysicalMemoryResource> consumers_;
};
/**
 * @brief Exact incremental mode costs, excluding table storage itself.
 *
 * Each plan contains only source-owned arena, mapped lookup, terminal executable
 * and bounded upload staging contributions. Contributions must not contain model,
 * expert or KV charges or retained credits. VRAM contains no hosted lookup channel.
 */
struct ImmutableRowTableModeContributions
{
    std::optional<PhysicalMemoryPlan> vram; ///< Present only after original upload-slot eligibility; eligible empty is zero-extra cost.
    PhysicalMemoryPlan ram;
    PhysicalMemoryPlan mmap;
    /** @return The exact immutable contribution for one concrete candidate. */
    const PhysicalMemoryPlan &forStorage(ImmutableRowStorage storage) const;
};
/** @brief One rank's setup-only final extension after rows and quotas are frozen. */
struct ImmutableRowTableAdmissionRequest
{
    ImmutableRowTablePolicy requested;
    ImmutableRowTablePlacementInput input;
    ImmutableRowTableModeContributions contributions;
    /** Source-authored execution already present in the pre-quota base; null only for direct incremental selector inputs. */
    std::shared_ptr<const ImmutableRowTableExecutionProvision> execution_provision;
};
/** @brief One selected concrete placement and its complete PMA candidate. */
struct ImmutableRowTablePlacementSelection
{
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement;
    PhysicalMemoryPlan physical_plan;
};
/** @brief Canonical setup-only table selector; runtime can only consume its sealed result. */
class ImmutableRowTablePlacementPlanner final
{
public:
    /**
     * @brief Finalize a production request whose mandatory execution was priced before quotas.
     * @param request Complete source-authored request and immutable provision.
     * @param higher_priority Frozen full row/quota plan; whole modes exclude the table, adaptive includes SSD/request execution and excludes optional cache slots.
     * @return Exact selected placement and the sole complete PMA candidate.
     * @throws std::invalid_argument For missing/stale provision or duplicate execution charges.
     */
    static ImmutableRowTablePlacementSelection select(const ImmutableRowTableAdmissionRequest &request,
        const PhysicalMemoryPlan &higher_priority);
    /**
     * @brief Select after all higher-priority placement and geometry decisions are frozen.
     * @param requested Lossless user preference and bounded mmap geometry.
     * @param input Exact original source, host resource and actual PLE consumers.
     * @param higher_priority Complete fitting plan without this rank's table/mode costs.
     * @param contributions Source-owned exact incremental mode BOMs.
     * @return First fitting VRAM/RAM/mmap candidate for Automatic, or the explicit candidate.
     * @throws std::invalid_argument For malformed/ineligible inputs or mixed accounting.
     * @throws PhysicalMemoryCapacityExhausted If every valid selected candidate exceeds its complete PMA capacity.
     * @throws std::overflow_error If canonical owner arithmetic cannot be represented.
     */
    static ImmutableRowTablePlacementSelection select(const ImmutableRowTablePolicy &requested,
        const ImmutableRowTablePlacementInput &input, const PhysicalMemoryPlan &higher_priority,
        const ImmutableRowTableModeContributions &contributions);
private:
    /** @brief Shared selector; only a validated typed provision permits prepriced lookup staging. */
    static ImmutableRowTablePlacementSelection selectImpl(const ImmutableRowTablePolicy &requested,
        const ImmutableRowTablePlacementInput &input,const PhysicalMemoryPlan &higher_priority,
        const ImmutableRowTableModeContributions &contributions,bool execution_prepriced);
};
}
