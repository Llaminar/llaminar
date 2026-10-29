/**
 * @file MoEExpertOverlayGraphScope.h
 * @brief Source-bound logical expert participants owned by one graph's setup.
 *
 * CPU storage identity does not encode a NUMA endpoint. This immutable view
 * resolves the exact declared address before materialization and preparation;
 * both consumers use the same domain/rank/index tuples. It owns no allocations,
 * mutable expert state, capacity, or alternate placement authority.
 */
#pragma once
#include "MoEExpertOverlayRuntimePlan.h"
#include "collective/ILocalTPContext.h"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief Explicit ownership of colocated CPU sparse endpoints by a GPU root graph. */
enum class MoEOverlayColocatedCPUOwnership { None, CurrentRank };

/** @brief Exact immutable preparation endpoint, independent of shared storage identity. */
struct MoEExpertOverlayGraphParticipant
{
    std::string domain;
    DeviceId device;
    int world_rank;
    int participant_index;
    bool operator==(const MoEExpertOverlayGraphParticipant &) const = default;
};

/** @brief Checked participant view of one retained runtime source, never a live registry. */
class MoEExpertOverlayGraphScope
{
public:
    /**
     * @brief Resolve graph ownership from an exact LocalTP address or unique device membership.
     * @param runtime Immutable original domain/member declarations.
     * @param device Physical graph execution/storage address space.
     * @param endpoint Exact LocalTP endpoint when hardware identity is shared.
     * @param cpu_ownership Explicit GPU continuation ownership of colocated CPU endpoints.
     * @return Source-bound complete participant selection, including empty expert ownership.
     * @throws std::invalid_argument For foreign, ambiguous, duplicated or incompatible identity.
     */
    static MoEExpertOverlayGraphScope resolve(
        const MoEExpertOverlayRuntimePlan &runtime, DeviceId device,
        std::optional<GlobalDeviceAddress> endpoint = std::nullopt,
        MoEOverlayColocatedCPUOwnership cpu_ownership = MoEOverlayColocatedCPUOwnership::None)
    {
        if (cpu_ownership != MoEOverlayColocatedCPUOwnership::None &&
            cpu_ownership != MoEOverlayColocatedCPUOwnership::CurrentRank)
            throw std::invalid_argument("ExpertOverlay graph scope has unknown colocated CPU ownership");
        if (!device.is_valid() || (endpoint && endpoint->toLocalDeviceId() != device))
            throw std::invalid_argument("ExpertOverlay graph scope has incompatible endpoint/storage identity");
        if (cpu_ownership == MoEOverlayColocatedCPUOwnership::CurrentRank &&
            (!device.is_gpu() || device != runtime.continuationDevice()))
            throw std::invalid_argument("ExpertOverlay colocated CPU ownership requires the GPU continuation root");
        MoEExpertOverlayGraphScope scope(runtime, device);
        bool endpoint_found = false;
        for (const auto &domain : runtime.domains())
        {
            size_t same_device_members = 0;
            for (const auto &member : domain.participants)
            {
                // The owner map enumerates every declared routed-tier member,
                // including tiers with no current experts. Preserve that exact
                // rank-proof boundary without rejecting unrelated unused domains.
                const bool routed_owner = domain.routed_tier_count > 0;
                const bool selected_endpoint = endpoint ? member.address == *endpoint : member.local_device == device;
                const bool selected_cpu = cpu_ownership == MoEOverlayColocatedCPUOwnership::CurrentRank &&
                    member.local_device.is_cpu();
                if (!member.world_rank_known && (routed_owner || selected_endpoint || selected_cpu))
                    throw std::invalid_argument("ExpertOverlay graph scope has an owner or selected endpoint without an owning MPI rank");
                if (member.world_rank_known && member.world_rank == runtime.currentWorldRank() &&
                    member.local_device == device)
                    ++same_device_members;
            }
            if (!endpoint && same_device_members > 1)
                throw std::invalid_argument("ExpertOverlay graph scope requires exact logical participant address");
            for (const auto &member : domain.participants)
            {
                if (!member.world_rank_known || member.world_rank != runtime.currentWorldRank())
                    continue;
                const bool own_endpoint = member.local_device == device &&
                    (!endpoint || member.address == *endpoint);
                endpoint_found |= own_endpoint;
                const bool colocated_cpu = cpu_ownership == MoEOverlayColocatedCPUOwnership::CurrentRank &&
                    member.local_device.is_cpu();
                if (!own_endpoint && !colocated_cpu)
                    continue;
                MoEExpertOverlayGraphParticipant selected{
                    domain.name, member.local_device, member.world_rank, member.participant_index};
                if (std::find(scope.participants_.begin(), scope.participants_.end(), selected) != scope.participants_.end())
                    throw std::invalid_argument("ExpertOverlay graph scope duplicates a logical participant");
                scope.participants_.push_back(std::move(selected));
            }
        }
        if (!endpoint_found)
            throw std::invalid_argument("ExpertOverlay graph scope endpoint is absent from its runtime source");
        return scope;
    }

    /**
     * @brief Resolve the exact logical endpoint already owned by a TP context.
     * @param runtime Original retained overlay source.
     * @param device Actual graph device, never a fabricated CPU ordinal.
     * @param context Existing local/global TP authority, or null for a single participant.
     * @param participant Existing index in that TP context's declaration.
     * @param cpu_ownership Explicit colocated CPU ownership by the continuation root.
     * @return The same source-bound scope used by materialization and graph lookup.
     * @throws std::invalid_argument For stale TP indices or absent/ambiguous endpoints.
     */
    static MoEExpertOverlayGraphScope resolveForTPParticipant(
        const MoEExpertOverlayRuntimePlan &runtime, DeviceId device, const ITPContext *context,
        int participant, MoEOverlayColocatedCPUOwnership cpu_ownership = MoEOverlayColocatedCPUOwnership::None)
    {
        std::optional<GlobalDeviceAddress> endpoint;
        if (context && context->isLocal())
        {
            const auto &local = *static_cast<const ILocalTPContext *>(context);
            if (participant < 0 || static_cast<size_t>(participant) >= local.devices().size())
                throw std::invalid_argument("ExpertOverlay graph scope has a stale LocalTP participant index");
            endpoint = local.devices()[participant];
        }
        return resolve(runtime, device, std::move(endpoint), cpu_ownership);
    }

    /**
     * @brief Reject use against another source or graph device before filtering any requests.
     * @param runtime Exact retained runtime from which this view was resolved.
     * @param device Actual calling graph endpoint.
     * @throws std::invalid_argument For a foreign runtime or graph endpoint.
     */
    void validate(const MoEExpertOverlayRuntimePlan &runtime, DeviceId device) const
    {
        if (source_ != &runtime || device_ != device)
            throw std::invalid_argument("ExpertOverlay graph scope belongs to another source or graph device");
    }
    /** @return Physical graph device retained by the checked setup view. */
    DeviceId device() const noexcept { return device_; }
    /** @return Complete immutable domain/device/rank/index selection. */
    const std::vector<MoEExpertOverlayGraphParticipant> &participants() const noexcept { return participants_; }
    /**
     * @brief Match complete logical identity without ordinal or insertion-order inference.
     * @param domain Original named expert domain.
     * @param device Exact request storage endpoint.
     * @param rank Owning MPI rank authenticated by the runtime source.
     * @param participant Original participant index in that domain.
     * @return Whether this graph owns that exact immutable request identity.
     */
    bool contains(const std::string &domain, DeviceId device, int rank, int participant) const
    {
        return std::find(participants_.begin(), participants_.end(),
            MoEExpertOverlayGraphParticipant{domain, device, rank, participant}) != participants_.end();
    }
private:
    /** @brief Create an empty source-bound view only inside checked resolution. */
    MoEExpertOverlayGraphScope(const MoEExpertOverlayRuntimePlan &runtime, DeviceId device)
        : source_(&runtime), device_(device) {}
    const MoEExpertOverlayRuntimePlan *source_;
    DeviceId device_;
    std::vector<MoEExpertOverlayGraphParticipant> participants_;
};
} // namespace llaminar2
