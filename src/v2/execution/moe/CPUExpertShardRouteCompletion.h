/**
 * @file CPUExpertShardRouteCompletion.h
 * @brief Closed CPU-owned witness of one actual original shard collective.
 *
 * The existing channel signal remains the only publication/reuse authority.
 * This stack-local value only proves that the exact source member completed
 * its real rooted gather before a joint return is allowed to be published.
 */
#pragma once
#include "MoEOverlayActivationEpochABI.h"
#include "MoEOverlayParticipantResidency.h"
#include <memory>
namespace llaminar2 {
class CPUExpertShardSourceSelection;
class IGlobalTPContext;
class MoECPUExpertShardRouteStage;
/** @brief Borrowed original publication identity plus retained exact native domain lifetime. */
class CPUExpertShardRouteCompletion final {
public:
    /**
     * @brief Authenticate this completed collective against the still-active original channel.
     * @param control Exact channel object whose CPU endpoint supplied the original packet.
     * @param identity Complete immutable original invocation, including return role.
     * @param stage_ordinal Original model-layer operation ordinal.
     * @return False for foreign/reused/mismatched/aborted source or context.
     */
    bool matches(const MoEOverlayActivationEpochControl &control,
                 const MoEOverlayActivationEpochIdentity &identity,
                 std::uint32_t stage_ordinal) const noexcept;
private:
    friend class MoECPUExpertShardRouteStage;
    /** @brief Only the real source-bound collective may attest completed work. */
    CPUExpertShardRouteCompletion(const MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEpochIdentity identity,std::uint32_t stage,
        std::shared_ptr<const CPUExpertShardSourceSelection> source,
        std::shared_ptr<IGlobalTPContext> context,int root,
        std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::CPUJointLayer> prepared)
        :control_(control),identity_(identity),stage_(stage),source_(std::move(source)),
         context_(std::move(context)),root_(root),prepared_(std::move(prepared)){}
    const MoEOverlayActivationEpochControl *control_;
    MoEOverlayActivationEpochIdentity identity_;
    std::uint32_t stage_;
    std::shared_ptr<const CPUExpertShardSourceSelection> source_;
    std::shared_ptr<IGlobalTPContext> context_;
    int root_;
    std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::CPUJointLayer> prepared_;
};
} // namespace llaminar2
