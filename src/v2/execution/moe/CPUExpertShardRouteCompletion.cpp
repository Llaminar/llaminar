/**
 * @file CPUExpertShardRouteCompletion.cpp
 * @brief Exact source/member/invocation checks before CPU joint return publication.
 */
#include "CPUExpertShardRouteCompletion.h"
#include "collective/GlobalTPContext.h"
#include "CPUExpertShardSourceSelection.h"
#include "MoEOverlayRouteTarget.h"
#include "MoEExpertOverlayPreparationPlan.h"
#include "../../collective/IGlobalTPContext.h"
namespace llaminar2 {
bool CPUExpertShardRouteCompletion::matches(const MoEOverlayActivationEpochControl &control,
    const MoEOverlayActivationEpochIdentity &identity,std::uint32_t stage) const noexcept {
    const auto* native=dynamic_cast<const GlobalTPContext*>(context_.get());
    if (!native || !prepared_ || !prepared_->sourceSelection() ||
        prepared_->sourceSelection() != source_ || !bank_ ||
        bank_->participant_id != prepared_->participantId() ||
        !bank_->containsModelLayer(prepared_->layer()) ||
        bank_->layerForModelLayer(prepared_->layer()).cpu_fragment_family != prepared_->family() ||
        bank_->layerForModelLayer(prepared_->layer())
            .cpu_fragment_family.owner_before(prepared_->family()) ||
        prepared_->family().owner_before(
            bank_->layerForModelLayer(prepared_->layer()).cpu_fragment_family) ||
        bank_->epoch != control.buffers[moeOverlayActivationBufferIndex(stage)]
                            .dispatch_descriptor.placement_epoch ||
        !native->authenticatesOriginalCPUExpertBorrow(context_) || control_ != &control ||
        identity_ != identity || stage_ != stage || !source_ || !context_ ||
        context_->isAbortRequested() ||
        identity.digest != moeOverlayActivationIdentityDigest(identity) ||
        identity.return_role != control.channel.return_role ||
        identity.cohort_target != control.channel.cohort_target ||
        identity.target_participant_id != source_->participant().participant_id ||
        identity.target_world_rank != source_->participant().world_rank ||
        source_->participantIndex() != static_cast<size_t>(context_->myIndex()) || root_ < 0 ||
        root_ >= context_->degree())
        return false;
    const auto original_map = prepared_->family()->preparation()->cpuJointOwnerMap();
    const auto *member =
        original_map ? original_map->participantForId(prepared_->participantId()) : nullptr;
    if (!member ||
        identity.cohort_target != MoEOverlayRouteTarget::cpuCohort(member->tier_idx).word())
        return false;
    const auto expected = source_->participantIndex() == static_cast<size_t>(root_)
                              ? MoEOverlayActivationReturnRole::JointRoot
                              : MoEOverlayActivationReturnRole::JointContributor;
    return identity.return_role==static_cast<std::uint32_t>(expected);
}
} // namespace llaminar2
