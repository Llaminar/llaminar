/**
 * @file MoEProjectionRuntimeBinding.cpp
 * @brief Publishes the projection family's immutable initial descriptor recipe.
 *
 * This code runs during graph preparation only. Runtime-table publication owns
 * the stream edges and reset template; there is no second epoch flag, device
 * download, inference allocation or host-authoritative movement path here.
 */
#include "execution/moe/MoEProjectionRuntimeBinding.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>

namespace llaminar2
{
void publishInitialMoEProjectionRuntimeLayer(DeviceMoERuntimeTable &runtime,
    const MoEExpertOwnerMap &owners, int layer,
    const std::vector<MoEOverlayPreparedExpertPayload> &payloads, void *stream)
{
    if (!stream || isGraphCaptureActive())
        throw std::invalid_argument("Projection baseline requires an explicit stream outside capture");
    const auto *fixed = runtime.fixedDownProjectionBank(layer);
    if (!fixed || runtime.movableProjections() != DeviceMoEProjectionSet::GateUp)
        throw std::invalid_argument("Projection baseline requires its runtime-owned fixed down bank");
    const auto &layout = fixed->ownership();
    const auto &geometry = layout.geometry();
    const int participant = fixed->participantId();
    const auto *endpoint = owners.participantForId(participant);
    if (!endpoint || endpoint->device != fixed->device() || participant != layout.participant() ||
        endpoint->domain_participant_index != participant ||
        owners.participants().size() != static_cast<std::size_t>(layout.participants()) ||
        layout.participants() > static_cast<int>(kDeviceMoEMaxParticipants) ||
        payloads.size() != static_cast<std::size_t>(geometry.experts))
        throw std::invalid_argument("Projection baseline has inconsistent native participant coordinates");

    MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = geometry.experts;
    update.participant_id = participant;
    update.participant_count = layout.participants();
    update.experts.resize(payloads.size());
    update.local_compute_mask.resize(payloads.size());
    update.replica_role.resize(payloads.size());
    update.resident_participant_mask.resize(payloads.size());
    update.overlay_route_participant.resize(payloads.size());
    for (int expert = 0; expert < geometry.experts; ++expert)
    {
        const auto *owner = owners.ownerFor(layer, expert);
        if (!owner || owner->owner_participant < 0 || owner->owner_participant >= layout.participants() ||
            owner->domain_name != endpoint->domain_name || owner->domain_participant_index != owner->owner_participant)
            throw std::invalid_argument("Projection baseline owner is outside its complete native domain");
        auto &desc = update.experts[expert];
        desc.logical_expert_id = expert;
        desc.owner_participant = owner->owner_participant;
        desc.local_slot = -1;
        desc.projection_set = DeviceMoEProjectionSet::GateUp;
        update.resident_participant_mask[expert] = 1u << owner->owner_participant;
        update.overlay_route_participant[expert] = owner->owner_participant;
        if (owner->owner_participant != participant)
        {
            if (!payloads[expert].empty())
                throw std::invalid_argument("Projection baseline has a payload for a nonresident owner");
            continue;
        }
        if (!payloads[expert].readyFor(DeviceMoEProjectionSet::GateUp) ||
            !exportDeviceMoEPreparedPayload(payloads[expert], desc))
            throw std::invalid_argument("Projection baseline is missing a prepared gate/up pair");
        desc.local_slot = expert;
        desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
            DeviceMoEExpertFlags::LocalCompute | DeviceMoEExpertFlags::PreferredOwner);
        update.local_compute_mask[expert] = 1;
        update.replica_role[expert] = static_cast<std::uint8_t>(DeviceMoEReplicaRole::Primary);
    }
    if (auto *parent = runtime.overlayPlacementSource())
        publishInitialMoEProjectionRuntimeLayer(*parent, owners, layer, payloads, stream);

    // The runtime owns this one-way setup transition. Its host recipe is not
    // a live device owner map; never inspect it to undo a later movement epoch.
    if (runtime.hasInitialLayerRuntimeState(layer)) return;
    if (!runtime.prepareInactiveBank(layer, update) || !runtime.flipActiveBank(layer, update.epoch, stream))
        throw std::runtime_error("Projection runtime could not publish its initial prepared bank");
}
} // namespace llaminar2
