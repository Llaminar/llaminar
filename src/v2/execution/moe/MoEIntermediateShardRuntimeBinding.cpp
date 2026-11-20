/**
 * @file MoEIntermediateShardRuntimeBinding.cpp
 * @brief Authenticate original fragment descriptors at the cold runtime publication boundary.
 *
 * Local/global participant coordinates remain distinct. Cohort route targets
 * come from the original closed owner map; all three physical projections come
 * from the original prepared fragment. Child graphs borrow the parent's durable
 * epoch while retaining private route scratch. The runtime's one-way initial
 * publication state prevents a later graph declaration from rewinding movement.
 */
#include "MoEIntermediateShardRuntimeBinding.h"
#include "MoEIntermediateShardDescriptorDeclaration.h"
#include "MoERuntimeTable.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEExpertOverlayPreparationPlan.h"
#include "MoEExpertOwnerMap.h"
#include <algorithm>
#include "DeviceMoEExpertDescriptorBuilder.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
namespace llaminar2
{
MoEIntermediateShardDescriptorDeclaration MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(
    std::shared_ptr<const MoEOverlayFragmentFamily> family)
{
    if(!family)throw std::invalid_argument("Fragment descriptor declaration requires its original family");
    MoEIntermediateShardDescriptorDeclaration result;
    result.original_=DeviceMoEIntermediateShardBinding::fromOriginalFamily(*family);
    const auto& cohort=*family->cohort();
    const auto member=family->participantIndex();
    result.device_=cohort.participants()[member].address.toLocalDeviceId();
    result.experts_=cohort.ownership(member).geometry().experts;
    for(const auto& projection:cohort.source().projections)
        result.sources_[static_cast<std::size_t>(projection.projection)]=projection.format;
    const bool native=result.sources_[0].isNativeQuantised();
    // Every transition retains the original routed roster. The original
    // accelerator-to-accelerator movement copies physical planes and cannot
    // manufacture a CPU-normalized representation. CPU participants may
    // produce that representation even when initially empty.
    const auto preparation=family->preparation();
    const auto owners=preparation?preparation->jointOwnerMap():nullptr;
    const bool canonical_gpu_only=owners&&!owners->participants().empty()&&
        std::all_of(owners->participants().begin(),owners->participants().end(),
            [](const auto& participant){return participant.device.is_gpu();});
    for(std::size_t role=0;role<result.sources_.size();++role) {
        const auto& format=result.sources_[role];
        if(!format.valid() || native!=format.isNativeQuantised())
            throw std::invalid_argument("Fragment descriptor declaration has mixed original arithmetic families");
        if(native) {
            const auto* source=native_quantised_formats::forSourceIdentity(
                format.native_quantised.codebook_id,format.native_quantised.is_superblock);
            auto& envelope=role==2?result.down_:result.gate_up_;
            if(!source || !addNativeQuantisedExecutionFormat(envelope,
                canonicalDeviceVnniCodebookId(*source),format.native_quantised,!canonical_gpu_only))
                throw std::invalid_argument("Fragment descriptor declaration has no canonical native execution envelope");
        } else if(format!=result.sources_[0])
            throw std::invalid_argument("Fragment descriptor declaration requires one original floating precision");
    }
    if(native) {
        if(!areNativeQuantisedGateUpCodebooksCompatible(
            canonicalDeviceVnniCodebookId(result.sources_[0].native_quantised),
            canonicalDeviceVnniCodebookId(result.sources_[1].native_quantised)))
            throw std::invalid_argument("Fragment descriptor declaration has incompatible original gate/up policies");
    } else {
        // The closed fragment source/arrival contract retains exact IEEE
        // source precision; no conversion policy is inferred from a resident.
        switch(result.sources_[0].kind) {
        case ExpertWeightFormatKind::FP16:result.format_=DeviceMoEWeightFormat::FP16;break;
        case ExpertWeightFormatKind::BF16:result.format_=DeviceMoEWeightFormat::BF16;break;
        case ExpertWeightFormatKind::FP32:result.format_=DeviceMoEWeightFormat::FP32;break;
        default:throw std::invalid_argument("Fragment descriptor declaration has unsupported floating source precision");
        }
    }
    result.family_=std::move(family);
    return result;
}

void publishInitialMoEIntermediateShardRuntimeLayer(DeviceMoERuntimeTable &runtime,
    const MoEExpertOwnerMap &original, const MoEOverlayParticipantLayerBank &source, int layer, void *stream)
{
    const auto &family = source.fragment_family;
    if (!stream || isGraphCaptureActive() || !family || runtime.fragmentFamily(layer) != family.get() ||
        !family->preparation() || !family->cohort() ||
        family->preparation()->jointOwnerMap().get() != &original ||
        family->cohort()->source().layer_idx != layer)
        throw std::invalid_argument("Intermediate-shard baseline requires its original family/map/global layer and explicit cold stream");
    const auto declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(family);
    const auto member = family->participantIndex();
    const auto &cohort = *family->cohort();
    const auto &layout = cohort.ownership(member);
    const auto &geometry = layout.geometry();
    if (!source.valid(geometry.experts) || !source.experts.empty() ||
        !declaration.validFor(runtime.deviceId(),geometry.experts,geometry.model_columns,
            layout.projection(WeightRole::MoEExpertUp).rows))
        throw std::invalid_argument("Intermediate-shard baseline has an incomplete original expert-axis bank");
    if (auto *parent = runtime.overlayPlacementSource())
        publishInitialMoEIntermediateShardRuntimeLayer(*parent, original, source, layer, stream);
    // This is the existing immutable setup publication bit, never a live
    // placement/epoch shadow. Once published, no new graph rewinds a bank.
    if (runtime.hasInitialLayerRuntimeState(layer)) return;
    MoEPlacementUpdate update;
    update.epoch = 1; update.expert_count = geometry.experts;
    update.participant_id = member; update.participant_count = cohort.participants().size();
    update.experts.resize(geometry.experts);
    update.local_compute_mask.assign(geometry.experts, 0);
    update.replica_role.assign(geometry.experts, 0);
    update.resident_participant_mask.assign(geometry.experts, 0);
    update.overlay_route_participant.reserve(geometry.experts);
    for (int expert = 0; expert < geometry.experts; ++expert)
    {
        const auto *placement = original.placementFor(layer, expert);
        if (!placement) throw std::invalid_argument("Intermediate-shard baseline omits an original logical expert");
        update.overlay_route_participant.push_back(placement->shard_cohort
            ? MoEOverlayRouteTarget::cohort(placement->tier_idx).word()
            : MoEOverlayRouteTarget::whole(placement->owner_participant).word());
        auto &descriptor = update.experts[expert];
        descriptor.logical_expert_id = expert; descriptor.owner_participant = -1; descriptor.local_slot = -1;
        descriptor.projection_set = DeviceMoEProjectionSet::IntermediateShard;
        if (!source.resident_mask[expert]) continue;
        if (placement->shard_cohort != family->cohort() || !family->accepts(source.fragments[expert], expert) ||
            !exportDeviceMoEFragmentWeightDescriptors(source.fragments[expert], descriptor))
            throw std::invalid_argument("Intermediate-shard baseline fragment changed its original cohort/source/local-F geometry");
        descriptor.owner_participant = -1; descriptor.local_slot = expert;
        descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
        update.local_compute_mask[expert] = 1;
    }
    if (!runtime.prepareInactiveBank(layer, update) || !runtime.flipActiveBank(layer, update.epoch, stream))
        throw std::runtime_error("Intermediate-shard runtime failed to publish its original initial fragment bank");
}
}
