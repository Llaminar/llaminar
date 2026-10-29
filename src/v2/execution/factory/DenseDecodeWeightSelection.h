/**
 * @file DenseDecodeWeightSelection.h
 * @brief Production selection of frozen dense decode replica requirements.
 *
 * A dense replica retains attention, router, shared experts and predictor
 * operands. Routed predictor parents remain owned by the exact authority
 * installed by the graph: a plain sidecar owns them, while ExpertOverlay
 * supplies prepared expert triplets without a second raw parent allocation.
 */
#pragma once

#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/WeightIdentity.h"

#include <string>
#include <unordered_set>

namespace llaminar2
{
    /**
     * @brief Select one source requirement for the actual dense decode replica.
     * @param weight_name Canonical validated source name.
     * @param mtp_source_layers Authenticated learned predictor block indices.
     * @param routed_authority Frozen physical owner of MTP routed parents.
     * @return Whether the participant must materialize this source requirement.
     * @throws std::invalid_argument For an unknown physical authority value.
     */
    [[nodiscard]] inline bool includeWeightInDenseDecodeReplicatedPlan(
        const std::string &weight_name,
        const std::unordered_set<int> &mtp_source_layers,
        MTPRoutedExpertWeightAuthority routed_authority)
    {
        const bool owns_routed_parents =
            mtpParticipantReplicaOwnsRoutedParents(routed_authority);
        if (!isRoutedExpertRole(inferWeightRole(weight_name)))
            return true;

        const int layer = inferWeightLayer(weight_name);
        const bool predictor_source =
            (layer >= 0 && mtp_source_layers.contains(layer)) ||
            weight_name.find(".nextn.") != std::string::npos ||
            weight_name.rfind("mtp.", 0) == 0;
        return predictor_source && owns_routed_parents;
    }
}
