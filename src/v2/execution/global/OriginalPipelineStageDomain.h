/**
 * @file OriginalPipelineStageDomain.h
 * @brief Borrow the original declared domain of a global pipeline action.
 *
 * Cross-rank actions carry one local device and deliberately have no local TP
 * device vector. Their full participant declaration belongs to the original
 * stage specification. This metadata view preserves that distinction without
 * discovering devices, constructing contexts or changing rank namespaces.
 */
#pragma once
#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/global_pp/GlobalPPRankPlan.h"
#include <span>
#include <stdexcept>
namespace llaminar2 {
/** @brief Borrowed exact domain declaration, valid only while its original plan lives. */
struct OriginalPipelineStageDomain {
    std::span<const GlobalDeviceAddress> participants;
    std::span<const float> weights;
    CollectiveBackendType backend;
    /**
     * @brief Authenticate the action against its original compiler-owned stage.
     * @param stage Original stage declaration, including every global participant.
     * @param action This actual rank's executable action; never a synthesized peer.
     * @return Exact semantic participant/weight order for binding and admission.
     * @throws std::invalid_argument for foreign stage, rank, device or geometry.
     */
    static OriginalPipelineStageDomain borrow(const GlobalPPStageSpec& stage,const RankStageAction& action) {
        if(stage.stage_id!=action.stage_id || stage.domain_name!=action.domain_name ||
            stage.is_global_tp!=action.is_global_tp || action.role!=RankStageAction::Role::EXECUTE ||
            stage.backend!=action.backend)
            throw std::invalid_argument("Pipeline action changed its original stage domain authority");
        if(!action.is_global_tp) return {action.devices,action.tp_weights,action.backend};
        if(stage.per_rank_devices.size()!=stage.participating_ranks.size() || stage.per_rank_devices.size()<2 ||
            action.tp_domain_size!=static_cast<int>(stage.per_rank_devices.size()) ||
            action.tp_rank_in_domain<0 || action.tp_rank_in_domain>=action.tp_domain_size ||
            action.device!=stage.per_rank_devices[action.tp_rank_in_domain] ||
            (!stage.tp_weights.empty() && stage.tp_weights.size()!=stage.per_rank_devices.size()))
            throw std::invalid_argument("Global pipeline binding requires every original MPI-domain endpoint");
        return {stage.per_rank_devices,stage.tp_weights,stage.backend};
    }
};
}
