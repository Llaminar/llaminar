/**
 * @file GraphExpertSourceAdmission.h
 * @brief Admit only the expert sources prepared by one participant-local graph.
 *
 * Sibling GPU factories prepare concurrently in a shared original registry.
 * Completion of one graph's weights grants no authority to inspect or publish
 * an unfinished sibling bank. The same typed endpoint scope used for loading
 * and preparation selects this cold publication; the existing registry owns
 * source identity, completeness, synchronization and retained bank lifetime.
 */
#pragma once

#include "execution/moe/MoEExpertOverlayGraphScope.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"

namespace llaminar2
{
    /**
     * @brief Publish this graph's complete original joint sources before graph declaration.
     * @param residency Existing original participant bank authority.
     * @param context Original model context that prepared this graph's weights.
     * @param runtime Original runtime used to resolve the graph's source scope.
     * @param scope Exact domain/device/rank/member selection shared with preparation.
     * @param error Precise source or completeness failure for a selected participant.
     * @return True when every selected joint bank is admitted; siblings are untouched.
     * @throws std::invalid_argument For a scope resolved from another runtime.
     */
    inline bool admitGraphLocalJointSources(MoEOverlayParticipantResidencyRegistry &residency,
        const std::shared_ptr<ModelContext> &context, const MoEExpertOverlayRuntimePlan &runtime,
        const MoEExpertOverlayGraphScope &scope, std::string *error = nullptr)
    {
        scope.validate(runtime, scope.device());
        if (error) error->clear();
        std::vector<int> selected;
        for (const int id : residency.localParticipantIds())
        {
            const auto *member = residency.initialOwnerMap().participantForId(id);
            if (member && residency.isJointParticipant(id) &&
                scope.contains(member->domain_name, member->device,
                    member->world_rank, member->domain_participant_index))
                selected.push_back(id);
        }
        if (selected.empty()) return true;
        if (!residency.bindOriginalPreparedContext(context, residency.projectionPreparation(), error))
            return false;
        const auto source = residency.acquireOriginalPreparedSource(error);
        if (!source) return false;
        for (const int id : selected)
            if (!residency.registerInitialJointBank(source->registry(), residency.originalModel(), id, error))
                return false;
        return true;
    }
}
