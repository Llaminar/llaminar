/**
 * @file MoEOptimizationPipeline.cpp
 * @brief Compose stage-local diagnostics without inventing execution authority.
 *
 * This code reads completed owner snapshots only. It performs no device I/O,
 * synchronization, movement, or placement accounting. Checked integer sums
 * report completed work; all policy receipts remain in their original scope.
 */
#include "MoEOptimizationPipeline.h"
#include "MoEExpertShardCohort.h"
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Add a completed-work counter without allowing wraparound. */
        void add(std::uint64_t &total, std::uint64_t value)
        {
            if (value > std::numeric_limits<std::uint64_t>::max() - total)
                throw std::overflow_error("Pipeline MoE completed-work counter overflow");
            total += value;
        }
    }

    MoEOptimizationStatus composeMoEOptimizationStatus(MoEOptimizationStages<MoEOptimizationStatus> stages)
    {
        if (stages.empty())
            throw std::invalid_argument("Pipeline MoE status requires every stage observation");
        MoEOptimizationStatus result{.authority = MoEOptimizationAuthority::Pipeline};
        bool failed = false, learning = false, active = false, drained = false, disabled = false;
        for (const auto &stage : stages.entries())
        {
            const auto &value = stage.value;
            if (value.authority != MoEOptimizationAuthority::None && value.authority != MoEOptimizationAuthority::Host &&
                value.authority != MoEOptimizationAuthority::Device)
                throw std::invalid_argument("Pipeline MoE stage does not name one publication authority");
            if (value.authority == MoEOptimizationAuthority::None &&
                (value.state != MoEOptimizationLifecycleState::NotApplicable || value.published_movement_waves ||
                 value.completed_movement.transactions || value.completed_movement.commands || value.completed_movement.physical_bytes ||
                 value.completed_movement.promotions || value.completed_movement.demotions || value.completed_movement.same_priority_moves ||
                 value.completed_decision_windows || value.last_decision || value.demand_window.valid()))
                throw std::invalid_argument("Pipeline MoE stage published work without an authority");
            switch (value.state)
            {
            case MoEOptimizationLifecycleState::Failed: failed = true; break;
            case MoEOptimizationLifecycleState::LearningEconomy: learning = true; break;
            case MoEOptimizationLifecycleState::Active: active = true; break;
            case MoEOptimizationLifecycleState::Drained: drained = true; break;
            case MoEOptimizationLifecycleState::MovementDisabled: disabled = true; break;
            case MoEOptimizationLifecycleState::NotApplicable: break;
            default: throw std::invalid_argument("Pipeline MoE stage has an unknown lifecycle");
            }
            add(result.published_movement_waves, value.published_movement_waves);
            add(result.completed_movement.transactions, value.completed_movement.transactions);
            add(result.completed_movement.commands, value.completed_movement.commands);
            add(result.completed_movement.physical_bytes, value.completed_movement.physical_bytes);
            add(result.completed_movement.promotions, value.completed_movement.promotions);
            add(result.completed_movement.demotions, value.completed_movement.demotions);
            add(result.completed_movement.same_priority_moves, value.completed_movement.same_priority_moves);
            add(result.completed_decision_windows, value.completed_decision_windows);
        }
        result.state = failed ? MoEOptimizationLifecycleState::Failed : learning ? MoEOptimizationLifecycleState::LearningEconomy :
            active ? MoEOptimizationLifecycleState::Active : disabled ? MoEOptimizationLifecycleState::MovementDisabled :
            drained ? MoEOptimizationLifecycleState::Drained : MoEOptimizationLifecycleState::NotApplicable;
        result.activity = failed ? MoEOptimizationActivityState::Failed :
            (learning || active) ? MoEOptimizationActivityState::StageOwned :
            (drained || disabled) ? MoEOptimizationActivityState::Dormant : MoEOptimizationActivityState::NotApplicable;
        // A stage epoch, demand window or last decision is not a model-wide
        // scalar. Keeping those fields empty forces consumers to name a stage.
        result.stages = std::move(stages);
        return result;
    }

    MoEOptimizationStatus drainedMoEOptimizationStatus(MoEOptimizationStatus status)
    {
        if (!status.stages.empty())
        {
            std::vector<MoEOptimizationStages<MoEOptimizationStatus>::Entry> stages;
            stages.reserve(status.stages.entries().size());
            for (const auto &stage : status.stages.entries())
                stages.push_back({stage.identity, drainedMoEOptimizationStatus(stage.value)});
            return composeMoEOptimizationStatus(MoEOptimizationStages<MoEOptimizationStatus>::seal(std::move(stages)));
        }
        if (status.authority != MoEOptimizationAuthority::None && !status.failed())
        {
            status.state = MoEOptimizationLifecycleState::Drained;
            status.activity = MoEOptimizationActivityState::Dormant;
        }
        return status;
    }

    MoEOptimizationMovementLedger composeMoEOptimizationMovementLedger(
        MoEOptimizationStages<MoEOptimizationMovementLedger> stages)
    {
        if (stages.empty())
            throw std::invalid_argument("Pipeline MoE movement requires every stage history");
        for (const auto &stage : stages.entries())
            for (const auto &edge : stage.value.edges)
            {
                const auto &scope = stage.identity;
                const auto &participants = scope.participants();
                const auto endpoint_belongs = [&](const MoEOptimizationMovementEndpoint& endpoint) {
                    if(endpoint.isWhole()) {
                        const auto& whole=endpoint.whole();
                        return whole.participant>=0 && static_cast<size_t>(whole.participant)<participants.size() &&
                            whole.device==participants[whole.participant].toLocalDeviceId();
                    }
                    if(!endpoint.isCPUCohort())return false;
                    for(const auto& member:endpoint.cohort().original->participants())
                        if(member.participant_id<0 || static_cast<size_t>(member.participant_id)>=participants.size() ||
                            member.address!=participants[member.participant_id])return false;
                    return true;
                };
                if (!edge.valid() || edge.layer < scope.firstLayer() || edge.layer >= scope.routedLastLayer() ||
                    !endpoint_belongs(edge.source) || !endpoint_belongs(edge.destination))
                    throw std::invalid_argument("Pipeline MoE movement contains a foreign layer or complete endpoint");
            }
        return {.stages = std::move(stages)};
    }

    MoEOptimizationMovementTopology composeMoEOptimizationMovementTopology(
        MoEOptimizationStages<MoEOptimizationMovementTopology> stages)
    {
        if (stages.empty())
            throw std::invalid_argument("Pipeline MoE topology requires every stage geometry");
        bool tier = false, participant = false;
        for (const auto &stage : stages.entries())
        {
            if (!stage.value.valid())
                throw std::invalid_argument("Pipeline MoE stage has invalid movement geometry");
            tier |= hasTierResidencyAxis(stage.value.axes);
            participant |= hasParticipantPlacementAxis(stage.value.axes);
        }
        return {.authority = MoEOptimizationAuthority::Pipeline,
            .axes = tier ? (participant ? MoEOptimizationMovementAxes::Both : MoEOptimizationMovementAxes::TierResidency)
                         : (participant ? MoEOptimizationMovementAxes::ParticipantPlacement : MoEOptimizationMovementAxes::None),
            .stages = std::move(stages)};
    }
}
