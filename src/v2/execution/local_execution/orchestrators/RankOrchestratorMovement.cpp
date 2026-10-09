/**
 * @file RankOrchestratorMovement.cpp
 * @brief Publish the elected TP root or independent, scoped PP movement owners.
 *
 * Native TP followers do not duplicate their elected controller's evidence.
 * PP children own disjoint layer intervals and must all survive publication;
 * their transaction and epoch counters are independent even on one host.
 * Reading these completed snapshots never launches work or reads device state.
 */
#include "RankOrchestrator.h"
#include "execution/moe/MoEOptimizationPipeline.h"
#include "execution/moe/MoEOverlayPipelineStageBinding.h"

namespace llaminar2
{
    namespace
    {
        /** @brief One observation per TP child, so root selection cannot read stale detail twice. */
        std::pair<const IInferenceRunner *, MoEOptimizationStatus> observeTP(
            const std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            std::pair<const IInferenceRunner *, MoEOptimizationStatus> result;
            for (const auto &runner : runners)
            {
                if (!runner)
                    throw std::logic_error("TP movement publication lost a participant runner");
                auto value = runner->moeOptimizationStatus();
                if (value.authority == MoEOptimizationAuthority::None)
                    continue;
                if (result.first || !value.stages.empty())
                    throw std::logic_error("TP MoE movement evidence has multiple publication authorities");
                result = {runner.get(), std::move(value)};
            }
            return result;
        }

        /** @return Exact admitted global scope, including terminal-only routed MTP rows. */
        MoEOptimizationStageIdentity stageIdentity(std::size_t index, const RankOrchestrator::PPStageConfig &config)
        {
            const int routed_end = config.moe_runtime
                ? config.moe_runtime->metadata().first_model_layer + config.moe_runtime->metadata().num_layers
                : config.last_layer;
            return {index, config.first_layer, config.last_layer, routed_end, config.stage_devices, config.has_lm_head};
        }
    }

    const IInferenceRunner *RankOrchestrator::moeOptimizationOwner() const
    {
        return observeTP(device_runners_).first;
    }

    MoEOptimizationStages<PrefixMovementEpochObservation>
    RankOrchestrator::prefixMovementStages() const
    {
        if (pp_stage_runners_.empty())
            return {};
        if (!device_runners_.empty() || pp_stage_runners_.size() != config_.pp_stages.size() ||
            last_pp_prefix_hits_.size() != pp_stage_runners_.size())
            throw std::logic_error("Pipeline prefix movement observation lost its complete admission");
        std::vector<MoEOptimizationStages<PrefixMovementEpochObservation>::Entry> stages;
        stages.reserve(pp_stage_runners_.size());
        for (std::size_t i = 0; i < pp_stage_runners_.size(); ++i)
        {
            const auto &runner = pp_stage_runners_[i];
            if (!runner)
                throw std::logic_error("Pipeline prefix movement observation lost its stage runner");
            PrefixMovementEpochObservation value{
                .admission = last_pp_prefix_hits_[i].placement_epochs,
                .completion = runner->moeRuntimeMovementEpoch()};
            (void)value.crossed(); // Reject a stale completion before publishing any stage.
            stages.push_back({stageIdentity(i, config_.pp_stages[i]), std::move(value)});
        }
        return MoEOptimizationStages<PrefixMovementEpochObservation>::seal(std::move(stages));
    }

    MoEOptimizationStatus RankOrchestrator::moeOptimizationStatus() const
    {
        if (pp_stage_runners_.empty())
            return observeTP(device_runners_).second;
        if (!device_runners_.empty() || pp_stage_runners_.size() != config_.pp_stages.size())
            throw std::logic_error("PP movement publication lost its complete stage topology");
        std::vector<MoEOptimizationStages<MoEOptimizationStatus>::Entry> stages;
        stages.reserve(pp_stage_runners_.size());
        bool applicable = false;
        for (std::size_t i = 0; i < pp_stage_runners_.size(); ++i)
        {
            if (!pp_stage_runners_[i])
                throw std::logic_error("PP movement publication lost a stage runner");
            auto value = pp_stage_runners_[i]->moeOptimizationStatus();
            const auto &stage = config_.pp_stages[i];
            if (stage.moe_runtime && value.authority == MoEOptimizationAuthority::None)
            {
                if (config_.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
                    throw std::logic_error("Dynamic PP MoE stage has no native movement publication authority");
                value.authority = describeMoEOptimizationMovementTopology(stage.moe_runtime->plan().get()).authority;
                value.state = MoEOptimizationLifecycleState::MovementDisabled;
                value.activity = MoEOptimizationActivityState::Dormant;
            }
            applicable |= value.authority != MoEOptimizationAuthority::None;
            stages.push_back({stageIdentity(i, stage), std::move(value)});
        }
        return applicable ? composeMoEOptimizationStatus(MoEOptimizationStages<MoEOptimizationStatus>::seal(std::move(stages)))
                          : MoEOptimizationStatus{};
    }

    MoEOptimizationMovementLedger RankOrchestrator::moeOptimizationMovementLedger() const
    {
        if (pp_stage_runners_.empty())
        {
            const auto *owner = moeOptimizationOwner();
            return owner ? owner->moeOptimizationMovementLedger() : MoEOptimizationMovementLedger{};
        }
        const auto status = moeOptimizationStatus();
        if (status.authority == MoEOptimizationAuthority::None)
            return {};
        std::vector<MoEOptimizationStages<MoEOptimizationMovementLedger>::Entry> stages;
        stages.reserve(pp_stage_runners_.size());
        for (std::size_t i = 0; i < pp_stage_runners_.size(); ++i)
            stages.push_back({stageIdentity(i, config_.pp_stages[i]), pp_stage_runners_[i]->moeOptimizationMovementLedger()});
        return composeMoEOptimizationMovementLedger(MoEOptimizationStages<MoEOptimizationMovementLedger>::seal(std::move(stages)));
    }
}
