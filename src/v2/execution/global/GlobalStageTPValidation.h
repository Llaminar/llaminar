/**
 * @file GlobalStageTPValidation.h
 * @brief Authenticate named-stage TP assignment before and after native context birth.
 *
 * Cross-rank pipelines have stage-local TP owners. Their enclosing rank never
 * creates a duplicate rank-wide LocalTP context merely to validate geometry.
 * Before memory admission, the sealed compiler result supplies each exact
 * stage assignment. At graph construction, the sole real context must retain
 * that same original endpoint order and explicit collective backend.
 */
#pragma once
#include "config/TPPPValidator.h"
#include "OriginalPipelineStageDomain.h"
#include "planning/ResolvedRankOrchestration.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include <algorithm>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Validate every local named-stage assignment against its original compiled action.
     * @param resolved Model-aware compiler result; cannot be constructed by a runtime caller.
     * @param model Original loaded model geometry and immutable source directory.
     * @param original_rank Actual original MPI rank whose local claims are checked.
     * @return Complete stage validation, including exact scope, TP order, weights and model geometry.
     * No collective context, native worker, device allocation or model weight is materialized here.
     */
    inline TPPPValidationResult validateResolvedGlobalStageTP(
        const ResolvedRankOrchestration &resolved, const IModelContext &model, int original_rank)
    {
        TPPPValidationResult result;
        if (!resolved.globalPipeline() || resolved.rankPlan().rank != original_rank || original_rank < 0 ||
            original_rank >= resolved.globalPipeline()->world_size || resolved.pipelineStages().empty())
        {
            result.addError("Global stage TP validation requires the exact original rank and sealed local stages");
            return result;
        }
        const auto actions = GlobalPPRankPlanBuilder::build(*resolved.globalPipeline(), original_rank);
        const auto local = actions.executeStages();
        if (local.size() != resolved.pipelineStages().size())
        {
            result.addError("Global stage TP validation omitted a local executable action");
            return result;
        }
        for (const auto &stage : resolved.pipelineStages())
        {
            const auto found = std::find_if(local.begin(), local.end(), [&](const auto *action) {
                return action->stage_id == stage.stageId();
            });
            const auto &plan = stage.rankPlan();
            const auto &config = stage.config();
            const auto reject = [&](const char *field) {
                result.addError("stage " + std::to_string(stage.stageId()) +
                    ": original compiled assignment differs at " + field);
            };
            if (found == local.end()) { reject("stage ID"); continue; }
            const auto &action = **found;
            const auto original_stage=std::find_if(resolved.globalPipeline()->stages.begin(),
                resolved.globalPipeline()->stages.end(),[&](const auto& original){return original.stage_id==stage.stageId();});
            if(original_stage==resolved.globalPipeline()->stages.end()) {reject("original stage domain");continue;}
            std::optional<OriginalPipelineStageDomain> original;
            try {original=OriginalPipelineStageDomain::borrow(*original_stage,action);}
            catch(const std::invalid_argument& error) {
                result.addError("stage "+std::to_string(stage.stageId())+": "+error.what());continue;
            }
            const bool global_cpu=action.is_global_tp;
            if(original->participants.empty() || (global_cpu &&
                (!action.devices.empty() || !action.tp_weights.empty() ||
                 !std::all_of(original->participants.begin(),original->participants.end(),[](const auto& endpoint){return endpoint.isCPU();}) ||
                 original_stage->participating_ranks[action.tp_rank_in_domain]!=original_rank))) {
                reject("original CPU execute role");continue;
            }
            const auto same=[](const auto& left,const auto& right){
                return left.size()==right.size() && std::equal(left.begin(),left.end(),right.begin(),right.end());
            };
            const bool local_tp = !global_cpu && action.devices.size() > 1;
            const std::vector<GlobalDeviceAddress> no_devices;
            const std::vector<float> no_weights;
            bool exact = true;
            const auto require = [&](bool valid, const char *field) {
                if (!valid) { reject(field); exact = false; }
            };
            require(plan.rank == original_rank, "original rank");
            require(plan.primary_device == (global_cpu?action.device:action.devices.front()), "primary endpoint");
            require(plan.local_tp_devices == (local_tp ? action.devices : no_devices), "rank LocalTP endpoint order");
            require(plan.local_tp_weights == (local_tp ? action.tp_weights : no_weights), "rank LocalTP weights");
            require(plan.local_tp_backend == action.backend, "rank collective backend");
            require(plan.tp_scope == (global_cpu?TPScope::NODE_LOCAL:local_tp?TPScope::RANK_LOCAL:TPScope::AUTO), "rank TP scope");
            if(global_cpu) {
                require(plan.usesGlobalTP() && !plan.usesLocalTP() &&
                    plan.global_tp_rank_in_domain==action.tp_rank_in_domain &&
                    plan.global_tp_domain_size==action.tp_domain_size,"original global TP coordinates");
                require(plan.weight_shard.validate().empty() &&
                    plan.weight_shard.shard_index==action.weight_shard.shard_index &&
                    plan.weight_shard.total_shards==action.weight_shard.total_shards &&
                    plan.weight_shard.work_fraction==action.weight_shard.work_fraction,"original dense weight shard");
                const auto parent=std::find_if(resolved.rankPlan().my_domains.begin(),resolved.rankPlan().my_domains.end(),
                    [&](const auto& domain){return domain.domain_name==action.domain_name;});
                require(parent!=resolved.rankPlan().my_domains.end(),"original parent domain participation");
                require(plan.my_domains.size()==1,"sole original stage domain participation");
                if(parent!=resolved.rankPlan().my_domains.end() && plan.my_domains.size()==1) {
                    const auto& member=plan.my_domains.front();
                    require(plan.global_tp_domain_id==std::optional{parent->domain_id} &&
                        member.domain_id==parent->domain_id && member.domain_name==action.domain_name &&
                        member.my_index_in_domain==action.tp_rank_in_domain &&
                        same(member.devices,original->participants) && same(member.weights,original->weights) &&
                        member.backend==original->backend,"original stage domain/shard authority");
                }
            }
            require(stage.scope().first_layer == action.first_layer &&
                stage.scope().last_layer == action.last_layer + 1, "owned model interval");
            require(stage.scope().has_embedding == action.has_embedding &&
                stage.scope().has_lm_head == action.has_lm_head, "embedding/vocabulary ownership");
            // SynthesizedSimpleTP retires selectors when the named overlay is
            // installed. Authenticate its sole canonical domain, rather than
            // expecting a second simple-TP topology to survive normalization.
            require(config.tp_devices.empty() && config.tp_weights.empty() &&
                config.tp_degree == 1 && config.tp_scope == TPScope::AUTO, "retired simple-TP selectors");
            if (!config.moe_routed_expert_plan || config.moe_routed_expert_plan->domains.size() != 1)
            {
                reject("canonical stage overlay domain");
                continue;
            }
            const auto &domain = config.moe_routed_expert_plan->domains.front();
            require(domain.name == config.moe_routed_expert_plan->effectiveBaseModelDomain(), "base-model domain identity");
            require(same(domain.participants,original->participants), "overlay endpoint order");
            require(same(domain.weights,original->weights), "overlay TP weights");
            require(domain.backend == action.backend, "overlay collective backend");
            require(global_cpu?(!domain.owner_rank && domain.world_ranks==original_stage->participating_ranks):
                (domain.owner_rank==original_rank && std::all_of(domain.world_ranks.begin(),domain.world_ranks.end(),
                    [&](int rank){return rank==original_rank;})),"overlay original rank ownership");
            require(domain.scope == (global_cpu?ExecutionDomainScope::NODE_LOCAL:local_tp?ExecutionDomainScope::RANK_LOCAL:ExecutionDomainScope::SINGLE),
                "overlay execution scope");
            require(config.domain_definitions.size() == 1, "normalized domain alias count");
            if (config.domain_definitions.size() == 1)
            {
                const auto &alias = config.domain_definitions.front();
                require(alias.name == domain.name && same(alias.devices,original->participants) &&
                    same(alias.weights,original->weights) && alias.backend==action.backend &&
                    alias.owner_rank==domain.owner_rank && alias.explicit_ranks==domain.world_ranks &&
                    alias.routed_compute_policy==domain.routed_compute_policy,
                    "normalized domain assignment");
                if(global_cpu) require(alias.scope==TPScope::NODE_LOCAL,"normalized original CPU domain scope");
            }
            require(config.routed_expert_compute_policy==domain.routed_compute_policy &&
                plan.runtime.routed_expert_compute_policy==domain.routed_compute_policy,"normalized stage compute policy");
            if(global_cpu) require(config.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy()==
                DenseParallelPolicy::TensorParallel,"original CPU dense TP continuation");
            if (!exact) continue;
            if(global_cpu) {
                try {
                    // Use the SAME rank-indexed CPU assignment factory as the
                    // actual GlobalTP runner. Device IDs represent local CPU
                    // allocators only; original MPI/address ownership was
                    // authenticated above and never becomes a LocalTP vector.
                    std::vector<DeviceId> allocators;
                    for(const auto& endpoint:original->participants) allocators.push_back(endpoint.toLocalDeviceId());
                    const auto assignment=TensorParallelConfig::equalSplit(action.tp_domain_size,
                        model.headCount(),model.headCountKV(),model.feedForwardLength(),model.vocabSize(),allocators);
                    require(assignment.worldSize()==action.tp_domain_size &&
                        assignment.forRank(action.tp_rank_in_domain).device==action.device.toLocalDeviceId(),
                        "original CPU model assignment");
                } catch(const std::exception& error) {
                    result.addError("stage "+std::to_string(stage.stageId())+": original CPU model assignment: "+error.what());
                }
            }
            auto checked = TPPPValidator::validate(config, model,
                TPPPAssignmentMembership::resolved(domain.participants));
            for (const auto &error : checked.errors)
                result.addError("stage " + std::to_string(stage.stageId()) + ": " + error);
            for (const auto &warning : checked.warnings)
                result.addWarning("stage " + std::to_string(stage.stageId()) + ": " + warning);
        }
        return result;
    }

    /**
     * @brief Authenticate the sole stage-local context after its actual native construction.
     * @param action Original compiled execute action consumed by the stage factory.
     * @param devices Live context's exact original endpoint order.
     * @param backend Live context's resolved collective backend.
     * @throws std::invalid_argument For reordered/missing endpoints, a non-execute role or changed explicit backend.
     * Weight normalization remains owned by LocalTPContext, which receives action.tp_weights directly.
     */
    inline void requireOriginalGlobalStageTPBirth(const RankStageAction &action,
        std::span<const GlobalDeviceAddress> devices, CollectiveBackendType backend)
    {
        if (action.role != RankStageAction::Role::EXECUTE || action.is_global_tp || action.devices.empty() ||
            !std::equal(action.devices.begin(), action.devices.end(), devices.begin(), devices.end()) ||
            (action.backend != CollectiveBackendType::AUTO && action.backend != backend))
            throw std::invalid_argument("Global stage LocalTP birth changed its original compiled membership or backend");
    }
}
