/**
 * @file MoEPipelinePreparedPlan.cpp
 * @brief Bind joint pipeline admission to source metadata and fresh runtime owners.
 *
 * Every stage reuses the existing model-aware placement resolver and checked
 * residency constructor. Only the immutable initial placements cross runner
 * lifetimes; the same PMA remains the sole allocator authority throughout.
 */
#include "MoEPipelinePreparedPlan.h"
#include "MoEOverlayPipelineStageBinding.h"
#include "MoEOverlayResidencySetup.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "interfaces/IMPIContext.h"
#include "loaders/ModelContext.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    std::shared_ptr<const MoEPipelinePreparedPlan> MoEPipelinePreparedPlan::create(
        std::shared_ptr<const AdmittedMoEPipelineMemory> admission, const std::shared_ptr<ModelContext> &model)
    {
        if (!admission || admission->stages().size() < 2 || !admission->physicalAdmission())
            throw std::invalid_argument("Prepared MoE pipeline requires a complete aggregate admission");
        if (!model || model->path() != admission->stages().front().topology().config().model_path)
            throw std::invalid_argument("Prepared MoE pipeline requires its admitted GGUF metadata source");
        auto result = std::shared_ptr<MoEPipelinePreparedPlan>(
            new MoEPipelinePreparedPlan(std::move(admission), model));
        for (const auto &stage : result->admission_->stages())
        {
            const auto &scope = stage.topology().scope();
            const auto metadata = resolveMoERoutedExpertModelMetadataForModel(*model, stage.rankPlan().runtime.mtp, scope);
            if (metadata.first_model_layer != stage.graphFamily().first_model_layer ||
                metadata.num_layers != stage.graphFamily().routedLayerCapacity() ||
                metadata.main_inference_layer_count != scope.last_layer ||
                metadata.num_experts != stage.capacity()->num_experts ||
                stage.capacity()->physical_memory_admission != result->admission_->physicalAdmission())
                throw std::invalid_argument("Prepared MoE pipeline metadata differs from its admitted source interval");
            if (buildMoEOverlayLayerWeightManifestFromGGUF(model->concreteLoader().getModel(), metadata.num_layers,
                    metadata.num_experts, metadata.first_model_layer) != stage.weightManifest())
                throw std::invalid_argument("Prepared MoE pipeline source projection geometry or format changed after admission");
            InferenceRunnerConfig config;
            config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
                MoEOverlayCapacityResolver::installResolvedQuotas(
                    *stage.topology().config().moe_routed_expert_plan, *stage.capacity()));
            config.mtp = stage.rankPlan().runtime.mtp;
            config.pp_stage_config = scope;
            auto placement = resolveMoERoutedExpertPlacementPlanForModel(*model, config);
            if (!placement || placement->placements.empty())
                throw std::logic_error("Prepared MoE pipeline did not freeze a complete stage placement");
            result->stages_.push_back({metadata, std::move(placement)});
        }
        return result;
    }

    std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>>
    MoEPipelinePreparedPlan::createRuntimeBindings(
        const ModelLoader &loader, std::shared_ptr<PhysicalMemoryAuthority> memory,
        std::shared_ptr<IMPIContext> context, const ResolvedRankOrchestration &current) const
    {
        const auto model = model_.lock();
        if (!model || &loader != &model->concreteLoader() || !memory || memory->admission() != admission_->physicalAdmission() ||
            !context || context->rank() != 0 || context->world_size() != 1 ||
            memory->worldRank() != context->rank() || current.pipelineStages().size() != stages_.size())
            throw std::invalid_argument("MoE pipeline runtime requires its retained local PMA and complete topology");
        if (pipelineRoutedWeightAuthorityRequestIdentity(current) !=
            pipelineRoutedWeightAuthorityRequestIdentity(*admission_))
            throw std::invalid_argument("MoE pipeline runtime policy differs from its retained prepared-weight admission");
        std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>> bindings;
        for (std::size_t index = 0; index < stages_.size(); ++index)
        {
            const auto &prepared = stages_[index];
            const auto &admitted = admission_->stages()[index];
            const auto &requested = current.pipelineStages()[index];
            const auto &plan = *prepared.placement;
            const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                [&](const auto &value) { return value.name == plan.effectiveBaseModelDomain(); });
            if (domain == plan.domains.end())
                throw std::logic_error("Prepared MoE pipeline lost its continuation domain");
            const auto setup = MoEOverlayResidencySetup::create({
                .plan = plan, .metadata = prepared.metadata, .loader = loader,
                .rebalance = requested.config().moe_rebalance, .storage_policy = admitted.storagePolicy(),
                .capacity = *admitted.capacity(), .memory = memory});
            auto binding = MoEOverlayPipelineStageBinding::seal(admitted.topology().scope(), prepared.metadata,
                domain->participants, {.authority = setup.authority(), .residency = setup.residency(),
                    .histogram = setup.histogram(), .mpi = context});
            const auto &requested_plan = *requested.config().moe_routed_expert_plan;
            const auto destination = std::find_if(requested_plan.domains.begin(), requested_plan.domains.end(),
                [&](const auto &value) { return value.name == requested_plan.effectiveBaseModelDomain(); });
            if (destination == requested_plan.domains.end())
                throw std::invalid_argument("MoE pipeline destination lost its continuation domain");
            binding->requireDestination(requested.scope(), destination->participants, destination->weights,
                destination->backend, requested.config().moe_rebalance.mode);
            bindings.push_back(std::move(binding));
        }
        return bindings;
    }
}
