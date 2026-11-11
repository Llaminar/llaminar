/**
 * @file MoEPipelinePreparedPlan.cpp
 * @brief Bind joint pipeline admission to source metadata and fresh runtime owners.
 *
 * Every stage reuses the existing model-aware placement resolver and checked
 * residency constructor. Only the immutable initial placements cross runner
 * lifetimes; the same PMA remains the sole allocator authority throughout.
 * CPU root declarations remain stage-owned across child projection and reuse.
 * Every fresh stage registry receives the identity of this same retained model,
 * so later prepared-source binding authenticates the actual original owner.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "MoEPipelinePreparedPlan.h"
#include "MoEOverlayRankBatchPreparation.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "MoEExpertOwnerMap.h"
#include <array>
#include <limits>
#include "MoEOverlayCPUHostMaintenance.h"
#include "ExpertHistogramServingRegime.h"
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
        if (!admission || admission->stages().empty() ||
            (!admission->globalPipeline() && admission->stages().size() < 2) || !admission->physicalAdmission())
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
            if (buildMoEOverlayLayerWeightManifestFromLoader(model->concreteLoader(), metadata.num_layers,
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
            !context || context->rank() != admission_->rankPlan().rank ||
            context->world_size() != (admission_->globalPipeline() ? admission_->globalPipeline()->world_size : 1) ||
            memory->worldRank() != context->rank() || current.pipelineStages().size() != stages_.size())
            throw std::invalid_argument("MoE pipeline runtime requires its retained rank PMA and complete topology");
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
                .capacity = *admitted.capacity(), .memory = memory,
                .world_rank = context->rank(), .world_size = context->world_size(),
                .original_model = ModelContextId{reinterpret_cast<uint64_t>(model.get())}});
            std::shared_ptr<MoEOverlayCPUHostMaintenance> host_maintenance;
            if (requested.config().moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
                plan.authority_execution == MoEOverlayAuthorityExecutionKind::HostResident &&
                !setup.authority()->noPeerResidency())
                host_maintenance = std::make_shared<MoEOverlayCPUHostMaintenance>(MoEOverlayCPUHostMaintenance::Config{
                    .metadata = prepared.metadata, .storage = admitted.storagePolicy(),
                    .rebalance = requested.config().moe_rebalance,
                    .topology = ExpertHistogramProductionTopology::forRetainedExecution(prepared.metadata.num_layers,
                        prepared.metadata.main_inference_layer_count, expertHistogramServingRegime(admitted.rankPlan().runtime.mtp),
                        prepared.metadata.first_model_layer),
                    .memory = memory, .admission = admitted.capacity()->physical_memory_admission,
                    .authority = setup.authority(), .residency = setup.residency(), .histogram = setup.histogram(),
                    .perf_device = "cpu_pipeline_stage_" + std::to_string(index)});
            auto binding = MoEOverlayPipelineStageBinding::seal(admitted.topology().scope(), prepared.metadata,
                domain->participants, {.authority = setup.authority(), .residency = setup.residency(),
                    .histogram = setup.histogram(), .mpi = context, .host_maintenance = std::move(host_maintenance),
                    .cpu_serial_expert_storage = admitted.capacity()->cpu_serial_expert_storage});
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
    std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>>
    MoEPipelinePreparedPlan::prepareOriginalCPUStageTransports(const ModelLoader& loader,
        std::shared_ptr<IMPIContext> context,const ClusterInventory& inventory,const RuntimeConfig& runtime,
        std::span<const std::shared_ptr<const MoEOverlayPipelineStageBinding>> bindings) const
    {
        const auto model=model_.lock();
        if(!model || &loader!=&model->concreteLoader() || !context ||
            context->rank()!=admission_->rankPlan().rank || context->world_size()!=inventory.world_size ||
            bindings.size()!=admission_->stages().size())
            throw std::invalid_argument("CPU pipeline transport setup requires its original model/rank/stage owners");
        std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>> result(bindings.begin(),bindings.end());
        if(!admission_->globalPipeline()) return result;
        const auto& topology=*admission_->globalPipeline();
        for(const auto& stage:topology.stages) {
            if(!stage.is_global_tp || stage.per_rank_devices.empty() ||
                !std::all_of(stage.per_rank_devices.begin(),stage.per_rank_devices.end(),[](const auto& p){return p.isCPU();})) continue;
            const auto source=stage.participating_ranks.front();
            const auto local=std::find_if(admission_->stages().begin(),admission_->stages().end(),
                [&](const auto& candidate){return candidate.topology().stageId()==stage.stage_id;});
            const auto local_index=local==admission_->stages().end()?result.size():std::size_t(local-admission_->stages().begin());
            std::string document;
            std::array<std::int64_t,8> header{};
            std::vector<int> sidecars;
            auto consensus=[&](auto identity,auto outcome){return MPIRankInitializationConsensus::reach(
                context->communicator(),identity,outcome);};
            auto frozen=RankInitializationLifecycle::execute({static_cast<std::uint32_t>(stage.stage_id),"cpu_pipeline_transport_source"},[&]{
                if(context->rank()!=source) return true;
                if(local_index==result.size() || !result[local_index])
                    throw std::logic_error("CPU pipeline transport source lost its actual local stage");
                const auto snapshot=result[local_index]->owners().authority->snapshot();
                if(!snapshot || !snapshot->valid() || snapshot->epoch!=result[local_index]->owners().residency->initialEpoch())
                    throw std::logic_error("CPU pipeline transport must freeze its original immutable initial placement");
                OrchestrationConfig envelope;
                envelope.moe_routed_expert_plan=std::make_shared<MoERoutedExpertPlacementPlan>(*snapshot->placement_plan);
                document=serializeOrchestrationConfig(envelope);
                if(document.empty() || document.size()>std::size_t(std::numeric_limits<int>::max()))
                    throw std::overflow_error("CPU pipeline transport source document exceeds original MPI count range");
                const auto& mtp=local->rankPlan().runtime.mtp;
                const int decode=retainsMTPGraphCapacity(mtp)?std::max(1,resolveMTPRetainedTargetQueryRows(mtp)):1;
                const auto family=resolveMoEOverlayInferenceGraphFamilyIdentity(loader,loader.architecture(),loader.blockCount(),
                    retainsMTPGraphCapacity(mtp)?MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars:MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                    1,std::max(runtime.overlay_prefill_schedule.graph_row_capacity,decode),decode,
                    std::max(1,runtime.batch_size),resolveMTPRetainedDraftCapacity(mtp)).forRoutedLayerInterval(
                        snapshot->placement_plan->first_model_layer,snapshot->placement_plan->placementLayerCapacity());
                header={static_cast<std::int64_t>(family.graph_family_generation),family.main_layer_count,family.max_graph_rows,
                    family.max_decode_rows,family.max_request_count,family.max_mtp_draft_depth,family.first_model_layer,
                    static_cast<std::int64_t>(family.mtp_source_layers.size())};
                sidecars=family.mtp_source_layers;
                return true;
            },consensus);
            if(!frozen.succeeded()) throw std::invalid_argument(frozen.diagnostic("CPU pipeline transport source"));
            int bytes=context->rank()==source?static_cast<int>(document.size()):0;
            if(MPI_Bcast(&bytes,1,MPI_INT,source,context->communicator())!=MPI_SUCCESS || bytes<=0 ||
                MPI_Bcast(header.data(),static_cast<int>(header.size()),MPI_INT64_T,source,context->communicator())!=MPI_SUCCESS)
                throw std::runtime_error("CPU pipeline original source metadata publication failed");
            const auto buffers=RankInitializationLifecycle::execute({static_cast<std::uint32_t>(stage.stage_id),"cpu_pipeline_transport_metadata_buffers"},[&]{
                if(header[7]<0 || header[7]>loader.blockCount())
                    throw std::invalid_argument("CPU pipeline source has a foreign retained-layer population");
                if(context->rank()!=source) {
                    document.resize(static_cast<std::size_t>(bytes));
                    sidecars.resize(static_cast<std::size_t>(header[7]));
                }
                return true;
            },consensus);
            if(!buffers.succeeded()) throw std::invalid_argument(buffers.diagnostic("CPU pipeline transport metadata buffers"));
            if(MPI_Bcast(document.data(),bytes,MPI_CHAR,source,context->communicator())!=MPI_SUCCESS)
                throw std::runtime_error("CPU pipeline original source placement publication failed");
            if(!sidecars.empty() && MPI_Bcast(sidecars.data(),static_cast<int>(sidecars.size()),MPI_INT,source,
                context->communicator())!=MPI_SUCCESS)
                throw std::runtime_error("CPU pipeline retained source-layer publication failed");
            std::shared_ptr<MoERoutedExpertPlacementPlan> placement;
            std::shared_ptr<const MoEExpertOwnerMap> owners;
            MoEOverlayInferenceGraphFamilyIdentity family;
            const auto accepted=RankInitializationLifecycle::execute({static_cast<std::uint32_t>(stage.stage_id),"cpu_pipeline_transport_admission"},[&]{
                placement=deserializeOrchestrationConfig(document).moe_routed_expert_plan;
                if(!placement || !placement->usesExpertOverlayAuthority())
                    throw std::invalid_argument("CPU pipeline source omitted its actual expert authority");
                for(std::size_t field=0;field<7;++field) if(header[field]<0 || header[field]>std::numeric_limits<int>::max())
                    throw std::invalid_argument("CPU pipeline source graph geometry exceeds native integer range");
                family={static_cast<std::uint64_t>(header[0]),static_cast<int>(header[1]),sidecars,
                    static_cast<int>(header[2]),static_cast<int>(header[3]),static_cast<int>(header[4]),
                    static_cast<int>(header[5]),static_cast<int>(header[6])};
                if(!family.valid() || family.first_model_layer!=stage.first_layer ||
                    family.main_layer_count!=stage.layerCount() || (!stage.has_lm_head && !family.mtp_source_layers.empty()) ||
                    placement->first_model_layer!=stage.first_layer)
                    throw std::invalid_argument("CPU pipeline source changed its original stage interval");
                const auto domain=std::find_if(placement->domains.begin(),placement->domains.end(),
                    [&](const auto& value){return value.name==placement->effectiveBaseModelDomain();});
                if(domain==placement->domains.end() || domain->participants!=stage.per_rank_devices ||
                    domain->world_ranks!=stage.participating_ranks || domain->scope!=ExecutionDomainScope::NODE_LOCAL)
                    throw std::invalid_argument("CPU pipeline source changed original MPI/NUMA stage membership");
                if(local_index!=result.size()) owners=result[local_index]->owners().authority->snapshot()->jointOwnerMap();
                if(!owners) {
                    if(MoEExpertOwnerMap::usesCPUJointFragments(*placement)) {
                        const auto manifest=buildMoEOverlayLayerWeightManifestFromLoader(loader,family.routedLayerCapacity(),
                            loader.getInt(loader.architecture()+".expert_count",0),family.first_model_layer);
                        owners=std::make_shared<const MoEExpertOwnerMap>(MoEExpertOwnerMap::buildShardPlacement(*placement,manifest));
                    }
                    else owners=std::make_shared<const MoEExpertOwnerMap>(MoEExpertOwnerMap::build(*placement));
                }
                return true;
            },consensus);
            if(!accepted.succeeded()) throw std::invalid_argument(accepted.diagnostic("CPU pipeline transport admission"));
            const auto registry=prepareOriginalMoEOverlayRankBatchTransports({
                .placement=*placement,.owner_map=*owners,.inventory=inventory,.graph_family=family,.original_context=context,
                .physical_memory_authority=model->concreteWeightManager()->physicalMemoryAuthority(),
                .activation_batch_inventory=local_index!=result.size()?admission_->stages()[local_index].capacity()->activation_batch_inventory:nullptr,
                .original_cpu_joint_owners=MoEExpertOwnerMap::usesCPUJointFragments(*placement)?owners:nullptr,
                .d_model=model->embeddingLength(),.top_k=loader.getInt(loader.architecture()+".expert_used_count",0)});
            const auto installed=RankInitializationLifecycle::execute({static_cast<std::uint32_t>(stage.stage_id),"cpu_pipeline_transport_binding"},[&]{
                if(local_index!=result.size() && registry)
                    result[local_index]=result[local_index]->withOriginalCPURankTransport(registry);
                return true;
            },consensus);
            if(!installed.succeeded()) throw std::invalid_argument(installed.diagnostic("CPU pipeline transport binding"));
        }
        return result;
    }

}
