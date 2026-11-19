/**
 * @file RankMemoryPlanInputs.cpp
 * @brief Canonical ordinary rank-plan to physical BOM input construction.
 *
 * Keep allocation geometry identical for plan/auto and runtime admission.
 * This module neither owns free-byte balances nor performs driver discovery:
 * the inventory is an observation and PhysicalMemoryAuthority remains the
 * sole ledger. Local TP uses the graph's exact proportional slice, not an
 * independent integer split reconstructed from the shard count. Ordinary and
 * ExpertOverlay pipelines share the same physical transfer-leader projection;
 * replicated weight views do not become extra channel owners. Authored CPU
 * NUMA participants retain their compute shards while sharing the one physical
 * host allocator; transfer ownership must not mistake that sharing for duplicate
 * compute membership.
 *
 * Source-qualified query KV intervals come from the same canonical assignment
 * used for weights and graphs; PMA alone accounts their actual physical bytes.
 * Retained graph owners follow declared participant domains and actual PP roles.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "planning/PlanningObservedResource.h"
#include "planning/RankMemoryPlanInputs.h"
#include "execution/global/HostedPipelineActivationStorage.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "backends/NativeContextPreparation.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "utils/DebugEnv.h"
#include "planning/ActivationBufferSizing.h"
#include "backends/DeviceAddressAdapter.h"
#include "config/BackendSelector.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "planning/WeightShardGeometry.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include "loaders/PreparedWeightIdentityContract.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include <span>

namespace llaminar2
{
    std::optional<NativeWeightLoadStagingGeometry> quoteCPUWeightSourceStaging(
        const ModelMemoryProfile& profile,const DevicePlanConfig& cfg,
        GPUWeightLoadMemoryPolicy policy,WeightPrecision precision,CPUStoragePlacement placement)
    {
        if(!cfg.device.is_cpu())throw std::invalid_argument("CPU source quote requires its original CPU participant");
        const auto schema=SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture);
        const auto terminal=SchemaFactoryRegistry::getFactory(profile.architecture)->terminalInputWeightNames();
        const int last=cfg.last_layer<0?profile.mainLayerCount()-1:cfg.last_layer;
        const bool continuation=cfg.execution_role==DeviceExecutionMemoryRole::ContinuationGraph;
        const bool terminal_owner=continuation && last>=profile.mainLayerCount()-1;
        const bool routed_owner=cfg.serial_routed_expert_participant_count>0 ||
            cfg.routed_expert_compact_buffer_lifetime==RoutedExpertCompactBufferLifetime::PerLayerGraphOwned;
        const bool tied_head=std::none_of(profile.tensors.begin(),profile.tensors.end(),[](const auto& tensor){
            return inferWeightRole(tensor.name)==WeightRole::LMHead;
        });
        WeightShardGeometryResolver source(profile,cfg.device,cfg.shard_index,cfg.total_shards,cfg.tensor_parallel_assignment);
        std::optional<NativeWeightLoadStagingGeometry> maximum;
        for(const auto& tensor:profile.tensors) {
            const auto original_role=inferWeightRole(tensor.name);
            const bool head_alias=original_role==WeightRole::Embedding && terminal_owner && tied_head;
            const auto role=head_alias?WeightRole::LMHead:original_role;
            auto semantic=tensor;
            if(head_alias)semantic.name="output.weight";
            const bool routed=isRoutedExpertRole(role);
            if(routed && !routed_owner)continue;
            if(!routed && !continuation)continue;
            if(tensor.layer_index>=0) {
                const bool main=tensor.layer_index>=cfg.first_layer && tensor.layer_index<=last &&
                    tensor.layer_index<profile.mainLayerCount();
                const bool predictor=terminal_owner && cfg.mtp_enabled && tensor.layer_index>=profile.mainLayerCount() &&
                    tensor.layer_index<profile.totalLayerCount();
                if(!main && !predictor)continue;
            } else {
                if(original_role==WeightRole::Embedding && !cfg.owns_embedding && !head_alias)continue;
                if(!terminal_owner && std::find(terminal.begin(),terminal.end(),tensor.name)!=terminal.end())continue;
            }
            const auto type=tensorTypeFromName(tensor.quant_type);
            if(!type)throw std::invalid_argument("CPU source quote has unknown original tensor format: "+tensor.quant_type);
            const auto joint=cfg.cpu_serial_expert_storage
                ? cfg.cpu_serial_expert_storage->envelope->admitted()->jointRoutes():std::optional<CPUJointExpertRouteStorageLayout>{};
            const auto shape=routed && joint
                ? source.resolve(semantic,joint->cohort()->ownership(joint->member()),size_t(profile.expert_count))
                : source.resolve(semantic,routed?std::optional<size_t>{size_t(profile.expert_count)}:std::nullopt);
            const size_t rows=shape.matrix()?shape.matrix()->rows:1;
            const size_t columns=shape.matrix()?shape.matrix()->columns:shape.elements();
            if(!rows || !columns)continue; // An actual empty shard has no source transaction.
            const auto classification=PreparedWeightIdentityContract::classifyOriginalMatrix(role,cfg.device);
            const bool prepared=routed || (shape.matrix() && classification!=PreparedGemmClassification::Rejected &&
                (classification!=PreparedGemmClassification::RequiresSchema || !schema.isNonGemmWeight(semantic.name)));
            const auto execution=PreparedWeightRepresentationContract::resolve(role,*type,tensor.name,profile.architecture,
                debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
            const auto operand=PreparedWeightRepresentationContract::operandPrecision(role,tensor.name,profile.architecture,
                debugEnv().gemm.gdn_projection_arithmetic,debugEnv().gemm.moe_router_arithmetic);
            const auto destination=PreparedWeightRepresentationContract::resolveCPUWeightPrecision(role,*type,precision,execution,operand);
            const auto quote=quoteCPUWeightPreparation(*type,rows,columns,destination,
                prepared?CPUWeightConsumer::PreparedGemm:CPUWeightConsumer::ResidentTensor,operand);
            if(!quote)continue; // Explicit direct writes into the final admitted resident root.
            const auto geometry=resolveNativeWeightLoadStaging(*quote,placement,
                policy.staging_budget_bytes?std::optional<size_t>{policy.staging_budget_bytes}:std::nullopt);
            if(!maximum || geometry.physical_bytes>maximum->physical_bytes)maximum=geometry;
        }
        return maximum;
    }

    PipelineStageTransferMemory PipelineStageTransferMemory::forStage(const RankExecutionPlan &parent, size_t stage)
    {
        const auto &devices = parent.local_pp_devices;
        const auto &boundaries = parent.local_pp_layer_boundaries;
        const auto &stage_tp = parent.local_pp_stage_tp_info;
        if (parent.rank < 0 || devices.size() < 2 || stage >= devices.size() ||
            boundaries.size() != devices.size() + 1 || boundaries.front() < 0 ||
            (!stage_tp.empty() && stage_tp.size() != devices.size()) ||
            !std::is_sorted(boundaries.begin(), boundaries.end()) ||
            std::adjacent_find(boundaries.begin(), boundaries.end()) != boundaries.end())
            throw std::invalid_argument("Pipeline transfer memory requires exact nonempty stage intervals and participants");
        PipelineStageTransferMemory result;
        result.world_rank_ = parent.rank;
        result.first_layer_ = boundaries[stage];
        result.last_layer_ = boundaries[stage + 1] - 1;
        const std::span<const GlobalDeviceAddress> members =
            stage_tp.empty() || stage_tp[stage].devices.empty()
                ? std::span<const GlobalDeviceAddress>(&devices[stage], 1)
                : std::span<const GlobalDeviceAddress>(stage_tp[stage].devices);
        if (members.front() != devices[stage])
            throw std::invalid_argument("Pipeline transfer leader differs from its authored TP domain");

        // Validate addresses before the lossy physical-allocator projection.
        // CPU NUMA endpoints are different workers, but DeviceId::cpu() is their
        // common storage authority. Conversely, changing a GPU's NUMA annotation
        // cannot create another physical GPU or another channel owner.
        const auto supported = [](const GlobalDeviceAddress &address) {
            return address.device_ordinal >= 0 &&
                (address.device_type == DeviceType::CPU || address.device_type == DeviceType::CUDA ||
                 address.device_type == DeviceType::ROCm);
        };
        for (const auto &leader : devices)
            if (!supported(leader) || leader.hostname != devices.front().hostname)
                throw std::invalid_argument("Local pipeline transfer has an invalid or foreign stage leader");
        for (size_t index = 0; index < members.size(); ++index)
        {
            const auto &member = members[index];
            if (!supported(member) || member.hostname != devices[stage].hostname)
                throw std::invalid_argument("Local pipeline transfer has an invalid or foreign TP participant");
            if (std::any_of(members.begin(), members.begin() + index, [&](const auto &previous) {
                    return previous == member ||
                        (member.device_type == DeviceType::CPU && previous.device_type == DeviceType::CPU &&
                         previous.numa_node == member.numa_node);
                }))
                throw std::invalid_argument("Pipeline transfer memory requires distinct authored TP participants");
            const DeviceId physical = member.toLocalDeviceId();
            const bool seen = std::find(result.participants_.begin(), result.participants_.end(), physical) !=
                result.participants_.end();
            if (!physical.is_valid() || (seen && !physical.is_cpu()))
                throw std::invalid_argument("Pipeline transfer memory requires distinct valid physical participants");
            if (!seen)
                result.participants_.push_back(physical);
        }
        // A complete homogeneous GPU pipeline retains one native PP context.
        // Stage-local TP contexts remain separate original compute authorities;
        // native byte endpoint incidence, not stage count, prices their edges.
        const bool native = devices.front().toLocalDeviceId().is_gpu() &&
            std::all_of(devices.begin(), devices.end(), [&](const auto &device) {
                return device.device_type == devices.front().device_type;
            }) && std::all_of(stage_tp.begin(), stage_tp.end(), [&](const auto &domain) {
                return std::all_of(domain.devices.begin(), domain.devices.end(), [&](const auto &member) {
                    return member.device_type == devices.front().device_type;
                });
            });
        if (native)
            result.native_backend_ = result.participants_.front().is_cuda()
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        const bool nested_native = native && (parent.runtime.ngram_table_policy.isFixedHeadSharded() ||
            std::any_of(stage_tp.begin(), stage_tp.end(), [](const auto &entry) {
                return entry.devices.size() > 1;
            }));
        if (nested_native) {
            const auto width = [&](std::size_t index) -> std::size_t {
                return stage_tp.empty() || stage_tp[index].devices.empty() ? 1 : stage_tp[index].devices.size();
            };
            result.native_byte_endpoints_.assign(result.participants_.size(), 0);
            for (std::size_t member = 0; member < result.participants_.size(); ++member) {
                auto &count = result.native_byte_endpoints_[member];
                // Incoming activations and reverse metadata reach every actual consumer.
                count = std::size_t(stage > 0) + std::size_t(stage + 1 < devices.size());
                if (member == 0) {
                    const auto add = [&](std::size_t peers) {
                        if (peers > std::numeric_limits<std::size_t>::max() - count)
                            throw std::overflow_error("Native pipeline endpoint incidence overflow");
                        count += peers;
                    };
                    if (stage > 0) add(width(stage - 1));
                    if (stage + 1 < devices.size()) add(width(stage + 1));
                }
            }
        }
        const auto leader = result.participants_.front();
        const auto cross_backend = [&](size_t peer) {
            const auto other = devices[peer].toLocalDeviceId();
            return leader.is_gpu() && other.is_gpu() && leader.type != other.type;
        };
        if (stage && cross_backend(stage - 1))
            result.captured_boundaries_.push_back(PipelineBoundarySide::LaterDomain);
        if (stage + 1 < devices.size() && cross_backend(stage + 1))
            result.captured_boundaries_.push_back(PipelineBoundarySide::EarlierDomain);
        result.native_activation_fanout_ = members.size() > 1 &&
            std::find(result.captured_boundaries_.begin(), result.captured_boundaries_.end(),
                PipelineBoundarySide::LaterDomain) != result.captured_boundaries_.end();
        return result;
    }

    PipelineStageTransferMemory PipelineStageTransferMemory::forGlobalStage(
        const GlobalPPTopology &topology, int world_rank, int stage_id)
    {
        if (world_rank < 0 || world_rank >= topology.world_size || !topology.validate().empty())
            throw std::invalid_argument("Global stage memory requires its complete canonical topology");
        const auto rank = GlobalPPRankPlanBuilder::build(topology, world_rank);
        const auto actions = rank.executeStages();
        const auto found = std::find_if(actions.begin(), actions.end(),
            [&](const auto *action) { return action->stage_id == stage_id; });
        if(found==actions.end()) throw std::invalid_argument("Global stage memory requires its original executable action");
        const auto& local=**found;
        const bool global_cpu=local.is_global_tp && local.device.isCPU();
        if((local.is_global_tp && !global_cpu) || (!local.is_global_tp && local.devices.empty()))
            throw std::invalid_argument("Global stage memory requires original local devices or its actual CPU MPI endpoint");
        PipelineStageTransferMemory result;
        result.world_rank_ = world_rank;
        result.first_layer_ = (*found)->first_layer;
        result.last_layer_ = (*found)->last_layer;
        const auto local_members=global_cpu?std::vector<GlobalDeviceAddress>{local.device}:local.devices;
        for (const auto &member : local_members)
        {
            const auto device = member.toLocalDeviceId();
            if (!device.is_valid() || std::find(result.participants_.begin(), result.participants_.end(), device) !=
                    result.participants_.end())
                throw std::invalid_argument("Global stage memory requires distinct physical members");
            result.participants_.push_back(device);
        }
        const bool hosted=std::any_of(topology.stages.begin(),topology.stages.end(),[](const auto& stage){
            return stage.is_global_tp ? std::any_of(stage.per_rank_devices.begin(),stage.per_rank_devices.end(),[](const auto& d){return d.isCPU();})
                : std::any_of(stage.devices.begin(),stage.devices.end(),[](const auto& d){return d.isCPU();});
        });
        result.hosted_activation_bank_owner_ = hosted &&
            HostedPipelineActivationStorage::requiresBank(topology, rank) &&
            !actions.empty() && actions.front()->stage_id == stage_id;
        // Original global adjacency, not the local action list, owns every
        // boundary. Only authenticated same-node GPU edges use this mapped
        // transport; CPU stages retain their separately declared host path.
        const auto &current = topology.stages.at(static_cast<size_t>(stage_id));
        const auto gpu_edge = [&](int peer, PipelineBoundarySide side) {
            const auto &other = topology.stages.at(static_cast<size_t>(peer));
            if (!result.participants_.front().is_gpu() || other.is_global_tp || other.devices.empty() ||
                !other.devices.front().toLocalDeviceId().is_gpu()) return;
            if (current.inner_mode == InnerParallelism::LOCAL_PP || other.inner_mode == InnerParallelism::LOCAL_PP)
                throw std::invalid_argument("Global captured boundary requires actual single/native TP stage domains");
            const auto *edge = topology.transferBetween(std::min(stage_id, peer), std::max(stage_id, peer));
            if (!edge || edge->physicalConnection().locality() == RankConnectionLocality::CrossNode)
                throw std::invalid_argument("Global captured GPU boundary requires original authenticated node-local ownership");
            result.captured_boundaries_.push_back(side);
        };
        if (stage_id > 0) gpu_edge(stage_id - 1, PipelineBoundarySide::LaterDomain);
        if (static_cast<size_t>(stage_id + 1) < topology.stages.size())
            gpu_edge(stage_id + 1, PipelineBoundarySide::EarlierDomain);
        result.native_activation_fanout_ = result.participants_.size() > 1 &&
            std::find(result.captured_boundaries_.begin(), result.captured_boundaries_.end(),
                PipelineBoundarySide::LaterDomain) != result.captured_boundaries_.end();
        return result;
    }

    bool PipelineStageTransferMemory::requiresHostMemory(DeviceId device) const noexcept
    {
        return (hosted_activation_bank_owner_ && device==participants_.front()) || (native_activation_fanout_ &&
                std::find(participants_.begin(), participants_.end(), device) != participants_.end()) ||
            (device == participants_.front() &&
             std::find(captured_boundaries_.begin(), captured_boundaries_.end(), PipelineBoundarySide::EarlierDomain) !=
                captured_boundaries_.end());
    }

    void PipelineStageTransferMemory::bind(DevicePlanConfig &config, int hosted_row_capacity) const
    {
        if (config.world_rank != world_rank_ || config.first_layer != first_layer_ || config.last_layer != last_layer_ ||
            config.execution_role != DeviceExecutionMemoryRole::ContinuationGraph || config.local_pipeline_backend ||
            config.pipeline_forward_only_graph_owner != PipelineForwardOnlyGraphOwner::None ||
            !config.captured_pipeline_boundaries.empty() || config.pipeline_native_activation_fanout || config.native_pipeline_byte_endpoints ||
            config.hosted_pipeline_activation_rows != 0 ||
            std::find(participants_.begin(), participants_.end(), config.device) == participants_.end())
            throw std::invalid_argument("Pipeline transfer memory cannot bind a foreign or already bound device graph");
        if (requiresHostMemory(config.device) && config.device.is_gpu() &&
            (!config.associated_host_memory || !config.associated_host_memory->valid() ||
             config.associated_host_memory->world_rank != world_rank_ || !config.associated_host_memory->device.is_cpu()))
            throw std::invalid_argument("Pipeline mapped channel owner requires its exact physical host authority");
        // Every physical GPU imports its own resident model body. A TP sibling
        // owns a parent even though only its leader owns boundary storage.
        config.pipeline_forward_only_graph_owner = config.device.is_gpu()
            ? PipelineForwardOnlyGraphOwner::Participant : PipelineForwardOnlyGraphOwner::None;
        config.pipeline_native_activation_fanout = native_activation_fanout_;
        if(hosted_activation_bank_owner_ && config.device==participants_.front()) {
            config.hosted_pipeline_activation_rows=hosted_row_capacity>0?hosted_row_capacity:config.activation_seq_len;
            if(config.hosted_pipeline_activation_rows<=0)
                throw std::invalid_argument("Hosted pipeline bank requires its exact admitted row capacity");
        }
        if (!native_byte_endpoints_.empty()) {
            const auto member = std::find(participants_.begin(), participants_.end(), config.device) - participants_.begin();
            config.native_pipeline_byte_endpoints = native_byte_endpoints_.at(member);
            config.local_pipeline_backend = native_backend_;
        }
        if (config.device == participants_.front())
        {
            config.local_pipeline_backend = native_backend_;
            config.captured_pipeline_boundaries = captured_boundaries_;
        }
    }

    ImmutableRowTableTPTopology immutableRowTableTPTopology(
        const ModelMemoryProfile &profile,const RankExecutionPlan &plan)
    {
        const auto &ngram=plan.runtime.ngram_table_policy;
        ngram.validate();
        const bool consumerLocal = ngram.isGPUAdaptive() &&
            ngram.gpu_adaptive.placement.mode == AdaptiveGPUCachePlacementMode::ConsumerLocal && plan.usesLocalTP();
        const bool sharedReaders=ngram.isGPUAdaptive() &&
            ngram.gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal && plan.usesLocalTP();
        if(ngram.isFixedHeadSharded() || consumerLocal || sharedReaders)
        {
            const auto &devices=plan.local_tp_devices;
            if(!profile.flash_next || !plan.usesLocalTP() || !plan.local_pp_devices.empty() ||
               plan.usesGlobalTP() || plan.prev_rank || plan.next_rank ||
               plan.first_layer!=0 || (plan.last_layer>=0 && plan.last_layer!=profile.mainLayerCount()-1) ||
               !plan.has_embedding || !plan.has_lm_head ||
               (devices.size()!=2 && devices.size()!=4 && devices.size()!=8))
                throw std::invalid_argument(sharedReaders
                    ? "Shared replicated adaptive lookup requires full-main rank-local homogeneous GPU TP degree2/4/8 with original addressed native completion"
                    : consumerLocal
                    ? "ConsumerLocal adaptive lookup requires full-main rank-local homogeneous GPU TP degree 2/4/8, without PP/global TP"
                    : "FixedHeadSharded requires full-main rank-local homogeneous GPU TP degree2/4/8, without PP/global TP");
            const auto type=devices.front().device_type;
            for(const auto &d:devices)
                if(!d.toLocalDeviceId().is_gpu() || d.device_type!=type || d.hostname!=plan.primary_device.hostname)
                    throw std::invalid_argument("Ngram TP has a foreign, CPU, or mixed-backend participant");
            if(sharedReaders){
                const auto& placement=ngram.gpu_adaptive.placement;
                const auto& coordinator=*placement.coordinator;
                const auto member=[&](DeviceId device){return std::any_of(devices.begin(),devices.end(),[&](const auto& value){return value.toLocalDeviceId()==device;});};
                if((type!=DeviceType::ROCm && type!=DeviceType::CUDA) ||
                   coordinator.world_rank!=plan.rank || !member(coordinator.device()))
                    throw std::invalid_argument("Shared replicated PLE requires authored homogeneous CUDA/ROCm coordinator membership");
                // Physical cache ownership is independent of original reader
                // membership. A native owner may be any original continuation;
                // every other owner uses the existing opposite-vendor channel.
                // Neither role grants a new communicator or execution authority.
                for (const auto& owner : placement.owners) {
                    if (owner.participant.world_rank != plan.rank)
                        throw std::invalid_argument("Shared PLE storage owner crossed its original rank authority");
                    if (owner.participant.device().type == type && !member(owner.participant.device()))
                        throw std::invalid_argument("Shared PLE same-vendor storage owner is outside the original native communicator; this topology is unqualified and unsupported");
                    if (!owner.participant.device().is_gpu())
                        throw std::invalid_argument("Shared PLE storage owner is not an original GPU participant");
                }
            }
        }
        return (ngram.isFixedHeadSharded() || consumerLocal || sharedReaders)?ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal:
            ImmutableRowTableTPTopology::None;
    }

    DeviceGenerationExecutionDomain declaredRankDeviceGenerationDomain(const RankExecutionPlan &plan)
    {
        DeviceGenerationDomainDeclaration generation;
        // A PP parent is a coordinator; its default primary slot is not an
        // additional CPU participant. TP likewise declares its complete members.
        if (!plan.usesLocalPP() && !plan.usesLocalTP())
            generation.observe(plan.primary_device.device_type);
        for (const auto &member : plan.local_tp_devices)
            generation.observe(member.device_type);
        for (const auto &member : plan.local_pp_devices)
            generation.observe(member.device_type);
        for (const auto &stage : plan.local_pp_stage_tp_info)
            for (const auto &member : stage.devices)
                generation.observe(member.device_type);
        for (const auto &domain : plan.my_domains)
            for (const auto &member : domain.devices)
                generation.observe(member.device_type);
        return generation.domain();
    }

    std::vector<DevicePlanConfig> buildRankMemoryPlanInputs(
        const RankMemoryPlanInputRequest &request)
    {
        if (request.inventory.rank != request.plan.rank || request.plan.rank < 0)
            throw std::invalid_argument("Memory BOM inventory does not name the execution rank");
        if (request.snapshot_capacity && !request.snapshot_capacity->valid())
            throw std::invalid_argument("Memory BOM snapshot policy has no capacity");
        const auto &profile = request.model;
        const auto &ngram=request.plan.runtime.ngram_table_policy;
        const auto ngram_topology=immutableRowTableTPTopology(profile,request.plan);
        /** @brief Physical observation used to fill a typed DevicePlanConfig. */
        struct DevicePlanningInventory
        {
            size_t total_bytes = 0;
            size_t free_bytes = 0;
            int compute_units = 0;
        };
        const auto inventoryForDevice = [&](DeviceId device)
        {
            const auto &rank = request.inventory;
            const auto observed = [&](size_t total, size_t available, int units)
            {
                // Validate the raw observation before a smaller explicit cap
                // could conceal corrupt geometry. PMA owns resource validity.
                if (!PhysicalMemoryResource{.world_rank = rank.rank, .device = device,
                        .total_bytes = total, .admission_available_bytes = available}.valid())
                    throw std::invalid_argument("Memory BOM has invalid observed capacity for " + device.toString());
                return DevicePlanningInventory{total, available, units};
            };
            if (device.is_cpu())
                return observed(
                    rank.cpu.memory_bytes > 0 ? rank.cpu.memory_bytes : rank.cpu_memory_bytes,
                    rank.cpu.free_memory_bytes, rank.cpu_worker_threads);
            for (const auto &gpu : rank.gpus)
                if (gpu.type == device.type && gpu.local_device_id == device.ordinal)
                    return observed(gpu.memory_bytes, gpu.free_memory_bytes, gpu.compute_units);
            throw std::invalid_argument("Memory BOM device absent from canonical rank inventory: " + device.toString());
        };

        std::vector<DevicePlanConfig> device_configs;
        auto makeConfigForDevice = [&](DeviceId device, int shard_index, int total_shards)
        {
            const DevicePlanningInventory inventory =
                inventoryForDevice(device);
            const int current_rank = request.plan.rank;

            DevicePlanConfig cfg;
            cfg.world_rank = current_rank;
            cfg.device = device;
            // Attention resolves below after the exact native KV/request geometry is known.
            cfg.qfn_hc_up_activation = profile.flash_next
                ? debugEnv().gemm.qfn_hc_up_activation
                : QuantizedActivationOperandPrecision::SingleQ8;
            cfg.qfn_gdn_output_activation = profile.flash_next
                ? debugEnv().gemm.qfn_gdn_output_activation
                : QuantizedActivationOperandPrecision::SingleQ8;
            cfg.qfn_fa_output_activation = profile.flash_next
                ? debugEnv().gemm.qfn_fa_output_activation
                : QuantizedActivationOperandPrecision::SingleQ8;
            cfg.qfn_gdn_qkv_activation = profile.flash_next
                ? debugEnv().gemm.qfn_gdn_qkv_activation
                : QuantizedActivationOperandPrecision::SingleQ8;
            cfg.device_total_bytes = inventory.total_bytes;
            {
                const auto observed = withObservedBacking(
                    {.world_rank = current_rank, .device = device, .total_bytes = inventory.total_bytes,
                     .admission_available_bytes = inventory.free_bytes}, request.inventory);
                cfg.device_memory_backing = observed.backing;
                cfg.device_pool_available_bytes = observed.pool_available_bytes;
            }
            cfg.device_free_bytes = PhysicalMemoryAuthority::admissionCapacity(inventory.free_bytes,
                device.is_cpu() ? request.max_cpu_memory_bytes : request.max_gpu_memory_bytes);
            cfg.device_compute_units = device.is_cpu()
                ? request.inventory.cpuWorkerThreads() : inventory.compute_units;
            if (device.is_cpu()) cfg.cpu_execution = request.inventory.cpu_execution;
            if (device.is_gpu() && request.snapshot_capacity.has_value())
            {
                cfg.graph_snapshot_memory =
                    *request.snapshot_capacity;
            }
            cfg.shard_index = shard_index;
            cfg.total_shards = total_shards;
            cfg.first_layer = request.plan.first_layer;
            cfg.last_layer = ngram.isFixedHeadSharded()?profile.mainLayerCount()-1:request.plan.last_layer;
            cfg.owns_embedding = request.plan.has_embedding;
            cfg.batch_size = request.plan.runtime.batch_size;
            cfg.ordinary_terminal_projection = request.plan.runtime.ordinary_terminal_projection;
            cfg.generation_request_capacity =
                std::max(1, request.plan.runtime.mtp.max_request_batch);
            cfg.max_seq_len = request.plan.runtime.max_seq_len;
            cfg.activation_seq_len = resolveActivationBufferSeqLen(cfg.max_seq_len, device);
            /* Memory planning describes resident capacity, not whether this
             * request executes MTP. An MTP-off request may deliberately retain
             * the same sidecar/checkpoint envelope as a later enabled request. */
            cfg.mtp_enabled =
                retainsMTPGraphCapacity(request.plan.runtime.mtp);
            cfg.mtp_shifted_kv_head_layout =
                resolveMTPShiftedKVHeadLayout(
                    request.plan.runtime.mtp,
                    /*dense_tensor_parallel=*/total_shards > 1,
                    total_shards);
            cfg.mtp_target_query_rows =
                cfg.mtp_enabled
                    ? resolveMTPRetainedTargetQueryRows(
                          request.plan.runtime.mtp)
                    : 1;
            cfg.mtp_terminal_logits_layout =
                resolveMTPTerminalLogitsLayout(
                    total_shards > 1,
                    request.plan.runtime.mtp.terminal_head_policy);
            /*
             * Ordinary TP graphs use GraphConfig's canonical tensor-parallel
             * dense policy.  Resolve every concurrently retained auxiliary
             * weight view from that same typed policy and the request's MTP
             * policy before PhysicalMemoryAuthority publishes its immutable
             * admission. A resolved mirrored terminal head remains live for the
             * serial decode oracle even when this request disables MTP; CPU's
             * resolved vocabulary-sharded policy does not retain that mirror.
             */
            cfg.additional_weight_sets =
                resolveAdditionalPersistentWeightSets(
                    DenseParallelPolicy::TensorParallel,
                    total_shards,
                    request.plan.runtime.mtp);

            if (device.is_gpu() && request.weight_load_geometry)
            {
                cfg.weight_load_staging = {
                    .device_bytes =
                        request.weight_load_geometry->staging_bytes,
                    .host_bytes =
                        request.weight_load_geometry->host_staging_bytes,
                };
                cfg.ordinary_weight_load_geometry = request.weight_load_geometry;
            }



            cfg.kv_precision = activationPrecisionToString(
                resolveKVCacheStoragePrecision(
                    request.plan.runtime.kv_cache_precision,
                    device.is_cpu()));
            cfg.prefix_cache = request.plan.runtime.prefix_cache;
            const bool owns_host_prefix_tier =
                cfg.prefix_cache.enabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Device;
            if (device.is_gpu() &&
                (owns_host_prefix_tier ||
                 cfg.weight_load_staging.host_bytes != 0u || profile.flash_next.has_value()))
            {
                // DeviceId::cpu() is the rank-local allocator used by runtime
                // owners. NUMA affinity may be unbound or a sparse physical ID;
                // it does not create a different CPU allocator ordinal.
                const DevicePlanningInventory host =
                    inventoryForDevice(DeviceId::cpu());
                cfg.associated_host_memory = PhysicalMemoryResource{
                    .world_rank = current_rank,
                    .device = DeviceId::cpu(),
                    .total_bytes = host.total_bytes,
                    .admission_available_bytes = PhysicalMemoryAuthority::admissionCapacity(
                        host.free_bytes, request.max_cpu_memory_bytes),
                };
            }

            if (total_shards > 1 && profile.n_kv_heads > 0)
            {
                cfg.local_kv_heads = profile.n_kv_heads / total_shards;
                if (cfg.local_kv_heads < 1)
                    cfg.local_kv_heads = 1;
            }
            return cfg;
        };

        /**
         * Bind the same exact rank-local TP assignment used by graph and
         * weight construction. Memory admission must not independently divide
         * model dimensions because GQA KV replication and remainder shards
         * make that reconstruction lossy.
         */
        const auto bindRankLocalTPAssignment = [&profile,&request](
            DevicePlanConfig &cfg,
            const std::vector<GlobalDeviceAddress> &participants,
            const std::vector<float> &configured_weights)
        {
            if (participants.size() <= 1u)
                return;

            std::vector<DeviceId> devices;
            devices.reserve(participants.size());
            for (const auto &participant : participants)
                devices.push_back(participant.toLocalDeviceId());

            std::vector<float> weights = configured_weights;
            if (weights.empty())
                weights.assign(devices.size(), 1.0f);
            const bool complete_main=request.plan.usesLocalTP() && !request.plan.usesLocalPP() &&
                !request.plan.usesGlobalTP() && !request.plan.prev_rank && !request.plan.next_rank &&
                request.plan.first_layer==0 && (request.plan.last_layer<0 || request.plan.last_layer==profile.mainLayerCount()-1);
            std::optional<TensorParallelAssignmentDeclaration> source;
            if(profile.architecture=="qwen4exp")source=resolveProfileTPDeclaration(profile,
                complete_main?TensorParallelMainPlacement::FullMainRankLocal:TensorParallelMainPlacement::OtherDeclared,
                request.plan.runtime.mtp,DenseParallelPolicy::TensorParallel,devices);
            const auto assignment=source?source->partition(devices,weights,profile.d_ff,profile.vocab_size):
                TensorParallelConfig::proportionalSplit(devices,weights,profile.n_heads,profile.n_kv_heads,profile.d_ff,profile.vocab_size);
            cfg.bindTensorParallelAssignment(
                assignment.forRank(cfg.shard_index));
        };

        if (request.plan.usesLocalPP())
        {
            // LOCAL PP: each PP stage has its own layer range. Create per-device
            // configs with the correct layer boundaries for each stage.
            const auto &pp_devices = request.plan.local_pp_devices;
            const auto &boundaries = request.plan.local_pp_layer_boundaries;
            const auto &stage_tp = request.plan.local_pp_stage_tp_info;
            if (boundaries.size() != pp_devices.size() + 1 ||
                !std::is_sorted(boundaries.begin(), boundaries.end()) ||
                std::adjacent_find(boundaries.begin(), boundaries.end()) != boundaries.end())
                throw std::invalid_argument("Pipeline memory BOM requires one nonempty interval per stage");

            // Cross-backend channels attach to domain leaders, not every TP
            // member. Source order also assigns host mapping ownership, so the
            // same physical slots cannot be charged once by each endpoint.
            const auto bindTransfers = [&](DevicePlanConfig &cfg, const PipelineStageTransferMemory &transfer) {
                if (transfer.requiresHostMemory(cfg.device) && !cfg.associated_host_memory)
                {
                    const auto host = inventoryForDevice(DeviceId::cpu());
                    cfg.associated_host_memory = PhysicalMemoryResource{
                        .world_rank = request.plan.rank, .device = DeviceId::cpu(),
                        .total_bytes = host.total_bytes,
                        .admission_available_bytes = PhysicalMemoryAuthority::admissionCapacity(
                            host.free_bytes, request.max_cpu_memory_bytes)};
                }
                transfer.bind(cfg);
            };

            for (size_t stage = 0; stage < pp_devices.size(); ++stage)
            {
                const auto transfer = PipelineStageTransferMemory::forStage(request.plan, stage);
                int stage_first = boundaries[stage];
                int stage_last = boundaries[stage + 1] - 1;

                // Check if this PP stage has TP composition (multiple devices per stage)
                if (stage < stage_tp.size() && stage_tp[stage].devices.size() > 1)
                {
                    // PP+TP: each device in this stage gets the stage's layer range + TP shard
                    const auto &tp_info = stage_tp[stage];
                    int tp_degree = static_cast<int>(tp_info.devices.size());
                    std::vector<DeviceId> tp_devices;
                    tp_devices.reserve(tp_info.devices.size());
                    for (const auto &participant : tp_info.devices)
                        tp_devices.push_back(participant.toLocalDeviceId());
                    const CollectiveBackendType resolved_backend =
                        BackendSelector::resolve(
                            tp_info.tp_backend, tp_devices);
                    for (int tp_idx = 0; tp_idx < tp_degree; ++tp_idx)
                    {
                        auto cfg = makeConfigForDevice(
                            tp_info.devices[tp_idx].toLocalDeviceId(),
                            tp_idx, tp_degree);
                        bindRankLocalTPAssignment(
                            cfg,
                            tp_info.devices,
                            tp_info.tp_weights);
                        cfg.local_tp_backend = resolved_backend;
                        cfg.first_layer = stage_first;
                        cfg.last_layer = stage_last;
                        cfg.owns_embedding =
                            request.plan.has_embedding && stage_first == 0;
                        bindTransfers(cfg, transfer);
                        device_configs.push_back(cfg);
                    }
                }
                else
                {
                    // PP only: single device per stage with that stage's full layer range
                    auto cfg = makeConfigForDevice(
                        pp_devices[stage].toLocalDeviceId(), 0, 1);
                    cfg.first_layer = stage_first;
                    cfg.last_layer = stage_last;
                    cfg.owns_embedding =
                        request.plan.has_embedding && stage_first == 0;
                    bindTransfers(cfg, transfer);
                    device_configs.push_back(cfg);
                }
            }
        }
        else if (request.plan.usesLocalTP())
        {
            const int total_shards = static_cast<int>(request.plan.local_tp_devices.size());
            std::vector<DeviceId> tp_devices;
            tp_devices.reserve(request.plan.local_tp_devices.size());
            for (const auto &participant : request.plan.local_tp_devices)
                tp_devices.push_back(participant.toLocalDeviceId());
            const CollectiveBackendType resolved_backend =
                BackendSelector::resolve(
                    request.plan.local_tp_backend, tp_devices);
            device_configs.reserve(request.plan.local_tp_devices.size());
            for (int index = 0; index < total_shards; ++index)
            {
                auto cfg = makeConfigForDevice(
                    request.plan.local_tp_devices[static_cast<size_t>(index)].toLocalDeviceId(),
                    index,
                    total_shards);
                bindRankLocalTPAssignment(
                    cfg,
                    request.plan.local_tp_devices,
                    request.plan.local_tp_weights);
                cfg.local_tp_backend = resolved_backend;
                cfg.ngram_tp_topology=ngram_topology;
                device_configs.push_back(std::move(cfg));
            }
        }
        else
        {
            DeviceId device = DeviceAddressAdapter::toDeviceId(request.plan.primary_device);
            device_configs.push_back(makeConfigForDevice(
                device,
                request.plan.weight_shard.shard_index,
                request.plan.weight_shard.total_shards));
        }

        if (request.captured_prefill_buckets)
        {
            const std::vector<int> configured_prefill_buckets =
                normalizePrefillGraphBuckets(*request.captured_prefill_buckets);
            if (configured_prefill_buckets.empty())
            {
                throw std::invalid_argument(
                    "Native serving graph memory admission has no configured prefill bucket inventory");
            }

            /*
             * Ordinary GPU setup retains every admitted prefill bucket, one
             * history-bearing serial decode graph, and—when the model keeps
             * an MTP sidecar family—one restored-prefix decode bridge. Native
             * drivers allocate opaque storage for those executables outside
             * BufferArena, so publish the exact setup inventory into the same
             * DevicePlanConfig that prices tensors and workspaces. Overlay
             * capacity has its own cross-rank executable inventory and must
             * not be charged a second time here.
             */
            // Observe authored members, including remote peers of this rank's
            // domains. A mixed complete domain requires hosted transactions on
            // every GPU; an inventory device outside that domain is irrelevant.
            const auto generation_domain = declaredRankDeviceGenerationDomain(request.plan);
            const auto query_programs = profile.sparseQueryProgramCount();
            for (auto &cfg : device_configs)
            {
                if (!cfg.device.is_gpu())
                    continue;
                const auto family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(
                    declaredDeviceGenerationExecutionPolicy(cfg.device.type, generation_domain)));
                // Only actual local PP nonterminal stages own follower parents.
                // A partial standalone graph is not a pipeline follower.
                const auto parent_role = request.plan.usesLocalPP() &&
                    cfg.last_layer < profile.mainLayerCount() - 1
                    ? MTPGenerationParentRole::PipelineFollower
                    : MTPGenerationParentRole::Controller;
                const auto graph_inventory = resolveCapturedServingGraphMemoryInventory(
                    configured_prefill_buckets,
                    request.plan.runtime.mtp,
                    family,
                    parent_role,
                    generation_domain,
                    query_programs);
                cfg.captured_serving_graphs = graph_inventory;
                if (cfg.graph_snapshot_memory.valid())
                {
                    const auto arenas = ServingSnapshotArenaInventory::ordinary(
                        !graph_inventory.prefill_bucket_rows.empty(),
                        graph_inventory.mtp_graph_owners.retainsGraphCapacity(),
                        graph_inventory.mtp_graph_owners.sidecarGraphSlots(),
                        graph_inventory.mtp_graph_owners.retainsGraphCapacity()
                            ? graph_inventory.mtp_graph_owners.requestCapacity() : 0u,
                        graph_inventory.sparse_query_program_count);
                    cfg.graph_snapshot_memory.serving_arenas = arenas;
                    cfg.graph_snapshot_memory.retained_main_forward_arena_count = arenas.mainOwners();
                    cfg.graph_snapshot_memory.retained_forward_arena_count = arenas.allOwners();
                }
                if (cfg.graph_snapshot_memory.effective_kv)
                {
                    // A per-executable bound also covers dedicated decode and
                    // sidecar arenas. Shared prefill/verifier alternatives can
                    // consume less, but cannot exceed this declared inventory.
                    cfg.graph_snapshot_memory.effective_kv->retained_arena_count =
                        cfg.graph_snapshot_memory.retained_forward_arena_count;
                }
            }
        }
        if (profile.flash_next)
        {
            const auto layer = static_cast<int>(profile.flash_next->ple_decoder_layer);
            std::vector<ImmutableRowTableReaderPeerAccess> reader_peer_access;
            // Original native NCCL/RCCL owns homogeneous peer selection;
            // serialized peer flags are not an alternate shared-cache route.
            for (auto &cfg : device_configs)
                if (cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph &&
                    cfg.first_layer <= layer && (cfg.last_layer < 0 || cfg.last_layer >= layer))
                {
                    cfg.ngram_execution_policy = request.plan.runtime.ngram_table_policy;
                    cfg.ngram_reader_peer_access = reader_peer_access;
                    if (request.plan.runtime.ngram_table_policy.isGPUAdaptive() &&
                        !(request.plan.usesLocalTP() && request.plan.runtime.ngram_table_policy.gpu_adaptive.placement.mode ==
                            AdaptiveGPUCachePlacementMode::DeclaredRankLocal))
                        cfg.ngram_gpu_peer_access = ImmutableRowTableDirectedTransportPolicy::observe(
                            request.plan.runtime.ngram_table_policy.gpu_adaptive.placement, request.inventory);
                    cfg.ngram_lookup_schedule = request.plan.runtime.ngram_lookup_schedule;
                    cfg.ngram_host_placement = ImmutableRowTableHostPlacement::fromObservedNUMANode(request.plan.numa_node);
                    cfg.ngram_table_placement = request.plan.runtime.ngram_table_placement;
                }
        }
        // PP overrides final layer ranges after makeConfigForDevice; resolve only now.
        for (auto &input : device_configs){
            if (request.native_source_placement) input.native_source_placement=*request.native_source_placement;
            input.native_context_preparation=request.plan.runtime.native_context_preparation;
            input.retained_host_services=request.plan.runtime.retained_host_services;
            if(input.native_context_preparation){
                const auto& original=input.native_context_preparation->mandatoryBOM();
                if(input.native_context_preparation->memory()->worldRank()!=input.world_rank)
                    throw std::logic_error("Rank planning received a foreign native source authority");
                if(input.device.is_gpu()){
                    input.native_context_preparation->requireSelectedDevice(input.device);
                    const auto* resource=original.find({input.world_rank,input.device});
                    if(!resource)throw std::logic_error("Native source omitted original GPU observation");
                    input.adoptDeviceResource(resource->resource());
                }
                if(input.associated_host_memory){
                    const auto* host=original.find({input.world_rank,DeviceId::cpu()});
                    if(!host)throw std::logic_error("Native source omitted original CPU0 observation");
                    input.associated_host_memory=host->resource();
                }
            }
            resolveQwenSparseAttentionDeviceProgram(profile, input, request.plan.runtime);
            if(input.device.is_cpu() && request.native_weight_source_policy) {
                if(!request.native_source_placement)
                    throw std::invalid_argument("Fresh CPU source staging requires its original loader allocation placement");
                input.native_weight_load_staging=quoteCPUWeightSourceStaging(profile,input,*request.native_weight_source_policy,
                    request.weight_precision,*request.native_source_placement);
                input.weight_load_staging.host_bytes=input.native_weight_load_staging
                    ? input.native_weight_load_staging->physical_bytes:0;
            }
        }
        return device_configs;
    }
}
