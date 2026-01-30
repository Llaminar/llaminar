/**
 * @file ExecutionPlanBuilder.cpp
 * @brief Implementation of ExecutionPlanBuilder
 *
 * Translates OrchestrationConfig into per-rank RankExecutionPlans.
 *
 * Two modes of operation:
 * 1. Named domains: Uses --define-domain and --pp-stage for complex configs
 * 2. Simple TP/PP: Uses --tp, --pp, --device for straightforward cases
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ExecutionPlanBuilder.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <numeric>
#include <set>

namespace llaminar2
{

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<IExecutionPlanBuilder> createExecutionPlanBuilder()
    {
        return std::make_unique<ExecutionPlanBuilder>();
    }

    // =========================================================================
    // Public Methods
    // =========================================================================

    std::vector<RankExecutionPlan> ExecutionPlanBuilder::buildAllPlans(
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const ClusterInventory &cluster_inventory)
    {
        LOG_DEBUG("Building execution plans for " << cluster_inventory.world_size
                                                  << " ranks, model " << model_config.name);

        std::vector<RankExecutionPlan> plans;
        plans.reserve(cluster_inventory.world_size);

        if (config.usesNamedDomains())
        {
            LOG_DEBUG("Using named domains mode");
            auto domains = resolveDomains(config, cluster_inventory);
            auto pp_stages = resolvePPStages(config, model_config, domains, cluster_inventory);

            for (int rank = 0; rank < cluster_inventory.world_size; ++rank)
            {
                plans.push_back(buildPlanWithDomains(
                    rank, config, model_config, cluster_inventory, domains, pp_stages));
            }
        }
        else
        {
            LOG_DEBUG("Using simple TP/PP mode");
            for (int rank = 0; rank < cluster_inventory.world_size; ++rank)
            {
                plans.push_back(buildSimplePlan(
                    rank, config, model_config, cluster_inventory));
            }
        }

        return plans;
    }

    RankExecutionPlan ExecutionPlanBuilder::buildPlanForRank(
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const ClusterInventory &cluster_inventory,
        int rank)
    {
        auto all_plans = buildAllPlans(config, model_config, cluster_inventory);
        if (rank < 0 || rank >= static_cast<int>(all_plans.size()))
        {
            LOG_ERROR("Invalid rank " << rank << " (world_size=" << all_plans.size() << ")");
            return RankExecutionPlan{};
        }
        return all_plans[rank];
    }

    std::vector<std::string> ExecutionPlanBuilder::validateConfig(
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const ClusterInventory &cluster_inventory)
    {
        std::vector<std::string> errors;

        // Validate model config
        auto model_errors = model_config.validate();
        errors.insert(errors.end(), model_errors.begin(), model_errors.end());

        // Validate orchestration config
        auto config_errors = config.validate();
        errors.insert(errors.end(), config_errors.begin(), config_errors.end());

        // Validate cluster has ranks
        if (cluster_inventory.world_size <= 0)
        {
            errors.push_back("ClusterInventory has no ranks");
        }

        // Check PP degree doesn't exceed ranks
        if (config.pp_degree > cluster_inventory.world_size)
        {
            errors.push_back("PP degree (" + std::to_string(config.pp_degree) +
                             ") exceeds world_size (" +
                             std::to_string(cluster_inventory.world_size) + ")");
        }

        // Check PP degree doesn't exceed layers
        if (config.pp_degree > model_config.n_layers)
        {
            errors.push_back("PP degree (" + std::to_string(config.pp_degree) +
                             ") exceeds n_layers (" +
                             std::to_string(model_config.n_layers) + ")");
        }

        // Validate named domains if used
        if (config.usesNamedDomains())
        {
            // Check all PP stages reference defined domains
            std::set<std::string> domain_names;
            for (const auto &domain : config.domain_definitions)
            {
                domain_names.insert(domain.name);
            }
            for (const auto &stage : config.pp_stage_definitions)
            {
                if (domain_names.find(stage.domain_name) == domain_names.end())
                {
                    errors.push_back("PP stage " + std::to_string(stage.stage_id) +
                                     " references undefined domain '" + stage.domain_name + "'");
                }
            }

            // Check layer ranges don't overlap and cover all layers
            if (!config.pp_stage_definitions.empty())
            {
                std::vector<bool> layer_covered(model_config.n_layers, false);
                for (const auto &stage : config.pp_stage_definitions)
                {
                    for (int l = stage.first_layer; l <= stage.last_layer && l < model_config.n_layers; ++l)
                    {
                        if (layer_covered[l])
                        {
                            errors.push_back("Layer " + std::to_string(l) +
                                             " assigned to multiple PP stages");
                        }
                        layer_covered[l] = true;
                    }
                }
                for (int l = 0; l < model_config.n_layers; ++l)
                {
                    if (!layer_covered[l])
                    {
                        errors.push_back("Layer " + std::to_string(l) +
                                         " not assigned to any PP stage");
                    }
                }
            }
        }

        return errors;
    }

    // =========================================================================
    // Domain Resolution
    // =========================================================================

    std::vector<ResolvedDomain> ExecutionPlanBuilder::resolveDomains(
        const OrchestrationConfig &config,
        const ClusterInventory &cluster_inventory)
    {
        std::vector<ResolvedDomain> resolved;

        for (size_t i = 0; i < config.domain_definitions.size(); ++i)
        {
            const auto &def = config.domain_definitions[i];
            ResolvedDomain domain;
            domain.id = static_cast<int>(i);
            domain.name = def.name;
            domain.devices = def.devices;
            domain.weights = def.weights;
            domain.backend = def.backend;

            // Find ranks for each device
            std::set<int> rank_set;
            for (const auto &device : domain.devices)
            {
                int rank = findRankForDevice(device, cluster_inventory);
                if (rank >= 0)
                {
                    rank_set.insert(rank);
                }
                else
                {
                    LOG_WARN("Device " << device.toString()
                                       << " not found in cluster inventory");
                }
            }
            domain.ranks.assign(rank_set.begin(), rank_set.end());

            // Build rank-to-index mapping
            int idx = 0;
            for (int rank : domain.ranks)
            {
                domain.rank_to_index[rank] = idx++;
            }

            LOG_DEBUG("Resolved domain '" << domain.name << "' with "
                                          << domain.devices.size() << " devices on "
                                          << domain.ranks.size() << " ranks");

            resolved.push_back(std::move(domain));
        }

        return resolved;
    }

    ResolvedDomain ExecutionPlanBuilder::createImplicitDomain(
        const OrchestrationConfig &config,
        const ClusterInventory &cluster_inventory)
    {
        ResolvedDomain domain;
        domain.id = 0;
        domain.name = "default";
        domain.backend = config.default_backend;

        // Use explicit TP devices if provided
        if (!config.tp_devices.empty())
        {
            domain.devices = config.tp_devices;
            domain.weights = config.tp_weights;
        }
        else
        {
            // Auto-detect: use all GPUs in cluster, or CPUs if no GPUs
            for (const auto &rank_inv : cluster_inventory.ranks)
            {
                for (const auto &gpu : rank_inv.gpus)
                {
                    domain.devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                        DeviceId(gpu.type, gpu.local_device_id),
                        rank_inv.hostname,
                        gpu.numa_node >= 0 ? gpu.numa_node : 0));
                }
            }

            // If no GPUs, use CPUs
            if (domain.devices.empty())
            {
                for (const auto &rank_inv : cluster_inventory.ranks)
                {
                    domain.devices.push_back(GlobalDeviceAddress::cpu(
                        rank_inv.numa_nodes > 0 ? 0 : rank_inv.local_rank,
                        rank_inv.hostname));
                }
            }
        }

        // Find participating ranks
        std::set<int> rank_set;
        for (const auto &device : domain.devices)
        {
            int rank = findRankForDevice(device, cluster_inventory);
            if (rank >= 0)
            {
                rank_set.insert(rank);
            }
        }
        domain.ranks.assign(rank_set.begin(), rank_set.end());

        // Build rank-to-index mapping
        int idx = 0;
        for (int rank : domain.ranks)
        {
            domain.rank_to_index[rank] = idx++;
        }

        return domain;
    }

    int ExecutionPlanBuilder::findRankForDevice(
        const GlobalDeviceAddress &device,
        const ClusterInventory &cluster_inventory)
    {
        for (const auto &rank_inv : cluster_inventory.ranks)
        {
            // Match hostname
            if (!device.isLocal() && rank_inv.hostname != device.hostname)
            {
                continue;
            }

            // For CPU devices, match by NUMA node
            if (device.isCPU())
            {
                // CPU device is typically on the rank's NUMA node
                // Simple heuristic: rank N owns CPU on NUMA node rank % numa_nodes
                if (device.numa_node == rank_inv.local_rank % std::max(1, rank_inv.numa_nodes))
                {
                    return rank_inv.rank;
                }
            }
            else
            {
                // For GPU devices, match by type and ordinal
                for (const auto &gpu : rank_inv.gpus)
                {
                    if (gpu.type == device.device_type &&
                        gpu.local_device_id == device.device_ordinal)
                    {
                        // Check NUMA if specified
                        if (gpu.numa_node < 0 || gpu.numa_node == device.numa_node)
                        {
                            return rank_inv.rank;
                        }
                    }
                }
            }
        }
        return -1; // Not found
    }

    // =========================================================================
    // PP Stage Resolution
    // =========================================================================

    std::vector<ResolvedPPStage> ExecutionPlanBuilder::resolvePPStages(
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const std::vector<ResolvedDomain> &domains,
        const ClusterInventory &cluster_inventory)
    {
        std::vector<ResolvedPPStage> stages;

        if (!config.pp_stage_definitions.empty())
        {
            // Use explicit PP stage definitions
            for (const auto &def : config.pp_stage_definitions)
            {
                ResolvedPPStage stage;
                stage.stage_id = def.stage_id;
                stage.domain_name = def.domain_name;
                stage.first_layer = def.first_layer;
                stage.last_layer = def.last_layer;

                // Find domain by name
                for (const auto &domain : domains)
                {
                    if (domain.name == def.domain_name)
                    {
                        stage.domain_id = domain.id;
                        stage.ranks = domain.ranks;
                        break;
                    }
                }

                stages.push_back(std::move(stage));
            }

            // Sort by stage_id
            std::sort(stages.begin(), stages.end(),
                      [](const ResolvedPPStage &a, const ResolvedPPStage &b)
                      { return a.stage_id < b.stage_id; });
        }
        else if (config.pp_degree > 1)
        {
            // Create simple PP stages with equal layer split
            stages = createSimplePPStages(
                config.pp_degree, model_config.n_layers, cluster_inventory);
        }
        else
        {
            // Single stage covering all layers
            ResolvedPPStage stage;
            stage.stage_id = 0;
            stage.domain_id = domains.empty() ? -1 : 0;
            stage.domain_name = domains.empty() ? "default" : domains[0].name;
            stage.first_layer = 0;
            stage.last_layer = model_config.n_layers - 1;
            for (int r = 0; r < cluster_inventory.world_size; ++r)
            {
                stage.ranks.push_back(r);
            }
            stages.push_back(std::move(stage));
        }

        return stages;
    }

    std::vector<ResolvedPPStage> ExecutionPlanBuilder::createSimplePPStages(
        int pp_degree,
        int n_layers,
        const ClusterInventory &cluster_inventory)
    {
        std::vector<ResolvedPPStage> stages;

        int layers_per_stage = n_layers / pp_degree;
        int remainder = n_layers % pp_degree;
        int ranks_per_stage = cluster_inventory.world_size / pp_degree;

        int layer = 0;
        int rank = 0;
        for (int s = 0; s < pp_degree; ++s)
        {
            ResolvedPPStage stage;
            stage.stage_id = s;
            stage.domain_id = s;
            stage.domain_name = "stage_" + std::to_string(s);
            stage.first_layer = layer;

            // Distribute extra layers to first stages
            int stage_layers = layers_per_stage + (s < remainder ? 1 : 0);
            stage.last_layer = layer + stage_layers - 1;
            layer += stage_layers;

            // Assign ranks to this stage
            for (int i = 0; i < ranks_per_stage && rank < cluster_inventory.world_size; ++i)
            {
                stage.ranks.push_back(rank++);
            }

            stages.push_back(std::move(stage));
        }

        // Distribute any remaining ranks to last stage
        while (rank < cluster_inventory.world_size)
        {
            if (!stages.empty())
            {
                stages.back().ranks.push_back(rank++);
            }
            else
            {
                ++rank;
            }
        }

        return stages;
    }

    // =========================================================================
    // Weight Sharding
    // =========================================================================

    WeightShardInfo ExecutionPlanBuilder::calculateWeightShard(
        int rank,
        const ResolvedDomain *primary_domain,
        const OrchestrationConfig &config)
    {
        WeightShardInfo shard;

        if (!primary_domain || primary_domain->ranks.size() <= 1)
        {
            // No TP sharding - this rank loads full weights
            shard.shard_index = 0;
            shard.total_shards = 1;
            shard.work_fraction = 1.0f;
        }
        else
        {
            // Find this rank's position in the domain
            auto it = primary_domain->rank_to_index.find(rank);
            if (it != primary_domain->rank_to_index.end())
            {
                shard.shard_index = it->second;
                shard.total_shards = static_cast<int>(primary_domain->ranks.size());

                // Calculate work fraction
                if (!primary_domain->weights.empty() &&
                    shard.shard_index < static_cast<int>(primary_domain->weights.size()))
                {
                    shard.work_fraction = primary_domain->weights[shard.shard_index];
                }
                else
                {
                    shard.work_fraction = 1.0f / shard.total_shards;
                }
            }
            else
            {
                // Rank not in domain - shouldn't happen
                LOG_WARN("Rank " << rank << " not found in primary domain");
                shard.shard_index = 0;
                shard.total_shards = 1;
                shard.work_fraction = 1.0f;
            }
        }

        return shard;
    }

    // =========================================================================
    // Plan Building
    // =========================================================================

    RankExecutionPlan ExecutionPlanBuilder::buildPlanWithDomains(
        int rank,
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const ClusterInventory &cluster_inventory,
        const std::vector<ResolvedDomain> &domains,
        const std::vector<ResolvedPPStage> &pp_stages)
    {
        RankExecutionPlan plan;

        // Identity
        plan.rank = rank;
        if (rank < static_cast<int>(cluster_inventory.ranks.size()))
        {
            const auto &rank_inv = cluster_inventory.ranks[rank];
            plan.hostname = rank_inv.hostname;
            plan.numa_node = rank_inv.local_rank; // Approximate
        }

        // Find which PP stage this rank belongs to
        const ResolvedPPStage *my_stage = nullptr;
        for (const auto &stage : pp_stages)
        {
            for (int r : stage.ranks)
            {
                if (r == rank)
                {
                    my_stage = &stage;
                    break;
                }
            }
            if (my_stage)
                break;
        }

        if (my_stage)
        {
            plan.pp_stage_id = my_stage->stage_id;
            plan.first_layer = my_stage->first_layer;
            plan.last_layer = my_stage->last_layer;

            // First stage has embedding
            plan.has_embedding = (my_stage->stage_id == 0);

            // Last stage has LM head
            bool is_last = true;
            for (const auto &stage : pp_stages)
            {
                if (stage.stage_id > my_stage->stage_id)
                {
                    is_last = false;
                    break;
                }
            }
            plan.has_lm_head = is_last;

            // PP neighbors
            for (const auto &stage : pp_stages)
            {
                if (stage.stage_id == my_stage->stage_id - 1 && !stage.ranks.empty())
                {
                    plan.prev_rank = stage.ranks[0]; // First rank of previous stage
                }
                if (stage.stage_id == my_stage->stage_id + 1 && !stage.ranks.empty())
                {
                    plan.next_rank = stage.ranks[0]; // First rank of next stage
                }
            }
        }
        else
        {
            // Rank not in any stage - give it all layers (shouldn't happen)
            plan.first_layer = 0;
            plan.last_layer = model_config.n_layers - 1;
            plan.has_embedding = true;
            plan.has_lm_head = true;
        }

        // Find domains this rank participates in
        const ResolvedDomain *primary_domain = nullptr;
        for (const auto &domain : domains)
        {
            auto it = domain.rank_to_index.find(rank);
            if (it != domain.rank_to_index.end())
            {
                TPDomainParticipation participation;
                participation.domain_id = domain.id;
                participation.domain_name = domain.name;
                participation.devices = domain.devices;
                participation.weights = domain.weights;
                participation.backend = domain.backend;
                participation.my_index_in_domain = it->second;
                plan.my_domains.push_back(std::move(participation));

                if (!primary_domain)
                {
                    primary_domain = &domain;
                }
            }
        }

        // Set up local TP devices (devices within this rank)
        if (primary_domain)
        {
            // Collect devices on this rank's host
            for (const auto &device : primary_domain->devices)
            {
                bool on_this_rank = false;
                if (rank < static_cast<int>(cluster_inventory.ranks.size()))
                {
                    const auto &rank_inv = cluster_inventory.ranks[rank];
                    if (device.isLocal() || device.hostname == rank_inv.hostname)
                    {
                        on_this_rank = true;
                    }
                }
                if (on_this_rank)
                {
                    plan.local_tp_devices.push_back(device);
                }
            }

            plan.local_tp_backend = selectBackend(plan.local_tp_devices);
        }

        // Set primary device
        if (!plan.local_tp_devices.empty())
        {
            plan.primary_device = plan.local_tp_devices[0];
        }
        else if (rank < static_cast<int>(cluster_inventory.ranks.size()))
        {
            const auto &rank_inv = cluster_inventory.ranks[rank];
            if (!rank_inv.gpus.empty())
            {
                const auto &gpu = rank_inv.gpus[0];
                plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId(gpu.type, gpu.local_device_id),
                    rank_inv.hostname,
                    gpu.numa_node >= 0 ? gpu.numa_node : 0);
            }
            else
            {
                plan.primary_device = GlobalDeviceAddress::cpu(0, rank_inv.hostname);
            }
        }

        // Global TP configuration
        if (primary_domain && primary_domain->ranks.size() > 1)
        {
            plan.global_tp_domain_id = primary_domain->id;
            auto it = primary_domain->rank_to_index.find(rank);
            plan.global_tp_rank_in_domain = (it != primary_domain->rank_to_index.end()) ? it->second : 0;
            plan.global_tp_domain_size = static_cast<int>(primary_domain->ranks.size());
        }

        // Weight sharding
        plan.weight_shard = calculateWeightShard(rank, primary_domain, config);

        // TP scope
        if (plan.usesLocalTP() && plan.usesGlobalTP())
        {
            plan.tp_scope = TPScope::HYBRID;
        }
        else if (plan.usesGlobalTP())
        {
            plan.tp_scope = TPScope::GLOBAL;
        }
        else if (plan.usesLocalTP())
        {
            plan.tp_scope = TPScope::LOCAL;
        }
        else
        {
            plan.tp_scope = TPScope::AUTO;
        }

        // Cross-rank backend
        plan.cross_rank_backend = selectCrossRankBackend(cluster_inventory);

        return plan;
    }

    RankExecutionPlan ExecutionPlanBuilder::buildSimplePlan(
        int rank,
        const OrchestrationConfig &config,
        const ModelConfig &model_config,
        const ClusterInventory &cluster_inventory)
    {
        RankExecutionPlan plan;

        // Identity
        plan.rank = rank;
        if (rank < static_cast<int>(cluster_inventory.ranks.size()))
        {
            const auto &rank_inv = cluster_inventory.ranks[rank];
            plan.hostname = rank_inv.hostname;
            plan.numa_node = rank_inv.local_rank;
        }

        // PP configuration
        int pp_degree = std::max(1, config.pp_degree);
        int ranks_per_stage = cluster_inventory.world_size / pp_degree;
        if (ranks_per_stage < 1)
            ranks_per_stage = 1;

        int stage_id = rank / ranks_per_stage;
        if (stage_id >= pp_degree)
            stage_id = pp_degree - 1;

        plan.pp_stage_id = stage_id;

        // Layer assignment
        int layers_per_stage = model_config.n_layers / pp_degree;
        int remainder = model_config.n_layers % pp_degree;

        plan.first_layer = 0;
        for (int s = 0; s < stage_id; ++s)
        {
            plan.first_layer += layers_per_stage + (s < remainder ? 1 : 0);
        }
        int stage_layers = layers_per_stage + (stage_id < remainder ? 1 : 0);
        plan.last_layer = plan.first_layer + stage_layers - 1;

        // Embedding and LM head
        plan.has_embedding = (stage_id == 0);
        plan.has_lm_head = (stage_id == pp_degree - 1);

        // PP neighbors
        if (stage_id > 0)
        {
            plan.prev_rank = (stage_id - 1) * ranks_per_stage;
        }
        if (stage_id < pp_degree - 1)
        {
            plan.next_rank = (stage_id + 1) * ranks_per_stage;
        }

        // Local TP devices
        if (!config.tp_devices.empty())
        {
            plan.local_tp_devices = config.tp_devices;
            plan.local_tp_weights = config.tp_weights;
        }
        else if (config.tp_degree > 1)
        {
            // Use GPUs on this rank if available
            if (rank < static_cast<int>(cluster_inventory.ranks.size()))
            {
                const auto &rank_inv = cluster_inventory.ranks[rank];
                for (int i = 0; i < std::min(config.tp_degree, static_cast<int>(rank_inv.gpus.size())); ++i)
                {
                    const auto &gpu = rank_inv.gpus[i];
                    plan.local_tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                        DeviceId(gpu.type, gpu.local_device_id),
                        rank_inv.hostname,
                        gpu.numa_node >= 0 ? gpu.numa_node : 0));
                }
            }
        }

        // Primary device
        if (!plan.local_tp_devices.empty())
        {
            plan.primary_device = plan.local_tp_devices[0];
        }
        else if (config.device_for_this_rank.has_value())
        {
            plan.primary_device = *config.device_for_this_rank;
        }
        else if (rank < static_cast<int>(cluster_inventory.ranks.size()))
        {
            const auto &rank_inv = cluster_inventory.ranks[rank];
            if (!rank_inv.gpus.empty())
            {
                const auto &gpu = rank_inv.gpus[0];
                plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId(gpu.type, gpu.local_device_id),
                    rank_inv.hostname,
                    gpu.numa_node >= 0 ? gpu.numa_node : 0);
            }
            else
            {
                plan.primary_device = GlobalDeviceAddress::cpu(0, rank_inv.hostname);
            }
        }

        // TP scope
        plan.tp_scope = config.tp_scope;
        if (plan.tp_scope == TPScope::AUTO)
        {
            if (plan.usesLocalTP())
            {
                plan.tp_scope = TPScope::LOCAL;
            }
            else if (cluster_inventory.world_size > 1 && config.tp_degree > 1)
            {
                plan.tp_scope = TPScope::GLOBAL;
            }
        }

        // Global TP for multi-rank TP
        if (config.tp_scope == TPScope::GLOBAL && cluster_inventory.world_size > 1)
        {
            plan.global_tp_domain_id = 0;
            plan.global_tp_rank_in_domain = rank % config.tp_degree;
            plan.global_tp_domain_size = std::min(config.tp_degree, cluster_inventory.world_size);
        }

        // Weight sharding
        int total_shards = 1;
        int shard_index = 0;

        if (plan.usesLocalTP())
        {
            total_shards = static_cast<int>(plan.local_tp_devices.size());
            shard_index = 0; // For local TP, this rank manages all local shards
        }
        else if (plan.usesGlobalTP())
        {
            total_shards = plan.global_tp_domain_size;
            shard_index = plan.global_tp_rank_in_domain;
        }

        plan.weight_shard.total_shards = total_shards;
        plan.weight_shard.shard_index = shard_index;
        plan.weight_shard.work_fraction = 1.0f / total_shards;

        // Local TP backend
        plan.local_tp_backend = selectBackend(plan.local_tp_devices);

        // Cross-rank backend
        plan.cross_rank_backend = selectCrossRankBackend(cluster_inventory);

        return plan;
    }

    // =========================================================================
    // Backend Selection
    // =========================================================================

    CollectiveBackendType ExecutionPlanBuilder::selectBackend(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        if (devices.size() <= 1)
        {
            return CollectiveBackendType::AUTO;
        }

        // Check for heterogeneous GPU types
        bool has_cuda = false;
        bool has_rocm = false;
        bool has_cpu = false;
        bool same_numa = true;
        bool same_host = true;

        int first_numa = -1;
        std::string first_host;

        for (const auto &dev : devices)
        {
            if (dev.isCUDA())
                has_cuda = true;
            if (dev.isROCm())
                has_rocm = true;
            if (dev.isCPU())
                has_cpu = true;

            if (first_numa < 0)
            {
                first_numa = dev.numa_node;
                first_host = dev.hostname;
            }
            else
            {
                if (dev.numa_node != first_numa)
                    same_numa = false;
                if (dev.hostname != first_host)
                    same_host = false;
            }
        }

        // Cross-host requires MPI
        if (!same_host)
        {
            return CollectiveBackendType::MPI;
        }

        // Heterogeneous GPUs on same host use PCIeBAR
        if (has_cuda && has_rocm)
        {
            return CollectiveBackendType::PCIE_BAR;
        }

        // GPU + CPU mix uses HOST staging
        if ((has_cuda || has_rocm) && has_cpu)
        {
            return CollectiveBackendType::HOST;
        }

        // Homogeneous CUDA uses NCCL
        if (has_cuda && !has_rocm && !has_cpu)
        {
            return CollectiveBackendType::NCCL;
        }

        // Homogeneous ROCm uses RCCL
        if (has_rocm && !has_cuda && !has_cpu)
        {
            return CollectiveBackendType::RCCL;
        }

        // Cross-NUMA CPUs use UPI
        if (has_cpu && !same_numa)
        {
            return CollectiveBackendType::UPI;
        }

        return CollectiveBackendType::AUTO;
    }

    CollectiveBackendType ExecutionPlanBuilder::selectCrossRankBackend(
        const ClusterInventory &cluster_inventory)
    {
        if (cluster_inventory.world_size <= 1)
        {
            return CollectiveBackendType::AUTO;
        }

        // Check if all ranks are on same host
        bool same_host = true;
        std::string first_host;
        for (const auto &rank_inv : cluster_inventory.ranks)
        {
            if (first_host.empty())
            {
                first_host = rank_inv.hostname;
            }
            else if (rank_inv.hostname != first_host)
            {
                same_host = false;
                break;
            }
        }

        // Cross-host always uses MPI
        if (!same_host)
        {
            return CollectiveBackendType::MPI;
        }

        // Same-host with GPUs could use NCCL/RCCL
        // But for cross-rank, MPI is typically used
        return CollectiveBackendType::MPI;
    }

} // namespace llaminar2
