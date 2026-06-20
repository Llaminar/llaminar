/**
 * @file HeterogeneousMultiDomainStrategy.cpp
 * @brief Implementation of heterogeneous multi-domain placement strategy
 * @author David Sanftenberg
 * @date January 2026
 */

#include "HeterogeneousMultiDomainStrategy.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaminar2
{

    // =============================================================================
    // DomainAssignment Implementation
    // =============================================================================

    std::string DomainAssignment::toString() const
    {
        std::ostringstream oss;
        oss << "Domain[" << domain_id << "] "
            << (type == TPDomainType::GPU_INTRA_RANK ? "GPU_TP" : "CPU_TP")
            << " node=" << node_id
            << " ranks=[";
        for (size_t i = 0; i < ranks.size(); ++i)
        {
            if (i > 0)
                oss << ",";
            oss << ranks[i];
        }
        oss << "] devices=[";
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (i > 0)
                oss << ",";
            oss << devices[i].toString();
        }
        oss << "] layers=[" << layer_start << "," << layer_end << ")"
            << " weight=" << compute_weight;
        return oss.str();
    }

    // =============================================================================
    // HeterogeneousPlan Implementation
    // =============================================================================

    const DomainAssignment *HeterogeneousPlan::getDomainForLayer(int layer_idx) const
    {
        for (const auto &domain : domains)
        {
            if (layer_idx >= domain.layer_start && layer_idx < domain.layer_end)
            {
                return &domain;
            }
        }
        return nullptr;
    }

    const PipelineStage *HeterogeneousPlan::getStageForLayer(int layer_idx) const
    {
        for (const auto &stage : stages)
        {
            if (layer_idx >= stage.layer_start && layer_idx < stage.layer_end)
            {
                return &stage;
            }
        }
        return nullptr;
    }

    std::vector<uint8_t> HeterogeneousPlan::serialize() const
    {
        // Simple binary serialization format:
        // [4B total_layers][4B world_size][4B node_count]
        // [4B num_stages][stages...]
        // [4B num_domains][domains...]

        std::vector<uint8_t> data;

        auto writeInt = [&data](int32_t val)
        {
            data.insert(data.end(),
                        reinterpret_cast<uint8_t *>(&val),
                        reinterpret_cast<uint8_t *>(&val) + sizeof(val));
        };

        auto writeFloat = [&data](float val)
        {
            data.insert(data.end(),
                        reinterpret_cast<uint8_t *>(&val),
                        reinterpret_cast<uint8_t *>(&val) + sizeof(val));
        };

        auto writeVector = [&](const std::vector<int> &vec)
        {
            writeInt(static_cast<int32_t>(vec.size()));
            for (int v : vec)
                writeInt(v);
        };

        // Header
        writeInt(total_layers);
        writeInt(world_size);
        writeInt(node_count);

        // Stages
        writeInt(static_cast<int32_t>(stages.size()));
        for (const auto &stage : stages)
        {
            writeInt(stage.node_id);
            writeInt(stage.stage_id);
            writeVector(stage.ranks);
            writeInt(stage.layer_start);
            writeInt(stage.layer_end);
            // Note: stage.domains indices are reconstructed from flat domains list
        }

        // Domains
        writeInt(static_cast<int32_t>(domains.size()));
        for (const auto &domain : domains)
        {
            writeInt(domain.domain_id);
            writeInt(static_cast<int32_t>(domain.type));
            writeVector(domain.ranks);
            writeInt(static_cast<int32_t>(domain.devices.size()));
            for (const auto &dev : domain.devices)
            {
                writeInt(static_cast<int32_t>(dev.type));
                writeInt(dev.ordinal);
            }
            writeInt(domain.node_id);
            writeInt(domain.layer_start);
            writeInt(domain.layer_end);
            writeFloat(domain.compute_weight);
        }

        return data;
    }

    HeterogeneousPlan HeterogeneousPlan::deserialize(const std::vector<uint8_t> &data)
    {
        HeterogeneousPlan plan;
        size_t offset = 0;

        auto readInt = [&data, &offset]() -> int32_t
        {
            int32_t val;
            std::memcpy(&val, data.data() + offset, sizeof(val));
            offset += sizeof(val);
            return val;
        };

        auto readFloat = [&data, &offset]() -> float
        {
            float val;
            std::memcpy(&val, data.data() + offset, sizeof(val));
            offset += sizeof(val);
            return val;
        };

        auto readVector = [&]() -> std::vector<int>
        {
            int32_t size = readInt();
            std::vector<int> vec(size);
            for (int i = 0; i < size; ++i)
                vec[i] = readInt();
            return vec;
        };

        // Header
        plan.total_layers = readInt();
        plan.world_size = readInt();
        plan.node_count = readInt();

        // Stages
        int32_t num_stages = readInt();
        plan.stages.resize(num_stages);
        for (auto &stage : plan.stages)
        {
            stage.node_id = readInt();
            stage.stage_id = readInt();
            stage.ranks = readVector();
            stage.layer_start = readInt();
            stage.layer_end = readInt();
        }

        // Domains
        int32_t num_domains = readInt();
        plan.domains.resize(num_domains);
        for (auto &domain : plan.domains)
        {
            domain.domain_id = readInt();
            domain.type = static_cast<TPDomainType>(readInt());
            domain.ranks = readVector();
            int32_t num_devices = readInt();
            domain.devices.resize(num_devices);
            for (auto &dev : domain.devices)
            {
                auto dev_type = static_cast<DeviceType>(readInt());
                int dev_idx = readInt();
                if (dev_type == DeviceType::CPU)
                {
                    dev = DeviceId::cpu();
                }
                else if (dev_type == DeviceType::CUDA)
                {
                    dev = DeviceId::cuda(dev_idx);
                }
                else if (dev_type == DeviceType::ROCm)
                {
                    dev = DeviceId::rocm(dev_idx);
                }
                else
                {
                    dev = DeviceId::cpu(); // Fallback
                }
            }
            domain.node_id = readInt();
            domain.layer_start = readInt();
            domain.layer_end = readInt();
            domain.compute_weight = readFloat();
        }

        // Reconstruct stage->domain references
        for (auto &stage : plan.stages)
        {
            for (const auto &domain : plan.domains)
            {
                if (domain.node_id == stage.node_id)
                {
                    stage.domains.push_back(domain);
                }
            }
        }

        return plan;
    }

    std::string HeterogeneousPlan::toString() const
    {
        std::ostringstream oss;
        oss << "HeterogeneousPlan {\n"
            << "  total_layers: " << total_layers << "\n"
            << "  world_size: " << world_size << "\n"
            << "  node_count: " << node_count << "\n"
            << "  stages: " << stages.size() << "\n";

        for (const auto &stage : stages)
        {
            oss << "    Stage[" << stage.stage_id << "] node=" << stage.node_id
                << " layers=[" << stage.layer_start << "," << stage.layer_end << ")"
                << " ranks=[";
            for (size_t i = 0; i < stage.ranks.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << stage.ranks[i];
            }
            oss << "] domains=" << stage.domains.size() << "\n";
        }

        oss << "  domains: " << domains.size() << "\n";
        for (const auto &domain : domains)
        {
            oss << "    " << domain.toString() << "\n";
        }

        oss << "}";
        return oss.str();
    }

    // =============================================================================
    // HeterogeneousMultiDomainStrategy Implementation
    // =============================================================================

    HeterogeneousMultiDomainStrategy::HeterogeneousMultiDomainStrategy(HeterogeneousConfig config)
        : config_(std::move(config))
    {
    }

    bool HeterogeneousMultiDomainStrategy::isApplicable(const PlacementInput &input) const
    {
        // Applicable when:
        // 1. Not forced CPU-only
        // 2. Either has multiple nodes OR has mixed GPU types OR has GPUs+CPUs

        if (input.force_cpu_only)
        {
            return false;
        }

        // Multi-node setup benefits from PP
        if (input.node_count > 1)
        {
            return true;
        }

        // Check for heterogeneous GPUs (NVIDIA + AMD on same rank)
        for (const auto &rank_inv : input.cluster_inventory.ranks)
        {
            auto [has_nvidia, has_amd] = detectGPUTypes(rank_inv);
            if (has_nvidia && has_amd)
            {
                return true;
            }
        }

        // Check for GPU + CPU hybrid potential (GPU for prefill, CPU for decode)
        if (input.any_rank_has_gpu && config_.enable_cpu_tp)
        {
            return true;
        }

        // Single-node homogeneous GPU: other strategies may be more appropriate
        return false;
    }

    std::pair<bool, bool> HeterogeneousMultiDomainStrategy::detectGPUTypes(
        const RankInventory &rank_inv) const
    {
        bool has_nvidia = false;
        bool has_amd = false;

        for (const auto &gpu : rank_inv.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
            {
                has_nvidia = true;
            }
            else if (gpu.type == DeviceType::ROCm)
            {
                has_amd = true;
            }
        }

        return {has_nvidia, has_amd};
    }

    float HeterogeneousMultiDomainStrategy::calculateGPUComputeWeight(
        const RankInventory &rank_inv) const
    {
        float weight = 0.0f;

        for (const auto &gpu : rank_inv.gpus)
        {
            // Use INT8 TOPS if available, otherwise FP16 TFLOPS
            if (gpu.tflops_int8 > 0)
            {
                weight += gpu.tflops_int8;
            }
            else if (gpu.tflops_fp16 > 0)
            {
                weight += gpu.tflops_fp16;
            }
            else
            {
                // Estimate from compute units
                // Rough heuristic: ~0.1 TFLOPS per SM/CU
                weight += gpu.compute_units * 0.1f;
            }
        }

        return weight;
    }

    float HeterogeneousMultiDomainStrategy::calculateCPUComputeWeight(
        const NodeInventory &node_inv,
        const ClusterInventory &cluster_inv) const
    {
        float weight = 0.0f;

        for (int rank : node_inv.ranks)
        {
            if (rank < 0 || rank >= static_cast<int>(cluster_inv.ranks.size()))
            {
                continue;
            }

            const auto &rank_inv = cluster_inv.ranks[rank];

            // Use CPU compute info if available
            if (rank_inv.cpu.tflops_fp16 > 0)
            {
                weight += rank_inv.cpu.tflops_fp16;
            }
            else if (rank_inv.cpu.tflops_int8 > 0)
            {
                weight += rank_inv.cpu.tflops_int8;
            }
            else
            {
                // Estimate from cores: ~0.01 TFLOPS per core for AVX-512
                weight += rank_inv.cpu_cores * 0.01f;
            }
        }

        return weight;
    }

    DomainAssignment HeterogeneousMultiDomainStrategy::createGPUDomain(
        const RankInventory &rank_inv,
        int domain_id) const
    {
        DomainAssignment domain;
        domain.domain_id = domain_id;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.ranks = {rank_inv.rank};
        domain.node_id = rank_inv.node_id;

        for (const auto &gpu : rank_inv.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
            {
                domain.devices.push_back(DeviceId::cuda(gpu.local_device_id));
            }
            else if (gpu.type == DeviceType::ROCm)
            {
                domain.devices.push_back(DeviceId::rocm(gpu.local_device_id));
            }
        }

        domain.compute_weight = calculateGPUComputeWeight(rank_inv);

        return domain;
    }

    DomainAssignment HeterogeneousMultiDomainStrategy::createCPUDomain(
        const NodeInventory &node_inv,
        const ClusterInventory &cluster_inv,
        int domain_id) const
    {
        DomainAssignment domain;
        domain.domain_id = domain_id;
        domain.type = TPDomainType::CPU_CROSS_RANK;
        domain.ranks = node_inv.ranks;
        domain.node_id = node_inv.node_id;

        // Add CPU device for each rank
        for (int rank : node_inv.ranks)
        {
            domain.devices.push_back(DeviceId::cpu());
        }

        domain.compute_weight = calculateCPUComputeWeight(node_inv, cluster_inv);

        return domain;
    }

    void HeterogeneousMultiDomainStrategy::distributeLayers(
        std::vector<DomainAssignment> &domains,
        int n_layers,
        bool gpu_domains_first) const
    {
        if (domains.empty() || n_layers <= 0)
        {
            return;
        }

        // Calculate total compute weight
        float total_weight = 0.0f;
        for (const auto &domain : domains)
        {
            total_weight += domain.compute_weight;
        }

        if (total_weight <= 0.0f)
        {
            // Fallback: equal distribution
            total_weight = static_cast<float>(domains.size());
            for (auto &domain : domains)
            {
                domain.compute_weight = 1.0f;
            }
        }

        // Separate GPU and CPU domains
        std::vector<DomainAssignment *> gpu_domains;
        std::vector<DomainAssignment *> cpu_domains;

        for (auto &domain : domains)
        {
            if (domain.type == TPDomainType::GPU_INTRA_RANK)
            {
                gpu_domains.push_back(&domain);
            }
            else
            {
                cpu_domains.push_back(&domain);
            }
        }

        // Calculate layer counts proportional to weight, respecting min_layers_per_domain
        int min_layers = config_.min_layers_per_domain;
        int remaining_layers = n_layers;

        // Ensure minimum layers for each domain
        int guaranteed_layers = min_layers * static_cast<int>(domains.size());
        if (guaranteed_layers > n_layers)
        {
            // Not enough layers for minimum per domain, reduce minimum
            min_layers = n_layers / static_cast<int>(domains.size());
            guaranteed_layers = min_layers * static_cast<int>(domains.size());
        }

        // Calculate proportional layers (beyond minimum)
        int distributable_layers = n_layers - guaranteed_layers;
        std::vector<int> layer_counts(domains.size());

        for (size_t i = 0; i < domains.size(); ++i)
        {
            float proportion = domains[i].compute_weight / total_weight;
            int extra_layers = static_cast<int>(proportion * distributable_layers);
            layer_counts[i] = min_layers + extra_layers;
            remaining_layers -= layer_counts[i];
        }

        // Distribute any remaining layers to highest-weight domains
        while (remaining_layers > 0)
        {
            size_t best_idx = 0;
            for (size_t i = 1; i < domains.size(); ++i)
            {
                if (domains[i].compute_weight > domains[best_idx].compute_weight)
                {
                    best_idx = i;
                }
            }
            layer_counts[best_idx]++;
            remaining_layers--;
        }

        // Assign layer ranges
        // Order: GPU domains first if configured, then CPU domains
        int current_layer = 0;

        auto assignLayers = [&](std::vector<DomainAssignment *> &domain_ptrs,
                                std::vector<int> &counts,
                                std::vector<size_t> &indices)
        {
            for (size_t idx : indices)
            {
                domain_ptrs[idx]->layer_start = current_layer;
                current_layer += counts[idx];
                domain_ptrs[idx]->layer_end = current_layer;
            }
        };

        // Build indices and counts for each group
        std::vector<int> gpu_counts, cpu_counts;
        std::vector<size_t> gpu_indices, cpu_indices;

        for (size_t i = 0; i < domains.size(); ++i)
        {
            if (domains[i].type == TPDomainType::GPU_INTRA_RANK)
            {
                size_t gpu_idx = 0;
                for (size_t j = 0; j < gpu_domains.size(); ++j)
                {
                    if (gpu_domains[j] == &domains[i])
                    {
                        gpu_idx = j;
                        break;
                    }
                }
                gpu_indices.push_back(gpu_idx);
                gpu_counts.push_back(layer_counts[i]);
            }
            else
            {
                size_t cpu_idx = 0;
                for (size_t j = 0; j < cpu_domains.size(); ++j)
                {
                    if (cpu_domains[j] == &domains[i])
                    {
                        cpu_idx = j;
                        break;
                    }
                }
                cpu_indices.push_back(cpu_idx);
                cpu_counts.push_back(layer_counts[i]);
            }
        }

        // Assign in order: GPU first if configured, otherwise CPU first
        if (gpu_domains_first)
        {
            // GPU domains get earlier layers (better for prefill)
            for (size_t i = 0; i < gpu_domains.size(); ++i)
            {
                gpu_domains[i]->layer_start = current_layer;
                // Find the layer count for this GPU domain
                for (size_t j = 0; j < domains.size(); ++j)
                {
                    if (&domains[j] == gpu_domains[i])
                    {
                        current_layer += layer_counts[j];
                        break;
                    }
                }
                gpu_domains[i]->layer_end = current_layer;
            }
            for (size_t i = 0; i < cpu_domains.size(); ++i)
            {
                cpu_domains[i]->layer_start = current_layer;
                for (size_t j = 0; j < domains.size(); ++j)
                {
                    if (&domains[j] == cpu_domains[i])
                    {
                        current_layer += layer_counts[j];
                        break;
                    }
                }
                cpu_domains[i]->layer_end = current_layer;
            }
        }
        else
        {
            // CPU domains get earlier layers
            for (size_t i = 0; i < cpu_domains.size(); ++i)
            {
                cpu_domains[i]->layer_start = current_layer;
                for (size_t j = 0; j < domains.size(); ++j)
                {
                    if (&domains[j] == cpu_domains[i])
                    {
                        current_layer += layer_counts[j];
                        break;
                    }
                }
                cpu_domains[i]->layer_end = current_layer;
            }
            for (size_t i = 0; i < gpu_domains.size(); ++i)
            {
                gpu_domains[i]->layer_start = current_layer;
                for (size_t j = 0; j < domains.size(); ++j)
                {
                    if (&domains[j] == gpu_domains[i])
                    {
                        current_layer += layer_counts[j];
                        break;
                    }
                }
                gpu_domains[i]->layer_end = current_layer;
            }
        }
    }

    std::vector<DomainAssignment> HeterogeneousMultiDomainStrategy::computeDomainAssignments(
        const ClusterInventory &inventory,
        int n_layers) const
    {
        std::vector<DomainAssignment> domains;
        int domain_id = 0;

        // Process each node
        for (const auto &node_inv : inventory.nodes)
        {
            // Create GPU domains for each rank with GPUs
            if (config_.enable_gpu_tp)
            {
                for (int rank : node_inv.ranks)
                {
                    if (rank < 0 || rank >= static_cast<int>(inventory.ranks.size()))
                    {
                        continue;
                    }

                    const auto &rank_inv = inventory.ranks[rank];
                    if (rank_inv.hasGPU())
                    {
                        domains.push_back(createGPUDomain(rank_inv, domain_id++));
                    }
                }
            }

            // Create CPU domain for the node (cross-rank if multiple ranks)
            if (config_.enable_cpu_tp)
            {
                // Only create CPU domain if:
                // 1. Node has CPU resources
                // 2. Either no GPU domains on this node, or cpu_compute_fraction > 0
                bool has_gpu_domain = false;
                for (const auto &d : domains)
                {
                    if (d.node_id == node_inv.node_id &&
                        d.type == TPDomainType::GPU_INTRA_RANK)
                    {
                        has_gpu_domain = true;
                        break;
                    }
                }

                if (!has_gpu_domain || config_.cpu_compute_fraction > 0.0f)
                {
                    domains.push_back(createCPUDomain(node_inv, inventory, domain_id++));
                }
            }
        }

        // If no domains created, create a single CPU domain as fallback
        if (domains.empty() && !inventory.nodes.empty())
        {
            domains.push_back(createCPUDomain(inventory.nodes[0], inventory, 0));
        }

        // Distribute layers across domains
        distributeLayers(domains, n_layers, config_.prefer_gpu_early_layers);

        return domains;
    }

    HeterogeneousPlan HeterogeneousMultiDomainStrategy::generatePlan(
        const ClusterInventory &inventory,
        int n_layers) const
    {
        HeterogeneousPlan plan;
        plan.total_layers = n_layers;
        plan.world_size = inventory.world_size;
        plan.node_count = inventory.node_count;

        // Compute domain assignments
        plan.domains = computeDomainAssignments(inventory, n_layers);

        // Build pipeline stages (one per node)
        int layers_per_node = n_layers / std::max(1, plan.node_count);
        int remaining = n_layers % std::max(1, plan.node_count);
        int current_layer = 0;

        for (int node_id = 0; node_id < plan.node_count; ++node_id)
        {
            PipelineStage stage;
            stage.node_id = node_id;
            stage.stage_id = node_id;

            // Find ranks on this node
            if (node_id < static_cast<int>(inventory.nodes.size()))
            {
                stage.ranks = inventory.nodes[node_id].ranks;
            }

            // Calculate layer range for this stage
            int stage_layers = layers_per_node + (node_id < remaining ? 1 : 0);
            stage.layer_start = current_layer;
            stage.layer_end = current_layer + stage_layers;
            current_layer = stage.layer_end;

            // Collect domains for this stage
            for (const auto &domain : plan.domains)
            {
                if (domain.node_id == node_id)
                {
                    stage.domains.push_back(domain);
                }
            }

            plan.stages.push_back(stage);
        }

        return plan;
    }

    PlacementPlan HeterogeneousMultiDomainStrategy::compute(const PlacementInput &input) const
    {
        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = input.any_rank_has_gpu;
        plan.total_gpu_memory = input.total_gpu_memory;
        plan.strategy_name = name();

        // Generate heterogeneous plan
        HeterogeneousPlan het_plan = generatePlan(input.cluster_inventory, input.n_layers);

        // Log the plan
        LOG_DEBUG("[HeterogeneousMultiDomainStrategy] Generated plan:");
        LOG_DEBUG("[HeterogeneousMultiDomainStrategy]   " << het_plan.node_count
                                                         << " nodes, " << het_plan.world_size << " ranks");
        LOG_DEBUG("[HeterogeneousMultiDomainStrategy]   " << het_plan.domains.size()
                                                         << " TP domains");
        for (const auto &domain : het_plan.domains)
        {
            LOG_DEBUG("[HeterogeneousMultiDomainStrategy]   " << domain.toString());
        }

        // Convert to PlacementPlan format
        plan.layers.resize(input.n_layers);

        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;

            // Find domain for this layer
            const DomainAssignment *domain = het_plan.getDomainForLayer(layer);

            if (domain && !domain->ranks.empty() && !domain->devices.empty())
            {
                lp.owner_rank = domain->ranks[0];

                // Set device based on domain type
                if (domain->type == TPDomainType::GPU_INTRA_RANK && !domain->devices.empty())
                {
                    const DeviceId &dev = domain->devices[0];
                    if (dev.type == DeviceType::CUDA || dev.type == DeviceType::ROCm)
                    {
                        lp.device = PlacementDevice::gpu(dev.ordinal);
                        lp.attention_device = lp.device;
                        lp.ffn_device = lp.device;
                    }
                    else
                    {
                        lp.device = PlacementDevice::cpu();
                        lp.attention_device = lp.device;
                        lp.ffn_device = lp.device;
                    }
                }
                else
                {
                    // CPU domain
                    lp.device = PlacementDevice::cpu();
                    lp.attention_device = PlacementDevice::cpu();
                    lp.ffn_device = PlacementDevice::cpu();
                }

                // CPU participation in decode for GPU layers
                if (domain->type == TPDomainType::GPU_INTRA_RANK && config_.enable_cpu_tp)
                {
                    lp.cpu_participates_in_decode = true;
                }
            }
            else
            {
                // Fallback to CPU
                lp.owner_rank = 0;
                lp.device = PlacementDevice::cpu();
                lp.attention_device = PlacementDevice::cpu();
                lp.ffn_device = PlacementDevice::cpu();
            }

            lp.split_attention_ffn = false;
        }

        // Global tensor placement
        // Put globals on the device of the first domain (usually GPU)
        if (!het_plan.domains.empty())
        {
            const auto &first_domain = het_plan.domains[0];
            if (first_domain.type == TPDomainType::GPU_INTRA_RANK &&
                !first_domain.devices.empty())
            {
                const DeviceId &dev = first_domain.devices[0];
                if (dev.type == DeviceType::CUDA || dev.type == DeviceType::ROCm)
                {
                    plan.global.embedding_device = PlacementDevice::gpu(dev.ordinal);
                    plan.global.lm_head_device = PlacementDevice::gpu(dev.ordinal);
                    plan.global.final_norm_device = PlacementDevice::gpu(dev.ordinal);
                }
                else
                {
                    plan.global.embedding_device = PlacementDevice::cpu();
                    plan.global.lm_head_device = PlacementDevice::cpu();
                    plan.global.final_norm_device = PlacementDevice::cpu();
                }
            }
            else
            {
                plan.global.embedding_device = PlacementDevice::cpu();
                plan.global.lm_head_device = PlacementDevice::cpu();
                plan.global.final_norm_device = PlacementDevice::cpu();
            }
        }
        else
        {
            plan.global.embedding_device = PlacementDevice::cpu();
            plan.global.lm_head_device = PlacementDevice::cpu();
            plan.global.final_norm_device = PlacementDevice::cpu();
        }

        // Sharding for large vocab
        plan.global.shard_embedding = (input.world_size > 1 && input.vocab_size > 100000);
        plan.global.shard_lm_head = (input.world_size > 1 && input.vocab_size > 100000);

        return plan;
    }

} // namespace llaminar2
