/**
 * @file PlacementStrategy.cpp
 * @brief Implementation of placement strategies
 * @author David Sanftenberg
 * @date December 2025
 */

#include "PlacementStrategy.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // CPUOnlyStrategy Implementation
    // =========================================================================

    bool CPUOnlyStrategy::isApplicable(const PlacementInput &input) const
    {
        // CPU-only is always applicable (every system has a CPU)
        // But we prefer other strategies if GPU is available and not forced
        return input.force_cpu_only || !input.any_rank_has_gpu;
    }

    PlacementPlan CPUOnlyStrategy::compute(const PlacementInput &input) const
    {
        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = false; // CPU-only strategy never uses GPU
        plan.total_gpu_memory = 0;
        plan.strategy_name = name();

        // Global tensors: All on CPU, shard if multi-rank
        plan.global.embedding_device = PlacementDevice::CPU;
        plan.global.lm_head_device = PlacementDevice::CPU;
        plan.global.final_norm_device = PlacementDevice::CPU;
        plan.global.shard_embedding = (input.world_size > 1 && input.vocab_size > 100000);
        plan.global.shard_lm_head = (input.world_size > 1 && input.vocab_size > 100000);

        // Allocate layer placements
        plan.layers.resize(input.n_layers);

        // Simple round-robin distribution of layers across ranks
        // For CPU-only TP, all ranks participate in all layers (row-parallel)
        // The "owner_rank" here is less meaningful - all ranks have all weights
        // but we track it for future pipeline parallelism support
        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;

            // For tensor parallelism, all ranks own all layers (row-sharded)
            // owner_rank = 0 means "all ranks" for TP
            // For pipeline parallelism (future), this would be layer / layers_per_rank
            lp.owner_rank = 0;

            lp.device = PlacementDevice::CPU;
            lp.attention_device = PlacementDevice::CPU;
            lp.ffn_device = PlacementDevice::CPU;
            lp.split_attention_ffn = false;
        }

        LOG_DEBUG("[CPUOnlyStrategy] Generated plan: " << input.n_layers << " layers, "
                                                       << input.world_size << " ranks, all CPU");

        return plan;
    }

    // =========================================================================
    // GPUFirstStrategy Implementation (Placeholder)
    // =========================================================================

    bool GPUFirstStrategy::isApplicable(const PlacementInput &input) const
    {
        // GPU-first requires at least one GPU and not forced CPU-only
        return !input.force_cpu_only && input.any_rank_has_gpu;
    }

    PlacementPlan GPUFirstStrategy::compute(const PlacementInput &input) const
    {
        // For now, GPUFirstStrategy is a placeholder that falls back to CPU
        // This will be implemented in Phase G1+

        LOG_WARN("[GPUFirstStrategy] GPU placement not yet implemented, falling back to CPU");

        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = false; // Not using GPU yet
        plan.total_gpu_memory = input.total_gpu_memory;
        plan.strategy_name = name() + " (CPU fallback)";

        // For now, just do CPU placement
        plan.global.embedding_device = PlacementDevice::CPU;
        plan.global.lm_head_device = PlacementDevice::CPU;
        plan.global.final_norm_device = PlacementDevice::CPU;
        plan.global.shard_embedding = (input.world_size > 1);
        plan.global.shard_lm_head = (input.world_size > 1);

        plan.layers.resize(input.n_layers);
        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;
            lp.owner_rank = 0;
            lp.device = PlacementDevice::CPU;
            lp.attention_device = PlacementDevice::CPU;
            lp.ffn_device = PlacementDevice::CPU;
            lp.split_attention_ffn = false;
        }

        return plan;
    }

    // =========================================================================
    // PlacementStrategyFactory Implementation
    // =========================================================================

    std::unique_ptr<PlacementStrategy> PlacementStrategyFactory::create(const std::string &name)
    {
        if (name == "CPUOnly" || name == "cpu" || name == "cpu_only")
        {
            return std::make_unique<CPUOnlyStrategy>();
        }
        if (name == "GPUFirst" || name == "gpu" || name == "gpu_first")
        {
            return std::make_unique<GPUFirstStrategy>();
        }

        LOG_WARN("[PlacementStrategyFactory] Unknown strategy: " << name);
        return nullptr;
    }

    std::unique_ptr<PlacementStrategy> PlacementStrategyFactory::autoSelect(const PlacementInput &input)
    {
        // Priority 1: User-specified strategy
        if (!input.preferred_strategy.empty())
        {
            auto strategy = create(input.preferred_strategy);
            if (strategy && strategy->isApplicable(input))
            {
                LOG_DEBUG("[PlacementStrategyFactory] Using user-specified strategy: "
                          << input.preferred_strategy);
                return strategy;
            }
            LOG_WARN("[PlacementStrategyFactory] User strategy '" << input.preferred_strategy
                                                                  << "' not applicable, auto-selecting");
        }

        // Priority 2: Force flags
        if (input.force_cpu_only)
        {
            LOG_DEBUG("[PlacementStrategyFactory] CPU-only mode forced");
            return std::make_unique<CPUOnlyStrategy>();
        }

        // Priority 3: Auto-select based on device availability
        // For now, always use CPU-only until GPU support is implemented
        if (input.any_rank_has_gpu && !input.force_cpu_only)
        {
            LOG_DEBUG("[PlacementStrategyFactory] GPU available but GPU strategies "
                      "not yet implemented, using CPUOnly");
        }

        return std::make_unique<CPUOnlyStrategy>();
    }

    std::vector<std::string> PlacementStrategyFactory::availableStrategies()
    {
        return {"CPUOnly", "GPUFirst"};
    }

} // namespace llaminar2
