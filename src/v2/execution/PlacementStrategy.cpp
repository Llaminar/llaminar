/**
 * @file PlacementStrategy.cpp
 * @brief Implementation of placement strategies
 * @author David Sanftenberg
 * @date December 2025
 */

#include "PlacementStrategy.h"
#include "../utils/Logger.h"
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // CPUOnlyStrategy Implementation
    // =========================================================================

    bool CPUOnlyStrategy::isApplicable(const PlacementInput &input) const
    {
        // CPU-only is ALWAYS applicable (every system has a CPU)
        // User can explicitly choose CPU even when GPU is available
        (void)input; // unused
        return true;
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
        // NO SILENT FALLBACK: GPU placement must be implemented or fail loudly
        // GPUFirstStrategy is not yet implemented - throw to make this clear
        throw std::runtime_error(
            "[GPUFirstStrategy] GPU placement strategy requested but not yet implemented. "
            "Use 'CPUOnly' strategy or implement GPU placement in Phase G1+.");
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
        // Priority 1: User-specified strategy - NO SILENT FALLBACK
        if (!input.preferred_strategy.empty())
        {
            auto strategy = create(input.preferred_strategy);
            if (!strategy)
            {
                throw std::runtime_error(
                    "[PlacementStrategyFactory] Unknown strategy: '" + input.preferred_strategy +
                    "'. Valid strategies: CPUOnly, GPUFirst.");
            }
            if (!strategy->isApplicable(input))
            {
                throw std::runtime_error(
                    "[PlacementStrategyFactory] Strategy '" + input.preferred_strategy +
                    "' is not applicable for current configuration. "
                    "Check GPU availability and force_cpu_only flag.");
            }
            LOG_DEBUG("[PlacementStrategyFactory] Using user-specified strategy: "
                      << input.preferred_strategy);
            return strategy;
        }

        // Priority 2: Force flags
        if (input.force_cpu_only)
        {
            LOG_DEBUG("[PlacementStrategyFactory] CPU-only mode forced");
            return std::make_unique<CPUOnlyStrategy>();
        }

        // Priority 3: Auto-select based on device availability
        // For GPU, we need GPU strategies to be implemented first
        if (input.any_rank_has_gpu)
        {
            // NO SILENT FALLBACK: If auto-selecting and GPU available, fail loudly
            // User should either use force_cpu_only or wait for GPU implementation
            throw std::runtime_error(
                "[PlacementStrategyFactory] GPU detected but GPU placement strategies not yet implemented. "
                "Set force_cpu_only=true or use CPUOnly strategy explicitly.");
        }

        // No GPU available, CPU-only is appropriate
        LOG_DEBUG("[PlacementStrategyFactory] No GPU available, using CPUOnly");
        return std::make_unique<CPUOnlyStrategy>();
    }

    std::vector<std::string> PlacementStrategyFactory::availableStrategies()
    {
        return {"CPUOnly", "GPUFirst"};
    }

} // namespace llaminar2
