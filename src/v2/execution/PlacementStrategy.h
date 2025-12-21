/**
 * @file PlacementStrategy.h
 * @brief Strategy interface for computing weight/compute placement
 *
 * PlacementStrategy computes a PlacementPlan from:
 * - Model info (architecture, n_layers, memory estimates)
 * - MPI topology (ranks, nodes, device capabilities)
 *
 * Different strategies optimize for different goals:
 * - CPUOnlyStrategy: All compute on CPU (current default)
 * - MemoryFirstStrategy: Fit as much as possible on GPU
 * - LatencyFirstStrategy: Minimize inter-device transfers
 * - ThroughputFirstStrategy: Maximize parallel utilization
 *
 * Strategy selection:
 * - Automatic: Based on available devices and model size
 * - Manual: User specifies via CLI flag or env var
 *
 * Design principle: All ranks compute the SAME placement plan deterministically.
 * This avoids needing to broadcast the plan - each rank runs the same algorithm
 * with the same inputs (post-capability-exchange) and gets the same output.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include "PlacementPlan.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class MPITopology;
    struct RankPlacement;
    struct DeviceCapability;

    /**
     * @brief Input parameters for placement strategy computation
     *
     * Contains all information needed to compute a PlacementPlan.
     * Gathered from ModelLoader metadata and MPITopology.
     */
    struct PlacementInput
    {
        // Model info
        std::string architecture;          ///< e.g., "qwen2", "llama"
        int n_layers = 0;                  ///< Number of transformer layers
        size_t d_model = 0;                ///< Hidden dimension
        size_t d_ff = 0;                   ///< FFN intermediate dimension
        size_t vocab_size = 0;             ///< Vocabulary size
        size_t n_heads = 0;                ///< Attention heads
        size_t n_kv_heads = 0;             ///< KV heads (for GQA)
        std::string quant_type;            ///< Quantization type (e.g., "Q4_0", "Q8_0")
        size_t estimated_memory_bytes = 0; ///< Estimated total model memory

        // MPI topology (from MPITopology after capability exchange)
        int world_size = 1;
        int ranks_per_node = 1;
        int node_count = 1;
        std::vector<float> rank_compute_weights; ///< Relative compute power per rank

        // Device availability (aggregated from all ranks)
        bool any_rank_has_gpu = false;
        size_t total_gpu_memory = 0;
        size_t total_cpu_memory = 0;

        // Constraints (from CLI or environment)
        bool force_cpu_only = false;    ///< --cpu-only flag
        bool force_gpu_only = false;    ///< --gpu-only flag
        int max_gpu_layers = -1;        ///< --n-gpu-layers N (-1 = no limit)
        std::string preferred_strategy; ///< User-specified strategy name
    };

    /**
     * @brief Abstract base class for placement strategies
     *
     * Subclasses implement compute() to generate a PlacementPlan
     * based on model info and device capabilities.
     */
    class PlacementStrategy
    {
    public:
        virtual ~PlacementStrategy() = default;

        /**
         * @brief Compute a placement plan from inputs
         * @param input Model and topology information
         * @return Computed placement plan
         *
         * IMPORTANT: This method must be DETERMINISTIC. All ranks call it
         * with the same inputs (after capability exchange) and must get
         * the exact same output. This avoids needing to broadcast the plan.
         */
        virtual PlacementPlan compute(const PlacementInput &input) const = 0;

        /**
         * @brief Get strategy name for logging/debugging
         */
        virtual std::string name() const = 0;

        /**
         * @brief Check if this strategy is applicable to the given input
         * @param input Model and topology information
         * @return true if strategy can generate a valid plan
         *
         * Used by automatic strategy selection to find applicable strategies.
         */
        virtual bool isApplicable(const PlacementInput &input) const = 0;
    };

    /**
     * @brief CPU-only strategy: All compute on CPU
     *
     * This is the current default strategy. All layers execute on CPU.
     * Weights are distributed across ranks for tensor parallelism,
     * but no GPU compute is used.
     *
     * Applicable when:
     * - force_cpu_only is true, OR
     * - No GPU is available on any rank
     */
    class CPUOnlyStrategy : public PlacementStrategy
    {
    public:
        PlacementPlan compute(const PlacementInput &input) const override;
        std::string name() const override { return "CPUOnly"; }
        bool isApplicable(const PlacementInput &input) const override;
    };

    /**
     * @brief GPU-first strategy: Fit as many layers on GPU as possible
     *
     * Places layers on GPU until GPU memory is exhausted, then spills
     * remaining layers to CPU. Optimizes for maximizing GPU utilization.
     *
     * NOT YET IMPLEMENTED - placeholder for Phase G1+
     */
    class GPUFirstStrategy : public PlacementStrategy
    {
    public:
        PlacementPlan compute(const PlacementInput &input) const override;
        std::string name() const override { return "GPUFirst"; }
        bool isApplicable(const PlacementInput &input) const override;
    };

    /**
     * @brief Factory for creating placement strategies
     *
     * Supports:
     * - Automatic selection based on capabilities
     * - Manual selection by name
     */
    class PlacementStrategyFactory
    {
    public:
        /**
         * @brief Create strategy by name
         * @param name Strategy name ("CPUOnly", "GPUFirst", etc.)
         * @return Strategy instance, or nullptr if unknown name
         */
        static std::unique_ptr<PlacementStrategy> create(const std::string &name);

        /**
         * @brief Auto-select best strategy for given input
         * @param input Model and topology information
         * @return Best applicable strategy
         *
         * Selection priority:
         * 1. User-specified strategy (input.preferred_strategy)
         * 2. Force flags (force_cpu_only, force_gpu_only)
         * 3. Automatic based on device availability
         */
        static std::unique_ptr<PlacementStrategy> autoSelect(const PlacementInput &input);

        /**
         * @brief Get list of all available strategy names
         */
        static std::vector<std::string> availableStrategies();
    };

} // namespace llaminar2
