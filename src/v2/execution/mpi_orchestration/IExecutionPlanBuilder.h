/**
 * @file IExecutionPlanBuilder.h
 * @brief Interface for building execution plans from orchestration config
 *
 * The execution plan builder translates cluster-level orchestration configuration
 * into per-rank execution plans. This interface enables mocking for unit tests.
 *
 * Workflow:
 * 1. Rank 0 gathers OrchestrationConfig + ClusterInventory
 * 2. IExecutionPlanBuilder::buildAllPlans() generates plans for all ranks
 * 3. Plans are broadcast to all ranks via MPI
 * 4. Each rank extracts its plan and executes it
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "RankExecutionPlan.h"
#include "DeviceInventory.h"
#include "../../config/OrchestrationConfig.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // ModelConfig (minimal for plan building)
    // =========================================================================

    /**
     * @brief Model configuration needed for execution plan building
     *
     * Contains the subset of model architecture information needed
     * to compute layer assignments and weight sharding.
     */
    struct ModelConfig
    {
        std::string name = "unknown";      ///< Model name (e.g., "Qwen2-7B")
        int n_layers = 0;                  ///< Number of transformer layers
        int n_heads = 0;                   ///< Number of attention heads
        int n_kv_heads = 0;                ///< Number of KV heads (for GQA)
        int hidden_size = 0;               ///< Hidden dimension (d_model)
        int intermediate_size = 0;         ///< FFN intermediate dimension
        int vocab_size = 0;                ///< Vocabulary size
        int head_dim = 0;                  ///< Per-head dimension (hidden_size / n_heads)
        size_t estimated_weight_bytes = 0; ///< Estimated weight size in bytes

        /**
         * @brief Create config for Qwen2-0.5B
         */
        static ModelConfig qwen2_0_5b()
        {
            return {"Qwen2-0.5B", 24, 14, 2, 896, 4864, 151936, 64,
                    500ULL * 1024 * 1024};
        }

        /**
         * @brief Create config for Qwen2-7B
         */
        static ModelConfig qwen2_7b()
        {
            return {"Qwen2-7B", 28, 28, 4, 3584, 18944, 152064, 128,
                    4ULL * 1024 * 1024 * 1024};
        }

        /**
         * @brief Create config for Qwen2-72B
         */
        static ModelConfig qwen2_72b()
        {
            return {"Qwen2-72B", 80, 64, 8, 8192, 29568, 152064, 128,
                    40ULL * 1024 * 1024 * 1024};
        }

        /**
         * @brief Validate model configuration
         * @return Empty vector if valid, otherwise list of errors
         */
        std::vector<std::string> validate() const
        {
            std::vector<std::string> errors;
            if (n_layers <= 0)
            {
                errors.push_back("n_layers must be > 0");
            }
            if (n_heads <= 0)
            {
                errors.push_back("n_heads must be > 0");
            }
            if (n_kv_heads <= 0)
            {
                errors.push_back("n_kv_heads must be > 0");
            }
            if (hidden_size <= 0)
            {
                errors.push_back("hidden_size must be > 0");
            }
            return errors;
        }

        /**
         * @brief String representation
         */
        std::string toString() const
        {
            std::ostringstream ss;
            ss << "ModelConfig{name='" << name << "'"
               << ", n_layers=" << n_layers
               << ", n_heads=" << n_heads
               << ", n_kv_heads=" << n_kv_heads
               << ", hidden_size=" << hidden_size
               << ", vocab_size=" << vocab_size << "}";
            return ss.str();
        }
    };

    // =========================================================================
    // IExecutionPlanBuilder Interface
    // =========================================================================

    /**
     * @brief Interface for building execution plans
     *
     * This interface is mockable for unit testing. The concrete implementation
     * (ExecutionPlanBuilder) handles all the logic for:
     * - Named domain resolution
     * - Simple TP/PP auto-configuration
     * - Weight shard calculation
     * - PP neighbor determination
     */
    class IExecutionPlanBuilder
    {
    public:
        virtual ~IExecutionPlanBuilder() = default;

        /**
         * @brief Build execution plans for ALL MPI ranks
         *
         * Called on rank 0 (or collaboratively). Generates a plan for every
         * rank in the cluster, which can then be broadcast.
         *
         * @param config Orchestration configuration from CLI/YAML
         * @param model_config Model architecture information
         * @param cluster_inventory Device inventory for all ranks
         * @return Vector of plans, one per rank (indexed by rank ID)
         */
        virtual std::vector<RankExecutionPlan> buildAllPlans(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory) = 0;

        /**
         * @brief Build execution plan for a specific rank
         *
         * Convenience method that builds all plans and returns the one
         * for the specified rank. Less efficient than buildAllPlans()
         * when plans for all ranks are needed.
         *
         * @param config Orchestration configuration from CLI/YAML
         * @param model_config Model architecture information
         * @param cluster_inventory Device inventory for all ranks
         * @param rank MPI rank to build plan for
         * @return Execution plan for the specified rank
         */
        virtual RankExecutionPlan buildPlanForRank(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory,
            int rank) = 0;

        /**
         * @brief Validate configuration can produce valid plans
         *
         * Checks that the configuration is internally consistent and
         * compatible with the cluster inventory, without fully building plans.
         *
         * @param config Orchestration configuration
         * @param model_config Model architecture information
         * @param cluster_inventory Cluster device inventory
         * @return Empty vector if valid, otherwise list of error messages
         */
        virtual std::vector<std::string> validateConfig(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory) = 0;
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create the default execution plan builder
     * @return Unique pointer to IExecutionPlanBuilder implementation
     */
    std::unique_ptr<IExecutionPlanBuilder> createExecutionPlanBuilder();

} // namespace llaminar2
