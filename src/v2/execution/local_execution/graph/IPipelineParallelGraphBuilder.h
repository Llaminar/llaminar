/**
 * @file IPipelineParallelGraphBuilder.h
 * @brief Interface for adding pipeline parallelism stages to compute graphs
 *
 * Part of Phase 3: Pipeline Parallelism Integration
 *
 * This interface enables insertion of Send/Recv stages at pipeline boundaries
 * to transfer activations between MPI ranks running different pipeline stages.
 *
 * Usage:
 *   auto pp_builder = createPipelineParallelGraphBuilder(mpi_ctx);
 *   pp_builder->insertPPStages(graph, execution_plan, input_buf, output_buf);
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../mpi_orchestration/RankExecutionPlan.h"
#include "IGraphExecutor.h" // For ComputeGraph forward declaration
#include "../../../utils/MPIContext.h"
#include <memory>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class ComputeGraph;

    /**
     * @brief Interface for adding pipeline parallelism stages to compute graphs
     *
     * The IPipelineParallelGraphBuilder is responsible for inserting the
     * appropriate Send/Recv stages at pipeline boundaries based on the
     * execution plan for the current rank.
     *
     * Design principles:
     * - First PP stage: No receive, only send (if not last)
     * - Last PP stage: No send, only receive (if not first)
     * - Middle PP stages: Both send and receive
     * - Single-stage PP (no parallelism): Neither send nor receive
     */
    class IPipelineParallelGraphBuilder
    {
    public:
        virtual ~IPipelineParallelGraphBuilder() = default;

        // =====================================================================
        // Core Operations
        // =====================================================================

        /**
         * @brief Insert receive stage at the beginning of the graph
         *
         * Only inserts a stage if the execution plan indicates this rank
         * receives activations from a previous PP stage (prev_rank.has_value()).
         *
         * The receive stage will be named "recv_from_stage_N" where N is the
         * previous PP stage ID.
         *
         * @param graph The compute graph to modify
         * @param plan The execution plan for this rank
         * @param input_buffer Buffer to receive activations into
         * @return true if a receive stage was inserted, false otherwise
         */
        virtual bool insertReceiveStage(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *input_buffer) = 0;

        /**
         * @brief Insert send stage at the end of the graph
         *
         * Only inserts a stage if the execution plan indicates this rank
         * sends activations to a next PP stage (next_rank.has_value()).
         *
         * The send stage will be named "send_to_stage_N" where N is the
         * next PP stage ID.
         *
         * @param graph The compute graph to modify
         * @param plan The execution plan for this rank
         * @param output_buffer Buffer containing activations to send
         * @return true if a send stage was inserted, false otherwise
         */
        virtual bool insertSendStage(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *output_buffer) = 0;

        /**
         * @brief Insert both PP stages as appropriate
         *
         * Convenience method that calls insertReceiveStage and insertSendStage
         * in the correct order. The receive stage becomes a root node, and
         * existing root nodes depend on it. The send stage depends on all
         * existing leaf nodes.
         *
         * @param graph The compute graph to modify
         * @param plan The execution plan for this rank
         * @param input_buffer Buffer for receiving activations
         * @param output_buffer Buffer for sending activations
         * @return Number of stages inserted (0, 1, or 2)
         */
        virtual int insertPPStages(
            ComputeGraph &graph,
            const RankExecutionPlan &plan,
            TensorBase *input_buffer,
            TensorBase *output_buffer) = 0;

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Enable/disable async MPI operations
         *
         * When enabled, Send/Recv stages use non-blocking MPI calls.
         * Useful for overlapping communication with computation.
         *
         * @param async Whether to use async operations
         */
        virtual void setAsyncMode(bool async) = 0;

        /**
         * @brief Check if async mode is enabled
         */
        virtual bool asyncMode() const = 0;
    };

    // =========================================================================
    // Factory
    // =========================================================================

    /**
     * @brief Create a pipeline parallel graph builder
     *
     * @param mpi_ctx MPI context for communication
     * @return Unique pointer to builder instance
     */
    std::unique_ptr<IPipelineParallelGraphBuilder> createPipelineParallelGraphBuilder(
        MPIContext *mpi_ctx);

} // namespace llaminar2
