/**
 * @file TreeToRunnerCompiler.h
 * @brief Compile ParallelismTree into nested IInferenceRunner hierarchy
 *
 * TreeToRunnerCompiler walks the parallelism tree bottom-up, producing
 * IInferenceRunner instances:
 *
 * - DEVICE leaf → DeviceGraphOrchestrator (or mock for testing)
 * - TP node (same rank) → RankOrchestrator(TP mode)
 * - PP node (same rank) → RankOrchestrator(PP mode)
 * - PP node (cross rank) → PipelineRunner
 *
 * Each rank compiles only its portion of the tree. Nodes not owned by
 * this rank are skipped (shouldCompile returns false).
 *
 * Example: For a 2-rank PP with TP on each rank:
 * ```
 * PP(global)
 * ├── TP(rank0, [cuda:0, cuda:1])  ← rank 0 compiles this subtree
 * └── TP(rank1, [cuda:0, cuda:1])  ← rank 1 compiles this subtree
 * ```
 *
 * Rank 0 gets: PipelineRunner with stage 0 = RankOrchestrator(TP)
 * Rank 1 gets: PipelineRunner with stage 1 = RankOrchestrator(TP)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ParallelismTree.h"
#include "PipelineRunner.h"
#include "../local_execution/orchestrators/IInferenceRunner.h"
#include <memory>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class IModelContext;
    class CommHierarchy;

    /**
     * @brief Compile ParallelismTree into nested IInferenceRunner hierarchy
     *
     * The compiler walks the tree and produces appropriate runner instances
     * for each node type. Cross-rank PP produces PipelineRunner, local TP/PP
     * produces RankOrchestrator.
     */
    class TreeToRunnerCompiler
    {
    public:
        // =====================================================================
        // Configuration Types
        // =====================================================================

        /**
         * @brief Factory for creating device-level runners
         *
         * Allows injection of mock runners for unit testing.
         * The factory receives:
         * - node: The DEVICE node to create runner for
         * - model_ctx: Model context (may be nullptr for tests)
         *
         * Returns: An IInferenceRunner for that device
         */
        using DeviceRunnerFactory = std::function<std::unique_ptr<IInferenceRunner>(
            const ParallelismNode &node,
            const std::shared_ptr<IModelContext> &model_ctx)>;

        /**
         * @brief Factory for creating TP runners
         *
         * Allows injection of mock runners for unit testing.
         *
         * Note: For pure TP-over-DEVICE trees, child_runners will be empty
         * because the compiler skips compiling DEVICE children of TP nodes
         * (the TP orchestrator creates its own TP-aware device runners
         * internally). child_runners is populated only when TP wraps
         * non-DEVICE children (e.g., PP sub-groups).
         */
        using TPRunnerFactory = std::function<std::unique_ptr<IInferenceRunner>(
            const ParallelismNode &node,
            std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
            const std::shared_ptr<IModelContext> &model_ctx)>;

        /**
         * @brief Factory for creating PP runners (same-rank)
         *
         * Allows injection of mock runners for unit testing.
         */
        using LocalPPRunnerFactory = std::function<std::unique_ptr<IInferenceRunner>(
            const ParallelismNode &node,
            std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
            const std::shared_ptr<IModelContext> &model_ctx)>;

        /**
         * @brief Context for compilation
         *
         * Contains all the information needed to compile a tree into runners.
         */
        struct CompileContext
        {
            /// Model context for weight loading (may be nullptr for testing)
            std::shared_ptr<IModelContext> model_ctx;

            /// This rank's MPI rank
            int my_rank = 0;

            /// Total MPI world size
            int world_size = 1;

            /// Maximum sequence length
            size_t max_seq_len = 4096;

            /// Batch size
            int batch_size = 1;

            /// Hidden dimension (for transfer buffers)
            int hidden_dim = 896;

            /// Vocabulary size
            int vocab_size = 151936;

            /// Pre-built communicator hierarchy (optional, for cross-rank TP)
            const CommHierarchy *comm_hierarchy = nullptr;

            /// Factory for creating device runners (default: real orchestrator)
            DeviceRunnerFactory device_runner_factory;

            /// Factory for creating TP runners (default: real RankOrchestrator)
            TPRunnerFactory tp_runner_factory;

            /// Factory for creating local PP runners (default: real RankOrchestrator)
            LocalPPRunnerFactory local_pp_runner_factory;
        };

        // =====================================================================
        // Main API
        // =====================================================================

        /**
         * @brief Compile the tree into a runner for this rank
         *
         * Walks the tree and produces an IInferenceRunner hierarchy for this
         * rank's portion of the tree.
         *
         * @param tree The parallelism tree (with layers assigned)
         * @param ctx Compilation context
         * @return Runner for this rank's portion, or nullptr if rank has no work
         */
        static std::unique_ptr<IInferenceRunner> compile(
            const ParallelismTree &tree,
            const CompileContext &ctx);

        /**
         * @brief Check if a node should be compiled by this rank
         *
         * A rank compiles a node if:
         * - It's a DEVICE leaf owned by this rank
         * - It's an interior node with descendants owned by this rank
         *
         * @param node Node to check
         * @param my_rank This rank's MPI rank
         * @return true if this rank should compile this node
         */
        static bool shouldCompile(const ParallelismNode &node, int my_rank);

        /**
         * @brief Check if a PP node is cross-rank
         *
         * A PP node is cross-rank if its children span multiple MPI ranks.
         *
         * @param node PP node to check
         * @return true if cross-rank PP
         */
        static bool isCrossRankPP(const ParallelismNode &node);

        /**
         * @brief Get the ranks covered by a node's subtree
         *
         * @param node Node to analyze
         * @return Set of MPI ranks in the subtree
         */
        static std::set<int> getSubtreeRanks(const ParallelismNode &node);

    private:
        /**
         * @brief Compile a single node recursively
         *
         * @param node Node to compile
         * @param ctx Compilation context
         * @return Runner for this node, or nullptr if not owned by this rank
         */
        static std::unique_ptr<IInferenceRunner> compileNode(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Compile a DEVICE leaf node
         *
         * @param node DEVICE node
         * @param ctx Compilation context
         * @return DeviceGraphOrchestrator (or mock)
         */
        static std::unique_ptr<IInferenceRunner> compileDevice(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Compile a TP node
         *
         * DEVICE children are NOT compiled — they are metadata specifying
         * which devices the TP group uses. The TP orchestrator (MDO) creates
         * its own TP-aware device runners internally. Only non-DEVICE children
         * (e.g., PP sub-groups) are compiled into child runners.
         *
         * @param node TP node
         * @param ctx Compilation context
         * @return RankOrchestrator(TP mode) (or mock)
         */
        static std::unique_ptr<IInferenceRunner> compileTP(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Compile a local (same-rank) PP node
         *
         * @param node PP node (all children same rank)
         * @param ctx Compilation context
         * @return RankOrchestrator(PP mode) (or mock)
         */
        static std::unique_ptr<IInferenceRunner> compileLocalPP(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Compile a cross-rank PP node
         *
         * @param node PP node (children span ranks)
         * @param ctx Compilation context
         * @return PipelineRunner
         */
        static std::unique_ptr<IInferenceRunner> compileCrossRankPP(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Build stage info for PipelineRunner from PP children
         *
         * @param node PP node
         * @param ctx Compilation context
         * @return Vector of StageInfo
         */
        static std::vector<PipelineRunner::StageInfo> buildStageInfos(
            const ParallelismNode &node,
            const CompileContext &ctx);

        /**
         * @brief Build transfer info between adjacent stages
         *
         * @param node PP node
         * @return Vector of TransferInfo
         */
        static std::vector<PipelineRunner::TransferInfo> buildTransferInfos(
            const ParallelismNode &node);
    };

} // namespace llaminar2
