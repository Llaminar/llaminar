/**
 * @file GraphExecutor.h
 * @brief Compute graph execution engine
 * @author David Sanftenberg
 * @date December 2025
 *
 * GraphExecutor coordinates the execution of ComputeStages within a compute graph,
 * managing:
 * - Device-aware stage scheduling
 * - Asynchronous execution with dependency tracking
 * - Multi-device work distribution via WorkDistributor
 *
 * Note: This was previously named LayerExecutor but renamed to better reflect
 * its purpose - it executes graphs, not layers specifically.
 *
 * Architecture:
 *   Pipeline / ModelExecutor
 *      |
 *      v
 *   GraphExecutor  <-- This component
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 *      |
 *      v
 *   DeviceContext (execution environment)
 */

#pragma once

#include "IGraphExecutor.h"
#include "compute_stages/ComputeStages.h"
#include "DeviceContext.h"
#include "WorkDistributor.h"
#include "GraphBufferManager.h"
#include "../backends/DeviceId.h"
#include "../utils/DebugEnv.h" // For LLAMINAR_ASSERTIONS_ACTIVE
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias

    /**
     * @brief Represents a node in the compute graph
     */
    struct ComputeNode
    {
        std::string name;                      ///< Node identifier
        std::unique_ptr<IComputeStage> stage;  ///< The compute stage
        std::vector<std::string> dependencies; ///< Names of nodes this depends on
        DeviceId device;                       ///< Target device for execution
        bool completed;                        ///< Execution complete flag

        ComputeNode() : device(DeviceId::cpu()), completed(false) {}
        ComputeNode(std::string n, std::unique_ptr<IComputeStage> s, DeviceId dev = DeviceId::cpu())
            : name(std::move(n)), stage(std::move(s)), device(dev), completed(false) {}
    };

    /**
     * @brief Compute graph for execution
     *
     * A directed acyclic graph (DAG) of ComputeNodes with dependency tracking.
     * Enables parallel execution of independent nodes and proper ordering of
     * dependent operations.
     */
    class ComputeGraph
    {
    public:
        ComputeGraph() = default;
        ~ComputeGraph() = default;

        // Non-copyable
        ComputeGraph(const ComputeGraph &) = delete;
        ComputeGraph &operator=(const ComputeGraph &) = delete;

        // Movable
        ComputeGraph(ComputeGraph &&) = default;
        ComputeGraph &operator=(ComputeGraph &&) = default;

        /**
         * @brief Add a node to the graph
         * @param name Unique node identifier
         * @param stage The compute stage to execute
         * @param device Target device (DeviceId::cpu() for auto/CPU)
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addNode(const std::string &name,
                              std::unique_ptr<IComputeStage> stage,
                              DeviceId device = DeviceId::cpu());

        /**
         * @brief Add a dependency between nodes
         * @param node_name The dependent node
         * @param depends_on The node that must complete first
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addDependency(const std::string &node_name,
                                    const std::string &depends_on);

        /**
         * @brief Get execution order respecting dependencies
         * @return Vector of node names in valid execution order
         */
        std::vector<std::string> getExecutionOrder() const;

        /**
         * @brief Get nodes that can execute in parallel (no unmet dependencies)
         * @return Vector of names of ready nodes
         */
        std::vector<std::string> getReadyNodes() const;

        /**
         * @brief Get a node by name
         * @param name Node identifier
         * @return Pointer to node (nullptr if not found)
         */
        ComputeNode *getNode(const std::string &name);
        const ComputeNode *getNode(const std::string &name) const;

        /**
         * @brief Mark a node as completed
         * @param name Node identifier
         */
        void markCompleted(const std::string &name);

        /**
         * @brief Reset all completion flags
         */
        void reset();

        /**
         * @brief Check if all nodes are completed
         */
        bool allCompleted() const;

        /**
         * @brief Get number of nodes
         */
        size_t size() const { return nodes_.size(); }

        /**
         * @brief Get total estimated FLOPs for all stages
         */
        size_t totalEstimatedFlops() const;

        /**
         * @brief Clear the graph
         */
        void clear();

        /**
         * @brief Merge another graph into this one
         *
         * Moves all nodes from the source graph into this graph.
         * If connect_from is specified, adds dependencies from nodes in the source
         * graph that have no dependencies to the connect_from node in this graph.
         *
         * @param other The graph to merge (will be emptied)
         * @param connect_from Optional node name to connect source graph roots to
         * @return Reference to this graph for chaining
         */
        ComputeGraph &merge(ComputeGraph &&other, const std::string &connect_from = "");

        /**
         * @brief Get the names of all root nodes (nodes with no dependencies)
         * @return Vector of root node names
         */
        std::vector<std::string> getRootNodes() const;

        /**
         * @brief Get the names of all leaf nodes (nodes with no dependents)
         * @return Vector of leaf node names
         */
        std::vector<std::string> getLeafNodes() const;

    private:
        std::vector<std::unique_ptr<ComputeNode>> nodes_;
        std::unordered_map<std::string, size_t> node_index_;
    };

    /**
     * @brief Orchestrates execution of compute stages within a graph
     *
     * GraphExecutor manages the execution of compute graphs by handling
     * topological ordering, device contexts, and execution modes.
     *
     * Usage:
     * @code
     * GraphExecutor executor(config);
     *
     * // Build a graph
     * ComputeGraph graph;
     * graph.addNode("norm", createRMSNormStage(...));
     * graph.addNode("proj", createGEMMStage(...));
     * graph.addDependency("proj", "norm");
     *
     * // Execute
     * executor.execute(graph, ctx);
     * @endcode
     */
    class GraphExecutor : public IGraphExecutor
    {
    public:
        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         */
        explicit GraphExecutor(const GraphExecutorConfig &config = GraphExecutorConfig());
        ~GraphExecutor() override;

        // Non-copyable, movable
        GraphExecutor(const GraphExecutor &) = delete;
        GraphExecutor &operator=(const GraphExecutor &) = delete;
        GraphExecutor(GraphExecutor &&) = default;
        GraphExecutor &operator=(GraphExecutor &&) = default;

        // =========================================================================
        // Configuration (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const GraphExecutorConfig &config() const override { return config_; }

        /**
         * @brief Set execution mode
         */
        void setExecutionMode(ExecutionMode mode) override { config_.mode = mode; }

        /**
         * @brief Get current execution mode
         */
        ExecutionMode executionMode() const override { return config_.mode; }

        /**
         * @brief Enable/disable profiling
         */
        void setProfilingEnabled(bool enabled) override { config_.enable_profiling = enabled; }

        /**
         * @brief Enable/disable output validation
         */
        void setValidationEnabled(bool enabled) override { config_.enable_validation = enabled; }

        /**
         * @brief Set snapshot callback for debugging
         */
        void setSnapshotCallback(StageSnapshotCallback callback) override { config_.snapshot_callback = std::move(callback); }

        /**
         * @brief Set the current layer context for stage dumping
         */
        void setCurrentLayerIdx(int layer_idx) override { config_.current_layer_idx = layer_idx; }

        /**
         * @brief Set the current iteration context for stage dumping
         */
        void setCurrentIteration(int iteration) override { config_.current_iteration = iteration; }

        /**
         * @brief Set the MPI rank for stage dumping
         */
        void setMPIRank(int rank) override { config_.mpi_rank = rank; }

        // =========================================================================
        // Execution (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx) override;

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of device_idx -> DeviceContext
         * @return true on success
         */
        bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<int, IDeviceContext *> &contexts) override;

        // =========================================================================
        // Statistics (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        const GraphExecutorStats &stats() const override { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() override { stats_.reset(); }

        // =========================================================================
        // Buffer Management
        // =========================================================================

        /**
         * @brief Set the buffer manager for managed execution
         *
         * When a buffer manager is set, executeWithBufferManagement() can be
         * used to automatically allocate buffers before execution.
         *
         * @param manager Buffer manager (not owned, must outlive executor)
         */
        void setBufferManager(GraphBufferManager *manager) { buffer_manager_ = manager; }

        /**
         * @brief Get the current buffer manager
         * @return Pointer to buffer manager (nullptr if not set)
         */
        GraphBufferManager *bufferManager() const { return buffer_manager_; }

        /**
         * @brief Execute a graph with automatic buffer management
         *
         * This method:
         * 1. Allocates all buffers via the buffer manager
         * 2. Executes the graph
         * 3. Leaves buffers allocated for retrieval
         *
         * Requires setBufferManager() to be called first.
         *
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool executeWithBufferManagement(ComputeGraph &graph, IDeviceContext *ctx);

    private:
        GraphExecutorConfig config_;
        GraphExecutorStats stats_;
        GraphBufferManager *buffer_manager_ = nullptr; ///< Optional buffer manager (not owned)

        // Internal execution helpers
        bool executeSequential(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeParallel(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeNode(ComputeNode &node, IDeviceContext *ctx);

        // Buffer validation (Debug/Integration builds only)
#if LLAMINAR_ASSERTIONS_ACTIVE
        /**
         * @brief Verify stage inputs before execution (ENTRY validation)
         *
         * Called automatically BEFORE stage execution when assertions are active.
         * Checks all INPUT buffers for null, NaN, or Inf data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_INPUTS=0.
         * Throws VerificationFailure on validation failure with full context.
         *
         * @param node The node whose inputs should be validated
         * @param layer_idx Current layer index for error reporting
         * @throws verification::VerificationFailure if validation fails
         */
        void verifyStageEntry(const ComputeNode &node, int layer_idx);

        /**
         * @brief Verify stage outputs after execution (EXIT validation)
         *
         * Called automatically AFTER stage execution when assertions are active.
         * Checks all OUTPUT buffers for null, NaN, Inf, or all-zero data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
         * Throws VerificationFailure on validation failure with full context.
         *
         * @param node The node whose outputs should be validated
         * @param layer_idx Current layer index for error reporting
         * @throws verification::VerificationFailure if validation fails
         */
        void verifyStageExit(const ComputeNode &node, int layer_idx);

        /**
         * @brief Validate stage outputs for zero/NaN tensors (legacy)
         *
         * Called automatically after stage execution when assertions are active.
         * Checks all OUTPUT buffers for uninitialized (zero) or corrupted (NaN/Inf) data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
         * NaN/Inf failure is enabled by default; zero-tensor warnings are not failures
         * unless LLAMINAR_FAIL_ON_ZERO=1 is set.
         *
         * @param node The node whose outputs should be validated
         * @return true if validation passes, false if errors detected
         *
         * @deprecated Use verifyStageExit() instead for exception-based validation
         */
        bool validateStageOutputs(const ComputeNode &node);
#endif

        // Workspace management
        std::vector<float> temp_buffer_;
        size_t temp_buffer_size_ = 0;

        float *getTemporaryBuffer(size_t elements);
    };

    // Backwards compatibility alias
    using LayerExecutor = GraphExecutor;

} // namespace llaminar2
