/**
 * @file ComputeGraph.h
 * @brief Compute graph data structure (DAG of ComputeNodes)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Extracted from DeviceGraphExecutor.h to allow consumers that only need
 * graph/node types to avoid pulling in the full executor dependency tree.
 */

#pragma once

#include "../../compute_stages/ComputeStages.h"
#include "../../../backends/DeviceId.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{

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

        // =====================================================================
        // Coherence fast-path flags (mutable for use in const execution context)
        // Once weights are on-device and outputs are allocated, subsequent
        // iterations can skip the entire coherence check + vector extraction.
        // =====================================================================
        mutable bool weights_cohered = false; ///< Weights confirmed on device
        mutable bool is_final_output = false; ///< Outputs will be read by CPU (needs event sync)

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
         * @return Reference to cached vector of node names in valid execution order
         *
         * The execution order is computed via topological sort on the first call
         * and cached until the graph is modified (addNode, addDependency, merge, clear).
         */
        const std::vector<std::string> &getExecutionOrder() const;

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

        /**
         * @brief Set the terminal node name for this graph
         *
         * Sub-graph builders (buildAttentionGraph, buildFFNGraph) set this to
         * indicate which node is the logical output of the sub-graph. Callers
         * use this instead of getLeafNodes() + name search.
         *
         * @param name Node name that represents the sub-graph's terminal output
         */
        void setTerminalNode(const std::string &name) { terminal_node_ = name; }

        /**
         * @brief Get the terminal node name
         *
         * Returns the terminal node set by setTerminalNode(), or falls back to
         * getLeafNodes().front() if not explicitly set.
         *
         * @return Terminal node name, or empty string if graph is empty
         */
        std::string terminalNode() const
        {
            if (!terminal_node_.empty())
                return terminal_node_;
            auto leaves = getLeafNodes();
            return leaves.empty() ? std::string{} : leaves.front();
        }

        // =====================================================================
        // Fast Schedule — pre-computed flat array for zero-overhead decode loops
        // Eliminates string hash lookups, markCompleted calls, and virtual
        // type() dispatch from the per-token hot path.
        // =====================================================================

        struct FastScheduleEntry
        {
            ComputeNode *node;  ///< Direct pointer (no hash lookup needed)
            bool is_collective; ///< Pre-computed: ALLREDUCE / ALLGATHER / ALLGATHER_V
        };

        /**
         * @brief Build a pre-computed fast schedule from the execution order
         *
         * Must be called after the graph is finalized. The collective_nodes set
         * (if non-null) takes priority for collective classification; otherwise
         * falls back to stage->type().
         *
         * Also marks the last node as is_final_output for event-based dirty marking.
         */
        void buildFastSchedule(const std::unordered_set<std::string> *collective_nodes = nullptr);

        const std::vector<FastScheduleEntry> &fastSchedule() const { return fast_schedule_; }
        bool hasFastSchedule() const { return !fast_schedule_.empty(); }

    private:
        std::vector<std::unique_ptr<ComputeNode>> nodes_;
        std::unordered_map<std::string, size_t> node_index_;
        mutable std::vector<std::string> cached_order_; ///< Cached topological order
        mutable bool order_dirty_ = true;               ///< Invalidated on graph mutation
        std::vector<FastScheduleEntry> fast_schedule_;   ///< Pre-computed decode schedule
        std::string terminal_node_;                      ///< Explicit terminal node (set by sub-graph builders)
    };

} // namespace llaminar2
