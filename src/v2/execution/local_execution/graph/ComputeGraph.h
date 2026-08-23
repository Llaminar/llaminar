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
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{

    /**
     * @brief Domain-wide identity for one participant-local capture unit.
     *
     * Heterogeneous graphs may legitimately contain different work on sibling
     * devices.  Native CUDA/HIP capture is nevertheless a domain lifecycle:
     * every LocalTP participant must enter and leave the same ordered capture
     * waves. The identity names the shared wave and the typed participation
     * role says whether this graph records work or joins without work. A
     * passive follower names a later sibling-only wave that this graph must
     * likewise join.
     *
     * The contract is graph topology, not compute-stage behavior.  Keeping it
     * on the node lets model lowering declare role asymmetry without teaching a
     * generic kernel stage about LocalTP scheduling.
     */
    enum class GraphCaptureWaveParticipation : uint8_t
    {
        Active,  ///< Record and launch the node as part of this native wave.
        Passive, ///< Join the wave without recording device work for this node.
    };

    /**
     * @brief Immutable participation contract for one graph node and its followers.
     *
     * Passive node participation is admitted only when the concrete stage
     * certifies @ref IComputeStage::isPassiveGraphCaptureNoOp. Following waves
     * have no local graph node; they represent later sibling-only work between
     * this node and the next participant-common device operation.
     */
    struct GraphCaptureWaveContract
    {
        std::string identity; ///< Shared cross-participant identity for this wave.
        GraphCaptureWaveParticipation participation =
            GraphCaptureWaveParticipation::Active; ///< This graph's immutable role in the wave.
        std::vector<std::string> passive_following_identities; ///< Ordered sibling-only waves joined after the active segment launches.
    };

    /**
     * @brief State why one heterogeneous captured unit ends at a graph node.
     *
     * Authority and follower graphs attach the same identity to different
     * participant-local terminal nodes. The authority then executes its manual
     * CPU ticket segment while the follower waits at the next LocalTP capture
     * rendezvous. The final disposition instead names the last captured unit
     * without manufacturing an empty successor. This contract deliberately
     * does not expose the fine-grained mapped GPU packet waves inside a unit:
     * their multi-stream event DAG must remain in one native executable.
     */
    enum class GraphHeterogeneousTicketUnitDisposition : uint8_t
    {
        BeforeManualBoundary, ///< Close this unit and start another after participant-local manual work.
        TransactionTerminal, ///< Name the final unit without creating an empty successor.
    };

    /** @brief Identity and closure semantics for one heterogeneous captured unit. */
    struct GraphHeterogeneousTicketUnitContract
    {
        std::string identity; ///< Cross-participant identity of the captured unit being closed.
        GraphHeterogeneousTicketUnitDisposition disposition =
            GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary; ///< Whether another captured unit must follow.
    };

    /**
     * @brief Graph-wide native-capture envelope selected by declarative lowering.
     *
     * Ordinary graphs may use per-node wave contracts to describe explicitly
     * segmented, role-asymmetric capture. A device-owned timeline transaction
     * instead embeds every heterogeneous wait/publication edge directly in the
     * endpoint stream capture. Splitting such a graph would reintroduce host
     * dispatch between those edges, so all participants must record one native
     * executable and legacy per-wave annotations become documentation only.
     *
     * A heterogeneous ticket lifecycle is the deliberately segmented case.
     * Its authority graph publishes one fixed immutable ticket, executes the
     * explicitly manual participant, and imports the result into a following
     * captured device unit. Every same-domain sibling uses the follower
     * envelope: it performs no manual work, but closes corresponding captured
     * units at typed participant-local cutpoints. Keeping those roles distinct
     * makes a missing ticket boundary, an accidental sibling CPU stage, or an
     * indivisible sibling graph invalid before capture begins.
     * Neither role permits generic eager replay.
     */
    enum class GraphNativeCaptureEnvelope : uint8_t
    {
        Ordinary = 0, ///< Per-node capture-wave and segment contracts apply.
        DeviceOwnedTimelineTransaction, ///< One indivisible native executable owns all ordering edges.
        HeterogeneousTicketAuthorityTransaction, ///< Captured device units surround one or more authenticated host-ticket boundaries.
        HeterogeneousTicketFollowerTransaction, ///< A no-manual-work sibling closes matching captured units around authority tickets.
    };

    /**
     * @brief Return whether an envelope requires exactly one native executable.
     * @param envelope Declarative graph-wide capture lifecycle.
     * @return True only for an indivisible device-owned timeline.
     */
    [[nodiscard]] constexpr bool requiresSingleNativeExecutable(
        GraphNativeCaptureEnvelope envelope) noexcept
    {
        return envelope ==
               GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction;
    }

    /**
     * @brief Return whether an envelope requires an explicit ticketed boundary.
     * @param envelope Declarative graph-wide capture lifecycle.
     * @return True for either typed role in a heterogeneous ticket lifecycle.
     */
    [[nodiscard]] constexpr bool requiresHeterogeneousTicketSegmentation(
        GraphNativeCaptureEnvelope envelope) noexcept
    {
        return envelope ==
                   GraphNativeCaptureEnvelope::
                       HeterogeneousTicketAuthorityTransaction ||
               envelope ==
                   GraphNativeCaptureEnvelope::
                       HeterogeneousTicketFollowerTransaction;
    }

    /**
     * @brief Return whether this graph owns the manual ticket boundary.
     * @param envelope Declarative graph-wide capture lifecycle.
     * @return True only for the heterogeneous ticket authority role.
     */
    [[nodiscard]] constexpr bool ownsHeterogeneousTicketBoundary(
        GraphNativeCaptureEnvelope envelope) noexcept
    {
        return envelope ==
               GraphNativeCaptureEnvelope::
                   HeterogeneousTicketAuthorityTransaction;
    }

    /**
     * @brief Return whether this graph follows a sibling ticket authority.
     * @param envelope Declarative graph-wide capture lifecycle.
     * @return True only for the no-manual-work heterogeneous follower role.
     */
    [[nodiscard]] constexpr bool followsHeterogeneousTicketBoundary(
        GraphNativeCaptureEnvelope envelope) noexcept
    {
        return envelope ==
               GraphNativeCaptureEnvelope::
                   HeterogeneousTicketFollowerTransaction;
    }

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
        std::optional<GraphCaptureWaveContract> graph_capture_wave; ///< Optional cross-participant capture schedule contract.
        std::optional<GraphHeterogeneousTicketUnitContract>
            heterogeneous_ticket_unit_contract; ///< Optional captured-unit identity aligned across authority and follower.

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
         * @brief Attach an explicit LocalTP capture-wave contract to one node.
         *
         * @param node_name Existing graph node that owns the contract.
         * @param contract Non-empty identity, an explicit participation role,
         *        and zero or more distinct non-empty passive follower identities.
         * @return Reference to this graph for fluent construction.
         * @throws std::invalid_argument when the contract is malformed.
         * @throws std::out_of_range when @p node_name does not exist.
         *
         * Contracts are consumed only while building a native graph replay
         * plan. They do not add a compute operation or alter dependency order.
         */
        ComputeGraph &setGraphCaptureWaveContract(
            const std::string &node_name,
            GraphCaptureWaveContract contract);

        /**
         * @brief Close a heterogeneous captured unit after one graph node.
         *
         * The boundary changes capture topology, not graph dependency order.
         * It is consumed only by the two heterogeneous ticket envelopes and is
         * rejected for ordinary or indivisible device-owned graphs.
         *
         * @param node_name Existing participant-local terminal node.
         * @param contract Non-empty identity plus boundary or terminal role.
         * @return Reference to this graph for fluent construction.
         * @throws std::invalid_argument when the identity is empty.
         * @throws std::out_of_range when @p node_name does not exist.
         */
        ComputeGraph &setHeterogeneousTicketUnitContract(
            const std::string &node_name,
            GraphHeterogeneousTicketUnitContract contract);

        /**
         * @brief Select the typed native-capture lifecycle for this graph.
         *
         * Repeated selection of the same envelope is idempotent. Merging graph
         * fragments propagates the stronger envelope; incompatible future
         * non-ordinary envelopes must be rejected rather than guessed.
         *
         * @param envelope Declarative graph-wide native-capture contract.
         * @return Reference to this graph for fluent construction.
         */
        ComputeGraph &setNativeCaptureEnvelope(
            GraphNativeCaptureEnvelope envelope);

        /** @return Graph-wide native-capture contract. */
        [[nodiscard]] GraphNativeCaptureEnvelope nativeCaptureEnvelope() const noexcept
        {
            return native_capture_envelope_;
        }

        /**
         * @brief Get execution order respecting dependencies
         * @return Reference to cached vector of node names in valid execution order
         *
         * The execution order is computed via topological sort on the first call
         * and cached until the graph is modified (addNode, addDependency, merge, clear).
         */
        const std::vector<std::string> &getExecutionOrder() const;

        /**
         * @brief Return direct stage pointers in finalized execution order.
         *
         * The graph owns both the stages and this cached view. Graph mutation
         * invalidates the view alongside the topological order; repeated decode
         * and MTP publication therefore avoid rebuilding a temporary vector and
         * hashing every node name on each transaction.
         *
         * Null stage pointers remain represented in the returned vector so
         * execution/publication boundaries can fail hard with the exact index
         * instead of silently changing graph cardinality.
         *
         * @return Stable direct-pointer view valid until the next graph mutation.
         */
        const std::vector<IComputeStage *> &getExecutionStages();

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
         * @brief Discover every graph node whose concrete stage participates in a collective.
         *
         * Collective behavior is an instance-level stage contract. Composite
         * stages such as phase-split MoE maintenance may contain NCCL/RCCL
         * operations even though their broad `ComputeStageType` is not one of
         * the dedicated collective enum values. Keeping discovery on the graph
         * prevents forward, verifier, and auxiliary maintenance launchers from
         * inventing separate and eventually inconsistent collective-node lists.
         *
         * @return Node names for which `IComputeStage::isCollectiveStage()`
         *         reports true.
         */
        std::unordered_set<std::string> collectiveNodeNames() const;

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
        void setTerminalNode(const std::string &name)
        {
            if (terminal_node_ == name)
                return;
            terminal_node_ = name;
            noteTopologyMutation();
        }

        /**
         * @brief Return the monotonic identity of this graph's declarative topology.
         *
         * Captured executables retain stage pointers, dependency order, and buffer
         * addresses derived from one finalized graph.  A retained replay plan can
         * compare this scalar instead of re-walking names and edges on every launch;
         * any graph mutation makes the old executable identity stale immediately.
         * Completion flags and request data deliberately do not advance it.
         */
        [[nodiscard]] uint64_t topologyGeneration() const noexcept
        {
            return topology_generation_;
        }

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
        /** @brief Advance the capture identity after one declarative mutation. */
        void noteTopologyMutation();

        std::vector<std::unique_ptr<ComputeNode>> nodes_;
        std::unordered_map<std::string, size_t> node_index_;
        mutable std::vector<std::string> cached_order_; ///< Cached topological order
        mutable bool order_dirty_ = true;               ///< Invalidated on graph mutation
        std::vector<IComputeStage *> cached_execution_stages_; ///< Direct stage view in topological order.
        bool execution_stages_dirty_ = true;                    ///< Invalidated whenever graph topology/ownership changes.
        std::vector<FastScheduleEntry> fast_schedule_;   ///< Pre-computed decode schedule
        std::string terminal_node_;                      ///< Explicit terminal node (set by sub-graph builders)
        GraphNativeCaptureEnvelope native_capture_envelope_ =
            GraphNativeCaptureEnvelope::Ordinary;        ///< Typed graph-wide capture lifecycle selected by lowering.
        uint64_t topology_generation_ = 1;               ///< Monotonic captured-topology identity; zero is never published.
    };

} // namespace llaminar2
