/**
 * @file GraphComposer.h
 * @brief Typed, fail-fast declarative graph composition API over ComputeGraph.
 *
 * GraphComposer eliminates scattered string-based `addNode`/`addDependency`
 * wiring by returning typed @ref GraphNodeRef handles bound to a single
 * @ref ComputeGraph and requiring every node's dependencies, edge roles,
 * capture-wave contract, and ticket-unit contract to be declared together at
 * node creation time.
 */

#pragma once

#include "ComputeGraph.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Non-owning typed reference to an authenticated node inside one ComputeGraph.
     *
     * Carrying the owning graph pointer prevents accidentally wiring a node
     * name from one subgraph into another graph or passing misspelled raw
     * strings to dependency declarations.
     */
    class GraphNodeRef
    {
    public:
        GraphNodeRef() = default;

        /** @return True when bound to a non-null graph with a non-empty node identity. */
        [[nodiscard]] bool valid() const noexcept { return graph_ != nullptr && !name_.empty(); }
        /** @return True when @ref valid() holds. */
        explicit operator bool() const noexcept { return valid(); }

        /** @return Exact node name inside its owning graph. */
        [[nodiscard]] const std::string &name() const noexcept { return name_; }

        /** @return True iff this handle belongs to @p graph and is non-empty. */
        [[nodiscard]] bool belongsTo(const ComputeGraph &graph) const noexcept
        {
            return graph_ == &graph && !name_.empty();
        }

        bool operator==(const GraphNodeRef &other) const noexcept
        {
            return graph_ == other.graph_ && name_ == other.name_;
        }
        bool operator!=(const GraphNodeRef &other) const noexcept { return !(*this == other); }

    private:
        friend class GraphComposer;
        friend class ComputeGraph;

        GraphNodeRef(const ComputeGraph *graph, std::string name)
            : graph_(graph), name_(std::move(name)) {}

        const ComputeGraph *graph_ = nullptr;
        std::string name_;
    };

    /**
     * @brief Complete declarative specification for adding one node to a ComputeGraph.
     *
     * Every incoming dependency is categorized explicitly by its semantic role
     * (@ref data_deps, @ref order_deps, @ref host_submission_deps,
     * @ref collective_deps), making data flow, scratch ordering, host dispatch
     * order, and collective barriers auditable at the declaration site.
     */
    struct GraphNodeDecl
    {
        std::string name;                                        ///< Unique node name (duplicates fail hard).
        std::unique_ptr<IComputeStage> stage;                    ///< Concrete compute stage (must be non-null).
        DeviceId device = DeviceId::cpu();                       ///< Target execution device.
        std::vector<GraphNodeRef> data_deps;                     ///< Upstream data-flow producers.
        std::vector<GraphNodeRef> order_deps;                    ///< Same-device scratch/stream ordering edges.
        std::vector<GraphNodeRef> host_submission_deps;          ///< Manual host submission order after captured cutpoints.
        std::vector<GraphNodeRef> collective_deps;               ///< Symmetric collective ordering edges.
        std::optional<GraphCaptureWaveContract> capture_wave;    ///< Optional LocalTP capture-wave contract.
        std::optional<GraphHeterogeneousTicketUnitContract> ticket_unit; ///< Optional ticket-unit closure contract.
    };

    /**
     * @brief Strict, declarative composition facade bound to one ComputeGraph.
     *
     * All validation runs before graph mutation: duplicate node names, missing
     * producers, foreign-graph handles, cycles, and malformed edge roles fail
     * immediately with an exception rather than logging and dropping edges.
     */
    class GraphComposer
    {
    public:
        /** @brief Bind a composer to @p graph (graph must outlive this composer). */
        explicit GraphComposer(ComputeGraph &graph) noexcept : graph_(graph) {}

        /** @return Underlying ComputeGraph being composed. */
        [[nodiscard]] ComputeGraph &graph() noexcept { return graph_; }
        [[nodiscard]] const ComputeGraph &graph() const noexcept { return graph_; }

        /**
         * @brief Authenticate an already-existing node in this graph and return its typed handle.
         * @param existing_node Exact node name that must already exist in @ref graph_.
         * @return Valid @ref GraphNodeRef bound to @ref graph_.
         * @throws std::out_of_range If @p existing_node is empty or absent from @ref graph_.
         */
        [[nodiscard]] GraphNodeRef ref(const std::string &existing_node) const;

        /**
         * @brief Return an optional handle for @p existing_node if non-empty, authenticating presence.
         * @param existing_node Node name; empty yields std::nullopt.
         * @throws std::out_of_range If non-empty and absent from @ref graph_.
         */
        [[nodiscard]] std::optional<GraphNodeRef> optionalRef(const std::string &existing_node) const;

        /**
         * @brief Atomically validate and insert one node with all declared typed edges and contracts.
         * @param decl Complete declarative node specification.
         * @return Typed handle for the newly inserted node.
         * @throws std::invalid_argument On empty/duplicate name, null stage, foreign handle,
         *         missing producer, or invalid host-submission edge endpoints.
         */
        GraphNodeRef add(GraphNodeDecl decl);

        /**
         * @brief Add one strictly validated typed dependency between existing nodes in this graph.
         * @param consumer Existing dependent node handle.
         * @param producer Existing upstream producer handle.
         * @param kind Explicit semantic dependency role.
         * @throws std::invalid_argument On invalid/foreign handles, self-loop, cycle, or invalid edge role.
         * @throws std::out_of_range If either referenced node was removed from the graph.
         */
        void connect(const GraphNodeRef &consumer,
                     const GraphNodeRef &producer,
                     GraphDependencyKind kind = GraphDependencyKind::DataFlow);

        /**
         * @brief Assert that @p to does NOT transitively depend on @p from.
         * @param from Upstream node that must not precede @p to.
         * @param to Downstream node that must not wait on @p from.
         * @param diagnostic Precise failure message describing the violated architectural invariant.
         * @throws std::logic_error If a path from @p from to @p to exists.
         */
        void requireNoPath(const GraphNodeRef &from,
                           const GraphNodeRef &to,
                           const char *diagnostic) const;

        /**
         * @brief Assert that @p a and @p b are mutually independent in the DAG.
         * @param a First node handle.
         * @param b Second node handle.
         * @param diagnostic Precise failure message describing the violated independence invariant.
         * @throws std::logic_error If either node transitively depends on the other.
         */
        void requireIndependent(const GraphNodeRef &a,
                                const GraphNodeRef &b,
                                const char *diagnostic) const;

        /** @brief Authenticate @p handle belongs to @ref graph_ and still exists. */
        void requireMember(const GraphNodeRef &handle, const char *role) const;

    private:
        ComputeGraph &graph_;
    };

} // namespace llaminar2
