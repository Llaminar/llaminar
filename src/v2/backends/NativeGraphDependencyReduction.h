/**
 * @file NativeGraphDependencyReduction.h
 * @brief Setup-only simplification of native CUDA/HIP completion dependencies.
 *
 * Captured stream joins can retain several edges for the same happens-before
 * relation. Native graph compilers charge scheduling storage for those edges.
 * This pass removes only a full-completion edge proved by another entirely
 * full-completion path. It never changes a node, arithmetic, memory ownership,
 * stream priority, or special launch/programmatic dependency. Nested/control
 * nodes are opaque operations; their bodies keep their own compiler lifecycle.
 */
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2::detail
{
    /** @brief Whether an edge proves complete execution and memory visibility. */
    enum class NativeDependencyKind
    {
        FullCompletion,
        PreserveNative,
    };

    /** @brief One edge in an immutable, densely indexed native graph snapshot. */
    struct NativeGraphDependency
    {
        std::size_t source;
        std::size_t destination;
        NativeDependencyKind kind = NativeDependencyKind::FullCompletion;
    };

    /**
     * @brief Identify redundant full-completion edges without touching a device.
     * @param node_count Number of nodes; indices occupy [0, node_count).
     * @param edges Complete immutable native edge inventory, in arbitrary order.
     * @return Original edge indices removable together, sorted by input index.
     * @throws std::invalid_argument For malformed endpoints, duplicate ordinary
     * edges, unknown semantics, or cycles (including native-preserved edges).
     *
     * Reverse topological bitset propagation proves reachability using retained
     * edges only. Special edges survive unchanged and never supply a proof.
     * Temporary host storage is O(V^2 / 64 + E); work is O(E * V / 64) plus
     * adjacency sorting. No graph-size heuristic, device allocation or replay
     * work is involved.
     */
    std::vector<std::size_t> redundantNativeGraphDependencies(
        std::size_t node_count, std::span<const NativeGraphDependency> edges);

    /** @brief Cold compiler observation, not a live physical-memory ledger. */
    struct NativeGraphReductionResult
    {
        std::size_t nodes;
        std::size_t edges;
        std::size_t removed;
    };

    /**
     * @brief Apply the shared proof to one sealed, exclusively owned native DAG.
     * @tparam API Thin CUDA/HIP operations, including lossless edge metadata.
     * @param graph Exact uninstantiated owner; never a borrowed recording view.
     * @return Original geometry and number of eliminated native dependencies.
     * @throws std::runtime_error If native enumeration/mutation fails or changes
     * beneath the owner. The caller must reject instantiation after any failure.
     *
     * Node handles and original edge metadata survive verbatim. Only the owning
     * capture's instantiate boundary calls this operation, after composition and
     * before native executable construction. Child/control graph contents are
     * neither traversed nor rewritten through a borrowed handle.
     */
    template <typename API>
    NativeGraphReductionResult reduceNativeGraphDependencies(typename API::Graph graph)
    {
        using Node = typename API::Node;
        using EdgeData = typename API::EdgeData;
        const auto require = [](auto status, const char *operation) {
            if (status != API::success)
                throw std::runtime_error(std::string(operation) + ": " + API::errorString(status));
        };
        std::size_t node_count = 0;
        require(API::nodes(graph, nullptr, &node_count), "dependency reduction node count");
        std::vector<Node> nodes(node_count);
        auto observed = node_count;
        require(API::nodes(graph, nodes.data(), &observed), "dependency reduction nodes");
        if (observed != node_count)
            throw std::runtime_error("dependency reduction: node inventory changed");

        std::unordered_map<Node, std::size_t> indices;
        indices.reserve(node_count);
        for (std::size_t i = 0; i < node_count; ++i)
            if (!nodes[i] || !indices.emplace(nodes[i], i).second)
                throw std::runtime_error("dependency reduction: null or duplicate native node");

        std::size_t edge_count = 0;
        require(API::edges(graph, nullptr, nullptr, nullptr, &edge_count),
                "dependency reduction edge count");
        std::vector<Node> from(edge_count), to(edge_count);
        std::vector<EdgeData> data(edge_count);
        observed = edge_count;
        require(API::edges(graph, from.data(), to.data(), data.data(), &observed),
                "dependency reduction edges");
        if (observed != edge_count)
            throw std::runtime_error("dependency reduction: edge inventory changed");

        std::vector<NativeGraphDependency> dependencies;
        dependencies.reserve(edge_count);
        for (std::size_t i = 0; i < edge_count; ++i)
        {
            const auto source = indices.find(from[i]);
            const auto destination = indices.find(to[i]);
            if (source == indices.end() || destination == indices.end())
                throw std::runtime_error("dependency reduction: foreign edge endpoint");
            dependencies.push_back({source->second, destination->second, API::kind(data[i])});
        }
        const auto redundant = redundantNativeGraphDependencies(node_count, dependencies);
        if (redundant.empty()) return {node_count, edge_count, 0};

        // Compact forwards in place. Sorted original indices ensure no unread
        // edge is overwritten; special metadata is preserved byte for byte.
        for (std::size_t i = 0; i < redundant.size(); ++i)
        {
            from[i] = from[redundant[i]];
            to[i] = to[redundant[i]];
            data[i] = data[redundant[i]];
        }
        require(API::remove(graph, from.data(), to.data(), data.data(), redundant.size()),
                "dependency reduction remove edges");
        observed = 0;
        require(API::edges(graph, nullptr, nullptr, nullptr, &observed),
                "dependency reduction final edge count");
        if (observed != edge_count - redundant.size())
            throw std::runtime_error("dependency reduction: native mutation count mismatch");
        return {node_count, edge_count, redundant.size()};
    }
}
