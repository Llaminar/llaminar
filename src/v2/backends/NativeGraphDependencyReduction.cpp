/**
 * @file NativeGraphDependencyReduction.cpp
 * @brief Device-free transitive reduction of full-completion graph edges.
 *
 * A retained edge is removed only when earlier retained successors already
 * prove its destination reachable. Successors are processed in topological
 * order and nodes in reverse topological order, so proofs cannot depend on an
 * edge that is later removed. Programmatic/launch edges are never completion
 * witnesses. Keeping that distinction prevents a scheduling optimization from
 * weakening producer publication or turning partial completion into readiness.
 */
#include "NativeGraphDependencyReduction.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace llaminar2::detail
{
    std::vector<std::size_t> redundantNativeGraphDependencies(
        std::size_t node_count, std::span<const NativeGraphDependency> edges)
    {
        if (edges.empty()) return {};
        std::vector<std::vector<std::size_t>> outgoing(node_count);
        std::vector<std::size_t> incoming(node_count, 0);
        for (std::size_t i = 0; i < edges.size(); ++i)
        {
            const auto &edge = edges[i];
            if (edge.source >= node_count || edge.destination >= node_count ||
                edge.source == edge.destination)
                throw std::invalid_argument("native graph dependency has invalid endpoints");
            if (edge.kind != NativeDependencyKind::FullCompletion &&
                edge.kind != NativeDependencyKind::PreserveNative)
                throw std::invalid_argument("native graph dependency has unknown semantics");
            outgoing[edge.source].push_back(i);
            ++incoming[edge.destination];
        }

        // Kahn's queue is also the topological order. Include special edges in
        // cycle validation, although they cannot prove full completion below.
        std::vector<std::size_t> order;
        order.reserve(node_count);
        for (std::size_t node = 0; node < node_count; ++node)
            if (incoming[node] == 0) order.push_back(node);
        for (std::size_t next = 0; next < order.size(); ++next)
            for (const auto index : outgoing[order[next]])
                if (--incoming[edges[index].destination] == 0)
                    order.push_back(edges[index].destination);
        if (order.size() != node_count)
            throw std::invalid_argument("native graph dependency inventory contains a cycle");

        std::vector<std::size_t> rank(node_count);
        for (std::size_t i = 0; i < order.size(); ++i) rank[order[i]] = i;
        bool has_completion_fork = false;
        for (auto &successors : outgoing)
        {
            // Non-completion edges stay in the native graph but are excluded
            // from the reachability proof, even if their endpoints also have
            // an ordinary path between them.
            std::erase_if(successors, [&](std::size_t i) {
                return edges[i].kind == NativeDependencyKind::PreserveNative;
            });
            std::sort(successors.begin(), successors.end(), [&](auto a, auto b) {
                return rank[edges[a].destination] < rank[edges[b].destination];
            });
            for (std::size_t i = 1; i < successors.size(); ++i)
                if (edges[successors[i - 1]].destination == edges[successors[i]].destination)
                    throw std::invalid_argument("native graph has duplicate full-completion edges");
            has_completion_fork |= successors.size() > 1;
        }
        if (!has_completion_fork) return {};

        // Store one bit per reachable node, not one byte or an O(V*E) DFS per
        // candidate edge. This is short-lived host compiler scratch only.
        constexpr std::size_t bits = std::numeric_limits<std::uint64_t>::digits;
        const std::size_t words = node_count / bits + (node_count % bits != 0);
        if (words != 0 && node_count > std::vector<std::uint64_t>().max_size() / words)
            throw std::length_error("native graph reachability geometry overflows host storage");
        std::vector<std::uint64_t> reachable(node_count * words, 0);
        std::vector<std::size_t> redundant;
        for (auto it = order.rbegin(); it != order.rend(); ++it)
        {
            auto *const row = reachable.data() + *it * words;
            for (const auto index : outgoing[*it])
            {
                const auto successor = edges[index].destination;
                const auto bit = std::uint64_t{1} << (successor % bits);
                if ((row[successor / bits] & bit) != 0)
                {
                    redundant.push_back(index);
                    continue;
                }
                row[successor / bits] |= bit;
                const auto *const child = reachable.data() + successor * words;
                for (std::size_t word = 0; word < words; ++word) row[word] |= child[word];
            }
        }
        // Native callers compact the original arrays using these indices.
        std::sort(redundant.begin(), redundant.end());
        return redundant;
    }
}
