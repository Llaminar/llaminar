/**
 * @file Test__NativeGraphDependencyReduction.cpp
 * @brief Device-free proof of the CUDA/HIP dependency simplification contract.
 *
 * Exhaustive typed DAGs compare the compiler with an independent reachability
 * oracle. Permuted node/edge identities, malformed inputs, long fork/join
 * graphs and native-enumeration failures cover cases hidden by one model's
 * convenient stream order. No device, model, timing threshold or profiler is
 * part of this functional gate.
 */
#include "backends/NativeGraphDependencyReduction.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <random>

using namespace llaminar2::detail;

namespace
{
    using Edge = NativeGraphDependency;
    constexpr auto full = NativeDependencyKind::FullCompletion;
    constexpr auto special = NativeDependencyKind::PreserveNative;

    /** @brief Independent Floyd-Warshall oracle; not the compiler's algorithm. */
    std::vector<bool> closure(std::size_t nodes, std::span<const Edge> edges)
    {
        std::vector<bool> reachable(nodes * nodes);
        for (const auto &edge : edges)
            if (edge.kind == full) reachable[edge.source * nodes + edge.destination] = true;
        for (std::size_t mid = 0; mid < nodes; ++mid)
            for (std::size_t from = 0; from < nodes; ++from)
                for (std::size_t to = 0; to < nodes; ++to)
                    reachable[from * nodes + to] = reachable[from * nodes + to] ||
                        (reachable[from * nodes + mid] && reachable[mid * nodes + to]);
        return reachable;
    }

    /** @brief Retain exactly the original edges not selected for deletion. */
    std::vector<Edge> retain(std::span<const Edge> edges, std::span<const std::size_t> removed)
    {
        std::vector<Edge> result;
        for (std::size_t i = 0; i < edges.size(); ++i)
            if (!std::binary_search(removed.begin(), removed.end(), i)) result.push_back(edges[i]);
        return result;
    }

    /** @brief Fake native graph for lossless adapter/error-path tests. */
    struct FakeGraph
    {
        std::vector<std::size_t> nodes{13, 7, 42};
        std::vector<Edge> edges{{13, 7}, {7, 42}, {13, 42}, {13, 42, special}};
        enum class Failure { None, Read, Remove } failure = Failure::None;
        std::size_t removals = 0;
    };

    /** @brief The same adapter surface used by the real native capture owners. */
    struct FakeAPI
    {
        using Graph = FakeGraph *;
        using Node = std::size_t;
        using EdgeData = NativeDependencyKind;
        static constexpr int success = 0;
        /** @brief Native status text stays in the propagated diagnostic. */
        static const char *errorString(int) { return "injected native error"; }
        /** @brief Enumerate stable native handles, not assumed dense indices. */
        static int nodes(Graph graph, Node *out, std::size_t *count)
        {
            if (out) std::copy(graph->nodes.begin(), graph->nodes.end(), out);
            *count = graph->nodes.size();
            return success;
        }
        /** @brief Return all edge annotations, including special duplicates. */
        static int edges(Graph graph, Node *from, Node *to, EdgeData *data, std::size_t *count)
        {
            if (graph->failure == FakeGraph::Failure::Read) return 1;
            if (from)
                for (std::size_t i = 0; i < graph->edges.size(); ++i)
                {
                    from[i] = graph->edges[i].source;
                    to[i] = graph->edges[i].destination;
                    data[i] = graph->edges[i].kind;
                }
            *count = graph->edges.size();
            return success;
        }
        /** @brief Keep typed native semantics without projecting them away. */
        static EdgeData kind(EdgeData data) { return data; }
        /** @brief Remove only exact source/destination/annotation matches. */
        static int remove(Graph graph, const Node *from, const Node *to,
                          const EdgeData *data, std::size_t count)
        {
            if (graph->failure == FakeGraph::Failure::Remove) return 1;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto match = std::find_if(graph->edges.begin(), graph->edges.end(), [&](const auto &edge) {
                    return edge.source == from[i] && edge.destination == to[i] && edge.kind == data[i];
                });
                if (match == graph->edges.end()) return 1;
                graph->edges.erase(match);
                ++graph->removals;
            }
            return success;
        }
    };
}

/** @test Every five-node DAG with absent/full/special edges preserves closure. */
TEST(NativeGraphDependencyReduction, ExhaustiveTypedDagsPreserveAllCompletionOrder)
{
    constexpr std::array<std::size_t, 5> permutation{3, 0, 4, 1, 2};
    constexpr std::size_t combinations = 59049; // 3^(5 choose 2)
    for (std::size_t pattern = 0; pattern < combinations; ++pattern)
    {
        auto code = pattern;
        std::vector<Edge> edges;
        for (std::size_t from = 0; from < permutation.size(); ++from)
            for (std::size_t to = from + 1; to < permutation.size(); ++to)
            {
                const auto type = code % 3; code /= 3;
                if (type != 0) edges.push_back({permutation[from], permutation[to], type == 1 ? full : special});
            }
        std::reverse(edges.begin(), edges.end());
        const auto removed = redundantNativeGraphDependencies(permutation.size(), edges);
        ASSERT_TRUE(std::is_sorted(removed.begin(), removed.end()));
        for (const auto index : removed) ASSERT_EQ(edges[index].kind, full);
        const auto retained = retain(edges, removed);
        ASSERT_EQ(closure(permutation.size(), edges), closure(permutation.size(), retained)) << pattern;
        ASSERT_TRUE(redundantNativeGraphDependencies(permutation.size(), retained).empty()) << pattern;
    }
}

/** @test Partial completion cannot witness a missing full-completion edge. */
TEST(NativeGraphDependencyReduction, SpecialPathsNeverAuthorizeDeletion)
{
    for (const auto edges : {std::vector<Edge>{{0, 1, special}, {1, 2}, {0, 2}},
                             std::vector<Edge>{{0, 1}, {1, 2, special}, {0, 2}},
                             std::vector<Edge>{{0, 1}, {1, 2}, {0, 2, special}}})
        EXPECT_TRUE(redundantNativeGraphDependencies(3, edges).empty());
}

/** @test Combined removal is minimal and independent of arbitrary identifiers. */
TEST(NativeGraphDependencyReduction, RandomNodeAndEdgePermutations)
{
    std::mt19937 random(73021);
    constexpr std::size_t nodes = 30;
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        std::vector<std::size_t> ids(nodes);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), random);
        std::vector<Edge> edges;
        for (std::size_t from = 0; from < nodes; ++from)
            for (std::size_t to = from + 1; to < nodes; ++to)
                if (random() % 4 == 0) edges.push_back({ids[from], ids[to], random() % 5 == 0 ? special : full});
        std::shuffle(edges.begin(), edges.end(), random);
        const auto removed = redundantNativeGraphDependencies(nodes, edges);
        const auto kept = retain(edges, removed);
        const auto original = closure(nodes, edges);
        ASSERT_EQ(original, closure(nodes, kept));
        for (std::size_t index = 0; index < kept.size(); ++index)
        {
            if (kept[index].kind == special) continue;
            auto one_less = kept;
            one_less.erase(one_less.begin() + index);
            ASSERT_NE(original, closure(nodes, one_less));
        }
    }
}

/** @test Long model-like fork/join chains reduce without recursion or topology caps. */
TEST(NativeGraphDependencyReduction, LargeForkJoinGraph)
{
    constexpr std::size_t layers = 4096;
    std::vector<Edge> edges;
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        const auto root = 3 * layer;
        edges.insert(edges.end(), {{root, root + 1}, {root, root + 2},
            {root + 1, root + 3}, {root + 2, root + 3}, {root, root + 3}});
    }
    const auto removed = redundantNativeGraphDependencies(3 * layers + 1, edges);
    ASSERT_EQ(removed.size(), layers);
    for (std::size_t i = 0; i < layers; ++i) EXPECT_EQ(removed[i], 5 * i + 4);
}

/** @test Reject malformed native inventories before any mutation is possible. */
TEST(NativeGraphDependencyReduction, InvalidInputsFailClosed)
{
    EXPECT_TRUE(redundantNativeGraphDependencies(0, {}).empty());
    EXPECT_TRUE(redundantNativeGraphDependencies(7, {}).empty());
    for (const auto &edges : {std::vector<Edge>{{0, 0}}, std::vector<Edge>{{0, 3}},
             std::vector<Edge>{{3, 0}}, std::vector<Edge>{{0, 1}, {0, 1}},
             std::vector<Edge>{{0, 1}, {1, 0}},
             std::vector<Edge>{{0, 1}, {1, 2}, {2, 0, special}},
             std::vector<Edge>{{0, 1, static_cast<NativeDependencyKind>(99)}}})
        EXPECT_THROW(redundantNativeGraphDependencies(3, edges), std::invalid_argument);
}

/** @test Lossless native compaction retains a special edge with identical endpoints. */
TEST(NativeGraphDependencyReduction, NativeAdapterRetainsMetadataAndIsIdempotent)
{
    FakeGraph graph;
    const auto result = reduceNativeGraphDependencies<FakeAPI>(&graph);
    EXPECT_EQ(result.nodes, 3);
    EXPECT_EQ(result.edges, 4);
    EXPECT_EQ(result.removed, 1);
    ASSERT_EQ(graph.edges.size(), 3);
    EXPECT_EQ(graph.edges.back().kind, special);
    EXPECT_EQ(reduceNativeGraphDependencies<FakeAPI>(&graph).removed, 0);
}

/** @test Invalid/stale native ownership and native API errors cannot be hidden. */
TEST(NativeGraphDependencyReduction, NativeAdapterRejectsBrokenInventory)
{
    FakeGraph graph;
    graph.nodes[0] = 0;
    EXPECT_THROW(reduceNativeGraphDependencies<FakeAPI>(&graph), std::runtime_error);
    graph = {}; graph.nodes[0] = graph.nodes[1];
    EXPECT_THROW(reduceNativeGraphDependencies<FakeAPI>(&graph), std::runtime_error);
    graph = {}; graph.edges[0].source = 100;
    EXPECT_THROW(reduceNativeGraphDependencies<FakeAPI>(&graph), std::runtime_error);
    graph = {}; graph.failure = FakeGraph::Failure::Read;
    EXPECT_THROW(reduceNativeGraphDependencies<FakeAPI>(&graph), std::runtime_error);
    graph = {}; graph.failure = FakeGraph::Failure::Remove;
    EXPECT_THROW(reduceNativeGraphDependencies<FakeAPI>(&graph), std::runtime_error);
    EXPECT_EQ(graph.removals, 0);
}
