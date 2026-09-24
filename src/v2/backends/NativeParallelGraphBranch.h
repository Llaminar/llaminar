/**
 * @file NativeParallelGraphBranch.h
 * @brief Shared CUDA/HIP cold assembly of one bounded parallel graph branch.
 *
 * The original graph is never cloned: CUDA conditional handles belong to that
 * exact native owner. Snapshot its complete root/terminal frontier before
 * adding three small child graphs. Only the terminal join waits for the worker;
 * the inference body must remain independent so it can publish Close.
 */
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace llaminar2::detail
{
    /** @brief Native error and the exact failed construction operation. */
    template <typename Error>
    struct NativeParallelGraphResult
    {
        Error error;
        const char *operation;
    };

    /**
     * @brief Add Open -> {body, worker}, body -> Close, {Close, worker} -> join.
     * @tparam API Exact vendor operations or a device-free graph test double.
     * @param body Non-empty, uninstantiated owner, retained in place.
     * @param fragments Open, worker and Close source graphs in that order.
     * @return Exact native status; failed partial assembly must be discarded.
     */
    template <typename API>
    NativeParallelGraphResult<typename API::Error> appendNativeParallelBranch(
        typename API::Graph body,
        const std::array<typename API::Graph, 3> &fragments)
    {
        using Node = typename API::Node;
        using Result = NativeParallelGraphResult<typename API::Error>;
        std::size_t count = 0;
        auto error = API::nodes(body, nullptr, &count);
        if (error != API::success) return Result{error, "query body nodes"};
        std::vector<Node> nodes(count);
        error = API::nodes(body, nodes.data(), &count);
        if (error != API::success) return Result{error, "read body nodes"};
        nodes.resize(count);
        std::vector<Node> roots;
        std::vector<Node> leaves;
        for (const auto node : nodes)
        {
            std::size_t degree = 0;
            error = API::dependencies(node, nullptr, &degree);
            if (error != API::success) return Result{error, "query body roots"};
            if (degree == 0) roots.push_back(node);
            degree = 0;
            error = API::dependents(node, nullptr, &degree);
            if (error != API::success) return Result{error, "query body terminals"};
            if (degree == 0) leaves.push_back(node);
        }
        if (roots.empty() || leaves.empty())
            return Result{API::invalid, "non-empty acyclic body frontier"};

        Node open{};
        error = API::child(&open, body, nullptr, 0u, fragments[0]);
        if (error != API::success) return Result{error, "append Open"};
        // Every original root consumes Open. Never depend on the worker here:
        // a bounded worker is allowed to wait for the original body's Close.
        std::vector<Node> opens(roots.size(), open);
        error = API::edges(body, opens.data(), roots.data(), roots.size());
        if (error != API::success) return Result{error, "order body after Open"};
        Node worker{};
        error = API::child(&worker, body, &open, 1u, fragments[1]);
        if (error != API::success) return Result{error, "append parallel worker"};
        Node close{};
        error = API::child(&close, body, leaves.data(), leaves.size(), fragments[2]);
        if (error != API::success) return Result{error, "append Close"};
        const Node terminals[] = {close, worker};
        Node join{};
        error = API::empty(&join, body, terminals, 2u);
        return Result{error, "join bounded worker retirement"};
    }
}
