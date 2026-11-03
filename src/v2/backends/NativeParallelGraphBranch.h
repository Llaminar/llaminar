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
#include <span>
#include <algorithm>

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
     * @brief Import finite siblings as independent roots/terminals of the SAME native parent.
     * @tparam API Original vendor graph operations or exact device-free DAG replay.
     * @param body Nonempty sealed native main body, preserved in place.
     * @param siblings Nonempty, unique graph-only finite bodies; none may wait for parent completion.
     * @return Original SDK status. Failure after first import invalidates the parent.
     * @note Native graph completion waits for every terminal. No artificial
     * Open/Close/progress operation is needed when native SDMA owns progress.
     */
    template<class API>
    NativeParallelGraphResult<typename API::Error> appendNativeFiniteBodySiblings(
        typename API::Graph body,std::span<const typename API::Graph> siblings)
    {
        using Result=NativeParallelGraphResult<typename API::Error>;
        using Node=typename API::Node;
        if(!body || siblings.empty())return Result{API::invalid,"nonempty finite body ownership"};
        std::size_t count=0;
        auto error=API::nodes(body,nullptr,&count);
        if(error!=API::success)return Result{error,"query original main body"};
        if(!count)return Result{API::invalid,"nonempty original main body"};
        // Validate the complete set before the first native mutation. Empty or
        // aliased definitions must not leave a partially decorated parent.
        for(std::size_t i=0;i<siblings.size();++i){
            if(!siblings[i] || siblings[i]==body ||
               std::find(siblings.begin(),siblings.begin()+i,siblings[i])!=siblings.begin()+i)
                return Result{API::invalid,"unique finite sibling ownership"};
            count=0;error=API::nodes(siblings[i],nullptr,&count);
            if(error!=API::success)return Result{error,"query finite sibling body"};
            if(!count)return Result{API::invalid,"nonempty finite sibling body"};
        }
        for(const auto sibling:siblings){
            Node node{};error=API::child(&node,body,nullptr,0u,sibling);
            if(error!=API::success)return Result{error,"append finite sibling root/terminal"};
        }
        return Result{API::success,"finite sibling roots/terminals complete"};
    }

    /**
     * @brief Add Open -> {body, worker}, body -> Close, {Close, worker} -> join.
     * @tparam API Exact vendor operations or a device-free graph test double.
     * @param body Non-empty, uninstantiated owner, retained in place.
     * @param fragments Open, worker and Close source graphs in that order.
     * @param body_siblings Finite graph-only work sharing Open and the body's Close frontier.
     *        A sibling must never require Close or parent completion to finish.
     * @return Exact native status; failed partial assembly must be discarded.
     */
    template <typename API>
    NativeParallelGraphResult<typename API::Error> appendNativeParallelBranch(
        typename API::Graph body,
        const std::array<typename API::Graph, 3> &fragments,
        std::span<const typename API::Graph> body_siblings = {})
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

        // Preserve the original body validation before any mutation. Siblings
        // are finite body work, not workers that wait for Close. Their terminal
        // must participate in Close so progress service cannot stop too early.
        for (std::size_t i = 0; i < body_siblings.size(); ++i)
            if (!body_siblings[i] || body_siblings[i] == body ||
                std::find(fragments.begin(), fragments.end(), body_siblings[i]) != fragments.end() ||
                std::find(body_siblings.begin(), body_siblings.begin() + i, body_siblings[i]) != body_siblings.begin() + i)
                return Result{API::invalid, "unique finite body siblings"};
        for (const auto sibling : body_siblings)
        {
            Node node{};
            error = API::child(&node, body, nullptr, 0u, sibling);
            if (error != API::success) return Result{error, "append finite body sibling"};
            roots.push_back(node);
            leaves.push_back(node);
        }

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
