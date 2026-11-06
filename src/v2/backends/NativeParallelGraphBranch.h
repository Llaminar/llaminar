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
#include <memory>

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
     * @brief Cold original-fragment to composed-parent exit mapping.
     *
     * A recording lifetime is replaced on reset/re-recording. Weak lifetime
     * borrows therefore reject native-handle or C++ address reuse without a
     * global nonce, source hash or mutable inference mirror. Repeated imports
     * are retained separately and cannot name an ambiguous producer frontier.
     */
    template<class Graph, class Node>
    class NativeCapturedFragmentFrontiers final
    {
    public:
        /** @brief Save one actual import after its original native lowering succeeds. */
        void remember(const void* original, const std::shared_ptr<const void>& lifetime,
                      Graph source, std::size_t source_nodes, std::span<const Node> exits)
        {
            entries_.push_back({original,lifetime,source,source_nodes,{exits.begin(),exits.end()}});
        }
        /** @return The unique exact-current exit in this parent, or an empty span on invalid ownership. */
        std::span<const Node> require(const void* original,
            const std::shared_ptr<const void>& lifetime, Graph source, std::size_t source_nodes) const
        {
            const Entry* found=nullptr;
            for(const auto& entry:entries_) {
                if(entry.original!=original)continue;
                if(found)return {};
                found=&entry;
            }
            if(!found || !lifetime || !source || !source_nodes || found->source!=source || found->source_nodes!=source_nodes ||
               !sameLifetime(found->lifetime,lifetime))return {};
            return found->exits;
        }
        /** @return Whether this exact auxiliary definition already belongs to this parent. */
        bool alreadyAttached(const void* original,const std::shared_ptr<const void>& lifetime,Graph source) const
        {
            for(const auto& entry:attached_)
                if(entry.original==original || (entry.source==source && sameLifetime(entry.lifetime,lifetime)))return true;
            return false;
        }
        /** @brief Retain the exact auxiliary identity only after native attachment succeeds. */
        void attached(const void* original,const std::shared_ptr<const void>& lifetime,Graph source)
        {attached_.push_back({original,lifetime,source,0,{}});}
        /** @brief Invalidate all associations when the original parent resets. */
        void clear() noexcept {entries_.clear();attached_.clear();}
    private:
        /** @brief One immutable original definition and its actual lowered parent nodes. */
        struct Entry {
            const void* original;
            std::weak_ptr<const void> lifetime;
            Graph source;
            std::size_t source_nodes;
            std::vector<Node> exits;
        };
        /** @return Same still-live original shared control block. */
        static bool sameLifetime(const std::weak_ptr<const void>& weak,
                                 const std::shared_ptr<const void>& expected)
        {
            const auto actual=weak.lock();
            return actual && expected && actual.get()==expected.get() &&
                !actual.owner_before(expected) && !expected.owner_before(actual);
        }
        std::vector<Entry> entries_,attached_;
    };

    /**
     * @brief Fork a finite body after an actual imported producer exit.
     * @param parent Original composed parent. @param child Same-device sealed finite definition.
     * @param producer Exits returned only by this parent's original-fragment mapping.
     * @return Original SDK status; later main nodes receive no dependency on this auxiliary.
     * The graph's normal terminal closure joins the new child. There is no edge
     * back to the producer fragment or to a subsequent LocalRead unit.
     */
    template<class API>
    NativeParallelGraphResult<typename API::Error> appendNativeFiniteBodyAfterFrontier(
        typename API::Graph parent, typename API::Graph child,
        std::span<const typename API::Node> producer)
    {
        using Result=NativeParallelGraphResult<typename API::Error>;
        using Node=typename API::Node;
        if(!parent || !child || parent==child || producer.empty())
            return Result{API::invalid,"exact finite producer/auxiliary ownership"};
        std::size_t count=0;
        auto error=API::nodes(parent,nullptr,&count);
        if(error!=API::success)return Result{error,"query composed parent"};
        std::vector<Node> nodes(count);
        error=API::nodes(parent,nodes.data(),&count);
        if(error!=API::success)return Result{error,"read composed parent"};
        nodes.resize(count);
        for(std::size_t i=0;i<producer.size();++i)
            if(!producer[i] || std::find(nodes.begin(),nodes.end(),producer[i])==nodes.end() ||
               std::find(producer.begin(),producer.begin()+i,producer[i])!=producer.begin()+i)
                return Result{API::invalid,"original producer exit belongs to this parent"};
        count=0;error=API::nodes(child,nullptr,&count);
        if(error!=API::success)return Result{error,"query finite auxiliary"};
        if(!count)return Result{API::invalid,"nonempty finite auxiliary"};
        Node appended{};
        error=API::child(&appended,parent,producer.data(),producer.size(),child);
        return Result{error,"fork finite auxiliary after original producer"};
    }

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
