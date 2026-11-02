/**
 * @file GraphLocalLookupFrame.h
 * @brief Exact-once construction contract for an issued current-invocation lookup.
 *
 * This lexical state names one graph and immutable declaration, never mutable
 * request values or device protocol decisions. The original stage/boundary
 * authenticates physical storage independently.
 */
#pragma once
#include "../../../local_execution/graph/ComputeGraph.h"
#include <stdexcept>
#include <utility>
namespace llaminar2 {
/** @brief A real Publish node's graph-local identity and one matching consumption. */
class GraphLocalLookupFrame final {
public:
    /** @brief Bind one actual producer after graph insertion and source retention. */
    GraphLocalLookupFrame(ComputeGraph& graph,std::string producer,std::uint64_t query,std::uint64_t hash)
        :graph_(&graph),source_(graph.retainedStorageBoundary()),producer_(std::move(producer)),query_(query),hash_(hash){
        if(!source_ || !graph.getNode(producer_) || !query_ || !hash_)
            throw std::invalid_argument("Lookup frame needs its actual retained source, Publish node and query identity");
    }
    GraphLocalLookupFrame(const GraphLocalLookupFrame&)=delete;
    GraphLocalLookupFrame& operator=(const GraphLocalLookupFrame&)=delete;
    /** @return Exact original publication dependency. */
    const std::string& producer() const noexcept { return producer_; }
    /**
     * @brief Validate all original declarations before issuing one delayed consumer.
     * @param graph Original graph retaining the exact source boundary.
     * @param query Complete immutable lookup declaration signature.
     * @param hash Exact token/history/count/hash-policy declaration signature.
     * @param trunk Existing independent trunk terminal producers.
     * @return Consumer dependencies containing every trunk terminal and Publish.
     * @throws std::invalid_argument Without changing state for foreign, repeated or incomplete consumption.
     */
    std::vector<std::string> consume(ComputeGraph& graph,std::uint64_t query,std::uint64_t hash,const std::vector<std::string>& trunk){
        if(&graph!=graph_ || graph.retainedStorageBoundary()!=source_ || state_!=State::Published ||
           query!=query_ || hash!=hash_ || !graph.getNode(producer_) || trunk.empty())
            throw std::invalid_argument("Lookup frame consume has foreign/consumed graph, source or query identity");
        for(const auto& node:trunk)if(node==producer_ || !graph.getNode(node))
            throw std::invalid_argument("Lookup Consume requires actual independent trunk readiness");
        auto dependencies=trunk;dependencies.push_back(producer_);
        state_=State::Consumed;return dependencies;
    }
    /** @brief Refuse a completed construction that did not consume its issued frame. */
    void requireConsumed() const { if(state_!=State::Consumed)throw std::logic_error("Issued lookup frame remains unconsumed"); }
private:
    enum class State { Published,Consumed };
    ComputeGraph* const graph_;
    const std::shared_ptr<IRetainedGraphStorageBoundary> source_;
    const std::string producer_;
    const std::uint64_t query_,hash_;
    State state_=State::Published;
};
}
