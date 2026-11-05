/**
 * @file PooledImmutableDemandFrontier.h
 * @brief Original protocol cut surrounding independent ordinary model work.
 *
 * Builders project actual source Publish, addressed seal-send or pure-reader
 * receive nodes. These declarations own no stream, allocation or live count.
 */
#pragma once
#include <string>
#include "execution/local_execution/graph/ComputeGraph.h"
#include <algorithm>
#include <vector>
#include <stdexcept>
namespace llaminar2 {
/** @brief Distinct original protocol roles; a pure reader publishes no demand. */
enum class PooledImmutableDemandRole { Source, InlineOwner, PureReader };
/** @brief Actual fixed protocol producer and first delayed response consumer. */
struct PooledImmutableDemandFrontier {
    PooledImmutableDemandRole role;
    std::string published; ///< Empty only for a pure query reader.
    std::string consume; ///< Original AcquireResponse or first response receive.
};
/** @brief Authenticate the actual source/inline/reader endpoint stage types before issuing a frame. */
void validatePooledDemandFrontier(const ComputeGraph& graph,const PooledImmutableDemandFrontier& frontier);
/** @brief Add only the consumed actual trunk join; inline protocol proof retains its original sole edge. */
void bindPooledDemandConsumption(ComputeGraph& graph,const PooledImmutableDemandFrontier& frontier,
    const std::vector<std::string>& trunk);
/**
 * @brief Exact-once graph-construction handoff around the original fixed protocol.
 *
 * This owns only immutable graph declarations. Existing source/reader plans
 * retain bytes, streams, one outstanding request and all runtime retirement.
 */
class PooledImmutableDemandHandoff final {
public:
    /** @brief Issue an actual stage-authenticated protocol frontier on its original source or reader graph. */
    PooledImmutableDemandHandoff(ComputeGraph& graph,PooledImmutableDemandFrontier frontier,
        std::uint64_t query,std::uint64_t hash)
        :graph_(&graph),boundary_(graph.retainedStorageBoundary()),domain_(graph.retainedParticipantDefinitionDomain()),
         frontier_(std::move(frontier)),query_(query),hash_(hash){
        if(!query_ || !hash_ || (!boundary_&&!domain_))throw std::invalid_argument("Pooled demand frame lost original source/reader authority");
        validatePooledDemandFrontier(graph,frontier_);
    }
    PooledImmutableDemandHandoff(const PooledImmutableDemandHandoff&)=delete;
    PooledImmutableDemandHandoff& operator=(const PooledImmutableDemandHandoff&)=delete;
    /** @return Actual source Publish or inline seal-send, empty for pure query readers. */
    const std::string& producer()const noexcept{return frontier_.published;}
    /** @brief Consume the exact original graph/query once, after every actual same-device trunk producer. */
    void consume(ComputeGraph& graph,std::uint64_t query,std::uint64_t hash,const std::vector<std::string>& trunk){
        if(consumed_ || &graph!=graph_ || graph.retainedStorageBoundary()!=boundary_ ||
           graph.retainedParticipantDefinitionDomain()!=domain_ || query!=query_ || hash!=hash_ || trunk.empty())
            throw std::invalid_argument("Pooled demand consumption changed original graph/source/query or was repeated");
        validatePooledDemandFrontier(graph,frontier_);
        const auto* receive=graph.getNode(frontier_.consume);
        std::vector<std::string> unique;
        for(const auto& name:trunk){
            const auto* node=graph.getNode(name);
            if(!node || node->device!=receive->device || name==frontier_.published || name==frontier_.consume ||
               std::find(unique.begin(),unique.end(),name)!=unique.end())
                throw std::invalid_argument("Pooled demand consumption requires distinct original trunk producers");
            unique.push_back(name);
            std::vector<std::string> pending{name},seen;bool found=frontier_.published.empty();
            while(!pending.empty()){
                auto current=std::move(pending.back());pending.pop_back();
                if(current==frontier_.consume)
                    throw std::invalid_argument("Pooled trunk cannot depend on its delayed response or protocol suffix");
                if(current==frontier_.published)found=true;
                if(std::find(seen.begin(),seen.end(),current)!=seen.end())continue;
                seen.push_back(current);const auto* actual=graph.getNode(current);
                if(actual)pending.insert(pending.end(),actual->dependencies.begin(),actual->dependencies.end());
            }
            if(!found)throw std::invalid_argument("Pooled trunk must follow its actual current demand prefix");
        }
        bindPooledDemandConsumption(graph,frontier_,trunk);consumed_=true;
    }
    /** @brief Require complete construction; reset/replay never changes this topology. */
    void requireConsumed()const{if(!consumed_)throw std::logic_error("Pooled current demand remains unconsumed");}
private:
    ComputeGraph* graph_;
    std::shared_ptr<IRetainedGraphStorageBoundary> boundary_;
    std::optional<RetainedGraphParticipantDefinitionDomain> domain_;
    PooledImmutableDemandFrontier frontier_;
    std::uint64_t query_,hash_;
    bool consumed_=false; ///< Cold construction state only.
};
}
