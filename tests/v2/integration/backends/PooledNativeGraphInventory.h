/**
 * @file PooledNativeGraphInventory.h
 * @brief Cold, test-only inspection of the actual flat lookup graph before instantiation.
 *
 * This diagnostic deliberately performs read-only SDK queries and uses temporary
 * host storage. Its observations include that overhead and are never timing or
 * admission limits. Child, conditional, host and allocation nodes are outside
 * this closed lookup scope; an unsupported graph produces a failed receipt.
 */
#pragma once
#include "PooledNativeProgramObservation.h"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace llaminar2::test::pooled_program_observation {
/** @brief Emit one fixed-size inventory record bound to the actual instantiate invocation. */
void inventory(const char* backend,const char* operation,std::uint64_t ordinal,
    const void* graph,const char* observation,std::size_t nodes,std::size_t edges,
    std::size_t index,const void* node,const void* other,int kind,const char* error)noexcept;

/** @brief Inspect one immutable flat graph without changing its original API arguments or outcome.
 * @tparam API Exact selected SDK query adapter; no alternate native runtime is searched.
 * @param graph Original definition passed to the actual instantiate call.
 * @param ordinal Original instantiate observation ordinal, not a new ownership identity.
 */
template<class API> void inspectFlatLookupGraph(typename API::Graph graph,std::uint64_t ordinal)noexcept{
    using Node=typename API::Node;
    const auto emit=[&](const char* observation,std::size_t count,std::size_t edges,
                       std::size_t index=0,Node node=nullptr,Node other=nullptr,int kind=-1,const char* error=""){
        inventory(API::backend,API::operation,ordinal,graph,observation,count,edges,index,node,other,kind,error);
    };
    const auto fail=[&](const char* reason){emit("failed",0,0,0,nullptr,nullptr,-1,reason);};
    try{
        std::size_t count=0,edgeCount=0;
        if(API::nodes(graph,nullptr,&count)!=API::success){fail("node_count_query");return;}
        // These are diagnostic safety bounds, not support or PMA capacity limits.
        if(!count || count>65536){fail("flat_lookup_node_extent");return;}
        std::vector<Node> nodes(count);
        const auto capacity=count;
        if(API::nodes(graph,nodes.data(),&count)!=API::success || count!=capacity){fail("node_inventory_changed");return;}
        std::unordered_set<Node> unique;
        for(const auto node:nodes)if(!node || !unique.insert(node).second){fail("duplicate_or_null_node");return;}
        if(API::edges(graph,nullptr,nullptr,&edgeCount)!=API::success || edgeCount>1048576){fail("edge_count_query");return;}
        std::vector<Node> from(edgeCount),to(edgeCount);
        const auto edgeCapacity=edgeCount;
        if(edgeCount && (API::edges(graph,from.data(),to.data(),&edgeCount)!=API::success || edgeCount!=edgeCapacity)){
            fail("edge_inventory_changed_or_nondefault");return;
        }
        std::vector<int> kinds; kinds.reserve(count);
        for(const auto node:nodes){
            typename API::Kind kind{};
            if(API::kind(node,&kind)!=API::success){fail("node_kind_query");return;}
            if(!API::flatLookupKind(kind)){fail("node_kind_outside_flat_lookup");return;}
            kinds.push_back(static_cast<int>(kind));
        }
        for(std::size_t i=0;i<edgeCount;++i)if(!unique.contains(from[i]) || !unique.contains(to[i]) || from[i]==to[i]){
            fail("foreign_or_self_edge");return;
        }
        emit("begin",count,edgeCount);
        for(std::size_t i=0;i<count;++i)emit("node",count,edgeCount,i,nodes[i],nullptr,kinds[i]);
        for(std::size_t i=0;i<edgeCount;++i)emit("edge",count,edgeCount,i,from[i],to[i]);
        emit("complete",count,edgeCount);
    }catch(...){fail("host_inspection_failure");}
}
}
