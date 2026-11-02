/**
 * @file GPUFlatGraphInspection.h
 * @brief One allocation-free cold DAG census for actual CUDA/HIP and recorded API replay.
 *
 * The caller preclaims all scratch on the original CPU/PMA. The adapter issues
 * only read-only native graph queries; unsupported kinds and lossy edges fail
 * before instantiate. This code performs no execution, transfer or synchronization.
 */
#pragma once
#include "backends/GPUFlatGraphStructure.h"
#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
namespace llaminar2::detail {
/** @brief Inspect one immutable actual graph in fixed caller-owned storage.
 * @tparam API Exact source-selected backend adapter, or a recorded native-query double.
 * @param api Adapter bound to the original sealed definition and owner device.
 * @param limits Original source-authored node/kind/edge scope.
 * @param scratch Same-PMA bounded storage retained by the original graph owner.
 * @param observed Receives a complete census only on success; cleared on failure.
 * @param error Optional first precise failure; no per-node strings are constructed.
 * @return True only for a complete unique-node/default-edge DAG within scope.
 */
template<class API> bool inspectFlatGraph(API& api,const GPUFlatGraphLimits& limits,
    std::span<std::byte> scratch,GPUFlatGraphStructure& observed,std::string* error){
    using Node=typename API::Node;
    static_assert(std::is_pointer_v<Node> && sizeof(Node)==sizeof(std::uintptr_t));
    static_assert(alignof(Node)<=alignof(std::uintptr_t));
    observed={};if(error)error->clear();
    const auto fail=[&](const char* reason){if(error)*error=reason;return false;};
    if(!limits.valid())return fail("Flat graph inspection has invalid source scope");
    const auto layout=GPUFlatGraphScratchLayout::fromLimits(limits);
    if(scratch.size()<layout.bytes || !scratch.data() || reinterpret_cast<std::uintptr_t>(scratch.data())%alignof(std::uintptr_t))
        return fail("Flat graph inspection lacks its original bounded aligned scratch");
    auto* nodes=reinterpret_cast<Node*>(scratch.data()+layout.nodes);
    auto* from=reinterpret_cast<Node*>(scratch.data()+layout.from);
    auto* to=reinterpret_cast<Node*>(scratch.data()+layout.to);
    auto* degree=reinterpret_cast<std::uint32_t*>(scratch.data()+layout.degree);
    auto* queue=reinterpret_cast<std::uint32_t*>(scratch.data()+layout.queue);
    // Begin the trivial array lifetimes explicitly; no heap or per-node owner exists.
    std::uninitialized_value_construct_n(nodes,limits.maximumNodes);
    std::uninitialized_value_construct_n(from,limits.maximumEdges);
    std::uninitialized_value_construct_n(to,limits.maximumEdges);
    std::uninitialized_value_construct_n(degree,limits.maximumNodes);
    std::uninitialized_value_construct_n(queue,limits.maximumNodes);
    std::size_t count=0,edges=0;
    if(!api.nodes(nullptr,&count))return fail("Flat graph native node-count query failed");
    if(!count || count>limits.maximumNodes)return fail("Flat graph native nodes exceed original source scope");
    const auto expected=count;
    if(!api.nodes(nodes,&count) || count!=expected)return fail("Flat graph native node inventory changed");
    for(std::size_t i=0;i<count;++i){
        if(!nodes[i] || std::find(nodes,nodes+i,nodes[i])!=nodes+i)return fail("Flat graph has a null or duplicate native node");
    }
    if(!api.edges(nullptr,nullptr,&edges))return fail("Flat graph native edge-count query failed");
    if(edges>limits.maximumEdges)return fail("Flat graph native edges exceed original source scope");
    const auto expectedEdges=edges;
    if(edges && (!api.edges(from,to,&edges) || edges!=expectedEdges))return fail("Flat graph edge inventory changed or has non-default data");
    GPUFlatGraphStructure result;result.nodes=static_cast<std::uint32_t>(count);result.edges=static_cast<std::uint32_t>(edges);
    for(std::size_t i=0;i<count;++i){
        GPUFlatGraphNodeKind kind=GPUFlatGraphNodeKind::Count;
        if(!api.kind(nodes[i],kind) || kind>=GPUFlatGraphNodeKind::Count || !(limits.allowedKinds&GPUFlatGraphLimits::bit(kind)))
            return fail("Flat graph has an unreadable or unsupported native kind");
        ++result.kinds[static_cast<unsigned>(kind)];
    }
    const auto index=[&](Node node){return static_cast<std::size_t>(std::find(nodes,nodes+count,node)-nodes);};
    for(std::size_t i=0;i<edges;++i){
        const auto source=index(from[i]),target=index(to[i]);
        if(source==count || target==count || source==target)return fail("Flat graph has a foreign or self dependency");
        for(std::size_t prior=0;prior<i;++prior)if(from[prior]==from[i] && to[prior]==to[i])return fail("Flat graph has a duplicate dependency");
        if(degree[target]==std::numeric_limits<std::uint32_t>::max())return fail("Flat graph dependency degree overflow");
        ++degree[target];
    }
    result.linearChain=edges+1==count;
    for(std::size_t node=0;node<count && result.linearChain;++node){
        const auto outgoing=std::count(from,from+edges,nodes[node]);
        if(degree[node]>1 || outgoing>1)result.linearChain=false;
    }
    std::size_t head=0,tail=0;
    for(std::size_t i=0;i<count;++i)if(!degree[i])queue[tail++]=static_cast<std::uint32_t>(i);
    while(head<tail){
        const auto source=nodes[queue[head++]];
        for(std::size_t i=0;i<edges;++i)if(from[i]==source){const auto target=index(to[i]);if(!--degree[target])queue[tail++]=static_cast<std::uint32_t>(target);}
    }
    if(head!=count)return fail("Flat graph native dependencies contain a cycle");
    observed=result;return true;
}
}
