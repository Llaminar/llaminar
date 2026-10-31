/**
 * @file PooledImmutableDirectoryLookup.h
 * @brief Read-only bounded Patricia lookup shared by the sole policy writer and parallel query producers.
 *
 * No reader mutates policy, failure state, pins or occupancy. The ordered caller
 * publishes a returned structural error; native parallel producers publish the
 * same error with an order-independent status CAS. Only fully Ready entries are
 * visible, and each walk is bounded by the original 64-bit row key.
 */
#pragma once
#include "PooledImmutableCoordinatorState.h"
namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/** @brief Pure lookup outcome; error is distinct from an ordinary absent row. */
struct DirectoryLookupResult { std::uint32_t entry;bool error; };
/** @return Exact read-only locator index or a structural failure without any mutation. */
QFN_ADAPTIVE_HD inline DirectoryLookupResult lookupDirectory(const DirectoryEntry* entries,const RadixNode* nodes,
    std::uint32_t capacity,std::uint32_t root,std::uint64_t row)
{
    if(!capacity)return {None,false};
    auto ref=root;
    for(unsigned steps=0;steps<=64;++steps){
        if(ref==None)return {None,false};
        if(ref&LeafTag){
            const auto at=ref&IndexMask;
            if(at>=capacity || !entries[at].present)return {None,true};
            return {entries[at].locator.row==row?at:None,false};
        }
        if(ref>=capacity-1 || nodes[ref].split_bit>=64)return {None,true};
        ref=nodes[ref].child[unsigned((row>>(63-nodes[ref].split_bit))&1)];
    }
    return {None,true};
}
}
