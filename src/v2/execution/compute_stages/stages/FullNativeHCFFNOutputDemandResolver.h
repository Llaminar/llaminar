/**
 * @file FullNativeHCFFNOutputDemandResolver.h
 * @brief Cold optional-output demand from one contracted HC/router/FFN graph.
 *
 * The final graph and canonical BufferArena remain the only topology/binding
 * authorities. This resolver owns no tensor, allocation, pointer cache or live
 * state. It follows the executor's original ordered producer model, validating
 * actual read/inout contracts against exact tensor identities before deciding
 * whether an outside node consumes this region's value or an older/later one.
 */
#pragma once
#include "kernels/common/FullNativeHCFFNOutputDemand.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "memory/BufferArena.h"
#include "tensors/TensorClasses.h"
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2 {
/**
 * @brief Original optional semantic value retained when its producer is contracted.
 * @details Original producer is provenance, while retained_producer passed to the
 * resolver is the one execution node. Current full regions use zero-offset
 * tensor views. The original owner seals tensor shape/storage and the final
 * graph generation; this record neither fabricates a descriptor nor extends a
 * tensor's lifetime. Multiple roles cannot claim the same transfer owner.
 */
struct FullNativeHCFFNOutputBinding {
    FullNativeHCFFNOutputRole role; ///< Original optional semantic role.
    std::string original_producer; ///< Original graph producer, retained for diagnostics.
    const TensorBase* tensor; ///< Exact original tensor, equal to canonical arena lookup.
    BufferId buffer; ///< Original arena role, not sufficient without tensor equality.
    std::size_t byte_offset; ///< Original view offset; the admitted full region requires zero.
    std::size_t byte_extent; ///< Original authored extent, bounded by that tensor's storage.
};

/**
 * @brief Resolve immutable optional publications from actual final graph readers.
 * @param graph Final contracted participant graph, before native preparation.
 * @param arena Canonical registered tensor lookup; no materialization is performed.
 * @param retained_producer Exact retained HC/router/FFN execution-node name.
 * @param candidates Original typed optional tensor/value declarations.
 * @param observations Roles selected by the existing cold snapshot filter.
 * @return Actual outside-read demand union selected observations.
 * @throws std::invalid_argument For absent/conflicting bindings, ambiguous aliases,
 * unordered readers/writers, invalid roles, or an observation without a value.
 * @details Reads preceding the region use old state. A later pure overwrite is
 * a new producer; its later readers do not require the region's old mirror.
 * Inouts first consume the previous producer, then become a new one. Unlike an
 * incidental topological order, graph ancestry proves these storage lifetimes.
 * Root stage/native owners seal the result with their existing final graph,
 * snapshot epoch and tensor leases; later mutation requires their normal fatal
 * identity validation. There is no second mutable binding/generation cache here.
 */
inline FullNativeHCFFNOutputDemand resolveFullNativeHCFFNOutputDemand(
    const ComputeGraph& graph,const BufferArena& arena,const std::string& retained_producer,
    std::span<const FullNativeHCFFNOutputBinding> candidates,
    FullNativeHCFFNOutputDemand observations=FullNativeHCFFNOutputDemand::none())
{
    const auto fail=[](const std::string& why) {throw std::invalid_argument("Full HC/FFN output demand: "+why);};
    const auto& order=graph.getExecutionOrder();
    std::unordered_map<std::string,std::size_t> indices;
    for(std::size_t i=0;i<order.size();++i)indices.emplace(order[i],i);
    const auto found=indices.find(retained_producer);
    if(found==indices.end()||!graph.getNode(retained_producer)->stage)fail("missing retained producer");
    const auto producer=found->second;
    std::array<const FullNativeHCFFNOutputBinding*,8> roles{};
    auto available=FullNativeHCFFNOutputDemand::none();
    for(const auto& value:candidates) {
        if(!FullNativeHCFFNOutputDemand::validRole(value.role))fail("invalid output role");
        if(value.role==FullNativeHCFFNOutputRole::RouterProbabilities)
            fail("router probability observation belongs to the original workspace owner, not a tensor candidate");
        const auto role=static_cast<unsigned>(value.role);
        if(roles[role]||value.original_producer.empty()||!value.tensor||value.byte_offset||!value.byte_extent||
            value.byte_extent>value.tensor->size_bytes()||arena.getTensor(value.buffer)!=value.tensor||
            !value.tensor->transferStorageOwner())fail("conflicting or noncanonical original binding");
        for(const auto* previous:roles)if(previous&&
            previous->tensor->transferStorageOwner()==value.tensor->transferStorageOwner())
            fail("optional roles alias one transfer owner");
        roles[role]=&value;available=available.with(value.role);
    }
    if((observations.bits()&~available.bits())!=0)fail("observation has no original candidate");
    auto result=observations;
    // Ancestry, not tie order in a valid topological sort, establishes version ownership.
    const auto ancestor=[&](std::size_t before,std::size_t after) {
        if(before==after)return true;
        std::vector<std::string> pending{order[after]};std::unordered_set<std::string> visited;
        while(!pending.empty()) {
            auto name=std::move(pending.back());pending.pop_back();
            if(!visited.insert(name).second)continue;
            const auto* node=graph.getNode(name);
            if(!node||!node->stage)fail("missing node in final graph");
            for(const auto& dependency:node->dependencies) {
                if(dependency==order[before])return true;
                pending.push_back(dependency);
            }
        }
        return false;
    };
    constexpr auto absent=std::numeric_limits<std::size_t>::max();
    std::array<std::size_t,8> latest;latest.fill(absent);
    const auto matching=[&](BufferId id,const std::string& name) -> unsigned {
        const auto* tensor=dynamic_cast<const TensorBase*>(arena.getTensor(id));
        if(!tensor||!tensor->transferStorageOwner()) {
            for(const auto* value:roles)if(value&&id==value->buffer)
                fail("unresolved candidate binding at "+name);
            // An unrelated unmaterialized registry entry cannot justify an
            // output demand. Existing arena/native owners still forbid later
            // rebinding after this cold seal; no tensor is created here.
            return unsigned(roles.size());
        }
        for(unsigned role=0;role<roles.size();++role)if(const auto* value=roles[role]) {
            if(tensor->transferStorageOwner()!=value->tensor->transferStorageOwner())continue;
            if(tensor!=value->tensor||id!=value->buffer)fail("ambiguous tensor/BufferId alias at "+name);
            return role;
        }
        return unsigned(roles.size());
    };
    for(std::size_t index=0;index<order.size();++index) {
        const auto* node=graph.getNode(order[index]);
        if(!node||!node->stage)fail("missing declared stage");
        const auto contract=node->stage->bufferContract();
        if(index==producer) {
            // Candidate metadata is retained even after optional stores leave the
            // stage's concrete output contract; it remains this region's value.
            for(unsigned role=0;role<roles.size();++role)if(roles[role])latest[role]=index;
            continue;
        }
        const auto requireOrdered=[&] {
            if(!ancestor(index,producer)&&!ancestor(producer,index))fail("unordered alias at "+order[index]);
        };
        for(const auto& binding:contract.allArenaReads()) {
            const auto role=matching(binding.id,order[index]);
            if(role==roles.size())continue;
            requireOrdered();
            if(ancestor(index,producer))continue; // Original value before this region.
            if(latest[role]==absent||!ancestor(latest[role],index))fail("ambiguous read version at "+order[index]);
            if(latest[role]==producer)result=result.with(roles[role]->role);
        }
        // An activation cannot silently enter the immutable raw/prepared-weight
        // authority. That would bypass the actual arena read/version declaration.
        const auto rejectWeightAlias=[&](const ITensor* weight) {
            const auto* tensor=dynamic_cast<const TensorBase*>(weight);
            if(!tensor)return;
            for(const auto* value:roles)if(value&&tensor->transferStorageOwner()==value->tensor->transferStorageOwner())
                fail("optional activation is declared as an immutable weight at "+order[index]);
        };
        for(const auto* weight:contract.weight_tensors)rejectWeightAlias(weight);
        for(const auto& weight:contract.prepared_weights)rejectWeightAlias(weight.source_tensor);
        for(const auto& binding:contract.allWrites()) {
            const auto role=matching(binding.id,order[index]);
            if(role==roles.size())continue;
            requireOrdered();
            if(latest[role]!=absent&&!ancestor(latest[role],index))fail("unordered write versions at "+order[index]);
            latest[role]=index;
        }
    }
    return result;
}
} // namespace llaminar2
