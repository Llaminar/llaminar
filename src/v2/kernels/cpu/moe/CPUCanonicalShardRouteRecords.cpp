/**
 * @file CPUCanonicalShardRouteRecords.cpp
 * @brief Complete compact-record validation followed by the canonical fixed-order combine.
 */
#include "CPUCanonicalShardRouteRecords.h"
#include "../../../execution/moe/CanonicalMoERouteRecord.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
namespace llaminar2::cpu::moe {
bool combineCanonicalShardRouteRecords(const CanonicalShardRouteRecordArgs &a) {
    const auto maximum=std::numeric_limits<size_t>::max();
    if (a.rows<0 || a.top_k<=0 || a.columns<=0 || a.original_expert_count<=0 ||
        a.source_ranges.size()<2 || a.source_ranges.size()>8) return false;
    const auto rows=static_cast<size_t>(a.rows),k=static_cast<size_t>(a.top_k),h=static_cast<size_t>(a.columns);
    if (rows>maximum/k || rows*k>maximum/h || rows*k*h>maximum/sizeof(float)) return false;
    const auto slots=rows*k,width=h+1,population=a.source_ranges.size();
    if (a.records.size()%width || !validRootedFloatRecordRanges(a.source_ranges,a.records.size()/width) ||
        (!a.sparse_routes && (a.route_indices.size()!=slots || a.route_weights.size()!=slots)) ||
        (a.sparse_routes && (!a.route_indices.empty() || !a.route_weights.empty() ||
            a.sparse_routes->original_slots.size()!=a.sparse_routes->expert_ids.size() ||
            a.sparse_routes->original_slots.size()!=a.sparse_routes->weights.size() ||
            a.sparse_routes->original_slots.size()>slots)) ||
        slots>maximum/population || a.participant_route_maps.size()<slots*population ||
        a.required_route_weights.size()<slots || a.combined_route_rows.size()<slots*h ||
        a.selected_expert_ids.empty()) return false;
    int last=-1;
    for (const int expert:a.selected_expert_ids) {
        if (expert<=last || expert>=a.original_expert_count) return false;
        last=expert;
    }
    // Scratch is mutable CPU execution state. Reject aliases before writing it;
    // a malformed map must never overwrite the immutable router or receive bank.
    const auto overlap=[](const void *x,size_t xb,const void *y,size_t yb) {
        if (!xb || !yb) return false;
        const auto u=reinterpret_cast<uintptr_t>(x),v=reinterpret_cast<uintptr_t>(y);
        if (!x || !y || xb>std::numeric_limits<uintptr_t>::max()-u ||
            yb>std::numeric_limits<uintptr_t>::max()-v) return true;
        return u<v+yb && v<u+xb;
    };
    const auto sparse=a.sparse_routes.value_or(CanonicalShardSparseRouteView{});
    const std::array<std::pair<const void*,size_t>,8> inputs{{
        {a.records.data(),a.records.size_bytes()}, {a.source_ranges.data(),a.source_ranges.size_bytes()},
        {a.route_indices.data(),a.route_indices.size_bytes()}, {a.route_weights.data(),a.route_weights.size_bytes()},
        {a.selected_expert_ids.data(),a.selected_expert_ids.size_bytes()},
        {sparse.original_slots.data(),sparse.original_slots.size_bytes()},
        {sparse.expert_ids.data(),sparse.expert_ids.size_bytes()},
        {sparse.weights.data(),sparse.weights.size_bytes()}}};
    const std::array<std::pair<const void*,size_t>,3> outputs{{
        {a.participant_route_maps.data(),slots*population*sizeof(int32_t)},
        {a.required_route_weights.data(),slots*sizeof(float)},
        {a.combined_route_rows.data(),slots*h*sizeof(float)}}};
    for (size_t i=0;i<outputs.size();++i) {
        if (outputs[i].second && !outputs[i].first) return false;
        for (const auto &input:inputs)
            if (overlap(outputs[i].first,outputs[i].second,input.first,input.second)) return false;
        for (size_t j=0;j<i;++j)
            if (overlap(outputs[i].first,outputs[i].second,outputs[j].first,outputs[j].second)) return false;
    }
    if(!a.sparse_routes) for (size_t slot=0;slot<slots;++slot) {
        const float index=a.route_indices[slot],weight=a.route_weights[slot];
        if (!std::isfinite(index) || index<0 || index>=static_cast<float>(a.original_expert_count) ||
            std::floor(index)!=index || !std::isfinite(weight)) return false;
    }
    if(a.sparse_routes) {
        int32_t prior=-1;
        for(size_t entry=0;entry<sparse.original_slots.size();++entry) {
            const auto slot=sparse.original_slots[entry],expert=sparse.expert_ids[entry];
            if(slot<=prior || static_cast<size_t>(slot)>=slots || expert<0 || expert>=a.original_expert_count ||
               !std::isfinite(sparse.weights[entry]) || sparse.weights[entry]==0.0f ||
               !std::binary_search(a.selected_expert_ids.begin(),a.selected_expert_ids.end(),expert))return false;
            prior=slot;
        }
    }
    std::fill_n(a.participant_route_maps.data(),slots*population,int32_t{-1});
    if(a.sparse_routes) {
        std::fill_n(a.required_route_weights.data(),slots,0.0f);
        for(const auto slot:sparse.original_slots)a.required_route_weights[slot]=1.0f;
    } else for (size_t slot=0;slot<slots;++slot)
        a.required_route_weights[slot]=a.route_weights[slot]!=0.0f &&
            std::binary_search(a.selected_expert_ids.begin(),a.selected_expert_ids.end(),
                               static_cast<int>(a.route_indices[slot]))?1.0f:0.0f;
    std::array<CanonicalRouteDownPartialRows,8> banks;
    for (size_t p=0;p<population;++p) {
        const auto range=a.source_ranges[p];
        if (range.record_count>slots || range.record_count>static_cast<size_t>(std::numeric_limits<int32_t>::max())) return false;
        auto map=a.participant_route_maps.subspan(p*slots,slots);
        const auto records=a.records.subspan(range.first_record*width,range.record_count*width);
        for (size_t record=0;record<range.record_count;++record) {
            const auto slot=canonical_moe_route_record::readFlatRouteSlot(records.data()+record*width,a.columns);
            if (slot>=slots || a.required_route_weights[slot]==0.0f || map[slot]!=-1) return false;
            map[slot]=static_cast<int32_t>(record);
        }
        for (size_t slot=0;slot<slots;++slot)
            if ((map[slot]>=0)!=(a.required_route_weights[slot]!=0.0f)) return false;
        banks[p]={records,map};
    }
    if (slots==0) return true;
    return combineCanonicalRouteDownPartialsOrdered({std::span(banks).first(population),
        a.combined_route_rows,a.required_route_weights.data(),a.rows,a.top_k,a.columns});
}
} // namespace llaminar2::cpu::moe
