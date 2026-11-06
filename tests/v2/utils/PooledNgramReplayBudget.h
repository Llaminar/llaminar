/**
 * @file PooledNgramReplayBudget.h
 * @brief Setup-only benchmark projection through the original complete PMA planner.
 *
 * Functional policies are immutable inputs. This helper authors smaller optional
 * byte ceilings in their original ratios, then asks the caller's canonical
 * MemoryPlanner/ImmutableRowTablePlacementPlanner path to resolve every trial.
 * It never computes slot bytes, reserves memory, samples free capacity or grants
 * materialization. Complete original plans remain the sole physical byte law.
 */
#pragma once
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include <algorithm>
#include <concepts>
#include <numeric>
#include <limits>
#include <type_traits>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

namespace llaminar2::test {
/** @brief One benchmark policy and its actual canonical complete plan, not an admission receipt. */
struct PooledNgramReplayBudgetSelection {
    ImmutableRowTablePolicy requested;
    ImmutableRowTablePlacementSelection selected;
    std::size_t target_bytes=0, residual_bytes=0;
    std::optional<ImmutableRowTablePlacementSelection> next_larger;
    bool within_alignment_tolerance=false;
};
namespace pooled_budget_detail {
/** @brief Visit only active hierarchy optional caps; row ceilings/policies/ownership stay untouched. */
template<class Visit> void caps(ImmutableRowTablePolicy& p,Visit&& visit) {
    if(!p.isGPUAdaptive())throw std::invalid_argument("Matched pooled budget requires explicit GPU adaptive policy");
    auto& g=p.gpu_adaptive;
    if(g.source_backing!=AdaptiveImmutableRowBacking::RAM || g.host.maximum_cache_rows || g.host.cache_physical_byte_limit)
        throw std::invalid_argument("Matched RAM replay cannot change source backing or add an SSD RAM cache");
    if(g.placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal)visit(g.gpu_cache_physical_byte_limit);
    else if(g.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
        for(auto& owner:g.placement.owners)visit(owner.cache_physical_byte_limit);
    else throw std::invalid_argument("Matched pooled budget requires original ConsumerLocal or declared pool ownership");
    if(g.local_hot_cache) {
        visit(g.local_hot_cache->cache_physical_byte_limit);
        for(auto& owner:g.local_hot_cache->consumer_overrides)visit(owner.cache_physical_byte_limit);
    }
}
/** @brief Require exactly the same authored allocator observations, not equal used bytes or aggregate capacities. */
inline void sameCeilings(const PhysicalMemoryPlan& original,const PhysicalMemoryPlan& candidate) {
    if(original.resources().size()!=candidate.resources().size())
        throw std::invalid_argument("Matched pooled budget changed allocator population");
    for(const auto& line:original.resources()) {
        const auto& r=line.resource();const auto* other=candidate.find({r.world_rank,r.device});
        if(!other || !r.sameAllocator(other->resource()))
            throw std::invalid_argument("Matched pooled budget changed original allocator ceilings");
    }
}
}
/**
 * @brief Match aggregate actual physical bytes without replacing production admission.
 * @param profiles Existing canonical functional policies; never changed in place.
 * @param maximum_residual_bytes Explicit tolerated physical alignment residual.
 * @param resolve Rebuild the original policy-bound source request and full higher-priority
 * plan, then call ImmutableRowTablePlacementPlanner::select. Reusing a stale provision
 * after changing policy is invalid and is rejected by that production planner.
 * @return Each scaled policy, complete plan, common target and exact residual/next-step witness.
 * After three refinements an unmatched band is returned explicitly with
 * within_alignment_tolerance=false; it must never be labeled matched physical bytes.
 * @throws std::invalid_argument For changed ceilings, infeasible floors, or erased positive cache profiles.
 * @note The target starts at the least full-profile physical footprint and decreases only
 * to actual selected totals for at most three canonical refinements. No per-device used-byte
 * equality, native execution, timing or materialization permission is implied.
 */
template<class Resolve>
requires std::same_as<std::invoke_result_t<Resolve&,const ImmutableRowTablePolicy&>,ImmutableRowTablePlacementSelection>
std::vector<PooledNgramReplayBudgetSelection> projectPooledNgramReplayBudgets(
    std::span<const ImmutableRowTablePolicy> profiles,std::size_t maximum_residual_bytes,Resolve&& resolve) {
    if(profiles.size()<2)throw std::invalid_argument("Matched pooled budget requires multiple original policies");
    std::vector<ImmutableRowTablePlacementSelection> full;
    std::size_t target=std::numeric_limits<std::size_t>::max();
    for(const auto& policy:profiles) {
        full.push_back(resolve(policy));
        if(!full.back().physical_plan.fits())throw std::invalid_argument("Canonical pooled plan does not fit original PMA ceilings");
        pooled_budget_detail::sameCeilings(full.front().physical_plan,full.back().physical_plan);
        target=std::min(target,full.back().physical_plan.totalBytes());
    }
    // Directory and policy roots have real discontinuities. Search a bounded
    // intersection of actual plan totals rather than calling a megabyte jump
    // "alignment" or raising the user's physical residual tolerance.
    for(unsigned refinement=0;refinement<3;++refinement) {
        std::vector<PooledNgramReplayBudgetSelection> result;
        std::size_t next_target=target;
        bool within_tolerance=true;
        for(std::size_t i=0;i<profiles.size();++i) {
            auto original=profiles[i];std::size_t quantum=0;
            pooled_budget_detail::caps(original,[&](auto& cap){quantum=std::gcd(quantum,std::size_t(cap));});
            if(!quantum)throw std::invalid_argument("Matched pooled profile has no positive optional byte ceiling");
            const auto scaled=[&](std::size_t units) {
                auto policy=profiles[i];
                pooled_budget_detail::caps(policy,[&](auto& cap){cap=(cap/quantum)*units;});
                policy.validate();return policy;
            };
            auto policy=scaled(0);auto selected=resolve(policy);
            pooled_budget_detail::sameCeilings(full.front().physical_plan,selected.physical_plan);
            if(selected.physical_plan.totalBytes()>target)
                throw std::invalid_argument("Matched pooled physical target cannot admit an original mandatory floor");
            std::size_t low=0,high=quantum;
            while(low<high) {
                const auto middle=low+(high-low+1)/2;auto trial=resolve(scaled(middle));
                pooled_budget_detail::sameCeilings(full.front().physical_plan,trial.physical_plan);
                if(!trial.physical_plan.fits())throw std::invalid_argument("Canonical pooled trial exceeds original PMA ceilings");
                if(trial.physical_plan.totalBytes()<=target)low=middle;else high=middle-1;
            }
            if(!low)throw std::invalid_argument("Matched pooled physical target would erase a positive cache profile");
            policy=scaled(low);selected=resolve(policy);
            bool positive=false;
            for(const auto& owner:selected.placement->gpuAdaptiveLayout().owners())
                positive=positive || (owner.key.allocator.device.is_gpu() && owner.cache_rows);
            for(const auto& local:selected.placement->gpuAdaptiveLocalHotCaches())positive=positive || local.slots();
            if(!positive)throw std::invalid_argument("Matched pooled physical target has no admitted positive GPU cache");
            const auto actual=selected.physical_plan.totalBytes();
            if(actual>target)throw std::invalid_argument("Canonical pooled search exceeded its physical target");
            next_target=std::min(next_target,actual);
            within_tolerance=within_tolerance && target-actual<=maximum_residual_bytes;
            std::optional<ImmutableRowTablePlacementSelection> next;
            if(low<quantum) {
                auto larger=resolve(scaled(low+1));
                pooled_budget_detail::sameCeilings(full.front().physical_plan,larger.physical_plan);
                next=std::move(larger);
                if(next->physical_plan.totalBytes()<=target)throw std::invalid_argument("Canonical pooled budget search did not preserve monotonic physical fit");
            }
            result.push_back({std::move(policy),std::move(selected),target,target-actual,std::move(next),target-actual<=maximum_residual_bytes});
        }
        if(within_tolerance || refinement==2)return result;
        if(next_target>=target)throw std::invalid_argument("Canonical pooled physical intersection made no progress");
        target=next_target;
    }
    throw std::logic_error("Unreachable bounded pooled budget projection");
}
}
