/**
 * @file CPUCanonicalShardRouteRecords.h
 * @brief Source-preserving validation of compact CPU expert down-partial records.
 *
 * The original rooted gather owns transport provenance. This helper consumes
 * its completed per-participant ranges and original router inputs, reconstructs
 * only selected CPU route mappings in admitted scratch, then invokes the same
 * fixed-order reducer used by dense expert-shard replay. It neither allocates
 * nor decides placement, and applies router weights only for an explicit borrowed compact output.
 * Its ordinary combine-only interface retains its original behavior.
 */
#pragma once
#include "CPUCanonicalRouteReducer.h"
#include "../../../collective/RootedFloatRecordRange.h"
#include <span>
#include <optional>
namespace llaminar2::cpu::moe {
/** @brief Original consumed activation entries; route slots retain their source router coordinates. */
struct CanonicalShardSparseRouteView {
    std::span<const int32_t> original_slots;
    std::span<const int32_t> expert_ids;
    std::span<const float> weights;
};
/** @brief Borrowed completed original gather and exact original logical CPU selection. */
struct CanonicalShardRouteRecordArgs {
    std::span<const float> records;
    std::span<const RootedFloatRecordRange> source_ranges;
    std::span<const float> route_indices;
    std::span<const float> route_weights;
    std::span<const int> selected_expert_ids;
    std::span<int32_t> participant_route_maps; ///< P * live slots; original-route scratch only.
    std::span<float> required_route_weights; ///< Live slots, 0/1 presence used by the combine.
    std::span<float> combined_route_rows; ///< Dense original route bank, unweighted.
    int original_expert_count=0;
    int rows=0;
    int top_k=0;
    int columns=0;
    std::optional<CanonicalShardSparseRouteView> sparse_routes; ///< Mutually exclusive with dense router inputs.
    std::optional<CanonicalRoutePreweightedOutput> preweighted_output; ///< Disjoint compact copy; slots/factor bytes must match the original router. Raw rows remain mandatory.
};
/**
 * @brief Validate all original members and route records before publishing combined rows.
 * @param args Actual completed compact inputs and caller-owned admitted scratch/output.
 * @return False without changing combined rows on malformed, missing or duplicate input.
 */
bool combineCanonicalShardRouteRecords(const CanonicalShardRouteRecordArgs &args);
} // namespace llaminar2::cpu::moe
