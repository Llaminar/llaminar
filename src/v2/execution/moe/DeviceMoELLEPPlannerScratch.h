/**
 * @file DeviceMoELLEPPlannerScratch.h
 * @brief Shared graph-lifetime scratch ABI for parallel GPU LLEP planning.
 *
 * CUDA and ROCm launch one independent planner block for every layer in the
 * active maintenance wave. Those blocks publish immutable candidate plans to
 * this backend-neutral record. A later stream-ordered controller kernel walks
 * the records in wave order and is the sole command-buffer mutator, preserving
 * deterministic byte order without atomics or host coordination.
 */

#pragma once

#include "LeastLoadedExpertAssignment.h"

#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    inline constexpr uint32_t kDeviceMoELLEPPlannerMaxExperts = 256u;
    inline constexpr uint32_t kDeviceMoELLEPPlannerMaxParticipants = 8u;

    /**
     * @brief Device-published authentication summary for durable transfer slots.
     *
     * The maintenance controller must prove that every transfer-backed expert
     * descriptor still names the same durable directory entry before it can
     * publish another ownership wave. GPU backends compute this summary with a
     * parallel, allocation-free preflight kernel. Keeping the complete failure
     * witness in the graph workspace lets the serial command publisher remain
     * small without weakening fail-fast diagnostics.
     */
    struct DeviceMoETransferSlotClaimSummary
    {
        uint32_t transient_placement_layers = 0;
        uint32_t active_claims = 0;
        uint32_t unique_claims = 0;
        uint32_t duplicate_claims = 0;
        uint32_t invalid_claims = 0;
        uint32_t max_slot = 0;
        uint32_t max_slot_layer = 0;
        uint32_t max_slot_expert = 0;
        uint32_t first_duplicate_slot = 0;
        uint32_t first_duplicate_layer = 0;
        uint32_t first_duplicate_expert = 0;
        uint32_t first_invalid_slot = 0xffffffffu;
        uint32_t first_invalid_layer = 0xffffffffu;
        uint32_t first_invalid_expert = 0xffffffffu;
        uint32_t first_invalid_reasons = 0;
        uint32_t first_invalid_flags = 0;
        uint32_t first_invalid_resident_mask = 0;
        int32_t first_invalid_owner = -1;
    };

    /**
     * @brief Device-published router economy evidence for one maintenance edge.
     *
     * Counters are accumulated over the configured routed-layer window. The
     * GPU preflight owns any requested counter reset, so the command publisher
     * consumes one immutable summary instead of serially revisiting every
     * model layer.
     */
    struct DeviceMoERouterBenefitSummary
    {
        uint64_t eligible_dispatches = 0;
        uint64_t used_dispatches = 0;
        uint64_t improved_dispatches = 0;
        uint64_t default_load_spread_total = 0;
        uint64_t actual_load_spread_total = 0;
        uint64_t load_spread_improvement_total = 0;
        uint64_t active_dispatches = 0;
        uint64_t miss_dispatches = 0;
        uint64_t selected_expert_slots = 0;
        uint64_t replicated_selected_expert_slots = 0;
        uint32_t hot_cache_active_layers = 0;
        uint32_t invalid_runtime_layers = 0;
    };

    /**
     * @brief Persistent output for one independently planned LLEP wave layer.
     *
     * The allocation belongs to the graph-lifetime workspace. Every planner
     * block owns one disjoint element, and the publication kernel consumes the
     * resulting array only after stream ordering proves all blocks complete.
     */
    struct alignas(16) DeviceMoELLEPLayerPlanScratch
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t ready = 0;
        uint32_t planned = 0;
        uint32_t window_index = 0;
        uint32_t layer = 0;
        uint32_t claim_summary_ready = 0;
        uint32_t router_summary_ready = 0;
        DeviceMoETransferSlotClaimSummary claim_summary{};
        DeviceMoERouterBenefitSummary router_summary{};
        uint64_t assigned_participant_load[
            kDeviceMoELLEPPlannerMaxParticipants] = {};
        least_loaded_ep::LeastLoadedExpertAssignmentStatus planner_status{};
        least_loaded_ep::LeastLoadedExpertWeightTransfer
            transfers[kDeviceMoELLEPPlannerMaxExperts] = {};
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoETransferSlotClaimSummary>);
    static_assert(std::is_trivially_copyable_v<DeviceMoERouterBenefitSummary>);
    static_assert(std::is_trivially_copyable_v<DeviceMoELLEPLayerPlanScratch>);
} // namespace llaminar2
