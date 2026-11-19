/**
 * @file ROCmNativeRuntimeBankView.h
 * @brief Exact existing GEMV runtime-bank ABI, shared with its fast-body adapter.
 *
 * Mechanical extraction only: every field, type, constant and offset assertion
 * below is retained from the original native GEMV translation unit. This view
 * owns no bytes, readiness, publication, epoch or descriptor certificate.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "execution/moe/DeviceMoEIntermediateShardBinding.h"
#include "execution/moe/DeviceMoEOverlayEpochABI.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
namespace llaminar2::rocm::native_runtime_abi {
constexpr int kDeviceMoEMaxExperts = llaminar2::moe_runtime_abi::kMaxExperts;
constexpr int kDeviceMoEMaxTopK = 16;
constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;

struct DeviceMoEExpertDescriptorView
{
    DeviceNativeQuantisedMatrixDesc gate;
    DeviceNativeQuantisedMatrixDesc up;
    DeviceNativeQuantisedMatrixDesc down;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_gate;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_up;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_down;
    int32_t logical_expert_id;
    int32_t owner_participant;
    int32_t local_slot;
    uint32_t flags;
    llaminar2::DeviceMoEWeightFormat weight_format;
    llaminar2::DeviceMoEWeightFormat floating_allocation_format;
    llaminar2::DeviceMoEProjectionSet projection_set;
};

static_assert(
    sizeof(DeviceMoEExpertDescriptorView) == 240,
    "ROCm GEMV expert descriptor view must match the shared runtime ABI");

static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_gate) == 168);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_up) == 184);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_down) == 200);
static_assert(offsetof(DeviceMoEExpertDescriptorView, logical_expert_id) == 216);
static_assert(offsetof(DeviceMoEExpertDescriptorView, weight_format) == 232);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_allocation_format) == 234);
static_assert(offsetof(DeviceMoEExpertDescriptorView, projection_set) == 236);

struct DeviceMoEPlacementBankView
{
    DeviceMoEExpertDescriptorView experts[kDeviceMoEMaxExperts];
    uint8_t local_compute_mask[kDeviceMoEMaxExperts];
    uint8_t replica_role[kDeviceMoEMaxExperts];
    uint32_t resident_participant_mask[kDeviceMoEMaxExperts];
    int32_t overlay_route_participant[kDeviceMoEMaxExperts];
    uint32_t epoch;
    uint32_t expert_count;
    uint32_t reserved[2];
};

static_assert(
    sizeof(DeviceMoEPlacementBankView) ==
    llaminar2::moe_runtime_abi::kPlacementBankBytes);
static_assert(
    offsetof(DeviceMoEPlacementBankView, overlay_route_participant) ==
    llaminar2::moe_runtime_abi::kOverlayRouteParticipantOffset);

struct DeviceMoELayerRuntimeView
{
    uint32_t active_bank;
    uint32_t active_epoch;
    uint32_t expert_count;
    uint32_t top_k;
    DeviceMoEPlacementBankView banks[2];
    int32_t topk_expert_ids[kDeviceMoEMaxTopK];
    float topk_weights[kDeviceMoEMaxTopK];
    uint64_t decode_histogram[kDeviceMoEMaxExperts];
    uint64_t decode_local_histogram[kDeviceMoEMaxExperts];
    uint64_t prefill_histogram[kDeviceMoEMaxExperts];
    uint64_t prefill_local_histogram[kDeviceMoEMaxExperts];
    uint64_t grouped_verifier_histogram[kDeviceMoEMaxExperts];
    uint64_t grouped_verifier_local_histogram[kDeviceMoEMaxExperts];
    uint64_t router_hot_cache_eligible_dispatches;
    uint64_t router_hot_cache_used_dispatches;
    uint64_t router_hot_cache_improved_dispatches;
    uint64_t router_hot_cache_default_load_spread_total;
    uint64_t router_hot_cache_actual_load_spread_total;
    uint64_t router_hot_cache_load_spread_improvement_total;
    uint64_t router_hot_cache_active_dispatches;
    uint64_t router_hot_cache_miss_dispatches;
    uint64_t router_hot_cache_selected_expert_slots;
    uint64_t router_hot_cache_replicated_selected_expert_slots;
    int32_t *route_expert_ids;
    float *route_weights;
    int32_t *route_participant_ids;
    int32_t *deferred_verifier_route_expert_ids;
    int32_t *deferred_verifier_route_participant_ids;
    int32_t *expert_counts;
    int32_t *expert_offsets;
    int32_t *grouped_token_ids;
    float *grouped_route_weights;
    float *grouped_gate_scratch;
    float *grouped_up_scratch;
    float *grouped_output_partials;
    void *decode_scratch;
    void *reserved_ptrs[3];
    uint64_t reserved_u64[4];
    uint32_t prefill_token_capacity;
    uint32_t prefill_route_capacity;
    uint32_t deferred_verifier_route_capacity;
    uint32_t participant_id;
    uint32_t participant_count;
    uint32_t current_batch_llep_movement_observed;
    uint32_t current_batch_llep_non_owner_assignment_observed;
    uint32_t current_batch_llep_transient_bank_active;
    llaminar2::moe_runtime_abi::DeviceMoERuntimeHistogramBank
        *runtime_histogram_banks;
    const uint32_t *runtime_histogram_active_bank;
    const llaminar2::DeviceMoEOverlayEpochTicket *overlay_epoch_ticket;
    const DeviceMoEPlacementBankView *overlay_placement_banks;
    const llaminar2::DeviceMoEOverlayEpochStatus *overlay_epoch_status;
    llaminar2::DeviceMoEIntermediateShardBinding intermediate_shard;
};

static_assert(offsetof(DeviceMoELayerRuntimeView, intermediate_shard) ==
    llaminar2::moe_runtime_abi::kIntermediateShardBindingOffset);
static_assert(
    sizeof(DeviceMoELayerRuntimeView) ==
    llaminar2::moe_runtime_abi::kLayerRuntimeBytes);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, route_participant_ids) ==
    llaminar2::moe_runtime_abi::kRouteParticipantIdsOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, participant_count) ==
    llaminar2::moe_runtime_abi::kParticipantCountOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_movement_observed) ==
    llaminar2::moe_runtime_abi::kCurrentBatchLLEPMovementObservedOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_non_owner_assignment_observed) ==
    llaminar2::moe_runtime_abi::
        kCurrentBatchLLEPNonOwnerAssignmentObservedOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_transient_bank_active) ==
    llaminar2::moe_runtime_abi::
        kCurrentBatchLLEPTransientBankActiveOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, runtime_histogram_banks) ==
    llaminar2::moe_runtime_abi::kRuntimeHistogramBanksOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, runtime_histogram_active_bank) ==
    llaminar2::moe_runtime_abi::kRuntimeHistogramActiveBankOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_epoch_ticket) ==
    llaminar2::moe_runtime_abi::kOverlayEpochTicketOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_placement_banks) ==
    llaminar2::moe_runtime_abi::kOverlayPlacementBanksOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_epoch_status) ==
    llaminar2::moe_runtime_abi::kOverlayEpochStatusOffset);

}
