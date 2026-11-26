/**
 * @file CUDAMoEKernelsCommon.cuh
 * @brief Shared CUDA MoE includes, runtime ABI view types, and inline device/launch helpers.
 */
#pragma once
/**
 * @file CUDAMoEKernels.cu
 * @brief CUDA launch bridges for MoE routing, grouping, and scatter/gather primitives.
 *
 * Alongside routing glue, fixed-tree floating expert projections retain the
 * cross-backend 256-lane arithmetic contract. Their grouped pipeline stores
 * SwiGLU once per intermediate and reuses existing arena scratch for the down
 * projection. Launch wrappers are C ABI functions consumed by CUDAMoEKernel.
 * Quantized gate/up borrows original token rows through the grouping map; it
 * never expands identical hidden activations into one copy per expert route.
 * Narrow SM86 Q5_1 Down publishes exact K32 contributions using independent
 * packed integer dots for the two halves and for the activation correction.
 * The SM86 verifier decodes selected weight planes four bytes at a time.
 * IQ4_XS physical format 24 preserves the original half base and raw six-bit
 * coefficient in separate 16-bit planes. Its LUT payload and source4 reduction
 * schedule are unchanged; the coefficient is storage metadata, never a minimum.
 *
 * Physical-half router loads round the original FP32 hidden to half RNE in
 * registers; all row modes reuse the same per-row FP32 accumulation/tree.
 * Four-expert tiles share hidden loads for prefill, while small row capacities
 * retain the measured lower-latency single-expert launch without new storage.
 * Shared rank-bank publication uses uniform direct copy/zero stores so
 * peer-bank zeroing owns no addressable private zero-vector temporary.
 * Independent decode slots share the serial normalization result and publish
 * routes in parallel, retaining live acquired-bank fragment authentication.
 * Decode descriptor publication touches only the selected packet; grouped
 * prefill retains its complete descriptor/list publication contract.
 * Fused decode emits Q8 directly from the ordered gate/up partition reducer.
 * SM86 reuses the original activation packet across paired Gate/Up projections
 * and vector-loads each K32 Down activation for the measured codebooks.
 * QFN decode specializes the paired Gate/Up schedule to five K32 packets
 * per partition, with both weight streams sharing register-held activations.
 * Ordered verifier tiles also resolve their measured pitch and packed
 * operand readers before capture and retain the serial split-K rounding tree.
 * The measured IQ1_S and IQ2_XXS tiles address the original activation
 * row directly from the validated route slot.
 * Wide SM86 verifier tiles expand a compact IQ1 grid and use whole-word
 * index and sign loads for IQ3. Compact IQ2 selectors combine magnitudes
 * and signs in registers. These launches resolve the physical pitch before K.
 * Serial Q5_1 resolves the measured ten-packet Down geometry before the dot.
 */

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "kernels/common/MoERuntimeDescriptorLaunch.h"
#include "CUDARouterQ8Prefill.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/__cccl/assert.h>

#include "CUDARetainedHCFFN.h"
#include "CUDARetainedGroupedFFN.h"
#include "CUDARetainedHCPrefix.cuh"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/cuda/gemm/CUDANativeQuantisedDecodeCommon.cuh"
#include "kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include "kernels/common/MoEIntermediateShardRowExecution.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include "execution/config/RoutedExpertPolicy.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/DeviceRouterHadamardQ8.h"
#include "kernels/common/RouterQ8Launch.h"
#include "kernels/common/DeviceOrderedRouteFold.h"
#include "kernels/common/MoEGroupedFloatingPrefillKernels.h"
#include "kernels/common/DeviceMoEFloatingWarp.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/MoERouterOwnedRows.h"
#include "kernels/common/MoERouterRowPacketDevice.inl"
#include "kernels/common/MoEGroupedSourceRows.h"
#include "kernels/common/DeviceIntegerBlockScan.h"
#include "kernels/common/MoEGroupingInitialization.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "kernels/common/DeviceSwiGLUNumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/cuda/moe/CUDAMoEBatchInvariantPolicy.h"
#include "kernels/cuda/moe/CUDAMoERouterPrefillPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoELLEPPlannerScratch.h"
#include "execution/moe/DeviceMoERebalanceABI.h"
#include "execution/moe/MoEGroupedPlanDemand.h"
#include "execution/moe/DeviceMoERebalancePublication.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/DeviceMoERebalanceMovementJournal.h"
#include "execution/moe/DeviceMoERebalanceCompletedDecision.h"
#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "execution/moe/DeviceMoEOverlayEpochABI.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "tensors/IQQuantTables.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cuda/atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
    namespace native_formats = llaminar2::native_quantised_formats;
    using DeviceMoELLEPLayerPlanScratchView =
        llaminar2::DeviceMoELLEPLayerPlanScratch;
    using DeviceMoETransferSlotClaimIndexView =
        llaminar2::DeviceMoETransferSlotClaimIndex;
    using DeviceMoETransferSlotClaimIndexEntryView =
        llaminar2::DeviceMoETransferSlotClaimIndexEntry;

    constexpr int kThreads = 256;
    static_assert(
        kThreads ==
            llaminar2::MoEProjectionNumericalContract::
                floating_ordered_k_partitions,
        "CUDA floating MoE launch geometry must retain the GPU-aligned expert tree");
    constexpr unsigned int kCudaGridYDimensionLimit = 65535u;

    /**
     * @brief Build a row-major CUDA grid with total positive-row coverage.
     *
     * CUDA's Y and Z launch dimensions are each limited to 65,535 blocks on
     * the supported devices. Production MoE work naturally reaches
     * `M * top_k` rows, so a 16K-token top-8 prefill already exceeds one Y
     * dimension. Kernels using this grid recover their logical row with
     * `logical_row_from_grid_yz()`; X remains available for column tiling.
     *
     * @param column_blocks Positive number of independent X tiles per row.
     * @param logical_rows Positive number of semantic rows to cover.
     * @return A legal three-dimensional launch geometry with no row holes.
     */
    dim3 make_row_major_grid(unsigned int column_blocks, int logical_rows)
    {
        assert(column_blocks > 0 && logical_rows > 0);
        const unsigned int rows = static_cast<unsigned int>(logical_rows);
        const unsigned int rows_y =
            std::min(rows, kCudaGridYDimensionLimit);
        const unsigned int rows_z = (rows + rows_y - 1u) / rows_y;
        assert(rows_z <= kCudaGridYDimensionLimit);
        return dim3(column_blocks, rows_y, rows_z);
    }

    /** @return The semantic row encoded across the Y/Z grid dimensions. */
    __device__ __forceinline__ int logical_row_from_grid_yz()
    {
        return static_cast<int>(
            blockIdx.y + blockIdx.z * static_cast<unsigned int>(gridDim.y));
    }
    /**
     * Number of scalar output columns owned by one canonical finalizer block.
     *
     * MTP verifier groups are deliberately short, so one block per token leaves
     * most GPU SMs idle at production hidden widths. A 256-column tile turns
     * Qwen3.6's 2,048-column row into eight independent output blocks while
     * retaining the same 256-thread gate-dot reduction in every block.
     */
    constexpr int kCanonicalMoEOutputTileColumns = 256;
    constexpr int kMaxExperts = 1024;
    constexpr int kDeviceMoEMaxExperts = llaminar2::moe_runtime_abi::kMaxExperts;
    constexpr uint32_t kDeviceMoEMaxTransferSlots = 0x7fffffffu;
    constexpr int kDeviceMoESmallGroupMaxSlots = 64;
    /**
     * Maximum route-table width owned by one runtime grouping block.
     *
     * The production MTP contract admits up to fifteen draft tokens plus the
     * target bonus row. Qwen's eight routed experts per token therefore require
     * at most 128 slots, while the extended M=31 regression requires 248. One
     * 256-thread block covers both without inter-block ordering or atomics.
     */
    constexpr int kDeviceMoERuntimeSmallGroupMaxSlots = 256;
    constexpr int kDeviceMoEMaxParticipants = 8;
    constexpr int kMaxTopK = 16;
    constexpr uint8_t kMixedCodebookSentinel = 0xffu;
    constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
    constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
    constexpr uint32_t kDeviceMoEFlagReplicated = 1u << 2;
    constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;
    constexpr uint32_t kDeviceMoEFlagTransferSlot = 1u << 5;
    constexpr uint32_t kDeviceMoEDirectoryFlagValid = 1u << 0;
    constexpr uint32_t kDeviceMoEDirectoryFlagResident = 1u << 1;
    constexpr uint32_t kDeviceMoEDirectoryFlagLocalCompute = 1u << 2;
    constexpr uint32_t kDeviceMoEDirectoryFlagTransferSlot = 1u << 3;
    constexpr uint32_t kDeviceMoEDirectoryFlagCopyComplete = 1u << 4;
    constexpr uint32_t kDeviceMoEReplicaRoleNone = 0u;
    constexpr uint32_t kDeviceMoEReplicaRolePrimary = 1u;
    constexpr uint32_t kDeviceMoEReplicaRoleReplica = 2u;
    constexpr uint32_t kDeviceMoERebalanceMagic = llaminar2::moe_rebalance_abi::kMagic;
    constexpr uint32_t kDeviceMoERebalanceVersion =
        llaminar2::moe_rebalance_abi::kVersion;
    constexpr uint32_t kDeviceMoERebalanceFlagHotReplicaCache = 1u << 0;
    constexpr uint32_t kDeviceMoERebalanceFlagResetHistograms = 1u << 1;
    constexpr uint32_t kDeviceMoERebalanceFlagPlanMissingArrivals = 1u << 2;
    constexpr uint32_t kDeviceMoERebalanceFlagDeferRuntimeApply = 1u << 3;
    constexpr uint32_t kDeviceMoERebalanceFlagCollectLoadStats = 1u << 4;
    constexpr uint32_t kDeviceMoERebalanceStatusOk = llaminar2::moe_rebalance_abi::kStatusOk;
    constexpr uint32_t kDeviceMoERebalanceStatusWindowNotReady = llaminar2::moe_rebalance_abi::kStatusWindowNotReady;
    constexpr uint32_t kDeviceMoERebalanceStatusInvalidConfig = 2u;
    constexpr uint32_t kDeviceMoERebalanceStatusInvalidRuntime = 3u;
    constexpr uint32_t kDeviceMoERebalanceStatusMissingHistogram = 4u;
    constexpr uint32_t kDeviceMoERebalanceStatusMissingTransferCompletion = 5u;
    constexpr uint32_t kDeviceMoERebalanceApplyStatusInvalidRuntime = 2u;
    constexpr uint32_t kDeviceMoERebalanceApplyStatusInProgress = 5u;
    constexpr uint32_t kDeviceMoECopyFailureInvalidStatusRecord = 1u << 0;
    constexpr uint32_t kDeviceMoECopyFailureNonOkStatusCode = 1u << 1;
    constexpr uint32_t kDeviceMoECopyFailureTransactionMismatch = 1u << 2;
    constexpr uint32_t kDeviceMoECopyFailureInvalidPlanEntry = 1u << 3;
    constexpr uint32_t kDeviceMoECopyFailureMissingSourceDescriptor = 1u << 4;
    constexpr uint32_t kDeviceMoECopyFailureMissingDestinationSlot = 1u << 5;
    constexpr uint32_t kDeviceMoECopyFailureDescriptorMismatch = 1u << 6;
    constexpr uint32_t kDeviceMoECopyFailureCopyIncomplete = 1u << 7;
    constexpr uint32_t kDeviceMoECopyFailureArrivalShortfall = 1u << 8;
    constexpr uint32_t kDeviceMoERebalancePlanExpertPayloadArrival = 1u;
    constexpr uint32_t kDeviceMoERebalancePlanResidentExpertAssignment = 2u;
    constexpr uint32_t kDeviceMoERebalancePlanOwnershipTransfer = 3u;
    constexpr uint32_t kDeviceMoERebalancePlanFlagCurrentBatchLLEP =
        llaminar2::moe_rebalance_abi::kPlanFlagCurrentBatchLLEP;
    constexpr uint32_t kDeviceMoERebalanceAssignmentLeastLoadedResident = 1u;
    constexpr uint32_t kDeviceMoERebalancePhasePlanAssignments = 2u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleIdle = 0u;
    constexpr uint32_t kDeviceMoERebalanceLifecyclePlanning = 1u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleTransferInFlight = 2u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleReadyToApply = 3u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleApplying = 4u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleApplied = 5u;
    constexpr uint32_t kDeviceMoERebalanceLifecycleError = 6u;
    constexpr uint32_t kDeviceMoEInvalidSlot = 0xffffffffu;
    /**
     * Internal claim value used while one device thread publishes first-error
     * provenance. Any non-zero controller error is terminal, so concurrent
     * graph consumers already stop while the winning publisher fills the
     * diagnostic fields and replaces this marker with the public status code.
     */
    constexpr uint32_t kDeviceMoERebalanceErrorPublicationInProgress =
        0xffffffffu;

    /**
     * @brief Apply one router weight with an explicit FP32 rounding boundary.
     *
     * Route publication may either retain each weighted row for a collective or
     * immediately add it to the final output.  Using the round-to-nearest
     * intrinsic here prevents the immediate path from contracting the multiply
     * into the following add while the retained path necessarily rounds at its
     * global-memory store.
     */
    __device__ __forceinline__ float moe_weight_route_rn(
        float route_weight,
        float expert_value)
    {
        return __fmul_rn(route_weight, expert_value);
    }

    /**
     * @brief Add one already-rounded contribution in canonical FP32 order.
     *
     * Keeping this operation explicit makes the route and K-part reduction tree
     * independent of compiler FMA contraction and kernel launch geometry.
     */
    __device__ __forceinline__ float moe_accumulate_rn(
        float accumulator,
        float contribution)
    {
        return __fadd_rn(accumulator, contribution);
    }

    __device__ __forceinline__ bool rebalance_plan_requires_payload(uint32_t op)
    {
        return op == kDeviceMoERebalancePlanExpertPayloadArrival ||
               op == kDeviceMoERebalancePlanOwnershipTransfer;
    }

    __device__ __forceinline__ bool rebalance_plan_applies_runtime(uint32_t op)
    {
        return rebalance_plan_requires_payload(op) ||
               op == kDeviceMoERebalancePlanResidentExpertAssignment;
    }

    using llaminar2::DeviceNativeQuantisedMatrixDesc;

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
        "CUDA expert descriptor view must match the shared runtime ABI");

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
        uint32_t multi_resident_expert_count;
        uint32_t transient_placement_observed;
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
        int32_t topk_expert_ids[kMaxTopK];
        float topk_weights[kMaxTopK];
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
        offsetof(DeviceMoELayerRuntimeView,
                 deferred_verifier_route_expert_ids) ==
        llaminar2::moe_runtime_abi::kDeferredVerifierExpertIdsOffset);
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

    /** Sentinel returned when an inference reader cannot prove its bank. */
    constexpr uint32_t kInvalidRuntimeExecutionBank = 0xffffffffu;

    /** @return Canonical durable placement banks for this runtime role. */
    __device__ __forceinline__ const DeviceMoEPlacementBankView *
    runtime_durable_placement_banks(const DeviceMoELayerRuntimeView *runtime)
    {
        return runtime && runtime->overlay_placement_banks
                   ? runtime->overlay_placement_banks
                   : (runtime ? runtime->banks : nullptr);
    }

    /** @return Placement banks visible to this exact captured child graph. */
    __device__ __forceinline__ const DeviceMoEPlacementBankView *
    runtime_placement_banks(const DeviceMoELayerRuntimeView *runtime)
    {
        if (!runtime)
            return nullptr;
        return runtime->current_batch_llep_transient_bank_active != 0u
                   ? runtime->banks
                   : runtime_durable_placement_banks(runtime);
    }

    /**
     * @brief Resolve the placement bank held by this request's epoch ticket.
     *
     * Non-overlay runtime tables explicitly leave the ticket null and retain
     * the legacy active-bank selector.  An overlay table must present a valid,
     * positive-generation ticket whose epoch still names the selected bank;
     * inference never substitutes the newer maintenance bank on failure.
     */
    __device__ __forceinline__ uint32_t runtime_execution_bank(
        const DeviceMoELayerRuntimeView *runtime)
    {
        if (!runtime)
            return kInvalidRuntimeExecutionBank;
        if (!runtime->overlay_epoch_ticket)
        {
            return runtime->active_bank <= 1u
                       ? runtime->active_bank
                       : kInvalidRuntimeExecutionBank;
        }

        const uint64_t epoch = runtime->overlay_epoch_ticket->epoch;
        const uint64_t selector = runtime->overlay_epoch_ticket->selector;
        const uint32_t bank = static_cast<uint32_t>(selector & 1u);
        const uint64_t generation = selector >> 1u;
        const DeviceMoEPlacementBankView *placement_banks =
            runtime_durable_placement_banks(runtime);
        if (epoch == 0u || generation == 0u || bank > 1u ||
            !placement_banks ||
            static_cast<uint64_t>(placement_banks[bank].epoch) != epoch)
        {
            return kInvalidRuntimeExecutionBank;
        }
        if (runtime->current_batch_llep_transient_bank_active != 0u)
        {
            const uint32_t transient_bank = runtime->active_bank;
            if (!runtime->overlay_placement_banks || transient_bank > 1u ||
                static_cast<uint64_t>(runtime->banks[transient_bank].epoch) !=
                    epoch)
            {
                return kInvalidRuntimeExecutionBank;
            }
            return transient_bank;
        }
        return bank;
    }

    /**
     * @brief Return the epoch that identifies a transfer transaction.
     *
     * Request-local LLEP children retain their canonical parent's durable
     * ticket while publishing private placement banks. Transfer slots must
     * therefore use the pinned ticket epoch, not the child's mutable
     * @c active_epoch metadata. Non-overlay tables retain their ordinary
     * participant-local placement epoch.
     */
    __device__ __forceinline__ uint32_t runtime_command_epoch(
        const DeviceMoELayerRuntimeView *runtime)
    {
        if (!runtime)
            return 0u;
        return runtime->overlay_epoch_ticket
                   ? static_cast<uint32_t>(
                         runtime->overlay_epoch_ticket->epoch)
                   : runtime->active_epoch;
    }

    /**
     * @brief Resolve one phase's writable selected-route counter array.
     *
     * External banks are model-lifetime state used by asynchronous overlay
     * maintenance. The embedded arrays remain the authority for CPU mirrors
     * and the graph-native same-domain rebalancer. A partially bound external
     * bank is a fatal ABI violation rather than permission to lose evidence.
     */
    __device__ __forceinline__ uint64_t *runtime_selected_histogram(
        DeviceMoELayerRuntimeView &runtime,
        uint32_t source)
    {
        if (runtime.runtime_histogram_banks ||
            runtime.runtime_histogram_active_bank)
        {
            if (!runtime.runtime_histogram_banks ||
                !runtime.runtime_histogram_active_bank ||
                source >= llaminar2::moe_runtime_abi::kHistogramSourceCount)
            {
                asm("trap;");
            }
            const uint32_t state = *runtime.runtime_histogram_active_bank;
            if ((state &
                 ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
                asm("trap;");
            const uint32_t bank =
                state &
                llaminar2::moe_runtime_abi::kHistogramWriterBankMask;
            return runtime.runtime_histogram_banks[bank].selected[source];
        }
        switch (source)
        {
        case 0u:
            return runtime.decode_histogram;
        case 1u:
            return runtime.prefill_histogram;
        case 2u:
            return runtime.grouped_verifier_histogram;
        default:
            asm("trap;");
            return nullptr;
        }
    }

    /** @brief Resolve one phase's writable locally-executed counter array. */
    __device__ __forceinline__ uint64_t *runtime_local_histogram(
        DeviceMoELayerRuntimeView &runtime,
        uint32_t source)
    {
        if (runtime.runtime_histogram_banks ||
            runtime.runtime_histogram_active_bank)
        {
            if (!runtime.runtime_histogram_banks ||
                !runtime.runtime_histogram_active_bank ||
                source >= llaminar2::moe_runtime_abi::kHistogramSourceCount)
            {
                asm("trap;");
            }
            const uint32_t state = *runtime.runtime_histogram_active_bank;
            if ((state &
                 ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
                asm("trap;");
            const uint32_t bank =
                state &
                llaminar2::moe_runtime_abi::kHistogramWriterBankMask;
            return runtime.runtime_histogram_banks[bank].local[source];
        }
        switch (source)
        {
        case 0u:
            return runtime.decode_local_histogram;
        case 1u:
            return runtime.prefill_local_histogram;
        case 2u:
            return runtime.grouped_verifier_local_histogram;
        default:
            asm("trap;");
            return nullptr;
        }
    }

    /** @return Whether the current external/embedded writer admits route rows. */
    __device__ __forceinline__ bool runtime_histogram_admits_rows(
        const DeviceMoELayerRuntimeView &runtime)
    {
        if (!runtime.runtime_histogram_active_bank)
            return true;
        const uint32_t state = *runtime.runtime_histogram_active_bank;
        if ((state &
             ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
        {
            asm("trap;");
        }
        return (state &
                llaminar2::moe_runtime_abi::kHistogramWriterQuarantineBit) ==
               0u;
    }

    /** @brief Clear one expert's exact production-phase demand counters. */
    __device__ __forceinline__ void reset_runtime_histogram_expert(
        DeviceMoELayerRuntimeView &runtime,
        uint32_t expert)
    {
        if (!runtime_histogram_admits_rows(runtime))
            return;
        for (uint32_t source = 0u;
             source < llaminar2::moe_runtime_abi::kHistogramSourceCount;
             ++source)
        {
            runtime_selected_histogram(runtime, source)[expert] = 0ULL;
            runtime_local_histogram(runtime, source)[expert] = 0ULL;
        }
    }

    /**
     * @brief Publish request-final LLEP evidence without reading live state on the host.
     *
     * One block scans the complete contiguous runtime table. Integer marker
     * counts use a fixed shared-memory tree, so publication is deterministic
     * and does not require atomics. Row zero owns this model-domain evidence;
     * later request rows are explicitly cleared to keep terminal aggregation
     * total for request-batched generation.
     */
    __global__ void
    publish_current_batch_llep_evidence_to_generation_control_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime_layers,
        int layer_count,
        int *__restrict__ generation_control,
        int generation_control_stride,
        int request_count,
        int movement_layer_count_index,
        int non_owner_assignment_layer_count_index)
    {
        constexpr int kEvidenceThreads = 256;
        __shared__ uint32_t movement_counts[kEvidenceThreads];
        __shared__ uint32_t non_owner_counts[kEvidenceThreads];

        uint32_t movement_count = 0;
        uint32_t non_owner_count = 0;
        for (int layer = static_cast<int>(threadIdx.x);
             layer < layer_count;
             layer += static_cast<int>(blockDim.x))
        {
            const auto &runtime = runtime_layers[layer];
            movement_count +=
                runtime.current_batch_llep_movement_observed != 0u ? 1u : 0u;
            non_owner_count +=
                runtime.current_batch_llep_non_owner_assignment_observed != 0u
                    ? 1u
                    : 0u;
        }

        movement_counts[threadIdx.x] = movement_count;
        non_owner_counts[threadIdx.x] = non_owner_count;
        __syncthreads();
        for (int offset = kEvidenceThreads / 2; offset > 0; offset >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < offset)
            {
                movement_counts[threadIdx.x] +=
                    movement_counts[threadIdx.x + offset];
                non_owner_counts[threadIdx.x] +=
                    non_owner_counts[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            for (int request = 0; request < request_count; ++request)
            {
                int *const control =
                    generation_control + request * generation_control_stride;
                control[movement_layer_count_index] =
                    request == 0
                        ? static_cast<int>(movement_counts[0])
                        : 0;
                control[non_owner_assignment_layer_count_index] =
                    request == 0
                        ? static_cast<int>(non_owner_counts[0])
                        : 0;
            }
        }
    }

    struct DeviceMoERebalanceConfigView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t num_layers;
        uint32_t num_experts;
        uint32_t top_k;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t root_participant;
        uint32_t window_size_tokens;
        uint32_t max_hot_replicas_per_participant;
        uint32_t layer_window_start;
        uint32_t layer_window_count;
        uint32_t layer_wave_count;
        uint32_t flags;
        uint32_t min_load_spread_improvement;
        uint32_t min_load_spread_improvement_divisor;
        uint32_t min_wave_spread_improvement_per_payload_slot;
        uint32_t min_foreign_rows_per_critical_path_payload_slot;
        uint32_t min_router_spread_improvement_per_payload_slot;
        uint32_t max_post_wave_load_spread_per_mille;
        uint32_t llep_alpha_numerator;
        uint32_t llep_alpha_denominator;
        uint32_t llep_lambda_numerator;
        uint32_t llep_lambda_denominator;
        uint32_t llep_enable_balanced_skip;
        uint32_t dynamic_imbalance_threshold_per_mille;
        uint32_t dynamic_min_improvement_per_mille;
        uint32_t dynamic_max_swaps_per_layer;
        uint32_t dynamic_max_plan_entries_per_wave;
        uint32_t dynamic_min_window_activations;
        uint32_t routed_assignment_policy;
        uint32_t active_transfer_slot_capacity;
        uint32_t transfer_slot_directory_capacity;
        uint32_t initial_maintenance_period_tokens;
        uint32_t maintenance_period_tokens;
    };
    static_assert(
        sizeof(DeviceMoERebalanceConfigView) ==
            llaminar2::moe_rebalance_abi::kConfigBytes,
        "CUDA rebalance config ABI must match the shared host/device record");

    struct DeviceMoERebalanceStatusView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t status_code;
        uint32_t last_epoch;
        uint32_t windows_observed;
        uint32_t windows_applied;
        uint32_t changed_layers;
        uint32_t selected_replicas;
        uint32_t skipped_not_ready;
        uint32_t skipped_no_resident;
        uint32_t invalid_runtime_layers;
        uint32_t planned_arrivals;
        uint32_t plan_overflow;
        uint32_t payload_bucket_requested_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t skipped_no_improvement;
        uint32_t dynamic_ownership_swap_attempts;
        uint32_t dynamic_ownership_swap_accepts;
        uint32_t dynamic_ownership_swap_rejections;
        uint32_t candidate_arrivals_considered;
        uint32_t candidate_arrivals_below_floor;
        uint32_t candidate_arrivals_pruned_by_count_bound;
        uint32_t skipped_busy_wave;
        uint32_t llep_assignment_span_count;
        uint32_t llep_weight_transfer_count;
        uint32_t llep_standard_ep_selected;
        uint32_t llep_skipped_balanced;
        uint32_t llep_skipped_insufficient_spread_improvement;
        uint32_t llep_skipped_insufficient_foreign_rows;
        uint32_t llep_min_chunk_skips;
        uint32_t llep_forced_spills;
        uint64_t candidate_load_spread_improvement_total;
        uint64_t candidate_load_spread_improvement_max;
        uint64_t accepted_load_spread_improvement_total;
        uint64_t accepted_load_spread_improvement_max;
        uint64_t llep_native_rows;
        uint64_t llep_spilled_rows;
        uint64_t llep_required_spread_improvement;
        uint64_t llep_required_foreign_rows;
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
        uint64_t pre_policy_load_total;
        uint64_t pre_policy_load_min;
        uint64_t pre_policy_load_max;
        uint64_t pre_policy_imbalance_numerator;
        uint64_t pre_policy_imbalance_denominator;
        uint64_t post_policy_load_total;
        uint64_t post_policy_load_min;
        uint64_t post_policy_load_max;
        uint64_t post_policy_imbalance_numerator;
        uint64_t post_policy_imbalance_denominator;
        uint64_t pre_policy_participant_load[kDeviceMoEMaxParticipants];
        uint64_t post_policy_participant_load[kDeviceMoEMaxParticipants];
        uint32_t window_ready_slots;
        uint32_t window_required_slots;
        uint32_t skipped_wave_cost_floor;
        uint32_t skipped_low_router_benefit;
        uint32_t skipped_post_load_spread_ceiling;
        uint32_t payload_source_participant_mask;
        uint32_t payload_destination_participant_mask;
        uint32_t prefill_current_batch_movement_layers;
        uint64_t payload_edge_mask;
        uint64_t pre_wave_load_total;
        uint64_t pre_wave_load_spread;
        uint64_t post_wave_load_total;
        uint64_t post_wave_load_spread;
        uint32_t skipped_participant_load_spread;
        uint32_t skipped_aggregate_load_spread;
        uint32_t skipped_configured_load_spread_ceiling;
        uint32_t capacity_limited_candidates;
        uint32_t prefill_active_transfer_slot_experts;
        uint32_t prefill_unique_transfer_slot_claims;
        uint32_t prefill_duplicate_transfer_slot_claims;
        uint32_t prefill_invalid_transfer_slot_claims;
        uint32_t prefill_max_transfer_slot;
        uint32_t prefill_max_transfer_slot_layer;
        uint32_t prefill_max_transfer_slot_expert;
        uint32_t prefill_first_duplicate_transfer_slot;
        uint32_t prefill_first_duplicate_layer;
        uint32_t prefill_first_duplicate_expert;
        uint32_t prefill_first_invalid_transfer_slot;
        uint32_t prefill_first_invalid_layer;
        uint32_t prefill_first_invalid_expert;
        uint32_t prefill_first_invalid_reasons;
        uint32_t prefill_first_invalid_flags;
        uint32_t prefill_first_invalid_resident_mask;
        int32_t prefill_first_invalid_owner;
        uint32_t prefill_current_batch_non_owner_assignment_layers;
    };
    static_assert(
        sizeof(DeviceMoERebalanceStatusView) ==
            llaminar2::moe_rebalance_abi::kStatusBytes,
        "CUDA rebalance status ABI must match the shared host/device record");
    static_assert(
        offsetof(DeviceMoERebalanceStatusView,
                 prefill_current_batch_non_owner_assignment_layers) ==
            llaminar2::moe_rebalance_abi::
                kPrefillCurrentBatchNonOwnerAssignmentLayersOffset,
        "CUDA resident-assignment evidence must occupy the shared status tail");

    /**
     * @brief Terminate CUDA execution with the exact violated LLEP invariant.
     *
     * A bare PTX `trap` correctly made an invalid post-transfer publication
     * fatal, but CUDA reported every such failure as an anonymous
     * `cudaErrorLaunchFailure`. That erased the distinction between an
     * incomplete collective, a non-resident assignment, and malformed grouped
     * route metadata precisely when the context became unusable.
     *
     * Calling CUDA's device assertion primitive directly preserves the required
     * fail-fast behavior in optimized builds while publishing the invariant,
     * caller function, and caller line through the ordinary CUDA error channel.
     * This function never repairs state, skips work, or selects another
     * execution path: a violated device-owned publication contract remains
     * unconditionally fatal.
     *
     * @param invariant Stable description of the violated device contract.
     * @param function  Device function that detected the violation.
     * @param line      Source line at which the violation was detected.
     */
    __device__ __forceinline__ void fail_fast_incomplete_llep_transfer(
        const char *invariant,
        const char *function,
        unsigned int line)
    {
        __assert_fail(invariant, __FILE__, line, function);
    }

#define FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(INVARIANT) \
    fail_fast_incomplete_llep_transfer((INVARIANT), __func__, __LINE__)

    /**
     * @brief Terminate CUDA execution when accepted verifier history has no ledger.
     *
     * The grouped verifier retains its speculative routes in a per-layer,
     * immutable-address ledger and consumes that ledger only after acceptance
     * is known.  Treating a missing or undersized ledger as an empty history
     * update would silently corrupt later LLEP routing evidence while the host
     * still observes a successful kernel enqueue.  A device assertion makes
     * that impossible and preserves the violated contract in CUDA diagnostics.
     *
     * @param invariant Stable description of the violated device contract.
     * @param function  Device function that detected the violation.
     * @param line      Source line at which the violation was detected.
     */
    __device__ __forceinline__ void fail_fast_invalid_grouped_verifier_commit(
        const char *invariant,
        const char *function,
        unsigned int line)
    {
        __assert_fail(invariant, __FILE__, line, function);
    }

#define FAIL_FAST_INVALID_GROUPED_VERIFIER_COMMIT(INVARIANT) \
    fail_fast_invalid_grouped_verifier_commit((INVARIANT), __func__, __LINE__)

    __device__ __forceinline__ bool prefill_llep_transfer_status_complete(
        const DeviceMoERebalanceStatusView *__restrict__ status,
        uint64_t expected_transfer_count,
        uint64_t expected_span_count)
    {
        if (!status ||
            status->magic != kDeviceMoERebalanceMagic ||
            status->version != kDeviceMoERebalanceVersion ||
            status->status_code != kDeviceMoERebalanceStatusOk ||
            status->plan_overflow != 0u ||
            status->payload_bucket_overflow != 0u ||
            expected_transfer_count > 0xffffffffULL ||
            expected_span_count > 0xffffffffULL)
        {
            return false;
        }

        const uint32_t transfer_count =
            static_cast<uint32_t>(expected_transfer_count);
        const uint32_t span_count =
            static_cast<uint32_t>(expected_span_count);
        return status->llep_weight_transfer_count == transfer_count &&
               status->planned_arrivals == transfer_count &&
               status->llep_assignment_span_count == span_count;
    }

    struct DeviceMoERebalancePlanEntryView
    {
        uint32_t op;
        uint32_t layer;
        uint32_t expert;
        uint32_t source_participant;
        uint32_t destination_participant;
        int32_t destination_overlay_participant;
        uint32_t source_resident_mask;
        uint32_t flags;
        uint32_t destination_slot;
        uint32_t payload_slot;
        uint32_t destination_previous_layer;
        uint32_t destination_previous_expert;
        uint32_t destination_generation;
        uint32_t reserved; // Canonical zero word; no implicit wire padding.
        uint64_t activation_count;
    };
    static_assert(std::is_standard_layout_v<DeviceMoERebalancePlanEntryView>);
    static_assert(std::has_unique_object_representations_v<DeviceMoERebalancePlanEntryView>,
                  "native command publication may not leave unnamed bytes");
    static_assert(offsetof(DeviceMoERebalancePlanEntryView, reserved) ==
                  llaminar2::moe_rebalance_abi::kPlanEntryReservedOffset);
    static_assert(offsetof(DeviceMoERebalancePlanEntryView, activation_count) ==
                  llaminar2::moe_rebalance_abi::kPlanEntryActivationCountOffset);
    static_assert(
        sizeof(DeviceMoERebalancePlanEntryView) ==
            llaminar2::moe_rebalance_abi::kPlanEntryBytes,
        "CUDA rebalance plan ABI must match the shared host record");

    /**
     * @brief Resolve one domain participant's stable overlay endpoint.
     *
     * The active bank is the device-owned namespace bridge: every expert owned
     * by a domain participant must name the same overlay route. Returning -1
     * rejects missing or contradictory mappings before an immutable transfer
     * command is published.
     */
    __device__ __forceinline__ int32_t
    rebalance_overlay_route_for_domain_participant(
        const DeviceMoEPlacementBankView &bank,
        uint32_t participant,
        uint32_t expert_count)
    {
        int32_t route = -1;
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            if (bank.experts[expert].owner_participant !=
                static_cast<int32_t>(participant))
            {
                continue;
            }
            const int32_t candidate =
                bank.overlay_route_participant[expert];
            if (candidate < 0 || (route >= 0 && route != candidate))
                return -1;
            route = candidate;
        }
        return route;
    }

    struct DeviceMoERebalanceCommandBufferHeaderView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t phase;
        uint32_t command_count;
        uint32_t command_capacity;
        uint32_t participant_id;
        uint32_t participant_count;
        llaminar2::DeviceMoERebalanceMovementDecision movement_decision;
    };
    static_assert(sizeof(DeviceMoERebalanceCommandBufferHeaderView) ==
                  llaminar2::moe_rebalance_abi::kCommandHeaderBytes);

    struct DeviceMoERebalanceWaveStateView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t next_start_layer;
        uint32_t planned_start_layer;
        uint32_t planned_layer_count;
        uint32_t command_capacity;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t requested_payload_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t reserved[3];
    };

    struct DeviceMoERebalanceWaveProgressView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t epoch;
        uint32_t state;
        uint32_t planned_start_layer;
        uint32_t planned_layer_count;
        uint32_t command_count;
        uint32_t copied_arrivals;
        uint32_t applied_arrivals;
        uint32_t applied_layer_count;
        uint32_t error_code;
        uint32_t requested_payload_slots;
        uint32_t payload_bucket_slots;
        uint32_t payload_bucket_index;
        uint32_t payload_bucket_overflow;
        uint32_t error_participant;
        uint64_t copied_payload_bytes;
    };

    struct DeviceMoERebalanceGraphControllerStateView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t participant_id;
        uint32_t participant_count;
        uint32_t next_epoch;
        uint32_t active_wave;
        uint32_t wave_count;
        uint32_t maintenance_launches;
        uint32_t decode_apply_polls;
        uint32_t decode_apply_hits;
        uint32_t last_error_code;
        uint32_t last_error_wave_index;
        uint32_t last_error_epoch;
        uint32_t last_error_expected_arrivals;
        uint32_t last_error_copied_arrivals;
        uint32_t last_error_copy_status_code;
        uint32_t last_error_copy_failure_flags;
        uint32_t last_error_copy_plan_entries_seen;
        uint32_t last_error_copy_skipped_wrong_destination;
        uint32_t last_error_missing_destination_slot;
        uint32_t last_error_missing_destination_layer;
        uint32_t last_error_missing_destination_expert;
        uint32_t last_error_missing_destination_source;
        uint32_t last_error_local_transfer_slot_count;
        uint32_t decode_rounds_committed;
        uint32_t decode_rounds_until_maintenance;
        uint32_t maintenance_period_rounds;
        uint32_t maintenance_due;
        uint32_t decode_boundary_advanced;
        DeviceMoERebalanceWaveProgressView waves[2];
        llaminar2::DeviceMoERebalanceMovementJournalState movement_journal;
        llaminar2::DeviceMoERebalanceDispatchTicket dispatch_ticket;
        llaminar2::DeviceMoERebalanceCompletedDecision completed_decision;
    };

    static_assert(sizeof(DeviceMoERebalanceWaveProgressView) ==
                  llaminar2::moe_rebalance_abi::kWaveProgressBytes);
    static_assert(sizeof(DeviceMoERebalanceGraphControllerStateView) ==
                  llaminar2::moe_rebalance_abi::kGraphControllerBytes);
    static_assert(offsetof(DeviceMoERebalanceGraphControllerStateView, completed_decision) ==
                  llaminar2::moe_rebalance_abi::kCompletedDecisionOffset);

    struct DeviceMoEExpertDirectoryEntryView
    {
        DeviceMoEExpertDescriptorView descriptor;
        uint32_t layer;
        uint32_t expert;
        uint32_t participant;
        uint32_t resident_mask;
        uint32_t epoch;
        uint32_t flags;
        uint32_t slot_index;
        uint32_t generation;
    };

    struct DeviceMoERebalanceApplyStatusView
    {
        uint32_t magic;
        uint32_t version;
        uint32_t status_code;
        uint32_t plan_entries_seen;
        uint32_t applied_arrivals;
        uint32_t skipped_wrong_destination;
        uint32_t invalid_plan_entries;
        uint32_t missing_source_descriptors;
        uint32_t missing_destination_slots;
        uint32_t descriptor_mismatches;
        uint32_t changed_layers;
        uint32_t copied_arrivals;
        uint32_t copy_incomplete;
        uint32_t post_apply_multi_resident_experts;
        uint32_t required_local_arrivals;
        uint32_t ready_local_arrivals;
        uint32_t first_missing_destination_slot;
        uint32_t first_missing_destination_layer;
        uint32_t first_missing_destination_expert;
        uint32_t first_missing_destination_source;
        uint32_t local_transfer_slot_count;
        uint32_t transaction_wave_index;
        uint32_t transaction_epoch;
        uint32_t transaction_command_count;
        uint64_t copied_payload_bytes;
    };

    static_assert(sizeof(DeviceMoERebalanceApplyStatusView) ==
                  llaminar2::moe_rebalance_abi::kApplyStatusBytes);

    /**
     * @brief Detect two local arrivals that cannot coexist in one transaction.
     *
     * A physical transfer slot has one logical occupant, and one logical
     * expert has one local payload publication. Duplicating either key within
     * a wave would make command order choose an accidental winner. Reject that
     * malformed plan before any inactive runtime bank is touched.
     */
    __device__ __forceinline__ bool
    rebalance_plan_duplicates_prior_local_arrival(
        const DeviceMoERebalancePlanEntryView *plans,
        uint32_t plan_index,
        const DeviceMoERebalancePlanEntryView &candidate,
        const DeviceMoERebalanceConfigView &config,
        int target_layer)
    {
        if (!plans ||
            !rebalance_plan_requires_payload(candidate.op) ||
            candidate.destination_participant !=
                config.participant_id)
        {
            return false;
        }

        for (uint32_t prior_index = 0u;
             prior_index < plan_index;
             ++prior_index)
        {
            const auto &prior = plans[prior_index];
            if (!rebalance_plan_requires_payload(prior.op) ||
                prior.destination_participant !=
                    config.participant_id ||
                (target_layer >= 0 &&
                 prior.layer !=
                     static_cast<uint32_t>(target_layer)))
            {
                continue;
            }
            if (prior.destination_slot ==
                    candidate.destination_slot ||
                (prior.layer == candidate.layer &&
                 prior.expert == candidate.expert))
            {
                return true;
            }
        }
        return false;
    }

    __device__ __forceinline__ bool prefill_llep_apply_status_complete(
        const DeviceMoERebalanceApplyStatusView *__restrict__ status,
        uint64_t expected_transfer_count)
    {
        if (!status ||
            status->magic != kDeviceMoERebalanceMagic ||
            status->version != kDeviceMoERebalanceVersion ||
            status->status_code != 0u ||
            expected_transfer_count > 0xffffffffULL)
        {
            return false;
        }

        const uint32_t transfer_count =
            static_cast<uint32_t>(expected_transfer_count);
        /*
         * `expected_transfer_count` is domain-wide, while apply status is
         * participant-local after command projection. Requiring every
         * participant to see every domain transfer rejects a valid split wave
         * (for example, two domain transfers with one local arrival on each
         * participant). Completeness is instead defined by the local contract:
         * every locally required payload is ready and published, with every
         * local payload count bounded by the domain total. `plan_entries_seen`
         * may be larger than the transfer count because the projected command
         * list can also contain resident-only assignments.
         */
        if (transfer_count > 0u &&
            (status->applied_arrivals > transfer_count ||
             status->required_local_arrivals > transfer_count ||
             status->ready_local_arrivals > transfer_count ||
             status->plan_entries_seen < status->required_local_arrivals))
        {
            return false;
        }

        return status->required_local_arrivals == status->ready_local_arrivals &&
               status->applied_arrivals == status->required_local_arrivals &&
               status->invalid_plan_entries == 0u &&
               status->missing_source_descriptors == 0u &&
               status->missing_destination_slots == 0u &&
               status->descriptor_mismatches == 0u &&
               status->copy_incomplete == 0u;
    }

    /**
     * @brief Return whether preflight found any condition that forbids commit.
     *
     * Arrival apply is one transaction even when a wave contains several
     * experts and displaces physical slots from several prior layers. The
     * kernel must not rebuild a subset of inactive banks while one payload is
     * still unpublished: doing so makes retry observe a mixed generation and
     * destroys the slot directory's single-occupant invariant.
     */
    __device__ __forceinline__ bool rebalance_apply_transaction_blocked(
        const DeviceMoERebalanceApplyStatusView *status)
    {
        return !status ||
               status->invalid_plan_entries != 0u ||
               status->missing_source_descriptors != 0u ||
               status->missing_destination_slots != 0u ||
               status->descriptor_mismatches != 0u ||
               status->copy_incomplete != 0u ||
               status->required_local_arrivals !=
                   status->ready_local_arrivals;
    }

    __device__ __forceinline__ bool rebalance_config_ok(
        const DeviceMoERebalanceConfigView &config)
    {
        return config.magic == kDeviceMoERebalanceMagic &&
               config.version == kDeviceMoERebalanceVersion &&
               config.num_layers > 0u &&
               config.num_experts > 0u &&
               config.num_experts <= static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
               config.top_k > 0u &&
               config.top_k <= static_cast<uint32_t>(kMaxTopK) &&
               config.participant_count > 0u &&
               config.participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
               config.participant_id < config.participant_count &&
               llaminar2::moe_rebalance_policy::hasValidRootParticipant(config) &&
               config.active_transfer_slot_capacity > 0u &&
               config.active_transfer_slot_capacity <=
                   kDeviceMoEMaxTransferSlots &&
               config.transfer_slot_directory_capacity > 0u &&
               config.transfer_slot_directory_capacity <=
                   kDeviceMoEMaxTransferSlots &&
               config.active_transfer_slot_capacity <=
                   config.transfer_slot_directory_capacity &&
               config.window_size_tokens > 0u &&
               config.initial_maintenance_period_tokens > 0u &&
               config.maintenance_period_tokens > 0u &&
               (config.layer_window_count == 0u ||
                config.layer_window_start < config.num_layers);
    }

    __device__ __forceinline__ uint32_t rebalance_command_buffer_count(
        uint32_t requested)
    {
        if (requested <= 1u)
            return 1u;
        return min(requested, 2u);
    }

    __device__ __forceinline__ bool rebalance_graph_controller_state_basic_ok(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        return state &&
               state->magic == kDeviceMoERebalanceMagic &&
               state->version == kDeviceMoERebalanceVersion &&
               state->participant_id == config.participant_id &&
               state->participant_count == config.participant_count &&
               state->wave_count == 2u &&
               ((state->decode_rounds_until_maintenance > 0u &&
                 state->maintenance_due == 0u) ||
                (state->decode_rounds_until_maintenance == 0u &&
                 state->maintenance_due == 1u)) &&
               state->decode_boundary_advanced <= 1u &&
               state->maintenance_period_rounds ==
                   config.maintenance_period_tokens;
    }

    __device__ __forceinline__ uint32_t rebalance_active_command_wave_index(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_buffer_count)
    {
        const uint32_t count = rebalance_command_buffer_count(command_buffer_count);
        if (count == 1u)
            return 0u;
        if (!rebalance_graph_controller_state_basic_ok(state, config))
            return 0u;
        return state->active_wave % count;
    }

    /** Advance one serial-visible decode round on the final routed layer. */
    __device__ __forceinline__ void
    advance_rebalance_serial_decode_round(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!rebalance_graph_controller_state_basic_ok(state, config) ||
            state->maintenance_due != 0u ||
            state->decode_rounds_until_maintenance == 0u)
        {
            if (state)
                state->maintenance_due = 2u;
            return;
        }

        ++state->decode_rounds_committed;
        --state->decode_rounds_until_maintenance;
        if (state->decode_rounds_until_maintenance == 0u)
            state->maintenance_due = 1u;
    }

    /** Consume one due edge and arm the recurring device-owned period. */
    __device__ __forceinline__ uint32_t
    consume_rebalance_maintenance_boundary(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!rebalance_graph_controller_state_basic_ok(state, config))
            return 2u;
        if (state->decode_boundary_advanced != 0u)
        {
            state->decode_boundary_advanced = 0u;
        }
        else
        {
            advance_rebalance_serial_decode_round(state, config);
        }
        if (state->maintenance_due == 0u)
            return 0u;
        if (state->maintenance_due != 1u ||
            state->decode_rounds_until_maintenance != 0u)
        {
            return 2u;
        }
        state->maintenance_due = 0u;
        state->decode_boundary_advanced = 0u;
        state->decode_rounds_until_maintenance =
            state->maintenance_period_rounds;
        return 1u;
    }

    /**
     * @brief Predict whether the next controller edge will consume maintenance.
     *
     * Parallel LLEP planning runs immediately before the publishing controller
     * on the same stream. It must avoid expensive policy work on ordinary
     * decode replays, but it must not mutate the request clock itself. This
     * predicate mirrors consume_rebalance_maintenance_boundary() without
     * writing any controller field; the following controller remains the sole
     * owner of the lifecycle transition.
     */
    __device__ __forceinline__ bool
    rebalance_maintenance_due_on_next_controller_edge(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!rebalance_graph_controller_state_basic_ok(state, config))
            return false;
        if (state->decode_boundary_advanced != 0u)
        {
            return state->maintenance_due == 1u &&
                   state->decode_rounds_until_maintenance == 0u;
        }
        return state->maintenance_due == 0u &&
               state->decode_rounds_until_maintenance == 1u;
    }

    /**
     * @brief Select one immutable root-projected command wave for transport.
     *
     * A controller may advance `active_wave` after planning so the alternate
     * command buffer can be prepared while payload transport is still in
     * flight. Transport kernels must not follow that mutable cursor. Projected
     * root headers are the domain-wide source of truth: a non-empty header
     * identifies the wave owned by this transaction on every participant.
     *
     * More than one non-empty header means the double-buffer ownership
     * contract has already been violated. Return the invalid sentinel so the
     * begin kernel poisons the copy status instead of choosing arbitrarily.
     * A no-work replay has no non-empty header, so it snapshots the controller
     * cursor once; later phases still consume the resulting immutable ticket.
     */
    __device__ __forceinline__ uint32_t
    rebalance_select_transfer_transaction_wave(
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_buffer_count)
    {
        if (!command_headers)
            return kDeviceMoEInvalidSlot;

        const uint32_t count =
            rebalance_command_buffer_count(command_buffer_count);
        uint32_t selected = kDeviceMoEInvalidSlot;
        for (uint32_t wave = 0u; wave < count; ++wave)
        {
            const auto &header = command_headers[wave];
            if (header.magic != kDeviceMoERebalanceMagic ||
                header.version != kDeviceMoERebalanceVersion ||
                header.phase != kDeviceMoERebalancePhasePlanAssignments ||
                header.participant_id != config.participant_id ||
                header.participant_count != config.participant_count ||
                header.epoch == 0u ||
                header.command_count == 0u)
            {
                continue;
            }
            if (selected != kDeviceMoEInvalidSlot)
                return kDeviceMoEInvalidSlot;
            selected = wave;
        }
        return selected != kDeviceMoEInvalidSlot
                   ? selected
                   : rebalance_active_command_wave_index(
                         state,
                         config,
                         command_buffer_count);
    }

    __device__ __forceinline__ bool rebalance_active_wave_busy_for_new_plan(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_wave_index)
    {
        if (state && state->last_error_code != 0u)
        {
            return true;
        }
        if (!rebalance_graph_controller_state_basic_ok(state, config) ||
            command_wave_index >= state->wave_count)
        {
            return false;
        }
        const auto &wave = state->waves[command_wave_index];
        if (wave.magic != kDeviceMoERebalanceMagic ||
            wave.version != kDeviceMoERebalanceVersion ||
            wave.epoch == 0u)
        {
            return false;
        }
        return wave.state == kDeviceMoERebalanceLifecyclePlanning ||
               wave.state == kDeviceMoERebalanceLifecycleTransferInFlight ||
               wave.state == kDeviceMoERebalanceLifecycleReadyToApply ||
               wave.state == kDeviceMoERebalanceLifecycleApplying ||
               wave.state == llaminar2::moe_rebalance_publication::kPreparedForPublication ||
               wave.state == kDeviceMoERebalanceLifecycleError;
    }

    __device__ __forceinline__ void reset_rebalance_wave_progress_device(
        DeviceMoERebalanceWaveProgressView &wave)
    {
        /*
         * Error is terminal for the request-owned controller. A later empty
         * maintenance replay must never launder a failed wave back to Idle.
         */
        if (wave.state == kDeviceMoERebalanceLifecycleError ||
            wave.error_code != 0u)
        {
            return;
        }
        DeviceMoERebalanceWaveProgressView zero{};
        wave = zero;
        wave.magic = kDeviceMoERebalanceMagic;
        wave.version = kDeviceMoERebalanceVersion;
        wave.state = kDeviceMoERebalanceLifecycleIdle;
        wave.error_participant = kDeviceMoEInvalidSlot;
        wave.copied_payload_bytes = 0;
    }

    /**
     * Publish the first fatal controller error and permanently poison replay.
     *
     * A claim marker serializes maintenance/apply streams without involving
     * the host. The winning publisher records complete provenance, fences it,
     * and only then exposes the public error code. Later failures preserve the
     * first cause and cannot reset or retarget the affected wave.
     */
    __device__ __forceinline__ void poison_rebalance_graph_controller(
        DeviceMoERebalanceGraphControllerStateView *state,
        uint32_t error_code,
        uint32_t wave_index,
        uint32_t epoch,
        uint32_t expected_arrivals,
        uint32_t copied_arrivals,
        uint32_t copy_status_code,
        uint32_t error_participant,
        uint32_t copy_failure_flags = 0u,
        uint32_t copy_plan_entries_seen = 0u,
        uint32_t copy_skipped_wrong_destination = 0u,
        uint32_t missing_destination_slot = kDeviceMoEInvalidSlot,
        uint32_t missing_destination_layer = kDeviceMoEInvalidSlot,
        uint32_t missing_destination_expert = kDeviceMoEInvalidSlot,
        uint32_t missing_destination_source = kDeviceMoEInvalidSlot,
        uint32_t local_transfer_slot_count = 0u)
    {
        if (!state || error_code == 0u)
            return;

        auto *error_word =
            reinterpret_cast<unsigned int *>(&state->last_error_code);
        if (atomicCAS(
                error_word,
                0u,
                kDeviceMoERebalanceErrorPublicationInProgress) != 0u)
        {
            return;
        }

        state->last_error_wave_index = wave_index;
        state->last_error_epoch = epoch;
        state->last_error_expected_arrivals = expected_arrivals;
        state->last_error_copied_arrivals = copied_arrivals;
        state->last_error_copy_status_code = copy_status_code;
        state->last_error_copy_failure_flags = copy_failure_flags;
        state->last_error_copy_plan_entries_seen =
            copy_plan_entries_seen;
        state->last_error_copy_skipped_wrong_destination =
            copy_skipped_wrong_destination;
        state->last_error_missing_destination_slot =
            missing_destination_slot;
        state->last_error_missing_destination_layer =
            missing_destination_layer;
        state->last_error_missing_destination_expert =
            missing_destination_expert;
        state->last_error_missing_destination_source =
            missing_destination_source;
        state->last_error_local_transfer_slot_count =
            local_transfer_slot_count;
        if (wave_index < state->wave_count && wave_index < 2u)
        {
            auto &wave = state->waves[wave_index];
            wave.magic = kDeviceMoERebalanceMagic;
            wave.version = kDeviceMoERebalanceVersion;
            wave.error_participant = error_participant;
            wave.error_code = error_code;
            wave.state = kDeviceMoERebalanceLifecycleError;
        }
        __threadfence();
        atomicExch(error_word, error_code);
    }

    __device__ __forceinline__ void clear_rebalance_command_header_device(
        DeviceMoERebalanceCommandBufferHeaderView *header)
    {
        if (!header)
            return;
        const uint32_t command_capacity = header->command_capacity;
        const uint32_t participant_id = header->participant_id;
        const uint32_t participant_count = header->participant_count;
        header->magic = kDeviceMoERebalanceMagic;
        header->version = kDeviceMoERebalanceVersion;
        header->epoch = 0u;
        header->phase = kDeviceMoERebalancePhasePlanAssignments;
        header->command_count = 0u;
        header->command_capacity = command_capacity;
        header->participant_id = participant_id;
        header->participant_count = participant_count;
        header->movement_decision = {};
    }

    __device__ __forceinline__ uint32_t rebalance_layer_window_count(
        const DeviceMoERebalanceConfigView &config)
    {
        return config.layer_window_count == 0u
                   ? config.num_layers
                   : min(config.layer_window_count, config.num_layers);
    }

    __device__ __forceinline__ uint32_t rebalance_layer_wave_count(
        const DeviceMoERebalanceConfigView &config)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        return config.layer_wave_count == 0u
                   ? window_count
                   : min(config.layer_wave_count, window_count);
    }

    __device__ __forceinline__ uint32_t rebalance_wave_start_offset(
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalanceWaveStateView *wave_state)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        if (!wave_state ||
            window_count == 0u ||
            config.num_layers == 0u ||
            wave_state->magic != kDeviceMoERebalanceMagic ||
            wave_state->version != kDeviceMoERebalanceVersion ||
            wave_state->participant_id != config.participant_id ||
            wave_state->participant_count != config.participant_count)
        {
            return 0u;
        }

        const uint32_t layer_window_start = config.layer_window_start % config.num_layers;
        const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
        const uint32_t candidate_offset =
            (candidate + config.num_layers - layer_window_start) % config.num_layers;
        return candidate_offset < window_count ? candidate_offset : 0u;
    }

    __device__ __forceinline__ uint32_t rebalance_actual_layer_for_wave_index(
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalanceWaveStateView *wave_state,
        uint32_t wave_layer)
    {
        const uint32_t window_count = rebalance_layer_window_count(config);
        if (window_count == 0u || config.num_layers == 0u)
            return 0u;
        const uint32_t layer_window_start = config.layer_window_start % config.num_layers;
        const uint32_t start_offset = rebalance_wave_start_offset(config, wave_state);
        return (layer_window_start + ((start_offset + wave_layer) % window_count)) %
               config.num_layers;
    }

    __device__ __forceinline__ unsigned long long rebalance_collected_state_word(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant,
        uint32_t wave_layer,
        uint32_t expert)
    {
        const unsigned long long participant_stride =
            static_cast<unsigned long long>(rebalance_layer_wave_count(config)) *
            static_cast<unsigned long long>(config.num_experts);
        const unsigned long long layer_stride =
            static_cast<unsigned long long>(config.num_experts);
        return histograms[
            static_cast<unsigned long long>(participant) * participant_stride +
            static_cast<unsigned long long>(wave_layer) * layer_stride +
            static_cast<unsigned long long>(expert)];
    }

    __device__ __forceinline__ unsigned long long rebalance_histogram_count(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant,
        uint32_t wave_layer,
        uint32_t expert)
    {
        return llaminar2::moe_rebalance_policy::collectedStateActivationCount(
            rebalance_collected_state_word(
                histograms, config, participant, wave_layer, expert));
    }

    __device__ __forceinline__ uint32_t
    rebalance_collected_active_transfer_slot_count(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant)
    {
        return llaminar2::moe_rebalance_policy::
            collectedStateActiveTransferSlots(
                rebalance_collected_state_word(
                    histograms, config, participant, 0u, 0u));
    }

    __device__ __forceinline__ uint32_t
    rebalance_collected_transfer_backed_participant_mask(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t wave_layer,
        uint32_t expert)
    {
        uint32_t mask = 0u;
        for (uint32_t participant = 0;
             participant < config.participant_count;
             ++participant)
        {
            if (llaminar2::moe_rebalance_policy::collectedStateTransferBacked(
                    rebalance_collected_state_word(
                        histograms,
                        config,
                        participant,
                        wave_layer,
                        expert)))
            {
                mask |= llaminar2::moe_rebalance_policy::participantBit(
                    participant);
            }
        }
        return mask;
    }

    __device__ __forceinline__ unsigned long long rebalance_global_count(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config,
        uint32_t wave_layer,
        uint32_t expert)
    {
        unsigned long long total = 0;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
            total += rebalance_histogram_count(histograms, config, participant, wave_layer, expert);
        return total;
    }

    __device__ __forceinline__ uint32_t rebalance_clamp_u64_to_u32(
        unsigned long long value)
    {
        return value > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(value);
    }

    /**
     * @brief Return whether one expert precedes another in canonical LLEP order.
     *
     * Experts are ordered by descending activation count and then ascending
     * logical expert id. Invalid padding lanes always follow real experts so
     * the fixed-width sorting network also supports smaller codebooks.
     */
    __device__ __forceinline__ bool rebalance_llep_expert_precedes(
        uint32_t lhs,
        uint32_t rhs,
        const uint64_t *expert_loads,
        uint32_t expert_count)
    {
        const bool lhs_valid = lhs < expert_count;
        const bool rhs_valid = rhs < expert_count;
        if (lhs_valid != rhs_valid)
            return lhs_valid;
        if (!lhs_valid)
            return lhs < rhs;

        const uint64_t lhs_load = expert_loads[lhs];
        const uint64_t rhs_load = expert_loads[rhs];
        return lhs_load > rhs_load ||
               (lhs_load == rhs_load && lhs < rhs);
    }

    /**
     * @brief Cooperatively sort all LLEP experts without per-thread local scratch.
     *
     * One thread owns each slot in a power-of-two bitonic network. Every
     * compare-exchange is integer-only and uses the same load/id ordering as
     * sortExpertsByLoadDescending(), preserving byte-stable assignment plans
     * while reducing the scalar O(E^2) insertion sort to O(log^2 E)
     * synchronized rounds. The active network width is rounded up from the
     * codebook size so smaller expert sets avoid irrelevant barriers.
     */
    __device__ __forceinline__ void rebalance_sort_llep_experts_parallel(
        uint32_t *sorted_experts,
        const uint64_t *expert_loads,
        uint32_t expert_count,
        uint32_t lane)
    {
        sorted_experts[lane] =
            lane < expert_count ? lane : kDeviceMoEInvalidSlot;
        __syncthreads();

        uint32_t network_width = 1u;
        while (network_width < expert_count)
            network_width <<= 1u;

        for (uint32_t sequence = 2u;
             sequence <= network_width;
             sequence <<= 1u)
        {
            for (uint32_t stride = sequence >> 1u;
                 stride > 0u;
                 stride >>= 1u)
            {
                const uint32_t peer = lane ^ stride;
                if (lane < network_width && peer > lane)
                {
                    const uint32_t first = sorted_experts[lane];
                    const uint32_t second = sorted_experts[peer];
                    const bool descending = (lane & sequence) == 0u;
                    const bool swap =
                        descending
                            ? rebalance_llep_expert_precedes(
                                  second, first, expert_loads, expert_count)
                            : rebalance_llep_expert_precedes(
                                  first, second, expert_loads, expert_count);
                    if (swap)
                    {
                        sorted_experts[lane] = second;
                        sorted_experts[peer] = first;
                    }
                }
                __syncthreads();
            }
        }
    }

    __device__ __forceinline__ unsigned long long rebalance_window_observed_slots(
        const unsigned long long *histograms,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!histograms || config.num_layers == 0u || config.top_k == 0u ||
            config.participant_count == 0u)
        {
            return 0ULL;
        }
        const uint32_t wave_count = rebalance_layer_wave_count(config);
        if (wave_count == 0u)
            return 0ULL;
        unsigned long long last_layer_slots = 0;
        const uint32_t last_layer = wave_count - 1u;
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                last_layer_slots += rebalance_histogram_count(
                    histograms, config, participant, last_layer, expert);
        return last_layer_slots;
    }

    __device__ __forceinline__ unsigned long long rebalance_window_required_slots(
        const DeviceMoERebalanceConfigView &config)
    {
        return static_cast<unsigned long long>(config.window_size_tokens) *
               static_cast<unsigned long long>(config.top_k);
    }

    __device__ __forceinline__ bool runtime_shape_ok(
        const DeviceMoELayerRuntimeView *runtime,
        uint32_t execution_bank,
        int num_experts,
        int top_k)
    {
        return runtime &&
               execution_bank <= 1u &&
               runtime->active_epoch != 0u &&
               runtime->expert_count == static_cast<uint32_t>(num_experts) &&
               runtime->top_k == static_cast<uint32_t>(top_k);
    }

    /**
     * @brief Resolve or fatally reject the immutable decode placement bank.
     *
     * This check belongs before `shape_ok`: a stale epoch ticket is a broken
     * ownership edge, not an alternate tensor shape that may skip publication.
     */
    __device__ __forceinline__ uint32_t
    runtime_require_decode_execution_bank(
        const DeviceMoELayerRuntimeView *runtime)
    {
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (execution_bank > 1u)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "ExpertOverlay decode ticket does not name a retained placement bank");
        }
        return execution_bank;
    }

    /** @return Whether decode can publish one final participant per route. */
    __device__ __forceinline__ bool runtime_route_assignment_ledger_ok(
        const DeviceMoELayerRuntimeView *runtime,
        int top_k)
    {
        return runtime && top_k > 0 && runtime->route_participant_ids &&
               runtime->prefill_route_capacity >=
                   static_cast<std::uint32_t>(top_k);
    }

    __device__ __forceinline__ int runtime_expert_owner(
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        return bank.experts[expert_id].owner_participant;
    }

    __device__ __forceinline__ uint32_t runtime_participant_bit(int participant)
    {
        return llaminar2::moe_rebalance_policy::participantBit(
            static_cast<uint32_t>(participant));
    }

    __device__ __forceinline__ uint32_t runtime_valid_participant_mask(uint32_t participant_count)
    {
        return llaminar2::moe_rebalance_policy::validParticipantMask(participant_count);
    }

    __device__ __forceinline__ int rebalance_first_resident_participant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int preferred_participant,
        int excluded_participant)
    {
        return llaminar2::moe_rebalance_policy::firstResidentParticipant(
            resident_mask,
            participant_count,
            preferred_participant,
            excluded_participant);
    }

    __device__ __forceinline__ bool rebalance_candidate_can_affect_local_compute(
        const DeviceMoEExpertDescriptorView &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfigView &config,
        uint32_t participant_bit,
        bool plan_missing_arrivals)
    {
        (void)participant_bit;
        return llaminar2::moe_rebalance_policy::candidateCanAffectLocalCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    __device__ __forceinline__ bool rebalance_candidate_can_affect_domain_compute(
        const DeviceMoEExpertDescriptorView &desc,
        uint32_t resident_mask,
        const DeviceMoERebalanceConfigView &config,
        bool plan_missing_arrivals)
    {
        return llaminar2::moe_rebalance_policy::candidateCanAffectDomainCompute(
            desc,
            resident_mask,
            config,
            plan_missing_arrivals);
    }

    __device__ __forceinline__ bool rebalance_append_plan_entry(
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalancePlanEntryView &entry,
        DeviceMoERebalanceStatusView *status)
    {
        if (!plan_entries || !plan_count)
            return false;
        const uint32_t index = *plan_count;
        if (index >= plan_capacity)
        {
            if (status)
                ++status->plan_overflow;
            return false;
        }
        plan_entries[index] = entry;
        plan_entries[index].reserved = 0u; // Complete canonical command representation.
        *plan_count = index + 1u;
        if (status)
            ++status->planned_arrivals;
        return true;
    }

    __device__ __forceinline__ void rebalance_command_layer_span(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t command_count,
        const DeviceMoERebalanceConfigView &config,
        uint32_t layer_window_start,
        uint32_t start_offset,
        uint32_t layer_window_count,
        uint32_t layer_wave_count,
        uint32_t *planned_start_layer,
        uint32_t *planned_layer_count)
    {
        if (!planned_start_layer || !planned_layer_count ||
            !plan_entries || command_count == 0u ||
            config.num_layers == 0u ||
            layer_window_count == 0u ||
            layer_wave_count == 0u)
        {
            return;
        }

        uint32_t first_wave_offset = layer_wave_count;
        uint32_t last_wave_offset = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const uint32_t layer = plan_entries[i].layer % config.num_layers;
            const uint32_t window_offset =
                (layer + config.num_layers - layer_window_start) % config.num_layers;
            if (window_offset >= layer_window_count)
                continue;
            const uint32_t wave_offset =
                (window_offset + layer_window_count - start_offset) % layer_window_count;
            if (wave_offset >= layer_wave_count)
                continue;
            if (first_wave_offset > wave_offset)
                first_wave_offset = wave_offset;
            if (last_wave_offset < wave_offset)
                last_wave_offset = wave_offset;
        }

        if (first_wave_offset >= layer_wave_count ||
            last_wave_offset < first_wave_offset)
        {
            return;
        }

        *planned_start_layer =
            (layer_window_start + ((start_offset + first_wave_offset) % layer_window_count)) %
            config.num_layers;
        *planned_layer_count = last_wave_offset - first_wave_offset + 1u;
    }

    __device__ __forceinline__ uint32_t runtime_expert_resident_mask(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        const uint32_t participant_count = runtime->participant_count;
        uint32_t mask = bank.resident_participant_mask[expert_id] &
                        runtime_valid_participant_mask(participant_count);
        if (bank.local_compute_mask[expert_id] != 0u &&
            runtime->participant_id < participant_count)
        {
            mask |= runtime_participant_bit(static_cast<int>(runtime->participant_id));
        }
        return mask;
    }

    __device__ __forceinline__ int runtime_resident_count(
        uint32_t resident_mask,
        uint32_t participant_count)
    {
        int count = 0;
        for (int participant = 0; participant < static_cast<int>(participant_count); ++participant)
        {
            if ((resident_mask & runtime_participant_bit(participant)) != 0u)
                ++count;
        }
        return count;
    }

    /**
     * @brief Count live transfer-storage claims across the participant runtime.
     *
     * Storage liveness is deliberately independent of the current row
     * assignment. An authoritative owner may receive zero rows in one LLEP
     * window while its bytes remain the only durable copy of that expert.
     * Counting only `local_compute_mask` therefore under-reports physical
     * occupancy and lets the planner publish more ownership arrivals than the
     * directory can retain.
     *
     * This helper examines active runtime banks and authenticates every
     * transfer-backed local resident by stable slot identity. A durable
     * descriptor naming transaction-only staging capacity is invalid, as is a
     * duplicated physical claim. The already-captured controller exports this
     * evidence through its request-boundary status publication.
     */
    using RebalanceTransferSlotClaimSummaryView =
        llaminar2::DeviceMoETransferSlotClaimSummary;
    using RebalanceRouterBenefitSummaryView =
        llaminar2::DeviceMoERouterBenefitSummary;

    /**
     * @brief Construct the sentinel-bearing empty transfer-slot summary.
     */
    __device__ __forceinline__ RebalanceTransferSlotClaimSummaryView
    rebalance_empty_transfer_slot_claim_summary()
    {
        RebalanceTransferSlotClaimSummaryView summary{};
        summary.first_invalid_slot = kDeviceMoEInvalidSlot;
        summary.first_invalid_layer = kDeviceMoEInvalidSlot;
        summary.first_invalid_expert = kDeviceMoEInvalidSlot;
        return summary;
    }

    /**
     * @brief Return whether one deterministic layer/expert witness precedes another.
     */
    __device__ __forceinline__ bool rebalance_claim_location_precedes(
        uint32_t lhs_layer,
        uint32_t lhs_expert,
        uint32_t rhs_layer,
        uint32_t rhs_expert)
    {
        return lhs_layer < rhs_layer ||
               (lhs_layer == rhs_layer && lhs_expert < rhs_expert);
    }

    /**
     * @brief Deterministically merge one disjoint claim scan into another.
     *
     * Sums are associative. Maximum-slot and first-failure witnesses use
     * explicit layer/expert tie breaks, so the parallel reduction reproduces
     * the byte result of the original ascending serial walk.
     */
    __device__ __forceinline__ void rebalance_merge_transfer_slot_claim_summary(
        RebalanceTransferSlotClaimSummaryView &destination,
        const RebalanceTransferSlotClaimSummaryView &source)
    {
        if (source.unique_claims != 0u &&
            (destination.unique_claims == 0u ||
             source.max_slot > destination.max_slot ||
             (source.max_slot == destination.max_slot &&
              rebalance_claim_location_precedes(
                  source.max_slot_layer,
                  source.max_slot_expert,
                  destination.max_slot_layer,
                  destination.max_slot_expert))))
        {
            destination.max_slot = source.max_slot;
            destination.max_slot_layer = source.max_slot_layer;
            destination.max_slot_expert = source.max_slot_expert;
        }
        if (source.duplicate_claims != 0u &&
            (destination.duplicate_claims == 0u ||
             rebalance_claim_location_precedes(
                 source.first_duplicate_layer,
                 source.first_duplicate_expert,
                 destination.first_duplicate_layer,
                 destination.first_duplicate_expert)))
        {
            destination.first_duplicate_slot = source.first_duplicate_slot;
            destination.first_duplicate_layer = source.first_duplicate_layer;
            destination.first_duplicate_expert = source.first_duplicate_expert;
        }
        if (source.invalid_claims != 0u &&
            (destination.invalid_claims == 0u ||
             rebalance_claim_location_precedes(
                 source.first_invalid_layer,
                 source.first_invalid_expert,
                 destination.first_invalid_layer,
                 destination.first_invalid_expert)))
        {
            destination.first_invalid_slot = source.first_invalid_slot;
            destination.first_invalid_layer = source.first_invalid_layer;
            destination.first_invalid_expert = source.first_invalid_expert;
            destination.first_invalid_reasons = source.first_invalid_reasons;
            destination.first_invalid_flags = source.first_invalid_flags;
            destination.first_invalid_resident_mask =
                source.first_invalid_resident_mask;
            destination.first_invalid_owner = source.first_invalid_owner;
        }

        destination.transient_placement_layers +=
            source.transient_placement_layers;
        destination.non_owner_assignment_layers +=
            source.non_owner_assignment_layers;
        destination.active_claims += source.active_claims;
        destination.unique_claims += source.unique_claims;
        destination.duplicate_claims += source.duplicate_claims;
        destination.invalid_claims += source.invalid_claims;
    }

    /**
     * @brief Merge disjoint routed-layer economy observations.
     */
    __device__ __forceinline__ void rebalance_merge_router_benefit_summary(
        RebalanceRouterBenefitSummaryView &destination,
        const RebalanceRouterBenefitSummaryView &source)
    {
        destination.eligible_dispatches += source.eligible_dispatches;
        destination.used_dispatches += source.used_dispatches;
        destination.improved_dispatches += source.improved_dispatches;
        destination.default_load_spread_total +=
            source.default_load_spread_total;
        destination.actual_load_spread_total +=
            source.actual_load_spread_total;
        destination.load_spread_improvement_total +=
            source.load_spread_improvement_total;
        destination.active_dispatches += source.active_dispatches;
        destination.miss_dispatches += source.miss_dispatches;
        destination.selected_expert_slots += source.selected_expert_slots;
        destination.replicated_selected_expert_slots +=
            source.replicated_selected_expert_slots;
        destination.hot_cache_active_layers +=
            source.hot_cache_active_layers;
        destination.invalid_runtime_layers += source.invalid_runtime_layers;
    }

    __device__ __forceinline__ RebalanceTransferSlotClaimSummaryView
    rebalance_transfer_slot_claim_summary_partial(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        uint32_t scan_lane,
        uint32_t scan_width)
    {
        auto summary = rebalance_empty_transfer_slot_claim_summary();
        if (!runtime_layers || !rebalance_config_ok(config) || scan_width == 0u)
            return summary;

        const uint32_t local_participant_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        for (uint32_t layer = scan_lane;
             layer < config.num_layers;
             layer += scan_width)
        {
            const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (runtime.active_bank > 1u ||
                runtime.active_epoch == 0u ||
                runtime.expert_count != config.num_experts ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                continue;
            }

            if (runtime.current_batch_llep_movement_observed != 0u)
                ++summary.transient_placement_layers;
            if (runtime.current_batch_llep_non_owner_assignment_observed != 0u)
                ++summary.non_owner_assignment_layers;
        }

        const uint64_t claim_count =
            static_cast<uint64_t>(config.num_layers) * config.num_experts;
        for (uint64_t flat_index = scan_lane;
             flat_index < claim_count;
             flat_index += scan_width)
        {
            const uint32_t layer =
                static_cast<uint32_t>(flat_index / config.num_experts);
            const uint32_t expert =
                static_cast<uint32_t>(flat_index % config.num_experts);
            const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (runtime.active_bank > 1u ||
                runtime.active_epoch == 0u ||
                runtime.expert_count != config.num_experts ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                continue;
            }

            const DeviceMoEPlacementBankView &bank =
                runtime.banks[runtime.active_bank];
            const auto &descriptor = bank.experts[expert];
                const uint32_t flags = descriptor.flags;
                const uint32_t resident_mask =
                    bank.resident_participant_mask[expert];
                const auto claim =
                    llaminar2::moe_rebalance_policy::
                        classifyTransferSlotClaim(
                            flags,
                            resident_mask,
                            local_participant_bit,
                            descriptor.owner_participant,
                            config.participant_id,
                            descriptor.local_slot,
                            kDeviceMoEMaxTransferSlots,
                            config.transfer_slot_directory_capacity,
                            kDeviceMoEFlagValid,
                            kDeviceMoEFlagResident,
                            kDeviceMoEFlagTransferSlot);
                if (claim.claims_storage)
                {
                    ++summary.active_claims;
                    const int32_t signed_slot = descriptor.local_slot;
                    if (!claim.valid())
                    {
                        if (summary.invalid_claims == 0u)
                        {
                            summary.first_invalid_slot =
                                signed_slot < 0
                                    ? kDeviceMoEInvalidSlot
                                    : static_cast<uint32_t>(signed_slot);
                            summary.first_invalid_layer = layer;
                            summary.first_invalid_expert = expert;
                            summary.first_invalid_reasons =
                                claim.invalid_reasons;
                            summary.first_invalid_flags = flags;
                            summary.first_invalid_resident_mask =
                                resident_mask;
                            summary.first_invalid_owner =
                                descriptor.owner_participant;
                        }
                        ++summary.invalid_claims;
                        continue;
                    }

                    const uint32_t slot =
                        static_cast<uint32_t>(signed_slot);
                    if (summary.unique_claims == 0u ||
                        slot > summary.max_slot)
                    {
                        summary.max_slot = slot;
                        summary.max_slot_layer = layer;
                        summary.max_slot_expert = expert;
                    }
                    if (!local_transfer_slots ||
                        local_transfer_slot_count !=
                            config.transfer_slot_directory_capacity ||
                        slot >= local_transfer_slot_count)
                    {
                        if (summary.invalid_claims == 0u)
                        {
                            summary.first_invalid_slot = slot;
                            summary.first_invalid_layer = layer;
                            summary.first_invalid_expert = expert;
                            summary.first_invalid_reasons =
                                llaminar2::moe_rebalance_policy::
                                    TransferSlotClaimDirectoryIdentityMismatch;
                            summary.first_invalid_flags = flags;
                            summary.first_invalid_resident_mask =
                                resident_mask;
                            summary.first_invalid_owner =
                                descriptor.owner_participant;
                        }
                        ++summary.invalid_claims;
                        continue;
                    }
                    const auto &directory_entry = local_transfer_slots[slot];
                    const bool physical_identity_valid =
                        directory_entry.participant ==
                            config.participant_id &&
                        directory_entry.slot_index == slot &&
                        directory_entry.descriptor.local_slot ==
                            signed_slot &&
                        directory_entry.generation != 0xffffffffu;
                    if (!physical_identity_valid)
                    {
                        if (summary.invalid_claims == 0u)
                        {
                            summary.first_invalid_slot = slot;
                            summary.first_invalid_layer = layer;
                            summary.first_invalid_expert = expert;
                            summary.first_invalid_reasons =
                                llaminar2::moe_rebalance_policy::
                                    TransferSlotClaimDirectoryIdentityMismatch;
                            summary.first_invalid_flags = flags;
                            summary.first_invalid_resident_mask =
                                resident_mask;
                            summary.first_invalid_owner =
                                descriptor.owner_participant;
                        }
                        ++summary.invalid_claims;
                        continue;
                    }

                    const bool occupant_matches =
                        directory_entry.layer == layer &&
                        directory_entry.expert == expert &&
                        directory_entry.descriptor.logical_expert_id ==
                            static_cast<int32_t>(expert) &&
                        directory_entry.descriptor.gate.payload ==
                            descriptor.gate.payload &&
                        directory_entry.descriptor.gate.scales ==
                            descriptor.gate.scales &&
                        directory_entry.descriptor.up.payload ==
                            descriptor.up.payload &&
                        directory_entry.descriptor.up.scales ==
                            descriptor.up.scales &&
                        directory_entry.descriptor.down.payload ==
                            descriptor.down.payload &&
                        directory_entry.descriptor.down.scales ==
                            descriptor.down.scales;
                    if (!occupant_matches)
                    {
                        if (summary.invalid_claims == 0u)
                        {
                            summary.first_invalid_slot = slot;
                            summary.first_invalid_layer = layer;
                            summary.first_invalid_expert = expert;
                            summary.first_invalid_reasons =
                                llaminar2::moe_rebalance_policy::
                                    TransferSlotClaimOccupantMismatch;
                            summary.first_invalid_flags = flags;
                            summary.first_invalid_resident_mask =
                                resident_mask;
                            summary.first_invalid_owner =
                                descriptor.owner_participant;
                        }
                        ++summary.invalid_claims;
                        continue;
                    }

                ++summary.unique_claims;
            }
        }
        return summary;
    }

    /**
     * @brief Preserve the serial helper for non-LLEP controller policies.
     *
     * LLEP uses the parallel preflight publication below. Other policies keep
     * this byte-identical implementation until their graph workspace adopts
     * the same typed preflight ABI.
     */
    __device__ __forceinline__ RebalanceTransferSlotClaimSummaryView
    rebalance_transfer_slot_claim_summary(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count)
    {
        return rebalance_transfer_slot_claim_summary_partial(
            runtime_layers,
            config,
            local_transfer_slots,
            local_transfer_slot_count,
            0u,
            1u);
    }

    __device__ __forceinline__ uint32_t
    rebalance_active_transfer_slot_expert_count(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!runtime_layers || !rebalance_config_ok(config))
            return 0u;

        const uint32_t local_participant_bit =
            runtime_participant_bit(
                static_cast<int>(config.participant_id));
        uint32_t active_claims = 0u;
        for (uint32_t layer = 0u; layer < config.num_layers; ++layer)
        {
            const auto &runtime = runtime_layers[layer];
            if (runtime.active_bank > 1u ||
                runtime.active_epoch == 0u ||
                runtime.expert_count != config.num_experts ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                continue;
            }
            const auto &bank = runtime.banks[runtime.active_bank];
            for (uint32_t expert = 0u;
                 expert < config.num_experts;
                 ++expert)
            {
                const auto &descriptor = bank.experts[expert];
                const auto claim =
                    llaminar2::moe_rebalance_policy::
                        classifyTransferSlotClaim(
                            descriptor.flags,
                            bank.resident_participant_mask[expert],
                            local_participant_bit,
                            descriptor.owner_participant,
                            config.participant_id,
                            descriptor.local_slot,
                            kDeviceMoEMaxTransferSlots,
                            config.transfer_slot_directory_capacity,
                            kDeviceMoEFlagValid,
                            kDeviceMoEFlagResident,
                            kDeviceMoEFlagTransferSlot);
                if (claim.claims_storage)
                    ++active_claims;
            }
        }
        return active_claims;
    }

    __device__ __forceinline__ void
    rebalance_publish_transfer_slot_claim_summary_fields(
        DeviceMoERebalanceStatusView *status,
        const RebalanceTransferSlotClaimSummaryView &summary)
    {
        if (!status)
            return;
        status->prefill_current_batch_movement_layers =
            summary.transient_placement_layers;
        status->prefill_current_batch_non_owner_assignment_layers =
            summary.non_owner_assignment_layers;
        status->prefill_active_transfer_slot_experts = summary.active_claims;
        status->prefill_unique_transfer_slot_claims = summary.unique_claims;
        status->prefill_duplicate_transfer_slot_claims =
            summary.duplicate_claims;
        status->prefill_invalid_transfer_slot_claims = summary.invalid_claims;
        status->prefill_max_transfer_slot = summary.max_slot;
        status->prefill_max_transfer_slot_layer = summary.max_slot_layer;
        status->prefill_max_transfer_slot_expert = summary.max_slot_expert;
        status->prefill_first_duplicate_transfer_slot =
            summary.first_duplicate_slot;
        status->prefill_first_duplicate_layer =
            summary.first_duplicate_layer;
        status->prefill_first_duplicate_expert =
            summary.first_duplicate_expert;
        status->prefill_first_invalid_transfer_slot =
            summary.first_invalid_slot;
        status->prefill_first_invalid_layer =
            summary.first_invalid_layer;
        status->prefill_first_invalid_expert =
            summary.first_invalid_expert;
        status->prefill_first_invalid_reasons =
            summary.first_invalid_reasons;
        status->prefill_first_invalid_flags =
            summary.first_invalid_flags;
        status->prefill_first_invalid_resident_mask =
            summary.first_invalid_resident_mask;
        status->prefill_first_invalid_owner =
            summary.first_invalid_owner;
    }

    __device__ __forceinline__ void
    rebalance_publish_transfer_slot_claim_summary(
        DeviceMoERebalanceStatusView *status,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count)
    {
        rebalance_publish_transfer_slot_claim_summary_fields(
            status,
            rebalance_transfer_slot_claim_summary(
                runtime_layers,
                config,
                local_transfer_slots,
                local_transfer_slot_count));
    }

    /** Return the variable-length entry array stored directly after its header. */
    __device__ __forceinline__ DeviceMoETransferSlotClaimIndexEntryView *
    rebalance_transfer_slot_claim_index_entries(
        DeviceMoETransferSlotClaimIndexView *index)
    {
        return reinterpret_cast<DeviceMoETransferSlotClaimIndexEntryView *>(
            index + 1);
    }

    /** Return the read-only variable-length entry array after its header. */
    __device__ __forceinline__ const DeviceMoETransferSlotClaimIndexEntryView *
    rebalance_transfer_slot_claim_index_entries(
        const DeviceMoETransferSlotClaimIndexView *index)
    {
        return reinterpret_cast<const DeviceMoETransferSlotClaimIndexEntryView *>(
            index + 1);
    }

    /** Validate one runtime layer before it participates in storage liveness. */
    __device__ __forceinline__ bool
    rebalance_claim_index_runtime_layer_ok(
        const DeviceMoELayerRuntimeView &runtime,
        const DeviceMoERebalanceConfigView &config)
    {
        return runtime.active_bank <= 1u &&
               runtime.active_epoch != 0u &&
               runtime.expert_count == config.num_experts &&
               runtime.participant_id == config.participant_id &&
               runtime.participant_count == config.participant_count;
    }

    /** Diagnose a valid-range claim against its physical directory entry. */
    __device__ __forceinline__ uint32_t
    rebalance_transfer_slot_directory_claim_error(
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        uint32_t slot,
        uint32_t layer,
        uint32_t expert,
        const DeviceMoEExpertDescriptorView &descriptor,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!local_transfer_slots ||
            local_transfer_slot_count !=
                config.transfer_slot_directory_capacity ||
            slot >= local_transfer_slot_count)
        {
            return llaminar2::moe_rebalance_policy::
                TransferSlotClaimDirectoryIdentityMismatch;
        }

        const auto &directory_entry = local_transfer_slots[slot];
        const bool physical_identity_valid =
            directory_entry.participant == config.participant_id &&
            directory_entry.slot_index == slot &&
            directory_entry.descriptor.local_slot ==
                descriptor.local_slot &&
            directory_entry.generation != 0xffffffffu;
        if (!physical_identity_valid)
        {
            return llaminar2::moe_rebalance_policy::
                TransferSlotClaimDirectoryIdentityMismatch;
        }

        const bool occupant_matches =
            directory_entry.layer == layer &&
            directory_entry.expert == expert &&
            directory_entry.descriptor.logical_expert_id ==
                static_cast<int32_t>(expert) &&
            directory_entry.descriptor.gate.payload ==
                descriptor.gate.payload &&
            directory_entry.descriptor.gate.scales ==
                descriptor.gate.scales &&
            directory_entry.descriptor.up.payload ==
                descriptor.up.payload &&
            directory_entry.descriptor.up.scales ==
                descriptor.up.scales &&
            directory_entry.descriptor.down.payload ==
                descriptor.down.payload &&
            directory_entry.descriptor.down.scales ==
                descriptor.down.scales;
        return occupant_matches
                   ? 0u
                   : llaminar2::moe_rebalance_policy::
                         TransferSlotClaimOccupantMismatch;
    }


    /** Authenticate that an index was rebuilt for this exact projection shape. */
    __device__ __forceinline__ bool rebalance_transfer_slot_claim_index_ok(
        const DeviceMoETransferSlotClaimIndexView *index,
        const DeviceMoERebalanceConfigView &config,
        uint32_t local_transfer_slot_count)
    {
        return index &&
               index->magic == kDeviceMoERebalanceMagic &&
               index->version == kDeviceMoERebalanceVersion &&
               index->ready == 1u &&
               index->participant_id == config.participant_id &&
               index->participant_count == config.participant_count &&
               index->num_layers == config.num_layers &&
               index->num_experts == config.num_experts &&
               index->slot_count == local_transfer_slot_count &&
               index->invalid_runtime_layers == 0u &&
               index->summary.invalid_claims == 0u &&
               index->summary.duplicate_claims == 0u;
    }

    __device__ __forceinline__ bool runtime_expert_replicated(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert_id)
    {
        if (!runtime)
            return false;
        const uint32_t resident_mask =
            runtime_expert_resident_mask(runtime, bank, expert_id);
        return runtime_resident_count(resident_mask, runtime->participant_count) > 1;
    }

    /**
     * @brief Retire one CUDA-local expert payload as an atomic publication unit.
     *
     * Matrix pointers, stable slot identity, resident mask, and execution flags
     * describe one physical allocation. Every ownership departure and slot reuse
     * must clear them together through this helper.
     */
    __device__ __forceinline__ void
    rebalance_retire_local_payload_publication(
        DeviceMoEExpertDescriptorView &descriptor,
        uint32_t &resident_mask,
        uint32_t local_participant_bit)
    {
        constexpr uint32_t kLocalPayloadFlags =
            kDeviceMoEFlagValid |
            kDeviceMoEFlagResident |
            kDeviceMoEFlagReplicated |
            kDeviceMoEFlagLocalCompute |
            kDeviceMoEFlagTransferSlot;
        llaminar2::moe_rebalance_policy::retireLocalPayloadPublication(
            descriptor,
            resident_mask,
            local_participant_bit,
            kLocalPayloadFlags);
    }

    __device__ __forceinline__ uint32_t runtime_multi_resident_expert_count(
        const DeviceMoEPlacementBankView &bank)
    {
        return bank.multi_resident_expert_count;
    }

    __device__ __forceinline__ uint32_t runtime_compute_multi_resident_expert_count(
        const DeviceMoEPlacementBankView &bank,
        uint32_t expert_count,
        uint32_t participant_count)
    {
        uint32_t count = 0u;
        const uint32_t valid_mask = runtime_valid_participant_mask(participant_count);
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            if (runtime_resident_count(bank.resident_participant_mask[expert] & valid_mask,
                                       participant_count) > 1)
            {
                ++count;
            }
        }
        return count;
    }

    __device__ __forceinline__ bool runtime_has_local_hot_replica(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        uint32_t expert_count)
    {
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
            const uint32_t resident_mask =
                runtime_expert_resident_mask(runtime, bank, static_cast<int>(expert));
            const bool local_resident =
                (resident_mask & runtime_participant_bit(
                                     static_cast<int>(runtime->participant_id))) != 0u;
            const bool owner_local =
                desc.owner_participant == static_cast<int32_t>(runtime->participant_id);
            const bool multi_resident =
                (resident_mask & (resident_mask - 1u)) != 0u;
            if (local_resident && !owner_local && multi_resident)
                return true;
        }
        return false;
    }

    __device__ __forceinline__ int runtime_choose_replicated_participant(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        int selected_slot,
        int expert_id,
        int32_t logical_position,
        const int *load)
    {
        const uint32_t participant_count = runtime->participant_count;
        const uint32_t resident_mask =
            runtime_expert_resident_mask(runtime, bank, expert_id);
        const uint64_t tie_turn =
            llaminar2::least_loaded_ep::residentAssignmentTieTurn(
                logical_position,
                expert_id,
                selected_slot);
        const int owner = runtime_expert_owner(bank, expert_id);
        const uint32_t fallback =
            owner >= 0 && static_cast<uint32_t>(owner) < participant_count
                ? static_cast<uint32_t>(owner)
                : runtime->participant_id;
        return static_cast<int>(
            llaminar2::least_loaded_ep::
                selectBatchInvariantResidentParticipant(
                    resident_mask,
                    load,
                    participant_count,
                    fallback,
                    tie_turn));
    }

    __device__ __forceinline__ uint64_t runtime_participant_load_spread(
        const int *load,
        uint32_t participant_count)
    {
        if (!load || participant_count == 0u)
            return 0ULL;
        int min_load = load[0];
        int max_load = load[0];
        for (uint32_t participant = 1; participant < participant_count; ++participant)
        {
            const int value = load[participant];
            if (value < min_load)
                min_load = value;
            if (value > max_load)
                max_load = value;
        }
        return static_cast<uint64_t>(max_load - min_load);
    }

    /** @brief Resolve only the original member's complete fragment; broken admission traps before routing. */
    __device__ __forceinline__ llaminar2::MoEIntermediateShardRowSelection
    runtime_intermediate_shard_row(const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank, int expert)
    {
        const auto selection = llaminar2::selectMoEIntermediateShardRow(
            runtime->intermediate_shard, bank, expert, static_cast<int>(runtime->participant_id));
        if (selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Invalid)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("original intermediate shard row has invalid family, route or local payload");
        }
        return selection;
    }

    /**
     * @brief Read the device-owned live count for a physical one-row route.
     * @param active_row_count_device Optional stable captured INT32 address; null means one live row.
     * @return Whether the row is live; values outside [0,1] are fatal before publication.
     */
    __device__ __forceinline__ bool runtime_decode_row_active(const int32_t *active_row_count_device)
    {
        const int32_t count = active_row_count_device ? *active_row_count_device : 1;
        if (count < 0 || count > 1)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("one-row MoE routing received a live count outside [0,1]");
            return false;
        }
        return count == 1;
    }

    /**
     * @brief Authenticate the immutable execution policy even for an empty counted row.
     * @param runtime Original captured runtime containing its closed source binding.
     * @param row_execution_policy Captured row policy, which must agree with that binding.
     */
    __device__ __forceinline__ void runtime_require_decode_row_policy(
        const DeviceMoELayerRuntimeView *runtime,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy)
    {
        if ((row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
             row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
             row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal) ||
            runtime->intermediate_shard.enabled() !=
                (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal) ||
            (runtime->intermediate_shard.enabled() && runtime->current_batch_llep_transient_bank_active != 0u))
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("decode row execution policy conflicts with the original fragment binding");
            return;
        }
    }

    /** @brief Temporary assignment state owned by the single decode route publisher. */
    struct RuntimeDecodeDispatchScratch
    {
        bool local_compute_flags[kMaxTopK];
        int assigned_participants[kMaxTopK];
        int default_load[kDeviceMoEMaxParticipants];
        int actual_load[kDeviceMoEMaxParticipants];
    };

    __device__ __forceinline__ void runtime_resolve_decode_dispatch(
        DeviceMoELayerRuntimeView *runtime,
        uint32_t execution_bank,
        const int *selected_experts,
        int num_experts,
        int top_k,
        int32_t logical_position,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        bool record_balance,
        RuntimeDecodeDispatchScratch &scratch)
    {
        if (!runtime_shape_ok(
                runtime, execution_bank, num_experts, top_k) ||
            !selected_experts)
        {
            return;
        }

        runtime_require_decode_row_policy(runtime, row_execution_policy);

        auto &local_compute_flags = scratch.local_compute_flags;
        auto &assigned_participants = scratch.assigned_participants;
        for (int slot = 0; slot < top_k; ++slot)
        {
            local_compute_flags[slot] = false;
            assigned_participants[slot] = -1;
        }

        const auto &bank = runtime_placement_banks(runtime)[execution_bank];
        const uint32_t participant_count = runtime->participant_count;
        const uint32_t participant_id = runtime->participant_id;
        if (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                if (selected_experts[slot] == -1) continue;
                const auto selected = runtime_intermediate_shard_row(runtime, bank, selected_experts[slot]);
                local_compute_flags[slot] = selected.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
                assigned_participants[slot] = selected.participant;
            }
            return;
        }

        const bool valid_participants =
            participant_count > 0u &&
            participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
            participant_id < participant_count;
        const bool has_multi_resident_experts =
            runtime_multi_resident_expert_count(bank) != 0u;
        if (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal || !has_multi_resident_experts)
        {
            /*
             * A fully replicated graph owns complete local expert payloads and
             * explicitly forbids participant assignment.  Preserve every
             * selected local route on every participant; otherwise a mirrored
             * MTP sidecar would silently compute complementary partial rows.
             */
            for (int slot = 0; slot < top_k; ++slot)
            {
                const int expert_id = selected_experts[slot];
                if (expert_id < 0 || expert_id >= num_experts)
                    continue;
                const bool local =
                    bank.local_compute_mask[expert_id] != 0u;
                assigned_participants[slot] = row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal
                                                  ? static_cast<int>(participant_id)
                                                  : runtime_expert_owner(
                                                        bank, expert_id);
                local_compute_flags[slot] = local;
            }
            return;
        }
        if (logical_position < 0)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "replicated decode routing requires a logical position");
        }
        const bool track_balance =
            record_balance &&
            valid_participants;

        auto &default_load = scratch.default_load;
        auto &actual_load = scratch.actual_load;
#pragma unroll
        for (int participant = 0; participant < kDeviceMoEMaxParticipants; ++participant)
        {
            default_load[participant] = 0;
            actual_load[participant] = 0;
        }
        bool eligible = false;
        bool used_remote_replica = false;
        uint64_t selected_expert_slots = 0ULL;
        uint64_t replicated_selected_expert_slots = 0ULL;
        if (track_balance)
            ++runtime->router_hot_cache_active_dispatches;

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_experts[slot];
            if (expert_id < 0 || expert_id >= num_experts)
                continue;
            if (track_balance)
                ++selected_expert_slots;

            const bool local_resident = bank.local_compute_mask[expert_id] != 0u;
            const bool replicated = runtime_expert_replicated(runtime, bank, expert_id);
            const int owner = runtime_expert_owner(bank, expert_id);
            const bool owner_valid =
                valid_participants &&
                owner >= 0 &&
                owner < static_cast<int>(participant_count);
            assigned_participants[slot] = owner_valid ? owner : -1;

            if (track_balance && owner_valid)
                ++default_load[owner];

            if (!replicated || !valid_participants || !owner_valid)
            {
                local_compute_flags[slot] = local_resident;
                if (!replicated && owner_valid)
                    ++actual_load[owner];
                continue;
            }

            if (track_balance)
            {
                const uint32_t resident_mask =
                    runtime_expert_resident_mask(runtime, bank, expert_id);
                if (runtime_resident_count(resident_mask, participant_count) > 1)
                {
                    eligible = true;
                    ++replicated_selected_expert_slots;
                }
            }
        }

        if (track_balance)
        {
            runtime->router_hot_cache_selected_expert_slots += selected_expert_slots;
            runtime->router_hot_cache_replicated_selected_expert_slots +=
                replicated_selected_expert_slots;
            if (!eligible)
                ++runtime->router_hot_cache_miss_dispatches;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_experts[slot];
            if (expert_id < 0 || expert_id >= num_experts ||
                !runtime_expert_replicated(runtime, bank, expert_id))
            {
                continue;
            }

            const bool local_resident = bank.local_compute_mask[expert_id] != 0u;
            const int owner = runtime_expert_owner(bank, expert_id);
            if (!valid_participants ||
                owner < 0 ||
                owner >= static_cast<int>(participant_count))
            {
                local_compute_flags[slot] = local_resident;
                continue;
            }

            const int selected_participant =
                runtime_choose_replicated_participant(
                    runtime,
                    bank,
                    slot,
                    expert_id,
                    logical_position,
                    actual_load);
            if (selected_participant >= 0 &&
                selected_participant < static_cast<int>(participant_count))
            {
                assigned_participants[slot] = selected_participant;
                local_compute_flags[slot] =
                    local_resident &&
                    selected_participant == static_cast<int>(participant_id);
                ++actual_load[selected_participant];
                if (track_balance && selected_participant != owner)
                    used_remote_replica = true;
            }
            else if (track_balance)
            {
                ++actual_load[owner];
            }
        }

        if (!track_balance || !eligible)
            return;

        const uint64_t default_spread =
            runtime_participant_load_spread(default_load, participant_count);
        const uint64_t actual_spread =
            runtime_participant_load_spread(actual_load, participant_count);
        ++runtime->router_hot_cache_eligible_dispatches;
        runtime->router_hot_cache_default_load_spread_total += default_spread;
        runtime->router_hot_cache_actual_load_spread_total += actual_spread;
        if (!used_remote_replica)
            return;
        ++runtime->router_hot_cache_used_dispatches;
        if (actual_spread < default_spread)
        {
            ++runtime->router_hot_cache_improved_dispatches;
            runtime->router_hot_cache_load_spread_improvement_total +=
                default_spread - actual_spread;
        }
    }


    __device__ __forceinline__ bool grouped_desc_supports_codebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:
        case 4:
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ bool grouped_desc_requires_mins(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        case 11: // Mandatory original IQ3_S source nibble.
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
        case 16:
        case 17:
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ bool grouped_desc_requires_emins(uint8_t codebook_id)
    {
        return codebook_id == 10;
    }

    __device__ __forceinline__ bool native_quantised_desc_shape_ok_dynamic(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int N,
        int K)
    {
        return desc.payload &&
               desc.scales &&
               desc.n == N &&
               desc.k == K &&
               desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
               grouped_desc_supports_codebook(desc.codebook_id) &&
               (!grouped_desc_requires_mins(desc.codebook_id) || desc.mins) &&
               (!grouped_desc_requires_emins(desc.codebook_id) || desc.emins);
    }

    template <uint8_t CodebookId>
    __device__ __forceinline__ bool native_quantised_desc_shape_ok(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int N,
        int K)
    {
        if constexpr (CodebookId == kMixedCodebookSentinel)
        {
            return native_quantised_desc_shape_ok_dynamic(desc, N, K);
        }
        else
        {
            return desc.payload &&
                   desc.scales &&
                   desc.n == N &&
                   desc.k == K &&
                   desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
                   llaminar2::cuda_native_quantised::native_quantised_descriptor_codebook_matches<CodebookId>(desc.codebook_id) &&
                   (!(llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::requires_secondary_metadata) ||
                    (CodebookId == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook && desc.codebook_id == 4) || desc.mins) &&
                   (!llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::is_dual_scale_asym ||
                    desc.emins);
        }
    }

    /**
     * @brief Admit independent IQ4 LUT factor programs in one selected family.
     * @tparam CodebookId Selected physical family;24 accepts both IQ4 payload members.
     * @param gate Original gate descriptor with explicit physical format identity.
     * @param up Original up descriptor, either the same format or IQ4 peer.
     * @param n Logical output columns.
     * @param k Logical input columns.
     * @return True only for complete descriptors with the declared payload family.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ bool native_quantised_pair_shape_ok(
        const DeviceNativeQuantisedMatrixDesc &gate,
        const DeviceNativeQuantisedMatrixDesc &up, int n, int k)
    {
        if (!native_quantised_desc_shape_ok<CodebookId>(gate, n, k))
            return false;
        if constexpr (llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::is_iq4_lut)
            return up.codebook_id == 4
                ? native_quantised_desc_shape_ok<4>(up, n, k)
                : up.codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook &&
                  native_quantised_desc_shape_ok<llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook>(up, n, k);
        return native_quantised_desc_shape_ok<CodebookId>(up, n, k);
    }

    __device__ __forceinline__ float silu(float x)
    {
        return llaminar2::device_swiglu_contract::siluValue(x);
    }

    bool finishLaunch(const char *name)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool finishGroupedPrefillLaunch(const char *name, cudaStream_t stream)
    {
        /*
         * Every grouped-prefill launch participates in the caller's explicit
         * stream order.  Completion is published at the graph or stage
         * ownership boundary; individual kernels must never block the host.
         */
        (void)stream;
        return finishLaunch(name);
    }

    inline int blocksFor(int count)
    {
        return (count + kThreads - 1) / kThreads;
    }


    __device__ __forceinline__ bool moe_topk_pair_better(
        float candidate_value,
        int candidate_id,
        float best_value,
        int best_id)
    {
        return candidate_value > best_value ||
               (candidate_value == best_value && candidate_id < best_id);
    }

    __device__ void moe_select_topk_probabilities_block(
        float *__restrict__ values,
        int num_experts,
        int top_k,
        int *__restrict__ selected,
        float *__restrict__ selected_weights,
        float *__restrict__ reductions,
        int *__restrict__ reduction_indices)
    {
        constexpr unsigned kFullWarpMask = 0xffffffffu;
        constexpr int kWarpWidth = 32;
        const int lane = threadIdx.x & (kWarpWidth - 1);
        const int warp_id = threadIdx.x / kWarpWidth;
        (void)reductions;
        (void)reduction_indices;

        /*
         * Router rows are independent and the grid supplies thousands of
         * blocks during prefill, so block-level parallelism is already ample.
         * Within a row, one warp can scan all supported experts with a stable
         * strided tournament.  This removes the two CTA barriers that the
         * former eight-warp merge paid for every selected slot.  The remaining
         * block barrier publishes the complete selected table once.
         *
         * This is not a reduced-capability path: lanes stride over the whole
         * `num_experts` range, including the supported 257..1024 geometries.
         */
        if (warp_id == 0)
        {
            for (int k = 0; k < top_k; ++k)
            {
                float best_value = -INFINITY;
                int best = kMaxExperts;
                for (int expert = lane;
                     expert < num_experts;
                     expert += kWarpWidth)
                {
                    const float value = values[expert];
                    if (moe_topk_pair_better(
                            value,
                            expert,
                            best_value,
                            best))
                    {
                        best_value = value;
                        best = expert;
                    }
                }

                for (int offset = kWarpWidth / 2;
                     offset > 0;
                     offset >>= 1)
                {
                    const float other_value =
                        __shfl_down_sync(
                            kFullWarpMask,
                            best_value,
                            offset);
                    const int other_id =
                        __shfl_down_sync(
                            kFullWarpMask,
                            best,
                            offset);
                    if (moe_topk_pair_better(
                            other_value,
                            other_id,
                            best_value,
                            best))
                    {
                        best_value = other_value;
                        best = other_id;
                    }
                }

                const int winner =
                    __shfl_sync(kFullWarpMask, best, 0);
                if (lane == 0)
                {
                    selected[k] = best;
                    selected_weights[k] = best_value;
                }
                for (int expert = lane;
                     expert < num_experts;
                     expert += kWarpWidth)
                {
                    if (expert == winner)
                        values[expert] = -INFINITY;
                }
                __syncwarp(kFullWarpMask);
            }
        }
        __syncthreads();
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    /**
     * @brief Merge eight warp-local ordered lists in one warp.
     * Map real nonnegative probabilities to keys above the consumed sentinel.
     * Both reductions apply original smallest-expert tie breaking. One CTA
     * join publishes the local lists before the final ordered merge.
     * @param probabilities Finished nonnegative FP32 probabilities.
     * @param top_k Number of slots in the original route ledger.
     * @param selected Canonical expert IDs in probability and tie order.
     * @param weights Canonical rounded probabilities before route normalization.
     * @param warp_values Shared storage for the sorted keys of all eight warps.
     * @param warp_experts Shared storage for the corresponding expert IDs.
     * @pre The probability bank contains 512 original experts and the CTA has
     *      256 lanes. Before another CTA lane reads the output, the caller
     *      joins the warp containing the final merge.
     */
    __device__ __forceinline__ void retainedTopK512(const float* probabilities, int top_k,
        int* selected, float* weights, float* warp_values, int* warp_experts)
    {
        constexpr unsigned lanes = 0xffffffffu;
        constexpr int width = 256;
        constexpr int experts = 512;
        constexpr int warps = width / 32;
        static_assert(warps * kMaxTopK <= width);
        const int first = threadIdx.x;
        const int second = first + width;
        const int warp = first / 32;
        const int lane = first % 32;
        const auto order_key = [](float value) {
            return value == 0.0f ? 1u : __float_as_uint(value) + 1u;
        };
        unsigned low = order_key(probabilities[first]);
        unsigned high = order_key(probabilities[second]);
        for (int slot = 0; slot < top_k; ++slot)
        {
            const bool use_second = high > low;
            const unsigned value = use_second ? high : low;
            const int expert = use_second ? second : first;
            const unsigned best = __reduce_max_sync(lanes, value);
            const int winner = __reduce_min_sync(lanes, value == best ? expert : experts);
            if (lane == 0)
            {
                warp_values[slot * warps + warp] = __uint_as_float(best);
                warp_experts[slot * warps + warp] = winner;
            }
            if (winner == first)
                low = 0;
            if (winner == second)
                high = 0;
        }
        __syncthreads();
        if (warp != 0)
            return;

        int head = 0;
        for (int slot = 0; slot < top_k; ++slot)
        {
            const int address = head * warps + lane;
            const unsigned value = lane < warps ? __float_as_uint(warp_values[address]) : 0u;
            const int expert = lane < warps ? warp_experts[address] : experts;
            const unsigned best = __reduce_max_sync(lanes, value);
            const int winner = __reduce_min_sync(lanes, value == best ? expert : experts);
            if (lane == 0)
            {
                selected[slot] = winner;
                weights[slot] = probabilities[winner];
            }
            if (winner == expert)
                ++head;
        }
    }
#endif

    __device__ __forceinline__ int clamp_effective_seq_len(
        int seq_len,
        const int *__restrict__ device_effective_seq_len)
    {
        if (!device_effective_seq_len)
            return seq_len;
        const int raw = *device_effective_seq_len;
        return raw < 0 ? 0 : (raw > seq_len ? seq_len : raw);
    }

} // namespace
extern "C" bool cudaMoE_internal_quantize_hidden_blockwise(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales, int K, void *stream);
