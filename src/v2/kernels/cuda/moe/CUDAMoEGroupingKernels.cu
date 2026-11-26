/**
 * @file CUDAMoEGroupingKernels.cu
 * @brief CUDA MoE token grouping, active-expert compaction, prefill LLEP assignment, and descriptor materialization kernels.
 */
#include "CUDAMoERebalanceCommon.cuh"

namespace
{
    /**
     * @brief Initialize grouping arrays in one coalesced, barrier-free launch.
     * @param binding Exact borrowed workspace arrays and independent extents.
     */
    __global__ void initialize_grouping_kernel(llaminar2::MoEGroupingInitialization binding)
    {
        binding.resetIndex(blockIdx.x * blockDim.x + threadIdx.x);
    }

    __global__ void count_per_expert_kernel(
        const int *__restrict__ routing_indices,
        int *__restrict__ expert_counts,
        int total_slots, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert >= 0 && expert < num_experts)
            atomicAdd(expert_counts + expert, 1);
    }

    /**
     * @brief Publish exact expert offsets with a cooperative integer scan.
     * @param expert_counts Immutable per-expert route counts.
     * @param expert_offsets Output offsets in ascending expert order.
     * @param num_experts Admitted expert count, including partial block tails.
     */
    __global__ void exclusive_scan_kernel(const int *__restrict__ expert_counts, int *__restrict__ expert_offsets, int num_experts)
    {
        __shared__ llaminar2::device_integer_scan::Scratch<kThreads> scan;
        llaminar2::device_integer_scan::arrayExclusive(
            expert_counts, expert_offsets, num_experts, scan);
    }

    __global__ void scatter_tokens_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ write_heads,
        const int *__restrict__ expert_offsets,
        int *__restrict__ grouped_token_indices,
        float *__restrict__ grouped_weights,
        int total_slots, int top_k, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;
        const int local = atomicAdd(write_heads + expert, 1);
        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        grouped_weights[dest] = routing_weights[idx];
    }

    constexpr int kActiveExpertWarpSize = 32;
    constexpr int kActiveExpertWarpCount = kThreads / kActiveExpertWarpSize;

    /**
     * @brief Persistent shared state for stable active-expert compaction.
     *
     * Expert ids are consumed downstream as an ordered compact list.  A plain
     * atomic append would make that order scheduling-dependent, so each warp
     * first counts its active lanes and lane zero assigns deterministic warp
     * offsets.  `total_active` carries the ordered prefix between 256-expert
     * chunks when the non-runtime helper is used with a larger expert domain.
     */
    struct ActiveExpertCompactionScratch
    {
        int warp_counts[kActiveExpertWarpCount];
        int warp_offsets[kActiveExpertWarpCount];
        int chunk_base;
        int total_active;
    };

    /**
     * @brief Initialize one block's stable compaction state.
     *
     * Every caller must execute this collectively before compacting a chunk.
     */
    __device__ __forceinline__ void initialize_active_expert_compaction(
        ActiveExpertCompactionScratch &scratch)
    {
        if (threadIdx.x == 0)
            scratch.total_active = 0;
        __syncthreads();
    }

    /**
     * @brief Append one 256-expert chunk in ascending expert-id order.
     *
     * @param is_active True when this lane's expert belongs in the output.
     * @param expert_id Ascending expert id represented by this lane.
     * @param max_active_experts Capacity of @p active_expert_ids.
     * @param active_expert_ids Ordered compact output, padded separately.
     * @param scratch Block-owned prefix state shared across chunks.
     */
    __device__ __forceinline__ void compact_active_expert_chunk(
        bool is_active,
        int expert_id,
        int max_active_experts,
        int *__restrict__ active_expert_ids,
        ActiveExpertCompactionScratch &scratch)
    {
        const int lane = static_cast<int>(threadIdx.x) &
                         (kActiveExpertWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) /
                         kActiveExpertWarpSize;
        const unsigned int votes = __ballot_sync(0xffffffffu, is_active);

        if (lane == 0)
            scratch.warp_counts[warp] = __popc(votes);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int chunk_active = 0;
            scratch.chunk_base = scratch.total_active;
            for (int current_warp = 0;
                 current_warp < kActiveExpertWarpCount;
                 ++current_warp)
            {
                scratch.warp_offsets[current_warp] = chunk_active;
                chunk_active += scratch.warp_counts[current_warp];
            }
            scratch.total_active += chunk_active;
        }
        __syncthreads();

        const unsigned int lower_lanes =
            lane == 0 ? 0u : ((1u << lane) - 1u);
        const int rank_in_warp = __popc(votes & lower_lanes);
        const int output_slot = scratch.chunk_base +
                                scratch.warp_offsets[warp] +
                                rank_in_warp;
        if (is_active && output_slot < max_active_experts)
            active_expert_ids[output_slot] = expert_id;

        // The next chunk reuses every shared field, so all writes must retire.
        __syncthreads();
    }

    /**
     * @brief Fill unused compact-list capacity with the canonical -1 sentinel.
     */
    __device__ __forceinline__ void pad_active_expert_list(
        int *__restrict__ active_expert_ids,
        int max_active_experts,
        const ActiveExpertCompactionScratch &scratch)
    {
        const int retained_active =
            scratch.total_active < max_active_experts
                ? scratch.total_active
                : max_active_experts;
        for (int slot = retained_active + static_cast<int>(threadIdx.x);
             slot < max_active_experts;
             slot += static_cast<int>(blockDim.x))
        {
            active_expert_ids[slot] = -1;
        }
    }

    /**
     * @brief Shared state for one complete runtime grouped-plan publication.
     *
     * Descriptor readiness and active-list compaction are collective block
     * operations. Keeping their shared state in one explicit object lets both
     * the complete descriptor publisher and the fused verifier grouping
     * publisher invoke exactly the same device implementation.
     */
    struct RuntimePrefillPlanPublicationScratch
    {
        int first_invalid_expert;
        ActiveExpertCompactionScratch active_experts;
    };

    /**
     * @brief Publish one expert's compact projection descriptors from a bank.
     *
     * Each lane owns exactly one expert id.  Keeping this operation separate
     * from active-list compaction allows the fused verifier transaction to
     * overlap descriptor publication with its count/offset publication while
     * the selected decode transaction continues to use the same exact
     * readiness and zero-fill contract.
     *
     * @param bank Active immutable placement bank selected by the caller.
     * @param local_bit Bit identifying the local participant in residency masks.
     * @param expert Expert id owned by this lane.
     * @param num_experts Number of logical experts in the layer.
     * @param gate_descs Compact gate descriptor table to publish.
     * @param up_descs Compact up descriptor table to publish.
     * @param down_descs Compact down descriptor table to publish.
     * @param expected_format Storage format shared by the layer's projections.
     * @param expected_projections Exact movable payload required by this plan.
     * @param gate_contract Fixed Gate/Up geometry validated before publishing both planes.
     * @return True when the expert is locally resident and compute-ready.
     */
    __device__ __forceinline__ bool publish_runtime_expert_descriptors_lane(
        const DeviceMoELayerRuntimeView *runtime,
        const DeviceMoEPlacementBankView &bank,
        uint32_t local_bit,
        int expert,
        int num_experts,
        DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ down_descs,
        llaminar2::DeviceMoEWeightFormat expected_format,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract = {})
    {
        if (expert >= num_experts)
            return false;

        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        // Gate/up-only banks have no movable down descriptor. The immutable
        // column bank owns down; absence alone never selects split execution.
        const bool descriptor_ready =
            llaminar2::deviceMoEExpertProjectionReady(desc, expected_projections);
        const bool local_ready = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
            ? runtime_intermediate_shard_row(runtime, bank, expert).disposition ==
                  llaminar2::MoEIntermediateShardRowDisposition::Local && desc.weight_format == expected_format
            : bank.local_compute_mask[expert] != 0u &&
            (bank.resident_participant_mask[expert] & local_bit) != 0u &&
            desc.local_slot >= 0 &&
            desc.weight_format == expected_format && descriptor_ready;
        if (expected_format ==
            llaminar2::DeviceMoEWeightFormat::NativeQuantised)
        {
            if (local_ready && gate_contract.enabled() &&
                (!gate_contract.accepts(desc.gate) || !gate_contract.accepts(desc.up)))
            {
                printf("qfn_gate_plan_invalid expert=%d expected_codebook=%u "
                       "gate_n=%d gate_k=%d gate_codebook=%u up_n=%d up_k=%d up_codebook=%u\n",
                       expert, static_cast<unsigned>(gate_contract.codebook()),
                       desc.gate.n, desc.gate.k, static_cast<unsigned>(desc.gate.codebook_id),
                       desc.up.n, desc.up.k, static_cast<unsigned>(desc.up.codebook_id));
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("QFN Gate/Up runtime plan has incompatible native operands");
                return false;
            }
            gate_descs[expert] = local_ready
                                     ? desc.gate
                                     : DeviceNativeQuantisedMatrixDesc{};
            up_descs[expert] = local_ready
                                   ? desc.up
                                   : DeviceNativeQuantisedMatrixDesc{};
            down_descs[expert] = local_ready
                                     ? desc.down
                                     : DeviceNativeQuantisedMatrixDesc{};
        }
        else
        {
            auto *floating_gate_descs =
                reinterpret_cast<llaminar2::DeviceMoEFloatingMatrixDesc *>(
                    gate_descs);
            auto *floating_up_descs =
                reinterpret_cast<llaminar2::DeviceMoEFloatingMatrixDesc *>(
                    up_descs);
            auto *floating_down_descs =
                reinterpret_cast<llaminar2::DeviceMoEFloatingMatrixDesc *>(
                    down_descs);
            floating_gate_descs[expert] = local_ready
                                              ? desc.floating_gate
                                              : llaminar2::DeviceMoEFloatingMatrixDesc{};
            floating_up_descs[expert] = local_ready
                                            ? desc.floating_up
                                            : llaminar2::DeviceMoEFloatingMatrixDesc{};
            floating_down_descs[expert] = local_ready
                                              ? desc.floating_down
                                              : llaminar2::DeviceMoEFloatingMatrixDesc{};
        }
        return local_ready;
    }

    /**
     * @brief Refresh only descriptors consumed by this exact one-row route packet.
     *
     * The arena retains a full expert-indexed table, but this invocation owns only
     * its selected entries. Every later consumer uses the same immutable ID packet
     * on this stream. Previously selected entries need no sweep: selecting them
     * again necessarily refreshes them from the currently acquired placement bank.
     * A selected remote expert is explicitly zeroed by the shared publisher.
     *
     * @param runtime Participant runtime whose acquired epoch owns the source bank.
     * @param expert_ids Device-owned selected IDs, with -1 for inactive routes.
     * @param gate_descs Existing graph-owned expert-indexed gate descriptor storage.
     * @param up_descs Existing graph-owned expert-indexed up descriptor storage.
     * @param down_descs Existing graph-owned expert-indexed down descriptor storage.
     * @param num_experts Logical source/table extent, independent of launch size.
     * @param top_k Exact route packet width; launch one lane per selected slot.
     * @param format Original native or floating descriptor family.
     * @param projections Exact admitted movable projection contract.
     */
    __global__ void materialize_runtime_selected_decode_descriptors_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ down_descs,
        int num_experts, int top_k, llaminar2::DeviceMoEWeightFormat format,
        llaminar2::DeviceMoEProjectionSet projections)
    {
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (!runtime || execution_bank > 1u ||
            runtime->intermediate_shard.enabled() !=
                (projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard))
        {
            if (threadIdx.x == 0) FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("selected decode descriptors require the acquired bank and projection contract");
            return;
        }
        const int slot = static_cast<int>(threadIdx.x);
        if (slot >= top_k) return;
        const int expert = expert_ids[slot];
        if (expert == -1) return;
        if (expert < 0 || expert >= num_experts)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("selected decode descriptor ID lies outside the admitted source");
            return;
        }
        // Ordinary top-k IDs are unique. An explicit packet may repeat an ID;
        // its first slot alone owns publication, avoiding even identical-write races.
        for (int earlier = 0; earlier < slot; ++earlier)
            if (expert_ids[earlier] == expert) return;
        const auto &bank = runtime_placement_banks(runtime)[execution_bank];
        const uint32_t local_bit =
            projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                ? 0u : runtime_participant_bit(static_cast<int>(runtime->participant_id));
        publish_runtime_expert_descriptors_lane(runtime, bank, local_bit, expert, num_experts,
            gate_descs, up_descs, down_descs, format, projections);
    }

    /**
     * @brief Materialize descriptor tables and the stable local active list.
     *
     * Every lane in the block must call this function. Ascending block-sized
     * chunks cover the complete expert domain even when it exceeds the launch
     * width. Descriptor publication reads one immutable bank entry per expert,
     * while active-list publication consumes grouped counts produced on the
     * stream or, in the fused small-M kernel, earlier in the same block. An
     * active expert without a ready local descriptor is a fatal lifecycle
     * violation because the immediately following grouped GEMM cannot execute
     * that row economically or correctly.
     */
    __device__ __forceinline__ void publish_runtime_prefill_plan_block(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ down_descs,
        int num_experts,
        int *__restrict__ active_expert_ids,
        int max_active_experts,
        llaminar2::DeviceMoEWeightFormat expected_format,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        RuntimePrefillPlanPublicationScratch &scratch)
    {
        const bool publish_active_experts = active_expert_ids != nullptr;
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (!runtime || !gate_descs || !up_descs || !down_descs ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            execution_bank > 1u ||
            !llaminar2::deviceMoEProjectionSetValid(expected_projections) ||
            (runtime && runtime->intermediate_shard.enabled() !=
                (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)) ||
            (expected_format !=
                 llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
             !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)) ||
            (publish_active_experts &&
             (!runtime->expert_counts || max_active_experts <= 0 ||
              max_active_experts > num_experts)))
        {
            if (threadIdx.x == 0)
            {
                const DeviceMoEPlacementBankView *const placement_banks =
                    runtime_durable_placement_banks(runtime);
                const llaminar2::DeviceMoEOverlayEpochTicket *const ticket =
                    runtime ? runtime->overlay_epoch_ticket : nullptr;
                const llaminar2::DeviceMoEOverlayEpochStatus *const status =
                    runtime ? runtime->overlay_epoch_status : nullptr;
                printf("runtime_prefill_invalid_contract "
                       "runtime=%p gate=%p up=%p down=%p experts=%d "
                       "active_ids=%p max_active=%d expert_counts=%p "
                       "expected_format=%u execution_bank=%u active_bank=%u "
                       "active_epoch=%u transient=%u ticket=%p "
                       "ticket_epoch=%llu ticket_selector=%llu "
                       "status=%p status_code=%u status_operation=%u "
                       "status_epoch=%llu status_selector=%llu status_bank=%u "
                       "status_state=%u placement_banks=%p bank0_epoch=%u "
                       "bank1_epoch=%u\n",
                       runtime,
                       gate_descs,
                       up_descs,
                       down_descs,
                       num_experts,
                       active_expert_ids,
                       max_active_experts,
                       runtime ? runtime->expert_counts : nullptr,
                       static_cast<unsigned>(expected_format),
                       execution_bank,
                       runtime ? runtime->active_bank : 0xffffffffu,
                       runtime ? runtime->active_epoch : 0u,
                       runtime
                           ? runtime->current_batch_llep_transient_bank_active
                           : 0xffffffffu,
                       ticket,
                       static_cast<unsigned long long>(
                           ticket ? ticket->epoch : 0u),
                       static_cast<unsigned long long>(
                           ticket ? ticket->selector : 0u),
                       status,
                       status ? status->code : 0xffffffffu,
                       status ? status->operation : 0xffffffffu,
                       static_cast<unsigned long long>(
                           status ? status->epoch : 0u),
                       static_cast<unsigned long long>(
                           status ? status->selector : 0u),
                       status ? status->bank : 0xffffffffu,
                       status ? status->observed_state : 0xffffffffu,
                       placement_banks,
                       placement_banks ? placement_banks[0].epoch : 0u,
                       placement_banks ? placement_banks[1].epoch : 0u);
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "runtime descriptor publication has an invalid device contract");
            }
            return;
        }

        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const uint32_t local_bit =
            expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                    ? 0u : runtime_participant_bit(static_cast<int>(runtime->participant_id));
        if (publish_active_experts)
            initialize_active_expert_compaction(scratch.active_experts);

        // The block width is a launch/resource choice, never the logical
        // expert capacity. Keep compaction state across ascending chunks so
        // later experts retain the same stable ordinal order as a serial scan.
        for (int chunk = 0; chunk < num_experts;
             chunk += static_cast<int>(blockDim.x))
        {
            const int expert = chunk + static_cast<int>(threadIdx.x);
            const bool local_ready = publish_runtime_expert_descriptors_lane(
                runtime,
                bank,
                local_bit,
                expert,
                num_experts,
                gate_descs,
                up_descs,
                down_descs,
                expected_format,
                expected_projections);

            // Descriptor-only decode still owes every later expert chunk. The
            // branch is uniform and skips only active-list collective work.
            if (!publish_active_experts)
                continue;

            const int count = expert < num_experts
                                  ? runtime->expert_counts[expert]
                                  : 0;
            const bool is_active = count > 0;
            if (threadIdx.x == 0)
                scratch.first_invalid_expert = num_experts;
            __syncthreads();
            if (is_active && !local_ready)
                atomicMin(&scratch.first_invalid_expert, expert);
            __syncthreads();

            if (scratch.first_invalid_expert < num_experts)
            {
                if (threadIdx.x == 0)
                {
                    const int invalid_expert = scratch.first_invalid_expert;
                    const DeviceMoEExpertDescriptorView &desc =
                        bank.experts[invalid_expert];
                    printf("runtime_prefill_active_expert_not_ready "
                           "participant=%u expert=%d count=%d execution_bank=%u "
                           "local_mask=%u resident_mask=%u local_bit=%u "
                           "local_slot=%d logical=%d owner=%d\\n",
                           runtime->participant_id,
                           invalid_expert,
                           runtime->expert_counts[invalid_expert],
                           execution_bank,
                           bank.local_compute_mask[invalid_expert],
                           bank.resident_participant_mask[invalid_expert],
                           local_bit,
                           desc.local_slot,
                           desc.logical_expert_id,
                           desc.owner_participant);
                    FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                        "runtime active expert is not locally resident and compute-ready");
                }
                return;
            }

            compact_active_expert_chunk(
                is_active,
                expert,
                max_active_experts,
                active_expert_ids,
                scratch.active_experts);
        }
        if (publish_active_experts)
        {
            pad_active_expert_list(
                active_expert_ids,
                max_active_experts,
                scratch.active_experts);
        }
    }

    /**
     * @brief Build a stable compact list from precomputed expert counts.
     *
     * One block processes the expert domain in ascending 256-id chunks.  The
     * output is therefore byte-identical to the former serial scan while count
     * reads, ballot classification, writes, and sentinel padding are parallel.
     */
    __global__ void build_active_expert_list_kernel(
        const int *__restrict__ expert_counts,
        int *__restrict__ active_expert_ids,
        int num_experts,
        int max_active_experts)
    {
        __shared__ ActiveExpertCompactionScratch scratch;
        initialize_active_expert_compaction(scratch);

        for (int chunk = 0; chunk < num_experts; chunk += blockDim.x)
        {
            const int expert = chunk + static_cast<int>(threadIdx.x);
            const bool is_active =
                expert < num_experts && expert_counts[expert] > 0;
            compact_active_expert_chunk(
                is_active,
                expert,
                max_active_experts,
                active_expert_ids,
                scratch);
        }
        pad_active_expert_list(
            active_expert_ids,
            max_active_experts,
            scratch);
    }

    __global__ void scatter_tokens_deterministic_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int total_slots,
        int top_k,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;

        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;

        int local = 0;
        for (int prev = 0; prev < idx; ++prev)
        {
            if (routing_indices[prev] == expert)
                ++local;
        }
        if (local >= expert_counts[expert])
            return;

        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        original_to_grouped[idx] = dest;
        original_expert_ids[idx] = expert;
        grouped_weights[dest] = routing_weights[idx];
    }

    /**
     * @brief Shared state for the verifier-sized expert prefix scan.
     *
     * The first scan component publishes each expert's stable route offset.
     * The optional second component publishes its rank among active experts.
     * Both use fixed warp order, so the result is identical to ascending serial
     * traversal while avoiding the former quadratic shared-memory scan.
     */
    struct RuntimeSmallGroupScanScratch
    {
        int count_warp_totals[(kDeviceMoEMaxExperts / 32)];
        int count_warp_prefixes[(kDeviceMoEMaxExperts / 32)];
        int active_warp_totals[(kDeviceMoEMaxExperts / 32)];
        int active_warp_prefixes[(kDeviceMoEMaxExperts / 32)];
        int total_active;
        int first_invalid_expert;
    };

    /**
     * @brief Compute deterministic block-wide count and active-list prefixes.
     *
     * Every lane must participate.  Each warp first scans its own values using
     * shuffle instructions. Warp zero then scans the block's warp totals in
     * ascending warp order. Integer addition is exact, and therefore this
     * fixed two-level tree publishes the same offsets and active ranks as the
     * serial expert-id traversal for every valid expert domain.
     *
     * @tparam ScanActive Whether to scan the active-expert predicate as well.
     * @param count Number of local routes assigned to this lane's expert.
     * @param scratch Block-owned scan state.
     * @param count_prefix Receives the exclusive route-count prefix.
     * @param active_prefix Receives the exclusive active-expert prefix.
     * @param total_active Receives the total number of active experts.
     */
    template <bool ScanActive>
    __device__ __forceinline__ void scan_runtime_small_group_experts(
        int count,
        RuntimeSmallGroupScanScratch &scratch,
        int &count_prefix,
        int &active_prefix,
        int &total_active)
    {
        constexpr unsigned int kFullWarpMask = 0xffffffffu;
        constexpr int kWarpSize = 32;
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        int count_inclusive = count;
        int active_inclusive = count > 0 ? 1 : 0;

#pragma unroll
        for (int delta = 1; delta < kWarpSize; delta <<= 1)
        {
            const int prior_count =
                __shfl_up_sync(kFullWarpMask, count_inclusive, delta);
            if (lane >= delta)
                count_inclusive += prior_count;
            if constexpr (ScanActive)
            {
                const int prior_active =
                    __shfl_up_sync(kFullWarpMask, active_inclusive, delta);
                if (lane >= delta)
                    active_inclusive += prior_active;
            }
        }

        if (lane == kWarpSize - 1)
        {
            scratch.count_warp_totals[warp] = count_inclusive;
            if constexpr (ScanActive)
                scratch.active_warp_totals[warp] = active_inclusive;
        }
        __syncthreads();

        if (warp == 0)
        {
            const int warp_count =
                lane < static_cast<int>(blockDim.x) / kWarpSize
                    ? scratch.count_warp_totals[lane]
                    : 0;
            const int warp_active =
                ScanActive && lane < static_cast<int>(blockDim.x) / kWarpSize
                    ? scratch.active_warp_totals[lane]
                    : 0;
            int count_warp_inclusive = warp_count;
            int active_warp_inclusive = warp_active;
#pragma unroll
            for (int delta = 1; delta < kWarpSize; delta <<= 1)
            {
                const int prior_count = __shfl_up_sync(
                    kFullWarpMask,
                    count_warp_inclusive,
                    delta);
                if (lane >= delta)
                    count_warp_inclusive += prior_count;
                if constexpr (ScanActive)
                {
                    const int prior_active = __shfl_up_sync(
                        kFullWarpMask,
                        active_warp_inclusive,
                        delta);
                    if (lane >= delta)
                        active_warp_inclusive += prior_active;
                }
            }

            if (lane < static_cast<int>(blockDim.x) / kWarpSize)
            {
                scratch.count_warp_prefixes[lane] =
                    count_warp_inclusive - warp_count;
                if constexpr (ScanActive)
                {
                    scratch.active_warp_prefixes[lane] =
                        active_warp_inclusive - warp_active;
                }
            }
            if constexpr (ScanActive)
            {
                if (lane == static_cast<int>(blockDim.x) / kWarpSize - 1)
                    scratch.total_active = active_warp_inclusive;
            }
        }
        __syncthreads();

        count_prefix = scratch.count_warp_prefixes[warp] +
                       count_inclusive - count;
        if constexpr (ScanActive)
        {
            const int active = count > 0 ? 1 : 0;
            active_prefix = scratch.active_warp_prefixes[warp] +
                            active_inclusive - active;
            total_active = scratch.total_active;
        }
        else
        {
            active_prefix = 0;
            total_active = 0;
        }
    }

    /** @brief Exact integer-prefix implementations selected by admitted geometry. */
    enum class SmallFloatGroupPrefixPolicy
    {
        IndependentExpert, ///< Short domains use independent ascending-expert prefixes.
        CooperativeScan, ///< Larger domains share the existing two-level integer scan.
    };

    /** @brief Maximum expert extent admitted to the independent-prefix specialization. */
    constexpr int kSmallFloatIndependentPrefixMaxExperts = 8;

    /**
     * @brief Select the prefix implementation before recording a complete operation.
     * @param num_experts Positive admitted expert extent, at most kDeviceMoEMaxExperts.
     * @return One fixed prefix policy for this immutable captured geometry.
     */
    constexpr SmallFloatGroupPrefixPolicy small_float_group_prefix_policy(int num_experts)
    {
        // Independent prefixes avoid cooperative-scan overhead for short domains;
        // beyond this extent the parallel scan bounds shared integer prefix work.
        return num_experts <= kSmallFloatIndependentPrefixMaxExperts
            ? SmallFloatGroupPrefixPolicy::IndependentExpert
            : SmallFloatGroupPrefixPolicy::CooperativeScan;
    }

    /**
     * @brief Specialize shared storage so mutually exclusive prefixes own no unused bytes.
     * @tparam PrefixPolicy One complete integer-prefix implementation.
     */
    template <SmallFloatGroupPrefixPolicy PrefixPolicy>
    struct SmallFloatGroupPrefixScratch;

    /** @brief Shared count publications read by independent ascending-expert prefixes. */
    template <>
    struct SmallFloatGroupPrefixScratch<SmallFloatGroupPrefixPolicy::IndependentExpert>
    {
        int counts[kSmallFloatIndependentPrefixMaxExperts]; ///< Only admitted expert lanes publish before readers.
    };

    /** @brief Exact existing scan storage with no independent-prefix count array. */
    template <>
    struct SmallFloatGroupPrefixScratch<SmallFloatGroupPrefixPolicy::CooperativeScan>
    {
        RuntimeSmallGroupScanScratch scan; ///< One block owns the existing two-level scan state.
    };

    /**
     * @brief Publish one compact stable expert-major grouping operation.
     *
     * One complete block owns every publication. Valid IDs determine route counts,
     * regardless of zero weight, and each route keeps its serial original-slot
     * rank. Prefix implementations differ only in integer scheduling and internal
     * shared storage; counts, offsets, maps, sentinels and copied weights have one
     * common initialization, publication and stable scatter body.
     *
     * @tparam PrefixPolicy Compile-time prefix implementation selected before capture.
     * @param routing_indices Original finite expert IDs.
     * @param routing_weights Original FP32 weights copied without arithmetic.
     * @param expert_counts Complete per-expert route-count publication.
     * @param expert_offsets Complete per-expert exclusive-offset publication.
     * @param grouped_token_indices Stable expert-major source token rows.
     * @param original_to_grouped Original-slot to grouped-slot inverse map.
     * @param original_expert_ids Original-slot valid expert IDs or minus one.
     * @param grouped_weights Stable expert-major copied FP32 route weights.
     * @param active_expert_ids Ascending active IDs with minus-one sentinels.
     * @param total_slots Positive admitted original route slots, at most 64.
     * @param num_experts Positive admitted expert extent, at most 512.
     * @param top_k Positive admitted route width dividing total_slots.
     * @param max_active_experts Declared active-list extent, at most min(total_slots, num_experts).
     */
    template <SmallFloatGroupPrefixPolicy PrefixPolicy>
    __global__ void group_tokens_small_float_kernel(
        const float *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ expert_counts,
        int *__restrict__ expert_offsets,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int *__restrict__ active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts)
    {
        __shared__ int shared_route_experts[kDeviceMoESmallGroupMaxSlots];
        __shared__ SmallFloatGroupPrefixScratch<PrefixPolicy> prefix_scratch;

        const int tid = threadIdx.x;

        if (tid < total_slots)
        {
            grouped_token_indices[tid] = 0;
            original_to_grouped[tid] = -1;
            original_expert_ids[tid] = -1;
            grouped_weights[tid] = 0.0f;

            const int expert = static_cast<int>(routing_indices[tid]);
            const int valid_expert =
                (expert >= 0 && expert < num_experts) ? expert : -1;
            shared_route_experts[tid] = valid_expert;
            original_expert_ids[tid] = valid_expert;
        }
        if (tid < max_active_experts)
            active_expert_ids[tid] = -1;
        __syncthreads();

        // Counts follow valid IDs, including routes with positive or negative
        // zero weights. Lanes outside the expert domain contribute zero.
        int count = 0;
        if (tid < num_experts)
            for (int slot = 0; slot < total_slots; ++slot)
                count += shared_route_experts[slot] == tid ? 1 : 0;

        int offset = 0;
        int active_rank = 0;
        if constexpr (PrefixPolicy == SmallFloatGroupPrefixPolicy::IndependentExpert)
        {
            // The selected domain fits the independent-prefix scratch. Only those
            // expert lanes write; every block lane still enters the barrier. This
            // short domain saves one cooperative-scan barrier and shuffles.
            if (tid < num_experts)
                prefix_scratch.counts[tid] = count;
            __syncthreads();
            if (tid < num_experts)
            {
                for (int expert = 0; expert < tid; ++expert)
                {
                    const int preceding_count = prefix_scratch.counts[expert];
                    offset += preceding_count;
                    active_rank += preceding_count > 0 ? 1 : 0;
                }
            }
        }
        else
        {
            // Every block lane enters both scan barriers. Bounded integer sums are
            // exact and preserve ascending expert order without quadratic reads.
            int total_active = 0;
            scan_runtime_small_group_experts<true>(
                count, prefix_scratch.scan, offset, active_rank, total_active);
        }
        if (tid < num_experts)
        {
            expert_counts[tid] = count;
            expert_offsets[tid] = offset;
            if (count > 0 && active_rank < max_active_experts)
                active_expert_ids[active_rank] = tid;
        }
        // Both prefix policies publish offsets before route workers consume them.
        __syncthreads();

        if (tid < total_slots)
        {
            const int expert = shared_route_experts[tid];
            if (expert < 0 || expert >= num_experts)
                return;

            int stable_local_rank = 0;
            for (int previous_slot = 0; previous_slot < tid; ++previous_slot)
                stable_local_rank +=
                    shared_route_experts[previous_slot] == expert ? 1 : 0;

            const int destination =
                expert_offsets[expert] + stable_local_rank;
            grouped_token_indices[destination] = tid / top_k;
            original_to_grouped[tid] = destination;
            grouped_weights[destination] = routing_weights[tid];
        }
    }

    /**
     * @brief Publish shared-expert grouping for only the controller's live rows.
     *
     * Initialize inactive inverse/descriptor entries on every replay so a
     * retained wider graph cannot compute stale rows after depth demotion.
     */
    __global__ void prepare_shared_expert_group_kernel(
        int *__restrict__ expert_offsets,
        int *__restrict__ expert_counts,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int *__restrict__ active_expert_ids,
        llaminar2::DeviceRowRange rows)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int active_rows = rows.activeRows();
        if (idx == 0)
        {
            expert_offsets[0] = 0;
            expert_counts[0] = active_rows;
            active_expert_ids[0] = active_rows > 0 ? 0 : -1;
        }
        if (idx < rows.physicalRows())
        {
            grouped_token_indices[idx] = idx;
            original_to_grouped[idx] = idx < active_rows ? idx : -1;
            original_expert_ids[idx] = idx < active_rows ? 0 : -1;
            grouped_weights[idx] = idx < active_rows ? 1.0f : 0.0f;
        }
    }

    __global__ void prefill_group_clear_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int *__restrict__ original_to_grouped,
        int max_slots,
        int num_experts,
        int clear_route_metadata)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;

        if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
        {
            runtime->expert_counts[idx] = 0;
            runtime->expert_offsets[idx] = 0;
        }

        if (idx < max_slots)
        {
            if (clear_route_metadata && runtime->route_expert_ids)
                runtime->route_expert_ids[idx] = -1;
            if (clear_route_metadata && runtime->route_weights)
                runtime->route_weights[idx] = 0.0f;
            if (clear_route_metadata && runtime->route_participant_ids)
                runtime->route_participant_ids[idx] = -1;
            if (runtime->grouped_token_ids)
                runtime->grouped_token_ids[idx] = 0;
            if (runtime->grouped_route_weights)
                runtime->grouped_route_weights[idx] = 0.0f;
            if (original_to_grouped)
                original_to_grouped[idx] = -1;
        }
    }

    /**
     * @brief Check whether StaticOwner runtime prefill may compute an expert locally.
     *
     * LocalTP verifier rows are global-route, local-compute: each shard sees
     * the same top-k expert ids, but only experts whose active placement-bank
     * descriptor is resident and LocalCompute-ready should enter this shard's
     * grouped GEMM.  Dynamic/LLEP leaves this filter disabled for the first
     * grouping pass so the planner can count every routed row before assigning
     * participants.
     */
    __device__ __forceinline__ bool prefill_static_local_runtime_ready(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        int expert_id,
        int num_experts,
        llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (!runtime ||
            expert_id < 0 ||
            expert_id >= num_experts ||
            expert_id >= kDeviceMoEMaxExperts ||
            execution_bank > 1u)
        {
            return false;
        }

        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert_id];
        if (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)
        return runtime_intermediate_shard_row(runtime, bank, expert_id).disposition ==
            llaminar2::MoEIntermediateShardRowDisposition::Local;
    const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(runtime->participant_id));
        return bank.local_compute_mask[expert_id] != 0u &&
               (bank.resident_participant_mask[expert_id] & local_bit) != 0u &&
               desc.local_slot >= 0 &&
               llaminar2::deviceMoEExpertProjectionReady(desc, expected_projections);
    }

    /**
     * @brief Publish a complete verifier-sized runtime grouping transaction.
     *
     * One block owns every route slot and every expert. The initial-grouping
     * specialization first converts FP32 router ids into the runtime route
     * ledger; the regrouping specialization consumes participant assignments
     * already written by the LLEP planner. Both specializations then derive
     * exact integer counts, ascending-expert offsets, stable grouped rows, and
     * the inverse original-to-grouped map in one launch.
     *
     * Stable route order is part of MTP correctness. A route thread computes its
     * destination by counting earlier local routes to the same expert, exactly
     * matching serial traversal. Every output has one writer; atomics only
     * count bounded integers or validate descriptor readiness. No floating-point
     * reduction or stable route ordering depends on their execution order.
     *
     * @tparam PublishRouterInputs Whether this launch owns initial router-output
     *         conversion (`true`) or post-LLEP regrouping (`false`).
     * @tparam PublishCompletePlan Publish the compact descriptors with the grouped routes.
     * @param gate_contract Fixed Gate/Up validation contract or the general validation mode.
     */
    template <bool PublishRouterInputs, bool PublishCompletePlan = false>
    __global__ void prefill_group_small_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        const float *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int filter_to_local_runtime_experts,
        int grouped_demand,
        DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ down_descs,
        int *__restrict__ active_expert_ids,
        int max_active_experts,
        llaminar2::DeviceMoEWeightFormat expected_format,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract)
    {
        // Complete-plan demand is distinct from route retention. Decode,
        // static and sidecar grouping must not count ordinary prefill.
        const int retain_routes_for_deferred_commit =
            grouped_demand == static_cast<int>(llaminar2::MoEGroupedPlanDemand::DeferredAcceptedRows);
        __shared__ int shared_route_experts[kDeviceMoERuntimeSmallGroupMaxSlots];
        __shared__ int shared_route_participants[kDeviceMoERuntimeSmallGroupMaxSlots];
        __shared__ float shared_route_weights[kDeviceMoERuntimeSmallGroupMaxSlots];
        __shared__ int shared_expert_offsets[kDeviceMoEMaxExperts];
        __shared__ RuntimeSmallGroupScanScratch scan_scratch;

        const int tid = static_cast<int>(threadIdx.x);
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        const bool invalid_contract =
            !runtime || !original_to_grouped ||
            blockDim.x != llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts) ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            max_slots > kDeviceMoERuntimeSmallGroupMaxSlots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK ||
            (PublishRouterInputs && (!routing_indices || !routing_weights)) ||
            execution_bank > 1u ||
            !llaminar2::deviceMoEProjectionSetValid(expected_projections) ||
            (runtime && runtime->intermediate_shard.enabled() !=
                (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)) ||
            runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots) ||
            !runtime->route_expert_ids || !runtime->route_weights ||
            !runtime->route_participant_ids || !runtime->expert_counts ||
            !runtime->expert_offsets || !runtime->grouped_token_ids ||
            !runtime->grouped_route_weights ||
            (PublishCompletePlan &&
             (!gate_descs || !up_descs || !down_descs ||
              !active_expert_ids || max_active_experts <= 0 ||
              max_active_experts > num_experts ||
              (expected_format !=
                   llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
               !llaminar2::deviceMoEWeightFormatIsFloating(
                   expected_format)))) ||
            (retain_routes_for_deferred_commit != 0 &&
             (!runtime->deferred_verifier_route_expert_ids ||
              !runtime->deferred_verifier_route_participant_ids ||
              runtime->deferred_verifier_route_capacity <
                  static_cast<uint32_t>(current_slots)));
        if (invalid_contract)
        {
            if (tid == 0)
            {
                const auto *const ticket =
                    runtime ? runtime->overlay_epoch_ticket : nullptr;
                const auto *const durable_banks =
                    runtime_durable_placement_banks(runtime);
                printf("runtime_small_group_invalid_contract "
                       "publish_inputs=%d complete_plan=%d current_slots=%d max_slots=%d "
                       "num_experts=%d top_k=%d block=%u bank=%u "
                       "runtime=%p indices=%p weights=%p inverse=%p "
                       "route_capacity=%u route_ids=%p route_weights=%p route_participants=%p "
                       "expert_counts=%p expert_offsets=%p grouped_ids=%p grouped_weights=%p "
                       "retain_deferred=%d deferred_capacity=%u deferred_ids=%p deferred_participants=%p "
                       "ticket=%p ticket_epoch=%llu ticket_selector=%llu durable_banks=%p "
                       "bank0_epoch=%u bank1_epoch=%u active_bank=%u transient=%u overlay_banks=%p\n",
                       PublishRouterInputs ? 1 : 0,
                       PublishCompletePlan ? 1 : 0,
                       current_slots,
                       max_slots,
                       num_experts,
                       top_k,
                       static_cast<unsigned>(blockDim.x),
                       execution_bank,
                       runtime,
                       routing_indices,
                       routing_weights,
                       original_to_grouped,
                       runtime ? runtime->prefill_route_capacity : 0u,
                       runtime ? runtime->route_expert_ids : nullptr,
                       runtime ? runtime->route_weights : nullptr,
                       runtime ? runtime->route_participant_ids : nullptr,
                       runtime ? runtime->expert_counts : nullptr,
                       runtime ? runtime->expert_offsets : nullptr,
                       runtime ? runtime->grouped_token_ids : nullptr,
                       runtime ? runtime->grouped_route_weights : nullptr,
                       retain_routes_for_deferred_commit,
                       runtime ? runtime->deferred_verifier_route_capacity : 0u,
                       runtime ? runtime->deferred_verifier_route_expert_ids : nullptr,
                       runtime ? runtime->deferred_verifier_route_participant_ids : nullptr,
                       ticket,
                       static_cast<unsigned long long>(ticket ? ticket->epoch : 0u),
                       static_cast<unsigned long long>(ticket ? ticket->selector : 0u),
                       durable_banks,
                       durable_banks ? durable_banks[0].epoch : 0u,
                       durable_banks ? durable_banks[1].epoch : 0u,
                       runtime ? runtime->active_bank : 0u,
                       runtime ? runtime->current_batch_llep_transient_bank_active : 0u,
                       runtime ? runtime->overlay_placement_banks : nullptr);
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "verifier-sized runtime grouping has an invalid device contract");
            }
            return;
        }

        // The exclusive-offset bank first holds exact bounded integer counts.
        // One local route adds one, avoiding an experts-by-slots shared-memory
        // scan. At most 256 increments reach a counter, independent of order.
        const int local_participant = static_cast<int>(runtime->participant_id);
        if (tid < num_experts)
            shared_expert_offsets[tid] = 0;
        __syncthreads();

        if (tid < max_slots)
        {
            runtime->grouped_token_ids[tid] = 0;
            runtime->grouped_route_weights[tid] = 0.0f;
            original_to_grouped[tid] = -1;

            int expert_id = -1;
            float route_weight = 0.0f;
            int participant_id = -1;
            if (tid < current_slots)
            {
                if constexpr (PublishRouterInputs)
                {
                    expert_id = static_cast<int>(routing_indices[tid]);
                    route_weight = routing_weights[tid];
                    if (expert_id < 0 || expert_id >= num_experts)
                    {
                        expert_id = -1;
                        route_weight = 0.0f;
                    }
                    else if (filter_to_local_runtime_experts != 0 ||
                    expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)
                    {
                        /* StaticOwner still publishes one globally canonical
                         * participant ledger on every shard.  Local grouping
                         * is a filtered view of that ledger; encoding a remote
                         * owner as -1 made mapped sparse reducers mistake a
                         * valid peer route for a heterogeneous/no-op route. */
                        const DeviceMoEPlacementBankView &bank =
                            runtime_placement_banks(runtime)[execution_bank];
                        participant_id = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                            ? runtime_intermediate_shard_row(runtime, bank, expert_id).participant
                            : runtime_expert_owner(bank, expert_id);
                        if (participant_id !=
                                static_cast<int>(runtime->participant_id) ||
                            !prefill_static_local_runtime_ready(
                                runtime,
                                expert_id,
                                num_experts,
                                expected_projections))
                        {
                            route_weight = 0.0f;
                        }
                    }
                    else
                    {
                        participant_id = static_cast<int>(runtime->participant_id);
                    }

                    runtime->route_expert_ids[tid] = expert_id;
                    runtime->route_weights[tid] = route_weight;
                    runtime->route_participant_ids[tid] = participant_id;
                }
                else
                {
                    expert_id = runtime->route_expert_ids[tid];
                    route_weight = runtime->route_weights[tid];
                    participant_id = runtime->route_participant_ids[tid];
                }

                if (retain_routes_for_deferred_commit != 0)
                {
                    runtime->deferred_verifier_route_expert_ids[tid] = expert_id;
                    runtime->deferred_verifier_route_participant_ids[tid] =
                        participant_id;
                }
            }
            else if constexpr (PublishRouterInputs)
            {
                runtime->route_expert_ids[tid] = -1;
                runtime->route_weights[tid] = 0.0f;
                runtime->route_participant_ids[tid] = -1;
            }

            shared_route_experts[tid] = expert_id;
            shared_route_weights[tid] = route_weight;
            shared_route_participants[tid] = participant_id;
            if (expert_id >= 0 && expert_id < num_experts &&
                participant_id == local_participant)
                atomicAdd(&shared_expert_offsets[expert_id], 1);
        }
        __syncthreads();

        /*
         * Only a complete ordinary-prefill plan is a serial-visible demand
         * publication boundary.  Route-only grouping may be followed by LLEP
         * participant assignment, while grouped verification must wait for
         * the accepted-prefix commit.  Folding the counter update into this
         * already-required small-plan kernel keeps that distinction exact
         * without adding a graph node.
         */
        if constexpr (PublishCompletePlan)
        {
            if (grouped_demand == static_cast<int>(llaminar2::MoEGroupedPlanDemand::OrdinaryPrefill) &&
                tid < current_slots &&
                runtime_histogram_admits_rows(*runtime))
            {
                const int expert_id = shared_route_experts[tid];
                const int participant_id = shared_route_participants[tid];
                if (expert_id >= 0 && expert_id < num_experts)
                {
                    atomicAdd(
                        reinterpret_cast<unsigned long long *>(
                            &runtime_selected_histogram(*runtime, 1u)[expert_id]),
                        1ULL);
                    if (participant_id ==
                        static_cast<int>(runtime->participant_id))
                    {
                        atomicAdd(
                            reinterpret_cast<unsigned long long *>(
                                &runtime_local_histogram(*runtime, 1u)[expert_id]),
                            1ULL);
                    }
                }
            }
        }

        // The publication barrier above completes the integer histogram. The
        // scan consumes every count before recycling this bank into offsets.
        const int count = tid < num_experts ? shared_expert_offsets[tid] : 0;

        if constexpr (PublishCompletePlan)
        {
            if (tid == 0)
                scan_scratch.first_invalid_expert = num_experts;
        }
        int expert_offset = 0;
        int active_rank = 0;
        int total_active = 0;
        scan_runtime_small_group_experts<PublishCompletePlan>(
            count,
            scan_scratch,
            expert_offset,
            active_rank,
            total_active);

        if (tid < num_experts)
        {
            shared_expert_offsets[tid] = expert_offset;
            runtime->expert_counts[tid] = count;
            runtime->expert_offsets[tid] = expert_offset;

            if constexpr (PublishCompletePlan)
            {
                const DeviceMoEPlacementBankView &bank =
                    runtime_placement_banks(runtime)[execution_bank];
                const uint32_t local_bit =
                    expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                    ? 0u : runtime_participant_bit(local_participant);
                const bool is_active = count > 0;

                /*
                 * The compact plan makes only active experts reachable. Avoid
                 * rewriting three 56-byte descriptors for every inactive
                 * expert on each verifier step; active lanes still publish
                 * exact bytes from the selected immutable placement bank.
                 */
                bool local_ready = true;
                if (is_active)
                {
                    local_ready = publish_runtime_expert_descriptors_lane(
                        runtime,
                        bank,
                        local_bit,
                        tid,
                        num_experts,
                        gate_descs,
                        up_descs,
                        down_descs,
                        expected_format,
                        expected_projections, gate_contract);
                }
                if (is_active)
                {
                    if (active_rank < max_active_experts)
                        active_expert_ids[active_rank] = tid;
                    if (!local_ready)
                    {
                        atomicMin(
                            &scan_scratch.first_invalid_expert,
                            tid);
                    }
                }
            }
        }

        if constexpr (PublishCompletePlan)
        {
            const int retained_active =
                total_active < max_active_experts
                    ? total_active
                    : max_active_experts;
            for (int slot = retained_active + tid;
                 slot < max_active_experts;
                 slot += static_cast<int>(blockDim.x))
            {
                active_expert_ids[slot] = -1;
            }
        }
        __syncthreads();

        if constexpr (PublishCompletePlan)
        {
            if (scan_scratch.first_invalid_expert < num_experts)
            {
                if (tid == 0)
                {
                    const int invalid_expert =
                        scan_scratch.first_invalid_expert;
                    const DeviceMoEPlacementBankView &bank =
                        runtime_placement_banks(runtime)[execution_bank];
                    const DeviceMoEExpertDescriptorView &desc =
                        bank.experts[invalid_expert];
                    const uint32_t local_bit =
                        expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                    ? 0u : runtime_participant_bit(local_participant);
                    const DeviceMoEPlacementBankView *const durable_banks =
                        runtime_durable_placement_banks(runtime);
                    const llaminar2::DeviceMoEOverlayEpochTicket *const ticket =
                        runtime->overlay_epoch_ticket;
                    printf("runtime_prefill_active_expert_not_ready "
                           "participant=%u expert=%d count=%d execution_bank=%u "
                           "active_bank=%u active_epoch=%u bank0_epoch=%u "
                           "bank1_epoch=%u local_mask=%u resident_mask=%u "
                           "local_bit=%u local_slot=%d logical=%d owner=%d "
                           "descriptor_flags=%u route_participant=%d "
                           "external_banks=%u transient=%u ticket_epoch=%llu "
                           "ticket_selector=%llu\\n",
                           runtime->participant_id,
                           invalid_expert,
                           runtime->expert_counts[invalid_expert],
                           execution_bank,
                           runtime->active_bank,
                           runtime->active_epoch,
                           durable_banks ? durable_banks[0].epoch : 0u,
                           durable_banks ? durable_banks[1].epoch : 0u,
                           bank.local_compute_mask[invalid_expert],
                           bank.resident_participant_mask[invalid_expert],
                           local_bit,
                           desc.local_slot,
                           desc.logical_expert_id,
                           desc.owner_participant,
                           desc.flags,
                           bank.overlay_route_participant[invalid_expert],
                           runtime->overlay_placement_banks ? 1u : 0u,
                           runtime->current_batch_llep_transient_bank_active,
                           static_cast<unsigned long long>(
                               ticket ? ticket->epoch : 0u),
                           static_cast<unsigned long long>(
                               ticket ? ticket->selector : 0u));
                    FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                        "runtime active expert is not locally resident and compute-ready");
                }
                return;
            }
        }

        if (tid < current_slots)
        {
            const int expert_id = shared_route_experts[tid];
            if (expert_id >= 0 && expert_id < num_experts &&
                shared_route_participants[tid] == local_participant)
            {
                int stable_local_rank = 0;
                for (int previous_slot = 0; previous_slot < tid; ++previous_slot)
                {
                    stable_local_rank +=
                        shared_route_experts[previous_slot] == expert_id &&
                                shared_route_participants[previous_slot] == local_participant
                            ? 1
                            : 0;
                }
                const int destination =
                    shared_expert_offsets[expert_id] + stable_local_rank;
                runtime->grouped_token_ids[destination] = tid;
                runtime->grouped_route_weights[destination] =
                    shared_route_weights[tid];
                original_to_grouped[tid] = destination;
            }
        }
        __syncthreads();

        /*
         * The former multi-launch grouping path validated this inverse map in a
         * separate kernel. Fusion must retain that fatal contract: a duplicate
         * destination, bad offset, or stale grouped row would otherwise feed a
         * different activation to the ordered verifier GEMM. Every route lane
         * checks its own expected membership after all grouped rows are visible,
         * proving that the two maps form a bijection without another launch.
         */
        if (tid < current_slots)
        {
            const int expert_id = shared_route_experts[tid];
            const int participant_id = shared_route_participants[tid];
            const bool should_be_local =
                expert_id >= 0 && expert_id < num_experts &&
                participant_id == local_participant;
            const int grouped_slot = original_to_grouped[tid];
            const bool valid_mapping =
                should_be_local
                    ? grouped_slot >= 0 && grouped_slot < max_slots &&
                          runtime->grouped_token_ids[grouped_slot] == tid &&
                          runtime->route_expert_ids[tid] == expert_id &&
                          runtime->route_participant_ids[tid] == local_participant
                    : grouped_slot == -1;
            if (!valid_mapping)
            {
                printf("runtime_small_group_invalid_row "
                       "participant=%d route_slot=%d grouped_slot=%d "
                       "expert=%d route_participant=%d grouped_route_slot=%d\n",
                       local_participant,
                       tid,
                       grouped_slot,
                       expert_id,
                       participant_id,
                       grouped_slot >= 0 && grouped_slot < max_slots
                           ? runtime->grouped_token_ids[grouped_slot]
                           : -1);
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "runtime grouped route row does not match its original route slot");
            }
        }

    }

    __global__ void prefill_group_cast_count_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        const float *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int current_slots,
        int max_slots,
        int num_experts,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || slot >= max_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->route_weights || !runtime->route_participant_ids)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        if (runtime->intermediate_shard.enabled() !=
            (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard))
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("scalable grouping has the wrong original fragment projection arm");
            return;
        }
        int expert_id = -1;
        float weight = 0.0f;
        int participant_id = -1;
        if (slot < current_slots)
        {
            expert_id = static_cast<int>(routing_indices[slot]);
            weight = routing_weights[slot];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            expert_id = -1;
            weight = 0.0f;
        }
        else if (filter_to_local_runtime_experts != 0 ||
                    expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)
        {
            /*
             * Preserve the authoritative StaticOwner participant even on a
             * non-owner shard. Local route weights remain the execution
             * filter; the participant ledger is shared by node-local sparse
             * exchange, accepted-history publication, and later controllers.
             */
            const uint32_t execution_bank = runtime_execution_bank(runtime);
            const DeviceMoEPlacementBankView &bank =
                runtime_placement_banks(runtime)[execution_bank];
            participant_id = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                            ? runtime_intermediate_shard_row(runtime, bank, expert_id).participant
                            : runtime_expert_owner(bank, expert_id);
            if (participant_id !=
                    static_cast<int>(runtime->participant_id) ||
                !prefill_static_local_runtime_ready(
                    runtime,
                    expert_id,
                    num_experts,
                    expected_projections))
            {
                weight = 0.0f;
            }
        }
        else
        {
            participant_id = static_cast<int>(runtime->participant_id);
            }
        }

        runtime->route_expert_ids[slot] = expert_id;
        runtime->route_weights[slot] = weight;
        runtime->route_participant_ids[slot] = participant_id;
        if (retain_routes_for_deferred_commit != 0 &&
            slot < current_slots &&
            runtime->deferred_verifier_route_expert_ids &&
            runtime->deferred_verifier_route_participant_ids &&
            runtime->deferred_verifier_route_capacity >=
                static_cast<uint32_t>(current_slots))
        {
            runtime->deferred_verifier_route_expert_ids[slot] = expert_id;
            runtime->deferred_verifier_route_participant_ids[slot] =
                participant_id;
        }
        /*
         * The scalable regime folds counting into route publication. Counts are
         * integers, so atomic arrival order cannot alter the result or the later
         * stable route ordering. The preceding clear launch is the graph edge
         * that makes every counter available before this publication begins.
         */
        if (expert_id >= 0 &&
            expert_id < num_experts &&
            participant_id == static_cast<int>(runtime->participant_id))
        {
            atomicAdd(runtime->expert_counts + expert_id, 1);
        }
    }

    __global__ void prefill_group_count_assigned_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int retain_routes_for_deferred_commit)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime || slot >= max_slots || slot >= current_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->route_participant_ids || !runtime->expert_counts)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        const int expert_id = runtime->route_expert_ids[slot];
        const int participant_id = runtime->route_participant_ids[slot];
        if (retain_routes_for_deferred_commit != 0 &&
            runtime->deferred_verifier_route_expert_ids &&
            runtime->deferred_verifier_route_participant_ids &&
            runtime->deferred_verifier_route_capacity >=
                static_cast<uint32_t>(current_slots))
        {
            runtime->deferred_verifier_route_expert_ids[slot] = expert_id;
            runtime->deferred_verifier_route_participant_ids[slot] =
                participant_id;
        }
        if (expert_id >= 0 &&
            expert_id < num_experts &&
            participant_id == static_cast<int>(runtime->participant_id))
        {
            atomicAdd(runtime->expert_counts + expert_id, 1);
        }
    }

    /**
     * @brief Publish only serial-visible grouped-verifier routing evidence.
     *
     * Route ids and participant assignments were produced by the verifier
     * graph for every padded physical row. Acceptance metadata is produced
     * later by the stochastic verifier reducer. One thread examines one route
     * slot and commits it only when its request-local token row falls inside
     * that request's accepted state prefix.
     *
     * The selected and local counters are independent because every
     * participant observes global route selection, while only the participant
     * that actually computed the route owns local demand. Unsigned integer
     * atomics make the result independent of thread execution order.
     */
    __global__ void commit_grouped_verifier_histograms_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int32_t *__restrict__ accepted_state_counts,
        const int32_t *__restrict__ publication_ok_flags,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_slots = total_rows * top_k;
        const bool ledger_is_complete =
            runtime != nullptr &&
            accepted_state_counts != nullptr &&
            publication_ok_flags != nullptr &&
            runtime->deferred_verifier_route_expert_ids != nullptr &&
            runtime->deferred_verifier_route_participant_ids != nullptr &&
            runtime->deferred_verifier_route_capacity >=
                static_cast<uint32_t>(total_slots);
        if (!ledger_is_complete)
        {
            if (slot == 0)
            {
                FAIL_FAST_INVALID_GROUPED_VERIFIER_COMMIT(
                    "accepted grouped-verifier history requires a complete "
                    "per-layer deferred route ledger");
            }
            return;
        }
        if (slot >= total_slots)
            return;

        const int token_row = slot / top_k;
        const int request = token_row / rows_per_request;
        const int request_row = token_row - request * rows_per_request;
        if (request < 0 ||
            request >= request_count ||
            publication_ok_flags[request] == 0)
        {
            return;
        }

        const int accepted_rows = accepted_state_counts[request];
        if (accepted_rows < 0 ||
            accepted_rows > rows_per_request ||
            request_row >= accepted_rows)
        {
            return;
        }

        const int expert_id =
            runtime->deferred_verifier_route_expert_ids[slot];
        if (expert_id < 0 || expert_id >= num_experts)
            return;
        if (!runtime_histogram_admits_rows(*runtime))
            return;

        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &runtime_selected_histogram(*runtime, 2u)[expert_id]),
            1ULL);
        if (runtime->deferred_verifier_route_participant_ids[slot] ==
            static_cast<int>(runtime->participant_id))
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &runtime_local_histogram(*runtime, 2u)[expert_id]),
                1ULL);
        }
    }

    /**
     * @brief Publish expert offsets and scatter routes in one deterministic launch.
     *
     * Block @c expert computes the exact ascending-expert prefix that the old
     * one-thread scan published for that expert, stores the offset, and then
     * scatters only that expert's routes. Prefixes are integer sums, so the
     * independently computed values are byte-identical to the former serial
     * scan and have no floating-point ordering concern. Keeping publication
     * and consumption in one block makes the offset dependency explicit and
     * removes one graph node from every MoE layer.
     * @param runtime Authoritative device route/count buffers, already published.
     * @param original_to_grouped Required inverse map indexed by original route slot.
     * @param current_slots Slots visible in this grouping transaction.
     * @param max_slots Admitted capacity; padded outputs are owned by prior clear.
     * @param num_experts Number of independent expert blocks.
     * @param top_k Router fanout retained by the common launch ABI.
     */
    __global__ void prefill_group_scan_scatter_deterministic_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int *__restrict__ original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k)
    {
        const int expert = blockIdx.x;
        if (!runtime || expert >= num_experts)
            return;
        if (!original_to_grouped ||
            !runtime->route_expert_ids || !runtime->route_weights ||
            !runtime->route_participant_ids ||
            !runtime->expert_offsets || !runtime->expert_counts ||
            !runtime->grouped_token_ids || !runtime->grouped_route_weights)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        __shared__ llaminar2::device_integer_scan::Scratch<kThreads> scan;
        int preceding_count = 0;
        for (int preceding_expert = static_cast<int>(threadIdx.x);
             preceding_expert < expert; preceding_expert += kThreads)
            preceding_count += runtime->expert_counts[preceding_expert];
        const int prefix = llaminar2::device_integer_scan::exclusive(preceding_count, scan).total;
        const int count = runtime->expert_counts[expert];
        if (prefix < 0 || count < 0 || prefix + count > max_slots)
        {
            if (threadIdx.x == 0)
            {
                printf("runtime_group_scan_scatter_invalid_range "
                       "participant=%u expert=%d prefix=%d count=%d max_slots=%d\n",
                       runtime->participant_id,
                       expert,
                       prefix,
                       count,
                       max_slots);
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "runtime grouped route prefix exceeds the published capacity");
            }
            return;
        }
        if (threadIdx.x == 0)
            runtime->expert_offsets[expert] = prefix;
        // Empty experts still publish their prefix, but do not rescan every
        // route. This return is block-uniform after the cooperative prefix.
        if (count == 0)
            return;
        const int expert_end = prefix + count;
        const int participant_id = static_cast<int>(runtime->participant_id);
        int running_count = 0;

        // Slots beyond current_slots cannot be retained. The preceding clear
        // owns padded outputs; grouping need not read that unused capacity.
        for (int chunk_start = 0; chunk_start < current_slots; chunk_start += kThreads)
        {
            const int slot = chunk_start + threadIdx.x;
            const int keep =
                slot < current_slots &&
                runtime->route_expert_ids[slot] == expert &&
                runtime->route_participant_ids[slot] == participant_id;

            const auto local = llaminar2::device_integer_scan::exclusive(keep, scan);

            if (keep)
            {
                const int dest = prefix + running_count + local.exclusive;
                if (dest < expert_end)
                {
                    runtime->grouped_token_ids[dest] = slot;
                    runtime->grouped_route_weights[dest] = runtime->route_weights[slot];
                    original_to_grouped[slot] = dest;
                }
            }
            // Every lane carries the same chunk total. Prefixes keep the exact
            // ascending route order without a lane-zero flag scan.
            running_count += local.total;
        }
    }

    __global__ void prefill_llep_count_expert_routes_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts)
    {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;
        if (slot >= current_slots || slot >= max_slots)
            return;
        if (!runtime->route_expert_ids || !runtime->expert_counts)
            return;
        if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        const int expert_id = runtime->route_expert_ids[slot];
        if (expert_id >= 0 && expert_id < num_experts)
            atomicAdd(runtime->expert_counts + expert_id, 1);
    }

    __global__ void prefill_llep_clear_current_batch_plan_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (!runtime)
            return;

        if (idx < num_experts && runtime->expert_counts)
            runtime->expert_counts[idx] = 0;

        int32_t *split_ends = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
        const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
        if (idx < split_items && split_ends)
            split_ends[idx] = 0;

        if (idx == 0)
        {
            runtime->reserved_u64[2] = 0ULL;
            runtime->reserved_u64[3] = 0ULL;
        }
    }

    /**
     * @brief Assign grouped resident routes with serial batch invariance.
     *
     * One CUDA lane owns one token row and keeps that row's participant loads
     * in registers. Route slots are consumed in their original router order.
     * Equal-load ties use the row's absolute logical position and route
     * identity, exactly like serial decode. Persistent demand histograms remain
     * maintenance evidence only, so rejected speculative rows cannot perturb
     * a future committed row's floating-point partition.
     *
     * The launch is allocation-free, atomic-free, transfer-free, and
     * graph-capturable. Every row is independent, so the GPU executes all rows
     * concurrently without imposing an artificial M limit.
     */
    __global__ void prefill_llep_assign_resident_rows_logical_position_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int32_t *__restrict__ absolute_position_ids,
        const int32_t *__restrict__ active_row_count,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k)
    {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        const int physical_rows = current_slots / top_k;

        if (!runtime)
        {
            if (row == 0)
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier assignment requires a runtime");
            return;
        }

        if (row == 0)
        {
            runtime->reserved_u64[2] = 0ULL;
            runtime->reserved_u64[3] = 0ULL;
        }

        const uint32_t execution_bank = runtime_execution_bank(runtime);
        const bool runtime_valid =
            runtime->route_expert_ids &&
            runtime->route_participant_ids &&
            absolute_position_ids &&
            active_row_count &&
            execution_bank <= 1u &&
            runtime->participant_count > 0u &&
            runtime->participant_count <= kDeviceMoEMaxParticipants &&
            current_slots >= 0 &&
            max_slots > 0 &&
            current_slots <= max_slots &&
            runtime->prefill_route_capacity >= static_cast<uint32_t>(max_slots);
        if (!runtime_valid)
        {
            if (row == 0)
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier assignment runtime is invalid");
            return;
        }
        const int logical_rows = *active_row_count;
        if (logical_rows < 0 || logical_rows > physical_rows)
        {
            if (row == 0)
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier active-row count exceeds physical width");
            return;
        }
        if (row >= logical_rows)
            return;
        const int32_t logical_position = absolute_position_ids[row];
        if (logical_position < 0)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "resident verifier logical position is negative");
        }

        const uint32_t participant_count = runtime->participant_count;
        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const int row_slot_base = row * top_k;
        int participant_load[kDeviceMoEMaxParticipants] = {};

        /*
         * Serial decode's first pass publishes every non-replicated route to
         * its owner and charges that owner before any replicated route is
         * selected. Preserve that two-pass ordering even when the single-owner
         * route appears later in top-k order.
         */
        for (int route = 0; route < top_k; ++route)
        {
            const int selected_slot = row_slot_base + route;
            const int selected_expert =
                runtime->route_expert_ids[selected_slot];
            if (selected_expert < 0 || selected_expert >= num_experts)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier route expert is out of range");
            }
            for (int prior_route = 0; prior_route < route; ++prior_route)
            {
                if (runtime->route_expert_ids[
                        row_slot_base + prior_route] == selected_expert)
                {
                    FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                        "resident verifier row contains duplicate experts");
                }
            }

            const int owner =
                bank.experts[selected_expert].owner_participant;
            if (owner < 0 ||
                static_cast<uint32_t>(owner) >= participant_count)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier route owner is invalid");
            }
            const uint32_t resident_mask =
                runtime_expert_resident_mask(
                    runtime,
                    bank,
                    selected_expert);
            if ((resident_mask &
                 runtime_participant_bit(owner)) == 0u)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier owner payload is not resident");
            }
            if (runtime_resident_count(
                    resident_mask,
                    participant_count) == 1)
            {
                runtime->route_participant_ids[selected_slot] = owner;
                ++participant_load[owner];
            }
        }

        for (int route = 0; route < top_k; ++route)
        {
            const int selected_slot = row_slot_base + route;
            const int selected_expert =
                runtime->route_expert_ids[selected_slot];
            const uint32_t runtime_resident_mask =
                runtime_expert_resident_mask(
                    runtime,
                    bank,
                    selected_expert);
            if (runtime_resident_count(
                    runtime_resident_mask,
                    participant_count) == 1)
            {
                continue;
            }

            const auto &desc = bank.experts[selected_expert];
            const int owner = desc.owner_participant;
            const uint32_t default_participant =
                (owner >= 0 &&
                 static_cast<uint32_t>(owner) < participant_count)
                    ? static_cast<uint32_t>(owner)
                    : (runtime->participant_id < participant_count
                           ? runtime->participant_id
                           : 0u);
            const uint32_t resident_mask =
                llaminar2::least_loaded_ep::normalizeResidentParticipantMask(
                    runtime_resident_mask,
                    owner,
                    runtime->participant_id,
                    participant_count);
            const uint32_t assigned_participant =
                llaminar2::least_loaded_ep::
                    selectBatchInvariantResidentParticipant(
                    resident_mask,
                    participant_load,
                    participant_count,
                    default_participant,
                    llaminar2::least_loaded_ep::
                        residentAssignmentTieTurn(
                            logical_position,
                            selected_expert,
                            route));
            if (assigned_participant >= participant_count)
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "resident verifier selected an invalid participant");

            ++participant_load[assigned_participant];
            runtime->route_participant_ids[selected_slot] =
                static_cast<int32_t>(assigned_participant);
        }
    }

    /**
     * @brief Return whether one expert precedes another in canonical LLEP order.
     *
     * The shared planner requires descending routed-row count with ascending
     * logical expert id as its exact tie break.  Invalid padding lanes sort
     * after every real expert so non-power-of-two codebooks can use the same
     * deterministic sorting network.
     */
    __device__ __forceinline__ bool current_batch_llep_expert_precedes(
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
     * @brief Cooperatively construct the canonical expert order in shared memory.
     *
     * Each block thread owns one sorting-network slot.  The network width is
     * the smallest power of two covering the active codebook, which keeps the
     * four- and eight-expert cases cheap while retaining complete support up
     * to the 256-expert runtime ABI limit.  Integer comparisons preserve the
     * planner's byte-exact deterministic ordering.
     */
    __device__ __forceinline__ void sort_current_batch_llep_experts_parallel(
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
                            ? current_batch_llep_expert_precedes(
                                  second, first, expert_loads, expert_count)
                            : current_batch_llep_expert_precedes(
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

    __global__ void prefill_llep_plan_current_batch_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        uint32_t min_chunk_tokens,
        uint32_t alpha_numerator,
        uint32_t alpha_denominator,
        uint32_t lambda_numerator,
        uint32_t lambda_denominator,
        uint64_t min_spread_improvement,
        uint32_t min_spread_improvement_divisor,
        uint64_t min_spread_improvement_per_critical_path_slot,
        uint64_t min_foreign_rows_per_critical_path_slot,
        uint32_t max_weight_transfers,
        uint32_t max_non_owner_experts_per_participant,
        int enable_balanced_skip)
    {
        __shared__ uint64_t expert_loads[kDeviceMoEMaxExperts];
        __shared__ uint32_t owner_participants[kDeviceMoEMaxExperts];
        __shared__ uint32_t resident_participant_masks[kDeviceMoEMaxExperts];
        __shared__ uint32_t sorted_experts[kDeviceMoEMaxExperts];
        __shared__ uint64_t pending_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t assigned_load[kDeviceMoEMaxParticipants];

        if (!runtime)
            return;

        const uint32_t execution_bank = runtime_execution_bank(runtime);
        const int lane = threadIdx.x;
        if (lane == 0)
        {
            runtime->reserved_u64[2] = 0ULL;
            runtime->reserved_u64[3] = 0ULL;
        }

        if (execution_bank > 1u ||
            runtime->participant_count == 0u ||
            runtime->participant_count > kDeviceMoEMaxParticipants ||
            current_slots < 0 ||
            max_slots <= 0 ||
            current_slots > max_slots ||
            num_experts <= 0 ||
            num_experts > kDeviceMoEMaxExperts ||
            runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots) ||
            !runtime->expert_counts ||
            !runtime->reserved_ptrs[1] ||
            !runtime->reserved_ptrs[2])
        {
            return;
        }

        const uint32_t participant_count = runtime->participant_count;
        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];

        if (lane < kDeviceMoEMaxParticipants)
        {
            pending_load[lane] = 0ULL;
            assigned_load[lane] = 0ULL;
        }

        for (int expert = lane; expert < num_experts; expert += blockDim.x)
        {
            const int raw_count = runtime->expert_counts[expert];
            expert_loads[expert] = raw_count > 0 ? static_cast<uint64_t>(raw_count) : 0ULL;
            const int owner = bank.experts[expert].owner_participant;
            owner_participants[expert] =
                (owner >= 0 && static_cast<uint32_t>(owner) < participant_count)
                    ? static_cast<uint32_t>(owner)
                    : (runtime->participant_id < participant_count ? runtime->participant_id : 0u);
            uint32_t resident_mask =
                bank.resident_participant_mask[expert] &
                runtime_valid_participant_mask(participant_count);
            resident_participant_masks[expert] = resident_mask;
        }
        sort_current_batch_llep_experts_parallel(
            sorted_experts,
            expert_loads,
            static_cast<uint32_t>(num_experts),
            static_cast<uint32_t>(lane));

        if (lane != 0)
            return;

        auto *spans = static_cast<llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan *>(
            runtime->reserved_ptrs[1]);
        auto *transfers = static_cast<llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
            runtime->reserved_ptrs[2]);
        const uint64_t span_capacity_u64 = runtime->reserved_u64[0];
        const uint64_t transfer_capacity_u64 = runtime->reserved_u64[1];
        const uint32_t span_capacity =
            span_capacity_u64 > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(span_capacity_u64);
        const uint32_t transfer_capacity =
            transfer_capacity_u64 > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(transfer_capacity_u64);

        llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig config{};
        config.expert_count = static_cast<uint32_t>(num_experts);
        config.participant_count = participant_count;
        config.min_chunk_tokens = min_chunk_tokens;
        config.alpha_numerator = alpha_numerator;
        config.alpha_denominator = alpha_denominator;
        config.lambda_numerator = lambda_numerator;
        config.lambda_denominator = lambda_denominator;
        config.min_spread_improvement = min_spread_improvement;
        config.min_spread_improvement_divisor = min_spread_improvement_divisor;
        config.min_spread_improvement_per_critical_path_slot =
            min_spread_improvement_per_critical_path_slot;
        config.min_foreign_rows_per_critical_path_slot =
            min_foreign_rows_per_critical_path_slot;
        config.max_weight_transfers = max_weight_transfers;
        config.max_non_owner_experts_per_participant =
            max_non_owner_experts_per_participant;
        config.enable_balanced_skip = enable_balanced_skip != 0;

        llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentWorkspace workspace{};
        workspace.sorted_experts = sorted_experts;
        workspace.pending_load = pending_load;
        workspace.assigned_load = assigned_load;

        llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus
            planner_status{};
        const bool ok = llaminar2::least_loaded_ep::planLeastLoadedExpertAssignment(
            expert_loads,
            owner_participants,
            config,
            workspace,
            spans,
            span_capacity,
            transfers,
            transfer_capacity,
            planner_status,
            resident_participant_masks,
            /*workspace_experts_are_sorted=*/true);
        if (ok && planner_status.overflow == 0u && planner_status.invalid_config == 0u)
        {
            runtime->reserved_u64[2] = planner_status.span_count;
            runtime->reserved_u64[3] = planner_status.weight_transfer_count;
            int32_t *span_bounds = static_cast<int32_t *>(runtime->reserved_ptrs[0]);
            if (span_bounds)
            {
                for (int expert = 0; expert < num_experts; ++expert)
                {
                    span_bounds[2 * expert] = -1;
                    span_bounds[2 * expert + 1] = -1;
                }
                for (uint32_t idx = 0; idx < planner_status.span_count; ++idx)
                {
                    const uint32_t expert = spans[idx].expert;
                    if (expert >= static_cast<uint32_t>(num_experts))
                        continue;
                    int32_t &begin = span_bounds[2 * static_cast<int>(expert)];
                    int32_t &end = span_bounds[2 * static_cast<int>(expert) + 1];
                    if (begin < 0)
                        begin = static_cast<int32_t>(idx);
                    end = static_cast<int32_t>(idx + 1u);
                }
            }
        }
    }

    __device__ __forceinline__ int prefill_llep_default_owner_participant(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        int expert)
    {
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (!runtime ||
            execution_bank > 1u ||
            runtime->participant_count == 0u ||
            runtime->participant_count > kDeviceMoEMaxParticipants ||
            expert < 0 ||
            expert >= kDeviceMoEMaxExperts)
        {
            return -1;
        }
        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const int owner = bank.experts[expert].owner_participant;
        return (owner >= 0 && static_cast<uint32_t>(owner) < runtime->participant_count)
                   ? owner
                   : static_cast<int>(
                         runtime->participant_id < runtime->participant_count
                             ? runtime->participant_id
                             : 0u);
    }

    __device__ __forceinline__ bool prefill_llep_local_expert_ready(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const DeviceMoEPlacementBankView &bank,
        int expert)
    {
        if (!runtime ||
            expert < 0 ||
            expert >= static_cast<int>(runtime->expert_count) ||
            expert >= kDeviceMoEMaxExperts)
        {
            return false;
        }
        const auto &desc = bank.experts[expert];
        const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(runtime->participant_id));
        return bank.local_compute_mask[expert] != 0u &&
               (bank.resident_participant_mask[expert] & local_bit) != 0u &&
               desc.local_slot >= 0 &&
               rebalance_expert_desc_ready(desc);
    }

    __device__ __forceinline__ bool prefill_llep_assignment_ready(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const DeviceMoEPlacementBankView &active_bank,
        int expert,
        int row,
        int assigned_participant)
    {
        if (!runtime ||
            expert < 0 ||
            expert >= static_cast<int>(runtime->expert_count) ||
            expert >= kDeviceMoEMaxExperts ||
            assigned_participant < 0 ||
            static_cast<uint32_t>(assigned_participant) >= runtime->participant_count)
        {
            printf("prefill_llep_invalid_assignment_participant "
                   "participant=%u expert=%d row=%d assigned=%d participant_count=%u\\n",
                   runtime ? runtime->participant_id : 0u,
                   expert,
                   row,
                   assigned_participant,
                   runtime ? runtime->participant_count : 0u);
            return false;
        }

        if (assigned_participant != static_cast<int32_t>(runtime->participant_id))
            return true;

        const uint32_t assigned_bit = runtime_participant_bit(assigned_participant);
        const int owner_participant =
            prefill_llep_default_owner_participant(runtime, expert);
        const uint32_t resident_mask =
            llaminar2::least_loaded_ep::normalizeResidentParticipantMask(
                active_bank.resident_participant_mask[expert],
                owner_participant,
                runtime->participant_id,
                runtime->participant_count);
        if ((resident_mask & assigned_bit) == 0u)
        {
            printf("prefill_llep_assignment_not_resident "
                   "participant=%u expert=%d row=%d assigned=%d assigned_bit=%u "
                   "resident_mask=%u raw_resident_mask=%u owner=%d local_ready=%u\\n",
                   runtime->participant_id,
                   expert,
                   row,
                   assigned_participant,
                   assigned_bit,
                   resident_mask,
                   active_bank.resident_participant_mask[expert],
                   owner_participant,
                   prefill_llep_local_expert_ready(runtime, active_bank, expert) ? 1u : 0u);
            return false;
        }

        if (assigned_participant == static_cast<int32_t>(runtime->participant_id) &&
            !prefill_llep_local_expert_ready(runtime, active_bank, expert))
        {
            printf("prefill_llep_assignment_local_not_ready "
                   "participant=%u expert=%d row=%d assigned=%d resident_mask=%u "
                   "raw_resident_mask=%u owner=%d\\n",
                   runtime->participant_id,
                   expert,
                   row,
                   assigned_participant,
                   resident_mask,
                   active_bank.resident_participant_mask[expert],
                   owner_participant);
            return false;
        }
        return true;
    }

    __global__ void prefill_llep_assign_routes_from_current_batch_spans_runtime_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int allow_pending_transfers,
        const DeviceMoERebalanceStatusView *__restrict__ transfer_status,
        const DeviceMoERebalanceApplyStatusView *__restrict__ apply_status)
    {
        const int expert = blockIdx.x;
        if (!runtime || expert >= num_experts)
            return;
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (execution_bank > 1u)
        {
            if (expert == 0 && threadIdx.x == 0)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "current-batch LLEP assignment has no valid execution epoch");
            }
            return;
        }
        if (!runtime->route_expert_ids || !runtime->route_participant_ids ||
            !runtime->expert_counts || !runtime->expert_offsets ||
            !runtime->grouped_token_ids || !runtime->reserved_ptrs[1])
        {
            return;
        }
        if (top_k <= 0 ||
            runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots))
            return;

        if (allow_pending_transfers)
        {
            const bool transfer_complete =
                prefill_llep_transfer_status_complete(
                    transfer_status,
                    runtime->reserved_u64[3],
                    runtime->reserved_u64[2]);
            const bool apply_complete =
                prefill_llep_apply_status_complete(
                    apply_status,
                    runtime->reserved_u64[3]);
            if (!transfer_complete || !apply_complete)
            {
                /*
                 * Every expert block observes the same publication-status
                 * objects.  Designate exactly one lane as the fatal witness so
                 * its diagnostic reaches the device printf stream before the
                 * assertion poisons the CUDA context.  Having every lane
                 * assert is equally fatal but floods the error channel and can
                 * hide the status fields needed to identify the broken
                 * ordering or publication contract.
                 */
                if (expert == 0 && threadIdx.x == 0)
                {
                    int32_t duplicate_slot = -1;
                    int32_t duplicate_first_expert = -1;
                    int32_t duplicate_second_expert = -1;
                    if (execution_bank <= 1u)
                    {
                        const auto &bank = runtime_placement_banks(runtime)[execution_bank];
                        const uint32_t local_bit =
                            runtime_participant_bit(
                                static_cast<int>(runtime->participant_id));
                        for (int32_t first = 0;
                             first < num_experts &&
                             duplicate_slot < 0;
                             ++first)
                        {
                            const auto &first_desc = bank.experts[first];
                            const bool first_live =
                                bank.local_compute_mask[first] != 0u &&
                                (bank.resident_participant_mask[first] &
                                 local_bit) != 0u &&
                                (first_desc.flags & kDeviceMoEFlagTransferSlot) != 0u &&
                                first_desc.local_slot >= 0;
                            if (!first_live)
                                continue;
                            for (int32_t second = first + 1;
                                 second < num_experts;
                                 ++second)
                            {
                                const auto &second_desc = bank.experts[second];
                                const bool second_live =
                                    bank.local_compute_mask[second] != 0u &&
                                    (bank.resident_participant_mask[second] &
                                     local_bit) != 0u &&
                                    (second_desc.flags &
                                     kDeviceMoEFlagTransferSlot) != 0u &&
                                    second_desc.local_slot >= 0;
                                if (second_live &&
                                    second_desc.local_slot ==
                                        first_desc.local_slot)
                                {
                                    duplicate_slot = first_desc.local_slot;
                                    duplicate_first_expert = first;
                                    duplicate_second_expert = second;
                                    break;
                                }
                            }
                        }
                    }
                    printf("prefill_llep_after_transfer_incomplete "
                           "participant=%u expected_transfers=%llu expected_spans=%llu "
                           "transfer_ok=%u transfer_magic=%u transfer_version=%u transfer_status=%u "
                           "plan_overflow=%u payload_overflow=%u planned=%u projected=%llu "
                           "transfer_count=%u span_count=%u capacity_limited=%u "
                           "payload_requested=%u payload_bucket=%u "
                           "duplicate_slot=%d duplicate_first=%d duplicate_second=%d\n",
                           runtime->participant_id,
                           static_cast<unsigned long long>(runtime->reserved_u64[3]),
                           static_cast<unsigned long long>(runtime->reserved_u64[2]),
                           transfer_complete ? 1u : 0u,
                           transfer_status ? transfer_status->magic : 0u,
                           transfer_status ? transfer_status->version : 0u,
                           transfer_status ? transfer_status->status_code : 0u,
                           transfer_status ? transfer_status->plan_overflow : 0u,
                           transfer_status ? transfer_status->payload_bucket_overflow : 0u,
                           transfer_status ? transfer_status->planned_arrivals : 0u,
                           static_cast<unsigned long long>(
                               transfer_status
                                   ? transfer_status->candidate_arrivals_considered
                                   : 0ULL),
                           transfer_status ? transfer_status->llep_weight_transfer_count : 0u,
                           transfer_status ? transfer_status->llep_assignment_span_count : 0u,
                           transfer_status
                               ? transfer_status->capacity_limited_candidates
                               : 0u,
                           transfer_status
                               ? transfer_status->payload_bucket_requested_slots
                               : 0u,
                           transfer_status
                               ? transfer_status->payload_bucket_slots
                               : 0u,
                           duplicate_slot,
                           duplicate_first_expert,
                           duplicate_second_expert);
                    printf("prefill_llep_after_apply_incomplete "
                           "participant=%u apply_ok=%u apply_magic=%u apply_version=%u "
                           "apply_status=%u seen=%u applied=%u required_local=%u "
                           "ready_local=%u invalid=%u missing_src=%u missing_dst=%u "
                           "mismatch=%u copy_incomplete=%u copied=%u\n",
                           runtime->participant_id,
                           apply_complete ? 1u : 0u,
                           apply_status ? apply_status->magic : 0u,
                           apply_status ? apply_status->version : 0u,
                           apply_status ? apply_status->status_code : 0u,
                           apply_status ? apply_status->plan_entries_seen : 0u,
                           apply_status ? apply_status->applied_arrivals : 0u,
                           apply_status ? apply_status->required_local_arrivals : 0u,
                           apply_status ? apply_status->ready_local_arrivals : 0u,
                           apply_status ? apply_status->invalid_plan_entries : 0u,
                           apply_status ? apply_status->missing_source_descriptors : 0u,
                           apply_status ? apply_status->missing_destination_slots : 0u,
                           apply_status ? apply_status->descriptor_mismatches : 0u,
                           apply_status ? apply_status->copy_incomplete : 0u,
                           apply_status ? apply_status->copied_arrivals : 0u);
                    FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                        "post-transfer LLEP status is incomplete or inconsistent");
                }
                return;
            }
        }
        else if (runtime->reserved_u64[3] != 0ULL)
        {
            return;
        }

        const uint64_t span_count_u64 = runtime->reserved_u64[2];
        const uint64_t span_capacity_u64 = runtime->reserved_u64[0];
        const uint32_t span_count =
            span_count_u64 > span_capacity_u64
                ? static_cast<uint32_t>(span_capacity_u64)
                : static_cast<uint32_t>(span_count_u64);
        const auto *spans =
            static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan *>(
                runtime->reserved_ptrs[1]);

        /*
         * Block zero is the single writer for the request-local evidence bit.
         * The marker is published by the same kernel that consumes the span
         * plan, after transfer/apply validation and before any expert block can
         * return for having no routed rows. A later failure is fatal, so a
         * successful request-boundary read proves that the assignment completed.
         * No atomic is needed because one exact lane owns the sticky transition.
         */
        if (expert == 0 && threadIdx.x == 0 &&
            llaminar2::least_loaded_ep::containsNonOwnerAssignmentRows(
                spans, span_count))
        {
            runtime->current_batch_llep_non_owner_assignment_observed = 1u;
        }

        const int expert_count = runtime->expert_counts[expert];
        const int expert_offset = runtime->expert_offsets[expert];
        if (expert_count <= 0 || expert_offset < 0)
            return;

        if (span_count == 0u)
        {
            const int assigned_participant =
                prefill_llep_default_owner_participant(runtime, expert);
            if (assigned_participant < 0)
                return;
            const DeviceMoEPlacementBankView &active_bank =
                runtime_placement_banks(runtime)[execution_bank];

            for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
            {
                if (!prefill_llep_assignment_ready(
                        runtime, active_bank, expert, row, assigned_participant))
                {
                    FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                        "default-owner LLEP assignment is not locally compute-ready");
                    return;
                }
                const int slot = runtime->grouped_token_ids[expert_offset + row];
                if (slot >= 0 && slot < current_slots && slot < max_slots &&
                    runtime->route_expert_ids[slot] == expert)
                {
                    runtime->route_participant_ids[slot] = assigned_participant;
                }
            }
            return;
        }

        const auto *span_bounds = static_cast<const int32_t *>(runtime->reserved_ptrs[0]);
        int span_begin = 0;
        int span_end = static_cast<int>(span_count);
        if (span_bounds)
        {
            span_begin = span_bounds[2 * expert];
            span_end = span_bounds[2 * expert + 1];
            if (span_begin < 0 || span_end <= span_begin)
            {
                const int assigned_participant =
                    prefill_llep_default_owner_participant(runtime, expert);
                if (assigned_participant < 0)
                    return;
                const DeviceMoEPlacementBankView &active_bank =
                    runtime_placement_banks(runtime)[execution_bank];

                for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
                {
                    if (!prefill_llep_assignment_ready(
                            runtime, active_bank, expert, row, assigned_participant))
                    {
                        FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                            "missing-span LLEP assignment is not locally compute-ready");
                        return;
                    }
                    const int slot = runtime->grouped_token_ids[expert_offset + row];
                    if (slot >= 0 && slot < current_slots && slot < max_slots &&
                        runtime->route_expert_ids[slot] == expert)
                    {
                        runtime->route_participant_ids[slot] = assigned_participant;
                    }
                }
                return;
            }
            if (span_begin > static_cast<int>(span_count))
                span_begin = static_cast<int>(span_count);
            if (span_end > static_cast<int>(span_count))
                span_end = static_cast<int>(span_count);
        }

        for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
        {
            int32_t assigned_participant = -1;
            for (int idx = span_begin; idx < span_end; ++idx)
            {
                const auto &span = spans[idx];
                if (span.expert != static_cast<uint32_t>(expert))
                    continue;
                if (static_cast<uint32_t>(row) >= span.route_row_begin &&
                    static_cast<uint32_t>(row) < span.route_row_end)
                {
                    assigned_participant = static_cast<int32_t>(span.destination_participant);
                    break;
                }
            }
            if (assigned_participant < 0)
            {
                assigned_participant =
                    prefill_llep_default_owner_participant(runtime, expert);
                if (assigned_participant < 0)
                    continue;
            }
            const DeviceMoEPlacementBankView &active_bank =
                runtime_placement_banks(runtime)[execution_bank];
            if (!prefill_llep_assignment_ready(
                    runtime, active_bank, expert, row, assigned_participant))
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "planned LLEP span assigns a non-ready local expert");
                return;
            }

            const int slot = runtime->grouped_token_ids[expert_offset + row];
            if (slot >= 0 && slot < current_slots && slot < max_slots &&
                runtime->route_expert_ids[slot] == expert)
            {
                runtime->route_participant_ids[slot] = assigned_participant;
            }
        }
    }

    /**
     * @brief Publish compact runtime descriptors and, for prefill, active ids.
     *
     * Runtime descriptor materialization already visits every expert with one
     * 256-thread block.  Stable active-list construction consumes the same
     * placement predicates and expert-count table, so publishing both products
     * here removes a launch and a duplicate descriptor walk from every MoE
     * layer.  Decode passes no active-list destination and receives only the
     * compact descriptor tables; grouped prefill uses the fused publication.
     */
    __global__ void materialize_runtime_prefill_descriptor_tables_kernel(
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        DeviceNativeQuantisedMatrixDesc *__restrict__ down_descs,
        int num_experts,
        int *__restrict__ active_expert_ids,
        int max_active_experts,
        int current_slots,
        int publish_prefill_histogram,
        llaminar2::DeviceMoEWeightFormat expected_format,
        llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        __shared__ RuntimePrefillPlanPublicationScratch scratch;
        publish_runtime_prefill_plan_block(
            runtime,
            gate_descs,
            up_descs,
            down_descs,
            num_experts,
            active_expert_ids,
            max_active_experts,
            expected_format,
            expected_projections,
            scratch);

        /*
         * Scalable grouping needs a separate descriptor-publication kernel
         * already.  Count final assigned routes here so route-only grouping
         * remains side-effect free and ordinary prefill adds no launch.
         * Aggregate per-expert hits in shared memory first when num_experts
         * fits in the 1024-entry local table so 5k+ slots flush with at most
         * one global/mapped 64-bit atomic per active expert instead of one per
         * route slot.
         */
        if (publish_prefill_histogram != 0 &&
            runtime_histogram_admits_rows(*runtime))
        {
            uint64_t *const sel_hist = runtime_selected_histogram(*runtime, 1u);
            uint64_t *const loc_hist = runtime_local_histogram(*runtime, 1u);
            const int local_pid = static_cast<int>(runtime->participant_id);
            constexpr int kSmemHistCap = 1024;
            if (num_experts <= kSmemHistCap)
            {
                __shared__ uint32_t s_sel[kSmemHistCap];
                __shared__ uint32_t s_loc[kSmemHistCap];
                for (int e = static_cast<int>(threadIdx.x);
                     e < num_experts;
                     e += static_cast<int>(blockDim.x))
                {
                    s_sel[e] = 0u;
                    s_loc[e] = 0u;
                }
                __syncthreads();
                for (int slot = static_cast<int>(threadIdx.x);
                     slot < current_slots;
                     slot += static_cast<int>(blockDim.x))
                {
                    const int expert_id = runtime->route_expert_ids[slot];
                    const int participant_id =
                        runtime->route_participant_ids[slot];
                    if (expert_id < 0 || expert_id >= num_experts)
                        continue;
                    atomicAdd(&s_sel[expert_id], 1u);
                    if (participant_id == local_pid)
                        atomicAdd(&s_loc[expert_id], 1u);
                }
                __syncthreads();
                for (int e = static_cast<int>(threadIdx.x);
                     e < num_experts;
                     e += static_cast<int>(blockDim.x))
                {
                    const uint32_t sc = s_sel[e];
                    if (sc != 0u)
                    {
                        atomicAdd(
                            reinterpret_cast<unsigned long long *>(&sel_hist[e]),
                            static_cast<unsigned long long>(sc));
                    }
                    const uint32_t lc = s_loc[e];
                    if (lc != 0u)
                    {
                        atomicAdd(
                            reinterpret_cast<unsigned long long *>(&loc_hist[e]),
                            static_cast<unsigned long long>(lc));
                    }
                }
            }
            else
            {
                for (int slot = static_cast<int>(threadIdx.x);
                     slot < current_slots;
                     slot += static_cast<int>(blockDim.x))
                {
                    const int expert_id = runtime->route_expert_ids[slot];
                    const int participant_id =
                        runtime->route_participant_ids[slot];
                    if (expert_id < 0 || expert_id >= num_experts)
                        continue;
                    atomicAdd(
                        reinterpret_cast<unsigned long long *>(&sel_hist[expert_id]),
                        1ULL);
                    if (participant_id == local_pid)
                    {
                        atomicAdd(
                            reinterpret_cast<unsigned long long *>(&loc_hist[expert_id]),
                            1ULL);
                    }
                }
            }
        }
    }

    /**
     * @brief Publish one uniform floating descriptor family from the active bank.
     *
     * The destination lives in the same graph-stable descriptor workspace used
     * by NativeQuantised. Floating descriptors are smaller, so selecting this typed
     * view changes neither arena capacity nor captured pointer identity. Every
     * lane overwrites its expert entry, including explicit zeros for remote
     * experts, before the following grouped projection consumes the table.
     */
    __global__ void materialize_runtime_floating_descriptor_tables_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (!runtime || !gate_descs || !up_descs || !down_descs ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            execution_bank > 1u ||
            !llaminar2::deviceMoEWeightFormatIsFloating(expected_format) ||
        runtime->intermediate_shard.enabled() != (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard))
        {
            if (threadIdx.x == 0)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "floating runtime descriptor publication has an invalid contract");
            }
            return;
        }
        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const uint32_t local_bit =
            runtime->intermediate_shard.enabled() ? 0u : runtime_participant_bit(static_cast<int>(runtime->participant_id));
        // Every admitted descriptor is overwritten on each replay, including
        // remote entries and logical experts above the fixed launch width.
        for (int expert = static_cast<int>(threadIdx.x); expert < num_experts;
             expert += static_cast<int>(blockDim.x))
        {
            const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
            const bool local_ready = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                ? runtime_intermediate_shard_row(runtime, bank, expert).disposition ==
                      llaminar2::MoEIntermediateShardRowDisposition::Local && desc.weight_format == expected_format
                : bank.local_compute_mask[expert] != 0u &&
                (bank.resident_participant_mask[expert] & local_bit) != 0u &&
                desc.local_slot >= 0 &&
                desc.weight_format == expected_format &&
                desc.floating_gate.valid() &&
                desc.floating_up.valid() &&
                desc.floating_down.valid();
            if (local_ready)
            {
                gate_descs[expert] = desc.floating_gate;
                up_descs[expert] = desc.floating_up;
                down_descs[expert] = desc.floating_down;
            }
            else
            {
                gate_descs[expert] = llaminar2::DeviceMoEFloatingMatrixDesc{};
                up_descs[expert] = llaminar2::DeviceMoEFloatingMatrixDesc{};
                down_descs[expert] = llaminar2::DeviceMoEFloatingMatrixDesc{};
            }
        }
    }

    __global__ void prefill_gather_expert_runtime_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (!runtime || !hidden || !batch_buffer || linear >= total)
            return;
        if (!runtime->expert_counts || !runtime->expert_offsets || !runtime->grouped_token_ids)
            return;
        if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count))
            return;
        if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens))
            return;

        const int row = linear / d_model;
        const int col = linear - row * d_model;
        const int count = runtime->expert_counts[expert_id];
        const int offset = runtime->expert_offsets[expert_id];

        float value = 0.0f;
        if (row < count)
        {
            const int route_slot = runtime->grouped_token_ids[offset + row];
            const int token_id =
                runtime->top_k > 0u ? route_slot / static_cast<int>(runtime->top_k) : -1;
            if (token_id >= 0 && token_id < max_tokens)
                value = hidden[token_id * d_model + col];
        }
        batch_buffer[linear] = value;
    }

    __global__ void prefill_scatter_expert_runtime_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (!runtime || !output || !expert_output || linear >= total)
            return;
        if (!runtime->expert_counts || !runtime->expert_offsets ||
            !runtime->grouped_token_ids || !runtime->grouped_route_weights)
            return;
        if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens))
            return;
        if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count))
            return;

        const int row = linear / d_model;
        const int col = linear - row * d_model;
        const int count = runtime->expert_counts[expert_id];
        if (row >= count)
            return;

        const int offset = runtime->expert_offsets[expert_id];
        const int route_slot = runtime->grouped_token_ids[offset + row];
        const int token_id =
            runtime->top_k > 0u ? route_slot / static_cast<int>(runtime->top_k) : -1;
        if (token_id < 0 || token_id >= max_tokens)
            return;
        const float weight = runtime->grouped_route_weights[offset + row];
        atomicAdd(output + token_id * d_model + col, weight * expert_output[linear]);
    }

    __global__ void gather_expert_fixed_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        float value = 0.0f;
        if (token_slot < count)
        {
            const int grouped = expert_offsets[expert_id] + token_slot;
            const int token = grouped_token_indices[grouped];
            value = hidden[static_cast<size_t>(token) * d_model + col];
        }
        batch_buffer[idx] = value;
    }

    __global__ void scatter_expert_fixed_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        if (token_slot >= count)
            return;

        const int grouped = expert_offsets[expert_id] + token_slot;
        const int token = grouped_token_indices[grouped];
        const float weight = grouped_weights[grouped];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col,
                  weight * expert_output[idx]);
    }

} // namespace

extern "C"
{
    /** @brief Validate and launch the shared grouping-reset contract; see its public declaration. */
    bool cudaMoE_initialize_grouping(
        llaminar2::MoEGroupingInitialization binding, int device_ordinal, void *stream)
    {
        if (!binding.valid() || !stream || cudaSetDevice(device_ordinal) != cudaSuccess)
            return false;
        const int blocks = 1 + (binding.workItems() - 1) / kThreads;
        initialize_grouping_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(binding);
        return finishLaunch("cudaMoE_initialize_grouping");
    }

    bool cudaMoE_count_per_expert(const int *routing_indices, int *expert_counts, int total_slots,
                                  int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        count_per_expert_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, expert_counts, total_slots, num_experts);
        return finishLaunch("cudaMoE_count_per_expert");
    }

    bool cudaMoE_exclusive_scan(const int *expert_counts, int *expert_offsets,
                                int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        exclusive_scan_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(expert_counts, expert_offsets, num_experts);
        return finishLaunch("cudaMoE_exclusive_scan");
    }

    bool cudaMoE_build_active_expert_list(
        const int *expert_counts,
        int *active_expert_ids,
        int num_experts,
        int max_active_experts,
        int device_idx,
        void *stream)
    {
        if (!expert_counts || !active_expert_ids || num_experts <= 0 ||
            max_active_experts <= 0 || max_active_experts > num_experts || !stream)
            return false;
        cudaSetDevice(device_idx);
        build_active_expert_list_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_counts, active_expert_ids, num_experts, max_active_experts);
        return finishLaunch("cudaMoE_build_active_expert_list");
    }

    bool cudaMoE_scatter_tokens(const int *routing_indices, const float *routing_weights,
                                int *write_heads, const int *expert_offsets,
                                int *grouped_token_indices, float *grouped_weights,
                                int total_slots, int top_k, int num_experts,
                                int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        scatter_tokens_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, write_heads, expert_offsets,
            grouped_token_indices, grouped_weights, total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens");
    }

    bool cudaMoE_scatter_tokens_deterministic(
        const int *routing_indices,
        const float *routing_weights,
        const int *expert_offsets,
        const int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int total_slots,
        int top_k,
        int num_experts,
        int device_idx,
        void *stream)
    {
        if (!stream)
            return false;
        cudaSetDevice(device_idx);
        scatter_tokens_deterministic_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, expert_offsets, expert_counts,
            grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
            total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens_deterministic");
    }

    /**
     * @brief Submit one complete grouping with a fixed geometry-selected integer prefix.
     *
     * Prefix selection changes only the compiled implementation. The original ABI,
     * publications, fixed block geometry and single-operation capture remain the
     * same, and an unsuccessful submission never tries another implementation.
     * @param routing_indices Original finite expert-ID device array.
     * @param routing_weights Original FP32 route-weight device array.
     * @param expert_counts Complete per-expert route-count publication.
     * @param expert_offsets Complete per-expert exclusive-offset publication.
     * @param grouped_token_indices Stable expert-major source token rows.
     * @param original_to_grouped Original-slot to grouped-slot inverse map.
     * @param original_expert_ids Original-slot valid expert IDs or minus one.
     * @param grouped_weights Stable expert-major copied FP32 weights.
     * @param active_expert_ids Declared sorted active-ID publication and sentinels.
     * @param total_slots Positive admitted route-slot count, at most 64.
     * @param num_experts Positive expert extent, at most 512.
     * @param top_k Positive route width under the existing kMaxTopK admission, dividing total_slots.
     * @param max_active_experts Positive active extent, at most min(total_slots, num_experts).
     * @param device_idx Exact native device owning the stream and buffers.
     * @param stream Exact non-null stream retained until all borrowed buffers retire.
     * @return True only after the selected native grouping launch succeeds.
     */
    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices,
        const float *routing_weights,
        int *expert_counts,
        int *expert_offsets,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int device_idx,
        void *stream)
    {
        if (!stream ||
            !routing_indices || !routing_weights ||
            !expert_counts || !expert_offsets ||
            !grouped_token_indices || !original_to_grouped ||
            !original_expert_ids || !grouped_weights ||
            !active_expert_ids ||
            total_slots <= 0 ||
            total_slots > kDeviceMoESmallGroupMaxSlots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK ||
            total_slots % top_k != 0 ||
            max_active_experts <= 0 ||
            max_active_experts > total_slots ||
            max_active_experts > num_experts)
        {
            std::fprintf(stderr, "CUDA MoE small float grouping invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        const auto prefix_policy = small_float_group_prefix_policy(num_experts);
        if (prefix_policy == SmallFloatGroupPrefixPolicy::IndependentExpert)
        {
            group_tokens_small_float_kernel<SmallFloatGroupPrefixPolicy::IndependentExpert><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0,
                static_cast<cudaStream_t>(stream)>>>(
                routing_indices, routing_weights, expert_counts, expert_offsets,
                grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
                active_expert_ids, total_slots, num_experts, top_k, max_active_experts);
        }
        else
        {
            group_tokens_small_float_kernel<SmallFloatGroupPrefixPolicy::CooperativeScan><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0,
                static_cast<cudaStream_t>(stream)>>>(
                routing_indices, routing_weights, expert_counts, expert_offsets,
                grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
                active_expert_ids, total_slots, num_experts, top_k, max_active_experts);
        }
        return finishLaunch("cudaMoE_group_tokens_small_float");
    }

    bool cudaMoE_prepare_shared_expert_group(
        int *expert_offsets,
        int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        llaminar2::DeviceRowRange rows,
        int device_idx,
        void *stream)
    {
        const int seq_len = rows.physicalRows();
        if (!stream || !expert_offsets || !expert_counts ||
            !grouped_token_indices || !original_to_grouped ||
            !original_expert_ids || !grouped_weights || !active_expert_ids)
            return false;
        cudaSetDevice(device_idx);
        prepare_shared_expert_group_kernel<<<blocksFor(seq_len), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_offsets, expert_counts, grouped_token_indices, original_to_grouped,
            original_expert_ids, grouped_weights, active_expert_ids, rows);
        return finishLaunch("cudaMoE_prepare_shared_expert_group");
    }

    bool cudaMoE_group_prefill_routes_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        if (!runtime || !llaminar2::deviceMoEProjectionSetValid(expected_projections) || !routing_indices || !routing_weights ||
            !original_to_grouped || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
        {
            prefill_group_small_runtime_kernel<true><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0, cuda_stream>>>(
                runtime_view,
                routing_indices,
                routing_weights,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                filter_to_local_runtime_experts,
                retain_routes_for_deferred_commit,
                /*gate_descs=*/nullptr,
                /*up_descs=*/nullptr,
                /*down_descs=*/nullptr,
                /*active_expert_ids=*/nullptr,
                /*max_active_experts=*/0,
                llaminar2::DeviceMoEWeightFormat::NativeQuantised,
                expected_projections, {});
            return finishGroupedPrefillLaunch(
                "cudaMoE_prefill_group_small_runtime", cuda_stream);
        }

        const int clear_items = max_slots > num_experts ? max_slots : num_experts;
        prefill_group_clear_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, original_to_grouped, max_slots, num_experts, 1);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_group_clear_runtime", cuda_stream))
            return false;

        prefill_group_cast_count_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, routing_indices, routing_weights,
            current_slots, max_slots, num_experts,
            filter_to_local_runtime_experts,
            retain_routes_for_deferred_commit,
            expected_projections);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_group_cast_count_runtime", cuda_stream))
            return false;

        prefill_group_scan_scatter_deterministic_runtime_kernel<<<
            num_experts, kThreads, 0, cuda_stream>>>(
            runtime_view, original_to_grouped,
            current_slots, max_slots, num_experts, top_k);
        return finishGroupedPrefillLaunch(
            "cudaMoE_prefill_group_scan_scatter_runtime", cuda_stream);
    }

    /**
     * @brief Publish router grouping, descriptor tables, and active ids as one plan.
     *
     * Verifier-sized route sets execute as one block and therefore publish the
     * complete grouped execution plan in one kernel. Larger prefill shapes keep
     * the scalable deterministic grouping geometry, then materialize descriptors
     * on the same stream. These are explicit geometry regimes of one total API;
     * neither path changes arithmetic or transfers state through the host.
     */
    __attribute__((visibility("default"))) bool
    cudaMoE_group_prefill_routes_and_materialize_plan_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int *original_to_grouped,
        DeviceNativeQuantisedMatrixDesc *gate_descs,
        DeviceNativeQuantisedMatrixDesc *up_descs,
        DeviceNativeQuantisedMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int filter_to_local_runtime_experts,
        int grouped_demand,
        llaminar2::DeviceMoEWeightFormat expected_format,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract)
    {
        const auto demand = static_cast<llaminar2::MoEGroupedPlanDemand>(grouped_demand);
        if (!llaminar2::validMoEGroupedPlanDemand(demand)) return false;
        const int retain_routes_for_deferred_commit =
            demand == llaminar2::MoEGroupedPlanDemand::DeferredAcceptedRows;
        if (!runtime || !llaminar2::deviceMoEProjectionSetValid(expected_projections) || !routing_indices || !routing_weights ||
            !original_to_grouped || !gate_descs || !up_descs || !down_descs ||
            !active_expert_ids || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK ||
            max_active_experts <= 0 || max_active_experts > num_experts ||
            (expected_format !=
                 llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
             !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)))
        {
            return false;
        }

        if (gate_contract.enabled() &&
            (expected_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised ||
             top_k != llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_routes_per_token ||
             current_slots != top_k * llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_gate_plan_rows ||
             max_slots != current_slots))
            return false;

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
        {
            prefill_group_small_runtime_kernel<true, true><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0, cuda_stream>>>(
                runtime_view,
                routing_indices,
                routing_weights,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                filter_to_local_runtime_experts,
                grouped_demand,
                gate_descs,
                up_descs,
                down_descs,
                active_expert_ids,
                max_active_experts,
                expected_format,
                expected_projections, gate_contract);
            return finishGroupedPrefillLaunch(
                "cudaMoE_prefill_group_and_plan_small_runtime", cuda_stream);
        }

        if (!cudaMoE_group_prefill_routes_runtime(
                routing_indices,
                routing_weights,
                runtime,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                filter_to_local_runtime_experts,
                retain_routes_for_deferred_commit,
                device_idx,
                stream,
                expected_projections))
        {
            return false;
        }

        materialize_runtime_prefill_descriptor_tables_kernel<<<
            1, kThreads, 0, cuda_stream>>>(
            runtime_view,
            gate_descs,
            up_descs,
            down_descs,
            num_experts,
            active_expert_ids,
            max_active_experts,
            current_slots,
            demand == llaminar2::MoEGroupedPlanDemand::OrdinaryPrefill ? 1 : 0,
            expected_format,
            expected_projections);
        return finishGroupedPrefillLaunch(
            "cudaMoE_prefill_group_and_plan_scalable_runtime", cuda_stream);
    }

    bool cudaMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream)
    {
        if (!runtime || !original_to_grouped || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
        {
            prefill_group_small_runtime_kernel<false><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0, cuda_stream>>>(
                runtime_view,
                nullptr,
                nullptr,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                /*filter_to_local_runtime_experts=*/0,
                retain_routes_for_deferred_commit,
                /*gate_descs=*/nullptr,
                /*up_descs=*/nullptr,
                /*down_descs=*/nullptr,
                /*active_expert_ids=*/nullptr,
                /*max_active_experts=*/0,
                llaminar2::DeviceMoEWeightFormat::NativeQuantised,
                llaminar2::DeviceMoEProjectionSet::CompleteExpert, {});
            return finishGroupedPrefillLaunch(
                "cudaMoE_prefill_regroup_small_runtime", cuda_stream);
        }

        const int clear_items = max_slots > num_experts ? max_slots : num_experts;
        prefill_group_clear_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, original_to_grouped, max_slots, num_experts, 0);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_regroup_clear_runtime", cuda_stream))
            return false;

        prefill_group_count_assigned_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts,
            retain_routes_for_deferred_commit);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_regroup_count_assigned_runtime", cuda_stream))
            return false;

        prefill_group_scan_scatter_deterministic_runtime_kernel<<<
            num_experts, kThreads, 0, cuda_stream>>>(
            runtime_view, original_to_grouped,
            current_slots, max_slots, num_experts, top_k);
        return finishGroupedPrefillLaunch(
            "cudaMoE_prefill_regroup_scan_scatter_runtime", cuda_stream);
    }

    /**
     * @brief Publish an assigned-route grouping and its complete execution plan.
     *
     * LLEP updates participant ids before this boundary. The verifier-sized
     * specialization consumes those final assignments and publishes grouped
     * rows, descriptors, and active ids in one block; scalable prefill retains
     * its multi-block deterministic grouping followed by one ordered descriptor
     * publication on the same stream.
     */
    __attribute__((visibility("default"))) bool
    cudaMoE_regroup_prefill_routes_and_materialize_plan_runtime(
        void *runtime,
        int *original_to_grouped,
        DeviceNativeQuantisedMatrixDesc *gate_descs,
        DeviceNativeQuantisedMatrixDesc *up_descs,
        DeviceNativeQuantisedMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int grouped_demand,
        llaminar2::DeviceMoEWeightFormat expected_format,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract)
    {
        const auto demand = static_cast<llaminar2::MoEGroupedPlanDemand>(grouped_demand);
        if (!llaminar2::validMoEGroupedPlanDemand(demand)) return false;
        const int retain_routes_for_deferred_commit =
            demand == llaminar2::MoEGroupedPlanDemand::DeferredAcceptedRows;
        if (!runtime || !llaminar2::deviceMoEProjectionSetValid(expected_projections) || !original_to_grouped ||
            !gate_descs || !up_descs || !down_descs ||
            !active_expert_ids || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK ||
            max_active_experts <= 0 || max_active_experts > num_experts ||
            (expected_format !=
                 llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
             !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)))
        {
            return false;
        }

        if (gate_contract.enabled() &&
            (expected_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised ||
             top_k != llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_routes_per_token ||
             current_slots != top_k * llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_gate_plan_rows ||
             max_slots != current_slots))
            return false;

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
        {
            prefill_group_small_runtime_kernel<false, true><<<
                1, llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts), 0, cuda_stream>>>(
                runtime_view,
                /*routing_indices=*/nullptr,
                /*routing_weights=*/nullptr,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                /*filter_to_local_runtime_experts=*/0,
                grouped_demand,
                gate_descs,
                up_descs,
                down_descs,
                active_expert_ids,
                max_active_experts,
                expected_format,
                expected_projections, gate_contract);
            return finishGroupedPrefillLaunch(
                "cudaMoE_prefill_regroup_and_plan_small_runtime", cuda_stream);
        }

        if (!cudaMoE_regroup_prefill_routes_runtime_assignments(
                runtime,
                original_to_grouped,
                current_slots,
                max_slots,
                num_experts,
                top_k,
                retain_routes_for_deferred_commit,
                device_idx,
                stream))
        {
            return false;
        }

        materialize_runtime_prefill_descriptor_tables_kernel<<<
            1, kThreads, 0, cuda_stream>>>(
            runtime_view,
            gate_descs,
            up_descs,
            down_descs,
            num_experts,
            active_expert_ids,
            max_active_experts,
            current_slots,
            demand == llaminar2::MoEGroupedPlanDemand::OrdinaryPrefill ? 1 : 0,
            expected_format,
            expected_projections);
        return finishGroupedPrefillLaunch(
            "cudaMoE_prefill_regroup_and_plan_scalable_runtime", cuda_stream);
    }

    bool cudaMoE_commit_grouped_verifier_histograms(
        void *runtime,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream)
    {
        if (!runtime ||
            !accepted_state_counts ||
            !publication_ok_flags ||
            !stream ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            total_rows != request_count * rows_per_request ||
            num_experts <= 0 ||
            num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 ||
            top_k > kMaxTopK)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int total_slots = total_rows * top_k;
        commit_grouped_verifier_histograms_runtime_kernel<<<
            blocksFor(total_slots),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(runtime),
            accepted_state_counts,
            publication_ok_flags,
            request_count,
            rows_per_request,
            total_rows,
            num_experts,
            top_k);
        return finishGroupedPrefillLaunch(
            "cudaMoE_commit_grouped_verifier_histograms",
            static_cast<cudaStream_t>(stream));
    }

    bool cudaMoE_assign_prefill_routes_least_loaded_resident(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        const int32_t *absolute_position_ids,
        const int32_t *active_row_count,
        int device_idx,
        void *stream)
    {
        if (!runtime || !absolute_position_ids || !active_row_count || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK || top_k > num_experts ||
            current_slots % top_k != 0 || max_slots % top_k != 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        constexpr int kAssignmentThreads = 32;
        const int current_rows = current_slots / top_k;
        const int assignment_blocks =
            current_rows > 0
                ? (current_rows + kAssignmentThreads - 1) / kAssignmentThreads
                : 1;
        prefill_llep_assign_resident_rows_logical_position_runtime_kernel
            <<<assignment_blocks, kAssignmentThreads, 0, cuda_stream>>>(
                runtime_view,
                absolute_position_ids,
                active_row_count,
                current_slots,
                max_slots,
                num_experts,
                top_k);
        return finishGroupedPrefillLaunch(
            "cudaMoE_prefill_llep_assign_resident_rows_logical_position",
            cuda_stream);
    }

    bool cudaMoE_plan_prefill_routes_least_loaded_current_batch(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        uint32_t min_chunk_tokens,
        uint32_t alpha_numerator,
        uint32_t alpha_denominator,
        uint32_t lambda_numerator,
        uint32_t lambda_denominator,
        uint64_t min_spread_improvement,
        uint32_t min_spread_improvement_divisor,
        uint64_t min_spread_improvement_per_critical_path_slot,
        uint64_t min_foreign_rows_per_critical_path_slot,
        uint32_t max_weight_transfers,
        uint32_t max_non_owner_experts_per_participant,
        int enable_balanced_skip,
        int device_idx,
        void *stream)
    {
        (void)top_k;
        if (!runtime || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            alpha_numerator == 0u || alpha_denominator == 0u ||
            lambda_numerator == 0u || lambda_denominator == 0u)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
        const int clear_items = split_items > num_experts ? split_items : num_experts;
        prefill_llep_clear_current_batch_plan_runtime_kernel<<<blocksFor(clear_items), kThreads, 0, cuda_stream>>>(
            runtime_view, num_experts);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_llep_current_batch_clear", cuda_stream))
            return false;

        prefill_llep_count_expert_routes_runtime_kernel<<<blocksFor(max_slots), kThreads, 0, cuda_stream>>>(
            runtime_view, current_slots, max_slots, num_experts);
        if (!finishGroupedPrefillLaunch("cudaMoE_prefill_llep_current_batch_count_routes", cuda_stream))
            return false;

        prefill_llep_plan_current_batch_runtime_kernel<<<1, kThreads, 0, cuda_stream>>>(
            runtime_view,
            current_slots,
            max_slots,
            num_experts,
            min_chunk_tokens,
            alpha_numerator,
            alpha_denominator,
            lambda_numerator,
            lambda_denominator,
            min_spread_improvement,
            min_spread_improvement_divisor,
            min_spread_improvement_per_critical_path_slot,
            min_foreign_rows_per_critical_path_slot,
            max_weight_transfers,
            max_non_owner_experts_per_participant,
            enable_balanced_skip);
        return finishGroupedPrefillLaunch("cudaMoE_prefill_llep_plan_current_batch", cuda_stream);
    }

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream)
    {
        if (!runtime || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        prefill_llep_assign_routes_from_current_batch_spans_runtime_kernel<<<
            num_experts, kThreads, 0, cuda_stream>>>(
            runtime_view,
            current_slots,
            max_slots,
            num_experts,
            top_k,
            0,
            nullptr,
            nullptr);
        return finishGroupedPrefillLaunch("cudaMoE_prefill_llep_assign_routes_from_current_batch_spans", cuda_stream);
    }

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        const void *transfer_status,
        const void *apply_status,
        int device_idx,
        void *stream)
    {
        if (!runtime || !transfer_status || !apply_status || !stream ||
            current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);

        prefill_llep_assign_routes_from_current_batch_spans_runtime_kernel<<<
            num_experts, kThreads, 0, cuda_stream>>>(
            runtime_view,
            current_slots,
            max_slots,
            num_experts,
            top_k,
            1,
            static_cast<const DeviceMoERebalanceStatusView *>(transfer_status),
            static_cast<const DeviceMoERebalanceApplyStatusView *>(apply_status));
        return finishGroupedPrefillLaunch("cudaMoE_prefill_llep_after_transfers_assign_routes_from_spans", cuda_stream);
    }

    /**
     * @brief Publish this decode packet's descriptors without scanning source capacity.
     * @param runtime Device runtime retaining the acquired placement epoch.
     * @param expert_ids Exact device ID packet also consumed by gate/up and down.
     * @param gate_descs Stable full-domain gate storage; only selected IDs are written.
     * @param up_descs Stable full-domain up storage; only selected IDs are written.
     * @param down_descs Stable full-domain down storage; only selected IDs are written.
     * @param num_experts Admitted logical table extent for bounds validation.
     * @param top_k Actual packet width, not expert or workspace capacity.
     * @param format Original descriptor scalar family; no conversion is performed.
     * @param device_idx Original local device ordinal.
     * @param stream Exact producer/consumer stream retained by capture.
     * @param projections Original movable projection ownership contract.
     * @return Whether admission and native submission succeeded.
     */
    __attribute__((visibility("default"))) bool cudaMoE_materialize_runtime_selected_decode_descriptors(
        const void *runtime, const int *expert_ids,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs, llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs, int num_experts, int top_k,
        llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet projections)
    {
        if (!runtime || !expert_ids || !gate_descs || !up_descs || !down_descs || !stream ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 || top_k > kMaxTopK || top_k > num_experts ||
            !llaminar2::deviceMoEProjectionSetValid(projections) ||
            (format != llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
             !llaminar2::deviceMoEWeightFormatIsFloating(format))) return false;
        if (cudaSetDevice(device_idx) != cudaSuccess) return false;
        materialize_runtime_selected_decode_descriptors_kernel<<<1, top_k, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime), expert_ids,
            reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(gate_descs), reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(up_descs), reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(down_descs), num_experts, top_k, format, projections);
        return finishGroupedPrefillLaunch("cudaMoE_materialize_runtime_selected_decode_descriptors",
            static_cast<cudaStream_t>(stream));
    }

    __attribute__((visibility("default"))) bool cudaMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int device_idx,
        void *stream, llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        if (!runtime || !gate_descs || !up_descs || !down_descs || !stream ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        materialize_runtime_prefill_descriptor_tables_kernel<<<1, kThreads, 0,
                                                               static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(
                const_cast<void *>(runtime)),
            reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(gate_descs),
            reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(up_descs),
            reinterpret_cast<DeviceNativeQuantisedMatrixDesc *>(down_descs),
            num_experts,
            /*active_expert_ids=*/nullptr,
            /*max_active_experts=*/0,
            /*current_slots=*/0,
            /*publish_prefill_histogram=*/0,
            llaminar2::DeviceMoEWeightFormat::NativeQuantised,
            expected_projections);
        return finishGroupedPrefillLaunch(
            "cudaMoE_materialize_runtime_prefill_descriptor_tables",
            static_cast<cudaStream_t>(stream));
    }

    /**
     * @brief Fuse runtime descriptor and active-expert publication for prefill.
     */
    __attribute__((visibility("default"))) bool cudaMoE_materialize_runtime_prefill_plan(
        const void *runtime,
        DeviceNativeQuantisedMatrixDesc *gate_descs,
        DeviceNativeQuantisedMatrixDesc *up_descs,
        DeviceNativeQuantisedMatrixDesc *down_descs,
        int *active_expert_ids,
        int num_experts,
        int max_active_experts,
        int device_idx,
        void *stream)
    {
        if (!runtime || !gate_descs || !up_descs || !down_descs ||
            !active_expert_ids || !stream ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            max_active_experts <= 0 || max_active_experts > num_experts)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        materialize_runtime_prefill_descriptor_tables_kernel<<<
            1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(
                const_cast<void *>(runtime)),
            gate_descs,
            up_descs,
            down_descs,
            num_experts,
            active_expert_ids,
            max_active_experts,
            /*current_slots=*/0,
            /*publish_prefill_histogram=*/0,
            llaminar2::DeviceMoEWeightFormat::NativeQuantised,
            llaminar2::DeviceMoEProjectionSet::CompleteExpert);
        return finishGroupedPrefillLaunch(
            "cudaMoE_materialize_runtime_prefill_plan",
            static_cast<cudaStream_t>(stream));
    }

    bool cudaMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream)
    {
        if (!runtime || !hidden || !batch_buffer || !stream ||
            expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        prefill_gather_expert_runtime_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime),
            hidden,
            batch_buffer,
            expert_id,
            max_tokens,
            d_model);
        return finishLaunch("cudaMoE_prefill_gather_expert_runtime");
    }

    bool cudaMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream)
    {
        if (!output || !expert_output || !runtime || !stream ||
            expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        prefill_scatter_expert_runtime_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            output,
            expert_output,
            static_cast<const DeviceMoELayerRuntimeView *>(runtime),
            expert_id,
            max_tokens,
            d_model);
        return finishLaunch("cudaMoE_prefill_scatter_expert_runtime");
    }

    bool cudaMoE_gather_expert_fixed(const float *hidden, float *batch_buffer,
                                     const int *expert_offsets, const int *expert_counts,
                                     const int *grouped_token_indices,
                                     int expert_id, int max_tokens, int d_model,
                                     int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        gather_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, batch_buffer, expert_offsets, expert_counts, grouped_token_indices,
            expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_gather_expert_fixed");
    }

    bool cudaMoE_scatter_expert_fixed(float *output, const float *expert_output,
                                      const int *expert_offsets, const int *expert_counts,
                                      const int *grouped_token_indices,
                                      const float *grouped_weights,
                                      int expert_id, int max_tokens, int d_model,
                                      int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        scatter_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            output, expert_output, expert_offsets, expert_counts, grouped_token_indices,
            grouped_weights, expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_scatter_expert_fixed");
    }


    /** @brief Materialize a graph-stable floating runtime descriptor family. */
    bool cudaMoE_materialize_runtime_floating_descriptor_tables(
        const void *d_runtime_layer,
        llaminar2::DeviceMoEFloatingMatrixDesc *d_gate_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *d_up_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *d_down_descs,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream, llaminar2::DeviceMoEProjectionSet expected_projections)
    {
        if (!d_runtime_layer || !d_gate_descs || !d_up_descs || !d_down_descs ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
            !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
        {
            std::fprintf(
                stderr,
                "[cudaMoE_materialize_runtime_floating_descriptor_tables] invalid arguments\n");
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        materialize_runtime_floating_descriptor_tables_kernel<<<
            1,
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
                static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer),
                d_gate_descs,
                d_up_descs,
                d_down_descs,
                num_experts,
                format, expected_projections);
        return finishLaunch(
            "cudaMoE_materialize_runtime_floating_descriptor_tables");
    }

} // extern "C"
