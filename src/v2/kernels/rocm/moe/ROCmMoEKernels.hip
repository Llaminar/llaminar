/**
 * @file ROCmMoEKernels.hip
 * @brief ROCm MoE token grouping, active-expert compaction, prefill LLEP assignment, floating MoE, and pointer staging kernels.
 */
#include "ROCmMoERebalanceCommon.hip.inc"

namespace
{
    constexpr int kMoEPointerStageMax = 16;

    struct MoEPointerStagePayload
    {
        const float *const_ptrs_a[kMoEPointerStageMax];
        const float *const_ptrs_b[kMoEPointerStageMax];
        float *mutable_ptrs_a[kMoEPointerStageMax];
        float *mutable_ptrs_b[kMoEPointerStageMax];
    };

    __global__ void moe_stage_const_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        const float **dst_a,
        const float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.const_ptrs_a[i];
        dst_b[i] = payload.const_ptrs_b[i];
    }

    __global__ void moe_stage_mutable_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        float **dst_a,
        float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.mutable_ptrs_a[i];
        dst_b[i] = payload.mutable_ptrs_b[i];
    }

} // namespace

// =========================================================================
// Phase 3 Kernels: Device-Side Token Grouping
// =========================================================================

// =========================================================================
// Kernel 10: Count Per Expert
//
// Counts how many token-slots go to each expert via atomicAdd.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

/**
 * @brief Initialize grouping arrays using the same publication as CUDA.
 * @param binding Persistent workspace arrays with independently bounded extents.
 */
__global__ void rocm_moe_initialize_grouping_kernel(llaminar2::MoEGroupingInitialization binding)
{
    binding.resetIndex(blockIdx.x * blockDim.x + threadIdx.x);
}

__global__ void rocm_moe_count_per_expert_kernel(
    const int *__restrict__ routing_indices, // [total_slots]
    int *__restrict__ expert_counts,         // [num_experts]
    int total_slots, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;

    int expert_id = routing_indices[idx];
    if (expert_id >= 0 && expert_id < num_experts)
    {
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// =========================================================================
// Kernel 11: Exclusive Scan
//
// Computes exclusive prefix sum on expert_counts → expert_offsets.
// One cooperative block processes ascending chunks without host-visible state.
// Expert counts remain immutable and integer addition preserves exact offsets.
// =========================================================================

/**
 * @brief Compute the exact expert prefix with coalesced cooperative work.
 * @param expert_counts Immutable route counts; the sum fits the admitted capacity.
 * @param expert_offsets Output offsets in ascending expert order.
 * @param num_experts Expert count, including counts beyond one workgroup width.
 */
__global__ void rocm_moe_exclusive_scan_kernel(
    int *__restrict__ expert_counts,   // [num_experts] IN (read, then overwritten with original counts)
    int *__restrict__ expert_offsets,   // [num_experts] OUT (exclusive prefix sum)
    int num_experts)
{
    __shared__ llaminar2::device_integer_scan::Scratch<kGateLogitsGroupedThreads> scan;
    llaminar2::device_integer_scan::arrayExclusive(
        expert_counts, expert_offsets, num_experts, scan);
}

// =========================================================================
// Kernel 11b: Max Expert Count (device-side reduction)
//
// Computes max(expert_counts[0..num_experts-1]) entirely on device.
// Single wavefront (64 threads), warp-shuffle reduction.
// Grid: 1 block, Block: 64 threads
// =========================================================================

__global__ void rocm_moe_max_expert_count_kernel(
    const int *__restrict__ expert_counts,  // [num_experts]
    int *__restrict__ d_max_out,            // [1] output scalar
    int num_experts)
{
    const int tid = threadIdx.x;
    int local_max = 0;

    // Each thread processes multiple experts (stride 64)
    for (int e = tid; e < num_experts; e += 64)
        local_max = max(local_max, expert_counts[e]);

    // Warp-shuffle reduction within a single wavefront (64 threads)
    for (int offset = 32; offset > 0; offset >>= 1)
        local_max = max(local_max, __shfl_down(local_max, offset));

    if (tid == 0)
        *d_max_out = local_max;
}

constexpr int kActiveExpertThreads = 256;
// gfx9 uses 64-lane waves, while newer HIP targets may use wave32.  Eight
// entries cover either mode for this fixed 256-thread launch.
constexpr int kActiveExpertMaxWaveCount = kActiveExpertThreads / 32;

/**
 * @brief Shared prefix state for stable active-expert compaction.
 *
 * Downstream grouped kernels require ascending expert ids.  Scheduling an
 * atomic append would violate that contract, so every wave publishes a count
 * and lane zero assigns wave offsets in deterministic wave order.
 */
struct ActiveExpertCompactionScratch
{
    int wave_counts[kActiveExpertMaxWaveCount];
    int wave_offsets[kActiveExpertMaxWaveCount];
    int chunk_base;
    int total_active;
};

/** @brief Initialize block-owned compaction state before the first chunk. */
__device__ __forceinline__ void rocm_initialize_active_expert_compaction(
    ActiveExpertCompactionScratch &scratch)
{
    if (threadIdx.x == 0)
        scratch.total_active = 0;
    __syncthreads();
}

/**
 * @brief Append one 256-expert chunk without changing ascending output order.
 *
 * @param is_active True when this lane's expert must be retained.
 * @param expert_id Ascending expert id represented by this lane.
 * @param max_active_experts Capacity of @p active_expert_ids.
 * @param active_expert_ids Ordered compact output, padded separately.
 * @param scratch Block-owned prefix state shared across chunks.
 */
__device__ __forceinline__ void rocm_compact_active_expert_chunk(
    bool is_active,
    int expert_id,
    int max_active_experts,
    int *__restrict__ active_expert_ids,
    ActiveExpertCompactionScratch &scratch)
{
    const int lane = static_cast<int>(threadIdx.x) % warpSize;
    const int wave = static_cast<int>(threadIdx.x) / warpSize;
    const int wave_count =
        (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
    const unsigned long long votes = __ballot(is_active);

    if (lane == 0)
        scratch.wave_counts[wave] = __popcll(votes);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int chunk_active = 0;
        scratch.chunk_base = scratch.total_active;
        for (int current_wave = 0;
             current_wave < wave_count;
             ++current_wave)
        {
            scratch.wave_offsets[current_wave] = chunk_active;
            chunk_active += scratch.wave_counts[current_wave];
        }
        scratch.total_active += chunk_active;
    }
    __syncthreads();

    const unsigned long long lower_lanes =
        lane == 0 ? 0ull : ((1ull << lane) - 1ull);
    const int rank_in_wave = __popcll(votes & lower_lanes);
    const int output_slot = scratch.chunk_base +
                            scratch.wave_offsets[wave] +
                            rank_in_wave;
    if (is_active && output_slot < max_active_experts)
        active_expert_ids[output_slot] = expert_id;

    // The next chunk reuses every shared field, so all writes must retire.
    __syncthreads();
}

/** @brief Fill unused compact-list capacity with the canonical -1 sentinel. */
__device__ __forceinline__ void rocm_pad_active_expert_list(
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
 * Descriptor readiness and active-list compaction are collective workgroup
 * operations. Keeping their state in one object lets complete descriptor
 * publication and fused verifier grouping share one exact implementation.
 */
struct RuntimePrefillPlanPublicationScratch
{
    int first_invalid_expert;
    ActiveExpertCompactionScratch active_experts;
};

/**
 * @brief Publish one expert's compact projection descriptors from a bank.
 *
 * Each lane owns exactly one expert id.  Separating descriptor publication
 * from active-list compaction lets the fused verifier transaction overlap it
 * with count/offset publication while selected decode continues to use
 * the same readiness and zero-fill contract.
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
 * @return True when the expert is locally resident and compute-ready.
 */
__device__ __forceinline__ bool rocm_publish_runtime_expert_descriptors_lane(
    const DeviceMoELayerRuntimeView *runtime,
    const DeviceMoEPlacementBankView &bank,
    uint32_t local_bit,
    int expert,
    int num_experts,
    DeviceNativeQuantisedMatrixDescView *__restrict__ gate_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ up_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ down_descs,
    llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections)
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
    if (expected_format == llaminar2::DeviceMoEWeightFormat::NativeQuantised)
    {
        gate_descs[expert] = local_ready
                                 ? desc.gate
                                 : DeviceNativeQuantisedMatrixDescView{};
        up_descs[expert] = local_ready
                               ? desc.up
                               : DeviceNativeQuantisedMatrixDescView{};
        down_descs[expert] = local_ready
                                 ? desc.down
                                 : DeviceNativeQuantisedMatrixDescView{};
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
__global__ void rocm_moe_materialize_runtime_selected_decode_descriptors_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const int *__restrict__ expert_ids,
    DeviceNativeQuantisedMatrixDescView *__restrict__ gate_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ up_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ down_descs,
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
    rocm_publish_runtime_expert_descriptors_lane(runtime, bank, local_bit, expert, num_experts,
        gate_descs, up_descs, down_descs, format, projections);
}

/**
 * @brief Publish runtime descriptors and the stable local active-expert list.
 *
 * Every lane in the workgroup must call this function. Descriptor publication
 * reads the immutable active placement bank, while active-list publication
 * consumes grouped counts authored earlier on the same stream or earlier in
 * the same fused workgroup. An active expert without a locally resident,
 * compute-ready descriptor is a fatal lifecycle violation.
 * @param runtime Original selected participant runtime and immutable placement bank.
 * @param gate_descs Exact pre-admitted gate table, overwritten over all experts.
 * @param up_descs Exact pre-admitted up table, overwritten over all experts.
 * @param down_descs Exact pre-admitted down table, overwritten over all experts.
 * @param num_experts Complete routed source interval, independent of block width.
 * @param active_expert_ids Optional stable ascending expert list; null selects descriptors only.
 * @param max_active_experts Admitted list capacity, validated when the list is present.
 * @param expected_format Original native/floating descriptor family.
 * @param expected_projections Exact original source projection-role contract.
 * @param scratch Block-owned readiness and cumulative chunk compaction state.
 */
__device__ __forceinline__ void rocm_publish_runtime_prefill_plan_block(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceNativeQuantisedMatrixDescView *__restrict__ gate_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ up_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ down_descs, int num_experts,
    int *__restrict__ active_expert_ids, int max_active_experts,
    llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections,
    RuntimePrefillPlanPublicationScratch &scratch)
{
    const bool publish_active_experts = active_expert_ids != nullptr;
    const uint32_t execution_bank = runtime_execution_bank(runtime);
    if (!runtime || !gate_descs || !up_descs || !down_descs || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || execution_bank > 1u ||
        !llaminar2::deviceMoEProjectionSetValid(expected_projections) ||
            (runtime && runtime->intermediate_shard.enabled() !=
                (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard)) ||
        (expected_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
         !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)) ||
        (publish_active_experts &&
         (!runtime->expert_counts || max_active_experts <= 0 || max_active_experts > num_experts)))
    {
        if (threadIdx.x == 0)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("prefill publication has an invalid runtime or projection contract");
        }
        return;
    }

    // Keep the admitted 256-thread shape. One ordered chunk per workgroup
    // visit covers the complete source expert interval, including upper ids.
    // Compaction owns a cumulative base across chunks, never a per-chunk list.
    if (publish_active_experts)
        rocm_initialize_active_expert_compaction(scratch.active_experts);
    for (int chunk = 0; chunk < num_experts; chunk += static_cast<int>(blockDim.x))
    {
        const int expert = chunk + static_cast<int>(threadIdx.x);
        const DeviceMoEPlacementBankView &bank = runtime_placement_banks(runtime)[execution_bank];
        const uint32_t local_bit =
            expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                    ? 0u : runtime_participant_bit(static_cast<int>(runtime->participant_id));
        const bool local_ready = rocm_publish_runtime_expert_descriptors_lane(
            runtime,
            bank, local_bit, expert, num_experts, gate_descs, up_descs, down_descs, expected_format,
            expected_projections);

        // Descriptor-only publication still visits every chunk. This branch is
        // uniform, so no lane can bypass a barrier needed by another lane.
        if (!publish_active_experts)
            continue;

        const int count = expert < num_experts ? runtime->expert_counts[expert] : 0;
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
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("prefill expert lacks admitted projection descriptors");
            }
            return;
        }

        rocm_compact_active_expert_chunk(is_active, expert, max_active_experts, active_expert_ids,
                                         scratch.active_experts);
    }
    if (!publish_active_experts)
        return;
    rocm_pad_active_expert_list(active_expert_ids, max_active_experts, scratch.active_experts);
}
/**
 * @brief Build a stable compact list from precomputed expert counts.
 *
 * One block processes ascending 256-id chunks, retaining byte identity with
 * the serial implementation while parallelizing reads, rank assignment,
 * writes, and sentinel padding.
 */
template<bool PackedOffsetSlots, bool ColumnSlots=false>
__global__ void rocm_moe_build_active_expert_list_kernel(
    const int *__restrict__ expert_counts,
    int *__restrict__ active_expert_ids,
    int num_experts,
    int max_active_experts,
    const int *__restrict__ expert_offsets,
    uint32_t *__restrict__ packed_directory,
    int total_slots)
{
    __shared__ ActiveExpertCompactionScratch scratch;
    // The existing initialization barrier orders all sentinel stores before
    // any expert publishes descriptors into its disjoint original interval.
    if constexpr (PackedOffsetSlots)
        for (int slot = threadIdx.x; slot < total_slots; slot += blockDim.x)
            packed_directory[slot] = UINT32_MAX;
    rocm_initialize_active_expert_compaction(scratch);

    for (int chunk = 0; chunk < num_experts; chunk += blockDim.x)
    {
        const int expert = chunk + static_cast<int>(threadIdx.x);
        const bool is_active =
            expert < num_experts && expert_counts[expert] > 0;
        if constexpr (PackedOffsetSlots)
        {
            if (expert < num_experts)
            {
                using Interval = std::conditional_t<ColumnSlots,
                    llaminar2::moe_grouped_directory::ColumnFloor8OffsetSlots,
                    llaminar2::moe_grouped_directory::WideFloor8OffsetSlots>;
                const auto interval = Interval::bind(expert_counts[expert], expert_offsets[expert], total_slots);
                for (int tile = 0; tile < interval.tiles(); ++tile)
                    packed_directory[interval.slot(tile)] = interval.descriptor(expert, num_experts, tile);
            }
        }
        rocm_compact_active_expert_chunk(
            is_active,
            expert,
            max_active_experts,
            active_expert_ids,
            scratch);
    }
    rocm_pad_active_expert_list(
        active_expert_ids,
        max_active_experts,
        scratch);
}

/**
 * @brief Build the route-slot to grouped-slot map for runtime grouped prefill.
 *
 * LLEP runtime grouping stores original route slots in @c grouped_token_ids
 * after it filters the batch down to this participant's assigned routes.  The
 * ordered scatter stage needs the inverse map so every token accumulates its
 * top-k choices in router order instead of in grouped expert order.  Invalid
 * scratch state traps immediately because continuing would publish numerically
 * unstable MoE output behind an apparently successful prefill.
 */
__global__ void rocm_moe_build_runtime_original_to_grouped_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int *__restrict__ original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts)
{
    const int expert = blockIdx.x;
    const uint32_t execution_bank = runtime_execution_bank(runtime);
    if (!runtime || !original_to_grouped ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        execution_bank > 1u ||
        runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots) ||
        !runtime->route_expert_ids ||
        !runtime->route_participant_ids ||
        !runtime->expert_counts ||
        !runtime->expert_offsets ||
        !runtime->grouped_token_ids)
    {
        FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("original-to-grouped route table lacks its admitted storage or geometry");
        return;
    }
    if (expert >= num_experts)
        return;

    const int count = runtime->expert_counts[expert];
    const int offset = runtime->expert_offsets[expert];
    if (count < 0 || offset < 0 || offset + count > max_slots)
    {
        FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("original-to-grouped expert span exceeds its admitted route capacity");
        return;
    }

    const int local_participant = static_cast<int>(runtime->participant_id);
    for (int row = threadIdx.x; row < count; row += blockDim.x)
    {
        const int grouped_slot = offset + row;
        const int route_slot = runtime->grouped_token_ids[grouped_slot];
        if (route_slot < 0 ||
            route_slot >= current_slots ||
            route_slot >= max_slots ||
            runtime->route_expert_ids[route_slot] != expert ||
            runtime->route_participant_ids[route_slot] != local_participant)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("original-to-grouped route maps a row to a foreign expert or participant");
            return;
        }
        original_to_grouped[route_slot] = grouped_slot;
    }
}

// =========================================================================
// Kernel 12: Scatter Tokens
//
// Scatters token indices and weights into expert-grouped arrays.
// Each thread handles one (token, slot) pair, uses atomicAdd on write_heads.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

__global__ void rocm_moe_scatter_tokens_kernel(
    const int *__restrict__ routing_indices,     // [total_slots]
    const float *__restrict__ routing_weights,   // [total_slots]
    const int *__restrict__ expert_offsets,       // [num_experts]
    int *__restrict__ write_heads,               // [num_experts] (init to 0)
    int *__restrict__ grouped_token_indices,     // [total_slots] OUT
    int *__restrict__ original_to_grouped,        // [total_slots] OUT, optional
    float *__restrict__ grouped_weights,         // [total_slots] OUT
    int total_slots, int num_experts, int top_k)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;

    int token_idx = slot / top_k;
    int expert_id = routing_indices[slot];

    if (expert_id >= 0 && expert_id < num_experts)
    {
        int write_pos = atomicAdd(&write_heads[expert_id], 1);
        int dest = expert_offsets[expert_id] + write_pos;
        grouped_token_indices[dest] = token_idx;
        if (original_to_grouped)
            original_to_grouped[slot] = dest;
        grouped_weights[dest] = routing_weights[slot];
    }
}

/**
 * @brief Shared state for the verifier-sized expert prefix scan.
 *
 * The first component publishes stable route offsets and the optional second
 * component publishes active-expert ranks.  Both are scanned in ascending wave
 * order, preserving the serial expert-id result without quadratic LDS reads.
 */
struct RuntimeSmallGroupScanScratch
{
    int count_wave_totals[kDeviceMoEMaxExperts / 32];
    int count_wave_prefixes[kDeviceMoEMaxExperts / 32];
    int active_wave_totals[kDeviceMoEMaxExperts / 32];
    int active_wave_prefixes[kDeviceMoEMaxExperts / 32];
    int total_active;
    int first_invalid_expert;
};

/**
 * @brief Compute deterministic workgroup-wide count and active-list prefixes.
 *
 * Every lane must participate. Each wave scans its own values with shuffle
 * instructions, then wave zero scans the per-wave totals in ascending order.
 * Integer addition is exact, so this fixed two-level tree reproduces the same
 * offsets and active ranks as serial expert traversal on wave32 and wave64.
 *
 * @tparam ScanActive Whether to scan the active-expert predicate as well.
 * @param count Number of local routes assigned to this lane's expert.
 * @param scratch Workgroup-owned scan state.
 * @param count_prefix Receives the exclusive route-count prefix.
 * @param active_prefix Receives the exclusive active-expert prefix.
 * @param total_active Receives the total number of active experts.
 */
template <bool ScanActive>
__device__ __forceinline__ void rocm_scan_runtime_small_group_experts(
    int count,
    RuntimeSmallGroupScanScratch &scratch,
    int &count_prefix,
    int &active_prefix,
    int &total_active)
{
    const int lane = static_cast<int>(threadIdx.x) % warpSize;
    const int wave = static_cast<int>(threadIdx.x) / warpSize;
    const int wave_count =
        (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
    int count_inclusive = count;
    int active_inclusive = count > 0 ? 1 : 0;

    for (int delta = 1; delta < warpSize; delta <<= 1)
    {
        const int prior_count = __shfl_up(count_inclusive, delta, warpSize);
        if (lane >= delta)
            count_inclusive += prior_count;
        if constexpr (ScanActive)
        {
            const int prior_active = __shfl_up(active_inclusive, delta, warpSize);
            if (lane >= delta)
                active_inclusive += prior_active;
        }
    }

    if (lane == warpSize - 1)
    {
        scratch.count_wave_totals[wave] = count_inclusive;
        if constexpr (ScanActive)
            scratch.active_wave_totals[wave] = active_inclusive;
    }
    __syncthreads();

    if (wave == 0)
    {
        const int wave_count_value =
            lane < wave_count ? scratch.count_wave_totals[lane] : 0;
        const int wave_active_value =
            ScanActive && lane < wave_count
                ? scratch.active_wave_totals[lane]
                : 0;
        int count_wave_inclusive = wave_count_value;
        int active_wave_inclusive = wave_active_value;
        for (int delta = 1; delta < warpSize; delta <<= 1)
        {
            const int prior_count =
                __shfl_up(count_wave_inclusive, delta, warpSize);
            if (lane >= delta)
                count_wave_inclusive += prior_count;
            if constexpr (ScanActive)
            {
                const int prior_active =
                    __shfl_up(active_wave_inclusive, delta, warpSize);
                if (lane >= delta)
                    active_wave_inclusive += prior_active;
            }
        }

        if (lane < wave_count)
        {
            scratch.count_wave_prefixes[lane] =
                count_wave_inclusive - wave_count_value;
            if constexpr (ScanActive)
            {
                scratch.active_wave_prefixes[lane] =
                    active_wave_inclusive - wave_active_value;
            }
        }
        if constexpr (ScanActive)
        {
            if (lane == wave_count - 1)
                scratch.total_active = active_wave_inclusive;
        }
    }
    __syncthreads();

    count_prefix = scratch.count_wave_prefixes[wave] +
                   count_inclusive - count;
    if constexpr (ScanActive)
    {
        const int active = count > 0 ? 1 : 0;
        active_prefix = scratch.active_wave_prefixes[wave] +
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
constexpr int kSmallFloatIndependentPrefixMaxExperts = 32;

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
 * @param top_k Positive original route width, at most 16.
 * @param max_active_experts Declared active-list extent, at most min(total_slots, num_experts).
 */
template <SmallFloatGroupPrefixPolicy PrefixPolicy>
__global__ void rocm_moe_group_tokens_small_float_kernel(
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
        /*
         * The small-M planner feeds two later grouped verifier contracts:
         * ordered scatter needs route-slot -> grouped-slot, while parallel
         * down publication needs route-slot -> expert-id.  Keep both maps
         * authored by this single grouping kernel so verifier replay cannot
         * accidentally consume stale scratch from a previous batch.
         */
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
        // The selected domain has at most 32 experts. Only those lanes write
        // the 32-entry scratch; every block lane still enters the barrier. The
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
        rocm_scan_runtime_small_group_experts<true>(
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
 * @brief Publish singleton shared-expert routes from the device-owned live prefix.
 *
 * The whole captured capacity is initialized on every replay. Inactive inverse
 * entries stay invalid, so both compact and tiled consumers skip padded work
 * even when a preceding replay admitted a larger MTP depth.
 */
__global__ void rocm_moe_prepare_shared_expert_group_kernel(
    int *__restrict__ expert_offsets,
    int *__restrict__ expert_counts,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
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
        grouped_weights[idx] = idx < active_rows ? 1.0f : 0.0f;
    }
}

__global__ void rocm_moe_prefill_group_clear_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int *__restrict__ original_to_grouped,
    int max_slots, int num_experts, int clear_route_metadata)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

    if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
    {
        runtime->expert_counts[idx] = 0;
        runtime->expert_offsets[idx] = 0;
    }

    if (idx < max_slots)
    {
        if (clear_route_metadata && runtime->route_expert_ids) runtime->route_expert_ids[idx] = -1;
        if (clear_route_metadata && runtime->route_weights) runtime->route_weights[idx] = 0.0f;
        if (clear_route_metadata && runtime->route_participant_ids) runtime->route_participant_ids[idx] = -1;
        if (runtime->grouped_token_ids) runtime->grouped_token_ids[idx] = 0;
        if (runtime->grouped_route_weights) runtime->grouped_route_weights[idx] = 0.0f;
        if (original_to_grouped) original_to_grouped[idx] = -1;
    }
}

/**
 * @brief Return whether a runtime-prefill route can be computed by this shard.
 *
 * StaticOwner LocalTP verifier prefill is intentionally a local partial.  The
 * router still produces global expert ids, but this participant must group
 * only experts whose active placement-bank descriptor is resident and marked
 * LocalCompute.  LLEP planning does not use this predicate during its first
 * grouping pass because it needs global route counts before assigning rows.
 */
__device__ __forceinline__ bool rocm_moe_prefill_static_local_ready(
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
 * @brief Publish one verifier-sized runtime grouping transaction.
 *
 * One 256/512-thread workgroup owns every route slot and every expert. The router
 * specialization converts FP32 router outputs into the device runtime ledger;
 * the regroup specialization consumes participant assignments already written
 * by the LLEP planner. Both derive exact counts, ascending expert offsets,
 * stable grouped rows, and the inverse route map without an order-dependent
 * floating-point atomic or a serial coordinator. Bounded integer histogram
 * additions are order-independent; stable route placement remains explicit.
 *
 * Stable route order is part of grouped-verifier correctness. Each route lane
 * counts earlier local routes to the same expert, reproducing serial traversal
 * exactly, and every output location has one deterministic writer.
 *
 * @tparam PublishRouterInputs True for initial router publication; false for
 *         regrouping after device-resident participant assignment.
 * @tparam PublishCompletePlan True when this launch also owns descriptor and
 *         active-expert publication for the immediately following compute.
 */
template <bool PublishRouterInputs, bool PublishCompletePlan = false>
// The public launch and validation below require exactly this workgroup size.
// Leaving HIP's default 1024-thread bound makes LLVM reserve registers for a
// launch that is invalid here, forcing otherwise register-resident values into
// private scratch. Publish the real contract to register allocation as well.
__global__ __launch_bounds__(kDeviceMoEMaxExperts)
void rocm_moe_prefill_group_small_runtime_kernel(
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
    DeviceNativeQuantisedMatrixDescView *__restrict__ gate_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ up_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ down_descs,
    int *__restrict__ active_expert_ids,
    int max_active_experts,
    llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections)
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
        !runtime ||
        blockDim.x != llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts) ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        max_slots > kDeviceMoERuntimeSmallGroupMaxSlots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
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
         (!original_to_grouped || !gate_descs || !up_descs || !down_descs ||
          !active_expert_ids || max_active_experts <= 0 ||
          max_active_experts > num_experts ||
          (expected_format !=
               llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
           !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)))) ||
        (retain_routes_for_deferred_commit != 0 &&
         (!runtime->deferred_verifier_route_expert_ids ||
          !runtime->deferred_verifier_route_participant_ids ||
          runtime->deferred_verifier_route_capacity <
              static_cast<uint32_t>(current_slots)));
    if (invalid_contract)
    {
        if (tid == 0)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("grouped prefill lacks its admitted runtime contract");
        }
        return;
    }

    // Reuse the eventual offset bank as a temporary integer histogram. At
    // most max_slots (<= 256) increments reach any counter, so every schedule
    // produces exactly the same count with no overflow or FP reassociation.
    const int local_participant = static_cast<int>(runtime->participant_id);
    if (tid < num_experts)
        shared_expert_offsets[tid] = 0;
    __syncthreads();

    if (tid < max_slots)
    {
        runtime->grouped_token_ids[tid] = 0;
        runtime->grouped_route_weights[tid] = 0.0f;
        if (original_to_grouped)
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
                    /* StaticOwner publishes the same authoritative owner on
                     * every shard.  A remote owner is not an inactive route:
                     * mapped sparse reduction needs that id to select its peer
                     * lane even though this shard performs no local GEMM. */
                    const DeviceMoEPlacementBankView &bank =
                        runtime_placement_banks(runtime)[execution_bank];
                    participant_id = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                            ? runtime_intermediate_shard_row(runtime, bank, expert_id).participant
                            : runtime_expert_owner(bank, expert_id);
                    if (participant_id !=
                            static_cast<int>(runtime->participant_id) ||
                        !rocm_moe_prefill_static_local_ready(
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
     * A complete ordinary-prefill plan is the only serial-visible demand
     * boundary here. Route-only grouping still awaits participant assignment,
     * and verifier routes still await accepted-prefix publication. Keeping the
     * update in this fused small-plan launch adds no graph node.
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

    // All route increments are now visible. The scan's workgroup barriers
    // consume these counts before replacing the bank with exclusive offsets.
    const int count = tid < num_experts ? shared_expert_offsets[tid] : 0;

    if constexpr (PublishCompletePlan)
    {
        if (tid == 0)
            scan_scratch.first_invalid_expert = num_experts;
    }
    int expert_offset = 0;
    int active_rank = 0;
    int total_active = 0;
    rocm_scan_runtime_small_group_experts<PublishCompletePlan>(
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
             * The compact plan's descriptor validity set is exactly its
             * active-expert list. Route-owned kernels index descriptors only
             * through validated active routes, while expert-row kernels walk
             * active_expert_ids. Publishing locally resident but inactive
             * experts therefore wrote three large descriptor records per lane
             * that no consumer could observe. At verifier depth four this
             * meant copying all 256 experts even though at most 32 routes can
             * be active.
             *
             * Leave inactive slots outside the transaction instead of
             * zero-filling or refreshing them. Every active slot is still
             * published from the selected immutable bank and validated before
             * the workgroup releases the plan, so a stale inactive descriptor
             * cannot become reachable through a valid grouped transaction.
             */
            bool local_ready = true;
            if (is_active)
            {
                local_ready = rocm_publish_runtime_expert_descriptors_lane(
                    runtime,
                    bank,
                    local_bit,
                    tid,
                    num_experts,
                    gate_descs,
                    up_descs,
                    down_descs,
                    expected_format,
                    expected_projections);
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
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("grouped prefill expert lacks admitted projection descriptors");
            }
            return;
        }
    }

    // Retain the independently computed destination across publication. The
    // validation below checks the published bijection; it need not perform
    // the same quadratic earlier-route search for a second time.
    int expected_grouped_slot = -1;
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
            expected_grouped_slot =
                shared_expert_offsets[expert_id] + stable_local_rank;
            runtime->grouped_token_ids[expected_grouped_slot] = tid;
            runtime->grouped_route_weights[expected_grouped_slot] =
                shared_route_weights[tid];
            if (original_to_grouped)
                original_to_grouped[tid] = expected_grouped_slot;
        }
    }
    __syncthreads();

    /*
     * Validate the grouped/original bijection inside the publication launch.
     * This retains the former standalone inverse-map fatal contract without a
     * second kernel or any host observation.
     */
    if (tid < current_slots)
    {
        const int expert_id = shared_route_experts[tid];
        const int participant_id = shared_route_participants[tid];
        const bool should_be_local =
            expert_id >= 0 && expert_id < num_experts &&
            participant_id == local_participant;
        const int published_grouped_slot = original_to_grouped
                                               ? original_to_grouped[tid]
                                               : expected_grouped_slot;
        const bool valid_mapping =
            should_be_local
                ? published_grouped_slot == expected_grouped_slot &&
                      expected_grouped_slot >= 0 &&
                      expected_grouped_slot < max_slots &&
                      runtime->grouped_token_ids[expected_grouped_slot] == tid &&
                      runtime->route_expert_ids[tid] == expert_id &&
                      runtime->route_participant_ids[tid] == local_participant
                : !original_to_grouped || published_grouped_slot == -1;
        if (!valid_mapping)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("grouped prefill route map disagrees with its selected expert");
            return;
        }
    }

}

/** @copydoc llaminar2::queryROCmRuntimeGroupResources */
std::optional<llaminar2::ROCmRuntimeGroupResources>
llaminar2::queryROCmRuntimeGroupResources(ROCmRuntimeGroupVariant variant, int experts)
{
    // Resolve the compiled function, not a source-level estimate: a compiler
    // change can introduce spills even when its launch geometry is unchanged.
    const int threads = moe_runtime_abi::expertGroupingThreads(experts);
    if (threads == 0) return std::nullopt;
    const void *function = nullptr;
    switch (variant)
    {
    case ROCmRuntimeGroupVariant::RouterRoutes:
        function = reinterpret_cast<const void *>(rocm_moe_prefill_group_small_runtime_kernel<true, false>);
        break;
    case ROCmRuntimeGroupVariant::AssignedRoutes:
        function = reinterpret_cast<const void *>(rocm_moe_prefill_group_small_runtime_kernel<false, false>);
        break;
    case ROCmRuntimeGroupVariant::RouterCompletePlan:
        function = reinterpret_cast<const void *>(rocm_moe_prefill_group_small_runtime_kernel<true, true>);
        break;
    case ROCmRuntimeGroupVariant::AssignedCompletePlan:
        function = reinterpret_cast<const void *>(rocm_moe_prefill_group_small_runtime_kernel<false, true>);
        break;
    default:
        return std::nullopt;
    }
    hipFuncAttributes attributes{};
    int resident_blocks = 0;
    if (hipFuncGetAttributes(&attributes, function) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, function, threads, 0) != hipSuccess)
        return std::nullopt;
    return ROCmRuntimeGroupResources{
        .registers_per_thread = attributes.numRegs,
        .private_bytes_per_thread = attributes.localSizeBytes,
        .shared_bytes_per_block = attributes.sharedSizeBytes,
        .launch_threads = threads,
        .maximum_threads_per_block = attributes.maxThreadsPerBlock,
        .resident_blocks_per_multiprocessor = resident_blocks,
    };
}

__global__ void rocm_moe_prefill_group_cast_count_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    int current_slots, int max_slots, int num_experts,
    int filter_to_local_runtime_experts,
    int retain_routes_for_deferred_commit,
    llaminar2::DeviceMoEProjectionSet expected_projections)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_weights || !runtime->route_participant_ids) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

        if (runtime->intermediate_shard.enabled() !=
            (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard))
        {
            fail_fast_incomplete_llep_transfer();
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
             * Keep the global StaticOwner assignment distinct from this
             * shard's execution filter.  Remote owners retain their identity
             * while a zero local weight excludes them from grouped compute.
             */
            const uint32_t execution_bank = runtime_execution_bank(runtime);
            const DeviceMoEPlacementBankView &bank =
                runtime_placement_banks(runtime)[execution_bank];
            participant_id = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                            ? runtime_intermediate_shard_row(runtime, bank, expert_id).participant
                            : runtime_expert_owner(bank, expert_id);
            if (participant_id !=
                    static_cast<int>(runtime->participant_id) ||
                !rocm_moe_prefill_static_local_ready(
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
     * Counts are exact integers, so atomic arrival order cannot alter either
     * their value or the stable route order constructed by the later scatter.
     * Folding this publication into conversion removes a full graph node from
     * scalable initial grouping.
     */
    if (expert_id >= 0 &&
        expert_id < num_experts &&
        participant_id == static_cast<int>(runtime->participant_id))
    {
        atomicAdd(&runtime->expert_counts[expert_id], 1);
    }
}

__global__ void rocm_moe_prefill_group_count_assigned_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts,
    int retain_routes_for_deferred_commit)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= max_slots || slot >= current_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_participant_ids || !runtime->expert_counts) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

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
        atomicAdd(&runtime->expert_counts[expert_id], 1);
    }
}

/**
 * @brief Commit accepted grouped-verifier rows to persistent routing history.
 *
 * The verifier graph snapshots route ids and final participant assignments in
 * this layer's persistent ledger before another layer reuses transient route
 * scratch. Acceptance metadata arrives later on the publication stream. This
 * kernel joins those two device-owned records and excludes every rejected or
 * padded row from the maintenance histograms.
 */
__global__ void rocm_moe_commit_grouped_verifier_histograms_kernel(
    void *__restrict__ runtime_layers,
    size_t layer_stride_bytes,
    const int32_t *__restrict__ accepted_state_counts,
    const int32_t *__restrict__ publication_ok_flags,
    int request_count,
    int rows_per_request,
    int total_rows,
    int num_experts,
    int top_k)
{
    auto *const runtime = runtime_layers != nullptr
        ? reinterpret_cast<DeviceMoELayerRuntimeView *>(
              static_cast<uint8_t *>(runtime_layers) +
              static_cast<size_t>(blockIdx.y) * layer_stride_bytes)
        : nullptr;
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
    if (slot >= total_slots) return;

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

    const int expert_id = runtime->deferred_verifier_route_expert_ids[slot];
    if (expert_id < 0 || expert_id >= num_experts) return;
    if (!runtime_histogram_admits_rows(*runtime)) return;

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
 * @brief Publish exact expert prefixes and stable route order cooperatively.
 * @param runtime Authoritative device route/count buffers, already published.
 * @param original_to_grouped Optional inverse map indexed by original route slot.
 * @param current_slots Slots visible in this grouping transaction.
 * @param max_slots Admitted capacity; padded outputs are owned by prior clear.
 * @param num_experts Number of independent expert blocks.
 * @param top_k Router fanout retained by the common launch ABI.
 */
__global__ void rocm_moe_prefill_group_scatter_deterministic_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int *__restrict__ original_to_grouped,
    int current_slots, int max_slots, int num_experts, int top_k)
{
    const int expert = blockIdx.x;
    if (!runtime || expert >= num_experts) return;
    if (!runtime->route_expert_ids || !runtime->route_weights ||
        !runtime->route_participant_ids ||
        !runtime->expert_offsets || !runtime->expert_counts ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    __shared__ llaminar2::device_integer_scan::Scratch<kGateLogitsGroupedThreads> scan;
    int preceding_count = 0;
    for (int preceding_expert = static_cast<int>(threadIdx.x);
         preceding_expert < expert; preceding_expert += kGateLogitsGroupedThreads)
        preceding_count += runtime->expert_counts[preceding_expert];
    const int prefix = llaminar2::device_integer_scan::exclusive(preceding_count, scan).total;
    const int count = runtime->expert_counts[expert];
    if (prefix < 0 || count < 0 || prefix + count > max_slots)
    {
        if (threadIdx.x == 0)
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("prefill scan span exceeds its admitted route capacity");
        }
        return;
    }
    if (threadIdx.x == 0)
        runtime->expert_offsets[expert] = prefix;
    // Keep the offset for empty experts, then skip their route-table walk.
    // Count is immutable and identical for every lane in this workgroup.
    if (count == 0)
        return;
    const int expert_end = prefix + count;
    const int participant_id = static_cast<int>(runtime->participant_id);
    int running_count = 0;

    // The prior clear publishes padded output. Only current slots can satisfy
    // the route predicate, so unused capture capacity needs no scan here.
    for (int chunk_start = 0; chunk_start < current_slots; chunk_start += kGateLogitsGroupedThreads)
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
                if (original_to_grouped)
                    original_to_grouped[slot] = dest;
            }
        }
        // The shared helper's final barrier protects the next chunk's reuse;
        // every lane carries the same total without a serial flag loop.
        running_count += local.total;
    }
}

__global__ void rocm_moe_prefill_llep_count_expert_routes_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= current_slots || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->expert_counts) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    const int expert_id = runtime->route_expert_ids[slot];
    if (expert_id >= 0 && expert_id < num_experts)
        atomicAdd(&runtime->expert_counts[expert_id], 1);
}

__global__ void rocm_moe_prefill_llep_clear_current_batch_plan_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

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
 * One ROCm lane owns one token row and keeps that row's participant loads in
 * registers. Route slots are consumed in their original router order.
 * Equal-load ties use the row's absolute logical position and route identity,
 * exactly like serial decode. Persistent demand histograms remain maintenance
 * evidence only, so rejected speculative rows cannot perturb a future
 * committed row's floating-point partition.
 *
 * The launch is allocation-free, atomic-free, transfer-free, and
 * graph-capturable. Every row is independent, so the GPU executes all rows
 * concurrently without imposing an artificial M limit.
 */
__global__ void rocm_moe_prefill_llep_assign_resident_rows_logical_position_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    const int32_t *__restrict__ absolute_position_ids,
    const int32_t *__restrict__ active_row_count,
    int current_slots, int max_slots,
    int num_experts, int top_k)
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
     * Serial decode's first pass publishes every non-replicated route to its
     * owner and charges that owner before any replicated route is selected.
     * Preserve that two-pass ordering even when the single-owner route appears
     * later in top-k order.
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
        if ((resident_mask & runtime_participant_bit(owner)) == 0u)
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

__global__ void rocm_moe_prefill_llep_plan_current_batch_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots,
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
    using PlannerStatus =
        llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus;
    __shared__ uint64_t planner_status_storage[
        (sizeof(PlannerStatus) + sizeof(uint64_t) - 1u) / sizeof(uint64_t)];
    auto &planner_status =
        *reinterpret_cast<PlannerStatus *>(planner_status_storage);

    if (!runtime) return;

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
    rebalance_sort_llep_experts_parallel(
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

__device__ __forceinline__ int rocm_moe_prefill_llep_default_owner_participant(
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

__device__ __forceinline__ bool rocm_moe_prefill_llep_local_expert_ready(
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

__device__ __forceinline__ bool rocm_moe_prefill_llep_assignment_ready(
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
        return false;
    }

    if (assigned_participant != static_cast<int32_t>(runtime->participant_id))
        return true;

    const uint32_t assigned_bit = runtime_participant_bit(assigned_participant);
    const int owner_participant =
        rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
    const uint32_t resident_mask =
        llaminar2::least_loaded_ep::normalizeResidentParticipantMask(
            active_bank.resident_participant_mask[expert],
            owner_participant,
            runtime->participant_id,
            runtime->participant_count);
    if ((resident_mask & assigned_bit) == 0u)
    {
        return false;
    }

    if (assigned_participant == static_cast<int32_t>(runtime->participant_id) &&
        !rocm_moe_prefill_llep_local_expert_ready(runtime, active_bank, expert))
    {
        return false;
    }
    return true;
}

// All three bridges use the same 256-lane expert-span workgroup. An accurate
// bound also keeps the fatal diagnostic branch from spilling live registers.
__global__ __launch_bounds__(256) void rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel(
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
            fail_fast_incomplete_llep_transfer();
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
             * Transfer and apply status are request-wide publications observed
             * identically by every expert block.  One designated witness traps on
             * the original status records. All other lanes return because
             * the HIP context is already irrecoverably failed.
             */
            if (expert == 0 && threadIdx.x == 0)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER("prefill route assignment requires a complete transfer and apply receipt");
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
     * One exact lane publishes durable proof from the production span
     * consumer. Stream ordering makes the write visible to maintenance
     * preflight, and a later assignment failure is fatal rather than tolerated.
     * This keeps resident-row evidence deterministic and atomic-free.
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
            rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
        if (assigned_participant < 0)
            return;
        const DeviceMoEPlacementBankView &active_bank =
            runtime_placement_banks(runtime)[execution_bank];

        for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
        {
            if (!rocm_moe_prefill_llep_assignment_ready(
                    runtime, active_bank, expert, row, assigned_participant))
            {
                fail_fast_incomplete_llep_transfer();
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
                rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
            if (assigned_participant < 0)
                return;
            const DeviceMoEPlacementBankView &active_bank =
                runtime_placement_banks(runtime)[execution_bank];

            for (int row = threadIdx.x; row < expert_count; row += blockDim.x)
            {
                if (!rocm_moe_prefill_llep_assignment_ready(
                        runtime, active_bank, expert, row, assigned_participant))
                {
                    fail_fast_incomplete_llep_transfer();
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
                rocm_moe_prefill_llep_default_owner_participant(runtime, expert);
            if (assigned_participant < 0)
                continue;
        }
        const DeviceMoEPlacementBankView &active_bank =
            runtime_placement_banks(runtime)[execution_bank];
        if (!rocm_moe_prefill_llep_assignment_ready(
                runtime, active_bank, expert, row, assigned_participant))
        {
            fail_fast_incomplete_llep_transfer();
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
 * @brief Publish compact runtime descriptors and optional active expert ids.
 *
 * Grouped prefill needs both products from the same complete placement bank.
 * Producing them in ordered 256-id chunks within one block removes a redundant launch;
 * full-domain descriptor diagnostics omit the active-list destination. Decode
 * instead publishes only its selected packet through the shared lane helper.
 * @param runtime Actual participant runtime with its retained selected placement bank.
 * @param gate_descs Full original gate descriptor output.
 * @param up_descs Full original up descriptor output.
 * @param down_descs Full original down descriptor output.
 * @param num_experts Complete source expert count, not the launch block width.
 * @param active_expert_ids Optional ordered active-list output.
 * @param max_active_experts Admitted list capacity when present.
 * @param current_slots Actual live route count for optional histogram publication.
 * @param publish_prefill_histogram Whether this transaction owns histogram publication.
 * @param expected_format Original source native/floating descriptor family.
 * @param expected_projections Exact admitted projection-role set.
 */
__global__ __launch_bounds__(kActiveExpertThreads) void rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    DeviceNativeQuantisedMatrixDescView *__restrict__ gate_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ up_descs,
    DeviceNativeQuantisedMatrixDescView *__restrict__ down_descs,
    int num_experts,
    int *__restrict__ active_expert_ids,
    int max_active_experts,
    int current_slots,
    int publish_prefill_histogram,
    llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections)
{
    __shared__ RuntimePrefillPlanPublicationScratch scratch;
    rocm_publish_runtime_prefill_plan_block(
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
     * Scalable grouping already requires this descriptor publication. Count
     * the final route ledger here so route-only conversion remains free of
     * persistent history effects and no extra launch enters the hot path.
     * Aggregate per-expert hits in LDS first when num_experts fits in the
     * 1024-entry local table (matching CUDA CUDAMoEGroupingKernels.cu) so 5k+
     * route slots flush with at most one global/mapped 64-bit atomic per active
     * expert instead of one per route slot.
     */
    if (publish_prefill_histogram != 0 &&
        runtime_histogram_admits_rows(*runtime))
    {
        uint64_t *const sel_hist = runtime_selected_histogram(*runtime, 1u);
        uint64_t *const loc_hist = runtime_local_histogram(*runtime, 1u);
        const int local_pid = static_cast<int>(runtime->participant_id);
        constexpr int kLdsHistCap = 1024;
        if (num_experts <= kLdsHistCap)
        {
            __shared__ uint32_t s_sel[kLdsHistCap];
            __shared__ uint32_t s_loc[kLdsHistCap];
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
                const int participant_id = runtime->route_participant_ids[slot];
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
                const int participant_id = runtime->route_participant_ids[slot];
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
 * The compact floating views occupy the beginning of the existing graph-owned
 * descriptor slots, preserving capture identity and arena capacity. Remote
 * entries are overwritten with zeros so a participant cannot consume stale
 * residency after an asynchronous placement epoch flips.
 * @param runtime Actual selected immutable placement bank and participant.
 * @param gate_descs Complete admitted gate descriptor output.
 * @param up_descs Complete admitted up descriptor output.
 * @param down_descs Complete admitted down descriptor output.
 * @param num_experts Original full source domain, independent of the fixed block width.
 * @param expected_format Exact original floating scalar family, never inferred from bytes.
 */
__global__ void rocm_moe_materialize_runtime_floating_descriptor_tables_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
    llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
    llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs, int num_experts,
    llaminar2::DeviceMoEWeightFormat expected_format,
    llaminar2::DeviceMoEProjectionSet expected_projections)
{
    const uint32_t execution_bank = runtime_execution_bank(runtime);
    if (!runtime || !gate_descs || !up_descs || !down_descs || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || execution_bank > 1u ||
        !llaminar2::deviceMoEWeightFormatIsFloating(expected_format) ||
        runtime->intermediate_shard.enabled() != (expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard))
    {
        if (threadIdx.x == 0)
            fail_fast_incomplete_llep_transfer();
        return;
    }
    // A fixed block publishes the full original table, including absent/remote
    // zeros, so placement changes cannot retain a stale upper-half descriptor.
    for (int expert = static_cast<int>(threadIdx.x); expert < num_experts;
         expert += static_cast<int>(blockDim.x))
    {

        const DeviceMoEPlacementBankView &bank = runtime_placement_banks(runtime)[execution_bank];
        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        const uint32_t local_bit =
            runtime->intermediate_shard.enabled() ? 0u : runtime_participant_bit(static_cast<int>(runtime->participant_id));
        const bool local_ready = expected_projections == llaminar2::DeviceMoEProjectionSet::IntermediateShard
            ? runtime_intermediate_shard_row(runtime, bank, expert).disposition ==
                  llaminar2::MoEIntermediateShardRowDisposition::Local && desc.weight_format == expected_format
            : bank.local_compute_mask[expert] != 0u &&
                                 (bank.resident_participant_mask[expert] & local_bit) != 0u &&
                                 desc.local_slot >= 0 && desc.weight_format == expected_format &&
                                 desc.floating_gate.valid() && desc.floating_up.valid() &&
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
/** @brief Load one contiguous expert weight using the published format tag. */
__device__ __forceinline__ float rocm_load_floating_moe_weight(
    const void *__restrict__ data,
    size_t index,
    llaminar2::DeviceMoEWeightFormat format)
{
    if (format == llaminar2::DeviceMoEWeightFormat::FP32)
        return static_cast<const float *>(data)[index];
    if (format == llaminar2::DeviceMoEWeightFormat::BF16)
        return static_cast<float>(static_cast<const hip_bfloat16 *>(data)[index]);
    return __half2float(static_cast<const __half *>(data)[index]);
}

/**
 * @brief Fixed-tree gate/up decode for a device-selected floating expert.
 *
 * One workgroup owns one `(route, output-column)` pair. All floating formats
 * use the same strided K walk and 256-thread reduction tree, keeping the
 * grouped verifier and ordinary M=1 decode arithmetic schedule identical.
 */
__global__ __launch_bounds__(kFloatingMoEThreads)
void rocm_grouped_floating_gate_up_decode_kernel(
    const float *__restrict__ hidden,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
    const int *__restrict__ expert_ids,
    float *const *__restrict__ gate_outputs,
    float *const *__restrict__ up_outputs,
    int num_active,
    int intermediate,
    int d_model,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format)
{
    __shared__ float gate_sums[kFloatingMoEThreads];
    __shared__ float up_sums[kFloatingMoEThreads];
    const int n = static_cast<int>(blockIdx.x);
    const int route = rocm_logical_row_from_grid_yz();
    if (route >= num_active || n >= intermediate)
        return;

    const int expert = expert_ids[route];
    const bool valid = expert >= 0 && expert < num_experts &&
                       gate_outputs[route] && up_outputs[route];
    llaminar2::DeviceMoEFloatingMatrixDesc gate{};
    llaminar2::DeviceMoEFloatingMatrixDesc up{};
    if (valid)
    {
        gate = gate_descs[expert];
        up = up_descs[expert];
    }
    const bool ready = valid && gate.data && up.data &&
                       gate.n == intermediate && gate.k == d_model &&
                       up.n == intermediate && up.k == d_model;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (ready)
    {
        const size_t weight_base =
            static_cast<size_t>(n) * static_cast<size_t>(d_model);
        for (int k = static_cast<int>(threadIdx.x);
             k < d_model;
             k += kFloatingMoEThreads)
        {
            const float activation = hidden[k];
            gate_sum = fmaf(
                activation,
                rocm_load_floating_moe_weight(
                    gate.data, weight_base + static_cast<size_t>(k), format),
                gate_sum);
            up_sum = fmaf(
                activation,
                rocm_load_floating_moe_weight(
                    up.data, weight_base + static_cast<size_t>(k), format),
                up_sum);
        }
    }
    gate_sums[threadIdx.x] = gate_sum;
    up_sums[threadIdx.x] = up_sum;
    __syncthreads();
    for (int stride = kFloatingMoEThreads / 2; stride > 0; stride >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < stride)
        {
            gate_sums[threadIdx.x] = rocm_native_quantised_accumulate_rn(
                gate_sums[threadIdx.x], gate_sums[threadIdx.x + stride]);
            up_sums[threadIdx.x] = rocm_native_quantised_accumulate_rn(
                up_sums[threadIdx.x], up_sums[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && valid)
    {
        gate_outputs[route][n] = ready ? gate_sums[0] : 0.0f;
        up_outputs[route][n] = ready ? up_sums[0] : 0.0f;
    }
}

/**
 * @brief Fixed-tree floating SwiGLU/down decode in canonical router order.
 *
 * Direct output walks all routes in order. Collective publication launches
 * one workgroup for each route and column and writes the already-rounded
 * weighted contribution for the shared canonical reduction.
 */
__global__ __launch_bounds__(kFloatingMoEThreads)
void rocm_grouped_floating_swiglu_down_decode_kernel(
    const float *const *__restrict__ gate_rows,
    const float *const *__restrict__ up_rows,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs,
    const int *__restrict__ expert_ids,
    const float *__restrict__ route_weights,
    float *__restrict__ output,
    float *__restrict__ canonical_routes,
    int num_active,
    int d_model,
    int intermediate,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format)
{
    __shared__ float sums[kFloatingMoEThreads];
    const int n = static_cast<int>(blockIdx.x);
    const int publication_row = rocm_logical_row_from_grid_yz();
    const bool canonical = canonical_routes != nullptr;
    if (n >= d_model || (canonical && publication_row >= num_active))
        return;

    const int route_begin = canonical ? publication_row : 0;
    const int route_end = canonical ? publication_row + 1 : num_active;
    float output_sum = 0.0f;
    for (int route = route_begin; route < route_end; ++route)
    {
        const int expert = expert_ids[route];
        const bool valid = expert >= 0 && expert < num_experts &&
                           gate_rows[route] && up_rows[route];
        llaminar2::DeviceMoEFloatingMatrixDesc down{};
        if (valid)
            down = down_descs[expert];
        const bool ready = valid && down.data &&
                           down.n == d_model && down.k == intermediate;
        float sum = 0.0f;
        if (ready)
        {
            const size_t weight_base =
                static_cast<size_t>(n) * static_cast<size_t>(intermediate);
            for (int k = static_cast<int>(threadIdx.x);
                 k < intermediate;
                 k += kFloatingMoEThreads)
            {
                const float gate = gate_rows[route][k];
                const float up = up_rows[route][k];
                const float swiglu =
                    llaminar2::rocm::decode_equivalent::swigluValue(
                        gate, up);
                sum = fmaf(
                    swiglu,
                    rocm_load_floating_moe_weight(
                        down.data,
                        weight_base + static_cast<size_t>(k),
                        format),
                    sum);
            }
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = kFloatingMoEThreads / 2; stride > 0; stride >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < stride)
            {
                sums[threadIdx.x] = rocm_native_quantised_accumulate_rn(
                    sums[threadIdx.x], sums[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            const float projected = ready ? sums[0] : 0.0f;
            output_sum = route_weights
                ? rocm_native_quantised_accumulate_rn(output_sum, ready
                    ? rocm_native_quantised_weight_route_rn(route_weights[route], projected) : 0.0f)
                : projected;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        if (canonical)
        {
            canonical_routes[
                static_cast<size_t>(publication_row) *
                    static_cast<size_t>(d_model) +
                static_cast<size_t>(n)] = output_sum;
        }
        else
        {
            output[n] = output_sum;
        }
    }
}

/**
 * @brief Project four floating columns per block with the exact serial tree.
 *
 * Each native wave64 owns one column; its four register accumulators
 * retain the 256 logical K partitions. Whole-subgroup tails and absent routes
 * are safe because no block barrier or cross-column state is used.
 */
__global__ __launch_bounds__(llaminar2::device_moe_floating_warp::threads)
void rocm_grouped_floating_gate_up_swiglu_prefill_kernel(
    const float *__restrict__ hidden,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
    const int *__restrict__ original_to_grouped,
    const int *__restrict__ original_expert_ids,
    float *__restrict__ grouped_swiglu,
    int total_slots,
    int top_k,
    int intermediate,
    int d_model,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format)
{
    namespace warp = llaminar2::device_moe_floating_warp;
    const int lane = threadIdx.x % warp::width;
    const int n = blockIdx.x * warp::columns_per_block + threadIdx.x / warp::width;
    const int original_slot = rocm_logical_row_from_grid_yz();
    if (original_slot >= total_slots || n >= intermediate)
        return;
    const int grouped_slot = original_to_grouped[original_slot];
    const int expert = original_expert_ids ? original_expert_ids[original_slot] : 0;
    if (grouped_slot < 0 || expert < 0 || expert >= num_experts)
        return;

    const auto gate = gate_descs[expert];
    const auto up = up_descs[expert];
    const bool ready = gate.data && up.data &&
                       gate.n == intermediate && gate.k == d_model &&
                       up.n == intermediate && up.k == d_model;
    float gate_sums[warp::partitions_per_lane]{};
    float up_sums[warp::partitions_per_lane]{};
    if (ready)
    {
        const float *row = hidden + static_cast<size_t>(original_slot / top_k) * d_model;
        const size_t weight_base = static_cast<size_t>(n) * d_model;
        // Register p emulates original thread lane+64*p. Moving between
        // columns changes scheduling, never that thread's FMA order.
        for (int base = 0; base < d_model; base += warp::partitions)
        {
#pragma unroll
            for (int partition = 0; partition < warp::partitions_per_lane; ++partition)
            {
                const int k = base + lane + partition * warp::width;
                if (k < d_model)
                {
                    const float activation = row[k];
                    gate_sums[partition] = fmaf(activation,
                        rocm_load_floating_moe_weight(gate.data, weight_base + k, format), gate_sums[partition]);
                    up_sums[partition] = fmaf(activation,
                        rocm_load_floating_moe_weight(up.data, weight_base + k, format), up_sums[partition]);
                }
            }
        }
    }
    const float gate_sum = warp::reduce(gate_sums);
    const float up_sum = warp::reduce(up_sums);
    if (lane == 0)
    {
        const size_t index = static_cast<size_t>(grouped_slot) * intermediate + n;
        grouped_swiglu[index] = ready ? llaminar2::rocm::decode_equivalent::swigluValue(gate_sum, up_sum) : 0.0f;
    }
}

/**
 * @brief Down-project four columns per block without shared reduction scratch.
 *
 * The preceding node already evaluated SwiGLU once. Each subgroup reads those
 * FP32 bits, emulates the serial 256-partition FMA/tree, and publishes either
 * one canonical route or the original ordered route sum. Missing routes still
 * contribute an explicit rounded +0, including signed-zero semantics.
 */
__global__ __launch_bounds__(llaminar2::device_moe_floating_warp::threads)
void rocm_grouped_floating_down_prefill_kernel(
    const float *__restrict__ grouped_swiglu,
    const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs,
    const int *__restrict__ original_to_grouped,
    const int *__restrict__ original_expert_ids,
    const float *__restrict__ grouped_weights,
    float *__restrict__ output,
    float *__restrict__ canonical_routes,
    int seq_len,
    int total_slots,
    int top_k,
    int d_model,
    int intermediate,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format)
{
    namespace warp = llaminar2::device_moe_floating_warp;
    const int lane = threadIdx.x % warp::width;
    const int n = blockIdx.x * warp::columns_per_block + threadIdx.x / warp::width;
    const int publication_row = rocm_logical_row_from_grid_yz();
    const bool canonical = canonical_routes != nullptr;
    if (n >= d_model || (canonical ? publication_row >= total_slots : publication_row >= seq_len))
        return;

    const int route_begin = canonical ? publication_row : publication_row * top_k;
    const int route_end = canonical ? publication_row + 1 : route_begin + top_k;
    float output_sum = 0.0f;
    for (int original_slot = route_begin; original_slot < route_end; ++original_slot)
    {
        const int grouped_slot = original_to_grouped[original_slot];
        const int expert = original_expert_ids ? original_expert_ids[original_slot] : 0;
        const bool valid = grouped_slot >= 0 && expert >= 0 && expert < num_experts;
        llaminar2::DeviceMoEFloatingMatrixDesc down{};
        if (valid)
            down = down_descs[expert];
        const bool ready = valid && down.data && down.n == d_model && down.k == intermediate;
        if (!ready)
        {
            if (lane == 0)
                output_sum = rocm_native_quantised_accumulate_rn(output_sum, 0.0f);
            continue;
        }
        float sums[warp::partitions_per_lane]{};
        const size_t row_base = static_cast<size_t>(grouped_slot) * intermediate;
        const size_t weight_base = static_cast<size_t>(n) * intermediate;
        for (int base = 0; base < intermediate; base += warp::partitions)
        {
#pragma unroll
            for (int partition = 0; partition < warp::partitions_per_lane; ++partition)
            {
                const int k = base + lane + partition * warp::width;
                if (k < intermediate)
                    sums[partition] = fmaf(grouped_swiglu[row_base + k],
                        rocm_load_floating_moe_weight(down.data, weight_base + k, format), sums[partition]);
            }
        }
        const float sum = warp::reduce(sums);
        if (lane == 0)
            output_sum = grouped_weights
                ? rocm_native_quantised_accumulate_rn(output_sum,
                    rocm_native_quantised_weight_route_rn(grouped_weights[grouped_slot], sum))
                : sum;
    }
    if (lane == 0)
    {
        const size_t index = static_cast<size_t>(publication_row) * d_model + n;
        if (canonical)
            canonical_routes[index] = output_sum;
        else
            output[index] = output_sum;
    }
}

__global__ void rocm_moe_prefill_gather_expert_runtime_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ hidden,
    float *__restrict__ batch_buffer,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !hidden || !batch_buffer || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets || !runtime->grouped_token_ids) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

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

__global__ void rocm_moe_prefill_scatter_expert_runtime_kernel(
    float *__restrict__ output,
    const float *__restrict__ expert_output,
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !output || !expert_output || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

    const int row = linear / d_model;
    const int col = linear - row * d_model;
    const int count = runtime->expert_counts[expert_id];
    if (row >= count) return;

    const int offset = runtime->expert_offsets[expert_id];
    const int route_slot = runtime->grouped_token_ids[offset + row];
    const int token_id =
        runtime->top_k > 0u ? route_slot / static_cast<int>(runtime->top_k) : -1;
    if (token_id < 0 || token_id >= max_tokens) return;

    const float weight = runtime->grouped_route_weights[offset + row];
    atomicAdd(&output[token_id * d_model + col], weight * expert_output[linear]);
}

// =========================================================================
// Phase 3 extern "C" bridge functions
// =========================================================================

extern "C"
{

/** @brief Validate and launch one HIP grouping reset; see the shared bridge declaration. */
bool hipMoE_initialize_grouping(
    llaminar2::MoEGroupingInitialization binding, int device_ordinal, void *stream)
{
    if (!binding.valid() || !stream || hipSetDevice(device_ordinal) != hipSuccess)
        return false;
    constexpr int threads = 256;
    const int blocks = 1 + (binding.workItems() - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_initialize_grouping_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream), binding);
    return hipGetLastError() == hipSuccess;
}

bool hipMoE_count_per_expert(
    const int *routing_indices, int *expert_counts,
    int total_slots, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_count_per_expert_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, expert_counts, total_slots, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE count_per_expert kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_exclusive_scan(
    int *expert_counts, int *expert_offsets,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_exclusive_scan_kernel,
        dim3(1), dim3(kGateLogitsGroupedThreads), 0, static_cast<hipStream_t>(stream),
        expert_counts, expert_offsets, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE exclusive_scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_max_expert_count(
    const int *expert_counts, int *d_max_out,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_max_expert_count_kernel,
        dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream),
        expert_counts, d_max_out, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE max_expert_count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_build_active_expert_list(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_build_active_expert_list_kernel<false>),
        dim3(1), dim3(kActiveExpertThreads), 0, static_cast<hipStream_t>(stream),
        expert_counts, active_expert_ids, num_experts, max_active_experts, nullptr, nullptr, 0);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE build_active_expert_list kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Publish raw active IDs and duplicate-safe packed work in the same grouping launch.
 * @param binding Validated original grouping banks and exact producer stream.
 * @return Whether the complete combined producer was accepted. */
bool hipMoE_build_active_expert_list_with_offset_slots(const llaminar2::MoEGroupingPackedDirectory &binding)
{
    if (!binding.device().is_rocm()) return false;
    llaminar2::HipDeviceGuard::setDevice(binding.device().ordinal);
    (void)hipGetLastError();
    const auto launch = [&]<bool Column>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_moe_build_active_expert_list_kernel<true,Column>),
            dim3(1), dim3(kActiveExpertThreads), 0, static_cast<hipStream_t>(binding.stream()),
            binding.counts(), const_cast<int *>(binding.activeIds()), binding.experts(),
            std::min(binding.totalSlots(), binding.experts()), binding.offsets(), binding.words(), binding.totalSlots());
    };
    if(binding.layout()==llaminar2::MoEGroupingPackedLayout::ColumnFloor8)launch.template operator()<true>();
    else launch.template operator()<false>();
    return hipGetLastError() == hipSuccess;
}

bool hipMoE_scatter_tokens(
    const int *routing_indices, const float *routing_weights,
    const int *expert_offsets, int *write_heads,
    int *grouped_token_indices, int *original_to_grouped,
    float *grouped_weights,
    int total_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_scatter_tokens_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights, expert_offsets, write_heads,
        grouped_token_indices, original_to_grouped, grouped_weights,
        total_slots, num_experts, top_k);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_build_runtime_original_to_grouped(
    const void *runtime,
    int *original_to_grouped,
    int current_slots,
    int max_slots,
    int num_experts,
    int top_k,
    int device_idx,
    void *stream)
{
    if (!runtime || !original_to_grouped || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    auto hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemsetAsync(
        original_to_grouped,
        0xff,
        static_cast<size_t>(max_slots) * sizeof(int),
        hip_stream);
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE runtime original_to_grouped memset failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_build_runtime_original_to_grouped_kernel,
        dim3(num_experts), dim3(256), 0, hip_stream,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        original_to_grouped,
        current_slots,
        max_slots,
        num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE build_runtime_original_to_grouped kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
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
 * @param top_k Positive route width, at most 16, dividing total_slots.
 * @param max_active_experts Positive active extent, at most min(total_slots, num_experts).
 * @param device_idx Exact native device owning the stream and buffers.
 * @param stream Exact non-null stream retained until all borrowed buffers retire.
 * @return True only after the selected native grouping launch succeeds.
 */
bool hipMoE_group_tokens_small_float(
    const float *routing_indices, const float *routing_weights,
    int *expert_counts, int *expert_offsets,
    int *grouped_token_indices, int *original_to_grouped,
    int *original_expert_ids,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots, int num_experts, int top_k,
    int max_active_experts,
    int device_idx, void *stream)
{
    /*
     * This ABI is intentionally stricter than the legacy prefill grouping
     * helper.  MTP verifier rows may publish through the original-route
     * parallel down kernel, so the caller must provide original_expert_ids
     * alongside original_to_grouped.  Returning success without that map would
     * make the grouped path descriptor-dependent on old workspace contents.
     */
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream ||
        !routing_indices || !routing_weights || !expert_counts || !expert_offsets ||
        !grouped_token_indices || !original_to_grouped ||
        !original_expert_ids ||
        !grouped_weights || !active_expert_ids ||
        total_slots <= 0 || total_slots > kDeviceMoESmallGroupMaxSlots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
        total_slots % top_k != 0 ||
        max_active_experts <= 0 || max_active_experts > total_slots ||
        max_active_experts > num_experts)
    {
        fprintf(stderr, "HIP MoE small float grouping invalid arguments\n");
        return false;
    }

    const auto prefix_policy = small_float_group_prefix_policy(num_experts);
    if (prefix_policy == SmallFloatGroupPrefixPolicy::IndependentExpert)
    {
        hipLaunchKernelGGL(
            (rocm_moe_group_tokens_small_float_kernel<SmallFloatGroupPrefixPolicy::IndependentExpert>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)),
            0, static_cast<hipStream_t>(stream), routing_indices, routing_weights,
            expert_counts, expert_offsets, grouped_token_indices, original_to_grouped,
            original_expert_ids, grouped_weights, active_expert_ids,
            total_slots, num_experts, top_k, max_active_experts);
    }
    else
    {
        hipLaunchKernelGGL(
            (rocm_moe_group_tokens_small_float_kernel<SmallFloatGroupPrefixPolicy::CooperativeScan>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)),
            0, static_cast<hipStream_t>(stream), routing_indices, routing_weights,
            expert_counts, expert_offsets, grouped_token_indices, original_to_grouped,
            original_expert_ids, grouped_weights, active_expert_ids,
            total_slots, num_experts, top_k, max_active_experts);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE small float grouping kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prepare_shared_expert_group(
    int *expert_offsets,
    int *expert_counts,
    int *grouped_token_indices,
    int *original_to_grouped,
    float *grouped_weights,
    int *active_expert_ids,
    llaminar2::DeviceRowRange rows,
    int device_idx,
    void *stream)
{
    const int seq_len = rows.physicalRows();
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !expert_offsets || !expert_counts ||
        !grouped_token_indices || !original_to_grouped ||
        !grouped_weights || !active_expert_ids ||
        seq_len <= 0)
    {
        fprintf(stderr, "HIP MoE shared expert grouping invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int blocks = (seq_len + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prepare_shared_expert_group_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_offsets, expert_counts,
        grouped_token_indices, original_to_grouped, grouped_weights,
        active_expert_ids, rows);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared expert grouping kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_prefill_routes_runtime(
    const float *routing_indices, const float *routing_weights,
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int filter_to_local_runtime_experts,
    int retain_routes_for_deferred_commit,
    int device_idx, void *stream,
    llaminar2::DeviceMoEProjectionSet expected_projections)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !llaminar2::deviceMoEProjectionSetValid(expected_projections) || !routing_indices || !routing_weights || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE group_prefill_routes invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_moe_prefill_group_small_runtime_kernel<true>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)), 0, hip_stream,
            runtime_view,
            routing_indices,
            routing_weights,
            /*original_to_grouped=*/nullptr,
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
            expected_projections);
        const hipError_t small_err = hipGetLastError();
        if (small_err != hipSuccess)
        {
            fprintf(stderr,
                    "HIP MoE verifier-sized prefill grouping launch failed: %s\n",
                    hipGetErrorString(small_err));
            return false;
        }
        return true;
    }

    const int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, /*original_to_grouped=*/nullptr,
        max_slots, num_experts, 1);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int slot_blocks = (max_slots + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_group_cast_count_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, routing_indices, routing_weights,
        current_slots, max_slots, num_experts,
        filter_to_local_runtime_experts,
        retain_routes_for_deferred_commit,
        expected_projections);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group cast/count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, /*original_to_grouped=*/nullptr,
        current_slots, max_slots, num_experts, top_k);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group scatter kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_regroup_prefill_routes_runtime_assignments(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int retain_routes_for_deferred_commit,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE regroup_prefill_routes invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_moe_prefill_group_small_runtime_kernel<false>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)), 0, hip_stream,
            runtime_view,
            /*routing_indices=*/nullptr,
            /*routing_weights=*/nullptr,
            /*original_to_grouped=*/nullptr,
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
            llaminar2::DeviceMoEProjectionSet::CompleteExpert);
        const hipError_t small_err = hipGetLastError();
        if (small_err != hipSuccess)
        {
            fprintf(stderr,
                    "HIP MoE verifier-sized prefill regrouping launch failed: %s\n",
                    hipGetErrorString(small_err));
            return false;
        }
        return true;
    }

    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    clear_items = (clear_items > split_items) ? clear_items : split_items;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    const int slot_blocks = (max_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, /*original_to_grouped=*/nullptr,
        max_slots, num_experts, 0);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_count_assigned_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts,
        retain_routes_for_deferred_commit);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup assigned-count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, /*original_to_grouped=*/nullptr,
        current_slots, max_slots, num_experts, top_k);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill regroup scatter kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

/**
 * @brief Publish router grouping and every runtime compute descriptor.
 *
 * Verifier-sized route sets execute as one workgroup and publish the complete
 * grouped execution plan in one launch. Larger prefills retain deterministic
 * scalable grouping, but author the inverse route map during scatter and then
 * publish descriptors on the same explicit stream. Both regimes expose one
 * total, graph-capturable API and never observe intermediate state on host.
 */
__attribute__((visibility("default"))) bool
hipMoE_group_prefill_routes_and_materialize_plan_runtime(
    const float *routing_indices,
    const float *routing_weights,
    void *runtime,
    int *original_to_grouped,
    DeviceNativeQuantisedMatrixDescView *gate_descs,
    DeviceNativeQuantisedMatrixDescView *up_descs,
    DeviceNativeQuantisedMatrixDescView *down_descs,
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
    llaminar2::DeviceMoEProjectionSet expected_projections)
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
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
        max_active_experts <= 0 || max_active_experts > num_experts ||
        (expected_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
         !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)))
    {
        fprintf(stderr, "HIP MoE complete router-plan publication invalid arguments\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    auto hip_stream = static_cast<hipStream_t>(stream);
    constexpr int threads = 256;

    if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_moe_prefill_group_small_runtime_kernel<true, true>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)), 0, hip_stream,
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
            expected_projections);
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "HIP MoE complete verifier router-plan launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    const int clear_items = max_slots > num_experts ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    hipLaunchKernelGGL(
        rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view,
        original_to_grouped,
        max_slots,
        num_experts,
        /*clear_route_metadata=*/1);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete router-plan clear launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    const int slot_blocks = (max_slots + threads - 1) / threads;
    hipLaunchKernelGGL(
        rocm_moe_prefill_group_cast_count_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view,
        routing_indices,
        routing_weights,
        current_slots,
        max_slots,
        num_experts,
        filter_to_local_runtime_experts,
        retain_routes_for_deferred_commit,
        expected_projections);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete router-plan conversion launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view,
        original_to_grouped,
        current_slots,
        max_slots,
        num_experts,
        top_k);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete router-plan scatter launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel,
        dim3(1), dim3(threads), 0, hip_stream,
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
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete router-plan descriptor launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Publish final LLEP assignments as one complete grouped compute plan.
 *
 * Device-side assignment has already finalized participant ids. The bounded
 * verifier specialization groups and publishes every compute product in one
 * launch; scalable prefill uses deterministic count/scatter plus one ordered
 * descriptor launch without rebuilding the inverse map afterward.
 */
__attribute__((visibility("default"))) bool
hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
    void *runtime,
    int *original_to_grouped,
    DeviceNativeQuantisedMatrixDescView *gate_descs,
    DeviceNativeQuantisedMatrixDescView *up_descs,
    DeviceNativeQuantisedMatrixDescView *down_descs,
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
    llaminar2::DeviceMoEProjectionSet expected_projections)
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
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
        max_active_experts <= 0 || max_active_experts > num_experts ||
        (expected_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
         !llaminar2::deviceMoEWeightFormatIsFloating(expected_format)))
    {
        fprintf(stderr, "HIP MoE complete assigned-plan publication invalid arguments\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    auto hip_stream = static_cast<hipStream_t>(stream);
    constexpr int threads = 256;

    if (max_slots <= kDeviceMoERuntimeSmallGroupMaxSlots)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_moe_prefill_group_small_runtime_kernel<false, true>),
            dim3(1), dim3(llaminar2::moe_runtime_abi::expertGroupingThreads(num_experts)), 0, hip_stream,
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
            expected_projections);
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "HIP MoE complete verifier assigned-plan launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    const int clear_items = max_slots > num_experts ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    hipLaunchKernelGGL(
        rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view,
        original_to_grouped,
        max_slots,
        num_experts,
        /*clear_route_metadata=*/0);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete assigned-plan clear launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    const int slot_blocks = (max_slots + threads - 1) / threads;
    hipLaunchKernelGGL(
        rocm_moe_prefill_group_count_assigned_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view,
        current_slots,
        max_slots,
        num_experts,
        retain_routes_for_deferred_commit);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete assigned-plan count launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view,
        original_to_grouped,
        current_slots,
        max_slots,
        num_experts,
        top_k);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete assigned-plan scatter launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel,
        dim3(1), dim3(threads), 0, hip_stream,
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
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE complete assigned-plan descriptor launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_commit_grouped_verifier_histograms_batch(
    void *runtime_layers,
    int layer_count,
    size_t layer_stride_bytes,
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
    if (!runtime_layers ||
        layer_count <= 0 ||
        (layer_count > 1 && layer_stride_bytes < sizeof(DeviceMoELayerRuntimeView)) ||
        !accepted_state_counts ||
        !publication_ok_flags ||
        !stream ||
        request_count <= 0 ||
        rows_per_request <= 0 ||
        total_rows != request_count * rows_per_request ||
        num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 ||
        top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    constexpr int threads = 256;
    const int total_slots = total_rows * top_k;
    const int blocks = (total_slots + threads - 1) / threads;
    hipLaunchKernelGGL(
        rocm_moe_commit_grouped_verifier_histograms_kernel,
        dim3(static_cast<unsigned int>(blocks),
             static_cast<unsigned int>(layer_count),
             1u),
        dim3(threads),
        0,
        static_cast<hipStream_t>(stream),
        runtime_layers,
        layer_stride_bytes,
        accepted_state_counts,
        publication_ok_flags,
        request_count,
        rows_per_request,
        total_rows,
        num_experts,
        top_k);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "HIP MoE committed grouped-verifier histogram batch launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_commit_grouped_verifier_histograms(
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
    return hipMoE_commit_grouped_verifier_histograms_batch(
        runtime,
        1,
        0u,
        accepted_state_counts,
        publication_ok_flags,
        request_count,
        rows_per_request,
        total_rows,
        num_experts,
        top_k,
        device_idx,
        stream);
}

bool hipMoE_assign_prefill_routes_least_loaded_resident(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    const int32_t *absolute_position_ids,
    const int32_t *active_row_count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !absolute_position_ids || !active_row_count || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts ||
        current_slots % top_k != 0 || max_slots % top_k != 0)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_least_loaded_resident invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    constexpr int assignment_threads = 64;
    const int current_rows = current_slots / top_k;
    const int assignment_blocks =
        current_rows > 0
            ? (current_rows + assignment_threads - 1) / assignment_threads
            : 1;
    hipLaunchKernelGGL(
        rocm_moe_prefill_llep_assign_resident_rows_logical_position_kernel,
        dim3(assignment_blocks), dim3(assignment_threads), 0, hip_stream,
        runtime_view,
        absolute_position_ids,
        active_row_count,
        current_slots, max_slots, num_experts, top_k);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "HIP MoE logical-position resident-row assignment kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_plan_prefill_routes_least_loaded_current_batch(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
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
    int device_idx, void *stream)
{
    (void)top_k;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        alpha_numerator == 0u || alpha_denominator == 0u ||
        lambda_numerator == 0u || lambda_denominator == 0u)
    {
        fprintf(stderr, "HIP MoE plan_prefill_routes_least_loaded_current_batch invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int split_items = num_experts * static_cast<int>(kDeviceMoEMaxParticipants);
    const int clear_items = (split_items > num_experts) ? split_items : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;
    const int slot_blocks = (max_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_clear_current_batch_plan_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_count_expert_routes_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_llep_plan_current_batch_kernel,
        dim3(1), dim3(threads), 0, hip_stream,
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

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch plan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_from_llep_current_batch_plan_no_transfers invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts, top_k, 0, nullptr, nullptr);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP current-batch span assignment kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    const void *transfer_status,
    const void *apply_status,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !transfer_status || !apply_status || !stream ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        fprintf(stderr, "HIP MoE assign_prefill_routes_from_llep_current_batch_plan_after_transfers invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;

    hipLaunchKernelGGL(rocm_moe_prefill_llep_assign_routes_from_current_batch_spans_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        runtime_view,
        current_slots,
        max_slots,
        num_experts,
        top_k,
        1,
        static_cast<const DeviceMoERebalanceStatusView *>(transfer_status),
        static_cast<const DeviceMoERebalanceApplyStatusView *>(apply_status));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill LLEP after-transfer span assignment kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
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
__attribute__((visibility("default"))) bool hipMoE_materialize_runtime_selected_decode_descriptors(
    const void *runtime, const int *expert_ids,
    llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs, llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
    llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs, int num_experts, int top_k,
    llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
    llaminar2::DeviceMoEProjectionSet projections)
{
    if (!runtime || !expert_ids || !gate_descs || !up_descs || !down_descs || !stream ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts ||
        !llaminar2::deviceMoEProjectionSetValid(projections) ||
        (format != llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
         !llaminar2::deviceMoEWeightFormatIsFloating(format))) return false;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(rocm_moe_materialize_runtime_selected_decode_descriptors_kernel, dim3(1), dim3(top_k), 0,
        static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime), expert_ids,
        reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(gate_descs), reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(up_descs), reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(down_descs), num_experts, top_k, format, projections);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        fprintf(stderr, "HIP MoE selected decode descriptor publication failed: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}

__attribute__((visibility("default"))) bool hipMoE_materialize_runtime_prefill_descriptor_tables(
    const void *runtime,
    llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
    llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
    llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
    int num_experts,
    int device_idx,
    void *stream, llaminar2::DeviceMoEProjectionSet expected_projections)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !gate_descs || !up_descs || !down_descs || !stream ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
    {
        fprintf(stderr, "HIP MoE materialize_runtime_prefill_descriptor_tables invalid arguments\n");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel,
        dim3(1), dim3(kActiveExpertThreads), 0, hip_stream,
        static_cast<DeviceMoELayerRuntimeView *>(const_cast<void *>(runtime)),
        reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(gate_descs),
        reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(up_descs),
        reinterpret_cast<DeviceNativeQuantisedMatrixDescView *>(down_descs),
        num_experts,
        /*active_expert_ids=*/nullptr,
        /*max_active_experts=*/0,
        /*current_slots=*/0,
        /*publish_prefill_histogram=*/0,
        llaminar2::DeviceMoEWeightFormat::NativeQuantised,
        expected_projections);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE runtime prefill descriptor materialize kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

/** @brief Materialize graph-stable floating descriptors from the active bank. */
__attribute__((visibility("default"))) bool hipMoE_materialize_runtime_floating_descriptor_tables(
    const void *runtime,
    llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
    llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
    llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format,
    int device_idx,
    void *stream, llaminar2::DeviceMoEProjectionSet expected_projections)
{
    if (!runtime || !gate_descs || !up_descs || !down_descs || !stream ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        !llaminar2::deviceMoEWeightFormatIsFloating(format))
    {
        fprintf(stderr, "HIP MoE floating descriptor materialization has invalid arguments\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        rocm_moe_materialize_runtime_floating_descriptor_tables_kernel,
        dim3(1),
        dim3(kFloatingMoEThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        gate_descs,
        up_descs,
        down_descs,
        num_experts,
        format, expected_projections);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE floating descriptor materialization failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Launch fixed-tree floating gate/up decode. */
__attribute__((visibility("default"))) bool rocmMoE_grouped_gate_up_floating_decode_table(
    const float *hidden,
    const llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
    const llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
    const int *expert_ids,
    float *const *gate_outputs,
    float *const *up_outputs,
    int num_active,
    int intermediate,
    int d_model,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format,
    int device_idx,
    void *stream)
{
    if (!hidden || !gate_descs || !up_descs || !expert_ids ||
        !gate_outputs || !up_outputs || num_active <= 0 ||
        intermediate <= 0 || d_model <= 0 || num_experts <= 0 ||
        !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        rocm_grouped_floating_gate_up_decode_kernel,
        make_rocm_row_major_grid(
            static_cast<unsigned int>(intermediate), num_active),
        dim3(kFloatingMoEThreads),
        0,
        static_cast<hipStream_t>(stream),
        hidden,
        gate_descs,
        up_descs,
        expert_ids,
        gate_outputs,
        up_outputs,
        num_active,
        intermediate,
        d_model,
        num_experts,
        format);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE floating gate/up decode launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Launch fixed-tree floating SwiGLU/down decode. */
__attribute__((visibility("default"))) bool rocmMoE_grouped_swiglu_down_floating_decode_table(
    const float *const *gate_rows,
    const float *const *up_rows,
    const llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
    const int *expert_ids,
    const float *route_weights,
    float *output,
    float *canonical_routes,
    int num_active,
    int d_model,
    int intermediate,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format,
    int device_idx,
    void *stream)
{
    if (!gate_rows || !up_rows || !down_descs || !expert_ids ||
        (!route_weights && (output || !canonical_routes)) || (!output && !canonical_routes) ||
        num_active <= 0 || d_model <= 0 || intermediate <= 0 ||
        num_experts <= 0 ||
        !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    const int publication_rows = canonical_routes ? num_active : 1;
    hipLaunchKernelGGL(
        rocm_grouped_floating_swiglu_down_decode_kernel,
        make_rocm_row_major_grid(
            static_cast<unsigned int>(d_model), publication_rows),
        dim3(kFloatingMoEThreads),
        0,
        static_cast<hipStream_t>(stream),
        gate_rows,
        up_rows,
        down_descs,
        expert_ids,
        route_weights,
        output,
        canonical_routes,
        num_active,
        d_model,
        intermediate,
        num_experts,
        format);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE floating SwiGLU/down decode launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Launch the declared fixed-tree floating phase on the exact stream.
 * @param grouped_swiglu Existing exclusive arena intermediate, in FP32.
 * @param execution Complete transaction, gate/up producer, or full-K down-column consumer.
 *
 * Complete execution retains two captured nodes. A split phase borrows the
 * same storage and serial arithmetic; it adds no conversion or allocation.
 */
__attribute__((visibility("default"))) bool rocmMoE_grouped_floating_prefill_pipeline(
    const float *hidden,
    const llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
    const llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
    const llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
    const int *original_to_grouped,
    const int *original_expert_ids,
    const float *grouped_weights,
    float *grouped_swiglu,
    float *output,
    float *canonical_routes,
    int seq_len,
    int total_slots,
    int top_k,
    int d_model,
    int intermediate,
    int num_experts,
    llaminar2::DeviceMoEWeightFormat format,
    int device_idx,
    void *stream,
    const llaminar2::MoEPrefillProjectionExecution &execution)
{
    const bool raw_cohort =
        execution.routeWeighting() == llaminar2::MoEExpertRouteWeighting::CompletedCohort;
    if (raw_cohort && (output || !canonical_routes))
        return false;
    if ((execution.executesGateUp() && (!hidden || !gate_descs || !up_descs)) ||
        (execution.executesDown() && (!down_descs || !grouped_weights ||
            (!output && !canonical_routes))) ||
        !original_to_grouped ||
        (!original_expert_ids && num_experts != 1) ||
        !grouped_swiglu || execution.modelColumns() != d_model ||
        seq_len <= 0 || top_k <= 0 || total_slots != static_cast<std::int64_t>(seq_len) * top_k ||
        d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
        !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    // Only independent output columns are partitioned; K's fixed tree is unchanged.
    if (execution.executesGateUp())
    {
        hipLaunchKernelGGL(
            rocm_grouped_floating_gate_up_swiglu_prefill_kernel,
            make_rocm_row_major_grid(
                static_cast<unsigned int>((intermediate + llaminar2::device_moe_floating_warp::columns_per_block - 1) /
                    llaminar2::device_moe_floating_warp::columns_per_block), total_slots),
            dim3(kFloatingMoEThreads),
            0,
            hip_stream,
            hidden,
            gate_descs,
            up_descs,
            original_to_grouped,
            original_expert_ids,
            grouped_swiglu,
            total_slots,
            top_k,
            intermediate,
            d_model,
            num_experts,
            format);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE floating prefill gate/up launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    if (!execution.executesDown())
        return true;
    const int down_columns = execution.columnCount();
    const int publication_rows = canonical_routes ? total_slots : seq_len;
    hipLaunchKernelGGL(
        rocm_grouped_floating_down_prefill_kernel,
        make_rocm_row_major_grid(
            static_cast<unsigned int>((down_columns + llaminar2::device_moe_floating_warp::columns_per_block - 1) /
                llaminar2::device_moe_floating_warp::columns_per_block), publication_rows),
        dim3(kFloatingMoEThreads),
        0,
        hip_stream,
        grouped_swiglu,
        down_descs,
        original_to_grouped,
        original_expert_ids,
        raw_cohort ? nullptr : grouped_weights,
        output,
        canonical_routes,
        seq_len,
        total_slots,
        top_k,
        down_columns,
        intermediate,
        num_experts,
        format);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE floating prefill down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Fuse runtime descriptor and active-expert publication for prefill.
 */
__attribute__((visibility("default"))) bool hipMoE_materialize_runtime_prefill_plan(
    const void *runtime,
    DeviceNativeQuantisedMatrixDescView *gate_descs,
    DeviceNativeQuantisedMatrixDescView *up_descs,
    DeviceNativeQuantisedMatrixDescView *down_descs,
    int *active_expert_ids,
    int num_experts,
    int max_active_experts,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !gate_descs || !up_descs || !down_descs ||
        !active_expert_ids || !stream ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        max_active_experts <= 0 || max_active_experts > num_experts)
    {
        fprintf(stderr, "HIP MoE materialize_runtime_prefill_plan invalid arguments\n");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(rocm_moe_materialize_runtime_prefill_descriptor_tables_kernel,
        dim3(1), dim3(kActiveExpertThreads), 0, hip_stream,
        static_cast<DeviceMoELayerRuntimeView *>(const_cast<void *>(runtime)),
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

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE runtime prefill plan materialize kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prefill_gather_expert_runtime(
    const void *runtime,
    const float *hidden,
    float *batch_buffer,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !hidden || !batch_buffer || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_gather_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        hidden, batch_buffer, expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prefill_scatter_expert_runtime(
    float *output,
    const float *expert_output,
    const void *runtime,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!output || !expert_output || !runtime || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_scatter_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 3)

// =========================================================================
// Phase 4: Tensor-aware utility kernels
// =========================================================================

/// Convert float array to int array on device (reverse: for GPU-side token grouping)
__global__ void rocm_moe_float_to_int_kernel(
    const float *__restrict__ input, int *__restrict__ output, int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count)
        output[idx] = static_cast<int>(input[idx]);
}

__global__ void rocm_moe_float_to_masked_int_kernel(
    const float *__restrict__ input,
    int *__restrict__ output,
    const uint8_t *__restrict__ expert_mask,
    int count,
    int num_experts)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;
    const int expert = static_cast<int>(input[idx]);
    output[idx] =
        (expert >= 0 && expert < num_experts && expert_mask && expert_mask[expert] != 0u)
            ? expert
            : -1;
}

extern "C" {

bool hipMoE_stage_mutable_pointer_arrays(
    float **d_a,
    float **d_b,
    float *const *h_a,
    float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE mutable pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.mutable_ptrs_a[i] = h_a[i];
        payload.mutable_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_mutable_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_stage_const_pointer_arrays(
    const float **d_a,
    const float **d_b,
    const float *const *h_a,
    const float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE const pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE const pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.const_ptrs_a[i] = h_a[i];
        payload.const_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_const_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE const pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_float_to_int(
    const float *d_input, int *d_output, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    // Convert FP32 routing IDs to integer grouping IDs on the explicit stream.
    // a trivial __device__ lambda would be ideal but HIP doesn't support
    // device lambdas in extern "C".  Instead, add a dedicated kernel below.
    // For now, use a trivial inline kernel via hipLaunchKernelGGL.
    // The kernel is defined right after this bridge.
    hipLaunchKernelGGL(rocm_moe_float_to_int_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE float_to_int kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_float_to_masked_int(
    const float *d_input,
    int *d_output,
    const uint8_t *d_expert_mask,
    int count,
    int num_experts,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_float_to_masked_int_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, d_expert_mask, count, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE float_to_masked_int kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 4)

