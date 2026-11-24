/**
 * @file DeviceMoERebalanceArrivalIndices.inl
 * @brief Compact validated live rebalance commands without scanning idle slots.
 *
 * Included inside each vendor's minimal device ABI scope. A single control CTA
 * visits only the published wave and preserves the original lowest-command-index
 * tie break for duplicate destination slots. Arrival ordering is immaterial
 * after arbitration: every emitted index has an exclusive destination.
 */

/** @return Whether this command owns potentially live payload work on this participant. */
__device__ __forceinline__ bool rebalance_local_payload_arrival(
    const DeviceMoERebalancePlanEntryView &plan, const DeviceMoERebalanceConfigView &config,
    uint32_t local_transfer_slot_count, uint32_t local_payload_slot_count)
{
    return rebalance_plan_requires_payload(plan.op) && plan.layer < config.num_layers &&
        plan.expert < config.num_experts && plan.source_participant < config.participant_count &&
        (plan.source_resident_mask & runtime_participant_bit(static_cast<int>(plan.source_participant))) != 0u &&
        plan.destination_participant == config.participant_id && plan.destination_slot < local_transfer_slot_count &&
        plan.payload_slot < local_payload_slot_count;
}

/** @return Exact number of distinct local arrival commands in the immutable copy-status wave. */
__device__ void rebalance_collect_local_arrival_indices(
    const DeviceMoERebalancePlanEntryView *plan_entries, const uint32_t *plan_count, uint32_t plan_capacity,
    const DeviceMoERebalanceCommandBufferHeaderView *command_header, uint32_t local_payload_slot_count,
    uint32_t local_transfer_slot_count, const DeviceMoERebalanceConfigView &config,
    DeviceMoERebalanceApplyStatusView *status, llaminar2::DeviceMoERebalanceArrivalScratch *scratch,
    uint32_t command_buffer_count)
{
    if (threadIdx.x == 0) scratch->live_count = 0u;
    const auto wave = status->transaction_wave_index;
    if (wave >= rebalance_command_buffer_count(command_buffer_count))
        return;
    plan_entries += static_cast<uint64_t>(wave) * plan_capacity;
    if (plan_count) plan_count += wave;
    if (command_header) command_header += wave;
    const uint32_t count = rebalance_command_count(plan_count, plan_capacity, command_header);
    auto *first_for_slot = reinterpret_cast<uint32_t *>(scratch + 1);
    auto *arrival_indices = first_for_slot + local_transfer_slot_count;
    if (threadIdx.x == 0)
    {
        atomicAdd(&status->plan_entries_seen, count);
        status->local_transfer_slot_count = local_transfer_slot_count;
    }

    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        const auto &plan = plan_entries[i];
        if (plan.op == 0u) continue;
        if (!rebalance_plan_applies_runtime(plan.op) || plan.layer >= config.num_layers ||
            plan.expert >= config.num_experts || plan.source_participant >= config.participant_count ||
            plan.destination_participant >= config.participant_count)
        {
            atomicAdd(&status->invalid_plan_entries, 1u);
            continue;
        }
        if (!rebalance_plan_requires_payload(plan.op)) continue;
        if ((plan.source_resident_mask & runtime_participant_bit(static_cast<int>(plan.source_participant))) == 0u)
        {
            atomicAdd(&status->invalid_plan_entries, 1u);
            continue;
        }
        if (plan.destination_participant != config.participant_id)
        {
            atomicAdd(&status->skipped_wrong_destination, 1u);
            continue;
        }
        if (plan.destination_slot >= local_transfer_slot_count)
        {
            atomicAdd(&status->missing_destination_slots, 1u);
            if (atomicCAS(&status->first_missing_destination_layer, kDeviceMoEInvalidSlot, plan.layer) == kDeviceMoEInvalidSlot)
            {
                status->first_missing_destination_slot = plan.destination_slot;
                status->first_missing_destination_expert = plan.expert;
                status->first_missing_destination_source = plan.source_participant;
            }
            continue;
        }
        if (plan.payload_slot >= local_payload_slot_count)
        {
            atomicAdd(&status->invalid_plan_entries, 1u);
            continue;
        }
        atomicExch(first_for_slot + plan.destination_slot, kDeviceMoEInvalidSlot);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        const auto &plan = plan_entries[i];
        if (rebalance_local_payload_arrival(plan, config, local_transfer_slot_count, local_payload_slot_count))
            atomicMin(first_for_slot + plan.destination_slot, i);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        const auto &plan = plan_entries[i];
        if (rebalance_local_payload_arrival(plan, config, local_transfer_slot_count, local_payload_slot_count) &&
            first_for_slot[plan.destination_slot] == i)
            arrival_indices[atomicAdd(&scratch->live_count, 1u)] = i;
    }
}
