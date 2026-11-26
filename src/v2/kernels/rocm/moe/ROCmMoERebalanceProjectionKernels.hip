/**
 * @file ROCmMoERebalanceProjectionKernels.hip
 * @brief ROCm overlay rebalance directory, claim index, domain-command projection, and LLEP command materialization kernels.
 */
#include "ROCmMoERebalanceCommon.hip.inc"

namespace
{
    /**
     * @brief Build a fresh reverse claim index from live ROCm runtime state.
     *
     * One 256-lane workgroup scans the complete layer/expert table once. Integer
     * atomics publish order-independent counts and the two smallest flattened
     * claim locations, so diagnostics are deterministic without serializing the
     * healthy path. The next same-stream projection consumes `ready` only after
     * this kernel has completed; no host mirror or synchronization participates.
     */
    __global__ void build_transfer_slot_claim_index_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoETransferSlotClaimIndexView *index)
    {
        if (!index)
            return;

        const uint32_t lane = threadIdx.x;
        auto *const entries =
            rebalance_transfer_slot_claim_index_entries(index);
        if (lane == 0u)
        {
            DeviceMoETransferSlotClaimIndexView empty{};
            empty.magic = kDeviceMoERebalanceMagic;
            empty.version = kDeviceMoERebalanceVersion;
            empty.participant_id = config.participant_id;
            empty.participant_count = config.participant_count;
            empty.num_layers = config.num_layers;
            empty.num_experts = config.num_experts;
            empty.slot_count = local_transfer_slot_count;
            empty.first_invalid_runtime_layer = kDeviceMoEInvalidSlot;
            empty.first_invalid_claim_flat = kDeviceMoEInvalidSlot;
            empty.first_duplicate_claim_flat = kDeviceMoEInvalidSlot;
            empty.summary = rebalance_empty_transfer_slot_claim_summary();
            *index = empty;
        }
        for (uint32_t slot = lane;
             slot < local_transfer_slot_count;
             slot += blockDim.x)
        {
            DeviceMoETransferSlotClaimIndexEntryView empty{};
            empty.first_claim_flat = kDeviceMoEInvalidSlot;
            empty.second_claim_flat = kDeviceMoEInvalidSlot;
            entries[slot] = empty;
        }
        __syncthreads();

        const bool geometry_ok =
            runtime_layers &&
            local_transfer_slots &&
            rebalance_config_ok(config) &&
            local_transfer_slot_count != 0u &&
            local_transfer_slot_count ==
                config.transfer_slot_directory_capacity;
        if (!geometry_ok)
        {
            if (lane == 0u)
            {
                index->invalid_runtime_layers = 1u;
                index->ready = 1u;
            }
            return;
        }

        for (uint32_t layer = lane;
             layer < config.num_layers;
             layer += blockDim.x)
        {
            const auto &runtime = runtime_layers[layer];
            if (!rebalance_claim_index_runtime_layer_ok(runtime, config))
            {
                atomicAdd(&index->invalid_runtime_layers, 1u);
                atomicMin(&index->first_invalid_runtime_layer, layer);
                continue;
            }
            if (runtime.current_batch_llep_movement_observed != 0u)
                atomicAdd(&index->summary.transient_placement_layers, 1u);
            if (runtime.current_batch_llep_non_owner_assignment_observed != 0u)
                atomicAdd(&index->summary.non_owner_assignment_layers, 1u);
        }

        const uint64_t descriptor_count =
            static_cast<uint64_t>(config.num_layers) * config.num_experts;
        const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        for (uint64_t flat64 = lane;
             flat64 < descriptor_count;
             flat64 += blockDim.x)
        {
            const uint32_t flat = static_cast<uint32_t>(flat64);
            const uint32_t layer = flat / config.num_experts;
            const uint32_t expert = flat % config.num_experts;
            const auto &runtime = runtime_layers[layer];
            if (!rebalance_claim_index_runtime_layer_ok(runtime, config))
                continue;

            const auto &bank = runtime.banks[runtime.active_bank];
            const auto &descriptor = bank.experts[expert];
            const uint32_t resident_mask =
                bank.resident_participant_mask[expert];
            const auto claim =
                llaminar2::moe_rebalance_policy::classifyTransferSlotClaim(
                    descriptor.flags,
                    resident_mask,
                    local_bit,
                    descriptor.owner_participant,
                    config.participant_id,
                    descriptor.local_slot,
                    kDeviceMoEMaxTransferSlots,
                    config.transfer_slot_directory_capacity,
                    kDeviceMoEFlagValid,
                    kDeviceMoEFlagResident,
                    kDeviceMoEFlagTransferSlot);
            if (!claim.claims_storage)
                continue;

            atomicAdd(&index->summary.active_claims, 1u);
            const uint32_t slot =
                descriptor.local_slot < 0
                    ? kDeviceMoEInvalidSlot
                    : static_cast<uint32_t>(descriptor.local_slot);
            const uint32_t directory_error =
                claim.valid()
                    ? rebalance_transfer_slot_directory_claim_error(
                          local_transfer_slots,
                          local_transfer_slot_count,
                          slot,
                          layer,
                          expert,
                          descriptor,
                          config)
                    : claim.invalid_reasons;

            /*
             * Index every in-range physical claim before authenticating its
             * logical directory occupant. A corrupt second descriptor must be
             * visible as both an occupant mismatch and a duplicate storage
             * claim; dropping it at the directory check would hide the very
             * alias the reverse index is responsible for diagnosing.
             */
            if (claim.valid())
            {
                auto &entry = entries[slot];
                atomicAdd(&entry.claim_count, 1u);
                const uint32_t previous_first =
                    atomicMin(&entry.first_claim_flat, flat);
                if (previous_first != kDeviceMoEInvalidSlot &&
                    previous_first != flat)
                {
                    atomicMin(
                        &entry.second_claim_flat,
                        previous_first > flat ? previous_first : flat);
                }
            }
            if (!claim.valid() || directory_error != 0u)
            {
                atomicAdd(&index->summary.invalid_claims, 1u);
                atomicMin(&index->first_invalid_claim_flat, flat);
                continue;
            }
        }
        __syncthreads();

        for (uint32_t slot = lane;
             slot < local_transfer_slot_count;
             slot += blockDim.x)
        {
            const auto &entry = entries[slot];
            if (entry.claim_count != 0u)
            {
                atomicAdd(&index->summary.unique_claims, 1u);
                atomicMax(&index->summary.max_slot, slot);
            }
            if (entry.claim_count > 1u)
            {
                atomicAdd(
                    &index->summary.duplicate_claims,
                    entry.claim_count - 1u);
                atomicMin(
                    &index->first_duplicate_claim_flat,
                    entry.second_claim_flat);
            }
        }
        __syncthreads();

        if (lane == 0u)
        {
            if (index->first_invalid_runtime_layer !=
                kDeviceMoEInvalidSlot)
            {
                const auto &invalid_runtime = runtime_layers[
                    index->first_invalid_runtime_layer];
                index->first_invalid_runtime_active_bank =
                    invalid_runtime.active_bank;
                index->first_invalid_runtime_active_epoch =
                    invalid_runtime.active_epoch;
                index->first_invalid_runtime_expert_count =
                    invalid_runtime.expert_count;
                index->first_invalid_runtime_participant_id =
                    invalid_runtime.participant_id;
                index->first_invalid_runtime_participant_count =
                    invalid_runtime.participant_count;
            }
            if (index->summary.unique_claims != 0u)
            {
                const uint32_t flat =
                    entries[index->summary.max_slot].first_claim_flat;
                index->summary.max_slot_layer = flat / config.num_experts;
                index->summary.max_slot_expert = flat % config.num_experts;
            }
            if (index->summary.duplicate_claims != 0u)
            {
                const uint32_t duplicate_flat =
                    index->first_duplicate_claim_flat;
                for (uint32_t slot = 0u;
                     slot < local_transfer_slot_count;
                     ++slot)
                {
                    if (entries[slot].second_claim_flat == duplicate_flat)
                    {
                        index->summary.first_duplicate_slot = slot;
                        break;
                    }
                }
                index->summary.first_duplicate_layer =
                    duplicate_flat / config.num_experts;
                index->summary.first_duplicate_expert =
                    duplicate_flat % config.num_experts;
            }
            if (index->summary.invalid_claims != 0u)
            {
                const uint32_t flat = index->first_invalid_claim_flat;
                const uint32_t layer = flat / config.num_experts;
                const uint32_t expert = flat % config.num_experts;
                const auto &runtime = runtime_layers[layer];
                const auto &bank = runtime.banks[runtime.active_bank];
                const auto &descriptor = bank.experts[expert];
                const uint32_t resident_mask =
                    bank.resident_participant_mask[expert];
                const auto claim =
                    llaminar2::moe_rebalance_policy::classifyTransferSlotClaim(
                        descriptor.flags,
                        resident_mask,
                        local_bit,
                        descriptor.owner_participant,
                        config.participant_id,
                        descriptor.local_slot,
                        kDeviceMoEMaxTransferSlots,
                        config.transfer_slot_directory_capacity,
                        kDeviceMoEFlagValid,
                        kDeviceMoEFlagResident,
                        kDeviceMoEFlagTransferSlot);
                const uint32_t slot =
                    descriptor.local_slot < 0
                        ? kDeviceMoEInvalidSlot
                        : static_cast<uint32_t>(descriptor.local_slot);
                index->summary.first_invalid_slot = slot;
                index->summary.first_invalid_layer = layer;
                index->summary.first_invalid_expert = expert;
                index->summary.first_invalid_reasons =
                    claim.valid()
                        ? rebalance_transfer_slot_directory_claim_error(
                              local_transfer_slots,
                              local_transfer_slot_count,
                              slot,
                              layer,
                              expert,
                              descriptor,
                              config)
                        : claim.invalid_reasons;
                index->summary.first_invalid_flags = descriptor.flags;
                index->summary.first_invalid_resident_mask = resident_mask;
                index->summary.first_invalid_owner =
                    descriptor.owner_participant;
            }
            __threadfence();
            index->ready = 1u;
        }
    }


    // Directory/source packing uses 256 lanes in the canonical bridge.
    __global__ __launch_bounds__(256) void pack_rebalance_directory_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        DeviceMoEExpertDirectoryEntryView *local_directory,
        DeviceMoERebalanceConfigView config)
    {
        if (!runtime_layers || !local_directory || !rebalance_config_ok(config))
            return;

        const uint32_t total = config.num_layers * config.num_experts;
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < total)
        {
            const uint32_t layer = idx / config.num_experts;
            const uint32_t expert = idx - layer * config.num_experts;
            local_directory[idx] = make_rebalance_source_entry(
                runtime_layers,
                config,
                layer,
                expert);
            idx += blockDim.x * gridDim.x;
        }
    }


    __global__ void init_rebalance_graph_controller_state_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        DeviceMoERebalanceConfigView config)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || !state)
            return;
        if (!rebalance_config_ok(config))
        {
            if (state->last_error_code == 0u)
                state->last_error_code = kDeviceMoERebalanceStatusInvalidConfig;
            return;
        }
        if (!rebalance_graph_controller_state_ok(state, config))
            init_rebalance_graph_controller_state_device(state, config);
    }

    /**
     * Start a fresh request without replacing the persistent controller buffer.
     *
     * Captured replay uses the preserving initializer above. This unconditional
     * kernel belongs only to the explicit request-reset stream after all prior
     * producers have joined it, keeping request lifetime and graph lifetime as
     * separate, visible transitions.
     */
    __global__ void reset_rebalance_graph_transaction_for_request_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        DeviceMoERebalanceWaveStateView *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        DeviceMoERebalanceConfigView config)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || !state ||
            !command_headers || !wave_states || !plan_counts)
            return;
        if (!rebalance_config_ok(config))
        {
            DeviceMoERebalanceGraphControllerStateView zero{};
            *state = zero;
            state->last_error_code =
                kDeviceMoERebalanceStatusInvalidConfig;
            return;
        }
        init_rebalance_graph_controller_state_device(state, config);
        const uint32_t count =
            rebalance_command_buffer_count(command_buffer_count);
        for (uint32_t wave = 0u; wave < count; ++wave)
        {
            command_headers[wave] =
                DeviceMoERebalanceCommandBufferHeaderView{};
            wave_states[wave] = DeviceMoERebalanceWaveStateView{};
            plan_counts[wave] = 0u;
        }
    }


    __global__ void publish_rebalance_transfer_complete_kernel(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceWaveStateView *wave_state,
        const DeviceMoERebalanceApplyStatusView *copy_status,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatusView *gathered_copy_status,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || !state)
            return;
        if (!rebalance_config_ok(config))
        {
            if (state->last_error_code == 0u)
                state->last_error_code = kDeviceMoERebalanceStatusInvalidConfig;
            return;
        }
        if (!rebalance_graph_controller_state_ok(state, config))
            init_rebalance_graph_controller_state_device(state, config);
        /*
         * A poisoned request cannot legally publish another transfer. The
         * maintenance graph remains launchable, but every replay becomes a
         * device-side no-op until request teardown replaces the controller.
         */
        if (state->last_error_code != 0u)
            return;
        ++state->maintenance_launches;
        const uint32_t wave_index =
            copy_status
                ? copy_status->transaction_wave_index
                : kDeviceMoEInvalidSlot;
        if (wave_index >=
            rebalance_command_buffer_count(command_buffer_count))
        {
            poison_rebalance_graph_controller(
                state,
                kDeviceMoERebalanceStatusInvalidRuntime,
                0u,
                copy_status ? copy_status->transaction_epoch : 0u,
                copy_status ? copy_status->transaction_command_count : 0u,
                0u,
                copy_status ? copy_status->status_code : 0u,
                config.participant_id);
            return;
        }
        if (command_header)
            command_header += wave_index;
        if (wave_state)
            wave_state += wave_index;
        if (!command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status ||
            !rebalance_command_header_ok(command_header, config))
        {
            poison_rebalance_graph_controller(
                state,
                kDeviceMoERebalanceStatusInvalidRuntime,
                wave_index,
                command_header ? command_header->epoch : 0u,
                0u,
                0u,
                copy_status ? copy_status->status_code : 0u,
                config.participant_id);
            return;
        }
        const uint32_t command_count =
            command_header->command_count <
                    (command_header->command_capacity < plan_capacity
                         ? command_header->command_capacity
                         : plan_capacity)
                ? command_header->command_count
                : (command_header->command_capacity < plan_capacity
                       ? command_header->command_capacity
                       : plan_capacity);
        if (copy_status->transaction_epoch != command_header->epoch ||
            copy_status->transaction_command_count != command_count)
        {
            poison_rebalance_graph_controller(
                state,
                kDeviceMoERebalanceStatusInvalidRuntime,
                wave_index,
                command_header->epoch,
                command_count,
                copy_status->transaction_command_count,
                copy_status->status_code,
                config.participant_id);
            return;
        }
        if (command_header->epoch == 0u || command_count == 0u)
        {
            if (wave_index < state->wave_count)
            {
                reset_rebalance_wave_progress_device(state->waves[wave_index]);
            }
            return;
        }

        auto &wave = state->waves[wave_index];
        const uint32_t requested_payload_slots = wave_state->requested_payload_slots;
        const uint32_t payload_bucket_slots = wave_state->payload_bucket_slots;
        const uint32_t payload_bucket_index = wave_state->payload_bucket_index;
        const uint32_t payload_bucket_overflow = wave_state->payload_bucket_overflow;
        if (wave.state == kDeviceMoERebalanceLifecycleReadyToApply &&
            wave.epoch != command_header->epoch)
        {
            poison_rebalance_graph_controller(
                state,
                kDeviceMoERebalanceStatusMissingTransferCompletion,
                wave_index,
                command_header->epoch,
                0u,
                0u,
                copy_status->status_code,
                config.participant_id);
            return;
        }
        const auto *wave_plan_entries =
            plan_entries + static_cast<unsigned long long>(wave_index) * plan_capacity;
        bool transfer_complete = true;
        uint32_t copied_arrivals_total = 0u;
        uint64_t copied_payload_bytes_total = 0u;
        uint32_t failing_participant = kDeviceMoEInvalidSlot;
        uint32_t failing_expected_arrivals = 0u;
        uint32_t failing_copied_arrivals = 0u;
        uint32_t failing_copy_status_code = 0u;
        uint32_t failing_copy_failure_flags = 0u;
        uint32_t failing_copy_plan_entries_seen = 0u;
        uint32_t failing_copy_skipped_wrong_destination = 0u;
        /*
         * Validate structural producer evidence before deriving destination
         * completeness.  A failed source pack necessarily causes a destination
         * to observe too few arrivals.  Reporting that downstream symptom
         * first hides the participant that actually violated the transfer
         * contract and makes a deterministic failure look like an ordering
         * race.  The allgather gives every participant the complete status
         * vector, so publication can establish the causal failure in two
         * explicit passes without host involvement or extra collectives.
         */
        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
        {
            const auto &peer_status = gathered_copy_status[participant];
            const bool transaction_matches =
                peer_status.transaction_wave_index == wave_index &&
                peer_status.transaction_epoch == command_header->epoch &&
                peer_status.transaction_command_count == command_count;
            const uint32_t copy_failure_flags =
                rebalance_copy_failure_flags(
                    peer_status,
                    transaction_matches);
            if (copy_failure_flags != 0u)
            {
                transfer_complete = false;
                failing_participant = participant;
                failing_expected_arrivals =
                    rebalance_expected_payload_arrivals(
                        wave_plan_entries,
                        command_count,
                        participant,
                        config);
                failing_copied_arrivals = peer_status.copied_arrivals;
                failing_copy_status_code =
                    transaction_matches
                        ? peer_status.status_code
                        : kDeviceMoERebalanceApplyStatusInvalidRuntime;
                failing_copy_failure_flags = copy_failure_flags;
                failing_copy_plan_entries_seen =
                    peer_status.plan_entries_seen;
                failing_copy_skipped_wrong_destination =
                    peer_status.skipped_wrong_destination;
                break;
            }
            copied_arrivals_total += peer_status.copied_arrivals;
            // This is the existing authenticated arrival collective, not a
            // byte estimate from the plan or an additional host observation.
            if (peer_status.copied_payload_bytes > UINT64_MAX - copied_payload_bytes_total)
            {
                transfer_complete = false;
                failing_participant = participant;
                failing_copy_status_code = kDeviceMoERebalanceApplyStatusInvalidRuntime;
                break;
            }
            copied_payload_bytes_total += peer_status.copied_payload_bytes;
        }
        if (transfer_complete)
        {
            for (uint32_t participant = 0;
                 participant < config.participant_count;
                 ++participant)
            {
                const auto &peer_status = gathered_copy_status[participant];
                const uint32_t expected =
                    rebalance_expected_payload_arrivals(
                        wave_plan_entries,
                        command_count,
                        participant,
                        config);
                if (peer_status.copied_arrivals < expected)
                {
                    transfer_complete = false;
                    failing_participant = participant;
                    failing_expected_arrivals = expected;
                    failing_copied_arrivals = peer_status.copied_arrivals;
                    failing_copy_status_code = peer_status.status_code;
                    failing_copy_failure_flags =
                        kDeviceMoECopyFailureArrivalShortfall;
                    failing_copy_plan_entries_seen =
                        peer_status.plan_entries_seen;
                    failing_copy_skipped_wrong_destination =
                        peer_status.skipped_wrong_destination;
                    break;
                }
            }
        }

        wave.magic = kDeviceMoERebalanceMagic;
        wave.version = kDeviceMoERebalanceVersion;
        wave.epoch = command_header->epoch;
        wave.state = kDeviceMoERebalanceLifecycleTransferInFlight;
        wave.planned_start_layer = wave_state->planned_start_layer;
        wave.planned_layer_count = wave_state->planned_layer_count;
        wave.command_count = command_count;
        wave.copied_arrivals = copied_arrivals_total;
        wave.copied_payload_bytes = copied_payload_bytes_total;
        wave.applied_arrivals = 0u;
        wave.applied_layer_count = 0u;
        wave.error_code = 0u;
        wave.requested_payload_slots = requested_payload_slots;
        wave.payload_bucket_slots = payload_bucket_slots;
        wave.payload_bucket_index = payload_bucket_index;
        wave.payload_bucket_overflow = payload_bucket_overflow;
        __threadfence();
        if (!transfer_complete)
        {
            const auto &failed_copy_status =
                gathered_copy_status[failing_participant];
            poison_rebalance_graph_controller(
                state,
                kDeviceMoERebalanceStatusMissingTransferCompletion,
                wave_index,
                command_header->epoch,
                failing_expected_arrivals,
                failing_copied_arrivals,
                failing_copy_status_code,
                failing_participant,
                failing_copy_failure_flags,
                failing_copy_plan_entries_seen,
                failing_copy_skipped_wrong_destination,
                failed_copy_status.first_missing_destination_slot,
                failed_copy_status.first_missing_destination_layer,
                failed_copy_status.first_missing_destination_expert,
                failed_copy_status.first_missing_destination_source,
                failed_copy_status.local_transfer_slot_count);
            return;
        }
        wave.state = kDeviceMoERebalanceLifecycleReadyToApply;
        state->next_epoch =
            state->next_epoch > command_header->epoch + 1u ? state->next_epoch : command_header->epoch + 1u;
    }


    __global__ __launch_bounds__(256) void project_rebalance_domain_commands_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntryView *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeaderView *local_command_headers,
        DeviceMoERebalanceStatusView *status,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveStateView *gathered_wave_states,
        DeviceMoERebalanceWaveStateView *local_wave_states,
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!gathered_plan_entries ||
            !gathered_command_headers ||
            !local_plan_entries ||
            !local_command_headers ||
            !runtime_layers ||
            !local_transfer_slots ||
            !claim_index ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u ||
            local_transfer_slot_count == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const uint32_t root_participant = config.root_participant;
        if (root_participant >= config.participant_count)
            return;

        for (uint32_t buffer_index = threadIdx.x;
             buffer_index < metadata_buffer_count;
             buffer_index += blockDim.x)
        {
            const auto &root_header =
                gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                             static_cast<unsigned long long>(metadata_buffer_count) +
                                         static_cast<unsigned long long>(buffer_index)];
            auto projected = DeviceMoERebalanceCommandBufferHeaderView{};
            projected.magic = kDeviceMoERebalanceMagic;
            projected.version = kDeviceMoERebalanceVersion;
            projected.phase = kDeviceMoERebalancePhasePlanAssignments;
            projected.command_capacity = plan_capacity;
            projected.participant_id = config.participant_id;
            projected.participant_count = config.participant_count;
            if (rebalance_command_header_for_participant_ok(
                    root_header,
                    root_participant,
                    config))
            {
                projected.epoch = root_header.epoch;
                projected.movement_decision = root_header.movement_decision;
                const uint32_t header_capacity =
                    root_header.command_capacity < plan_capacity
                        ? root_header.command_capacity
                        : plan_capacity;
                projected.command_count =
                    root_header.command_count < header_capacity
                        ? root_header.command_count
                        : header_capacity;
            }
            local_command_headers[buffer_index] = projected;

            if (gathered_wave_states && local_wave_states)
            {
                const auto &root_wave =
                    gathered_wave_states[static_cast<unsigned long long>(root_participant) *
                                             static_cast<unsigned long long>(metadata_buffer_count) +
                                         static_cast<unsigned long long>(buffer_index)];
                auto projected_wave = DeviceMoERebalanceWaveStateView{};
                projected_wave.magic = kDeviceMoERebalanceMagic;
                projected_wave.version = kDeviceMoERebalanceVersion;
                projected_wave.command_capacity = plan_capacity;
                projected_wave.participant_id = config.participant_id;
                projected_wave.participant_count = config.participant_count;
                if (root_wave.magic == kDeviceMoERebalanceMagic &&
                    root_wave.version == kDeviceMoERebalanceVersion &&
                    root_wave.participant_id == root_participant &&
                    root_wave.participant_count == config.participant_count)
                {
                    projected_wave = root_wave;
                    projected_wave.command_capacity = plan_capacity;
                    projected_wave.participant_id = config.participant_id;
                    projected_wave.participant_count = config.participant_count;
                }
                local_wave_states[buffer_index] = projected_wave;
            }
        }

        const unsigned long long total_entries =
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long idx = static_cast<unsigned long long>(threadIdx.x);
             idx < total_entries;
             idx += static_cast<unsigned long long>(blockDim.x))
        {
            local_plan_entries[idx] = DeviceMoERebalancePlanEntryView{};
        }
        __syncthreads();

        /**
         * @brief Compact root-domain transfer commands after checking physical source residency.
         *
         * The root planner decides which experts should move, but only a
         * participant with a resident descriptor in the active runtime bank can
         * pack the payload.  Copying owner-only root commands into every local
         * command buffer makes invalid sources look runnable until the pack
         * kernel dereferences empty descriptor payloads.  Filtering here keeps
         * the command buffer, wave payload metadata, and status masks aligned
         * with work that can really be transferred.
         */
        if (threadIdx.x == 0)
        {
            if (!rebalance_transfer_slot_claim_index_ok(
                    claim_index,
                    config,
                    local_transfer_slot_count))
            {
                for (uint32_t buffer_index = 0u;
                     buffer_index < metadata_buffer_count;
                     ++buffer_index)
                {
                    local_command_headers[buffer_index].command_count = 0u;
                }
                if (status)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    status->invalid_runtime_layers +=
                        claim_index->invalid_runtime_layers != 0u
                            ? claim_index->invalid_runtime_layers
                            : 1u;
                    status->plan_overflow = 1u;
                    rebalance_publish_transfer_slot_claim_summary_fields(
                        status,
                        claim_index->summary);
                }
                return;
            }
            if (status)
            {
                rebalance_publish_transfer_slot_claim_summary_fields(
                    status,
                    claim_index->summary);
            }
            uint32_t total_command_count = 0u;
            uint32_t last_projected_epoch = 0u;
            uint32_t requested_by_source[kDeviceMoEMaxParticipants] = {};
            uint32_t payload_source_participant_mask = 0u;
            uint32_t payload_destination_participant_mask = 0u;
            uint64_t payload_edge_mask = 0ULL;
            uint32_t invalid_runtime_layers = 0u;
            uint32_t destination_allocation_failures = 0u;
            for (uint32_t buffer_index = 0; buffer_index < metadata_buffer_count; ++buffer_index)
            {
                /*
                 * The two command buffers are alternative ping-pong waves, not
                 * concurrent consumers of the transfer directory. The copy
                 * ticket selects exactly one buffer for this graph replay; all
                 * buffers are projected again after that wave applies.
                 *
                 * Keep destination reservations local to one wave. The
                 * projected entries already emitted for this buffer are the
                 * exact lease ledger, so no slot-count-dependent bitmap is
                 * needed and an inactive wave cannot reserve active storage.
                 */
                uint32_t output_count = 0u;
                DeviceMoERebalancePlanEntryView *projected_wave_entries =
                    &local_plan_entries[
                        static_cast<unsigned long long>(buffer_index) *
                        static_cast<unsigned long long>(plan_capacity)];
                uint32_t buffer_requested_by_source[kDeviceMoEMaxParticipants] = {};
                const auto &root_header =
                    gathered_command_headers[static_cast<unsigned long long>(root_participant) *
                                                 static_cast<unsigned long long>(metadata_buffer_count) +
                                             static_cast<unsigned long long>(buffer_index)];
                if (rebalance_command_header_for_participant_ok(
                        root_header,
                        root_participant,
                        config))
                {
                    const uint32_t header_capacity =
                        root_header.command_capacity < plan_capacity
                            ? root_header.command_capacity
                            : plan_capacity;
                    const uint32_t command_count =
                        root_header.command_count < header_capacity
                        ? root_header.command_count
                        : header_capacity;
                    if (root_header.epoch != 0u)
                        last_projected_epoch = root_header.epoch;
                    for (uint32_t plan_index = 0; plan_index < command_count; ++plan_index)
                    {
                        const unsigned long long root_plan_index =
                            (static_cast<unsigned long long>(root_participant) *
                                 static_cast<unsigned long long>(metadata_buffer_count) +
                             static_cast<unsigned long long>(buffer_index)) *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(plan_index);
                        const auto &plan = gathered_plan_entries[root_plan_index];
                        DeviceMoERebalancePlanEntryView projected_plan = plan;
                        const bool source_in_range =
                            plan.source_participant < config.participant_count &&
                            plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                        const bool destination_in_range =
                            plan.destination_participant < config.participant_count &&
                            plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                        const uint32_t source_participant_bit =
                            source_in_range
                                ? llaminar2::moe_rebalance_policy::participantBit(
                                      plan.source_participant)
                                : 0u;
                        const uint32_t destination_participant_bit =
                            destination_in_range
                                ? llaminar2::moe_rebalance_policy::participantBit(
                                      plan.destination_participant)
                                : 0u;
                        const bool common_fields_valid =
                            rebalance_plan_applies_runtime(plan.op) &&
                            plan.layer < config.num_layers &&
                            plan.expert < config.num_experts &&
                            source_in_range &&
                            destination_in_range;
                        const bool payload_transfer_valid =
                            rebalance_plan_requires_payload(plan.op) &&
                            plan.source_participant != plan.destination_participant &&
                            plan.payload_slot != kDeviceMoEInvalidSlot &&
                            (plan.source_resident_mask & source_participant_bit) != 0u;
                        const bool resident_assignment_valid =
                            plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                            plan.payload_slot == kDeviceMoEInvalidSlot &&
                            plan.destination_slot == kDeviceMoEInvalidSlot &&
                            (plan.source_resident_mask & destination_participant_bit) != 0u;
                        const bool valid =
                            common_fields_valid &&
                            (payload_transfer_valid || resident_assignment_valid);
                        if (!valid)
                        {
                            ++invalid_runtime_layers;
                            continue;
                        }
                        if (output_count >= plan_capacity)
                        {
                            ++invalid_runtime_layers;
                            continue;
                        }

                        if (rebalance_plan_requires_payload(plan.op) &&
                            plan.destination_participant == config.participant_id &&
                            !rebalance_lease_local_transfer_slot(
                                projected_plan,
                                runtime_layers,
                                local_transfer_slots,
                                local_transfer_slot_count,
                                config,
                                projected_wave_entries,
                                output_count,
                                &gathered_plan_entries[
                                    (static_cast<unsigned long long>(root_participant) *
                                         static_cast<unsigned long long>(metadata_buffer_count) +
                                     static_cast<unsigned long long>(buffer_index)) *
                                    static_cast<unsigned long long>(plan_capacity)],
                                command_count,
                                claim_index))
                        {
                            projected_plan.destination_slot =
                                kDeviceMoEInvalidSlot;
                            projected_plan.destination_previous_layer =
                                kDeviceMoEInvalidSlot;
                            projected_plan.destination_previous_expert =
                                kDeviceMoEInvalidSlot;
                            projected_plan.destination_generation = 0u;
                            ++invalid_runtime_layers;
                            ++destination_allocation_failures;
                        }

                        const unsigned long long local_plan_index =
                            static_cast<unsigned long long>(buffer_index) *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(output_count);
                        local_plan_entries[local_plan_index] = projected_plan;
                        ++output_count;
                        ++total_command_count;

                        /*
                         * ResidentExpertAssignment is metadata-only: the
                         * destination already owns the descriptor and bytes.
                         * Retain it in the apply buffer, but never let it
                         * inflate payload bucket sizing or collective edges.
                         */
                        if (rebalance_plan_requires_payload(plan.op))
                        {
                            const uint32_t requested = plan.payload_slot + 1u;
                            if (requested_by_source[plan.source_participant] < requested)
                                requested_by_source[plan.source_participant] = requested;
                            if (buffer_requested_by_source[plan.source_participant] < requested)
                                buffer_requested_by_source[plan.source_participant] = requested;
                            payload_source_participant_mask |=
                                llaminar2::moe_rebalance_policy::participantBit(
                                    plan.source_participant);
                            payload_destination_participant_mask |=
                                llaminar2::moe_rebalance_policy::participantBit(
                                    plan.destination_participant);
                            payload_edge_mask |=
                                llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                                    plan.source_participant,
                                    plan.destination_participant,
                                    static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                        }
                    }
                }

                local_command_headers[buffer_index].command_count = output_count;
                uint32_t buffer_requested_payload_slots = 0u;
                for (uint32_t participant = 0;
                     participant < config.participant_count &&
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    if (buffer_requested_by_source[participant] > buffer_requested_payload_slots)
                        buffer_requested_payload_slots = buffer_requested_by_source[participant];
                }
                if (local_wave_states)
                {
                    auto wave = local_wave_states[buffer_index];
                    wave.requested_payload_slots = buffer_requested_payload_slots;
                    wave.payload_bucket_slots =
                        llaminar2::moe_rebalance_policy::payloadBucketSlots(
                            buffer_requested_payload_slots,
                            payload_slot_capacity);
                    wave.payload_bucket_index =
                        llaminar2::moe_rebalance_policy::payloadBucketIndex(
                            wave.payload_bucket_slots);
                    wave.payload_bucket_overflow =
                        buffer_requested_payload_slots > wave.payload_bucket_slots ? 1u : 0u;
                    local_wave_states[buffer_index] = wave;
                }
            }

            if (status)
            {
                uint32_t requested_payload_slots = 0u;
                for (uint32_t participant = 0;
                     participant < config.participant_count &&
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    if (requested_by_source[participant] > requested_payload_slots)
                        requested_payload_slots = requested_by_source[participant];
                }
                const uint32_t payload_bucket_slots =
                    llaminar2::moe_rebalance_policy::payloadBucketSlots(
                        requested_payload_slots,
                        payload_slot_capacity);
                const uint32_t payload_bucket_index =
                    llaminar2::moe_rebalance_policy::payloadBucketIndex(
                        payload_bucket_slots);
                const uint32_t payload_bucket_overflow =
                    requested_payload_slots > payload_bucket_slots ? 1u : 0u;

                if (status->magic != kDeviceMoERebalanceMagic ||
                    status->version != kDeviceMoERebalanceVersion)
                {
                    DeviceMoERebalanceStatusView zero_status{};
                    *status = zero_status;
                    status->magic = kDeviceMoERebalanceMagic;
                    status->version = kDeviceMoERebalanceVersion;
                    status->status_code = kDeviceMoERebalanceStatusOk;
                }
                if (status->status_code == kDeviceMoERebalanceStatusOk)
                {
                    status->windows_observed = status->windows_observed > 1u ? status->windows_observed : 1u;
                    status->planned_arrivals = total_command_count;
                    status->invalid_runtime_layers += invalid_runtime_layers;
                    status->capacity_limited_candidates +=
                        destination_allocation_failures;
                    if (destination_allocation_failures != 0u)
                        status->plan_overflow = 1u;
                    if (total_command_count != 0u)
                    {
                        status->windows_applied = status->windows_applied > 1u ? status->windows_applied : 1u;
                        if (status->last_epoch == 0u)
                            status->last_epoch = last_projected_epoch;
                    }
                    else
                    {
                        status->windows_applied = 0u;
                    }
                    status->payload_bucket_requested_slots = requested_payload_slots;
                    status->payload_bucket_slots = payload_bucket_slots;
                    status->payload_bucket_index = payload_bucket_index;
                    status->payload_bucket_overflow =
                        payload_bucket_overflow != 0u ||
                                destination_allocation_failures != 0u
                            ? 1u
                            : 0u;
                    status->payload_source_participant_mask = payload_source_participant_mask;
                    status->payload_destination_participant_mask =
                        payload_destination_participant_mask;
                    status->payload_edge_mask = payload_edge_mask;
                }
            }
        }
    }

    __global__ void project_prefill_llep_domain_commands_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntryView *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeaderView *local_command_headers,
        DeviceMoERebalanceStatusView *status,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceConfigView config,
        uint32_t command_buffer_count,
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!gathered_plan_entries ||
            !gathered_command_headers ||
            !local_plan_entries ||
            !local_plan_count ||
            !local_command_headers ||
            !status ||
            !runtime_layers ||
            !local_transfer_slots ||
            !claim_index ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u ||
            payload_slot_capacity == 0u ||
            local_transfer_slot_count == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const unsigned long long total_entries =
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long idx = static_cast<unsigned long long>(threadIdx.x);
             idx < total_entries;
             idx += static_cast<unsigned long long>(blockDim.x))
        {
            local_plan_entries[idx] = DeviceMoERebalancePlanEntryView{};
        }
        for (uint32_t buffer_index = threadIdx.x;
             buffer_index < metadata_buffer_count;
             buffer_index += blockDim.x)
        {
            DeviceMoERebalanceCommandBufferHeaderView header{};
            header.magic = kDeviceMoERebalanceMagic;
            header.version = kDeviceMoERebalanceVersion;
            header.phase = kDeviceMoERebalancePhasePlanAssignments;
            header.command_capacity = plan_capacity;
            header.participant_id = config.participant_id;
            header.participant_count = config.participant_count;
            local_command_headers[buffer_index] = header;
            local_plan_count[buffer_index] = 0u;
        }
        __syncthreads();

        if (threadIdx.x != 0u)
            return;

        if (!rebalance_transfer_slot_claim_index_ok(
                claim_index,
                config,
                local_transfer_slot_count))
        {
            DeviceMoERebalanceStatusView invalid_status{};
            invalid_status.magic = kDeviceMoERebalanceMagic;
            invalid_status.version = kDeviceMoERebalanceVersion;
            invalid_status.status_code =
                kDeviceMoERebalanceStatusInvalidRuntime;
            invalid_status.invalid_runtime_layers =
                claim_index->invalid_runtime_layers != 0u
                    ? claim_index->invalid_runtime_layers
                    : 1u;
            invalid_status.plan_overflow = 1u;
            rebalance_publish_transfer_slot_claim_summary_fields(
                &invalid_status,
                claim_index->summary);
            *status = invalid_status;
            return;
        }

        const DeviceMoERebalanceStatusView prior_status = *status;
        DeviceMoERebalanceStatusView projected_status{};
        projected_status.magic = kDeviceMoERebalanceMagic;
        projected_status.version = kDeviceMoERebalanceVersion;
        projected_status.status_code =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.status_code
                : kDeviceMoERebalanceStatusOk;
        projected_status.windows_observed = 1u;
        projected_status.plan_overflow =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.plan_overflow
                : 0u;
        projected_status.payload_bucket_overflow =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.payload_bucket_overflow
                : 0u;
        projected_status.capacity_limited_candidates =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.capacity_limited_candidates
                : 0u;
        projected_status.prefill_active_transfer_slot_experts =
            (prior_status.magic == kDeviceMoERebalanceMagic &&
             prior_status.version == kDeviceMoERebalanceVersion)
                ? prior_status.prefill_active_transfer_slot_experts
                : 0u;
        projected_status.prefill_unique_transfer_slot_claims =
            prior_status.prefill_unique_transfer_slot_claims;
        projected_status.prefill_duplicate_transfer_slot_claims =
            prior_status.prefill_duplicate_transfer_slot_claims;
        projected_status.prefill_invalid_transfer_slot_claims =
            prior_status.prefill_invalid_transfer_slot_claims;
        projected_status.prefill_max_transfer_slot =
            prior_status.prefill_max_transfer_slot;
        projected_status.prefill_max_transfer_slot_layer =
            prior_status.prefill_max_transfer_slot_layer;
        projected_status.prefill_max_transfer_slot_expert =
            prior_status.prefill_max_transfer_slot_expert;
        projected_status.prefill_first_duplicate_transfer_slot =
            prior_status.prefill_first_duplicate_transfer_slot;
        projected_status.prefill_first_duplicate_layer =
            prior_status.prefill_first_duplicate_layer;
        projected_status.prefill_first_duplicate_expert =
            prior_status.prefill_first_duplicate_expert;
        rebalance_publish_transfer_slot_claim_summary_fields(
            &projected_status,
            claim_index->summary);

        uint32_t total_command_count = 0u;
        uint32_t last_epoch = 0u;
        uint32_t requested_payload_slots = 0u;
        uint32_t payload_source_participant_mask = 0u;
        uint32_t payload_destination_participant_mask = 0u;
        uint64_t payload_edge_mask = 0ULL;
        for (uint32_t buffer_index = 0u;
             buffer_index < metadata_buffer_count;
             ++buffer_index)
        {
            uint32_t source_payload_counts[kDeviceMoEMaxParticipants] = {};
            uint32_t output_count = 0u;
            uint32_t buffer_epoch = 0u;
            DeviceMoERebalancePlanEntryView *projected_wave_entries =
                &local_plan_entries[
                    static_cast<unsigned long long>(buffer_index) *
                    static_cast<unsigned long long>(plan_capacity)];

            for (uint32_t participant = 0u;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                const unsigned long long participant_buffer_base =
                    static_cast<unsigned long long>(participant) *
                        static_cast<unsigned long long>(metadata_buffer_count) +
                    static_cast<unsigned long long>(buffer_index);
                const auto &header =
                    gathered_command_headers[participant_buffer_base];
                if (!rebalance_command_header_for_participant_ok(
                        header,
                        participant,
                        config))
                {
                    continue;
                }
                if (header.epoch != 0u)
                    buffer_epoch = max(buffer_epoch, header.epoch);
                const uint32_t command_count =
                    min(header.command_count,
                        min(header.command_capacity, plan_capacity));
                for (uint32_t plan_index = 0u;
                     plan_index < command_count;
                     ++plan_index)
                {
                    const auto &plan =
                        gathered_plan_entries[
                            participant_buffer_base *
                                static_cast<unsigned long long>(plan_capacity) +
                            static_cast<unsigned long long>(plan_index)];
                    const uint32_t source_participant_bit =
                        llaminar2::moe_rebalance_policy::participantBit(
                            plan.source_participant);
                    const bool valid =
                        rebalance_plan_requires_payload(plan.op) &&
                        plan.layer < config.num_layers &&
                        plan.expert < config.num_experts &&
                        plan.source_participant < config.participant_count &&
                        plan.destination_participant == participant &&
                        plan.destination_participant < config.participant_count &&
                        plan.source_participant != plan.destination_participant &&
                        plan.source_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        plan.destination_participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                        (plan.source_resident_mask & source_participant_bit) != 0u;
                    if (!valid)
                    {
                        ++projected_status.invalid_runtime_layers;
                        continue;
                    }

                    if (output_count >= plan_capacity ||
                        source_payload_counts[plan.source_participant] >=
                            payload_slot_capacity)
                    {
                        projected_status.plan_overflow = 1u;
                        projected_status.payload_bucket_overflow = 1u;
                        continue;
                    }

                    DeviceMoERebalancePlanEntryView projected = plan;
                    /*
                     * Every gathered command is logical, including portable
                     * prefix-runtime rehydration. Rolling transfer-directory
                     * indices are graph-lifetime allocator state and never
                     * cross the RAM/disk prefix boundary. Only the destination
                     * participant can lease stable VRAM because only it sees
                     * the complete local directory and active runtime claims.
                     */
                    projected.destination_slot = kDeviceMoEInvalidSlot;
                    projected.destination_previous_layer =
                        kDeviceMoEInvalidSlot;
                    projected.destination_previous_expert =
                        kDeviceMoEInvalidSlot;
                    projected.destination_generation = 0u;
                    if (projected.destination_participant ==
                            config.participant_id &&
                        !rebalance_lease_prefill_transfer_slot(
                            projected,
                            runtime_layers,
                            local_transfer_slots,
                            local_transfer_slot_count,
                            config,
                            projected_wave_entries,
                            output_count,
                            claim_index))
                    {
                        projected_status.plan_overflow = 1u;
                        projected_status.payload_bucket_overflow = 1u;
                        ++projected_status.capacity_limited_candidates;
                        ++projected_status.invalid_runtime_layers;
                    }
                    projected.payload_slot =
                        source_payload_counts[plan.source_participant]++;

                    projected_wave_entries[output_count] = projected;
                    ++output_count;

                    total_command_count += 1u;
                    requested_payload_slots =
                        max(requested_payload_slots, projected.payload_slot + 1u);
                    payload_source_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            projected.source_participant);
                    payload_destination_participant_mask |=
                        llaminar2::moe_rebalance_policy::participantBit(
                            projected.destination_participant);
                    payload_edge_mask |=
                        llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                            projected.source_participant,
                            projected.destination_participant,
                            static_cast<uint32_t>(kDeviceMoEMaxParticipants));
                }
            }

            local_plan_count[buffer_index] = output_count;
            local_command_headers[buffer_index].epoch = buffer_epoch;
            local_command_headers[buffer_index].command_count = output_count;
            last_epoch = max(last_epoch, buffer_epoch);
        }

        const uint32_t payload_bucket_slots =
            llaminar2::moe_rebalance_policy::payloadBucketSlots(
                requested_payload_slots,
                payload_slot_capacity);
        projected_status.last_epoch =
            prior_status.last_epoch != 0u ? prior_status.last_epoch : last_epoch;
        projected_status.planned_arrivals = total_command_count;
        projected_status.windows_applied =
            total_command_count != 0u ? 1u : 0u;
        projected_status.payload_bucket_requested_slots = requested_payload_slots;
        projected_status.payload_bucket_slots = payload_bucket_slots;
        projected_status.payload_bucket_index =
            llaminar2::moe_rebalance_policy::payloadBucketIndex(
                payload_bucket_slots);
        if (requested_payload_slots > payload_bucket_slots ||
            projected_status.plan_overflow != 0u)
        {
            projected_status.payload_bucket_overflow = 1u;
        }
        projected_status.payload_source_participant_mask =
            payload_source_participant_mask;
        projected_status.payload_destination_participant_mask =
            payload_destination_participant_mask;
        projected_status.payload_edge_mask = payload_edge_mask;
        projected_status.candidate_arrivals_considered = total_command_count;
        projected_status.llep_assignment_span_count =
            prior_status.llep_assignment_span_count;
        projected_status.llep_weight_transfer_count =
            prior_status.llep_weight_transfer_count;
        *status = projected_status;
    }

    /**
     * @brief Expand one deterministic current-batch plan into all participant slices.
     *
     * Every homogeneous LocalTP participant owns the same transfer list because
     * routing rows, planner arithmetic, and the logical residency masks are
     * mirrored. Building the participant-major request envelope locally avoids
     * two fixed-latency metadata collectives per MoE layer. Physical destination
     * slots are deliberately left unresolved for the following projection
     * kernel, which remains participant-local.
     */
    __global__ void materialize_prefill_llep_mirrored_domain_commands_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceMoERebalancePlanEntryView *__restrict__ mirrored_plan_entries,
        DeviceMoERebalanceCommandBufferHeaderView *__restrict__ mirrored_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalanceStatusView *__restrict__ status,
        DeviceMoERebalanceConfigView config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx)
    {
        const uint32_t lane = threadIdx.x;
        const unsigned long long total_entries =
            static_cast<unsigned long long>(config.participant_count) *
            static_cast<unsigned long long>(plan_capacity);
        for (unsigned long long index = lane;
             index < total_entries;
             index += blockDim.x)
        {
            mirrored_plan_entries[index] = DeviceMoERebalancePlanEntryView{};
        }
        for (uint32_t participant = lane;
             participant < config.participant_count;
             participant += blockDim.x)
        {
            DeviceMoERebalanceCommandBufferHeaderView header{};
            header.magic = kDeviceMoERebalanceMagic;
            header.version = kDeviceMoERebalanceVersion;
            header.phase = kDeviceMoERebalancePhasePlanAssignments;
            header.command_capacity = plan_capacity;
            header.participant_id = participant;
            header.participant_count = config.participant_count;
            mirrored_command_headers[participant] = header;
        }
        if (lane == 0u)
        {
            *status = DeviceMoERebalanceStatusView{};
            status->magic = kDeviceMoERebalanceMagic;
            status->version = kDeviceMoERebalanceVersion;
            status->status_code = kDeviceMoERebalanceStatusOk;
        }
        __syncthreads();

        if (lane != 0u)
            return;

        if (!runtime || !mirrored_plan_entries ||
            !mirrored_command_headers || !status ||
            !rebalance_config_ok(config) || plan_capacity == 0u ||
            payload_slot_capacity == 0u ||
            config.participant_count == 0u ||
            config.participant_count >
                static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
            layer_idx >= config.num_layers ||
            runtime_execution_bank(runtime) > 1u ||
            runtime->participant_count != config.participant_count ||
            !runtime->reserved_ptrs[2])
        {
            status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
            status->invalid_runtime_layers = 1u;
            return;
        }

        const uint32_t transfer_count =
            runtime->reserved_u64[3] > 0xffffffffULL
                ? 0xffffffffu
                : static_cast<uint32_t>(runtime->reserved_u64[3]);
        const uint32_t span_count =
            runtime->reserved_u64[2] > 0xffffffffULL
                ? 0xffffffffu
                : static_cast<uint32_t>(runtime->reserved_u64[2]);
        status->llep_assignment_span_count = span_count;
        status->llep_weight_transfer_count = transfer_count;
        if (transfer_count > static_cast<uint32_t>(kDeviceMoEMaxExperts))
        {
            status->plan_overflow = 1u;
            status->payload_bucket_overflow = 1u;
            fail_fast_incomplete_llep_transfer();
            return;
        }

        const auto *transfers =
            static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
                runtime->reserved_ptrs[2]);
        const auto &active_bank =
            runtime_placement_banks(runtime)[runtime_execution_bank(runtime)];
        uint32_t command_counts[kDeviceMoEMaxParticipants] = {};
        uint32_t source_payload_counts[kDeviceMoEMaxParticipants]
                                      [kDeviceMoEMaxParticipants] = {};
        uint32_t total_command_count = 0u;
        uint32_t requested_payload_slots = 0u;
        uint32_t overflow = 0u;
        uint32_t source_mask = 0u;
        uint32_t destination_mask = 0u;
        uint64_t edge_mask = 0ULL;

        for (uint32_t transfer_index = 0u;
             transfer_index < transfer_count;
             ++transfer_index)
        {
            const auto transfer = transfers[transfer_index];
            uint32_t resident_mask = 0u;
            if (transfer.expert < static_cast<uint32_t>(kDeviceMoEMaxExperts))
            {
                resident_mask =
                    active_bank.resident_participant_mask[transfer.expert];
            }
            const uint32_t source_bit =
                llaminar2::moe_rebalance_policy::participantBit(
                    transfer.source_participant);
            const bool valid =
                transfer.expert < config.num_experts &&
                transfer.source_participant < config.participant_count &&
                transfer.destination_participant < config.participant_count &&
                transfer.source_participant != transfer.destination_participant &&
                (resident_mask & source_bit) != 0u;
            if (!valid)
            {
                overflow = 1u;
                ++status->invalid_runtime_layers;
                continue;
            }

            const uint32_t destination = transfer.destination_participant;
            const uint32_t source = transfer.source_participant;
            const uint32_t output_index = command_counts[destination];
            const uint32_t payload_slot =
                source_payload_counts[destination][source];
            if (output_index >= plan_capacity ||
                payload_slot >= payload_slot_capacity)
            {
                overflow = 1u;
                continue;
            }
            ++command_counts[destination];
            ++source_payload_counts[destination][source];

            DeviceMoERebalancePlanEntryView entry{};
            entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
            entry.flags = kDeviceMoERebalancePlanFlagCurrentBatchLLEP;
            entry.layer = layer_idx;
            entry.expert = transfer.expert;
            entry.source_participant = source;
            entry.destination_participant = destination;
            entry.source_resident_mask = resident_mask;
            entry.destination_slot = kDeviceMoEInvalidSlot;
            entry.payload_slot = payload_slot;
            mirrored_plan_entries[
                static_cast<unsigned long long>(destination) *
                    static_cast<unsigned long long>(plan_capacity) +
                static_cast<unsigned long long>(output_index)] = entry;

            ++total_command_count;
            requested_payload_slots =
                max(requested_payload_slots, payload_slot + 1u);
            source_mask |=
                llaminar2::moe_rebalance_policy::participantBit(source);
            destination_mask |=
                llaminar2::moe_rebalance_policy::participantBit(destination);
            edge_mask |=
                llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                    source,
                    destination,
                    static_cast<uint32_t>(kDeviceMoEMaxParticipants));
        }

        const uint32_t epoch = runtime_command_epoch(runtime);
        for (uint32_t participant = 0u;
             participant < config.participant_count;
             ++participant)
        {
            mirrored_command_headers[participant].epoch = epoch;
            mirrored_command_headers[participant].command_count =
                command_counts[participant];
        }
        const uint32_t payload_bucket_slots =
            llaminar2::moe_rebalance_policy::payloadBucketSlots(
                requested_payload_slots,
                payload_slot_capacity);
        status->last_epoch = epoch;
        status->planned_arrivals = total_command_count;
        status->plan_overflow = overflow;
        status->payload_bucket_requested_slots = requested_payload_slots;
        status->payload_bucket_slots = payload_bucket_slots;
        status->payload_bucket_index =
            llaminar2::moe_rebalance_policy::payloadBucketIndex(
                payload_bucket_slots);
        status->payload_bucket_overflow =
            overflow != 0u || requested_payload_slots > payload_bucket_slots
                ? 1u
                : 0u;
        status->payload_source_participant_mask = source_mask;
        status->payload_destination_participant_mask = destination_mask;
        status->payload_edge_mask = edge_mask;
    }

    __global__ void materialize_prefill_llep_transfer_commands_kernel(
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        DeviceMoERebalancePlanEntryView *__restrict__ plan_entries,
        uint32_t *__restrict__ plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *__restrict__ command_headers,
        DeviceMoERebalanceStatusView *__restrict__ status,
        DeviceMoERebalanceConfigView config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        __shared__ uint32_t source_payload_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t compact_command_count;
        __shared__ uint32_t overflow;
        __shared__ uint32_t valid;
        __shared__ uint32_t transfer_count;
        __shared__ uint32_t span_count;
        __shared__ uint32_t fast_payload_source_participant_mask;
        __shared__ uint32_t fast_payload_destination_participant_mask;
        __shared__ unsigned long long fast_payload_edge_mask;
        __shared__ uint32_t shared_transfer_experts[256];
        __shared__ uint32_t shared_transfer_sources[256];
        __shared__ uint32_t shared_transfer_destinations[256];
        __shared__ uint32_t shared_transfer_valid[256];
        __shared__ uint32_t shared_transfer_output_indices[256];
        __shared__ uint32_t shared_transfer_payload_slots[256];

        const uint32_t lane = threadIdx.x;
        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);

        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
             participant += blockDim.x)
        {
            source_payload_counts[participant] = 0u;
        }

        for (uint32_t idx = lane; idx < metadata_buffer_count; idx += blockDim.x)
        {
            DeviceMoERebalanceCommandBufferHeaderView header{};
            header.magic = kDeviceMoERebalanceMagic;
            header.version = kDeviceMoERebalanceVersion;
            header.phase = kDeviceMoERebalancePhasePlanAssignments;
            header.command_capacity = plan_capacity;
            header.participant_id = config.participant_id;
            header.participant_count = config.participant_count;
            command_headers[idx] = header;
        }

        if (lane == 0u)
        {
            compact_command_count = 0u;
            overflow = 0u;
            valid = 0u;
            transfer_count = 0u;
            span_count = 0u;
            fast_payload_source_participant_mask = 0u;
            fast_payload_destination_participant_mask = 0u;
            fast_payload_edge_mask = 0ULL;
            if (plan_count)
                *plan_count = 0u;
            if (status)
            {
                *status = DeviceMoERebalanceStatusView{};
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
            }

            if (runtime &&
                plan_entries &&
                plan_count &&
                command_headers &&
                status &&
                rebalance_config_ok(config) &&
                payload_slot_capacity > 0u &&
                payload_slot_capacity <=
                    static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
                plan_capacity > 0u &&
                layer_idx < config.num_layers &&
                runtime_execution_bank(runtime) <= 1u &&
                runtime->participant_count == config.participant_count &&
                runtime->participant_count <= static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                runtime->reserved_ptrs[2])
            {
                transfer_count =
                    runtime->reserved_u64[3] > 0xffffffffULL
                        ? 0xffffffffu
                        : static_cast<uint32_t>(runtime->reserved_u64[3]);
                span_count =
                    runtime->reserved_u64[2] > 0xffffffffULL
                        ? 0xffffffffu
                        : static_cast<uint32_t>(runtime->reserved_u64[2]);
                valid = 1u;
            }
            else if (status)
            {
                valid = 0u;
                status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
                status->invalid_runtime_layers = 1u;
            }
        }
        __syncthreads();

        if (valid == 0u)
            return;

        const auto *transfers =
            static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
                runtime->reserved_ptrs[2]);
    const uint32_t execution_bank = runtime_execution_bank(runtime);
    const auto &active_bank =
        runtime_placement_banks(runtime)[execution_bank];

    if (transfer_count <= static_cast<uint32_t>(blockDim.x))
    {
        if (lane < transfer_count)
        {
            const auto transfer = transfers[lane];
            shared_transfer_experts[lane] = transfer.expert;
            shared_transfer_sources[lane] = transfer.source_participant;
            shared_transfer_destinations[lane] = transfer.destination_participant;
        }
        __syncthreads();

        if (lane == 0u)
        {
            for (uint32_t transfer_idx = 0u;
                 transfer_idx < transfer_count;
                 ++transfer_idx)
            {
                shared_transfer_valid[transfer_idx] = 0u;
                shared_transfer_output_indices[transfer_idx] =
                    kDeviceMoEInvalidSlot;
                shared_transfer_payload_slots[transfer_idx] =
                    kDeviceMoEInvalidSlot;

                const uint32_t transfer_expert =
                    shared_transfer_experts[transfer_idx];
                const uint32_t transfer_source =
                    shared_transfer_sources[transfer_idx];
                const uint32_t transfer_destination =
                    shared_transfer_destinations[transfer_idx];
                uint32_t resident_mask = 0u;
                if (transfer_expert <
                    static_cast<uint32_t>(kDeviceMoEMaxExperts))
                {
                    resident_mask =
                        active_bank
                            .resident_participant_mask[transfer_expert];
                }
                const uint32_t transfer_source_bit =
                    llaminar2::moe_rebalance_policy::participantBit(
                        transfer_source);
                const bool transfer_valid =
                    transfer_expert < config.num_experts &&
                    transfer_source < config.participant_count &&
                    transfer_destination < config.participant_count &&
                    transfer_source <
                        static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    transfer_destination <
                        static_cast<uint32_t>(kDeviceMoEMaxParticipants) &&
                    transfer_source != transfer_destination &&
                    (resident_mask & transfer_source_bit) != 0u;
                if (!transfer_valid)
                {
                    overflow = 1u;
                    continue;
                }
                /* Materialization is the pre-allgather request phase: a
                 * participant publishes only arrivals it will receive.
                 * Source-side packing is derived after the allgather by
                 * domain projection, which merges these destination-local
                 * requests without duplicating global transfers in every
                 * participant buffer. */
                if (transfer_destination != config.participant_id)
                    continue;
                if (compact_command_count >= plan_capacity)
                {
                    overflow = 1u;
                    continue;
                }

                const uint32_t payload_slot =
                    source_payload_counts[transfer_source];
                if (payload_slot >= payload_slot_capacity)
                {
                    overflow = 1u;
                    continue;
                }
                ++source_payload_counts[transfer_source];

                shared_transfer_valid[transfer_idx] = 1u;
                shared_transfer_output_indices[transfer_idx] =
                    compact_command_count++;
                shared_transfer_payload_slots[transfer_idx] =
                    payload_slot;
                fast_payload_source_participant_mask |=
                    llaminar2::moe_rebalance_policy::participantBit(
                        transfer_source);
                fast_payload_destination_participant_mask |=
                    llaminar2::moe_rebalance_policy::participantBit(
                        transfer_destination);
                fast_payload_edge_mask |=
                    static_cast<unsigned long long>(
                        llaminar2::moe_rebalance_policy::directedParticipantEdgeBit(
                            transfer_source,
                            transfer_destination,
                            static_cast<uint32_t>(
                                kDeviceMoEMaxParticipants)));
            }
        }
        __syncthreads();

        for (uint32_t transfer_idx = lane;
             transfer_idx < transfer_count;
             transfer_idx += blockDim.x)
        {
            if (shared_transfer_valid[transfer_idx] == 0u)
                continue;

            const uint32_t transfer_expert =
                shared_transfer_experts[transfer_idx];
            const uint32_t transfer_source =
                shared_transfer_sources[transfer_idx];
            const uint32_t transfer_destination =
                shared_transfer_destinations[transfer_idx];
            DeviceMoERebalancePlanEntryView entry{};
            entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
            entry.flags = kDeviceMoERebalancePlanFlagCurrentBatchLLEP;
            entry.layer = layer_idx;
            entry.expert = transfer_expert;
            entry.source_participant = transfer_source;
            entry.destination_participant = transfer_destination;
            entry.source_resident_mask =
                active_bank.resident_participant_mask[transfer_expert];
            entry.destination_slot = kDeviceMoEInvalidSlot;
            entry.payload_slot =
                shared_transfer_payload_slots[transfer_idx];
            plan_entries[
                shared_transfer_output_indices[transfer_idx]] = entry;
        }
        __syncthreads();

        if (lane == 0u)
        {
            const uint32_t command_count =
                compact_command_count < plan_capacity
                    ? compact_command_count
                    : plan_capacity;
            *plan_count = command_count;
            command_headers[0].command_count = command_count;
            command_headers[0].epoch = runtime_command_epoch(runtime);

            uint32_t requested_payload_slots = 0u;
            for (uint32_t participant = 0u;
                 participant < config.participant_count &&
                 participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                 ++participant)
            {
                if (requested_payload_slots < source_payload_counts[participant])
                    requested_payload_slots = source_payload_counts[participant];
            }
            const uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);

            status->last_epoch = command_headers[0].epoch;
            status->planned_arrivals = command_count;
            status->plan_overflow = overflow != 0u ? 1u : 0u;
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            status->payload_bucket_overflow =
                requested_payload_slots > payload_bucket_slots ||
                        status->plan_overflow != 0u
                    ? 1u
                    : 0u;
            status->payload_source_participant_mask =
                fast_payload_source_participant_mask;
            status->payload_destination_participant_mask =
                fast_payload_destination_participant_mask;
            status->payload_edge_mask = static_cast<uint64_t>(fast_payload_edge_mask);
            status->llep_assignment_span_count = span_count;
            status->llep_weight_transfer_count = transfer_count;
        }
        return;
    }

    if (lane == 0u)
    {
        *plan_count = 0u;
        command_headers[0].command_count = 0u;
        command_headers[0].epoch = runtime_command_epoch(runtime);
        status->last_epoch = command_headers[0].epoch;
        status->planned_arrivals = 0u;
        status->plan_overflow = 1u;
        status->payload_bucket_overflow = 1u;
        status->llep_assignment_span_count = span_count;
        status->llep_weight_transfer_count = transfer_count;
        fail_fast_incomplete_llep_transfer();
    }
    }

    __global__ __launch_bounds__(256) void pack_rebalance_source_descriptors_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntryView *local_source_descriptors,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        (void)controller_state;
        if (!runtime_layers ||
            !plan_entries ||
            !command_headers ||
            !local_source_descriptors ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u)
        {
            return;
        }

        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const unsigned long long total =
            static_cast<unsigned long long>(config.participant_count) *
            static_cast<unsigned long long>(metadata_buffer_count) *
            static_cast<unsigned long long>(plan_capacity);
        unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) *
                                 static_cast<unsigned long long>(blockDim.x) +
                                 static_cast<unsigned long long>(threadIdx.x);
        while (idx < total)
        {
            local_source_descriptors[idx] = DeviceMoEExpertDirectoryEntryView{};

            const uint32_t plan_index =
                static_cast<uint32_t>(idx % static_cast<unsigned long long>(plan_capacity));
            const unsigned long long tmp =
                idx / static_cast<unsigned long long>(plan_capacity);
            const uint32_t command_buffer_index =
                static_cast<uint32_t>(tmp % static_cast<unsigned long long>(metadata_buffer_count));
            const uint32_t destination_participant =
                static_cast<uint32_t>(tmp / static_cast<unsigned long long>(metadata_buffer_count));
            const auto &header =
                command_headers[static_cast<unsigned long long>(command_buffer_index)];
            if (!rebalance_command_header_ok(&header, config))
            {
                idx += static_cast<unsigned long long>(blockDim.x) *
                       static_cast<unsigned long long>(gridDim.x);
                continue;
            }
            const uint32_t header_capacity =
                header.command_capacity < plan_capacity ? header.command_capacity : plan_capacity;
            const uint32_t command_count =
                header.command_count < header_capacity ? header.command_count : header_capacity;
            if (plan_index >= command_count)
            {
                idx += static_cast<unsigned long long>(blockDim.x) *
                       static_cast<unsigned long long>(gridDim.x);
                continue;
            }

            const auto &plan =
                plan_entries[static_cast<unsigned long long>(command_buffer_index) *
                                 static_cast<unsigned long long>(plan_capacity) +
                             static_cast<unsigned long long>(plan_index)];
            if (rebalance_plan_requires_payload(plan.op) &&
                plan.source_participant == config.participant_id &&
                plan.destination_participant == destination_participant &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts)
            {
                local_source_descriptors[idx] = make_rebalance_source_entry(
                    runtime_layers,
                    config,
                    plan.layer,
                    plan.expert);
            }

            idx += static_cast<unsigned long long>(blockDim.x) *
                   static_cast<unsigned long long>(gridDim.x);
        }
    }

} // namespace

extern "C"
{
bool hipMoE_pack_rebalance_directory(
    void *runtime_layers,
    void *local_directory,
    const void *config,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !local_directory || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const int total = static_cast<int>(cfg.num_layers * cfg.num_experts);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(pack_rebalance_directory_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<DeviceMoEExpertDirectoryEntryView *>(local_directory),
        cfg);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_directory kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_pack_rebalance_source_descriptors(
    void *runtime_layers,
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_source_descriptors,
    const void *config,
    void *controller_state,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !gathered_plan_entries || !gathered_command_headers ||
        !local_source_descriptors || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const uint32_t metadata_buffer_count =
        command_buffer_count <= 1u ? 1u : (command_buffer_count < 2u ? command_buffer_count : 2u);
    const unsigned long long total =
        static_cast<unsigned long long>(cfg.participant_count) *
        static_cast<unsigned long long>(metadata_buffer_count) *
        static_cast<unsigned long long>(plan_capacity);
    const int threads = 256;
    const int blocks =
        static_cast<int>((total + static_cast<unsigned long long>(threads) - 1ULL) /
                         static_cast<unsigned long long>(threads));
    if (blocks > 0)
    {
        hipLaunchKernelGGL(pack_rebalance_source_descriptors_kernel,
            dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
            plan_capacity,
            static_cast<DeviceMoEExpertDirectoryEntryView *>(local_source_descriptors),
            cfg,
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_source_descriptors kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_project_rebalance_domain_commands(
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_plan_entries,
    void *local_command_headers,
    const void *config,
    void *status,
    uint32_t payload_slot_capacity,
    uint32_t command_buffer_count,
    const void *gathered_wave_states,
    void *local_wave_states,
    void *runtime_layers,
    const void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
    void *transfer_slot_claim_index,
    int device_idx,
    void *stream)
{
    if (!gathered_plan_entries || !gathered_command_headers ||
        !local_plan_entries || !local_command_headers ||
        !config || !runtime_layers || !local_transfer_slots ||
        !transfer_slot_claim_index ||
        !stream || plan_capacity == 0u || local_transfer_slot_count == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(build_transfer_slot_claim_index_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(
            local_transfer_slots),
        local_transfer_slot_count,
        cfg,
        static_cast<DeviceMoETransferSlotClaimIndexView *>(
            transfer_slot_claim_index));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE transfer-slot claim-index preflight kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    hipLaunchKernelGGL(project_rebalance_domain_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
        plan_capacity,
        static_cast<DeviceMoERebalancePlanEntryView *>(local_plan_entries),
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(local_command_headers),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        payload_slot_capacity,
        cfg,
        command_buffer_count,
        static_cast<const DeviceMoERebalanceWaveStateView *>(gathered_wave_states),
        static_cast<DeviceMoERebalanceWaveStateView *>(local_wave_states),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
        local_transfer_slot_count,
        static_cast<const DeviceMoETransferSlotClaimIndexView *>(
            transfer_slot_claim_index));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE project_rebalance_domain_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_project_prefill_llep_domain_commands(
    const void *gathered_plan_entries,
    const void *gathered_command_headers,
    uint32_t plan_capacity,
    void *local_plan_entries,
    uint32_t *local_plan_count,
    void *local_command_header,
    const void *config,
    void *status,
    uint32_t payload_slot_capacity,
    uint32_t command_buffer_count,
    void *runtime_layers,
    const void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
    void *transfer_slot_claim_index,
    int device_idx,
    void *stream)
{
    if (!gathered_plan_entries || !gathered_command_headers ||
        !local_plan_entries || !local_plan_count || !local_command_header ||
        !config || !status || !runtime_layers || !local_transfer_slots ||
        !transfer_slot_claim_index ||
        !stream || plan_capacity == 0u || payload_slot_capacity == 0u ||
        local_transfer_slot_count == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(build_transfer_slot_claim_index_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(
            local_transfer_slots),
        local_transfer_slot_count,
        cfg,
        static_cast<DeviceMoETransferSlotClaimIndexView *>(
            transfer_slot_claim_index));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE transfer-slot claim-index preflight kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    hipLaunchKernelGGL(project_prefill_llep_domain_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
        static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
        plan_capacity,
        static_cast<DeviceMoERebalancePlanEntryView *>(local_plan_entries),
        local_plan_count,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(local_command_header),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        payload_slot_capacity,
        cfg,
        command_buffer_count,
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        static_cast<const DeviceMoEExpertDirectoryEntryView *>(
            local_transfer_slots),
        local_transfer_slot_count,
        static_cast<const DeviceMoETransferSlotClaimIndexView *>(
            transfer_slot_claim_index));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE project_prefill_llep_domain_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_materialize_prefill_llep_transfer_commands(
    const void *runtime_layer,
    void *plan_entries,
    uint32_t *plan_count,
    uint32_t plan_capacity,
    void *command_header,
    void *status,
    const void *config,
    uint32_t payload_slot_capacity,
    uint32_t layer_idx,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layer || !plan_entries || !plan_count || !command_header ||
        !status || !config || !stream || plan_capacity == 0u ||
        payload_slot_capacity == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(materialize_prefill_llep_transfer_commands_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layer),
        static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
        plan_capacity,
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
        static_cast<DeviceMoERebalanceStatusView *>(status),
        cfg,
        payload_slot_capacity,
        layer_idx,
        command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE materialize_prefill_llep_transfer_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_materialize_prefill_llep_mirrored_domain_commands(
    const void *runtime_layer,
    void *mirrored_plan_entries,
    void *mirrored_command_headers,
    uint32_t plan_capacity,
    void *status,
    const void *config,
    uint32_t payload_slot_capacity,
    uint32_t layer_idx,
    int device_idx,
    void *stream)
{
    if (!runtime_layer || !mirrored_plan_entries ||
        !mirrored_command_headers || !status || !config || !stream ||
        plan_capacity == 0u || payload_slot_capacity == 0u)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg =
        *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(
        materialize_prefill_llep_mirrored_domain_commands_kernel,
        dim3(1),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layer),
        static_cast<DeviceMoERebalancePlanEntryView *>(
            mirrored_plan_entries),
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
            mirrored_command_headers),
        plan_capacity,
        static_cast<DeviceMoERebalanceStatusView *>(status),
        cfg,
        payload_slot_capacity,
        layer_idx);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE materialize_prefill_llep_mirrored_domain_commands kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}


bool hipMoE_init_rebalance_graph_controller_state(
    void *controller_state,
    const void *config,
    int device_idx,
    void *stream)
{
    if (!controller_state || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(init_rebalance_graph_controller_state_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        cfg);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE init_rebalance_graph_controller_state kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_reset_rebalance_graph_transaction_for_request(
    void *controller_state,
    void *command_headers,
    void *wave_states,
    uint32_t *plan_counts,
    uint32_t command_buffer_count,
    const void *config,
    int device_idx,
    void *stream)
{
    if (!controller_state || !command_headers || !wave_states ||
        !plan_counts || command_buffer_count == 0u ||
        command_buffer_count > 2u || !config || !stream)
        return false;
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(reset_rebalance_graph_transaction_for_request_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_headers),
        static_cast<DeviceMoERebalanceWaveStateView *>(wave_states),
        plan_counts,
        command_buffer_count,
        cfg);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "HIP MoE reset_rebalance_graph_transaction_for_request kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_publish_rebalance_transfer_complete(
    void *controller_state,
    const void *command_header,
    const void *wave_state,
    const void *copy_status,
    const void *plan_entries,
    uint32_t plan_capacity,
    const void *gathered_copy_status,
    const void *config,
    uint32_t command_buffer_count,
    int device_idx,
    void *stream)
{
    if (!controller_state || !command_header || !wave_state ||
        !copy_status || !plan_entries || plan_capacity == 0u ||
        !gathered_copy_status || !config || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    hipLaunchKernelGGL(publish_rebalance_transfer_complete_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<const DeviceMoERebalanceApplyStatusView *>(copy_status),
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_capacity,
            static_cast<const DeviceMoERebalanceApplyStatusView *>(gathered_copy_status),
            cfg,
            command_buffer_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE publish_rebalance_transfer_complete kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
