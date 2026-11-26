/**
 * @file CUDAMoERebalanceApplyKernels.cu
 * @brief CUDA overlay rebalance payload pack/unpack, arrival apply, ready-wave apply, and publication finalization kernels.
 */
#include "CUDAMoERebalanceCommon.cuh"

namespace
{
    __global__ void init_rebalance_apply_status_kernel(DeviceMoERebalanceApplyStatusView *status)
    {
        if (!status || blockIdx.x != 0 || threadIdx.x != 0)
            return;
        init_rebalance_apply_status_device(status);
    }


    /**
     * @brief Begin one immutable device-owned payload-copy transaction.
     *
     * This kernel is enqueued immediately before payload packing on the
     * transfer stream. It snapshots the root-projected command-buffer wave,
     * epoch, and count into the status object that is already allgathered for
     * publication. Every later transfer phase reads this ticket; none may
     * re-sample the controller's mutable `active_wave`.
     */
    __global__ void begin_rebalance_copy_transaction_kernel(
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!status || blockIdx.x != 0 || threadIdx.x != 0)
            return;

        init_rebalance_apply_status_device(status);
        if (!command_headers ||
            !rebalance_config_ok(config) ||
            plan_capacity == 0u)
        {
            status->status_code =
                kDeviceMoERebalanceApplyStatusInvalidRuntime;
            return;
        }

        const uint32_t wave =
            rebalance_select_transfer_transaction_wave(
                command_headers,
                controller_state,
                config,
                command_buffer_count);
        const uint32_t count =
            rebalance_command_buffer_count(command_buffer_count);
        if (wave >= count)
        {
            status->status_code =
                kDeviceMoERebalanceApplyStatusInvalidRuntime;
            return;
        }

        const auto &header = command_headers[wave];
        if (!rebalance_command_header_ok(&header, config))
        {
            status->status_code =
                kDeviceMoERebalanceApplyStatusInvalidRuntime;
            return;
        }

        status->transaction_wave_index = wave;
        status->transaction_epoch = header.epoch;
        status->transaction_command_count =
            min(header.command_count,
                min(header.command_capacity, plan_capacity));
        __threadfence();
    }


    __global__ void pack_rebalance_collective_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntryView *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!status || !gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;
        const uint32_t global_slot = blockIdx.x;
        if (global_slot >= local_payload_slot_count)
            return;

        uint8_t *payload_slot =
            local_payload + static_cast<unsigned long long>(global_slot) * payload_slot_bytes;
        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) =
                DeviceMoEExpertDirectoryEntryView{};
        __syncthreads();

        const uint32_t destination_participant = global_slot / plan_capacity;
        const uint32_t plan_index = global_slot % plan_capacity;
        if (destination_participant >= config.participant_count)
            return;
        const uint32_t metadata_buffer_count =
            rebalance_command_buffer_count(command_buffer_count);
        const uint32_t active_wave =
            status->transaction_wave_index;
        if (active_wave >= metadata_buffer_count)
            return;
        const unsigned long long metadata_participant_base =
            static_cast<unsigned long long>(destination_participant) *
            static_cast<unsigned long long>(metadata_buffer_count);
        const auto &header = gathered_command_headers[metadata_participant_base + active_wave];
        if (!rebalance_command_header_for_participant_ok(
                header,
                destination_participant,
                config))
            return;
        const uint32_t count = min(header.command_count,
                                   min(header.command_capacity, plan_capacity));
        if (plan_index >= count)
            return;

        const auto &plan =
            gathered_plan_entries[(metadata_participant_base + active_wave) *
                                      static_cast<unsigned long long>(plan_capacity) +
                                  static_cast<unsigned long long>(plan_index)];
        const uint32_t source_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        if (plan.op == 0u ||
            !rebalance_plan_requires_payload(plan.op) ||
            plan.destination_participant != destination_participant ||
            plan.source_participant != config.participant_id ||
            plan.layer >= config.num_layers ||
            plan.expert >= config.num_experts)
        {
            return;
        }
        if ((plan.source_resident_mask & source_bit) == 0u)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->invalid_plan_entries, 1u);
            return;
        }

        const auto &src =
            local_directory[static_cast<unsigned long long>(plan.layer) *
                                static_cast<unsigned long long>(config.num_experts) +
                            static_cast<unsigned long long>(plan.expert)];
        if (!rebalance_source_entry_ready(src, config.participant_id, plan.layer, plan.expert))
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->missing_source_descriptors, 1u);
            return;
        }
        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        if (rebalance_expert_payload_bytes(src) > payload_data_bytes)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->descriptor_mismatches, 1u);
            return;
        }

        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) = src;
        __syncthreads();

        uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        (void)rebalance_pack_expert_to_payload(src, payload_data);
    }

    __global__ void pack_rebalance_compact_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntryView *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        if (!status || !plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload ||
            !rebalance_config_ok(config) || plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;

        const uint32_t global_slot = blockIdx.x;
        if (global_slot >= local_payload_slot_count)
            return;

        uint8_t *payload_slot =
            local_payload + static_cast<unsigned long long>(global_slot) * payload_slot_bytes;
        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) =
                DeviceMoEExpertDirectoryEntryView{};
        __syncthreads();

    const uint32_t metadata_buffer_count =
        rebalance_command_buffer_count(command_buffer_count);
        const uint32_t active_wave =
            status->transaction_wave_index;
        if (active_wave >= metadata_buffer_count)
            return;
        const auto &header = command_headers[active_wave];
        if (!rebalance_command_header_ok(&header, config))
            return;
        const uint32_t command_count =
            min(header.command_count,
                min(header.command_capacity, plan_capacity));
        const uint32_t source_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));

        __shared__ uint32_t shared_plan_index;
        if (threadIdx.x == 0)
            shared_plan_index = kDeviceMoEInvalidSlot;
        __syncthreads();
        for (uint32_t plan_index = threadIdx.x;
             plan_index < command_count;
             plan_index += blockDim.x)
        {
            const auto &plan =
                plan_entries[static_cast<unsigned long long>(active_wave) *
                                 static_cast<unsigned long long>(plan_capacity) +
                             static_cast<unsigned long long>(plan_index)];
            const bool matches_local_source_slot =
                rebalance_plan_requires_payload(plan.op) &&
                plan.source_participant == config.participant_id &&
                plan.destination_participant < config.participant_count &&
                plan.payload_slot == global_slot &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts;
            if (matches_local_source_slot &&
                (plan.source_resident_mask & source_bit) == 0u)
            {
                atomicAdd(&status->invalid_plan_entries, 1u);
                continue;
            }
            if (matches_local_source_slot)
            {
                atomicCAS(&shared_plan_index, kDeviceMoEInvalidSlot, plan_index);
            }
        }
        __syncthreads();
        if (shared_plan_index == kDeviceMoEInvalidSlot)
            return;

        const auto &selected_plan =
            plan_entries[static_cast<unsigned long long>(active_wave) *
                             static_cast<unsigned long long>(plan_capacity) +
                         static_cast<unsigned long long>(shared_plan_index)];
        const auto &src = local_source_descriptors[
            rebalance_source_descriptor_local_index(
                config,
                metadata_buffer_count,
                plan_capacity,
                selected_plan.destination_participant,
                active_wave,
                shared_plan_index)];
        if (!rebalance_source_entry_ready(src,
                                          config.participant_id,
                                          selected_plan.layer,
                                          selected_plan.expert))
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(&status->missing_source_descriptors, 1u);
            }
            return;
        }

        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        if (rebalance_expert_payload_bytes(src) > payload_data_bytes)
        {
            if (threadIdx.x == 0)
                atomicAdd(&status->descriptor_mismatches, 1u);
            return;
        }

        if (threadIdx.x == 0)
            *reinterpret_cast<DeviceMoEExpertDirectoryEntryView *>(payload_slot) = src;
        __syncthreads();

        uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        (void)rebalance_pack_expert_to_payload(src, payload_data);
    }

    __global__ void unpack_rebalance_collective_payloads_kernel(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count)
    {
        const uint32_t command_wave_index =
            status->transaction_wave_index;
        if (command_wave_index >=
            rebalance_command_buffer_count(command_buffer_count))
        {
            return;
        }
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;

        if (!status || !plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots ||
            plan_capacity == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return;

        const uint32_t count = rebalance_command_count(plan_count, plan_capacity, command_header);

        if (blockIdx.x == 0)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(&status->plan_entries_seen, count);
                status->local_transfer_slot_count =
                    local_transfer_slot_count;
            }
            for (uint32_t plan_index = threadIdx.x;
                 plan_index < count;
                 plan_index += blockDim.x)
            {
                const auto &plan = plan_entries[plan_index];
                if (plan.op == 0u)
                    continue;
                if (!rebalance_plan_applies_runtime(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    plan.destination_participant >= config.participant_count)
                {
                    atomicAdd(&status->invalid_plan_entries, 1u);
                    continue;
                }

                /*
                 * ResidentExpertAssignment is already fully represented by
                 * command metadata. It must reach the apply kernel, but this
                 * byte-unpack phase has no payload work to validate or copy.
                 */
                if (!rebalance_plan_requires_payload(plan.op))
                    continue;

                if ((plan.source_resident_mask &
                     runtime_participant_bit(static_cast<int>(plan.source_participant))) == 0u)
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
                    if (atomicCAS(
                            &status->first_missing_destination_layer,
                            kDeviceMoEInvalidSlot,
                            plan.layer) == kDeviceMoEInvalidSlot)
                    {
                        status->first_missing_destination_slot =
                            plan.destination_slot;
                        status->first_missing_destination_expert =
                            plan.expert;
                        status->first_missing_destination_source =
                            plan.source_participant;
                    }
                    continue;
                }
                if (plan.payload_slot >= local_payload_slot_count)
                {
                    atomicAdd(&status->invalid_plan_entries, 1u);
                    continue;
                }
            }
            return;
        }

        const uint32_t destination_slot = blockIdx.x - 1u;
        if (destination_slot >= local_transfer_slot_count)
            return;

        __shared__ uint32_t shared_plan_index;
        if (threadIdx.x == 0)
            shared_plan_index = kDeviceMoEInvalidSlot;
        __syncthreads();

        for (uint32_t plan_index = threadIdx.x;
             plan_index < count;
             plan_index += blockDim.x)
        {
            const auto &plan = plan_entries[plan_index];
            if (plan.op != 0u &&
                rebalance_plan_requires_payload(plan.op) &&
                plan.layer < config.num_layers &&
                plan.expert < config.num_experts &&
                plan.source_participant < config.participant_count &&
                (plan.source_resident_mask &
                 runtime_participant_bit(static_cast<int>(plan.source_participant))) != 0u &&
                plan.destination_participant == config.participant_id &&
                plan.destination_slot == destination_slot &&
                plan.payload_slot < local_payload_slot_count)
            {
                atomicMin(&shared_plan_index, plan_index);
            }
        }
        __syncthreads();
        if (shared_plan_index == kDeviceMoEInvalidSlot)
            return;

        const auto &plan = plan_entries[shared_plan_index];

        const unsigned long long gathered_slot =
            (static_cast<unsigned long long>(plan.source_participant) *
                 static_cast<unsigned long long>(local_payload_slot_count) +
             static_cast<unsigned long long>(plan.payload_slot));
        const uint8_t *payload_slot = gathered_payload + gathered_slot * payload_slot_bytes;
        const auto &src =
            *reinterpret_cast<const DeviceMoEExpertDirectoryEntryView *>(payload_slot);
        const unsigned long long payload_data_bytes =
            payload_slot_bytes - sizeof(DeviceMoEExpertDirectoryEntryView);
        auto &dst = local_transfer_slots[destination_slot];

        __shared__ uint32_t shared_arrival_valid;
        if (threadIdx.x == 0)
        {
            shared_arrival_valid = 1u;
            const bool lease_matches =
                plan.destination_generation != 0xffffffffu &&
                dst.generation == plan.destination_generation &&
                dst.layer == plan.destination_previous_layer &&
                dst.expert == plan.destination_previous_expert &&
                rebalance_transfer_slot_identity_ok(
                    dst,
                    destination_slot,
                    config);
            if (!lease_matches)
            {
                shared_arrival_valid = 0u;
                atomicAdd(&status->descriptor_mismatches, 1u);
            }
            else if (!rebalance_source_entry_ready(
                         src,
                         plan.source_participant,
                         plan.layer,
                         plan.expert))
            {
                shared_arrival_valid = 0u;
                atomicAdd(&status->missing_source_descriptors, 1u);
            }
            else if (!rebalance_directory_fits_transfer_capacity(src, dst) ||
                     rebalance_expert_payload_bytes(src) > payload_data_bytes)
            {
                shared_arrival_valid = 0u;
                atomicAdd(&status->descriptor_mismatches, 1u);
            }
        }
        __syncthreads();
        if (shared_arrival_valid == 0u)
            return;

        /*
         * Destructive mutation begins only after the complete source, capacity,
         * and generation lease have been validated. A malformed or stale
         * arrival therefore leaves the previous directory publication intact.
         */
        if (threadIdx.x == 0)
        {
            dst.layer = kDeviceMoEInvalidSlot;
            dst.expert = kDeviceMoEInvalidSlot;
            dst.descriptor.logical_expert_id = -1;
            dst.resident_mask &= ~runtime_participant_bit(static_cast<int>(config.participant_id));
            dst.flags &= ~(kDeviceMoEDirectoryFlagResident |
                           kDeviceMoEDirectoryFlagCopyComplete);
        }
        __syncthreads();

        if (threadIdx.x == 0)
            rebalance_retarget_transfer_directory(dst, src);
        __syncthreads();

        unsigned long long offset = 0ULL;
        const uint8_t *payload_data = payload_slot + sizeof(DeviceMoEExpertDirectoryEntryView);
        offset = rebalance_unpack_expert_from_payload(payload_data, dst);
        (void)offset;
        __syncthreads();
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0)
        {
            dst.layer = plan.layer;
            dst.expert = plan.expert;
            dst.descriptor.logical_expert_id = static_cast<int32_t>(plan.expert);
            /*
             * The transfer directory and the runtime bank are two publications
             * of the same physical slot.  An ownership-transfer arrival changes
             * the authoritative participant as soon as the destination bytes
             * are installed.  Publishing the source's old owner here while the
             * apply kernel published the destination owner left those views
             * permanently incoherent: capacity accounting still saw a live
             * transfer-backed expert, but a later destination-slot lease
             * rejected the directory entry because its owner disagreed with the
             * runtime descriptor.
             *
             * Payload-only replication deliberately preserves the source owner.
             * F fragments retain joint ownership: the directory's physical
             * destination participant is never a scalar logical expert owner.
             * Both cases are derived from the immutable plan so no host mirror
             * or post-apply repair is involved.
             */
            dst.descriptor.owner_participant =
                src.descriptor.projection_set == llaminar2::DeviceMoEProjectionSet::IntermediateShard
                    ? -1 : plan.op == kDeviceMoERebalancePlanOwnershipTransfer
                    ? static_cast<int32_t>(plan.destination_participant)
                    : src.descriptor.owner_participant;
            dst.descriptor.flags |= kDeviceMoEFlagTransferSlot;
            dst.resident_mask |= runtime_participant_bit(static_cast<int>(config.participant_id));
            dst.epoch = command_header ? command_header->epoch : 0u;
            dst.generation = plan.destination_generation + 1u;
            dst.flags |= kDeviceMoEDirectoryFlagResident |
                         kDeviceMoEDirectoryFlagCopyComplete;
            atomicAdd(&status->copied_arrivals, 1u);
            atomicAdd(reinterpret_cast<unsigned long long *>(&status->copied_payload_bytes), offset);
        }
    }


    __global__ void apply_rebalance_arrivals_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        int target_layer,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        bool require_ready_wave,
        uint32_t command_buffer_count,
        const llaminar2::DeviceMoEOverlayEpochStatus *
            overlay_reservation_status)
    {
        if (blockIdx.x != 0 || !status)
            return;

        __shared__ uint8_t changed_layer[kDeviceMoEMaxExperts];
        __shared__ uint8_t current_batch_child_layer[kDeviceMoEMaxExperts];
        __shared__ uint32_t reduce_scratch[kThreads];
        __shared__ uint32_t shared_valid;
        __shared__ uint32_t shared_count;
        __shared__ uint32_t shared_participant_bit;
        __shared__ uint32_t shared_valid_mask;
        __shared__ DeviceMoERebalanceWaveProgressView *shared_ready_wave;
        __shared__ uint32_t shared_ready_wave_index;
        __shared__ const DeviceMoERebalancePlanEntryView *shared_plan_entries;
        __shared__ const uint32_t *shared_plan_count;
        __shared__ DeviceMoERebalanceCommandBufferHeaderView *shared_command_header;
        __shared__ uint32_t shared_overlay_enabled;
        __shared__ uint32_t shared_overlay_published_bank;
        __shared__ uint32_t shared_overlay_candidate_bank;
        __shared__ uint64_t shared_overlay_candidate_epoch;

        for (uint32_t layer = threadIdx.x;
             layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             layer += blockDim.x)
        {
            changed_layer[layer] = 0u;
            current_batch_child_layer[layer] = 0u;
        }
        reduce_scratch[threadIdx.x] = 0u;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            if (require_ready_wave)
                init_rebalance_apply_status_in_progress_device(status);

            bool setup_ok = true;
            shared_valid = 0u;
            shared_count = 0u;
            shared_participant_bit = 0u;
            shared_valid_mask = 0u;
            shared_ready_wave = nullptr;
            shared_ready_wave_index = 0u;
            shared_plan_entries = plan_entries;
            shared_plan_count = plan_count;
            shared_command_header = command_header;
            shared_overlay_enabled = 0u;
            shared_overlay_published_bank = 0u;
            shared_overlay_candidate_bank = 0u;
            shared_overlay_candidate_epoch = 0u;

            if (overlay_reservation_status)
            {
                const uint32_t reserve_code =
                    overlay_reservation_status->code;
                const uint32_t reserve_operation =
                    overlay_reservation_status->operation;
                const uint32_t candidate_bank =
                    overlay_reservation_status->bank;
                const uint64_t candidate_epoch =
                    overlay_reservation_status->epoch;
                const uint64_t selector =
                    overlay_reservation_status->selector;
                const uint32_t published_bank =
                    static_cast<uint32_t>(selector & 1u);
                const uint64_t generation = selector >> 1u;
                if (reserve_code != static_cast<uint32_t>(
                                        llaminar2::DeviceMoEOverlayEpochStatusCode::Success))
                {
                    /* Busy retirement and no-capacity conditions are expected
                     * maintenance poll misses. Leave placement untouched. */
                    if (require_ready_wave)
                        status->status_code = 0u;
                    setup_ok = false;
                }
                else if (reserve_operation != static_cast<uint32_t>(
                                                   llaminar2::DeviceMoEOverlayEpochOperation::ReserveCandidate) ||
                         target_layer >= 0 || candidate_bank > 1u ||
                         published_bank > 1u || candidate_bank == published_bank ||
                         candidate_epoch <= 1u ||
                         candidate_epoch > 0xffffffffull ||
                         generation == 0u)
                {
                    if (require_ready_wave)
                        status->status_code =
                            kDeviceMoERebalanceApplyStatusInvalidRuntime;
                    setup_ok = false;
                }
                else
                {
                    shared_overlay_enabled = 1u;
                    shared_overlay_published_bank = published_bank;
                    shared_overlay_candidate_bank = candidate_bank;
                    shared_overlay_candidate_epoch = candidate_epoch;
                }
            }

            if (require_ready_wave)
            {
                if (!controller_state)
                {
                    setup_ok = false;
                }
                else
                {
                    if ((config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u &&
                        rebalance_graph_controller_state_ok(controller_state, config))
                        ++controller_state->decode_apply_polls;
                    const DeviceMoEReadyWaveSelection ready = rebalance_find_ready_wave(
                        controller_state,
                        command_header,
                        config,
                        target_layer,
                        command_buffer_count);
                    shared_ready_wave = ready.wave;
                    shared_ready_wave_index = ready.wave_index;
                    if (!shared_ready_wave)
                    {
                        /*
                         * An empty ready-wave poll is a successful no-op.  The
                         * status was initialized to InProgress above so a real
                         * asynchronous apply can never look complete early;
                         * publish Ok before returning when there is no wave to
                         * consume so terminal diagnostics do not mistake an
                         * idle poll for unfinished device work.
                         */
                        status->status_code = 0u;
                        setup_ok = false;
                    }
                    else
                    {
                        shared_plan_entries =
                            plan_entries + static_cast<unsigned long long>(shared_ready_wave_index) * plan_capacity;
                        if (plan_count)
                            shared_plan_count = plan_count + shared_ready_wave_index;
                        if (command_header)
                            shared_command_header = command_header + shared_ready_wave_index;
                        shared_ready_wave->state = kDeviceMoERebalanceLifecycleApplying;
                        status->transaction_wave_index = shared_ready_wave_index;
                        status->transaction_epoch = shared_ready_wave->epoch;
                        status->transaction_command_count = shared_ready_wave->command_count;
                        ++controller_state->decode_apply_hits;
                    }
                }
            }

            if (setup_ok && !rebalance_config_ok(config))
            {
                status->status_code = 1u;
                if (shared_ready_wave)
                {
                    poison_rebalance_graph_controller(
                        controller_state,
                        1u,
                        shared_ready_wave_index,
                        shared_ready_wave->epoch,
                        0u,
                        0u,
                        status->status_code,
                        config.participant_id);
                }
                setup_ok = false;
            }
            if (setup_ok && !runtime_layers)
            {
                status->status_code = 2u;
                if (shared_ready_wave)
                {
                    poison_rebalance_graph_controller(
                        controller_state,
                        2u,
                        shared_ready_wave_index,
                        shared_ready_wave->epoch,
                        0u,
                        0u,
                        status->status_code,
                        config.participant_id);
                }
                setup_ok = false;
            }
            if (setup_ok &&
                (!shared_plan_entries || (!shared_plan_count && !shared_command_header) ||
                 !rebalance_command_header_ok(shared_command_header, config)))
            {
                status->status_code = 3u;
                if (shared_ready_wave)
                {
                    poison_rebalance_graph_controller(
                        controller_state,
                        3u,
                        shared_ready_wave_index,
                        shared_ready_wave->epoch,
                        0u,
                        0u,
                        status->status_code,
                        config.participant_id);
                }
                setup_ok = false;
            }

            if (setup_ok)
            {
                shared_count = rebalance_command_count(shared_plan_count, plan_capacity, shared_command_header);
                shared_participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
                shared_valid_mask = runtime_valid_participant_mask(config.participant_count);
                shared_valid = 1u;
            }
        }
        __syncthreads();
        if (shared_valid == 0u)
            return;

        if (threadIdx.x == 0)
        {
            for (uint32_t i = 0; i < shared_count; ++i)
            {
                ++status->plan_entries_seen;
                const auto &plan = shared_plan_entries[i];
                if (plan.op == 0u)
                    continue;
                if (target_layer >= 0 &&
                    plan.layer != static_cast<uint32_t>(target_layer))
                    continue;
                if (!rebalance_plan_applies_runtime(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    plan.destination_participant >= config.participant_count ||
                    rebalance_plan_duplicates_prior_local_arrival(
                        shared_plan_entries,
                        i,
                        plan,
                        config,
                        target_layer))
                {
                    ++status->invalid_plan_entries;
                    continue;
                }
                const bool destination_local =
                    plan.destination_participant == config.participant_id;
                const uint32_t destination_bit =
                    runtime_participant_bit(static_cast<int>(plan.destination_participant));
                const bool current_batch_plan =
                    (plan.flags &
                     kDeviceMoERebalancePlanFlagCurrentBatchLLEP) != 0u;

                auto &runtime = runtime_layers[plan.layer];
                const uint32_t execution_bank =
                    runtime_execution_bank(&runtime);
                /* Durable maintenance starts only after the inference reader
                 * has released its epoch ticket. In that transaction the
                 * successful reservation status, not a deliberately invalid
                 * request ticket, authorizes reads from published_bank. A
                 * request-local apply has no such reservation and must still
                 * prove its exact ticket-selected execution bank. */
                if (runtime.active_bank > 1u ||
                    (shared_overlay_enabled == 0u && execution_bank > 1u) ||
                    (shared_overlay_enabled != 0u &&
                     runtime.active_bank != shared_overlay_published_bank) ||
                    runtime.expert_count != config.num_experts ||
                    runtime.top_k != config.top_k ||
                    runtime.participant_id != config.participant_id ||
                    runtime.participant_count != config.participant_count ||
                    (plan.op == kDeviceMoERebalancePlanOwnershipTransfer &&
                     runtime.overlay_epoch_ticket &&
                     plan.destination_overlay_participant < 0) ||
                    (current_batch_plan &&
                     (!runtime.overlay_placement_banks ||
                      !runtime.overlay_epoch_ticket ||
                      !shared_command_header ||
                      shared_command_header->epoch !=
                          runtime_command_epoch(&runtime))))
                {
                    ++status->invalid_plan_entries;
                    continue;
                }

                const uint32_t source_bank =
                    shared_overlay_enabled != 0u
                        ? shared_overlay_published_bank
                        : execution_bank;
                const auto &active =
                    shared_overlay_enabled != 0u
                        ? runtime.banks[source_bank]
                        : runtime_placement_banks(&runtime)[source_bank];
                uint32_t resident_mask =
                    (active.resident_participant_mask[plan.expert] |
                     plan.source_resident_mask |
                     destination_bit) &
                    shared_valid_mask;
                if (rebalance_plan_requires_payload(plan.op) &&
                    destination_local)
                {
                    ++status->required_local_arrivals;
                    if (!local_transfer_slots ||
                        plan.destination_slot >= local_transfer_slot_count)
                    {
                        ++status->missing_destination_slots;
                        continue;
                    }

                    const auto &slot = local_transfer_slots[plan.destination_slot];
                    if (!rebalance_transfer_slot_copy_complete_for_plan(
                            slot,
                            plan,
                            config.participant_id,
                            shared_command_header ? shared_command_header->epoch : 0u))
                    {
                        ++status->copy_incomplete;
                        continue;
                    }
                    ++status->ready_local_arrivals;
                    if (plan.op == kDeviceMoERebalancePlanOwnershipTransfer)
                        resident_mask = destination_bit & shared_valid_mask;
                    else
                        resident_mask |= shared_participant_bit;
                }
                else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                         destination_local &&
                         (resident_mask & shared_participant_bit) == 0u)
                {
                    ++status->missing_source_descriptors;
                    continue;
                }

                changed_layer[plan.layer] = 1u;
                if (current_batch_plan)
                {
                    current_batch_child_layer[plan.layer] = 1u;
                }
                if (destination_local &&
                    rebalance_plan_requires_payload(plan.op) &&
                    plan.destination_previous_layer < config.num_layers &&
                    plan.destination_previous_expert < config.num_experts &&
                    (plan.destination_previous_layer != plan.layer ||
                     plan.destination_previous_expert != plan.expert))
                {
                    changed_layer[plan.destination_previous_layer] = 1u;
                    auto &previous_runtime =
                        runtime_layers[plan.destination_previous_layer];
                    if (current_batch_plan &&
                        (!previous_runtime.overlay_placement_banks ||
                         !previous_runtime.overlay_epoch_ticket ||
                         !shared_command_header ||
                         shared_command_header->epoch !=
                             runtime_command_epoch(&previous_runtime)))
                    {
                        ++status->invalid_plan_entries;
                        continue;
                    }
                    if (current_batch_plan)
                    {
                        current_batch_child_layer[
                            plan.destination_previous_layer] = 1u;
                    }
                }
            }

            /*
             * Preflight is the transaction's only commit gate. In particular,
             * a ReadyToApply wave may race the final directory visibility on
             * another stream, but no inactive bank may be assembled or flipped
             * until every local arrival is visible in this stream's order.
             */
            if (rebalance_apply_transaction_blocked(status))
            {
                const bool permanent_plan_error =
                    status->invalid_plan_entries != 0u ||
                    status->missing_source_descriptors != 0u ||
                    status->missing_destination_slots != 0u ||
                    status->descriptor_mismatches != 0u;
                if (permanent_plan_error)
                {
                    status->status_code =
                        status->invalid_plan_entries != 0u
                            ? 3u
                            : 4u;
                    if (shared_ready_wave)
                    {
                        poison_rebalance_graph_controller(
                            controller_state,
                            status->status_code,
                            shared_ready_wave_index,
                            shared_ready_wave->epoch,
                            status->required_local_arrivals,
                            status->ready_local_arrivals,
                            status->status_code,
                            config.participant_id);
                    }
                }
                else if (shared_ready_wave)
                {
                    shared_ready_wave->state =
                        kDeviceMoERebalanceLifecycleReadyToApply;
                }
                shared_valid = 0u;
                __threadfence();
            }
        }
        __syncthreads();
        if (shared_valid == 0u)
            return;

        for (uint32_t layer = 0;
             layer < config.num_layers && layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             ++layer)
        {
            if (changed_layer[layer] == 0u)
                continue;

            auto &runtime = runtime_layers[layer];
            const uint32_t source_bank =
                shared_overlay_enabled != 0u
                    ? shared_overlay_published_bank
                    : runtime_execution_bank(&runtime);
            const uint32_t inactive_bank =
                shared_overlay_enabled != 0u
                    ? shared_overlay_candidate_bank
                    : 1u - source_bank;
            const auto &active =
                runtime_placement_banks(&runtime)[source_bank];
            auto &next = runtime.banks[inactive_bank];
            if (threadIdx.x == 0)
            {
                next.epoch =
                    shared_overlay_enabled != 0u
                        ? static_cast<uint32_t>(shared_overlay_candidate_epoch)
                        : (current_batch_child_layer[layer] != 0u
                               ? static_cast<uint32_t>(
                                     runtime.overlay_epoch_ticket->epoch)
                               : runtime.active_epoch + 1u);
                next.expert_count = config.num_experts;
                next.multi_resident_expert_count =
                    active.multi_resident_expert_count;
                next.transient_placement_observed =
                    active.transient_placement_observed;
            }
            for (uint32_t expert = threadIdx.x; expert < config.num_experts; expert += blockDim.x)
            {
                next.experts[expert] = active.experts[expert];
                next.overlay_route_participant[expert] =
                    active.overlay_route_participant[expert];
                auto &base_desc = next.experts[expert];
                uint32_t resident_mask =
                    active.resident_participant_mask[expert] & shared_valid_mask;
                if (rebalance_ready_arrival_reuses_runtime_slot(
                        shared_plan_entries,
                        shared_count,
                        local_transfer_slots,
                        local_transfer_slot_count,
                        config,
                        shared_command_header ? shared_command_header->epoch : 0u,
                        layer,
                        expert,
                        base_desc))
                {
                    rebalance_retire_reused_transfer_slot_replica(
                        base_desc,
                        resident_mask,
                        shared_participant_bit,
                        config.participant_id);
                }
                const bool owner_local =
                    base_desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool local_resident = (resident_mask & shared_participant_bit) != 0u;
                const bool multi_resident =
                    runtime_resident_count(resident_mask, config.participant_count) > 1;
                if (multi_resident)
                    base_desc.flags |= kDeviceMoEFlagReplicated;
                else
                    base_desc.flags &= ~kDeviceMoEFlagReplicated;
                if (local_resident)
                {
                    base_desc.flags |= kDeviceMoEFlagValid |
                                       kDeviceMoEFlagResident |
                                       kDeviceMoEFlagLocalCompute;
                }
                else
                {
                    if (owner_local)
                    {
                        FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                            "authoritative CUDA apply owner lost its local payload");
                    }
                    rebalance_retire_local_payload_publication(
                        base_desc,
                        resident_mask,
                        shared_participant_bit);
                }
                next.local_compute_mask[expert] = local_resident ? 1u : 0u;
                next.replica_role[expert] =
                    local_resident
                        ? (owner_local
                               ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                               : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica))
                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
                next.resident_participant_mask[expert] = resident_mask;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            for (uint32_t i = 0; i < shared_count; ++i)
            {
                const auto &plan = shared_plan_entries[i];
                if (plan.op == 0u)
                    continue;
                if (target_layer >= 0 &&
                    plan.layer != static_cast<uint32_t>(target_layer))
                    continue;
                if (!rebalance_plan_applies_runtime(plan.op) ||
                    plan.layer >= config.num_layers ||
                    plan.expert >= config.num_experts ||
                    plan.source_participant >= config.participant_count ||
                    plan.destination_participant >= config.participant_count ||
                    (plan.op == kDeviceMoERebalancePlanOwnershipTransfer &&
                     runtime_layers[plan.layer].overlay_epoch_ticket &&
                     plan.destination_overlay_participant < 0) ||
                    changed_layer[plan.layer] == 0u)
                {
                    continue;
                }
                const bool destination_local =
                    plan.destination_participant == config.participant_id;
                const uint32_t destination_bit =
                    runtime_participant_bit(static_cast<int>(plan.destination_participant));
                const bool ownership_transfer =
                    plan.op == kDeviceMoERebalancePlanOwnershipTransfer;
                auto &runtime = runtime_layers[plan.layer];
                if (runtime.active_bank > 1u ||
                    runtime.expert_count != config.num_experts ||
                    runtime.top_k != config.top_k ||
                    runtime.participant_id != config.participant_id ||
                    runtime.participant_count != config.participant_count)
                {
                    continue;
                }

                const uint32_t source_bank =
                    shared_overlay_enabled != 0u
                        ? shared_overlay_published_bank
                        : runtime_execution_bank(&runtime);
                const auto &active =
                    runtime_placement_banks(&runtime)[source_bank];
                auto desc = active.experts[plan.expert];
                uint32_t resident_mask =
                    (active.resident_participant_mask[plan.expert] |
                     plan.source_resident_mask |
                     destination_bit) &
                    shared_valid_mask;
                if (ownership_transfer)
                    resident_mask = destination_bit & shared_valid_mask;
                if (rebalance_plan_requires_payload(plan.op) &&
                    destination_local)
                {
                    if (!local_transfer_slots ||
                        plan.destination_slot >= local_transfer_slot_count)
                        continue;
                    const auto &slot = local_transfer_slots[plan.destination_slot];
                    if (!rebalance_transfer_slot_copy_complete_for_plan(
                            slot,
                            plan,
                            config.participant_id,
                            shared_command_header ? shared_command_header->epoch : 0u))
                        continue;
                    desc = slot.descriptor;
                    desc.owner_participant =
                        ownership_transfer
                            ? static_cast<int32_t>(plan.destination_participant)
                            : active.experts[plan.expert].owner_participant;
                    resident_mask = ownership_transfer
                                        ? destination_bit & shared_valid_mask
                                        : resident_mask | shared_participant_bit;
                }
                else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                         destination_local &&
                         (resident_mask & shared_participant_bit) == 0u)
                {
                    continue;
                }

                const uint32_t inactive_bank =
                    shared_overlay_enabled != 0u
                        ? shared_overlay_candidate_bank
                        : 1u - source_bank;
                auto &next = runtime.banks[inactive_bank];
                if (ownership_transfer)
                {
                    desc.owner_participant =
                        static_cast<int32_t>(plan.destination_participant);
                    desc.flags &= ~kDeviceMoEFlagReplicated;
                }
                else
                {
                    desc.flags |= kDeviceMoEFlagReplicated;
                }
                if (destination_local)
                {
                    desc.flags |= kDeviceMoEFlagValid |
                                  kDeviceMoEFlagResident |
                                  kDeviceMoEFlagLocalCompute;
                    next.local_compute_mask[plan.expert] = 1u;
                    next.replica_role[plan.expert] =
                        ownership_transfer ||
                                desc.owner_participant == static_cast<int32_t>(config.participant_id)
                            ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                            : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica);
                    ++status->applied_arrivals;
                }
                else if (ownership_transfer)
                {
                    rebalance_retire_local_payload_publication(
                        desc,
                        resident_mask,
                        shared_participant_bit);
                    next.local_compute_mask[plan.expert] = 0u;
                    next.replica_role[plan.expert] =
                        static_cast<uint8_t>(kDeviceMoEReplicaRoleNone);
                }
                next.experts[plan.expert] = desc;
                if (ownership_transfer)
                {
                    next.overlay_route_participant[plan.expert] =
                        plan.destination_overlay_participant;
                }
                next.resident_participant_mask[plan.expert] =
                    resident_mask & shared_valid_mask;
                if (rebalance_plan_requires_payload(plan.op))
                {
                    /*
                     * The transfer plan is domain-wide even though only its
                     * destination owns the rolling payload descriptor. Keep a
                     * sticky global durability marker in every participant's
                     * bank so portable prefix capture cannot diverge by local
                     * slot ownership.
                     */
                    next.transient_placement_observed = 1u;
                    if ((plan.flags &
                         kDeviceMoERebalancePlanFlagCurrentBatchLLEP) != 0u)
                    {
                        runtime.current_batch_llep_movement_observed = 1u;
                    }
                }
            }
        }
        __syncthreads();

        for (uint32_t layer = 0;
             layer < config.num_layers && layer < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             ++layer)
        {
            if (changed_layer[layer] == 0u)
                continue;

            auto &runtime = runtime_layers[layer];
            const uint32_t source_bank =
                shared_overlay_enabled != 0u
                    ? shared_overlay_published_bank
                    : runtime_execution_bank(&runtime);
            const uint32_t inactive_bank =
                shared_overlay_enabled != 0u
                    ? shared_overlay_candidate_bank
                    : 1u - source_bank;
            auto &next = runtime.banks[inactive_bank];
            uint32_t local_multi_resident = 0u;
            for (uint32_t expert = threadIdx.x; expert < config.num_experts; expert += blockDim.x)
            {
                const uint32_t mask = next.resident_participant_mask[expert] & shared_valid_mask;
                if (runtime_resident_count(mask, config.participant_count) > 1u)
                    ++local_multi_resident;
                if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
                {
                    reset_runtime_histogram_expert(runtime, expert);
                }
            }
            reduce_scratch[threadIdx.x] = local_multi_resident;
            __syncthreads();
            for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u)
            {
                if (threadIdx.x < stride)
                    reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                next.multi_resident_expert_count = reduce_scratch[0];
            }
            __syncthreads();
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                /* A durable overlay apply only prepares the reserved peer.
                 * The request-visible selector and the controller's active
                 * metadata remain on the published bank until the all-layer
                 * finalizer has cloned every untouched layer. */
                if (shared_overlay_enabled == 0u)
                {
                    runtime.active_bank = inactive_bank;
                    runtime.active_epoch = next.epoch;
                    if (current_batch_child_layer[layer] != 0u)
                    {
                        /* Publish the request-local override only after its
                         * complete bank is globally visible. The exact parent
                         * ticket remains the epoch proof used by readers. */
                        runtime.current_batch_llep_transient_bank_active = 1u;
                    }
                }
                ++status->changed_layers;
                status->post_apply_multi_resident_experts += reduce_scratch[0];
            }
            __syncthreads();
        }

        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            status->status_code = 0u;
            if (shared_ready_wave)
            {
                shared_ready_wave->applied_arrivals += status->applied_arrivals;
                const uint32_t layer_increment = target_layer >= 0 ? 1u : shared_ready_wave->planned_layer_count;
                shared_ready_wave->applied_layer_count =
                    min(shared_ready_wave->planned_layer_count,
                        shared_ready_wave->applied_layer_count + layer_increment);
                if (shared_ready_wave->applied_layer_count >= shared_ready_wave->planned_layer_count)
                {
                    if (shared_overlay_enabled != 0u)
                    {
                        // Candidate preparation is not durable publication.
                        // Keep the commands and wave leased to the finalizer.
                        shared_ready_wave->state =
                            llaminar2::moe_rebalance_publication::kPreparedForPublication;
                    }
                    else
                    {
                        clear_rebalance_command_header_device(shared_command_header);
                        shared_ready_wave->state = kDeviceMoERebalanceLifecycleApplied;
                        controller_state->active_wave =
                            (controller_state->active_wave + 1u) %
                            rebalance_command_buffer_count(command_buffer_count);
                    }
                }
                else
                {
                    shared_ready_wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
                }
            }
            __threadfence();
        }
    }


    /**
     * @brief Complete one all-layer RCU publication after rebalance apply.
     *
     * Exactly one block owns the complete family. Each layer is normalized
     * before the publication selector changes: changed layers already occupy
     * the reserved peer bank, while untouched layers are cloned from the old
     * published bank. Main and every MTP sidecar then observe the same immutable
     * bank through their shared ticket and canonical placement pointer.
     */
    __global__ void finalize_overlay_rebalance_publication_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        uint32_t layer_count,
        uint32_t expert_count,
        llaminar2::DeviceMoEOverlayEpochControl *control,
        uint64_t *candidate_epoch_ptr,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        const DeviceMoERebalanceApplyStatusView *apply_status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        uint32_t command_buffer_count,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t plan_capacity,
        uint64_t estimated_expert_bytes,
        llaminar2::DeviceMoERebalanceMovementJournalView journal)
    {
        if (blockIdx.x != 0u || !runtime_layers || !control ||
            !candidate_epoch_ptr || !status || !apply_status ||
            !controller_state || !command_headers)
        {
            return;
        }

        __shared__ uint32_t valid;
        __shared__ uint32_t published_bank;
        __shared__ uint32_t candidate_bank;
        __shared__ uint64_t candidate_epoch;
        __shared__ uint64_t old_selector;
        // One writer owns reservation/commit; edge copies use all block lanes.
        // Shared storage prevents a per-thread copy of the sealed policy proof.
        __shared__ uint64_t journal_append_storage[
            (sizeof(llaminar2::DeviceMoEMovementJournalAppend) + sizeof(uint64_t) - 1u) /
            sizeof(uint64_t)];
        auto &journal_append =
            *reinterpret_cast<llaminar2::DeviceMoEMovementJournalAppend *>(journal_append_storage);
        __shared__ uint32_t journal_required;
        __shared__ uint64_t plan_offset;
        // One root writer owns the completion receipt. Shared scratch avoids
        // replicating this cold maintenance proof in every cooperative lane.
        __shared__ uint64_t decision_append_storage[
            (sizeof(llaminar2::DeviceMoECompletedDecisionAppend) + sizeof(uint64_t) - 1u) / sizeof(uint64_t)];
        auto &decision_append = *reinterpret_cast<llaminar2::DeviceMoECompletedDecisionAppend *>(decision_append_storage);

        if (threadIdx.x == 0u)
        {
            journal_required = 0u;
            plan_offset = 0u;
            decision_append.disposition = llaminar2::DeviceMoECompletedDecisionDisposition::Inactive;
            /* Capture the reserve result before this same record becomes the
             * final publication result. A failed/busy reserve is deliberately
             * left intact for an ordered diagnostic consumer. */
            const uint32_t reserve_code = status->code;
            const uint32_t reserve_operation = status->operation;
            candidate_bank = status->bank;
            candidate_epoch = status->epoch;
            old_selector = status->selector;
            published_bank = static_cast<uint32_t>(old_selector & 1u);
            valid =
                reserve_code == static_cast<uint32_t>(
                                    llaminar2::DeviceMoEOverlayEpochStatusCode::Success) &&
                        reserve_operation == static_cast<uint32_t>(
                                                 llaminar2::DeviceMoEOverlayEpochOperation::ReserveCandidate)
                    ? 1u
                    : 0u;

            if (valid != 0u)
            {
                const bool apply_record_valid =
                    apply_status->magic == kDeviceMoERebalanceMagic &&
                    apply_status->version == kDeviceMoERebalanceVersion &&
                    apply_status->status_code == kDeviceMoERebalanceStatusOk &&
                    (apply_status->changed_layers == 0u ||
                     llaminar2::moe_rebalance_publication::preparedWaveMatches(
                         controller_state, command_headers, command_buffer_count,
                         apply_status));
                const bool topology_valid =
                    layer_count > 0u &&
                    layer_count <= static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
                    expert_count > 0u &&
                    expert_count <= static_cast<uint32_t>(kDeviceMoEMaxExperts) &&
                    candidate_bank <= 1u && published_bank <= 1u &&
                    candidate_bank != published_bank &&
                    (old_selector >> 1u) != 0u && candidate_epoch > 1u &&
                    candidate_epoch < 0xffffffffull &&
                    *candidate_epoch_ptr == candidate_epoch &&
                    overlay_atomic_load64(&control->bank_epochs[candidate_bank]) ==
                        candidate_epoch &&
                    overlay_atomic_load32(&control->bank_states[candidate_bank]) ==
                        static_cast<uint32_t>(
                            llaminar2::DeviceMoEOverlayEpochBankState::Candidate) &&
                    overlay_atomic_load32(&control->bank_states[published_bank]) ==
                        static_cast<uint32_t>(
                            llaminar2::DeviceMoEOverlayEpochBankState::Published) &&
                    overlay_atomic_load64(&control->published_selector) ==
                        old_selector;

                bool journal_valid = journal.state == &controller_state->movement_journal &&
                    journal.valid();
                if (apply_record_valid && topology_valid &&
                    apply_status->changed_layers != 0u)
                {
                    const auto &header = command_headers[apply_status->transaction_wave_index];
                    const auto kind = header.movement_decision.kind;
                    journal_required = kind ==
                        llaminar2::DeviceMoERebalanceMovementDecisionKind::NativeOwnershipSpread;
                    journal_valid = journal_valid && (journal_required != 0u || kind ==
                        llaminar2::DeviceMoERebalanceMovementDecisionKind::None);
                    if (journal_required != 0u)
                    {
                        const auto &wave = controller_state->waves[apply_status->transaction_wave_index];
                        journal_append = llaminar2::prepareDeviceMoEMovementJournalAppend(
                            journal, candidate_epoch, header.epoch, header.command_count,
                            header.movement_decision.load, wave.copied_payload_bytes);
                        journal_valid = journal_valid && plan_entries && plan_capacity != 0u &&
                            header.command_count <= plan_capacity && estimated_expert_bytes != 0u &&
                            journal_append.disposition !=
                                llaminar2::DeviceMoEMovementJournalDisposition::Invalid;
                        plan_offset = static_cast<uint64_t>(apply_status->transaction_wave_index) *
                            plan_capacity;
                        // This bounded metadata walk is performed only for an
                        // accepted maintenance wave, never for each decode token.
                        for (uint32_t i = 0u; journal_valid && i < header.command_count; i += 2u)
                            journal_valid = llaminar2::nativeMovementPairValid(
                                plan_entries[plan_offset + i], plan_entries[plan_offset + i + 1u],
                                layer_count, expert_count, controller_state->participant_count,
                                kDeviceMoERebalancePlanOwnershipTransfer);
                    }
                }

                if (apply_record_valid && topology_valid && journal_valid && journal.decision_status)
                {
                    const auto &policy_status = *reinterpret_cast<const DeviceMoERebalanceStatusView *>(
                        journal.decision_status);
                    // Moving provenance is retained with commands, not later
                    // planner scratch. Busy reservation may defer this exact wave.
                    if (journal_required != 0u)
                        decision_append = llaminar2::prepareDeviceMoECompletedOwnershipDecision(
                            *controller_state,
                            command_headers[apply_status->transaction_wave_index].movement_decision.load,
                            candidate_epoch, apply_status->transaction_epoch, apply_status->transaction_command_count);
                    else
                        decision_append = llaminar2::prepareDeviceMoECompletedDecision(
                            *controller_state, policy_status, candidate_epoch,
                            apply_status->changed_layers ? apply_status->transaction_epoch : 0u,
                            apply_status->changed_layers ? apply_status->transaction_command_count : 0u);
                    journal_valid = decision_append.disposition !=
                        llaminar2::DeviceMoECompletedDecisionDisposition::Invalid;
                }

                if (!apply_record_valid || !topology_valid || !journal_valid)
                {
                    /* The published bank remains authoritative. Abandon a
                     * structurally valid reservation, but never index a bank
                     * supplied by a corrupt status record. */
                    uint32_t prior = static_cast<uint32_t>(
                        llaminar2::DeviceMoEOverlayEpochBankState::Empty);
                    if (candidate_bank <= 1u)
                    {
                        prior = atomicCAS(
                            reinterpret_cast<unsigned int *>(
                                &control->bank_states[candidate_bank]),
                            static_cast<uint32_t>(
                                llaminar2::DeviceMoEOverlayEpochBankState::Candidate),
                            static_cast<uint32_t>(
                                llaminar2::DeviceMoEOverlayEpochBankState::Empty));
                        if (prior == static_cast<uint32_t>(
                                         llaminar2::DeviceMoEOverlayEpochBankState::Candidate))
                        {
                            overlay_atomic_store64(
                                &control->bank_epochs[candidate_bank], 0u);
                        }
                    }
                    finish_overlay_publication_status(
                        status,
                        llaminar2::DeviceMoEOverlayEpochOperation::AbortCandidate,
                        llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                        candidate_epoch,
                        old_selector,
                        candidate_bank,
                        static_cast<llaminar2::DeviceMoEOverlayEpochBankState>(prior));
                    valid = 0u;
                }

                if (valid != 0u && apply_status->changed_layers == 0u)
                {
                    const uint32_t prior = atomicCAS(
                        reinterpret_cast<unsigned int *>(
                            &control->bank_states[candidate_bank]),
                        static_cast<uint32_t>(
                            llaminar2::DeviceMoEOverlayEpochBankState::Candidate),
                        static_cast<uint32_t>(
                            llaminar2::DeviceMoEOverlayEpochBankState::Empty));
                    if (prior == static_cast<uint32_t>(
                                     llaminar2::DeviceMoEOverlayEpochBankState::Candidate))
                    {
                        overlay_atomic_store64(
                            &control->bank_epochs[candidate_bank], 0u);
                        // A valid no-move policy result completes only when its
                        // reservation really retires. Replaying this finalizer
                        // cannot recount: status is no longer ReserveCandidate.
                        const bool decision_committed = llaminar2::commitDeviceMoECompletedDecision(
                            controller_state->completed_decision, decision_append);
                        finish_overlay_publication_status(
                            status,
                            llaminar2::DeviceMoEOverlayEpochOperation::AbortCandidate,
                            decision_committed ? llaminar2::DeviceMoEOverlayEpochStatusCode::Success
                                               : llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                            candidate_epoch,
                            old_selector,
                            candidate_bank,
                            llaminar2::DeviceMoEOverlayEpochBankState::Empty);
                    }
                    else
                    {
                        finish_overlay_publication_status(
                            status,
                            llaminar2::DeviceMoEOverlayEpochOperation::AbortCandidate,
                            llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                            candidate_epoch,
                            old_selector,
                            candidate_bank,
                            static_cast<llaminar2::DeviceMoEOverlayEpochBankState>(prior));
                    }
                    valid = 0u;
                }

                /* The controller metadata remains on the published source.
                 * Candidate epoch equality is the durable changed-layer mark;
                 * monotonic epochs make it impossible to confuse stale data. */
                for (uint32_t layer = 0u;
                     valid != 0u && layer < layer_count;
                     ++layer)
                {
                    const auto &runtime = runtime_layers[layer];
                    if (runtime.expert_count != expert_count ||
                        runtime.active_bank != published_bank ||
                        runtime.active_epoch !=
                            static_cast<uint32_t>(candidate_epoch - 1u) ||
                        runtime.banks[published_bank].epoch !=
                            static_cast<uint32_t>(candidate_epoch - 1u))
                    {
                        const uint32_t prior = atomicCAS(
                            reinterpret_cast<unsigned int *>(
                                &control->bank_states[candidate_bank]),
                            static_cast<uint32_t>(
                                llaminar2::DeviceMoEOverlayEpochBankState::Candidate),
                            static_cast<uint32_t>(
                                llaminar2::DeviceMoEOverlayEpochBankState::Empty));
                        if (prior == static_cast<uint32_t>(
                                         llaminar2::DeviceMoEOverlayEpochBankState::Candidate))
                        {
                            overlay_atomic_store64(
                                &control->bank_epochs[candidate_bank], 0u);
                        }
                        finish_overlay_publication_status(
                            status,
                            llaminar2::DeviceMoEOverlayEpochOperation::AbortCandidate,
                            llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                            candidate_epoch,
                            old_selector,
                            candidate_bank,
                            static_cast<llaminar2::DeviceMoEOverlayEpochBankState>(prior));
                        valid = 0u;
                    }
                }
            }
        }
        __syncthreads();
        if (valid == 0u)
        {
            // A failed reservation/no-work poll is benign; corrupt publication
            // is terminal. Preserve its commands and poison the existing owner
            // so later replay cannot turn this into an indefinitely busy wave.
            if (threadIdx.x == 0u && status->code == static_cast<uint32_t>(
                    llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl))
            {
                poison_rebalance_graph_controller(
                    controller_state, kDeviceMoERebalanceStatusInvalidRuntime,
                    apply_status->transaction_wave_index, apply_status->transaction_epoch,
                    0u, 0u, apply_status->status_code, controller_state->participant_id);
            }
            return;
        }

        // Reserved payload bytes are not visible history until the selector
        // publishes. Exhaustion skips writes but records the whole missing wave.
        if (journal_required != 0u && journal_append.disposition ==
            llaminar2::DeviceMoEMovementJournalDisposition::Record)
        {
            for (uint32_t i = threadIdx.x; i < journal_append.edge_count; i += blockDim.x)
            {
                const auto &plan = plan_entries[plan_offset + i];
                journal.edges[journal_append.first_edge + i] = {
                    plan.layer, plan.expert, plan.source_participant, plan.destination_participant,
                    plan.activation_count, estimated_expert_bytes};
            }
        }

        for (uint32_t layer = 0u; layer < layer_count; ++layer)
        {
            auto &runtime = runtime_layers[layer];
            auto &candidate = runtime.banks[candidate_bank];
            const auto &published = runtime.banks[published_bank];
            const bool clone_untouched =
                candidate.epoch != static_cast<uint32_t>(candidate_epoch);

            if (clone_untouched)
            {
                for (uint32_t expert = threadIdx.x;
                     expert < expert_count;
                     expert += blockDim.x)
                {
                    candidate.experts[expert] = published.experts[expert];
                    candidate.local_compute_mask[expert] =
                        published.local_compute_mask[expert];
                    candidate.replica_role[expert] =
                        published.replica_role[expert];
                    candidate.resident_participant_mask[expert] =
                        published.resident_participant_mask[expert];
                    candidate.overlay_route_participant[expert] =
                        published.overlay_route_participant[expert];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0u)
            {
                if (clone_untouched)
                {
                    candidate.expert_count = published.expert_count;
                    candidate.multi_resident_expert_count =
                        published.multi_resident_expert_count;
                    candidate.transient_placement_observed =
                        published.transient_placement_observed;
                }
                candidate.epoch = static_cast<uint32_t>(candidate_epoch);
            }
            __syncthreads();
        }

        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            /* Active-bank fields are maintenance metadata only for ticketed
             * inference, but keeping them aligned makes the next controller
             * wave and diagnostics consume the newly published source. */
            for (uint32_t layer = 0u; layer < layer_count; ++layer)
            {
                runtime_layers[layer].active_bank = candidate_bank;
                runtime_layers[layer].active_epoch =
                    static_cast<uint32_t>(candidate_epoch);
            }
            __threadfence();

            const uint32_t candidate_prior = atomicCAS(
                reinterpret_cast<unsigned int *>(
                    &control->bank_states[candidate_bank]),
                static_cast<uint32_t>(
                    llaminar2::DeviceMoEOverlayEpochBankState::Candidate),
                static_cast<uint32_t>(
                    llaminar2::DeviceMoEOverlayEpochBankState::Ready));
            if (candidate_prior != static_cast<uint32_t>(
                                       llaminar2::DeviceMoEOverlayEpochBankState::Candidate))
            {
                finish_overlay_publication_status(
                    status,
                    llaminar2::DeviceMoEOverlayEpochOperation::PublishCandidate,
                    llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                    candidate_epoch,
                    old_selector,
                    candidate_bank,
                    llaminar2::DeviceMoEOverlayEpochBankState::Ready);
                poison_rebalance_graph_controller(
                    controller_state, kDeviceMoERebalanceStatusInvalidRuntime,
                    apply_status->transaction_wave_index, apply_status->transaction_epoch,
                    0u, 0u, apply_status->status_code, controller_state->participant_id);
                return;
            }

            overlay_atomic_store32(
                &control->bank_states[candidate_bank],
                static_cast<uint32_t>(
                    llaminar2::DeviceMoEOverlayEpochBankState::Published));
            const uint64_t next_selector =
                (((old_selector >> 1u) + 1u) << 1u) |
                static_cast<uint64_t>(candidate_bank);
            __threadfence();
            overlay_atomic_store64(
                &control->published_selector, next_selector);
            /* Readers may still finish acquisition against the old selector;
             * Retiring remains acquireable and its guard prevents reclamation. */
            overlay_atomic_store32(
                &control->bank_states[published_bank],
                static_cast<uint32_t>(
                    llaminar2::DeviceMoEOverlayEpochBankState::Retiring));
            *candidate_epoch_ptr = candidate_epoch + 1u;
            // The same device owner publishes placement and its immutable
            // receipt. No telemetry enablement can alter this lifecycle edge.
            if (journal_required != 0u &&
                !llaminar2::commitDeviceMoEMovementJournalAppend(journal, journal_append))
            {
                finish_overlay_publication_status(
                    status, llaminar2::DeviceMoEOverlayEpochOperation::PublishCandidate,
                    llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                    candidate_epoch, next_selector, candidate_bank,
                    llaminar2::DeviceMoEOverlayEpochBankState::Published);
                poison_rebalance_graph_controller(
                    controller_state, kDeviceMoERebalanceStatusInvalidRuntime,
                    apply_status->transaction_wave_index, apply_status->transaction_epoch,
                    0u, 0u, apply_status->status_code, controller_state->participant_id);
                return;
            }
            if (!llaminar2::commitDeviceMoECompletedDecision(
                    controller_state->completed_decision, decision_append))
            {
                finish_overlay_publication_status(
                    status, llaminar2::DeviceMoEOverlayEpochOperation::PublishCandidate,
                    llaminar2::DeviceMoEOverlayEpochStatusCode::InvalidControl,
                    candidate_epoch, next_selector, candidate_bank,
                    llaminar2::DeviceMoEOverlayEpochBankState::Published);
                poison_rebalance_graph_controller(
                    controller_state, kDeviceMoERebalanceStatusInvalidRuntime,
                    apply_status->transaction_wave_index, apply_status->transaction_epoch,
                    0u, 0u, apply_status->status_code, controller_state->participant_id);
                return;
            }
            // The selector is now published. Only this successful finalizer
            // can release the exact command generation and advertise Applied.
            const uint32_t completed_wave = apply_status->transaction_wave_index;
            clear_rebalance_command_header_device(&command_headers[completed_wave]);
            controller_state->waves[completed_wave].state =
                kDeviceMoERebalanceLifecycleApplied;
            controller_state->active_wave =
                (completed_wave + 1u) % command_buffer_count;
            __threadfence();
            finish_overlay_publication_status(
                status,
                llaminar2::DeviceMoEOverlayEpochOperation::PublishCandidate,
                llaminar2::DeviceMoEOverlayEpochStatusCode::Success,
                candidate_epoch,
                next_selector,
                candidate_bank,
                llaminar2::DeviceMoEOverlayEpochBankState::Published);
        }
    }

} // namespace

extern "C"
{
    bool cudaMoE_pack_rebalance_compact_payloads(
        const void *plan_entries,
        const void *command_headers,
        uint32_t plan_capacity,
        const void *local_source_descriptors,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
    {
        if (!plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload || !config || !status || !stream ||
            plan_capacity == 0u || local_payload_slot_count == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        /*
         * Begin the immutable copy transaction before any payload block samples
         * command metadata. The unpack and publication kernels consume this
         * same status ticket, so a later controller-wave advance cannot retarget
         * them.
         */
        begin_rebalance_copy_transaction_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_headers),
            plan_capacity,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        pack_rebalance_compact_payloads_kernel<<<
            static_cast<int>(local_payload_slot_count),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_headers),
            plan_capacity,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_source_descriptors),
            static_cast<uint8_t *>(local_payload),
            local_payload_slot_count,
            payload_slot_bytes,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        return finishLaunch("cudaMoE_pack_rebalance_compact_payloads");
    }

    bool cudaMoE_pack_rebalance_collective_payloads(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        const void *local_directory,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
    {
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || !config || !status ||
            !stream || plan_capacity == 0u || local_payload_slot_count == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        const uint32_t metadata_buffer_count =
            command_buffer_count <= 1u
                ? 1u
                : std::min<uint32_t>(command_buffer_count, 2u);
        const auto *root_command_headers =
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(
                gathered_command_headers) +
            static_cast<unsigned long long>(cfg.root_participant) *
                metadata_buffer_count;
        auto root_cfg = cfg;
        root_cfg.participant_id = cfg.root_participant;
        begin_rebalance_copy_transaction_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            root_command_headers,
            plan_capacity,
            root_cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        pack_rebalance_collective_payloads_kernel<<<
            static_cast<int>(local_payload_slot_count),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoERebalancePlanEntryView *>(gathered_plan_entries),
            static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(gathered_command_headers),
            plan_capacity,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_directory),
            static_cast<uint8_t *>(local_payload),
            local_payload_slot_count,
            payload_slot_bytes,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        return finishLaunch("cudaMoE_pack_rebalance_collective_payloads");
    }

    bool cudaMoE_unpack_rebalance_collective_payloads(
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *gathered_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream)
    {
        if (!plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots || !config || !status ||
            !stream || plan_capacity == 0u || local_payload_slot_count == 0u ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntryView))
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        if (local_transfer_slot_count == 0xffffffffu)
            return false;
        const uint32_t unpack_block_count = local_transfer_slot_count + 1u;
        // Deliberately preserve the status initialized by the matching pack
        // call.  Pack and unpack are two halves of one transfer wave and both
        // must contribute counters to the same status record.
        unpack_rebalance_collective_payloads_kernel<<<
            static_cast<int>(unpack_block_count),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            const_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
                static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header)),
            static_cast<const uint8_t *>(gathered_payload),
            local_payload_slot_count,
            payload_slot_bytes,
            static_cast<DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
            local_transfer_slot_count,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count);
        return finishLaunch("cudaMoE_unpack_rebalance_collective_payloads");
    }

    bool cudaMoE_apply_rebalance_arrivals(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        int target_layer,
        int device_idx,
        void *stream)
    {
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !local_transfer_slots || !config || !status || !stream)
            return false;
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        init_rebalance_apply_status_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoERebalanceApplyStatusView *>(status));
        apply_rebalance_arrivals_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            const_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
                static_cast<const DeviceMoERebalanceCommandBufferHeaderView *>(command_header)),
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
            local_transfer_slot_count,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            target_layer,
            nullptr,
            false,
            1u,
            nullptr);
        return finishLaunch("cudaMoE_apply_rebalance_arrivals");
    }


    bool cudaMoE_apply_ready_rebalance_wave(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        int target_layer,
        uint32_t command_buffer_count,
        const void *overlay_reservation_status,
        int device_idx,
        void *stream)
    {
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !config || !status || !controller_state || !stream)
        {
            return false;
        }
        cudaSetDevice(device_idx);
        const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
        apply_rebalance_arrivals_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(local_transfer_slots),
            local_transfer_slot_count,
            cfg,
            static_cast<DeviceMoERebalanceApplyStatusView *>(status),
            target_layer,
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            true,
            command_buffer_count,
            static_cast<const llaminar2::DeviceMoEOverlayEpochStatus *>(
                overlay_reservation_status));
        return finishLaunch("cudaMoE_apply_ready_rebalance_wave");
    }

    bool cudaMoE_finalize_overlay_rebalance_publication(
        void *runtime_layers,
        uint32_t layer_count,
        uint32_t expert_count,
        llaminar2::DeviceMoEOverlayEpochControl *control,
        uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        const void *apply_status,
        void *controller_state,
        void *command_headers,
        uint32_t command_buffer_count,
        const void *plan_entries,
        uint32_t plan_capacity,
        uint64_t estimated_expert_bytes,
        llaminar2::DeviceMoERebalanceMovementJournalView journal,
        int device_idx,
        void *stream)
    {
        if (!runtime_layers || layer_count == 0u || expert_count == 0u ||
            !control || !candidate_epoch || !status || !apply_status ||
            !controller_state || !command_headers || command_buffer_count == 0u ||
            command_buffer_count > 2u || !plan_entries || plan_capacity == 0u ||
            !stream)
        {
            return false;
        }
        cudaSetDevice(device_idx);
        finalize_overlay_rebalance_publication_kernel<<<
            1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            layer_count,
            expert_count,
            control,
            candidate_epoch,
            status,
            static_cast<const DeviceMoERebalanceApplyStatusView *>(
                apply_status),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_headers),
            command_buffer_count,
            static_cast<const DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_capacity, estimated_expert_bytes, journal);
        return finishLaunch(
            "cudaMoE_finalize_overlay_rebalance_publication");
    }

} // extern "C"
