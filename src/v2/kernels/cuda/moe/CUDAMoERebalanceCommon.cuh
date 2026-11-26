/**
 * @file CUDAMoERebalanceCommon.cuh
 * @brief Shared inline CUDA device helpers for overlay rebalance planning, projection, payload packing, and wave apply.
 */
#pragma once
#include "CUDAMoEKernelsCommon.cuh"

namespace
{
    __device__ __forceinline__ bool rebalance_matrix_desc_ready(
        const DeviceNativeQuantisedMatrixDesc &desc)
    {
        return desc.payload != nullptr &&
               desc.scales != nullptr &&
               desc.n > 0 &&
               desc.k > 0 &&
               desc.blocks_per_row > 0u;
    }

    __device__ __forceinline__ bool rebalance_expert_desc_ready(
        const DeviceMoEExpertDescriptorView &desc)
    {
        // Complete-FFN consumers retain their original, stronger contract.
        if (desc.projection_set != llaminar2::DeviceMoEProjectionSet::CompleteExpert)
            return false;
        const bool native_ready =
            desc.weight_format == llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
            rebalance_matrix_desc_ready(desc.gate) &&
            rebalance_matrix_desc_ready(desc.up) &&
            rebalance_matrix_desc_ready(desc.down);
        const bool floating_ready =
            llaminar2::deviceMoEWeightFormatIsFloating(desc.weight_format) &&
            desc.floating_gate.valid() &&
            desc.floating_up.valid() &&
            desc.floating_down.valid();
        return desc.logical_expert_id >= 0 &&
               (native_ready || floating_ready);
    }

    __device__ __forceinline__ bool rebalance_transfer_desc_ready(
        const DeviceMoEExpertDescriptorView &desc)
    {
        if (!llaminar2::deviceMoEProjectionPayloadValid(desc))
            return false;
        // An original local-F transfer retains the ready-row producer promise.
        // Retargeting preserves these source fields and checks destination spans
        // before copying; its host-authority bank publication still authenticates
        // the destination family/shape. Whole-expert/GateUp APIs stay unchanged.
        if(desc.projection_set==llaminar2::DeviceMoEProjectionSet::IntermediateShard &&
           desc.weight_format==llaminar2::DeviceMoEWeightFormat::NativeQuantised) {
            const auto ready=[](const auto& matrix) {
                return matrix.source_identity_present==1u &&
                    llaminar2::NativeQuantisedDescriptorValidation::publishable(matrix);
            };
            if(!ready(desc.gate)||!ready(desc.up)||!ready(desc.down))return false;
        }

        if (llaminar2::deviceMoEWeightFormatIsFloating(desc.weight_format))
            return llaminar2::deviceMoEFloatingExpertCopyReady(desc);
        return desc.weight_format ==
                   llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
               rebalance_matrix_desc_ready(desc.gate) &&
               rebalance_matrix_desc_ready(desc.up) &&
               (desc.projection_set == llaminar2::DeviceMoEProjectionSet::GateUp ||
                rebalance_matrix_desc_ready(desc.down));
    }

    __device__ __forceinline__ bool rebalance_format_for_codebook(
        uint8_t codebook_id,
        uint8_t &payload_bytes,
        uint8_t &is_asymmetric,
        uint8_t &has_emins)
    {
        payload_bytes = 0u;
        is_asymmetric = 0u;
        has_emins = 0u;
        switch (codebook_id)
        {
        case 0:
        case 4:
            payload_bytes = 16u;
            return true;
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
            payload_bytes = 16u;
            is_asymmetric = 1u; // Physical secondary presence, not additive arithmetic.
            return true;
        case 5:
            payload_bytes = 16u;
            is_asymmetric = 1u;
            return true;
        case 6:
            payload_bytes = 20u;
            return true;
        case 7:
            payload_bytes = 20u;
            is_asymmetric = 1u;
            return true;
        case 8:
            payload_bytes = 24u;
            is_asymmetric = 1u;
            return true;
        case 9:
            payload_bytes = 12u;
            is_asymmetric = 1u;
            return true;
        case 10:
            payload_bytes = 8u;
            is_asymmetric = 1u;
            has_emins = 1u;
            return true;
        case 11:
            payload_bytes = 13u;
            is_asymmetric = 1u; // Physical raw source nibble; no additive minimum.
            return true;
        case 12:
            payload_bytes = 12u;
            return true;
        case 13:
        case 14:
            payload_bytes = 9u;
            is_asymmetric = 1u;
            return true;
        case 15:
            payload_bytes = 8u;
            return true;
        case 16:
        case 17:
            payload_bytes = 6u;
            is_asymmetric = 1u;
            return true;
        case 19:
        case 20:
            payload_bytes = 32u;
            return true;
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            payload_bytes = 32u;
            is_asymmetric = 1u;
            return true;
        default:
            return false;
        }
    }

    __device__ __forceinline__ unsigned long long rebalance_block_count(
        const DeviceNativeQuantisedMatrixDesc &desc)
    {
        return static_cast<unsigned long long>(desc.blocks_per_row) *
               static_cast<unsigned long long>(desc.n);
    }

    __device__ __forceinline__ bool rebalance_projection_format(
        const DeviceNativeQuantisedMatrixDesc &desc,
        uint8_t &payload_bytes,
        uint8_t &is_asymmetric,
        uint8_t &has_emins)
    {
        return rebalance_format_for_codebook(
            desc.codebook_id,
            payload_bytes,
            is_asymmetric,
            has_emins);
    }

    __device__ __forceinline__ bool rebalance_projection_allocation_capacity(
        const DeviceNativeQuantisedMatrixDesc &desc,
        uint8_t &payload_bytes,
        uint8_t &has_mins,
        uint8_t &has_emins)
    {
        if (desc.allocation_payload_bytes_per_block != 0u)
        {
            payload_bytes = desc.allocation_payload_bytes_per_block;
            has_mins = desc.allocation_has_mins;
            has_emins = desc.allocation_has_emins;
            return true;
        }

        uint8_t is_asymmetric = 0u;
        if (!rebalance_projection_format(
                desc,
                payload_bytes,
                is_asymmetric,
                has_emins))
        {
            return false;
        }
        has_mins = is_asymmetric;
        return true;
    }

    __device__ __forceinline__ bool rebalance_matrix_fits_transfer_capacity(
        const DeviceNativeQuantisedMatrixDesc &src,
        const DeviceNativeQuantisedMatrixDesc &dst)
    {
        if (src.n != dst.n ||
            src.k != dst.k ||
            src.blocks_per_row != dst.blocks_per_row)
        {
            return false;
        }

        uint8_t src_payload_bytes = 0u;
        uint8_t src_is_asymmetric = 0u;
        uint8_t src_has_emins = 0u;
        uint8_t dst_payload_capacity = 0u;
        uint8_t dst_has_mins = 0u;
        uint8_t dst_has_emins = 0u;
        return rebalance_projection_format(
                   src,
                   src_payload_bytes,
                   src_is_asymmetric,
                   src_has_emins) &&
               rebalance_projection_allocation_capacity(
                   dst,
                   dst_payload_capacity,
                   dst_has_mins,
                   dst_has_emins) &&
               src_payload_bytes <= dst_payload_capacity &&
               (src_is_asymmetric == 0u || dst_has_mins != 0u) &&
               (src_has_emins == 0u || dst_has_emins != 0u);
    }

    __device__ __forceinline__ void rebalance_retarget_transfer_matrix(
        DeviceNativeQuantisedMatrixDesc &dst,
        const DeviceNativeQuantisedMatrixDesc &src)
    {
        dst.n = src.n;
        dst.k = src.k;
        dst.blocks_per_row = src.blocks_per_row;
        dst.codebook_id = src.codebook_id;
        dst.source_codebook_id = src.source_codebook_id;
        dst.source_is_superblock = src.source_is_superblock;
        dst.source_identity_present = src.source_identity_present;
    }

    __device__ __forceinline__ bool rebalance_matrix_copy_ready(
        const DeviceNativeQuantisedMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_matrix_desc_ready(desc) ||
            !rebalance_projection_format(desc, payload_bytes, is_asymmetric, has_emins) ||
            payload_bytes == 0u)
            return false;
        if (is_asymmetric != 0u && desc.mins == nullptr)
            return false;
        if (has_emins != 0u && desc.emins == nullptr)
            return false;
        return true;
    }

    __device__ __forceinline__ bool rebalance_directory_copy_ready(
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        if (!llaminar2::deviceMoEProjectionPayloadValid(entry.descriptor))
            return false;
        if (llaminar2::deviceMoEWeightFormatIsFloating(entry.descriptor.weight_format))
            return llaminar2::deviceMoEFloatingExpertCopyReady(entry.descriptor);
        return entry.descriptor.weight_format == llaminar2::DeviceMoEWeightFormat::NativeQuantised &&
               rebalance_matrix_copy_ready(entry.descriptor.gate, entry) &&
               rebalance_matrix_copy_ready(entry.descriptor.up, entry) &&
               (entry.descriptor.projection_set == llaminar2::DeviceMoEProjectionSet::GateUp ||
                rebalance_matrix_copy_ready(entry.descriptor.down, entry));
    }

    __device__ __forceinline__ bool rebalance_source_entry_ready(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        const uint32_t required = kDeviceMoEDirectoryFlagValid |
                                  kDeviceMoEDirectoryFlagResident;
        return entry.participant == participant &&
               entry.layer == layer &&
               entry.expert == expert &&
               (entry.resident_mask &
                runtime_participant_bit(static_cast<int>(participant))) != 0u &&
               entry.slot_index != kDeviceMoEInvalidSlot &&
               entry.descriptor.local_slot >= 0 &&
               (entry.flags & required) == required &&
               entry.descriptor.logical_expert_id >= 0 &&
               rebalance_transfer_desc_ready(entry.descriptor) &&
               rebalance_directory_copy_ready(entry);
    }

    __device__ __forceinline__ bool rebalance_transfer_slot_ready(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        (void)layer;
        (void)expert;
        const uint32_t required = kDeviceMoEDirectoryFlagValid |
                                  kDeviceMoEDirectoryFlagTransferSlot;
        return entry.participant == participant &&
               entry.slot_index != kDeviceMoEInvalidSlot &&
               entry.descriptor.local_slot >= 0 &&
               (entry.flags & required) == required &&
               rebalance_transfer_desc_ready(entry.descriptor) &&
               rebalance_directory_copy_ready(entry);
    }

    __device__ __forceinline__ bool rebalance_transfer_slot_copy_complete(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t participant,
        uint32_t layer,
        uint32_t expert)
    {
        return rebalance_transfer_slot_ready(entry, participant, layer, expert) &&
               entry.layer == layer &&
               entry.expert == expert &&
               entry.descriptor.logical_expert_id == static_cast<int32_t>(expert) &&
               (entry.flags & kDeviceMoEDirectoryFlagResident) != 0u &&
               (entry.flags & kDeviceMoEDirectoryFlagCopyComplete) != 0u;
    }

    /**
     * @brief Authenticate a completed copy as the publication for one plan wave.
     *
     * Logical identity alone is insufficient because a transfer slot can retain
     * a valid completed copy from an older wave.  Apply may consume the slot only
     * when unpack performed this plan's exact generation transition and stamped
     * the current command epoch.
     */
    __device__ __forceinline__ bool rebalance_transfer_slot_copy_complete_for_plan(
        const DeviceMoEExpertDirectoryEntryView &entry,
        const DeviceMoERebalancePlanEntryView &plan,
        uint32_t participant,
        uint32_t command_epoch)
    {
        return plan.destination_generation != 0xffffffffu &&
               entry.generation == plan.destination_generation + 1u &&
               entry.epoch == command_epoch &&
               rebalance_transfer_slot_copy_complete(
                   entry,
                   participant,
                   plan.layer,
                   plan.expert);
    }

    __device__ __forceinline__ bool rebalance_directory_fits_transfer_capacity(
        const DeviceMoEExpertDirectoryEntryView &src,
        const DeviceMoEExpertDirectoryEntryView &dst)
    {
        if (src.descriptor.projection_set != dst.descriptor.projection_set ||
            !llaminar2::deviceMoEProjectionPayloadValid(src.descriptor) ||
            !llaminar2::deviceMoEProjectionPayloadValid(dst.descriptor))
            return false;
        if (llaminar2::deviceMoEWeightFormatIsFloating(src.descriptor.weight_format))
            return llaminar2::deviceMoEFloatingExpertFitsTransferCapacity(src.descriptor, dst.descriptor);
        if (src.descriptor.weight_format != llaminar2::DeviceMoEWeightFormat::NativeQuantised)
            return false;
        return rebalance_matrix_fits_transfer_capacity(
                   src.descriptor.gate,
                   dst.descriptor.gate) &&
               rebalance_matrix_fits_transfer_capacity(
                   src.descriptor.up,
                   dst.descriptor.up) &&
               (src.descriptor.projection_set == llaminar2::DeviceMoEProjectionSet::GateUp ||
                rebalance_matrix_fits_transfer_capacity(
                   src.descriptor.down,
                   dst.descriptor.down));
    }

    __device__ __forceinline__ void rebalance_retarget_transfer_directory(
        DeviceMoEExpertDirectoryEntryView &dst,
        const DeviceMoEExpertDirectoryEntryView &src)
    {
        // The destination owns its pointers/capacity; only occupant metadata
        // changes after the complete transaction has passed validation.
        dst.descriptor.weight_format = src.descriptor.weight_format;
        if (llaminar2::deviceMoEWeightFormatIsFloating(src.descriptor.weight_format))
            return;
        rebalance_retarget_transfer_matrix(
            dst.descriptor.gate,
            src.descriptor.gate);
        rebalance_retarget_transfer_matrix(
            dst.descriptor.up,
            src.descriptor.up);
        if (llaminar2::deviceMoEProjectionSetHasDown(src.descriptor.projection_set))
            rebalance_retarget_transfer_matrix(
                dst.descriptor.down,
                src.descriptor.down);
    }

    __device__ __forceinline__ DeviceMoEExpertDirectoryEntryView make_rebalance_source_entry(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        uint32_t layer,
        uint32_t expert)
    {
        DeviceMoEExpertDirectoryEntryView entry{};
        entry.layer = layer;
        entry.expert = expert;
        entry.participant = config.participant_id;
        entry.slot_index = kDeviceMoEInvalidSlot;
        if (!runtime_layers ||
            !rebalance_config_ok(config) ||
            layer >= config.num_layers ||
            expert >= config.num_experts)
        {
            return entry;
        }

        const uint32_t participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
        const bool runtime_ok =
            runtime.active_bank <= 1u &&
            runtime.expert_count == config.num_experts &&
            runtime.top_k == config.top_k &&
            runtime.participant_id == config.participant_id &&
            runtime.participant_count == config.participant_count;
        if (!runtime_ok)
            return entry;

        const DeviceMoEPlacementBankView &bank = runtime.banks[runtime.active_bank];
        const DeviceMoEExpertDescriptorView &desc = bank.experts[expert];
        const uint32_t resident_mask =
            runtime_expert_resident_mask(&runtime, bank, static_cast<int>(expert)) & valid_mask;
        const bool local_resident = (resident_mask & participant_bit) != 0u;
        const bool local_slot_resident = desc.local_slot >= 0;
        entry.epoch = runtime.active_epoch;
        entry.generation = runtime.active_epoch;
        entry.resident_mask = resident_mask;
        if (desc.local_slot >= 0)
            entry.slot_index = static_cast<uint32_t>(desc.local_slot);

        if (local_resident && local_slot_resident && desc.logical_expert_id >= 0 &&
            rebalance_transfer_desc_ready(desc))
        {
            entry.descriptor = desc;
            if (rebalance_directory_copy_ready(entry))
            {
                entry.flags = kDeviceMoEDirectoryFlagValid |
                              kDeviceMoEDirectoryFlagResident;
                if (bank.local_compute_mask[expert] != 0u)
                    entry.flags |= kDeviceMoEDirectoryFlagLocalCompute;
            }
            else
            {
                entry.descriptor = DeviceMoEExpertDescriptorView{};
            }
        }
        return entry;
    }


    __device__ __forceinline__ void init_rebalance_apply_status_device(DeviceMoERebalanceApplyStatusView *status)
    {
        DeviceMoERebalanceApplyStatusView zero{};
        *status = zero;
        status->magic = kDeviceMoERebalanceMagic;
        status->version = kDeviceMoERebalanceVersion;
        status->status_code = 0u;
        status->first_missing_destination_slot = kDeviceMoEInvalidSlot;
        status->first_missing_destination_layer = kDeviceMoEInvalidSlot;
        status->first_missing_destination_expert = kDeviceMoEInvalidSlot;
        status->first_missing_destination_source = kDeviceMoEInvalidSlot;
        status->transaction_wave_index = kDeviceMoEInvalidSlot;
    }

    __device__ __forceinline__ void init_rebalance_apply_status_in_progress_device(DeviceMoERebalanceApplyStatusView *status)
    {
        init_rebalance_apply_status_device(status);
        status->status_code = kDeviceMoERebalanceApplyStatusInProgress;
    }


    __device__ __forceinline__ bool rebalance_command_header_ok(
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceConfigView &config);


    __device__ __forceinline__ bool rebalance_graph_controller_state_ok(
        const DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        return rebalance_graph_controller_state_basic_ok(state, config);
    }

    __device__ void init_rebalance_graph_controller_state_device(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceConfigView &config)
    {
        DeviceMoERebalanceGraphControllerStateView zero{};
        *state = zero;
        state->magic = kDeviceMoERebalanceMagic;
        state->version = kDeviceMoERebalanceVersion;
        state->participant_id = config.participant_id;
        state->participant_count = config.participant_count;
        state->next_epoch = 1u;
        state->active_wave = 0u;
        state->wave_count = 2u;
        state->last_error_wave_index = kDeviceMoEInvalidSlot;
        state->last_error_missing_destination_slot = kDeviceMoEInvalidSlot;
        state->last_error_missing_destination_layer = kDeviceMoEInvalidSlot;
        state->last_error_missing_destination_expert = kDeviceMoEInvalidSlot;
        state->last_error_missing_destination_source = kDeviceMoEInvalidSlot;
        state->decode_rounds_committed = 0u;
        state->decode_rounds_until_maintenance =
            config.initial_maintenance_period_tokens;
        state->maintenance_period_rounds =
            config.maintenance_period_tokens;
        state->maintenance_due = 0u;
        for (uint32_t i = 0; i < 2u; ++i)
        {
            auto &wave = state->waves[i];
            wave.magic = kDeviceMoERebalanceMagic;
            wave.version = kDeviceMoERebalanceVersion;
            wave.state = kDeviceMoERebalanceLifecycleIdle;
            wave.error_participant = kDeviceMoEInvalidSlot;
        }
    }


    __device__ __forceinline__ uint32_t rebalance_wave_next_layer(
        const DeviceMoERebalanceWaveProgressView &wave,
        const DeviceMoERebalanceConfigView &config)
    {
        if (config.num_layers == 0u || wave.planned_layer_count == 0u)
            return config.num_layers;
        const uint32_t layer_offset = min(wave.applied_layer_count, wave.planned_layer_count);
        if (layer_offset >= wave.planned_layer_count)
            return config.num_layers;
        return (wave.planned_start_layer + layer_offset) % config.num_layers;
    }

    struct DeviceMoEReadyWaveSelection
    {
        DeviceMoERebalanceWaveProgressView *wave;
        uint32_t wave_index;
    };

    __device__ DeviceMoEReadyWaveSelection rebalance_find_ready_wave(
        DeviceMoERebalanceGraphControllerStateView *state,
        const DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        const DeviceMoERebalanceConfigView &config,
        int target_layer,
        uint32_t command_buffer_count)
    {
        DeviceMoEReadyWaveSelection selection{nullptr, 0u};
        const uint32_t count = rebalance_command_buffer_count(command_buffer_count);
        if (!rebalance_graph_controller_state_ok(state, config) ||
            state->last_error_code != 0u ||
            !command_headers)
        {
            return selection;
        }

        for (uint32_t i = 0; i < state->wave_count && i < count; ++i)
        {
            auto &wave = state->waves[i];
            const auto &command_header = command_headers[i];
            if (wave.magic != kDeviceMoERebalanceMagic ||
                wave.version != kDeviceMoERebalanceVersion ||
                !rebalance_command_header_ok(&command_header, config) ||
                command_header.epoch == 0u ||
                wave.epoch != command_header.epoch ||
                wave.state != kDeviceMoERebalanceLifecycleReadyToApply)
            {
                continue;
            }
            if (target_layer >= 0)
            {
                const uint32_t next_layer = rebalance_wave_next_layer(wave, config);
                if (next_layer != static_cast<uint32_t>(target_layer))
                    continue;
            }
            selection.wave = &wave;
            selection.wave_index = i;
            return selection;
        }
        return selection;
    }

    /**
     * @brief Encode every structural defect in a participant copy-status row.
     *
     * The transfer-complete publisher consumes an allgather of per-participant
     * copy status records.  A participant can have no destination arrivals for a
     * wave and still be the source for another participant's payload slot, so
     * checking only `copied_arrivals` is not enough.  Source-side pack failures
     * such as missing descriptors must poison the whole transfer wave before any
     * apply kernel can observe an empty payload slot as valid work. Returning a
     * bitset preserves every simultaneous defect in first-error provenance.
     */
    __device__ __forceinline__ uint32_t rebalance_copy_failure_flags(
        const DeviceMoERebalanceApplyStatusView &status,
        bool transaction_matches)
    {
        uint32_t flags = 0u;
        if (status.magic != kDeviceMoERebalanceMagic ||
            status.version != kDeviceMoERebalanceVersion)
            flags |= kDeviceMoECopyFailureInvalidStatusRecord;
        if (status.status_code != 0u)
            flags |= kDeviceMoECopyFailureNonOkStatusCode;
        if (!transaction_matches)
            flags |= kDeviceMoECopyFailureTransactionMismatch;
        if (status.invalid_plan_entries != 0u)
            flags |= kDeviceMoECopyFailureInvalidPlanEntry;
        if (status.missing_source_descriptors != 0u)
            flags |= kDeviceMoECopyFailureMissingSourceDescriptor;
        if (status.missing_destination_slots != 0u)
            flags |= kDeviceMoECopyFailureMissingDestinationSlot;
        if (status.descriptor_mismatches != 0u)
            flags |= kDeviceMoECopyFailureDescriptorMismatch;
        if (status.copy_incomplete != 0u)
            flags |= kDeviceMoECopyFailureCopyIncomplete;
        return flags;
    }

    __device__ __forceinline__ uint32_t rebalance_expected_payload_arrivals(
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t command_count,
        uint32_t participant,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!plan_entries || participant >= config.participant_count)
            return 0u;
        uint32_t expected = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = plan_entries[i];
            if (plan.op == 0u ||
                !rebalance_plan_requires_payload(plan.op) ||
                plan.layer >= config.num_layers ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant != participant)
            {
                continue;
            }
            ++expected;
        }
        return expected;
    }


    __device__ __forceinline__ bool rebalance_command_header_ok(
        const DeviceMoERebalanceCommandBufferHeaderView *command_header,
        const DeviceMoERebalanceConfigView &config)
    {
        return !command_header ||
               (command_header->magic == kDeviceMoERebalanceMagic &&
                command_header->version == kDeviceMoERebalanceVersion &&
                command_header->phase == kDeviceMoERebalancePhasePlanAssignments &&
                command_header->participant_id == config.participant_id &&
                command_header->participant_count == config.participant_count);
    }

    __device__ __forceinline__ bool rebalance_command_header_for_participant_ok(
        const DeviceMoERebalanceCommandBufferHeaderView &command_header,
        uint32_t participant,
        const DeviceMoERebalanceConfigView &config)
    {
        return command_header.magic == kDeviceMoERebalanceMagic &&
               command_header.version == kDeviceMoERebalanceVersion &&
               command_header.phase == kDeviceMoERebalancePhasePlanAssignments &&
               command_header.participant_id == participant &&
               command_header.participant_count == config.participant_count;
    }

    __device__ __forceinline__ uint32_t rebalance_command_count(
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeaderView *command_header)
    {
        if (command_header)
            return min(command_header->command_count,
                       min(command_header->command_capacity, plan_capacity));
        return plan_count ? min(*plan_count, plan_capacity) : 0u;
    }

    __device__ __forceinline__ unsigned long long rebalance_source_descriptor_local_index(
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_buffer_count,
        uint32_t plan_capacity,
        uint32_t destination_participant,
        uint32_t command_buffer_index,
        uint32_t plan_index)
    {
        (void)config;
        return ((static_cast<unsigned long long>(destination_participant) *
                 static_cast<unsigned long long>(command_buffer_count) +
                 static_cast<unsigned long long>(command_buffer_index)) *
                static_cast<unsigned long long>(plan_capacity)) +
               static_cast<unsigned long long>(plan_index);
    }

    /**
     * @brief Validate the immutable physical identity of one transfer slot.
     *
     * The logical expert stored in a transfer slot changes over time, while the
     * allocation, slot index, participant, and pointer-bearing descriptor stay
     * fixed for the lifetime of the graph.  Projection must reject a malformed
     * directory entry before it can mint a lease for that allocation.
     */
    __device__ __forceinline__ bool rebalance_transfer_slot_identity_ok(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t slot_index,
        const DeviceMoERebalanceConfigView &config)
    {
        /*
         * `slot_index` is the subscript in the transfer-directory array.
         * `entry.slot_index` is the stable local expert-allocation identity.
         * Production currently numbers both from zero, but that is a layout
         * choice rather than an ABI invariant.  The descriptor and directory
         * entry must agree about the stable allocation; they need not equal the
         * directory subscript used to locate this entry.
         */
        (void)slot_index;
        return entry.participant == config.participant_id &&
               entry.slot_index != kDeviceMoEInvalidSlot &&
               entry.descriptor.local_slot == static_cast<int32_t>(entry.slot_index) &&
               entry.generation != 0xffffffffu &&
               rebalance_transfer_slot_ready(
                   entry,
                   config.participant_id,
                   /*layer=*/0u,
                   /*expert=*/0u);
    }

    /**
     * @brief Test whether a directory occupant is still published by runtime.
     *
     * A decode-maintenance copy must never overwrite storage referenced by an
     * active placement bank.  This remains true when another command in the
     * same wave intends to transfer ownership away: unpack happens before the
     * per-layer apply point, so early overwrite would leave the old descriptor
     * temporarily pointing at another expert's bytes.
     */
    __device__ __forceinline__ bool rebalance_transfer_slot_has_active_runtime_claim(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t slot_index,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!runtime_layers ||
            !claim_index ||
            slot_index >= claim_index->slot_count ||
            entry.layer >= config.num_layers ||
            entry.expert >= config.num_experts ||
            (entry.flags & (kDeviceMoEDirectoryFlagResident |
                            kDeviceMoEDirectoryFlagCopyComplete)) !=
                (kDeviceMoEDirectoryFlagResident |
                 kDeviceMoEDirectoryFlagCopyComplete))
        {
            return false;
        }

        const auto *const claims =
            rebalance_transfer_slot_claim_index_entries(claim_index);
        const uint32_t expected_flat =
            entry.layer * config.num_experts + entry.expert;
        if (claims[slot_index].claim_count != 1u ||
            claims[slot_index].first_claim_flat != expected_flat)
        {
            return false;
        }

        const auto &runtime = runtime_layers[entry.layer];
        const uint32_t execution_bank =
            runtime_execution_bank(&runtime);
        if (execution_bank > 1u ||
            runtime.expert_count != config.num_experts ||
            runtime.participant_id != config.participant_id ||
            runtime.participant_count != config.participant_count)
        {
            return false;
        }

        const auto &active =
            runtime_placement_banks(&runtime)[execution_bank];
        const auto &descriptor = active.experts[entry.expert];
        const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        return descriptor.logical_expert_id == static_cast<int32_t>(entry.expert) &&
               descriptor.local_slot == entry.descriptor.local_slot &&
               descriptor.owner_participant == entry.descriptor.owner_participant &&
               (descriptor.flags & kDeviceMoEFlagTransferSlot) != 0u &&
               (active.resident_participant_mask[entry.expert] & local_bit) != 0u;
    }

    /**
     * @brief Return whether maintenance may replace one cold cached replica.
     *
     * A full bounded directory may contain remote-owned replicas retained from
     * earlier placement waves. They remain real runtime claims, so the retired
     * generation pass must protect them. A replica becomes replaceable only
     * when exactly one active runtime descriptor names the allocation, that
     * descriptor is the directory entry's declared occupant, the local
     * participant is not its authoritative owner, and the current placement
     * does not execute the expert locally.
     *
     * The projected plan records the old layer, expert, and generation. Apply
     * authenticates the completed copy against that ticket and retires this
     * exact replica while publishing the incoming expert. Any duplicate or
     * incoherent claim fails closed here.
     */
    __device__ __forceinline__ bool
    rebalance_transfer_slot_is_evictable_cached_replica(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t slot_index,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!runtime_layers ||
            !rebalance_transfer_slot_has_active_runtime_claim(
                entry,
                slot_index,
                runtime_layers,
                config,
                claim_index))
        {
            return false;
        }

        const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        const auto &runtime = runtime_layers[entry.layer];
        const auto &active = runtime.banks[runtime.active_bank];
        const auto occupancy =
            llaminar2::moe_rebalance_policy::classifyTransferSlotOccupancy(
                /*transfer_backed=*/true,
                /*assigned_locally=*/
                    active.local_compute_mask[entry.expert] != 0u,
                active.resident_participant_mask[entry.expert],
                local_bit,
                active.experts[entry.expert].owner_participant,
                config.participant_id);
        return occupancy.occupied && !occupancy.protected_from_reuse;
    }

    /**
     * @brief Decide whether a slot generation has no live device owner.
     *
     * Directory epochs belong to the domain-wide transfer transaction, while
     * runtime active epochs are participant-local placement-bank counters.
     * Comparing those unrelated counters can permanently pin an applied slot:
     * transaction epoch 23 is not "newer" than placement epoch 4 in any useful
     * sense. Liveness is explicit instead: a runtime descriptor protects its
     * physical slot. Projection executes at the graph-ordered transaction
     * boundary, after the previous copy/apply wave has either completed or
     * failed fatally, so command-header epoch equality is not ownership. In
     * particular, projection has already published the next ping-pong headers
     * before it leases slots; treating those new header numbers as prior-wave
     * owners falsely pins stale entries whenever transaction epochs coincide.
     * With no active runtime owner, the old logical publication is retired and
     * its allocation may be leased.
     */
    __device__ __forceinline__ bool rebalance_transfer_slot_generation_retired(
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint32_t slot_index,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!runtime_layers ||
            !claim_index ||
            slot_index >= claim_index->slot_count ||
            rebalance_transfer_slot_claim_index_entries(
                claim_index)[slot_index].claim_count != 0u)
        {
            return false;
        }

        const bool empty =
            entry.layer == kDeviceMoEInvalidSlot &&
            entry.expert == kDeviceMoEInvalidSlot &&
            (entry.flags & (kDeviceMoEDirectoryFlagResident |
                            kDeviceMoEDirectoryFlagCopyComplete)) == 0u;
        if (empty)
            return true;

        if (entry.layer >= config.num_layers ||
            entry.expert >= config.num_experts)
        {
            return false;
        }
        return true;
    }

    __device__ __forceinline__ bool rebalance_transfer_slot_selected(
        const DeviceMoERebalancePlanEntryView *selected_plans,
        uint32_t selected_plan_count,
        uint32_t slot_index,
        uint32_t local_participant)
    {
        if (!selected_plans)
            return false;
        for (uint32_t index = 0u; index < selected_plan_count; ++index)
        {
            const auto &plan = selected_plans[index];
            if (rebalance_plan_requires_payload(plan.op) &&
                plan.destination_participant == local_participant &&
                plan.destination_slot == slot_index)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Return whether this slot's live occupant leaves in the projected wave.
     *
     * A bounded transfer-slot pool can legitimately execute a swap with one slot
     * per participant.  The outgoing payload is packed into the persistent
     * collective buffer before any destination unpack mutates a transfer slot, so
     * an ownership transfer out of this participant is a stream-ordered release
     * of the occupant's physical storage.  Merely sending a replica is not enough:
     * only OwnershipTransfer removes the local runtime claim during apply.
     *
     * @param entry             Current physical transfer-slot publication.
     * @param wave_plan_entries Immutable root commands for this transaction wave.
     * @param wave_plan_count   Number of valid commands in the root wave.
     * @param config            Local participant and domain geometry.
     * @return true only when the exact live occupant is transferred away.
     */
    __device__ __forceinline__ bool
    rebalance_transfer_slot_occupant_departs_in_wave(
        const DeviceMoEExpertDirectoryEntryView &entry,
        const DeviceMoERebalancePlanEntryView *wave_plan_entries,
        uint32_t wave_plan_count,
        const DeviceMoERebalanceConfigView &config)
    {
        if (!wave_plan_entries ||
            entry.layer >= config.num_layers ||
            entry.expert >= config.num_experts)
        {
            return false;
        }

        const uint32_t local_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        for (uint32_t plan_index = 0u;
             plan_index < wave_plan_count;
             ++plan_index)
        {
            const auto &outgoing = wave_plan_entries[plan_index];
            if (outgoing.op == kDeviceMoERebalancePlanOwnershipTransfer &&
                outgoing.layer == entry.layer &&
                outgoing.expert == entry.expert &&
                outgoing.source_participant == config.participant_id &&
                outgoing.destination_participant < config.participant_count &&
                outgoing.destination_participant != config.participant_id &&
                (outgoing.source_resident_mask & local_bit) != 0u)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Lease a destination-local transfer slot from live device state.
     *
     * Root plans describe logical movement only.  The destination participant
     * owns physical slot assignment because only it can observe its current
     * directory and runtime banks.  The returned command carries the exact
     * prior occupant and generation so unpack can perform compare-and-replace
     * instead of trusting a stale integer slot index.
     */
    __device__ __forceinline__ bool rebalance_lease_local_transfer_slot(
        DeviceMoERebalancePlanEntryView &plan,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalancePlanEntryView *selected_plans,
        uint32_t selected_plan_count,
        const DeviceMoERebalancePlanEntryView *wave_plan_entries,
        uint32_t wave_plan_count,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (!runtime_layers ||
            !local_transfer_slots ||
            (selected_plan_count != 0u && !selected_plans) ||
            local_transfer_slot_count == 0u ||
            local_transfer_slot_count > kDeviceMoEMaxTransferSlots)
        {
            return false;
        }

        uint32_t selected_slot = kDeviceMoEInvalidSlot;

        /*
         * Reusing the current slot for the same logical expert preserves the
         * runtime pointer identity and is preferable to consuming a new slot.
         */
        for (uint32_t slot = 0u;
             slot < local_transfer_slot_count;
             ++slot)
        {
            const auto &entry = local_transfer_slots[slot];
            if (!rebalance_transfer_slot_selected(
                    selected_plans,
                    selected_plan_count,
                    slot,
                    config.participant_id) &&
                rebalance_transfer_slot_identity_ok(entry, slot, config) &&
                entry.layer == plan.layer &&
                entry.expert == plan.expert &&
                rebalance_transfer_slot_has_active_runtime_claim(
                    entry,
                    slot,
                    runtime_layers,
                    config,
                    claim_index))
            {
                selected_slot = slot;
                break;
            }
        }

        /*
         * Otherwise consume only storage whose previous publication is already
         * retired.
         */
        for (uint32_t slot = 0u;
             selected_slot == kDeviceMoEInvalidSlot &&
             slot < local_transfer_slot_count;
             ++slot)
        {
            const auto &entry = local_transfer_slots[slot];
            if (!rebalance_transfer_slot_selected(
                    selected_plans,
                    selected_plan_count,
                    slot,
                    config.participant_id) &&
                rebalance_transfer_slot_identity_ok(entry, slot, config) &&
                rebalance_transfer_slot_generation_retired(
                    entry,
                    slot,
                    runtime_layers,
                    config,
                    claim_index))
            {
                selected_slot = slot;
            }
        }

        /*
         * If no retired storage exists, permit a true same-wave swap.
         * Projection only mints the lease; it does not mutate the slot. Stage
         * ordering is:
         *
         *   project lease -> pack outgoing bytes -> collective -> unpack incoming
         *
         * so the live occupant remains readable until its bytes are safely in the
         * collective payload. The generation ticket below then authenticates that
         * no other transaction changed the slot before incoming unpack.
         */
        for (uint32_t slot = 0u;
             selected_slot == kDeviceMoEInvalidSlot &&
             slot < local_transfer_slot_count;
             ++slot)
        {
            const auto &entry = local_transfer_slots[slot];
            if (!rebalance_transfer_slot_selected(
                    selected_plans,
                    selected_plan_count,
                    slot,
                    config.participant_id) &&
                rebalance_transfer_slot_identity_ok(entry, slot, config) &&
                rebalance_transfer_slot_has_active_runtime_claim(
                    entry,
                    slot,
                    runtime_layers,
                    config,
                    claim_index) &&
                rebalance_transfer_slot_occupant_departs_in_wave(
                    entry,
                    wave_plan_entries,
                    wave_plan_count,
                    config))
            {
                selected_slot = slot;
            }
        }

        /*
         * A full cache may still contain cold remote-owned replicas. Reuse one
         * only after a complete runtime scan proves that no owner, active
         * assignment, or duplicate descriptor protects the allocation.
         */
        for (uint32_t slot = 0u;
             selected_slot == kDeviceMoEInvalidSlot &&
             slot < local_transfer_slot_count;
             ++slot)
        {
            const auto &entry = local_transfer_slots[slot];
            if (!rebalance_transfer_slot_selected(
                    selected_plans,
                    selected_plan_count,
                    slot,
                    config.participant_id) &&
                rebalance_transfer_slot_identity_ok(entry, slot, config) &&
                rebalance_transfer_slot_is_evictable_cached_replica(
                    entry,
                    slot,
                    runtime_layers,
                    config,
                    claim_index))
            {
                selected_slot = slot;
            }
        }

        if (selected_slot == kDeviceMoEInvalidSlot)
            return false;

        const auto &prior = local_transfer_slots[selected_slot];
        plan.destination_slot = selected_slot;
        plan.destination_previous_layer = prior.layer;
        plan.destination_previous_expert = prior.expert;
        plan.destination_generation = prior.generation;
        return true;
    }

    /**
     * @brief Lease physical storage for one projected prefill LLEP arrival.
     *
     * Current-batch materialization owns only logical movement and compact
     * collective payload numbering. Physical transfer-slot selection belongs
     * here, on the destination participant, because this projection point sees
     * the complete local directory and every layer's active runtime bank.
     *
     * Ordinary lease selection first preserves an existing copy of the same
     * expert, then consumes empty or retired storage. If the bounded active
     * cache is full, prefill may replace one remote-owned replica from the same
     * layer only when that expert has no rows in the current device-owned
     * assignment. Authoritative local experts and replicas used by the current
     * grouped pass are never evictable.
     */
    __device__ __forceinline__ bool rebalance_lease_prefill_transfer_slot(
        DeviceMoERebalancePlanEntryView &plan,
        const DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfigView &config,
        const DeviceMoERebalancePlanEntryView *selected_plans,
        uint32_t selected_plan_count,
        const DeviceMoETransferSlotClaimIndexView *claim_index)
    {
        if (rebalance_lease_local_transfer_slot(
                plan,
                runtime_layers,
                local_transfer_slots,
                local_transfer_slot_count,
                config,
                selected_plans,
                selected_plan_count,
                /*wave_plan_entries=*/nullptr,
                /*wave_plan_count=*/0u,
                claim_index))
        {
            return true;
        }

        if (!runtime_layers ||
            !local_transfer_slots ||
            (selected_plan_count != 0u && !selected_plans) ||
            plan.layer >= config.num_layers)
        {
            return false;
        }

        const auto &target_runtime = runtime_layers[plan.layer];
        if (target_runtime.active_bank > 1u ||
            target_runtime.expert_count != config.num_experts ||
            target_runtime.participant_id != config.participant_id ||
            target_runtime.participant_count != config.participant_count)
        {
            return false;
        }

        const auto *spans =
            static_cast<const llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentSpan *>(
                target_runtime.reserved_ptrs[1]);
        const uint32_t span_count =
            target_runtime.reserved_u64[2] > 0xffffffffULL
                ? 0xffffffffu
                : static_cast<uint32_t>(target_runtime.reserved_u64[2]);
        uint32_t selected_slot = kDeviceMoEInvalidSlot;
        for (uint32_t slot_index = 0u;
             slot_index < local_transfer_slot_count;
             ++slot_index)
        {
            const auto &prior = local_transfer_slots[slot_index];
            if (rebalance_transfer_slot_selected(
                    selected_plans,
                    selected_plan_count,
                    slot_index,
                    config.participant_id) ||
                !rebalance_transfer_slot_identity_ok(
                    prior,
                    slot_index,
                    config))
            {
                continue;
            }

            /*
             * Request reset intentionally preserves model-lifetime transfer
             * allocations while removing every request-local runtime claim.
             * Their directory generations can therefore be newer than the
             * restored runtime epoch. The allocator derives liveness from the
             * active device runtime, never by comparing those unrelated epoch
             * domains. Prefill additionally uses the next route assignment to
             * identify cross-layer replicas that are no longer read by this
             * forward pass.
             *
             * Keep malformed logical identities out of the lease pool. Empty
             * entries were already consumed by the ordinary allocator above.
             */
            if (prior.layer >= config.num_layers ||
                prior.expert >= config.num_experts)
            {
                continue;
            }
            const bool has_active_runtime_claim =
                rebalance_transfer_slot_has_active_runtime_claim(
                    prior,
                    slot_index,
                    runtime_layers,
                    config,
                    claim_index);
            if (!has_active_runtime_claim)
            {
                selected_slot = slot_index;
                break;
            }

            bool assigned_locally = false;
            if (prior.layer == plan.layer)
            {
                for (uint32_t span_index = 0u;
                     spans && span_index < span_count;
                     ++span_index)
                {
                    const auto &span = spans[span_index];
                    if (span.expert == prior.expert &&
                        span.destination_participant == config.participant_id &&
                        span.route_row_end > span.route_row_begin)
                    {
                        assigned_locally = true;
                        break;
                    }
                }
            }

            /*
             * The compute-ready event before projection establishes that every
             * earlier layer has finished reading its payload. Future layers
             * will publish a fresh assignment before they execute. Therefore
             * only a same-layer assignment can make this non-owner replica live
             * for the current prefill stage; apply carries the previous layer,
             * expert, and generation and retires that exact old publication.
             */
            assigned_locally =
                llaminar2::moe_rebalance_policy::
                    prefillAssignmentReadsTransferSlotOccupant(
                        plan.layer,
                        prior.layer,
                        assigned_locally);
            const auto &occupant_runtime = runtime_layers[prior.layer];
            if (occupant_runtime.active_bank > 1u ||
                occupant_runtime.expert_count != config.num_experts ||
                occupant_runtime.participant_id != config.participant_id ||
                occupant_runtime.participant_count != config.participant_count)
            {
                continue;
            }
            const auto &active_bank =
                occupant_runtime.banks[occupant_runtime.active_bank];
            const auto occupancy =
                llaminar2::moe_rebalance_policy::classifyTransferSlotOccupancy(
                    /*transfer_backed=*/true,
                    assigned_locally,
                    active_bank.resident_participant_mask[prior.expert],
                    llaminar2::moe_rebalance_policy::participantBit(
                        config.participant_id),
                    prior.descriptor.owner_participant,
                    config.participant_id);
            if (occupancy.occupied && !occupancy.protected_from_reuse)
            {
                selected_slot = slot_index;
                break;
            }
        }

        if (selected_slot == kDeviceMoEInvalidSlot)
            return false;

        const auto &prior = local_transfer_slots[selected_slot];
        plan.destination_slot = selected_slot;
        plan.destination_previous_layer = prior.layer;
        plan.destination_previous_expert = prior.expert;
        plan.destination_generation = prior.generation;
        return true;
    }


    __device__ void rebalance_copy_bytes(const void *src, const void *dst_const, unsigned long long bytes)
    {
        const auto *src_bytes = static_cast<const uint8_t *>(src);
        auto *dst_bytes = const_cast<uint8_t *>(static_cast<const uint8_t *>(dst_const));
        if (!src_bytes || !dst_bytes || bytes == 0ULL)
            return;
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0xFULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint4 *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint4 *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint4);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x7ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint2 *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint2 *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint2);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x3ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint32_t *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint32_t *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint32_t);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        if ((((reinterpret_cast<uintptr_t>(src_bytes) |
               reinterpret_cast<uintptr_t>(dst_bytes) |
               static_cast<uintptr_t>(bytes)) &
              0x1ULL) == 0u))
        {
            const auto *src_vec = reinterpret_cast<const uint16_t *>(src_bytes);
            auto *dst_vec = reinterpret_cast<uint16_t *>(dst_bytes);
            const unsigned long long vec_count = bytes / sizeof(uint16_t);
            for (unsigned long long i = threadIdx.x; i < vec_count; i += blockDim.x)
                dst_vec[i] = src_vec[i];
            return;
        }
        for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
            dst_bytes[i] = src_bytes[i];
    }

    __device__ __forceinline__ unsigned long long rebalance_projection_payload_bytes(
        const DeviceNativeQuantisedMatrixDesc &desc,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(desc, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(desc);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        return blocks * static_cast<unsigned long long>(payload_bytes_per_block) +
               scales_bytes +
               (is_asymmetric != 0u ? scales_bytes : 0ULL) +
               (has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL);
    }

    __device__ __forceinline__ unsigned long long rebalance_expert_payload_bytes(
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        if (llaminar2::deviceMoEWeightFormatIsFloating(entry.descriptor.weight_format))
        {
            const auto &expert = entry.descriptor;
            return llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_gate, expert.weight_format) +
                   llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_up, expert.weight_format) +
                   llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_down, expert.weight_format);
        }
        return rebalance_projection_payload_bytes(entry.descriptor.gate, entry) +
               rebalance_projection_payload_bytes(entry.descriptor.up, entry) +
               (llaminar2::deviceMoEProjectionSetHasDown(entry.descriptor.projection_set)
                    ? rebalance_projection_payload_bytes(entry.descriptor.down, entry) : 0ULL);
    }

    __device__ unsigned long long rebalance_pack_projection_to_payload(
        const DeviceNativeQuantisedMatrixDesc &src,
        const DeviceMoEExpertDirectoryEntryView &entry,
        uint8_t *payload)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(src, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(src);
        const unsigned long long payload_bytes =
            blocks * static_cast<unsigned long long>(payload_bytes_per_block);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        const unsigned long long mins_bytes =
            is_asymmetric != 0u ? scales_bytes : 0ULL;
        const unsigned long long emins_bytes =
            has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL;

        unsigned long long offset = 0ULL;
        rebalance_copy_bytes(src.payload, payload + offset, payload_bytes);
        offset += payload_bytes;
        rebalance_copy_bytes(src.scales, payload + offset, scales_bytes);
        offset += scales_bytes;
        if (mins_bytes > 0ULL)
        {
            rebalance_copy_bytes(src.mins, payload + offset, mins_bytes);
            offset += mins_bytes;
        }
        if (emins_bytes > 0ULL)
        {
            rebalance_copy_bytes(src.emins, payload + offset, emins_bytes);
            offset += emins_bytes;
        }
        return offset;
    }

    __device__ unsigned long long rebalance_unpack_projection_from_payload(
        const uint8_t *payload,
        const DeviceNativeQuantisedMatrixDesc &dst,
        const DeviceMoEExpertDirectoryEntryView &entry)
    {
        (void)entry;
        uint8_t payload_bytes_per_block = 0u;
        uint8_t is_asymmetric = 0u;
        uint8_t has_emins = 0u;
        if (!rebalance_projection_format(dst, payload_bytes_per_block, is_asymmetric, has_emins))
            return 0ULL;
        const unsigned long long blocks = rebalance_block_count(dst);
        const unsigned long long payload_bytes =
            blocks * static_cast<unsigned long long>(payload_bytes_per_block);
        const unsigned long long scales_bytes = blocks * sizeof(uint16_t);
        const unsigned long long mins_bytes =
            is_asymmetric != 0u ? scales_bytes : 0ULL;
        const unsigned long long emins_bytes =
            has_emins != 0u ? blocks * sizeof(uint32_t) : 0ULL;

        unsigned long long offset = 0ULL;
        rebalance_copy_bytes(payload + offset, dst.payload, payload_bytes);
        offset += payload_bytes;
        rebalance_copy_bytes(payload + offset, dst.scales, scales_bytes);
        offset += scales_bytes;
        if (mins_bytes > 0ULL)
        {
            rebalance_copy_bytes(payload + offset, dst.mins, mins_bytes);
            offset += mins_bytes;
        }
        if (emins_bytes > 0ULL)
        {
            rebalance_copy_bytes(payload + offset, dst.emins, emins_bytes);
            offset += emins_bytes;
        }
        return offset;
    }

    /** @brief Pack an exact floating or quantized expert using the same vector copy. */
    __device__ unsigned long long rebalance_pack_expert_to_payload(
        const DeviceMoEExpertDirectoryEntryView &src, uint8_t *payload)
    {
        unsigned long long offset = 0;
        if (llaminar2::deviceMoEWeightFormatIsFloating(src.descriptor.weight_format))
        {
            const auto &expert = src.descriptor;
            const auto gate_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_gate, expert.weight_format);
            const auto up_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_up, expert.weight_format);
            const auto down_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_down, expert.weight_format);
            rebalance_copy_bytes(expert.floating_gate.data, payload, gate_bytes);
            rebalance_copy_bytes(expert.floating_up.data, payload + gate_bytes, up_bytes);
            rebalance_copy_bytes(expert.floating_down.data, payload + gate_bytes + up_bytes, down_bytes);
            return gate_bytes + up_bytes + down_bytes;
        }
        offset += rebalance_pack_projection_to_payload(src.descriptor.gate, src, payload + offset);
        offset += rebalance_pack_projection_to_payload(src.descriptor.up, src, payload + offset);
        if (llaminar2::deviceMoEProjectionSetHasDown(src.descriptor.projection_set))
            offset += rebalance_pack_projection_to_payload(src.descriptor.down, src, payload + offset);
        return offset;
    }

    /**
     * @brief Copy a validated arrival into destination-owned pointers.
     * Retargeting has already selected the source dtype without replacing the
     * destination allocation capacity. Publication remains after the existing
     * copy-complete fence, regardless of arithmetic family.
     */
    __device__ unsigned long long rebalance_unpack_expert_from_payload(
        const uint8_t *payload, const DeviceMoEExpertDirectoryEntryView &dst)
    {
        unsigned long long offset = 0;
        if (llaminar2::deviceMoEWeightFormatIsFloating(dst.descriptor.weight_format))
        {
            const auto &expert = dst.descriptor;
            const auto gate_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_gate, expert.weight_format);
            const auto up_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_up, expert.weight_format);
            const auto down_bytes = llaminar2::deviceMoEFloatingMatrixBytes(expert.floating_down, expert.weight_format);
            rebalance_copy_bytes(payload, const_cast<void *>(expert.floating_gate.data), gate_bytes);
            rebalance_copy_bytes(payload + gate_bytes, const_cast<void *>(expert.floating_up.data), up_bytes);
            rebalance_copy_bytes(payload + gate_bytes + up_bytes, const_cast<void *>(expert.floating_down.data), down_bytes);
            return gate_bytes + up_bytes + down_bytes;
        }
        offset += rebalance_unpack_projection_from_payload(payload + offset, dst.descriptor.gate, dst);
        offset += rebalance_unpack_projection_from_payload(payload + offset, dst.descriptor.up, dst);
        if (llaminar2::deviceMoEProjectionSetHasDown(dst.descriptor.projection_set))
            offset += rebalance_unpack_projection_from_payload(payload + offset, dst.descriptor.down, dst);
        return offset;
    }


    /**
     * @brief Report whether a ready local arrival replaces an expert's slot.
     *
     * Transfer-directory allocations are stable, but their logical expert
     * contents are not. Runtime publication must therefore remove the exact
     * previous logical occupant stamped into the destination lease whenever a
     * ready arrival reuses the same stable `local_slot`.  Looking up the prior
     * identity from the post-copy directory is too late: that directory already
     * names the new expert.
     *
     * @param plans                 Current device-owned rebalance plan.
     * @param plan_count            Number of entries in @p plans.
     * @param transfer_slots        Destination transfer directory.
     * @param transfer_slot_count  Number of stable destination slots.
     * @param config                Participant and model geometry.
     * @param layer                 Runtime layer being rebuilt.
     * @param expert                Existing logical expert under inspection.
     * @param descriptor            Existing expert descriptor.
     * @return true when another expert's ready arrival owns the same slot.
     */
    __device__ __forceinline__ bool
    rebalance_ready_arrival_reuses_runtime_slot(
        const DeviceMoERebalancePlanEntryView *plans,
        uint32_t plan_count,
        const DeviceMoEExpertDirectoryEntryView *transfer_slots,
        uint32_t transfer_slot_count,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_epoch,
        uint32_t layer,
        uint32_t expert,
        const DeviceMoEExpertDescriptorView &descriptor,
        int target_arrival_layer = -1)
    {
        if (!plans ||
            !transfer_slots ||
            descriptor.local_slot < 0 ||
            (descriptor.flags & kDeviceMoEFlagTransferSlot) == 0u)
        {
            return false;
        }

        for (uint32_t plan_index = 0u;
             plan_index < plan_count;
             ++plan_index)
        {
            const auto &plan = plans[plan_index];
            if ((target_arrival_layer >= 0 &&
                 plan.layer !=
                     static_cast<uint32_t>(target_arrival_layer)) ||
                plan.destination_previous_layer != layer ||
                plan.destination_previous_expert != expert ||
                (plan.layer == layer && plan.expert == expert) ||
                plan.expert >= config.num_experts ||
                plan.destination_participant != config.participant_id ||
                !rebalance_plan_requires_payload(plan.op) ||
                plan.destination_slot >= transfer_slot_count)
            {
                continue;
            }

            const auto &slot = transfer_slots[plan.destination_slot];
            if (slot.descriptor.local_slot == descriptor.local_slot &&
                rebalance_transfer_slot_copy_complete_for_plan(
                    slot,
                    plan,
                    config.participant_id,
                    command_epoch))
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Retire a cached replica whose stable transfer slot was reused.
     *
     * The logical descriptor remains in the runtime inventory so its owner and
     * resident-domain metadata continue to describe the expert, but it no
     * longer advertises any local payload pointer or compute eligibility.
     * Authoritative local copies are never evictable: seeing one here means the
     * materializer violated its protection contract and is fatal.
     */
    __device__ __forceinline__ void
    rebalance_retire_reused_transfer_slot_replica(
        DeviceMoEExpertDescriptorView &descriptor,
        uint32_t &resident_mask,
        uint32_t local_participant_bit,
        uint32_t local_participant)
    {
        if (descriptor.owner_participant ==
            static_cast<int32_t>(local_participant))
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "LLEP arrival attempted to overwrite an authoritative local transfer slot");
        }

        rebalance_retire_local_payload_publication(
            descriptor,
            resident_mask,
            local_participant_bit);
    }

    /**
     * @brief Retire cross-layer occupants reused by one graph-applied layer.
     *
     * The atomic apply kernel rebuilds every changed layer in one launch. Decode
     * graph apply advances one target layer at a time, so it must explicitly
     * rebuild any different layer whose transfer allocation is replaced by an
     * arrival for the current target. Otherwise the old bank keeps a descriptor
     * for bytes that the transfer directory has already overwritten.
     *
     * @return Number of additional runtime layers published by retirement.
     */
    __device__ __forceinline__ uint32_t
    rebalance_retire_cross_layer_slot_occupants_thread0(
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plans,
        uint32_t plan_count,
        const DeviceMoEExpertDirectoryEntryView *transfer_slots,
        uint32_t transfer_slot_count,
        const DeviceMoERebalanceConfigView &config,
        uint32_t command_epoch,
        uint32_t target_layer)
    {
        if (threadIdx.x != 0 ||
            !runtime_layers ||
            !plans ||
            !transfer_slots ||
            target_layer >= config.num_layers)
        {
            return 0u;
        }

        const uint32_t local_participant_bit =
            runtime_participant_bit(
                static_cast<int>(config.participant_id));
        const uint32_t valid_mask =
            runtime_valid_participant_mask(config.participant_count);
        uint32_t changed_layers = 0u;

        for (uint32_t prior_layer = 0u;
             prior_layer < config.num_layers;
             ++prior_layer)
        {
            if (prior_layer == target_layer)
                continue;

            bool layer_reused = false;
            bool current_batch_llep_reuse = false;
            for (uint32_t plan_index = 0u;
                 plan_index < plan_count;
                 ++plan_index)
            {
                const auto &plan = plans[plan_index];
                if (plan.layer == target_layer &&
                    plan.destination_previous_layer == prior_layer &&
                    plan.destination_previous_expert <
                        config.num_experts &&
                    plan.destination_participant ==
                        config.participant_id &&
                    rebalance_plan_requires_payload(plan.op) &&
                    plan.destination_slot < transfer_slot_count &&
                    rebalance_transfer_slot_copy_complete_for_plan(
                        transfer_slots[plan.destination_slot],
                        plan,
                        config.participant_id,
                        command_epoch))
                {
                    layer_reused = true;
                    current_batch_llep_reuse =
                        (plan.flags &
                         kDeviceMoERebalancePlanFlagCurrentBatchLLEP) != 0u;
                    break;
                }
            }
            if (!layer_reused)
                continue;

            auto &runtime = runtime_layers[prior_layer];
            const uint32_t execution_bank =
                runtime_execution_bank(&runtime);
            const bool current_batch_child =
                current_batch_llep_reuse &&
                runtime.overlay_placement_banks &&
                runtime.overlay_epoch_ticket;
            if (runtime.active_bank > 1u ||
                execution_bank > 1u ||
                runtime.expert_count != config.num_experts ||
                runtime.participant_id != config.participant_id ||
                runtime.participant_count != config.participant_count)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "cross-layer CUDA slot retirement found invalid runtime metadata");
            }
            if (current_batch_llep_reuse && !current_batch_child)
            {
                FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                    "cross-layer CUDA LLEP slot retirement lost its ExpertOverlay child authority");
            }

            const auto &active =
                runtime_placement_banks(&runtime)[execution_bank];
            const uint32_t inactive_bank = 1u - execution_bank;
            auto &next = runtime.banks[inactive_bank];
            next.epoch = current_batch_child
                             ? runtime_command_epoch(&runtime)
                             : runtime.active_epoch + 1u;
            next.expert_count = config.num_experts;
            next.transient_placement_observed =
                active.transient_placement_observed;

            uint32_t multi_resident = 0u;
            for (uint32_t expert = 0u;
                 expert < config.num_experts;
                 ++expert)
            {
                next.experts[expert] = active.experts[expert];
                next.overlay_route_participant[expert] =
                    active.overlay_route_participant[expert];
                next.local_compute_mask[expert] =
                    active.local_compute_mask[expert];
                next.replica_role[expert] =
                    active.replica_role[expert];
                uint32_t resident_mask =
                    active.resident_participant_mask[expert] &
                    valid_mask;
                if (rebalance_ready_arrival_reuses_runtime_slot(
                        plans,
                        plan_count,
                        transfer_slots,
                        transfer_slot_count,
                        config,
                        command_epoch,
                        prior_layer,
                        expert,
                        next.experts[expert],
                        static_cast<int>(target_layer)))
                {
                    rebalance_retire_reused_transfer_slot_replica(
                        next.experts[expert],
                        resident_mask,
                        local_participant_bit,
                        config.participant_id);
                    next.local_compute_mask[expert] = 0u;
                    next.replica_role[expert] =
                        static_cast<uint8_t>(
                            kDeviceMoEReplicaRoleNone);
                }
                next.resident_participant_mask[expert] =
                    resident_mask;
                if (runtime_resident_count(
                        resident_mask,
                        config.participant_count) > 1u)
                {
                    ++multi_resident;
                }
            }
            next.multi_resident_expert_count = multi_resident;
            __threadfence();
            runtime.active_bank = inactive_bank;
            runtime.active_epoch = next.epoch;
            if (current_batch_child)
            {
                runtime.current_batch_llep_transient_bank_active = 1u;
            }
            ++changed_layers;
        }
        return changed_layers;
    }


    /** @return Atomic acquire-style load of one overlay 64-bit word. */
    __device__ __forceinline__ uint64_t overlay_atomic_load64(
        const uint64_t *value)
    {
        return static_cast<uint64_t>(atomicAdd(
            reinterpret_cast<unsigned long long *>(
                const_cast<uint64_t *>(value)),
            0ull));
    }

    /** @return Atomic acquire-style load of one overlay 32-bit word. */
    __device__ __forceinline__ uint32_t overlay_atomic_load32(
        const uint32_t *value)
    {
        return atomicAdd(
            reinterpret_cast<unsigned int *>(
                const_cast<uint32_t *>(value)),
            0u);
    }

    /** @brief Atomic release store for one overlay 64-bit word. */
    __device__ __forceinline__ void overlay_atomic_store64(
        uint64_t *destination,
        uint64_t value)
    {
        atomicExch(
            reinterpret_cast<unsigned long long *>(destination),
            static_cast<unsigned long long>(value));
    }

    /** @brief Atomic release store for one overlay 32-bit word. */
    __device__ __forceinline__ void overlay_atomic_store32(
        uint32_t *destination,
        uint32_t value)
    {
        atomicExch(reinterpret_cast<unsigned int *>(destination), value);
    }

    /**
     * @brief Publish one complete semantic result after all provenance fields.
     */
    __device__ __forceinline__ void finish_overlay_publication_status(
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        llaminar2::DeviceMoEOverlayEpochOperation operation,
        llaminar2::DeviceMoEOverlayEpochStatusCode code,
        uint64_t epoch,
        uint64_t selector,
        uint32_t bank,
        llaminar2::DeviceMoEOverlayEpochBankState state)
    {
        status->epoch = epoch;
        status->selector = selector;
        status->operation = static_cast<uint32_t>(operation);
        status->bank = bank;
        status->observed_state = static_cast<uint32_t>(state);
        __threadfence();
        overlay_atomic_store32(&status->code, static_cast<uint32_t>(code));
    }


    __device__ __forceinline__ void try_apply_ready_rebalance_wave_for_layer_thread0(
        DeviceMoELayerRuntimeView *runtime_layers,
        const DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_headers,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalanceApplyStatusView *status,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        int target_layer,
        uint32_t command_buffer_count)
    {
        if (threadIdx.x != 0 || !runtime_layers ||
            !plan_entries || !command_headers || !controller_state ||
            !rebalance_config_ok(config))
        {
            return;
        }

        if ((config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u &&
            rebalance_graph_controller_state_ok(controller_state, config))
            ++controller_state->decode_apply_polls;
        const DeviceMoEReadyWaveSelection ready = rebalance_find_ready_wave(
            controller_state,
            command_headers,
            config,
            target_layer,
            command_buffer_count);
        if (!ready.wave)
            return;

        DeviceMoERebalanceWaveProgressView *wave = ready.wave;
        const uint32_t wave_index = ready.wave_index;
        auto *command_header = command_headers + wave_index;
        if (!rebalance_command_header_ok(command_header, config))
            return;

        const auto *wave_plan_entries =
            plan_entries + static_cast<unsigned long long>(wave_index) * plan_capacity;
        const uint32_t command_count =
            rebalance_command_count(nullptr, plan_capacity, command_header);
        if (command_count == 0u)
            return;

        if (status)
            init_rebalance_apply_status_in_progress_device(status);
        wave->state = kDeviceMoERebalanceLifecycleApplying;
        ++controller_state->decode_apply_hits;

        uint32_t layer = target_layer >= 0
                             ? static_cast<uint32_t>(target_layer)
                             : config.num_layers;
        if (target_layer < 0)
        {
            const uint32_t layer_offset =
                min(wave->applied_layer_count, wave->planned_layer_count);
            if (layer_offset < wave->planned_layer_count && config.num_layers > 0u)
            {
                layer =
                    (wave->planned_start_layer + layer_offset) %
                    config.num_layers;
            }
            if (layer >= config.num_layers)
            {
                clear_rebalance_command_header_device(command_header);
                wave->state = kDeviceMoERebalanceLifecycleApplied;
                controller_state->active_wave =
                    (controller_state->active_wave + 1u) %
                    rebalance_command_buffer_count(command_buffer_count);
                __threadfence();
                return;
            }
        }
        const uint32_t participant_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        auto &runtime = runtime_layers[layer];
        const uint32_t execution_bank =
            runtime_execution_bank(&runtime);
        const bool current_batch_child =
            runtime.overlay_placement_banks &&
            runtime.overlay_epoch_ticket;
        const bool incomplete_child_authority =
            (runtime.overlay_placement_banks != nullptr) !=
            (runtime.overlay_epoch_ticket != nullptr);
        const bool durable_overlay_requires_atomic_apply =
            runtime.overlay_epoch_ticket &&
            !runtime.overlay_placement_banks;
        if (runtime.active_bank > 1u ||
            execution_bank > 1u ||
            incomplete_child_authority ||
            durable_overlay_requires_atomic_apply ||
            (current_batch_child &&
             command_header->epoch != runtime_command_epoch(&runtime)) ||
            runtime.expert_count != config.num_experts ||
            runtime.top_k != config.top_k ||
            runtime.participant_id != config.participant_id ||
            runtime.participant_count != config.participant_count)
        {
            if (status)
                status->status_code = 2u;
            poison_rebalance_graph_controller(
                controller_state,
                2u,
                wave_index,
                wave->epoch,
                0u,
                0u,
                status ? status->status_code : 2u,
                config.participant_id);
            return;
        }

        const auto &active =
            runtime_placement_banks(&runtime)[execution_bank];
        uint32_t valid_plan_count = 0u;
        uint32_t ready_plan_count = 0u;
        bool blocked_on_arrival = false;
        bool permanent_plan_error = false;

        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = wave_plan_entries[i];
            if (status)
                ++status->plan_entries_seen;
            if (plan.op == 0u || plan.layer != layer)
                continue;
            const bool current_batch_plan =
                (plan.flags &
                 kDeviceMoERebalancePlanFlagCurrentBatchLLEP) != 0u;
            if (current_batch_plan != current_batch_child ||
                !rebalance_plan_applies_runtime(plan.op) ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant >= config.participant_count ||
                (plan.op == kDeviceMoERebalancePlanOwnershipTransfer &&
                 runtime.overlay_epoch_ticket &&
                 plan.destination_overlay_participant < 0) ||
                rebalance_plan_duplicates_prior_local_arrival(
                    wave_plan_entries,
                    i,
                    plan,
                    config,
                    static_cast<int>(layer)))
            {
                if (status)
                    ++status->invalid_plan_entries;
                permanent_plan_error = true;
                continue;
            }
            const bool destination_local =
                plan.destination_participant == config.participant_id;
            ++valid_plan_count;

            const uint32_t destination_bit =
                runtime_participant_bit(static_cast<int>(plan.destination_participant));
            uint32_t resident_mask =
                (active.resident_participant_mask[plan.expert] |
                 plan.source_resident_mask |
                 destination_bit) &
                valid_mask;
            if (plan.op == kDeviceMoERebalancePlanOwnershipTransfer)
                resident_mask = destination_bit & valid_mask;

            if (rebalance_plan_requires_payload(plan.op) &&
                destination_local)
            {
                if (status)
                    ++status->required_local_arrivals;
                if (!local_transfer_slots ||
                    plan.destination_slot >= local_transfer_slot_count)
                {
                    if (status)
                        ++status->missing_destination_slots;
                    blocked_on_arrival = true;
                    permanent_plan_error = true;
                    continue;
                }
                const auto &slot = local_transfer_slots[plan.destination_slot];
                if (!rebalance_transfer_slot_copy_complete_for_plan(
                        slot,
                        plan,
                        config.participant_id,
                        command_header->epoch))
                {
                    if (status)
                        ++status->copy_incomplete;
                    blocked_on_arrival = true;
                    continue;
                }
                if (status)
                    ++status->ready_local_arrivals;
            }
            else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                     destination_local &&
                     (resident_mask & participant_bit) == 0u)
            {
                if (status)
                    ++status->missing_source_descriptors;
                blocked_on_arrival = true;
                permanent_plan_error = true;
                continue;
            }

            ++ready_plan_count;
        }

        if (valid_plan_count == 0u)
        {
            wave->applied_layer_count =
                min(wave->planned_layer_count, wave->applied_layer_count + 1u);
            wave->state =
                wave->applied_layer_count >= wave->planned_layer_count
                    ? kDeviceMoERebalanceLifecycleApplied
                    : kDeviceMoERebalanceLifecycleReadyToApply;
            if (wave->state == kDeviceMoERebalanceLifecycleApplied)
            {
                clear_rebalance_command_header_device(command_header);
                controller_state->active_wave =
                    (controller_state->active_wave + 1u) %
                    rebalance_command_buffer_count(command_buffer_count);
            }
            __threadfence();
            return;
        }
        if (permanent_plan_error)
        {
            if (status)
            {
                status->status_code =
                    status->invalid_plan_entries != 0u
                        ? 3u
                        : 4u;
            }
            poison_rebalance_graph_controller(
                controller_state,
                status ? status->status_code : 3u,
                wave_index,
                wave->epoch,
                valid_plan_count,
                ready_plan_count,
                status ? status->status_code : 3u,
                config.participant_id);
            __threadfence();
            return;
        }
        if (blocked_on_arrival || ready_plan_count == 0u)
        {
            wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
            __threadfence();
            return;
        }

        const uint32_t retired_cross_layer_count =
            rebalance_retire_cross_layer_slot_occupants_thread0(
                runtime_layers,
                wave_plan_entries,
                command_count,
                local_transfer_slots,
                local_transfer_slot_count,
                config,
                command_header->epoch,
                layer);

        const uint32_t inactive_bank = 1u - execution_bank;
        auto &next = runtime.banks[inactive_bank];
        next.epoch = current_batch_child
                         ? runtime_command_epoch(&runtime)
                         : runtime.active_epoch + 1u;
        next.expert_count = config.num_experts;
        next.multi_resident_expert_count =
            active.multi_resident_expert_count;
        next.transient_placement_observed =
            active.transient_placement_observed;
        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
        {
            next.experts[expert] = active.experts[expert];
            next.overlay_route_participant[expert] =
                active.overlay_route_participant[expert];
            auto &base_desc = next.experts[expert];
            uint32_t resident_mask =
                active.resident_participant_mask[expert] & valid_mask;
            if (rebalance_ready_arrival_reuses_runtime_slot(
                    wave_plan_entries,
                    command_count,
                    local_transfer_slots,
                    local_transfer_slot_count,
                    config,
                    command_header->epoch,
                    layer,
                    expert,
                    base_desc,
                    static_cast<int>(layer)))
            {
                rebalance_retire_reused_transfer_slot_replica(
                    base_desc,
                    resident_mask,
                    participant_bit,
                    config.participant_id);
            }
            const bool owner_local =
                base_desc.owner_participant == static_cast<int32_t>(config.participant_id);
            const bool local_resident = (resident_mask & participant_bit) != 0u;
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
                        "authoritative CUDA graph-apply owner lost its local payload");
                }
                rebalance_retire_local_payload_publication(
                    base_desc,
                    resident_mask,
                    participant_bit);
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

        uint32_t applied_arrivals = 0u;
        for (uint32_t i = 0; i < command_count; ++i)
        {
            const auto &plan = wave_plan_entries[i];
            if (plan.op == 0u || plan.layer != layer)
                continue;
            if (!rebalance_plan_applies_runtime(plan.op) ||
                plan.expert >= config.num_experts ||
                plan.source_participant >= config.participant_count ||
                plan.destination_participant >= config.participant_count ||
                (plan.op == kDeviceMoERebalancePlanOwnershipTransfer &&
                 runtime.overlay_epoch_ticket &&
                 plan.destination_overlay_participant < 0))
            {
                continue;
            }

            const bool destination_local =
                plan.destination_participant == config.participant_id;
            const uint32_t destination_bit =
                runtime_participant_bit(static_cast<int>(plan.destination_participant));
            const bool ownership_transfer =
                plan.op == kDeviceMoERebalancePlanOwnershipTransfer;
            auto desc = active.experts[plan.expert];
            uint32_t resident_mask =
                (active.resident_participant_mask[plan.expert] |
                 plan.source_resident_mask |
                 destination_bit) &
                valid_mask;
            if (ownership_transfer)
                resident_mask = destination_bit & valid_mask;

            if (rebalance_plan_requires_payload(plan.op) &&
                destination_local)
            {
                if (!local_transfer_slots ||
                    plan.destination_slot >= local_transfer_slot_count)
                {
                    if (status)
                        ++status->missing_destination_slots;
                    continue;
                }
                const auto &slot = local_transfer_slots[plan.destination_slot];
                if (!rebalance_transfer_slot_copy_complete_for_plan(
                        slot,
                        plan,
                        config.participant_id,
                        command_header->epoch))
                {
                    if (status)
                        ++status->copy_incomplete;
                    continue;
                }
                desc = slot.descriptor;
                desc.owner_participant =
                    ownership_transfer
                        ? static_cast<int32_t>(plan.destination_participant)
                        : active.experts[plan.expert].owner_participant;
                resident_mask = ownership_transfer
                                    ? destination_bit & valid_mask
                                    : resident_mask | participant_bit;
            }
            else if (plan.op == kDeviceMoERebalancePlanResidentExpertAssignment &&
                     destination_local &&
                     (resident_mask & participant_bit) == 0u)
            {
                if (status)
                    ++status->missing_source_descriptors;
                continue;
            }

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
                ++applied_arrivals;
            }
            else if (ownership_transfer)
            {
                rebalance_retire_local_payload_publication(
                    desc,
                    resident_mask,
                    participant_bit);
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
            next.resident_participant_mask[plan.expert] = resident_mask & valid_mask;
            if (rebalance_plan_requires_payload(plan.op))
            {
                next.transient_placement_observed = 1u;
                if ((plan.flags &
                     kDeviceMoERebalancePlanFlagCurrentBatchLLEP) != 0u)
                {
                    runtime.current_batch_llep_movement_observed = 1u;
                }
            }
        }

        uint32_t multi_resident = 0u;
        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
        {
            const uint32_t mask = next.resident_participant_mask[expert] & valid_mask;
            if (runtime_resident_count(mask, config.participant_count) > 1u)
                ++multi_resident;
            if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
            {
                reset_runtime_histogram_expert(runtime, expert);
            }
        }
        next.multi_resident_expert_count = multi_resident;
        __threadfence();
        runtime.active_bank = inactive_bank;
        runtime.active_epoch = next.epoch;
        if (current_batch_child)
        {
            runtime.current_batch_llep_transient_bank_active = 1u;
        }
        if (status)
        {
            status->status_code = 0u;
            status->changed_layers =
                1u + retired_cross_layer_count;
            status->applied_arrivals = applied_arrivals;
            status->post_apply_multi_resident_experts = multi_resident;
        }
        wave->applied_arrivals += applied_arrivals;
        const uint32_t next_applied = wave->applied_layer_count + 1u;
        wave->applied_layer_count =
            next_applied < wave->planned_layer_count
                ? next_applied
                : wave->planned_layer_count;
        if (wave->applied_layer_count >= wave->planned_layer_count)
        {
            clear_rebalance_command_header_device(command_header);
            wave->state = kDeviceMoERebalanceLifecycleApplied;
            controller_state->active_wave =
                (controller_state->active_wave + 1u) %
                rebalance_command_buffer_count(command_buffer_count);
        }
        else
        {
            wave->state = kDeviceMoERebalanceLifecycleReadyToApply;
        }
        __threadfence();
    }

} // namespace
