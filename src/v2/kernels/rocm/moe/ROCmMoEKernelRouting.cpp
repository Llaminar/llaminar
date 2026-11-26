/**
 * @file ROCmMoEKernelRouting.cpp
 * @brief ROCmMoEKernel router logits, rebalance controller, token grouping, and descriptor tables.
 */
#include "ROCmMoEKernelInternal.h"

namespace llaminar2
{

    bool ROCmMoEKernel::publishDeviceRebalanceTransferComplete(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceApplyStatus *copy_status,
        const DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatus *gathered_copy_status,
        const DeviceMoERebalanceConfig &config,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::publishDeviceRebalanceTransferComplete] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status)
        {
            LOG_ERROR("[ROCmMoEKernel::publishDeviceRebalanceTransferComplete] controller state, command header, wave state, copy status, plan entries, and gathered copy status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "publishDeviceRebalanceTransferComplete");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "publishDeviceRebalanceTransferComplete"))
            return false;

        return hipMoE_publish_rebalance_transfer_complete(
            controller_state,
            command_header,
            wave_state,
            copy_status,
            plan_entries,
            plan_capacity,
            gathered_copy_status,
            &config,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::applyReadyDeviceRebalanceWave(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer,
        uint32_t command_buffer_count,
        const DeviceMoEOverlayEpochStatus *overlay_reservation_status)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::applyReadyDeviceRebalanceWave] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !status || !controller_state)
        {
            LOG_ERROR("[ROCmMoEKernel::applyReadyDeviceRebalanceWave] runtime, plan, status, controller state, and command ABI must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyReadyDeviceRebalanceWave");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "applyReadyDeviceRebalanceWave"))
            return false;

        return hipMoE_apply_ready_rebalance_wave(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            target_layer,
            command_buffer_count,
            overlay_reservation_status,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::finalizeMoEOverlayRebalancePublication(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer_count,
        std::uint32_t expert_count,
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *reservation_and_publication_status,
        const DeviceMoERebalanceApplyStatus *apply_status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        std::uint32_t command_buffer_count,
        const DeviceMoERebalancePlanEntry *plan_entries,
        std::uint32_t plan_capacity,
        std::uint64_t estimated_expert_bytes,
        DeviceMoERebalanceMovementJournalView journal)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finalizeMoEOverlayRebalancePublication");
        if (!runtime_layers || layer_count == 0u || expert_count == 0u ||
            !control || !candidate_epoch ||
            !reservation_and_publication_status || !apply_status || !stream ||
            !controller_state || !command_headers || command_buffer_count == 0u ||
            command_buffer_count > 2u || !plan_entries || plan_capacity == 0u ||
            !journal.state || !journal.waves || !journal.edges ||
            journal.wave_capacity == 0u || journal.edge_capacity == 0u)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::finalizeMoEOverlayRebalancePublication] "
                "runtime family, epoch state, apply status, and explicit stream are required");
            return false;
        }
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "finalizeMoEOverlayRebalancePublication"))
        {
            return false;
        }
        return hipMoE_finalize_overlay_rebalance_publication(
            runtime_layers,
            layer_count,
            expert_count,
            control,
            candidate_epoch,
            reservation_and_publication_status,
            apply_status,
            controller_state,
            command_headers,
            command_buffer_count,
            plan_entries,
            plan_capacity,
            estimated_expert_bytes,
            journal,
            device_ordinal_,
            stream);
    }

    void ROCmMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        if (!setMoEDevice(device_ordinal_, "zeroBuffer"))
            return;

        void *ptr = tensor->gpu_data_ptr();
        if (!ptr)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] tensor has no device allocation");
            return;
        }
        hipError_t err = hipMemsetAsync(ptr, 0, bytes, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] hipMemsetAsync failed: " << hipGetErrorString(err));
            return;
        }
        markDeviceWritten(
            tensor,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    void ROCmMoEKernel::gatherTokenBatchFromTensors(
        ITensor *hidden, ITensor *batch_buffer,
        const int *host_token_indices, int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatchFromTensors"))
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());

        if (!h || !b)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] null device pointer");
            return;
        }
        // Upload host token indices to device staging buffer
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }

        gatherTokenBatch(h, b, d_staging_indices_, num_tokens, d_model);
        markDeviceWritten(
            batch_buffer,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    bool ROCmMoEKernel::copyTokenRowFromTensor(
        ITensor *source, ITensor *row_buffer,
        int row_index, int row_width)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (row_index < 0 || row_width <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "copyTokenRowFromTensor"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(source, device, stream, "source", "copyTokenRowFromTensor") ||
            !requireOutputOnDevice(row_buffer, device, stream, "row_buffer", "copyTokenRowFromTensor"))
        {
            return false;
        }

        const auto *src = static_cast<const float *>(source->gpu_data_ptr());
        auto *dst = static_cast<float *>(row_buffer->gpu_data_ptr());
        if (!hipMoE_copy_token_row(src, dst, row_index, row_width, device_ordinal_, stream))
            return false;

        markDeviceWritten(row_buffer, device, stream);
        return true;
    }

    void ROCmMoEKernel::scatterAddWeightedFromTensors(
        ITensor *output, ITensor *expert_output,
        const int *host_token_indices, const float *host_weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeightedFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *e = static_cast<const float *>(expert_output->gpu_data_ptr());

        if (!o || !e)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] null device pointer");
            return;
        }
        // Upload host indices + weights to device staging
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }
        err = hipMemcpyAsync(
            d_staging_weights_, host_weights,
            num_tokens * sizeof(float), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D weights failed: "
                      << hipGetErrorString(err));
            return;
        }

        scatterAddWeighted(o, e, d_staging_indices_, d_staging_weights_,
                           num_tokens, d_model);
        markDeviceWritten(
            output,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    bool ROCmMoEKernel::writeTokenRowToTensor(
        ITensor *destination, ITensor *row_buffer,
        int row_index, int row_width)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (row_index < 0 || row_width <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "writeTokenRowToTensor"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireOutputOnDevice(destination, device, stream, "destination", "writeTokenRowToTensor") ||
            !requireTensorOnDevice(row_buffer, device, stream, "row_buffer", "writeTokenRowToTensor"))
        {
            return false;
        }

        auto *dst = static_cast<float *>(destination->gpu_data_ptr());
        const auto *src = static_cast<const float *>(row_buffer->gpu_data_ptr());
        if (!hipMoE_write_token_row(dst, src, row_index, row_width, device_ordinal_, stream))
            return false;

        markDeviceWritten(destination, device, stream);
        return true;
    }

    void ROCmMoEKernel::sharedExpertGateFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateFromTensors"))
            return;

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateFromTensors") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateFromTensors") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateFromTensors"))
            return;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());

        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensors] null device pointer");
            return;
        }

        sharedExpertGate(in, gi, so, seq_len, d_model);
        markDeviceWritten(shared_output, device, stream);
    }

    bool ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateFromTensorsEffectiveSeqLen"))
            return false;
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateFromTensorsEffectiveSeqLen"))
            return false;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (seq_len == 1)
        {
            // Keep the existing fused M1 reduction tree; its optional count
            // suppresses dot work and overwrites the shared row when empty.
            if (!hipMoE_shared_expert_gate_decode_fused(in, gi, so, d_model,
                    device_ordinal_, stream, device_effective_seq_len))
                return false;
            markDeviceWritten(shared_output, device, stream);
            return true;
        }

        if (!ensureSharedGateScratchCapacity(seq_len))
            return false;
        if (!hipMoE_shared_expert_gate_effective_seq_len(
                in, gi, so, d_shared_gate_scratch_,
                seq_len, d_model, device_effective_seq_len,
                device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        return true;
    }

    void ROCmMoEKernel::sharedExpertGateAddFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateAddFromTensors"))
            return;

        if (seq_len <= 0)
            return;

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual", "sharedExpertGateAddFromTensors") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output", "sharedExpertGateAddFromTensors"))
            return;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const float *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        float *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensors] null device pointer");
            return;
        }

        if (!hipMoE_shared_expert_gate_add(in, gi, so, rr, co,
                                           seq_len, d_model,
                                           device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensors] fused gate-add kernel launch failed");
            return;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
    }

    bool ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateAddFromTensorsEffectiveSeqLen"))
            return false;
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output", "sharedExpertGateAddFromTensorsEffectiveSeqLen"))
            return false;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const float *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        float *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!hipMoE_shared_expert_gate_add_effective_seq_len(
                in, gi, so, rr, co, seq_len, d_model,
                device_effective_seq_len, device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] fused gate-add kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        return true;
    }

    void ROCmMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        if (!setMoEDevice(device_ordinal_, "swiGLUFromTensors"))
            return;

        float *g = static_cast<float *>(gate->gpu_data_ptr());
        float *u = static_cast<float *>(up->gpu_data_ptr());

        if (!g || !u)
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLUFromTensors] null device pointer");
            return;
        }

        swiGLU(g, u, count);
        markDeviceWritten(
            gate,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    void ROCmMoEKernel::weightedAddFromTensors(
        ITensor *output, ITensor *input, float weight, int count)
    {
        if (!setMoEDevice(device_ordinal_, "weightedAddFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *in = static_cast<const float *>(input->gpu_data_ptr());

        if (!o || !in)
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAddFromTensors] null device pointer");
            return;
        }

        weightedAdd(o, in, weight, count);
        markDeviceWritten(
            output,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable(
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!down_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (intermediate % 32) != 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped down descriptor table"))
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertDownDescriptorTable"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (original_family && !original_family->matchesNativeDown(desc)) return -1;

            if (!desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }

            if (!validateGroupedDownDesc(desc, d_model, intermediate))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
            if (!original_family && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        uint8_t codebook_id = kROCmMoEMixedCodebookSentinel;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            const uint32_t bit = nativeQuantisedCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }

        // The IQ4 LUT family may contain source-tuple24 and ordinary4.
        // One template24 launch owns the family; each descriptor retains its
        // independent factor words and the original physical mask is unchanged.
        const int iq4_family_codebook = singleCodebookFromMask(codebook_mask);
        if (iq4_family_codebook >= 0)
            codebook_id = static_cast<uint8_t>(iq4_family_codebook);

        // Publish immutable IQ tables before accepting or reusing a retained
        // descriptor table. A cache hit must never bypass device prerequisites.
        if (groupedPrefillMaskNeedsIQTables(codebook_mask) &&
            !rocm::ensureIQGridTablesInitialized(device_ordinal_))
        {
            LOG_ERROR(
                "[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] "
                "IQ grid table initialization failed on ROCm device "
                << device_ordinal_);
            return -1;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_descs.data(), down_descs, desc_bytes) == 0)
                return static_cast<int>(index);
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    /** @copydoc IMoEKernel::prepareOwnerGateUpCaptureBinding */
    std::uint64_t ROCmMoEKernel::prepareOwnerGateUpCaptureBinding(int table_id, int rows,
                                                                    std::uint64_t expected_intent)
    {
        if (isGraphCaptureActive()) throw std::logic_error("owner gate/up capability must be bound before native capture");
        if (table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) || rows <= 0)
            throw std::invalid_argument("owner gate/up preparation requires its original table and physical rows");
        auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.deviceReady()) throw std::logic_error("owner gate/up table is not prepared");
        const bool selected = rows == 1 && table.weight_format == DeviceMoEWeightFormat::NativeQuantised &&
            (table.descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable);
        if (!selected)
        {
            if (expected_intent) throw std::logic_error("owner gate/up intent names an unsupported physical operation");
            return 0;
        }
        if (!expected_intent)
        {
            if (table.owner_m1_admission || table.owner_m1_intent)
                throw std::logic_error("retained owner capability cannot be replaced by an unbound component policy");
            return 0; // Original explicitly unbound component policy; model declarations validate their own nonzero intent.
        }
        const auto plan = rocm::planMoERouteOwnedGateUp(rocm::MoERouteOwnedGateUpGeometry::TN64,
            {table.codebook_mask, table.policy_codebook_mask}, device_ordinal_,
            rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes,
            {.output_columns = table.intermediate, .reduction_columns = table.d_model});
        if (plan.identity() != expected_intent)
            throw std::logic_error("owner gate/up table does not match its predeclared immutable intent");
        if (table.owner_m1_admission)
        {
            if (table.owner_m1_intent != expected_intent)
                throw std::logic_error("owner gate/up capability cannot change behind a retained graph");
            return table.owner_m1_intent;
        }
        auto admission = rocm::admitMoERouteOwnedGateUpGeometry(plan);
        // Publish only the complete native capability. The original table and
        // graph retirement own this host metadata; no memory reserve is added.
        table.owner_m1_admission = std::move(admission);
        table.owner_m1_intent = expected_intent;
        return expected_intent;
    }

    /** @copydoc IMoEKernel::ownerGateUpCaptureBindingSignature */
    std::uint64_t ROCmMoEKernel::ownerGateUpCaptureBindingSignature(int table_id, int rows) const
    {
        if (table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) || rows <= 0)
            throw std::logic_error("owner gate/up capture binding lost its descriptor table");
        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.deviceReady()) throw std::logic_error("owner gate/up capture binding lost its prepared storage");
        const bool selected = rows == 1 && table.weight_format == DeviceMoEWeightFormat::NativeQuantised &&
            (table.descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable);
        if (!selected) return 0;
        if (table.owner_m1_admission.has_value() != (table.owner_m1_intent != 0))
            throw std::logic_error("owner gate/up table lost its complete immutable capability");
        return table.owner_m1_intent;
    }

    /** @copydoc ROCmMoEKernel::ownerGateUpAdmissionForCall */
    const rocm::MoERouteOwnedGateUpAdmission *ROCmMoEKernel::ownerGateUpAdmissionForCall(
        int table_id, int rows, const MoEPrefillProjectionExecution &execution,
        rocm::MoERouteOwnedProjectionShape shape) const
    {
        if (!execution.executesGateUp() || execution.executesDown() || rows != 1 ||
            table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size())) return nullptr;
        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            table.descriptor_source != MoEDecodeDescriptorSource::RuntimePlacementTable) return nullptr;
        if (!table.owner_m1_admission && !table.owner_m1_intent)
            return nullptr; // Existing unbound component path, not a declared graph policy.
        if (!table.owner_m1_admission || !table.owner_m1_intent)
            throw std::logic_error("owner gate/up table lost its complete immutable capability");
        if (rocm::currentMoERouteOwnedGateUpGeometry())
            throw std::logic_error("prepared model owner capability cannot be overridden by a diagnostic scope");
        // A bound operation cannot silently execute under a different shape.
        if (shape.output_columns != table.intermediate || shape.reduction_columns != table.d_model)
            throw std::logic_error("owner M1 gate/up requires its exact cold model capability");
        return &*table.owner_m1_admission;
    }

    /** @copydoc ROCmMoEKernel::recordOwnerGateUpCaptureSubmissions */
    void ROCmMoEKernel::recordOwnerGateUpCaptureSubmissions(
        const rocm::MoERouteOwnedGateUpLaunchSet &completed, std::uint64_t intent) const
    {
        if (!PerfStatsCollector::isDomainEnabled("kernel")) return;
        for (std::size_t index = 0; index < completed.leaf_count; ++index)
        {
            const auto &leaf = completed.leaves[index];
            PerfStatsCollector::addCounter("kernel", "rocm_moe_owner_m1_partition_submissions", 1.0,
                "submission", DeviceId::rocm(device_ordinal_).toString(),
                {{"observation", "host_execute_or_capture"}, {"partition_execution", "four_worker_planes"},
                 {"intent", std::to_string(intent)}, {"execution_codebook", std::to_string(leaf.execution_codebook)},
                 {"n", std::to_string(leaf.n)}, {"k", std::to_string(leaf.k)},
                 {"worker_planes", std::to_string(leaf.partition_worker_planes)},
                 {"grid_x", std::to_string(leaf.grid_x)}, {"grid_y", std::to_string(leaf.grid_y)}, {"grid_z", std::to_string(leaf.grid_z)},
                 {"block_threads", std::to_string(leaf.block_x)},
                 {"dynamic_shared_bytes", std::to_string(leaf.resources.dynamic_shared_memory_bytes)},
                 {"native_function", std::to_string(reinterpret_cast<std::uintptr_t>(leaf.resources.function))}});
        }
    }

    int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!gate_descs || !up_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (d_model % 32) != 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped gate/up descriptor tables"))
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertGateUpDescriptorTables"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeGateUpEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete sparse descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (original_family && !original_family->matchesNativeGateUp(gate_desc,up_desc)) return -1;
            if (!gate_desc.valid() || !up_desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (!areNativeQuantisedGateUpCodebooksCompatible(gate_desc.codebook_id, up_desc.codebook_id))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Gate/up codebook mismatch for expert "
                          << expert_id << ": gate=" << static_cast<int>(gate_desc.codebook_id)
                          << " up=" << static_cast<int>(up_desc.codebook_id));
                return -1;
            }

            if (!validateGroupedGateUpDesc(gate_desc, d_model, intermediate) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid descriptor pair for expert "
                          << expert_id);
                return -1;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeQuantisedMatrixDesc &desc)
            {
                return addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!original_family && (!add_descriptor(gate_desc) || !add_descriptor(up_desc)))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        uint8_t codebook_id = kROCmMoEMixedCodebookSentinel;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            const uint32_t bit = nativeQuantisedCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }
        const int iq4_family_codebook = singleCodebookFromMask(codebook_mask);
        if (iq4_family_codebook >= 0)
            codebook_id = static_cast<uint8_t>(iq4_family_codebook);

        // Gate/up graphs use the same device-constant tables as down graphs.
        // Establish the prerequisite before a cached table can be returned.
        if (groupedPrefillMaskNeedsIQTables(codebook_mask) &&
            !rocm::ensureIQGridTablesInitialized(device_ordinal_))
        {
            LOG_ERROR(
                "[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] "
                "IQ grid table initialization failed on ROCm device "
                << device_ordinal_);
            return -1;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_gate_descs ||
                !existing.device_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_gate_descs.size() != static_cast<size_t>(num_experts) ||
                existing.host_up_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_gate_descs.data(), gate_descs, desc_bytes) == 0 &&
                std::memcmp(existing.host_up_descs.data(), up_descs, desc_bytes) == 0)
                return static_cast<int>(index);
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    int ROCmMoEKernel::uploadGroupedExpertFloatingDownDescriptorTable(
        const DeviceMoEFloatingMatrixDesc *down_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!down_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped floating down descriptor table") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingDownDescriptorTable"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &desc = down_descs[expert];
            if (isBlankGroupedFloatingDesc(desc))
                continue;
            if (original_family && !original_family->matchesFloatingDown(desc)) return -1;
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return -1;
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_descs.data(),
                    down_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_floating_descs.assign(
            down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped floating expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int ROCmMoEKernel::uploadGroupedExpertFloatingGateUpDescriptorTables(
        const DeviceMoEFloatingMatrixDesc *gate_descs,
        const DeviceMoEFloatingMatrixDesc *up_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!gate_descs || !up_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped floating gate/up descriptor tables") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingGateUpDescriptorTables"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool gate_blank = isBlankGroupedFloatingDesc(gate_descs[expert]);
            const bool up_blank = isBlankGroupedFloatingDesc(up_descs[expert]);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return -1;
            }
            if (original_family && !original_family->matchesFloatingGateUp(gate_descs[expert],up_descs[expert])) return -1;
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return -1;
            }
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_gate_descs ||
                !existing.device_floating_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_gate_descs.size() !=
                    static_cast<size_t>(num_experts) ||
                existing.host_floating_up_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_gate_descs.data(),
                    gate_descs,
                    desc_bytes) == 0 &&
                std::memcmp(
                    existing.host_floating_up_descs.data(),
                    up_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_floating_gate_descs.assign(
            gate_descs, gate_descs + num_experts);
        table.host_floating_up_descs.assign(
            up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped floating expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool ROCmMoEKernel::updateGroupedExpertDownDescriptorTable(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (intermediate % 32) != 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped down descriptor table"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertDownDescriptorTable"))
            return false;

        auto &table = grouped_down_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !table.device_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(table.descriptor_source);
        NativeQuantisedExecutionFormatEnvelope format_envelope = table.original_family.valid()
            ? table.original_family.nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (table.original_family.valid() && !table.original_family.matchesNativeDown(desc)) return false;
            if (!desc.valid() || !validateGroupedDownDesc(desc, d_model, intermediate))
                return false;
            if (!table.original_family.valid() && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return false;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return false;
        const int sole_codebook = singleCodebookFromMask(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kROCmMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_descs,
                        down_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "descriptor refresh failed: "
                                  << hipGetErrorString(err));
                        return false;
                    }
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_descs.assign(down_descs, down_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (d_model % 32) != 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped gate/up descriptor tables"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertGateUpDescriptorTables"))
            return false;

        auto &table = grouped_gateup_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !table.device_gate_descs ||
            !table.device_up_descs || !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(table.descriptor_source);
        NativeQuantisedExecutionFormatEnvelope format_envelope = table.original_family.valid()
            ? table.original_family.nativeGateUpEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return false;
            }
            if (table.original_family.valid() && !table.original_family.matchesNativeGateUp(gate_desc,up_desc)) return false;
            if (!gate_desc.valid() || !up_desc.valid() ||
                !areNativeQuantisedGateUpCodebooksCompatible(gate_desc.codebook_id, up_desc.codebook_id) ||
                !validateGroupedGateUpDesc(gate_desc, d_model, intermediate) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate))
            {
                return false;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeQuantisedMatrixDesc &desc)
            {
                return addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!table.original_family.valid() && (!add_descriptor(gate_desc) || !add_descriptor(up_desc)))
            {
                return false;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return false;
        const int sole_codebook = singleCodebookFromMask(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kROCmMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    gate_descs,
                    up_descs,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_gate_descs,
                        gate_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "gate descriptor refresh failed: "
                                  << hipGetErrorString(err));
                        return false;
                    }
                    err = hipMemcpyAsync(
                        table.device_up_descs,
                        up_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "up refresh failed after gate submission: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertFloatingDownDescriptorTable(
        int descriptor_table_id,
        const DeviceMoEFloatingMatrixDesc *down_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            descriptor_table_id < 0 ||
            descriptor_table_id >=
                static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped floating down descriptor table") ||
            !setMoEDevice(
                device_ordinal_,
                "updateGroupedExpertFloatingDownDescriptorTable"))
        {
            return false;
        }

        auto &table = grouped_down_desc_tables_[
            static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || table.weight_format != weight_format ||
            !table.device_floating_descs || !table.workspace_publication ||
            table.num_experts != num_experts || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &desc = down_descs[expert];
            if (isBlankGroupedFloatingDesc(desc))
                continue;
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return false;
            any_live = true;
        }
        if (!any_live)
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
            return false;
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedFloatingDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    weight_format,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_floating_descs,
                        down_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        return false;
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    return true;
                }))
        {
            return false;
        }
        table.host_floating_descs.assign(
            down_descs, down_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertFloatingGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceMoEFloatingMatrixDesc *gate_descs,
        const DeviceMoEFloatingMatrixDesc *up_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            descriptor_table_id < 0 ||
            descriptor_table_id >=
                static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped floating gate/up descriptor tables") ||
            !setMoEDevice(
                device_ordinal_,
                "updateGroupedExpertFloatingGateUpDescriptorTables"))
        {
            return false;
        }

        auto &table = grouped_gateup_desc_tables_[
            static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || table.weight_format != weight_format ||
            !table.device_floating_gate_descs ||
            !table.device_floating_up_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool gate_blank = isBlankGroupedFloatingDesc(gate_descs[expert]);
            const bool up_blank = isBlankGroupedFloatingDesc(up_descs[expert]);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return false;
            }
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return false;
            }
            any_live = true;
        }
        if (!any_live)
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
            return false;
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedFloatingDescriptorPublicationKey(
                    gate_descs,
                    up_descs,
                    weight_format,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_floating_gate_descs,
                        gate_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        return false;
                    err = hipMemcpyAsync(
                        table.device_floating_up_descs,
                        up_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    return true;
                }))
        {
            return false;
        }
        table.host_floating_gate_descs.assign(
            gate_descs, gate_descs + num_experts);
        table.host_floating_up_descs.assign(
            up_descs, up_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromTable(
        const TensorBase *input,
        const int *expert_ids,
        int descriptor_table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !expert_ids || descriptor_table_id < 0 || num_active <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << descriptor_table_id
                      << " valid=" << table.valid
                      << " device_ready=" << table.deviceReady()
                      << " format="
                      << static_cast<std::uint32_t>(table.weight_format)
                      << " table_d_model=" << table.d_model
                      << " requested_d_model=" << d_model
                      << " table_intermediate=" << table.intermediate
                      << " requested_intermediate=" << intermediate);
            return false;
        }
        if (!floating && (d_model % 32) != 0)
            return false;

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        // Floating weights have no Q8/K-part intermediate dependency, including
        // their geometry constraints. Metadata/pointer publication is shared.
        if (!floating && !ensureGroupedGateUpCapacity(
                num_active, d_model, intermediate))
            return false;

        const int *fixed_device_expert_ids = nullptr;
        if (!resolveFixedTableGateUpMetadata(
                table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                num_active,
                &fixed_device_expert_ids))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromTable"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        if (!d_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Input tensor has no device pointer");
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_output_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_output_ptrs = {};
        for (int i = 0; i < num_active; ++i)
        {
            if (!gate_outputs[i] || !up_outputs[i])
                return false;
            gate_output_ptrs[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            up_output_ptrs[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!gate_output_ptrs[i] || !up_output_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Null gate/up output pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float **d_gate_output_ptrs = nullptr;
        float **d_up_output_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                num_active, gate_output_ptrs, up_output_ptrs,
                &d_gate_output_ptrs, &d_up_output_ptrs))
        {
            return false;
        }

        /* Table decode shares stable metadata/pointer publications between
         * packed and contiguous formats; dispatch only selects the arithmetic
         * kernel appropriate for the descriptor family. */
        const bool reuse_required_hidden=router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source();
        int8_t* original_hidden_codes=d_grouped_hidden_int8_;
        float* original_hidden_scales=d_grouped_hidden_scales_;
        if(reuse_required_hidden) {
            const auto& publication=*router_q8_hidden_publication_;
            if(floating)throw std::logic_error("Required M1 hidden Q8 reader has incompatible floating projection arithmetic");
            if(router_q8_publication_access_!=MoERouterQ8PublicationAccess::RequiredConsumer) {
                if(!canReuseRouterQ8Hidden(d_hidden,1,d_model))throw std::logic_error("Required M1 routed hidden Q8 reader is absent");
                original_hidden_codes=publication.original_m1_binding->codes();
                original_hidden_scales=publication.original_m1_binding->scales();
            } else {
            if(!original_m1_hidden_q8_read_||publication.backend!=DeviceType::ROCm||publication.device_ordinal!=device_ordinal_||
               publication.d_model_capacity<d_model||publication.blocks_per_row_capacity<d_model/32)
                throw std::logic_error("Required M1 shared hidden Q8 reader was not cold prepared on its exact participant/width");
            original_m1_hidden_q8_read_->validate(*input,getStream(),workspace_);
            original_m1_hidden_q8_read_->requirePayload(publication.source_rows,publication.quantized_rows,publication.row_scales,
                publication.published_rows,
                publication.capture_recorded?OriginalM1HiddenQ8Use::CaptureRecording:OriginalM1HiddenQ8Use::OrderedEager,
                isGraphCaptureActive()?OriginalM1HiddenQ8Use::CaptureRecording:OriginalM1HiddenQ8Use::OrderedEager);
            // The original low-level ABI combines writable independent scratch
            // with prequantized input. Only this sealed read view selects true;
            // that arm never writes codes/scales. No caller supplies raw reuse.
            original_hidden_codes=const_cast<int8_t*>(original_m1_hidden_q8_read_->codes());
            original_hidden_scales=const_cast<float*>(original_m1_hidden_q8_read_->scales());
            }
        }

        const bool ok = floating
            ? rocmMoE_grouped_gate_up_floating_decode_table(
                  d_hidden,
                  table.device_floating_gate_descs,
                  table.device_floating_up_descs,
                  fixed_device_expert_ids,
                  d_gate_output_ptrs,
                  d_up_output_ptrs,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.weight_format,
                  device_ordinal_,
                  getStream())
            : rocmMoE_grouped_gate_up_native_quantised_decode_table(
                  d_hidden,
                  table.device_gate_descs,
                  table.device_up_descs,
                  fixed_device_expert_ids,
                  d_gate_output_ptrs,
                  d_up_output_ptrs,
                  nullptr,
                  nullptr,
                  original_hidden_codes,
                  original_hidden_scales,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  reuse_required_hidden,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.codebook_id,
                  table.policy_codebook_mask,
                  device_ordinal_,
                  getStream());

        if (ok)
        {
            for (int i = 0; i < num_active; ++i)
            {
                markDeviceWritten(
                    gate_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
                markDeviceWritten(
                    up_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !routing_indices || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRouting"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden &&
            rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromRouting"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices &&
            !requireTensorOnDevice(routing_indices, device, getStream(),
                                  "routing_indices", "groupedExpertGateUpDecodeFromRouting"))
        {
            return false;
        }
        d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_hidden || !d_routing_indices)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Missing input/routing device pointer");
            return false;
        }

        host_grouped_gate_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        host_grouped_up_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        for (int i = 0; i < top_k; ++i)
        {
            if (!requireOutputOnDevice(gate_outputs[i], device, getStream(),
                                      "gate_output", "groupedExpertGateUpDecodeFromRouting") ||
                !requireOutputOnDevice(up_outputs[i], device, getStream(),
                                      "up_output", "groupedExpertGateUpDecodeFromRouting"))
            {
                return false;
            }
            host_grouped_gate_output_ptrs_[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            host_grouped_up_output_ptrs_[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!host_grouped_gate_output_ptrs_[i] || !host_grouped_up_output_ptrs_[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Null gate/up output pointer for slot "
                          << i);
                return false;
            }
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_output_ptrs_, host_grouped_gate_output_ptrs_.data(),
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_output_ptrs_, host_grouped_up_output_ptrs_.data(),
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        /*
         * The ROCm MTP verifier deliberately replays decode-equivalent rows from
         * device routing tensors. In LocalTP routed-expert overlays the route
         * tensor names global experts, while this participant should compute only
         * its mask-active shard before the MoE output allreduce.  The masked
         * conversion writes -1 for nonlocal top-k slots into backend scratch; the
         * grouped table kernels already skip those inactive slots.
         */
        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!hipMoE_float_to_masked_int(
                    d_routing_indices,
                    d_grouped_gateup_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    getStream()))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     d_routing_indices, d_grouped_gateup_expert_ids_, top_k,
                     device_ordinal_, getStream()))
        {
            return false;
        }

        /*
         * Gate/up owns one increasing-K accumulator per route and output column.
         * That arithmetic is the decode contract for both explicit routing and
         * runtime placement. Split-K variants are deliberately absent because
         * reducing partition totals changes the FP32 addition tree.
         */
        const bool ok = rocmMoE_grouped_gate_up_native_quantised_decode_table(
            d_hidden,
            table.device_gate_descs,
            table.device_up_descs,
            d_grouped_gateup_expert_ids_,
            d_grouped_gate_output_ptrs_,
            d_grouped_up_output_ptrs_,
            nullptr,
            nullptr,
            d_grouped_hidden_int8_,
            d_grouped_hidden_scales_,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            false,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                markDeviceWritten(gate_outputs[i], device, getStream());
                markDeviceWritten(up_outputs[i], device, getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !input || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden &&
            rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromRuntime"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Missing input/runtime device pointer");
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_output_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_output_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_output_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_output_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_output_ptrs[slot] || !up_output_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Null gate/up output pointer for slot "
                          << slot);
                return false;
            }
        }

        float **d_gate_output_ptrs = nullptr;
        float **d_up_output_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                top_k, gate_output_ptrs, up_output_ptrs,
                &d_gate_output_ptrs, &d_up_output_ptrs))
        {
            return false;
        }

        const bool reuse_router_q8_hidden =
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        int8_t *gateup_hidden_int8 = reuse_router_q8_hidden ? d_router_q8_hidden_ : d_grouped_hidden_int8_;
        float *gateup_hidden_scales = reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : d_grouped_hidden_scales_;

        const bool ok = rocmMoE_grouped_gate_up_native_quantised_decode_runtime(
            d_hidden,
            runtime_layer,
            d_expert_ids,
            d_gate_output_ptrs,
            d_up_output_ptrs,
            gateup_hidden_int8,
            gateup_hidden_scales,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            reuse_router_q8_hidden,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                markDeviceWritten(
                    gate_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
                markDeviceWritten(
                    up_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask,
        ITensor *canonical_route_contributions,
        DeviceMoELayerRuntime *runtime_layer,
        MoEDecodeDescriptorSource descriptor_source,
        MoEExplicitRouteDemand demand)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source) || !validMoEExplicitRouteDemand(demand)) return false;

        if (!input || !routing_indices || !routing_weights ||
            gateup_descriptor_table_id < 0 || down_descriptor_table_id < 0 ||
            top_k <= 0 || !output || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            gateup_descriptor_table_id >=
                static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table =
            grouped_gateup_desc_tables_[gateup_descriptor_table_id];
        if (static_cast<size_t>(down_descriptor_table_id) >= grouped_down_desc_tables_.size()) return false;
        const auto &down_table = grouped_down_desc_tables_[down_descriptor_table_id];
        if (!down_table.valid ||
            (gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        if (!gateup_table.valid || gateup_table.num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "gate/up descriptor table is unavailable");
            return false;
        }
        const bool use_runtime_descriptors =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if ((use_runtime_descriptors && (!runtime_layer || expert_mask)) ||
            (!use_runtime_descriptors && runtime_layer))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "runtime placement requires exactly one runtime layer and no host mask");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_, "groupedExpertDecodeFromRouting") ||
            !ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
        {
            return false;
        }

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "an explicit stream is required");
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const bool capture_active = isDecodeGraphCaptureActive();
        const float *device_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *device_routing_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if ((!device_routing_indices || !device_routing_weights) &&
            capture_active)
        {
            rejectDecodeStagingDuringCapture(
                "fused explicit-routing tensors");
            return false;
        }
        if (!device_routing_indices &&
            !requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "groupedExpertDecodeFromRouting"))
        {
            return false;
        }
        if (!device_routing_weights &&
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "groupedExpertDecodeFromRouting"))
        {
            return false;
        }

        device_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        device_routing_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!device_routing_indices || !device_routing_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "routing tensors have no device publication");
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(
                    expert_mask, gateup_table.num_experts) ||
                !hipMoE_float_to_masked_int(
                    device_routing_indices,
                    d_grouped_gateup_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    gateup_table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     device_routing_indices,
                     d_grouped_gateup_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        const int *resolved_expert_ids = d_grouped_gateup_expert_ids_;
        const float *resolved_weights = device_routing_weights;
        if (use_runtime_descriptors)
        {
            /* Keep route publication, placement filtering, and histogram
             * accounting in one exact captured follower stream. */
            if (!hipMoE_decode_route_select_runtime(
                    d_grouped_gateup_expert_ids_,
                    device_routing_weights,
                    runtime_layer,
                    /*legacy_indices=*/nullptr,
                    /*legacy_weights=*/nullptr,
                    gateup_table.num_experts,
                    top_k,
                    /*write_legacy_outputs=*/false,
                    /*update_runtime_histogram=*/demand == MoEExplicitRouteDemand::PublishDecode,
                    descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                        ? RoutedExpertRowExecutionPolicy::IntermediateShardLocal : RoutedExpertRowExecutionPolicy::ParticipantAssigned,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
            resolved_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
            resolved_weights = runtimeTopKWeightsDevice(runtime_layer);
        }

        return groupedExpertDecodeResolved(
            runtime_layer,
            input,
            gateup_descriptor_table_id,
            down_descriptor_table_id,
            top_k,
            output,
            d_model,
            intermediate,
            resolved_expert_ids,
            resolved_weights,
            descriptor_source,
            /*allow_router_q8_reuse=*/false,
            descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                ? "runtime_intermediate_shard" : use_runtime_descriptors ? "routing_runtime" : "routing",
            canonical_route_contributions);
    }

    const ROCmMoEKernel::PairedRouterDescriptors &ROCmMoEKernel::requirePairedRouterDescriptors(
        DeviceMoELayerRuntime *runtime, int num_experts, int top_k) const
    {
        if (!paired_router_descriptors_)
            throw std::logic_error("Missing original paired router descriptor preparation");
        const auto &binding = *paired_router_descriptors_;
        if (runtime != binding.runtime || workspace_ != binding.workspace || !workspace_ ||
            workspace_->id() != binding.workspace_id || getStream() != binding.stream ||
            num_experts != binding.outputs.expert_count || top_k != binding.top_k)
            throw std::logic_error("Paired router descriptors changed runtime, workspace, stream or geometry");
        return binding;
    }

    MoEDecodeDescriptorPublisher ROCmMoEKernel::preparePairedRouterDecodeDescriptors(
        DeviceMoELayerRuntime *runtime, int gateup_id, int down_id, int top_k)
    {
        if (isGraphCaptureActive() || !workspace_ || !runtime ||
            reinterpret_cast<std::uintptr_t>(getStream()) <= 2 ||
            gateup_id < 0 || down_id < 0 || top_k <= 0 || top_k > int(kDeviceMoEMaxTopK) ||
            gateup_id >= int(grouped_gateup_desc_tables_.size()) || down_id >= int(grouped_down_desc_tables_.size()))
            throw std::logic_error("Paired router descriptors require complete cold graph preparation");
        const auto &gate = grouped_gateup_desc_tables_[gateup_id];
        const auto &down = grouped_down_desc_tables_[down_id];
        if (!gate.valid || !gate.deviceReady() || !down.valid || !down.deviceReady() ||
            gate.descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
            down.descriptor_source != gate.descriptor_source || gate.weight_format != down.weight_format ||
            gate.num_experts != down.num_experts || gate.d_model != down.d_model ||
            gate.intermediate != down.intermediate || top_k > gate.num_experts)
            throw std::logic_error("Paired router descriptors require the same original fragment family");
        const bool ordinary=rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::OrdinaryPhases||
            rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::NativeLayoutRouter;
        const auto &policy = debugEnv().rocm;
        if(ordinary&&(!policy.moe_router_wave_topk||policy.moe_router_kpart_decode||
            gate.num_experts>int(kDeviceMoEMaxExperts)))
            throw std::invalid_argument("Ordinary FFN requires the original paired wave-topK descriptor publisher");
        if (paired_router_descriptors_)
        {
            const auto &original = requirePairedRouterDescriptors(runtime, gate.num_experts, top_k);
            if (original.gateup_table != gateup_id || original.down_table != down_id)
                throw std::logic_error("Paired router descriptors changed original table handles");
            return MoEDecodeDescriptorPublisher::Router;
        }
        // This is a cold performance-policy choice, not recovery after a failed
        // operation. The ordinary FFN candidate explicitly consumes this
        // publication on every HIP target; defaults retain the measured gfx906
        // selection and other architectures' independent two-node operation.
        if (!ordinary&&(!rocmHardwareMetadata().at(device_ordinal_).isa_target.starts_with("gfx906") ||
            !policy.moe_router_wave_topk || policy.moe_router_kpart_decode ||
            gate.num_experts > int(kDeviceMoEMaxExperts)))
            return MoEDecodeDescriptorPublisher::ExpertConsumer;
        MoEDecodeDescriptorPublication outputs{
            .expert_count = gate.num_experts,
            .format = gate.weight_format,
            .projections = DeviceMoEProjectionSet::IntermediateShard};
        if (!bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gate.workspace_slot, gate.num_experts, &outputs.gate, "paired router gate descriptors") ||
            !bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gate.workspace_slot, gate.num_experts, &outputs.up, "paired router up descriptors") ||
            !bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down.workspace_slot, down.num_experts, &outputs.down, "paired router down descriptors") || !outputs.valid())
            throw std::logic_error("Paired router descriptors lack the original admitted arena destinations");
        paired_router_descriptors_ = PairedRouterDescriptors{
            runtime, workspace_, workspace_->id(), getStream(), gateup_id, down_id, top_k, outputs};
        return MoEDecodeDescriptorPublisher::Router;
    }

    bool ROCmMoEKernel::prepareGroupedRuntimeDecodeLaunchState(
        int gateup_table_id,
        int down_table_id,
        int top_k,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] Runtime grouped-decode preparation "
                      "requires the exact non-null producer stream");
            return false;
        }
        if (isDecodeGraphCaptureActive())
        {
            LOG_ERROR("[ROCmMoEKernel] Runtime grouped-decode launch state must "
                      "be prepared before graph capture begins");
            return false;
        }
        if (gateup_table_id < 0 || down_table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || intermediate <= 0 ||
            gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model ||
            down_table.intermediate != intermediate ||
            !setMoEDevice(device_ordinal_,
                          "prepareGroupedRuntimeDecodeLaunchState") ||
            !ensureGroupedGateUpCapacity(top_k, d_model, intermediate) ||
            !ensureGroupedDecodeCapacity(top_k, intermediate, d_model) ||
            !ensureGroupedPrefillScratchCapacity(
                top_k, d_model, intermediate))
        {
            return false;
        }
        if (!floating &&
            ((d_model % 32) != 0 || (intermediate % 32) != 0))
        {
            return false;
        }

        if (moEDecodeUsesRuntimeDescriptors(descriptor_source))
        {
            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "ROCm runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "ROCm runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "ROCm runtime decode down descriptors"))
            {
                return false;
            }
        }

        const int max_dim = std::max(d_model, intermediate);
        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] =
                d_prefill_gate_ + static_cast<size_t>(slot) * max_dim;
            up_ptrs[slot] =
                d_prefill_up_ + static_cast<size_t>(slot) * intermediate;
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }

        float **device_gate_ptrs = nullptr;
        float **device_up_ptrs = nullptr;
        const float **device_down_gate_ptrs = nullptr;
        const float **device_down_up_ptrs = nullptr;
        return stageRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               stageRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool ROCmMoEKernel::prepareGroupedTableDecodeLaunchState(
        const int *expert_ids,
        const float *expert_weights,
        int gateup_table_id,
        int down_table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped-decode preparation "
                      "requires the exact non-null producer stream");
            return false;
        }
        if (isDecodeGraphCaptureActive())
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped-decode launch state "
                      "must be prepared before graph capture begins");
            return false;
        }
        if (!expert_ids || !expert_weights || !gate_outputs || !up_outputs ||
            !output || gateup_table_id < 0 || down_table_id < 0 ||
            num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || intermediate <= 0 ||
            gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        if (gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model ||
            down_table.intermediate != intermediate ||
            !setMoEDevice(device_ordinal_,
                          "prepareGroupedTableDecodeLaunchState"))
        {
            return false;
        }
        if (!floating &&
            ((d_model % 32) != 0 || (intermediate % 32) != 0))
        {
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 ||
                expert_ids[slot] >= gateup_table.num_experts ||
                expert_ids[slot] >= down_table.num_experts)
            {
                return false;
            }
        }
        const int *fixed_gateup_expert_ids = nullptr;
        const int *fixed_down_expert_ids = nullptr;
        const float *fixed_down_expert_weights = nullptr;
        if ((!floating &&
             (!ensureGroupedGateUpCapacity(num_active, d_model, intermediate) ||
              !ensureGroupedDecodeCapacity(num_active, intermediate, d_model))) ||
            !resolveFixedTableGateUpMetadata(
                gateup_table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                num_active,
                &fixed_gateup_expert_ids) ||
            !resolveFixedTableDownMetadata(
                down_table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                expert_weights,
                num_active,
                &fixed_down_expert_ids,
                &fixed_down_expert_weights))
        {
            return false;
        }
        // Both families publish immutable metadata and pointer arrays. Only
        // quantized arithmetic above binds Q8/K-part intermediate workspaces.
        (void)fixed_gateup_expert_ids;
        (void)fixed_down_expert_ids;
        (void)fixed_down_expert_weights;

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            gate_ptrs[slot] = gate_outputs[slot]
                                  ? static_cast<float *>(
                                        gate_outputs[slot]->gpu_data_ptr())
                                  : nullptr;
            up_ptrs[slot] = up_outputs[slot]
                                ? static_cast<float *>(
                                      up_outputs[slot]->gpu_data_ptr())
                                : nullptr;
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped decode has no "
                          "persistent gate/up scratch pointer for slot "
                          << slot);
                return false;
            }
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }
        if (!output->gpu_data_ptr())
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped decode output is not "
                      "bound to persistent device storage before capture");
            return false;
        }

        float **device_gate_ptrs = nullptr;
        float **device_up_ptrs = nullptr;
        const float **device_down_gate_ptrs = nullptr;
        const float **device_down_up_ptrs = nullptr;
        return stageRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               stageRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool ROCmMoEKernel::groupedExpertDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        ITensor *canonical_route_contributions, MoEExpertRouteWeighting weighting)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        if (!runtime_layer)
            return false;

        return groupedExpertDecodeResolved(
            runtime_layer,
            input,
            gateup_descriptor_table_id,
            down_descriptor_table_id,
            top_k,
            output,
            d_model,
            intermediate,
            runtimeTopKExpertIdsDevice(runtime_layer),
            runtimeTopKWeightsDevice(runtime_layer),
            descriptor_source,
            /*allow_router_q8_reuse=*/true,
            descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                ? "runtime_intermediate_shard" : moEDecodeUsesRuntimeDescriptors(descriptor_source)
                ? "runtime_compact_table"
                : "runtime_static_table",
            canonical_route_contributions, weighting);
    }

    bool ROCmMoEKernel::groupedExpertDecodeResolved(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const int *d_expert_ids,
        const float *d_weights,
        MoEDecodeDescriptorSource descriptor_source,
        bool allow_router_q8_reuse,
        const char *counter_source,
        ITensor *canonical_route_contributions, MoEExpertRouteWeighting weighting)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;
        const bool cohort_weights = weighting == MoEExpertRouteWeighting::CompletedCohort;
        if (weighting != MoEExpertRouteWeighting::LocalProjection &&
            (!cohort_weights || !canonical_route_contributions || output ||
             descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;


        const bool use_runtime_descriptors = moEDecodeUsesRuntimeDescriptors(descriptor_source);

        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || gateup_descriptor_table_id < 0 ||
            down_descriptor_table_id < 0 || top_k <= 0 || (!output && !canonical_route_contributions) ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (!d_expert_ids || !d_weights ||
            (use_runtime_descriptors && !runtime_layer))
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
        {
            return false;
        }
        if (gateup_descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_descriptor_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_descriptor_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model || gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model || down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        const hipStream_t stream = static_cast<hipStream_t>(getStream());
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRuntime"))
            return false;

        /*
         * This fused wrapper is intentionally workspace-native. The stage no
         * longer has to materialize one TensorBase per top-k slot, and captured
         * graphs replay stable pointer-table slots instead of host-owned scratch
         * tensors. If a graph would need to upload/allocate here, fail loudly.
         */
        if (!ensureGroupedGateUpCapacity(
                top_k, d_model, intermediate) ||
            !ensureGroupedDecodeCapacity(top_k, intermediate, d_model) ||
            !ensureGroupedPrefillScratchCapacity(top_k, d_model, intermediate))
        {
            return false;
        }

        /*
         * Runtime placement changes descriptor values while the graph-owned
         * workspace identity remains stable.  Materialize the active device
         * bank into compact projection tables before launching decode GEMVs.
         *
         * The former runtime kernels indexed the full DeviceMoELayerRuntime
         * object from every output lane.  That structure contains two complete
         * 256-expert banks plus routing and histogram state, so a single
         * transfer-backed expert forced every decode projection through a
         * cache-hostile descriptor walk.  LLEP consequently made otherwise
         * ordinary Qwen 3.6 layers tens of milliseconds slower.
         *
         * This publication is a device-to-device kernel on the exact producer
         * stream.  It is graph capturable, performs no allocation or host
         * transfer, and feeds the same compact table kernels used by immutable
         * placement.  Therefore mutable and immutable placement differ only in
         * where the table values are published, not in GEMV arithmetic.
         */
        const DeviceNativeQuantisedMatrixDesc *decode_gate_descs =
            gateup_table.device_gate_descs;
        const DeviceNativeQuantisedMatrixDesc *decode_up_descs =
            gateup_table.device_up_descs;
        const DeviceNativeQuantisedMatrixDesc *decode_down_descs =
            down_table.device_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_gate_descs =
            gateup_table.device_floating_gate_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_up_descs =
            gateup_table.device_floating_up_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_down_descs =
            down_table.device_floating_descs;
        if (use_runtime_descriptors)
        {
            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "ROCm runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "ROCm runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "ROCm runtime decode down descriptors"))
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] "
                          "failed to bind graph-owned runtime descriptor slots");
                return false;
            }
            // External route packets deliberately own their selected IDs and
            // descriptor publication, even when this kernel instance also
            // prepared an original-router graph. Only the original runtime
            // route entry can consume that router's paired publication.
            if (paired_router_descriptors_ && allow_router_q8_reuse)
            {
                const auto &binding = requirePairedRouterDescriptors(runtime_layer, gateup_table.num_experts, top_k);
                if (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
                    binding.gateup_table != gateup_descriptor_table_id || binding.down_table != down_descriptor_table_id ||
                    binding.outputs.gate != runtime_gate_descs || binding.outputs.up != runtime_up_descs ||
                    binding.outputs.down != runtime_down_descs || binding.outputs.format != gateup_table.weight_format ||
                    d_expert_ids != runtimeTopKExpertIdsDevice(runtime_layer) || d_weights != runtimeTopKWeightsDevice(runtime_layer))
                    throw std::logic_error("Paired descriptor consumer changed its original producer/table contract");
            }
            else
            {
                const bool materialized = hipMoE_materialize_runtime_selected_decode_descriptors(
                    runtime_layer, d_expert_ids, runtime_gate_descs, runtime_up_descs,
                    runtime_down_descs, gateup_table.num_experts, top_k,
                    gateup_table.weight_format, device_ordinal_, stream,
                    moEDecodeProjectionSet(descriptor_source));
                if (!materialized)
                {
                    LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] "
                              "failed to materialize compact runtime descriptor tables");
                    return false;
                }
            }
            decode_gate_descs = runtime_gate_descs;
            decode_up_descs = runtime_up_descs;
            decode_down_descs = runtime_down_descs;
            decode_floating_gate_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_gate_descs);
            decode_floating_up_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_up_descs);
            decode_floating_down_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_down_descs);
        }

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && rejectDecodeStagingDuringCapture("fused grouped decode input tensor"))
            return false;
        if (!d_hidden)
        {
            TransferEngine::requireDeviceInput(
                const_cast<TensorBase *>(input),
                device,
                stream);
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical route contribution tensor"
                                                  : "fused grouped decode output tensor";
        float *d_publication_output = static_cast<float *>(
            publication_output->gpu_data_ptr());
        if (!d_publication_output &&
            rejectDecodeStagingDuringCapture(publication_output_name))
            return false;
        if (!d_publication_output)
        {
            TransferEngine::requireDeviceOutput(
                publication_output,
                device,
                stream);
            d_publication_output = static_cast<float *>(
                publication_output->gpu_data_ptr());
        }
        const auto* down_completion=downCompletionForCall(down_descriptor_table_id,top_k);
        if(down_completion&&(!output||!canonical_route_contributions||floating))
            throw std::logic_error("Down completion requires both original canonical route and routed output planes");
        float *d_output = canonical_route_contributions
                              ? nullptr
                              : d_publication_output;
        float *d_canonical_route_contributions = canonical_route_contributions
                                                     ? d_publication_output
                                                     : nullptr;

        if (!d_hidden || !d_expert_ids || !d_weights ||
            !d_prefill_gate_ || !d_prefill_up_ ||
            (!d_output && !d_canonical_route_contributions))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        if(down_completion)
        {
            d_output=static_cast<float*>(output->gpu_data_ptr());
            if(!d_output||d_output==d_canonical_route_contributions)
                throw std::logic_error("Down completion lost its distinct materialized original routed output");
        }

        const int max_dim = std::max(d_model, intermediate);
        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = d_prefill_gate_ + static_cast<size_t>(slot) * max_dim;
            up_ptrs[slot] = d_prefill_up_ + static_cast<size_t>(slot) * intermediate;
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                gateup_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        /*
         * Gate/up follows the same complete increasing-K dot-product contract
         * as serial decode. Partition totals are not an interchangeable result:
         * adding those totals changes FP32 grouping and therefore token bytes.
         */
        const bool reuse_router_q8_hidden =
            !floating && allow_router_q8_reuse &&
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "rocm_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }
        int8_t *gateup_hidden_int8 = reuse_router_q8_hidden ? d_router_q8_hidden_ : d_grouped_hidden_int8_;
        float *gateup_hidden_scales = reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : d_grouped_hidden_scales_;
        /*
         * NativeQuantised decode has no consumer for the intermediate FP32 gate/up
         * rows.  Publish the exact decode-equivalent Q8 SwiGLU rows directly
         * from the ordered partial reducer.  Floating-point weights retain
         * their own fixed-tree FP32 contract and therefore do not enter this
         * NativeQuantised publication path.
         */
        const bool gateup_ok = floating
            ? rocmMoE_grouped_gate_up_floating_decode_table(
                  d_hidden,
                  decode_floating_gate_descs,
                  decode_floating_up_descs,
                  d_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  top_k,
                  intermediate,
                  d_model,
                  gateup_table.num_experts,
                  gateup_table.weight_format,
                  device_ordinal_,
                  stream)
            : rocmMoE_grouped_gate_up_native_quantised_decode_table(
                  d_hidden,
                  decode_gate_descs,
                  decode_up_descs,
                  d_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  gateup_hidden_int8,
                  gateup_hidden_scales,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  reuse_router_q8_hidden,
                  top_k,
                  intermediate,
                  d_model,
                  gateup_table.num_experts,
                  gateup_table.codebook_id,
                  gateup_table.policy_codebook_mask,
                  device_ordinal_,
                  stream,gateUpCompletionForCall(gateup_descriptor_table_id,top_k));
        if (!gateup_ok)
            return false;

        const float **d_down_gate_ptrs = nullptr;
        const float **d_down_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                down_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                top_k, const_gate_ptrs, const_up_ptrs,
                &d_down_gate_ptrs, &d_down_up_ptrs))
        {
            return false;
        }

        /*
         * The NativeQuantised branch consumes the Q8 rows published immediately
         * above on this exact stream.  No event or synchronization is needed
         * inside one stream; the boolean is a strict data-contract selector,
         * not permission to infer readiness from pointer non-nullness.
         */
        const bool down_ok = floating
            ? rocmMoE_grouped_swiglu_down_floating_decode_table(
                  d_down_gate_ptrs,
                  d_down_up_ptrs,
                  decode_floating_down_descs,
                  d_expert_ids,
                  cohort_weights ? nullptr : d_weights,
                  d_output,
                  d_canonical_route_contributions,
                  top_k,
                  d_model,
                  intermediate,
                  down_table.num_experts,
                  down_table.weight_format,
                  device_ordinal_,
                  stream)
            : rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
                  d_down_gate_ptrs,
                  d_down_up_ptrs,
                  decode_down_descs,
                  d_expert_ids,
                  cohort_weights ? nullptr : d_weights,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  true,
                  d_output,
                  d_canonical_route_contributions,
                  d_grouped_down_partials_,
                  top_k,
                  d_model,
                  intermediate,
                  down_table.num_experts,
                  down_table.codebook_id,
                  down_table.policy_codebook_mask,
                  device_ordinal_,
                  stream,down_completion);
        if (!down_ok)
            return false;

        markDeviceWritten(
            canonical_route_contributions
                ? canonical_route_contributions
                : output,
            device,
            stream);
        recordGroupedDecodeCounter(
            "rocm_moe_grouped_decode_fused_calls",
            counter_source,
            top_k, d_model, intermediate,
            floating ? "floating_fixed_tree"
                     : "ordered_route_parallel_down");
        return true;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromTable(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        int descriptor_table_id,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !output || descriptor_table_id < 0 || num_active <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << descriptor_table_id
                      << " valid=" << table.valid
                      << " device_ready=" << table.deviceReady()
                      << " format="
                      << static_cast<std::uint32_t>(table.weight_format)
                      << " table_d_model=" << table.d_model
                      << " requested_d_model=" << d_model
                      << " table_intermediate=" << table.intermediate
                      << " requested_intermediate=" << intermediate);
            return false;
        }
        if (!floating && (intermediate % 32) != 0)
            return false;

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        // Table-owned floating inputs bypass quantized activation workspaces.
        if (!floating && !ensureGroupedDecodeCapacity(num_active, intermediate, d_model))
            return false;

        const int *fixed_device_expert_ids = nullptr;
        const float *fixed_device_expert_weights = nullptr;
        if (!resolveFixedTableDownMetadata(
                table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                expert_weights,
                num_active,
                &fixed_device_expert_ids,
                &fixed_device_expert_weights))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int i = 0; i < num_active; ++i)
        {
            if (!gate_tensors[i] || !up_tensors[i])
                return false;
            gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!gate_ptrs[i] || !up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output && rejectDecodeStagingDuringCapture("grouped down output tensor"))
            return false;
        if (!d_output)
        {
            TransferEngine::requireDeviceOutput(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
            d_output = static_cast<float *>(output->gpu_data_ptr());
        }
        if (!d_output)
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                num_active, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
            return false;

        const bool ok = floating
            ? rocmMoE_grouped_swiglu_down_floating_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_floating_descs,
                  fixed_device_expert_ids,
                  fixed_device_expert_weights,
                  d_output,
                  nullptr,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.weight_format,
                  device_ordinal_,
                  getStream())
            : rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  fixed_device_expert_ids,
                  fixed_device_expert_weights,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  false,
                  d_output,
                  nullptr,
                  d_grouped_down_partials_,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.codebook_id,
                  table.policy_codebook_mask,
                  device_ordinal_,
                  getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRouting(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !routing_indices || !routing_weights ||
            !output || descriptor_table_id < 0 || top_k <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate, d_model))
            return false;

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < top_k; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Null gate/up device pointer for slot "
                          << i);
                return false;
            }
        }

        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices &&
            !requireTensorOnDevice(routing_indices, device, getStream(),
                                  "routing_indices", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_weights &&
            !requireTensorOnDevice(routing_weights, device, getStream(),
                                  "routing_weights", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output &&
            !requireOutputOnDevice(output, device, getStream(),
                                  "moe_output", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_routing_indices || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Missing routing/output device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!hipMoE_float_to_masked_int(
                    d_routing_indices,
                    d_grouped_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    getStream()))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     d_routing_indices, d_grouped_expert_ids_, top_k,
                     device_ordinal_, getStream()))
        {
            return false;
        }

        /*
         * Routes execute concurrently into unique rows. The canonical helper
         * then folds those rows in original top-k order, including zero rows for
         * non-local experts, so masked and unmasked decode share one reduction
         * tree without floating-point atomics.
         */
        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            table.device_descs,
            d_grouped_expert_ids_,
            d_weights,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            false,
            d_output,
            nullptr,
            d_grouped_down_partials_,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
            markDeviceWritten(output, device, getStream());
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRuntime(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        DeviceMoELayerRuntime *runtime_layer,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !runtime_layer || !output ||
            descriptor_table_id < 0 || top_k <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate, d_model))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Null gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Missing runtime/output device pointer");
            return false;
        }

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime(
            d_gate_ptrs,
            d_up_ptrs,
            runtime_layer,
            d_expert_ids,
            d_weights,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_output,
            d_grouped_down_partials_,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecode(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !down_descs || !output || num_active <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16 || (intermediate % 32) != 0)
        {
            return false;
        }

        uint8_t codebook_id = down_descs[0].codebook_id;
        for (int i = 0; i < num_active; ++i)
        {
            if (expert_ids[i] < 0 || !down_descs[i].valid() ||
                down_descs[i].n != d_model || down_descs[i].k != intermediate ||
                down_descs[i].blocks_per_row != static_cast<uint32_t>(intermediate / 32) ||
                !areNativeQuantisedGateUpCodebooksCompatible(
                    down_descs[i].codebook_id, codebook_id) ||
                (groupedDecodeRequiresMins(down_descs[i].codebook_id) && !down_descs[i].mins) ||
                (groupedDecodeRequiresEmins(down_descs[i].codebook_id) && !down_descs[i].emins))
            {
                return false;
            }
        }

        // Preserve the old IQ4_NL/IQ4_XS mixed-down domain: the LUT decode
        // is shared, but physical24 resolves only its own raw source factors.
        for (int i = 0; i < num_active; ++i)
            if (down_descs[i].codebook_id == kNativeQuantisedIQ4XSSourceScaleCodebook)
                codebook_id = kNativeQuantisedIQ4XSSourceScaleCodebook;

        if (!groupedDecodeSupportsCodebook(codebook_id))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Unsupported native-quantised codebook "
                      << static_cast<int>(codebook_id));
            return false;
        }

        if (!ensureGroupedDecodeCapacity(num_active, intermediate, d_model))
            return false;

        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < num_active; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Output tensor has no device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(num_active) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_decode_weights_, expert_weights,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_down_descs_, down_descs,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeQuantisedMatrixDesc),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            d_grouped_down_descs_,
            d_grouped_decode_weights_,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_grouped_down_partials_,
            d_output,
            num_active,
            d_model,
            intermediate,
            codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    // =========================================================================
    // Phase 4: GPU-side expert dispatch for prefill
    // =========================================================================

    bool ROCmMoEKernel::groupPrefillRoutes(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices, ITensor *routing_weights,
        int current_tokens, int max_tokens,
        int num_experts, int top_k,
        bool filter_to_local_runtime_experts,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !routing_indices || !routing_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] null runtime or routing tensor");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] invalid dimensions current_tokens=" << current_tokens
                                                                                               << " max_tokens=" << max_tokens
                                                                                               << " num_experts=" << num_experts
                                                                                               << " top_k=" << top_k);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "groupPrefillRoutes"))
            return false;

        const float *d_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_indices || !d_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] routing tensors have no device pointers");
            return false;
        }

        return hipMoE_group_prefill_routes_runtime(
            d_indices,
            d_weights,
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            filter_to_local_runtime_experts ? 1 : 0,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            getStream(),
            DeviceMoEProjectionSet::CompleteExpert);
    }

    bool ROCmMoEKernel::regroupPrefillRoutesForDiagnostics(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(getStream()));
        if (!runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 ||
            current_tokens > max_tokens || num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::regroupPrefillRoutesForDiagnostics] "
                "invalid route-only diagnostic contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "regroupPrefillRoutesForDiagnostics"))
        {
            return false;
        }

        return hipMoE_regroup_prefill_routes_runtime_assignments(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            getStream());
    }

    bool ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRouter(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        bool filter_to_local_runtime_experts,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(getStream()));
        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer || !routing_indices || !routing_weights ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRouter] "
                      "invalid complete-plan contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "publishCompleteGroupedPrefillPlanFromRouter"))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            return false;
        }

        const int max_slots = max_tokens * top_k;
        if (max_slots > group_slots_cap_ ||
            !d_group_active_expert_ids_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_active_expert_ids_),
                    MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                    static_cast<size_t>(num_experts) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRouter(active_ids)") ||
                !bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_original_to_grouped_),
                    MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                    static_cast<size_t>(max_slots) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRouter(inverse_map)"))
            {
                return false;
            }
            group_slots_cap_ = max_slots;
        }

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm complete runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm complete runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm complete runtime prefill down descriptors"))
        {
            return false;
        }

        const auto *device_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const auto *device_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        const int active_expert_slots = std::min(max_slots, num_experts);
        return device_indices && device_weights &&
               hipMoE_group_prefill_routes_and_materialize_plan_runtime(
                   device_indices,
                   device_weights,
                   static_cast<void *>(runtime_layer),
                   d_group_original_to_grouped_,
                   runtime_gate_descs,
                   runtime_up_descs,
                   runtime_down_descs,
                   d_group_active_expert_ids_,
                   current_tokens * top_k,
                   max_slots,
                   num_experts,
                   top_k,
                   active_expert_slots,
                   filter_to_local_runtime_experts ? 1 : 0,
                   static_cast<int>(demand),
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream(), expected_projections);
    }

    bool ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments] "
                      "invalid complete-plan contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "publishCompleteGroupedPrefillPlanFromRuntimeAssignments"))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            return false;
        }

        const int max_slots = max_tokens * top_k;
        if (max_slots > group_slots_cap_ ||
            !d_group_active_expert_ids_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_active_expert_ids_),
                    MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                    static_cast<size_t>(num_experts) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRuntimeAssignments(active_ids)") ||
                !bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_original_to_grouped_),
                    MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                    static_cast<size_t>(max_slots) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRuntimeAssignments(inverse_map)"))
            {
                return false;
            }
            group_slots_cap_ = max_slots;
        }

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm assigned runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm assigned runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm assigned runtime prefill down descriptors"))
        {
            return false;
        }

        const int active_expert_slots = std::min(max_slots, num_experts);

        return hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
                   static_cast<void *>(runtime_layer),
                   d_group_original_to_grouped_,
                   runtime_gate_descs,
                   runtime_up_descs,
                   runtime_down_descs,
                   d_group_active_expert_ids_,
                   current_tokens * top_k,
                   max_slots,
                   num_experts,
                   top_k,
                   active_expert_slots,
                   static_cast<int>(demand),
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream(), expected_projections);
    }

    bool ROCmMoEKernel::commitGroupedVerifierHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "commitGroupedVerifierHistograms");
        if (!stream)
            return false;
        if (!runtime_layer ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            total_rows != request_count * rows_per_request ||
            num_experts <= 0 ||
            top_k <= 0)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::commitGroupedVerifierHistograms] invalid "
                "device publication contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "commitGroupedVerifierHistograms"))
        {
            return false;
        }

        return hipMoE_commit_grouped_verifier_histograms(
            static_cast<void *>(runtime_layer),
            accepted_state_counts_device,
            publication_ok_flags_device,
            request_count,
            rows_per_request,
            total_rows,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens, int num_experts, int top_k,
        const int32_t *absolute_position_ids_device,
        const int32_t *active_row_count_device)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesLeastLoadedResident");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));

        if (!runtime_layer || !absolute_position_ids_device ||
            !active_row_count_device)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident] "
                "runtime, device position row, and device active-row count "
                "must be non-null");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesLeastLoadedResident"))
            return false;

        return hipMoE_assign_prefill_routes_least_loaded_resident(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            absolute_position_ids_device,
            active_row_count_device,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens, int num_experts, int top_k,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "planPrefillRoutesLeastLoadedCurrentBatch");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));

        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (config.expert_count != static_cast<uint32_t>(num_experts) ||
            config.participant_count == 0u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u ||
            config.lambda_numerator == 0u ||
            config.lambda_denominator == 0u)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid LLEP config");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "planPrefillRoutesLeastLoadedCurrentBatch"))
            return false;

        return hipMoE_plan_prefill_routes_least_loaded_current_batch(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            config.min_chunk_tokens,
            config.alpha_numerator,
            config.alpha_denominator,
            config.lambda_numerator,
            config.lambda_denominator,
            config.min_spread_improvement,
            config.min_spread_improvement_divisor,
            config.min_spread_improvement_per_critical_path_slot,
            config.min_foreign_rows_per_critical_path_slot,
            config.max_weight_transfers,
            config.max_non_owner_experts_per_participant,
            config.enable_balanced_skip ? 1 : 0,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers");
        if (!stream)
            return false;
        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers"))
            return false;

        return hipMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const DeviceMoERebalanceStatus *transfer_status,
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers");
        if (!stream)
            return false;
        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] null runtime");
            return false;
        }
        if (!transfer_status || !apply_status)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] transfer and apply status are required");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"))
            return false;
        if (!validateDevicePointerOrLog(
                runtime_layer,
                device_ordinal_,
                "runtime_layer",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers") ||
            !validateDevicePointerOrLog(
                transfer_status,
                device_ordinal_,
                "transfer_status",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers") ||
            !validateDevicePointerOrLog(
                apply_status,
                device_ordinal_,
                "apply_status",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"))
        {
            return false;
        }

        return hipMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            transfer_status,
            apply_status,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !batch_buffer)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                          << " max_tokens=" << max_tokens
                                                                                                          << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "gatherPrefillExpertBatchFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_batch = static_cast<float *>(batch_buffer->gpu_data_ptr());
        if (!d_hidden || !d_batch)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_gather_expert_runtime(
            static_cast<const void *>(runtime_layer),
            d_hidden,
            d_batch,
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
        {
            markDeviceWritten(
                batch_buffer,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime(
        ITensor *output, ITensor *expert_results,
        DeviceMoELayerRuntime *runtime_layer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (!output || !expert_results || !runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                             << " max_tokens=" << max_tokens
                                                                                                             << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "scatterPrefillExpertResultsFromRuntime"))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const float *d_expert_output = static_cast<const float *>(expert_results->gpu_data_ptr());
        if (!d_output || !d_expert_output)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_scatter_expert_runtime(
            d_output,
            d_expert_output,
            static_cast<const void *>(runtime_layer),
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    // =========================================================================
    // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
    // =========================================================================

    bool ROCmMoEKernel::prepareExpertGroupsAsync(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginAllRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0 || seq_len > std::numeric_limits<int>::max()/top_k)
            return false;

        if (!setMoEDevice(device_ordinal_, "prepareExpertGroupsAsync"))
            return false;

        const int total_slots = seq_len * top_k;

        // 1. Join both routing producers onto this planner's explicit stream.
        //
        // A transfer that omits the consumer stream may return before this
        // planner reads the route table. The stream-aware helper either uploads
        // on this stream or joins the tensor's existing producer event, making
        // that race structurally impossible.
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        if (!requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "prepareExpertGroupsAsync") ||
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "prepareExpertGroupsAsync"))
        {
            return false;
        }

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] null device pointers");
            return false;
        }

        // 2. Lazy-allocate grouping buffers
        if (total_slots > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroupsAsync(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroupsAsync(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        const bool use_small_grouping =
            total_slots <= 64 &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            top_k <= 16;
        if (const auto *original = ScopedMoERoutedTransaction::current())
        {
            auto actual = *original;
            actual.owner = this; actual.indices = routing_indices; actual.weights = routing_weights;
            actual.rows = seq_len; actual.experts = num_experts; actual.top_k = top_k;
            actual.stream = stream; actual.workspace = workspace_;
            ScopedMoERoutedTransaction::requireMatching(actual);
        }
        const auto directory_source = rocm::selectedMoEGroupedPrefillPolicy(
            device_ordinal_, originalRoutedPopulationKey()).directory;
        std::optional<MoEGroupingPackedDirectory> packed_directory;
        if (directory_source != rocm::MoEGroupedDirectorySource::RolePlanner)
        {
            if ((seq_len != 9 && seq_len != 16) || num_experts != 256 || top_k != 8)
                throw std::invalid_argument("grouping offset-slot candidate requires physical M9/16, E256, top8");
            if (!ensureGroupedPrefillWorkDirectoryCapacity(total_slots, num_experts)) return false;
            packed_directory.emplace(groupingDirectoryBinding(seq_len, top_k, num_experts, directory_source));
        }
        if (use_small_grouping)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            /*
             * Small verifier batches use a single graph-capturable planner
             * kernel that publishes every piece of route metadata consumed by
             * grouped execution.  d_group_int_indices_ is deliberately passed
             * as original route-slot expert ids; executeGroupedPrefillPipeline
             * later gives that same buffer to the parallel down publication
             * kernel when ROCm serial decode is using parallel down.
             */
            if (!hipMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_group_int_indices_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] small-M grouping failed");
                group_active_expert_slots_ = 0;
                return false;
            }
            group_active_expert_slots_ = active_expert_slots;

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "rocm_moe_small_prefill_grouping_calls",
                    1.0,
                    "moe",
                    DeviceId::rocm(device_ordinal_).to_string(),
                    PerfStatsCollector::Tags{
                        {"total_slots", std::to_string(total_slots)},
                        {"num_experts", std::to_string(num_experts)},
                        {"top_k", std::to_string(top_k)}});
            }

            std::move(grouping_publication).publish();
            return true;
        }

        // 3. Convert float indices → int on device
        group_active_expert_slots_ = 0;
        if (!hipMoE_float_to_int(d_float_indices, d_group_int_indices_,
                                 total_slots, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] float_to_int failed");
            return false;
        }

        // 4. Group tokens by expert (counts, offsets, scatter) — all on device
        if (!enqueueExpertGrouping(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_, packed_directory ? &*packed_directory : nullptr))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] groupTokensByExpertDevice failed");
            return false;
        }
        group_active_expert_slots_ = std::min(total_slots, num_experts);

        // Launch device-side max-reduction over expert counts (async, no sync).
        // Result in d_group_max_tokens_ — consumed by GEMM grid early-exit.
        if (!hipMoE_max_expert_count(d_group_counts_, d_group_max_tokens_,
                                     num_experts, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] max_expert_count failed");
            return false;
        }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            /*
             * Wider runtime-M verifier groups intentionally use the scalable
             * count/scan/scatter planner. Publishing this route separately
             * lets correctness gates prove that a missing compact-planner
             * counter is a policy transition, not an unobserved fallback.
             */
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_general_prefill_grouping_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"total_slots", std::to_string(total_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"top_k", std::to_string(top_k)}});
        }

        // NO D2H copy, NO hipStreamSynchronize — data stays on device
        // for consumption by executeGroupedPrefillPipeline()

        std::move(grouping_publication).publish(std::move(packed_directory));
        return true;
    }

    bool ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask(
        ITensor *routing_indices,
        ITensor *routing_weights,
        int seq_len,
        int num_experts,
        int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginSelectedRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(
                device_ordinal_,
                "prepareExpertGroupsAsyncUsingPublishedMask"))
            return false;

        const int total_slots = seq_len * top_k;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        if (!requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "prepareExpertGroupsAsyncUsingPublishedMask") ||
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "prepareExpertGroupsAsyncUsingPublishedMask"))
        {
            return false;
        }

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "null device pointers");
            return false;
        }

        if (total_slots > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        /*
         * The graph path consumes only the backend-owned mask. Missing
         * publication is a lifecycle error, not an invitation to stage a host
         * pointer while HIP is recording graph nodes.
         */
        if (!group_expert_mask_published_ ||
            !d_group_expert_mask_ ||
            group_expert_mask_cap_ < num_experts ||
            group_expert_mask_num_experts_ != num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "fixed-topology expert mask was not published for the current "
                      "workspace and expert count before graph execution"
                      << " requested_experts=" << num_experts
                      << " published_experts=" << group_expert_mask_num_experts_
                      << " capacity=" << group_expert_mask_cap_);
            return false;
        }

        group_active_expert_slots_ = 0;
        if (!hipMoE_float_to_masked_int(
                d_float_indices,
                d_group_int_indices_,
                d_group_expert_mask_,
                total_slots,
                num_experts,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "float_to_masked_int failed");
            return false;
        }

        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "groupTokensByExpertDevice failed");
            return false;
        }
        /*
         * Masked LocalTP grouping emits a compact active-id prefix followed by
         * -1 padding.  Bound the stable grouped grid by the number of experts
         * this participant can own instead of launching planes for every global
         * expert.  This mirrors CUDA and keeps masked publication economical.
         */
        group_active_expert_slots_ =
            std::min(total_slots, group_expert_mask_active_experts_);

        if (!hipMoE_max_expert_count(d_group_counts_, d_group_max_tokens_,
                                     num_experts, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "max_expert_count failed");
            return false;
        }

        PerfStatsCollector::addCounter(
            "kernel", "rocm_moe_masked_prefill_grouping_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"num_experts", std::to_string(num_experts)},
             {"mask_active_experts", std::to_string(group_expert_mask_active_experts_)},
             {"expert_grid_slots", std::to_string(group_active_expert_slots_)}});
        std::move(grouping_publication).publish();
        return true;
    }
} // namespace llaminar2
