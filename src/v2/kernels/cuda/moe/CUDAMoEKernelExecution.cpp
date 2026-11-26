/**
 * @file CUDAMoEKernelExecution.cpp
 * @brief CUDAMoEKernel grouped decode, grouped prefill, floating MoE, and retained HC/FFN execution.
 */
#include "CUDAMoEKernelInternal.h"

namespace llaminar2
{
    bool CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            (d_model % 32) != 0)
        {
            return false;
        }
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "update grouped gate/up descriptor tables"))
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
                !validateCudaGroupedDescShape(gate_desc, intermediate, d_model) ||
                !validateCudaGroupedDescShape(up_desc, intermediate, d_model))
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
        const int sole_codebook = singleCUDAMoECodebook(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kCudaMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(getStream());
        if (!workspace_->rewritePersistentPublication(
                kCudaGroupedGateUpDescriptorLeaseDomain,
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
                    cudaError_t err = cudaMemcpyAsync(
                        table.device_gate_descs,
                        gate_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "gate descriptor refresh failed: "
                                  << cudaGetErrorString(err));
                        return false;
                    }
                    err = cudaMemcpyAsync(
                        table.device_up_descs,
                        up_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "up refresh failed after gate submission: "
                                  << cudaGetErrorString(err));
                        std::terminate();
                    }
                    err = cudaEventRecord(
                        static_cast<cudaEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << cudaGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        return true;
    }

    bool CUDAMoEKernel::updateGroupedExpertFloatingDownDescriptorTable(
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
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "update grouped floating down descriptor table") ||
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

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        const cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        if (!stream)
            return false;
        if (!workspace_->rewritePersistentPublication(
                kCudaGroupedDownDescriptorLeaseDomain,
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
                    cudaError_t err = cudaMemcpyAsync(
                        table.device_floating_descs,
                        down_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                        return false;
                    err = cudaEventRecord(
                        static_cast<cudaEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != cudaSuccess)
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

    bool CUDAMoEKernel::updateGroupedExpertFloatingGateUpDescriptorTables(
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
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "update grouped floating gate/up descriptor tables") ||
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

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        const cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        if (!stream)
            return false;
        if (!workspace_->rewritePersistentPublication(
                kCudaGroupedGateUpDescriptorLeaseDomain,
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
                    cudaError_t err = cudaMemcpyAsync(
                        table.device_floating_gate_descs,
                        gate_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                        return false;
                    err = cudaMemcpyAsync(
                        table.device_floating_up_descs,
                        up_descs,
                        desc_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
                    if (err != cudaSuccess)
                        std::terminate();
                    err = cudaEventRecord(
                        static_cast<cudaEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != cudaSuccess)
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

    bool CUDAMoEKernel::prepareExpertGroupsAsync(ITensor *routing_indices, ITensor *routing_weights,
                                                 int seq_len, int num_experts, int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginAllRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareExpertGroupsAsync");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(total_slots, num_experts))
            return false;

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
            return false;

        group_active_expert_slots_ = 0;
        const bool verifier_sized_group =
            total_slots <= 64 &&
            seq_len <= 4 &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            top_k <= static_cast<int>(kDeviceMoEMaxTopK);
        if (verifier_sized_group)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            if (!cudaMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_group_original_expert_ids_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    stream))
            {
                return false;
            }

            group_active_expert_slots_ = active_expert_slots;
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_small_prefill_grouping_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"num_experts", std::to_string(num_experts)},
                 {"total_slots", std::to_string(total_slots)}});
            std::move(grouping_publication).publish();
            return true;
        }

        if (!cudaMoE_float_to_int(d_float_indices, d_group_int_indices_, total_slots, device_ordinal_, stream))
            return false;
        if (!groupTokensByExpertDevice(d_group_int_indices_, d_float_weights,
                                       seq_len, num_experts, top_k,
                                       d_group_offsets_, d_group_counts_,
                                       d_group_token_indices_, d_group_weights_))
            return false;

        group_active_expert_slots_ = std::min(total_slots, num_experts);
        /*
         * Runtime-M verifier batches above the compact planner's fixed work
         * envelope use the scalable count/scan/scatter pipeline.  Publish a
         * distinct counter so correctness tests can prove an extended-depth
         * row batch did not accidentally reuse the M<=4 planner or skip
         * grouping altogether.
         */
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_general_prefill_grouping_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"num_experts", std::to_string(num_experts)},
             {"total_slots", std::to_string(total_slots)}});
        std::move(grouping_publication).publish();
        return true;
    }

    bool CUDAMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask(
        ITensor *routing_indices,
        ITensor *routing_weights,
        int seq_len,
        int num_experts,
        int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginSelectedRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        void *stream = requireStream(
            "CUDAMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(total_slots, num_experts))
            return false;

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
            return false;

        /*
         * Mask publication is intentionally not recoverable from this method.
         * A missing or mismatched publication means the stage lifecycle is
         * invalid; copying a host mask here would make graph replay depend on
         * transient host memory and recreate the coherence bug this contract
         * exists to prevent.
         */
        if (!group_expert_mask_published_ ||
            !d_group_expert_mask_ ||
            group_expert_mask_cap_ < num_experts ||
            group_expert_mask_num_experts_ != num_experts)
        {
            LOG_ERROR("[CUDAMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "fixed-topology expert mask was not published for the current "
                      "workspace and expert count before graph execution"
                      << " requested_experts=" << num_experts
                      << " published_experts=" << group_expert_mask_num_experts_
                      << " capacity=" << group_expert_mask_cap_);
            return false;
        }

        group_active_expert_slots_ = 0;
        if (!cudaMoE_float_to_masked_int(
                d_float_indices,
                d_group_int_indices_,
                d_group_expert_mask_,
                total_slots,
                num_experts,
                device_ordinal_,
                stream))
        {
            return false;
        }

        if (!groupTokensByExpertDevice(d_group_int_indices_, d_float_weights,
                                       seq_len, num_experts, top_k,
                                       d_group_offsets_, d_group_counts_,
                                       d_group_token_indices_, d_group_weights_))
            return false;

        /*
         * The active-id list is compacted in expert order and padded with -1.
         * A LocalTP participant can never execute more distinct experts than
         * its static ownership mask contains, so use that tighter graph-stable
         * upper bound for the grouped expert grid.  The old num_experts bound
         * launched known-nonlocal no-op expert planes and made masked telemetry
         * indistinguishable from the unmasked route.
         */
        group_active_expert_slots_ =
            std::min(total_slots, group_expert_mask_active_experts_);
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_masked_prefill_grouping_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"num_experts", std::to_string(num_experts)},
             {"mask_active_experts", std::to_string(group_expert_mask_active_experts_)},
             {"expert_grid_slots", std::to_string(group_active_expert_slots_)}});
        std::move(grouping_publication).publish();
        return true;
    }

    bool CUDAMoEKernel::updateGroupedPrefillExpertMask(
        const uint8_t *expert_mask,
        int num_experts)
    {
        if (num_experts <= 0 || !expert_mask)
            return false;

        void *stream = requireStream("CUDAMoEKernel::updateGroupedPrefillExpertMask");
        if (!stream)
            return false;

        uint64_t mask_hash = 1469598103934665603ull;
        int active_experts = 0;
        for (int i = 0; i < num_experts; ++i)
        {
            mask_hash ^= static_cast<uint64_t>(expert_mask[i]);
            mask_hash *= 1099511628211ull;
            active_experts += expert_mask[i] != 0u ? 1 : 0;
        }
        mask_hash ^= static_cast<uint64_t>(num_experts);
        mask_hash *= 1099511628211ull;

        /*
         * Captured decode may consume an immutable mask published during
         * warmup.  Check that exact identity before applying the mutation
         * guard: reusing an already-published device table performs no host
         * transfer and is therefore a valid graph-construction operation.
         */
        if (group_expert_mask_published_ &&
            group_expert_mask_hash_ == mask_hash &&
            group_expert_mask_num_experts_ == num_experts)
            return true;

        if (rejectCudaPersistentMetadataMutationDuringCapture(
                stream, "update grouped prefill expert mask"))
        {
            return false;
        }

        if (!d_group_expert_mask_ || group_expert_mask_cap_ < num_experts)
        {
            if (!workspace_)
            {
                LOG_ERROR("[CUDAMoEKernel::updateGroupedPrefillExpertMask] "
                          "graph-owned workspace is required before mask publication");
                return false;
            }

            auto lease = workspace_->acquirePersistentSlot(
                kCudaGroupedExpertMaskLeaseDomain,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            void *mask_table = nullptr;
            const size_t required_table_bytes =
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) *
                static_cast<size_t>(num_experts) * sizeof(uint8_t);
            if (!lease ||
                !bindWorkspaceBuffer(
                    &mask_table,
                    MoEWorkspaceBuffers::GROUP_EXPERT_MASK,
                    required_table_bytes,
                    "updateGroupedPrefillExpertMask(group_expert_mask_table)"))
            {
                d_group_expert_mask_ = nullptr;
                group_expert_mask_workspace_lease_.reset();
                group_expert_mask_cap_ = 0;
                group_expert_mask_hash_ = 0;
                group_expert_mask_num_experts_ = 0;
                group_expert_mask_active_experts_ = 0;
                group_expert_mask_published_ = false;
                return false;
            }

            const size_t table_bytes =
                workspace_->getBufferSize(MoEWorkspaceBuffers::GROUP_EXPERT_MASK);
            const size_t slot_stride =
                table_bytes /
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            if (slot_stride < static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[CUDAMoEKernel::updateGroupedPrefillExpertMask] "
                          "leased expert-mask slot is too narrow: stride="
                          << slot_stride << " requested_experts=" << num_experts);
                d_group_expert_mask_ = nullptr;
                group_expert_mask_workspace_lease_.reset();
                group_expert_mask_cap_ = 0;
                group_expert_mask_published_ = false;
                return false;
            }

            d_group_expert_mask_ =
                static_cast<uint8_t *>(mask_table) + lease->slot() * slot_stride;
            group_expert_mask_workspace_lease_ = std::move(lease);
            group_expert_mask_cap_ = static_cast<int>(slot_stride);
            group_expert_mask_hash_ = 0;
            group_expert_mask_num_experts_ = 0;
            group_expert_mask_active_experts_ = 0;
            group_expert_mask_published_ = false;
        }

        cudaError_t err = cudaMemcpyAsync(
            d_group_expert_mask_,
            expert_mask,
            static_cast<size_t>(num_experts) * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedPrefillExpertMask] H2D mask copy failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        group_expert_mask_hash_ = mask_hash;
        group_expert_mask_num_experts_ = num_experts;
        group_expert_mask_active_experts_ = active_experts;
        group_expert_mask_published_ = true;
        return true;
    }

    bool CUDAMoEKernel::prepareSharedExpertPrefillGroup(DeviceRowRange rows, MoESharedExpertGroupKind kind)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginSharedExpert(kind, rows);
        const int seq_len = rows.physicalRows();
        if (seq_len <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareSharedExpertPrefillGroup");
        if (!ensureGroupingBufferCapacity(seq_len, /*num_experts=*/1))
            return false;
        if (!cudaMoE_prepare_shared_expert_group(
                d_group_offsets_,
                d_group_counts_,
                d_group_token_indices_,
                d_group_original_to_grouped_,
                d_group_original_expert_ids_,
                d_group_weights_,
                d_group_active_expert_ids_,
                rows,
                device_ordinal_,
                stream))
        {
            return false;
        }
        group_active_expert_slots_ = 1;
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_shared_expert_prefill_group_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", "1"},
             {"active_expert_slots", "1"}});
        std::move(grouping_publication).publish();
        return true;
    }

    bool CUDAMoEKernel::exportGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packet, int participant)
    {
        // These pointers are owned by the already-admitted grouped workspace.
        // Do not expose names or pointer arithmetic to model graph callers.
        if (!layout.valid() || !route_participants || !packet ||
            packet->size_bytes() < layout.packetBytes() ||
            participant < 0 || static_cast<std::uint32_t>(participant) >= layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ || !setMoEDevice(device_ordinal_, "exportGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::cuda(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "exportGroupedPrefillIntermediates(packet)"))
            return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        const MoEGroupedIntermediatePackLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .grouped_values = floating ? static_cast<const void *>(d_prefill_gate_)
                                      : static_cast<const void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_,
            .packet = static_cast<std::uint32_t *>(packet->gpu_data_ptr()),
            .participant = participant};
        if (!cuda::packGroupedIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool CUDAMoEKernel::importGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packets)
    {
        if (!layout.valid() || !route_participants || !packets ||
            packets->size_bytes() < layout.packetBytes() * layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ || !setMoEDevice(device_ordinal_, "importGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::cuda(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packets, device, stream, "importGroupedPrefillIntermediates(packets)"))
            return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        const MoEGroupedIntermediateConsumeLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .participant_packets = static_cast<const std::uint32_t *>(packets->gpu_data_ptr()),
            .grouped_values = floating ? static_cast<void *>(d_prefill_gate_)
                                      : static_cast<void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_};
        return cuda::consumeGroupedIntermediate(launch, stream);
    }

    bool CUDAMoEKernel::exportCompactGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const DeviceMoELayerRuntime &runtime, ITensor *packet,
        std::uint64_t *packet_bytes)
    {
        if (!layout.compactValid() || !packet || !packet_bytes ||
            reinterpret_cast<std::uintptr_t>(packet_bytes) % alignof(std::uint64_t) ||
            packet->size_bytes() < layout.compactCapacityBytes() ||
            runtime.participant_count != layout.participants || runtime.participant_id >= layout.participants ||
            !runtime.expert_count || runtime.expert_count > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            runtime.prefill_route_capacity < layout.route_capacity ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !runtime.expert_offsets || !runtime.expert_counts || !runtime.route_participant_ids ||
            !d_group_original_to_grouped_ ||
            !setMoEDevice(device_ordinal_, "exportCompactGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::cuda(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "exportCompactGroupedPrefillIntermediates(packet)")) return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        // Runtime-plan execution reads runtime-owned counts, not the kernel's
        // private grouping scratch. That scratch may still describe a prior
        // layer. Bind the same authoritative addresses used by gate/up, with
        // the actual expert geometry rather than the allocation's capacity.
        const MoECompactIntermediatePackLaunch launch{
            .payload = {
                .layout = layout,
                .route_owners = runtime.route_participant_ids,
                .original_to_grouped = d_group_original_to_grouped_,
                .grouped_values = floating ? static_cast<const void *>(d_prefill_gate_)
                                          : static_cast<const void *>(d_prefill_swiglu_int8_),
                .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_,
                .packet = static_cast<std::uint32_t *>(packet->gpu_data_ptr()),
                .participant = static_cast<std::int32_t>(runtime.participant_id)},
            .last_group_offset = runtime.expert_offsets + runtime.expert_count - 1,
            .last_group_count = runtime.expert_counts + runtime.expert_count - 1,
            .packet_bytes = packet_bytes};
        if (!cuda::packCompactIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool CUDAMoEKernel::importCompactGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packet,
        std::size_t packet_byte_offset, const std::uint64_t *packet_bytes, int participant)
    {
        if (!layout.compactValid() || !route_participants || !packet || !packet_bytes ||
            reinterpret_cast<std::uintptr_t>(packet_bytes) % alignof(std::uint64_t) ||
            packet_byte_offset % alignof(std::uint32_t) || packet_byte_offset > packet->size_bytes() ||
            layout.compactCapacityBytes() > packet->size_bytes() - packet_byte_offset ||
            participant < 0 || static_cast<std::uint32_t>(participant) >= layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ ||
            !setMoEDevice(device_ordinal_, "importCompactGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::cuda(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packet, device, stream, "importCompactGroupedPrefillIntermediates(packet)")) return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        // Regrouping has already rebound the original router slots. Import
        // changes only this producer's live rows, so peers and empty packets
        // cannot overwrite each other's contributions or any unused capacity.
        // The offset selects bytes, not a new tensor/coherence authority. The
        // capture ledger above authenticates the published arena owner itself.
        const MoECompactIntermediateConsumeLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .packet = static_cast<const std::uint32_t *>(packet->gpu_data_ptr()) + packet_byte_offset / sizeof(std::uint32_t),
            .packet_bytes = packet_bytes,
            .participant = participant,
            .grouped_values = floating ? static_cast<void *>(d_prefill_gate_)
                                      : static_cast<void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_};
        return cuda::consumeCompactIntermediate(launch, stream);
    }

    bool CUDAMoEKernel::executeGroupedPrefillProjection(
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        const MoEPrefillProjectionExecution &execution,
        ITensor *canonical_route_contributions)
    {
        if (execution.isIntermediateShard() ||
            (gateup_desc_table_id >= 0 && static_cast<size_t>(gateup_desc_table_id) < grouped_gateup_desc_tables_.size() &&
             grouped_gateup_desc_tables_[gateup_desc_table_id].descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_desc_table_id >= 0 && static_cast<size_t>(down_desc_table_id) < grouped_down_desc_tables_.size() &&
             grouped_down_desc_tables_[down_desc_table_id].descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
        {
            LOG_ERROR("Intermediate-shard partial execution requires the acquired runtime plan and original cohort binding");
            return false;
        }

if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
            return false;
        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            (execution.executesDown() && down_table.d_model != execution.columnCount()) ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] descriptor table shape mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::executeGroupedPrefillProjection");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        const int max_tokens_per_expert = seq_len;
        const int splitk_tile_rows =
            std::min(seq_len, MoEWorkspaceBuffers::kVerifierSplitKTileRows);
        const int splitk_route_slots = splitk_tile_rows * top_k;
        const int active_expert_slots = group_active_expert_slots_;
        const int *d_active_expert_ids =
            (active_expert_slots > 0) ? d_group_active_expert_ids_ : nullptr;

        /*
         * Floating experts retain the same ordered grouping transaction but
         * consume their typed descriptor family directly. The fixed-tree
         * kernels deliberately bypass router-Q8 reuse and codebook policy:
         * neither is meaningful for a contiguous FP16/BF16/FP32 matrix.
         */
        if (floating)
        {
            ITensor *publication_output = canonical_route_contributions
                                              ? canonical_route_contributions
                                              : output;
            const char *publication_output_name =
                canonical_route_contributions
                    ? "canonical_route_contributions"
                    : "output";
            if (!d_group_original_to_grouped_ ||
                !d_group_original_expert_ids_ || !d_group_weights_ ||
                !ensureGroupedPrefillScratchCapacity(
                    total_slots, d_model, intermediate) ||
                (execution.executesGateUp() && !requireTensorOnDevice(hidden, device, stream, "hidden")) ||
                (execution.executesDown() && !requireOutputOnDevice(
                    publication_output,
                    device,
                    stream,
                    publication_output_name)))
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped-prefill publication is incomplete");
                return false;
            }

            const bool ok = execution.overwriteOutputThenPublish(
                execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
                execution.executesDown() ? publication_output->size_bytes() : 0u,
                seq_len, top_k,
                canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                              : MoEPrefillOutputLayout::TokenRows,
                stream,
                [&](std::span<float> live_output, void *producer_stream)
                {
                    return cudaMoE_grouped_floating_prefill_pipeline(
                        execution.executesGateUp() ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr,
                        gateup_table.device_floating_gate_descs,
                        gateup_table.device_floating_up_descs,
                        down_table.device_floating_descs,
                        d_group_original_to_grouped_,
                        d_group_original_expert_ids_,
                        d_group_weights_,
                        d_prefill_gate_,
                        canonical_route_contributions ? nullptr : live_output.data(),
                        canonical_route_contributions ? live_output.data() : nullptr,
                        seq_len,
                        total_slots,
                        top_k,
                        d_model,
                        intermediate,
                        num_experts,
                        gateup_table.weight_format,
                        device_ordinal_,
                        producer_stream, execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped CUDA pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_floating_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_expert_slots", std::to_string(active_expert_slots)},
                 {"descriptor_source", "static_table"},
                 {"weight_format", std::to_string(static_cast<uint32_t>(
                                       gateup_table.weight_format))}});
            return true;
        }

        const auto projection_engine =
            CUDAMoEBatchInvariantPolicy::groupedProjectionEngine(seq_len);
        const bool use_grouped_imma =
            projection_engine ==
            CUDAMoEBatchInvariantPolicy::GroupedProjectionEngine::
                TensorCoreImmaPrefill;
        const bool use_gateup_kpart =
            active_expert_slots > 0 && !use_grouped_imma;
        CUDAMoEProductionPrefillPolicy prefill_policy =
            selectCUDAMoEProductionPrefillPolicy(
                execution.measuredOperation(
                    canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                                  : MoEPrefillOutputLayout::TokenRows,
                    grouped_prefill_grouping_.scopeForProjection(seq_len, top_k)),
                gateup_table.codebook_mask,
                down_table.codebook_mask,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k,
                use_gateup_kpart, device_ordinal_);
        if (use_gateup_kpart &&
            !validCUDAMoEGateUpOrderedTileN(prefill_policy.gateup_tile_n))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "capture-time prefill policy is invalid"
                      << " seq_len=" << seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate
                      << " num_experts=" << num_experts
                      << " top_k=" << top_k);
            return false;
        }
        if (use_grouped_imma &&
            (!cuda::moe::validGroupedImmaGateUpColumns(
                 prefill_policy.imma_gateup_columns) ||
             !cuda::moe::validGroupedImmaDownColumns(
                 prefill_policy.imma_down_columns) ||
             !cuda::moe::validGroupedImmaGateUpSchedule(
                 prefill_policy.imma_gateup_schedule)))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "capture-time IMMA geometry is invalid");
            return false;
        }
        if (use_gateup_kpart &&
            !ensureGroupedGateUpKPartScratchCapacity(
                splitk_route_slots,
                CUDAMoEBatchInvariantPolicy::gate_up_k_partitions,
                intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "verifier grouped gate/up split-K scratch allocation failed");
            return false;
        }
        if (active_expert_slots > 0 &&
            (!d_group_original_to_grouped_ || !d_group_original_expert_ids_))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "decode-equivalent grouped prefill requires ordered route maps");
            return false;
        }
        const bool use_down_ordered_kpart =
            active_expert_slots > 0 && !use_grouped_imma;
        const bool publishes_canonical_routes =
            canonical_route_contributions != nullptr;
        if (use_down_ordered_kpart && publishes_canonical_routes &&
            !ensureGroupedDownKPartScratchCapacity(
                CUDAMoEBatchInvariantPolicy::down_k_partitions,
                d_model,
                splitk_route_slots))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "verifier grouped down split-K scratch allocation failed");
            return false;
        }
        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            (use_grouped_imma &&
             !ensureGroupedImmaDirectoryCapacity(total_slots, num_experts)) ||
            (execution.executesGateUp() && !requireTensorOnDevice(hidden, device, stream, "hidden")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name)))
            return false;

        const float *d_hidden = execution.executesGateUp()
            ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr;
        float *d_output = !execution.executesDown() || canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            execution.executesDown() && canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;
        if ((execution.executesGateUp() && !d_hidden) ||
            (execution.executesDown() && !d_output && !d_canonical_route_contributions))
            return false;
        const char *router_q8_reuse_block_reason =
            routerQ8HiddenReuseBlockReason(d_hidden, seq_len, d_model);
        const bool reuse_router_q8_hidden =
            execution.executesGateUp() && active_expert_slots > 0 &&
            router_q8_reuse_block_reason == nullptr;
        if (execution.executesGateUp() && router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::RequiredConsumer &&
            !reuse_router_q8_hidden)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] required "
                      "router Q8 publication is unavailable: "
                      << (router_q8_reuse_block_reason
                              ? router_q8_reuse_block_reason
                              : "no_active_experts"));
            return false;
        }
        if (execution.executesGateUp() && !reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_router_q8_reuse_blocked_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", "static_table"},
                 {"reason", active_expert_slots <= 0
                                ? "no_active_experts"
                                : (router_q8_reuse_block_reason
                                       ? router_q8_reuse_block_reason
                                       : "unpublished")}});
        }

        /*
         * Ordered scatter is the only active-route publication path. Every
         * token/column has one owner, so no floating-point atomic accumulation
         * or destination pre-zero is involved. A participant with no local
         * routes still publishes an all-zero contribution for the following
         * collective; that no-work case is the sole reason to clear output.
         */
        const bool ordered_scatter_overwrites_output =
            active_expert_slots > 0 && d_group_original_to_grouped_ != nullptr;
        if (active_expert_slots > 0 && !ordered_scatter_overwrites_output)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] "
                      "active grouped prefill routes require ordered scatter ownership");
            return false;
        }

        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                if (execution.executesDown() && active_expert_slots == 0)
                {
                    cudaStream_t cuda_stream = static_cast<cudaStream_t>(producer_stream);
                    cudaError_t err = cudaMemsetAsync(
                        live_output.data(),
                        0,
                        live_output.size_bytes(),
                        cuda_stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] output memset failed: "
                                  << cudaGetErrorString(err));
                        return false;
                    }
                }
                return cudaMoE_grouped_prefill_pipeline(
                    d_hidden,
                    reuse_router_q8_hidden
                        ? router_q8_hidden_publication_->quantized_rows
                        : nullptr,
                    reuse_router_q8_hidden
                        ? router_q8_hidden_publication_->row_scales
                        : nullptr,
                    gateup_table.device_gate_descs,
                    gateup_table.device_up_descs,
                    down_table.device_descs,
                    d_group_counts_,
                    d_group_offsets_,
                    use_grouped_imma ? d_prefill_imma_directory_ : nullptr,
                    use_grouped_imma
                        ? prefill_imma_directory_entries_cap_
                        : 0,
                    d_group_token_indices_,
                    ordered_scatter_overwrites_output ? d_group_original_to_grouped_ : nullptr,
                    use_gateup_kpart || use_down_ordered_kpart
                        ? d_group_original_expert_ids_
                        : nullptr,
                    d_active_expert_ids,
                    d_group_weights_,
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    use_gateup_kpart ? d_grouped_gateup_gate_partials_ : nullptr,
                    use_gateup_kpart ? d_grouped_gateup_up_partials_ : nullptr,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    use_down_ordered_kpart && publishes_canonical_routes
                        ? d_grouped_down_partials_
                        : nullptr,
                    d_prefill_gate_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    max_tokens_per_expert,
                    total_slots,
                    top_k,
                    active_expert_slots,
                    0,
                    gateup_table.codebook_id,
                    down_table.codebook_id,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    use_gateup_kpart ? CUDAMoEBatchInvariantPolicy::gate_up_k_partitions : 0,
                    prefill_policy.gateup_tile_n,
                    use_down_ordered_kpart ? CUDAMoEBatchInvariantPolicy::down_k_partitions : 0,
                    splitk_tile_rows,
                    projection_engine,
                    prefill_policy.imma_gateup_columns,
                    prefill_policy.imma_down_columns,
                    prefill_policy.imma_gateup_schedule,
                    device_ordinal_,
                    producer_stream, execution);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjection] grouped CUDA pipeline failed");
            return false;
        }

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", "static_table"}});
        }

        const int selected_tile_m = use_grouped_imma
                                        ? cuda::moe::kGroupedImmaTileRows
                                        : (use_gateup_kpart
                                               ? 0
                                               : selectGroupedPrefillTileM(
                                                     debugEnv().gemm.cuda_moe_prefill_tile_m,
                                                     max_tokens_per_expert));
        const int selected_tile_n = prefill_policy.gateup_tile_n;
        recordGroupedPrefillCounters(
            seq_len,
            top_k,
            num_experts,
            active_expert_slots,
            selected_tile_m,
            selected_tile_n,
            gateup_table.codebook_mask,
            down_table.codebook_mask,
            gateup_table.policy_codebook_mask,
            down_table.policy_codebook_mask,
            use_grouped_imma,
            use_gateup_kpart,
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu,
            use_down_ordered_kpart,
            ordered_scatter_overwrites_output,
            canonical_route_contributions != nullptr,
            splitk_tile_rows,
            prefill_policy.imma_gateup_columns,
            prefill_policy.imma_down_columns,
            prefill_policy.imma_gateup_schedule,
            prefill_policy.source);
        return true;
    }

    bool CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan(
        DeviceMoELayerRuntime *device_runtime_layer,
        const DeviceMoELayerRuntime &runtime_host_layer,
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        const MoEPrefillProjectionExecution &execution,
        ITensor *canonical_route_contributions)
    {
        if (execution.isIntermediateShard() != runtime_host_layer.intermediate_shard.enabled() ||
            (execution.isIntermediateShard() &&
             (runtime_host_layer.intermediate_shard.modelColumns() != static_cast<uint32_t>(d_model) ||
              runtime_host_layer.intermediate_shard.localIntermediateColumns() != static_cast<uint32_t>(intermediate) ||
              !canonical_route_contributions)))
            return false;

        const uint64_t publication_generation = grouped_prefill_grouping_.generation();
        const auto published_gate = std::exchange(runtime_gate_plan_, std::nullopt);
        grouped_prefill_grouping_.invalidate();
        if (!device_runtime_layer)
            return false;
        // Mutable placement never owns immutable column slices. Binding a
        // local-N hint to its full-N packed matrix would change indexing.
        if (execution.executesDown() && execution.columnCount() != d_model)
        {
            LOG_ERROR("Sliced MoE down execution requires the immutable fixed-down descriptor table");
            return false;
        }
if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
            return false;
        if (runtime_host_layer.expert_count != static_cast<uint32_t>(num_experts) ||
            runtime_host_layer.top_k != static_cast<uint32_t>(top_k) ||
            runtime_host_layer.prefill_token_capacity < static_cast<uint32_t>(seq_len) ||
            runtime_host_layer.prefill_route_capacity < static_cast<uint32_t>(seq_len * top_k) ||
            !runtime_host_layer.expert_counts ||
            !runtime_host_layer.expert_offsets ||
            !runtime_host_layer.grouped_token_ids ||
            !runtime_host_layer.grouped_route_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid runtime scratch contract"
                      << " expert_count=" << runtime_host_layer.expert_count
                      << " expected_experts=" << num_experts
                      << " top_k=" << runtime_host_layer.top_k
                      << " expected_top_k=" << top_k
                      << " token_capacity=" << runtime_host_layer.prefill_token_capacity
                      << " seq_len=" << seq_len
                      << " route_capacity=" << runtime_host_layer.prefill_route_capacity
                      << " total_slots=" << (seq_len * top_k));
            return false;
        }
        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()))
            return false;

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            (execution.executesDown() && down_table.d_model != execution.columnCount()) ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] descriptor table shape mismatch");
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        const int max_tokens_per_expert = seq_len;
        const int splitk_tile_rows =
            std::min(seq_len, MoEWorkspaceBuffers::kVerifierSplitKTileRows);
        const int splitk_route_slots = splitk_tile_rows * top_k;
        const int active_expert_slots = std::min(total_slots, num_experts);

        if (floating)
        {
            if (!runtime_host_layer.route_expert_ids ||
                !ensureGroupingBufferCapacity(total_slots, num_experts) ||
                !ensureGroupedPrefillScratchCapacity(
                    total_slots, d_model, intermediate))
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                          "floating runtime grouping workspace is incomplete");
                return false;
            }

            ITensor *publication_output = canonical_route_contributions
                                              ? canonical_route_contributions
                                              : output;
            const char *publication_output_name =
                canonical_route_contributions
                    ? "canonical_route_contributions"
                    : "output";
            if ((execution.executesGateUp() && !requireTensorOnDevice(hidden, device, stream, "hidden")) ||
                (execution.executesDown() && !requireOutputOnDevice(
                    publication_output,
                    device,
                    stream,
                    publication_output_name)))
            {
                return false;
            }

            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    num_experts,
                    &runtime_gate_descs,
                    "CUDA floating runtime prefill gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    num_experts,
                    &runtime_up_descs,
                    "CUDA floating runtime prefill up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    num_experts,
                    &runtime_down_descs,
                    "CUDA floating runtime prefill down descriptors"))
            {
                return false;
            }

            group_active_expert_slots_ = active_expert_slots;
            const bool ok = execution.overwriteOutputThenPublish(
                execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
                execution.executesDown() ? publication_output->size_bytes() : 0u,
                seq_len, top_k,
                canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                              : MoEPrefillOutputLayout::TokenRows,
                stream,
                [&](std::span<float> live_output, void *producer_stream)
                {
                    return cudaMoE_grouped_floating_prefill_pipeline(
                        execution.executesGateUp() ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr,
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_gate_descs),
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_up_descs),
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_down_descs),
                        d_group_original_to_grouped_,
                        runtime_host_layer.route_expert_ids,
                        runtime_host_layer.grouped_route_weights,
                        d_prefill_gate_,
                        canonical_route_contributions ? nullptr : live_output.data(),
                        canonical_route_contributions ? live_output.data() : nullptr,
                        seq_len,
                        total_slots,
                        top_k,
                        d_model,
                        intermediate,
                        num_experts,
                        gateup_table.weight_format,
                        device_ordinal_,
                        producer_stream, execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                          "floating grouped CUDA pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_floating_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_expert_slots", std::to_string(active_expert_slots)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"},
                 {"weight_format", std::to_string(static_cast<uint32_t>(
                                       gateup_table.weight_format))}});
            return true;
        }

        const auto projection_engine =
            CUDAMoEBatchInvariantPolicy::groupedProjectionEngine(seq_len);
        const bool use_grouped_imma =
            projection_engine ==
            CUDAMoEBatchInvariantPolicy::GroupedProjectionEngine::
                TensorCoreImmaPrefill;
        const bool use_gateup_kpart =
            active_expert_slots > 0 && !use_grouped_imma;
        CUDAMoEProductionPrefillPolicy prefill_policy =
            selectCUDAMoEProductionPrefillPolicy(
                execution.measuredOperation(
                    canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                                  : MoEPrefillOutputLayout::TokenRows,
                    MoEPrefillRouteScope::SelectedRoutes),
                gateup_table.codebook_mask,
                down_table.codebook_mask,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k,
                use_gateup_kpart, device_ordinal_);
        if (use_gateup_kpart &&
            !validCUDAMoEGateUpOrderedTileN(prefill_policy.gateup_tile_n))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "capture-time prefill policy is invalid"
                      << " seq_len=" << seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate
                      << " num_experts=" << num_experts
                      << " top_k=" << top_k);
            return false;
        }
        if (use_grouped_imma &&
            (!cuda::moe::validGroupedImmaGateUpColumns(
                 prefill_policy.imma_gateup_columns) ||
             !cuda::moe::validGroupedImmaDownColumns(
                 prefill_policy.imma_down_columns) ||
             !cuda::moe::validGroupedImmaGateUpSchedule(
                 prefill_policy.imma_gateup_schedule)))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "capture-time IMMA geometry is invalid");
            return false;
        }
        /*
         * Runtime grouped verifier prefill must mirror the public M=1 runtime
         * decode route.  CUDA serial decode uses split-K gate/up for tiny MoE
         * rows by default, so every runtime-M publication has to use the same
         * split-K partial ordering before it evaluates SwiGLU and quantizes the
         * down input. Fixed-size row tiles bound scratch independently of M;
         * this remains one grouped, graph-capturable device sequence rather
         * than serial row replay.
         */
        if (use_gateup_kpart &&
            !ensureGroupedGateUpKPartScratchCapacity(
                splitk_route_slots,
                CUDAMoEBatchInvariantPolicy::gate_up_k_partitions,
                intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "verifier grouped gate/up split-K scratch allocation failed");
            return false;
        }
        const bool use_down_ordered_kpart =
            active_expert_slots > 0 && !use_grouped_imma;
        const bool publishes_canonical_routes =
            canonical_route_contributions != nullptr;
        if ((use_gateup_kpart || use_down_ordered_kpart) &&
            !runtime_host_layer.route_expert_ids)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "verifier grouped down split-K requires runtime route expert ids");
            return false;
        }
        if (use_down_ordered_kpart && publishes_canonical_routes &&
            !ensureGroupedDownKPartScratchCapacity(
                CUDAMoEBatchInvariantPolicy::down_k_partitions,
                d_model,
                splitk_route_slots))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "verifier grouped down split-K scratch allocation failed");
            return false;
        }
        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if (!ensureGroupingBufferCapacity(total_slots, num_experts) ||
            !ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            (use_grouped_imma &&
             !ensureGroupedImmaDirectoryCapacity(total_slots, num_experts)) ||
            (execution.executesGateUp() && !requireTensorOnDevice(hidden, device, stream, "hidden")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name)))
        {
            return false;
        }

        /*
         * Complete plan publication has already written these graph-owned
         * descriptor slots. The compute consumer resolves the same immutable
         * slot identities without launching a hidden materialization step.
         */
        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "CUDA runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "CUDA runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "CUDA runtime prefill down descriptors"))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "failed to bind graph-owned runtime descriptor slots");
            return false;
        }

        const float *d_hidden = execution.executesGateUp()
            ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr;
        float *d_output = !execution.executesDown() || canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            execution.executesDown() && canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;
        if ((execution.executesGateUp() && !d_hidden) ||
            (execution.executesDown() && !d_output && !d_canonical_route_contributions))
            return false;
        const char *router_q8_reuse_block_reason =
            routerQ8HiddenReuseBlockReason(d_hidden, seq_len, d_model);
        const bool reuse_router_q8_hidden =
            execution.executesGateUp() && active_expert_slots > 0 &&
            router_q8_reuse_block_reason == nullptr;
        if (execution.executesGateUp() && !reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_router_q8_reuse_blocked_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"},
                 {"reason", active_expert_slots <= 0
                                ? "no_active_experts"
                                : (router_q8_reuse_block_reason
                                       ? router_q8_reuse_block_reason
                                       : "unpublished")}});
        }

        group_active_expert_slots_ = active_expert_slots;

        /*
         * groupPrefillRoutes() or the post-assignment regroup transaction has
         * already published d_group_original_to_grouped_ beside counts, offsets,
         * and grouped rows. Treating those arrays as one ordered transaction
         * prevents a consumer from observing a grouping table with a stale
         * inverse map and removes reconstruction work from every MoE layer.
         */

        CUDAMoERuntimeGateContract gate_contract;
        if (execution.executesGateUp() && published_gate)
        {
            const auto expected = runtimeGateContract(gateup_table, seq_len, seq_len, top_k);
            if (published_gate->generation != publication_generation || !expected.enabled() ||
                expected.codebook() != published_gate->contract.codebook() ||
                published_gate->runtime != device_runtime_layer ||
                published_gate->gate != runtime_gate_descs || published_gate->up != runtime_up_descs ||
                published_gate->original_to_grouped != d_group_original_to_grouped_ ||
                published_gate->stream != stream || published_gate->experts != num_experts)
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                          "QFN Gate/Up validation belongs to a different runtime publication");
                return false;
            }
            gate_contract = published_gate->contract;
        }

        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                return cudaMoE_grouped_prefill_pipeline(
                    d_hidden,
                    reuse_router_q8_hidden ? d_decode_hidden_int8_ : nullptr,
                    reuse_router_q8_hidden ? d_decode_hidden_scales_ : nullptr,
                    runtime_gate_descs,
                    runtime_up_descs,
                    runtime_down_descs,
                    runtime_host_layer.expert_counts,
                    runtime_host_layer.expert_offsets,
                    use_grouped_imma ? d_prefill_imma_directory_ : nullptr,
                    use_grouped_imma
                        ? prefill_imma_directory_entries_cap_
                        : 0,
                    runtime_host_layer.grouped_token_ids,
                    d_group_original_to_grouped_,
                    use_gateup_kpart || use_down_ordered_kpart
                        ? runtime_host_layer.route_expert_ids
                        : nullptr,
                    d_group_active_expert_ids_,
                    runtime_host_layer.grouped_route_weights,
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    use_gateup_kpart ? d_grouped_gateup_gate_partials_ : nullptr,
                    use_gateup_kpart ? d_grouped_gateup_up_partials_ : nullptr,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    use_down_ordered_kpart && publishes_canonical_routes
                        ? d_grouped_down_partials_
                        : nullptr,
                    d_prefill_gate_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    max_tokens_per_expert,
                    total_slots,
                    top_k,
                    active_expert_slots,
                    1,
                    gateup_table.codebook_id,
                    down_table.codebook_id,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    use_gateup_kpart ? CUDAMoEBatchInvariantPolicy::gate_up_k_partitions : 0,
                    prefill_policy.gateup_tile_n,
                    use_down_ordered_kpart ? CUDAMoEBatchInvariantPolicy::down_k_partitions : 0,
                    splitk_tile_rows,
                    projection_engine,
                    prefill_policy.imma_gateup_columns,
                    prefill_policy.imma_down_columns,
                    prefill_policy.imma_gateup_schedule,
                    device_ordinal_,
                    producer_stream, execution, gate_contract);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] grouped CUDA pipeline failed");
            return false;
        }

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"}});
        }

        recordGroupedPrefillCounters(
            seq_len,
            top_k,
            num_experts,
            active_expert_slots,
            use_grouped_imma
                ? cuda::moe::kGroupedImmaTileRows
                : (use_gateup_kpart
                       ? 0
                       : selectGroupedPrefillTileM(
                             debugEnv().gemm.cuda_moe_prefill_tile_m,
                             max_tokens_per_expert)),
            prefill_policy.gateup_tile_n,
            gateup_table.codebook_mask,
            down_table.codebook_mask,
            gateup_table.policy_codebook_mask,
            down_table.policy_codebook_mask,
            use_grouped_imma,
            use_gateup_kpart,
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu,
            use_down_ordered_kpart,
            true,
            canonical_route_contributions != nullptr,
            splitk_tile_rows,
            prefill_policy.imma_gateup_columns,
            prefill_policy.imma_down_columns,
            prefill_policy.imma_gateup_schedule,
            prefill_policy.source);
        return true;
    }

    bool CUDAMoEKernel::reduceCanonicalRouteContributions(
        ITensor *canonical_route_contributions,
        ITensor *output,
        int seq_len,
        int top_k,
        int d_model)
    {
        if (!canonical_route_contributions || !output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0)
        {
            return false;
        }

        void *stream =
            requireStream("CUDAMoEKernel::reduceCanonicalRouteContributions");
        const DeviceId device = deviceId();
        if (!setMoEDevice(
                device_ordinal_,
                "reduceCanonicalRouteContributions") ||
            !requireTensorOnDevice(
                canonical_route_contributions,
                device,
                stream,
                "canonical_route_contributions") ||
            !requireOutputOnDevice(output, device, stream, "output"))
        {
            return false;
        }

        const float *d_contributions = static_cast<const float *>(
            canonical_route_contributions->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!cudaMoE_reduce_canonical_route_contributions(
                d_contributions,
                d_output,
                seq_len,
                top_k,
                d_model,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(output, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_moe_canonical_route_reduce_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"d_model", std::to_string(d_model)}});
        return true;
    }

    bool CUDAMoEKernel::publishSharedExpertRankBank(
        ITensor *shared_output,
        ITensor *canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *device_effective_seq_len)
    {
        if (!shared_output || !canonical_publication ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0 || participant_index < 0 ||
            participant_index >= participant_count)
        {
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::publishSharedExpertRankBank");
        const DeviceId device = deviceId();
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_elements =
            row_elements *
            (static_cast<size_t>(top_k) +
             static_cast<size_t>(participant_count));
        if (shared_output->numel() < row_elements ||
            canonical_publication->numel() < required_elements ||
            !setMoEDevice(device_ordinal_, "publishSharedExpertRankBank") ||
            !requireTensorOnDevice(
                shared_output,
                device,
                stream,
                "shared_output") ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication"))
        {
            return false;
        }

        if (!cudaMoE_publish_shared_expert_rank_bank(
                static_cast<const float *>(shared_output->gpu_data_ptr()),
                static_cast<float *>(canonical_publication->gpu_data_ptr()),
                seq_len,
                top_k,
                d_model,
                participant_index,
                participant_count,
                device_effective_seq_len,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(canonical_publication, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_moe_shared_rank_bank_publish_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

    bool CUDAMoEKernel::finalizeCanonicalMoEPublication(
        ITensor *input,
        ITensor *gate_inp,
        ITensor *canonical_publication,
        ITensor *routed_output,
        ITensor *shared_output,
        ITensor *combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *device_effective_seq_len)
    {
        if (!input || !gate_inp || !canonical_publication ||
            !routed_output || !shared_output || !combined_output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0)
        {
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::finalizeCanonicalMoEPublication");
        const DeviceId device = deviceId();
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_elements =
            row_elements *
            (static_cast<size_t>(top_k) +
             static_cast<size_t>(participant_count));
        if (input->numel() < row_elements ||
            gate_inp->numel() < static_cast<size_t>(d_model) ||
            canonical_publication->numel() < required_elements ||
            routed_output->numel() < row_elements ||
            shared_output->numel() < row_elements ||
            combined_output->numel() < row_elements ||
            !setMoEDevice(
                device_ordinal_,
                "finalizeCanonicalMoEPublication") ||
            !requireTensorOnDevice(input, device, stream, "input") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication") ||
            !requireOutputOnDevice(
                routed_output,
                device,
                stream,
                "routed_output") ||
            !requireOutputOnDevice(
                shared_output,
                device,
                stream,
                "shared_output") ||
            !requireOutputOnDevice(
                combined_output,
                device,
                stream,
                "combined_output"))
        {
            return false;
        }

        if (!cudaMoE_finalize_canonical_publication(
                static_cast<const float *>(input->gpu_data_ptr()),
                static_cast<const float *>(gate_inp->gpu_data_ptr()),
                static_cast<const float *>(
                    canonical_publication->gpu_data_ptr()),
                static_cast<float *>(routed_output->gpu_data_ptr()),
                static_cast<float *>(shared_output->gpu_data_ptr()),
                static_cast<float *>(combined_output->gpu_data_ptr()),
                seq_len,
                top_k,
                d_model,
                participant_count,
                device_effective_seq_len,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(routed_output, device, stream);
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_moe_canonical_publication_finalize_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(
        const TensorBase *input,
        const int *expert_ids,
        int table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        if (!input || !expert_ids || table_id < 0 || num_active <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
            return false;
        if (num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << table_id
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
        /* NativeQuantised quantizes in 32-element blocks; contiguous floating
         * tables use the exact published K width and do not inherit that
         * packing constraint. */
        if (!floating && (d_model % 32) != 0)
            return false;
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
                return false;
        }
        const int *fixed_device_expert_ids = nullptr;
        if (!resolveFixedTableGateUpMetadata(
                table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                num_active,
                &fixed_device_expert_ids))
            return false;

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromTable");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;
        if (!requireTensorType(input, TensorType::FP32, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireMatrixCapacity(input, 1, d_model, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireTensorElements(input, static_cast<size_t>(d_model), "input", "groupedExpertGateUpDecodeFromTable"))
            return false;

        const int k_partitions =
            CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if (!floating &&
            !ensureGroupedGateUpKPartScratchCapacity(
                num_active, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] "
                      "mandatory K-part gate/up scratch allocation failed");
            return false;
        }
        // Floating descriptors never consume Q8 publications. Requiring their
        // scratch would incorrectly impose quantized block-width constraints.
        if (!floating && !ensureGroupedGateUpDecodeCapacity(num_active, d_model))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] input upload required during graph capture");
            return false;
        }
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    device,
                    stream,
                    "input"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        if (!d_hidden)
            return false;
        if (!requireCudaDevicePointer(
                d_hidden, device_ordinal_, "input",
                "groupedExpertGateUpDecodeFromTable", stream))
            return false;
        if (!floating &&
            (!requireCudaDevicePointer(
                 d_decode_hidden_int8_, device_ordinal_,
                 "decode hidden int8",
                 "groupedExpertGateUpDecodeFromTable", stream) ||
             !requireCudaDevicePointer(
                 d_decode_hidden_scales_, device_ordinal_,
                 "decode hidden scales",
                 "groupedExpertGateUpDecodeFromTable", stream) ||
             !requireCudaDevicePointer(
                 d_grouped_gateup_gate_partials_, device_ordinal_,
                 "gate partials",
                 "groupedExpertGateUpDecodeFromTable", stream) ||
             !requireCudaDevicePointer(
                 d_grouped_gateup_up_partials_, device_ordinal_,
                 "up partials",
                 "groupedExpertGateUpDecodeFromTable", stream)))
            return false;
        if (!requireTensorElements(
                input, static_cast<size_t>(d_model), "input",
                "groupedExpertGateUpDecodeFromTable"))
            return false;

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
                return false;
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                capture_active)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] output allocation required during graph capture");
                return false;
            }
            if (!requireOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !requireOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
                return false;
            auto *gate_output_base = requireTensor(gate_outputs[slot], "groupedExpertGateUpDecodeFromTable gate_output");
            auto *up_output_base = requireTensor(up_outputs[slot], "groupedExpertGateUpDecodeFromTable up_output");
            if (!requireTensorType(gate_output_base, TensorType::FP32, "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorType(up_output_base, TensorType::FP32, "up_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorElements(gate_outputs[slot], static_cast<size_t>(intermediate), "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorElements(up_outputs[slot], static_cast<size_t>(intermediate), "up_output", "groupedExpertGateUpDecodeFromTable"))
                return false;
            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
            if (!requireAlignedPointer(gate_ptrs[slot], 16, "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireAlignedPointer(up_ptrs[slot], 16, "up_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireCudaDevicePointer(gate_ptrs[slot], device_ordinal_, "gate_output", "groupedExpertGateUpDecodeFromTable", stream) ||
                !requireCudaDevicePointer(up_ptrs[slot], device_ordinal_, "up_output", "groupedExpertGateUpDecodeFromTable", stream))
                return false;
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                                              num_active, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;

        /* Both descriptor families use the same graph-owned expert-id and
         * output-pointer publications. Only the arithmetic kernel differs. */
        const bool reuse_required_hidden=router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source();
        int8_t* original_hidden_codes=d_decode_hidden_int8_;
        float* original_hidden_scales=d_decode_hidden_scales_;
        if(reuse_required_hidden) {
            const auto& publication=*router_q8_hidden_publication_;
            if(floating)throw std::logic_error("Required M1 hidden Q8 reader has incompatible floating projection arithmetic");
            if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer) {
                if(!canReuseRouterQ8Hidden(d_hidden,1,d_model))throw std::logic_error("Required M1 routed hidden Q8 reader is absent");
                original_hidden_codes=publication.original_m1_binding->codes();
                original_hidden_scales=publication.original_m1_binding->scales();
            } else {
            if(!original_m1_hidden_q8_read_||publication.backend!=DeviceType::CUDA||publication.device_ordinal!=device_ordinal_||
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
            ? cudaMoE_grouped_gate_up_floating_decode_table(
                  d_hidden,
                  table.device_floating_gate_descs,
                  table.device_floating_up_descs,
                  fixed_device_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.weight_format,
                  device_ordinal_,
                  stream)
            : cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart(
                  d_hidden,
                  table.device_gate_descs,
                  table.device_up_descs,
                  fixed_device_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  original_hidden_codes,
                  original_hidden_scales,
                  reuse_required_hidden,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.codebook_id,
                  k_partitions,
                  device_ordinal_,
                  stream);
        if (ok)
        {
            for (int slot = 0; slot < num_active; ++slot)
            {
                markDeviceWritten(gate_outputs[slot], device, stream);
                markDeviceWritten(up_outputs[slot], device, stream);
            }
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_gateup_calls", "table", num_active,
                d_model, intermediate,
                floating ? "floating_fixed_tree" : "kpart");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromTable(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        int table_id,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            table_id < 0 || num_active <= 0 || !output ||
            d_model <= 0 || intermediate <= 0)
            return false;
        if (num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << table_id
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
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
                return false;
        }
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

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromTable");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if ((!floating && !ensureGroupedDownDecodeCapacity(num_active, intermediate)) ||
            !setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromTable"))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
                return false;
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
        }

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] output allocation required during graph capture");
            return false;
        }
        if (!requireOutputOnDevice(output, device, stream, "moe_output"))
            return false;
        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                                            num_active, gate_ptrs, up_ptrs,
                                            &d_gate_ptrs, &d_up_ptrs))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const int k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if (!floating &&
            !ensureGroupedDownKPartScratchCapacity(
                k_partitions, d_model, num_active))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] "
                      "mandatory K-part down scratch allocation failed");
            return false;
        }

        const bool ok = floating
            ? cudaMoE_grouped_swiglu_down_floating_decode_table(
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
                  stream)
            : cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  fixed_device_expert_ids,
                  fixed_device_expert_weights,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_grouped_down_partials_,
                  d_output,
                  nullptr,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.codebook_id,
                  k_partitions,
                  device_ordinal_,
                  stream);
        if (ok)
        {
            markDeviceWritten(output, device, stream);
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_down_calls", "table", num_active,
                d_model, intermediate,
                floating ? "floating_fixed_tree" : "kpart");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        int table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        if (!input || !routing_indices || table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
        {
            return false;
        }
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRouting"))
            return false;
        if (!ensureRoutingDecodeMetadataCapacity(top_k) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model))
        {
            return false;
        }

        const int k_partitions = CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if (!ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] "
                      "mandatory K-part gate/up scratch allocation failed");
            return false;
        }

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] input upload required during graph capture");
            return false;
        }
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    device,
                    stream,
                    "input"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        if (!routing_indices->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] routing upload required during graph capture");
            return false;
        }
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices"))
            return false;
        const float *d_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_hidden || !d_routing_indices)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] missing input/routing device pointer");
            return false;
        }

        /*
         * Dynamic/LLEP verifier rows still consume the device-owned routing
         * tensor, but each LocalTP participant must contribute only the experts
         * it owns.  The masked conversion preserves the original routing tensor
         * for histograms/runtime publication and marks nonlocal top-k slots as
         * inactive in backend-owned scratch.  The grouped kernels already treat
         * expert id -1 as a no-op slot.
         */
        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!cudaMoE_float_to_masked_int(
                    d_routing_indices,
                    d_routing_decode_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!cudaMoE_float_to_int(
                     d_routing_indices,
                     d_routing_decode_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] null output tensor for slot "
                          << slot);
                return false;
            }
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                capture_active)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] output allocation required during graph capture");
                return false;
            }
            if (!requireOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !requireOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
            {
                return false;
            }

            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        const bool ok = cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart(
            d_hidden,
            table.device_gate_descs,
            table.device_up_descs,
            d_routing_decode_expert_ids_,
            d_gate_ptrs,
            d_up_ptrs,
            d_decode_hidden_int8_,
            d_decode_hidden_scales_,
            false,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            k_partitions,
            device_ordinal_,
            stream);
        if (ok)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                markDeviceWritten(gate_outputs[slot], device, stream);
                markDeviceWritten(up_outputs[slot], device, stream);
            }
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_gateup_calls", "routing", top_k,
                d_model, intermediate, "kpart");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromRouting(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        if (!gate_tensors || !up_tensors || !routing_indices || !routing_weights ||
            table_id < 0 || top_k <= 0 || !output ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
        {
            return false;
        }
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromRouting");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromRouting"))
            return false;
        if (!ensureRoutingDecodeMetadataCapacity(top_k) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
        {
            return false;
        }

        if ((!routing_indices->gpu_data_ptr() || !routing_weights->gpu_data_ptr()) &&
            capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] routing upload required during graph capture");
            return false;
        }
        if (!requireTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(routing_weights, device, stream, "routing_weights"))
        {
            return false;
        }

        const float *d_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_routing_indices || !d_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] missing routing device pointer");
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!cudaMoE_float_to_masked_int(
                    d_routing_indices,
                    d_routing_decode_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!cudaMoE_float_to_int(
                     d_routing_indices,
                     d_routing_decode_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] null gate/up tensor for slot "
                          << slot);
                return false;
            }
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] missing gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] output allocation required during graph capture");
            return false;
        }
        if (!requireOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
            return false;

        const int k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if (!ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] "
                      "mandatory K-part down scratch allocation failed");
            return false;
        }

        const bool ok = cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart(
            d_gate_ptrs,
            d_up_ptrs,
            table.device_descs,
            d_routing_decode_expert_ids_,
            d_weights,
            d_decode_swiglu_int8_,
            d_decode_swiglu_scales_,
            d_grouped_down_partials_,
            d_output,
            nullptr,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            k_partitions,
            device_ordinal_,
            stream);
        if (ok)
        {
            markDeviceWritten(output, device, stream);
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_down_calls", "routing", top_k,
                d_model, intermediate, "kpart");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int gateup_table_id,
        int down_table_id,
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
            gateup_table_id < 0 || down_table_id < 0 || top_k <= 0 ||
            !output || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        if (static_cast<size_t>(down_table_id) >= grouped_down_desc_tables_.size()) return false;
        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if (!down_table.valid ||
            (gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        if (!gateup_table.valid || gateup_table.num_experts <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRouting] "
                      "gate/up descriptor table is unavailable");
            return false;
        }
        const bool use_runtime_descriptors =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if ((use_runtime_descriptors && (!runtime_layer || expert_mask)) ||
            (!use_runtime_descriptors && runtime_layer))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRouting] "
                      "runtime placement requires exactly one runtime layer and no host mask");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDecodeFromRouting");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRouting") ||
            !ensureRoutingDecodeMetadataCapacity(top_k))
        {
            return false;
        }

        if ((!routing_indices->gpu_data_ptr() || !routing_weights->gpu_data_ptr()) &&
            capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRouting] "
                      "routing upload is forbidden during graph capture");
            return false;
        }
        if (!requireTensorOnDevice(
                routing_indices, device, stream, "routing_indices") ||
            !requireTensorOnDevice(
                routing_weights, device, stream, "routing_weights"))
        {
            return false;
        }

        const float *device_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *device_routing_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!device_routing_indices || !device_routing_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRouting] "
                      "routing tensors have no device publication");
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(
                    expert_mask, gateup_table.num_experts) ||
                !cudaMoE_float_to_masked_int(
                    device_routing_indices,
                    d_routing_decode_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    gateup_table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!cudaMoE_float_to_int(
                     device_routing_indices,
                     d_routing_decode_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        const int *resolved_expert_ids = d_routing_decode_expert_ids_;
        const float *resolved_weights = device_routing_weights;
        if (use_runtime_descriptors)
        {
            /*
             * The explicit packet remains the route-value producer, while the
             * epoch-pinned runtime bank is the sole placement authority. This
             * existing selector drops non-local experts, publishes compact
             * runtime top-k state, and publishes selected/local histograms only when this packet owns demand in
             * the same captured stream transaction.
             */
            if (!cudaMoE_decode_route_select_runtime(
                    d_routing_decode_expert_ids_,
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
            gateup_table_id,
            down_table_id,
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

    bool CUDAMoEKernel::prepareGroupedRuntimeDecodeLaunchState(
        int gateup_table_id,
        int down_table_id,
        int top_k,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        void *stream = requireStream(
            "CUDAMoEKernel::prepareGroupedRuntimeDecodeLaunchState");
        if (isCudaMoEDecodeCaptureActive(stream))
        {
            LOG_ERROR("[CUDAMoEKernel] Runtime grouped-decode launch state must be "
                      "prepared before graph capture begins");
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
                          "prepareGroupedRuntimeDecodeLaunchState"))
        {
            return false;
        }
        if (!floating &&
            ((d_model % 32) != 0 || (intermediate % 32) != 0))
        {
            return false;
        }

        const int gateup_k_partitions =
            CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        const int down_k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if ((!floating &&
             (!ensureGroupedGateUpKPartScratchCapacity(
                  top_k, gateup_k_partitions, intermediate) ||
              !ensureGroupedDownKPartScratchCapacity(
                  down_k_partitions, d_model, top_k))) ||
            !ensureGroupedPrefillScratchCapacity(
                top_k, d_model, intermediate) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
        {
            return false;
        }

        if (moEDecodeUsesRuntimeDescriptors(descriptor_source))
        {
            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "CUDA runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "CUDA runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "CUDA runtime decode down descriptors"))
            {
                return false;
            }
        }

        // Native decode binds only the Q8 planes; floating decode still
        // publishes the original separate FP32 gate and up operands.
        if (!floating)
            return true;

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
        return ensureRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               ensureRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool CUDAMoEKernel::prepareGroupedTableDecodeLaunchState(
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
        void *stream = requireStream(
            "CUDAMoEKernel::prepareGroupedTableDecodeLaunchState");
        if (isCudaMoEDecodeCaptureActive(stream))
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table grouped-decode launch state "
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

        const int gateup_k_partitions =
            CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        const int down_k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        const int *fixed_gateup_expert_ids = nullptr;
        const int *fixed_down_expert_ids = nullptr;
        const float *fixed_down_expert_weights = nullptr;
        if ((!floating &&
             (!ensureGroupedGateUpKPartScratchCapacity(
                  num_active, gateup_k_partitions, intermediate) ||
              !ensureGroupedDownKPartScratchCapacity(
                  down_k_partitions, d_model, num_active) ||
              !ensureGroupedGateUpDecodeCapacity(num_active, d_model) ||
              !ensureGroupedDownDecodeCapacity(num_active, intermediate))) ||
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
                LOG_ERROR("[CUDAMoEKernel] Fixed-table grouped decode has no "
                          "persistent gate/up scratch pointer for slot "
                          << slot);
                return false;
            }
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }
        if (!output->gpu_data_ptr())
        {
            LOG_ERROR("[CUDAMoEKernel] Fixed-table grouped decode output is not "
                      "bound to persistent device storage before capture");
            return false;
        }

        float **device_gate_ptrs = nullptr;
        float **device_up_ptrs = nullptr;
        const float **device_down_gate_ptrs = nullptr;
        const float **device_down_up_ptrs = nullptr;
        return ensureRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               ensureRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool CUDAMoEKernel::groupedExpertDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_table_id,
        int down_table_id,
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
            gateup_table_id,
            down_table_id,
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
                ? "runtime"
                : "runtime_static_table",
            canonical_route_contributions, weighting);
    }

    bool CUDAMoEKernel::groupedExpertDecodeResolved(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_table_id,
        int down_table_id,
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

        if (!input || gateup_table_id < 0 || down_table_id < 0 ||
            top_k <= 0 || (!output && !canonical_route_contributions) || d_model <= 0 || intermediate <= 0)
            return false;
        if (!d_expert_ids || !d_weights ||
            (use_runtime_descriptors && !runtime_layer))
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
            return false;
        if (gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

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
            gateup_table.d_model != d_model || gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model || down_table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDecodeFromRuntime");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRuntime"))
            return false;

        const int gateup_k_partitions =
            CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if (!floating && !ensureGroupedGateUpKPartScratchCapacity(
                             top_k, gateup_k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                      "mandatory K-part gate/up scratch allocation failed");
            return false;
        }
        const int down_k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if (!floating && !ensureGroupedDownKPartScratchCapacity(
                             down_k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                      "mandatory K-part down scratch allocation failed");
            return false;
        }

        if (!ensureGroupedPrefillScratchCapacity(top_k, d_model, intermediate) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
            return false;

        /*
         * Mutable placement publishes descriptor values from the active device
         * runtime bank while immutable placement reads the uploaded model
         * tables. Keep both modes on the same compact-table GEMV launch path:
         * the runtime mode first materializes its active bank into stable
         * graph-owned workspace slots on this exact producer stream.
         *
         * Besides avoiding a cache-hostile walk through the complete runtime
         * object from every output lane, this gives CUDA and ROCm one placement
         * contract. A descriptor is executable only when the active bank says
         * the expert is both locally computable and resident on this
         * participant. DeviceMoERuntimeTable enforces that invariant when it
         * publishes a bank; the materializer preserves it during every graph
         * replay without host inspection, allocation, or synchronization.
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
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "CUDA runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "CUDA runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "CUDA runtime decode down descriptors"))
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                          "failed to bind graph-owned runtime descriptor slots");
                return false;
            }
            // Selected decode rewrites these same compact descriptor slots.
            runtime_gate_plan_.reset();
            if (!cudaMoE_materialize_runtime_selected_decode_descriptors(
                runtime_layer, d_expert_ids, runtime_gate_descs, runtime_up_descs,
                runtime_down_descs, gateup_table.num_experts, top_k,
                gateup_table.weight_format, device_ordinal_, stream,
                moEDecodeProjectionSet(descriptor_source)))
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                          "failed to materialize compact runtime descriptor tables");
                return false;
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
        if (!d_hidden && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] input upload required during graph capture");
            return false;
        }
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    device,
                    stream,
                    "input"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        if (!d_hidden || (floating && (!d_prefill_gate_ || !d_prefill_up_)))
            return false;

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "moe_output";
        if (!publication_output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                      << publication_output_name
                      << " allocation required during graph capture");
            return false;
        }
        if (!requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name))
        {
            return false;
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        const float **d_down_gate_ptrs = nullptr;
        const float **d_down_up_ptrs = nullptr;
        if (floating)
        {
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

            if (!ensureRuntimeGateUpPointerArrays(gateup_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                                                  top_k, gate_ptrs, up_ptrs,
                                                  &d_gate_ptrs, &d_up_ptrs))
                return false;
            if (!ensureRuntimeDownPointerArrays(down_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                                                top_k, const_gate_ptrs, const_up_ptrs,
                                                &d_down_gate_ptrs, &d_down_up_ptrs))
                return false;
        }

        float *d_output = canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;
        if (!d_expert_ids || !d_weights ||
            (!d_output && !d_canonical_route_contributions))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        const bool reuse_router_q8_hidden =
            !floating && allow_router_q8_reuse &&
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }
        const bool gateup_ok = floating
            ? cudaMoE_grouped_gate_up_floating_decode_table(
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
            : cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart(
                  d_hidden,
                  decode_gate_descs,
                  decode_up_descs,
                  d_expert_ids,
                  nullptr,
                  nullptr,
                  d_decode_hidden_int8_,
                  d_decode_hidden_scales_,
                  reuse_router_q8_hidden,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  top_k,
                  intermediate,
                  d_model,
                  gateup_table.num_experts,
                  gateup_table.codebook_id,
                  gateup_k_partitions,
                  device_ordinal_,
                  stream,
                  {d_decode_swiglu_int8_, d_decode_swiglu_scales_});
        if (!gateup_ok)
            return false;

        const bool down_ok = floating
            ? cudaMoE_grouped_swiglu_down_floating_decode_table(
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
            : cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart(
                  nullptr,
                  nullptr,
                  decode_down_descs,
                  d_expert_ids,
                  cohort_weights ? nullptr : d_weights,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_grouped_down_partials_,
                  d_output,
                  d_canonical_route_contributions,
                  top_k,
                  d_model,
                  intermediate,
                  down_table.num_experts,
                  down_table.codebook_id,
                  down_k_partitions,
                  device_ordinal_,
                  stream,
                  CUDAMoEDecodeDownInput::PublishedSwiGLU);
        if (!down_ok)
            return false;

        markDeviceWritten(
            canonical_route_contributions
                ? canonical_route_contributions
                : output,
            device,
            stream);
        recordGroupedDecodeCounter(
            "cuda_moe_grouped_decode_fused_calls",
            counter_source,
            top_k, d_model, intermediate,
            floating ? "floating_fixed_tree" : "fused_block_down");
        if (!floating)
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_swiglu_publication_calls",
                counter_source, top_k, d_model, intermediate, "ordered_gate_reducer");
        return true;
    }

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        if (!runtime_layer || !input || table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.num_experts <= 0 || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
            return false;

        // Ordered K-part decode is the production contract. If its partial
        // scratch cannot be sized, fail instead of changing arithmetic.
        const int k_partitions = CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if (!ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] "
                      "mandatory K-part gate/up scratch allocation failed");
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (capture_active)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] input upload required during graph capture");
                return false;
            }
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    device,
                    stream,
                    "input"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] null output tensor for slot "
                          << slot);
                return false;
            }
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                capture_active)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] output allocation required during graph capture");
                return false;
            }
            if (!requireOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !requireOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
                return false;

            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] missing output device pointer for slot "
                          << slot);
                return false;
            }
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                                              top_k, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] missing runtime/input device pointer");
            return false;
        }

        const bool reuse_router_q8_hidden =
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }

        const bool ok = cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart(
            d_hidden,
            runtime_layer,
            d_expert_ids,
            d_gate_ptrs,
            d_up_ptrs,
            d_decode_hidden_int8_,
            d_decode_hidden_scales_,
            reuse_router_q8_hidden,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            k_partitions,
            device_ordinal_,
            stream);

        if (ok)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                markDeviceWritten(gate_outputs[slot], device, stream);
                markDeviceWritten(up_outputs[slot], device, stream);
            }
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_gateup_calls", "runtime", top_k,
                d_model, intermediate, "kpart");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromRuntime(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        DeviceMoELayerRuntime *runtime_layer,
        int table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        if (!gate_tensors || !up_tensors || !runtime_layer || !output ||
            table_id < 0 || top_k <= 0 || d_model <= 0 || intermediate <= 0)
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromRuntime");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!ensureGroupedDownDecodeCapacity(top_k, intermediate))
            return false;
        if (!setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromRuntime"))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] null gate/up tensor for slot "
                          << slot);
                return false;
            }
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] missing gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] output allocation required during graph capture");
            return false;
        }
        if (!requireOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                                            top_k, gate_ptrs, up_ptrs,
                                            &d_gate_ptrs, &d_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        // Split-K raises occupancy while its ordered reduction preserves the
        // serial-decode arithmetic tree used by grouped verifier batches.
        const int k_partitions = CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if (!ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] "
                      "mandatory K-part down scratch allocation failed");
            return false;
        }

        const bool ok = cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart(
            d_gate_ptrs,
            d_up_ptrs,
            runtime_layer,
            d_expert_ids,
            d_weights,
            d_decode_swiglu_int8_,
            d_decode_swiglu_scales_,
            d_grouped_down_partials_,
            d_output,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            k_partitions,
            device_ordinal_,
            stream);

        if (ok)
        {
            markDeviceWritten(output, device, stream);
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_down_calls", "runtime", top_k,
                d_model, intermediate, "kpart");
        }
        return ok;
    }

} // namespace llaminar2
