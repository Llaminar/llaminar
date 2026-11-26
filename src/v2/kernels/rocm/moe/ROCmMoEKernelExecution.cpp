/**
 * @file ROCmMoEKernelExecution.cpp
 * @brief ROCmMoEKernel grouped decode, grouped prefill, last-producer bindings, and retained resident HC/FFN execution.
 */
#include "ROCmMoEKernelInternal.h"

namespace llaminar2
{

    bool ROCmMoEKernel::updateGroupedPrefillExpertMask(
        const uint8_t *expert_mask,
        int num_experts)
    {
        if (num_experts <= 0 || !expert_mask)
            return false;

        if (!setMoEDevice(device_ordinal_, "updateGroupedPrefillExpertMask"))
            return false;
        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                      "an explicit stream is required");
            return false;
        }

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
         * Warmup publishes immutable participant ownership once. Graph
         * construction may reuse that exact device table, but any attempt to
         * change or allocate it while capture is active remains fatal.
         */
        if (group_expert_mask_published_ &&
            group_expert_mask_hash_ == mask_hash &&
            group_expert_mask_num_experts_ == num_experts)
            return true;

        if (isGraphCaptureActive() || isHipStreamCapturing(stream))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                      "host expert-mask publication is forbidden during graph capture");
            return false;
        }

        if (!d_group_expert_mask_ || group_expert_mask_cap_ < num_experts)
        {
            if (!workspace_)
            {
                LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                          "graph-owned workspace is required before mask publication");
                return false;
            }

            auto lease = workspace_->acquirePersistentSlot(
                kROCmGroupedExpertMaskLeaseDomain,
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
                LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
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

        hipError_t err = hipMemcpyAsync(
            d_group_expert_mask_,
            expert_mask,
            static_cast<size_t>(num_experts) * sizeof(uint8_t),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] H2D mask copy failed: "
                      << hipGetErrorString(err));
            return false;
        }

        group_expert_mask_hash_ = mask_hash;
        group_expert_mask_num_experts_ = num_experts;
        group_expert_mask_active_experts_ = active_experts;
        group_expert_mask_published_ = true;
        return true;
    }

    bool ROCmMoEKernel::prepareSharedExpertPrefillGroup(DeviceRowRange rows, MoESharedExpertGroupKind kind)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginSharedExpert(kind, rows);
        const int seq_len = rows.physicalRows();
        if (seq_len <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "prepareSharedExpertPrefillGroup"))
            return false;

        if (seq_len > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(seq_len) * sizeof(float),
                                     "prepareSharedExpertPrefillGroup(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = seq_len;
        }
        if (group_experts_cap_ < 1)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = 1;
        }

        if (!hipMoE_prepare_shared_expert_group(
                d_group_offsets_,
                d_group_counts_,
                d_group_token_indices_,
                d_group_original_to_grouped_,
                d_group_weights_,
                d_group_active_expert_ids_,
                rows,
                device_ordinal_,
                getStream()))
        {
            return false;
        }

        group_active_expert_slots_ = 1;
        if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_shared_expert_prefill_group_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", "1"},
                    {"active_expert_slots", "1"}});
        }
        std::move(grouping_publication).publish();
        return true;
    }

    /**
     * @brief Resolve preallocated grouped scratch without changing arena size.
     * @param total_slots Physical routed-row capacity of the captured graph.
     * @param d_model Hidden projection width, including floating-point tails.
     * @param intermediate Expert intermediate width.
     * @return True when every required named arena region has sufficient space.
     */
    bool ROCmMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        if (!setMoEDevice(device_ordinal_, "ensureGroupedPrefillScratchCapacity"))
            return false;

        const bool need_realloc = (total_slots > prefill_slots_cap_ ||
                                   d_model > prefill_d_model_cap_ ||
                                   intermediate > prefill_intermediate_cap_);
        if (!need_realloc &&
            d_prefill_A_int8_ &&
            d_prefill_A_scales_ &&
            d_prefill_swiglu_int8_ &&
            d_prefill_swiglu_scales_ &&
            d_prefill_gate_ &&
            d_prefill_up_)
        {
            return true;
        }

        const int max_dim = (d_model > intermediate) ? d_model : intermediate;
        // Match the workspace BOM and CUDA binding. Floating matrices need not
        // be multiples of a quantized block; flooring here can request a zero
        // length binding even though the arena already owns the padded region.
        const int max_blocks = (max_dim + 31) / 32;
        const int intermediate_blocks = (intermediate + 31) / 32;

        // Hidden-row Q8 data remains live through gate/up, while the exact
        // SwiGLU Q8 rows remain live through down projection. Those lifetimes
        // overlap across CTAs, so they require distinct device buffers. Once
        // SwiGLU has consumed the FP32 gate rows, long-prefill down projection
        // reuses the larger gate allocation for deterministic route partials;
        // the ordered publisher then folds those rows in original top-k order.
        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_int8_),
                                 MoEWorkspaceBuffers::PREFILL_A_INT8,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t),
                                 "ensureGroupedPrefillScratchCapacity(a_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_scales_),
                                 MoEWorkspaceBuffers::PREFILL_A_SCALES,
                                 static_cast<size_t>(total_slots) * max_blocks * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(a_scales)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_swiglu_int8_),
                                 MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,
                                 static_cast<size_t>(total_slots) * intermediate * sizeof(int8_t),
                                 "ensureGroupedPrefillScratchCapacity(swiglu_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_swiglu_scales_),
                                 MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                                 static_cast<size_t>(total_slots) * intermediate_blocks * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(swiglu_scales)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_gate_),
                                 MoEWorkspaceBuffers::PREFILL_GATE,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(gate)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_up_),
                                 MoEWorkspaceBuffers::PREFILL_UP,
                                 static_cast<size_t>(total_slots) * intermediate * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(up)"))
        {
            d_prefill_A_int8_ = nullptr;
            d_prefill_A_scales_ = nullptr;
            d_prefill_swiglu_int8_ = nullptr;
            d_prefill_swiglu_scales_ = nullptr;
            d_prefill_gate_ = nullptr;
            d_prefill_up_ = nullptr;
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

        prefill_slots_cap_ = total_slots;
        prefill_d_model_cap_ = d_model;
        prefill_intermediate_cap_ = intermediate;
        return true;
    }

    /**
     * @brief Borrow the current original grouping banks and exact named workspace extent.
     *
     * The directory is not interchangeable with the compact active-expert-id
     * list. The combined producer retains raw IDs separately and stores packed
     * descriptors into original exclusive intervals. Successful binding does
     * not publish authority; only the complete grouping preparation can do so.
     */
    MoEGroupingPackedDirectory ROCmMoEKernel::groupingDirectoryBinding(int rows, int top_k, int experts,
        rocm::MoEGroupedDirectorySource source) const
    {
        if (source != rocm::MoEGroupedDirectorySource::GroupingOffsetSlots &&
            source != rocm::MoEGroupedDirectorySource::GroupingColumnSlots)
            throw std::invalid_argument("packed grouping binding requires a resolved packed directory producer");
        const auto factory=source==rocm::MoEGroupedDirectorySource::GroupingColumnSlots
            ? MoEGroupingPackedDirectory::columnFloor8 : MoEGroupingPackedDirectory::wideFloor8;
        return factory(d_group_counts_, d_group_offsets_, d_group_active_expert_ids_,
            d_prefill_work_directory_, workspace_ ? workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY) / sizeof(uint32_t) : 0,
            rows, top_k, experts, DeviceId::rocm(device_ordinal_), workspace_, ExplicitGPUStream(getStream()));
    }

    rocm::MoEGroupedPopulationKey ROCmMoEKernel::originalRoutedPopulationKey() const
    {
        const auto *original = ScopedMoERoutedTransaction::current();
        if (!original) return {};
        if (original->owner != this || original->stream != getStream() || original->workspace != workspace_)
            throw std::logic_error("original routed transaction changed kernel, stream or workspace");
        // Unmeasured geometries keep their ordinary first-class policy. Only
        // immutable descriptor metadata is inspected; live routes stay on device.
        if (original->contributions || original->rows != 16 || original->hidden_width != 2048 ||
            original->intermediate != 512 || original->experts != 256 || original->top_k != 8 ||
            original->gateup_table >= int(grouped_gateup_desc_tables_.size()) ||
            original->down_table >= int(grouped_down_desc_tables_.size())) return {};
        const auto &gu = grouped_gateup_desc_tables_[original->gateup_table];
        const auto &down = grouped_down_desc_tables_[original->down_table];
        if (!gu.valid || !down.valid || !gu.deviceReady() || !down.deviceReady() ||
            gu.descriptor_source != MoEDecodeDescriptorSource::StaticDescriptorTable ||
            down.descriptor_source != MoEDecodeDescriptorSource::StaticDescriptorTable ||
            deviceMoEWeightFormatIsFloating(gu.weight_format) || gu.weight_format != down.weight_format ||
            gu.num_experts != 256 || down.num_experts != 256 || gu.d_model != 2048 || down.d_model != 2048 ||
            gu.intermediate != 512 || down.intermediate != 512 ||
            !canReuseRouterQ8Hidden(static_cast<const float *>(original->hidden->gpu_data_ptr()), 16, 2048)) return {};
        return {.operation = MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
            .input = rocm::MoEGroupedPopulationInput::OriginalTokens,
            .preparation = rocm::MoEGroupedPopulationPreparation::PublishedQ8,
            .rows = 16, .hidden = 2048, .intermediate = 512, .experts = 256, .top_k = 8,
            .gateup_physical = gu.codebook_mask, .down_physical = down.codebook_mask,
            .gateup_source = gu.policy_codebook_mask, .down_source = down.policy_codebook_mask,
            .routed_sources = rocm::routedVerifierOriginalSources(gu.host_gate_descs, gu.host_up_descs, down.host_descs)};
    }

    /** @brief Bind the existing named adaptive directory and population banks before capture. */
    bool ROCmMoEKernel::ensureGroupedPrefillWorkDirectoryCapacity(
        int total_slots,
        int num_experts)
    {
        if (total_slots <= 0 || num_experts <= 0)
            return false;
        const std::size_t words =
            MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(
                static_cast<std::size_t>(total_slots),
                static_cast<std::size_t>(num_experts));
        if (!bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_prefill_work_directory_),
                MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY,
                words * sizeof(uint32_t),
                "ensureGroupedPrefillWorkDirectoryCapacity") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_population_selection_),
                MoEWorkspaceBuffers::ROCM_GROUPED_POPULATION_SELECTION, sizeof(uint32_t),
                "ensureGroupedPrefillWorkDirectoryCapacity(population)"))
        {
            d_prefill_work_directory_ = nullptr;
            d_grouped_population_selection_ = nullptr;
            return false;
        }
        return true;
    }

    bool ROCmMoEKernel::exportGroupedPrefillIntermediates(
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
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "packet", "exportGroupedPrefillIntermediates"))
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
        if (!rocm::packGroupedIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::importGroupedPrefillIntermediates(
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
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packets, device, stream, "packets", "importGroupedPrefillIntermediates"))
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
        return rocm::consumeGroupedIntermediate(launch, stream);
    }

    bool ROCmMoEKernel::exportCompactGroupedPrefillIntermediates(
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
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "packet", "exportCompactGroupedPrefillIntermediates")) return false;
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
        if (!rocm::packCompactIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::importCompactGroupedPrefillIntermediates(
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
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packet, device, stream, "packet", "importCompactGroupedPrefillIntermediates")) return false;
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
        return rocm::consumeCompactIntermediate(launch, stream);
    }

    bool ROCmMoEKernel::executeGroupedPrefillProjection(
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

        if (const auto *original = ScopedMoERoutedTransaction::current())
        {
            auto actual = *original;
            actual.owner = this; actual.hidden = hidden; actual.output = output;
            actual.contributions = canonical_route_contributions;
            actual.gateup_table = gateup_desc_table_id; actual.down_table = down_desc_table_id;
            actual.rows = seq_len; actual.hidden_width = d_model; actual.intermediate = intermediate;
            actual.experts = num_experts; actual.top_k = top_k;
            actual.stream = getStream(); actual.workspace = workspace_;
            ScopedMoERoutedTransaction::requireMatching(actual);
            if (!execution.executesGateUp() || !execution.executesDown() ||
                execution.firstColumn() != 0 || execution.columnCount() != d_model)
                throw std::logic_error("original routed transaction cannot lend its identity to a projection phase");
        }
        const auto original_routed_key = originalRoutedPopulationKey();
        const auto directory_source = rocm::selectedMoEGroupedPrefillPolicy(device_ordinal_, original_routed_key).directory;
        const bool grouping_offset_slots = directory_source != rocm::MoEGroupedDirectorySource::RolePlanner;
        if (!grouping_offset_slots) grouped_prefill_grouping_.revokeDirectory();
        const auto *owner_admission = ownerGateUpAdmissionForCall(gateup_desc_table_id, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model});
        rocm::MoERouteOwnedGateUpLaunchSet owner_completed{};
        rocm::validateMoERouteOwnedGateUpScopeBinding(device_ordinal_, gpu_stream_, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model}, owner_admission);
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
            return false;

        if (!setMoEDevice(device_ordinal_, "executeGroupedPrefillProjection"))
            return false;

        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] invalid gateup descriptor table id "
                      << gateup_desc_table_id);
            return false;
        }
        if (down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] invalid down descriptor table id "
                      << down_desc_table_id);
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        rocm::validateMoERouteOwnedGateUpCodebook(gateup_table.codebook_mask, gateup_table.policy_codebook_mask, owner_admission);
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (grouping_offset_slots && floating)
            throw std::invalid_argument("grouping offset-slot candidate requires native quantised descriptors");

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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] descriptor tables not valid");
            return false;
        }

        const int total_slots = seq_len * top_k;
        const int active_expert_slots = group_active_expert_slots_;

        // Ensure scratch buffers
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate))
            return false;
        if (!ensureGroupedPrefillWorkDirectoryCapacity(
                total_slots, num_experts))
            return false;

        // Join the hidden-state producer and the exact publication target to
        // this pipeline. Canonical LocalTP publication does not own or inspect
        // the later reducer destination.
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if ((execution.executesGateUp() && !requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillProjection")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillProjection")))
        {
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

        if (isGraphCaptureActive() &&
            ((execution.executesGateUp() && !d_hidden) ||
             (execution.executesDown() && !d_output && !d_canonical_route_contributions)))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] null device pointers during graph capture");
            return false;
        }

        const bool shared_singleton_expert = (num_experts == 1 && top_k == 1);
        const int *d_original_expert_ids_for_pipeline =
            shared_singleton_expert ? nullptr : d_group_int_indices_;
        if (floating)
        {
            if (!d_group_original_to_grouped_ || !d_group_weights_ ||
                (!shared_singleton_expert &&
                 !d_original_expert_ids_for_pipeline))
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped-prefill route publication is incomplete");
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
                    return rocmMoE_grouped_floating_prefill_pipeline(
                        d_hidden,
                        gateup_table.device_floating_gate_descs,
                        gateup_table.device_floating_up_descs,
                        down_table.device_floating_descs,
                        d_group_original_to_grouped_,
                        d_original_expert_ids_for_pipeline,
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
                        producer_stream,
                        execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_floating_calls",
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

        const bool reuse_router_q8_hidden = execution.executesGateUp() &&
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        if (execution.executesGateUp() && router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::RequiredConsumer &&
            !reuse_router_q8_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] required "
                      "router Q8 publication is unavailable"
                      << " rows=" << seq_len
                      << " d_model=" << d_model);
            return false;
        }
        /*
         * Ordered publication is the sole grouped-prefill contract. One lane
         * owns each token/column and accumulates routes in top-k order. Small
         * verifier buckets compute each route directly; long prefill first
         * materializes deterministic route partials so decoded weights can be
         * reused across expert-row tiles. Neither variant relies on a
         * scheduler-dependent floating-point atomic reduction.
         */
        const bool ordered_scatter_overwrites_output =
            active_expert_slots > 0 && d_group_original_to_grouped_ != nullptr;
        if (!ordered_scatter_overwrites_output ||
            (!shared_singleton_expert && !d_group_int_indices_))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                      "batch-invariant grouped prefill requires the original route map and expert ids");
            return false;
        }
        // Launch the fixed, fully grouped pipeline without a host synchronization.
        // These retained descriptors are immutable preparation metadata. Runtime
        // placement tables cannot provide a host source authority for selection.
        rocm::MoEGroupedSourceTuple singleton_sources;
        if (num_experts == 1 && gateup_table.descriptor_source == MoEDecodeDescriptorSource::StaticDescriptorTable &&
            down_table.descriptor_source == MoEDecodeDescriptorSource::StaticDescriptorTable &&
            gateup_table.host_gate_descs.size() == 1 && gateup_table.host_up_descs.size() == 1 &&
            down_table.host_descs.size() == 1)
        {
            const auto source = [](const DeviceNativeQuantisedMatrixDesc &descriptor) {
                return NativeQuantisedSourceIdentity{descriptor.source_codebook_id,
                    descriptor.source_is_superblock != 0, descriptor.source_identity_present != 0};
            };
            singleton_sources = {source(gateup_table.host_gate_descs.front()),
                source(gateup_table.host_up_descs.front()), source(down_table.host_descs.front())};
        }
        rocm::MoEGroupedPrefillSubmission grouped_submitted;
        const auto grouping_directory = grouped_prefill_grouping_.borrowDirectory();
        if (grouping_offset_slots)
            (void)grouping_directory.requireMatching(groupingDirectoryBinding(seq_len, top_k, num_experts, directory_source));
        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                return rocmMoE_grouped_prefill_pipeline(
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
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_original_expert_ids_for_pipeline,
                    d_group_weights_,
                    reinterpret_cast<int *>(d_prefill_work_directory_),
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    total_slots,
                    top_k,
                    0,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    gateup_table.policy_codebook_mask,
                    down_table.policy_codebook_mask,
                    device_ordinal_,
                    producer_stream,
                    execution,
                    grouped_prefill_grouping_.scopeForProjection(seq_len, top_k), owner_admission,
                    owner_admission ? &owner_completed : nullptr, d_grouped_population_selection_, &grouped_submitted,
                    singleton_sources, grouping_offset_slots ? &grouping_directory : nullptr, original_routed_key.routed_sources);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });

        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] pipeline failed for layer");
            return false;
        }

        if (owner_admission)
            recordOwnerGateUpCaptureSubmissions(owner_completed, gateup_table.owner_m1_intent);

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", "static_table"}});
        }

            if (PerfStatsCollector::isDomainEnabled("kernel") &&
                active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                submittedMoEPrefillPairPolicyTags(grouped_submitted);
            const std::string common_row_tile =
                policy.gateup.tile_m == policy.down.tile_m
                    ? policy.gateup.tile_m
                    : "mixed";
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_batch_invariant_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", std::to_string(top_k)},
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"gateup_codebook_mask",
                     codebookMaskTag(gateup_table.codebook_mask)},
                    {"down_codebook_mask",
                     codebookMaskTag(down_table.codebook_mask)},
                    {"gateup_policy_codebook_mask",
                     codebookMaskTag(gateup_table.policy_codebook_mask)},
                    {"down_policy_codebook_mask",
                     codebookMaskTag(down_table.policy_codebook_mask)},
                    {"gateup_route", groupedRouteRoster(grouped_submitted.gateup,grouped_submitted.gateup_route_owned)},
                    {"down_route", groupedRouteRoster(grouped_submitted.down,grouped_submitted.down_route_owned)},
                    {"hidden_preparation", !execution.executesGateUp() ? "not_submitted" :
                        reuse_router_q8_hidden ? "router_q8" : "original_row_quant"},
                    {"dot_policy", groupedEngineTag(grouped_submitted.policy.engine)},
                    {"directory_source", grouped_submitted.policy.directory == rocm::MoEGroupedDirectorySource::GroupingOffsetSlots
                        ? "grouping_offset_slots" : grouped_submitted.policy.directory == rocm::MoEGroupedDirectorySource::GroupingColumnSlots
                        ? "grouping_column_slots" : "role_planner"},
                    {"gateup_submitted_tiles", groupedTileRoster(grouped_submitted.gateup)},
                    {"down_submitted_tiles", groupedTileRoster(grouped_submitted.down)},
                    {"gateup_submitted_tiled_leaves", std::to_string(grouped_submitted.gateup.leaves())},
                    {"down_submitted_tiled_leaves", std::to_string(grouped_submitted.down.leaves())},
                    {"counter_semantics", "host_submissions"},
                    {"gateup_tile_m", policy.gateup.tile_m},
                    {"gateup_tile_n", policy.gateup.tile_n},
                    {"down_tile_m", policy.down.tile_m},
                    {"down_tile_n", policy.down.tile_n},
                    {"policy_source", policy.source},
                    {"row_tile", common_row_tile},
                    // These tags describe the captured envelope, not a host
                    // observation of mutable device-owned live row counts.
                    {"geometry_role", "physical_capacity"},
                    {"route_admission", "device_group_plan"},
                    {"grouping", "static"}});
        }
        return true;
    }

    bool ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan(
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
        return executeGroupedRuntimeProjection(device_runtime_layer,runtime_host_layer,hidden,output,
            gateup_desc_table_id,down_desc_table_id,seq_len,d_model,intermediate,num_experts,top_k,execution,
            canonical_route_contributions,nullptr);
    }

    /** @copydoc ROCmMoEKernel::executeGroupedFFNFrontierFromPublishedRuntimePlan */
    bool ROCmMoEKernel::executeGroupedFFNFrontierFromPublishedRuntimePlan(DeviceMoELayerRuntime* runtime,
        const DeviceMoELayerRuntime& host,ITensor* hidden,int gate_id,int down_id,ITensor* route_partials,
        const rocm::grouped_ffn::LocalPublication& publication)
    {
        namespace geometry=rocm::grouped_ffn;
        return executeGroupedRuntimeProjection(runtime,host,hidden,nullptr,gate_id,down_id,
            publication.rows.physicalRows(),geometry::hidden,geometry::routedColumns(publication.program),geometry::experts,geometry::topK,
            MoEPrefillProjectionExecution::intermediateShard(geometry::hidden),route_partials,&publication);
    }

    bool ROCmMoEKernel::executeGroupedRuntimeProjection(
        DeviceMoELayerRuntime *device_runtime_layer,
        const DeviceMoELayerRuntime &runtime_host_layer,
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        const MoEPrefillProjectionExecution &execution,
        ITensor *canonical_route_contributions,
        const rocm::grouped_ffn::LocalPublication* frontier)
    {
        if (execution.isIntermediateShard() != runtime_host_layer.intermediate_shard.enabled() ||
            (execution.isIntermediateShard() &&
             (runtime_host_layer.intermediate_shard.modelColumns() != static_cast<uint32_t>(d_model) ||
              runtime_host_layer.intermediate_shard.localIntermediateColumns() != static_cast<uint32_t>(intermediate) ||
              !canonical_route_contributions)))
            return false;

        const auto *owner_admission = ownerGateUpAdmissionForCall(gateup_desc_table_id, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model});
        rocm::MoERouteOwnedGateUpLaunchSet owner_completed{};
        rocm::validateMoERouteOwnedGateUpScopeBinding(device_ordinal_, gpu_stream_, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model}, owner_admission);
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
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
        {
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan"))
            return false;
        if (runtime_host_layer.expert_count != static_cast<uint32_t>(num_experts) ||
            runtime_host_layer.top_k != static_cast<uint32_t>(top_k) ||
            runtime_host_layer.prefill_token_capacity < static_cast<uint32_t>(seq_len) ||
            runtime_host_layer.prefill_route_capacity < static_cast<uint32_t>(seq_len * top_k) ||
            !runtime_host_layer.expert_counts ||
            !runtime_host_layer.expert_offsets ||
            !runtime_host_layer.grouped_token_ids ||
            !runtime_host_layer.grouped_route_weights ||
            !runtime_host_layer.route_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid runtime scratch contract"
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()))
            return false;

        rocm::validateMoERouteOwnedGateUpCodebook(gateup_table.codebook_mask, gateup_table.policy_codebook_mask, owner_admission);
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] descriptor table shape mismatch");
            return false;
        }

        const int total_slots = seq_len * top_k;
        if (!ensureGroupedPrefillScratchCapacity(
                total_slots, d_model, intermediate) ||
            !ensureGroupedPrefillWorkDirectoryCapacity(
                total_slots, num_experts))
            return false;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();

        /*
         * Resolve both fused-publication outputs before launch.  These are
         * persistent workspace addresses selected by graph identity; binding
         * performs validation and pointer arithmetic only, never allocation.
         */
        if (total_slots > group_slots_cap_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "executeGroupedPrefillProjectionFromPublishedRuntimePlan(group_original_to_grouped)"))
            {
                d_group_original_to_grouped_ = nullptr;
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] runtime grouping workspace is required");
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        const int active_expert_slots = std::min(total_slots, num_experts);

        /*
         * Complete plan publication has already written these graph-owned
         * descriptor slots and the ordered inverse map. The compute consumer
         * resolves the same persistent addresses without hidden publication
         * kernels of its own.
         */
        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm runtime prefill down descriptors"))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "failed to bind graph-owned runtime descriptor slots");
            return false;
        }

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if ((execution.executesGateUp() && !requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan")))
        {
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
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] null device pointers");
            return false;
        }

        group_active_expert_slots_ = active_expert_slots;

        if(frontier && floating)
            throw std::logic_error("Retained grouped FFN lost its original native quantised source");

        if (floating)
        {
            const bool ok = execution.overwriteOutputThenPublish(
                execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
                execution.executesDown() ? publication_output->size_bytes() : 0u,
                seq_len, top_k,
                canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                              : MoEPrefillOutputLayout::TokenRows,
                stream,
                [&](std::span<float> live_output, void *producer_stream)
                {
                    return rocmMoE_grouped_floating_prefill_pipeline(
                        d_hidden,
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
                        producer_stream,
                        execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_floating_calls",
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

        const bool reuse_router_q8_hidden = execution.executesGateUp() &&
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        if(frontier) {
            namespace program=rocm::grouped_ffn;
            const auto [gate_format,down_format]=program::routedFormats(frontier->program);
            if(!reuse_router_q8_hidden || owner_admission || !execution.isIntermediateShard() ||
               gateup_table.codebook_mask!=gate_format.execution_codebook_mask ||
               gateup_table.policy_codebook_mask!=gate_format.policy_codebook_mask ||
               down_table.codebook_mask!=down_format.execution_codebook_mask ||
               down_table.policy_codebook_mask!=down_format.policy_codebook_mask ||
               frontier->rows.physicalRows()!=seq_len || !frontier->rows.countOwner() ||
               !workspace_ || !d_prefill_A_int8_ || !d_prefill_A_scales_ ||
               d_prefill_A_int8_==d_router_q8_hidden_ || d_prefill_A_scales_==d_router_q8_hidden_scales_ ||
               workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_SHARED_GATE)<std::size_t(seq_len)*sizeof(float))
                throw std::logic_error("Retained grouped FFN requires its original Q8/router descriptor contract:"
                    " reuse_q8="+std::to_string(reuse_router_q8_hidden)+
                    " gateup_owned="+std::to_string(owner_admission!=nullptr)+
                    " intermediate_shard="+std::to_string(execution.isIntermediateShard())+
                    " gate_mask="+std::to_string(gateup_table.codebook_mask)+
                    "/"+std::to_string(gate_format.execution_codebook_mask)+
                    " gate_policy="+std::to_string(gateup_table.policy_codebook_mask)+
                    "/"+std::to_string(gate_format.policy_codebook_mask)+
                    " down_mask="+std::to_string(down_table.codebook_mask)+
                    "/"+std::to_string(down_format.execution_codebook_mask)+
                    " down_policy="+std::to_string(down_table.policy_codebook_mask)+
                    "/"+std::to_string(down_format.policy_codebook_mask)+
                    " rows="+std::to_string(frontier->rows.physicalRows())+
                    "/"+std::to_string(seq_len)+
                    " publication="+std::to_string(router_q8_hidden_publication_?router_q8_hidden_publication_->published_rows:0)+
                    " shared_gate_bytes="+std::to_string(workspace_?workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_SHARED_GATE):0));
            if(!requireTensorOnDevice(frontier->sharedInputGate,device,stream,"shared input gate","grouped FFN frontier"))
                return false;
            const std::array outputs{frontier->sharedGateScratch,frontier->sharedUpScratch,frontier->routedOutput,
                frontier->sharedOutput,frontier->combined};
            for(auto* result:outputs)
                if(!requireOutputOnDevice(result,device,stream,"original local FFN output","grouped FFN frontier"))
                    return false;
            const program::RoutePlan plan{frontier->rows.countOwner(),d_group_original_to_grouped_,
                runtime_host_layer.route_expert_ids,runtime_host_layer.grouped_route_weights,d_router_q8_hidden_,
                d_router_q8_hidden_scales_,d_hidden,runtime_gate_descs,runtime_up_descs,runtime_down_descs};
            const auto address=[](TensorBase* tensor){return static_cast<float*>(tensor->gpu_data_ptr());};
            const program::Output publication{d_prefill_swiglu_int8_,d_prefill_swiglu_scales_,
                d_prefill_A_int8_,d_prefill_A_scales_,
                static_cast<float*>(workspace_->getBuffer(MoEWorkspaceBuffers::ROCM_SHARED_GATE)),
                address(frontier->sharedInputGate),
                address(frontier->sharedGateScratch),address(frontier->sharedUpScratch),address(frontier->routedOutput),
                address(frontier->sharedOutput),address(frontier->combined),d_canonical_route_contributions};
            program::DirectBankInput direct;
            if(frontier->directRouter.indices) {
                const auto& router=frontier->directRouter;
                if((router.demand!=MoEGroupedPlanDemand::None && router.demand!=MoEGroupedPlanDemand::DeferredAcceptedRows) ||
                   !requireTensorOnDevice(router.indices,device,stream,"original router indices","grouped FFN direct bank") ||
                   !requireTensorOnDevice(router.weights,device,stream,"original router weights","grouped FFN direct bank") ||
                   !d_prefill_work_directory_ ||
                   workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY)<program::bankIdentityBytes)
                    return false;
                direct={device_runtime_layer,static_cast<const float*>(router.indices->gpu_data_ptr()),
                    static_cast<const float*>(router.weights->gpu_data_ptr()),router.demand,
                    reinterpret_cast<std::uint64_t*>(d_prefill_work_directory_)};
                if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumerWithExactK32Sums) {
                    const auto* raw=workspace_->getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8);
                    if(!program::usesExactK32Sums(frontier->program) || raw!=d_prefill_A_int8_ ||
                       workspace_->getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8)<program::k32SumWorkspaceBytes)
                        throw std::logic_error("Exact K32 sum consumer lost the original router workspace");
                    direct.affineSums=program::exactK32Sums(d_prefill_A_int8_);
                }
            }
            const auto* formats=std::get_if<program::SourceFormatProgram>(&frontier->program);
            if(formats) {
                if(!frontier->boundSourceProgram ||
                   !program::launchFormatProgram(plan,frontier->rows,frontier->sharedWeights,publication,device_ordinal_,stream,
                        direct,*frontier->boundSourceProgram))
                    return false;
            } else {
                const auto& measured=std::get<program::MeasuredProgram>(frontier->program);
                if(!program::launch(plan,frontier->sharedWeights,publication,device_ordinal_,stream,direct,
                        measured.geometry,measured.decoder,
                        frontier->boundSourceProgram?&*frontier->boundSourceProgram:nullptr))
                    return false;
            }
            markDeviceWritten(publication_output,device,stream);
            for(auto* result:outputs)markDeviceWritten(result,device,stream);
            PerfStatsCollector::addCounter("kernel","rocm_grouped_ffn_frontier_calls",1.0,"moe",device.toString(),
                {{"physical_rows",std::to_string(seq_len)},{"routed_width",std::to_string(intermediate)},
                 {"shared_width",std::to_string(program::sharedColumns(frontier->program))},
                 {"verifier_dispatch_table",formats?native_quantised::verifierDispatchTableName(
                     frontier->boundSourceProgram->dispatchTable()):"fixed"},
                 {"program",formats?"all_formats":"fixed"},{"decoder_family",formats?std::to_string(formats->family()):"1"}});
            return true;
        }
        rocm::MoEGroupedPrefillSubmission grouped_submitted;
        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                return rocmMoE_grouped_prefill_pipeline(
                    d_hidden,
                    reuse_router_q8_hidden ? d_router_q8_hidden_ : nullptr,
                    reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : nullptr,
                    runtime_gate_descs,
                    runtime_up_descs,
                    runtime_down_descs,
                    runtime_host_layer.expert_counts,
                    runtime_host_layer.expert_offsets,
                    runtime_host_layer.grouped_token_ids,
                    d_group_original_to_grouped_,
                    runtime_host_layer.route_expert_ids,
                    runtime_host_layer.grouped_route_weights,
                    reinterpret_cast<int *>(d_prefill_work_directory_),
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    total_slots,
                    top_k,
                    1,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    gateup_table.policy_codebook_mask,
                    down_table.policy_codebook_mask,
                    device_ordinal_,
                    producer_stream,
                    execution,
                    MoEPrefillRouteScope::SelectedRoutes, owner_admission,
                    owner_admission ? &owner_completed : nullptr, d_grouped_population_selection_, &grouped_submitted);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });
        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] grouped ROCm pipeline failed");
            return false;
        }

        if (owner_admission)
            recordOwnerGateUpCaptureSubmissions(owner_completed, gateup_table.owner_m1_intent);

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"}});
        }

            if (PerfStatsCollector::isDomainEnabled("kernel") &&
                active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                submittedMoEPrefillPairPolicyTags(grouped_submitted);
            const std::string common_row_tile =
                policy.gateup.tile_m == policy.down.tile_m
                    ? policy.gateup.tile_m
                    : "mixed";
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_batch_invariant_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", std::to_string(top_k)},
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"gateup_codebook_mask",
                     codebookMaskTag(gateup_table.codebook_mask)},
                    {"down_codebook_mask",
                     codebookMaskTag(down_table.codebook_mask)},
                    {"gateup_policy_codebook_mask",
                     codebookMaskTag(gateup_table.policy_codebook_mask)},
                    {"down_policy_codebook_mask",
                     codebookMaskTag(down_table.policy_codebook_mask)},
                    {"gateup_route", groupedRouteRoster(grouped_submitted.gateup,grouped_submitted.gateup_route_owned)},
                    {"down_route", groupedRouteRoster(grouped_submitted.down,grouped_submitted.down_route_owned)},
                    {"hidden_preparation", !execution.executesGateUp() ? "not_submitted" :
                        reuse_router_q8_hidden ? "router_q8" : "original_row_quant"},
                    {"dot_policy", groupedEngineTag(grouped_submitted.policy.engine)},
                    {"gateup_submitted_tiles", groupedTileRoster(grouped_submitted.gateup)},
                    {"down_submitted_tiles", groupedTileRoster(grouped_submitted.down)},
                    {"gateup_submitted_tiled_leaves", std::to_string(grouped_submitted.gateup.leaves())},
                    {"down_submitted_tiled_leaves", std::to_string(grouped_submitted.down.leaves())},
                    {"counter_semantics", "host_submissions"},
                    {"gateup_tile_m", policy.gateup.tile_m},
                    {"gateup_tile_n", policy.gateup.tile_n},
                    {"down_tile_m", policy.down.tile_m},
                    {"down_tile_n", policy.down.tile_n},
                    {"policy_source", policy.source},
                    {"row_tile", common_row_tile},
                    {"geometry_role", "physical_capacity"},
                    {"route_admission", "device_group_plan"},
                    {"grouping", "runtime"}});
        }
        return true;
    }

    bool ROCmMoEKernel::reduceCanonicalRouteContributions(
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
        if (!setMoEDevice(
                device_ordinal_,
                "reduceCanonicalRouteContributions"))
        {
            return false;
        }

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::reduceCanonicalRouteContributions] "
                      "an explicit stream is required");
            return false;
        }
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(
                canonical_route_contributions,
                device,
                stream,
                "canonical_route_contributions",
                "reduceCanonicalRouteContributions") ||
            !requireOutputOnDevice(
                output,
                device,
                stream,
                "output",
                "reduceCanonicalRouteContributions"))
        {
            return false;
        }

        const float *d_contributions = static_cast<const float *>(
            canonical_route_contributions->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!rocmMoE_reduce_canonical_route_contributions(
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
            "rocm_moe_canonical_route_reduce_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"d_model", std::to_string(d_model)}});
        return true;
    }

    bool ROCmMoEKernel::publishSharedExpertRankBank(
        ITensor *shared_output,
        ITensor *canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *device_effective_seq_len)
    {
        constexpr const char *kContext =
            "publishSharedExpertRankBank";
        if (!shared_output || !canonical_publication ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0 || participant_index < 0 ||
            participant_index >= participant_count)
        {
            return false;
        }

        void *stream = requireStream(
            "ROCmMoEKernel::publishSharedExpertRankBank");
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_elements =
            row_elements *
            (static_cast<size_t>(top_k) +
             static_cast<size_t>(participant_count));
        if (shared_output->numel() < row_elements ||
            canonical_publication->numel() < required_elements ||
            !setMoEDevice(device_ordinal_, kContext) ||
            !requireTensorOnDevice(
                shared_output,
                device,
                stream,
                "shared_output",
                kContext) ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication",
                kContext))
        {
            return false;
        }

        if (!rocmMoE_publish_shared_expert_rank_bank(
                static_cast<const float *>(shared_output->gpu_data_ptr()),
                static_cast<float *>(
                    canonical_publication->gpu_data_ptr()),
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
            "rocm_moe_shared_rank_bank_publish_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

    bool ROCmMoEKernel::finalizeCanonicalMoEPublication(
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
        constexpr const char *kContext =
            "finalizeCanonicalMoEPublication";
        if (!input || !gate_inp || !canonical_publication ||
            !routed_output || !shared_output || !combined_output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0)
        {
            return false;
        }

        void *stream = requireStream(
            "ROCmMoEKernel::finalizeCanonicalMoEPublication");
        const DeviceId device = DeviceId::rocm(device_ordinal_);
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
            !setMoEDevice(device_ordinal_, kContext) ||
            !requireTensorOnDevice(
                input, device, stream, "input", kContext) ||
            !requireTensorOnDevice(
                gate_inp, device, stream, "gate_inp", kContext) ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication",
                kContext) ||
            !requireOutputOnDevice(
                routed_output,
                device,
                stream,
                "routed_output",
                kContext) ||
            !requireOutputOnDevice(
                shared_output,
                device,
                stream,
                "shared_output",
                kContext) ||
            !requireOutputOnDevice(
                combined_output,
                device,
                stream,
                "combined_output",
                kContext))
        {
            return false;
        }

        if (!rocmMoE_finalize_canonical_publication(
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
            "rocm_moe_canonical_publication_finalize_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

namespace
{
/** @brief One retained existing named arena interval, never an allocation authority. */
struct ResidentFFNScratchBinding
{
    const char* name=nullptr;
    void* pointer=nullptr;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::size_t bytes=0;
};
/** @brief Existing immutable singleton table handles and exact captured descriptor leases. */
struct ResidentFFNSharedTables
{
    int gate_up=-1,down=-1;
    std::array<std::shared_ptr<const WorkspaceBufferLease>,3> leases;
};
}
/** @brief Immutable borrowed counter/table/scratch seal; the arena remains allocation authority. */
struct ROCmMoEKernel::GateUpLastProducerBinding
{
    rocm::GateUpProducerLoads loads;
    DeviceWorkspaceManager* workspace;
    std::uint64_t generation;
    void* stream;
    int table,top_k,hidden,intermediate,experts;
    std::uint32_t execution_mask,policy_mask;
    NativeTileCompletionView view;
    std::size_t counter_bytes;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::shared_ptr<GroupedDescriptorWorkspacePublication> publication;
    const void* original_gate;
    const void* original_up;
    std::size_t descriptor_slot;
    MoEIntermediateShardDescriptorDeclaration original;
    struct ScratchSeal
    {
        const char* name;
        const void* pointer;
        std::size_t bytes;
        std::shared_ptr<const WorkspaceBufferLease> lease;
    };
    std::vector<ScratchSeal> scratch;
    std::uint64_t signature;
};

/** @copydoc IMoEKernel::prepareGateUpPublication */
std::uint64_t ROCmMoEKernel::prepareGateUpPublication(int table,int top_k,MoEGateUpPublicationPolicy policy)
{
    if(::llaminar2::isGraphCaptureActive()||!workspace_||reinterpret_cast<std::uintptr_t>(getStream())<=2||
        table<0||table>=int(grouped_gateup_desc_tables_.size())||top_k<=0||top_k>int(kDeviceMoEMaxTopK))
        throw std::logic_error("Gate/up publication requires cold original table/stream/workspace admission");
    if(policy==MoEGateUpPublicationPolicy::Staged)
    {
        if(gate_up_last_producer_.contains(table))
            throw std::logic_error("A prepared last-producer gate/up owner cannot become Staged behind its captured graph");
        return 0;
    }
    if(policy!=MoEGateUpPublicationPolicy::LastProducer)
        throw std::invalid_argument("Unknown gate/up publication policy");
    // Binding a borrowed manager does not select a HIP device on this host
    // thread. Reuse the canonical backend/device guard before setup traffic.
    if(!validateROCmWorkspaceBinding(workspace_,device_ordinal_,"prepareGateUpPublication") ||
        !setMoEDevice(device_ordinal_,"prepareGateUpPublication"))
        throw std::logic_error("Last-producer gate/up lost its original ROCm workspace/device");
    if(gate_up_last_producer_.contains(table))
    {
        if(gate_up_last_producer_.at(table)->loads!=rocm::currentGateUpProducerLoads())
            throw std::logic_error("Prepared gate/up producer loads cannot be reselected");
        return gateUpPublicationSignature(table,top_k);
    }
    const auto& source=grouped_gateup_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        top_k>source.num_experts||!source.workspace_publication)
        throw std::invalid_argument("Last-producer gate/up requires an original native M1 descriptor table");
    const auto plan=rocm::gateUpLastProducerPlan(source.intermediate,source.d_model,top_k,source.policy_codebook_mask);
    auto* counters=static_cast<std::uint32_t*>(workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION));
    if(!counters||workspace_->getBufferSize(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)<plan.bytes())
        throw std::invalid_argument("Last-producer gate/up has no sufficient named PMA counter bank");
    auto binding=std::make_shared<GateUpLastProducerBinding>(GateUpLastProducerBinding{
        rocm::currentGateUpProducerLoads(),workspace_,workspace_->id(),getStream(),table,top_k,source.d_model,source.intermediate,source.num_experts,
        source.codebook_mask,source.policy_codebook_mask,plan.bind(counters),plan.bytes(),
        workspace_->retainBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION,0,plan.bytes()),source.workspace_publication,
        source.device_gate_descs,source.device_up_descs,source.workspace_slot,source.original_family,{},{}});
    for(const auto* name:std::array{MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
        MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
        MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
        MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
        MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES})
    {
        const auto* pointer=workspace_->getBuffer(name);
        if(!pointer)throw std::logic_error("Last-producer gate/up original scratch/table preparation is incomplete");
        const auto bytes=workspace_->getBufferSize(name);
        binding->scratch.push_back({name,pointer,bytes,workspace_->retainBuffer(name,0,bytes)});
    }
    // Source tensors, partial banks and counters are distinct admitted names.
    // The public cold metadata boundary proves completion of this exact setup
    // copy before host staging retires. A retained leaf may later be imported
    // into a parent launched on another stream, so capture-stream order alone
    // is insufficient. Only this one-time publication waits on its own event;
    // every subsequent reset remains device-owned in the last producer CTA.
    const std::vector<std::byte> initial_counters(plan.bytes(),std::byte{0});
    TransferEngine::instance().initializeWorkspaceMetadata(*binding->lease,initial_counters,getStream());
    std::uint64_t signature=kFNV1a64OffsetBasis;
    const auto mix=[&](std::uint64_t value){signature=(signature^value)*kFNV1a64Prime;};
    for(const auto* pointer:std::array<const void*,7>{workspace_,getStream(),counters,source.device_gate_descs,
        source.device_up_descs,source.workspace_publication.get(),rocm::gateUpLastProducerKernelFunction()})
        mix(reinterpret_cast<std::uintptr_t>(pointer));
    mix(workspace_->id());mix(static_cast<std::uint64_t>(binding->loads));mix(table);mix(top_k);mix(source.d_model);mix(source.intermediate);mix(source.num_experts);
    mix(source.codebook_mask);mix(source.policy_codebook_mask);mix(source.workspace_slot);
    mix(plan.tiles());mix(plan.contributors());mix(plan.bytes());
    for(const auto& seal:binding->scratch){mix(reinterpret_cast<std::uintptr_t>(seal.pointer));mix(seal.bytes);}
    binding->signature=signature?signature:1;
    gate_up_last_producer_.emplace(table,std::move(binding));
    return gateUpPublicationSignature(table,top_k);
}

/** @copydoc IMoEKernel::gateUpPublicationSignature */
std::uint64_t ROCmMoEKernel::gateUpPublicationSignature(int table,int top_k) const
{
    const auto found=gate_up_last_producer_.find(table);
    if(found==gate_up_last_producer_.end())return 0;
    const auto& binding=*found->second;
    if(!workspace_||workspace_!=binding.workspace||workspace_->id()!=binding.generation||getStream()!=binding.stream||
        table!=binding.table||top_k!=binding.top_k||table<0||table>=int(grouped_gateup_desc_tables_.size()))
        throw std::logic_error("Last-producer gate/up changed its original owner/generation/stream/table");
    const auto& source=grouped_gateup_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        source.d_model!=binding.hidden||source.intermediate!=binding.intermediate||source.num_experts!=binding.experts||
        source.codebook_mask!=binding.execution_mask||source.policy_codebook_mask!=binding.policy_mask||
        source.workspace_slot!=binding.descriptor_slot||source.workspace_publication!=binding.publication||
        source.device_gate_descs!=binding.original_gate||source.device_up_descs!=binding.original_up||
        !source.original_family.sameIdentity(binding.original))
        throw std::logic_error("Last-producer gate/up changed its original descriptor/source publication");
    if(workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)!=binding.view.counters||
        workspace_->getBufferSize(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)<binding.counter_bytes)
        throw std::logic_error("Last-producer gate/up counter bank was rebound or released");
    for(const auto& seal:binding.scratch)
        if(workspace_->getBuffer(seal.name)!=seal.pointer||workspace_->getBufferSize(seal.name)<seal.bytes)
            throw std::logic_error("Last-producer gate/up original scratch/table storage changed");
    return binding.signature;
}

/** @copydoc gateUpCompletionForCall */
const NativeTileCompletionView* ROCmMoEKernel::gateUpCompletionForCall(int table,int top_k) const
{
    const auto signature=gateUpPublicationSignature(table,top_k);
    if(signature&&gate_up_last_producer_.at(table)->loads!=rocm::currentGateUpProducerLoads())
        throw std::logic_error("Prepared gate/up producer loads cannot change behind retained capture identity");
    return signature?&gate_up_last_producer_.at(table)->view:nullptr;
}

/** @brief Immutable borrowed counter/table/scratch seal; the arena remains allocation authority. */
struct ROCmMoEKernel::DownLastProducerBinding
{
    DeviceWorkspaceManager* workspace;
    std::uint64_t generation;
    void* stream;
    int table,top_k,hidden,intermediate,experts;
    std::uint32_t execution_mask,policy_mask;
    NativeDownCompletionView view;
    std::size_t counter_bytes;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::shared_ptr<GroupedDescriptorWorkspacePublication> publication;
    const void* original_down;
    std::size_t descriptor_slot;
    MoEIntermediateShardDescriptorDeclaration original;
    struct ScratchSeal
    {
        const char* name;
        const void* pointer;
        std::size_t bytes;
        std::shared_ptr<const WorkspaceBufferLease> lease;
    };
    std::vector<ScratchSeal> scratch;
    std::uint64_t signature;
};

/** @copydoc IMoEKernel::prepareDownPublication */
std::uint64_t ROCmMoEKernel::prepareDownPublication(int table,int top_k,MoEDownPublicationPolicy policy)
{
    if(::llaminar2::isGraphCaptureActive()||!workspace_||reinterpret_cast<std::uintptr_t>(getStream())<=2||
        table<0||table>=int(grouped_down_desc_tables_.size())||top_k<=0||top_k>int(kDeviceMoEMaxTopK))
        throw std::logic_error("Down/route publication requires cold original table/stream/workspace admission");
    if(policy==MoEDownPublicationPolicy::Staged)
    {
        if(down_last_producer_.contains(table))
            throw std::logic_error("A prepared last-producer down/route owner cannot become Staged behind its captured graph");
        return 0;
    }
    if(policy!=MoEDownPublicationPolicy::LastProducerRouteFold)
        throw std::invalid_argument("Unknown down/route publication policy");
    // Binding a borrowed manager does not select a HIP device on this host
    // thread. Reuse the canonical backend/device guard before setup traffic.
    if(!validateROCmWorkspaceBinding(workspace_,device_ordinal_,"prepareDownPublication") ||
        !setMoEDevice(device_ordinal_,"prepareDownPublication"))
        throw std::logic_error("Last-producer down/route lost its original ROCm workspace/device");
    if(down_last_producer_.contains(table))return downPublicationSignature(table,top_k);
    const auto& source=grouped_down_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        top_k>source.num_experts||!source.workspace_publication)
        throw std::invalid_argument("Last-producer down/route requires an original native M1 descriptor table");
    const auto plan=rocm::downLastProducerPlan(source.d_model,source.intermediate,top_k,source.policy_codebook_mask);
    auto* counters=static_cast<std::uint32_t*>(workspace_->getBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION));
    if(!counters||workspace_->getBufferSize(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)<plan.bytes())
        throw std::invalid_argument("Last-producer down/route has no sufficient named PMA counter bank");
    auto binding=std::make_shared<DownLastProducerBinding>(DownLastProducerBinding{
        workspace_,workspace_->id(),getStream(),table,top_k,source.d_model,source.intermediate,source.num_experts,
        source.codebook_mask,source.policy_codebook_mask,plan.bind(counters),plan.bytes(),
        workspace_->retainBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION,0,plan.bytes()),source.workspace_publication,
        source.device_descs,source.workspace_slot,source.original_family,{},{}});
    for(const auto* name:std::array{MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
        MoEWorkspaceBuffers::DOWN_PARTIALS,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
        MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES})
    {
        const auto* pointer=workspace_->getBuffer(name);
        if(!pointer)throw std::logic_error("Last-producer down/route original scratch/table preparation is incomplete");
        const auto bytes=workspace_->getBufferSize(name);
        binding->scratch.push_back({name,pointer,bytes,workspace_->retainBuffer(name,0,bytes)});
    }
    // Source tensors, partial banks and counters are distinct admitted names.
    // The public cold metadata boundary proves completion of this exact setup
    // copy before host staging retires. A retained leaf may later be imported
    // into a parent launched on another stream, so capture-stream order alone
    // is insufficient. Only this one-time publication waits on its own event;
    // every subsequent reset remains device-owned in the last producer CTA.
    const std::vector<std::byte> initial_counters(plan.bytes(),std::byte{0});
    TransferEngine::instance().initializeWorkspaceMetadata(*binding->lease,initial_counters,getStream());
    std::uint64_t signature=kFNV1a64OffsetBasis;
    const auto mix=[&](std::uint64_t value){signature=(signature^value)*kFNV1a64Prime;};
    for(const auto* pointer:std::array<const void*,6>{workspace_,getStream(),counters,source.device_descs,
        source.workspace_publication.get(),rocm::downLastProducerKernelFunction()})
        mix(reinterpret_cast<std::uintptr_t>(pointer));
    mix(workspace_->id());mix(table);mix(top_k);mix(source.d_model);mix(source.intermediate);mix(source.num_experts);
    mix(source.codebook_mask);mix(source.policy_codebook_mask);mix(source.workspace_slot);
    mix(plan.partitions().tiles());mix(plan.partitions().contributors());mix(plan.bytes());
    for(const auto& seal:binding->scratch){mix(reinterpret_cast<std::uintptr_t>(seal.pointer));mix(seal.bytes);}
    binding->signature=signature?signature:1;
    down_last_producer_.emplace(table,std::move(binding));
    return downPublicationSignature(table,top_k);
}

/** @copydoc IMoEKernel::downPublicationSignature */
std::uint64_t ROCmMoEKernel::downPublicationSignature(int table,int top_k) const
{
    const auto found=down_last_producer_.find(table);
    if(found==down_last_producer_.end())return 0;
    const auto& binding=*found->second;
    if(!workspace_||workspace_!=binding.workspace||workspace_->id()!=binding.generation||getStream()!=binding.stream||
        table!=binding.table||top_k!=binding.top_k||table<0||table>=int(grouped_down_desc_tables_.size()))
        throw std::logic_error("Last-producer down/route changed its original owner/generation/stream/table");
    const auto& source=grouped_down_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        source.d_model!=binding.hidden||source.intermediate!=binding.intermediate||source.num_experts!=binding.experts||
        source.codebook_mask!=binding.execution_mask||source.policy_codebook_mask!=binding.policy_mask||
        source.workspace_slot!=binding.descriptor_slot||source.workspace_publication!=binding.publication||
        source.device_descs!=binding.original_down||
        !source.original_family.sameIdentity(binding.original))
        throw std::logic_error("Last-producer down/route changed its original descriptor/source publication");
    if(workspace_->getBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)!=binding.view.partitions.counters||
        workspace_->getBufferSize(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)<binding.counter_bytes)
        throw std::logic_error("Last-producer down/route counter bank was rebound or released");
    for(const auto& seal:binding.scratch)
        if(workspace_->getBuffer(seal.name)!=seal.pointer||workspace_->getBufferSize(seal.name)<seal.bytes)
            throw std::logic_error("Last-producer down/route original scratch/table storage changed");
    return binding.signature;
}

/** @copydoc downCompletionForCall */
const NativeDownCompletionView* ROCmMoEKernel::downCompletionForCall(int table,int top_k) const
{
    const auto signature=downPublicationSignature(table,top_k);
    return signature?&down_last_producer_.at(table)->view:nullptr;
}

#include "ROCmFullNativeHCFFNOwner.inc"

/** @brief Sealed native program plus original tensor/workspace lifetime observations. */
class ROCmMoEKernel::PreparedResidentFFN final : public PreparedResidentMoEFFN
{
public:
    /**
     * @brief Retain original FFN authority and one exact already-admitted native program.
     * @param owner Original backend source/table/workspace authority.
     * @param request Original tensor/source declaration retained by the stage.
     * @param args Complete admitted FFN argument snapshot.
     * @param admission Actual standalone or full function's immutable resource/admission evidence.
     * @param stream Exact original non-default submission stream.
     * @param scratch Existing named FFN leases and consumed byte extents.
     * @param shared_tables Original immutable shared descriptor publication handles.
     * @param router Optional original router/suffix snapshot.
     * @param router_tensors Original router/ID/weight tensor identity observations.
     * @param barrier_slot Existing exclusive instance slot, retained until graph retirement.
     * @param prefix Optional admitted HC phase; its original stage retains and validates physical leases.
     */
    PreparedResidentFFN(ROCmMoEKernel& owner,ResidentMoEFFNRequest request,rocm::ResidentMoEFFNArgs args,
        rocm::PreparedMoEFFNProgram admission,void* stream,std::vector<ResidentFFNScratchBinding> scratch,
        ResidentFFNSharedTables shared_tables,std::optional<rocm::NativeLayoutRouterFFNArgs> router={},
        std::array<TensorBase*,3> router_tensors={},
        std::shared_ptr<PersistentWorkspaceSlotLease> barrier_slot={},
        std::optional<rocm::FullNativeHCFFNPrefixArgs> prefix={})
        :owner_(&owner),request_(std::move(request)),args_(args),admission_(admission),stream_(stream),
         workspace_(owner.getWorkspace()),workspace_id_(workspace_->id()),scratch_(std::move(scratch)),
         shared_tables_(std::move(shared_tables)),gate_publication_(owner.grouped_gateup_desc_tables_.at(request_.gate_up_table).workspace_publication),
         down_publication_(owner.grouped_down_desc_tables_.at(request_.down_table).workspace_publication),
         shared_gate_publication_(owner.grouped_gateup_desc_tables_.at(shared_tables_.gate_up).workspace_publication),
         shared_down_publication_(owner.grouped_down_desc_tables_.at(shared_tables_.down).workspace_publication),
         barrier_slot_(std::move(barrier_slot)),native_router_(std::move(router)),full_prefix_(std::move(prefix)),router_tensors_(router_tensors),router_publication_((admission.evidence().kind==ResidentMoEFFNLaunchKind::OrdinaryPhases||native_router_)
            ?owner.paired_router_descriptors_:std::nullopt)
    {
        const auto mix=[&](std::uint64_t value){signature_=(signature_^value)*kFNV1a64Prime;};
        for(const void* pointer:std::array<const void*,23>{args_.runtime,args_.hidden,args_.gates,args_.ups,args_.downs,
            args_.shared_gate,args_.shared_up,args_.shared_down,
            args_.shared_input_gate,args_.hidden_q8,args_.hidden_scales,args_.gate_partials,args_.up_partials,args_.down_partials,
            args_.swiglu_q8,args_.swiglu_scales,args_.route_contributions,args_.routed_output,args_.shared_output,args_.combined_output,
            args_.shared_gate_output,args_.shared_up_output,stream_})mix(reinterpret_cast<std::uintptr_t>(pointer));
        for(const auto& matrix:std::array{request_.shared.gate,request_.shared.up,request_.shared.down})
        {
            for(const void* pointer:std::array<const void*,4>{matrix.payload,matrix.scales,matrix.mins,matrix.emins})
                mix(reinterpret_cast<std::uintptr_t>(pointer));
            mix(matrix.n);mix(matrix.k);mix(matrix.blocks_per_row);mix(matrix.codebook_id);
            mix(matrix.source_identity_present);mix(matrix.source_codebook_id);mix(matrix.source_is_superblock);
            mix(matrix.allocation_payload_bytes_per_block);mix(matrix.allocation_has_mins);mix(matrix.allocation_has_emins);
        }
        const auto rows=args_.rows.captureIdentity();
        mix(reinterpret_cast<std::uintptr_t>(rows.count));mix(rows.count_capacity);mix(rows.first_physical_row);
        mix(rows.physical_rows);mix(rows.repeats_per_row);mix(rows.schema_version);
        mix(args_.hidden_columns);mix(args_.routed_columns);mix(args_.shared_columns);mix(args_.experts);mix(args_.top_k);mix(args_.resident_grid);
        mix(args_.execution.routed_gate);mix(args_.execution.routed_down);
        mix(args_.execution.shared_gate);mix(args_.execution.shared_down);
        mix(args_.original.firstIntermediateColumn());mix(args_.original.intermediateColumns());
        mix(args_.original.target().participantId());mix(args_.original.target().cohortTarget());
        const auto& evidence=admission_.evidence();
        sealRetainedModelReceipt();
        mix(static_cast<std::uint64_t>(evidence.kind));mix(evidence.generation);mix(evidence.nativeEntries().size());
        mix(static_cast<std::uint64_t>(evidence.device.type));mix(evidence.device.ordinal);
        for(const auto& entry:evidence.nativeEntries())
        {
            mix(reinterpret_cast<std::uintptr_t>(entry.function));mix(entry.grid);mix(entry.threads);mix(entry.dynamic_shared_bytes);
            mix(entry.resources.registers_per_thread);mix(entry.resources.maximum_threads_per_block);
            mix(entry.resources.blocks_per_processor);mix(entry.resources.private_bytes);mix(entry.resources.static_shared_bytes);
        }
        if(router_publication_)
        {
            mix(router_publication_->gateup_table);mix(router_publication_->down_table);
            mix(static_cast<unsigned>(router_publication_->outputs.format));
            mix(static_cast<unsigned>(router_publication_->outputs.projections));
        }
        for(const auto& binding:scratch_)
        {mix(reinterpret_cast<std::uintptr_t>(binding.pointer));mix(binding.bytes);}
        mix(workspace_id_);
        mix(shared_tables_.gate_up);mix(shared_tables_.down);
        for(const auto& p:std::array{args_.routed_gate_policy,args_.routed_down_policy,args_.shared_gate_policy,args_.shared_down_policy})
        {mix(p.p0);mix(p.p1);mix(p.p2);mix(p.p3);mix(p.maximum);}
        if(native_router_)
        {
            mix(reinterpret_cast<std::uintptr_t>(native_router_->barrier_state));
            for(std::size_t i=0;i<router_tensors_.size();++i)
            {
                if(!router_tensors_[i])throw std::logic_error("Native-layout router tensor owner is absent");
                router_shapes_[i]=router_tensors_[i]->shape();
                mix(reinterpret_cast<std::uintptr_t>(router_tensors_[i]));
                for(const auto extent:router_shapes_[i])mix(extent);
            }
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_weights));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_logits));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_indices));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_weights_output));
            mix(native_router_->normalize_router_weights);mix(native_router_->write_router_outputs);
            // Full-entry windows are immutable captured policy, independent of
            // current row count or routes. Standalone entries retain zero fields.
            mix(native_router_->full_entry_panel.rows);mix(native_router_->full_entry_panel.blocks);
        }
        if(full_prefix_)
        {
            if(!native_router_)throw std::logic_error("Full native HC requires its existing router/FFN owner");
            validateFullNativeHCFFNPrefixBinding(*full_prefix_,*native_router_);
            mixFullNativeHCFFNPrefix(*full_prefix_,mix);
        }
        if(!signature_)signature_=1;
    }
    /** @copydoc PreparedResidentMoEFFN::programEvidence */
    const ResidentMoEFFNProgramEvidence& programEvidence() const noexcept override {return admission_.evidence();}
    /** @copydoc PreparedResidentMoEFFN::captureSignature */
    std::uint64_t captureSignature() const noexcept override {return signature_;}
    /** @copydoc PreparedResidentMoEFFN::validateBinding */
    void validateBinding() const override
    {
        if(owner_->getStream()!=stream_||owner_->getWorkspace()!=workspace_||workspace_->id()!=workspace_id_)
            throw std::logic_error("Resident FFN changed its original stream/workspace owner");
        for(const auto& binding:scratch_)
            if(workspace_->getBuffer(binding.name)!=binding.pointer||workspace_->getBufferSize(binding.name)<binding.bytes)
                throw std::logic_error("Resident FFN original scratch storage was rebound");
        if(request_.gate_up_table>=int(owner_->grouped_gateup_desc_tables_.size())||
            request_.down_table>=int(owner_->grouped_down_desc_tables_.size())||
            shared_tables_.gate_up>=int(owner_->grouped_gateup_desc_tables_.size())||
            shared_tables_.down>=int(owner_->grouped_down_desc_tables_.size()))
            throw std::logic_error("Resident FFN original descriptor table retired");
        const auto& gate=owner_->grouped_gateup_desc_tables_[request_.gate_up_table];
        const auto& down=owner_->grouped_down_desc_tables_[request_.down_table];
        if(!gate.valid||!down.valid||gate.workspace_publication!=gate_publication_||
            down.workspace_publication!=down_publication_||((!router_publication_||native_router_)&&gate.device_gate_descs!=args_.gates)||
            ((!router_publication_||native_router_)&&(gate.device_up_descs!=args_.ups||down.device_descs!=args_.downs))||
            !gate.original_family.sameIdentity(request_.original)||!down.original_family.sameIdentity(request_.original)||
            gate.codebook_mask!=args_.execution.routed_gate||down.codebook_mask!=args_.execution.routed_down||
            gate.codebook_mask!=request_.original.nativeGateUpEnvelope().execution_codebook_mask||
            down.codebook_mask!=request_.original.nativeDownEnvelope().execution_codebook_mask||
            gate.policy_codebook_mask!=request_.original.nativeGateUpEnvelope().policy_codebook_mask||
            down.policy_codebook_mask!=request_.original.nativeDownEnvelope().policy_codebook_mask||
            gate.num_experts!=args_.experts||down.num_experts!=args_.experts||gate.d_model!=args_.hidden_columns||
            down.d_model!=args_.hidden_columns||gate.intermediate!=args_.routed_columns||down.intermediate!=args_.routed_columns)
            throw std::logic_error("Resident FFN original descriptor storage or source family changed");
        if(router_publication_&&!native_router_)
        {
            const auto& original=*router_publication_;
            const auto& current=owner_->requirePairedRouterDescriptors(request_.runtime,args_.experts,args_.top_k);
            if(current.gateup_table!=original.gateup_table||current.down_table!=original.down_table||
                current.gateup_table!=request_.gate_up_table||current.down_table!=request_.down_table||
                current.outputs.gate!=args_.gates||current.outputs.up!=args_.ups||current.outputs.down!=args_.downs||
                current.outputs.format!=DeviceMoEWeightFormat::NativeQuantised||
                current.outputs.projections!=DeviceMoEProjectionSet::IntermediateShard)
                throw std::logic_error("Ordinary FFN original router descriptor publication changed");
        }
        if(native_router_)
        {
            const auto barrier_bytes=full_prefix_?rocm::fullNativeHCFFNSlotBytes(full_prefix_->implementation):NativeLayoutBarrierLayout::bytes;
            auto* barrier_base=barrier_slot_?static_cast<std::uint32_t*>(workspace_->getPersistentSlotBuffer(
                MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                barrier_slot_->slot(),barrier_bytes)):nullptr;
            if(!barrier_base||native_router_->barrier_state!=barrier_base+
                (full_prefix_?FullNativeHCFFNBarrierLayout::suffix_word_offset:0))
                throw std::logic_error("Native-layout barrier slot binding changed");
            if(full_prefix_)
            {
                if(full_prefix_->barrier_state!=barrier_base)
                    throw std::logic_error("Full native HC prefix barrier slot binding changed");
                validateFullNativeHCFFNPrefixBinding(*full_prefix_,*native_router_);
            }
            const std::array<const void*,3> expected{native_router_->router_weights,native_router_->router_indices,native_router_->router_weights_output};
            for(std::size_t i=0;i<router_tensors_.size();++i)
                if(router_tensors_[i]->gpu_data_ptr()!=expected[i]||router_tensors_[i]->shape()!=router_shapes_[i])
                    throw std::logic_error("Native-layout router tensor storage or shape changed");
            const auto& publisher=owner_->requirePairedRouterDescriptors(request_.runtime,args_.experts,args_.top_k);
            if(!router_publication_||publisher.runtime!=router_publication_->runtime||
                publisher.workspace!=router_publication_->workspace||publisher.workspace_id!=router_publication_->workspace_id||
                publisher.stream!=router_publication_->stream||publisher.outputs.gate!=router_publication_->outputs.gate||
                publisher.outputs.up!=router_publication_->outputs.up||publisher.outputs.down!=router_publication_->outputs.down)
                throw std::logic_error("Native-layout router publication owner changed");
            if(publisher.gateup_table!=request_.gate_up_table||publisher.down_table!=request_.down_table||
                publisher.outputs.expert_count!=native_router_->router_publication.expert_count||
                publisher.outputs.format!=native_router_->router_publication.format||
                publisher.outputs.projections!=native_router_->router_publication.projections||
                publisher.outputs.gate!=native_router_->router_publication.gate||publisher.outputs.up!=native_router_->router_publication.up||
                publisher.outputs.down!=native_router_->router_publication.down||owner_->d_route_logits_!=native_router_->router_logits)
                throw std::logic_error("Native-layout FFN original router publication changed");
        }
        const auto& shared_gate=owner_->grouped_gateup_desc_tables_[shared_tables_.gate_up];
        const auto& shared_down=owner_->grouped_down_desc_tables_[shared_tables_.down];
        // Immutable source metadata is checked field by field, never by a live
        // source-data digest or a device readback. Table rewrites must reject
        // even when their publication object and device addresses survive.
        const auto same=[](const DeviceNativeQuantisedMatrixDesc& a,const DeviceNativeQuantisedMatrixDesc& b) {
            return a.payload==b.payload && a.scales==b.scales && a.mins==b.mins && a.emins==b.emins &&
                a.n==b.n && a.k==b.k && a.blocks_per_row==b.blocks_per_row && a.codebook_id==b.codebook_id &&
                a.allocation_payload_bytes_per_block==b.allocation_payload_bytes_per_block &&
                a.allocation_has_mins==b.allocation_has_mins && a.allocation_has_emins==b.allocation_has_emins &&
                a.source_codebook_id==b.source_codebook_id && a.source_is_superblock==b.source_is_superblock &&
                a.source_identity_present==b.source_identity_present && a.reserved==b.reserved && a.reserved_tail==b.reserved_tail;
        };
        if(!shared_gate.valid||!shared_down.valid||shared_gate.num_experts!=1||shared_down.num_experts!=1||
            shared_gate.weight_format!=DeviceMoEWeightFormat::NativeQuantised||shared_down.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
            shared_gate.descriptor_source!=MoEDecodeDescriptorSource::StaticDescriptorTable||
            shared_down.descriptor_source!=MoEDecodeDescriptorSource::StaticDescriptorTable||
            shared_gate.workspace_publication!=shared_gate_publication_||shared_down.workspace_publication!=shared_down_publication_||
            shared_gate.device_gate_descs!=args_.shared_gate||shared_gate.device_up_descs!=args_.shared_up||
            shared_gate.codebook_mask!=args_.execution.shared_gate||shared_down.codebook_mask!=args_.execution.shared_down||
            shared_down.device_descs!=args_.shared_down||shared_gate.d_model!=args_.hidden_columns||
            shared_down.d_model!=args_.hidden_columns||shared_gate.intermediate!=args_.shared_columns||shared_down.intermediate!=args_.shared_columns||
            shared_gate.host_gate_descs.size()!=1||shared_gate.host_up_descs.size()!=1||shared_down.host_descs.size()!=1||
            !same(shared_gate.host_gate_descs[0],request_.shared.gate)||!same(shared_gate.host_up_descs[0],request_.shared.up)||
            !same(shared_down.host_descs[0],request_.shared.down))
            throw std::logic_error("Resident FFN original immutable shared descriptor publication changed");
        // Re-evaluate the arena's exact named table intervals. Retained leases
        // preserve old bytes after growth, but never authorize replaying them.
        const auto tableMatches=[&](const char* name,std::size_t slot,const void* pointer,std::size_t experts) {
            const auto available=workspace_->getBufferSize(name);
            const auto stride=(available/MoEWorkspaceBuffers::kGroupedDescriptorTableSlots/
                sizeof(DeviceNativeQuantisedMatrixDesc))*sizeof(DeviceNativeQuantisedMatrixDesc);
            const auto base=reinterpret_cast<std::uintptr_t>(workspace_->getBuffer(name));
            return base && slot<MoEWorkspaceBuffers::kGroupedDescriptorTableSlots &&
                stride>=experts*sizeof(DeviceNativeQuantisedMatrixDesc) &&
                (slot+1)*stride<=available &&
                reinterpret_cast<std::uintptr_t>(pointer)==base+slot*stride;
        };
        if(router_publication_&&!native_router_&&(
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,gate.workspace_slot,args_.gates,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,gate.workspace_slot,args_.ups,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,down.workspace_slot,args_.downs,args_.experts)))
            throw std::logic_error("Ordinary FFN original router descriptor arena changed");
        if(!tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,gate.workspace_slot,gate.device_gate_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,gate.workspace_slot,gate.device_up_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,down.workspace_slot,down.device_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,shared_gate.workspace_slot,args_.shared_gate,1)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,shared_gate.workspace_slot,args_.shared_up,1)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,shared_down.workspace_slot,args_.shared_down,1))
            throw std::logic_error("Resident FFN original named descriptor arena changed");
        const auto tensors=std::array<ITensor*,7>{request_.hidden,request_.shared_input_gate,request_.route_contributions,
            request_.routed_output,request_.shared_output,request_.combined_output,request_.shared_gate_scratch};
        const auto addresses=std::array<const void*,7>{args_.hidden,args_.shared_input_gate,args_.route_contributions,
            args_.routed_output,args_.shared_output,args_.combined_output,args_.shared_gate_output};
        for(std::size_t i=0;i<tensors.size();++i)
            if(addresses[i]&&(!tensors[i]||tensors[i]->gpu_data_ptr()!=addresses[i]))
                throw std::logic_error("Resident FFN public tensor storage changed");
        if(args_.shared_up_output&&(!request_.shared_up_scratch||request_.shared_up_scratch->gpu_data_ptr()!=args_.shared_up_output))
            throw std::logic_error("Resident FFN shared up storage changed");
    }
    /** @copydoc PreparedResidentMoEFFN::enqueue */
    void enqueue(void* stream) const override
    {
        if(stream!=stream_)throw std::logic_error("Resident FFN submission changed its original stream");
        validateBinding();
        if(full_prefix_)
        {
            const rocm::FullNativeHCFFNArgs full{*native_router_,*full_prefix_};
            rocm::launchFullNativeHCFFN(admission_,full,stream,workspace_id_);
            if(!retained_model_receipt_.empty()&&::llaminar2::isGraphCaptureActive())
                PerfStatsCollector::addCounter("gpu_graph_inventory","retained_m1_layer_binding",1.0,
                    "graph_setup",admission_.evidence().device.toString(),retained_model_receipt_);
            PerfStatsCollector::addCounter("kernel","rocm_full_native_hc_ffn_calls",1.0,"moe",
                admission_.evidence().device.toString(),{{"resident_grid",std::to_string(args_.resident_grid)}});
        }
        else if(native_router_)rocm::launchNativeLayoutRouterFFN(admission_,*native_router_,stream,workspace_id_);
        else rocm::launchResidentMoEFFN(admission_,args_,stream,workspace_id_);
        PerfStatsCollector::addCounter("kernel","rocm_resident_moe_ffn_calls",1.0,"moe",admission_.evidence().device.toString(),
            {{"top_k",std::to_string(args_.top_k)},{"routed_f",std::to_string(args_.routed_columns)},
             {"shared_f",std::to_string(args_.shared_columns)},{"resident_grid",std::to_string(args_.resident_grid)},{"native_launches",std::to_string(admission_.evidence().nativeEntries().size())}});
    }
private:
    /**
     * @brief Seal an opt-in diagnostic census from original model, detector and prepared launch owners.
     * @pre The exact registered HIP entry and its stream were admitted cold by the original backend.
     * @details Captured replay never resolves function names or constructs tags.
     */
    void sealRetainedModelReceipt()
    {
        if(!request_.model_main_inference ||
           !PerfStatsCollector::isRecordFamilyEnabled("gpu_graph_inventory","retained_m1_layer_binding"))return;
        if(!full_prefix_)
            throw std::logic_error("Retained model receipt requires its original full HC/FFN preparation");

        const auto sourceName=[](const ExpertWeightFormat& role) -> std::string {
            const auto source=role.native_quantised;
            if(role.isNativeQuantised() && source.present) {
                if(source.codebook_id==5 && source.is_superblock)return "Q4_K";
                if(source.codebook_id==7)return source.is_superblock?"Q5_K":"Q5_1";
                if(source.codebook_id==19 && !source.is_superblock)return "Q8_0";
                return std::string("codebook_") + std::to_string(source.codebook_id) +
                    (source.is_superblock ? "_sb" : "");
            }
            throw std::logic_error("Retained model receipt lost its original GGUF source identity");
        };
        const auto& mode=*request_.model_main_inference;
        const auto& sources=request_.original.sourceFormats();
        const auto& evidence=admission_.evidence();
        const auto& entry=evidence.nativeEntries().front();
        const auto* symbol=hipKernelNameRefByPtr(entry.function,static_cast<hipStream_t>(stream_));
        if(!symbol||!*symbol||mode.source_layer<0||!barrier_slot_)
            throw std::logic_error("Retained model receipt requires its resolved HIP kernel and source layer");
        const auto& hardware=rocmHardwareMetadata().at(evidence.device.ordinal);
        retained_model_receipt_={
            {"layer",std::to_string(mode.source_layer)},
            {"h",std::to_string(args_.hidden_columns)},{"r",std::to_string(full_prefix_->hc.publication.columns)},
            {"f",std::to_string(args_.routed_columns)},{"s",std::to_string(args_.shared_columns)},
            {"experts",std::to_string(args_.experts)},{"top_k",std::to_string(args_.top_k)},
            {"f_start",std::to_string(args_.original.firstIntermediateColumn())},
            {"s_start",std::to_string(mode.shared_source_first_column)},
            {"gate",sourceName(sources[0])},{"up",sourceName(sources[1])},{"down",sourceName(sources[2])},
            {"pci",hardware.pci_address},{"phase","decode"},{"role","main_inference"},
            {"rows",std::to_string(args_.rows.physicalRows())},{"requests","1"},
            {"scratch_slot",std::to_string(barrier_slot_->slot())},
            {"kernel",symbol},{"grid",std::to_string(entry.grid)+"x1x1"},{"block",std::to_string(entry.threads)+"x1x1"},
        };
    }

    ROCmMoEKernel* owner_;
    const ResidentMoEFFNRequest request_;
    PerfStatsCollector::Tags retained_model_receipt_; ///< Cold optional model census; no device state is cached here.
    const rocm::ResidentMoEFFNArgs args_;
    const rocm::PreparedMoEFFNProgram admission_;
    const std::shared_ptr<PersistentWorkspaceSlotLease> barrier_slot_; ///< Exclusive barrier bytes until graph retirement.
    const std::optional<rocm::NativeLayoutRouterFFNArgs> native_router_;
    const std::optional<rocm::FullNativeHCFFNPrefixArgs> full_prefix_; ///< Original stage-owned phase snapshot, not another source/table owner.
    const std::array<TensorBase*,3> router_tensors_;
    std::array<std::vector<std::size_t>,3> router_shapes_;
    void* const stream_;
    DeviceWorkspaceManager* const workspace_;
    const std::uint64_t workspace_id_;
    const std::vector<ResidentFFNScratchBinding> scratch_;
    const ResidentFFNSharedTables shared_tables_;
    const std::optional<PairedRouterDescriptors> router_publication_; ///< Same original router's existing borrowed destinations.
    const std::shared_ptr<GroupedDescriptorWorkspacePublication> gate_publication_,down_publication_,
        shared_gate_publication_,shared_down_publication_;
    std::uint64_t signature_=kFNV1a64OffsetBasis;
};

/** @copydoc IResidentMoEFFNPreparation::prepareResidentMoEFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareResidentMoEFFN(
    const ResidentMoEFFNRequest& request,void* stream)
{
    return prepareResidentMoEFFNImpl(request,stream,nullptr,nullptr,nullptr,false);
}

/** @copydoc ROCmMoEKernel::prepareNativeLayoutRouterFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareNativeLayoutRouterFFN(
    const ResidentMoEFFNRequest& request,TensorBase* router,TensorBase* indices,TensorBase* weights,bool normalize,void* stream)
{
    if(!router||router->native_type()!=TensorType::FP16||!indices||!weights||
        indices->native_type()!=TensorType::FP32||weights->native_type()!=TensorType::FP32)
        throw std::invalid_argument("Native-layout router FFN requires the original prepared FP16 router and public outputs");
    const rocm::ScopedResidentMoEFFNProgram selection(rocm::ResidentMoEFFNProgram::NativeLayoutRouter);
    preparePairedRouterDecodeDescriptors(request.runtime,request.gate_up_table,request.down_table,request.top_k);
    return prepareResidentMoEFFNImpl(request,stream,router,indices,weights,normalize);
}

/** @copydoc ROCmMoEKernel::prepareFullNativeHCFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareFullNativeHCFFN(
    const ResidentMoEFFNRequest& request,TensorBase* router,TensorBase* indices,TensorBase* weights,
    bool normalize,const rocm::FullNativeHCFFNPrefixArgs& prefix,void* stream)
{
    if(!router||router->native_type()!=TensorType::FP16||!indices||!weights||
        indices->native_type()!=TensorType::FP32||weights->native_type()!=TensorType::FP32||prefix.barrier_state)
        throw std::invalid_argument("Full native HC requires original FP16 router outputs and an unbound barrier");
    const rocm::ScopedResidentMoEFFNProgram selection(rocm::ResidentMoEFFNProgram::NativeLayoutRouter);
    preparePairedRouterDecodeDescriptors(request.runtime,request.gate_up_table,request.down_table,request.top_k);
    return prepareResidentMoEFFNImpl(request,stream,router,indices,weights,normalize,&prefix);
}

/** @brief Share the original source/table/arena resolver without preparing another native program. */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareResidentMoEFFNImpl(
    const ResidentMoEFFNRequest& request,void* stream,TensorBase* router,TensorBase* indices,TensorBase* weights,bool normalize,
    const rocm::FullNativeHCFFNPrefixArgs* prefix)
{
    const auto device=DeviceId::rocm(device_ordinal_);
    if(!workspace_||!stream||stream!=getStream()||::llaminar2::isGraphCaptureActive())
        throw std::logic_error("Resident FFN preparation requires its original workspace/stream outside capture");
    const auto& original=request.original.originalBinding();
    const int H=int(original.modelColumns()),F=int(original.localIntermediateColumns());
    if(!request.original.valid()||request.original.weightFormat()!=DeviceMoEWeightFormat::NativeQuantised||
        request.rows.physicalRows()!=1||request.rows.capacity()!=1||!request.runtime||!request.hidden||request.top_k<=0||
        request.top_k>int(kDeviceMoEMaxTopK)||request.gate_up_table<0||request.down_table<0||
        request.gate_up_table>=int(grouped_gateup_desc_tables_.size())||request.down_table>=int(grouped_down_desc_tables_.size()))
        throw std::invalid_argument("Resident FFN requires an original native-quantised ordinary M1 source declaration");
    const auto& gate=grouped_gateup_desc_tables_[request.gate_up_table];
    const auto& down=grouped_down_desc_tables_[request.down_table];
    if(!gate.valid||!down.valid||!gate.deviceReady()||!down.deviceReady()||gate.num_experts!=down.num_experts||
        gate.d_model!=H||down.d_model!=H||gate.intermediate!=F||down.intermediate!=F||
        gate.descriptor_source!=MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable||
        down.descriptor_source!=MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable||
        !gate.original_family.sameIdentity(request.original)||!down.original_family.sameIdentity(request.original)||
        !request.original.validFor(device,gate.num_experts,H,F))
        throw std::invalid_argument("Resident FFN changed original private table/family identities");
    const auto routed_gate_envelope=request.original.nativeGateUpEnvelope();
    const auto routed_down_envelope=request.original.nativeDownEnvelope();
    if(!routed_gate_envelope.valid()||!routed_down_envelope.valid()||
        gate.codebook_mask!=routed_gate_envelope.execution_codebook_mask||
        down.codebook_mask!=routed_down_envelope.execution_codebook_mask||
        gate.policy_codebook_mask!=routed_gate_envelope.policy_codebook_mask||
        down.policy_codebook_mask!=routed_down_envelope.policy_codebook_mask)
        throw std::invalid_argument("Resident FFN table masks do not match the complete original source envelopes");
    const int shared_f=request.shared.gate.n;
    if(request.shared.weight_format!=DeviceMoEWeightFormat::NativeQuantised||shared_f<=0||shared_f%32||
        !validateGroupedGateUpDesc(request.shared.gate,H,shared_f)||!validateGroupedGateUpDesc(request.shared.up,H,shared_f)||
        !validateGroupedDownDesc(request.shared.down,H,shared_f))
        throw std::invalid_argument("Resident FFN shared source has invalid original native projection geometry");
    // The original prepared owner seals physical validity before these SAME
    // immutable shared tables become available to a full-entry consumer. This
    // does not certify allocation spans independently of the prepared owner.
    for(const auto& descriptor:std::array{request.shared.gate,request.shared.up,request.shared.down})
        if(!NativeQuantisedDescriptorValidation::publishable(descriptor))
            throw std::invalid_argument("Resident FFN shared publication has incomplete native physical planes");
    NativeQuantisedExecutionFormatEnvelope shared_gate,shared_down;
    const auto add=[](NativeQuantisedExecutionFormatEnvelope& envelope,const DeviceNativeQuantisedMatrixDesc& desc){
        return addNativeQuantisedExecutionFormat(envelope,desc.codebook_id,
            {.codebook_id=desc.source_codebook_id,.is_superblock=bool(desc.source_is_superblock),.present=bool(desc.source_identity_present)},false);
    };
    if(!add(shared_gate,request.shared.gate)||!add(shared_gate,request.shared.up)||!add(shared_down,request.shared.down))
        throw std::invalid_argument("Resident FFN shared source has no original arithmetic-policy identity");
    const auto pointer=[](ITensor* tensor,std::size_t elements)->float* {
        if(!tensor||tensor->native_type()!=TensorType::FP32||tensor->size_bytes()/sizeof(float)<elements||!tensor->gpu_data_ptr())
            throw std::invalid_argument("Resident FFN requires original materialized FP32 tensor storage");
        return static_cast<float*>(tensor->gpu_data_ptr());
    };
    rocm::ResidentMoEFFNArgs args;
    args.runtime=request.runtime;args.original=original;args.rows=request.rows;
    args.hidden=pointer(request.hidden,H);args.shared_input_gate=pointer(request.shared_input_gate,H);
    args.gates=gate.device_gate_descs;args.ups=gate.device_up_descs;args.downs=down.device_descs;
    const bool ordinary=rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::OrdinaryPhases;
    if(ordinary)
    {
        const auto& publisher=requirePairedRouterDescriptors(request.runtime,gate.num_experts,request.top_k);
        if(publisher.gateup_table!=request.gate_up_table||publisher.down_table!=request.down_table||
            publisher.outputs.format!=DeviceMoEWeightFormat::NativeQuantised||
            publisher.outputs.projections!=DeviceMoEProjectionSet::IntermediateShard)
            throw std::logic_error("Ordinary FFN requires its exact original router descriptor producer");
        args.gates=publisher.outputs.gate;args.ups=publisher.outputs.up;args.downs=publisher.outputs.down;
    }
    args.hidden_columns=H;args.routed_columns=F;args.shared_columns=shared_f;args.experts=gate.num_experts;args.top_k=request.top_k;
    const auto output_demand=prefix?prefix->output_demand:FullNativeHCFFNOutputDemand::all();
    const auto output=[&](FullNativeHCFFNOutputRole role,ITensor* tensor,std::size_t elements) {
        return output_demand.contains(role)?pointer(tensor,elements):nullptr;
    };
    args.route_contributions=output(FullNativeHCFFNOutputRole::RouteContributions,request.route_contributions,size_t(request.top_k)*H);
    args.routed_output=output(FullNativeHCFFNOutputRole::Routed,request.routed_output,H);
    args.shared_output=output(FullNativeHCFFNOutputRole::Shared,request.shared_output,H);
    args.combined_output=pointer(request.combined_output,H);
    args.shared_gate_output=output(FullNativeHCFFNOutputRole::SharedGate,request.shared_gate_scratch,shared_f);
    args.shared_up_output=output(FullNativeHCFFNOutputRole::SharedUp,request.shared_up_scratch,shared_f);
    struct Span{const void* pointer;std::size_t bytes;};
    const std::array public_reads{Span{args.hidden,size_t(H)*sizeof(float)},Span{args.shared_input_gate,size_t(H)*sizeof(float)}};
    const auto publication=[](const void* pointer,std::size_t bytes){return Span{pointer,pointer?bytes:0};};
    const std::array public_writes{publication(args.route_contributions,size_t(request.top_k)*H*sizeof(float)),
        publication(args.routed_output,size_t(H)*sizeof(float)),publication(args.shared_output,size_t(H)*sizeof(float)),
        publication(args.combined_output,size_t(H)*sizeof(float)),publication(args.shared_gate_output,size_t(shared_f)*sizeof(float)),
        publication(args.shared_up_output,size_t(shared_f)*sizeof(float))};
    const auto disjoint=[](Span a,Span b){const auto x=reinterpret_cast<std::uintptr_t>(a.pointer),y=reinterpret_cast<std::uintptr_t>(b.pointer);
        return !a.bytes||!b.bytes||(a.bytes<=std::numeric_limits<std::uintptr_t>::max()-x&&b.bytes<=std::numeric_limits<std::uintptr_t>::max()-y&&
            (x+a.bytes<=y||y+b.bytes<=x));};
    for(size_t i=0;i<public_writes.size();++i)
    {
        for(const auto& read:public_reads)if(!disjoint(public_writes[i],read))throw std::invalid_argument("Resident FFN output aliases an original input");
        for(size_t j=0;j<i;++j)if(!disjoint(public_writes[i],public_writes[j]))throw std::invalid_argument("Resident FFN public outputs alias");
    }
    std::vector<ResidentFFNScratchBinding> scratch;
    const auto bind=[&](const char* name,size_t bytes)->void* {
        auto* value=workspace_->getBuffer(name);
        if(!value||bytes>workspace_->getBufferSize(name))throw std::invalid_argument(std::string("Resident FFN original arena is insufficient: ")+name);
        const Span span{value,bytes};
        for(const auto& read:public_reads)if(!disjoint(span,read))throw std::invalid_argument("Resident FFN scratch aliases an original input");
        for(const auto& output:public_writes)if(!disjoint(span,output))throw std::invalid_argument("Resident FFN scratch aliases a public output");
        scratch.push_back({name,value,workspace_->retainBuffer(name,0,bytes),bytes});return value;
    };
    if(ordinary)
    {
        // These are the existing expert-indexed producer destinations, not a
        // second table or a new allocation. Retain their original named bytes.
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE));
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE));
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE));
    }
    const size_t intermediate_values=std::max(size_t(request.top_k)*F,size_t(shared_f));
    args.hidden_q8=static_cast<int8_t*>(bind(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,H));
    args.hidden_scales=static_cast<float*>(bind(MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,size_t(H/32)*sizeof(float)));
    args.gate_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
        intermediate_values*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.up_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
        intermediate_values*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.down_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::DOWN_PARTIALS,
        size_t(request.top_k)*H*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.swiglu_q8=static_cast<int8_t*>(bind(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,intermediate_values));
    args.swiglu_scales=static_cast<float*>(bind(MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,intermediate_values/32*sizeof(float)));
    const rocm::ResidentMoEFFNPolicyMasks masks{
        .routed_gate=gate.policy_codebook_mask,.routed_down=down.policy_codebook_mask,
        .shared_gate=shared_gate.policy_codebook_mask,.shared_down=shared_down.policy_codebook_mask,
        .execution={routed_gate_envelope.execution_codebook_mask,routed_down_envelope.execution_codebook_mask,
            shared_gate.execution_codebook_mask,shared_down.execution_codebook_mask}};
    // Reuse the original immutable descriptor publication infrastructure and
    // admitted slots. These are the same singleton shared tables the staged
    // SharedExpertFFNStage prepares; neither weights nor allocation authority
    // are duplicated. Appending these host vectors can invalidate gate/down
    // references above, so all original scalars/pointers are already copied.
    ResidentFFNSharedTables shared_tables;
    shared_tables.gate_up=uploadGroupedExpertGateUpDescriptorTables(&request.shared.gate,&request.shared.up,1,H,shared_f);
    shared_tables.down=uploadGroupedExpertDownDescriptorTable(&request.shared.down,1,H,shared_f);
    if(shared_tables.gate_up<0||shared_tables.down<0)
        throw std::runtime_error("Resident FFN could not publish its original immutable shared descriptors");
    const auto& shared_gate_table=grouped_gateup_desc_tables_[shared_tables.gate_up];
    const auto& shared_down_table=grouped_down_desc_tables_[shared_tables.down];
    args.shared_gate=shared_gate_table.device_gate_descs;args.shared_up=shared_gate_table.device_up_descs;
    args.shared_down=shared_down_table.device_descs;
    const auto retain=[&](const char* name,std::size_t slot) {
        const auto bytes=workspace_->getBufferSize(name);
        const auto stride=(bytes/MoEWorkspaceBuffers::kGroupedDescriptorTableSlots/sizeof(DeviceNativeQuantisedMatrixDesc))*
            sizeof(DeviceNativeQuantisedMatrixDesc);
        if(stride<sizeof(DeviceNativeQuantisedMatrixDesc)||slot>=MoEWorkspaceBuffers::kGroupedDescriptorTableSlots)
            throw std::logic_error("Resident FFN original immutable descriptor slot is incomplete");
        return workspace_->retainBuffer(name,slot*stride,sizeof(DeviceNativeQuantisedMatrixDesc));
    };
    shared_tables.leases={retain(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,shared_gate_table.workspace_slot),
        retain(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,shared_gate_table.workspace_slot),
        retain(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,shared_down_table.workspace_slot)};
    if(router)
    {
        // Shared descriptor registration above may grow the owning vectors.
        // Use the already copied geometry, never the earlier gate-table reference.
        const auto& publication=requirePairedRouterDescriptors(request.runtime,args.experts,request.top_k);
        if(!ensureRouteBufferCapacity(args.experts))throw std::logic_error("Native-layout router logits arena is unavailable");
        bind(MoEWorkspaceBuffers::ROUTE_LOGITS,std::size_t(args.experts)*sizeof(float));
        for(const auto* name:{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
            MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE})
            bind(name,workspace_->getBufferSize(name));
        rocm::NativeLayoutRouterFFNArgs combined;
        auto barrier_slot=workspace_->acquirePersistentSlot(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
        if(!barrier_slot)throw std::logic_error("Native-layout barrier already has a retained owner");
        const auto barrier_bytes=prefix?rocm::fullNativeHCFFNSlotBytes(prefix->implementation):NativeLayoutBarrierLayout::bytes;
        const auto barrier_bank_bytes=workspace_->getBufferSize(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER);
        if(barrier_bank_bytes<barrier_bytes*MoEWorkspaceBuffers::kGroupedDescriptorTableSlots)
            throw std::invalid_argument("Native-layout barrier bank is smaller than its admitted slot payloads");
        // Slot addresses use the actual merged bank stride, not this caller's
        // payload width. A standalone 16-byte owner can occupy a late slot in a
        // bank enlarged for 56-byte full owners, so retain the actual named extent.
        bind(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,barrier_bank_bytes);
        auto* barrier_base=static_cast<std::uint32_t*>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
            barrier_slot->slot(),barrier_bytes));
        if(!barrier_base)throw std::logic_error("Native-layout barrier slot is unavailable");
        combined.barrier_state=barrier_base+(prefix?FullNativeHCFFNBarrierLayout::suffix_word_offset:0);
        const auto barrier_offset=reinterpret_cast<const std::byte*>(barrier_base)-
            static_cast<const std::byte*>(workspace_->getBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER));
        const auto barrier_lease=workspace_->retainBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,barrier_offset,barrier_bytes);
        if(!prefix)TransferEngine::instance().initializeWorkspaceMetadata(*barrier_lease,
            std::vector<std::byte>(barrier_bytes,std::byte{0}),stream);
        combined.ffn=args;combined.router_weights=router->gpu_data_ptr();combined.router_logits=d_route_logits_;
        combined.router_indices=static_cast<float*>(indices->gpu_data_ptr());
        combined.router_weights_output=static_cast<float*>(weights->gpu_data_ptr());
        combined.router_publication=publication.outputs;combined.normalize_router_weights=normalize;combined.write_router_outputs=true;
        if(router->size_bytes()!=std::size_t(args.experts)*H*sizeof(uint16_t)||
            indices->size_bytes()<std::size_t(request.top_k)*sizeof(float)||weights->size_bytes()<std::size_t(request.top_k)*sizeof(float)||
            !combined.router_weights||!combined.router_indices||!combined.router_weights_output)
            throw std::invalid_argument("Native-layout router original tensor extents are invalid");
        const std::array router_writes{Span{combined.router_indices,std::size_t(request.top_k)*sizeof(float)},
            Span{combined.router_weights_output,std::size_t(request.top_k)*sizeof(float)},
            Span{combined.router_logits,std::size_t(args.experts)*sizeof(float)}};
        for(std::size_t i=0;i<router_writes.size();++i)
        {
            if(!disjoint(router_writes[i],Span{combined.router_weights,router->size_bytes()}))
                throw std::invalid_argument("Native-layout router output aliases its original router weights");
            for(const auto& read:public_reads)if(!disjoint(router_writes[i],read))
                throw std::invalid_argument("Native-layout router output aliases original input");
            for(const auto& output:public_writes)if(!disjoint(router_writes[i],output))
                throw std::invalid_argument("Native-layout router output aliases FFN output");
            for(std::size_t j=0;j<i;++j)if(!disjoint(router_writes[i],router_writes[j]))
                throw std::invalid_argument("Native-layout router public outputs alias");
        }
        for(const auto& output:public_writes)
            if(!disjoint(output,Span{combined.router_weights,router->size_bytes()}))
                throw std::invalid_argument("Native-layout FFN output aliases original router weights");
        // Original source-policy tables are resolved by the same native authority.
        // The full branch admits only the actual complete function, never a
        // standalone HC or suffix function before replacing that admission.
        if(prefix)
        {
            rocm::FullNativeHCFFNArgs full{combined,*prefix};
            full.prefix.barrier_state=barrier_base;
            const auto admission=rocm::prepareFullNativeHCFFN(device,full,masks,workspace_->id());
            validateFullNativeHCFFNPrefixBinding(full.prefix,full.suffix);
            validateFullNativeHCFFNStorage(full.prefix,full.suffix,scratch,*workspace_);
            TransferEngine::instance().initializeWorkspaceMetadata(*barrier_lease,
                std::vector<std::byte>(barrier_bytes,std::byte{0}),stream);
            return std::make_unique<PreparedResidentFFN>(*this,request,full.suffix.ffn,admission,stream,std::move(scratch),
                std::move(shared_tables),full.suffix,std::array<TensorBase*,3>{router,indices,weights},
                std::move(barrier_slot),full.prefix);
        }
        const auto admission=rocm::prepareNativeLayoutRouterFFN(device,combined,masks,workspace_->id());
        return std::make_unique<PreparedResidentFFN>(*this,request,combined.ffn,admission,stream,std::move(scratch),
            std::move(shared_tables),combined,std::array<TensorBase*,3>{router,indices,weights},std::move(barrier_slot));
    }
    const auto admission=rocm::prepareResidentMoEFFN(device,args,masks,workspace_->id());
    return std::make_unique<PreparedResidentFFN>(*this,request,args,admission,stream,std::move(scratch),std::move(shared_tables));
}

} // namespace llaminar2
