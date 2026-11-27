/**
 * @file MoEExpertComputeStage.cpp
 * @brief MoEExpertComputeStage core lifecycle, trace authority, and primary execution paths.
 */
#include "MoEExpertComputeStageInternal.h"

namespace llaminar2
{
    namespace mec_detail
    {
        uint64_t nextPrefillAssignmentTraceSequence()
        {
            static std::atomic<uint64_t> sequence{0};
            return sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
        }

        bool tracePrefillAssignmentRuntime(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const char *tag,
            DeviceMoELayerRuntime *device_runtime,
            int layer_idx,
            int seq_len,
            int num_experts,
            int top_k)
        {
            if (!tracePrefillAssignmentEnabled() ||
                !tracePrefillAssignmentLayerMatches(layer_idx) ||
                !tracePrefillAssignmentCheckpointMatches(tag))
            {
                return true;
            }

            /*
             * Assignment diagnostics are host-readback diagnostics, not graph
             * nodes. Native CUDA/HIP capture must retain a fully device-owned
             * execution path, so the request-boundary diagnostic publication
             * owns any later host inspection. Returning here is not an
             * execution fallback: the production kernels and graph continue
             * unchanged, and PerfStats records that this optional observation
             * point was intentionally deferred.
             */
            if (isGraphCaptureActive())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_llep_prefill_assignment_trace_deferred",
                    1.0,
                    "moe",
                    device.to_string(),
                    {{"tag", tag ? tag : "<null>"},
                     {"layer", std::to_string(layer_idx)},
                     {"reason", "graph_capture_device_owned"}});
                return true;
            }

            if (!device.is_gpu())
                return true;
            if (!backend || !stream || !device_runtime)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace requires backend, stream, and runtime"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " backend=" << static_cast<void *>(backend)
                          << " stream=" << stream
                          << " runtime=" << static_cast<void *>(device_runtime));
                return false;
            }

            DeviceMoELayerRuntime runtime{};
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    device_runtime,
                    &runtime,
                    sizeof(runtime),
                    "runtime"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy runtime"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t route_count =
                static_cast<uint64_t>(std::max(0, seq_len)) *
                static_cast<uint64_t>(std::max(0, top_k));
            if (route_count > static_cast<uint64_t>(runtime.prefill_route_capacity))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace route capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " route_count=" << route_count
                          << " capacity=" << runtime.prefill_route_capacity);
                return false;
            }

            const uint32_t expert_count =
                std::min<uint32_t>(
                    static_cast<uint32_t>(std::max(0, num_experts)),
                    kDeviceMoEMaxExperts);
            std::vector<int32_t> route_experts(static_cast<size_t>(route_count));
            std::vector<float> route_weights(static_cast<size_t>(route_count));
            std::vector<int32_t> route_participants(static_cast<size_t>(route_count));
            std::vector<int32_t> expert_counts(expert_count);
            std::vector<least_loaded_ep::LeastLoadedExpertAssignmentSpan> spans;
            std::vector<least_loaded_ep::LeastLoadedExpertWeightTransfer> transfers;

            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_expert_ids,
                    route_experts.data(),
                    route_experts.size() * sizeof(int32_t),
                    "route_expert_ids") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_weights,
                    route_weights.data(),
                    route_weights.size() * sizeof(float),
                    "route_weights") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_participant_ids,
                    route_participants.data(),
                    route_participants.size() * sizeof(int32_t),
                    "route_participant_ids") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.expert_counts,
                    expert_counts.data(),
                    expert_counts.size() * sizeof(int32_t),
                    "expert_counts"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy route metadata"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t span_count = runtime.reserved_u64[2];
            const uint64_t span_capacity = runtime.reserved_u64[0];
            if (span_count > span_capacity)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace span capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " span_count=" << span_count
                          << " span_capacity=" << span_capacity);
                return false;
            }
            spans.resize(static_cast<size_t>(span_count));
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.reserved_ptrs[1],
                    spans.data(),
                    spans.size() * sizeof(least_loaded_ep::LeastLoadedExpertAssignmentSpan),
                    "llep_assignment_spans"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy assignment spans"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t transfer_count = runtime.reserved_u64[3];
            const uint64_t transfer_capacity = runtime.reserved_u64[1];
            if (transfer_count > transfer_capacity)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace transfer capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " transfer_count=" << transfer_count
                          << " transfer_capacity=" << transfer_capacity);
                return false;
            }
            transfers.resize(static_cast<size_t>(transfer_count));
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.reserved_ptrs[2],
                    transfers.data(),
                    transfers.size() * sizeof(least_loaded_ep::LeastLoadedExpertWeightTransfer),
                    "llep_weight_transfers"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy weight transfers"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            if (!backend->synchronizeStream(stream, device.toKernelDeviceIndex()))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace stream synchronization failed"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            std::array<uint64_t, kDeviceMoEMaxParticipants> participant_load{};
            uint64_t invalid_participant_rows = 0;
            for (int32_t participant : route_participants)
            {
                if (participant >= 0 &&
                    static_cast<size_t>(participant) < participant_load.size())
                {
                    ++participant_load[static_cast<size_t>(participant)];
                }
                else
                {
                    ++invalid_participant_rows;
                }
            }

            /*
             * Reconstruct the counterfactual StaticOwner load from the same
             * immutable router output and placement bank.  The final
             * participant row is already available above, but without this
             * baseline an exactly balanced LLEP result says nothing about how
             * much critical-path work it actually removed.  This remains
             * inside the explicitly enabled diagnostic path; production graph
             * execution never downloads or mirrors router assignments.
             */
            std::array<uint64_t, kDeviceMoEMaxParticipants>
                standard_participant_load{};
            uint64_t invalid_standard_rows = 0;
            if (runtime.active_bank <= 1u)
            {
                const auto &bank = runtime.banks[runtime.active_bank];
                for (int32_t expert : route_experts)
                {
                    if (expert < 0 ||
                        static_cast<uint32_t>(expert) >= expert_count)
                    {
                        ++invalid_standard_rows;
                        continue;
                    }
                    const int32_t owner =
                        bank.experts[static_cast<size_t>(expert)]
                            .owner_participant;
                    if (owner < 0 ||
                        static_cast<size_t>(owner) >=
                            standard_participant_load.size())
                    {
                        ++invalid_standard_rows;
                        continue;
                    }
                    ++standard_participant_load[static_cast<size_t>(owner)];
                }
            }
            else
            {
                invalid_standard_rows = route_experts.size();
            }
            uint64_t standard_load_min = 0;
            uint64_t standard_load_max = 0;
            uint64_t standard_load_spread = 0;
            uint64_t assigned_load_min = 0;
            uint64_t assigned_load_max = 0;
            uint64_t assigned_load_spread = 0;
            least_loaded_ep::summarizeParticipantLoads(
                standard_participant_load.data(),
                runtime.participant_count,
                &standard_load_min,
                &standard_load_max,
                &standard_load_spread);
            least_loaded_ep::summarizeParticipantLoads(
                participant_load.data(),
                runtime.participant_count,
                &assigned_load_min,
                &assigned_load_max,
                &assigned_load_spread);

            uint64_t expert_count_sum = 0;
            int32_t max_expert_count = 0;
            uint32_t nonzero_experts = 0;
            for (int32_t count : expert_counts)
            {
                if (count > 0)
                {
                    expert_count_sum += static_cast<uint64_t>(count);
                    max_expert_count = std::max(max_expert_count, count);
                    ++nonzero_experts;
                }
            }

            std::vector<uint32_t> placement_semantics;
            placement_semantics.reserve(expert_count * 4u);
            if (runtime.active_bank <= 1u)
            {
                const auto &bank = runtime.banks[runtime.active_bank];
                for (uint32_t expert = 0; expert < expert_count; ++expert)
                {
                    placement_semantics.push_back(static_cast<uint32_t>(
                        std::max(0, bank.experts[expert].owner_participant)));
                    placement_semantics.push_back(bank.resident_participant_mask[expert]);
                    placement_semantics.push_back(static_cast<uint32_t>(bank.local_compute_mask[expert]));
                    placement_semantics.push_back(static_cast<uint32_t>(bank.replica_role[expert]));
                }
            }

            LOG_INFO("[MoEExpertComputeStage] prefill assignment trace"
                     << " tag=" << (tag ? tag : "<null>")
                     << " device=" << device.to_string()
                     << " layer=" << layer_idx
                     << " seq_len=" << seq_len
                     << " top_k=" << top_k
                     << " route_count=" << route_count
                     << " active_bank=" << runtime.active_bank
                     << " active_epoch=" << runtime.active_epoch
                     << " participant_id=" << runtime.participant_id
                     << " participant_count=" << runtime.participant_count
                     << " expert_count=" << runtime.expert_count
                     << " route_expert_hash=" << hashTraceVector(route_experts)
                     << " route_weight_hash=" << hashTraceVector(route_weights)
                     << " route_participant_hash=" << hashTraceVector(route_participants)
                     << " expert_counts_hash=" << hashTraceVector(expert_counts)
                     << " spans_hash=" << hashTraceVector(spans)
                     << " transfers_hash=" << hashTraceVector(transfers)
                     << " placement_hash=" << hashPlacementBankSemantics(runtime, expert_count)
                     << " placement_owner_mask_hash=" << hashTraceVector(placement_semantics)
                     << " span_count=" << span_count
                     << " span_capacity=" << span_capacity
                     << " transfer_count=" << transfer_count
                     << " transfer_capacity=" << transfer_capacity
                     << " expert_count_sum=" << expert_count_sum
                     << " max_expert_count=" << max_expert_count
                     << " nonzero_experts=" << nonzero_experts
                     << " invalid_participants=" << invalid_participant_rows
                     << " invalid_standard_rows=" << invalid_standard_rows
                     << " standard_load_min=" << standard_load_min
                     << " standard_load_max=" << standard_load_max
                     << " standard_load_spread=" << standard_load_spread
                     << " assigned_load_min=" << assigned_load_min
                     << " assigned_load_max=" << assigned_load_max
                     << " assigned_load_spread=" << assigned_load_spread
                     << " standard_p0=" << standard_participant_load[0]
                     << " standard_p1=" << standard_participant_load[1]
                     << " standard_p2=" << standard_participant_load[2]
                     << " standard_p3=" << standard_participant_load[3]
                     << " p0=" << participant_load[0]
                     << " p1=" << participant_load[1]
                     << " p2=" << participant_load[2]
                     << " p3=" << participant_load[3]);

            /*
             * A transferred expert is useful only when its published runtime
             * descriptor names the destination participant's stable VRAM slot.
             * Correct payload bytes can otherwise mask a remote-pointer
             * publication bug until decode becomes PCIe-bound. Keep the exact
             * pointer provenance in the opt-in trace so one diagnostic pass can
             * distinguish transfer cost from post-transfer compute cost.
             */
            if (tag &&
                std::strcmp(tag, "after_llep_transfer_apply") == 0 &&
                runtime.active_bank <= 1u)
            {
                const auto &bank = runtime.banks[runtime.active_bank];
                for (const auto &transfer : transfers)
                {
                    if (transfer.expert >= expert_count)
                        continue;
                    const auto &descriptor = bank.experts[transfer.expert];
                    LOG_INFO("[MoEExpertComputeStage] prefill transfer descriptor trace"
                             << " device=" << device.to_string()
                             << " layer=" << layer_idx
                             << " expert=" << transfer.expert
                             << " source_participant=" << transfer.source_participant
                             << " destination_participant=" << transfer.destination_participant
                             << " runtime_participant=" << runtime.participant_id
                             << " owner_participant=" << descriptor.owner_participant
                             << " local_slot=" << descriptor.local_slot
                             << " flags=" << descriptor.flags
                             << " resident_mask="
                             << bank.resident_participant_mask[transfer.expert]
                             << " local_compute="
                             << static_cast<uint32_t>(
                                    bank.local_compute_mask[transfer.expert])
                             << " gate_payload="
                             << static_cast<const void *>(descriptor.gate.payload)
                             << " gate_scales=" << descriptor.gate.scales
                             << " up_payload="
                             << static_cast<const void *>(descriptor.up.payload)
                             << " up_scales=" << descriptor.up.scales
                             << " down_payload="
                             << static_cast<const void *>(descriptor.down.payload)
                             << " down_scales=" << descriptor.down.scales);
                }
            }
            return true;
        }

        bool tracePrefillLLEPStatus(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const char *tag,
            const DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceApplyStatus *apply_status)
        {
            if (!tracePrefillLLEPStatusEnabled() ||
                !backend ||
                !device.is_gpu() ||
                !stream ||
                !status ||
                !apply_status)
            {
                return true;
            }

            /*
             * Host readback and stream synchronization are intentionally
             * forbidden while CUDA/HIP records a graph. The status buffers are
             * already persistent device-owned graph state and are exported by
             * the request-boundary PerfStats readback after replay. Attempting
             * to inspect them here poisons native stream capture before the
             * first LLEP event edge is recorded.
             */
            if (isGraphCaptureActive())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_llep_prefill_status_trace_deferred",
                    1.0,
                    "moe",
                    device.to_string(),
                    {{"tag", tag ? tag : "<null>"},
                     {"reason", "graph_capture_device_owned"}});
                return true;
            }

            DeviceMoERebalanceStatus host_status{};
            DeviceMoERebalanceApplyStatus host_apply_status{};
            const int device_ordinal = device.toKernelDeviceIndex();
            const bool copied =
                backend->deviceToHostOnStream(
                    &host_status,
                    status,
                    sizeof(host_status),
                    device_ordinal,
                    stream) &&
                backend->deviceToHostOnStream(
                    &host_apply_status,
                    apply_status,
                    sizeof(host_apply_status),
                    device_ordinal,
                    stream) &&
                backend->synchronizeStream(stream, device_ordinal);
            if (!copied)
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to trace prefill LLEP status"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string());
                return false;
            }

            LOG_INFO("[MoEExpertComputeStage] prefill LLEP status"
                     << " tag=" << (tag ? tag : "<null>")
                     << " device=" << device.to_string()
                     << " transfer_status=" << host_status.status_code
                     << " planned=" << host_status.planned_arrivals
                     << " transfer_count=" << host_status.llep_weight_transfer_count
                     << " span_count=" << host_status.llep_assignment_span_count
                     << " candidates=" << host_status.candidate_arrivals_considered
                     << " invalid_runtime=" << host_status.invalid_runtime_layers
                     << " standard_ep=" << host_status.llep_standard_ep_selected
                     << " skipped_balanced=" << host_status.llep_skipped_balanced
                     << " skipped_spread=" << host_status.llep_skipped_insufficient_spread_improvement
                     << " skipped_foreign=" << host_status.llep_skipped_insufficient_foreign_rows
                     << " plan_overflow=" << host_status.plan_overflow
                     << " payload_overflow=" << host_status.payload_bucket_overflow
                     << " payload_requested=" << host_status.payload_bucket_requested_slots
                     << " payload_slots=" << host_status.payload_bucket_slots
                     << " payload_bucket_index=" << host_status.payload_bucket_index
                     << " source_mask=" << host_status.payload_source_participant_mask
                     << " destination_mask=" << host_status.payload_destination_participant_mask
                     << " apply_status=" << host_apply_status.status_code
                     << " seen=" << host_apply_status.plan_entries_seen
                     << " applied=" << host_apply_status.applied_arrivals
                     << " required_local=" << host_apply_status.required_local_arrivals
                     << " ready_local=" << host_apply_status.ready_local_arrivals
                     << " copied=" << host_apply_status.copied_arrivals
                     << " invalid=" << host_apply_status.invalid_plan_entries
                     << " missing_src=" << host_apply_status.missing_source_descriptors
                     << " missing_dst=" << host_apply_status.missing_destination_slots
                     << " mismatch=" << host_apply_status.descriptor_mismatches
                     << " copy_incomplete=" << host_apply_status.copy_incomplete
                     << " changed_layers=" << host_apply_status.changed_layers);
            return true;
        }
    } // namespace mec_detail


    CPUGroupedMoESerialWorkspace::InvocationLease::~InvocationLease()
    {
        release();
    }

    CPUGroupedMoESerialWorkspace::InvocationLease::InvocationLease(
        InvocationLease &&other) noexcept
        : owner_(std::exchange(other.owner_, nullptr))
    {
    }

    CPUGroupedMoESerialWorkspace::InvocationLease &
    CPUGroupedMoESerialWorkspace::InvocationLease::operator=(
        InvocationLease &&other) noexcept
    {
        if (this != &other)
        {
            release();
            owner_ = std::exchange(other.owner_, nullptr);
        }
        return *this;
    }

    void CPUGroupedMoESerialWorkspace::InvocationLease::release() noexcept
    {
        if (!owner_)
            return;
        owner_->invocation_active_.clear(std::memory_order_release);
        owner_ = nullptr;
    }

    CPUGroupedMoESerialWorkspace::StorageLayout
    CPUGroupedMoESerialWorkspace::storageLayout(const Config &config)
    {
        // The admitted enclosing declaration owns physical placement and
        // initialization. Rebuild every execution extent independently while
        // retaining that exact policy; a default heap declaration would differ
        // from the original NUMA fragment even when all logical widths match.
        const auto placement = config.declared_storage
            ? config.declared_storage->batch.placement() : CPUStoragePlacement::local();
        const auto initialization = config.declared_storage
            ? config.declared_storage->batch.initialization() : CPUStorageInitialization::Zero;
        return StorageLayout::declare(config.row_capacity, config.d_model,
            config.expert_intermediate, config.num_experts, config.routing_top_k, config.device_id,
            placement, initialization);
    }

    size_t CPUGroupedMoESerialWorkspace::plannedAllocationBytes(const Config &config)
    {
        const auto authored = storageLayout(config);
        if (config.declared_storage && *config.declared_storage != authored)
            throw std::logic_error("CPU grouped-MoE declared storage differs from execution geometry");
        return config.declared_storage ? config.declared_storage->physicalBytes() : authored.physicalBytes();
    }

    CPUGroupedMoESerialWorkspace::CPUGroupedMoESerialWorkspace(Config config)
        : row_capacity_(config.row_capacity), d_model_(config.d_model),
          expert_intermediate_(config.expert_intermediate), num_experts_(config.num_experts),
          routing_top_k_(config.routing_top_k),
          debug_name_(config.debug_name.empty() ? "cpu_grouped_moe_serial_workspace" : config.debug_name)
    {
        const auto authored = storageLayout(config);
        if (config.declared_storage && *config.declared_storage != authored)
            throw std::logic_error("CPU grouped-MoE declared storage differs from execution geometry");
        const auto &layout = config.declared_storage ? *config.declared_storage : authored;
        if (!config.physical_memory_authority)
            throw std::logic_error("CPU grouped-MoE workspace requires physical-memory admission");
        route_capacity_ = layout.route_capacity;
        const auto &memory = config.physical_memory_authority;
        backing_ = std::make_shared<CPUGroupedMoEStorageBacking>(layout, memory);
        moe_kernel_ = KernelFactory::createMoEKernel(config.device_id);
        auto *const cpu_kernel = dynamic_cast<CPUMoEKernel *>(moe_kernel_.get());
        if (!cpu_kernel)
            throw std::runtime_error("CPU grouped-MoE workspace requires the CPU router kernel");
        cpu_kernel->bindRouterQ8HiddenStorage(backing_->kernel_router_q8);
        allocation_bytes_ = layout.physicalBytes();

        PerfStatsCollector::addCounter(
            "memory",
            "cpu_grouped_moe_serial_workspace_bytes",
            static_cast<double>(allocation_bytes_),
            "model_setup",
            config.device_id.toString(),
            {{"debug_name", debug_name_},
             {"row_capacity", std::to_string(row_capacity_)},
             {"route_capacity", std::to_string(route_capacity_)},
             {"d_model", std::to_string(d_model_)},
             {"expert_intermediate", std::to_string(expert_intermediate_)},
             {"num_experts", std::to_string(num_experts_)},
             {"routing_top_k", std::to_string(routing_top_k_)},
             {"ownership", "per_cpu_participant_serial_graph_family"},
             {"first_touched_during_setup", "true"}});
    }

    CPUGroupedMoESerialWorkspace::~CPUGroupedMoESerialWorkspace()
    {
        if (invocation_active_.test(std::memory_order_acquire))
            std::terminate();
    }

    bool CPUGroupedMoESerialWorkspace::supports(
        int rows,
        int route_width,
        int d_model,
        int expert_intermediate,
        int num_experts) const noexcept
    {
        return rows > 0 && static_cast<size_t>(rows) <= row_capacity_ &&
               route_width > 0 && route_width <= routing_top_k_ &&
               d_model == d_model_ &&
               expert_intermediate == expert_intermediate_ &&
               num_experts == num_experts_;
    }

    CPUGroupedMoESerialWorkspace::InvocationLease
    CPUGroupedMoESerialWorkspace::acquire(
        int rows,
        int route_width,
        int d_model,
        int expert_intermediate,
        int num_experts,
        int layer_idx)
    {
        if (!supports(
                rows, route_width, d_model, expert_intermediate, num_experts))
        {
            throw std::invalid_argument(
                "CPU grouped-MoE invocation exceeds serial workspace plan at layer " +
                std::to_string(layer_idx));
        }
        if (invocation_active_.test_and_set(std::memory_order_acquire))
        {
            throw std::logic_error(
                "CPU grouped-MoE serial workspace was acquired concurrently at layer " +
                std::to_string(layer_idx));
        }
        return InvocationLease(this);
    }

    // =========================================================================
    // MoEExpertComputeStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEExpertComputeStage::MoEExpertComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        /*
         * Capture the immutable upper bound before a serial CPU endpoint starts
         * rebinding smaller live route widths. GPU graph families never mutate
         * this scalar, but sharing the proof makes the binding API total for
         * both ownership regimes.
         */
        if (params_.cpu_joint_layer) {
            if(params_.prepared_engine_binding_lifecycle==MoEPreparedEngineBindingLifecycle::SparseOverlayInvocationPublished) {
            const auto &publication=*params_.cpu_joint_layer;
            if (publication.bankLease() || !publication.sourceSelection() || !publication.family() || publication.layer()!=params_.layer_idx ||
                publication.family()->cohort()!=publication.sourceSelection()->cohort() ||
                publication.family()->participantIndex()!=publication.sourceSelection()->participantIndex())
                throw std::invalid_argument("CPU joint invocation has no original published source family");
            if (!params_.device_id.is_cpu() ||
                params_.expert_weight_resolution_policy!=MoEExpertWeightResolutionPolicy::PreparedRegistryOnly ||
                params_.prepared_engine_binding_lifecycle!=MoEPreparedEngineBindingLifecycle::SparseOverlayInvocationPublished ||
                params_.canonical_route_arithmetic!=MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA ||
                !params_.canonical_route_contributions)
                throw std::invalid_argument("CPU joint expert invocation requires its source-bound packet-published canonical producer");
            const auto &source=*publication.sourceSelection();const auto &geometry=source.ownership().geometry();
            const auto empty=[](const auto& engines){return std::all_of(engines.begin(),engines.end(),[](auto* engine){return engine==nullptr;});};
            if(source.layer()!=params_.layer_idx || geometry.experts!=params_.num_experts || geometry.model_columns!=params_.d_model ||
               source.ownership().projection(WeightRole::MoEExpertUp).rows!=params_.expert_intermediate ||
               params_.expert_mask.size()!=static_cast<size_t>(params_.num_experts) ||
               std::any_of(params_.expert_mask.begin(),params_.expert_mask.end(),[](bool active){return active;}) ||
               !empty(params_.prepared_gate_gemm) || !empty(params_.prepared_up_gemm) || !empty(params_.prepared_down_gemm))
                throw std::invalid_argument("CPU joint expert setup changed original source geometry or supplied epoch engine pointers");
            } else if(params_.prepared_engine_binding_lifecycle==MoEPreparedEngineBindingLifecycle::CompleteAtConstruction) {
                // The generic complete-bank graph remains a distinct explicit
                // cold contract. It cannot adopt a retired bootstrap bank.
            const auto &publication=*params_.cpu_joint_layer;
            const auto construction=publication.acquireInitialConstruction();
            if (!publication.sourceSelection() || !construction || publication.layer()!=params_.layer_idx)
                throw std::invalid_argument("CPU joint invocation has no original published layer");
            const auto &lease=construction->layerForModelLayer(params_.layer_idx);
            if (!params_.device_id.is_cpu() ||
                params_.expert_weight_resolution_policy!=MoEExpertWeightResolutionPolicy::PreparedRegistryOnly ||
                params_.canonical_route_arithmetic!=MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA ||
                !params_.canonical_route_contributions)
                throw std::invalid_argument("CPU joint expert invocation requires its source-bound unweighted canonical producer");
            const auto &source=*publication.sourceSelection();
            const auto &geometry=source.ownership().geometry();
            if (source.layer()!=params_.layer_idx || geometry.experts!=params_.num_experts ||
                geometry.model_columns!=params_.d_model ||
                source.ownership().projection(WeightRole::MoEExpertUp).rows!=params_.expert_intermediate ||
                params_.expert_mask.size()!=static_cast<size_t>(params_.num_experts) ||
                lease.fragments.size()!=static_cast<size_t>(params_.num_experts) ||
                params_.prepared_gate_gemm.size()!=lease.fragments.size() ||
                params_.prepared_up_gemm.size()!=lease.fragments.size() || params_.prepared_down_gemm.size()!=lease.fragments.size())
                throw std::invalid_argument("CPU joint expert invocation changed original source/member geometry");
            for (int expert=0;expert<params_.num_experts;++expert) {
                const auto index=static_cast<size_t>(expert);
                const auto &engines=lease.fragments[index];
                if ((params_.expert_mask[index] && !source.contains(expert)) ||
                    params_.prepared_gate_gemm[index]!=engines.gate().get() ||
                    params_.prepared_up_gemm[index]!=engines.up().get() ||
                    params_.prepared_down_gemm[index]!=engines.down().get() ||
                    (source.contains(expert) && !engines.valid()))
                    throw std::invalid_argument("CPU joint expert invocation replaced an original selected native engine");
            }
            } else {
                throw std::invalid_argument("Unknown CPU joint prepared-engine binding lifecycle");
            }
        }
        sparse_overlay_route_width_capacity_ = params_.top_k;
        cpu_workspace_row_capacity_ = params_.seq_len;
        switch (params_.canonical_route_diagnostic_publication)
        {
        case MoECanonicalRouteDiagnosticPublication::CompleteBank:
            break;
        case MoECanonicalRouteDiagnosticPublication::ContinuationLocalBank:
            if (!params_.canonical_route_contributions)
                throw std::invalid_argument("Local canonical route diagnostics require their authored bank");
            break;
        default:
            throw std::invalid_argument("Unknown canonical route diagnostic publication identity");
        }
        if (params_.expert_weight_resolution_policy ==
                MoEExpertWeightResolutionPolicy::PreparedRegistryOnly &&
            (params_.gate_exps || params_.up_exps || params_.down_exps))
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] registry-only expert resolution forbids raw parent bindings");
        }

        if (params_.routed_row_execution_policy ==
            RoutedExpertRowExecutionPolicy::FullyReplicatedLocal)
        {
            if (!hasFullLocalExpertOwnership() || !expertMaskAllEnabled())
            {
                throw std::invalid_argument(
                    "[MoEExpertComputeStage] fully replicated local row execution "
                    "requires every logical expert to be locally executable");
            }
            if (params_.canonical_route_contributions)
            {
                throw std::invalid_argument(
                    "[MoEExpertComputeStage] fully replicated local row execution "
                    "forbids canonical route contribution publication");
            }
        }

        if (params_.device_id.is_gpu() &&
            params_.routed_assignment_policy ==
                RoutedExpertAssignmentPolicy::LeastLoadedResident &&
            params_.prefill_llep_tp_ctx &&
            params_.prefill_llep_assignment_mode ==
                PrefillLLEPAssignmentMode::LogicalPositionResidentOnly &&
            !params_.absolute_position_ids_device)
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] logical-position resident LLEP "
                "requires the graph-local device position row");
        }

        if (params_.device_id.is_gpu() &&
            params_.seq_len > 1 &&
            params_.routed_assignment_policy ==
                RoutedExpertAssignmentPolicy::LeastLoadedResident &&
            params_.prefill_llep_tp_ctx &&
            params_.prefill_llep_assignment_mode ==
                PrefillLLEPAssignmentMode::LogicalPositionResidentOnly &&
            !params_.active_row_count_device)
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] logical-position resident LLEP "
                "requires the graph-local device active-row count");
        }

        if (params_.combine_shared_expert_in_verifier &&
            params_.shared_gate_inp &&
            params_.shared_gate_inp->native_type() != TensorType::FP32)
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] GPU combined verifier shared-gate input "
                "must be normalized to FP32 by model-weight preparation");
        }

        /*
         * Runtime placement publication indexes these tables by global expert
         * id even when a participant currently owns no experts. Full model
         * builders normally pre-size the vectors while preparing weights, but
         * placement is a stage-level contract and must not depend on that
         * incidental builder ordering. A fixed-size null table also lets a
         * newly built graph adopt engines from PreparedWeightStore without
         * allocating or reshaping metadata in the token hot path.
         */
        const size_t expert_count =
            static_cast<size_t>(std::max(0, params_.num_experts));
        params_.prepared_gate_gemm.resize(expert_count, nullptr);
        params_.prepared_up_gemm.resize(expert_count, nullptr);
        params_.prepared_down_gemm.resize(expert_count, nullptr);
        // This existing CPU schedule also lends completed expert IDs to metrics.
        // Reserve its same bounded capacity during construction, not per packet.
        if (params_.device_id.is_cpu())
            cpu_grouped_active_experts_.reserve(expert_count);
        const bool invocation_published_registry_bank =
            params_.prepared_engine_binding_lifecycle ==
            MoEPreparedEngineBindingLifecycle::
                SparseOverlayInvocationPublished;
        const bool empty_explicit_setup_mask =
            params_.expert_mask.size() == expert_count &&
            std::none_of(
                params_.expert_mask.begin(),
                params_.expert_mask.end(),
                [](bool enabled) { return enabled; });
        if (invocation_published_registry_bank &&
            (params_.expert_weight_resolution_policy !=
                 MoEExpertWeightResolutionPolicy::PreparedRegistryOnly ||
             !params_.device_id.is_cpu() ||
             params_.cpu_router_q8_input_publication !=
                 CPURouterQ8InputPublicationPolicy::PublishTransportedRows ||
             !params_.cpu_grouped_serial_workspace ||
             !empty_explicit_setup_mask))
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] invocation-published prepared engines require an empty setup-primed serial CPU ExpertOverlay endpoint");
        }
        if (params_.cpu_grouped_serial_workspace &&
            (!params_.device_id.is_cpu() ||
             params_.cpu_router_q8_input_publication !=
                 CPURouterQ8InputPublicationPolicy::PublishTransportedRows ||
             !params_.cpu_grouped_serial_workspace->supports(
                 params_.seq_len,
                 params_.top_k,
                 params_.d_model,
                 params_.expert_intermediate,
                 params_.num_experts)))
        {
            throw std::invalid_argument(
                "CPU grouped-MoE serial workspace does not match the transported overlay stage");
        }
        if (params_.device_id.is_cpu() &&
            params_.cpu_router_q8_input_publication ==
                CPURouterQ8InputPublicationPolicy::PublishTransportedRows)
        {
            if (params_.seq_len <= 0)
            {
                throw std::invalid_argument(
                    "CPU transported-row expert execution requires a positive setup row capacity");
            }
            sparse_overlay_cpu_compact_to_source_row_.resize(
                static_cast<size_t>(params_.seq_len), -1);
        }
        if (params_.expert_weight_resolution_policy ==
                MoEExpertWeightResolutionPolicy::PreparedRegistryOnly &&
            !invocation_published_registry_bank &&
            !hasPreparedExpertGemmEnginesForLocalOwnership())
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] registry-only expert resolution requires a complete prepared triplet for every locally executable expert");
        }
        invalidateFixedTopologyMaskPublication();

        DeviceMoEOverlayServiceTelemetryBinding service_telemetry =
            params_.overlay_service_telemetry;
        if (!service_telemetry.empty() && !service_telemetry.valid())
        {
            throw std::invalid_argument(
                "[MoEExpertComputeStage] Sparse service telemetry binding is partial");
        }

        if (params_.moe_runtime_table && params_.layer_idx >= 0)
        {
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            const auto table_telemetry =
                params_.moe_runtime_table
                    ->deviceOverlayServiceTelemetryBinding(
                        params_.layer_idx);
            if (!table_telemetry.empty())
            {
                if (!service_telemetry.empty() &&
                    !(service_telemetry == table_telemetry))
                {
                    throw std::invalid_argument(
                        "[MoEExpertComputeStage] Explicit service telemetry disagrees with the runtime table");
                }
                service_telemetry = table_telemetry;
            }
            /*
             * Construction has no executor-owned stream, so it may inspect an
             * already published bank but must never publish a new one. A cold
             * decode bank is initialized by executeSingleToken() after the
             * executor binds the exact producer stream.
             */
            moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank();
        }
        if (!service_telemetry.empty())
        {
            if (!params_.device_id.is_gpu() || !service_telemetry.valid())
            {
                throw std::invalid_argument(
                    "[MoEExpertComputeStage] Dynamic service telemetry requires one complete GPU observation binding");
            }
            overlay_service_runtime_layer_ =
                service_telemetry.runtime_layer;
            overlay_service_telemetry_layer_ =
                service_telemetry.layer_telemetry;
            overlay_service_sample_ = service_telemetry.sample;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "device_service_stage_bindings",
                1.0,
                "model_setup",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"rows", std::to_string(params_.seq_len)},
                 {"sparse_endpoint",
                  sparse_overlay_invocation_bound_ ? "true" : "false"},
                 {"telemetry_allocation",
                  std::to_string(reinterpret_cast<std::uintptr_t>(
                      overlay_service_telemetry_layer_ -
                      static_cast<std::size_t>(params_.layer_idx) *
                          kDeviceMoEOverlayServicePhaseCount))},
                 {"sample_allocation",
                  std::to_string(reinterpret_cast<std::uintptr_t>(
                      overlay_service_sample_ - params_.layer_idx))}});
        }

        // Ordinary payloads materialize in the finalized workspace domain.
        // Explicit serial endpoints already own the same eight-root backing.

        if (params_.cpu_grouped_serial_workspace)
            bindCPUGroupedStorage(params_.cpu_grouped_serial_workspace->backing_);

    }

    bool MoEExpertComputeStage::usesGraphStableRuntimeDecodePlacement() const
    {
        return supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
               params_.seq_len == 1 &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               params_.moe_runtime_table != nullptr &&
               params_.layer_idx >= 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0;
    }

    bool MoEExpertComputeStage::usesGraphStableFixedTopologyPrefillPlacement() const
    {
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               canUseFixedTopologyGroupedPrefill();
    }

    bool MoEExpertComputeStage::usesGraphStableMoEPlacement() const
    {
        return usesGraphStableRuntimeDecodePlacement() ||
               usesGraphStableFixedTopologyPrefillPlacement();
    }

    bool MoEExpertComputeStage::expertMaskMatches(
        const std::vector<bool> &candidate) const noexcept
    {
        return params_.expert_mask == candidate;
    }

    bool MoEExpertComputeStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        if (params_.combine_shared_expert_in_verifier)
        {
            if (!params_.prepared_store ||
                !params_.shared_gate_w ||
                !params_.shared_up_w ||
                !params_.shared_down_w ||
                !params_.shared_gate_inp ||
                !params_.prepared_shared_ref_gate.has_value() ||
                !params_.prepared_shared_ref_up.has_value() ||
                !params_.prepared_shared_ref_down.has_value())
            {
                return fail(
                    "combined shared verifier requires gate/up/down source weights, "
                    "gate input, PreparedWeightStore, and all prepared refs");
            }
            if (!params_.prepared_store->contains(*params_.prepared_shared_ref_gate) ||
                !params_.prepared_store->contains(*params_.prepared_shared_ref_up) ||
                !params_.prepared_store->contains(*params_.prepared_shared_ref_down))
            {
                return fail(
                    "combined shared verifier PreparedWeightStore is missing a "
                    "gate/up/down ref");
            }
        }

        const bool has_slab_ref = params_.gate_slab_ref.has_value() ||
                                  params_.up_slab_ref.has_value() ||
                                  params_.down_slab_ref.has_value();
        if (!has_slab_ref)
        {
            if (error)
                error->clear();
            return true;
        }

        if (!params_.prepared_store)
            return fail("PreparedWeightStore is required for MoE expert slab refs");
        if (!params_.gate_slab_ref.has_value() ||
            !params_.up_slab_ref.has_value() ||
            !params_.down_slab_ref.has_value())
        {
            return fail("gate/up/down ExpertSlabRefs must be provided together");
        }

        auto check_slab = [&](const char *name, const ExpertSlabRef &ref)
        {
            const auto availability = params_.prepared_store->expertAvailabilityMask(ref);
            if (availability.empty())
                return fail(std::string("PreparedWeightStore does not contain expert slab for ") + name);
            if (params_.num_experts > 0 && availability.size() != static_cast<size_t>(params_.num_experts))
            {
                return fail(std::string("expert slab size mismatch for ") + name +
                            ": got " + std::to_string(availability.size()) +
                            ", expected " + std::to_string(params_.num_experts));
            }
            return true;
        };

        if (!check_slab("gate", *params_.gate_slab_ref))
            return false;
        if (!check_slab("up", *params_.up_slab_ref))
            return false;
        if (!check_slab("down", *params_.down_slab_ref))
            return false;

        if (error)
            error->clear();
        return true;
    }

    bool MoEExpertComputeStage::updateExpertMask(const std::vector<bool> &mask)
    {
        if (static_cast<int>(mask.size()) != params_.num_experts)
        {
            LOG_ERROR("[MoEExpertComputeStage] Expert mask size " << mask.size()
                                                                  << " != num_experts " << params_.num_experts);
            return false;
        }
        params_.expert_mask = mask;
        if (params_.replica_set.num_replicated > 0 && !params_.expert_mask.empty())
            params_.replica_set.buildPrefillMask(params_.my_socket_id, params_.expert_mask, params_.layer_idx);
        else
            params_.replica_set.prefill_mask.clear();
        grouped_gateup_desc_table_dirty_ = true;
        grouped_down_desc_table_dirty_ = true;
        moe_runtime_table_initialized_ = false;
        invalidateFixedTopologyMaskPublication();
        return refreshGraphStablePlacement(/*preserve_capture_ready=*/true);
    }

    ExpertPackedWeights MoEExpertComputeStage::detachPreparedExpert(int expert_id)
    {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::detachPreparedExpert(ctx, expert_id);
    }

    ExpertPackedWeights MoEExpertComputeStage::clonePreparedExpert(
        int expert_id) const
    {
        auto ctx =
            const_cast<MoEExpertComputeStage *>(this)->buildWeightContext();
        return MoEExpertWeightService::clonePreparedExpert(ctx, expert_id);
    }

    ExpertWeightBlobs MoEExpertComputeStage::serializeExpert(int expert_id) const
    {
        auto ctx = const_cast<MoEExpertComputeStage *>(this)->buildWeightContext();
        return MoEExpertWeightService::serializeExpert(ctx, expert_id);
    }

    std::vector<int> MoEExpertComputeStage::transferExpertsGPUDirectFrom(
        MoEExpertComputeStage &source,
        const std::vector<int> &expert_ids,
        void *source_producer_stream)
    {
        std::vector<int> satisfied_expert_ids;
        if (!source_producer_stream)
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer requires an explicit source producer stream"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }

        auto src_ctx = source.buildWeightContext();
        auto dst_ctx = buildWeightContext();
        GpuDirectTransferCompletion completion;
        const bool ok = MoEExpertWeightService::transferExpertsGPUDirect(
            src_ctx,
            dst_ctx,
            expert_ids,
            params_.layer_idx,
            source_producer_stream,
            &satisfied_expert_ids,
            &completion);
        params_.gate_slab_ref = dst_ctx.gate_slab_ref;
        params_.up_slab_ref = dst_ctx.up_slab_ref;
        params_.down_slab_ref = dst_ctx.down_slab_ref;
        if (!satisfied_expert_ids.empty())
        {
            if (completion.valid())
                addPendingGpuDirectTransfer(std::move(completion));
            bindPreparedExpertEnginesForExperts(satisfied_expert_ids);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        if (!ok && satisfied_expert_ids.empty())
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer did not satisfy any requested experts"
                      << " for layer " << params_.layer_idx);
        }
        return satisfied_expert_ids;
    }

    std::vector<int> MoEExpertComputeStage::missingPreparedExpertIds(
        const std::vector<int> &expert_ids) const
    {
        std::vector<int> missing;
        missing.reserve(expert_ids.size());
        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            const size_t idx = static_cast<size_t>(expert_id);
            const bool has_all =
                idx < params_.prepared_gate_gemm.size() &&
                idx < params_.prepared_up_gemm.size() &&
                idx < params_.prepared_down_gemm.size() &&
                params_.prepared_gate_gemm[idx] &&
                params_.prepared_up_gemm[idx] &&
                params_.prepared_down_gemm[idx];
            if (!has_all)
                missing.push_back(expert_id);
        }
        return missing;
    }

    std::vector<int> MoEExpertComputeStage::stageExpertsGPUDirectToTransferSlotsFrom(
        MoEExpertComputeStage &source,
        const std::vector<int> &expert_ids,
        void *source_producer_stream,
        GpuDirectTransferSlotArrivals *staged_arrivals,
        size_t active_arrival_capacity,
        size_t staging_pool_capacity,
        std::vector<std::shared_ptr<GpuExpertTransferStagingPool>> *transfer_staging_pools)
    {
        std::vector<int> satisfied_expert_ids;
        if (staged_arrivals)
            *staged_arrivals = GpuDirectTransferSlotArrivals{};
        if (!source_producer_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer-slot staging requires an explicit source producer stream"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }
        if (!staged_arrivals)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer-slot staging requires an output carrier"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }

        auto src_ctx = source.buildWeightContext();
        auto dst_ctx = buildWeightContext();
        const bool ok = MoEExpertWeightService::stageExpertsGPUDirectToTransferSlots(
            src_ctx,
            dst_ctx,
            expert_ids,
            params_.layer_idx,
            source_producer_stream,
            staged_arrivals,
            &satisfied_expert_ids,
            active_arrival_capacity,
            staging_pool_capacity,
            transfer_staging_pools);
        params_.gate_slab_ref = dst_ctx.gate_slab_ref;
        params_.up_slab_ref = dst_ctx.up_slab_ref;
        params_.down_slab_ref = dst_ctx.down_slab_ref;
        if (!ok && satisfied_expert_ids.empty())
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer-slot staging did not satisfy requested experts"
                      << " for layer " << params_.layer_idx);
        }
        return satisfied_expert_ids;
    }

    std::vector<int> MoEExpertComputeStage::activateGpuDirectTransferSlotArrivals(
        const GpuDirectTransferSlotArrivals &arrivals,
        void *activation_stream,
        GpuDirectTransferCompletion *completion_out,
        bool retain_pending_completion)
    {
        std::vector<int> activated_expert_ids;
        if (arrivals.empty())
            return activated_expert_ids;
        if (!activation_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-slot activation requires an explicit stream"
                      << " for layer " << params_.layer_idx);
            return activated_expert_ids;
        }

        auto ctx = buildWeightContext();
        GpuDirectStagedExpertArrivals activated_arrivals;
        GpuDirectTransferCompletion completion;
        const bool activated = MoEExpertWeightService::activateGpuDirectTransferSlotArrivals(
            ctx,
            arrivals,
            activation_stream,
            &activated_arrivals,
            &completion);
        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        if (!activated)
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to activate GPU-direct transfer slots"
                      << " for layer " << params_.layer_idx
                      << " on " << params_.device_id.to_string());
            return activated_expert_ids;
        }

        if (completion.valid())
            activated_arrivals.completion = completion;

        if (!MoEExpertWeightService::installActivatedGpuDirectArrivals(ctx, activated_arrivals))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to publish activated GPU-direct arrivals"
                      << " for layer " << params_.layer_idx
                      << " on " << params_.device_id.to_string());
            return activated_expert_ids;
        }

        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        activated_expert_ids = activated_arrivals.expertIds();
        if (!activated_expert_ids.empty())
        {
            if (completion_out)
                *completion_out = completion;
            if (retain_pending_completion && completion.valid())
                addPendingGpuDirectTransfer(std::move(completion));
            bindPreparedExpertEnginesForExperts(activated_expert_ids);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        return activated_expert_ids;
    }

    void MoEExpertComputeStage::bindPreparedExpertEnginesForExperts(
        const std::vector<int> &expert_ids)
    {
        auto bind = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            bindStageStream(gemm);
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            if (expert_id < static_cast<int>(params_.prepared_gate_gemm.size()))
                bind(params_.prepared_gate_gemm[expert_id]);
            if (expert_id < static_cast<int>(params_.prepared_up_gemm.size()))
                bind(params_.prepared_up_gemm[expert_id]);
            if (expert_id < static_cast<int>(params_.prepared_down_gemm.size()))
                bind(params_.prepared_down_gemm[expert_id]);
        }
    }

    void MoEExpertComputeStage::addPendingGpuDirectTransfer(
        GpuDirectTransferCompletion completion)
    {
        if (!completion.valid())
            return;
        const void *event_ptr = completion.ready_event.get();
        for (const auto &pending : pending_gpu_direct_transfers_)
        {
            if (pending.ready_event.get() == event_ptr)
                return;
        }
        pending_gpu_direct_transfers_.push_back(std::move(completion));
    }

    void MoEExpertComputeStage::addPendingGpuDirectTransfersFromStore(
        const std::vector<int> &expert_ids)
    {
        if (!params_.prepared_store)
            return;

        auto adopt_from_slab = [&](const std::optional<ExpertSlabRef> &slab_ref,
                                   int expert_id)
        {
            if (!slab_ref.has_value())
                return;
            auto completion =
                params_.prepared_store->expertGpuDirectCompletion(*slab_ref, expert_id);
            if (completion.has_value())
                addPendingGpuDirectTransfer(std::move(*completion));
        };

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            adopt_from_slab(params_.gate_slab_ref, expert_id);
            adopt_from_slab(params_.up_slab_ref, expert_id);
            adopt_from_slab(params_.down_slab_ref, expert_id);
        }
    }

    size_t MoEExpertComputeStage::releaseRawExpertWeights()
    {
        if (raw_weights_released_)
            return 0;

        const auto active_expert = [&](size_t expert)
        {
            if (!params_.expert_mask.empty())
            {
                return expert < params_.expert_mask.size() &&
                       params_.expert_mask[expert];
            }
            const int local_start = std::max(0, params_.local_expert_start);
            const int local_count =
                params_.local_expert_count > 0
                    ? params_.local_expert_count
                    : params_.num_experts;
            return expert >= static_cast<size_t>(local_start) &&
                   expert < static_cast<size_t>(
                                std::min(params_.num_experts,
                                         local_start + local_count));
        };
        const auto table_releases_source =
            [&](const std::vector<ITensorGemm *> &engines)
        {
            if (engines.size() !=
                static_cast<size_t>(std::max(0, params_.num_experts)))
            {
                return false;
            }
            for (size_t expert = 0; expert < engines.size(); ++expert)
            {
                if (!active_expert(expert))
                    continue;
                if (!engines[expert] ||
                    !engines[expert]->canReleaseSourceWeightTensor())
                {
                    return false;
                }
            }
            return true;
        };

        if (!table_releases_source(params_.prepared_gate_gemm) ||
            !table_releases_source(params_.prepared_up_gemm) ||
            !table_releases_source(params_.prepared_down_gemm))
        {
            /*
             * This is not an execution fallback. The prepared engines remain
             * the only compute path; source tensors are retained solely because
             * at least one engine explicitly declares that it still owns a view
             * into those bytes.
             */
            LOG_TRACE("[MoEExpertComputeStage] Retaining raw expert source storage"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string()
                      << " reason=prepared_engine_references_source");
            return 0;
        }

        auto ctx = buildWeightContext();
        size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);
        raw_weights_released_ = true;
        return freed;
    }

    // ── Phased rebalance API ─────────────────────────────────────────────

    std::vector<const TensorBase *> MoEExpertComputeStage::releaseDepartedExperts(
        const std::vector<bool> &new_mask)
    {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    bool MoEExpertComputeStage::registerAndPrepareNewExperts(
        const std::vector<bool> &new_mask,
        const std::unordered_map<int, ExpertWeightBlobs> *received_weights,
        const std::unordered_map<int, PreparedExpertEngines> *
            received_prepared_experts)
    {
        auto ctx = buildWeightContext();
        const bool ok = MoEExpertWeightService::registerAndPrepareNewExperts(
            ctx,
            new_mask,
            received_weights,
            received_prepared_experts);
        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        if (ok)
        {
            std::vector<int> active_experts;
            active_experts.reserve(new_mask.size());
            for (size_t expert_id = 0; expert_id < new_mask.size(); ++expert_id)
            {
                if (new_mask[expert_id])
                    active_experts.push_back(static_cast<int>(expert_id));
            }
            bindPreparedExpertEnginesForExperts(active_experts);
            addPendingGpuDirectTransfersFromStore(active_experts);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        return ok;
    }

    void MoEExpertComputeStage::applyExpertMask(const std::vector<bool> &new_mask)
    {
        clearCompletedCPUActiveExpertView();
        params_.expert_mask = new_mask;
        if (params_.replica_set.num_replicated > 0 && !params_.expert_mask.empty())
            params_.replica_set.buildPrefillMask(params_.my_socket_id, params_.expert_mask, params_.layer_idx);
        else
            params_.replica_set.prefill_mask.clear();
        cached_gate_gemm_ = params_.prepared_gate_gemm;
        cached_up_gemm_ = params_.prepared_up_gemm;
        cached_down_gemm_ = params_.prepared_down_gemm;
        grouped_gateup_desc_table_dirty_ = true;
        grouped_down_desc_table_dirty_ = true;
        moe_runtime_table_initialized_ = false;
        invalidateFixedTopologyMaskPublication();
        if (!refreshGraphStablePlacement(/*preserve_capture_ready=*/true))
        {
            throw std::runtime_error(
                "MoE expert mask publication failed to refresh graph-stable runtime tables");
        }
    }

    MoEWeightContext MoEExpertComputeStage::buildWeightContext()
    {
        const CPUExpertNUMAPlacement cpu_placement =
            CPUExpertNUMAPlacement::forDevice(params_.device_id);

        return MoEWeightContext{
            params_.device_id,
            params_.num_experts,
            params_.expert_intermediate,
            params_.d_model,
            params_.local_expert_start,
            params_.local_expert_count,
            params_.layer_idx,
            params_.expert_mask,
            params_.gate_exps,
            params_.up_exps,
            params_.down_exps,
            params_.expert_gate_views,
            params_.expert_up_views,
            params_.expert_down_views,
            params_.prepared_gate_gemm,
            params_.prepared_up_gemm,
            params_.prepared_down_gemm,
            params_.moe_owned_kernels,
            params_.moe_packed_gate_lifetime,
            params_.moe_packed_up_lifetime,
            params_.moe_packed_down_lifetime,
            payload_provider_,
            params_.prepared_store,
            params_.expert_registry,
            params_.gate_slab_ref,
            params_.up_slab_ref,
            params_.down_slab_ref,
            true,
            &params_.gpu_direct_slot_pool,
            cpu_placement,
            params_.expert_weight_resolution_policy,
            params_.persistent_weight_owner};
    }

    IMoEKernel *MoEExpertComputeStage::acquireMoEKernelObject() const
    {
        if (!moe_kernel_)
        {
            if (params_.cpu_grouped_serial_workspace)
            {
                if (!params_.device_id.is_cpu() ||
                    !params_.cpu_grouped_serial_workspace->moe_kernel_)
                {
                    LOG_ERROR(
                        "[MoEExpertComputeStage] Serial grouped-MoE workspace has no CPU kernel");
                    return nullptr;
                }
                moe_kernel_ =
                    params_.cpu_grouped_serial_workspace->moe_kernel_.get();
            }
            else if (params_.routed_pipeline_kernel_owner)
            {
                if (!params_.routed_pipeline_kernel_owner->kernel)
                {
                    params_.routed_pipeline_kernel_owner->kernel =
                        KernelFactory::createMoEKernel(params_.device_id);
                }
                moe_kernel_ = params_.routed_pipeline_kernel_owner->kernel.get();
            }
            else
            {
                owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
                moe_kernel_ = owned_moe_kernel_.get();
            }
        }

        return moe_kernel_;
    }

    IMoEKernel *MoEExpertComputeStage::ensureMoEKernel() const
    {
        if (!acquireMoEKernelObject())
            return nullptr;

        if (params_.device_id.is_gpu() &&
            params_.routed_pipeline_kernel_owner)
        {
            const auto &publication =
                params_.routed_pipeline_kernel_owner->router_q8_publication;
            if (!moe_kernel_ ||
                !moe_kernel_->bindRouterQ8HiddenPublication(
                    publication,
                    MoERouterQ8PublicationAccess::ProducerAndConsumer))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to bind the graph-local "
                          "router Q8 publication on "
                          << params_.device_id.to_string());
                return nullptr;
            }
        }
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
            {
                consumer->bindWorkspace(bound_workspace_);
            }
        }
        return kernel;
    }

    bool MoEExpertComputeStage::publishCommittedGroupedVerifierHistograms(
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        void *producer_stream)
    {
        /* This diagnostic wrapper is explicitly outside captured execution.
         * Production graph owners make the same typed preparation transition
         * from their prepareGraphLaunch() boundary. */
        if (!prepareGroupedVerifierHistogramProducer(producer_stream))
            return false;
        if (!enqueueCommittedGroupedVerifierHistograms(
                accepted_state_counts_device,
                publication_ok_flags_device,
                request_count,
                rows_per_request,
                producer_stream))
        {
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::prepareGroupedVerifierHistogramProducer(
        void *producer_stream)
    {
        void *const publication_stream =
            groupedVerifierHistogramPublicationStream();
        if (groupedVerifierHistogramRole() !=
                MoEGroupedVerifierHistogramRole::DeferredAcceptedRows ||
            !params_.moe_runtime_table || !producer_stream ||
            !publication_stream || producer_stream != publication_stream)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Deferred verifier histogram producer "
                "preparation requires the table-owned publication stream"
                << " layer=" << params_.layer_idx
                << " stream=" << producer_stream
                << " expected=" << publication_stream);
            return false;
        }
        // Accepted-state publication owns its table stream independently of
        // the verifier's compute stream. Cold retained widths may never execute
        // before this child is recorded. Acquire only object ownership here;
        // native enqueue below receives its exact stream explicitly.
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Histogram kernel preparation must precede native capture");
            return false;
        }
        if (!acquireMoEKernelObject())
        {
            LOG_ERROR("[MoEExpertComputeStage] Histogram publication could not prepare its graph-owned kernel");
            return false;
        }
        try
        {
            params_.moe_runtime_table->prepareDecodeHistogramProducerStream(
                producer_stream);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Deferred verifier histogram producer "
                "preparation failed"
                << " layer=" << params_.layer_idx
                << " reason=" << error.what());
            return false;
        }
        return true;
    }

    bool MoEExpertComputeStage::
        transitionGroupedVerifierHistogramProducerCapture(
            void *producer_stream,
            RuntimeHistogramProducerCaptureTransition transition)
    {
        void *const publication_stream =
            groupedVerifierHistogramPublicationStream();
        if (groupedVerifierHistogramRole() !=
                MoEGroupedVerifierHistogramRole::DeferredAcceptedRows ||
            !params_.moe_runtime_table || !producer_stream ||
            !publication_stream || producer_stream != publication_stream)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Deferred verifier histogram capture "
                "transition requires the table-owned publication stream"
                << " layer=" << params_.layer_idx
                << " stream=" << producer_stream
                << " expected=" << publication_stream);
            return false;
        }
        try
        {
            params_.moe_runtime_table
                ->transitionDecodeHistogramProducerCapture(
                    producer_stream, transition);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Deferred verifier histogram capture "
                "transition failed"
                << " layer=" << params_.layer_idx
                << " reason=" << error.what());
            return false;
        }
        return true;
    }

    bool MoEExpertComputeStage::enqueueCommittedGroupedVerifierHistograms(
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        void *producer_stream)
    {
        if (groupedVerifierHistogramRole() !=
            MoEGroupedVerifierHistogramRole::DeferredAcceptedRows)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Committed grouped-verifier histogram "
                "publication was requested from a non-verifier GPU stage");
            return false;
        }
        // The accepted-state graph borrows the table-owned publication stream,
        // even when no asynchronous histogram drain is currently installed.
        void *const publication_stream = groupedVerifierHistogramPublicationStream();
        if (!producer_stream || !publication_stream || producer_stream != publication_stream ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            params_.seq_len != request_count * rows_per_request)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Invalid committed grouped-verifier "
                "histogram publication shape"
                << " layer=" << params_.layer_idx
                << " request_count=" << request_count
                << " rows_per_request=" << rows_per_request
                << " stage_rows=" << params_.seq_len
                << " stream=" << producer_stream);
            return false;
        }
        if (!moe_runtime_layer_ || !params_.moe_runtime_table)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Committed grouped-verifier histogram "
                "publication requires an initialized per-layer device route ledger"
                << " layer=" << params_.layer_idx);
            return false;
        }
        // Preparation established this immutable object before capture. Do not
        // bind the verifier's compute stream or construct a kernel here: the
        // accepted-state transaction supplies its table-owned launch stream.
        IMoEKernel *const kernel = moe_kernel_;
        if (!kernel)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Committed grouped-verifier histogram "
                "publication requires its graph-owned kernel prepared before capture"
                << " layer=" << params_.layer_idx);
            return false;
        }

        try
        {
            /* Capture-safe identity validation only. Persistent events and the
             * initialization edge were installed by the publication stage's
             * prepareGraphLaunch() transition. */
            params_.moe_runtime_table->recordDecodeHistogramProducerStream(
                producer_stream);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Deferred verifier histogram producer "
                "was not prepared before enqueue"
                << " layer=" << params_.layer_idx
                << " reason=" << error.what());
            return false;
        }

        const MoEKernelLaunchContext launch{
            .stream = producer_stream,
            .workspace = bound_workspace_};
        if (!kernel->commitGroupedVerifierHistograms(
                launch,
                moe_runtime_layer_,
                accepted_state_counts_device,
                publication_ok_flags_device,
                request_count,
                rows_per_request,
                params_.seq_len,
                params_.num_experts,
                params_.top_k))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Backend rejected committed "
                "grouped-verifier histogram publication"
                << " layer=" << params_.layer_idx);
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::waitForPendingGpuDirectTransfers()
    {
        if (pending_gpu_direct_transfers_.empty())
            return true;

        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] Pending GPU-direct expert transfers on non-GPU stage"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        void *consumer_stream = gpuStream();
        if (!consumer_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Pending GPU-direct expert transfers require an explicit compute stream"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        IBackend *backend = getBackendFor(params_.device_id);
        if (!backend)
        {
            LOG_ERROR("[MoEExpertComputeStage] No backend available to consume GPU-direct transfer event"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        const int device_ordinal = params_.device_id.is_cuda()
                                       ? params_.device_id.cuda_ordinal()
                                       : params_.device_id.rocm_ordinal();
        int waited_events = 0;
        for (const auto &completion : pending_gpu_direct_transfers_)
        {
            if (!completion.valid())
                continue;
            if (completion.device_id != params_.device_id ||
                completion.device_ordinal != device_ordinal)
            {
                LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer completion targets "
                          << completion.device_id.to_string()
                          << " ordinal=" << completion.device_ordinal
                          << " but stage is " << params_.device_id.to_string()
                          << " ordinal=" << device_ordinal
                          << " layer=" << params_.layer_idx);
                return false;
            }
            if (!backend->streamWaitEvent(consumer_stream,
                                          completion.ready_event.get(),
                                          device_ordinal))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to wait on GPU-direct transfer completion event"
                          << " layer=" << params_.layer_idx
                          << " device=" << params_.device_id.to_string());
                return false;
            }
            ++waited_events;
        }

        if (waited_events > 0)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_consumer_event_waits",
                static_cast<double>(waited_events),
                "rebalance",
                params_.device_id.to_string(),
                {{"layer", std::to_string(params_.layer_idx)}});
        }
        pending_gpu_direct_transfers_.clear();
        return true;
    }

    bool MoEExpertComputeStage::refreshGraphStablePlacement(bool preserve_capture_ready)
    {
        return refreshRuntimeGroupedDecodePlacement(preserve_capture_ready) &&
               refreshFixedTopologyGroupedPrefillPlacement();
    }

    void MoEExpertComputeStage::invalidateFixedTopologyMaskPublication() noexcept
    {
        fixed_topology_mask_publication_state_ =
            usesPublishedFixedTopologyMaskGrouping()
                ? FixedTopologyMaskPublicationState::NeedsPublication
                : FixedTopologyMaskPublicationState::NotRequired;
    }

    bool MoEExpertComputeStage::publishFixedTopologyMaskBeforeCapture(IMoEKernel *kernel)
    {
        if (!usesPublishedFixedTopologyMaskGrouping())
        {
            fixed_topology_mask_publication_state_ =
                FixedTopologyMaskPublicationState::NotRequired;
            return true;
        }

        if (fixed_topology_mask_publication_state_ ==
            FixedTopologyMaskPublicationState::Published)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Fixed-topology expert mask was not "
                      "published before graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }
        if (!kernel)
        {
            LOG_ERROR("[MoEExpertComputeStage] Fixed-topology mask publication "
                      "requires a backend kernel"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }
        (void)requireGPUStream();

        const std::vector<uint8_t> mask = fixedTopologyPrefillExpertMaskBytes();
        if (!kernel->updateGroupedPrefillExpertMask(mask.data(), params_.num_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to publish grouped prefill "
                      "expert mask before graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        fixed_topology_mask_publication_state_ =
            FixedTopologyMaskPublicationState::Published;
        return true;
    }

    bool MoEExpertComputeStage::refreshRuntimeGroupedDecodePlacement(bool preserve_capture_ready)
    {
        if (!supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) ||
            params_.seq_len != 1 ||
            !params_.moe_runtime_table ||
            params_.layer_idx < 0 ||
            params_.d_model <= 0 ||
            params_.expert_intermediate <= 0 ||
            params_.num_experts <= 0)
        {
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_launch_state_prepared_ = false;
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Refusing to refresh MoE runtime placement during graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_launch_state_prepared_ = false;
            return false;
        }

        const bool was_capture_ready = runtime_grouped_decode_launch_state_prepared_;
        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
        {
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_launch_state_prepared_ = false;
            return false;
        }

        all_expert_ids_.clear();
        all_expert_ids_.reserve(static_cast<size_t>(params_.num_experts));
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (expertComputesLocally(expert_id))
                all_expert_ids_.push_back(expert_id);
        }

        if (all_expert_ids_.empty() ||
            !ensureGemmEnginesForExperts(all_expert_ids_) ||
            !ensureGroupedGateUpDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !ensureGroupedDownDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !initializeMoERuntimeTableForGroupedDecode())
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to refresh graph-stable MoE runtime placement"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_launch_state_prepared_ = false;
            return false;
        }

        moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank();
        runtime_grouped_decode_launch_state_prepared_ =
            preserve_capture_ready && was_capture_ready &&
            grouped_gateup_desc_table_id_ >= 0 &&
            grouped_down_desc_table_id_ >= 0 &&
            moe_runtime_table_initialized_;
        return moe_runtime_table_initialized_;
    }

    bool MoEExpertComputeStage::refreshFixedTopologyGroupedPrefillPlacement()
    {
        if (!usesGraphStableFixedTopologyPrefillPlacement())
            return true;

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Refusing to refresh fixed-topology MoE prefill placement during graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;

        const std::vector<int> active_expert_ids = fixedTopologyPrefillExpertIds();
        if (active_expert_ids.empty() ||
            !ensureGemmEnginesForExperts(active_expert_ids) ||
            !ensureGroupedGateUpDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !ensureGroupedDownDescriptorTable(kernel, params_.d_model, params_.expert_intermediate))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to refresh graph-stable fixed-topology prefill placement"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        if (usesPublishedFixedTopologyMaskGrouping())
        {
            if (!publishFixedTopologyMaskBeforeCapture(kernel))
                return false;
        }

        if (params_.use_runtime_row_grouping &&
            !initializeMoERuntimeTableForGroupedRows())
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to prepare persistent "
                      "runtime grouping state before graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        return true;
    }

    MoEOverlayServicePhaseHint
    MoEExpertComputeStage::serviceTelemetryPhaseHint() const noexcept
    {
        if (params_.force_decode_equivalent_verifier_prefill ||
            params_.force_grouped_verifier_prefill_for_decode)
        {
            return MoEOverlayServicePhaseHint::GroupedVerifier;
        }
        if (sparse_overlay_invocation_bound_)
        {
            /*
             * A retained participant-local sparse graph has already named its
             * immutable semantic phase through the typed binding. In
             * particular, the one-row predictor sidecar is MTP-routed work,
             * not ordinary decode. Preserve Auto only for mapped follower
             * families whose device-owned authenticated graph role changes
             * between replays; the finish kernel resolves that role before it
             * considers histogram deltas.
             */
            return sparse_overlay_service_phase_;
        }
        if (params_.service_phase != MoEOverlayServicePhaseHint::Auto)
            return params_.service_phase;
        return params_.seq_len == 1
                   ? MoEOverlayServicePhaseHint::Decode
                   : MoEOverlayServicePhaseHint::Prefill;
    }

    void MoEExpertComputeStage::releaseCPUJointInvocation() noexcept
    {
        if(!params_.cpu_joint_layer)return;
        if(params_.prepared_engine_binding_lifecycle!=MoEPreparedEngineBindingLifecycle::SparseOverlayInvocationPublished) {
            cpu_joint_invocation_layer_ = nullptr;
            cpu_joint_invocation_layer_has_local_ = false;
            cpu_joint_invocation_bank_.reset();
            return;
        }
        // CPU prepared engines take scratch as an invocation argument and
        // retain no stage workspace (CPUInvocationWorkspaceConsumer). When
        // direct RCU layer borrowing is active, the vector tables stayed empty
        // so revoking the layer view and lease is O(1) with zero stale pointers.
        if (cpu_joint_tables_dirty_)
        {
            std::fill(params_.prepared_gate_gemm.begin(),params_.prepared_gate_gemm.end(),nullptr);
            std::fill(params_.prepared_up_gemm.begin(),params_.prepared_up_gemm.end(),nullptr);
            std::fill(params_.prepared_down_gemm.begin(),params_.prepared_down_gemm.end(),nullptr);
            std::fill(cached_gate_gemm_.begin(),cached_gate_gemm_.end(),nullptr);
            std::fill(cached_up_gemm_.begin(),cached_up_gemm_.end(),nullptr);
            std::fill(cached_down_gemm_.begin(),cached_down_gemm_.end(),nullptr);
            std::fill(params_.expert_mask.begin(),params_.expert_mask.end(),false);
            cpu_joint_tables_dirty_ = false;
        }
        grouped_gateup_desc_table_dirty_=true;grouped_down_desc_table_dirty_=true;
        runtime_grouped_decode_launch_state_prepared_=false;
        cpu_joint_invocation_layer_ = nullptr;
        cpu_joint_invocation_layer_has_local_ = false;
        cpu_joint_invocation_bank_.reset();
    }

    bool MoEExpertComputeStage::execute(IDeviceContext *ctx)
    {
        clearCompletedCPUActiveExpertView();
        if(params_.cpu_joint_layer && !cpu_joint_invocation_bank_) {
            LOG_ERROR("CPU fragment execution requires its exact bound packet-epoch RCU lease");return false;
        }
        struct InvocationBankScope {
            MoEExpertComputeStage &stage;
            ~InvocationBankScope(){stage.releaseCPUJointInvocation();}
        } release_after_gemm{*this};
        if (!overlay_service_telemetry_layer_)
        {
            // Construction admits device service telemetry only for a complete
            // GPU binding. Every CPU invocation necessarily completes here.
            const bool completed = executeWithoutServiceTelemetry(ctx);
            cpu_active_expert_view_published_ = completed && params_.device_id.is_cpu();
            if (!completed) cpu_grouped_active_experts_.clear();
            return completed;
        }
        if (!ctx || !overlay_service_sample_ ||
            !overlay_service_runtime_layer_)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Dynamic service telemetry lost its exact runtime binding");
            return false;
        }

        IMoEKernel *const kernel = ensureMoEKernel();
        void *const stream = gpuStream();
        const MoEKernelLaunchContext launch{
            .stream = stream,
            .workspace = bound_workspace_,
        };
        if (!kernel || !stream ||
            !kernel->beginMoEOverlayServiceTelemetry(
                launch, overlay_service_sample_))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Could not enqueue the device-local service start marker"
                << " layer=" << params_.layer_idx
                << " device=" << params_.device_id.toString());
            return false;
        }

        /* This setup/capture-side witness proves that the retained graph
         * actually received both service markers. Device publications remain
         * the authority for executions and accumulated arithmetic. */
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_service_marker_pairs_enqueued",
            1.0,
            "graph_capture",
            params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)},
             {"rows", std::to_string(params_.seq_len)},
             {"sparse_endpoint",
              sparse_overlay_invocation_bound_ ? "true" : "false"}});

        if (!executeWithoutServiceTelemetry(ctx))
            return false;
        if (!kernel->finishMoEOverlayServiceTelemetry(
                launch,
                overlay_service_runtime_layer_,
                overlay_service_telemetry_layer_,
                overlay_service_sample_,
                static_cast<std::uint32_t>(params_.num_experts),
                serviceTelemetryPhaseHint(),
                params_.runtime_service_graph_role_device))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Could not enqueue the device-local service finish marker"
                << " layer=" << params_.layer_idx
                << " device=" << params_.device_id.toString());
            return false;
        }
        return true;
    }

    bool MoEExpertComputeStage::executeWithoutServiceTelemetry(
        IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.output)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null input/output tensor");
            return false;
        }

        if (params_.device_id.is_gpu() &&
            !params_.output_registered_in_arena &&
            !params_.canonical_route_contributions)
        {
            auto *output_tensor = dynamic_cast<TensorBase *>(params_.output);
            const auto output_device =
                output_tensor ? output_tensor->current_device() : std::nullopt;
            if (!output_tensor ||
                !output_tensor->gpu_data_ptr() ||
                output_device != params_.device_id)
            {
                throw std::runtime_error(
                    "MoEExpertComputeStage standalone GPU output must be "
                    "preallocated on the exact stage device; execute() cannot "
                    "allocate graph-visible storage");
            }
        }

        if (!waitForPendingGpuDirectTransfers())
            return false;

        const bool has_complete_expert_views =
            params_.expert_gate_views.size() == static_cast<size_t>(params_.num_experts) &&
            params_.expert_up_views.size() == static_cast<size_t>(params_.num_experts) &&
            params_.expert_down_views.size() == static_cast<size_t>(params_.num_experts);
        const bool has_complete_raw_expert_tensors =
            params_.gate_exps && params_.up_exps && params_.down_exps;
        const bool has_prepared_expert_slabs =
            params_.prepared_store &&
            params_.gate_slab_ref.has_value() &&
            params_.up_slab_ref.has_value() &&
            params_.down_slab_ref.has_value();
        const bool has_locally_complete_prepared_engines =
            hasPreparedExpertGemmEnginesForLocalOwnership();
        if (!has_complete_raw_expert_tensors &&
            !has_complete_expert_views &&
            !has_prepared_expert_slabs &&
            !has_locally_complete_prepared_engines)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Missing a complete expert weight source: "
                "provide raw tensors, global expert views, prepared slabs, or "
                "gate/up/down GEMM engines for every locally owned expert");
            return false;
        }

        if (!supportsRequestedRoutedAssignmentPolicy())
        {
            LOG_ERROR("[MoEExpertComputeStage] Routed expert assignment policy "
                      << routedExpertAssignmentPolicyToString(params_.routed_assignment_policy)
                      << " is not wired for this execution path. Refusing to execute as StaticOwner.");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;
        const bool is_gpu = params_.device_id.is_gpu();

        if (params_.canonical_route_contributions)
        {
            const auto required_arithmetic =
                is_gpu
                    ? MoECanonicalRouteArithmeticPolicy::
                          PreweightedContributionThenOrderedAdd
                    : MoECanonicalRouteArithmeticPolicy::
                          UnweightedExpertRowThenOrderedFMA;
            if (params_.canonical_route_arithmetic != required_arithmetic)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage] Canonical route publication has "
                    "an incompatible arithmetic contract"
                    << " device=" << params_.device_id.to_string()
                    << " configured="
                    << moeCanonicalRouteArithmeticPolicyToString(
                           params_.canonical_route_arithmetic)
                    << " required="
                    << moeCanonicalRouteArithmeticPolicyToString(
                           required_arithmetic)
                    << " layer=" << params_.layer_idx);
                return false;
            }
            if (params_.canonical_route_layout ==
                    MoECanonicalRoutePublicationLayout::Unspecified ||
                (is_gpu &&
                 params_.canonical_route_layout !=
                     MoECanonicalRoutePublicationLayout::
                         DenseOriginalRouteSlots))
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage] Canonical route publication has "
                    "an incompatible physical layout"
                    << " device=" << params_.device_id.to_string()
                    << " layout="
                    << moeCanonicalRoutePublicationLayoutToString(
                           params_.canonical_route_layout)
                    << " layer=" << params_.layer_idx);
                return false;
            }
        }
        else if (params_.canonical_route_arithmetic !=
                     MoECanonicalRouteArithmeticPolicy::Unspecified ||
                 params_.canonical_route_layout !=
                     MoECanonicalRoutePublicationLayout::Unspecified)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Canonical route policy was set "
                "without a canonical publication tensor"
                << " device=" << params_.device_id.to_string()
                << " arithmetic="
                << moeCanonicalRouteArithmeticPolicyToString(
                       params_.canonical_route_arithmetic)
                << " layout="
                << moeCanonicalRoutePublicationLayoutToString(
                       params_.canonical_route_layout)
                << " layer=" << params_.layer_idx);
            return false;
        }

        if (params_.prefix_runtime_device_rehydration)
        {
            /*
             * This graph is the sole consumer of the transfer list published by
             * portable prefix restore. Execute it before every routing regime,
             * including M=1 decode. The operation reuses the compact LLEP
             * transport lane, fixed workspace, NCCL/RCCL collectives, and event
             * handoffs used by ordinary prefill movement; it performs no host
             * query, allocation, synchronization, or payload round trip.
             */
            if (params_.layer_idx == 0)
            {
                LOG_INFO("[MoEExpertComputeStage] Executing captured "
                         "prefix-runtime device rehydration transaction"
                         << " device=" << params_.device_id.to_string()
                         << " seq_len=" << params_.seq_len);
            }
            IMoEKernel *rehydration_kernel = ensureMoEKernel();
            DeviceMoERebalanceStatus *transfer_status = nullptr;
            DeviceMoERebalanceApplyStatus *apply_status = nullptr;
            if (!params_.prefix_runtime_rehydration_transfer_state ||
                !rehydration_kernel ||
                !executePrefixRuntimeRehydrationPayloadMovement(
                    rehydration_kernel,
                    &transfer_status,
                    &apply_status,
                    params_.prefix_runtime_rehydration_transfer_state.get()) ||
                !transfer_status ||
                !apply_status)
            {
                throw std::runtime_error(
                    "MoE prefix-runtime device rehydration failed before routed expert execution");
            }
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "moe_layer_device_payload_rehydrations",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)}});
        }

        if (params_.force_decode_equivalent_verifier_prefill)
        {
            return executeDecodeEquivalentVerifierPrefill(ctx);
        }

        /*
         * M=1 retains its latency-specialized routing/publication front end.
         * Transported CPU execution consolidates the projection phases inside
         * executeSingleToken(), without paying the general CSR scheduler used
         * for M>1 packets.
         */
        const bool cpu_host_packet =
            !is_gpu &&
            sparse_overlay_binding_kind_ ==
                SparseOverlayBindingKind::HostPacketPublication;
        if (params_.seq_len == 1 &&
            !params_.force_grouped_verifier_prefill_for_decode)
        {
            return executeSingleToken(ctx, serviceTelemetryPhaseHint());
        }

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights");
            return false;
        }

        const bool has_prepared_expert_state =
            (cpu_joint_invocation_layer_ != nullptr) ||
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views or prepared expert slabs. "
                      "Call extractExpertViews()/prepareExpertGemmEngines() at graph build time.");
            return false;
        }

        // Get device-appropriate MoE kernel for gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // =====================================================================
        // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
        // All 5 kernels launched with zero host sync, counts stay on device.
        // This is the ONLY ROCm prefill path — no per-expert fallback.
        // =====================================================================
        const bool use_fixed_topology_grouped_prefill =
            is_gpu && canUseFixedTopologyGroupedPrefill();
        if (use_fixed_topology_grouped_prefill)
        {
            /*
             * The grouped prefill kernel owns output initialization because
             * its final scatter-add path accumulates routed expert slots into
             * the dense output. Keep that clear inside the backend pipeline so
             * graph replay captures exactly one initialization node per MoE
             * layer. The generic zero below remains for the older paths where
             * the stage itself owns accumulation setup.
             */
            // Ensure descriptor tables are built (lazy, only first call)
            bool tables_ready = true;
            if (static_cast<int>(all_expert_ids_.size()) != num_experts)
            {
                all_expert_ids_.resize(static_cast<size_t>(num_experts));
                std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
            }

            if (canUseSafeCombinedSharedVerifierComposite())
            {
                tables_ready = ensureGemmEnginesForExperts(all_expert_ids_) &&
                               ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                               ensureGroupedDownDescriptorTable(kernel, d_model, intermediate) &&
                               ensureCombinedSharedVerifierResources(kernel, d_model, intermediate);
            }
            else if (grouped_gateup_desc_table_id_ < 0 ||
                     grouped_down_desc_table_id_ < 0 ||
                     grouped_gateup_desc_table_dirty_ ||
                     grouped_down_desc_table_dirty_)
            {
                const auto prefill_expert_ids = fixedTopologyPrefillExpertIds();
                tables_ready = ensureGemmEnginesForExperts(prefill_expert_ids) &&
                               ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                               ensureGroupedDownDescriptorTable(kernel, d_model, intermediate);
            }

            if (!tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] Grouped prefill: descriptor table build failed, layer "
                          << params_.layer_idx);
                return false;
            }

            const bool grouped_ok = canUseSafeCombinedSharedVerifierComposite()
                                        ? executeSafeCombinedSharedVerifierComposite(kernel)
                                        : executeFixedTopologyGroupedPrefill(kernel, seq_len);
            if (!grouped_ok)
            {
                LOG_ERROR("[MoEExpertComputeStage] Grouped prefill pipeline failed, layer "
                          << params_.layer_idx);
                return false;
            }

            if (params_.device_id.is_gpu())
                publishPublicationTarget();
            return true;
        }

        const bool forced_verifier_decode_replay =
            params_.seq_len == 1 && params_.force_grouped_verifier_prefill_for_decode;
        if (forced_verifier_decode_replay && isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] MTP verifier correction replay requested grouped "
                      "prefill for seq_len=1 inside graph capture, but the fixed-topology grouped path "
                      "is unavailable: "
                      "device=" << params_.device_id.to_string()
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
                                << ", layer=" << params_.layer_idx);
            return false;
        }

        /*
         * Every supported GPU MoE mode must enter the same economical grouped
         * pipeline. The removed branch below downloaded counts/offsets and
         * drove one expert at a time from the host whenever capture was not yet
         * active. That made warmup semantically different from capture and
         * allowed stale host routing state to steer production execution.
         */
        if (params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU MoE execution requires the fully device-resident grouped pipeline: "
                      << "seq_len=" << params_.seq_len
                      << ", fullOwnership=" << hasFullLocalExpertOwnership()
                      << ", allEnabled=" << expertMaskAllEnabled()
                      << ", fixedMask=" << hasFixedTopologyPrefillExpertMask()
                      << ", replicas=" << params_.replica_set.num_replicated
                      << ", layer=" << params_.layer_idx);
            return false;
        }

        if (forced_verifier_decode_replay)
        {
            /*
             * MTP correction replay rows are part of the verifier publication
             * contract.  Full-ownership graphs use the fixed descriptor-table
             * route above; LocalTP/overlay graphs intentionally use this
             * mask-aware device grouping route.  Do not fall through to the
             * host-routed CPU grouping path when device grouping is unavailable,
             * because that would split verifier state ownership and hide a real
             * graph/runtime capability gap.
             */
            LOG_ERROR("[MoEExpertComputeStage] MTP verifier correction replay could not use "
                      "the required device grouped path for seq_len=1: "
                      "device=" << params_.device_id.to_string()
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
                                << ", layer=" << params_.layer_idx);
            return false;
        }

        /*
         * CPU prefill is a first-class grouped, serial-row-equivalent route.
         * It shares the same arithmetic implementation as MTP verification,
         * while ordinary-prefill ownership remains static and explicit.
         */
        CPUGroupedRowsPurpose cpu_purpose =
            CPUGroupedRowsPurpose::OrdinaryPrefill;
        if (cpu_host_packet)
        {
            switch (sparse_overlay_service_phase_)
            {
            case MoEOverlayServicePhaseHint::Decode:
                cpu_purpose = CPUGroupedRowsPurpose::Decode;
                break;
            case MoEOverlayServicePhaseHint::Prefill:
                cpu_purpose = CPUGroupedRowsPurpose::OrdinaryPrefill;
                break;
            case MoEOverlayServicePhaseHint::GroupedVerifier:
                cpu_purpose = CPUGroupedRowsPurpose::MTPVerifier;
                break;
            case MoEOverlayServicePhaseHint::Auto:
                LOG_ERROR(
                    "[MoEExpertComputeStage] CPU host packet has no authenticated inference phase");
                return false;
            }
        }
        return executeCPUGroupedDecodeEquivalentRows(ctx, cpu_purpose);
    }

    // =========================================================================
    // MoEExpertComputeStage::executeSingleToken — Optimized decode path (seq_len=1)
    //
    // Eliminates per-expert overhead:
    // - No gather (input IS the single token)
    // - No scatter (direct weighted accumulation into output)
    // - No vector allocations (stack arrays for top_k ≤ 16)
    // - No expert_token_lists grouping
    // - Reuses a single pair of scratch buffers across all experts
    // =========================================================================

    ITensor *MoEExpertComputeStage::publicationTarget() const
    {
        ITensor *target = params_.canonical_route_contributions
                              ? params_.canonical_route_contributions
                              : params_.output;
        if (!target)
        {
            throw std::logic_error(
                "MoEExpertComputeStage has no routed-expert publication target");
        }
        return target;
    }

    void MoEExpertComputeStage::publishPublicationTarget() const
    {
        gpuExecution().publish(publicationTarget());
    }

    bool MoEExpertComputeStage::executeSingleToken(IDeviceContext *ctx, MoEOverlayServicePhaseHint phase)
    {
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;
        const bool is_gpu = params_.device_id.is_gpu();

        const bool has_prepared_expert_state =
            (cpu_joint_invocation_layer_ != nullptr) ||
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views or prepared expert slabs.");
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();

        /*
         * GPU fused decode overwrites exactly one publication target: either
         * the final MoE row or the canonical route-contribution tensor. It
         * must not initialize the inactive later-reducer output, and an extra
         * memset would add a needless captured launch. CPU decode accumulates
         * into the final row below, so only that host-owned path needs an
         * explicit zero before projection work begins.
         */
        float *cpu_canonical_rows = nullptr;
        float *cpu_packed_records = nullptr;
        if (!is_gpu)
        {
            TensorBase *const publication =
                params_.canonical_route_contributions
                    ? params_.canonical_route_contributions
                    : params_.output;
            if (params_.canonical_route_contributions)
            {
                cpu_canonical_rows = publication->mutable_data();
                if (!cpu_canonical_rows)
                    throw std::logic_error("CPU decode lacks its canonical route publication");
            }
            const bool packed_canonical_records =
                params_.canonical_route_contributions &&
                params_.canonical_route_layout ==
                    MoECanonicalRoutePublicationLayout::
                        PackedIndexedRouteRows;
            const bool host_packet_dense_canonical =
                params_.canonical_route_contributions &&
                params_.canonical_route_layout ==
                    MoECanonicalRoutePublicationLayout::
                        DenseOriginalRouteSlots &&
                sparse_overlay_binding_kind_ ==
                    SparseOverlayBindingKind::HostPacketPublication;
            if (packed_canonical_records)
            {
                float *const records = cpu_canonical_rows;
                const size_t capacity =
                    canonical_moe_route_record::recordCapacity(
                        publication->numel(),
                        d_model);
                if (capacity < static_cast<size_t>(top_k) ||
                    !canonical_moe_route_record::writeRecordCount(
                        records,
                        publication->numel(),
                        0u))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU decode packed route "
                              "publication buffer is undersized"
                              << " layer=" << params_.layer_idx
                              << " elements=" << publication->numel()
                              << " capacity=" << capacity
                              << " required_records=" << top_k);
                    return false;
                }
                cpu_packed_records = records;
            }
            else
            {
                const size_t publication_rows = cpu_canonical_rows ? static_cast<size_t>(top_k) : 1u;
                const size_t publication_elements = publication_rows * static_cast<size_t>(d_model);
                if (cpu_canonical_rows && publication->numel() < publication_elements)
                    throw std::logic_error("CPU decode dense canonical publication is undersized");
                if (!host_packet_dense_canonical)
                    kernel->zeroBuffer(publication, publication_elements * sizeof(float));
            }
        }

        // Expert-ID ownership range.
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        // Use input tensor directly (no gather needed for 1 token)
        const TensorBase *input_tensor = params_.input;

        if (supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table && params_.layer_idx >= 0 && !moe_runtime_layer_)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        bool runtime_decode_bank_active_for_expert_stage =
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            moe_runtime_layer_ &&
            runtimeTableHasActiveGroupedDecodeBank();
        if (supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table &&
            params_.layer_idx >= 0 &&
            (!moe_runtime_table_initialized_ || !runtime_decode_bank_active_for_expert_stage))
        {
            moe_runtime_table_initialized_ = !moe_runtime_table_initialized_
                                                 ? initializeMoERuntimeTableForGroupedDecode()
                                                 : (runtime_decode_bank_active_for_expert_stage ||
                                                    initializeMoERuntimeTableForGroupedDecode());
            /*
             * The first decode step after clear_cache() may initialize the
             * placement/runtime bank itself.  Treat that freshly initialized
             * bank as available for the same warmup pass so the fused runtime
             * grouped path is warmed before cached graph capture is armed.
             */
            runtime_decode_bank_active_for_expert_stage =
                moe_runtime_table_initialized_ &&
                moe_runtime_layer_ &&
                runtimeTableHasActiveGroupedDecodeBank();
        }

        const bool can_try_device_routed_decode =
            !params_.require_device_routing_tensor_decode &&
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table &&
            moe_runtime_layer_ &&
            moe_runtime_table_initialized_ &&
            runtime_decode_bank_active_for_expert_stage &&
            top_k > 0 && top_k <= 16;

        if (params_.device_id.is_gpu() &&
            params_.seq_len == 1 &&
            params_.moe_runtime_table)
        {
            PerfStatsCollector::addCounter(
                "moe_runtime_decode",
                can_try_device_routed_decode ? "predicate_ready" : "predicate_skipped",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"backend_supported", perfBool(supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id))},
                 {"runtime_layer", perfBool(moe_runtime_layer_ != nullptr)},
                 {"initialized", perfBool(moe_runtime_table_initialized_)},
                 {"active_bank", perfBool(runtime_decode_bank_active_for_expert_stage)},
                 {"descriptor_source",
                  usesRuntimePlacementWeightDescriptors()
                      ? "runtime"
                      : "static_table"},
                 {"top_k", std::to_string(top_k)},
                 {"replicas", std::to_string(params_.replica_set.num_replicated)}});
        }

        if (can_try_device_routed_decode)
        {
            bool device_routed_done = false;
            const MoEDecodeDescriptorSource descriptor_source =
                params_.weight_descriptor_source;
            const bool have_grouped_tables =
                grouped_gateup_desc_table_id_ >= 0 &&
                grouped_gateup_desc_table_num_experts_ == num_experts &&
                grouped_gateup_desc_table_d_model_ == d_model &&
                grouped_gateup_desc_table_intermediate_ == intermediate &&
                grouped_down_desc_table_id_ >= 0 &&
                grouped_down_desc_table_num_experts_ == num_experts &&
                grouped_down_desc_table_d_model_ == d_model &&
                grouped_down_desc_table_intermediate_ == intermediate;

            bool grouped_tables_ready = have_grouped_tables;
            if (!grouped_tables_ready)
            {
                all_expert_ids_.clear();
                all_expert_ids_.reserve(static_cast<size_t>(num_experts));
                for (int expert_id = 0; expert_id < num_experts; ++expert_id)
                {
                    if (expertComputesLocally(expert_id))
                        all_expert_ids_.push_back(expert_id);
                }

                const PerfStatsCollector::Tags setup_tags{
                    {"layer", std::to_string(params_.layer_idx)},
                    {"num_experts", std::to_string(num_experts)},
                    {"top_k", std::to_string(top_k)}};
                PerfStatsCollector::addCounter(
                    "moe_runtime_decode", "local_expert_count",
                    static_cast<double>(all_expert_ids_.size()),
                    "decode", params_.device_id.toString(), setup_tags);

                if (all_expert_ids_.empty())
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "no_local_experts",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGemmEnginesForExperts(all_expert_ids_))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "ensure_engines_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "gateup_descriptor_table_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGroupedDownDescriptorTable(kernel, d_model, intermediate))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "down_descriptor_table_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else
                {
                    grouped_tables_ready = true;
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "descriptor_tables_ready",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
            }

            if (!grouped_tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] GPU runtime grouped decode descriptor tables "
                          "are unavailable for layer "
                          << params_.layer_idx);
                return false;
            }

            /*
             * CUDA and ROCm own their complete runtime decode scratch in the
             * stage-bound persistent workspace. The fused entry point is the one
             * production path: constructing per-slot TensorBase scratch here
             * would reintroduce dynamic allocation and a second decode contract.
             */
            device_routed_done = kernel->groupedExpertDecodeFromRuntime(
                moe_runtime_layer_,
                input_tensor,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                top_k,
                params_.output,
                d_model,
                intermediate,
                descriptor_source,
                params_.canonical_route_contributions);

            if (!device_routed_done)
            {
                LOG_ERROR("[MoEExpertComputeStage] Mandatory fused runtime grouped decode failed for layer "
                          << params_.layer_idx);
                return false;
            }

            if (!isGraphCaptureActive())
                runtime_grouped_decode_launch_state_prepared_ = true;
            publishPublicationTarget();
            return true;
        }

        const bool require_device_routing_tensor_decode =
            params_.require_device_routing_tensor_decode;
        std::vector<uint8_t> device_routing_expert_mask;
        const uint8_t *device_routing_expert_mask_ptr = nullptr;
        std::vector<int> device_routing_required_expert_ids;
        if (is_gpu &&
            !params_.expert_mask.empty() &&
            params_.expert_mask.size() == static_cast<size_t>(num_experts) &&
            !expertMaskAllEnabled())
        {
            device_routing_expert_mask.resize(static_cast<size_t>(num_experts), 0u);
            device_routing_required_expert_ids.reserve(static_cast<size_t>(num_experts));
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                if (!params_.expert_mask[static_cast<size_t>(expert_id)])
                    continue;
                device_routing_expert_mask[static_cast<size_t>(expert_id)] = 1u;
                device_routing_required_expert_ids.push_back(expert_id);
            }
            if (!usesRuntimePlacementWeightDescriptors() &&
                !device_routing_required_expert_ids.empty())
                device_routing_expert_mask_ptr = device_routing_expert_mask.data();
        }

        /*
         * Explicit-routing M=1 decode binds one already-produced routing row as
         * device tensors and asks the backend to select descriptors without
         * reading a stale host mirror. Static expert-ID-apportioned LocalTP and
         * Dynamic/LLEP overlays deliberately give each participant only a
         * subset of complete experts. The device route therefore converts
         * masked-off top-k slots to -1 and computes exactly the participant-
         * local partial result consumed by the subsequent MoE allreduce.
         *
         * Do not require full local ownership here. Descriptor-table upload
         * preserves one global expert-id slot per model expert, leaves remote
         * slots empty, and prepares engines only for mask-active local experts.
         * Requiring every expert locally would make the mandatory apportioned
         * production topology impossible even though the backend contract is
         * explicitly mask-aware.
         */
        const bool can_try_device_routing_tensor_decode =
            is_gpu &&
            supportsDeviceRoutingTensorDecodeExecutionBackend(params_.device_id) &&
            params_.routing_indices &&
            params_.routing_weights &&
            top_k > 0 && top_k <= 16 &&
            params_.replica_set.num_replicated == 0 &&
            (params_.expert_mask.empty() ||
             params_.expert_mask.size() == static_cast<size_t>(num_experts));

        if (can_try_device_routing_tensor_decode)
        {
            const bool have_grouped_tables =
                grouped_gateup_desc_table_id_ >= 0 &&
                grouped_gateup_desc_table_num_experts_ == num_experts &&
                grouped_gateup_desc_table_d_model_ == d_model &&
                grouped_gateup_desc_table_intermediate_ == intermediate &&
                grouped_down_desc_table_id_ >= 0 &&
                grouped_down_desc_table_num_experts_ == num_experts &&
                grouped_down_desc_table_d_model_ == d_model &&
                grouped_down_desc_table_intermediate_ == intermediate;

            bool grouped_tables_ready = have_grouped_tables;
            if (!grouped_tables_ready)
            {
                if (static_cast<int>(all_expert_ids_.size()) != num_experts)
                {
                    all_expert_ids_.resize(static_cast<size_t>(num_experts));
                    std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
                }

                const std::vector<int> &required_expert_ids =
                    device_routing_expert_mask_ptr ? device_routing_required_expert_ids
                                                   : all_expert_ids_;

                grouped_tables_ready = ensureGemmEnginesForExperts(required_expert_ids) &&
                                       ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                                       ensureGroupedDownDescriptorTable(kernel, d_model, intermediate);
            }

            if (!grouped_tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] Explicit-routing GPU decode "
                          "descriptor tables are unavailable for layer "
                          << params_.layer_idx);
                return false;
            }

            if (!kernel->groupedExpertDecodeFromRouting(
                    input_tensor,
                    params_.routing_indices,
                    params_.routing_weights,
                    grouped_gateup_desc_table_id_,
                    grouped_down_desc_table_id_,
                    top_k,
                    params_.output,
                    d_model,
                    intermediate,
                    device_routing_expert_mask_ptr,
                    params_.canonical_route_contributions,
                    usesRuntimePlacementWeightDescriptors()
                        ? moe_runtime_layer_
                        : nullptr,
                    params_.weight_descriptor_source))
            {
                LOG_ERROR("[MoEExpertComputeStage] Mandatory fused explicit-routing "
                          "GPU decode failed for layer "
                          << params_.layer_idx
                          << " (tables_ready=" << grouped_tables_ready
                          << ", top_k=" << top_k
                          << ", full_ownership=" << hasFullLocalExpertOwnership()
                          << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                          << ", replicas=" << params_.replica_set.num_replicated << ")");
                return false;
            }

            if (!isGraphCaptureActive())
            {
                /*
                 * The eager launch has now exercised the same workspace-native
                 * fused route used by capture. Record that its stable pointer
                 * arrays and scratch bindings are warm; prepareGraphLaunch()
                 * will still revalidate them against the exact capture stream.
                 */
                runtime_grouped_decode_launch_state_prepared_ = true;
            }
            publishPublicationTarget();
            return true;
        }

        if (is_gpu)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU decode has neither a usable "
                      "runtime-table route nor a usable explicit-routing fused path for layer "
                      << params_.layer_idx
                      << " (device=" << params_.device_id.toString()
                      << ", explicit_required=" << require_device_routing_tensor_decode
                      << ", grouped_decode=" << debugEnv().rocm.moe_grouped_decode
                      << ", device_routed_decode=" << debugEnv().rocm.moe_device_routed_decode
                      << ", top_k=" << top_k
                      << ", has_indices=" << (params_.routing_indices != nullptr)
                      << ", has_weights=" << (params_.routing_weights != nullptr)
                      << ", full_ownership=" << hasFullLocalExpertOwnership()
                      << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                      << ", replicas=" << params_.replica_set.num_replicated << ")");
            return false;
        }

        /*
         * CPU decode owns reusable host tensors because CPU GEMV APIs consume
         * TensorBase outputs directly. GPU execution has already returned
         * through one of the fused workspace-native contracts above and can
         * never reach these allocations.
         */
        if (params_.device_id.is_cpu())
        {
            if (!cpu_grouped_backing_ ||
                static_cast<int>(scratch_gate_batch_.size()) < top_k || !scratch_out_)
                throw std::logic_error("CPU MoE decode requires setup-bound declared storage");
        }
        else
        {
            if (static_cast<int>(scratch_gate_batch_.size()) < top_k)
            {
                scratch_gate_batch_.resize(top_k);
                scratch_up_batch_.resize(top_k);
                for (int i = 0; i < top_k; ++i)
                {
                    scratch_gate_batch_[i] =
                        makeScratchFP32(1, intermediate, params_.device_id);
                    scratch_up_batch_[i] =
                        makeScratchFP32(1, intermediate, params_.device_id);
                }
            }
            if (!scratch_out_)
                scratch_out_ = makeScratchFP32(1, d_model, params_.device_id);
        }

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights (single-token)");
            return false;
        }

        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();

        // ---------------------------------------------------------------
        // Phase 1: Batch all experts' gate+up into ONE fused GEMV call.
        // CPU NativeQuantised consumes the Q8_1 row already published by routing;
        // GPU and floating CPU bundles retain their backend-native activation
        // preparation.  Every route still uses one parallel projection region.
        // ---------------------------------------------------------------
        struct ActiveExpert
        {
            int expert_id;
            float weight;
            int batch_idx;
            int route_slot;
        };
        ActiveExpert active_experts[16]; // stack-allocated, max top_k
        int num_active = 0;

        batch_projections_.clear();
        batch_projections_.reserve(top_k * 2);

        // Per-token dynamic dispatch for replicated experts.
        // When replicas are active, use ExpertReplicaSet::assignForToken()
        // to deterministically decide which socket computes each expert.
        // TODO(gpu-moe-rebalance): after the GPU rebalancing correctness gates
        // are proven, move this replica assignment into the device-side routing
        // path so homogeneous GPU domains do not round-trip token ownership
        // decisions through host-side ExpertReplicaSet logic.
        bool compute_here[16]; // stack-allocated, max top_k

        // Convert float indices to int for replica dispatch
        int routing_int_indices[16]; // stack-allocated, max top_k
        for (int k = 0; k < top_k; ++k)
        {
            const float weight = routing_wt_data[k];
            if (!std::isfinite(weight))
            {
                LOG_ERROR("[MoEExpertComputeStage] Non-finite decode route weight at slot "
                          << k << " for layer " << params_.layer_idx);
                return false;
            }
            if (weight == 0.0f)
            {
                routing_int_indices[k] = -1;
                continue;
            }

            routing_int_indices[k] = static_cast<int>(routing_idx_data[k]);
            if (routing_int_indices[k] < 0 || routing_int_indices[k] >= num_experts)
            {
                LOG_ERROR("[MoEExpertComputeStage] Invalid routed expert id " << routing_int_indices[k]
                                                                              << " at decode slot " << k
                                                                              << " for layer " << params_.layer_idx
                                                                              << " (num_experts=" << num_experts << ")");
                return false;
            }
        }

        if (params_.replica_set.num_replicated > 0)
        {
            params_.replica_set.assignForToken(
                routing_int_indices,
                routing_wt_data,
                top_k,
                params_.my_socket_id,
                cpu_joint_invocation_layer_ ? cpu_joint_invocation_layer_->resident_mask : params_.expert_mask,
                compute_here,
                params_.layer_idx);
        }
        else
        {
            // No replicas — use simple mask/range check
            for (int k = 0; k < top_k; ++k)
            {
                if (routing_wt_data[k] == 0.0f)
                {
                    compute_here[k] = false;
                    continue;
                }
                compute_here[k] = activeExpertResident(routing_int_indices[k]);
            }
        }

        if (is_gpu)
        {
            std::vector<int> active_expert_ids;
            active_expert_ids.reserve(top_k);
            for (int k = 0; k < top_k; ++k)
                if (compute_here[k])
                    active_expert_ids.push_back(routing_int_indices[k]);
            if (!ensureGemmEnginesForExperts(active_expert_ids))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to prepare GPU GEMM engines for decode experts");
                return false;
            }
        }
        else if (!cpu_joint_invocation_layer_)
        {
            ensureGemmEnginesCached();
        }

        for (int k = 0; k < top_k; ++k)
        {
            if (!compute_here[k])
                continue;

            const int expert_id = routing_int_indices[k];

            ITensorGemm *gate_gemm = activeGateGemm(expert_id);
            ITensorGemm *up_gemm = activeUpGemm(expert_id);

            if (!gate_gemm || !up_gemm)
            {
                LOG_ERROR("[MoEExpertComputeStage] FATAL: Null gate/up GEMM engine for expert "
                          << expert_id << " (layer " << params_.layer_idx
                          << ", mask=" << (int)activeExpertResident(expert_id)
                          << ", replicated=" << params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id)
                          << ", prepared_gate=" << (bool)gate_gemm << ")");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            batch_projections_.push_back(
                {gate_gemm, scratch_gate_batch_[num_active].get(), intermediate, nullptr, "gate"});
            batch_projections_.push_back(
                {up_gemm, scratch_up_batch_[num_active].get(), intermediate, nullptr, "up"});

            active_experts[num_active] = {
                expert_id,
                routing_wt_data[k],
                num_active,
                k};
            num_active++;
        }

        // Bind the original slot metadata before any Down producer writes its
        // disjoint record. The count remains zero until the CPU transaction retires.
        if (cpu_packed_records)
        {
            for (int route = 0; route < num_active; ++route)
            {
                const auto &info = active_experts[route];
                if (info.batch_idx != route || info.route_slot < 0 || info.route_slot >= top_k)
                    throw std::logic_error("CPU decode lost its original packed route order");
                float *const record = canonical_moe_route_record::record(cpu_packed_records, route, d_model);
                if (!canonical_moe_route_record::writeFlatRouteSlot(record, d_model,
                        static_cast<size_t>(info.route_slot)))
                    throw std::logic_error("CPU decode lost its original packed route slot");
            }
        }

        // The final canonical row is bound before the Down workshare. A
        // complete local owner instead retains an unweighted row for its
        // separately ordered top-k accumulation.
        const auto canonical_destination = [&](const ActiveExpert &route) -> float * {
            if (cpu_packed_records)
                return canonical_moe_route_record::record(cpu_packed_records, route.batch_idx, d_model);
            return cpu_canonical_rows
                ? cpu_canonical_rows + static_cast<size_t>(route.route_slot) * d_model
                : nullptr;
        };

        // Single fused call: router-Q8 reuse on CPU NativeQuantised, then one
        // projection region for every active expert's gate/up pair.
        bool cpu_prepared_decode_completed = false;
        if (num_active > 0)
        {
            bool projected = false;
            if (!is_gpu)
            {
                bool any_native = false;
                bool all_native = true;
                for (const auto &projection : batch_projections_)
                {
                    const bool native =
                        dynamic_cast<CPUNativeQuantisedGemmKernel *>(projection.kernel) != nullptr;
                    any_native = any_native || native;
                    all_native = all_native && native;
                }
                if (any_native != all_native)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU decode cannot mix "
                              "NativeQuantised and floating gate/up projection contracts in layer "
                              << params_.layer_idx);
                    return false;
                }

                if (all_native)
                {
                    auto *cpu_moe_kernel = dynamic_cast<CPUMoEKernel *>(kernel);
                    auto *native_gate = dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                        batch_projections_[0].kernel);
                    const float *input_rows = input_tensor->data();
                    /*
                     * A graph-native ExpertOverlay participant receives this
                     * one compact row from sparse transport instead of from a
                     * local router.  The NativeQuantised projection contract is
                     * still exactly the router's Q8_1 quantization, so publish
                     * it at the declared transport boundary before looking it
                     * up.  This mirrors the grouped-row path below; omitting
                     * it made a one-row cold-tier packet fail while multi-row
                     * packets succeeded.
                     */
                    if (params_.cpu_router_q8_input_publication ==
                        CPURouterQ8InputPublicationPolicy::PublishTransportedRows)
                    {
                        if (!cpu_moe_kernel || !input_rows ||
                            !cpu_moe_kernel->publishTransportedRouterQ8Hidden(
                                input_rows,
                                /*rows=*/1,
                                d_model))
                        {
                            LOG_ERROR(
                                "[MoEExpertComputeStage] CPU ExpertOverlay transport "
                                "failed to publish canonical serial router Q8_1 row for layer "
                                << params_.layer_idx << " d_model=" << d_model);
                            return false;
                        }
                        if (PerfStatsCollector::isDomainEnabled("moe_overlay"))
                        {
                            PerfStatsCollector::addCounter(
                                "moe_overlay",
                                "cpu_transport_router_q8_publications",
                                1.0,
                                "decode",
                                params_.device_id.toString(),
                                {{"layer", std::to_string(params_.layer_idx)},
                                 {"rows", "1"},
                                 {"d_model", std::to_string(d_model)},
                                 {"source", "compact_transport_serial_row"}});
                        }
                    }
                    const Q8_1Block *published_router_q8 =
                        cpu_moe_kernel && input_rows
                            ? cpu_moe_kernel->publishedRouterQ8Hidden(
                                  input_rows,
                                  /*rows=*/1,
                                  d_model)
                            : nullptr;
                    if (!native_gate || !published_router_q8)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CPU NativeQuantised decode is "
                                  "missing its router Q8_1 publication for layer "
                                  << params_.layer_idx);
                        return false;
                    }
                    const size_t routes = static_cast<size_t>(num_active);
                    if (phase == MoEOverlayServicePhaseHint::Decode &&
                        CPUNativeQuantisedGemmKernel::standaloneMoEOneRowWorkers(d_model, intermediate, routes))
                    {
                        // Keep the prepared schedule's operand staging and
                        // stack frame independent of the existing grouped code.
                        const auto complete_prepared = [&]() __attribute__((noinline)) {
                            using Projection = CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
                            constexpr size_t capacity = std::extent_v<decltype(active_experts)>;
                            if (!bound_workspace_) return false;
                            const size_t blocks = static_cast<size_t>(intermediate) / Q8_1Block::BLOCK_SIZE;
                            std::array<Projection, capacity> down;
                            for (size_t route = 0; route < routes; ++route)
                            {
                                down[route].kernel = dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                                    activeDownGemm(active_experts[route].expert_id));
                                if (!down[route].kernel) return false;
                            }
                            if (!cpu_canonical_rows && scratch_down_batch_.size() < routes)
                                throw std::logic_error("CPU decode exceeds its declared route output capacity");
                            auto activation = CPUInvocationWorkspace::available<Q8_1Block>(bound_workspace_, kCPUProjectionQ8);
                            if (activation.size() < routes * blocks)
                                throw std::logic_error("CPU one-row decode lacks its admitted Down activation workspace");
                            std::array<Projection, 2 * capacity> gate_up;
                            for (size_t route = 0; route < routes; ++route)
                            {
                                float *const destination = cpu_canonical_rows
                                    ? canonical_destination(active_experts[route])
                                    : scratch_down_batch_[route]->mutable_data();
                                down[route] = {.kernel = down[route].kernel, .input_q8 = activation.data() + route * blocks,
                                    .output = destination, .rows = 1, .n = d_model, .ldc = d_model};
                            }
                            for (size_t index = 0; index < 2 * routes; ++index)
                            {
                                const auto &source = batch_projections_[index];
                                gate_up[index] = {.kernel = static_cast<CPUNativeQuantisedGemmKernel *>(source.kernel),
                                    .input_q8 = published_router_q8, .output = source.output->mutable_data(),
                                    .rows = 1, .n = intermediate, .ldc = intermediate};
                            }
                            const auto candidate = CPUNativeQuantisedGemmKernel::tryExecuteStandaloneMoEOneRow(
                                std::span(gate_up).first(2 * routes), std::span(down).first(routes), activation,
                                d_model, intermediate);
                            if (candidate.has_value() && !*candidate)
                                throw std::runtime_error("Prepared standalone CPU MoE did not complete its fixed worker team");
                            return candidate.value_or(false);
                        };
                        cpu_prepared_decode_completed = complete_prepared();
                    }
                    projected = cpu_prepared_decode_completed || native_gate->
                        multiply_fused_router_q8_hidden_grouped_decode_equivalent(published_router_q8,
                            batch_projections_, /*m=*/1, d_model,
                            CPUInvocationWorkspace::available<float>(bound_workspace_, kCPUProjectionPartials));
                    if (projected && PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "cpu_moe_decode_router_q8_reuse_calls",
                            1.0,
                            "moe",
                            "cpu",
                            {{"top_k", std::to_string(top_k)},
                             {"active_experts", std::to_string(num_active)}});
                    }
                }
                else
                {
                    projected = batch_projections_[0].kernel->multiply_fused_tensor(
                        input_tensor,
                        batch_projections_,
                        /*m=*/1,
                        d_model,
                        nullptr,
                        getWorkspace());
                }
            }
            else
            {
                projected = batch_projections_[0].kernel->multiply_fused_tensor(
                    input_tensor,
                    batch_projections_,
                    /*m=*/1,
                    d_model,
                    nullptr,
                    getWorkspace());
            }

            if (!projected)
            {
                LOG_ERROR("[MoEExpertComputeStage] Decode gate/up batched projection failed for layer "
                          << params_.layer_idx);
                return false;
            }
            if (is_gpu)
            {
                for (const auto &projection : batch_projections_)
                    gpuExecution().publish(projection.output);
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Fused SwiGLU + Down projection + weighted accumulate
        //
        // GPU path: per-expert fusedSwigluDown (tensor-based, runs on GPU)
        //           + GPU weightedAdd (avoids all D2H transfers).
        // CPU path: batch SwiGLU + fused multi-input down projections
        //           + vec_axpy accumulation in a single OMP region.
        // ---------------------------------------------------------------

        if (is_gpu)
        {
            bool grouped_down_done = false;
            if (debugEnv().rocm.moe_grouped_decode && num_active > 0 && num_active <= 16)
            {
                ITensor *gate_tensors[16] = {};
                ITensor *up_tensors[16] = {};
                int grouped_expert_ids[16] = {};
                float grouped_weights[16] = {};
                DeviceNativeQuantisedMatrixDesc down_descs[16] = {};
                bool grouped_supported = true;

                for (int i = 0; i < num_active; ++i)
                {
                    const auto &info = active_experts[i];
                    ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];
                    if (!down_gemm)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                                  << info.expert_id << " (layer " << params_.layer_idx << ")");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    DeviceNativeQuantisedMatrixDesc desc;
                    if (!down_gemm->exportNativeQuantisedMatrixDesc(desc) ||
                        desc.n != d_model || desc.k != intermediate)
                    {
                        grouped_supported = false;
                        break;
                    }

                    gate_tensors[i] = scratch_gate_batch_[info.batch_idx].get();
                    up_tensors[i] = scratch_up_batch_[info.batch_idx].get();
                    grouped_expert_ids[i] = info.expert_id;
                    grouped_weights[i] = info.weight;
                    down_descs[i] = desc;
                }

                if (grouped_supported)
                {
                    grouped_down_done = kernel->groupedExpertDownDecode(
                        gate_tensors,
                        up_tensors,
                        grouped_expert_ids,
                        grouped_weights,
                        down_descs,
                        num_active,
                        params_.output,
                        d_model,
                        intermediate);
                    if (!grouped_down_done)
                    {
                        LOG_DEBUG("[MoEExpertComputeStage] Grouped ROCm decode down path unavailable for layer "
                                  << params_.layer_idx << "; using per-expert fallback");
                    }
                }
            }

            if (!grouped_down_done)
            {
                // GPU Phase 2 fallback: sequential per-expert SwiGLU+Down on GPU + GPU accumulate.
                // fusedSwigluDown uses multiply_tensor_with_fused_swiglu (tensor-based,
                // executed on GPU by ROCmQuantisedGemmKernel).  No D2H transfers occur.
                if (!scratch_out_)
                    scratch_out_ = makeScratchFP32(1, d_model, params_.device_id);

                for (int i = 0; i < num_active; ++i)
                {
                    const auto &info = active_experts[i];
                    ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];

                    if (!down_gemm)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                                  << info.expert_id << " (layer " << params_.layer_idx << ")");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    // Fused SwiGLU + Down on GPU (tensor-based).
                    // After Phase 1, scratch_gate/up are DEVICE_AUTHORITATIVE.
                    // fusedSwigluDown primary path calls multiply_tensor_with_fused_swiglu
                    // which handles tensor coherence internally.
                    if (!fusedSwigluDown(
                            *this,
                            scratch_gate_batch_[info.batch_idx].get(),
                            scratch_up_batch_[info.batch_idx].get(),
                            scratch_out_.get(),
                            down_gemm, /*m=*/1, d_model, intermediate,
                            getWorkspace()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CUDA decode SwiGLU/down projection failed for expert "
                                  << info.expert_id << " layer " << params_.layer_idx);
                        return false;
                    }

                    // GPU weighted accumulate: output += weight * scratch_out
                    kernel->weightedAddFromTensors(
                        params_.output, scratch_out_.get(), info.weight, d_model);
                    gpuExecution().publish(params_.output);
                }
            }
        }
        else
        {
            // CPU Phase 2: batch SwiGLU + fused down projections, followed by
            // either the serial ordered fold or raw canonical-slot publication.

            float *output = params_.output->mutable_data();
            if (!output)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU decode could not access its local output");
                return false;
            }
            auto publish_route = [&](const ActiveExpert &info,
                                     const float *route_output)
            {
                if (float *destination = canonical_destination(info))
                    std::copy_n(route_output, d_model, destination);
                else
                {
                    primitives::vec_axpy(
                        output,
                        route_output,
                        info.weight,
                        d_model);
                }
            };

            bool down_published_directly = cpu_prepared_decode_completed && cpu_canonical_rows;
            if (cpu_prepared_decode_completed)
            {
                if (!cpu_canonical_rows)
                    for (int i = 0; i < num_active; ++i)
                        publish_route(active_experts[i], scratch_down_batch_[i]->data());
            }
            else
            {
                if (!cpu_canonical_rows && static_cast<int>(scratch_down_batch_.size()) < num_active)
                    throw std::logic_error("CPU MoE decode exceeds its declared route capacity");

                // Validate down GEMM engines
                for (int i = 0; i < num_active; ++i)
                {
                    if (!activeDownGemm(active_experts[i].expert_id))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                                  << active_experts[i].expert_id << " (layer " << params_.layer_idx << ")");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }

                // Phase 2a: Apply SwiGLU for all experts.
                for (int i = 0; i < num_active; ++i)
                {
                    const auto &info = active_experts[i];
                    const float *gate_fp32 =
                        scratch_gate_batch_[info.batch_idx]->data();
                    const float *up_fp32 =
                        scratch_up_batch_[info.batch_idx]->data();
                    if (static_cast<int>(swiglu_scratch_batch_.size()) <= i ||
                        swiglu_scratch_batch_[i].size() < static_cast<size_t>(intermediate))
                        throw std::logic_error("CPU MoE decode exceeds declared FP32 SwiGLU storage");

                    /*
                     * Every CPU expert uses the GPU-aligned SwiGLU program.
                     * Placement, migration eligibility, and local route count
                     * cannot select arithmetic: a distributed participant may
                     * see one route while another sees several, yet both must
                     * materialize the same logical expert activation bytes.
                     */
                    primitives::compute_swiglu_gpu_aligned_expert_serial(
                        gate_fp32,
                        up_fp32,
                        swiglu_scratch_batch_[i].data(),
                        intermediate);
                }

                // Phase 2b: Use the canonical expert down-projection launcher
                // for every non-empty route set.  Route ownership is allowed to
                // leave a participant with one local expert, so cardinality
                // must not select a different arithmetic implementation.
                bool fused_ok = false;
                if (num_active >= 1)
                {
                    ITensorGemm::FusedExpertDownDesc down_descs[16];
                    for (int i = 0; i < num_active && i < 16; ++i)
                    {
                        const auto &info = active_experts[i];
                        down_descs[i].kernel =
                            activeDownGemm(info.expert_id);
                        down_descs[i].input =
                            swiglu_scratch_batch_[i].data();
                        down_descs[i].output = cpu_canonical_rows
                            ? canonical_destination(info)
                            : scratch_down_batch_[i]->mutable_data();
                        down_descs[i].n = d_model;
                    }
                    fused_ok =
                        activeDownGemm(active_experts[0].expert_id)
                            ->multiply_fused_expert_down(
                                down_descs,
                                num_active,
                                1,
                                intermediate,
                                bound_workspace_);
                }

                if (fused_ok)
                {
                    down_published_directly = cpu_canonical_rows != nullptr;
                    if (!down_published_directly)
                        for (int i = 0; i < num_active; ++i)
                            publish_route(active_experts[i], scratch_down_batch_[i]->data());
                }
                else
                {
                    // Standalone non-transaction CPU decode retains the exact
                    // per-expert SwiGLU/down implementation.
                    float *scratch_out_ptr = scratch_out_->mutable_data();
                    for (int i = 0; i < num_active; ++i)
                    {
                        const auto &info = active_experts[i];
                        ITensorGemm *down_gemm =
                            activeDownGemm(info.expert_id);

                        if (!fusedSwigluDown(
                                *this,
                                scratch_gate_batch_[info.batch_idx].get(),
                                scratch_up_batch_[info.batch_idx].get(),
                                scratch_out_.get(),
                                down_gemm,
                                /*m=*/1,
                                d_model,
                                intermediate,
                                getWorkspace()))
                        {
                            LOG_ERROR("[MoEExpertComputeStage] Decode SwiGLU/down projection failed for expert "
                                      << info.expert_id << " layer " << params_.layer_idx);
                            return false;
                        }

                        publish_route(info, scratch_out_ptr);
                    }
                }
            }

            if (down_published_directly &&
                PerfStatsCollector::isRecordFamilyEnabled("kernel", "cpu_moe_single_row_down_direct_publication"))
            {
                PerfStatsCollector::addCounter("kernel", "cpu_moe_single_row_down_direct_publication",
                    static_cast<double>(num_active), "gemm", "cpu",
                    {{"layout", cpu_packed_records ? "packed" : "dense"}});
            }
            if (cpu_packed_records &&
                !canonical_moe_route_record::writeRecordCount(
                    cpu_packed_records,
                    params_.canonical_route_contributions->numel(),
                    static_cast<size_t>(num_active)))
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU decode could not publish "
                          "its packed route record count"
                          << " layer=" << params_.layer_idx
                          << " count=" << num_active);
                return false;
            }

        } // end CPU Phase 2 else block

        if (!is_gpu)
        {
            // M=1 has its own real ActiveExpert entries, not the grouped CSR
            // schedule. Publish those completed selections into the same
            // retained view, deduplicating at most top_k<=16 without a node set.
            for (int active = 0; active < num_active; ++active)
            {
                const int expert = active_experts[active].expert_id;
                auto position = std::lower_bound(cpu_grouped_active_experts_.begin(),
                    cpu_grouped_active_experts_.end(), expert);
                if (position == cpu_grouped_active_experts_.end() || *position != expert)
                    cpu_grouped_active_experts_.insert(position, expert);
            }
        }
        LOG_TRACE("[MoEExpertComputeStage] Single-token decode (batched gate+up): " << num_active << " experts");
        if (is_gpu && !params_.output_registered_in_arena)
            gpuExecution().publish(params_.output);
        return true;
    }

    bool MoEExpertComputeStage::executeCPUGroupedDecodeEquivalentRows(
        IDeviceContext *ctx,
        CPUGroupedRowsPurpose purpose)
    {
        (void)ctx;
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row executor called for GPU device "
                      << params_.device_id.to_string());
            return false;
        }
        if (!params_.input || !params_.output ||
            !params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row executor missing tensors");
            return false;
        }

        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();
        if (!routing_idx_data || !routing_wt_data)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row executor could not access routing tensors");
            return false;
        }

        if (seq_len < 1 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts || top_k > 16)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row executor received invalid geometry: "
                      << "M=" << seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate
                      << " experts=" << num_experts
                      << " top_k=" << top_k);
            return false;
        }

        const bool borrowed_transport_hidden =
            sparse_overlay_binding_kind_ ==
                SparseOverlayBindingKind::HostPacketPublication &&
            sparse_overlay_cpu_hidden_source_ != nullptr &&
            sparse_overlay_cpu_hidden_source_row_capacity_ > 0 &&
            sparse_overlay_cpu_compact_to_source_row_.size() >=
                static_cast<size_t>(seq_len);
        if (sparse_overlay_binding_kind_ ==
                SparseOverlayBindingKind::HostPacketPublication &&
            !borrowed_transport_hidden)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] CPU host packet has no complete borrowed hidden-row binding");
            return false;
        }

        CPUGroupedMoESerialWorkspace::InvocationLease serial_workspace_lease;
        CPUGroupedMoESerialWorkspace *const serial_workspace =
            params_.cpu_grouped_serial_workspace.get();
        if (serial_workspace)
        {
            serial_workspace_lease = serial_workspace->acquire(
                seq_len,
                top_k,
                d_model,
                intermediate,
                num_experts,
                params_.layer_idx);
        }

        if (!cpu_grouped_backing_)
            throw std::logic_error("CPU MoE grouped execution requires setup-bound declared storage");
        auto &grouped_scratch_batch = cpu_grouped_backing_->batch;
        auto &grouped_scratch_gate = cpu_grouped_backing_->gate;
        auto &grouped_scratch_up = cpu_grouped_backing_->up;
        auto &grouped_scratch_out = cpu_grouped_backing_->output;
        auto &grouped_router_q8 = cpu_grouped_backing_->router_q8;
        auto &grouped_swiglu_q8 = cpu_grouped_backing_->swiglu_q8;
        auto &grouped_route_outputs = cpu_grouped_backing_->route_outputs;

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;

        const bool record_cpu_moe_phase_timings =
            PerfStatsCollector::isDomainEnabled("cpu_moe_grouped_rows");
        const auto route_schedule_start = record_cpu_moe_phase_timings
                                              ? PerfStatsCollector::Clock::now()
                                              : PerfStatsCollector::Clock::time_point{};

        const size_t route_slot_capacity =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (route_slot_capacity > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::logic_error("CPU grouped route schedule exceeds its signed row index representation");
        cpu_grouped_route_slots_.clear();
        cpu_grouped_route_slots_.reserve(route_slot_capacity);
        cpu_grouped_original_to_local_.assign(route_slot_capacity, -1);
        cpu_grouped_expert_counts_.assign(static_cast<size_t>(num_experts), 0);

        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        const bool verifier_rows =
            purpose == CPUGroupedRowsPurpose::MTPVerifier;
        const bool decode_rows =
            purpose == CPUGroupedRowsPurpose::Decode;
        const char *const grouped_phase = verifier_rows
            ? "verifier"
            : (decode_rows ? "decode" : "prefill");
        const bool has_replicas = params_.replica_set.num_replicated > 0;
        const bool has_prefill_mask =
            !params_.replica_set.prefill_mask.empty();
        if (!params_.expert_mask.empty() &&
            params_.expert_mask.size() != static_cast<size_t>(num_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row expert mask size "
                      << params_.expert_mask.size()
                      << " does not match num_experts=" << num_experts
                      << " layer=" << params_.layer_idx);
            return false;
        }
        if (has_prefill_mask &&
            params_.replica_set.prefill_mask.size() !=
                static_cast<size_t>(num_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row prefill mask size "
                      << params_.replica_set.prefill_mask.size()
                      << " does not match num_experts=" << num_experts
                      << " layer=" << params_.layer_idx);
            return false;
        }
        if (has_replicas &&
            (!params_.replica_set.hasLayerReplicaPlacement() ||
             params_.replica_set.base_ownership.expertCount() != num_experts ||
             params_.layer_idx < 0 ||
             params_.layer_idx >= params_.replica_set.base_ownership.layerCount()))
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row layered replica ownership does not match num_experts="
                      << num_experts
                      << " layer=" << params_.layer_idx);
            return false;
        }
        /*
         * Preserve full pre-ownership routing evidence for static apportionment
         * diagnosis.  The vector is stage-owned and grow-only, so enabling
         * PerfStats does not recreate one heap object per expert invocation.
         */
        const bool record_route_distribution =
            purpose == CPUGroupedRowsPurpose::OrdinaryPrefill &&
            PerfStatsCollector::isDomainEnabled("moe_static_apportionment") &&
            params_.my_socket_id == 0;
        if (borrowed_transport_hidden)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay",
                "cpu_transport_hidden_borrowed_rows",
                static_cast<double>(seq_len),
                grouped_phase,
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"source_row_capacity",
                  std::to_string(
                      sparse_overlay_cpu_hidden_source_row_capacity_)},
                 {"materialized_compact_rows", "0"},
                 {"ownership", "synchronous_host_packet"}});
        }
        if (record_route_distribution)
        {
            cpu_grouped_routed_rows_by_expert_.assign(
                static_cast<size_t>(num_experts), 0u);
        }

        for (int row = 0; row < seq_len; ++row)
        {
            int routing_int_indices[16] = {};
            bool compute_here[16] = {};
            const float *row_weights =
                routing_wt_data + static_cast<size_t>(row) * static_cast<size_t>(top_k);

            for (int route = 0; route < top_k; ++route)
            {
                const size_t flat_slot =
                    static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                    static_cast<size_t>(route);
                const float weight = row_weights[route];
                if (!std::isfinite(weight))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row non-finite route weight"
                              << " row=" << row
                              << " route=" << route
                              << " layer=" << params_.layer_idx);
                    return false;
                }
                if (weight == 0.0f)
                {
                    routing_int_indices[route] = -1;
                    continue;
                }

                const int expert_id = static_cast<int>(routing_idx_data[flat_slot]);
                routing_int_indices[route] = expert_id;
                if (expert_id < 0 || expert_id >= num_experts)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row invalid expert id "
                              << expert_id << " row=" << row
                              << " route=" << route
                              << " layer=" << params_.layer_idx
                              << " num_experts=" << num_experts);
                    return false;
                }
                if (record_route_distribution)
                {
                    ++cpu_grouped_routed_rows_by_expert_[
                        static_cast<size_t>(expert_id)];
                }
            }

            if (borrowed_transport_hidden)
            {
                /*
                 * The sparse collective already authenticated and filtered
                 * this participant's routes. Reapplying prefill masks or
                 * replica assignment here would create a second placement
                 * authority and could silently discard an admitted route.
                 */
                for (int route = 0; route < top_k; ++route)
                {
                    if (row_weights[route] == 0.0f)
                    {
                        compute_here[route] = false;
                        continue;
                    }
                    const int expert_id = routing_int_indices[route];
                    compute_here[route] = activeExpertResident(expert_id);
                }
            }
            else if (verifier_rows && has_replicas)
            {
                params_.replica_set.assignForToken(
                    routing_int_indices,
                    row_weights,
                    top_k,
                    params_.my_socket_id,
                    cpu_joint_invocation_layer_ ? cpu_joint_invocation_layer_->resident_mask : params_.expert_mask,
                    compute_here,
                    params_.layer_idx);
            }
            else
            {
                const auto &active_mask =
                    cpu_joint_invocation_layer_
                        ? cpu_joint_invocation_layer_->resident_mask
                        : params_.expert_mask;
                for (int route = 0; route < top_k; ++route)
                {
                    if (row_weights[route] == 0.0f)
                    {
                        compute_here[route] = false;
                        continue;
                    }
                    const int expert_id = routing_int_indices[route];
                    if (!verifier_rows && has_prefill_mask)
                    {
                        compute_here[route] =
                            params_.replica_set.prefill_mask[
                                static_cast<size_t>(expert_id)];
                    }
                    else if (!active_mask.empty())
                    {
                        bool computes_locally =
                            active_mask[static_cast<size_t>(expert_id)];
                        if (!verifier_rows && computes_locally && has_replicas &&
                            params_.replica_set.isReplicatedForLayer(
                                params_.layer_idx, expert_id) &&
                            params_.replica_set.ownerParticipant(
                                params_.layer_idx, expert_id) !=
                                params_.my_socket_id)
                        {
                            computes_locally = false;
                        }
                        compute_here[route] = computes_locally;
                    }
                    else
                    {
                        compute_here[route] =
                            expert_id >= local_start && expert_id < local_end;
                    }
                }
            }

            for (int route = 0; route < top_k; ++route)
            {
                if (!compute_here[route])
                    continue;

                const int expert_id = routing_int_indices[route];
                const int local_slot =
                    static_cast<int>(cpu_grouped_route_slots_.size());
                const size_t flat_slot =
                    static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                    static_cast<size_t>(route);
                cpu_grouped_original_to_local_[flat_slot] = local_slot;
                ++cpu_grouped_expert_counts_[static_cast<size_t>(expert_id)];
                cpu_grouped_route_slots_.push_back(
                    {row, route, expert_id, row_weights[route]});
            }
        }

        if (record_route_distribution)
        {
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                const uint32_t routed_rows =
                    cpu_grouped_routed_rows_by_expert_[
                        static_cast<size_t>(expert_id)];
                if (routed_rows == 0)
                    continue;
                PerfStatsCollector::addCounter(
                    "moe_static_apportionment",
                    "routed_rows",
                    static_cast<double>(routed_rows),
                    "prefill",
                    params_.device_id.toString(),
                    {{"layer", std::to_string(params_.layer_idx)},
                     {"expert", std::to_string(expert_id)},
                     {"tokens", std::to_string(seq_len)},
                     {"top_k", std::to_string(top_k)}});
            }
        }

        /*
         * Convert original-order route slots into one flat CSR schedule.  The
         * second pass is stable: slots for each expert appear in the same
         * row/top-k order in which routing produced them.
         */
        cpu_grouped_expert_offsets_.resize(
            static_cast<size_t>(num_experts) + 1u);
        cpu_grouped_expert_offsets_[0] = 0;
        cpu_grouped_active_experts_.clear();
        cpu_grouped_active_experts_.reserve(static_cast<size_t>(num_experts));
        int max_batch = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const int count =
                cpu_grouped_expert_counts_[static_cast<size_t>(expert_id)];
            cpu_grouped_expert_offsets_[static_cast<size_t>(expert_id) + 1u] =
                cpu_grouped_expert_offsets_[static_cast<size_t>(expert_id)] +
                count;
            if (count > 0)
                cpu_grouped_active_experts_.push_back(expert_id);
            max_batch = std::max(max_batch, count);
        }
        cpu_grouped_expert_write_offsets_ = cpu_grouped_expert_offsets_;
        if (cpu_grouped_expert_local_slots_.size() <
            cpu_grouped_route_slots_.size())
        {
            cpu_grouped_expert_local_slots_.resize(
                cpu_grouped_route_slots_.size());
        }
        if (cpu_grouped_local_to_expert_position_.size() <
            cpu_grouped_route_slots_.size())
        {
            cpu_grouped_local_to_expert_position_.resize(
                cpu_grouped_route_slots_.size());
        }
        for (size_t local_slot = 0;
             local_slot < cpu_grouped_route_slots_.size();
             ++local_slot)
        {
            const int expert_id =
                cpu_grouped_route_slots_[local_slot].expert_id;
            const int destination =
                cpu_grouped_expert_write_offsets_[
                    static_cast<size_t>(expert_id)]++;
            cpu_grouped_expert_local_slots_[static_cast<size_t>(destination)] =
                static_cast<int>(local_slot);
            cpu_grouped_local_to_expert_position_[local_slot] = destination;
        }

        /*
         * Materialize the exact expert-major source-row schedule once. The
         * transported CPU publication consumes this map directly, while the
         * floating path reuses the same storage for per-expert FP32 gathers.
         * The vector is grow-only and therefore allocates only when a graph
         * family first observes a larger certified route prefix.
         */
        if (cpu_grouped_token_indices_.size() <
            cpu_grouped_route_slots_.size())
        {
            cpu_grouped_token_indices_.resize(
                cpu_grouped_route_slots_.size());
        }
        if (cpu_grouped_router_row_indices_.size() <
            cpu_grouped_route_slots_.size())
        {
            cpu_grouped_router_row_indices_.resize(
                cpu_grouped_route_slots_.size());
        }
        for (size_t expert_position = 0;
             expert_position < cpu_grouped_route_slots_.size();
             ++expert_position)
        {
            const int local_slot =
                cpu_grouped_expert_local_slots_[expert_position];
            const int compact_row =
                cpu_grouped_route_slots_[
                    static_cast<size_t>(local_slot)].row;
            cpu_grouped_router_row_indices_[expert_position] = compact_row;
            cpu_grouped_token_indices_[expert_position] =
                borrowed_transport_hidden
                    ? sparse_overlay_cpu_compact_to_source_row_[
                          static_cast<size_t>(compact_row)]
                    : compact_row;
        }

        if (!ensureGemmEnginesForExperts(cpu_grouped_active_experts_))
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row execution is missing prepared GEMM engines");
            return false;
        }

        const size_t output_bytes =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model) * sizeof(float);
        TensorBase *const publication =
            params_.canonical_route_contributions
                ? params_.canonical_route_contributions
                : params_.output;
        const bool packed_canonical_records =
            params_.canonical_route_contributions &&
            params_.canonical_route_layout ==
                MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
        const size_t publication_bytes =
            params_.canonical_route_contributions
                ? output_bytes * static_cast<size_t>(top_k)
                : output_bytes;
        float *const canonical_routes = params_.canonical_route_contributions
            ? publication->mutable_data()
            : nullptr;
        if (params_.canonical_route_contributions && !canonical_routes)
            return false;
        if (canonical_routes && !packed_canonical_records &&
            publication_bytes / sizeof(float) > publication->numel())
            throw std::logic_error("CPU grouped route destinations exceed their declared publication");
        const auto publication_prepare_start =
            record_cpu_moe_phase_timings
                ? PerfStatsCollector::Clock::now()
                : PerfStatsCollector::Clock::time_point{};
        if (packed_canonical_records)
        {
            const size_t packed_capacity =
                canonical_moe_route_record::recordCapacity(
                    publication->numel(),
                    d_model);
            if (packed_capacity < route_slot_capacity ||
                !canonical_moe_route_record::writeRecordCount(
                    canonical_routes, publication->numel(), 0))
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU grouped packed route "
                          "publication buffer is undersized"
                          << " layer=" << params_.layer_idx
                          << " elements=" << publication->numel()
                          << " capacity=" << packed_capacity
                          << " required_records=" << route_slot_capacity
                          << " active_records="
                          << cpu_grouped_route_slots_.size());
                return false;
            }

            /*
             * The route schedule is complete and immutable at this point. Encode
             * every original slot before entering the OpenMP projection so
             * worker threads cannot throw through an OpenMP region or expose a
             * half-described packed transaction.
             */
            for (size_t local_slot = 0;
                 local_slot < cpu_grouped_route_slots_.size();
                 ++local_slot)
            {
                const auto &slot = cpu_grouped_route_slots_[local_slot];
                if (slot.row < 0 || slot.row >= seq_len ||
                    slot.route < 0 || slot.route >= top_k)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped packed route "
                              "schedule contains an invalid row or route"
                              << " layer=" << params_.layer_idx
                              << " local_slot=" << local_slot
                              << " row=" << slot.row
                              << " route=" << slot.route
                              << " seq_len=" << seq_len
                              << " top_k=" << top_k);
                    return false;
                }
                const size_t flat_slot =
                    static_cast<size_t>(slot.row) *
                        static_cast<size_t>(top_k) +
                    static_cast<size_t>(slot.route);
                float *const route_record =
                    canonical_moe_route_record::record(
                        canonical_routes,
                        local_slot,
                        d_model);
                if (flat_slot >= route_slot_capacity ||
                    !canonical_moe_route_record::writeFlatRouteSlot(
                        route_record,
                        d_model,
                        flat_slot))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped route slot "
                              "cannot be represented by packed publication"
                              << " layer=" << params_.layer_idx
                              << " local_slot=" << local_slot
                              << " flat_slot=" << flat_slot
                              << " route_slots=" << route_slot_capacity);
                    return false;
                }
            }
        }
        else if (params_.canonical_route_contributions &&
                 !(params_.canonical_route_layout ==
                       MoECanonicalRoutePublicationLayout::
                           DenseOriginalRouteSlots &&
                   sparse_overlay_binding_kind_ ==
                       SparseOverlayBindingKind::HostPacketPublication))
        {
            kernel->zeroBuffer(publication, publication_bytes);
        }
        const auto publication_prepare_finish =
            record_cpu_moe_phase_timings
                ? PerfStatsCollector::Clock::now()
                : PerfStatsCollector::Clock::time_point{};
        if (cpu_grouped_route_slots_.empty())
        {
            if (!params_.canonical_route_contributions)
                kernel->zeroBuffer(publication, publication_bytes);
            // Consult the original immutable domain policy BEFORE constructing
            // counter strings/tags; the collector cannot undo argument work.
            if (PerfStatsCollector::isDomainEnabled(verifier_rows ? "mtp" : (decode_rows ? "decode" : "prefill")))
            {
                PerfStatsCollector::addCounter(
                    verifier_rows ? "mtp" : (decode_rows ? "decode" : "prefill"),
                    verifier_rows
                        ? "moe_routed_grouped_decode_equivalent_verifier_prefill_rows"
                        : (decode_rows
                               ? "moe_routed_grouped_decode_rows"
                               : "moe_routed_grouped_decode_equivalent_prefill_rows"),
                    static_cast<double>(seq_len),
                    grouped_phase,
                    params_.device_id.toString(),
                    {{"stage", "routed_expert"},
                     {"route", "cpu_expert_slot_grouped"},
                     {"local_route_slots", "0"},
                     {"active_experts", "0"}});
            }
            return true;
        }

        /*
         * NativeQuantised gate/up/down projections share one activation contract.
         * Require the immediately preceding CPU router publication once for
         * the complete runtime batch, then gather canonical blocks for each
         * expert. Floating-point bundles retain their own grouped exact
         * implementation. Mixing either contract within an expert or across a
         * layer is fatal because route-dependent arithmetic would otherwise be
         * possible.
         */
        bool any_native_bundle = false;
        bool all_native_bundles = true;
        // Cache resolved NativeQuantised pointers once per active expert to
        // avoid 6 redundant RTTI dynamic_casts per active expert on hot prefill layers.
        struct ResolvedNativeExpertTriple
        {
            CPUNativeQuantisedGemmKernel *gate = nullptr;
            CPUNativeQuantisedGemmKernel *up = nullptr;
            CPUNativeQuantisedGemmKernel *down = nullptr;
        };
        thread_local std::vector<ResolvedNativeExpertTriple> resolved_native_experts;
        resolved_native_experts.resize(cpu_grouped_active_experts_.size());
        for (size_t idx = 0; idx < cpu_grouped_active_experts_.size(); ++idx)
        {
            const int expert_id = cpu_grouped_active_experts_[idx];
            auto *native_gate =
                dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                    activeGateGemm(expert_id));
            auto *native_up =
                dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                    activeUpGemm(expert_id));
            auto *native_down =
                dynamic_cast<CPUNativeQuantisedGemmKernel *>(
                    activeDownGemm(expert_id));
            resolved_native_experts[idx] = {native_gate, native_up, native_down};
            const bool any_native = native_gate || native_up || native_down;
            const bool all_native = native_gate && native_up && native_down;
            if (any_native != all_native)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row execution found mixed "
                          "gate/up/down kernel contracts for expert "
                          << expert_id << " layer=" << params_.layer_idx);
                return false;
            }
            any_native_bundle = any_native_bundle || all_native;
            all_native_bundles = all_native_bundles && all_native;
        }
        if (any_native_bundle != all_native_bundles)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row execution cannot mix "
                      "NativeQuantised and floating expert bundles in layer "
                      << params_.layer_idx);
            return false;
        }

        const int local_route_rows =
            static_cast<int>(cpu_grouped_route_slots_.size());
        const Q8_1Block *published_router_q8 = nullptr;
        CPUMoEKernel *cpu_moe_kernel = nullptr;
        const float *input_rows = nullptr;
        const bool direct_transport_publication =
            all_native_bundles &&
            params_.cpu_router_q8_input_publication ==
                CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
        const int router_q8_blocks_per_row =
            (d_model + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        if (all_native_bundles)
        {
            cpu_moe_kernel = dynamic_cast<CPUMoEKernel *>(kernel);
            input_rows = borrowed_transport_hidden
                             ? sparse_overlay_cpu_hidden_source_
                             : params_.input->data();
            if (!cpu_moe_kernel || !input_rows)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU NativeQuantised grouped-row execution "
                          "requires the production CPUMoEKernel router");
                return false;
            }
            if (!direct_transport_publication)
            {
                published_router_q8 =
                    cpu_moe_kernel->publishedRouterQ8Hidden(
                        input_rows,
                        seq_len,
                        d_model);
                if (!published_router_q8)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU NativeQuantised grouped-row execution "
                              "is missing the matching router Q8_1 publication for layer "
                              << params_.layer_idx << " seq_len=" << seq_len
                              << " d_model=" << d_model);
                    return false;
                }
            }
        }
        auto record_cpu_moe_phase_ns =
            [&](const char *name, uint64_t duration_ns)
        {
            if (!record_cpu_moe_phase_timings)
                return;
            PerfStatsCollector::recordTimingNs(
                "cpu_moe_grouped_rows",
                name,
                duration_ns,
                grouped_phase,
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"seq_len", std::to_string(seq_len)},
                 {"local_route_rows", std::to_string(local_route_rows)},
                 {"active_experts",
                  std::to_string(cpu_grouped_active_experts_.size())},
                 {"purpose", grouped_phase}});
        };
        auto record_cpu_moe_phase =
            [&](const char *name,
                PerfStatsCollector::Clock::time_point start)
        {
            if (!record_cpu_moe_phase_timings)
                return;
            const auto finish = PerfStatsCollector::Clock::now();
            const auto duration_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    finish - start).count();
            record_cpu_moe_phase_ns(
                name,
                static_cast<uint64_t>(duration_ns));
        };
        if (record_cpu_moe_phase_timings)
        {
            const auto route_schedule_finish =
                PerfStatsCollector::Clock::now();
            const auto complete_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    route_schedule_finish - route_schedule_start).count();
            const auto publication_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    publication_prepare_finish -
                    publication_prepare_start).count();
            record_cpu_moe_phase_ns(
                "route_schedule_and_bundle_validation",
                static_cast<uint64_t>(
                    std::max<int64_t>(
                        0,
                        complete_ns - publication_ns)));
            record_cpu_moe_phase_ns(
                "publication_prepare",
                static_cast<uint64_t>(publication_ns));
        }

        const auto workspace_prepare_start = record_cpu_moe_phase_timings
                                                 ? PerfStatsCollector::Clock::now()
                                                 : PerfStatsCollector::Clock::time_point{};
        const int scratch_rows =
            all_native_bundles ? local_route_rows : max_batch;
        if (!all_native_bundles && !canonical_routes)
        {
            const size_t required_route_output_elements =
                static_cast<size_t>(local_route_rows) *
                static_cast<size_t>(d_model);
            if (grouped_route_outputs.size() <
                required_route_output_elements)
            {
                throw std::logic_error("CPU MoE route outputs exceed declared storage");
            }
        }

        auto scratch_has_shape = [](const std::shared_ptr<FP32Tensor> &tensor,
                                    int min_rows,
                                    int cols) -> bool
        {
            if (!tensor)
                return false;
            const auto shape = tensor->shape();
            return shape.size() == 2u &&
                   shape[0] >= static_cast<size_t>(min_rows) &&
                   shape[1] == static_cast<size_t>(cols);
        };

        if (!all_native_bundles &&
            !scratch_has_shape(grouped_scratch_batch, max_batch, d_model))
        {
            throw std::logic_error("CPU MoE batch rows exceed declared storage");
        }
        if (!scratch_has_shape(grouped_scratch_gate, scratch_rows, intermediate))
        {
            throw std::logic_error("CPU MoE gate rows exceed declared storage");
        }
        if (!scratch_has_shape(grouped_scratch_up, scratch_rows, intermediate))
        {
            throw std::logic_error("CPU MoE up rows exceed declared storage");
        }
        if (!scratch_has_shape(grouped_scratch_out, scratch_rows, d_model))
        {
            throw std::logic_error("CPU MoE out rows exceed declared storage");
        }
        if ((!all_native_bundles && !grouped_scratch_batch) ||
            !grouped_scratch_gate || !grouped_scratch_up ||
            !grouped_scratch_out)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row scratch allocation failed");
            return false;
        }

        if (all_native_bundles)
        {
            const size_t required_router_q8_blocks =
                static_cast<size_t>(local_route_rows) *
                static_cast<size_t>(router_q8_blocks_per_row);
            if (grouped_router_q8.size() < required_router_q8_blocks)
                throw std::logic_error("CPU MoE router rows exceed declared storage");
            const int swiglu_q8_blocks_per_row =
                (intermediate + Q8_1Block::BLOCK_SIZE - 1) /
                Q8_1Block::BLOCK_SIZE;
            const size_t required_swiglu_q8_blocks =
                static_cast<size_t>(local_route_rows) *
                static_cast<size_t>(swiglu_q8_blocks_per_row);
            if (grouped_swiglu_q8.size() < required_swiglu_q8_blocks)
                throw std::logic_error("CPU MoE SwiGLU rows exceed declared storage");
        }
        else
        {
            if (cpu_grouped_token_indices_.size() <
                static_cast<size_t>(max_batch))
            {
                cpu_grouped_token_indices_.resize(
                    static_cast<size_t>(max_batch));
            }
        }
        batch_projections_.clear();
        batch_projections_.reserve(2u);
        record_cpu_moe_phase(
            "workspace_prepare_and_output_zero",
            workspace_prepare_start);

        if (all_native_bundles)
        {
            /*
             * Quantize only the <= seq_len unique live rows once on transported
             * endpoints (or borrow published_router_q8 directly on ordinary CPU
             * endpoints) instead of materializing an expert-major Q8 copy. The
             * single-OMP fused MoE FFN megakernel indexes those unique rows
             * directly via cpu_grouped_router_row_indices_ and fuses Gate+Up ->
             * SwiGLU -> Q8_1 -> Down with zero gate/up DRAM round-trip.
             */
            const auto router_q8_prepare_start =
                record_cpu_moe_phase_timings
                    ? PerfStatsCollector::Clock::now()
                    : PerfStatsCollector::Clock::time_point{};
            const Q8_1Block *unique_router_q8 = published_router_q8;
            if (direct_transport_publication)
            {
                const int source_rows = borrowed_transport_hidden
                    ? sparse_overlay_cpu_hidden_source_row_capacity_
                    : seq_len;
                const int *source_indices = borrowed_transport_hidden
                    ? sparse_overlay_cpu_compact_to_source_row_.data()
                    : nullptr;
                if (!cpu_moe_kernel->quantizeRouterQ8Rows(
                        input_rows,
                        source_rows,
                        d_model,
                        source_indices,
                        seq_len,
                        grouped_router_q8.data(),
                        grouped_router_q8.size()))
                {
                    LOG_ERROR(
                        "[MoEExpertComputeStage] CPU ExpertOverlay transport "
                        "failed to quantize unique canonical Q8_1 rows for layer "
                        << params_.layer_idx << " seq_len=" << seq_len
                        << " d_model=" << d_model);
                    return false;
                }
                unique_router_q8 = grouped_router_q8.data();
                if (PerfStatsCollector::isDomainEnabled("moe_overlay"))
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay",
                        "cpu_transport_router_q8_publications",
                        1.0,
                        grouped_phase,
                        params_.device_id.toString(),
                        {{"layer", std::to_string(params_.layer_idx)},
                         {"source_rows", std::to_string(source_rows)},
                         {"route_rows", std::to_string(local_route_rows)},
                         {"unique_rows", std::to_string(seq_len)},
                         {"d_model", std::to_string(d_model)},
                         {"layout", "unique_row_major"},
                         {"source", "fixed_capacity_ticket"}});
                }
                record_cpu_moe_phase(
                    "transported_router_q8_unique_publication",
                    router_q8_prepare_start);
            }

            using BatchedProjection =
                CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
            std::array<
                BatchedProjection,
                CPUNativeQuantisedGemmKernel::kMaxBatchedPrequantizedProjections>
                gate_up_projection_descriptors = {};
            std::array<
                BatchedProjection,
                CPUNativeQuantisedGemmKernel::kMaxBatchedPrequantizedProjections>
                down_projection_descriptors = {};
            int gate_up_projection_count = 0;
            int down_projection_count = 0;
            // Routing may make a prompt batch as small as one row or an MTP
            // verifier batch unusually large. Carry the graph's typed purpose
            // rather than guessing scheduling intent from the expert's M.
            const auto projection_purpose = [&]()
            {
                using ProjectionPurpose = cpu::native_quantised::ProjectionRowsPurpose;
                switch (purpose)
                {
                case CPUGroupedRowsPurpose::Decode: return ProjectionPurpose::Decode;
                case CPUGroupedRowsPurpose::OrdinaryPrefill: return ProjectionPurpose::Prefill;
                case CPUGroupedRowsPurpose::MTPVerifier: return ProjectionPurpose::MTPVerifier;
                }
                throw std::logic_error("CPU grouped expert execution has an unknown typed purpose");
            }();
            float *gate_rows = grouped_scratch_gate->mutable_data();
            float *up_rows = grouped_scratch_up->mutable_data();
            float *down_rows = canonical_routes ? canonical_routes : grouped_scratch_out->mutable_data();
            if (!gate_rows || !up_rows || !down_rows)
                return false;

            const int down_stride = d_model + static_cast<int>(packed_canonical_records);
            std::span<float> route_bank;
            std::optional<CPUNativeQuantisedGemmKernel::GroupedDownRouteOrder> down_route_order;
            if (canonical_routes)
            {
                const size_t bank_elements = packed_canonical_records
                    ? publication->numel() - canonical_moe_route_record::kTrailerElements
                    : publication_bytes / sizeof(float);
                route_bank = {canonical_routes, bank_elements};
                // The activation gather has completed. Reuse its index span
                // to bind one final destination per expert-major Down row.
                for (int local_slot = 0; local_slot < local_route_rows; ++local_slot)
                {
                    const auto &slot = cpu_grouped_route_slots_[local_slot];
                    const int expert_position = cpu_grouped_local_to_expert_position_[local_slot];
                    cpu_grouped_token_indices_[expert_position] =
                        packed_canonical_records ? local_slot : slot.row * top_k + slot.route;
                }
                const auto live_routes = static_cast<size_t>(local_route_rows);
                down_route_order = CPUNativeQuantisedGemmKernel::GroupedDownRouteOrder{
                    .expert_major_destinations = {cpu_grouped_token_indices_.data(), live_routes},
                    .local_to_expert = {cpu_grouped_local_to_expert_position_.data(), live_routes},
                    .destination_to_local = packed_canonical_records ? std::span<const int>{}
                        : std::span<const int>{cpu_grouped_original_to_local_.data(), route_slot_capacity},
                };
            }

            const int swiglu_q8_blocks_per_row =
                (intermediate + Q8_1Block::BLOCK_SIZE - 1) /
                Q8_1Block::BLOCK_SIZE;
            for (size_t idx = 0; idx < cpu_grouped_active_experts_.size(); ++idx)
            {
                const int expert_id = cpu_grouped_active_experts_[idx];
                const int expert_begin =
                    cpu_grouped_expert_offsets_[static_cast<size_t>(expert_id)];
                const int expert_end =
                    cpu_grouped_expert_offsets_[
                        static_cast<size_t>(expert_id) + 1u];
                const int expert_rows = expert_end - expert_begin;
                auto *native_gate = resolved_native_experts[idx].gate;
                auto *native_up = resolved_native_experts[idx].up;
                auto *native_down = resolved_native_experts[idx].down;
                if (!native_gate || !native_up || !native_down ||
                    gate_up_projection_count + 2 >
                        CPUNativeQuantisedGemmKernel::
                            kMaxBatchedPrequantizedProjections ||
                    down_projection_count >=
                        CPUNativeQuantisedGemmKernel::
                            kMaxBatchedPrequantizedProjections)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU NativeQuantised batched "
                              "FFN descriptor capacity exceeded in layer "
                              << params_.layer_idx);
                    return false;
                }

                const Q8_1Block *expert_input =
                    grouped_router_q8.data() +
                    static_cast<size_t>(expert_begin) *
                        static_cast<size_t>(router_q8_blocks_per_row);
                gate_up_projection_descriptors[
                    static_cast<size_t>(gate_up_projection_count++)] = {
                    .kernel = native_gate,
                    .input_q8 = expert_input,
                    .output = gate_rows +
                        static_cast<size_t>(expert_begin) *
                            static_cast<size_t>(intermediate),
                    .bias = nullptr,
                    .rows = expert_rows,
                    .n = intermediate,
                    .ldc = intermediate,
                    .purpose = projection_purpose,
                };
                gate_up_projection_descriptors[
                    static_cast<size_t>(gate_up_projection_count++)] = {
                    .kernel = native_up,
                    .input_q8 = expert_input,
                    .output = up_rows +
                        static_cast<size_t>(expert_begin) *
                            static_cast<size_t>(intermediate),
                    .bias = nullptr,
                    .rows = expert_rows,
                    .n = intermediate,
                    .ldc = intermediate,
                    .purpose = projection_purpose,
                };
                down_projection_descriptors[
                    static_cast<size_t>(down_projection_count++)] = {
                    .kernel = native_down,
                    .input_q8 = grouped_swiglu_q8.data() +
                        static_cast<size_t>(expert_begin) *
                            static_cast<size_t>(swiglu_q8_blocks_per_row),
                    .output = canonical_routes
                        ? down_rows
                        : down_rows + static_cast<size_t>(expert_begin) * d_model,
                    .bias = nullptr,
                    .rows = expert_rows,
                    .n = d_model,
                    .ldc = down_stride,
                    .purpose = projection_purpose,
                    .output_rows = canonical_routes
                        ? cpu::native_quantised::CPUProjectionOutputRows::indexed(
                              route_bank, down_stride, d_model,
                              {cpu_grouped_token_indices_.data() + expert_begin,
                               static_cast<size_t>(expert_rows)})
                        : cpu::native_quantised::CPUProjectionOutputRows{},
                };
            }

            const auto grouped_ffn_start = record_cpu_moe_phase_timings
                                               ? PerfStatsCollector::Clock::now()
                                               : PerfStatsCollector::Clock::time_point{};
            if (!CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn(
                    {gate_up_projection_descriptors.data(), static_cast<size_t>(gate_up_projection_count)},
                    unique_router_q8,
                    seq_len,
                    router_q8_blocks_per_row,
                    {cpu_grouped_router_row_indices_.data(), static_cast<size_t>(local_route_rows)},
                    d_model,
                    grouped_swiglu_q8.data(),
                    local_route_rows,
                    intermediate,
                    swiglu_q8_blocks_per_row,
                    {down_projection_descriptors.data(), static_cast<size_t>(down_projection_count)},
                    down_route_order))
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU NativeQuantised single-OMP "
                          "fused grouped FFN transaction failed in layer "
                          << params_.layer_idx);
                return false;
            }
            record_cpu_moe_phase(
                "native_quantised_grouped_ffn_transaction",
                grouped_ffn_start);

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_moe_native_quantised_layer_batched_expert_calls",
                    1.0,
                    grouped_phase,
                    "cpu",
                    {{"layer", std::to_string(params_.layer_idx)},
                     {"active_experts",
                      std::to_string(cpu_grouped_active_experts_.size())},
                     {"local_route_rows", std::to_string(local_route_rows)},
                     {"team_lifetime", "complete_moe_ffn"}});
            }
        }
        else
        {
            for (int expert_id : cpu_grouped_active_experts_)
            {
                const int expert_begin =
                    cpu_grouped_expert_offsets_[static_cast<size_t>(expert_id)];
                const int expert_end =
                    cpu_grouped_expert_offsets_[
                        static_cast<size_t>(expert_id) + 1u];
                const int expert_rows = expert_end - expert_begin;
                ITensorGemm *gate_gemm = activeGateGemm(expert_id);
                ITensorGemm *up_gemm = activeUpGemm(expert_id);
                ITensorGemm *down_gemm = activeDownGemm(expert_id);
                if (!gate_gemm || !up_gemm || !down_gemm)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row execution "
                              "found null GEMM for expert "
                              << expert_id << " layer=" << params_.layer_idx);
                    return false;
                }

                if (borrowed_transport_hidden)
                {
                    kernel->gatherTokenBatch(
                        sparse_overlay_cpu_hidden_source_,
                        grouped_scratch_batch->mutable_data(),
                        cpu_grouped_token_indices_.data() + expert_begin,
                        expert_rows,
                        d_model);
                }
                else
                {
                    for (int expert_row = 0;
                         expert_row < expert_rows;
                         ++expert_row)
                    {
                        const int local_slot =
                            cpu_grouped_expert_local_slots_[
                                static_cast<size_t>(expert_begin + expert_row)];
                        cpu_grouped_token_indices_[
                            static_cast<size_t>(expert_row)] =
                            cpu_grouped_route_slots_[
                                static_cast<size_t>(local_slot)].row;
                    }
                    kernel->gatherTokenBatchFromTensors(
                        params_.input,
                        grouped_scratch_batch.get(),
                        cpu_grouped_token_indices_.data(),
                        expert_rows,
                        d_model);
                }

                batch_projections_.clear();
                batch_projections_.push_back(
                    {gate_gemm, grouped_scratch_gate.get(), intermediate, nullptr, "gate"});
                batch_projections_.push_back(
                    {up_gemm, grouped_scratch_up.get(), intermediate, nullptr, "up"});

                const bool projected = verifier_rows && expert_rows > 1
                    ? gate_gemm->multiply_fused_verifier_rows_decode_equivalent(
                          grouped_scratch_batch.get(),
                          batch_projections_,
                          expert_rows,
                          d_model,
                          nullptr,
                          getWorkspace())
                    : gate_gemm->multiply_fused_tensor(
                          grouped_scratch_batch.get(),
                          batch_projections_,
                          expert_rows,
                          d_model,
                          nullptr,
                          getWorkspace());
                if (!projected)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row gate/up projection failed"
                              << " expert=" << expert_id
                              << " rows=" << expert_rows
                              << " layer=" << params_.layer_idx);
                    return false;
                }

                const bool down_ok = verifier_rows && expert_rows > 1
                    ? down_gemm->
                          multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                              grouped_scratch_gate.get(),
                              grouped_scratch_up.get(),
                              grouped_scratch_out.get(),
                              expert_rows,
                              d_model,
                              intermediate,
                              1.0f,
                              0.0f,
                              getWorkspace())
                    : fusedSwigluDown(
                          *this,
                          grouped_scratch_gate.get(),
                          grouped_scratch_up.get(),
                          grouped_scratch_out.get(),
                          down_gemm,
                          expert_rows,
                          d_model,
                          intermediate,
                          getWorkspace());
                if (!down_ok)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped-row SwiGLU/down projection failed"
                              << " expert=" << expert_id
                              << " rows=" << expert_rows
                              << " layer=" << params_.layer_idx);
                    return false;
                }

                const float *expert_down_rows = grouped_scratch_out->data();
                if (!expert_down_rows)
                    return false;
                for (int expert_row = 0;
                     expert_row < expert_rows;
                     ++expert_row)
                {
                    const int local_slot =
                        cpu_grouped_expert_local_slots_[
                            static_cast<size_t>(expert_begin + expert_row)];
                    float *destination = nullptr;
                    if (packed_canonical_records)
                    {
                        destination = canonical_moe_route_record::record(
                            canonical_routes, static_cast<size_t>(local_slot), d_model);
                    }
                    else if (canonical_routes)
                    {
                        const auto &slot = cpu_grouped_route_slots_[local_slot];
                        const auto flat_slot = static_cast<size_t>(slot.row) * top_k + slot.route;
                        destination = canonical_routes + flat_slot * d_model;
                    }
                    else
                    {
                        destination = grouped_route_outputs.data() + static_cast<size_t>(local_slot) * d_model;
                    }
                    // The floating projection owns an expert-major matrix.
                    // Transfer each finished row straight to its consumer.
                    std::copy_n(
                        expert_down_rows +
                            static_cast<size_t>(expert_row) *
                                static_cast<size_t>(d_model),
                        d_model, destination);
                }
            }
        }

        float *output = params_.output->mutable_data();
        if (!output)
            return false;
        const bool down_published_directly = all_native_bundles && canonical_routes;
        const float *native_down_rows =
            all_native_bundles && !down_published_directly ? grouped_scratch_out->data() : nullptr;
        if (all_native_bundles && !down_published_directly && !native_down_rows)
            return false;

        /*
         * Rows are independent. A complete local owner folds routes directly
         * with the exact serial top-k FMA order. A distributed CPU owner instead
         * publishes the raw expert result into its original route slot; the
         * collective preserves that slot identity and the canonical reducer
         * performs the same serial FMA loop after all owners are visible.
         */
        const auto accumulate_start = record_cpu_moe_phase_timings
                                          ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
        auto accumulate_rows = [&]()
        {
#pragma omp for schedule(static)
            for (int row = 0; row < seq_len; ++row)
            {
                float *row_output =
                    output + static_cast<size_t>(row) *
                        static_cast<size_t>(d_model);
                std::fill_n(row_output, d_model, 0.0f);
                for (int route = 0; route < top_k; ++route)
                {
                    const size_t flat_slot =
                        static_cast<size_t>(row) *
                            static_cast<size_t>(top_k) +
                        static_cast<size_t>(route);
                    const int local_slot =
                        cpu_grouped_original_to_local_[flat_slot];
                    if (local_slot < 0)
                        continue;
                    const auto &slot =
                        cpu_grouped_route_slots_[
                            static_cast<size_t>(local_slot)];
                    const int output_position = all_native_bundles
                        ? cpu_grouped_local_to_expert_position_[
                              static_cast<size_t>(local_slot)]
                        : local_slot;
                    const float *route_output = all_native_bundles
                        ? native_down_rows +
                              static_cast<size_t>(output_position) *
                                  static_cast<size_t>(d_model)
                        : grouped_route_outputs.data() +
                              static_cast<size_t>(output_position) *
                                  static_cast<size_t>(d_model);
                    primitives::vec_axpy(row_output, route_output, slot.weight, d_model);
                }
            }
        };
        if (!canonical_routes)
            OMP_WORKSHARE_REGION(accumulate_rows);
        if (packed_canonical_records &&
            !canonical_moe_route_record::writeRecordCount(
                canonical_routes, publication->numel(), static_cast<size_t>(local_route_rows)))
            throw std::logic_error("CPU grouped Down could not commit its admitted packed route count");
        if (down_published_directly && PerfStatsCollector::isDomainEnabled("kernel"))
            PerfStatsCollector::addCounter(
                "kernel", "cpu_moe_grouped_down_direct_publication_rows",
                static_cast<double>(local_route_rows), grouped_phase, "cpu",
                {{"layout", packed_canonical_records ? "packed" : "dense"}});
        record_cpu_moe_phase(
            canonical_routes
                ? "canonical_route_slot_publication"
                : "ordered_topk_accumulation",
            accumulate_start);

        if (PerfStatsCollector::isDomainEnabled(verifier_rows ? "mtp" : (decode_rows ? "decode" : "prefill")))
        {
            PerfStatsCollector::addCounter(
                verifier_rows ? "mtp" : (decode_rows ? "decode" : "prefill"),
                verifier_rows
                    ? "moe_routed_grouped_decode_equivalent_verifier_prefill_rows"
                    : (decode_rows
                           ? "moe_routed_grouped_decode_rows"
                           : "moe_routed_grouped_decode_equivalent_prefill_rows"),
                static_cast<double>(seq_len),
                grouped_phase,
                params_.device_id.toString(),
                {{"stage", "routed_expert"},
                 {"route", "cpu_expert_slot_grouped"},
                 {"local_route_slots", std::to_string(cpu_grouped_route_slots_.size())},
                 {"active_experts", std::to_string(cpu_grouped_active_experts_.size())},
                 {"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)}});
        }
        if (all_native_bundles)
        {
            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    verifier_rows
                        ? "cpu_moe_grouped_verifier_router_q8_reuse_calls"
                        : (decode_rows
                               ? "cpu_moe_grouped_decode_router_q8_reuse_calls"
                               : "cpu_moe_prefill_router_q8_reuse_calls"),
                    1.0,
                    "moe",
                    "cpu",
                    {{"seq_len", std::to_string(seq_len)},
                     {"top_k", std::to_string(top_k)},
                     {"active_experts", std::to_string(cpu_grouped_active_experts_.size())},
                     {"route", "cpu_expert_slot_grouped"},
                     {"purpose", grouped_phase}});
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;

        const bool is_gpu = params_.device_id.is_gpu();

        if (seq_len < 1)
            return false;
        if (!params_.input || !params_.output ||
            !params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill missing tensors");
            return false;
        }

        const bool has_prepared_expert_state =
            (cpu_joint_invocation_layer_ != nullptr) ||
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill requires prepared expert engines");
            return false;
        }
        if (PerfStatsCollector::isDomainEnabled("mtp"))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_decode_equivalent_verifier_prefill_runs",
                1.0,
                "verifier",
                params_.device_id.toString(),
                {{"stage", "routed_expert"},
                 {"route", seq_len == 1
                               ? (is_gpu ? "grouped_prefill"
                                         : "single_row_decode")
                               : (is_gpu ? "grouped_prefill"
                                         : "cpu_expert_slot_grouped")},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"seq_len", std::to_string(params_.seq_len)},
                 {"top_k", std::to_string(params_.top_k)}});
        }
        if (top_k > 16)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill top_k="
                      << top_k << " exceeds stack capacity");
            return false;
        }

        if (seq_len == 1)
        {
            struct ScopedSingleVerifierRow
            {
                Params &params;
                bool force_grouped_verifier_prefill_for_decode;
                bool force_decode_equivalent_verifier_prefill;
                bool require_device_routing_tensor_decode;
                int seq_len;

                ~ScopedSingleVerifierRow()
                {
                    params.force_grouped_verifier_prefill_for_decode =
                        force_grouped_verifier_prefill_for_decode;
                    params.force_decode_equivalent_verifier_prefill =
                        force_decode_equivalent_verifier_prefill;
                    params.require_device_routing_tensor_decode =
                        require_device_routing_tensor_decode;
                    params.seq_len = seq_len;
                }
            } restore{
                params_,
                params_.force_grouped_verifier_prefill_for_decode,
                params_.force_decode_equivalent_verifier_prefill,
                params_.require_device_routing_tensor_decode,
                params_.seq_len};

            /*
             * M=1 is the degenerate verifier bucket: there is no grouping economy
             * to harvest, so the production contract is the backend's exact
             * one-row decode implementation.  GPU participants still stay
             * device-resident by requiring the explicit routing tensor path; a
             * missing device route is a hard failure rather than a host replay.
             */
            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;
            params_.require_device_routing_tensor_decode = is_gpu;

            return executeSingleToken(ctx, MoEOverlayServicePhaseHint::GroupedVerifier);
        }

        if (!is_gpu)
        {
            return executeCPUGroupedDecodeEquivalentRows(
                ctx,
                CPUGroupedRowsPurpose::MTPVerifier);
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;
        if (!canUseFixedTopologyGroupedPrefill())
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent GPU verifier M="
                      << seq_len
                      << " requires the grouped fixed-topology prefill path; "
                      << "row replay is not a production fallback"
                      << " device=" << params_.device_id.to_string()
                      << " layer=" << params_.layer_idx
                      << " fullOwnership=" << hasFullLocalExpertOwnership()
                      << " allEnabled=" << expertMaskAllEnabled()
                      << " replicas=" << params_.replica_set.num_replicated);
            return false;
        }

        if (static_cast<int>(all_expert_ids_.size()) != num_experts)
        {
            all_expert_ids_.resize(static_cast<size_t>(num_experts));
            std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
        }

        const auto prefill_expert_ids = fixedTopologyPrefillExpertIds();
        const bool tables_ready =
            ensureGemmEnginesForExperts(prefill_expert_ids) &&
            ensureGroupedGateUpDescriptorTable(kernel, d_model, params_.expert_intermediate) &&
            ensureGroupedDownDescriptorTable(kernel, d_model, params_.expert_intermediate);
        if (!tables_ready)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent GPU verifier failed to prepare grouped tables"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        const bool grouped_ok = executeFixedTopologyGroupedPrefill(kernel, seq_len);
        if (grouped_ok)
            publishPublicationTarget();
        return grouped_ok;
    }

    void MoEExpertComputeStage::ensureGemmEnginesCached()
    {
        if (!cached_gate_gemm_.empty())
            return;

        const int num_experts = params_.num_experts;

        // Use pre-resolved engines from graph build time if available
        if (!params_.prepared_gate_gemm.empty())
        {
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            return;
        }

        // Phase C: Resolve from PreparedWeightStore if slab refs are cached
        if (params_.prepared_store && params_.gate_slab_ref.has_value())
        {
            cached_gate_gemm_.resize(num_experts, nullptr);
            cached_up_gemm_.resize(num_experts, nullptr);
            cached_down_gemm_.resize(num_experts, nullptr);

            for (int e = 0; e < num_experts; ++e)
            {
                if (!expertComputesLocally(e))
                    continue;
                cached_gate_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.gate_slab_ref, e);
                cached_up_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.up_slab_ref, e);
                cached_down_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.down_slab_ref, e);
            }

            LOG_DEBUG("[MoEExpertComputeStage] Resolved GEMM engines from PreparedWeightStore"
                      << " (layer " << params_.layer_idx << ")");
            return;
        }

        // All local experts must be prepared at graph-build time.
        // If we reach here, it means prepareExpertGemmEngines() was not called.
        LOG_ERROR("[MoEExpertComputeStage] GEMM engines not pre-resolved for layer "
                  << params_.layer_idx << ". All experts must be prepared at graph build time. "
                  << "Ensure prepareExpertGemmEngines() is called during graph construction.");

        // Initialize empty cache to avoid repeated error logging
        cached_gate_gemm_.resize(num_experts, nullptr);
        cached_up_gemm_.resize(num_experts, nullptr);
        cached_down_gemm_.resize(num_experts, nullptr);
    }

    bool MoEExpertComputeStage::ensureGemmEnginesForExperts(const std::vector<int> &expert_ids)
    {
        const int num_experts = params_.num_experts;
        if (expert_ids.empty())
            return true;

        if (!params_.device_id.is_gpu())
        {
            if (!cpu_joint_invocation_layer_)
                ensureGemmEnginesCached();
            for (int expert_id : expert_ids)
            {
                if (expert_id < 0 || expert_id >= num_experts)
                {
                    LOG_ERROR("[MoEExpertComputeStage] Invalid expert id " << expert_id
                                                                           << " for layer " << params_.layer_idx);
                    return false;
                }
                if (!activeGateGemm(expert_id) ||
                    !activeUpGemm(expert_id) ||
                    !activeDownGemm(expert_id))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Missing prepared CPU GEMM engine for expert "
                              << expert_id << " layer " << params_.layer_idx
                              << "; CPU expert repack fallback is disabled");
                    return false;
                }
            }
            return true;
        }

        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(num_experts))
        {
            params_.prepared_gate_gemm.assign(num_experts, nullptr);
            params_.prepared_up_gemm.assign(num_experts, nullptr);
            params_.prepared_down_gemm.assign(num_experts, nullptr);
        }

        std::vector<bool> needed(num_experts, false);
        bool has_missing = false;
        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= num_experts)
            {
                LOG_ERROR("[MoEExpertComputeStage] Invalid expert id " << expert_id
                                                                       << " for layer " << params_.layer_idx);
                return false;
            }
            needed[expert_id] = true;
            if (!params_.prepared_gate_gemm[expert_id] ||
                !params_.prepared_up_gemm[expert_id] ||
                !params_.prepared_down_gemm[expert_id])
            {
                has_missing = true;
            }
        }

        if (has_missing)
        {
            // Missing experts MUST come from transfer blobs (dynamic rebalancing).
            // No fallback to raw host tensor data — all local experts are prepared
            // upfront at graph-build time. Missing engines at this point means either
            // a newly-arrived expert via rebalancing (must have a transfer blob) or
            // an initialization error.
            if (!payload_provider_)
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing GEMM engines for layer "
                          << params_.layer_idx << " but no payload provider available. "
                          << "All local experts must be prepared at graph-build time.");
                return false;
            }

            auto provider_payloads = payload_provider_->payloadsForLayer(params_.layer_idx);
            if (provider_payloads.empty())
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing GEMM engines for layer "
                          << params_.layer_idx << " and no transfer blobs available. "
                          << "Ensure the unified registry prepared all local experts, "
                          << "or dynamic expert transfer blobs were registered.");
                return false;
            }

            auto ctx = buildWeightContext();
            if (!MoEExpertWeightService::registerAndPrepareNewExperts(ctx, needed, &provider_payloads))
                return false;
        }

        cached_gate_gemm_ = params_.prepared_gate_gemm;
        cached_up_gemm_ = params_.prepared_up_gemm;
        cached_down_gemm_ = params_.prepared_down_gemm;

        addPendingGpuDirectTransfersFromStore(expert_ids);
        if (!waitForPendingGpuDirectTransfers())
            return false;

        auto bind_if_needed = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            /*
             * Workspace planning resolves prepared engines before the executor
             * binds a graph stream. Stream propagation is therefore conditional
             * here; execute() performs the mandatory propagation immediately
             * before any kernel launch. Calling gpuStream() merely to test for a
             * binding would turn harmless planning into an implicit-stream
             * failure and blur the scheduler/compute ownership boundary.
             */
            if (!params_.device_id.is_gpu() || hasGPUStream())
                bindStageStream(gemm);
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };

        for (int expert_id : expert_ids)
        {
            if (!cached_gate_gemm_[expert_id] ||
                !cached_up_gemm_[expert_id] ||
                !cached_down_gemm_[expert_id])
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing prepared GPU GEMM engine for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }
            bind_if_needed(cached_gate_gemm_[expert_id]);
            bind_if_needed(cached_up_gemm_[expert_id]);
            bind_if_needed(cached_down_gemm_[expert_id]);
        }
        return true;
    }
} // namespace llaminar2
