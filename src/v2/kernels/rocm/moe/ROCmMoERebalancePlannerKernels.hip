/**
 * @file ROCmMoERebalancePlannerKernels.hip
 * @brief ROCm overlay rebalance planner, LLEP preplanner, dynamic ownership controller, and histogram packing kernels.
 */
#include "ROCmMoERebalanceCommon.hip.inc"

namespace
{
    /**
     * @brief Authenticate all durable transfer-slot claims in parallel.
     *
     * One lane scans a disjoint strided subset of the flattened
     * layer-by-expert directory. A deterministic tree reduction publishes one
     * complete diagnostic witness into graph-lifetime scratch. The kernel is
     * due-gated without consuming the controller lifecycle; the following
     * publisher remains the sole transaction-state mutator.
     */
    __global__ void device_rebalance_llep_claim_preflight_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        DeviceMoERebalanceConfigView config,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        DeviceMoELLEPLayerPlanScratchView *layer_plans)
    {
        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        if (!layer_plans)
            return;

        DeviceMoELLEPLayerPlanScratchView &publication = layer_plans[0];
        if (leader)
        {
            publication.claim_summary_ready = 0u;
            publication.claim_summary =
                rebalance_empty_transfer_slot_claim_summary();
        }
        __syncthreads();

        if (blockDim.x != static_cast<uint32_t>(kDeviceMoEMaxExperts) ||
            !runtime_layers ||
            !rebalance_config_ok(config) ||
            config.routed_assignment_policy !=
                kDeviceMoERebalanceAssignmentLeastLoadedResident ||
            !rebalance_maintenance_due_on_next_controller_edge(
                controller_state, config))
        {
            return;
        }

        __shared__ uint64_t shared_summary_storage[
            (sizeof(RebalanceTransferSlotClaimSummaryView) *
                 kDeviceMoEMaxExperts +
             sizeof(uint64_t) - 1u) /
            sizeof(uint64_t)];
        auto *const shared_summaries =
            reinterpret_cast<RebalanceTransferSlotClaimSummaryView *>(
                shared_summary_storage);
        shared_summaries[lane] =
            rebalance_transfer_slot_claim_summary_partial(
                runtime_layers,
                config,
                local_transfer_slots,
                local_transfer_slot_count,
                lane,
                blockDim.x);
        __syncthreads();

        for (uint32_t stride =
                 static_cast<uint32_t>(kDeviceMoEMaxExperts) / 2u;
             stride > 0u;
             stride >>= 1u)
        {
            if (lane < stride)
            {
                rebalance_merge_transfer_slot_claim_summary(
                    shared_summaries[lane],
                    shared_summaries[lane + stride]);
            }
            __syncthreads();
        }

        if (leader)
        {
            publication.claim_summary = shared_summaries[0];
            publication.claim_summary_ready = 1u;
        }
    }

    /**
     * @brief Collect routed-layer economy counters with one workgroup per layer.
     *
     * Each workgroup exclusively owns one runtime layer, including the optional
     * counter reset. Expert lanes cooperatively detect local hot replicas and
     * publish one immutable integer summary into graph-lifetime scratch.
     */
    __global__ void device_rebalance_llep_router_collect_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        DeviceMoERebalanceConfigView config,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        DeviceMoELLEPLayerPlanScratchView *layer_plans)
    {
        const uint32_t window_index = blockIdx.x;
        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : min(config.layer_window_count, config.num_layers);
        if (!layer_plans || window_index >= layer_window_count)
            return;

        DeviceMoELLEPLayerPlanScratchView &publication =
            layer_plans[window_index];
        if (leader)
        {
            publication.router_summary_ready = 0u;
            publication.router_summary = RebalanceRouterBenefitSummaryView{};
        }
        __syncthreads();

        if (blockDim.x != static_cast<uint32_t>(kDeviceMoEMaxExperts) ||
            !runtime_layers ||
            !rebalance_config_ok(config) ||
            config.routed_assignment_policy !=
                kDeviceMoERebalanceAssignmentLeastLoadedResident ||
            !rebalance_maintenance_due_on_next_controller_edge(
                controller_state, config))
        {
            return;
        }

        const uint32_t layer_window_start =
            config.num_layers == 0u
                ? 0u
                : config.layer_window_start % config.num_layers;
        const uint32_t layer =
            (layer_window_start + window_index) % config.num_layers;
        DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
        __shared__ uint64_t shared_summary_storage[
            (sizeof(RebalanceRouterBenefitSummaryView) +
             sizeof(uint64_t) - 1u) /
            sizeof(uint64_t)];
        auto &shared_summary =
            *reinterpret_cast<RebalanceRouterBenefitSummaryView *>(
                shared_summary_storage);
        __shared__ uint32_t shared_runtime_valid;
        __shared__ uint32_t shared_hot_replica[kDeviceMoEMaxExperts];

        if (leader)
        {
            shared_summary = RebalanceRouterBenefitSummaryView{};
            shared_runtime_valid =
                runtime.active_bank <= 1u &&
                        runtime.expert_count == config.num_experts &&
                        runtime.top_k == config.top_k &&
                        runtime.participant_id == config.participant_id &&
                        runtime.participant_count == config.participant_count
                    ? 1u
                    : 0u;
            if (shared_runtime_valid != 0u)
            {
                shared_summary.eligible_dispatches =
                    runtime.router_hot_cache_eligible_dispatches;
                shared_summary.used_dispatches =
                    runtime.router_hot_cache_used_dispatches;
                shared_summary.improved_dispatches =
                    runtime.router_hot_cache_improved_dispatches;
                shared_summary.default_load_spread_total =
                    runtime.router_hot_cache_default_load_spread_total;
                shared_summary.actual_load_spread_total =
                    runtime.router_hot_cache_actual_load_spread_total;
                shared_summary.load_spread_improvement_total =
                    runtime.router_hot_cache_load_spread_improvement_total;
                shared_summary.active_dispatches =
                    runtime.router_hot_cache_active_dispatches;
                shared_summary.miss_dispatches =
                    runtime.router_hot_cache_miss_dispatches;
                shared_summary.selected_expert_slots =
                    runtime.router_hot_cache_selected_expert_slots;
                shared_summary.replicated_selected_expert_slots =
                    runtime.router_hot_cache_replicated_selected_expert_slots;

                if ((config.flags &
                     kDeviceMoERebalanceFlagResetHistograms) != 0u)
                {
                    runtime.router_hot_cache_eligible_dispatches = 0ULL;
                    runtime.router_hot_cache_used_dispatches = 0ULL;
                    runtime.router_hot_cache_improved_dispatches = 0ULL;
                    runtime.router_hot_cache_default_load_spread_total = 0ULL;
                    runtime.router_hot_cache_actual_load_spread_total = 0ULL;
                    runtime.router_hot_cache_load_spread_improvement_total =
                        0ULL;
                    runtime.router_hot_cache_active_dispatches = 0ULL;
                    runtime.router_hot_cache_miss_dispatches = 0ULL;
                    runtime.router_hot_cache_selected_expert_slots = 0ULL;
                    runtime.router_hot_cache_replicated_selected_expert_slots =
                        0ULL;
                }
            }
            else
            {
                shared_summary.invalid_runtime_layers = 1u;
            }
        }
        __syncthreads();

        if (shared_runtime_valid == 0u)
        {
            if (leader)
            {
                publication.router_summary = shared_summary;
                publication.router_summary_ready = 1u;
            }
            return;
        }

        const uint32_t local_participant_bit =
            runtime_participant_bit(static_cast<int>(config.participant_id));
        uint32_t hot_replica = 0u;
        if (lane < config.num_experts)
        {
            const DeviceMoEPlacementBankView &active =
                runtime.banks[runtime.active_bank];
            const DeviceMoEExpertDescriptorView &descriptor =
                active.experts[lane];
            const uint32_t resident_mask =
                runtime_expert_resident_mask(
                    &runtime,
                    active,
                    static_cast<int>(lane));
            const bool local_resident =
                (resident_mask & local_participant_bit) != 0u;
            const bool owner_local =
                descriptor.owner_participant ==
                static_cast<int32_t>(config.participant_id);
            const bool multi_resident =
                (resident_mask & (resident_mask - 1u)) != 0u;
            hot_replica =
                local_resident && !owner_local && multi_resident ? 1u : 0u;
        }
        shared_hot_replica[lane] = hot_replica;
        __syncthreads();
        for (uint32_t stride =
                 static_cast<uint32_t>(kDeviceMoEMaxExperts) / 2u;
             stride > 0u;
             stride >>= 1u)
        {
            if (lane < stride)
            {
                shared_hot_replica[lane] |=
                    shared_hot_replica[lane + stride];
            }
            __syncthreads();
        }

        if (leader)
        {
            shared_summary.hot_cache_active_layers =
                shared_hot_replica[0];
            publication.router_summary = shared_summary;
            publication.router_summary_ready = 1u;
        }
    }

    /** Reduce independent routed-layer summaries into one publication. */
    __global__ void device_rebalance_llep_router_reduce_kernel(
        DeviceMoERebalanceConfigView config,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        DeviceMoELLEPLayerPlanScratchView *layer_plans)
    {
        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        if (!layer_plans ||
            blockDim.x != static_cast<uint32_t>(kDeviceMoEMaxExperts) ||
            !rebalance_config_ok(config) ||
            config.routed_assignment_policy !=
                kDeviceMoERebalanceAssignmentLeastLoadedResident ||
            !rebalance_maintenance_due_on_next_controller_edge(
                controller_state, config))
        {
            return;
        }

        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : min(config.layer_window_count, config.num_layers);
        RebalanceRouterBenefitSummaryView local{};
        if (lane < layer_window_count)
        {
            const auto &publication = layer_plans[lane];
            if (publication.router_summary_ready == 1u)
                local = publication.router_summary;
            else
                local.invalid_runtime_layers = 1u;
        }

        __shared__ uint64_t shared_summary_storage[
            (sizeof(RebalanceRouterBenefitSummaryView) *
                 kDeviceMoEMaxExperts +
             sizeof(uint64_t) - 1u) /
            sizeof(uint64_t)];
        auto *const shared_summaries =
            reinterpret_cast<RebalanceRouterBenefitSummaryView *>(
                shared_summary_storage);
        shared_summaries[lane] = local;
        __syncthreads();
        for (uint32_t stride =
                 static_cast<uint32_t>(kDeviceMoEMaxExperts) / 2u;
             stride > 0u;
             stride >>= 1u)
        {
            if (lane < stride)
            {
                rebalance_merge_router_benefit_summary(
                    shared_summaries[lane],
                    shared_summaries[lane + stride]);
            }
            __syncthreads();
        }

        if (leader)
        {
            layer_plans[0].router_summary_ready = 0u;
            layer_plans[0].router_summary = shared_summaries[0];
            layer_plans[0].router_summary_ready = 2u;
        }
    }

    /**
     * @brief Plan every LLEP wave layer concurrently into persistent scratch.
     *
     * One workgroup owns one layer and one lane owns one logical expert. The
     * planner does not mutate runtime placement or the public command buffer;
     * it publishes immutable candidates that the following controller consumes
     * in ascending wave order. Layer-level parallelism therefore cannot perturb
     * command order, transfer-slot assignment, or byte parity.
     */
    __global__ void device_rebalance_llep_preplan_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        const unsigned long long *gathered_histograms,
        DeviceMoERebalanceConfigView config,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        const DeviceMoERebalanceWaveStateView *wave_states,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count,
        DeviceMoELLEPLayerPlanScratchView *layer_plans)
    {
        const uint32_t window_index = blockIdx.x;
        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : min(config.layer_window_count, config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : min(config.layer_wave_count, layer_window_count);
        if (!layer_plans || window_index >= layer_wave_count)
            return;

        DeviceMoELLEPLayerPlanScratchView &output = layer_plans[window_index];
        if (leader)
        {
            output.magic = kDeviceMoERebalanceMagic;
            output.version = kDeviceMoERebalanceVersion;
            output.ready = 0u;
            output.planned = 0u;
            output.window_index = window_index;
            output.layer = kDeviceMoEInvalidSlot;
        }

        if (!runtime_layers || !gathered_histograms ||
            !rebalance_config_ok(config) ||
            config.routed_assignment_policy !=
                kDeviceMoERebalanceAssignmentLeastLoadedResident ||
            config.participant_id != config.root_participant ||
            !rebalance_maintenance_due_on_next_controller_edge(
                controller_state, config))
        {
            return;
        }

        const uint32_t command_wave_index =
            rebalance_active_command_wave_index(
                controller_state, config, command_buffer_count);
        const DeviceMoERebalanceWaveStateView *wave_state =
            wave_states ? wave_states + command_wave_index : nullptr;
        const uint32_t layer_window_start =
            config.layer_window_start % config.num_layers;
        uint32_t start_offset = 0u;
        if (wave_state && layer_window_count > 0u)
        {
            const uint32_t candidate =
                wave_state->next_start_layer % config.num_layers;
            const uint32_t candidate_offset =
                (candidate + config.num_layers - layer_window_start) %
                config.num_layers;
            if (candidate_offset < layer_window_count)
                start_offset = candidate_offset;
        }
        const uint32_t layer =
            (layer_window_start +
             ((start_offset + window_index) % layer_window_count)) %
            config.num_layers;
        const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
        const bool runtime_ok =
            runtime.active_bank <= 1u &&
            runtime.expert_count == config.num_experts &&
            runtime.top_k == config.top_k &&
            runtime.participant_id == config.participant_id &&
            runtime.participant_count == config.participant_count;
        if (!runtime_ok)
            return;

        __shared__ uint64_t shared_expert_counts[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_owner_participants[kDeviceMoEMaxExperts];
        __shared__ uint32_t
            shared_resident_participant_masks[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_sorted_experts[kDeviceMoEMaxExperts];
        __shared__ uint64_t
            shared_pending_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t
            shared_assigned_load[kDeviceMoEMaxParticipants];
        static_assert(
            sizeof(llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer) ==
            sizeof(uint4));
        __shared__ uint4 shared_transfer_storage[kDeviceMoEMaxExperts];
        auto *const shared_transfers =
            reinterpret_cast<
                llaminar2::least_loaded_ep::LeastLoadedExpertWeightTransfer *>(
                shared_transfer_storage);
        using LLEPStatus =
            llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus;
        __shared__ uint64_t shared_status_storage[
            (sizeof(LLEPStatus) + sizeof(uint64_t) - 1u) / sizeof(uint64_t)];
        auto &shared_status =
            *reinterpret_cast<LLEPStatus *>(shared_status_storage);
        __shared__ uint32_t shared_planned;

        const DeviceMoEPlacementBankView &active =
            runtime.banks[runtime.active_bank];
        if (lane < config.num_experts)
        {
            const uint64_t count =
                rebalance_global_count(
                    gathered_histograms, config, window_index, lane);
            const auto &descriptor = active.experts[lane];
            const uint32_t resident_mask =
                active.resident_participant_mask[lane] &
                runtime_valid_participant_mask(config.participant_count);
            const int32_t owner =
                descriptor.owner_participant >= 0 &&
                        descriptor.owner_participant <
                            static_cast<int32_t>(config.participant_count)
                    ? descriptor.owner_participant
                    : rebalance_first_resident_participant(
                          resident_mask,
                          config.participant_count,
                          -1,
                          -1);
            shared_expert_counts[lane] = count;
            shared_owner_participants[lane] =
                owner >= 0 ? static_cast<uint32_t>(owner) : 0u;
            shared_resident_participant_masks[lane] = resident_mask;
        }
        __syncthreads();

        rebalance_sort_llep_experts_parallel(
            shared_sorted_experts,
            shared_expert_counts,
            config.num_experts,
            lane);

        if (leader)
        {
            llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentConfig
                llep_config{};
            llep_config.expert_count = config.num_experts;
            llep_config.participant_count = config.participant_count;
            llep_config.alpha_numerator = config.llep_alpha_numerator;
            llep_config.alpha_denominator = config.llep_alpha_denominator;
            llep_config.lambda_numerator = config.llep_lambda_numerator;
            llep_config.lambda_denominator = config.llep_lambda_denominator;
            llep_config.min_spread_improvement =
                config.min_load_spread_improvement;
            llep_config.min_spread_improvement_divisor =
                config.min_load_spread_improvement_divisor;
            llep_config.min_spread_improvement_per_critical_path_slot =
                config.min_wave_spread_improvement_per_payload_slot;
            llep_config.min_foreign_rows_per_critical_path_slot =
                config.min_foreign_rows_per_critical_path_payload_slot;
            llep_config.enable_balanced_skip =
                config.llep_enable_balanced_skip != 0u;
            llep_config.max_weight_transfers =
                min(payload_slot_capacity, plan_capacity);
            if (config.dynamic_max_plan_entries_per_wave > 0u)
            {
                llep_config.max_weight_transfers = min(
                    llep_config.max_weight_transfers,
                    config.dynamic_max_plan_entries_per_wave);
            }

            llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentWorkspace
                workspace{};
            workspace.sorted_experts = shared_sorted_experts;
            workspace.pending_load = shared_pending_load;
            workspace.assigned_load = shared_assigned_load;
            shared_planned =
                llaminar2::least_loaded_ep::planLeastLoadedExpertWeightTransfers(
                    shared_expert_counts,
                    shared_owner_participants,
                    llep_config,
                    workspace,
                    shared_transfers,
                    kDeviceMoEMaxExperts,
                    shared_status,
                    shared_resident_participant_masks,
                    /*workspace_experts_are_sorted=*/true)
                    ? 1u
                    : 0u;
        }
        __syncthreads();

        if (lane < static_cast<uint32_t>(kDeviceMoEMaxExperts))
        {
            output.transfers[lane] =
                lane < shared_status.weight_transfer_count
                    ? shared_transfers[lane]
                    : llaminar2::least_loaded_ep::
                          LeastLoadedExpertWeightTransfer{};
        }
        if (lane < static_cast<uint32_t>(kDeviceMoEMaxParticipants))
        {
            output.assigned_participant_load[lane] =
                lane < config.participant_count
                    ? shared_assigned_load[lane]
                    : 0ULL;
        }
        __syncthreads();
        if (leader)
        {
            output.planner_status = shared_status;
            output.layer = layer;
            output.planned = shared_planned;
            output.ready = 1u;
        }
    }

    /**
     * @brief Publish one captured device-owned rebalance transaction.
     *
     * @tparam LeastLoadedAssignment Whether this graph was captured for the
     *         LLEP least-loaded-resident policy. Assignment policy is immutable
     *         graph topology, so specialization removes unrelated static-owner
     *         candidate state from the production LLEP kernel.
     */
    template <bool LeastLoadedAssignment>
    __global__ __launch_bounds__(256, 1)
    void device_rebalance_controller_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const unsigned long long *gathered_histograms,
        DeviceMoERebalanceStatusView *status,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        DeviceMoERebalanceWaveStateView *wave_state,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        const DeviceMoELLEPLayerPlanScratchView *llep_layer_plans,
        DeviceMoEPlacementBankView *placement_plan_scratch,
        uint32_t command_buffer_count,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count)
    {
        if (blockIdx.x != 0)
            return;

        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        __shared__ uint32_t shared_command_wave_index;
        __shared__ uint32_t shared_abort;
        __shared__ uint32_t shared_layer_invalid;
        __shared__ uint32_t shared_selected[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_selected_count;
        __shared__ uint32_t shared_local_replicas;
        __shared__ uint32_t shared_plan_epoch;
        __shared__ unsigned long long shared_candidate_values[kDeviceMoEMaxExperts];
        __shared__ unsigned long long shared_candidate_counts[kDeviceMoEMaxExperts];
        __shared__ unsigned long long shared_candidate_improvements[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_experts[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_arrival_considered[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_candidate_arrival_below_floor[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_stop_rank;
        __shared__ uint32_t shared_post_policy_resident_mask[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_physical_source_resident_mask[kDeviceMoEMaxExperts];
        __shared__ uint64_t shared_current_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_candidate_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_pre_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_post_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_owner_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_expert_counts[kDeviceMoEMaxExperts];
        __shared__ int32_t shared_expert_owners[kDeviceMoEMaxExperts];
        using LLEPStatus =
            llaminar2::least_loaded_ep::LeastLoadedExpertAssignmentStatus;
        __shared__ uint64_t llep_status_storage[
            (sizeof(LLEPStatus) + sizeof(uint64_t) - 1u) / sizeof(uint64_t)];
        auto &llep_status =
            *reinterpret_cast<LLEPStatus *>(llep_status_storage);
        __shared__ uint32_t
            shared_expert_transfer_backed_participant_mask[kDeviceMoEMaxExperts];
        __shared__ uint64_t shared_required_load_spread_improvement;
        __shared__ uint32_t shared_count_bound_prunes;
        __shared__ uint32_t shared_destination_replica_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_destination_transfer_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_source_payload_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t
            shared_active_transfer_slot_counts[kDeviceMoEMaxParticipants];
        static_assert(
            kDeviceMoEMaxExperts % kDeviceMoEControllerWaveSize == 0);
        __shared__ RebalanceRouterTelemetryPartial
            shared_router_telemetry[
                kDeviceMoEMaxExperts / kDeviceMoEControllerWaveSize];

        if (leader)
        {
            shared_command_wave_index =
                rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
            shared_abort = 0u;
        }
        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
            participant += blockDim.x)
        {
            shared_current_policy_load[participant] = 0ULL;
            shared_candidate_policy_load[participant] = 0ULL;
            shared_pre_policy_load[participant] = 0ULL;
            shared_post_policy_load[participant] = 0ULL;
            shared_owner_policy_load[participant] = 0ULL;
            shared_destination_replica_counts[participant] = 0u;
            shared_destination_transfer_slot_counts[participant] = 0u;
            shared_source_payload_slot_counts[participant] = 0u;
            shared_active_transfer_slot_counts[participant] =
                gathered_histograms &&
                        config.num_experts > 0u &&
                        rebalance_layer_wave_count(config) > 0u &&
                        participant < config.participant_count
                    ? rebalance_collected_active_transfer_slot_count(
                          gathered_histograms, config, participant)
                    : 0u;
        }
        for (int expert = static_cast<int>(lane);
             expert < kDeviceMoEMaxExperts;
             expert += static_cast<int>(blockDim.x))
        {
            shared_expert_counts[expert] = 0ULL;
            shared_expert_owners[expert] = -1;
            shared_expert_transfer_backed_participant_mask[expert] = 0u;
        }
        __syncthreads();

        const uint32_t command_wave_index = shared_command_wave_index;
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;
        if (wave_state)
            wave_state += command_wave_index;

        if (leader)
        {
            if (status)
            {
                DeviceMoERebalanceStatusView zero_status{};
                *status = zero_status;
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
            }
            else
            {
                shared_abort = 1u;
            }

            /*
             * Physical residency is an invariant, not a scheduling hint.
             * Audit it before cadence/window early-outs so corrupt transfer
             * claims cannot hide behind WindowNotReady and current-batch
             * movement evidence remains observable on a skipped window.
             */
            if (shared_abort == 0u && !rebalance_config_ok(config))
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidConfig;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && !runtime_layers)
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && !gathered_histograms)
            {
                status->status_code = kDeviceMoERebalanceStatusMissingHistogram;
                shared_abort = 1u;
            }
            if (shared_abort == 0u)
            {
                if constexpr (LeastLoadedAssignment)
                {
                    const bool preflight_valid =
                        llep_layer_plans &&
                        llep_layer_plans[0].magic == kDeviceMoERebalanceMagic &&
                        llep_layer_plans[0].version ==
                            kDeviceMoERebalanceVersion &&
                        llep_layer_plans[0].claim_summary_ready == 1u;
                    if (!preflight_valid)
                    {
                        status->status_code =
                            kDeviceMoERebalanceStatusInvalidRuntime;
                        shared_abort = 1u;
                    }
                    else
                    {
                        rebalance_publish_transfer_slot_claim_summary_fields(
                            status,
                            llep_layer_plans[0].claim_summary);
                    }
                }
                else
                {
                    rebalance_publish_transfer_slot_claim_summary(
                        status,
                        runtime_layers,
                        config,
                        local_transfer_slots,
                        local_transfer_slot_count);
                }
            }
            if (shared_abort == 0u &&
                (status->prefill_duplicate_transfer_slot_claims != 0u ||
                 status->prefill_invalid_transfer_slot_claims != 0u ||
                 status->prefill_active_transfer_slot_experts >
                     config.active_transfer_slot_capacity))
            {
                status->status_code =
                    kDeviceMoERebalanceStatusInvalidRuntime;
                shared_abort = 1u;
            }

            if (shared_abort == 0u)
            {
                const uint32_t boundary =
                    consume_rebalance_maintenance_boundary(
                        controller_state,
                        config);
                if (boundary == 0u)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
                else if (boundary != 1u)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    shared_abort = 1u;
                }
            }

            if (shared_abort == 0u &&
                rebalance_active_wave_busy_for_new_plan(
                    controller_state,
                    config,
                    command_wave_index))
            {
                status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                status->skipped_not_ready = 1u;
                status->skipped_busy_wave = 1u;
                shared_abort = 1u;
            }

            if (plan_count)
                *plan_count = 0u;

            if (command_header)
            {
                command_header->magic = kDeviceMoERebalanceMagic;
                command_header->version = kDeviceMoERebalanceVersion;
                command_header->epoch = 0u;
                command_header->phase = kDeviceMoERebalancePhasePlanAssignments;
                command_header->command_count = 0u;
                command_header->command_capacity = plan_capacity;
                command_header->participant_id = config.participant_id;
                command_header->participant_count = config.participant_count;
            }

            if (wave_state)
            {
                const bool keep_cursor =
                    wave_state->magic == kDeviceMoERebalanceMagic &&
                    wave_state->version == kDeviceMoERebalanceVersion &&
                    wave_state->participant_id == config.participant_id &&
                    wave_state->participant_count == config.participant_count;
                const uint32_t next_start = keep_cursor ? wave_state->next_start_layer : 0u;
                wave_state->magic = kDeviceMoERebalanceMagic;
                wave_state->version = kDeviceMoERebalanceVersion;
                wave_state->epoch = 0u;
                wave_state->next_start_layer = next_start;
                wave_state->planned_start_layer = 0u;
                wave_state->planned_layer_count = 0u;
                wave_state->command_capacity = plan_capacity;
                wave_state->participant_id = config.participant_id;
                wave_state->participant_count = config.participant_count;
                wave_state->requested_payload_slots = 0u;
                wave_state->payload_bucket_slots = 0u;
                wave_state->payload_bucket_index = 0u;
                wave_state->payload_bucket_overflow = 0u;
            }

            if (shared_abort == 0u)
            {
                const unsigned long long observed_slots =
                    rebalance_window_observed_slots(gathered_histograms, config);
                const unsigned long long required_slots =
                    rebalance_window_required_slots(config);
                status->window_ready_slots =
                    rebalance_clamp_u64_to_u32(observed_slots);
                status->window_required_slots =
                    rebalance_clamp_u64_to_u32(required_slots);
                if (observed_slots < required_slots)
                {
                    status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
            }
        }
        __syncthreads();
        if (shared_abort != 0u)
            return;

        const uint32_t participant_bit = runtime_participant_bit(static_cast<int>(config.participant_id));
        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const bool plan_missing_arrivals =
            (config.flags & kDeviceMoERebalanceFlagPlanMissingArrivals) != 0u;
        const bool defer_runtime_apply =
            (config.flags & kDeviceMoERebalanceFlagDeferRuntimeApply) != 0u;
        if (defer_runtime_apply && !placement_plan_scratch)
        {
            if (leader && status)
                status->status_code = kDeviceMoERebalanceStatusInvalidRuntime;
            return;
        }
        const bool domain_root_planning = defer_runtime_apply && plan_missing_arrivals;
        constexpr bool least_loaded_assignment = LeastLoadedAssignment;
        const bool hot_replica_cache =
            (config.flags & kDeviceMoERebalanceFlagHotReplicaCache) != 0u;
        const bool collect_load_stats =
            (config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u;
        uint32_t changed_layers = 0;
        uint32_t selected_replicas = 0;
        uint32_t skipped_no_resident = 0;
        uint32_t skipped_no_improvement = 0;
        uint32_t dynamic_ownership_swap_attempts = 0;
        uint32_t dynamic_ownership_swap_accepts = 0;
        uint32_t dynamic_ownership_swap_rejections = 0;
        uint32_t candidate_arrivals_considered = 0;
        uint32_t candidate_arrivals_below_floor = 0;
        uint32_t candidate_arrivals_pruned_by_count_bound = 0;
        uint32_t llep_assignment_span_count = 0u;
        uint32_t llep_weight_transfer_count = 0u;
        uint32_t llep_standard_ep_selected = 0u;
        uint32_t llep_skipped_balanced = 0u;
        uint32_t llep_skipped_insufficient_spread_improvement = 0u;
        uint32_t llep_skipped_insufficient_foreign_rows = 0u;
        uint32_t llep_min_chunk_skips = 0u;
        uint32_t llep_forced_spills = 0u;
        uint64_t candidate_load_spread_improvement_total = 0ULL;
        uint64_t candidate_load_spread_improvement_max = 0ULL;
        uint64_t accepted_load_spread_improvement_total = 0ULL;
        uint64_t accepted_load_spread_improvement_max = 0ULL;
        uint64_t llep_native_rows = 0ULL;
        uint64_t llep_spilled_rows = 0ULL;
        uint64_t llep_required_spread_improvement = 0ULL;
        uint64_t llep_required_foreign_rows = 0ULL;
        uint64_t router_hot_cache_eligible_dispatches = 0ULL;
        uint64_t router_hot_cache_used_dispatches = 0ULL;
        uint64_t router_hot_cache_improved_dispatches = 0ULL;
        uint64_t router_hot_cache_default_load_spread_total = 0ULL;
        uint64_t router_hot_cache_actual_load_spread_total = 0ULL;
        uint64_t router_hot_cache_load_spread_improvement_total = 0ULL;
        uint64_t router_hot_cache_active_dispatches = 0ULL;
        uint64_t router_hot_cache_miss_dispatches = 0ULL;
        uint64_t router_hot_cache_selected_expert_slots = 0ULL;
        uint64_t router_hot_cache_replicated_selected_expert_slots = 0ULL;
        uint32_t hot_cache_active_layers = 0u;
        uint64_t pre_wave_load_total = 0ULL;
        uint64_t pre_wave_load_spread = 0ULL;
        uint64_t post_wave_load_total = 0ULL;
        uint64_t post_wave_load_spread = 0ULL;
        uint32_t invalid_layers = 0;
        uint32_t last_epoch = 0;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : ((config.layer_window_count < config.num_layers)
                       ? config.layer_window_count
                       : config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : ((config.layer_wave_count < layer_window_count)
                       ? config.layer_wave_count
                       : layer_window_count);
        const uint32_t layer_window_start =
            config.num_layers == 0u ? 0u : (config.layer_window_start % config.num_layers);
        uint32_t start_offset = 0u;
        if (wave_state && layer_window_count > 0u)
        {
            const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
            const uint32_t candidate_offset =
                (candidate + config.num_layers - layer_window_start) % config.num_layers;
            if (candidate_offset < layer_window_count)
                start_offset = candidate_offset;
        }
        if (leader)
        {
            shared_plan_epoch =
                (defer_runtime_apply &&
                 rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                 controller_state->next_epoch > 0u)
                    ? controller_state->next_epoch
                    : 1u;
        }
        __syncthreads();

        /*
         * Router-benefit counters describe the previously active replica set.
         * Aggregate them over the full routed layer window before planning the
         * next layer wave so rolling waves cannot advance past their own payback
         * signal.
         */
        if constexpr (LeastLoadedAssignment)
        {
            if (leader)
            {
                const bool router_preflight_valid =
                    llep_layer_plans &&
                    llep_layer_plans[0].magic == kDeviceMoERebalanceMagic &&
                    llep_layer_plans[0].version ==
                        kDeviceMoERebalanceVersion &&
                    llep_layer_plans[0].router_summary_ready == 2u;
                if (!router_preflight_valid)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    shared_abort = 1u;
                }
                else
                {
                    const auto &router_summary =
                        llep_layer_plans[0].router_summary;
                    router_hot_cache_eligible_dispatches =
                        router_summary.eligible_dispatches;
                    router_hot_cache_used_dispatches =
                        router_summary.used_dispatches;
                    router_hot_cache_improved_dispatches =
                        router_summary.improved_dispatches;
                    router_hot_cache_default_load_spread_total =
                        router_summary.default_load_spread_total;
                    router_hot_cache_actual_load_spread_total =
                        router_summary.actual_load_spread_total;
                    router_hot_cache_load_spread_improvement_total =
                        router_summary.load_spread_improvement_total;
                    router_hot_cache_active_dispatches =
                        router_summary.active_dispatches;
                    router_hot_cache_miss_dispatches =
                        router_summary.miss_dispatches;
                    router_hot_cache_selected_expert_slots =
                        router_summary.selected_expert_slots;
                    router_hot_cache_replicated_selected_expert_slots =
                        router_summary.replicated_selected_expert_slots;
                    hot_cache_active_layers =
                        router_summary.hot_cache_active_layers;
                }
            }
        }
        else
        {
            for (uint32_t window_index = lane;
                 window_index < layer_window_count;
                 window_index += blockDim.x)
            {
                const uint32_t layer =
                    (layer_window_start + window_index) % config.num_layers;
                DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
                const bool runtime_ok =
                    runtime.active_bank <= 1u &&
                    runtime.expert_count == config.num_experts &&
                    runtime.top_k == config.top_k &&
                    runtime.participant_id == config.participant_id &&
                    runtime.participant_count == config.participant_count;
                if (!runtime_ok)
                    continue;

            router_hot_cache_eligible_dispatches +=
                runtime.router_hot_cache_eligible_dispatches;
            router_hot_cache_used_dispatches +=
                runtime.router_hot_cache_used_dispatches;
            router_hot_cache_improved_dispatches +=
                runtime.router_hot_cache_improved_dispatches;
            router_hot_cache_default_load_spread_total +=
                runtime.router_hot_cache_default_load_spread_total;
            router_hot_cache_actual_load_spread_total +=
                runtime.router_hot_cache_actual_load_spread_total;
            router_hot_cache_load_spread_improvement_total +=
                runtime.router_hot_cache_load_spread_improvement_total;
            router_hot_cache_active_dispatches +=
                runtime.router_hot_cache_active_dispatches;
            router_hot_cache_miss_dispatches +=
                runtime.router_hot_cache_miss_dispatches;
            router_hot_cache_selected_expert_slots +=
                runtime.router_hot_cache_selected_expert_slots;
            router_hot_cache_replicated_selected_expert_slots +=
                runtime.router_hot_cache_replicated_selected_expert_slots;

            const DeviceMoEPlacementBankView &active =
                runtime.banks[runtime.active_bank];
            if (runtime_has_local_hot_replica(
                    &runtime, active, config.num_experts))
            {
                ++hot_cache_active_layers;
            }

            if ((config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
            {
                runtime.router_hot_cache_eligible_dispatches = 0ULL;
                runtime.router_hot_cache_used_dispatches = 0ULL;
                runtime.router_hot_cache_improved_dispatches = 0ULL;
                runtime.router_hot_cache_default_load_spread_total = 0ULL;
                runtime.router_hot_cache_actual_load_spread_total = 0ULL;
                runtime.router_hot_cache_load_spread_improvement_total = 0ULL;
                runtime.router_hot_cache_active_dispatches = 0ULL;
                runtime.router_hot_cache_miss_dispatches = 0ULL;
                runtime.router_hot_cache_selected_expert_slots = 0ULL;
                runtime.router_hot_cache_replicated_selected_expert_slots = 0ULL;
            }
        }

        const uint32_t wave_lane =
            lane % kDeviceMoEControllerWaveSize;
        const uint32_t wave_index =
            lane / kDeviceMoEControllerWaveSize;
        const uint64_t wave_eligible =
            rebalance_wave_sum(router_hot_cache_eligible_dispatches);
        const uint64_t wave_used =
            rebalance_wave_sum(router_hot_cache_used_dispatches);
        const uint64_t wave_improved =
            rebalance_wave_sum(router_hot_cache_improved_dispatches);
        const uint64_t wave_default_spread =
            rebalance_wave_sum(router_hot_cache_default_load_spread_total);
        const uint64_t wave_actual_spread =
            rebalance_wave_sum(router_hot_cache_actual_load_spread_total);
        const uint64_t wave_spread_improvement =
            rebalance_wave_sum(
                router_hot_cache_load_spread_improvement_total);
        const uint64_t wave_active =
            rebalance_wave_sum(router_hot_cache_active_dispatches);
        const uint64_t wave_miss =
            rebalance_wave_sum(router_hot_cache_miss_dispatches);
        const uint64_t wave_selected =
            rebalance_wave_sum(router_hot_cache_selected_expert_slots);
        const uint64_t wave_replicated =
            rebalance_wave_sum(
                router_hot_cache_replicated_selected_expert_slots);
        const uint32_t wave_hot_layers =
            rebalance_wave_sum(hot_cache_active_layers);
        if (wave_lane == 0u)
        {
            RebalanceRouterTelemetryPartial &partial =
                shared_router_telemetry[wave_index];
            partial.eligible_dispatches = wave_eligible;
            partial.used_dispatches = wave_used;
            partial.improved_dispatches = wave_improved;
            partial.default_load_spread_total = wave_default_spread;
            partial.actual_load_spread_total = wave_actual_spread;
            partial.load_spread_improvement_total =
                wave_spread_improvement;
            partial.active_dispatches = wave_active;
            partial.miss_dispatches = wave_miss;
            partial.selected_expert_slots = wave_selected;
            partial.replicated_selected_expert_slots = wave_replicated;
            partial.hot_cache_active_layers = wave_hot_layers;
        }
        __syncthreads();

        if (leader)
        {
            router_hot_cache_eligible_dispatches = 0ULL;
            router_hot_cache_used_dispatches = 0ULL;
            router_hot_cache_improved_dispatches = 0ULL;
            router_hot_cache_default_load_spread_total = 0ULL;
            router_hot_cache_actual_load_spread_total = 0ULL;
            router_hot_cache_load_spread_improvement_total = 0ULL;
            router_hot_cache_active_dispatches = 0ULL;
            router_hot_cache_miss_dispatches = 0ULL;
            router_hot_cache_selected_expert_slots = 0ULL;
            router_hot_cache_replicated_selected_expert_slots = 0ULL;
            hot_cache_active_layers = 0u;
            for (uint32_t wave = 0;
                 wave < blockDim.x / kDeviceMoEControllerWaveSize;
                 ++wave)
            {
                const RebalanceRouterTelemetryPartial &partial =
                    shared_router_telemetry[wave];
                router_hot_cache_eligible_dispatches +=
                    partial.eligible_dispatches;
                router_hot_cache_used_dispatches +=
                    partial.used_dispatches;
                router_hot_cache_improved_dispatches +=
                    partial.improved_dispatches;
                router_hot_cache_default_load_spread_total +=
                    partial.default_load_spread_total;
                router_hot_cache_actual_load_spread_total +=
                    partial.actual_load_spread_total;
                router_hot_cache_load_spread_improvement_total +=
                    partial.load_spread_improvement_total;
                router_hot_cache_active_dispatches +=
                    partial.active_dispatches;
                router_hot_cache_miss_dispatches +=
                    partial.miss_dispatches;
                router_hot_cache_selected_expert_slots +=
                    partial.selected_expert_slots;
                router_hot_cache_replicated_selected_expert_slots +=
                    partial.replicated_selected_expert_slots;
                hot_cache_active_layers +=
                    partial.hot_cache_active_layers;
            }
        }
        }
        __syncthreads();
        if (shared_abort != 0u)
            return;

        /*
         * In deferred transfer-slot mode the root participant produces the
         * domain-wide assignment plan and project_rebalance_domain_commands()
         * later broadcasts that root plan into every participant's local apply
         * ABI. Non-root participants must still advance the rolling wave cursor
         * so their next histogram pack stays aligned with the root, but running
         * the full candidate-selection loop here is wasted work.
         */
        if (domain_root_planning && config.participant_id != config.root_participant)
        {
            if (leader)
            {
                if (status)
                {
                    status->status_code = kDeviceMoERebalanceStatusOk;
                    status->windows_observed = 1u;
                    status->windows_applied = 0u;
                    status->changed_layers = 0u;
                    status->last_epoch = 0u;
                    status->router_hot_cache_eligible_dispatches =
                        router_hot_cache_eligible_dispatches;
                    status->router_hot_cache_used_dispatches =
                        router_hot_cache_used_dispatches;
                    status->router_hot_cache_improved_dispatches =
                        router_hot_cache_improved_dispatches;
                    status->router_hot_cache_default_load_spread_total =
                        router_hot_cache_default_load_spread_total;
                    status->router_hot_cache_actual_load_spread_total =
                        router_hot_cache_actual_load_spread_total;
                    status->router_hot_cache_load_spread_improvement_total =
                        router_hot_cache_load_spread_improvement_total;
                    status->router_hot_cache_active_dispatches =
                        router_hot_cache_active_dispatches;
                    status->router_hot_cache_miss_dispatches =
                        router_hot_cache_miss_dispatches;
                    status->router_hot_cache_selected_expert_slots =
                        router_hot_cache_selected_expert_slots;
                    status->router_hot_cache_replicated_selected_expert_slots =
                        router_hot_cache_replicated_selected_expert_slots;
                }
                if (command_header)
                {
                    command_header->epoch = 0u;
                    command_header->command_count = 0u;
                }
                if (wave_state)
                {
                    const uint32_t start_layer =
                        (layer_window_start + start_offset) % config.num_layers;
                    wave_state->epoch = 0u;
                    wave_state->planned_start_layer = start_layer;
                    wave_state->planned_layer_count = layer_wave_count;
                    const uint32_t next_offset =
                        layer_window_count > 0u
                            ? ((start_offset + layer_wave_count) % layer_window_count)
                            : 0u;
                    wave_state->next_start_layer =
                        (layer_window_start + next_offset) % config.num_layers;
                    wave_state->requested_payload_slots = 0u;
                    wave_state->payload_bucket_slots = 0u;
                    wave_state->payload_bucket_index = 0u;
                    wave_state->payload_bucket_overflow = 0u;
                }
            }
            return;
        }

        for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)
        {
            const uint32_t layer =
                (layer_window_start + ((start_offset + window_index) % layer_window_count)) %
                config.num_layers;
            DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (leader)
            {
                shared_layer_invalid =
                    (runtime.active_bank > 1u ||
                     runtime.expert_count != config.num_experts ||
                     runtime.top_k != config.top_k ||
                     runtime.participant_id != config.participant_id ||
                     runtime.participant_count != config.participant_count)
                        ? 1u
                        : 0u;
                if (shared_layer_invalid != 0u)
                    ++invalid_layers;
            }
            __syncthreads();
            if (shared_layer_invalid != 0u)
                continue;

            uint32_t layer_command_count_before = 0u;
            if (leader && defer_runtime_apply && plan_count)
                layer_command_count_before =
                    *plan_count < plan_capacity ? *plan_count : plan_capacity;

            const uint32_t inactive_bank = 1u - runtime.active_bank;
            const DeviceMoEPlacementBankView &active = runtime.banks[runtime.active_bank];
            /* Deferred planning owns scratch, never an unreserved RCU bank. */
            DeviceMoEPlacementBankView &next =
                defer_runtime_apply
                    ? *placement_plan_scratch
                    : runtime.banks[inactive_bank];
            if (leader)
            {
                next.epoch = runtime.active_epoch + 1u;
                next.expert_count = config.num_experts;
                next.multi_resident_expert_count =
                    active.multi_resident_expert_count;
                next.transient_placement_observed =
                    active.transient_placement_observed;
            }

            for (uint32_t expert = lane; expert < config.num_experts; expert += blockDim.x)
            {
                next.experts[expert] = active.experts[expert];
                /* Intra-domain replica policy cannot rewrite the overlay-wide
                 * sparse-packet authority. Preserve it with the bank clone. */
                next.overlay_route_participant[expert] =
                    active.overlay_route_participant[expert];
                DeviceMoEExpertDescriptorView &desc = next.experts[expert];
                uint32_t resident_mask =
                    active.resident_participant_mask[expert] & valid_mask;
                const bool owner_local =
                    desc.owner_participant == static_cast<int32_t>(config.participant_id);
                const bool local_resident = (resident_mask & participant_bit) != 0u;
                const bool multi_resident =
                    runtime_resident_count(resident_mask, config.participant_count) > 1;
                if (multi_resident)
                    desc.flags |= kDeviceMoEFlagReplicated;
                else
                    desc.flags &= ~kDeviceMoEFlagReplicated;
                if (local_resident)
                {
                    desc.flags |= kDeviceMoEFlagValid |
                                  kDeviceMoEFlagResident |
                                  kDeviceMoEFlagLocalCompute;
                }
                else
                {
                    if (owner_local)
                    {
                        FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                            "authoritative ROCm expert owner lost its local payload");
                    }
                    rebalance_retire_local_payload_publication(
                        desc,
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
                /*
                 * Transfer commands are published after the controller has
                 * optimistically updated next.resident_participant_mask for
                 * this wave.  Keep a separate wave-start physical mask for
                 * choosing copy sources, otherwise a later command can try to
                 * copy from a participant that only becomes resident after an
                 * earlier command in the same wave.
                 */
                shared_physical_source_resident_mask[expert] = resident_mask;
                shared_post_policy_resident_mask[expert] =
                    next.resident_participant_mask[expert] & valid_mask;
                const uint64_t count =
                    rebalance_global_count(
                        gathered_histograms,
                        config,
                        window_index,
                        expert);
                shared_expert_counts[expert] = count;
                shared_expert_transfer_backed_participant_mask[expert] =
                    rebalance_collected_transfer_backed_participant_mask(
                        gathered_histograms,
                        config,
                        window_index,
                        expert);
                shared_expert_owners[expert] =
                    desc.owner_participant >= 0 &&
                            desc.owner_participant <
                                static_cast<int32_t>(config.participant_count)
                        ? desc.owner_participant
                        : -1;
            }
            __syncthreads();

            if (lane < config.participant_count)
            {
                uint64_t participant_policy_load = 0ULL;
                uint64_t participant_owner_load = 0ULL;
                for (uint32_t expert = 0;
                     expert < config.num_experts;
                     ++expert)
                {
                    const uint64_t count = shared_expert_counts[expert];
                    const uint32_t resident_mask =
                        shared_post_policy_resident_mask[expert] & valid_mask;
                    participant_policy_load +=
                        llaminar2::moe_rebalance_policy::
                            projectedParticipantLoadForExpert(
                                count,
                                resident_mask,
                                config.participant_count,
                                lane);
                    if (shared_expert_owners[expert] ==
                        static_cast<int32_t>(lane))
                    {
                        participant_owner_load += count;
                    }
                }
                shared_current_policy_load[lane] = participant_policy_load;
                shared_owner_policy_load[lane] = participant_owner_load;
            }
            __syncthreads();
            if (leader)
            {
                uint64_t layer_pre_total = 0ULL;
                uint64_t layer_pre_min = 0ULL;
                uint64_t layer_pre_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_current_policy_load,
                    config.participant_count,
                    layer_pre_total,
                    layer_pre_min,
                    layer_pre_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_pre_policy_load[participant] += shared_current_policy_load[participant];
                pre_wave_load_total += layer_pre_total;
                pre_wave_load_spread += layer_pre_max - layer_pre_min;
            }
            __syncthreads();

            const bool plan_llep =
                least_loaded_assignment &&
                domain_root_planning &&
                plan_missing_arrivals &&
                payload_slot_capacity > 0u;

            if (leader && plan_llep)
            {
                const DeviceMoELLEPLayerPlanScratchView *layer_plan =
                    llep_layer_plans ? llep_layer_plans + window_index : nullptr;
                const bool preplan_valid =
                    layer_plan &&
                    layer_plan->magic == kDeviceMoERebalanceMagic &&
                    layer_plan->version == kDeviceMoERebalanceVersion &&
                    layer_plan->ready == 1u &&
                    layer_plan->window_index == window_index &&
                    layer_plan->layer == layer;
                if (!preplan_valid)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    shared_abort = 1u;
                    llep_status = {};
                }
                else
                {
                    llep_status = layer_plan->planner_status;
                    for (uint32_t participant = 0;
                         participant < config.participant_count;
                         ++participant)
                    {
                        shared_candidate_policy_load[participant] =
                            layer_plan->assigned_participant_load[participant];
                    }
                }
                const bool planned_llep =
                    preplan_valid && layer_plan->planned != 0u;
                if (!hot_replica_cache)
                {
                    for (uint32_t participant = 0;
                         participant < config.participant_count;
                         ++participant)
                    {
                        shared_candidate_policy_load[participant] =
                            shared_current_policy_load[participant];
                    }
                }
                if (!planned_llep || llep_status.overflow != 0u)
                {
                    if (status)
                        ++status->plan_overflow;
                }
                candidate_arrivals_considered += llep_status.weight_transfer_count;
                llep_assignment_span_count += llep_status.span_count;
                llep_weight_transfer_count += llep_status.weight_transfer_count;
                llep_standard_ep_selected += llep_status.standard_ep_selected;
                llep_skipped_balanced += llep_status.skipped_balanced;
                llep_skipped_insufficient_spread_improvement +=
                    llep_status.skipped_insufficient_spread_improvement;
                llep_skipped_insufficient_foreign_rows +=
                    llep_status.skipped_insufficient_foreign_rows;
                llep_min_chunk_skips += llep_status.min_chunk_skips;
                llep_forced_spills += llep_status.forced_spills;
                llep_native_rows += llep_status.native_rows;
                llep_spilled_rows += llep_status.spilled_rows;
                llep_required_spread_improvement +=
                    llep_status.required_spread_improvement;
                llep_required_foreign_rows += llep_status.required_foreign_rows;
                candidate_load_spread_improvement_total +=
                    llep_status.assigned_load_spread_improvement;
                if (llep_status.assigned_load_spread_improvement >
                    candidate_load_spread_improvement_max)
                {
                    candidate_load_spread_improvement_max =
                        llep_status.assigned_load_spread_improvement;
                }

                uint32_t accepted_transfers = 0u;
                if (planned_llep &&
                    llep_status.overflow == 0u &&
                    llep_status.weight_transfer_count > 0u)
                {
                    for (uint32_t transfer_index = 0;
                         transfer_index < llep_status.weight_transfer_count;
                         ++transfer_index)
                    {
                        const auto transfer =
                            layer_plan->transfers[transfer_index];
                        if (transfer.expert >= config.num_experts ||
                            transfer.source_participant >= config.participant_count ||
                            transfer.destination_participant >= config.participant_count ||
                            transfer.source_participant == transfer.destination_participant)
                        {
                            ++skipped_no_resident;
                            continue;
                        }

                        const uint32_t destination_bit =
                            runtime_participant_bit(static_cast<int>(transfer.destination_participant));
                        const uint32_t source_bit =
                            runtime_participant_bit(static_cast<int>(transfer.source_participant));
                        uint32_t resident_mask =
                            shared_post_policy_resident_mask[transfer.expert] & valid_mask;
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[transfer.expert] & valid_mask;
                        if ((physical_source_resident_mask & source_bit) == 0u)
                        {
                            /*
                             * The LLEP transfer planner can only name a useful
                             * source if that participant already owns resident
                             * bytes for this expert before this wave starts.
                             * Keep the command stream honest by rejecting
                             * owner-only or same-wave sources instead of
                             * manufacturing residency in the transfer metadata.
                             */
                            ++skipped_no_resident;
                            continue;
                        }
                        if ((resident_mask & destination_bit) != 0u)
                            continue;

                        const bool ownership_transfer = !hot_replica_cache;
                        if (ownership_transfer &&
                            (shared_expert_owners[transfer.expert] !=
                                 static_cast<int32_t>(
                                     transfer.source_participant) ||
                             !llaminar2::moe_rebalance_policy::
                                 ownershipTransferFitsPersistentCapacity(
                                     shared_active_transfer_slot_counts,
                                     shared_expert_transfer_backed_participant_mask,
                                     transfer.expert,
                                     transfer.source_participant,
                                     transfer.destination_participant,
                                     config.participant_count,
                                     config.active_transfer_slot_capacity)))
                        {
                            if (status)
                                ++status->capacity_limited_candidates;
                            continue;
                        }

                        if (transfer.destination_participant >=
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                            transfer.source_participant >=
                                static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                            shared_destination_transfer_slot_counts[
                                transfer.destination_participant] >= payload_slot_capacity ||
                            shared_source_payload_slot_counts[
                                transfer.source_participant] >= payload_slot_capacity)
                        {
                            if (status)
                                ++status->capacity_limited_candidates;
                            break;
                        }

                        DeviceMoERebalancePlanEntryView entry{};
                        entry.op = ownership_transfer
                                       ? kDeviceMoERebalancePlanOwnershipTransfer
                                       : kDeviceMoERebalancePlanExpertPayloadArrival;
                        entry.layer = layer;
                        entry.expert = transfer.expert;
                        entry.source_participant = transfer.source_participant;
                        entry.destination_participant = transfer.destination_participant;
                        entry.destination_overlay_participant =
                            ownership_transfer
                                ? rebalance_overlay_route_for_domain_participant(
                                      active,
                                      transfer.destination_participant,
                                      config.num_experts)
                                : -1;
                        if (ownership_transfer &&
                            runtime.overlay_epoch_ticket &&
                            entry.destination_overlay_participant < 0)
                        {
                            status->status_code =
                                kDeviceMoERebalanceStatusInvalidRuntime;
                            shared_abort = 1u;
                            break;
                        }
                        entry.source_resident_mask =
                            ownership_transfer ? physical_source_resident_mask : resident_mask;
                        entry.destination_slot =
                            shared_destination_transfer_slot_counts[
                                transfer.destination_participant];
                        entry.payload_slot =
                            shared_source_payload_slot_counts[
                                transfer.source_participant];
                        if (!rebalance_append_plan_entry(
                                plan_entries,
                                plan_count,
                                plan_capacity,
                                entry,
                                status))
                        {
                            break;
                        }

                        ++shared_destination_transfer_slot_counts[
                            transfer.destination_participant];
                        ++shared_source_payload_slot_counts[
                            transfer.source_participant];
                        const uint32_t next_resident_mask =
                            ownership_transfer
                                ? (destination_bit & valid_mask)
                                : ((resident_mask | destination_bit) & valid_mask);
                        if (ownership_transfer)
                        {
                            const uint64_t count =
                                shared_expert_counts[transfer.expert];
                            for (uint32_t participant = 0;
                                 participant < config.participant_count;
                                 ++participant)
                            {
                                const uint64_t old_contribution =
                                    llaminar2::moe_rebalance_policy::
                                        projectedParticipantLoadForExpert(
                                            count,
                                            resident_mask,
                                            config.participant_count,
                                            participant);
                                const uint64_t new_contribution =
                                    llaminar2::moe_rebalance_policy::
                                        projectedParticipantLoadForExpert(
                                            count,
                                            next_resident_mask,
                                            config.participant_count,
                                            participant);
                                shared_candidate_policy_load[participant] =
                                    shared_candidate_policy_load[participant] -
                                    old_contribution + new_contribution;
                            }
                        }
                        resident_mask = next_resident_mask;
                        shared_post_policy_resident_mask[transfer.expert] =
                            resident_mask;
                        next.resident_participant_mask[transfer.expert] =
                            resident_mask;
                        if (ownership_transfer)
                        {
                            next.experts[transfer.expert].owner_participant =
                                static_cast<int32_t>(transfer.destination_participant);
                            next.overlay_route_participant[transfer.expert] =
                                entry.destination_overlay_participant;
                            shared_expert_owners[transfer.expert] =
                                static_cast<int32_t>(transfer.destination_participant);
                            if (!llaminar2::moe_rebalance_policy::
                                    applyOwnershipTransferOccupancy(
                                        shared_active_transfer_slot_counts,
                                        shared_expert_transfer_backed_participant_mask,
                                        transfer.expert,
                                        transfer.source_participant,
                                        transfer.destination_participant))
                            {
                                status->status_code =
                                    kDeviceMoERebalanceStatusInvalidRuntime;
                                shared_abort = 1u;
                                break;
                            }
                        }
                        ++accepted_transfers;
                    }
                }

                if (accepted_transfers > 0u)
                {
                    uint64_t accepted_before_total = 0ULL;
                    uint64_t accepted_before_min = 0ULL;
                    uint64_t accepted_before_max = 0ULL;
                    llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                        shared_current_policy_load,
                        config.participant_count,
                        accepted_before_total,
                        accepted_before_min,
                        accepted_before_max);
                    (void)accepted_before_total;

                    const bool exact_llep_assignment_is_materialized =
                        hot_replica_cache &&
                        accepted_transfers == llep_status.weight_transfer_count;
                    if (hot_replica_cache &&
                        !exact_llep_assignment_is_materialized)
                    {
                        for (uint32_t participant = 0;
                             participant < config.participant_count;
                             ++participant)
                        {
                            shared_candidate_policy_load[participant] = 0ULL;
                        }
                        for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                        {
                            const uint32_t resident_mask =
                                shared_post_policy_resident_mask[expert] & valid_mask;
                            for (uint32_t participant = 0;
                                 participant < config.participant_count;
                                 ++participant)
                            {
                                shared_candidate_policy_load[participant] +=
                                    llaminar2::moe_rebalance_policy::projectedParticipantLoadForExpert(
                                        shared_expert_counts[expert],
                                        resident_mask,
                                        config.participant_count,
                                        participant);
                            }
                        }
                    }
                    /*
                     * The LLEP planner already wrote its exact row-span loads
                     * into shared_candidate_policy_load.  When every required
                     * replica arrival is present, those loads describe the
                     * production apportioned-row schedule exactly.  Replacing
                     * them with an equal resident split can invert the economy
                     * decision for uneven groups and reject a genuinely useful
                     * transfer.  Whole-expert ownership moves and partial waves
                     * cannot execute the ideal row schedule, so they retain the
                     * physical-residency projection above.
                     */

                    uint64_t accepted_after_total = 0ULL;
                    uint64_t accepted_after_min = 0ULL;
                    uint64_t accepted_after_max = 0ULL;
                    llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                        shared_candidate_policy_load,
                        config.participant_count,
                        accepted_after_total,
                        accepted_after_min,
                        accepted_after_max);
                    (void)accepted_after_total;
                    const uint64_t accepted_before_spread =
                        accepted_before_max >= accepted_before_min
                            ? accepted_before_max - accepted_before_min
                            : 0ULL;
                    const uint64_t accepted_after_spread =
                        accepted_after_max >= accepted_after_min
                            ? accepted_after_max - accepted_after_min
                            : 0ULL;
                    const uint64_t accepted_improvement =
                        accepted_before_spread > accepted_after_spread
                            ? accepted_before_spread - accepted_after_spread
                            : 0ULL;

                    for (uint32_t participant = 0;
                         participant < config.participant_count;
                         ++participant)
                    {
                        shared_current_policy_load[participant] =
                            shared_candidate_policy_load[participant];
                    }
                    accepted_load_spread_improvement_total += accepted_improvement;
                    if (accepted_improvement > accepted_load_spread_improvement_max)
                    {
                        accepted_load_spread_improvement_max = accepted_improvement;
                    }
                    if (hot_replica_cache)
                        selected_replicas += accepted_transfers;
                }
                else if (llep_status.standard_ep_selected != 0u ||
                         llep_status.weight_transfer_count == 0u)
                {
                    ++skipped_no_improvement;
                }
            }
            __syncthreads();
            if (shared_abort != 0u)
                return;

            if (leader && !least_loaded_assignment && plan_missing_arrivals && payload_slot_capacity > 0u)
            {
                const uint32_t max_swaps =
                    config.dynamic_max_swaps_per_layer == 0u
                        ? 0u
                        : config.dynamic_max_swaps_per_layer;
                for (uint32_t swap_iter = 0; swap_iter < max_swaps; ++swap_iter)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    const auto admission =
                        llaminar2::moe_rebalance_policy::dynamicPlanAdmission(
                            command_count,
                            /*required_entries=*/2u,
                            config.dynamic_max_plan_entries_per_wave,
                            plan_capacity);
                    if (admission == llaminar2::moe_rebalance_policy::
                                         DynamicPlanAdmission::ConfiguredWaveLimit)
                    {
                        if (status)
                            ++status->capacity_limited_candidates;
                        break;
                    }
                    if (admission == llaminar2::moe_rebalance_policy::
                                         DynamicPlanAdmission::PhysicalPlanOverflow)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }

                    ++dynamic_ownership_swap_attempts;
                    const auto swap_choice =
                        llaminar2::moe_rebalance_policy::bestDynamicOwnershipSwap(
                            shared_owner_policy_load,
                            shared_expert_counts,
                            shared_expert_owners,
                            config.num_experts,
                            config.participant_count,
                            llaminar2::moe_rebalance_policy::
                                dynamicOwnershipEvidenceFromParticipantLoads(
                                    shared_owner_policy_load,
                                    config.participant_count,
                                    config.dynamic_min_window_activations),
                            config.dynamic_imbalance_threshold_per_mille,
                            config.dynamic_min_improvement_per_mille,
                            shared_expert_transfer_backed_participant_mask,
                            shared_active_transfer_slot_counts,
                            config.active_transfer_slot_capacity);
                    if (!swap_choice.valid)
                    {
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }
                    const uint32_t heavy_source = swap_choice.overloaded_participant;
                    const uint32_t heavy_destination = swap_choice.underloaded_participant;
                    const uint32_t light_source = swap_choice.underloaded_participant;
                    const uint32_t light_destination = swap_choice.overloaded_participant;
                    const uint32_t heavy_physical_source_mask =
                        shared_physical_source_resident_mask[swap_choice.heavy_expert] & valid_mask;
                    const uint32_t light_physical_source_mask =
                        shared_physical_source_resident_mask[swap_choice.light_expert] & valid_mask;
                    const int32_t heavy_destination_overlay_participant =
                        rebalance_overlay_route_for_domain_participant(
                            next,
                            heavy_destination,
                            config.num_experts);
                    const int32_t light_destination_overlay_participant =
                        rebalance_overlay_route_for_domain_participant(
                            next,
                            light_destination,
                            config.num_experts);
                    if (runtime.overlay_epoch_ticket &&
                        (heavy_destination_overlay_participant < 0 ||
                         light_destination_overlay_participant < 0))
                    {
                        status->status_code =
                            kDeviceMoERebalanceStatusInvalidRuntime;
                        shared_abort = 1u;
                        break;
                    }
                    const uint32_t heavy_source_bit =
                        runtime_participant_bit(static_cast<int>(heavy_source));
                    const uint32_t light_source_bit =
                        runtime_participant_bit(static_cast<int>(light_source));
                    if (heavy_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        heavy_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        shared_source_payload_slot_counts[heavy_source] >= payload_slot_capacity ||
                        shared_source_payload_slot_counts[light_source] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[heavy_destination] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[light_destination] >= payload_slot_capacity)
                    {
                        if (status)
                            ++status->capacity_limited_candidates;
                        break;
                    }
                    if ((heavy_physical_source_mask & heavy_source_bit) == 0u ||
                        (light_physical_source_mask & light_source_bit) == 0u)
                    {
                        /*
                         * Ownership swaps still move full expert payloads through the
                         * compact transfer path.  A logical old owner or same-wave
                         * destination is not enough; each source participant must
                         * physically hold the payload before we publish the paired swap.
                         */
                        ++skipped_no_resident;
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }
                    ++dynamic_ownership_swap_accepts;

                    DeviceMoERebalancePlanEntryView heavy_entry{};
                    heavy_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    heavy_entry.layer = layer;
                    heavy_entry.expert = swap_choice.heavy_expert;
                    heavy_entry.activation_count = shared_expert_counts[swap_choice.heavy_expert];
                    heavy_entry.source_participant = heavy_source;
                    heavy_entry.destination_participant = heavy_destination;
                    heavy_entry.destination_overlay_participant =
                        heavy_destination_overlay_participant;
                    heavy_entry.source_resident_mask = heavy_physical_source_mask;
                    heavy_entry.destination_slot =
                        shared_destination_transfer_slot_counts[heavy_destination];
                    heavy_entry.payload_slot =
                        shared_source_payload_slot_counts[heavy_source];

                    DeviceMoERebalancePlanEntryView light_entry{};
                    light_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    light_entry.layer = layer;
                    light_entry.expert = swap_choice.light_expert;
                    light_entry.activation_count = shared_expert_counts[swap_choice.light_expert];
                    light_entry.source_participant = light_source;
                    light_entry.destination_participant = light_destination;
                    light_entry.destination_overlay_participant =
                        light_destination_overlay_participant;
                    light_entry.source_resident_mask = light_physical_source_mask;
                    light_entry.destination_slot =
                        shared_destination_transfer_slot_counts[light_destination];
                    light_entry.payload_slot =
                        shared_source_payload_slot_counts[light_source];

                    if (!rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            heavy_entry,
                            status) ||
                        !rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            light_entry,
                            status))
                    {
                        break;
                    }

                    ++shared_source_payload_slot_counts[heavy_source];
                    ++shared_source_payload_slot_counts[light_source];
                    ++shared_destination_transfer_slot_counts[heavy_destination];
                    ++shared_destination_transfer_slot_counts[light_destination];

                    next.experts[swap_choice.heavy_expert].owner_participant =
                        static_cast<int32_t>(heavy_destination);
                    next.experts[swap_choice.light_expert].owner_participant =
                        static_cast<int32_t>(light_destination);
                    next.overlay_route_participant[swap_choice.heavy_expert] =
                        heavy_destination_overlay_participant;
                    next.overlay_route_participant[swap_choice.light_expert] =
                        light_destination_overlay_participant;
                    shared_post_policy_resident_mask[swap_choice.heavy_expert] =
                        runtime_participant_bit(static_cast<int>(heavy_destination));
                    shared_post_policy_resident_mask[swap_choice.light_expert] =
                        runtime_participant_bit(static_cast<int>(light_destination));
                    next.resident_participant_mask[swap_choice.heavy_expert] =
                        shared_post_policy_resident_mask[swap_choice.heavy_expert];
                    next.resident_participant_mask[swap_choice.light_expert] =
                        shared_post_policy_resident_mask[swap_choice.light_expert];

                    llaminar2::moe_rebalance_policy::applyDynamicOwnershipSwap(
                        shared_owner_policy_load,
                        shared_expert_owners,
                        swap_choice);
                    llaminar2::moe_rebalance_policy::
                        applyDynamicOwnershipSwapTransferOccupancy(
                            shared_active_transfer_slot_counts,
                            shared_expert_transfer_backed_participant_mask,
                            swap_choice);
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                        shared_current_policy_load[participant] = shared_owner_policy_load[participant];

                    accepted_load_spread_improvement_total += swap_choice.improvement;
                    if (swap_choice.improvement > accepted_load_spread_improvement_max)
                        accepted_load_spread_improvement_max = swap_choice.improvement;
                }
            }
            __syncthreads();

            const uint32_t max_local_replicas =
                (config.flags & kDeviceMoERebalanceFlagHotReplicaCache) != 0u
                    ? config.max_hot_replicas_per_participant
                    : 0u;
            const uint32_t max_selected_replicas =
                domain_root_planning
                    ? max_local_replicas * config.participant_count
                    : max_local_replicas;
            if (leader)
            {
                shared_selected_count = 0u;
                shared_local_replicas = 0u;
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_destination_replica_counts[participant] = 0u;
            }
            __syncthreads();

            for (uint32_t rank = 0; !least_loaded_assignment && rank < config.num_experts; ++rank)
            {
                if (shared_local_replicas >= max_selected_replicas)
                    break;
                if (leader)
                {
                    shared_stop_rank = 0u;
                    shared_count_bound_prunes = 0u;
                    uint64_t current_policy_total = 0ULL;
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                        current_policy_total += shared_current_policy_load[participant];
                    shared_required_load_spread_improvement =
                        llaminar2::moe_rebalance_policy::requiredLoadSpreadImprovement(
                            current_policy_total,
                            config.min_load_spread_improvement,
                            config.min_load_spread_improvement_divisor);
                    if (plan_count &&
                        llaminar2::moe_rebalance_policy::commandBufferFull(
                            *plan_count,
                            plan_capacity))
                    {
                        if (status)
                            ++status->plan_overflow;
                        shared_stop_rank = 1u;
                    }
                }
                __syncthreads();
                if (shared_stop_rank != 0u)
                    break;

                // Retain the proven 256-thread controller and its register
                // budget. Each thread evaluates disjoint expert IDs in order;
                // shared storage then exposes all 512 candidates to one fixed
                // tree. No candidate state scales with the expert count.
                for (uint32_t candidate_lane = lane;
                     candidate_lane < static_cast<uint32_t>(kDeviceMoEMaxExperts);
                     candidate_lane += blockDim.x)
                {
                    uint32_t candidate_expert = static_cast<uint32_t>(kDeviceMoEMaxExperts);
                    unsigned long long candidate_value = 0ULL;
                    unsigned long long candidate_count = 0ULL;
                    unsigned long long candidate_improvement = 0ULL;
                    uint32_t candidate_arrival_considered = 0u;
                    uint32_t candidate_arrival_below_floor = 0u;
                    if (candidate_lane < config.num_experts)
                    {
                        bool already_selected = false;
                        for (uint32_t i = 0; i < shared_selected_count; ++i)
                        {
                            if (shared_selected[i] == candidate_lane)
                            {
                                already_selected = true;
                                break;
                            }
                        }
                        if (!already_selected)
                        {
                            const DeviceMoEExpertDescriptorView &candidate_desc =
                                next.experts[candidate_lane];
                            uint32_t candidate_resident_mask =
                                next.resident_participant_mask[candidate_lane] & valid_mask;
                            const uint32_t candidate_physical_source_mask =
                                shared_physical_source_resident_mask[candidate_lane] & valid_mask;
                            const bool candidate_can_affect =
                                domain_root_planning
                                    ? rebalance_candidate_can_affect_domain_compute(
                                          candidate_desc,
                                          candidate_resident_mask,
                                          config,
                                          plan_missing_arrivals)
                                    : rebalance_candidate_can_affect_local_compute(
                                          candidate_desc,
                                          candidate_resident_mask,
                                          config,
                                          participant_bit,
                                          plan_missing_arrivals);
                            if (candidate_can_affect)
                            {
                                const unsigned long long count =
                                    rebalance_global_count(gathered_histograms, config, window_index, candidate_lane);
                                if (count > 0ULL)
                                {
                                    const bool candidate_local_resident =
                                        (candidate_resident_mask & participant_bit) != 0u;
                                    const bool candidate_owner_local =
                                        candidate_desc.owner_participant ==
                                        static_cast<int32_t>(config.participant_id);
                                    if (domain_root_planning)
                                    {
                                        if (shared_required_load_spread_improvement != 0ULL &&
                                            count < shared_required_load_spread_improvement)
                                        {
                                            atomicAdd(&shared_count_bound_prunes, 1u);
                                        }
                                        else
                                        {
                                            const auto destination_choice =
                                                llaminar2::moe_rebalance_policy::bestDynamicMissingResidentDestination(
                                                    shared_current_policy_load,
                                                    count,
                                                    candidate_resident_mask,
                                                    config.participant_count,
                                                    candidate_desc.owner_participant,
                                                    shared_destination_replica_counts,
                                                    max_local_replicas,
                                                    config.window_size_tokens,
                                                    config.min_load_spread_improvement,
                                                    config.min_load_spread_improvement_divisor);
                                            candidate_arrival_considered = 1u;
                                            candidate_improvement = destination_choice.delta.improvement;
                                            const int source_participant =
                                                destination_choice.valid
                                                    ? rebalance_first_resident_participant(
                                                          candidate_physical_source_mask,
                                                          config.participant_count,
                                                          candidate_desc.owner_participant,
                                                          static_cast<int>(destination_choice.destination_participant))
                                                    : -1;
                                            const bool destination_has_transfer_slot =
                                                destination_choice.valid &&
                                                destination_choice.destination_participant < kDeviceMoEMaxParticipants &&
                                                shared_destination_transfer_slot_counts[
                                                    destination_choice.destination_participant] < payload_slot_capacity;
                                            const bool source_has_payload_slot =
                                                source_participant >= 0 &&
                                                static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                                shared_source_payload_slot_counts[
                                                    static_cast<uint32_t>(source_participant)] < payload_slot_capacity;
                                            const bool source_is_physically_resident =
                                                source_participant >= 0 &&
                                                static_cast<uint32_t>(source_participant) < config.participant_count &&
                                                (candidate_physical_source_mask &
                                                 runtime_participant_bit(source_participant)) != 0u;
                                            const auto source_delta =
                                                source_is_physically_resident
                                                    ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
                                                          shared_current_policy_load,
                                                          count,
                                                          candidate_resident_mask,
                                                          destination_choice.proposed_resident_mask,
                                                          config.participant_count,
                                                          static_cast<uint32_t>(source_participant),
                                                          destination_choice.destination_participant,
                                                          config.window_size_tokens,
                                                          config.min_load_spread_improvement,
                                                          config.min_load_spread_improvement_divisor)
                                                    : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                                            candidate_improvement = source_delta.improvement;
                                            if (destination_choice.valid &&
                                                destination_has_transfer_slot &&
                                                source_has_payload_slot &&
                                                source_is_physically_resident &&
                                                source_delta.meets_floor)
                                            {
                                                candidate_value = source_delta.improvement;
                                                candidate_count = count;
                                                candidate_expert = candidate_lane;
                                            }
                                            else if ((candidate_resident_mask & (candidate_resident_mask - 1u)) == 0u)
                                            {
                                                candidate_arrival_below_floor = 1u;
                                            }
                                            else
                                            {
                                                candidate_value = count;
                                                candidate_count = count;
                                                candidate_expert = candidate_lane;
                                            }
                                        }
                                    }
                                    else if (plan_missing_arrivals &&
                                        !candidate_local_resident &&
                                        !candidate_owner_local)
                                    {
                                        if (shared_required_load_spread_improvement != 0ULL &&
                                            count < shared_required_load_spread_improvement)
                                        {
                                            atomicAdd(&shared_count_bound_prunes, 1u);
                                        }
                                        else
                                        {
                                            const int source_participant =
                                                rebalance_first_resident_participant(
                                                    candidate_resident_mask,
                                                    config.participant_count,
                                                    candidate_desc.owner_participant,
                                                    static_cast<int>(config.participant_id));
                                            const uint32_t proposed_resident_mask =
                                                (candidate_resident_mask | participant_bit) & valid_mask;
                                                const auto delta =
                                                    source_participant >= 0
                                                        ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
                                                              shared_current_policy_load,
                                                              count,
                                                              candidate_resident_mask,
                                                              proposed_resident_mask,
                                                              config.participant_count,
                                                              static_cast<uint32_t>(source_participant),
                                                              config.participant_id,
                                                              config.window_size_tokens,
                                                              config.min_load_spread_improvement,
                                                              config.min_load_spread_improvement_divisor)
                                                        : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                                            candidate_arrival_considered = 1u;
                                            candidate_improvement = delta.improvement;
                                            if (!delta.meets_floor)
                                            {
                                                candidate_arrival_below_floor = 1u;
                                            }
                                            else
                                            {
                                                const bool source_has_payload_slot =
                                                    source_participant >= 0 &&
                                                    static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                                    shared_source_payload_slot_counts[
                                                        static_cast<uint32_t>(source_participant)] < payload_slot_capacity;
                                                if (config.participant_id < kDeviceMoEMaxParticipants &&
                                                    shared_destination_transfer_slot_counts[config.participant_id] <
                                                        payload_slot_capacity &&
                                                    source_has_payload_slot)
                                                {
                                                    candidate_value = delta.improvement;
                                                    candidate_count = count;
                                                    candidate_expert = candidate_lane;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        candidate_value = count;
                                        candidate_count = count;
                                        candidate_expert = candidate_lane;
                                    }
                                }
                            }
                        }
                    }

                    shared_candidate_values[candidate_lane] = candidate_value;
                    shared_candidate_counts[candidate_lane] = candidate_count;
                    shared_candidate_improvements[candidate_lane] = candidate_improvement;
                    shared_candidate_experts[candidate_lane] = candidate_expert;
                    shared_candidate_arrival_considered[candidate_lane] = candidate_arrival_considered;
                    shared_candidate_arrival_below_floor[candidate_lane] = candidate_arrival_below_floor;
                }
                __syncthreads();

                if (leader)
                {
                    for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                    {
                        candidate_arrivals_considered +=
                            shared_candidate_arrival_considered[expert];
                        candidate_arrivals_below_floor +=
                            shared_candidate_arrival_below_floor[expert];
                        skipped_no_improvement +=
                            shared_candidate_arrival_below_floor[expert];
                        candidate_load_spread_improvement_total +=
                            shared_candidate_improvements[expert];
                        if (shared_candidate_improvements[expert] >
                            candidate_load_spread_improvement_max)
                        {
                            candidate_load_spread_improvement_max =
                                shared_candidate_improvements[expert];
                        }
                    }
                }
                __syncthreads();

                for (uint32_t stride = kDeviceMoEMaxExperts / 2u; stride > 0u; stride >>= 1u)
                {
                    if (lane < stride)
                    {
                        const uint32_t other = lane + stride;
                        const unsigned long long lhs_value = shared_candidate_values[lane];
                        const unsigned long long rhs_value = shared_candidate_values[other];
                        const unsigned long long lhs_count = shared_candidate_counts[lane];
                        const unsigned long long rhs_count = shared_candidate_counts[other];
                        const uint32_t lhs_expert = shared_candidate_experts[lane];
                        const uint32_t rhs_expert = shared_candidate_experts[other];
                        if (llaminar2::moe_rebalance_policy::candidateValueIsBetter(
                                rhs_value,
                                rhs_count,
                                rhs_expert,
                                lhs_value,
                                lhs_count,
                                lhs_expert))
                        {
                            shared_candidate_values[lane] = rhs_value;
                            shared_candidate_counts[lane] = rhs_count;
                            shared_candidate_experts[lane] = rhs_expert;
                        }
                    }
                    __syncthreads();
                }

                if (leader)
                {
                    const uint32_t best_expert = shared_candidate_experts[0];
                    const unsigned long long best_count = shared_candidate_counts[0];
                    shared_stop_rank =
                        (best_expert >= config.num_experts || best_count == 0ULL) ? 1u : 0u;
                    candidate_arrivals_pruned_by_count_bound += shared_count_bound_prunes;
                    skipped_no_improvement += shared_count_bound_prunes;
                    if (shared_stop_rank == 0u)
                        shared_selected[shared_selected_count++] = best_expert;
                }
                __syncthreads();
                if (shared_stop_rank != 0u)
                    break;

                if (leader)
                {
                    const uint32_t best_expert = shared_selected[shared_selected_count - 1u];
                    DeviceMoEExpertDescriptorView &desc = next.experts[best_expert];
                    uint32_t resident_mask = next.resident_participant_mask[best_expert] & valid_mask;
                    next.resident_participant_mask[best_expert] = resident_mask;

                    const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
                    const bool local_resident = (resident_mask & participant_bit) != 0u;
                    const bool owner_local =
                        desc.owner_participant == static_cast<int32_t>(config.participant_id);
                    if (domain_root_planning)
                    {
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[best_expert] & valid_mask;
                        const unsigned long long count =
                            rebalance_global_count(gathered_histograms, config, window_index, best_expert);
                        const auto destination_choice =
                            llaminar2::moe_rebalance_policy::bestDynamicMissingResidentDestination(
                                shared_current_policy_load,
                                count,
                                resident_mask,
                                config.participant_count,
                                desc.owner_participant,
                                shared_destination_replica_counts,
                                max_local_replicas,
                                config.window_size_tokens,
                                config.min_load_spread_improvement,
                                config.min_load_spread_improvement_divisor);
                        if (destination_choice.valid)
                        {
                            const int source_participant =
                                rebalance_first_resident_participant(
                                    physical_source_resident_mask,
                                    config.participant_count,
                                    desc.owner_participant,
                                    static_cast<int>(destination_choice.destination_participant));
                            const bool source_is_physically_resident =
                                source_participant >= 0 &&
                                static_cast<uint32_t>(source_participant) < config.participant_count &&
                                (physical_source_resident_mask &
                                 runtime_participant_bit(source_participant)) != 0u;
                            if (!source_is_physically_resident)
                            {
                                /*
                                 * A transfer command is only valid when the chosen source
                                 * already owns resident bytes in this runtime bank.  Static
                                 * ownership can guide preference, but it cannot authorize a
                                 * compact payload copy by itself.
                                 */
                                ++skipped_no_resident;
                            }
                            else if (!llaminar2::moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                                    shared_current_policy_load,
                                    count,
                                    resident_mask,
                                    destination_choice.proposed_resident_mask,
                                    config.participant_count,
                                    static_cast<uint32_t>(source_participant),
                                    destination_choice.destination_participant,
                                    shared_candidate_policy_load,
                                    config.window_size_tokens,
                                    config.min_load_spread_improvement,
                                    config.min_load_spread_improvement_divisor))
                            {
                                ++skipped_no_improvement;
                            }
                            else
                            {
                                uint32_t destination_slot = kDeviceMoEInvalidSlot;
                                uint32_t payload_slot = kDeviceMoEInvalidSlot;
                                if (destination_choice.destination_participant < kDeviceMoEMaxParticipants &&
                                    shared_destination_transfer_slot_counts[
                                        destination_choice.destination_participant] < payload_slot_capacity &&
                                    source_participant >= 0 &&
                                    static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                    shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)] < payload_slot_capacity)
                                {
                                    destination_slot =
                                        shared_destination_transfer_slot_counts[
                                            destination_choice.destination_participant];
                                    payload_slot =
                                        shared_source_payload_slot_counts[
                                            static_cast<uint32_t>(source_participant)];
                                }
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    source_participant >= 0 ? static_cast<uint32_t>(source_participant) : 0u;
                                entry.destination_participant =
                                    destination_choice.destination_participant;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = destination_slot;
                                entry.payload_slot = payload_slot;
                                if (destination_slot != kDeviceMoEInvalidSlot &&
                                    payload_slot != kDeviceMoEInvalidSlot &&
                                    source_participant >= 0 &&
                                    rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    ++shared_destination_transfer_slot_counts[
                                        destination_choice.destination_participant];
                                    ++shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)];
                                    shared_post_policy_resident_mask[best_expert] =
                                        destination_choice.proposed_resident_mask;
                                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                                        shared_current_policy_load[participant] =
                                            shared_candidate_policy_load[participant];
                                    accepted_load_spread_improvement_total +=
                                        destination_choice.delta.improvement;
                                    if (destination_choice.delta.improvement >
                                        accepted_load_spread_improvement_max)
                                    {
                                        accepted_load_spread_improvement_max =
                                            destination_choice.delta.improvement;
                                    }
                                    ++shared_destination_replica_counts[
                                        destination_choice.destination_participant];
                                    ++shared_local_replicas;
                                    ++selected_replicas;
                                }
                                else
                                {
                                    if (destination_slot == kDeviceMoEInvalidSlot && status)
                                        ++status->capacity_limited_candidates;
                                    ++skipped_no_resident;
                                }
                            }
                        }
                        else if (!replicated)
                        {
                            ++skipped_no_resident;
                        }
                        else
                        {
                            bool resident_plan_ready = false;
                            for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                            {
                                if ((resident_mask & runtime_participant_bit(static_cast<int>(participant))) == 0u)
                                    continue;
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanResidentExpertAssignment;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    desc.owner_participant >= 0
                                        ? static_cast<uint32_t>(desc.owner_participant)
                                        : participant;
                                entry.destination_participant = participant;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = kDeviceMoEInvalidSlot;
                                entry.payload_slot = kDeviceMoEInvalidSlot;
                                if (rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    resident_plan_ready = true;
                                    if (desc.owner_participant != static_cast<int32_t>(participant))
                                    {
                                        ++shared_destination_replica_counts[participant];
                                        ++shared_local_replicas;
                                    }
                                }
                            }
                            if (resident_plan_ready)
                            {
                                shared_post_policy_resident_mask[best_expert] =
                                    resident_mask & valid_mask;
                                ++selected_replicas;
                            }
                            else
                            {
                                ++skipped_no_resident;
                            }
                        }
                    }
                    else if (plan_missing_arrivals && !local_resident && !owner_local)
                    {
                        const uint32_t physical_source_resident_mask =
                            shared_physical_source_resident_mask[best_expert] & valid_mask;
                        const unsigned long long count =
                            rebalance_global_count(gathered_histograms, config, window_index, best_expert);
                        const int source_participant = rebalance_first_resident_participant(
                            physical_source_resident_mask,
                            config.participant_count,
                            desc.owner_participant,
                            static_cast<int>(config.participant_id));
                        const uint32_t proposed_resident_mask =
                            (resident_mask | participant_bit) & valid_mask;
	                        const auto delta =
	                            source_participant >= 0
	                                ? llaminar2::moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
	                                      shared_current_policy_load,
	                                      count,
	                                      resident_mask,
	                                      proposed_resident_mask,
	                                      config.participant_count,
	                                      static_cast<uint32_t>(source_participant),
	                                      config.participant_id,
	                                      config.window_size_tokens,
	                                      config.min_load_spread_improvement,
	                                      config.min_load_spread_improvement_divisor)
	                                : llaminar2::moe_rebalance_policy::LoadSpreadDelta{};
                        if (!delta.meets_floor)
                        {
                            ++skipped_no_improvement;
                        }
                        else if (!llaminar2::moe_rebalance_policy::addingResidentImprovesDynamicSpread(
                                shared_current_policy_load,
                                count,
                                resident_mask,
                                proposed_resident_mask,
                                config.participant_count,
                                static_cast<uint32_t>(source_participant),
                                config.participant_id,
                                shared_candidate_policy_load,
                                config.window_size_tokens,
                                config.min_load_spread_improvement,
                                config.min_load_spread_improvement_divisor))
                        {
                            ++skipped_no_improvement;
                        }
                        else
                        {
                            uint32_t destination_slot = kDeviceMoEInvalidSlot;
                            uint32_t payload_slot = kDeviceMoEInvalidSlot;
                            if (config.participant_id < kDeviceMoEMaxParticipants &&
                                shared_destination_transfer_slot_counts[config.participant_id] < payload_slot_capacity &&
                                source_participant >= 0 &&
                                static_cast<uint32_t>(source_participant) < kDeviceMoEMaxParticipants &&
                                shared_source_payload_slot_counts[
                                    static_cast<uint32_t>(source_participant)] < payload_slot_capacity)
                            {
                                destination_slot =
                                    shared_destination_transfer_slot_counts[config.participant_id];
                                payload_slot =
                                    shared_source_payload_slot_counts[
                                        static_cast<uint32_t>(source_participant)];
                            }
                            DeviceMoERebalancePlanEntryView entry{};
                            entry.op = kDeviceMoERebalancePlanExpertPayloadArrival;
                            entry.layer = layer;
                            entry.expert = best_expert;
                            entry.source_participant =
                                source_participant >= 0 ? static_cast<uint32_t>(source_participant) : 0u;
                            entry.destination_participant = config.participant_id;
                            entry.source_resident_mask = resident_mask;
                            entry.destination_slot = destination_slot;
                            entry.payload_slot = payload_slot;
                            if (destination_slot != kDeviceMoEInvalidSlot &&
                                payload_slot != kDeviceMoEInvalidSlot &&
                                source_participant >= 0 &&
                                rebalance_append_plan_entry(
                                    plan_entries,
                                    plan_count,
                                    plan_capacity,
                                    entry,
                                    status))
                            {
                                ++shared_destination_transfer_slot_counts[config.participant_id];
                                ++shared_source_payload_slot_counts[
                                    static_cast<uint32_t>(source_participant)];
                                shared_post_policy_resident_mask[best_expert] =
                                    proposed_resident_mask;
                                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                                    shared_current_policy_load[participant] =
                                        shared_candidate_policy_load[participant];
                                accepted_load_spread_improvement_total += delta.improvement;
                                if (delta.improvement > accepted_load_spread_improvement_max)
                                    accepted_load_spread_improvement_max = delta.improvement;
                                ++shared_local_replicas;
                            }
                            else
                            {
                                if (destination_slot == kDeviceMoEInvalidSlot && status)
                                    ++status->capacity_limited_candidates;
                                ++skipped_no_resident;
                            }
                        }
                    }
                    else if (!replicated)
                    {
                        ++skipped_no_resident;
                    }
                    else
                    {
                        desc.flags |= kDeviceMoEFlagReplicated;
                        if (local_resident)
                        {
                            bool resident_plan_ready = true;
                            if (defer_runtime_apply)
                            {
                                DeviceMoERebalancePlanEntryView entry{};
                                entry.op = kDeviceMoERebalancePlanResidentExpertAssignment;
                                entry.layer = layer;
                                entry.expert = best_expert;
                                entry.source_participant =
                                    desc.owner_participant >= 0
                                        ? static_cast<uint32_t>(desc.owner_participant)
                                        : config.participant_id;
                                entry.destination_participant = config.participant_id;
                                entry.source_resident_mask = resident_mask;
                                entry.destination_slot = kDeviceMoEInvalidSlot;
                                entry.payload_slot = kDeviceMoEInvalidSlot;
                                if (!rebalance_append_plan_entry(
                                        plan_entries,
                                        plan_count,
                                        plan_capacity,
                                        entry,
                                        status))
                                {
                                    ++skipped_no_resident;
                                    resident_plan_ready = false;
                                }
                            }
                            if (resident_plan_ready)
                            {
                                next.local_compute_mask[best_expert] = 1u;
                                next.replica_role[best_expert] =
                                    owner_local
                                        ? static_cast<uint8_t>(kDeviceMoEReplicaRolePrimary)
                                        : static_cast<uint8_t>(kDeviceMoEReplicaRoleReplica);
                                if (!owner_local)
                                    ++shared_local_replicas;
                                ++selected_replicas;
                            }
                            shared_post_policy_resident_mask[best_expert] =
                                next.resident_participant_mask[best_expert] & valid_mask;
                        }
                    }
                }
                __syncthreads();
            }

            if (!defer_runtime_apply)
            {
                /*
                 * Planning can move authoritative ownership after the initial
                 * bank copy above. Normalize every departed source before the
                 * inactive bank is published; otherwise its old transfer slot
                 * and matrix pointers survive under a remote owner.
                 */
                for (uint32_t expert = lane;
                     expert < config.num_experts;
                     expert += blockDim.x)
                {
                    auto &desc = next.experts[expert];
                    uint32_t resident_mask =
                        next.resident_participant_mask[expert] & valid_mask;
                    const bool local_resident =
                        (resident_mask & participant_bit) != 0u;
                    const bool owner_local =
                        desc.owner_participant ==
                        static_cast<int32_t>(config.participant_id);
                    if (!local_resident)
                    {
                        if (owner_local)
                        {
                            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                                "ROCm rebalance plan published a nonresident local owner");
                        }
                        rebalance_retire_local_payload_publication(
                            desc,
                            resident_mask,
                            participant_bit);
                        next.local_compute_mask[expert] = 0u;
                        next.replica_role[expert] =
                            static_cast<uint8_t>(
                                kDeviceMoEReplicaRoleNone);
                        next.resident_participant_mask[expert] =
                            resident_mask;
                    }
                }
                __syncthreads();
            }

            if (leader)
            {
                if (defer_runtime_apply)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    if (command_count > layer_command_count_before)
                    {
                        last_epoch = shared_plan_epoch;
                        ++changed_layers;
                    }
                }
                else
                {
                    next.multi_resident_expert_count =
                        runtime_compute_multi_resident_expert_count(
                            next, config.num_experts, config.participant_count);
                    runtime.active_bank = inactive_bank;
                    runtime.active_epoch = next.epoch;
                    last_epoch = runtime.active_epoch;
                    ++changed_layers;
                }
            }

            if (!defer_runtime_apply &&
                (config.flags & kDeviceMoERebalanceFlagResetHistograms) != 0u)
            {
                for (uint32_t expert = lane; expert < config.num_experts; expert += blockDim.x)
                {
                    reset_runtime_histogram_expert(runtime, expert);
                }
                if (leader)
                {
                    runtime.router_hot_cache_eligible_dispatches = 0ULL;
                    runtime.router_hot_cache_used_dispatches = 0ULL;
                    runtime.router_hot_cache_improved_dispatches = 0ULL;
                    runtime.router_hot_cache_default_load_spread_total = 0ULL;
                    runtime.router_hot_cache_actual_load_spread_total = 0ULL;
                    runtime.router_hot_cache_load_spread_improvement_total = 0ULL;
                    runtime.router_hot_cache_active_dispatches = 0ULL;
                    runtime.router_hot_cache_miss_dispatches = 0ULL;
                    runtime.router_hot_cache_selected_expert_slots = 0ULL;
                    runtime.router_hot_cache_replicated_selected_expert_slots = 0ULL;
                }
            }
            if (leader)
            {
                uint64_t layer_post_total = 0ULL;
                uint64_t layer_post_min = 0ULL;
                uint64_t layer_post_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_current_policy_load,
                    config.participant_count,
                    layer_post_total,
                    layer_post_min,
                    layer_post_max);
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_post_policy_load[participant] += shared_current_policy_load[participant];
                post_wave_load_total += layer_post_total;
                post_wave_load_spread += layer_post_max - layer_post_min;
            }
            __syncthreads();
        }

        if (leader)
        {
            uint64_t pre_total = 0ULL;
            uint64_t pre_min = 0ULL;
            uint64_t pre_max = 0ULL;
            uint64_t post_total = 0ULL;
            uint64_t post_min = 0ULL;
            uint64_t post_max = 0ULL;
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_pre_policy_load,
                config.participant_count,
                pre_total,
                pre_min,
                pre_max);
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_post_policy_load,
                config.participant_count,
                post_total,
                post_min,
                post_max);
            status->status_code = invalid_layers > 0u
                                      ? kDeviceMoERebalanceStatusInvalidRuntime
                                      : kDeviceMoERebalanceStatusOk;
            status->windows_observed = 1u;
            status->windows_applied = changed_layers > 0u ? 1u : 0u;
            status->changed_layers = changed_layers;
            status->selected_replicas = selected_replicas;
            status->skipped_no_resident = skipped_no_resident;
            status->skipped_no_improvement = skipped_no_improvement;
            status->dynamic_ownership_swap_attempts = dynamic_ownership_swap_attempts;
            status->dynamic_ownership_swap_accepts = dynamic_ownership_swap_accepts;
            status->dynamic_ownership_swap_rejections = dynamic_ownership_swap_rejections;
            status->candidate_arrivals_considered = candidate_arrivals_considered;
            status->candidate_arrivals_below_floor = candidate_arrivals_below_floor;
            status->candidate_arrivals_pruned_by_count_bound =
                candidate_arrivals_pruned_by_count_bound;
            status->llep_assignment_span_count = llep_assignment_span_count;
            status->llep_weight_transfer_count = llep_weight_transfer_count;
            status->llep_standard_ep_selected = llep_standard_ep_selected;
            status->llep_skipped_balanced = llep_skipped_balanced;
            status->llep_skipped_insufficient_spread_improvement =
                llep_skipped_insufficient_spread_improvement;
            status->llep_skipped_insufficient_foreign_rows =
                llep_skipped_insufficient_foreign_rows;
            status->llep_min_chunk_skips = llep_min_chunk_skips;
            status->llep_forced_spills = llep_forced_spills;
            status->candidate_load_spread_improvement_total =
                candidate_load_spread_improvement_total;
            status->candidate_load_spread_improvement_max =
                candidate_load_spread_improvement_max;
            status->accepted_load_spread_improvement_total =
                accepted_load_spread_improvement_total;
            status->accepted_load_spread_improvement_max =
                accepted_load_spread_improvement_max;
            status->llep_native_rows = llep_native_rows;
            status->llep_spilled_rows = llep_spilled_rows;
            status->llep_required_spread_improvement =
                llep_required_spread_improvement;
            status->llep_required_foreign_rows = llep_required_foreign_rows;
            status->router_hot_cache_eligible_dispatches =
                router_hot_cache_eligible_dispatches;
            status->router_hot_cache_used_dispatches =
                router_hot_cache_used_dispatches;
            status->router_hot_cache_improved_dispatches =
                router_hot_cache_improved_dispatches;
            status->router_hot_cache_default_load_spread_total =
                router_hot_cache_default_load_spread_total;
            status->router_hot_cache_actual_load_spread_total =
                router_hot_cache_actual_load_spread_total;
            status->router_hot_cache_load_spread_improvement_total =
                router_hot_cache_load_spread_improvement_total;
            status->router_hot_cache_active_dispatches =
                router_hot_cache_active_dispatches;
            status->router_hot_cache_miss_dispatches =
                router_hot_cache_miss_dispatches;
            status->router_hot_cache_selected_expert_slots =
                router_hot_cache_selected_expert_slots;
            status->router_hot_cache_replicated_selected_expert_slots =
                router_hot_cache_replicated_selected_expert_slots;
            status->invalid_runtime_layers = invalid_layers;
            status->last_epoch = last_epoch;
            status->window_ready_slots =
                rebalance_clamp_u64_to_u32(
                    rebalance_window_observed_slots(gathered_histograms, config));
            status->window_required_slots =
                rebalance_clamp_u64_to_u32(rebalance_window_required_slots(config));
            if (leader && collect_load_stats)
            {
                status->pre_policy_load_total = pre_total;
                status->pre_policy_load_min = pre_min;
                status->pre_policy_load_max = pre_max;
                status->pre_policy_imbalance_numerator = pre_max - pre_min;
                status->pre_policy_imbalance_denominator = pre_total;
                status->post_policy_load_total = post_total;
                status->post_policy_load_min = post_min;
                status->post_policy_load_max = post_max;
                status->post_policy_imbalance_numerator = post_max - post_min;
                status->post_policy_imbalance_denominator = post_total;
                for (uint32_t participant = 0;
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    status->pre_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_pre_policy_load[participant]
                            : 0ULL;
                    status->post_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_post_policy_load[participant]
                            : 0ULL;
                }
            }

            uint32_t command_count =
                plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
            llaminar2::moe_rebalance_policy::PayloadPlanDemand payload_demand;
            for (uint32_t i = 0; plan_entries && i < command_count; ++i)
            {
                const auto &plan = plan_entries[i];
                llaminar2::moe_rebalance_policy::accumulatePayloadPlanDemand(
                    payload_demand, plan.op, plan.source_participant,
                    plan.destination_participant, plan.payload_slot,
                    config.participant_count);
            }
            uint32_t requested_payload_slots = payload_demand.requested_slots;
            uint32_t payload_source_participant_mask = payload_demand.source_mask;
            uint32_t payload_destination_participant_mask = payload_demand.destination_mask;
            uint64_t payload_edge_mask = payload_demand.edge_mask;
            uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);
            uint32_t payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            uint32_t payload_bucket_overflow =
                (status->plan_overflow != 0u ||
                 requested_payload_slots > payload_bucket_slots)
                    ? 1u
                    : 0u;
            const bool wave_cost_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
                    accepted_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_wave_spread_improvement_per_payload_slot);
            const bool router_benefit_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsRealizedRouterBenefitFloor(
                    router_hot_cache_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_router_spread_improvement_per_payload_slot,
                    hot_cache_active_layers > 0u);
            const bool realized_router_payback =
                requested_payload_slots > 0u &&
                router_hot_cache_used_dispatches > 0ULL &&
                llaminar2::moe_rebalance_policy::transferWaveMeetsRealizedRouterBenefitFloor(
                    router_hot_cache_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_router_spread_improvement_per_payload_slot,
                    true);
            const bool durable_llep_ownership_wave =
                least_loaded_assignment && !hot_replica_cache;
            /*
             * Layers execute serially, so the sum of per-layer participant
             * spreads is the relevant compute critical path.  Whole-request
             * participant totals can tie when opposite layer skews cancel;
             * do not let that lossy aggregate veto a dynamic ownership wave
             * whose retained plan demonstrably reduces the layer-wise sum.
             */
            const bool realized_layer_critical_path_payback =
                requested_payload_slots > 0u &&
                dynamic_ownership_swap_accepts > 0u &&
                post_wave_load_spread < pre_wave_load_spread;
            const bool participant_load_spread_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
                     pre_max - pre_min,
                     post_max - post_min,
                     pre_total,
                     post_total,
                     requested_payload_slots,
                     realized_router_payback ||
                         realized_layer_critical_path_payback);
            const bool aggregate_load_spread_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
                     pre_wave_load_spread,
                     post_wave_load_spread,
                     pre_wave_load_total,
                     post_wave_load_total,
                     requested_payload_slots);
            const bool configured_load_spread_ceiling_rejected =
                requested_payload_slots > 0u &&
                !durable_llep_ownership_wave &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
                     post_wave_load_spread,
                     post_wave_load_total,
                     requested_payload_slots,
                     config.max_post_wave_load_spread_per_mille);
            const bool post_load_spread_ceiling_rejected =
                participant_load_spread_rejected ||
                aggregate_load_spread_rejected ||
                configured_load_spread_ceiling_rejected;
            const bool wave_rejected =
                wave_cost_floor_rejected ||
                router_benefit_floor_rejected ||
                post_load_spread_ceiling_rejected;
            if (wave_rejected)
            {
                /* Report only ownership swaps retained in the published wave. */
                dynamic_ownership_swap_rejections +=
                    dynamic_ownership_swap_accepts;
                dynamic_ownership_swap_accepts = 0u;
                uint32_t resident_command_count = 0u;
                command_count =
                    llaminar2::moe_rebalance_policy::prunePayloadArrivalsPreservingResidentAssignments(
                        plan_entries,
                        command_count,
                        &resident_command_count);
                if (plan_count)
                    *plan_count = command_count;
                skipped_no_improvement +=
                    selected_replicas > resident_command_count
                        ? selected_replicas - resident_command_count
                        : 1u;
                selected_replicas = resident_command_count;
                accepted_load_spread_improvement_total = 0ULL;
                accepted_load_spread_improvement_max = 0ULL;
                requested_payload_slots = 0u;
                payload_bucket_slots = 0u;
                payload_bucket_index = 0u;
                payload_bucket_overflow = 0u;
                payload_source_participant_mask = 0u;
                payload_destination_participant_mask = 0u;
                payload_edge_mask = 0ULL;
                if (command_count == 0u)
                {
                    changed_layers = 0u;
                    last_epoch = 0u;
                }
            }
            if (wave_rejected)
            {
                status->windows_applied = command_count > 0u ? 1u : 0u;
                status->changed_layers = changed_layers;
                status->selected_replicas = selected_replicas;
                status->skipped_no_improvement = skipped_no_improvement;
                status->accepted_load_spread_improvement_total =
                    accepted_load_spread_improvement_total;
                status->accepted_load_spread_improvement_max =
                    accepted_load_spread_improvement_max;
                status->last_epoch = last_epoch;
                status->planned_arrivals = 0u;
                if (collect_load_stats)
                {
                    status->post_policy_load_total = status->pre_policy_load_total;
                    status->post_policy_load_min = status->pre_policy_load_min;
                    status->post_policy_load_max = status->pre_policy_load_max;
                    status->post_policy_imbalance_numerator =
                        status->pre_policy_imbalance_numerator;
                    status->post_policy_imbalance_denominator =
                        status->pre_policy_imbalance_denominator;
                    for (uint32_t participant = 0;
                         participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                         ++participant)
                    {
                        status->post_policy_participant_load[participant] =
                            status->pre_policy_participant_load[participant];
                    }
                }
            }
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index = payload_bucket_index;
            status->payload_bucket_overflow = payload_bucket_overflow;
            status->skipped_wave_cost_floor = wave_cost_floor_rejected ? 1u : 0u;
            status->skipped_low_router_benefit =
                router_benefit_floor_rejected ? 1u : 0u;
            status->skipped_post_load_spread_ceiling =
                post_load_spread_ceiling_rejected ? 1u : 0u;
            status->skipped_participant_load_spread =
                participant_load_spread_rejected ? 1u : 0u;
            status->skipped_aggregate_load_spread =
                aggregate_load_spread_rejected ? 1u : 0u;
            status->skipped_configured_load_spread_ceiling =
                configured_load_spread_ceiling_rejected ? 1u : 0u;
            /* The terminal economy gate may prune provisional commands.  The
             * published counters describe only commands that survived it. */
            status->dynamic_ownership_swap_accepts =
                dynamic_ownership_swap_accepts;
            status->dynamic_ownership_swap_rejections =
                dynamic_ownership_swap_rejections;
            status->payload_source_participant_mask = payload_source_participant_mask;
            status->payload_destination_participant_mask =
                payload_destination_participant_mask;
            status->payload_edge_mask = payload_edge_mask;
            status->pre_wave_load_total = pre_wave_load_total;
            status->pre_wave_load_spread = pre_wave_load_spread;
            status->post_wave_load_total = post_wave_load_total;
            status->post_wave_load_spread = post_wave_load_spread;
            const uint32_t start_layer =
                (layer_window_start + start_offset) % config.num_layers;
            uint32_t planned_start_layer = start_layer;
            uint32_t planned_layer_count = layer_wave_count;
            if (command_count > 0u)
            {
                rebalance_command_layer_span(
                    plan_entries,
                    command_count,
                    config,
                    layer_window_start,
                    start_offset,
                    layer_window_count,
                    layer_wave_count,
                    &planned_start_layer,
                    &planned_layer_count);
            }
            if (command_header)
            {
                command_header->epoch = defer_runtime_apply && command_count > 0u
                                            ? shared_plan_epoch
                                            : last_epoch;
                command_header->command_count = command_count;
                command_header->movement_decision = {};
                // The general planner can also emit an ownership-only wave.
                // Seal it at the same post-admission boundary as the dedicated
                // ownership kernel; mixed replica/assignment waves are distinct.
                if (!LeastLoadedAssignment && command_count > 0u &&
                    dynamic_ownership_swap_accepts <= UINT32_MAX / 2u &&
                    command_count == dynamic_ownership_swap_accepts * 2u)
                {
                    auto &decision = command_header->movement_decision;
                    decision.kind = llaminar2::DeviceMoERebalanceMovementDecisionKind::NativeOwnershipSpread;
                    decision.load = {
                        accepted_load_spread_improvement_total,
                        pre_wave_load_spread, post_wave_load_spread,
                        pre_wave_load_total, post_wave_load_total,
                        pre_max - pre_min, post_max - post_min, pre_total, post_total,
                        requested_payload_slots, config.min_wave_spread_improvement_per_payload_slot,
                        config.max_post_wave_load_spread_per_mille, dynamic_ownership_swap_accepts};
                }
            }
            if (wave_state)
            {
                wave_state->epoch = last_epoch;
                wave_state->planned_start_layer = planned_start_layer;
                wave_state->planned_layer_count = planned_layer_count;
                wave_state->requested_payload_slots = requested_payload_slots;
                wave_state->payload_bucket_slots = payload_bucket_slots;
                wave_state->payload_bucket_index = payload_bucket_index;
                wave_state->payload_bucket_overflow = payload_bucket_overflow;
                const uint32_t next_offset =
                    layer_window_count > 0u
                        ? ((start_offset + layer_wave_count) % layer_window_count)
                        : 0u;
                wave_state->next_start_layer =
                    (layer_window_start + next_offset) % config.num_layers;
            }
            if (defer_runtime_apply &&
                command_count > 0u &&
                controller_state &&
                rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                command_wave_index < controller_state->wave_count)
            {
                const bool transfer_payload_required = requested_payload_slots > 0u;
                auto &wave = controller_state->waves[command_wave_index];
                wave.magic = kDeviceMoERebalanceMagic;
                wave.version = kDeviceMoERebalanceVersion;
                wave.epoch = shared_plan_epoch;
                wave.state = kDeviceMoERebalanceLifecyclePlanning;
                wave.planned_start_layer =
                    wave_state ? wave_state->planned_start_layer : planned_start_layer;
                wave.planned_layer_count =
                    wave_state ? wave_state->planned_layer_count : planned_layer_count;
                wave.command_count = command_count;
                wave.copied_arrivals = transfer_payload_required ? 0u : command_count;
                wave.applied_arrivals = 0u;
                wave.applied_layer_count = 0u;
                wave.error_code = 0u;
                wave.requested_payload_slots = requested_payload_slots;
                wave.payload_bucket_slots = payload_bucket_slots;
                wave.payload_bucket_index = payload_bucket_index;
                wave.payload_bucket_overflow = payload_bucket_overflow;
                __threadfence();
                wave.state = transfer_payload_required
                                 ? kDeviceMoERebalanceLifecyclePlanning
                                 : kDeviceMoERebalanceLifecycleReadyToApply;
                controller_state->next_epoch = shared_plan_epoch + 1u;
            }
            else if (defer_runtime_apply &&
                     command_count == 0u &&
                     controller_state &&
                     rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                     command_wave_index < controller_state->wave_count)
            {
                reset_rebalance_wave_progress_device(controller_state->waves[command_wave_index]);
            }
        }
    }

    // The public bridge always launches exactly one expert-width workgroup.
    __global__ __launch_bounds__(kDeviceMoEMaxExperts) void device_rebalance_dynamic_ownership_controller_kernel(
        DeviceMoELayerRuntimeView *runtime_layers,
        const unsigned long long *gathered_histograms,
        DeviceMoERebalanceStatusView *status,
        DeviceMoERebalanceConfigView config,
        DeviceMoERebalancePlanEntryView *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeaderView *command_header,
        DeviceMoERebalanceWaveStateView *wave_state,
        DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count,
        const DeviceMoEExpertDirectoryEntryView *local_transfer_slots,
        uint32_t local_transfer_slot_count)
    {
        if (blockIdx.x != 0) return;

        const uint32_t lane = threadIdx.x;
        const bool leader = lane == 0u;
        __shared__ uint32_t shared_command_wave_index;
        __shared__ uint32_t shared_abort;
        __shared__ uint32_t shared_layer_invalid;
        __shared__ uint32_t shared_plan_epoch;
        __shared__ uint64_t shared_owner_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_pre_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_post_policy_load[kDeviceMoEMaxParticipants];
        __shared__ uint64_t shared_expert_counts[kDeviceMoEMaxExperts];
        __shared__ int32_t shared_expert_owners[kDeviceMoEMaxExperts];
        __shared__ uint32_t
            shared_expert_transfer_backed_participant_mask[kDeviceMoEMaxExperts];
        __shared__ uint32_t shared_destination_transfer_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t shared_source_payload_slot_counts[kDeviceMoEMaxParticipants];
        __shared__ uint32_t
            shared_active_transfer_slot_counts[kDeviceMoEMaxParticipants];

        if (leader)
        {
            shared_command_wave_index =
                rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
            shared_abort = 0u;
        }
        for (uint32_t participant = lane;
             participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
             participant += blockDim.x)
        {
            shared_owner_policy_load[participant] = 0ULL;
            shared_pre_policy_load[participant] = 0ULL;
            shared_post_policy_load[participant] = 0ULL;
            shared_destination_transfer_slot_counts[participant] = 0u;
            shared_source_payload_slot_counts[participant] = 0u;
            shared_active_transfer_slot_counts[participant] =
                gathered_histograms &&
                        config.num_experts > 0u &&
                        rebalance_layer_wave_count(config) > 0u &&
                        participant < config.participant_count
                    ? rebalance_collected_active_transfer_slot_count(
                          gathered_histograms, config, participant)
                    : 0u;
        }
        for (uint32_t expert = lane;
             expert < static_cast<uint32_t>(kDeviceMoEMaxExperts);
             expert += blockDim.x)
        {
            shared_expert_counts[expert] = 0ULL;
            shared_expert_owners[expert] = -1;
            shared_expert_transfer_backed_participant_mask[expert] = 0u;
        }
        __syncthreads();

        const uint32_t command_wave_index = shared_command_wave_index;
        if (plan_entries)
            plan_entries += static_cast<unsigned long long>(command_wave_index) * plan_capacity;
        if (plan_count)
            plan_count += command_wave_index;
        if (command_header)
            command_header += command_wave_index;
        if (wave_state)
            wave_state += command_wave_index;

        if (leader)
        {
            if (status)
            {
                DeviceMoERebalanceStatusView zero_status{};
                *status = zero_status;
                status->magic = kDeviceMoERebalanceMagic;
                status->version = kDeviceMoERebalanceVersion;
                status->status_code = kDeviceMoERebalanceStatusOk;
                rebalance_publish_transfer_slot_claim_summary(
                    status,
                    runtime_layers,
                    config,
                    local_transfer_slots,
                    local_transfer_slot_count);
                if (status->prefill_duplicate_transfer_slot_claims != 0u ||
                    status->prefill_invalid_transfer_slot_claims != 0u ||
                    status->prefill_active_transfer_slot_experts >
                        config.active_transfer_slot_capacity)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    shared_abort = 1u;
                }
            }
            else
            {
                shared_abort = 1u;
            }

            if (shared_abort == 0u)
            {
                const uint32_t boundary =
                    consume_rebalance_maintenance_boundary(
                        controller_state,
                        config);
                if (boundary == 0u)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
                else if (boundary != 1u)
                {
                    status->status_code =
                        kDeviceMoERebalanceStatusInvalidRuntime;
                    shared_abort = 1u;
                }
            }

            if (shared_abort == 0u &&
                rebalance_active_wave_busy_for_new_plan(
                    controller_state,
                    config,
                    command_wave_index))
            {
                status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                status->skipped_not_ready = 1u;
                status->skipped_busy_wave = 1u;
                shared_abort = 1u;
            }

            if (plan_count)
                *plan_count = 0u;

            if (command_header)
            {
                command_header->magic = kDeviceMoERebalanceMagic;
                command_header->version = kDeviceMoERebalanceVersion;
                command_header->epoch = 0u;
                command_header->phase = kDeviceMoERebalancePhasePlanAssignments;
                command_header->command_count = 0u;
                command_header->command_capacity = plan_capacity;
                command_header->participant_id = config.participant_id;
                command_header->participant_count = config.participant_count;
            }

            if (wave_state)
            {
                const bool keep_cursor =
                    wave_state->magic == kDeviceMoERebalanceMagic &&
                    wave_state->version == kDeviceMoERebalanceVersion &&
                    wave_state->participant_id == config.participant_id &&
                    wave_state->participant_count == config.participant_count;
                const uint32_t next_start = keep_cursor ? wave_state->next_start_layer : 0u;
                wave_state->magic = kDeviceMoERebalanceMagic;
                wave_state->version = kDeviceMoERebalanceVersion;
                wave_state->epoch = 0u;
                wave_state->next_start_layer = next_start;
                wave_state->planned_start_layer = 0u;
                wave_state->planned_layer_count = 0u;
                wave_state->command_capacity = plan_capacity;
                wave_state->participant_id = config.participant_id;
                wave_state->participant_count = config.participant_count;
                wave_state->requested_payload_slots = 0u;
                wave_state->payload_bucket_slots = 0u;
                wave_state->payload_bucket_index = 0u;
                wave_state->payload_bucket_overflow = 0u;
            }

            if (shared_abort == 0u && !rebalance_config_ok(config))
            {
                status->status_code = kDeviceMoERebalanceStatusInvalidConfig;
                shared_abort = 1u;
            }
            if (shared_abort == 0u && (!runtime_layers || !gathered_histograms))
            {
                status->status_code = !runtime_layers
                                          ? kDeviceMoERebalanceStatusInvalidRuntime
                                          : kDeviceMoERebalanceStatusMissingHistogram;
                shared_abort = 1u;
            }
            if (shared_abort == 0u)
            {
                const unsigned long long observed_slots =
                    rebalance_window_observed_slots(gathered_histograms, config);
                const unsigned long long required_slots =
                    rebalance_window_required_slots(config);
                status->window_ready_slots =
                    rebalance_clamp_u64_to_u32(observed_slots);
                status->window_required_slots =
                    rebalance_clamp_u64_to_u32(required_slots);
                if (observed_slots < required_slots)
                {
                    status->status_code = kDeviceMoERebalanceStatusWindowNotReady;
                    status->skipped_not_ready = 1u;
                    status->skipped_busy_wave = 0u;
                    shared_abort = 1u;
                }
            }
        }
        __syncthreads();
        if (shared_abort != 0u)
            return;

        const uint32_t valid_mask = runtime_valid_participant_mask(config.participant_count);
        const bool collect_load_stats =
            (config.flags & kDeviceMoERebalanceFlagCollectLoadStats) != 0u;
        const uint32_t layer_window_count =
            config.layer_window_count == 0u
                ? config.num_layers
                : ((config.layer_window_count < config.num_layers)
                       ? config.layer_window_count
                       : config.num_layers);
        const uint32_t layer_wave_count =
            config.layer_wave_count == 0u
                ? layer_window_count
                : ((config.layer_wave_count < layer_window_count)
                       ? config.layer_wave_count
                       : layer_window_count);
        const uint32_t layer_window_start =
            config.num_layers == 0u ? 0u : (config.layer_window_start % config.num_layers);
        uint32_t start_offset = 0u;
        if (wave_state && layer_window_count > 0u)
        {
            const uint32_t candidate = wave_state->next_start_layer % config.num_layers;
            const uint32_t candidate_offset =
                (candidate + config.num_layers - layer_window_start) % config.num_layers;
            if (candidate_offset < layer_window_count)
                start_offset = candidate_offset;
        }

        if (leader)
        {
            shared_plan_epoch =
                (rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                 controller_state->next_epoch > 0u)
                    ? controller_state->next_epoch
                    : 1u;
        }
        __syncthreads();

        uint32_t changed_layers = 0u;
        uint32_t invalid_layers = 0u;
        uint32_t last_epoch = 0u;
        uint32_t dynamic_ownership_swap_attempts = 0u;
        uint32_t dynamic_ownership_swap_accepts = 0u;
        uint32_t dynamic_ownership_swap_rejections = 0u;
        uint32_t skipped_no_resident = 0u;
        uint32_t skipped_no_improvement = 0u;
        uint64_t accepted_load_spread_improvement_total = 0ULL;
        uint64_t accepted_load_spread_improvement_max = 0ULL;
        uint64_t pre_wave_load_total = 0ULL;
        uint64_t pre_wave_load_spread = 0ULL;
        uint64_t post_wave_load_total = 0ULL;
        uint64_t post_wave_load_spread = 0ULL;

        if (config.participant_id != config.root_participant)
        {
            if (leader)
            {
                status->status_code = kDeviceMoERebalanceStatusOk;
                status->windows_observed = 1u;
                if (command_header)
                    command_header->command_count = 0u;
                if (wave_state)
                {
                    const uint32_t start_layer =
                        (layer_window_start + start_offset) % config.num_layers;
                    wave_state->planned_start_layer = start_layer;
                    wave_state->planned_layer_count = layer_wave_count;
                    const uint32_t next_offset =
                        layer_window_count > 0u
                            ? ((start_offset + layer_wave_count) % layer_window_count)
                            : 0u;
                    wave_state->next_start_layer =
                        (layer_window_start + next_offset) % config.num_layers;
                }
            }
            return;
        }

        /*
         * Build the wave-wide participant baseline before selecting any
         * layer-local swap. A layer can call participant 0 "overloaded" while
         * earlier or later layers already make participant 1 the aggregate
         * bottleneck. Planning directly into this shared working vector makes
         * every accepted command improve the actual multi-layer device load;
         * the final economy gate is then a verifier, not a place where an
         * entire payload wave is discarded after doing speculative work.
         */
        if (leader)
        {
            for (uint32_t window_index = 0;
                 window_index < layer_wave_count;
                 ++window_index)
            {
                const uint32_t layer =
                    (layer_window_start +
                     ((start_offset + window_index) % layer_window_count)) %
                    config.num_layers;
                const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
                if (runtime.active_bank > 1u ||
                    runtime.expert_count != config.num_experts ||
                    runtime.top_k != config.top_k ||
                    runtime.participant_id != config.participant_id ||
                    runtime.participant_count != config.participant_count)
                {
                    ++invalid_layers;
                    continue;
                }

                for (uint32_t participant = 0;
                     participant < config.participant_count;
                     ++participant)
                {
                    shared_owner_policy_load[participant] = 0ULL;
                }
                const DeviceMoEPlacementBankView &active =
                    runtime.banks[runtime.active_bank];
                for (uint32_t expert = 0;
                     expert < config.num_experts;
                     ++expert)
                {
                    const int32_t owner =
                        active.experts[expert].owner_participant;
                    if (owner >= 0 &&
                        owner < static_cast<int32_t>(config.participant_count))
                    {
                        shared_owner_policy_load[static_cast<uint32_t>(owner)] +=
                            rebalance_global_count(
                                gathered_histograms,
                                config,
                                window_index,
                                expert);
                    }
                }

                uint64_t layer_total = 0ULL;
                uint64_t layer_min = 0ULL;
                uint64_t layer_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_owner_policy_load,
                    config.participant_count,
                    layer_total,
                    layer_min,
                    layer_max);
                for (uint32_t participant = 0;
                     participant < config.participant_count;
                     ++participant)
                {
                    shared_pre_policy_load[participant] +=
                        shared_owner_policy_load[participant];
                }
                pre_wave_load_total += layer_total;
                pre_wave_load_spread += layer_max - layer_min;
            }
            for (uint32_t participant = 0;
                 participant < config.participant_count;
                 ++participant)
            {
                shared_post_policy_load[participant] =
                    shared_pre_policy_load[participant];
            }
        }
        __syncthreads();

        for (uint32_t window_index = 0; window_index < layer_wave_count; ++window_index)
        {
            const uint32_t layer =
                (layer_window_start + ((start_offset + window_index) % layer_window_count)) %
                config.num_layers;
            DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            if (leader)
            {
                shared_layer_invalid =
                    (runtime.active_bank > 1u ||
                     runtime.expert_count != config.num_experts ||
                     runtime.top_k != config.top_k ||
                     runtime.participant_id != config.participant_id ||
                     runtime.participant_count != config.participant_count)
                        ? 1u
                        : 0u;
            }
            __syncthreads();
            if (shared_layer_invalid != 0u)
                continue;

            uint32_t layer_command_count_before = 0u;
            if (leader && plan_count)
                layer_command_count_before =
                    *plan_count < plan_capacity ? *plan_count : plan_capacity;

            const DeviceMoEPlacementBankView &active = runtime.banks[runtime.active_bank];
            if (leader)
            {
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    shared_owner_policy_load[participant] = 0ULL;
                for (uint32_t expert = 0; expert < config.num_experts; ++expert)
                {
                    const DeviceMoEExpertDescriptorView &desc = active.experts[expert];
                    const unsigned long long count =
                        rebalance_global_count(gathered_histograms, config, window_index, expert);
                    shared_expert_counts[expert] = count;
                    shared_expert_transfer_backed_participant_mask[expert] =
                        rebalance_collected_transfer_backed_participant_mask(
                            gathered_histograms,
                            config,
                            window_index,
                            expert);
                    shared_expert_owners[expert] =
                        desc.owner_participant >= 0 &&
                                desc.owner_participant < static_cast<int32_t>(config.participant_count)
                            ? desc.owner_participant
                            : -1;
                    if (shared_expert_owners[expert] >= 0)
                        shared_owner_policy_load[static_cast<uint32_t>(shared_expert_owners[expert])] += count;
                }

                const uint32_t max_swaps = config.dynamic_max_swaps_per_layer;
                for (uint32_t swap_iter = 0; swap_iter < max_swaps; ++swap_iter)
                {
                    const uint32_t command_count =
                        plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                    const auto admission =
                        llaminar2::moe_rebalance_policy::dynamicPlanAdmission(
                            command_count,
                            /*required_entries=*/2u,
                            config.dynamic_max_plan_entries_per_wave,
                            plan_capacity);
                    if (admission == llaminar2::moe_rebalance_policy::
                                         DynamicPlanAdmission::ConfiguredWaveLimit)
                    {
                        if (status)
                            ++status->capacity_limited_candidates;
                        break;
                    }
                    if (admission == llaminar2::moe_rebalance_policy::
                                         DynamicPlanAdmission::PhysicalPlanOverflow)
                    {
                        if (status)
                            ++status->plan_overflow;
                        break;
                    }

                    ++dynamic_ownership_swap_attempts;
                    const auto swap_choice =
                        llaminar2::moe_rebalance_policy::bestDynamicOwnershipSwap(
                            shared_owner_policy_load,
                            shared_expert_counts,
                            shared_expert_owners,
                            config.num_experts,
                            config.participant_count,
                            llaminar2::moe_rebalance_policy::
                                dynamicOwnershipEvidenceFromParticipantLoads(
                                    shared_owner_policy_load,
                                    config.participant_count,
                                    config.dynamic_min_window_activations),
                            config.dynamic_imbalance_threshold_per_mille,
                            config.dynamic_min_improvement_per_mille,
                            shared_expert_transfer_backed_participant_mask,
                            shared_active_transfer_slot_counts,
                            config.active_transfer_slot_capacity);
                    if (!swap_choice.valid)
                    {
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }

                    const uint32_t heavy_source = swap_choice.overloaded_participant;
                    const uint32_t heavy_destination = swap_choice.underloaded_participant;
                    const uint32_t light_source = swap_choice.underloaded_participant;
                    const uint32_t light_destination = swap_choice.overloaded_participant;
                    const uint32_t heavy_physical_source_mask =
                        active.resident_participant_mask[swap_choice.heavy_expert] & valid_mask;
                    const uint32_t light_physical_source_mask =
                        active.resident_participant_mask[swap_choice.light_expert] & valid_mask;
                    const int32_t heavy_destination_overlay_participant =
                        rebalance_overlay_route_for_domain_participant(
                            active,
                            heavy_destination,
                            config.num_experts);
                    const int32_t light_destination_overlay_participant =
                        rebalance_overlay_route_for_domain_participant(
                            active,
                            light_destination,
                            config.num_experts);
                    if (runtime.overlay_epoch_ticket &&
                        (heavy_destination_overlay_participant < 0 ||
                         light_destination_overlay_participant < 0))
                    {
                        status->status_code =
                            kDeviceMoERebalanceStatusInvalidRuntime;
                        shared_abort = 1u;
                        break;
                    }
                    const uint32_t heavy_source_bit =
                        runtime_participant_bit(static_cast<int>(heavy_source));
                    const uint32_t light_source_bit =
                        runtime_participant_bit(static_cast<int>(light_source));
                    if (heavy_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        heavy_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_source >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        light_destination >= static_cast<uint32_t>(kDeviceMoEMaxParticipants) ||
                        shared_source_payload_slot_counts[heavy_source] >= payload_slot_capacity ||
                        shared_source_payload_slot_counts[light_source] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[heavy_destination] >= payload_slot_capacity ||
                        shared_destination_transfer_slot_counts[light_destination] >= payload_slot_capacity)
                    {
                        if (status)
                            ++status->capacity_limited_candidates;
                        break;
                    }
                    if ((heavy_physical_source_mask & heavy_source_bit) == 0u ||
                        (light_physical_source_mask & light_source_bit) == 0u)
                    {
                        /*
                         * The graph-captured controller publishes commands that the
                         * payload packer consumes later.  Reject an ownership swap
                         * immediately if either old owner is not a physical resident.
                         */
                        ++skipped_no_resident;
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }

                    const auto aggregate_delta =
                        llaminar2::moe_rebalance_policy::
                            evaluateDynamicOwnershipSwapAgainstAggregateLoad(
                                shared_post_policy_load,
                                config.participant_count,
                                swap_choice);
                    if (!aggregate_delta.improves)
                    {
                        /* Keep the command buffer free of globally harmful work. */
                        ++skipped_no_improvement;
                        ++dynamic_ownership_swap_rejections;
                        break;
                    }

                    DeviceMoERebalancePlanEntryView heavy_entry{};
                    heavy_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    heavy_entry.layer = layer;
                    heavy_entry.expert = swap_choice.heavy_expert;
                    heavy_entry.source_participant = heavy_source;
                    heavy_entry.activation_count = shared_expert_counts[swap_choice.heavy_expert];
                    heavy_entry.destination_participant = heavy_destination;
                    heavy_entry.destination_overlay_participant =
                        heavy_destination_overlay_participant;
                    heavy_entry.source_resident_mask = heavy_physical_source_mask;
                    heavy_entry.destination_slot =
                        shared_destination_transfer_slot_counts[heavy_destination];
                    heavy_entry.payload_slot =
                        shared_source_payload_slot_counts[heavy_source];

                    DeviceMoERebalancePlanEntryView light_entry{};
                    light_entry.op = kDeviceMoERebalancePlanOwnershipTransfer;
                    light_entry.layer = layer;
                    light_entry.expert = swap_choice.light_expert;
                    light_entry.source_participant = light_source;
                    light_entry.activation_count = shared_expert_counts[swap_choice.light_expert];
                    light_entry.destination_participant = light_destination;
                    light_entry.destination_overlay_participant =
                        light_destination_overlay_participant;
                    light_entry.source_resident_mask = light_physical_source_mask;
                    light_entry.destination_slot =
                        shared_destination_transfer_slot_counts[light_destination];
                    light_entry.payload_slot =
                        shared_source_payload_slot_counts[light_source];

                    if (!rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            heavy_entry,
                            status) ||
                        !rebalance_append_plan_entry(
                            plan_entries,
                            plan_count,
                            plan_capacity,
                            light_entry,
                            status))
                    {
                        break;
                    }

                    ++dynamic_ownership_swap_accepts;
                    ++shared_source_payload_slot_counts[heavy_source];
                    ++shared_source_payload_slot_counts[light_source];
                    ++shared_destination_transfer_slot_counts[heavy_destination];
                    ++shared_destination_transfer_slot_counts[light_destination];
                    /*
                     * Commit the proposal that was authenticated immediately
                     * above.  Re-evaluating it after publishing the commands
                     * would create an impossible-to-rollback failure edge.
                     */
                    const uint64_t aggregate_shift =
                        swap_choice.heavy_count - swap_choice.light_count;
                    shared_post_policy_load[swap_choice.overloaded_participant] -=
                        aggregate_shift;
                    shared_post_policy_load[swap_choice.underloaded_participant] +=
                        aggregate_shift;
                    llaminar2::moe_rebalance_policy::applyDynamicOwnershipSwap(
                        shared_owner_policy_load,
                        shared_expert_owners,
                        swap_choice);
                    llaminar2::moe_rebalance_policy::
                        applyDynamicOwnershipSwapTransferOccupancy(
                            shared_active_transfer_slot_counts,
                            shared_expert_transfer_backed_participant_mask,
                            swap_choice);
                    accepted_load_spread_improvement_total += swap_choice.improvement;
                    if (swap_choice.improvement > accepted_load_spread_improvement_max)
                        accepted_load_spread_improvement_max = swap_choice.improvement;
                }

                uint64_t layer_post_total = 0ULL;
                uint64_t layer_post_min = 0ULL;
                uint64_t layer_post_max = 0ULL;
                llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                    shared_owner_policy_load,
                    config.participant_count,
                    layer_post_total,
                    layer_post_min,
                    layer_post_max);
                post_wave_load_total += layer_post_total;
                post_wave_load_spread += layer_post_max - layer_post_min;

                const uint32_t command_count =
                    plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
                if (command_count > layer_command_count_before)
                {
                    last_epoch = shared_plan_epoch;
                    ++changed_layers;
                }
            }
            __syncthreads();
        }

        if (leader)
        {
            uint64_t pre_total = 0ULL;
            uint64_t pre_min = 0ULL;
            uint64_t pre_max = 0ULL;
            uint64_t post_total = 0ULL;
            uint64_t post_min = 0ULL;
            uint64_t post_max = 0ULL;
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_pre_policy_load,
                config.participant_count,
                pre_total,
                pre_min,
                pre_max);
            llaminar2::moe_rebalance_policy::finalizeLoadSpread(
                shared_post_policy_load,
                config.participant_count,
                post_total,
                post_min,
                post_max);

            uint32_t command_count =
                plan_count ? ((*plan_count < plan_capacity) ? *plan_count : plan_capacity) : 0u;
            llaminar2::moe_rebalance_policy::PayloadPlanDemand payload_demand;
            for (uint32_t i = 0; plan_entries && i < command_count; ++i)
            {
                const auto &plan = plan_entries[i];
                llaminar2::moe_rebalance_policy::accumulatePayloadPlanDemand(
                    payload_demand, plan.op, plan.source_participant,
                    plan.destination_participant, plan.payload_slot,
                    config.participant_count);
            }
            uint32_t requested_payload_slots = payload_demand.requested_slots;
            uint32_t payload_source_participant_mask = payload_demand.source_mask;
            uint32_t payload_destination_participant_mask = payload_demand.destination_mask;
            uint64_t payload_edge_mask = payload_demand.edge_mask;
            uint32_t payload_bucket_slots =
                llaminar2::moe_rebalance_policy::payloadBucketSlots(
                    requested_payload_slots,
                    payload_slot_capacity);
            uint32_t payload_bucket_index =
                llaminar2::moe_rebalance_policy::payloadBucketIndex(
                    payload_bucket_slots);
            uint32_t payload_bucket_overflow =
                (status->plan_overflow != 0u ||
                 requested_payload_slots > payload_bucket_slots)
                    ? 1u
                    : 0u;

            const bool wave_cost_floor_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
                    accepted_load_spread_improvement_total,
                    requested_payload_slots,
                    config.min_wave_spread_improvement_per_payload_slot);
            const bool realized_layer_critical_path_payback =
                requested_payload_slots > 0u &&
                dynamic_ownership_swap_accepts > 0u &&
                post_wave_load_spread < pre_wave_load_spread;
            const bool participant_load_spread_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
                     pre_max - pre_min,
                     post_max - post_min,
                     pre_total,
                     post_total,
                     requested_payload_slots,
                     realized_layer_critical_path_payback);
            const bool aggregate_load_spread_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
                     pre_wave_load_spread,
                     post_wave_load_spread,
                     pre_wave_load_total,
                     post_wave_load_total,
                     requested_payload_slots);
            const bool configured_load_spread_ceiling_rejected =
                requested_payload_slots > 0u &&
                !llaminar2::moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
                     post_wave_load_spread,
                     post_wave_load_total,
                     requested_payload_slots,
                     config.max_post_wave_load_spread_per_mille);
            const bool post_load_spread_ceiling_rejected =
                participant_load_spread_rejected ||
                aggregate_load_spread_rejected ||
                configured_load_spread_ceiling_rejected;
            const bool wave_rejected =
                wave_cost_floor_rejected || post_load_spread_ceiling_rejected;
            if (wave_rejected)
            {
                dynamic_ownership_swap_rejections +=
                    dynamic_ownership_swap_accepts;
                dynamic_ownership_swap_accepts = 0u;
                command_count = 0u;
                if (plan_count)
                    *plan_count = 0u;
                skipped_no_improvement += 1u;
                changed_layers = 0u;
                last_epoch = 0u;
                accepted_load_spread_improvement_total = 0ULL;
                accepted_load_spread_improvement_max = 0ULL;
                requested_payload_slots = 0u;
                payload_bucket_slots = 0u;
                payload_bucket_index = 0u;
                payload_bucket_overflow = 0u;
                payload_source_participant_mask = 0u;
                payload_destination_participant_mask = 0u;
                payload_edge_mask = 0ULL;
                post_total = pre_total;
                post_min = pre_min;
                post_max = pre_max;
            }

            status->status_code = invalid_layers > 0u
                                      ? kDeviceMoERebalanceStatusInvalidRuntime
                                      : kDeviceMoERebalanceStatusOk;
            status->windows_observed = 1u;
            status->windows_applied = command_count > 0u ? 1u : 0u;
            status->changed_layers = changed_layers;
            status->skipped_no_resident = skipped_no_resident;
            status->skipped_no_improvement = skipped_no_improvement;
            status->dynamic_ownership_swap_attempts = dynamic_ownership_swap_attempts;
            status->dynamic_ownership_swap_accepts = dynamic_ownership_swap_accepts;
            status->dynamic_ownership_swap_rejections = dynamic_ownership_swap_rejections;
            status->accepted_load_spread_improvement_total =
                accepted_load_spread_improvement_total;
            status->accepted_load_spread_improvement_max =
                accepted_load_spread_improvement_max;
            status->invalid_runtime_layers = invalid_layers;
            status->last_epoch = last_epoch;
            status->window_ready_slots =
                rebalance_clamp_u64_to_u32(
                    rebalance_window_observed_slots(gathered_histograms, config));
            status->window_required_slots =
                rebalance_clamp_u64_to_u32(rebalance_window_required_slots(config));
            status->payload_bucket_requested_slots = requested_payload_slots;
            status->payload_bucket_slots = payload_bucket_slots;
            status->payload_bucket_index = payload_bucket_index;
            status->payload_bucket_overflow = payload_bucket_overflow;
            status->skipped_wave_cost_floor = wave_cost_floor_rejected ? 1u : 0u;
            status->skipped_post_load_spread_ceiling =
                post_load_spread_ceiling_rejected ? 1u : 0u;
            status->skipped_participant_load_spread =
                participant_load_spread_rejected ? 1u : 0u;
            status->skipped_aggregate_load_spread =
                aggregate_load_spread_rejected ? 1u : 0u;
            status->skipped_configured_load_spread_ceiling =
                configured_load_spread_ceiling_rejected ? 1u : 0u;
            status->payload_source_participant_mask = payload_source_participant_mask;
            status->payload_destination_participant_mask =
                payload_destination_participant_mask;
            status->payload_edge_mask = payload_edge_mask;
            status->pre_wave_load_total = pre_wave_load_total;
            status->pre_wave_load_spread = pre_wave_load_spread;
            status->post_wave_load_total = post_wave_load_total;
            status->post_wave_load_spread = post_wave_load_spread;
            if (collect_load_stats)
            {
                status->pre_policy_load_total = pre_total;
                status->pre_policy_load_min = pre_min;
                status->pre_policy_load_max = pre_max;
                status->pre_policy_imbalance_numerator = pre_max - pre_min;
                status->pre_policy_imbalance_denominator = pre_total;
                status->post_policy_load_total = post_total;
                status->post_policy_load_min = post_min;
                status->post_policy_load_max = post_max;
                status->post_policy_imbalance_numerator = post_max - post_min;
                status->post_policy_imbalance_denominator = post_total;
                for (uint32_t participant = 0;
                     participant < static_cast<uint32_t>(kDeviceMoEMaxParticipants);
                     ++participant)
                {
                    status->pre_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_pre_policy_load[participant]
                            : 0ULL;
                    status->post_policy_participant_load[participant] =
                        participant < config.participant_count
                            ? shared_post_policy_load[participant]
                            : 0ULL;
                }
            }

            const uint32_t start_layer =
                (layer_window_start + start_offset) % config.num_layers;
            uint32_t planned_start_layer = start_layer;
            uint32_t planned_layer_count = layer_wave_count;
            if (command_count > 0u)
            {
                rebalance_command_layer_span(
                    plan_entries,
                    command_count,
                    config,
                    layer_window_start,
                    start_offset,
                    layer_window_count,
                    layer_wave_count,
                    &planned_start_layer,
                    &planned_layer_count);
            }
            if (command_header)
            {
                command_header->epoch = command_count > 0u ? shared_plan_epoch : last_epoch;
                command_header->command_count = command_count;
                // Seal policy inputs beside the commands before reusable
                // status/histograms can observe another planning window.
                command_header->movement_decision = {};
                if (command_count > 0u)
                {
                    auto &decision = command_header->movement_decision;
                    decision.kind = llaminar2::DeviceMoERebalanceMovementDecisionKind::NativeOwnershipSpread;
                    decision.load = {
                        accepted_load_spread_improvement_total,
                        pre_wave_load_spread, post_wave_load_spread,
                        pre_wave_load_total, post_wave_load_total,
                        pre_max - pre_min, post_max - post_min, pre_total, post_total,
                        requested_payload_slots, config.min_wave_spread_improvement_per_payload_slot,
                        config.max_post_wave_load_spread_per_mille, dynamic_ownership_swap_accepts};
                }
            }
            if (wave_state)
            {
                wave_state->epoch = last_epoch;
                wave_state->planned_start_layer = planned_start_layer;
                wave_state->planned_layer_count = planned_layer_count;
                wave_state->requested_payload_slots = requested_payload_slots;
                wave_state->payload_bucket_slots = payload_bucket_slots;
                wave_state->payload_bucket_index = payload_bucket_index;
                wave_state->payload_bucket_overflow = payload_bucket_overflow;
                const uint32_t next_offset =
                    layer_window_count > 0u
                        ? ((start_offset + layer_wave_count) % layer_window_count)
                        : 0u;
                wave_state->next_start_layer =
                    (layer_window_start + next_offset) % config.num_layers;
            }
            if (command_count > 0u &&
                controller_state &&
                rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                command_wave_index < controller_state->wave_count)
            {
                auto &wave = controller_state->waves[command_wave_index];
                wave.magic = kDeviceMoERebalanceMagic;
                wave.version = kDeviceMoERebalanceVersion;
                wave.epoch = shared_plan_epoch;
                wave.state = kDeviceMoERebalanceLifecyclePlanning;
                wave.planned_start_layer =
                    wave_state ? wave_state->planned_start_layer : planned_start_layer;
                wave.planned_layer_count =
                    wave_state ? wave_state->planned_layer_count : planned_layer_count;
                wave.command_count = command_count;
                wave.copied_arrivals = requested_payload_slots > 0u ? 0u : command_count;
                wave.applied_arrivals = 0u;
                wave.applied_layer_count = 0u;
                wave.error_code = 0u;
                wave.requested_payload_slots = requested_payload_slots;
                wave.payload_bucket_slots = payload_bucket_slots;
                wave.payload_bucket_index = payload_bucket_index;
                wave.payload_bucket_overflow = payload_bucket_overflow;
                __threadfence();
                wave.state = requested_payload_slots > 0u
                                 ? kDeviceMoERebalanceLifecyclePlanning
                                 : kDeviceMoERebalanceLifecycleReadyToApply;
                controller_state->next_epoch = shared_plan_epoch + 1u;
            }
            else if (command_count == 0u &&
                     controller_state &&
                     rebalance_graph_controller_state_basic_ok(controller_state, config) &&
                     command_wave_index < controller_state->wave_count)
            {
                reset_rebalance_wave_progress_device(controller_state->waves[command_wave_index]);
            }
        }
    }

    __global__ void pack_rebalance_histograms_kernel(
        const DeviceMoELayerRuntimeView *runtime_layers,
        unsigned long long *local_histograms,
        DeviceMoERebalanceConfigView config,
        const DeviceMoERebalanceWaveStateView *wave_state,
        const DeviceMoERebalanceGraphControllerStateView *controller_state,
        uint32_t command_buffer_count,
        uint32_t histogram_source_mask,
        unsigned long long *previous_activation_counts)
    {
        if (!runtime_layers || !local_histograms || !rebalance_config_ok(config))
            return;

        const uint32_t command_wave_index =
            rebalance_active_command_wave_index(controller_state, config, command_buffer_count);
        const DeviceMoERebalanceWaveStateView *selected_wave_state =
            wave_state ? wave_state + command_wave_index : nullptr;
        const uint32_t packed_layers = rebalance_layer_wave_count(config);
        const uint32_t total =
            packed_layers * config.num_experts;
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < total)
        {
            const uint32_t wave_layer = idx / config.num_experts;
            const uint32_t expert = idx - wave_layer * config.num_experts;
            const uint32_t layer =
                rebalance_actual_layer_for_wave_index(config, selected_wave_state, wave_layer);
            const DeviceMoELayerRuntimeView &runtime = runtime_layers[layer];
            unsigned long long value = 0ULL;
            if (runtime.active_bank <= 1u &&
                runtime.expert_count == config.num_experts &&
                runtime.top_k == config.top_k &&
                runtime.participant_id == config.participant_id &&
                runtime.participant_count == config.participant_count)
            {
                const DeviceMoEPlacementBankView &bank =
                    runtime.banks[runtime.active_bank];
                constexpr uint32_t kAppliedResidentFlags =
                    kDeviceMoEFlagValid |
                    kDeviceMoEFlagResident;
                const uint32_t descriptor_flags =
                    bank.experts[expert].flags;
                const bool physically_resident =
                    (bank.resident_participant_mask[expert] &
                     runtime_participant_bit(
                         static_cast<int>(config.participant_id))) != 0u &&
                    (descriptor_flags & kAppliedResidentFlags) ==
                        kAppliedResidentFlags;
                const bool transfer_backed =
                    physically_resident &&
                    (descriptor_flags & kDeviceMoEFlagTransferSlot) != 0u;
                const bool authoritative_owner =
                    bank.experts[expert].owner_participant ==
                    static_cast<int32_t>(config.participant_id);
                const uint32_t active_transfer_slots =
                    idx == 0u
                        ? rebalance_active_transfer_slot_expert_count(
                              runtime_layers, config)
                        : 0u;
                unsigned long long cumulative_count = 0ULL;
                for (uint32_t source = 0u;
                     source < llaminar2::moe_runtime_abi::kHistogramSourceCount;
                     ++source)
                {
                    if ((histogram_source_mask & (1u << source)) == 0u)
                        continue;
                    auto *const counter = reinterpret_cast<unsigned long long *>(
                        &runtime_local_histogram(
                            const_cast<DeviceMoELayerRuntimeView &>(runtime),
                            source)[expert]);
                    // Match inference's 64-bit atomic writer so a concurrent
                    // retained graph cannot expose a torn cumulative counter.
                    cumulative_count += atomicAdd(counter, 0ULL);
                }
                unsigned long long window_count = cumulative_count;
                if (previous_activation_counts)
                {
                    const uint32_t baseline_index =
                        layer * config.num_experts + expert;
                    const unsigned long long previous =
                        previous_activation_counts[baseline_index];
                    window_count = cumulative_count >= previous
                                       ? cumulative_count - previous
                                       : cumulative_count;
                    previous_activation_counts[baseline_index] =
                        cumulative_count;
                }
                value =
                    llaminar2::moe_rebalance_policy::packCollectedState(
                        window_count,
                        active_transfer_slots,
                        physically_resident,
                        transfer_backed,
                        authoritative_owner);
            }
            local_histograms[idx] = value;
            idx += blockDim.x * gridDim.x;
        }
    }

} // namespace

extern "C"
{
bool hipMoE_device_rebalance_controller(
    void *runtime_layers,
    const unsigned long long *gathered_histograms,
    void *status,
    const void *config,
    void *plan_entries,
    uint32_t *plan_count,
    uint32_t plan_capacity,
    uint32_t payload_slot_capacity,
    void *command_header,
    void *wave_state,
    void *controller_state,
    void *llep_layer_plans,
    void *placement_plan_scratch,
    uint32_t command_buffer_count,
    const void *local_transfer_slots,
    uint32_t local_transfer_slot_count,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !gathered_histograms || !status || !config ||
        !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    if ((cfg.flags & kDeviceMoERebalanceFlagDeferRuntimeApply) != 0u &&
        !placement_plan_scratch)
    {
        return false;
    }
    const auto finish_launch = [](const char *kernel_name)
    {
        const hipError_t error = hipGetLastError();
        if (error == hipSuccess)
            return true;
        fprintf(stderr,
                "HIP MoE %s kernel launch failed: %s\n",
                kernel_name,
                hipGetErrorString(error));
        return false;
    };
    const bool dynamic_ownership_fast_path =
        cfg.routed_assignment_policy != kDeviceMoERebalanceAssignmentLeastLoadedResident &&
        (cfg.flags & kDeviceMoERebalanceFlagHotReplicaCache) == 0u &&
        (cfg.flags & kDeviceMoERebalanceFlagPlanMissingArrivals) != 0u &&
        (cfg.flags & kDeviceMoERebalanceFlagDeferRuntimeApply) != 0u;
    if (dynamic_ownership_fast_path)
    {
        hipLaunchKernelGGL(device_rebalance_dynamic_ownership_controller_kernel,
            dim3(1), dim3(kDeviceMoEMaxExperts), 0, static_cast<hipStream_t>(stream),
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            gathered_histograms,
            static_cast<DeviceMoERebalanceStatusView *>(status),
            cfg,
            static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            command_buffer_count,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(
                local_transfer_slots),
            local_transfer_slot_count);

        return finish_launch(
            "device_rebalance_dynamic_ownership_controller");
    }

    if (cfg.routed_assignment_policy ==
        kDeviceMoERebalanceAssignmentLeastLoadedResident)
    {
        if (!llep_layer_plans)
            return false;
        const uint32_t layer_window_count =
            cfg.layer_window_count == 0u
                ? cfg.num_layers
                : min(cfg.layer_window_count, cfg.num_layers);
        const uint32_t layer_wave_count =
            cfg.layer_wave_count == 0u
                ? layer_window_count
                : min(cfg.layer_wave_count, layer_window_count);

        hipLaunchKernelGGL(device_rebalance_llep_claim_preflight_kernel,
            dim3(1), dim3(kDeviceMoEMaxExperts), 0,
            static_cast<hipStream_t>(stream),
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            cfg,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(
                local_transfer_slots),
            local_transfer_slot_count,
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(
                controller_state),
            static_cast<DeviceMoELLEPLayerPlanScratchView *>(
                llep_layer_plans));
        if (!finish_launch("device_rebalance_llep_claim_preflight"))
            return false;

        hipLaunchKernelGGL(device_rebalance_llep_router_collect_kernel,
            dim3(layer_window_count), dim3(kDeviceMoEMaxExperts), 0,
            static_cast<hipStream_t>(stream),
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            cfg,
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(
                controller_state),
            static_cast<DeviceMoELLEPLayerPlanScratchView *>(
                llep_layer_plans));
        if (!finish_launch("device_rebalance_llep_router_collect"))
            return false;

        hipLaunchKernelGGL(device_rebalance_llep_router_reduce_kernel,
            dim3(1), dim3(kDeviceMoEMaxExperts), 0,
            static_cast<hipStream_t>(stream),
            cfg,
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(
                controller_state),
            static_cast<DeviceMoELLEPLayerPlanScratchView *>(
                llep_layer_plans));
        if (!finish_launch("device_rebalance_llep_router_reduce"))
            return false;

        hipLaunchKernelGGL(device_rebalance_llep_preplan_kernel,
            dim3(layer_wave_count), dim3(kDeviceMoEMaxExperts), 0,
            static_cast<hipStream_t>(stream),
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            gathered_histograms,
            cfg,
            plan_capacity,
            payload_slot_capacity,
            static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<const DeviceMoERebalanceGraphControllerStateView *>(
                controller_state),
            command_buffer_count,
            static_cast<DeviceMoELLEPLayerPlanScratchView *>(
                llep_layer_plans));
        if (!finish_launch("device_rebalance_llep_preplan"))
            return false;

        hipLaunchKernelGGL((device_rebalance_controller_kernel<true>),
            dim3(1), dim3(256), 0,
            static_cast<hipStream_t>(stream),
            static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
            gathered_histograms,
            static_cast<DeviceMoERebalanceStatusView *>(status),
            cfg,
            static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(
                command_header),
            static_cast<DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(
                controller_state),
            static_cast<const DeviceMoELLEPLayerPlanScratchView *>(
                llep_layer_plans),
            static_cast<DeviceMoEPlacementBankView *>(
                placement_plan_scratch),
            command_buffer_count,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(
                local_transfer_slots),
            local_transfer_slot_count);
        return finish_launch("device_rebalance_llep_controller");
    }

    hipLaunchKernelGGL((device_rebalance_controller_kernel<false>),
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),
        gathered_histograms,
        static_cast<DeviceMoERebalanceStatusView *>(status),
        cfg,
        static_cast<DeviceMoERebalancePlanEntryView *>(plan_entries),
        plan_count,
            plan_capacity,
            payload_slot_capacity,
            static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(command_header),
            static_cast<DeviceMoERebalanceWaveStateView *>(wave_state),
            static_cast<DeviceMoERebalanceGraphControllerStateView *>(controller_state),
            nullptr,
            static_cast<DeviceMoEPlacementBankView *>(
                placement_plan_scratch),
            command_buffer_count,
            static_cast<const DeviceMoEExpertDirectoryEntryView *>(
                local_transfer_slots),
            local_transfer_slot_count);

    return finish_launch("device_rebalance_controller");
}

bool hipMoE_pack_rebalance_histograms(
    void *runtime_layers,
    unsigned long long *local_histograms,
    const void *config,
    const void *wave_state,
    const void *controller_state,
    uint32_t command_buffer_count,
    uint32_t histogram_source_mask,
    unsigned long long *previous_activation_counts,
    int device_idx,
    void *stream)
{
    if (!runtime_layers || !local_histograms || !config || !stream ||
        !llaminar2::moe_runtime_abi::validHistogramSourceMask(
            histogram_source_mask))
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const auto cfg = *static_cast<const DeviceMoERebalanceConfigView *>(config);
    const uint32_t window_count =
        cfg.layer_window_count == 0u
            ? cfg.num_layers
            : ((cfg.layer_window_count < cfg.num_layers)
                   ? cfg.layer_window_count
                   : cfg.num_layers);
    const uint32_t wave_count =
        cfg.layer_wave_count == 0u
            ? window_count
            : ((cfg.layer_wave_count < window_count)
                   ? cfg.layer_wave_count
                   : window_count);
    const int total = static_cast<int>(wave_count * cfg.num_experts);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(pack_rebalance_histograms_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
        local_histograms,
        cfg,
        static_cast<const DeviceMoERebalanceWaveStateView *>(wave_state),
        static_cast<const DeviceMoERebalanceGraphControllerStateView *>(controller_state),
        command_buffer_count,
        histogram_source_mask,
        previous_activation_counts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE pack_rebalance_histograms kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
