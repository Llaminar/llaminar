/**
 * @file DeviceGraphOrchestrator.cpp
 * @brief DeviceGraphOrchestrator core lifecycle, rendezvous, trace logger, and initialization.
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{
    namespace dgo_internal
    {
        bool appendDeviceMoERebalanceTraceJsonl(
            const std::string &path,
            const std::string &device_key,
            const PerfStatsCollector::Tags &maintenance_tags,
            const std::string &workspace_name,
            const MoEDeviceRebalanceStage::Params &params,
            const DeviceMoERebalanceStatus &status,
            const DeviceMoERebalanceGraphControllerState &controller_state,
            const DeviceMoERebalanceApplyStatus &copy_status,
            bool valid_copy_status,
            const DeviceMoERebalanceApplyStatus &apply_status,
            const std::vector<DeviceMoERebalanceCommandBufferHeader> &command_headers,
            const std::vector<DeviceMoERebalanceWaveState> &wave_states,
            const std::vector<uint32_t> &plan_counts,
            const std::vector<DeviceMoERebalancePlanEntry> &plan_entries,
            uint32_t plan_capacity,
            const DeviceMoERebalanceTransferCostEstimate &transfer_cost,
            uint32_t transfer_copied_arrivals,
            uint32_t transfer_applied_arrivals,
            uint32_t transfer_payload_bucket_slots,
            uint64_t transfer_payload_edge_mask,
            const std::vector<uint64_t> &trace_gathered_histograms,
            uint32_t trace_histogram_layer_count,
            const std::vector<int32_t> &trace_owner_participants,
            const DeviceMoERuntimeTable *trace_runtime_table)
        {
            static std::mutex trace_mutex;
            std::ostringstream out;
            out << '{';
            bool first = true;
            appendJsonStringField(out, first, "schema", "llaminar.moe_rebalance.trace.v1");
            appendJsonStringField(out, first, "event", "maintenance_export");
            appendJsonStringField(out, first, "device", device_key);
            appendJsonStringField(out, first, "workspace", workspace_name);
            appendJsonStringField(out, first, "stage", params.stage_name);
            appendJsonNumberField(out, first, "transfer_mode", static_cast<uint32_t>(params.transfer_mode));
            appendJsonNumberField(out, first, "phase", static_cast<uint32_t>(params.phase));
            appendJsonNumberField(out, first, "local_transfer_slot_count", params.local_transfer_slot_count);
            appendJsonNumberField(out, first, "collective_payload_slot_capacity", params.collective_payload_slot_capacity);
            appendJsonNumberField(out, first, "collective_payload_slot_bytes", params.collective_payload_slot_bytes);

            if (!first)
                out << ',';
            first = false;
            out << "\"tags\":";
            appendJsonTagsObject(out, maintenance_tags);

            if (!first)
                out << ',';
            first = false;
            out << "\"config\":{";
            bool config_first = true;
            appendJsonNumberField(out, config_first, "num_layers", params.config.num_layers);
            appendJsonNumberField(out, config_first, "num_experts", params.config.num_experts);
            appendJsonNumberField(out, config_first, "top_k", params.config.top_k);
            appendJsonNumberField(out, config_first, "participant_id", params.config.participant_id);
            appendJsonNumberField(out, config_first, "participant_count", params.config.participant_count);
            appendJsonNumberField(out, config_first, "root_participant", params.config.root_participant);
            appendJsonNumberField(out, config_first, "window_size_tokens", params.config.window_size_tokens);
            appendJsonNumberField(out, config_first, "max_hot_replicas_per_participant", params.config.max_hot_replicas_per_participant);
            appendJsonNumberField(out, config_first, "layer_window_start", params.config.layer_window_start);
            appendJsonNumberField(out, config_first, "layer_window_count", params.config.layer_window_count);
            appendJsonNumberField(out, config_first, "layer_wave_count", params.config.layer_wave_count);
            appendJsonNumberField(out, config_first, "flags", params.config.flags);
            appendJsonNumberField(out, config_first, "min_load_spread_improvement", params.config.min_load_spread_improvement);
            appendJsonNumberField(out, config_first, "min_load_spread_improvement_divisor", params.config.min_load_spread_improvement_divisor);
            appendJsonNumberField(out, config_first, "min_wave_spread_improvement_per_payload_slot", params.config.min_wave_spread_improvement_per_payload_slot);
            appendJsonNumberField(
                out,
                config_first,
                "min_foreign_rows_per_critical_path_payload_slot",
                params.config
                    .min_foreign_rows_per_critical_path_payload_slot);
            appendJsonNumberField(out, config_first, "min_router_spread_improvement_per_payload_slot", params.config.min_router_spread_improvement_per_payload_slot);
            appendJsonNumberField(out, config_first, "max_post_wave_load_spread_per_mille", params.config.max_post_wave_load_spread_per_mille);
            out << '}';

            if (!first)
                out << ',';
            first = false;
            out << "\"status\":{";
            bool status_first = true;
            appendJsonNumberField(out, status_first, "status_code", status.status_code);
            appendJsonNumberField(out, status_first, "last_epoch", status.last_epoch);
            appendJsonNumberField(out, status_first, "windows_observed", status.windows_observed);
            appendJsonNumberField(out, status_first, "windows_applied", status.windows_applied);
            appendJsonNumberField(out, status_first, "changed_layers", status.changed_layers);
            appendJsonNumberField(out, status_first, "selected_replicas", status.selected_replicas);
            appendJsonNumberField(out, status_first, "planned_arrivals", status.planned_arrivals);
            appendJsonNumberField(out, status_first, "skipped_no_resident", status.skipped_no_resident);
            appendJsonNumberField(out, status_first, "skipped_no_improvement", status.skipped_no_improvement);
            appendJsonNumberField(out, status_first, "dynamic_ownership_swap_attempts", status.dynamic_ownership_swap_attempts);
            appendJsonNumberField(out, status_first, "dynamic_ownership_swap_accepts", status.dynamic_ownership_swap_accepts);
            appendJsonNumberField(out, status_first, "dynamic_ownership_swap_rejections", status.dynamic_ownership_swap_rejections);
            appendJsonNumberField(out, status_first, "candidate_arrivals_considered", status.candidate_arrivals_considered);
            appendJsonNumberField(out, status_first, "candidate_arrivals_below_floor", status.candidate_arrivals_below_floor);
            appendJsonNumberField(out, status_first, "candidate_arrivals_pruned_by_count_bound", status.candidate_arrivals_pruned_by_count_bound);
            appendJsonNumberField(out, status_first, "llep_assignment_span_count", status.llep_assignment_span_count);
            appendJsonNumberField(out, status_first, "llep_weight_transfer_count", status.llep_weight_transfer_count);
            appendJsonNumberField(out, status_first, "llep_standard_ep_selected", status.llep_standard_ep_selected);
            appendJsonNumberField(out, status_first, "llep_skipped_balanced", status.llep_skipped_balanced);
            appendJsonNumberField(out, status_first, "llep_skipped_insufficient_spread_improvement", status.llep_skipped_insufficient_spread_improvement);
            appendJsonNumberField(out, status_first, "llep_skipped_insufficient_foreign_rows", status.llep_skipped_insufficient_foreign_rows);
            appendJsonNumberField(out, status_first, "llep_min_chunk_skips", status.llep_min_chunk_skips);
            appendJsonNumberField(out, status_first, "llep_forced_spills", status.llep_forced_spills);
            appendJsonNumberField(out, status_first, "candidate_load_spread_improvement_total", status.candidate_load_spread_improvement_total);
            appendJsonNumberField(out, status_first, "candidate_load_spread_improvement_max", status.candidate_load_spread_improvement_max);
            appendJsonNumberField(out, status_first, "accepted_load_spread_improvement_total", status.accepted_load_spread_improvement_total);
            appendJsonNumberField(out, status_first, "accepted_load_spread_improvement_max", status.accepted_load_spread_improvement_max);
            appendJsonNumberField(out, status_first, "llep_native_rows", status.llep_native_rows);
            appendJsonNumberField(out, status_first, "llep_spilled_rows", status.llep_spilled_rows);
            appendJsonNumberField(out, status_first, "llep_required_spread_improvement", status.llep_required_spread_improvement);
            appendJsonNumberField(out, status_first, "llep_required_foreign_rows", status.llep_required_foreign_rows);
            appendJsonNumberField(out, status_first, "router_hot_cache_eligible_dispatches", status.router_hot_cache_eligible_dispatches);
            appendJsonNumberField(out, status_first, "router_hot_cache_used_dispatches", status.router_hot_cache_used_dispatches);
            appendJsonNumberField(out, status_first, "router_hot_cache_improved_dispatches", status.router_hot_cache_improved_dispatches);
            appendJsonNumberField(out, status_first, "router_hot_cache_default_load_spread_total", status.router_hot_cache_default_load_spread_total);
            appendJsonNumberField(out, status_first, "router_hot_cache_actual_load_spread_total", status.router_hot_cache_actual_load_spread_total);
            appendJsonNumberField(out, status_first, "router_hot_cache_load_spread_improvement_total", status.router_hot_cache_load_spread_improvement_total);
            appendJsonNumberField(out, status_first, "router_hot_cache_active_dispatches", status.router_hot_cache_active_dispatches);
            appendJsonNumberField(out, status_first, "router_hot_cache_miss_dispatches", status.router_hot_cache_miss_dispatches);
            appendJsonNumberField(out, status_first, "router_hot_cache_selected_expert_slots", status.router_hot_cache_selected_expert_slots);
            appendJsonNumberField(out, status_first, "router_hot_cache_replicated_selected_expert_slots", status.router_hot_cache_replicated_selected_expert_slots);
            appendJsonNumberField(out, status_first, "pre_policy_load_total", status.pre_policy_load_total);
            appendJsonNumberField(out, status_first, "pre_policy_load_min", status.pre_policy_load_min);
            appendJsonNumberField(out, status_first, "pre_policy_load_max", status.pre_policy_load_max);
            appendJsonNumberField(out, status_first, "post_policy_load_total", status.post_policy_load_total);
            appendJsonNumberField(out, status_first, "post_policy_load_min", status.post_policy_load_min);
            appendJsonNumberField(out, status_first, "post_policy_load_max", status.post_policy_load_max);
            appendJsonNumberField(out, status_first, "pre_wave_load_total", status.pre_wave_load_total);
            appendJsonNumberField(out, status_first, "pre_wave_load_spread", status.pre_wave_load_spread);
            appendJsonNumberField(out, status_first, "post_wave_load_total", status.post_wave_load_total);
            appendJsonNumberField(out, status_first, "post_wave_load_spread", status.post_wave_load_spread);
            appendJsonNumberField(out, status_first, "window_ready_slots", status.window_ready_slots);
            appendJsonNumberField(out, status_first, "window_required_slots", status.window_required_slots);
            appendJsonNumberField(out, status_first, "payload_bucket_requested_slots", status.payload_bucket_requested_slots);
            appendJsonNumberField(out, status_first, "payload_bucket_slots", status.payload_bucket_slots);
            appendJsonNumberField(out, status_first, "prefill_current_batch_movement_layers", status.prefill_current_batch_movement_layers);
            appendJsonNumberField(out, status_first, "prefill_current_batch_non_owner_assignment_layers", status.prefill_current_batch_non_owner_assignment_layers);
            appendJsonNumberField(out, status_first, "skipped_wave_cost_floor", status.skipped_wave_cost_floor);
            appendJsonNumberField(out, status_first, "skipped_low_router_benefit", status.skipped_low_router_benefit);
            appendJsonNumberField(out, status_first, "skipped_post_load_spread_ceiling", status.skipped_post_load_spread_ceiling);
            appendJsonNumberField(out, status_first, "skipped_participant_load_spread", status.skipped_participant_load_spread);
            appendJsonNumberField(out, status_first, "skipped_aggregate_load_spread", status.skipped_aggregate_load_spread);
            appendJsonNumberField(out, status_first, "skipped_configured_load_spread_ceiling", status.skipped_configured_load_spread_ceiling);
            appendJsonNumberField(out, status_first, "payload_edge_mask", status.payload_edge_mask);
            out << '}';

            auto append_load_array = [&](const char *name, const uint64_t *values)
            {
                if (!first)
                    out << ',';
                first = false;
                out << '"' << name << "\":[";
                const uint32_t participant_count =
                    std::min<uint32_t>(params.config.participant_count, kDeviceMoEMaxParticipants);
                for (uint32_t participant = 0; participant < participant_count; ++participant)
                {
                    if (participant != 0)
                        out << ',';
                    out << values[participant];
                }
                out << ']';
            };
            append_load_array("pre_policy_participant_load", status.pre_policy_participant_load);
            append_load_array("post_policy_participant_load", status.post_policy_participant_load);

            if (!trace_gathered_histograms.empty() &&
                trace_histogram_layer_count > 0u &&
                params.config.num_experts > 0u &&
                params.config.participant_count > 0u)
            {
                const uint32_t participant_count =
                    std::min<uint32_t>(params.config.participant_count, kDeviceMoEMaxParticipants);
                const uint32_t expert_count =
                    std::min<uint32_t>(params.config.num_experts, kDeviceMoEMaxExperts);
                const size_t layer_stride = static_cast<size_t>(expert_count);
                const size_t participant_stride =
                    static_cast<size_t>(trace_histogram_layer_count) * layer_stride;

                if (!first)
                    out << ',';
                first = false;
                out << "\"gathered_histogram\":{";
                bool histogram_first = true;
                appendJsonStringField(out,
                                      histogram_first,
                                      "layout",
                                      "participant,wave_layer,expert");
                appendJsonNumberField(out,
                                      histogram_first,
                                      "participant_count",
                                      participant_count);
                appendJsonNumberField(out,
                                      histogram_first,
                                      "wave_layer_count",
                                      trace_histogram_layer_count);
                appendJsonNumberField(out,
                                      histogram_first,
                                      "num_experts",
                                      expert_count);
                appendJsonStringField(
                    out,
                    histogram_first,
                    "word_encoding",
                    "activation_count_with_residency_metadata");

                if (!histogram_first)
                    out << ',';
                histogram_first = false;
                out << "\"nonzero\":[";
                bool first_histogram_entry = true;
                for (uint32_t participant = 0; participant < participant_count; ++participant)
                {
                    for (uint32_t wave_layer = 0; wave_layer < trace_histogram_layer_count; ++wave_layer)
                    {
                        for (uint32_t expert = 0; expert < expert_count; ++expert)
                        {
                            const size_t idx =
                                static_cast<size_t>(participant) * participant_stride +
                                static_cast<size_t>(wave_layer) * layer_stride +
                                static_cast<size_t>(expert);
                            if (idx >= trace_gathered_histograms.size())
                                continue;
                            const uint64_t packed_state =
                                trace_gathered_histograms[idx];
                            if (packed_state == 0u)
                                continue;
                            /*
                             * The collective word is not a raw histogram: its
                             * high bits carry physical residency and transfer-
                             * slot evidence. Exporting that word as `count`
                             * made ordinary resident experts appear to have
                             * 2^63 activations and poisoned corpus economics.
                             * Preserve the raw word for forensic replay, but
                             * expose each semantic component independently so
                             * CSV/JSON consumers never have to reverse-engineer
                             * the device ABI.
                             */
                            const uint64_t count =
                                moe_rebalance_policy::
                                    collectedStateActivationCount(
                                        packed_state);
                            if (!first_histogram_entry)
                                out << ',';
                            first_histogram_entry = false;
                            out << '{';
                            bool entry_first = true;
                            appendJsonNumberField(out, entry_first, "participant", participant);
                            appendJsonNumberField(out, entry_first, "wave_layer", wave_layer);
                            appendJsonNumberField(out, entry_first, "expert", expert);
                            appendJsonNumberField(out, entry_first, "count", count);
                            appendJsonNumberField(
                                out,
                                entry_first,
                                "active_transfer_slots",
                                moe_rebalance_policy::
                                    collectedStateActiveTransferSlots(
                                        packed_state));
                            appendJsonNumberField(
                                out,
                                entry_first,
                                "physically_resident",
                                moe_rebalance_policy::
                                        collectedStatePhysicallyResident(
                                            packed_state)
                                    ? 1
                                    : 0);
                            appendJsonNumberField(
                                out,
                                entry_first,
                                "transfer_backed",
                                moe_rebalance_policy::
                                        collectedStateTransferBacked(
                                            packed_state)
                                    ? 1
                                    : 0);
                            appendJsonNumberField(
                                out,
                                entry_first,
                                "packed_state",
                                packed_state);
                            out << '}';
                        }
                    }
                }
                out << "],\"global_expert_loads\":[";
                bool first_global_entry = true;
                for (uint32_t wave_layer = 0; wave_layer < trace_histogram_layer_count; ++wave_layer)
                {
                    for (uint32_t expert = 0; expert < expert_count; ++expert)
                    {
                        uint64_t count = 0;
                        for (uint32_t participant = 0; participant < participant_count; ++participant)
                        {
                            const size_t idx =
                                static_cast<size_t>(participant) * participant_stride +
                                static_cast<size_t>(wave_layer) * layer_stride +
                                static_cast<size_t>(expert);
                            if (idx < trace_gathered_histograms.size())
                            {
                                count +=
                                    moe_rebalance_policy::
                                        collectedStateActivationCount(
                                            trace_gathered_histograms[idx]);
                            }
                        }
                        if (count == 0u)
                            continue;
                        if (!first_global_entry)
                            out << ',';
                        first_global_entry = false;
                        out << '{';
                        bool entry_first = true;
                        appendJsonNumberField(out, entry_first, "wave_layer", wave_layer);
                        appendJsonNumberField(out, entry_first, "expert", expert);
                        appendJsonNumberField(out, entry_first, "count", count);
                        out << '}';
                    }
                }
                out << "]}";
            }

            if (!trace_owner_participants.empty() &&
                trace_histogram_layer_count > 0u &&
                params.config.num_experts > 0u)
            {
                const uint32_t expert_count =
                    std::min<uint32_t>(params.config.num_experts, kDeviceMoEMaxExperts);
                if (!first)
                    out << ',';
                first = false;
                out << "\"expert_owner_participants\":[";
                for (uint32_t wave_layer = 0; wave_layer < trace_histogram_layer_count; ++wave_layer)
                {
                    if (wave_layer != 0u)
                        out << ',';
                    out << "{\"wave_layer\":" << wave_layer << ",\"owners\":[";
                    for (uint32_t expert = 0; expert < expert_count; ++expert)
                    {
                        if (expert != 0u)
                            out << ',';
                        const size_t idx =
                            static_cast<size_t>(wave_layer) * static_cast<size_t>(expert_count) +
                            static_cast<size_t>(expert);
                        const int32_t owner =
                            idx < trace_owner_participants.size()
                                ? trace_owner_participants[idx]
                                : -1;
                        out << owner;
                    }
                    out << "]}";
                }
                out << ']';
            }

            if (!first)
                out << ',';
            first = false;
            out << "\"controller\":{";
            bool controller_first = true;
            appendJsonNumberField(out, controller_first, "participant_id", controller_state.participant_id);
            appendJsonNumberField(out, controller_first, "participant_count", controller_state.participant_count);
            appendJsonNumberField(out, controller_first, "next_epoch", controller_state.next_epoch);
            appendJsonNumberField(out, controller_first, "active_wave", controller_state.active_wave);
            appendJsonNumberField(out, controller_first, "wave_count", controller_state.wave_count);
            appendJsonNumberField(out, controller_first, "maintenance_launches", controller_state.maintenance_launches);
            appendJsonNumberField(out, controller_first, "decode_apply_polls", controller_state.decode_apply_polls);
            appendJsonNumberField(out, controller_first, "decode_apply_hits", controller_state.decode_apply_hits);
            appendJsonNumberField(out, controller_first, "last_error_code", controller_state.last_error_code);
            appendJsonNumberField(out, controller_first, "last_error_wave_index", controller_state.last_error_wave_index);
            appendJsonNumberField(out, controller_first, "last_error_epoch", controller_state.last_error_epoch);
            appendJsonNumberField(out, controller_first, "last_error_expected_arrivals", controller_state.last_error_expected_arrivals);
            appendJsonNumberField(out, controller_first, "last_error_copied_arrivals", controller_state.last_error_copied_arrivals);
            appendJsonNumberField(out, controller_first, "last_error_copy_status_code", controller_state.last_error_copy_status_code);
            appendJsonNumberField(out,
                                  controller_first,
                                  "last_error_copy_failure_flags",
                                  controller_state.last_error_copy_failure_flags);
            appendJsonNumberField(out,
                                  controller_first,
                                  "last_error_copy_plan_entries_seen",
                                  controller_state.last_error_copy_plan_entries_seen);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_copy_skipped_wrong_destination",
                controller_state
                    .last_error_copy_skipped_wrong_destination);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_missing_destination_slot",
                controller_state.last_error_missing_destination_slot);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_missing_destination_layer",
                controller_state.last_error_missing_destination_layer);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_missing_destination_expert",
                controller_state.last_error_missing_destination_expert);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_missing_destination_source",
                controller_state.last_error_missing_destination_source);
            appendJsonNumberField(
                out,
                controller_first,
                "last_error_local_transfer_slot_count",
                controller_state.last_error_local_transfer_slot_count);
            const uint32_t controller_error_participant =
                controller_state.last_error_wave_index < 2u
                    ? controller_state
                          .waves[controller_state.last_error_wave_index]
                          .error_participant
                    : kDeviceMoEInvalidSlot;
            appendJsonNumberField(out, controller_first, "last_error_participant", controller_error_participant);
            out << '}';

            if (!first)
                out << ',';
            first = false;
            out << "\"transfer\":{";
            bool transfer_first = true;
            appendJsonNumberField(out, transfer_first, "copied_arrivals", transfer_copied_arrivals);
            appendJsonNumberField(out, transfer_first, "applied_arrivals", transfer_applied_arrivals);
            appendJsonNumberField(out, transfer_first, "payload_bucket_slots", transfer_payload_bucket_slots);
            appendJsonNumberField(out, transfer_first, "payload_edge_mask", transfer_payload_edge_mask);
            appendJsonNumberField(out, transfer_first, "selected_payload_gathered_capacity_bytes", transfer_cost.selected_payload_gathered_capacity_bytes);
            appendJsonNumberField(out, transfer_first, "selected_payload_transport_capacity_bytes", transfer_cost.selected_payload_transport_capacity_bytes);
            appendJsonNumberField(out, transfer_first, "useful_payload_bytes", transfer_cost.useful_payload_bytes);
            appendJsonNumberField(out, transfer_first, "wasted_payload_capacity_bytes", transfer_cost.wasted_payload_capacity_bytes);
            out << '}';

            if (!first)
                out << ',';
            first = false;
            out << "\"copy_status\":{";
            bool copy_first = true;
            appendJsonNumberField(out, copy_first, "valid", valid_copy_status ? 1u : 0u);
            appendJsonNumberField(out, copy_first, "status_code", copy_status.status_code);
            appendJsonNumberField(out,
                                  copy_first,
                                  "transaction_wave_index",
                                  copy_status.transaction_wave_index);
            appendJsonNumberField(out,
                                  copy_first,
                                  "transaction_epoch",
                                  copy_status.transaction_epoch);
            appendJsonNumberField(out,
                                  copy_first,
                                  "transaction_command_count",
                                  copy_status.transaction_command_count);
            appendJsonNumberField(out, copy_first, "plan_entries_seen", copy_status.plan_entries_seen);
            appendJsonNumberField(out, copy_first, "applied_arrivals", copy_status.applied_arrivals);
            appendJsonNumberField(out, copy_first, "copied_arrivals", copy_status.copied_arrivals);
            appendJsonNumberField(out, copy_first, "copy_incomplete", copy_status.copy_incomplete);
            appendJsonNumberField(out,
                                  copy_first,
                                  "required_local_arrivals",
                                  copy_status.required_local_arrivals);
            appendJsonNumberField(out,
                                  copy_first,
                                  "ready_local_arrivals",
                                  copy_status.ready_local_arrivals);
            appendJsonNumberField(out,
                                  copy_first,
                                  "missing_source_descriptors",
                                  copy_status.missing_source_descriptors);
            appendJsonNumberField(out,
                                  copy_first,
                                  "missing_destination_slots",
                                  copy_status.missing_destination_slots);
            appendJsonNumberField(out,
                                  copy_first,
                                  "descriptor_mismatches",
                                  copy_status.descriptor_mismatches);
            appendJsonNumberField(out,
                                  copy_first,
                                  "invalid_plan_entries",
                                  copy_status.invalid_plan_entries);
            appendJsonNumberField(out,
                                  copy_first,
                                  "skipped_wrong_destination",
                                  copy_status.skipped_wrong_destination);
            appendJsonNumberField(out, copy_first, "changed_layers", copy_status.changed_layers);
            appendJsonNumberField(out,
                                  copy_first,
                                  "post_apply_multi_resident_experts",
                                  copy_status.post_apply_multi_resident_experts);
            out << '}';

            if (!first)
                out << ',';
            first = false;
            out << "\"apply_status\":{";
            bool apply_first = true;
            appendJsonNumberField(out, apply_first, "status_code", apply_status.status_code);
            appendJsonNumberField(out, apply_first, "plan_entries_seen", apply_status.plan_entries_seen);
            appendJsonNumberField(out, apply_first, "applied_arrivals", apply_status.applied_arrivals);
            appendJsonNumberField(out, apply_first, "copied_arrivals", apply_status.copied_arrivals);
            appendJsonNumberField(out, apply_first, "copy_incomplete", apply_status.copy_incomplete);
            appendJsonNumberField(out,
                                  apply_first,
                                  "required_local_arrivals",
                                  apply_status.required_local_arrivals);
            appendJsonNumberField(out,
                                  apply_first,
                                  "ready_local_arrivals",
                                  apply_status.ready_local_arrivals);
            appendJsonNumberField(out,
                                  apply_first,
                                  "missing_source_descriptors",
                                  apply_status.missing_source_descriptors);
            appendJsonNumberField(out,
                                  apply_first,
                                  "missing_destination_slots",
                                  apply_status.missing_destination_slots);
            appendJsonNumberField(out,
                                  apply_first,
                                  "descriptor_mismatches",
                                  apply_status.descriptor_mismatches);
            appendJsonNumberField(out,
                                  apply_first,
                                  "invalid_plan_entries",
                                  apply_status.invalid_plan_entries);
            appendJsonNumberField(out,
                                  apply_first,
                                  "skipped_wrong_destination",
                                  apply_status.skipped_wrong_destination);
            appendJsonNumberField(out, apply_first, "changed_layers", apply_status.changed_layers);
            appendJsonNumberField(out,
                                  apply_first,
                                  "post_apply_multi_resident_experts",
                                  apply_status.post_apply_multi_resident_experts);
            out << '}';

            if (!first)
                out << ',';
            first = false;
            out << "\"waves\":[";
            const uint32_t wave_count = std::min<uint32_t>(controller_state.wave_count, 2u);
            for (uint32_t i = 0; i < wave_count; ++i)
            {
                if (i != 0)
                    out << ',';
                const auto &wave = controller_state.waves[i];
                out << '{';
                bool wave_first = true;
                appendJsonNumberField(out, wave_first, "wave", i);
                appendJsonNumberField(out, wave_first, "epoch", wave.epoch);
                appendJsonNumberField(out, wave_first, "state", wave.state);
                appendJsonNumberField(out, wave_first, "planned_start_layer", wave.planned_start_layer);
                appendJsonNumberField(out, wave_first, "planned_layer_count", wave.planned_layer_count);
                appendJsonNumberField(out, wave_first, "command_count", wave.command_count);
                appendJsonNumberField(out, wave_first, "copied_arrivals", wave.copied_arrivals);
                appendJsonNumberField(out, wave_first, "applied_arrivals", wave.applied_arrivals);
                appendJsonNumberField(out, wave_first, "applied_layer_count", wave.applied_layer_count);
                appendJsonNumberField(out, wave_first, "requested_payload_slots", wave.requested_payload_slots);
                appendJsonNumberField(out, wave_first, "payload_bucket_slots", wave.payload_bucket_slots);
                appendJsonNumberField(out, wave_first, "error_code", wave.error_code);
                out << '}';
            }
            out << ']';

            if (!first)
                out << ',';
            first = false;
            out << "\"command_buffers\":[";
            for (uint32_t i = 0; i < command_headers.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << '{';
                bool buffer_first = true;
                appendJsonNumberField(out, buffer_first, "buffer_slot", i);
                appendJsonNumberField(out, buffer_first, "epoch", command_headers[i].epoch);
                appendJsonNumberField(out, buffer_first, "command_count", command_headers[i].command_count);
                appendJsonNumberField(out, buffer_first, "command_capacity", command_headers[i].command_capacity);
                appendJsonNumberField(out, buffer_first, "plan_count", i < plan_counts.size() ? plan_counts[i] : 0u);
                appendJsonNumberField(out, buffer_first, "wave_epoch", i < wave_states.size() ? wave_states[i].epoch : 0u);
                appendJsonNumberField(out, buffer_first, "wave_planned_start_layer", i < wave_states.size() ? wave_states[i].planned_start_layer : 0u);
                appendJsonNumberField(out, buffer_first, "wave_planned_layer_count", i < wave_states.size() ? wave_states[i].planned_layer_count : 0u);
                out << '}';
            }
            out << ']';

            if (!first)
                out << ',';
            first = false;
            out << "\"commands\":[";
            bool first_command = true;
            uint32_t noop_command_entries = 0;
            for (uint32_t buffer = 0; buffer < command_headers.size(); ++buffer)
            {
                const uint32_t header_count = command_headers[buffer].command_count;
                const uint32_t plan_count = buffer < plan_counts.size() ? plan_counts[buffer] : 0u;
                const uint32_t wave_count_for_buffer =
                    buffer < wave_count ? controller_state.waves[buffer].command_count : 0u;
                const uint32_t command_count =
                    std::min<uint32_t>(
                        std::max({header_count, plan_count, wave_count_for_buffer}),
                        plan_capacity);
                for (uint32_t entry_index = 0; entry_index < command_count; ++entry_index)
                {
                    const size_t flat_index =
                        static_cast<size_t>(buffer) * static_cast<size_t>(plan_capacity) +
                        static_cast<size_t>(entry_index);
                    if (flat_index >= plan_entries.size())
                        break;
                    const auto &entry = plan_entries[flat_index];
                    if (entry.op == 0u)
                    {
                        ++noop_command_entries;
                        continue;
                    }
                    if (!first_command)
                        out << ',';
                    first_command = false;
                    out << '{';
                    bool command_first = true;
                    appendJsonNumberField(out, command_first, "buffer_slot", buffer);
                    appendJsonNumberField(out, command_first, "entry_index", entry_index);
                    appendJsonNumberField(out, command_first, "op", entry.op);
                    appendJsonNumberField(out, command_first, "layer", entry.layer);
                    appendJsonNumberField(out, command_first, "expert", entry.expert);
                    appendJsonNumberField(out, command_first, "source_participant", entry.source_participant);
                    appendJsonNumberField(out, command_first, "destination_participant", entry.destination_participant);
                    appendJsonNumberField(out, command_first, "source_resident_mask", entry.source_resident_mask);
                    appendJsonNumberField(out, command_first, "flags", entry.flags);
                    appendJsonNumberField(out, command_first, "destination_slot", entry.destination_slot);
                    appendJsonNumberField(
                        out,
                        command_first,
                        "destination_previous_layer",
                        entry.destination_previous_layer);
                    appendJsonNumberField(
                        out,
                        command_first,
                        "destination_previous_expert",
                        entry.destination_previous_expert);
                    appendJsonNumberField(
                        out,
                        command_first,
                        "destination_generation",
                        entry.destination_generation);
                    appendJsonNumberField(out, command_first, "payload_slot", entry.payload_slot);
                    out << '}';
                }
            }
            out << ']';

            /*
             * A maintenance trace is an explicit diagnostic boundary at which
             * the completion event has already been consumed. Record both RCU
             * banks for only the logical experts named by the current command
             * family. This lets a developer distinguish a malformed candidate
             * publication from a later request-reset or prefix-restore rewrite
             * without turning the host mirror into an inference authority.
             */
            if (trace_runtime_table)
            {
                if (!first)
                    out << ',';
                first = false;
                out << "\"runtime_command_experts\":[";
                bool first_runtime_expert = true;
                for (uint32_t buffer = 0; buffer < command_headers.size(); ++buffer)
                {
                    const uint32_t header_count =
                        command_headers[buffer].command_count;
                    const uint32_t buffered_plan_count =
                        buffer < plan_counts.size() ? plan_counts[buffer] : 0u;
                    const uint32_t wave_count_for_buffer =
                        buffer < wave_count
                            ? controller_state.waves[buffer].command_count
                            : 0u;
                    const uint32_t command_count =
                        std::min<uint32_t>(
                            std::max({header_count,
                                      buffered_plan_count,
                                      wave_count_for_buffer}),
                            plan_capacity);
                    for (uint32_t entry_index = 0;
                         entry_index < command_count;
                         ++entry_index)
                    {
                        const size_t flat_index =
                            static_cast<size_t>(buffer) *
                                static_cast<size_t>(plan_capacity) +
                            static_cast<size_t>(entry_index);
                        if (flat_index >= plan_entries.size())
                            break;
                        const auto &entry = plan_entries[flat_index];
                        if (entry.op == 0u ||
                            entry.layer >= static_cast<uint32_t>(
                                               trace_runtime_table->layerCount()) ||
                            entry.expert >= static_cast<uint32_t>(
                                                trace_runtime_table->expertCount()))
                        {
                            continue;
                        }

                        const auto &runtime =
                            trace_runtime_table->hostLayerState(
                                static_cast<int>(entry.layer));
                        if (!first_runtime_expert)
                            out << ',';
                        first_runtime_expert = false;
                        out << '{';
                        bool runtime_first = true;
                        appendJsonNumberField(
                            out, runtime_first, "buffer_slot", buffer);
                        appendJsonNumberField(
                            out,
                            runtime_first,
                            "entry_index",
                            entry_index);
                        appendJsonNumberField(
                            out, runtime_first, "layer", entry.layer);
                        appendJsonNumberField(
                            out, runtime_first, "expert", entry.expert);
                        appendJsonNumberField(
                            out,
                            runtime_first,
                            "active_bank",
                            runtime.active_bank);
                        appendJsonNumberField(
                            out,
                            runtime_first,
                            "active_epoch",
                            runtime.active_epoch);
                        for (uint32_t bank_index = 0u;
                             bank_index < 2u;
                             ++bank_index)
                        {
                            const auto &bank = runtime.banks[bank_index];
                            const auto &descriptor =
                                bank.experts[entry.expert];
                            const std::string prefix =
                                "bank" + std::to_string(bank_index) + "_";
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "epoch").c_str(),
                                bank.epoch);
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "owner").c_str(),
                                descriptor.owner_participant);
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "local_slot").c_str(),
                                descriptor.local_slot);
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "flags").c_str(),
                                descriptor.flags);
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "local_compute").c_str(),
                                static_cast<unsigned int>(
                                    bank.local_compute_mask[entry.expert]));
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "resident_mask").c_str(),
                                bank.resident_participant_mask[entry.expert]);
                            appendJsonNumberField(
                                out,
                                runtime_first,
                                (prefix + "route_participant").c_str(),
                                bank.overlay_route_participant[entry.expert]);
                        }
                        out << '}';
                    }
                }
                out << ']';
            }
            appendJsonNumberField(out, first, "noop_command_entries", noop_command_entries);
            out << "}\n";

            std::lock_guard<std::mutex> lock(trace_mutex);
            std::ofstream file(path, std::ios::out | std::ios::app);
            if (!file)
            {
                LOG_ERROR("[DGO] Failed to open MoE rebalance trace JSONL at " << path);
                return false;
            }
            file << out.str();
            if (!file)
            {
                LOG_ERROR("[DGO] Failed to append MoE rebalance trace JSONL at " << path);
                return false;
            }
            return true;
        }
    } // namespace dgo_internal

    ForwardGraphExecutionRendezvous::ForwardGraphExecutionRendezvous(
        size_t expected_participants,
        std::string label)
        : expected_participants_(std::max<size_t>(1, expected_participants)),
          label_(std::move(label))
    {
    }

    bool ForwardGraphExecutionRendezvous::arriveAndWait(
        DeviceId device,
        int timeout_ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (failed_)
            return false;
        if (released_)
            return true;

        ++arrivals_;
        if (arrivals_ > expected_participants_)
        {
            failed_ = true;
            cv_.notify_all();
            LOG_ERROR("[ForwardGraphExecutionRendezvous] too many arrivals"
                      << " label=" << label_
                      << " expected=" << expected_participants_
                      << " device=" << device.toString());
            return false;
        }

        if (arrivals_ == expected_participants_)
        {
            released_ = true;
            cv_.notify_all();
            return true;
        }

        const auto predicate = [this]()
        {
            return released_ || failed_;
        };

        bool ok = true;
        if (timeout_ms > 0)
        {
            ok = cv_.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                predicate);
        }
        else
        {
            cv_.wait(lock, predicate);
        }

        if (!ok)
        {
            failed_ = true;
            cv_.notify_all();
            LOG_ERROR("[ForwardGraphExecutionRendezvous] timeout"
                      << " label=" << label_
                      << " expected=" << expected_participants_
                      << " arrivals=" << arrivals_
                      << " timeout_ms=" << timeout_ms
                      << " device=" << device.toString());
            return false;
        }

        return released_ && !failed_;
    }

    // =========================================================================
    // Shared Executor Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::configureExecutor()
    {
        GraphExecutorConfig exec_config = graph_builder_->config().executor_config;
        exec_config.default_device = graph_builder_->config().default_device;

        const auto &env = debugEnv();
        exec_config.enable_profiling = exec_config.enable_profiling || graph_builder_->config().enable_profiling || env.execution.executor_profiling;
        exec_config.enable_validation = exec_config.enable_validation || graph_builder_->config().enable_validation;
        exec_config.require_snapshot_memory_authority =
            memory_authority_mode_ ==
            MemoryAuthorityMode::ProductionRequired;

        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = DeviceGraphExecutor(exec_config);
    }

    bool DeviceGraphOrchestrator::validateConfigurationForForward() const
    {
        bool valid = true;

        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] graph_builder_ is null");
            valid = false;
        }

        if (!arena_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] BufferArena not initialized "
                      "(call initializeInferenceStateFromArena() first)");
            valid = false;
        }

        if (!managed_buffers_.current_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No current_hidden buffer "
                      "(call initializeInferenceStateFromArena() first)");
            valid = false;
        }

        // Weight check: either graph builder has weights or weight_manager is set
        bool has_weights = (graph_builder_ && graph_builder_->isInitialized()) ||
                           weight_manager_ != nullptr;
        if (!has_weights)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No weights loaded "
                      "(call setWeights() or setWeightManager())");
            valid = false;
        }

        return valid;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        Dependencies deps)
        : DeviceGraphOrchestrator(
              std::move(deps),
              MemoryAuthorityMode::ProductionRequired)
    {
    }

    std::unique_ptr<DeviceGraphOrchestrator>
    DeviceGraphOrchestrator::createForTest(Dependencies deps)
    {
        return std::unique_ptr<DeviceGraphOrchestrator>(
            new DeviceGraphOrchestrator(
                std::move(deps),
                MemoryAuthorityMode::IsolatedTest));
    }

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        Dependencies deps,
        MemoryAuthorityMode memory_authority_mode)
        : graph_builder_(std::move(deps.graph_builder)),
          mpi_ctx_(std::move(deps.mpi_ctx)),
          cache_config_(deps.cache_config),
          injected_model_ctx_(std::move(deps.model_ctx)),
          injected_topology_(std::move(deps.topology)),
          injected_collective_ctx_(std::move(deps.collective_ctx)),
          turboquant_ctx_(std::move(deps.turboquant_ctx)),
          kv_rotation_(std::move(deps.kv_rotation)),
          pp_stage_config_(std::move(deps.pp_stage_config)),
          domain_tp_contexts_(std::move(deps.domain_tp_contexts)),
          weight_streamer_(std::move(deps.weight_streamer)),
          weight_manager_(std::move(deps.weight_manager)),
          weight_placement_map_(std::move(deps.weight_placement_map)),
          tp_config_(std::move(deps.tp_config)),
          domain_config_(std::move(deps.domain_config)),
          reusable_execution_workspaces_(
              std::move(deps.reusable_execution_workspaces)),
          physical_memory_authority_(
              std::move(deps.physical_memory_authority)),
          memory_authority_mode_(memory_authority_mode)
    {
        if (!injected_model_ctx_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator Dependencies requires a valid model_ctx");
        }

        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator Dependencies requires a valid graph_builder");
        }

        // Graph builders can also be used by isolated graph-construction tests.
        // Production admission must reject their legacy multi-device graph form
        // before installing callbacks, contexts, weights or serving state.
        if (graph_builder_->config().hasUnifiedPP())
            throw std::invalid_argument(
                "DeviceGraphOrchestrator requires a participant-local graph; "
                "pipeline composition belongs to RankOrchestrator/GlobalOrchestrator");

        if (mpi_ctx_ && injected_topology_)
        {
            throw std::invalid_argument(
                "DeviceGraphOrchestrator Dependencies cannot provide both mpi_ctx and topology authorities");
        }

        for (const auto &[name, context] : domain_tp_contexts_)
            if (name.empty() || !context)
                throw std::invalid_argument(
                    "DeviceGraphOrchestrator requires named, non-null injected TP contexts");

        graph_builder_->bindStorageMemoryAuthority(physical_memory_authority_);
        graph_builder_->setCPUCurrentBatchLLEPPhysicalExecutor(this);

        configureExecutor();

        for (auto &[name, ctx] : domain_tp_contexts_)
        {
            graph_builder_->setTPContext(name, ctx.get());
        }
        // Propagate the one distributed rank authority to stage diagnostics.
        if (injected_topology_)
        {
            executor_.setMPIRank(injected_topology_->rank());
        }
        else if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        // Wire CollectiveContext to executor for GPU-native collectives (RCCL/NCCL/HOST)
        if (injected_collective_ctx_)
        {
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_DEBUG("[DeviceGraphOrchestrator] Wired CollectiveContext to DeviceGraphExecutor");
        }

        // Validate PP stage config if provided
        if (pp_stage_config_.has_value() && !pp_stage_config_->isValid())
        {
            throw std::invalid_argument("Invalid FactoryPPStageConfig in Dependencies: "
                                        "first_layer=" +
                                        std::to_string(pp_stage_config_->first_layer) +
                                        ", last_layer=" + std::to_string(pp_stage_config_->last_layer));
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized with dependencies, caching="
                  << (cache_config_.enabled ? "enabled" : "disabled")
                  << ", mpi_context=" << (mpi_ctx_ ? "provided" : "none")
                  << ", topology=" << (injected_topology_ ? "provided" : "none")
                  << ", collective=" << (injected_collective_ctx_ ? "provided" : "none")
                  << ", turboquant=" << (turboquant_ctx_ ? "provided" : "none")
                  << ", pp_stage=" << (pp_stage_config_.has_value() ? "configured" : "none")
                  << ", participant-local graph ownership");
    }

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        std::shared_ptr<IGraphBuilder> graph_builder,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::move(graph_builder)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config),
          memory_authority_mode_(MemoryAuthorityMode::IsolatedTest)
    {
        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator requires a valid graph builder");
        }

        graph_builder_->setCPUCurrentBatchLLEPPhysicalExecutor(this);

        configureExecutor();

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized with graph builder, caching="
                  << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    DeviceGraphOrchestrator::~DeviceGraphOrchestrator()
    {
        retireExecutionTopologyBeforeSharedStorageDetach();
        // A singleton DGO is its own complete parent owner. Shared participants
        // are detached by Rank only AFTER every child has crossed this phase.
        if (auto model = std::dynamic_pointer_cast<ModelContext>(injected_model_ctx_)) {
            std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> original;
            {
                std::lock_guard lock(model->immutable_row_table_mutex_);
                const auto found = model->adaptive_lookup_consumers_.find(state_.device_id);
                if (found != model->adaptive_lookup_consumers_.end()) {
                    const auto candidate = found->second.participants;
                    std::size_t consumers = 0;
                    for (const auto& [device, consumer] : model->adaptive_lookup_consumers_)
                        if (candidate && consumer.participants == candidate) ++consumers;
                    if (consumers == 1) original = candidate;
                }
            }
            if (original) model->retireAdaptiveImmutableLookupParticipants(original);
        }

        /*
         * Captured executables were the final borrowers of named workspace
         * pointers. Only now may the allocator discard its publication map and
         * expose the unchanged primary allocation to another runner. The
         * registry retains failure state so OrchestrationRunner can invalidate
         * the enclosing model-context contract after this destructor returns.
         */
        if (reusable_workspace_lease_)
        {
            std::string reuse_error;
            if (!reusable_workspace_lease_->publishReusable(&reuse_error))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Could not seal reusable execution workspace on "
                    << state_.device_id.toString() << ": "
                    << (reuse_error.empty()
                            ? std::string("unknown workspace lifecycle failure")
                            : reuse_error));
            }
        }
    }

    void DeviceGraphOrchestrator::retireExecutionTopologyBeforeSharedStorageDetach() noexcept
    {
        /*
         * Close every published GPU ownership edge before reverse-order member
         * destruction can release BufferArena storage ahead of graph caches and
         * their streams.  This is intentionally event-scoped: no device-wide
         * synchronization is permitted at the runner lifetime boundary.
         */
        retirePublishedDeviceWorkBeforeArenaRelease();

        /*
         * A request can finish after queueing a prefix restore but before the
         * next graph consumes its live-state handoff. Move that final event into
         * the retirement queue, then wait only for the exact restore events.
         * This is a teardown boundary, not a hot-path device synchronization.
         */
        clearPendingLivePrefixMutationReady();
        retirePendingPrefixPayloadUses(/*wait_for_all=*/true);

        /* Runtime-table producer events must close while the exact graph
         * streams they name still belong to live device contexts. This typed
         * edge deliberately precedes graph/context destruction; relying on
         * reverse member order left Dynamic teardown with stale HIP handles. */
        retireGraphBorrowedExecutionStreamsBeforeTopologyRelease();

        /*
         * Event retirement proves that no device work can still dereference an
         * arena pointer.  Destroy the executable topology while the arena is
         * nevertheless still its valid backing owner: HIP may defer a free for
         * memory embedded by a live graph, which would make the next runner pay
         * for both the old and new arena even though the old pointer has left
         * Llaminar's allocation registry.
         */
        destroyCapturedExecutionTopologyBeforeArenaRelease();

        // The source/channel may outlive this runner. Retire only its exact
        // canonical controller borrow, after every producer and native graph.
        // Recurring participants retain their separate definition frontier;
        // Rank closes that before any arena storage may become reusable.
        retireRetainedStorageGenerationBinding();

    }

    void DeviceGraphOrchestrator::retirePublishedDeviceWorkBeforeArenaRelease() noexcept
    {
        if (!state_.device_id.is_gpu())
            return;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot retire published GPU work without a backend for "
                      << state_.device_id.toString());
            std::terminate();
        }

        struct PublishedEvent
        {
            const char *name;
            const std::shared_ptr<void> *event;
            bool valid;
        };

        std::vector<PublishedEvent> published;
        published.reserve(
            stochastic_target_sample_ready_.size() +
            stochastic_draft_sample_ready_.size() +
            11);

        const auto append =
            [&](const char *name,
                const std::shared_ptr<void> &event,
                bool valid)
        {
            published.push_back(PublishedEvent{
                .name = name,
                .event = &event,
                .valid = valid,
            });
        };

        for (const auto &ready : stochastic_target_sample_ready_)
            append("stochastic_target_sample", ready.event, ready.valid);
        for (const auto &ready : stochastic_draft_sample_ready_)
            append("stochastic_draft_sample", ready.event, ready.valid);

        append("shifted_mtp_kv",
               shifted_mtp_kv_ready_.event,
               shifted_mtp_kv_ready_.valid);
        append("all_position_verifier",
               all_position_verifier_state_ready_.event,
               all_position_verifier_state_ready_.valid);
        append("accepted_spec_publication",
               accepted_spec_publication_ready_.event,
               accepted_spec_publication_ready_.valid);
        append("live_prefix_checkpoint",
               live_prefix_checkpoint_ready_.event,
               live_prefix_checkpoint_ready_.valid);
        append("live_prefix_mutation",
               live_prefix_mutation_ready_.event,
               live_prefix_mutation_ready_.valid);
        append("mtp_prefill_terminal_archive",
               mtp_prefill_terminal_archive_ready_.event,
               mtp_prefill_terminal_archive_ready_.valid);
        append("request_state_reset",
               request_state_reset_ready_.event,
               request_state_reset_ready_.valid);
        append("main_graph_build_device_state",
               main_graph_build_device_state_ready_.event,
               main_graph_build_device_state_ready_.valid);
        append("mtp_graph_build_device_state",
               mtp_graph_build_device_state_ready_.event,
               mtp_graph_build_device_state_ready_.valid);
        append("request_input_admission",
               request_input_admission_ready_.event,
               request_input_admission_ready_.valid);
        append("request_input_reuse",
               request_input_reuse_ready_.event,
               request_input_reuse_ready_.valid);
        append("forward_graph_output_main",
               forward_graph_output_ready_.main_forward.event,
               forward_graph_output_ready_.main_forward.valid);
        append("forward_graph_output_grouped_verifier",
               forward_graph_output_ready_.grouped_verifier.event,
               forward_graph_output_ready_.grouped_verifier.valid);

        std::unordered_set<void *> retired_events;
        retired_events.reserve(published.size());
        for (const PublishedEvent &publication : published)
        {
            if (!publication.valid)
                continue;
            if (!publication.event || !*publication.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Published GPU lifecycle state has no event"
                          << " state=" << publication.name
                          << " device=" << state_.device_id.toString());
                std::terminate();
            }

            void *event = publication.event->get();
            if (!retired_events.insert(event).second)
                continue;
            if (!backend->waitForEvent(event, state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to retire published GPU lifecycle event"
                          << " state=" << publication.name
                          << " device=" << state_.device_id.toString());
                std::terminate();
            }
        }

        /*
         * A request that never reaches its next maintenance boundary (static
         * overlay, prefill-only execution, or an early runner teardown) can
         * still own one durable epoch reader.  Every published inference edge
         * above is now complete, so close that reader on its exact producer
         * stream before the arena and retained boundary graph are destroyed.
         * releaseMoEOverlayEpochForExternalTransaction() inserts the acquired-
         * event wait itself; it therefore also covers an acquire submitted just
         * before an otherwise empty/error path without inventing a new stream.
         */
        try
        {
            const auto lease_state =
                moe_overlay_epoch_lease_lifecycle_->load();
            if (lease_state != MoEOverlayEpochLeaseState::Idle &&
                lease_state != MoEOverlayEpochLeaseState::ExternalReader &&
                lease_state !=
                    MoEOverlayEpochLeaseState::ReleasePublished)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay teardown found an incomplete forward epoch submission on "
                    << state_.device_id.toString()
                    << " lease_state="
                    << moeOverlayEpochLeaseStateName(lease_state));
                std::terminate();
            }
            if (lease_state == MoEOverlayEpochLeaseState::ExternalReader)
            {
                if (!moe_overlay_epoch_acquire_producer_stream_ ||
                    !releaseMoEOverlayEpochForExternalTransaction(
                        moe_overlay_epoch_acquire_producer_stream_,
                        "orchestrator_teardown"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Could not close the active ExpertOverlay epoch reader during teardown on "
                              << state_.device_id.toString());
                    std::terminate();
                }
            }
            if (moe_overlay_external_sequence_identity_.valid())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay teardown found "
                    "an execution-sequence identity without its live sequence "
                    "lease on "
                    << state_.device_id.toString());
                std::terminate();
            }
            moe_overlay_epoch_forward_submission_stream_ = nullptr;
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch teardown threw on "
                      << state_.device_id.toString() << ": "
                      << exception.what());
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch teardown threw an unknown exception on "
                      << state_.device_id.toString());
            std::terminate();
        }

        if (moe_overlay_epoch_lease_lifecycle_->load() ==
            MoEOverlayEpochLeaseState::ReleasePublished)
        {
            auto submission =
                moe_overlay_epoch_lease_lifecycle_->beginSubmission();
            if (!moe_overlay_epoch_released_event_ ||
                !submission.publishedReleaseProducerStream() ||
                submission.state() !=
                    MoEOverlayEpochLeaseState::ReleasePublished ||
                !backend->waitForEvent(
                    moe_overlay_epoch_released_event_.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Could not retire the ExpertOverlay epoch release event during teardown on "
                          << state_.device_id.toString());
                std::terminate();
            }
            if (!submission.commit(MoEOverlayEpochLeaseState::Idle))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay teardown lost release receipt ownership on "
                          << state_.device_id.toString()
                          << " lease_state="
                          << moeOverlayEpochLeaseStateName(
                                 submission.state()));
                std::terminate();
            }
        }
    }

    void DeviceGraphOrchestrator::
        retireGraphBorrowedExecutionStreamsBeforeTopologyRelease() noexcept
    {
        if (!graph_builder_)
            return;

        try
        {
            graph_builder_->retireBorrowedExecutionStreams(
                state_.device_id);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Could not retire graph-borrowed execution streams before topology release on "
                << state_.device_id.toString() << ": " << error.what());
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Could not retire graph-borrowed execution streams before topology release on "
                << state_.device_id.toString() << ": unknown exception");
            std::terminate();
        }
    }

    void DeviceGraphOrchestrator::destroyCapturedExecutionTopologyBeforeArenaRelease() noexcept
    {
        try
        {
            /*
             * The generation controller embeds borrowed child-capture
             * identities.  Retire that parent first so no surviving native
             * executable names a child while the child family is destroyed.
             */
            releaseRetainedStorageGenerationTopology();

            mtp_publication_admission_authority_.reset();
            // Publication hooks borrow verifier stages. Retire both children
            // before the forward graph owners, borrowers before stream owner.
            for (auto cache = mtp_speculative_state_publication_graphs_.rbegin();
                 cache != mtp_speculative_state_publication_graphs_.rend(); ++cache) cache->invalidate();

            if (forward_engine_)
                forward_engine_->discardAllCachedGraphs();

            for (auto &cache : layer_graph_cache_)
                cache.invalidate();

            mtp_sidecar_depth0_cache_.invalidate();
            mtp_sidecar_depth0_device_token_cache_.invalidate();
            mtp_sidecar_depth0_chained_cache_.invalidate();
            mtp_sidecar_depth0_chained_device_token_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_device_token_cache_.invalidate();
            for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            {
                if (cache)
                    cache->invalidate();
            }

            mtp_terminal_hidden_row_select_cache_.invalidate();
            mtp_terminal_hidden_rows_select_cache_.invalidate();
            for (auto &cache :
                 mtp_terminal_hidden_contiguous_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_terminal_hidden_device_accepted_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_terminal_hidden_request_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_shifted_prefill_hidden_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }

            // The rank's frozen topology outlives participants, but this native
            // recording embeds participant arena pointers. Release its lease
            // after all enclosing parents and before workspace reuse admission.
            pipeline_publication_transport_.reset();
            mtp_direct_token_history_graph_.invalidate();
            for (auto &cache : mtp_draft_token_publication_graphs_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache : mtp_verifier_preparation_graphs_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache : mtp_stochastic_outcome_graphs_) cache.invalidate();
            for (auto &cache : mtp_stochastic_target_distribution_graphs_) cache.invalidate();
            moe_overlay_finite_placement_graphs_.invalidate();
            moe_overlay_epoch_acquire_graph_.invalidate();
            moe_overlay_epoch_release_graph_.invalidate();
            device_moe_rebalance_maintenance_graph_.invalidate();

            last_mirrored_layer_checkpoint_prefix_.clear();

            /*
             * Graph caches own captures/streams created from these contexts.
             * Context destruction is therefore the final topology transition,
             * immediately before the caller may release arena storage.
             */
            device_contexts_.clear();
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Captured execution topology teardown threw before arena release: "
                << exception.what());
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Captured execution topology teardown threw an unknown exception before arena release");
            std::terminate();
        }
    }

    void DeviceGraphOrchestrator::setForwardGraphExecutionRendezvous(
        std::shared_ptr<ForwardGraphExecutionRendezvous> rendezvous)
    {
        forward_execution_rendezvous_ = std::move(rendezvous);
    }

    bool DeviceGraphOrchestrator::waitBeforeForwardGraphExecution(
        const ForwardInput &input,
        DeviceId execution_device,
        bool cache_miss)
    {
        (void)input;
        (void)cache_miss;
        auto rendezvous = forward_execution_rendezvous_;
        if (!rendezvous)
            return true;

        const int timeout_ms =
            collective_timeout_policy::effectiveCollectTimeoutMs(
                debugEnv().tp_collect_timeout_ms);
        const bool ok = rendezvous->arriveAndWait(execution_device, timeout_ms);
        if (!ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Forward graph pre-execution rendezvous failed on "
                      << execution_device.toString());
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::waitAtPrefillGraphCaptureBoundary(const ForwardInput& input,DeviceId device,
        const std::string& name,void* stream)
    {return waitAtPrefillGraphCaptureBoundaryImpl(input,device,name,stream,nullptr);}
    bool DeviceGraphOrchestrator::waitAtPrefillGraphCaptureBoundary(const ForwardInput& input,DeviceId device,
        const std::string& name,void* stream,LocalTPCaptureSourceDefinition& definition)
    {return waitAtPrefillGraphCaptureBoundaryImpl(input,device,name,stream,&definition);}

    bool DeviceGraphOrchestrator::waitAtPrefillGraphCaptureBoundaryImpl(
        const ForwardInput &input,
        DeviceId execution_device,
        const std::string &boundary_name,
        void *capture_stream,LocalTPCaptureSourceDefinition* definition)
    {
        (void)input;
        if (pipeline_forward_edges_){
            if(definition)throw std::logic_error("Source definition handoff requires its original LocalTP capture boundary");
            return pipeline_forward_edges_->captureBoundary(boundary_name, capture_stream);
        }
        if (!graph_builder_){
            if(definition)throw std::logic_error("Source definition handoff lost its original graph builder");
            return true;
        }

        auto *local_tp =
            dynamic_cast<ILocalTPContext *>(graph_builder_->config().tp_ctx);
        if (!local_tp || local_tp->degree() <= 1){
            if(!definition)return true;
            if(!definition->publication || definition->publication->stream()!=capture_stream ||
                reinterpret_cast<std::uintptr_t>(capture_stream)<=2 || definition->domain.source!=execution_device ||
                definition->publication->definition().domain!=definition->domain)
                throw std::logic_error("Singleton source definition lost its actual source cache");
            definition->publication->requireDefinition();definition->result=definition->publication;return true;
        }

        const int participant = graph_builder_->config().tp_device_idx;
        if (participant < 0 || participant >= local_tp->degree())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill graph capture boundary rejected: "
                      << "invalid LocalTP participant index " << participant
                      << " degree=" << local_tp->degree()
                      << " boundary=" << boundary_name
                      << " device=" << execution_device.toString());
            return false;
        }

        const auto &devices = local_tp->devices();
        if (static_cast<int>(devices.size()) != local_tp->degree())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill graph capture boundary rejected: "
                      << "LocalTP device list size " << devices.size()
                      << " does not match degree=" << local_tp->degree()
                      << " boundary=" << boundary_name);
            return false;
        }

        const DeviceId participant_device = devices[static_cast<size_t>(participant)].toLocalDeviceId();
        if (!(participant_device == execution_device))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill graph capture boundary rejected: "
                      << "participant " << participant
                      << " owns " << participant_device.toString()
                      << " but execution device is " << execution_device.toString()
                      << " boundary=" << boundary_name);
            return false;
        }

        const int timeout_ms =
            collective_timeout_policy::effectiveCollectTimeoutMs(
                debugEnv().tp_collect_timeout_ms);
        const std::string domain_boundary =
            "prefill_graph:" + boundary_name +
            ":domain=" + prefillGraphDomainId() +
            ":topology=" + std::to_string(prefillGraphTopologySignature());
        if (!capture_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill graph capture boundary requires "
                      << "an explicit GPU stream on " << execution_device.toString()
                      << " boundary=" << domain_boundary);
            return false;
        }
        const bool ok = definition?local_tp->graphCaptureBoundaryOnStream(domain_boundary,participant,capture_stream,timeout_ms,*definition):
            local_tp->graphCaptureBoundaryOnStream(domain_boundary,participant,capture_stream,timeout_ms);
        if (!ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill graph capture boundary rendezvous failed on "
                      << execution_device.toString()
                      << " boundary=" << domain_boundary
                      << " participant=" << participant);
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::waitAtDecodeGraphCaptureBoundary(const ForwardInput& input,DeviceId device,
        const std::string& name,void* stream)
    {return waitAtDecodeGraphCaptureBoundaryImpl(input,device,name,stream,nullptr);}
    bool DeviceGraphOrchestrator::waitAtDecodeGraphCaptureBoundary(const ForwardInput& input,DeviceId device,
        const std::string& name,void* stream,LocalTPCaptureSourceDefinition& definition)
    {return waitAtDecodeGraphCaptureBoundaryImpl(input,device,name,stream,&definition);}

    bool DeviceGraphOrchestrator::waitAtDecodeGraphCaptureBoundaryImpl(
        const ForwardInput &input,
        DeviceId execution_device,
        const std::string &boundary_name,
        void *capture_stream,LocalTPCaptureSourceDefinition* definition)
    {
        (void)input;
        if (!graph_builder_){
            if(definition)throw std::logic_error("Source definition handoff lost its original graph builder");
            return true;
        }

        auto *local_tp =
            dynamic_cast<ILocalTPContext *>(graph_builder_->config().tp_ctx);
        if (!local_tp || local_tp->degree() <= 1){
            if(!definition)return true;
            if(!definition->publication || definition->publication->stream()!=capture_stream ||
                reinterpret_cast<std::uintptr_t>(capture_stream)<=2 || definition->domain.source!=execution_device ||
                definition->publication->definition().domain!=definition->domain)
                throw std::logic_error("Singleton source definition lost its actual source cache");
            definition->publication->requireDefinition();definition->result=definition->publication;return true;
        }

        const int participant = graph_builder_->config().tp_device_idx;
        if (participant < 0 || participant >= local_tp->degree())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Decode graph capture boundary rejected: "
                      << "invalid LocalTP participant index " << participant
                      << " degree=" << local_tp->degree()
                      << " boundary=" << boundary_name
                      << " device=" << execution_device.toString());
            return false;
        }

        const auto &devices = local_tp->devices();
        if (static_cast<int>(devices.size()) != local_tp->degree())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Decode graph capture boundary rejected: "
                      << "LocalTP device list size " << devices.size()
                      << " does not match degree=" << local_tp->degree()
                      << " boundary=" << boundary_name);
            return false;
        }

        const DeviceId participant_device = devices[static_cast<size_t>(participant)].toLocalDeviceId();
        if (!(participant_device == execution_device))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Decode graph capture boundary rejected: "
                      << "participant " << participant
                      << " owns " << participant_device.toString()
                      << " but execution device is " << execution_device.toString()
                      << " boundary=" << boundary_name);
            return false;
        }

        const int timeout_ms =
            collective_timeout_policy::effectiveCollectTimeoutMs(
                debugEnv().tp_collect_timeout_ms);
        const std::string domain_boundary =
            "decode_graph:" + boundary_name +
            ":domain=" + prefillGraphDomainId() +
            ":topology=" + std::to_string(prefillGraphTopologySignature());
        if (!capture_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Decode graph capture boundary requires "
                      << "an explicit GPU stream on " << execution_device.toString()
                      << " boundary=" << domain_boundary);
            return false;
        }
        const bool ok = definition?local_tp->graphCaptureBoundaryOnStream(domain_boundary,participant,capture_stream,timeout_ms,*definition):
            local_tp->graphCaptureBoundaryOnStream(domain_boundary,participant,capture_stream,timeout_ms);
        if (!ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Decode graph capture boundary rendezvous failed on "
                      << execution_device.toString()
                      << " boundary=" << domain_boundary
                      << " participant=" << participant);
        }
        return ok;
    }

    // =========================================================================
    // Device Context Management
    // =========================================================================

    IDeviceContext *DeviceGraphOrchestrator::getDeviceContext(DeviceId device)
    {
        auto it = device_contexts_.find(device);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context using DeviceId
        auto ctx = IDeviceContext::create(device);
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to create device context for device " << device.toString());
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device] = std::move(ctx);

        LOG_DEBUG("[DeviceGraphOrchestrator] Created device context for device " << device.to_string());
        return raw_ptr;
    }

    IWorkerGPUContext *DeviceGraphOrchestrator::getWorkerGPUContext(DeviceId device)
    {
        if (!device.is_gpu())
            return nullptr;
        return &GPUDeviceContextPool::instance().getContext(device);
    }

    bool DeviceGraphOrchestrator::workerGPUContextUsesProcessPool(DeviceId device) const
    {
        return device.is_gpu();
    }

    bool DeviceGraphOrchestrator::isMoeRebalancingActive() const
    {
        if (!graph_builder_)
            return false;

        const auto &moe = graph_builder_->config().moe;
        if (!moe.usesExpertOverlayDurableResidencyAuthority() ||
            !moe.routed_expert_plan ||
            moe.rebalance_config.mode != MoERebalanceRuntimeMode::Dynamic)
        {
            return false;
        }

        if (moe.expert_overlay_residency_authority &&
            moe.expert_overlay_residency_authority->stationaryTensorShardResidency())
            return false;

        /* A one-participant tier has no peer to rebalance with. Treating its
         * Dynamic policy as active would require a collective maintenance
         * graph that cannot perform any movement and would turn an intentional
         * no-op topology into a request-time failure. */
        if (moe.routed_expert_plan->routed_tiers.size() == 1u)
        {
            const auto &tier =
                moe.routed_expert_plan->routed_tiers.front();
            const auto domain = std::find_if(
                moe.routed_expert_plan->domains.begin(),
                moe.routed_expert_plan->domains.end(),
                [&](const RoutedExpertDomain &candidate)
                {
                    return candidate.name == tier.domain;
                });
            if (domain != moe.routed_expert_plan->domains.end() &&
                domain->participants.size() <= 1u)
            {
                return false;
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::isMoeRebalancingGraphStableForPrefillCapture() const
    {
        return usesGraphStableMoEOverlayResidency();
    }

    bool DeviceGraphOrchestrator::prefillGraphCaptureDisabledByHost() const
    {
        return false;
    }

    bool DeviceGraphOrchestrator::usesGraphStableMoEOverlayResidency() const
    {
        if (!graph_builder_)
            return false;

        const auto &moe = graph_builder_->config().moe;
        /*
         * Both authority executors publish through model-lifetime epoch and
         * runtime-bank addresses. Graph identity therefore depends on the
         * arena generation, not on the currently selected expert bank.
         */
        return moe.usesExpertOverlayDurableResidencyAuthority() &&
               moe.routed_expert_plan &&
               moe.authority_execution !=
                   MoEOverlayAuthorityExecutionKind::Unresolved;
    }

    MoEOverlayAuthorityExecutionKind
    DeviceGraphOrchestrator::moeOverlayAuthorityExecution() const
    {
        if (!graph_builder_)
            return MoEOverlayAuthorityExecutionKind::Unresolved;

        const auto &moe = graph_builder_->config().moe;
        if (!moe.usesExpertOverlayDurableResidencyAuthority())
            return MoEOverlayAuthorityExecutionKind::Unresolved;
        if (!moe.routed_expert_plan ||
            moe.authority_execution ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
        {
            throw std::logic_error(
                "ExpertOverlay graph participant has no frozen authority execution backend");
        }
        return moe.authority_execution;
    }

    bool DeviceGraphOrchestrator::
        usesDeviceResidentMoEOverlayAuthority() const
    {
        return moeOverlayAuthorityExecution() ==
               MoEOverlayAuthorityExecutionKind::
                   DeviceResident;
    }

    bool DeviceGraphOrchestrator::
        usesTopologyWideDeviceMoEOverlayController() const noexcept
    {
        if (!graph_builder_ || !state_.device_id.is_gpu())
            return false;

        const auto &moe = graph_builder_->config().moe;
        return moe.device_controller_fabric != nullptr &&
               moe.authority_execution ==
                   MoEOverlayAuthorityExecutionKind::DeviceResident;
    }

    bool DeviceGraphOrchestrator::
        usesParticipantLocalDeviceMoERebalanceController() const noexcept
    {
        const auto policy =
            deviceMoERebalanceMaintenanceExecutionPolicy();
        return policy ==
                   DeviceMoERebalanceMaintenanceExecutionPolicy::
                       NativeConditionalGraph ||
               policy ==
                   DeviceMoERebalanceMaintenanceExecutionPolicy::
                       HostScheduledCapturedMaintenance;
    }

    bool DeviceGraphOrchestrator::
        deviceResidentMoEOverlayMaintenanceReady() const
    {
        if (!usesParticipantLocalDeviceMoERebalanceController())
            return false;

        const auto &cache = device_moe_rebalance_maintenance_graph_;
        return state_.device_id.is_gpu() &&
               static_cast<bool>(
                   moe_overlay_epoch_execution_binding_) &&
               cache.graph != nullptr &&
               !cache.collective_nodes.empty() &&
               cache.workspace_generation != 0u;
    }

    uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const
    {
        /*
         * Graph-stable GPU MoE rebalance publishes into persistent runtime
         * descriptor tables and transfer slots. Captured decode graphs record
         * those table pointers, so changing expert ownership is data mutation,
         * not graph topology. CPU or non-runtime-table GPU rebalancing keeps the
         * epoch in the graph key and must recapture when placement changes.
         */
        if (usesGraphStableMoEOverlayResidency())
            return 0;

        uint64_t epoch = std::max(current_expert_replica_epoch_, current_expert_mask_epoch_);
        if (!moe_rebalance_controller_)
            return epoch;
        return std::max(epoch, moe_rebalance_controller_->placementEpoch());
    }

    uint64_t DeviceGraphOrchestrator::moeRuntimeMovementEpoch() const
    {
        uint64_t epoch =
            std::max(moePlacementEpoch(), moe_runtime_movement_epoch_);
        if (!graph_builder_)
            return epoch;

        const auto &authority = graph_builder_->config()
                                    .moe
                                    .expert_overlay_residency_authority;
        if (!authority)
            return epoch;

        /*
         * A heterogeneous ExpertOverlay has one host-side RCU authority.  Its
         * immutable publication is the live placement value; copying that
         * epoch into a device-owned shadow would create a second authority and
         * a coherence race.  Prefix lookup and diagnostics instead sample the
         * current RCU snapshot directly at the request boundary.
         */
        const auto snapshot = authority->snapshot();
        if (!snapshot || !snapshot->valid())
        {
            throw std::logic_error(
                "ExpertOverlay runtime movement epoch has no valid authoritative snapshot");
        }
        return std::max(epoch, snapshot->epoch);
    }

    std::string DeviceGraphOrchestrator::prefillGraphDomainId() const
    {
        if (!moe_rebalance_controller_)
            return "single";
        return moe_rebalance_controller_->domainId();
    }

    int DeviceGraphOrchestrator::prefillGraphParticipantId() const
    {
        return moeRebalanceParticipantId();
    }

    PrefillChunkMaintenanceState DeviceGraphOrchestrator::prefillChunkMaintenanceState(
        const PrefillChunkPlan &chunk) const
    {
        PrefillChunkMaintenanceState state;
        state.chunk_index = chunk.chunk_index;
        state.histograms_merged = true;
        state.manual_boundaries_complete = true;
        state.graph_capture_active = isGraphCaptureActive();
        state.graph_replay_active = false;
        state.participants_at_same_boundary = true;

        if (!moe_rebalance_controller_)
            return state;

        if (auto *histogram = moe_rebalance_controller_->histogram())
        {
            state.histograms_merged = histogram->syncRuntimeHistograms();
        }

        state.rebalance_requested =
            state.histograms_merged && moe_rebalance_controller_->shouldRebalance();
        return state;
    }

    bool DeviceGraphOrchestrator::onPrefillChunkMaintenance(
        const PrefillChunkPlan &chunk,
        const PrefillChunkMaintenanceDecision &decision)
    {
        if (!decision.ok)
            return false;
        if (!decision.can_run || !moe_rebalance_controller_)
            return true;
        if (!moe_rebalance_controller_->shouldRebalance())
            return true;

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            LOG_WARN("[DGO] Prefill chunk maintenance reached a multi-rank MoE rebalance "
                     "boundary at chunk "
                     << chunk.chunk_index
                     << "; deferring to rank/runner-level coordination");
            return !decision.required;
        }

        if (moe_rebalance_controller_->maxReplicasPerSocket() > 0 ||
            debugEnv().moe_rebalance.gpu_cache_experts_per_layer > 0)
        {
            LOG_WARN("[DGO] Prefill chunk maintenance reached a MoE replica/cache rebalance "
                     "boundary at chunk "
                     << chunk.chunk_index
                     << "; replica and GPU-cache placement require runner-level coordination");
            return !decision.required;
        }

        try
        {
            const auto old_ownership =
                moe_rebalance_controller_->currentOwnership();
            const auto ownership_changes =
                moe_rebalance_controller_->rebalance();

            if (ownership_changes.empty())
                return true;

            invalidateSparseQueryParents();
            ExpertTransferResult received;
            auto manifest = ExpertWeightTransfer::buildManifest(
                old_ownership,
                moe_rebalance_controller_->currentOwnership());
            if (!manifest.empty())
                received = transferExpertWeights(manifest);

            const int participant_id = mpi_ctx_ ? mpi_ctx_->local_rank() : 0;
            auto masks = moe_rebalance_controller_->computeExpertMasksForParticipant(participant_id);
            applyExpertMasks(
                masks, received.serialized, received.prepared);

            retireMTPPublicationVerifierBorrows();
            if (forward_engine_)
                forward_engine_->discardAllCachedGraphs();
            mtp_sidecar_depth0_cache_.invalidate();
            mtp_sidecar_depth0_device_token_cache_.invalidate();
            mtp_sidecar_depth0_chained_cache_.invalidate();
            mtp_sidecar_depth0_chained_device_token_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_device_token_cache_.invalidate();
            for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
                cache->invalidate();
            mtp_terminal_hidden_row_select_cache_.invalidate();
            mtp_terminal_hidden_rows_select_cache_.invalidate();
            for (auto &cache :
                 mtp_terminal_hidden_contiguous_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_terminal_hidden_device_accepted_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_terminal_hidden_request_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache :
                 mtp_shifted_prefill_hidden_rows_select_caches_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache : mtp_verifier_preparation_graphs_)
            {
                if (cache)
                    cache->invalidate();
            }
            for (auto &cache : layer_graph_cache_)
                cache.invalidate();
            device_moe_rebalance_maintenance_graph_.invalidate();
            resetKernelDynamicState();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DGO] Prefill chunk MoE maintenance failed at chunk "
                      << chunk.chunk_index << ": " << e.what());
            return false;
        }

        return true;
    }

    bool DeviceGraphOrchestrator::beginPrefillChunkGraphSubmission(
        ForwardInput &chunk_input,
        const PrefillChunkPlan &chunk,
        std::unique_ptr<IPrefillChunkGraphSubmissionLease> *lease,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!lease)
        {
            if (error)
                *error =
                    "Prefill chunk graph submission requires a lease output";
            return false;
        }
        lease->reset();

        if (!moe_overlay_inference_transaction_coordinator_)
            return true;
        if (chunk.chunk_index < 0 || chunk.real_count <= 0 ||
            chunk.bucket_seq_len < chunk.real_count ||
            chunk_input.execution_role !=
                ForwardExecutionRole::MainInference ||
            chunk_input.execution_phase !=
                ForwardExecutionPhase::Prefill ||
            chunk_input.batch_size != 1 ||
            chunk_input.real_seq_len != chunk.real_count ||
            chunk_input.seq_len != chunk.bucket_seq_len ||
            chunk_input.bucket_seq_len != chunk.bucket_seq_len)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay prefill chunk submission received "
                    "divergent scheduler and forward-input geometry";
            }
            return false;
        }

        std::string admission_error;
        if (!moe_overlay_inference_transaction_coordinator_
                 ->admitSerialPrefillGraph(
                     moe_overlay_inference_transaction_participant_index_,
                     &admission_error))
        {
            if (error)
            {
                *error = admission_error.empty()
                             ? "ExpertOverlay serial prefill graph-sequence "
                               "admission failed"
                             : std::move(admission_error);
            }
            return false;
        }

        MoEOverlayInferenceExecutionDescriptor descriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .logical_step_id = 0,
            .placement_epoch =
                moe_overlay_inference_transaction_coordinator_
                    ->currentPlacementEpoch(),
            .request_count = 1,
            .logical_rows_per_request = chunk.real_count,
            .physical_rows_per_request = chunk.bucket_seq_len,
            .draft_depth = -1,
            .sidecar_depth = -1,
        };
        auto overlay_lease =
            std::make_unique<
                MoEOverlayPrefillChunkGraphSubmissionLease>(
                moe_overlay_inference_transaction_coordinator_,
                descriptor,
                moe_overlay_inference_transaction_participant_index_);
        if (!overlay_lease->ready())
        {
            if (error)
            {
                *error = overlay_lease->error().empty()
                             ? "ExpertOverlay transaction authority rejected "
                               "the scheduled prefill graph"
                             : overlay_lease->error();
            }
            return false;
        }
        if (!overlay_lease->bind(
                chunk_input,
                moe_overlay_collective_request_generation_,
                error))
        {
            return false;
        }

        *lease = std::move(overlay_lease);
        return true;
    }

    /**
     * @brief Give one captured prefill chunk a unique diagnostic namespace.
     *
     * Every graph launch still receives the same ForwardInput and uses the same
     * captured stream/event edges. The namespace is consumed only by the
     * post-launch SnapshotCapture callback, where it prevents the next chunk
     * from overwriting the previous chunk's mathematical checkpoint.
     */
    void DeviceGraphOrchestrator::beginPrefillChunkSnapshotDiagnostics(
        const PrefillChunkPlan &chunk)
    {
        if (!snapshot_enabled_)
            return;

        if (prefill_chunk_snapshot_previous_context_.has_value())
        {
            throw std::logic_error(
                "prefill chunk snapshot scope was entered before its prior "
                "scope completed");
        }
        if (chunk.chunk_index < 0 ||
            static_cast<size_t>(chunk.chunk_index) !=
                prefill_chunk_snapshot_sequence_.size() ||
            chunk.real_count <= 0)
        {
            throw std::logic_error(
                "prefill chunk snapshot order or real-row count is invalid");
        }

        const std::string context =
            "PREFILL_CHUNK_" + std::to_string(chunk.chunk_index);
        prefill_chunk_snapshot_sequence_.push_back({
            .context = context,
            .logical_rows = static_cast<size_t>(chunk.real_count),
        });
        /*
         * Allocate/record the diagnostic descriptor before changing the active
         * callback namespace. If allocation throws, no partially-entered
         * namespace remains for a later graph publication to inherit.
         */
        prefill_chunk_snapshot_previous_context_ = snapshot_context_;
        snapshot_context_ = context;
    }

    /**
     * @brief Restore the prior diagnostics namespace after one chunk attempt.
     *
     * ForwardExecutionEngine owns this pair with an RAII scope. A failed graph
     * is therefore unable to leak its chunk ordinal into a later diagnostic
     * publication, even though failure handling exits before final aggregation.
     */
    void DeviceGraphOrchestrator::endPrefillChunkSnapshotDiagnostics(
        const PrefillChunkPlan &chunk)
    {
        (void)chunk;
        if (!snapshot_enabled_)
            return;

        if (!prefill_chunk_snapshot_previous_context_.has_value())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk snapshot scope "
                      "ended without a matching begin");
            return;
        }

        snapshot_context_ = std::move(*prefill_chunk_snapshot_previous_context_);
        prefill_chunk_snapshot_previous_context_.reset();
    }

    /**
     * @brief Reconstruct full-prompt parity checkpoints after a segmented replay.
     *
     * Padded GPU buckets have already been projected to each chunk's live rows
     * by DeviceGraphExecutor. SnapshotCapture now validates that every
     * sequence-shaped stage published every chunk and concatenates those values
     * in scheduler order. This remains diagnostic-only: it runs after all
     * graph launches, does not add a graph node, and never alters KV/model
     * state.
     */
    bool DeviceGraphOrchestrator::finalizePrefillChunkSnapshotDiagnostics(
        const PrefillChunkSchedule &schedule)
    {
        if (!snapshot_enabled_)
            return true;

        if (prefill_chunk_snapshot_previous_context_.has_value() ||
            schedule.chunks.size() != prefill_chunk_snapshot_sequence_.size())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk snapshot sequence "
                      "is incomplete before aggregation");
            return false;
        }

        for (size_t index = 0; index < schedule.chunks.size(); ++index)
        {
            const auto &chunk = schedule.chunks[index];
            const auto &captured = prefill_chunk_snapshot_sequence_[index];
            if (chunk.chunk_index != static_cast<int>(index) ||
                chunk.real_count <= 0 ||
                captured.logical_rows !=
                    static_cast<size_t>(chunk.real_count))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk snapshot "
                          "geometry disagrees with the executed scheduler");
                return false;
            }
        }

        const SnapshotChunkSequenceAggregation aggregation =
            snapshot_capture_.aggregateSequentialChunkSnapshots(
                prefill_chunk_snapshot_sequence_);
        prefill_chunk_snapshot_sequence_.clear();
        if (!aggregation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not aggregate segmented "
                      "prefill parity snapshots: " << aggregation.error);
            return false;
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "prefill_chunk_snapshot_sequence_keys",
            static_cast<double>(aggregation.aggregated_sequence_keys),
            "prefill",
            state_.device_id.toString(),
            {
                {"chunks", std::to_string(schedule.chunks.size())},
                {"terminal_or_nonsequence_keys",
                 std::to_string(aggregation.terminal_or_nonsequence_keys)},
                {"diagnostic_only", "true"},
            });
        return true;
    }

    /**
     * @brief Discard partial segmented-prefill evidence after an unsuccessful request.
     *
     * A partial collection is not a valid full-prompt checkpoint, so retaining
     * it would make a later parity comparison look more complete than it is.
     * Clearing the diagnostic store is safe here because capture exists only
     * for the failed request; production tensor/graph ownership is unaffected.
     */
    void DeviceGraphOrchestrator::cancelPrefillChunkSnapshotDiagnostics() noexcept
    {
        prefill_chunk_snapshot_previous_context_.reset();
        prefill_chunk_snapshot_sequence_.clear();
        snapshot_context_.clear();
        if (snapshot_enabled_)
            snapshot_capture_.clear();
    }

    // =========================================================================
    // Weight and Buffer Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::requireOpenWeightPublication(const char *caller) const
    {
        const auto reject = [&](const char *owner) {
            throw std::logic_error(std::string(caller) +
                " cannot replace graph-bound model weights: " + owner);
        };
        // Both CPU stages and GPU recordings embed these source addresses.
        // Freeze before capture, when the first owned graph is constructed;
        // request reset does not reopen an immutable model binding lifetime.
        if (serving_graph_family_plan_ ||
            serving_graph_family_lifecycle_ != ServingGraphFamilyLifecycle::AwaitingMaterialization)
            reject("serving graph family admission");
        if (forward_engine_ && !forward_engine_->cacheEmpty())
            reject("main forward graph cache");
        for (const auto &layer : layer_graph_cache_)
            if (layer.attention_decode || layer.ffn_decode)
                reject("layer graph cache");
        for (const auto *cache : {&mtp_sidecar_depth0_cache_, &mtp_sidecar_depth0_device_token_cache_,
                 &mtp_sidecar_depth0_chained_cache_, &mtp_sidecar_depth0_chained_device_token_cache_,
                 &mtp_sidecar_depth0_kv_only_cache_, &mtp_sidecar_depth0_kv_only_device_token_cache_})
            if (cache->graph)
                reject("scalar MTP sidecar graph cache");
        for (const auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            if (cache && cache->graph)
                reject("batched MTP sidecar graph cache");
    }

    void DeviceGraphOrchestrator::setWeights(const ModelWeights &weights)
    {
        requireOpenWeightPublication("setWeights");
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set weights: graph builder not initialized");
            return;
        }

        // Phase 6: Pre-resolve all layer weights to freeze weight resolution.
        // After this, graph construction reads pre-resolved pointers rather than
        // calling getWeightForDevice() lazily during graph build.
        const auto &cfg = graph_builder_->config();
        const int n_layers = cfg.n_layers;
        if (n_layers > 0 && weights.get_layer_weights)
        {
            // For PP stages, graph builders use global layer indices
            // (pp_layer_offset .. pp_layer_offset + n_layers - 1).
            // For single-device, pp_layer_offset=0 and indices are 0..n_layers-1.
            const int first_layer = cfg.pp_layer_offset;
            const int last_layer = first_layer + n_layers;

            ModelWeights frozen_weights = weights;
            auto resolved = std::make_shared<std::unordered_map<int, LayerWeights>>();
            resolved->reserve(n_layers);
            for (int i = first_layer; i < last_layer; ++i)
                (*resolved)[i] = weights.get_layer_weights(i);

            frozen_weights.get_layer_weights = [resolved](int layer_idx) -> LayerWeights
            {
                auto it = resolved->find(layer_idx);
                if (it != resolved->end())
                    return it->second;
                return {};
            };

            // Phase 6 (continued): Build FrozenModelWeightSet for audit, validation,
            // and Phase 7 PreparedWeightStore integration.
            buildFrozenWeightSet(weights, *resolved, first_layer, last_layer);

            if (frozen_weight_set_)
            {
                auto weight_bindings = makeModelWeightBindings(
                    *frozen_weight_set_, state_.device_id);
                graph_builder_->setWeightBindings(weight_bindings);
                graph_builder_->setWeights(toLegacyModelWeights(weight_bindings));
            }
            else
            {
                graph_builder_->setWeights(frozen_weights);
            }

            LOG_DEBUG("[DeviceGraphOrchestrator] Model weights frozen for " << n_layers
                                                                            << " layers [" << first_layer << ", " << last_layer << ")");
        }
        else
        {
            graph_builder_->setWeights(weights);
            LOG_DEBUG("[DeviceGraphOrchestrator] Model weights configured for full forward pass");
        }
    }

    void DeviceGraphOrchestrator::setFrozenWeightSet(std::unique_ptr<FrozenModelWeightSet> weight_set)
    {
        requireOpenWeightPublication("setFrozenWeightSet");
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set frozen weights: graph builder not initialized");
            return;
        }
        if (!weight_set)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set frozen weights: null weight set");
            return;
        }

        weight_set->validateForGraph();
        cached_mtp_weight_bindings_.reset();
        cached_mtp_weight_bindings_source_ = nullptr;
        frozen_weight_set_ = std::move(weight_set);

        auto weight_bindings = makeModelWeightBindings(
            *frozen_weight_set_, state_.device_id);
        graph_builder_->setWeightBindings(weight_bindings);
        graph_builder_->setWeights(toLegacyModelWeights(weight_bindings));

        LOG_DEBUG("[DeviceGraphOrchestrator] FrozenModelWeightSet configured directly with "
                  << frozen_weight_set_->bindings().size() << " bindings");
    }

    const FrozenModelWeightSet *DeviceGraphOrchestrator::selectMTPDecodeWeightSet(
        const char *caller) const
    {
        const char *context = (caller && caller[0] != '\0') ? caller : "MTP decode";
        const bool use_replicated_decode_weights =
            graph_builder_ &&
            graph_builder_->config().dense_tp_enabled &&
            (graph_builder_->config().dense_tp_decode_replicated ||
             graph_builder_->config().mtpUsesReplicatedDenseSidecarBinding());

        if (use_replicated_decode_weights)
        {
            if (!decode_replicated_dense_weight_set_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] " << context
                                                       << " requires replicated dense decode MTP weights");
                return nullptr;
            }
            return decode_replicated_dense_weight_set_.get();
        }

        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] " << context
                                                   << " requires frozen MTP weight bindings");
            return nullptr;
        }
        return frozen_weight_set_.get();
    }

    const ModelWeightBindings *DeviceGraphOrchestrator::selectMTPDecodeWeightBindings(
        const char *caller) const
    {
        const FrozenModelWeightSet *weight_set = selectMTPDecodeWeightSet(caller);
        if (!weight_set)
            return nullptr;
        if (cached_mtp_weight_bindings_ &&
            cached_mtp_weight_bindings_source_ == weight_set &&
            cached_mtp_weight_bindings_device_ == state_.device_id)
        {
            return &*cached_mtp_weight_bindings_;
        }
        cached_mtp_weight_bindings_ = makeModelWeightBindings(*weight_set, state_.device_id);
        cached_mtp_weight_bindings_source_ = weight_set;
        cached_mtp_weight_bindings_device_ = state_.device_id;
        return &*cached_mtp_weight_bindings_;
    }

    void DeviceGraphOrchestrator::setDecodeReplicatedDenseWeightSet(std::unique_ptr<FrozenModelWeightSet> weight_set)
    {
        requireOpenWeightPublication("setDecodeReplicatedDenseWeightSet");
        cached_mtp_weight_bindings_.reset();
        cached_mtp_weight_bindings_source_ = nullptr;
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set replicated dense decode weights: graph builder not initialized");
            return;
        }

        if (!weight_set)
        {
            decode_replicated_dense_weight_set_.reset();
            graph_builder_->setDecodeReplicatedDenseWeightBindings({});
            return;
        }

        weight_set->validateForGraph();
        auto weight_bindings = makeModelWeightBindings(
            *weight_set, state_.device_id);
        const auto &graph_config = graph_builder_->config();
        const bool requires_full_replicated_dense =
            graph_config.dense_tp_enabled &&
            graph_config.dense_tp_decode_replicated;
        const bool requires_mirrored_embedding =
            graph_config.dense_tp_enabled &&
            graph_config.dense_tp_decode_mirrored_embedding;
        const bool requires_mirrored_mtp_head =
            graph_config.mtpUsesMirroredTerminalHeadBinding();
        const bool requires_replicated_mtp_sidecar =
            graph_config.mtpUsesReplicatedDenseSidecarBinding();
        const bool owns_embedding_stage =
            !pp_stage_config_.has_value() || pp_stage_config_->has_embedding;
        const bool owns_lm_head_stage =
            !pp_stage_config_.has_value() || pp_stage_config_->has_lm_head;
        if (requires_full_replicated_dense ||
            requires_mirrored_embedding ||
            requires_mirrored_mtp_head ||
            requires_replicated_mtp_sidecar)
        {
            std::vector<std::string> missing;
            if ((requires_mirrored_embedding ||
                 (requires_full_replicated_dense && owns_embedding_stage)) &&
                !weight_bindings.embedding_table)
            {
                missing.emplace_back("token_embd.weight");
            }
            if ((requires_full_replicated_dense ||
                 requires_mirrored_mtp_head) && owns_lm_head_stage)
            {
                // Auxiliary planning already uses this model-owned inventory.
                // Validate that same contract here: a branch-residual terminal
                // mixer cannot be replaced by an ordinary final-norm binding.
                const auto schema = SchemaFactoryRegistry::getFactory(
                    graph_builder_->architectureName());
                for (const char *name : schema->terminalInputWeightNames())
                {
                    const auto *binding = optionalGlobalBinding(*weight_set, name);
                    if (!binding || !binding->tensor)
                    {
                        missing.emplace_back(name);
                    }
                }
            }
            if ((requires_full_replicated_dense ||
                 requires_mirrored_mtp_head) &&
                owns_lm_head_stage &&
                !weight_bindings.lm_head)
            {
                missing.emplace_back("output.weight");
            }
            if (retainsMTPGraphCapacity(graph_config.mtp) &&
                (requires_full_replicated_dense ||
                 requires_replicated_mtp_sidecar) &&
                owns_lm_head_stage &&
                (weight_bindings.mtp.empty() || weight_bindings.mtp.depths.empty()))
            {
                /*
                 * MTP verifies decode rows and therefore belongs to the
                 * replicated dense decode view.  Failing here catches an
                 * incomplete weight plan while the runner is still being built,
                 * instead of letting the first speculative step discover a
                 * missing sidecar block after expensive graph initialization.
                 */
                missing.emplace_back("MTP sidecar weights");
            }

            if (!missing.empty())
            {
                std::ostringstream joined;
                for (size_t i = 0; i < missing.size(); ++i)
                {
                    if (i > 0)
                        joined << ", ";
                    joined << missing[i];
                }
                throw std::runtime_error(
                    "[DeviceGraphOrchestrator] Replicated dense/MTP auxiliary weight set is incomplete for " +
                    state_.device_id.toString() + ": missing " + joined.str() +
                    "; available globals: " + summarizeFrozenWeightSetGlobals(*weight_set));
            }
        }

        decode_replicated_dense_weight_set_ = std::move(weight_set);
        graph_builder_->setDecodeReplicatedDenseWeightBindings(weight_bindings);

        LOG_DEBUG("[DeviceGraphOrchestrator] Replicated dense decode FrozenModelWeightSet configured with "
                  << decode_replicated_dense_weight_set_->bindings().size()
                  << " bindings"
                  << " embedding=" << (weight_bindings.embedding_table ? "yes" : "no")
                  << " final_norm=" << (weight_bindings.final_norm ? "yes" : "no")
                  << " lm_head=" << (weight_bindings.lm_head ? "yes" : "no"));
    }

    void DeviceGraphOrchestrator::buildFrozenWeightSet(
        const ModelWeights &weights,
        const std::unordered_map<int, LayerWeights> &resolved_layers,
        int first_layer, int last_layer)
    {
        requireOpenWeightPublication("buildFrozenWeightSet");
        // Determine strategy from current orchestrator state
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.pp_stages = 1;
        strategy.tp_degree = 1;
        if (prepared_weight_store_ && prepared_weight_store_->modelId().value != 0)
        {
            strategy.model_id = prepared_weight_store_->modelId();
        }
        else if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
        {
            if (auto store = concrete_weight_manager->preparedWeightStoreIfInitialized();
                store && store->modelId().value != 0)
            {
                strategy.model_id = store->modelId();
            }
        }
        if (graph_builder_)
        {
            const auto &cfg = graph_builder_->config();
            strategy.devices.push_back(cfg.default_device);
        }
        const DeviceId target_device = strategy.devices.empty()
                                           ? DeviceId::cpu()
                                           : strategy.devices.front();
        const bool can_infer_prepared_metadata =
            static_cast<bool>(std::dynamic_pointer_cast<WeightManager>(weight_manager_));
        const auto preparedKindFor = [&](TensorBase *tensor, WeightRole role) -> PreparedWeightKind
        {
            if (!can_infer_prepared_metadata || !tensor)
                return PreparedWeightKind::None;
            if (role == WeightRole::Embedding)
                return PreparedWeightKind::PreparedEmbedding;
            if (role == WeightRole::Norm ||
                role == WeightRole::Bias ||
                role == WeightRole::SharedExpertInputGate ||
                tensor->shape().size() != 2)
                return PreparedWeightKind::None;
            if (target_device.is_cuda())
                return PreparedWeightKind::CudaInt8PackedGemm;
            if (target_device.is_rocm())
                return PreparedWeightKind::RocmInt8PackedGemm;
            return PreparedWeightKind::CpuPackedGemm;
        };

        // Build bindings from global weights + resolved layer weights
        ModelWeightSetBuilder builder(strategy);

        // Global weights
        auto addGlobal = [&](TensorBase *tensor, const std::string &name, WeightRole role)
        {
            if (!tensor)
                return;
            WeightIdentity id;
            id.canonical_name = name;
            id.model_id = strategy.model_id;
            id.role = role;
            id.logical_id = stableWeightLogicalId(name);
            WeightBinding binding;
            binding.binding_id = stableLegacyGraphBindingId(
                strategy.model_id,
                name,
                role,
                -1,
                target_device);
            binding.identity = id;
            binding.tensor = tensor;
            binding.immutable = true;
            binding.residency.home_device = target_device;
            binding.residency.resident_device = target_device;
            if (const auto kind = preparedKindFor(tensor, role);
                kind != PreparedWeightKind::None)
            {
                binding.prepared = PreparedWeightRef{
                    strategy.model_id,
                    0,
                    kind,
                    target_device};
            }
            builder.addBinding(std::move(binding));
        };

        addGlobal(weights.embedding_table, "token_embd.weight", WeightRole::Embedding);
        addGlobal(weights.final_norm, "output_norm.weight", WeightRole::OutputNorm);
        addGlobal(weights.lm_head, "output.weight", WeightRole::LMHead);

        // Per-layer weights
        for (int layer_idx = first_layer; layer_idx < last_layer; ++layer_idx)
        {
            auto it = resolved_layers.find(layer_idx);
            if (it == resolved_layers.end())
                continue;
            const auto &lw = it->second;

            auto addLayer = [&](TensorBase *tensor, const std::string &suffix, WeightRole role)
            {
                if (!tensor)
                    return;
                std::string canonical = "blk." + std::to_string(layer_idx) + "." + suffix;
                WeightIdentity id;
                id.canonical_name = canonical;
                id.model_id = strategy.model_id;
                id.role = role;
                id.layer = layer_idx;
                id.logical_id = stableWeightLogicalId(canonical);
                WeightBinding binding;
                binding.binding_id = stableLegacyGraphBindingId(
                    strategy.model_id,
                    canonical,
                    role,
                    layer_idx,
                    target_device);
                binding.identity = id;
                binding.tensor = tensor;
                binding.immutable = true;
                binding.residency.home_device = target_device;
                binding.residency.resident_device = target_device;
                if (const auto kind = preparedKindFor(tensor, role);
                    kind != PreparedWeightKind::None)
                {
                    binding.prepared = PreparedWeightRef{
                        strategy.model_id,
                        0,
                        kind,
                        target_device};
                }
                builder.addBinding(std::move(binding));
            };

            addLayer(lw.wq, "attn_q.weight", WeightRole::AttentionQ);
            addLayer(lw.wk, "attn_k.weight", WeightRole::AttentionK);
            addLayer(lw.wv, "attn_v.weight", WeightRole::AttentionV);
            addLayer(lw.wo, "attn_output.weight", WeightRole::AttentionWO);
            addLayer(lw.attn_norm, "attn_norm.weight", WeightRole::Norm);
            addLayer(lw.q_bias, "attn_q.bias", WeightRole::Bias);
            addLayer(lw.k_bias, "attn_k.bias", WeightRole::Bias);
            addLayer(lw.v_bias, "attn_v.bias", WeightRole::Bias);
            addLayer(lw.q_norm, "attn_q_norm.weight", WeightRole::Norm);
            addLayer(lw.k_norm, "attn_k_norm.weight", WeightRole::Norm);
            addLayer(lw.attn_qkv, "attn_qkv.weight", WeightRole::FusedQKV);
            addLayer(lw.attn_gate, "attn_gate.weight", WeightRole::GDNProjection);
            addLayer(lw.ssm_alpha, "ssm_alpha.weight", WeightRole::GDNAlphaBetaProjection);
            addLayer(lw.ssm_beta, "ssm_beta.weight", WeightRole::GDNAlphaBetaProjection);
            addLayer(lw.ssm_conv1d, "ssm_conv1d.weight", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_dt_bias, "ssm_dt.bias", WeightRole::Bias);
            addLayer(lw.ssm_a, "ssm_a", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_norm, "ssm_norm.weight", WeightRole::Norm);
            addLayer(lw.ssm_out, "ssm_out.weight", WeightRole::GDNProjection);
            addLayer(lw.gate_proj, "ffn_gate.weight", WeightRole::FFNGate);
            addLayer(lw.up_proj, "ffn_up.weight", WeightRole::FFNUp);
            addLayer(lw.down_proj, "ffn_down.weight", WeightRole::FFNDown);
            addLayer(lw.ffn_norm, "ffn_norm.weight", WeightRole::Norm);
            addLayer(lw.moe_gate, "ffn_gate_inp.weight", WeightRole::MoERouter);
            addLayer(lw.moe_gate_exps, "ffn_gate_exps.weight", WeightRole::MoEExpertGate);
            addLayer(lw.moe_up_exps, "ffn_up_exps.weight", WeightRole::MoEExpertUp);
            addLayer(lw.moe_down_exps, "ffn_down_exps.weight", WeightRole::MoEExpertDown);
            addLayer(lw.shared_expert_gate, "ffn_gate_shexp.weight", WeightRole::SharedExpertGate);
            addLayer(lw.shared_expert_up, "ffn_up_shexp.weight", WeightRole::SharedExpertUp);
            addLayer(lw.shared_expert_down, "ffn_down_shexp.weight", WeightRole::SharedExpertDown);
            addLayer(lw.shared_expert_gate_inp, "ffn_gate_inp_shexp.weight", WeightRole::SharedExpertInputGate);
        }

        cached_mtp_weight_bindings_.reset();
        cached_mtp_weight_bindings_source_ = nullptr;
        frozen_weight_set_ = std::make_unique<FrozenModelWeightSet>(
            strategy, builder.freezeBindings());
        frozen_weight_set_->validateForGraph();
        LOG_DEBUG("[DeviceGraphOrchestrator] FrozenModelWeightSet built with "
                  << frozen_weight_set_->bindings().size() << " bindings");
    }

    void DeviceGraphOrchestrator::setBuffers(const ModelBuffers &buffers)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set buffers: graph builder not initialized");
            return;
        }
        graph_builder_->setBuffers(buffers);
        managed_buffers_ = buffers;
        LOG_DEBUG("[DeviceGraphOrchestrator] Model buffers configured for full forward pass");
    }

    bool DeviceGraphOrchestrator::hasGlobalWeights() const
    {
        if (!graph_builder_)
        {
            return false;
        }
        // Check if the builder's isInitialized returns true AND we have global weights
        // isInitialized() checks get_layer_weights, but we also need embedding_table etc.
        // For now, rely on the graph builder's internal state
        return graph_builder_->isInitialized();
    }

    // =========================================================================
    // Graph Buffer Management (Phase 3 - moved from QwenStandardGraph)
    // =========================================================================

    bool DeviceGraphOrchestrator::initializeBuffers(int seq_len)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] initializeBuffers called but graph_builder not set");
            return false;
        }

        const auto &config = graph_builder_->config();
        if (!config.use_graph_buffer_management)
        {
            LOG_WARN("[DeviceGraphOrchestrator] initializeBuffers called but use_graph_buffer_management=false");
            return false;
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initializing buffers with graph management, seq_len=" << seq_len);

        // Get schema and resolver config from graph builder
        GraphSchema schema = graph_builder_->getSchema();
        GraphResolverConfig resolver_config = graph_builder_->getResolverConfig(seq_len);

        // Verify TensorFactory is set
        if (!tensor_factory_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] TensorFactory not set. Call setTensorFactory() before initializeBuffers()");
            return false;
        }

        // =====================================================================
        // Configure ArenaConfig for BufferArena allocation
        // =====================================================================
        ArenaConfig arena_config;
        arena_config.factory = tensor_factory_;
        arena_config.physical_memory_authority = physical_memory_authority_;

        /*
         * Snapshot publication does not alter activation placement. Selected
         * stage outputs are copied into the graph-owned VRAM snapshot arena and
         * acquired in one ordered D2H transaction after replay. This keeps
         * ordinary logits and scratch writes off GPU-visible host pages, whose
         * retry faults can overwhelm a ROCm secondary interrupt ring.
         */

        if (pipeline_maintenance_admission_)
            throw std::logic_error("Arena replacement cannot retain an original pipeline maintenance admission family");
        // Reinitialization replaces the arena and therefore invalidates every
        // non-owning typed view into its old allocation before the new one is
        // constructed.
        if (sparseQueryParentLaunched() || compiling_sparse_query_program_)
            throw std::logic_error("Arena replacement cannot retire a live sparse-query parent family");
        releaseRetainedStorageGenerationTopology();
        clearDeviceResidentLogicalSequenceStateMailbox();
        device_resident_logical_state_access_epoch_.clear();
        device_resident_logical_sequence_state_storage_.clear();
        device_generation_storage_.clear();
        active_device_generation_admission_.reset();
        device_generation_state_ready_ = {};
        last_device_generation_dispatch_ticket_.reset();
        hosted_device_generation_cursor_.reset();

        request_token_ids_dev_ = nullptr;
        request_position_ids_dev_ = nullptr;
        prefill_chunk_token_ids_dev_ = nullptr;
        prefill_chunk_position_ids_dev_ = nullptr;
        prefill_chunk_geometry_dev_ = nullptr;
        prefill_chunk_real_rows_dev_ = nullptr;
        prefill_chunk_row_stride_dev_ = nullptr;
        mtp_shifted_prefill_token_ids_dev_ = nullptr;
        mtp_shifted_prefill_position_ids_dev_ = nullptr;
        mtp_shifted_prefill_append_lengths_dev_ = nullptr;
        request_input_row_capacity_ = 0;
        prefill_chunk_row_capacity_ = 0;
        mtp_shifted_prefill_row_capacity_ = 0;
        request_padding_token_ids_host_.clear();
        request_setup_zero_rows_host_.clear();
        request_batch_geometry_layout_ = {};
        request_batch_geometry_dev_ = nullptr;
        request_sequence_lengths_dev_ = nullptr;
        request_row_stride_dev_ = nullptr;
        request_sequence_lengths_capacity_ = 0;
        request_sequence_lengths_active_count_ = 0;
        request_batch_geometry_host_.clear();
        request_state_reset_ready_ = {};
        request_input_admission_ready_ = {};
        request_input_reuse_ready_ = {};
        mtp_verifier_position_ids_dev_ = nullptr;
        mtp_verifier_request_lengths_dev_ = nullptr;

        // Create BufferArena with config
        arena_ = std::make_unique<BufferArena>(arena_config);

        // Admission and allocation resolve one physical root inventory. Alias
        // annotations do not merge storage; unused attention scratch is omitted
        // by the common resolver, and continuation mailboxes are explicit roots.
        const bool retained_mtp = retainsMTPGraphCapacity(config.mtp);
        const auto arena_layout = GraphArenaStorageLayout::resolve(schema, resolver_config, {
            .full_prefill_kv_handoff = config.dense_tp_enabled &&
                config.dense_tp_decode_replicated && config.qkv_column_parallel,
            .terminal_hidden_rows = retained_mtp
                ? static_cast<size_t>(resolveMTPTerminalHiddenRowCapacity(state_.batch_size, config.mtp)) : 0,
            .terminal_hidden_columns = retained_mtp ? static_cast<size_t>(config.residualWidth()) : 0});
        for (const auto &entry : arena_layout.entries())
        {
            if (!arena_->registerBuffer(entry.id, entry.descriptor))
                throw std::logic_error("Graph arena failed to register declared root: " + entry.descriptor.name);
            LOG_DEBUG("[DeviceGraphOrchestrator] Registered arena root: '" << entry.descriptor.name
                << "' -> " << bufferIdName(entry.id) << " [" << arena_->getRows(entry.id)
                << "x" << arena_->getCols(entry.id) << "]");
        }

        // =====================================================================
        // Register argmax partial-reduction scratch (GPU greedy sampling)
        // =====================================================================
        // The two-pass GPU argmax (sampleGreedyOnDevice) needs a small device
        // scratch holding one (value, index) partial per pass-1 block. We allocate
        // it through the arena (the V2 central buffer manager) instead of doing a
        // lazy cudaMalloc inside the backend. Capacity bounds the pass-1 grid;
        // 1024 partials far exceeds the ~74 blocks a 152K vocab needs.
        /*
         * Resolve MTP shape ownership for every backend before entering the
         * GPU-only allocation block. CPU grouped sidecars use the same graph
         * and row-selection helpers, so leaving these fields at historical
         * depth-one defaults silently reintroduced an M<=2 CPU limit.
         *
         * `mtp_max_verifier_rows_` is the per-request target row count. The
         * sidecar slot width is the flattened capture capacity across all
         * configured requests. Neither value is an implementation maximum:
         * changing runtime policy produces a correspondingly larger graph.
         */
        // The same declared execution projection is consumed by native memory admission.
        mtp_verifier_graph_family_ = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole());
        mtp_graph_owner_plan_ = MTPGraphOwnerPlan(config.mtp, mtpVerifierGraphFamily(),
            pp_stage_config_ && !pp_stage_config_->has_lm_head
                ? MTPGenerationParentRole::PipelineFollower : MTPGenerationParentRole::Controller,
            sparseQueryProgramCount());
        for (auto &cache : mtp_speculative_state_publication_graphs_) cache.invalidate();
        mtp_speculative_state_publication_graphs_.clear();
        mtp_speculative_state_publication_graphs_.resize(std::max(sparseQueryProgramCount(),
            mtp_graph_owner_plan_.speculativePublicationGraphSlots()));
        mtp_publication_selected_width_.fill(0u);
        for (auto &cache : mtp_stochastic_outcome_graphs_) cache.invalidate();
        for (auto &cache : mtp_stochastic_target_distribution_graphs_) cache.invalidate();
        mtp_stochastic_outcome_graphs_.clear();
        mtp_stochastic_target_distribution_graphs_.clear();
        const auto sampler_widths = std::max(std::size_t{1}, mtp_graph_owner_plan_.samplerHelperWidthCount());
        mtp_stochastic_outcome_graphs_.resize(sampler_widths);
        mtp_stochastic_target_distribution_graphs_.resize(sampler_widths);
        mtp_max_draft_depth_ = mtp_graph_owner_plan_.draftDepth();
        mtp_max_verifier_rows_ =
            mtp_graph_owner_plan_.verifierRowsPerRequest();
        for (auto &program : mtp_retained_verifier_signatures_)
        {
            program.clear();
            program.resize(static_cast<std::size_t>(mtp_max_verifier_rows_ - 1));
        }
        stochastic_batch_output_token_stride_ = mtp_max_verifier_rows_;
        stochastic_batch_output_request_capacity_ =
            mtp_graph_owner_plan_.requestCapacity();
        mtp_sidecar_condition_token_slot_width_ =
            mtp_graph_owner_plan_.flattenedTargetRows();
        mtp_sidecar_capture_layout_ =
            MTPSidecarCaptureLayout(mtp_sidecar_condition_token_slot_width_);
        mtp_terminal_hidden_contiguous_rows_select_caches_.clear();
        mtp_terminal_hidden_contiguous_rows_select_caches_.reserve(
            mtp_graph_owner_plan_.terminalHiddenContiguousRegistryCapacity());
        for (size_t slot = 0;
             slot <
                 mtp_graph_owner_plan_.terminalHiddenContiguousRegistryCapacity();
             ++slot)
        {
            mtp_terminal_hidden_contiguous_rows_select_caches_.push_back(
                std::make_unique<MTPTerminalHiddenRowsSelectGraphCache>());
        }
        mtp_terminal_hidden_device_accepted_rows_select_caches_.clear();
        mtp_terminal_hidden_device_accepted_rows_select_caches_.reserve(
            mtp_graph_owner_plan_
                .terminalHiddenDeviceAcceptedRegistryCapacity());
        for (size_t slot = 0;
             slot < mtp_graph_owner_plan_
                        .terminalHiddenDeviceAcceptedRegistryCapacity();
             ++slot)
        {
            mtp_terminal_hidden_device_accepted_rows_select_caches_.push_back(
                std::make_unique<MTPTerminalHiddenRowsSelectGraphCache>());
        }
        mtp_terminal_hidden_request_rows_select_caches_.clear();
        mtp_terminal_hidden_request_rows_select_caches_.reserve(
            mtp_graph_owner_plan_
                .terminalHiddenRequestTerminalRegistryCapacity());
        for (size_t slot = 0;
             slot < mtp_graph_owner_plan_
                        .terminalHiddenRequestTerminalRegistryCapacity();
             ++slot)
        {
            mtp_terminal_hidden_request_rows_select_caches_.push_back(
                std::make_unique<MTPTerminalHiddenRowsSelectGraphCache>());
        }
        mtp_shifted_prefill_hidden_rows_select_caches_.clear();
        const size_t shifted_prefill_selector_count =
            mtp_graph_owner_plan_
                .terminalHiddenShiftedPrefillRegistryCapacity();
        mtp_shifted_prefill_hidden_rows_select_caches_.reserve(
            shifted_prefill_selector_count);
        for (size_t cache_index = 0;
             cache_index < shifted_prefill_selector_count;
             ++cache_index)
        {
            mtp_shifted_prefill_hidden_rows_select_caches_.push_back(
                std::make_unique<MTPTerminalHiddenRowsSelectGraphCache>());
        }
        mtp_sidecar_depth0_kv_only_batch_caches_.clear();
        mtp_sidecar_depth0_kv_only_batch_caches_.reserve(
            mtp_graph_owner_plan_.sidecarKVOnlyBatchRegistryCapacity());
        for (size_t cache_index = 0;
             cache_index <
                 mtp_graph_owner_plan_.sidecarKVOnlyBatchRegistryCapacity();
             ++cache_index)
        {
            mtp_sidecar_depth0_kv_only_batch_caches_.push_back(
                std::make_unique<MTPSidecarGraphCache>());
        }

        if (state_.device_id.is_gpu())
        {
            /*
             * `initializeBuffers()` runs before the remainder of
             * `initializeInferenceStateFromArena()` publishes its final state
             * fields.  Size this persistent device owner from both immutable
             * capacities instead of relying on whichever transient batch size
             * happens to be visible during arena construction.  The MTP limit
             * is included explicitly because captured request-batched graphs
             * retain this address for their entire lifetime.
             */
            const size_t request_length_capacity =
                static_cast<size_t>(std::max(
                    {1,
                     state_.batch_size,
                     config.mtp.max_request_batch}));
            request_batch_geometry_layout_ =
                DeviceRequestBatchGeometryLayout(
                    static_cast<int>(request_length_capacity));
            if (!request_batch_geometry_layout_.valid())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to declare persistent request-batch geometry layout");
                return false;
            }
            /*
             * Request admission and graph activations have deliberately
             * different capacities. A long prompt must cross the host/device
             * boundary once in full, while each captured replay consumes only
             * one activation-width window. Conflating the two widths forced
             * the old scheduler to upload a new host slice per chunk.
             */
            const size_t active_request_capacity =
                static_cast<size_t>(std::max(1, state_.batch_size));
            const size_t request_input_capacity =
                active_request_capacity *
                static_cast<size_t>(
                    std::max({1, seq_len, state_.max_seq_len}));
            const size_t activation_input_capacity =
                active_request_capacity *
                static_cast<size_t>(std::max(1, seq_len));
            if (!arena_->registerBuffer(BufferId::REQUEST_TOKEN_IDS,
                                        1,
                                        request_input_capacity,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::REQUEST_POSITION_IDS,
                                        1,
                                        request_input_capacity,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::PREFILL_CHUNK_TOKEN_IDS,
                    1,
                    activation_input_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::PREFILL_CHUNK_POSITION_IDS,
                    1,
                    activation_input_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::PREFILL_CHUNK_GEOMETRY,
                    1,
                    2,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS,
                    1,
                    activation_input_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_SHIFTED_PREFILL_POSITION_IDS,
                    1,
                    activation_input_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_SHIFTED_PREFILL_APPEND_LENGTHS,
                    1,
                    request_length_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(BufferId::REQUEST_BATCH_GEOMETRY,
                                        1,
                                        request_batch_geometry_layout_.scalarCount(),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                    DeviceResidentLogicalSequenceStateStorage::kFieldCount,
                    request_length_capacity,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_GENERATION_RESPONSE_TOKENS,
                    request_length_capacity,
                    static_cast<size_t>(state_.max_seq_len),
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_GENERATION_CONTROL,
                    request_length_capacity,
                    sampling_math::kDeviceGenerationControlCount,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_GENERATION_DISPATCH_TICKETS,
                    request_length_capacity,
                    sizeof(sampling_math::DeviceGenerationDispatchTicket) /
                        sizeof(int32_t),
                    "INT32",
                    state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register persistent device request-state buffers");
                return false;
            }
            if(const auto terminalBytes=localHotTerminalMetadataBytes(request_length_capacity)){
                if(!arena_->registerBuffer(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION,1,terminalBytes/sizeof(int),"INT32",state_.device_id))
                    throw std::logic_error("Local terminal observation could not declare its exact separate arena root");
            }
            mtp_diagnostic_record_mode_ = sampling_math::resolveMTPDiagnosticRecordMode(
                debugEnv().runtime_debug.mtp_device_phase_snapshots,
                DebugEnv::isTruthyEnv("LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS"),
                debugEnv().runtime_debug.mtp_depth_transaction_trace);
            const bool device_logical_state_diagnostics_requested =
                debugEnv().runtime_debug.mtp_device_phase_snapshots ||
                PrefixProbeCapturePolicy::fromEnvironment()
                    .capture_device_logical_state;
            if (device_logical_state_diagnostics_requested &&
                !arena_->registerBuffer(
                    BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS,
                    DeviceResidentLogicalStateDiagnosticStorage::kRowCount,
                    request_length_capacity,
                    "INT32",
                    state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register device-only MTP logical-state phase snapshots");
                return false;
            }
            if (config.vocab_size <= 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic buffers require a positive vocab size");
                return false;
            }
            IBackend *sampling_backend = getBackendFor(state_.device_id);
            if (!sampling_backend ||
                !sampling_backend->prepareLogitPenaltyWorkspace(
                    config.vocab_size,
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to prepare persistent GPU logit-penalty workspace");
                return false;
            }
            const size_t stochastic_vocab_cols =
                static_cast<size_t>(config.vocab_size);
            /*
             * Target slots include the bonus row for every request, while
             * draft slots pack only the compared speculative rows.  Keeping
             * these capacities separate avoids both stale scalar scratch and
             * unnecessary bonus-sized draft buffers.
             */
            stochastic_target_row_capacity_ =
                mtp_graph_owner_plan_.stochasticTargetRows();
            stochastic_draft_row_capacity_ =
                mtp_graph_owner_plan_.stochasticDraftRows();
            stochastic_target_top_k_.assign(
                static_cast<size_t>(stochastic_target_row_capacity_),
                0);
            stochastic_draft_top_k_.assign(
                static_cast<size_t>(stochastic_draft_row_capacity_),
                0);
            stochastic_target_row_formats_.assign(
                static_cast<size_t>(stochastic_target_row_capacity_),
                StochasticRowFormat::Empty);
            stochastic_draft_row_formats_.assign(
                static_cast<size_t>(stochastic_draft_row_capacity_),
                StochasticRowFormat::Empty);
            stochastic_target_distribution_streams_.assign(
                static_cast<size_t>(stochastic_target_row_capacity_),
                nullptr);
            stochastic_draft_distribution_streams_.assign(
                static_cast<size_t>(stochastic_draft_row_capacity_),
                nullptr);
            stochastic_target_sample_ready_.assign(
                static_cast<size_t>(stochastic_target_row_capacity_),
                StochasticSampleReadyState{});
            stochastic_draft_sample_ready_.assign(
                static_cast<size_t>(stochastic_draft_row_capacity_),
                StochasticSampleReadyState{});
            mtp_direct_token_history_graph_.invalidate();
            mtp_draft_token_publication_graphs_.clear();
            mtp_draft_token_publication_graphs_.reserve(
                mtp_graph_owner_plan_.draftPublicationRegistryCapacity());
            for (size_t slot = 0;
                 slot < mtp_graph_owner_plan_.draftPublicationRegistryCapacity();
                 ++slot)
            {
                mtp_draft_token_publication_graphs_.push_back(
                    std::make_unique<
                        MTPDraftTokenPublicationGraphCache>());
            }
            mtp_verifier_preparation_graphs_.clear();
            invalidateSparseQueryParents();
            const size_t verifier_preparation_registry_size =
                mtp_graph_owner_plan_.retainsGraphCapacity()
                    ? mtp_graph_owner_plan_
                          .verifierPreparationRegistryCapacity()
                    : kMTPVerifierPreparationControlPolicyCount *
                          static_cast<size_t>(
                              stochastic_batch_output_request_capacity_) *
                          static_cast<size_t>(mtp_max_verifier_rows_);
            mtp_verifier_preparation_graphs_.reserve(
                verifier_preparation_registry_size);
            for (size_t registry_index = 0;
                 registry_index < verifier_preparation_registry_size;
                 ++registry_index)
            {
                mtp_verifier_preparation_graphs_.push_back(
                    std::make_unique<
                        MTPVerifierPreparationGraphCache>());
            }
            mtp_verifier_preparation_token_row_scratch_.clear();
            mtp_verifier_preparation_token_row_scratch_.reserve(
                static_cast<size_t>(
                    stochastic_batch_output_request_capacity_));
            mtp_verifier_preparation_checkpoint_scratch_.clear();
            mtp_verifier_preparation_checkpoint_scratch_.reserve(
                static_cast<size_t>(
                    stochastic_batch_output_request_capacity_));
            /*
             * Reserve the largest retained semantic family for allocation-free
             * parent composition. HIP needs every ticket-selected branch;
             * native CUDA uses only the compact maximum-width linear subset.
             * These descriptors do not own native executables and therefore
             * never enter graph-driver memory admission.
             */
            const size_t device_generation_dynamic_fragment_capacity =
                resolveMTPRetainedDeviceGenerationFragmentCapacity(
                    mtp_max_draft_depth_);
            mtp_device_generation_loop_fragment_scratch_.clear();
            mtp_device_generation_loop_fragment_scratch_.reserve(
                device_generation_dynamic_fragment_capacity);
            for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
            {
                auto &parent = sparse_query_generation_parents_[program];
                parent.source_fragments.clear();
                parent.source_fragments.reserve(device_generation_dynamic_fragment_capacity);
            }
            mtp_device_generation_loop_hosted_fragment_scratch_.clear();
            mtp_device_generation_loop_hosted_fragment_scratch_.reserve(
                device_generation_dynamic_fragment_capacity);
            for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
            {
                auto &parent = sparse_query_generation_parents_[program];
                parent.hosted_fragments.clear();
                parent.hosted_fragments.reserve(device_generation_dynamic_fragment_capacity);
            }
            const size_t stochastic_topk_partial_capacity =
                static_cast<size_t>(stochastic_target_row_capacity_) *
                kStochasticTopKPartialBlocks *
                kStochasticTopKSmallKCap;
            const size_t stochastic_draft_topk_partial_capacity =
                static_cast<size_t>(stochastic_draft_row_capacity_) *
                kStochasticTopKPartialBlocks *
                kStochasticTopKSmallKCap;
            constexpr size_t kArgmaxPartialCapacity = 1024;
            if (!arena_->registerBuffer(BufferId::ARGMAX_PARTIAL_VALS, 1, kArgmaxPartialCapacity,
                                        "FP32", state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register ARGMAX_PARTIAL_VALS buffer");
                return false;
            }
            if (!arena_->registerBuffer(BufferId::ARGMAX_PARTIAL_IDXS, 1, kArgmaxPartialCapacity,
                                        "INT32", state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register ARGMAX_PARTIAL_IDXS buffer");
                return false;
            }
            if (!arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_TOKEN_IDS,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        kStochasticDistributionMaxK,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_PROBS,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        kStochasticDistributionMaxK,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS,
                                        static_cast<size_t>(stochastic_draft_row_capacity_),
                                        kStochasticDistributionMaxK,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_PROBS,
                                        static_cast<size_t>(stochastic_draft_row_capacity_),
                                        kStochasticDistributionMaxK,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_PROCESSED_LOGITS,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        stochastic_vocab_cols,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES,
                                        static_cast<size_t>(stochastic_draft_row_capacity_),
                                        stochastic_vocab_cols,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS,
                                        1,
                                        static_cast<size_t>(stochastic_draft_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS,
                                        1,
                                        static_cast<size_t>(stochastic_draft_row_capacity_),
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_CONDITION_TOKEN,
                                        1,
                                        static_cast<size_t>(
                                            mtp_sidecar_condition_token_slot_width_) *
                                            static_cast<size_t>(
                                                mtp_sidecar_capture_layout_
                                                    .conditionTokenSlotCount()),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_POSITION_IDS,
                                        1,
                                        static_cast<size_t>(
                                            mtp_sidecar_condition_token_slot_width_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_VERIFIER_INPUT_TOKENS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_VERIFIER_STOP_TOKENS,
                    1,
                    sampling_math::kSpeculativeBatchMaxStopTokens,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_GREEDY_PENALTY_POLICY,
                    1,
                    static_cast<size_t>(kMTPGreedyPenaltyPolicyWords),
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_GENERATED_TOKEN_COUNTS,
                    static_cast<size_t>(std::max(
                        state_.batch_size, stochastic_batch_output_request_capacity_)),
                    stochastic_vocab_cols,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::SAMPLING_REQUEST_SEEDS,
                    GenerationRequestSeedGeometry(state_.batch_size,
                        stochastic_batch_output_request_capacity_).requests(),
                    GenerationRequestSeedGeometry::words_per_request,
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_VERIFIER_POSITION_IDS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_VERIFIER_REQUEST_LENGTHS,
                    1,
                    static_cast<size_t>(
                        stochastic_batch_output_request_capacity_),
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS,
                                        1,
                                        stochastic_topk_partial_capacity,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS,
                                        1,
                                        stochastic_topk_partial_capacity,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS,
                                        1,
                                        stochastic_draft_topk_partial_capacity,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS,
                                        1,
                                        stochastic_draft_topk_partial_capacity,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_TOKENS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_ACCEPTED,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_THRESHOLDS,
                                        1,
                                        static_cast<size_t>(stochastic_target_row_capacity_),
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS,
                                        static_cast<size_t>(stochastic_batch_output_request_capacity_),
                                        static_cast<size_t>(
                                            stochastic_batch_output_token_stride_),
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_BATCH_OUTPUT_META,
                                        static_cast<size_t>(stochastic_batch_output_request_capacity_),
                                        sampling_math::kSpeculativeBatchMetaCount,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC,
                    1,
                    sizeof(sampling_math::MTPDiagnosticRecordStorage) /
                        sizeof(int32_t),
                    "INT32",
                    state_.device_id) ||
                !arena_->registerBuffer(
                    BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                    static_cast<size_t>(
                        stochastic_batch_output_request_capacity_),
                    sizeof(
                        sampling_math::
                            MTPCommittedVerifierIdentityRecord) /
                        sizeof(int32_t),
                    "INT32",
                    state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register stochastic MTP sampling buffers");
                return false;
            }
        }

        // =====================================================================
        // Allocate all registered buffers
        // =====================================================================
        logOrchestratorVramTrace(state_.device_id, "arena.before_allocate");
        if (!arena_->allocate())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] BufferArena allocation failed");
            return false;
        }
        logOrchestratorVramTrace(state_.device_id, "arena.after_allocate");

        /*
         * Resolve the capacity-complete terminal-hidden owner immediately
         * after arena allocation.  CPU stages retain host addresses just as
         * captured GPU nodes retain device addresses, so both backends obey
         * the same immutable-owner lifetime even though only GPU needs an
         * explicit device-storage allocation here.
         */
        if (retainsMTPGraphCapacity(graph_builder_->config().mtp))
        {
            if (!arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP requires an arena-owned terminal-hidden archive");
                return false;
            }
            if (state_.device_id.is_gpu() &&
                !arena_->allocateDeviceStorage(
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to reserve device storage for the MTP terminal-hidden archive");
                return false;
            }

            state_.prefix_terminal_hidden =
                arena_->getSharedTensor(BufferId::PREFIX_TERMINAL_HIDDEN);
            const auto terminal_hidden_device =
                state_.prefix_terminal_hidden
                    ? state_.prefix_terminal_hidden->current_device()
                    : std::optional<DeviceId>{};
            const bool storage_ready =
                state_.prefix_terminal_hidden &&
                (state_.device_id.is_gpu()
                     ? state_.prefix_terminal_hidden->gpu_data_ptr() != nullptr &&
                           terminal_hidden_device.has_value() &&
                           *terminal_hidden_device == state_.device_id
                     : state_.prefix_terminal_hidden->raw_data() != nullptr);
            if (!storage_ready ||
                state_.prefix_terminal_hidden->isMapped() ||
                state_.prefix_terminal_hidden->rows() <
                    static_cast<size_t>(
                        resolveMTPTerminalHiddenRowCapacity(
                            state_.batch_size,
                            graph_builder_->config().mtp)) ||
                state_.prefix_terminal_hidden->cols() <
                    static_cast<size_t>(state_.d_model))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to resolve the persistent MTP terminal-hidden archive");
                return false;
            }
            state_.mtp_terminal_hidden_publication.invalidate();
        }

        /*
         * Typed GPU MTP graphs bind every scratch address before graph
         * construction starts. Materialize the complete schema-owned family
         * while allocation is still legal so main prefill, grouped verifier,
         * and live condition captures all retain stable arena addresses.
         *
         * Allocation alone deliberately leaves each tensor UNINITIALIZED. Its
         * captured producer publishes device authority on the exact execution
         * stream after writing; initialization must never claim that unwritten
         * scratch bytes are coherent. Optional topology-specific members are
         * absent from the schema and therefore intentionally skipped here.
         */
        if (state_.device_id.is_gpu() &&
            retainsMTPGraphCapacity(graph_builder_->config().mtp))
        {
            // Canonical terminal rows are registered arena roots too. Materialize
            // their original admitted storage here, before selectors capture
            // addresses; workspace sizing alone does not materialize tensors.
            constexpr std::array<BufferId, 24> kMTPGraphTensorFamily{
                BufferId::LOGITS,
                BufferId::LOGITS_LOCAL,
                BufferId::MTP_EMBEDDING,
                BufferId::MTP_NORM_HIDDEN,
                BufferId::MTP_NORM_EMBEDDING,
                BufferId::MTP_CONCAT,
                BufferId::MTP_PROJECTED,
                BufferId::MTP_HIDDEN,
                BufferId::MTP_Q_PROJ,
                BufferId::MTP_K_PROJ,
                BufferId::MTP_V_PROJ,
                BufferId::MTP_K_FULL_PREFILL,
                BufferId::MTP_V_FULL_PREFILL,
                BufferId::MTP_FA_Q_RAW,
                BufferId::MTP_FA_GATE,
                BufferId::MTP_Q_ROPE,
                BufferId::MTP_K_ROPE,
                BufferId::MTP_ATTN_OUTPUT,
                BufferId::MTP_ATTN_PROJ,
                BufferId::MTP_GATE_PROJ,
                BufferId::MTP_UP_PROJ,
                BufferId::MTP_FFN_OUTPUT,
                BufferId::MTP_LOGITS,
                BufferId::MTP_LOGITS_GATHERED,
            };
            for (const BufferId id : kMTPGraphTensorFamily)
            {
                if (!arena_->isRegistered(id))
                    continue;
                arena_->allocateDeviceStorage(id, state_.device_id);
                const auto tensor = arena_->getSharedTensor(id);
                const auto tensor_device =
                    tensor ? tensor->current_device()
                           : std::optional<DeviceId>{};
                if (!tensor || tensor->isMapped() ||
                    !tensor->gpu_data_ptr() ||
                    !tensor_device.has_value() ||
                    *tensor_device != state_.device_id)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to reserve device-local MTP graph tensor storage for "
                              << bufferIdName(id));
                    return false;
                }
            }
            if (!arena_->isRegistered(BufferId::MTP_LOGITS))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP requires a schema-owned MTP_LOGITS capacity");
                return false;
            }
            const auto mtp_logits =
                arena_->getSharedTensor(BufferId::MTP_LOGITS);
            const auto logits_device =
                mtp_logits ? mtp_logits->current_device()
                           : std::optional<DeviceId>{};
            if (!mtp_logits ||
                mtp_logits->isMapped() ||
                !mtp_logits->gpu_data_ptr() ||
                !logits_device.has_value() ||
                *logits_device != state_.device_id)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to reserve device-local MTP logits output storage");
                return false;
            }

        }

        /*
         * Resolve every external-request input owner before any graph is built.
         * Their addresses remain stable for the arena lifetime, so captured
         * embedding, RoPE, short-conv, and GDN stages can retain them directly.
         */
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::REQUEST_TOKEN_IDS) &&
            arena_->isRegistered(BufferId::REQUEST_POSITION_IDS) &&
            arena_->isRegistered(BufferId::REQUEST_BATCH_GEOMETRY))
        {
            arena_->allocateDeviceStorage(
                BufferId::REQUEST_TOKEN_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::REQUEST_POSITION_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::PREFILL_CHUNK_TOKEN_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::PREFILL_CHUNK_POSITION_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::PREFILL_CHUNK_GEOMETRY,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_SHIFTED_PREFILL_POSITION_IDS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_SHIFTED_PREFILL_APPEND_LENGTHS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::REQUEST_BATCH_GEOMETRY,
                state_.device_id);
            request_token_ids_dev_ = arena_->getDevicePtr(
                BufferId::REQUEST_TOKEN_IDS,
                state_.device_id);
            request_position_ids_dev_ = arena_->getDevicePtr(
                BufferId::REQUEST_POSITION_IDS,
                state_.device_id);
            prefill_chunk_token_ids_dev_ = arena_->getDevicePtr(
                BufferId::PREFILL_CHUNK_TOKEN_IDS,
                state_.device_id);
            prefill_chunk_position_ids_dev_ = arena_->getDevicePtr(
                BufferId::PREFILL_CHUNK_POSITION_IDS,
                state_.device_id);
            prefill_chunk_geometry_dev_ = arena_->getDevicePtr(
                BufferId::PREFILL_CHUNK_GEOMETRY,
                state_.device_id);
            prefill_chunk_real_rows_dev_ =
                static_cast<int32_t *>(prefill_chunk_geometry_dev_);
            prefill_chunk_row_stride_dev_ =
                prefill_chunk_real_rows_dev_
                    ? prefill_chunk_real_rows_dev_ + 1
                    : nullptr;
            mtp_shifted_prefill_token_ids_dev_ = arena_->getDevicePtr(
                BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS,
                state_.device_id);
            mtp_shifted_prefill_position_ids_dev_ = arena_->getDevicePtr(
                BufferId::MTP_SHIFTED_PREFILL_POSITION_IDS,
                state_.device_id);
            mtp_shifted_prefill_append_lengths_dev_ = arena_->getDevicePtr(
                BufferId::MTP_SHIFTED_PREFILL_APPEND_LENGTHS,
                state_.device_id);
            request_input_row_capacity_ = static_cast<int>(
                arena_->getCols(BufferId::REQUEST_TOKEN_IDS));
            prefill_chunk_row_capacity_ = static_cast<int>(
                arena_->getCols(BufferId::PREFILL_CHUNK_TOKEN_IDS));
            mtp_shifted_prefill_row_capacity_ = static_cast<int>(
                arena_->getCols(
                    BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS));
            request_padding_token_ids_host_.assign(
                static_cast<size_t>(
                    std::max(0, prefill_chunk_row_capacity_)),
                static_cast<int32_t>(
                    debugEnv().execution.prefill_graph_pad_token_id));
            request_setup_zero_rows_host_.assign(
                static_cast<size_t>(
                    std::max(0, prefill_chunk_row_capacity_)),
                int32_t{0});
            request_batch_geometry_dev_ = arena_->getDevicePtr(
                BufferId::REQUEST_BATCH_GEOMETRY,
                state_.device_id);
            request_sequence_lengths_dev_ =
                request_batch_geometry_layout_.sequenceLengths(
                    static_cast<int32_t *>(request_batch_geometry_dev_));
            request_row_stride_dev_ =
                request_batch_geometry_layout_.rowStride(
                    static_cast<int32_t *>(request_batch_geometry_dev_));
            request_sequence_lengths_capacity_ =
                request_batch_geometry_layout_.requestCapacity();
            request_batch_geometry_host_.assign(
                request_batch_geometry_layout_.scalarCount(),
                0);
            if (!request_token_ids_dev_ ||
                !request_position_ids_dev_ ||
                !prefill_chunk_token_ids_dev_ ||
                !prefill_chunk_position_ids_dev_ ||
                !prefill_chunk_geometry_dev_ ||
                !prefill_chunk_real_rows_dev_ ||
                !prefill_chunk_row_stride_dev_ ||
                !mtp_shifted_prefill_token_ids_dev_ ||
                !mtp_shifted_prefill_position_ids_dev_ ||
                !mtp_shifted_prefill_append_lengths_dev_ ||
                request_input_row_capacity_ <= 0 ||
                prefill_chunk_row_capacity_ <= 0 ||
                mtp_shifted_prefill_row_capacity_ <= 0 ||
                request_padding_token_ids_host_.size() !=
                    static_cast<size_t>(prefill_chunk_row_capacity_) ||
                request_setup_zero_rows_host_.size() !=
                    static_cast<size_t>(prefill_chunk_row_capacity_) ||
                arena_->getCols(BufferId::REQUEST_POSITION_IDS) !=
                    static_cast<size_t>(request_input_row_capacity_) ||
                arena_->getCols(BufferId::PREFILL_CHUNK_POSITION_IDS) !=
                    static_cast<size_t>(prefill_chunk_row_capacity_) ||
                arena_->getCols(BufferId::PREFILL_CHUNK_GEOMETRY) != 2 ||
                arena_->getCols(BufferId::MTP_SHIFTED_PREFILL_TOKEN_IDS) !=
                    static_cast<size_t>(
                        mtp_shifted_prefill_row_capacity_) ||
                arena_->getCols(BufferId::MTP_SHIFTED_PREFILL_POSITION_IDS) !=
                    static_cast<size_t>(
                        mtp_shifted_prefill_row_capacity_) ||
                mtp_shifted_prefill_row_capacity_ !=
                    prefill_chunk_row_capacity_ ||
                arena_->getCols(BufferId::MTP_SHIFTED_PREFILL_APPEND_LENGTHS) !=
                    static_cast<size_t>(request_sequence_lengths_capacity_) ||
                !request_batch_geometry_dev_ ||
                !request_sequence_lengths_dev_ ||
                !request_row_stride_dev_ ||
                request_sequence_lengths_capacity_ <= 0 ||
                arena_->getCols(BufferId::REQUEST_BATCH_GEOMETRY) !=
                    request_batch_geometry_layout_.scalarCount() ||
                request_batch_geometry_host_.size() !=
                    request_batch_geometry_layout_.scalarCount())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to resolve persistent device request-input buffers");
                return false;
            }

            IBackend *backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Request-input admission could not resolve its GPU backend");
                return false;
            }
            const int device_ordinal = state_.device_id.gpu_ordinal();
            auto make_request_input_event =
                [backend, device_ordinal]() -> std::shared_ptr<void>
            {
                void *raw_event = backend->createEvent(device_ordinal);
                if (!raw_event)
                    return {};
                return std::shared_ptr<void>(
                    raw_event,
                    [backend, device_ordinal](void *event)
                    {
                        if (event)
                            backend->destroyEvent(event, device_ordinal);
                    });
            };

            request_input_admission_ready_.event =
                make_request_input_event();
            request_input_reuse_ready_.event =
                make_request_input_event();
            device_generation_state_ready_.event =
                make_request_input_event();
            request_state_reset_ready_.event =
                make_request_input_event();
            main_graph_build_device_state_ready_.event =
                make_request_input_event();
            mtp_graph_build_device_state_ready_.event =
                make_request_input_event();
            moe_overlay_initial_runtime_source_ready_event_ =
                make_request_input_event();
            moe_overlay_initial_runtime_published_event_ =
                make_request_input_event();
            if (!request_input_admission_ready_.event ||
                !request_input_reuse_ready_.event ||
                !request_state_reset_ready_.event ||
                !main_graph_build_device_state_ready_.event ||
                !mtp_graph_build_device_state_ready_.event ||
                !moe_overlay_initial_runtime_source_ready_event_ ||
                !moe_overlay_initial_runtime_published_event_ ||
                !device_generation_state_ready_.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate request-boundary and graph-build ordering events");
                return false;
            }
        }

        /*
         * Resolve the durable logical-state owner independently of stochastic
         * verifier scratch. Metadata derivation and request-batch prefill write
         * these rows directly, and every later mailbox only publishes this
         * stable address. The arena outlives all graph-workspace reallocations.
         */
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::MTP_LOGICAL_SEQUENCE_STATE))
        {
            arena_->allocateDeviceStorage(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                state_.device_id);
            void *logical_state_base = arena_->getDevicePtr(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                state_.device_id);
            const size_t logical_state_rows = arena_->getRows(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE);
            const int logical_state_capacity = static_cast<int>(arena_->getCols(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE));
            if (logical_state_rows !=
                    DeviceResidentLogicalSequenceStateStorage::kFieldCount ||
                !device_resident_logical_sequence_state_storage_.bind(
                    logical_state_base,
                    logical_state_capacity))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind arena-owned MTP logical sequence state");
                return false;
            }
        }
        if(state_.device_id.is_gpu() && arena_->isRegistered(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION))
            arena_->allocateDeviceStorage(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION,state_.device_id);
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::MTP_GENERATION_RESPONSE_TOKENS) &&
            arena_->isRegistered(BufferId::MTP_GENERATION_CONTROL) &&
            arena_->isRegistered(
                BufferId::MTP_GENERATION_DISPATCH_TICKETS))
        {
            arena_->allocateDeviceStorage(
                BufferId::MTP_GENERATION_RESPONSE_TOKENS,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_GENERATION_CONTROL,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_GENERATION_DISPATCH_TICKETS,
                state_.device_id);
            if (!device_generation_storage_.bind(
                    arena_->getDevicePtr(
                        BufferId::MTP_GENERATION_RESPONSE_TOKENS,
                        state_.device_id),
                    static_cast<int>(arena_->getRows(
                        BufferId::MTP_GENERATION_RESPONSE_TOKENS)),
                    static_cast<int>(arena_->getCols(
                        BufferId::MTP_GENERATION_RESPONSE_TOKENS)),
                    arena_->getDevicePtr(
                        BufferId::MTP_GENERATION_CONTROL,
                        state_.device_id),
                    static_cast<int>(arena_->getRows(
                        BufferId::MTP_GENERATION_CONTROL)),
                    static_cast<int>(arena_->getCols(
                        BufferId::MTP_GENERATION_CONTROL)),
                    arena_->getDevicePtr(
                        BufferId::MTP_GENERATION_DISPATCH_TICKETS,
                        state_.device_id),
                    static_cast<int>(arena_->getRows(
                        BufferId::MTP_GENERATION_DISPATCH_TICKETS)),
                    static_cast<int>(arena_->getCols(
                        BufferId::MTP_GENERATION_DISPATCH_TICKETS))))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind persistent device generation response/controller/ticket rows");
                return false;
            }
        }
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS))
        {
            arena_->allocateDeviceStorage(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS,
                state_.device_id);
            void *snapshot_base = arena_->getDevicePtr(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS,
                state_.device_id);
            const size_t snapshot_rows = arena_->getRows(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS);
            const int snapshot_capacity = static_cast<int>(arena_->getCols(
                BufferId::MTP_LOGICAL_SEQUENCE_STATE_DIAGNOSTIC_SNAPSHOTS));
            if (snapshot_rows !=
                    DeviceResidentLogicalStateDiagnosticStorage::kRowCount ||
                !device_resident_logical_state_diagnostic_storage_.bind(
                    snapshot_base,
                    snapshot_capacity))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind device-only MTP logical-state phase snapshots");
                return false;
            }
        }

        // Resolve the argmax partial scratch device pointers once, up front, so
        // the per-decode-step greedy-sampling hot path never touches the arena.
        // prepareForWrite forces device-side allocation; the buffers are pure
        // scratch (overwritten every call) so no coherence tracking is needed.
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::ARGMAX_PARTIAL_VALS) &&
            arena_->isRegistered(BufferId::ARGMAX_PARTIAL_IDXS))
        {
            arena_->allocateDeviceStorage(BufferId::ARGMAX_PARTIAL_VALS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::ARGMAX_PARTIAL_IDXS, state_.device_id);
            argmax_partial_vals_dev_ = arena_->getDevicePtr(BufferId::ARGMAX_PARTIAL_VALS, state_.device_id);
            argmax_partial_idxs_dev_ = arena_->getDevicePtr(BufferId::ARGMAX_PARTIAL_IDXS, state_.device_id);
            if (argmax_partial_vals_dev_ && argmax_partial_idxs_dev_)
                argmax_partial_capacity_ = static_cast<int>(arena_->getCols(BufferId::ARGMAX_PARTIAL_VALS));
        }

        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_TOKEN_IDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_PROCESSED_LOGITS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS) &&
            arena_->isRegistered(BufferId::MTP_CONDITION_TOKEN) &&
            arena_->isRegistered(BufferId::MTP_POSITION_IDS) &&
            arena_->isRegistered(BufferId::MTP_VERIFIER_INPUT_TOKENS) &&
            arena_->isRegistered(BufferId::MTP_VERIFIER_STOP_TOKENS) &&
            arena_->isRegistered(BufferId::MTP_GREEDY_PENALTY_POLICY) &&
            arena_->isRegistered(BufferId::MTP_GENERATED_TOKEN_COUNTS) &&
            arena_->isRegistered(BufferId::SAMPLING_REQUEST_SEEDS) &&
            arena_->isRegistered(BufferId::MTP_VERIFIER_POSITION_IDS) &&
            arena_->isRegistered(BufferId::MTP_VERIFIER_REQUEST_LENGTHS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_ACCEPTED) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_THRESHOLDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_BATCH_OUTPUT_META) &&
            arena_->isRegistered(
                BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC) &&
            arena_->isRegistered(
                BufferId::MTP_COMMITTED_VERIFIER_IDENTITY))
        {
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_TARGET_TOKEN_IDS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_TARGET_PROBS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_PROBS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_PROCESSED_LOGITS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::MTP_CONDITION_TOKEN, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::MTP_POSITION_IDS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::MTP_VERIFIER_INPUT_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::MTP_VERIFIER_STOP_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_GREEDY_PENALTY_POLICY,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_GENERATED_TOKEN_COUNTS,
                state_.device_id);
            arena_->allocateDeviceStorage(BufferId::SAMPLING_REQUEST_SEEDS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::MTP_VERIFIER_POSITION_IDS, state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_VERIFIER_REQUEST_LENGTHS,
                state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_VERIFY_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_VERIFY_ACCEPTED, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_VERIFY_THRESHOLDS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS, state_.device_id);
            arena_->allocateDeviceStorage(BufferId::STOCHASTIC_BATCH_OUTPUT_META, state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC,
                state_.device_id);
            arena_->allocateDeviceStorage(
                BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                state_.device_id);

            stochastic_target_token_ids_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_TOKEN_IDS, state_.device_id);
            stochastic_target_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_PROBS, state_.device_id);
            stochastic_draft_token_ids_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS, state_.device_id);
            stochastic_draft_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_PROBS, state_.device_id);
            stochastic_processed_logits_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_PROCESSED_LOGITS, state_.device_id);
            stochastic_inverse_rejection_samples_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES, state_.device_id);
            stochastic_target_sample_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id);
            stochastic_draft_sample_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS, state_.device_id);
            stochastic_draft_sample_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS, state_.device_id);
            mtp_sidecar_condition_token_dev_ =
                arena_->getDevicePtr(BufferId::MTP_CONDITION_TOKEN, state_.device_id);
            mtp_sidecar_condition_token_capacity_ =
                static_cast<int>(arena_->getCols(BufferId::MTP_CONDITION_TOKEN));
            mtp_sidecar_position_ids_dev_ =
                arena_->getDevicePtr(BufferId::MTP_POSITION_IDS, state_.device_id);
            mtp_verifier_input_tokens_dev_ =
                arena_->getDevicePtr(BufferId::MTP_VERIFIER_INPUT_TOKENS, state_.device_id);
            mtp_verifier_stop_tokens_dev_ =
                arena_->getDevicePtr(
                    BufferId::MTP_VERIFIER_STOP_TOKENS,
                    state_.device_id);
            mtp_greedy_penalty_policy_dev_ =
                arena_->getDevicePtr(
                    BufferId::MTP_GREEDY_PENALTY_POLICY,
                    state_.device_id);
            mtp_generated_token_counts_dev_ =
                arena_->getDevicePtr(
                    BufferId::MTP_GENERATED_TOKEN_COUNTS,
                    state_.device_id);
            mtp_generated_token_count_capacity_ =
                static_cast<int>(
                    arena_->getCols(
                        BufferId::MTP_GENERATED_TOKEN_COUNTS));
            mtp_verifier_position_ids_dev_ =
                arena_->getDevicePtr(BufferId::MTP_VERIFIER_POSITION_IDS, state_.device_id);
            mtp_verifier_request_lengths_dev_ =
                arena_->getDevicePtr(
                    BufferId::MTP_VERIFIER_REQUEST_LENGTHS,
                    state_.device_id);
            stochastic_topk_partial_vals_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS, state_.device_id);
            stochastic_topk_partial_idxs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS, state_.device_id);
            if (stochastic_topk_partial_vals_dev_ && stochastic_topk_partial_idxs_dev_)
                stochastic_topk_partial_capacity_ =
                    static_cast<int>(arena_->getCols(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS));
            stochastic_draft_topk_partial_vals_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS, state_.device_id);
            stochastic_draft_topk_partial_idxs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_IDXS, state_.device_id);
            if (stochastic_draft_topk_partial_vals_dev_ &&
                stochastic_draft_topk_partial_idxs_dev_)
                stochastic_draft_topk_partial_capacity_ =
                    static_cast<int>(arena_->getCols(BufferId::STOCHASTIC_DRAFT_TOPK_PARTIAL_VALS));
            stochastic_verify_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_TOKENS, state_.device_id);
            stochastic_verify_accepted_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_ACCEPTED, state_.device_id);
            stochastic_verify_accept_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS, state_.device_id);
            stochastic_verify_thresholds_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_THRESHOLDS, state_.device_id);
            stochastic_batch_output_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS, state_.device_id);
            stochastic_batch_output_meta_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_BATCH_OUTPUT_META, state_.device_id);
            mtp_diagnostic_record_storage_dev_ = static_cast<sampling_math::MTPDiagnosticRecordStorage *>(
                arena_->getDevicePtr(BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC, state_.device_id));
            if (!mtp_diagnostic_record_storage_dev_ ||
                arena_->getRows(BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC) != 1 ||
                arena_->getCols(BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC) * sizeof(int32_t) != sizeof(sampling_math::MTPDiagnosticRecordStorage))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Existing diagnostic bank has incompatible physical storage");
                return false;
            }
            mtp_first_transaction_diagnostic_dev_ = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::FirstTransaction
                ? &mtp_diagnostic_record_storage_dev_->first_transaction : nullptr;
            mtp_depth_transaction_trace_dev_ = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::DepthTransactionTrace
                ? &mtp_diagnostic_record_storage_dev_->depth_transaction_trace : nullptr;
            mtp_committed_verifier_identity_dev_ =
                static_cast<
                    sampling_math::MTPCommittedVerifierIdentityRecord *>(
                    arena_->getDevicePtr(
                        BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                        state_.device_id));

            MTPVerifierOutcomeGraphBinding outcome_binding;
            outcome_binding.verifier_input_tokens_device =
                static_cast<const int32_t *>(
                    mtp_verifier_input_tokens_dev_);
            outcome_binding.active_verifier_row_count_device =
                static_cast<const int32_t *>(
                    mtp_verifier_request_lengths_dev_);
            outcome_binding.transaction_commit_budget_device =
                reinterpret_cast<const uint32_t *>(
                    device_generation_storage_.control_device +
                    sampling_math::
                        kDeviceGenerationControlTransactionCommitBudget);
            outcome_binding.next_leading_committed_output_count_device =
                device_generation_storage_.control_device +
                sampling_math::
                    kDeviceGenerationControlNextLeadingCommittedOutputCount;
            outcome_binding.stop_tokens_device =
                static_cast<const int32_t *>(
                    mtp_verifier_stop_tokens_dev_);
            outcome_binding.penalty_policy_device =
                static_cast<const MTPGreedyPenaltyPolicy *>(
                    mtp_greedy_penalty_policy_dev_);
            outcome_binding.generated_token_counts_device =
                static_cast<int32_t *>(
                    mtp_generated_token_counts_dev_);
            outcome_binding.generated_token_count_capacity =
                mtp_generated_token_count_capacity_;
            outcome_binding.verifier_tokens_device =
                static_cast<int32_t *>(stochastic_verify_tokens_dev_);
            outcome_binding.argmax_values_device =
                static_cast<float *>(stochastic_verify_accept_probs_dev_);
            outcome_binding.argmax_partial_values_device =
                static_cast<float *>(argmax_partial_vals_dev_);
            outcome_binding.argmax_partial_indices_device =
                static_cast<int32_t *>(argmax_partial_idxs_dev_);
            outcome_binding.argmax_partial_capacity =
                argmax_partial_capacity_;
            outcome_binding.output_tokens_device =
                static_cast<int32_t *>(
                    stochastic_batch_output_tokens_dev_);
            outcome_binding.output_meta_device =
                static_cast<int32_t *>(
                    stochastic_batch_output_meta_dev_);
            outcome_binding.output_token_capacity =
                stochastic_batch_output_token_stride_;
            outcome_binding.output_meta_capacity =
                sampling_math::kSpeculativeBatchMetaCount;
            if (!mtp_committed_verifier_identity_dev_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind persistent committed MTP verifier identity storage on "
                          << state_.device_id.toString());
                return false;
            }
            if (!outcome_binding.validForGreedy() ||
                !graph_builder_ ||
                !graph_builder_->setMTPVerifierOutcomeGraphBinding(
                    outcome_binding))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to install persistent graph-owned MTP outcome bindings on "
                          << state_.device_id.toString());
                return false;
            }

            IBackend *backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to resolve backend for stochastic pinned host scratch on "
                          << state_.device_id.toString());
                return false;
            }
            auto scratch = std::make_unique<PinnedHostScratch>();
            if (!scratch->allocate(
                    backend,
                    state_.device_id.gpu_ordinal(),
                    stochastic_batch_output_request_capacity_,
                    stochastic_batch_output_token_stride_,
                    sampling_math::kSpeculativeBatchMetaCount))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate pinned host scratch for stochastic MTP outcomes on "
                          << state_.device_id.toString());
                return false;
            }
            stochastic_batch_output_host_scratch_ = std::move(scratch);

            auto terminal_scratch = std::make_unique<PinnedHostScratch>();
            if (!device_generation_storage_.validFor(
                    stochastic_batch_output_request_capacity_) ||
                !terminal_scratch->allocate(
                    backend,
                    state_.device_id.gpu_ordinal(),
                    device_generation_storage_.request_capacity,
                    device_generation_storage_.response_token_stride,
                    device_generation_storage_.control_stride,
                    localHotTerminalMetadataBytes(device_generation_storage_.request_capacity)?physical_memory_authority_:nullptr,
                    localHotTerminalMetadataBytes(device_generation_storage_.request_capacity)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate pinned terminal response scratch on "
                          << state_.device_id.toString());
                return false;
            }
            device_generation_terminal_host_scratch_ =
                std::move(terminal_scratch);

            auto dispatch_ticket_scratch =
                std::make_unique<PinnedDispatchTicketScratch>();
            if (!dispatch_ticket_scratch->allocate(
                    backend,
                    state_.device_id.gpu_ordinal(),
                    device_generation_storage_.request_capacity))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate the pinned device-generation dispatch-ticket observer on "
                          << state_.device_id.toString());
                return false;
            }
            device_generation_dispatch_ticket_host_scratch_ =
                std::move(dispatch_ticket_scratch);

            void *const raw_dispatch_ticket_event =
                backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_dispatch_ticket_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate the device-generation dispatch-ticket event on "
                          << state_.device_id.toString());
                return false;
            }
            const int dispatch_ticket_device_ordinal =
                state_.device_id.gpu_ordinal();
            device_generation_dispatch_ticket_ready_event_.reset(
                raw_dispatch_ticket_event,
                [backend, dispatch_ticket_device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(
                            event,
                            dispatch_ticket_device_ordinal);
                });

            void *const raw_terminal_host_event =
                backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_terminal_host_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate the terminal device-generation host event on "
                          << state_.device_id.toString());
                return false;
            }
            device_generation_terminal_host_ready_event_.reset(
                raw_terminal_host_event,
                [backend, dispatch_ticket_device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(
                            event,
                            dispatch_ticket_device_ordinal);
                });

            void *raw_response_stream =
                backend->createStream(state_.device_id.gpu_ordinal());
            if (!raw_response_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate the terminal response bridge stream on "
                          << state_.device_id.toString());
                return false;
            }
            const int response_device_ordinal =
                state_.device_id.gpu_ordinal();
            stochastic_outcome_response_bridge_stream_.reset(
                raw_response_stream,
                [backend, response_device_ordinal](void *stream)
                {
                    if (stream)
                        backend->destroyStream(
                            stream,
                            response_device_ordinal);
                });

            releaseRetainedStorageGenerationTopology();
            void *raw_loop_stream =
                backend->createStream(state_.device_id.gpu_ordinal());
            if (!raw_loop_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate the device-generation loop stream on "
                          << state_.device_id.toString());
                return false;
            }
            const int loop_device_ordinal = state_.device_id.gpu_ordinal();
            sparse_query_generation_parents_[0].stream.reset(
                raw_loop_stream,
                [backend, loop_device_ordinal](void *stream)
                {
                    if (stream)
                        backend->destroyStream(stream, loop_device_ordinal);
                });
            try
            {
                auto &gpu_context =
                    GPUDeviceContextPool::instance().getContext(
                        state_.device_id);
                for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
                {
                    auto &parent = sparse_query_generation_parents_[program];
                    parent.stream = sparse_query_generation_parents_[0].stream;
                    parent.capture = gpu_context.createGraphCapture(raw_loop_stream);
                    if (!parent.capture)
                        throw std::runtime_error("Sparse-query generation family has no exact native capture owner");
                }
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create the persistent device-generation graph owner on "
                          << state_.device_id.toString() << ": "
                          << exception.what());
                releaseRetainedStorageGenerationTopology();
                return false;
            }
            if (!deviceGenerationLoopGraph().capture)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU context returned no device-generation graph owner on "
                          << state_.device_id.toString());
                releaseRetainedStorageGenerationTopology();
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_response_bridge_stream_initializations",
                1.0,
                "initialization",
                state_.device_id.toString());
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_loop_stream_initializations",
                1.0,
                "initialization",
                state_.device_id.toString(),
                {{"backend",
                  deviceGenerationLoopGraph().capture->backendName()}});
        }
        else
        {
            stochastic_batch_output_host_scratch_.reset();
            device_generation_terminal_host_scratch_.reset();
            device_generation_dispatch_ticket_host_scratch_.reset();
            device_generation_dispatch_ticket_ready_event_.reset();
            device_generation_terminal_host_ready_event_.reset();
            stochastic_outcome_response_bridge_stream_.reset();
            releaseRetainedStorageGenerationTopology();
        }

        /*
         * Every GPU forward publishes its complete live-state write set through
         * one orchestrator-owned event. The event must predate the first request:
         * graph-cache invalidation is allowed to destroy a capture stream, while
         * archive and diagnostic consumers still need a durable dependency on the
         * work that stream submitted.
         */
        if (state_.device_id.is_gpu() &&
            !initializeForwardGraphOutputReadyEvent())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize the persistent forward-output timeline event");
            return false;
        }

        /*
         * All generation policies publish a logical frontier. Allocate its
         * reader/writer edges before admission, independently of MTP. Shifted
         * prefill additionally needs its terminal-archive handoff; it is not
         * permission to make common generation events predictor-dependent.
         */
        if (state_.device_id.is_gpu() &&
            (!initializePersistentDeviceGenerationEvents() ||
             (retainsMTPGraphCapacity(config.mtp) && !initializeMTPPrefillTerminalArchiveReadyEvent())))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize persistent generation stream-handoff events");
            return false;
        }

        // Wire arena directly to graph builder (replaces bindArenaToManagedBuffers + setBuffers shim)
        graph_builder_->setArena(arena_.get());

        // Wire arena to executor for contract-based coherence
        executor_.setArena(arena_.get());

        /*
         * Resolve ExpertOverlay epoch topology only after the backend arena and
         * executor exist, but before any request graph can be captured. The graph
         * builder owns this allocation alongside every runtime table that embeds
         * its ticket address.
         */
        if (!initializeMoEOverlayEpochExecutionBinding())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize the durable ExpertOverlay epoch binding");
            return false;
        }

        // Log per-buffer allocation details
        arena_->logAllocationSummary();

        // Log theoretical aliasing savings
        auto [original, optimized] = BufferAllocator::estimateMemorySavings(schema, resolver_config);
        double savings = (original > 0) ? 100.0 * (original - optimized) / original : 0.0;
        LOG_DEBUG("[DeviceGraphOrchestrator] Theoretical aliasing savings: "
                  << (original / 1024.0) << " KB -> " << (optimized / 1024.0) << " KB"
                  << " (" << savings << "% reduction)");

        return true;
    }

    bool DeviceGraphOrchestrator::initializeMTPKVCaches(
        int batch_size,
        int max_seq_len,
        ActivationPrecision kv_cache_prec,
        KVCacheLayoutMode kv_layout_mode,
        DeviceId device,
        const std::shared_ptr<IMPIContext> &local_mpi_ctx,
        bool use_sharded_cache,
        bool has_tp,
        bool is_global_tp)
    {
        const auto &config = graph_builder_->config();
        state_.mtp_kv_caches.clear();

        /*
         * A replicated MTP predictor owns complete K/V projections and must
         * therefore own a complete shifted cache on every participant. The
         * committed main cache may remain TP-sharded: these are separate state
         * authorities with separate graph families. Derive this here from the
         * typed graph policy so callers cannot accidentally pass main-cache
         * geometry into a replicated sidecar.
         */
        const int tp_participant_count =
            has_tp
                ? config.tp_ctx->degree()
                : (is_global_tp ? mpi_ctx_->world_size() : 1);
        const MTPShiftedKVHeadLayout sidecar_head_layout =
            resolveMTPShiftedKVHeadLayout(
                config.mtp,
                config.dense_tp_enabled,
                tp_participant_count);
        const int primary_local_kv_heads =
            use_sharded_cache && (has_tp || is_global_tp)
                ? config.local_n_kv_heads
                : config.n_kv_heads;
        const int sidecar_local_kv_heads =
            resolveMTPShiftedKVLocalHeadCount(
                sidecar_head_layout,
                config.n_kv_heads,
                primary_local_kv_heads);

        const auto query_sidecar_heads = resolveQueryIntervalCacheHeads(config, device,
            sidecar_head_layout == MTPShiftedKVHeadLayout::FullModelPerParticipant
                ? KVHeadConsumerDemand::FullModel : KVHeadConsumerDemand::AssignedQueryInterval);

        // Qwen3.5/Qwen3.6 support is currently D=1. Additional depths need
        // distinct sidecar weights before they can have independent caches.
        constexpr int kSupportedMTPDepth = 1;
        state_.mtp_kv_caches.reserve(kSupportedMTPDepth);

        for (int depth = 0; depth < kSupportedMTPDepth; ++depth)
        {
            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = kv_cache_prec;
            kv_config.device = device;
            kv_config.num_layers = 1;
            kv_config.first_layer_index = 0;
            kv_config.batch_size = batch_size;
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = config.n_kv_heads;
            kv_config.local_n_kv_heads = query_sidecar_heads ? query_sidecar_heads->count : sidecar_local_kv_heads;
            kv_config.head_dim = config.head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();
            kv_config.turboquant_ctx = config.turboquant_ctx;
            kv_config.physical_memory_authority =
                physical_memory_authority_;

            if (query_sidecar_heads)
                kv_config.kv_head_start = query_sidecar_heads->first;
            else if (sidecar_head_layout ==
                    MTPShiftedKVHeadLayout::PrimaryTensorParallelShard &&
                use_sharded_cache &&
                (has_tp || is_global_tp))
            {
                kv_config.kv_head_start = has_tp
                                               ? config.tp_device_idx * config.local_n_kv_heads
                                               : mpi_ctx_->rank() * config.local_n_kv_heads;
            }

            auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
            if (!cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create MTP KV cache depth " << depth);
                return false;
            }
            if (sidecar_head_layout ==
                    MTPShiftedKVHeadLayout::FullModelPerParticipant &&
                cache->is_sharded())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Replicated MTP sidecar constructed a sharded KV cache at depth "
                          << depth);
                return false;
            }
            cache->bindStateOwnership({
                .role = IKVCache::StateRole::MTPShiftedSidecar,
                .mtp_depth = depth,
                .shifted_origin = graph_builder_->config().mtpShiftedCacheOrigin(),
            });
            logKVCacheBom(("mtp_depth_" + std::to_string(depth)).c_str(),
                          device,
                          cache.get(),
                          batch_size);
            state_.mtp_kv_caches.push_back(std::move(cache));
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Created " << state_.mtp_kv_caches.size()
                                                       << " request-local MTP KV cache(s)");
        return true;
    }

    size_t DeviceGraphOrchestrator::localLogitsVocabColumns(const TensorBase *tensor) const
    {
        size_t storage_cols = 0;
        if (tensor)
        {
            const auto &shape = tensor->shape();
            storage_cols = shape.size() >= 2 ? shape[1] : 0;
        }

        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel)
        {
            const int configured_vocab_local = graph_builder_->config().vocab_local;
            if (configured_vocab_local > 0)
            {
                const size_t semantic_cols = static_cast<size_t>(configured_vocab_local);
                if (storage_cols == 0 || semantic_cols <= storage_cols)
                    return semantic_cols;

                LOG_WARN("[DeviceGraphOrchestrator] Configured local logits vocab "
                         << semantic_cols << " exceeds backing storage columns "
                         << storage_cols << "; using storage width");
            }
        }

        return storage_cols;
    }

    size_t DeviceGraphOrchestrator::localLogitsRowStrideColumns(const TensorBase *tensor) const
    {
        if (!tensor)
            return 0;
        const auto &shape = tensor->shape();
        const size_t storage_cols = shape.size() >= 2 ? shape[1] : 0;
        return storage_cols > 0 ? storage_cols : localLogitsVocabColumns(tensor);
    }

    int DeviceGraphOrchestrator::localLogitsVocabOffset() const
    {
        if (!graph_builder_)
            return 0;
        return vocabOffsetForTPConfig(graph_builder_->config());
    }

    bool DeviceGraphOrchestrator::activeMainLogitsAreColumnParallel() const
    {
        if (!state_.logits_local || !graph_builder_)
            return false;

        const auto &config = graph_builder_->config();
        if (!config.lm_head_column_parallel || config.hasNativeLocalVocabularyGather())
            return false;

        /*
         * A request-batched GPU prefill temporarily turns on row-indexed
         * all-position logits and projects one compact terminal row per
         * request. Under mirrored MTP terminal-head policy, that temporary graph
         * binds the replicated full-vocabulary head and writes
         * `all_position_logits`; the ordinary `logits_local` allocation still
         * exists only because it belongs to the runner's maximum-capacity
         * arena. Classify the output that was actually produced, rather than
         * inferring ownership from that dormant allocation.
         *
         * The compact-row count remains positive from successful prefill until
         * `publishMainLogitsBatchSamplesToDeviceResidentState()` consumes the
         * rows, so it is the
         * transaction-lifetime proof that the full-vocabulary result is active.
         */
        if (request_batched_prefill_logits_row_count_ > 0 &&
            publishesFullVocabularyMTPLogits())
        {
            return false;
        }

        /*
         * A TP runner with mirrored MTP-head policy binds that replicated
         * head for every main terminal projection. Ordinary prefill/decode
         * projects one row, compact request prefill projects one row per
         * request, and verifier graphs project their declared compact rows.
         * LOGITS_LOCAL remains allocated only as maximum-capacity arena state;
         * it is never authoritative while this policy is active.
         */
        if (publishesFullVocabularyMTPLogits())
        {
            return false;
        }

        /*
         * Prefill-TP/decode-replicated execution reserves LOGITS_LOCAL with full-width storage so
         * prefill can use column-parallel LM-head output, but decode writes the
         * authoritative replicated row to LOGITS.  Treating LOGITS_LOCAL as
         * active during decode reuses the last prefill row.
         */
        if (current_phase_ == InferencePhase::DECODE &&
            config.dense_tp_decode_replicated)
            return false;

        return true;
    }

    bool DeviceGraphOrchestrator::mtpSidecarLogitsAreColumnParallel() const
    {
        return graph_builder_ && graph_builder_->config().mtpParticipantLogitsAreVocabularySharded() &&
            !graph_builder_->config().mtpPublishesNativeGatheredVocabulary();
    }

    BufferId DeviceGraphOrchestrator::mtpSamplingLogitsBufferId() const
    {
        return graph_builder_ && graph_builder_->config().mtpPublishesNativeGatheredVocabulary()
            ? BufferId::MTP_LOGITS_GATHERED : BufferId::MTP_LOGITS;
    }

    bool DeviceGraphOrchestrator::mtpSidecarRequiresLogitsGather(bool kv_cache_only) const
    {
        if (!graph_builder_) return false;
        const auto *global_tp = globalTPContextForMTPCoordination();
        return resolveMTPTerminalLogitsCollective({
            .layout = graph_builder_->config().mtpTerminalLogitsLayout(),
            .sidecar_produces_logits = !kv_cache_only,
            .spans_multiple_global_ranks = global_tp && global_tp->degree() > 1,
            .native_local_gpu_tp = graph_builder_->config().hasNativeLocalVocabularyGather(),
        }) != MTPTerminalLogitsCollective::None;
    }

    bool DeviceGraphOrchestrator::allPositionVerifierGraphWritesLocalLogits(
        int graph_token_count) const
    {
        if (!graph_builder_)
            return false;

        const auto &config = graph_builder_->config();
        if (!config.lm_head_column_parallel)
            return false;

        /*
         * Decode-replicated LocalTP makes the dense stack and LM head full and
         * serial-equivalent for short verifier rows.  Advertising a local shard
         * here would make the graph builder select the primary sharded LM-head
         * binding, which is a different numerical path from ordinary serial
         * decode even when the verifier hidden rows are identical.
         */
        if (config.dense_tp_enabled &&
            config.dense_tp_decode_replicated)
            return false;

        if (config.mtpParticipantOwnsFullVocabulary())
        {
            return false;
        }

        return true;
    }

    bool DeviceGraphOrchestrator::activeAllPositionLogitsAreColumnParallel(
        int graph_token_count) const
    {
        return state_.all_position_logits_local != nullptr && graph_builder_ &&
               !graph_builder_->config().hasNativeLocalVocabularyGather() &&
               allPositionVerifierGraphWritesLocalLogits(graph_token_count);
    }

    std::shared_ptr<TensorBase>
    DeviceGraphOrchestrator::preplannedMTPLogitsBuffer(
        ForwardExecutionRole execution_role,
        BufferId buffer_id,
        size_t rows,
        size_t columns) const
    {
        const bool typed_mtp_main_graph =
            execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier ||
            execution_role ==
                ForwardExecutionRole::MTPCondition;
        if (!typed_mtp_main_graph ||
            !state_.device_id.is_gpu())
        {
            return {};
        }

        const auto it = state_.extension_buffers.find(buffer_id);
        if (it == state_.extension_buffers.end() || !it->second)
            return {};

        const TensorBase *candidate = it->second.get();
        const auto &shape = candidate->shape();
        const auto device = candidate->current_device();
        if (shape.size() != 2 ||
            shape[0] < rows ||
            shape[1] != columns ||
            !device.has_value() ||
            *device != state_.device_id ||
            candidate->isMapped() ||
            candidate->gpu_data_ptr() == nullptr)
        {
            return {};
        }

        return it->second;
    }

    void DeviceGraphOrchestrator::requireNativeTerminalPublicationCapacity(std::size_t rows) const
    {
        if (!state_.device_id.is_gpu() || !graph_builder_ ||
            !graph_builder_->config().hasNativeLocalVocabularyGather() ||
            (pp_stage_config_ && !pp_stage_config_->has_lm_head))
            return;
        if (rows == 0 || rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Native terminal publication rows exceed the graph ABI");
        if (!allPositionVerifierGraphWritesLocalLogits(static_cast<int>(rows)))
            return;
        const auto &config = graph_builder_->config();
        // MainInference can also be a retained-MTP parity verifier; roles and
        // enabled flags cannot replace its true retained row-capacity union.
        config.ordinary_terminal_projection.requirePublication(
            static_cast<int>(rows), state_.batch_size, config.max_activation_rows,
            retainsMTPGraphCapacity(config.mtp) ? resolveMTPRetainedTargetQueryRows(config.mtp) : 0);
    }

    void DeviceGraphOrchestrator::requireActiveNativeTerminalPublicationCapacity(
        int requests, int sequence_rows) const
    {
        if ((!compute_all_position_logits_ && !live_mtp_request_batch_condition_) ||
            !state_.device_id.is_gpu() || !graph_builder_ ||
            !graph_builder_->config().hasNativeLocalVocabularyGather() ||
            !allPositionVerifierGraphWritesLocalLogits(1) ||
            (pp_stage_config_ && !pp_stage_config_->has_lm_head))
            return;
        const int input_rows = OrdinaryTerminalProjectionCapacity::checkedFlattenedRows(requests, sequence_rows);
        const int projected_rows = compute_all_position_logits_ && compute_row_indexed_all_position_logits_
            ? row_indexed_all_position_logits_row_count_ : input_rows;
        if (projected_rows <= 0 || projected_rows > input_rows)
            throw std::invalid_argument("Native terminal projected rows exceed the actual input population");
        requireNativeTerminalPublicationCapacity(static_cast<std::size_t>(projected_rows));
    }

    bool DeviceGraphOrchestrator::bindAllPositionLogitsOutputs(
        ForwardExecutionRole execution_role,
        size_t rows,
        TensorBase *&logits_output,
        TensorBase *&logits_local_output)
    {
        if (rows == 0 || !arena_ || !tensor_factory_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] All-position logits require a "
                "positive row count, arena, and initialized TensorFactory");
            return false;
        }

        // Layer-partitioned followers publish hidden activations, never logits.
        // A typed condition/verifier role does not grant them a vocabulary head
        // or permission to allocate a substitute all-position output bank.
        if (pp_stage_config_ && !pp_stage_config_->has_lm_head)
        {
            logits_output = nullptr;
            logits_local_output = nullptr;
            return true;
        }
        requireNativeTerminalPublicationCapacity(rows);
        const int rows_key = static_cast<int>(rows);
        const bool typed_gpu_mtp_logits =
            state_.device_id.is_gpu() &&
            (execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
             execution_role == ForwardExecutionRole::MTPCondition);
        auto resolve_logits_owner =
            [&](std::unordered_map<int, std::shared_ptr<TensorBase>> &owners,
                BufferId prepared_buffer,
                size_t columns,
                const char *buffer_name) -> std::shared_ptr<TensorBase>
        {
            if (typed_gpu_mtp_logits)
            {
                auto prepared = preplannedMTPLogitsBuffer(
                    execution_role,
                    prepared_buffer,
                    rows,
                    columns);
                if (!prepared)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Typed GPU MTP graph is "
                        "missing preplanned logits capacity"
                        << " role=" << static_cast<int>(execution_role)
                        << " buffer=" << bufferIdName(prepared_buffer)
                        << " rows=" << rows
                        << " columns=" << columns
                        << " device=" << state_.device_id.toString());
                }
                return prepared;
            }

            const auto existing = owners.find(rows_key);
            if (existing != owners.end() && existing->second)
            {
                const auto &shape = existing->second->shape();
                if (shape.size() == 2 && shape[0] == rows &&
                    shape[1] == columns)
                {
                    return existing->second;
                }
            }

            auto tensor = tensor_factory_->createFP32(
                {rows, columns},
                state_.device_id);
            auto owner = std::shared_ptr<TensorBase>(tensor.release());
            if (owner)
            {
                owners.insert_or_assign(rows_key, owner);
                logVramBomLine(
                    "dynamic_tensor_buffer",
                    "device=" + state_.device_id.toString() +
                        " name=" + buffer_name +
                        " rows=" + std::to_string(rows) +
                        " cols=" + std::to_string(columns) +
                        " dtype=FP32 " + vramBomBytes(owner->size_bytes()));
            }
            return owner;
        };

        if (allPositionVerifierGraphWritesLocalLogits(
                static_cast<int>(rows)))
        {
            if (!state_.logits_local)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] All-position local logits "
                    "require logits_local");
                return false;
            }
            const size_t local_vocab =
                localLogitsVocabColumns(state_.logits_local.get());
            if (local_vocab == 0)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] All-position local logits "
                    "require a non-zero local vocab");
                return false;
            }

            auto local_owner = resolve_logits_owner(
                state_.all_position_logits_local_by_rows,
                BufferId::MTP_LOGITS,
                local_vocab,
                "all_position_logits_local");
            if (!local_owner)
                return false;
            state_.all_position_logits_local = std::move(local_owner);
            logits_local_output = state_.all_position_logits_local.get();
            if (!arena_->bindExternalBuffer(
                    BufferId::ALL_POSITION_LOGITS_LOCAL,
                    logits_local_output))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to bind all-position "
                    "local logits buffer");
                return false;
            }

            auto full_owner = resolve_logits_owner(
                state_.all_position_logits_by_rows,
                BufferId::MTP_LOGITS_GATHERED,
                static_cast<size_t>(state_.vocab_size),
                "all_position_logits");
            if (!full_owner)
                return false;
            state_.all_position_logits = std::move(full_owner);
            logits_output = state_.all_position_logits.get();
            if (!arena_->bindExternalBuffer(
                    BufferId::ALL_POSITION_LOGITS,
                    logits_output))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to bind all-position "
                    "logits buffer");
                return false;
            }
            return true;
        }

        state_.all_position_logits_local.reset();
        auto full_owner = resolve_logits_owner(
            state_.all_position_logits_by_rows,
            graph_builder_->config().mtpParticipantLogitsAreVocabularySharded()
                ? BufferId::MTP_LOGITS_GATHERED
                : BufferId::MTP_LOGITS,
            static_cast<size_t>(state_.vocab_size),
            "all_position_logits");
        if (!full_owner)
            return false;
        state_.all_position_logits = std::move(full_owner);
        logits_output = state_.all_position_logits.get();
        if (!arena_->bindExternalBuffer(
                BufferId::ALL_POSITION_LOGITS,
                logits_output))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to bind all-position logits buffer");
            return false;
        }
        return true;
    }

    void DeviceGraphOrchestrator::releaseBuffers()
    {
        /*
         * Public buffer release is a lifetime boundary just like destruction.
         * Retire the exact published events and close any static/prefill-only
         * epoch reader before invalidating its retained graphs or event owners.
         * This is intentionally event-scoped and never synchronizes the device.
        */
        retirePublishedDeviceWorkBeforeArenaRelease();
        const auto lease_state =
            moe_overlay_epoch_lease_lifecycle_->load();
        if (lease_state != MoEOverlayEpochLeaseState::Idle ||
            moe_overlay_external_sequence_identity_.valid() ||
            moe_overlay_epoch_forward_submission_stream_)
        {
            throw std::logic_error(
                std::string(
                    "DeviceGraphOrchestrator buffer retirement left an ExpertOverlay epoch lease active: ") +
                moeOverlayEpochLeaseStateName(lease_state));
        }

        /*
         * Every native executable embeds arena addresses.  Destroy the whole
         * parent/child topology before arena_.reset(); freeing the arena first
         * lets HIP defer its physical release behind those live graph owners,
         * which makes a subsequently constructed runner allocate a second
         * complete workspace family against the same VRAM.
         */
        destroyCapturedExecutionTopologyBeforeArenaRelease();

        moe_overlay_epoch_acquired_event_.reset();
        moe_overlay_epoch_released_event_.reset();
        moe_overlay_epoch_acquire_producer_stream_ = nullptr;
        moe_overlay_external_sequence_identity_.reset();
        moe_overlay_epoch_forward_submission_stream_ = nullptr;
        moe_overlay_epoch_execution_binding_ = {};
        if (arena_)
        {
            arena_.reset();
            LOG_DEBUG("[DeviceGraphOrchestrator] BufferArena released");
        }

        owned_buffers_.clear();

        // Clear buffer pointers
        managed_buffers_ = ModelBuffers{};
    }

    ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers()
    {
        return managed_buffers_.layer_buffers;
    }

    const ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers() const
    {
        return managed_buffers_.layer_buffers;
    }

    const ModelBuffers &DeviceGraphOrchestrator::getModelBuffers() const
    {
        return managed_buffers_;
    }

    const ArenaAllocationStats *DeviceGraphOrchestrator::bufferStats() const
    {
        if (!arena_)
        {
            return nullptr;
        }
        return &arena_->stats();
    }

    // NOTE: Legacy initializeArena() was removed. The arena is now exclusively
    // created and populated by the schema-driven initializeBuffers() path.
    // The legacy path only registered ~15 standard buffers and missed
    // model-specific ones (GDN_QKV, FA_GATE, etc.), causing crashes for
    // architectures like Qwen3.5.

    // =========================================================================
    // Execution Methods
    // =========================================================================

    bool DeviceGraphOrchestrator::executeForward(
        const ForwardInput &input,
        ForwardOutput &output,
        std::optional<ForwardGraphSignature> *materialized_signature)
    {
        // Enable device-scoped logging if not already set by caller (e.g., from forward())
        ScopedDeviceLog device_log(input.device);

        // Validate that all required configuration has been set
        if (!validateConfigurationForForward())
        {
            return false;
        }

        // Token input OR activation input is required
        bool has_token_input = input.token_ids || input.token_ids_device || input.batches;
        bool has_activation_input =
            external_hidden_state_input_ != nullptr ||
            input.external_hidden_state != nullptr;

        // For PP stages without embedding, activation input is required instead of tokens
        bool is_pp_middle_stage = pp_stage_config_.has_value() &&
                                  !pp_stage_config_.value().has_embedding;

        if (!has_token_input && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No token or activation input provided");
            return false;
        }

        if (is_pp_middle_stage && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] PP middle stage requires activation input "
                      "via setHiddenState()");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[DeviceGraphOrchestrator] executeForward: batch_size=" << input.batch_size
                                                                          << ", seq_len=" << input.seq_len
                                                                          << ", device=" << input.device);

        /*
         * Synthesize host positions only when the caller supplied no position
         * authority at all.  Device-resident prefill and decode inputs own a
         * permanent position row whose address is part of native graph-cache
         * identity; manufacturing a host row here would create two authorities
         * and make the captured chunk binding disagree with the graph input.
         */
        std::vector<int> position_ids_storage;
        ForwardInput effective_input = input;
        effective_input.sparse_query_execution_program = sparseQueryForwardProgram(input.execution_phase);
        if (input.execution_phase == ForwardExecutionPhase::Prefill && sparseQueryProgramCount() > 1u) {
            if (compiling_sparse_query_program_)
                effective_input.sparse_query_prefill_proof = SparseQueryPrefillProof::materialization(*compiling_sparse_query_program_);
            else if (input.sparse_query_prefill_proof)
                effective_input.sparse_query_execution_program = input.sparse_query_prefill_proof->program(
                    qwen4exp::QwenSparseFirstRankingLogicalRow);
        }

        if (!input.position_ids && !input.position_ids_device)
        {
            position_ids_storage = IGraphBuilder::buildPositionIds(
                input.seq_len, input.batch_size, input.position_offset);
            effective_input.position_ids = position_ids_storage.data();
        }

        // Pass external hidden state to graph builder input for PP middle/final stages
        if (external_hidden_state_input_)
        {
            effective_input.external_hidden_state = external_hidden_state_input_;
            // setHiddenState is the explicit host-owned segmented boundary.
            // Seal its current live rows before bucket planning changes capacity.
            effective_input.host_pipeline_activation_rows = HostPipelineActivationRows::admitted(
                external_hidden_state_input_, input.device, input.batch_size,
                input.real_seq_len > 0 ? input.real_seq_len : input.seq_len);
            LOG_DEBUG("[DeviceGraphOrchestrator] Using external hidden state input: "
                      << external_hidden_state_input_->numel() << " elements");
            external_hidden_state_input_ = nullptr; // single-use semantics
        }

        if(!executableForwardSubmissionIntent(effective_input.graph_submission_intent))
            throw std::invalid_argument("DGO cannot submit a workspace declaration or unknown forward intent");
        if(effective_input.device.is_gpu() && effective_input.batch_size==1 &&
            effective_input.execution_role==ForwardExecutionRole::MainInference &&
            effective_input.execution_phase==ForwardExecutionPhase::Prefill && !effective_input.device_prefill_chunk &&
            (effective_input.device_request_query || (effective_input.token_ids_device==request_token_ids_dev_ &&
                effective_input.position_ids_device==request_position_ids_dev_ && effective_input.sequence_lengths_device==request_sequence_lengths_dev_)))
        {
            const auto original=DeviceRequestQueryBinding::fromArena(*arena_,effective_input.device);
            if(effective_input.device_request_query && effective_input.device_request_query->captureIdentity()!=original.captureIdentity())
                throw std::invalid_argument("DGO prefill received a foreign original-request query binding");
            effective_input.device_request_query=original;
            requireDeviceRequestQueryOwnsInput(effective_input);
        }

        // Ensure engine is initialized with current config
        ensureForwardEngine();

        // Delegate to ForwardExecutionEngine
        const bool success = forward_engine_->execute(effective_input, output, *this, materialized_signature);
        if (!success)
        {
            if (mtp_verifier_outcome_graph_mode_ !=
                MTPVerifierOutcomeGraphMode::Disabled)
            {
                greedy_verifier_outcome_graph_transaction_ = {};
                mtp_verifier_outcome_graph_mode_ =
                    MTPVerifierOutcomeGraphMode::Disabled;
            }
            return false;
        }
        if (effective_input.execution_phase == ForwardExecutionPhase::Decode &&
            effective_input.graph_submission_intent == ForwardGraphSubmissionIntent::Execute &&
            sparseQueryProgramCount() > 1u)
            recordQwenSparseQueryDecodeForward(state_.device_id, effective_input.sparse_query_execution_program,
                effective_input.execution_role, effective_input.seq_len);
        if (effective_input.execution_phase == ForwardExecutionPhase::Prefill &&
            effective_input.graph_submission_intent == ForwardGraphSubmissionIntent::Execute &&
            sparseQueryProgramCount() > 1u && effective_input.sparse_query_prefill_proof &&
            effective_input.sparse_query_prefill_proof->namesRange(effective_input.position_offset, effective_input.seq_len))
            PerfStatsCollector::addCounter("qsa", "query_program_forward_submissions", 1.0,
                "prefill", state_.device_id.toString(),
                {{"program", effective_input.sparse_query_prefill_proof->program(qwen4exp::QwenSparseFirstRankingLogicalRow)==
                    SparseQueryExecutionProgram::ProvenNoRanking?"proven_no_ranking":"ranking_possible"},
                 {"role", std::to_string(static_cast<int>(effective_input.execution_role))},
                 {"physical_rows", std::to_string(effective_input.seq_len)},
                 {"logical_rows", std::to_string(effective_input.seq_len)},
                 {"logical_begin", std::to_string(effective_input.position_offset)},
                 {"width_policy", "exact_retained"},
                 {"observation", "successful_forward_submission_not_native_replay_count"}});
        /*
         * Expert maintenance is a decode-transaction operation, not a raw
         * forward epilogue.  In particular, an MTP grouped verifier forward is
         * followed by accepted-state publication or rollback.  Moving experts
         * here would race that publication, while requiring seq_len == 1 here
         * silently omits maintenance for the grouped-only MTP path.  The
         * orchestration boundary calls maybeApplyDecodeBoundaryMaintenance()
         * after the complete decode transaction instead.
         */
        return true;
    }

    void DeviceGraphOrchestrator::commitSuccessfulForwardOutput(
        const ForwardOutput &output)
    {
        if (mtp_verifier_outcome_graph_mode_ !=
            MTPVerifierOutcomeGraphMode::Disabled)
        {
            if (mtp_verifier_outcome_graph_mode_ !=
                    MTPVerifierOutcomeGraphMode::Greedy ||
                greedy_verifier_outcome_graph_transaction_.state !=
                    GreedyVerifierOutcomeGraphState::Armed ||
                !output.execution.valid ||
                output.execution.execution_role !=
                    ForwardExecutionRole::GroupedMTPVerifier ||
                !output.execution.all_position_logits)
            {
                throw std::runtime_error(
                    "Successful verifier forward did not satisfy its armed "
                    "graph-owned MTP outcome transaction");
            }
            greedy_verifier_outcome_graph_transaction_.state =
                GreedyVerifierOutcomeGraphState::Produced;
        }
        if (!publishForwardGraphOutputReady(output))
        {
            throw std::runtime_error(
                "Successful GPU forward could not publish its durable "
                "ForwardGraphOutputReady dependency");
        }
    }

    bool DeviceGraphOrchestrator::maybeApplyDecodeBoundaryMaintenance(
        uint64_t committed_tokens)
    {
        if (committed_tokens == 0u)
        {
            throw std::invalid_argument(
                "Device MoE maintenance cannot retire an empty decode transaction");
        }
        if (!usesParticipantLocalDeviceMoERebalanceController())
            return true;

        if (compute_all_position_logits_ ||
            compute_row_indexed_all_position_logits_)
        {
            throw std::runtime_error(
                "Device MoE maintenance reached a decode boundary before "
                "grouped verifier mode was closed");
        }

        /*
         * Reaching this method is the commit signal.  Do not consult
         * state_.positions: grouped GPU MTP owns its logical position on the
         * device and intentionally leaves that legacy host mirror stale.  The
         * captured graph consumes only device-resident readiness and routing
         * state; the host supplies no position, token, or planning metadata.
         */
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_participant_committed_tokens",
            static_cast<double>(committed_tokens),
            "decode",
            state_.device_id.toString());
        /*
         * The accepted-state/sampling graph has already advanced the canonical
         * device controller by the exact grouped-MTP commit count. One enqueue
         * observes that durable state; replaying once per token would count the
         * transaction twice and serialize otherwise grouped work.
         */
        return maybeRunDeviceMoERebalanceMaintenanceGraph();
    }

    DeviceMoERebalanceMaintenanceExecutionPolicy
    DeviceGraphOrchestrator::
        deviceMoERebalanceMaintenanceExecutionPolicy() const noexcept
    {
        if (!isMoeRebalancingActive() ||
            !usesDeviceResidentMoEOverlayAuthority())
        {
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;
        }

        /*
         * The topology-wide service owns one controller graph spanning every
         * mapped homogeneous group.  Advertising a participant-local policy
         * here would attach the retired CUDA conditional or ROCm ticket
         * authority to the same runtime table, producing two epoch writers.
         */
        if (usesTopologyWideDeviceMoEOverlayController())
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;

        if (state_.device_id.is_cuda())
        {
            return DeviceMoERebalanceMaintenanceExecutionPolicy::
                NativeConditionalGraph;
        }
        if (state_.device_id.is_rocm())
        {
            return DeviceMoERebalanceMaintenanceExecutionPolicy::
                HostScheduledCapturedMaintenance;
        }
        return DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
    }

    DeviceMoERebalanceHostedObservationSchedule
    DeviceGraphOrchestrator::
        deviceMoERebalanceHostedObservationSchedule() const noexcept
    {
        if (deviceMoERebalanceMaintenanceExecutionPolicy() !=
            DeviceMoERebalanceMaintenanceExecutionPolicy::
                HostScheduledCapturedMaintenance)
        {
            return {};
        }
        return device_moe_rebalance_maintenance_graph_
            .hosted_observation_schedule;
    }

    std::vector<MoEOverlayDeviceControllerRuntimeBinding>
    DeviceGraphOrchestrator::moeOverlayDeviceControllerRuntimeBindings() const
    {
        if (!graph_builder_)
            return {};
        auto binding =
            graph_builder_->deviceMoEOverlayControllerRuntimeBinding(
                state_.device_id);
        if (binding.publicationValid())
        {
            // The pointer is setup-stable for this orchestrator's lifetime.
            // It carries no policy or epoch value; it only exposes the exact
            // participant-local producer/release timeline to the controller.
            binding.inference_boundary = const_cast<
                DeviceGraphOrchestrator *>(this);
            binding.initial_runtime_publisher = const_cast<
                DeviceGraphOrchestrator *>(this);
        }
        return binding.valid()
                   ? std::vector<MoEOverlayDeviceControllerRuntimeBinding>{
                         binding}
                   : std::vector<MoEOverlayDeviceControllerRuntimeBinding>{};
    }

    bool DeviceGraphOrchestrator::publishMoEOverlayDeviceInitialRuntime(
        void *controller_stream)
    {
        if (!state_.device_id.is_gpu() || !controller_stream ||
            !graph_builder_ ||
            !moe_overlay_initial_runtime_source_ready_event_ ||
            !moe_overlay_initial_runtime_published_event_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay initial runtime "
                "publication has incomplete device, stream, graph, or event ownership");
            return false;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay initial runtime "
                "publication cannot resolve its GPU backend");
            return false;
        }
        const auto edge = DeviceEventEdge::at(
                              DeviceTimelinePoint::
                                  MoEOverlayInitialRuntimeReady)
                              .from(
                                  DeviceTimelineRole::
                                      MoEOverlayRuntimePublication);

        if (moe_overlay_initial_runtime_published_)
        {
            if (!moe_overlay_initial_runtime_publication_stream_ ||
                !edge.to(DeviceTimelineRole::MoERebalanceMaintenance)
                     .enqueueWait(
                         *backend,
                         state_.device_id,
                         moe_overlay_initial_runtime_published_event_.get(),
                         moe_overlay_initial_runtime_publication_stream_,
                         controller_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay could not order "
                    "a replacement controller after its published initial runtime");
                return false;
            }
            return true;
        }

        void *const graph_build_stream = explicitGPUStreamForOperation(
            "moe_overlay_initial_runtime_source");
        if (!graph_build_stream ||
            !edge.publish(
                *backend,
                state_.device_id,
                moe_overlay_initial_runtime_source_ready_event_.get(),
                graph_build_stream) ||
            !edge.to(DeviceTimelineRole::MoEOverlayRuntimePublication)
                 .enqueueWait(
                     *backend,
                     state_.device_id,
                     moe_overlay_initial_runtime_source_ready_event_.get(),
                     graph_build_stream,
                     controller_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay could not join its "
                "graph-build runtime producer to the controller stream");
            return false;
        }

        if (!graph_builder_->finalizeMoEOverlayDeviceControllerRuntime(
                state_.device_id,
                controller_stream) ||
            !edge.publish(
                *backend,
                state_.device_id,
                moe_overlay_initial_runtime_published_event_.get(),
                controller_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay could not publish "
                "the complete retained-layer runtime family");
            return false;
        }

        moe_overlay_initial_runtime_publication_stream_ = controller_stream;
        moe_overlay_initial_runtime_published_ = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "initial_runtime_publications",
            1.0,
            "model_setup",
            state_.device_id.toString(),
            {{"source", "continuation_graph_builder"},
             {"ordering", "event_wait"},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::installMoEOverlayTransferProgressEpoch(
        std::shared_ptr<MappedTransferProgressEpoch> epoch)
    {
        if (!epoch || !state_.device_id.is_gpu() ||
            epoch->device() != state_.device_id)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Rejected a retained transfer epoch with mismatched device ownership");
            return false;
        }
        if (moe_overlay_transfer_progress_epoch_ &&
            moe_overlay_transfer_progress_epoch_ != epoch)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Rejected replacement of an installed retained transfer epoch on "
                << state_.device_id.toString());
            return false;
        }
        moe_overlay_transfer_progress_epoch_ = std::move(epoch);
        return true;
    }

    MoEOverlayInferenceBoundaryStatus DeviceGraphOrchestrator::
        enqueueMoEOverlayDeviceInferenceBoundary(
            void *maintenance_stream,
            MoEOverlayInferenceBoundaryRequest request)
    {
        const auto *const moe = graph_builder_
                                    ? &graph_builder_->config().moe
                                    : nullptr;
        const bool observational_boundary =
            request.purpose == MoEOverlayInferenceBoundaryPurpose::
                                   ServiceTelemetrySnapshot;
        const bool overlay_authority_bound =
            moe && moe->usesExpertOverlayDurableResidencyAuthority() &&
            moe->routed_expert_plan &&
            moe->authority_execution !=
                MoEOverlayAuthorityExecutionKind::Unresolved;
        const bool device_policy_boundary =
            !observational_boundary && moe &&
            moe->device_controller_fabric &&
            usesDeviceResidentMoEOverlayAuthority();
        if (!state_.device_id.is_gpu() || !maintenance_stream ||
            (observational_boundary && !overlay_authority_bound) ||
            (!observational_boundary && !device_policy_boundary))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Topology-wide ExpertOverlay "
                "boundary has incomplete device, stream, authority, or purpose ownership");
            return MoEOverlayInferenceBoundaryStatus::Failed;
        }

        const char *const name = request.name();
        const auto defer_for_lease =
            [&](MoEOverlayEpochLeaseState lease_state)
            -> MoEOverlayInferenceBoundaryStatus
        {
            /*
             * Inference and maintenance share this one atomic authority. Any
             * pending/forward/hosted phase means inference owns the lease; the
             * controller retries after its ordinary terminal publication. It
             * never joins, clears, or releases a partially submitted forward.
             */
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "inference_boundary_deferrals",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"boundary", name},
                 {"reason", moeOverlayEpochLeaseStateName(lease_state)},
                 {"blocking", "false"}});
            return MoEOverlayInferenceBoundaryStatus::Deferred;
        };
        const auto initial_lease_state =
            moe_overlay_epoch_lease_lifecycle_->load();
        if (moeOverlayEpochObservationAction(initial_lease_state) ==
            MoEOverlayEpochObservationAction::DeferToInferenceOwner)
        {
            return defer_for_lease(initial_lease_state);
        }
        if (compute_all_position_logits_ ||
            compute_row_indexed_all_position_logits_)
        {
            /*
             * Grouped verification is one live inference transaction. The
             * background controller is allowed to reach this boundary before
             * acceptance/rollback closes its all-position logits view, but it
             * must not turn that ordinary overlap into either a fatal error or
             * a wait on the inference thread. Retry the complete topology
             * preflight after the verifier publishes its normal closure edge.
             */
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "inference_boundary_deferrals",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"boundary",
                  name},
                 {"reason", "grouped_verifier_live"},
                 {"blocking", "false"}});
            return MoEOverlayInferenceBoundaryStatus::Deferred;
        }
        bool replacement_writer_pending = false;
        if (!waitForDeviceResidentLogicalSequenceStateMailboxForObservation(
                maintenance_stream,
                name,
                &replacement_writer_pending))
        {
            if (replacement_writer_pending)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "inference_boundary_deferrals",
                    1.0,
                    perfPhaseName(),
                    state_.device_id.toString(),
                    {{"boundary", name},
                     {"reason", "logical_state_replacement_writer"},
                     {"blocking", "false"}});
                return MoEOverlayInferenceBoundaryStatus::Deferred;
            }
            return MoEOverlayInferenceBoundaryStatus::Failed;
        }
        if (!waitForLiveInferenceStateReadyForObservation(
                maintenance_stream,
                name,
                DeviceTimelineRole::MoERebalanceMaintenance,
                /*logical_state_mailbox_already_joined=*/true))
        {
            return MoEOverlayInferenceBoundaryStatus::Failed;
        }

        /*
         * Observation may consume an immutable terminal receipt, but it never
         * closes a live reader. ExternalReader is the admitted next inference
         * transaction and its forward must be the sole authority that promotes
         * or releases the corresponding device ticket.
         */
        {
            /*
             * Re-open the lifecycle under its submission authority after all
             * live-state joins.  The next inference acquire is allowed to
             * consume this same immutable event, so a naked atomic state load
             * is insufficient: it could observe `ReleasePublished` while that
             * acquire was clearing the associated raw stream payload.  The
             * typed submission pins the state and receipt as one pair while we
             * enqueue this independent consumer wait.
             */
            auto joined_submission =
                moe_overlay_epoch_lease_lifecycle_->beginSubmission();
            const auto joined_lease_state = joined_submission.state();
            const auto observation_action =
                moeOverlayEpochObservationAction(joined_lease_state);
            if (observation_action ==
                MoEOverlayEpochObservationAction::DeferToInferenceOwner)
            {
                return defer_for_lease(joined_lease_state);
            }
            if (observation_action ==
                MoEOverlayEpochObservationAction::WaitForPublishedRelease)
            {
                /*
                 * Ordinary topology-wide forward submissions append their
                 * captured release before returning to the execution engine.
                 * The release kernel may still be queued, so consume its
                 * immutable event without recording a late fence on the
                 * mutable inference stream. Event waits are multi-consumer and
                 * do not transfer ownership.
                 */
                IBackend *const backend = getBackendFor(state_.device_id);
                if (!backend || !moe_overlay_epoch_released_event_ ||
                    !joined_submission
                         .publishedReleaseProducerStream() ||
                    !backend->streamWaitEvent(
                        maintenance_stream,
                        moe_overlay_epoch_released_event_.get(),
                        state_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Topology-wide ExpertOverlay maintenance could not consume the completed forward-reader release on "
                        << state_.device_id.toString());
                    return MoEOverlayInferenceBoundaryStatus::Failed;
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "inference_boundary_release_event_waits",
                    1.0,
                    perfPhaseName(),
                    state_.device_id.toString(),
                    {{"boundary", name},
                     {"blocking", "false"},
                     {"stream", "exact_participant_maintenance"}});
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "inference_boundaries_enqueued",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"stream", "exact_participant_maintenance"},
             {"blocking", "false"},
             {"host_epoch_mirror", "false"}});
        return MoEOverlayInferenceBoundaryStatus::Submitted;
    }

    bool DeviceGraphOrchestrator::
        submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary()
    {
        auto &cache = device_moe_rebalance_maintenance_graph_;
        if (deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !cache.hosted_observation_schedule.valid() ||
            !cache.cadence_transaction_capability_checked ||
            !cache.hosted_decode_commit_boundary_embedded ||
            device_moe_rebalance_dispatch_ticket_copy_pending_ ||
            last_device_moe_rebalance_dispatch_ticket_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP known-non-due MoE boundary has incomplete or overlapping ownership on "
                      << state_.device_id.toString());
            return false;
        }

        /*
         * The immutable rank countdown proves this boundary cannot be due.
         * Ordinary serial decode already published and acknowledged the edge
         * at the end of its complete captured graph. Grouped MTP published the
         * exact accepted-round count and its next admission owns the non-due
         * acknowledgement. No graph, event, transfer, or overlay writer is
         * therefore legal at this host boundary.
         */
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_participant_boundary_calls",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"scheduler", "authenticated_dispatch_ticket"},
             {"ticket_observation", "elided"}});
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_scheduler_transactions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"backend", "ROCm"},
             {"scheduler", "decode_graph_embedded_cadence"},
             {"maintenance_due", "false"},
             {"ticket_d2h", "false"},
             {"standalone_graph_launch", "false"},
             {"host_state_upload", "false"}});
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_moe_rebalance_dispatch_ticket_observations_elided",
            1.0,
            "decode",
            state_.device_id.toString());
        return true;
    }
} // namespace llaminar2
