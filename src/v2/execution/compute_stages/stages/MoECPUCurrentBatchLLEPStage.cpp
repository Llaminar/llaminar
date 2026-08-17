/**
 * @file MoECPUCurrentBatchLLEPStage.cpp
 * @brief CPU current-batch least-loaded expert transaction implementation.
 */

#include "MoECPUCurrentBatchLLEPStage.h"

#include "../../../collective/IGlobalTPContext.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
    constexpr uint32_t kUnassignedParticipant =
        least_loaded_ep::kInvalidParticipant;

    uint64_t hashWord(uint64_t hash, uint64_t value) noexcept
    {
        constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);
        for (int byte = 0; byte < 8; ++byte)
        {
            hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
            hash *= kFnvPrime;
        }
        return hash;
    }

    uint64_t initialHash() noexcept
    {
        return UINT64_C(1469598103934665603);
    }

    bool integralExpertId(float value, int num_experts, int &expert_id) noexcept
    {
        if (!std::isfinite(value))
            return false;
        const float rounded = std::round(value);
        if (std::fabs(value - rounded) > 1e-4f)
            return false;
        expert_id = static_cast<int>(rounded);
        return expert_id >= 0 && expert_id < num_experts;
    }

} // namespace

    uint32_t CPUCurrentBatchLLEPTransactionState::destinationForFlatRoute(
        size_t flat_route) const
    {
        if (!active)
        {
            throw std::logic_error(
                "CPU current-batch LLEP route publication is not active");
        }
        if (flat_route >= destination_by_flat_route.size())
        {
            throw std::out_of_range(
                "CPU current-batch LLEP flat route is out of range");
        }
        const uint32_t destination = destination_by_flat_route[flat_route];
        if (destination >= static_cast<uint32_t>(participant_count))
        {
            throw std::logic_error(
                "CPU current-batch LLEP route has no valid destination");
        }
        return destination;
    }

    MoECPUCurrentBatchLLEPStage::MoECPUCurrentBatchLLEPStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.stage_name.empty())
        {
            params_.stage_name =
                params_.phase == CPUCurrentBatchLLEPPhase::Begin
                    ? "moe_cpu_current_batch_llep_begin"
                    : "moe_cpu_current_batch_llep_restore";
        }

        if (params_.phase != CPUCurrentBatchLLEPPhase::Begin)
            return;

        if (!params_.state)
        {
            params_.state =
                std::make_shared<CPUCurrentBatchLLEPTransactionState>();
        }
        if (params_.seq_len <= 0 || params_.top_k <= 0 ||
            params_.num_experts <= 0 || params_.participant_count <= 1 ||
            params_.participant_count > 32 || params_.participant_id < 0 ||
            params_.participant_id >= params_.participant_count ||
            params_.layer_idx < 0 || !params_.physical_executor ||
            !params_.residency_authority || params_.residency_domain.empty())
        {
            throw std::invalid_argument(
                "CPU current-batch LLEP graph has invalid geometry, executor, "
                "or ExpertOverlay parent authority");
        }

        auto &state = *params_.state;
        state.layer_idx = params_.layer_idx;
        state.seq_len = params_.seq_len;
        state.top_k = params_.top_k;
        state.num_experts = params_.num_experts;
        state.participant_count = params_.participant_count;
        state.participant_id = params_.participant_id;
        state.owner_participants.assign(
            static_cast<size_t>(params_.num_experts), 0u);
        state.owner_mask.assign(
            static_cast<size_t>(params_.num_experts), false);
        state.transient_resident_mask = state.owner_mask;

        const size_t route_slots = state.routeSlotCount();
        const size_t expert_count = static_cast<size_t>(params_.num_experts);
        const size_t participant_count =
            static_cast<size_t>(params_.participant_count);
        const size_t span_capacity =
            expert_count * (participant_count + 1u);
        const size_t transfer_capacity =
            expert_count * (participant_count - 1u);

        state.expert_loads.assign(expert_count, 0u);
        state.sorted_experts.resize(expert_count);
        state.pending_load.assign(participant_count, 0u);
        state.assigned_load.assign(participant_count, 0u);
        state.spans.resize(span_capacity);
        state.transfers.resize(transfer_capacity);
        state.expert_route_offsets.assign(expert_count + 1u, 0u);
        state.expert_route_cursors.assign(expert_count, 0u);
        state.destination_by_expert_occurrence.assign(
            route_slots, kUnassignedParticipant);
        state.destination_by_flat_route.assign(
            route_slots, kUnassignedParticipant);
        state.gathered_route_hashes.assign(participant_count, 0u);
        state.gathered_plan_hashes.assign(participant_count, 0u);

        params_.planner_config.expert_count =
            static_cast<uint32_t>(params_.num_experts);
        params_.planner_config.participant_count =
            static_cast<uint32_t>(params_.participant_count);
        if (params_.planner_config.max_weight_transfers == 0u)
        {
            params_.planner_config.max_weight_transfers =
                static_cast<uint32_t>(transfer_capacity);
        }
        if (params_.planner_config.max_non_owner_experts_per_participant == 0u)
        {
            params_.planner_config.max_non_owner_experts_per_participant =
                static_cast<uint32_t>(
                    (expert_count + participant_count - 1u) /
                    participant_count);
        }
    }

    bool MoECPUCurrentBatchLLEPStage::validateBindings() const
    {
        if (!params_.device_id.is_cpu() || !params_.physical_executor ||
            !params_.residency_authority || params_.residency_domain.empty() ||
            !params_.tp_ctx || !params_.expert_consumer || !params_.state)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Missing CPU transaction binding"
                      << " stage=" << params_.stage_name
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const auto *global_tp =
            dynamic_cast<const IGlobalTPContext *>(params_.tp_ctx);
        if (!global_tp || global_tp->degree() <= 1 ||
            global_tp->degree() != params_.participant_count ||
            global_tp->myIndex() != params_.participant_id)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] CPU LLEP requires an exact "
                      "cross-rank TP domain"
                      << " stage=" << params_.stage_name
                      << " configured_participants="
                      << params_.participant_count
                      << " configured_participant=" << params_.participant_id);
            return false;
        }

        const auto &state = *params_.state;
        if (state.layer_idx != params_.layer_idx ||
            state.participant_count != params_.participant_count ||
            state.participant_id != params_.participant_id)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Transaction identity mismatch"
                      << " stage=" << params_.stage_name
                      << " layer=" << params_.layer_idx);
            return false;
        }
        return true;
    }

    bool MoECPUCurrentBatchLLEPStage::verifyHashAgreement(
        uint64_t local_hash,
        std::vector<uint64_t> &gathered,
        const char *publication) const
    {
        auto *global_tp = dynamic_cast<IGlobalTPContext *>(params_.tp_ctx);
        if (!global_tp ||
            gathered.size() != static_cast<size_t>(global_tp->degree()) ||
            !global_tp->allgatherBytes(
                &local_hash,
                gathered.data(),
                sizeof(local_hash)))
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Failed to gather "
                      << publication << " hash for layer "
                      << params_.layer_idx);
            return false;
        }
        const bool identical =
            std::all_of(
                gathered.begin(), gathered.end(),
                [&](uint64_t hash) { return hash == local_hash; });
        if (!identical)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Cross-rank "
                      << publication << " mismatch for layer "
                      << params_.layer_idx);
        }
        return identical;
    }

    bool MoECPUCurrentBatchLLEPStage::executeBegin()
    {
        auto &state = *params_.state;
        if (state.active)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Refusing to begin an "
                      "already-active transaction for layer "
                      << params_.layer_idx);
            return false;
        }
        if (state.durable_epoch_lease.has_value() ||
            state.durable_parent_epoch != 0)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Refusing to begin with "
                      "an unreleased durable parent lease for layer "
                      << params_.layer_idx);
            return false;
        }
        if (!params_.routing_indices || !params_.routing_weights ||
            params_.routing_indices->native_type() != TensorType::FP32 ||
            params_.routing_weights->native_type() != TensorType::FP32 ||
            params_.routing_indices->numel() < state.routeSlotCount() ||
            params_.routing_weights->numel() < state.routeSlotCount())
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Begin requires complete "
                      "FP32 routing tensors for layer " << params_.layer_idx);
            return false;
        }

        const float *indices = params_.routing_indices->data();
        const float *weights = params_.routing_weights->data();
        if (!indices || !weights)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Routing tensors have no "
                      "CPU-authoritative data for layer " << params_.layer_idx);
            return false;
        }

        std::fill(state.expert_loads.begin(), state.expert_loads.end(), 0u);
        std::fill(state.destination_by_flat_route.begin(),
                  state.destination_by_flat_route.end(),
                  kUnassignedParticipant);
        uint64_t route_hash = initialHash();
        route_hash = hashWord(route_hash, static_cast<uint64_t>(params_.layer_idx));
        route_hash = hashWord(route_hash, static_cast<uint64_t>(params_.seq_len));
        route_hash = hashWord(route_hash, static_cast<uint64_t>(params_.top_k));

        for (size_t flat_route = 0;
             flat_route < state.routeSlotCount();
             ++flat_route)
        {
            const float weight = weights[flat_route];
            if (!std::isfinite(weight))
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Non-finite route weight"
                          << " layer=" << params_.layer_idx
                          << " flat_route=" << flat_route);
                return false;
            }
            const bool active = weight != 0.0f;
            int expert_id = -1;
            if (active &&
                !integralExpertId(
                    indices[flat_route], params_.num_experts, expert_id))
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Invalid expert id"
                          << " layer=" << params_.layer_idx
                          << " flat_route=" << flat_route
                          << " value=" << indices[flat_route]);
                return false;
            }
            route_hash = hashWord(route_hash, active ? 1u : 0u);
            route_hash = hashWord(
                route_hash,
                active ? static_cast<uint64_t>(expert_id) : UINT64_MAX);
            if (active)
                ++state.expert_loads[static_cast<size_t>(expert_id)];
        }

        std::string lease_error;
        auto durable_lease =
            params_.residency_authority
                ->tryAcquireCurrentBatchLLEPLease(
                    params_.layer_idx,
                    params_.residency_domain,
                    params_.participant_count,
                    state.owner_participants,
                    &lease_error);
        if (!durable_lease)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Parent epoch lease "
                      "rejected for layer "
                      << params_.layer_idx << ": " << lease_error);
            return false;
        }
        const uint64_t durable_parent_epoch = durable_lease->epoch();
        std::fill(state.owner_mask.begin(), state.owner_mask.end(), false);
        for (int expert = 0; expert < params_.num_experts; ++expert)
        {
            const uint32_t owner =
                state.owner_participants[static_cast<size_t>(expert)];
            if (owner >= static_cast<uint32_t>(params_.participant_count))
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Authority published "
                          "an invalid domain-local owner for layer "
                          << params_.layer_idx << " expert=" << expert);
                return false;
            }
            state.owner_mask[static_cast<size_t>(expert)] =
                owner == static_cast<uint32_t>(params_.participant_id);
        }
        state.transient_resident_mask = state.owner_mask;

        /*
         * Route identity alone is insufficient when Dynamic maintenance may
         * publish E+1 between requests. Fold the exact parent epoch and owner
         * row into the first consensus record so no rank can plan against a
         * different durable generation or domain-local translation.
         */
        route_hash = hashWord(route_hash, durable_parent_epoch);
        for (const uint32_t owner : state.owner_participants)
            route_hash = hashWord(route_hash, owner);
        if (!verifyHashAgreement(
                route_hash, state.gathered_route_hashes, "router/epoch publication"))
        {
            return false;
        }

        least_loaded_ep::LeastLoadedExpertAssignmentWorkspace workspace{
            .sorted_experts = state.sorted_experts.data(),
            .pending_load = state.pending_load.data(),
            .assigned_load = state.assigned_load.data(),
        };
        const bool planned =
            least_loaded_ep::planLeastLoadedExpertAssignment(
                state.expert_loads.data(),
                state.owner_participants.data(),
                params_.planner_config,
                workspace,
                state.spans.data(),
                static_cast<uint32_t>(state.spans.size()),
                state.transfers.data(),
                static_cast<uint32_t>(state.transfers.size()),
                state.status);
        if (!planned || state.status.invalid_config != 0u ||
            state.status.overflow != 0u ||
            state.status.span_count > state.spans.size() ||
            state.status.weight_transfer_count > state.transfers.size())
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Deterministic planner failed"
                      << " layer=" << params_.layer_idx
                      << " invalid_config=" << state.status.invalid_config
                      << " overflow=" << state.status.overflow
                      << " spans=" << state.status.span_count
                      << " transfers=" << state.status.weight_transfer_count);
            return false;
        }

        state.expert_route_offsets[0] = 0u;
        for (int expert = 0; expert < params_.num_experts; ++expert)
        {
            state.expert_route_offsets[static_cast<size_t>(expert) + 1u] =
                state.expert_route_offsets[static_cast<size_t>(expert)] +
                state.expert_loads[static_cast<size_t>(expert)];
        }
        if (state.expert_route_offsets.back() >
            state.destination_by_expert_occurrence.size())
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Routed-row publication "
                      "exceeds graph-owned capacity for layer "
                      << params_.layer_idx);
            return false;
        }
        std::fill(state.destination_by_expert_occurrence.begin(),
                  state.destination_by_expert_occurrence.end(),
                  kUnassignedParticipant);

        if (state.status.standard_ep_selected != 0u)
        {
            for (int expert = 0; expert < params_.num_experts; ++expert)
            {
                const uint64_t begin =
                    state.expert_route_offsets[static_cast<size_t>(expert)];
                const uint64_t end =
                    state.expert_route_offsets[static_cast<size_t>(expert) + 1u];
                std::fill(
                    state.destination_by_expert_occurrence.begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    state.destination_by_expert_occurrence.begin() +
                        static_cast<std::ptrdiff_t>(end),
                    state.owner_participants[static_cast<size_t>(expert)]);
            }
        }
        else
        {
            for (uint32_t span_index = 0;
                 span_index < state.status.span_count;
                 ++span_index)
            {
                const auto &span = state.spans[span_index];
                if (span.expert >= static_cast<uint32_t>(params_.num_experts) ||
                    span.destination_participant >=
                        static_cast<uint32_t>(params_.participant_count) ||
                    span.route_row_begin > span.route_row_end ||
                    span.route_row_end > state.expert_loads[span.expert])
                {
                    LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Planner published "
                              "an invalid assignment span for layer "
                              << params_.layer_idx);
                    return false;
                }
                const uint64_t expert_begin =
                    state.expert_route_offsets[span.expert];
                for (uint64_t occurrence = span.route_row_begin;
                     occurrence < span.route_row_end;
                     ++occurrence)
                {
                    uint32_t &destination =
                        state.destination_by_expert_occurrence[
                            static_cast<size_t>(expert_begin + occurrence)];
                    if (destination != kUnassignedParticipant)
                    {
                        LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Planner spans "
                                  "overlap for layer " << params_.layer_idx);
                        return false;
                    }
                    destination = span.destination_participant;
                }
            }
        }

        std::fill(state.expert_route_cursors.begin(),
                  state.expert_route_cursors.end(), 0u);
        for (size_t flat_route = 0;
             flat_route < state.routeSlotCount();
             ++flat_route)
        {
            if (weights[flat_route] == 0.0f)
                continue;
            int expert_id = -1;
            if (!integralExpertId(
                    indices[flat_route], params_.num_experts, expert_id))
            {
                return false;
            }
            const uint64_t occurrence =
                state.expert_route_cursors[static_cast<size_t>(expert_id)]++;
            const uint64_t occurrence_index =
                state.expert_route_offsets[static_cast<size_t>(expert_id)] +
                occurrence;
            if (occurrence_index >=
                state.destination_by_expert_occurrence.size())
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Route occurrence "
                          "exceeds publication capacity for layer "
                          << params_.layer_idx);
                return false;
            }
            const uint32_t destination =
                state.destination_by_expert_occurrence[
                    static_cast<size_t>(occurrence_index)];
            if (destination >=
                static_cast<uint32_t>(params_.participant_count))
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Planner left an active "
                          "route unassigned for layer " << params_.layer_idx);
                return false;
            }
            state.destination_by_flat_route[flat_route] = destination;
        }

        uint64_t plan_hash = initialHash();
        plan_hash = hashWord(plan_hash, state.status.span_count);
        plan_hash = hashWord(plan_hash, state.status.weight_transfer_count);
        for (size_t flat_route = 0;
             flat_route < state.routeSlotCount();
             ++flat_route)
        {
            plan_hash = hashWord(
                plan_hash, state.destination_by_flat_route[flat_route]);
        }
        for (uint32_t transfer_index = 0;
             transfer_index < state.status.weight_transfer_count;
             ++transfer_index)
        {
            const auto &transfer = state.transfers[transfer_index];
            plan_hash = hashWord(plan_hash, transfer.expert);
            plan_hash = hashWord(plan_hash, transfer.source_participant);
            plan_hash = hashWord(plan_hash, transfer.destination_participant);
        }
        if (!verifyHashAgreement(
                plan_hash, state.gathered_plan_hashes, "assignment plan"))
        {
            return false;
        }

        state.transient_resident_mask = state.owner_mask;
        for (uint32_t transfer_index = 0;
             transfer_index < state.status.weight_transfer_count;
             ++transfer_index)
        {
            const auto &transfer = state.transfers[transfer_index];
            if (transfer.expert >= static_cast<uint32_t>(params_.num_experts) ||
                transfer.source_participant >=
                    static_cast<uint32_t>(params_.participant_count) ||
                transfer.destination_participant >=
                    static_cast<uint32_t>(params_.participant_count) ||
                transfer.source_participant == transfer.destination_participant)
            {
                LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Planner published an "
                          "invalid transfer for layer " << params_.layer_idx);
                return false;
            }
            if (transfer.destination_participant ==
                static_cast<uint32_t>(params_.participant_id))
            {
                state.transient_resident_mask[transfer.expert] = true;
            }
        }

        auto *global_tp = dynamic_cast<IGlobalTPContext *>(params_.tp_ctx);
        state.durable_parent_epoch = durable_parent_epoch;
        state.durable_domain = params_.residency_domain;
        const bool materialized =
            params_.physical_executor->materializeCPUCurrentBatchLLEPTransaction(
                state, *params_.expert_consumer, *global_tp);
        if (!materialized || !state.active)
        {
            state.durable_parent_epoch = 0;
            state.durable_domain.clear();
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Physical executor "
                      "did not publish an active transaction for layer "
                      << params_.layer_idx);
            return false;
        }
        state.durable_epoch_lease.emplace(std::move(*durable_lease));

        const auto tags = PerfStatsCollector::Tags{
            {"layer", std::to_string(params_.layer_idx)},
            {"participant", std::to_string(params_.participant_id)},
            {"participants", std::to_string(params_.participant_count)},
            {"seq_len", std::to_string(params_.seq_len)},
            {"top_k", std::to_string(params_.top_k)}};
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_plan_calls", 1.0,
            "prefill", params_.device_id.toString(), tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_assignment_spans",
            static_cast<double>(state.status.span_count),
            "prefill", params_.device_id.toString(), tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_weight_transfers",
            static_cast<double>(state.status.weight_transfer_count),
            "prefill", params_.device_id.toString(), tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_critical_path_payload_slots",
            static_cast<double>(
                state.status.critical_path_transfer_slots),
            "prefill", params_.device_id.toString(), tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_non_owner_rows",
            static_cast<double>(state.status.spilled_rows),
            "prefill", params_.device_id.toString(), tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_standard_ep_selections",
            static_cast<double>(state.status.standard_ep_selected),
            "prefill", params_.device_id.toString(), tags);
        return true;
    }

    bool MoECPUCurrentBatchLLEPStage::executeRestore()
    {
        auto &state = *params_.state;
        if (!state.active)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Restore requires one exact "
                      "active transaction for layer " << params_.layer_idx);
            return false;
        }
        if (!state.durable_epoch_lease.has_value() ||
            state.durable_parent_epoch == 0 ||
            state.durable_domain != params_.residency_domain)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Restore lost the exact "
                      "ExpertOverlay parent lease for layer "
                      << params_.layer_idx);
            return false;
        }
        auto *global_tp = dynamic_cast<IGlobalTPContext *>(params_.tp_ctx);
        const bool restored =
            params_.physical_executor->restoreCPUCurrentBatchLLEPPhysicalState(
                state, *params_.expert_consumer, *global_tp);
        if (!restored || state.active)
        {
            LOG_ERROR("[MoECPUCurrentBatchLLEPStage] Physical executor "
                      "failed to restore owner residency for layer "
                      << params_.layer_idx);
            return false;
        }
        const uint64_t durable_parent_epoch = state.durable_parent_epoch;
        state.durable_epoch_lease.reset();
        state.durable_parent_epoch = 0;
        state.durable_domain.clear();
        PerfStatsCollector::addCounter(
            "moe_rebalance", "cpu_llep_restore_calls", 1.0,
            "prefill", params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)},
             {"participant", std::to_string(params_.participant_id)},
             {"transaction_epoch",
              std::to_string(state.transaction_epoch)},
             {"durable_parent_epoch",
              std::to_string(durable_parent_epoch)}});
        return true;
    }

    bool MoECPUCurrentBatchLLEPStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoECPUCurrentBatchLLEPStage") ||
            ctx->deviceId() != params_.device_id || !validateBindings())
        {
            return false;
        }
        return params_.phase == CPUCurrentBatchLLEPPhase::Begin
                   ? executeBegin()
                   : executeRestore();
    }

    bool MoECPUCurrentBatchLLEPStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements
    MoECPUCurrentBatchLLEPStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_indices)
        {
            requirements.addInput(
                "routing_indices",
                params_.routing_indices->shape(),
                toBufferTensorType(params_.routing_indices->native_type()));
        }
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_weights)
        {
            requirements.addInput(
                "routing_weights",
                params_.routing_weights->shape(),
                toBufferTensorType(params_.routing_weights->native_type()));
        }
        return requirements;
    }

    StageBufferContract MoECPUCurrentBatchLLEPStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_indices_buffer_id)
        {
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        }
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_weights_buffer_id)
        {
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        }
        return contract;
    }

    StageDumpInfo MoECPUCurrentBatchLLEPStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_indices)
        {
            info.addInput(
                "routing_indices", params_.routing_indices,
                static_cast<size_t>(params_.seq_len),
                static_cast<size_t>(params_.top_k));
        }
        if (params_.phase == CPUCurrentBatchLLEPPhase::Begin &&
            params_.routing_weights)
        {
            info.addInput(
                "routing_weights", params_.routing_weights,
                static_cast<size_t>(params_.seq_len),
                static_cast<size_t>(params_.top_k));
        }
        info.addScalarInt("phase", static_cast<int>(params_.phase));
        info.addScalarInt("layer", params_.layer_idx);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("participant", params_.participant_id);
        info.addScalarInt("participants", params_.participant_count);
        if (params_.state)
        {
            info.addScalarBool("active", params_.state->active);
            info.addScalarInt(
                "assignment_spans",
                static_cast<int>(params_.state->status.span_count));
            info.addScalarInt(
                "weight_transfers",
                static_cast<int>(
                    params_.state->status.weight_transfer_count));
        }
        return info;
    }

} // namespace llaminar2
