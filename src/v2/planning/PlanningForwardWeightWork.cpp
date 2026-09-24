/**
 * @file PlanningForwardWeightWork.cpp
 * @brief Bind model operands to exact admitted TP/PP and phase-specific ownership.
 *
 * WeightShardGeometryResolver remains the one semantic sharding calculation;
 * PreparedWeightRepresentationContract remains the one conversion policy.
 * Expert operands use a single whole instance even when a tier has no initial
 * residency: the routing/capacity authority determines whether work is issued.
 * No inferred transfer path, physical allocation or execution timing lives here.
 */
#include "planning/PlanningForwardWeightWork.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    PlanningOrdinaryWeightKind planningOrdinaryWeightKind(WeightRole role) noexcept
    {
        switch (role)
        {
        case WeightRole::Embedding: return PlanningOrdinaryWeightKind::Embedding;
        case WeightRole::MoERouter: return PlanningOrdinaryWeightKind::Router;
        case WeightRole::LMHead: case WeightRole::AttentionQ: case WeightRole::AttentionK:
        case WeightRole::AttentionV: case WeightRole::AttentionWO: case WeightRole::FusedQKV:
        case WeightRole::GDNProjection: case WeightRole::GDNAlphaBetaProjection:
        case WeightRole::FFNGate: case WeightRole::FFNUp: case WeightRole::FFNDown:
        case WeightRole::SharedExpertGate: case WeightRole::SharedExpertUp: case WeightRole::SharedExpertDown:
            return PlanningOrdinaryWeightKind::Projection;
        case WeightRole::Norm: case WeightRole::OutputNorm: case WeightRole::Bias:
        case WeightRole::GDNSsmParam: case WeightRole::SharedExpertInputGate:
            return PlanningOrdinaryWeightKind::NonProjection;
        default: return PlanningOrdinaryWeightKind::Unclassified;
        }
    }

    PlanningExpertExecutionShare::PlanningExpertExecutionShare(
        PlanningExpertExecution execution, int experts, int assignment_participants)
        : execution_(execution), experts_(experts), assignment_participants_(assignment_participants)
    {
        if (experts < 0 || assignment_participants <= 0)
            throw std::invalid_argument("Expert work requires nonnegative population and positive assignment degree");
        switch (execution)
        {
        case PlanningExpertExecution::OwnedExperts:
        case PlanningExpertExecution::ReplicatedExperts:
            if (assignment_participants != 1)
                throw std::invalid_argument("Owned or replicated expert work cannot be divided by participant count");
            break;
        case PlanningExpertExecution::BalancedReplicaAssignment:
            break;
        default:
            throw std::invalid_argument("Unknown expert work execution policy");
        }
    }

    PlanningExpertWorkExpectation PlanningExpertExecutionShare::uniformExpectation(
        int model_experts, int top_k, size_t token_rows) const
    {
        if (model_experts <= 0 || top_k <= 0 || top_k > model_experts || experts_ > model_experts)
            throw std::invalid_argument("Uniform expert work requires a valid router population, top-k and eligible subset");
        if (token_rows == 0 || experts_ == 0) return {0.0, 0.0};
        const double selected = static_cast<double>(top_k) / model_experts;
        const double eligible = static_cast<double>(experts_) / assignment_participants_;
        // Only independence across tokens is assumed. Each token's top-k is
        // without replacement, so one particular expert is selected with K/E.
        // log1p/expm1 avoid losing tiny hit probabilities through cancellation.
        const double hit = top_k == model_experts ? 1.0 :
            -std::expm1(static_cast<double>(token_rows) * std::log1p(-selected));
        return {eligible * selected * static_cast<double>(token_rows), eligible * hit};
    }

    PlanningExpertWorkExpectation PlanningRoutedExpertWeightWork::uniformExpectation(size_t token_rows) const
    {
        if (execution_shares.empty())
            throw std::logic_error("Routed work has no phase-specific execution ownership");
        PlanningExpertWorkExpectation result{0.0, 0.0};
        for (const auto &share : execution_shares)
        {
            const auto work = share.uniformExpectation(expert_count, routes_per_token, token_rows);
            result.routed_rows += work.routed_rows;
            result.nonempty_experts += work.nonempty_experts;
        }
        return result;
    }

    namespace
    {
        /**
         * @brief Join a compiled endpoint to admitted logical tier quotas, not aggregate physical copies.
         *
         * One physical endpoint may participate in more than one logical tier.
         * Keep those populations separate: each tier has its own execution and
         * replica-assignment policy. Discovery rank IDs are not used in this
         * join because admission has already compacted execution membership.
         */
        std::vector<PlanningExpertExecutionShare> executionShares(
            const AdmittedOrchestrationCandidate &candidate, const DevicePlanConfig &device,
            std::span<const MoEOverlayBoundTierParticipant> participants, int layer,
            int model_experts, size_t ordinary_experts, PlanningMainForwardPhase phase)
        {
            const auto &capacity = candidate.overlayCapacity();
            if (!capacity)
            {
                const auto policy = candidate.rankPlans().at(device.world_rank).runtime.routed_expert_compute_policy;
                if (policy == RoutedExpertComputePolicy::Replicated)
                    return {{PlanningExpertExecution::ReplicatedExperts, model_experts}};
                if (policy != RoutedExpertComputePolicy::Apportioned || ordinary_experts > static_cast<size_t>(model_experts))
                    throw std::logic_error("Whole-expert planning work requires replicated or apportioned execution");
                return {{PlanningExpertExecution::OwnedExperts, static_cast<int>(ordinary_experts)}};
            }
            const auto &plan = *candidate.config().moe_routed_expert_plan;
            const auto footprint = std::find_if(capacity->layer_footprints.begin(), capacity->layer_footprints.end(),
                [&](const auto &entry) { return entry.layer_idx == layer; });
            if (footprint == capacity->layer_footprints.end() || capacity->num_experts != model_experts)
                throw std::logic_error("Expert work has no matching admitted layer/population");
            const size_t layer_index = static_cast<size_t>(footprint - capacity->layer_footprints.begin());
            std::vector<PlanningExpertExecutionShare> result;
            for (const auto &participant : participants)
            {
                if (participant.world_rank != device.world_rank || participant.device != device.device) continue;
                const auto &tier = plan.routed_tiers.at(participant.tier_index);
                const auto *quota = capacity->tier(participant.tier_index);
                const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                    [&](const auto &entry) { return entry.name == tier.domain; });
                if (!quota || domain == plan.domains.end())
                    throw std::logic_error("Expert work references an absent admitted tier/domain");
                const int copies = quota->participant_live_copies.at(participant.domain_participant_index).at(layer_index);
                if (domain->routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
                    result.emplace_back(PlanningExpertExecution::OwnedExperts, copies);
                else if (domain->routed_compute_policy == RoutedExpertComputePolicy::Replicated)
                {
                    if (copies != quota->live_experts_per_layer.at(layer_index))
                        throw std::logic_error("Replicated expert work requires the complete admitted tier population");
                    // Production already owns the phase predicate. Prefill's
                    // balanced assignment is an explicit prediction; decode
                    // executes the complete resident population on every replica.
                    const bool assigned = phase == PlanningMainForwardPhase::Prefill && domain->usesParticipantAssignedPrefill();
                    result.emplace_back(assigned ? PlanningExpertExecution::BalancedReplicaAssignment :
                        PlanningExpertExecution::ReplicatedExperts, copies,
                        assigned ? static_cast<int>(domain->participants.size()) : 1);
                }
                else throw std::logic_error("Overlay whole-expert work has an unsupported compute policy");
            }
            if (result.empty()) throw std::logic_error("Routed work endpoint is absent from its admitted tier topology");
            return result;
        }

        /** @return Whether this exact participant retains a named alternate weight authority. */
        bool hasSet(const DevicePlanConfig &device, AdditionalPersistentWeightSet set)
        {
            return std::find(device.additional_weight_sets.begin(), device.additional_weight_sets.end(), set) !=
                device.additional_weight_sets.end();
        }

        /** @brief Reject vector/parent metadata where a single projection is required. */
        void requireMatrix(const PlanningWeightOperand &weight)
        {
            const auto &matrix = weight.geometry.matrix();
            if (!matrix || matrix->instances != 1 || matrix->rows == 0 || matrix->columns == 0)
                throw std::invalid_argument("Main-forward projection has no complete single-matrix shape: " + weight.source_name);
        }

        /** @return Ordinary arithmetic category, with unknown roles retained explicitly. */
        PlanningOrdinaryWeightWork classify(PlanningWeightOperand weight)
        {
            switch (planningOrdinaryWeightKind(weight.role))
            {
            case PlanningOrdinaryWeightKind::Embedding:
                requireMatrix(weight);
                return PlanningEmbeddingWeight{std::move(weight)};
            case PlanningOrdinaryWeightKind::Projection:
                requireMatrix(weight);
                return PlanningProjectionWeight{std::move(weight)};
            case PlanningOrdinaryWeightKind::Router:
                requireMatrix(weight);
                return PlanningRouterWeight{std::move(weight)};
            case PlanningOrdinaryWeightKind::NonProjection:
                return PlanningNonProjectionWeight{std::move(weight)};
            default:
                return PlanningUnclassifiedWeight{std::move(weight)};
            }
        }
    }

    std::vector<PlanningParticipantWeightWork> compilePlanningForwardWeightWork(
        const PlanningModelMetadata &model, const AdmittedOrchestrationCandidate &candidate,
        PlanningMainForwardPhase phase)
    {
        if (phase != PlanningMainForwardPhase::Prefill && phase != PlanningMainForwardPhase::Decode)
            throw std::invalid_argument("Unknown main-forward planning phase");
        const auto &profile = model.memoryProfile();
        const int main_layers = model.mainLayerCount();
        const auto &ranks = candidate.rankPlans();
        const auto &membership = candidate.membership().discoveryRanks();
        const bool explicit_head = std::any_of(profile.tensors.begin(), profile.tensors.end(), [](const auto &tensor) {
            return inferWeightRole(tensor.name) == WeightRole::LMHead;
        });
        const auto embedding = std::find_if(profile.tensors.begin(), profile.tensors.end(), [](const auto &tensor) {
            return inferWeightRole(tensor.name) == WeightRole::Embedding;
        });
        std::vector<PlanningParticipantWeightWork> result;
        result.reserve(candidate.devicePlans().size());
        const auto participants = candidate.overlayCapacity()
            ? MoEOverlayCapacityAdmission::boundParticipants(*candidate.config().moe_routed_expert_plan)
            : std::vector<MoEOverlayBoundTierParticipant>{};
        for (const auto &device : candidate.devicePlans())
        {
            if (device.world_rank < 0 || static_cast<size_t>(device.world_rank) >= ranks.size() ||
                ranks[device.world_rank].rank != device.world_rank || device.activation_seq_len <= 0)
                throw std::logic_error("Main-forward work requires exact admitted rank and row geometry");
            const auto &rank = ranks[device.world_rank];
            const int first = std::max(0, device.first_layer);
            const int last = std::min(main_layers - 1, device.last_layer < 0 ? main_layers - 1 : device.last_layer);
            if (first > last)
                throw std::logic_error("Main-forward participant has no main-model layer interval");
            const bool continuation = device.execution_role == DeviceExecutionMemoryRole::ContinuationGraph;
            const bool routed = !candidate.overlayCapacity() || device.serial_routed_expert_participant_count > 0;
            const bool head = continuation && rank.has_lm_head && last == main_layers - 1;
            const bool decode = phase == PlanningMainForwardPhase::Decode;
            const bool full_decode = decode && hasSet(device, AdditionalPersistentWeightSet::ReplicatedDenseDecode);
            const bool full_embedding = full_decode || (decode && hasSet(device, AdditionalPersistentWeightSet::MirroredDecodeEmbedding));
            // Terminal ownership also applies to prefill's compact final-row
            // projection. It does not follow the transformer's row count or
            // whether this request currently enables speculative generation.
            const bool full_head = full_decode || device.mtp_terminal_logits_layout ==
                MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
            WeightShardGeometryResolver sharded(profile, device.device, device.shard_index, device.total_shards,
                device.tensor_parallel_assignment);
            WeightShardGeometryResolver replicated(profile, device.device);
            PlanningParticipantWeightWork work{device.world_rank, membership.at(device.world_rank), device.device,
                first, last, device.activation_seq_len, {}, {}};
            std::map<int, std::array<std::optional<PlanningWeightOperand>, 3>> experts;
            std::map<int, size_t> ordinary_expert_counts;

            // Materialized decode mirrors replace the primary invocation view;
            // they do not add a second execution of each projection. A tied
            // output head similarly reuses source bytes with different semantics.
            const auto operand = [&](const TensorSizeInfo &tensor, WeightRole role, bool complete,
                                     std::optional<size_t> expert_instances = {}) {
                const auto &resolver = complete ? replicated : sharded;
                auto geometry = [&] {
                    if (role == WeightRole::LMHead && inferWeightRole(tensor.name) == WeightRole::Embedding)
                    {
                        auto semantic = tensor;
                        // Ask the canonical schema for head, not embedding,
                        // sharding. All other source metadata stays borrowed.
                        semantic.name = "output.weight";
                        return resolver.resolve(semantic, expert_instances);
                    }
                    return resolver.resolve(tensor, expert_instances);
                }();
                return PlanningWeightOperand{tensor.name, role, tensor.layer_index, tensor.quant_type,
                    PreparedWeightRepresentationContract::resolve(role, tensor.quant_type),
                    std::move(geometry)};
            };
            for (const auto &tensor : profile.tensors)
            {
                // Learned predictors are a separate transaction. Retaining depth
                // fifteen does not execute fifteen extra main blocks per token.
                if (tensor.layer_index >= main_layers ||
                    (tensor.layer_index >= 0 && (tensor.layer_index < first || tensor.layer_index > last))) continue;
                const auto role = inferWeightRole(tensor.name);
                if (isRoutedExpertRole(role))
                {
                    // A dense-only continuation domain owns router work, not
                    // an expert service endpoint. Zero initial quota is a
                    // different case: its bound endpoint still has this shape.
                    if (!routed) continue;
                    if (tensor.layer_index < 0 || profile.expert_count <= 0 || profile.expert_used_count <= 0 ||
                        profile.expert_used_count > profile.expert_count)
                        throw std::invalid_argument("Routed work has invalid layer or top-k geometry");
                    const size_t slot = role == WeightRole::MoEExpertGate ? 0 : role == WeightRole::MoEExpertUp ? 1 : 2;
                    auto &entry = experts[tensor.layer_index][slot];
                    if (entry) throw std::invalid_argument("Duplicate routed projection: " + tensor.name);
                    entry = operand(tensor, role, true, 1);
                    requireMatrix(*entry);
                    if (!candidate.overlayCapacity())
                    {
                        const auto local = sharded.resolve(tensor).matrix();
                        if (!local) throw std::invalid_argument("Routed source has no participant geometry");
                        const auto [position, inserted] = ordinary_expert_counts.emplace(tensor.layer_index, local->instances);
                        if (!inserted && position->second != local->instances)
                            throw std::invalid_argument("Routed projections disagree on participant expert ownership");
                    }
                    continue;
                }
                if (!continuation) continue;
                if (role == WeightRole::Embedding && !device.owns_embedding) continue;
                if ((role == WeightRole::LMHead || role == WeightRole::OutputNorm) && !head) continue;
                const bool complete = role == WeightRole::Embedding ? full_embedding : role == WeightRole::LMHead ? full_head : full_decode;
                work.ordinary.push_back(classify(operand(tensor, role, complete)));
            }
            if (head && !explicit_head)
            {
                if (embedding == profile.tensors.end())
                    throw std::invalid_argument("Main-forward output has neither explicit nor tied source weights");
                work.ordinary.push_back(classify(operand(*embedding, WeightRole::LMHead, full_head)));
            }
            for (auto &[layer, triplet] : experts)
            {
                if (!triplet[0] || !triplet[1] || !triplet[2])
                    throw std::invalid_argument("Incomplete routed FFN at layer " + std::to_string(layer));
                const auto &gate = *triplet[0]->geometry.matrix();
                const auto &up = *triplet[1]->geometry.matrix();
                const auto &down = *triplet[2]->geometry.matrix();
                if (gate != up || gate.rows != down.columns || gate.columns != down.rows)
                    throw std::invalid_argument("Incompatible routed FFN projection geometry at layer " + std::to_string(layer));
                work.routed.push_back({layer, profile.expert_count, profile.expert_used_count,
                    {std::move(*triplet[0]), std::move(*triplet[1]), std::move(*triplet[2])},
                    executionShares(candidate, device, participants, layer, profile.expert_count,
                        candidate.overlayCapacity() ? 0 : ordinary_expert_counts.at(layer), phase)});
            }
            result.push_back(std::move(work));
        }
        if (result.empty()) throw std::logic_error("Main-forward work has no admitted participants");
        return result;
    }
}
