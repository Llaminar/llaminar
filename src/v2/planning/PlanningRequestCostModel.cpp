/**
 * @file PlanningRequestCostModel.cpp
 * @brief Pure request composition over one bounded discovery-time measurement batch.
 *
 * Compute uses per-format source samples and independent streaming bandwidth.
 * State and scalar operations use explicitly qualified FP32 roofline proxies.
 * Communication predictions retain physical endpoints and report when a native
 * group or host request/reply is a volume proxy rather than the exact production
 * protocol. Predictions never change that protocol, precision or placement.
 */
#include "PlanningRequestCostModel.h"
#include "AutomaticPlanningStartup.h"
#include "PlanningForwardStateWork.h"
#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @return Positive representable whole-byte payload for a fractional routing expectation. */
        size_t payload(double bytes)
        {
            if (!std::isfinite(bytes) || bytes < 0 ||
                static_cast<long double>(bytes) >= static_cast<long double>(std::numeric_limits<size_t>::max()))
                throw std::overflow_error("Planning request payload is not representable");
            // Even an expected empty packet participates in a control epoch.
            // This one-byte query charges the measured control floor, not a
            // fabricated zero-bandwidth message or a reserved maximum buffer.
            return std::max(size_t{1}, static_cast<size_t>(std::ceil(bytes)));
        }

        /** @return Whether this phase executes the primary sharded dense view. */
        bool denseSharded(const DevicePlanConfig &device, PlanningMainForwardPhase phase)
        {
            return device.total_shards > 1 && !(phase == PlanningMainForwardPhase::Decode &&
                std::find(device.additional_weight_sets.begin(), device.additional_weight_sets.end(),
                    AdditionalPersistentWeightSet::ReplicatedDenseDecode) != device.additional_weight_sets.end());
        }
    }

    struct PlanningRequestCostModel::InvocationCost
    {
        double compute = 0;
        double communication = 0;
        std::set<std::string> qualifications;

        /** @return Dependent compute and communication; no speculative overlap credit. */
        double seconds() const { return planningSerialSeconds(std::array{compute, communication}); }
    };

    std::optional<AutomaticOrchestrationPlanner::Evaluate> PlanningRequestCostModel::prepare(
        const AutomaticPlanningPreparation &context)
    {
        auto weights = PlanningWeightServiceModel::collect(context);
        std::optional<AutomaticOrchestrationRequest> request;
        acceptPlanningCostPreparation(context.mpi(), [&] {
            request = std::get<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(context.request()));
        });
        auto arithmetic = PlanningKernelServiceCatalog::collect(context.mpi(), context.inventory(), *request,
            PlanningFP32ArithmeticPlan{});
        // This is a bounded FP32 link-service basis, not a precision override
        // for inference. The final evidence labels volume proxies explicitly.
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        auto communication = PlanningCommunicationService::collect(context.mpi(), context.inventory(), *request,
            context.metadata().memoryProfile().d_model, 64, precisions);
        if (!context.isRoot()) return std::nullopt;
        auto service = std::make_shared<const PlanningRequestCostModel>(context.metadata(), context.inventory(),
            std::move(*weights), std::move(*arithmetic), PlanningCommunicationCost(context.inventory(),
                communication->native(), communication->mpi(), communication->hostDevice()));
        // The closure owns immutable evidence. It borrows neither the root
        // GGUF reader nor the preparation stack and retains no GPU resources.
        return [service](const auto &candidate, const auto &workload) { return service->evaluate(candidate, workload); };
    }

    PlanningRequestCostModel::PlanningRequestCostModel(PlanningModelMetadata model, ClusterInventory inventory,
        PlanningWeightServiceModel weights, PlanningKernelServiceCatalog arithmetic,
        PlanningCommunicationCost communication)
        : model_(std::move(model)), inventory_(std::move(inventory)), weights_(std::move(weights)),
          arithmetic_(std::move(arithmetic)), communication_(std::move(communication))
    {
        const auto &memory = weights_.streamingEvidence();
        if (!std::holds_alternative<PlanningFP32ArithmeticPlan>(arithmetic_.source()) ||
            arithmetic_.records().size() != memory.records().size())
            throw std::invalid_argument("Request cost requires a complete dedicated FP32 arithmetic basis");
        for (const auto &record : memory.records())
        {
            const auto &observer = record.observer;
            const auto &compute = arithmetic_.serviceFor(observer.discoveryRank(), observer.device());
            if (compute.observer != observer)
                throw std::invalid_argument("Request cost arithmetic and streaming observations have different physical owners");
            const auto &fp32 = std::get<PlanningFP32ArithmeticObservations>(compute.observation);
            const auto &stream = std::get<PlanningMemoryBandwidthObservation>(record.observation);
            if (observer.device().is_cpu() && (!fp32.cpu || fp32.cpu->workers != stream.request.workers() ||
                    fp32.cpu->execution != stream.request.cpuGeometry()))
                throw std::invalid_argument("Request cost arithmetic and streaming observations have different CPU workshares");
        }
        for (const auto &tensor : model_.memoryProfile().tensors)
        {
            if (!tensor.elements || !tensor.native_bytes ||
                !source_bytes_per_element_.emplace(tensor.name, double(tensor.native_bytes) / tensor.elements).second)
                throw std::invalid_argument("Request cost requires a complete unique native tensor inventory");
        }
    }

    double PlanningRequestCostModel::parameterBytes(const PlanningWeightOperand &weight) const
    {
        const auto found = source_bytes_per_element_.find(weight.source_name);
        if (found == source_bytes_per_element_.end())
            throw std::invalid_argument("Request operand does not belong to its measured model");
        return weight.geometry.elements() * (weight.representation == ModelPreparedWeightRepresentation::FP32
            ? sizeof(float) : found->second);
    }

    double PlanningRequestCostModel::scalarSeconds(int rank, DeviceId device, int rows,
        double operations, double bytes) const
    {
        const auto &observation = std::get<PlanningFP32ArithmeticObservations>(arithmetic_.serviceFor(rank, device).observation);
        const auto &decode = observation.phases.at(0);
        const auto &prefill = observation.phases.at(1);
        return std::max(planningArithmeticSeconds(decode.service, prefill.rows, prefill.service, rows, operations),
            weights_.memorySeconds(rank, device, bytes));
    }

    PlanningRequestCostModel::InvocationCost PlanningRequestCostModel::invocation(
        const AdmittedOrchestrationCandidate &candidate, PlanningMainForwardPhase phase, int rows, double context) const
    {
        const auto work = compilePlanningForwardWeightWork(model_, candidate, phase);
        const auto &devices = candidate.devicePlans();
        if (work.size() != devices.size()) throw std::logic_error("Request work lost compiled participant identity");
        const auto &profile = model_.memoryProfile();
        const int layers = model_.mainLayerCount();
        InvocationCost result;
        result.qualifications.insert("same-format shape extrapolation; uniform distinct top-k; no future migration benefit");
        result.qualifications.insert("FP32 scalar/attention/GDN roofline proxies with ideal invocation-local state reuse");
        result.qualifications.insert("coarse layer joins; dependent layers and communications; no transfer/compute overlap credit");

        /** @brief Candidate-local arithmetic only; no physical byte reservations or live accounting. */
        struct ParticipantCost
        {
            double entry = 0, terminal = 0;
            std::vector<double> ordinary, experts;
            std::vector<const PlanningRoutedExpertWeightWork *> routed;
        };
        std::vector<ParticipantCost> local(work.size());
        for (size_t index = 0; index < work.size(); ++index)
        {
            const auto &participant = work[index];
            if (participant.execution_rank != devices[index].world_rank || participant.device != devices[index].device)
                throw std::logic_error("Request work and compiled device order disagree");
            auto &cost = local[index];
            cost.ordinary.resize(layers);
            cost.experts.resize(layers);
            cost.routed.resize(layers);
            for (const auto &operation : participant.ordinary)
                std::visit([&](const auto &value) {
                    using Operation = std::decay_t<decltype(value)>;
                    const auto &weight = value.weight;
                    const int count = weight.role == WeightRole::LMHead || weight.role == WeightRole::OutputNorm ? 1 : rows;
                    double seconds = 0;
                    if constexpr (std::is_same_v<Operation, PlanningUnclassifiedWeight>)
                        throw std::invalid_argument("Unclassified request work: " + weight.source_name);
                    else if constexpr (std::is_same_v<Operation, PlanningProjectionWeight> ||
                                       std::is_same_v<Operation, PlanningRouterWeight>)
                    {
                        seconds = weights_.projectionSeconds(participant.discovery_rank, participant.device, weight, count);
                        const auto &shape = *weight.geometry.matrix();
                        if constexpr (std::is_same_v<Operation, PlanningRouterWeight>)
                        {
                            result.qualifications.insert("router uses same-format projection plus FP32 top-k work proxy");
                            seconds += scalarSeconds(participant.discovery_rank, participant.device, count,
                                double(count) * shape.rows * (5 + profile.expert_used_count),
                                sizeof(float) * double(count) * (shape.rows + 2 * profile.expert_used_count));
                        }
                        if (weight.role == WeightRole::FFNGate || weight.role == WeightRole::SharedExpertGate)
                            seconds += scalarSeconds(participant.discovery_rank, participant.device, count,
                                6.0 * count * shape.rows, 3.0 * sizeof(float) * count * shape.rows);
                        if (weight.role == WeightRole::LMHead)
                            seconds += scalarSeconds(participant.discovery_rank, participant.device, 1,
                                5.0 * profile.vocab_size, 3.0 * sizeof(float) * profile.vocab_size);
                    }
                    else if constexpr (std::is_same_v<Operation, PlanningEmbeddingWeight>)
                    {
                        const auto &shape = weight.geometry.matrix();
                        if (!shape || !shape->rows) throw std::invalid_argument("Embedding work lacks its table geometry");
                        // One indexed row per token, never the entire table.
                        // A vocabulary-sharded participant sees the uniform
                        // token-share proxy; its collective is priced separately.
                        const double share = double(shape->rows) / profile.vocab_size;
                        seconds = scalarSeconds(participant.discovery_rank, participant.device, count,
                            2.0 * count * shape->columns * share,
                            count * (parameterBytes(weight) / shape->rows * share + sizeof(float) * profile.d_model));
                    }
                    else
                    {
                        const double elements = weight.geometry.elements();
                        const bool norm = weight.role == WeightRole::Norm || weight.role == WeightRole::OutputNorm;
                        const double coordinates = norm ? std::max(elements, double(profile.d_model)) : elements;
                        const double operations = weight.role == WeightRole::GDNSsmParam ? 0 :
                            count * coordinates * (norm ? 8.0 : weight.role == WeightRole::SharedExpertInputGate ? 2.0 : 1.0);
                        seconds = scalarSeconds(participant.discovery_rank, participant.device, count, operations,
                            parameterBytes(weight) + 2.0 * sizeof(float) * count * coordinates);
                    }
                    if (weight.role == WeightRole::Embedding) cost.entry += seconds;
                    else if (weight.role == WeightRole::LMHead || weight.role == WeightRole::OutputNorm) cost.terminal += seconds;
                    else
                    {
                        if (weight.layer < 0 || weight.layer >= layers)
                            throw std::invalid_argument("Ordinary request operation has no main-layer owner");
                        cost.ordinary[weight.layer] += seconds;
                    }
                }, operation);

            const auto &fp32 = std::get<PlanningFP32ArithmeticObservations>(
                arithmetic_.serviceFor(participant.discovery_rank, participant.device).observation);
            const auto &memory = std::get<PlanningMemoryBandwidthObservation>(
                weights_.streamingEvidence().serviceFor(participant.discovery_rank, participant.device).observation);
            for (const auto &state : compilePlanningForwardStateWork(model_, devices[index], phase, {rows, context}))
            {
                cost.ordinary[state.layer] += planningStateServiceSeconds(state, fp32, memory);
                // Two residual additions per transformer block. Parameterized
                // norms are already charged by their own semantic operands.
                cost.ordinary[state.layer] += scalarSeconds(participant.discovery_rank, participant.device, rows,
                    2.0 * rows * profile.d_model, 6.0 * sizeof(float) * rows * profile.d_model);
            }
            for (const auto &expert : participant.routed)
            {
                cost.experts.at(expert.layer) = weights_.expertSeconds(participant.discovery_rank, participant.device, expert, rows);
                cost.routed.at(expert.layer) = &expert;
            }
        }

        // A link estimate never changes physical host membership or the actual
        // selected transport. Same-node mapped epochs and remote host MPI stay
        // distinct. Point-to-point native links use a clearly qualified group
        // service proxy: we do not claim to have sampled every candidate pair.
        const auto exchange = [&](size_t source, size_t target, size_t outgoing, size_t incoming, bool mapped) {
            const auto &a = work[source];
            const auto &b = work[target];
            if (a.discovery_rank == b.discovery_rank && a.device == b.device) return 0.0;
            const auto topology = inventory_.connectionBetweenRanks(a.discovery_rank, b.discovery_rank);
            const bool same_node = topology.locality() != RankConnectionLocality::CrossNode;
            if (a.discovery_rank == b.discovery_rank && a.device.is_gpu() && b.device.type == a.device.type)
            {
                const std::array group{a.device, b.device};
                result.qualifications.insert("same-process GPU point-to-point traffic uses a native FP32 group-service volume proxy");
                const size_t row_bytes = size_t(profile.d_model) * sizeof(float);
                const size_t bytes = std::max(outgoing, incoming);
                const size_t equivalent_rows = bytes / row_bytes + (bytes % row_bytes != 0);
                if (equivalent_rows > size_t(std::numeric_limits<int>::max()))
                    throw std::overflow_error("Planning native traffic exceeds representable row geometry");
                const auto prediction = communication_.nativeAllreduce(a.discovery_rank, group, profile.d_model,
                    static_cast<int>(std::max(size_t{1}, equivalent_rows)), PlanningAllreducePrecision::FP32);
                result.qualifications.insert(prediction.evidence);
                return prediction.seconds;
            }
            double seconds = 0;
            const auto staging = [&](const auto &gpu, int pages, MappedTransferDirection direction, size_t bytes) {
                return communication_.hostTransfer(gpu.discovery_rank, gpu.device, pages,
                    mapped && same_node ? PlanningHostTransferMechanism::MappedKernel : PlanningHostTransferMechanism::DMA,
                    direction, bytes).seconds;
            };
            if (mapped && same_node)
            {
                result.qualifications.insert("node-local activation epochs use directed mapped-byte primitives; packet/control overhead not separately fitted");
                if (a.device.is_gpu())
                {
                    seconds += staging(a, b.discovery_rank, MappedTransferDirection::DeviceToHost, outgoing);
                    seconds += staging(a, a.discovery_rank, MappedTransferDirection::HostToDevice, incoming);
                }
                if (b.device.is_gpu())
                {
                    seconds += staging(b, b.discovery_rank, MappedTransferDirection::HostToDevice, outgoing);
                    seconds += staging(b, a.discovery_rank, MappedTransferDirection::DeviceToHost, incoming);
                }
                if (a.device.is_cpu()) seconds += weights_.memorySeconds(a.discovery_rank, a.device, double(outgoing) + incoming);
                if (b.device.is_cpu()) seconds += weights_.memorySeconds(b.discovery_rank, b.device, double(outgoing) + incoming);
            }
            else
            {
                if (a.discovery_rank != b.discovery_rank)
                {
                    result.qualifications.insert("host exchanges retain one measured directed control RTT; collective/PP traffic uses an explicit serial request-reply proxy");
                    seconds += communication_.mpiExchange({a.discovery_rank, b.discovery_rank, outgoing, incoming}).seconds;
                }
                if (a.device.is_gpu())
                {
                    seconds += staging(a, a.discovery_rank, MappedTransferDirection::DeviceToHost, outgoing);
                    seconds += staging(a, a.discovery_rank, MappedTransferDirection::HostToDevice, incoming);
                }
                if (b.device.is_gpu())
                {
                    seconds += staging(b, b.discovery_rank, MappedTransferDirection::HostToDevice, outgoing);
                    seconds += staging(b, b.discovery_rank, MappedTransferDirection::DeviceToHost, incoming);
                }
            }
            if (!std::isfinite(seconds) || seconds <= 0)
                throw std::invalid_argument("A nonlocal request edge has no positive measured service basis");
            return seconds;
        };

        const auto reduction = [&](const std::vector<size_t> &group) {
            if (group.size() <= 1) return 0.0;
            const auto &root = work[group.front()];
            const bool native = root.device.is_gpu() && std::all_of(group.begin(), group.end(), [&](size_t index) {
                return work[index].discovery_rank == root.discovery_rank && work[index].device.type == root.device.type;
            });
            if (native)
            {
                std::vector<DeviceId> endpoints;
                for (size_t index : group) endpoints.push_back(work[index].device);
                result.qualifications.insert("TP reductions use measured native FP32 group service; schema precision and canonical-order kernels are volume proxies, not sampled model collectives");
                const auto prediction = communication_.nativeAllreduce(root.discovery_rank, endpoints, profile.d_model, rows,
                    PlanningAllreducePrecision::FP32);
                result.qualifications.insert(prediction.evidence);
                return prediction.seconds;
            }
            double seconds = 0;
            const size_t bytes = payload(double(rows) * profile.d_model * sizeof(float));
            for (size_t index = 1; index < group.size(); ++index)
                seconds += exchange(group.front(), group[index], bytes, bytes, false);
            return seconds;
        };

        std::vector<double> entries, terminals;
        for (const auto &cost : local) { entries.push_back(cost.entry); terminals.push_back(cost.terminal); }
        result.compute = planningIndependentSeconds(entries) + planningIndependentSeconds(terminals);
        std::optional<size_t> overlay_root;
        if (candidate.overlayCapacity())
        {
            const auto &placement = *candidate.config().moe_routed_expert_plan;
            const auto bindings = MoEOverlayCapacityAdmission::boundParticipants(placement);
            const auto root = std::find_if(bindings.begin(), bindings.end(), [&](const auto &binding) {
                return binding.participant_id == placement.continuation_domain_spec.logical_root_participant;
            });
            if (root == bindings.end()) throw std::logic_error("Request overlay has no bound logical root");
            for (size_t index = 0; index < work.size(); ++index)
                if (work[index].execution_rank == root->world_rank && work[index].device == root->device)
                    overlay_root = index;
            if (!overlay_root) throw std::logic_error("Request overlay root is not a compiled participant");
        }
        std::vector<size_t> previous;
        for (int layer = 0; layer < layers; ++layer)
        {
            std::vector<size_t> continuation, tp;
            std::vector<double> ordinary, experts;
            for (size_t index = 0; index < work.size(); ++index)
            {
                if (work[index].first_layer > layer || work[index].last_layer < layer) continue;
                ordinary.push_back(local[index].ordinary[layer]);
                experts.push_back(local[index].experts[layer]);
                if (devices[index].execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
                {
                    continuation.push_back(index);
                    if (denseSharded(devices[index], phase)) tp.push_back(index);
                }
            }
            if (continuation.empty()) throw std::logic_error("Request layer has no compiled continuation owner");
            if (!previous.empty() && previous != continuation)
                result.communication += exchange(previous.front(), continuation.front(),
                    payload(double(rows) * profile.d_model * sizeof(float)), 1, false);
            if (layer == 0)
            {
                const bool embedding_sharded = std::any_of(continuation.begin(), continuation.end(), [&](size_t i) {
                    return std::any_of(work[i].ordinary.begin(), work[i].ordinary.end(), [&](const auto &operation) {
                        const auto *embedding = std::get_if<PlanningEmbeddingWeight>(&operation);
                        return embedding && embedding->weight.geometry.matrix()->rows < size_t(profile.vocab_size);
                    });
                });
                if (embedding_sharded) result.communication += reduction(continuation);
            }
            result.compute += planningIndependentSeconds(ordinary) + planningIndependentSeconds(experts);
            result.communication += 2 * reduction(tp); // Attention and FFN row-parallel joins.
            // Ordinary TP replicas do not become a sparse overlay: their
            // row-parallel joins were already charged above. Overlay routing
            // uses its bound logical root, never vector order or discovery rank.
            for (size_t index = 0; index < work.size(); ++index)
            {
                const auto *expert = local[index].routed[layer];
                if (!overlay_root || !expert || index == *overlay_root ||
                    devices[index].execution_role != DeviceExecutionMemoryRole::RoutedExpertParticipant) continue;
                const auto expected = expert->uniformExpectation(rows);
                // Expected routes bound compact unique rows. Canonical route
                // return preserves each expert contribution rather than an
                // unrelated maximum packet/buffer capacity.
                const auto entries = static_cast<uint64_t>(std::ceil(expected.routed_rows));
                const auto unique_rows = std::min<uint64_t>(rows, entries);
                const auto dispatch = moeOverlayDispatchPayloadBytes(unique_rows, entries, profile.d_model);
                const auto reply = moeOverlayReturnPayloadBytes(entries, profile.d_model);
                if (entries && (!dispatch || !reply))
                    throw std::overflow_error("Planning overlay traffic does not fit the production packet ABI");
                result.communication += exchange(*overlay_root, index, payload(dispatch), payload(reply), true);
            }
            previous = std::move(continuation);
        }
        return result;
    }

    OrchestrationCostEstimate PlanningRequestCostModel::evaluate(const AdmittedOrchestrationCandidate &candidate,
        const OrchestrationPlanningWorkload &workload) const
    {
        workload.requireFitsContext(candidate.config().max_seq_len);
        int capacity = workload.promptTokens();
        for (const auto &device : candidate.devicePlans())
        {
            if (device.activation_seq_len <= 0) throw std::logic_error("Request cost has no admitted row capacity");
            capacity = std::min(capacity, device.activation_seq_len);
            if (device.serial_routed_expert_compact_rows > 0)
                capacity = std::min(capacity, device.serial_routed_expert_compact_rows);
        }
        for (const auto &rank : candidate.rankPlans())
            if (rank.runtime.overlay_prefill_schedule.enabled())
                capacity = std::min(capacity, rank.runtime.overlay_prefill_schedule.graph_row_capacity);
        const int complete = workload.promptTokens() / capacity;
        const int tail = workload.promptTokens() % capacity;
        // Full chunks have identical weights/geometry. Evaluate their mean
        // context once, plus the tail, rather than simulate thousands of
        // synthetic tokens for every candidate. Roofline crossover is an
        // explicit approximation; causal arithmetic retains the exact sum.
        auto prefill = invocation(candidate, PlanningMainForwardPhase::Prefill, capacity,
            0.5 * (complete - 1.0) * capacity);
        prefill.compute *= complete;
        prefill.communication *= complete;
        if (tail)
        {
            const auto remainder = invocation(candidate, PlanningMainForwardPhase::Prefill, tail, double(complete) * capacity);
            prefill.compute += remainder.compute;
            prefill.communication += remainder.communication;
            prefill.qualifications.insert(remainder.qualifications.begin(), remainder.qualifications.end());
        }
        const auto decode = invocation(candidate, PlanningMainForwardPhase::Decode, 1,
            workload.promptTokens() + 0.5 * (workload.generationTokens() - 1.0));
        std::ostringstream evidence;
        evidence << "bounded measured ranking; predictions, not model benchmarks; serial main-forward baseline (no assumed MTP acceptance); "
                 << "prefill_compute_s=" << prefill.compute << "; prefill_interconnect_s=" << prefill.communication
                 << "; mean_decode_compute_s=" << decode.compute << "; mean_decode_interconnect_s=" << decode.communication
                 << "; admitted_prefill_rows=" << capacity << "; mean-context roofline approximation";
        prefill.qualifications.insert(decode.qualifications.begin(), decode.qualifications.end());
        for (const auto &qualification : prefill.qualifications) evidence << "; " << qualification;
        return {workload, prefill.seconds(), decode.seconds(), evidence.str()};
    }
}
