/**
 * @file PlanningRequestCostModel.cpp
 * @brief Pure request composition over one bounded discovery-time measurement batch.
 *
 * Compute uses per-format source samples and independent streaming bandwidth.
 * State and scalar operations use explicitly qualified FP32 roofline proxies.
 * Same-class GPU prediction pooling removes short-sample placement noise without
 * rewriting raw endpoint evidence, physical admission or directed link costs.
 * Communication predictions retain physical endpoints and report when a native
 * group or host request/reply is a volume proxy rather than the exact production
 * protocol. Predictions never change that protocol, precision or placement.
 */
#include "PlanningRequestCostModel.h"
#include "AutomaticPlanningStartup.h"
#include "PlanningForwardStateWork.h"
#include "PlanningImmutableLookupWork.h"
#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEGroupedIntermediateExchangeABI.h"
#include "collective/DeviceCountedAllGather.h"
#include "interfaces/IMPIContext.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "utils/Logger.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

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
        auto phase_begin = std::chrono::steady_clock::now();
        // Include preparation, publication and resource retirement in each
        // evidence family. Native sample intervals alone hide startup costs.
        const auto completed_phase = [&](const char *name) {
            const auto now = std::chrono::steady_clock::now();
            LOG_INFO("[PlanningEvidenceWall] discovery_rank=" << (context.mpi() ? context.mpi()->rank() : 0)
                << " phase=" << name << " seconds="
                << std::chrono::duration<double>(now - phase_begin).count());
            phase_begin = now;
        };
        auto weights = PlanningWeightServiceModel::collect(context);
        completed_phase("weights_and_streaming");
        std::optional<AutomaticOrchestrationRequest> request;
        acceptPlanningCostPreparation(context.mpi(), [&] {
            request = std::get<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(context.request()));
        });
        auto arithmetic = PlanningKernelServiceCatalog::collect(context.mpi(), context.inventory(), *request,
            PlanningFP32ArithmeticPlan{});
        completed_phase("fp32_arithmetic");
        // This is a bounded FP32 link-service basis, not a precision override
        // for inference. The final evidence labels volume proxies explicitly.
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        auto communication = PlanningCommunicationService::collect(context.mpi(), context.inventory(), *request,
            context.metadata().memoryProfile().d_model, 64, precisions,
            context.request().routed_expert_compute_policy,
            PlanningLocalTPMeasurementMode::NativeCollective);
        completed_phase("communication");
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
        prepareComputeCohorts();
        for (const auto &tensor : model_.memoryProfile().tensors)
        {
            if (!tensor.elements || !tensor.native_bytes ||
                !source_bytes_per_element_.emplace(tensor.name, double(tensor.native_bytes) / tensor.elements).second)
                throw std::invalid_argument("Request cost requires a complete unique native tensor inventory");
        }
    }

    void PlanningRequestCostModel::prepareComputeCohorts()
    {
        // PCI location and measured speed are deliberately absent: they remain
        // physical-link evidence, not compute-class identity. Unknown required
        // hardware fields never authorize pooling with another unknown device.
        using Class=std::tuple<int,int,int,DeviceType,std::string,std::string,int,int,int,size_t,size_t,float,float,float>;
        std::map<Class,std::vector<const PlanningServiceObserver*>> classes;
        for(const auto &record:weights_.streamingEvidence().records()) {
            const auto &observer=record.observer;const auto device=observer.device();
            if(!device.is_gpu())continue;
            compute_cohorts_[{observer.discoveryRank(),device}]={device};
            const auto rank=std::find_if(inventory_.ranks.begin(),inventory_.ranks.end(),
                [&](const auto &r){return r.rank==observer.discoveryRank();});
            if(rank==inventory_.ranks.end() || rank->node_id!=observer.physicalNode())
                throw std::invalid_argument("Compute class lost its original discovery owner");
            const auto found=std::find_if(rank->gpus.begin(),rank->gpus.end(),[&](const auto &g){
                return g.type==device.type && g.local_device_id==device.ordinal;});
            if(found==rank->gpus.end() || found->uuid!=observer.uuid() || found->numa_node!=observer.numaNode())
                throw std::invalid_argument("Compute class differs from authenticated physical observer");
            const auto &g=*found;
            const bool isa=device.is_cuda() ? g.compute_capability_major>0 && g.compute_capability_minor>=0 :
                device.is_rocm() && !g.native_isa.empty();
            if(!isa || g.name.empty() || g.uuid.empty() || g.numa_node<0 || g.compute_units<=0 ||
               !g.memory_bytes || !g.last_level_cache_bytes ||
               !std::isfinite(g.tflops_fp16) || !std::isfinite(g.tflops_int8) || !std::isfinite(g.memory_bandwidth_gbps) ||
               g.tflops_fp16<0 || g.tflops_int8<0 || g.memory_bandwidth_gbps<0)continue;
            classes[{observer.discoveryRank(),observer.physicalNode(),observer.numaNode(),device.type,g.name,g.native_isa,
                g.compute_units,g.compute_capability_major,g.compute_capability_minor,g.memory_bytes,g.last_level_cache_bytes,
                g.tflops_fp16,g.tflops_int8,g.memory_bandwidth_gbps}].push_back(&observer);
        }
        std::ostringstream evidence;
        for(const auto &[key,members]:classes) {
            if(members.size()<2)continue;
            std::vector<DeviceId> devices;devices.reserve(members.size());
            for(const auto *member:members)devices.push_back(member->device());
            evidence << "; original-rank=" << members.front()->discoveryRank()
                << "/node=" << members.front()->physicalNode() << "/NUMA=" << members.front()->numaNode() << " UUIDs=[";
            bool first=true;
            for(const auto *member:members) {
                compute_cohorts_.at({member->discoveryRank(),member->device()})=devices;
                if(!first)evidence << ',';
                first=false;evidence << member->uuid();
            }
            evidence << ']';
        }
        if(!evidence.str().empty())compute_cohort_evidence_=
            "same-class GPU median complete compute predictions over original per-endpoint receipts; "
            "same source family/precision/local geometry; physical admission and links unpooled; CPU exact"+evidence.str();
    }

    double PlanningRequestCostModel::medianSeconds(int rank,DeviceId device,
        const std::function<double(int,DeviceId)> &estimate) const
    {
        if(!device.is_gpu())return estimate(rank,device);
        // Catalog aliases authenticate physical identity. Resolve the original
        // measured owner before choosing its class or interpreting any ordinal.
        const auto &observer=weights_.streamingEvidence().serviceFor(rank,device).observer;
        const int observed_rank=observer.discoveryRank();
        const auto found=compute_cohorts_.find({observed_rank,observer.device()});
        if(found==compute_cohorts_.end())throw std::invalid_argument("No authenticated GPU compute cohort");
        if(found->second.size()==1)return estimate(observed_rank,observer.device());
        std::vector<double> seconds;seconds.reserve(found->second.size());
        for(const auto member:found->second) {
            const auto value=estimate(observed_rank,member);
            if(!std::isfinite(value) || value<0)throw std::invalid_argument("Invalid same-class compute prediction");
            seconds.push_back(value);
        }
        std::sort(seconds.begin(),seconds.end());
        const auto middle=seconds.size()/2;
        return seconds.size()%2 ? seconds[middle] : seconds[middle-1]+(seconds[middle]-seconds[middle-1])/2;
    }

    double PlanningRequestCostModel::parameterBytes(const PlanningWeightOperand &weight) const
    {
        const auto found = source_bytes_per_element_.find(weight.source_name);
        if (found == source_bytes_per_element_.end())
            throw std::invalid_argument("Request operand does not belong to its measured model");
        return weight.geometry.elements() * (weight.representation == ModelPreparedWeightRepresentation::FP32
            ? sizeof(float) : weight.representation == ModelPreparedWeightRepresentation::FP16
                ? sizeof(uint16_t) : found->second);
    }

    double PlanningRequestCostModel::scalarSeconds(int rank, DeviceId device, int rows,
        double operations, double bytes) const
    {
        return medianSeconds(rank,device,[&](int observed_rank,DeviceId observed) {
            const auto &observation=std::get<PlanningFP32ArithmeticObservations>(arithmetic_.serviceFor(observed_rank,observed).observation);
            const auto &decode=observation.phases.at(0);const auto &prefill=observation.phases.at(1);
            return std::max(planningArithmeticSeconds(decode.service,prefill.rows,prefill.service,rows,operations),
                weights_.memorySeconds(observed_rank,observed,bytes));
        });
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
        if(!compute_cohort_evidence_.empty())result.qualifications.insert(compute_cohort_evidence_);
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
                    const int count = planningWeightInvocationRows(weight.role, rows);
                    double seconds = 0;
                    if constexpr (std::is_same_v<Operation, PlanningUnclassifiedWeight>)
                        throw std::invalid_argument("Unclassified request work: " + weight.source_name);
                    else if constexpr (std::is_same_v<Operation, PlanningProjectionWeight> ||
                                       std::is_same_v<Operation, PlanningRouterWeight>)
                    {
                        seconds = medianSeconds(participant.discovery_rank,participant.device,[&](int observed_rank,DeviceId observed) {
                            return weights_.projectionSeconds(observed_rank,observed,weight,count);
                        });
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
                        const bool extension_norm = weight.role == WeightRole::HyperConnectionNorm ||
                            weight.role == WeightRole::SparseIndexerNorm || weight.role == WeightRole::PerLayerEmbeddingNorm;
                        const bool norm = extension_norm || weight.role == WeightRole::Norm || weight.role == WeightRole::OutputNorm;
                        // QSA gains are reused per head; their source extent is
                        // not the transformer's hidden width. HC/PLE gains
                        // already span their full branch or lexical vector.
                        double coordinates = extension_norm ? elements : norm ? std::max(elements, double(profile.d_model)) : elements;
                        if (weight.role == WeightRole::SparseIndexerNorm && profile.flash_next)
                            coordinates *= weight.source_name.ends_with(".indexer.q_norm.weight")
                                ? profile.flash_next->indexer_query_heads : profile.flash_next->indexer_key_heads;
                        const double operations = weight.role == WeightRole::GDNSsmParam ? 0 :
                            count * coordinates * (norm ? 8.0 :
                                weight.role == WeightRole::SharedExpertInputGate || weight.role == WeightRole::PerLayerEmbeddingConvolution ? 2.0 : 1.0);
                        seconds = scalarSeconds(participant.discovery_rank, participant.device, count, operations,
                            parameterBytes(weight) + 2.0 * sizeof(float) * count * coordinates);
                    }
                    if (weight.role == WeightRole::Embedding) cost.entry += seconds;
                    else if (planningTerminalWeight(weight.role, weight.layer)) cost.terminal += seconds;
                    else
                    {
                        if (weight.layer < 0 || weight.layer >= layers)
                            throw std::invalid_argument("Ordinary request operation has no main-layer owner");
                        cost.ordinary[weight.layer] += seconds;
                    }
                }, operation);

            for (const auto &state : compilePlanningForwardStateWork(model_, devices[index], phase, {rows, context}))
            {
                cost.ordinary[state.layer] += medianSeconds(participant.discovery_rank,participant.device,[&](int observed_rank,DeviceId observed) {
                    const auto &fp32=std::get<PlanningFP32ArithmeticObservations>(
                        arithmetic_.serviceFor(observed_rank,observed).observation);
                    const auto &memory=std::get<PlanningMemoryBandwidthObservation>(
                        weights_.streamingEvidence().serviceFor(observed_rank,observed).observation);
                    return planningStateServiceSeconds(state,fp32,memory);
                });
                // Two residual additions per transformer block. Parameterized
                // norms are already charged by their own semantic operands.
                cost.ordinary[state.layer] += scalarSeconds(participant.discovery_rank, participant.device, rows,
                    2.0 * rows * profile.d_model, 6.0 * sizeof(float) * rows * profile.d_model);
            }
            for (const auto &expert : participant.routed)
            {
                cost.experts.at(expert.layer) = medianSeconds(participant.discovery_rank,participant.device,[&](int observed_rank,DeviceId observed) {
                    return weights_.expertSeconds(observed_rank,observed,expert,rows);
                });
                cost.routed.at(expert.layer) = &expert;
            }
        }

        // A link estimate never changes physical host membership or the actual
        // selected transport. Same-node mapped epochs and remote host MPI stay
        // distinct. Native links use the original directed pair matrix, not
        // another subgroup's allreduce timing or a reverse-direction estimate.
        const auto exchange = [&](size_t source, size_t target, size_t outgoing, size_t incoming, bool mapped) {
            const auto &a = work[source];
            const auto &b = work[target];
            if (a.discovery_rank == b.discovery_rank && a.device == b.device) return 0.0;
            const auto topology = inventory_.connectionBetweenRanks(a.discovery_rank, b.discovery_rank);
            const bool same_node = topology.locality() != RankConnectionLocality::CrossNode;
            if (!mapped && a.discovery_rank == b.discovery_rank && a.device.is_gpu() && b.device.type == a.device.type)
            {
                result.qualifications.insert("same-process GPU point-to-point placement uses physical peer copies or measured host-leg composition; request and reply remain dependent; native collective protocol timing is separate");
                const auto request = communication_.physicalPeerTransfer(a.discovery_rank, a.device, b.device, outgoing);
                const auto reply = communication_.physicalPeerTransfer(a.discovery_rank, b.device, a.device, incoming);
                result.qualifications.insert(request.evidence);
                result.qualifications.insert(reply.evidence);
                return request.seconds + reply.seconds;
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
                result.qualifications.insert("TP reductions retain measured native allreduce fixed latency and topology-scaled live FP32 payload service, with shared PCIe cut bounds");
                const auto prediction = communication_.calibratedNativeAllreduce(root.discovery_rank, endpoints,
                    profile.d_model, rows, PlanningAllreducePrecision::FP32);
                result.qualifications.insert(prediction.evidence);
                return prediction.seconds;
            }
            double seconds = 0;
            const size_t bytes = payload(double(rows) * profile.d_model * sizeof(float));
            for (size_t index = 1; index < group.size(); ++index)
                seconds += exchange(group.front(), group[index], bytes, bytes, false);
            return seconds;
        };

        /**
         * @brief Price the declared projection exchange, not a complete-FFN output sum.
         *
         * The production counted gather broadcasts each owner's extent and
         * live records through its original native communicator. Original
         * physical peer/host service supplies a serial-broadcast proxy; shared
         * PCIe cuts account for aggregate unique fragments. Column publication
         * is a lossless gather, not another floating reduction. No inferred
         * native forwarding schedule is presented as a measured algorithm.
         */
        const auto projectionCommunication = [&](const std::vector<size_t> &group, int layer) {
            if (group.size() < 2) throw std::logic_error("Projection cost has no complete GPU domain");
            const auto &root = work[group.front()];
            const auto *first = local[group.front()].routed[layer];
            if (!first || !first->projection_ownership)
                throw std::logic_error("Projection cost has no admitted source ownership");
            const auto topology = devices[group.front()].projection_exchange_topology;
            if (!topology) throw std::logic_error("Projection cost is missing original counted topology");
            std::vector<DeviceId> endpoints;
            std::vector<double> records;
            const auto geometry = first->projection_ownership->geometry();
            if (rows > std::numeric_limits<int>::max() / first->routes_per_token)
                throw std::overflow_error("Projection live route geometry exceeds its packet ABI");
            const bool floating = !native_quantised_formats::forQuantType(first->gate_up_down[2].executionFormat());
            const MoEGroupedIntermediateLayout layout{
                floating ? MoEGroupedIntermediateEncoding::FP32 : MoEGroupedIntermediateEncoding::BlockQ8FP32Scales,
                static_cast<uint32_t>(geometry.intermediate_columns),
                static_cast<uint32_t>(rows * first->routes_per_token), static_cast<uint32_t>(group.size())};
            if (!layout.compactValid()) throw std::logic_error("Projection cost has invalid packet geometry");
            for (size_t index : group)
            {
                const auto *expert = local[index].routed[layer];
                if (!expert || !expert->projection_ownership || work[index].discovery_rank != root.discovery_rank ||
                    work[index].device.type != root.device.type || (!devices[index].projection_exchange_topology ||
                     !devices[index].projection_exchange_topology->sameRoutes(*topology)) ||
                    expert->projection_ownership->geometry() != geometry ||
                    expert->projection_ownership->participants() != static_cast<int>(group.size()))
                    throw std::logic_error("Projection cost cannot substitute foreign domain/ownership evidence");
                endpoints.push_back(work[index].device);
                records.push_back(expert->uniformExpectation(rows).routed_rows);
            }
            const auto nativeVolume = [&](std::span<const size_t> owner_bytes) {
                const auto prediction = communication_.calibratedNativeCountedAllgather(root.discovery_rank, endpoints, owner_bytes);
                result.qualifications.insert(prediction.evidence);
                return prediction.seconds;
            };
            std::vector<size_t> intermediate_bytes;
            for (double count : records)
                intermediate_bytes.push_back(count > 0 ? payload(count * layout.compactRecordWords() * sizeof(uint32_t)) : 0);
            // The measured counted protocol already contains every root's
            // extent, payload and completion. Charging an additional metadata
            // gather would count that control twice, including empty owners.
            result.qualifications.insert("projection intermediate uses measured native counted gather; live extents and completion charged once");
            const double intermediate = nativeVolume(intermediate_bytes);
            result.qualifications.insert("projection per-role work extrapolates complete-FFN measured rate; gate/up quota and all-expert down slices remain distinct");
            const size_t column_bytes = payload(double(rows) * geometry.model_columns * sizeof(float) / group.size());
            const auto columns = communication_.calibratedNativeAllgather(root.discovery_rank, endpoints, column_bytes);
            result.qualifications.insert(columns.evidence);
            return intermediate + columns.seconds;
        };

        std::vector<double> entries, terminals;
        for (const auto &cost : local) { entries.push_back(cost.entry); terminals.push_back(cost.terminal); }
        result.compute = planningIndependentSeconds(entries) + planningIndependentSeconds(terminals);
        std::vector<size_t> previous;
        for (int layer = 0; layer < layers; ++layer)
        {
            std::optional<size_t> overlay_root;
            if (const auto overlay = candidate.expertLayer(layer))
            {
                const auto &placement = overlay->placement;
                const auto bindings = MoEOverlayCapacityAdmission::boundParticipants(placement);
                const auto root = std::find_if(bindings.begin(), bindings.end(), [&](const auto &binding) {
                    return binding.participant_id == placement.continuation_domain_spec.logical_root_participant;
                });
                if (root == bindings.end()) throw std::logic_error("Request overlay has no bound logical root");
                for (size_t index = 0; index < work.size(); ++index)
                    if (work[index].execution_rank == root->world_rank && work[index].device == root->device &&
                        work[index].first_layer <= layer && work[index].last_layer >= layer)
                        overlay_root = index;
                if (!overlay_root) throw std::logic_error("Request overlay root is not a compiled participant");
            }
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
            {
                // PP carries the residual trunk between complete decoder stages.
                // HC expands that trunk without expanding compute-H allreduces.
                const auto width = profile.flash_next ? profile.flash_next->residual_width : profile.d_model;
                if (!width) throw std::invalid_argument("Pipeline boundary has no residual-trunk width");
                result.communication += exchange(previous.front(), continuation.front(),
                    payload(double(rows) * width * sizeof(float)), 1, false);
            }
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
            // The shared FFN retains a native sum/reduce-scatter volume proxy
            // above. Projection mode additionally exchanges intermediates and
            // assembles independent down columns; whole-expert mode does not.
            const bool projection = std::any_of(continuation.begin(), continuation.end(), [&](size_t index) {
                return local[index].routed[layer] && local[index].routed[layer]->projection_ownership.has_value();
            });
            if (projection) result.communication += projectionCommunication(continuation, layer);
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
        // Ordinary operands deliberately exclude the immutable table's
        // capacity. This mandatory typed pass accounts for each live native
        // read, decoded output and admitted source/consumer channel before a
        // request estimate can escape. Missing services remain fatal.
        const auto lookup = pricePlanningImmutableLookupWork(model_, candidate, static_cast<size_t>(rows), weights_, communication_);
        result.compute += lookup.compute_seconds;
        result.communication += lookup.communication_seconds;
        result.qualifications.insert(lookup.qualifications.begin(), lookup.qualifications.end());
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
