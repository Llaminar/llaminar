/**
 * @file PlanningKernelServiceCatalog.cpp
 * @brief One all-rank source/admission/service transaction before candidate pricing.
 *
 * Physical endpoint resolution is pure inventory projection. Native source bytes
 * are broadcast once, then each node measures one reporting rank at a time so
 * overlapping CPU visibility cannot manufacture contention. Samplers on separate
 * physical nodes may run concurrently. Setup storage is admitted once through
 * the canonical PMA envelope and completely retires before returning evidence.
 */
#include "PlanningKernelServiceCatalog.h"
#include "PlanningObservedResource.h"
#include "PlanningExecutionMeasurement.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "utils/CPUFeatures.h"
#include "utils/MPIContext.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <omp.h>

namespace llaminar2
{
    PlanningProjectionServicePlan::PlanningProjectionServicePlan(PlanningMatrixSamplePlan matrix, int prefill_rows)
        : matrix_(std::move(matrix)), prefill_rows_(prefill_rows)
    {
        if (prefill_rows <= 0)
            throw std::invalid_argument("Planning projection service requires positive prefill rows");
    }

    namespace
    {
        using Json = nlohmann::json;
        using ExpertObservation = std::variant<PlanningCPUExpertObservations, PlanningGPUExpertObservations>;
        constexpr const char *kSchema = "llaminar.planning-kernel-service.v4";

        /** @brief Reject incomplete evidence rather than treating a failed endpoint as slow. */
        void require(bool condition, const char *message)
        {
            if (!condition) throw std::invalid_argument(std::string("Planning kernel service: ") + message);
        }

        /** @return A bounded nonnegative integer without JSON's implicit truncation/wrapping. */
        size_t integer(const Json &value, size_t maximum = std::numeric_limits<size_t>::max())
        {
            require(value.is_number_integer(), "expected integer field");
            require(value.is_number_unsigned() || value.get<int64_t>() >= 0, "negative integer field");
            const auto result = value.get<uint64_t>();
            require(result <= maximum, "integer field out of range");
            return static_cast<size_t>(result);
        }

        /** @return Source selection and operation geometry, not merely matching byte lengths. */
        Json sourceIdentity(const PlanningKernelSamplePlan &plan)
        {
            return std::visit([](const auto &source) -> Json {
                if constexpr (std::is_same_v<std::decay_t<decltype(source)>, PlanningExpertSamplePlan>)
                    return {{"kind", "expert"}, {"source", source.serialize()}};
                else if constexpr (std::is_same_v<std::decay_t<decltype(source)>, PlanningProjectionServicePlan>)
                    return {{"kind", "projection"}, {"source", source.matrix().serialize()},
                    {"execution_format", source.matrix().executionFormat()},
                    {"prefill_rows", source.prefillRows()}};
                else if constexpr (std::is_same_v<std::decay_t<decltype(source)>, PlanningFP32ArithmeticPlan>)
                    return {{"kind", "fp32-arithmetic-proxy"}, {"N", source.kN}, {"K", source.kK},
                        {"prefill_rows", source.kPrefillRows}};
                else return {{"kind", "streaming-memory"}};
            }, plan);
        }

        /** @return Complete topology-derived assignment; every rank checks this before allocating. */
        std::vector<uint8_t> assignment(const ClusterInventory &inventory,
            const AutomaticOrchestrationRequest &request, const PlanningKernelSamplePlan &plan)
        {
            Json wire{{"source", sourceIdentity(plan)}, {"observers", Json::array()}, {"membership", Json::array()}};
            for (int rank = 0; rank < inventory.world_size; ++rank)
                wire["membership"].push_back({rank, inventory.connectionBetweenRanks(rank, rank).sourceNode()});
            for (const auto &observer : PlanningKernelServiceCatalog::observers(inventory, request))
                wire["observers"].push_back({{"rank", observer.discoveryRank()}, {"node", observer.physicalNode()},
                    {"backend", int(observer.device().type)}, {"ordinal", observer.device().ordinal},
                    {"numa", observer.numaNode()}, {"uuid", observer.uuid()}});
            const auto text = wire.dump();
            return {text.begin(), text.end()};
        }

        /** @return Exact sample identity; names alone do not identify an expert slice. */
        bool sameSource(const PlanningExpertSampleRequest &left, const PlanningExpertSampleRequest &right)
        {
            for (size_t i = 0; i < 3; ++i)
            {
                const auto *a = left.projections()[i], *b = right.projections()[i];
                const auto *ae = std::get_if<PlanningExpertMatrix>(&a->selection);
                const auto *be = std::get_if<PlanningExpertMatrix>(&b->selection);
                if (a->tensor_name != b->tensor_name || !ae || !be || ae->index != be->index) return false;
            }
            return true;
        }

        /** @brief Validate the existing sampler's complete phase contract without extrapolation. */
        void validateExpert(const ExpertObservation &value, const PlanningExpertSamplePlan &plan)
        {
            std::visit([&](const auto &observation) {
                using Observation = std::decay_t<decltype(observation)>;
                constexpr bool cpu = std::is_same_v<Observation, PlanningCPUExpertObservations>;
                const auto &description = plan.description();
                require(cpu ? observation.device == DeviceId::cpu() : observation.device.is_gpu(), "wrong execution regime");
                require(observation.input_width == description.matrices[0].k &&
                    observation.intermediate_width == description.matrices[0].n &&
                    observation.formats == description.formats && sameSource(observation.source, plan.request()),
                    "observation does not match source geometry/format/selection");
                require(observation.prepared_bytes > 0 && observation.phases.size() == 3, "incomplete prepared service");
                if constexpr (cpu)
                    require(observation.worker_threads > 0 &&
                        observation.isa >= ISALevel::Scalar && observation.isa <= ISALevel::AVX512 &&
                        observation.compiled_isa >= ISALevel::AVX2 && observation.compiled_isa <= ISALevel::AVX512 &&
                        observation.isa <= observation.compiled_isa, "invalid CPU worker/ISA identity");
                for (size_t i = 0; i < observation.phases.size(); ++i)
                {
                    const auto &phase = observation.phases[i];
                    require(expertHistogramProductionSourceIndex(phase.phase) == i &&
                        phase.rows == MoEOverlayCPUServiceMeasurement::rowsForPhase(phase.phase), "wrong phase/row geometry");
                    if constexpr (!cpu) require(phase.graph_nodes > 0, "GPU observation did not execute a captured graph");
                    const int invocations = cpu ? MoEOverlayCPUServiceMeasurement::kMeasuredSamples :
                        PlanningExecutionMeasurement::kTimedInvocations;
                    require(phase.service.unit() == PlanningWorkUnit::ArithmeticOperations &&
                        phase.service.completedWork() == 6.0 * phase.rows * observation.input_width *
                            observation.intermediate_width * invocations, "wrong completed work or units");
                }
            }, value);
        }

        /** @brief Byte service may only name completed streaming passes, never cached arithmetic. */
        void validateStreaming(const PlanningMemoryBandwidthObservation &value)
        {
            require(value.request.device().is_gpu() ? value.graph_nodes > 0 : value.graph_nodes == 0,
                "streaming graph evidence differs from execution backend");
            require(value.service.unit() == PlanningWorkUnit::Bytes &&
                value.service.completedWork() == double(value.request.usefulBytes()) * PlanningExecutionMeasurement::kTimedInvocations,
                "wrong completed streaming bytes or work units");
        }

        /** @return The sole device identity stored by the operation-specific observation. */
        DeviceId observedDevice(const PlanningKernelObservation &observation)
        {
            return std::visit([](const auto &value) {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, PlanningMemoryBandwidthObservation>)
                    return value.request.device();
                else return value.device;
            }, observation);
        }

        /** @return Wire observations with source verified before serialization. */
        Json encodeExpert(const ExpertObservation &value, const PlanningExpertSamplePlan &plan)
        {
            validateExpert(value, plan);
            return std::visit([](const auto &observation) {
                Json result{{"backend", int(observation.device.type)}, {"ordinal", observation.device.ordinal},
                    {"prepared_bytes", observation.prepared_bytes}, {"phases", Json::array()}};
                if constexpr (std::is_same_v<std::decay_t<decltype(observation)>, PlanningCPUExpertObservations>)
                    result["cpu"] = {{"workers", observation.worker_threads}, {"isa", int(observation.isa)},
                        {"compiled_isa", int(observation.compiled_isa)}};
                for (const auto &phase : observation.phases)
                {
                    Json entry{{"phase", int(phase.phase)}, {"rows", phase.rows},
                        {"work", phase.service.completedWork()}, {"seconds", phase.service.elapsedSeconds()},
                        {"provenance", phase.service.provenance()}};
                    if constexpr (requires { phase.graph_nodes; }) entry["graph_nodes"] = phase.graph_nodes;
                    result["phases"].push_back(std::move(entry));
                }
                return result;
            }, value);
        }

        /** @return Fully checked observation; topology/rank are deliberately absent from this codec. */
        ExpertObservation decodeExpert(const Json &wire, const PlanningExpertSamplePlan &plan)
        {
            const auto backend = static_cast<DeviceType>(integer(wire.at("backend"), int(DeviceType::Metal)));
            require(backend == DeviceType::CPU || backend == DeviceType::CUDA || backend == DeviceType::ROCm,
                "unsupported execution backend");
            const bool cpu = backend == DeviceType::CPU;
            require(wire.is_object() && wire.size() == (cpu ? 5 : 4), "unknown observation fields");
            const DeviceId device{backend, static_cast<int>(integer(wire.at("ordinal"), std::numeric_limits<int>::max()))};
            const size_t prepared = integer(wire.at("prepared_bytes"));
            const auto &description = plan.description();
            ExpertObservation result = PlanningGPUExpertObservations{device,
                description.matrices[0].k, description.matrices[0].n, prepared, description.formats, plan.request(), {}};
            if (cpu)
            {
                const auto &scope = wire.at("cpu");
                require(scope.is_object() && scope.size() == 3, "invalid CPU execution scope");
                result = PlanningCPUExpertObservations{device,
                    static_cast<int>(integer(scope.at("workers"), std::numeric_limits<int>::max())),
                    static_cast<ISALevel>(integer(scope.at("isa"), int(ISALevel::AVX512))),
                    static_cast<ISALevel>(integer(scope.at("compiled_isa"), int(ISALevel::AVX512))),
                    description.matrices[0].k, description.matrices[0].n, prepared, description.formats, plan.request(), {}};
            }
            require(wire.at("phases").is_array() && wire.at("phases").size() == 3, "missing service phases");
            for (const auto &phase : wire.at("phases"))
            {
                require(phase.is_object() && phase.size() == (cpu ? 5 : 6), "unknown phase fields");
                const auto kind = static_cast<ExpertHistogramSource>(integer(phase.at("phase"), 2));
                const int rows = static_cast<int>(integer(phase.at("rows"), std::numeric_limits<int>::max()));
                PlanningServiceObservation service(PlanningWorkUnit::ArithmeticOperations,
                    phase.at("work").get<double>(), phase.at("seconds").get<double>(), phase.at("provenance").get<std::string>());
                if (cpu) std::get<PlanningCPUExpertObservations>(result).phases.push_back({kind, rows, std::move(service)});
                else std::get<PlanningGPUExpertObservations>(result).phases.push_back(
                    {kind, rows, integer(phase.at("graph_nodes")), std::move(service)});
            }
            validateExpert(result, plan);
            return result;
        }

        /** @return Exact CPU execution identity without re-running discovery on the receiving rank. */
        Json cpuExecutionIdentity(const CPUExecutionGeometry &execution)
        {
            return Json::array({execution.cache.private_l2_bytes, execution.cache.shared_l3_bytes,
                execution.cache.private_l2_ways, execution.cache.shared_l3_ways, execution.maximum_native_row_tile});
        }

        /** @brief Authenticate the shared projection ABI without assigning a source identity. */
        template<class Observation>
        void validateProjectionExecution(const Observation &value, size_t n, size_t k, int prefill_rows, TensorType type)
        {
            const bool cpu = value.device == DeviceId::cpu();
            require(cpu || value.device.is_gpu(), "unsupported projection execution device");
            require(bool(value.cpu) == cpu && value.phases.size() == 2, "incomplete projection execution scope");
            if (cpu)
            {
                require(value.cpu->workers > 0 && value.cpu->isa >= ISALevel::Scalar &&
                    value.cpu->isa <= ISALevel::AVX512 && value.cpu->compiled_isa >= ISALevel::AVX2 &&
                    value.cpu->compiled_isa <= ISALevel::AVX512 && value.cpu->isa <= value.cpu->compiled_isa,
                    "invalid projection CPU workshare/ISA");
                const bool floating = type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16;
                require(floating ? value.prepared_bytes == 0 : value.prepared_bytes > 0,
                    "CPU projection prepared ownership differs from its execution format");
            }
            else require(value.prepared_bytes > 0, "GPU projection has no prepared payload");
            for (size_t i = 0; i < value.phases.size(); ++i)
            {
                const auto &phase = value.phases[i];
                require(phase.rows == (i == 0 ? 1 : prefill_rows) &&
                    (cpu ? phase.graph_nodes == 0 : phase.graph_nodes > 0), "wrong projection M or graph execution");
                require(phase.service.unit() == PlanningWorkUnit::ArithmeticOperations &&
                    phase.service.completedWork() == 2.0 * phase.rows * n * k *
                        PlanningExecutionMeasurement::kTimedInvocations, "wrong projection work or units");
            }
        }

        /** @brief Preserve the exact GGUF source and representation in addition to the execution ABI. */
        void validateProjection(const PlanningProjectionObservations &value, const PlanningProjectionServicePlan &plan)
        {
            require(value.source.serialize() == plan.matrix().serialize(), "foreign projection source selection");
            validateProjectionExecution(value, plan.matrix().geometry().n, plan.matrix().geometry().k,
                plan.prefillRows(), plan.matrix().executionType());
        }

        /** @return One operation-qualified receipt, with topology deliberately excluded. */
        Json encodeObservation(const PlanningKernelObservation &value, const PlanningKernelSamplePlan &plan)
        {
            return std::visit([&](const auto &observation) -> Json {
                if constexpr (std::is_same_v<std::decay_t<decltype(observation)>, PlanningMemoryBandwidthObservation>)
                {
                    require(std::holds_alternative<PlanningStreamingServicePlan>(plan),
                        "streaming byte observation cannot price a source-kernel request");
                    validateStreaming(observation);
                    const auto &request = observation.request;
                    return {{"backend", int(request.device().type)}, {"ordinal", request.device().ordinal},
                        {"cache_bytes", request.cacheBytes()}, {"stream_bytes", request.streamBytes()},
                        {"workers", request.workers()}, {"execution", cpuExecutionIdentity(request.cpuGeometry())},
                        {"graph_nodes", observation.graph_nodes}, {"work", observation.service.completedWork()},
                        {"seconds", observation.service.elapsedSeconds()}, {"provenance", observation.service.provenance()}};
                }
                else if constexpr (std::is_same_v<std::decay_t<decltype(observation)>, PlanningProjectionObservations> ||
                    std::is_same_v<std::decay_t<decltype(observation)>, PlanningFP32ArithmeticObservations>)
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(observation)>, PlanningProjectionObservations>)
                    {
                        const auto *projection = std::get_if<PlanningProjectionServicePlan>(&plan);
                        require(projection != nullptr, "source projection cannot substitute for another operation");
                        validateProjection(observation, *projection);
                    }
                    else
                    {
                        require(std::holds_alternative<PlanningFP32ArithmeticPlan>(plan),
                            "FP32 arithmetic proxy cannot substitute for model or byte service");
                        validateProjectionExecution(observation, PlanningFP32ArithmeticPlan::kN, PlanningFP32ArithmeticPlan::kK,
                            PlanningFP32ArithmeticPlan::kPrefillRows, TensorType::FP32);
                    }
                    Json wire{{"backend", int(observation.device.type)}, {"ordinal", observation.device.ordinal},
                        {"prepared_bytes", observation.prepared_bytes}, {"phases", Json::array()}};
                    if (observation.cpu)
                        wire["cpu"] = {{"workers", observation.cpu->workers}, {"isa", int(observation.cpu->isa)},
                            {"compiled_isa", int(observation.cpu->compiled_isa)},
                            {"execution", cpuExecutionIdentity(observation.cpu->execution)}};
                    for (const auto &phase : observation.phases)
                        wire["phases"].push_back({{"rows", phase.rows}, {"graph_nodes", phase.graph_nodes},
                            {"work", phase.service.completedWork()}, {"seconds", phase.service.elapsedSeconds()},
                            {"provenance", phase.service.provenance()}});
                    return wire;
                }
                else
                {
                    const auto *expert = std::get_if<PlanningExpertSamplePlan>(&plan);
                    require(expert != nullptr, "expert observation cannot price an ordinary projection request");
                    return encodeExpert(ExpertObservation(observation), *expert);
                }
            }, value);
        }

        /** @return Strictly decoded source-matched evidence, never a partially populated result. */
        PlanningKernelObservation decodeObservation(const Json &wire, const PlanningKernelSamplePlan &plan,
            const RankInventory &rank)
        {
            if (std::holds_alternative<PlanningStreamingServicePlan>(plan))
            {
                require(wire.is_object() && wire.size() == 10, "unknown streaming fields");
                const auto backend = static_cast<DeviceType>(integer(wire.at("backend"), int(DeviceType::Metal)));
                const DeviceId device{backend, static_cast<int>(integer(wire.at("ordinal"), INT_MAX))};
                // Transport supplies rank identity; geometry comes from that
                // rank's immutable inventory, not from a self-asserted receipt.
                auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, device);
                require(integer(wire.at("cache_bytes")) == request.cacheBytes() &&
                    integer(wire.at("stream_bytes")) == request.streamBytes() &&
                    integer(wire.at("workers"), INT_MAX) == size_t(request.workers()) &&
                    wire.at("execution") == cpuExecutionIdentity(request.cpuGeometry()),
                    "streaming scope differs from the reporting rank's inventory");
                PlanningMemoryBandwidthObservation result{std::move(request), integer(wire.at("graph_nodes")),
                    PlanningServiceObservation(PlanningWorkUnit::Bytes, wire.at("work").get<double>(),
                        wire.at("seconds").get<double>(), wire.at("provenance").get<std::string>())};
                validateStreaming(result);
                return result;
            }
            if (const auto *expert = std::get_if<PlanningExpertSamplePlan>(&plan))
                return std::visit([](auto observation) -> PlanningKernelObservation { return observation; },
                    decodeExpert(wire, *expert));
            const auto backend = static_cast<DeviceType>(integer(wire.at("backend"), int(DeviceType::Metal)));
            require(backend == DeviceType::CPU || backend == DeviceType::CUDA || backend == DeviceType::ROCm,
                "unsupported projection backend");
            const DeviceId device{backend, static_cast<int>(integer(wire.at("ordinal"), INT_MAX))};
            const bool cpu = backend == DeviceType::CPU;
            require(wire.is_object() && wire.size() == (cpu ? 5 : 4), "unknown projection fields");
            // Source-backed and source-free observations share one execution
            // codec, but are sealed into different types after authentication.
            const auto decode_projection = [&]<class Observation>(Observation result) -> PlanningKernelObservation {
                if (cpu)
                {
                    const auto &scope = wire.at("cpu"), &geometry = scope.at("execution");
                    require(scope.is_object() && scope.size() == 4 && geometry.is_array() && geometry.size() == 5,
                        "incomplete projection CPU scope");
                    CPUExecutionGeometry execution{{integer(geometry[0]), integer(geometry[1]),
                        static_cast<uint32_t>(integer(geometry[2], UINT32_MAX)),
                        static_cast<uint32_t>(integer(geometry[3], UINT32_MAX))},
                        static_cast<uint32_t>(integer(geometry[4], UINT32_MAX))};
                    result.cpu = PlanningProjectionCPUObservation{static_cast<int>(integer(scope.at("workers"), INT_MAX)),
                        static_cast<ISALevel>(integer(scope.at("isa"), int(ISALevel::AVX512))),
                        static_cast<ISALevel>(integer(scope.at("compiled_isa"), int(ISALevel::AVX512))), execution};
                }
                require(wire.at("phases").is_array() && wire.at("phases").size() == 2, "missing projection phases");
                for (const auto &phase : wire.at("phases"))
                {
                    require(phase.is_object() && phase.size() == 5, "unknown projection phase fields");
                    result.phases.push_back({static_cast<int>(integer(phase.at("rows"), INT_MAX)),
                        integer(phase.at("graph_nodes")), PlanningServiceObservation(PlanningWorkUnit::ArithmeticOperations,
                            phase.at("work").get<double>(), phase.at("seconds").get<double>(),
                            phase.at("provenance").get<std::string>())});
                }
                if constexpr (std::is_same_v<Observation, PlanningProjectionObservations>)
                    validateProjection(result, std::get<PlanningProjectionServicePlan>(plan));
                else
                    validateProjectionExecution(result, PlanningFP32ArithmeticPlan::kN, PlanningFP32ArithmeticPlan::kK,
                        PlanningFP32ArithmeticPlan::kPrefillRows, TensorType::FP32);
                return result;
            };
            if (const auto *projection = std::get_if<PlanningProjectionServicePlan>(&plan))
                return decode_projection(PlanningProjectionObservations{
                    projection->matrix(), device, integer(wire.at("prepared_bytes")), {}, {}});
            require(std::holds_alternative<PlanningFP32ArithmeticPlan>(plan), "unknown arithmetic operation");
            return decode_projection(PlanningFP32ArithmeticObservations{device, integer(wire.at("prepared_bytes")), {}, {}});
        }

        /**
         * @brief Admit this rank's source and mutually exclusive measurement demands through PMA.
         *
         * Source-free probes may assign no observer to a discovery rank.  Such
         * a rank owns no planning byte and must not manufacture an empty PMA
         * certificate merely to participate in the control collective.  A null
         * result is therefore an explicit no-allocation state; every caller
         * that will load or measure still requires a non-null authority.
         *
         * @return Canonical rank-local admission authority, or null when this
         *         source-free probe has no rank-local allocation or observer.
         */
        std::shared_ptr<PhysicalMemoryAuthority> admit(const ClusterInventory &inventory, int rank,
            const PlanningKernelSamplePlan &plan, std::span<const PlanningServiceObserver> observers)
        {
            const auto &observed = inventory.ranks.at(rank);
            if (std::any_of(observers.begin(), observers.end(), [&](const auto &observer) {
                    return observer.discoveryRank() == rank && observer.device().is_cpu();
                }))
            {
                // Check before source I/O and before a kernel can consume a
                // workspace sized for a different team. Receipt validation is
                // a second boundary, not permission to execute with stale
                // discovery geometry and reject the resulting number later.
                require(!omp_in_parallel() && !omp_get_dynamic() &&
                    omp_get_max_threads() == observed.cpuWorkerThreads() &&
                    omp_get_thread_limit() >= observed.cpuWorkerThreads() &&
                    CPUExecutionGeometry::local() == observed.cpu_execution,
                    "local CPU execution geometry/workshare changed after discovery");
            }
            const auto host = planningObservedResource(observed, DeviceId::cpu());
            const auto origin = rank == 0 ? PlanningSampleOrigin::LocalGGUF : PlanningSampleOrigin::PublishedPayload;
            PhysicalMemoryPlanBuilder source;
            std::visit([&](const auto &sample) {
                if constexpr (std::is_same_v<std::decay_t<decltype(sample)>, PlanningExpertSamplePlan>)
                {
                    source.add(host, PhysicalMemoryOwner::ModelSourcePayload, sample.description().source_bytes);
                    source.add(host, PhysicalMemoryOwner::WeightLoadStaging,
                        rank == 0 ? sample.description().largest_source_bytes : 0);
                }
                else if constexpr (std::is_same_v<std::decay_t<decltype(sample)>, PlanningProjectionServicePlan>)
                {
                    source.add(host, PhysicalMemoryOwner::ModelSourcePayload, sample.matrix().geometry().source_bytes);
                    source.add(host, PhysicalMemoryOwner::WeightLoadStaging,
                        rank == 0 ? sample.matrix().geometry().source_bytes : 0);
                }
            }, plan);
            std::vector<PhysicalMemoryPlan> alternatives{source.build()};
            for (const auto &observer : observers)
            {
                if (observer.discoveryRank() != rank) continue;
                PhysicalMemoryPlanBuilder builder;
                const auto device = observer.device();
                const auto contribute = [&] {
                    std::visit([&](const auto &sample) {
                        if constexpr (std::is_same_v<std::decay_t<decltype(sample)>, PlanningExpertSamplePlan>)
                        {
                            if (device.is_cpu()) PlanningCPUExpertMeasurement::contributeMemory(sample, origin,
                                host, builder, observed.cpu_execution, observed.cpuWorkerThreads());
                            else PlanningGPUExpertMeasurement::contributeMemory(sample, origin,
                                host, planningObservedResource(observed, device), builder);
                        }
                        else if constexpr (std::is_same_v<std::decay_t<decltype(sample)>, PlanningProjectionServicePlan> ||
                            std::is_same_v<std::decay_t<decltype(sample)>, PlanningFP32ArithmeticPlan>)
                        {
                            std::optional<PlanningProjectionCPUObservation> cpu;
                            // Only geometry/workers contribute to scratch. ISA
                            // identity is read on the observing rank, not root.
                            if (device.is_cpu()) cpu = PlanningProjectionCPUObservation{
                                observed.cpuWorkerThreads(), activeISALevel(),
#if LLAMINAR_COMPILED_WITH_AVX512
                                ISALevel::AVX512,
#else
                                ISALevel::AVX2,
#endif
                                observed.cpu_execution};
                            if constexpr (std::is_same_v<std::decay_t<decltype(sample)>, PlanningProjectionServicePlan>)
                                PlanningProjectionMeasurement::contributeMemory(sample.matrix(), origin,
                                    host, planningObservedResource(observed, device), builder, sample.prefillRows(), cpu);
                            else PlanningProjectionMeasurement::contributeMemory(sample,
                                host, planningObservedResource(observed, device), builder, cpu);
                        }
                        else PlanningMemoryBandwidthMeasurement::contributeMemory(
                            PlanningMemoryBandwidthRequest::fromInventory(observed, device),
                            host, planningObservedResource(observed, device), builder);
                    }, plan);
                };
                if (device.is_cpu()) contribute();
                else GPUDeviceContextPool::instance().getContext(device).submitAndWait(contribute);
                alternatives.push_back(builder.build());
            }
            PhysicalMemoryPlanBuilder complete;
            complete.addMutuallyExclusive(alternatives);
            const auto admitted = complete.build();
            if (admitted.resources().empty()) return {};
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admitted), rank);
        }
    }

    std::vector<PlanningServiceObserver> PlanningKernelServiceCatalog::observers(
        const ClusterInventory &inventory, const AutomaticOrchestrationRequest &request)
    {
        require(inventory.world_size > 0 && inventory.ranks.size() == size_t(inventory.world_size), "invalid discovery membership");
        // Reuse the canonical physical aggregation validator. It checks alias
        // capacities and geometry; the copy is validation, not fresh discovery.
        auto validated = inventory;
        validated.buildNodeAggregations();
        std::vector<PlanningServiceObserver> result;
        using Key = std::tuple<int, DeviceType, std::string>;
        std::map<Key, PlanningServiceObserver> gpus;
        for (int rank = 0; rank < inventory.world_size; ++rank)
        {
            const auto &observed = inventory.ranks.at(rank);
            const int node = inventory.connectionBetweenRanks(rank, rank).sourceNode();
            if (request.allows(DeviceType::CPU) && observed.cpu.memory_bytes > 0)
                result.push_back(PlanningServiceObserver(rank, node, DeviceId::cpu(), observed.cpu.numa_node, {}));
            std::set<std::pair<DeviceType, int>> ordinals;
            for (const auto &gpu : observed.gpus)
            {
                if (!request.allows(gpu.type)) continue;
                require(gpu.type == DeviceType::CUDA || gpu.type == DeviceType::ROCm, "unsupported GPU backend");
                require(!gpu.uuid.empty() && gpu.local_device_id >= 0, "GPU has no physical identity");
                require(ordinals.emplace(gpu.type, gpu.local_device_id).second, "duplicate rank-local GPU ordinal");
                const Key key{node, gpu.type, gpu.uuid};
                PlanningServiceObserver candidate(rank, node, {gpu.type, gpu.local_device_id}, gpu.numa_node, gpu.uuid);
                const auto previous = gpus.find(key);
                require(previous == gpus.end() || previous->second.numaNode() < 0 || gpu.numa_node < 0 ||
                    previous->second.numaNode() == gpu.numa_node, "GPU aliases disagree on physical NUMA affinity");
                // Affinity selects the reporter only. It cannot change physical
                // membership, select an inference topology or fabricate a cost.
                const bool local = gpu.numa_node >= 0 && observed.cpu.numa_node == gpu.numa_node;
                const bool previous_local = previous != gpus.end() && gpu.numa_node >= 0 &&
                    inventory.ranks.at(previous->second.discoveryRank()).cpu.numa_node == gpu.numa_node;
                if (previous == gpus.end()) gpus.emplace(key, std::move(candidate));
                else if (local && !previous_local) previous->second = std::move(candidate);
            }
        }
        for (auto &[key, observer] : gpus) result.push_back(std::move(observer));
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return std::tuple{a.discoveryRank(), a.device().type, a.device().ordinal} <
                std::tuple{b.discoveryRank(), b.device().type, b.device().ordinal};
        });
        return result;
    }

    std::vector<uint8_t> PlanningKernelServiceCatalog::encode(const PlanningKernelSamplePlan &plan,
        std::span<const PlanningKernelObservation> observations)
    {
        Json wire{{"schema", kSchema}, {"source", sourceIdentity(plan)}, {"observations", Json::array()}};
        for (const auto &observation : observations) wire["observations"].push_back(encodeObservation(observation, plan));
        const auto text = wire.dump();
        return {text.begin(), text.end()};
    }

    PlanningKernelServiceCatalog PlanningKernelServiceCatalog::accept(const ClusterInventory &inventory,
        const AutomaticOrchestrationRequest &request, const PlanningKernelSamplePlan &plan,
        std::span<const RankPlanningSample> receipts)
    {
        const auto expected = observers(inventory, request);
        require(receipts.size() == size_t(inventory.world_size), "incomplete rank receipts");
        std::vector<PlanningKernelServiceRecord> records;
        std::set<int> seen_ranks;
        for (const auto &receipt : receipts)
        {
            inventory.connectionBetweenRanks(receipt.discovery_rank, receipt.discovery_rank);
            require(seen_ranks.insert(receipt.discovery_rank).second, "duplicate reporting rank");
            const auto wire = Json::parse(receipt.bytes.begin(), receipt.bytes.end());
            require(wire.is_object() && wire.size() == 3 && wire.at("schema") == kSchema &&
                wire.at("source") == sourceIdentity(plan) && wire.at("observations").is_array(), "foreign source or receipt schema");
            std::vector<PlanningServiceObserver> local;
            for (const auto &observer : expected)
                if (observer.discoveryRank() == receipt.discovery_rank) local.push_back(observer);
            require(wire.at("observations").size() == local.size(), "missing or extra endpoint observation");
            for (size_t i = 0; i < local.size(); ++i)
            {
                auto observation = decodeObservation(wire.at("observations").at(i), plan,
                    inventory.ranks.at(receipt.discovery_rank));
                require(observedDevice(observation) == local[i].device(),
                    "observation reported by the wrong device/rank");
                std::visit([&](const auto &projection) {
                    if constexpr (requires { projection.cpu; })
                        if (projection.cpu)
                            require(projection.cpu->execution == inventory.ranks.at(receipt.discovery_rank).cpu_execution &&
                                projection.cpu->workers == inventory.ranks.at(receipt.discovery_rank).cpuWorkerThreads(),
                                "projection execution geometry/workshare differs from the observing rank's inventory");
                }, observation);
                if (const auto *expert = std::get_if<PlanningCPUExpertObservations>(&observation))
                    require(expert->worker_threads == inventory.ranks.at(receipt.discovery_rank).cpuWorkerThreads(),
                        "expert workshare differs from the observing rank's inventory");
                records.push_back({local[i], std::move(observation)});
            }
        }
        std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
            return std::tuple{a.observer.discoveryRank(), a.observer.device().type, a.observer.device().ordinal} <
                std::tuple{b.observer.discoveryRank(), b.observer.device().type, b.observer.device().ordinal};
        });
        std::map<EndpointKey, size_t> endpoints;
        for (size_t index = 0; index < records.size(); ++index)
        {
            const auto &observer = records[index].observer;
            if (observer.device().is_cpu())
                endpoints.emplace(EndpointKey{observer.discoveryRank(), DeviceType::CPU, 0}, index);
            else
                for (const auto &rank : inventory.ranks)
                {
                    if (inventory.connectionBetweenRanks(rank.rank, rank.rank).sourceNode() != observer.physicalNode()) continue;
                    for (const auto &gpu : rank.gpus)
                        if (gpu.type == observer.device().type && gpu.uuid == observer.uuid())
                            require(endpoints.emplace(EndpointKey{rank.rank, gpu.type, gpu.local_device_id}, index).second,
                                "ambiguous physical GPU evidence alias");
                }
        }
        return PlanningKernelServiceCatalog(plan, std::move(records), std::move(endpoints));
    }

    const PlanningKernelServiceRecord &PlanningKernelServiceCatalog::serviceFor(int discovery_rank, DeviceId device) const
    {
        const auto found = endpoints_.find(EndpointKey{discovery_rank, device.type, device.ordinal});
        if (found == endpoints_.end())
            throw std::out_of_range("Planning kernel service has no evidence for discovery rank=" +
                std::to_string(discovery_rank) + " device=" + device.toString());
        return records_.at(found->second);
    }

    std::optional<PlanningKernelServiceCatalog> PlanningKernelServiceCatalog::collect(
        const std::shared_ptr<IMPIContext> &mpi, const ClusterInventory &inventory,
        const AutomaticOrchestrationRequest &request, const PlanningKernelSamplePlan &plan,
        const std::function<PlanningLoadedKernelSample(const std::shared_ptr<PhysicalMemoryAuthority> &)> &root_load)
    {
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        std::vector<PlanningServiceObserver> assigned;
        const int rank = mpi ? mpi->rank() : 0;
        int node_slot = 0, rounds = 0;
        // Both topology validation and allocation admission are fallible local
        // work inside the shared consensus. Never throw on one rank immediately
        // before another rank enters the native payload broadcast.
        exchangePlanningSamples(mpi, [&](int count) {
            require(count == inventory.world_size, "inventory/communicator size mismatch");
            return std::vector<std::vector<uint8_t>>(count, assignment(inventory, request, plan));
        }, [&](auto envelope) {
            require(std::vector<uint8_t>(envelope.begin(), envelope.end()) == assignment(inventory, request, plan),
                "source/topology/participant assignment disagreement");
            require(!mpi || (mpi->clusterInventory() && mpi->clusterInventory().get() == &inventory),
                "collection requires the context-owned discovery inventory");
            const bool source_free = std::holds_alternative<PlanningStreamingServicePlan>(plan) ||
                std::holds_alternative<PlanningFP32ArithmeticPlan>(plan);
            require(source_free ? !root_load : bool(root_load),
                "source reader must match the requested service kind");
            assigned = observers(inventory, request);
            memory = admit(inventory, rank, plan, assigned);
            const bool local_observer = std::any_of(assigned.begin(), assigned.end(),
                [&](const PlanningServiceObserver &observer) {
                    return observer.discoveryRank() == rank;
                });
            // No-observer source-free ranks join the same publication schedule
            // but have no allocator work.  Every other rank will either publish
            // source bytes or invoke a measurement and must carry PMA proof.
            require(source_free && !local_observer ? !memory : bool(memory),
                "planning allocation authority disagrees with local source/observer ownership");
            std::map<int, int> slots;
            for (const auto &member : inventory.ranks)
            {
                const int slot = slots[member.node_id]++;
                if (member.rank == rank) node_slot = slot;
                rounds = std::max(rounds, slot + 1);
            }
            return std::vector<uint8_t>{1};
        }, [](auto) {});
        const auto sample = std::visit([&](const auto &source) -> PlanningLoadedKernelSample {
            if constexpr (std::is_same_v<std::decay_t<decltype(source)>, PlanningExpertSamplePlan>)
                return PlanningExpertSamplePublication::publish(mpi, source, memory, DeviceId::cpu(),
                    [&] { return std::get<PlanningLoadedExpertSample>(root_load(memory)); });
            else if constexpr (std::is_same_v<std::decay_t<decltype(source)>, PlanningProjectionServicePlan>)
                return PlanningMatrixSamplePublication::publish(mpi, source.matrix(), memory, DeviceId::cpu(),
                    [&] { return std::get<PlanningLoadedMatrixSample>(root_load(memory)); });
            else return std::monostate{};
        }, plan);
        std::vector<PlanningKernelObservation> completed;
        std::optional<PlanningKernelServiceCatalog> result;
        for (int round = 0; round < rounds; ++round)
            exchangePlanningSamples(mpi, [&](int count) {
                const auto text = std::to_string(round);
                return std::vector<std::vector<uint8_t>>(count, {text.begin(), text.end()});
            }, [&](auto envelope) {
                require(std::string(envelope.begin(), envelope.end()) == std::to_string(round), "sampling schedule disagreement");
                if (node_slot == round)
                    for (const auto &observer : assigned)
                    {
                        if (observer.discoveryRank() != rank) continue;
                        const auto device = observer.device();
                        const auto measure = [&] {
                            return std::visit([&](const auto &loaded) -> PlanningKernelObservation {
                                if constexpr (std::is_same_v<std::decay_t<decltype(loaded)>, PlanningLoadedExpertSample>)
                                {
                                    if (device.is_cpu()) return PlanningCPUExpertMeasurement::measure(loaded, device, memory);
                                    return PlanningGPUExpertMeasurement::measure(loaded, device, memory);
                                }
                                else if constexpr (std::is_same_v<std::decay_t<decltype(loaded)>, PlanningLoadedMatrixSample>)
                                    return PlanningProjectionMeasurement::measure(loaded, device, memory,
                                        std::get<PlanningProjectionServicePlan>(plan).prefillRows());
                                else
                                {
                                    if (std::holds_alternative<PlanningFP32ArithmeticPlan>(plan))
                                        return PlanningProjectionMeasurement::measure(PlanningFP32ArithmeticPlan{}, device, memory);
                                    return PlanningMemoryBandwidthMeasurement::measure(
                                        PlanningMemoryBandwidthRequest::fromInventory(inventory.ranks.at(rank), device), memory);
                                }
                            }, sample);
                        };
                        if (device.is_cpu()) completed.emplace_back(measure());
                        else GPUDeviceContextPool::instance().getContext(device).submitAndWait([&] {
                            completed.emplace_back(measure());
                        });
                    }
                return encode(plan, completed);
            }, [&](auto receipts) {
                if (round + 1 == rounds) result = accept(inventory, request, plan, receipts);
            });
        return result;
    }
}
