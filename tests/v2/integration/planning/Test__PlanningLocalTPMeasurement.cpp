/**
 * @file Test__PlanningLocalTPMeasurement.cpp
 * @brief Real NCCL/RCCL retained-graph sampling, resource rejection and retirement proofs.
 *
 * This functional gate has no throughput threshold. Both native transport
 * precisions, odd row/column tails, reversed group order and repeated group
 * construction must produce complete positive device-clock observations. An exhausted
 * endpoint must fail without stranding its peer or retaining physical owners.
 * Separate startup sample cases retain raw clock receipts and wall times in
 * GTest XML, without a performance threshold or per-edge logging overhead.
 * Context-storage replay uses the production four-graph timing-family bound;
 * context leases survive that family's retirement until explicit device reset.
 */
#include "planning/PlanningLocalTPMeasurement.h"
#include "collective/DeviceCountedAllGather.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "backends/GPUGraphMemoryContract.h"
#include "planning/PlanningPeerCopyAccess.h"
#include "planning/RankHardwareOwnership.h"
#include "utils/MPIContext.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "transfer/TransferEngine.h"
#include "transfer/PreparedDevicePeerCopy.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include <array>
#include <cstring>
#include <optional>
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace llaminar2;

namespace
{
    /**
     * @return Every backend participant owned by this original MPI rank, in observed order.
     *
     * The canonical preparation main initializes MPI and publishes FullInventory.
     * Visibility alone grants no ownership: filtering retains UUID evidence and
     * excludes GPUs whose physical NUMA node belongs to another rank/socket.
     */
    std::vector<DeviceId> ownedDevices(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        if (!inventory) throw std::runtime_error("Native LocalTP fixture requires canonical published inventory");
        const auto &rank = inventory->ranks.at(mpi->rank());
        if (!hasCPUBackend()) initCPUBackend(rank.cpu.numa_node);
        std::vector<DeviceId> devices;
        std::string uuids;
        for (const auto &gpu : rank.gpus)
            if (gpu.type == backend && rankOwnsGPUDevice(rank, gpu))
            {
                if (gpu.uuid.empty()) throw std::runtime_error("Owned native GPU has no observed UUID");
                devices.emplace_back(gpu.type, gpu.local_device_id);
                uuids += (uuids.empty() ? "" : ",") + gpu.uuid;
            }
        ::testing::Test::RecordProperty("original_ordered_gpu_uuids", uuids);
        ::testing::Test::RecordProperty("original_cpu_numa", rank.cpu.numa_node);
        return devices;
    }

    /** @return Small explicit test ceiling; production contributor owns every demand. */
    PhysicalMemoryResource resource(DeviceId device)
    {
        if (device.is_gpu())
        {
            auto *backend = getBackendFor(device);
            if (!backend) throw std::runtime_error("Native LocalTP proof lost its physical backend");
            return {.world_rank = MPIContextFactory::global()->rank(), .device = device,
                .total_bytes = backend->deviceMemoryTotal(device.ordinal),
                .admission_available_bytes = backend->deviceMemoryFree(device.ordinal)};
        }
        return {.world_rank = MPIContextFactory::global()->rank(), .device = device, .total_bytes = 256ull << 20,
            .admission_available_bytes = 256ull << 20};
    }

    /** @return Ordinary canonical authority with one collective group admitted. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const PlanningLocalTPRequest &request,
        PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective)
    {
        PhysicalMemoryPlanBuilder builder;
        std::vector<PhysicalMemoryResource> resources;
        for (const auto device : request.devices()) resources.push_back(resource(device));
        PlanningLocalTPMeasurement::contributeMemory(request, resource(DeviceId::cpu()), resources, builder, mode);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), MPIContextFactory::global()->rank());
    }

    /** @brief Supply the exact original-rank counted protocol authority to the production measurement. */
    PlanningLocalTPObservations measureOriginalNative(const PlanningLocalTPRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory,PlanningPairSampling pairs=PlanningPairSampling::Disabled,
        PlanningMeasurementValidation validation=PlanningMeasurementValidation::None,std::span<const uint8_t> access={})
    {
        const auto mpi=MPIContextFactory::global();const auto inventory=mpi->clusterInventory();
        if(!inventory)throw std::runtime_error("Native gather replay requires original discovery inventory");
        const auto topology=DeviceCountedAllGatherTopology::resolve(inventory->ranks.at(mpi->rank()),request.devices());
        return PlanningLocalTPMeasurement::measure(request,memory,pairs,validation,access,
            PlanningLocalTPMeasurementMode::NativeCollective,&topology);
    }

    /** @brief No sampled payload, graph reservation or collective scratch escapes the transaction. */
    void expectRetired(const PlanningLocalTPRequest &request, const PhysicalMemoryAuthority &memory)
    {
        auto devices = request.devices();
        devices.push_back(DeviceId::cpu());
        for (const auto device : devices)
            for (const auto owner : {PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::LocalCollective, PhysicalMemoryOwner::ActivationTransportStaging, PhysicalMemoryOwner::NativeGraphExecutable})
            {
                EXPECT_EQ(memory.claimedBytes(device, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory.reservedBytes(device, owner), 0u);
            }
    }

    /** @brief Replay the production pair sampler with exact source/tail observations outside timing. */
    void proveDirectedPairs(DeviceType backend,
        PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective)
    {
        const auto devices = ownedDevices(backend);
        ASSERT_GE(devices.size(), 2u) << "Exact rank-owned native GPU group is required";
        const PlanningLocalTPRequest request(devices, 257, 31, PlanningAllreducePrecision::FP32);
        auto memory = memoryFor(request, mode);
        const auto mpi = MPIContextFactory::global();
        const auto access = planningDirectedPeerCopyAccess(mpi->clusterInventory()->ranks.at(mpi->rank()), devices);
        const size_t edges = std::count(access.begin(), access.end(), std::uint8_t{1});
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            const auto result = mode==PlanningLocalTPMeasurementMode::NativeCollective
                ? measureOriginalNative(request,memory,PlanningPairSampling::Enabled,PlanningMeasurementValidation::ObservePayloads,access)
                : PlanningLocalTPMeasurement::measure(request,memory,PlanningPairSampling::Enabled,
                    PlanningMeasurementValidation::ObservePayloads,access,mode);
            EXPECT_EQ(result.request.devices(), devices);
            EXPECT_EQ(result.mode, mode);
            EXPECT_EQ(result.phases.size(), mode == PlanningLocalTPMeasurementMode::PhysicalPeersOnly ? 0u : 2u);
            EXPECT_NO_THROW(result.pair_sampling.validate(request));
            EXPECT_EQ(result.pair_sampling.directed_pairs.size(),
                edges * request.pairPayloadAnchors().size());
            EXPECT_EQ(result.pair_sampling.capture_count, edges);
            EXPECT_EQ(result.pair_sampling.measured_launches, result.pair_sampling.capture_count);
            EXPECT_GT(result.pair_sampling.setup_wall_seconds, 0);
            EXPECT_GT(result.pair_sampling.measurement_wall_seconds, 0);
            ::testing::Test::RecordProperty("pair_setup_seconds_repeat_" + std::to_string(repeat),
                std::to_string(result.pair_sampling.setup_wall_seconds));
            ::testing::Test::RecordProperty("pair_replay_seconds_repeat_" + std::to_string(repeat),
                std::to_string(result.pair_sampling.measurement_wall_seconds));
            expectRetired(request, *memory);
        }
        if (mode == PlanningLocalTPMeasurementMode::NativeCollective)
        {
            const auto disabled = measureOriginalNative(request, memory, PlanningPairSampling::Disabled);
            EXPECT_TRUE(disabled.pair_sampling.directed_pairs.empty());
            EXPECT_EQ(disabled.pair_sampling.capture_count, 0u);
        }
        else
            EXPECT_THROW(PlanningLocalTPMeasurement::measure(request, memory, PlanningPairSampling::Disabled,
                PlanningMeasurementValidation::None, {}, mode), std::invalid_argument);
        expectRetired(request, *memory);
    }

    /**
     * @brief Replay the production native allreduce's captured-clock observation and exact PMA retirement.
     *
     * Native observations expose endpoint latency and original body nodes, not
     * a payload download or execution counter. The four-body ordering itself
     * is proven by the shared prepared-parent replay tests; this gate proves
     * the collective producer actually uses its admitted four-graph path.
     */
    void proveNativeCollectiveCapturedClock(DeviceType backend)
    {
        const auto devices=ownedDevices(backend);
        ASSERT_GE(devices.size(),2u)<<"Native captured-clock replay requires one original rank-owned group";
        nlohmann::json observations=nlohmann::json::array();
        for(const auto precision:{PlanningAllreducePrecision::FP32,PlanningAllreducePrecision::FP16}){
            const PlanningLocalTPRequest request(devices,257,31,precision);
            auto memory=memoryFor(request);std::weak_ptr<PhysicalMemoryAuthority> retired=memory;
            for(const auto device:devices)
                EXPECT_EQ(memory->plannedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable),
                    PlanningExecutionMeasurement::kPreparedGraphCount*GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            for(int replay=0;replay<2;++replay){
                const auto result=measureOriginalNative(request,memory,PlanningPairSampling::Disabled);
                EXPECT_EQ(result.mode,PlanningLocalTPMeasurementMode::NativeCollective);
                EXPECT_EQ(result.request.devices(),devices);EXPECT_EQ(result.request.precision(),precision);
                EXPECT_TRUE(result.pair_sampling.directed_pairs.empty());EXPECT_EQ(result.pair_sampling.capture_count,0u);
                ASSERT_EQ(result.phases.size(),2u);
                for(size_t phase_index=0;phase_index<result.phases.size();++phase_index){
                    const auto &phase=result.phases[phase_index];
                    EXPECT_EQ(phase.rows,phase_index?31:1);EXPECT_EQ(phase.payload_bytes,request.payloadBytes(phase.rows));
                    ASSERT_EQ(phase.endpoints.size(),devices.size());double slowest=0;
                    for(size_t index=0;index<devices.size();++index){const auto &endpoint=phase.endpoints[index];
                        EXPECT_EQ(endpoint.device,devices[index]);EXPECT_GT(endpoint.graph_nodes,0u);
                        EXPECT_TRUE(std::isfinite(endpoint.seconds_per_collective));EXPECT_GT(endpoint.seconds_per_collective,0.);
                        slowest=std::max(slowest,endpoint.seconds_per_collective);
                        observations.push_back({{"precision",int(precision)},{"replay",replay},{"rows",phase.rows},
                            {"ordinal",endpoint.device.ordinal},{"body_graph_nodes",endpoint.graph_nodes},
                            {"seconds_per_collective",endpoint.seconds_per_collective},{"live_bytes",phase.payload_bytes}});
                    }
                    EXPECT_DOUBLE_EQ(phase.secondsPerCollective(),slowest);
                }
                expectRetired(request,*memory);
            }
            memory.reset();EXPECT_TRUE(retired.expired());
        }
        ::testing::Test::RecordProperty("native_collective_captured_clock_observations",observations.dump());
        ::testing::Test::RecordProperty("native_collective_clock_scope",
            "original-rank group; four-graph prepared parent; two fresh collections per precision; no host enqueue timestamps or latency threshold");
    }

    /** @brief Replay the actual fixed/count native gather calibration; no pair-copy observation can substitute. */
    void proveNativeGatherCapturedClock(DeviceType backend)
    {
        const auto devices=ownedDevices(backend);ASSERT_GE(devices.size(),2u);
        const PlanningLocalTPRequest request(devices,257,31,PlanningAllreducePrecision::FP32);
        auto memory=memoryFor(request);nlohmann::json records=nlohmann::json::array();
        EXPECT_THROW(PlanningLocalTPMeasurement::measure(request,memory),std::invalid_argument);
        expectRetired(request,*memory);
        for(int replay=0;replay<2;++replay){
            const auto observed=measureOriginalNative(request,memory,PlanningPairSampling::Disabled,PlanningMeasurementValidation::ObservePayloads);
            ASSERT_EQ(observed.gather_phases.size(),4u);
            for(size_t phase_index=0;phase_index<4;++phase_index){const auto &phase=observed.gather_phases[phase_index];
                EXPECT_EQ(phase.protocol,phase_index<2?PlanningNativeGatherProtocol::FixedAllgather:PlanningNativeGatherProtocol::CountedAllgather);
                EXPECT_EQ(phase.payload_bytes,phase_index%2?request.activationBytes():(phase_index<2?1u:0u));
                EXPECT_EQ(phase.participant_count,devices.size());ASSERT_EQ(phase.endpoints.size(),devices.size());
                double slowest=0;
                for(size_t index=0;index<devices.size();++index){const auto &endpoint=phase.endpoints[index];
                    EXPECT_EQ(endpoint.device,devices[index]);EXPECT_GT(endpoint.graph_nodes,0u);
                    EXPECT_TRUE(std::isfinite(endpoint.seconds_per_collective));EXPECT_GT(endpoint.seconds_per_collective,0.);
                    slowest=std::max(slowest,endpoint.seconds_per_collective);
                    records.push_back({{"replay",replay},{"payload_bytes",phase.payload_bytes},{"participants",phase.participant_count},
                        {"ordinal",endpoint.device.ordinal},{"body_graph_nodes",endpoint.graph_nodes},
                        {"seconds_per_group_invocation",endpoint.seconds_per_collective}});
                }
                EXPECT_DOUBLE_EQ(phase.secondsPerCollective(),slowest);
            }
            EXPECT_TRUE(observed.pair_sampling.directed_pairs.empty());expectRetired(request,*memory);
        }
        ::testing::Test::RecordProperty("native_gather_fixed_clock_observations",records.dump());
    }

    /**
     * @brief Observe one production-geometry startup sample and publish its exact receipt in XML.
     * @param backend Homogeneous CUDA or ROCm group owned by this original discovery rank.
     *
     * The total interval includes canonical request admission and the native
     * sampler, but excludes inventory/backend discovery and XML formatting.
     * Validation downloads are disabled as in production. Transaction owners
     * must retire; persistent backend-context ownership is not represented as
     * retired model/sample storage.
     */
    void recordDirectedPairStartup(DeviceType backend,
        PlanningLocalTPMeasurementMode mode = PlanningLocalTPMeasurementMode::NativeCollective)
    {
        const auto devices = ownedDevices(backend);
        ASSERT_GE(devices.size(), 2u) << "Startup sample requires the exact rank-owned native GPU group";
        const PlanningLocalTPRequest request(devices, 2560, 64, PlanningAllreducePrecision::FP32);
        const auto begin = std::chrono::steady_clock::now();
        auto memory = memoryFor(request, mode);
        const auto admitted = std::chrono::steady_clock::now();
        const auto original_mpi = MPIContextFactory::global();
        const auto access = planningDirectedPeerCopyAccess(
            original_mpi->clusterInventory()->ranks.at(original_mpi->rank()), devices);
        const auto result = mode==PlanningLocalTPMeasurementMode::NativeCollective
            ? measureOriginalNative(request,memory,PlanningPairSampling::Enabled,PlanningMeasurementValidation::None,access)
            : PlanningLocalTPMeasurement::measure(request,memory,PlanningPairSampling::Enabled,PlanningMeasurementValidation::None,access,mode);
        const auto completed = std::chrono::steady_clock::now();
        EXPECT_EQ(result.request.devices(), devices);
        EXPECT_EQ(result.mode, mode);
        EXPECT_EQ(result.phases.size(), mode == PlanningLocalTPMeasurementMode::PhysicalPeersOnly ? 0u : 2u);
        EXPECT_NO_THROW(result.pair_sampling.validate(request));
        expectRetired(request, *memory);

        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        using Json = nlohmann::json;
        Json endpoints = Json::array(), edges = Json::array(), retirement = Json::array();
        for (const auto device : devices)
        {
            const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
            ASSERT_NE(found, rank.gpus.end());
            ASSERT_TRUE(rankOwnsGPUDevice(rank, *found));
            endpoints.push_back({{"backend", int(device.type)}, {"ordinal", device.ordinal},
                {"uuid", found->uuid}, {"numa_node", found->numa_node}});
        }
        for (const auto &edge : result.pair_sampling.directed_pairs)
            edges.push_back({{"source_index", edge.source_index}, {"destination_index", edge.destination_index},
                {"live_bytes", edge.payload_bytes}, {"seconds_per_transfer", edge.seconds_per_transfer},
                {"graph_nodes", edge.graph_nodes}});
        bool retired = true;
        auto resources = devices; resources.push_back(DeviceId::cpu());
        for (const auto device : resources)
            for (const auto owner : {PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryOwner::LocalCollective,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                const auto claimed = memory->claimedBytes(device, owner, PhysicalMemoryMaterializationKind::NewAllocation);
                const auto reserved = memory->reservedBytes(device, owner);
                retired = retired && claimed == 0 && reserved == 0;
                if (mode == PlanningLocalTPMeasurementMode::PhysicalPeersOnly && owner == PhysicalMemoryOwner::LocalCollective)
                    EXPECT_EQ(memory->plannedBytes(device, owner), 0u);
                retirement.push_back({{"backend", int(device.type)}, {"ordinal", device.ordinal},
                    {"owner", int(owner)}, {"planned_bytes", memory->plannedBytes(device, owner)},
                    {"claimed_bytes", claimed}, {"reserved_bytes", reserved}});
            }
        const auto &pairs = result.pair_sampling;
        const double native_complete_seconds = std::chrono::duration<double>(completed - begin).count();
        const double admission_seconds = std::chrono::duration<double>(admitted - begin).count();
        Json receipt{{"schema", "llaminar.physical-peer-startup-sample.v2"},
            {"discovery_rank", mpi->rank()}, {"physical_node", rank.node_id}, {"cpu_numa", rank.cpu.numa_node},
            {"ordered_endpoints", endpoints}, {"hidden_width", 2560}, {"prefill_rows", 64},
            {"measurement_mode", mode == PlanningLocalTPMeasurementMode::PhysicalPeersOnly ? "PhysicalPeersOnly" : "NativeCollective"},
            {"native_collective_sample_count", result.phases.size()},
            {"timing_scope", "request PMA admission through sampler native retirement and setup-authority destruction; includes receipt assembly; excludes prior inventory/backend bootstrap and final XML serialization"},
            {"wire_precision", "FP32"}, {"payload_validation", "None"},
            {"raw_method", "captured fixed-frequency device clock delta divided by transfers_per_window; one graph per direction; destination-owned physical peer copy; native collective timing is separate"},
            {"terminal_clock_bytes_per_endpoint", 2 * pairs.payload_anchors.size() * PlanningLocalTPPairSampling::kSamples * sizeof(std::uint64_t)},
            {"raw_samples", PlanningLocalTPPairSampling::kSamples},
            {"transfers_per_window", PlanningLocalTPPairSampling::kBatchTransfers},
            {"windows_per_graph", pairs.payload_anchors.size() * PlanningLocalTPPairSampling::kSamples},
            {"payload_anchors", pairs.payload_anchors}, {"direct_peer_access", pairs.direct_peer_access}, {"directed_observations", edges},
            {"captures", pairs.capture_count}, {"measured_launches", pairs.measured_launches},
            {"admission_wall_seconds", admission_seconds}, {"native_complete_wall_seconds", native_complete_seconds},
            {"pair_setup_wall_seconds", pairs.setup_wall_seconds}, {"pair_replay_wall_seconds", pairs.measurement_wall_seconds},
            {"transaction_pma_retired", retired}, {"transaction_pma_resources", retirement},
            {"retirement_scope", "sample workspace, collective and graph claims; persistent backend contexts retain separate ownership"}};
        std::weak_ptr<PhysicalMemoryAuthority> lifetime = memory;
        memory.reset();
        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        receipt["total_wall_seconds"] = total_seconds;
        receipt["setup_authority_retired"] = lifetime.expired();
        EXPECT_TRUE(retired); EXPECT_TRUE(lifetime.expired());
        ::testing::Test::RecordProperty("startup_total_wall_seconds", std::to_string(total_seconds));
        ::testing::Test::RecordProperty("startup_pair_setup_wall_seconds", std::to_string(pairs.setup_wall_seconds));
        ::testing::Test::RecordProperty("startup_pair_replay_wall_seconds", std::to_string(pairs.measurement_wall_seconds));
        ::testing::Test::RecordProperty("startup_native_directed_pair_receipt", receipt.dump());
    }

    /** @brief Exercise exact backend/group evidence without inferring a link type from its speed. */
    void prove(DeviceType backend)
    {
        auto devices = ownedDevices(backend);
        ASSERT_GE(devices.size(), 2u) << "Native collective planning proof requires two owned GPUs";
        // Every owned member participates; no hard-coded CUDA/ROCm degree.
        for (const auto precision : {PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16})
        {
            const PlanningLocalTPRequest request(devices, 257, 31, precision);
            auto memory = memoryFor(request);
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                const auto result = measureOriginalNative(request, memory);
                ASSERT_EQ(result.phases.size(), 2u);
                EXPECT_EQ(result.request.devices(), devices);
                EXPECT_EQ(result.request.precision(), precision);
                for (size_t phase_index = 0; phase_index < result.phases.size(); ++phase_index)
                {
                    const auto &phase = result.phases[phase_index];
                    EXPECT_EQ(phase.rows, phase_index ? 31 : 1);
                    EXPECT_EQ(phase.payload_bytes, request.payloadBytes(phase.rows));
                    ASSERT_EQ(phase.endpoints.size(), devices.size());
                    EXPECT_GT(phase.secondsPerCollective(), 0.0);
                    for (size_t index = 0; index < devices.size(); ++index)
                    {
                        EXPECT_EQ(phase.endpoints[index].device, devices[index]);
                        EXPECT_GT(phase.endpoints[index].graph_nodes, 0u);
                        EXPECT_TRUE(std::isfinite(phase.endpoints[index].seconds_per_collective));
                    }
                    ::testing::Test::RecordProperty((precision == PlanningAllreducePrecision::FP32 ? "fp32_" : "fp16_") +
                        std::string(phase_index ? "prefill_us" : "decode_us"), std::to_string(phase.secondsPerCollective() * 1e6));
                }
                expectRetired(request, *memory);
            }
            std::reverse(devices.begin(), devices.end());
        }
        const PlanningLocalTPRequest request(devices, 256, 32, PlanningAllreducePrecision::FP32);
        auto memory = memoryFor(request);
        // Exhaust only the final endpoint: earlier endpoint claims must unwind,
        // not leave another worker waiting in capture or leak its reservation.
        for (const auto owner : {PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryOwner::NativeGraphExecutable,
            PhysicalMemoryOwner::LocalCollective})
        {
            {
                auto occupied = memory->reserveNewAllocations(devices.back(), owner,
                    memory->plannedBytes(devices.back(), owner));
                EXPECT_THROW(measureOriginalNative(request, memory), std::exception);
            }
            expectRetired(request, *memory);
        }
        // A previous failed transaction cannot poison the next group's lifecycle.
        EXPECT_NO_THROW(measureOriginalNative(request, memory));
        expectRetired(request, *memory);
    }
}

#ifdef HAVE_CUDA
TEST(PlanningExecutionMeasurementIntegration, CUDA_DirectedPairStartupSample)
{
    ensureNvidiaFactoryRegistered();
    recordDirectedPairStartup(DeviceType::CUDA);
}

TEST(PlanningExecutionMeasurementIntegration, CUDA_DirectedPairMatrixReplay)
{
    ensureNvidiaFactoryRegistered();
    proveDirectedPairs(DeviceType::CUDA);
}

TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTP)
{
    ensureNvidiaFactoryRegistered();
    prove(DeviceType::CUDA);
}

/**
 * @test Cold production-size native kernels retain context bytes after the four-graph timing family retires.
 *
 * The original small odd-tail sample selected a lower-stack native algorithm
 * and did not expose the second growth event. Start this case in its own cold
 * process, use the real 2048-wide/64-row service geometry, and repeat with the
 * exact same admitted graph units. The context's canonical leases survive the
 * setup authority; repeating the family must not grow them again. Only the
 * public exclusive model-retirement transaction may release them, and the next
 * context generation must admit its own new leases before the same sample runs.
 */
namespace
{
/**
 * @brief Prove cold admission, graph retirement and successor ownership at an exact collective geometry.
 * @param columns Actual projection width communicated by each sample.
 * @param rows Exact admitted prefill/collective capacity of the sample family.
 */
void proveContextStorageLifetime(int columns, int rows)
{
    ensureNvidiaFactoryRegistered();
    const auto devices = ownedDevices(DeviceType::CUDA);
    ASSERT_GE(devices.size(), 2u);
    auto *backend = dynamic_cast<CUDABackend *>(getBackendFor(devices.front()));
    ASSERT_NE(backend, nullptr);
    for (const auto device : devices)
        ASSERT_EQ(backend->nativeExecutionContextStorageBytes(device.ordinal), 0u)
            << "Run this cold-context proof as its independently registered preflight case";
    const PlanningLocalTPRequest request(devices, columns, rows, PlanningAllreducePrecision::FP32);
    std::vector<std::size_t> retained;
    std::weak_ptr<PhysicalMemoryAuthority> setup_lifetime;
    {
        auto memory = memoryFor(request);
        setup_lifetime = memory;
        const auto result = measureOriginalNative(request, memory);
        ASSERT_EQ(result.phases.size(), 2u);
        expectRetired(request, *memory);
        for (const auto device : devices)
        {
            const auto bytes = backend->nativeExecutionContextStorageBytes(device.ordinal);
            EXPECT_GT(bytes, 0u);
            retained.push_back(bytes);
            EXPECT_EQ(memory->plannedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable),
                PlanningExecutionMeasurement::kPreparedGraphCount *
                    GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        }
    }
    EXPECT_TRUE(setup_lifetime.expired());
    for (std::size_t index = 0; index < devices.size(); ++index)
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), retained[index]);
    auto replay_memory = memoryFor(request);
    EXPECT_NO_THROW(measureOriginalNative(request, replay_memory));
    expectRetired(request, *replay_memory);
    for (std::size_t index = 0; index < devices.size(); ++index)
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), retained[index]);

    // Destroying graphs was deliberately insufficient. End the complete
    // native-context generation through the same public transaction as model
    // unload, rather than calling cudaDeviceReset behind the lifecycle owner.
    TransferEngine transfers;
    constexpr std::size_t model_workspace_bytes = 4096u;
    std::vector<PhysicalMemoryAllocationLease> model_claims;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> model_buffers;
    for (const auto device : devices)
    {
        model_claims.push_back(replay_memory->claimNewAllocation(device,
            PhysicalMemoryOwner::ExecutionWorkspace, model_workspace_bytes));
        model_buffers.push_back(transfers.allocateDeviceTransferBuffer(model_workspace_bytes, device));
    }
    // A retirement ticket requires a real positive model BOM. These small
    // PMA-owned buffers stand for its final live workspace; release backing
    // before releasing its ledger claims, and both before completing unload.
    std::vector<ExclusiveModelRetirementTicket> tickets;
    for (const auto device : devices)
        tickets.push_back(transfers.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{.device = device, .reusable_workspace_bytes = model_workspace_bytes}));
    model_buffers.clear();
    model_claims.clear();
    const auto receipts = transfers.completeExclusiveModelRetirements(std::move(tickets));
    ASSERT_EQ(receipts.size(), devices.size());
    for (std::size_t index = 0; index < devices.size(); ++index)
    {
        EXPECT_EQ(receipts[index].device, devices[index]);
        EXPECT_TRUE(receipts[index].runtime_reset_invoked);
        EXPECT_EQ(receipts[index].runtime_post_reset_state, DeviceRuntimePostResetState::Quiescent);
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), 0u);
    }

    // Old certificates cannot lend a stale context lease to the successor.
    // Its first captured native kernels must create fresh canonical ownership.
    auto successor_memory = memoryFor(request);
    EXPECT_NO_THROW(measureOriginalNative(request, successor_memory));
    expectRetired(request, *successor_memory);
    for (const auto device : devices)
        EXPECT_GT(backend->nativeExecutionContextStorageBytes(device.ordinal), 0u);
}
}

TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTPContextStorageLifetime)
{
    proveContextStorageLifetime(2048, 64);
}

/** @test The failing large prefill sample owns exact context bytes in both native generations. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTPContextStoragePrefillBucket)
{
    proveContextStorageLifetime(5120, 384);
}

#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration, ROCm_DirectedPairStartupSample)
{
    ensureAMDFactoryRegistered();
    recordDirectedPairStartup(DeviceType::ROCm);
}

TEST(PlanningExecutionMeasurementIntegration, ROCm_DirectedPairMatrixReplay)
{
    ensureAMDFactoryRegistered();
    proveDirectedPairs(DeviceType::ROCm);
}

TEST(PlanningExecutionMeasurementIntegration, ROCm_NativeLocalTP)
{
    ensureAMDFactoryRegistered();
    prove(DeviceType::ROCm);
}
#endif

// Prepared direct-peer contract coverage is independent of startup timing
// receipts.

namespace {
/** @brief Existing admitted payload and native graph owners; no second payload
 * allocation. */
struct PeerContractOwners final {
  PhysicalMemoryAllocationLease host_claim, device_claim;
  PhysicalMemoryOwnerReservation graph_claim;
  std::shared_ptr<FP32Tensor> tensor;
  std::unique_ptr<IGPUGraphCapture> graph;
  std::optional<PreparedDevicePeerCopy> copy;
  void *stream = nullptr;
  PeerContractOwners(const PlanningLocalTPRequest &request, DeviceId device,
                     const std::shared_ptr<PhysicalMemoryAuthority> &memory)
      : host_claim(memory->claimNewAllocation(
            DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
            request.storageBytes())),
        device_claim(memory->claimNewAllocation(
            device, PhysicalMemoryOwner::ExecutionWorkspace,
            request.storageBytes())),
        graph_claim(memory->reserveNewAllocations(
            device, PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(
                    device))) {}
};
/** @brief Throw through the original worker future on a failed native lifecycle
 * step. */
void peerContractRequire(bool success, const char *message) {
  if (!success)
    throw std::runtime_error(message);
}
/**
 * @brief Exercise the public prepared-copy contract with real captured bytes.
 * No performance assertions, model loading or communicator construction occur.
 * Unavailable directed P2P is an explicit negative case, never a copied-path
 * proof.
 */
void provePreparedDevicePeerCopyContract(DeviceType backend) {
  const auto owned = ownedDevices(backend);
  ASSERT_GE(owned.size(), 2u)
      << "Prepared-copy gate requires two original rank-owned GPUs";
  const auto mpi = MPIContextFactory::global();
  const auto access = planningDirectedPeerCopyAccess(
      mpi->clusterInventory()->ranks.at(mpi->rank()), owned);
  size_t source_index = 0, destination_index = 1;
  bool direct = false;
  for (size_t source = 0; source < owned.size() && !direct; ++source)
    for (size_t destination = 0; destination < owned.size(); ++destination)
      if (source != destination &&
          access[source * owned.size() + destination]) {
        source_index = source;
        destination_index = destination;
        direct = true;
        break;
      }
  const std::array devices{owned[source_index], owned[destination_index]};
  const PlanningLocalTPRequest request({devices[0], devices[1]}, 64, 1,
                                       PlanningAllreducePrecision::FP32);
  auto memory = memoryFor(request);
  TransferEngine transfers;
  std::array<std::unique_ptr<PeerContractOwners>, 2> owners;
  for (size_t index = 0; index < 2; ++index)
    owners[index] =
        std::make_unique<PeerContractOwners>(request, devices[index], memory);
  auto &pool = GPUDeviceContextPool::instance();
  const auto on = [&](size_t index,
                      const std::function<void(IWorkerGPUContext &)> &body) {
    auto &worker = pool.getContext(devices[index]);
    worker.submitAsync([&worker, &body] { body(worker); }).get();
  };
  const auto retire = [&] {
    // Cross-endpoint tensor borrows die before either payload claim. Keep
    // both original workers alive until their graphs and copies are gone.
    for (size_t index = 0; index < 2; ++index)
      on(index, [&](auto &) {
        owners[index]->graph.reset();
        owners[index]->copy.reset();
      });
    for (size_t index = 0; index < 2; ++index)
      on(index, [&](auto &) { owners[index].reset(); });
  };
  try {
    for (size_t index = 0; index < 2; ++index)
      on(index, [&, index](auto &worker) {
        auto &owner = *owners[index];
        owner.stream = worker.defaultStream();
        owner.tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{request.storageBytes() / sizeof(float)});
        std::memset(owner.tensor->mutable_data(), index ? 0xa5 : 0x39,
                    request.storageBytes());
        if (index || !direct)
          TransferEngine::prepareDeviceInput(owner.tensor.get(), devices[index],
                                             owner.stream);
        else
          TransferEngine::prepareDeviceOutput(owner.tensor.get(),
                                              devices[index], owner.stream);
      });
    if (!direct) {
      on(1, [&](auto &) {
        try {
          auto unexpected = transfers.prepareDevicePeerCopy(
              owners[0]->tensor, owners[1]->tensor, devices[0], devices[1],
              ExplicitGPUStream(owners[1]->stream));
          FAIL() << "Observed unavailable edge was accepted";
        } catch (const std::runtime_error &error) {
          EXPECT_NE(std::string(error.what()).find("unavailable physical P2P"),
                    std::string::npos);
        }
      });
      ::testing::Test::RecordProperty("positive_direct_peer_contract_exercised",
                                      false);
    } else {
      on(0, [&](auto &worker) {
        auto &owner = *owners[0];
        owner.graph = worker.createGraphCapture(owner.stream);
        peerContractRequire(bool(owner.graph),
                            "Missing-publication graph owner missing");
        ScopedBackendGraphCapture capture(
            worker, *owner.graph, "prepared peer missing publication control");
        peerContractRequire(capture.begin(),
                            "Missing-publication capture begin failed");
        auto *native = getBackendFor(devices[0]);
        peerContractRequire(
            native && native->memset(owner.tensor->gpu_data_ptr(), 0x39,
                                     request.storageBytes(), devices[0].ordinal,
                                     owner.stream),
            "Missing-publication capture memset failed");
        TransferEngine::publishDeviceWrite(owner.tensor.get(), devices[0],
                                           owner.stream);
        capture.finish(); // Deliberately never launched: no producer event can
                          // prove completion.
      });
      on(1, [&](auto &) {
        try {
          auto unexpected = transfers.prepareDevicePeerCopy(
              owners[0]->tensor, owners[1]->tensor, devices[0], devices[1],
              ExplicitGPUStream(owners[1]->stream));
          FAIL() << "Unpublished source was accepted";
        } catch (const std::logic_error &error) {
          const std::string reason = error.what();
          EXPECT_TRUE(
              reason.find("producer event") != std::string::npos ||
              reason.find("existing valid physical tensor allocations") !=
                  std::string::npos)
              << "Unexecuted captured output was rejected for an unrelated "
                 "reason: "
              << reason;
        }
      });
      on(0, [&](auto &) {
        owners[0]->graph.reset();
        // Replace the deliberately unexecuted output with an actual external
        // GPU producer. A host read cannot acquire unpublished device bytes.
        auto *native = getBackendFor(devices[0]);
        peerContractRequire(
            native && native->memset(owners[0]->tensor->gpu_data_ptr(), 0x39,
                                     request.storageBytes(), devices[0].ordinal,
                                     owners[0]->stream),
            "Prepared peer source reset producer failed");
        TransferEngine::publishDeviceWrite(owners[0]->tensor.get(), devices[0],
                                           owners[0]->stream);
      });
      on(1, [&](auto &) {
        EXPECT_THROW((void)transfers.prepareDevicePeerCopy(
                         owners[0]->tensor, owners[1]->tensor, devices[0],
                         devices[1], ExplicitGPUStream(owners[0]->stream)),
                     std::invalid_argument);
        owners[1]->copy.emplace(transfers.prepareDevicePeerCopy(
            owners[0]->tensor, owners[1]->tensor, devices[0], devices[1],
            ExplicitGPUStream(owners[1]->stream)));
        const size_t capacity = request.storageBytes();
        auto &copy = *owners[1]->copy;
        EXPECT_NO_THROW(copy.enqueue(capacity, capacity, 0));
        EXPECT_THROW(copy.enqueue(capacity + 1, 0, 0), std::out_of_range);
        EXPECT_THROW(copy.enqueue(0, capacity + 1, 0), std::out_of_range);
        EXPECT_THROW(copy.enqueue(capacity, 0, 1), std::out_of_range);
        EXPECT_THROW(copy.enqueue(0, capacity, 1), std::out_of_range);
        EXPECT_THROW(copy.enqueue(0, 0, capacity + 1), std::out_of_range);
        auto moved = std::move(copy);
        EXPECT_THROW(copy.enqueue(0, 0, 0), std::logic_error);
        copy = std::move(moved);
      });
      on(0, [&](auto &) {
        EXPECT_THROW(owners[1]->copy->enqueue(0, 0, 1), std::logic_error);
      });
      // Large-to-small independent captures reuse the same original
      // payloads/PMA.
      for (const size_t bytes : {37u, 7u})
        on(1, [&](auto &worker) {
          auto &owner = *owners[1];
          owner.graph.reset();
          owner.copy.reset();
          // Terminal host observation may retire its publication event. Each
          // independent capture cold-prepares a fresh exact-stream capability;
          // payload allocations and the original PMA remain unchanged.
          owner.copy.emplace(transfers.prepareDevicePeerCopy(
              owners[0]->tensor, owner.tensor, devices[0], devices[1],
              ExplicitGPUStream(owner.stream)));
          owner.graph = worker.createGraphCapture(owner.stream);
          peerContractRequire(bool(owner.graph),
                              "Prepared peer graph owner missing");
          ScopedBackendGraphCapture capture(
              worker, *owner.graph, "prepared peer exact live tail replay");
          peerContractRequire(capture.begin(),
                              "Prepared peer capture begin failed");
          EXPECT_THROW((void)transfers.prepareDevicePeerCopy(
                           owners[0]->tensor, owners[1]->tensor, devices[0],
                           devices[1], ExplicitGPUStream(owner.stream)),
                       std::logic_error);
          auto *native = getBackendFor(devices[1]);
          owner.copy->enqueue(3, 5, bytes);
          capture.finish();
          // A physical copy must be represented by a retained node. In
          // particular HIP PeerAsync may execute eagerly and leave no node.
          EXPECT_GE(owner.graph->nodeCount(), 1u);
          peerContractRequire(
              owner.graph->prepareRuntimeContextStorage(memory) &&
                  owner.graph->nodeCount() && owner.graph->instantiate(),
              "Prepared peer graph materialization failed");
          for (int replay = 0; replay < 2; ++replay) {
            // Poison AFTER recording and before EVERY replay. Capture-time
            // eager work cannot satisfy this oracle; only a replayed copy can.
            peerContractRequire(
                native && native->memset(owner.tensor->gpu_data_ptr(), 0xa5,
                                         request.storageBytes(),
                                         devices[1].ordinal, owner.stream),
                "Prepared peer pre-replay guard poison failed");
            TransferEngine::publishDeviceWrite(owner.tensor.get(), devices[1],
                                               owner.stream);
            peerContractRequire(owner.graph->launchOnStream(owner.stream),
                                "Prepared peer replay failed");
            TransferEngine::publishDeviceWrite(owner.tensor.get(), devices[1],
                                               owner.stream);
            peerContractRequire(
                owner.tensor->ensureOnHost(owner.stream),
                "Prepared peer terminal byte observation failed");
            const auto *observed =
                reinterpret_cast<const unsigned char *>(owner.tensor->data());
            for (size_t offset = 0; offset < request.storageBytes(); ++offset)
              EXPECT_EQ(observed[offset],
                        offset >= 5 && offset < 5 + bytes ? 0x39 : 0xa5)
                  << "live=" << bytes << " offset=" << offset
                  << " replay=" << replay;
          }
        });
      on(1, [&](auto &) { owners[1]->graph.reset(); });
      on(0, [&](auto &) {
        peerContractRequire(
            owners[0]->tensor->releaseDeviceMemory(),
            "Prepared peer identity control failed to release source storage");
      });
      on(1, [&](auto &) {
        EXPECT_THROW(owners[1]->copy->enqueue(0, 0, 1), std::logic_error);
        EXPECT_THROW(owners[1]->copy->enqueue(0, 0, 0), std::logic_error);
      });
      ::testing::Test::RecordProperty("positive_direct_peer_contract_exercised",
                                      true);
    }
  } catch (...) {
    retire();
    throw;
  }
  retire();
  expectRetired(request, *memory);
}
} // namespace
#ifdef HAVE_CUDA
TEST(PlanningExecutionMeasurementIntegration,
     CUDA_PreparedDevicePeerCopyContract) {
  ensureNvidiaFactoryRegistered();
  provePreparedDevicePeerCopyContract(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration,
     ROCm_PreparedDevicePeerCopyContract) {
  ensureAMDFactoryRegistered();
  provePreparedDevicePeerCopyContract(DeviceType::ROCm);
}
#endif

#ifdef HAVE_CUDA
/** @brief Replayed physical-only captures prove copied bytes without a native protocol sample. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_PhysicalPeerOnlyReplay)
{
    ensureNvidiaFactoryRegistered();
    proveDirectedPairs(DeviceType::CUDA, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
}
#endif
#ifdef HAVE_ROCM
/** @brief HIP must retain actual copy nodes; eager capture-time copies cannot satisfy this oracle. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_PhysicalPeerOnlyReplay)
{
    ensureAMDFactoryRegistered();
    proveDirectedPairs(DeviceType::ROCm, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
}
#endif

#ifdef HAVE_CUDA
/** @brief Full physical-only collector wall includes allocation, recording and native/PMA retirement. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_PhysicalPeerOnlyStartupSample)
{
    ensureNvidiaFactoryRegistered();
    recordDirectedPairStartup(DeviceType::CUDA, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
}
#endif
#ifdef HAVE_ROCM
/** @brief Full HIP collector timing excludes all unused RCCL communicator/calibration work. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_PhysicalPeerOnlyStartupSample)
{
    ensureAMDFactoryRegistered();
    recordDirectedPairStartup(DeviceType::ROCm, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
}
#endif

#ifdef HAVE_CUDA
/** @test Production NCCL allreduce uses the fixed-clock parent and retires every repeated observation. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeCollectiveCapturedClockReplay)
{
    ensureNvidiaFactoryRegistered();proveNativeCollectiveCapturedClock(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
/** @test Production RCCL allreduce stays within original rank-local NUMA ownership through reset/replay. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_NativeCollectiveCapturedClockReplay)
{
    ensureAMDFactoryRegistered();proveNativeCollectiveCapturedClock(DeviceType::ROCm);
}
#endif

#ifdef HAVE_CUDA
/** @test Actual NCCL gather protocols retain control and live-byte calibration across complete native reset/replay. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeGatherCapturedClockReplay)
{ensureNvidiaFactoryRegistered();proveNativeGatherCapturedClock(DeviceType::CUDA);}
#endif
#ifdef HAVE_ROCM
/** @test Actual RCCL gather sampling remains rank-local and retires every repeated captured whole-group gather. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_NativeGatherCapturedClockReplay)
{ensureAMDFactoryRegistered();proveNativeGatherCapturedClock(DeviceType::ROCm);}
#endif
