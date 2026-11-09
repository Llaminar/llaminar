/**
 * @file Test__PlanningKernelServiceCatalog.cpp
 * @brief Device-free proof that evidence never invents topology or execution scope.
 *
 * Synthetic positive times exercise only the codec/identity contract. Ordinal
 * aliases, reversed receipt order, missing results and malformed work/graph/ISA
 * fields must not become a usable catalog. No test here opens a device.
 */
#include "../../utils/ScopedGDNProjectionArithmetic.h"
#include "planning/PlanningKernelServiceCatalog.h"
#include "planning/RankHardwareOwnership.h"
#include "planning/PlanningWeightServiceModel.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "planning/PlanningCommunicationService.h"
#include "planning/PlanningCommunicationCost.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PlanningRequestCostModel.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/CPUFeatures.h"
#include "utils/DebugEnv.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @return Two aliases of a local GPU and an identically named remote GPU. */
    ClusterInventory inventory()
    {
        ClusterInventory result;
        result.world_size = 3;
        for (int rank = 0; rank < 3; ++rank)
        {
            RankInventory member;
            member.rank = rank;
            member.node_id = rank == 2 ? 1 : 0;
            member.local_rank = rank == 2 ? 0 : rank;
            member.hostname = "same-name-is-not-membership";
            member.cpu.type = DeviceType::CPU;
            member.cpu.memory_bytes = member.cpu.free_memory_bytes = size_t{1} << 30;
            member.cpu.numa_node = rank == 0 ? -1 : 0;
            member.cpu.last_level_cache_bytes = 32u << 20;
            member.cpu_memory_bytes = size_t{1} << 30;
            member.cpu_cores = 16;
            member.cpu_worker_threads = 2;
            member.cpu_execution = kSyntheticCPUExecutionGeometry;
            CPUSocketInfo socket;
            socket.numa_node = 0;
            socket.socket_id = 0;
            socket.memory_bytes = size_t{1} << 30;
            socket.physical_cores = {0, 1};
            member.cpu_socket_info.push_back(socket);
            for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                DeviceInfo gpu;
                gpu.type = backend;
                gpu.local_device_id = rank == 0 ? 4 : 0;
                gpu.uuid = "same-uuid-distinct-node-and-vendor";
                gpu.numa_node = 0;
                gpu.memory_bytes = gpu.free_memory_bytes = size_t{1} << 30;
                gpu.last_level_cache_bytes = 6u << 20;
                member.gpus.push_back(gpu);
            }
            result.ranks.push_back(member);
        }
        return result;
    }

    /** @return Partially overlapping local visibility plus a physically distinct remote group. */
    ClusterInventory communicationInventory()
    {
        auto result = inventory();
        for (auto &rank : result.ranks)
        {
            // Known remote host scope: this rank may first-touch pages, but
            // its NUMA7 does not authorize the visible NUMA0 GPUs.
            if (rank.rank == 0)
            {
                rank.cpu.numa_node = 7;
                rank.cpu_socket_info.front().numa_node = 7;
                rank.cpu_socket_info.front().physical_cores = {2, 3};
            }
            const auto first = rank.gpus;
            for (const auto &gpu : first)
                for (int index = 1; index < (rank.rank == 1 ? 3 : 2); ++index)
                {
                    auto extra = gpu;
                    extra.local_device_id += index * 10;
                    extra.uuid = "0-" + gpu.uuid + "-" + std::to_string(index);
                    rank.gpus.push_back(extra);
                }
            for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                auto &width = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
                auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
                width = std::count_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) { return gpu.type == backend; });
                matrix.assign(size_t(width) * width, true); // Explicit synthetic physical access, independent of ordinal order.
            }
        }
        return result;
    }

    /** @return Synthetic completed native observations, grouped by authenticated reporting rank. */
    std::vector<std::vector<uint8_t>> nativeReceipts(const PlanningCommunicationSamplePlan &plan, int count)
    {
        std::vector<std::vector<std::pair<size_t, PlanningLocalTPObservations>>> groups(count);
        std::vector<std::vector<std::pair<size_t, PlanningHostDeviceObservations>>> host_groups(count);
        for (size_t index = 0; index < plan.native().size(); ++index)
        {
            const auto &sample = plan.native()[index];
            PlanningLocalTPObservations observed{sample.request, {}};
            observed.mode = sample.mode;
            if (sample.mode == PlanningLocalTPMeasurementMode::NativeCollective)
            for (int rows : {1, sample.request.prefillRows()})
            {
                PlanningLocalTPPhaseObservation phase{rows, sample.request.payloadBytes(rows), {}};
                for (const auto device : sample.request.devices())
                    phase.endpoints.push_back({device, 3, .0001 * (phase.endpoints.size() + 1)});
                observed.phases.push_back(std::move(phase));
            }
            if(sample.mode==PlanningLocalTPMeasurementMode::NativeCollective)
                for(const auto protocol:{PlanningNativeGatherProtocol::FixedAllgather,PlanningNativeGatherProtocol::CountedAllgather})
                    for(const size_t bytes:{protocol==PlanningNativeGatherProtocol::FixedAllgather?size_t{1}:size_t{0},sample.request.activationBytes()}){
                        PlanningLocalTPGatherPhaseObservation phase{protocol,bytes,sample.request.devices().size(),{}};
                        for(const auto device:sample.request.devices())phase.endpoints.push_back({device,std::max<size_t>(3,sample.request.devices().size()),
                            .0001*(phase.endpoints.size()+1)*(protocol==PlanningNativeGatherProtocol::FixedAllgather?1.:2.)*(bytes==sample.request.activationBytes()?64.:1.)});
                        observed.gather_phases.push_back(std::move(phase));
                    }
            if (sample.pair_sampling == PlanningPairSampling::Enabled)
            {
                auto &pairs = observed.pair_sampling;
                pairs.payload_anchors = sample.request.pairPayloadAnchors();
                pairs.direct_peer_access = sample.direct_peer_access;
                for (size_t source = 0; source < sample.request.devices().size(); ++source)
                    for (size_t destination = 0; destination < sample.request.devices().size(); ++destination)
                        if (pairs.direct_peer_access[source * sample.request.devices().size() + destination])
                            for (const auto bytes : pairs.payload_anchors)
                                pairs.directed_pairs.push_back({source, destination, bytes,
                                    {.00001, .00002, .00003}, 4});
                pairs.setup_wall_seconds = .01;
                pairs.measurement_wall_seconds = .02;
                pairs.capture_count = pairs.directed_pairs.size() / pairs.payload_anchors.size();
                pairs.measured_launches = pairs.capture_count;
            }
            groups.at(sample.discovery_rank).emplace_back(index, std::move(observed));
        }
        for (size_t index = 0; index < plan.hostDevice().size(); ++index)
        {
            const auto &request = plan.hostDevice()[index];
            PlanningHostDeviceObservations observed{request, {}};
            for (auto mechanism : {PlanningHostTransferMechanism::DMA, PlanningHostTransferMechanism::MappedKernel})
                for (auto direction : {MappedTransferDirection::HostToDevice, MappedTransferDirection::DeviceToHost})
                    for (size_t payload : {size_t{1}, request.bulkBytes()})
                        observed.phases.push_back({mechanism, direction, payload, 1,
                            PlanningServiceObservation(PlanningWorkUnit::Bytes, 3.0 * payload, .0003,
                                "synthetic completed GPU/host codec sample")});
            host_groups[request.gpuRank()].emplace_back(index, std::move(observed));
        }
        std::vector<std::vector<uint8_t>> result;
        for (size_t rank = 0; rank < groups.size(); ++rank)
            result.push_back(PlanningCommunicationService::encodeLocal(plan, groups[rank], host_groups[rank]));
        return result;
    }

    /** @return Expert source used only to authenticate immutable sample metadata. */
    PlanningExpertSampleRequest request()
    {
        return {{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
    }

    /** @return Explicit synthetic observation for codec tests, not a measured performance rate. */
    PlanningKernelObservation observation(const PlanningServiceObserver &observer, const PlanningExpertSamplePlan &plan)
    {
        const auto &shape = plan.description();
        PlanningKernelObservation result = PlanningGPUExpertObservations{observer.device(),
            shape.matrices[0].k, shape.matrices[0].n, 1024, shape.formats, plan.request(), {}};
        if (observer.device().is_cpu())
            result = PlanningCPUExpertObservations{observer.device(), 2,
                ISALevel::AVX2, ISALevel::AVX512, shape.matrices[0].k, shape.matrices[0].n, 1024,
                shape.formats, plan.request(), {}};
        for (auto phase : kPlanningExpertStartupPhases)
        {
            const int rows = MoEOverlayCPUServiceMeasurement::rowsForPhase(phase);
            PlanningServiceObservation service(PlanningWorkUnit::ArithmeticOperations,
                6.0 * rows * shape.matrices[0].n * shape.matrices[0].k * 3,
                observer.physicalNode() == 0 ? 1.0 : 0.00001, "synthetic codec unit fixture");
            if (observer.device().is_cpu()) std::get<PlanningCPUExpertObservations>(result).phases.push_back({phase, rows, service});
            else std::get<PlanningGPUExpertObservations>(result).phases.push_back({phase, rows, 10, service});
        }
        return result;
    }

    /** @return Receipts retained in owning buffers, with no source payload or device allocation. */
    std::vector<std::vector<uint8_t>> receipts(const ClusterInventory &cluster, const PlanningExpertSamplePlan &plan)
    {
        std::vector<std::vector<PlanningKernelObservation>> values(cluster.world_size);
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}))
            values[observer.discoveryRank()].push_back(observation(observer, plan));
        std::vector<std::vector<uint8_t>> result;
        for (const auto &local : values) result.push_back(PlanningKernelServiceCatalog::encode(plan, local));
        return result;
    }

    /** @return Borrowed transport-ranked receipts; rank identity is outside the JSON body. */
    std::vector<RankPlanningSample> borrow(const std::vector<std::vector<uint8_t>> &owned)
    {
        std::vector<RankPlanningSample> result;
        for (size_t i = 0; i < owned.size(); ++i) result.push_back({int(i), owned[i]});
        return result;
    }

    /** @return Source-free byte receipts; topology comes from transport and inventory, not timings. */
    std::vector<std::vector<uint8_t>> streamingReceipts(const ClusterInventory &cluster,
        const AutomaticOrchestrationRequest &policy = AutomaticOrchestrationRequest{})
    {
        std::vector<std::vector<PlanningKernelObservation>> rows(cluster.world_size);
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, policy))
        {
            const auto request = PlanningMemoryBandwidthRequest::fromInventory(
                cluster.ranks[observer.discoveryRank()], observer.device());
            rows[observer.discoveryRank()].push_back(PlanningMemoryBandwidthObservation{request,
                observer.device().is_gpu() ? 1u : 0u,
                PlanningServiceObservation(PlanningWorkUnit::Bytes, 3.0 * request.usefulBytes(),
                    observer.physicalNode() ? .0001 : .1, "synthetic streaming codec fixture")});
        }
        std::vector<std::vector<uint8_t>> result;
        for (const auto &local : rows) result.push_back(PlanningKernelServiceCatalog::encode(PlanningStreamingServicePlan{}, local));
        return result;
    }

    /** @return Source-free FP32 receipts without inventing a model tensor or another endpoint inventory. */
    std::vector<std::vector<uint8_t>> arithmeticReceipts(const ClusterInventory &cluster)
    {
        const PlanningFP32ArithmeticPlan plan;
        std::vector<std::vector<PlanningKernelObservation>> rows(cluster.world_size);
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}))
        {
            PlanningFP32ArithmeticObservations value{observer.device(), observer.device().is_cpu() ? 0u : 1024u, {}, {}};
            if (observer.device().is_cpu()) value.cpu = PlanningProjectionCPUObservation{
                cluster.ranks[observer.discoveryRank()].cpuWorkerThreads(), ISALevel::AVX512, ISALevel::AVX512,
                cluster.ranks[observer.discoveryRank()].cpu_execution};
            for (int count : {1, plan.kPrefillRows})
                value.phases.push_back({count, observer.device().is_gpu() ? 4u : 0u,
                    PlanningServiceObservation(PlanningWorkUnit::ArithmeticOperations,
                        3.0 * 2 * count * plan.kN * plan.kK, .0002, "synthetic codec fixture, FP32 arithmetic proxy")});
            rows[observer.discoveryRank()].push_back(std::move(value));
        }
        std::vector<std::vector<uint8_t>> result;
        for (const auto &local : rows) result.push_back(PlanningKernelServiceCatalog::encode(plan, local));
        return result;
    }

    /** @return Synthetic operation-specific receipts, using each rank's published CPU geometry. */
    std::vector<std::vector<uint8_t>> projectionReceipts(const ClusterInventory &cluster,
        const PlanningProjectionServicePlan &plan,
        const AutomaticOrchestrationRequest &policy = AutomaticOrchestrationRequest{})
    {
        std::vector<std::vector<PlanningKernelObservation>> rows(cluster.world_size);
        const auto shape = plan.matrix().geometry();
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, policy))
        {
            PlanningProjectionObservations value{plan.matrix(), observer.device(), 1024, {}, {}};
            if (observer.device().is_cpu()) value.cpu = PlanningProjectionCPUObservation{
                cluster.ranks[observer.discoveryRank()].cpuWorkerThreads(), ISALevel::AVX512, ISALevel::AVX512,
                cluster.ranks[observer.discoveryRank()].cpu_execution};
            const auto type = plan.matrix().executionType();
            if (value.cpu && (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16))
                value.prepared_bytes = 0; // Ordinary floating CPU kernels borrow the native source.
            for (int count : {1, plan.prefillRows()})
                value.phases.push_back({count, observer.device().is_gpu() ? 4u : 0u,
                    PlanningServiceObservation(PlanningWorkUnit::ArithmeticOperations,
                        3.0 * 2 * count * shape.n * shape.k, .0002, "synthetic ordinary projection fixture")});
            rows[observer.discoveryRank()].push_back(std::move(value));
        }
        std::vector<std::vector<uint8_t>> result;
        for (const auto &local : rows) result.push_back(PlanningKernelServiceCatalog::encode(plan, local));
        return result;
    }

}

TEST(PlanningKernelServiceCatalog, SourceFreeArithmeticRetainsPhysicalEndpointAndExactWorkshare)
{
    const auto cluster = inventory();
    const auto owned = arithmeticReceipts(cluster);
    const auto catalog = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
        PlanningFP32ArithmeticPlan{}, borrow(owned));
    EXPECT_TRUE(std::holds_alternative<PlanningFP32ArithmeticPlan>(catalog.source()));
    for (const auto &record : catalog.records())
    {
        const auto &value = std::get<PlanningFP32ArithmeticObservations>(record.observation);
        EXPECT_EQ(record.observer.device(), value.device);
        if (value.cpu) EXPECT_EQ(value.cpu->workers, cluster.ranks.at(record.observer.discoveryRank()).cpuWorkerThreads());
        EXPECT_EQ(value.phases.size(), 2u);
        const std::array<PlanningKernelObservation, 1> observations{value};
        EXPECT_THROW(PlanningKernelServiceCatalog::encode(PlanningStreamingServicePlan{}, observations), std::invalid_argument);
    }
}

TEST(PlanningKernelServiceCatalog, ArithmeticProxyRejectsForeignGeometryWorkRowsAndObserver)
{
    const auto cluster = inventory();
    const auto original = arithmeticReceipts(cluster);
    for (int defect = 0; defect != 9; ++defect)
    {
        auto owned = original;
        auto wire = nlohmann::json::parse(owned[0]);
        if (defect == 0) wire["source"]["N"] = 512;
        if (defect == 1) wire["source"]["kind"] = "projection";
        if (defect == 2) wire["observations"][0]["phases"][1]["rows"] = 32;
        if (defect == 3) wire["observations"][0]["phases"][0]["work"] = 1;
        if (defect == 4) wire["observations"][0]["cpu"]["workers"] = 16;
        if (defect == 5) wire["observations"][0]["phases"][0]["graph_nodes"] = 1;
        if (defect == 6) wire["observations"][0]["prepared_bytes"] = 100;
        if (defect == 7) wire["schema"] = "llaminar.planning-kernel-service.v3";
        if (defect == 8) wire["source"]["K"] = 2048;
        const auto text = wire.dump();
        owned[0] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
            PlanningFP32ArithmeticPlan{}, borrow(owned)), std::exception) << defect;
    }
}

TEST(PlanningKernelServiceCatalog, FP32ModelSourceIsNotSourceFreeArithmeticEvidence)
{
    PlanningGGUFFixture fixture(false, false, GGUFTensorType::F32);
    PlanningModelSource source(fixture.path());
    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}}), 64);
    const auto cluster = inventory();
    const auto source_receipts = projectionReceipts(cluster, plan);
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
        PlanningFP32ArithmeticPlan{}, borrow(source_receipts)), std::invalid_argument);
    const auto arithmetic_receipts = arithmeticReceipts(cluster);
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
        plan, borrow(arithmetic_receipts)), std::invalid_argument);
}

TEST(PlanningKernelServiceCatalog, StreamingBytesPreserveCacheWorkshareAndPhysicalIdentity)
{
    const auto cluster = inventory();
    const auto owned = streamingReceipts(cluster);
    auto ranked = borrow(owned);
    std::reverse(ranked.begin(), ranked.end());
    const auto catalog = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
        PlanningStreamingServicePlan{}, ranked);
    ASSERT_EQ(catalog.records().size(), 7u);
    EXPECT_EQ(&catalog.serviceFor(0, DeviceId::cuda(4)), &catalog.serviceFor(1, DeviceId::cuda(0)));
    EXPECT_NE(&catalog.serviceFor(0, DeviceId::cuda(4)), &catalog.serviceFor(2, DeviceId::cuda(0)));
    for (const auto &record : catalog.records())
    {
        const auto &value = std::get<PlanningMemoryBandwidthObservation>(record.observation);
        EXPECT_EQ(value.request.rank(), record.observer.discoveryRank());
        EXPECT_EQ(value.request.device(), record.observer.device());
        EXPECT_EQ(value.service.unit(), PlanningWorkUnit::Bytes);
        EXPECT_EQ(value.service.completedWork(), 3.0 * value.request.usefulBytes());
        EXPECT_EQ(record.observer.physicalNode(), cluster.connectionBetweenRanks(0, value.request.rank()).destinationNode());
    }
}

TEST(PlanningKernelServiceCatalog, StreamingRejectsStaleGeometryFabricatedWorkAndSourceSubstitution)
{
    const auto cluster = inventory();
    const auto original = streamingReceipts(cluster);
    for (int defect = 0; defect < 15; ++defect)
    {
        SCOPED_TRACE(defect);
        auto owned = original;
        auto wire = nlohmann::json::parse(owned[1]);
        auto &cpu = wire["observations"][0], &gpu = wire["observations"][1];
        if (defect == 0) cpu["cache_bytes"] = 32;
        if (defect == 1) gpu["stream_bytes"] = 64;
        if (defect == 2) cpu["workers"] = 1;
        if (defect == 3) cpu["execution"][0] = 17;
        if (defect == 4) gpu["graph_nodes"] = 0;
        if (defect == 5) cpu["graph_nodes"] = 1;
        if (defect == 6) cpu["work"] = 1;
        if (defect == 7) gpu["work"] = 0;
        if (defect == 8) cpu["seconds"] = -1;
        if (defect == 9) gpu["ordinal"] = 4;
        if (defect == 10) wire["source"]["kind"] = "projection";
        if (defect == 11) wire["observations"].erase(0);
        if (defect == 12) gpu["rank"] = 2;
        if (defect == 13) cpu["workers"] = 2.5;
        if (defect == 14) cpu["provenance"] = "";
        const auto text = wire.dump();
        owned[1] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
            PlanningStreamingServicePlan{}, borrow(owned)), std::exception);
    }
    PlanningGGUFFixture fixture(true);
    PlanningModelSource source(fixture.path());
    const auto expert = PlanningExpertSamplePlan::resolve(source, request());
    const auto observers = PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{});
    const std::array wrong_operation{observation(observers.front(), expert)};
    EXPECT_THROW(PlanningKernelServiceCatalog::encode(PlanningStreamingServicePlan{}, wrong_operation), std::invalid_argument);
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, expert,
        borrow(original)), std::invalid_argument);
}

/** @test Ordinary projections share reporter identity without being mistaken for expert service. */
TEST(PlanningKernelServiceCatalog, ProjectionSourceShardsAndCPUExecutionScopeRoundTrip)
{
    PlanningGGUFFixture fixture(false, false, GGUFTensorType::Q6_K);
    PlanningModelSource source(fixture.path());
    const auto matrix = PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ffn_down.weight", PlanningMatrixColumns{256, 512}});
    EXPECT_THROW(PlanningProjectionServicePlan(matrix, 0), std::invalid_argument);
    EXPECT_THROW(PlanningProjectionServicePlan(matrix, -1), std::invalid_argument);
    const PlanningProjectionServicePlan plan(matrix, 32);
    const auto cluster = inventory();
    const auto owned = projectionReceipts(cluster, plan);
    auto ranked = borrow(owned);
    std::reverse(ranked.begin(), ranked.end());
    const auto catalog = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, ranked);
    ASSERT_EQ(catalog.records().size(), 7u);
    EXPECT_EQ(std::get<PlanningProjectionServicePlan>(catalog.source()).prefillRows(), 32);
    // Rank 0's ordinal 4 and rank 1's ordinal 0 name one physical device.
    // The remote host deliberately repeats the same UUID: it must not alias.
    EXPECT_EQ(&catalog.serviceFor(0, DeviceId::cuda(4)), &catalog.serviceFor(1, DeviceId::cuda(0)));
    EXPECT_NE(&catalog.serviceFor(0, DeviceId::cuda(4)), &catalog.serviceFor(2, DeviceId::cuda(0)));
    EXPECT_NE(&catalog.serviceFor(0, DeviceId::cuda(4)), &catalog.serviceFor(0, DeviceId::rocm(4)));
    EXPECT_NE(&catalog.serviceFor(0, DeviceId::cpu()), &catalog.serviceFor(1, DeviceId::cpu()));
    EXPECT_THROW(catalog.serviceFor(0, DeviceId::cuda(0)), std::out_of_range);
    EXPECT_THROW(catalog.serviceFor(3, DeviceId::cpu()), std::out_of_range);
    EXPECT_THROW(catalog.serviceFor(-1, DeviceId::cpu()), std::out_of_range);
    EXPECT_THROW(catalog.serviceFor(0, DeviceId::invalid()), std::out_of_range);
    for (const auto &record : catalog.records())
    {
        const auto *value = std::get_if<PlanningProjectionObservations>(&record.observation);
        ASSERT_NE(value, nullptr);
        EXPECT_EQ(value->source.serialize(), matrix.serialize());
        if (value->cpu)
            EXPECT_EQ(value->cpu->execution, cluster.ranks[record.observer.discoveryRank()].cpu_execution);
        EXPECT_EQ(record.observer.physicalNode(), cluster.connectionBetweenRanks(0,
            record.observer.discoveryRank()).destinationNode());
    }
}

/** @test Every projection identity field is authenticated before it can influence a cost. */
TEST(PlanningKernelServiceCatalog, RejectsWrongProjectionOperationRowsGraphsAndRankGeometry)
{
    PlanningGGUFFixture fixture(false, false, GGUFTensorType::BF16);
    PlanningModelSource source(fixture.path());
    const auto matrix = PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ffn_gate.weight", PlanningMatrixRows{64, 129}});
    const PlanningProjectionServicePlan plan(matrix, 7);
    const auto cluster = inventory();
    const auto original = projectionReceipts(cluster, plan);
    for (int defect = 0; defect < 16; ++defect)
    {
        SCOPED_TRACE(defect);
        auto owned = original;
        auto wire = nlohmann::json::parse(owned[1]);
        auto &cpu = wire["observations"][0], &gpu = wire["observations"][1];
        if (defect == 0) wire["source"]["kind"] = "expert";
        if (defect == 1) wire["source"]["prefill_rows"] = 8;
        if (defect == 2) cpu["cpu"]["execution"][0] = 17;
        if (defect == 3) cpu["cpu"]["execution"].push_back(1);
        if (defect == 4) cpu["cpu"]["workers"] = 0;
        if (defect == 5) cpu["phases"][0]["graph_nodes"] = 1;
        if (defect == 6) gpu["phases"][0]["graph_nodes"] = 0;
        if (defect == 7) gpu["phases"][1]["work"] = 3.0 * 6 * 7 * 65 * 256; // FFN work cannot stand in for GEMM.
        if (defect == 8) gpu["phases"][1]["rows"] = 8;
        if (defect == 9) gpu["cpu"] = cpu["cpu"];
        if (defect == 10) cpu["phases"][0]["rows"] = 1.5;
        if (defect == 11) cpu["phases"][0]["seconds"] = 0;
        if (defect == 12) cpu["cpu"]["execution"][2] = uint64_t(UINT32_MAX) + 1;
        if (defect == 13) wire["observations"].erase(0);
        if (defect == 14) cpu["prepared_bytes"] = 1;
        if (defect == 15) cpu["cpu"]["workers"] = 1; // Positive but not the published team.
        const auto text = wire.dump();
        owned[1] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(owned)),
            std::exception);
    }
    PlanningGGUFFixture expert_fixture(true);
    PlanningModelSource expert_source(expert_fixture.path());
    const auto expert = PlanningExpertSamplePlan::resolve(expert_source, request());
    const auto observers = PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{});
    const std::array wrong_operation{observation(observers.front(), expert)};
    EXPECT_THROW(PlanningKernelServiceCatalog::encode(plan, wrong_operation), std::invalid_argument);
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, expert, borrow(original)),
        std::invalid_argument);
}

TEST(PlanningKernelServiceCatalog, PhysicalGPUAliasesDeduplicateButNodesAndVendorsRemainDistinct)
{
    const auto cluster = inventory();
    const auto observers = PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{});
    ASSERT_EQ(observers.size(), 7u); // Three rank workshares and four physical GPUs.
    for (const auto &observer : observers)
        if (observer.device().is_gpu())
        {
            EXPECT_NE(observer.discoveryRank(), 0); // Rank 1 has the observed NUMA affinity.
            EXPECT_EQ(observer.device().ordinal, 0);
            EXPECT_EQ(observer.physicalNode(), observer.discoveryRank() == 2 ? 1 : 0);
        }
    const AutomaticOrchestrationRequest cuda_only({.only_backends = std::vector{DeviceType::CUDA}});
    EXPECT_EQ(PlanningKernelServiceCatalog::observers(cluster, cuda_only).size(), 2u);
    auto reversed_affinity = cluster;
    reversed_affinity.ranks[0].cpu.numa_node = 0;
    reversed_affinity.ranks[1].cpu.numa_node = -1;
    const auto changed = PlanningKernelServiceCatalog::observers(reversed_affinity, cuda_only);
    EXPECT_EQ(changed[0].discoveryRank(), 0);
    EXPECT_EQ(changed[0].device(), DeviceId::cuda(4));
}

TEST(PlanningKernelServiceCatalog, RemoteSpeedAndReceiptOrderCannotChangePhysicalMembershipOrCPUWorkers)
{
    PlanningGGUFFixture fixture(true, false, GGUFTensorType::BF16, 512);
    PlanningModelSource source(fixture.path());
    const auto plan = PlanningExpertSamplePlan::resolve(source, request());
    const auto cluster = inventory();
    const auto owned = receipts(cluster, plan);
    auto ranked = borrow(owned);
    std::reverse(ranked.begin(), ranked.end());
    const auto catalog = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, ranked);
    ASSERT_EQ(catalog.records().size(), 7u);
    for (const auto &record : catalog.records())
    {
        const int rank = record.observer.discoveryRank();
        EXPECT_EQ(record.observer.physicalNode(), cluster.connectionBetweenRanks(0, rank).destinationNode());
        if (record.observer.device().is_cpu())
        {
            EXPECT_EQ(std::get<PlanningCPUExpertObservations>(record.observation).worker_threads, cluster.ranks[rank].cpuWorkerThreads());
            EXPECT_EQ(record.observer.numaNode(), cluster.ranks[rank].cpu.numa_node);
        }
    }
}

TEST(PlanningKernelServiceCatalog, RejectsAmbiguousInventoryBeforeAssigningObservers)
{
    for (int defect = 0; defect < 5; ++defect)
    {
        SCOPED_TRACE(defect);
        auto cluster = inventory();
        if (defect == 0) cluster.ranks[1].rank = 0;
        if (defect == 1) cluster.ranks[1].gpus[0].uuid.clear();
        if (defect == 2) cluster.ranks[1].gpus[0].numa_node = 1;
        if (defect == 3) cluster.ranks[1].gpus[0].local_device_id = -1;
        if (defect == 4)
        {
            auto duplicate = cluster.ranks[1].gpus[0];
            duplicate.uuid = "different-physical-device-same-ordinal";
            cluster.ranks[1].gpus.push_back(duplicate);
        }
        EXPECT_THROW(PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}), std::invalid_argument);
    }
}

TEST(PlanningKernelServiceCatalog, RejectsPartialForeignAndMalformedEvidence)
{
    PlanningGGUFFixture fixture(true, false, GGUFTensorType::F16, 512);
    PlanningModelSource source(fixture.path());
    const auto plan = PlanningExpertSamplePlan::resolve(source, request());
    const auto cluster = inventory();
    const auto original = receipts(cluster, plan);
    for (int defect = 0; defect < 18; ++defect)
    {
        SCOPED_TRACE(defect);
        auto owned = original;
        auto wire = nlohmann::json::parse(owned[1]);
        auto &cpu = wire["observations"][0];
        auto &gpu = wire["observations"][1];
        if (defect == 0) wire["schema"] = "unrecognized";
        if (defect == 1) wire["node"] = 17;
        if (defect == 2) wire["source"] = nlohmann::json::array();
        if (defect == 3) wire["observations"].erase(1);
        if (defect == 4) gpu["ordinal"] = 4;
        if (defect == 5) gpu["phases"][0]["graph_nodes"] = 0;
        if (defect == 6) cpu["cpu"]["workers"] = 0;
        if (defect == 7) cpu["cpu"]["isa"] = 99;
        if (defect == 8) cpu["phases"][0]["rows"] = 8;
        if (defect == 9) cpu["phases"][0]["work"] = 0;
        if (defect == 10) cpu["phases"][0]["seconds"] = -1;
        if (defect == 11) cpu["phases"][0]["phase"] = 3;
        if (defect == 12) cpu["prepared_bytes"] = -1;
        if (defect == 13) cpu["cpu"]["workers"] = 1.5;
        if (defect == 14) cpu["phases"][0]["provenance"] = "  ";
        if (defect == 17) cpu["cpu"]["workers"] = 1; // Cannot borrow a different workshare's rate.
        const auto text = wire.dump();
        owned[1] = {text.begin(), text.end()};
        auto ranked = borrow(owned);
        if (defect == 15) ranked.pop_back();
        if (defect == 16) ranked[2].discovery_rank = 1;
        EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, ranked), std::exception);
    }
    auto ranked = borrow(original);
    std::swap(ranked[0].discovery_rank, ranked[1].discovery_rank);
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, ranked), std::invalid_argument);
}

namespace
{
    /** @return Independently timed synthetic streaming evidence, with an optional controlled slowdown. */
    PlanningKernelServiceCatalog memoryCatalog(const ClusterInventory &cluster, double slowdown = 1)
    {
        auto owned = streamingReceipts(cluster);
        for (auto &bytes : owned)
        {
            auto wire = nlohmann::json::parse(bytes);
            for (auto &entry : wire["observations"])
                entry["seconds"] = entry["seconds"].get<double>() * slowdown;
            const auto text = wire.dump();
            bytes = {text.begin(), text.end()};
        }
        return PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
            PlanningStreamingServicePlan{}, borrow(owned));
    }

    /** @return Compiled whole-source geometry without requiring a runner or device allocation. */
    PlanningWeightOperand sourceOperand(const PlanningModelSource &source, const std::string &name,
        DeviceId device, std::optional<size_t> experts = {})
    {
        const auto &profile = source.metadata().memoryProfile();
        const auto found = std::find_if(profile.tensors.begin(), profile.tensors.end(),
            [&](const auto &tensor) { return tensor.name == name; });
        if (found == profile.tensors.end()) throw std::logic_error("No source operand in fixture");
        const auto role = inferWeightRole(name);
        return {name, role, found->layer_index, found->quant_type,
            PreparedWeightRepresentationContract::resolve(role, found->quant_type),
            WeightShardGeometryResolver(profile, device).resolve(*found, experts)};
    }

    /** @return Canonical serialized identities for comparison, never payload hashes. */
    std::vector<std::vector<uint8_t>> sampleIdentities(const PlanningModelSource &source,
        const PlanningModelMetadata &metadata)
    {
        std::vector<std::vector<uint8_t>> result;
        for (const auto &sample : PlanningWeightServiceModel::samples(metadata))
            result.push_back(std::visit([&](const auto &request) {
                if constexpr (std::is_same_v<std::decay_t<decltype(request)>, PlanningModelSampleRequest>)
                    return PlanningMatrixSamplePlan::resolve(source, request).serialize();
                else return PlanningExpertSamplePlan::resolve(source, request).serialize();
            }, sample));
        return result;
    }
}

TEST(PlanningWeightServiceModel, SamplingIsBoundedByFamiliesAndIndependentOfDirectoryOrder)
{
    for (bool moe : {false, true})
    {
        PlanningGGUFFixture file(moe, true, GGUFTensorType::Q6_K, 512);
        PlanningModelSource source(file.path());
        const auto original = sampleIdentities(source, source.metadata());
        // Dense: F32 and Q6_K projections. MoE: F32 plus one complete Q6_K FFN.
        EXPECT_EQ(original.size(), 2u);
        auto profile = source.metadata().memoryProfile();
        std::reverse(profile.tensors.begin(), profile.tensors.end());
        EXPECT_EQ(sampleIdentities(source, PlanningModelMetadata(profile, 2)), original);
        for (const auto &sample : PlanningWeightServiceModel::samples(source.metadata()))
            std::visit([&](const auto &request) {
                const auto &name = [&]() -> const std::string & {
                    if constexpr (std::is_same_v<std::decay_t<decltype(request)>, PlanningModelSampleRequest>)
                        return request.tensor_name;
                    else return request.gate.tensor_name;
                }();
                EXPECT_EQ(name.find("blk.2."), std::string::npos);
            }, sample);
    }
}

TEST(PlanningWeightServiceModel, PromotedSourcesMeasureAndPriceTheirExecutedFamily)
{
    const test::ScopedGDNProjectionArithmetic arithmetic(GDNProjectionArithmetic::Native);
    PlanningGGUFFixture file(false, false, GGUFTensorType::Q8_0, 256, GGUFTensorType::Q8_0);
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    // A model with no native F32 projection must still acquire F32 service
    // through its actual promoted source, not through invented F32 weights.
    std::erase_if(profile.tensors, [](const auto &tensor) { return tensor.quant_type == "F32"; });
    const auto selected = PlanningWeightServiceModel::samples(PlanningModelMetadata(profile, 2));
    ASSERT_EQ(selected.size(), 2u); // Native Q8 FFN and promoted F32 alpha/beta.
    const auto &request = std::get<PlanningModelSampleRequest>(selected.front());
    EXPECT_EQ(request.tensor_name, "blk.0.ssm_alpha.weight");
    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source, request), 7);
    EXPECT_EQ(plan.matrix().format(), "Q8_0");
    EXPECT_EQ(plan.matrix().executionFormat(), "F32");
    const auto cluster = inventory();
    const auto owned = projectionReceipts(cluster, plan);
    const auto compute = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(owned));
    const PlanningWeightServiceModel service(memoryCatalog(cluster, 1000), {compute});
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto weight = sourceOperand(source, request.tensor_name, device);
        EXPECT_GT(service.projectionSeconds(1, device, weight, 1), 0);
        EXPECT_THROW(service.projectionSeconds(1, device,
            sourceOperand(source, "blk.0.ffn_gate.weight", device), 1), std::invalid_argument);
    }
    // CPU kernels borrow converted storage, so their traffic floor must use
    // its FP32 extent even though the receipt has zero owned packed bytes.
    const auto &shape = plan.matrix().geometry();
    EXPECT_DOUBLE_EQ(service.projectionSeconds(1, DeviceId::cpu(),
        sourceOperand(source, request.tensor_name, DeviceId::cpu()), 1),
        service.memorySeconds(1, DeviceId::cpu(), plan.matrix().executionPayloadBytes() +
            sizeof(float) * (shape.n + shape.k)));
    auto malformed = owned;
    auto wire = nlohmann::json::parse(malformed[0]);
    wire["source"]["execution_format"] = "Q8_0";
    const auto text = wire.dump();
    malformed[0] = {text.begin(), text.end()};
    EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan,
        borrow(malformed)), std::invalid_argument);
}

TEST(PlanningWeightServiceModel, LargeProjectionSamplesBoundRowsWithoutChangingKOrFormat)
{
    PlanningGGUFFixture file(false, false, GGUFTensorType::IQ2_S);
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    // Metadata-only expansion exercises selection policy, not an actual source
    // load; native resolution still validates real requests before any I/O.
    for (auto &tensor : profile.tensors)
        if (tensor.quant_type == "IQ2_S") tensor.elements *= 32;
    const auto selected = PlanningWeightServiceModel::samples(PlanningModelMetadata(profile, 2));
    bool bounded = false;
    for (const auto &sample : selected)
        if (const auto *matrix = std::get_if<PlanningModelSampleRequest>(&sample))
            if (const auto *rows = std::get_if<PlanningMatrixRows>(&matrix->selection))
            {
                EXPECT_EQ(rows->first, 0u);
                EXPECT_EQ(rows->last, 1024u);
                bounded = true;
            }
    EXPECT_TRUE(bounded);
}

namespace
{
    /** @return Codec topology augmented with the physical facts required by production admission. */
    ClusterInventory requestInventory(bool multiple_gpus)
    {
        auto cluster = multiple_gpus ? communicationInventory() : inventory();
        for (auto &rank : cluster.ranks)
        {
            // Request arithmetic chooses rank zero; author its actual bound
            // NUMA rather than borrowing the codec fixture's separate host scope.
            rank.cpu.numa_node = 0;
            rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = size_t{16} << 30;
            for (auto &socket : rank.cpu_socket_info)
            {
                socket.numa_node = rank.cpu.numa_node;
                socket.memory_bytes = size_t{16} << 30;
            }
            for (auto &gpu : rank.gpus)
            {
                gpu.memory_bytes = gpu.free_memory_bytes = size_t{16} << 30;
                gpu.compute_units = 32;
            }
            // Explicit physical topology, independent of the synthetic service
            // receipts: projection fixtures begin with proven absent P2P.
            for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                const auto count = std::count_if(rank.gpus.begin(), rank.gpus.end(),
                    [&](const auto &gpu) { return gpu.type == backend; });
                auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
                auto &width = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
                width = count;
                matrix.assign(static_cast<size_t>(count * count), false);
                for (size_t i = 0; i < static_cast<size_t>(count); ++i) matrix[i * count + i] = true;
            }
        }
        cluster.buildNodeAggregations();
        return cluster;
    }

    /** @brief Perturb only raw measured service; identities, work and format remain authentic fixture inputs. */
    void scaleComputeReceipts(std::vector<std::vector<uint8_t>> &receipts,
        const std::map<DeviceId,double> &factors)
    {
        for(auto &receipt:receipts) {
            auto wire=nlohmann::json::parse(receipt);
            for(auto &observation:wire.at("observations")) {
                const DeviceId device{static_cast<DeviceType>(observation.at("backend").get<int>()),
                    observation.at("ordinal").get<int>()};
                if(const auto found=factors.find(device);found!=factors.end())
                    for(auto &phase:observation.at("phases"))
                        phase["seconds"]=phase.at("seconds").get<double>()*found->second;
            }
            const auto text=wire.dump();receipt.assign(text.begin(),text.end());
        }
    }

    /** @return Complete synthetic source-family evidence for pure request arithmetic tests. */
    PlanningWeightServiceModel requestWeights(const PlanningModelSource &source, const ClusterInventory &cluster,
        const std::map<DeviceId,double> &factors = {})
    {
        std::vector<PlanningKernelServiceCatalog> kernels;
        for (const auto &sample : PlanningWeightServiceModel::samples(source.metadata()))
            std::visit([&](const auto &request) {
                using Request = std::decay_t<decltype(request)>;
                if constexpr (std::is_same_v<Request, PlanningModelSampleRequest>)
                {
                    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source, request), 64);
                    auto encoded = projectionReceipts(cluster, plan);
                    scaleComputeReceipts(encoded,factors);
                    kernels.push_back(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
                        plan, borrow(encoded)));
                }
                else
                {
                    const auto plan = PlanningExpertSamplePlan::resolve(source, request);
                    auto encoded = receipts(cluster, plan);
                    scaleComputeReceipts(encoded,factors);
                    kernels.push_back(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
                        plan, borrow(encoded)));
                }
            }, sample);
        return {memoryCatalog(cluster, .000001), std::move(kernels)};
    }

    /** @return Exact admitted rank-zero candidate; no model payload or accelerator is loaded. */
    AdmittedOrchestrationCandidate requestCandidate(const PlanningModelSource &source, const ClusterInventory &cluster,
        DeviceType backend, OrchestrationStrategy strategy,
        MoERebalanceRuntimeMode movement = MoERebalanceRuntimeMode::Off,
        RoutedExpertComputePolicy routed = RoutedExpertComputePolicy::Apportioned,
        const PlanningHostServiceBudgets &host_services = {})
    {
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 8192;
        request.prefill_max_bucket_size = 64;
        // Costing predicts initial placement without assuming future movement
        // gains; admission must still retain the exact requested maintenance.
        request.moe_rebalance.mode = movement;
        request.routed_expert_compute_policy = routed;
        request.automatic_planning.only_backends = std::vector{backend};
        request.automatic_planning.only_strategies = std::vector{strategy};
        std::optional<AdmittedOrchestrationCandidate> found;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), cluster, [&](auto candidate) {
            if (found || candidate.membership.discoveryRanks() != std::vector<int>{0}) return;
            found = AdmittedOrchestrationCandidate::admit(std::move(candidate), source,
                {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
                 .maximum_cached_prefill_buckets = 8, .retained_host_services = host_services});
        });
        if (!found) throw std::logic_error("No request-cost fixture candidate");
        return std::move(*found);
    }

    /** @return Complete source-free proxy matching the codec fixture's physical observers. */
    PlanningKernelServiceCatalog requestArithmetic(const ClusterInventory &cluster,
        const std::map<DeviceId,double> &factors = {})
    {
        auto encoded = arithmeticReceipts(cluster);
        scaleComputeReceipts(encoded,factors);
        return PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
            PlanningFP32ArithmeticPlan{}, borrow(encoded));
    }
}

/**
 * @test CPU allocation identity is independent of whole-host or bound NUMA affinity.
 *
 * A rank's -1 affinity denotes an observed whole-host CPU resource, not an
 * invalid process CPU allocator. GPU execution additionally requires exact
 * locality, so the unqualified GPU arms are negative controls. Binding NUMA3
 * likewise cannot invent CPU ordinal3. Exercise the actual model-aware compiler and complete BOM on every backend;
 * no model payload or accelerator is loaded by these metadata fixtures.
 */
TEST(PlanningRequestCostModel, CPUAllocatorDoesNotUseNUMAAffinityAsDeviceOrdinal)
{
    PlanningGGUFFixture file(false, false, GGUFTensorType::F32);
    PlanningModelSource source(file.path());
    for (const int affinity : {-1, 3})
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(::testing::Message() << affinity << '/' << int(backend));
            auto observed = requestInventory(false);
            for (auto &rank : observed.ranks)
            {
                rank.cpu.numa_node = affinity;
                rank.numa_nodes = affinity < 0 ? 1 : 4;
                for (auto &socket : rank.cpu_socket_info) socket.numa_node = affinity < 0 ? 0 : affinity;
                if (affinity >= 0)
                    for (auto &gpu : rank.gpus) gpu.numa_node = affinity;
            }
            observed.buildNodeAggregations();
            if (affinity < 0 && backend != DeviceType::CPU)
            {
                // Whole-host CPU ownership cannot authorize any GPU executor.
                EXPECT_THROW(requestCandidate(source, observed, backend,
                    OrchestrationStrategy::SingleDevice), std::logic_error);
                continue;
            }
            std::optional<AdmittedOrchestrationCandidate> admitted;
            EXPECT_NO_THROW(admitted.emplace(requestCandidate(source, observed, backend,
                OrchestrationStrategy::SingleDevice)));
            if (!admitted) continue;
            ASSERT_EQ(admitted->rankPlans().size(), 1u);
            EXPECT_EQ(admitted->rankPlans().front().numa_node, affinity);
            ASSERT_EQ(admitted->devicePlans().size(), 1u);
            const auto &device = admitted->devicePlans().front();
            if (backend == DeviceType::CPU)
                EXPECT_EQ(device.device, DeviceId::cpu());
            else
            {
                ASSERT_TRUE(device.associated_host_memory);
                EXPECT_EQ(device.associated_host_memory->device, DeviceId::cpu());
                EXPECT_EQ(device.associated_host_memory->world_rank, 0);
            }
            const auto &physical = admitted->physicalAdmission().plan();
            EXPECT_NE(physical.find({0, DeviceId::cpu()}), nullptr);
            if (affinity == 3)
                EXPECT_EQ(physical.find({0, DeviceId{DeviceType::CPU, 3}}), nullptr);
        }
}

TEST(PlanningRequestCostModel, SingleDeviceHasNoLinksAndLiveContextCostsMoreAcrossBackendsAndFormats)
{
    const auto cluster = requestInventory(false);
    for (bool moe : {false, true})
        for (auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16, GGUFTensorType::Q8_0})
        {
            PlanningGGUFFixture file(moe, true, format);
            PlanningModelSource source(file.path());
            const PlanningRequestCostModel costs(source.metadata(), cluster, requestWeights(source, cluster),
                requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}));
            for (auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            {
                SCOPED_TRACE(std::to_string(int(backend)) + "/" + std::to_string(int(format)) + "/" + std::to_string(moe));
                const auto candidate = requestCandidate(source, cluster, backend, OrchestrationStrategy::SingleDevice);
                const auto short_context = costs.evaluate(candidate, {64, 16});
                const auto long_context = costs.evaluate(candidate, {2048, 16});
                const auto tail = costs.evaluate(candidate, {65, 16});
                EXPECT_GT(long_context.decodeSecondsPerToken(), short_context.decodeSecondsPerToken());
                EXPECT_GT(long_context.prefillSeconds(), tail.prefillSeconds());
                EXPECT_GT(tail.prefillSeconds(), short_context.prefillSeconds());
                EXPECT_NE(short_context.evidence().find("prefill_interconnect_s=0"), std::string::npos);
                EXPECT_NE(short_context.evidence().find("mean_decode_interconnect_s=0"), std::string::npos);
                EXPECT_NE(short_context.evidence().find("no assumed MTP acceptance"), std::string::npos);
                EXPECT_THROW(costs.evaluate(candidate, {8192, 1}), std::invalid_argument);
            }
        }
}

TEST(PlanningRequestCostModel, TPRequiresNativeEvidenceAndPricesTheTypedDecodePolicy)
{
    const auto cluster = requestInventory(true);
    constexpr std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 256, 64, precision);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    const auto observed = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(encoded));
    for (bool moe : {false, true})
    {
        PlanningGGUFFixture file(moe);
        PlanningModelSource source(file.path());
        const PlanningRequestCostModel costs(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, observed.native, {}, observed.host_device));
        const PlanningRequestCostModel missing(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}, observed.host_device));
        // This inventory explicitly has no direct peers. Retaining native
        // collective evidence alone cannot price the physical host-staged edge.
        const PlanningRequestCostModel missing_host(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, observed.native, {}));
        for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            const auto candidate = requestCandidate(source, cluster, backend, OrchestrationStrategy::TensorParallel);
            const auto cost = costs.evaluate(candidate, {64, 16});
            const bool replicated_decode = std::any_of(candidate.devicePlans().begin(), candidate.devicePlans().end(),
                [](const auto &device) {
                    return std::find(device.additional_weight_sets.begin(), device.additional_weight_sets.end(),
                        AdditionalPersistentWeightSet::ReplicatedDenseDecode) != device.additional_weight_sets.end();
                });
            EXPECT_GT(cost.requestSeconds(), 0);
            // TP prefill always consumes the measured native collective.  The
            // homogeneous MoE default deliberately switches decode to its
            // admitted replicated weight view, so only that typed policy has
            // zero decode-interconnect cost.
            EXPECT_EQ(cost.evidence().find("prefill_interconnect_s=0;"), std::string::npos);
            if (replicated_decode)
                EXPECT_NE(cost.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
            else
                EXPECT_EQ(cost.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
            EXPECT_NE(cost.evidence().find("measured native allreduce fixed latency"), std::string::npos);
            EXPECT_EQ(cost.evidence().find("point-to-point traffic"), std::string::npos);
            EXPECT_THROW(missing.evaluate(candidate, {64, 16}), std::invalid_argument);
            EXPECT_THROW(missing_host.evaluate(candidate, {64, 16}), std::invalid_argument);
        }
    }
}

/** @test Real candidate scoring consumes directed measurements and preserves workload-dependent placement. */
TEST(PlanningRequestCostModel, MeasuredPairsOverrideSwitchPreferenceAcrossWorkloads)
{
    PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        // One actual rank owns three GPUs on its own NUMA node. The first
        // two share a switch; the third belongs to another local switch.
        // Fabric times below are a decision oracle, not performance evidence.
        auto cluster = requestInventory(true);
        auto rank = cluster.ranks[1];
        rank.rank = rank.local_rank = rank.node_id = 0;
        std::erase_if(rank.gpus, [&](const auto &gpu) { return gpu.type != backend; });
        ASSERT_EQ(rank.gpus.size(), 3u);
        std::sort(rank.gpus.begin(), rank.gpus.end(), [](const auto &a, const auto &b) {
            return a.local_device_id < b.local_device_id;
        });
        std::vector<int> ordinals;
        for (size_t i = 0; i < rank.gpus.size(); ++i)
        {
            auto &gpu = rank.gpus[i];
            gpu.pcie_upstream_bdfs = {i < 2 ? "0000:86:00.0" : "0000:af:00.0", "0000:80:00.0"};
            ordinals.push_back(gpu.local_device_id);
        }
        rank.p2p_cuda_count = backend == DeviceType::CUDA ? 3 : 0;
        rank.p2p_rocm_count = backend == DeviceType::ROCm ? 3 : 0;
        rank.p2p_cuda.assign(backend == DeviceType::CUDA ? 9 : 0, true);
        rank.p2p_rocm.assign(backend == DeviceType::ROCm ? 9 : 0, true);
        cluster.ranks = {std::move(rank)}; cluster.world_size = 1; cluster.buildNodeAggregations();
        OrchestrationConfig request;
        request.model_path = source.path(); request.max_seq_len = 8192;
        request.prefill_max_bucket_size = 64;
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.automatic_planning.only_backends = std::vector{backend};
        request.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::TensorParallel};
        request.automatic_planning.device_counts = {{backend, 2}};
        const auto constraints = std::get<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(request));
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, constraints, 256, 64, precisions);
        const auto encoded = nativeReceipts(plan, 1);
        auto original = PlanningCommunicationService::acceptLocal(plan, 1, borrow(encoded)).native;
        // This ranking fixture observes actual payload-dependent excess above
        // the native small-message floor; a flat curve cannot price subgroup improvement.
        for(auto &receipt:original)for(auto &phase:receipt.observation.phases)
            for(auto &endpoint:phase.endpoints)endpoint.seconds_per_collective*=phase.rows;
        const auto weights = requestWeights(source, cluster);
        const auto arithmetic = requestArithmetic(cluster);
        for (const bool ties : {true, false})
        {
            auto measured = original;
            for (auto &receipt : measured)
                for (auto &edge : receipt.observation.pair_sampling.directed_pairs)
                {
                    const auto low = std::min(edge.source_index, edge.destination_index);
                    const auto high = std::max(edge.source_index, edge.destination_index);
                    const bool bulk = edge.payload_bytes == receipt.sample.request.activationBytes();
                    const bool quick = high == 2 && low == (bulk ? 1u : 0u);
                    edge.seconds_per_transfer.assign(PlanningLocalTPPairSampling::kSamples,
                        ties ? .00001 : quick ? .000001 : .001);
                }
            const PlanningRequestCostModel costs(source.metadata(), cluster, weights, arithmetic,
                PlanningCommunicationCost(cluster, measured, {}));
            for (const auto workload : {OrchestrationPlanningWorkload{1, 256}, OrchestrationPlanningWorkload{32, 256}, OrchestrationPlanningWorkload{256, 1}})
            {
                const auto selected = AutomaticOrchestrationPlanner::select(request, source, cluster,
                    {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
                     .maximum_cached_prefill_buckets = 8}, workload,
                    [&](const auto &candidate, const auto &work) { return costs.evaluate(candidate, work); });
                std::set<int> actual;
                for (const auto &device : selected.candidate().devicePlans()) actual.insert(device.device.ordinal);
                const std::set<int> expected = (ties || workload.promptTokens()==1) ? std::set{ordinals[0], ordinals[1]} :
                    std::set{ordinals[workload.generationTokens() > 1 ? 0 : 1], ordinals[2]};
                EXPECT_EQ(actual, expected) << "backend=" << int(backend) << ", ties=" << ties;
                EXPECT_NE(selected.cost().evidence().find("measured native allreduce fixed latency"), std::string::npos);
            }
        }
    }
}

namespace {
    /** @return Fully identified homogeneous compute devices, with explicit immutable peer facts. */
    ClusterInventory medianComputeInventory(DeviceType backend,size_t count)
    {
        auto cluster=requestInventory(true);auto rank=cluster.ranks[1];rank.rank=rank.local_rank=rank.node_id=0;
        std::erase_if(rank.gpus,[&](const auto &gpu){return gpu.type!=backend;});
        const auto prototype=rank.gpus.front();rank.gpus.clear();
        for(size_t i=0;i<count;++i){auto gpu=prototype;gpu.local_device_id=int(i);
            gpu.uuid="median-fixture-"+std::to_string(int(backend))+"-"+std::to_string(i);
            gpu.name="Recorded homogeneous GPU";gpu.native_isa=backend==DeviceType::ROCm?"gfx906":"sm_80";
            gpu.compute_capability_major=8;gpu.compute_capability_minor=0;gpu.compute_units=32;
            gpu.pcie_upstream_bdfs={i<2?"0000:86:00.0":"0000:af:00.0","0000:80:00.0"};rank.gpus.push_back(gpu);}
        rank.p2p_cuda_count=backend==DeviceType::CUDA?int(count):0;rank.p2p_rocm_count=backend==DeviceType::ROCm?int(count):0;
        rank.p2p_cuda.assign(backend==DeviceType::CUDA?count*count:0,true);
        rank.p2p_rocm.assign(backend==DeviceType::ROCm?count*count:0,true);
        cluster.ranks={std::move(rank)};cluster.world_size=1;cluster.buildNodeAggregations();return cluster;
    }
    /** @return Exact admitted singleton at the requested original ordinal, without a substitute topology. */
    AdmittedOrchestrationCandidate medianComputeCandidate(const PlanningModelSource &source,const ClusterInventory &cluster,DeviceType backend,int ordinal,int discovery_rank=0)
    {
        OrchestrationConfig request;request.model_path=source.path();request.max_seq_len=8192;
        request.prefill_max_bucket_size=64;request.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        request.automatic_planning.only_backends=std::vector{backend};
        request.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::SingleDevice};
        request.automatic_planning.device_counts={{backend,1}};
        std::optional<AdmittedOrchestrationCandidate> selected;
        visitAutomaticOrchestrationCandidates(request,source.metadata(),cluster,[&](auto proposal){
            if(selected || proposal.membership.discoveryRanks()!=std::vector<int>{discovery_rank})return;
            auto admitted=AdmittedOrchestrationCandidate::admit(std::move(proposal),source,
                {.prefill_bucket_rows={32,64},.minimum_prefill_sequence_rows=1,.maximum_cached_prefill_buckets=8});
            if(admitted.devicePlans().size()==1 && admitted.devicePlans()[0].device.ordinal==ordinal)selected.emplace(std::move(admitted));});
        if(!selected)throw std::logic_error("No exact median compute fixture candidate");return std::move(*selected);
    }
}

/** @test Same-class raw timing permutations cannot reshuffle compute placement; actual faster peer service decides. */
TEST(PlanningRequestCostModel, SameClassMedianComputeKeepsMeasuredFastestGroup)
{
    PlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        const auto cluster=medianComputeInventory(backend,3);
        OrchestrationConfig request;request.model_path=source.path();request.max_seq_len=8192;request.prefill_max_bucket_size=64;
        request.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;request.automatic_planning.only_backends=std::vector{backend};
        request.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::TensorParallel};request.automatic_planning.device_counts={{backend,2}};
        const auto policy=std::get<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(request));
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        const auto plan=PlanningCommunicationSamplePlan::resolve(cluster,policy,256,64,precisions);
        const auto wire=nativeReceipts(plan,1);auto native=PlanningCommunicationService::acceptLocal(plan,1,borrow(wire)).native;
        // This ranking fixture observes actual payload-dependent excess above
        // the native small-message floor; a flat curve cannot price subgroup improvement.
        for(auto &receipt:native)for(auto &phase:receipt.observation.phases)
            for(auto &endpoint:phase.endpoints)endpoint.seconds_per_collective*=phase.rows;
        for(const bool cross_fast:{false,true}){
            auto links=native;for(auto &receipt:links)for(auto &edge:receipt.observation.pair_sampling.directed_pairs){
                const auto low=std::min(edge.source_index,edge.destination_index),high=std::max(edge.source_index,edge.destination_index);
                const bool fastest=cross_fast?(low==0 && high==2):(low==0 && high==1);
                edge.seconds_per_transfer.assign(PlanningLocalTPPairSampling::kSamples,fastest?.00003:.00008);}
            std::optional<double> baseline;
            for(const bool reverse:{false,true}){
                const std::map<DeviceId,double> factors{{{backend,0},reverse?100.:.01},{{backend,1},1.},{{backend,2},reverse?.01:100.}};
                const auto weights=requestWeights(source,cluster,factors);const auto arithmetic=requestArithmetic(cluster,factors);
                const auto raw=std::get<PlanningFP32ArithmeticObservations>(arithmetic.serviceFor(0,{backend,0}).observation).phases[0].service.elapsedSeconds();
                EXPECT_DOUBLE_EQ(raw,.0002*factors.at(DeviceId{backend,0}));
                const PlanningRequestCostModel costs(source.metadata(),cluster,weights,arithmetic,PlanningCommunicationCost(cluster,links,{}));
                const auto selected=AutomaticOrchestrationPlanner::select(request,source,cluster,
                    {.prefill_bucket_rows={32,64},.minimum_prefill_sequence_rows=1,.maximum_cached_prefill_buckets=8},
                    {256,256},[&](const auto &candidate,const auto &work){return costs.evaluate(candidate,work);});
                std::set<int> actual;for(const auto &device:selected.candidate().devicePlans())actual.insert(device.device.ordinal);
                EXPECT_EQ(actual,(cross_fast?std::set<int>{0,2}:std::set<int>{0,1}));
                if(baseline)EXPECT_DOUBLE_EQ(selected.cost().requestSeconds(),*baseline);else baseline=selected.cost().requestSeconds();
                EXPECT_DOUBLE_EQ(std::get<PlanningFP32ArithmeticObservations>(arithmetic.serviceFor(0,{backend,0}).observation).phases[0].service.elapsedSeconds(),raw);
            }
        }
    }
}

/** @test Singleton, odd and larger cohorts retain one class median without changing original raw catalogs. */
TEST(PlanningRequestCostModel, SameClassMedianComputeSupportsArbitraryCohortSizes)
{
    PlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const size_t count:{1u,3u,9u}){
        const auto cluster=medianComputeInventory(DeviceType::ROCm,count);const auto candidate=medianComputeCandidate(source,cluster,DeviceType::ROCm,0);
        std::optional<double> baseline;
        for(const bool reverse:{false,true}){
            std::map<DeviceId,double> factors;for(size_t i=0;i<count;++i)factors.emplace(DeviceId::rocm(int(i)),double((reverse?count-1-i:i)+1));
            const auto weights=requestWeights(source,cluster,factors);const auto arithmetic=requestArithmetic(cluster,factors);
            const PlanningRequestCostModel costs(source.metadata(),cluster,weights,arithmetic,PlanningCommunicationCost(cluster,{},{}));
            const auto estimate=costs.evaluate(candidate,{256,256});
            if(baseline)EXPECT_DOUBLE_EQ(estimate.requestSeconds(),*baseline);else baseline=estimate.requestSeconds();
        }
    }
}

/** @test A permitted compute receipt alias resolves its original observer before class membership or ordinal lookup. */
TEST(PlanningRequestCostModel, SameClassMedianComputeResolvesOriginalObserverAliases)
{
    PlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        auto cluster=requestInventory(true);
        for(auto &rank:cluster.ranks)for(auto &gpu:rank.gpus){
            gpu.name="Recorded alias GPU";gpu.native_isa=gpu.type==DeviceType::CUDA?"sm_80":"gfx906";
            gpu.compute_capability_major=8;gpu.compute_capability_minor=0;}
        cluster.buildNodeAggregations();
        const auto canonical=medianComputeCandidate(source,cluster,backend,4,0);
        const auto alias=medianComputeCandidate(source,cluster,backend,0,1);
        const std::map<DeviceId,double> factors{{{backend,4},100.},{{backend,0},.01},{{backend,10},1.},{{backend,20},2.}};
        const PlanningRequestCostModel costs(source.metadata(),cluster,requestWeights(source,cluster,factors),
            requestArithmetic(cluster,factors),PlanningCommunicationCost(cluster,{},{}));
        EXPECT_DOUBLE_EQ(costs.evaluate(canonical,{256,256}).requestSeconds(),costs.evaluate(alias,{256,256}).requestSeconds());
    }
}

/** @test Different or incomplete physical compute facts cannot pool independent endpoint evidence. */
TEST(PlanningRequestCostModel, SameClassMedianComputePreservesHardwareDistinctions)
{
    PlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(int defect=0;defect<12;++defect){
        SCOPED_TRACE(int(backend));
        SCOPED_TRACE(defect);auto cluster=medianComputeInventory(backend,3);
        auto &gpu=cluster.ranks[0].gpus[1];
        if(defect==0)gpu.name+=" other";
        if(defect==1)gpu.native_isa+="other";
        if(defect==2)++gpu.compute_units;
        if(defect==3)++gpu.compute_capability_major;
        if(defect==4)++gpu.compute_capability_minor;
        if(defect==5)gpu.memory_bytes+=4096;
        if(defect==6)gpu.last_level_cache_bytes+=4096;
        if(defect==7)gpu.tflops_fp16=123.;
        if(defect==8)gpu.tflops_int8=123.;
        if(defect==9)gpu.memory_bandwidth_gbps=123.;
        if(defect==10)for(auto &device:cluster.ranks[0].gpus)device.name.clear();
        if(defect==11)for(auto &device:cluster.ranks[0].gpus)device.native_isa.clear();
        cluster.buildNodeAggregations();const auto candidate=medianComputeCandidate(source,cluster,backend,0);
        std::array<double,2> predictions{};
        for(size_t reverse=0;reverse<2;++reverse){
            const std::map<DeviceId,double> factors{{{backend,0},reverse?100.:.01},{{backend,1},reverse?.01:100.},{{backend,2},1.}};
            const PlanningRequestCostModel costs(source.metadata(),cluster,requestWeights(source,cluster,factors),requestArithmetic(cluster,factors),PlanningCommunicationCost(cluster,{},{}));
            predictions[reverse]=costs.evaluate(candidate,{256,256}).requestSeconds();
        }
        if(defect==11 && backend==DeviceType::CUDA)EXPECT_DOUBLE_EQ(predictions[0],predictions[1]);
        else EXPECT_NE(predictions[0],predictions[1]);
    }
}

/** @test A supported GPU/CPU expert epoch requires measured mapped service, regardless of GPU vendor. */
TEST(PlanningRequestCostModel, MappedOverlayDoesNotSubstituteNativePeerMeasurements)
{
    auto cluster = requestInventory(true);
    cluster.world_size = 1; cluster.ranks.resize(1); cluster.buildNodeAggregations();
    PlanningGGUFFixture file(true);
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        const AutomaticOrchestrationRequest constraints({.only_backends = {{backend, DeviceType::CPU}},
            .only_strategies = {{OrchestrationStrategy::ExpertOverlay}}});
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, constraints, 256, 64, precisions);
        const auto encoded = nativeReceipts(plan, 1);
        const auto observed = PlanningCommunicationService::acceptLocal(plan, 1, borrow(encoded));
        OrchestrationConfig request;
        request.model_path = source.path(); request.max_seq_len = 8192; request.prefill_max_bucket_size = 64;
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.automatic_planning = constraints.options();
        std::optional<AdmittedOrchestrationCandidate> selected;
        size_t proposals = 0;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), cluster, [&](auto proposal) {
            ++proposals;
            if (selected || proposal.membership.discoveryRanks() != std::vector<int>{0}) return;
            const auto &overlay = proposal.config.moe_routed_expert_plan;
            if (!overlay || overlay->domains.size() != 2 || overlay->routed_tiers.size() != 2 ||
                overlay->domains[0].participants.size() != 1 || overlay->domains[1].participants.size() != 1 ||
                overlay->domains[0].participants[0].device_type != backend ||
                !overlay->domains[1].participants[0].isCPU()) return;
            // Keep a real positive CPU expert cohort: unconstrained GPU capacity
            // would otherwise consume all experts and leave no mapped epoch.
            proposal.config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(*overlay);
            proposal.config.moe_routed_expert_plan->routed_tiers[0].max_experts_per_layer = 4;
            selected.emplace(AdmittedOrchestrationCandidate::admit(std::move(proposal), source,
                {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
                 .maximum_cached_prefill_buckets = 8,
                 .retained_host_services = planningHostServiceBudgets(cluster.world_size)}));
        });
        ASSERT_TRUE(selected) << "Expected one GPU continuation plus a host-owned CPU expert tier; proposals=" << proposals;
        const auto &candidate = *selected;
        ASSERT_EQ(candidate.rankPlans().size(), 1u);
        const auto &runtime = candidate.rankPlans().front().runtime;
        ASSERT_TRUE(runtime.retained_host_services);
        EXPECT_EQ(runtime.retained_host_services->worldRank(), 0);
        EXPECT_TRUE(runtime.retained_host_services->contains(RetainedHostServiceInstance{
            .worldRank = 0,
            .consumer = candidate.rankPlans().front().primary_device.toLocalDeviceId(),
            .role = RetainedHostServiceRole::ExecutorParentTicket}));
        ASSERT_TRUE(candidate.overlayCapacity());
        ASSERT_EQ(candidate.overlayCapacity()->tiers.size(), 2u);
        for (const auto &tier : candidate.overlayCapacity()->tiers)
        {
            ASSERT_FALSE(tier.live_experts_per_layer.empty());
            for (const int experts : tier.live_experts_per_layer) EXPECT_EQ(experts, 4);
        }
        const PlanningRequestCostModel mapped(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}, observed.host_device));
        const auto estimate = mapped.evaluate(candidate, {64, 16});
        EXPECT_NE(estimate.evidence().find("node-local activation epochs use directed mapped-byte primitives"), std::string::npos);
        EXPECT_EQ(estimate.evidence().find("native grouped-P2P INT8 directed placement proxy"), std::string::npos);
        const PlanningRequestCostModel wrong(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, observed.native, {}));
        EXPECT_THROW(wrong.evaluate(candidate, {64, 16}), std::invalid_argument);
    }
}

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @brief Auto admission preserves Dynamic policy for every floating source family. */
TEST(PlanningRequestCostModel, FloatingHomogeneousMoETPAdmitsDynamicWithoutChangingPolicy)
{
    const auto cluster = requestInventory(true);
    for (const auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16})
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(::testing::Message() << int(format) << "/" << int(backend));
            PlanningGGUFFixture file(true, true, format);
            PlanningModelSource source(file.path());
            const auto candidate = requestCandidate(source, cluster, backend,
                OrchestrationStrategy::TensorParallel, MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(candidate.config().moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
        }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

/** @brief Four/eight physical owners use the same scoped request-cost composition. */
class PipelineRequestCost : public ::testing::TestWithParam<int> {};

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test A later PP expert stage must use its own root, participants and quotas. */
TEST_P(PipelineRequestCost, HybridMoEUsesStageScopedEvidence)
{
    auto cluster = requestInventory(true);
    cluster.world_size = 1;
    cluster.ranks.resize(1);
    auto &rank = cluster.ranks.front();
    // Every participant retains its configured RAM tier. This observation must
    // cover the complete transaction, including eight independently owned tiers.
    rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = size_t{256} << 30;
    for (auto &socket : rank.cpu_socket_info) socket.memory_bytes = size_t{256} << 30;
    const int width = GetParam() / 2;
    const auto templates = rank.gpus;
    rank.gpus.clear();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto base = *std::find_if(templates.begin(), templates.end(),
            [&](const auto &gpu) { return gpu.type == backend; });
        for (int index = 0; index < width; ++index)
        {
            auto gpu = base;
            gpu.local_device_id = 4 + index * 10;
            gpu.uuid = std::string(deviceTypeToString(backend)) + "-cost-owner-" + std::to_string(index);
            rank.gpus.push_back(std::move(gpu));
        }
        auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        (backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count) = width;
        matrix.assign(size_t(width * width), false);
        for (int index = 0; index < width; ++index) matrix[index * width + index] = true;
    }
    cluster.buildNodeAggregations();
    PlanningGGUFFixture file(true, true, GGUFTensorType::IQ4_XS);
    PlanningModelSource source(file.path());
    constexpr std::array precisions{PlanningAllreducePrecision::FP32};
    const auto sample = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{},
        256, 64, precisions, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
    const auto encoded = nativeReceipts(sample, 1);
    const auto observed = PlanningCommunicationService::acceptLocal(sample, 1, borrow(encoded));
    const auto weights = requestWeights(source, cluster);
    const auto arithmetic = requestArithmetic(cluster);
    const PlanningRequestCostModel costs(source.metadata(), cluster, weights, arithmetic,
        PlanningCommunicationCost(cluster, observed.native, {}, observed.host_device));
    const PlanningRequestCostModel missing_links(source.metadata(), cluster, weights, arithmetic,
        PlanningCommunicationCost(cluster, observed.native, {}));
    for (const bool reverse : {false, true})
    {
        const auto domain = [width](std::string name, bool rocm) {
            std::string members;
            for (int index = width - 1; index >= 0; --index)
                members += (members.empty() ? "" : ",") + std::string(rocm ? "rocm:" : "cuda:") + std::to_string(4 + index * 10);
            return name + "=" + members + ";scope=rank_local;backend=" + (rocm ? "rccl" : "nccl") + ";owner=0";
        };
        std::vector<std::string> args{"llaminar2", "--define-domain", domain("head", reverse),
            "--define-domain", domain("tail", !reverse), "--pp-stage", "0=head:0-0", "--pp-stage", "1=tail:1-1"};
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.model_path = source.path();
        config.max_seq_len = 8192;
        config.prefill_max_bucket_size = 64;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.routed_expert_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        const auto admitted = AdmittedOrchestrationCandidate::admit(
            {OrchestrationStrategy::PipelineParallel, ExecutionRankMembership(cluster, {0}), std::move(config)}, source,
            {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1, .maximum_cached_prefill_buckets = 8});
        const auto estimate = costs.evaluate(admitted, {64, 16});
        EXPECT_GT(estimate.requestSeconds(), 0);
        EXPECT_NE(estimate.evidence().find("all-expert down slices"), std::string::npos);
        EXPECT_EQ(estimate.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
        EXPECT_THROW(missing_links.evaluate(admitted, {64, 16}), std::invalid_argument);
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Every test of this suite requires a compiled GPU backend family.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
INSTANTIATE_TEST_SUITE_P(DeviceCounts, PipelineRequestCost, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });
#endif

TEST(PlanningWeightServiceModel, InterpolatesInvocationTimeAndNeverReusesCachedComputeAsBandwidth)
{
    PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto cluster = inventory();
    const auto matrix = PlanningMatrixSamplePlan::resolve(source, {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}});
    const PlanningProjectionServicePlan plan(matrix, 7);
    const auto owned = projectionReceipts(cluster, plan);
    const auto compute = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(owned));
    const PlanningWeightServiceModel fast(memoryCatalog(cluster, .000001), {compute});
    const PlanningWeightServiceModel slow(memoryCatalog(cluster, 1000000), {compute});
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto weight = sourceOperand(source, matrix.request().tensor_name, device);
        // The synthetic observation timed three launches in .0002 seconds,
        // at both endpoints. Every interpolation point must retain that latency.
        for (size_t rows = 1; rows <= 7; ++rows)
            EXPECT_NEAR(fast.projectionSeconds(1, device, weight, rows), .0002 / 3, 1e-12);
        EXPECT_NEAR(fast.projectionSeconds(1, device, weight, 14), 2 * .0002 / 3, 1e-12);
        EXPECT_GT(slow.projectionSeconds(1, device, weight, 1), fast.projectionSeconds(1, device, weight, 1) * 100);
        EXPECT_THROW(fast.projectionSeconds(1, device, weight, 0), std::invalid_argument);
        auto absent = weight;
        absent.source_format = "Q6_K";
        EXPECT_THROW(fast.projectionSeconds(1, device, absent, 1), std::invalid_argument);
        // A legitimately promoted operand consumes F32 service, not its Q8 source label.
        absent.source_format = "Q8_0";
        absent.representation = ModelPreparedWeightRepresentation::FP32;
        EXPECT_DOUBLE_EQ(fast.projectionSeconds(1, device, absent, 1), fast.projectionSeconds(1, device, weight, 1));
    }
    const auto weight = sourceOperand(source, matrix.request().tensor_name, DeviceId::cuda(0));
    EXPECT_DOUBLE_EQ(fast.projectionSeconds(0, DeviceId::cuda(4), weight, 1),
        fast.projectionSeconds(1, DeviceId::cuda(0), weight, 1));
    EXPECT_THROW(fast.projectionSeconds(99, DeviceId::cpu(), weight, 1), std::out_of_range);
}

TEST(PlanningWeightServiceModel, ExpertGroupingAndReplicationPreserveAllNativeFormats)
{
    using T = GGUFTensorType;
    const std::vector formats{T::F32, T::F16, T::BF16, T::Q4_0, T::Q4_1, T::Q5_0, T::Q5_1, T::Q8_0,
        T::Q2_K, T::Q3_K, T::Q4_K, T::Q5_K, T::Q6_K, T::Q8_K, T::IQ1_S, T::IQ1_M,
        T::IQ2_XXS, T::IQ2_XS, T::IQ2_S, T::IQ3_XXS, T::IQ3_S, T::IQ4_NL, T::IQ4_XS};
    const auto cluster = inventory();
    for (auto format : formats)
    {
        SCOPED_TRACE(static_cast<int>(format));
        PlanningGGUFFixture file(true, true, format, 512);
        PlanningModelSource source(file.path());
        const auto plan = PlanningExpertSamplePlan::resolve(source, request());
        const auto owned = receipts(cluster, plan);
        const auto kernel = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(owned));
        const PlanningWeightServiceModel service(memoryCatalog(cluster, .000001), {kernel});
        const auto selected = PlanningWeightServiceModel::samples(source.metadata());
        EXPECT_EQ(selected.size(), 2u);
        EXPECT_TRUE(std::holds_alternative<PlanningExpertSampleRequest>(selected.back()));
        for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            PlanningRoutedExpertWeightWork work{0, 8, 2,
                {sourceOperand(source, plan.request().gate.tensor_name, device, 1),
                 sourceOperand(source, plan.request().up.tensor_name, device, 1),
                 sourceOperand(source, plan.request().down.tensor_name, device, 1)},
                {{PlanningExpertExecution::OwnedExperts, 8}}};
            const auto full = service.expertSeconds(1, device, work, 1);
            EXPECT_GT(full, 0);
            EXPECT_DOUBLE_EQ(service.expertSeconds(1, device, work, 0), 0);
            work.execution_shares = {{PlanningExpertExecution::OwnedExperts, 4}};
            EXPECT_NEAR(service.expertSeconds(1, device, work, 1), full / 2, 1e-12);
            work.execution_shares = {{PlanningExpertExecution::ReplicatedExperts, 8}};
            EXPECT_DOUBLE_EQ(service.expertSeconds(1, device, work, 1), full);
            const auto grouped = service.expertSeconds(1, device, work, 64);
            EXPECT_GT(grouped, full);
            EXPECT_LT(grouped, full * 64); // Repeated routes group; they do not reread one FFN per token.
            work.execution_shares = {{PlanningExpertExecution::BalancedReplicaAssignment, 8, 2}};
            EXPECT_NEAR(service.expertSeconds(1, device, work, 64), grouped / 2, 1e-12);
            work.execution_shares = {{PlanningExpertExecution::OwnedExperts, 0}};
            EXPECT_DOUBLE_EQ(service.expertSeconds(1, device, work, 64), 0);
        }
        // Ordinary projections have a distinct prepared layout and service
        // family. Exercise every source format there too, not just on experts.
        PlanningGGUFFixture dense_file(false, false, format);
        PlanningModelSource dense_source(dense_file.path());
        const PlanningProjectionServicePlan projection(PlanningMatrixSamplePlan::resolve(dense_source,
            {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}}), 7);
        const auto projected = projectionReceipts(cluster, projection);
        const auto ordinary = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, projection, borrow(projected));
        const PlanningWeightServiceModel ordinary_service(memoryCatalog(cluster, .000001), {ordinary});
        for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            const auto weight = sourceOperand(dense_source, projection.matrix().request().tensor_name, device);
            for (const size_t rows : {1u, 2u, 7u, 64u, 8192u})
                EXPECT_GT(ordinary_service.projectionSeconds(1, device, weight, rows), 0);
        }
    }
}

/** @brief Zero movable quota never erases the fixed all-expert down population. */
TEST(PlanningWeightServiceModel, ProjectionWorkChargesDownAtZeroOwnersAllNativeFormats)
{
    using T = GGUFTensorType;
    const std::vector formats{T::F32, T::F16, T::BF16, T::Q4_0, T::Q4_1, T::Q5_0, T::Q5_1, T::Q8_0,
        T::Q2_K, T::Q3_K, T::Q4_K, T::Q5_K, T::Q6_K, T::Q8_K, T::IQ1_S, T::IQ1_M,
        T::IQ2_XXS, T::IQ2_XS, T::IQ2_S, T::IQ3_XXS, T::IQ3_S, T::IQ4_NL, T::IQ4_XS};
    const auto cluster = inventory();
    for (const auto format : formats)
    {
        PlanningGGUFFixture file(true, true, format, 512);
        PlanningModelSource source(file.path());
        const auto plan = PlanningExpertSamplePlan::resolve(source, request());
        const auto encoded = receipts(cluster, plan);
        const auto catalog = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(encoded));
        // Make the arithmetic witness the limiting basis, so exact operation
        // fractions are testable independently of format-dependent streaming.
        const PlanningWeightServiceModel service(memoryCatalog(cluster, 1e-12), {catalog});
        for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const int degree : {2, 4, 8})
        {
            SCOPED_TRACE(::testing::Message() << int(format) << "/" << device.toString() << "/" << degree);
            PlanningRoutedExpertWeightWork work{0, 8, 2,
                {sourceOperand(source, plan.request().gate.tensor_name, device, 1),
                 sourceOperand(source, plan.request().up.tensor_name, device, 1),
                 sourceOperand(source, plan.request().down.tensor_name, device, 1)},
                {{PlanningExpertExecution::OwnedExperts, 8}}};
            for (size_t rows : {1u, 2u, 16u, 33u, 64u, 8192u})
            {
                work.projection_ownership.reset();
                work.execution_shares = {{PlanningExpertExecution::OwnedExperts, 8}};
                const auto complete = service.expertSeconds(1, device, work, rows);
                work.projection_ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 256, 512}, 0, degree);
                work.execution_shares = {{PlanningExpertExecution::OwnedExperts, 8 / degree}};
                EXPECT_NEAR(service.expertSeconds(1, device, work, rows), complete / degree, complete * 1e-12);
                work.execution_shares = {{PlanningExpertExecution::OwnedExperts, 0}};
                EXPECT_NEAR(service.expertSeconds(1, device, work, rows), complete / (3 * degree), complete * 1e-12);
                EXPECT_DOUBLE_EQ(service.expertSeconds(1, device, work, 0), 0);
            }
            EXPECT_THROW(service.expertSeconds(1, DeviceId::cpu(), work, 1), std::invalid_argument);
        }
    }
}

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @brief Both declared fabrics require their own evidence; timing cannot pick a transport. */
TEST(PlanningRequestCostModel, ProjectionPolicyRequiresActualFabricAndColumnPublication)
{
    for (const bool p2p : {false, true})
    {
        auto cluster = requestInventory(true);
        if (p2p)
            for (auto &rank : cluster.ranks)
            {
                std::fill(rank.p2p_cuda.begin(), rank.p2p_cuda.end(), true);
                std::fill(rank.p2p_rocm.begin(), rank.p2p_rocm.end(), true);
            }
        cluster.buildNodeAggregations();
        constexpr std::array precisions{PlanningAllreducePrecision::FP32};
        for (const auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16, GGUFTensorType::Q8_0,
                                 GGUFTensorType::Q6_K, GGUFTensorType::IQ2_S})
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(::testing::Message() << p2p << "/" << int(format) << "/" << int(backend));
            // Reproduce the public topology's TP-only search. An unrestricted
            // search also samples PP/overlay host links and hid this defect.
            const AutomaticOrchestrationRequest constraints({.only_backends = {{backend}},
                .only_strategies = {{OrchestrationStrategy::TensorParallel}}});
            const auto sample = PlanningCommunicationSamplePlan::resolve(cluster, constraints, 256, 64, precisions,
                RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            const auto encoded = nativeReceipts(sample, cluster.world_size);
            const auto observations = PlanningCommunicationService::acceptLocal(sample, cluster.world_size, borrow(encoded));
            PlanningGGUFFixture file(true, true, format, 512);
            PlanningModelSource source(file.path());
            const auto candidate = requestCandidate(source, cluster, backend, OrchestrationStrategy::TensorParallel,
                MoERebalanceRuntimeMode::Dynamic, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            const PlanningRequestCostModel costs(source.metadata(), cluster, requestWeights(source, cluster),
                requestArithmetic(cluster), PlanningCommunicationCost(cluster, observations.native, {}, observations.host_device));
            const auto estimate = costs.evaluate(candidate, {64, 16});
            EXPECT_GT(estimate.requestSeconds(), 0);
            EXPECT_NE(estimate.evidence().find("all-expert down slices"), std::string::npos);
            EXPECT_NE(estimate.evidence().find("native fixed allgather; equal DeviceRows"), std::string::npos);
            EXPECT_NE(estimate.evidence().find("projection intermediate uses measured native counted gather; live extents and completion charged once"), std::string::npos);
            EXPECT_EQ(estimate.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
            const PlanningRequestCostModel missing_native(source.metadata(), cluster, requestWeights(source, cluster),
                requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}, observations.host_device));
            EXPECT_THROW(missing_native.evaluate(candidate, {64, 16}), std::invalid_argument);
            if (!p2p)
            {
                const PlanningRequestCostModel missing_mapped(source.metadata(), cluster, requestWeights(source, cluster),
                    requestArithmetic(cluster), PlanningCommunicationCost(cluster, observations.native, {}));
                EXPECT_THROW(missing_mapped.evaluate(candidate, {64, 16}), std::invalid_argument);
            }
        }
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

/** @brief TP-only projection searches retain exact host first-touch scopes, unlike whole-expert TP. */
TEST(PlanningCommunicationService, ProjectionOnlyTPSamplesItsActualHostScopes)
{
    const auto cluster = communicationInventory();
    constexpr std::array precision{PlanningAllreducePrecision::FP32};
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(int(backend));
        const AutomaticOrchestrationRequest tp({.only_backends = {{backend}},
            .only_strategies = {{OrchestrationStrategy::TensorParallel}}});
        EXPECT_TRUE(PlanningCommunicationSamplePlan::resolve(cluster, tp, 257, 31, precision).hostDevice().empty());
        const auto projection = PlanningCommunicationSamplePlan::resolve(cluster, tp, 257, 31, precision,
            RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        ASSERT_EQ(projection.hostDevice().size(), 8u); // Owned GPUs retain every original same-machine host role.
        const auto automatic = PlanningCommunicationSamplePlan::resolve(cluster, tp, 257, 31, precision,
            RoutedExpertComputePolicy::Automatic);
        ASSERT_EQ(automatic.hostDevice().size(), projection.hostDevice().size());
        for (size_t index = 0; index < projection.hostDevice().size(); ++index)
            EXPECT_EQ(automatic.hostDevice()[index], projection.hostDevice()[index]);
        std::set<std::tuple<int, int, int>> scopes;
        for (const auto &sample : projection.hostDevice())
        {
            EXPECT_EQ(sample.device().type, backend);
            EXPECT_EQ(sample, PlanningHostDeviceRequest::fromInventory(cluster,
                {sample.gpuRank(), sample.device()}, {sample.hostRank()}, 257 * 31 * sizeof(float)));
            EXPECT_TRUE(scopes.emplace(sample.gpuRank(), sample.hostRank(), sample.device().ordinal).second);
        }
        const AutomaticOrchestrationRequest single({.only_backends = {{backend}},
            .only_strategies = {{OrchestrationStrategy::SingleDevice}}});
        EXPECT_TRUE(PlanningCommunicationSamplePlan::resolve(cluster, single, 257, 31, precision,
            RoutedExpertComputePolicy::GateUpOwnedDownColumns).hostDevice().empty());
        EXPECT_NE(PlanningCommunicationSamplePlan::resolve(cluster, single, 257, 31, precision).serialize(),
            PlanningCommunicationSamplePlan::resolve(cluster, single, 257, 31, precision,
                RoutedExpertComputePolicy::GateUpOwnedDownColumns).serialize());
        EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, tp, 257, 31, precision,
            static_cast<RoutedExpertComputePolicy>(255)), std::invalid_argument);
    }
}

TEST(PlanningWeightServiceModel, RejectsIncompleteDuplicateAndSubstitutedServiceFamilies)
{
    PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto cluster = inventory();
    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}}), 7);
    const auto owned = projectionReceipts(cluster, plan);
    const auto compute = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(owned));
    const auto memory = memoryCatalog(cluster);
    EXPECT_THROW(PlanningWeightServiceModel(memory, {}), std::invalid_argument);
    EXPECT_THROW(PlanningWeightServiceModel(compute, {compute}), std::invalid_argument);
    EXPECT_THROW(PlanningWeightServiceModel(memory, {memory}), std::invalid_argument);
    EXPECT_THROW(PlanningWeightServiceModel(memory, {compute, compute}), std::invalid_argument);
    auto changed = cluster;
    for (auto &rank : changed.ranks) for (auto &gpu : rank.gpus) gpu.uuid += "another-observation";
    EXPECT_THROW(PlanningWeightServiceModel(memoryCatalog(changed), {compute}), std::invalid_argument);
    changed = cluster;
    for (auto &rank : changed.ranks) ++rank.cpu_worker_threads;
    EXPECT_THROW(PlanningWeightServiceModel(memoryCatalog(changed), {compute}), std::invalid_argument);
    changed = cluster;
    for (auto &rank : changed.ranks) rank.cpu_execution.cache.private_l2_ways += 1;
    EXPECT_THROW(PlanningWeightServiceModel(memoryCatalog(changed), {compute}), std::invalid_argument);
}

TEST(PlanningCommunicationService, SampleBasisIsPhysicalBoundedAndConstraintAware)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precisions);
    EXPECT_EQ(plan.native().size(), 8u); // Two physical nodes, two vendors, two precisions.
    EXPECT_EQ(plan.mpi().size(), 18u); // Control plus two independently varied payload directions.
    EXPECT_EQ(plan.hostDevice().size(), 16u); // Eight original executor/host pairs per backend.
    for (const auto &sample : plan.native())
    {
        EXPECT_EQ(sample.discovery_rank, sample.physical_node ? 2 : 1);
        EXPECT_EQ(sample.request.devices().size(), sample.physical_node ? 2u : 3u);
        EXPECT_EQ(sample.uuids.size(), sample.request.devices().size());
        EXPECT_TRUE(std::is_sorted(sample.request.devices().begin(), sample.request.devices().end(),
            [](const auto a, const auto b) { return a.ordinal < b.ordinal; }));
        EXPECT_FALSE(std::is_sorted(sample.uuids.begin(), sample.uuids.end()));
    }
    for (const auto &sample : plan.mpi())
    {
        EXPECT_NE(sample.initiator, sample.responder);
        EXPECT_TRUE(sample.reply_bytes == 1 || sample.request_bytes == 1);
        EXPECT_TRUE(sample.request_bytes == 1 || sample.request_bytes == 257 * 31 * sizeof(float));
        EXPECT_TRUE(sample.reply_bytes == 1 || sample.reply_bytes == 257 * 31 * sizeof(float));
    }
    for (int from = 0; from < cluster.world_size; ++from)
        for (int to = 0; to < cluster.world_size; ++to)
            if (from != to)
                for (const auto &[outbound, reply] : std::array<std::pair<size_t, size_t>, 3>{
                    {{1, 1}, {257 * 31 * sizeof(float), 1}, {1, 257 * 31 * sizeof(float)}}})
                    EXPECT_EQ(std::count(plan.mpi().begin(), plan.mpi().end(),
                        PlanningMPITransferRequest{from, to, outbound, reply}), 1);
    const auto cpu = PlanningCommunicationSamplePlan::resolve(cluster,
        AutomaticOrchestrationRequest({.only_backends = {{DeviceType::CPU}}}), 257, 31, precisions);
    EXPECT_TRUE(cpu.native().empty());
    EXPECT_TRUE(cpu.hostDevice().empty());
    EXPECT_EQ(cpu.mpi().size(), 18u);
    const auto rocm = PlanningCommunicationSamplePlan::resolve(cluster,
        AutomaticOrchestrationRequest({.only_backends = {{DeviceType::ROCm}}}), 257, 31, precisions);
    ASSERT_EQ(rocm.native().size(), 4u);
    for (const auto &sample : rocm.native()) EXPECT_EQ(sample.request.backend(), CollectiveBackendType::RCCL);
    const auto single = PlanningCommunicationSamplePlan::resolve(cluster,
        AutomaticOrchestrationRequest({.only_strategies = {{OrchestrationStrategy::SingleDevice}}}), 257, 31, precisions);
    EXPECT_TRUE(single.native().empty());
    EXPECT_TRUE(single.mpi().empty());
    EXPECT_TRUE(single.hostDevice().empty());
    auto reordered = cluster;
    for (auto &rank : reordered.ranks) std::reverse(rank.gpus.begin(), rank.gpus.end());
    EXPECT_EQ(PlanningCommunicationSamplePlan::resolve(reordered, AutomaticOrchestrationRequest{},
        257, 31, precisions).serialize(), plan.serialize());
}

TEST(PlanningCommunicationService, RejectsInvalidGeometryAndPolicyWithoutDevices)
{
    const auto cluster = communicationInventory();
    const std::array precision{PlanningAllreducePrecision::FP32};
    const AutomaticOrchestrationRequest request;
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, request, 0, 1, precision), std::invalid_argument);
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, request, INT_MAX, INT_MAX, precision), std::invalid_argument);
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, request, 1, 1, {}), std::invalid_argument);
    const std::array duplicate{PlanningAllreducePrecision::FP16, PlanningAllreducePrecision::FP16};
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, request, 1, 1, duplicate), std::invalid_argument);
    const std::array unknown{static_cast<PlanningAllreducePrecision>(99)};
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, request, 1, 1, unknown), std::invalid_argument);
    auto aliased = cluster;
    auto duplicate_gpu = aliased.ranks[0].gpus.front();
    duplicate_gpu.local_device_id = 99;
    aliased.ranks[0].gpus.push_back(duplicate_gpu);
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(aliased, request, 1, 1, precision), std::invalid_argument);
}

TEST(PlanningCommunicationService, SingleDeviceSearchDoesNotInitializeTransportOrRequireAnMPIContext)
{
    auto cluster = inventory();
    cluster.world_size = 1;
    cluster.ranks.resize(1);
    cluster.ranks[0].gpus.clear();
    const AutomaticOrchestrationRequest request({.only_backends = {{DeviceType::CPU}},
        .only_strategies = {{OrchestrationStrategy::SingleDevice}}});
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto service = PlanningCommunicationService::collect(nullptr, cluster, request, 256, 64, precision);
    ASSERT_TRUE(service.has_value());
    EXPECT_TRUE(service->native().empty());
    EXPECT_TRUE(service->mpi().empty());
    EXPECT_TRUE(service->hostDevice().empty());
}

TEST(PlanningCommunicationService, CompleteNativeReceiptsPreserveRankNodeAndSlowestEndpoint)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precisions);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    auto references = borrow(encoded);
    std::reverse(references.begin(), references.end());
    const auto all = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, references);
    const auto &accepted = all.native;
    EXPECT_EQ(all.host_device.size(), plan.hostDevice().size());
    ASSERT_EQ(accepted.size(), plan.native().size());
    for (size_t index = 0; index < accepted.size(); ++index)
    {
        EXPECT_EQ(accepted[index].sample.discovery_rank, plan.native()[index].discovery_rank);
        EXPECT_EQ(accepted[index].sample.physical_node, plan.native()[index].physical_node);
        EXPECT_EQ(accepted[index].sample.uuids, plan.native()[index].uuids);
        for (const auto &phase : accepted[index].observation.phases)
            EXPECT_DOUBLE_EQ(phase.secondsPerCollective(), phase.endpoints.size() * .0001);
    }
}

/** @test One physical ordered group measures directed pairs only once across wire precisions. */
TEST(PlanningCommunicationService, DirectedPairWireSamplesOncePerPhysicalGroupAcrossPrecisions)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precisions);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    const auto accepted = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(encoded));
    using Group = std::tuple<int, int, DeviceType, std::vector<std::string>>;
    std::map<Group, std::pair<size_t, size_t>> counts;
    for (const auto &service : accepted.native)
    {
        const auto &sample = service.sample;
        auto &count = counts[{sample.discovery_rank, sample.physical_node,
            sample.request.devices().front().type, sample.uuids}];
        const auto &pairs = service.observation.pair_sampling;
        EXPECT_EQ(sample.pair_sampling, count.first + count.second == 0
            ? PlanningPairSampling::Enabled : PlanningPairSampling::Disabled);
        if (sample.pair_sampling == PlanningPairSampling::Enabled)
        {
            ++count.first;
            EXPECT_EQ(pairs.payload_anchors, sample.request.pairPayloadAnchors());
            ASSERT_EQ(pairs.payload_anchors.size(), 3u);
            EXPECT_EQ(pairs.directed_pairs.size(), sample.uuids.size() * (sample.uuids.size() - 1) * 3);
            EXPECT_EQ(pairs.capture_count, pairs.directed_pairs.size() / pairs.payload_anchors.size());
            EXPECT_EQ(pairs.measured_launches, pairs.capture_count);
            for (const auto &edge : pairs.directed_pairs)
                EXPECT_EQ(edge.seconds_per_transfer, (std::vector<double>{.00001, .00002, .00003}));
        }
        else
        {
            ++count.second;
            EXPECT_TRUE(pairs.payload_anchors.empty()); EXPECT_TRUE(pairs.directed_pairs.empty());
            EXPECT_EQ(pairs.capture_count, 0u); EXPECT_EQ(pairs.measured_launches, 0u);
            EXPECT_EQ(pairs.setup_wall_seconds, 0); EXPECT_EQ(pairs.measurement_wall_seconds, 0);
        }
    }
    ASSERT_EQ(counts.size(), 4u);
    for (const auto &entry : counts)
    {
        EXPECT_EQ(entry.second.first, 1u);
        EXPECT_EQ(entry.second.second, 1u);
    }
}

/** @test Strict v6 receipts reject altered direction, completion, ownership and duplicate work. */
TEST(PlanningCommunicationService, DirectedPairWireRejectsMissingInvalidAndSubstitutedEvidence)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precisions);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    for (int defect = 0; defect < 16; ++defect)
    {
        SCOPED_TRACE(defect); auto broken = encoded;
        auto wire = nlohmann::json::parse(broken[1]);
        nlohmann::json *enabled = nullptr, *disabled = nullptr;
        for (auto &entry : wire["observations"])
        {
            if (entry["pairs"]["edges"].empty()) disabled = &entry;
            else enabled = &entry;
        }
        ASSERT_NE(enabled, nullptr); ASSERT_NE(disabled, nullptr);
        auto &pairs = (*enabled)["pairs"]; auto &edges = pairs["edges"];
        if (defect == 0) edges.erase(0);
        if (defect == 1) edges.push_back(edges[0]);
        if (defect == 2) edges[0][0] = edges[0][1]; // Self-edge cannot replace one direction.
        if (defect == 3) std::swap(edges[0][0], edges[0][1]); // Duplicates reverse; original direction is missing.
        if (defect == 4) edges[0][3][0] = -1;
        if (defect == 5) edges[0][3][0] = std::numeric_limits<double>::infinity();
        if (defect == 6) edges[0][3].erase(0);
        if (defect == 7) edges[0][4] = 0;
        if (defect == 8) pairs["anchors"].erase(0);
        if (defect == 9) pairs["captures"] = 0;
        if (defect == 10) pairs["launches"] = 0;
        if (defect == 11) (*disabled)["pairs"] = pairs;
        if (defect == 12) edges[0][0] = 999; // Foreign source index is not another rank's alias.
        if (defect == 13) wire["plan"]["native"][(*enabled)["index"].get<size_t>()]["uuids"][0] = "substituted-source";
        if (defect == 14) wire["plan"]["native"][(*enabled)["index"].get<size_t>()]["rank"] = 2;
        if (defect == 15) pairs["measurement_seconds"] = -1;
        const auto text = wire.dump(); broken[1] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(broken)), std::exception);
    }
    auto wrong_reporter = borrow(encoded);
    std::swap(wrong_reporter[1].discovery_rank, wrong_reporter[2].discovery_rank);
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, wrong_reporter), std::invalid_argument);
}

TEST(PlanningCommunicationService, MalformedOrSubstitutedNativeEvidenceNeverBecomesACost)
{
    const auto cluster = communicationInventory();
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    for (int defect = 0; defect < 14; ++defect)
    {
        SCOPED_TRACE(defect);
        auto broken = encoded;
        auto wire = nlohmann::json::parse(broken[1]);
        if (defect == 0) wire["schema"] = "stale";
        if (defect == 1) wire["plan"]["rank_nodes"][2] = 0;
        if (defect == 2) wire["plan"]["native"][0]["uuids"][0] = "other-physical-gpu";
        if (defect == 3) wire["observations"].erase(0);
        if (defect == 4) wire["observations"].push_back(wire["observations"][0]);
        if (defect == 5) wire["observations"][0]["index"] = -1;
        if (defect == 6) wire["observations"][0]["index"] = .5;
        auto &phase = wire["observations"][0]["phases"][0];
        if (defect == 7) phase["rows"] = 31;
        if (defect == 8) phase["bytes"] = 1;
        if (defect == 9) phase["endpoints"][0][1] = 99;
        if (defect == 10) phase["endpoints"][0][2] = 0;
        if (defect == 11) phase["endpoints"][0][3] = -1;
        if (defect == 12) phase["endpoints"].erase(0);
        if (defect == 13) wire["observations"][0]["phases"].erase(0);
        const auto text = wire.dump();
        broken[1] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(broken)), std::exception);
    }
    auto wrong_owner = borrow(encoded);
    std::swap(wrong_owner[1].discovery_rank, wrong_owner[2].discovery_rank);
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, wrong_owner), std::invalid_argument);
    auto missing = borrow(encoded);
    missing.pop_back();
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, missing), std::invalid_argument);
}

TEST(PlanningCommunicationCost, NativeQueriesPreservePhysicalAliasesPrecisionAndQualifiedSubsetCosts)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precisions);
    const auto wire = nativeReceipts(plan, cluster.world_size);
    auto native = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wire)).native;
    for (auto &service : native)
        for (size_t index = 0; index < service.observation.phases.size(); ++index)
            for (auto &endpoint : service.observation.phases[index].endpoints)
                endpoint.seconds_per_collective = (index ? 6 : 3) *
                    (service.sample.physical_node ? 100.0 : 1.0) *
                    (service.sample.request.precision() == PlanningAllreducePrecision::FP16 ? .5 : 1.0);
    const PlanningCommunicationCost cost(cluster, native, {});
    const std::array full{DeviceId::cuda(0), DeviceId::cuda(10), DeviceId::cuda(20)};
    const auto decode = cost.nativeAllreduce(1, full, 257, 1, PlanningAllreducePrecision::FP32);
    EXPECT_EQ(decode.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
    EXPECT_DOUBLE_EQ(decode.seconds, 3);
    EXPECT_DOUBLE_EQ(cost.nativeAllreduce(1, full, 257, 16, PlanningAllreducePrecision::FP32).seconds, 4.5);
    EXPECT_DOUBLE_EQ(cost.nativeAllreduce(1, full, 257, 31, PlanningAllreducePrecision::FP32).seconds, 6);
    EXPECT_DOUBLE_EQ(cost.nativeAllreduce(1, full, 257, 62, PlanningAllreducePrecision::FP32).seconds, 12);
    EXPECT_DOUBLE_EQ(cost.nativeAllreduce(1, full, 257, 31, PlanningAllreducePrecision::FP16).seconds, 3);
    const std::array aliases{DeviceId::cuda(4), DeviceId::cuda(14)};
    const auto subset = cost.nativeAllreduce(0, aliases, 257, 1, PlanningAllreducePrecision::FP32);
    EXPECT_EQ(subset.basis, PlanningCommunicationBasis::ContainingNativeGroup);
    EXPECT_DOUBLE_EQ(subset.seconds, 3); // Not divided by 3/2, and not sourced from remote node 1.
    const std::array reverse{full[2], full[1], full[0]};
    const auto reordered = cost.nativeAllreduce(1, reverse, 257, 1, PlanningAllreducePrecision::FP32);
    EXPECT_EQ(reordered.basis, PlanningCommunicationBasis::ReorderedNativeGroup);
    EXPECT_DOUBLE_EQ(reordered.seconds, 3);
    const std::array remote{DeviceId::cuda(0), DeviceId::cuda(10)};
    EXPECT_DOUBLE_EQ(cost.nativeAllreduce(2, remote, 257, 1, PlanningAllreducePrecision::FP32).seconds, 300);
    auto wrong_rank = cluster;
    wrong_rank.ranks[0].gpus.clear();
    EXPECT_THROW(PlanningCommunicationCost(wrong_rank, native, {}).nativeAllreduce(
        0, aliases, 257, 1, PlanningAllreducePrecision::FP32), std::invalid_argument);
}

TEST(PlanningCommunicationCost, OneDeviceHasNoLinkTaxAndMissingEvidenceIsNotAProtocolSubstitution)
{
    const auto cluster = communicationInventory();
    const PlanningCommunicationCost empty(cluster, {}, {});
    const std::array one{DeviceId::cuda(4)};
    const auto single = empty.nativeAllreduce(0, one, 257, 31, PlanningAllreducePrecision::FP32);
    EXPECT_EQ(single.basis, PlanningCommunicationBasis::NoCommunication);
    EXPECT_EQ(single.seconds, 0);
    const std::array two{DeviceId::cuda(4), DeviceId::cuda(14)};
    EXPECT_THROW(empty.nativeAllreduce(0, two, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(empty.mpiExchange({0, 1, 1024, 4}), std::invalid_argument);
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto wire = nativeReceipts(plan, cluster.world_size);
    const auto native = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wire)).native;
    const PlanningCommunicationCost cost(cluster, native, {});
    EXPECT_THROW(cost.nativeAllreduce(0, two, 257, 31, PlanningAllreducePrecision::FP16), std::invalid_argument);
    const std::array mixed{DeviceId::cuda(4), DeviceId::rocm(14)};
    const std::array duplicate{DeviceId::cuda(4), DeviceId::cuda(4)};
    EXPECT_THROW(cost.nativeAllreduce(0, mixed, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(0, duplicate, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(0, {}, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(-1, two, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(3, two, 257, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(0, one, 0, 31, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(0, one, INT_MAX, INT_MAX, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(cost.nativeAllreduce(0, one, 257, 31, static_cast<PlanningAllreducePrecision>(9)), std::invalid_argument);
}

TEST(PlanningCommunicationCost, ChangedMembershipAndCorruptOrDuplicateReceiptsFailBeforeQueries)
{
    const auto cluster = communicationInventory();
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto wire = nativeReceipts(plan, cluster.world_size);
    const auto native = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wire)).native;
    for (int defect = 0; defect < 8; ++defect)
    {
        SCOPED_TRACE(defect);
        auto broken = native;
        auto inventory = cluster;
        if (defect == 0) inventory.world_size = 4;
        if (defect == 1) inventory.ranks[1].node_id = 123;
        if (defect == 2) broken[0].sample.physical_node = 123;
        if (defect == 3) broken[0].sample.uuids[0] = "another GPU";
        if (defect == 4) broken[0].sample.uuids.pop_back();
        if (defect == 5) broken[0].observation.phases.pop_back();
        if (defect == 6) broken[0].observation.phases[0].endpoints[0].seconds_per_collective = 0;
        if (defect == 7) broken.push_back(broken[0]);
        EXPECT_THROW(PlanningCommunicationCost(inventory, broken, {}), std::invalid_argument);
    }
}

TEST(PlanningCommunicationCost, HostPrimitivesRetainFirstTouchScopeDirectionAndMechanismAcrossGPUAliases)
{
    auto cluster = communicationInventory();
    // Both first-touch scopes are real local executors of this physical GPU.
    // This variation gives the otherwise remote host NUMA-local GPU authority.
    cluster.ranks[0].cpu.numa_node = 0;
    cluster.ranks[0].cpu_socket_info.front().numa_node = 0;
    cluster.ranks[0].cpu_socket_info.front().physical_cores = {0, 1};
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto wire = nativeReceipts(plan, cluster.world_size);
    auto evidence = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wire));
    for (auto &observed : evidence.host_device)
        for (auto &phase : observed.phases)
        {
            const double time = .001 * (1 + 100 * observed.request.hostRank() + 10 * int(phase.mechanism) + 2 * int(phase.direction));
            phase.service = {PlanningWorkUnit::Bytes, 3.0 * phase.payload_bytes,
                3 * time * (phase.payload_bytes == 1 ? 1 : 2), "synthetic distinct direction/mechanism/first-touch times"};
        }
    const PlanningCommunicationCost costs(cluster, evidence.native, {}, evidence.host_device);
    for (auto mechanism : {PlanningHostTransferMechanism::DMA, PlanningHostTransferMechanism::MappedKernel})
        for (auto direction : {MappedTransferDirection::HostToDevice, MappedTransferDirection::DeviceToHost})
        {
            const double expected = .001 * (101 + 10 * int(mechanism) + 2 * int(direction));
            const auto alias = costs.hostTransfer(0, DeviceId::cuda(4), 1, mechanism, direction, 1);
            EXPECT_DOUBLE_EQ(alias.seconds, expected);
            EXPECT_EQ(alias.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
            EXPECT_DOUBLE_EQ(costs.hostTransfer(1, DeviceId::cuda(0), 1, mechanism, direction, 1).seconds, expected);
            EXPECT_DOUBLE_EQ(costs.hostTransfer(0, DeviceId::cuda(4), 1, mechanism, direction, 257 * 31 * 4).seconds, 2 * expected);
            EXPECT_LT(costs.hostTransfer(0, DeviceId::cuda(4), 0, mechanism, direction, 1).seconds, expected);
            EXPECT_THROW(costs.hostTransfer(0, DeviceId::cuda(4), 2, mechanism, direction, 1), std::invalid_argument);
        }
    EXPECT_THROW(costs.hostTransfer(0, DeviceId::cuda(4), 1, static_cast<PlanningHostTransferMechanism>(99),
        MappedTransferDirection::HostToDevice, 1), std::invalid_argument);
    EXPECT_THROW(costs.hostTransfer(0, DeviceId::cuda(4), 1, PlanningHostTransferMechanism::DMA,
        MappedTransferDirection::HostToDevice, 0), std::invalid_argument);
    auto incomplete = evidence.host_device;
    std::erase_if(incomplete, [](const auto &sample) {
        return sample.request.gpuRank() == 0 && sample.request.hostRank() == 1 &&
            sample.request.device() == DeviceId::cuda(4);
    });
    const PlanningCommunicationCost missing_executor_scope(cluster, evidence.native, {}, incomplete);
    EXPECT_THROW(missing_executor_scope.hostTransfer(0, DeviceId::cuda(4), 1, PlanningHostTransferMechanism::DMA,
        MappedTransferDirection::HostToDevice, 1), std::invalid_argument);
    const PlanningCommunicationCost no_host(cluster, evidence.native, {});
    EXPECT_THROW(no_host.hostTransfer(0, DeviceId::cuda(4), 0, PlanningHostTransferMechanism::DMA,
        MappedTransferDirection::HostToDevice, 1), std::invalid_argument);
    auto changed = cluster;
    changed.ranks[1].cpu.numa_node = 7;
    EXPECT_THROW(PlanningCommunicationCost(changed, {}, {}, evidence.host_device), std::invalid_argument);
    auto duplicate = evidence.host_device;
    duplicate.push_back(duplicate.front());
    EXPECT_THROW(PlanningCommunicationCost(cluster, {}, {}, duplicate), std::invalid_argument);
}

TEST(PlanningCommunicationService, MalformedHostDeviceReceiptsCannotEscapeTheSharedCollectionGate)
{
    const auto cluster = communicationInventory();
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    ASSERT_FALSE(plan.hostDevice().empty());
    const auto reporter = static_cast<std::size_t>(plan.hostDevice().front().gpuRank());
    {
        auto substituted = encoded;
        auto source = nlohmann::json::parse(substituted[reporter]);
        const int host = plan.hostDevice().front().hostRank();
        ASSERT_NE(int(reporter), host);
        auto destination = nlohmann::json::parse(substituted[host]);
        destination["host_device"].push_back(source["host_device"][0]);
        source["host_device"].erase(0);
        const auto from = source.dump(), to = destination.dump();
        substituted[reporter] = {from.begin(), from.end()};
        substituted[host] = {to.begin(), to.end()};
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(substituted)), std::exception);
    }
    for (int defect = 0; defect < 12; ++defect)
    {
        SCOPED_TRACE(defect);
        auto broken = encoded;
        auto wire = nlohmann::json::parse(broken[reporter]);
        if (defect == 0) wire["host_device"].erase(0);
        if (defect == 1) wire["host_device"].push_back(wire["host_device"][0]);
        auto &sample = wire["host_device"][0];
        if (defect == 2) sample["index"] = plan.hostDevice().size();
        if (defect == 3) sample["phases"].erase(0);
        auto &phase = sample["phases"][0];
        if (defect == 4) phase[0] = 7;
        if (defect == 5) phase[1] = 257;
        if (defect == 6) phase[2] = 0;
        if (defect == 7) phase[3] = 0;
        if (defect == 8) phase[4] = 1;
        if (defect == 9) phase[5] = 0;
        if (defect == 10) phase[6] = "";
        if (defect == 11) wire["plan"]["host_device"][0]["host_numa"] = 99;
        const auto text = wire.dump();
        broken[reporter] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(broken)), std::exception);
    }
}

/** @brief Equal half storage does not merge Native FP32-A and GDN half-A evidence. */
TEST(PlanningWeightServiceModel, GDNHalfOperandsKeepSeparateServiceFamilies)
{
    PlanningGGUFFixture file(false, false, GGUFTensorType::F16, 256, GGUFTensorType::F16);
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    profile.architecture = "qwen35moe";
    struct Restore {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.gdn_projection_arithmetic = previous; }
    } restore;
    for (const auto mode : {GDNProjectionArithmetic::FP16, GDNProjectionArithmetic::FP16AlphaBeta, GDNProjectionArithmetic::FP16MoEAlphaBeta})
    {
        mutableDebugEnv().gemm.gdn_projection_arithmetic = mode;
        const auto selected = PlanningWeightServiceModel::samples(PlanningModelMetadata(profile, 2));
        bool native_half = false, gdn_half = false;
        for (const auto &sample : selected)
            if (const auto *request = std::get_if<PlanningModelSampleRequest>(&sample))
            {
                native_half |= request->tensor_name == "blk.0.ffn_gate.weight" ||
                    request->tensor_name == "blk.0.ffn_down.weight" || request->tensor_name == "blk.0.ffn_up.weight";
                gdn_half |= request->tensor_name == "blk.0.ssm_alpha.weight" || request->tensor_name == "blk.0.ssm_beta.weight";
            }
        EXPECT_TRUE(native_half);
        EXPECT_TRUE(gdn_half);
    }
}

/** @test Calibration may observe remote hardware but cannot execute on its remote visibility. */
TEST(PlanningKernelServiceCatalog, RankLocalGPUOwnershipFiltersNativeAndHostDeviceCalibration)
{
    auto cluster = communicationInventory();
    const auto all = PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{});
    for (const auto &observer : all)
        if (observer.device().is_gpu())
            EXPECT_NO_THROW(requireRankLocalGPUOwnership(cluster.ranks.at(observer.discoveryRank()), observer.device(), "sample reporter"));
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31,
        std::array{PlanningAllreducePrecision::FP32});
    for (const auto &sample : plan.native())
        for (const auto device : sample.request.devices())
            EXPECT_NO_THROW(requireRankLocalGPUOwnership(cluster.ranks.at(sample.discovery_rank), device, "native collective reporter"));
    for (const auto &sample : plan.hostDevice())
        EXPECT_NO_THROW(requireRankLocalGPUOwnership(cluster.ranks.at(sample.gpuRank()), sample.device(), "host/device reporter"));
    // CPU participants retain their legitimate cross-rank protocol samples.
    EXPECT_EQ(plan.mpi().size(), 18u);
    EXPECT_EQ(cluster.ranks[0].gpus.size(), 4u);
    // If no reporting rank has the GPU's NUMA, do not substitute a remote one.
    for (auto &rank : cluster.ranks) rank.cpu.numa_node = 19;
    const auto none = PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{});
    EXPECT_TRUE(std::none_of(none.begin(), none.end(), [](const auto &observer) { return observer.device().is_gpu(); }));
    const auto no_native = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31,
        std::array{PlanningAllreducePrecision::FP32});
    EXPECT_TRUE(no_native.native().empty());
    EXPECT_TRUE(no_native.hostDevice().empty());
    EXPECT_EQ(no_native.mpi().size(), 18u);
}

/** @test Locality filtering cannot turn corrupt visible inventory into an empty valid roster. */
TEST(PlanningKernelServiceCatalog, RankLocalGPUOwnershipRejectsMalformedRemoteAliases)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const bool duplicate_uuid : {false, true})
    {
        auto cluster = inventory();
        for (auto &rank : cluster.ranks) rank.cpu.numa_node = 19;
        auto &gpus = cluster.ranks[1].gpus;
        const auto gpu = std::find_if(gpus.begin(), gpus.end(), [&](const auto &value) { return value.type == backend; });
        ASSERT_NE(gpu, gpus.end());
        if (duplicate_uuid)
        {
            auto duplicate = *gpu;
            duplicate.local_device_id += 99;
            gpus.push_back(duplicate);
        }
        else gpu->numa_node = 1;
        EXPECT_THROW(PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}), std::invalid_argument);
        EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 1, 1,
            std::array{PlanningAllreducePrecision::FP32}), std::invalid_argument);
    }
}

/**
 * @brief Replay terminal HC costing through the actual measured-service compositor.
 *
 * The existing source-free FP32 observations provide service, never a claimed
 * QFN timing. Actual HC N/K observations are replayed as floating operands to
 * isolate row/stage dispatch without manufacturing a second kernel sampler.
 */
TEST(PlanningRequestCostModel, QFNHyperconnectionWorkIsPricedAndReplayStable)
{
    const auto cluster = requestInventory(false);
    PlanningGGUFFixture file(false, true, GGUFTensorType::F32);
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    for (const auto &tensor : std::vector<TensorSizeInfo>{
             {"output_hc_down.weight", 320u * 10240u * 4u, "F32", 320u * 10240u, 10240, -1},
             {"output_hc_up.weight", 10240u * 320u * 4u, "F32", 10240u * 320u, 320, -1},
             {"output_hc_norm.weight", 10240u * 4u, "F32", 10240, 10240, -1},
             {"blk.0.hc_attn_inject.weight", 4u * 10240u * 4u, "F32", 4u * 10240u, 10240, 0}})
        profile.tensors.push_back(tensor);
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto candidate = requestCandidate(source, cluster, backend, OrchestrationStrategy::SingleDevice);
        const PlanningRequestCostModel original(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}));
        const PlanningRequestCostModel extensions(PlanningModelMetadata(profile, source.metadata().mainLayerCount()),
            cluster, requestWeights(source, cluster), requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}));
        for (const auto workload : {OrchestrationPlanningWorkload{1, 1}, OrchestrationPlanningWorkload{64, 16},
                                    OrchestrationPlanningWorkload{65, 16}})
        {
            const auto baseline = original.evaluate(candidate, workload);
            const auto priced = extensions.evaluate(candidate, workload);
            const auto replay = extensions.evaluate(candidate, workload);
            EXPECT_GT(priced.prefillSeconds(), baseline.prefillSeconds());
            EXPECT_GT(priced.decodeSecondsPerToken(), baseline.decodeSecondsPerToken());
            EXPECT_DOUBLE_EQ(priced.prefillSeconds(), replay.prefillSeconds());
            EXPECT_DOUBLE_EQ(priced.decodeSecondsPerToken(), replay.decodeSecondsPerToken());
        }
        auto unknown = profile;
        unknown.tensors.push_back({"blk.0.future_op.weight", 256u * 4u, "F32", 256, 256, 0});
        const PlanningRequestCostModel rejected(PlanningModelMetadata(unknown, source.metadata().mainLayerCount()),
            cluster, requestWeights(source, cluster), requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}));
        EXPECT_THROW((void)rejected.evaluate(candidate, {64, 16}), std::invalid_argument);
    }
}


/** @brief Storage-only CPU bandwidth cannot widen GPU compute or substitute a projection sample. */
TEST(PlanningWeightServiceModel, ImmutableHostSourceStreamingDoesNotAdmitCPUCompute)
{
    PlanningGGUFFixture file(false, false, GGUFTensorType::F32);
    PlanningModelSource source(file.path());
    const auto cluster = inventory();
    AutomaticOrchestrationOptions gpu_options, host_options;
    gpu_options.only_backends = std::vector{DeviceType::ROCm};
    host_options.only_backends = std::vector{DeviceType::CPU};
    const AutomaticOrchestrationRequest gpu_policy(gpu_options), host_policy(host_options);
    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}}), 7);
    const auto gpu_stream_bytes = streamingReceipts(cluster, gpu_policy);
    const auto cpu_stream_bytes = streamingReceipts(cluster, host_policy);
    const auto gpu_kernel_bytes = projectionReceipts(cluster, plan, gpu_policy);
    const auto gpu_stream = PlanningKernelServiceCatalog::accept(cluster, gpu_policy,
        PlanningStreamingServicePlan{}, borrow(gpu_stream_bytes));
    const auto cpu_stream = PlanningKernelServiceCatalog::accept(cluster, host_policy,
        PlanningStreamingServicePlan{}, borrow(cpu_stream_bytes));
    const auto gpu_kernel = PlanningKernelServiceCatalog::accept(cluster, gpu_policy, plan, borrow(gpu_kernel_bytes));
    const PlanningWeightServiceModel no_host(gpu_stream, {gpu_kernel});
    const PlanningWeightServiceModel with_host(gpu_stream, {gpu_kernel}, cpu_stream);
    EXPECT_EQ(with_host.streamingEvidence().records().size(), gpu_stream.records().size());
    for (const auto &record : with_host.streamingEvidence().records())
        EXPECT_TRUE(record.observer.device().is_rocm());
    for (const auto &record : cpu_stream.records())
    {
        const auto rank = record.observer.discoveryRank();
        const auto device = record.observer.device();
        const auto &observed = std::get<PlanningMemoryBandwidthObservation>(record.observation);
        for (const double live_bytes : {1.0, 4096.0})
        {
            EXPECT_THROW((void)no_host.memorySeconds(rank, device, live_bytes), std::exception);
            EXPECT_DOUBLE_EQ(with_host.memorySeconds(rank, device, live_bytes),
                observed.service.secondsFor(PlanningWorkUnit::Bytes, live_bytes));
        }
        EXPECT_THROW((void)with_host.projectionSeconds(rank, device,
            sourceOperand(source, "blk.0.ffn_gate.weight", device), 1), std::exception);
    }
    EXPECT_THROW((void)PlanningWeightServiceModel(gpu_stream, {gpu_kernel}, gpu_kernel), std::invalid_argument);
    EXPECT_THROW((void)PlanningWeightServiceModel(gpu_stream, {gpu_kernel}, gpu_stream), std::invalid_argument);
    const auto all_kernel_bytes = projectionReceipts(cluster, plan);
    const auto all_kernel = PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{}, plan, borrow(all_kernel_bytes));
    EXPECT_THROW((void)PlanningWeightServiceModel(memoryCatalog(cluster), {all_kernel}, cpu_stream), std::invalid_argument);
    // Existing CPU compute already has one independently authenticated streaming
    // source; the optional host arm must never replace or double-count it.
    const PlanningWeightServiceModel existing_cpu(memoryCatalog(cluster), {all_kernel});
    EXPECT_GT(existing_cpu.memorySeconds(1, DeviceId::cpu(), 4096), 0.0);
}

/** @brief Physical evidence cannot be promoted into an unmeasured collective protocol. */
TEST(PlanningCommunicationService, PhysicalOnlyWireKeepsCopiesWithoutCollectiveSamples)
{
    const auto cluster = communicationInventory();
    const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31,
        precisions, RoutedExpertComputePolicy::Apportioned, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
    const auto encoded = nativeReceipts(plan, cluster.world_size);
    const auto accepted = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(encoded));
    ASSERT_FALSE(accepted.native.empty());
    std::set<std::pair<int, DeviceType>> observed;
    for (const auto &service : accepted.native)
    {
        EXPECT_EQ(service.sample.mode, PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
        EXPECT_EQ(service.observation.mode, service.sample.mode);
        EXPECT_TRUE(service.observation.phases.empty());
        EXPECT_EQ(service.sample.pair_sampling, PlanningPairSampling::Enabled);
        EXPECT_TRUE(observed.emplace(service.sample.discovery_rank, service.sample.request.devices().front().type).second);
        EXPECT_NO_THROW(service.observation.pair_sampling.validate(service.sample.request));
    }
    const PlanningCommunicationCost cost(cluster, accepted.native, {});
    const auto &first = accepted.native.front().sample;
    EXPECT_THROW(cost.nativeAllreduce(first.discovery_rank, first.request.devices(), 257, 31,
        first.request.precision()), std::invalid_argument);
    EXPECT_GT(cost.physicalPeerTransfer(first.discovery_rank, first.request.devices()[0],
        first.request.devices()[1], 16).seconds, 0);
    PhysicalMemoryPlanBuilder builder;
    const PhysicalMemoryResource host{.world_rank=first.discovery_rank, .device=DeviceId::cpu(),
        .total_bytes=1ull<<40, .admission_available_bytes=1ull<<40};
    std::vector<PhysicalMemoryResource> resources;
    for (const auto device : first.request.devices())
        resources.push_back({.world_rank=first.discovery_rank, .device=device,
            .total_bytes=1ull<<40, .admission_available_bytes=1ull<<40});
    PlanningLocalTPMeasurement::contributeMemory(first.request, host, resources, builder,
        PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
    const auto physical_bom = builder.build();
    for (const auto &resource : physical_bom.resources())
        EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::LocalCollective), 0u);
    auto wrong_mode = encoded;
    for (auto &receipt : wrong_mode)
    {
        auto wire = nlohmann::json::parse(receipt);
        if (wire["observations"].empty()) continue;
        wire["observations"][0]["mode"] = int(PlanningLocalTPMeasurementMode::NativeCollective);
        const auto text = wire.dump(); receipt.assign(text.begin(), text.end());
        break;
    }
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wrong_mode)), std::invalid_argument);
    auto invented_collective = encoded;
    for (auto &receipt : invented_collective)
    {
        auto wire = nlohmann::json::parse(receipt);
        if (wire["observations"].empty()) continue;
        wire["observations"][0]["phases"].push_back({{"rows", 1}, {"bytes", 1}, {"endpoints", nlohmann::json::array()}});
        const auto text = wire.dump(); receipt.assign(text.begin(), text.end());
        break;
    }
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(invented_collective)), std::invalid_argument);
    EXPECT_THROW(PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31,
        precisions, RoutedExpertComputePolicy::Apportioned, static_cast<PlanningLocalTPMeasurementMode>(99)), std::invalid_argument);
}

/** @test Shared source declaration is independent of any private GPU preparation and unchanged by publication origin. */
TEST(PlanningKernelServiceCatalog, ProjectionSharedSourceBOMIsSingleAndSeparateFromGPUPrivate)
{
    PlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto plan=PlanningMatrixSamplePlan::resolve(source,{"blk.0.ffn_gate.weight",PlanningWholeMatrix{}});
    const PhysicalMemoryResource host{.world_rank=0,.device=DeviceId::cpu(),
        .total_bytes=1ull<<30,.admission_available_bytes=1ull<<30};
    PhysicalMemoryPlanBuilder original,received;
    PlanningProjectionMeasurement::contributeSourceMemory(plan,PlanningSampleOrigin::LocalGGUF,host,original);
    PlanningProjectionMeasurement::contributeSourceMemory(plan,PlanningSampleOrigin::PublishedPayload,host,received);
    const auto declared=original.build();const auto published=received.build();
    ASSERT_EQ(declared.resources().size(),1u);ASSERT_EQ(published.resources().size(),1u);
    EXPECT_EQ(declared.resources()[0].bytes(PhysicalMemoryOwner::ModelSourcePayload),plan.geometry().physicalBytes());
    EXPECT_EQ(declared.summary(),published.summary());
    for(const auto owner:{PhysicalMemoryOwner::PrimaryModelWeights,PhysicalMemoryOwner::ExecutionWorkspace,
        PhysicalMemoryOwner::WeightLoadStaging,PhysicalMemoryOwner::NativeGraphExecutable})
        EXPECT_EQ(declared.resources()[0].bytes(owner),0u);
    EXPECT_THROW(PlanningProjectionMeasurement::contributeSourceMemory(plan,static_cast<PlanningSampleOrigin>(255),host,original),std::invalid_argument);
    EXPECT_THROW(PlanningProjectionMeasurement::contributeGPUParticipantMemory(plan,host,host,original,64),std::invalid_argument);
}

/** @test Startup ranking consumes exactly decode/prefill; missing, duplicate and obsolete verifier evidence never becomes a rate. */
TEST(PlanningWeightServiceModel, StartupExpertTwoPhaseCoverageRejectsObsoleteGroupedEvidence)
{
    PlanningGGUFFixture file(true,false,GGUFTensorType::BF16,512);PlanningModelSource source(file.path());
    const auto plan=PlanningExpertSamplePlan::resolve(source,request());const auto cluster=inventory();
    const auto encoded=receipts(cluster,plan);
    const auto accepted=PlanningKernelServiceCatalog::accept(cluster,AutomaticOrchestrationRequest{},plan,borrow(encoded));
    for(const auto &record:accepted.records())std::visit([](const auto &observed){
        if constexpr(requires{observed.formats;}){
            ASSERT_EQ(observed.phases.size(),2u);
            EXPECT_EQ(observed.phases[0].phase,ExpertHistogramSource::DecodeToken);
            EXPECT_EQ(observed.phases[1].phase,ExpertHistogramSource::PrefillChunk);
        }
    },record.observation);
    auto slowed=encoded;
    for(auto &receipt:slowed){auto wire=nlohmann::json::parse(receipt);
        for(auto &observed:wire["observations"])observed["phases"][1]["seconds"]=observed["phases"][1]["seconds"].get<double>()*100;
        const auto text=wire.dump();receipt.assign(text.begin(),text.end());}
    const auto slow_catalog=PlanningKernelServiceCatalog::accept(cluster,AutomaticOrchestrationRequest{},plan,borrow(slowed));
    const PlanningWeightServiceModel fast(memoryCatalog(cluster,.000001),{accepted}),slow(memoryCatalog(cluster,.000001),{slow_catalog});
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)}){
        PlanningRoutedExpertWeightWork work{0,8,2,{
            sourceOperand(source,plan.request().gate.tensor_name,device,1),
            sourceOperand(source,plan.request().up.tensor_name,device,1),
            sourceOperand(source,plan.request().down.tensor_name,device,1)},
            {{PlanningExpertExecution::OwnedExperts,8}}};
        const double decode=fast.expertSeconds(1,device,work,1);
        // Occupancy uses log1p/expm1: a one-row group can round just above
        // one, exposing a few ULPs of the deliberately exaggerated slope.
        EXPECT_NEAR(decode,slow.expertSeconds(1,device,work,1),
            64*std::numeric_limits<double>::epsilon()*decode);
        EXPECT_GT(slow.expertSeconds(1,device,work,128),fast.expertSeconds(1,device,work,128));
    }
    for(int defect=0;defect<5;++defect){SCOPED_TRACE(defect);auto broken=encoded;
        auto wire=nlohmann::json::parse(broken[0]);auto &phases=wire["observations"][0]["phases"];
        if(defect==0)phases.erase(1);
        if(defect==1)phases[1]=phases[0];
        if(defect==2){auto obsolete=phases[0];obsolete["phase"]=int(ExpertHistogramSource::GroupedVerifier);
            obsolete["rows"]=MoEOverlayCPUServiceMeasurement::rowsForPhase(ExpertHistogramSource::GroupedVerifier);phases.push_back(obsolete);}
        if(defect==3)wire["source"]["startup_phases"]={0,1,2};
        if(defect==4)wire["schema"]="llaminar.planning-kernel-service.v4";
        const auto text=wire.dump();broken[0].assign(text.begin(),text.end());
        EXPECT_THROW(PlanningKernelServiceCatalog::accept(cluster,AutomaticOrchestrationRequest{},plan,borrow(broken)),std::invalid_argument);
    }
}

/** @test Live service scope still retains grouped-verifier work; the startup-only workspace entry does not alter it. */
TEST(PlanningWeightServiceModel, StartupExpertWorkspaceKeepsLiveVerifierContractSeparate)
{
    static_assert(kPlanningExpertStartupPhases.size()==2);
    static_assert(kExpertHistogramProductionSourceCount==3);
    EXPECT_EQ(expertHistogramProductionSourceIndex(ExpertHistogramSource::GroupedVerifier),2u);
    EXPECT_GT(MoEOverlayCPUServiceMeasurement::rowsForPhase(ExpertHistogramSource::GroupedVerifier),1);
    PlanningGGUFFixture file(true,false);PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    WorkspaceMemoryGeometry geometry;
    geometry.device=DeviceId::rocm(7);geometry.first_layer=0;geometry.last_layer=source.metadata().mainLayerCount()-1;
    geometry.apportioned_routed_experts=true;
    const auto startup=WorkspaceMemoryEstimator::estimateStartupExpertService(profile,geometry);
    const auto live=WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry);
    EXPECT_GT(startup,0u);EXPECT_GE(live,startup);
    EXPECT_EQ(WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry),live);
}

/** @test Residual-trunk geometry changes PP traffic while identical compute-H TP collectives stay unchanged. */
TEST(PlanningRequestCostModel, PipelineResidualTrunkPayloadIsIndependentOfComputeWidth)
{
    auto cluster=requestInventory(true);
    cluster.world_size=1;cluster.ranks.resize(1);cluster.buildNodeAggregations();
    Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    // Complete original QFN layer kinds, HC tensors and sparse geometry are retained.
    for(auto &rank:cluster.ranks){rank.cpu_memory_bytes=rank.cpu.memory_bytes=rank.cpu.free_memory_bytes=1ull<<40;
        for(auto &socket:rank.cpu_socket_info)socket.memory_bytes=socket.available_memory_bytes=1ull<<40;
        for(auto &gpu:rank.gpus)gpu.memory_bytes=gpu.free_memory_bytes=1ull<<40;}
    cluster.buildNodeAggregations();
    constexpr std::array precisions{PlanningAllreducePrecision::FP32};
    const auto sample=PlanningCommunicationSamplePlan::resolve(cluster,AutomaticOrchestrationRequest{},256,64,precisions);
    const auto encoded=nativeReceipts(sample,1);
    auto observed=PlanningCommunicationService::acceptLocal(sample,1,borrow(encoded));
    // Both decode widths lie below this fixture's bulk anchor. Give each
    // independently identified host mechanism/direction a positive byte slope;
    // a flat latency floor cannot witness their different live payloads.
    for(auto &host:observed.host_device)for(auto &phase:host.phases)
        phase.service=PlanningServiceObservation(PlanningWorkUnit::Bytes,3.0*phase.payload_bytes,
            .0003+3.0*phase.payload_bytes*1e-8,"authored PP payload-dependent host transfer service");
    const auto weights=requestWeights(source,cluster);const auto arithmetic=requestArithmetic(cluster);
    const auto makeCost=[&](std::optional<uint32_t> trunk) {
        auto profile=source.metadata().memoryProfile();
        if(!profile.flash_next)throw std::logic_error("PP residual test requires complete authenticated QFN geometry");
        profile.flash_next->residual_width=trunk.value_or(profile.d_model);
        return PlanningRequestCostModel(PlanningModelMetadata(std::move(profile),source.metadata().mainLayerCount()),
            cluster,weights,arithmetic,PlanningCommunicationCost(cluster,observed.native,{},observed.host_device));
    };
    // Compare only the residual transport dimension over the same complete
    // QFN source inventory; compute-H and all other authenticated facts stay fixed.
    // This is a cost-composition diagnostic, not numerical model evidence.
    const auto ordinary=makeCost(std::nullopt);
    const auto expanded=makeCost(4*source.metadata().memoryProfile().d_model);
    const auto invalid=makeCost(0);
    std::vector<std::string> args{"llaminar2","--define-domain","head=cuda:4;scope=rank_local;backend=nccl;owner=0",
        "--define-domain","tail=rocm:4;scope=rank_local;backend=rccl;owner=0",
        "--pp-stage","0=head:0-0","--pp-stage","1=tail:1-1"};
    const int midpoint=source.metadata().mainLayerCount()/2;
    args[6]="0=head:0-"+std::to_string(midpoint-1);
    args[8]="1=tail:"+std::to_string(midpoint)+"-"+std::to_string(source.metadata().mainLayerCount()-1);
    std::vector<char*> argv;for(auto &arg:args)argv.push_back(arg.data());
    auto config=OrchestrationConfigParser{}.parseArgs(argv.size(),argv.data());
    config.model_path=source.path();config.max_seq_len=8192;config.prefill_max_bucket_size=64;
    const auto services=planningHostServiceBudgets(cluster.world_size);
    const auto pipeline=AdmittedOrchestrationCandidate::admit(
        {OrchestrationStrategy::PipelineParallel,ExecutionRankMembership(cluster,{0}),std::move(config)},source,
        {.prefill_bucket_rows={32,64},.minimum_prefill_sequence_rows=1,.maximum_cached_prefill_buckets=8,.retained_host_services=services});
    const OrchestrationPlanningWorkload workload{64,16};
    const auto small=ordinary.evaluate(pipeline,workload),large=expanded.evaluate(pipeline,workload);
    EXPECT_GT(large.prefillSeconds(),small.prefillSeconds());
    EXPECT_GT(large.decodeSecondsPerToken(),small.decodeSecondsPerToken());
    EXPECT_GT(large.prefillSeconds()-small.prefillSeconds(),
        large.decodeSecondsPerToken()-small.decodeSecondsPerToken());
    EXPECT_THROW(invalid.evaluate(pipeline,workload),std::invalid_argument);
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        const auto tp=requestCandidate(source,cluster,backend,OrchestrationStrategy::TensorParallel,
            MoERebalanceRuntimeMode::Off,RoutedExpertComputePolicy::Apportioned,services);
        EXPECT_DOUBLE_EQ(ordinary.evaluate(tp,workload).requestSeconds(),expanded.evaluate(tp,workload).requestSeconds());
    }
}

/** @test Native gather receipts require both real protocols, exact payload anchors and complete ordered original participants. */
TEST(PlanningCommunicationService, NativeGatherWireRequiresBothProtocolsAndOriginalParticipants)
{
    const auto cluster=communicationInventory();const AutomaticOrchestrationRequest request;
    constexpr std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan=PlanningCommunicationSamplePlan::resolve(cluster,request,256,64,precision);
    const auto original=nativeReceipts(plan,cluster.world_size);ASSERT_FALSE(plan.native().empty());
    const int owner=plan.native().front().discovery_rank;
    EXPECT_NO_THROW(PlanningCommunicationService::acceptLocal(plan,cluster.world_size,borrow(original)));
    for(int defect=0;defect<11;++defect){SCOPED_TRACE(defect);auto changed=original;
        auto wire=nlohmann::json::parse(changed.at(owner));auto &entry=wire["observations"][0];
        if(defect==0)entry.erase("gather_phases");
        if(defect==1)entry["gather_phases"]=nlohmann::json::array();
        if(defect==2)entry["gather_phases"][0]["bytes"]=2;
        if(defect==3)entry["gather_phases"][0]["participants"]=0;
        if(defect==4)entry["gather_phases"][0]["participants"]=999;
        if(defect==5)entry["gather_phases"][0]["endpoints"][0][1]=999;
        if(defect==6)entry["gather_phases"][0]["endpoints"].erase(entry["gather_phases"][0]["endpoints"].begin());
        if(defect==7)entry["gather_phases"][0]["endpoints"][0][3]=-1;
        if(defect==8)std::swap(entry["gather_phases"][0],entry["gather_phases"][1]);
        if(defect==9)entry["gather_phases"][0]["protocol"]=99;
        if(defect==10)entry["gather_phases"][2]["bytes"]=1;
        const auto text=wire.dump();changed[owner].assign(text.begin(),text.end());
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan,cluster.world_size,borrow(changed)),std::invalid_argument);
    }
    const auto physical=PlanningCommunicationSamplePlan::resolve(cluster,request,256,64,precision,
        RoutedExpertComputePolicy::Apportioned,PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
    auto extra=nativeReceipts(physical,cluster.world_size);const int physical_owner=physical.native().front().discovery_rank;
    auto wire=nlohmann::json::parse(extra[physical_owner]);
    wire["observations"][0]["gather_phases"]=nlohmann::json::parse(original[owner])["observations"][0]["gather_phases"];
    const auto text=wire.dump();extra[physical_owner].assign(text.begin(),text.end());
    EXPECT_THROW(PlanningCommunicationService::acceptLocal(physical,cluster.world_size,borrow(extra)),std::invalid_argument);
}
