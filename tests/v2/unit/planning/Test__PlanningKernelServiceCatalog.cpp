/**
 * @file Test__PlanningKernelServiceCatalog.cpp
 * @brief Device-free proof that evidence never invents topology or execution scope.
 *
 * Synthetic positive times exercise only the codec/identity contract. Ordinal
 * aliases, reversed receipt order, missing results and malformed work/graph/ISA
 * fields must not become a usable catalog. No test here opens a device.
 */
#include "planning/PlanningKernelServiceCatalog.h"
#include "planning/PlanningWeightServiceModel.h"
#include "planning/PlanningCommunicationService.h"
#include "planning/PlanningCommunicationCost.h"
#include "planning/PlanningRequestCostModel.h"
#include "utils/CPUFeatures.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <algorithm>

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
            const auto first = rank.gpus;
            for (const auto &gpu : first)
                for (int index = 1; index < (rank.rank == 1 ? 3 : 2); ++index)
                {
                    auto extra = gpu;
                    extra.local_device_id += index * 10;
                    extra.uuid = "0-" + gpu.uuid + "-" + std::to_string(index);
                    rank.gpus.push_back(extra);
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
            for (int rows : {1, sample.request.prefillRows()})
            {
                PlanningLocalTPPhaseObservation phase{rows, sample.request.payloadBytes(rows), {}};
                for (const auto device : sample.request.devices())
                    phase.endpoints.push_back({device, 3, .0001 * (phase.endpoints.size() + 1)});
                observed.phases.push_back(std::move(phase));
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
            host_groups[request.rank()].emplace_back(index, std::move(observed));
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
        for (auto phase : {ExpertHistogramSource::DecodeToken, ExpertHistogramSource::PrefillChunk,
                           ExpertHistogramSource::GroupedVerifier})
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
    std::vector<std::vector<uint8_t>> streamingReceipts(const ClusterInventory &cluster)
    {
        std::vector<std::vector<PlanningKernelObservation>> rows(cluster.world_size);
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}))
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
        const PlanningProjectionServicePlan &plan)
    {
        std::vector<std::vector<PlanningKernelObservation>> rows(cluster.world_size);
        const auto shape = plan.matrix().geometry();
        for (const auto &observer : PlanningKernelServiceCatalog::observers(cluster, AutomaticOrchestrationRequest{}))
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
            rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = size_t{16} << 30;
            for (auto &socket : rank.cpu_socket_info) socket.memory_bytes = size_t{16} << 30;
            for (auto &gpu : rank.gpus)
            {
                gpu.memory_bytes = gpu.free_memory_bytes = size_t{16} << 30;
                gpu.compute_units = 32;
            }
        }
        cluster.buildNodeAggregations();
        return cluster;
    }

    /** @return Complete synthetic source-family evidence for pure request arithmetic tests. */
    PlanningWeightServiceModel requestWeights(const PlanningModelSource &source, const ClusterInventory &cluster)
    {
        std::vector<PlanningKernelServiceCatalog> kernels;
        for (const auto &sample : PlanningWeightServiceModel::samples(source.metadata()))
            std::visit([&](const auto &request) {
                using Request = std::decay_t<decltype(request)>;
                if constexpr (std::is_same_v<Request, PlanningModelSampleRequest>)
                {
                    const PlanningProjectionServicePlan plan(PlanningMatrixSamplePlan::resolve(source, request), 64);
                    const auto encoded = projectionReceipts(cluster, plan);
                    kernels.push_back(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
                        plan, borrow(encoded)));
                }
                else
                {
                    const auto plan = PlanningExpertSamplePlan::resolve(source, request);
                    const auto encoded = receipts(cluster, plan);
                    kernels.push_back(PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
                        plan, borrow(encoded)));
                }
            }, sample);
        return {memoryCatalog(cluster, .000001), std::move(kernels)};
    }

    /** @return Exact admitted rank-zero candidate; no model payload or accelerator is loaded. */
    AdmittedOrchestrationCandidate requestCandidate(const PlanningModelSource &source, const ClusterInventory &cluster,
        DeviceType backend, OrchestrationStrategy strategy,
        MoERebalanceRuntimeMode movement = MoERebalanceRuntimeMode::Off)
    {
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 8192;
        request.prefill_max_bucket_size = 64;
        // Costing predicts initial placement without assuming future movement
        // gains; admission must still retain the exact requested maintenance.
        request.moe_rebalance.mode = movement;
        request.automatic_planning.only_backends = std::vector{backend};
        request.automatic_planning.only_strategies = std::vector{strategy};
        std::optional<AdmittedOrchestrationCandidate> found;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), cluster, [&](auto candidate) {
            if (found || candidate.membership.discoveryRanks() != std::vector<int>{0}) return;
            found = AdmittedOrchestrationCandidate::admit(std::move(candidate), source,
                {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
                 .maximum_cached_prefill_buckets = 8});
        });
        if (!found) throw std::logic_error("No request-cost fixture candidate");
        return std::move(*found);
    }

    /** @return Complete source-free proxy matching the codec fixture's physical observers. */
    PlanningKernelServiceCatalog requestArithmetic(const ClusterInventory &cluster)
    {
        const auto encoded = arithmeticReceipts(cluster);
        return PlanningKernelServiceCatalog::accept(cluster, AutomaticOrchestrationRequest{},
            PlanningFP32ArithmeticPlan{}, borrow(encoded));
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
    const auto native = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(encoded)).native;
    for (bool moe : {false, true})
    {
        PlanningGGUFFixture file(moe);
        PlanningModelSource source(file.path());
        const PlanningRequestCostModel costs(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, native, {}));
        const PlanningRequestCostModel missing(source.metadata(), cluster, requestWeights(source, cluster),
            requestArithmetic(cluster), PlanningCommunicationCost(cluster, {}, {}));
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
            EXPECT_NE(cost.evidence().find("measured native FP32 group service"), std::string::npos);
            EXPECT_EQ(cost.evidence().find("point-to-point traffic"), std::string::npos);
            EXPECT_THROW(missing.evaluate(candidate, {64, 16}), std::invalid_argument);
        }
    }
}

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
    EXPECT_EQ(plan.hostDevice().size(), 14u); // Every rank's distinct first-touch scope is retained.
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
    const auto cluster = communicationInventory();
    const std::array precision{PlanningAllreducePrecision::FP32};
    const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, AutomaticOrchestrationRequest{}, 257, 31, precision);
    const auto wire = nativeReceipts(plan, cluster.world_size);
    auto evidence = PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(wire));
    for (auto &observed : evidence.host_device)
        for (auto &phase : observed.phases)
        {
            const double time = .001 * (1 + 100 * observed.request.rank() + 10 * int(phase.mechanism) + 2 * int(phase.direction));
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
    for (int defect = 0; defect < 12; ++defect)
    {
        SCOPED_TRACE(defect);
        auto broken = encoded;
        auto wire = nlohmann::json::parse(broken[0]);
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
        broken[0] = {text.begin(), text.end()};
        EXPECT_THROW(PlanningCommunicationService::acceptLocal(plan, cluster.world_size, borrow(broken)), std::exception);
    }
}
