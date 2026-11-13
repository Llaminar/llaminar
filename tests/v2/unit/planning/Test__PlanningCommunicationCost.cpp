/**
 * @file Test__PlanningCommunicationCost.cpp
 * @brief Device-free native pair placement pricing and strict observer binding.
 *
 * Authored completed receipts exercise the actual public costing API. They are
 * protocol/identity regressions, not hardware performance measurements. Pair
 * placement scores remain distinct from full-group collective observations.
 */
#include "planning/PlanningCommunicationCost.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PlanningPeerCopyAccess.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
    /** @return Two discovery ranks with physical aliases, only one of which supplies receipts. */
    ClusterInventory inventory()
    {
        ClusterInventory result; result.world_size = 2;
        for (int index = 0; index < 2; ++index)
        {
            RankInventory rank; rank.rank = index; rank.node_id = 0; rank.local_rank = index;
            rank.hostname = "physical-node"; rank.cpu.numa_node = 1;
            rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = 64ull << 30;
            // Both original observers have real first-touch ownership on NUMA1,
            // with distinct two-core workshares on that same physical socket.
            rank.cpu.type = DeviceType::CPU;
            rank.cpu_cores = rank.cpu.compute_units = rank.cpu_worker_threads = 2;
            rank.cpu_memory_bytes = rank.cpu.memory_bytes;
            rank.numa_nodes = 1;
            CPUSocketInfo socket; socket.socket_id = 1; socket.numa_node = 1;
            socket.memory_bytes = rank.cpu.memory_bytes;
            socket.physical_cores = {2 * index, 2 * index + 1};
            rank.cpu_socket_info.push_back(std::move(socket));
            for (int device = 0; device < 4; ++device)
                rank.gpus.push_back({.type = DeviceType::ROCm, .local_device_id = device + 1,
                    .memory_bytes = 32ull << 30, .free_memory_bytes = 30ull << 30,
                    .compute_units = 64, .uuid = "gpu-" + std::to_string(device), .numa_node = 1});
            rank.p2p_rocm_count = 4; rank.p2p_rocm.assign(16, true);
            result.ranks.push_back(std::move(rank));
        }
        result.buildNodeAggregations(); return result;
    }

    /** @return Complete authored receipt with independent original-observer event times. */
    PlanningNativeCollectiveService recordedSample(PlanningNativeCollectiveSample sample, const std::vector<uint8_t> *access = nullptr)
    {
        const size_t degree = sample.uuids.size();
        if (sample.direct_peer_access.empty())
        {
            sample.direct_peer_access.assign(degree * degree, 1);
            for (size_t index = 0; index < degree; ++index) sample.direct_peer_access[index * degree + index] = 0;
        }
        if (access) sample.direct_peer_access = *access;
        PlanningLocalTPObservations observed{sample.request, {}};
        observed.mode=sample.mode;
        if(sample.mode==PlanningLocalTPMeasurementMode::NativeCollective)
        for (const int rows : {1, 64})
        {
            PlanningLocalTPPhaseObservation phase{rows, sample.request.payloadBytes(rows), {}};
            for (const auto device : sample.request.devices()) phase.endpoints.push_back({device, 2, .1 * rows});
            observed.phases.push_back(std::move(phase));
        }
        if(sample.mode==PlanningLocalTPMeasurementMode::NativeCollective)
            for(const auto protocol:{PlanningNativeGatherProtocol::FixedAllgather,PlanningNativeGatherProtocol::CountedAllgather})
                for(const size_t bytes:{protocol==PlanningNativeGatherProtocol::FixedAllgather?size_t{1}:size_t{0},sample.request.activationBytes()}){
                    PlanningLocalTPGatherPhaseObservation phase{protocol,bytes,degree,{}};
                    const double floor=protocol==PlanningNativeGatherProtocol::FixedAllgather?.1:.2;
                    for(const auto device:sample.request.devices())phase.endpoints.push_back({device,std::max<size_t>(2,degree),bytes==sample.request.activationBytes()?floor*64:floor});
                    observed.gather_phases.push_back(std::move(phase));
                }
        if (sample.pair_sampling == PlanningPairSampling::Enabled)
        {
            auto &sampling = observed.pair_sampling;
            sampling.payload_anchors = sample.request.pairPayloadAnchors();
            sampling.direct_peer_access = sample.direct_peer_access;
            for (size_t source = 0; source < sample.uuids.size(); ++source)
                for (size_t destination = 0; destination < sample.uuids.size(); ++destination)
                    if (sampling.direct_peer_access[source * degree + destination])
                        for (const size_t bytes : sampling.payload_anchors)
                        {
                            // The same-switch-labelled {0,1} pair is slower than
                            // cross-switch-labelled {0,2}, with independent reverse service.
                            const double seconds = ((source == 0 && destination == 1 ? 10e-6 : 2e-6) + bytes * 1e-9) *
                                (sample.discovery_rank + 1);
                            sampling.directed_pairs.push_back({source, destination, bytes,
                                {seconds * .8, seconds, seconds * 1.2}, 4});
                        }
            sampling.setup_wall_seconds = .01; sampling.measurement_wall_seconds = .02;
            sampling.capture_count = sampling.directed_pairs.size() / sampling.payload_anchors.size();
            sampling.measured_launches = sampling.capture_count;
        }
        return {std::move(sample), std::move(observed)};
    }

    /** @return Complete physical four-GPU receipt with precision-independent directed samples. */
    PlanningNativeCollectiveService service(PlanningAllreducePrecision precision = PlanningAllreducePrecision::FP32,
        bool pairs = true)
    {
        return recordedSample({0, 0, {"gpu-0", "gpu-1", "gpu-2", "gpu-3"},
            PlanningLocalTPRequest({DeviceId::rocm(1), DeviceId::rocm(2), DeviceId::rocm(3), DeviceId::rocm(4)},
                16, 64, precision), pairs ? PlanningPairSampling::Enabled : PlanningPairSampling::Disabled});
    }

    /** @return Exact same-rank measured host legs, independent of physical peer-copy observations. */
    std::vector<PlanningHostDeviceObservations> hostServices(const ClusterInventory &cluster)
    {
        std::vector<PlanningHostDeviceObservations> result;
        for (const auto &gpu : cluster.ranks[0].gpus)
        {
            const DeviceId device{gpu.type, gpu.local_device_id};
            PlanningHostDeviceObservations observed{
                PlanningHostDeviceRequest::fromInventory(cluster, {0, device}, {0}, 4096), {}};
            for (auto mechanism : {PlanningHostTransferMechanism::DMA, PlanningHostTransferMechanism::MappedKernel})
                for (auto direction : {MappedTransferDirection::HostToDevice, MappedTransferDirection::DeviceToHost})
                    for (const size_t bytes : {size_t{1}, size_t{4096}})
                    {
                        const double seconds = gpu.local_device_id * 1e-6 + bytes * 1e-9 +
                            (direction == MappedTransferDirection::HostToDevice ? 3e-6 : 7e-6);
                        observed.phases.push_back({mechanism, direction, bytes, 1,
                            PlanningServiceObservation(PlanningWorkUnit::Bytes,
                                PlanningExecutionMeasurement::kTimedInvocations * double(bytes),
                                PlanningExecutionMeasurement::kTimedInvocations * seconds,
                                "authored physical-copy host leg")});
                    }
            result.push_back(std::move(observed));
        }
        return result;
    }
}

TEST(PlanningCommunicationCost, CommonNativePairBasisSeparatesPlacementFromCollectiveTiming)
{
    const auto receipt = service(); const std::array receipts{receipt};
    const PlanningCommunicationCost costs(inventory(), receipts, {});
    const std::array slow{DeviceId::rocm(1), DeviceId::rocm(2)};
    const std::array fast{DeviceId::rocm(1), DeviceId::rocm(3)};
    const auto slow_score = costs.physicalGroupPlacement(0, slow, 64);
    const auto fast_score = costs.physicalGroupPlacement(0, fast, 64);
    EXPECT_GT(slow_score.seconds, fast_score.seconds);
    EXPECT_EQ(slow_score.basis, PlanningCommunicationBasis::PhysicalGroupWorstPeerPlacementProxy);
    EXPECT_NE(slow_score.evidence.find("not collective latency"), std::string::npos);
    const auto full = costs.physicalGroupPlacement(0, receipt.sample.request.devices(), 64);
    EXPECT_DOUBLE_EQ(full.seconds, slow_score.seconds); // Same pair basis even for an exact measured group.
    const auto collective = costs.nativeAllreduce(0, receipt.sample.request.devices(), 16, 1,
        PlanningAllreducePrecision::FP32);
    EXPECT_DOUBLE_EQ(collective.seconds, .1);
    EXPECT_EQ(collective.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
    EXPECT_NE(collective.seconds, full.seconds);
    const auto reverse = costs.physicalPeerTransfer(0, slow[1], slow[0], 64);
    EXPECT_LT(reverse.seconds, slow_score.seconds);
    EXPECT_EQ(reverse.basis, PlanningCommunicationBasis::PhysicalDirectedPeerPlacementProxy);
}

TEST(PlanningCommunicationCost, NativePairPacketizationRetainsExactLiveTailWithoutExtrapolatedRate)
{
    const std::array receipts{service()}; const PlanningCommunicationCost costs(inventory(), receipts, {});
    const auto &anchors = receipts[0].observation.pair_sampling.payload_anchors;
    ASSERT_EQ(anchors.size(), 3u);
    const auto price = [&](size_t bytes) { return costs.physicalPeerTransfer(0, DeviceId::rocm(1), DeviceId::rocm(3), bytes); };
    EXPECT_DOUBLE_EQ(price(1).seconds, price(anchors.front()).seconds);
    EXPECT_NEAR(price(40).seconds, 2e-6 + 40e-9, 1e-18);
    EXPECT_DOUBLE_EQ(price(anchors.back() * 2).seconds, 2 * price(anchors.back()).seconds);
    const auto tail = price(anchors.back() * 2 + 7);
    EXPECT_DOUBLE_EQ(tail.seconds, 2 * price(anchors.back()).seconds + price(7).seconds);
    EXPECT_NE(tail.evidence.find("full-max-chunks=2"), std::string::npos);
    EXPECT_NE(tail.evidence.find("tail-live-bytes=7"), std::string::npos);
    EXPECT_NE(tail.evidence.find("not actual graph schedule"), std::string::npos);
    EXPECT_THROW(price(0), std::invalid_argument);
}

TEST(PlanningCommunicationCost, NativePairEvidenceCannotAliasAnotherDiscoveryRankOrTransport)
{
    const std::array receipts{service()}; const PlanningCommunicationCost costs(inventory(), receipts, {});
    const auto source = DeviceId::rocm(1), destination = DeviceId::rocm(2);
    // Rank1 sees the same physical UUIDs but supplied no native pair observations.
    EXPECT_THROW(costs.physicalPeerTransfer(1, source, destination, 64), std::invalid_argument);
    EXPECT_THROW(costs.physicalPeerTransfer(0, source, source, 64), std::invalid_argument);
    EXPECT_THROW(costs.physicalPeerTransfer(0, source, DeviceId::cuda(2), 64), std::invalid_argument);
    EXPECT_THROW(costs.physicalPeerTransfer(0, DeviceId::cpu(), destination, 64), std::invalid_argument);
    const std::array duplicate{source, source};
    EXPECT_THROW(costs.physicalGroupPlacement(0, duplicate, 64), std::invalid_argument);
    const std::array singleton{source};
    EXPECT_EQ(costs.physicalGroupPlacement(1, singleton, 64).seconds, 0);
    const std::array collective_only{service(PlanningAllreducePrecision::FP32, false)};
    const PlanningCommunicationCost missing(inventory(), collective_only, {});
    EXPECT_GT(missing.nativeAllreduce(0, receipts[0].sample.request.devices(), 16, 1,
        PlanningAllreducePrecision::FP32).seconds, 0);
    EXPECT_THROW(missing.physicalPeerTransfer(0, source, destination, 64), std::invalid_argument);
}

TEST(PlanningCommunicationCost, PairMatricesAreBoundOnceAcrossPrecisionsAndRejectSubstitution)
{
    const std::array receipts{service(), service(PlanningAllreducePrecision::FP16, false)};
    const PlanningCommunicationCost costs(inventory(), receipts, {});
    EXPECT_GT(costs.physicalPeerTransfer(0, DeviceId::rocm(1), DeviceId::rocm(2), 64).seconds, 0);
    auto duplicate = receipts; duplicate[1] = service(PlanningAllreducePrecision::FP16, true);
    EXPECT_THROW((void)PlanningCommunicationCost(inventory(), duplicate, {}), std::invalid_argument);
    auto stale = inventory(); stale.ranks[0].gpus[0].uuid = "replacement-device";
    EXPECT_THROW((void)PlanningCommunicationCost(stale, receipts, {}), std::invalid_argument);
    auto incomplete = receipts; incomplete[0].observation.pair_sampling.directed_pairs.pop_back();
    EXPECT_THROW((void)PlanningCommunicationCost(inventory(), incomplete, {}), std::invalid_argument);
}

/** @test Alternate eligible MPI owners retain independent native receipts, including subset visibility. */
TEST(PlanningCommunicationCost, OriginalRankMatricesRetainSameNUMAAliasesAndContainedOwnerGroups)
{
    const AutomaticOrchestrationRequest request({.only_backends = {{DeviceType::ROCm}},
        .only_strategies = {{OrchestrationStrategy::TensorParallel}}});
    constexpr std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
    for (const bool subset : {false, true})
    {
        SCOPED_TRACE(subset);
        auto cluster = inventory();
        if (subset) { cluster.ranks[1].gpus.resize(2); cluster.ranks[1].p2p_rocm_count = 2; cluster.ranks[1].p2p_rocm.assign(4, true); }
        cluster.buildNodeAggregations();
        const auto plan = PlanningCommunicationSamplePlan::resolve(cluster, request, 16, 64, precisions);
        ASSERT_EQ(plan.native().size(), 4u) << "Each eligible original owner requires its own two precisions";
        std::array<size_t, 2> enabled{}, disabled{};
        std::vector<PlanningNativeCollectiveService> receipts;
        for (const auto &sample : plan.native())
        {
            ASSERT_GE(sample.discovery_rank, 0); ASSERT_LT(sample.discovery_rank, 2);
            EXPECT_EQ(sample.request.devices().size(), subset && sample.discovery_rank == 1 ? 2u : 4u);
            if (sample.pair_sampling == PlanningPairSampling::Enabled) ++enabled[sample.discovery_rank];
            else ++disabled[sample.discovery_rank];
            receipts.push_back(recordedSample(sample));
        }
        EXPECT_EQ(enabled, (std::array<size_t, 2>{1, 1}));
        EXPECT_EQ(disabled, (std::array<size_t, 2>{1, 1}));
        const PlanningCommunicationCost costs(cluster, receipts, {});
        const auto first = costs.physicalPeerTransfer(0, DeviceId::rocm(1), DeviceId::rocm(2), 64);
        const auto second = costs.physicalPeerTransfer(1, DeviceId::rocm(1), DeviceId::rocm(2), 64);
        EXPECT_DOUBLE_EQ(second.seconds, 2 * first.seconds);
        EXPECT_NE(first.evidence.find("observer=0"), std::string::npos);
        EXPECT_NE(second.evidence.find("observer=1"), std::string::npos);
        const std::array group{DeviceId::rocm(1), DeviceId::rocm(2)};
        EXPECT_DOUBLE_EQ(costs.physicalGroupPlacement(1, group, 64).seconds, second.seconds);
        std::erase_if(receipts, [](const auto &service) { return service.sample.discovery_rank == 1; });
        const PlanningCommunicationCost missing(cluster, receipts, {});
        EXPECT_THROW(missing.physicalGroupPlacement(1, group, 64), std::invalid_argument);
        EXPECT_GT(missing.physicalGroupPlacement(0, group, 64).seconds, 0);
    }
}

/** @test Destination-read eligibility preserves direction and prices absent peers with both measured host legs. */
TEST(PlanningCommunicationCost, PhysicalPeerDirectionAndHostCompositionRemainDistinct)
{
    auto cluster = inventory();
    for (auto &rank : cluster.ranks)
    {
        rank.p2p_rocm.assign(16, false);
        rank.p2p_rocm[1 * 4 + 0] = true; // Destination1 can read source0; data0->1 only.
    }
    const auto original = service();
    const auto access = planningDirectedPeerCopyAccess(cluster.ranks[0], original.sample.request.devices());
    ASSERT_EQ(access[1], 1u);
    ASSERT_EQ(access[4], 0u);
    const std::array receipts{recordedSample(original.sample, &access)};
    const auto host = hostServices(cluster);
    const PlanningCommunicationCost costs(cluster, receipts, {}, host);
    const auto forward = costs.physicalPeerTransfer(0, DeviceId::rocm(1), DeviceId::rocm(2), 64);
    const auto reverse = costs.physicalPeerTransfer(0, DeviceId::rocm(2), DeviceId::rocm(1), 64);
    EXPECT_EQ(forward.basis, PlanningCommunicationBasis::PhysicalDirectedPeerPlacementProxy);
    EXPECT_EQ(reverse.basis, PlanningCommunicationBasis::PhysicalHostStagedPlacementProxy);
    const double expected = costs.hostTransfer(0, DeviceId::rocm(2), 0, PlanningHostTransferMechanism::DMA,
        MappedTransferDirection::DeviceToHost, 64).seconds +
        costs.hostTransfer(0, DeviceId::rocm(1), 0, PlanningHostTransferMechanism::DMA,
            MappedTransferDirection::HostToDevice, 64).seconds;
    EXPECT_DOUBLE_EQ(reverse.seconds, expected);
    EXPECT_NE(reverse.evidence.find("source D2H then destination H2D"), std::string::npos);
    const std::array group{DeviceId::rocm(1), DeviceId::rocm(2)};
    EXPECT_DOUBLE_EQ(costs.physicalGroupPlacement(0, group, 64).seconds, std::max(forward.seconds, reverse.seconds));
    const PlanningCommunicationCost missing(cluster, receipts, {});
    EXPECT_GT(missing.physicalPeerTransfer(0, group[0], group[1], 64).seconds, 0);
    EXPECT_THROW(missing.physicalPeerTransfer(0, group[1], group[0], 64), std::invalid_argument);
    auto partial_host = host;
    std::erase_if(partial_host, [](const auto &observation) { return observation.request.device() == DeviceId::rocm(1); });
    const PlanningCommunicationCost missing_leg(cluster, receipts, {}, partial_host);
    EXPECT_THROW(missing_leg.physicalPeerTransfer(0, group[1], group[0], 64), std::invalid_argument);
    EXPECT_THROW(costs.physicalPeerTransfer(0, group[0], group[1], 0), std::invalid_argument);
    EXPECT_THROW(costs.physicalPeerTransfer(1, group[0], group[1], 64), std::invalid_argument);
    EXPECT_THROW((void)PlanningCommunicationCost(inventory(), receipts, {}, host), std::invalid_argument);
    auto substituted = receipts;
    substituted[0].observation.pair_sampling.direct_peer_access[1] = 0;
    EXPECT_THROW((void)PlanningCommunicationCost(cluster, substituted, {}, host), std::invalid_argument);
}

/** @test An explicit empty peer mask is valid, but neither missing host evidence nor fabricated direct samples are valid. */
TEST(PlanningCommunicationCost, EmptyPhysicalPeerMaskRequiresMeasuredHostService)
{
    auto cluster = inventory();
    for (auto &rank : cluster.ranks) rank.p2p_rocm.assign(16, false);
    const auto original = service();
    const auto access = planningDirectedPeerCopyAccess(cluster.ranks[0], original.sample.request.devices());
    const std::array receipts{recordedSample(original.sample, &access)};
    ASSERT_TRUE(receipts[0].observation.pair_sampling.directed_pairs.empty());
    const auto host = hostServices(cluster);
    const PlanningCommunicationCost costs(cluster, receipts, {}, host);
    const auto price = costs.physicalGroupPlacement(0, original.sample.request.devices(), 64);
    EXPECT_GT(price.seconds, 0);
    EXPECT_NE(price.evidence.find("composed same-rank-host DMA"), std::string::npos);
    const PlanningCommunicationCost missing(cluster, receipts, {});
    EXPECT_THROW(missing.physicalGroupPlacement(0, original.sample.request.devices(), 64), std::invalid_argument);
    auto fabricated = receipts;
    fabricated[0].observation.pair_sampling.directed_pairs.push_back({0, 1, 16, {1e-6, 1e-6, 1e-6}, 4});
    EXPECT_THROW((void)PlanningCommunicationCost(cluster, fabricated, {}, host), std::invalid_argument);
}

/** @test Native control latency survives faster subset links; only measured payload excess is calibrated. */
TEST(PlanningCommunicationCost, CalibratedAllreduceRetainsNativeFloorAndExactGroupCurve)
{
    const auto receipt=service();const std::array receipts{receipt};const auto cluster=inventory();
    const PlanningCommunicationCost cost(cluster,receipts,{});
    const auto &full=receipt.sample.request.devices();const std::array fast{DeviceId::rocm(1),DeviceId::rocm(3)};
    for(const int rows:{1,32,64}){
        EXPECT_DOUBLE_EQ(cost.calibratedNativeAllreduce(0,full,16,rows,PlanningAllreducePrecision::FP32).seconds,
            cost.nativeAllreduce(0,full,16,rows,PlanningAllreducePrecision::FP32).seconds);
    }
    const auto small=cost.calibratedNativeAllreduce(0,fast,16,1,PlanningAllreducePrecision::FP32);
    EXPECT_DOUBLE_EQ(small.seconds,.1); // Native minimum-payload latency cannot be discounted.
    const auto large=cost.calibratedNativeAllreduce(0,fast,16,32,PlanningAllreducePrecision::FP32);
    const double expected=.1+(3.2-.1)*(2e-6+2048e-9)/(10e-6+2048e-9);
    EXPECT_DOUBLE_EQ(large.seconds,expected);EXPECT_GT(large.seconds,.1);EXPECT_LT(large.seconds,3.2);
    EXPECT_EQ(large.basis,PlanningCommunicationBasis::CalibratedNativeAllreduceProxy);
}

/** @test Equal link service gives no degree discount, and another original observer cannot lend native evidence. */
TEST(PlanningCommunicationCost, CalibratedAllreduceKeepsDegreeAndOriginalObserverAuthority)
{
    auto receipt=service();for(auto &edge:receipt.observation.pair_sampling.directed_pairs)
        edge.seconds_per_transfer.assign(PlanningLocalTPPairSampling::kSamples,1e-5);
    const std::array receipts{receipt};const PlanningCommunicationCost cost(inventory(),receipts,{});
    const std::array subset{DeviceId::rocm(1),DeviceId::rocm(3)};
    EXPECT_DOUBLE_EQ(cost.calibratedNativeAllreduce(0,subset,16,32,PlanningAllreducePrecision::FP32).seconds,3.2);
    EXPECT_THROW(cost.calibratedNativeAllreduce(1,subset,16,32,PlanningAllreducePrecision::FP32),std::invalid_argument);
    const std::array singleton{DeviceId::rocm(1)};
    EXPECT_DOUBLE_EQ(cost.calibratedNativeAllreduce(0,singleton,16,1,PlanningAllreducePrecision::FP32).seconds,0.);
}

/** @test Negotiated cut facts contradicting a measured reference interval are rejected instead of repaired. */
TEST(PlanningCommunicationCost, CalibratedAllreduceRejectsReferenceBelowObservedWireFloor)
{
    auto cluster=inventory();for(auto &rank:cluster.ranks)for(size_t i=0;i<rank.gpus.size();++i){auto &gpu=rank.gpus[i];
        gpu.pci_bus_address="0000:"+std::to_string(20+i)+":00.0";
        gpu.pcie_bottleneck_bdf=i<2?"0000:10:00.0":"0000:11:00.0";
        gpu.pcie_upstream_bdfs={gpu.pcie_bottleneck_bdf,"0000:01:00.0"};gpu.pcie_speed_gts=2.5;gpu.pcie_width=1;}
    cluster.buildNodeAggregations();auto receipt=service();
    for(auto &phase:receipt.observation.phases)for(auto &endpoint:phase.endpoints)endpoint.seconds_per_collective=1e-12*phase.rows;
    const std::array receipts{receipt};const PlanningCommunicationCost cost(cluster,receipts,{});
    EXPECT_THROW(cost.calibratedNativeAllreduce(0,receipt.sample.request.devices(),16,64,PlanningAllreducePrecision::FP32),std::invalid_argument);
}

/** @test Actual whole-group native protocols retain their own control floors, without adding a second header charge. */
TEST(PlanningCommunicationCost, NativeGatherProtocolsKeepMeasuredUniformCostAndZeroOwnerControl)
{
    const auto receipt=service(PlanningAllreducePrecision::FP32,false);const std::array receipts{receipt};
    const PlanningCommunicationCost cost(inventory(),receipts,{});const auto &devices=receipt.sample.request.devices();
    const std::array<size_t,4> zero{},bulk{4096,4096,4096,4096};
    // Exact measured groups need no independently sampled physical-copy basis.
    EXPECT_DOUBLE_EQ(cost.calibratedNativeAllgather(0,devices,1).seconds,.1);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeAllgather(0,devices,4096).seconds,6.4);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeCountedAllgather(0,devices,zero).seconds,.2);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeCountedAllgather(0,devices,bulk).seconds,12.8);
    EXPECT_THROW(cost.calibratedNativeAllgather(0,devices,0),std::invalid_argument);
    const std::array<size_t,2> short_shape{1,2};
    EXPECT_THROW(cost.calibratedNativeCountedAllgather(0,devices,short_shape),std::invalid_argument);
    EXPECT_THROW(cost.calibratedNativeCountedAllgather(1,devices,zero),std::invalid_argument);
}

/** @test Unequal owners each retain measured control, and their exact live fragments contribute once per root sequence. */
TEST(PlanningCommunicationCost, NativeGatherUnequalAndEmptyOwnersUseWholeGroupReceiptCoordinates)
{
    const auto receipt=service();const std::array receipts{receipt};const PlanningCommunicationCost cost(inventory(),receipts,{});
    const auto &devices=receipt.sample.request.devices();const std::array<size_t,4> owners{17,0,31,0},permuted{0,31,0,17};
    const double expected=.2+12.6*48/(4.*4096);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeCountedAllgather(0,devices,owners).seconds,expected);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeCountedAllgather(0,devices,permuted).seconds,expected);
}

/** @test Contained-group estimates use bulk measured fabric pressure; tiny peer noise cannot become native control latency. */
TEST(PlanningCommunicationCost, NativeGatherSubgroupKeepsNativeFloorsAndBulkMeasuredLinkPreference)
{
    auto cluster=inventory();for(auto &rank:cluster.ranks)for(size_t i=0;i<rank.gpus.size();++i)
        rank.gpus[i].pcie_upstream_bdfs={i<2?"0000:86:00.0":"0000:af:00.0"};
    const auto receipt=service();const std::array receipts{receipt};const PlanningCommunicationCost cost(cluster,receipts,{});
    const std::array same{DeviceId::rocm(1),DeviceId::rocm(2)},cross{DeviceId::rocm(1),DeviceId::rocm(3)};
    const std::array<size_t,2> zero{},bulk{4096,4096};
    EXPECT_DOUBLE_EQ(cost.calibratedNativeAllgather(0,cross,1).seconds,.1);
    EXPECT_DOUBLE_EQ(cost.calibratedNativeCountedAllgather(0,cross,zero).seconds,.1); // Two of four measured root sequences.
    EXPECT_LT(cost.calibratedNativeCountedAllgather(0,cross,bulk).seconds,cost.calibratedNativeCountedAllgather(0,same,bulk).seconds);
    const auto estimate=cost.calibratedNativeAllgather(0,cross,4096);
    EXPECT_NE(estimate.evidence.find("estimate"),std::string::npos);
    auto noisy=receipt;for(auto &edge:noisy.observation.pair_sampling.directed_pairs)
        if(edge.payload_bytes!=noisy.sample.request.activationBytes())edge.seconds_per_transfer.assign(PlanningLocalTPPairSampling::kSamples,100.);
    const std::array changed{noisy};const PlanningCommunicationCost noisy_cost(cluster,changed,{});
    EXPECT_DOUBLE_EQ(noisy_cost.calibratedNativeAllgather(0,cross,4096).seconds,estimate.seconds);
    EXPECT_DOUBLE_EQ(noisy_cost.calibratedNativeCountedAllgather(0,cross,bulk).seconds,
        cost.calibratedNativeCountedAllgather(0,cross,bulk).seconds);
}

/** @test Physical peer measurements or another native protocol cannot substitute for actual native gather observations. */
TEST(PlanningCommunicationCost, NativeGatherMissingProtocolCannotUsePairCopiesAsLatency)
{
    const auto original=service();const std::array<size_t,4> owners{};
    for(const auto absent:{PlanningNativeGatherProtocol::FixedAllgather,PlanningNativeGatherProtocol::CountedAllgather}){
        auto receipt=original;std::erase_if(receipt.observation.gather_phases,[&](const auto &phase){return phase.protocol==absent;});
        const std::array receipts{receipt};
        EXPECT_THROW(([&]{const PlanningCommunicationCost cost(inventory(),receipts,{});
            if(absent==PlanningNativeGatherProtocol::FixedAllgather)(void)cost.calibratedNativeAllgather(0,receipt.sample.request.devices(),1);
            else (void)cost.calibratedNativeCountedAllgather(0,receipt.sample.request.devices(),owners);}()),std::invalid_argument);
    }
    auto physical=original;physical.sample.mode=physical.observation.mode=PlanningLocalTPMeasurementMode::PhysicalPeersOnly;
    physical.observation.phases.clear();physical.observation.gather_phases.clear();const std::array receipts{physical};
    const PlanningCommunicationCost cost(inventory(),receipts,{});
    EXPECT_THROW(cost.calibratedNativeCountedAllgather(0,physical.sample.request.devices(),owners),std::invalid_argument);
    EXPECT_THROW(cost.calibratedNativeAllgather(0,physical.sample.request.devices(),1),std::invalid_argument);
}

/** @test Calibrating excess cannot push a candidate below its own known physical wire floor. */
TEST(PlanningCommunicationCost, CalibratedAllreduceClampsCandidateToItsOwnCutFloor)
{
    auto cluster=inventory();for(auto &rank:cluster.ranks)for(size_t i=0;i<rank.gpus.size();++i){auto &gpu=rank.gpus[i];
        gpu.pci_bus_address="0000:"+std::to_string(20+i)+":00.0";
        gpu.pcie_bottleneck_bdf=i<2?"0000:10:00.0":"0000:11:00.0";
        gpu.pcie_upstream_bdfs={gpu.pcie_bottleneck_bdf,"0000:01:00.0"};gpu.pcie_speed_gts=2.5;gpu.pcie_width=1;}
    cluster.buildNodeAggregations();auto receipt=service();
    for(auto &phase:receipt.observation.phases)for(auto &endpoint:phase.endpoints)
        endpoint.seconds_per_collective=phase.rows==1?1e-6:21e-6;
    for(auto &edge:receipt.observation.pair_sampling.directed_pairs)
        edge.seconds_per_transfer.assign(PlanningLocalTPPairSampling::kSamples,
            edge.source_index==0 && edge.destination_index==1?.001:1e-9);
    const std::array receipts{receipt};const PlanningCommunicationCost cost(cluster,receipts,{});
    const std::array subset{DeviceId::rocm(1),DeviceId::rocm(3)};
    const auto result=cost.calibratedNativeAllreduce(0,subset,16,32,PlanningAllreducePrecision::FP32);
    EXPECT_DOUBLE_EQ(result.seconds,2048./(2.5e9*.8/8.));
}
