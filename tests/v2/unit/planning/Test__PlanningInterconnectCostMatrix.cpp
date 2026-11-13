/**
 * @file Test__PlanningInterconnectCostMatrix.cpp
 * @brief Device-free recorded physical-peer placement cost contracts.
 *
 * Authored raw observations exercise the actual pure matrix implementation;
 * these tests neither benchmark hardware nor certify collective performance.
 * Schedules are explicit test graphs, never claims about NCCL/RCCL algorithms.
 */
#include "planning/PlanningInterconnectCostMatrix.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;
namespace
{
    /** @return One original NUMA-local observer with arbitrary physical GPU count. */
    PlanningInterconnectIdentity identity(size_t count = 4)
    {
        PlanningInterconnectIdentity result{1, 0, DeviceType::ROCm, {}};
        for (size_t index = 0; index < count; ++index) result.uuids.push_back("gpu-" + std::to_string(index));
        return result;
    }

    /** @return Explicit source-major full direct-peer mask, excluding self edges. */
    std::vector<uint8_t> mask(size_t count = 4)
    {
        std::vector<uint8_t> result(count * count, 1);
        for (size_t index = 0; index < count; ++index) result[index * count + index] = 0;
        return result;
    }

    /** @return Complete directed observations, preserving all three completed raw intervals. */
    std::vector<PlanningLocalTPPairObservation> observations(size_t count = 4)
    {
        std::vector<PlanningLocalTPPairObservation> result;
        for (size_t source = 0; source < count; ++source)
            for (size_t destination = 0; destination < count; ++destination)
                if (source != destination)
                    for (const size_t bytes : {size_t{16}, size_t{1024}})
                        result.push_back({source, destination, bytes, {0.8e-6, 1e-6, 1.2e-6}, 2});
        return result;
    }

    /** @brief Set a recorded direction's anchor mean without assuming reverse symmetry. */
    void set(std::vector<PlanningLocalTPPairObservation> &samples, size_t source, size_t destination,
        size_t bytes, double seconds)
    {
        for (auto &sample : samples)
            if (sample.source_index == source && sample.destination_index == destination && sample.payload_bytes == bytes)
                sample.seconds_per_transfer = {seconds * 0.8, seconds, seconds * 1.2};
    }

    /** @return Explicit full-duplex cycle for one occurrence of a phase, supporting odd counts. */
    PlanningInterconnectPhaseDemand cycle(const std::vector<std::string> &group, size_t bytes, uint64_t occurrences = 1)
    {
        PlanningInterconnectRound round;
        for (size_t index = 0; index < group.size(); ++index)
            round.push_back({group[index], group[(index + 1) % group.size()], bytes});
        return {{std::move(round)}, occurrences};
    }
}

TEST(PlanningInterconnectCostMatrix, RecordedSlowerSameSwitchDoesNotBeatFasterCrossSwitch)
{
    const auto physical = identity(); auto samples = observations();
    // The authored topology calls {0,1} same-switch and {0,2} cross-switch.
    // Topology labels are deliberately not inputs to measured cost arithmetic.
    for (const auto bytes : {size_t{16}, size_t{1024}})
        for (const auto direction : {std::pair{size_t{0}, size_t{1}}, std::pair{size_t{1}, size_t{0}}})
            set(samples, direction.first, direction.second, bytes, 9e-6);
    const PlanningInterconnectCostMatrix matrix(physical, samples, mask());
    const std::vector<std::string> same{physical.uuids[0], physical.uuids[1]}, cross{physical.uuids[0], physical.uuids[2]};
    const auto slow = matrix.groupWorkloadScore(physical, same, {cycle(same, 1024, 3), cycle(same, 16, 9)});
    const auto fast = matrix.groupWorkloadScore(physical, cross, {cycle(cross, 1024, 3), cycle(cross, 16, 9)});
    EXPECT_GT(slow.seconds(), fast.seconds());
    EXPECT_NEAR(fast.prefill_seconds, 3e-6, 1e-18);
    EXPECT_NEAR(fast.decode_seconds, 9e-6, 1e-18);
    EXPECT_EQ(matrix.observations()[0].seconds_per_transfer, samples[0].seconds_per_transfer);
}

TEST(PlanningInterconnectCostMatrix, DirectedPayloadCurvesAndWorkloadRegimesRemainIndependent)
{
    const auto physical = identity(); auto samples = observations();
    for (const auto pair : {std::pair{size_t{0}, size_t{1}}, std::pair{size_t{1}, size_t{0}}})
    {
        set(samples, pair.first, pair.second, 16, 1e-6);
        set(samples, pair.first, pair.second, 1024, 10e-6);
    }
    for (const auto pair : {std::pair{size_t{2}, size_t{3}}, std::pair{size_t{3}, size_t{2}}})
    {
        set(samples, pair.first, pair.second, 16, 5e-6);
        set(samples, pair.first, pair.second, 1024, 2e-6);
    }
    set(samples, 0, 2, 16, 7e-6);
    const PlanningInterconnectCostMatrix matrix(physical, samples, mask());
    EXPECT_NEAR(matrix.directedService(physical, "gpu-0", "gpu-2", 16).seconds, 7e-6, 1e-18);
    EXPECT_NEAR(matrix.directedService(physical, "gpu-2", "gpu-0", 16).seconds, 1e-6, 1e-18);
    const auto midpoint = matrix.directedService(physical, "gpu-0", "gpu-1", 520);
    EXPECT_NEAR(midpoint.seconds, 5.5e-6, 1e-18);
    EXPECT_EQ(midpoint.lower_payload_bytes, 16u); EXPECT_EQ(midpoint.upper_payload_bytes, 1024u);
    const std::vector<std::string> a{"gpu-0", "gpu-1"}, b{"gpu-2", "gpu-3"};
    const auto score = [&](const auto &group, uint64_t prefill, uint64_t decode) {
        return matrix.groupWorkloadScore(physical, group, {cycle(group, 1024, prefill), cycle(group, 16, decode)}).seconds();
    };
    EXPECT_LT(score(a, 1, 100), score(b, 1, 100));
    EXPECT_GT(score(a, 100, 1), score(b, 100, 1));
}

TEST(PlanningInterconnectCostMatrix, ExplicitRoundsSupportSingletonOddUnequalAndLargeGroups)
{
    const auto physical = identity(11); const PlanningInterconnectCostMatrix matrix(physical, observations(11), mask(11));
    for (const size_t count : {size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{8}, size_t{11}})
    {
        SCOPED_TRACE(count);
        const std::vector<std::string> group(physical.uuids.begin(), physical.uuids.begin() + count);
        auto phase = cycle(group, 16, 3); phase.rounds.push_back(phase.rounds.front());
        EXPECT_NEAR(matrix.groupWorkloadScore(physical, group, {{}, phase}).seconds(), 6e-6, 1e-18);
    }
    const auto single = identity(1); const PlanningInterconnectCostMatrix singleton(single, {}, mask(1));
    EXPECT_EQ(singleton.groupWorkloadScore(single, single.uuids, {}).seconds(), 0);
    EXPECT_THROW(singleton.directedService(single, "gpu-0", "gpu-0", 16), std::invalid_argument);
    // Disjoint stage widths need no shared degree or power-of-two assumption.
    const std::vector<std::string> three(physical.uuids.begin(), physical.uuids.begin() + 3);
    const std::vector<std::string> eight(physical.uuids.begin() + 3, physical.uuids.end());
    EXPECT_EQ(matrix.groupWorkloadScore(physical, three, {{}, cycle(three, 16)}).seconds(),
        matrix.groupWorkloadScore(physical, eight, {{}, cycle(eight, 16)}).seconds());
}

TEST(PlanningInterconnectCostMatrix, MissingStaleDuplicateAndIncompatibleEvidenceFails)
{
    const auto physical = identity(); const auto samples = observations();
    for (int defect = 0; defect < 8; ++defect)
    {
        SCOPED_TRACE(defect); auto broken = samples;
        if (defect == 0) broken.erase(broken.begin());
        if (defect == 1) broken.push_back(broken.front());
        if (defect == 2) broken.front().source_index = physical.uuids.size();
        if (defect == 3) broken.front().seconds_per_transfer[0] = std::numeric_limits<double>::quiet_NaN();
        if (defect == 4) broken.front().seconds_per_transfer.pop_back();
        if (defect == 5) broken.front().graph_nodes = 0;
        if (defect == 6) broken.front().payload_bytes = 0;
        if (defect == 7) broken.front().seconds_per_transfer[0] = 0;
        EXPECT_THROW((void)PlanningInterconnectCostMatrix(physical, broken, mask()), std::invalid_argument);
    }
    const PlanningInterconnectCostMatrix matrix(physical, samples, mask());
    for (int defect = 0; defect < 5; ++defect)
    {
        auto stale = physical;
        if (defect == 0) ++stale.discovery_rank;
        if (defect == 1) ++stale.physical_node;
        if (defect == 2) stale.backend = DeviceType::CUDA;
        if (defect == 3) std::swap(stale.uuids[0], stale.uuids[1]);
        if (defect == 4) stale.uuids[0] = "replaced-card";
        EXPECT_THROW(matrix.directedService(stale, "gpu-0", "gpu-1", 16), std::invalid_argument);
    }
    auto cpu = physical; cpu.backend = DeviceType::CPU;
    EXPECT_THROW((void)PlanningInterconnectCostMatrix(cpu, samples, mask()), std::invalid_argument);
    for (size_t bytes : {size_t{0}, size_t{15}, size_t{1025}})
        EXPECT_THROW(matrix.directedService(physical, "gpu-0", "gpu-1", bytes), std::invalid_argument);
    EXPECT_THROW(matrix.directedService(physical, "gpu-0", "absent", 16), std::invalid_argument);
}

TEST(PlanningInterconnectCostMatrix, IncompatiblePartialSchedulesNeverBecomeZeroCostPlacement)
{
    const auto physical = identity(); const PlanningInterconnectCostMatrix matrix(physical, observations(), mask());
    const std::vector<std::string> group{"gpu-0", "gpu-1", "gpu-2"};
    EXPECT_THROW(matrix.groupWorkloadScore(physical, group, {}), std::invalid_argument);
    auto workload = PlanningInterconnectWorkload{{}, cycle(group, 16)};
    workload.decode.rounds.front().push_back({"gpu-0", "gpu-2", 16});
    EXPECT_THROW(matrix.groupWorkloadScore(physical, group, workload), std::invalid_argument);
    workload.decode = cycle(group, 16); workload.decode.rounds.front()[0].destination_uuid = "gpu-3";
    EXPECT_THROW(matrix.groupWorkloadScore(physical, group, workload), std::invalid_argument);
    workload.decode = cycle(group, 16); workload.decode.occurrences = 0;
    EXPECT_THROW(matrix.groupWorkloadScore(physical, group, workload), std::invalid_argument);
    workload.decode = cycle(group, 16); workload.decode.rounds.push_back({});
    EXPECT_THROW(matrix.groupWorkloadScore(physical, group, workload), std::invalid_argument);
    const std::vector<std::string> duplicate{"gpu-0", "gpu-0"};
    EXPECT_THROW(matrix.groupWorkloadScore(physical, duplicate, workload), std::invalid_argument);
}

/** @test Only observed destination-can-read-source edges admit physical peer-copy curves. */
TEST(PlanningInterconnectCostMatrix, ExplicitAsymmetricAndEmptyPeerMasksControlCoverage)
{
    const auto physical = identity(3);
    std::vector<uint8_t> access(9, 0); access[1] = 1; // Data0->1, destination1 reads source0.
    const std::vector<PlanningLocalTPPairObservation> samples{
        {0, 1, 16, {1e-6, 1e-6, 1e-6}, 3}, {0, 1, 1024, {2e-6, 2e-6, 2e-6}, 3}};
    const PlanningInterconnectCostMatrix matrix(physical, samples, access);
    EXPECT_TRUE(matrix.hasDirectPeer("gpu-0", "gpu-1"));
    EXPECT_FALSE(matrix.hasDirectPeer("gpu-1", "gpu-0"));
    EXPECT_GT(matrix.directedService(physical, "gpu-0", "gpu-1", 16).seconds, 0);
    EXPECT_THROW(matrix.directedService(physical, "gpu-1", "gpu-0", 16), std::invalid_argument);
    const PlanningInterconnectCostMatrix none(physical, {}, std::vector<uint8_t>(9, 0));
    EXPECT_FALSE(none.hasDirectPeer("gpu-0", "gpu-1"));
    EXPECT_THROW(none.directedService(physical, "gpu-0", "gpu-1", 16), std::invalid_argument);
    auto incomplete = samples; incomplete.clear();
    EXPECT_THROW((void)PlanningInterconnectCostMatrix(physical, incomplete, access), std::invalid_argument);
    auto extra = samples; extra.push_back({1, 0, 16, {1e-6, 1e-6, 1e-6}, 3});
    EXPECT_THROW((void)PlanningInterconnectCostMatrix(physical, extra, access), std::invalid_argument);
    for (int defect = 0; defect < 3; ++defect)
    {
        auto broken = access;
        if (defect == 0) broken.pop_back();
        if (defect == 1) broken[0] = 1;
        if (defect == 2) broken[1] = 2;
        EXPECT_THROW((void)PlanningInterconnectCostMatrix(physical, samples, broken), std::invalid_argument);
    }
}

/** @test A single finite scheduling outlier remains visible but does not dominate typical-link selection. */
TEST(PlanningInterconnectCostMatrix, MedianObservedServiceRetainsRawOutlierWithoutChangingSampleBudget)
{
    const auto physical = identity();
    auto samples = observations();
    for (auto &sample : samples)
    {
        if ((sample.source_index == 0 && sample.destination_index == 1) ||
            (sample.source_index == 1 && sample.destination_index == 0))
            sample.seconds_per_transfer = {6e-6, 1000e-6, 5e-6};
        if ((sample.source_index == 2 && sample.destination_index == 3) ||
            (sample.source_index == 3 && sample.destination_index == 2))
            sample.seconds_per_transfer = {8e-6, 8e-6, 8e-6};
    }
    const PlanningInterconnectCostMatrix matrix(physical, samples, mask());
    const auto price = matrix.directedService(physical, "gpu-0", "gpu-1", 16);
    EXPECT_DOUBLE_EQ(price.seconds, 6e-6);
    EXPECT_DOUBLE_EQ(matrix.directedService(physical, "gpu-0", "gpu-1", 520).seconds, 6e-6);
    const std::vector<std::string> typical{"gpu-0", "gpu-1"}, slower{"gpu-2", "gpu-3"};
    EXPECT_LT(matrix.groupWorkloadScore(physical, typical, {cycle(typical, 1024), cycle(typical, 16, 10)}).seconds(),
        matrix.groupWorkloadScore(physical, slower, {cycle(slower, 1024), cycle(slower, 16, 10)}).seconds());
    EXPECT_EQ(matrix.observations().front().seconds_per_transfer, (std::vector<double>{6e-6, 1000e-6, 5e-6}));
    // Replaying identical observations changes neither ranking nor raw evidence.
    const PlanningInterconnectCostMatrix replay(physical, matrix.observations(), mask());
    EXPECT_DOUBLE_EQ(replay.directedService(physical, "gpu-0", "gpu-1", 16).seconds, price.seconds);
    for (int defect = 0; defect < 3; ++defect)
    {
        auto invalid = samples;
        if (defect == 0) invalid[0].seconds_per_transfer[1] = std::numeric_limits<double>::infinity();
        if (defect == 1) invalid[0].seconds_per_transfer.pop_back();
        if (defect == 2) invalid[0].seconds_per_transfer[1] = 0;
        EXPECT_THROW((void)PlanningInterconnectCostMatrix(physical, invalid, mask()), std::invalid_argument);
    }
}
