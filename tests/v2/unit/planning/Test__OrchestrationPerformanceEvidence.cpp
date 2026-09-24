/**
 * @file Test__OrchestrationPerformanceEvidence.cpp
 * @brief Device-free regressions for evidence validation and critical-path pricing.
 *
 * All rates below are explicitly synthetic mathematical fixtures, not hardware
 * measurements. The independent discrete-event oracle checks pipeline fill,
 * steady-state service and partial drains without sharing the production closed
 * form. Unit tests issue no MPI, accelerator, allocator, or model payload calls.
 */
#include "planning/OrchestrationPerformanceEvidence.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @return Completed synthetic byte-service observation, in base units per second. */
    PlanningServiceObservation bytes(double rate)
    {
        return {PlanningWorkUnit::Bytes, rate, 1.0, "synthetic completed byte sample"};
    }
    /** @return Completed synthetic arithmetic observation, with one FMA counted as two. */
    PlanningServiceObservation operations(double rate)
    {
        return {PlanningWorkUnit::ArithmeticOperations, rate, 1.0, "synthetic completed arithmetic sample"};
    }
    /**
     * @brief Independent FIFO event simulation for a small chunked path.
     * @return Last completion time after explicitly issuing each chunk through every leg.
     */
    double simulate(const std::vector<PlanningTransferLeg> &legs,
        std::size_t payload, std::size_t chunk_bytes)
    {
        std::vector<double> free_at(legs.size(), 0.0);
        if (payload == 0)
        {
            double result = 0.0;
            for (const auto &leg : legs) result += leg.seconds(0);
            return result;
        }
        while (payload != 0)
        {
            const auto next = std::min(payload, chunk_bytes);
            double arrival = 0.0;
            for (std::size_t stage = 0; stage < legs.size(); ++stage)
            {
                free_at[stage] = std::max(arrival, free_at[stage]) + legs[stage].seconds(next);
                arrival = free_at[stage];
            }
            payload -= next;
        }
        return free_at.back();
    }
}

TEST(OrchestrationPerformanceEvidence, IncompleteObservationsHaveNoDefaultOrZeroRate)
{
    static_assert(!std::is_default_constructible_v<PlanningServiceObservation>);
    static_assert(!std::is_default_constructible_v<PlanningKernelService>);
    static_assert(!std::is_default_constructible_v<PlanningTransferLeg>);
    static_assert(!std::is_default_constructible_v<PlanningChunkPipeline>);
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(PlanningServiceObservation(PlanningWorkUnit::Bytes, invalid, 1.0, "fixture"), std::invalid_argument);
        EXPECT_THROW(PlanningServiceObservation(PlanningWorkUnit::Bytes, 1.0, invalid, "fixture"), std::invalid_argument);
    }
    EXPECT_THROW(PlanningServiceObservation(PlanningWorkUnit::Bytes, 1.0, 1.0, " \t\r\n\f\v"), std::invalid_argument);
    EXPECT_THROW(PlanningServiceObservation(static_cast<PlanningWorkUnit>(99), 1.0, 1.0, "fixture"), std::invalid_argument);
    const auto largest = std::numeric_limits<double>::max();
    const auto smallest = std::numeric_limits<double>::denorm_min();
    EXPECT_THROW(PlanningServiceObservation(PlanningWorkUnit::Bytes, largest, smallest, "overflow"), std::invalid_argument);
    EXPECT_THROW(PlanningServiceObservation(PlanningWorkUnit::Bytes, smallest, largest, "underflow"), std::invalid_argument);
}

TEST(OrchestrationPerformanceEvidence, UnitsAndProvenanceRemainIndivisible)
{
    const PlanningServiceObservation sample(PlanningWorkUnit::Bytes, 8e6, .002, "rank=3 ROCm=1 sample=7");
    EXPECT_DOUBLE_EQ(sample.unitsPerSecond(), 4e9);
    EXPECT_DOUBLE_EQ(sample.completedWork(), 8e6);
    EXPECT_DOUBLE_EQ(sample.elapsedSeconds(), .002);
    EXPECT_EQ(sample.provenance(), "rank=3 ROCm=1 sample=7");
    EXPECT_DOUBLE_EQ(sample.secondsFor(PlanningWorkUnit::Bytes, 4e6), .001);
    EXPECT_DOUBLE_EQ(sample.secondsFor(PlanningWorkUnit::Bytes, 0), 0);
    EXPECT_THROW(sample.secondsFor(PlanningWorkUnit::ArithmeticOperations, 1), std::invalid_argument);
    EXPECT_THROW(PlanningKernelService(bytes(1), bytes(1), 0), std::invalid_argument);
    EXPECT_THROW(PlanningKernelService(operations(1), operations(1), 0), std::invalid_argument);
    EXPECT_THROW(PlanningTransferLeg(operations(1), 0), std::invalid_argument);
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(sample.secondsFor(PlanningWorkUnit::Bytes, invalid), std::invalid_argument);
        EXPECT_THROW(PlanningKernelService(operations(1), bytes(1), invalid), std::invalid_argument);
        EXPECT_THROW(PlanningTransferLeg(bytes(1), invalid), std::invalid_argument);
        EXPECT_THROW(planningSerialSeconds(std::array{1.0, invalid}), std::invalid_argument);
        EXPECT_THROW(planningIndependentSeconds(std::array{1.0, invalid}), std::invalid_argument);
    }
}

TEST(OrchestrationPerformanceEvidence, PayloadCurveIsMonotoneAndChargesBulkBeyondTheObservedRange)
{
    const PlanningPayloadServiceCurve curve(10, 2, 30, 6);
    EXPECT_DOUBLE_EQ(curve.seconds(1), 2);
    EXPECT_DOUBLE_EQ(curve.seconds(10), 2);
    EXPECT_DOUBLE_EQ(curve.seconds(20), 4);
    EXPECT_DOUBLE_EQ(curve.seconds(30), 6);
    EXPECT_DOUBLE_EQ(curve.seconds(60), 12);
    // Noise can flatten the interval but must not create free bulk service.
    const PlanningPayloadServiceCurve noisy(10, 6, 30, 2);
    const PlanningPayloadServiceCurve repeated(10, 2, 10, 6);
    EXPECT_DOUBLE_EQ(noisy.seconds(20), 6);
    EXPECT_DOUBLE_EQ(noisy.seconds(60), 12);
    EXPECT_DOUBLE_EQ(repeated.seconds(10), 6);
    EXPECT_DOUBLE_EQ(repeated.seconds(20), 12);
    for (double small : {.00001, 1.0, 100.0})
        for (double large : {.00001, 1.0, 100.0})
        {
            const PlanningPayloadServiceCurve varied(1, small, 100, large);
            double previous = 0;
            for (size_t bytes = 1; bytes <= 1000; ++bytes)
            {
                const auto seconds = varied.seconds(bytes);
                ASSERT_GE(seconds, previous);
                ASSERT_GT(seconds, 0);
                previous = seconds;
            }
        }
}

TEST(OrchestrationPerformanceEvidence, PayloadCurveRejectsInvalidAndUnrepresentableCosts)
{
    const auto infinity = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    for (double invalid : {0.0, -1.0, infinity, nan})
    {
        EXPECT_THROW(PlanningPayloadServiceCurve(1, invalid, 20, 1), std::invalid_argument);
        EXPECT_THROW(PlanningPayloadServiceCurve(1, 1, 20, invalid), std::invalid_argument);
    }
    EXPECT_THROW(PlanningPayloadServiceCurve(0, 1, 20, 1), std::invalid_argument);
    EXPECT_THROW(PlanningPayloadServiceCurve(21, 1, 20, 1), std::invalid_argument);
    const PlanningPayloadServiceCurve curve(1, 1, 20, 1);
    EXPECT_THROW(curve.seconds(0), std::invalid_argument);
    const PlanningPayloadServiceCurve huge(1, 1, 2, std::numeric_limits<double>::max());
    EXPECT_THROW(huge.seconds(3), std::overflow_error);
    const PlanningPayloadServiceCurve tiny(1, std::numeric_limits<double>::denorm_min(),
        2, std::numeric_limits<double>::denorm_min());
    EXPECT_GT(tiny.seconds(1), 0);
    EXPECT_GT(tiny.seconds(SIZE_MAX), 0);
}

TEST(OrchestrationPerformanceEvidence, ArithmeticRowCurvePreservesInvocationTimeAndEndpointRates)
{
    // Both endpoint invocations last one second. Every intermediate M must
    // remain one second rather than acquiring an artificial quadratic slowdown.
    for (int rows : {1, 2, 4, 8, 16, 32, 64})
        EXPECT_DOUBLE_EQ(planningArithmeticSeconds(operations(100), 64, operations(6400), rows, 100.0 * rows), 1.0);
    EXPECT_DOUBLE_EQ(planningArithmeticSeconds(operations(100), 64, operations(6400), 128, 12800), 2.0);
    EXPECT_DOUBLE_EQ(planningArithmeticSeconds(operations(100), 64, operations(6400), 3.5, 350), 1.0);
    EXPECT_DOUBLE_EQ(planningArithmeticSeconds(operations(100), 64, operations(6400), 1, 0), 0);
    EXPECT_THROW(planningArithmeticSeconds(bytes(100), 64, operations(6400), 1, 100), std::invalid_argument);
    EXPECT_THROW(planningArithmeticSeconds(operations(100), 64, bytes(6400), 1, 100), std::invalid_argument);
    EXPECT_THROW(planningArithmeticSeconds(operations(100), 1, operations(100), 1, 100), std::invalid_argument);
    EXPECT_THROW(planningArithmeticSeconds(operations(100), 64, operations(6400), 0, 100), std::invalid_argument);
    EXPECT_THROW(planningArithmeticSeconds(operations(100), 64, operations(6400), 1, -100), std::invalid_argument);
}

TEST(OrchestrationPerformanceEvidence, ArithmeticNeverOverflowsOrErasesPositiveWork)
{
    const auto largest = std::numeric_limits<double>::max();
    const auto smallest = std::numeric_limits<double>::denorm_min();
    EXPECT_THROW(bytes(smallest).secondsFor(PlanningWorkUnit::Bytes, 1), std::overflow_error);
    EXPECT_THROW(bytes(largest).secondsFor(PlanningWorkUnit::Bytes, smallest), std::overflow_error);
    EXPECT_THROW(planningSerialSeconds(std::array{largest, largest}), std::overflow_error);
    EXPECT_THROW(PlanningKernelService(operations(1), bytes(1), largest).seconds(largest, 0), std::overflow_error);
    EXPECT_THROW(PlanningTransferLeg(bytes(1.0 / largest), largest).seconds(1), std::overflow_error);
    const std::array huge{PlanningTransferLeg(bytes(1), largest)};
    EXPECT_THROW(planningTransferSeconds(huge, 2, PlanningChunkPipeline(1)), std::overflow_error);
}

TEST(OrchestrationPerformanceEvidence, RooflineIsPerInvocationNotAnImaginaryCrossKernelOverlap)
{
    const PlanningKernelService kernel(operations(100), bytes(10), .25);
    EXPECT_DOUBLE_EQ(kernel.seconds(100, 100), 10.25); // Memory-bound decode.
    EXPECT_DOUBLE_EQ(kernel.seconds(10000, 100), 100.25); // Compute-bound grouped work.
    EXPECT_DOUBLE_EQ(kernel.seconds(0, 0), .25); // Issued empty work still has dispatch cost.
    EXPECT_EQ(kernel.arithmetic().provenance(), operations(100).provenance());
    EXPECT_EQ(kernel.memory().provenance(), bytes(10).provenance());
    const std::array stages{kernel.seconds(100, 100), kernel.seconds(10000, 1)};
    EXPECT_DOUBLE_EQ(planningSerialSeconds(stages), 110.5);
    EXPECT_DOUBLE_EQ(planningIndependentSeconds(stages), 100.25);
    EXPECT_DOUBLE_EQ(planningSerialSeconds({}), 0);
    EXPECT_DOUBLE_EQ(planningIndependentSeconds({}), 0);
}

TEST(OrchestrationPerformanceEvidence, SerialHostStagingChargesBothLegsAndBothMessageLatencies)
{
    const std::array legs{PlanningTransferLeg(bytes(8e9), .000002),
                          PlanningTransferLeg(bytes(8e9), .000003)};
    EXPECT_NEAR(planningTransferSeconds(legs, 8'000'000, PlanningSerialTransfer{}), .002005, 1e-15);
    EXPECT_NEAR(planningTransferSeconds(legs, 0, PlanningSerialTransfer{}), .000005, 1e-15);
    EXPECT_NEAR(planningTransferSeconds(legs, 0, PlanningChunkPipeline(4096)), .000005, 1e-15);
    const std::array reverse{PlanningTransferLeg(bytes(2e9), .000007)};
    EXPECT_NEAR(planningTransferSeconds(reverse, 8'000'000, PlanningSerialTransfer{}), .004007, 1e-15);
    EXPECT_THROW(planningTransferSeconds({}, 1, PlanningSerialTransfer{}), std::invalid_argument);
    EXPECT_THROW(PlanningChunkPipeline(0), std::invalid_argument);
}

TEST(OrchestrationPerformanceEvidence, ChunkPipelineAccountsForFillSteadyStateAndDrain)
{
    const std::array legs{PlanningTransferLeg(bytes(100), .1), PlanningTransferLeg(bytes(50), .2)};
    EXPECT_DOUBLE_EQ(planningTransferSeconds(legs, 100, PlanningSerialTransfer{}), 3.3);
    EXPECT_DOUBLE_EQ(planningTransferSeconds(legs, 100, PlanningChunkPipeline(100)), 3.3);
    // Full-chunk stages are .6 and 1.2 seconds; two chunks finish at 3.0s,
    // not payload/min(bandwidth), nor a two-leg serial sum for each chunk.
    EXPECT_DOUBLE_EQ(planningTransferSeconds(legs, 100, PlanningChunkPipeline(50)), 3.0);
    EXPECT_DOUBLE_EQ(planningTransferSeconds(legs, 75, PlanningChunkPipeline(50)), 2.5);
}

TEST(OrchestrationPerformanceEvidence, ClosedFormMatchesIndependentEventOracleForEveryTailAndBottleneck)
{
    for (int stage_count = 1; stage_count <= 8; ++stage_count)
    for (int slow_stage = 0; slow_stage < stage_count; ++slow_stage)
    for (std::size_t chunk : {1u, 3u, 16u, 31u})
    {
        std::vector<PlanningTransferLeg> legs;
        for (int stage = 0; stage < stage_count; ++stage)
            legs.emplace_back(bytes(stage == slow_stage ? 3.0 : 11.0 + stage), .01 * (stage + 1));
        for (std::size_t payload = 0; payload <= chunk * 4 + 1; ++payload)
        {
            SCOPED_TRACE(::testing::Message() << "stages=" << stage_count << " bottleneck=" << slow_stage
                << " chunk=" << chunk << " payload=" << payload);
            const double expected = simulate(legs, payload, chunk);
            EXPECT_NEAR(planningTransferSeconds(legs, payload, PlanningChunkPipeline(chunk)),
                expected, std::max(1.0, expected) * 1e-13);
        }
    }
}

TEST(OrchestrationPerformanceEvidence, HugePayloadDoesNotAllocateOrWalkOneEntryPerChunk)
{
    const std::array legs{PlanningTransferLeg(bytes(1e9), 0), PlanningTransferLeg(bytes(2e9), 0)};
    const auto payload = std::numeric_limits<std::size_t>::max();
    const auto result = planningTransferSeconds(legs, payload, PlanningChunkPipeline(1));
    EXPECT_NEAR(result, static_cast<double>(payload) / 1e9, 1e-5);
}
