/**
 * @file Test__MoEOverlayInferenceInterferenceProbe.cpp
 * @brief Device-free adversarial tests for live inference timing publication.
 */

#include "execution/moe/MoEOverlayInferenceInterferenceProbe.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Construct one valid request without stringly topology data. */
        MoEOverlayInterferenceProbeRequest request(
            ExpertHistogramSource source,
            MoEOverlayInterferenceProbeMode mode,
            std::uint64_t sequence = 1)
        {
            return {
                .coordinate = {
                    .source_participant = 2,
                    .destination_participant = 0,
                    .layer = 7,
                },
                .source = source,
                .mode = mode,
                .calibration_sequence = sequence,
            };
        }

        /** @brief Construct one exact allocation-free production workload key. */
        MoEOverlayInferenceWorkloadIdentity workload(
            ExpertHistogramSource source,
            int rows = 1,
            int speculative_depth = 0,
            std::uint64_t fingerprint = 101)
        {
            return {
                .source = source,
                .real_rows = rows,
                .execution_rows = rows,
                .transaction_count = 1,
                .speculative_depth = speculative_depth,
                .schedule_fingerprint = fingerprint,
            };
        }
    } // namespace

    TEST(
        MoEOverlayInferenceInterferenceProbe,
        PublishesOneExactPhaseIntervalWithoutBlocking)
    {
        MoEOverlayInferenceInterferenceProbe probe;
        EXPECT_TRUE(probe.idle());

        auto invalid = request(
            ExpertHistogramSource::SyntheticTest,
            MoEOverlayInterferenceProbeMode::Baseline);
        EXPECT_FALSE(probe.arm(invalid));

        const auto armed = request(
            ExpertHistogramSource::PrefillChunk,
            MoEOverlayInterferenceProbeMode::Baseline,
            41);
        ASSERT_TRUE(probe.arm(armed));
        EXPECT_EQ(
            probe.progress(armed),
            MoEOverlayInterferenceProbeProgress::Armed);
        EXPECT_EQ(
            probe.progress(request(
                ExpertHistogramSource::PrefillChunk,
                MoEOverlayInterferenceProbeMode::Baseline,
                42)),
            MoEOverlayInterferenceProbeProgress::Missing);
        EXPECT_FALSE(probe.idle());
        EXPECT_FALSE(
            probe.beginSample(workload(
                ExpertHistogramSource::DecodeToken)).valid());

        const auto ticket =
            probe.beginSample(workload(
                ExpertHistogramSource::PrefillChunk, 8));
        ASSERT_TRUE(ticket.valid());
        EXPECT_EQ(
            probe.progress(armed),
            MoEOverlayInterferenceProbeProgress::Running);
        EXPECT_EQ(ticket.calibration_sequence, 41u);
        EXPECT_FALSE(
            probe.beginSample(workload(
                ExpertHistogramSource::PrefillChunk, 8)).valid());
        EXPECT_FALSE(probe.arm(armed));
        EXPECT_FALSE(probe.cancelArmed());
        ASSERT_TRUE(probe.finishSample(ticket));
        EXPECT_EQ(
            probe.progress(armed),
            MoEOverlayInterferenceProbeProgress::Completed);
        EXPECT_FALSE(probe.finishSample(ticket));

        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe.consume(&sample));
        EXPECT_TRUE(sample.valid());
        EXPECT_EQ(sample.request.coordinate, armed.coordinate);
        EXPECT_EQ(sample.request.source, ExpertHistogramSource::PrefillChunk);
        EXPECT_EQ(
            sample.workload,
            workload(ExpertHistogramSource::PrefillChunk, 8));
        EXPECT_EQ(sample.request.mode, MoEOverlayInterferenceProbeMode::Baseline);
        EXPECT_EQ(sample.request.calibration_sequence, 41u);
        EXPECT_GT(sample.durationNanoseconds(), 0u);
        EXPECT_TRUE(sample.whollyContainedBy(
            sample.begin_steady_nanoseconds,
            sample.end_steady_nanoseconds));
        EXPECT_FALSE(sample.whollyContainedBy(
            sample.begin_steady_nanoseconds + 1u,
            sample.end_steady_nanoseconds));
        EXPECT_TRUE(sample.whollyContains(
            sample.begin_steady_nanoseconds,
            sample.end_steady_nanoseconds));
        EXPECT_FALSE(sample.whollyContains(
            sample.begin_steady_nanoseconds - 1u,
            sample.end_steady_nanoseconds));
        EXPECT_TRUE(probe.idle());
        EXPECT_EQ(
            probe.progress(armed),
            MoEOverlayInterferenceProbeProgress::Missing);
        EXPECT_FALSE(probe.consume(&sample));

        const auto stats = probe.stats();
        EXPECT_EQ(stats.arms, 1u);
        EXPECT_EQ(stats.claims, 1u);
        EXPECT_EQ(stats.completions, 1u);
        EXPECT_EQ(stats.consumptions, 1u);
        EXPECT_EQ(stats.phase_misses, 1u);
        EXPECT_GE(stats.rejected_operations, 4u);
    }

    TEST(
        MoEOverlayInferenceInterferenceProbe,
        ConcurrentArmAcceptsOnlyTheExactBaselineWorkloadIdentity)
    {
        MoEOverlayInferenceInterferenceProbe probe;
        const auto exact = workload(
            ExpertHistogramSource::PrefillChunk, 64, 0, 707);
        auto paired = request(
            ExpertHistogramSource::PrefillChunk,
            MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            9);
        paired.require_exact_workload = true;
        paired.required_workload = exact;
        ASSERT_TRUE(probe.arm(paired));

        auto wrong = exact;
        wrong.execution_rows = 128;
        wrong.schedule_fingerprint = 808;
        EXPECT_FALSE(probe.beginSample(wrong).valid());
        EXPECT_FALSE(
            probe.beginSample(workload(
                ExpertHistogramSource::DecodeToken)).valid());

        const auto ticket = probe.beginSample(exact);
        ASSERT_TRUE(ticket.valid());
        ASSERT_TRUE(probe.finishSample(ticket));
        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe.consume(&sample));
        EXPECT_EQ(sample.workload, exact);
        EXPECT_EQ(probe.stats().workload_misses, 1u);
        EXPECT_EQ(probe.stats().phase_misses, 1u);
    }

    TEST(
        MoEOverlayInferenceInterferenceProbe,
        ExactlyOneConcurrentInferenceCallerClaimsTheArmedGeneration)
    {
        MoEOverlayInferenceInterferenceProbe probe;
        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::GroupedVerifier,
            MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            73)));

        constexpr std::size_t kCallers = 16;
        std::array<std::thread, kCallers> callers;
        std::atomic<bool> release{false};
        std::atomic<std::uint64_t> winners{0};
        for (auto &caller : callers)
        {
            caller = std::thread(
                [&]
                {
                    while (!release.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    const auto ticket = probe.beginSample(
                        workload(
                            ExpertHistogramSource::GroupedVerifier,
                            4,
                            3));
                    if (!ticket.valid())
                        return;
                    winners.fetch_add(1, std::memory_order_relaxed);
                    EXPECT_TRUE(probe.finishSample(ticket));
                });
        }
        release.store(true, std::memory_order_release);
        for (auto &caller : callers)
            caller.join();

        EXPECT_EQ(winners.load(std::memory_order_relaxed), 1u);
        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe.consume(&sample));
        EXPECT_EQ(
            sample.request.mode,
            MoEOverlayInterferenceProbeMode::ConcurrentMovement);
        EXPECT_EQ(sample.request.calibration_sequence, 73u);
        EXPECT_EQ(probe.stats().claims, 1u);
    }

    TEST(
        MoEOverlayInferenceInterferenceProbe,
        CancellationOnlyReleasesAnUnclaimedRequestAndNextArmGetsNewGeneration)
    {
        MoEOverlayInferenceInterferenceProbe probe;
        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::DecodeToken,
            MoEOverlayInterferenceProbeMode::Baseline,
            1)));
        ASSERT_TRUE(probe.cancelArmed());
        EXPECT_TRUE(probe.idle());

        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::DecodeToken,
            MoEOverlayInterferenceProbeMode::Baseline,
            2)));
        const auto second =
            probe.beginSample(workload(
                ExpertHistogramSource::DecodeToken));
        ASSERT_TRUE(second.valid());
        EXPECT_GT(second.probe_generation, 1u);
        EXPECT_EQ(second.calibration_sequence, 2u);
        EXPECT_FALSE(probe.cancelArmed());
        ASSERT_TRUE(probe.finishSample(second));

        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe.consume(&sample));
        EXPECT_EQ(sample.request.calibration_sequence, 2u);
        EXPECT_EQ(probe.stats().cancellations, 1u);

        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::GroupedVerifier,
            MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            3)));
        {
            MoEOverlayInferenceInterferenceScope scope(
                &probe,
                workload(
                    ExpertHistogramSource::GroupedVerifier,
                    4,
                    3));
            ASSERT_TRUE(scope.active());
            scope.discard();
        }
        EXPECT_TRUE(probe.idle());
        EXPECT_EQ(probe.stats().running_discards, 1u);

        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::DecodeToken,
            MoEOverlayInterferenceProbeMode::Baseline,
            4)));
        {
            MoEOverlayInferenceInterferenceScope scope(
                &probe,
                workload(ExpertHistogramSource::DecodeToken));
            ASSERT_TRUE(scope.active());
        }
        ASSERT_TRUE(probe.consume(&sample));
        EXPECT_EQ(sample.request.calibration_sequence, 4u);

        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::PrefillChunk,
            MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            5)));
        const auto running = probe.beginSample(workload(
            ExpertHistogramSource::PrefillChunk,
            8));
        ASSERT_TRUE(running.valid());
        EXPECT_FALSE(probe.discardAvailable());
        EXPECT_FALSE(probe.idle());
        ASSERT_TRUE(probe.finishSample(running));
        EXPECT_TRUE(probe.discardAvailable());
        EXPECT_TRUE(probe.idle());
        EXPECT_EQ(probe.stats().completed_discards, 1u);

        ASSERT_TRUE(probe.arm(request(
            ExpertHistogramSource::DecodeToken,
            MoEOverlayInterferenceProbeMode::Baseline,
            6)));
        EXPECT_TRUE(probe.discardAvailable());
        EXPECT_TRUE(probe.idle());
        EXPECT_EQ(probe.stats().cancellations, 2u);
    }
} // namespace llaminar2::test
