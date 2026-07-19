/**
 * @file Test__NativeVNNITrainerEvidence.cpp
 * @brief CPU-only contract tests for cross-backend trainer evidence math.
 */

#include "utils/NativeVNNITrainerEvidence.h"
#include "utils/NativeVNNIPrefillProbePlan.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{
    using llaminar2::test::trainer::compareFP32;
    using llaminar2::test::trainer::expectedCPUPrefillPhysicalCandidateId;
    using llaminar2::test::trainer::evaluateAdaptiveTiming;
    using llaminar2::test::trainer::adaptiveTimingNeedsAnotherSample;
    using llaminar2::test::trainer::anchoredCompleteRoundTimingEvidence;
    using llaminar2::test::trainer::nativeByteDigest;
    using llaminar2::test::trainer::nativeByteMismatchCount;
    using llaminar2::test::trainer::rowSoftmaxSymmetricKLDivergence;
    using llaminar2::test::trainer::summarizeSortedTimingSamples;
    using llaminar2::test::trainer::timingCandidateParticipatesInActiveRound;

    TEST(Test__NativeVNNITrainerEvidence,
         PrefillProbePlanCollapsesFullKRequestsOntoSerialKPart)
    {
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.two_row_pair_grid.nbc16.full_k",
                /*serial_m1_uses_kpart=*/true,
                /*candidate_uses_kpart=*/false,
                /*candidate_requests_wide_rows=*/false,
                /*effective_runtime_is_avx512=*/true,
                /*M=*/15,
                /*N=*/4096,
                /*threads=*/28),
            llaminar2::test::trainer::kCPUPrefillKPartPairwiseCandidate);

        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
                /*serial_m1_uses_kpart=*/true,
                /*candidate_uses_kpart=*/true,
                /*candidate_requests_wide_rows=*/true,
                /*effective_runtime_is_avx512=*/true,
                /*M=*/15,
                /*N=*/4096,
                /*threads=*/28),
            llaminar2::test::trainer::kCPUPrefillKPartWideRowsCandidate);

        // AVX2 and two-row launches both normalize the logical wide-row
        // request to the pairwise physical implementation.
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
                true,
                true,
                true,
                false,
                15,
                4096,
                28),
            llaminar2::test::trainer::kCPUPrefillKPartPairwiseCandidate);
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
                true,
                true,
                true,
                true,
                2,
                4096,
                28),
            llaminar2::test::trainer::kCPUPrefillKPartPairwiseCandidate);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         PrefillProbePlanCollapsesKPartRequestsOntoSerialFullK)
    {
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
                /*serial_m1_uses_kpart=*/false,
                /*candidate_uses_kpart=*/true,
                /*candidate_requests_wide_rows=*/false,
                /*effective_runtime_is_avx512=*/true,
                /*M=*/64,
                /*N=*/256,
                /*threads=*/28),
            llaminar2::test::trainer::kCPUPrefillRowChunkCandidate);
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
                false,
                true,
                false,
                true,
                64,
                4096,
                28),
            llaminar2::test::trainer::kCPUPrefillTwoRowNbc1Candidate);
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
                false,
                true,
                false,
                true,
                64,
                64,
                1),
            llaminar2::test::trainer::kCPUPrefillTwoRowNbc1Candidate);

        constexpr std::string_view compatible =
            "cpu.nvnni.prefill.two_row_pair_grid.nbc8.full_k";
        EXPECT_EQ(
            expectedCPUPrefillPhysicalCandidateId(
                compatible,
                false,
                false,
                false,
                true,
                64,
                4096,
                28),
            compatible);
    }

    TEST(Test__NativeVNNITrainerEvidence, NativeBytesDistinguishSignedZero)
    {
        const std::vector<float> positive{0.0f, 1.0f};
        const std::vector<float> negative{-0.0f, 1.0f};

        EXPECT_NE(nativeByteDigest(positive), nativeByteDigest(negative));
        EXPECT_EQ(nativeByteMismatchCount(positive, negative), 1u);

        const auto evidence = compareFP32(positive, negative, positive.size());
        EXPECT_EQ(evidence.mismatch_count, 1u);
        EXPECT_EQ(evidence.first_mismatch_index, 0u);
        EXPECT_DOUBLE_EQ(evidence.max_abs, 0.0);
        EXPECT_FALSE(evidence.bitwiseEqual());
    }

    TEST(Test__NativeVNNITrainerEvidence, TimingSummaryUsesTrainerRankPolicy)
    {
        const std::vector<double> samples{1.0, 2.0, 3.0, 4.0, 100.0};
        const auto summary = summarizeSortedTimingSamples(samples);

        EXPECT_DOUBLE_EQ(summary.min, 1.0);
        EXPECT_DOUBLE_EQ(summary.median, 3.0);
        EXPECT_DOUBLE_EQ(summary.p95, 100.0);
        EXPECT_DOUBLE_EQ(summary.mad, 1.0);
        EXPECT_GT(summary.cv, 1.0);
        EXPECT_EQ(summary.digest, nativeByteDigest(samples));
    }

    TEST(Test__NativeVNNITrainerEvidence, AdaptiveTimingRequiresElapsedStableEvidence)
    {
        const auto too_short = evaluateAdaptiveTiming(
            {400000.0, 401000.0, 399000.0, 400500.0, 399500.0},
            5,
            30,
            2500000.0,
            0.02);
        EXPECT_TRUE(too_short.sample_floor_reached);
        EXPECT_FALSE(too_short.elapsed_floor_reached);
        EXPECT_FALSE(too_short.median_stable);
        EXPECT_FALSE(too_short.promotion_evidence);
        EXPECT_FALSE(too_short.should_stop);
        EXPECT_EQ(too_short.stationary_sample_count, 0u);

        const auto enough = evaluateAdaptiveTiming(
            {500000.0, 501000.0, 499000.0, 500500.0, 499500.0},
            5,
            30,
            2000000.0,
            0.02);
        EXPECT_TRUE(enough.elapsed_floor_reached);
        EXPECT_TRUE(enough.median_stable);
        EXPECT_TRUE(enough.promotion_evidence);
        EXPECT_TRUE(enough.should_stop);
        EXPECT_LT(enough.median_relative_drift, 0.02);
        EXPECT_EQ(enough.stationary_window_begin, 0u);
        EXPECT_EQ(enough.stationary_sample_count, 5u);
        EXPECT_DOUBLE_EQ(enough.stationary_duration_us, 2500000.0);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         TwoTimingWindowsCannotEndALateFrequencyTransition)
    {
        const auto drifting = evaluateAdaptiveTiming(
            {100.0, 101.0, 102.0, 130.0, 131.0, 132.0},
            5,
            6,
            250.0,
            0.02);
        EXPECT_TRUE(drifting.sample_floor_reached);
        EXPECT_TRUE(drifting.elapsed_floor_reached);
        EXPECT_FALSE(drifting.median_stable);
        EXPECT_FALSE(drifting.promotion_evidence);
        EXPECT_FALSE(drifting.should_stop);
        EXPECT_GT(drifting.median_relative_drift, 0.20);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         AdaptiveTimingExposesDriftAtTheFourWindowCeiling)
    {
        const auto drifting = evaluateAdaptiveTiming(
            {100.0, 101.0, 102.0, 130.0, 131.0, 132.0},
            5,
            6,
            150.0,
            0.02);
        EXPECT_TRUE(drifting.sample_floor_reached);
        EXPECT_TRUE(drifting.elapsed_floor_reached);
        EXPECT_FALSE(drifting.median_stable);
        EXPECT_FALSE(drifting.promotion_evidence);
        EXPECT_TRUE(drifting.should_stop);
        EXPECT_GT(drifting.median_relative_drift, 0.20);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         SampleCeilingCannotStopBeforeElapsedEvidence)
    {
        const auto sample_only = evaluateAdaptiveTiming(
            {10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
            5,
            6,
            100.0,
            0.02);

        EXPECT_TRUE(sample_only.sample_floor_reached);
        EXPECT_FALSE(sample_only.elapsed_floor_reached);
        EXPECT_FALSE(sample_only.promotion_evidence);
        EXPECT_FALSE(sample_only.should_stop);
    }

    TEST(Test__NativeVNNITrainerEvidence, StableTailRecoversFromStartupTransient)
    {
        std::vector<double> samples(40u, 1300.0);
        samples.insert(samples.end(), 80u, 1000.0);

        const auto recovered = evaluateAdaptiveTiming(
            samples,
            5,
            180,
            50000.0,
            0.02);

        EXPECT_TRUE(recovered.promotion_evidence);
        EXPECT_TRUE(recovered.median_stable);
        EXPECT_EQ(recovered.stationary_window_begin, 70u);
        EXPECT_EQ(recovered.stationary_sample_count, 50u);
        EXPECT_DOUBLE_EQ(recovered.stationary_duration_us, 50000.0);
        EXPECT_DOUBLE_EQ(recovered.median_relative_drift, 0.0);
    }

    TEST(Test__NativeVNNITrainerEvidence, StableShortTailCannotHideEarlierDrift)
    {
        std::vector<double> samples(40u, 1300.0);
        samples.insert(samples.end(), 20u, 1000.0);

        const auto still_drifting = evaluateAdaptiveTiming(
            samples,
            5,
            180,
            50000.0,
            0.02);

        EXPECT_FALSE(still_drifting.promotion_evidence);
        EXPECT_FALSE(still_drifting.median_stable);
        EXPECT_EQ(still_drifting.stationary_window_begin, 16u);
        EXPECT_EQ(still_drifting.stationary_sample_count, 44u);
        EXPECT_GE(still_drifting.stationary_duration_us, 50000.0);
        EXPECT_GT(still_drifting.median_relative_drift, 0.20);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         PromotedCandidateFreezesWhileAnotherCandidateKeepsTiming)
    {
        std::vector<double> stable_prefix(100u, 1000.0);
        const auto promoted = evaluateAdaptiveTiming(
            stable_prefix,
            5,
            180,
            50000.0,
            0.02);
        ASSERT_TRUE(promoted.promotion_evidence);
        EXPECT_FALSE(adaptiveTimingNeedsAnotherSample(
            promoted,
            stable_prefix.size(),
            180u,
            50000.0));

        // This later tail models unrelated collection rounds that used to be
        // appended while a peer candidate was still converging. Re-evaluating
        // the stream would revoke valid evidence, which is precisely why the
        // production loop must stop launching the promoted candidate.
        std::vector<double> accidentally_extended = stable_prefix;
        accidentally_extended.insert(
            accidentally_extended.end(), 10u, 2000.0);
        const auto revoked = evaluateAdaptiveTiming(
            accidentally_extended,
            5,
            180,
            50000.0,
            0.02);
        EXPECT_FALSE(revoked.promotion_evidence);
        EXPECT_GT(revoked.median_relative_drift, 0.20);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         UnstableCandidateStopsAtBoundedRecoveryCeiling)
    {
        // The complete trace reaches four 50 ms evidence windows. Its latest
        // 50 ms suffix still straddles a 1 ms -> 2 ms transition, so recovery
        // must terminate collection without claiming stationarity.
        std::vector<double> drifting(180u, 1000.0);
        drifting.insert(drifting.end(), 10u, 2000.0);
        const auto decision = evaluateAdaptiveTiming(
            drifting,
            5,
            180,
            50000.0,
            0.02);

        ASSERT_FALSE(decision.promotion_evidence);
        ASSERT_GE(decision.measured_duration_us, 4.0 * 50000.0);
        EXPECT_FALSE(adaptiveTimingNeedsAnotherSample(
            decision,
            drifting.size(),
            180u,
            50000.0));
    }

    TEST(Test__NativeVNNITrainerEvidence,
         RequestedStationaryPopulationRejectsAChanceStableTail)
    {
        std::vector<double> samples(15u, 1300.0);
        samples.insert(samples.end(), 15u, 1000.0);

        const auto chance_tail = evaluateAdaptiveTiming(
            samples,
            2,
            180,
            2000.0,
            0.02);
        ASSERT_TRUE(chance_tail.promotion_evidence);
        ASSERT_EQ(chance_tail.stationary_sample_count, 2u);

        const auto representative_tail = evaluateAdaptiveTiming(
            samples,
            30,
            180,
            2000.0,
            0.02);
        EXPECT_FALSE(representative_tail.promotion_evidence);
        EXPECT_EQ(representative_tail.stationary_sample_count, 30u);
        EXPECT_GT(representative_tail.median_relative_drift, 0.20);
    }

    TEST(Test__NativeVNNITrainerEvidence,
         AnchoredEpochCannotStopAtSampleCeilingBeforeElapsedFloor)
    {
        EXPECT_FALSE(anchoredCompleteRoundTimingEvidence(
            180u, 15u, 97000.0, 100000.0));
        EXPECT_TRUE(anchoredCompleteRoundTimingEvidence(
            186u, 15u, 100100.0, 100000.0));
    }

    TEST(Test__NativeVNNITrainerEvidence,
         AnchoredCandidateRemainsInEveryGloballyActiveRound)
    {
        const auto locally_complete = evaluateAdaptiveTiming(
            {25000.0, 25000.0, 25000.0, 25000.0, 25000.0},
            5u,
            180u,
            100000.0,
            0.02);
        ASSERT_TRUE(locally_complete.promotion_evidence);

        // Ordinary evidence freezes here. An anchored candidate must still
        // launch while a slower candidate or MPI peer keeps the epoch open.
        EXPECT_FALSE(timingCandidateParticipatesInActiveRound(
            false,
            locally_complete,
            5u,
            180u,
            100000.0));
        EXPECT_TRUE(timingCandidateParticipatesInActiveRound(
            true,
            locally_complete,
            5u,
            180u,
            100000.0));
    }

    TEST(Test__NativeVNNITrainerEvidence, SymmetricKLIsActuallySymmetric)
    {
        const std::vector<float> lhs{3.0f, 1.0f, -2.0f};
        const std::vector<float> rhs{2.5f, 1.5f, -1.0f};

        const double forward = rowSoftmaxSymmetricKLDivergence(
            lhs.data(), rhs.data(), lhs.size());
        const double reverse = rowSoftmaxSymmetricKLDivergence(
            rhs.data(), lhs.data(), lhs.size());
        EXPECT_GT(forward, 0.0);
        EXPECT_NEAR(forward, reverse, 1.0e-15);
        EXPECT_DOUBLE_EQ(
            rowSoftmaxSymmetricKLDivergence(
                lhs.data(), lhs.data(), lhs.size()),
            0.0);
    }

    TEST(Test__NativeVNNITrainerEvidence, ExactRowsPassEveryEvidenceGate)
    {
        const std::vector<float> values{0.25f, -1.5f, 7.0f, 0.0f};
        const auto evidence = compareFP32(values, values, 2);

        EXPECT_TRUE(evidence.bitwiseEqual());
        EXPECT_EQ(evidence.mismatch_count, 0u);
        EXPECT_DOUBLE_EQ(evidence.cosine, 1.0);
        EXPECT_DOUBLE_EQ(evidence.relative_l2, 0.0);
        EXPECT_DOUBLE_EQ(evidence.max_abs, 0.0);
        EXPECT_DOUBLE_EQ(evidence.symmetric_kld, 0.0);
        EXPECT_EQ(evidence.actual_digest, evidence.expected_digest);
    }
} // namespace
