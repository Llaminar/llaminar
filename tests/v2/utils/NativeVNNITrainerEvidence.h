/**
 * @file NativeVNNITrainerEvidence.h
 * @brief Backend-neutral evidence primitives for NativeVNNI policy trainers.
 *
 * Learned dispatch is only trustworthy when every backend emits fields with
 * identical semantics. This header centralizes native-byte fingerprints,
 * replay comparison, robust timing aggregation, and FP32 diagnostics for the
 * CUDA, ROCm, and CPU trainer harnesses. Correctness promotion still comes from
 * exact byte comparison; cosine, relative L2, maximum absolute error, and
 * symmetric KL divergence are diagnostic coordinates for locating drift.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2::test::trainer
{
    /**
     * @brief Maximum acquisition duration measured in stationary windows.
     *
     * AVX-512 frequency-state transitions can consume more than two complete
     * timing windows after source preconditioning. Four windows allow the
     * first observation window plus three replacement windows while retaining
     * a deterministic bound for a genuinely nonstationary kernel.
     */
    inline constexpr double ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER = 4.0;

    /**
     * @brief Robust aggregate and provenance for sorted timing samples.
     */
    struct TimingEvidence
    {
        double min = 0.0;    ///< Minimum observed latency in caller units.
        double median = 0.0; ///< Upper median, matching trainer selection.
        double p95 = 0.0;    ///< Nearest-rank 95th percentile.
        double mad = 0.0;    ///< Median absolute deviation from @ref median.
        double cv = 0.0;     ///< Population coefficient of variation.
        std::string digest;  ///< Native-double byte fingerprint.
    };

    /**
     * @brief Decision produced by the elapsed-evidence timing protocol.
     *
     * NativeVNNI training spans sub-millisecond decode kernels and CPU GEMMs
     * whose individual launches can take hundreds of milliseconds.  A fixed
     * repetition count therefore either undersamples the former or spends
     * minutes repeatedly measuring the latter.  This value reports whether a
     * candidate has accumulated both an elapsed-time floor and a stable timing
     * window, while retaining a hard sample ceiling for bounded collection.
     */
    struct AdaptiveTimingDecision
    {
        bool sample_floor_reached = false;
        bool elapsed_floor_reached = false;
        bool median_stable = false;
        bool promotion_evidence = false;
        bool should_stop = false;
        double measured_duration_us = 0.0;
        size_t stationary_window_begin = 0;
        size_t stationary_sample_count = 0;
        double stationary_duration_us = 0.0;
        double median_relative_drift = std::numeric_limits<double>::infinity();
    };

    /**
     * @brief Test whether one anchored complete-round epoch has enough evidence.
     *
     * Candidate expansion times every new schedule and its existing route
     * anchor in the same shuffled rounds. Its bounded acquisition is complete
     * only when both the launch population and elapsed-kernel floor hold for
     * every participant. The requested maximum sample count is a recovery
     * trigger, not permission to publish a 97 ms epoch against a 100 ms floor;
     * inexpensive candidates may therefore execute a few additional common
     * rounds until elapsed evidence catches up.
     *
     * @param sample_count Number of common rounds retained for the candidate.
     * @param minimum_samples Required complete-round population.
     * @param measured_duration_us Sum of retained candidate launch durations.
     * @param minimum_duration_us Required elapsed kernel duration.
     * @return True only when both independent evidence floors are satisfied.
     */
    inline bool anchoredCompleteRoundTimingEvidence(
        size_t sample_count,
        size_t minimum_samples,
        double measured_duration_us,
        double minimum_duration_us) noexcept
    {
        return sample_count >= minimum_samples &&
            measured_duration_us >= minimum_duration_us;
    }

    /**
     * @brief Evaluate whether acquisition-order timings contain enough evidence.
     *
     * The stability statistic compares upper medians from the first and second
     * halves of the shortest trailing acquisition window that independently
     * satisfies both the sample and elapsed-time floors.  A trailing window is
     * essential here: CPU frequency can take several seconds to settle after a
     * new matrix geometry starts. Comparing the complete historical first half
     * against the complete second half makes a finite startup transient poison
     * the decision forever, even after hundreds of stationary launches.
     *
     * The selected suffix remains in acquisition order so that sustained clock
     * or thermal drift is visible. Callers use @ref stationary_window_begin to
     * derive policy-facing aggregates from exactly the same stationary suffix,
     * while retaining the complete acquisition sequence as diagnostic evidence.
     *
     * @param acquisition_samples_us Positive launch latencies in collection order.
     * @param minimum_samples Smallest statistically useful sample population.
     * @param maximum_samples Hard collection ceiling for inexpensive kernels.
     * @param minimum_duration_us Required cumulative measured kernel duration.
     * @param maximum_median_relative_drift Maximum admitted half-window drift.
     */
    inline AdaptiveTimingDecision evaluateAdaptiveTiming(
        const std::vector<double> &acquisition_samples_us,
        size_t minimum_samples,
        size_t maximum_samples,
        double minimum_duration_us,
        double maximum_median_relative_drift)
    {
        AdaptiveTimingDecision result;
        result.measured_duration_us = std::accumulate(
            acquisition_samples_us.begin(), acquisition_samples_us.end(), 0.0);
        result.sample_floor_reached =
            acquisition_samples_us.size() >= minimum_samples;
        result.elapsed_floor_reached =
            result.measured_duration_us >= minimum_duration_us;

        if (result.sample_floor_reached && result.elapsed_floor_reached)
        {
            size_t begin = acquisition_samples_us.size();
            double duration_us = 0.0;
            while (begin > 0u &&
                   (acquisition_samples_us.size() - begin < minimum_samples ||
                    duration_us < minimum_duration_us))
            {
                --begin;
                duration_us += acquisition_samples_us[begin];
            }

            result.stationary_window_begin = begin;
            result.stationary_sample_count =
                acquisition_samples_us.size() - begin;
            result.stationary_duration_us = duration_us;
        }

        if (result.stationary_sample_count >= 2u)
        {
            const size_t midpoint = result.stationary_window_begin +
                result.stationary_sample_count / 2u;
            std::vector<double> first(
                acquisition_samples_us.begin() + static_cast<std::ptrdiff_t>(
                    result.stationary_window_begin),
                acquisition_samples_us.begin() + static_cast<std::ptrdiff_t>(midpoint));
            std::vector<double> second(
                acquisition_samples_us.begin() + static_cast<std::ptrdiff_t>(midpoint),
                acquisition_samples_us.end());
            std::sort(first.begin(), first.end());
            std::sort(second.begin(), second.end());
            const double first_median = first[first.size() / 2u];
            const double second_median = second[second.size() / 2u];
            const double denominator = std::max(first_median, second_median);
            if (denominator > 0.0 && std::isfinite(denominator))
            {
                result.median_relative_drift =
                    std::abs(first_median - second_median) / denominator;
                result.median_stable =
                    result.median_relative_drift <=
                    maximum_median_relative_drift;
            }
        }

        result.promotion_evidence =
            result.sample_floor_reached && result.elapsed_floor_reached &&
            result.median_stable;
        result.should_stop = result.promotion_evidence ||
            (acquisition_samples_us.size() >= maximum_samples &&
             result.measured_duration_us >=
                 ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER *
                     minimum_duration_us);
        return result;
    }

    /**
     * @brief Decide whether an independently timed candidate needs another launch.
     *
     * Ordinary trainer evidence is terminal once a candidate has produced a
     * promotable stationary window.  Continuing to sample that candidate while
     * a slower peer is still converging can replace its valid suffix with a
     * later chance fluctuation.  With many candidates in one cell, requiring
     * every latest suffix to be stable simultaneously becomes a multiple-testing
     * gate rather than an independent stationarity test.
     *
     * A candidate that has not converged remains active until it reaches both
     * the hard sample floor and the bounded recovery-duration ceiling.  The
     * caller may then retain its complete diagnostic trace, but must not promote
     * it as installable timing evidence.
     *
     * Anchored candidate-expansion collection deliberately does not use this
     * helper: every candidate in an anchored epoch must observe the same number
     * of shuffled complete rounds so its contemporaneous normalization proof is
     * meaningful.
     *
     * @param decision Latest adaptive timing decision for this candidate.
     * @param sample_count Number of acquisition-order samples retained so far.
     * @param maximum_samples Hard sample floor for recovery-ceiling admission.
     * @param minimum_duration_us Requested stationary-window duration.
     * @return True only while another candidate launch can add required evidence.
     */
    inline bool adaptiveTimingNeedsAnotherSample(
        const AdaptiveTimingDecision &decision,
        size_t sample_count,
        size_t maximum_samples,
        double minimum_duration_us) noexcept
    {
        if (decision.promotion_evidence)
            return false;

        const bool recovery_ceiling_reached =
            sample_count >= maximum_samples &&
            decision.measured_duration_us >=
                ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER *
                    minimum_duration_us;
        return !recovery_ceiling_reached;
    }

    /**
     * @brief Decide whether one candidate participates in an active timing round.
     *
     * Ordinary evidence streams retire independently as soon as they either
     * become promotable or reach the bounded diagnostic ceiling. Anchored
     * candidate expansion has a different contract: every forceable candidate
     * must execute once in every globally active round, including rounds kept
     * open by a slower local candidate or another MPI rank. This common cadence
     * is what makes a contemporaneous anchor ratio meaningful.
     *
     * The caller invokes this helper only while the MPI coordinator reports
     * that the global timing epoch remains active. Consequently anchored mode
     * always returns true; local completion controls whether this rank asks to
     * close the next round, but never removes a candidate from a round that a
     * peer still requires.
     *
     * @param anchored_candidate_expansion True for anchor-normalized evidence.
     * @param decision Latest adaptive timing decision for this candidate.
     * @param sample_count Number of acquisition-order samples retained so far.
     * @param maximum_samples Hard sample floor for recovery-ceiling admission.
     * @param minimum_duration_us Requested stationary-window duration.
     * @return True when the candidate must launch in the active global round.
     */
    inline bool timingCandidateParticipatesInActiveRound(
        bool anchored_candidate_expansion,
        const AdaptiveTimingDecision &decision,
        size_t sample_count,
        size_t maximum_samples,
        double minimum_duration_us) noexcept
    {
        return anchored_candidate_expansion ||
            adaptiveTimingNeedsAnotherSample(
                decision,
                sample_count,
                maximum_samples,
                minimum_duration_us);
    }

    /**
     * @brief Full native-byte and numerical comparison of two FP32 tensors.
     */
    struct FP32Evidence
    {
        size_t mismatch_count = 0;       ///< Number of differing FP32 words.
        size_t first_mismatch_index = 0; ///< First differing word when nonzero.
        double max_abs = 0.0;
        double relative_l2 = 0.0;
        double cosine = 0.0;
        double symmetric_kld = 0.0; ///< Maximum row-wise symmetric softmax KL.
        size_t nonfinite_count = 0;
        std::string actual_digest;
        std::string expected_digest;

        /** Return true only for finite, native-byte-identical outputs. */
        bool bitwiseEqual() const
        {
            return mismatch_count == 0 && nonfinite_count == 0;
        }
    };

    /**
     * @brief Fingerprint the exact native bytes of a typed value span.
     *
     * FNV-1a is an audit fingerprint, not the correctness oracle. The complete
     * byte mismatch count remains authoritative. The offset basis intentionally
     * matches the existing ROCm MoE trainer so old and new sidecars compare.
     */
    template <typename T>
    std::string nativeByteDigest(const T *values, size_t count)
    {
        constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
        constexpr uint64_t kPrime = 1099511628211ULL;
        uint64_t hash = kOffsetBasis;
        const auto *bytes = reinterpret_cast<const uint8_t *>(values);
        const size_t byte_count = count * sizeof(T);
        for (size_t index = 0; index < byte_count; ++index)
        {
            hash ^= bytes[index];
            hash *= kPrime;
        }
        std::ostringstream stream;
        stream << "fnv1a64:" << std::hex << std::setw(16)
               << std::setfill('0') << hash;
        return stream.str();
    }

    /** Convenience overload for an owning vector. */
    template <typename T>
    std::string nativeByteDigest(const std::vector<T> &values)
    {
        return nativeByteDigest<T>(values.data(), values.size());
    }

    /**
     * @brief Count differing bytes between two equally typed native spans.
     */
    template <typename T>
    size_t nativeByteMismatchCount(
        const T *lhs,
        size_t lhs_count,
        const T *rhs,
        size_t rhs_count)
    {
        if (lhs_count != rhs_count)
            return std::max(lhs_count, rhs_count) * sizeof(T);
        const auto *lhs_bytes = reinterpret_cast<const uint8_t *>(lhs);
        const auto *rhs_bytes = reinterpret_cast<const uint8_t *>(rhs);
        const size_t byte_count = lhs_count * sizeof(T);
        size_t mismatches = 0;
        for (size_t index = 0; index < byte_count; ++index)
            mismatches += lhs_bytes[index] != rhs_bytes[index] ? 1u : 0u;
        return mismatches;
    }

    /** Convenience overload for owning vectors. */
    template <typename T>
    size_t nativeByteMismatchCount(
        const std::vector<T> &lhs,
        const std::vector<T> &rhs)
    {
        return nativeByteMismatchCount<T>(
            lhs.data(), lhs.size(), rhs.data(), rhs.size());
    }

    /**
     * @brief Aggregate an already sorted, non-empty latency sample vector.
     */
    inline TimingEvidence summarizeSortedTimingSamples(
        const std::vector<double> &sorted)
    {
        TimingEvidence result;
        if (sorted.empty())
            return result;

        result.min = sorted.front();
        result.median = sorted[sorted.size() / 2u];
        const size_t p95_rank = static_cast<size_t>(
            std::ceil(0.95 * static_cast<double>(sorted.size())));
        result.p95 = sorted[
            std::min(sorted.size() - 1u, std::max<size_t>(1u, p95_rank) - 1u)];

        std::vector<double> deviations;
        deviations.reserve(sorted.size());
        for (double value : sorted)
            deviations.push_back(std::abs(value - result.median));
        std::sort(deviations.begin(), deviations.end());
        result.mad = deviations[deviations.size() / 2u];

        const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                            static_cast<double>(sorted.size());
        if (mean > 0.0)
        {
            double squared_error = 0.0;
            for (double value : sorted)
            {
                const double delta = value - mean;
                squared_error += delta * delta;
            }
            result.cv = std::sqrt(
                squared_error / static_cast<double>(sorted.size())) / mean;
        }
        result.digest = nativeByteDigest(sorted);
        return result;
    }

    /**
     * @brief Compute symmetric KL between two row-wise softmax distributions.
     */
    inline double rowSoftmaxSymmetricKLDivergence(
        const float *actual,
        const float *expected,
        size_t width)
    {
        if (!actual || !expected || width == 0)
            return std::numeric_limits<double>::infinity();
        double max_actual = -std::numeric_limits<double>::infinity();
        double max_expected = -std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < width; ++index)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[index]));
            max_expected = std::max(max_expected, static_cast<double>(expected[index]));
        }

        std::vector<double> actual_probability(width);
        std::vector<double> expected_probability(width);
        double actual_sum = 0.0;
        double expected_sum = 0.0;
        for (size_t index = 0; index < width; ++index)
        {
            actual_probability[index] =
                std::exp(static_cast<double>(actual[index]) - max_actual);
            expected_probability[index] =
                std::exp(static_cast<double>(expected[index]) - max_expected);
            actual_sum += actual_probability[index];
            expected_sum += expected_probability[index];
        }

        constexpr double kEpsilon = 1.0e-300;
        double actual_to_expected = 0.0;
        double expected_to_actual = 0.0;
        for (size_t index = 0; index < width; ++index)
        {
            const double p = actual_probability[index] /
                             std::max(actual_sum, kEpsilon);
            const double q = expected_probability[index] /
                             std::max(expected_sum, kEpsilon);
            actual_to_expected += p * (
                std::log(std::max(p, kEpsilon)) -
                std::log(std::max(q, kEpsilon)));
            expected_to_actual += q * (
                std::log(std::max(q, kEpsilon)) -
                std::log(std::max(p, kEpsilon)));
        }
        return 0.5 * (actual_to_expected + expected_to_actual);
    }

    /**
     * @brief Compare complete FP32 outputs and retain row-wise diagnostics.
     *
     * @param actual Candidate output in native host memory.
     * @param expected Serial-M1 oracle output in native host memory.
     * @param row_width Logical row width for symmetric-KL diagnostics.
     */
    inline FP32Evidence compareFP32(
        const float *actual,
        size_t actual_count,
        const float *expected,
        size_t expected_count,
        size_t row_width)
    {
        FP32Evidence result;
        result.actual_digest = nativeByteDigest(actual, actual_count);
        result.expected_digest = nativeByteDigest(expected, expected_count);
        if (actual_count != expected_count)
        {
            result.mismatch_count = std::max(actual_count, expected_count);
            result.nonfinite_count = result.mismatch_count;
            result.relative_l2 = std::numeric_limits<double>::infinity();
            result.max_abs = std::numeric_limits<double>::infinity();
            result.symmetric_kld = std::numeric_limits<double>::infinity();
            return result;
        }

        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        double difference_norm = 0.0;
        for (size_t index = 0; index < actual_count; ++index)
        {
            if (std::memcmp(
                    actual + index,
                    expected + index,
                    sizeof(float)) != 0)
            {
                if (result.mismatch_count == 0)
                    result.first_mismatch_index = index;
                ++result.mismatch_count;
            }
            if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
            {
                ++result.nonfinite_count;
                continue;
            }
            const double lhs = actual[index];
            const double rhs = expected[index];
            const double difference = lhs - rhs;
            dot += lhs * rhs;
            actual_norm += lhs * lhs;
            expected_norm += rhs * rhs;
            difference_norm += difference * difference;
            result.max_abs = std::max(result.max_abs, std::abs(difference));
        }
        result.cosine =
            (actual_norm < 1.0e-30 && expected_norm < 1.0e-30)
                ? 1.0
                : dot / (std::sqrt(actual_norm) * std::sqrt(expected_norm) + 1.0e-30);
        result.relative_l2 =
            expected_norm < 1.0e-30
                ? (difference_norm < 1.0e-30
                       ? 0.0
                       : std::numeric_limits<double>::infinity())
                : std::sqrt(difference_norm / expected_norm);

        if (row_width != 0 && actual_count % row_width == 0)
        {
            const size_t rows = actual_count / row_width;
            for (size_t row = 0; row < rows; ++row)
            {
                result.symmetric_kld = std::max(
                    result.symmetric_kld,
                    rowSoftmaxSymmetricKLDivergence(
                        actual + row * row_width,
                        expected + row * row_width,
                        row_width));
            }
        }
        return result;
    }

    /** Convenience overload for owning FP32 vectors. */
    inline FP32Evidence compareFP32(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t row_width)
    {
        return compareFP32(
            actual.data(),
            actual.size(),
            expected.data(),
            expected.size(),
            row_width);
    }
} // namespace llaminar2::test::trainer
