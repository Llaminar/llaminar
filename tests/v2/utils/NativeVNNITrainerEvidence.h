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
