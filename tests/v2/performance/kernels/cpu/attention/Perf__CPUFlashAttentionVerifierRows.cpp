/**
 * @file Perf__CPUFlashAttentionVerifierRows.cpp
 * @brief Strict CPU attention verifier-row equivalence and economy microbench.
 *
 * MTP verification needs small groups of target rows (normally M=2..4) to be
 * computed together while remaining decode-equivalent to running one decode
 * row at a time.  This benchmark isolates the CPU flash-attention stage of
 * that contract:
 *
 * - The grouped path calls the production
 *   `CPUFlashAttentionKernelT::compute_verifier_rows_decode_equivalent()` API.
 * - The oracle calls the same production decode path M times with `seq_len=1`.
 * - Future verifier KV rows are present in both paths and hidden by the causal
 *   mask, matching how an all-row verifier graph sees draft/bonus rows.
 *
 * The test reports speedup and verifies full output vectors with cosine,
 * relative L2, max absolute error, and a softmax-space symmetric KL check.  The
 * KL is over the output vector rather than attention probabilities; it is a
 * drift tripwire, not a semantic claim about the output distribution.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    struct Metrics
    {
        double cosine = 1.0;
        double relative_l2 = 0.0;
        double max_abs = 0.0;
        double min_row_cosine = 1.0;
        double max_row_relative_l2 = 0.0;
        double max_row_symmetric_kl = 0.0;
        size_t worst_row = 0;
    };

    struct Timing
    {
        double min_us = 0.0;
        double mean_us = 0.0;
    };

    int envInt(const char *name, int fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        return (end != value && parsed > 0) ? static_cast<int>(parsed) : fallback;
    }

    void fillDeterministic(std::vector<float> &values, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.35f, 0.35f);
        for (float &v : values)
            v = dist(rng);
    }

    /**
     * @brief KL(lhs || rhs) after stable softmax over one output row.
     */
    double rowKL(const float *lhs, const float *rhs, size_t width)
    {
        double max_l = -std::numeric_limits<double>::infinity();
        double max_r = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < width; ++i)
        {
            max_l = std::max(max_l, static_cast<double>(lhs[i]));
            max_r = std::max(max_r, static_cast<double>(rhs[i]));
        }

        double sum_l = 0.0;
        double sum_r = 0.0;
        for (size_t i = 0; i < width; ++i)
        {
            sum_l += std::exp(static_cast<double>(lhs[i]) - max_l);
            sum_r += std::exp(static_cast<double>(rhs[i]) - max_r);
        }

        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < width; ++i)
        {
            const double p =
                std::exp(static_cast<double>(lhs[i]) - max_l) / std::max(sum_l, kEps);
            const double q =
                std::exp(static_cast<double>(rhs[i]) - max_r) / std::max(sum_r, kEps);
            kl += p * (std::log(std::max(p, kEps)) - std::log(std::max(q, kEps)));
        }
        return kl;
    }

    Metrics compareRows(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        size_t row_width)
    {
        EXPECT_EQ(actual.size(), expected.size());
        Metrics metrics;

        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        double diff2 = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(actual[i])) << "actual[" << i << "]";
            EXPECT_TRUE(std::isfinite(expected[i])) << "expected[" << i << "]";
            const double a = actual[i];
            const double e = expected[i];
            const double d = a - e;
            dot += a * e;
            actual_norm += a * a;
            expected_norm += e * e;
            diff2 += d * d;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(d));
        }

        metrics.cosine =
            dot / (std::sqrt(actual_norm) * std::sqrt(expected_norm) + 1.0e-30);
        metrics.relative_l2 =
            std::sqrt(diff2) / (std::sqrt(expected_norm) + 1.0e-30);

        const size_t rows = row_width == 0 ? 0 : actual.size() / row_width;
        for (size_t row = 0; row < rows; ++row)
        {
            const float *a_row = actual.data() + row * row_width;
            const float *e_row = expected.data() + row * row_width;
            double row_dot = 0.0;
            double row_actual_norm = 0.0;
            double row_expected_norm = 0.0;
            double row_diff2 = 0.0;
            for (size_t i = 0; i < row_width; ++i)
            {
                const double a = a_row[i];
                const double e = e_row[i];
                const double d = a - e;
                row_dot += a * e;
                row_actual_norm += a * a;
                row_expected_norm += e * e;
                row_diff2 += d * d;
            }
            const double row_cos =
                row_dot / (std::sqrt(row_actual_norm) *
                               std::sqrt(row_expected_norm) +
                           1.0e-30);
            const double row_rel =
                std::sqrt(row_diff2) / (std::sqrt(row_expected_norm) + 1.0e-30);
            const double sym_kl =
                0.5 * (rowKL(a_row, e_row, row_width) +
                       rowKL(e_row, a_row, row_width));
            if (row_cos < metrics.min_row_cosine ||
                row_rel > metrics.max_row_relative_l2 ||
                sym_kl > metrics.max_row_symmetric_kl)
            {
                metrics.worst_row = row;
            }
            metrics.min_row_cosine = std::min(metrics.min_row_cosine, row_cos);
            metrics.max_row_relative_l2 =
                std::max(metrics.max_row_relative_l2, row_rel);
            metrics.max_row_symmetric_kl =
                std::max(metrics.max_row_symmetric_kl, sym_kl);
        }
        return metrics;
    }

    template <typename Fn>
    Timing timeFn(int warmup, int iterations, Fn &&fn)
    {
        for (int i = 0; i < warmup; ++i)
            fn();

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(iterations));
        for (int i = 0; i < iterations; ++i)
        {
            const auto start = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::micro>(end - start).count());
        }

        const double sum =
            std::accumulate(samples.begin(), samples.end(), 0.0);
        return {
            *std::min_element(samples.begin(), samples.end()),
            sum / static_cast<double>(samples.size())};
    }

    /**
     * @brief Run one Qwen3.6-shaped attention verifier-row case.
     */
    void runCase(int rows)
    {
        constexpr int n_heads = 40;
        constexpr int n_kv_heads = 8;
        constexpr int head_dim = 128;
        constexpr bool causal = true;

        const int context_len =
            envInt("LLAMINAR_CPU_ATTN_VERIFIER_CONTEXT", 2048);
        const int warmup =
            envInt("LLAMINAR_CPU_ATTN_VERIFIER_WARMUP", 3);
        const int iterations =
            envInt("LLAMINAR_CPU_ATTN_VERIFIER_ITERS", 20);

        const int kv_len = context_len + rows;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const size_t q_size = static_cast<size_t>(rows) * q_stride;
        const size_t kv_size = static_cast<size_t>(kv_len) * kv_stride;
        const size_t out_size = static_cast<size_t>(rows) * q_stride;

        std::vector<float> Q(q_size);
        std::vector<float> K(kv_size);
        std::vector<float> V(kv_size);
        std::vector<float> grouped(out_size);
        std::vector<float> serial(out_size);

        fillDeterministic(Q, 101u + static_cast<uint32_t>(rows));
        fillDeterministic(K, 202u + static_cast<uint32_t>(rows));
        fillDeterministic(V, 303u + static_cast<uint32_t>(rows));

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        auto q_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows),
                                static_cast<size_t>(q_stride)});
        auto k_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len),
                                static_cast<size_t>(kv_stride)});
        auto v_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len),
                                static_cast<size_t>(kv_stride)});
        auto grouped_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows),
                                static_cast<size_t>(q_stride)});
        std::copy(Q.begin(), Q.end(), q_tensor->mutable_data());
        std::copy(K.begin(), K.end(), k_tensor->mutable_data());
        std::copy(V.begin(), V.end(), v_tensor->mutable_data());

        auto run_grouped = [&]()
        {
            ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(
                q_tensor.get(),
                k_tensor.get(),
                v_tensor.get(),
                grouped_tensor.get(),
                rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                causal));
        };

        auto run_serial = [&]()
        {
            for (int row = 0; row < rows; ++row)
            {
                ASSERT_TRUE(kernel.compute_decode(
                    Q.data() + static_cast<size_t>(row) * q_stride,
                    K.data(),
                    V.data(),
                    serial.data() + static_cast<size_t>(row) * q_stride,
                    1, kv_len, n_heads, n_kv_heads, head_dim,
                    causal, context_len + row));
            }
        };

        run_grouped();
        std::copy(grouped_tensor->data(),
                  grouped_tensor->data() + grouped.size(),
                  grouped.begin());
        run_serial();
        const Metrics metrics = compareRows(grouped, serial, q_stride);
        EXPECT_GE(metrics.cosine, 0.999999)
            << "rel_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_symmetric_kl=" << metrics.max_row_symmetric_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.relative_l2, 1.0e-5)
            << "cosine=" << metrics.cosine
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_symmetric_kl=" << metrics.max_row_symmetric_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_GE(metrics.min_row_cosine, 0.999999)
            << "cosine=" << metrics.cosine
            << " rel_l2=" << metrics.relative_l2
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_symmetric_kl=" << metrics.max_row_symmetric_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_relative_l2, 1.0e-5)
            << "cosine=" << metrics.cosine
            << " rel_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_symmetric_kl=" << metrics.max_row_symmetric_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_symmetric_kl, 1.0e-9)
            << "cosine=" << metrics.cosine
            << " rel_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " worst_row=" << metrics.worst_row;

        const Timing grouped_timing = timeFn(warmup, iterations, run_grouped);
        const Timing serial_timing = timeFn(warmup, iterations, run_serial);
        const double speedup =
            grouped_timing.min_us > 0.0
                ? serial_timing.min_us / grouped_timing.min_us
                : 0.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "backend,phase,m,context,grouped_min_us,serial_min_us,"
                     "speedup,cosine,relative_l2,max_abs,min_row_cosine,"
                     "max_row_relative_l2,max_row_symmetric_kl,worst_row\n"
                  << "cpu,flash_attention_verifier_rows,"
                  << rows << ','
                  << context_len << ','
                  << grouped_timing.min_us << ','
                  << serial_timing.min_us << ','
                  << speedup << ','
                  << metrics.cosine << ','
                  << metrics.relative_l2 << ','
                  << metrics.max_abs << ','
                  << metrics.min_row_cosine << ','
                  << metrics.max_row_relative_l2 << ','
                  << metrics.max_row_symmetric_kl << ','
                  << metrics.worst_row << '\n';

        /*
         * Attention is not expected to produce a GEMV-like M=4 multiplier: the
         * grouped path still does causal work for each verifier row.  It must,
         * however, avoid being worse than the serial verifier oracle.
         */
        EXPECT_GT(speedup, 1.0)
            << "CPU attention verifier rows are correct but not economical at M="
            << rows;
    }
}

TEST(Perf__CPUFlashAttentionVerifierRows, M234_GroupedDecodeMatchesSerial)
{
    for (int rows : {2, 3, 4})
        runCase(rows);
}
