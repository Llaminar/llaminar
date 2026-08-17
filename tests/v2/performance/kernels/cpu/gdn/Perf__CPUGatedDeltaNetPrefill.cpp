/**
 * @file Perf__CPUGatedDeltaNetPrefill.cpp
 * @brief Release microbenchmark for byte-exact CPU GDN prefill recurrence.
 *
 * The Qwen 3.6 prefill graph advances Gated Delta Net state in strict token
 * order. This harness isolates that recurrence from model loading, projections,
 * collectives, and routing so kernel changes can be judged against stable
 * warmed measurements. It covers both the full single-socket head geometry and
 * the per-rank geometry produced by two-way tensor parallelism.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "tensors/AlignedVector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <omp.h>

using namespace llaminar2;

namespace
{
    /** @brief Complete persistent input and state storage for one geometry. */
    struct PrefillFixture
    {
        int rows = 0;
        int n_heads = 0;
        int d_k = 0;
        int d_v = 0;
        AlignedVector<float> q;
        AlignedVector<float> k;
        AlignedVector<float> v;
        AlignedVector<float> alpha;
        AlignedVector<float> beta;
        AlignedVector<float> a_log;
        AlignedVector<float> dt_bias;
        AlignedVector<float> initial_state;
        AlignedVector<float> state;
        AlignedVector<float> output;

        PrefillFixture(int row_count, int head_count, int key_width, int value_width)
            : rows(row_count),
              n_heads(head_count),
              d_k(key_width),
              d_v(value_width),
              q(static_cast<size_t>(rows) * n_heads * d_k),
              k(q.size()),
              v(static_cast<size_t>(rows) * n_heads * d_v),
              alpha(static_cast<size_t>(rows) * n_heads),
              beta(alpha.size()),
              a_log(static_cast<size_t>(n_heads)),
              dt_bias(static_cast<size_t>(n_heads)),
              initial_state(static_cast<size_t>(n_heads) * d_k * d_v),
              state(initial_state.size()),
              output(v.size())
        {
            for (size_t i = 0; i < q.size(); ++i)
            {
                q[i] = 0.0021f * static_cast<float>(static_cast<int>(i % 29) - 14);
                k[i] = -0.0017f * static_cast<float>(static_cast<int>(i % 31) - 15);
            }
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = 0.0029f * static_cast<float>(static_cast<int>(i % 23) - 11);
            for (size_t i = 0; i < alpha.size(); ++i)
            {
                alpha[i] = -0.17f + 0.0031f * static_cast<float>(i % 37);
                beta[i] = 0.11f - 0.0023f * static_cast<float>(i % 41);
            }
            for (int head = 0; head < n_heads; ++head)
            {
                a_log[static_cast<size_t>(head)] = -0.35f - 0.002f * head;
                dt_bias[static_cast<size_t>(head)] =
                    0.01f * static_cast<float>((head % 7) - 3);
            }
            for (size_t i = 0; i < initial_state.size(); ++i)
            {
                initial_state[i] =
                    0.0001f * static_cast<float>(static_cast<int>(i % 43) - 21);
            }
        }
    };

    /**
     * @brief Measure warmed recurrence calls and return median milliseconds.
     *
     * State reset occurs outside the timed interval. `CPUGatedDeltaNet` is
     * retained across every launch, so its persistent scratch reaches steady
     * capacity during warmup exactly as it does in a captured model graph.
     */
    double benchmarkPrefill(PrefillFixture &fixture, int warmups, int iterations)
    {
        CPUGatedDeltaNet kernel;
        auto invoke = [&]()
        {
            std::memcpy(
                fixture.state.data(),
                fixture.initial_state.data(),
                fixture.initial_state.size() * sizeof(float));
            return kernel.chunk_forward(
                fixture.q.data(),
                fixture.k.data(),
                fixture.v.data(),
                fixture.alpha.data(),
                fixture.beta.data(),
                fixture.a_log.data(),
                fixture.dt_bias.data(),
                fixture.output.data(),
                fixture.state.data(),
                fixture.rows,
                fixture.n_heads,
                fixture.d_k,
                fixture.d_v,
                /*chunk_size=*/64,
                /*use_qk_l2norm=*/true);
        };

        for (int warmup = 0; warmup < warmups; ++warmup)
            EXPECT_TRUE(invoke());

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(iterations));
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            std::memcpy(
                fixture.state.data(),
                fixture.initial_state.data(),
                fixture.initial_state.size() * sizeof(float));
            const auto begin = std::chrono::steady_clock::now();
            EXPECT_TRUE(kernel.chunk_forward(
                fixture.q.data(), fixture.k.data(), fixture.v.data(),
                fixture.alpha.data(), fixture.beta.data(),
                fixture.a_log.data(), fixture.dt_bias.data(),
                fixture.output.data(), fixture.state.data(),
                fixture.rows, fixture.n_heads, fixture.d_k, fixture.d_v,
                /*chunk_size=*/64,
                /*use_qk_l2norm=*/true));
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(end - begin).count());
        }

        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }
} // namespace

/**
 * @brief Report the warmed Qwen 3.6 prefill recurrence economy at TP=1/2.
 */
TEST(Perf__CPUGatedDeltaNetPrefill, Qwen36SingleAndTwoWayTPHeadGeometries)
{
    omp_set_dynamic(0);
    int threads = std::max(2, omp_get_max_threads());
    if (const char *configured = std::getenv("LLAMINAR_CPU_GDN_PERF_THREADS"))
        threads = std::max(2, std::stoi(configured));
    omp_set_num_threads(threads);

    constexpr int rows = 434;
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    constexpr int warmups = 3;
    constexpr int iterations = 11;

    PrefillFixture single(rows, /*n_heads=*/16, d_k, d_v);
    PrefillFixture tp2_rank(rows, /*n_heads=*/8, d_k, d_v);
    const double single_ms = benchmarkPrefill(single, warmups, iterations);
    const double tp2_rank_ms = benchmarkPrefill(tp2_rank, warmups, iterations);

    const float checksum = std::accumulate(
        tp2_rank.output.begin(), tp2_rank.output.end(), 0.0f);
    ASSERT_TRUE(std::isfinite(checksum));
    ASSERT_GT(single_ms, 0.0);
    ASSERT_GT(tp2_rank_ms, 0.0);

    std::cout << "CPU GDN prefill M=" << rows
              << " d_k=" << d_k
              << " d_v=" << d_v
              << " threads=" << threads
              << " single_heads16_ms=" << single_ms
              << " tp2_local_heads8_ms=" << tp2_rank_ms
              << " head_halving_speedup=" << (single_ms / tp2_rank_ms)
              << '\n';
}
