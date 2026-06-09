#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "kernels/common/SamplingMath.h"
#include "kernels/cpu/sampling/CPUSamplerPrimitives.h"
#include "utils/Sampler.h"

using namespace llaminar2;

namespace
{
    volatile double g_sampler_perf_sink = 0.0;

    std::vector<float> make_logits(size_t vocab_size)
    {
        std::vector<float> logits(vocab_size);
        std::mt19937 rng(1337);
        std::normal_distribution<float> noise(0.0f, 2.5f);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            const float periodic = std::sin(static_cast<float>(i) * 0.00491f) * 3.0f;
            const float local = static_cast<float>((i * 7919u) % 997u) * 0.00019f;
            logits[i] = periodic + local + noise(rng);
        }
        return logits;
    }

    double checksum_distribution(const std::vector<SamplingDistributionEntry> &distribution)
    {
        double checksum = 0.0;
        for (const auto &entry : distribution)
        {
            checksum += static_cast<double>(entry.token_id + 1) *
                        static_cast<double>(entry.probability);
        }
        return checksum;
    }

    std::vector<SamplingDistributionEntry> old_partial_sort_distribution(
        const float *logits,
        size_t vocab_size,
        const SamplingParams &params)
    {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            candidates.emplace_back(logits[i], static_cast<int>(i));
        }

        const int k = std::min<int>(params.top_k, static_cast<int>(candidates.size()));
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + k,
            candidates.end(),
            [](const auto &a, const auto &b)
            {
                return a.first > b.first;
            });

        std::vector<float> sorted_logits(static_cast<size_t>(k));
        std::vector<int> sorted_ids(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i)
        {
            sorted_logits[static_cast<size_t>(i)] = candidates[static_cast<size_t>(i)].first;
            sorted_ids[static_cast<size_t>(i)] = candidates[static_cast<size_t>(i)].second;
        }

        std::vector<float> scratch(static_cast<size_t>(k), 0.0f);
        std::vector<int> out_ids(static_cast<size_t>(k), -1);
        std::vector<float> out_probs(static_cast<size_t>(k), 0.0f);
        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            k,
            params.top_p,
            params.temperature,
            out_ids.data(),
            out_probs.data(),
            scratch.data());

        std::vector<SamplingDistributionEntry> distribution;
        distribution.reserve(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i)
        {
            if (out_ids[static_cast<size_t>(i)] >= 0 &&
                out_probs[static_cast<size_t>(i)] > 0.0f)
            {
                distribution.push_back({out_ids[static_cast<size_t>(i)],
                                        out_probs[static_cast<size_t>(i)]});
            }
        }
        return distribution;
    }

    std::vector<SamplingDistributionEntry> new_isa_distribution(
        const float *logits,
        size_t vocab_size,
        const SamplingParams &params)
    {
        const int k = std::min<int>(params.top_k, static_cast<int>(vocab_size));
        std::vector<float> sorted_logits(static_cast<size_t>(k));
        std::vector<int> sorted_ids(static_cast<size_t>(k));
        const int selected = cpu_sampling::select_topk(
            logits,
            static_cast<int>(vocab_size),
            k,
            sorted_logits.data(),
            sorted_ids.data());
        if (selected <= 0)
        {
            return {};
        }

        std::vector<float> scratch(static_cast<size_t>(selected), 0.0f);
        std::vector<int> out_ids(static_cast<size_t>(selected), -1);
        std::vector<float> out_probs(static_cast<size_t>(selected), 0.0f);
        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            selected,
            params.top_p,
            params.temperature,
            out_ids.data(),
            out_probs.data(),
            scratch.data());

        std::vector<SamplingDistributionEntry> distribution;
        distribution.reserve(static_cast<size_t>(selected));
        for (int i = 0; i < selected; ++i)
        {
            if (out_ids[static_cast<size_t>(i)] >= 0 &&
                out_probs[static_cast<size_t>(i)] > 0.0f)
            {
                distribution.push_back({out_ids[static_cast<size_t>(i)],
                                        out_probs[static_cast<size_t>(i)]});
            }
        }
        return distribution;
    }

    template <typename Fn>
    double time_ms(Fn &&fn, int warmup_iters, int bench_iters)
    {
        for (int i = 0; i < warmup_iters; ++i)
        {
            g_sampler_perf_sink += fn();
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; ++i)
        {
            g_sampler_perf_sink += fn();
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() /
               static_cast<double>(bench_iters);
    }

    void print_header()
    {
        std::cout << "\nCPU Sampler Top-k Distribution Perf\n";
        std::cout << "vocab,top_k,top_p,old_partial_sort_ms,new_isa_ms,speedup\n";
    }

    void print_row(
        size_t vocab_size,
        int top_k,
        float top_p,
        double old_ms,
        double new_ms)
    {
        std::cout << vocab_size << ","
                  << top_k << ","
                  << std::fixed << std::setprecision(2) << top_p << ","
                  << std::setprecision(6) << old_ms << ","
                  << new_ms << ","
                  << (old_ms / std::max(new_ms, 1e-9)) << "\n";
    }
} // namespace

TEST(Perf__CPUSamplerTopK, QwenStyleTopKDistributionBeforeAfter)
{
    constexpr size_t kQwenStyleVocab = 151936;
    const auto logits = make_logits(kQwenStyleVocab);

    print_header();
    for (int top_k : {20, 40, sampling_math::kMaxTopK})
    {
        SamplingParams params;
        params.temperature = 0.7f;
        params.top_k = top_k;
        params.top_p = 0.95f;

        const auto old_distribution =
            old_partial_sort_distribution(logits.data(), logits.size(), params);
        const auto new_distribution =
            new_isa_distribution(logits.data(), logits.size(), params);
        ASSERT_EQ(new_distribution.size(), old_distribution.size());
        for (size_t i = 0; i < old_distribution.size(); ++i)
        {
            ASSERT_EQ(new_distribution[i].token_id, old_distribution[i].token_id);
            ASSERT_NEAR(new_distribution[i].probability,
                        old_distribution[i].probability,
                        1e-6f);
        }

        const int bench_iters = top_k >= 128 ? 80 : 160;
        const double old_ms = time_ms(
            [&]() -> double
            {
                return checksum_distribution(
                    old_partial_sort_distribution(logits.data(), logits.size(), params));
            },
            5,
            bench_iters);
        const double new_ms = time_ms(
            [&]() -> double
            {
                return checksum_distribution(
                    new_isa_distribution(logits.data(), logits.size(), params));
            },
            5,
            bench_iters);

        print_row(logits.size(), top_k, params.top_p, old_ms, new_ms);
    }

    EXPECT_NE(g_sampler_perf_sink, 0.0);
}
