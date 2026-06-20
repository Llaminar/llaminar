#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "execution/mtp/MTPDepthController.h"

using namespace llaminar2;

namespace
{
    volatile uint64_t g_depth_controller_sink = 0;

    MTPDepthPolicyConfig make_dynamic_config(
        int initial_depth,
        int max_depth,
        int window_size,
        int min_depth = 1)
    {
        MTPDepthPolicyConfig config;
        config.mode = MTPDepthPolicyMode::Dynamic;
        config.min_depth = min_depth;
        config.max_depth = max_depth;
        config.initial_depth = initial_depth;
        config.window_size = window_size;
        config.min_samples = window_size;
        config.cooldown_steps = 0;
        config.promote_consecutive_windows = 2;
        config.promote_full_accept_rate = 1.0;
        config.demote_zero_accept_rate = 0.30;
        config.demote_acceptance_rate = 0.55;
        return config;
    }

    MTPDepthObservation observation_for_step(int step, int depth)
    {
        const int accepted =
            step % 9 == 0 ? 0
                          : (step % 5 == 0 ? std::max(0, depth - 1) : depth);
        return MTPDepthObservation{
            .requested_depth = depth,
            .effective_depth = depth,
            .accepted_speculative_prefix = accepted,
            .budget_limited = false,
            .rollback = accepted < depth,
        };
    }

    template <typename Fn>
    double time_ns_per_op(Fn &&fn, int warmup_iters, int bench_iters)
    {
        for (int i = 0; i < warmup_iters; ++i)
        {
            g_depth_controller_sink += static_cast<uint64_t>(fn(i));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; ++i)
        {
            g_depth_controller_sink += static_cast<uint64_t>(fn(i));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ns =
            std::chrono::duration<double, std::nano>(end - start).count();
        return total_ns / static_cast<double>(bench_iters);
    }

    void print_row(const char *name, int iterations, double ns_per_op)
    {
        std::cout << name << ","
                  << iterations << ","
                  << std::fixed << std::setprecision(3)
                  << ns_per_op << "\n";
    }
}

TEST(Perf__MTPDepthController, RecordStepIsTinyConstantTimeBookkeeping)
{
    constexpr int kWarmupIters = 10000;
    constexpr int kBenchIters = 5000000;

    std::cout << "\nMTP Depth Controller Perf\n";
    std::cout << "workload,iterations,ns_per_op\n";

    MTPDepthController window16(
        make_dynamic_config(/*initial_depth=*/1, /*max_depth=*/3, /*window_size=*/16),
        /*configured_draft_tokens=*/3);
    const double window16_ns = time_ns_per_op(
        [&](int step) -> int
        {
            const int depth = window16.requestedDepthForStep();
            const auto decision = window16.recordStep(observation_for_step(step, depth));
            return depth + decision.new_depth + static_cast<int>(decision.reason);
        },
        kWarmupIters,
        kBenchIters);
    print_row("dynamic_window16_record_step", kBenchIters, window16_ns);

    MTPDepthController window1(
        make_dynamic_config(/*initial_depth=*/3, /*max_depth=*/3, /*window_size=*/1),
        /*configured_draft_tokens=*/3);
    const double window1_ns = time_ns_per_op(
        [&](int step) -> int
        {
            const int depth = window1.requestedDepthForStep();
            const auto decision = window1.recordStep(observation_for_step(step, depth));
            return depth + decision.new_depth + static_cast<int>(decision.reason);
        },
        kWarmupIters,
        kBenchIters);
    print_row("dynamic_window1_record_step", kBenchIters, window1_ns);

    MTPDepthController depth0(
        make_dynamic_config(
            /*initial_depth=*/0,
            /*max_depth=*/3,
            /*window_size=*/4,
            /*min_depth=*/0),
        /*configured_draft_tokens=*/3);
    const double bypass_ns = time_ns_per_op(
        [&](int) -> int
        {
            const auto decision = depth0.recordBypassStep();
            return depth0.requestedDepthForStep() + decision.new_depth;
        },
        kWarmupIters,
        kBenchIters);
    print_row("dynamic_depth0_bypass_step", kBenchIters, bypass_ns);

    EXPECT_LT(window16_ns, 1000.0);
    EXPECT_LT(window1_ns, 1000.0);
    EXPECT_LT(bypass_ns, 1000.0);
}
