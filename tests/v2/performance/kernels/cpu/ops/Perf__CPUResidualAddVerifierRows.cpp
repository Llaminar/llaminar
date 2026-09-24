/**
 * @file Perf__CPUResidualAddVerifierRows.cpp
 * @brief CPU residual-add thread-team crossover benchmark for MTP verifier rows.
 *
 * Qwen MoE combines its routed and shared-expert branches with an elementwise
 * residual add. For grouped verification this is only M times d_model values,
 * so opening a whole-socket OpenMP team can cost much more than the arithmetic.
 * This harness measures that crossover directly for the Qwen3.6-35B-A3B
 * hidden width. It compares the installed production policy, a vectorized
 * caller-thread loop, and bounded OpenMP teams; checks every output byte
 * against the same serial arithmetic; and reports the winning policy for M=1
 * through prefill-sized work.
 *
 * The benchmark deliberately times a fresh team per invocation because that is
 * the production stage boundary being optimized. It performs no MPI work and
 * should be run with physical-core placement, for example:
 *
 * @code
 * OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores \
 *   ./build_v2_integration/tests/v2/v2_perf_cpu_residual_add_verifier_rows
 * @endcode
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <fort.hpp>
#include <omp.h>

#include "kernels/cpu/ops/CPUResidualAddKernelT.h"
#include "tensors/Tensors.h"

namespace
{
    constexpr int kQwen36MoEHiddenWidth = 2048;

    /**
     * @brief Execute the exact elementwise arithmetic on the caller thread.
     *
     * Each output element has one addition and no cross-element reduction, so
     * SIMD changes neither operation order nor output bytes.
     */
    void addSerial(
        const float *input,
        const float *residual,
        float *output,
        size_t element_count)
    {
#pragma omp simd
        for (size_t element = 0; element < element_count; ++element)
            output[element] = input[element] + residual[element];
    }

    /**
     * @brief Execute one residual add with a newly entered bounded OpenMP team.
     *
     * @param thread_count Number of workers participating in this invocation.
     * The caller clamps the value to the process's configured OpenMP maximum.
     */
    void addParallel(
        const float *input,
        const float *residual,
        float *output,
        size_t element_count,
        int thread_count)
    {
#pragma omp parallel for schedule(static) num_threads(thread_count)
        for (size_t element = 0; element < element_count; ++element)
            output[element] = input[element] + residual[element];
    }

    /**
     * @brief Return the minimum per-invocation latency of a repeated candidate.
     *
     * Short rows are repeated inside each clocked sample so timer resolution is
     * not mistaken for kernel economy. The output is consumed after timing to
     * keep the compiler from proving the measured stores dead.
     */
    template <typename Candidate>
    double minimumLatencyUs(
        size_t element_count,
        const float *output,
        Candidate &&candidate)
    {
        const int repetitions = std::clamp(
            static_cast<int>((1u << 18) / std::max<size_t>(element_count, 1)),
            1,
            24);
        for (int warmup = 0; warmup < 3; ++warmup)
            candidate();

        double minimum_us = std::numeric_limits<double>::infinity();
        for (int sample = 0; sample < 9; ++sample)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int repetition = 0; repetition < repetitions; ++repetition)
                candidate();
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed_us =
                std::chrono::duration<double, std::micro>(stop - start).count() /
                static_cast<double>(repetitions);
            minimum_us = std::min(minimum_us, elapsed_us);
        }

        volatile float output_witness = output[element_count / 2];
        (void)output_witness;
        return minimum_us;
    }

    /**
     * @brief Format one latency without hiding sub-microsecond differences.
     */
    std::string latencyText(double microseconds)
    {
        char buffer[32]{};
        std::snprintf(buffer, sizeof(buffer), "%.2f", microseconds);
        return buffer;
    }
} // namespace

TEST(Perf__CPUResidualAddVerifierRows, ThreadTeamCrossoverIsByteExact)
{
    omp_set_dynamic(0);
    const int configured_threads = std::max(1, omp_get_max_threads());
    const std::array<int, 9> row_counts = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const std::array<int, 5> requested_teams = {2, 4, 8, 14, 28};

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "M" << "traffic KiB" << "production us"
          << "serial us";
    for (const int requested : requested_teams)
        table << ("T" + std::to_string(std::min(requested, configured_threads)) + " us");
    table << "winner" << fort::endr;

    for (const int rows : row_counts)
    {
        const size_t element_count =
            static_cast<size_t>(rows) * kQwen36MoEHiddenWidth;
        std::vector<float> input(element_count);
        std::vector<float> residual(element_count);
        std::vector<float> expected(element_count);
        std::vector<float> output(element_count);

        llaminar2::FP32Tensor production_input(
            {static_cast<size_t>(rows),
             static_cast<size_t>(kQwen36MoEHiddenWidth)});
        llaminar2::FP32Tensor production_residual(
            {static_cast<size_t>(rows),
             static_cast<size_t>(kQwen36MoEHiddenWidth)});
        llaminar2::FP32Tensor production_output(
            {static_cast<size_t>(rows),
             static_cast<size_t>(kQwen36MoEHiddenWidth)});

        std::mt19937 generator(0x51A17u + static_cast<unsigned>(rows));
        std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
        for (size_t element = 0; element < element_count; ++element)
        {
            input[element] = distribution(generator);
            residual[element] = distribution(generator);
        }
        addSerial(input.data(), residual.data(), expected.data(), element_count);
        std::copy(input.begin(), input.end(), production_input.mutable_data());
        std::copy(
            residual.begin(), residual.end(),
            production_residual.mutable_data());

        llaminar2::CPUResidualAddKernelT<
            llaminar2::ActivationPrecision::FP32>
            production_kernel;
        bool production_ok = true;
        const double production_us = minimumLatencyUs(
            element_count,
            production_output.data(),
            [&]()
            {
                production_ok = production_kernel.apply_tensor(
                                    &production_input,
                                    &production_residual,
                                    &production_output,
                                    element_count,
                                    nullptr,
                                    -1) &&
                                production_ok;
            });
        ASSERT_TRUE(production_ok)
            << "production residual-add invocation failed for M=" << rows;
        ASSERT_EQ(
            std::memcmp(
                production_output.data(),
                expected.data(),
                element_count * sizeof(float)),
            0)
            << "production arithmetic changed bytes for M=" << rows;

        double best_us = minimumLatencyUs(
            element_count,
            output.data(),
            [&]()
            {
                addSerial(
                    input.data(), residual.data(), output.data(), element_count);
            });
        ASSERT_EQ(
            std::memcmp(
                output.data(),
                expected.data(),
                element_count * sizeof(float)),
            0)
            << "serial SIMD arithmetic changed bytes for M=" << rows;
        std::string winner = "serial";

        table << rows
              << latencyText(
                     static_cast<double>(element_count * sizeof(float) * 3) /
                     1024.0)
              << latencyText(production_us)
              << latencyText(best_us);

        for (const int requested : requested_teams)
        {
            const int team = std::min(requested, configured_threads);
            const double candidate_us = minimumLatencyUs(
                element_count,
                output.data(),
                [&]()
                {
                    addParallel(
                        input.data(), residual.data(), output.data(),
                        element_count, team);
                });
            ASSERT_EQ(
                std::memcmp(
                    output.data(),
                    expected.data(),
                    element_count * sizeof(float)),
                0)
                << "parallel arithmetic changed bytes for M=" << rows
                << " threads=" << team;

            table << latencyText(candidate_us);
            if (candidate_us < best_us)
            {
                best_us = candidate_us;
                winner = "T" + std::to_string(team);
            }
        }
        table << winner << fort::endr;
    }

    std::cout << table.to_string() << std::flush;
}
