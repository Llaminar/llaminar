/**
 * @file Perf__CPUExpertQ8Publication.cpp
 * @brief Isolated production Q8 publication speedometer with no model setup.
 *
 * One invocation measures one row/width geometry. Profiler mode repeats only
 * that kernel, never an oracle or a second candidate. Canonical timings have
 * no profiler and keep input construction and warmup outside the samples.
 * Functional certification lives in CPUExplicitRoundingContract, not a noisy
 * latency threshold in production preflight.
 */
#include "kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
/**
 * @brief Read a positive per-process benchmark geometry or iteration count.
 * @param name Environment setting owned by this isolated harness.
 * @param default_value Value used when the setting is absent.
 * @return Validated positive count.
 * @throws std::invalid_argument for malformed or out-of-range input.
 */
int publicationSetting(const char *name, int default_value)
{
    const char *raw = std::getenv(name);
    if (!raw) return default_value;
    char *end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end || value < 1 || value > 100000000)
        throw std::invalid_argument(name);
    return int(value);
}

/**
 * @brief Publish exactly the active blocks, retaining an isolated profiler symbol.
 * @param input Contiguous FP32 active rows.
 * @param output Admitted Q8 destination of rows * ceil(width/32) blocks.
 * @param rows Active rows, never buffer capacity.
 * @param width FP32 elements per row.
 */
[[gnu::noinline]] void publishExpertQ8(const float *input, Q8_1Block *output,
                                     int rows, int width)
{
    const int blocks = (width + 31) / 32;
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
            cpu::gpu_aligned_expert_q8::quantizeBlock(
                input + size_t(row) * width + block * 32,
                output[size_t(row) * blocks + block],
                std::min(32, width - block * 32));
}

/** @test Time one byte-certified production publication geometry. */
TEST(Perf_CPUExpertQ8Publication, ActiveRows)
{
    const int rows = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_ROWS", 20);
    const int width = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_WIDTH", 2048);
    std::vector<float> source(size_t(rows) * width);
    std::vector<Q8_1Block> output(size_t(rows) * ((width + 31) / 32));
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = float(int((i * 71 + i / 13) % 257) - 128) * 0.003125f;
    auto run = [&] { publishExpertQ8(source.data(), output.data(), rows, width); };
    for (int warmup = 0; warmup < 10; ++warmup) run();
    if (std::getenv("LLAMINAR_CPU_EXPERT_Q8_PROFILE_ITERATIONS"))
    {
        const int iterations = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_PROFILE_ITERATIONS", 1);
        for (int iteration = 0; iteration < iterations; ++iteration) run();
        std::cout << "profile_only,rows=" << rows << ",width=" << width
                  << ",iterations=" << iterations << '\n';
        return;
    }
    constexpr int iterations = 512;
    std::vector<double> samples;
    for (int sample = 0; sample < 9; ++sample)
    {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) run();
        samples.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - begin).count() / iterations);
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "rows,width,median_us,min_us,max_us\n" << rows << ',' << width
              << ',' << samples[4] << ',' << samples.front() << ',' << samples.back() << '\n';
}

/**
 * @test Measure the exact transported ExpertOverlay publication entry point.
 *
 * Expert-major routes may repeat source tokens. The source capacity, active
 * route count and hidden width are independent, just as in the production
 * ticket. All storage is fixed before timing, and an independent serial block
 * walk checks every output byte. Run this test alone for profiler attribution;
 * it includes the production worksharing decision, not a hand-written loop.
 */
TEST(Perf_CPUExpertQ8Publication, TransportedExpertMajor)
{
    const int rows = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_ROWS", 20);
    const int source_rows = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_SOURCE_ROWS", 5);
    const int width = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_WIDTH", 2048);
    const int blocks = (width + 31) / 32;
    const bool profile_only = std::getenv("LLAMINAR_CPU_EXPERT_Q8_PROFILE_ITERATIONS") != nullptr;
    std::vector<float> source(size_t(source_rows) * width);
    std::vector<int> indices(rows);
    std::vector<Q8_1Block> output(size_t(rows) * blocks);
    std::vector<Q8_1Block> expected(output.size());
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = float(int((i * 71 + i / 13) % 257) - 128) * 0.003125f;
    for (int row = 0; row < rows; ++row)
    {
        indices[row] = row % source_rows;
        if (!profile_only)
            publishExpertQ8(source.data() + size_t(indices[row]) * width,
                           expected.data() + size_t(row) * blocks, 1, width);
    }
    CPUMoEKernel kernel;
    auto run = [&] {
        return kernel.publishTransportedRouterQ8HiddenExpertMajor(
            source.data(), source_rows, width, indices, output);
    };
    ASSERT_TRUE(run());
    if (!profile_only)
        ASSERT_EQ(std::memcmp(output.data(), expected.data(), output.size() * sizeof(Q8_1Block)), 0);
    const auto warmup_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    do { ASSERT_TRUE(run()); } while (std::chrono::steady_clock::now() < warmup_end);
    if (profile_only)
    {
        const int iterations = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_PROFILE_ITERATIONS", 1);
        for (int iteration = 0; iteration < iterations; ++iteration)
            ASSERT_TRUE(run());
        std::cout << "profile_only,rows=" << rows << ",source_rows=" << source_rows
                  << ",width=" << width << ",iterations=" << iterations << '\n';
        return;
    }
    const int iterations = publicationSetting("LLAMINAR_CPU_EXPERT_Q8_ITERS", 512);
    std::vector<double> samples;
    for (int sample = 0; sample < 9; ++sample)
    {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
            ASSERT_TRUE(run());
        samples.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - begin).count() / iterations);
    }
    ASSERT_EQ(std::memcmp(output.data(), expected.data(), output.size() * sizeof(Q8_1Block)), 0);
    std::sort(samples.begin(), samples.end());
    std::cout << "rows,source_rows,width,median_us,min_us,max_us\n"
              << rows << ',' << source_rows << ',' << width << ',' << samples[4]
              << ',' << samples.front() << ',' << samples.back() << '\n';
}
} // namespace llaminar2::test
