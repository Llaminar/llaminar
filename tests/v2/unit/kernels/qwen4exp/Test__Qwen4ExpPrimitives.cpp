/**
 * @file Test__Qwen4ExpPrimitives.cpp
 * @brief Device-free arithmetic and adversarial history regressions for GR/PLE.
 *
 * Independent scalar formulas authenticate branch order and signed-square-root
 * gating. Chunked and serial production calls must then agree byte-for-byte at
 * every accepted prefix, including zero acceptance, EOS boundaries, short
 * chunks, partial SIMD tails and shrinking replay. No model or GPU is loaded.
 */
#include <gtest/gtest.h>
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "utils/OpenMPUtils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Small exact-integer fixture whose multipliers exceed float64 precision. */
NgramHashPolicy hashPolicy()
{
    NgramHashPolicy policy;
    policy.multipliers = {9007199254740993ULL, 9007199254740995ULL, 9007199254740997ULL};
    policy.moduli = {1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097};
    std::uint64_t offset = 0;
    for (int head = 0; head < 16; ++head)
    {
        policy.offsets[head] = offset;
        offset += policy.moduli[head];
    }
    return policy;
}
/** @brief Deterministic finite FP32 fixtures with channel- and row-dependent signs. */
std::vector<float> values(std::size_t count, int phase = 0)
{
    std::vector<float> output(count);
    for (std::size_t index = 0; index < count; ++index)
        output[index] = static_cast<float>((static_cast<int>(index % 53) * 17 + phase) % 61 - 30) / 37.0f;
    return output;
}
/** @brief Assert raw FP32 identity, retaining signed-zero and tail evidence. */
void expectBytes(const std::vector<float> &a, const std::vector<float> &b)
{
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
}
} // namespace

/** @test Each branch uses its own actual gain, with no second zero-centered offset. */
TEST(Qwen4ExpPrimitives, BranchRMSNormDistinctActualGainsAndGroupedBytes)
{
    auto kernel = createCPUQwen4ExpKernels();
    for (int threads : {1, 3})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        for (int hidden : {1, 7, 19, 257, 2560})
        {
            constexpr int rows = 15;
            const int channels = hidden * 4;
            auto input = values(rows * channels);
            auto gain = values(channels, 31);
            std::vector<float> grouped(rows * channels), serial(rows * channels);
            for (int branch = 0; branch < 4; ++branch)
                for (int channel = 0; channel < hidden; ++channel)
                    gain[branch * hidden + channel] += branch * 0.5f;
            kernel->branchRMSNorm({{rows}, hidden, 1e-6f, input.data(), gain.data(), grouped.data()}, nullptr);
            for (int row = 0; row < rows; ++row)
                kernel->branchRMSNorm({{1}, hidden, 1e-6f, input.data() + row * channels, gain.data(),
                                       serial.data() + row * channels}, nullptr);
            expectBytes(grouped, serial);
            for (int task = 0; task < rows * 4; ++task)
            {
                double sum = 0;
                for (int channel = 0; channel < hidden; ++channel)
                {
                    const double value = input[task * hidden + channel];
                    sum += value * value;
                }
                const double inverse = 1 / std::sqrt(sum / hidden + 1e-6);
                for (int channel = 0; channel < hidden; ++channel)
                    EXPECT_NEAR(grouped[task * hidden + channel],
                        input[task * hidden + channel] * inverse * gain[(task % 4) * hidden + channel], 2e-6);
            }
            auto inPlace = input;
            kernel->branchRMSNorm({{rows}, hidden, 1e-6f, inPlace.data(), gain.data(), inPlace.data()}, nullptr);
            expectBytes(inPlace, grouped);
        }
    }
}

/** @test The checkpoint's /4 placement, branch layout and residual additions are explicit. */
TEST(Qwen4ExpPrimitives, ResidualEquationsAndBranchOrder)
{
    llaminar::v2::ThreadCountGuard team(1);
    auto kernel = createCPUQwen4ExpKernels();
    constexpr int hidden = 19;
    auto input = values(hidden), branches = values(hidden * 4, 3), logits = values(hidden * 4, 9);
    std::vector<float> expanded(hidden * 4), mixed(hidden), written(hidden * 4), down(hidden);
    kernel->residualExpand({{1}, hidden, input.data(), expanded.data()}, nullptr);
    for (int branch = 0; branch < 4; ++branch)
        for (int channel = 0; channel < hidden; ++channel)
            EXPECT_EQ(expanded[branch * hidden + channel], input[channel]);
    kernel->residualDown({{1}, hidden, input.data(), down.data()}, nullptr);
    for (int channel = 0; channel < hidden; ++channel)
    {
        const float scaled = input[channel] / 4;
        EXPECT_NEAR(down[channel], scaled / (1 + std::exp(-scaled)), 1e-6);
    }
    kernel->residualRead({{1}, hidden, branches.data(), logits.data(), mixed.data()}, nullptr);
    for (int channel = 0; channel < hidden; ++channel)
    {
        float sum = 0;
        for (int branch = 0; branch < 4; ++branch)
            sum += branches[branch * hidden + channel] / (1 + std::exp(-logits[branch * hidden + channel]));
        EXPECT_NEAR(mixed[channel], sum / 4, 1e-6);
    }
    const float inject[] = {-4, 0, 4, 8};
    kernel->residualWrite({{1}, hidden, branches.data(), input.data(), inject, written.data()}, nullptr);
    for (int branch = 0; branch < 4; ++branch)
        for (int channel = 0; channel < hidden; ++channel)
            EXPECT_NEAR(written[branch * hidden + channel],
                branches[branch * hidden + channel] + 2 * input[channel] / (1 + std::exp(-inject[branch] / 4)), 1e-6);
}

/** @test Zero, positive and negative normalized similarities preserve the sign-zero convention. */
TEST(Qwen4ExpPrimitives, SignedSquareRootGateIndependentOracle)
{
    llaminar::v2::ThreadCountGuard team(1);
    auto kernel = createCPUQwen4ExpKernels();
    constexpr int hidden = 4;
    std::vector<float> key(hidden * 4, 1), query(hidden * 4, 1), value(hidden, 2), gated(hidden * 4);
    for (int channel = 0; channel < hidden; ++channel)
    {
        query[hidden + channel] = -1;
        query[2 * hidden + channel] = channel % 2 ? -1 : 1;
        query[3 * hidden + channel] = 1e-12f;
    }
    kernel->pleGate({{1}, hidden, key.data(), query.data(), value.data(), gated.data()}, nullptr);
    const float expected[] = {2 / (1 + std::exp(-std::sqrt(2.0f))),
                              2 / (1 + std::exp(std::sqrt(2.0f))), 1,
                              2 / (1 + std::exp(-0.001f))};
    for (int branch = 0; branch < 4; ++branch)
        for (int channel = 0; channel < hidden; ++channel)
            EXPECT_FLOAT_EQ(gated[branch * hidden + channel], expected[branch]);
}

/** @test Exact unsigned hashing, EOS truncation and one/two-token commits agree with an independent oracle. */
TEST(Qwen4ExpPrimitives, HashConstantsEOSAndEveryAcceptedPrefix)
{
    llaminar::v2::ThreadCountGuard team(1);
    auto kernel = createCPUQwen4ExpKernels();
    const auto policy = hashPolicy();
    const std::vector<std::int32_t> tokens{11, static_cast<std::int32_t>(policy.eosToken), 13, 14, 15, static_cast<std::int32_t>(policy.eosToken), 17};
    const std::int32_t prior[] = {7, 9};
    std::vector<std::int64_t> ids(tokens.size() * 16, -1);
    kernel->ngramHash({{static_cast<int>(tokens.size())}, tokens.data(), prior, ids.data(), policy}, nullptr);
    for (std::size_t row = 0; row < tokens.size(); ++row)
        for (int head = 0; head < 16; ++head)
        {
            std::uint64_t hash = 0;
            bool crossed = false;
            for (int position = 0; position < (head < 8 ? 2 : 3); ++position)
            {
                const auto source = static_cast<std::ptrdiff_t>(row) - position;
                const auto token = crossed ? policy.eosToken : (source < 0 ? prior[source + 2] : tokens[source]);
                hash ^= static_cast<std::uint64_t>(token) * policy.multipliers[position];
                // The current EOS never erases preceding tokens in its own hash.
                if (position > 0 && token == policy.eosToken) crossed = true;
            }
            EXPECT_EQ(ids[row * 16 + head], static_cast<std::int64_t>(hash % policy.moduli[head] + policy.offsets[head]));
        }
    // Golden constants were calculated with Python's arbitrary-precision uint64
    // mask, independently of both C++ backends and floating-point conversion.
    EXPECT_EQ(ids[0], 647);
    EXPECT_EQ(ids[8], 8340);
    for (int accepted = 0; accepted <= static_cast<int>(tokens.size()); ++accepted)
    {
        std::int32_t output[2]{};
        kernel->commitTokenHistory({{{static_cast<int>(tokens.size())}, &accepted}, prior, tokens.data(), output}, nullptr);
        for (int index = 0; index < 2; ++index)
        {
            const int source = accepted - 2 + index;
            EXPECT_EQ(output[index], source < 0 ? prior[source + 2] : tokens[source]);
        }
    }
    // Every split, including one/two-token chunks, uses exactly the same raw context.
    for (int split = 1; split < static_cast<int>(tokens.size()); ++split)
    {
        std::int32_t history[2];
        kernel->commitTokenHistory({{{static_cast<int>(tokens.size())}, nullptr, split}, prior, tokens.data(), history}, nullptr);
        std::vector<std::int64_t> suffix((tokens.size() - split) * 16);
        kernel->ngramHash({{static_cast<int>(tokens.size()) - split}, tokens.data() + split, history, suffix.data(), policy}, nullptr);
        EXPECT_TRUE(std::equal(suffix.begin(), suffix.end(), ids.begin() + split * 16));
    }
}

/** @test Dilation-three convolution uses oldest-to-newest taps, not adjacent positions. */
TEST(Qwen4ExpPrimitives, DilatedConvolutionIndependentTapOracle)
{
    llaminar::v2::ThreadCountGuard team(1);
    auto kernel = createCPUQwen4ExpKernels();
    constexpr int rows = 12, channels = 12;
    auto input = values(rows * channels), history = values(9 * channels, 7), weights = values(channels * 4, 3);
    auto residual = values(rows * channels, 13), gated = values(rows * channels, 29);
    std::vector<float> output(rows * channels);
    kernel->pleConv({{rows}, channels, input.data(), history.data(), weights.data(), gated.data(), residual.data(), output.data()}, nullptr);
    for (int row = 0; row < rows; ++row)
        for (int channel = 0; channel < channels; ++channel)
        {
            float sum = 0;
            for (int tap = 0; tap < 4; ++tap)
            {
                const int position = row - 9 + tap * 3;
                const float value = position < 0 ? history[(position + 9) * channels + channel] : input[position * channels + channel];
                sum = std::fma(value, weights[channel * 4 + tap], sum);
            }
            EXPECT_NEAR(output[row * channels + channel],
                residual[row * channels + channel] + (gated[row * channels + channel] + sum / (1 + std::exp(-sum))), 2e-6);
        }
}

/** @test All primitives keep row arithmetic invariant under thread count, shape and grouping. */
TEST(Qwen4ExpPrimitives, SerialGroupedBytesAndShrinkingReplay)
{
    auto kernel = createCPUQwen4ExpKernels();
    for (const int threads : {1, 3})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        for (const int hidden : {1, 7, 19, 257})
            for (const int rows : {1, 2, 3, 4, 7, 9, 15, 16, 31})
            {
                SCOPED_TRACE(::testing::Message() << "threads=" << threads << " H=" << hidden << " M=" << rows);
                const int channels = hidden * 4;
                auto normalized = values(rows * channels), key = values(rows * channels, 11), logits = values(rows * channels, 3);
                auto block = values(rows * hidden, 7), inject = values(rows * 4, 17), history = values(9 * channels, 19);
                auto weights = values(channels * 4, 23), residual = values(rows * channels, 29);
                std::vector<float> grouped(rows * channels, -99), serial(rows * channels, -99), mixed(rows * hidden), serialMixed(rows * hidden);
                kernel->pleGate({{rows}, hidden, key.data(), normalized.data(), block.data(), grouped.data()}, nullptr);
                for (int row = 0; row < rows; ++row)
                    kernel->pleGate({{1}, hidden, key.data() + row * channels, normalized.data() + row * channels,
                                    block.data() + row * hidden, serial.data() + row * channels}, nullptr);
                expectBytes(grouped, serial);
                kernel->residualRead({{rows}, hidden, normalized.data(), logits.data(), mixed.data()}, nullptr);
                for (int row = 0; row < rows; ++row)
                    kernel->residualRead({{1}, hidden, normalized.data() + row * channels, logits.data() + row * channels,
                                         serialMixed.data() + row * hidden}, nullptr);
                expectBytes(mixed, serialMixed);
                kernel->residualWrite({{rows}, hidden, residual.data(), block.data(), inject.data(), grouped.data()}, nullptr);
                for (int row = 0; row < rows; ++row)
                    kernel->residualWrite({{1}, hidden, residual.data() + row * channels, block.data() + row * hidden,
                                          inject.data() + row * 4, serial.data() + row * channels}, nullptr);
                expectBytes(grouped, serial);
                kernel->pleConv({{rows}, channels, normalized.data(), history.data(), weights.data(), key.data(), residual.data(), grouped.data()}, nullptr);
                auto serialHistory = history;
                std::vector<float> nextHistory(9 * channels);
                for (int row = 0; row < rows; ++row)
                {
                    kernel->pleConv({{1}, channels, normalized.data() + row * channels, serialHistory.data(), weights.data(),
                                     key.data() + row * channels, residual.data() + row * channels, serial.data() + row * channels}, nullptr);
                    kernel->commitConvHistory({{{1}, nullptr, 1}, channels, serialHistory.data(),
                                               normalized.data() + row * channels, nextHistory.data()}, nullptr);
                    serialHistory.swap(nextHistory);
                }
                expectBytes(grouped, serial);
                kernel->commitConvHistory({{{rows}, nullptr, rows}, channels, history.data(), normalized.data(), nextHistory.data()}, nullptr);
                expectBytes(serialHistory, nextHistory);
                for (int accepted = 0; accepted <= rows; ++accepted)
                {
                    kernel->commitConvHistory({{{rows}, &accepted}, channels, history.data(), normalized.data(), nextHistory.data()}, nullptr);
                    for (int slot = 0; slot < 9; ++slot)
                    {
                        const int source = accepted - 9 + slot;
                        const float *expected = source < 0 ? history.data() + (source + 9) * channels : normalized.data() + source * channels;
                        EXPECT_EQ(std::memcmp(expected, nextHistory.data() + slot * channels, channels * sizeof(float)), 0);
                    }
                }
                int live = std::min(rows, 2);
                std::fill(grouped.begin(), grouped.end(), -99);
                kernel->pleConv({{rows, &live}, channels, normalized.data(), history.data(), weights.data(), key.data(), residual.data(), grouped.data()}, nullptr);
                EXPECT_TRUE(std::all_of(grouped.begin() + live * channels, grouped.end(), [](float value) { return value == -99; }));
            }
    }
}

/** @test Invalid capacities, pointer aliases and dynamic decisions fail before publication. */
TEST(Qwen4ExpPrimitives, RejectsMalformedBindingsAndCounts)
{
    llaminar::v2::ThreadCountGuard team(1);
    auto kernel = createCPUQwen4ExpKernels();
    auto policy = hashPolicy();
    EXPECT_NO_THROW(policy.validate());
    policy.moduli[0] = 0;
    EXPECT_THROW(policy.validate(), std::invalid_argument);
    policy = hashPolicy();
    policy.offsets[1] = policy.offsets[0];
    EXPECT_THROW(policy.validate(), std::invalid_argument);
    policy = hashPolicy();
    policy.offsets[15] = std::numeric_limits<std::int64_t>::max();
    EXPECT_THROW(policy.validate(), std::invalid_argument);
    EXPECT_THROW(kernel->residualRead({}, nullptr), std::invalid_argument);
    std::vector<float> input(4), output(4), history(36), next(36);
    int live = 2;
    EXPECT_THROW(kernel->residualExpand({{1, &live}, 1, input.data(), output.data()}, nullptr), std::invalid_argument);
    int accepted = -1;
    EXPECT_THROW(kernel->commitConvHistory({{{1}, &accepted}, 4, history.data(), input.data(), next.data()}, nullptr), std::invalid_argument);
    accepted = 2;
    EXPECT_THROW(kernel->commitConvHistory({{{1}, &accepted}, 4, history.data(), input.data(), next.data()}, nullptr), std::invalid_argument);
    accepted = 0;
    EXPECT_THROW(kernel->commitConvHistory({{{1}, &accepted}, 4, history.data(), input.data(), history.data()}, nullptr), std::invalid_argument);
}
