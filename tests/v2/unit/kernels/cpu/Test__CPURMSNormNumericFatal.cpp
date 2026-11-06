/**
 * @file Test__CPURMSNormNumericFatal.cpp
 * @brief Device-free RMSNorm numeric-fatality, half-policy and identity-gamma regressions.
 *
 * Every CPU RMS column program exercised here must terminate through one
 * explicit release-safe abort sink when a nonfinite summary or an
 * unrepresentable inverse is already computed at a phase boundary: exceptions
 * cannot cross an OpenMP collective, so a failing row must not publish
 * NaN/Inf/repair bytes. Previously published rows are never rolled back, and
 * no blanket before-any-byte or all-leaves claim is made: Q8_1 integer-space
 * and legacy FP32→INT8 quantization remain outside this coverage. Death
 * assertions use EXPECT_EXIT with KilledBySignal(SIGABRT) and the exact single
 * fatal line (prefix + typed cause + complete detail + newline), and the child
 * distinguishes an escaped exception (exit 97) and std::terminate (exit 98)
 * from the real abort. Retained OpenMP bodies are defined in lambdas outside
 * the EXPECT_EXIT argument so no #pragma sits inside a macro argument. This
 * target is device-free and does not certify factories, model inference or
 * economy.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "utils/CPUFeatures.h"
#include "tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
using namespace primitives;

namespace
{
/** @brief Complete single-line numeric fatal diagnostic: prefix, typed cause, detail, newline.
 * @details POSIX ERE (gtest REG_EXTENDED) compatible: capturing groups only,
 * literal escaped brackets, real newline character encoded in the C++ string.
 */
constexpr const char *kLineSummary =
    "CPU RMS numeric contract violation \\[NonfiniteSummary\\]: "
    "(CPU (FP32 |BF16 |FP16 )?RMS|Native RMS|((Scalar|AVX2|AVX512) Q16 RMS|Native Q16 RMS)) "
    "observed a nonfinite (original-row squared summary|FP32-square summary)\n";
/** @brief Complete single-line nonfinite/representability inverse diagnostic (POSIX ERE). */
constexpr const char *kLineInverse =
    "CPU RMS numeric contract violation \\[NonfiniteInverse\\]: "
    "(CPU (FP32 |BF16 |FP16 )?RMS|Native RMS|((Scalar|AVX2|AVX512) Q16 RMS|Native Q16 RMS)) "
    "(FP32 inverse is not representable after the declared rounding|"
    "original-row inverse is not representable|inverse is not representable)\n";
/** @brief Complete single-line retained-policy diagnostic (POSIX ERE). */
constexpr const char *kLinePolicy =
    "CPU RMS unsupported policy in retained collective: (BF16|FP16) RMS has no declared T5 FP32 accumulation program\n";

/** @brief Encode one original FP16 word through the portable storage contract. */
uint16_t encodeFP16(float value) { return ::llaminar2::fp32_to_fp16(value); }
/** @brief Encode one original BF16 word through the portable storage contract. */
uint16_t encodeBF16(float value)
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

/** @brief Keep requested retained teams exact without leaking caller OpenMP policy. */
class TestTeamPolicy final
{
public:
    /** @brief Retain caller policy and disable dynamic shrinking for this fixture. */
    TestTeamPolicy() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) { omp_set_dynamic(0); }
    /** @brief Restore the original caller ICVs even on an assertion failure. */
    ~TestTeamPolicy() { omp_set_num_threads(threads_); omp_set_dynamic(dynamic_); }
private:
    int threads_, dynamic_;
};

/** @brief Restore the caller's death-test style after a focused death sequence. */
class DeathStyleGuard final
{
public:
    DeathStyleGuard() : previous_(GTEST_FLAG_GET(death_test_style)) { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
    ~DeathStyleGuard() { GTEST_FLAG_SET(death_test_style, previous_); }
private:
    std::string previous_;
};

/** @brief Require an admitted compiled/runtime native profile before native-path fixtures. */
void requireNativeProfile()
{
    const auto isa = activeISALevel();
    if (isa == ISALevel::Scalar || !isNativeCPUExecutionISAPair(compiledCPUISALevel(), isa))
        throw std::runtime_error("CPURMSNormNumericFatal native fixture requires an admitted native profile");
}
} // namespace

/** @test FP32 matrix nonfinite summaries terminate through the sink across native row families. */
TEST(CPURMSNormNumericFatal, FP32MatrixNonfiniteSummaryIsExplicitFatal)
{
    const DeathStyleGuard style;
    requireNativeProfile();
    std::array<float, 16> gamma;
    gamma.fill(1.0f);
    for (const bool scalar : {false, true})
        for (const std::size_t rows : {1u, 2u, 3u, 4u, 5u})
        {
            const RMSNormExecOptions options{.allow_parallel = false, .force_scalar = scalar};
            std::vector<float> input(rows * 16, 1.0f), output(rows * 16, -19.0f);
            input[(rows - 1) * 16] = std::numeric_limits<float>::infinity();
            SCOPED_TRACE(::testing::Message() << "scalar=" << scalar << " M=" << rows);
            EXPECT_EXIT({
                std::set_terminate([]() { std::_Exit(98); });
                try { rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), rows, 16, 1e-5f, options); }
                catch (...) { std::_Exit(97); }
            }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        }
    for (const bool scalar : {false, true})
    {
        const RMSNormExecOptions options{.allow_parallel = false, .force_scalar = scalar};
        std::vector<float> input(16, 1.0f), output(16, -19.0f);
        input[0] = std::numeric_limits<float>::quiet_NaN();
        SCOPED_TRACE(::testing::Message() << "scalar=" << scalar << " NaN");
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, options); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
    }
}

/** @test The same nonfinite FP32 matrix input is fatal inside actually retained teams. */
TEST(CPURMSNormNumericFatal, FP32MatrixNonfiniteRetainedTeamIsExplicitFatal)
{
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    requireNativeProfile();
    std::array<float, 16> input, gamma, output;
    input.fill(1.0f); gamma.fill(1.0f); output.fill(-19.0f);
    input[1] = std::numeric_limits<float>::infinity();
    const auto actualTeam = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(actualTeam(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
    // Serialized and inactive retained teams are still lexical collectives.
    const auto serialized = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(1)
            { if (omp_get_level() <= 0) std::_Exit(95); rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false}); }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(serialized(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
    const auto inactive = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel if(false)
            { if (omp_get_level() <= 0) std::_Exit(95); rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false}); }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(inactive(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
}

/** @test Serialized nested lexical collectives still reach the numeric sink. */
TEST(CPURMSNormNumericFatal, FP32MatrixNonfiniteNestedRetainedIsExplicitFatal)
{
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    requireNativeProfile();
    std::array<float, 16> input, gamma, output;
    input.fill(1.0f); gamma.fill(1.0f); output.fill(-19.0f);
    input[2] = std::numeric_limits<float>::infinity();
    const auto nested = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(2)
            {
                if (omp_get_num_threads() != 2) std::_Exit(94);
#pragma omp parallel num_threads(1)
                { if (omp_get_level() <= 1) std::_Exit(95); rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false}); }
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(nested(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
}

/** @test The FP32 per-row FP32-mean program treats an undefined zero denominator as fatal. */
TEST(CPURMSNormNumericFatal, FP32DirectRowZeroDenominatorIsExplicitFatal)
{
    const DeathStyleGuard style;
    std::array<float, 8> input, gamma, output;
    input.fill(0.0f); gamma.fill(1.0f); output.fill(-19.0f);
    EXPECT_EXIT({
        std::set_terminate([]() { std::_Exit(98); });
        try { rmsnorm_fused_row_scalar(input.data(), gamma.data(), output.data(), input.size(), 0.0f); }
        catch (...) { std::_Exit(97); }
    }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    // The positive-epsilon witness must remain legal and publish exact zeros.
    rmsnorm_fused_row_scalar(input.data(), gamma.data(), output.data(), input.size(), 1e-5f);
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
}

/** @test BF16/FP16 matrix nonfinite summaries terminate through the sink on the actual column program. */
TEST(CPURMSNormNumericFatal, HalfMatrixNonfiniteSummaryIsExplicitFatal)
{
    const DeathStyleGuard style;
    requireNativeProfile();
    for (const bool bf16 : {true, false})
    {
        std::vector<float> gamma(16, 1.0f);
        for (const bool scalar : {false, true})
            for (const std::size_t rows : {1u, 2u, 3u, 4u, 5u})
            {
                const RMSNormExecOptions options{.allow_parallel = false, .force_scalar = scalar};
                std::vector<std::uint16_t> input(rows * 16, 0x3c00), output(rows * 16, 0xa55a);
                input[(rows - 1) * 16] = bf16 ? encodeBF16(std::numeric_limits<float>::infinity())
                                              : encodeFP16(std::numeric_limits<float>::infinity());
                SCOPED_TRACE(::testing::Message() << (bf16 ? "bf16" : "fp16") << " scalar=" << scalar << " M=" << rows);
                EXPECT_EXIT({
                    std::set_terminate([]() { std::_Exit(98); });
                    try {
                        if (bf16) rmsnorm_fused_bf16_vectorized(input.data(), gamma.data(), output.data(), rows, 16, 1e-5f, options);
                        else rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), rows, 16, 1e-5f, options);
                    } catch (...) { std::_Exit(97); }
                }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
            }
        for (const bool scalar : {false, true})
        {
            const RMSNormExecOptions options{.allow_parallel = false, .force_scalar = scalar};
            std::vector<std::uint16_t> input(16, 0x3c00), output(16, 0xa55a);
            input[0] = bf16 ? encodeBF16(std::numeric_limits<float>::quiet_NaN())
                            : encodeFP16(std::numeric_limits<float>::quiet_NaN());
            SCOPED_TRACE(::testing::Message() << (bf16 ? "bf16" : "fp16") << " scalar=" << scalar << " NaN");
            EXPECT_EXIT({
                std::set_terminate([]() { std::_Exit(98); });
                try {
                    if (bf16) rmsnorm_fused_bf16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, options);
                    else rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, options);
                } catch (...) { std::_Exit(97); }
            }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        }
    }
}

/** @test Half matrix nonfinite input is fatal inside an actually retained 3-thread team. */
TEST(CPURMSNormNumericFatal, HalfMatrixNonfiniteRetainedTeamIsExplicitFatal)
{
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    requireNativeProfile();
    std::vector<std::uint16_t> input(64, 0x3c00), output(64, 0xa55a);
    std::vector<float> gamma(16, 1.0f);
    input[3 * 16] = encodeFP16(std::numeric_limits<float>::infinity());
    const auto child = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 4, 16, 1e-5f, {.allow_parallel = false});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(child(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
}

/** @test Half zero denominator is fatal while a positive epsilon keeps the legal zero row. */
TEST(CPURMSNormNumericFatal, HalfZeroDenominatorIsFatalAndPositiveEpsilonLegal)
{
    const DeathStyleGuard style;
    std::array<std::uint16_t, 8> input, output;
    std::array<float, 8> gamma;
    input.fill(0x0000); gamma.fill(1.0f); output.fill(0xa55a);
    EXPECT_EXIT({
        std::set_terminate([]() { std::_Exit(98); });
        try { rmsnorm_fused_row_fp16_scalar(input.data(), gamma.data(), output.data(), input.size(), 0.0f); }
        catch (...) { std::_Exit(97); }
    }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    EXPECT_EXIT({
        std::set_terminate([]() { std::_Exit(98); });
        try { rmsnorm_fused_row_bf16_scalar(input.data(), gamma.data(), output.data(), input.size(), 0.0f); }
        catch (...) { std::_Exit(97); }
    }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    rmsnorm_fused_row_fp16_scalar(input.data(), gamma.data(), output.data(), input.size(), 1e-5f);
    for (auto word : output) EXPECT_EQ(word, 0x0000u);
    rmsnorm_fused_row_bf16_scalar(input.data(), gamma.data(), output.data(), input.size(), 1e-5f);
    for (auto word : output) EXPECT_EQ(word, 0x0000u);
}

/** @test Half T5 is rejected cold: recoverable standalone, explicit fatal in retained collectives. */
TEST(CPURMSNormNumericFatal, HalfT5RejectedColdAndFatalInRetainedTeam)
{
    std::vector<std::uint16_t> input(16, 0x3c00), output(16, 0xa55a);
    std::vector<float> gamma(16, 1.0f);
    const RMSNormExecOptions t5{.allow_parallel = false, .t5_compat_mode = true};
    EXPECT_THROW(rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5),
                 std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_bf16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5),
                 std::invalid_argument);
    // FP32 retains its declared T5 program instead of being rejected.
    std::array<float, 16> f32_input, f32_output;
    f32_input.fill(0.5f); f32_output.fill(-19.0f);
    rmsnorm_fused_vectorized(f32_input.data(), gamma.data(), f32_output.data(), 1, 16, 1e-6f, {.allow_parallel = false, .t5_compat_mode = true});
    for (auto value : f32_output) EXPECT_TRUE(std::isfinite(value));
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    const auto fp16Three = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5);
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(fp16Three(), ::testing::KilledBySignal(SIGABRT), kLinePolicy);
    const auto bf16Three = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_bf16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5);
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(bf16Three(), ::testing::KilledBySignal(SIGABRT), kLinePolicy);
    // A serialized retained team (`num_threads(1)`) is still a retained
    // structured region: the cold admission must abort, never throw across it.
    const auto serialized = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(1)
            { if (omp_get_level() <= 0) std::_Exit(95); rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5); }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(serialized(), ::testing::KilledBySignal(SIGABRT), kLinePolicy);
    // The same applies to an inactive `if(false)` retained region.
    const auto inactive = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel if(false)
            { if (omp_get_level() <= 0) std::_Exit(95); rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5); }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(inactive(), ::testing::KilledBySignal(SIGABRT), kLinePolicy);
    // A nested retained team (outer active, inner serialized) is still lexical.
    const auto nested = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(2)
            {
                if (omp_get_num_threads() != 2) std::_Exit(94);
#pragma omp parallel num_threads(1)
                { if (omp_get_level() <= 1) std::_Exit(95); rmsnorm_fused_fp16_vectorized(input.data(), gamma.data(), output.data(), 1, 16, 1e-5f, t5); }
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(nested(), ::testing::KilledBySignal(SIGABRT), kLinePolicy);
}

/** @test Null gamma is an admitted first-class identity on half matrices, rows and retained teams. */
TEST(CPURMSNormNumericFatal, HalfNullGammaIdentityMatchesExplicitOnes)
{
    const TestTeamPolicy team;
    requireNativeProfile();
    for (const bool bf16 : {true, false})
        for (const std::size_t columns : {1u, 7u, 8u, 9u, 17u, 33u})
            for (const bool scalar : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "bf16=" << bf16 << " K=" << columns << " scalar=" << scalar);
                std::vector<std::uint16_t> input(columns), identity(columns), original(columns);
                std::vector<float> ones(columns, 1.0f);
                for (std::size_t c = 0; c < columns; ++c)
                {
                    const float value = float(int(c * 13 + 7) % 31 - 15) * 0.125f;
                    input[c] = bf16 ? encodeBF16(value) : encodeFP16(value);
                }
                const RMSNormExecOptions opts{.allow_parallel = false, .force_scalar = scalar};
                const auto invoke = [&](const float *gamma, std::vector<std::uint16_t> &output) {
                    if (bf16) rmsnorm_fused_bf16_vectorized(input.data(), gamma, output.data(), 1, columns, 1e-5f, opts);
                    else rmsnorm_fused_fp16_vectorized(input.data(), gamma, output.data(), 1, columns, 1e-5f, opts);
                };
                invoke(nullptr, identity);
                invoke(ones.data(), original);
                EXPECT_EQ(identity, original);
            }
    // Retained actual team: identity bytes must equal the serial TRUE-ones witness,
    // while a separately weighted gamma must discriminate from identity.
    constexpr std::size_t columns = 17;
    std::array<std::uint16_t, columns> input, identity, original, weighted_output;
    std::array<float, columns> ones, weighted;
    for (std::size_t c = 0; c < columns; ++c)
    {
        input[c] = encodeFP16(float(c + 1) * 0.25f);
        ones[c] = 1.0f;
        weighted[c] = 0.5f + float(c) * 0.0625f;
    }
    rmsnorm_fused_fp16_vectorized(input.data(), ones.data(), original.data(), 1, columns, 1e-5f, {.allow_parallel = false});
    rmsnorm_fused_fp16_vectorized(input.data(), weighted.data(), weighted_output.data(), 1, columns, 1e-5f, {.allow_parallel = false});
    ASSERT_NE(original, weighted_output) << "weighted gamma must discriminate from identity";
    int actualTeam = 0;
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) actualTeam = omp_get_num_threads();
        rmsnorm_fused_fp16_vectorized(input.data(), nullptr, identity.data(), 1, columns, 1e-5f, {.allow_parallel = false});
    }
    ASSERT_EQ(actualTeam, 3);
    EXPECT_EQ(identity, original);
}

/** @test Direct FP32/half row entries treat null gamma as identity without reading null. */
TEST(CPURMSNormNumericFatal, RowNullGammaIdentityMatchesExplicitOnes)
{
    std::array<float, 9> f32_input, f32_identity, f32_original, ones;
    for (std::size_t c = 0; c < f32_input.size(); ++c)
    {
        f32_input[c] = float(int(c * 11 + 3) % 37 - 18) * 0.5f;
        ones[c] = 1.0f;
    }
    rmsnorm_fused_row_scalar(f32_input.data(), ones.data(), f32_original.data(), f32_input.size(), 1e-5f);
    rmsnorm_fused_row_scalar(f32_input.data(), nullptr, f32_identity.data(), f32_input.size(), 1e-5f);
    EXPECT_EQ(f32_identity, f32_original);

    std::array<std::uint16_t, 9> half_input, half_identity, half_original;
    for (std::size_t c = 0; c < half_input.size(); ++c)
        half_input[c] = encodeFP16(f32_input[c]);
    rmsnorm_fused_row_fp16_scalar(half_input.data(), ones.data(), half_original.data(), half_input.size(), 1e-5f);
    rmsnorm_fused_row_fp16_scalar(half_input.data(), nullptr, half_identity.data(), half_input.size(), 1e-5f);
    EXPECT_EQ(half_identity, half_original);
    rmsnorm_fused_row_bf16_scalar(half_input.data(), ones.data(), half_original.data(), half_input.size(), 1e-5f);
    rmsnorm_fused_row_bf16_scalar(half_input.data(), nullptr, half_identity.data(), half_input.size(), 1e-5f);
    EXPECT_EQ(half_identity, half_original);
}

/** @test The guarded FP32 zero row is legal and FP64 large finite rows remain finite positives. */
TEST(CPURMSNormNumericFatal, GuardedZeroAndLargeFinitePositivesRemainLegal)
{
    std::array<float, 32> input, gamma, output;
    gamma.fill(1.0f);
    input.fill(0.0f);
    output.fill(-19.0f);
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false});
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
    output.fill(-19.0f);
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, {.allow_parallel = false});
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);

    // A finite FP64 square summary (1e25^2 per element) must stay positive.
    input.fill(1e25f);
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false});
    for (auto value : output) EXPECT_TRUE(std::isfinite(value) && value > 0.0f);
    for (auto value : output)
    {
        // Every column is the same legal positive; no NaN/Inf repair occurred.
        EXPECT_EQ(std::bit_cast<std::uint32_t>(value), std::bit_cast<std::uint32_t>(output[0]));
    }
}

/** @test T5 FP32 matrix changes accumulation only; a zero row with zero epsilon publishes legal guarded zeros. */
TEST(CPURMSNormNumericFatal, T5ZeroRowPublishesLegalGuardedZeros)
{
    const TestTeamPolicy team;
    requireNativeProfile();
    std::array<float, 16> input, gamma, output;
    input.fill(0.0f); gamma.fill(1.0f);
    const RMSNormExecOptions t5{.allow_parallel = false, .t5_compat_mode = true};
    for (const bool scalar : {false, true})
    {
        const RMSNormExecOptions options{.allow_parallel = false, .t5_compat_mode = true, .force_scalar = scalar};
        output.fill(-19.0f);
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, options);
        SCOPED_TRACE(::testing::Message() << "scalar=" << scalar);
        for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
    }
    // Retained actual team: the same guarded zeros, T5 accumulation unchanged.
    output.fill(-19.0f);
    int actualTeam = 0;
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) actualTeam = omp_get_num_threads();
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, t5);
    }
    ASSERT_EQ(actualTeam, 3);
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
}

/** @test T5 FP32 accumulation overflow on large finite input is a nonfinite-summary fatal. */
TEST(CPURMSNormNumericFatal, T5FiniteOverflowSummaryIsFatal)
{
    const DeathStyleGuard style;
    requireNativeProfile();
    std::array<float, 16> input, gamma, output;
    input.fill(1e30f); gamma.fill(1.0f); output.fill(-19.0f);
    for (const bool scalar : {false, true})
    {
        const RMSNormExecOptions options{.allow_parallel = false, .t5_compat_mode = true, .force_scalar = scalar};
        SCOPED_TRACE(::testing::Message() << "scalar=" << scalar);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, options); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
    }
}

/** @test A finite FP64 inverse that rounds to FP32 infinity is fatal on generic and native paths. */
TEST(CPURMSNormNumericFatal, TinyOnlyRowRoundedFP32InverseOverflowIsFatal)
{
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    std::array<float, 16> input, gamma, output;
    // Every value is a small FP32 subnormal: the FP64 squared mean is positive
    // and finite, but its FP32-rounded inverse exceeds FLT_MAX (1e39 > 3.4e38).
    input.fill(std::ldexp(1.0f, -130));
    gamma.fill(1.0f);
    output.fill(-19.0f);
    for (const bool scalar : {false, true})
    {
        const RMSNormExecOptions options{.allow_parallel = false, .force_scalar = scalar};
        SCOPED_TRACE(::testing::Message() << "scalar=" << scalar);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, options); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    }
    // Retained controls: actual native team and a retained forced-scalar team;
    // both must reach the fixed generic inverse check (served by software,
    // not by a native-only early exit).
    const auto retainedActual = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, {.allow_parallel = false});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(retainedActual(), ::testing::KilledBySignal(SIGABRT), kLineInverse);
    const auto retainedScalar = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, {.allow_parallel = false, .force_scalar = true});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(retainedScalar(), ::testing::KilledBySignal(SIGABRT), kLineInverse);
}

/** @test Actual compiled direct SIMD leaves gate full blocks, tails and zero-width no-ops. */
TEST(CPURMSNormNumericFatal, DirectSIMDLeavesGateRangesAndTails)
{
    const DeathStyleGuard style;
    requireNativeProfile();
    // Zero-width direct leaves are no-ops: no source/gamma/output is touched.
    rmsnorm_fused_row_scalar(nullptr, nullptr, nullptr, 0, 0.0f);
    rmsnorm_fused_row_fp16_scalar(nullptr, nullptr, nullptr, 0, 0.0f);
    rmsnorm_fused_row_bf16_scalar(nullptr, nullptr, nullptr, 0, 0.0f);
#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        // Genuine native FP32 direct leaf: four-lane fragments plus a tail.
        constexpr std::size_t columns = 9;
        std::array<float, columns> input, gamma, ones, output, identity;
        input.fill(1.0f); gamma.fill(1.0f); ones.fill(1.0f); output.fill(-19.0f);
        rmsnorm_fused_row_neon(input.data(), ones.data(), output.data(), columns, 1e-5f);
        rmsnorm_fused_row_neon(input.data(), nullptr, identity.data(), columns, 1e-5f);
        EXPECT_EQ(identity, output);
        input[columns - 1] = std::numeric_limits<float>::infinity();
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_neon(input.data(), gamma.data(), output.data(), columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
    }
#endif
#if defined(__AVX2__)
    rmsnorm_fused_row_avx2(nullptr, nullptr, nullptr, 0, 0.0f);
    {
        // FP32 AVX2 row: 64-element summary blocks, then 8-element blocks -> K=65.
        constexpr std::size_t columns = 65;
        std::vector<float> input(columns, 1.0f), gamma(columns, 1.0f), ones(columns, 1.0f), output(columns, -19.0f), identity(columns);
        rmsnorm_fused_row_avx2(input.data(), ones.data(), output.data(), columns, 1e-5f);
        rmsnorm_fused_row_avx2(input.data(), nullptr, identity.data(), columns, 1e-5f);
        EXPECT_EQ(identity, output);
        input[columns - 1] = std::numeric_limits<float>::infinity();
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_avx2(input.data(), gamma.data(), output.data(), columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        input.assign(columns, 0.0f);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_avx2(input.data(), gamma.data(), output.data(), columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    }
#endif
#if defined(__AVX512F__)
    rmsnorm_fused_row_avx512(nullptr, nullptr, nullptr, 0, 0.0f);
    rmsnorm_fused_row_bf16_avx512(nullptr, nullptr, nullptr, 0, 0.0f);
    rmsnorm_fused_row_fp16_avx512(nullptr, nullptr, nullptr, 0, 0.0f);
    fused_residual_rmsnorm_row_avx512(nullptr, nullptr, nullptr, nullptr, 0, 0.0f);
    {
        // FP32 AVX512 row: 32-element summary blocks -> K=33.
        constexpr std::size_t columns = 33;
        std::vector<float> input(columns, 1.0f), gamma(columns, 1.0f), ones(columns, 1.0f), output(columns, -19.0f), identity(columns);
        rmsnorm_fused_row_avx512(input.data(), ones.data(), output.data(), columns, 1e-5f);
        rmsnorm_fused_row_avx512(input.data(), nullptr, identity.data(), columns, 1e-5f);
        EXPECT_EQ(identity, output);
        input[columns - 1] = std::numeric_limits<float>::infinity();
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_avx512(input.data(), gamma.data(), output.data(), columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        input.assign(columns, 0.0f);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_avx512(input.data(), gamma.data(), output.data(), columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
        // BF16 AVX512 row: 32-element summary blocks -> K=33.
        constexpr std::size_t half_columns = 33;
        std::vector<std::uint16_t> half_input(half_columns), half_output(half_columns, 0xa55a), half_identity(half_columns);
        std::vector<float> half_ones(half_columns, 1.0f);
        for (std::size_t c = 0; c < half_columns; ++c)
            half_input[c] = encodeBF16(float(int(c * 7 + 3) % 29 - 14) * 0.25f);
        rmsnorm_fused_row_bf16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 1e-5f);
        rmsnorm_fused_row_bf16_avx512(half_input.data(), nullptr, half_identity.data(), half_columns, 1e-5f);
        EXPECT_EQ(half_identity, half_output);
        half_input[half_columns - 1] = encodeBF16(std::numeric_limits<float>::infinity());
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_bf16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        half_input.assign(half_columns, 0x0000);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_bf16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
        // FP16 AVX512 row: 32-element summary blocks -> K=33.
        for (std::size_t c = 0; c < half_columns; ++c)
            half_input[c] = encodeFP16(float(int(c * 7 + 3) % 29 - 14) * 0.25f);
        rmsnorm_fused_row_fp16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 1e-5f);
        rmsnorm_fused_row_fp16_avx512(half_input.data(), nullptr, half_identity.data(), half_columns, 1e-5f);
        EXPECT_EQ(half_identity, half_output);
        half_input[half_columns - 1] = encodeFP16(std::numeric_limits<float>::infinity());
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_fp16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        half_input.assign(half_columns, 0x0000);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_fp16_avx512(half_input.data(), half_ones.data(), half_output.data(), half_columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
        // Residual AVX512 row: nonempty 32-element pass 1 and undefined zero denominator.
        std::vector<float> residual(columns, 1.0f), fused_output(columns, -19.0f);
        fused_residual_rmsnorm_row_avx512(input.data(), residual.data(), ones.data(), fused_output.data(), columns, 1e-5f);
        for (auto value : fused_output) EXPECT_TRUE(std::isfinite(value));
        residual.assign(columns, 0.0f);
        input.assign(columns, 0.0f);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { fused_residual_rmsnorm_row_avx512(input.data(), residual.data(), ones.data(), fused_output.data(), columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    }
#endif
#if defined(__AVX2__) && defined(__F16C__)
    rmsnorm_fused_row_fp16_avx2(nullptr, nullptr, nullptr, 0, 0.0f);
    {
        // FP16 AVX2+F16C row: 16-element summary blocks -> K=17.
        constexpr std::size_t columns = 17;
        std::vector<std::uint16_t> input(columns), output(columns, 0xa55a), identity(columns);
        std::vector<float> ones(columns, 1.0f);
        for (std::size_t c = 0; c < columns; ++c)
            input[c] = encodeFP16(float(int(c * 11 + 5) % 37 - 18) * 0.5f);
        rmsnorm_fused_row_fp16_avx2(input.data(), ones.data(), output.data(), columns, 1e-5f);
        rmsnorm_fused_row_fp16_avx2(input.data(), nullptr, identity.data(), columns, 1e-5f);
        EXPECT_EQ(identity, output);
        input[columns - 1] = encodeFP16(std::numeric_limits<float>::infinity());
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_fp16_avx2(input.data(), ones.data(), output.data(), columns, 1e-5f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
        input.assign(columns, 0x0000);
        EXPECT_EXIT({
            std::set_terminate([]() { std::_Exit(98); });
            try { rmsnorm_fused_row_fp16_avx2(input.data(), ones.data(), output.data(), columns, 0.0f); }
            catch (...) { std::_Exit(97); }
        }, ::testing::KilledBySignal(SIGABRT), kLineInverse);
    }
#endif
}

/** @test The guarded FP32 zero publication yields zero output with an eps=0 guard even with nonfinite gamma. */
TEST(CPURMSNormNumericFatal, GuardedZeroRowOutputRemainsZeroWithNonfiniteGamma)
{
    std::array<float, 16> input, gamma, output;
    input.fill(0.0f);
    gamma.fill(std::numeric_limits<float>::quiet_NaN());
    output.fill(-19.0f);
    // Epsilon 0 is the guarded inverse-zero boundary; with a positive inverse
    // the declared (0*inverse)*gamma arithmetic would legitimately produce NaN,
    // so the nonfinite-gamma fixture is constrained to the guard.
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 0.0f, {.allow_parallel = false});
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
}

/** @test A legal positive-epsilon zero row keeps a finite positive inverse with valid gamma. */
TEST(CPURMSNormNumericFatal, ZeroRowWithPositiveEpsilonAndValidGammaStaysLegal)
{
    std::array<float, 16> input, gamma, output;
    input.fill(0.0f);
    gamma.fill(1.0f);
    output.fill(-19.0f);
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, input.size(), 1e-5f, {.allow_parallel = false});
    for (auto value : output) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0x00000000u);
}

/** @test Q16 nonfinite scale is fatal through the shared sink, standalone and retained. */
TEST(CPURMSNormNumericFatal, Q16NonfiniteScaleIsExplicitFatal)
{
    const TestTeamPolicy team;
    requireNativeProfile();
    std::vector<Q16_1Block> input(1, Q16_1Block{});
    std::array<float, 32> gamma, output;
    gamma.fill(1.0f); output.fill(-41.0f);
    input[0].d = 1.0f;
    input[0].qs[0] = 1;
    const DeathStyleGuard style;
    input[0].d = std::numeric_limits<float>::infinity();
    EXPECT_EXIT({
        std::set_terminate([]() { std::_Exit(98); });
        try { rmsnorm_q16_1_fp32_fused(input.data(), gamma.data(), output.data(), 1, 32, 1e-5f, {.allow_parallel = false}); }
        catch (...) { std::_Exit(97); }
    }, ::testing::KilledBySignal(SIGABRT), kLineSummary);
    input[0].d = std::numeric_limits<float>::quiet_NaN();
    const auto retained = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_q16_1_fp32_fused(input.data(), gamma.data(), output.data(), 1, 32, 1e-5f, {.allow_parallel = false});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(retained(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
}

/** @test A corrupt M=12 matrix in a retained 3-team yields one intact typed fatal diagnostic. */
TEST(CPURMSNormNumericFatal, MultiOwnerMatrixEmitsIntactTypedFatal)
{
    const DeathStyleGuard style;
    const TestTeamPolicy team;
    requireNativeProfile();
    constexpr std::size_t rows = 12, columns = 16;
    std::vector<float> input(rows * columns, 1.0f), gamma(columns, 1.0f), output(rows * columns, -19.0f);
    // One corrupt row inside each of the three native four-row tiles; at least
    // one worker must emit one complete typed fatal line via the single fprintf
    // before any abort. No claim that every tile reached the sink is made.
    input[1 * columns] = std::numeric_limits<float>::infinity();
    input[5 * columns] = std::numeric_limits<float>::infinity();
    input[9 * columns] = std::numeric_limits<float>::infinity();
    const auto child = [&]() {
        std::set_terminate([]() { std::_Exit(98); });
        try {
#pragma omp parallel num_threads(3)
            {
                if (omp_get_num_threads() != 3) std::_Exit(96);
                rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), rows, columns, 1e-5f, {.allow_parallel = false});
            }
        } catch (...) { std::_Exit(97); }
    };
    EXPECT_EXIT(child(), ::testing::KilledBySignal(SIGABRT), kLineSummary);
}

/** @test Cold invalid epsilon admission stays a recoverable standalone invalid_argument. */
TEST(CPURMSNormNumericFatal, ColdEpsilonPolicyRemainsStandaloneRecoverable)
{
    std::array<float, 8> input, gamma, output;
    input.fill(1.0f); gamma.fill(1.0f); output.fill(-19.0f);
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, 8, -1.0f), std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), 1, 8, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(rmsnorm_fused_row_scalar(input.data(), gamma.data(), output.data(), 8, -1.0f), std::invalid_argument);
    std::array<std::uint16_t, 8> half, half_output;
    half.fill(0x3c00); half_output.fill(0xa55a);
    EXPECT_THROW(rmsnorm_fused_fp16_vectorized(half.data(), gamma.data(), half_output.data(), 1, 8, std::numeric_limits<float>::infinity()), std::invalid_argument);
    EXPECT_EQ(half_output, (std::array<std::uint16_t, 8>{0xa55a, 0xa55a, 0xa55a, 0xa55a, 0xa55a, 0xa55a, 0xa55a, 0xa55a}));
    EXPECT_EQ(output, (std::array<float, 8>{-19.0f, -19.0f, -19.0f, -19.0f, -19.0f, -19.0f, -19.0f, -19.0f}));
}
} // namespace llaminar2::test
