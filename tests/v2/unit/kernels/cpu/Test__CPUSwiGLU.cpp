/**
 * @file Test__CPUSwiGLU.cpp
 * @brief Production SwiGLU geometry, native arithmetic, storage and workshare proofs.
 *
 * The test links SwiGLUPrimitives.cpp rather than a test-only implementation.
 * ARM ordinary words use an independent memory-rounded polynomial reference;
 * expert words use the backend-neutral device contract. Native tails, original
 * serial rows, typed aliases and retained teams must publish the same bytes.
 * These bounded fixtures certify no model graph, serving path or performance.
 */
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "utils/CPUFeatures.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <chrono>
#include <vector>
#if LLAMINAR_COMPILED_WITH_NEON
#include "kernels/common/DeviceSwiGLUNumericalContract.h"
#include "kernels/cpu/simd/NEONDeviceSwiGLU.h"
#include "tensors/NEONFP16.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUExplicitRoundingOracle.h"
#endif

namespace llaminar2::test
{
namespace
{
/** @brief Restore the calling test's real OpenMP policy after bounded team tests. */
class SwiGLUBudget final
{
public:
    /** @brief Start with one real worker and disable dynamic team shrinking. */
    SwiGLUBudget() : previous_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore the caller's worker and dynamic policies. */
    ~SwiGLUBudget() { omp_set_num_threads(previous_); omp_set_dynamic(dynamic_); }
private:
    int previous_, dynamic_;
};

/** @test Every incomplete Q8 extent rejects before even reading a source pointer. */
TEST(CPUSwiGLU, CompleteBlockGeometryRejectsBeforeMutation)
{
    SwiGLUBudget budget;
    std::array<Q8_1Block, 4> output;
    std::memset(output.data(), 0x65, sizeof(output));
    const auto original = output;
    const auto *unreadable = reinterpret_cast<const void *>(std::uintptr_t{1});
    for (int count = -3; count < 96; ++count)
    {
        if (count >= 0 && count % 32 == 0) continue;
        EXPECT_THROW(primitives::compute_swiglu_q8_1(unreadable, unreadable, output.data(), count), std::invalid_argument);
        EXPECT_EQ(std::memcmp(output.data(), original.data(), sizeof(output)), 0);
    }
    EXPECT_NO_THROW(primitives::compute_swiglu_q8_1(nullptr, nullptr, nullptr, 0));
    EXPECT_THROW(primitives::compute_swiglu(nullptr, nullptr, nullptr, 1), std::invalid_argument);
    EXPECT_THROW(primitives::compute_swiglu_serial(nullptr, nullptr, nullptr, -1), std::invalid_argument);
    EXPECT_THROW(primitives::compute_swiglu_fp16(nullptr, nullptr, nullptr, 1), std::invalid_argument);
    EXPECT_THROW(primitives::compute_swiglu_bf16(nullptr, nullptr, nullptr, -1), std::invalid_argument);
}

/** @test Partial source/output aliases reject before any native or scalar implementation writes. */
TEST(CPUSwiGLU, PartialStorageAliasesRejectBeforeMutation)
{
    SwiGLUBudget budget;
    std::array<float, 128> floating{}, multiplier{};
    std::memset(floating.data(), 0x63, sizeof(floating));
    const auto original = floating;
    for (auto operation : {primitives::compute_swiglu, primitives::compute_swiglu_serial,
                           primitives::compute_swiglu_gpu_aligned_expert, primitives::compute_swiglu_gpu_aligned_expert_serial})
    {
        EXPECT_THROW(operation(floating.data() + 8, multiplier.data(), floating.data() + 9, 17), std::invalid_argument);
        EXPECT_THROW(operation(floating.data() + 9, multiplier.data(), floating.data() + 8, 17), std::invalid_argument);
        EXPECT_THROW(operation(multiplier.data(), floating.data() + 8, floating.data() + 9, 17), std::invalid_argument);
        EXPECT_EQ(std::memcmp(floating.data(), original.data(), sizeof(floating)), 0);
    }
    std::array<std::uint16_t, 128> words{}, other_words{};
    std::memset(words.data(), 0x65, sizeof(words));
    const auto original_words = words;
    for (auto operation : {primitives::compute_swiglu_fp16, primitives::compute_swiglu_bf16})
    {
        EXPECT_THROW(operation(words.data() + 8, other_words.data(), words.data() + 9, 17), std::invalid_argument);
        EXPECT_THROW(operation(words.data() + 9, other_words.data(), words.data() + 8, 17), std::invalid_argument);
        EXPECT_EQ(std::memcmp(words.data(), original_words.data(), sizeof(words)), 0);
    }
    std::array<Q8_1Block, 8> blocks{}, other_blocks{};
    std::memset(blocks.data(), 0x68, sizeof(blocks));
    const auto original_blocks = blocks;
    EXPECT_THROW(primitives::compute_swiglu_q8_1(blocks.data(), other_blocks.data(), blocks.data() + 1, 96), std::invalid_argument);
    EXPECT_THROW(primitives::compute_swiglu_q8_1(other_blocks.data(), blocks.data() + 1, blocks.data(), 96), std::invalid_argument);
    EXPECT_EQ(std::memcmp(blocks.data(), original_blocks.data(), sizeof(blocks)), 0);
}

/** @test Every ISA rejects malformed fused-publication policy/storage before touching either bank. */
TEST(CPUSwiGLU, FusedPublicationRejectsPolicyAndSourceAliasesBeforeMutation)
{
    using cpu::native_vnni::swiglu_quantize_activations_to_q8_1;
    SwiGLUBudget budget;
    std::vector<float> gate(64, 0.5f), up(64, 0.75f);
    const auto original_gate = gate, original_up = up;
    std::vector<Q8_1Block> output(2);
    std::memset(output.data(), 0x65, output.size() * sizeof(Q8_1Block));
    const auto original_output = output;
    EXPECT_THROW(swiglu_quantize_activations_to_q8_1(gate.data(), up.data(), output.data(), 2, 32, 1,
        static_cast<CPUProjectionNumericalPolicy>(255)), std::invalid_argument);
    EXPECT_THROW(swiglu_quantize_activations_to_q8_1(gate.data(), up.data(), reinterpret_cast<Q8_1Block *>(gate.data() + 1), 2, 32, 1), std::invalid_argument);
    EXPECT_THROW(swiglu_quantize_activations_to_q8_1(gate.data(), up.data(), reinterpret_cast<Q8_1Block *>(up.data() + 1), 2, 32, 1), std::invalid_argument);
    EXPECT_EQ(std::memcmp(gate.data(), original_gate.data(), gate.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(up.data(), original_up.data(), up.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(output.data(), original_output.data(), output.size() * sizeof(Q8_1Block)), 0);
}

/** @test Empty public workshares release the team only after its delayed last observer arrives. */
TEST(CPUSwiGLU, EmptyRetainedWorksharesCompleteBeforeReturn)
{
    SwiGLUBudget budget;
    enum class Operation { Ordinary, Expert, BFloat16, Half, Q8 };
    for (auto operation : {Operation::Ordinary, Operation::Expert, Operation::BFloat16, Operation::Half, Operation::Q8})
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            std::atomic<int> entered{0};
            int observed_at_first_return = 0;
            const auto invoke = [&] {
                switch (operation)
                {
                case Operation::Ordinary: primitives::compute_swiglu(nullptr, nullptr, nullptr, 0); break;
                case Operation::Expert: primitives::compute_swiglu_gpu_aligned_expert(nullptr, nullptr, nullptr, 0); break;
                case Operation::BFloat16: primitives::compute_swiglu_bf16(nullptr, nullptr, nullptr, 0); break;
                case Operation::Half: primitives::compute_swiglu_fp16(nullptr, nullptr, nullptr, 0); break;
                case Operation::Q8: primitives::compute_swiglu_q8_1(nullptr, nullptr, nullptr, 0); break;
                }
            };
#pragma omp parallel num_threads(3)
            {
                if (omp_get_thread_num() == 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                entered.fetch_add(1, std::memory_order_relaxed);
                invoke();
                // Sample before single's own completion barrier. A no-op early
                // return exposes 1/2 arrivals here, even though the enclosing
                // parallel-region barrier would later hide the missing edge.
#pragma omp single
                observed_at_first_return = entered.load(std::memory_order_relaxed);
            }
            ASSERT_EQ(observed_at_first_return, 3) << static_cast<int>(operation) << ", " << iteration;
        }
}

/** @test Every ISA publishes each Q8 block once in retained teams, including exact aliases and empty banks. */
TEST(CPUSwiGLU, Q8RetainedTeamsPublishOnceAndCompleteEmpty)
{
    SwiGLUBudget budget;
    for (int blocks : {0, 1, 2, 3, 5})
    {
        std::vector<Q8_1Block> gate(blocks + 2), up(blocks + 2);
        for (int block = 0; block < blocks + 2; ++block)
        {
            gate[block].d = fp32_to_fp16(0.25f); up[block].d = fp32_to_fp16(0.125f);
            for (int i = 0; i < 32; ++i)
            {
                gate[block].qs[i] = static_cast<std::int8_t>((block * 17 + i * 73) % 256 - 128);
                up[block].qs[i] = static_cast<std::int8_t>((block * 13 + i * 37) % 256 - 128);
                gate[block].sum_qs += gate[block].qs[i]; up[block].sum_qs += up[block].qs[i];
            }
        }
        auto reference = gate;
        omp_set_num_threads(1);
        for (int block = 0; block < blocks; ++block)
            primitives::compute_swiglu_q8_1(gate.data() + block, up.data() + block, reference.data() + block, 32);
        if (blocks) ASSERT_NE(std::memcmp(reference.data(), gate.data(), blocks * sizeof(Q8_1Block)), 0);
        for (int workers : {1, 3, 10})
            for (bool alias_up : {false, true})
            {
                auto actual = alias_up ? up : gate;
                auto expected = actual;
                std::memcpy(expected.data(), reference.data(), blocks * sizeof(Q8_1Block));
                const auto *g = alias_up ? gate.data() : actual.data();
                const auto *u = alias_up ? actual.data() : up.data();
                int observed_workers = 0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed_workers = omp_get_num_threads();
                    primitives::compute_swiglu_q8_1(g, u, actual.data(), 0);
                    // Make duplicated whole-bank execution observable rather
                    // than relying on coincident vector loads hiding the race.
                    if (omp_get_thread_num() != 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    primitives::compute_swiglu_q8_1(g, u, actual.data(), blocks * 32);
                    primitives::compute_swiglu_q8_1(g, u, actual.data(), 0);
                }
                ASSERT_EQ(observed_workers, workers);
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Q8_1Block)), 0);
            }
    }
}

#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Fail admission if the compilation envelope is unavailable on this host. */
class NativeSwiGLU : public ::testing::Test
{
protected:
    /** @brief Use the real runtime/compiler ISA pair, never a spoofed processor. */
    static void SetUpTestSuite()
    { ASSERT_TRUE(activeISALevel() == ISALevel::NEON || activeISALevel() == ISALevel::NEONDotProd); }
};

/** @brief Test-only source of deterministic finite inputs, not a dispatch/timing corpus.
 * @param state Explicit caller-owned generator state.
 * @return Next complete unsigned word.
 */
std::uint32_t next(std::uint32_t &state)
{ state = state * 1664525u + 1013904223u; return state; }

/** @brief Independently evaluate the declared native ordinary polynomial through memory rounding.
 * @param gate Finite gate value.
 * @param up Finite multiplier value.
 * @return Original-order ordinary native result, without calling a native helper.
 */
float ordinaryReference(float gate, float up)
{
    const bool positive = gate >= 0.0f;
    const float argument = positive ? -gate : gate;
    const float exponential = [&] {
        if (argument >= 0.0f) return 1.0f;
        if (argument <= -87.0f) return 0.0f;
        const float scaled = memoryRoundedMultiply(argument, 0x1.715476p+0f);
        const int exponent = static_cast<int>(memoryRoundedAdd(scaled, scaled >= 0 ? 0.5f : -0.5f));
        float remainder = std::fma(-static_cast<float>(exponent), 0x1.62e400p-1f, argument);
        remainder = std::fma(-static_cast<float>(exponent), 0x1.7f7d1cp-20f, remainder);
        float polynomial = std::fma(0x1.a01a02p-16f, remainder, 0x1.a01a02p-13f);
        for (float coefficient : {0x1.6c16c2p-10f, 0x1.111112p-7f, 0x1.555556p-5f,
                                  0x1.555556p-3f, 0x1.000000p-1f, 1.0f, 1.0f})
            polynomial = std::fma(polynomial, remainder, coefficient);
        return memoryRoundedMultiply(polynomial, std::scalbn(1.0f, exponent));
    }();
    const float denominator = memoryRoundedAdd(1.0f, exponential);
    volatile float reciprocal = 1.0f / denominator;
    const float sigmoid = positive ? static_cast<float>(reciprocal) : memoryRoundedMultiply(exponential, reciprocal);
    return memoryRoundedMultiply(memoryRoundedMultiply(gate, sigmoid), up);
}

/** @brief Evaluate the expert program with independent observable-memory edges.
 * @param gate Original finite gate projection.
 * @param up Original finite up projection.
 * @return Separately rounded expert value, without the production register helper.
 */
float expertReference(float gate, float up)
{
    const bool positive = gate >= 0.0f;
    const float argument = positive ? -gate : gate;
    const float exponential = [&] {
        if (argument >= 0.0f) return 1.0f;
        if (argument <= -87.0f) return 0.0f;
        const float scaled = memoryRoundedMultiply(argument, 0x1.715476p+0f);
        const int exponent = static_cast<int>(memoryRoundedAdd(scaled, scaled >= 0 ? 0.5f : -0.5f));
        float remainder = std::fma(-static_cast<float>(exponent), 0x1.62e400p-1f, argument);
        remainder = std::fma(-static_cast<float>(exponent), 0x1.7f7d1cp-20f, remainder);
        float polynomial = std::fma(0x1.a01a02p-16f, remainder, 0x1.a01a02p-13f);
        for (float coefficient : {0x1.6c16c2p-10f, 0x1.111112p-7f, 0x1.555556p-5f,
                                  0x1.555556p-3f, 0x1.000000p-1f, 1.0f, 1.0f})
            polynomial = std::fma(polynomial, remainder, coefficient);
        return memoryRoundedMultiply(polynomial, std::scalbn(1.0f, exponent));
    }();
    const float denominator = memoryRoundedAdd(1.0f, exponential);
    float reciprocal = 0.75f;
    for (int iteration = 0; iteration < 5; ++iteration)
    {
        const float product = memoryRoundedMultiply(denominator, reciprocal);
        reciprocal = memoryRoundedMultiply(reciprocal, memoryRoundedAdd(2.0f, -product));
    }
    const float sigmoid = positive ? reciprocal : memoryRoundedMultiply(exponential, reciprocal);
    return memoryRoundedMultiply(memoryRoundedMultiply(gate, sigmoid), up);
}

/** @brief Require complete physical byte equality, including untouched trailing canaries.
 * @tparam T Trivial published element type.
 * @param actual Actual output storage.
 * @param expected Independent reference storage.
 */
template <typename T> void sameBytes(const std::vector<T> &actual, const std::vector<T> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0);
}

/** @test Native narrowing retains every half word and all FP32 exponent/rounding domains. */
TEST_F(NativeSwiGLU, PortableHalfNarrowingAllExponentAndMantissaDomains)
{
    const auto check = [](const std::array<std::uint32_t, 4> &words)
    {
        std::uint16_t actual[4];
        vst1_u16(actual, neon_fp16::narrow(vreinterpretq_f32_u32(vld1q_u32(words.data()))));
        for (int lane = 0; lane < 4; ++lane)
            ASSERT_EQ(actual[lane], fp32_to_fp16(std::bit_cast<float>(words[lane]))) << std::hex << words[lane];
    };
    for (std::uint32_t half = 0; half < 65536; half += 4)
    {
        std::array<std::uint32_t, 4> words;
        for (int lane = 0; lane < 4; ++lane) words[lane] = std::bit_cast<std::uint32_t>(fp16_to_fp32(static_cast<std::uint16_t>(half + lane)));
        check(words);
    }
    constexpr std::uint32_t mantissas[] = {0, 1, 0x0ffe, 0x0fff, 0x1000, 0x1001, 0x1fff,
        0x2000, 0x2fff, 0x3000, 0x3001, 0x3fff, 0x4000, 0x3fffff, 0x400000, 0x7fdfff,
        0x7fe000, 0x7fefff, 0x7ff000, 0x7ff001, 0x7fffff};
    for (std::uint32_t exponent = 0; exponent < 256; ++exponent)
        for (auto mantissa : mantissas)
            check({(exponent << 23) | mantissa, 0x80000000u | (exponent << 23) | mantissa,
                   (exponent << 23) | (mantissa ^ 0x2000u), 0x80000000u | (exponent << 23) | (mantissa ^ 0x2000u)});
    std::uint32_t state = 0x14b70021u;
    for (int sample = 0; sample < 32768; ++sample) check({next(state), next(state), next(state), next(state)});
}

/** @test Native expert lanes retain the shared scalar contract at range, sign and rounding boundaries. */
TEST_F(NativeSwiGLU, DeviceContractWordsAndExceptionalDomains)
{
    constexpr float edges[] = {-1000.0f, -87.00001f, -87.0f, -86.99999f, -64.0f, -0.00001f,
        -std::numeric_limits<float>::denorm_min(), -0.0f, 0.0f, std::numeric_limits<float>::denorm_min(),
        0.00001f, 1.0f, 32.0f, 87.0f, 1000.0f};
    std::uint32_t state = 0x8721cc31u;
    for (int sample = 0; sample < 8192; ++sample)
    {
        float gate[4], up[4], actual[4], exponential[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            gate[lane] = sample < static_cast<int>(std::size(edges)) ? edges[sample] :
                (static_cast<int>(next(state) % 65536) - 32768) * (1.0f / 256.0f);
            up[lane] = (static_cast<int>(next(state) % 65536) - 32768) * (1.0f / 128.0f);
        }
        vst1q_f32(actual, neon::device_swiglu(vld1q_f32(gate), vld1q_f32(up)));
        vst1q_f32(exponential, neon::device_exp_non_positive(vld1q_f32(gate)));
        for (int lane = 0; lane < 4; ++lane)
        {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[lane]), std::bit_cast<std::uint32_t>(device_swiglu_contract::swigluValue(gate[lane], up[lane])));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(exponential[lane]), std::bit_cast<std::uint32_t>(device_swiglu_contract::expNonPositive(gate[lane])));
        }
    }
    const float exceptional[4] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(), -0.0f};
    float result[4];
    vst1q_f32(result, neon::device_exp_non_positive(vld1q_f32(exceptional)));
    EXPECT_TRUE(std::isnan(result[0])); EXPECT_EQ(result[1], 1.0f); EXPECT_EQ(result[2], 0.0f); EXPECT_EQ(result[3], 1.0f);
}

/** @test Inlining cannot fuse a Newton product/subtraction or move an exponent rounding edge. */
TEST_F(NativeSwiGLU, DeviceNewtonRoundingSurvivesInlining)
{
    SwiGLUBudget budget;
    const float first_gate = std::bit_cast<float>(0x41550000u);
    const float first_up = std::bit_cast<float>(0xc37f2200u);
    ASSERT_EQ(std::bit_cast<std::uint32_t>(expertReference(first_gate, first_up)), 0xc5544734u);

    const auto check = [](const std::array<float, 4> &gate, const std::array<float, 4> &up)
    {
        std::array<float, 4> actual{}, direct{};
        vst1q_f32(direct.data(), neon::device_swiglu(vld1q_f32(gate.data()), vld1q_f32(up.data())));
        for (int count = 1; count <= 4; ++count)
        {
            for (auto operation : {primitives::compute_swiglu_gpu_aligned_expert,
                                   primitives::compute_swiglu_gpu_aligned_expert_serial})
            {
                operation(gate.data(), up.data(), actual.data(), count);
                for (int lane = 0; lane < count; ++lane)
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[lane]),
                              std::bit_cast<std::uint32_t>(expertReference(gate[lane], up[lane])))
                        << "count=" << count << " lane=" << lane << " gate=" << gate[lane];
            }
        }
        for (int lane = 0; lane < 4; ++lane)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(direct[lane]),
                      std::bit_cast<std::uint32_t>(expertReference(gate[lane], up[lane])));
    };
    check({first_gate, -first_gate, first_gate, -first_gate}, {first_up, first_up, -first_up, -first_up});
    for (int exponent = -125; exponent < 0; ++exponent)
    {
        const float boundary = (float(exponent) + 0.5f) / 0x1.715476p+0f;
        const float below = std::nextafter(boundary, -std::numeric_limits<float>::infinity());
        const float above = std::nextafter(boundary, std::numeric_limits<float>::infinity());
        check({boundary, below, above, -boundary}, {first_up, -first_up, 1.0f, -1.0f});
    }
}

/** @test The native ordinary polynomial also satisfies an independent mathematical accuracy envelope. */
TEST_F(NativeSwiGLU, OrdinaryAccuracyAgainstDoubleExponential)
{
    SwiGLUBudget budget;
    for (int sample = -4096; sample <= 4096; ++sample)
    {
        const float gate = sample * (80.0f / 4096.0f);
        const float up = (sample % 173 - 86) * 0.5f;
        float actual = 0;
        primitives::compute_swiglu_serial(&gate, &up, &actual, 1);
        const double reference = static_cast<double>(gate) / (1.0 + std::exp(-static_cast<double>(gate))) * up;
        ASSERT_LE(std::abs(static_cast<double>(actual) - reference), 2e-6 * std::abs(reference) + 2e-37)
            << gate << ", " << up;
    }
}

/** @test Real public FP32 APIs retain the same words at every vector, row and task fringe. */
TEST_F(NativeSwiGLU, OriginalRowsTailsAliasesAndNativeEntryPoints)
{
    SwiGLUBudget budget;
    for (int width : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025})
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 31, 33})
        {
            const int count = width * rows;
            std::vector<float> gate(count + 4, 919.0f), up(count + 4, -811.0f), expected(count + 4, 733.0f), actual = expected;
            for (int i = 0; i < count; ++i)
            {
                gate[i] = (i % 267 - 133) * 0.125f;
                up[i] = (i % 113 - 56) * 0.25f;
                expected[i] = ordinaryReference(gate[i], up[i]);
            }
            primitives::compute_swiglu_serial(gate.data(), up.data(), actual.data(), count);
            sameBytes(actual, expected);
            std::fill(actual.begin(), actual.end(), 733.0f);
            for (int row = 0; row < rows; ++row)
                primitives::compute_swiglu_serial(gate.data() + row * width, up.data() + row * width, actual.data() + row * width, width);
            sameBytes(actual, expected);
            omp_set_num_threads(3);
            primitives::compute_swiglu(gate.data(), up.data(), actual.data(), count);
            sameBytes(actual, expected);
            auto alias = gate;
            primitives::compute_swiglu(alias.data(), up.data(), alias.data(), count);
            for (int i = count; i < count + 4; ++i) expected[i] = gate[i];
            sameBytes(alias, expected);
            alias = up;
            primitives::compute_swiglu(gate.data(), alias.data(), alias.data(), count);
            for (int i = count; i < count + 4; ++i) expected[i] = up[i];
            sameBytes(alias, expected);
            for (int i = 0; i < count; ++i) expected[i] = device_swiglu_contract::swigluValue(gate[i], up[i]);
            for (int i = count; i < count + 4; ++i) expected[i] = 733.0f;
            std::fill(actual.begin(), actual.end(), 733.0f);
            primitives::compute_swiglu_gpu_aligned_expert(gate.data(), up.data(), actual.data(), count);
            sameBytes(actual, expected);
            for (int row = 0; row < rows; ++row)
                primitives::compute_swiglu_gpu_aligned_expert_serial(gate.data() + row * width, up.data() + row * width, actual.data() + row * width, width);
            sameBytes(actual, expected);
            alias = gate;
            primitives::compute_swiglu_gpu_aligned_expert(alias.data(), up.data(), alias.data(), count);
            for (int i = count; i < count + 4; ++i) expected[i] = gate[i];
            sameBytes(alias, expected);
            alias = up;
            primitives::compute_swiglu_gpu_aligned_expert(gate.data(), alias.data(), alias.data(), count);
            for (int i = count; i < count + 4; ++i) expected[i] = up[i];
            sameBytes(alias, expected);
        }
}

/** @test Every typed nonlinear route performs native arithmetic with portable storage and exact aliases. */
TEST_F(NativeSwiGLU, TypedStorageAndQ8ScaleThresholdPrograms)
{
    SwiGLUBudget budget;
    omp_set_num_threads(3);
    for (bool half : {false, true})
        for (int count : {1, 2, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1025})
        {
            std::vector<std::uint16_t> gate(count + 4, 0x6331), up(count + 4, 0x6331), expected(count + 4, 0x6331), actual = expected;
            for (int i = 0; i < count; ++i)
            {
                const float g = (i % 131 - 65) * 0.25f, u = (i % 97 - 48) * 0.5f;
                gate[i] = half ? fp32_to_fp16(g) : simd::fp32_to_bf16(g);
                up[i] = half ? fp32_to_fp16(u) : simd::fp32_to_bf16(u);
                const float decoded_g = half ? fp16_to_fp32(gate[i]) : simd::bf16_to_fp32(gate[i]);
                const float decoded_u = half ? fp16_to_fp32(up[i]) : simd::bf16_to_fp32(up[i]);
                const float result = ordinaryReference(decoded_g, decoded_u);
                expected[i] = half ? fp32_to_fp16(result) : simd::fp32_to_bf16(result);
            }
            const auto operation = half ? primitives::compute_swiglu_fp16 : primitives::compute_swiglu_bf16;
            operation(gate.data(), up.data(), actual.data(), count); sameBytes(actual, expected);
            auto alias = gate; operation(alias.data(), up.data(), alias.data(), count); sameBytes(alias, expected);
        }
    for (float gate_scale : {0.0f, 0x1p-24f, 1e-6f, 0.001f, 0.25f, 16.0f})
        for (float up_scale : {0x1p-24f, 1e-6f, 0.001f, 0.5f, 31.0f})
        {
            constexpr int blocks = 33;
            std::vector<Q8_1Block> gate(blocks + 1), up(blocks + 1), expected(blocks + 1), actual(blocks + 1);
            std::memset(expected.data(), 0x61, expected.size() * sizeof(Q8_1Block)); actual = expected;
            for (int block = 0; block < blocks; ++block)
            {
                gate[block].d = fp32_to_fp16(gate_scale); up[block].d = fp32_to_fp16(up_scale);
                float values[32], maximum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    gate[block].qs[i] = static_cast<std::int8_t>((block * 13 + i * 73) % 256 - 128);
                    up[block].qs[i] = static_cast<std::int8_t>((block * 31 + i * 17) % 256 - 128);
                    gate[block].sum_qs += gate[block].qs[i];
                    up[block].sum_qs += up[block].qs[i];
                    values[i] = ordinaryReference(memoryRoundedMultiply(fp16_to_fp32(gate[block].d), gate[block].qs[i]),
                                                  memoryRoundedMultiply(fp16_to_fp32(up[block].d), up[block].qs[i]));
                    maximum = std::max(maximum, std::abs(values[i]));
                }
                const float d = maximum / 127.0f, inverse = d > 1e-10f ? 1.0f / d : 0.0f;
                expected[block].d = fp32_to_fp16(d);
                int sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    const float scaled = memoryRoundedMultiply(values[i], inverse);
                    const float bounded = std::clamp(scaled, -127.0f, 127.0f);
                    expected[block].qs[i] = static_cast<std::int8_t>(std::nearbyint(bounded));
                    sum += expected[block].qs[i];
                }
                expected[block].sum_qs = static_cast<std::int16_t>(sum);
            }
            primitives::compute_swiglu_q8_1(gate.data(), up.data(), actual.data(), blocks * 32); sameBytes(actual, expected);
            auto alias = gate; primitives::compute_swiglu_q8_1(alias.data(), up.data(), alias.data(), blocks * 32);
            std::memcpy(&expected.back(), &gate.back(), sizeof(Q8_1Block)); sameBytes(alias, expected);
        }
}

/** @test Native 16-bit storage routes retain exact aliases and completion inside real teams. */
TEST_F(NativeSwiGLU, TypedRetainedTeamsAndEmptyCompletion)
{
    SwiGLUBudget budget;
    for (bool half : {false, true})
        for (int count : {0, 1, 3, 4, 5, 255, 256, 257})
        {
            std::vector<std::uint16_t> gate(count + 4, 0x6331), up(count + 4, 0x6331);
            for (int i = 0; i < count; ++i)
            {
                const float g = (i % 131 - 65) * 0.25f, u = (i % 97 - 48) * 0.5f;
                gate[i] = half ? fp32_to_fp16(g) : simd::fp32_to_bf16(g);
                up[i] = half ? fp32_to_fp16(u) : simd::fp32_to_bf16(u);
            }
            const auto operation = half ? primitives::compute_swiglu_fp16 : primitives::compute_swiglu_bf16;
            auto reference = gate;
            omp_set_num_threads(1); operation(gate.data(), up.data(), reference.data(), count);
            for (int workers : {1, 3, 10})
                for (bool alias_up : {false, true})
                {
                    auto actual = alias_up ? up : gate, expected = actual;
                    std::copy_n(reference.data(), count, expected.data());
                    const auto *g = alias_up ? gate.data() : actual.data();
                    const auto *u = alias_up ? actual.data() : up.data();
                    int observed_workers = 0;
#pragma omp parallel num_threads(workers)
                    {
#pragma omp single
                        observed_workers = omp_get_num_threads();
                        operation(g, u, actual.data(), 0);
                        operation(g, u, actual.data(), count);
                        operation(g, u, actual.data(), 0);
                    }
                    ASSERT_EQ(observed_workers, workers);
                    sameBytes(actual, expected);
                }
        }
}

/** @test Real standalone and retained teams publish serial-row bytes at every positive host worker budget. */
TEST_F(NativeSwiGLU, AllPositiveWorkersThroughTenAndRetainedTeams)
{
    SwiGLUBudget budget;
    constexpr int width = 1031, rows = 33, count = width * rows;
    std::vector<float> gate(count + 4, 711), up(count + 4, 711), actual(count + 4, 711), expected(count + 4, 711);
    for (int i = 0; i < count; ++i) { gate[i] = (i % 131 - 65) * 0.25f; up[i] = (i % 97 - 48) * 0.5f; }
    for (int workers = 1; workers <= 10; ++workers)
    {
        omp_set_num_threads(workers);
        for (bool expert : {false, true})
        {
            const auto operation = expert ? primitives::compute_swiglu_gpu_aligned_expert : primitives::compute_swiglu;
            for (int i = 0; i < count; ++i) expected[i] = expert ? device_swiglu_contract::swigluValue(gate[i], up[i]) : ordinaryReference(gate[i], up[i]);
            operation(gate.data(), up.data(), actual.data(), count); sameBytes(actual, expected);
            std::fill(actual.begin(), actual.end(), 711);
            int observed_workers = 0;
#pragma omp parallel num_threads(workers)
            {
#pragma omp single
                observed_workers = omp_get_num_threads();
                operation(gate.data(), up.data(), actual.data(), 0);
                operation(gate.data(), up.data(), actual.data(), count);
                operation(gate.data(), up.data(), actual.data(), 0);
            }
            ASSERT_EQ(observed_workers, workers);
            sameBytes(actual, expected);
        }
    }
}
#endif
} // namespace
} // namespace llaminar2::test
