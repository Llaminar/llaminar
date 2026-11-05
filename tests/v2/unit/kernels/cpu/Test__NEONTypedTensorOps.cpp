/**
 * @file Test__NEONTypedTensorOps.cpp
 * @brief Byte proof of native typed tensor arithmetic through the public ISA dispatcher.
 *
 * Complete-block Q8/Q16 operations compare with the retained scalar diagnostic
 * program, including in-place aliases, thresholds, signed extrema and ties.
 * Floating outputs also prove logical tails against independently memory-rounded
 * decode/add references; canaries detect writes beyond the authored extent.
 */
#include "tensors/SIMDHelpers.h"
#include "utils/ARMCPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
namespace
{
class NEONTypedTensorOps : public ::testing::Test
{
protected:
    /** @brief Admit every instruction the translation unit may execute. */
    static void SetUpTestSuite()
    {
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
#endif
    }
};

/** @brief Compare complete typed publications, including scale and integer-sum words.
 * @tparam T One trivially stored block/array element type.
 * @param actual Native publication.
 * @param expected Independently selected diagnostic publication.
 */
template <typename T> void sameBytes(const T &actual, const T &expected)
{ EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(T)), 0); }

/** @brief Populate finite inputs spanning full integer and floating scale domains.
 * @param seed Deterministic independent fixture identity.
 * @param q8 Receives complete Q8 source blocks.
 * @param q16 Receives complete Q16 source blocks.
 * @param delta Receives complete FP32 delta blocks.
 */
void inputs(int seed, std::array<Q8_1Block, 3> &q8, std::array<Q16_1Block, 3> &q16,
    std::array<float, 96> &delta)
{
    constexpr std::array<std::uint16_t, 7> half{0, 0x8000, 1, 0x03ff, 0x0401, 0x3555, 0x4321};
    constexpr std::array<float, 7> full{0.0f, -0.0f, 1e-15f, 1e-8f, 0.00012341f, 0.25f, 31.03125f};
    for (int block = 0; block < 3; ++block)
    {
        q8[block].d = half[(seed + block) % half.size()];
        q16[block].d = full[(seed * 3 + block) % full.size()];
        int sum8 = 0, sum16 = 0;
        for (int i = 0; i < 32; ++i)
        {
            q8[block].qs[i] = static_cast<std::int8_t>((seed * 41 + block * 13 + i * 73) % 256 - 128);
            q16[block].qs[i] = static_cast<std::int16_t>((seed * 727 + block * 129 + i * 4099) % 65536 - 32768);
            sum8 += q8[block].qs[i];
            sum16 += q16[block].qs[i];
            delta[block * 32 + i] = float(((seed + i * 7 + block * 37) % 255) - 127) * full[(seed + block + 2) % full.size()];
        }
        q8[block].sum_qs = static_cast<std::int16_t>(sum8);
        q16[block].sum_qs = sum16;
    }
}
} // namespace

/** @test Native block arithmetic and every exact in-place alias match the original program. */
TEST_F(NEONTypedTensorOps, Q8Q16FullBlocksAndAliasesMatchDiagnosticBytes)
{
    for (int seed = 0; seed < 256; ++seed)
    {
        SCOPED_TRACE(seed);
        std::array<Q8_1Block, 3> a8{}, b8{}, actual8{}, expected8{};
        std::array<Q16_1Block, 3> a16{}, b16{}, actual16{}, expected16{};
        std::array<float, 96> delta, ignored;
        inputs(seed, a8, a16, delta);
        inputs(seed + 19, b8, b16, ignored);

        simd::q8_1_add_q8_1_scalar(a8.data(), b8.data(), expected8.data(), 96);
        simd::q8_1_add_q8_1(a8.data(), b8.data(), actual8.data(), 96);
        sameBytes(actual8, expected8);
        actual8 = a8;
        simd::q8_1_add_q8_1(actual8.data(), b8.data(), actual8.data(), 96);
        sameBytes(actual8, expected8);
        actual8 = b8;
        simd::q8_1_add_q8_1(a8.data(), actual8.data(), actual8.data(), 96);
        sameBytes(actual8, expected8);

        simd::q16_1_add_q16_1_scalar(a16.data(), b16.data(), expected16.data(), 96);
        simd::q16_1_add_q16_1(a16.data(), b16.data(), actual16.data(), 96);
        sameBytes(actual16, expected16);
        actual16 = a16;
        simd::q16_1_add_q16_1(actual16.data(), b16.data(), actual16.data(), 96);
        sameBytes(actual16, expected16);

        expected16 = a16;
        simd::q16_1_add_fp32_scalar(expected16.data(), delta.data(), 96);
        actual16 = a16;
        simd::q16_1_add_fp32(actual16.data(), delta.data(), 96);
        sameBytes(actual16, expected16);

        expected16 = a16;
        simd::q16_1_add_q8_1_scalar(expected16.data(), b8.data(), 96);
        actual16 = a16;
        simd::q16_1_add_q8_1(actual16.data(), b8.data(), 96);
        sameBytes(actual16, expected16);

        simd::q16_1_to_q8_1_packed_scalar(a16.data(), expected8.data(), 3);
        simd::q16_1_to_q8_1_packed(a16.data(), actual8.data(), 3);
        sameBytes(actual8, expected8);
        simd::quantize_fp32_to_q8_1_blocks_scalar(delta.data(), expected8.data(), 96);
        simd::quantize_fp32_to_q8_1_blocks(delta.data(), actual8.data(), 96);
        sameBytes(actual8, expected8);
    }
}

/** @test Scale thresholds, scale saturation and ties-away boundaries remain explicit. */
TEST_F(NEONTypedTensorOps, TensorQuantizationThresholdTiesAndScaleClamp)
{
    constexpr std::array<float, 8> magnitudes{0.0f, 1e-12f, 9.999e-7f, 1e-6f, 1.0001e-6f, 1.0f, 127.0f, 1e9f};
    for (const auto magnitude : magnitudes)
    {
        std::array<float, 32> values;
        for (int k = 0; k < 32; ++k) values[k] = magnitude * float(k - 16) / 16.0f;
        Q8_1Block actual{}, expected{};
        simd::quantize_fp32_to_q8_1_blocks_scalar(values.data(), &expected, 32);
        simd::quantize_fp32_to_q8_1_blocks(values.data(), &actual, 32);
        sameBytes(actual, expected);
    }
    std::array<float, 32> ties;
    for (int k = 0; k < 31; ++k) ties[k] = float(k - 15) + 0.5f;
    ties[31] = 127.0f; // Scale/inverse are exactly one, so ties cannot move in preprocessing.
    Q8_1Block actual{}, expected{};
    simd::quantize_fp32_to_q8_1_blocks_scalar(ties.data(), &expected, 32);
    simd::quantize_fp32_to_q8_1_blocks(ties.data(), &actual, 32);
    sameBytes(actual, expected);
}

/** @test Full activation blocks retain SIMD ties-even; bounded partial blocks retain ties-away. */
TEST_F(NEONTypedTensorOps, ActivationBlockUsesItsDistinctFullAndPartialPrograms)
{
    for (const float magnitude : {0.0f, 9.99e-11f, 1e-10f, 1e-7f, 1.0f, 127.0f, 10000.0f})
    {
        SCOPED_TRACE(magnitude);
        std::array<float, 32> values;
        for (int k = 0; k < 31; ++k) values[k] = magnitude * (float(k - 15) + 0.5f);
        values[31] = magnitude * 127.0f;
        const float maximum = std::fabs(values[31]);
        Q8_1Block actual{}, expected{};
        if (maximum >= 1e-10f)
        {
            const float scale = maximum / 127.0f;
            // This ordinary activation program uses division, not the
            // expert contract's independently tested five-step Newton tree.
            volatile float inverse_word = 1.0f / scale;
            const float inverse = inverse_word;
            expected.d = fp32_to_fp16(scale);
            int sum = 0;
            for (int k = 0; k < 32; ++k)
            {
                const float scaled = memoryRoundedMultiply(values[k], inverse);
                const float absolute = std::fabs(scaled), lower = std::floor(absolute);
                const float fraction = absolute - lower;
                int integer = static_cast<int>(lower);
                if (fraction > 0.5f || (fraction == 0.5f && integer % 2)) ++integer;
                integer = std::clamp(scaled < 0.0f ? -integer : integer, -127, 127);
                expected.qs[k] = static_cast<std::int8_t>(integer);
                sum += integer;
            }
            expected.sum_qs = static_cast<std::int16_t>(sum);
        }
        simd::quantize_single_block(values.data(), actual);
        sameBytes(actual, expected);
        for (int live = 0; live < 32; ++live)
        {
            simd::quantize_single_block_scalar(values.data(), expected, live);
            simd::quantize_single_block(values.data(), actual, live);
            sameBytes(actual, expected);
        }
    }
}

/** @test Floating outputs preserve source multiplication edges and every bounded tail. */
TEST_F(NEONTypedTensorOps, FloatingResidualsAndDecodeAuthorOnlyLiveExtent)
{
    std::array<Q8_1Block, 3> q8;
    std::array<Q16_1Block, 3> q16;
    std::array<float, 96> input;
    inputs(27, q8, q16, input);
    std::array<float, 96> residual;
    std::array<std::uint16_t, 96> half, brain;
    for (int i = 0; i < 96; ++i)
    {
        residual[i] = float(i - 47) * 0.013457f;
        half[i] = fp32_to_fp16(residual[i]);
        brain[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(residual[i]) >> 16);
    }
    constexpr float canary = 123456.0f;
    for (const int count : {0, 1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 96})
    {
        SCOPED_TRACE(count);
        std::vector<float> actual(count + 4, canary), expected(count + 4, canary);
        simd::fused_fp32_residual_add_scalar(residual.data(), input.data(), expected.data(), count);
        simd::fused_fp32_residual_add(residual.data(), input.data(), actual.data(), count);
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        simd::fused_bf16_residual_add_scalar(brain.data(), input.data(), expected.data(), count);
        simd::fused_bf16_residual_add(brain.data(), input.data(), actual.data(), count);
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        simd::fused_fp16_residual_add_scalar(half.data(), input.data(), expected.data(), count);
        simd::fused_fp16_residual_add(half.data(), input.data(), actual.data(), count);
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        for (int i = 0; i < count; ++i)
            expected[i] = memoryRoundedAdd(memoryRoundedMultiply(fp16_to_fp32(q8[i / 32].d), float(q8[i / 32].qs[i % 32])), input[i]);
        // The private native leaf has bounded tails. The existing public Q8
        // residual interface publishes full blocks on every CPU ISA.
        simd::fused_q8_1_residual_add_neon(q8.data(), input.data(), actual.data(), count);
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        if (count % 32 == 0)
        {
            simd::fused_q8_1_residual_add(q8.data(), input.data(), actual.data(), count);
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
        for (int i = 0; i < count; ++i)
            expected[i] = memoryRoundedMultiply(fp16_to_fp32(q8[i / 32].d), float(q8[i / 32].qs[i % 32]));
        simd::dequantize_q8_1_to_fp32_neon(q8.data(), actual.data(), count);
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        if (count % 32 == 0)
        {
            simd::dequantize_q8_1_to_fp32(q8.data(), actual.data(), count);
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
    }
}

/** @test Block-only APIs never silently drop partial logical values. */
TEST_F(NEONTypedTensorOps, RejectsPartialBlockCountsAndAcceptsEmptyOperations)
{
    std::array<Q8_1Block, 2> q8{};
    std::array<Q16_1Block, 2> q16{};
    std::array<float, 64> values{};
    EXPECT_THROW(simd::q8_1_add_q8_1(q8.data(), q8.data(), q8.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::q16_1_add_q16_1(q16.data(), q16.data(), q16.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::q16_1_add_fp32(q16.data(), values.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::q16_1_add_q8_1(q16.data(), q8.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::quantize_fp32_to_q8_1_blocks(values.data(), q8.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::fused_q8_1_residual_add(q8.data(), values.data(), values.data(), 33), std::invalid_argument);
    EXPECT_THROW(simd::dequantize_q8_1_to_fp32(q8.data(), values.data(), 33), std::invalid_argument);
    simd::q8_1_add_q8_1(nullptr, nullptr, nullptr, 0);
    simd::q16_1_add_q16_1(nullptr, nullptr, nullptr, 0);
    simd::quantize_fp32_to_q8_1_blocks(nullptr, nullptr, 0);
}
} // namespace llaminar2::test
