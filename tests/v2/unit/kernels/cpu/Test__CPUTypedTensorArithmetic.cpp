/**
 * @file Test__CPUTypedTensorArithmetic.cpp
 * @brief Independent byte regression for separately rounded typed residual additions.
 *
 * Comparing a native route only with an optimized scalar diagnostic can hide a
 * shared contraction defect. These tests derive every decoded product and sum
 * through observable FP32 memory before independent ties-away requantization.
 * They exercise both the real public ISA dispatcher and the scalar diagnostic;
 * source banks remain original, and complete scale/sum/payload bytes are checked.
 */
#include "tensors/SIMDHelpers.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace llaminar2::test
{
namespace
{
/** @brief Distinct original addend programs; none selects a different quantizer. */
enum class Addend { Q16, FP32, Q8 };
/** @brief Small independent source bank, never used as production workspace. */
struct Inputs
{
    std::array<Q16_1Block, 3> q16{};
    std::array<Q8_1Block, 3> q8{};
    std::array<float, 96> fp32{};
};

/** @brief Generate original extrema and scales that expose fused decode/add ties.
 * @param seed Deterministic source identity, not a measured dispatch corpus.
 * @return Complete original source blocks and FP32 addends.
 */
Inputs inputs(int seed)
{
    constexpr std::array<std::uint16_t, 7> half{0, 0x8000, 1, 0x03ff, 0x0401, 0x3555, 0x4321};
    constexpr std::array<float, 7> full{0.0f, -0.0f, 1e-15f, 1e-8f, 0.00012341f, 0.25f, 31.03125f};
    Inputs result;
    for (int block = 0; block < 3; ++block)
    {
        auto &q16 = result.q16[block];
        auto &q8 = result.q8[block];
        q16.d = full[(seed * 3 + block) % full.size()];
        q8.d = half[(seed + block) % half.size()];
        for (int i = 0; i < 32; ++i)
        {
            q16.qs[i] = static_cast<std::int16_t>((seed * 727 + block * 129 + i * 4099) % 65536 - 32768);
            q8.qs[i] = static_cast<std::int8_t>((seed * 41 + block * 13 + i * 73) % 256 - 128);
            q16.sum_qs += q16.qs[i];
            q8.sum_qs = static_cast<std::int16_t>(q8.sum_qs + q8.qs[i]);
            result.fp32[block * 32 + i] =
                float(((seed + i * 7 + block * 37) % 255) - 127) * full[(seed + block + 2) % full.size()];
        }
    }
    return result;
}

/** @brief Reconstruct the declared decode/add/requantize program independently.
 * @param a Original Q16 residual block.
 * @param b Original addend bank, never rewritten before reference consumption.
 * @param fp32 Original floating addend bank (the residual's fixture).
 * @param block Original source block index.
 * @param addend Original addend storage program.
 * @return Complete expected Q16 publication, including exact signed sum.
 */
Q16_1Block reference(const Q16_1Block &a, const Inputs &b, const Inputs &fp32, int block, Addend addend)
{
    std::array<float, 32> values{};
    float maximum = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        const float rhs = addend == Addend::Q16 ? memoryRoundedMultiply(b.q16[block].d, float(b.q16[block].qs[i])) :
            addend == Addend::Q8 ? memoryRoundedMultiply(fp16_to_fp32(b.q8[block].d), float(b.q8[block].qs[i])) :
            fp32.fp32[block * 32 + i];
        values[i] = memoryRoundedAdd(memoryRoundedMultiply(a.d, float(a.qs[i])), rhs);
        maximum = std::max(maximum, std::abs(values[i]));
    }
    Q16_1Block output{};
    if (maximum < 1e-10f) return output;
    output.d = maximum / 32767.0f;
    const float inverse = 32767.0f / maximum;
    for (int i = 0; i < 32; ++i)
    {
        output.qs[i] = static_cast<std::int16_t>(std::clamp(
            static_cast<int>(std::round(memoryRoundedMultiply(values[i], inverse))), -32767, 32767));
        output.sum_qs += output.qs[i];
    }
    return output;
}

/** @brief Check one compiled route without deriving expected bytes from that route.
 * @tparam ScalarDiagnostic Whether to call the explicitly named scalar diagnostic.
 * @details Native calls use normal production dispatch and flags. Every source
 * product rounds before its add, irrespective of inlining or FMA availability.
 */
template <bool ScalarDiagnostic> void verify()
{
    for (int seed = 0; seed < 256; ++seed)
    {
        SCOPED_TRACE(seed);
        const auto a = inputs(seed);
        const auto b = inputs(seed + 19);
        for (auto addend : {Addend::Q16, Addend::FP32, Addend::Q8})
        {
            SCOPED_TRACE(static_cast<int>(addend));
            auto actual = a.q16;
            if constexpr (ScalarDiagnostic)
            {
                if (addend == Addend::Q16) simd::q16_1_add_q16_1_scalar(a.q16.data(), b.q16.data(), actual.data(), 96);
                if (addend == Addend::FP32) simd::q16_1_add_fp32_scalar(actual.data(), a.fp32.data(), 96);
                if (addend == Addend::Q8) simd::q16_1_add_q8_1_scalar(actual.data(), b.q8.data(), 96);
            }
            else
            {
                if (addend == Addend::Q16) simd::q16_1_add_q16_1(a.q16.data(), b.q16.data(), actual.data(), 96);
                if (addend == Addend::FP32) simd::q16_1_add_fp32(actual.data(), a.fp32.data(), 96);
                if (addend == Addend::Q8) simd::q16_1_add_q8_1(actual.data(), b.q8.data(), 96);
            }
            for (int block = 0; block < 3; ++block)
            {
                SCOPED_TRACE(block);
                const auto expected = reference(a.q16[block], b, a, block, addend);
                EXPECT_EQ(std::memcmp(&actual[block], &expected, sizeof(expected)), 0);
            }
        }
    }
}
} // namespace

/** @test Production ISA decode/add edges retain the original separate FP32 program. */
TEST(CPUTypedTensorArithmetic, NativeDecodedAdditionsMatchMemoryRoundedBytes) { verify<false>(); }
/** @test An equally contracted scalar diagnostic cannot bless the native implementation. */
TEST(CPUTypedTensorArithmetic, ScalarDiagnosticMatchesMemoryRoundedBytes) { verify<true>(); }
} // namespace llaminar2::test
