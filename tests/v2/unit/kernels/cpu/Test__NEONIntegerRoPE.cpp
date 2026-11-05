/**
 * @file Test__NEONIntegerRoPE.cpp
 * @brief Native ratio/Q15/fixed-cache RoPE proofs against independent widened arithmetic.
 *
 * Tests invoke real production dispatchers and independently model metadata
 * rounding, 64-bit products, the two Q15 bias programs, clamps and signed sums.
 * Exact-reused Q16 storage is compared with disjoint output to expose premature
 * scale publication. Diagnostic full-head oracles are test-only, not production
 * head mirrors, scalar substitutes or serving/performance certificates.
 */
#include "kernels/cpu/primitives/NEONIntegerRoPE.h"
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "utils/CPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llaminar2::test
{
namespace
{
namespace nr = primitives::neon_rope;
/** @brief Use actual native compilation/runtime admission, not a mocked ISA. */
class NEONIntegerRoPE : public ::testing::Test
{
protected:
    /** @brief Fail on a host that cannot execute the compiled envelope. */
    static void SetUpTestSuite()
    { ASSERT_TRUE(activeISALevel() == ISALevel::NEON || activeISALevel() == ISALevel::NEONDotProd); }
};
/** @brief Restore caller OpenMP state after bounded real-worker tests. */
class Budget final
{
public:
    /** @brief Keep real requested teams intact and use one standalone worker initially. */
    Budget() : old_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore unchanged caller policy. */
    ~Budget() { omp_set_num_threads(old_); omp_set_dynamic(dynamic_); }
private:
    int old_, dynamic_;
};
/** @brief Compare entire storage, including untouched canary blocks.
 * @tparam T Trivial original stored type.
 * @param actual Actual output bank; @param expected Independent complete oracle.
 */
template <typename T> void bytes(const std::vector<T> &actual, const std::vector<T> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0);
}
/** @brief Independently round a scalar metadata fixed-point ratio through memory.
 * @param source Original scale; @param destination Positive common scale.
 * @param factor Exact fixed-point multiplier.
 * @return Ties-away signed metadata ratio.
 */
std::int32_t referenceRatio(float source, float destination, float factor)
{
    volatile float quotient = source / destination;
    return static_cast<std::int32_t>(std::round(memoryRoundedMultiply(quotient, factor)));
}
/** @brief Original independent widened Q15 word program.
 * @param x First scaled input; @param y Second scaled input.
 * @param c Original cosine; @param s Original sine.
 * @param second Select the mathematical second result, not a storage precision.
 * @param signed_bias Select the explicitly tested legacy signed-bias program.
 * @param limit Exact saturation limit.
 * @return Saturated original-order word.
 */
std::int16_t referenceRotate(std::int64_t x, std::int64_t y, std::int16_t c, std::int16_t s,
    bool second, bool signed_bias, int limit)
{
    std::int64_t value = second ? x * s + y * c : x * c - y * s;
    value = (value + (signed_bias && value < 0 ? -(1LL << 14) : 1LL << 14)) >> 15;
    return static_cast<std::int16_t>(std::clamp(value, -std::int64_t(limit), std::int64_t(limit)));
}
/** @brief Seed real complete Q8 blocks, including signed extrema and zero-payload/nonzero-scale blocks.
 * @param dim Logical complete head width; @param seed Deterministic diagnostic identity.
 * @return Original head plus one retained canary block.
 */
std::vector<Q8_1Block> q8(int dim, int seed)
{
    std::vector<Q8_1Block> result(dim / 32 + 1);
    std::memset(result.data(), 0x61, result.size() * sizeof(Q8_1Block));
    for (int block = 0; block < dim / 32; ++block)
    {
        result[block].d = fp32_to_fp16((block + 1) * 0.001234f);
        int sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            result[block].qs[i] = static_cast<std::int8_t>((block * 13 + i * 73 + seed * 17) % 256 - 128);
            if ((block + seed) % 7 == 0) result[block].qs[i] = 0;
            sum += result[block].qs[i];
        }
        result[block].sum_qs = static_cast<std::int16_t>(sum);
    }
    return result;
}
/** @brief Seed original Q16 words and varying signed block scales.
 * @tparam Block Actual Q16 physical encoding.
 * @param blocks Live physical blocks; @param seed Deterministic original identity.
 * @return Original complete head followed by one canary block.
 */
template <nr::Q16Block Block> std::vector<Block> q16(int blocks, int seed)
{
    std::vector<Block> result(blocks + 1);
    std::memset(result.data(), 0x61, result.size() * sizeof(Block));
    for (int block = 0; block < blocks; ++block)
    {
        result[block].d = ((block + seed) % 7 == 0 ? 0.0f : (block + 1) * 0.0001234f) * (block % 2 ? -1.0f : 1.0f);
        int sum = 0;
        for (int i = 0; i < int(Block::BLOCK_SIZE); ++i)
        { result[block].qs[i] = static_cast<std::int16_t>((block * 1301 + seed * 971 + i * 731) % 65536 - 32768); sum += result[block].qs[i]; }
        result[block].sum_qs = sum;
    }
    return result;
}
/** @brief Materialize independent original Q15 words over the complete signed domain.
 * @param dim Original logical head width; @param seed Diagnostic identity.
 * @return Original cosine/sine planes.
 */
std::pair<std::vector<std::int16_t>, std::vector<std::int16_t>> coefficients(int dim, int seed)
{
    std::vector<std::int16_t> c(dim / 2), s(c.size());
    for (std::size_t i = 0; i < c.size(); ++i)
    {
        c[i] = static_cast<std::int16_t>((int(i) * 1301 + seed * 201) % 65536 - 32768);
        s[i] = static_cast<std::int16_t>((int(i) * 3203 + seed * 13) % 65536 - 32768);
    }
    return {c, s};
}
/** @brief Publish diagnostic metadata after independently authored words.
 * @tparam Block Actual output encoding.
 * @param output Complete writable diagnostic bank; @param blocks Live physical blocks.
 * @param scale Independent common scale.
 */
template <nr::Q16Block Block> void finish(std::vector<Block> &output, int blocks, float scale)
{
    for (int block = 0; block < blocks; ++block)
    {
        int sum = 0; for (auto value : output[block].qs) sum += value;
        output[block].d = scale; output[block].sum_qs = sum;
    }
}
/** @brief Independently evaluate normalized or legacy Q8 conversion over diagnostic original coordinates.
 * @tparam Out Original Q16 encoding.
 * @param input Original Q8 head; @param output Diagnostic output bank including canary.
 * @param dim Original logical width; @param c Original cosine words; @param s Original sine words.
 * @param legacy Explicit legacy signed-bias/common-scale-return case.
 * @return Common and stored scale pair.
 */
template <nr::Q16Block Out> std::pair<float, float> referenceQ8(const std::vector<Q8_1Block> &input,
    std::vector<Out> &output, int dim, const std::vector<std::int16_t> &c, const std::vector<std::int16_t> &s, bool legacy)
{
    float maximum = 0;
    for (int block = 0; block < dim / 32; ++block)
        for (auto value : input[block].qs)
            maximum = std::max(maximum, std::abs(memoryRoundedMultiply(float(value), fp16_to_fp32(input[block].d))));
    maximum = std::max(maximum, 1e-20f);
    volatile float common = (legacy ? maximum : memoryRoundedMultiply(maximum, 1.415f)) / 127.0f;
    std::vector<std::int32_t> ratios(dim / 32);
    for (int block = 0; block < dim / 32; ++block)
    {
        bool nonzero = false; for (auto value : input[block].qs) nonzero |= value != 0;
        ratios[block] = nonzero ? referenceRatio(fp16_to_fp32(input[block].d), common, 256.0f) : 0;
    }
    const auto put = [&](int index, std::int16_t value) { output[index / Out::BLOCK_SIZE].qs[index % Out::BLOCK_SIZE] = value; };
    for (int i = 0; i < dim / 2; ++i)
    {
        const int j = i + dim / 2;
        const std::int64_t x = std::int64_t(input[i / 32].qs[i % 32]) * ratios[i / 32];
        const std::int64_t y = std::int64_t(input[j / 32].qs[j % 32]) * ratios[j / 32];
        put(i, referenceRotate(x, y, c[i], s[i], false, legacy, 32767));
        put(j, referenceRotate(x, y, c[i], s[i], true, legacy, 32767));
    }
    const float stored = common / 256.0f;
    finish(output, dim / int(Out::BLOCK_SIZE), stored);
    return {common, stored};
}

/** @brief Check normalized Q8 integer conversion and distinct FP32 fixed-cache conversion.
 * @tparam Out Actual Q16 output encoding.
 */
template <nr::Q16Block Out> void conversionCases()
{
    for (int blocks : {1, 2, 3, 4, 5, 9})
        for (int seed = 0; seed < 16; ++seed)
        {
            const int dim = blocks * int(Out::BLOCK_SIZE);
            const auto input = q8(dim, seed);
            const auto [c, s] = coefficients(dim, seed);
            std::vector<Out> expected(blocks + 1), actual(blocks + 1);
            std::memset(expected.data(), 0x65, expected.size() * sizeof(Out)); actual = expected;
            const auto scales = referenceQ8(input, expected, dim, c, s, false);
            const float scale = primitives::apply_rope_q8_1_to_q16_head<Out>(input.data(), actual.data(), dim, c.data(), s.data());
            bytes(actual, expected); EXPECT_EQ(std::bit_cast<std::uint32_t>(scale), std::bit_cast<std::uint32_t>(scales.second));
            constexpr float cache = 32.0f;
            volatile float fixed = cache / 32767.0f, inverse = 32767.0f / cache;
            std::memset(expected.data(), 0x65, expected.size() * sizeof(Out)); actual = expected;
            const auto quantize = [inverse](float value) {
                return static_cast<std::int16_t>(std::round(std::clamp(memoryRoundedMultiply(value, inverse), -16383.0f, 16383.0f)));
            };
            for (int i = 0; i < dim / 2; ++i)
            {
                const int j = i + dim / 2;
                const float x = memoryRoundedMultiply(float(input[i / 32].qs[i % 32]), fp16_to_fp32(input[i / 32].d));
                const float y = memoryRoundedMultiply(float(input[j / 32].qs[j % 32]), fp16_to_fp32(input[j / 32].d));
                const float cosine = memoryRoundedMultiply(float(c[i]), 1.0f / 32767.0f), sine = memoryRoundedMultiply(float(s[i]), 1.0f / 32767.0f);
                expected[i / Out::BLOCK_SIZE].qs[i % Out::BLOCK_SIZE] = quantize(memoryRoundedAdd(memoryRoundedMultiply(x, cosine), -memoryRoundedMultiply(y, sine)));
                expected[j / Out::BLOCK_SIZE].qs[j % Out::BLOCK_SIZE] = quantize(memoryRoundedAdd(memoryRoundedMultiply(x, sine), memoryRoundedMultiply(y, cosine)));
            }
            finish(expected, blocks, fixed);
            primitives::apply_rope_q8_1_to_q16_head_fixed_scale<Out>(input.data(), actual.data(), dim, c.data(), s.data(), cache);
            bytes(actual, expected);
        }
}
/** @test Every physical output encoding handles original single/odd/even conversion geometry and both distinct programs. */
TEST_F(NEONIntegerRoPE, Q8ConversionsMatchIndependentMetadataAndIntegerPrograms)
{
    conversionCases<Q16_1Block>(); conversionCases<Q16_1Block_64>(); conversionCases<Q16_1Block_128>();
}

/** @brief Check widened Q16 ratio arithmetic and metadata deferral under exact input reuse.
 * @tparam Block Actual retained physical encoding.
 */
template <nr::Q16Block Block> void uniformCases()
{
    for (int blocks : {1, 2, 3, 4, 5, 9})
        for (int seed = 0; seed < 16; ++seed)
            for (bool dynamic : {false, true})
            {
                const int dim = blocks * int(Block::BLOCK_SIZE);
                const auto input = q16<Block>(blocks, seed);
                const auto [c, s] = coefficients(dim, seed);
                float scale = 128.0f / 32767.0f;
                if (dynamic)
                {
                    scale = 1e-20f;
                    for (int b = 0; b < blocks; ++b) scale = std::max(scale, std::abs(input[b].d));
                }
                auto expected = input;
                for (int i = 0; i < dim / 2; ++i)
                {
                    const int j = i + dim / 2;
                    const auto ra = referenceRatio(input[i / Block::BLOCK_SIZE].d, scale, 65536.0f);
                    const auto rb = referenceRatio(input[j / Block::BLOCK_SIZE].d, scale, 65536.0f);
                    const auto x = (std::int64_t(input[i / Block::BLOCK_SIZE].qs[i % Block::BLOCK_SIZE]) * ra) >> 16;
                    const auto y = (std::int64_t(input[j / Block::BLOCK_SIZE].qs[j % Block::BLOCK_SIZE]) * rb) >> 16;
                    expected[i / Block::BLOCK_SIZE].qs[i % Block::BLOCK_SIZE] = referenceRotate(x, y, c[i], s[i], false, false, dynamic ? 32767 : 16383);
                    expected[j / Block::BLOCK_SIZE].qs[j % Block::BLOCK_SIZE] = referenceRotate(x, y, c[i], s[i], true, false, dynamic ? 32767 : 16383);
                }
                finish(expected, blocks, scale);
                for (bool reuse : {false, true})
                {
                    auto actual = input;
                    const auto *source = reuse ? actual.data() : input.data();
                    float published = -919.0f;
                    if (dynamic)
                        primitives::apply_rope_q16_to_q16_head_dynamic_scale<Block>(source, actual.data(), dim, c.data(), s.data(), &published);
                    else
                        primitives::apply_rope_q16_to_q16_head_fixed_scale<Block>(source, actual.data(), dim, c.data(), s.data(), 128.0f);
                    bytes(actual, expected);
                    if (dynamic) EXPECT_EQ(std::bit_cast<std::uint32_t>(published), std::bit_cast<std::uint32_t>(scale));
                }
            }
}
/** @test Exact Q16 reuse never overwrites a scale before the final original-half consumption. */
TEST_F(NEONIntegerRoPE, FixedDynamicQ16WidenedProductsAndDeferredScales)
{
    uniformCases<Q16_1Block>(); uniformCases<Q16_1Block_64>(); uniformCases<Q16_1Block_128>();
}

/** @test Legacy signed-bias Q8 conversion retains its distinct common return scale and stored d. */
TEST_F(NEONIntegerRoPE, LegacySignedBiasAndScaleIdentity)
{
    for (int blocks : {1, 2, 3, 4, 5, 9})
    {
        const int dim = blocks * 32;
        const auto input = q8(dim, 3);
        const auto [c, s] = coefficients(dim, 7);
        std::vector<Q16_1Block> expected(blocks + 1), actual(blocks + 1);
        std::memset(expected.data(), 0x61, expected.size() * sizeof(Q16_1Block)); actual = expected;
        const auto scales = referenceQ8(input, expected, dim, c, s, true);
        const float common = nr::q8_legacy_head(input.data(), actual.data(), blocks, c.data(), s.data());
        bytes(actual, expected); EXPECT_EQ(std::bit_cast<std::uint32_t>(common), std::bit_cast<std::uint32_t>(scales.first));
        if (blocks == 2)
        {
            actual = expected;
            std::memset(actual.data(), 0x62, 2 * sizeof(Q16_1Block));
            nr::q8_pair_to_q16(input[0], input[1], actual[0], actual[1], c.data(), s.data(), common);
            bytes(actual, expected);
        }
    }
    const int32_t x[4] = {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 32768, -32768};
    const int32_t y[4] = {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), -32768, 32768};
    const std::int16_t c[4] = {-32768, 32767, -32768, 32767}, s[4] = {-32768, -32768, 32767, 32767};
    const auto result = nr::integer_rotate4<nr::Q15Bias::SignedHalf, 32767>(vld1q_s32(x), vld1q_s32(y), c, s);
    std::int32_t a[4], b[4]; vst1q_s32(a, result.first); vst1q_s32(b, result.second);
    for (int i = 0; i < 4; ++i)
    { EXPECT_EQ(a[i], referenceRotate(x[i], y[i], c[i], s[i], false, true, 32767)); EXPECT_EQ(b[i], referenceRotate(x[i], y[i], c[i], s[i], true, true, 32767)); }
}

/** @test Signed ratio endpoints with either byte extreme widen before Q15, rather than wrapping a scaled int32 lane. */
TEST_F(NEONIntegerRoPE, Q8CommonScalePairRetainsWidenedSignedByteExtrema)
{
    constexpr float common = 0x1p-16f;
    for (float source_scale : {-1.0f, 1.0f})
        for (int payload : {-128, 127})
            for (int coefficient : {-32768, 32767})
            {
                Q8_1Block a{}, b{};
                a.d = fp32_to_fp16(source_scale);
                for (int i = 0; i < 32; ++i) a.qs[i] = static_cast<std::int8_t>(payload);
                a.sum_qs = static_cast<std::int16_t>(payload * 32);
                std::array<std::int16_t, 32> c{}, s{};
                c.fill(static_cast<std::int16_t>(coefficient));
                std::vector<Q16_1Block> actual(3), expected(3);
                std::memset(actual.data(), 0x65, actual.size() * sizeof(Q16_1Block)); expected = actual;
                const auto ratio = referenceRatio(source_scale, common, 256.0f);
                const auto x = std::int64_t(payload) * ratio;
                for (int i = 0; i < 32; ++i)
                {
                    expected[0].qs[i] = referenceRotate(x, 0, c[i], s[i], false, true, 32767);
                    expected[1].qs[i] = referenceRotate(x, 0, c[i], s[i], true, true, 32767);
                }
                finish(expected, 2, common / 256.0f);
                nr::q8_pair_to_q16(a, b, actual[0], actual[1], c.data(), s.data(), common);
                bytes(actual, expected);
            }
}

/** @test Native hybrid pairs match independent decoded words and separate FP32 arithmetic beyond the former head limit. */
TEST_F(NEONIntegerRoPE, HybridPairsMatchIndependentMemoryRoundedBytes)
{
    for (int dim : {32, 64, 96, 128, 256, 512, 1024})
    {
        const auto input = q8(dim, 3);
        std::vector<float> cosine(dim / 2), sine(cosine.size()), expected(dim + 4, 919.25f), actual = expected;
        for (int i = 0; i < dim / 2; ++i)
        {
            cosine[i] = std::cos(float(i * 13 + 7) * 0.137f); sine[i] = std::sin(float(i * 13 + 7) * 0.137f);
            const int j = i + dim / 2;
            const float x = memoryRoundedMultiply(float(input[i / 32].qs[i % 32]), fp16_to_fp32(input[i / 32].d));
            const float y = memoryRoundedMultiply(float(input[j / 32].qs[j % 32]), fp16_to_fp32(input[j / 32].d));
            expected[i] = memoryRoundedAdd(memoryRoundedMultiply(x, cosine[i]), -memoryRoundedMultiply(y, sine[i]));
            expected[j] = memoryRoundedAdd(memoryRoundedMultiply(x, sine[i]), memoryRoundedMultiply(y, cosine[i]));
        }
        nr::q8_to_fp32_head(input.data(), actual.data(), dim, cosine.data(), sine.data());
        bytes(actual, expected);
    }
}

/** @test Malformed geometry, scalar publication aliases and nonfinite scales fail before affected storage is authored. */
TEST_F(NEONIntegerRoPE, GeometryAliasesAndInvalidScaleRejection)
{
    auto input = q16<Q16_1Block_64>(3, 1);
    auto output = input;
    const auto original = input;
    const auto [c, s] = coefficients(192, 0);
    EXPECT_THROW(primitives::apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block_64>(input.data(), output.data(), 192, c.data(), s.data(), &input[0].d), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block_64>(input.data(), output.data() + 1, 191, c.data(), s.data(), 128.0f), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block_64>(input.data(), input.data() + 1, 192, c.data(), s.data(), 128.0f), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block_64>(input.data(), output.data(), 192, c.data(), s.data(), 0.0f), std::invalid_argument);
    bytes(input, original); bytes(output, original);
    auto source_q8 = q8(192, 2);
    source_q8[0].d = 0x7c00;
    EXPECT_THROW(primitives::apply_rope_q8_1_to_q16_head<Q16_1Block_64>(source_q8.data(), output.data(), 192, c.data(), s.data()), std::invalid_argument);
    bytes(output, original);
}
} // namespace
} // namespace llaminar2::test
