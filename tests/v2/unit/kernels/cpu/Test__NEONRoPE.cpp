/**
 * @file Test__NEONRoPE.cpp
 * @brief Native original-coordinate RoPE arithmetic, decode and cache-workshare proofs.
 *
 * This executable links the real RoPE translation unit. A memory-rounded pair
 * oracle checks native floating storage and bounded tails; actual production M1
 * calls certify grouped position/head scheduling without serial replay in the
 * optimized path. Angle planes are borrowed and shared by the calling team, as
 * in the CPU kernel. No model graph, full factory or performance certificate is implied.
 */
#include "kernels/cpu/primitives/NEONRoPEPrimitives.h"
#include "kernels/cpu/primitives/NEONQuantizedRoPE.h"
#include "kernels/cpu/primitives/RoPEInline.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/RoPEAngleTestStorage.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
namespace
{
namespace nr = primitives::neon_rope;
/** @brief Scope real OpenMP worker policy without changing placement/capabilities. */
class RoPEBudget final
{
public:
    /** @brief Save caller policy and use explicit physical worker budgets. */
    RoPEBudget() : previous_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore the unchanged caller policy. */
    ~RoPEBudget() { omp_set_num_threads(previous_); omp_set_dynamic(dynamic_); }
private:
    int previous_, dynamic_;
};

/** @brief Admit the actual native compilation envelope, never a spoofed ISA. */
class NEONRoPE : public ::testing::Test
{
protected:
    /** @brief An unavailable native instruction envelope is a hard test failure. */
    static void SetUpTestSuite()
    { ASSERT_TRUE(activeISALevel() == ISALevel::NEON || activeISALevel() == ISALevel::NEONDotProd); }
};

/** @brief Test-only portable encoding of one scalar result.
 * @tparam Format Explicit original tensor storage.
 * @param value Binary32 value to store through its canonical bit program.
 * @return Original storage word.
 */
template <nr::Storage Format> nr::Word<Format> encode(float value)
{
    if constexpr (Format == nr::Storage::FP32) return value;
    else if constexpr (Format == nr::Storage::Half) return fp32_to_fp16(value);
    else return simd::fp32_to_bf16(value);
}
/** @brief Independently decode one retained word for the arithmetic oracle.
 * @tparam Format Explicit original storage.
 * @param value Original stored word.
 * @return Portable binary32 value.
 */
template <nr::Storage Format> float decode(nr::Word<Format> value)
{
    if constexpr (Format == nr::Storage::FP32) return value;
    else if constexpr (Format == nr::Storage::Half) return fp16_to_fp32(value);
    else return simd::bf16_to_fp32(value);
}
/** @brief Construct finite original data plus recognizable trailing words.
 * @tparam Format Original tensor encoding.
 * @param count Number of live words.
 * @param salt Distinct query/key identity.
 * @return Live words followed by four untouched canaries.
 */
template <nr::Storage Format> std::vector<nr::Word<Format>> data(std::size_t count, int salt)
{
    std::vector<nr::Word<Format>> result(count + 4, encode<Format>(93.25f));
    for (std::size_t i = 0; i < count; ++i)
        result[i] = encode<Format>(float((int(i % 251) * 73 + salt) % 251 - 125) * 0.03125f);
    return result;
}
/** @brief Compare complete physical storage, not approximate values or only useful rows.
 * @tparam T Trivial original storage type.
 * @param actual Actual native output plus canaries.
 * @param expected Independent expected storage.
 */
template <typename T> void bytes(const std::vector<T> &actual, const std::vector<T> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0);
}
/** @brief Execute the real single/contiguous-position production API.
 * @tparam Format Original storage and corresponding API.
 * @param q Query storage in original physical coordinates.
 * @param k Original key storage.
 * @param rows Physical row count.
 * @param dim Complete original head width.
 * @param q_heads Query heads; @param k_heads Key heads.
 * @param position Absolute first position.
 * @param angles Angle planes for rows x dim/2 pairs, shared by the calling team.
 */
template <nr::Storage Format> void full(nr::Word<Format> *q, nr::Word<Format> *k,
    int rows, int dim, int q_heads, int k_heads, int position, const primitives::RoPEAngleRows &angles)
{
    if constexpr (Format == nr::Storage::FP32)
        primitives::apply_rope_vectorized(q, k, rows, dim, q_heads, k_heads, position, 10000.0f, angles);
    else if constexpr (Format == nr::Storage::Half)
        primitives::apply_rope_fp16(q, k, rows, dim, q_heads, k_heads, position, 10000.0f, angles);
    else
        primitives::apply_rope_bf16(q, k, rows, dim, q_heads, k_heads, position, 10000.0f, angles);
}
/** @brief Execute the actual grouped verifier API without compacting positions or heads.
 * @tparam Format Original retained storage.
 * @param q Query physical bank; @param k Key physical bank.
 * @param positions Original position words, including inactive rows.
 * @param rows Original physical M; @param dim Original head width.
 * @param angles Angle planes for rows x dim/2 pairs, shared by the calling team.
 */
template <nr::Storage Format> void grouped(nr::Word<Format> *q, nr::Word<Format> *k,
    const int *positions, int rows, int dim, const primitives::RoPEAngleRows &angles)
{
    if constexpr (Format == nr::Storage::FP32)
        primitives::apply_rope_decode_equivalent_rows(q, k, positions, rows, dim, 0, 3, 1, 0, 10000.0f, angles);
    else if constexpr (Format == nr::Storage::Half)
        primitives::apply_rope_fp16_decode_equivalent_rows(q, k, positions, rows, dim, 3, 1, 0, 10000.0f, angles);
    else
        primitives::apply_rope_bf16_decode_equivalent_rows(q, k, positions, rows, dim, 3, 1, 0, 10000.0f, angles);
}

/** @brief Exercise memory-rounded pairs across every native vector/storage fringe.
 * @tparam Format Explicit original storage.
 */
template <nr::Storage Format> void pairOracle()
{
    for (int pairs : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257})
    {
        auto actual = data<Format>(std::size_t(pairs) * 2, 17), expected = actual;
        std::vector<float> cosine(pairs), sine(pairs);
        for (int i = 0; i < pairs; ++i)
        {
            cosine[i] = std::cos(float(i * 17 + 13) * 0.137f);
            sine[i] = std::sin(float(i * 17 + 13) * 0.137f);
            const float x = decode<Format>(actual[i]), y = decode<Format>(actual[i + pairs]);
            expected[i] = encode<Format>(memoryRoundedAdd(memoryRoundedMultiply(x, cosine[i]), -memoryRoundedMultiply(y, sine[i])));
            expected[i + pairs] = encode<Format>(memoryRoundedAdd(memoryRoundedMultiply(x, sine[i]), memoryRoundedMultiply(y, cosine[i])));
        }
        nr::rotate_head<Format>(actual.data(), cosine.data(), sine.data(), pairs * 2);
        bytes(actual, expected);
        if constexpr (Format == nr::Storage::FP32)
        {
            actual = data<Format>(std::size_t(pairs) * 2, 17);
            apply_rope_to_head_inline(actual.data(), cosine.data(), sine.data(), pairs * 2);
            bytes(actual, expected);
        }
    }
}
/** @test Actual native pairs retain explicit rounding, portable encoding and all bounded tails. */
TEST_F(NEONRoPE, FloatingStoragePairsMatchMemoryRoundedBytes)
{
    pairOracle<nr::Storage::FP32>(); pairOracle<nr::Storage::Half>(); pairOracle<nr::Storage::BFloat16>();
}

/** @brief Exercise original grouped coordinates against real serial M1 production calls.
 * @tparam Format Original retained tensor encoding.
 */
template <nr::Storage Format> void originalRows()
{
    const RoPEBudget budget;
    for (int dim : {2, 6, 14, 32, 64, 128, 192, 256, 258, 514})
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 31, 33})
        {
            auto query = data<Format>(std::size_t(rows) * 3 * dim, 13), key = data<Format>(std::size_t(rows) * dim, 71);
            auto expected_q = query, expected_k = key;
            std::vector<int> positions(rows);
            RoPEAngleTestStorage reference(1, dim / 2);
            for (int row = 0; row < rows; ++row)
            {
                positions[row] = row % 7 == 2 ? -1 : (row % 3 == 0 ? 17231 : row % 3 == 1 ? 11 : 17231);
                if (positions[row] >= 0)
                    full<Format>(expected_q.data() + std::size_t(row) * 3 * dim, expected_k.data() + std::size_t(row) * dim,
                        1, dim, 3, 1, positions[row], reference.view(1, dim / 2));
            }
            auto actual_q = query, actual_k = key;
            RoPEAngleTestStorage storage(rows, dim / 2);
            const auto angles = storage.view(rows, dim / 2);
            omp_set_num_threads(3);
            grouped<Format>(actual_q.data(), actual_k.data(), positions.data(), rows, dim, angles);
            bytes(actual_q, expected_q); bytes(actual_k, expected_k);
            actual_q = query; actual_k = key;
#pragma omp parallel num_threads(3)
            {
                grouped<Format>(actual_q.data(), actual_k.data(), positions.data(), rows, dim, angles);
            }
            bytes(actual_q, expected_q); bytes(actual_k, expected_k);
        }
}
/** @test Original active/inactive rows, repeated positions and prefix restores retain serial bytes. */
TEST_F(NEONRoPE, OriginalVerifierRowsAndRestoredPositionsRetainSerialBytes)
{
    originalRows<nr::Storage::FP32>(); originalRows<nr::Storage::Half>(); originalRows<nr::Storage::BFloat16>();
}

/** @brief Independently rotate and requantize a diagnostic full-head oracle.
 * @tparam Block Original physical Q8/Q16 encoding.
 * @param original Immutable complete original head plus canary block.
 * @param blocks Number of live physical blocks.
 * @param cosine Original full-half cosine words.
 * @param sine Matching original sine words.
 * @return Expected independently rounded storage; no production cycle/provider is called.
 */
template <primitives::QuantizedRoPEBlock Block>
std::vector<Block> quantizedOracle(const std::vector<Block> &original, int blocks,
    const std::vector<std::int16_t> &cosine, const std::vector<std::int16_t> &sine)
{
    constexpr int width = static_cast<int>(Block::BLOCK_SIZE);
    constexpr int limit = std::is_same_v<Block, Q8_1Block> ? 127 : 32767;
    const int half = blocks * width / 2;
    std::vector<float> decoded(std::size_t(blocks) * width), rotated(decoded.size());
    auto expected = original;
    for (int block = 0; block < blocks; ++block)
    {
        const float scale = [&] {
            if constexpr (std::is_same_v<Block, Q8_1Block>) return fp16_to_fp32(original[block].d);
            else return original[block].d;
        }();
        for (int index = 0; index < width; ++index)
            decoded[std::size_t(block) * width + index] = memoryRoundedMultiply(float(original[block].qs[index]), scale);
    }
    for (int index = 0; index < half; ++index)
    {
        const float x = decoded[index], y = decoded[index + half];
        const float c = memoryRoundedMultiply(float(cosine[index]), 1.0f / 32767.0f);
        const float s = memoryRoundedMultiply(float(sine[index]), 1.0f / 32767.0f);
        rotated[index] = std::fma(x, c, -memoryRoundedMultiply(y, s));
        rotated[index + half] = std::fma(x, s, memoryRoundedMultiply(y, c));
    }
    for (int block = 0; block < blocks; ++block)
    {
        float maximum = 0;
        for (int index = 0; index < width; ++index) maximum = std::max(maximum, std::abs(rotated[std::size_t(block) * width + index]));
        const float scale = std::max(maximum / float(limit), 1e-20f);
        volatile float inverse = 1.0f / scale;
        int sum = 0;
        for (int index = 0; index < width; ++index)
        {
            const float value = memoryRoundedMultiply(rotated[std::size_t(block) * width + index], inverse);
            const int q = static_cast<int>(std::nearbyint(std::clamp(value, -float(limit), float(limit))));
            expected[block].qs[index] = static_cast<std::remove_reference_t<decltype(expected[block].qs[index])>>(q);
            sum += q;
        }
        if constexpr (std::is_same_v<Block, Q8_1Block>) expected[block].d = fp32_to_fp16(scale);
        else expected[block].d = scale;
        expected[block].sum_qs = static_cast<decltype(expected[block].sum_qs)>(sum);
    }
    return expected;
}

/** @brief Exercise all single/odd/even in-place head cycles and original metadata lifetimes.
 * @tparam Block Actual retained Q8/Q16 physical block.
 */
template <primitives::QuantizedRoPEBlock Block> void quantizedCycles()
{
    constexpr int width = static_cast<int>(Block::BLOCK_SIZE);
    for (int blocks : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17})
        for (int seed = 0; seed < 32; ++seed)
        {
            std::vector<Block> original(blocks + 1);
            std::memset(original.data(), 0x61, original.size() * sizeof(Block));
            for (int block = 0; block < blocks; ++block)
            {
                const float scale = (block + seed) % 7 == 0 ? 0.0f : (block + 1) * 0.001234f;
                if constexpr (std::is_same_v<Block, Q8_1Block>) original[block].d = fp32_to_fp16(scale);
                else original[block].d = scale;
                int sum = 0;
                for (int index = 0; index < width; ++index)
                {
                    const int q = std::is_same_v<Block, Q8_1Block> ? (block * 13 + seed * 17 + index * 73) % 256 - 128 :
                        (block * 1301 + seed * 971 + index * 731) % 65536 - 32768;
                    original[block].qs[index] = static_cast<std::remove_reference_t<decltype(original[block].qs[index])>>(q);
                    sum += q;
                }
                original[block].sum_qs = static_cast<decltype(original[block].sum_qs)>(sum);
            }
            std::vector<std::int16_t> cosine(std::size_t(blocks) * width / 2), sine(cosine.size());
            for (std::size_t index = 0; index < cosine.size(); ++index)
            {
                cosine[index] = static_cast<std::int16_t>((int(index) * 1301 + seed * 201) % 65536 - 32768);
                sine[index] = static_cast<std::int16_t>((int(index) * 3203 + seed * 13) % 65536 - 32768);
            }
            const auto expected = quantizedOracle(original, blocks, cosine, sine);
            auto actual = original;
            if constexpr (std::is_same_v<Block, Q8_1Block>)
                primitives::apply_rope_q8_1_integer_head(actual.data(), blocks, cosine.data(), sine.data());
            else primitives::apply_rope_q16_integer_head<Block>(actual.data(), blocks, cosine.data(), sine.data());
            bytes(actual, expected);
            if constexpr (std::is_same_v<Block, Q16_1Block>)
            {
                actual = original;
                primitives::apply_rope_q16_1_integer_head(actual.data(), blocks, cosine.data(), sine.data());
                bytes(actual, expected);
            }
        }
}
/** @test Native cycles cover every retained block encoding, including the formerly dropped single/odd block geometries. */
TEST_F(NEONRoPE, QuantizedSingleOddEvenBlockCyclesRetainIndependentBytes)
{
    ASSERT_EQ(std::fegetround(), FE_TONEAREST);
    quantizedCycles<Q8_1Block>(); quantizedCycles<Q16_1Block>();
    quantizedCycles<Q16_1Block_64>(); quantizedCycles<Q16_1Block_128>();
}

/** @test Native Q8 never publishes infinite half-scale metadata when a valid input rotates beyond half storage. */
TEST_F(NEONRoPE, Q8OutputScaleOverflowIsFatalBeforeBlockPublication)
{
    std::vector<Q8_1Block> head(1);
    head[0].d = 0x7bff;
    for (int i = 0; i < 32; ++i) head[0].qs[i] = 127;
    head[0].sum_qs = 127 * 32;
    const auto original = head;
    std::array<std::int16_t, 16> cosine{}, sine{};
    cosine.fill(23170); sine.fill(23170);
    EXPECT_THROW(primitives::apply_rope_q8_1_integer_head(head.data(), 1, cosine.data(), sine.data()), std::invalid_argument);
    bytes(head, original);
}

/** @test Invalid native pair geometry and coefficient aliases reject before mutation. */
TEST_F(NEONRoPE, InvalidPairGeometryAndAliasesRejectBeforeMutation)
{
    std::array<float, 32> head{}, cosine{}, sine{};
    const auto original = head;
    EXPECT_THROW(nr::rotate_head<nr::Storage::FP32>(head.data(), cosine.data(), sine.data(), 7), std::invalid_argument);
    EXPECT_THROW(nr::rotate_head<nr::Storage::FP32>(head.data(), head.data() + 1, sine.data(), 8), std::invalid_argument);
    EXPECT_THROW(nr::rotate_pairs<nr::Storage::FP32>(head.data(), head.data() + 1, cosine.data(), sine.data(), 4), std::invalid_argument);
    EXPECT_THROW(nr::rotate_pairs<nr::Storage::FP32>(nullptr, head.data(), cosine.data(), sine.data(), 4), std::invalid_argument);
    EXPECT_EQ(std::memcmp(head.data(), original.data(), sizeof(head)), 0);
}
} // namespace
} // namespace llaminar2::test
