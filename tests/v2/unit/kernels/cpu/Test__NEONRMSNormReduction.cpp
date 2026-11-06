/**
 * @file Test__NEONRMSNormReduction.cpp
 * @brief Independent native RMS reduction programs over original noncontiguous storage.
 *
 * Finite FP32/FP16/BF16/full-range INT32 fixtures reconstruct the declared lane
 * tree independently, then compare grouped rows with actual native M=1 calls.
 * Native T5 has its own increasing-column FP32 FMA oracle. These math/borrow
 * tests do not certify the still-unwired complete RMSNorm kernel or its team,
 * factory, scratch, publication, model or performance paths.
 */
#include "kernels/cpu/primitives/NEONRMSNormReduction.h"
#include "tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace primitives::neon_rmsnorm;

/** @brief Produce finite original storage with wide exponent and integer domains.
 * @tparam Encoding Original physical encoding.
 * @param word Deterministic raw word, not a decoded row shadow.
 * @return One original storage element.
 */
template<Storage Encoding>
Unit<Encoding> fixtureWord(std::uint32_t word)
{
    if constexpr (Encoding == Storage::FP32)
        return std::bit_cast<float>((word & 0x807fffffu) | (((word >> 23) % 131u + 50u) << 23));
    else if constexpr (Encoding == Storage::Int32) return std::bit_cast<std::int32_t>(word);
    else if constexpr (Encoding == Storage::FP16) return std::uint16_t((word & 0x8000u) | (word % 0x7c00u));
    else return std::uint16_t((word & 0x8000u) | (word % 0x7f80u));
}

/** @brief Decode one oracle value through the canonical scalar storage contract.
 * @tparam Encoding Declared storage kind, independent of native vector helpers.
 * @param value Original word.
 * @return Exactly widened value used by the FP64 oracle.
 */
template<Storage Encoding>
double referenceValue(Unit<Encoding> value)
{
    if constexpr (Encoding == Storage::FP32 || Encoding == Storage::Int32) return double(value);
    else if constexpr (Encoding == Storage::FP16) return double(fp16_to_fp32(value));
    else return double(std::bit_cast<float>(std::uint32_t(value) << 16));
}

/** @brief Reconstruct the exact eight-lane/four-chain arithmetic without native helpers.
 * @tparam Encoding Original storage kind.
 * @param row Complete original logical row.
 * @return Independently ordered FP64 summary.
 */
template<Storage Encoding>
double referenceSum(std::span<const Unit<Encoding>> row)
{
    std::array<double, 8> lanes{};
    for (std::size_t column = 0; column < row.size(); ++column)
    {
        const double value = referenceValue<Encoding>(row[column]);
        lanes[column % 8] = std::fma(value, value, lanes[column % 8]);
    }
    const double low = (lanes[0] + lanes[2]) + (lanes[4] + lanes[6]);
    const double high = (lanes[1] + lanes[3]) + (lanes[5] + lanes[7]);
    return low + high;
}

/** @brief Sweep physical grouped rows and original padded-prefix addressing.
 * @tparam Encoding Original physical storage kind.
 * @tparam Count Shared-traversal row count1..4.
 */
template<Storage Encoding, std::size_t Count>
void checkOriginalRows()
{
    using Cell = Unit<Encoding>;
    using Bank = OriginalRows<Encoding, Count>;
    for (std::size_t columns : {0u,1u,2u,3u,4u,5u,7u,8u,9u,15u,16u,17u,31u,32u,33u,
                              63u,64u,65u,127u,128u,129u,255u,256u,257u,511u,512u,513u,8192u})
    {
        std::array<std::vector<Cell>, Count> owners;
        std::array<std::span<const Cell>, Count> originals;
        std::uint32_t state = 0x6a09e667u + std::uint32_t(columns);
        for (std::size_t row = 0; row < Count; ++row)
        {
            owners[row].resize(columns + row + 19);
            for (auto &value : owners[row])
            {
                state = state * 1664525u + 1013904223u;
                value = fixtureWord<Encoding>(state);
            }
            if constexpr (Encoding == Storage::Int32)
                if (columns >= 3)
                {
                    owners[row][row + 3] = std::numeric_limits<std::int32_t>::min();
                    owners[row][row + 4] = std::numeric_limits<std::int32_t>::max();
                    owners[row][row + 5] = 16777217;
                }
            originals[row] = {owners[row].data() + row + 3, columns};
        }
        const auto before = owners;
        const auto grouped = squaredSums(Bank::bind(originals, columns));
        for (std::size_t row = 0; row < Count; ++row)
        {
            const auto serial = squaredSums(OriginalRows<Encoding, 1>::bind({originals[row]}, columns));
            const auto expected = referenceSum<Encoding>(originals[row]);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(grouped[row]), std::bit_cast<std::uint64_t>(expected))
                << "encoding=" << int(Encoding) << " M=" << Count << " columns=" << columns << " row=" << row;
            EXPECT_EQ(std::bit_cast<std::uint64_t>(grouped[row]), std::bit_cast<std::uint64_t>(serial[0]));
        }
        EXPECT_EQ(owners, before);
        if constexpr (Encoding == Storage::FP32)
        {
            const auto grouped_t5 = t5SquaredSums(Bank::bind(originals, columns));
            for (std::size_t row = 0; row < Count; ++row)
            {
                float expected = 0.0f;
                for (float value : originals[row]) expected = std::fma(value, value, expected);
                const auto serial = t5SquaredSums(OriginalRows<Storage::FP32, 1>::bind({originals[row]}, columns));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(grouped_t5[row]), std::bit_cast<std::uint64_t>(double(expected)));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(grouped_t5[row]), std::bit_cast<std::uint64_t>(serial[0]));
            }
        }
    }
}
} // namespace

/** @test Every floating storage/fringe retains independent arithmetic and actual M=1 bytes. */
TEST(NEONRMSNormReduction, FloatingOriginalRowsMatchDeclaredAndSerialPrograms)
{
    checkOriginalRows<Storage::FP32,1>(); checkOriginalRows<Storage::FP32,2>();
    checkOriginalRows<Storage::FP32,3>(); checkOriginalRows<Storage::FP32,4>();
    checkOriginalRows<Storage::FP16,1>(); checkOriginalRows<Storage::FP16,2>();
    checkOriginalRows<Storage::FP16,3>(); checkOriginalRows<Storage::FP16,4>();
    checkOriginalRows<Storage::BF16,1>(); checkOriginalRows<Storage::BF16,2>();
    checkOriginalRows<Storage::BF16,3>(); checkOriginalRows<Storage::BF16,4>();
}

/** @test Full INT32 words widen before arithmetic; low bits above2^24 are not lost through FP32. */
TEST(NEONRMSNormReduction, IntegerOriginalRowsRetainFullWidthAndSerialBytes)
{
    checkOriginalRows<Storage::Int32,1>(); checkOriginalRows<Storage::Int32,2>();
    checkOriginalRows<Storage::Int32,3>(); checkOriginalRows<Storage::Int32,4>();
}

/** @test Malformed borrows reject without touching owner storage; zero-sized work needs no pointer. */
TEST(NEONRMSNormReduction, OriginalBorrowGeometryRejectsBeforeArithmetic)
{
    using Bank = OriginalRows<Storage::FP32, 1>;
    std::array<float, 3> original{1,2,3};
    const auto before = original;
    EXPECT_THROW((void)Bank::bind({std::span<const float>(original)},4),std::invalid_argument);
    EXPECT_THROW((void)Bank::bind({std::span<const float>{}},1),std::invalid_argument);
    EXPECT_THROW((void)Bank::bind({std::span<const float>{}},std::numeric_limits<std::size_t>::max()),std::overflow_error);
    EXPECT_EQ(squaredSums(Bank::bind({std::span<const float>{}},0))[0],0.0);
    EXPECT_EQ(original,before);
}
} // namespace llaminar2::test
