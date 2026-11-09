/**
 * @file Test__TypedTensorBlockGeometry.cpp
 * @brief Cross-ISA rejection of incomplete typed residual publication geometry.
 *
 * Invalid counts fail before feature dispatch, pointer reads or output writes.
 * A shared interface check prevents the old count/32 truncation from leaving a
 * live tail unchanged on any CPU ISA. No tensor/model/device is constructed.
 */
#include "tensors/SIMDHelpers.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace llaminar2::test
{
/** @test Every block-only public entry rejects incomplete metadata/value extents. */
TEST(TypedTensorBlockGeometry, RejectsPartialCountsBeforeDispatchOrMutation)
{
    std::array<Q8_1Block, 3> q8;
    std::array<Q16_1Block, 3> q16;
    std::array<float, 96> values;
    values.fill(42.25f);
    const auto before_values = values;
    std::memset(q8.data(), 0x5a, sizeof(q8));
    std::memset(q16.data(), 0x39, sizeof(q16));
    const auto before8 = q8;
    const auto before16 = q16;
    for (std::size_t count = 1; count < 96; ++count)
    {
        if (count % 32 == 0) continue;
        EXPECT_THROW(simd::q8_1_add_q8_1(nullptr, nullptr, q8.data(), count), std::invalid_argument);
        EXPECT_THROW(simd::q16_1_add_q16_1(nullptr, nullptr, q16.data(), count), std::invalid_argument);
        EXPECT_THROW(simd::q16_1_add_fp32(q16.data(), nullptr, count), std::invalid_argument);
        EXPECT_THROW(simd::q16_1_add_q8_1(q16.data(), nullptr, count), std::invalid_argument);
        EXPECT_THROW(simd::quantize_fp32_to_q8_1_blocks(values.data(), q8.data(), count), std::invalid_argument);
        EXPECT_THROW(simd::fused_q8_1_residual_add(nullptr, nullptr, values.data(), count), std::invalid_argument);
        EXPECT_THROW(simd::dequantize_q8_1_to_fp32(nullptr, values.data(), count), std::invalid_argument);
    }
    for (const int count : {-1, 33, 127})
        EXPECT_THROW(simd::quantize_single_block(nullptr, q8[0], count), std::invalid_argument);
    EXPECT_EQ(std::memcmp(q8.data(), before8.data(), sizeof(q8)), 0);
    EXPECT_EQ(std::memcmp(q16.data(), before16.data(), sizeof(q16)), 0);
    EXPECT_EQ(std::memcmp(values.data(), before_values.data(), sizeof(values)), 0);
}
} // namespace llaminar2::test
