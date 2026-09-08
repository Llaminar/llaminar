/**
 * @file Test__NativeVNNIPackedBits.cpp
 * @brief Exhaustive device-free proof of compact Q5 high-plane expansion.
 *
 * An independent element-by-element oracle covers every high nibble and every
 * four-element low-nibble word. CUDA/ROCm all-format GEMM gates separately
 * exercise compiler lowering with real prepared tensors and captured graphs.
 */
#include "kernels/common/NativeVNNIPackedBits.h"

#include <gtest/gtest.h>
#include <cstdint>

namespace
{
    using llaminar2::native_vnni::q5HighBitsToPackedBytes;

    /** Construct the four destination bits independently of the packed trick. */
    constexpr std::uint32_t highPlaneOracle(std::uint32_t nibble)
    {
        std::uint32_t result = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            result |= ((nibble >> byte) & 1u) << (byte * 8 + 4);
        return result;
    }

    TEST(NativeVNNIPackedBits, EveryQ5HighPlaneAndInputBitIsExact)
    {
        for (std::uint32_t nibble = 0; nibble < 16; ++nibble)
        {
            const auto expected = highPlaneOracle(nibble);
            EXPECT_EQ(q5HighBitsToPackedBytes(nibble), expected);
            // Bits outside the declared nibble may not leak into another byte.
            for (unsigned bit = 4; bit < 32; ++bit)
                EXPECT_EQ(q5HighBitsToPackedBytes(
                    nibble | (std::uint32_t{1} << bit)), expected);
            EXPECT_EQ(q5HighBitsToPackedBytes(
                nibble | 0xfffffff0u), expected);
        }
    }

    TEST(NativeVNNIPackedBits, ExhaustiveFourElementQ5Words)
    {
        // There are exactly 2^20 combinations of four five-bit values.
        for (std::uint32_t low = 0; low < 65536; ++low)
        {
            std::uint32_t packed_low = 0;
            for (unsigned byte = 0; byte < 4; ++byte)
                packed_low |= ((low >> (byte * 4)) & 15u) << (byte * 8);
            for (std::uint32_t high = 0; high < 16; ++high)
            {
                std::uint32_t expected = 0;
                for (unsigned byte = 0; byte < 4; ++byte)
                {
                    const auto value = ((low >> (byte * 4)) & 15u) |
                        (((high >> byte) & 1u) << 4);
                    expected |= value << (byte * 8);
                }
                const auto actual = packed_low |
                    q5HighBitsToPackedBytes(high);
                ASSERT_EQ(actual, expected) << "low=" << low << " high=" << high;
            }
        }
    }
}
