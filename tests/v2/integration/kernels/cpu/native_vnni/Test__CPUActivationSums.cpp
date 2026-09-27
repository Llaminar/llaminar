/**
 * @file Test__CPUActivationSums.cpp
 * @brief Independent integer oracle for compact-format activation corrections.
 *
 * Serial/grouped agreement cannot detect a shared correction bug. These tests
 * compare the actual SIMD primitives with independent scalar sums, including
 * INT8_MIN, every input byte position, unaligned rows, and signed cancellation.
 * Build this small target under both production CPU ISAs; it needs no model,
 * engine startup, accelerator, or allocation ledger fixture.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <random>

#include "kernels/cpu/gemm/CPUNativeVNNIActivationSums.h"

namespace
{
    using namespace llaminar2::cpu::native_vnni;

    /** Compare both reduction geometries without sharing implementation arithmetic. */
    void verify(const std::int8_t *values)
    {
        std::array<int, 4> expected{};
        for (int index = 0; index < 32; ++index)
            expected[index / 8] += static_cast<int>(values[index]);
        ASSERT_EQ(activationQuarterSums(values), expected);
        ASSERT_EQ(activationHalfSum(values), expected[0] + expected[1]);
        ASSERT_EQ(activationHalfSum(values + 16), expected[2] + expected[3]);
    }

    /** Exhaust signed values uniformly and at each independently varied byte lane. */
    TEST(CPUActivationSums, ExhaustiveSignedValuesAndPositions)
    {
        std::array<std::int8_t, 32> values{};
        for (int value = -128; value <= 127; ++value)
        {
            SCOPED_TRACE(value);
            values.fill(static_cast<std::int8_t>(value));
            verify(values.data());
            for (int index = 0; index < 32; ++index)
            {
                SCOPED_TRACE(index);
                // Opposite extrema expose saturation, sign handling and any
                // accidental carry between the independent eight-byte sums.
                for (int lane = 0; lane < 32; ++lane)
                    values[lane] = static_cast<std::int8_t>((lane & 1) ? -128 : 127);
                values[index] = static_cast<std::int8_t>(value);
                verify(values.data());
            }
        }
    }

    /** Cover every byte alignment and independently randomized half/quarter signs. */
    TEST(CPUActivationSums, UnalignedMixedSigns)
    {
        alignas(64) std::array<std::int8_t, 96> storage{};
        std::mt19937 random(0xAC71A710u);
        for (int alignment = 0; alignment < 64; ++alignment)
        {
            SCOPED_TRACE(alignment);
            for (int sample = 0; sample < 256; ++sample)
            {
                for (int lane = 0; lane < 32; ++lane)
                    storage[alignment + lane] = static_cast<std::int8_t>(
                        static_cast<int>(random() % 256) - 128);
                verify(storage.data() + alignment);
            }
        }
    }

}
