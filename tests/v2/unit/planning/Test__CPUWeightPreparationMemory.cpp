/**
 * @file Test__CPUWeightPreparationMemory.cpp
 * @brief Device-free totality and checked geometry for CPU preparation BOM terms.
 *
 * Actual all-format prepared allocation equality belongs to the CPU planning
 * integration group. These tests exercise admission inputs without preparing
 * weights, invoking kernels, or acquiring devices.
 */
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

TEST(CPUWeightPreparationMemory, FloatingNativeBytesNeedNoPackingShadow)
{
    for (const auto format : {"F16", "BF16", "F32"})
    {
        const auto demand = CPUWeightPreparationMemory::sourceNative(format, 65, 259);
        EXPECT_EQ(demand.persistent_bytes, 65u * 259u * (std::string_view(format) == "F32" ? 4 : 2));
        EXPECT_EQ(demand.temporary_bytes, 0u);
    }
}

TEST(CPUWeightPreparationMemory, EveryNativeCodebookHasPaddedAllocationGeometry)
{
    for (const auto &format : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(format.quant_type);
        const auto full = CPUWeightPreparationMemory::sourceNative(format.quant_type, 64, 256);
        const auto tail = CPUWeightPreparationMemory::sourceNative(format.quant_type, 65, 256);
        EXPECT_GT(full.persistent_bytes, 0u);
        EXPECT_EQ(tail.persistent_bytes, 2 * full.persistent_bytes);
        EXPECT_EQ(tail.temporary_bytes, 2 * full.temporary_bytes);
    }
}

TEST(CPUWeightPreparationMemory, DistinguishesNibbleOracleFromConsumedCacheFootprint)
{
    const auto q4 = CPUWeightPreparationMemory::sourceNative("Q4_0", 64, 32);
    EXPECT_EQ(q4.persistent_bytes, 1280u + 64u * 16u);
    EXPECT_EQ(q4.temporary_bytes, 64u * (16u + 8u));
    const auto q6 = CPUWeightPreparationMemory::sourceNative("Q6_K", 64, 256);
    EXPECT_EQ(q6.persistent_bytes, 8u * 1792u);
    EXPECT_EQ(q6.temporary_bytes, 8u * 64u * (24u + 4u));
    EXPECT_EQ(CPUWeightPreparationMemory::sourceNative("IQ2_S", 64, 256).temporary_bytes, 0u);
}

TEST(CPUWeightPreparationMemory, RejectsInvalidSourceBeforeAllocation)
{
    for (const auto &[n, k] : {std::pair<size_t, size_t>{0, 256}, {256, 0},
            {std::numeric_limits<size_t>::max(), 256}, {256, std::numeric_limits<size_t>::max()}})
        EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("F32", n, k), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("unknown", 256, 256), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("Q4_0", 256, 31), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("Q6_K", 256, 32), std::invalid_argument);
}
