/**
 * @file Test__NativeVNNIPrefillSchedule.cpp
 * @brief Device-free coverage of CPU prefill worksharing and wave occupancy.
 *
 * These tests cover schedule totality and validation without allocating weights
 * or running kernels. The integration sweep separately proves serial-row byte
 * equality for every format through the selected production launcher.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIPrefillSchedule.h"

#include <gtest/gtest.h>
#include <limits>

namespace
{
    using namespace llaminar2::cpu::native_vnni;

    TEST(NativeVNNIPrefillSchedule, DistributesUnderfilledWavesAcrossRowPairs)
    {
        for (int rows : {32, 512, 4096})
        for (int columns : {2048, 4096})
            EXPECT_EQ(resolvePrefillSchedule(
                PrefillSchedulePolicy::Auto, {rows, columns, columns, 1, 28}),
                PrefillSchedulePolicy::TwoRowPairGrid);
    }

    TEST(NativeVNNIPrefillSchedule, RetainsWeightReuseForBalancedWaves)
    {
        for (int workers = 1; workers <= 128; ++workers)
        for (int waves : {1, 2, 3, 5, 17})
        for (int rows : {1, 2, 3, 32, 512})
        {
            const int columns = workers * waves * 64;
            EXPECT_EQ(resolvePrefillSchedule(
                PrefillSchedulePolicy::Auto, {rows, columns, columns, 1, workers}),
                PrefillSchedulePolicy::TwoRowNMajor);
        }
    }

    TEST(NativeVNNIPrefillSchedule, DoesNotPayGridSetupForSmallBatches)
    {
        for (int rows = 1; rows <= 6; ++rows)
            EXPECT_EQ(resolvePrefillSchedule(
                PrefillSchedulePolicy::Auto, {rows, 2048, 2048, 1, 28}),
                PrefillSchedulePolicy::TwoRowNMajor);
    }

    TEST(NativeVNNIPrefillSchedule, PreservesNarrowRowsAndExplicitCandidates)
    {
        for (int columns : {1, 31, 63, 64, 128})
        {
            const PrefillScheduleGeometry shape{32, columns, columns, 1, 28,
                CPUNativeVNNIEncoding::ExpandedInt8, PrefillRowKernelSet::WideRows};
            EXPECT_EQ(resolvePrefillSchedule(PrefillSchedulePolicy::Auto, shape),
                      PrefillSchedulePolicy::RowChunkGrid);
            for (auto forced : {PrefillSchedulePolicy::RowChunkGrid,
                                PrefillSchedulePolicy::TwoRowNMajor,
                                PrefillSchedulePolicy::TwoRowPairGrid,
                                PrefillSchedulePolicy::FourRowGrid})
                EXPECT_EQ(resolvePrefillSchedule(forced, shape), forced);
        }
    }

    TEST(NativeVNNIPrefillSchedule, EveryPositiveWorkerBudgetHasATotalSchedule)
    {
        for (int workers = 1; workers <= 128; ++workers)
        for (int rows : {1, 2, 3, 4, 15, 16, 31, 32, 127, 512, 8192})
        for (int columns : {1, 63, 65, 511, 1024, 1792, 2048, 4096, 8193, 151936})
        for (int block_chunks : {1, 2, 4, 8, 16})
            EXPECT_NE(resolvePrefillSchedule(
                PrefillSchedulePolicy::Auto,
                {rows, columns, columns, block_chunks, workers}),
                PrefillSchedulePolicy::Auto);
        constexpr int largest = std::numeric_limits<int>::max();
        EXPECT_NO_THROW((void)resolvePrefillSchedule(
            PrefillSchedulePolicy::Auto, {largest, largest, largest, largest, largest}));
        EXPECT_NO_THROW((void)resolvePrefillSchedule(
            PrefillSchedulePolicy::Auto, {largest, largest, largest, 1, 1}));
    }

    TEST(NativeVNNIPrefillSchedule, LongCompactGridReusesFourRowsOnlyOnWideISA)
    {
        for (const auto encoding : {CPUNativeVNNIEncoding::NibbleLUT,
                                   CPUNativeVNNIEncoding::ExpandedInt8,
                                   CPUNativeVNNIEncoding::Q6KNativeDualScale,
                                   CPUNativeVNNIEncoding::CompactMultiScale})
        for (const auto kernels : {PrefillRowKernelSet::Pairwise, PrefillRowKernelSet::WideRows})
        for (int rows : {32, 64, 127, 128, 129, 511, 512, 4096})
        {
            const bool wide = rows >= 128 && kernels == PrefillRowKernelSet::WideRows &&
                (encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale ||
                 encoding == CPUNativeVNNIEncoding::CompactMultiScale);
            EXPECT_EQ(resolvePrefillSchedule(PrefillSchedulePolicy::Auto,
                {rows, 2048, 2048, 1, 28, encoding, kernels}),
                wide ? PrefillSchedulePolicy::FourRowGrid : PrefillSchedulePolicy::TwoRowPairGrid);
        }
        // Explicit candidates remain forceable and retain their physical name.
        const PrefillScheduleGeometry wide{512, 2048, 2048, 1, 28,
            CPUNativeVNNIEncoding::Q6KNativeDualScale, PrefillRowKernelSet::WideRows};
        EXPECT_EQ(resolvePrefillSchedule(PrefillSchedulePolicy::TwoRowPairGrid, wide),
                  PrefillSchedulePolicy::TwoRowPairGrid);
        EXPECT_THROW((void)resolvePrefillSchedule(PrefillSchedulePolicy::FourRowGrid,
            {512, 2048, 2048, 1, 28}), std::invalid_argument);
    }

    TEST(NativeVNNIPrefillSchedule, RejectsInvalidGeometryAndPolicy)
    {
        for (const auto shape : {
                 PrefillScheduleGeometry{0, 64, 64, 1, 1}, {1, 0, 64, 1, 1},
                 {1, 64, 63, 1, 1}, {1, 64, 64, 0, 1}, {1, 64, 64, 1, 0}})
            EXPECT_THROW((void)resolvePrefillSchedule(PrefillSchedulePolicy::Auto, shape),
                         std::invalid_argument);
        EXPECT_THROW((void)resolvePrefillSchedule(
            static_cast<PrefillSchedulePolicy>(99), {1, 64, 64, 1, 1}),
            std::invalid_argument);
    }
}
