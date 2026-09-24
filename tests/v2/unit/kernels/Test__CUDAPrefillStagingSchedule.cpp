/**
 * @file Test__CUDAPrefillStagingSchedule.cpp
 * @brief Device-free totality checks for physical prefill staging admission.
 *
 * The exact same predicate guards host dispatch and device instantiation.
 * Sweep every serialized enum byte, packed block width and owner boundary so
 * unsupported codebooks or tiles cannot quietly select another schedule.
 */
#include "kernels/cuda/gemm/CUDANativeVNNIPrefillSchedule.h"
#include <gtest/gtest.h>
#include <climits>

namespace
{
using namespace llaminar2::cuda::prefill;

TEST(CUDAPrefillStagingSchedule, EverySerializedScheduleIsExplicit)
{
    for (int value = 0; value <= 255; ++value)
    {
        const auto schedule = static_cast<PrefillStagingSchedule>(value);
        EXPECT_EQ(isKnownPrefillStagingSchedule(schedule), value < 4);
        EXPECT_EQ(supportsPrefillStaging(schedule, 16, 2, 128, 256), value < 4);
    }
}

TEST(CUDAPrefillStagingSchedule, AllPayloadWidthsAndCopyOwnerBoundaries)
{
    for (int value = 0; value < 4; ++value)
        for (int bytes = 1; bytes <= 64; ++bytes)
            for (int slots : {1, 2, 3})
                for (int columns : {1, 32, 64, 128})
                    for (int threads : {1, 31, 32, 63, 64, 127, 128, 255, 256, 512})
                    {
                        // Independent small-integer oracle: the async copy has
                        // one unique owner for both halves of every column.
                        const bool expected = value == 0 ||
                            (slots == 2 && threads >= 2 * columns &&
                             (bytes == 16 || bytes == 20));
                        ASSERT_EQ(supportsPrefillStaging(
                            static_cast<PrefillStagingSchedule>(value),
                            bytes, slots, columns, threads), expected)
                            << value << ' ' << bytes << ' ' << slots << ' '
                            << columns << ' ' << threads;
                    }
}

TEST(CUDAPrefillStagingSchedule, MalformedGeometryCannotOverflowOrAdmit)
{
    for (int value = 0; value < 4; ++value)
    {
        const auto schedule = static_cast<PrefillStagingSchedule>(value);
        for (int invalid : {INT_MIN, -1, 0})
        {
            EXPECT_FALSE(supportsPrefillStaging(schedule, invalid, 2, 128, 256));
            EXPECT_FALSE(supportsPrefillStaging(schedule, 16, invalid, 128, 256));
            EXPECT_FALSE(supportsPrefillStaging(schedule, 16, 2, invalid, 256));
            EXPECT_FALSE(supportsPrefillStaging(schedule, 16, 2, 128, invalid));
        }
    }
    EXPECT_FALSE(supportsPrefillStaging(
        PrefillStagingSchedule::AsyncPayload, 16, 2, INT_MAX, INT_MAX));
    EXPECT_TRUE(supportsPrefillStaging(
        PrefillStagingSchedule::AsyncPayload, 16, 2, INT_MAX / 2, INT_MAX));
}
} // namespace
