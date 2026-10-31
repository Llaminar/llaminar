/**
 * @file Test__ImmutableRowTableUploadGeometry.cpp
 * @brief Pure canonical staging eligibility, without device or inference dependencies.
 *
 * The table may consume only the ordinary loader's original slot, including an
 * explicitly tiny cap. Zero incremental capacity means temporal reuse, never
 * permission to synthesize a full-table slot or infer geometry from a BOM sum.
 */
#include "loaders/ImmutableRowTableUploadGeometry.h"
#include <gtest/gtest.h>
#include <limits>
using namespace llaminar2;
TEST(ImmutableRowTableUploadGeometry, ExactOriginalSlotBoundsNativeRowsWithoutTableSizing)
{
    const auto ordinary=resolveGPUWeightLoadMemoryGeometry(1024*1024,
        {.staging_stream_count=2,.staging_budget_bytes=1024});
    EXPECT_EQ(ordinary.staging_slot_bytes,512u);
    EXPECT_TRUE(residentImmutableRowUploadFits(1,ordinary));
    EXPECT_TRUE(residentImmutableRowUploadFits(512,ordinary));
    EXPECT_FALSE(residentImmutableRowUploadFits(513,ordinary));
    EXPECT_FALSE(residentImmutableRowUploadFits(1024*1024,ordinary));
    const auto incremental=residentImmutableRowUploadIncrementalBOM();
    EXPECT_TRUE(incremental.resources().empty());
    const auto absent=resolveGPUWeightLoadMemoryGeometry(0,{.staging_stream_count=1});
    EXPECT_FALSE(residentImmutableRowUploadFits(1,absent));
}
TEST(ImmutableRowTableUploadGeometry, MalformedOrOverflowingOriginalGeometryCannotAuthorizeUpload)
{
    const auto valid=resolveGPUWeightLoadMemoryGeometry(4096,
        {.staging_stream_count=1,.staging_budget_bytes=512});
    EXPECT_THROW((void)residentImmutableRowUploadFits(0,valid),std::invalid_argument);
    auto wrong=valid;wrong.host_staging_bytes++;
    EXPECT_THROW((void)residentImmutableRowUploadFits(1,wrong),std::invalid_argument);
    wrong=valid;wrong.staging_slot_stride_bytes++;
    EXPECT_THROW((void)residentImmutableRowUploadFits(1,wrong),std::invalid_argument);
    wrong=valid;wrong.staging_slot_bytes=std::numeric_limits<std::size_t>::max();
    wrong.staging_stream_count=2;
    EXPECT_THROW((void)residentImmutableRowUploadFits(1,wrong),std::invalid_argument);
}
