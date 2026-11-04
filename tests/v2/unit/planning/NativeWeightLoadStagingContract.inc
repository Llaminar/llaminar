/**
 * @file NativeWeightLoadStagingContract.inc
 * @brief Actual bounded geometry contract for shared native input and consumer scratch.
 *
 * These metadata-only tests do not materialize model tensors or claim native
 * execution readiness. Source units model indivisible native blocks/rows.
 */
#include "loaders/NativeWeightLoadStaging.h"

namespace llaminar2 {
TEST(NativeWeightLoadStagingContract, SourceAndSimultaneousScratchShareOneExactPhysicalCap)
{
    const auto placement=CPUStoragePlacement::onNode(7); // Metadata, not an observed socket.
    EXPECT_EQ(configuredNativeWeightLoadStagingBudgetBytes(),
        gpuPerDeviceLoadStagingBudgetBytes(debugEnv().rocm.repack_budget_mb));
    for(const size_t native:{size_t{2},size_t{4},size_t{18},size_t{34},size_t{144},size_t{210}})
        for(const size_t scratch:{size_t{0},size_t{128},size_t{1024}}) {
            SCOPED_TRACE(native);
            SCOPED_TRACE(scratch);
            const NativeWeightLoadStagingRequest request{native,scratch,96,1000000};
            const auto geometry=resolveNativeWeightLoadStaging(request,placement,8192);
            ASSERT_GT(geometry.maximum_units_per_window,0u);
            EXPECT_LT(geometry.maximum_units_per_window,request.maximum_units);
            EXPECT_EQ(geometry.native_bytes,native*geometry.maximum_units_per_window);
            EXPECT_EQ(geometry.scratch_bytes,scratch*geometry.maximum_units_per_window+96);
            EXPECT_EQ(geometry.scratch_offset%64,0u);
            EXPECT_GE(geometry.scratch_offset,geometry.native_bytes);
            EXPECT_EQ(geometry.physical_bytes,placement.physicalBytes<uint8_t>(geometry.scratch_offset+geometry.scratch_bytes));
            EXPECT_LE(geometry.physical_bytes,8192u);
            const auto one=resolveNativeWeightLoadStaging({native,scratch,96,1},placement,std::nullopt);
            EXPECT_THROW((void)resolveNativeWeightLoadStaging(request,placement,one.physical_bytes-1),std::invalid_argument);
            const auto next=resolveNativeWeightLoadStaging({native,scratch,96,geometry.maximum_units_per_window+1},placement,std::nullopt);
            EXPECT_GT(next.physical_bytes,8192u);
        }
}

TEST(NativeWeightLoadStagingContract, LargeSmallEmptyInvalidAndOverflowDoNotInventWholeTensorFitLimit)
{
    const auto placement=CPUStoragePlacement::local();
    const auto large=resolveNativeWeightLoadStaging({144,1024,128,std::numeric_limits<size_t>::max()},placement,4096);
    EXPECT_GT(large.maximum_units_per_window,0u);
    EXPECT_LE(large.physical_bytes,4096u);
    const auto small=resolveNativeWeightLoadStaging({144,1024,128,1},placement,4096);
    EXPECT_EQ(small.maximum_units_per_window,1u);
    EXPECT_EQ(resolveNativeWeightLoadStaging({144,1024,128,std::numeric_limits<size_t>::max()},placement,4096),large);
    EXPECT_THROW((void)resolveNativeWeightLoadStaging({0,0,0,1},placement,4096),std::invalid_argument);
    EXPECT_THROW((void)resolveNativeWeightLoadStaging({1,0,0,0},placement,4096),std::invalid_argument);
    EXPECT_THROW((void)resolveNativeWeightLoadStaging({1,0,0,1},placement,0),std::invalid_argument);
    EXPECT_THROW((void)resolveNativeWeightLoadStaging({1,1,std::numeric_limits<size_t>::max(),1},placement,std::nullopt),std::overflow_error);
    EXPECT_THROW((void)resolveNativeWeightLoadStaging({std::numeric_limits<size_t>::max(),1,0,2},placement,std::nullopt),std::overflow_error);
}

} // namespace llaminar2
