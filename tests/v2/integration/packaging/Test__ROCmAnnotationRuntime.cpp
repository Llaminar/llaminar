/**
 * @file Test__ROCmAnnotationRuntime.cpp
 * @brief Prove that the installed test image carries its ROCm annotation ABI.
 *
 * Native kernel harnesses link ROCTX even when a functional preflight test is
 * selected instead of a profiling run. A full development SDK used to conceal
 * this missing dependency in the minimal image. This test intentionally links
 * only GoogleTest and the actual annotation package: no HIP initialization,
 * accelerator, profiler, or Llaminar core is needed to prove the ELF closure.
 */

#include <gtest/gtest.h>
#include <rocprofiler-sdk-roctx/roctx.h>

/** @brief Exercise nested and process ranges without an attached profiler. */
TEST(ROCmAnnotationRuntime, RangesWorkWithoutProfilerOrDevice) {
    // These functions remain part of the executable's dynamic dependency even
    // when profiling is not requested. Resolve and call the real shared API,
    // rather than checking for a filename that might have missing dependencies.
    EXPECT_EQ(roctxRangePushA("preflight:outer"), 0);
    EXPECT_EQ(roctxRangePushA("preflight:inner"), 1);
    roctxMarkA("preflight:annotation-runtime");
    EXPECT_EQ(roctxRangePop(), 1);
    EXPECT_EQ(roctxRangePop(), 0);
    EXPECT_LT(roctxRangePop(), 0);

    // Process ranges use a separate lifetime from the thread-local stack.
    const roctx_range_id_t range = roctxRangeStartA("preflight:process");
    roctxRangeStop(range);
}
