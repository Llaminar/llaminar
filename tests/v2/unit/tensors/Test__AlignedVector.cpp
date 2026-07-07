/**
 * @file Test__AlignedVector.cpp
 * @brief Regression coverage for the cache/page alignment guarantees provided
 *        by the AlignedVector tensor utility.
 *
 * MoE CPU expert preparation uses strict NUMA page-placement checks for the
 * large NativeVNNI interleaved weight buffers stored in AlignedVector.  Those
 * checks operate at page granularity, so large buffers must begin on a page
 * boundary; otherwise the migration range can accidentally include neighboring
 * heap allocations on the first or last page and fail for reasons unrelated to
 * the expert weight buffer being prepared.
 */

#include "tensors/AlignedVector.h"

#include <gtest/gtest.h>

#include <cstdint>
#ifdef __linux__
#include <unistd.h>
#endif

namespace
{
    /**
     * @brief Return the operating-system page size for alignment assertions.
     *
     * A non-positive return value means the platform did not expose a usable
     * runtime page size, in which case Linux-specific page-alignment assertions
     * should be skipped rather than guessed.
     */
    long runtimePageSize()
    {
#ifdef __linux__
        return sysconf(_SC_PAGESIZE);
#else
        return -1;
#endif
    }
} // namespace

TEST(Test__AlignedVector, SmallAllocationsRemainCacheLineAligned)
{
    llaminar2::AlignedVector<uint8_t> buffer;
    buffer.resize_uninitialized(257);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) %
                  llaminar2::AlignedVector<uint8_t>::ALIGNMENT,
              0u);
}

TEST(Test__AlignedVector, LargeAllocationsArePageAlignedForStrictNUMAPlacement)
{
    const long page_size = runtimePageSize();
    if (page_size <= 0)
    {
        GTEST_SKIP() << "runtime page size is unavailable on this platform";
    }

    llaminar2::AlignedVector<uint8_t> buffer;
    buffer.resize_uninitialized(static_cast<size_t>(page_size) * 3u + 137u);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) %
                  static_cast<uintptr_t>(page_size),
              0u);
}
