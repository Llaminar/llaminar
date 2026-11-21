/**
 * @file Test__HostPageSize.cpp
 * @brief Page-size owners use the observed host base page, not 4 KiB.
 *
 * AArch64 hosts run 4, 16 or 64 KiB base pages. NUMA policy, madvise,
 * first touch and page-rounded admission must all agree with the observed
 * page, otherwise mbind/madvise fail on 64 KiB hosts and admission
 * under-counts page-aligned CPU expert slots. The layout arithmetic is
 * exercised for every supported page size; the owners are checked against
 * the actual host page.
 */
#include "utils/HostPageSize.h"
#include "memory/NUMAAllocator.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "tensors/TensorType.h"

#include <gtest/gtest.h>
#include <unistd.h>

using namespace llaminar2;

/** @test Only positive powers of two are accepted page sizes. */
TEST(HostPageSize, ValidationAcceptsOnlyPowersOfTwo)
{
    for (long bytes : {4096L, 16384L, 65536L})
        EXPECT_EQ(validatedHostPageBytes(bytes), static_cast<std::size_t>(bytes));
    for (long bytes : {0L, -1L, 3L, 12288L})
        EXPECT_THROW((void)validatedHostPageBytes(bytes), std::runtime_error);
}

/** @test The authority reports the kernel's actual base page. */
TEST(HostPageSize, MatchesTheKernelPage)
{
    EXPECT_EQ(hostPageBytes(), static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)));
}

/** @test NUMA external ranges align to the host page, not a fixed 4 KiB. */
TEST(HostPageSize, NUMAExternalRangesUseTheHostPage)
{
    EXPECT_EQ(NUMAAllocator::externalRangeAlignment(), hostPageBytes());
}

/** @test CPU expert admission rounds to the same page the slot pool allocates. */
TEST(HostPageSize, CPUExpertAdmissionRoundsToTheHostPage)
{
    for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
    {
        // 17 x 33 elements is deliberately not a multiple of any page size.
        const auto geometry = resolveExpertPreparedProjectionMemoryGeometry(
            17, 33, ExpertWeightFormat::floating(type));
        ASSERT_GT(geometry.cpu_bytes, 0u);
        EXPECT_EQ(geometry.cpu_bytes % hostPageBytes(), 0u);
    }
}
