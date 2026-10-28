/**
 * @file Test__HostRegistrationPageRange.cpp
 * @brief Device-free whole-registration geometry and deterministic VMA policy.
 *
 * Verify all touched pages, including interior PMDs, without waiting for THP
 * collapse or a rate-limited driver warning. Partial boundary pages are covered;
 * wholly adjacent pages retain eligibility and payload bytes stay unchanged.
 * An explicit edge-only negative control reproduces the former policy defect.
 */
#include "memory/HostRegistrationPageRange.h"
#include "../../utils/HostPageMappingTestUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

using namespace llaminar2;

namespace
{
    /**
     * @brief Check every overlapping VMA, not merely sampled boundary pages.
     * @param begin First page of the authenticated half-open registration span.
     * @param end Exclusive last page boundary.
     * @return True only for complete gap-free coverage by NOHUGEPAGE VMAs.
     */
    bool allRegistrationVMAsExcludeHugePages(std::uintptr_t begin, std::uintptr_t end)
    {
        std::ifstream maps("/proc/self/smaps");
        if (!maps) throw std::runtime_error("Cannot inspect registration VMA policy");
        auto covered = begin;
        std::uintptr_t selected_end = 0;
        bool selected = false;
        for (std::string line; std::getline(maps, line);)
        {
            unsigned long start = 0, stop = 0;
            if (std::sscanf(line.c_str(), "%lx-%lx", &start, &stop) == 2)
            {
                selected = stop > begin && start < end;
                if (selected && start > covered) return false;
                selected_end = std::min<std::uintptr_t>(stop, end);
            }
            else if (selected && line.starts_with("VmFlags:"))
            {
                const auto flags = line + " ";
                if (flags.find(" nh ") == std::string::npos ||
                    flags.find(" hg ") != std::string::npos)
                    return false;
                covered = std::max(covered, selected_end);
                if (covered == end) return true;
                selected = false;
            }
        }
        return false;
    }
}

TEST(HostRegistrationPageRange, ExactFullSpanForEveryOffsetAndTail)
{
    constexpr std::size_t page = 4096u;
    for (std::size_t offset = 0; offset < page; ++offset)
        for (const auto bytes : {1u, 13u, 4096u, 8193u, 2621440u})
        {
            const auto range = HostRegistrationPageRange::forRange(0x10000u + offset, bytes, page);
            const auto expected_bytes = ((offset + bytes - 1u) / page + 1u) * page;
            EXPECT_EQ(range.pageBytes(), page);
            EXPECT_EQ(range.beginAddress(), 0x10000u);
            EXPECT_EQ(range.endAddress(), 0x10000u + expected_bytes);
            EXPECT_EQ(range.sizeBytes(), expected_bytes);
        }
}

TEST(HostRegistrationPageRange, RejectsLogicalAndRoundedEndOverflowBeforeAdvice)
{
    using Range = HostRegistrationPageRange;
    constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
    EXPECT_THROW((void)Range::forRange(0u, 1u, 4096u), std::invalid_argument);
    EXPECT_THROW((void)Range::forRange(1u, 0u, 4096u), std::invalid_argument);
    EXPECT_THROW((void)Range::forRange(1u, 1u, 0u), std::invalid_argument);
    EXPECT_THROW((void)Range::forRange(1u, 1u, 4095u), std::invalid_argument);
    EXPECT_THROW((void)Range::forRange(maximum - 7u, 9u, 4096u), std::overflow_error);
    // The final logical byte fits, but rounding its page end would wrap.
    EXPECT_THROW((void)Range::forRange(maximum, 1u, 4096u), std::overflow_error);
    EXPECT_THROW((void)Range::forRange(maximum - 4095u, 1u, 4096u), std::overflow_error);
    EXPECT_THROW((void)Range::forRange(maximum, 1u, 1u), std::overflow_error);
    const auto final_representable_page = Range::forRange(maximum - 8191u, 4096u, 4096u);
    EXPECT_EQ(final_representable_page.sizeBytes(), 4096u);
    EXPECT_EQ(final_representable_page.endAddress(), maximum - 4095u);
    const auto byte_page = Range::forRange(13u, 1u, 1u);
    EXPECT_EQ(byte_page.beginAddress(), 13u);
    EXPECT_EQ(byte_page.endAddress(), 14u);
}

TEST(HostRegistrationPageRange, FullVMASpanPreservesPartialPagesPayloadAndNeighbours)
{
    test::HostPageMapping mapping;
    auto *const first = mapping.data();
    const auto page = test::HostPageMapping::pageBytes();
    auto *const logical = first + page + 13u;
    const auto bytes = test::HostPageMapping::live_bytes - 2u * page - 26u;
    const auto range = HostRegistrationPageRange::forRange(
        reinterpret_cast<std::uintptr_t>(logical), bytes, page);
    ASSERT_TRUE(test::HostPageMapping::hasFlag(first, "hg"));
    HostRegistrationPageRange::prepare(logical, bytes);
    EXPECT_TRUE(allRegistrationVMAsExcludeHugePages(range.beginAddress(), range.endAddress()));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first + 2u * test::HostPageMapping::huge_bytes, "nh"));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first, "hg"));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(reinterpret_cast<void *>(range.endAddress()), "hg"));
    EXPECT_TRUE(std::all_of(first, first + test::HostPageMapping::payload_bytes,
        [](unsigned char byte) { return byte == 0x53; }));
    HostRegistrationPageRange::prepare(logical, bytes);
    mapping.reclaimNeighbour();
    EXPECT_TRUE(allRegistrationVMAsExcludeHugePages(range.beginAddress(), range.endAddress()));
    EXPECT_TRUE(std::all_of(first, first + test::HostPageMapping::live_bytes,
        [](unsigned char byte) { return byte == 0x53; }));
}

TEST(HostRegistrationPageRange, EdgeOnlyNegativeControlLeavesInteriorEligible)
{
    test::HostPageMapping mapping;
    auto *const first = mapping.data();
    const auto page = test::HostPageMapping::pageBytes();
    const auto bytes = test::HostPageMapping::live_bytes;
    const auto range = HostRegistrationPageRange::forRange(
        reinterpret_cast<std::uintptr_t>(first), bytes, page);
    // These are the old production policy's two exact advice operations.
    // Smaps proves its defect deterministically without relying on khugepaged.
    ASSERT_EQ(::madvise(first, page, MADV_NOHUGEPAGE), 0);
    ASSERT_EQ(::madvise(first + bytes - page, page, MADV_NOHUGEPAGE), 0);
    EXPECT_FALSE(allRegistrationVMAsExcludeHugePages(range.beginAddress(), range.endAddress()));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first + 2u * test::HostPageMapping::huge_bytes, "hg"));
    HostRegistrationPageRange::prepare(first, bytes);
    EXPECT_TRUE(allRegistrationVMAsExcludeHugePages(range.beginAddress(), range.endAddress()));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first + bytes, "hg"));
}
