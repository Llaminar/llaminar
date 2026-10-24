/**
 * @file Test__DeviceRequestRowRanges.cpp
 * @brief Device-free admission and indexing proof for captured request rows.
 *
 * A length pointer is an identity only: deliberately invalid addresses prove
 * factories and pure validation never dereference device metadata on the host.
 * Empty requests preserve following requests' physical coordinates.
 * Merged recurrence admission also proves disjoint source/output byte ranges
 * while preserving the separate, intentional in-place state alias.
 */
#include <gtest/gtest.h>
#include "kernels/common/DeviceRequestRowRanges.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include "kernels/gdn/GDNMergedRecurrence.h"
#include <array>
#include <limits>

using namespace llaminar2;

TEST(DeviceRequestRowRanges, FixedRowsHaveNoMutableAuthority)
{
    const auto rows = DeviceRequestRowRanges::fullyActive(16);
    EXPECT_EQ(rows.requests(), 1);
    EXPECT_EQ(rows.rowsPerRequest(), 16);
    EXPECT_EQ(rows.physicalRows(), 16);
    EXPECT_EQ(rows.lengthOwner(), nullptr);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(0), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(-1), std::invalid_argument);
}

TEST(DeviceRequestRowRanges, RequestPrefixesRemainDisjointAndChecked)
{
    const auto *identity = reinterpret_cast<const std::int32_t *>(std::uintptr_t{0x12340});
    const auto rows = DeviceRequestRowRanges::deviceCounted(3, 16, identity);
    EXPECT_EQ(rows.lengthOwner(), identity);
    EXPECT_EQ(rows.physicalRows(), 48);
    for (int request = 0; request < 3; ++request)
        for (int length = 0; length <= 16; ++length)
            EXPECT_EQ(rows.activeRowsFor(request, length), length);
    EXPECT_EQ(rows.activeRowsFor(0, 0), 0);
    EXPECT_EQ(rows.activeRowsFor(2, 16), 16);
    EXPECT_EQ(rows.activeRowsFor(-1, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(3, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, -1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, 17), -1);
}

TEST(DeviceRequestRowRanges, InvalidAdmissionCannotOverflowOrOmitAuthority)
{
    const std::int32_t owner = 0;
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(1, 16, nullptr), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(0, 16, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, 0, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, std::numeric_limits<int>::max(), &owner),
        std::invalid_argument);
    const auto largest = DeviceRequestRowRanges::deviceCounted(1, std::numeric_limits<int>::max(), &owner);
    EXPECT_EQ(largest.physicalRows(), std::numeric_limits<int>::max());
}

TEST(DeviceRequestRowRanges, DeinterleaveGeometryRejectsUnrepresentableMatrices)
{
    const std::int32_t owner = 0;
    const auto rows = DeviceRequestRowRanges::deviceCounted(2, 16, &owner);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, 128), 16 * 40 * 128 * 3);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 0, 40, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, -1, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 0, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, -1), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(DeviceRequestRowRanges::fullyActive(
        std::numeric_limits<int>::max()), 1, 1, 1, 1), 0);
}

/** @test Half-open source/output ranges cover every physical request row. */
TEST(DeviceRequestRowRanges, MergedSourceOutputIntervalsPreserveOnlyStateAlias)
{
    constexpr int physical_rows = 6;
    constexpr int source_elements = physical_rows * (2 * 64 + 2 * 65);
    constexpr int output_elements = physical_rows * 2 * 65;
    std::array<float, source_elements + output_elements> storage{};
    const auto *lengths = reinterpret_cast<const int *>(std::uintptr_t{0x12340});
    gdn::MergedRecurrenceInput input{
        .rows = DeviceRequestRowRanges::deviceCounted(2, 3, lengths),
        .key_heads = 1, .value_heads = 2, .key_width = 64, .value_width = 65,
        .merged = storage.data(),
        .output = storage.data() + source_elements,
        .initial_state = storage.data(), .updated_state = storage.data(),
    };
    EXPECT_TRUE(input.sourceOutputDisjoint());
    --input.output;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = storage.data();
    EXPECT_FALSE(input.sourceOutputDisjoint());

    input.merged = storage.data() + output_elements;
    EXPECT_TRUE(input.sourceOutputDisjoint());
    --input.merged;
    EXPECT_FALSE(input.sourceOutputDisjoint());
}

/** @test Invalid shape and pointer-end overflow fail before any payload access. */
TEST(DeviceRequestRowRanges, MergedSourceOutputRejectsUnrepresentableRanges)
{
    const auto source = std::uintptr_t{0x10000};
    const auto destination = std::uintptr_t{0x100000};
    gdn::MergedRecurrenceInput input{
        .rows = DeviceRequestRowRanges::fullyActive(1),
        .key_heads = 1, .value_heads = 2, .key_width = 64, .value_width = 65,
        .merged = reinterpret_cast<const float *>(source),
        .output = reinterpret_cast<float *>(destination),
    };
    ASSERT_TRUE(input.sourceOutputDisjoint());
    input.merged = nullptr;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.merged = reinterpret_cast<const float *>(source);
    input.output = nullptr;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = reinterpret_cast<float *>(destination);
    const auto near_end = std::numeric_limits<std::uintptr_t>::max() - sizeof(float);
    input.merged = reinterpret_cast<const float *>(near_end);
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.merged = reinterpret_cast<const float *>(source);
    input.output = reinterpret_cast<float *>(near_end);
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = reinterpret_cast<float *>(destination);
    input.key_width = std::numeric_limits<int>::max();
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.key_width = 64;
    input.value_heads = 0;
    EXPECT_FALSE(input.sourceOutputDisjoint());
}
