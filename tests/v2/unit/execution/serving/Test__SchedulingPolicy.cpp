/**
 * @file Test__SchedulingPolicy.cpp
 * @brief §11 deterministic granted-row selection policy tests.
 *
 * Selection is a pure function of (age, admission ordinal, grant usability):
 * highest saturating age wins, stable admission-order tie-break, no grant
 * selected, empty returns nullopt, and no wall-clock input exists in the
 * signature. Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/SchedulingPolicy.h"

namespace llaminar2::test
{
    TEST(SchedulingPolicy, HighestAgeWinsWithStableTieBreak)
    {
        std::vector<SelectionRow> rows;
        rows.push_back({0, 1, 5, 0, true});  // age 5, ordinal 0
        rows.push_back({1, 1, 9, 1, true});  // age 9 -> selected
        rows.push_back({2, 1, 9, 2, true});  // same age, higher ordinal
        rows.push_back({3, 1, 0, 3, false}); // no grant: never selected
        EXPECT_EQ(selectNextGrantedRow(rows), std::optional<std::size_t>{1});
    }

    TEST(SchedulingPolicy, SameAgeTieBreaksByAdmissionOrdinal)
    {
        std::vector<SelectionRow> rows;
        rows.push_back({0, 1, 4, 7, true});
        rows.push_back({1, 1, 4, 3, true}); // lower ordinal wins
        EXPECT_EQ(selectNextGrantedRow(rows), std::optional<std::size_t>{1});
    }

    TEST(SchedulingPolicy, NoGrantAndEmptyCases)
    {
        EXPECT_EQ(selectNextGrantedRow({}), std::nullopt);
        std::vector<SelectionRow> rows;
        rows.push_back({0, 1, 99, 0, false});
        rows.push_back({1, 1, 0, 1, false});
        EXPECT_EQ(selectNextGrantedRow(rows), std::nullopt);
    }
} // namespace llaminar2::test
