/**
 * @file Test__RaggedRowCompaction.cpp
 * @brief §12.4 ragged-bank compaction validation.
 */

#include <gtest/gtest.h>

#include "execution/serving/RaggedRowCompaction.h"

namespace llaminar2::test
{
    TEST(RaggedRowCompaction, ExactPerRowCompactionAdmits)
    {
        const std::vector<std::uint64_t> live{3, 0, 5};
        RaggedCompaction compaction;
        compaction.candidates = {{0, 3}, {2, 5}};
        EXPECT_NO_THROW(validateRaggedBank(live, compaction));
    }

    TEST(RaggedRowCompaction, MismatchAndHostAuthoredRefused)
    {
        const std::vector<std::uint64_t> live{3, 0, 5};
        RaggedCompaction wrong;
        wrong.candidates = {{0, 4}, {2, 5}}; // live mismatch
        EXPECT_THROW(validateRaggedBank(live, wrong), std::logic_error);
        RaggedCompaction host;
        host.candidates = {{0, 3}, {2, 5}};
        host.device_authored = false;
        EXPECT_THROW(validateRaggedBank(live, host), std::logic_error);
        RaggedCompaction extra;
        extra.candidates = {{0, 3}, {2, 5}, {1, 0}}; // counts an inactive row
        EXPECT_THROW(validateRaggedBank(live, extra), std::logic_error);
    }

    TEST(RaggedRowCompaction, WholeBankSingleCountIsForbidden)
    {
        RaggedCompaction any;
        EXPECT_THROW(rejectWholeBankSingleLiveCount(3, any), std::logic_error);
    }

    TEST(RaggedRowCompaction, DuplicateAndInactiveMappingsRefused)
    {
        const std::vector<std::uint64_t> live{3, 0, 5};
        // Both negatives pass cardinality (2 candidates for the 2 LIVE rows):
        // they discriminate the bijection fix, not the count check.
        RaggedCompaction dup;
        dup.candidates = {{0, 3}, {0, 3}}; // row 0 mapped twice (row 2 absent)
        EXPECT_THROW(validateRaggedBank(live, dup), std::logic_error);
        RaggedCompaction inactive;
        inactive.candidates = {{0, 3}, {1, 0}}; // inactive row 1 mapped (row 2 absent)
        EXPECT_THROW(validateRaggedBank(live, inactive), std::logic_error);
    }
} // namespace llaminar2::test
