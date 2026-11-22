/**
 * @file Test__PathEvidenceCounters.cpp
 * @brief Path-evidence observability contract.
 */

#include <gtest/gtest.h>

#include "execution/serving/PathEvidenceCounters.h"

namespace llaminar2::test
{
    TEST(PathEvidenceCounters, MonotonicNamedCounters)
    {
        PathEvidenceCounters counters;
        counters.record(PathEvidenceKind::FullGraphCapture, 3);
        counters.record(PathEvidenceKind::FullGraphCapture);
        EXPECT_EQ(counters.get(PathEvidenceKind::FullGraphCapture), 4u);
        EXPECT_EQ(counters.get(PathEvidenceKind::MTPDynamicDepth), 0u);
        counters.assertPathRan(PathEvidenceKind::FullGraphCapture);
        EXPECT_THROW(counters.assertPathRan(PathEvidenceKind::MTPDynamicDepth), std::logic_error);
    }
} // namespace llaminar2::test
