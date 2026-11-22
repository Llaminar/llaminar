/**
 * @file Test__PhysicalModeClassification.cpp
 * @brief §9.1-§9.3 fail-closed physical-mode classification.
 */

#include <gtest/gtest.h>

#include "execution/serving/PhysicalModeClassification.h"

namespace llaminar2::test
{
    TEST(PhysicalModeClassification, QualifiedModeAdmits)
    {
        const auto c = classifyPhysicalMode(PhysicalMode::FixedSlotFullContext);
        EXPECT_TRUE(c.isQualified());
        EXPECT_NO_THROW(c.requireQualified());
    }

    TEST(PhysicalModeClassification, UnqualifiedModeFailsClosed)
    {
        const auto c = classifyPhysicalMode(PhysicalMode::PagedKV);
        EXPECT_FALSE(c.isQualified());
        EXPECT_THROW(c.requireQualified(), std::logic_error);
    }
} // namespace llaminar2::test
