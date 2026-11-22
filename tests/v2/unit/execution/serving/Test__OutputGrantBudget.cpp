#include <gtest/gtest.h>

#include "execution/serving/OutputGrantBudget.h"

namespace llaminar2::test
{
    TEST(OutputGrantBudget, RemainingAndFitChecks)
    {
        OutputGrantBudget budget{128};
        EXPECT_EQ(budget.remaining(0), 128u);
        EXPECT_EQ(budget.remaining(100), 28u);
        EXPECT_EQ(budget.remaining(200), 0u); // clamped at zero
        WindowGrant grant;
        grant.grant_id = 1;
        grant.serving_instance = 7;
        grant.destination_reservation = 0xabcd;
        grant.window = IntrinsicWindowPlan{16, 128};
        EXPECT_NO_THROW(budget.requireFits(grant, 20));
        // Partial final window: c=120, remaining 8 (< quantum 16) is ALLOWED.
        EXPECT_NO_THROW(budget.requireFits(grant, 120));
        OutputGrantBudget exhausted{8};
        EXPECT_THROW(exhausted.requireFits(grant, 8), std::logic_error); // nothing left
    }
} // namespace llaminar2::test
