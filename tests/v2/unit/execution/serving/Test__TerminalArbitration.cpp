/**
 * @file Test__TerminalArbitration.cpp
 * @brief §10.3 separate execution/delivery outcomes + AlreadyTerminal tests.
 */

#include <gtest/gtest.h>

#include "execution/serving/TerminalArbitration.h"

namespace llaminar2::test
{
    TEST(TerminalArbitration, CompletionRequiresModelTerminalButNotDelivery)
    {
        TerminalArbitration arb;
        EXPECT_THROW(arb.complete(), std::logic_error); // no model observation
        arb.observeModelTerminal();
        // Delivery is a SEPARATE authority: completion is NOT blocked by it.
        EXPECT_NO_THROW(arb.complete());
        EXPECT_EQ(arb.execution, ExecutionOutcome::Completed);
        arb.markDeliveryDrained(); // independent delivery completes later
        EXPECT_EQ(arb.delivery, DeliveryOutcome::Drained);
    }

    TEST(TerminalArbitration, LateCancellationYieldsAlreadyTerminal)
    {
        TerminalArbitration arb;
        arb.observeModelTerminal();
        arb.complete();
        EXPECT_THROW(arb.cancel(), AlreadyTerminalError);
        EXPECT_EQ(arb.execution, ExecutionOutcome::Completed); // history never rewritten
        EXPECT_THROW(arb.retire(), AlreadyTerminalError);
    }

    TEST(TerminalArbitration, CancelDuringModelWorkIsLegalAndIndependentOfDelivery)
    {
        TerminalArbitration arb;
        arb.cancel(); // mid-window: no observation required
        EXPECT_EQ(arb.execution, ExecutionOutcome::Cancelled);
        EXPECT_EQ(arb.delivery, DeliveryOutcome::Pending); // delivery separate
        arb.markDeliveryAbandoned();
        EXPECT_EQ(arb.delivery, DeliveryOutcome::Abandoned);
    }
} // namespace llaminar2::test
