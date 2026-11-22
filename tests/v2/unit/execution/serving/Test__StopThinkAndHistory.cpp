/**
 * @file Test__StopThinkAndHistory.cpp
 * @brief Stop/think row + per-request sequence-history contract tests.
 *
 * Reasoning budget never negative; forced sequence enters/exits exactly once;
 * terminal stop observed exactly once; history emission is exactly-once per
 * logical position and bounded by the intrinsic response limit. Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/SequenceHistory.h"
#include "execution/serving/StopThinkRow.h"

namespace llaminar2::test
{
    TEST(StopThinkRow, BudgetNeverNegativeAndForcedSequenceExactlyOnce)
    {
        StopThinkRow row;
        row.reasoning_budget_remaining = 2;
        row.enterThinking();
        row.consumeReasoningUnit();
        row.consumeReasoningUnit();
        row.consumeReasoningUnit(); // saturates at zero
        EXPECT_EQ(row.reasoning_budget_remaining, 0u);
        row.enterForcedSequence();
        EXPECT_TRUE(row.forced_sequence_active);
        EXPECT_THROW(row.enterForcedSequence(), std::logic_error);
        row.exitForcedSequence();
        EXPECT_THROW(row.exitForcedSequence(), std::logic_error);
        row.observeTerminal();
        EXPECT_THROW(row.observeTerminal(), std::logic_error);
    }

    TEST(StopThinkRow, ThinkingRequiresPositiveBudget)
    {
        StopThinkRow row;
        EXPECT_THROW(row.enterThinking(), std::logic_error);
    }

    TEST(SequenceHistory, EmissionIsExactlyOncePerPositionAndBounded)
    {
        SequenceHistory history;
        history.window = IntrinsicWindowPlan{16, 128};
        history.emitOne(7, 0);
        history.emitOne(9, 1);
        EXPECT_EQ(history.committed(), 2u);
        EXPECT_THROW(history.emitOne(7, 1), std::logic_error); // position 1 already emitted
        SequenceHistory limited;
        limited.window = IntrinsicWindowPlan{16, 2};
        limited.emitOne(1, 0);
        limited.emitOne(2, 1);
        EXPECT_THROW(limited.emitOne(3, 2), std::logic_error); // response limit
    }

    TEST(StopThinkRow, ForcedPositionClippingSkipsDepthLearningExactlyOnce)
    {
        StopThinkRow row;
        row.reasoning_budget_remaining = 1;
        row.enterThinking();
        row.noteForcedPositionClipped();
        EXPECT_TRUE(row.depth_learning_skipped);
        EXPECT_THROW(row.noteForcedPositionClipped(), std::logic_error); // exactly once
    }

    TEST(StopThinkRow, OwnedTransactionBoundaryLifecycle)
    {
        StopThinkRow row;
        row.reasoning_budget_remaining = 1;
        EXPECT_NO_THROW(row.beginTransaction()); // clean row: boundary accepted
        row.enterThinking();
        row.endThinking(); // back to Normal
        EXPECT_NO_THROW(row.beginTransaction()); // owned reset boundary after a clean transaction
        // Forced sequence must EXIT before ending the transaction.
        row.enterThinking();
        row.enterForcedSequence();
        EXPECT_THROW(row.endThinking(), std::logic_error); // exit first (no stale flag in Normal)
        row.exitForcedSequence();
        row.endThinking();
        row.beginTransaction(); // next transaction starts clean
        EXPECT_NO_THROW(row.beginTransaction());
    }
} // namespace llaminar2::test
