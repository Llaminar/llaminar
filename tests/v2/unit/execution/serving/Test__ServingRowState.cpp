/**
 * @file Test__ServingRowState.cpp
 * @brief §6 ABI substrate: slot/generation identity, admission kind, continuation law.
 *
 * Device-free contract tests: the identity is slot+generation (no reuse across
 * replacement), continuations are only allowed from an admitted NewRequest
 * with identical slot/generation/policy identity, and the committed count
 * carries the predecessor's terminal value.
 */

#include <gtest/gtest.h>

#include "execution/serving/ServingRowState.h"

namespace llaminar2::test
{
    static ServingRowState newRow(std::uint32_t slot, std::uint64_t generation)
    {
        ServingRowState row;
        row.slot = slot;
        row.generation = generation;
        row.kind = AdmissionKind::NewRequest;
        row.policy_identity = 42;
        row.response_limit = 128;
        row.window = IntrinsicWindowPlan{16, 128};
        row.rng.seed = resolveRequestSeed(777, 0);
        return row;
    }

    TEST(ServingRowState, IdentityIsSlotAndGeneration)
    {
        const auto a = newRow(0, 1);
        ServingRowState b = newRow(0, 2); // replacement = new generation
        EXPECT_FALSE(a.samePhysicalSlotIdentity(b));
        ServingRowState c = newRow(0, 1);
        EXPECT_TRUE(a.samePhysicalSlotIdentity(c));
    }

    TEST(ServingRowState, ContinuationRequiresExactIdentityAndPolicy)
    {
        auto row = newRow(0, 1);
        row.response_limit = 16;
        row.committed = 8; // NON-terminal predecessor (has budget)
        ServingRowState cont = newRow(0, 1);
        cont.kind = AdmissionKind::ContinueResponse;
        cont.response_limit = 16; // same immutable limit as the predecessor
        cont.window = IntrinsicWindowPlan{16, 128};
        // cont must preserve the immutable RNG seed + policy + limit.
        admitContinuation(cont, row);
        EXPECT_EQ(cont.committed, 8u); // carries the predecessor's current count
        // Foreign generation / policy refused.
        ServingRowState bad = newRow(0, 1);
        bad.kind = AdmissionKind::ContinueResponse;
        bad.generation = 2;
        EXPECT_THROW(admitContinuation(bad, row), std::logic_error);
        ServingRowState bad2 = newRow(0, 1);
        bad2.kind = AdmissionKind::ContinueResponse;
        bad2.policy_identity = 43;
        EXPECT_THROW(admitContinuation(bad2, row), std::logic_error);
    }

    TEST(ServingRowState, SuccessiveContinuationsAreAllowed)
    {
        // §14: successive continuations of the SAME admitted request are valid.
        auto first = newRow(0, 1);
        first.response_limit = 16;
        first.committed = 8;
        ServingRowState second = newRow(0, 1);
        second.kind = AdmissionKind::ContinueResponse;
        second.response_limit = 16; // same immutable limit
        second.window = IntrinsicWindowPlan{16, 128};
        EXPECT_NO_THROW(admitContinuation(second, first));
        ServingRowState third = newRow(0, 1);
        third.kind = AdmissionKind::ContinueResponse;
        third.response_limit = 16;
        third.window = IntrinsicWindowPlan{16, 128};
        EXPECT_NO_THROW(admitContinuation(third, second)); // continuation of a continuation
    }

    TEST(ServingRowState, ContinuationRejectsTerminalPredecessor)
    {
        auto nonterminal = newRow(0, 1);
        nonterminal.response_limit = 16;
        nonterminal.committed = 8;
        ServingRowState cont = newRow(0, 1);
        cont.kind = AdmissionKind::ContinueResponse;
        cont.response_limit = 16;
        EXPECT_NO_THROW(admitContinuation(cont, nonterminal));
        nonterminal.committed = 16; // terminal: a finished response is NOT continued
        EXPECT_THROW(admitContinuation(cont, nonterminal), std::logic_error);
    }
} // namespace llaminar2::test
