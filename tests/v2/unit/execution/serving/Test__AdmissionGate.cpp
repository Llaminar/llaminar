/**
 * @file Test__AdmissionGate.cpp
 * @brief Admission gate contract: exactly one outcome; continuation requires identity; backpressure refuses without settling mutable state.
 */

#include <gtest/gtest.h>

#include "execution/serving/AdmissionGate.h"

namespace llaminar2::test
{
    static ServingRowState newRow(std::uint32_t slot, std::uint64_t generation)
    {
        ServingRowState row;
        row.slot = slot;
        row.generation = generation;
        row.kind = AdmissionKind::NewRequest;
        return row;
    }

    TEST(AdmissionGate, ExactlyOneOutcome)
    {
        AdmissionGate gate;
        gate.admitNew(0, 1);
        EXPECT_EQ(gate.outcome, AdmissionOutcome::AcceptedNewRequest);
        EXPECT_THROW(gate.refuse(), std::logic_error);
        EXPECT_THROW(gate.admitNew(1, 1), std::logic_error);
    }

    TEST(AdmissionGate, ContinuationKeepsIdentity)
    {
        AdmissionGate gate;
        gate.admitContinuation(0, 1, newRow(0, 1));
        EXPECT_EQ(gate.outcome, AdmissionOutcome::AcceptedContinuation);
        // Refuse before settle is allowed; after settle it is not.
        AdmissionGate gate2;
        gate2.refuse();
        EXPECT_EQ(gate2.outcome, AdmissionOutcome::RefusedBackpressure);
        // Refusal is RETRYABLE: the same request may be admitted afterwards.
        gate2.admitNew(0, 1);
        EXPECT_EQ(gate2.outcome, AdmissionOutcome::AcceptedNewRequest);
        // Continuation with foreign identity refused.
        AdmissionGate gate3;
        EXPECT_THROW(gate3.admitContinuation(0, 2, newRow(0, 1)), std::logic_error);
    }
} // namespace llaminar2::test
