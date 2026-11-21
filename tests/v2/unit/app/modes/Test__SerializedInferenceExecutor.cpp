/**
 * @file Test__SerializedInferenceExecutor.cpp
 * @brief Unit tests for typed serving admission in the serialized HTTP executor.
 *
 * The executor is the response-lease admission authority at the serving
 * boundary (design §9.4). These tests prove that both typed axes are enforced:
 * the response-lease total R (capacity) and the queue/preparing bound Q, where
 * the single execution-slot occupant is never counted against Q. An authored
 * Q = 0 therefore still admits the one response occupying the worker while
 * refusing every additional accepted response.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "app/modes/SerializedInferenceExecutor.h"

namespace llaminar2::test
{
    TEST(SerializedInferenceExecutor, QueueZeroAdmitsOnlyTheExecutionSlot)
    {
        SerializedInferenceExecutor executor(4, 0);
        auto first = executor.tryReserve();
        EXPECT_TRUE(first);
        // Q = 0: beyond the execution slot nothing may be accepted, even though
        // R = 4 would allow it.
        EXPECT_FALSE(executor.tryReserve());
        // The execution-slot occupant may still submit and run.
        first->run([]{});
        // Release ends the response lease; a new single response is admitted.
        first.reset();
        auto replacement = executor.tryReserve();
        EXPECT_TRUE(replacement);
        EXPECT_FALSE(executor.tryReserve());
        replacement->run([]{});
    }

    TEST(SerializedInferenceExecutor, QueueBoundLimitsAcceptedBeyondExecutionSlot)
    {
        SerializedInferenceExecutor executor(8, 1);
        auto slot = executor.tryReserve();
        EXPECT_TRUE(slot);
        auto extra = executor.tryReserve();
        EXPECT_TRUE(extra);
        EXPECT_FALSE(executor.tryReserve()); // Q = 1: only one beyond the slot
        slot->run([]{});
        extra->run([]{});
    }

    TEST(SerializedInferenceExecutor, ResponseLeaseTotalStillBoundsAdmission)
    {
        SerializedInferenceExecutor executor(2, 1);
        auto first = executor.tryReserve();
        EXPECT_TRUE(first);
        auto second = executor.tryReserve();
        EXPECT_TRUE(second);
        EXPECT_FALSE(executor.tryReserve()); // R = 2 reached
        first->run([]{});
        second->run([]{});
    }

    TEST(SerializedInferenceExecutor, QueueCapacityValidatesConstructor)
    {
        EXPECT_THROW((SerializedInferenceExecutor(8, 1025)), std::invalid_argument);
        EXPECT_THROW((SerializedInferenceExecutor(0, 0)), std::invalid_argument);
    }

    TEST(SerializedInferenceExecutor, DefaultsPreserveQualifiedPoolSemantics)
    {
        SerializedInferenceExecutor executor;
        EXPECT_EQ(executor.snapshot().capacity, SerializedInferenceExecutor::kDefaultCapacity);
        // Automatic envelope: R = 32, Q = R - 1 -> behavior identical to the
        // historical 32-total admission.
        std::vector<std::shared_ptr<SerializedInferenceExecutor::Reservation>> held;
        for (int i = 0; i < 32; ++i)
        {
            auto reservation = executor.tryReserve();
            EXPECT_TRUE(reservation) << i;
            held.push_back(std::move(reservation));
        }
        // 33rd is refused by the response-lease total.
        EXPECT_FALSE(executor.tryReserve());
    }
} // namespace llaminar2::test
