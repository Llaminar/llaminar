/**
 * @file Test__OrchestrationInitializationLifecycle.cpp
 * @brief Device-free regression for admission-only and prepared runner retirement.
 *
 * A real hybrid dry run admitted its complete memory plan and then attempted
 * to seal nonexistent prepared weights. These tests exercise the production
 * completion authority, including illegal promotion, repeated publication and
 * shutdown/reinitialization, without loading a model or opening a GPU context.
 */
#include "execution/runner/OrchestrationInitializationLifecycle.h"

#include <gtest/gtest.h>

namespace llaminar2::test
{
    using Lifecycle = OrchestrationInitializationLifecycle;
    using Completion = Lifecycle::Completion;

    /** @brief A plan can be admitted without granting graph or weight-retention access. */
    TEST(Test__OrchestrationInitializationLifecycle, DryRunNeverBecomesInferenceReady)
    {
        Lifecycle lifecycle;
        EXPECT_FALSE(lifecycle.admissionComplete());
        EXPECT_FALSE(lifecycle.readyForInference());
        lifecycle.publish(Completion::Admission);
        EXPECT_EQ(lifecycle.completion(), Completion::Admission);
        EXPECT_TRUE(lifecycle.admissionComplete());
        EXPECT_FALSE(lifecycle.readyForInference());
        lifecycle.retire();
        EXPECT_EQ(lifecycle.completion(), Completion::None);
    }

    /** @brief Retained admission cannot impersonate successful graph preparation. */
    TEST(Test__OrchestrationInitializationLifecycle, AdmissionRejectsPromotionAndDuplicatePublication)
    {
        Lifecycle lifecycle;
        lifecycle.publish(Completion::Admission);
        for (const auto next : {Completion::Admission, Completion::Inference})
        {
            EXPECT_THROW(lifecycle.publish(next), std::logic_error);
            EXPECT_EQ(lifecycle.completion(), Completion::Admission);
            EXPECT_FALSE(lifecycle.readyForInference());
        }
    }

    /** @brief Prepared inference stays distinct and cannot be downgraded by another initializer. */
    TEST(Test__OrchestrationInitializationLifecycle, InferenceOwnsPreparedRetirement)
    {
        Lifecycle lifecycle;
        lifecycle.publish(Completion::Inference);
        EXPECT_TRUE(lifecycle.admissionComplete());
        EXPECT_TRUE(lifecycle.readyForInference());
        for (const auto next : {Completion::Admission, Completion::Inference})
        {
            EXPECT_THROW(lifecycle.publish(next), std::logic_error);
            EXPECT_EQ(lifecycle.completion(), Completion::Inference);
        }
    }

    /** @brief Only full retirement allows a different preparation kind to begin. */
    TEST(Test__OrchestrationInitializationLifecycle, ShutdownSeparatesRepeatedLifetimes)
    {
        Lifecycle lifecycle;
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            lifecycle.publish(Completion::Admission);
            EXPECT_FALSE(lifecycle.readyForInference());
            lifecycle.retire();
            lifecycle.retire();
            lifecycle.publish(Completion::Inference);
            EXPECT_TRUE(lifecycle.readyForInference());
            lifecycle.retire();
            EXPECT_FALSE(lifecycle.admissionComplete());
        }
    }

    /** @brief Invalid publications cannot change the state or manufacture a ready runner. */
    TEST(Test__OrchestrationInitializationLifecycle, RejectsUnknownAndAbsentCompletions)
    {
        Lifecycle lifecycle;
        for (const auto invalid : {Completion::None, static_cast<Completion>(99)})
        {
            EXPECT_THROW(lifecycle.publish(invalid), std::invalid_argument);
            EXPECT_EQ(lifecycle.completion(), Completion::None);
        }
        EXPECT_THROW((Lifecycle{static_cast<Completion>(99)}), std::invalid_argument);
        Lifecycle injected(Completion::Inference);
        EXPECT_TRUE(injected.readyForInference());
        injected.retire();
        EXPECT_FALSE(injected.readyForInference());
    }
}
