/**
 * @file Test__ServingCapacityIntent.cpp
 * @brief Knob disambiguation: generation/benchmark knobs are NOT serving capacity.
 *
 * A-2c: only the three §9.4 axes (--serving-resident-slots,
 * --serving-response-leases, --serving-queue-capacity) express serving
 * capacity intent. Decode batch size, generation length, RNG seed and MTP
 * depth are generation knobs and must never be read as capacity intent, so no
 * future policy can overload them. Device-free.
 */

#include <gtest/gtest.h>

#include "config/OrchestrationConfig.h"

namespace llaminar2::test
{
    TEST(ServingCapacityIntent, GenerationKnobsAreNotServingCapacity)
    {
        OrchestrationConfig config;
        config.batch_size = 8;   // decode batch
        config.n_predict = 256;  // generation length
        config.seed = 42;        // RNG
        EXPECT_FALSE(config.hasServingCapacityIntent());
    }

    TEST(ServingCapacityIntent, OnlyTheThreeAxesExpressServingCapacity)
    {
        OrchestrationConfig config;
        config.serving_resident_slots = 1;
        EXPECT_TRUE(config.hasServingCapacityIntent());
        config = OrchestrationConfig{};
        config.serving_response_leases = 8;
        EXPECT_TRUE(config.hasServingCapacityIntent());
        config = OrchestrationConfig{};
        config.serving_queue_capacity = 4;
        EXPECT_TRUE(config.hasServingCapacityIntent());
    }
} // namespace llaminar2::test
