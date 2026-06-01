#include "loaders/GPUVramPreflight.h"

#include <gtest/gtest.h>

#include <string>

namespace llaminar2::test
{
    namespace
    {
        bool contains(const std::string &text, const std::string &needle)
        {
            return text.find(needle) != std::string::npos;
        }
    } // namespace

    TEST(Test__GPUVramPreflight, DenseResidentWeightsCanRecommendStreamingWhenDisabled)
    {
        const std::string message =
            gpuPipelineVramPreflightMitigations(
                /*weight_streaming_enabled=*/false,
                /*includes_resident_moe_experts=*/false);

        EXPECT_TRUE(contains(message, "set LLAMINAR_WEIGHT_STREAMING=1"))
            << message;
        EXPECT_TRUE(contains(message, "reduce context/KV cache pressure"))
            << message;
    }

    TEST(Test__GPUVramPreflight, ResidentMoEExpertsAvoidUnsupportedStreamingSuggestion)
    {
        const std::string message =
            gpuPipelineVramPreflightMitigations(
                /*weight_streaming_enabled=*/false,
                /*includes_resident_moe_experts=*/true);

        EXPECT_FALSE(contains(message, "set LLAMINAR_WEIGHT_STREAMING=1"))
            << message;
        EXPECT_TRUE(contains(message, "reduce resident experts"))
            << message;
    }

    TEST(Test__GPUVramPreflight, StreamingEnabledResidentMoEReportsAlreadyEnabled)
    {
        const std::string message =
            gpuPipelineVramPreflightMitigations(
                /*weight_streaming_enabled=*/true,
                /*includes_resident_moe_experts=*/true);

        EXPECT_TRUE(contains(message, "LLAMINAR_WEIGHT_STREAMING=1 is already enabled"))
            << message;
        EXPECT_TRUE(contains(message, "resident MoE expert streaming is not active"))
            << message;
        EXPECT_FALSE(contains(message, "set LLAMINAR_WEIGHT_STREAMING=1"))
            << message;
    }
} // namespace llaminar2::test
