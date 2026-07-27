#include "loaders/GPUVramPreflight.h"
#include "loaders/GPUHostLoadPreflight.h"

#include <gtest/gtest.h>

#include <string>

namespace llaminar2::test
{
    static constexpr size_t kMiB = 1024ULL * 1024ULL;

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

    TEST(Test__GPUVramPreflight, GpuDirectRebalanceUsesSmallRuntimeMargin)
    {
        EXPECT_EQ(gpuDirectRebalanceVramSafetyMarginBytes(), 16ULL * kMiB);
    }

    TEST(Test__GPUVramPreflight, MappedGpuLoadUsesBoundedPinnedWorkingSet)
    {
        const size_t model_bytes = 8ULL * 1024ULL * kMiB;
        const size_t staging_bytes = 512ULL * kMiB;

        EXPECT_EQ(
            gpuHostLoadWorkingSetBytes(
                model_bytes,
                /*target_is_gpu=*/true,
                /*uses_mmap=*/true,
                staging_bytes),
            staging_bytes);
    }

    TEST(Test__GPUVramPreflight, BoundedRingDoesNotInflateSmallModels)
    {
        const size_t model_bytes = 128ULL * kMiB;
        const size_t staging_bytes = 512ULL * kMiB;

        EXPECT_EQ(
            gpuHostLoadWorkingSetBytes(
                model_bytes,
                /*target_is_gpu=*/true,
                /*uses_mmap=*/true,
                staging_bytes),
            model_bytes);
    }

    TEST(Test__GPUVramPreflight, CpuAndNonMappedLoadsRequireFullEagerBytes)
    {
        const size_t model_bytes = 8ULL * 1024ULL * kMiB;
        const size_t staging_bytes = 512ULL * kMiB;

        EXPECT_EQ(
            gpuHostLoadWorkingSetBytes(
                model_bytes,
                /*target_is_gpu=*/false,
                /*uses_mmap=*/true,
                staging_bytes),
            model_bytes);
        EXPECT_EQ(
            gpuHostLoadWorkingSetBytes(
                model_bytes,
                /*target_is_gpu=*/true,
                /*uses_mmap=*/false,
                staging_bytes),
            model_bytes);
    }

    TEST(Test__GPUVramPreflight, ZeroBudgetPreservesExplicitUnlimitedMode)
    {
        const size_t model_bytes = 8ULL * 1024ULL * kMiB;

        EXPECT_EQ(
            gpuHostLoadWorkingSetBytes(
                model_bytes,
                /*target_is_gpu=*/true,
                /*uses_mmap=*/true,
                /*staging_budget_bytes=*/0),
            model_bytes);
    }
} // namespace llaminar2::test
