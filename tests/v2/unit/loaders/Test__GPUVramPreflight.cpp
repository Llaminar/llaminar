#include "loaders/GPUVramPreflight.h"
#include "loaders/GPUHostLoadPreflight.h"
#include "utils/DebugEnv.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

namespace llaminar2::test
{
    static constexpr size_t kMiB = 1024ULL * 1024ULL;

    namespace
    {
        class ScopedEnvironmentVariable
        {
          public:
            ScopedEnvironmentVariable(const char *name, const char *value)
                : name_(name)
            {
                if (const char *current = std::getenv(name))
                    previous_ = current;
                ::setenv(name_.c_str(), value, 1);
                mutableDebugEnv().reload();
            }

            ~ScopedEnvironmentVariable()
            {
                if (previous_.has_value())
                    ::setenv(name_.c_str(), previous_->c_str(), 1);
                else
                    ::unsetenv(name_.c_str());
                mutableDebugEnv().reload();
            }

            ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
            ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;

          private:
            std::string name_;
            std::optional<std::string> previous_;
        };

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

    TEST(Test__GPUVramPreflight, DebugEnvStagingBudgetIsPerGpuAndNeverDividedByDeviceCount)
    {
        const auto per_gpu_budget = gpuPerDeviceLoadStagingBudgetBytes(512);

        ASSERT_TRUE(per_gpu_budget.has_value());
        EXPECT_EQ(*per_gpu_budget, 512ULL * kMiB);

        // Two independently loading GPUs each receive the complete configured
        // ring. Device count is deliberately absent from the production helper,
        // so adding participants cannot silently shrink an individual pipeline.
        constexpr size_t gpu_count = 2;
        EXPECT_EQ(gpu_count * *per_gpu_budget, 1024ULL * kMiB);
    }

    TEST(Test__GPUVramPreflight, DebugEnvOverridesPerGpuStagingBudget)
    {
        ScopedEnvironmentVariable staging_budget(
            "LLAMINAR_GPU_LOAD_STAGING_MB", "768");

        EXPECT_EQ(debugEnv().rocm.repack_budget_mb, 768);
        const auto bytes = gpuPerDeviceLoadStagingBudgetBytes(
            debugEnv().rocm.repack_budget_mb);
        ASSERT_TRUE(bytes.has_value());
        EXPECT_EQ(*bytes, 768ULL * kMiB);
    }

    TEST(Test__GPUVramPreflight, NonPositiveDebugEnvStagingBudgetMeansUnlimited)
    {
        EXPECT_FALSE(gpuPerDeviceLoadStagingBudgetBytes(0).has_value());
        EXPECT_FALSE(gpuPerDeviceLoadStagingBudgetBytes(-1).has_value());
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
