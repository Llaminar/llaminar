#include "loaders/GPUVramPreflight.h"
#include "loaders/GPUHostLoadPreflight.h"
#include "utils/DebugEnv.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
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

    TEST(Test__GPUVramPreflight,
         InitialLoadBomOwnsRingReserveAndExactFitEquation)
    {
        constexpr size_t kTotal = 24124ULL * kMiB;
        constexpr size_t kWeights = 16971ULL * kMiB;
        constexpr size_t kLargestSource = 2ULL * 1024ULL * kMiB;
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 3,
            .staging_budget_bytes = 512ULL * kMiB,
            .safety_margin_percent = 5,
            .minimum_safety_margin_bytes = 512ULL * kMiB,
        };
        const size_t expected_slot = (512ULL * kMiB) / 3ULL;
        const size_t expected_staging = expected_slot * 3ULL;
        const size_t expected_safety = (kTotal * 5ULL) / 100ULL;
        const size_t expected_required =
            kWeights + expected_staging + expected_safety;

        const auto exact = gpuWeightLoadMemoryBOM(
            kWeights,
            kLargestSource,
            expected_required,
            kTotal,
            policy);
        EXPECT_EQ(exact.staging_slot_bytes, expected_slot);
        EXPECT_EQ(exact.staging_bytes, expected_staging);
        EXPECT_EQ(exact.load_bytes, kWeights + expected_staging);
        EXPECT_EQ(exact.safety_margin_bytes, expected_safety);
        EXPECT_EQ(exact.required_bytes, expected_required);
        EXPECT_TRUE(exact.fits());

        const auto one_byte_short = gpuWeightLoadMemoryBOM(
            kWeights,
            kLargestSource,
            expected_required - 1,
            kTotal,
            policy);
        EXPECT_FALSE(one_byte_short.fits());
        EXPECT_EQ(
            one_byte_short.availableAfterSafetyReserve(),
            expected_required - 1 - expected_safety);
    }

    TEST(Test__GPUVramPreflight,
         UnlimitedPolicyPricesEveryPhysicalSlot)
    {
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 4,
            .staging_budget_bytes = 0,
            .safety_margin_override_bytes = 17,
        };
        const auto bill = gpuWeightLoadMemoryBOM(
            /*planned_weight_bytes=*/23,
            /*maximum_source_bytes=*/101,
            /*free_vram_bytes=*/1000,
            /*total_vram_bytes=*/2000,
            policy);

        EXPECT_EQ(bill.staging_slot_bytes, 101u);
        EXPECT_EQ(bill.staging_bytes, 404u);
        EXPECT_EQ(bill.safety_margin_bytes, 17u);
        EXPECT_EQ(bill.required_bytes, 444u);
        EXPECT_TRUE(bill.fits());
    }

    TEST(Test__GPUVramPreflight, BomArithmeticRejectsOverflow)
    {
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 2,
            .staging_budget_bytes = 0,
        };
        EXPECT_THROW(
            (void)gpuWeightLoadMemoryBOM(
                /*planned_weight_bytes=*/0,
                std::numeric_limits<size_t>::max(),
                /*free_vram_bytes=*/0,
                /*total_vram_bytes=*/0,
                policy),
            std::overflow_error);
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
