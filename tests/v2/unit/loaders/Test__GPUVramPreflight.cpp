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

    TEST(Test__GPUVramPreflight,
         InitialLoadBomOwnsEveryRingSlotAndExactFitEquation)
    {
        constexpr size_t kWeights = 16971ULL * kMiB;
        constexpr size_t kLargestSource = 2ULL * 1024ULL * kMiB;
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 3,
            .staging_budget_bytes = 512ULL * kMiB,
        };
        const size_t expected_slot =
            ((512ULL * kMiB) / 3ULL) &
            ~(kGPUWeightLoadAllocationAlignment - 1u);
        const size_t expected_staging = expected_slot * 3ULL;
        const size_t expected_required = kWeights + expected_staging;
        const auto geometry = resolveGPUWeightLoadMemoryGeometry(
            kLargestSource, policy);

        const auto exact = gpuWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = -1,
                .device = DeviceId::cuda(0),
                .total_bytes = expected_required,
                .admission_available_bytes = expected_required,
            },
            kWeights,
            geometry);
        EXPECT_EQ(geometry.staging_slot_bytes, expected_slot);
        EXPECT_EQ(geometry.staging_slot_stride_bytes, expected_slot);
        EXPECT_EQ(
            exact.bytes(PhysicalMemoryOwner::WeightLoadStaging),
            expected_staging);
        EXPECT_EQ(exact.incrementalBytes(), expected_required);
        EXPECT_TRUE(exact.fits());

        const auto one_byte_short = gpuWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = -1,
                .device = DeviceId::cuda(0),
                .total_bytes = expected_required,
                .admission_available_bytes = expected_required - 1u,
            },
            kWeights,
            geometry);
        EXPECT_FALSE(one_byte_short.fits());
    }

    TEST(Test__GPUVramPreflight,
         UnlimitedPolicyPricesEveryPhysicalSlot)
    {
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 4,
            .staging_budget_bytes = 0,
        };
        const auto geometry = resolveGPUWeightLoadMemoryGeometry(
            /*maximum_source_bytes=*/101, policy);
        const auto bill = gpuWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = -1,
                .device = DeviceId::rocm(0),
                .total_bytes = 1280,
                .admission_available_bytes = 1280,
            },
            /*planned_weight_bytes=*/23,
            geometry);

        EXPECT_EQ(geometry.staging_slot_bytes, 101u);
        EXPECT_EQ(geometry.staging_slot_stride_bytes, 256u);
        EXPECT_EQ(
            bill.bytes(PhysicalMemoryOwner::WeightLoadStaging), 1024u);
        EXPECT_EQ(bill.incrementalBytes(), 1280u);
        EXPECT_TRUE(bill.fits());
    }

    TEST(Test__GPUVramPreflight, BomArithmeticRejectsOverflow)
    {
        const GPUWeightLoadMemoryPolicy policy{
            .staging_stream_count = 2,
            .staging_budget_bytes = 0,
        };
        EXPECT_THROW(
            (void)resolveGPUWeightLoadMemoryGeometry(
                std::numeric_limits<size_t>::max(),
                policy),
            std::overflow_error);
    }

    TEST(Test__GPUVramPreflight, MappedGpuLoadUsesBoundedPinnedWorkingSet)
    {
        const auto geometry = resolveGPUWeightLoadMemoryGeometry(
            /*maximum_source_bytes=*/2ULL * 1024ULL * kMiB,
            GPUWeightLoadMemoryPolicy{
                .staging_stream_count = 3,
                .staging_budget_bytes = 512ULL * kMiB,
            });
        const auto bom = hostWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = 16ULL * 1024ULL * kMiB,
                .admission_available_bytes =
                    16ULL * 1024ULL * kMiB,
            },
            /*eager_weight_bytes=*/8ULL * 1024ULL * kMiB,
            /*target_is_gpu=*/true,
            /*uses_mmap=*/true,
            geometry.host_staging_bytes);

        EXPECT_EQ(
            bom.bytes(PhysicalMemoryOwner::WeightLoadStaging),
            geometry.host_staging_bytes);
        EXPECT_EQ(
            bom.bytes(PhysicalMemoryOwner::ModelSourcePayload), 0u);
        EXPECT_LE(geometry.host_staging_bytes, 512ULL * kMiB);
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

    TEST(Test__GPUVramPreflight, EveryPinnedSlotIsChargedForSmallModels)
    {
        const auto geometry = resolveGPUWeightLoadMemoryGeometry(
            /*maximum_source_bytes=*/128ULL * kMiB,
            GPUWeightLoadMemoryPolicy{
                .staging_stream_count = 3,
                .staging_budget_bytes = 512ULL * kMiB,
            });
        EXPECT_EQ(geometry.host_staging_bytes, 384ULL * kMiB);
    }

    TEST(Test__GPUVramPreflight,
         CpuAndNonMappedGpuLoadsChargeDistinctPhysicalOwners)
    {
        const size_t model_bytes = 8ULL * 1024ULL * kMiB;
        constexpr size_t kPinned = 512ULL * kMiB;
        const PhysicalMemoryResource host{
            .world_rank = 2,
            .device = DeviceId::cpu(),
            .total_bytes = 32ULL * 1024ULL * kMiB,
            .admission_available_bytes = 32ULL * 1024ULL * kMiB,
        };

        const auto cpu = hostWeightLoadMemoryBOM(
            host,
            model_bytes,
            /*target_is_gpu=*/false,
            /*uses_mmap=*/true,
            /*pinned_ring_bytes=*/0);
        EXPECT_EQ(
            cpu.bytes(PhysicalMemoryOwner::PrimaryModelWeights),
            model_bytes);
        EXPECT_EQ(cpu.incrementalBytes(), model_bytes);

        const auto gpu = hostWeightLoadMemoryBOM(
            host,
            model_bytes,
            /*target_is_gpu=*/true,
            /*uses_mmap=*/false,
            kPinned);
        EXPECT_EQ(
            gpu.bytes(PhysicalMemoryOwner::ModelSourcePayload),
            model_bytes);
        EXPECT_EQ(
            gpu.bytes(PhysicalMemoryOwner::WeightLoadStaging), kPinned);
        EXPECT_EQ(gpu.incrementalBytes(), model_bytes + kPinned);
    }

    TEST(Test__GPUVramPreflight,
         UnlimitedModeStillChargesEveryConcretePinnedSlot)
    {
        const auto geometry = resolveGPUWeightLoadMemoryGeometry(
            /*maximum_source_bytes=*/2ULL * 1024ULL * kMiB,
            GPUWeightLoadMemoryPolicy{
                .staging_stream_count = 3,
                .staging_budget_bytes = 0,
            });
        EXPECT_EQ(
            geometry.host_staging_bytes,
            6ULL * 1024ULL * kMiB);
    }
} // namespace llaminar2::test
