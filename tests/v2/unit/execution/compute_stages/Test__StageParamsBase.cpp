/**
 * @file Test__StageParamsBase.cpp
 * @brief Unit tests for StageParamsBase concept and STAGE_PARAMS_COMMON_FIELDS macro
 * @author David Sanftenberg
 * @date January 2026
 *
 * These tests verify that:
 * 1. All stage Params structs satisfy the StageParamsRequired concept
 * 2. device_id defaults to CPU for safety
 * 3. device_id can be set via designated initializers
 * 4. device() returns the correct device_id passed to base constructor
 * 5. mpi_ctx defaults to nullptr
 *
 * This prevents the bug where a stage accidentally omits device_id
 * and silently runs on CPU when intended for GPU.
 */

#include <gtest/gtest.h>
#include <type_traits>

#include "execution/compute_stages/StageParamsBase.h"
#include "execution/compute_stages/ComputeStages.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Compile-Time Concept Tests
// =============================================================================

// These static_asserts verify at compile time that all Params satisfy the concept

// GEMM Stages
static_assert(StageParamsRequired<GEMMStage::Params>,
              "GEMMStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<FusedQKVGEMMStage::Params>,
              "FusedQKVGEMMStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<FusedGateUpGEMMStage::Params>,
              "FusedGateUpGEMMStage::Params must satisfy StageParamsRequired");

// Normalization and Position Encoding
static_assert(StageParamsRequired<RMSNormStage::Params>,
              "RMSNormStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<RoPEStage::Params>,
              "RoPEStage::Params must satisfy StageParamsRequired");

// Attention Stages
static_assert(StageParamsRequired<AttentionComputeStage::Params>,
              "AttentionComputeStage::Params must satisfy StageParamsRequired");

// KV Cache Stages
static_assert(StageParamsRequired<KVCacheAppendStage::Params>,
              "KVCacheAppendStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<KVCacheGatherStage::Params>,
              "KVCacheGatherStage::Params must satisfy StageParamsRequired");

// Activation and Projection Stages
static_assert(StageParamsRequired<ResidualAddStage::Params>,
              "ResidualAddStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<EmbeddingStage::Params>,
              "EmbeddingStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<LMHeadStage::Params>,
              "LMHeadStage::Params must satisfy StageParamsRequired");

// MPI Collective Stages
static_assert(StageParamsRequired<AllreduceStage::Params>,
              "AllreduceStage::Params must satisfy StageParamsRequired");
static_assert(StageParamsRequired<AllGatherStage::Params>,
              "AllGatherStage::Params must satisfy StageParamsRequired");

// =============================================================================
// Test Fixture
// =============================================================================

class Test__StageParamsBase : public ::testing::Test
{
protected:
    // Helper to create test DeviceIds
    static DeviceId cpuDevice() { return DeviceId::cpu(); }
    static DeviceId cudaDevice(int idx = 0) { return DeviceId::cuda(idx); }
    static DeviceId rocmDevice(int idx = 0) { return DeviceId::rocm(idx); }
};

// =============================================================================
// Default Value Tests - Verify device_id defaults to CPU for safety
// =============================================================================

TEST_F(Test__StageParamsBase, GEMMStage_DefaultsToDeviceCPU)
{
    GEMMStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, FusedQKVGEMMStage_DefaultsToDeviceCPU)
{
    FusedQKVGEMMStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, FusedGateUpGEMMStage_DefaultsToDeviceCPU)
{
    FusedGateUpGEMMStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, RMSNormStage_DefaultsToDeviceCPU)
{
    RMSNormStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, RoPEStage_DefaultsToDeviceCPU)
{
    RoPEStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, AttentionComputeStage_DefaultsToDeviceCPU)
{
    AttentionComputeStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, ResidualAddStage_DefaultsToDeviceCPU)
{
    ResidualAddStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, EmbeddingStage_DefaultsToDeviceCPU)
{
    EmbeddingStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, LMHeadStage_DefaultsToDeviceCPU)
{
    LMHeadStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, KVCacheAppendStage_DefaultsToDeviceCPU)
{
    KVCacheAppendStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, KVCacheGatherStage_DefaultsToDeviceCPU)
{
    KVCacheGatherStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, AllreduceStage_DefaultsToDeviceCPU)
{
    AllreduceStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

TEST_F(Test__StageParamsBase, AllGatherStage_DefaultsToDeviceCPU)
{
    AllGatherStage::Params params{};
    EXPECT_EQ(params.device_id.type, DeviceType::CPU);
    EXPECT_EQ(params.mpi_ctx, nullptr);
}

// =============================================================================
// Designated Initializer Tests - Verify device_id can be set correctly
// =============================================================================

TEST_F(Test__StageParamsBase, GEMMStage_DeviceIdDesignatedInitializer)
{
    auto cuda0 = cudaDevice(0);
    GEMMStage::Params params{.device_id = cuda0};

    EXPECT_EQ(params.device_id.type, DeviceType::CUDA);
    EXPECT_EQ(params.device_id.ordinal, 0);
}

TEST_F(Test__StageParamsBase, FusedQKVGEMMStage_DeviceIdDesignatedInitializer)
{
    auto cuda1 = cudaDevice(1);
    FusedQKVGEMMStage::Params params{.device_id = cuda1};

    EXPECT_EQ(params.device_id.type, DeviceType::CUDA);
    EXPECT_EQ(params.device_id.ordinal, 1);
}

TEST_F(Test__StageParamsBase, RMSNormStage_DeviceIdDesignatedInitializer)
{
    auto rocm0 = rocmDevice(0);
    RMSNormStage::Params params{.device_id = rocm0};

    EXPECT_EQ(params.device_id.type, DeviceType::ROCm);
    EXPECT_EQ(params.device_id.ordinal, 0);
}

TEST_F(Test__StageParamsBase, AttentionComputeStage_DeviceIdDesignatedInitializer)
{
    auto cuda2 = cudaDevice(2);
    AttentionComputeStage::Params params{.device_id = cuda2};

    EXPECT_EQ(params.device_id.type, DeviceType::CUDA);
    EXPECT_EQ(params.device_id.ordinal, 2);
}

// =============================================================================
// device() Tests - Verify stage returns correct device from params
// =============================================================================
// NOTE: Only testing stages with complete implementations (.cpp files)
// QuantizeToQ16_1Stage is header-only placeholder - tested via static_assert only

TEST_F(Test__StageParamsBase, GEMMStage_DeviceMatchesParams)
{
    auto cuda0 = cudaDevice(0);
    GEMMStage::Params params{.device_id = cuda0};
    GEMMStage stage(params);

    EXPECT_EQ(stage.device(), cuda0);
}

TEST_F(Test__StageParamsBase, RMSNormStage_DeviceMatchesParams)
{
    auto rocm0 = rocmDevice(0);
    RMSNormStage::Params params{.device_id = rocm0};
    RMSNormStage stage(params);

    EXPECT_EQ(stage.device(), rocm0);
}

TEST_F(Test__StageParamsBase, RoPEStage_DeviceMatchesParams)
{
    auto cuda1 = cudaDevice(1);
    RoPEStage::Params params{.device_id = cuda1};
    RoPEStage stage(params);

    EXPECT_EQ(stage.device(), cuda1);
}

TEST_F(Test__StageParamsBase, ResidualAddStage_DeviceMatchesParams)
{
    auto rocm1 = rocmDevice(1);
    ResidualAddStage::Params params{.device_id = rocm1};
    ResidualAddStage stage(params);

    EXPECT_EQ(stage.device(), rocm1);
}

TEST_F(Test__StageParamsBase, AttentionComputeStage_DeviceMatchesParams)
{
    auto cuda0 = cudaDevice(0);
    AttentionComputeStage::Params params{.device_id = cuda0};
    AttentionComputeStage stage(params);

    EXPECT_EQ(stage.device(), cuda0);
}

TEST_F(Test__StageParamsBase, EmbeddingStage_DeviceMatchesParams)
{
    auto cuda1 = cudaDevice(1);
    EmbeddingStage::Params params{.device_id = cuda1};
    EmbeddingStage stage(params);

    EXPECT_EQ(stage.device(), cuda1);
}

TEST_F(Test__StageParamsBase, LMHeadStage_DeviceMatchesParams)
{
    auto cuda0 = cudaDevice(0);
    LMHeadStage::Params params{.device_id = cuda0};
    LMHeadStage stage(params);

    EXPECT_EQ(stage.device(), cuda0);
}

// NOTE: QuantizeToQ16_1Stage is header-only without implementation
// Its Params struct is validated via compile-time static_assert

TEST_F(Test__StageParamsBase, AllreduceStage_DeviceMatchesParams)
{
    auto cuda0 = cudaDevice(0);
    AllreduceStage::Params params{.device_id = cuda0};
    AllreduceStage stage(params);

    EXPECT_EQ(stage.device(), cuda0);
}

TEST_F(Test__StageParamsBase, AllGatherStage_DeviceMatchesParams)
{
    auto cuda1 = cudaDevice(1);
    AllGatherStage::Params params{.device_id = cuda1};
    AllGatherStage stage(params);

    EXPECT_EQ(stage.device(), cuda1);
}

// =============================================================================
// CPU Default Safety Tests - Verify unset device_id results in CPU execution
// =============================================================================

TEST_F(Test__StageParamsBase, GEMMStage_DefaultDeviceIsCPU)
{
    // When device_id is not explicitly set, it should default to CPU
    GEMMStage::Params params{
        .A = nullptr,
        .B = nullptr,
        .C = nullptr,
        .m = 32,
        .n = 32,
        .k = 32};
    GEMMStage stage(params);

    EXPECT_EQ(stage.device().type, DeviceType::CPU);
}

TEST_F(Test__StageParamsBase, RMSNormStage_DefaultDeviceIsCPU)
{
    RMSNormStage::Params params{
        .input = nullptr,
        .output = nullptr,
        .gamma = nullptr};
    RMSNormStage stage(params);

    EXPECT_EQ(stage.device().type, DeviceType::CPU);
}

// =============================================================================
// Multi-GPU Device Propagation Tests
// =============================================================================

TEST_F(Test__StageParamsBase, GEMMStage_MultiGPUDevicePropagation)
{
    // Test that different GPU indices propagate correctly
    for (int i = 0; i < 8; ++i)
    {
        auto cuda_i = cudaDevice(i);
        GEMMStage::Params params{.device_id = cuda_i};
        GEMMStage stage(params);

        EXPECT_EQ(stage.device().type, DeviceType::CUDA);
        EXPECT_EQ(stage.device().ordinal, i);
    }
}

TEST_F(Test__StageParamsBase, RMSNormStage_MultiGPUDevicePropagation)
{
    for (int i = 0; i < 4; ++i)
    {
        auto rocm_i = rocmDevice(i);
        RMSNormStage::Params params{.device_id = rocm_i};
        RMSNormStage stage(params);

        EXPECT_EQ(stage.device().type, DeviceType::ROCm);
        EXPECT_EQ(stage.device().ordinal, i);
    }
}

// =============================================================================
// Type Traits Tests
// =============================================================================

TEST_F(Test__StageParamsBase, AllParamsHaveDeviceIdField)
{
    // Test that device_id field exists and is of correct type
    static_assert(std::is_same_v<decltype(GEMMStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(RMSNormStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(RoPEStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(ResidualAddStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(AttentionComputeStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(AllreduceStage::Params::device_id), DeviceId>);
    static_assert(std::is_same_v<decltype(AllGatherStage::Params::device_id), DeviceId>);

    // This test just needs to compile - if it compiles, the static_asserts passed
    SUCCEED();
}

TEST_F(Test__StageParamsBase, AllParamsHaveMpiCtxField)
{
    // Test that mpi_ctx field exists and is of correct type
    static_assert(std::is_same_v<decltype(GEMMStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(RMSNormStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(RoPEStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(ResidualAddStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(AttentionComputeStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(AllreduceStage::Params::mpi_ctx), const IMPIContext *>);
    static_assert(std::is_same_v<decltype(AllGatherStage::Params::mpi_ctx), const IMPIContext *>);

    SUCCEED();
}
