/**
 * @file Test__TPAllreduceStage.cpp
 * @brief Unit tests for TPAllreduceStage (unified LOCAL + GLOBAL TP allreduce)
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for TP all-reduce stage supporting both LOCAL and GLOBAL TP:
 * - Stage construction and properties
 * - Stage type and name
 * - Buffer requirements
 * - Parameter validation
 * - Single-device no-op behavior
 * - LOCAL TP context tests
 *
 * Note: Global TP tests requiring MPI are in the integration test:
 *       tests/v2/integration/execution/compute_stages/stages/Test__TPAllreduceStage_MPI.cpp
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "collective/LocalTPContext.h"
#include "collective/ITPContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../mocks/MockComputeStage.h"
#include "../../../../mocks/MockLocalTPContext.h"

using namespace llaminar2;
using namespace llaminar2::testing;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
                old_value_ = std::string(old);
            ::setenv(name_.c_str(), value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (old_value_)
                ::setenv(name_.c_str(), old_value_->c_str(), 1);
            else
                ::unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::optional<std::string> old_value_;
    };
} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class Test__TPAllreduceStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices for LOCAL TP
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);

        // Create MPI context (mock - no real MPI needed for unit tests)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

        // Create device context
        ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        // Create tensor factory
        TensorFactory factory(*mpi_ctx_);
        test_tensor_ = factory.createFP32({4, 128}, DeviceId::cpu());
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    std::shared_ptr<IMPIContext> mpi_ctx_;
    std::unique_ptr<MockDeviceContext> ctx_;
    std::unique_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// Stage Type and Properties Tests
// =============================================================================

/**
 * @test Stage type is ALLREDUCE
 */
TEST_F(Test__TPAllreduceStage, StageTypeIsAllreduce)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::ALLREDUCE);
}

/**
 * @test Stage name is TPAllreduce
 */
TEST_F(Test__TPAllreduceStage, StageNameCorrect)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->name(), "TPAllreduce");
}

/**
 * @test Stage requires allreduce
 */
TEST_F(Test__TPAllreduceStage, RequiresAllreduce)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_TRUE(stage->requiresAllreduce());
}

/**
 * @test Coherence policy is OUTPUT
 *
 * Collective stages use OUTPUT policy so the executor marks outputs dirty
 * (flags only) after execution. The stage itself records a completion event
 * via mark_device_dirty_with_event() for fine-grained synchronization.
 */
TEST_F(Test__TPAllreduceStage, CoherencePolicyIsOutput)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::OUTPUT);
}

/**
 * @test Stage supports all backends
 */
TEST_F(Test__TPAllreduceStage, SupportsAllBackends)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
}

// =============================================================================
// Accessor Tests
// =============================================================================

/**
 * @test getTPContext returns correct LOCAL context
 */
TEST_F(Test__TPAllreduceStage, GetTPContextReturnsLocalContext)
{
    // Use HOST backend for unit tests - no GPU initialization needed
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->getTPContext(), tp_ctx.get());
    EXPECT_EQ(stage->getTPContext()->degree(), 2);
    EXPECT_FALSE(stage->getTPContext()->isGlobal());
}

/**
 * @test getTensor returns correct tensor
 */
TEST_F(Test__TPAllreduceStage, GetTensorReturnsCorrect)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->getTensor(), tensor);
}

// =============================================================================
// ITPContext Interface Polymorphism Tests
// =============================================================================

/**
 * @test Stage works with ITPContext* pointing to LocalTPContext
 */
TEST_F(Test__TPAllreduceStage, WorksWithLocalTPContextViaInterface)
{
    auto local_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ITPContext *tp_ctx = local_ctx.get(); // Upcast to base interface

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx;
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_FALSE(stage->getTPContext()->isGlobal());
    // Note: AUTO gets resolved to NCCL for CUDA devices
    EXPECT_NE(stage->getTPContext()->backend(), CollectiveBackendType::UPI);

    // Single device no-op should succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));
}

// =============================================================================
// Buffer Requirements Tests
// =============================================================================

/**
 * @test Buffer requirements with valid tensor
 */
TEST_F(Test__TPAllreduceStage, BufferRequirementsWithTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    auto reqs = stage->getBufferRequirements();

    ASSERT_GE(reqs.buffers.size(), 1);
    EXPECT_EQ(reqs.buffers[0].role, BufferRole::INOUT);
    ASSERT_GE(reqs.buffers[0].shape.size(), 2);
    EXPECT_EQ(reqs.buffers[0].shape[0], 4);   // rows
    EXPECT_EQ(reqs.buffers[0].shape[1], 128); // cols
}

/**
 * @test Buffer requirements with null tensor
 */
TEST_F(Test__TPAllreduceStage, BufferRequirementsWithNullTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    auto reqs = stage->getBufferRequirements();

    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// Execute Tests
// =============================================================================

/**
 * @test Execute fails with null tp_ctx
 */
TEST_F(Test__TPAllreduceStage, ExecuteFailsWithNullContext)
{
    TPAllreduceStage::Params params;
    params.tp_ctx = nullptr;
    params.tensor = test_tensor_.get();

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute fails with null tensor
 */
TEST_F(Test__TPAllreduceStage, ExecuteFailsWithNullTensor)
{
    // Use HOST backend for unit tests - no GPU initialization needed
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Execute succeeds for single device LOCAL TP (no-op)
 */
TEST_F(Test__TPAllreduceStage, ExecuteSucceedsSingleDeviceLocalTP)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    // Initialize tensor with known values
    auto *tensor = test_tensor_.get();
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    // Single device - should be no-op and succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));

    // Data should be unchanged
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

TEST_F(Test__TPAllreduceStage, GpuAllreduceFailsFastWithoutExplicitStream)
{
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;
    params.count = tensor->numel();
    params.stage_name = "layer0_wo_allreduce";
    params.precision = "fp16";

    TPAllreduceStage stage(params);
    MockDeviceContext cuda_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    EXPECT_FALSE(stage.execute(&cuda_ctx));
}

TEST_F(Test__TPAllreduceStage, EmptyPrecisionResolvesBeforeOnStreamCollective)
{
    llaminar2::test::MockLocalTPContext tp_ctx;
    tp_ctx.setDevices({cuda0_, cuda1_});
    tp_ctx.setBackend(CollectiveBackendType::HOST);

    auto *tensor = test_tensor_.get();
    TPAllreduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.tp_ctx = &tp_ctx;
    params.tensor = tensor;
    params.count = 256;
    params.stage_name = "embedding_allreduce";
    params.precision = "";

    TPAllreduceStage stage(params);
    void *explicit_stream = reinterpret_cast<void *>(0x1234);
    stage.setGPUStream(explicit_stream);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    const auto calls = tp_ctx.getAllreduceCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls.front().stage_name, "embedding_allreduce");
    EXPECT_EQ(calls.front().count, 256u);
    EXPECT_EQ(calls.front().stream, explicit_stream);
    EXPECT_EQ(calls.front().precision, "fp32");
}

// =============================================================================
// Dump Info Tests
// =============================================================================

// Note: The test "ExecuteLocalMultiDeviceFailsWhenTensorNotOnTPDevice" has been
// moved to integration tests (Test__TPAllreduceStage_LocalNCCL.cpp) because it
// requires NCCL backend initialization which touches real GPUs.

/**
 * @test buildDumpInfoImpl includes tensor info
 */
TEST_F(Test__TPAllreduceStage, DumpInfoIncludesTensor)
{
    // Use HOST backend for unit tests - no GPU initialization needed
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tensor as both input and output (in-place)
    EXPECT_GE(dump_info.inputs.size(), 1);
    EXPECT_GE(dump_info.outputs.size(), 1);
}

/**
 * @test buildDumpInfoImpl includes scalars for LOCAL TP
 */
TEST_F(Test__TPAllreduceStage, DumpInfoIncludesScalarsLocalTP)
{
    // Use HOST backend for unit tests - no GPU initialization needed
    auto tp_ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);
    auto *tensor = test_tensor_.get();

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    auto dump_info = stage->buildDumpInfoImpl();

    // Should have tp_degree, backend, and tp_scope scalars
    EXPECT_GE(dump_info.scalars.size(), 3);

    // Find tp_scope scalar and verify it's LOCAL (0)
    bool found_tp_scope = false;
    for (const auto &scalar : dump_info.scalars)
    {
        if (scalar.name == std::string_view("tp_scope"))
        {
            found_tp_scope = true;
            EXPECT_EQ(static_cast<int>(scalar.value), static_cast<int>(TPScope::LOCAL));
        }
    }
    EXPECT_TRUE(found_tp_scope);
}

// =============================================================================
// setParams Tests
// =============================================================================

/**
 * @test setParams updates tensor
 */
TEST_F(Test__TPAllreduceStage, SetParamsUpdatesTensor)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = nullptr;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    EXPECT_EQ(stage->getTensor(), nullptr);

    // Update params with a tensor
    auto *tensor = test_tensor_.get();
    params.tensor = tensor;
    stage->setParams(params);

    EXPECT_EQ(stage->getTensor(), tensor);
}

// =============================================================================
// Custom Count Tests
// =============================================================================

/**
 * @test Custom count parameter is respected
 */
TEST_F(Test__TPAllreduceStage, CustomCountIsRespected)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    auto *tensor = test_tensor_.get();

    // Tensor has 4 * 128 = 512 elements, but we only want to reduce 256
    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = tensor;
    params.count = 256;

    auto stage = std::make_unique<TPAllreduceStage>(params);

    // Single device - no-op, but count should still be set correctly
    // We can verify via dump info
    auto dump_info = stage->buildDumpInfoImpl();

    // Input/output rows should be calculated based on custom count
    // With 128 cols and 256 count, rows = 256 / 128 = 2
    ASSERT_GE(dump_info.inputs.size(), 1);
    EXPECT_EQ(dump_info.inputs[0].rows, 2); // 256 / 128 cols
}

/**
 * @test Stage name parameter is stored
 */
TEST_F(Test__TPAllreduceStage, StageNameParameterStored)
{
    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();
    params.stage_name = "layer0_wo_proj_allreduce";

    auto stage = std::make_unique<TPAllreduceStage>(params);

    // Single device execute should succeed
    EXPECT_TRUE(stage->execute(ctx_.get()));
}

/**
 * @test Execute records a per-buffer bill of materials entry when perf stats are enabled
 */
TEST_F(Test__TPAllreduceStage, RecordsBillOfMaterialsForMoERoutedAllreduce)
{
    ScopedEnv enable_perf_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();
    params.count = 256;
    params.stage_name = "layer0_moe_expert_overlay_fast_allreduce";
    params.precision = "fp16";

    auto stage = std::make_unique<TPAllreduceStage>(params);

    ASSERT_TRUE(stage->execute(ctx_.get()));

    const auto records = PerfStatsCollector::snapshot({"tp_allreduce_bom"});
    const PerfStatRecord *bytes = nullptr;
    const PerfStatRecord *stages = nullptr;
    for (const auto &record : records)
    {
        if (record.name == "bytes")
            bytes = &record;
        else if (record.name == "stages")
            stages = &record;
    }

    ASSERT_NE(bytes, nullptr);
    EXPECT_EQ(bytes->domain, "tp_allreduce_bom");
    EXPECT_DOUBLE_EQ(bytes->value, 256.0 * sizeof(float));
    EXPECT_EQ(bytes->tags.at("role"), "moe_routed_expert");
    EXPECT_EQ(bytes->tags.at("stage"), "layer0_moe_expert_overlay_fast_allreduce");
    EXPECT_EQ(bytes->tags.at("elements"), "256");
    EXPECT_EQ(bytes->tags.at("element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("tensor_element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("tensor_numel"), std::to_string(test_tensor_->numel()));
    EXPECT_EQ(bytes->tags.at("tensor_type"), "FP32");
    EXPECT_EQ(bytes->tags.at("precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("requested_transport_precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("transport_precision"), "fp32");
    EXPECT_EQ(bytes->tags.at("scope"), "local");
    EXPECT_EQ(bytes->tags.at("degree"), "1");
    EXPECT_EQ(bytes->tags.at("no_op"), "true");

    ASSERT_NE(stages, nullptr);
    EXPECT_DOUBLE_EQ(stages->value, 1.0);
    EXPECT_EQ(stages->tags.at("role"), "moe_routed_expert");
    EXPECT_EQ(stages->tags.at("elements"), "256");
    EXPECT_EQ(stages->tags.at("precision"), "fp16");
    EXPECT_EQ(stages->tags.at("no_op"), "true");

    PerfStatsCollector::reset();
}

/**
 * @test FP16 transport threshold reports tiny allreduces as FP32 in the BOM
 */
TEST_F(Test__TPAllreduceStage, BillOfMaterialsHonorsFP16MinimumElementThreshold)
{
    ScopedEnv enable_perf_stats("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnv fp16_min("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS", "1024");
    PerfStatsCollector::reset();

    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();
    params.count = 256;
    params.stage_name = "layer0_moe_combined_allreduce";
    params.precision = "fp16";

    TPAllreduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    const auto records = PerfStatsCollector::snapshot({"tp_allreduce_bom"});
    const PerfStatRecord *bytes = nullptr;
    for (const auto &record : records)
    {
        if (record.name == "bytes")
            bytes = &record;
    }

    ASSERT_NE(bytes, nullptr);
    EXPECT_DOUBLE_EQ(bytes->value, 256.0 * sizeof(float));
    EXPECT_EQ(bytes->tags.at("element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("tensor_element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("requested_transport_precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("transport_precision"), "fp32");

    PerfStatsCollector::reset();
}

/**
 * @test Captured graph replay records the same allreduce BOM without re-entering execute()
 */
TEST_F(Test__TPAllreduceStage, GraphReplayRecordsBillOfMaterials)
{
    ScopedEnv enable_perf_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto tp_ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();
    params.tensor = test_tensor_.get();
    params.count = 128;
    params.stage_name = "layer0_moe_combined_allreduce";
    params.precision = "fp16";

    TPAllreduceStage stage(params);

    ASSERT_TRUE(stage.needsOnGraphReplayed());
    stage.onGraphReplayed();

    const auto records = PerfStatsCollector::snapshot({"tp_allreduce_bom"});
    const PerfStatRecord *bytes = nullptr;
    const PerfStatRecord *stages = nullptr;
    for (const auto &record : records)
    {
        if (record.name == "bytes")
            bytes = &record;
        else if (record.name == "stages")
            stages = &record;
    }

    ASSERT_NE(bytes, nullptr);
    EXPECT_DOUBLE_EQ(bytes->value, 128.0 * sizeof(float));
    EXPECT_EQ(bytes->tags.at("role"), "moe_combined");
    EXPECT_EQ(bytes->tags.at("stage"), "layer0_moe_combined_allreduce");
    EXPECT_EQ(bytes->tags.at("elements"), "128");
    EXPECT_EQ(bytes->tags.at("element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("tensor_element_bytes"), std::to_string(sizeof(float)));
    EXPECT_EQ(bytes->tags.at("precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("requested_transport_precision"), "fp16");
    EXPECT_EQ(bytes->tags.at("transport_precision"), "fp32");
    EXPECT_EQ(bytes->tags.at("no_op"), "true");

    ASSERT_NE(stages, nullptr);
    EXPECT_DOUBLE_EQ(stages->value, 1.0);
    EXPECT_EQ(stages->tags.at("role"), "moe_combined");
    EXPECT_EQ(stages->tags.at("elements"), "128");
    EXPECT_EQ(stages->tags.at("precision"), "fp16");

    PerfStatsCollector::reset();
}
