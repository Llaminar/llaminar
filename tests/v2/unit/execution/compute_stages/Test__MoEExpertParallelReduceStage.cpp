#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace llaminar2::test
{
namespace
{

    std::unique_ptr<FP32Tensor> fp32Tensor(size_t rows, size_t cols, const std::vector<float> &values)
    {
        auto tensor = TestTensorFactory::createFP32({rows, cols});
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    }

    std::vector<float> snapshot(const FP32Tensor &tensor)
    {
        const float *data = tensor.data();
        return std::vector<float>(data, data + tensor.numel());
    }

    void expectTensorValues(const FP32Tensor &tensor, const std::vector<float> &expected)
    {
        ASSERT_EQ(tensor.numel(), expected.size());
        const float *data = tensor.data();
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_FLOAT_EQ(data[i], expected[i]) << "index=" << i;
        }
    }

    MoEExpertParallelReduceStage::Params reduceParams(
        std::vector<const ITensor *> partials,
        ITensor *output,
        size_t rows = 0,
        size_t cols = 0)
    {
        MoEExpertParallelReduceStage::Params params;
        params.device_id = DeviceId::cpu();
        params.partials = std::move(partials);
        params.output = output;
        params.rows = rows;
        params.cols = cols;
        return params;
    }

} // namespace

class Test__MoEExpertParallelReduceStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    std::unique_ptr<llaminar2::testing::MockDeviceContext> ctx_;
};

TEST_F(Test__MoEExpertParallelReduceStage, SharedAndThreeRoutedTierPartialsSumExactlyAndInputsStayUnchanged)
{
    auto shared = fp32Tensor(2, 3, {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f});
    auto routed0 = fp32Tensor(2, 3, {10.0f, 20.0f, 30.0f,
                                    40.0f, 50.0f, 60.0f});
    auto routed1 = fp32Tensor(2, 3, {-1.0f, -2.0f, -3.0f,
                                    -4.0f, -5.0f, -6.0f});
    auto routed2 = fp32Tensor(2, 3, {0.5f, 1.5f, 2.5f,
                                    3.5f, 4.5f, 5.5f});
    auto output = fp32Tensor(2, 3, {99.0f, 99.0f, 99.0f,
                                   99.0f, 99.0f, 99.0f});

    const auto before_shared = snapshot(*shared);
    const auto before_routed0 = snapshot(*routed0);
    const auto before_routed1 = snapshot(*routed1);
    const auto before_routed2 = snapshot(*routed2);

    auto params = reduceParams({shared.get(), routed0.get(), routed1.get(), routed2.get()}, output.get(), 2, 3);
    MoEExpertParallelReduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_PARALLEL_REDUCE);
    EXPECT_EQ(output->shape(), (std::vector<size_t>{2, 3}));
    expectTensorValues(*output, {10.5f, 21.5f, 32.5f,
                                43.5f, 54.5f, 65.5f});

    EXPECT_EQ(snapshot(*shared), before_shared);
    EXPECT_EQ(snapshot(*routed0), before_routed0);
    EXPECT_EQ(snapshot(*routed1), before_routed1);
    EXPECT_EQ(snapshot(*routed2), before_routed2);
}

TEST_F(Test__MoEExpertParallelReduceStage, HostStagedModeRecordsCrossDomainDiagnostics)
{
    auto shared = fp32Tensor(2, 2, {1.0f, 2.0f,
                                   3.0f, 4.0f});
    auto rocm = fp32Tensor(2, 2, {10.0f, 20.0f,
                                 30.0f, 40.0f});
    auto cpu = fp32Tensor(2, 2, {-1.0f, -2.0f,
                                -3.0f, -4.0f});
    auto output = fp32Tensor(2, 2, {0.0f, 0.0f,
                                   0.0f, 0.0f});
    MoEExpertParallelReduceDiagnostics diagnostics;

    auto params = reduceParams({shared.get(), rocm.get(), cpu.get()}, output.get(), 2, 2);
    params.mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
    params.continuation_domain = "cuda_fast";
    params.continuation_device = DeviceId::cpu();
    params.partial_infos = {
        {.name = "shared_expert", .source_domain = "cuda_fast", .source_device = DeviceId::cuda(0)},
        {.name = "hot", .source_domain = "rocm_hot", .source_device = DeviceId::rocm(0)},
        {.name = "cold", .source_domain = "cpu_cold", .source_device = DeviceId::cpu()},
    };
    params.diagnostics = &diagnostics;
    MoEExpertParallelReduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    expectTensorValues(*output, {10.0f, 20.0f,
                                30.0f, 40.0f});
    EXPECT_EQ(diagnostics.mode, MoEExpertParallelReduceMode::HostStagedCorrectness);
    EXPECT_TRUE(diagnostics.host_staged);
    EXPECT_EQ(diagnostics.continuation_domain, "cuda_fast");
    EXPECT_EQ(diagnostics.partial_count, 3u);
    EXPECT_EQ(diagnostics.input_bytes, 3u * 4u * sizeof(float));
    EXPECT_EQ(diagnostics.host_staged_read_bytes, diagnostics.input_bytes);
    EXPECT_EQ(diagnostics.device_to_host_bytes, 2u * 4u * sizeof(float));
    EXPECT_EQ(diagnostics.host_to_device_bytes, 0u);
    EXPECT_EQ(diagnostics.total_transfer_bytes, diagnostics.device_to_host_bytes);
    EXPECT_TRUE(diagnostics.output_resident_on_continuation);
    ASSERT_EQ(diagnostics.partials.size(), 3u);
    EXPECT_TRUE(diagnostics.partials[0].source_is_continuation);
    EXPECT_FALSE(diagnostics.partials[1].source_is_continuation);
    EXPECT_FALSE(diagnostics.partials[2].source_is_continuation);
    EXPECT_EQ(diagnostics.partials[0].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
    EXPECT_EQ(diagnostics.partials[1].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
    EXPECT_EQ(diagnostics.partials[2].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
}

TEST_F(Test__MoEExpertParallelReduceStage, OptimizedContinuationModeFallsBackToHostStagedForCpuContinuation)
{
    auto partial = fp32Tensor(1, 2, {1.0f, 2.0f});
    auto second = fp32Tensor(1, 2, {3.0f, 4.0f});
    auto output = fp32Tensor(1, 2, {0.0f, 0.0f});
    MoEExpertParallelReduceDiagnostics diagnostics;

    auto params = reduceParams({partial.get(), second.get()}, output.get(), 1, 2);
    params.mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;
    params.continuation_domain = "cpu_continuation";
    params.continuation_device = DeviceId::cpu();
    params.partial_infos = {
        {.name = "first", .source_domain = "rocm_hot", .source_device = DeviceId::rocm(0)},
        {.name = "second", .source_domain = "cpu_cold", .source_device = DeviceId::cpu()},
    };
    params.diagnostics = &diagnostics;
    MoEExpertParallelReduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    expectTensorValues(*output, {4.0f, 6.0f});
    EXPECT_EQ(diagnostics.mode, MoEExpertParallelReduceMode::HostStagedCorrectness);
    EXPECT_TRUE(diagnostics.host_staged);
    EXPECT_EQ(diagnostics.continuation_domain, "cpu_continuation");
    EXPECT_EQ(diagnostics.continuation_device, DeviceId::cpu());
    EXPECT_EQ(diagnostics.partial_count, 2u);
    EXPECT_EQ(diagnostics.input_bytes, 2u * 2u * sizeof(float));
    EXPECT_EQ(diagnostics.host_staged_read_bytes, diagnostics.input_bytes);
    EXPECT_EQ(diagnostics.host_to_device_bytes, 0u);
    ASSERT_EQ(diagnostics.partials.size(), 2u);
    EXPECT_EQ(diagnostics.partials[0].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
    EXPECT_EQ(diagnostics.partials[1].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
}

TEST_F(Test__MoEExpertParallelReduceStage, ZeroPartialsFillOutputWithZerosAndPreserveShape)
{
    auto output = fp32Tensor(2, 2, {7.0f, 8.0f,
                                   9.0f, 10.0f});
    auto params = reduceParams({}, output.get(), 2, 2);
    MoEExpertParallelReduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(output->shape(), (std::vector<size_t>{2, 2}));
    expectTensorValues(*output, {0.0f, 0.0f,
                                0.0f, 0.0f});
}

TEST_F(Test__MoEExpertParallelReduceStage, CapacitySizedOutputReducesIntoLivePrefix)
{
    auto partial0 = fp32Tensor(2, 3, {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f});
    auto partial1 = fp32Tensor(4, 3, {10.0f, 20.0f, 30.0f,
                                     40.0f, 50.0f, 60.0f,
                                     70.0f, 80.0f, 90.0f,
                                     100.0f, 110.0f, 120.0f});
    auto output = fp32Tensor(4, 3, {99.0f, 99.0f, 99.0f,
                                   99.0f, 99.0f, 99.0f,
                                   77.0f, 77.0f, 77.0f,
                                   77.0f, 77.0f, 77.0f});

    auto params = reduceParams({partial0.get(), partial1.get()}, output.get(), 2, 3);
    MoEExpertParallelReduceStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    expectTensorValues(*output, {11.0f, 22.0f, 33.0f,
                                44.0f, 55.0f, 66.0f,
                                77.0f, 77.0f, 77.0f,
                                77.0f, 77.0f, 77.0f});
}

TEST_F(Test__MoEExpertParallelReduceStage, ShapeMismatchFailsClearly)
{
    auto partial = fp32Tensor(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    auto output = fp32Tensor(2, 2, {0.0f, 0.0f,
                                   0.0f, 0.0f});
    auto params = reduceParams({partial.get()}, output.get(), 2, 2);
    MoEExpertParallelReduceStage stage(params);

    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, TypeMismatchFailsClearly)
{
    auto partial = TestTensorFactory::createFP16({2, 2});
    auto output = fp32Tensor(2, 2, {0.0f, 0.0f,
                                   0.0f, 0.0f});
    auto params = reduceParams({partial.get()}, output.get(), 2, 2);
    MoEExpertParallelReduceStage stage(params);

    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, ExpectedOutputShapeMismatchFailsClearly)
{
    auto partial = fp32Tensor(2, 2, {1.0f, 2.0f,
                                    3.0f, 4.0f});
    auto output = fp32Tensor(2, 2, {0.0f, 0.0f,
                                   0.0f, 0.0f});
    auto params = reduceParams({partial.get()}, output.get(), 2, 3);
    MoEExpertParallelReduceStage stage(params);

    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, AdvertisesCrossDomainBackendSupport)
{
    auto output = fp32Tensor(1, 1, {0.0f});
    auto params = reduceParams({}, output.get(), 1, 1);
    MoEExpertParallelReduceStage stage(params);

    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
}

TEST_F(Test__MoEExpertParallelReduceStage, ManualCoherenceBridgeDoesNotAdvertiseArenaContract)
{
    auto output = fp32Tensor(1, 1, {0.0f});
    auto params = reduceParams({}, output.get(), 1, 1);
    params.device_id = DeviceId::cuda(0);
    params.continuation_device = DeviceId::cuda(0);
    params.mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;

    MoEExpertParallelReduceStage stage(params);

    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::NONE);
    EXPECT_TRUE(stage.bufferContract().empty());
}

TEST_F(Test__MoEExpertParallelReduceStage, FactoryCreatesReduceStage)
{
    auto output = fp32Tensor(1, 1, {0.0f});
    auto params = reduceParams({}, output.get(), 1, 1);

    auto stage = ComputeStageFactory::createMoEExpertParallelReduce(params);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::MOE_EXPERT_PARALLEL_REDUCE);
    EXPECT_STREQ(computeStageTypeName(stage->type()), "MOE_EXPERT_PARALLEL_REDUCE");
}

// ============================================================================
// Bridge Phase 7A: sparse partial row interface tests
// ============================================================================

TEST_F(Test__MoEExpertParallelReduceStage, SparsePartialScatterAddsSelectedRowsAndLeavesOtherRowsZero)
{
    // Partial covers only rows 1 and 3 out of [0..3] in a 4-row, 3-col output.
    // The partial tensor has shape [2, 3] (compact: only selected rows).
    auto sparse_partial = fp32Tensor(2, 3, {10.0f, 20.0f, 30.0f,
                                            40.0f, 50.0f, 60.0f});
    auto output = fp32Tensor(4, 3, {0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f});

    MoEExpertParallelReduceDiagnostics diagnostics;
    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {sparse_partial.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "cold_sparse",
            .source_domain = "cpu_cold",
            .source_device = DeviceId::cpu(),
            .selected_rows = {1, 3},
        },
    };
    params.output = output.get();
    params.rows = 4;
    params.cols = 3;
    params.mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
    params.diagnostics = &diagnostics;

    MoEExpertParallelReduceStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Row 0 and Row 2 must remain zero (not selected).
    expectTensorValues(*output, {0.0f, 0.0f, 0.0f,
                                 10.0f, 20.0f, 30.0f,
                                 0.0f, 0.0f, 0.0f,
                                 40.0f, 50.0f, 60.0f});

    // Diagnostics must report sparse partial.
    ASSERT_EQ(diagnostics.partials.size(), 1u);
    EXPECT_TRUE(diagnostics.partials[0].is_sparse);
    EXPECT_EQ(diagnostics.partials[0].sparse_row_count, 2u);
    EXPECT_EQ(diagnostics.sparse_partial_count, 1u);
    EXPECT_EQ(diagnostics.partials[0].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);
    // Bytes reported should reflect compact layout (2 rows not 4).
    EXPECT_EQ(diagnostics.partials[0].bytes, 2u * 3u * sizeof(float));
}

TEST_F(Test__MoEExpertParallelReduceStage, MixedSparseAndDensePartialsReduceCorrectly)
{
    // Dense partial covers all rows (continuation tier output).
    auto dense = fp32Tensor(3, 2, {1.0f, 2.0f,
                                   3.0f, 4.0f,
                                   5.0f, 6.0f});
    // Sparse partial: only row 1 (one selected row, compact shape [1, 2]).
    auto sparse = fp32Tensor(1, 2, {10.0f, 20.0f});
    auto output = fp32Tensor(3, 2, {0.0f, 0.0f,
                                   0.0f, 0.0f,
                                   0.0f, 0.0f});

    MoEExpertParallelReduceDiagnostics diagnostics;
    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {dense.get(), sparse.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "continuation",
            .source_domain = "cuda_fast",
            .source_device = DeviceId::cpu(),
        },
        MoEExpertParallelReducePartialInfo{
            .name = "cold_tier",
            .source_domain = "cpu_cold",
            .source_device = DeviceId::cpu(),
            .selected_rows = {1},
        },
    };
    params.output = output.get();
    params.rows = 3;
    params.cols = 2;
    params.mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
    params.diagnostics = &diagnostics;

    MoEExpertParallelReduceStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Row 0: only dense contributes.
    // Row 1: dense + sparse scatter.
    // Row 2: only dense contributes.
    expectTensorValues(*output, {1.0f, 2.0f,
                                 13.0f, 24.0f,
                                 5.0f, 6.0f});

    ASSERT_EQ(diagnostics.partials.size(), 2u);
    EXPECT_FALSE(diagnostics.partials[0].is_sparse);
    EXPECT_EQ(diagnostics.partials[0].sparse_row_count, 0u);
    EXPECT_TRUE(diagnostics.partials[1].is_sparse);
    EXPECT_EQ(diagnostics.partials[1].sparse_row_count, 1u);
    EXPECT_EQ(diagnostics.sparse_partial_count, 1u);
    EXPECT_EQ(diagnostics.partial_count, 2u);
    // Dense bytes = 3 rows * 2 cols * sizeof(float)
    EXPECT_EQ(diagnostics.partials[0].bytes, 3u * 2u * sizeof(float));
    // Sparse bytes = 1 row (compact) * 2 cols * sizeof(float)
    EXPECT_EQ(diagnostics.partials[1].bytes, 1u * 2u * sizeof(float));
}

TEST_F(Test__MoEExpertParallelReduceStage, SparsePartialShapeMismatchFailsClearly)
{
    // Partial claims 2 selected rows, but the tensor only has 1 row.
    auto bad_sparse = fp32Tensor(1, 3, {1.0f, 2.0f, 3.0f});
    auto output = fp32Tensor(4, 3, {0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f});

    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {bad_sparse.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "bad",
            .source_domain = "cpu_cold",
            .source_device = DeviceId::cpu(),
            .selected_rows = {0, 2},  // 2 rows claimed, but tensor has only 1 row.
        },
    };
    params.output = output.get();
    params.rows = 4;
    params.cols = 3;

    MoEExpertParallelReduceStage stage(std::move(params));
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, SparsePartialOutOfRangeSelectedRowFailsClearly)
{
    auto sparse = fp32Tensor(1, 3, {1.0f, 2.0f, 3.0f});
    auto output = fp32Tensor(4, 3, {0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f});

    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {sparse.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "bad_row",
            .source_domain = "cpu_cold",
            .source_device = DeviceId::cpu(),
            .selected_rows = {4},
        },
    };
    params.output = output.get();
    params.rows = 4;
    params.cols = 3;

    MoEExpertParallelReduceStage stage(std::move(params));
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, SparsePartialDuplicateSelectedRowFailsClearly)
{
    auto sparse = fp32Tensor(2, 3, {1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f});
    auto output = fp32Tensor(4, 3, {0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f});

    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {sparse.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "duplicate_row",
            .source_domain = "cpu_cold",
            .source_device = DeviceId::cpu(),
            .selected_rows = {1, 1},
        },
    };
    params.output = output.get();
    params.rows = 4;
    params.cols = 3;

    MoEExpertParallelReduceStage stage(std::move(params));
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertParallelReduceStage, DiagnosticsDistinguishesHostStagedTransportFromHostSummedCorrectness)
{
    // HostStagedCorrectness mode: all paths are HostSummedCorrectnessFallback.
    // This verifies the diagnostic label distinction:
    //   - HostSummedCorrectnessFallback = correctness bridge (host owns summation)
    //   - HostStagedThenDeviceAccumulated = transport via host, accumulation on device
    // The latter requires real GPU hardware (tested by LayoutA/B integration tests).
    // This test verifies the correctness fallback path label is accurate.
    auto partial = fp32Tensor(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f});
    auto output = fp32Tensor(2, 4, {0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f});

    MoEExpertParallelReduceDiagnostics diagnostics;
    MoEExpertParallelReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.partials = {partial.get()};
    params.partial_infos = {
        MoEExpertParallelReducePartialInfo{
            .name = "hot_rocm",
            .source_domain = "rocm_hot",
            .source_device = DeviceId::rocm(0),
        },
    };
    params.output = output.get();
    params.rows = 2;
    params.cols = 4;
    params.mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
    params.continuation_domain = "cuda_fast";
    params.continuation_device = DeviceId::cpu();  // CPU proxy; real CUDA tested by LayoutB integration test.
    params.diagnostics = &diagnostics;

    MoEExpertParallelReduceStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Host-staged correctness: summation on host, path is HostSummedCorrectnessFallback.
    EXPECT_EQ(diagnostics.mode, MoEExpertParallelReduceMode::HostStagedCorrectness);
    EXPECT_TRUE(diagnostics.host_staged);
    ASSERT_EQ(diagnostics.partials.size(), 1u);
    EXPECT_EQ(diagnostics.partials[0].accumulation_path,
              MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback)
        << "HostStagedCorrectness must label all paths HostSummedCorrectnessFallback to "
           "distinguish from HostStagedThenDeviceAccumulated (transport via host, "
           "accumulation on continuation device), which requires real GPU hardware.";
    EXPECT_FALSE(diagnostics.partials[0].is_sparse);
    EXPECT_EQ(diagnostics.sparse_partial_count, 0u);
}

} // namespace llaminar2::test
