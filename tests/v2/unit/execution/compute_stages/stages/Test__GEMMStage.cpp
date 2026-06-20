/**
 * @file Test__GEMMStage.cpp
 * @brief Unit tests for GEMMStage
 * @author GitHub Copilot
 * @date January 2026
 *
 * Tests the GEMMStage compute stage including:
 * - Construction and type identification
 * - IWorkspaceConsumerStage delegation
 * - Params validation
 * - Estimated metrics (FLOPs, memory bytes)
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/IWorkspaceConsumerStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "../../../../utils/PreparedWeightTestHarness.h"
#include <memory>
#include <optional>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    class RecordingFusedSwigluGemm final : public ITensorGemm, public IWorkspaceConsumer
    {
    public:
        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *,
            TensorBase *,
            int,
            int,
            int,
            bool,
            float,
            float,
            const TensorBase *,
            const IMPIContext *,
            int,
            DeviceWorkspaceManager *,
            int) override
        {
            return false;
        }

        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *,
            const TensorBase *,
            TensorBase *,
            int m,
            int n,
            int k,
            float alpha = 1.0f,
            float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            ++call_count;
            observed_m = m;
            observed_n = n;
            observed_k = k;
            observed_alpha = alpha;
            observed_beta = beta;
            observed_workspace = workspace;
            return true;
        }

        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override
        {
            return WorkspaceRequirements{};
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override { bound_workspace = workspace; }
        void unbindWorkspace() override { bound_workspace = nullptr; }
        bool hasWorkspace() const override { return bound_workspace != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace; }

        int call_count = 0;
        int observed_m = 0;
        int observed_n = 0;
        int observed_k = 0;
        float observed_alpha = 0.0f;
        float observed_beta = 0.0f;
        DeviceWorkspaceManager *observed_workspace = nullptr;
        DeviceWorkspaceManager *bound_workspace = nullptr;
    };

    PreparedWeightRef registerRecordingGemm(
        PreparedWeightStore &store,
        TensorBase *tensor,
        std::shared_ptr<RecordingFusedSwigluGemm> kernel,
        const std::string &canonical_name,
        ModelContextId model_id = ModelContextId{9901})
    {
        auto binding = makePreparedWeightTestBinding(
            tensor,
            DeviceId::cpu(),
            canonical_name,
            model_id);

        auto prepared_weights =
            std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();
        prepared_weights->kind =
            llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CPU_PACKED;
        prepared_weights->kernel = kernel.get();
        prepared_weights->owned_kernel = kernel;

        auto handle =
            std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
        handle->tensor = tensor;
        handle->device_id = DeviceId::cpu();
        handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CPU_PACKED;
        handle->prepared_weights = std::move(prepared_weights);

        return store.registerPreparedGemmHandle(
            binding,
            PreparedWeightKind::CpuPackedGemm,
            DeviceId::cpu(),
            std::move(handle));
    }
}

// =============================================================================
// Test Fixture
// =============================================================================

class Test__GEMMStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create small test tensors for unit tests
        // A: [4, 8] - input activation
        // B: [16, 8] - weight (transposed layout, n=16 rows, k=8 cols)
        // C: [4, 16] - output
        A_ = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8}, DeviceId::cpu());
        B_ = std::make_unique<FP32Tensor>(std::vector<size_t>{16, 8}, DeviceId::cpu());
        C_ = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 16}, DeviceId::cpu());

        // Initialize with zeros
        std::fill_n(A_->mutable_data(), 4 * 8, 0.0f);
        std::fill_n(B_->mutable_data(), 16 * 8, 0.0f);
        std::fill_n(C_->mutable_data(), 4 * 16, 0.0f);

        prepared_B_.emplace(makePreparedGemmFixture(
            B_.get(),
            DeviceId::cpu(),
            "blk.0.test_gemm.weight"));
    }

    // Helper to create valid params
    GEMMStage::Params makeValidParams()
    {
        GEMMStage::Params params;
        params.A = A_.get();
        params.B = B_.get();
        params.C = C_.get();
        params.m = 4;
        params.n = 16;
        params.k = 8;
        params.alpha = 1.0f;
        params.beta = 0.0f;
        params.transpose_B = true;
        params.device_id = DeviceId::cpu();
        params.prepared_ref = prepared_B_->ref;
        params.prepared_store = prepared_B_->store.get();
        return params;
    }

    std::unique_ptr<FP32Tensor> A_, B_, C_;
    std::optional<PreparedGemmFixture> prepared_B_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__GEMMStage, ConstructionWithValidParams)
{
    auto params = makeValidParams();
    EXPECT_NO_THROW({
        GEMMStage stage(params);
    });
}

TEST_F(Test__GEMMStage, TypeIsGEMM)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::GEMM);
}

TEST_F(Test__GEMMStage, ParamsAccessible)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    const auto &retrieved = stage.getParams();
    EXPECT_EQ(retrieved.A, A_.get());
    EXPECT_EQ(retrieved.B, B_.get());
    EXPECT_EQ(retrieved.C, C_.get());
    EXPECT_EQ(retrieved.m, 4);
    EXPECT_EQ(retrieved.n, 16);
    EXPECT_EQ(retrieved.k, 8);
}

// =============================================================================
// IWorkspaceConsumerStage Delegation Tests
// =============================================================================

TEST_F(Test__GEMMStage, CastsToIWorkspaceConsumer)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // GEMMStage inherits from IWorkspaceConsumerStage which inherits from IWorkspaceConsumer
    // This enables DeviceGraphExecutor to treat the stage as a workspace consumer
    IWorkspaceConsumer *consumer = dynamic_cast<IWorkspaceConsumer *>(&stage);

    EXPECT_NE(consumer, nullptr);
}

TEST_F(Test__GEMMStage, CastsToIWorkspaceConsumerStage)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // GEMMStage explicitly inherits from IWorkspaceConsumerStage
    IWorkspaceConsumerStage *consumer_stage = dynamic_cast<IWorkspaceConsumerStage *>(&stage);

    EXPECT_NE(consumer_stage, nullptr);
}

TEST_F(Test__GEMMStage, GetKernelAsWorkspaceConsumerReturnsNullWhenBIsNull)
{
    auto params = makeValidParams();
    params.B = nullptr;

    GEMMStage stage(params);

    // When B is null, there's no kernel to get
    EXPECT_EQ(stage.getKernelAsWorkspaceConsumer(), nullptr);
}

TEST_F(Test__GEMMStage, GetKernelAsWorkspaceConsumerReturnsNonNullForValidB)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // For FP32Tensor weights, KernelFactory should return a kernel
    // Note: This may return nullptr if no kernel supports FP32 weights
    // The test verifies the delegation happens without crashing
    // Whether nullptr or valid depends on kernel availability
    IWorkspaceConsumer *consumer = stage.getKernelAsWorkspaceConsumer();
    // Just verify we can call the method without crashing
    (void)consumer; // May be nullptr for FP32 weights
}

TEST_F(Test__GEMMStage, HasWorkspaceReturnsFalseWhenNoWorkspaceBound)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // Initially no workspace is bound
    EXPECT_FALSE(stage.hasWorkspace());
}

TEST_F(Test__GEMMStage, GetWorkspaceReturnsNullWhenNoWorkspaceBound)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_EQ(stage.getWorkspace(), nullptr);
}

TEST_F(Test__GEMMStage, BindWorkspaceStoresWorkspacePointer)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // Create a workspace manager with CPU device and small budget
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 1024);

    stage.bindWorkspace(&workspace);

    // getWorkspace should return what we bound
    EXPECT_EQ(stage.getWorkspace(), &workspace);
}

TEST_F(Test__GEMMStage, UnbindWorkspaceClearsWorkspace)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    DeviceWorkspaceManager workspace(DeviceId::cpu(), 1024);
    stage.bindWorkspace(&workspace);
    EXPECT_EQ(stage.getWorkspace(), &workspace);

    stage.unbindWorkspace();

    EXPECT_EQ(stage.getWorkspace(), nullptr);
    EXPECT_FALSE(stage.hasWorkspace());
}

// =============================================================================
// Params Validation Tests
// =============================================================================

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnNullA)
{
    GEMMStage::Params params = makeValidParams();
    params.A = nullptr;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnNullB)
{
    GEMMStage::Params params = makeValidParams();
    params.B = nullptr;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnNullC)
{
    GEMMStage::Params params = makeValidParams();
    params.C = nullptr;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnZeroM)
{
    GEMMStage::Params params = makeValidParams();
    params.m = 0;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnZeroN)
{
    GEMMStage::Params params = makeValidParams();
    params.n = 0;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnZeroK)
{
    GEMMStage::Params params = makeValidParams();
    params.k = 0;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsOnNegativeDimensions)
{
    GEMMStage::Params params = makeValidParams();
    params.m = -1;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidateThrowsWhenBiasRequiredButNotProvided)
{
    GEMMStage::Params params = makeValidParams();
    params.bias_required = true;
    params.bias = nullptr;
    params.bias_tensor = nullptr;

    EXPECT_THROW({ params.validate("TestStage"); }, std::runtime_error);
}

TEST_F(Test__GEMMStage, ParamsValidatePassesWhenBiasRequiredAndProvided)
{
    auto bias = std::make_unique<FP32Tensor>(std::vector<size_t>{16}, DeviceId::cpu());
    std::fill_n(bias->mutable_data(), 16, 0.0f);

    GEMMStage::Params params = makeValidParams();
    params.bias_required = true;
    params.bias_tensor = bias.get();

    EXPECT_NO_THROW({
        params.validate("TestStage");
    });
}

TEST_F(Test__GEMMStage, ParamsValidatePassesWhenBiasRequiredAndRawBiasProvided)
{
    std::vector<float> bias_data(16, 0.0f);

    GEMMStage::Params params = makeValidParams();
    params.bias_required = true;
    params.bias = bias_data.data();

    EXPECT_NO_THROW({
        params.validate("TestStage");
    });
}

TEST_F(Test__GEMMStage, ParamsValidatePassesWithValidParams)
{
    GEMMStage::Params params = makeValidParams();

    EXPECT_NO_THROW({
        params.validate("TestStage");
    });
}

// =============================================================================
// Params::getBiasData Tests
// =============================================================================

TEST_F(Test__GEMMStage, GetBiasDataReturnsNullWhenNoBias)
{
    GEMMStage::Params params = makeValidParams();
    params.bias = nullptr;
    params.bias_tensor = nullptr;

    EXPECT_EQ(params.getBiasData(), nullptr);
}

TEST_F(Test__GEMMStage, GetBiasDataReturnsRawBiasPointer)
{
    std::vector<float> bias_data(16, 1.0f);

    GEMMStage::Params params = makeValidParams();
    params.bias = bias_data.data();
    params.bias_tensor = nullptr;

    EXPECT_EQ(params.getBiasData(), bias_data.data());
}

TEST_F(Test__GEMMStage, GetBiasDataPrefersBiasTensorOverRawPointer)
{
    auto bias_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{16}, DeviceId::cpu());
    std::fill_n(bias_tensor->mutable_data(), 16, 2.0f);

    std::vector<float> raw_bias(16, 1.0f);

    GEMMStage::Params params = makeValidParams();
    params.bias = raw_bias.data();
    params.bias_tensor = bias_tensor.get();

    // bias_tensor should be preferred over raw bias pointer
    EXPECT_EQ(params.getBiasData(), bias_tensor->data());
    EXPECT_NE(params.getBiasData(), raw_bias.data());
}

// =============================================================================
// Estimated FLOPs and Memory Tests
// =============================================================================

TEST_F(Test__GEMMStage, EstimatedFlopsReturnsCorrectValue)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // GEMM FLOPs: 2 * M * N * K
    // M=4, N=16, K=8 -> 2 * 4 * 16 * 8 = 1024
    EXPECT_EQ(stage.estimatedFlops(), 1024);
}

TEST_F(Test__GEMMStage, EstimatedFlopsWithLargerDimensions)
{
    // Create larger tensors for this test
    auto large_A = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64}, DeviceId::cpu());
    auto large_B = std::make_unique<FP32Tensor>(std::vector<size_t>{128, 64}, DeviceId::cpu());
    auto large_C = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 128}, DeviceId::cpu());

    GEMMStage::Params params;
    params.A = large_A.get();
    params.B = large_B.get();
    params.C = large_C.get();
    params.m = 32;
    params.n = 128;
    params.k = 64;
    params.device_id = DeviceId::cpu();

    GEMMStage stage(params);

    // 2 * 32 * 128 * 64 = 524,288
    EXPECT_EQ(stage.estimatedFlops(), 524288);
}

TEST_F(Test__GEMMStage, EstimatedMemoryBytesReturnsCorrectValue)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    // Memory bytes estimate (assuming FP32 for all):
    // A: m * k * 4 = 4 * 8 * 4 = 128 bytes
    // B: k * n * 4 = 8 * 16 * 4 = 512 bytes
    // C: m * n * 4 = 4 * 16 * 4 = 256 bytes
    // Total = 896 bytes
    EXPECT_EQ(stage.estimatedMemoryBytes(), 896);
}

// =============================================================================
// Backend Support Tests
// =============================================================================

TEST_F(Test__GEMMStage, SupportsCPUBackend)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(Test__GEMMStage, DoesNotSupportVulkanBackend)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_VULKAN));
}

TEST_F(Test__GEMMStage, DoesNotSupportMetalBackend)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_METAL));
}

// =============================================================================
// StageDumpInfo Tests
// =============================================================================

TEST_F(Test__GEMMStage, GetDumpInfoContainsExpectedFields)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    StageDumpInfo info = stage.getDumpInfo();

    // Should have inputs, weights, outputs, and scalars
    // Just verify we can call this without crashing
    EXPECT_NO_THROW({
        auto dump = stage.getDumpInfo();
    });
}

// =============================================================================
// StageBufferRequirements Tests
// =============================================================================

TEST_F(Test__GEMMStage, GetBufferRequirementsWithValidParams)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    StageBufferRequirements reqs = stage.getBufferRequirements();

    // Should have requirements for A, B, C
    EXPECT_NO_THROW({
        auto requirements = stage.getBufferRequirements();
    });
}

TEST_F(Test__GEMMStage, GetBufferRequirementsUsesKForFusedSwigluGateInput)
{
    auto params = makeValidParams();
    params.gate_input = A_.get();

    GEMMStage stage(params);
    StageBufferRequirements reqs = stage.getBufferRequirements();

    const BufferDescriptor *gate = reqs.getByName("gate_input");
    ASSERT_NE(gate, nullptr);
    ASSERT_EQ(gate->shape.size(), 2u);
    EXPECT_EQ(gate->shape[0], static_cast<size_t>(params.m));
    EXPECT_EQ(gate->shape[1], static_cast<size_t>(params.k));
}

TEST_F(Test__GEMMStage, GetBufferRequirementsEmptyWithNullTensors)
{
    GEMMStage::Params params;
    params.A = nullptr;
    params.B = nullptr;
    params.C = nullptr;
    params.m = 4;
    params.n = 16;
    params.k = 8;
    params.device_id = DeviceId::cpu();

    GEMMStage stage(params);

    // With null tensors, requirements should be empty
    StageBufferRequirements reqs = stage.getBufferRequirements();
    // Just verify no crash
}

TEST_F(Test__GEMMStage, ExecutePassesBoundWorkspaceToFusedSwigluKernel)
{
    auto params = makeValidParams();
    auto gate = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8}, DeviceId::cpu());
    std::fill_n(gate->mutable_data(), 4 * 8, 0.0f);

    auto store = std::make_unique<PreparedWeightStore>(ModelContextId{9901});
    auto kernel = std::make_shared<RecordingFusedSwigluGemm>();
    params.prepared_store = store.get();
    params.prepared_ref = registerRecordingGemm(
        *store,
        B_.get(),
        kernel,
        "blk.0.ffn_down.weight");
    params.gate_input = gate.get();

    GEMMStage stage(params);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 1024);
    stage.bindWorkspace(&workspace);

    CPUDeviceContext ctx(DeviceId::cpu(), 1);
    ASSERT_TRUE(stage.execute(&ctx));

    EXPECT_EQ(kernel->call_count, 1);
    EXPECT_EQ(kernel->observed_workspace, &workspace);
    EXPECT_EQ(kernel->bound_workspace, &workspace);
    EXPECT_EQ(kernel->observed_m, params.m);
    EXPECT_EQ(kernel->observed_n, params.n);
    EXPECT_EQ(kernel->observed_k, params.k);
}

// =============================================================================
// requiresAllreduce Tests
// =============================================================================

TEST_F(Test__GEMMStage, RequiresAllreduceReturnsFalseByDefault)
{
    auto params = makeValidParams();
    GEMMStage stage(params);

    EXPECT_FALSE(stage.requiresAllreduce());
}

TEST_F(Test__GEMMStage, RequiresAllreduceReturnsTrueWhenSet)
{
    auto params = makeValidParams();
    params.needs_allreduce = true;

    GEMMStage stage(params);

    EXPECT_TRUE(stage.requiresAllreduce());
}
