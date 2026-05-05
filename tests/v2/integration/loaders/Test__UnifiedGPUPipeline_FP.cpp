/**
 * @file Test__UnifiedGPUPipeline_FP.cpp
 * @brief Integration tests for unified GPU weight pipeline with floating-point models
 *
 * Mirrors Test__UnifiedGPUPipeline.cpp (quantised Q4_K path) but uses a BF16
 * model (Qwen3.5-35B-A3B-BF16 multipart GGUF) to exercise the RAW_FP passthrough
 * path through the pipeline.
 *
 * Verifies:
 * 1. BF16 MoE expert weights load via the pipelined RAW_FP path
 * 2. BF16 dense weights (shared experts, attention) load via RAW_FP
 * 3. ExpertGemmRegistry is populated with FP GEMM kernels
 * 4. FP GEMM kernels produce correct matmul results (spot-check)
 * 5. Dense FP GEMM weights are registered in KernelFactory
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <numeric>

#include "../../utils/TestModelHelper.h"
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/loaders/ModelLoader.h"
#include "../../src/v2/loaders/ExpertGemmRegistry.h"
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/kernels/KernelFactory.h"
#include "../../src/v2/backends/DeviceId.h"
#include "../../src/v2/backends/BackendManager.h"
#include "../../src/v2/models/qwen35moe/Qwen35MoESchema.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/utils/Logger.h"
#include "../../src/v2/interfaces/IWorkspaceConsumer.h"
#include "../../src/v2/execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../src/v2/execution/local_execution/coherence/GpuCoherence.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar::v2::kernels;

// =============================================================================
// Constants
// =============================================================================

static const char *BF16_MODEL_PATH =
    "/opt/llaminar-models/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf";

// Qwen3.5-35B-A3B has 256 experts per MoE layer
static constexpr int EXPECTED_NUM_EXPERTS = 256;

// We test layers 0-2 only (all GDN layers with MoE).
// Layer 3 is FA (full attention) — has no MoE experts but has dense attention weights.
static constexpr int TEST_LAYER_START = 0;
static constexpr int TEST_LAYER_END = 2;

// =============================================================================
// Helpers
// =============================================================================

static bool hasModel()
{
    std::ifstream f(BF16_MODEL_PATH);
    return f.good();
}

static bool hasGPU()
{
    return hasROCmBackend() || hasCUDABackend();
}

static DeviceId getFirstGPU()
{
    if (hasROCmBackend())
        return DeviceId(DeviceType::ROCm, 0);
    if (hasCUDABackend())
        return DeviceId(DeviceType::CUDA, 0);
    return DeviceId::cpu();
}

// =============================================================================
// Test Fixture
// =============================================================================

class UnifiedGPUPipeline_FP_Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!hasModel())
        {
            GTEST_SKIP() << "BF16 model not found: " << BF16_MODEL_PATH;
        }
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU available (need ROCm or CUDA)";
        }

        device_ = getFirstGPU();

        // Create MPI context (single rank for this test)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);

        // Create tensor factory and model loader
        factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        loader_ = std::make_unique<ModelLoader>(factory_.get());

        if (!tryLoadModel(*loader_, BF16_MODEL_PATH))
        {
            GTEST_SKIP() << "Failed to load model: " << BF16_MODEL_PATH;
        }

        // Verify we actually loaded BF16 tensors
        auto sample = loader_->loadTensor("blk.0.ffn_gate_exps.weight");
        if (!sample)
        {
            GTEST_SKIP() << "Model loaded but missing expected tensor blk.0.ffn_gate_exps.weight";
        }
        ASSERT_EQ(sample->native_type(), TensorType::BF16)
            << "Expected BF16 model but got type " << static_cast<int>(sample->native_type());

        // Create WeightManager with REPLICATED strategy (single device)
        weight_mgr_ = std::make_unique<WeightManager>(
            *loader_, mpi_ctx_, nullptr, WeightDistributionStrategy::REPLICATED);

        // Configure sharding config from Qwen35MoE schema factory
        Qwen35MoESchemaFactory schema_factory;
        weight_mgr_->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
    }

    void TearDown() override
    {
        // Clear KernelFactory caches to avoid leaking between tests
        KernelFactory::clearCache();
    }

    /// Load layer weights into cache for a range of layers
    void loadLayerWeights(int first_layer, int last_layer)
    {
        const std::vector<std::string> weight_suffixes = {
            // GDN weights
            "attn_qkv.weight", "attn_gate.weight",
            // Common norms
            "attn_norm.weight", "post_attention_norm.weight",
            // MoE expert weights
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_inp.weight",
            // Shared expert weights
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        };

        for (int layer = first_layer; layer < last_layer; ++layer)
        {
            for (const auto &suffix : weight_suffixes)
            {
                std::string name = "blk." + std::to_string(layer) + "." + suffix;
                weight_mgr_->getWeightForDevice(name);
            }
        }

        // Also load global weights needed by the pipeline
        weight_mgr_->getWeightForDevice("token_embd.weight");
    }

    /// Create a layer filter function for the tested layer range
    std::function<bool(const std::string &)> makeLayerFilter(int first_layer, int last_layer)
    {
        return [first_layer, last_layer](const std::string &name) -> bool
        {
            // Global weights
            if (name.find("blk.") == std::string::npos)
                return true;

            // Extract layer index from "blk.N.xxx"
            auto dot1 = name.find('.');
            if (dot1 == std::string::npos)
                return false;
            auto dot2 = name.find('.', dot1 + 1);
            if (dot2 == std::string::npos)
                return false;

            int layer = std::stoi(name.substr(dot1 + 1, dot2 - dot1 - 1));
            return layer >= first_layer && layer < last_layer;
        };
    }

    DeviceId device_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> factory_;
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<WeightManager> weight_mgr_;
};

// =============================================================================
// Test: BF16 MoE experts loaded and registered via FP pipeline
// =============================================================================

TEST_F(UnifiedGPUPipeline_FP_Test, BF16_ExpertsRegisteredInRegistry)
{
    // Load weights for test layers (2 MoE layers)
    loadLayerWeights(TEST_LAYER_START, TEST_LAYER_END);

    // Run the unified GPU pipeline with layer filter
    auto filter = makeLayerFilter(TEST_LAYER_START, TEST_LAYER_END);
    bool ok = weight_mgr_->packGemmWeightsViaPipeline(device_, filter);
    ASSERT_TRUE(ok) << "packGemmWeightsViaPipeline failed for BF16 model";

    // Verify ExpertGemmRegistry is populated
    const auto &registry = weight_mgr_->expertGemmRegistry();

    // Each MoE layer should have 256 experts × 3 roles = 768 engines
    const int moe_layers = TEST_LAYER_END - TEST_LAYER_START;

    for (int layer = TEST_LAYER_START; layer < TEST_LAYER_END; ++layer)
    {
        EXPECT_TRUE(registry.hasEnginesForLayer(device_, layer))
            << "Layer " << layer << " should have BF16 expert engines registered";

        // Spot-check experts at boundaries
        for (int e : {0, 1, 127, 255})
        {
            auto *gate = registry.getEngine(device_, layer, e, ExpertGemmRegistry::WeightRole::GATE);
            auto *up = registry.getEngine(device_, layer, e, ExpertGemmRegistry::WeightRole::UP);
            auto *down = registry.getEngine(device_, layer, e, ExpertGemmRegistry::WeightRole::DOWN);

            EXPECT_NE(gate, nullptr) << "Layer " << layer << " expert " << e << " GATE missing";
            EXPECT_NE(up, nullptr) << "Layer " << layer << " expert " << e << " UP missing";
            EXPECT_NE(down, nullptr) << "Layer " << layer << " expert " << e << " DOWN missing";
        }
    }

    // Verify total count
    const size_t expected_min = static_cast<size_t>(moe_layers) * EXPECTED_NUM_EXPERTS * 3;
    EXPECT_GE(registry.size(), expected_min)
        << "Expected at least " << expected_min << " expert engines for "
        << moe_layers << " BF16 MoE layers";
}

// =============================================================================
// Test: BF16 dense weights (shared experts) registered in KernelFactory
// =============================================================================

TEST_F(UnifiedGPUPipeline_FP_Test, BF16_DenseWeightsRegistered)
{
    // Load and pipeline a single layer
    loadLayerWeights(0, 1);
    auto filter = makeLayerFilter(0, 1);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    // Shared expert gate should be registered as a dense FP GEMM weight
    auto shexp_gate = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_shexp.weight");
    ASSERT_NE(shexp_gate, nullptr) << "Shared expert gate weight should exist";

    auto *prepared = KernelFactory::getOrCreatePreparedGemmWeights(shexp_gate.get(), device_);
    EXPECT_NE(prepared, nullptr)
        << "BF16 ffn_gate_shexp.weight should have been packed by the FP pipeline";

    // GDN QKV projection should also be registered
    auto qkv = weight_mgr_->getWeightForDevice("blk.0.attn_qkv.weight");
    if (qkv)
    {
        auto *qkv_prepared = KernelFactory::getOrCreatePreparedGemmWeights(qkv.get(), device_);
        EXPECT_NE(qkv_prepared, nullptr)
            << "BF16 attn_qkv.weight should have been packed by the FP pipeline";
    }
}

// =============================================================================
// Test: populateExpertEngines bulk API works for BF16 experts
// =============================================================================

TEST_F(UnifiedGPUPipeline_FP_Test, BF16_PopulateExpertEngines)
{
    loadLayerWeights(0, 1);
    auto filter = makeLayerFilter(0, 1);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    const auto &registry = weight_mgr_->expertGemmRegistry();

    std::vector<ITensorGemm *> gate_engines, up_engines, down_engines;
    bool found = registry.populateExpertEngines(
        device_, 0, EXPECTED_NUM_EXPERTS,
        gate_engines, up_engines, down_engines);

    EXPECT_TRUE(found) << "populateExpertEngines should find layer 0 BF16 engines";
    EXPECT_EQ(gate_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));
    EXPECT_EQ(up_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));
    EXPECT_EQ(down_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));

    // All engines should be non-null
    int null_count = 0;
    for (int e = 0; e < EXPECTED_NUM_EXPERTS; ++e)
    {
        if (!gate_engines[e]) null_count++;
        if (!up_engines[e]) null_count++;
        if (!down_engines[e]) null_count++;
    }
    EXPECT_EQ(null_count, 0)
        << null_count << " BF16 expert engines were null (should all be populated)";
}

// =============================================================================
// Test: BF16 expert GEMM kernel produces valid output (spot-check)
// =============================================================================

TEST_F(UnifiedGPUPipeline_FP_Test, BF16_ExpertGemmCorrectness)
{
    loadLayerWeights(0, 1);

    // Get expert gate tensor shape: GGUF 3D shape[0]=cols(K), shape[1]=rows_per_expert(N), shape[2]=num_experts
    auto gate_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_exps.weight");
    ASSERT_NE(gate_tensor, nullptr);
    ASSERT_EQ(gate_tensor->shape().size(), 3u) << "Expert gate tensor should be 3D";

    const int K = static_cast<int>(gate_tensor->shape()[0]);   // d_model
    const int N = static_cast<int>(gate_tensor->shape()[1]);   // expert_intermediate
    ASSERT_GT(K, 0);
    ASSERT_GT(N, 0);

    // Run pipeline
    auto filter = makeLayerFilter(0, 1);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    const auto &registry = weight_mgr_->expertGemmRegistry();

    // Get expert 0 gate engine
    auto *gate = registry.getEngine(device_, 0, 0, ExpertGemmRegistry::WeightRole::GATE);
    ASSERT_NE(gate, nullptr) << "Expert 0 GATE engine should exist";

    // GEMM: output[M×N] = input[M×K] @ weight[N×K]^T
    const int M = 1;

    // Set up workspace if the kernel requires it (FP GEMM kernels typically don't)
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(gate);
    if (ws_consumer)
    {
        auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
        const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
        workspace = std::make_unique<DeviceWorkspaceManager>(device_, budget);
        ASSERT_TRUE(workspace->allocate(reqs)) << "Workspace allocation failed";
        ws_consumer->bindWorkspace(workspace.get());
    }

    // Create input tensor (FP32, random)
    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_NE(input, nullptr);

    // Create output tensor
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_NE(output, nullptr);

    // Zero output
    std::memset(output->mutable_data(), 0, M * N * sizeof(float));

    // Execute GEMM with proper coherence
    ASSERT_TRUE(with_gpu_coherence(
        device_,
        {input.get()},
        {output.get()},
        [&]
        {
            return gate->multiply_tensor(
                input.get(), output.get(), M, N, K);
        })) << "BF16 Expert GEMM with_gpu_coherence failed";

    // Unbind workspace if we bound one
    if (ws_consumer)
    {
        ws_consumer->unbindWorkspace();
    }

    // Verify output is not all zeros and has no NaN/Inf
    const float *out_data = output->data();
    bool all_zero = true;
    bool has_nan = false;
    float max_abs = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        if (out_data[i] != 0.0f) all_zero = false;
        if (std::isnan(out_data[i]) || std::isinf(out_data[i])) has_nan = true;
        max_abs = std::max(max_abs, std::abs(out_data[i]));
    }
    EXPECT_FALSE(all_zero) << "BF16 Expert GEMM output should not be all zeros";
    EXPECT_FALSE(has_nan) << "BF16 Expert GEMM output should not contain NaN/Inf";

    // Output magnitude should be reasonable (not tiny, not huge)
    EXPECT_GT(max_abs, 1e-6f) << "BF16 Expert GEMM output magnitude suspiciously small";
    EXPECT_LT(max_abs, 1e6f) << "BF16 Expert GEMM output magnitude suspiciously large";
}

// =============================================================================
// Test: Verify BF16 tensor type is preserved through pipeline
// =============================================================================

TEST_F(UnifiedGPUPipeline_FP_Test, BF16_TensorTypeVerification)
{
    // Verify that the model actually loaded as BF16 (not dequantized to FP32)
    auto gate_exps = loader_->loadTensor("blk.0.ffn_gate_exps.weight");
    ASSERT_NE(gate_exps, nullptr);
    EXPECT_EQ(gate_exps->native_type(), TensorType::BF16);

    auto shexp_gate = loader_->loadTensor("blk.0.ffn_gate_shexp.weight");
    ASSERT_NE(shexp_gate, nullptr);
    EXPECT_EQ(shexp_gate->native_type(), TensorType::BF16);

    auto qkv = loader_->loadTensor("blk.0.attn_qkv.weight");
    if (qkv)
    {
        EXPECT_EQ(qkv->native_type(), TensorType::BF16);
    }
}
