/**
 * @file Test__UnifiedGPUPipeline.cpp
 * @brief Integration tests for unified GPU weight pipeline (dense + MoE in one shot)
 *
 * Verifies Phase 2 acceptance criteria:
 * 1. Load a real MoE model (Qwen3.5-35B-A3B Q4_K_XL)
 * 2. Dense weights AND expert weights flow through the same VRAM pool
 * 3. ExpertGemmRegistry is populated for all tested layers/experts
 * 4. Expert GEMM engines produce correct matmul results (spot-check)
 * 5. FA layer weights are packed alongside GDN MoE layers
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

static const char *MOE_MODEL_PATH = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";

// Qwen3.5-35B-A3B has 256 experts per MoE layer
static constexpr int EXPECTED_NUM_EXPERTS = 256;

// We test layers 0-3 inclusive: layers 0,1,2 are GDN, layer 3 is FA (full attention).
// This ensures both GDN and FA weight paths are exercised.
static constexpr int TEST_LAYER_START = 0;
static constexpr int TEST_LAYER_END = 4;

// Layer 3 is the first full-attention layer (full_attention_interval=4)
static constexpr int FIRST_FA_LAYER = 3;

// =============================================================================
// Helpers
// =============================================================================

static bool hasModel()
{
    std::ifstream f(MOE_MODEL_PATH);
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

static int getROCmDeviceCountForTest()
{
#ifdef HAVE_ROCM
    if (auto *backend = getROCmBackend())
        return backend->deviceCount();
#endif
    return 0;
}

static std::vector<DeviceId> firstROCmDevices(int count)
{
    std::vector<DeviceId> devices;
    devices.reserve(static_cast<size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        devices.emplace_back(DeviceType::ROCm, ordinal);
    }
    return devices;
}

static void verifyLayer0ExpertsReadyOnDevices(WeightManager &weight_mgr,
                                              const std::vector<DeviceId> &devices);

// =============================================================================
// Test Fixture
// =============================================================================

class UnifiedGPUPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!hasModel())
        {
            GTEST_SKIP() << "Model not found: " << MOE_MODEL_PATH;
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

        if (!tryLoadModel(*loader_, MOE_MODEL_PATH))
        {
            GTEST_SKIP() << "Failed to load model: " << MOE_MODEL_PATH;
        }

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
            // FA (Full Attention) weights — only present on FA layers
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "attn_q_norm.weight", "attn_k_norm.weight",
            // GDN weights — only present on GDN layers
            "attn_qkv.weight", "attn_gate.weight",
            // Common norms
            "attn_norm.weight", "post_attention_norm.weight",
            // MoE expert weights (present on all layers)
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_inp.weight",
            // Shared expert weights (present on all layers)
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        };

        for (int layer = first_layer; layer < last_layer; ++layer)
        {
            for (const auto &suffix : weight_suffixes)
            {
                std::string name = "blk." + std::to_string(layer) + "." + suffix;
                // Load into cache (returns nullptr for missing tensors, which is fine)
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

    void runFinalizeForRocmDevices(int device_count)
    {
        const int available = getROCmDeviceCountForTest();
        if (available < device_count)
        {
            GTEST_SKIP() << "Need at least " << device_count << " ROCm GPUs, found " << available;
        }

        auto devices = firstROCmDevices(device_count);
        loadLayerWeights(0, 1);

        ASSERT_TRUE(weight_mgr_->finalizeForDevices(devices, /*release_host_data=*/false))
            << "finalizeForDevices should prepare independent GPU pipelines for "
            << device_count << " ROCm devices";

        verifyLayer0ExpertsReadyOnDevices(*weight_mgr_, devices);
    }

    DeviceId device_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> factory_;
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<WeightManager> weight_mgr_;
};

static void verifyLayer0ExpertsReadyOnDevices(WeightManager &weight_mgr,
                                              const std::vector<DeviceId> &devices)
{
    const auto &registry = weight_mgr.expertGemmRegistry();
    for (const auto &device : devices)
    {
        EXPECT_TRUE(registry.hasCompleteLayer(device, 0, EXPECTED_NUM_EXPERTS))
            << "Layer 0 experts should be complete on " << device.to_string();
        EXPECT_NE(registry.getEngine(device, 0, 0, ExpertGemmRegistry::WeightRole::GATE), nullptr)
            << "Missing gate expert 0 on " << device.to_string();
        EXPECT_NE(registry.getEngine(device, 0, 0, ExpertGemmRegistry::WeightRole::UP), nullptr)
            << "Missing up expert 0 on " << device.to_string();
        EXPECT_NE(registry.getEngine(device, 0, 0, ExpertGemmRegistry::WeightRole::DOWN), nullptr)
            << "Missing down expert 0 on " << device.to_string();
    }
}

// =============================================================================
// Test: ExpertGemmRegistry is populated after pipeline
// =============================================================================

TEST_F(UnifiedGPUPipelineTest, FinalizeForDevicesTwoWayRocmPreparesExpertsOnEachGpu)
{
    runFinalizeForRocmDevices(2);
}

TEST_F(UnifiedGPUPipelineTest, FinalizeForDevicesFourWayRocmPreparesExpertsOnEachGpu)
{
    runFinalizeForRocmDevices(4);
}

TEST_F(UnifiedGPUPipelineTest, RegistryPopulatedForAllExperts)
{
    // Load weights for test layers
    loadLayerWeights(TEST_LAYER_START, TEST_LAYER_END);

    // Run the unified GPU pipeline with layer filter
    auto filter = makeLayerFilter(TEST_LAYER_START, TEST_LAYER_END);
    bool ok = weight_mgr_->packGemmWeightsViaPipeline(device_, filter);
    ASSERT_TRUE(ok) << "packGemmWeightsViaPipeline failed";

    // Verify ExpertGemmRegistry is populated
    const auto &registry = weight_mgr_->expertGemmRegistry();

    // Each MoE layer should have 256 experts × 3 roles = 768 engines
    const int expected_per_layer = EXPECTED_NUM_EXPERTS * 3;
    const int total_moe_layers = TEST_LAYER_END - TEST_LAYER_START;
    const size_t expected_total = static_cast<size_t>(expected_per_layer) * total_moe_layers;

    EXPECT_GE(registry.size(), expected_total)
        << "Registry should have at least " << expected_total
        << " entries (" << total_moe_layers << " layers × " << EXPECTED_NUM_EXPERTS
        << " experts × 3 roles)";

    // Verify each layer has engines registered
    for (int layer = TEST_LAYER_START; layer < TEST_LAYER_END; ++layer)
    {
        EXPECT_TRUE(registry.hasEnginesForLayer(device_, layer))
            << "Layer " << layer << " should have engines registered";

        // Spot-check a few experts
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
}

// =============================================================================
// Test: Dense and MoE weights in same pipeline invocation
// =============================================================================

TEST_F(UnifiedGPUPipelineTest, DenseAndMoEInSamePipeline)
{
    // Load weights for test layers
    loadLayerWeights(TEST_LAYER_START, TEST_LAYER_END);

    auto layer0_gate_exps = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_exps.weight");
    auto layer1_gate_exps = weight_mgr_->getWeightForDevice("blk.1.ffn_gate_exps.weight");
    ASSERT_NE(layer0_gate_exps, nullptr);
    ASSERT_NE(layer1_gate_exps, nullptr);
    ASSERT_FALSE(layer0_gate_exps->is_raw_data_released());
    ASSERT_FALSE(layer1_gate_exps->is_raw_data_released());

    // Run pipeline for layer 0 only
    auto filter = makeLayerFilter(0, 1);
    bool ok = weight_mgr_->packGemmWeightsViaPipeline(device_, filter);
    ASSERT_TRUE(ok) << "packGemmWeightsViaPipeline failed";

    // Verify dense GEMM weights are packed (shared expert gate is a dense GEMM weight)
    auto shexp_gate = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_shexp.weight");
    ASSERT_NE(shexp_gate, nullptr) << "Shared expert gate weight should exist";

    // Dense weights should have a PreparedGemmWeights entry
    auto *prepared = KernelFactory::findPreparedGemmWeights(shexp_gate.get(), device_);
    EXPECT_NE(prepared, nullptr)
        << "Dense ffn_gate_shexp.weight should have been packed by the pipeline";

    // MoE expert weights should also be registered
    const auto &registry = weight_mgr_->expertGemmRegistry();
    EXPECT_TRUE(registry.hasEnginesForLayer(device_, 0))
        << "MoE experts for layer 0 should be registered alongside dense weights";
    EXPECT_TRUE(registry.hasCompleteRole(device_, 0, EXPECTED_NUM_EXPERTS,
                                         ExpertGemmRegistry::WeightRole::GATE))
        << "Layer 0 gate experts should be complete after the layer 0 pipeline run";
    EXPECT_FALSE(registry.hasCompleteRole(device_, 1, EXPECTED_NUM_EXPERTS,
                                          ExpertGemmRegistry::WeightRole::GATE))
        << "Layer 1 gate experts should not be complete after a layer 0-only pipeline run";

    weight_mgr_->markMaterializationComplete();
    weight_mgr_->markDevicePreparationComplete();
    weight_mgr_->markGraphMaterializationComplete();
    const size_t released = weight_mgr_->releaseAllHostWeightData();
    EXPECT_GT(released, 0u) << "Expected prepared layer 0 host data to be released";
    EXPECT_FALSE(layer1_gate_exps->is_raw_data_released())
        << "Unprepared layer 1 MoE parent must be retained despite layer 0 registry entries";
}

// =============================================================================
// Test: populateExpertEngines returns consistent data for graph builder
// =============================================================================

TEST_F(UnifiedGPUPipelineTest, PopulateExpertEnginesForGraphBuilder)
{
    // Load and pipeline
    loadLayerWeights(TEST_LAYER_START, TEST_LAYER_END);
    auto filter = makeLayerFilter(TEST_LAYER_START, TEST_LAYER_END);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    const auto &registry = weight_mgr_->expertGemmRegistry();

    // Use the bulk populateExpertEngines API (used by graph builders)
    std::vector<ITensorGemm *> gate_engines, up_engines, down_engines;
    bool found = registry.populateExpertEngines(
        device_, 0, EXPECTED_NUM_EXPERTS,
        gate_engines, up_engines, down_engines);

    EXPECT_TRUE(found) << "populateExpertEngines should find layer 0 engines";
    EXPECT_EQ(gate_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));
    EXPECT_EQ(up_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));
    EXPECT_EQ(down_engines.size(), static_cast<size_t>(EXPECTED_NUM_EXPERTS));

    // All engines should be non-null
    int null_count = 0;
    for (int e = 0; e < EXPECTED_NUM_EXPERTS; ++e)
    {
        if (!gate_engines[e])
            ++null_count;
        if (!up_engines[e])
            ++null_count;
        if (!down_engines[e])
            ++null_count;
    }
    EXPECT_EQ(null_count, 0)
        << "All " << EXPECTED_NUM_EXPERTS << " × 3 expert engines should be non-null";
}

// =============================================================================
// Test: FA (Full Attention) layer weights are packed alongside MoE
// =============================================================================

TEST_F(UnifiedGPUPipelineTest, FALayerAttentionWeightsPacked)
{
    // Load weights including FA layer 3
    loadLayerWeights(TEST_LAYER_START, TEST_LAYER_END);

    // Run pipeline for all test layers (includes GDN layers 0-2 + FA layer 3)
    auto filter = makeLayerFilter(TEST_LAYER_START, TEST_LAYER_END);
    bool ok = weight_mgr_->packGemmWeightsViaPipeline(device_, filter);
    ASSERT_TRUE(ok) << "packGemmWeightsViaPipeline failed";

    // Verify FA attention GEMM weights are packed for layer 3
    const std::vector<std::string> fa_weights = {
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
    };

    for (const auto &name : fa_weights)
    {
        auto tensor = weight_mgr_->getWeightForDevice(name);
        ASSERT_NE(tensor, nullptr) << "FA weight " << name << " should exist";

        auto *prepared = KernelFactory::findPreparedGemmWeights(tensor.get(), device_);
        EXPECT_NE(prepared, nullptr)
            << "FA weight " << name << " should have been packed by the pipeline";
    }

    // FA layer should ALSO have MoE expert engines registered
    const auto &registry = weight_mgr_->expertGemmRegistry();
    EXPECT_TRUE(registry.hasEnginesForLayer(device_, FIRST_FA_LAYER))
        << "FA layer " << FIRST_FA_LAYER << " should also have MoE expert engines";

    // Spot-check an expert on the FA layer
    auto *gate = registry.getEngine(device_, FIRST_FA_LAYER, 0, ExpertGemmRegistry::WeightRole::GATE);
    EXPECT_NE(gate, nullptr) << "FA layer expert 0 GATE should be registered";
}

// =============================================================================
// Test: Expert GEMM engine produces correct matmul (spot-check)
// =============================================================================

TEST_F(UnifiedGPUPipelineTest, ExpertGemmProducesCorrectOutput)
{
    // Load layer 0 and get expert weight dimensions BEFORE pipeline
    loadLayerWeights(0, 1);

    // Get expert gate tensor shape: GGUF 3D shape[0]=cols(K), shape[1]=rows_per_expert(N), shape[2]=num_experts
    auto gate_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_exps.weight");
    ASSERT_NE(gate_tensor, nullptr);
    ASSERT_EQ(gate_tensor->shape().size(), 3u) << "Expert gate tensor should be 3D";

    const int K = static_cast<int>(gate_tensor->shape()[0]);   // d_model (2048)
    const int N = static_cast<int>(gate_tensor->shape()[1]);   // expert_intermediate (512)
    ASSERT_GT(K, 0);
    ASSERT_GT(N, 0);

    // Run pipeline
    auto filter = makeLayerFilter(0, 1);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    const auto &registry = weight_mgr_->expertGemmRegistry();

    // Get one expert GATE engine for layer 0, expert 0
    auto *gate_engine = registry.getEngine(
        device_, 0, 0, ExpertGemmRegistry::WeightRole::GATE);
    ASSERT_NE(gate_engine, nullptr) << "Expert 0 GATE engine should exist";

    // GEMM: output[M×N] = input[M×K] @ weight[N×K]^T
    const int M = 1; // single token

    // ROCm kernels require workspace binding before execution.
    // Cast to IWorkspaceConsumer to set up workspace.
    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(gate_engine);
    ASSERT_NE(ws_consumer, nullptr) << "GEMM kernel should implement IWorkspaceConsumer";

    auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
    const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device_, budget);
    ASSERT_TRUE(workspace->allocate(reqs)) << "Workspace allocation failed";
    ws_consumer->bindWorkspace(workspace.get());

    // Create input tensor (FP32, random)
    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_NE(input, nullptr);

    // Create output tensor
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_NE(output, nullptr);

    // Zero output
    std::memset(output->mutable_data(), 0, M * N * sizeof(float));

    // Execute GEMM with proper coherence (upload input, mark output device-dirty)
    ASSERT_TRUE(with_gpu_coherence(
        device_,
        {input.get()},
        {output.get()},
        [&]
        {
            return gate_engine->multiply_tensor(
                input.get(), output.get(), M, N, K);
        })) << "Expert GEMM with_gpu_coherence failed";

    // Unbind workspace
    ws_consumer->unbindWorkspace();

    // Verify output is non-zero (a random input × real weights should produce non-zero output)
    const float *out_data = output->data();
    float sum_abs = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        sum_abs += std::fabs(out_data[i]);
    }
    EXPECT_GT(sum_abs, 0.0f)
        << "Expert GEMM output should be non-zero for random input";

    // Verify no NaN/Inf
    bool has_nan_inf = false;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::isnan(out_data[i]) || std::isinf(out_data[i]))
        {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Expert GEMM output should not contain NaN or Inf";
}
