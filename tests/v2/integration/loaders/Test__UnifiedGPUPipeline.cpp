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
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../../utils/TestModelHelper.h"
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/loaders/PreparedWeightStore.h"
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

/// @brief Returns cosine similarity for dense FP32 result comparisons.
static double cosineSimilarity(const float *actual, const float *reference, size_t count)
{
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;
    for (size_t elem_idx = 0; elem_idx < count; ++elem_idx)
    {
        dot += static_cast<double>(actual[elem_idx]) * reference[elem_idx];
        actual_norm += static_cast<double>(actual[elem_idx]) * actual[elem_idx];
        reference_norm += static_cast<double>(reference[elem_idx]) * reference[elem_idx];
    }
    if (actual_norm < 1.0e-30 || reference_norm < 1.0e-30)
        return 0.0;
    return dot / (std::sqrt(actual_norm) * std::sqrt(reference_norm));
}

/// @brief Returns relative L2 error for dense FP32 result comparisons.
static double relativeL2Error(const float *actual, const float *reference, size_t count)
{
    double error_sq = 0.0;
    double reference_sq = 0.0;
    for (size_t elem_idx = 0; elem_idx < count; ++elem_idx)
    {
        const double diff = static_cast<double>(actual[elem_idx]) - reference[elem_idx];
        error_sq += diff * diff;
        reference_sq += static_cast<double>(reference[elem_idx]) * reference[elem_idx];
    }
    if (reference_sq < 1.0e-30)
        return 0.0;
    return std::sqrt(error_sq / reference_sq);
}

/// @brief Returns maximum absolute difference for diagnostic logging.
static double maxAbsDiff(const float *actual, const float *reference, size_t count)
{
    double max_diff = 0.0;
    for (size_t elem_idx = 0; elem_idx < count; ++elem_idx)
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(actual[elem_idx]) - reference[elem_idx]));
    return max_diff;
}

/// @brief Checks whether a dense FP32 tensor buffer contains non-finite values.
static bool hasNaNOrInf(const float *data, size_t count)
{
    for (size_t elem_idx = 0; elem_idx < count; ++elem_idx)
    {
        if (std::isnan(data[elem_idx]) || std::isinf(data[elem_idx]))
            return true;
    }
    return false;
}

/// @brief Computes input[M,K] x weight[N,K]^T using the tensor's dequantized FP32 weights.
static std::vector<float> computeDenseMatmulReference(
    const float *input,
    const float *weights,
    int rows,
    int output_cols,
    int input_cols)
{
    std::vector<float> reference(static_cast<size_t>(rows) * output_cols, 0.0f);
    for (int row_idx = 0; row_idx < rows; ++row_idx)
    {
        const float *input_row = input + static_cast<size_t>(row_idx) * input_cols;
        for (int output_idx = 0; output_idx < output_cols; ++output_idx)
        {
            const float *weight_row = weights + static_cast<size_t>(output_idx) * input_cols;
            double accum = 0.0;
            for (int input_idx = 0; input_idx < input_cols; ++input_idx)
                accum += static_cast<double>(input_row[input_idx]) * weight_row[input_idx];
            reference[static_cast<size_t>(row_idx) * output_cols + output_idx] = static_cast<float>(accum);
        }
    }
    return reference;
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

static void verifyExpertGemmAgainstCpuReference(ITensorGemm *engine,
                                                TensorBase *parent_weight,
                                                DeviceId device,
                                                int expert_id,
                                                int rows,
                                                int output_cols,
                                                int input_cols,
                                                uint32_t input_seed,
                                                const std::string &label);

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
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            // GDN weights — only present on GDN layers
            "attn_qkv.weight",
            "attn_gate.weight",
            // Common norms
            "attn_norm.weight",
            "post_attention_norm.weight",
            // MoE expert weights (present on all layers)
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp.weight",
            // Shared expert weights (present on all layers)
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
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

static void verifyExpertGemmAgainstCpuReference(ITensorGemm *engine,
                                                TensorBase *parent_weight,
                                                DeviceId device,
                                                int expert_id,
                                                int rows,
                                                int output_cols,
                                                int input_cols,
                                                uint32_t input_seed,
                                                const std::string &label)
{
    SCOPED_TRACE(label + " expert=" + std::to_string(expert_id));
    ASSERT_NE(engine, nullptr);
    ASSERT_NE(parent_weight, nullptr);

    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(engine);
    ASSERT_NE(workspace_consumer, nullptr) << "GEMM kernel should implement IWorkspaceConsumer";

    auto requirements = workspace_consumer->getWorkspaceRequirements(rows, output_cols, input_cols);
    const size_t budget = requirements.total_bytes_with_alignment() + (8 * 1024 * 1024);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, budget);
    ASSERT_TRUE(workspace->allocate(requirements)) << "Workspace allocation failed for " << label;
    workspace_consumer->bindWorkspace(workspace.get());

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(rows), static_cast<size_t>(input_cols)}, -0.25f, 0.25f, input_seed);
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(output_cols)});
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    const size_t expert_offset = static_cast<size_t>(expert_id) * output_cols * input_cols;
    auto expert_view = parent_weight->create_view(
        {static_cast<size_t>(output_cols), static_cast<size_t>(input_cols)}, expert_offset);
    ASSERT_NE(expert_view, nullptr);

    const std::vector<float> reference = computeDenseMatmulReference(
        input->data(), expert_view->data(), rows, output_cols, input_cols);

    ASSERT_TRUE(with_gpu_coherence(
        device,
        {input.get()},
        {output.get()},
        [&]
        {
            return engine->multiply_tensor(input.get(), output.get(), rows, output_cols, input_cols);
        }))
        << "Expert GEMM execution failed for " << label;
    workspace_consumer->unbindWorkspace();

    const float *actual = output->data();
    ASSERT_FALSE(hasNaNOrInf(actual, reference.size()));

    const double cosine = cosineSimilarity(actual, reference.data(), reference.size());
    const double rel_l2 = relativeL2Error(actual, reference.data(), reference.size());
    const double max_abs = maxAbsDiff(actual, reference.data(), reference.size());

    EXPECT_GE(cosine, 0.995) << "relative L2=" << rel_l2 << " max_abs=" << max_abs;
    EXPECT_LE(rel_l2, 0.08) << "cosine=" << cosine << " max_abs=" << max_abs;

    std::cout << "[UnifiedGPUPipelineTest.RealExpertGemm] " << label
              << " expert=" << expert_id
              << " cosine=" << std::fixed << std::setprecision(6) << cosine
              << " rel_l2=" << rel_l2
              << " max_abs=" << max_abs << std::endl;
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

    // Dense weights should have been prepared by the pipeline. Graph-time lookup
    // is intentionally binding-id based; this tensor flag only verifies that the
    // pipeline attached prepared state to the resident weight object.
    EXPECT_TRUE(shexp_gate->hasPreparedDeviceState())
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

        EXPECT_TRUE(tensor->hasPreparedDeviceState())
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
    const bool full_pack_replay = std::getenv("LLAMINAR_QWEN35_MOE_REPLAY_FULL_PACK") &&
                                  std::string(std::getenv("LLAMINAR_QWEN35_MOE_REPLAY_FULL_PACK")) == "1";
    if (full_pack_replay)
        loadLayerWeights(0, static_cast<int>(loader_->blockCount()));
    else
        loadLayerWeights(0, 1);

    // Get expert gate tensor shape: GGUF 3D shape[0]=cols(K), shape[1]=rows_per_expert(N), shape[2]=num_experts
    auto gate_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_exps.weight");
    ASSERT_NE(gate_tensor, nullptr);
    ASSERT_EQ(gate_tensor->shape().size(), 3u) << "Expert gate tensor should be 3D";

    const int K = static_cast<int>(gate_tensor->shape()[0]); // d_model (2048)
    const int N = static_cast<int>(gate_tensor->shape()[1]); // expert_intermediate (512)
    ASSERT_GT(K, 0);
    ASSERT_GT(N, 0);

    // Run pipeline
    if (full_pack_replay)
        ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, nullptr));
    else
    {
        auto filter = makeLayerFilter(0, 1);
        ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));
    }

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
        }))
        << "Expert GEMM with_gpu_coherence failed";

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

TEST_F(UnifiedGPUPipelineTest, Layer0RouteTableExpertGEMMsMatchCpuDequantReference)
{
    if (!hasROCmBackend())
    {
        GTEST_SKIP() << "This diagnostic targets the ROCm packed expert path";
    }
    device_ = DeviceId(DeviceType::ROCm, 0);

    loadLayerWeights(0, 1);

    auto gate_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_gate_exps.weight");
    auto up_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_up_exps.weight");
    auto down_tensor = weight_mgr_->getWeightForDevice("blk.0.ffn_down_exps.weight");
    ASSERT_NE(gate_tensor, nullptr);
    ASSERT_NE(up_tensor, nullptr);
    ASSERT_NE(down_tensor, nullptr);
    ASSERT_EQ(gate_tensor->shape().size(), 3u);
    ASSERT_EQ(up_tensor->shape().size(), 3u);
    ASSERT_EQ(down_tensor->shape().size(), 3u);

    const int d_model = static_cast<int>(gate_tensor->shape()[0]);
    const int intermediate = static_cast<int>(gate_tensor->shape()[1]);
    ASSERT_EQ(static_cast<int>(gate_tensor->shape()[2]), EXPECTED_NUM_EXPERTS);
    ASSERT_EQ(static_cast<int>(up_tensor->shape()[0]), d_model);
    ASSERT_EQ(static_cast<int>(up_tensor->shape()[1]), intermediate);
    ASSERT_EQ(static_cast<int>(down_tensor->shape()[0]), intermediate);
    ASSERT_EQ(static_cast<int>(down_tensor->shape()[1]), d_model);

    auto filter = makeLayerFilter(0, 1);
    ASSERT_TRUE(weight_mgr_->packGemmWeightsViaPipeline(device_, filter));

    const auto &registry = weight_mgr_->expertGemmRegistry();
    const std::vector<int> selected_experts = {112, 181, 251, 250};
    constexpr int rows = 3;

    for (int expert_id : selected_experts)
    {
        verifyExpertGemmAgainstCpuReference(
            registry.getEngine(device_, 0, expert_id, ExpertGemmRegistry::WeightRole::GATE),
            gate_tensor.get(), device_, expert_id, rows, intermediate, d_model,
            static_cast<uint32_t>(41000 + expert_id), "layer0_gate");
        verifyExpertGemmAgainstCpuReference(
            registry.getEngine(device_, 0, expert_id, ExpertGemmRegistry::WeightRole::UP),
            up_tensor.get(), device_, expert_id, rows, intermediate, d_model,
            static_cast<uint32_t>(42000 + expert_id), "layer0_up");
        verifyExpertGemmAgainstCpuReference(
            registry.getEngine(device_, 0, expert_id, ExpertGemmRegistry::WeightRole::DOWN),
            down_tensor.get(), device_, expert_id, rows, d_model, intermediate,
            static_cast<uint32_t>(43000 + expert_id), "layer0_down");
    }
}
