/**
 * @file Test__MultiDeviceOrchestrator_PP.cpp
 * @brief Integration tests for MultiDeviceOrchestrator Pipeline Parallelism mode
 *
 * Tests the PP mode of MultiDeviceOrchestrator:
 * - PP stage runner creation with layer-partitioned model contexts
 * - Sequential forward execution through stages
 * - Activation transfer between stages via LocalPPContext
 * - Logits output from final stage
 *
 * Note: These tests run on CPU (no GPU required) to verify the orchestration
 * logic. GPU-specific PP tests are in parity test suite.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/config/RuntimeConfig.h"
#include "loaders/ModelContext.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"
#include "collective/ILocalPPContext.h"
#include "kernels/KernelFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// Forward-declare cudaGetDeviceCount to avoid CUDA/HIP header conflicts
#ifdef HAVE_CUDA
extern "C" int cudaGetDeviceCount(int *count);
static constexpr int cudaSuccess_v = 0;
#endif

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MultiDeviceOrchestrator_PP : public ::testing::Test
{
protected:
    static constexpr const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    static constexpr int MAX_SEQ_LEN = 64;
    static constexpr int BATCH_SIZE = 1;
    static constexpr int NUM_LAYERS = 24;

    void SetUp() override
    {
        // Check model exists
        std::ifstream f(TEST_MODEL_PATH);
        if (!f.good())
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }

        // Load full model context for reference
        model_ctx_ = ModelContext::create(TEST_MODEL_PATH);
        ASSERT_NE(model_ctx_, nullptr);
        ASSERT_EQ(model_ctx_->blockCount(), NUM_LAYERS) << "Expected 24 layer model";
    }

    void TearDown() override
    {
        // Destroy orchestrators before clearing caches so GPU memory is freed
        // in the correct order (orchestrator → kernels → device handles).
        model_ctx_.reset();

        // Clear global KernelFactory caches to prevent stale pointer hits
        // when mmap reuses the same virtual addresses across tests.
        llaminar::v2::kernels::KernelFactory::clearCache();
    }

    bool hasCUDADevice() const
    {
#ifdef HAVE_CUDA
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess_v && count > 0;
#else
        return false;
#endif
    }

    int rocmDeviceCount() const
    {
#ifdef HAVE_ROCM
        int count = 0;
        return (hipGetDeviceCount(&count) == hipSuccess) ? count : 0;
#else
        return 0;
#endif
    }

    static float cosineSimilarity(const float *a, const float *b, int n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < n; ++i)
        {
            dot += double(a[i]) * double(b[i]);
            na += double(a[i]) * double(a[i]);
            nb += double(b[i]) * double(b[i]);
        }
        if (na == 0.0 || nb == 0.0)
            return 0.0f;
        return float(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    /**
     * @brief Create PPStageConfig for a given layer range
     */
    MultiDeviceOrchestrator::PPStageConfig createPPStageConfig(
        int first_layer, int last_layer, DeviceId device,
        bool has_embedding = false, bool has_lm_head = false)
    {
        MultiDeviceOrchestrator::PPStageConfig config;
        config.first_layer = first_layer;
        config.last_layer = last_layer;
        config.has_embedding = has_embedding;
        config.has_lm_head = has_lm_head;
        config.stage_devices.push_back(GlobalDeviceAddress::cpu());
        return config;
    }

    /**
     * @brief Create 2-stage PP config on CPU
     */
    MultiDeviceOrchestrator::Config create2StageCPUConfig()
    {
        MultiDeviceOrchestrator::Config config;
        config.max_seq_len = MAX_SEQ_LEN;
        config.batch_size = BATCH_SIZE;
        config.activation_precision = ActivationPrecision::FP32;
        config.mode = MultiDeviceOrchestrator::ParallelismMode::PP;

        // Stage 0: layers [0, 12) with embedding
        config.pp_stages.push_back(createPPStageConfig(
            0, 12, DeviceId::cpu(), true, false));

        // Stage 1: layers [12, 24) with LM head
        config.pp_stages.push_back(createPPStageConfig(
            12, 24, DeviceId::cpu(), false, true));

        return config;
    }

    std::shared_ptr<ModelContext> model_ctx_;
};

// =============================================================================
// PP Configuration Tests
// =============================================================================

/**
 * @test PP mode detection from config
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConfigDetectsMode_PP)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::AUTO;

    // No TP devices, only PP stages
    config.pp_stages.push_back(createPPStageConfig(0, 12, DeviceId::cpu(), true, false));
    config.pp_stages.push_back(createPPStageConfig(12, 24, DeviceId::cpu(), false, true));

    EXPECT_EQ(config.detectMode(), MultiDeviceOrchestrator::ParallelismMode::PP);
}

/**
 * @test 2-stage PP config validates successfully
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_2Stage_IsValid)
{
    auto config = create2StageCPUConfig();
    EXPECT_TRUE(config.validate()) << "2-stage CPU PP config should be valid";
}

/**
 * @test PP config with layer gap fails validation
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_LayerGap_IsInvalid)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::PP;

    // Gap: [0, 10), [12, 24) - missing layers 10-11
    config.pp_stages.push_back(createPPStageConfig(0, 10, DeviceId::cpu(), true, false));
    config.pp_stages.push_back(createPPStageConfig(12, 24, DeviceId::cpu(), false, true));

    EXPECT_FALSE(config.validate()) << "PP config with layer gap should be invalid";
}

/**
 * @test PP config layer boundaries extraction
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_LayerBoundaries)
{
    auto config = create2StageCPUConfig();
    auto boundaries = config.buildLayerBoundaries();

    ASSERT_EQ(boundaries.size(), 3);
    EXPECT_EQ(boundaries[0], 0);  // Start of stage 0
    EXPECT_EQ(boundaries[1], 12); // Start of stage 1
    EXPECT_EQ(boundaries[2], 24); // End of stage 1
}

// =============================================================================
// PP Orchestrator Creation Tests (require model loading)
// =============================================================================

/**
 * @test MultiDeviceOrchestrator construction in PP mode (CPU)
 *
 * This test verifies that the orchestrator can be constructed in PP mode
 * and creates the expected number of stage runners.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConstructionCreatesStageRunners)
{
    auto config = create2StageCPUConfig();

    // Create orchestrator - this should create 2 PP stage runners
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Verify PP mode was selected
    EXPECT_EQ(orchestrator->effectiveMode(), MultiDeviceOrchestrator::ParallelismMode::PP);

    // Verify vocabulary size is available (implies successful initialization)
    EXPECT_GT(orchestrator->vocab_size(), 0) << "Should have valid vocab_size from model";
}

// =============================================================================
// PP Forward Execution Tests (require model loading + inference)
// NOTE: These tests are DISABLED because DeviceGraphOrchestrator::forward()
// currently validates that token_ids is non-null even for PP middle stages
// that receive hidden state input via setHiddenState(). This is a limitation
// in the graph build session validation that needs to be addressed in
// DeviceGraphOrchestrator before these tests can pass.
// =============================================================================

/**
 * @test PP forward produces logits
 *
 * Verifies that forward() in PP mode produces non-null logits output.
 * This is an integration test that exercises the full PP execution path.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Forward_ProducesLogits)
{
    auto config = create2StageCPUConfig();
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Run forward with a simple prompt
    std::vector<int> tokens = {151644, 8948, 198}; // Simple system token sequence
    bool success = orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(success) << "PP forward should succeed";

    // Check logits are available
    const float *logits = orchestrator->logits();
    EXPECT_NE(logits, nullptr) << "PP forward should produce logits";
}

/**
 * @test PP forward vs single-device forward produces similar results
 *
 * This is a smoke test to verify PP doesn't produce wildly different results.
 * More rigorous parity testing is done in the parity test suite.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Forward_SimilarToSingleDevice)
{
    // Create PP orchestrator
    auto pp_config = create2StageCPUConfig();
    auto pp_orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, pp_config);
    ASSERT_NE(pp_orchestrator, nullptr);

    // Create single-device reference with its OWN ModelContext.
    // The PP MDO's weight preparation modifies the shared WeightManager
    // (packing state), so the TP reference needs an independent WM.
    auto ref_model_ctx = ModelContext::create(TEST_MODEL_PATH);
    ASSERT_NE(ref_model_ctx, nullptr);

    MultiDeviceOrchestrator::Config tp_config;
    tp_config.max_seq_len = MAX_SEQ_LEN;
    tp_config.batch_size = BATCH_SIZE;
    tp_config.activation_precision = ActivationPrecision::FP32;
    tp_config.mode = MultiDeviceOrchestrator::ParallelismMode::TP;
    tp_config.devices.push_back(GlobalDeviceAddress::cpu());

    auto tp_orchestrator = std::make_unique<MultiDeviceOrchestrator>(ref_model_ctx, tp_config);
    ASSERT_NE(tp_orchestrator, nullptr);

    // Run same tokens through both
    std::vector<int> tokens = {151644, 8948, 198};

    bool pp_success = pp_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(pp_success);

    bool tp_success = tp_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(tp_success);

    // Compare logits - they should be close (not necessarily identical due to
    // potential numerical differences in activation transfer)
    const float *pp_logits = pp_orchestrator->logits();
    const float *tp_logits = tp_orchestrator->logits();
    ASSERT_NE(pp_logits, nullptr);
    ASSERT_NE(tp_logits, nullptr);

    // Find argmax for both - they should pick the same top token
    int vocab = pp_orchestrator->vocab_size();
    ASSERT_GT(vocab, 0);

    int pp_argmax = 0, tp_argmax = 0;
    float pp_max = pp_logits[0], tp_max = tp_logits[0];
    for (int i = 1; i < vocab; ++i)
    {
        if (pp_logits[i] > pp_max)
        {
            pp_max = pp_logits[i];
            pp_argmax = i;
        }
        if (tp_logits[i] > tp_max)
        {
            tp_max = tp_logits[i];
            tp_argmax = i;
        }
    }

    // They should predict the same top token (strong requirement for correct PP)
    EXPECT_EQ(pp_argmax, tp_argmax)
        << "PP and single-device should predict same top token. "
        << "PP=" << pp_argmax << " TP=" << tp_argmax;
}

// =============================================================================
// PP Cache Clear Tests
// =============================================================================

/**
 * @test PP clear_cache propagates to all stage runners
 *
 * DISABLED: Requires DeviceGraphOrchestrator fix for PP middle stage forward()
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ClearCache_PropagatestoStages)
{
    auto config = create2StageCPUConfig();
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Run forward to populate caches
    std::vector<int> tokens = {151644};
    EXPECT_TRUE(orchestrator->forward(tokens.data(), 1));

    // Clear should not throw
    EXPECT_NO_THROW(orchestrator->clear_cache());

    // Should be able to run forward again after clear
    EXPECT_TRUE(orchestrator->forward(tokens.data(), 1));
}

// =============================================================================
// Heterogeneous PP + TP Tests (CUDA + ROCm TP domain)
// =============================================================================

/**
 * @test TP_PP mode detection: CUDA single-device stage + ROCm 2-device TP stage
 *
 * Verifies that the config auto-detects TP_PP mode when one PP stage
 * has multiple devices (a TP domain).
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConfigDetectsMode_TP_PP)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::AUTO;

    // Stage 0: single CUDA device (layers 0-12)
    MultiDeviceOrchestrator::PPStageConfig stage0;
    stage0.first_layer = 0;
    stage0.last_layer = 12;
    stage0.has_embedding = true;
    stage0.has_lm_head = false;
    stage0.stage_devices.push_back(GlobalDeviceAddress::cuda(0));
    config.pp_stages.push_back(stage0);

    // Stage 1: 2-device ROCm TP domain (layers 12-24)
    MultiDeviceOrchestrator::PPStageConfig stage1;
    stage1.first_layer = 12;
    stage1.last_layer = 24;
    stage1.has_embedding = false;
    stage1.has_lm_head = true;
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(0));
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(1));
    stage1.tp_backend = CollectiveBackendType::RCCL;
    config.pp_stages.push_back(stage1);

    EXPECT_EQ(config.detectMode(), MultiDeviceOrchestrator::ParallelismMode::TP_PP)
        << "One single-device stage + one TP-domain stage should detect TP_PP mode";

    EXPECT_FALSE(stage0.isTPDomain());
    EXPECT_TRUE(stage1.isTPDomain());
}

/**
 * @test Heterogeneous PP+TP construction: CUDA prefix + ROCm TP suffix
 *
 * Creates a MultiDeviceOrchestrator in TP_PP mode:
 * - Stage 0: CUDA:0 runs layers [0, 12) with embedding (single device)
 * - Stage 1: ROCm:0 + ROCm:1 run layers [12, 24) with LM head (TP domain)
 *
 * Stage 1 is a nested MDO in TP mode that shards weights across 2 ROCm devices.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Construction_CUDAPrefix_ROCmTPSuffix)
{
    if (!hasCUDADevice())
        GTEST_SKIP() << "No CUDA device available";
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "Need at least 2 ROCm devices for TP domain";

    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.activation_precision = ActivationPrecision::FP32;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::TP_PP;

    // Stage 0: CUDA single device
    MultiDeviceOrchestrator::PPStageConfig stage0;
    stage0.first_layer = 0;
    stage0.last_layer = NUM_LAYERS / 2;
    stage0.has_embedding = true;
    stage0.has_lm_head = false;
    stage0.stage_devices.push_back(GlobalDeviceAddress::cuda(0));
    config.pp_stages.push_back(stage0);

    // Stage 1: 2-device ROCm TP domain
    MultiDeviceOrchestrator::PPStageConfig stage1;
    stage1.first_layer = NUM_LAYERS / 2;
    stage1.last_layer = NUM_LAYERS;
    stage1.has_embedding = false;
    stage1.has_lm_head = true;
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(0));
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(1));
    stage1.tp_backend = CollectiveBackendType::RCCL;
    config.pp_stages.push_back(stage1);

    EXPECT_TRUE(config.validate()) << "CUDA + ROCm TP PP config should be valid";

    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    EXPECT_EQ(orchestrator->effectiveMode(), MultiDeviceOrchestrator::ParallelismMode::TP_PP);
    EXPECT_GT(orchestrator->vocab_size(), 0);

    std::cout << "CUDA + ROCm(2) TP_PP orchestrator constructed successfully, vocab_size="
              << orchestrator->vocab_size() << std::endl;
}

/**
 * @test Heterogeneous PP+TP forward: CUDA prefix + ROCm TP suffix vs CPU reference
 *
 * End-to-end parity test:
 * - Reference: Single-device CPU full model (24 layers)
 * - Under test: 2-stage PP where CUDA handles first half and ROCm TP domain handles second half
 *
 * Compares final logit distribution via cosine similarity.
 * Expected cosine > 0.95 (quantized GEMM + cross-vendor transfer + TP reductions
 * introduce more numerical divergence than single-vendor tests).
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Forward_CUDAPrefix_ROCmTPSuffix_VsCPU)
{
    if (!hasCUDADevice())
        GTEST_SKIP() << "No CUDA device available";
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "Need at least 2 ROCm devices for TP domain";

    std::vector<int> tokens(32);
    for (int i = 0; i < 32; ++i)
        tokens[i] = i % 1024;

    // =========================================================================
    // Reference: single-device CPU full model
    // Uses its OWN ModelContext — the CPU reference's finalizeForDevices()
    // releases host weight data, which would starve the PP+TP MDO if they
    // shared the same WeightManager.
    // =========================================================================
    std::vector<float> ref_logits_copy;
    int vocab = 0;
    {
        auto ref_model_ctx = ModelContext::create(TEST_MODEL_PATH);
        ASSERT_NE(ref_model_ctx, nullptr);

        MultiDeviceOrchestrator::Config ref_config;
        ref_config.max_seq_len = MAX_SEQ_LEN;
        ref_config.batch_size = BATCH_SIZE;
        ref_config.activation_precision = ActivationPrecision::FP32;
        ref_config.mode = MultiDeviceOrchestrator::ParallelismMode::TP;
        ref_config.devices.push_back(GlobalDeviceAddress::cpu());

        auto ref_orchestrator = std::make_unique<MultiDeviceOrchestrator>(ref_model_ctx, ref_config);
        ASSERT_NE(ref_orchestrator, nullptr);

        ASSERT_TRUE(ref_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size())))
            << "CPU reference forward failed";

        const float *ref_logits = ref_orchestrator->logits();
        ASSERT_NE(ref_logits, nullptr);
        vocab = ref_orchestrator->vocab_size();
        ASSERT_GT(vocab, 0);
        ref_logits_copy.assign(ref_logits, ref_logits + vocab);
    }

    // =========================================================================
    // Under test: CUDA prefix + ROCm TP suffix
    // =========================================================================
    MultiDeviceOrchestrator::Config test_config;
    test_config.max_seq_len = MAX_SEQ_LEN;
    test_config.batch_size = BATCH_SIZE;
    test_config.activation_precision = ActivationPrecision::FP32;
    test_config.mode = MultiDeviceOrchestrator::ParallelismMode::TP_PP;

    // Stage 0: CUDA:0 → layers [0, 12) with embedding
    MultiDeviceOrchestrator::PPStageConfig stage0;
    stage0.first_layer = 0;
    stage0.last_layer = NUM_LAYERS / 2;
    stage0.has_embedding = true;
    stage0.has_lm_head = false;
    stage0.stage_devices.push_back(GlobalDeviceAddress::cuda(0));
    test_config.pp_stages.push_back(stage0);

    // Stage 1: ROCm:0 + ROCm:1 → layers [12, 24) with LM head, TP sharded
    MultiDeviceOrchestrator::PPStageConfig stage1;
    stage1.first_layer = NUM_LAYERS / 2;
    stage1.last_layer = NUM_LAYERS;
    stage1.has_embedding = false;
    stage1.has_lm_head = true;
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(0));
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(1));
    stage1.tp_backend = CollectiveBackendType::RCCL;
    test_config.pp_stages.push_back(stage1);

    auto test_orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, test_config);
    ASSERT_NE(test_orchestrator, nullptr);

    ASSERT_TRUE(test_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size())))
        << "CUDA + ROCm TP forward failed";

    // =========================================================================
    // Compare logits
    // =========================================================================
    const float *test_logits = test_orchestrator->logits();
    ASSERT_NE(test_logits, nullptr);
    ASSERT_EQ(vocab, test_orchestrator->vocab_size());

    float cosine = cosineSimilarity(ref_logits_copy.data(), test_logits, vocab);
    std::cout << "CUDA + ROCm(2) TP_PP vs CPU reference logit cosine: " << cosine << std::endl;

    // Find argmax for both
    int ref_argmax = static_cast<int>(std::distance(ref_logits_copy.begin(),
                                                    std::max_element(ref_logits_copy.begin(), ref_logits_copy.end())));
    int test_argmax = static_cast<int>(std::distance(test_logits, std::max_element(test_logits, test_logits + vocab)));
    std::cout << "  CPU argmax=" << ref_argmax << "  CUDA+ROCm_TP argmax=" << test_argmax << std::endl;

    EXPECT_GT(cosine, 0.95f)
        << "Heterogeneous PP+TP logits should be close to CPU reference "
        << "(cross-vendor quantized GEMM + TP reductions introduce some divergence)";
}

/**
 * @test TP_PP mode detection with all-single-device stages stays PP
 *
 * Verifies that when all PP stages have exactly 1 device, the mode
 * is PP (not TP_PP), even on a heterogeneous CUDA+ROCm config.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConfigDetectsMode_HeterogeneousPP_NotTP_PP)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::AUTO;

    // Stage 0: single CUDA device
    MultiDeviceOrchestrator::PPStageConfig stage0;
    stage0.first_layer = 0;
    stage0.last_layer = 12;
    stage0.has_embedding = true;
    stage0.has_lm_head = false;
    stage0.stage_devices.push_back(GlobalDeviceAddress::cuda(0));
    config.pp_stages.push_back(stage0);

    // Stage 1: single ROCm device (NOT a TP domain)
    MultiDeviceOrchestrator::PPStageConfig stage1;
    stage1.first_layer = 12;
    stage1.last_layer = 24;
    stage1.has_embedding = false;
    stage1.has_lm_head = true;
    stage1.stage_devices.push_back(GlobalDeviceAddress::rocm(0));
    config.pp_stages.push_back(stage1);

    // All stages are single-device → PP mode, not TP_PP
    EXPECT_EQ(config.detectMode(), MultiDeviceOrchestrator::ParallelismMode::PP);
}
