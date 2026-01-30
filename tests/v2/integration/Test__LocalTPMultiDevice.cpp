/**
 * @file Test__LocalTPMultiDevice.cpp
 * @brief Integration tests for real LOCAL TP with MultiDeviceOrchestrator
 *
 * These tests require actual GPU hardware and test the full inference pipeline
 * with tensor parallelism across multiple devices within a single MPI rank.
 *
 * Test categories:
 * 1. Hardware Detection - Verify GPU enumeration and device discovery
 * 2. CUDA LOCAL TP - Tests requiring 2+ CUDA GPUs (NCCL backend)
 * 3. ROCm LOCAL TP - Tests requiring 2+ ROCm GPUs (RCCL backend)
 * 4. Heterogeneous LOCAL TP - Tests requiring 1 CUDA + 1 ROCm (PCIeBAR backend)
 *
 * All tests gracefully skip if hardware requirements are not met.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>

#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

using namespace llaminar2;

// =============================================================================
// Global Test Environment - Initialize DeviceManager once
// =============================================================================

class LocalTPMultiDeviceEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        // Initialize DeviceManager to enumerate all devices
        DeviceManager::instance().initialize(-1);

        auto &devices = DeviceManager::instance().devices();
        std::cout << "\n[LocalTPMultiDeviceEnvironment] DeviceManager initialized with "
                  << devices.size() << " device(s)\n";
    }
};

// Register global environment
::testing::Environment *const local_tp_multi_device_env =
    ::testing::AddGlobalTestEnvironment(new LocalTPMultiDeviceEnvironment);

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPMultiDevice : public ::testing::Test
{
protected:
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    static constexpr float MSE_THRESHOLD = 1e-3f;     ///< MSE threshold for logit comparison
    static constexpr float COSINE_THRESHOLD = 0.999f; ///< Cosine similarity threshold

    void SetUp() override
    {
        // Get GPU counts via BackendManager
#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        if (cuda_backend != nullptr)
        {
            cuda_count_ = cuda_backend->deviceCount();
            for (int i = 0; i < cuda_count_; ++i)
            {
                cuda_names_.push_back(cuda_backend->deviceName(i));
            }
        }
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend != nullptr)
        {
            rocm_count_ = rocm_backend->deviceCount();
            for (int i = 0; i < rocm_count_; ++i)
            {
                rocm_names_.push_back(rocm_backend->deviceName(i));
            }
        }
#endif

        // Log device info
        std::cout << "Test__LocalTPMultiDevice: Found "
                  << cuda_count_ << " CUDA GPU(s), "
                  << rocm_count_ << " ROCm GPU(s)" << std::endl;

        // Check if model exists
        model_available_ = std::ifstream(MODEL_PATH).good();
        if (!model_available_)
        {
            std::cout << "  Model not found: " << MODEL_PATH << std::endl;
        }
    }

    void TearDown() override
    {
        // Synchronize all GPUs
#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        if (cuda_backend != nullptr)
        {
            for (int i = 0; i < cuda_count_; ++i)
            {
                cuda_backend->synchronize(i);
            }
        }
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend != nullptr)
        {
            for (int i = 0; i < rocm_count_; ++i)
            {
                rocm_backend->synchronize(i);
            }
        }
#endif
    }

    // =========================================================================
    // Skip Helpers
    // =========================================================================

    void skipIfNoGPUs()
    {
        if (cuda_count_ == 0 && rocm_count_ == 0)
        {
            GTEST_SKIP() << "No GPUs available";
        }
    }

    void skipIfNoCUDA()
    {
        if (cuda_count_ == 0)
        {
            GTEST_SKIP() << "No CUDA GPUs available";
        }
    }

    void skipIfNoROCm()
    {
        if (rocm_count_ == 0)
        {
            GTEST_SKIP() << "No ROCm GPUs available";
        }
    }

    void skipIfLessThan2CUDA()
    {
        if (cuda_count_ < 2)
        {
            GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_count_;
        }
    }

    void skipIfLessThan2ROCm()
    {
        if (rocm_count_ < 2)
        {
            GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_count_;
        }
    }

    void skipIfNoHeterogeneous()
    {
        if (cuda_count_ == 0 || rocm_count_ == 0)
        {
            GTEST_SKIP() << "Requires 1+ CUDA and 1+ ROCm GPUs for heterogeneous test";
        }
    }

    void skipIfNoModel()
    {
        if (!model_available_)
        {
            GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
        }
    }

    // =========================================================================
    // Model Loading
    // =========================================================================

    std::shared_ptr<ModelContext> loadModel()
    {
        return ModelContext::create(MODEL_PATH, nullptr);
    }

    // =========================================================================
    // Metric Computation
    // =========================================================================

    /**
     * @brief Compute Mean Squared Error between two logit arrays
     */
    static float computeMSE(const float *a, const float *b, size_t count)
    {
        if (count == 0)
            return 0.0f;

        double sum = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += diff * diff;
        }
        return static_cast<float>(sum / static_cast<double>(count));
    }

    /**
     * @brief Compute cosine similarity between two arrays
     */
    static float computeCosineSimilarity(const float *a, const float *b, size_t count)
    {
        if (count == 0)
            return 0.0f;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    /**
     * @brief Find argmax in array
     */
    static int findArgmax(const float *data, size_t count)
    {
        if (count == 0)
            return -1;
        int max_idx = 0;
        float max_val = data[0];
        for (size_t i = 1; i < count; ++i)
        {
            if (data[i] > max_val)
            {
                max_val = data[i];
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }

    // =========================================================================
    // Device Info
    // =========================================================================

    int cuda_count_ = 0;
    int rocm_count_ = 0;
    std::vector<std::string> cuda_names_;
    std::vector<std::string> rocm_names_;
    bool model_available_ = false;
};

// =============================================================================
// Hardware Detection Tests
// =============================================================================

/**
 * @test Verify CUDA device detection and enumeration
 */
TEST_F(Test__LocalTPMultiDevice, DetectsAvailableCUDADevices)
{
    skipIfNoCUDA();

    std::cout << "Detected " << cuda_count_ << " CUDA device(s):" << std::endl;
    for (int i = 0; i < cuda_count_; ++i)
    {
        std::cout << "  CUDA:" << i << " - " << cuda_names_[i] << std::endl;
    }

    // Verify we can create GlobalDeviceAddress for each
    for (int i = 0; i < cuda_count_; ++i)
    {
        auto addr = GlobalDeviceAddress::cuda(i);
        EXPECT_EQ(addr.device_type, DeviceType::CUDA);
        EXPECT_EQ(addr.device_ordinal, i);
        EXPECT_FALSE(addr.isCPU());
        EXPECT_TRUE(addr.isGPU());
    }
}

/**
 * @test Verify ROCm device detection and enumeration
 */
TEST_F(Test__LocalTPMultiDevice, DetectsAvailableROCmDevices)
{
    skipIfNoROCm();

    std::cout << "Detected " << rocm_count_ << " ROCm device(s):" << std::endl;
    for (int i = 0; i < rocm_count_; ++i)
    {
        std::cout << "  ROCm:" << i << " - " << rocm_names_[i] << std::endl;
    }

    // Verify we can create GlobalDeviceAddress for each
    for (int i = 0; i < rocm_count_; ++i)
    {
        auto addr = GlobalDeviceAddress::rocm(i);
        EXPECT_EQ(addr.device_type, DeviceType::ROCm);
        EXPECT_EQ(addr.device_ordinal, i);
        EXPECT_FALSE(addr.isCPU());
        EXPECT_TRUE(addr.isGPU());
    }
}

/**
 * @test Verify detection of heterogeneous GPU setup
 */
TEST_F(Test__LocalTPMultiDevice, DetectsHeterogeneousSetup)
{
    skipIfNoGPUs();

    int total_gpus = cuda_count_ + rocm_count_;
    bool is_heterogeneous = (cuda_count_ > 0 && rocm_count_ > 0);

    std::cout << "GPU Setup: " << total_gpus << " total GPUs" << std::endl;
    std::cout << "  CUDA: " << cuda_count_ << std::endl;
    std::cout << "  ROCm: " << rocm_count_ << std::endl;
    std::cout << "  Heterogeneous: " << (is_heterogeneous ? "YES" : "NO") << std::endl;

    EXPECT_GT(total_gpus, 0);

    if (is_heterogeneous)
    {
        // Verify we can create addresses for both types
        auto cuda_addr = GlobalDeviceAddress::cuda(0);
        auto rocm_addr = GlobalDeviceAddress::rocm(0);
        EXPECT_NE(cuda_addr.device_type, rocm_addr.device_type);
    }
}

// =============================================================================
// CUDA LOCAL TP Tests (require 2+ CUDA GPUs)
// =============================================================================

/**
 * @test Verify LocalTPContext creation with 2 CUDA GPUs
 */
TEST_F(Test__LocalTPMultiDevice, CUDA_TwoGPU_LocalTPContextCreation)
{
    skipIfLessThan2CUDA();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto tp_ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);

    ASSERT_NE(tp_ctx, nullptr) << "Failed to create LocalTPContext";
    EXPECT_EQ(tp_ctx->degree(), 2);
    EXPECT_EQ(tp_ctx->devices().size(), 2u);

    // Verify equal weights by default
    auto weights = tp_ctx->weights();
    EXPECT_NEAR(weights[0], 0.5f, 0.01f);
    EXPECT_NEAR(weights[1], 0.5f, 0.01f);

    std::cout << "Created LocalTPContext with 2 CUDA GPUs, backend="
              << static_cast<int>(tp_ctx->backend()) << std::endl;
}

/**
 * @test Verify forward pass produces output with correct shape
 *
 * NOTE: This test requires MultiDeviceOrchestrator weight sharding to be fully
 * implemented. Currently, device runners are created but weights aren't loaded
 * with proper LOCAL TP sharding, causing kernel device errors.
 *
 * TODO: Enable once MultiDeviceOrchestrator::initializeDeviceRunners() properly
 * loads sharded weights for each device.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_CUDA_TwoGPU_ForwardProducesSameShape)
{
    skipIfLessThan2CUDA();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr) << "Failed to load model";

    // Create multi-device config
    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    config.backend = CollectiveBackendType::NCCL;
    config.max_seq_len = 512;

    // Create orchestrator
    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);

    ASSERT_NE(multi_orch, nullptr);
    EXPECT_EQ(multi_orch->device_count(), 2);

    // Run forward pass with test tokens
    std::vector<int> tokens = {1, 2, 3, 4, 5}; // Simple test sequence
    int seq_len = static_cast<int>(tokens.size());

    bool success = multi_orch->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success) << "Forward pass failed";

    // Verify logits shape
    const float *logits = multi_orch->logits();
    ASSERT_NE(logits, nullptr) << "Logits are null after forward";

    int vocab_size = multi_orch->vocab_size();
    EXPECT_GT(vocab_size, 0) << "Invalid vocab size";

    std::cout << "Forward pass successful: vocab_size=" << vocab_size << std::endl;

    // Verify logits are not all zeros or NaN
    float sum = 0.0f;
    bool has_nan = false;
    for (int i = 0; i < vocab_size; ++i)
    {
        if (std::isnan(logits[i]))
            has_nan = true;
        sum += std::abs(logits[i]);
    }
    EXPECT_FALSE(has_nan) << "Logits contain NaN values";
    EXPECT_GT(sum, 0.0f) << "Logits are all zeros";
}

/**
 * @test Compare multi-device vs single-device logits (should match closely)
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 * See CUDA_TwoGPU_ForwardProducesSameShape for details.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_CUDA_TwoGPU_LogitsMatchSingleDevice)
{
    skipIfLessThan2CUDA();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr) << "Failed to load model";

    // Test tokens
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208}; // "The quick brown fox jumps"
    int seq_len = static_cast<int>(tokens.size());

    // =========================================================================
    // Single-device baseline (CUDA:0)
    // =========================================================================
    InferenceRunnerConfig single_config;
    single_config.max_seq_len = 512;
    single_config.activation_precision = ActivationPrecision::FP32;

    auto single_runner = createInferenceRunner(
        model_ctx, nullptr, DeviceId::cuda(0), single_config);
    ASSERT_NE(single_runner, nullptr) << "Failed to create single-device runner";

    bool single_success = single_runner->forward(tokens.data(), seq_len);
    ASSERT_TRUE(single_success) << "Single-device forward failed";

    const float *single_logits = single_runner->logits();
    ASSERT_NE(single_logits, nullptr);
    int vocab_size = single_runner->vocab_size();

    // Copy single-device logits
    std::vector<float> single_logits_copy(single_logits, single_logits + vocab_size);

    // =========================================================================
    // Multi-device (CUDA:0 + CUDA:1)
    // =========================================================================
    MultiDeviceOrchestrator::Config multi_config;
    multi_config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    multi_config.backend = CollectiveBackendType::NCCL;
    multi_config.max_seq_len = 512;
    multi_config.activation_precision = ActivationPrecision::FP32;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, multi_config);
    ASSERT_NE(multi_orch, nullptr);

    bool multi_success = multi_orch->forward(tokens.data(), seq_len);
    ASSERT_TRUE(multi_success) << "Multi-device forward failed";

    const float *multi_logits = multi_orch->logits();
    ASSERT_NE(multi_logits, nullptr);
    EXPECT_EQ(multi_orch->vocab_size(), vocab_size);

    // =========================================================================
    // Compare logits
    // =========================================================================
    float mse = computeMSE(single_logits_copy.data(), multi_logits, vocab_size);
    float cosine = computeCosineSimilarity(single_logits_copy.data(), multi_logits, vocab_size);

    int single_argmax = findArgmax(single_logits_copy.data(), vocab_size);
    int multi_argmax = findArgmax(multi_logits, vocab_size);

    std::cout << "Logits comparison:" << std::endl;
    std::cout << "  MSE: " << mse << " (threshold: " << MSE_THRESHOLD << ")" << std::endl;
    std::cout << "  Cosine: " << cosine << " (threshold: " << COSINE_THRESHOLD << ")" << std::endl;
    std::cout << "  Argmax single: " << single_argmax << std::endl;
    std::cout << "  Argmax multi: " << multi_argmax << std::endl;

    EXPECT_LT(mse, MSE_THRESHOLD) << "MSE too high";
    EXPECT_GT(cosine, COSINE_THRESHOLD) << "Cosine similarity too low";
    EXPECT_EQ(single_argmax, multi_argmax) << "Argmax mismatch (different predicted token)";
}

/**
 * @test Verify cache clearing works across devices
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_CUDA_TwoGPU_ClearCacheWorks)
{
    skipIfLessThan2CUDA();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    config.backend = CollectiveBackendType::NCCL;
    config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
    ASSERT_NE(multi_orch, nullptr);

    // First forward
    std::vector<int> tokens = {1, 2, 3};
    ASSERT_TRUE(multi_orch->forward(tokens.data(), tokens.size()));
    EXPECT_GT(multi_orch->get_position(), 0);

    // Clear cache
    multi_orch->clear_cache();
    EXPECT_EQ(multi_orch->get_position(), 0) << "Position not reset after clear_cache";

    // Second forward should work
    ASSERT_TRUE(multi_orch->forward(tokens.data(), tokens.size()));
    EXPECT_GT(multi_orch->get_position(), 0);

    std::cout << "Cache clearing verified" << std::endl;
}

/**
 * @test Verify multiple forward passes work correctly
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_CUDA_TwoGPU_MultipleForwardPasses)
{
    skipIfLessThan2CUDA();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    config.backend = CollectiveBackendType::NCCL;
    config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
    ASSERT_NE(multi_orch, nullptr);

    // Run multiple forward passes (simulating decode)
    std::vector<int> prefill_tokens = {785, 3974, 13876}; // "The quick brown"
    ASSERT_TRUE(multi_orch->forward(prefill_tokens.data(), prefill_tokens.size()));

    int vocab_size = multi_orch->vocab_size();
    std::vector<int> generated_tokens;

    for (int step = 0; step < 5; ++step)
    {
        const float *logits = multi_orch->logits();
        ASSERT_NE(logits, nullptr) << "Logits null at step " << step;

        int next_token = findArgmax(logits, vocab_size);
        generated_tokens.push_back(next_token);

        // Single token decode
        ASSERT_TRUE(multi_orch->forward(&next_token, 1))
            << "Forward failed at decode step " << step;
    }

    std::cout << "Generated " << generated_tokens.size() << " tokens:" << std::endl;
    std::cout << "  ";
    for (int tok : generated_tokens)
    {
        std::cout << tok << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(generated_tokens.size(), 5u);
}

// =============================================================================
// ROCm LOCAL TP Tests (require 2+ ROCm GPUs)
// =============================================================================

/**
 * @test Verify LocalTPContext creation with 2 ROCm GPUs
 */
TEST_F(Test__LocalTPMultiDevice, ROCm_TwoGPU_LocalTPContextCreation)
{
    skipIfLessThan2ROCm();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto tp_ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);

    ASSERT_NE(tp_ctx, nullptr) << "Failed to create LocalTPContext";
    EXPECT_EQ(tp_ctx->degree(), 2);

    std::cout << "Created LocalTPContext with 2 ROCm GPUs, backend="
              << static_cast<int>(tp_ctx->backend()) << std::endl;
}

/**
 * @test Verify forward pass produces output with correct shape on ROCm
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_ROCm_TwoGPU_ForwardProducesSameShape)
{
    skipIfLessThan2ROCm();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr) << "Failed to load model";

    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};
    config.backend = CollectiveBackendType::RCCL;
    config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
    ASSERT_NE(multi_orch, nullptr);
    EXPECT_EQ(multi_orch->device_count(), 2);

    std::vector<int> tokens = {1, 2, 3, 4, 5};
    bool success = multi_orch->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Forward pass failed";

    const float *logits = multi_orch->logits();
    ASSERT_NE(logits, nullptr);

    int vocab_size = multi_orch->vocab_size();
    EXPECT_GT(vocab_size, 0);

    std::cout << "ROCm forward pass successful: vocab_size=" << vocab_size << std::endl;
}

/**
 * @test Compare ROCm multi-device vs single-device logits
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_ROCm_TwoGPU_LogitsMatchSingleDevice)
{
    skipIfLessThan2ROCm();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208};
    int seq_len = static_cast<int>(tokens.size());

    // Single-device baseline (ROCm:0)
    InferenceRunnerConfig single_config;
    single_config.max_seq_len = 512;

    auto single_runner = createInferenceRunner(
        model_ctx, nullptr, DeviceId::rocm(0), single_config);
    ASSERT_NE(single_runner, nullptr);

    ASSERT_TRUE(single_runner->forward(tokens.data(), seq_len));
    const float *single_logits = single_runner->logits();
    int vocab_size = single_runner->vocab_size();
    std::vector<float> single_logits_copy(single_logits, single_logits + vocab_size);

    // Multi-device (ROCm:0 + ROCm:1)
    MultiDeviceOrchestrator::Config multi_config;
    multi_config.devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};
    multi_config.backend = CollectiveBackendType::RCCL;
    multi_config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, multi_config);
    ASSERT_NE(multi_orch, nullptr);

    ASSERT_TRUE(multi_orch->forward(tokens.data(), seq_len));
    const float *multi_logits = multi_orch->logits();

    // Compare
    float mse = computeMSE(single_logits_copy.data(), multi_logits, vocab_size);
    float cosine = computeCosineSimilarity(single_logits_copy.data(), multi_logits, vocab_size);
    int single_argmax = findArgmax(single_logits_copy.data(), vocab_size);
    int multi_argmax = findArgmax(multi_logits, vocab_size);

    std::cout << "ROCm logits comparison:" << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    std::cout << "  Cosine: " << cosine << std::endl;

    EXPECT_LT(mse, MSE_THRESHOLD);
    EXPECT_GT(cosine, COSINE_THRESHOLD);
    EXPECT_EQ(single_argmax, multi_argmax);
}

// =============================================================================
// Heterogeneous LOCAL TP Tests (require 1 CUDA + 1 ROCm)
// =============================================================================

/**
 * @test Verify LocalTPContext creation with heterogeneous GPUs
 */
TEST_F(Test__LocalTPMultiDevice, Heterogeneous_LocalTPContextCreation)
{
    skipIfNoHeterogeneous();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    // PCIeBAR is the only backend that supports heterogeneous GPU types
    auto tp_ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);

    ASSERT_NE(tp_ctx, nullptr) << "Failed to create heterogeneous LocalTPContext";
    EXPECT_EQ(tp_ctx->degree(), 2);
    EXPECT_EQ(tp_ctx->backend(), CollectiveBackendType::PCIE_BAR);

    // Verify device types differ
    EXPECT_NE(tp_ctx->devices()[0].device_type, tp_ctx->devices()[1].device_type);

    std::cout << "Created heterogeneous LocalTPContext: CUDA + ROCm" << std::endl;
}

/**
 * @test Verify heterogeneous forward pass succeeds
 *
 * NOTE: Re-enabled to test CUDA/HIP coexistence after CPU reduction workaround.
 */
TEST_F(Test__LocalTPMultiDevice, Heterogeneous_CUDAROCm_ForwardSucceeds)
{
    skipIfNoHeterogeneous();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    config.backend = CollectiveBackendType::PCIE_BAR;
    config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
    ASSERT_NE(multi_orch, nullptr);
    EXPECT_EQ(multi_orch->device_count(), 2);

    std::vector<int> tokens = {1, 2, 3, 4, 5};
    bool success = multi_orch->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Heterogeneous forward pass failed";

    const float *logits = multi_orch->logits();
    ASSERT_NE(logits, nullptr);

    int vocab_size = multi_orch->vocab_size();
    EXPECT_GT(vocab_size, 0);

    // Verify no NaN
    bool has_nan = false;
    for (int i = 0; i < vocab_size; ++i)
    {
        if (std::isnan(logits[i]))
        {
            has_nan = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan) << "Heterogeneous logits contain NaN";

    std::cout << "Heterogeneous forward pass successful" << std::endl;
}

/**
 * @test Verify heterogeneous logits are reasonable (may have larger divergence)
 *
 * Note: Heterogeneous execution may have larger numerical differences due to:
 * - Different GPU architectures (NVIDIA vs AMD)
 * - Different floating-point implementations
 * - Different quantization precision characteristics
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_Heterogeneous_CUDAROCm_LogitsReasonable)
{
    skipIfNoHeterogeneous();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208};
    int seq_len = static_cast<int>(tokens.size());

    // Single-device baseline (CUDA:0)
    InferenceRunnerConfig single_config;
    single_config.max_seq_len = 512;

    auto single_runner = createInferenceRunner(
        model_ctx, nullptr, DeviceId::cuda(0), single_config);
    ASSERT_NE(single_runner, nullptr);

    ASSERT_TRUE(single_runner->forward(tokens.data(), seq_len));
    const float *single_logits = single_runner->logits();
    int vocab_size = single_runner->vocab_size();
    std::vector<float> single_logits_copy(single_logits, single_logits + vocab_size);

    // Heterogeneous (CUDA:0 + ROCm:0)
    MultiDeviceOrchestrator::Config multi_config;
    multi_config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    multi_config.backend = CollectiveBackendType::PCIE_BAR;
    multi_config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, multi_config);
    ASSERT_NE(multi_orch, nullptr);

    ASSERT_TRUE(multi_orch->forward(tokens.data(), seq_len));
    const float *multi_logits = multi_orch->logits();

    // Compare with relaxed thresholds for heterogeneous
    constexpr float HETERO_MSE_THRESHOLD = 1e-2f;     // 10x relaxed
    constexpr float HETERO_COSINE_THRESHOLD = 0.995f; // Slightly relaxed

    float mse = computeMSE(single_logits_copy.data(), multi_logits, vocab_size);
    float cosine = computeCosineSimilarity(single_logits_copy.data(), multi_logits, vocab_size);
    int single_argmax = findArgmax(single_logits_copy.data(), vocab_size);
    int multi_argmax = findArgmax(multi_logits, vocab_size);

    std::cout << "Heterogeneous logits comparison:" << std::endl;
    std::cout << "  MSE: " << mse << " (threshold: " << HETERO_MSE_THRESHOLD << ")" << std::endl;
    std::cout << "  Cosine: " << cosine << " (threshold: " << HETERO_COSINE_THRESHOLD << ")" << std::endl;
    std::cout << "  Argmax CUDA-only: " << single_argmax << std::endl;
    std::cout << "  Argmax heterogeneous: " << multi_argmax << std::endl;

    EXPECT_LT(mse, HETERO_MSE_THRESHOLD) << "Heterogeneous MSE too high";
    EXPECT_GT(cosine, HETERO_COSINE_THRESHOLD) << "Heterogeneous cosine too low";

    // Argmax may differ slightly for heterogeneous, just log it
    if (single_argmax != multi_argmax)
    {
        std::cout << "  NOTE: Argmax differs (expected for heterogeneous execution)" << std::endl;
    }
}

/**
 * @test Verify proportional weights work with heterogeneous setup
 *
 * NOTE: Disabled until MultiDeviceOrchestrator weight sharding is implemented.
 */
TEST_F(Test__LocalTPMultiDevice, DISABLED_Heterogeneous_ProportionalWeights)
{
    skipIfNoHeterogeneous();
    skipIfNoModel();

    auto model_ctx = loadModel();
    ASSERT_NE(model_ctx, nullptr);

    // 73% CUDA, 27% ROCm (common for NVIDIA vs AMD performance ratio)
    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    config.weights = {0.73f, 0.27f};
    config.backend = CollectiveBackendType::PCIE_BAR;
    config.max_seq_len = 512;

    auto multi_orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
    ASSERT_NE(multi_orch, nullptr);

    // Verify weights via TP context
    auto *tp_ctx = multi_orch->localTPContext();
    ASSERT_NE(tp_ctx, nullptr);

    auto weights = tp_ctx->weights();
    EXPECT_NEAR(weights[0], 0.73f, 0.01f);
    EXPECT_NEAR(weights[1], 0.27f, 0.01f);

    // Forward should still work
    std::vector<int> tokens = {1, 2, 3};
    bool success = multi_orch->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success);

    std::cout << "Proportional weights test passed: 73%/27% split" << std::endl;
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

/**
 * @test Verify configuration validation catches invalid weights
 */
TEST_F(Test__LocalTPMultiDevice, ConfigValidation_InvalidWeights)
{
    skipIfLessThan2CUDA();

    MultiDeviceOrchestrator::Config config;
    config.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    // Weights don't sum to 1.0
    config.weights = {0.5f, 0.3f};

    EXPECT_FALSE(config.validate()) << "Should reject weights that don't sum to 1.0";

    // Mismatched count
    config.weights = {0.5f};
    EXPECT_FALSE(config.validate()) << "Should reject mismatched weight count";

    // Valid weights
    config.weights = {0.6f, 0.4f};
    EXPECT_TRUE(config.validate());
}

/**
 * @test Verify empty device list is rejected
 */
TEST_F(Test__LocalTPMultiDevice, ConfigValidation_EmptyDevices)
{
    MultiDeviceOrchestrator::Config config;
    config.devices = {};

    EXPECT_FALSE(config.validate()) << "Should reject empty device list";
}

#endif // HAVE_CUDA || HAVE_ROCM

// =============================================================================
// Main (required for standalone execution)
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
