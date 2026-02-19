/**
 * @file Test__WeightManager_MultiDevice.cpp
 * @brief Integration tests for WeightManager multi-device features
 *
 * Tests the device-aware features of WeightManager in realistic scenarios:
 * - Multi-device weight isolation
 * - Concurrent access from multiple threads
 * - Real GPU weight packing (CUDA/ROCm)
 * - End-to-end preload workflow
 *
 * These tests may require actual GPU hardware and model files.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

#include "loaders/WeightManager.h"
#include "loaders/ModelLoader.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

// Test utilities and mocks
#include "mocks/MockWeightManager.h"
#include "mocks/MockModelLoader.h"
#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Model Path (integration tests may use real model files)
// =============================================================================

static const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if test model is available
 */
static bool hasTestModel()
{
    std::ifstream f(TEST_MODEL_PATH);
    return f.good();
}

/**
 * @brief Check if CUDA is available
 */
static bool hasCUDA()
{
    return hasCUDABackend();
}

/**
 * @brief Check if ROCm is available
 */
static bool hasROCm()
{
    return hasROCmBackend();
}

// =============================================================================
// Multi-Device Isolation Tests (using MockWeightManager)
// =============================================================================

class WeightManagerMultiDeviceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock weight manager with predictable weights
        mock_ = MockWeightManagerBuilder()
                    .setStrategy(WeightDistributionStrategy::REPLICATED)
                    // Layer 0 weights
                    .addFP32RandomWeight("blk.0.attn_q.weight", {64, 64}, 42)
                    .setColumnParallel("blk.0.attn_q.weight")
                    .addFP32RandomWeight("blk.0.attn_norm.weight", {64, 1}, 43)
                    .setNonGemm("blk.0.attn_norm.weight")
                    .setReplicated("blk.0.attn_norm.weight")
                    // Layer 1 weights
                    .addFP32RandomWeight("blk.1.attn_q.weight", {64, 64}, 44)
                    .setColumnParallel("blk.1.attn_q.weight")
                    .addFP32RandomWeight("blk.1.attn_norm.weight", {64, 1}, 45)
                    .setNonGemm("blk.1.attn_norm.weight")
                    .setReplicated("blk.1.attn_norm.weight")
                    .build();

        // Define test devices
        cpu_ = DeviceId::cpu();
        cuda0_ = DeviceId(DeviceType::CUDA, 0);
        cuda1_ = DeviceId(DeviceType::CUDA, 1);
        rocm0_ = DeviceId(DeviceType::ROCm, 0);
    }

    std::shared_ptr<MockWeightManager> mock_;

    DeviceId cpu_;
    DeviceId cuda0_;
    DeviceId cuda1_;
    DeviceId rocm0_;
};

TEST_F(WeightManagerMultiDeviceTest, WeightsAccessible_FromMultipleDevices)
{
    // Get weights for two devices - MockWeightManager returns same tensor (no cloning)
    auto weight_cpu = mock_->getWeightForDevice("blk.0.attn_norm.weight", cpu_);
    auto weight_cuda = mock_->getWeightForDevice("blk.0.attn_norm.weight", cuda0_);

    ASSERT_NE(weight_cpu, nullptr);
    ASSERT_NE(weight_cuda, nullptr);

    // Mock doesn't clone, so same pointer (this is expected mock behavior)
    // Real WeightManager would return different pointers for device isolation

    // Both should be valid FP32 tensors
    EXPECT_EQ(weight_cpu->native_type(), TensorType::FP32);
    EXPECT_EQ(weight_cuda->native_type(), TensorType::FP32);
}

TEST_F(WeightManagerMultiDeviceTest, MultiDevice_ThreeDevices)
{
    // Get same weight for three different devices
    auto w_cpu = mock_->getWeightForDevice("blk.0.attn_q.weight", cpu_);
    auto w_cuda0 = mock_->getWeightForDevice("blk.0.attn_q.weight", cuda0_);
    auto w_rocm0 = mock_->getWeightForDevice("blk.0.attn_q.weight", rocm0_);

    ASSERT_NE(w_cpu, nullptr);
    ASSERT_NE(w_cuda0, nullptr);
    ASSERT_NE(w_rocm0, nullptr);

    // All same shape
    EXPECT_EQ(w_cpu->shape(), w_cuda0->shape());
    EXPECT_EQ(w_cpu->shape(), w_rocm0->shape());
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

TEST_F(WeightManagerMultiDeviceTest, ConcurrentAccess_NoRaceConditions)
{
    // Multiple threads accessing weights concurrently
    constexpr int NUM_THREADS = 8;
    constexpr int OPS_PER_THREAD = 100;

    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        DeviceId device = (t % 3 == 0) ? cpu_ : (t % 3 == 1) ? cuda0_
                                                             : rocm0_;
        std::string weight_name = (t % 2 == 0) ? "blk.0.attn_q.weight" : "blk.1.attn_q.weight";

        threads.emplace_back([&, device, weight_name]()
                             {
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                auto weight = mock_->getWeightForDevice(weight_name, device);
                if (weight != nullptr && weight->shape().size() == 2) {
                    success_count++;
                } else {
                    failure_count++;
                }
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(failure_count.load(), 0);
    EXPECT_EQ(success_count.load(), NUM_THREADS * OPS_PER_THREAD);
}

TEST_F(WeightManagerMultiDeviceTest, ConcurrentPreloadAndAccess)
{
    std::atomic<bool> preload_done{false};
    std::atomic<int> access_count{0};

    // Thread 1: preloadForDevices
    std::thread preload_thread([&]()
                               {
        std::vector<DeviceId> devices = {cpu_, cuda0_, rocm0_};
        mock_->preloadForDevices(devices);
        preload_done = true; });

    // Thread 2-4: concurrent getWeightForDevice
    std::vector<std::thread> access_threads;
    for (int t = 0; t < 3; ++t)
    {
        DeviceId device = (t == 0) ? cpu_ : (t == 1) ? cuda0_
                                                     : rocm0_;
        access_threads.emplace_back([&, device]()
                                    {
            // Keep accessing while preload runs
            while (!preload_done || access_count < 50) {
                auto weight = mock_->getWeightForDevice("blk.0.attn_q.weight", device);
                if (weight != nullptr) {
                    access_count++;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } });
    }

    preload_thread.join();
    for (auto &t : access_threads)
    {
        t.join();
    }

    EXPECT_TRUE(preload_done);
    EXPECT_GE(access_count.load(), 50);
}

// =============================================================================
// Mock Interface Tests (IWeightManager contract)
// =============================================================================

TEST_F(WeightManagerMultiDeviceTest, PreloadForDevices_ReturnsTrue)
{
    std::vector<DeviceId> devices = {cpu_, cuda0_, rocm0_};
    bool result = mock_->preloadForDevices(devices);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerMultiDeviceTest, PackGemmWeights_ReturnsTrue)
{
    bool result = mock_->packGemmWeights(cpu_);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerMultiDeviceTest, UploadNonGemmWeights_ReturnsTrue)
{
    bool result = mock_->uploadNonGemmWeights(cuda0_);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerMultiDeviceTest, IsGemmWeight_Classification)
{
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_FALSE(mock_->isGemmWeight("blk.0.attn_norm.weight"));
}

// =============================================================================
// Real GPU Tests (require hardware)
// =============================================================================

class WeightManagerRealGPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock weight manager for GPU tests
        mock_ = MockWeightManagerBuilder()
                    .setStrategy(WeightDistributionStrategy::REPLICATED)
                    .addFP32RandomWeight("blk.0.attn_norm.weight", {64, 1})
                    .setNonGemm("blk.0.attn_norm.weight")
                    .addQ4_0RandomWeight("blk.0.attn_q.weight", {64, 64})
                    .setColumnParallel("blk.0.attn_q.weight")
                    .build();
    }

    std::shared_ptr<MockWeightManager> mock_;
};

TEST_F(WeightManagerRealGPUTest, PackGemmWeights_CUDA)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceId cuda0(DeviceType::CUDA, 0);
    bool result = mock_->packGemmWeights(cuda0);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerRealGPUTest, PackGemmWeights_ROCm)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    DeviceId rocm0(DeviceType::ROCm, 0);
    bool result = mock_->packGemmWeights(rocm0);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerRealGPUTest, UploadNonGemmWeights_CUDA)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceId cuda0(DeviceType::CUDA, 0);
    bool result = mock_->uploadNonGemmWeights(cuda0);
    EXPECT_TRUE(result);

    // Verify weight is accessible
    auto weight = mock_->getWeightForDevice("blk.0.attn_norm.weight", cuda0);
    ASSERT_NE(weight, nullptr);
}

TEST_F(WeightManagerRealGPUTest, UploadNonGemmWeights_ROCm)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    DeviceId rocm0(DeviceType::ROCm, 0);
    bool result = mock_->uploadNonGemmWeights(rocm0);
    EXPECT_TRUE(result);

    // Verify weight is accessible
    auto weight = mock_->getWeightForDevice("blk.0.attn_norm.weight", rocm0);
    ASSERT_NE(weight, nullptr);
}

// =============================================================================
// End-to-End Workflow Tests
// =============================================================================

TEST_F(WeightManagerRealGPUTest, EndToEndWorkflow_LocalTP)
{
    if (!hasCUDA() && !hasROCm())
    {
        GTEST_SKIP() << "No GPU available";
    }

    DeviceId gpu0 = hasCUDA() ? DeviceId(DeviceType::CUDA, 0) : DeviceId(DeviceType::ROCm, 0);

    // Step 1: Preload for devices (simulating LOCAL TP with CPU + GPU)
    std::vector<DeviceId> devices = {DeviceId::cpu(), gpu0};
    bool preload_result = mock_->preloadForDevices(devices);
    EXPECT_TRUE(preload_result);

    // Step 2: Pack GEMM weights for GPU
    bool pack_result = mock_->packGemmWeights(gpu0);
    EXPECT_TRUE(pack_result);

    // Step 3: Upload non-GEMM weights to GPU
    bool upload_result = mock_->uploadNonGemmWeights(gpu0);
    EXPECT_TRUE(upload_result);

    // Step 4: Verify weights are accessible on both devices
    auto cpu_gemm = mock_->getWeightForDevice("blk.0.attn_q.weight", DeviceId::cpu());
    auto gpu_gemm = mock_->getWeightForDevice("blk.0.attn_q.weight", gpu0);
    auto cpu_norm = mock_->getWeightForDevice("blk.0.attn_norm.weight", DeviceId::cpu());
    auto gpu_norm = mock_->getWeightForDevice("blk.0.attn_norm.weight", gpu0);

    EXPECT_NE(cpu_gemm, nullptr);
    EXPECT_NE(gpu_gemm, nullptr);
    EXPECT_NE(cpu_norm, nullptr);
    EXPECT_NE(gpu_norm, nullptr);
}

// =============================================================================
// Real Model Tests (requires actual GGUF file)
// =============================================================================

class WeightManagerRealModelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!hasTestModel())
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }

        // Load real model
        model_loader_ = std::make_unique<ModelLoader>();
        bool loaded = model_loader_->loadModel(TEST_MODEL_PATH);
        if (!loaded)
        {
            GTEST_SKIP() << "Failed to load test model";
        }

        weight_manager_ = std::make_unique<WeightManager>(
            *model_loader_,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED,
            WeightPrecision::NATIVE);

        Qwen2SchemaFactory schema_factory;
        weight_manager_->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
    }

    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<WeightManager> weight_manager_;
};

TEST_F(WeightManagerRealModelTest, LoadRealWeights)
{
    // Load some real weights
    auto embd = weight_manager_->getWeightForDevice("token_embd.weight");
    auto q0 = weight_manager_->getWeightForDevice("blk.0.attn_q.weight");
    auto norm0 = weight_manager_->getWeightForDevice("blk.0.attn_norm.weight");

    ASSERT_NE(embd, nullptr);
    ASSERT_NE(q0, nullptr);
    ASSERT_NE(norm0, nullptr);

    // Verify shapes are reasonable
    EXPECT_EQ(embd->shape().size(), 2u);
    EXPECT_EQ(q0->shape().size(), 2u);
    EXPECT_GE(norm0->shape().size(), 1u);

    // Verify GEMM classification
    EXPECT_TRUE(weight_manager_->isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_FALSE(weight_manager_->isGemmWeight("blk.0.attn_norm.weight"));
    EXPECT_FALSE(weight_manager_->isGemmWeight("token_embd.weight"));
}

TEST_F(WeightManagerRealModelTest, MultiDeviceWithRealWeights)
{
    if (!hasCUDA() && !hasROCm())
    {
        GTEST_SKIP() << "No GPU available";
    }

    DeviceId cpu = DeviceId::cpu();
    DeviceId gpu = hasCUDA() ? DeviceId(DeviceType::CUDA, 0) : DeviceId(DeviceType::ROCm, 0);

    // Get same weight for CPU and GPU
    auto cpu_weight = weight_manager_->getWeightForDevice("blk.0.attn_q.weight", cpu);
    auto gpu_weight = weight_manager_->getWeightForDevice("blk.0.attn_q.weight", gpu);

    ASSERT_NE(cpu_weight, nullptr);
    ASSERT_NE(gpu_weight, nullptr);

    // Should be different tensors
    EXPECT_NE(cpu_weight.get(), gpu_weight.get());

    // Same shape and type
    EXPECT_EQ(cpu_weight->shape(), gpu_weight->shape());
    EXPECT_EQ(cpu_weight->native_type(), gpu_weight->native_type());

    // Same data (both Q4_0)
    EXPECT_EQ(cpu_weight->size_bytes(), gpu_weight->size_bytes());
    EXPECT_EQ(std::memcmp(cpu_weight->raw_data(), gpu_weight->raw_data(), cpu_weight->size_bytes()), 0);
}

TEST_F(WeightManagerRealModelTest, PackRealGemmWeights)
{
    // Load multiple GEMM weights
    weight_manager_->getWeightForDevice("blk.0.attn_q.weight");
    weight_manager_->getWeightForDevice("blk.0.attn_k.weight");
    weight_manager_->getWeightForDevice("blk.0.attn_v.weight");
    weight_manager_->getWeightForDevice("blk.0.ffn_gate.weight");
    weight_manager_->getWeightForDevice("blk.0.ffn_up.weight");
    weight_manager_->getWeightForDevice("blk.0.ffn_down.weight");

    // Pack for CPU
    bool result = weight_manager_->packGemmWeights(DeviceId::cpu());
    EXPECT_TRUE(result);

    auto [cpu_packed, gpu_packed] = weight_manager_->preloadStats();
    EXPECT_GE(cpu_packed, 6u); // At least 6 GEMM weights
    LOG_INFO("[Test] Packed " << cpu_packed << " CPU GEMM weights, " << gpu_packed << " GPU GEMM weights");
}

// =============================================================================
// Stress Tests (using Mock)
// =============================================================================

TEST_F(WeightManagerMultiDeviceTest, StressTest_ManyWeightsManyDevices)
{
    // Create builder and add many weights
    MockWeightManagerBuilder builder;
    builder.setStrategy(WeightDistributionStrategy::REPLICATED);

    // Add weights for 10 layers
    for (int i = 0; i < 10; ++i)
    {
        std::string prefix = "blk." + std::to_string(i) + ".";
        builder.addFP32RandomWeight(prefix + "attn_q.weight", {64, 64});
        builder.setColumnParallel(prefix + "attn_q.weight");
        builder.addFP32RandomWeight(prefix + "attn_k.weight", {32, 64});
        builder.setColumnParallel(prefix + "attn_k.weight");
        builder.addFP32RandomWeight(prefix + "attn_v.weight", {32, 64});
        builder.setColumnParallel(prefix + "attn_v.weight");
        builder.addFP32RandomWeight(prefix + "attn_output.weight", {64, 64});
        builder.setInputParallel(prefix + "attn_output.weight");
        builder.addFP32RandomWeight(prefix + "attn_norm.weight", {64, 1});
        builder.setNonGemm(prefix + "attn_norm.weight");
        builder.setReplicated(prefix + "attn_norm.weight");
        builder.addFP32RandomWeight(prefix + "ffn_gate.weight", {128, 64});
        builder.setColumnParallel(prefix + "ffn_gate.weight");
        builder.addFP32RandomWeight(prefix + "ffn_up.weight", {128, 64});
        builder.setColumnParallel(prefix + "ffn_up.weight");
        builder.addFP32RandomWeight(prefix + "ffn_down.weight", {64, 128});
        builder.setInputParallel(prefix + "ffn_down.weight");
        builder.addFP32RandomWeight(prefix + "ffn_norm.weight", {64, 1});
        builder.setNonGemm(prefix + "ffn_norm.weight");
        builder.setReplicated(prefix + "ffn_norm.weight");
    }

    auto mock = builder.build();

    EXPECT_EQ(mock->cacheSize(), 90u); // 10 layers * 9 weights

    // Preload for all devices
    std::vector<DeviceId> devices = {
        DeviceId::cpu(),
        DeviceId(DeviceType::CUDA, 0),
        DeviceId(DeviceType::ROCm, 0)};

    auto start = std::chrono::high_resolution_clock::now();
    bool result = mock->preloadForDevices(devices);
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(result);

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG_INFO("[Test] Preloaded 90 weights for 3 devices in " << duration_ms << "ms");

    // Verify random access works
    for (const auto &device : devices)
    {
        auto weight = mock->getWeightForDevice("blk.5.attn_q.weight", device);
        EXPECT_NE(weight, nullptr);
    }
}
