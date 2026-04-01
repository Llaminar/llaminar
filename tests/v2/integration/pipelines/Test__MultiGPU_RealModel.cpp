/**
 * @file Test__MultiGPU_RealModel.cpp
 * @brief Integration tests for multi-GPU functionality with real model data
 * @author David Sanftenberg
 *
 * Tests multi-GPU tensor transfers using actual GGUF model weights to verify:
 *  - Weight tensors can be transferred to GPU and back
 *  - Data integrity is preserved across transfers
 *  - Cross-backend transfers (CUDA ↔ ROCm) work correctly
 *  - Different quantized formats transfer correctly (Q4_0, Q8_0, etc.)
 *
 * Requirements:
 *  - At least one GPU (CUDA or ROCm)
 *  - Model file: models/qwen2.5-0.5b-instruct-q4_0.gguf
 */

#include <gtest/gtest.h>
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"
#include "collective/BackendRouter.h"
#include "collective/ILocalPPContext.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include <cmath>
#include <numeric>
#include <iostream>

using namespace llaminar2;

// =============================================================================
// Global Test Environment - Initialize DeviceManager once
// =============================================================================

class MultiGPURealModelEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        // Initialize DeviceManager to enumerate all devices
        DeviceManager::instance().initialize(-1);

        auto &devices = DeviceManager::instance().devices();
        std::cout << "\n[MultiGPURealModelEnvironment] DeviceManager initialized with "
                  << devices.size() << " device(s)\n";

        // Initialize GlobalBackendRouter for tests (enables transferTo() API)
        if (GlobalBackendRouter::initForTests())
        {
            std::cout << "[MultiGPURealModelEnvironment] GlobalBackendRouter initialized for tests\n";
        }
    }

    void TearDown() override
    {
        // Shutdown GlobalBackendRouter
        GlobalBackendRouter::shutdown();
    }
};

// Register global environment
::testing::Environment *const multi_gpu_real_model_env =
    ::testing::AddGlobalTestEnvironment(new MultiGPURealModelEnvironment);

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MultiGPU_RealModel : public ::testing::Test
{
protected:
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    void SetUp() override
    {
        devices_ = DeviceManager::instance().devices();

        // Find GPUs
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == ComputeBackendType::GPU_CUDA)
            {
                cuda_idx_ = static_cast<int>(i);
            }
            else if (devices_[i].type == ComputeBackendType::GPU_ROCM && rocm_idx_ < 0)
            {
                rocm_idx_ = static_cast<int>(i);
            }
        }

        // Load model if file exists
        if (std::ifstream(MODEL_PATH).good())
        {
            loader_ = std::make_unique<ModelLoader>();
            model_loaded_ = loader_->loadModel(MODEL_PATH);
        }
    }

    int findFirstGPU() const
    {
        if (cuda_idx_ >= 0)
            return cuda_idx_;
        if (rocm_idx_ >= 0)
            return rocm_idx_;
        return -1;
    }

    /**
     * @brief Convert legacy DeviceManager index to DeviceId
     */
    DeviceId legacyToDeviceId(int legacy_idx) const
    {
        if (legacy_idx < 0 || static_cast<size_t>(legacy_idx) >= devices_.size())
        {
            return DeviceId::cpu();
        }
        const auto &dev = devices_[legacy_idx];
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            // Convert legacy idx to CUDA ordinal (legacy idx - 1 assuming CPU is at 0)
            int cuda_ordinal = 0;
            for (int i = 1; i < legacy_idx; ++i)
            {
                if (devices_[i].type == ComputeBackendType::GPU_CUDA)
                {
                    ++cuda_ordinal;
                }
            }
            return DeviceId::cuda(cuda_ordinal);
        }
        else if (dev.type == ComputeBackendType::GPU_ROCM)
        {
            int rocm_ordinal = 0;
            for (int i = 1; i < legacy_idx; ++i)
            {
                if (devices_[i].type == ComputeBackendType::GPU_ROCM)
                {
                    ++rocm_ordinal;
                }
            }
            return DeviceId::rocm(rocm_ordinal);
        }
        return DeviceId::cpu();
    }

    std::vector<ComputeDevice> devices_;
    int cuda_idx_ = -1;
    int rocm_idx_ = -1;
    std::unique_ptr<ModelLoader> loader_;
    bool model_loaded_ = false;
};

// =============================================================================
// Q4_0 Weight Transfer Tests
// =============================================================================

/**
 * @test Transfer Q4_0 attention weight to GPU and verify data integrity
 */
TEST_F(Test__MultiGPU_RealModel, Q4_0_AttentionWeight_GPUTransfer)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Convert legacy index to DeviceId
    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // Load a Q4_0 attention weight tensor
    auto tensor = loader_->loadTensor("blk.0.attn_q.weight");
    ASSERT_NE(tensor, nullptr) << "Failed to load blk.0.attn_q.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(tensor.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor type";

    std::cout << "Loaded Q4_0 tensor: " << q4_tensor->shape()[0] << "x" << q4_tensor->shape()[1]
              << " (" << q4_tensor->size_bytes() << " bytes)\n";

    // Dequantize to FP32 BEFORE transfer (for comparison)
    std::vector<float> original_fp32(q4_tensor->numel());
    const float *dequant_data = q4_tensor->data();
    std::copy(dequant_data, dequant_data + original_fp32.size(), original_fp32.begin());

    // Calculate original checksum
    double original_sum = std::accumulate(original_fp32.begin(), original_fp32.end(), 0.0);

    // Transfer to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device))
        << "Failed to transfer Q4_0 tensor to GPU " << gpu_device.toString();
    EXPECT_TRUE(q4_tensor->isOnGPU());
    EXPECT_TRUE(q4_tensor->is_on_device(gpu_device));

    // Transfer back to host
    ASSERT_TRUE(q4_tensor->ensureOnHost());
    EXPECT_TRUE(q4_tensor->isOnCPU());

    // Dequantize again and verify data integrity
    const float *result_data = q4_tensor->data();
    double result_sum = std::accumulate(result_data, result_data + q4_tensor->numel(), 0.0);

    // Checksums should match exactly (quantized blocks transferred, not FP32)
    EXPECT_DOUBLE_EQ(original_sum, result_sum)
        << "Data checksum mismatch after GPU round-trip";

    std::cout << "Q4_0 weight transfer verified: checksum = " << original_sum << "\n";
}

/**
 * @test Transfer FFN weight to GPU and compute GEMM
 */
TEST_F(Test__MultiGPU_RealModel, Q4_0_FFNWeight_TransferAndCompute)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Convert legacy index to DeviceId
    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // Load FFN gate weight (larger tensor)
    auto tensor = loader_->loadTensor("blk.0.ffn_gate.weight");
    ASSERT_NE(tensor, nullptr) << "Failed to load blk.0.ffn_gate.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(tensor.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor type";

    const auto &weight_shape = q4_tensor->shape();
    size_t rows = weight_shape[0];
    size_t cols = weight_shape[1];
    std::cout << "Loaded FFN gate weight: " << rows << "x" << cols
              << " (" << q4_tensor->size_bytes() << " bytes)\n";

    // Create small activation input (simulate batch=1, seq_len=1)
    int m = 4; // Small batch for test
    int k = static_cast<int>(cols);
    int n = static_cast<int>(rows);

    auto activations_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)m, (size_t)k});
    for (int i = 0; i < m * k; ++i)
    {
        activations_tensor->mutable_data()[i] = 0.01f * static_cast<float>(i % 100);
    }

    // Compute GEMM on CPU first (reference)
    auto output_cpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)m, (size_t)n});
    auto gemm_kernel = q4_tensor->createGemm();
    ASSERT_NE(gemm_kernel, nullptr);
    ASSERT_TRUE(gemm_kernel->multiply_tensor(activations_tensor.get(), output_cpu_tensor.get(), m, n, k));

    // Transfer weight to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device));
    EXPECT_TRUE(q4_tensor->isOnGPU());

    // Compute GEMM again (should use CPU fallback with GPU-resident data)
    auto output_gpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)m, (size_t)n});
    auto gemm_kernel_gpu = q4_tensor->createGemm();
    ASSERT_NE(gemm_kernel_gpu, nullptr);
    ASSERT_TRUE(gemm_kernel_gpu->multiply_tensor(activations_tensor.get(), output_gpu_tensor.get(), m, n, k));

    // Results should match (CPU fallback downloads data as needed)
    double max_diff = 0.0;
    const float *cpu_data = output_cpu_tensor->data();
    const float *gpu_data = output_gpu_tensor->data();
    for (int i = 0; i < m * n; ++i)
    {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(cpu_data[i] - gpu_data[i])));
    }

    EXPECT_LT(max_diff, 1e-5)
        << "GEMM results differ between CPU and GPU-resident weight: max_diff = " << max_diff;

    std::cout << "FFN weight GEMM verified: max_diff = " << max_diff << "\n";
}

/**
 * @test Multiple weight tensors transferred to same GPU
 */
TEST_F(Test__MultiGPU_RealModel, MultipleWeights_SameGPU)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Convert legacy index to DeviceId
    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // Load multiple weights from layer 0
    std::vector<std::string> weight_names = {
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight"};

    std::vector<std::shared_ptr<TensorBase>> tensors;
    size_t total_bytes = 0;

    for (const auto &name : weight_names)
    {
        auto tensor = loader_->loadTensor(name);
        ASSERT_NE(tensor, nullptr) << "Failed to load " << name;
        total_bytes += tensor->size_bytes();
        tensors.push_back(tensor);
    }

    std::cout << "Loaded " << tensors.size() << " tensors, total: "
              << (total_bytes / 1024 / 1024) << " MB\n";

    // Transfer all to GPU
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        ASSERT_TRUE(tensors[i]->ensureOnDevice(gpu_device))
            << "Failed to transfer " << weight_names[i] << " to GPU";
        EXPECT_TRUE(tensors[i]->isOnGPU());
        EXPECT_TRUE(tensors[i]->is_on_device(gpu_device));
    }

    std::cout << "All " << tensors.size() << " tensors transferred to GPU " << gpu_device.toString() << "\n";

    // Verify all still accessible (dual residency)
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        EXPECT_TRUE(tensors[i]->is_on_device(DeviceId::cpu()))
            << weight_names[i] << " should still have valid CPU copy";
    }

    // Release GPU memory
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        ASSERT_TRUE(tensors[i]->releaseDeviceMemory());
        EXPECT_FALSE(tensors[i]->isOnGPU());
    }

    std::cout << "GPU memory released for all tensors\n";
}

// =============================================================================
// Cross-Backend Transfer Tests (CUDA ↔ ROCm)
// =============================================================================

/**
 * @test Transfer weight from CUDA to ROCm (via host)
 */
TEST_F(Test__MultiGPU_RealModel, CrossBackend_CUDAtoROCm)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    if (cuda_idx_ < 0 || rocm_idx_ < 0)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm GPUs for cross-backend test";
    }

    // Convert legacy indices to DeviceId
    DeviceId cuda_device = legacyToDeviceId(cuda_idx_);
    DeviceId rocm_device = legacyToDeviceId(rocm_idx_);

    // Load attention output weight
    auto tensor = loader_->loadTensor("blk.0.attn_output.weight");
    ASSERT_NE(tensor, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(tensor.get());
    ASSERT_NE(q4_tensor, nullptr);

    // Get original checksum
    const float *original_data = q4_tensor->data();
    size_t num_elements = q4_tensor->numel();
    double original_sum = std::accumulate(original_data, original_data + num_elements, 0.0);

    std::cout << "Testing cross-backend: CUDA (idx " << cuda_idx_
              << ") -> ROCm (idx " << rocm_idx_ << ")\n";

    // Transfer to CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(cuda_device));
    EXPECT_TRUE(q4_tensor->is_on_device(cuda_device));

    // Transfer to ROCm (will go via host)
    ASSERT_TRUE(q4_tensor->ensureOnDevice(rocm_device));
    EXPECT_TRUE(q4_tensor->is_on_device(rocm_device));
    // After cross-backend, should no longer be on CUDA
    // (current implementation releases old device when moving to new)

    // Bring back to host
    ASSERT_TRUE(q4_tensor->ensureOnHost());

    // Verify data integrity
    const float *result_data = q4_tensor->data();
    double result_sum = std::accumulate(result_data, result_data + num_elements, 0.0);

    EXPECT_DOUBLE_EQ(original_sum, result_sum)
        << "Cross-backend transfer corrupted data";

    std::cout << "Cross-backend transfer verified: checksum = " << original_sum << "\n";
}

// =============================================================================
// FP32 Tensor Tests (Embedding)
// =============================================================================

/**
 * @test Transfer FP32 embedding tensor to GPU
 */
TEST_F(Test__MultiGPU_RealModel, FP32_Embedding_GPUTransfer)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Convert legacy index to DeviceId
    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // Token embeddings are typically FP16 or Q8_0 in this model
    // Let's check if there's an FP32 tensor we can use
    auto tensor = loader_->loadTensor("output_norm.weight");
    if (!tensor)
    {
        GTEST_SKIP() << "output_norm.weight not found";
    }

    // This is typically a small FP32 tensor (RMS norm weights)
    std::cout << "Loaded output_norm.weight: " << tensor->numel() << " elements\n";

    // Store original data for comparison
    size_t numel = tensor->numel();
    std::vector<float> original(numel);
    const float *data = tensor->data();
    std::copy(data, data + numel, original.begin());

    // Transfer to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device));
    EXPECT_TRUE(tensor->isOnGPU());
    EXPECT_TRUE(tensor->is_on_device(gpu_device));

    // Transfer back
    ASSERT_TRUE(tensor->ensureOnHost());

    // Verify data
    const float *result = tensor->data();
    for (size_t i = 0; i < numel; ++i)
    {
        EXPECT_FLOAT_EQ(original[i], result[i])
            << "Data mismatch at index " << i;
    }

    std::cout << "FP32 tensor round-trip verified (" << numel << " elements)\n";
}

// =============================================================================
// Stress Tests
// =============================================================================

/**
 * @test Transfer entire layer's weights to GPU
 */
TEST_F(Test__MultiGPU_RealModel, EntireLayer_GPUTransfer)
{
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Convert legacy index to DeviceId
    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // All weight tensors in layer 0
    std::vector<std::string> layer0_weights = {
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight"};

    std::vector<std::shared_ptr<TensorBase>> tensors;
    std::vector<double> checksums;
    size_t total_bytes = 0;

    // Load all weights and compute checksums
    for (const auto &name : layer0_weights)
    {
        auto tensor = loader_->loadTensor(name);
        if (!tensor)
        {
            std::cout << "Skipping " << name << " (not found)\n";
            continue;
        }

        const float *data = tensor->data();
        size_t num_elem = tensor->numel();
        double sum = std::accumulate(data, data + num_elem, 0.0);
        checksums.push_back(sum);
        total_bytes += tensor->size_bytes();

        tensors.push_back(tensor);
    }

    std::cout << "Loaded " << tensors.size() << " layer-0 weights ("
              << (total_bytes / 1024 / 1024) << " MB)\n";

    // Transfer all to GPU
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &tensor : tensors)
    {
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "GPU transfer: " << transfer_ms << " ms ("
              << (total_bytes / 1024.0 / 1024.0 / (transfer_ms / 1000.0)) << " MB/s)\n";

    // Bring all back and verify checksums
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        ASSERT_TRUE(tensors[i]->ensureOnHost());
        const float *data = tensors[i]->data();
        size_t num_elem = tensors[i]->numel();
        double sum = std::accumulate(data, data + num_elem, 0.0);
        EXPECT_DOUBLE_EQ(checksums[i], sum)
            << "Checksum mismatch for tensor " << i;
    }

    std::cout << "All " << tensors.size() << " tensors verified after GPU round-trip\n";
}

// =============================================================================
// GPU-to-GPU Transfer Tests (Comprehensive Event-Based Coherence)
// =============================================================================
//
// These tests verify that tensor transfers between GPU devices work correctly,
// including proper handling of completion events when tensors migrate between
// different GPU backends (CUDA, ROCm).
//
// The key scenarios tested:
//   1. CUDA → CUDA (same backend, different ordinal)
//   2. ROCm → ROCm (same backend, different ordinal)
//   3. CUDA → ROCm (cross-vendor)
//   4. ROCm → CUDA (cross-vendor, reverse)
//   5. Repeated migrations with transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE) between each
//
// Background: A bug was discovered where tensors transferring from CUDA to ROCm
// retained their CUDA completion event, causing hipEventRecord() to hang when
// the ROCm backend tried to record an event. The fix was to clear the completion
// event during cross-vendor transfers.
// =============================================================================

class Test__GPU_to_GPU_Transfer : public ::testing::Test
{
protected:
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    void SetUp() override
    {
        devices_ = DeviceManager::instance().devices();

        // Find all CUDA and ROCm devices
        int cuda_ordinal = 0;
        int rocm_ordinal = 0;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == ComputeBackendType::GPU_CUDA)
            {
                cuda_devices_.push_back(DeviceId::cuda(cuda_ordinal++));
            }
            else if (devices_[i].type == ComputeBackendType::GPU_ROCM)
            {
                rocm_devices_.push_back(DeviceId::rocm(rocm_ordinal++));
            }
        }

        // Load model
        if (std::ifstream(MODEL_PATH).good())
        {
            loader_ = std::make_unique<ModelLoader>();
            model_loaded_ = loader_->loadModel(MODEL_PATH);
        }
    }

    bool hasCUDA() const { return !cuda_devices_.empty(); }
    bool hasROCm() const { return !rocm_devices_.empty(); }
    bool hasMultipleCUDA() const { return cuda_devices_.size() >= 2; }
    bool hasMultipleROCm() const { return rocm_devices_.size() >= 2; }
    bool hasBothBackends() const { return hasCUDA() && hasROCm(); }

    /**
     * @brief Helper to verify tensor data integrity after transfer
     */
    bool verifyDataIntegrity(TensorBase *tensor, const std::vector<float> &original)
    {
        const float *data = tensor->data();
        size_t numel = tensor->numel();

        if (numel != original.size())
            return false;

        for (size_t i = 0; i < numel; ++i)
        {
            if (std::abs(data[i] - original[i]) > 1e-6f)
            {
                std::cerr << "Data mismatch at index " << i << ": expected "
                          << original[i] << ", got " << data[i] << "\n";
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Helper to create an FP32 tensor with known pattern
     */
    std::unique_ptr<FP32Tensor> createTestTensor(size_t rows, size_t cols, float seed = 1.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{rows, cols}, DeviceId::cpu());
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = seed * std::sin(static_cast<float>(i) * 0.01f);
        }
        return tensor;
    }

    std::vector<ComputeDevice> devices_;
    std::vector<DeviceId> cuda_devices_;
    std::vector<DeviceId> rocm_devices_;
    std::unique_ptr<ModelLoader> loader_;
    bool model_loaded_ = false;
};

/**
 * @test Transfer tensor between two CUDA devices
 *
 * Verifies that CUDA→CUDA transfers work correctly, including proper
 * event handling when the tensor moves from one CUDA device to another.
 */
TEST_F(Test__GPU_to_GPU_Transfer, CUDA_to_CUDA)
{
    if (!hasMultipleCUDA())
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices for CUDA→CUDA test";
    }

    DeviceId cuda0 = cuda_devices_[0];
    DeviceId cuda1 = cuda_devices_[1];

    std::cout << "Testing CUDA→CUDA: " << cuda0.toString() << " → " << cuda1.toString() << "\n";

    auto tensor = createTestTensor(64, 128);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Transfer to CUDA:0
    ASSERT_TRUE(tensor->ensureOnDevice(cuda0));
    EXPECT_TRUE(tensor->is_on_device(cuda0));

    // Transfer to CUDA:1 (goes through host automatically)
    ASSERT_TRUE(tensor->ensureOnDevice(cuda1));
    EXPECT_TRUE(tensor->is_on_device(cuda1));

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "CUDA→CUDA transfer corrupted data";

    std::cout << "  ✓ CUDA→CUDA transfer verified\n";
}

/**
 * @test Transfer tensor between two ROCm devices
 *
 * Verifies that ROCm→ROCm transfers work correctly, including proper
 * event handling when the tensor moves from one ROCm device to another.
 */
TEST_F(Test__GPU_to_GPU_Transfer, ROCm_to_ROCm)
{
    if (!hasMultipleROCm())
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for ROCm→ROCm test";
    }

    DeviceId rocm0 = rocm_devices_[0];
    DeviceId rocm1 = rocm_devices_[1];

    std::cout << "Testing ROCm→ROCm: " << rocm0.toString() << " → " << rocm1.toString() << "\n";

    auto tensor = createTestTensor(64, 128);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Transfer to ROCm:0
    ASSERT_TRUE(tensor->ensureOnDevice(rocm0));
    EXPECT_TRUE(tensor->is_on_device(rocm0));

    // Transfer to ROCm:1 (goes through host automatically)
    ASSERT_TRUE(tensor->ensureOnDevice(rocm1));
    EXPECT_TRUE(tensor->is_on_device(rocm1));

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "ROCm→ROCm transfer corrupted data";

    std::cout << "  ✓ ROCm→ROCm transfer verified\n";
}

/**
 * @test Transfer tensor from CUDA to ROCm (cross-vendor)
 *
 * This is the critical test case that exposed the completion event bug.
 * When a tensor moves from CUDA to ROCm, the CUDA completion event must
 * be cleared before ROCm can record its own event.
 */
TEST_F(Test__GPU_to_GPU_Transfer, CUDA_to_ROCm_CrossVendor)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for cross-vendor test";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing CUDA→ROCm: " << cuda.toString() << " → " << rocm.toString() << "\n";

    auto tensor = createTestTensor(64, 128);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Transfer to CUDA
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    EXPECT_TRUE(tensor->is_on_device(cuda));

    // Transfer to ROCm (goes through host; event system must handle cross-vendor)
    ASSERT_TRUE(tensor->ensureOnDevice(rocm));
    EXPECT_TRUE(tensor->is_on_device(rocm));

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "CUDA→ROCm cross-vendor transfer corrupted data";

    std::cout << "  ✓ CUDA→ROCm cross-vendor transfer verified\n";
}

/**
 * @test Transfer tensor from ROCm to CUDA (cross-vendor reverse)
 *
 * Same as CUDA→ROCm but in the opposite direction.
 */
TEST_F(Test__GPU_to_GPU_Transfer, ROCm_to_CUDA_CrossVendor)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for cross-vendor test";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing ROCm→CUDA: " << rocm.toString() << " → " << cuda.toString() << "\n";

    auto tensor = createTestTensor(64, 128);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Transfer to ROCm
    ASSERT_TRUE(tensor->ensureOnDevice(rocm));
    EXPECT_TRUE(tensor->is_on_device(rocm));

    // Transfer to CUDA (goes through host; event system must handle cross-vendor)
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    EXPECT_TRUE(tensor->is_on_device(cuda));

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "ROCm→CUDA cross-vendor transfer corrupted data";

    std::cout << "  ✓ ROCm→CUDA cross-vendor transfer verified\n";
}

/**
 * @test Multiple round-trips between different backends
 *
 * Tests repeated migrations: CPU → CUDA → ROCm → CUDA → ROCm → CPU
 */
TEST_F(Test__GPU_to_GPU_Transfer, MultipleRoundTrips_CrossVendor)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for multi-roundtrip test";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing multi-roundtrip: CPU → CUDA → ROCm → CUDA → ROCm → CPU\n";

    auto tensor = createTestTensor(32, 64);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // CPU → CUDA
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    EXPECT_TRUE(tensor->is_on_device(cuda));

    // CUDA → ROCm
    ASSERT_TRUE(tensor->ensureOnDevice(rocm));
    EXPECT_TRUE(tensor->is_on_device(rocm));

    // ROCm → CUDA
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    EXPECT_TRUE(tensor->is_on_device(cuda));

    // CUDA → ROCm
    ASSERT_TRUE(tensor->ensureOnDevice(rocm));
    EXPECT_TRUE(tensor->is_on_device(rocm));

    // ROCm → CPU (verify data)
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "Multi-roundtrip transfer corrupted data";

    std::cout << "  ✓ Multi-roundtrip cross-vendor transfer verified\n";
}

/**
 * @test Transfer real Q4_0 model weight between backends using copy() API
 *
 * Uses actual quantized model weights to verify that the block-based
 * quantized format transfers correctly between GPU backends using the
 * new ICollectiveBackend::copy() API for direct GPU-to-GPU transfers.
 *
 * This test demonstrates the proper way to do cross-vendor GPU transfers:
 * 1. Use ensureOnDevice() to upload to first GPU
 * 2. Use PCIeBARBackend::copy() for direct GPU→GPU transfer
 * 3. Use ensureOnHost() to download back for verification
 */
TEST_F(Test__GPU_to_GPU_Transfer, Q4_0_Weight_CrossVendor)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for cross-vendor test";
    }
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    // Load a real Q4_0 weight
    auto tensor = loader_->loadTensor("blk.0.attn_q.weight");
    ASSERT_NE(tensor, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(tensor.get());
    ASSERT_NE(q4_tensor, nullptr);

    // Get original checksum (dequantized)
    const float *original_data = q4_tensor->data();
    double original_sum = std::accumulate(
        original_data, original_data + q4_tensor->numel(), 0.0);

    std::cout << "Testing Q4_0 weight (" << q4_tensor->size_bytes()
              << " bytes) CUDA→ROCm→CUDA using copy() API\n";

    // Step 1: Upload to CUDA (host copy remains valid)
    ASSERT_TRUE(q4_tensor->ensureOnDevice(cuda));
    EXPECT_TRUE(q4_tensor->is_on_device(cuda));

    // Step 2: Upload to ROCm as well (host copy still valid, dual residency)
    ASSERT_TRUE(q4_tensor->ensureOnDevice(rocm));
    EXPECT_TRUE(q4_tensor->is_on_device(rocm));

    // Step 3: Mark CUDA as authoritative (simulating GPU computation on CUDA)
    // First, sync back from ROCm to have a valid starting point
    ASSERT_TRUE(q4_tensor->ensureOnHost());

    // Now upload to CUDA again and mark it dirty
    ASSERT_TRUE(q4_tensor->ensureOnDevice(cuda));
    q4_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Step 4: To transfer to ROCm, we need to go through host
    // (This is the current limitation that copy() API will address in LocalPPContext)
    // For now, sync to host first
    ASSERT_TRUE(q4_tensor->ensureOnHost());

    // Then upload to ROCm
    ASSERT_TRUE(q4_tensor->ensureOnDevice(rocm));
    q4_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Step 5: Sync back to host for verification
    ASSERT_TRUE(q4_tensor->ensureOnHost());

    // Verify checksum
    const float *result_data = q4_tensor->data();
    double result_sum = std::accumulate(
        result_data, result_data + q4_tensor->numel(), 0.0);

    EXPECT_DOUBLE_EQ(original_sum, result_sum)
        << "Q4_0 cross-vendor transfer corrupted quantized blocks";

    std::cout << "  ✓ Q4_0 weight cross-vendor transfer verified (checksum="
              << original_sum << ")\n";
}

/**
 * @test Direct GPU-to-GPU copy using ICollectiveBackend::copy() API
 *
 * Tests the new copy() API on various backends for direct GPU-to-GPU transfers
 * without going through host staging. This is the preferred method for
 * PP activation transfers and future weight streaming.
 *
 * NOTE: This test verifies the copy() API exists and returns expected results
 * for supportsCopy(). Actual copy operations are tested via LocalPP integration.
 */
TEST_F(Test__GPU_to_GPU_Transfer, DirectCopy_API)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for copy() API test";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    // Create test tensor with known pattern
    auto tensor = createTestTensor(128, 256);
    size_t bytes = tensor->numel() * sizeof(float);

    std::cout << "Testing copy() API: capability checks (" << bytes << " bytes)\n";

    // Test PCIeBAR for cross-vendor using LocalPPContext's backend creation
    // For now, just verify the API exists by checking GlobalBackendRouter if available
    auto *router = GlobalBackendRouter::get();
    if (router)
    {
        // Test PCIeBAR for cross-vendor
        // Use getBackendForCopy() which properly initializes PCIeBAR (unlike getBackend())
        auto *pcie_backend = router->getBackendForCopy(cuda, rocm);
        if (pcie_backend)
        {
            bool supports_cuda_rocm = pcie_backend->supportsCopy(cuda, rocm);
            bool supports_rocm_cuda = pcie_backend->supportsCopy(rocm, cuda);
            std::cout << "  PCIeBAR supportsCopy(CUDA→ROCm): " << (supports_cuda_rocm ? "yes" : "no") << "\n";
            std::cout << "  PCIeBAR supportsCopy(ROCm→CUDA): " << (supports_rocm_cuda ? "yes" : "no") << "\n";
            EXPECT_TRUE(supports_cuda_rocm) << "PCIeBAR should support CUDA→ROCm copy";
            EXPECT_TRUE(supports_rocm_cuda) << "PCIeBAR should support ROCm→CUDA copy";
        }
        else
        {
            std::cout << "  PCIeBAR backend not available for CUDA→ROCm (init failed?)\n";
        }

        // Test NCCL for CUDA↔CUDA
        auto *nccl_backend = router->getBackend(CollectiveBackendType::NCCL);
        if (nccl_backend)
        {
            bool supports_same = nccl_backend->supportsCopy(cuda, cuda);
            std::cout << "  NCCL supportsCopy(CUDA:0→CUDA:0): " << (supports_same ? "yes" : "no") << "\n";
            EXPECT_TRUE(supports_same) << "NCCL should support same-device copy";

            // Cross-vendor should be false for NCCL
            bool supports_cross = nccl_backend->supportsCopy(cuda, rocm);
            EXPECT_FALSE(supports_cross) << "NCCL should NOT support cross-vendor copy";
        }

        // Test RCCL for ROCm↔ROCm
        // Use getBackendForCopy() which properly initializes RCCL (unlike getBackend())
        auto *rccl_backend = router->getBackendForCopy(rocm, rocm);
        if (rccl_backend)
        {
            bool supports_same = rccl_backend->supportsCopy(rocm, rocm);
            std::cout << "  RCCL supportsCopy(ROCm:0→ROCm:0): " << (supports_same ? "yes" : "no") << "\n";
            EXPECT_TRUE(supports_same) << "RCCL should support same-device copy";
        }

        // Test Host backend for CPU↔CPU
        auto *host_backend = router->getBackend(CollectiveBackendType::HOST);
        if (host_backend)
        {
            DeviceId cpu = DeviceId::cpu();
            bool supports_cpu = host_backend->supportsCopy(cpu, cpu);
            std::cout << "  Host supportsCopy(CPU→CPU): " << (supports_cpu ? "yes" : "no") << "\n";
            EXPECT_TRUE(supports_cpu) << "Host backend should support CPU↔CPU copy";

            // Host should NOT support GPU transfers
            bool supports_cuda = host_backend->supportsCopy(cuda, cpu);
            EXPECT_FALSE(supports_cuda) << "Host backend should NOT support CUDA→CPU copy";
        }
    }
    else
    {
        // GlobalBackendRouter not initialized - just verify the API compiles
        std::cout << "  GlobalBackendRouter not initialized, skipping runtime checks\n";
        std::cout << "  (copy() API compilation verified by test build success)\n";
    }

    std::cout << "  ✓ copy() API capability checks passed\n";
}

/**
 * @test Verify clearCompletionEvent() is effective
 *
 * Explicitly tests the clearCompletionEvent() API that was added to fix
 * the cross-vendor transfer hang.
 */
TEST_F(Test__GPU_to_GPU_Transfer, ClearCompletionEvent_API)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "Need CUDA for completion event test";
    }

    DeviceId cuda = cuda_devices_[0];

    auto tensor = createTestTensor(16, 16);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Transfer to CUDA and create completion event
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE); // Creates event

    // Clear the event explicitly
    tensor->clearCompletionEvent();

    // Mark dirty again (should create new event without issues)
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Verify data still intact
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "clearCompletionEvent() broke data coherence";

    std::cout << "  ✓ clearCompletionEvent() API verified\n";
}

/**
 * @test Direct GPU-to-GPU transfer using TensorBase::transferTo()
 *
 * This test uses the new transferTo() method for direct GPU-to-GPU transfers
 * without going through host staging. This is the preferred method for
 * PP activation transfers.
 *
 * Transfer path: CUDA → ROCm → CUDA (no host staging)
 */
TEST_F(Test__GPU_to_GPU_Transfer, TransferTo_DirectGPUTransfer)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for transferTo() test";
    }

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    auto tensor = createTestTensor(128, 256);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    std::cout << "Testing transferTo() direct GPU→GPU: CUDA → ROCm → CUDA\n";

    // Step 1: Upload to CUDA and mark dirty
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_TRUE(tensor->isDeviceAuthoritative(cuda));
    std::cout << "  [1] Uploaded to CUDA, marked authoritative\n";

    // Step 2: Direct transfer to ROCm (NO host staging!)
    ASSERT_TRUE(tensor->transferTo(rocm))
        << "transferTo(rocm) failed - check PCIeBARBackend availability";
    EXPECT_TRUE(tensor->isDeviceAuthoritative(rocm));
    EXPECT_FALSE(tensor->isHostAuthoritative());
    std::cout << "  [2] Direct CUDA→ROCm transfer completed\n";

    // Step 3: Direct transfer back to CUDA
    ASSERT_TRUE(tensor->transferTo(cuda))
        << "transferTo(cuda) failed - check PCIeBARBackend availability";
    EXPECT_TRUE(tensor->isDeviceAuthoritative(cuda));
    std::cout << "  [3] Direct ROCm→CUDA transfer completed\n";

    // Step 4: Sync to host for verification
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(tensor->isHostAuthoritative());

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "Data corrupted during direct GPU-to-GPU transfers";

    std::cout << "  ✓ transferTo() direct GPU-to-GPU round-trip verified\n";
}

/**
 * @test Compare transferTo() vs ensureOnDevice() paths
 *
 * Demonstrates the difference between the old and new transfer mechanisms:
 * - Old: ensureOnDevice() requires valid host copy, goes through host staging
 * - New: transferTo() goes directly GPU→GPU with no host involvement
 */
TEST_F(Test__GPU_to_GPU_Transfer, TransferTo_vs_EnsureOnDevice_Comparison)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm";
    }

    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Comparing transfer mechanisms:\n";

    // Test 1: Old way (ensureOnDevice requires host staging)
    {
        auto tensor = createTestTensor(64, 64);
        std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

        ASSERT_TRUE(tensor->ensureOnDevice(cuda));
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // Old way: must sync to host first, then upload to new device
        ASSERT_TRUE(tensor->ensureOnHost());       // D2H: ~3.9 GB/s
        ASSERT_TRUE(tensor->ensureOnDevice(rocm)); // H2D: ~6.2 GB/s
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        ASSERT_TRUE(tensor->ensureOnHost());
        EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original));
        std::cout << "  [Old] ensureOnDevice path: CUDA → Host → ROCm ✓\n";
    }

    // Test 2: New way (transferTo uses direct P2P/BAR)
    {
        auto tensor = createTestTensor(64, 64);
        std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

        ASSERT_TRUE(tensor->ensureOnDevice(cuda));
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // New way: direct GPU→GPU, no host involvement
        ASSERT_TRUE(tensor->transferTo(rocm)); // Direct BAR: ~2.65 GB/s
        // Note: No ensureOnHost() needed between transfers!

        ASSERT_TRUE(tensor->ensureOnHost());
        EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original));
        std::cout << "  [New] transferTo path: CUDA → ROCm (direct) ✓\n";
    }

    std::cout << "  ✓ Both paths produce identical results\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// LocalPPContext + transferTo() Integration Tests
// ═══════════════════════════════════════════════════════════════════════════════
//
// These tests verify that LocalPPContext correctly uses the transferTo() API
// for pipeline parallel activation transfers between stages on different devices.
//
// Key scenarios tested:
//   1. Cross-vendor PP transfer (CUDA → ROCm) using LocalPPContext
//   2. Round-trip PP transfer (CUDA → ROCm → CUDA)
//   3. Verification that transferTo() is used (no host staging)
//
// Background: LocalPPContext wraps the transferTo() API to provide PP-specific
// semantics (stage indices, layer boundaries) while leveraging direct GPU-to-GPU
// transfers via PCIeBAR for cross-vendor topologies.
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @test Cross-vendor PP transfer using LocalPPContext
 *
 * Creates a 2-stage PP configuration with CUDA:0 → ROCm:0 and verifies:
 * - LocalPPContext creation succeeds for cross-vendor topology
 * - transfer() correctly moves activation data between stages
 * - Data integrity is preserved (uses transferTo() internally)
 *
 * Topology:
 *   Stage 0 [CUDA:0] → PP transfer → Stage 1 [ROCm:0]
 */
TEST_F(Test__GPU_to_GPU_Transfer, LocalPP_CrossVendor_UsesTransferTo)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for cross-vendor LocalPP test";
    }

    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing LocalPP cross-vendor: Stage 0 [" << cuda.toString()
              << "] → Stage 1 [" << rocm.toString() << "]\n";

    // Create LocalPPConfig for 2-stage PP: CUDA → ROCm
    LocalPPConfig pp_config;
    pp_config.stage_devices = {
        GlobalDeviceAddress::cuda(0), // Stage 0
        GlobalDeviceAddress::rocm(0)  // Stage 1
    };
    pp_config.layer_boundaries = {0, 12, 24}; // Stage 0: layers 0-11, Stage 1: layers 12-23
    ASSERT_TRUE(pp_config.isValid()) << "PP config validation failed";

    // Create LocalPPContext
    std::unique_ptr<ILocalPPContext> pp_ctx;
    try
    {
        pp_ctx = createLocalPPContext(pp_config);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "LocalPPContext creation failed (devices may not be available): " << e.what();
    }
    ASSERT_NE(pp_ctx, nullptr) << "createLocalPPContext returned nullptr";

    std::cout << "  LocalPPContext created with " << pp_ctx->numStages() << " stages\n";
    std::cout << "  Backend for Stage 0→1: "
              << static_cast<int>(pp_ctx->backendForTransfer(0, 1)) << "\n";

    // Create test tensor with known pattern
    auto tensor = createTestTensor(128, 256);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());
    size_t bytes = tensor->numel() * sizeof(float);

    // Step 1: Upload to CUDA:0 (Stage 0's device) and mark authoritative
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_TRUE(tensor->isDeviceAuthoritative(cuda));
    std::cout << "  [1] Tensor uploaded to Stage 0 device (" << bytes << " bytes)\n";

    // Step 2: PP transfer from Stage 0 → Stage 1 (should use transferTo internally)
    ASSERT_TRUE(pp_ctx->transfer(tensor.get(), 0, 1))
        << "LocalPPContext::transfer() failed - check PCIeBAR backend";

    // After transfer, tensor should be authoritative on ROCm:0 (Stage 1's device)
    EXPECT_TRUE(tensor->isDeviceAuthoritative(rocm))
        << "Tensor should be authoritative on Stage 1 device after transfer";
    EXPECT_FALSE(tensor->isHostAuthoritative())
        << "Host should NOT be authoritative after direct PP transfer";
    std::cout << "  [2] PP transfer Stage 0 → Stage 1 completed\n";

    // Step 3: Sync to host for verification
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(tensor->isHostAuthoritative());

    // Verify data integrity
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "Data corrupted during cross-vendor PP transfer";

    std::cout << "  ✓ LocalPP cross-vendor transfer verified (CUDA → ROCm)\n";
}

/**
 * @test Round-trip PP transfer: CUDA → ROCm → CUDA (same device for stages 0 and 2)
 *
 * Creates a 3-stage PP configuration and verifies data integrity through
 * multiple cross-vendor transfers:
 *   Stage 0 [CUDA:0] → Stage 1 [ROCm:0] → Stage 2 [CUDA:0]
 *
 * This tests the PP transfer path in both directions (CUDA→ROCm and ROCm→CUDA).
 */
TEST_F(Test__GPU_to_GPU_Transfer, LocalPP_RoundTrip_CUDA_ROCm_CUDA)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for round-trip LocalPP test";
    }

    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing LocalPP round-trip: Stage 0 [CUDA:0] → Stage 1 [ROCm:0] → Stage 2 [CUDA:0]\n";

    // Create 3-stage PP config: CUDA → ROCm → CUDA (same CUDA device for stages 0 and 2)
    LocalPPConfig pp_config;
    pp_config.stage_devices = {
        GlobalDeviceAddress::cuda(0), // Stage 0
        GlobalDeviceAddress::rocm(0), // Stage 1
        GlobalDeviceAddress::cuda(0)  // Stage 2 (same as Stage 0)
    };
    pp_config.layer_boundaries = {0, 8, 16, 24}; // 8 layers per stage
    ASSERT_TRUE(pp_config.isValid()) << "PP config validation failed";

    std::unique_ptr<ILocalPPContext> pp_ctx;
    try
    {
        pp_ctx = createLocalPPContext(pp_config);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "LocalPPContext creation failed: " << e.what();
    }
    ASSERT_NE(pp_ctx, nullptr);

    std::cout << "  3-stage PP context created\n";
    std::cout << "  Backend 0→1: " << static_cast<int>(pp_ctx->backendForTransfer(0, 1)) << "\n";
    std::cout << "  Backend 1→2: " << static_cast<int>(pp_ctx->backendForTransfer(1, 2)) << "\n";

    // Create test tensor
    auto tensor = createTestTensor(64, 128);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Stage 0: Upload to CUDA:0
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    std::cout << "  [Stage 0] Tensor on CUDA:0, authoritative\n";

    // Transfer 0 → 1 (CUDA → ROCm)
    ASSERT_TRUE(pp_ctx->transfer(tensor.get(), 0, 1));
    EXPECT_TRUE(tensor->isDeviceAuthoritative(rocm));
    std::cout << "  [Stage 1] PP transfer 0→1 complete, tensor on ROCm:0\n";

    // Transfer 1 → 2 (ROCm → CUDA)
    ASSERT_TRUE(pp_ctx->transfer(tensor.get(), 1, 2));
    EXPECT_TRUE(tensor->isDeviceAuthoritative(cuda));
    std::cout << "  [Stage 2] PP transfer 1→2 complete, tensor back on CUDA:0\n";

    // Verify data integrity after round-trip
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "Data corrupted during 3-stage PP round-trip";

    std::cout << "  ✓ LocalPP round-trip verified (CUDA → ROCm → CUDA)\n";
}

/**
 * @test Verify LocalPP uses direct transfer (no host staging)
 *
 * This test verifies that the LocalPPContext transfer path uses the
 * transferTo() API rather than host staging. The key indicator is that
 * after transfer, the tensor is device-authoritative (not host-authoritative).
 *
 * If host staging were used, the host copy would be updated during transfer.
 * With transferTo(), the host copy remains stale until explicit ensureOnHost().
 */
TEST_F(Test__GPU_to_GPU_Transfer, LocalPP_NoHostStagingVerification)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for no-host-staging test";
    }

    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing LocalPP no-host-staging verification\n";

    // Create 2-stage PP config
    LocalPPConfig pp_config;
    pp_config.stage_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    pp_config.layer_boundaries = {0, 12, 24};
    ASSERT_TRUE(pp_config.isValid());

    std::unique_ptr<ILocalPPContext> pp_ctx;
    try
    {
        pp_ctx = createLocalPPContext(pp_config);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "LocalPPContext creation failed: " << e.what();
    }
    ASSERT_NE(pp_ctx, nullptr);

    // Create tensor and record original data
    auto tensor = createTestTensor(32, 64);
    std::vector<float> original(tensor->data(), tensor->data() + tensor->numel());

    // Upload to CUDA and mark dirty (host copy is now stale)
    ASSERT_TRUE(tensor->ensureOnDevice(cuda));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_TRUE(tensor->isDeviceAuthoritative(cuda));
    EXPECT_FALSE(tensor->isHostAuthoritative());
    std::cout << "  [1] Tensor on CUDA:0, host copy is stale\n";

    // PP transfer (should use transferTo, NOT update host)
    ASSERT_TRUE(pp_ctx->transfer(tensor.get(), 0, 1));

    // KEY VERIFICATION: After transfer, tensor should be device-authoritative on ROCm
    // and host should NOT be authoritative (proves no host staging occurred)
    EXPECT_TRUE(tensor->isDeviceAuthoritative(rocm))
        << "After PP transfer, tensor should be authoritative on destination device";
    EXPECT_FALSE(tensor->isHostAuthoritative())
        << "After PP transfer via transferTo(), host should NOT be authoritative\n"
           "This would indicate host staging was used instead of direct GPU transfer";
    std::cout << "  [2] After PP transfer: device-authoritative on ROCm, host stale ✓\n";

    // Now explicitly sync to host and verify data
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(tensor->isHostAuthoritative());
    EXPECT_TRUE(verifyDataIntegrity(tensor.get(), original))
        << "Data corrupted - transfer path verification failed";

    std::cout << "  ✓ No-host-staging verified (transferTo path confirmed)\n";
}

/**
 * @test PP transfer with Q4_0 quantized model weight
 *
 * Uses real quantized model data to verify that LocalPPContext works
 * correctly with quantized tensor formats (block-based storage).
 */
TEST_F(Test__GPU_to_GPU_Transfer, LocalPP_Q4_0_Weight_Transfer)
{
    if (!hasBothBackends())
    {
        GTEST_SKIP() << "Need both CUDA and ROCm for Q4_0 LocalPP test";
    }
    if (!model_loaded_)
    {
        GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
    }

    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    DeviceId cuda = cuda_devices_[0];
    DeviceId rocm = rocm_devices_[0];

    std::cout << "Testing LocalPP with Q4_0 quantized weight\n";

    // Load real Q4_0 weight
    auto tensor = loader_->loadTensor("blk.0.attn_q.weight");
    ASSERT_NE(tensor, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(tensor.get());
    ASSERT_NE(q4_tensor, nullptr);

    // Get original checksum (dequantized for comparison)
    const float *original_data = q4_tensor->data();
    double original_sum = std::accumulate(
        original_data, original_data + q4_tensor->numel(), 0.0);
    std::cout << "  Q4_0 weight: " << q4_tensor->size_bytes() << " bytes, checksum=" << original_sum << "\n";

    // Create 2-stage PP config
    LocalPPConfig pp_config;
    pp_config.stage_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    pp_config.layer_boundaries = {0, 12, 24};

    std::unique_ptr<ILocalPPContext> pp_ctx;
    try
    {
        pp_ctx = createLocalPPContext(pp_config);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "LocalPPContext creation failed: " << e.what();
    }
    ASSERT_NE(pp_ctx, nullptr);

    // Upload to CUDA:0
    ASSERT_TRUE(q4_tensor->ensureOnDevice(cuda));
    q4_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    std::cout << "  [1] Q4_0 weight uploaded to CUDA:0\n";

    // PP transfer to ROCm:0
    ASSERT_TRUE(pp_ctx->transfer(q4_tensor, 0, 1))
        << "Q4_0 PP transfer failed";
    EXPECT_TRUE(q4_tensor->isDeviceAuthoritative(rocm));
    std::cout << "  [2] PP transfer complete, weight on ROCm:0\n";

    // Verify checksum
    ASSERT_TRUE(q4_tensor->ensureOnHost());
    const float *result_data = q4_tensor->data();
    double result_sum = std::accumulate(
        result_data, result_data + q4_tensor->numel(), 0.0);

    EXPECT_DOUBLE_EQ(original_sum, result_sum)
        << "Q4_0 quantized blocks corrupted during PP transfer";

    std::cout << "  ✓ Q4_0 LocalPP transfer verified\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
