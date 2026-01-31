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

    std::vector<float> activations(m * k);
    for (int i = 0; i < m * k; ++i)
    {
        activations[i] = 0.01f * static_cast<float>(i % 100);
    }

    // Compute GEMM on CPU first (reference)
    std::vector<float> output_cpu(m * n, 0.0f);
    auto gemm_kernel = q4_tensor->createGemm();
    ASSERT_NE(gemm_kernel, nullptr);
    ASSERT_TRUE(gemm_kernel->multiply(activations.data(), output_cpu.data(), m, n, k));

    // Transfer weight to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device));
    EXPECT_TRUE(q4_tensor->isOnGPU());

    // Compute GEMM again (should use CPU fallback with GPU-resident data)
    std::vector<float> output_gpu(m * n, 0.0f);
    auto gemm_kernel_gpu = q4_tensor->createGemm();
    ASSERT_NE(gemm_kernel_gpu, nullptr);
    ASSERT_TRUE(gemm_kernel_gpu->multiply(activations.data(), output_gpu.data(), m, n, k));

    // Results should match (CPU fallback downloads data as needed)
    double max_diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(output_cpu[i] - output_gpu[i])));
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
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
